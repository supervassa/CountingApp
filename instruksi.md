# MASTER INSTRUCTION

## AI People Flow + Face Recognition System (v2)

Anda bertindak sebagai **Senior AI/Computer Vision Engineer + Edge AI Engineer**. Bangun sistem penghitung orang masuk/keluar gedung berbasis **YOLO + ByteTrack + Face Recognition** yang berjalan pada **NVIDIA Jetson Nano 4GB (B01)** dengan **2× kamera CSI IMX219-120**.

Dokumen ini adalah instruksi kerja. Rujuk [PRD.md](PRD.md) untuk detail requirement.

---

## 1. Tujuan Sistem

**Fitur utama: menghitung orang `IN`/`OUT` melalui satu pintu.**
**Fitur tambahan (diminta pimpinan): pengenalan wajah** untuk menautkan identitas (`idpersonal`) ke setiap event — untuk mengurangi risiko kriminal dan (Fase 2) presensi kedinasan.

Sistem harus:

- Mendeteksi orang/kepala melalui 2 kamera.
- Melacak orang antar-frame (per kamera).
- Mengenali identitas orang terdaftar (identifikasi 1:N open-set) atau menandai `UNKNOWN`.
- Menentukan arah pergerakan melalui pita dua garis virtual.
- Mencatat event `IN` dan `OUT` dengan `idpersonal`/`UNKNOWN`, `camera_id`, timestamp sistem.
- Mencegah penghitungan ganda (per track, per kamera, antar kamera).
- Menyimpan histori ke SQLite lokal.
- Mengambil nama/info orang dari PostgreSQL kepegawaian secara **asinkron** (di luar loop real-time).
- Menampilkan rekap harian + occupancy.

Contoh output:

```
[07:31:20] cam_out  a4f9… (Juris)  → IN
[08:20:11] cam_in   a4f9… (Juris)  → OUT
[08:41:05] cam_out  UNKNOWN        → IN
```

---

## 2. Arsitektur Utama

```
2× CSI Camera (IMX219-120)
   ↓  (per kamera)
Frame Capture  → (opsional) undistort lensa 120°
   ↓
YOLO Person/Head Detection
   ↓
ByteTrack
   ↓
Face Detection / Alignment  (setiap N frame, dalam ROI)
   ↓
Face Embedding (ArcFace)
   ↓
Gallery Matching (cosine, threshold + margin)
   ↓
Identity: idpersonal / UNKNOWN  → di-vote → dikunci ke track_id
   ↓
Virtual Line Crossing (pita 2 garis) + State Machine + Direction Filter
   ↓
IN / OUT Event
   ↓
SQLite lokal
   ↓
├─ Rekap harian / Occupancy / Cross-check antar kamera
└─ Enrichment worker (asinkron) → PostgreSQL person.get_info_person($1) → persons_cache
```

Prinsip wajib:

```
Detection ≠ Tracking ≠ Recognition ≠ Counting
Counting INDEPENDEN dari Recognition
```

Setiap komponen = module/class terpisah. Recognition boleh gagal (`UNKNOWN`) tanpa merusak counting.

### Tata letak dua kamera

| Kamera | Pemasangan | Tanggung jawab |
|---|---|---|
| `cam_out` | luar pintu, menghadap approach luar | event `IN` + identitas orang masuk |
| `cam_in` | dalam pintu, menghadap approach dalam | event `OUT` + identitas orang keluar |

Satu kamera hanya melihat wajah satu arah — itu sebabnya perlu dua. Occupancy tunggal disuplai kedua kamera dengan filter arah + dedup `(waktu, arah)`.

---

## 3. Target Hardware

```
Device       : NVIDIA Jetson Nano 4GB (B01, 2 port CSI)
Architecture : ARM64 / aarch64
JetPack      : R32.7.1
CUDA         : 10.2
Python       : 3.6.9 (Jetson) / 3.9+ (laptop dev)
Kamera       : 2× IMX219-120 CSI, 1280×720 atau 1920×1080
Power        : barrel jack 5V/4A, mode 10W/MAXN
Cooling      : kipas aktif wajib
```

Jetson = **inference device**, bukan mesin training. Optimalkan RAM, VRAM, CPU, GPU, latency. Kendala: FFC IMX219-120 ±15 cm → Jetson dalam ~1 m dari kamera.

**Pengembangan dilakukan di laptop dulu** (pipeline benar dulu), baru port ke Jetson + TensorRT. Modul `app/camera/` memakai `CameraSource` pluggable: `webcam` | `file` | `csi`.

---

## 4. Model AI

### Person / Head Detection

- YOLO untuk kelas `person`. Untuk kondisi ramai, sediakan opsi model deteksi **kepala** (CrowdHuman) — kepala lebih terpisah dari atas.
- Jangan face recognition ke seluruh frame — hanya ROI relevan.

### Tracking

- **ByteTrack**. Output: `track_id`, `bbox`, `confidence`, `trajectory`.
- `track_buffer` besar (45–60 frame) untuk menjembatani occlusion singkat (tailgating).
- `track_id` ≠ `idpersonal`. Dua informasi berbeda.

### Face Recognition — TANPA TRAINING

- Pendekatan **face embedding** (bukan classifier per orang).
- **Jangan latih / fine-tune model.** Model ArcFace pretrained dipakai apa adanya. Alasan: pretrained sudah general dari jutaan wajah; jumlah orang terdaftar tak cukup untuk training; enrollment ≠ training (enrollment hanya menghasilkan embedding untuk gallery).
- Model kandidat: ArcFace / InsightFace (R50 / MobileFaceNet) format ONNX.
- **Verifikasi kompatibilitas** dengan JetPack R32.7.1, CUDA 10.2, Python 3.6, ARM64 sebelum memilih implementasi final. Jangan asumsikan library modern kompatibel dengan Jetson Nano. `insightface` pip kemungkinan gagal di Python 3.6 — rencana cadangan: ONNX mentah + preprocessing manual.

### Identity Matching (1:N open-set)

```
query_embedding → cosine similarity ke seluruh gallery → top-k
   keputusan: (top1 ≥ threshold) AND (top1 − top2 ≥ margin) → idpersonal
              selain itu                                    → UNKNOWN
   hasil di-vote (vote_window) sebelum dikunci ke track_id
```

- Mayoritas orang lewat TIDAK terdaftar → `UNKNOWN` harus sering & benar.
- Jangan pilih identitas similarity tertinggi bila di bawah threshold.
- Threshold + margin dari config, dikalibrasi (FAR/FRR/EER). Jangan hard-code.

---

## 5. Enrollment

- **Rekam ulang dengan IMX219** pada posisi pintu terpasang (domain cocok). Dataset lama `raw/` **di-shelve** (mayoritas 1 foto/orang, domain selfie) — hanya dipakai sebagai pool impostor untuk uji FAR. Audit: `scripts/ingest_raw.py`.
- Target: **15–20 sampel/orang**, ditautkan ke `idpersonal` asli (dipilih dari daftar PostgreSQL).
- Variasi: pose (frontal, yaw ±30°, pitch), ekspresi, kacamata, pencahayaan (normal/terang/redup/backlight), jarak (~1/2/3 m). Jangan 20 foto identik.
- Quality check otomatis: confidence detector, ketajaman (var-of-Laplacian), ukuran wajah ≥ 112 px, yaw/pitch < 30°, exposure wajar, dedup (cosine > 0,97 ditolak).
- Simpan **semua** embedding per-sample + mean embedding per orang di `data/gallery/embeddings.npz`.
- Enrollment di-gate ke Capaian 1c (kamera harus terpasang). Sebelum itu, dev pakai webcam.

### Gallery vs Probe

```
dataset/gallery/            enrollment 15–20/orang
dataset/probe/calib/        kalibrasi threshold
dataset/probe/test/         metrik final — JANGAN disentuh saat kalibrasi
        {clean,pose,illumination,occlusion,combined}/
```

---

## 6. Line Crossing — Pita Dua Garis

```
─────────── garis LUAR    urutan LUAR→DALAM = IN
   (pita)
─────────── garis DALAM   urutan DALAM→LUAR = OUT
```

- Arah dari **urutan crossing**, bukan posisi satu frame. Pakai trajectory/centroid.
- Anchor = **kepala** (`counting.crossing_anchor`), bukan pusat bbox badan.
- Koordinat garis = fraksi frame, di-set per kamera pada Capaian 1c. Jangan hard-code.
- Poligon zona (inside/outside, walk-in/walk-out/pass-by) = alternatif setara, murni geometri gambar.

### State machine (per track)

```
UNKNOWN → OUTSIDE → (crossing pita) → INSIDE → (crossing pita) → OUTSIDE
```

Event hanya saat transisi valid: `OUTSIDE→INSIDE` di `cam_out` = `IN`; `INSIDE→OUTSIDE` di `cam_in` = `OUT`.

---

## 7. Anti Double Counting

Implementasikan **semua**:

- tracking (`track_id` stabil);
- multi-frame confirmation (`counting.confirm_frames`);
- `min_track_len` sebelum boleh menghasilkan event;
- **cooldown per `track_id`** setelah event (`counting.cooldown_s`) — JANGAN cooldown global (dua orang beriringan 0,5 detik akan terbuang);
- identity confidence (similarity rendah tidak dicatat sebagai identitas valid);
- **dedup antar kamera**: occupancy tunggal + filter arah + dedup `(waktu, arah)`;
- track lahir di tengah / lewat garis tanpa histori approach → jangan langsung hitung, atau hitung sebagai `UNKNOWN` low-confidence.

**Event diproses per-track independen tiap frame.** Tidak ada "satu event dalam satu waktu". Dua orang nyebrang bareng = dua transisi = dua event.

---

## 8. Skenario Banyak Orang

| Skenario | Counting | Recognition |
|---|---|---|
| Satu-satu | ~97–99% | baik |
| Beriringan rapat | ~90–95% (tracker di-tune) | orang belakang → `UNKNOWN`, tetap terhitung |
| Gerombolan padat | turun signifikan | mayoritas gagal |

Batas fundamental kamera tunggal per arah. Mitigasi kode: deteksi kepala + tracker anti-occlusion + konfirmasi multi-frame + cooldown per track. Mitigasi fisik terkuat: persempit pintu jadi jalur satu-satu. **Ukur** per skenario pada probe kamera-pintu (missed/false/duplicate count, ID switch, identity accuracy) — jangan diasumsikan.

---

## 9. Database

### 9.1 SQLite lokal (offline-first)

Tabel: `persons`, `events`, `occupancy_state`, `persons_cache`. Skema lengkap di [PRD.md §14.1](PRD.md). Poin penting:

- `events` menyimpan `idpersonal` (NULL bila UNKNOWN), `identity`, `event_type`, `camera_id`, `similarity`, `all_scores` (JSON top-k untuk re-kalibrasi), `track_id`, `timestamp` (waktu **sistem**, bukan client), `date`, `image_path`.
- Embedding di `data/gallery/embeddings.npz`, bukan BLOB tabel.

### 9.2 PostgreSQL kepegawaian (eksternal)

- `SELECT * FROM person.get_info_person($1)` dengan `$1 = idpersonal`.
- **Tidak pernah** di jalur real-time. Worker enrichment asinkron: `idpersonal` baru → panggil fungsi → `persons_cache`.
- Aturan: DSN dari env var (bukan source/git); user **read-only**; **selalu parameterized** (`$1`); validasi `idpersonal` = UUID valid dulu; timeout pendek + retry backoff + circuit breaker; TLS bila lewat jaringan.
- Dashboard/laporan = JOIN `events × persons_cache`. PostgreSQL putus → event tetap tercatat, nama menyusul.

---

## 10. Daily Summary & Occupancy

Format: `idpersonal | Nama | IN | OUT | Current Status`. Status dari event terakhir (`occupancy_state`). Occupancy dibedakan known vs unknown. Akhir hari: `idpersonal` yang masih `INSIDE` dengan `last_event='IN'` = anomali → laporkan.

---

## 11. Cross-check Antar Kamera

Occupancy versi `cam_out` dan `cam_in` harus konsisten dengan occupancy gabungan. Selisih > ambang → alarm (crossing terlewat / kamera bergeser / terhalang).

---

## 12. Privasi (WAJIB)

Pemrosesan biometrik di bawah UU PDP No. 27/2022. Terapkan:

- papan pemberitahuan di pintu;
- kebijakan retensi terdokumentasi (`privacy.retention_days_events`, `database.keep_snapshots_days`);
- simpan embedding, bukan foto wajah mentah, bila memungkinkan;
- purpose limitation;
- kontrol akses DB; user PostgreSQL read-only;
- enrollment hanya untuk orang yang memberi persetujuan.

---

## 13. Performance

Target engineering awal (Jetson Nano 4GB, **2 kamera**):

```
Detection      : ≥ 8 FPS (engine di-share, bergantian antar kamera)
Pipeline/kamera : ≥ 5 FPS efektif
Recognition    : ≤ 300 ms/wajah, tiap every_n_frames
RAM            : ≤ 3 GB (satu set engine di-share)
Precision      : FP16 (TensorRT)
```

Target engineering, bukan jaminan. Bila rendah, berurutan: turunkan resolusi → optimalkan YOLO → TensorRT → FP16 → kurangi frekuensi recognition → frame skipping → batasi ROI → frame stagger antar kamera. Jangan ganti arsitektur tanpa profiling.

---

## 14. Struktur Project

Lihat [PRD.md §22](PRD.md). Ringkas:

```
main.py  config/config.yaml
app/{config,logging_setup,pipeline}.py
app/camera/{base,webcam,filesource,csi,factory}.py
app/{detection,tracking,recognition,counting,database,enrollment}/
scripts/ingest_raw.py   tests/   models/   data/   dataset/   logs/
```

Jangan buat satu file Python besar untuk seluruh pipeline.

---

## 15. Strategy Pengembangan (Capaian)

Bertahap. Setiap capaian menghasilkan sistem yang bisa dijalankan + diuji.

| # | Capaian | Fokus |
|---|---|---|
| 0 | Scaffold | struktur, config loader + validasi, `main.py --check`, tests |
| 1 | Camera | `CameraSource` dual-source, FPS stabil, shutdown bersih, recover |
| 1b | Kalibrasi lensa | matrix + dist_coeffs per kamera (CSI, barrel 120°) |
| 1c | Site survey | kamera terpasang, rekam klip semua skenario, set garis + ROI |
| 2 | Detection | YOLO person/head, engine di-share |
| 3 | Tracking | ByteTrack, ID stabil, tahan occlusion singkat, ukur ID switch |
| 4 | Face Recognition | enroll IMX219, matching 1:N, kalibrasi threshold pada probe kamera-pintu |
| 5 | Counting | pita 2 garis + state machine + filter arah + cooldown per track + dedup antar kamera |
| 6 | Database | event + timestamp sistem + camera_id + idpersonal; worker enrichment PostgreSQL |
| 7 | Daily Summary | rekap per idpersonal + occupancy known/unknown + anomali |
| 8 | Optimization | ONNX→TensorRT FP16, engine di-share, 2 kamera dalam anggaran Nano |
| 9 | Cross-check | occupancy cam_out vs cam_in + alarm |

Jangan implementasi seluruh sistem sekaligus.

---

## 16. Testing

Setiap module diuji independen.

- **Detection:** accuracy, false detection, missed detection.
- **Tracking:** ID switch, lost track, multiple people, occlusion.
- **Recognition:** known, unknown, pose, illumination, occlusion, distance.
- **Counting:** IN, OUT, reversal, multiple people, crossing simultan, tailgating, gerombolan.
- **System:** duplicate event, camera disconnect, database failure, PostgreSQL failure, low FPS, Jetson memory pressure, cross-camera occupancy drift.

---

## 17. Evaluation — pisahkan dua level

**Face recognition:** Accuracy, Precision, Recall, FAR, FRR, ROC, AUC, EER, TAR@FAR — dilaporkan pada `probe/test/` (bukan `calib/`).

**People counting:** IN accuracy, OUT accuracy, Missed count, False count, Duplicate count, ID switch — per skenario §8.

Jangan campur evaluasi face recognition dan people-flow counting.

---

## 18. Coding Rules

1. Jangan asumsikan library yang belum diverifikasi (terutama kompatibilitas Jetson Nano).
2. Prioritaskan kompatibilitas Jetson Nano. Kriteria pilih library: Compatibility → Performance → Memory → Accuracy → Maintenance. Jangan pilih karena paling populer/baru.
3. Dependency seminimal mungkin.
4. Konfigurasi terpisah dari source (`config/config.yaml`).
5. Jangan hard-code threshold, margin, koordinat garis, ROI.
6. Jangan simpan API key / password / DSN di source atau git — pakai environment variable.
7. Query PostgreSQL selalu parameterized (`$1`). Validasi `idpersonal` = UUID.
8. Timestamp event dari sistem, bukan client.
9. Tambahkan logging + error handling di setiap modul. Kode dapat di-debug.
10. Jangan premature optimization. Profiling sebelum optimasi.
11. Setiap perubahan arsitektur harus punya alasan teknis.
12. Jangan hapus fitur existing tanpa alasan + konfirmasi.
13. Jangan ubah environment Jetson secara destruktif.
14. Di Jetson: pakai OpenCV sistem (build JetPack dengan GStreamer + CUDA). JANGAN `pip install opencv-python` di Jetson.
15. Counting tidak boleh bergantung pada Recognition.

---

## 19. Aturan Khusus AI Coding Agent

Sebelum membuat kode:

1. Periksa struktur project yang ada.
2. Periksa versi Python, CUDA, TensorRT, OpenCV, GStreamer.
3. Periksa dependency terinstall.
4. Identifikasi library kompatibel Jetson Nano.
5. Jelaskan dependency baru sebelum memasang.

Sebelum perubahan besar, jelaskan: **Problem → Cause → Proposed Solution → Impact.**

Setiap tahap harus menghasilkan sistem yang dapat dijalankan dan diuji.

---

## 20. Prinsip Utama

Sistem ini bukan sekadar `YOLO = people counter`. Melainkan:

```
Camera → WHO IS THERE? → TRACK → WHO IS THE PERSON? (idpersonal/UNKNOWN)
       → WHERE ARE THEY GOING? → DID THEY CROSS THE BAND?
       → IN / OUT → STORE EVENT → (async) RESOLVE NAME → REPORT
```

Semua implementasi mempertahankan arsitektur tersebut, dengan **Counting sebagai fitur utama yang tidak boleh gagal** dan **Recognition sebagai lapisan pengaya yang boleh menghasilkan `UNKNOWN`**.
