# Product Requirements Document (PRD)

## Sistem Penghitungan Keluar-Masuk Orang + Pengenalan Wajah Berbasis Kamera, YOLO, Face Recognition, dan Jetson Nano

**Versi:** 2.0
**Platform:** NVIDIA Jetson Nano Developer Kit 4GB (B01, 2× port CSI)
**Kamera:** 2× IMX219-160 (CSI / MIPI)
**Target OS:** JetPack / L4T R32.7.1
**Python:** 3.6.9 (Jetson) / 3.9+ (pengembangan di laptop)
**CUDA:** 10.2
**Mode:** Edge AI / Offline-first

> Perubahan besar dari v1.0: dua kamera CSI (satu per arah), penghitungan memakai pita dua garis, pengenalan wajah sebagai identifikasi 1:N open-set, dua basis data (SQLite lokal + PostgreSQL kepegawaian), dataset `raw/` di-*shelve* dan diganti perekaman ulang IMX219, bagian privasi UU PDP, presensi kedinasan dipisah menjadi Fase 2.

---

# 1. Ringkasan Produk

Sistem computer vision berbasis edge AI. Dua kamera memantau satu pintu gedung/ruangan untuk:

1. **Menghitung** jumlah orang yang masuk (`IN`) dan keluar (`OUT`) — fitur utama.
2. **Mengenali identitas** orang yang lewat dan menautkannya ke `idpersonal` — lapisan pengaya di atas penghitungan.

Penghitungan dan pengenalan adalah dua lapisan terpisah. Pengenalan boleh gagal (menghasilkan `UNKNOWN`) tanpa merusak angka penghitungan.

Hasil yang dapat diturunkan sistem:

- siapa yang masuk dan keluar;
- berapa kali seseorang masuk/keluar dalam sehari;
- kapan setiap event terjadi;
- siapa saja yang sedang berada di dalam (occupancy);
- jumlah orang tak dikenal yang lewat.

Contoh rekap:

| idpersonal | Nama  | Masuk | Keluar | Di Dalam |
|---|---|---:|---:|---:|
| a4f9… | Juris | 1 | 1 | Tidak |
| c8b1… | Gusti | 2 | 2 | Tidak |
| f22e… | Rizki | 1 | 0 | Ya |

Seluruh proses inti (deteksi, tracking, recognition, counting, penyimpanan event) berjalan di Jetson Nano tanpa internet. Resolusi nama/informasi orang dari basis data kepegawaian bersifat opsional dan asinkron.

---

# 2. Tujuan Produk

## 2.1 Tujuan Utama

Sistem otomatis untuk menghitung dan mencatat aktivitas keluar-masuk individu melalui satu pintu menggunakan computer vision, dengan identitas orang terdaftar tertaut ke setiap event bila wajah berhasil dikenali.

## 2.2 Tujuan Teknis

Sistem harus mampu:

1. menangkap video dari 2 kamera CSI secara bersamaan;
2. mendeteksi orang/kepala pada frame;
3. melakukan tracking antar-frame;
4. mengenali identitas individu terdaftar (identifikasi 1:N), atau menandai `UNKNOWN`;
5. mendeteksi crossing terhadap pita dua garis virtual;
6. menentukan arah `IN` atau `OUT` dari urutan crossing;
7. mencegah double counting (per track, per kamera, dan antar kamera);
8. menyimpan event ke SQLite lokal beserta `idpersonal`/`UNKNOWN`, `camera_id`, dan timestamp sistem;
9. menghasilkan rekap harian dan occupancy;
10. mengambil nama/informasi orang dari PostgreSQL kepegawaian secara asinkron (tidak di jalur real-time);
11. berjalan dalam anggaran sumber daya Jetson Nano 4GB.

## 2.3 Motivasi

Penghitungan saja dinilai kurang oleh pimpinan. Pengenalan wajah ditambahkan untuk:

- mengurangi risiko kriminal (mengetahui siapa yang masuk, alarm bila orang tak dikenal menumpuk di luar jam kerja);
- **potensi** presensi kedinasan (Fase 2, lihat §26).

---

# 3. Non-Goals

Versi ini tidak bertujuan untuk:

- surveillance area luas atau banyak pintu;
- mengenali seluruh populasi tanpa registrasi;
- menggantikan sistem keamanan profesional;
- identifikasi dari wajah sangat kecil / tertutup sepenuhnya;
- melatih atau fine-tune model face recognition (memakai model pretrained apa adanya, lihat §8.3);
- cloud inference sebagai komponen utama;
- penghitungan yang tahan gerombolan padat pada konfigurasi satu kamera per arah (lihat §12 untuk batasan);
- presensi kedinasan pada rilis pertama (Fase 2).

---

# 4. Prinsip Arsitektur Utama

```
DETECTION ≠ TRACKING ≠ RECOGNITION ≠ COUNTING
```

| Modul | Pertanyaan yang dijawab |
|---|---|
| YOLO | "Di mana orangnya?" |
| Tracker (ByteTrack) | "Apakah ini orang yang sama antar-frame?" |
| Face Recognition | "Siapa orang ini? (idpersonal / UNKNOWN)" |
| Line Crossing | "Dia bergerak masuk atau keluar?" |
| Database | "Apa histori orang ini hari ini? Berapa occupancy?" |

Aturan tambahan yang wajib dijaga di seluruh implementasi:

- **Counting independen dari Recognition.** Penghitungan = tracking + crossing. Tidak butuh wajah. Recognition gagal → event tetap tercatat sebagai `UNKNOWN`, angka tetap benar.
- **`track_id` bukan identitas.** `track_id` bersifat sementara per kamera. `idpersonal` berasal dari face recognition dan ditautkan ke track setelah *voting* beberapa hasil.
- Setiap modul adalah class/module terpisah sehingga model dapat diganti tanpa mengubah keseluruhan sistem.

---

# 5. Arsitektur Sistem

## 5.1 Tata Letak Dua Kamera

Satu kamera hanya melihat wajah untuk satu arah. Karena itu dipakai dua kamera pada satu pintu:

| Kamera | Pemasangan | Tanggung jawab |
|---|---|---|
| `cam_out` | **di luar** pintu, menghadap jalur pendekatan luar | event `IN` + identitas orang **masuk** |
| `cam_in` | **di dalam** pintu, menghadap jalur pendekatan dalam | event `OUT` + identitas orang **keluar** |

Occupancy tunggal disuplai kedua kamera dengan dedup berdasarkan `(waktu, arah)`. Filter arah: `cam_out` hanya mencatat transisi `LUAR→DALAM`, `cam_in` hanya `DALAM→LUAR`.

## 5.2 Pipeline per Kamera

```
CSI CAMERA (IMX219-160)
        │
        ▼
  Frame Capture  ── (opsional) undistort lensa 160°
        │
        ▼
  YOLO Detection (person / head)
        │
        ▼
  ByteTrack  ─────────────► track_id, bbox, trajectory
        │
        ▼
  Face Detect + Align (setiap N frame, dalam ROI)
        │
        ▼
  Face Embedding (ArcFace)
        │
        ▼
  Gallery Matching (cosine, threshold + margin)
        │
        ▼
  Identity: idpersonal / UNKNOWN  ──► di-vote lalu dikunci ke track_id
        │
        ▼
  Line Crossing (pita 2 garis) + State Machine + Direction Filter
        │
        ▼
  IN / OUT Event  { idpersonal|UNKNOWN, camera_id, ts, confidence, track_id, snapshot? }
        │
        ▼
  SQLite lokal
        │
        ├──► Rekap harian / Occupancy / Cross-check antar kamera
        │
        └──► (asinkron, best-effort) Enrichment worker
                     │
                     ▼
             PostgreSQL kepegawaian: SELECT * FROM person.get_info_person($1)
                     │
                     ▼
             persons_cache (nama + info) di SQLite lokal
                     │
                     ▼
             Dashboard / Laporan (JOIN event × persons_cache)
```

## 5.3 Efisiensi Komputasi (dua kamera, satu Jetson)

- **Engine model di-*share*.** Satu engine YOLO dan satu engine face embedding dipakai bergantian untuk frame kedua kamera. Hanya tracker + state yang per-kamera. Ini menjaga jejak memori setara satu set model.
- Face recognition tidak dijalankan setiap frame (`recognition.every_n_frames`).
- Deteksi pada resolusi diturunkan (`detection.input_size`).
- Opsi *frame stagger*: kamera A diproses pada frame genap, kamera B pada frame ganjil.

---

# 6. Target Hardware

| Parameter | Nilai |
|---|---|
| Device | NVIDIA Jetson Nano Developer Kit 4GB (B01) |
| Arsitektur | ARM64 / aarch64 |
| JetPack | R32.7.1 |
| CUDA | 10.2 |
| Python (Jetson) | 3.6.9 |
| Kamera | 2× IMX219-160 CSI, mode 1280×720 atau 1920×1080 |
| Power | Barrel jack 5V/4A (jangan micro-USB), mode 10W/MAXN |
| Pendinginan | Kipas aktif wajib (beban GPU + 2 kamera berkelanjutan) |
| Penyimpanan | ≥ 32 GB (OS + model + event + snapshot) |

Jetson adalah **inference device**, bukan mesin training. Kendala fisik: FFC bawaan IMX219-160 ±15 cm — Jetson harus berada dalam ~1 m dari kamera, atau memakai FFC/extender berkualitas (kabel buruk = noise gambar). Pertimbangkan enclosure Jetson tepat di atas pintu.

---

# 7. Layout Kamera & Pemasangan

## 7.1 Posisi

- `cam_out`: di atas kusen sisi luar, tinggi target **2,1–2,3 m**, sudut nunduk **15–25°**, menghadap jalur orang mendekati pintu dari luar.
- `cam_in`: cermin dari `cam_out` di sisi dalam.
- Serendah mungkin di atas kusen. Setiap +10 cm tinggi = sudut lebih curam = wajah lebih *top-down* = akurasi recognition turun. Di atas ~2,7 m dengan sudut curam, wajah tertutup dahi/rambut.
- Zona wajah tajam diarahkan ke **1,5–3 m** di depan pintu.

## 7.2 Backlight

Pintu kaca gedung = backlight. Ini mode gagal yang tercatat pada sistem people-counting komersial (untuk penghitungan **dan** recognition). IMX219 memiliki WDR/HDR lemah. Mitigasi:

- atur sudut agar menghindari garis matahari langsung;
- pertimbangkan fill light sisi dalam;
- bila memungkinkan, pilih waktu perekaman probe di kondisi cahaya terburuk agar threshold dikalibrasi konservatif.

## 7.3 Layout Fisik Pintu (mitigasi terkuat untuk gerombolan)

Bila layout memungkinkan, persempit jalur masuk menjadi **satu-satu** (koridor sempit / tali / turnstile). Ini peningkatan keandalan terbesar dan termurah untuk counting maupun recognition (lihat §12).

## 7.4 Site Survey (Capaian 1c)

Setelah kamera terpasang fisik:

1. Rekam klip 1280×720 dari `cam_out` dan `cam_in` di posisi asli: orang lewat `IN` dan `OUT`, skenario satu-satu / beriringan 2 / beriringan 3 / gerombolan 4–5 / sejajar 2.
2. Ukur tinggi wajah dalam piksel saat berada di pita garis → harus ≥ 112 px.
3. Cek sudut wajah saat approach → pitch < ~25°.
4. Set `counting.band.<cam>` (garis luar/dalam), `recognition.roi`, parameter undistort.
5. Klip menjadi *probe set* untuk Capaian 4 (kalibrasi threshold) dan Capaian 5 (uji counting).

---

# 8. Komponen Computer Vision

## 8.1 Object / Head Detection

YOLO mendeteksi target pada frame.

- Kelas dasar: `person`.
- Opsi untuk kondisi ramai: model deteksi **kepala** (mis. dilatih pada CrowdHuman). Kepala lebih terpisah dari atas dibanding badan.
- Output minimal per deteksi: `class`, `confidence`, `bbox [x1,y1,x2,y2]`.
- Face recognition tidak dijalankan pada seluruh frame — hanya pada ROI relevan.

Contoh output:

```json
{ "class": "person", "confidence": 0.94, "bbox": [320, 120, 520, 620] }
```

Model dioptimalkan untuk Jetson Nano (ONNX → TensorRT FP16, Fase Optimasi).

## 8.2 Tracking

**ByteTrack** (alternatif: SORT, DeepSORT).

- Menghasilkan: `track_id`, `bbox`, `confidence`, `trajectory`.
- `track_buffer` besar (mis. 45–60 frame) untuk menjembatani occlusion singkat (tailgating) sehingga `track_id` orang di belakang tetap bertahan.
- `track_id` ≠ `idpersonal`. Keduanya informasi terpisah.

Tujuan tracking: mencegah double counting, mengetahui trajectory, menentukan crossing, dan menautkan hasil face recognition ke objek.

## 8.3 Face Recognition

Menggunakan **face embedding** (bukan classifier per orang).

```
Face → Face Alignment (5-titik, 112×112) → Face Embedding (ArcFace) → Vektor embedding
```

**Tidak ada training / fine-tuning model.** Alasan:

- Model ArcFace pretrained sudah general dari jutaan wajah.
- Jumlah orang terdaftar (puluhan–ratusan) jauh dari cukup untuk melatih apa pun; memaksakan training = overfitting parah.
- *Enrollment* seseorang bukan training. Enrollment hanya menghasilkan embedding yang dimasukkan ke gallery.

Model kandidat: ArcFace / InsightFace (R50 / MobileFaceNet) dalam format ONNX. Sebelum implementasi final, verifikasi kompatibilitas dengan JetPack R32.7.1, CUDA 10.2, Python 3.6, ARM64. Jangan mengasumsikan library modern otomatis kompatibel dengan Jetson Nano. Catatan: `insightface` via pip kemungkinan gagal di Python 3.6 Jetson — rencana cadangan: memakai model ONNX mentah + preprocessing manual.

## 8.4 Identity Matching (Identifikasi 1:N Open-set)

```
query_embedding → cosine similarity ke seluruh gallery → top-k
        → keputusan: (top1 ≥ threshold) AND (top1 − top2 ≥ margin)  → idpersonal
                     selain itu                                     → UNKNOWN
        → hasil di-vote sepanjang jendela (recognition.vote_window) sebelum dikunci ke track_id
```

- Mayoritas orang lewat **tidak terdaftar** → sistem harus sering menghasilkan `UNKNOWN` dengan benar (open-set).
- Semakin besar gallery, semakin besar peluang wajah asing menyangkut ke suatu entri → FAR per-perbandingan harus rendah (≈ 1e-4 s/d 1e-5). `margin` mencegah pemilihan identitas terdekat yang lemah.
- Jangan pernah memaksakan identitas dengan similarity tertinggi bila di bawah threshold.
- Threshold dan margin **dapat dikonfigurasi** dan **dikalibrasi** (lihat §10). Tidak boleh hard-code.
- Search 1:N (ratusan–ribuan vektor cosine per pemanggilan) murah di Jetson. FAISS belum diperlukan; catat untuk skala > 100k.

---

# 9. Gallery & Enrollment

## 9.1 Kebijakan Data

- **Perekaman ulang dengan IMX219** pada posisi pintu terpasang. Ini menghilangkan *domain gap* (selfie 40 cm ≠ kamera pintu 2–3 m nunduk fisheye), memberi kontrol kualitas, dan label identitas pasti benar.
- Dataset lama `raw/` (dump enrollment aplikasi mobile terdahulu) **di-*shelve***: 1420 folder `idpersonal`, mayoritas hanya 1 foto usable, domain selfie. Tetap disimpan (gitignore) — satu-satunya kegunaan tersisa: **pool impostor** untuk uji FAR. Diaudit oleh `scripts/ingest_raw.py`.

## 9.2 Protokol Enrollment

Target: **15–20 citra per orang**, ditautkan ke `idpersonal` asli (dipilih dari daftar orang di PostgreSQL saat capture).

Variasi wajib (jangan 20 citra hampir identik):

| Dimensi | Nilai |
|---|---|
| Pose | frontal, yaw ±30° kiri/kanan, pitch sedikit atas/bawah |
| Ekspresi | netral, bicara, senyum |
| Kacamata | dengan / tanpa (bila dipakai) |
| Pencahayaan | normal, terang, redup, backlight |
| Jarak | ~1 m, ~2 m (operasional), ~3 m |

## 9.3 Quality Check Otomatis (tolak bila gagal)

1. confidence face detector ≥ ambang;
2. ketajaman (variance of Laplacian) ≥ ambang — tolak motion blur;
3. ukuran wajah ≥ `recognition.min_face_px` (112 px);
4. yaw/pitch < 30°;
5. exposure wajar (tidak terlalu gelap / *blown*);
6. dedup: tolak bila cosine similarity > 0,97 terhadap sample yang sudah diterima.

## 9.4 Agregasi Embedding

- Simpan **semua** embedding per-sample untuk tiap `idpersonal` + satu *mean embedding* (L2-normalized).
- Matching: max cosine similarity ke salah satu sample, atau mean top-k.
- Path: `gallery.embeddings_path` (`data/gallery/embeddings.npz`).

## 9.5 Perkayaan Bertahap (opsional)

Saat terjadi match confident di pintu, embedding kamera-pintu itu boleh ditambahkan ke gallery `idpersonal` tersebut (*template update*) dengan guardrail (confidence tinggi, margin lebar, batas jumlah per hari). Tujuan: gallery makin cocok dengan domain kamera pintu seiring waktu.

## 9.6 Pemisahan Gallery dan Probe

```
dataset/
├── gallery/            enrollment 15–20/orang (dari kamera pintu)
└── probe/
    ├── calib/          untuk kalibrasi threshold
    └── test/           metrik final — JANGAN disentuh saat kalibrasi
        ├── clean/  pose/  illumination/  occlusion/  combined/
```

Robustness tidak boleh dinilai memakai data yang terlalu mirip dengan data enrollment.

---

# 10. Kalibrasi Ambang (Threshold)

Threshold cosine similarity + margin adalah **satu-satunya** parameter yang "di-tuning" pada sistem ini.

1. Susun pasangan **genuine** (orang sama) dan **impostor** (orang beda) dari `probe/calib/` — utamakan citra dari kamera pintu; `raw/` boleh sebagai pool impostor tambahan.
2. Hitung cosine similarity tiap pasangan.
3. Sapu (sweep) `τ` dari 0 ke 1; hitung FAR(τ) dan FRR(τ).
4. `τ_EER` = titik FAR(τ) = FRR(τ) → baseline.
5. Pintu perlu keamanan lebih tinggi → geser `τ` naik: FAR turun, FRR naik sedikit. Dokumentasikan sebagai keputusan desain.
6. **Validasi ulang `τ`** pada kondisi kamera pintu sebenarnya (cahaya, sudut, kualitas sensor) — distribusi similarity dapat bergeser.
7. Laporkan EER final pada `probe/test/` (bukan `calib/`) sebagai metrik akhir, supaya evaluasi tidak overfitting terhadap ambang.

Metrik: Accuracy, FAR, FRR, ROC, AUC, EER, TAR@FAR.

---

# 11. Line Crossing (Pita Dua Garis)

Area pintu dikonfigurasi dengan **dua garis sejajar** membentuk pita:

```
─────────── garis LUAR
   (pita)                urutan LUAR → DALAM  = IN
─────────── garis DALAM  urutan DALAM → LUAR  = OUT
```

- Arah ditentukan dari **urutan crossing** kedua garis, bukan posisi satu frame.
- Anchor: **kepala / centroid** (`counting.crossing_anchor`), bukan pusat bbox badan.
- Koordinat garis = fraksi frame, di-set per kamera pada Capaian 1c. Tidak boleh hard-code.
- Poligon zona (mis. `inside`/`outside` + `walk-in`/`walk-out`/`pass-by`) adalah alternatif setara — murni geometri gambar pada satu kamera.

## 11.1 Event State Machine (per track)

```
UNKNOWN → OUTSIDE → (crossing pita) → INSIDE → (crossing pita) → OUTSIDE
```

Event hanya dibuat ketika transisi state valid:

- `OUTSIDE → INSIDE` pada `cam_out` → `<identitas> IN`
- `INSIDE → OUTSIDE` pada `cam_in` → `<identitas> OUT`

## 11.2 Anti Double Counting

| Mekanisme | Keterangan |
|---|---|
| Tracking | orang sama mempertahankan `track_id` |
| Multi-frame confirmation | crossing dikonfirmasi `counting.confirm_frames`, bukan satu frame |
| Cooldown per track | setelah event, track yang sama tidak langsung menerima event berikutnya (`counting.cooldown_s`) |
| `min_track_len` | track harus eksis minimal N frame sebelum boleh menghasilkan event |
| Identity confidence | identitas similarity rendah tidak dicatat sebagai identitas valid |
| Dedup antar kamera | occupancy tunggal + filter arah + dedup `(waktu, arah)` |
| Track lahir di tengah/lewat garis tanpa histori approach | tidak langsung dihitung, atau dihitung sebagai `UNKNOWN` low-confidence |

---

# 12. Skenario Banyak Orang — Kemampuan dan Batas

| Skenario | Counting | Recognition |
|---|---|---|
| Satu-satu, ada jarak | ~97–99% | baik |
| Beriringan rapat (1–2 di belakang) | ~90–95% bila tracker di-tune | orang belakang sering gagal → `UNKNOWN`, tetap terhitung |
| Gerombolan padat (3+ sejajar, saling menutup) | turun signifikan (undercount karena bbox menyatu, ID switch) | mayoritas gagal |

**Akar masalah:** satu kamera tidak bisa optimal untuk dua hal sekaligus. Face recognition butuh kamera menghadap wajah, nunduk ~15–25°. Penghitungan gerombolan yang andal butuh kamera *top-down* ~60–90° (dari atas kepala tidak saling menutup — sistem people-counter komersial memakai top-down + deteksi kepala + zona, dan sengaja **tidak** melakukan face recognition).

**Mitigasi yang dipilih:** deteksi kepala (bukan badan) + tracker anti-occlusion + konfirmasi crossing multi-frame + cooldown per track. **Peningkatan terbesar dan termurah = mempersempit pintu menjadi jalur satu-satu (§7.3).**

**Wajib diukur** (bukan diasumsikan) pada probe kamera-pintu: missed count, false count, duplicate count, ID switch, identity accuracy — per skenario. Angka ini menjadi dasar keputusan pimpinan (menerima error gerombolan, atau menambah kamera ketiga *top-down* khusus penghitungan).

---

# 13. Unknown Person

Bila wajah tidak cocok dengan gallery (`similarity < threshold` atau `margin` tidak terpenuhi):

```
identity = UNKNOWN
```

Sistem membedakan: **Known Person**, **Unknown Person**, **No Face**, **Low Confidence**.

- Event `UNKNOWN IN` / `UNKNOWN OUT` tetap disimpan (statistik anonim + occupancy).
- Alarm opsional: `UNKNOWN` menumpuk di luar jam kerja.
- Tamu: bila perlu dibedakan dari "orang asing", diperlukan alur pra-registrasi tamu (enroll sementara, kedaluwarsa). Bila tidak, tamu = `UNKNOWN`.

---

# 14. Database

## 14.1 SQLite Lokal (Jetson, offline-first)

### `persons` (cache identitas lokal + metadata gallery)

```sql
CREATE TABLE persons (
    idpersonal   TEXT PRIMARY KEY,          -- UUID dari sistem kepegawaian
    enrolled_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    n_samples    INTEGER NOT NULL DEFAULT 0,
    source       TEXT DEFAULT 'imx219'       -- imx219 | raw | mixed
);
```

Embedding disimpan di `data/gallery/embeddings.npz` (bukan BLOB di tabel), di-*key* oleh `idpersonal`.

### `events`

```sql
CREATE TABLE events (
    id          INTEGER PRIMARY KEY,
    idpersonal  TEXT,                        -- NULL bila UNKNOWN
    identity    TEXT NOT NULL,               -- idpersonal atau 'UNKNOWN'
    event_type  TEXT NOT NULL CHECK (event_type IN ('IN','OUT')),
    camera_id   TEXT NOT NULL CHECK (camera_id IN ('cam_out','cam_in')),
    confidence  REAL,
    similarity  REAL,                        -- top-1 cosine similarity saat keputusan
    all_scores  TEXT,                        -- JSON top-k skor, untuk re-kalibrasi threshold nanti
    track_id    INTEGER,
    timestamp   DATETIME NOT NULL,           -- waktu SISTEM, bukan dari client
    date        DATE NOT NULL,
    image_path  TEXT
);
```

### `occupancy_state`

```sql
CREATE TABLE occupancy_state (
    idpersonal   TEXT PRIMARY KEY,
    entered_at   DATETIME NOT NULL,
    last_event   TEXT NOT NULL                -- 'IN' | 'OUT'
);
```

### `persons_cache` (hasil enrichment dari PostgreSQL)

```sql
CREATE TABLE persons_cache (
    idpersonal  TEXT PRIMARY KEY,
    name        TEXT,
    info_json   TEXT,                         -- baris get_info_person() sebagai JSON opaque
    fetched_at  DATETIME NOT NULL
);
```

## 14.2 PostgreSQL Kepegawaian (eksternal)

- Query: `SELECT * FROM person.get_info_person($1)` dengan `$1 = idpersonal`.
- **Tidak pernah** dipanggil di jalur real-time. Worker enrichment asinkron: untuk `idpersonal` baru pada `events` → panggil fungsi → simpan ke `persons_cache`.
- PostgreSQL putus/lambat → event tetap tercatat dengan `idpersonal`, nama menyusul.
- Aturan:
  - DSN dari environment variable (`personnel_db.dsn_env`), bukan di source, bukan di git;
  - user PostgreSQL **read-only**;
  - query selalu parameterized (`$1`) — jangan format string `idpersonal` ke SQL;
  - validasi `idpersonal` = UUID valid sebelum query;
  - timeout pendek + retry backoff + circuit breaker;
  - TLS bila melewati jaringan.
- Diperlukan sebelum Capaian 6: host/port/kredensial, daftar kolom yang dikembalikan `get_info_person()`, dan (bila ada) fungsi untuk melisting seluruh orang (untuk UI enrollment admin).

---

# 15. Rekap Harian & Occupancy

```
================================
DAILY PEOPLE FLOW — 2026-09-08
================================

Juris  (a4f9…)   IN: 1   OUT: 1   INSIDE: NO
Gusti  (c8b1…)   IN: 2   OUT: 2   INSIDE: NO
Rizki  (f22e…)   IN: 1   OUT: 0   INSIDE: YES
UNKNOWN          IN: 5   OUT: 4

--------------------------------
Current Occupancy : 1 known + 1 unknown
--------------------------------
```

- `INSIDE` / status ditentukan dari event terakhir per `idpersonal` (`occupancy_state`).
- Akhir hari: `idpersonal` yang masih di `occupancy_state` dengan `last_event='IN'` = anomali (exit tidak terbaca / orang menginap) → laporkan.

---

# 16. Dashboard & Live Monitoring (prioritas Could)

Dashboard minimal:

```
┌────────────────────────────────────┐
│ PEOPLE FLOW MONITORING             │
├────────────────────────────────────┤
│ Orang di dalam        1 (+1 unk)   │
│ Total IN              17           │
│ Total OUT             14           │
├────────────────────────────────────┤
│ Nama    IN  OUT  Status            │
│ Juris    1   1   OUTSIDE           │
│ Gusti    2   2   OUTSIDE           │
│ Rizki    1   0   INSIDE            │
└────────────────────────────────────┘
```

Live monitoring: stream tiap kamera + bounding box (`idpersonal`/nama, `track_id`, similarity, `IN`/`OUT`) + garis pita virtual.

---

# 17. Cross-check Occupancy Antar Kamera

- Occupancy versi `cam_out` (Σ IN − Σ OUT yang teramati di FOV-nya) dan `cam_in` harus konsisten dengan occupancy gabungan.
- Selisih di atas ambang → alarm ("ada crossing terlewat di salah satu kamera").
- Berguna untuk mendeteksi kamera bergeser, terhalang, atau gagal.

---

# 18. Privasi & Kepatuhan

Pengenalan wajah semua orang yang masuk + penautan ke basis data identitas + (Fase 2) presensi = **pemrosesan data biometrik**. UU PDP No. 27/2022: biometrik = data pribadi bersifat spesifik.

Persyaratan (bukan opsional):

- **Papan pemberitahuan** di pintu masuk (tujuan pemrosesan, pengendali data, kontak).
- **Kebijakan retensi** terdokumentasi: `privacy.retention_days_events`, `database.keep_snapshots_days`. Snapshot retensi pendek.
- **Simpan embedding, bukan foto wajah mentah** bila memungkinkan (`privacy.store_face_crops: false`).
- **Purpose limitation**: data tidak dipakai ulang untuk tujuan lain tanpa dasar baru.
- **Kontrol akses** ke SQLite lokal dan PostgreSQL; user PostgreSQL read-only.
- Enrollment hanya untuk orang yang memberikan persetujuan untuk pintu ini.

---

# 19. Performance Requirement

Target engineering awal pada Jetson Nano 4GB, **dua kamera**:

| Parameter | Target |
|---|---|
| Resolusi input | 640×480 (deteksi) dari sumber 1280×720 |
| Detection | ≥ 8 FPS (engine di-share, bergantian antar kamera) |
| End-to-end per kamera | ≥ 5 FPS efektif (cukup untuk kecepatan jalan kaki) |
| Recognition latency | ≤ 300 ms per wajah, dijalankan tiap `every_n_frames` |
| RAM | ≤ 3 GB (satu set engine di-share) |
| Precision | TensorRT FP16 |
| GPU | TensorRT |

Angka ini **target engineering**, bukan jaminan. Nilai final diukur pada model, kamera, dan konfigurasi Jetson yang dipakai.

Bila performa rendah, berurutan: turunkan resolusi → optimalkan/ganti model YOLO → TensorRT FP16 → kurangi frekuensi face recognition → perbesar frame skipping → batasi ROI → *frame stagger* antar kamera. Jangan mengganti seluruh arsitektur tanpa profiling.

---

# 20. Optimasi Jetson Nano

Tahap awal (pipeline benar dulu di laptop):

```
Python → OpenCV → YOLO → Face Recognition
```

Setelah pipeline benar:

```
ONNX → TensorRT → FP16 → engine di-share antar kamera
```

Optimasi terhadap: input resolution, detection frequency, recognition frequency, batch size, TensorRT engine, memory allocation, frame skipping, frame stagger.

Contoh alokasi:

```
YOLO        → tiap frame (bergantian antar kamera)
Tracker     → tiap frame, per kamera
Recognition → tiap 5–10 frame, dalam ROI
Line        → berdasarkan tracker
Enrichment  → worker asinkron, di luar loop
```

---

# 21. Software Stack

```
OS
└── JetPack / L4T R32.7.1

Runtime
├── Python 3.6.9 (Jetson) / 3.9+ (laptop dev)
├── CUDA 10.2
├── OpenCV (Jetson: build sistem dengan GStreamer + CUDA — JANGAN pip install opencv-python di Jetson)
├── GStreamer (nvarguscamerasrc untuk CSI)
└── TensorRT

Computer Vision
├── YOLO (person / head)
├── ByteTrack
└── ArcFace / InsightFace (ONNX)

Storage
├── SQLite (lokal, event + gallery meta + cache)
└── PostgreSQL (eksternal, kepegawaian — read-only)

Optional Backend
├── FastAPI / Flask
└── REST API + Dashboard web
```

Pemilihan versi library disesuaikan dengan CUDA 10.2, JetPack R32.7.1, Python 3.6, ARM64. Kriteria pemilihan berurutan: **Compatibility → Performance → Memory → Accuracy → Maintenance**. Jangan memilih library hanya karena paling populer/baru.

---

# 22. Struktur Project

```
CountingApp/
├── main.py                     entry point (--check untuk validasi config)
├── requirements.txt
├── README.md
├── config/
│   └── config.yaml             seluruh tuning runtime
├── app/
│   ├── config.py               load + validasi YAML
│   ├── logging_setup.py
│   ├── pipeline.py             orchestrator (detect/track/recognize/count menempel di sini)
│   ├── camera/
│   │   ├── base.py             CameraSource ABC + Frame
│   │   ├── webcam.py  filesource.py  csi.py
│   │   └── factory.py
│   ├── detection/              YOLO wrapper
│   ├── tracking/               ByteTrack wrapper
│   ├── recognition/            detector, embedder, matcher, gallery
│   ├── counting/               line/band, crossing, state machine, dedup
│   ├── database/               sqlite repo, models, enrichment worker
│   └── enrollment/             capture.py, generate_embedding.py
├── scripts/
│   └── ingest_raw.py           audit dataset raw/ (shelved)
├── models/                     *.onnx, *.engine (gitignore)
├── data/                       gallery/, snapshots/, events.db (gitignore)
├── dataset/                    gallery/, probe/ (gitignore)
├── tests/
└── logs/
```

Jangan membuat satu file Python besar yang menangani seluruh pipeline.

---

# 23. Functional Requirements

| ID | Requirement | Prioritas |
|---|---|---|
| FR-001 | Sistem menerima input dari 2 kamera CSI bersamaan | Must |
| FR-002 | Sistem mendeteksi orang/kepala | Must |
| FR-003 | Sistem melakukan tracking per kamera | Must |
| FR-004 | Sistem mengenali individu (1:N) atau menandai `UNKNOWN` | Must |
| FR-005 | Sistem menentukan `IN`/`OUT` dari pita dua garis + filter arah | Must |
| FR-006 | Sistem mencegah double counting (per track, per kamera, antar kamera) | Must |
| FR-007 | Sistem menyimpan event ke SQLite dengan `idpersonal`/`UNKNOWN`, `camera_id`, timestamp sistem | Must |
| FR-008 | Sistem menghasilkan rekap harian + occupancy | Must |
| FR-009 | Sistem menangani open-set (mayoritas `UNKNOWN`) dengan benar | Must |
| FR-010 | Sistem menyediakan enrollment dari kamera IMX219 (15–20 sampel/orang, quality check) | Must |
| FR-011 | Threshold + margin dapat dikonfigurasi dan dikalibrasi (FAR/FRR/EER) | Must |
| FR-012 | Worker asinkron mengambil nama/info dari `person.get_info_person($1)` ke `persons_cache` | Must |
| FR-013 | Sistem berjalan tanpa internet (inti CV + penyimpanan event) | Must |
| FR-014 | Cross-check occupancy `cam_out` vs `cam_in` + alarm selisih | Should |
| FR-015 | Live monitoring (stream + overlay) | Should |
| FR-016 | Sistem menyimpan snapshot event + `all_scores` untuk re-kalibrasi | Should |
| FR-017 | Perkayaan gallery bertahap (template update) dengan guardrail | Could |
| FR-018 | REST API | Could |
| FR-019 | Dashboard web | Could |
| FR-020 | Alur pra-registrasi tamu | Could |
| FR-021 | Modul presensi kedinasan (Fase 2) | Won't (rilis 1) |

---

# 24. Non-Functional Requirements

- **Performance:** real-time / near-real-time pada Jetson Nano 4GB dengan 2 kamera (§19).
- **Reliability:** kehilangan satu/beberapa frame tidak boleh menghasilkan event `IN`/`OUT` palsu. Kamera terputus → pipeline berhenti bersih + log, bukan crash. PostgreSQL terputus → enrichment tertunda, counting jalan terus.
- **Privacy:** §18. Data wajah/embedding disimpan aman, retensi terbatas.
- **Offline:** inti computer vision + penyimpanan event berjalan tanpa internet.
- **Maintainability:** Detection / Tracking / Recognition / Counting / Database terpisah; model dapat diganti tanpa mengubah keseluruhan sistem. Konfigurasi terpisah dari source. Tidak ada threshold/koordinat hard-code. Logging + error handling di setiap modul.
- **Reproducibility:** `requirements.txt` / `pip freeze` disimpan; versi model + opset ONNX + versi TensorRT didokumentasikan.

---

# 25. Roadmap Implementasi (Capaian)

Setiap capaian harus menghasilkan sistem yang dapat dijalankan dan diuji sebelum lanjut. Implementasi bertahap — jangan sekaligus.

| # | Capaian | Acceptance Criteria |
|---|---|---|
| 0 | Scaffold | `python main.py --check` memuat + memvalidasi `config.yaml`, keluar bersih. `tests/` lulus. |
| 1 | Camera | `CameraSource` pluggable (`webcam`/`file`/`csi`); dua sumber berjalan bersamaan; FPS terukur & stabil; shutdown bersih; recover saat kamera dicabut. |
| 1b | Kalibrasi lensa | Per kamera: `camera_matrix` + `dist_coeffs` untuk barrel 160° (khusus CSI). Frame ter-undistort. |
| 1c | Site survey | Kamera terpasang fisik. Klip `cam_out` + `cam_in` untuk semua skenario (§7.4). Set garis pita + ROI recognition per kamera. Tinggi wajah di garis ≥ 112 px. |
| 2 | Detection | YOLO person/head + confidence tampil; filter kelas; FPS terukur; engine di-share antar kamera. |
| 3 | Tracking | `track_id` stabil; 2 orang sejajar = 2 ID; ID bertahan saat occlusion singkat (tailgating). Ukur ID switch. |
| 4 | Face Recognition | Enroll dari IMX219; terdaftar → `idpersonal`, lainnya → `UNKNOWN`; threshold + margin dikalibrasi pada probe kamera-pintu; laporkan EER pada `probe/test/`. |
| 5 | Counting | Pita 2 garis + state machine + filter arah + cooldown per track; satu crossing = satu event; uji semua skenario §12; dedup antar kamera. |
| 6 | Database | Event tersimpan dengan timestamp sistem + `camera_id` + `idpersonal`; worker enrichment PostgreSQL → `persons_cache`; validasi UUID + parameterized + read-only. |
| 7 | Daily Summary | Rekap per `idpersonal` (IN/OUT/status) + occupancy known/unknown; anomali akhir hari. |
| 8 | Optimization | ONNX → TensorRT FP16; engine di-share; 2 kamera dalam anggaran Nano (RAM ≤ 3 GB, ≥ 5 FPS/kamera, tanpa thermal throttling). |
| 9 | Cross-check | Occupancy `cam_out` vs `cam_in` konsisten; alarm saat selisih > ambang. |

---

# 26. Fase Produk

## Fase 1 — MVP: Monitoring & Keamanan

```
2× IMX219 → YOLO → ByteTrack → Face Recognition → Pita 2 Garis → SQLite → Console/UI sederhana
```

Output:

```
[07:31:20] cam_out  a4f9… (Juris)  → IN   sim=0.71
[08:20:11] cam_in   a4f9… (Juris)  → OUT  sim=0.63
[08:41:05] cam_out  UNKNOWN        → IN
```

Recognition bersifat **membantu**, ada manusia yang memverifikasi. Alarm `UNKNOWN` di luar jam kerja.

## Fase 2 — Presensi Kedinasan

Ditambahkan **hanya setelah** akurasi identitas pada `probe/test/` kamera-pintu memenuhi ambang (mis. TAR ≥ 0,98 pada FAR ≤ 1e-3 per orang) **dan** ada fallback (tap kartu/QR atau koreksi manual). Toleransi error presensi jauh lebih ketat (salah tandai hadir/absen = konsekuensi nyata). Jangan menjanjikan akurasi grade-presensi sebelum terukur.

Setelah MVP stabil: REST API → Dashboard → Reporting → Remote monitoring.

---

# 27. Acceptance Criteria Produk

Produk dianggap berhasil (Fase 1) apabila:

1. Kedua kamera berjalan kontinu.
2. Sistem mendeteksi orang/kepala dan mempertahankan tracking.
3. Sistem mengenali orang terdaftar (`idpersonal`) dan menandai lainnya `UNKNOWN` (open-set benar).
4. Sistem membedakan `IN` dan `OUT` dari pita dua garis + filter arah.
5. Satu crossing menghasilkan tepat satu event (tidak ada double count per track / antar kamera).
6. Event tersimpan dengan timestamp sistem, `camera_id`, dan `idpersonal`/`UNKNOWN`.
7. Rekap harian + occupancy (known & unknown) dapat dihitung; anomali akhir hari terdeteksi.
8. Worker enrichment mengisi `persons_cache` dari PostgreSQL saat online; sistem tetap jalan saat offline.
9. Sistem berjalan pada Jetson Nano 4GB dua kamera dalam anggaran sumber daya.
10. Tingkat kesalahan counting pada kondisi deployment berada pada level yang **telah diukur dan disepakati** per skenario (§12).
11. Papan pemberitahuan privasi terpasang; kebijakan retensi aktif.

---

# 28. Contoh Output Akhir

Aktivitas satu hari:

```
07:31:20  cam_out  Juris   IN
07:42:10  cam_out  Gusti   IN
08:01:31  cam_out  Rizki   IN
08:15:04  cam_out  UNKNOWN IN
09:15:20  cam_in   Juris   OUT
10:12:10  cam_in   Gusti   OUT
11:30:20  cam_out  Gusti   IN
12:10:32  cam_in   Gusti   OUT
```

Sistem menghasilkan:

```
================================
DAILY PEOPLE FLOW — 2026-09-08
================================

Juris   IN: 1   OUT: 1   INSIDE: NO
Gusti   IN: 2   OUT: 2   INSIDE: NO
Rizki   IN: 1   OUT: 0   INSIDE: YES
UNKNOWN IN: 1   OUT: 0

--------------------------------
Current Occupancy : 1 known + 1 unknown = 2
--------------------------------
```
