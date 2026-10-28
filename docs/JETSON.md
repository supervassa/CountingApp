# Uji di Jetson Nano

Panduan bring-up pertama: dari kode di laptop sampai `main.py` jalan dengan 2 kamera IMX219.

## Status jujur: apa yang sudah dan belum diuji

| Sudah | Belum |
|---|---|
| 35 test lulus di Python 3.9 dan 3.8 (laptop) | **Belum pernah jalan di Python 3.6 / JetPack asli** |
| Pipeline penuh klip → deteksi → tracking → counting → SQLite → rekap (3.8) | Kamera CSI IMX219 (cuma dites string GStreamer, belum hardware) |
| `vermin`: sintaks kode kompatibel 3.6 (sisa `dataclasses` → pakai backport) | `onnxruntime 1.10` + `cv2 4.1.1` bawaan JetPack |
| SQLite tanpa sintaks upsert (Ubuntu 18.04 bawa 3.22) | Kecepatan inferensi di Nano |

Jadi langkah pertama di Jetson **bukan** `main.py`, tapi `scripts/jetson_check.py` + test. Kalau ada yang merah,
kirim outputnya — itu justru informasi yang dicari.

## 1. Prasyarat di Jetson

- JetPack 4.6.x (L4T R32.7.x), `python3 --version` → 3.6.9
- Dua kamera IMX219 terpasang di port CSI0 dan CSI1 (board B01), Jetson **mati** waktu pasang kabel
- Power: barrel jack 5V/4A (jumper J48), kipas aktif
- `ls /dev/video*` menampilkan `/dev/video0` dan `/dev/video1`

## 2. Pindahin kode + model

`models/` dan `data/` di-gitignore, jadi model ONNX dikirim manual.

```bash
# dari laptop — kode (tanpa venv/data/raw), lewat rsync
rsync -av --exclude .venv --exclude raw --exclude data --exclude logs --exclude .git --exclude __pycache__ \
  /Users/upatikmacbook/Documents/pythonpro/CountingApp/.claude/worktrees/capaian-awal-aplikasi-e0b6e8/ \
  USER@JETSON_IP:~/CountingApp/

# dari laptop — model (opset 12, IR 7: aman buat onnxruntime lama)
ssh USER@JETSON_IP 'mkdir -p ~/CountingApp/models'
scp models/yolov8n.onnx USER@JETSON_IP:~/CountingApp/models/
```

Alternatif: `git clone` + `git checkout <branch>` kalau branch sudah di-push / PR sudah di-merge.

## 3. Setup environment (di Jetson)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv libopenblas-dev
cd ~/CountingApp
python3 -m venv --system-site-packages .venv      # --system-site-packages: biar cv2 JetPack kelihatan
. .venv/bin/activate
pip install --upgrade pip==21.3.1 setuptools wheel
pip install -r requirements-jetson.txt
```

Kalau `pip` ngecompile `scipy`/`numpy` lebih dari ~10 menit: batalkan, `sudo apt install python3-scipy python3-numpy`
dan hapus pin-nya dari `requirements-jetson.txt`.

**onnxruntime GPU (disarankan):** build CPU dari PyPI lambat di Nano. Ambil wheel `onnxruntime-gpu` untuk
JetPack 4.6 + Python 3.6 dari [Jetson Zoo](https://elinux.org/Jetson_Zoo) (bagian ONNX Runtime), lalu:

```bash
pip uninstall -y onnxruntime
pip install /path/ke/onnxruntime_gpu-*-cp36-*-linux_aarch64.whl
```

(Link persisnya gua nggak bisa verifikasi dari sini — cek halaman Jetson Zoo.)

## 4. Urutan uji

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks       # mode MAXN
python3 scripts/jetson_check.py                 # 1. environment + kamera + kecepatan
for t in tests/test_*.py; do python3 $t || break; done    # 2. semua test harus "ok"
python3 main.py -c config/config.yaml -c config/jetson.yaml --check    # 3. config valid
python3 main.py -c config/config.yaml -c config/jetson.yaml            # 4. jalan (headless)
python3 main.py -c config/config.yaml -c config/jetson.yaml --summary  # 5. rekap hari ini
```

**Langkah 1** menulis cuplikan tiap kamera ke `data/jetson_check/cam0.jpg` dan `cam1.jpg`. Tarik ke laptop
dan lihat framing-nya — itu bahan buat naro garis band:

```bash
scp USER@JETSON_IP:~/CountingApp/data/jetson_check/cam*.jpg .
```

**Langkah 4:** jalan lewat SSH tanpa monitor (`display: false`). Yang dicari di log:
`[cam_out] frame N … fps` (FPS per kamera) dan baris `EVENT IN/OUT` waktu lo jalan lewat band.
Untuk lihat kenapa crossing dibuang, naikin log: tambah `logging: {level: DEBUG}` di `config/jetson.yaml`.

## 5. Yang harus disetel dari hasil uji

| Setelan | Kapan | Cara |
|---|---|---|
| `counting.band.<cam>` (garis luar/dalam) | **wajib**, default cuma placeholder | pecahan tinggi frame dari cuplikan `cam*.jpg`. `cam_out`: garis luar di atas garis dalam. `cam_in`: kebalikannya (lihat komentar di `config.yaml`) |
| `counting.confirm_frames` / `min_track_len` | FPS per kamera < 5 | turunkan ke `2` / `3` — di 3 FPS orang cuma beberapa frame di band |
| `detection.input_size` | FPS rendah | 480 → 416 / 320 (turun akurasi) |
| `cameras.<cam>.width/height/fps` | kamera nggak mau buka | `jetson_check.py` nyoba 1280×720@30/60, 1920×1080@30, 1640×1232@30 dan bilang yang jalan |

Target engineering: ≥ 5 FPS per kamera. Kalau jauh di bawah itu pakai wheel GPU dulu; TensorRT FP16 itu Capaian 8.

## 6. Troubleshooting

| Gejala | Kemungkinan | Aksi |
|---|---|---|
| `cannot open CSI sensor-id=N` | argus bengong / kabel | `sudo systemctl restart nvargus-daemon`; cek `ls /dev/video*`; pasang ulang kabel (Jetson mati) |
| `jetson_check`: cv2 built WITHOUT GStreamer | ada `opencv-python` dari pip nutupin cv2 JetPack | `pip uninstall opencv-python opencv-python-headless`; venv harus `--system-site-packages` |
| Satu kamera buka, yang kedua gagal | argus belum lepas sensor | tunggu 2 detik, ulang; `nvargus-daemon` restart |
| `SyntaxError` / `ModuleNotFoundError: dataclasses` | pip install belum jalan / bukan venv | aktifkan `.venv`, `pip install -r requirements-jetson.txt` |
| IN kehitung, **OUT nggak pernah** | band `cam_in` kebalik, atau `cam_in` nggak lihat orang | tukar `outer`/`inner` di band `cam_in`; cek `cam1.jpg` |
| IN dobel / sering ngawur | band terlalu tipis, FPS rendah | lebarin jarak outer–inner; turunin `confirm_frames` |
| Proses mati sendiri (`Killed`) | RAM habis | tambah swap ≥ 2 GB, matikan GUI desktop: `sudo systemctl set-default multi-user.target` |
| `UNKNOWN` semua | **normal** — face recognition (Capaian 4) belum dibangun | — |

## 7. Setelah ini

Hasil uji (output `jetson_check.py`, FPS per kamera, hitungan IN/OUT lawan jumlah orang yang sebenarnya lewat)
jadi masukan buat Capaian 1b (kalibrasi lensa), 1c (set band + ROI), dan 8 (TensorRT).
