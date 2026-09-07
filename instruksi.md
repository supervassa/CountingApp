# ****MASTER INSTRUCTION****

## ****AI People Flow & Face Recognition System****

Anda bertindak sebagai ****Senior AI/Computer Vision Engineer + Edge AI Engineer****. Bangun sistem penghitung orang masuk/keluar ruangan berbasis ****YOLO + ByteTrack + Face Recognition**** yang berjalan pada ****NVIDIA Jetson Nano 4GB****.

## ****1\. Tujuan Sistem****

Sistem harus mampu:

-   Mendeteksi orang melalui kamera.
-   Melacak orang antar-frame.
-   Mengenali identitas orang yang terdaftar.
-   Menentukan arah pergerakan melalui virtual line.
-   Mencatat event `IN` dan `OUT`.
-   Mencegah penghitungan ganda.
-   Menyimpan histori ke SQLite.
-   Menampilkan rekap harian per orang.

Contoh output:

Juris  : IN 1 | OUT 1  
Gusti  : IN 2 | OUT 2  
Rizki  : IN 1 | OUT 1

# ****2\. Arsitektur Utama****

Gunakan pipeline berikut dan jangan menggabungkan tanggung jawab antar-modul:

Camera  
   ↓  
Frame Capture  
   ↓  
YOLO Person Detection  
   ↓  
ByteTrack  
   ↓  
Face Detection / Face Alignment  
   ↓  
Face Embedding  
   ↓  
Gallery Matching  
   ↓  
Identity  
   ↓  
Virtual Line Crossing  
   ↓  
IN / OUT Event  
   ↓  
SQLite  
   ↓  
API / Dashboard

Prinsip wajib:

```
Detection ≠ Tracking ≠ Recognition ≠ Counting
```

Setiap komponen harus memiliki module/class yang terpisah.

# ****3\. Target Hardware****

Target utama:

Device       : NVIDIA Jetson Nano 4GB  
Architecture : ARM64 / aarch64  
JetPack      : R32.7.1  
CUDA         : 10.2  
Python       : 3.6.9

Jetson digunakan sebagai ****inference device****, bukan sebagai mesin training.

Optimalkan penggunaan:

-   RAM
-   VRAM
-   CPU
-   GPU
-   inference latency

Prioritaskan model yang realistis untuk Jetson Nano 4GB.

# ****4\. Model AI****

## ****Person Detection****

Gunakan YOLO untuk mendeteksi:

```
class = person
```

Jangan melakukan face recognition terhadap seluruh frame.

Face recognition hanya dilakukan pada area yang relevan untuk menghemat resource.

## ****Tracking****

Gunakan:

```
ByteTrack
```

Tracking harus menghasilkan:

track\_id  
bbox  
confidence  
trajectory

`track_id` tidak boleh dianggap sebagai identitas seseorang.

Contoh:

track\_id = 17  
identity = Juris

adalah dua informasi berbeda.

# ****5\. Face Recognition****

Gunakan pendekatan ****face embedding****, bukan classifier per orang.

Pipeline:

Face  
 ↓  
Face Alignment  
 ↓  
Face Embedding  
 ↓  
Cosine Similarity  
 ↓  
Gallery Matching  
 ↓  
Identity / UNKNOWN

Gunakan model pretrained yang sesuai, misalnya keluarga:

```
ArcFace / InsightFace
```

Namun sebelum memilih implementasi final, verifikasi kompatibilitas model/library dengan:

JetPack R32.7.1  
CUDA 10.2  
Python 3.6  
ARM64

Jangan mengasumsikan library modern otomatis kompatibel dengan Jetson Nano.

# ****6\. Enrollment****

Enrollment dilakukan melalui aplikasi mobile:

Flutter Android  
Flutter iOS

Flow:

Register Person  
      ↓  
Capture Face Samples  
      ↓  
Quality Check  
      ↓  
Upload  
      ↓  
Generate Embedding  
      ↓  
Store Face Gallery

Target awal:

```
15–20 sample/person
```

Sample harus memiliki variasi:

-   frontal
-   kiri
-   kanan
-   sedikit atas
-   sedikit bawah
-   ekspresi berbeda
-   pencahayaan berbeda
-   jarak berbeda
-   kondisi penggunaan sebenarnya

Jangan mengambil 20 foto yang hampir identik.

# ****7\. Gallery dan Probe****

Pisahkan:

Gallery  
Probe/Test

Gallery digunakan untuk enrollment.

Probe digunakan untuk menguji sistem.

Contoh:

gallery/  
    juris/  
    gusti/  
    rizki/  
  
probe/  
    clean/  
    pose/  
    illumination/  
    occlusion/  
    combined/

Jangan menggunakan seluruh data testing sebagai gallery karena dapat menyebabkan evaluasi tidak valid.

# ****8\. Training****

Jangan melakukan training model utama di Jetson Nano.

Gunakan:

PC GPU  
Kaggle GPU  
Cloud GPU

untuk:

-   training
-   fine-tuning
-   eksperimen
-   evaluasi
-   export ONNX

Kemudian:

ONNX  
 ↓  
TensorRT  
 ↓  
Jetson Nano

Jika model pretrained sudah cukup baik, ****jangan melakukan training tambahan tanpa alasan eksperimental yang jelas****.

Enrollment seseorang bukan training model.

Enrollment hanya menghasilkan:

```
face embedding
```

yang kemudian dimasukkan ke gallery.

# ****9\. Identity Matching****

Gunakan cosine similarity.

Konsep:

query\_embedding  
        ↓  
compare  
        ↓  
gallery embeddings  
        ↓  
highest similarity  
        ↓  
threshold  
        ↓  
identity / UNKNOWN

Jangan selalu memilih identity dengan similarity tertinggi.

Contoh:

Juris   0.91  
Gusti   0.62  
Rizki   0.55

Jika threshold = `0.70`:

```
Juris
```

Jika:

Juris   0.61  
Gusti   0.59  
Rizki   0.54

hasil:

```
UNKNOWN
```

Threshold harus dapat dikonfigurasi dan dievaluasi menggunakan dataset validation.

# ****10\. Line Crossing****

Gunakan virtual line:

\-------------------------  
        LINE  
\-------------------------

Tentukan dua area:

OUTSIDE  
INSIDE

Aturan:

OUTSIDE → INSIDE = IN  
  
INSIDE → OUTSIDE = OUT

Gunakan trajectory/centroid tracking, bukan hanya posisi satu frame.

# ****11\. Anti Double Counting****

Sistem tidak boleh menghitung seseorang berkali-kali hanya karena berada dekat line.

Implementasikan:

-   tracking
-   multi-frame confirmation
-   trajectory history
-   crossing state
-   event cooldown
-   identity confidence

Contoh state:

UNKNOWN  
   ↓  
OUTSIDE  
   ↓  
CROSSING  
   ↓  
INSIDE

atau:

INSIDE  
   ↓  
CROSSING  
   ↓  
OUTSIDE

Event hanya dibuat ketika transisi valid terjadi.

# ****12\. Event Database****

Gunakan SQLite pada tahap awal.

Minimal tabel:

persons  
events

### ****persons****

id  
name  
external\_id  
embedding  
created\_at  
updated\_at

### ****events****

id  
person\_id  
identity  
event\_type  
confidence  
timestamp  
date  
track\_id  
image\_path

`event_type`:

IN  
OUT

Timestamp event harus berasal dari sistem dan tidak boleh menggunakan waktu yang dimanipulasi oleh client.

# ****13\. Daily Summary****

Sistem harus dapat menghasilkan:

```
Person | IN | OUT | Current Status
```

Contoh:

Juris | 1 | 1 | OUTSIDE  
Gusti | 2 | 2 | OUTSIDE  
Rizki | 1 | 0 | INSIDE

Current status dapat ditentukan berdasarkan event terakhir.

# ****14\. Unknown Person****

Jika wajah tidak dapat dikenali:

```
identity = UNKNOWN
```

Jangan memaksakan identitas terdekat.

Sistem harus membedakan:

Known Person  
Unknown Person  
No Face  
Low Confidence

# ****15\. Performance****

Target awal engineering:

Input          : 640x480 atau 1280x720  
Detection FPS  : ≥ 8 FPS  
Pipeline FPS   : ≥ 5 FPS  
Recognition    : ≤ 300 ms  
RAM            : ≤ 3 GB  
Precision      : FP16 jika memungkinkan

Angka tersebut adalah ****target engineering****, bukan jaminan performa.

Jika performa rendah:

1.  turunkan resolusi
2.  optimalkan YOLO
3.  gunakan TensorRT
4.  gunakan FP16
5.  kurangi frekuensi face recognition
6.  gunakan tracking antar-frame
7.  batasi ROI
8.  kurangi ukuran input model

Jangan langsung mengganti seluruh arsitektur tanpa profiling.

# ****16\. Struktur Project****

Gunakan struktur modular:

people-flow/  
│  
├── app/  
│   ├── camera/  
│   ├── detection/  
│   ├── tracking/  
│   ├── recognition/  
│   ├── counting/  
│   ├── database/  
│   ├── enrollment/  
│   └── api/  
│  
├── models/  
│   ├── yolo/  
│   ├── face/  
│   └── tensorrt/  
│  
├── config/  
│   └── config.yaml  
│  
├── data/  
│   ├── gallery/  
│   └── snapshots/  
│  
├── tests/  
│  
├── scripts/  
│  
├── main.py  
├── requirements.txt  
└── README.md

Jangan membuat satu file Python besar yang menangani seluruh pipeline.

# ****17\. Development Strategy****

Implementasikan secara bertahap.

### ****Phase 1****

Camera:

```
Camera → Frame
```

Pastikan FPS stabil.

### ****Phase 2****

Detection:

```
Camera → YOLO → Bounding Box
```

### ****Phase 3****

Tracking:

```
YOLO → ByteTrack → track_id
```

### ****Phase 4****

Recognition:

```
Face → Embedding → Identity
```

### ****Phase 5****

Counting:

```
Tracking → Line Crossing → IN/OUT
```

### ****Phase 6****

Database:

```
IN/OUT → SQLite
```

### ****Phase 7****

Dashboard/API:

```
SQLite → API → Dashboard
```

### ****Phase 8****

Optimization:

```
Profiling → TensorRT → FP16 → Optimization
```

Jangan mengimplementasikan seluruh sistem sekaligus.

# ****18\. Testing****

Setiap module harus dapat diuji secara independen.

Minimal testing:

### ****Detection****

person detection accuracy  
false detection  
missed detection

### ****Tracking****

ID switch  
lost track  
multiple people  
occlusion

### ****Recognition****

known person  
unknown person  
pose  
illumination  
occlusion  
distance

### ****Counting****

IN  
OUT  
reversal  
multiple people  
crossing simultaneously

### ****System****

duplicate event  
camera disconnect  
database failure  
low FPS  
Jetson memory pressure

# ****19\. Evaluation****

Face recognition:

Accuracy  
Precision  
Recall  
FAR  
FRR  
ROC  
AUC  
EER  
TAR@FAR

People counting:

IN Accuracy  
OUT Accuracy  
Missed Count  
False Count  
Duplicate Count  
ID Switch

Pisahkan evaluasi ****face recognition**** dan ****people-flow counting****.

# ****20\. Coding Rules****

Saat menulis kode:

1.  Jangan membuat asumsi tentang library yang belum diverifikasi.
2.  Prioritaskan kompatibilitas Jetson Nano.
3.  Gunakan dependency seminimal mungkin.
4.  Pisahkan konfigurasi dari source code.
5.  Jangan hard-code threshold.
6.  Jangan hard-code koordinat line.
7.  Jangan menyimpan API key/password di source code.
8.  Tambahkan logging.
9.  Tambahkan error handling.
10.  Buat kode yang dapat di-debug.
11.  Jangan melakukan premature optimization.
12.  Profiling dilakukan sebelum optimasi.
13.  Setiap perubahan arsitektur harus memiliki alasan teknis.
14.  Jangan menghapus fitur existing tanpa alasan dan konfirmasi.
15.  Jangan mengubah environment Jetson secara destruktif.

# ****21\. Aturan Khusus AI Coding Agent****

Sebelum membuat kode:

1.  Periksa struktur project yang sudah ada.
2.  Periksa versi Python.
3.  Periksa CUDA.
4.  Periksa TensorRT.
5.  Periksa OpenCV.
6.  Periksa dependency yang sudah terinstall.
7.  Identifikasi library yang kompatibel dengan Jetson Nano.
8.  Jelaskan dependency baru sebelum memasangnya.

Jika terdapat beberapa alternatif library/model, pilih berdasarkan:

Compatibility  
→ Performance  
→ Memory usage  
→ Accuracy  
→ Maintenance

Jangan memilih library hanya karena paling populer atau paling baru.

Sebelum melakukan perubahan besar, jelaskan:

Problem  
Cause  
Proposed Solution  
Impact

Setiap tahap harus menghasilkan sistem yang dapat dijalankan dan diuji.

# ****22\. Prinsip Utama****

Sistem ini bukan sekadar:

```
YOLO = people counter
```

Tetapi:

YOLO  
  ↓  
WHO IS THERE?  
  ↓  
TRACK  
  ↓  
WHO IS THE PERSON?  
  ↓  
WHERE ARE THEY GOING?  
  ↓  
DID THEY CROSS THE LINE?  
  ↓  
IN / OUT  
  ↓  
STORE EVENT

Target akhir:

Camera  
   ↓  
Detect  
   ↓  
Track  
   ↓  
Recognize  
   ↓  
Determine Direction  
   ↓  
Generate Event  
   ↓  
Store  
   ↓  
Report

Semua implementasi harus mempertahankan arsitektur tersebut.