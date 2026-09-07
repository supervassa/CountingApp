\# Product Requirements Document (PRD)

  

\## Sistem Penghitungan Keluar-Masuk Orang Berbasis Kamera, YOLO, Face Recognition, dan Jetson Nano

  

\*\*Versi:\*\* 1.0

\*\*Platform:\*\* NVIDIA Jetson Nano Developer Kit 4GB

\*\*Target OS:\*\* JetPack / L4T R32.7.1

\*\*Python:\*\* 3.6.9

\*\*CUDA:\*\* 10.2

\*\*Mode:\*\* Edge AI / Offline-first

  

\---

  

\# 1. Ringkasan Produk

  

Sistem merupakan aplikasi computer vision berbasis edge AI yang menggunakan kamera untuk memantau aktivitas keluar-masuk orang melalui sebuah pintu.

  

Sistem tidak hanya menghitung jumlah orang, tetapi juga melakukan identifikasi terhadap orang yang melewati pintu sehingga dapat menghasilkan histori:

  

\- siapa yang masuk,

\- siapa yang keluar,

\- berapa kali seseorang masuk,

\- berapa kali seseorang keluar,

\- kapan event terjadi,

\- dan siapa saja yang sedang berada di dalam.

  

Contoh:

  

| Nama | Masuk | Keluar | Di Dalam |

|---|---:|---:|---:|

| Juris | 1 | 1 | Tidak |

| Gusti | 2 | 2 | Tidak |

| Rizki | 1 | 1 | Tidak |

  

Sistem dirancang untuk berjalan langsung pada NVIDIA Jetson Nano 4GB sehingga proses deteksi, tracking, dan face recognition dapat dilakukan di edge tanpa ketergantungan terhadap koneksi internet.

  

\---

  

\# 2. Tujuan Produk

  

\## 2.1 Tujuan Utama

  

Membangun sistem otomatis untuk menghitung dan mencatat aktivitas keluar-masuk individu melalui pintu menggunakan computer vision.

  

\## 2.2 Tujuan Teknis

  

Sistem harus mampu:

  

1\. menangkap video dari kamera;

2\. mendeteksi wajah/orang;

3\. melakukan tracking terhadap individu;

4\. mengenali identitas individu;

5\. mendeteksi crossing terhadap garis virtual;

6\. menentukan arah \`IN\` atau \`OUT\`;

7\. mencegah double counting;

8\. menyimpan event ke database;

9\. menghasilkan rekap harian;

10\. berjalan pada Jetson Nano 4GB.

  

\---

  

\# 3. Non-Goals

  

Versi pertama sistem tidak bertujuan untuk:

  

\- melakukan surveillance terhadap area yang luas;

\- mengenali seluruh populasi tanpa registrasi;

\- menggantikan sistem keamanan profesional;

\- melakukan identification dari wajah yang sangat kecil/tidak terlihat;

\- mengenali seseorang ketika wajah sepenuhnya tertutup;

\- menggunakan cloud inference sebagai komponen utama.

  

\---

  

\# 4. Target Pengguna

  

\### 4.1 Administrator

  

Bertanggung jawab terhadap:

  

\- registrasi pengguna;

\- pengambilan foto wajah;

\- konfigurasi kamera;

\- konfigurasi virtual line;

\- melihat histori;

\- melihat statistik.

  

\### 4.2 Operator

  

Menggunakan sistem untuk:

  

\- monitoring kondisi pintu;

\- melihat jumlah orang di dalam;

\- melihat event masuk/keluar.

  

\### 4.3 Individu Terdaftar

  

Orang yang wajahnya telah diregistrasikan ke sistem.

  

Contoh:

  

\`\`\`text

Juris

Gusti

Rizki

\`\`\`

  

\---

  

\# 5. Arsitektur Sistem

  

\`\`\`text

USB CAMERA

│

▼

┌─────────────┐

│ Frame Input │

└──────┬──────┘

│

▼

┌─────────────┐

│ YOLO │

│ Detection │

└──────┬──────┘

│

▼

┌─────────────┐

│ Tracker │

│ ByteTrack │

└──────┬──────┘

│

▼

┌─────────────┐

│ Face │

│ Recognition │

└──────┬──────┘

│

▼

┌─────────────┐

│ Similarity │

│ Matching │

└──────┬──────┘

│

▼

Juris/Gusti/Rizki

│

▼

┌─────────────┐

│Line Crossing│

└──────┬──────┘

│

┌─────┴─────┐

▼ ▼

IN OUT

│ │

└─────┬─────┘

▼

┌─────────────┐

│ SQLite │

└──────┬──────┘

│

▼

Dashboard/API

\`\`\`

  

\---

  

\# 6. Komponen Computer Vision

  

\## 6.1 Object / Face Detection

  

YOLO digunakan untuk mendeteksi objek target pada frame.

  

Output minimal:

  

\`\`\`text

class

confidence

x1

y1

x2

y2

\`\`\`

  

Contoh:

  

\`\`\`json

{

"class": "person",

"confidence": 0.94,

"bbox": \[320, 120, 520, 620\]

}

\`\`\`

  

Model harus dioptimalkan untuk perangkat Jetson Nano.

  

\---

  

\# 7. Tracking

  

Tracker digunakan untuk mempertahankan identitas sementara suatu objek antar-frame.

  

Contoh:

  

\`\`\`text

Frame 1 → Track ID 17

Frame 2 → Track ID 17

Frame 3 → Track ID 17

Frame 4 → Track ID 17

\`\`\`

  

Tujuan utama tracking:

  

\- mencegah double counting;

\- mengetahui trajectory;

\- menentukan crossing;

\- menghubungkan hasil face recognition dengan objek.

  

Tracker yang direkomendasikan:

  

\*\*ByteTrack\*\*

  

Alternatif:

  

\- SORT;

\- DeepSORT.

  

\---

  

\# 8. Face Recognition

  

YOLO dan tracker tidak bertanggung jawab terhadap identitas manusia.

  

Sistem membutuhkan model face recognition.

  

Pipeline:

  

\`\`\`text

Face

↓

Face Alignment

↓

Face Embedding Model

↓

Embedding Vector

↓

Similarity Matching

↓

Identity

\`\`\`

  

Contoh:

  

\`\`\`text

Input Face

↓

Embedding

↓

\[0.12, -0.31, 0.84, ...\]

↓

Compare Gallery

↓

Juris = 0.87

Gusti = 0.32

Rizki = 0.28

↓

Juris

\`\`\`

  

Model yang dapat digunakan pada tahap implementasi:

  

\- ArcFace / InsightFace;

\- model face embedding lain yang kompatibel dengan ONNX/TensorRT.

  

Model harus dipilih berdasarkan kompatibilitas dengan:

  

\- CUDA 10.2;

\- JetPack R32.7.1;

\- Python 3.6;

\- ARM64;

\- Jetson Nano 4GB.

  

\---

  

\# 9. Face Gallery / Enrollment

  

Setiap individu harus diregistrasikan terlebih dahulu.

  

Contoh:

  

\`\`\`text

gallery/

├── juris/

│ ├── 001.jpg

│ ├── 002.jpg

│ ├── 003.jpg

│ └── ...

│

├── gusti/

│ ├── 001.jpg

│ ├── 002.jpg

│ └── ...

│

└── rizki/

├── 001.jpg

├── 002.jpg

└── ...

\`\`\`

  

Sistem kemudian menghasilkan embedding.

  

\`\`\`text

Juris

↓

20 sample

↓

Face Embedding

↓

Representative Embedding

\`\`\`

  

\---

  

\# 10. Enrollment Protocol

  

Enrollment harus memperhatikan kondisi nyata kamera.

  

Target awal:

  

\*\*15–20 foto per orang.\*\*

  

Variasi yang direkomendasikan:

  

\### Pose

  

\- frontal;

\- kiri ±30°;

\- kanan ±30°;

\- sedikit menunduk;

\- sedikit mendongak.

  

\### Illumination

  

\- normal;

\- terang;

\- redup;

\- shadow;

\- backlight.

  

\### Appearance

  

\- tanpa kacamata;

\- menggunakan kacamata;

\- masker jika memang digunakan di lingkungan deployment.

  

\### Distance

  

\- dekat;

\- jarak operasional;

\- sedikit lebih jauh.

  

Foto tidak boleh seluruhnya diambil secara berurutan dengan pose identik.

  

Tujuan enrollment adalah memperoleh \*\*keragaman representasi identitas\*\*, bukan sekadar jumlah gambar.

  

\---

  

\# 11. Gallery dan Probe

  

Data harus dipisahkan.

  

\`\`\`text

dataset/

│

├── gallery/

│ ├── juris/

│ ├── gusti/

│ └── rizki/

│

└── probe/

├── clean/

├── pose/

├── illumination/

├── occlusion/

└── combined/

\`\`\`

  

Gallery digunakan untuk registrasi.

  

Probe digunakan untuk pengujian.

  

Hal ini penting agar robustness sistem tidak dinilai menggunakan data yang terlalu mirip dengan data enrollment.

  

\---

  

\# 12. Line Crossing

  

Area pintu dikonfigurasi menggunakan virtual line.

  

\`\`\`text

AREA DALAM

  

│

│

─────────┼─────────

│

│

  

AREA LUAR

\`\`\`

  

Trajectory centroid/anchor point digunakan untuk menentukan crossing.

  

\### IN

  

\`\`\`text

OUTSIDE

↓

LINE

↓

INSIDE

\`\`\`

  

Event:

  

\`\`\`text

IN

\`\`\`

  

\### OUT

  

\`\`\`text

INSIDE

↓

LINE

↓

OUTSIDE

\`\`\`

  

Event:

  

\`\`\`text

OUT

\`\`\`

  

\---

  

\# 13. Event State Machine

  

Untuk menghindari double counting, setiap track memiliki state.

  

Contoh:

  

\`\`\`text

UNKNOWN

│

▼

OUTSIDE

│

│ crossing

▼

INSIDE

│

│ crossing

▼

OUTSIDE

\`\`\`

  

Event hanya dibuat ketika terjadi transisi state yang valid.

  

Contoh:

  

\`\`\`text

OUTSIDE → INSIDE

\`\`\`

  

menghasilkan:

  

\`\`\`text

Juris IN

\`\`\`

  

Sedangkan:

  

\`\`\`text

INSIDE → OUTSIDE

\`\`\`

  

menghasilkan:

  

\`\`\`text

Juris OUT

\`\`\`

  

\---

  

\# 14. Anti Double Counting

  

Sistem harus memiliki beberapa mekanisme:

  

\### Tracking

  

Orang yang sama mempertahankan \`track\_id\`.

  

\### Crossing confirmation

  

Crossing harus dikonfirmasi berdasarkan beberapa frame, bukan satu frame.

  

\### Cooldown

  

Setelah event:

  

\`\`\`text

Juris → IN

\`\`\`

  

sistem tidak langsung menerima event IN berikutnya dari track yang sama.

  

\### Identity confidence

  

Identity dengan confidence/similarity rendah tidak boleh langsung dicatat sebagai identitas valid.

  

\---

  

\# 15. Unknown Person

  

Jika wajah tidak cocok dengan gallery:

  

\`\`\`text

Similarity < threshold

\`\`\`

  

maka:

  

\`\`\`text

identity = UNKNOWN

\`\`\`

  

Contoh:

  

\`\`\`text

UNKNOWN → IN

UNKNOWN → OUT

\`\`\`

  

Event tetap dapat disimpan sebagai statistik anonim apabila dibutuhkan.

  

Sistem tidak boleh memaksa seseorang menjadi Juris hanya karena similarity tertinggi jika nilai similarity sebenarnya berada di bawah threshold.

  

\---

  

\# 16. Database

  

SQLite digunakan pada edge device untuk versi awal.

  

\## \`persons\`

  

\`\`\`sql

CREATE TABLE persons (

id INTEGER PRIMARY KEY,

name TEXT NOT NULL,

embedding BLOB,

created\_at DATETIME DEFAULT CURRENT\_TIMESTAMP

);

\`\`\`

  

\## \`events\`

  

\`\`\`sql

CREATE TABLE events (

id INTEGER PRIMARY KEY,

person\_id INTEGER,

identity TEXT,

event\_type TEXT NOT NULL,

confidence REAL,

timestamp DATETIME NOT NULL,

date DATE NOT NULL,

track\_id INTEGER,

image\_path TEXT,

FOREIGN KEY(person\_id) REFERENCES persons(id)

);

\`\`\`

  

Contoh:

  

\`\`\`text

id | identity | event | timestamp

\---+----------+-------+-------------------

1 | Juris | IN | 07:30:21

2 | Gusti | IN | 07:35:12

3 | Rizki | IN | 07:42:18

4 | Juris | OUT | 08:20:11

\`\`\`

  

\---

  

\# 17. Rekap Harian

  

Sistem harus menyediakan:

  

\`\`\`text

Tanggal: 7 September 2026

  

Juris

IN : 1

OUT : 1

Current: OUT

  

Gusti

IN : 2

OUT : 2

Current: OUT

  

Rizki

IN : 1

OUT : 1

Current: OUT

\`\`\`

  

Status \`Current\` dapat ditentukan dari event terakhir.

  

\---

  

\# 18. Dashboard

  

Dashboard minimal menampilkan:

  

\`\`\`text

┌────────────────────────────────────┐

│ PEOPLE FLOW MONITORING │

├────────────────────────────────────┤

│ │

│ Orang di dalam 3 │

│ Total IN 17 │

│ Total OUT 14 │

│ │

├────────────────────────────────────┤

│ Nama IN OUT Status │

│ Juris 1 1 OUT │

│ Gusti 2 2 OUT │

│ Rizki 1 1 OUT │

└────────────────────────────────────┘

\`\`\`

  

\---

  

\# 19. Live Monitoring

  

Operator dapat melihat:

  

\`\`\`text

┌─────────────────────────────┐

│ │

│ CAMERA STREAM │

│ │

│ ┌───────────────┐ │

│ │ Juris │ │

│ │ IN │ │

│ └───────────────┘ │

│ │

│──────── VIRTUAL LINE ───────│

│ │

└─────────────────────────────┘

\`\`\`

  

Bounding box dapat menampilkan:

  

\`\`\`text

Juris

ID: 17

0.87

IN

\`\`\`

  

\---

  

\# 20. Dataset dan Data Collection

  

Data collection harus dilakukan dalam kondisi deployment sebenarnya.

  

Setiap individu sebaiknya memiliki variasi:

  

\### Pose

  

\- frontal;

\- yaw kiri;

\- yaw kanan;

\- pitch atas/bawah.

  

\### Illumination

  

\- normal;

\- low-light;

\- bright;

\- shadow;

\- backlight.

  

\### Occlusion

  

\- glasses;

\- mask;

\- partial occlusion;

\- objek yang dibawa.

  

\### Distance

  

\- dekat;

\- normal;

\- jauh.

  

\### Direction

  

\- masuk;

\- keluar.

  

\---

  

\# 21. Prinsip Dataset

  

Augmentasi tidak menggantikan data nyata.

  

Augmentasi digunakan untuk:

  

\- membantu robustness;

\- regularisasi;

\- memperbanyak variasi training jika model memang dilatih/fine-tuned.

  

Sedangkan data nyata digunakan untuk:

  

\- enrollment;

\- validasi;

\- pengujian deployment;

\- mengukur robustness sebenarnya.

  

\---

  

\# 22. Evaluation Metrics

  

Sistem harus dievaluasi pada dua level.

  

\## 22.1 Face Recognition

  

Minimal:

  

\- Accuracy;

\- FAR;

\- FRR;

\- ROC;

\- AUC;

\- EER;

\- TAR pada FAR tertentu.

  

\## 22.2 People Flow

  

Minimal:

  

\- IN counting accuracy;

\- OUT counting accuracy;

\- identity accuracy;

\- missed detection;

\- false crossing;

\- duplicate event;

\- ID switch.

  

Contoh:

  

\`\`\`text

Ground Truth:

Juris IN 1

Juris OUT 1

  

System:

Juris IN 1

Juris OUT 1

  

Result:

Correct

\`\`\`

  

\---

  

\# 23. Performance Requirement

  

Target awal pada Jetson Nano 4GB:

  

| Parameter | Target |

|---|---:|

| Resolution | 640×480 / 1280×720 |

| Detection | ≥ 8 FPS |

| End-to-end | ≥ 5 FPS |

| Recognition latency | ≤ 300 ms |

| RAM | ≤ 3 GB |

| GPU | TensorRT FP16 |

| Storage | ≥ 16 GB |

| Camera | USB / CSI |

  

Angka tersebut merupakan \*\*target engineering awal\*\*, bukan jaminan performa. Nilai final harus diukur pada model, kamera, dan konfigurasi Jetson yang digunakan.

  

\---

  

\# 24. Optimasi Jetson Nano

  

Tahap awal:

  

\`\`\`text

Python

↓

OpenCV

↓

YOLO

↓

Face Recognition

\`\`\`

  

Setelah pipeline benar:

  

\`\`\`text

ONNX

↓

TensorRT

↓

FP16

\`\`\`

  

Optimasi dilakukan terhadap:

  

\- input resolution;

\- detection frequency;

\- recognition frequency;

\- batch size;

\- TensorRT engine;

\- memory allocation;

\- frame skipping.

  

Face recognition tidak perlu dijalankan pada setiap frame.

  

Contoh:

  

\`\`\`text

YOLO → setiap frame

Tracker → setiap frame

Recognition → setiap 5–10 frame

Line → berdasarkan tracker

\`\`\`

  

\---

  

\# 25. Software Stack

  

Baseline:

  

\`\`\`text

OS

└── JetPack / L4T R32.7.1

  

Runtime

├── Python 3.6.9

├── CUDA 10.2

├── OpenCV

└── TensorRT

  

Computer Vision

├── YOLO

├── ByteTrack

└── Face Recognition / ArcFace

  

Storage

└── SQLite

  

Optional Backend

├── FastAPI / Flask

└── REST API

  

Dashboard

└── Web application

\`\`\`

  

Pemilihan versi library harus disesuaikan dengan CUDA 10.2, JetPack R32.7.1, Python 3.6, dan ARM64.

  

\---

  

\# 26. Struktur Project

  

\`\`\`text

people-flow/

│

├── app.py

│

├── camera/

│ ├── camera.py

│ └── stream.py

│

├── detection/

│ ├── detector.py

│ └── yolo.py

│

├── tracking/

│ ├── tracker.py

│ └── bytetrack.py

│

├── recognition/

│ ├── detector.py

│ ├── embedder.py

│ ├── matcher.py

│ └── gallery.py

│

├── counting/

│ ├── line.py

│ ├── crossing.py

│ └── state.py

│

├── database/

│ ├── database.py

│ ├── models.py

│ └── repository.py

│

├── enrollment/

│ ├── capture.py

│ └── generate\_embedding.py

│

├── config/

│ └── config.yaml

│

├── models/

│ ├── yolo.engine

│ └── face.engine

│

├── data/

│ ├── gallery/

│ ├── events/

│ └── snapshots/

│

└── logs/

\`\`\`

  

\---

  

\# 27. Functional Requirements

  

| ID | Requirement | Priority |

|---|---|---|

| FR-001 | Sistem dapat menerima input kamera | Must |

| FR-002 | Sistem dapat mendeteksi wajah/orang | Must |

| FR-003 | Sistem dapat melakukan tracking | Must |

| FR-004 | Sistem dapat mengenali individu | Must |

| FR-005 | Sistem dapat menentukan IN/OUT | Must |

| FR-006 | Sistem mencegah double counting | Must |

| FR-007 | Sistem menyimpan event | Must |

| FR-008 | Sistem menghasilkan rekap harian | Must |

| FR-009 | Sistem mendukung UNKNOWN | Must |

| FR-010 | Sistem menyediakan enrollment | Must |

| FR-011 | Sistem menyediakan live monitoring | Should |

| FR-012 | Sistem menyimpan snapshot event | Should |

| FR-013 | Sistem menyediakan REST API | Could |

| FR-014 | Sistem menyediakan dashboard web | Could |

  

\---

  

\# 28. Non-Functional Requirements

  

\## Performance

  

Sistem harus mampu berjalan secara real-time atau near-real-time pada Jetson Nano 4GB.

  

\## Reliability

  

Kehilangan satu atau beberapa frame tidak boleh menghasilkan event IN/OUT palsu.

  

\## Privacy

  

Data wajah dan embedding harus disimpan secara aman.

  

\## Offline

  

Core computer vision harus dapat berjalan tanpa internet.

  

\## Maintainability

  

Setiap komponen harus dipisahkan:

  

\`\`\`text

Detection

Tracking

Recognition

Counting

Database

\`\`\`

  

sehingga model dapat diganti tanpa mengubah keseluruhan sistem.

  

\---

  

\# 29. Roadmap Implementasi

  

\## Phase 1 — Camera

  

\`\`\`text

Camera

↓

OpenCV

↓

Display

\`\`\`

  

Acceptance criteria:

  

\- kamera stabil;

\- FPS dapat diukur;

\- tidak terjadi frame drop berlebihan.

  

\---

  

\## Phase 2 — YOLO

  

\`\`\`text

Camera

↓

YOLO

↓

Bounding Box

\`\`\`

  

Acceptance:

  

\- orang dapat dideteksi;

\- confidence dapat ditampilkan;

\- FPS terukur.

  

\---

  

\## Phase 3 — Tracking

  

\`\`\`text

YOLO

↓

ByteTrack

↓

Track ID

\`\`\`

  

Acceptance:

  

\`\`\`text

Person A → ID 1

Person B → ID 2

\`\`\`

  

ID tidak berubah secara berlebihan.

  

\---

  

\## Phase 4 — Face Recognition

  

\`\`\`text

Face

↓

Embedding

↓

Gallery

↓

Identity

\`\`\`

  

Acceptance:

  

\`\`\`text

Juris → Juris

Gusti → Gusti

Rizki → Rizki

Unknown → Unknown

\`\`\`

  

\---

  

\## Phase 5 — Line Crossing

  

\`\`\`text

Tracking

+

Virtual Line

↓

IN / OUT

\`\`\`

  

Acceptance:

  

\`\`\`text

OUTSIDE → INSIDE = IN

INSIDE → OUTSIDE = OUT

\`\`\`

  

\---

  

\## Phase 6 — Database

  

\`\`\`text

IN/OUT

↓

SQLite

\`\`\`

  

Acceptance:

  

\`\`\`text

Juris IN

Juris OUT

Gusti IN

...

\`\`\`

  

tersimpan dengan timestamp.

  

\---

  

\## Phase 7 — Daily Summary

  

\`\`\`text

Database

↓

Aggregation

↓

Daily Report

\`\`\`

  

Acceptance:

  

\`\`\`text

Juris 1 IN 1 OUT

Gusti 2 IN 2 OUT

Rizki 1 IN 1 OUT

\`\`\`

  

\---

  

\## Phase 8 — Optimization

  

\`\`\`text

Baseline

↓

ONNX

↓

TensorRT

↓

FP16

↓

Jetson Optimization

\`\`\`

  

Acceptance:

  

\- FPS meningkat;

\- latency turun;

\- memory stabil;

\- tidak terjadi thermal throttling.

  

\---

  

\# 30. Acceptance Criteria Produk

  

Produk dianggap berhasil apabila:

  

1\. Kamera dapat berjalan secara kontinu.

2\. Sistem dapat mendeteksi orang/wajah.

3\. Sistem dapat mempertahankan tracking.

4\. Sistem dapat mengenali orang terdaftar.

5\. Orang yang tidak dikenal diberi label \`UNKNOWN\`.

6\. Sistem dapat membedakan IN dan OUT.

7\. Satu crossing hanya menghasilkan satu event.

8\. Event tersimpan dengan timestamp.

9\. Rekap harian dapat dihitung.

10\. Sistem dapat berjalan pada Jetson Nano 4GB.

11\. Sistem tetap dapat beroperasi tanpa koneksi internet.

12\. Pengujian menunjukkan tingkat kesalahan counting yang dapat diterima pada kondisi deployment.

  

\---

  

\# 31. Contoh Output Akhir

  

Misalnya aktivitas satu hari:

  

\`\`\`text

07:31:20 Juris IN

07:42:10 Gusti IN

08:01:31 Rizki IN

  

09:15:20 Juris OUT

10:12:10 Gusti OUT

  

11:30:20 Gusti IN

12:10:32 Gusti OUT

  

13:20:11 Rizki OUT

\`\`\`

  

Sistem menghasilkan:

  

\`\`\`text

\================================

DAILY PEOPLE FLOW

\================================

  

Juris

IN : 1

OUT : 1

INSIDE : NO

  

Gusti

IN : 2

OUT : 2

INSIDE : NO

  

Rizki

IN : 1

OUT : 1

INSIDE : NO

  

\--------------------------------

Current Occupancy : 0

\--------------------------------

\`\`\`

  

\---

  

\# 32. Prinsip Arsitektur Utama

  

Sistem harus mengikuti prinsip:

  

\`\`\`text

DETECTION ≠ TRACKING ≠ RECOGNITION ≠ COUNTING

\`\`\`

  

Masing-masing memiliki tanggung jawab berbeda.

  

\`\`\`text

YOLO

"Di mana orangnya?"

  

Tracker

"Apakah ini orang yang sama?"

  

Face Recognition

"Siapa orang ini?"

  

Line Crossing

"Dia bergerak masuk atau keluar?"

  

Database

"Apa histori orang ini hari ini?"

\`\`\`

  

Dengan pemisahan ini, sistem menjadi lebih mudah diuji, dioptimalkan, dan dikembangkan.

  

\---

  

\# 33. Versi MVP

  

MVP tidak perlu langsung memiliki dashboard web.

  

Target MVP:

  

\`\`\`text

USB Camera

↓

YOLO

↓

ByteTrack

↓

Face Recognition

↓

Line Crossing

↓

SQLite

↓

Console / Simple UI

\`\`\`

  

Output:

  

\`\`\`text

\[07:31:20\] Juris → IN

\[09:15:20\] Juris → OUT

\[07:42:10\] Gusti → IN

\[10:12:10\] Gusti → OUT

\[11:30:20\] Gusti → IN

\[12:10:32\] Gusti → OUT

\`\`\`

  

Setelah MVP stabil, baru ditambahkan:

  

\`\`\`text

REST API

↓

Dashboard

↓

Reporting

↓

Remote monitoring

\`\`\`