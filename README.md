# People Flow + Face Recognition

Count people IN/OUT of a doorway and attach identity, on an **NVIDIA Jetson Nano 4GB**
with **2× IMX219-120 CSI** cameras. Counting is the primary feature; recognition is an
enrichment layer that may fail (→ `UNKNOWN`) without breaking the count.

- Design & requirements: [PRD.md](PRD.md), [instruksi.md](instruksi.md)
- Prod: 2 CSI cameras, one each side of the door. Dev: laptop webcam / recorded clips.

## Two-camera layout

| Camera    | Mount                         | Owns              |
|-----------|-------------------------------|-------------------|
| `cam_out` | outside, faces the approach   | `IN` events + identity of people entering |
| `cam_in`  | inside, faces the approach    | `OUT` events + identity of people leaving |

One camera only ever sees faces for one direction, so two are needed for identity on
both check-in and check-out.

## Data stores

- **SQLite** (local, offline-first): events, gallery embeddings, occupancy state.
- **PostgreSQL** (external personnel DB): name + info via `SELECT * FROM person.get_info_person($1)`.
  Never queried in the capture loop — an async worker resolves `idpersonal` → `persons_cache`.

## Dev setup (laptop)

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install ultralytics                       # export-only (pulls torch)
./.venv/bin/python scripts/export_yolo.py --imgsz 480     # -> models/yolov8n.onnx
./.venv/bin/python main.py --check      # load + validate config/config.yaml
./.venv/bin/python main.py              # capture + detection (needs a webcam)
```

`models/` is gitignored — regenerate `yolov8n.onnx` with `scripts/export_yolo.py`.
Without it the capture loop still runs (detection logs "disabled").

`config/config.yaml` holds everything tunable — cameras, lines, thresholds. Nothing
operational is hard-coded.

## Capaian (milestones — each must run and be testable before the next)

| # | Milestone | Acceptance |
|---|-----------|------------|
| 0 | Scaffold | `main.py --check` loads config, exits clean ✅ |
| 1 | Camera | dual-source capture (webcam/file/csi), stable FPS, clean shutdown |
| 1b | Lens calibration | per-camera matrix + dist coeffs for the 120° barrel (CSI only) |
| 1c | Site survey | mount cameras, record IN/OUT clips, set band lines + recog ROI per camera |
| 2 | Detection | YOLOv8n via onnxruntime, shared across cameras; person boxes + confidence ✅ |
| 3 | Tracking | ByteTrack per camera; stable `track_id` through crossing + short occlusion; `track_buffer` recovers gaps ✅ |
| 5 | Counting | 2-line band + per-track state machine + direction filter (cam_out=IN, cam_in=OUT) + per-track cooldown + occupancy; one crossing = one event ✅ |
| 6 | Database | SQLite events (system time, `camera_id`, `idpersonal`) + occupancy_state (survives restart); async PostgreSQL enrichment worker → persons_cache (off by default) ✅ |
| 7 | Daily summary | `main.py --summary [DATE]` — per-person IN/OUT/status, occupancy known/unknown, end-of-day anomaly list (still marked inside) ✅ |
| 4 | Face recognition | enrolled → idpersonal, others → `UNKNOWN`; threshold calibrated on door-cam probe (after cameras mounted) |
| 9 | Cross-check | `cam_out` vs `cam_in` consistency alarms (negative / too-high occupancy, silent camera) → WARNING log + `alarms` table, rate-limited ✅ |
| 8 | Optimization | ONNX → TensorRT FP16, shared engines, 2-cam within Nano budget (on Jetson) |

## Status

Capaian 0, 1, 2, 3, 5, 6, 7, 9 are built and tested (27 tests, `main.py --check` OK).
The remaining ones need physical hardware:

- **1b lens calibration** — once the cameras are mounted: capture a checkerboard,
  fill `cameras.<cam>.camera_matrix` / `dist_coeffs`, set `undistort: true`.
- **1c site survey** — mount both cameras, record IN/OUT and crowd clips, then set
  `counting.band.<cam>` lines and `recognition.roi` per camera from those clips.
- **4 face recognition** — not built yet. Needs face detect + align (112px), an
  ArcFace ONNX embedder, a gallery (`data/gallery/embeddings.npz`), a cosine
  matcher (threshold + margin), and vote-and-lock of identity onto a `track_id`.
  Enrollment (`app/enrollment/`) captures 15–20 shots per person from the mounted
  IMX219, keyed to their `idpersonal`; the threshold is calibrated on a door-cam
  probe (FAR/FRR/EER). Turn on `personnel_db` with credentials + the column list
  from `person.get_info_person()` for names.
- **8 optimization** — on the Jetson: ONNX → TensorRT FP16, shared engines, verify
  the 2-camera pipeline fits the Nano budget.

## Enrollment

Fresh capture from the mounted IMX219 (domain-matched), 15–20 varied shots per person,
keyed to their real `idpersonal`. The legacy `raw/` dump is **shelved** (mostly 1 photo
per person, selfie domain) — kept only as a potential impostor pool for FAR testing.
`scripts/ingest_raw.py` audits it.

## Repo layout

```
main.py                 entry point (run | --check | --summary)
config/config.yaml      all runtime tuning (cameras, band lines, thresholds, personnel_db)
app/
  config.py             load + validate yaml
  logging_setup.py
  pipeline.py           orchestrator: capture -> detect -> track -> crossing -> occupancy -> db
  camera/               CameraSource: webcam / filesource / csi (GStreamer) + factory
  detection/            Detector + YoloOnnxDetector + factory
  tracking/             Tracker + ByteTracker + Kalman + factory
  counting/             band, counter (state machine), occupancy, crosscheck + factory
  database/             schema.sql, repository, enrichment worker + factory
  recognition/ enrollment/   (Capaian 4 — not built yet)
scripts/
  ingest_raw.py         legacy dataset audit
  export_yolo.py        yolov8n.pt -> ONNX (one-time)
tests/                  config, bytetrack, counting, crosscheck, database  (27 tests)
```
