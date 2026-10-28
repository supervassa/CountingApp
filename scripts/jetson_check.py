#!/usr/bin/env python3
"""First thing to run on the Jetson: is the environment ready for this app?

Checks Python / L4T / libraries / the two IMX219 CSI cameras / YOLO speed and prints
PASS / WARN / FAIL per item. Written for Python 3.6 (stdlib only for the checks
themselves; cv2 / onnxruntime / numpy are imported lazily and reported if missing).

  python3 scripts/jetson_check.py
  python3 scripts/jetson_check.py --no-camera      # skip the CSI probe
  python3 scripts/jetson_check.py --no-model       # skip the inference benchmark

Snapshots from each camera are written to data/jetson_check/ so you can see what
each camera actually looks at (use them to place the counting band).
"""
import argparse
import os
import platform
import sqlite3
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (width, height, fps) modes the IMX219 exposes through nvarguscamerasrc, best-first for us
CSI_MODES = [(1280, 720, 30), (1280, 720, 60), (1920, 1080, 30), (1640, 1232, 30)]

_results = []


def report(level, item, detail=""):
    _results.append(level)
    print("  [%-4s] %-28s %s" % (level, item, detail))


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                       universal_newlines=True).strip()
    except Exception:
        return ""


def check_platform():
    print("\n== platform ==")
    py = sys.version_info
    report("PASS" if py[:2] == (3, 6) else "WARN", "python", "%s (code targets 3.6)" % platform.python_version())
    arch = platform.machine()
    report("PASS" if arch == "aarch64" else "WARN", "arch", arch + ("" if arch == "aarch64" else " — not a Jetson?"))
    tegra = sh("head -n1 /etc/nv_tegra_release")
    report("PASS" if tegra else "WARN", "L4T / JetPack", tegra or "/etc/nv_tegra_release not found")
    mode = sh("nvpmodel -q 2>/dev/null | tail -n1")
    report("PASS" if mode in ("0", "MAXN") else "WARN", "power mode",
           (mode or "unknown") + "   (MAXN = `sudo nvpmodel -m 0 && sudo jetson_clocks`)")
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            mem[k] = int(v.split()[0]) // 1024
    except IOError:
        pass                                          # not Linux (laptop dry-run)
    if mem:
        report("PASS" if mem.get("SwapTotal", 0) >= 2000 else "WARN", "RAM / swap",
               "%d MB RAM, %d MB swap  (add >= 2 GB swap for 2 cameras + model)"
               % (mem.get("MemTotal", 0), mem.get("SwapTotal", 0)))
    else:
        report("WARN", "RAM / swap", "/proc/meminfo unavailable")
    report("PASS", "sqlite", sqlite3.sqlite_version + "  (app avoids upsert syntax, needs nothing newer)")


def check_libs():
    print("\n== libraries ==")
    for mod, need in (("numpy", True), ("yaml", True), ("scipy", True), ("dataclasses", True)):
        try:
            m = __import__(mod)
            report("PASS", mod, getattr(m, "__version__", "ok"))
        except ImportError:
            hint = "pip3 install dataclasses" if mod == "dataclasses" else "pip3 install " + ("PyYAML" if mod == "yaml" else mod)
            report("FAIL", mod, "missing — " + hint)
    try:
        import cv2
        info = cv2.getBuildInformation()
        gst = [l.strip() for l in info.splitlines() if "GStreamer:" in l]
        ok = bool(gst) and "YES" in gst[0]
        report("PASS" if ok else "FAIL", "cv2 " + cv2.__version__,
               "GStreamer YES" if ok else "built WITHOUT GStreamer — CSI cannot open. Use JetPack's cv2; do not pip install opencv-python")
    except ImportError:
        report("FAIL", "cv2", "missing — `sudo apt install python3-opencv`, venv needs --system-site-packages")
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
        gpu = "CUDAExecutionProvider" in prov or "TensorrtExecutionProvider" in prov
        report("PASS" if gpu else "WARN", "onnxruntime " + ort.__version__,
               ",".join(prov) + ("" if gpu else "   CPU only — slow; install the Jetson Zoo onnxruntime-gpu wheel"))
    except ImportError:
        report("FAIL", "onnxruntime", "missing — see docs/JETSON.md")


def probe_csi(sensor_id):
    import cv2
    from app.camera.csi import gst_pipeline
    for (w, h, fps) in CSI_MODES:
        cap = cv2.VideoCapture(gst_pipeline(sensor_id, w, h, fps, 0), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            time.sleep(1.0)
            continue
        t0, n, last = time.time(), 0, None
        while n < 30 and time.time() - t0 < 8:
            ok, img = cap.read()
            if ok and img is not None:
                n, last = n + 1, img
        dt = time.time() - t0
        cap.release()
        time.sleep(1.0)                              # let argus free the sensor before the next open
        if last is None:
            continue
        snap_dir = os.path.join(ROOT, "data", "jetson_check")
        os.makedirs(snap_dir, exist_ok=True)
        snap = os.path.join(snap_dir, "cam%d.jpg" % sensor_id)
        cv2.imwrite(snap, last)
        return (w, h, fps), n / dt if dt > 0 else 0.0, float(last.mean()), snap, last
    return None


def check_cameras():
    print("\n== CSI cameras (IMX219) ==")
    devs = sh("ls /dev/video* 2>/dev/null")
    report("PASS" if devs else "FAIL", "/dev/video*", devs.replace("\n", " ") or "none — cameras not detected (reseat the ribbon cables, contacts toward the heatsink side per the board silk, power off first)")
    frames = {}
    try:
        import cv2  # noqa: F401
    except ImportError:
        return frames
    for sid in (0, 1):
        got = probe_csi(sid)
        if got is None:
            report("FAIL", "sensor-id=%d" % sid, "no mode opened — check `sudo systemctl restart nvargus-daemon`, cable seating")
            continue
        mode, fps, bright, snap, img = got
        frames[sid] = img
        dark = bright < 20
        report("WARN" if dark else "PASS", "sensor-id=%d" % sid,
               "%dx%d@%d  measured %.1f fps  brightness %.0f%s  -> %s"
               % (mode[0], mode[1], mode[2], fps, bright, "  (very dark: lens cap?)" if dark else "",
                  os.path.relpath(snap, ROOT)))
    return frames


def check_model(frames, model_rel="models/yolov8n.onnx"):
    print("\n== YOLO inference speed ==")
    model = os.path.join(ROOT, model_rel)
    if not os.path.isfile(model):
        report("FAIL", model_rel, "not found — scp it from the laptop (python scripts/export_yolo.py there)")
        return
    try:
        import numpy as np
        from app.detection.yolo_onnx import YoloOnnxDetector
        det = YoloOnnxDetector(model, input_size=480, conf_threshold=0.35, nms_iou=0.7, classes=["person"])
    except Exception as e:
        report("FAIL", "load model", "%s: %s" % (type(e).__name__, e))
        return
    img = frames[0] if 0 in frames else (frames[1] if 1 in frames else None)
    if img is None:
        img = np.full((720, 1280, 3), 114, np.uint8)
    for _ in range(3):
        det.detect(img)
    n, t0 = 20, time.time()
    for _ in range(n):
        det.detect(img)
    ms = (time.time() - t0) * 1000.0 / n
    per_cam = 1000.0 / (2 * ms)                      # one shared engine, two cameras take turns
    level = "PASS" if per_cam >= 5 else ("WARN" if per_cam >= 2 else "FAIL")
    report(level, "detect (480px)", "%.0f ms/frame -> ~%.1f FPS per camera with 2 cameras (target >= 5)" % (ms, per_cam))
    if per_cam < 5:
        print("         slow: use the onnxruntime-gpu wheel (CUDA EP) or a TensorRT engine; "
              "meanwhile lower detection.input_size and set counting.confirm_frames/min_track_len to 2/3.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()

    print("Jetson readiness check — %s" % ROOT)
    check_platform()
    check_libs()
    frames = {} if a.no_camera else check_cameras()
    if not a.no_model:
        check_model(frames)
    print("\nsummary: %d PASS, %d WARN, %d FAIL" % tuple(_results.count(k) for k in ("PASS", "WARN", "FAIL")))
    return 1 if "FAIL" in _results else 0


if __name__ == "__main__":
    sys.exit(main())
