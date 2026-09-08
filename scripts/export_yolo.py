#!/usr/bin/env python3
"""One-time: export a YOLOv8 .pt to ONNX for app/detection/yolo_onnx.py.

Needs `ultralytics` (pulls torch) — a DEV/EXPORT dependency only, never at
runtime. On Jetson the ONNX is converted further to a TensorRT engine.

  pip install ultralytics
  python scripts/export_yolo.py --weights yolov8n.pt --imgsz 480 --out models/yolov8n.onnx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8n.pt", help="ultralytics weights (auto-downloads)")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--opset", type=int, default=12, help="keep low for JetPack onnxruntime/TensorRT")
    ap.add_argument("--out", type=Path, default=Path("models/yolov8n.onnx"))
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics  (export-only dependency)")

    model = YOLO(a.weights)
    path = model.export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), a.out)
    print(f"exported -> {a.out}  (imgsz={a.imgsz}, opset={a.opset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
