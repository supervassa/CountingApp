"""nms_xyxy — hand-computed cases (no OpenCV dependency, runs on the Jetson too).

Run: python tests/test_nms.py   (or pytest)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detection.yolo_onnx import nms_xyxy  # noqa: E402


def _b(*rows):
    return np.array(rows, dtype=np.float32)


def test_empty():
    assert nms_xyxy(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.7) == []


def test_overlap_keeps_highest_score():
    boxes = _b([0, 0, 100, 200], [5, 5, 105, 205])          # IoU ~0.9
    assert nms_xyxy(boxes, np.array([0.6, 0.9], np.float32), 0.7) == [1]


def test_disjoint_keeps_both_best_first():
    boxes = _b([0, 0, 50, 50], [300, 300, 350, 350])
    assert nms_xyxy(boxes, np.array([0.5, 0.8], np.float32), 0.7) == [1, 0]


def test_threshold_boundary():
    boxes = _b([0, 0, 100, 100], [50, 0, 150, 100])         # IoU = 1/3
    s = np.array([0.9, 0.8], np.float32)
    assert nms_xyxy(boxes, s, 0.5) == [0, 1]                 # below thr -> both kept
    assert nms_xyxy(boxes, s, 0.2) == [0]                    # above thr -> suppressed


if __name__ == "__main__":
    mod = sys.modules[__name__]
    for n in [x for x in dir(mod) if x.startswith("test_")]:
        getattr(mod, n)()
        print("ok", n)
