"""YOLOv8 detector on onnxruntime.

Dev backend. On Jetson, swap for a TensorRT backend implementing the same
Detector interface (Fase Optimasi). One instance is shared across both cameras
(§5.3 PRD) — do not construct per camera.

Expected ONNX: single output [1, 4+nc, N] (YOLOv8 export default), bbox in
xywh, input-pixel space, no separate objectness. Person = COCO class 0.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .base import Detection, Detector

log = logging.getLogger(__name__)

# minimal COCO name map — only what we might use
_COCO = {0: "person"}


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy class-agnostic NMS. boxes Nx4 (x1,y1,x2,y2) -> kept indices, best first.

    Plain numpy on purpose: cv2.dnn.NMSBoxes changed its accepted box types between
    OpenCV releases and JetPack 4.x ships an old one.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class YoloOnnxDetector(Detector):
    def __init__(self, model_path: str, input_size: int = 480, conf_threshold: float = 0.35,
                 nms_iou: float = 0.7, classes: Optional[List[str]] = None):
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            raise ImportError("onnxruntime required: pip install onnxruntime")

        p = Path(model_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"detection model not found: {p}. Run scripts/export_yolo.py first."
            )
        self.size = int(input_size)
        self.conf = float(conf_threshold)
        self.iou = float(nms_iou)
        want = set(classes or ["person"])
        self.keep_cls = {cid for cid, name in _COCO.items() if name in want}

        providers = ort.get_available_providers()
        pref = [pr for pr in ("CUDAExecutionProvider", "CPUExecutionProvider") if pr in providers]
        self.sess = ort.InferenceSession(str(p), providers=pref or None)
        self.inp = self.sess.get_inputs()[0].name
        log.info("YOLO onnx: %s | size=%d | providers=%s", p.name, self.size, pref or providers)

    # --- pre/post ---
    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def detect(self, image: np.ndarray) -> List[Detection]:
        canvas, r, pad_x, pad_y = self._letterbox(image)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # 1,3,S,S

        out = self.sess.run(None, {self.inp: blob})[0]  # 1, 4+nc, N
        pred = np.squeeze(out, 0).T                     # N, 4+nc
        if pred.shape[1] < 5:
            return []

        xywh, scores = pred[:, :4], pred[:, 4:]
        cls_id = scores.argmax(1)
        cls_score = scores.max(1)

        m = cls_score >= self.conf
        if self.keep_cls:
            m &= np.isin(cls_id, list(self.keep_cls))
        if not m.any():
            return []
        xywh, cls_score, cls_id = xywh[m], cls_score[m], cls_id[m]

        # xywh (input space) -> xyxy (original image space)
        cx, cy, bw, bh = xywh.T
        x1 = (cx - bw / 2 - pad_x) / r
        y1 = (cy - bh / 2 - pad_y) / r
        x2 = (cx + bw / 2 - pad_x) / r
        y2 = (cy + bh / 2 - pad_y) / r

        idx = nms_xyxy(np.stack([x1, y1, x2, y2], 1), cls_score, self.iou)

        H, W = image.shape[:2]
        dets: List[Detection] = []
        for i in idx:
            dets.append(Detection(
                x1=float(np.clip(x1[i], 0, W - 1)), y1=float(np.clip(y1[i], 0, H - 1)),
                x2=float(np.clip(x2[i], 0, W - 1)), y2=float(np.clip(y2[i], 0, H - 1)),
                score=float(cls_score[i]), cls=int(cls_id[i]),
                label=_COCO.get(int(cls_id[i]), str(int(cls_id[i]))),
            ))
        return dets
