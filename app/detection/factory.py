"""Build the shared Detector from config."""
from __future__ import annotations

from .base import Detector
from .yolo_onnx import YoloOnnxDetector


def build_detector(node) -> Detector:
    # only onnx backend for now; TensorRT backend slots in here on Jetson
    return YoloOnnxDetector(
        model_path=node.model,
        input_size=node.input_size,
        conf_threshold=node.conf_threshold,
        nms_iou=node.nms_iou,
        classes=list(node.classes) if getattr(node, "classes", None) else ["person"],
    )
