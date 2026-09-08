"""Build a CameraSource from a camera config node."""
from __future__ import annotations

from .base import CameraSource
from .csi import CsiSource
from .filesource import FileSource
from .webcam import WebcamSource


def build_camera(cam_id: str, node) -> CameraSource:
    src = node.source
    if src == "webcam":
        return WebcamSource(cam_id, index=node.webcam_index,
                            width=node.width, height=node.height)
    if src == "file":
        return FileSource(cam_id, path=node.file_path)
    if src == "csi":
        return CsiSource(cam_id, sensor_id=node.csi_sensor_id, width=node.width,
                         height=node.height, fps=node.fps, flip_method=node.flip_method)
    raise ValueError(f"[{cam_id}] unknown source: {src!r}")
