"""IMX219 (and other CSI) source for Jetson — via GStreamer nvarguscamerasrc.

Tegra-only. On a laptop this import path is never taken (factory picks webcam/file).
The pipeline hands OpenCV BGR frames through appsink.
"""
from __future__ import annotations

import logging

import cv2

from .base import CameraSource, Frame

log = logging.getLogger(__name__)


def gst_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
    # capture at sensor res/fps, let the ISP debayer, convert to BGR for OpenCV
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class CsiSource(CameraSource):
    def __init__(self, cam_id: str, sensor_id: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 30, flip_method: int = 0):
        super().__init__(cam_id)
        self.sensor_id, self.width, self.height = sensor_id, width, height
        self.fps, self.flip_method = fps, flip_method
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        pipe = gst_pipeline(self.sensor_id, self.width, self.height, self.fps, self.flip_method)
        log.info("[%s] gst: %s", self.cam_id, pipe)
        self._cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[{self.cam_id}] cannot open CSI sensor-id={self.sensor_id}. "
                "OpenCV built with GStreamer? camera seated? `ls /dev/video*`?"
            )
        log.info("[%s] CSI sensor-id=%d opened %dx%d@%d", self.cam_id,
                 self.sensor_id, self.width, self.height, self.fps)

    def read(self) -> Frame | None:
        if not self._cap:
            return None
        ok, img = self._cap.read()
        if not ok or img is None:
            log.warning("[%s] CSI frame read failed", self.cam_id)
            return None
        return self._wrap(img)

    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
