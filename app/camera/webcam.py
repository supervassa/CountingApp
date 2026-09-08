"""Webcam / integer-index capture (laptop development)."""
from __future__ import annotations

import logging

import cv2

from .base import CameraSource, Frame

log = logging.getLogger(__name__)


class WebcamSource(CameraSource):
    def __init__(self, cam_id: str, index: int = 0, width: int = 1280, height: int = 720):
        super().__init__(cam_id)
        self.index, self.width, self.height = index, width, height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"[{self.cam_id}] cannot open webcam index {self.index}")
        log.info("[%s] webcam %d opened %dx%d", self.cam_id, self.index,
                 int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self) -> Frame | None:
        if not self._cap:
            return None
        ok, img = self._cap.read()
        if not ok or img is None:
            log.warning("[%s] frame read failed", self.cam_id)
            return None
        return self._wrap(img)

    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
