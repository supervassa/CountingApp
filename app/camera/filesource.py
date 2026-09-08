"""Video-file source — replay recorded IMX219 clips on the laptop.

Paces itself to the file's FPS so downstream timing behaves like a live camera.
Loops by default (handy for dev); set loop=False for one-shot probe runs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from .base import CameraSource, Frame

log = logging.getLogger(__name__)


class FileSource(CameraSource):
    def __init__(self, cam_id: str, path: str, loop: bool = True):
        super().__init__(cam_id)
        self.path = Path(path)
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._spf = 1 / 30
        self._next_t = 0.0

    def open(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"[{self.cam_id}] clip not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"[{self.cam_id}] cannot open clip: {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._spf = 1 / fps
        self._next_t = time.time()
        log.info("[%s] file %s opened (%.1f fps)", self.cam_id, self.path.name, fps)

    def read(self) -> Frame | None:
        if not self._cap:
            return None
        now = time.time()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t += self._spf

        ok, img = self._cap.read()
        if not ok or img is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok or img is None:
                log.info("[%s] end of clip", self.cam_id)
                return None
        return self._wrap(img)

    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
