"""CameraSource interface — the rest of the pipeline never knows if a frame
came from a webcam, a video file, or an IMX219 on the CSI bus.

Contract: read() returns a Frame or None (transient failure -> caller may retry;
persistent failure -> is_open() goes False).
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    image: np.ndarray            # BGR HxWx3 uint8
    ts: float = field(default_factory=time.time)   # capture wall-clock
    index: int = 0               # monotonically increasing per source
    cam_id: str = ""             # "cam_out" | "cam_in"


class CameraSource(abc.ABC):
    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self._index = 0

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Frame | None: ...

    @abc.abstractmethod
    def is_open(self) -> bool: ...

    @abc.abstractmethod
    def release(self) -> None: ...

    def _wrap(self, image) -> Frame:
        f = Frame(image=image, index=self._index, cam_id=self.cam_id)
        self._index += 1
        return f

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()
