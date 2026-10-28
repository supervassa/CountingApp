"""Detector interface + Detection record.

Backends (onnxruntime now, TensorRT on Jetson later) implement Detector so the
pipeline never depends on a specific inference engine.
"""

import abc
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int
    label: str = ""

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def wh(self) -> Tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def anchor_head(self) -> Tuple[float, float]:
        """Top-centre of the box — proxy for the head, used by line crossing."""
        return (self.x1 + self.x2) / 2, self.y1

    @property
    def anchor_centroid(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """image: BGR HxWx3 uint8 -> detections in image pixel coords."""
