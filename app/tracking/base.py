"""Tracker interface + Track record.

One Tracker instance PER camera (state is per-camera); the detector is shared.
"""
from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int = 0
    age: int = 0                 # frames since first seen
    hits: int = 0               # total updates
    time_since_update: int = 0  # frames since last matched to a detection
    trajectory: deque = field(default_factory=lambda: deque(maxlen=64))

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def anchor_head(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, self.y1

    @property
    def anchor_centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class Tracker(abc.ABC):
    @abc.abstractmethod
    def update(self, detections, frame_shape: tuple[int, int]) -> list[Track]:
        """detections: list[app.detection.base.Detection]; returns active tracks."""
