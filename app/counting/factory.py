"""Build per-camera CrossingCounter + the shared Occupancy from config."""
from __future__ import annotations

from .band import Band
from .counter import CrossingCounter
from .occupancy import Occupancy


def build_band(cfg_counting, cam_id: str) -> Band:
    node = getattr(cfg_counting.band, cam_id)
    return Band(node.outer, node.inner)


def build_counter(cfg_counting, cam_id: str, role: str) -> CrossingCounter:
    return CrossingCounter(
        camera_id=cam_id,
        role=role,
        band=build_band(cfg_counting, cam_id),
        anchor=getattr(cfg_counting, "crossing_anchor", "head"),
        min_track_len=getattr(cfg_counting, "min_track_len", 5),
        confirm_frames=getattr(cfg_counting, "confirm_frames", 3),
        cooldown_s=getattr(cfg_counting, "cooldown_s", 2.0),
    )


def build_occupancy(cfg_counting) -> Occupancy:
    return Occupancy(dedup_window_s=getattr(cfg_counting, "dedup_window_s", 2.0))
