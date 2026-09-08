"""Load and validate config/config.yaml.

Access is attribute-style: cfg.cameras.cam_out.source
Fail loud and early on a missing/wrong key — better than a KeyError deep in the loop.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

_REQUIRED = (
    "app.target_fps",
    "cameras.cam_out.source",
    "cameras.cam_in.source",
    "counting.cooldown_s",
    "database.sqlite_path",
)
_VALID_SOURCES = {"webcam", "file", "csi"}


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _get(ns, dotted: str):
    cur = ns
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"config missing: {dotted}")
        cur = getattr(cur, part)
    return cur


def load_config(path: str | Path = "config/config.yaml") -> SimpleNamespace:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = _to_ns(raw)

    for key in _REQUIRED:
        _get(cfg, key)  # raises if absent

    for cam_name in ("cam_out", "cam_in"):
        cam = getattr(cfg.cameras, cam_name)
        if cam.source not in _VALID_SOURCES:
            raise ValueError(f"cameras.{cam_name}.source={cam.source!r} not in {_VALID_SOURCES}")
        if cam.source == "file" and not getattr(cam, "file_path", None):
            raise ValueError(f"cameras.{cam_name}.source=file but file_path is empty")

    cfg._path = str(path)
    return cfg
