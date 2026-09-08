"""Pipeline orchestrator: open both cameras, pull frames, detect, track.

Recognition / counting hang off process_frame() in later capaian.
The Detector is shared across both cameras (§5.3 PRD); each camera keeps its
own Tracker (state is per-camera).
"""
from __future__ import annotations

import logging
import signal
import time

import cv2

from .camera.factory import build_camera
from .detection.factory import build_detector
from .tracking.factory import build_tracker

log = logging.getLogger(__name__)

# deterministic-ish colour per track id
_PALETTE = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
            (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]


def _color(tid: int):
    return _PALETTE[tid % len(_PALETTE)]


class FpsMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self._t: list[float] = []

    def tick(self) -> float:
        now = time.time()
        self._t.append(now)
        if len(self._t) > self.window:
            self._t.pop(0)
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cams = {
            name: build_camera(name, getattr(cfg.cameras, name))
            for name in ("cam_out", "cam_in")
        }
        self.meters = {name: FpsMeter() for name in self.cams}
        self._stop = False

        # shared detector; keep running the capture loop even if the model is absent
        self.detector = None
        try:
            self.detector = build_detector(cfg.detection)
        except (FileNotFoundError, ImportError) as e:
            log.warning("detection disabled: %s", e)

        # one tracker per camera
        self.trackers = {name: build_tracker(cfg.tracking) for name in self.cams}

    def _install_signals(self):
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, lambda *_: setattr(self, "_stop", True))

    def process_frame(self, name, frame):
        """Detect, then track. Recognition/counting hook in here later."""
        img = frame.image
        if self.detector is None:
            return img
        dets = self.detector.detect(img)
        tracks = self.trackers[name].update(dets, img.shape)

        out = img.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.xyxy
            col = _color(t.track_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            tag = f"#{t.track_id} {t.score:.2f}"
            if t.time_since_update:
                tag += f" (lost {t.time_since_update})"
            cv2.putText(out, tag, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            pts = list(t.trajectory)
            for a, b in zip(pts, pts[1:]):
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)
        if frame.index % 30 == 0:
            log.info("[%s] frame %d: %d det -> %d track(s) %s",
                     name, frame.index, len(dets), len(tracks),
                     sorted(t.track_id for t in tracks))
        return out

    def run(self):
        self._install_signals()
        for cam in self.cams.values():
            cam.open()
        display = bool(getattr(self.cfg.app, "display", False))
        target_dt = 1.0 / max(1, int(self.cfg.app.target_fps))
        log.info("pipeline start (display=%s, target_fps=%s)", display, self.cfg.app.target_fps)

        try:
            while not self._stop:
                loop_t = time.time()
                for name, cam in self.cams.items():
                    if not cam.is_open():
                        log.error("[%s] camera closed, stopping", name)
                        self._stop = True
                        break
                    frame = cam.read()
                    if frame is None:
                        continue
                    annotated = self.process_frame(name, frame)
                    fps = self.meters[name].tick()
                    if frame.index % 30 == 0:
                        log.info("[%s] frame %d  %.1f fps", name, frame.index, fps)
                    if display:
                        cv2.putText(annotated, f"{name} {fps:5.1f} fps", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(name, annotated)
                if display and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self._stop = True
                slack = target_dt - (time.time() - loop_t)
                if slack > 0:
                    time.sleep(slack)
        finally:
            for cam in self.cams.values():
                cam.release()
            if display:
                cv2.destroyAllWindows()
            log.info("pipeline stopped")
