"""Capaian 1 orchestrator: open both cameras, pull frames, measure FPS.

Detection / tracking / recognition / counting hang off process_frame() in later
capaian. For now it just proves the dual-camera capture loop is stable.
"""
from __future__ import annotations

import logging
import signal
import time

import cv2

from .camera.factory import build_camera

log = logging.getLogger(__name__)


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

    def _install_signals(self):
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, lambda *_: setattr(self, "_stop", True))

    def process_frame(self, name, frame):
        """Hook for later capaian (detect/track/recognize/count). No-op for now."""
        return frame.image

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
