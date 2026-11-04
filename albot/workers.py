"""
Background threads that keep the newest Scene and Status published.

Every worker survives errors: a failed frame is logged and the next one tried.
In the original, an exception ended the thread and the bot carried on acting on
the last image it had, indefinitely.
"""
from __future__ import annotations

import logging
import threading
import time

from .state import Latest, Scene, Status
from .safety import ZoneGuard

log = logging.getLogger(__name__)


class Workers:
    def __init__(self, perceiver, config) -> None:
        self.perceiver = perceiver
        self.config = config
        self.scenes: Latest[Scene] = Latest()
        self.statuses: Latest[Status] = Latest()
        self.zone_guard = getattr(perceiver, "zone_guard", None) or ZoneGuard(config)
        self.running = threading.Event()   # cleared while paused
        self.stopping = threading.Event()
        # set by the farm behavior while it waits for a loading bar, so the loading
        # bar model only costs GPU time when the answer is needed
        self.want_loading_bar = threading.Event()
        self.errors = 0
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.running.set()
        for i in range(self.config.perception.scene_workers):
            self._spawn(f"scene-{i + 1}", self._scene_loop)
        self._spawn("status", self._status_loop)

    def _spawn(self, name: str, target) -> None:
        t = threading.Thread(target=self._guarded, args=(name, target), name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _guarded(self, name: str, step) -> None:
        log.debug("%s started", name)
        consecutive = 0
        while not self.stopping.is_set():
            if not self.running.is_set():
                self.running.wait(0.5)
                continue
            try:
                step()
                consecutive = 0
            except Exception:
                self.errors += 1
                consecutive += 1
                # log the full trace once, then just count, so a persistent failure
                # doesn't flood the log
                if consecutive == 1:
                    log.exception("%s failed; retrying", name)
                elif consecutive % 50 == 0:
                    log.error("%s has failed %d times in a row", name, consecutive)
                time.sleep(min(2.0, 0.1 * consecutive))
        log.debug("%s stopped", name)

    def _scene_loop(self) -> None:
        self.scenes.publish(self.perceiver.scan_scene())

    def _status_loop(self) -> None:
        start = time.monotonic()
        status = self.perceiver.scan_status(self.want_loading_bar.is_set())
        if self.perceiver.window.is_active():
            self.zone_guard.observe(status, time.monotonic())
        self.statuses.publish(status)
        period = 1.0 / max(0.1, self.config.perception.status_hz)
        rest = period - (time.monotonic() - start)
        if rest > 0:
            self.stopping.wait(rest)

    def pause(self) -> None:
        self.running.clear()

    def resume(self) -> None:
        self.running.set()

    def stop(self) -> None:
        self.stopping.set()
        self.running.set()
        for t in self._threads:
            t.join(timeout=5)
