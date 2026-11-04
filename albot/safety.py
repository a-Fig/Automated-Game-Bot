"""Death episodes and journey damage/retreat guards, driven by fresh HUD readings."""
from __future__ import annotations

import logging
import math
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SafetyStop(Exception):
    """Unwind a behavior when safety suspends it or hands control to retreat."""


class ZoneGuard:
    """The first visible zone label is trusted; loading/change latches until restart."""
    def __init__(self, config):
        self.s, self.stale_after = config.safety, config.perception.stale_after
        x0, y0, x1, y1 = self.s.zone_label_rect
        w, h = self.s.zone_label_size
        if (not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h
                or not 0 < self.s.zone_label_threshold <= 255 or self.s.zone_label_seconds <= 0
                or not 0 <= self.s.zone_loading_mean <= 255):
            raise ValueError("invalid zone guard rectangle or thresholds")
        self.reference = None
        self.changed_at = self.timestamp = None
        self.debug_at = None
        self.reason = ""  # written last; the decision thread only reads worker-owned state

    def observe(self, status, now):
        if not self.s.zone_guard or self.reason:
            return self.reason
        if status is None or not 0 <= now-status.timestamp <= self.stale_after:
            self.changed_at = None
            return ""
        if self.timestamp is not None and status.timestamp <= self.timestamp:
            return ""
        if self.timestamp is not None and status.timestamp-self.timestamp > self.stale_after:
            self.changed_at = None
        self.timestamp = status.timestamp
        frame = status.frame
        if frame is None:
            self.changed_at = None
            return ""
        pixels = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
        if pixels.ndim != 3 or pixels.shape[2] < 3 or not pixels.size:
            self.changed_at = None
            return ""
        mean = float(pixels[:, :, :3].mean())
        if mean < self.s.zone_loading_mean:
            self.reason = f"loading frame (mean {mean:.1f})"
            return self.reason
        x0, y0, x1, y1 = self.s.zone_label_rect
        w, h = self.s.zone_label_size
        height, width = pixels.shape[:2]
        a, b, c, d = round(x0*width/w), round(y0*height/h), round(x1*width/w), round(y1*height/h)
        if c <= a or d <= b:
            self.changed_at = None
            return ""
        crop = pixels[b:d, a:c, :3]
        if not hasattr(frame, "convert"):
            crop = crop[:, :, ::-1]  # numpy captures are BGR; compare in RGB
        crop = cv2.resize(crop, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.reference is None:
            self.reference = crop.copy()
            self.debug_at = status.timestamp
            return ""
        difference = float(np.abs(crop-self.reference).mean())
        if status.timestamp-self.debug_at >= 30:
            log.debug("zone label diff=%.1f threshold=%.1f", difference, self.s.zone_label_threshold)
            self.debug_at = status.timestamp
        if difference > self.s.zone_label_threshold:
            if self.changed_at is None:
                self.changed_at = status.timestamp
            if status.timestamp-self.changed_at >= self.s.zone_label_seconds:
                self.reason = f"zone label difference {difference:.1f} held {status.timestamp-self.changed_at:.1f}s"
        else:
            self.changed_at = None
        return self.reason


class TravelGuard:
    """Fresh HP drops outside active fights signal an attacker during a journey."""
    def __init__(self, config) -> None:
        self.s = config.safety
        self.stale_after = config.perception.stale_after
        if (not 0 < self.s.travel_abort_health < 1 or not 0 < self.s.attack_health_drop < 1
                or self.s.attack_window_seconds <= 0 or self.s.attack_hold_seconds <= 0
                or self.s.retreat_calm_seconds <= 0 or self.s.critical_retreat_seconds <= 0
                or not 0 <= self.s.retreat_drop_epsilon < 1 or self.s.retreat_dash_cooldown < 0):
            raise ValueError("invalid travel safety settings")
        self.samples = deque()
        self.last_timestamp = None
        self.until = 0.0
        self.last_hp = None
        self.last_drop_at = 0.0
        self.retreat_started = None
        self.critical_since = None

    def under_attack(self, now: float) -> bool:
        return now < self.until

    def valid(self, status, now: float) -> bool:
        return (status is not None and 0 <= now-status.timestamp <= self.stale_after
                and status.health_fraction is not None and math.isfinite(status.health_fraction)
                and 0 <= status.health_fraction <= 1)

    def abort(self, status, now: float) -> bool:
        return self.valid(status, now) and status.health_fraction < self.s.travel_abort_health

    def start_retreat(self, now: float, critical: bool = False) -> None:
        if self.retreat_started is None:
            self.retreat_started = now
            self.last_drop_at = now
        if critical and self.critical_since is None:
            self.critical_since = now

    def retreat_expired(self, now: float) -> bool:
        return self.critical_since is not None and now-self.critical_since >= self.s.critical_retreat_seconds

    def calm(self, now: float) -> bool:
        return now-self.last_drop_at >= self.s.retreat_calm_seconds

    def finish_retreat(self) -> None:
        self.retreat_started = self.critical_since = None
        self.samples.clear()
        if self.last_timestamp is not None and self.last_hp is not None:
            self.samples.append((self.last_timestamp, self.last_hp))
        self.until = 0.0

    def observe(self, status, now: float, enabled: bool, fighting: bool) -> bool:
        if not enabled:
            self.samples.clear()
            self.last_timestamp, self.until = None, 0.0
            self.last_hp = None
            self.retreat_started = self.critical_since = None
            return False
        if not self.valid(status, now):
            self.samples.clear()
            return False
        if self.last_timestamp is not None and status.timestamp > self.last_timestamp:
            if self.last_hp is not None and self.last_hp-status.health_fraction > self.s.retreat_drop_epsilon:
                self.last_drop_at = now
        if self.last_timestamp is None or status.timestamp > self.last_timestamp:
            self.last_hp = status.health_fraction
        if fighting:
            self.samples.clear()  # damage during a fight is expected, not a new travel incident
            self.samples.append((status.timestamp, status.health_fraction))
            self.last_timestamp = status.timestamp
            return False
        if self.last_timestamp is not None and status.timestamp <= self.last_timestamp:
            return False
        self.last_timestamp = status.timestamp
        while self.samples and status.timestamp-self.samples[0][0] > self.s.attack_window_seconds:
            self.samples.popleft()
        drop = max((hp-status.health_fraction for _, hp in self.samples), default=0.0)
        self.samples.append((status.timestamp, status.health_fraction))
        if drop > self.s.attack_health_drop + 1e-9:
            new = not self.under_attack(now)
            self.until = now+self.s.attack_hold_seconds
            return new
        return False


class DeathGuard:
    def __init__(self, config) -> None:
        self.s = config.safety
        self.stale_after = config.perception.stale_after
        if (self.s.death_limit < 1 or self.s.death_window_seconds <= 0
                or self.s.minimum_health_seconds <= 0 or self.s.recovery_seconds <= 0
                or not 0 <= self.s.zero_health <= self.s.minimum_health < self.s.recovery_health <= 1):
            raise ValueError("invalid death guard settings")
        self.deaths: list[float] = []
        self.dead = False
        self.minimum_since = None
        self.recovery_since = None
        self.last_timestamp = None

    @property
    def unsafe(self) -> bool:
        return self.dead

    def observe(self, status, now: float) -> bool:
        """True only for a newly detected death. Stale samples cannot confirm a hold."""
        if status is None or now - status.timestamp > self.stale_after:
            self.minimum_since = self.recovery_since = None
            return False
        if self.last_timestamp is not None and status.timestamp <= self.last_timestamp:
            return False
        if self.last_timestamp is not None and status.timestamp - self.last_timestamp > self.stale_after:
            self.minimum_since = self.recovery_since = None
        self.last_timestamp = status.timestamp
        health = status.health_fraction
        if health is None or not math.isfinite(health) or not 0 <= health <= 1:
            self.minimum_since = self.recovery_since = None
            return False
        if self.dead:
            if health >= self.s.recovery_health:
                if self.recovery_since is None:
                    self.recovery_since = status.timestamp
                if status.timestamp - self.recovery_since >= self.s.recovery_seconds:
                    self.dead = False
                    self.recovery_since = None
            else:
                self.recovery_since = None
            return False
        if health <= self.s.minimum_health:
            if self.minimum_since is None:
                self.minimum_since = status.timestamp
            if (health <= self.s.zero_health
                    or status.timestamp - self.minimum_since >= self.s.minimum_health_seconds):
                self.dead = True
                self.minimum_since = None
                self.deaths = [t for t in self.deaths if now - t <= self.s.death_window_seconds]
                self.deaths.append(now)
                return True
        else:
            self.minimum_since = None
        return False
