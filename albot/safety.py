"""Death episodes: count once, and require sustained recovery before re-arming."""
from __future__ import annotations

import math


class SafetyStop(Exception):
    """Unwind the current behavior without sending another input."""


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
