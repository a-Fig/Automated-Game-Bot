"""World-input exclusion and minimap recovery proposals; no input is sent here."""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np


def hud_reason(point, size, safety):
    w, h = size
    bw, bh = safety.hud_size
    x, y = point.x * bw / w, point.y * bh / h
    if not (0 <= x < bw and 0 <= y < bh):
        return "outside the window"
    for rect in safety.hud_rects:
        x0, y0, x1, y1 = rect
        if x0 <= x < x1 and y0 <= y < y1:
            return f"HUD rectangle {list(rect)}"
    return ""


def minimap_panel_state(crop):
    """Look for long tan runs on the diamond frame, not the arrow or terrain.

    These three frame segments avoid compass letters, buttons and the label.
    A visible frame on even one side vetoes recovery. Weak frame evidence and
    flat/black panels are unknown, rather than reasons to toggle a visible map.
    The two real run-13 crops calibrate this geometry at 420x370.
    """
    image = cv2.resize(crop, (420, 370), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    tan = ((hue >= 12) & (hue <= 30) & (saturation >= 45)
           & (saturation <= 190) & (value >= 25))  # Tab overlay dims the real frame below 75
    scores = []
    for a, b in (((85, 166), (209, 82)), ((85, 203), (210, 285)), ((244, 285), (375, 198))):
        xy = np.round(np.linspace(a, b, 100)).astype(int)
        scores.append(max(float(tan[xy[:, 1]+dy, xy[:, 0]].mean()) for dy in range(-2, 3)))
    if max(scores) >= 0.55:
        return "visible"
    if max(scores) < 0.25 and float(value.mean()) > 25 and float(value.std()) > 8:
        return "hidden"
    return "unknown"


class MinimapRecovery:
    """Fresh readable HUD + missing arrow + missing frame must agree before a toggle.

    A toggle is a proposal, counted only when the focused input path dispatches
    it. Three unsuccessful attempts latch a stop; a real arrow fix resets the
    failure count. Visible parchment with an occluded arrow is never toggled.
    """
    def __init__(self, config):
        from .navigation import MapLocator

        self.s, self.stale_after = config.safety, config.perception.stale_after
        s, nav = self.s, config.navigation
        if (s.minimap_missing_seconds <= 0 or s.minimap_toggle_interval <= 0
                or s.minimap_toggle_attempts < 1):
            raise ValueError("invalid minimap recovery timing or attempts")
        x0, y0, x1, y1 = s.minimap_panel_rect
        bw, bh = s.hud_size
        if not (0 <= x0 < x1 <= bw and 0 <= y0 < y1 <= bh):
            raise ValueError("invalid minimap panel rectangle")
        # Locate in the small normalized panel, including at a scaled window size.
        # The locator still sees its configured minimap pixels, not a new map space.
        sx, sy = bw / nav.window_size[0], bh / nav.window_size[1]
        rect = tuple(round(v * scale - offset) for v, scale, offset in
                     zip(nav.minimap_rect, (sx, sy, sx, sy), (x0, y0, x0, y0)))
        self.locator = MapLocator(replace(nav, window_size=(x1-x0, y1-y0), minimap_rect=rect))
        self.timestamp = self.missing_at = None
        self.toggled_at = -math.inf
        self.attempts = 0
        self.reason = ""

    def observe(self, status, now):
        if not self.s.minimap_recovery or self.reason:
            return False
        if (status is None or now-status.timestamp > self.stale_after
                or status.health_fraction is None or status.frame is None):
            self.missing_at = None
            return False
        if self.timestamp is not None and status.timestamp <= self.timestamp:
            return False
        if self.timestamp is not None and status.timestamp-self.timestamp > self.stale_after:
            self.missing_at = None
        self.timestamp = status.timestamp
        from .navigation import as_bgr

        try:
            image = as_bgr(status.frame)
        except ValueError:
            self.missing_at = None
            return False
        x0, y0, x1, y1 = self.s.minimap_panel_rect
        bw, bh = self.s.hud_size
        h, w = image.shape[:2]
        left, top, right, bottom = (round(x0*w/bw), round(y0*h/bh), round(x1*w/bw), round(y1*h/bh))
        crop = image[top:bottom, left:right]
        if not crop.size:
            self.missing_at = None
            return False
        panel = cv2.resize(crop, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
        # Custom navigation rectangles may not fit this panel; do not guess a toggle.
        nx0, ny0, nx1, ny1 = self.locator.config.minimap_rect
        if not (0 <= nx0 < nx1 <= panel.shape[1] and 0 <= ny0 < ny1 <= panel.shape[0]):
            return False
        if self.locator.locate(panel, status.timestamp) is not None:
            self.missing_at = None
            self.attempts = 0
            return False
        if minimap_panel_state(panel) != "hidden":
            self.missing_at = None
            return False
        if self.missing_at is None:
            self.missing_at = status.timestamp
        if status.timestamp-max(self.missing_at, self.toggled_at) < self.s.minimap_missing_seconds:
            return False
        if self.attempts >= self.s.minimap_toggle_attempts:
            self.reason = f"minimap hidden after {self.attempts} failed toggles"
            return False
        return now-self.toggled_at >= self.s.minimap_toggle_interval

    def toggled(self, now):
        self.attempts += 1
        self.toggled_at = now
