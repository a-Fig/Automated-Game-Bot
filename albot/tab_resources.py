"""Read the five fixed Tab-header slots from saved frames; never send input.

The header covers T2+, not T1. Faded/--- slots are not all-tier negatives;
ambient T1 occurrences are retained separately and do not define zone types.
Icon/label templates come from fg_04, with fg_01..03 as held-out layout checks.
Only the II-III caption is calibrated; other clear tier captions retain types
with null tiers rather than guessing a Roman numeral the templates cannot read.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .navigation import as_bgr

TEMPLATES = Path(__file__).resolve().parents[1] / "assets/ui/tab_resources"


def _score(image, template):
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        return -1.0, (0, 0)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, position = cv2.minMaxLoc(result)
    return float(score), position


def extract_tab_resources(frame, capture, captured_at, templates=TEMPLATES):
    folder = Path(templates)
    layout = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
    image = as_bgr(frame)
    if list(image.shape[1::-1]) != layout["window_size"]:
        raise ValueError("Tab resource templates require 2240x1400; recalibrate for another HUD scale")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    captions = [(slot["caption_tiers"], cv2.imread(str(folder / slot["caption"]), cv2.IMREAD_GRAYSCALE))
                for slot in layout["slots"]]
    resources, absent, readings, warnings = [], [], {}, []
    for slot in layout["slots"]:
        name = slot["type"]
        x0, y0, x1, y1 = slot["icon_rect"]
        icon = cv2.imread(str(folder / slot["icon"]), cv2.IMREAD_GRAYSCALE)
        score, (dx, dy) = _score(gray[y0-4:y1+4, x0-4:x1+4], icon)
        if score < 0.83:
            raise ValueError(f"{name} icon not recognized ({score:.3f}); require a clear Tab header")
        dx, dy = dx-4, dy-4
        a, b, c, d = slot["caption_rect"]
        caption = gray[b+dy:d+dy, a+dx:c+dx]
        absent_score = max(_score(caption, patch)[0] for tiers, patch in captions if tiers is None)
        tier_score, matched_tiers = max((_score(caption, patch)[0], tiers)
                                       for tiers, patch in captions if tiers is not None)
        if absent_score >= 0.97 and absent_score > tier_score:
            absent.append(name)
            state, tiers, confidence = "absent_t2plus", None, absent_score
        elif tier_score >= 0.985 and tier_score > absent_score:
            state, tiers, confidence = "present", matched_tiers, tier_score
        else:
            # Two tall glyph groups around a short central dash establish a tier
            # caption, but not which numerals. Icon shape alone cannot read tiers.
            ink = (caption < np.median(caption)*0.6).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
            tall = [s for s in stats[1:] if s[3] >= 6]
            dash = any(18 <= s[0] <= 34 and s[2] >= 2 and s[3] <= 3 for s in stats[1:])
            if len(tall) < 2 or not dash or not any(s[0] < 22 for s in tall) or not any(s[0] > 25 for s in tall):
                raise ValueError(f"{name} caption unreadable; catalogue left unchanged")
            state, tiers, confidence = "present", None, None
            warnings.append(f"{name}: tier caption uncalibrated; tiers left null")
        readings[name] = {"state": state, "icon_score": round(score, 4),
                          "caption_score": round(confidence, 4) if confidence is not None else None}
        if state == "present":
            resources.append({"type": name, "tiers": tiers, "role": None})
    return {"source": "ingame_tab", "capture": str(capture), "captured_at": captured_at,
            "extracted_at": datetime.now(timezone.utc).isoformat(), "tab_header_min_tier": 2,
            "complete_t2plus": True, "resources": resources, "absent_t2plus": absent,
            "readings": readings, "warnings": warnings}


def update_catalog(path, zone, observation, name=None):
    if not re.fullmatch(r"[a-z0-9_]+", zone):
        raise ValueError("zone must use lowercase letters, digits and underscores")
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "zones": {}}
    if data.get("version") != 1:
        raise ValueError("unsupported zone resource catalogue version")
    entry = data["zones"].setdefault(zone, {"name": name or zone.replace("_", " ").title(), "observations": []})
    # Refresh only this capture, preserving user/T1/wiki observations and other zones.
    key = lambda o: (o.get("source"), Path(o.get("capture", "").replace("\\", "/")).name, o.get("captured_at"))
    entry["observations"] = [o for o in entry["observations"] if key(o) != key(observation)] + [observation]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)
