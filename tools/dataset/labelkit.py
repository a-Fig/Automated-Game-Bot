"""
Shared pieces for the dataset tools: class schema, HUD masks, box helpers and a cached
client for the cheap vision-language model.

Nothing here prints or stores the API key.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# --------------------------------------------------------------------------- schema

# "red" (AoE telegraph) was dropped after the first full run: it fired on shadows and red leaves, not telegraphs.
# To bring it back put "red" between "fiber_depleted" and "silver" here and in OLD_TO_NEW.
CLASSES = ["player", "mob", "hide", "fiber", "fiber_depleted", "silver"]
CLASS_ID = {n: i for i, n in enumerate(CLASSES)}

# Old detector class -> new class, used as a hint only (hide/mob were polluted, so both of those
# map to "mob" and the VLM decides alive vs dead later). Old ore/stone/tree are not in the schema.
OLD_TO_NEW = {"player": "player", "mob": "mob", "hide": "mob", "fiber": "fiber", "silver": "silver"}

# --------------------------------------------------------------------------- HUD

REF_SIZE = (2240, 1400)
# (x1, y1, x2, y2) in 2240x1400 reference pixels. One place to edit.
HUD_REGIONS = {
    "top_left_hud": (0, 0, 300, 160),
    "top_right_icons": (1650, 0, 2240, 100),
    "right_buttons": (2150, 80, 2240, 260),
    "action_bar": (680, 1290, 1580, 1400),
    "minimap_ping": (1780, 1000, 2240, 1400),
    "chat_button": (0, 1340, 120, 1400),
}


def hud_boxes(size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """HUD rectangles scaled to a frame of size (w, h)."""
    sx, sy = size[0] / REF_SIZE[0], size[1] / REF_SIZE[1]
    return [(a * sx, b * sy, c * sx, d * sy) for a, b, c, d in HUD_REGIONS.values()]


def in_hud(cx: float, cy: float, size: tuple[int, int]) -> bool:
    return any(a <= cx <= c and b <= cy <= d for a, b, c, d in hud_boxes(size))


# --------------------------------------------------------------------------- boxes
# Boxes are (x1, y1, x2, y2) floats in frame pixels.

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(a: Sequence[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def clip_box(b: Sequence[float], size: tuple[int, int]) -> tuple[float, float, float, float]:
    return (max(0.0, b[0]), max(0.0, b[1]), min(float(size[0]), b[2]), min(float(size[1]), b[3]))


def center(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def contained(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of the smaller box that lies inside the other (1.0 = fully nested)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1.0, min(area(a), area(b)))


def nms(items: list[dict], thr: float, contain: float = 1.1) -> list[dict]:
    """
    Greedy same-label NMS over dicts with 'label', 'box' and 'score'. A box is also dropped when
    at least `contain` of the smaller box lies inside an already-kept one (default: never).
    """
    items = sorted(items, key=lambda d: -d.get("score", 0.0))
    keep: list[dict] = []
    for it in items:
        if all(not (k["label"] == it["label"]
                    and (iou(k["box"], it["box"]) >= thr or contained(k["box"], it["box"]) >= contain)) for k in keep):
            keep.append(it)
    return keep


# --------------------------------------------------------------------------- env / keys

def user_env(name: str) -> str | None:
    """An environment variable, falling back to the Windows *user* environment (registry)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            v, _ = winreg.QueryValueEx(k, name)
            if v:
                os.environ[name] = v
                return v
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- VLM client

API_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3-contributor"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.20 / 1e6  # USD per token


def _api_key() -> str:
    p = Path.home() / ".config" / "muse" / "auth.json"
    return json.loads(p.read_text(encoding="utf-8"))["providers"]["meta"]["api_key"]


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class Vlm:
    """
    Cached chat-completion client. The cache key is sha256(image bytes + prompt + model + effort),
    so changing a prompt re-queries only that prompt. Token counts are stored with each cached
    answer; `spent` counts only calls actually made in this run, `cached` the ones served from disk.
    """

    def __init__(self, cache_dir: Path, effort: str = "low", max_tokens: int = 4000, workers: int = 10) -> None:
        self.dir = Path(cache_dir) / "vlm"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.effort, self.max_tokens, self.workers = effort, max_tokens, workers
        self._key: str | None = None
        self._lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(workers)  # global cap on in-flight requests
        self.spent = {"calls": 0, "in": 0, "out": 0}
        self.cached = {"calls": 0, "in": 0, "out": 0}
        self.failures = 0
        self.rate_limited = 0

    def _auth(self) -> str:
        if self._key is None:
            self._key = _api_key()
        return self._key

    def key_for(self, jpeg: bytes, prompt: str, effort: str, extra: Sequence[bytes] = ()) -> str:
        h = hashlib.sha256()
        for e in extra:
            h.update(hashlib.sha256(e).digest())
        h.update(jpeg)
        h.update(b"\0" + prompt.encode("utf-8") + b"\0" + MODEL.encode() + b"\0" + effort.encode())
        return h.hexdigest()

    def ask(self, jpeg: bytes, prompt: str, effort: str | None = None, extra: Sequence[bytes] = (), retries: int = 6) -> str | None:
        """Text answer for one image + prompt (`extra` images, e.g. a reference sheet, come first), or None if every attempt failed."""
        import requests
        effort = effort or self.effort
        key = self.key_for(jpeg, prompt, effort, extra)
        path = self.dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            with self._lock:
                self.cached["calls"] += 1
                self.cached["in"] += rec["in"]
                self.cached["out"] += rec["out"]
            return rec["text"]
        body = {
            "model": MODEL,
            "reasoning_effort": effort,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                  for j in (*extra, jpeg)],
            ]}],
        }
        delay = 2.0
        for attempt in range(retries):
            try:
                with self._sem:
                    r = requests.post(API_URL, json=body, timeout=120,
                                      headers={"Authorization": f"Bearer {self._auth()}"})
                if r.status_code == 200:
                    j = r.json()
                    text = j["choices"][0]["message"].get("content")
                    u = j.get("usage") or {}
                    tin, tout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
                    with self._lock:
                        self.spent["calls"] += 1
                        self.spent["in"] += tin
                        self.spent["out"] += tout
                    if text:  # null content means the reasoning ate max_tokens; don't cache, retry
                        path.write_text(json.dumps({"text": text, "in": tin, "out": tout}), encoding="utf-8")
                        return text
                elif r.status_code == 429:
                    with self._lock:
                        self.rate_limited += 1
                    time.sleep(delay + 5)  # extra back-off on top of the loop's own
                elif r.status_code not in (408, 409, 425, 429) and r.status_code < 500:
                    # a real client error (bad request, auth): retrying won't help
                    with self._lock:
                        self.failures += 1
                    return None
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 2, 30)
        with self._lock:
            self.failures += 1
        return None

    def map(self, jobs: Iterable[tuple], fn: Callable[..., Any] | None = None) -> list[Any]:
        """Run ask(*job) for every job in parallel, in order."""
        jobs = list(jobs)
        with ThreadPoolExecutor(max(1, min(len(jobs), self.workers))) as ex:
            return list(ex.map(lambda j: self.ask(*j), jobs))

    def cost_line(self) -> str:
        def usd(d: dict) -> float:
            return d["in"] * PRICE_IN + d["out"] * PRICE_OUT
        s, c = self.spent, self.cached
        return (f"VLM this run: {s['calls']} calls, {s['in']} in / {s['out']} out tokens, ${usd(s):.4f}; "
                f"served from cache: {c['calls']} calls ({c['in']} in / {c['out']} out, ${usd(c):.4f} saved); "
                f"failures: {self.failures}, rate-limited (429) responses: {self.rate_limited}")


def parse_json(text: str | None) -> Any:
    """Pull the first JSON array/object out of a model answer (tolerates code fences and chatter)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    pairs = [("[", "]"), ("{", "}")]
    pairs.sort(key=lambda p: t.find(p[0]) if t.find(p[0]) != -1 else len(t))  # whichever opens first
    for opener, closer in pairs:
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    return None
