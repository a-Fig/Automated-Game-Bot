"""
Auto-label game frames into a YOLO dataset using two noisy sources that check each other.

    albot-venv\\Scripts\\python.exe tools\\dataset\\autolabel.py <frames_dir> <out_dir>

A  the old Roboflow detector (confidence >= 0.20; its class names are hints only)
B  a cheap vision-language model (VLM) run over overlapping tiles of the frame
Both are merged by IoU. A box is accepted without a further call only when A and B agree on a
"stable" class (player). Everything else (single-source boxes,
disagreements, and every other class: mob / hide / fiber / fiber_depleted, where the old detector
can't tell alive from dead or live from harvested, and red / silver, which are easy to hallucinate) gets a
padded crop sent to the VLM together with a reference sheet of labelled examples (tools/dataset/refs/,
named <class>__<description>.png; "none__" files are look-alikes that must not be labelled). It picks a
class or "none"; "none" boxes are dropped.

Output (out_dir): images/, labels/, data.yaml, preview/, review.csv. VLM answers and detector
output are cached under tools/dataset/cache (override with --cache) keyed by image hash + prompt,
so re-runs are free.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labelkit import (CLASS_ID, CLASSES, HERE, REF_SIZE, OLD_TO_NEW, REPO_ROOT, Vlm, area, center, clip_box, contained,  # noqa: E402
                      encode_jpeg, hud_boxes, iou, in_hud, nms, parse_json, user_env)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STABLE = {"player"}  # A+B agreement is enough for these; every other class gets a crop check
DET_CONF = 0.20
REF_W = REF_SIZE[0]
# Largest plausible box side per class, in px of a 2240-wide frame (scaled for other sizes). Bigger boxes are
# VLM hallucinations (a whole bush cluster called fiber_depleted, ...).
MAX_SIDE = {"player": 360, "mob": 450, "hide": 450, "fiber": 260, "fiber_depleted": 260, "red": 900, "silver": 160}
# Classes that are two states of one thing: the crop check decides between them, and two boxes of
# different states on one spot are the same object.
FAMILY = {"mob": "creature", "hide": "creature", "fiber": "plant", "fiber_depleted": "plant"}
REFS_DIR = HERE / "refs"

# --------------------------------------------------------------------------- prompts

CLASS_GUIDE = """\
- player: a humanoid PLAYER character (hooded/armoured figure, often with a health bar above it; also when crouching to gather or partly hidden by text). Include the player at the centre of the screen and any others.
- mob: a LIVING creature that is standing, walking, running or fighting: antelope/deer/stag, the Moabird (a big flightless bird), wolf, boar, undead, bandit, monster. Dark or hard-to-see animals that are upright/moving are still mob. A selected mob has a red ring under it: the ring is part of the mob, not a separate object.
- hide: a FRESH animal CORPSE lying motionless on its side on the ground, fur or feathers still intact, no health bar. It can be skinned. It is NOT standing or moving.
- fiber: a harvestable fiber plant with flowers or puffs on it. Cotton: low plant with WHITE or PINK puffs. Flax (green oasis areas, often at night): dark-green leafy plant with small pale-BLUE flowers.
- fiber_depleted: the same fiber plants (cotton or flax) but with bare purple/brown/blue stems and NO flowers or puffs (already harvested).
- red: a red/orange telegraph ring or AoE circle drawn flat on the ground (an enemy ability warning). NOT the thin selection ring right under a single targeted creature.
- silver: a dropped loot bag, silver pouch or pile of coins lying on the ground. A yellow sparkle or glint alone is NOT silver."""

NEGATIVES = """\
Label NOTHING (answer none) for: the bare white SKELETON or bones left after an animal is skinned or long dead (not hide); \
the big leafy yellow-green desert bushes with spiky fronds and the agave-like rosette plants (scenery, not fiber); rocks, boulders, trees, \
cliffs and paving; HUD/UI, floating text, health bars, icons; buildings, market stalls; NPC vendors (a person sitting at a desk/stall with \
books or a lantern); shadows, grass, empty sand."""

TILE_PROMPT = f"""This is one tile cut from a top-down isometric screenshot of the game Albion Online (desert or green oasis zone, day or night).
Find every object of the classes below that is FULLY or mostly visible in this tile and return ONLY a JSON array, no other text:
[{{"label": "<class>", "box": [x1, y1, x2, y2]}}, ...]
Box coordinates are pixels in THIS image (origin top-left). Boxes must be TIGHT around the object (for a creature: the body only, not its shadow, health bar or selection ring). If there are none, return [].

Classes:
{CLASS_GUIDE}

{NEGATIVES}"""

CROP_PROMPT = f"""You get TWO images. Image 1 is a REFERENCE SHEET of labelled examples from this game (each caption says what the example is; "none" examples must get the label none). Image 2 is a small crop from a top-down isometric screenshot of the game Albion Online. A MAGENTA rectangle marks ONE candidate object. Judge only the object inside the magenta rectangle; ignore neighbouring things, even if they are close or overlap the rectangle.
Which class is it? Choose exactly one:
{CLASS_GUIDE}
- none: none of the above.

{NEGATIVES}
Key distinctions: standing/walking/fighting animal = mob; animal lying on its side with intact fur or feathers = hide; bare white bones = none. Plant with flowers/puffs = fiber; same plant bare = fiber_depleted.
Also give a TIGHT box around just that object, in pixel coordinates of THIS crop image (it may be smaller or shifted from the magenta rectangle; exclude shadows, health bars and selection rings). Use [] when the label is none.
Answer with ONLY a JSON object: {{"seen": "<what is really inside the rectangle, max 12 words; say 'empty sand' or 'shadow' if that is all>", "label": "<class or none>", "box": [x1, y1, x2, y2]}}"""


def build_ref_sheet(refs_dir: Path = REFS_DIR, cell_h: int = 190, cols: int = 3) -> bytes | None:
    """One image with every reference crop and a caption (file name: <class>__<description>.png)."""
    files = sorted(refs_dir.glob("*.png")) + sorted(refs_dir.glob("*.jpg"))
    if not files:
        return None
    cells = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if im is None:
            continue
        cls, _, desc = f.stem.partition("__")
        im = cv2.resize(im, (max(1, int(im.shape[1] * cell_h / im.shape[0])), cell_h), interpolation=cv2.INTER_AREA)
        cap = f"{cls.upper()}: {desc.replace('_', ' ')}"
        cells.append((im, cap))
    cw = max(c[0].shape[1] for c in cells)
    cap_h = 26
    rows = []
    for i in range(0, len(cells), cols):
        row = []
        for im, cap in cells[i:i + cols]:
            cell = np.full((cell_h + cap_h, cw, 3), 255, np.uint8)
            cell[cap_h:, :im.shape[1]] = im
            cv2.putText(cell, cap, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            row.append(cell)
        while len(row) < cols:
            row.append(np.full((cell_h + cap_h, cw, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    return encode_jpeg(np.vstack(rows), 92)


# --------------------------------------------------------------------------- A: old detector


class OldDetector:
    """Lazy wrapper: the model is only loaded if some frame is missing from the cache."""

    def __init__(self, cache_dir: Path) -> None:
        self.dir = cache_dir / "det"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _load(self) -> None:
        import os
        for k in ("ROBOFLOW_KEY1", "ROBOFLOW_KEY2"):
            user_env(k)
        sys.path.insert(0, str(REPO_ROOT))
        cwd = os.getcwd()
        os.chdir(REPO_ROOT)  # albot.toml is resolved relative to the repo root
        try:
            from albot.config import load
            import inference
            cfg = load("albot.toml")
            t = time.time()
            self.model = inference.get_model(cfg.models.scene, cfg.key_for_model("scene"))
            print(f"[det] loaded {cfg.models.scene} in {time.time() - t:.0f}s")
        finally:
            os.chdir(cwd)

    def detect(self, img: np.ndarray, key: str) -> list[dict]:
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.model is None:
            self._load()
        preds = self.model.infer(img, confidence=DET_CONF)[0].predictions
        out = []
        for p in preds:
            if p.confidence < DET_CONF:
                continue
            out.append({"old": p.class_name, "conf": round(float(p.confidence), 4),
                        "box": [p.x - p.width / 2, p.y - p.height / 2, p.x + p.width / 2, p.y + p.height / 2]})
        path.write_text(json.dumps(out), encoding="utf-8")
        return out


# --------------------------------------------------------------------------- B: tiles


def tile_origins(size: tuple[int, int], tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Tiles of at most `tile` px that cover the frame with at least `overlap` px between neighbours."""
    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        k = math.ceil((n - tile) / (tile - overlap)) + 1
        return [round(i * (n - tile) / (k - 1)) for i in range(k)]
    return [(x, y, min(x + tile, size[0]), min(y + tile, size[1]))
            for y in starts(size[1]) for x in starts(size[0])]


def vlm_tiles(vlm: Vlm, img: np.ndarray, tile_sizes: list[int], overlap: int) -> list[dict]:
    """Tile passes at each size (a second, smaller size recovers objects the first pass missed)."""
    h, w = img.shape[:2]
    tiles = [t for ts in tile_sizes for t in tile_origins((w, h), ts, overlap)]
    answers = vlm.map([(encode_jpeg(img[y1:y2, x1:x2]), TILE_PROMPT) for x1, y1, x2, y2 in tiles])
    out: list[dict] = []
    edge = 4
    for (x1, y1, x2, y2), ans in zip(tiles, answers):
        data = parse_json(ans)
        if not isinstance(data, list):
            continue
        for d in data:
            try:
                label = str(d["label"]).strip().lower().replace(" ", "_")
                bx = [float(v) for v in d["box"]]
            except Exception:
                continue
            if label not in CLASS_ID or len(bx) != 4:
                continue
            bx = [min(bx[0], bx[2]), min(bx[1], bx[3]), max(bx[0], bx[2]), max(bx[1], bx[3])]
            # A box touching an interior tile edge is cut off; the neighbouring tile sees it whole.
            cut = ((bx[0] <= edge and x1 > 0) or (bx[1] <= edge and y1 > 0)
                   or (bx[2] >= x2 - x1 - edge and x2 < w) or (bx[3] >= y2 - y1 - edge and y2 < h))
            if cut:
                continue
            gb = (bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1)
            cx, cy = center(bx)
            # prefer the tile in which the object is most central
            score = 1.0 - max(abs(cx / (x2 - x1) - 0.5), abs(cy / (y2 - y1) - 0.5))
            out.append({"label": label, "box": gb, "score": score})
    return nms(out, 0.4, contain=0.7)


# --------------------------------------------------------------------------- merge + verify


def match_boxes(a_items: list[dict], b_items: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """One-to-one greedy matching by IoU (>= 0.3, or one box mostly inside the other)."""
    pairs = []
    for i, a in enumerate(a_items):
        for j, b in enumerate(b_items):
            v = iou(a["box"], b["box"])
            ix = min(a["box"][2], b["box"][2]) - max(a["box"][0], b["box"][0])
            iy = min(a["box"][3], b["box"][3]) - max(a["box"][1], b["box"][1])
            inside = (ix * iy) / max(1.0, min(area(a["box"]), area(b["box"]))) if ix > 0 and iy > 0 else 0.0
            if v >= 0.3 or (inside >= 0.8 and v >= 0.15):
                pairs.append((v + 0.5 * (inside >= 0.8), i, j))
    pairs.sort(reverse=True)
    ua, ub, out = set(), set(), []
    for _, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i), ub.add(j)
        out.append((a_items[i], b_items[j]))
    out += [(a, None) for i, a in enumerate(a_items) if i not in ua]
    out += [(None, b) for j, b in enumerate(b_items) if j not in ub]
    return out


def crop_for(img: np.ndarray, box, pad: int = 30, min_side: int = 256):
    """Padded crop with the candidate box outlined in magenta. Returns (crop, x0, y0, scale)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad)), int(min(w, box[2] + pad)), int(min(h, box[3] + pad))
    c = img[y1:y2, x1:x2].copy()
    s = max(1.0, min_side / max(1, min(c.shape[:2])))
    if s > 1.0:
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    p1 = (int(round((box[0] - x1) * s)), int(round((box[1] - y1) * s)))
    p2 = (int(round((box[2] - x1) * s)), int(round((box[3] - y1) * s)))
    cv2.rectangle(c, p1, p2, (255, 0, 255), 2)
    return c, x1, y1, s


def parse_crop_answer(text: str | None):
    """-> (label | 'none' | None, box in crop pixels | None)"""
    label = parse_class(text)
    d = parse_json(text)
    if isinstance(d, list) and d and isinstance(d[0], dict):
        d = d[0]
    box = None
    if isinstance(d, dict) and isinstance(d.get("box"), list) and len(d["box"]) == 4:
        try:
            b = [float(v) for v in d["box"]]
            box = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
        except Exception:
            box = None
    return label, box


def parse_class(text: str | None) -> str | None:
    """'none', a class name, or None when the answer was unusable."""
    d = parse_json(text)
    label = None
    if isinstance(d, dict):
        label = d.get("label")
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        label = d[0].get("label")
    if label is None and text:
        low = text.lower()
        found = [c for c in sorted(CLASSES, key=len, reverse=True) if c in low]
        label = found[0] if found else ("none" if "none" in low else None)
    if label is None:
        return None
    label = str(label).strip().lower().replace(" ", "_")
    return label if label in CLASS_ID or label == "none" else None


def process_frame(path: Path, img: np.ndarray, key: str, det: list[dict], vlm: Vlm, args) -> list[dict]:
    """Returns decision records: {class, source, conf, box, decision, note, keep}."""
    h, w = img.shape[:2]
    size = (w, h)

    a_items = []
    for d in det:
        hint = OLD_TO_NEW.get(d["old"])
        if hint is None:
            continue  # cup of piss / wisp caged: not in the new schema
        a_items.append({"box": tuple(d["box"]), "hint": hint, "old": d["old"], "conf": d["conf"]})
    b_items = [] if args.no_vlm_tiles else vlm_tiles(vlm, img, args.tiles, args.overlap)

    cands: list[dict] = []
    for a, b in match_boxes(a_items, b_items):
        box = a["box"] if a else b["box"]  # the detector's box is usually tighter
        cands.append({"a": a, "b": b, "box": box})

    records: list[dict] = []
    verify: list[dict] = []
    for c in cands:
        a, b = c["a"], c["b"]
        cx, cy = center(c["box"])
        src = ("A+B" if a and b else "A" if a else "B")
        conf = f"old:{a['old']}:{a['conf']:.2f}" if a else ""
        bl = f" vlm:{b['label']}" if b else ""
        rec = {"source": src, "conf": (conf + bl).strip(), "box": clip_box(c["box"], size), "note": ""}
        bw, bh = rec["box"][2] - rec["box"][0], rec["box"][3] - rec["box"][1]
        if in_hud(cx, cy, size):
            records.append({**rec, "class": a["hint"] if a else b["label"], "decision": "drop_hud", "keep": False})
            continue
        if bw < 10 or bh < 10 or area(rec["box"]) > 0.35 * w * h:
            records.append({**rec, "class": a["hint"] if a else b["label"], "decision": "drop_size", "keep": False})
            continue
        if a and b and a["hint"] == b["label"] and b["label"] in STABLE:
            records.append({**rec, "class": b["label"], "decision": "accept_agree", "keep": True})
            continue
        if a and not b and a["hint"] in STABLE and a["conf"] >= args.trust_a:
            records.append({**rec, "class": a["hint"], "decision": "accept_detector", "keep": True})
            continue
        c["rec"] = rec
        verify.append(c)

    # crop verification
    crops = [crop_for(img, c["rec"]["box"], args.pad) for c in verify]
    extra = (args.ref_sheet,) if args.ref_sheet else ()
    answers = vlm.map([(encode_jpeg(cr[0]), CROP_PROMPT, args.crop_effort, extra) for cr in crops])
    for c, cr, ans in zip(verify, crops, answers):
        rec, a, b = c["rec"], c["a"], c["b"]
        label, cbox = parse_crop_answer(ans)
        guess = (b["label"] if b else a["hint"])
        if args.debug_crops:
            d = args.out_dir / "crops"
            d.mkdir(exist_ok=True)
            tag = (label or "noanswer")
            cv2.imwrite(str(d / f"{path.stem}_{int(rec['box'][0])}_{int(rec['box'][1])}_{guess}_{tag}.jpg"), cr[0])
            (d / f"{path.stem}_{int(rec['box'][0])}_{int(rec['box'][1])}_{guess}_{tag}.txt").write_text(f"{ans}\ncrop origin {cr[1]},{cr[2]} scale {cr[3]:.3f}", encoding="utf-8")
        if label is None:
            records.append({**rec, "class": guess, "decision": "drop_no_answer", "keep": False})
        elif label == "none":
            records.append({**rec, "class": guess, "decision": "drop_none", "keep": False})
        else:
            note = "" if label == guess else f"guess was {guess}"
            if a and b and a["hint"] == b["label"] and label != guess and FAMILY.get(label) != FAMILY.get(guess):
                # detector and tile pass agree, the crop contradicts them across families (e.g. mob vs player):
                # two votes beat one. Within a family (alive/dead, flowers/bare) the crop is the arbiter.
                note = f"crop said {label}, kept {guess}"
                label = guess
            # B-only boxes are loose: take the VLM's tightened box when it is sane. A's box stays.
            if a is None and cbox is not None and not args.no_refine:
                ch, cw = cr[0].shape[:2]
                if cbox[2] > cw * 1.25 or cbox[3] > ch * 1.25:  # the model sometimes answers in 0-1000 normalised units
                    cbox = (cbox[0] * cw / 1000, cbox[1] * ch / 1000, cbox[2] * cw / 1000, cbox[3] * ch / 1000)
                _, x0, y0, s = cr
                nb = (x0 + cbox[0] / s, y0 + cbox[1] / s, x0 + cbox[2] / s, y0 + cbox[3] / s)
                if iou(nb, rec["box"]) >= 0.3 and area(nb) <= 1.6 * area(rec["box"]) and area(nb) >= 0.15 * area(rec["box"]):
                    rec = {**rec, "box": clip_box(nb, size)}
                    note = (note + " box refined").strip()
            records.append({**rec, "class": label, "decision": "crop_verified", "keep": True, "note": note})

    # collapse duplicates that ended up with the same class (e.g. old 'mob' and 'hide' on one animal)
    kept = [r for r in records if r["keep"]]
    for r in kept:
        r["score"] = (2.0 if r["source"] == "A+B" else 1.0 if r["source"] == "A" else 0.5) + (0.0 if ("guess was" in r["note"] or "crop said" in r["note"]) else 0.2)
        r["label"] = r["class"]
    survivors = nms(kept, 0.25, contain=0.5)
    ids = {id(r) for r in survivors}
    for r in kept:
        if id(r) not in ids:
            r["keep"] = False
            r["decision"] = "drop_duplicate"
            # two loose VLM-only boxes on one object usually each cover part of it: grow the survivor
            if r["source"] == "B":
                for k in survivors:
                    if k["label"] == r["label"] and k["source"] == "B" and (iou(k["box"], r["box"]) >= 0.25 or contained(k["box"], r["box"]) >= 0.5):
                        u = (min(k["box"][0], r["box"][0]), min(k["box"][1], r["box"][1]), max(k["box"][2], r["box"][2]), max(k["box"][3], r["box"][3]))
                        if area(u) <= 1.5 * max(area(k["box"]), area(r["box"])):
                            k["box"] = u
                            k["note"] = (k["note"] + " merged with duplicate").strip()
                        break

    for r in records:
        if r["keep"]:
            side = max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1])
            if side > MAX_SIDE[r["class"]] * w / REF_W:
                r["keep"] = False
                r["decision"] = "drop_size"
                r["note"] = (r["note"] + " too big for class").strip()

    # class-agnostic pass: one object must not carry two labels (deer as mob + hide, or two mob boxes).
    # Best-supported box wins. A box mostly inside another of similar size counts as the same object;
    # red is exempt from the containment rule because a telegraph legitimately surrounds creatures.
    live = sorted((r for r in records if r["keep"]), key=lambda r: -r["score"])
    won: list[dict] = []
    for r in live:
        for k in won:
            kin = FAMILY.get(k["class"]) is not None and FAMILY.get(k["class"]) == FAMILY.get(r["class"])
            same = iou(k["box"], r["box"]) > (0.2 if kin else 0.55)
            if not same and "red" not in (k["class"], r["class"]):
                small, big = sorted((k["box"], r["box"]), key=area)
                same = contained(small, big) >= (0.4 if kin else 0.8) and area(small) >= 0.3 * area(big) * (1 if kin else 4 / 3)
            if same:
                r["keep"] = False
                r["decision"] = "drop_overlap"
                r["note"] = (r["note"] + f" overlaps {k['class']}").strip()
                break
        else:
            won.append(r)

    # a red ring directly under one creature is its selection ring, not an AoE telegraph
    creatures = [r for r in won if r["class"] in ("mob", "player", "hide")]
    for r in won:
        if r["class"] == "red":
            for m in creatures:
                if area(r["box"]) <= args.ring_ratio * area(m["box"]) and (iou(r["box"], m["box"]) >= 0.1 or contained(r["box"], m["box"]) >= 0.3):
                    r["keep"] = False
                    r["decision"] = "drop_selection_ring"
                    break
    records.sort(key=lambda r: (not r["keep"], r["box"][1], r["box"][0]))
    return records


# --------------------------------------------------------------------------- output


def yolo_line(cls: str, box, size) -> str:
    w, h = size
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    return f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {(box[2] - box[0]) / w:.6f} {(box[3] - box[1]) / h:.6f}"


COLORS = {  # BGR
    "player": (255, 160, 0), "mob": (0, 0, 255), "hide": (0, 140, 255), "fiber": (0, 200, 0), "fiber_depleted": (160, 160, 0),
    "red": (0, 0, 160), "silver": (0, 255, 255),
}


def draw_preview(img: np.ndarray, records: list[dict], max_w: int = 1600) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    for b in hud_boxes((img.shape[1], img.shape[0])):
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (90, 90, 90), -1)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    for r in records:
        if not r["keep"]:
            continue
        b = [int(v) for v in r["box"]]
        col = COLORS[r["class"]]
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), col, 3)
        label = r["class"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        y0 = max(th + 6, b[1])
        cv2.rectangle(out, (b[0], y0 - th - 8), (b[0] + tw + 6, y0), col, -1)
        cv2.putText(out, label, (b[0] + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    s = min(1.0, max_w / out.shape[1])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--tiles", type=lambda v: [int(x) for x in v.split(",")], default=[840, 560],
                    help="comma-separated tile sizes in px; one VLM pass per size (default 840,560)")
    ap.add_argument("--overlap", type=int, default=100, help="minimum overlap between tiles in px")
    ap.add_argument("--pad", type=int, default=30, help="crop padding for verification calls")
    ap.add_argument("--workers", type=int, default=10, help="concurrent VLM requests")
    ap.add_argument("--frame-workers", type=int, default=3, help="frames processed concurrently")
    ap.add_argument("--no-vlm-tiles", action="store_true", help="skip proposals B (detector + crop verification only)")
    ap.add_argument("--trust-a", type=float, default=0.75,
                    help="accept detector-only boxes of stable classes (player, ore, ...) at/above this confidence without a crop call; >1 disables")
    ap.add_argument("--ring-ratio", type=float, default=3.0,
                    help="a red box this many times smaller than an overlapping creature box is treated as its selection ring and dropped")
    ap.add_argument("--no-refs", action="store_true", help="don't send the reference sheet with crop calls")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames")
    ap.add_argument("--crop-effort", default="low", choices=["minimal", "low", "medium"], help="VLM reasoning effort for crop calls")
    ap.add_argument("--no-refine", action="store_true", help="don't replace loose B-only boxes by the VLM's tightened box")
    ap.add_argument("--debug-crops", action="store_true", help="save every verification crop to <out_dir>/crops")
    args = ap.parse_args()

    frames = sorted(p for p in args.frames_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit(f"no images in {args.frames_dir}")
    out = args.out_dir
    for sub in ("images", "labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    vlm = Vlm(args.cache, workers=args.workers)
    args.ref_sheet = None if args.no_refs else build_ref_sheet()
    if args.ref_sheet:
        (args.cache / "ref_sheet.jpg").write_bytes(args.ref_sheet)
    detector = OldDetector(args.cache)
    t0 = time.time()

    # stage 1: detector, serial (GPU), cached. Images are not kept in memory.
    loaded = []
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable {p.name}")
            continue
        key = hashlib.sha256(p.read_bytes()).hexdigest()[:24]
        loaded.append((p, key, detector.detect(img, key)))
        if (i + 1) % 100 == 0:
            print(f"[det] {i + 1}/{len(frames)}", flush=True)
    t_det = time.time() - t0

    # stage 2+3: VLM tiles + crop verification, frames in parallel; each worker writes its own outputs
    rows, totals, errors = [], {}, []
    lock = threading.Lock()
    done = [0]

    def work(item):
        p, key, det = item
        t = time.time()
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            recs = process_frame(p, img, key, det, vlm, args)
            kept = [r for r in recs if r["keep"]]
            shutil.copyfile(p, out / "images" / p.name)
            (out / "labels" / f"{p.stem}.txt").write_text(
                "".join(yolo_line(r["class"], r["box"], (w, h)) + "\n" for r in kept), encoding="utf-8")
            cv2.imwrite(str(out / "preview" / f"{p.stem}.jpg"), draw_preview(img, recs), [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as e:  # one bad frame shouldn't sink an overnight run; it gets no label file
            with lock:
                errors.append((p.name, repr(e)))
            print(f"ERROR {p.name}: {e!r}", flush=True)
            return
        counts = {}
        for r in kept:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        with lock:
            for c, k in counts.items():
                totals[c] = totals.get(c, 0) + k
            for r in recs:
                rows.append([p.name, r["class"], r["source"], r["conf"], *[round(v, 1) for v in r["box"]],
                             r["decision"], r["note"]])
            done[0] += 1
            print(f"[{done[0]}/{len(loaded)}] {p.name}: {len(kept)} labels {counts} "
                  f"({sum(not r['keep'] for r in recs)} dropped, {time.time() - t:.0f}s)", flush=True)

    with ThreadPoolExecutor(args.frame_workers) as ex:
        list(ex.map(work, loaded))
    rows.sort(key=lambda r: (r[0], r[5] == "drop_hud", r[7], r[6]))

    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "class", "source", "conf", "x1", "y1", "x2", "y2", "decision", "note"])
        wr.writerows(rows)
    (out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")

    n = max(1, len(loaded))
    if errors:
        print(f"{len(errors)} frame(s) failed: {errors[:5]}")
    el = time.time() - t0
    print(f"\n{n} frames in {el:.0f}s ({el / n:.1f}s/frame; detector stage {t_det:.0f}s). Totals: {totals}")
    print(vlm.cost_line())
    total = {k: vlm.spent[k] + vlm.cached[k] for k in ("calls", "in", "out")}
    from labelkit import PRICE_IN, PRICE_OUT
    print(f"Full price of this run (cached or not): ${total['in'] * PRICE_IN + total['out'] * PRICE_OUT:.4f} "
          f"= ${(total['in'] * PRICE_IN + total['out'] * PRICE_OUT) / n:.4f}/frame, {total['calls'] / n:.1f} calls/frame")


if __name__ == "__main__":
    main()
