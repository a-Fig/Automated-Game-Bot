"""
Class-count stats and contact sheets for an autolabel.py output folder.

    albot-venv\\Scripts\\python.exe tools\\dataset\\stats.py <out_dir> [--sheets 1] [--per-sheet 24] [--seed 0]

Prints per-class box counts, frames per class, empty frames and decision counts from review.csv, and writes
<out_dir>/sheet_*.jpg: random previews tiled in a grid, at most 1600 px wide.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labelkit import CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--sheets", type=int, default=1)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir

    labels = sorted((out / "labels").glob("*.txt"))
    boxes, frames_with, per_frame = Counter(), Counter(), []
    for f in labels:
        ids = [int(ln.split()[0]) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        per_frame.append(len(ids))
        for c in ids:
            boxes[CLASSES[c]] += 1
        for c in set(ids):
            frames_with[CLASSES[c]] += 1
    n = len(labels)
    print(f"{n} labelled frames, {sum(boxes.values())} boxes, {sum(1 for k in per_frame if k == 0)} frames with no labels, "
          f"mean {np.mean(per_frame):.1f} boxes/frame")
    print(f"{'class':16s} {'boxes':>6s} {'frames':>7s}")
    for c in CLASSES:
        print(f"{c:16s} {boxes[c]:6d} {frames_with[c]:7d}")
    csv_path = out / "review.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            dec = Counter((r["decision"]) for r in csv.DictReader(f))
        print("decisions:", dict(dec.most_common()))

    prev = sorted((out / "preview").glob("*.jpg"))
    rng = random.Random(args.seed)
    rng.shuffle(prev)
    cols = args.cols
    for s in range(args.sheets):
        pick = prev[s * args.per_sheet:(s + 1) * args.per_sheet]
        if not pick:
            break
        cw = 1600 // cols
        ch = int(cw * 1400 / 2240)
        rows = []
        for i in range(0, len(pick), cols):
            row = []
            for p in pick[i:i + cols]:
                im = cv2.resize(cv2.imread(str(p)), (cw, ch), interpolation=cv2.INTER_AREA)
                cv2.putText(im, p.stem, (4, ch - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(im, p.stem, (4, ch - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                row.append(im)
            while len(row) < cols:
                row.append(np.zeros((ch, cw, 3), np.uint8))
            rows.append(np.hstack(row))
        path = out / f"sheet_{s + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", path)


if __name__ == "__main__":
    main()
