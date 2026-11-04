"""
Upload an autolabel.py output folder to a Roboflow object-detection project.

    uv run --no-project --python 3.12 --with roboflow --with opencv-python python tools/dataset/upload_roboflow.py <out_dir> --project <name> [--yes]

Default is a DRY RUN: it prints the plan (project, splits, per-class counts) and uploads nothing.
Pass --yes to create the project (if missing) and upload. It never generates a version or trains.

Splits are 80/15/5 (train/valid/test) by frame GROUP so that frames from one recording session never
straddle splits. A frame's group is its file stem without the trailing frame index:
    sessionA_0001, sessionA_0002 -> group "sessionA";  rec_20261104_0042 -> "rec_20261104";
    obs_full (no index) -> its own group.
Override with --group-regex (one capture group = the group name).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_GROUP_RE = r"^(.*?)(?:[_-]f?\d+)(?:_full)?$"
SPLITS = {"train": 0.80, "valid": 0.15, "test": 0.05}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def user_env(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:  # Windows user environment, for shells that predate the variable
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0] or None
    except Exception:
        return None


def read_names(out_dir: Path) -> list[str]:
    import yaml  # ships with roboflow; fall back to a tiny parser if absent
    text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    try:
        names = yaml.safe_load(text)["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    except Exception:
        return [m.group(2) for m in re.finditer(r"^\s+(\d+):\s*(\S+)\s*$", text, re.M)]


def group_of(stem: str, rx: re.Pattern, block: int = 0) -> str:
    """Group name for a frame. With block > 0 a long single recording (s1_00000 ... s1_02044) is cut into
    runs of `block` consecutive frames, so neighbouring near-duplicate frames still share a split."""
    m = rx.match(stem)
    g = m.group(1) if m and m.group(1) else stem
    if block > 0:
        n = re.search(r"[_-]f?(\d+)(?:_full)?$", stem)
        if n:
            g = f"{g}_blk{int(n.group(1)) // block:04d}"
    return g


def assign_splits(groups: dict[str, list[str]], seed: int) -> dict[str, str]:
    """Greedy: shuffle groups, then give each to the split furthest below its image-count target."""
    total = sum(len(v) for v in groups.values())
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -len(groups[g]))  # big groups first so small ones can even things out
    have = Counter()
    out = {}
    for g in names:
        deficit = {s: SPLITS[s] * total - have[s] for s in SPLITS}
        s = max(deficit, key=lambda k: (deficit[k], SPLITS[k]))
        out[g] = s
        have[s] += len(groups[g])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="autolabel.py output folder (images/, labels/, data.yaml)")
    ap.add_argument("--project", required=True, help="Roboflow project name (created if missing)")
    ap.add_argument("--workspace", default="albion-edu-workspace")
    ap.add_argument("--key-env", default="ROBOFLOW_KEY2", help="environment variable holding the API key for the workspace")
    ap.add_argument("--license", default="CC BY 4.0", help="project license (only used when creating)")
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_RE)
    ap.add_argument("--block", type=int, default=0,
                    help="split a long recording into groups of this many consecutive frames (e.g. 40); needed when all frames share one session prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-name", default=None, help="Roboflow upload batch name (default: the out_dir name)")
    ap.add_argument("--include-empty", action="store_true", help="also upload frames with no labels (as negatives)")
    ap.add_argument("--yes", action="store_true", help="really create/upload; without it this is a dry run")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry run (this is the default)")
    args = ap.parse_args()
    dry = args.dry_run or not args.yes

    out = args.out_dir
    names = read_names(out)
    rx = re.compile(args.group_regex)
    frames = []
    for p in sorted((out / "images").iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lab = out / "labels" / f"{p.stem}.txt"
        rows = [ln.split() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()] if lab.exists() else []
        frames.append({"image": p, "label": lab, "rows": rows, "group": group_of(p.stem, rx, args.block)})
    if not frames:
        sys.exit(f"no images in {out / 'images'}")
    skipped = [f for f in frames if not f["rows"] and not args.include_empty]
    frames = [f for f in frames if f["rows"] or args.include_empty]

    groups: dict[str, list[str]] = defaultdict(list)
    for f in frames:
        groups[f["group"]].append(f["image"].name)
    split_of_group = assign_splits(groups, args.seed)
    for f in frames:
        f["split"] = split_of_group[f["group"]]

    slug = re.sub(r"[^a-z0-9_-]+", "-", args.project.lower()).strip("-")
    print(f"{'DRY RUN - nothing will be uploaded. ' if dry else ''}workspace={args.workspace} project={args.project} (slug {slug}) key=${args.key_env}")
    print(f"classes ({len(names)}): " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
    print(f"{len(frames)} frames in {len(groups)} group(s)" + (f"; skipping {len(skipped)} frame(s) with no labels (use --include-empty)" if skipped else ""))
    for s in SPLITS:
        fs = [f for f in frames if f["split"] == s]
        c = Counter(names[int(r[0])] for f in fs for r in f["rows"])
        print(f"  {s:5s} {len(fs):4d} frames ({len(fs) / len(frames):5.1%}, target {SPLITS[s]:.0%}), {sum(c.values())} boxes  {dict(c)}")
    for g in sorted(groups):
        print(f"  group {g!r}: {len(groups[g])} frame(s) -> {split_of_group[g]}")
    if len(groups) < 3:
        print("  note: fewer than 3 groups, so some splits will be empty; that's expected for tiny test runs")

    key = user_env(args.key_env)
    if not key:
        print(f"{args.key_env} is not set; cannot check or create the project.")
        return 0 if dry else 1
    try:
        import roboflow
    except ImportError:
        print("roboflow isn't installed here (use the uv command at the top of this file).")
        return 0 if dry else 1

    ws = roboflow.Roboflow(api_key=key).workspace(args.workspace)
    exists = any(pid.rsplit("/", 1)[-1] == slug for pid in ws.projects())
    if dry:
        print(f"project '{slug}' {'already exists, images would be added to it' if exists else 'does not exist and would be created (object-detection)'}")
        print("re-run with --yes to do it. No version will be generated and nothing will be trained.")
        return 0

    if exists:
        project = ws.project(slug)
    else:
        project = ws.create_project(project_name=args.project, project_type="object-detection",
                                    project_license=args.license, annotation="albion-game-objects")
        print(f"created project {project.id}")
    labelmap = {i: n for i, n in enumerate(names)}
    batch = args.batch_name or out.resolve().name
    ok, failed = 0, []
    for i, f in enumerate(frames, 1):
        try:
            project.single_upload(
                image_path=str(f["image"]),
                annotation_path=str(f["label"]) if f["rows"] else None,
                annotation_labelmap=labelmap,
                split=f["split"], batch_name=batch, tag_names=[f"group:{f['group']}"], num_retry_uploads=3)
            ok += 1
        except Exception as e:  # report and continue; one bad frame shouldn't stop the batch
            failed.append((f["image"].name, str(e)[:200]))
        if i % 10 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} uploaded ({len(failed)} failed)")
    print(f"done: {ok} uploaded, {len(failed)} failed. Project: https://app.roboflow.com/{args.workspace}/{slug}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
    print("No version was generated and nothing was trained.")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
