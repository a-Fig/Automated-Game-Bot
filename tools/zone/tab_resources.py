"""Extract a saved Tab header, or an explicitly requested live screenshot. No input."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from albot.config import load
from albot.tab_resources import extract_tab_resources, update_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--live", action="store_true", help="one screenshot only; operator opens Tab first")
    parser.add_argument("--save-image", type=Path, help="required with --live for reproducible provenance")
    parser.add_argument("--config")
    parser.add_argument("--zone", required=True, help="operator-verified zone id; this tool does not OCR its name")
    parser.add_argument("--captured-at", help="ISO capture date/time; saved-file mtime when omitted")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2]/"assets/zones/resources.json")
    parser.add_argument("--dry-run", action="store_true", help="print observation without updating catalogue")
    args = parser.parse_args(argv)
    try:
        if args.live:
            if args.save_image is None:
                raise ValueError("--live requires --save-image; no capture taken")
            from albot.vision import Capture
            from albot.window import GameWindow
            cfg = load(args.config)
            window = GameWindow(cfg.window_title)
            window.wait_for()
            frame = Capture(cfg.perception.capture_backend).grab(window.rect())
            from albot.navigation import as_bgr
            frame = as_bgr(frame)
            args.save_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_image), frame):
                raise OSError("could not save capture")
            capture = args.save_image
        else:
            capture = args.image
            frame = cv2.imread(str(capture))
            if frame is None:
                raise ValueError(f"cannot read {capture}")
        date = args.captured_at or datetime.fromtimestamp(capture.stat().st_mtime).isoformat(timespec="seconds")
        datetime.fromisoformat(date)  # reject an invalid provenance date before writing
        observation = extract_tab_resources(frame, capture.as_posix(), date)
        observation["capture_date_basis"] = "operator" if args.captured_at else "file_mtime"
        if not args.dry_run:
            update_catalog(args.out, args.zone, observation)
        print(json.dumps(observation, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
