"""
Command line entry point.

  python -m albot run [--dry-run] [--config albot.toml]
      Play. F8 pauses and resumes, F9 quits, right shift saves a screenshot.
  python -m albot check
      Load the models, find the window, run one scan, and report what was seen.
  python -m albot replay <folder> [--out annotated]
      Run the models and the decision logic on saved screenshots, without the
      game, and write annotated copies showing what the bot would have done.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import config as config_mod

log = logging.getLogger("albot")


def setup_logging(cfg) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(cfg.debug.log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    if cfg.debug.log_file:
        f = logging.FileHandler(cfg.debug.log_file, encoding="utf-8")
        f.setFormatter(fmt)
        root.addHandler(f)


def report_providers() -> None:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        log.info("onnxruntime %s, device %s, providers %s", ort.__version__, ort.get_device(), providers)
        if "CUDAExecutionProvider" not in providers:
            log.warning("no CUDA provider: models will run on the CPU and scans will be slow")
    except Exception:
        log.warning("could not query onnxruntime")


def build(cfg, dry_run: bool):
    from .actuator import DryRunInput, Input
    from .brain import Brain, Context
    from .datacollect import Saver
    from .vision import Capture, Models, Perceiver
    from .window import GameWindow
    from .workers import Workers

    window = GameWindow(cfg.window_title)
    window.wait_for()
    log.info("loading models (first run downloads them)")
    models = Models(cfg)
    perceiver = Perceiver(cfg, models, Capture(cfg.perception.capture_backend), window)
    workers = Workers(perceiver, cfg)
    saver = Saver(cfg.data.folder, lambda: perceiver.grab()[0])
    input_ = DryRunInput() if dry_run else Input(window, cfg)
    ctx = Context(cfg, input_, workers, window, saver=saver)
    return Brain(ctx), workers, perceiver, saver


def cmd_run(cfg, args) -> int:
    dry_run = args.dry_run or cfg.dry_run
    report_providers()
    brain, workers, perceiver, saver = build(cfg, dry_run)

    import keyboard
    keyboard.add_hotkey(cfg.keys.pause, brain.toggle_pause)
    keyboard.add_hotkey(cfg.keys.quit, brain.quit.set)
    keyboard.add_hotkey(cfg.keys.screenshot, lambda: saver.save("manual_data"))

    workers.start()
    if cfg.debug.view:
        from .debug_view import LiveView
        LiveView(workers, brain.ctx).start()

    log.info("waiting for the first scans")
    while (workers.scenes.get() is None or workers.statuses.get() is None) and not brain.quit.is_set():
        time.sleep(0.2)
    status = workers.statuses.get()
    if status is not None and not status.spells_calibrated:
        log.warning("spell bar not calibrated yet: spells stay unused until a frame shows all "
                    "%d slots", len(cfg.keys.slots))

    log.info("running%s. %s pause/resume, %s quit, %s screenshot",
             " (DRY RUN: no input is sent)" if dry_run else "",
             cfg.keys.pause.upper(), cfg.keys.quit.upper(), cfg.keys.screenshot)

    import threading
    threading.Thread(target=_status_line, args=(brain, workers), daemon=True).start()
    started = time.monotonic()
    try:
        brain.run()
    except KeyboardInterrupt:
        pass
    finally:
        brain.quit.set()
        workers.stop()
        keyboard.unhook_all_hotkeys()
        minutes = (time.monotonic() - started) / 60
        log.info("stopped after %.1f minutes. %s | screenshots saved %d",
                 minutes, brain.summary(), saver.saved)
    return 0


def _status_line(brain, workers) -> None:
    while not brain.quit.wait(15):
        if brain.paused.is_set():
            continue
        scene, status = workers.scenes.get(), workers.statuses.get()
        if scene is None or status is None:
            continue
        log.info("scan %.1f/s (capture %.0fms, models %.0fms) status %.1f/s | spells %s | %s",
                 workers.scenes.rate(), scene.capture_ms, scene.inference_ms, workers.statuses.rate(),
                 "".join("+" if s == "up" else "-" for s in status.spells)
                 if status.spells_calibrated else "uncalibrated",
                 brain.summary())


def cmd_check(cfg, args) -> int:
    report_providers()
    from .vision import Capture, Models, Perceiver
    from .window import GameWindow
    window = GameWindow(cfg.window_title)
    window.wait_for()
    r = window.rect()
    log.info("window at %d,%d size %dx%d, focused: %s", r.left, r.top, r.width, r.height, window.is_active())
    models = Models(cfg)
    perceiver = Perceiver(cfg, models, Capture(cfg.perception.capture_backend), window)
    for i in range(3):
        scene = perceiver.scan_scene()
        status = perceiver.scan_status(want_loading_bar=True)
        log.info("scan %d: capture %.0fms, models %.0fms", i + 1, scene.capture_ms, scene.inference_ms)
    log.info("player %s at %.0f,%.0f", "detected" if scene.player_detected else "NOT detected (using offset)",
             scene.player.x, scene.player.y)
    log.info("mobs %d, threats %d, farmables %d, pickups %d, wisps %d, obstacles %d, all detections %d",
             len(scene.mobs), len(scene.threats), len(scene.farmables), len(scene.pickups),
             len(scene.wisps), len(scene.obstacles), len(scene.detections))
    log.info("spells %s (calibrated: %s), low health %s, loading bar %s",
             status.spells, status.spells_calibrated, status.low_health, status.loading_bar)
    return 0


def cmd_replay(cfg, args) -> int:
    """No game needed: models plus decisions on saved screenshots."""
    import numpy as np
    from PIL import Image

    from .actuator import DryRunInput
    from .brain import Brain, Context
    from .debug_view import annotate
    from .state import Latest
    from .vision import Models, Perceiver
    from .window import FixedWindow

    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        log.error("no images in %s", args.folder)
        return 1
    os.makedirs(args.out, exist_ok=True)
    models = Models(cfg)

    class Holder:  # stands in for Workers
        def __init__(self):
            import threading
            self.scenes, self.statuses = Latest(), Latest()
            self.want_loading_bar = threading.Event()

    holder = Holder()
    window = FixedWindow(1, 1)
    perceiver = Perceiver(cfg, models, None, window)
    ctx = Context(cfg, DryRunInput(), holder, window)
    brain = Brain(ctx)
    tally = {}
    for i, name in enumerate(files):
        frame = np.asarray(Image.open(os.path.join(args.folder, name)).convert("RGB"))[:, :, ::-1].copy()
        size = (frame.shape[1], frame.shape[0])
        scene = perceiver.scene_from(frame, size, timestamp=float(i + 1))
        status = perceiver.status_from(frame, float(i + 1), want_loading_bar=False)
        holder.scenes.publish(scene)
        holder.statuses.publish(status)
        choice = brain.choose(scene, status)
        plan = choice[1] if choice else None
        what = plan.description if plan else ("nothing detected" if scene.is_empty() else "idle")
        tally[plan.behavior if plan else what] = tally.get(plan.behavior if plan else what, 0) + 1
        header = f"{what} | spells {''.join('+' if s == 'up' else '-' for s in status.spells)}"
        if status.low_health:
            header += " | LOW HEALTH"
        Image.fromarray(annotate(frame, scene, plan, header)[:, :, ::-1]).save(os.path.join(args.out, name))
        log.info("%-40s %s", name, header)
    log.info("decisions: %s", ", ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])))
    log.info("annotated images written to %s", args.out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="albot", description="Albion Online bot")
    parser.add_argument("--config", default="albot.toml", help="config file (default albot.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play")
    run.add_argument("--dry-run", action="store_true", help="decide and log, but send no input")
    sub.add_parser("check", help="test models, window and capture")
    replay = sub.add_parser("replay", help="run decisions on saved screenshots")
    replay.add_argument("folder")
    replay.add_argument("--out", default="replay_out")
    nav = sub.add_parser("nav", help="preview map navigation without sending input")
    nav.add_argument("--to", required=True, help="portal:NE or minimap x,y")
    nav.add_argument("--dry-run", action="store_true", help="required: print proposals only")
    nav.add_argument("--image", help="saved frame instead of live capture")
    nav.add_argument("--zone", help="zone asset folder name")
    nav.add_argument("--calibration", help="JSON from nav-calibrate --samples")
    calibrate = sub.add_parser("nav-calibrate", help="measure or solve the map-to-screen transform")
    calibrate.add_argument("--probe", help="one screen displacement, e.g. 350,0")
    calibrate.add_argument("--execute", action="store_true", help="allow one live calibration click")
    calibrate.add_argument("--samples", nargs="+", help="solve saved probe JSON files, no input")
    calibrate.add_argument("--out", required=True, help="output sample or calibration JSON")
    args = parser.parse_args(argv)

    cfg = config_mod.load(args.config)
    setup_logging(cfg)
    if not os.path.exists(args.config):
        log.info("no %s found, using defaults (copy albot.example.toml to change them)", args.config)
    if args.command in ("nav", "nav-calibrate"):
        from .navigation_cli import cmd_calibrate, cmd_nav
        try:
            return (cmd_nav if args.command == "nav" else cmd_calibrate)(cfg, args)
        except (ValueError, OSError) as e:
            log.error("%s", e)
            return 1
    return {"run": cmd_run, "check": cmd_check, "replay": cmd_replay}[args.command](cfg, args)
