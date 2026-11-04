# The rewrite (branch `claude/clever-hamilton-nt7r40-rewrite`)

Written 2026-11-04 by Claude, working only from the original source. **None of this
has run against the game.** There was no Albion client, no Windows machine, no GPU
and no Roboflow key where it was written. What *has* run: 22 tests covering the
geometry, spell mapping, config loading, every behavior's decisions and timing
(with a fake clock), and an end-to-end test of the real worker threads and decision
loop against fake models. Replay mode was also run against generated images. The
first real session should be `python -m albot check`, then `run --dry-run`.

The light-fix branch (`claude/clever-hamilton-nt7r40`) keeps your structure and
fixes the bugs in place. This branch keeps the game logic (same behaviors, same
priority order, same timings and keys as defaults) and rebuilds everything around
it.

## Layout

| File | What it does | Replaces |
|---|---|---|
| `albot/config.py` | Every tunable value, loaded from `albot.toml` | constants spread across all files |
| `albot/vision.py` | Capture, models, turning model output into a `Scene` / `Status` | `objectFinder.py`, `window_info.albion_screenshot` |
| `albot/workers.py` | Background threads publishing the newest Scene and Status | `multiprocessing_class.py`, `globals.py` |
| `albot/state.py` | Immutable `Scene`, `Status`, and `Latest`, a newest-item holder | the image lists and their locks |
| `albot/geometry.py` | Obstacle checks and path planning | `obstacle_instance_detection.py` |
| `albot/spells.py` | Calibrated spell slots | the sort-and-pad in `create_spell_image` |
| `albot/behaviors.py` | One class per activity: plan, then run | `player_actions.py`, `weapon_spell_casting.py`, the if-chain in `main.py` |
| `albot/brain.py` | Decision loop, pausing, focus, errors | `main.py`, `helper.pause` |
| `albot/actuator.py` | Keyboard and mouse, with a focus check and dry run | `helper.clickOn`, scattered `kb.press_and_release` |
| `albot/debug_view.py` | Draws detections, obstacles and the current plan | nothing |
| `albot/app.py` | `run`, `check` and `replay` commands | nothing |

## What should play better

These are behavior changes. Each one rests on reasoning from the code, not on
observed play.

1. **Spell slots don't shift.** Same calibration as the light branch.
2. **Healing waits for health.** In the original, `danger()` counted low health as
   danger, so the heal wait ended at once. Same fix as the light branch.
3. **Blocked targets get a path, not a skip.** The original dropped any target whose
   straight line crossed an obstacle. Now an A* search over obstacle corners finds
   a way around, which handles thick obstacles and several in a row. The bot walks
   to the first waypoint, then re-plans from a fresh frame, because the camera
   follows the player and every screen position moves as it walks. Benchmark on 10
   obstacles with 80-point outlines: median 0.3 ms, worst 57 ms, with a 60 ms cap.
   `walk_speed_px_per_s` (how long to walk before re-planning) is a guess.
4. **The right player.** The original used the first `player` detection, which can
   be another player. Now it's the one nearest the usual spot (centre plus offset).
5. **Stuck detection.** Failing at the same screen spot 3 times in a row (clicking a
   node that never starts gathering, an item that stays put) skips that spot for
   15 seconds. The camera follows the player, so a target only stays at the same
   screen spot when the player isn't moving, which is what makes this signal mean
   "stuck". Fights are excluded, since a mob standing still while you shoot it is
   normal.
6. **Waits react to danger.** Pickups, wisps, detour walks and wandering all stop
   waiting when a threat appears. The original used plain `sleep` in several of these.
7. **Fresher spell and health data.** Status (spells + health) is read at 5 per
   second from one shared screenshot. The original took separate screenshots and
   ran spells about 2 per second and health about 1 per second. The loading-bar model
   now only runs while farming is waiting for it.
8. **No crash paths from mixed snapshots.** Each `Scene` is immutable, and a
   behavior's plan is built from one snapshot. While running, behaviors still read
   the newest data, as the original intended.

## What should be nicer to use

- **Hotkeys:** F8 pause/resume, F9 quit, right shift screenshot. No typing into the
  console.
- **Focus handling:** alt-tab pauses automatically and resumes 2 seconds after the game
  is focused again. Input is refused whenever the game isn't focused, even mid-action,
  so the bot can't type into another window.
- **Settings file:** keys, weapon rotation, thresholds, timings and behavior order are
  in `albot.toml`. A misspelled setting is an error, not silently ignored. The four
  original weapon rotations are in the example file.
- **`check`** loads the models, finds the window, runs three scans and reports what
  was seen and how long it took.
- **`run --dry-run`** decides and logs everything without sending input.
- **`replay <folder>`** runs the models and decisions on saved screenshots and writes
  annotated copies. Your existing training screenshots become a test set for both
  the models and the logic.
- **Debug view** (`debug.view = true`): a live window with boxes, obstacles, the
  player and the planned path. Put it on a second monitor: if it takes focus, the bot
  pauses.
- **Logging** with timestamps, to the console and `albot.log`. A status line every 15
  seconds shows scan rates, timings, the spell bar and action counts. A summary
  prints on exit.
- **Errors don't stop it.** A failed frame is logged once and retried. A failed tick
  is logged and the loop carries on.

## Unverified assumptions to check first

- **Colour order with `mss`.** mss frames go to the models as numpy arrays in BGR
  order, which the `inference` package should treat as OpenCV images. If detections
  look worse than before, set `capture_backend = "pyautogui"` (what the original used)
  and compare with `check` or `replay`.
- **Health model output.** Read as `predictions[0].class_name == "low"`, as the
  original did.
- **Move button.** Wander and detours walk with the right mouse button, as the
  original's wander did.
- **Speed.** Scan rate is mostly model time on the GPU, which this doesn't change.
  Capture with mss should be faster than pyautogui, but that's unmeasured here.

## Left out on purpose

Nothing here tries to hide the bot from anti-cheat (no humanised mouse paths,
randomised timings or similar). That wasn't in the original either.
