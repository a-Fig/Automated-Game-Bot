# Overnight report, 2026-11-04

Everything below is on branch `claude/albot-overnight` (not merged to main). Times are local.

## Bottom line

- **Character is fine.** No real deaths. One near-death at 01:38 (26/2351 HP, 1.1%) from a standalone
  navigation test walking into a bandit; that kind of walking is now inside the bot with combat first.
- **Silver spent: 0** (cap 50m). Gold not touched. Closed a Premium ad after maintenance with its X.
- **New scene model live:** `albiondetector/albion-v2/1` (YOLOv11s), trained on 725 auto-labelled frames
  with clean classes. With a stricter fiber threshold it fails about 15-22% of actions, against about 30%
  for the old model (44% before the threshold fix).
- **Map walking works partly.** The bot now plans from a walkability mask built from the Tab zone map,
  registered to the live minimap, and learns cliffs and walked ground across runs. It has walked from
  the oasis to the road below the ramp, but has not reached the NE portal (Bridgewatch) yet.

## How to run

```
# farming with the new model (local, gitignored config)
python -m albot --config albot_v2.toml run
# travel to the NE portal, then fall back to farming if it gives up
python -m albot --config travel_v2.toml run
# inspect or remove learned cliff edges
python -m albot nav-edges list --zone fractured_ground
python -m albot nav-edges retract --zone fractured_ground --near 238.57,143.05 --radius 1.5
```

`albot_v2.toml` = `albot.toml` with `models.scene = "albion-v2/1"`, `scene_key_env = "ROBOFLOW_KEY1"`,
`min_confidence = 0.55`, `class_confidence = { fiber = 0.7, hide = 0.6 }`, `pickup_classes = ["silver"]`,
`threat_classes = []`. `travel_v2.toml` adds `travel_goal = "portal:NE"` and the `[navigation]`
calibration (`calibrated = true`, `minimap_to_screen = [[373.34, 274.96], [-59.97, 341.67]]`,
`map_speed_px_per_s = 0.529`, `retreat_tolerance = 0.8`).

## Incidents

| Time | What | Outcome / fix |
|---|---|---|
| 00:37 | Watchdog "death" | False: the game darkened the screen. Health reading fixed (ae1cec3). |
| 01:38 | Near-death: standalone nav loop walked into a shielded bandit; nothing fought back | Fled with the boots dash; full HP, gear on. Travel moved into the bot with combat first (c69d934). |
| 02:01 | Daily server maintenance (10:00 UTC), client at login screen | Bot suspended itself; logged back in at 02:26. |
| 03:31 | Bot pressed Esc (player hidden behind a palm) and opened the game menu (Logout/Close) | Closed it. With the HUD readable the bot now steps out of cover instead (ce12f65). |
| 04:14 | Travel left-clicked a rock and gathered "Rough Stone" | Travel walks with right-click now (eacb3e3). |
| 05:44 | Farming bot wandered through Fractured Ground's NW exit into Vixen Tor (tier IV) and farmed there ~6 min; travel run 10 (05:50) then planned on the wrong zone's map | Stopped at 05:54, walked back by hand (ford, road, gate) and re-entered Fractured Ground at 07:02, HP full. Zone guard (label-change safety stop + exit no-go discs) requested from sol. |
| 05:4x | New model called meadow lavender/wildflowers "fiber", so farming in Vixen Tor kept failing with "no gathering bar" | Labelling prompt fixed and meadow look-alikes added to the reference sheet (b193ded); dataset v2 relabelled. |

## The image model

- Root problem: the old dataset mixed hide and mob, labelled whole-desktop screenshots, and had no
  "depleted" class, so the bot attacked nothing, missed things, and walked to gathered plants.
- New dataset `albiondetector/albion-v2` (the edu workspace is at its 10-project limit): classes
  player, mob, hide, fiber, fiber_depleted, silver. 725 frames, 2,681 boxes, labelled by
  `tools/dataset/autolabel.py` (old detector + Meta `muse-spark-1.3-contributor` tiles + crop checks
  against reference crops in `tools/dataset/refs/`). Cost $4.57.
- Weakest calls: fiber vs fiber_depleted, mob vs hide. "red" was dropped (its labels were wrong).
- A second batch of 590 frames from a later session is labelling now for version 2.

## Navigation (with sol, the GPT-6.1 T3 thread)

- Minimap arrow locator plus a calibrated minimap-to-screen matrix (two long walks).
- `assets/zones/fractured_ground/map.png` was misregistered by about 8 px against the live minimap;
  re-registered from four Tab-map captures (0.3-0.8 px arrow error).
- Walkability from the Tab map: both ground levels open, cliff faces, rock blobs and blue water
  blocked, the orange road always open (ramps). Checked against 1,259 positions the character really
  stood on.
- Travel is a bot behaviour after shield/heal/fight_close: it walks with right-click once a second,
  retreats along walked breadcrumbs when hit by something undetected, quits below 25% HP after an
  8 s retreat, gives up after 30 s without a move and goes back to farming.
- Learning: walked positions and stall edges persist in `assets/zones/fractured_ground/learned.json`
  (gitignored). Road crossings need two stalls before they become an edge.
- Sol's backtracking (b7a0eee) is merged: zero-movement detection, cliff "fans", and up to two walks back along the trail per stuck spot. It worked once live in run 10, but that run was in the wrong zone (see Incidents).
- Best run (run 7): oasis to (238.6, 143.1) on the road below the ramp. Run 9 climbed onto a rock
  plateau at (239.4, 122.0) and could not get down; sol is adding zero-movement detection and
  backtracking along the trail for that case.

## Other fixes

- Health read by how far the red fill reaches (ae1cec3); gathering bar by its green pixels (86f7830).
- Farm walk timeout scales with distance (a109e61).
- Sideways minimap arrow accepted (4ab2c2f).

## Spend

- Silver: 0. Gold: 0.
- Meta API: about $8 (two labelling batches).
- Roboflow: one YOLOv11s training (fast) in `albiondetector`.
- Modal: $0.

## Left running / not restarted

- The farming bot, on the new model.
- Background helpers that hit the 2-hour limit and were not restarted: the gsrv death watchdog,
  the focus keeper, and the frame recorder. The bot's own safety (death limit 2 per 30 min, HUD checks)
  still applies.
- Power settings unchanged (AC sleep was already "never").
