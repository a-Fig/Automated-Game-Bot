# Fractured Ground: live minimap coordinates

Every point uses the live 400x360 minimap crop, window rectangle
`[1840,1040,2240,1400]` at 2240x1400. The original map was roughly eight pixels too
high in Y. This revision registers Tab terrain to the minimap visible in the same
captures and independently checks the Tab-closed player fixes. Navigation
calibration, visited.json and runtime arrow fixes remain unchanged.

- `map.png`: re-registered Tab image. Moving player icons are removed before
  combining captures; terrain behind them is inpainted, not known.
- `mask.png`: binary 400x360 walkability. Both flat ground levels and neutral/warm
  grey basin patches are open. Level boundaries, very dark patches, blue-hued
  water/structure candidates and the exterior rim are provisionally blocked.
  Roads form corridors through boundaries (ramps). This remains a visual hypothesis
  requiring live validation, not ground-truth collision geometry.
- `mask_hires.png`: 1600x1440 planning grid, four cells per map pixel. The scaled
  grid API converts all points, distances, clearance, blocks and waypoints to/from
  the original minimap coordinates. mask.png remains a 400x360 summary. The fine
  grid preserves passages around small grey rocks that disappear when downsampled.
- `road.png`: road/ramp corridor, for debugging and route checks.
- `ramp_overlay.png` / `ramp_validation.json`: the local bend with all eight saved
  run-8 edges retained (magenta), the road (green) and a verified mask route (white).
- `mask_overlay.png`: red = blocked, green = road/ramp, cyan = independent live
  fixes, white = NE route, magenta = portal approaches. Display coordinates are x4.
- `transform.json`: feature statistics, independent arrow errors, live/asset drift
  checks, parameters, observed-fix audit and route check. Fixes never adjust registration.
- `observation_validation.json`: classification audit before/after, every effective
  fallback disc and its opened-cell count, and ordered-walk connectivity results.
- `walk_ne_fixes.json`: 536 hand-walk fixes in walk order; visited.json supplies
  723 other standing positions. mask_before_basin.png preserves the previous
  planning mask for a reproducible audit. These files are not used at runtime.
- `zone.json`: corrected portal positions, locator polygon and Tab transform.
  Its clearance default is 1.0; an explicit navigation.clearance overrides it.
- `override.png`: neutral grey. The old override mixed misregistered road colours
  with live visited discs and could reopen cliffs. Keep visited.json as evidence,
  not an instruction to paint every nearby cell open. Override support remains:
  0 = blocked, 255 = open, 128 = retain the generated mask. Disable antialiasing.

Rebuild from the original captures (reads images only, never game input):

```powershell
python tools/zone/tabmap_mask.py --captures "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_01.png" "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_02.png" "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_03.png" "C:\Users\smash\PycharmProjects\datasets\albion_v2\tabmap\fg_04.png" --out-dir assets/zones/fractured_ground --check-learned tests/fixtures/navigation/learned_after_run8.json
```

Each adjacent JSON supplies minimap_before, minimap_after and window. Use --fixes
to name them separately. The zone metadata loads both observed fix sets and their
baseline automatically. To reproduce registration from the old asset, add
`--reference tests/fixtures/tabmap/reference_old.png` while keeping current zone
metadata, which includes the observation sources. Already-live portal coordinates
are preserved on rebuilds. --observed-fixes accepts alternative fix files;
--ordered-walks names which of them require consecutive-pair connectivity.

Thresholds and widths are CLI parameters: --level-value 215, --ground-value 165,
--boundary-width 0.25, --dark-rock-value 120, --rock-max-area 0,
--water-saturation 75, --road-saturation 180, --road-value 230, --road-gap 4,
--road-half-width 1.5, --rim-margin 15 and --scale 4. Neutral-grey blocking is off;
--block-neutral-grey opts in. Widths use map pixels, areas map pixels squared.
Road width includes planning clearance. Lower thresholds or wider roads can reopen
cliffs: inspect the overlay before live use. Poor feature fits, independent arrow
errors over 1.2px and asset/live drift over 1px are rejected. Drift means median
terrain displacement and drift at the matched terrain centre, not a similarity's
extrapolated translation at the off-map origin.

The road/ramp is a hypothesis, not a proven crossing at (241,142). With all eight
run-8 edges, the checked route from the gully heads south around the cliff near
(240,146.5), follows the bend near (247.5,143.25), then goes NE. Live validation is
still needed. Clearance 2.0 over-inflates fine
rocks and disconnects it; remove a legacy explicit clearance=2.0 to inherit 1.0.
Zones without their own clearance still default to 2.0.

## Basin ground truth

Runtime travel learning is separate: gitignored `learned.json` holds new standing
discs and suspected stall edges. Zone.load applies it without modifying these
generated assets. See docs/NAVIGATION.md for persistence, reset and log replay.

The previous classifier rejected 267/536 hand-walk samples and 374/723 visited
samples. Its "water" rule confused warm/neutral grey with impassable water. Its
dark threshold and small-blob rule also rejected traversed floor, and a brightness
cutoff at 205 made ordinary brown-floor texture into false level boundaries.
The observed hues were H8-13, not blue water hues; these patches are traversable
at the recorded positions. The map alone cannot say whether they are shallow
water, flat rock or another kind of floor.

With grey blocking off, no size-only blob blocking, a very-dark cutoff of 120,
and the 215 level cutoff, all 536 hand-walk samples are safe before corrections.
The original basin correction used 18 native cells in five discs. After the four-
capture registration and road correction, two visited samples need edge corrections:
radius-1-map-px discs change only 2 native 4x cells (0.125 map-pixel squared) in one
effective disc. All 1,259 samples are then directly safe; all 535
consecutive hand-walk pairs connect. The report lists each disc centre and count,
and tests enforce the 2-cell count. No blanket route/visited override is restored.
The discs override the classifier's estimated edge where standing evidence
contradicts it. Their radius is capped at 1.0; --disc-radius 0 disables them and
fails validation if any observed position is unsafe. Unknown blue-hued regions
remain blocked, but their actual game meaning has not been established here.

## Run 8 road bend check

fg_04 is included in fitting and independent validation. Terrain features are now
detected at half Tab resolution rather than quarter, avoiding an unstable rotation
fit when the fourth capture is added. Independent arrow errors are 0.26-0.84 map
pixels; fg_04 is 0.48. Around (235-250,130-145), its six terrain matches have median
error 1.14 pixels, mostly Y. That local sampling uncertainty is reported rather
than calling the whole bend subpixel-accurate. Portal coordinates and live movement
calibration are unchanged.

The road rule now rejects lower-saturation floor pixels and dilates by 1.5 rather
than 2.8 map pixels. Crucially, the old floor-derived playable contour clipped the
orange road through dark ground; only broad dilation could connect around that
false cut. Roads now use the outer playable footprint instead, preserving the
narrow orange bend without widening away known cliffs. All eight run-8 edges stay
active in the saved route check. No edge is removed to manufacture connectivity.

Bushes, trees and mobs still require local avoidance. Unmarked ramps may be omitted.
Icon anchors have subpixel uncertainty; overrides remain useful for verified
corrections. The old low-resolution Zone.build_mask helper is disabled for these
assets: use full-resolution Tab captures with the generator.
