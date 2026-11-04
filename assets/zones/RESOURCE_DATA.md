# Zone resource observations

`resources.json` is an optional, versioned catalogue keyed by navigation zone id.
It is separate from walkability `zone.json`. Each zone contains independently
attributed `observations`; refreshes preserve other sources and zones. Runtime
reads this data locally and caches it for the process. Restart after refreshing.

Zone resource types mean **T2+**, per Tyler's clarification. Ambient T1 rough logs,
rough stone and scraps of hide remain as `t1_ambient: true` records, with their
tiers, source and date. They do not define a zone's resource types. The verified
Fractured Ground header covers T2+: hide/fiber/ore II–III and faded wood/stone
slots. Its 04:14 Rough Stone and 11:08 Rough Logs reports are retained separately.

Only `ingame_tab` and `user`/`user (...)` observations may authorize exclusions,
using `absent_t2plus` or `complete_t2plus`. Any T2+ positive from any source vetoes
an exclusion. Wiki observations have URLs and fetch dates and remain advisory;
third-party records must use `unverified:<name>`. Missing files/zones and zones
with only advisory/unknown data do not change farming. An unknown list is `null`,
not `[]`. No wiki tier/role is copied into a user observation with unknown tiers.

The current catalogue includes Fractured Ground, Vixen Tor, Bridgewatch,
Prospector's Hope and Steppe Cross. The four neighbour names come from the
[Fractured Ground wiki page](https://wiki.albiononline.com/wiki/Fractured_Ground),
not verified exit-direction mappings. Bridgewatch/Steppe Cross pages do not
establish harvestable lists, so their resources stay unknown. Vixen Tor's user
type statement and advisory wiki agree on wood/stone/ore and no fiber.

The filter intersects configured farm classes in scene grouping and again in
target selection. It does not enable classes or recognize tiers on world nodes.
Aliases `cotton`→`fiber`, `tree`/`log`→`wood` share type rules. The current model's
hide/fiber remain enabled in FG; configured wood/stone would be excluded as T2+
zone targets despite their ambient T1 occurrences.

## Refresh from a saved Tab capture

```powershell
python tools/zone/tab_resources.py --zone fractured_ground --image "..\datasets\albion_v2\tabmap\fg_04.png" --captured-at 2026-11-04 --dry-run
python tools/zone/tab_resources.py --zone fractured_ground --image "..\datasets\albion_v2\tabmap\fg_04.png" --captured-at 2026-11-04
```

`--out file.json` writes another catalogue. `--zone` is operator-verified; this
tool does not OCR the zone name. An invalid header cannot overwrite the catalogue.
`--captured-at` accepts an ISO date/time; otherwise the file modification time is
recorded explicitly as the date basis. Updates are atomic and send no game input.

Optional operator-run capture (open Tab manually first; use a unique filename):

```powershell
python tools/zone/tab_resources.py --zone fractured_ground --live --save-image captures/fg_tab_20261104.png --captured-at 2026-11-04 --config albot_v2.toml
```

This takes one screenshot, saves it and extracts the header. It does not press
Tab, focus/move the window, move the cursor, send keys/clicks or start the bot.
No live capture was used during development.

Templates/layout live in `assets/ui/tab_resources`, calibrated at 2240x1400 on
fg_04 and checked against held-out fg_01–03. Five icon matches establish the
header; label templates distinguish faded `---` slots from a tier range. Only
II–III range templates are currently bundled. A clear other tier caption keeps
the resource type with `tiers: null` and a warning; ambiguous captions reject the
update. Add a labelled caption PNG and its `caption_tiers` in layout.json to
extend tier reading. Other scales/layouts require new templates and fixtures.
`labelroads-3inbp/5` was not run or benchmarked: the template option passes the
saved header test without a model/key, GPU load or new dependencies.
