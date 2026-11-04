# Roadsmapper reuse for albot

Read-only review on 2026-11-04 of `C:/Users/smash/PycharmProjects/roadsmapper2`
([a-Fig/roadsmapper](https://github.com/a-Fig/roadsmapper)), HEAD
`f8ae887552a20d574160129d802067bc7aaf1d35`. `roadsmodel.py` already had local edits;
those working files were inspected without importing/running or modifying them.

**Highest value: add expected-zone OCR alongside the fast zone guard.**
`roadsofavalon-1ueto/1` finds `zone_current`/`map_menu`; `labelroads-3inbp/5` finds
`zone_name` and map icons. Both load through `inference.get_model` with
`ROBOFLOW_KEY2`. Their labels locate a crop; OCR supplies its name. Reuse that
separation with albot's saved frames and small HUD-name crop. Start with local
Tesseract (configured executable path), normalize against known zone names, and
require repeated confident agreement. OCR can reject a wrong zone **at startup**,
which a reference pixel diff cannot identify. These models were written for map
UI, not proven on the tiny bottom-right label; validate on FG/VT crops first.
Keep loading detection, the cheap pixel-diff stop and the no-auto-resume latch.
Unreadable OCR must not authorize a zone switch or load another zone's evidence.

**Next: harvest exit labels/connections from operator-supplied Tab/tooltips.**
`roadsmodel.Roadsimage` crops `open_connection`/`ava_connection`, OCRs them, and
`googlevision.extract_connection_info` parses the destination after "exit to".
The portal parser also reads Roads expiry. Reuse crop→OCR→validated connection
records, then register a gate marker to albot map coordinates. A tooltip box is
not itself a safe gate waypoint; the current parser also discards compass
direction. Royal exits are stable; Roads connections expire. Neither graph
connectivity nor a tooltip establishes local traversability across cliffs.

The resource labeler counts `cotton`, `hide`, `log`, `ore`, `stone`, `water` and
`large_*` icons as arbitrary .01/1.0 increments. It does not read tier numerals or
implement the five-slot Tab-header layout. Map labels `cotton`/`log` map to
albot's `fiber`/`wood`, but domain/scale accuracy needs saved-frame evaluation.
The separate template refresh tool now reads this header; no labeler inference
claim is made. The operator's current `albion-v2/2` has **no ore/wood/stone classes**
(older model survey results are different). Adding an ore class is the larger
immediate world-perception gain: ore is FG's tertiary and Vixen Tor's secondary
resource in the advisory wiki records. Model work remains Claude/Tyler's task.

`ZoneImages.csv` has 406 references, including Bridgewatch, but **zero referenced
images exist** and `images/maps` is empty in this checkout. No map images were
reused. Restored Tab images could seed masks only after independent registration
to live minimap fixes; palette/road/elevation interpretation still needs ground
truth and learned cliffs. Roads names use a different punctuation/space sanitizer
from albot's underscore ids; use explicit mappings rather than dropping characters.

Port the pure pieces, not module imports: model initialization, window access,
Google client setup and environment mutation occur in the existing wrappers.
`Roadsimage(input_img=...)` still reads window location. Google OCR makes network
calls and may save random debug images; `localvision` includes a fixed Windows
Tesseract path and a separate HTTP OCR client. No OCR/model service was called,
no credentials read, no pinned-venv changes made, and no game input sent.
