# Navigation crops

400x360 minimap crops from user-provided 2240x1400 recordings on 2026-11-04.
Tests insert them at full-frame `[1840:2240,1040:1400]`; pixel values retain the
original PNG/JPEG-decoded colours. `s1_00179` and `s1_00316` are darkened frames;
`tabmap_open_2` is a map overlay with no trustworthy minimap fix.

Approximate arrow centres: `(235,145)`, `(235,145)`, `(236,145)`, `(225,125)`,
`(217,123)` for frame_minimap_1, s1_00039, s1_00079, s1_00179 and s1_00316.
The castle's blue patch is at about `(157,161)` and must not be selected.

`run15_edge19.json` is a local read-only snapshot of the run-15 collision record
(`source_index: 19`) and 18 observed fixes around x250–275/y120–140, taken from
the supplied runtime evidence after Vixen Tor cleanup. Its single fixture edge
has local index 0. Tests reproduce the road cut and retract only temporary copies;
they never read or change the live gitignored evidence.
