# xsql-aef-nlcd-deck.py: plan and state (2026-08-24, night)

**STATUS: sections 1-3 DONE and driven later the same night** (playwright, marimo run
--headless: ramp draws, highlight reverses it, legend labels swap, zero console errors;
committed). Kept for the record and for the TODO at the bottom.

## What was built (as left mid-work, before the drive)

`xsql-aef-nlcd-deck.py` has edits on top of `6012769` (the Photon search commit).
`marimo check` passed after the color edits; nothing since has been run in a browser.

### 1. Color by agreement (built, not driven)

- Constants cell: `AGREE_CMAP = "viridis"`, `RAMPS` (viridis + cividis, 32 stops each,
  matplotlib's tables embedded as hex strings; no matplotlib import), `ALPHA_RAMP = 225`.
  All three exported from the cell's return.
- Frame cell: `_RAMP` 256-entry LUT interpolated from the stops, `RAMP_HEX` (16
  swatches for the legend bar), `rgb_ramp` / `rgb_ramp_inv` per frame (unscored cells
  grey 128). `fill(paint, sel, hit, inv, ramp=False)`: on the agreement paint with
  `ramp` the color is the LUT on the agreement value (reversed when `inv`), alpha flat
  at `ALPHA_RAMP`; coverage scaling unchanged (size and color say the same thing).
- `legend_for(frame, paint, ramp=False, inv=False)`: prepends `{"ramp": RAMP_HEX,
  "cmap", "lo", "hi"}` when coloring by agreement; the bar is always cool -> warm
  left to right, the END LABELS swap on highlight (default lo = disagreement, hi =
  agreement; highlight flips them so warm = disagreement).
- Strip JS: a `color by agreement` toggle button after the highlight checkbox
  (dimmed to .5 opacity when the paint is not agreement), `acol` in every ctl send;
  `renderLegend` draws an item with `ramp` as a linear-gradient bar with its labels.
- Wiring: `HOLD["acol"]` (False), read from ctl in `_on_ctl`, passed to `fill` and
  `legend_for` in `_paint`.
- Why viridis: no red anywhere, the warm end is yellow, so neither direction of the
  flip lands on Stephen's weak leg; a blue-white-red cool/warm would. cividis is the
  swap-in via `AGREE_CMAP`.

### 2. Runner-up wording (partly applied)

Stephen: the "looks more like X" is the class whose PER-VIEW prototype the cell's
vector sits closest to, a suggestion relative to this scene, not a classification;
say "AEF suggests it could be this".

- DONE: pick panel -> `AlphaEarth suggests it could be <i>X</i> (relative to this
  view)`; selection panel word -> `AlphaEarth usually suggests`; the SQL cell's
  column -> `aef_usually_suggests`.
- NOT DONE (the sed was interrupted): the analyze table's header cell still reads
  `usually looks like` (two `<th>` occurrences near `_analyze_html`, ~line 2142);
  rename to `AlphaEarth usually suggests` with a title attribute saying it is relative
  to the scene.

### 3. To finish

1. Rename the `<th>` above.
2. Drive under `marimo run --headless --port 2808` with the scratch harness
   (`drive_acol.py` was written in this session's scratchpad but never ran: search
   to Folsom Lake, toggle color on, screenshot, tick highlight, screenshot, toggle
   off): confirm the ramp draws, the flip reverses it, the legend bar and labels
   follow, zero console errors.
3. Commit; add the CLAUDE.md bullets (color toggle + wording) under the deck
   notebook's section, and the memory note.

## TODO, Stephen's (not today)

- **res offset and variable zoom are not tidy.** The ladder (BASE_RES 6 / ZOOM0 6.2 /
  PER_RES 1.4, res 5-11, CELL_BUDGET coarsening), the "zoom in inside the served box
  never refolds" rule, and the strip's `res − / +` offset (resets only when the camera
  leaves the box) all work but do not read as one design; the mapterhorn explorer has
  the same open item ("the res-to-zoom ratio isn't right"). To be worked as a piece of
  its own: what the zoom should buy automatically, what the offset is for, and whether
  the reset should key on lon/lat/zoom rather than the footprint.
