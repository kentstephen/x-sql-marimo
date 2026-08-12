# xsql-canopy-deforest: deforestation 2002-2022 x canopy height today

**STATUS: PARKED**, in `archive/xsql-canopy-deforest.py`. The machinery works (both
headless runs below), but as built it only COMPARES the two rasters: a toggle between
two ramps leaves the synthesis to the viewer and answers no question. The way back, if
it comes back, is recorded from the parking discussion: cross the layers per cell into
FOUR STATES (cleared-and-still-bare, cleared-and-regrown, intact-and-tall,
low-loss-low-canopy) drawn as one categorical map, and repoint the drawn-box ranking
at "share of cleared ground still bare" (a permanence scorecard) rather than mean
loss. Quotability of any recovery number is gated on reading the per-tile acquisition
dates (`tiles.geojson`): the CHM is 2018-2020 and the loss layer runs to 2022, so
"cleared but tall" can be stale imagery. Known cosmetic gap at parking: toggling the
`colour` dropdown below zoom 13 correctly keeps the deforestation hexes but only
explains itself in the status line after the next draw, so it can read as a dead
control. Canopy ramp is matplotlib Greens (pale -> deep green with height), Stephen's
own pick; his colour issue is red (protan-type) and mono-green ramps read normally,
recorded in CLAUDE.md.

A whole-file fork of `xsql-deforest-divisions.py` that adds Meta & WRI's High
Resolution Canopy Height Maps as a second per-cell quantity at deep zoom, plus the HFP
notebook's measured-viewport ruler (the parent still carries the VIEW_W/VIEW_H guess
and its fullscreen defect). The CHM reader is the one built for the parked fire-risk
fork, carried over by the shared-by-copy rule; the dataset recon (BigTIFF one-row
strips, predictor 2, quadkey grid, the unusable msk sidecars, strip-size measurements)
is recorded in `docs/canopy-firerisk-notes.md` and not repeated here.

## Why this pairing and not fire risk

Canopy height added nothing to a hazard model whose fuel inputs already contain canopy
structure, and it misleads in grass country. Here it is the subject, not a proxy: the
deforestation layer says what share of ground was CLEARED 2002-2022, the CHM says what
STANDS THERE NOW. Cleared-and-regrown, cleared-and-gone and intact-and-tall become
three distinguishable statements, and zero canopy over a high-loss cell is a real
measurement, not a risk claim.

## The two ladders, and the parent join that bridges them

The datasets' resolutions are two orders of magnitude apart, so they get different
ladders on the same zoom formula:

- **Deforestation** caps at res 8 (its 100 m pixels give ~18 per cell at L1); that cap
  is the parent notebook's and is untouched.
- **Canopy** continues where that stops: res 10 at z13 (~660 strided px/cell), res 11
  at z14.4 (~94), res 12 at z15.2 with the pixel stride relaxed from 4 to 2 (~54).
  Res 13 is possible (~30 unstrided px) but res 12 hexagons are already a few px on
  screen when they arrive.

The bridge is `h3_cell_to_parent` in DuckDB (Uber's C, same reasoning as the
polyfill): every fine canopy cell LEFT-joins the deforested share of the res-8 cell it
sits in, so a canopy hexagon's tooltip reads "12.4 m standing, on ground 8% cleared".
LEFT matters twice: the deforestation fold drops zero cells, so a missing parent row
means "nothing measured", surfaced as null, never 0%.

## Interaction model

A `colour` select (Unicode trait, per the proven-trait-types rule) switches the
hexagons between the two quantities above zoom 13; below it the switch shows a status
hint and the hexagons stay on deforestation. `HOLD["mode"]` tracks what the cells
layer currently shows, both switch directions are answered from kept layer tables in
`_instant` (no read), and the divisions choropleth plus the drawn-box ranking stay
deforestation-based in either mode: the deforestation fold always runs, canopy only
swaps the hexagon paint.

The canopy ramp is linear 0-25 m on matplotlib Greens (pale for bare, deep green for
tall) with ZERO INSIDE THE RAMP (the HFP lesson: zero canopy is the bottom of a
continuum, and it is exactly what a fresh clearcut looks like). Hue names the dataset
(yellow-topped cividis = cleared share, green = standing canopy), luminance names the
value in both. There is no no-data swatch: an absent CHM tile folds no cells, so the
hexagon is missing rather than grey.

## Measured

- Rondônia at z13.6, 1400x620 seed canvas: canopy res 10, 3,688 cells, 36 MB of strip
  span, on top of the parent's unchanged deforestation fold. Dense-canopy strips run
  ~16 KB compressed; the archive-wide average is 320 B, so most places are far
  cheaper. `CANOPY_BUDGET` (160 MB) refuses beyond that with a status note.
- The world view is byte-identical in behaviour to the parent (38,517 cells, res 4).

## Known limits

- The canopy memo is coverage-based on the padded viewport box, so a pan that leaves
  the box refetches; there is no chunk-grid sharing like the COG side, because the CHM
  has no tiles worth sharing (1-row strips).
- Canopy vintage is per Maxar acquisition (2018-2020 mostly): a fresh clearcut can
  still wear last year's trees. The acquisition-date layer exists
  (`CHM_acquisition_date.tif`, `tiles.geojson`) and is not yet read; it is the first
  thing to add if the pairing is ever used to claim change.
- A zoomed-out canopy view needs the missing pyramid built offline (Zarr/Icechunk
  repack, ~1.2 TB global); recorded as an idea, not planned.
