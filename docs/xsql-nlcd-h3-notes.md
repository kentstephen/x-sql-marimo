# Annual NLCD in H3: notes

Notes for `xsql-nlcd-zoom.py`. The previous version of this file described a notebook that
did not work; that notebook has been replaced. What follows is what the rebuild established,
with the numbers it was established by.

Run: `uv run marimo edit xsql-nlcd-zoom.py`

## The shape of the thing

Nothing is read until the camera asks for it. Every fold reads **only the padded viewport**,
from the overview that matches the H3 resolution it is about to build, registers that window
with xarray-sql and folds it in SQL. The camera never re-runs a marimo cell: it schedules a
coroutine that swaps traits on one live layer.

The counter-intuitive part, and the reason res 11 is reachable at all: **the finest views are
the cheapest reads**, because the viewport shrinks faster than the resolution grows.

| res | overview | m/px | read px | read | fold | cells | px/hex |
|-----|----------|------|---------|------|------|-------|--------|
| 5 | L5 | 960 | 16,245,000 | 0.26s | 0.45s | 31,629 | 277 |
| 6 | L4 | 480 | 4,898,214 | 0.18s | 0.16s | 31,262 | 157 |
| 7 | L4 | 480 | 672,560 | 0.16s | 0.05s | 30,539 | 22.0 |
| 8 | L3 | 240 | 383,720 | 0.29s | 0.05s | 30,705 | 12.5 |
| 9 | L2 | 120 | 220,527 | 0.36s | 0.04s | 30,919 | 7.1 |
| 10 | L1 | 60 | 126,880 | 0.15s | 0.03s | 31,072 | 4.1 |
| 11 | L0 | 30 | 72,890 | 0.17s | 0.04s | 31,183 | 2.3 |

Python RSS stays flat (~0.68 GB) because one fixed table name means DataFusion holds exactly
one window at a time. `ctx.deregister_table("lc")` before each `from_dataset`, or it errors
with "table lc already exists".

**The res 5 and res 6 rows of that table are superseded.** See below: those two reads were
sized against a pyramid one level shorter than the one in the file.

## The pyramid has SEVEN levels, and the table above only knew about six

The `LEVEL_FOR_RES` comment described the source as "L0 30 m, L1 60, L2 120, L3 240, L4 480,
L5 960" and the mapping stopped at L5. Parsing the BigTIFF IFD chain of
`Annual_NLCD_LndCov_2024_CU_C1V1.tif` directly says otherwise:

| IFD | size | m/px |
|-----|------|------|
| 0 | 160000 x 105000 | 30 |
| 1-5 | ... | 60 / 120 / 240 / 480 / 960 |
| **6** | **2500 x 1640** | **1920** |

The chain terminates after IFD 6, so L6 is genuinely the last one. `_levels = [_g,
*_g.overviews]` already contained index 6; nothing ever selected it. `LEVEL_FOR_RES` is now
`{5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1, 11: 0}`, which is also just `11 - res` and drops the
odd doubled-up `{6: 4, 7: 4}`.

| res | was | now | px/hex | read px |
|-----|-----|-----|--------|---------|
| 5 | L5 | **L6** | 277 -> ~69 | 16.2M -> ~4.1M |
| 6 | L4 | **L5** | 157 -> ~39 | 4.9M -> ~1.2M |

Res 7 down are untouched. The whole of L6 is 4.1M uint8 pixels for the conterminous US,
about 4 MB, so the opening whole-country draw stops being the one read that costs anything.

Two things worth keeping:

- **The oversampling was visible in this file the whole time.** 277 px/hex sat in the table
  next to 2.3 px/hex at res 11, and the fine end being usable at 2.3 is the proof that 277
  is roughly a hundred times more than a modal class needs. The number was recorded and not
  read as a finding.
- **Coarse overviews of CATEGORICAL data are only safe if the pyramid is nearest or mode
  resampled.** An `average` over class codes blends 41 and 82 into 61, which is a legal-looking
  number and a meaningless one. Verified before the change rather than assumed: one 512x512
  tile decoded (deflate, predictor 1) from L6, L5 and L0, and every distinct value is a legal
  NLCD code, `{11,21,22,23,24,31,41,42,43,52,71,81,82,90,95}` plus 250 at L5. Any future
  dataset dropped into this pipeline needs the same check before its coarse levels are used.

The general lesson for other rasters: being a COG does not mean "read fewer pixels", it means
"read the right pyramid level and only the tiles under the viewport". The tile half was
already right, and `TILE = 512` matching the file's internal tile size is why there is no read
amplification. The level half was loose at the coarse end only.

**res 11 is the floor, not a cap.** A res 11 hexagon holds 2.3 pixels of 30 m NLCD; res 12
would hold 0.6 and the map would hole out. The ceiling belongs to the data.

## Sizing: tune by SCREEN PIXELS, never by cell count

This was the biggest self-inflicted wound. `PER_RES = 1.4` is correct (each H3 step is 2.65x
linear, `log2(2.65) = 1.4`), so hexagon size on screen stays constant across zooms. But
`BASE_RES` was tuned to give ~215k cells per fold and that was reported as a virtue. **215k
cells on a 1400x620 viewport is one hexagon per four pixels by definition.** Measured, at the
band start (the worst case, since a hexagon grows 2.65x before the next step):

| BASE_RES | band start | band end | cells |
|----------|-----------|----------|-------|
| 7 | 0.69px | 1.82px | 692,000 |
| 6 | 1.83px | 4.81px | 99,000 |
| **5** | **4.84px** | 12.68px | 14,000 |
| 4 | 12.77px | 33.47px | 2,000 |

At 0.7px you are not looking at hexagons, you are looking at aliasing. Settled on
`BASE_RES = 5`. `math.floor`, not `int()`: int truncates toward zero, which collapsed every
zoom below ZOOM0 onto the floor and jumped the map 4,626 -> 216,896 cells in one zoom step.

## Colour: use NLCD's own, and check the LUMINANCE of the COMMON classes

A palette was invented here on a teal-to-brown axis, with the three forest classes separated
by lightness. It made the map unreadable. Measured over a real southeast viewport:

| class | share of view | invented lum | NLCD lum |
|-------|--------------|--------------|----------|
| Deciduous forest | **39.7%** | 0.103 | 0.329 |
| Evergreen forest | 9.2% | 0.034 | 0.086 |
| Water | 2.1% | 0.032 | 0.143 |
| Pasture | 21.7% | 0.558 | 0.651 |

**52% of the map came out below 0.18 luminance.** The map read as black, with every lighter
patch apparently outlined in it. Lightness ended up inversely correlated with frequency: the
single most common class in the region got the darkest colour.

NLCD ships its own colormap **inside the COG**: `GeoTIFF.colormap.as_dict()`. Those values
are now written into `GROUPS` so the legend and the fill cannot drift. The lesson is not
"invented palettes are bad", it is: **weight a palette by how much of the map each class
actually covers before judging it.**

(NLCD's `23`/`24` developed are pure reds against forest greens, which is the one pairing a
deuteranope cannot use. They were 1.1% of that view. Left as NLCD ships them, deliberately.)

## Clusters: dissolve, then stroke the boundary

`h3ronpy.vector.cells_to_wkb_polygons(cells, link_cells=True)` does the dissolve **and** the
connected-component split in one call: neighbours merge, disconnected groups come back as
separate polygons. 13,036 contiguous cells -> 1 polygon in 0.01s.

Two encodings were tried and both failed for reasons worth recording:

1. **Stroking every rim cell.** At res 9, 75% of cells touch an unlike neighbour (50% at
   res 7), so "outline the boundary cells" outlines nearly the whole map. It came out a
   honeycomb. No threshold fixes it, because the fragments *are* the data.
2. **Filling the dissolved polygon in the class colour.** The polygon *is* those cells, so a
   fill in their own colour is invisible at any opacity, including 1.0. Self-cancelling by
   construction. This should have been obvious before it was built.

What works is **stroking the dissolved boundary**: a run of 40,000 cells becomes one line.

Dissolving everything is not viable; the speckle has to go first. Union-find over k-ring-1
adjacency gives run sizes in 0.49s, and then (res 8, 267,511 cells):

| min run | cells kept | polygons | WKB | dissolve |
|---------|-----------|----------|-----|----------|
| 1 | 267,511 (100%) | 43,329 | 13.2 MB | 22.5s |
| 20 | 174,036 (65%) | 1,153 | 5.2 MB | 1.9s |
| 100 | 136,192 (51%) | 181 | 3.6 MB | 1.3s |
| 500 | 104,493 (39%) | 34 | 2.5 MB | 1.0s |
| 2000 | 82,196 (31%) | 8 | 1.8 MB | 0.7s |

The polygon count collapses 1,200x while the cells covered only halve: almost every run is a
handful of cells.

The adjacency is SQL. `unnest(h3_ring1(hex))` joined back to the cell table, 0.04s at 216k
cells. The UDF must declare **`pa.large_list(pa.uint64())`**, not `pa.list_` -- h3ronpy
returns LargeList and DataFusion rejects the mismatch outright. `c.hex > r.hex` halves the
edge list losslessly (grid_disk emits every adjacency twice; verified 1,003,266 edges ->
501,633 unordered-unique, identical partitions either way).

## Things that cost a session each

**lonboard latches `_rows_per_chunk` in `__init__` and never recomputes it.**
`layer/_base.py:397`. Every later assignment still rechunks through it
(`traits/_table.py:106`, `_h3.py:130`, `_color.py:140`) and `serialize_table_to_parquet`
writes **one Parquet file per chunk**. Build a layer against a 1-row placeholder table and
`infer_rows_per_chunk` returns 1, latched for the layer's life, so every fold serialises one
complete Parquet file **per hexagon**. Measured over four folds of a zoom-in: **673,581
Parquet files and 621.94 MB, against 12 files and 6.89 MB** with `_rows_per_chunk`
recomputed before each assignment. Nothing errors. The map just silently ships 90x the bytes,
and it cost a machine restart. Only the reassign-in-place pattern is exposed; rebuilding the
layer each update re-infers a sane value and hides it entirely.

Corollary: **never seed a layer with a placeholder row.** Either construct from the first
real table or pass `_rows_per_chunk=` explicitly (a real constructor kwarg).

**A zero-row table kills deck.** Used to blank the layer on a resolution change; the map went
white on the first zoom and did not come back on a re-run. `infer_rows_per_chunk` also
returns 0 for it and lonboard asserts `max_chunksize > 0`. Blanking now goes through the
1-row seed table the layer is constructed with, which is the only shape known to survive.

**geoarrow: `from_wkb` alone is not enough.** It yields the generic `geoarrow.geometry`
union and lonboard rejects it with "Expected one of geoarrow.polygon, geoarrow.multipolygon".
Needs `to_type=multipolygon("xy", crs="EPSG:4326")`. A *single* polygon downcasts on its own,
which is why a one-geometry test passed and the real fold did not. Passing the crs also stops
lonboard warning that it cannot tell whether the data is WGS84. And `pa.array(geo_array)`
**strips the extension metadata** -- build the table with `ArroArray.from_arrow` /
`ArroTable.from_arrays`.

**marimo does not render classic ipywidgets.** `ipywidgets.HTML` produces a "please migrate
this widget to anywidget" banner. It still *serialises* into `marimo export html`, so the
headless check does not catch it. Replaced with a ~12-line `anywidget.AnyWidget` carrying a
synced `traitlets.Unicode`.

**`Map(show_tooltip=True)` is hover; `show_side_panel` (default True) is click.**
`show_tooltip` defaults to False, so inspection is click-only until you set it.

**Positron labels over the hexes** need to be a deck layer, not the basemap: the basemap
paints under every deck layer, so place names on it sit beneath an opaque cell. Carto serves
the `positron-labels-only` style as `light_only_labels`; `@2x` with `tile_size=512`.

## Stale state is the recurring failure mode

Everything the camera drives is asynchronous, so anything left on screen from the previous
fold is a lie that looks like a bug:

- **Stale cells.** Zoom in and the old coarse cells sit at the wrong size; zoom OUT and the
  old FINE cells are suddenly sub-pixel and alias into a black mush that reads as corruption.
  Handled by blanking on a resolution change, plus a per-resolution cache so returning to a
  level already folded is a dict lookup rather than a read.
- **Stale outlines.** The wash/outline is dissolved from one fold's cells. When different
  cells go up it is stale, and it does not *look* stale: it is clean lines in the right
  colours over the wrong place, or a flat tint with a straight edge where the old padded box
  stopped, which reads as a rectangular distortion across the map. `_show` now hides the
  outline layer every time it paints cells.

`SETTLE = 0.25` debounces the camera, since every fold is a network read. Verified in
isolation: a 120-event 60fps drag produces one fold, of the final position. No threads, no
timers -- the debounce is an await on the kernel's own loop.

## Open

## TODO: analytics

Nothing here is started. Written down so the framing survives, not as a plan.

### What the polygons are actually for

**Hexagons are a sample grid. Polygons are objects.** A cell knows what class it is; it
does not know what it is part of. That line decides which questions need the dissolve and
which do not, and it is worth checking against before building anything:

- **Cells answer these, cheaper and better.** How much of each class is on screen. Purity.
  Anything joined on cell id at any resolution, including another dataset folded to the
  same grid. Year-on-year change as a per-cell diff. If the analytic is area or
  composition, the polygons add NOTHING and are decoration.
- **Only objects answer these.** How many separate forests, rather than how much forest.
  The size distribution of those patches. Whether a class is one block or ten thousand
  scraps, which is invisible in a class total: two views can have identical composition and
  completely different structure. What borders what, and over how much edge. What is
  enclosed by what. Whether two areas are connected, which is the union-find already
  running in `to_cluster_table` and thrown away after it is used to drop speckle.

### The caveat that would quietly corrupt half of it

**The polygon boundaries are hex staircases, not real edges.** Area is trustworthy: it is
the cell count. **Perimeter is not.** The zigzag inflates it by a factor that depends on
resolution, so perimeter-to-area ratios are comparable WITHIN a resolution and misleading
ACROSS one. Any fragmentation metric either normalises for this or stays inside one res.
Anything that reports a perimeter in metres without saying this is lying politely.

### The ladder, cheapest first

1. ~~A stats line for the current view.~~ **DONE, and it became the drawn box instead.**
   `selected_bounds` (lonboard's own draw control) captures a box AND the H3 resolution the
   map was on when it was drawn, and the cell below folds that AOI across all 40 years:
   area per class per year in SQL over a (year, y, x) cube, and patch counts from
   `patch_stats` on the same H3 cells. Measured on a 0.6 x 0.4 degree box at res 8, 51,546
   px/year: **300 ms** to read all 40 years warm, ~3.7 s cold while the COG headers are
   still opening, which is why they are prefetched at startup. Per-year cost is the thing
   to remember: the box is small and the overview matches the res, so 40 years of it is
   cheaper than one whole-country view.

   It works, in the sense that it says something composition alone cannot. Kentucky,
   1985 to 2024: Pasture/Hay loses 2.9 points of area while going from 90 patches to 121,
   and Cultivated crops gains 2.0 points while going from 23 to 52. Deciduous forest holds
   both. That is the argument for the dissolve, in one table.
2. **Hover or click a cluster.** Its class, area in km², its rank by size, its hole count.
   The polygons are already there; `pickable=False` only to keep hovers off the cells.
3. **A distribution, rendered.** Patch sizes as a small chart under the map. This is where
   speckle-versus-structure becomes visible rather than a number.
4. **Fragmentation as the fill colour.** Recolour clusters by patch density or size instead
   of class, so the map answers "how broken up is this" rather than "what is here". Same
   fold, different question. A lightness ramp suits it and stays colourblind-safe.
5. **Two years overlaid.** The expensive one: a second read of everything. Change as
   geometry (what a run gained, lost, or split into) rather than as a cell diff. The year is
   pinned at 2024 with the slider deliberately out, so this is a decision, not an omission.

### A class selector in the console, defaulting to forest

**The UI, which is the small part.** A dropdown in the `Controls` widget alongside the
existing checkboxes and sliders, choosing which land cover class the map is about. It opens
on **forest** (41 deciduous, 42 evergreen, 43 mixed) rather than on everything. One more
synced traitlet plus a `select()` helper next to the `check()` / `slider()` / `steps()`
helpers already in the widget's JS, and a `WHERE` in the fold.

Forest is the right default because it is usually the largest thing on screen: deciduous
alone was 39.7% of the measured southeast viewport in the colour section above. The map
therefore shows something immediately rather than opening on an empty filter.

**Why it is worth more than a filter.** Everything gets sharper when only one class is in
play:

- The dissolve stops being "runs of whatever class" and becomes forest patches, directly.
  `WASH_SQL` needs no change; it groups by class already.
- `MIN_CLUSTER` can be tuned for one class instead of compromised across sixteen.
- It is the precondition for the join below, which is the actual point.

### Joining NDVI to a filtered class (the reason for the selector)

**Source.** The Earth Genome seamless cloud free Sentinel-2 mosaic. Two properties settle
the design, and both come from the data rather than from preference:

- It is **pre-composited and seamless**, which is what makes it fit the camera driven read
  at all. Per scene Sentinel-2 would mean a search plus a cloud composite per viewport, and
  that breaks the one read per camera move model this notebook is built on.
- **Only NDVI is seamless. The other bands have seams.** So read the NDVI band and never
  derive NDVI from red and NIR. Two independent reasons point the same way: the seams are
  visible, and deriving from an overview gives NDVI of averaged bands rather than the
  average of NDVI, which is a different number (largest exactly at edges). The cost is that
  this source yields exactly one index. No NDWI, NBR or EVI, since those need the seamed
  bands.

**The join is on H3 cell id, and that is the whole trick.** NLCD is Albers 5070, the mosaic
is not, and it does not matter: fold both to the same resolution and the cell id absorbs the
reprojection. No resampling, no warp, no shared grid to agree on. This is the argument for a
second dataset generally, and it is stronger than "another map".

**Resolutions do not match, so the join picks the coarser.** Sentinel-2 10 m floors at H3
res 12 (100 m² per pixel against a 307 m² hexagon, about 3 px/hex; res 13 is 0.44 and holes
out, the same wall res 11 hits on 30 m NLCD). NLCD floors at res 11. **Join at res 11.**

**Palette, and this one is a trap.** The conventional NDVI ramp is brown to green, which is
the red/green axis this project is not allowed to use, and every tutorial uses it. Viridis
or cividis, per `CLAUDE.md`. NDVI is the one layer here where the standard choice is the
wrong choice.

Water is legitimately negative NDVI rather than nodata. Bin it, do not mask it.

**Two questions this answers, neither of which either dataset answers alone:**

1. **Forest stress.** Cells that are anomalously low NDVI *for their own class*. Never pool
   41 and 42: deciduous and evergreen have completely different NDVI seasonality, so an
   anomaly only means anything measured within its own class. Dissolving the anomaly flag
   then gives stress patches as objects, with a count and a size distribution. That is the
   case where the dissolve earns its keep on continuous data, which raw elevation never gave
   us.
2. **Urban greening.** Filter to developed and ask whether cities have gained or lost green
   over the Sentinel-2 era.

**Three things that would quietly corrupt the urban version:**

- **Fixed cohort, not per year class.** Take the cells that were developed in the first
  year and track *those same cells* forward. Using each year's own NLCD mixes greening with
  urban expansion: a cell that converted from cropland to developed in 2020 would read as an
  urban NDVI change when it is really a land cover change. The fixed cohort is also cheaper,
  since the cell set resolves once and every later year is an NDVI read against a fixed hex
  list.
- **Split the developed subclasses.** 21 open space is mostly lawn and park and is naturally
  high NDVI; 23 and 24 are high intensity, where a gain means street trees. Pooling them
  averages two different phenomena. The class column is already there, so this is free.
- **Interannual weather dominates the signal.** A wet spring against a dry one moves regional
  NDVI further than a decade of planting does, and composite date windows may differ year to
  year on top of that. A raw urban NDVI series is a rainfall chart. The fix costs nothing
  because the control cells are already in the fold: track urban NDVI **relative to**
  non urban cells of the same class in the same view and the same year. City rises while the
  countryside is flat is greening; both rise together is weather.

**Scope note.** Sentinel-2 starts 2015, so this lives in roughly 2017 to 2024. It does not
span the 40 year axis the NLCD box fold gets, and any chart showing both has to say so.

### Also open, unranked

- **Query targets.** A dissolved run is a natural AOI: click a forest and it becomes the
  boundary for the next read. Elevation inside it, imagery clipped to it, an Overture join
  against the footprints that fall in it.
- **Holes are information.** A hole in a crop polygon is a town, a lake, a woodlot.
  `cells_to_wkb_polygons(link_cells=True)` already returns interior rings and nothing
  downstream looks at them.
- **Export.** They are proper WKB polygons and could leave as GeoParquet or FlatGeobuf,
  which makes the fold a data product rather than a picture.

- **A glitch on zoom remains, unresolved.** Reported repeatedly: wrong resolution and
  incomplete coverage, transiently. The blanking and the cache reduce it but do not remove
  it. Not diagnosed.
- `VIEW_W, VIEW_H = 1400, 620` is an **assumption** about the map's pixel size, not a
  measurement. If the real viewport is wider, the folded box is smaller than the screen and
  the edges go unfilled permanently. Worth reading the real size back from the browser.
- `_note_jump` counts camera moves too large to be a drag and names where they went, to
  separate "a marimo cell re-ran and rebuilt the Map" from a JS-side camera move. It has not
  yet been used in anger.
- lonboard hands deck an **uncontrolled `initialViewState`** from the same model
  `onViewStateChange` writes back to every frame. A stale echo would snap the camera. This is
  a candidate for the jumps, not a measurement.

## Process notes

Two habits caused most of the damage in the rebuild, and both are cheap to avoid:

1. **A headless `marimo export html` that exits 0 proves the Python ran, and nothing else.**
   It caught neither the 90x serialization blowup, nor the ipywidgets banner, nor any visual
   problem. The 126 MB export file *was* the blowup, sitting in plain sight, dismissed
   without being opened.
2. **Computed estimates were presented in tables that read as measurements.** Say which is
   which, or measure it.
