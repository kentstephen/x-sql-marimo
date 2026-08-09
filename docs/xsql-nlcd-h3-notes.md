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

1. **A stats line for the current view.** START HERE. Class composition, patch count,
   largest run, all from tables already in memory: no new read, no new geometry, and it
   answers whether the patch numbers are interesting enough to justify the rest. The signal
   to look for is the ratio of patches to cells: at res 8 over one viewport, min run 1 gives
   43,329 polygons against 176,539 cells; at res 5 over the whole country, 1,774. That ratio
   IS the fragmentation signal and nothing currently reads it.
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
