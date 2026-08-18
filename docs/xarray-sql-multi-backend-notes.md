# xarray-sql 0.4.0 pre-release: multi-engine backends, measured (2026-08-18)

Test bed: `xarray-sql-multi-backend-test/` at the repo root (the `archive/` pattern:
its own `pyproject.toml`, `uv.lock`, `.venv`, gitignored by the root ignore). The
dependency list is the root pyproject's verbatim except `xarray-sql[duckdb]==0.4.0rc1`,
with `[tool.uv] prerelease = "explicit"` (a global `allow` also pulled a duckdb 1.6 dev
build; explicit keeps duckdb at 1.5.5). The rc pins `datafusion==54.0.0` exactly. Run
things with `uv run --project xarray-sql-multi-backend-test ...` from the root.

Scripts in the folder, all synthetic and offline: `smoke_test.py`, `bench_scan.py`,
`bench_fold.py`, `bench_dissolve.py`. Numbers below are from Stephen's Mac (arm64).

## What the rc is (PR #227, the release body is one line)

xarray-sql becomes the translator between a lazy `xr.Dataset` and any engine's own
connection. Two seams:

- `xql.register(con, name, ds, chunks=...)` dispatches on connection type. DataFusion
  delegates to the existing table provider (so `XarrayContext.from_dataset` is
  unchanged); DuckDB gets `XarrayPushdownDataset`, a `pyarrow.dataset.Dataset` subclass
  with projection pushdown, per-dimension chunk pruning (Arrow guarantee
  simplification over shadow fragments, no expression parsing), coalescing, and a
  bounded prefetch pool. `xql.arrow_dataset(ds)` is the same object for Polars, Dask,
  Ibis. The DuckDB path REQUIRES a chunked dataset or explicit `chunks=`.
- `xql.to_dataset(result, template=ds, chunks=..., spill=...)` takes a DuckDB relation,
  DataFusion DataFrame, Polars frame, or anything with `__arrow_c_stream__` back to a
  labelled (optionally lazy, chunked) `xr.Dataset`.

Also: `register(..., geometry=("x","y"))` emits a GeoArrow point column (pixel
CENTRES, WKB for DuckDB GEOMETRY), `xql.bbox_conjuncts`, extras `[duckdb]`, `[polars]`,
`[geo]` (pyproj). Roadmap item to watch: "optional H3 cell column emitted at
registration", which would precompute the cell in Python; that is the shape rejected
for heat hex (the fold must be the UDF inside the SQL, CLAUDE.md).

## Benchmarks

### Scan throughput, DataFusion path, 0.3.2 vs rc (`bench_scan.py`)

Synthetic (168, 500, 500) float32 = 42M rows, the notebook's `t x 45 x 45` block shape,
warm runs.

| query | 0.3.2 | 0.4.0rc1 |
|---|---|---|
| `count(*)` | 0.05 s | 0.01 s (chunk arithmetic, no scan) |
| `avg(v)` | 0.10 s, 420 M rows/s | 0.05 s, 780 M rows/s |
| `GROUP BY t, avg(v)` | 0.16 s, 265 M rows/s | 0.11 s, 390 M rows/s |

~1.5-1.9x on the pivot+scan slice with no code change. A week of HRRR (~320M rows) is
roughly a second saved: under the wire at home, a small visible piece on molab. Reason
enough to move the maintained notebooks to 0.4.0 when it goes final.

### The fold, both engines via `register` (`bench_fold.py`)

6M rows (24 x 500 x 500), a `(y, x, lat, lon)` lookup joined in as the heat hex
notebook does, `h3_latlng_to_cell(lat, lon, 6)` in the GROUP BY, 5.46M cells out,
identical row counts on both engines.

| engine | time |
|---|---|
| DataFusion + h3ronpy UDF | 0.23 s |
| DuckDB + duckdb-h3 (Arrow collect) | 1.75 s |

~7.5x, the same ratio as the NLCD measurement (70 vs 462 ms). The multi-backend does
not change which engine folds; it confirms it. Registering one lazy Dataset on both
engines from one script and running the same SQL text on both works.

Where DuckDB-on-the-cube would earn a place: pixel-field thresholds that need no H3
(rain footprints from `prate > 0` with time-window pruning), and aggregates DataFusion
cannot spill (the four-accumulator heat hex case). Not for the fold.

### Point fold vs footprint polyfill (reasoning, not measured)

`latlng_to_cell` + GROUP BY is a pure function per pixel plus a hash aggregate: cells
partition the pixels, the mean is exact, and it is right whenever the cell is at least
as big as the pixel; finer than the pixel it holes out (no centre lands). Polyfilling
each pixel's FOOTPRINT to fine cells (Stephen's raster-polyfill work) fills every cell a
pixel covers and can area-weight, at 10-100x the work and with corner geometry (HRRR
corners are Lambert, not a degree rectangle). `register(..., geometry=)` gives centres
only, so "polyfill the zarr in DuckDB" via the rc is the point fold on the slower H3,
not the footprint polyfill. Heat hex at res 6 gains nothing from switching; the case for
the footprint route is drawing HRRR at res 8-9 without holes.

## The dome / outline experiment (design, then the deciding benchmark)

Idea (Stephen, 2026-08-18): outline regions of similar sustained heat in the heat hex
film ("waves of heat"), storms as the cooling holes, polygons for domes; where does
DuckDB come in.

Layout that the numbers settle: DataFusion fold (as now) -> numpy accumulator + bands
on the frame matrix -> one Arrow `(frame, cell, band)` table into DuckDB ->
`h3_cells_to_multi_polygon_wkb` per (frame, band), `ST_Dump` for blob rows. Nothing in
it uses the rc.

Browser-side pieces (slider-reactive, zero bridge bytes): per-cell 6-neighbour index
once, then a frame's outline is `mask[i] && !mask[nbr[i][k]]` as a PathLayer of edges;
nested bands (1, 3, 5, 10 degC excess) read as dome + core; onset isochrones (first
hour over X), duration (hours over X), advancing/retreating front (over now, not an
hour ago), rain/flush footprints as holes.

Kernel/DuckDB pieces (identity, numbers): dissolved blobs with area, centroid,
lifetime; a self-join frame f to f+1 on overlap for dome tracks; per-dome series.

The accumulator is a plain window function (the recurrence unrolls to a weighted
cumulative sum): `(1-a) * pow(a,f) * sum(greatest(hi-thr,0) * pow(a,-f)) OVER
(PARTITION BY cell ORDER BY f)`, `a = 2^(-1/half_life)`; `pow(a,-f)` reaches ~2^28
over 336 h, fine in float64. Measured 30x slower than the numpy loop, so numpy it is.

### Dissolve cost (`bench_dissolve.py`), the live-vs-button question

Pessimistic fake: 359k res 6 cells (1.7x the real 210k), 168 frames, three moving
domes plus per-cell noise so band edges speckle into ~420 blobs per group; a real
sustained-heat field is smoother. 60M answer rows in, 10M cell-frames over the
1 degC line, 411 (frame, band) groups.

| step | whole film | one frame |
|---|---|---|
| accumulator, DuckDB window fn | 2.4 s | |
| accumulator, numpy | 0.08 s | |
| banding | 0.06 s | |
| A. `h3_cells_to_multi_polygon_wkb` per group | 6.0 s | 37 ms |
| B. `ST_Union_Agg(cell boundary) + ST_Dump` | 191 s | 1.1 s |
| C. boundary directed edges (in mask, neighbour not) | 2.3 s | |
| D. A then `ST_Dump` + `ST_Area_Spheroid` + centroid | 8.0 s | |

- The dissolve is A, not B: H3's own outer-boundary walk is 30x faster than the spatial
  union of hexagons and gives the same dissolved geometry. B was right for a
  per-viewport slice (the NLCD notebooks); it is wrong for a film.
- Per-blob identity does not need the union: `ST_Dump` over A's multipolygon (D) is
  +2 s for 173k blob rows with area and centroid.
- 37 ms per frame means a slider-driven outline of the current frame can round-trip
  the kernel inside a debounce; the whole film re-polygonises in ~6 s.
- 105 MB of WKB for the noisy film is the speckle; measure on real output before
  choosing between shipping polygons or shipping the mask and edge-testing in JS.

### D on the real fold (`dome_real.py`, 2026-08-18)

`dome_real.py` runs the heat hex notebook headless (marimo `app.run()` on a copy with
constants patched: `DAYS = 7`, no rain/wind; `FULL=1` in the env runs its own dome
week) and does numpy accumulator -> bands (1, 3, 5, 10 degC excess, cumulative sets)
-> A -> D on the fold's `cell_hour`. Outputs (polys, blobs, frames) land in the
scratchpad as parquet/npz. Two script lessons: marimo's cell decorator needs the
notebook source ON DISK (`inspect.getsourcelines`; exec of a patched string fails
"source code not available"), and `ST_Area_Spheroid` returned NaN on the H3
multipolygons, so area is `ST_Area(ST_Transform(g, 4326 -> 5070 Albers))`.

Two runs. East dome week (`FULL=1`, 168 h, 5 variables): notebook 279 s of which
fold 263 s (a full store chunk, rain and wind on: the notebook's known cost, not the
dissolve), 35.4M cell-hour rows, A 2.8 s / 550 rows / 132 MB WKB, D 2.6 s. Young
chunk (2026-08-12 .. 08-18 14Z, 159 h, 2 variables): fold 44 s, 33.5M rows,
peak sustained excess 14.3 degC, cell-frames per band 12.7M / 8.7M / 5.8M / 111k,
**A 2.0 s (602 rows, 96 MB WKB), D 0.5 s**, ~4 ms per frame implied.

What the blobs are: median blob 41-42 km2 at every band, i.e. single res 6 cells
(a cell is ~36 km2): the real field speckles at band edges too, so most blob ROWS are
one-cell islands and a `km2 > k` filter (or a 1-ring open/close on the mask) is
wanted before tracking. The dome itself is unmistakable: the >=3 degC band's largest
blob is 1.5-2.6M km2 (the South from Texas to the Carolinas), sitting over the lower
Mississippi valley for five days and drifting from -93 to -90 lon, breathing between
~2.5M km2 at 00Z and ~1.6-2.0M at 12Z. The >=10 band peaks at 69k km2 over the
Texas coast. Bytes: median 210 kB WKB per frame for the >=1 band, so ~33 MB for a
film of one band and ~96 MB for four, against a 33 MB frame matrix; per-frame on
demand is cheap, the whole-film polygon push is not obviously better than shipping
the mask and edge-testing in JS.

Directed edges (C) vs `cells_to_multi_polygon` (A), the same thing at two stages:
H3's `cellsToLinkedMultiPolygon` collects every cell's six edges, cancels the ones
shared by two cells in the set, and LINKS the survivors into ordered loops (outer
ring + holes). C is exactly the survivors before linking: the same boundary
geometry as an unordered bag of segments (5.9M edges in the fake film vs 411 linked
multipolygons), cheaper to compute (2.3 s vs 6.0 s) and what the browser-side
neighbour test produces for free, but not fillable and not a per-blob object. A is
C plus the walk; D is A plus `ST_Dump`. (`polygon_to_cells_experimental` is the
OTHER direction, polygon -> cells with containment modes, the polyfill.)

Next: a size filter / morphological open on the mask before D, then frame-to-frame
overlap tracking of the surviving blobs (dome id, lifetime, track).

## Pointers to talk about (2026-08-18, from Stephen, unassessed)

- https://github.com/xpublish-community/xpublish-zarr ("Serve Xpublish datasets as
  Zarr", small) and https://github.com/earth-mover/xpublish-tiles (Earthmover's tile
  router for Xpublish, active Aug 2026). Alex's framing: move data to a tabular
  representation, run the SQL, move back to an `xr.Dataset` (`xql.to_dataset` is that
  seam in the rc), then some xpublish variant serves the map (tiles). Stephen: "not
  sure if they're relevant at all". To discuss after the benchmarking.

## `xsql-hrrr-heat-domes.py` (2026-08-18)

The heat hex film rebuilt on the rc, Stephen's ask ("a new version of hrrr hex with this
implementation ... see the boundaries move on their own for sustained heat, using the
multi backend from the pre release"). Three additions, all recorded in CLAUDE.md's
section for it: the `ENGINE` switch (DataFusion default, DuckDB via `xql.register`,
same fold SQL), the browser-side moving boundaries (neighbour index + edge test +
PathLayer, zero bridge bytes, slider-live), and the DuckDB dome table (A + D, size
filtered, largest blob per level and the second level's track).

The DuckDB-registered fold on the REAL cube (`fold_duckdb_real.py`): young chunk,
159 h, 2 variables, `threads = 8`: 179.2 s against DataFusion's 38.8 s in the same
process, identical 33,505,116 rows and mean temperature. Untuned (prefetch, coalesce,
threads); the h3 step alone predicts ~4-5x from the synthetic ratio, so most of the
gap is that, not the pushdown scan.

Browser test in node on the real 210,724 cells (`scratchpad/bnd_test.mjs`, not kept):
neighbour index 549 ms; boundary edges at frame 150 of the young-chunk film: level 1
21,852 edges, 3: 11,320, 5: 9,450, 10: 2,570; 34 / 11 / 8 / 3 ms cold (coord cache
filling), 1.7 ms warm.

Headless export on the rc venv: fold 82 s on datafusion (5 variables, young chunk),
dome dissolve + dump 2.4 s, tables built, no cell errors. NOT FLOWN: the flight is
play with boundaries on over the sustained heat field, drag the threshold slider and
watch the lines follow, B to toggle, then `ENGINE = "duckdb"` once for the record.
