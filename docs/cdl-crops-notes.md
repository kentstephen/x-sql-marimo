# xsql-cdl-crops.py — working notes

USDA Cropland Data Layer (chill/usda-cropland-data-layer on source.coop, icechunk,
2008-2025 at 30 m + a block-majority pyramid 2x..256x, class names/colors in the
attrs) as a DuckDB-only marimo notebook on xarray-sql 0.4.0rc1: every pyramid
level registered as a table, marimo SQL cells for the analytics, and a map whose
serve is one SQL query. Built 2026-08-20 in one long day with Stephen; this file
is the record of what was measured, what broke, and what the final shape is.

## The store

- `s3://chill/usda-cropland-data-layer/v0.1.0.icechunk` via https://data.source.coop
  (anonymous, force_path_style, region us-east-1). Groups `30m` (2008-2025,
  105,432 x 160,171, 304 GB logical) and `10m` (2024-2025, unused so far).
- zarr v3 sharded, inner chunks (1, 512, 512), shards (1, 8192, 8192). Time depth
  is 1: the HRRR chunk-depth problem does not exist here; the pyramid replaces
  the whole zoom problem.
- `0` = Background (also fill), `81` = Clouds/No Data: always dropped. Landcover
  classes (Developed*, forest, water, wetlands, grassland...) drop only when
  "crops only" is on (name-matched set in the store cell).
- 30 m for 2024+ is NASS's own resampling of the native 10 m product.

## Measured (home link, 2026-08-20)

| thing | time |
|---|---|
| register a level (xql.register, whole-plane blocks) | ~0.0 s |
| CONUS class histogram, one year, 64x | 0.4 s |
| FULL 18-year CONUS scan, 64x | 1.3 s |
| corn/soy rotation self-join, CONUS, 64x | 0.6 s |
| native 30 m window 20x20 km x 3 years | 0.9 s |
| map serve, any rung incl. native (query+layer) | 0.5-1.7 s |
| analyze timelapse (18-yr GROUP BY over the box) | 0.8 s at CONUS/256x |

Block layout per level matters: whole-plane per year for k>=32 (scanned whole),
2048^2 below so x/y predicates prune fragments. A 2048 block expands to ~4.2M
rows; never register fine levels whole-plane.

## The map serve (final shape)

One SQL query per view:

    SELECT ST_Transform(ST_MakeEnvelope(x-half, y-half, x+half, y+half),
                        'EPSG:5070','EPSG:4326', always_xy := true) AS geometry,
           [c.r, c.g, c.b]::UTINYINT[3] AS color, t.crop_type
    FROM cdl_{k} t JOIN classes c ON c.code = t.crop_type
    WHERE year = ? AND crop_type NOT IN (...) AND x BETWEEN ... AND y BETWEEN ...

`UTINYINT[3]` arrives as arrow FixedSizeList<uint8>[3]: `get_fill_color` is the
table's own column. `PolygonLayer.from_duckdb` does all conversion in its
__init__ (WKB parse, interleave, reproject); the serve keeps only its table,
rechunked to ONE chunk, and assigns it onto the persistent layer under
hold_sync. No private lonboard imports anywhere (an earlier iteration used four;
Stephen: "is all this shit really necessary" — it wasn't).

- Level pick: floor rule (finest k with pixel >= PX_PER screen px), then the
  count-based budget — the box's cell count is the UPPER BOUND of drawn rows, so
  when it fits ROW_BUDGET no count query runs at all (every deep zoom skips it);
  only a box that could exceed the budget pays for a real count (background
  dominates: CONUS crops at 128x is 68k rows in a 1.2M-cell box).
- Held view: same level/year/filter and the camera still inside the served box
  (MARGIN 0.35) skips the serve entirely — most pans and small zooms are free.
- The fold box is a DENSIFIED boundary transform (9 samples/edge) clamped to the
  array's Albers bbox: an EPSG:5070 parallel bows with its lowest y at the -96
  central meridian, so a 4-corner min clipped south TX / the Gulf / Florida in
  an arc at CONUS-wide zoom (the "doesn't the whole map load on open" defect).
- Serve runs ON THE KERNEL LOOP (deforest's machinery: SETTLE debounce,
  busy/pending coalescing, _spawn with run_coroutine_threadsafe fallback).
  Widgets or trait updates from threading.Timer or run_in_executor threads are
  lost under marimo edit even when marimo run + playwright looks fine.

## The lonboard layer-id saga (read before adding a second layer)

Under marimo, every lonboard 0.16 deck layer gets `id: undefined` (its JS reads
`this.model.model_id`, which marimo's anywidget bridge does not provide). What
that does, established by a minimal repro + playwright console capture:

- ONE layer: fine.
- TWO STATIC layers passed to Map() at construction: fine (labels render), with
  benign "Multiple new layers with same id undefined" + per-tile BitmapLayer
  assertion noise in the console. Console errors are NOT evidence of a dead
  layer; screenshots are (twice this day the errors were misread as breakage —
  and once real breakage was misread as cosmetic).
- TWO layers where one UPDATES (a live serve): deck's differ cannot tell the
  colliding ids apart and the tile layer's children die at init, every path:
  visible-trait toggle (unwatched trait, never syncs), layer-membership
  reassignment (a removed layer never remounts), opacity toggle, mounting both
  at construction. A labels overlay CANNOT coexist with a live-updating layer
  under stock lonboard 0.16 + marimo.
- A 3-line bundle patch (unique per-model ids + no id on tile children) fixes it
  at the source — built, verified on the repro, and REVERTED at Stephen's
  direction ("adds too much complexity"). If a second deck layer is ever needed,
  that patch is the way; it also applies to deforest, whose dark_only_labels
  overlay is equally dead today (verified headless; unnoticed because dim
  dark-on-dark labels vanish quietly).
- Resolution here: labels come from the BASEMAP (CartoStyle.Positron, WITH
  labels, under the pixels) and the map has exactly one deck layer.

## The strip (controls under the map)

Deforest Controls/Status idiom: 12px ui-sans-serif flex row, transparent
bordered buttons, 12.5px ui-monospace status line. year slider (native range,
no locale commas) + ◀ ▶ step arrows, crops-only checkbox (starts OFF), analyze
button, and a Photon SEARCH FIELD (flood's client moved into the strip: urllib
on a thread, camera-biased, first hit flies via `deck.fly_to` — assigning
`view_state` kernel-side is ignored — extent picks the zoom, and the refold
follows). Analyze fills the strip with top-10 classes (chips, M acres, share)
plus an 18-year timelapse of the box as an inline SVG (top 6, class colors,
direct labels at line ends), with a × clear button. In browser fullscreen the
strip re-parents into the fullscreen element (found via
shadowRoot.fullscreenElement descent — document.fullscreenElement reports the
shadow HOST) as a docked bottom bar with its own white backdrop. lonboard's
draw-box toolbar is hidden by deforest's aria-label walk.

Trait contract (proven types only): `ctl` Unicode browser->kernel, JSON
{act: set|analyze|search, year, crops, q, n}; `status`/`panel` Unicode
kernel->browser. marimo re-running the wiring cell (which reads hud.widget.ctl)
IS the year/toggle/search dispatch.

## marimo gotchas paid for here

- One duckdb connection cannot serve marimo SQL cells and the map at once
  (streaming Arrow results interleave: "Can't 'FetchRaw' from
  ArrowQueryResult"); the map has its own connection + lock.
- Cursors do not see xql.register's registrations (per-connection views).
- A cell-level underscore def referenced from a sibling closure hits marimo's
  name mangling (NameError `_cell_*`); nest it.
- `uv add` upgraded marimo 0.23.16 -> 0.24.0 as a side effect (0.24 also
  disposes task-created widget models: "Model not found for key", black map
  with a healthy kernel); pinned marimo==0.23.16 in the rc project.
- uv installs by hardlink from its cache: editing a venv file in place edits
  the cached wheel copy, and reinstall re-links the edited file.

## Verification harness

Playwright driving headless Chromium against `marimo run --headless` (drive
scripts in the session scratchpad; the pattern is worth recreating): recursive
shadow-root walks to read the strip/status, scrollIntoView BEFORE mouse.wheel
(the canvas sits below the fold; the first zoom test passed vacuously), console
capture for layer errors, screenshots as the ground truth. The playwright pip
version must match the cached chromium or `playwright install
chromium-headless-shell` fetches one.

## Analytics cells

`App(sql_output="native")`: mo.sql returns duckdb relations (no polars/pandas
shipped; the altair chart eats `rel.arrow().read_all()`). Area time series =
one 18-year scan at 64x; rotation matrix = (y,x) self-join PIVOT between two
dropdown years. Block-majority pyramid counts are approximate (corn at 64x
reads ~119M acres vs ~90M planted): trends and transitions at a fixed level are
honest, absolute acreage wants native over a window.

## The legend (pickable, per-view)

The strip's right side holds a legend of the classes actually in view (top 14
by count, unfiltered mix so every chip stays reachable), refreshed by each
serve via a `legend` Unicode trait (JSON). Chips are BUTTONS: click isolates
that class on the map (the selection joins the serve/count/analyze predicates
as `crop_type IN (...)`, busts held views and memo keys), multi-select
toggles, `× all` resets. Measured: Corn from the CONUS view = 9,015 px in
678 ms; unpick restores from memo in 284 ms. JS gotcha that cost a round: the
legend renders once at build time, so its state (`sel`) must be declared
BEFORE the render call — a TDZ ReferenceError there kills the whole widget
silently (no status line, nothing).

## Unbuilt / later

- **Segment the pixels with DuckDB** (Stephen, 2026-08-20). THE PLAN:
  - Two candidate mechanisms, both pure DuckDB:
    1. DISSOLVE: `ST_Union_Agg(ST_MakeEnvelope(...))` per class over the
       view's pixels -> multipolygons, drawn transparent-fill + stroked.
       The result is still POLYGONS, so it rides the existing single
       persistent PolygonLayer as a "boundaries" MODE TOGGLE in the strip
       (fill vs outlines) and never trips the one-layer constraint.
       `ST_Dump` on the unions gives per-blob identity (area, centroid) for
       free, the heat-domes dome-table pattern; a blob table cell could rank
       the largest contiguous fields of each crop in view.
    2. NEIGHBOR-JOIN edge extraction: an edge is a boundary where the
       adjacent pixel differs (self-join on x +/- 30k, y +/- 30k at level k);
       emits segments directly, no union, near-linear. BLOCKED for drawing:
       segments want a PathLayer, i.e. a SECOND deck layer, which cannot
       coexist with the updating pixels layer without the declined layer-id
       bundle patch (see the id saga above).
  - Therefore v1 = route 1 as a mode toggle. Build order: (a) 15-minute
    BENCHMARK of ST_Union_Agg at realistic viewport sizes (50k / 200k / 400k
    squares, few classes vs many) -- this is the go/no-go; the repo has a
    prior of ST_Union_Agg being 30x slower than alternatives on hexagons;
    (b) `boundaries` flag through `ctl` -> serve branch: dissolve query,
    `get_line_color` from the class color column, transparent fill,
    line_width_units "pixels" (repo lesson: metres is the default and floors
    the width); (c) memo/held keys gain the mode; (d) playwright pass.
  - Estimate: ~1-2 h of iteration after a sane benchmark; if the union is
    too slow, route 2 reopens the second-layer/patch question first.
  - Pairs with the pickable legend: isolate a class, outline its regions.
- **The 10m group**: only `30m` is read today; 10 m (2024-2025) could serve
  the deepest zoom rungs for those years, or the 10m-vs-30m comparison.

County stats (duckdb spatial x Overture counties), cropland->developed
conversion, 18-year persistence map. A
/dataviz-validated crop palette for the chart was drafted and dropped (NASS
colors fail CVD validation: Spring Wheat vs Fallow ΔE 3.9; the validated 8-hue
order is in the 2026-08-20 session log if wanted).
