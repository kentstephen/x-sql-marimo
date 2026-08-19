# CLAUDE.md

Guidance for Claude Code working in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory location, colorblind-safe encodings).

## Repository layout

**Two interactive notebooks are the repo**: deforestation divisions and the terrain
3D experiment, plus one maintained one-shot (`xsql-deforest-conus-counties.py`, the
deforestation fold as a static CONUS county choropleth, below) and, since 2026-08-15,
`xsql-hrrr-counties.py` (HRRR temperature per county as an animated film, below). Everything else is in `archive/`, kept for reference and not
maintained. (The HFP pair, the flood-buildings experiment, the canopy notebook and
the fire-risk buildings notebook moved there on 2026-08-13, Stephen's call; their
full sections live under the archive heading below, undiminished.)

`xsql-deforest-divisions.py` folds a global deforestation COG to H3 and joins the cells
onto Overture divisions for a zoomable choropleth plus a drawn-box ranking. The raster is
Vizzuality / LandGriffon's `deforest_100m_cog.tif` (CC-BY 4.0) from the source.coop
repository [`vizzuality/lg-land-carbon-data`](https://source.coop/vizzuality/lg-land-carbon-data),
read from `s3://us-west-2.opendata.source.coop/vizzuality/lg-land-carbon-data/`. Divisions
come from Overture's own PMTiles build of the pinned release
(`overturemaps-extras-us-west-2/tiles/<release>/divisions.pmtiles`), NOT the GeoParquet:
the GeoParquet layout makes geometry (99% of the bytes) unprunable, measured at ~190 MB
per viewport against ~0.8 MB from tiles. The PMTiles reader is hand-rolled (ported from
the parked terrain notebook), the MVT decode too; tile-clipped pieces are dissolved per
`division_id` in DuckDB before drawing or the stroke shows tile seams. Full record in
`docs/deforest-divisions-notes.md`.

Since 2026-08-14 the deforestation PAINT is a lonboard RasterLayer serving the COG's own
pyramid as ramp-coloured PNG tiles (kernel-side fetch/render callbacks; the layer is
built directly, not via `from_geotiff`, because from_geotiff's fetch is not sparse-aware
and its zoom clamp ships commented out in 0.16, wrapping overzoom onto coarse overviews).
The H3 hexagon layer is COMMENTED OUT in the map cell, not deleted; the fold still runs
because the divisions join and the ranking consume its cells. Zero and NaN pixels are
both transparent, matching the fold's `HAVING avg(v) > 0`. Needs `lonboard[geotiff]`
(morecantile) in the header and pyproject. First flight 2026-08-14: tiles render, but
the world view was covered in horizontal streaks that read as a projection bug and are
NOT one: `boundless=False` clips edge tiles and deck stretches the clipped PNG across
the full tile quad, and at coarse levels nearly every tile is an edge tile. Fixed with
`boundless=True` (padding arrives as 0.0, measured, which the zero-transparent render
already hides); that fix flew and held. Boundary fill opacity is now a Controls
slider (stepped 0.1-1.0) plus a free 0-1 number box, crossing the bridge as a Unicode
string per the proven-trait-types rule; the alpha lives in `HOLD["fill_alpha"]` (the
old FILL_ALPHA constant is deleted), new division pairs read it at build time and the
current pair is re-tinted in place (`_refill`, whole-table `pa.table(tbl)` per the
terrain recolor lesson). The map/wiring CELL SPLIT is applied (2026-08-14, the flood
notebook's pattern), closing the re-run-loses-the-fill report: the map cell builds
widgets/layers/Map only and must never re-run (VIEW_W/VIEW_H and HOME moved into it so
constants edits cannot reach it); the wiring cell re-runs freely, un-observing old
handlers via HOLD["h_*"] refs. The RasterLayer is inserted via `deck.layers` FROM THE
WIRING CELL, not passed to Map(), so a ramp or constants edit rebuilds the raster
layer and rewires without destroying the Map. The split passes headless, unflown.
OPEN DEFECT, STILL OPEN AFTER THE TRANSPARENT-PNG FIX: the raster still
stretches/smears at LOW ZOOMS ON ZOOM OUT (Stephen: the "splatter"). Absent tiles
render as a real transparent PNG instead of None (a None child never arrives, so deck
keeps stretched neighbour-zoom tiles up); that fix has now been FLOWN (2026-08-14)
and DID NOT close the smear, so the absent-tile theory is dead as the whole story.
Next suspects and Stephen's NaN hunch are in the notes doc.

`xsql-deforest-conus-counties.py` (2026-08-14) is the deforestation fold as a ONE-SHOT
on the xsql-hfp-conus chassis: no camera, no widgets, one CONUS box, one L2 read folded
to res 7 (~32 px/cell, the ladder's own row), Overture counties from the divisions
PMTiles at z8 (their minzoom; ~1,008 tiles), DuckDB dissolve + center polyfill,
DataFusion zonal join, one static PolygonLayer choropleth (log-cividis ramp with the
zero swatch; Stephen turned the stroke OFF). Flown and committed. Things to know:

- **The clip to CONUS is the join itself.** County pieces are filtered at decode
  (country US, is_land, region not AK/HI) and the polyfill is 'center'-ruled, so cells
  centred in Canada/Mexico/water are in no county and drop out; nothing else is drawn.
  No country-polygon clip step exists.
- **Zero cells are KEPT** (no `HAVING avg(v) > 0`), a deliberate departure from the
  interactive fold: that filter is a hexagon-render economy, and a county's mean share
  must include ground that lost nothing or it is a different, inflated number.
- **DuckDB's replacement scan DOES NOT WORK from marimo cell bodies**: marimo mangles
  underscore-prefixed cell locals to cell-private names, so the frame name never
  matches the SQL name and DuckDB reports the table missing. Both geometry steps use
  `con.register(...)` explicitly. (The interactive notebooks get away with the scan
  because their SQL runs inside nested functions over non-underscore locals.)
- Measured: 16,114x6,986 px window, 2.10M res-7 cells (6.4 s), 6,815 pieces -> 3,108
  counties (7.6 s), 1.48M filled cells, ALL 3,108 counties catch a centre (1.1 s).
  Headline numbers: CONUS mean 1.8%, median county 0.68%, top of the table is the
  Hurricane Michael panhandle trio (Calhoun FL 31%, Gulf, Bay) then the GA pine belt.
- **marimo `export html` runs the cells but the lonboard map does NOT render in the
  exported page** (Stephen: "map doesn't load"), so export is a smoke test, not a
  screenshot pipeline; screenshots come from the live notebook. A pretty title/legend/
  stats export layer was built 2026-08-14 and REVERTED at Stephen's direction (his
  notes before the revert: fit one screen, no rounded corners).

`xsql-hrrr-counties.py` (2026-08-15, BUILT AND HEADLESS-PASSED, NOT YET FLOWN) is
Stephen's HRRR idea: dynamical.org's hourly HRRR analysis read straight from its Zarr
with xarray-sql, every pixel labelled with its H3 res 7 cell from the store's own 2-D
lat/lon (no pyproj), Overture counties polyfilled in DuckDB at the same res, one
DataFusion join + group by giving 2 m temperature per county per hour for the last
`DAYS` days, and the WHOLE FILM shipped to the browser once: a bespoke anywidget with
deck.gl + `@geoarrow/deck.gl-layers` (the layers lonboard renders with) that owns the
clock (play/pause, scrub, fps, hover, click-to-series chart, all client side; the
kernel is idle while it plays). Route 2 of the three in `docs/hrrr-counties-notes.md`,
picked by Stephen ("geoarrow deck.gl if possible"; slices hourly / daily mean / daily
max; res 7, fall back to 6 on performance; diverging blue-yellow-orange ramp). Things
to know:

- **Two archives.** The analysis is icechunk v2 in `s3://dynamical-noaa-hrrr` (AWS Open
  Data, anonymous, NOT on source.coop), chunks 2,160 h x 45 x 45 px: any CONUS window
  up to 90 days costs about the same fetch (~20 s through xarray-sql; the full 2,160 h
  depth is 149 s). The 48 h forecast IS on source.coop as plain Zarr v3 (obstore, all
  49 leads x CONUS in 2.2 s); `SOURCE` in the constants cell switches, the pipeline is
  identical from the fold on. Both carry `spatial_ref` WKT and 27 variables.
- **No dask.** The store is opened `chunks=None` (lazily indexed) and xarray-sql cuts
  it into blocks itself (`cube_chunks`: the whole time window x 45 x 45, one block
  per store column), which DataFusion pulls in parallel; measured identical to a
  dask-backed open (20.9 s for 168 h). The block must span the whole window: a block
  grid narrower than the window decodes each 2,160 h store chunk once per block it
  touches (measured 111.7 s for the same window with 168 h blocks laid unaligned).
- **Res 7 is finer than the 3 km pixel** (1.00 px per cell, measured; res 6 is 4.2, res
  5 is 29), so the fold is a relabel and the polyfill decides which pixel belongs to
  which county. 3,107 of 3,108 counties catch a pixel (Lexington VA does not; drawn
  hollow). `pix2c(y, x, county)` is a static lookup built once (879,420 pixel rows);
  the per-frame SQL is `SELECT t, id, avg(v) FROM cube JOIN pix2c USING (y, x) GROUP
  BY 1, 2`, 168 frames x CONUS in 19.8 s headless, aggregate nearly free.
- **The widget's esm.sh imports pin EVERY deck package to one version (9.3.10, the
  newest at build time) AND pin `?deps=` so that all of them resolve to ONE
  `@deck.gl/core` module.** esm.sh hashes a module's URL by its deps list, and it
  resolves the packages' own caret ranges to the newest release: the first flight
  (2026-08-15) pinned 9.1.14 and died on `geo-layers -> mesh-layers@^9.1.0` landing
  on 9.3, which asked the 9.1 core for `phongMaterial`. The fix is pin-to-newest plus
  a crawl of the whole module graph (200 modules: one core, one luma set, one
  geo-layers; crawler in `docs/hrrr-counties-notes.md`). Re-crawl if any version
  moves. The fallback if esm.sh misbehaves is a local esbuild bundle (node 26 is
  installed), lonboard's own approach.
- **Geometry crosses as one Arrow IPC stream with INTERLEAVED coords**
  (`multipolygon("xy", coord_type="interleaved")`, one record batch): geoarrow-rust's
  default is separated `struct<x,y>`, which the JS layers do not read (lonboard
  converts to interleaved for the same reason). The extension metadata survives
  pyarrow IPC (verified). Bytes traits kernel -> browser are how lonboard ships its
  tables, so that direction is proven under marimo; the JS copies the DataView's
  buffer before viewing it as Float32 (alignment). Browser -> kernel is `clicked`, a
  Unicode row index, per the proven-trait-types rule.
- **The map cell builds the widget with geometry only and must not re-run**; the
  wiring cell pushes `config` (labels, ramp lo/mid/hi, stops, fps, height, title,
  subtitle, meta) then `frames` (float32 F x N, NaN = no pixel) and re-runs freely.
  One ramp per film: pivot at the median, span to the wider of p2/p98, `PIVOT`/`SPAN`
  pin them. Daily frames are UTC days (first/last partial if the window is).
- **The HUD is MINIMAL and on the map** (2026-08-15, after a flight of a fuller
  dashboard that Stephen called dizzying and that froze the notebook: "want
  minimal"): one panel top-left (title, legend, county mean for the frame, and a
  clicked county's line only after a click, × clears) with its own hide toggle
  (H), and the transport across the bottom (step / play / step, slider with UTC day
  ticks, timestamp, fps, ⛶ fullscreen / F). ⛶ is the DECK ELEMENT's own browser
  fullscreen (`mapEl.requestFullscreen()`, not marimo's); the HUD is inside the
  element so it comes along; deck polls its canvas size, no resize handler.
  Everything is computed in the browser from the frame matrix (no
  hyparquet/hightable). Space plays, arrows step.
- **The widget SHARES THE PAGE'S STYLESHEET, so every class is `cf-` prefixed.**
  The "hide freezes the notebook and exits fullscreen" report (two flights) was a
  root class literally named `hidden`: marimo's Tailwind owns `.hidden { display:
  none }`, so the toggle blanked the whole widget, the fullscreen element vanished
  (browser exits fullscreen) and the page read as frozen. Never use a bare
  utility-looking class name in an anywidget here.
- **ONE thing crosses back to the kernel from the widget: `window`.** The fuller HUD
  synced the clicked county to a `clicked` trait; nothing consumed it and marimo
  re-runs cells that reference a displayed widget whose value changed, so it is
  gone. The date range is different: since 2026-08-15 (night) it lives IN THE HUD
  PANEL (Stephen: "this needs to be in the map, how would anyone see it there"),
  two `<input type=date>` + mode select + load button, and "load" sets `window`
  (Unicode JSON `{"d0","d1","mode"}`); marimo re-running the cells that read
  `film` IS the refold. The widget is wrapped `mo.ui.anywidget(...)` explicitly
  in the map cell (setattr/getattr forward, so `film.config = ...` still lands);
  the window cell reads `film.widget.window`, NOT `film.value` (that packs every
  synced trait, county bytes included). Limits are checked in the JS before send
  (button disabled with the reason) and again kernel-side (`mo.stop`, the guard).
  The kernel states span/served window/limits in `config.win`; the JS reflects
  it on each film. Round trip measured in a headless Chrome via playwright:
  load pressed -> new 3-day film on screen 25 s later. The old marimo submit form
  cell is deleted. Anything else per-county wanted kernel-side must still not go
  through a synced trait.
- **Picking is GEOMETRIC, in JS, not deck's.** deck's GPU picking returned null for
  every click on three flights here (`pick: none (null)` in the ruler; hover never
  showed either), inside marimo's shadow DOM with deck 9.3.10 + geoarrow layers
  0.3.2. So a click is unprojected (`deck.getViewports()[0].unproject`) and tested
  against the county rings the browser already holds (arrow data walk:
  multipolygon `valueOffsets` -> polygon -> ring -> `FixedSizeList<f64,2>` values;
  bbox reject, even-odd over every ring; index 4 ms, lookup 0.1 ms, verified in node
  on the real table). A press that starts on the HUD or moves > 4 px is a drag.
  There is no hover readout for the same reason.
- **Counties are NOT stroked** (Stephen: "then dont stroke the counties"). Adjacent
  counties from the z8 tiles do not quite meet at deep zoom (~30 m vertex
  quantisation, 1-3 px hairlines at z10, his "gaps btw counties"); a same-colour 1 px
  stroke would hide them and was declined. The other fix, if wanted, is finer county
  geometry (z10 tiles, 16x the fetch), not a stroke.
- **The window is the HUD's load button** (see the `window` bullet above; the
  marimo `.form()` cell it replaced was above the map and invisible in practice),
  UTC days inclusive, end clipped to the newest hour; opening default is the last
  `DAYS` (7) days ending now. Limits per mode: hourly <= `HOURLY_MAX_DAYS` (14; 336 frames), daily mean/max
  <= `DAILY_MAX_DAYS` (92; the read cost, one full store chunk is 149 s); over the
  limit `mo.stop`s with the reason rather than clamping. The fold is memoised in
  `HOLD` on (SOURCE, t0, t1) so re-submitting the same dates or switching
  hourly/daily never refetches. The forecast source ignores the window (one init's
  49 leads, coords rewritten to valid times).
- **Startup cost, measured:** store 2.8 s, counties 7.3 s (1,008 ranged GETs + Python
  MVT decode; dissolve 0.1 s), polyfill + lookup 1.5-3 s, fold ~20 s. The fold is the
  floor and FROM STEPHEN'S MACHINE IT IS THE WIRE (2026-08-17, corrected: an earlier
  note here said decode): a 7-day CONUS window is 960 subchunks, 489 MB compressed,
  and the link to us-west-2 measures ~24 MB/s whether one object or eight stream
  (bandwidth, not request latency), so DataFusion partitions 32/96, zarr
  `async.concurrency` 64 and icechunk `max_concurrent_requests` 64/256 all land at
  ~20 s because the pipe is already full. Decode is real but hidden under the fetch:
  each 45x45 store chunk is 2,160 h deep, so any window decodes 960 x 17.5 MB
  (16.8 GB), ~13 s in zarr-python's blosc pipeline, 1.2 s on zarrista's Rust pool
  (developmentseed's `zarrs` binding, 0.1.0, beta; takes an icechunk Session
  directly; measured end to end 17.6 s vs zarr-python 18.7 s here). NOT ADOPTED: at
  home it buys ~1 s; near the data (molab, us-west-2) decode becomes the floor and it
  would be the ~10x lever, replacing the read cell only (numpy cube into
  XarrayContext; fold and join untouched). zarr-python's own zarrista engine is PR
  #4064, an open draft. Numbers in `docs/hrrr-counties-notes.md`. Do not try obstore
  for icechunk (same Rust object_store underneath, and no seam). The counties are
  cached as parquet in the OS temp dir (`CACHE_DIR`, Stephen: tmp not .cache; None
  disables): warm run reads them in 0.0 s.
- **Status (2026-08-15, evening): FLOWN AND WORKING, pushed as f95c637.** Map and
  film play, hide/fullscreen work once the Tailwind `.hidden` collision was fixed,
  clicking a county outlines it (yellow PathLayer from its own rings; a one-row
  GeoArrow layer via `table.slice` outlined EVERY county because the layer reads the
  full offsets under a sliced table) and shows its value and line; clicking it again,
  or off any county, clears. Headless: store 2.8 s, counties 7.4 s cold / 0.0 s from
  the tmp cache, polyfill + lookup 1.5-3 s, fold ~20 s for a 7-UTC-day window.
  Carries an "Open in molab" badge cell (first cell after the imports); Stephen
  ran it in molab from the badge and it works there (2026-08-15).
  Stephen: "probably the coolest notebook I've shared so far, but the slowest";
  thirty seconds accepted for a LinkedIn demo, no cloud spend, no publishing a
  derived cube (the idea and its numbers are in the notes). Known: 1-3 px seams
  between counties at deep zoom (z8 geometry), left as is. `marimo export html` does
  not render the widget (same as lonboard).

`xsql-hrrr-heat-domes.py` (2026-08-18, HEADLESS-PASSED, NOT YET FLOWN) is the heat hex
film on the xarray-sql 0.4.0 PRE-RELEASE (`xarray-sql[duckdb]==0.4.0rc1` in its
header; run it from the rc venv, `uv run --project xarray-sql-multi-backend-test
marimo edit xsql-hrrr-heat-domes.py`, or `--sandbox`), with three additions:

- **`ENGINE` in the constants cell**: `"datafusion"` (default; XarrayContext, the
  h3ronpy UDF) or `"duckdb"` (the rc's `xql.register(con, "cube", ds, chunks=...)`,
  a pushdown pyarrow dataset on the notebook's own DuckDB connection, duckdb-h3's
  `h3_latlng_to_cell` in the GROUP BY). Same fold SQL text apart from the res literal.
  Measured on the real cube, young chunk, 159 h, 2 variables: 39 s vs 179 s,
  identical 33.5M rows and means. The DuckDB branch is verified by script
  (`xarray-sql-multi-backend-test/fold_duckdb_real.py`), not yet inside the notebook.
- **Boundaries of the sustained heat, moving with the film**: at each level in
  `CONTOURS` (1, 3, 5, 10 degC of sustained excess) the browser draws the edge of the
  set of cells at or above it, every frame, on either field, following the sliders.
  Mechanism: a per-cell 6-neighbour index built once from h3-js
  (`originToDirectedEdges` / `getDirectedEdgeDestination`, 0.55 s for 210k cells),
  then a frame is one pass over the load matrix (edge on the boundary when the
  destination is below the level or off land), edge coords cached lazily
  (`directedEdgeToBoundary`), one PathLayer per level with binary attributes.
  ~2 ms per level per frame warm; zero bytes across the bridge. Toggle: the
  "boundaries" button in the fields row, or B. Gold lines (inferno's
  #f7d13d, Stephen's call), thin to thick and more opaque by level.
- **The dome table (DuckDB, kernel-side)**: numpy runs the accumulator at the
  constants-cell defaults, DuckDB dissolves each cumulative (frame, level) set with
  `h3_cells_to_multi_polygon_wkb`, `ST_Dump`s to blobs with Albers area
  (`ST_Transform` to EPSG:5070; `ST_Area_Spheroid` returns NaN on these) and
  centroid, drops blobs under `DOME_MIN_KM2` (500; the median blob is one cell of
  speckle), and shows the largest blob per level plus the hour-by-hour track of the
  second level's dome. 2.4 s for a week. `ST_Union_Agg` of hexagons is 30x slower on
  the same input and is not used. Full numbers in
  `docs/xarray-sql-multi-backend-notes.md`.

- **`CHUNK_CACHE_GB` (6): the store is opened with an icechunk chunk-bytes cache.**
  icechunk's default retains nothing (a repeat read of the same shard measured the
  same as cold, 19.7 s); with the budget the repeat is 0.3-0.5 s, and through
  xarray-sql a second 168-h window in the same 90-day store chunk measured 2.1 s
  against 120 s cold. The layout itself cannot be filtered finer: inner chunks are
  (2160, 45, 45), the whole time depth, so a window always fetches every filled
  hour of the columns it touches; land pruning is the only filter. Portable by hand
  to the counties film and heat hex. Numbers in the notes doc.

- **`MIRROR_DIR`: a disk mirror of full time shards** (`MirrorStore`, a read-only
  zarr v3 Store cell wrapping the icechunk session store; keyed by (key, byte range)
  exactly as the sharding codec asks; overrides `get` and `get_ranges`; only keys
  the store cell marks mirrorable are written: the read variables' shards with time
  index below the youngest, which grows hourly and stays live). Measured on the
  East dome week, 2 variables, full chunk: cold 183 s (fold 172 s), a FRESH PROCESS
  afterwards 18 s (fold 7.0 s, 2,092 ranges from disk, 0 fetched), same rows. 3.4 GB
  in `$TMPDIR/x-sql-marimo/hrrr-mirror/<store version>/` per full chunk for two
  variables (1,046 inner chunks + 16 shard indexes). `None` disables. `fold_stats`
  reports hits/fetched. Lessons: tmp files need unique names (two blocks fetch the
  same shard index at once; same-pid tmp names collided and `os.replace` raised
  inside DataFusion), and a marimo cell may not re-import a name another cell
  defines (`asyncio` is the imports cell's; the class cell takes it as a ref).
  PORTED 2026-08-19 with `CHUNK_CACHE_GB` to xsql-hrrr-counties.py (analysis
  source only; the forecast branch sets `mirror = None`) and xsql-hrrr-heat-hex.py,
  same class by copy, same dir: heat hex's East dome week then read T + RH from the
  mirror heat-domes had written (3,618 ranges from disk, 1,612 fetched for rain +
  wind), fold 108.9 s against 263 s measured before. Counties' opening window is in
  the youngest shard, so it reports 0/0 until a window reaches an older shard.

The original `xsql-hrrr-heat-hex.py` is unchanged and stays on 0.3.x.

`xsql-mapterhorn-explorer.py` (EXPERIMENTAL, open defects below) draws Mapterhorn terrain
worldwide as extruded H3 columns: the DEM half of the parked
`archive/xsql-duckdb-terrain-h3.py` standing alone, on the canopy notebook's chassis
(ruler Status widget, camera machinery, `_instant`/`refresh`). No DuckDB, no pyproj;
the fold is xarray-sql + the h3 UDF, `avg(elev)` per cell. Things to know:

- **Two archive tiers, one reader.** `planet.pmtiles` (z0-12) serves everything up to
  res 11; res 12 and 13 route to the regional `6-{x}-{y}.pmtiles` archives (z13-18,
  z17 over flat country, 457 land-only files; an ocean key is an ABSENT OBJECT, not
  an empty archive). The PMTiles client is the parked notebook's, generalised to
  archive-per-path (`_pm_open`); a tile at z > 12 belongs to the regional archive at
  `(x >> (z-6), y >> (z-6))`, opened lazily as a task per key. res 12 reads z14
  (~23 px/hex), res 13 reads z15 (~13 px/hex).
- **No bathymetry, measured.** Open-ocean tiles are absent from ~z6 up; where ocean
  exists at coarse zooms it reads ~0 m (mid-Pacific z4: 99.7% of pixels within 1 m of
  zero). The fold drops `|elev| <= 1` and `elev <= -500`; Death Valley (-86) and the
  Dead Sea shore (-430) survive as signed metres.
- **The fold box is deck's camera footprint, ray-cast exactly** (2026-08-15).
  `view_to_bbox` casts the four screen corners of deck's pinhole camera (1.5
  screen-heights from the focal point, half-fov 18.4 degrees) onto the ground,
  rotates by the bearing, and takes the Mercator bounding box; pitch 0 collapses
  to the flat box. It replaced a sin(pitch) horizon heuristic in `_pad` and the
  `_cam_ok` tolerances (both deleted): at pitch 60 the screen sees 2.37
  view-heights past the centre and 2.36 widths wide (~7x the flat area) and the
  heuristic reached ~1.98 heights and ~1.0 widths, which was the missing horizon
  band and empty far corners on tilt, orbit and fullscreen. `_pad` is now only
  PAD's symmetric slack; coverage is plain box containment; pitch >= 35 folds
  one H3 step coarser (7x fewer cells, which is what pays for the ~7x area).
  Table of footprints per pitch and the frame conventions are in
  `docs/mapterhorn-explorer-notes.md`. Headless passes (res 4 · 82,387 cells);
  the flight to run is fullscreen, pitch 60, full orbit.
- **The ladder** is BASE_RES 7 / ZOOM0 6.2 / PER_RES 1.4, MIN_RES 4, MAX_RES 13, plus
  a `res offset` +/- 2 slider in the panel (commit debounced 350 ms after the thumb
  stops, because Safari and Firefox fire `change` DURING a drag and every stop is a
  refold; a dim warning beside it says each + step refolds ~7x the cells, and
  MOVING THE MAP RESETS A RAISED OFFSET to 0: +1/+2 is a statement about the view
  it was set on, negative offsets survive a move). The
  scale opens at a FIXED 20x at every zoom: a continuous quadratic slider (half the
  track covers 0-50) plus a typed number box that takes any non-negative float. The
  AUTO FIT IS DELETED (2026-08-13, Stephen's call: it fit relief to ~25 px and read
  flat next to the 20x default). HOME is THE WORLD, flat at zoom 1.6 (was the
  pitched Alps; opening fold res 4 · ~84k cells · 14.7M px, measured headless).
  The colormap is a panel DROPDOWN over the CB-safe shortlist (gist_heat, viridis,
  cividis, magma, inferno, Greens, Blues; `RAMP["name"]` is only the seed), reverse serves
  the matplotlib `_r` twin and DEFAULTS ON, RELATIVE COLORS (panel button) respends
  the ramp on the p2-p98 of the ground in view with the legend following each
  serve, and repaints are generation-counted (`RAMP["gen"]`) so stale cached tables
  recolour lazily on serve. Any ramp change (flip, mode, cmap pick) repaints
  through the ORDINARY SERVE PATH (recolour + `put_cells`), with the kernel-side
  cost printed as `repaint N ms` in the status line.
- **The "reverse cmap is slow / doesn't stick" defect had a found root cause**
  (2026-08-13): a stale static copy of the legend HTML was rebuilt late in the map
  cell and SHADOWED the HtmlLine legend widget, so `legend.value = ...` in the
  observer hit a plain str and raised AttributeError inside a comm handler, where
  exceptions are silent; the button died before its repaint line and the map only
  caught up on the next fold. Both the shadow and the bespoke colours-only repaint
  path are deleted. A SECOND layer surfaced on the first flight after that fix
  ("TypeError: arro3.core._core.ChunkedArray is not a sequence"): `recolor()` had
  never actually run, because it fed arro3 ChunkedArray columns into pyarrow's
  `pa.table()`, which converts dict values as sequences and rejects them (arro3
  exposes the C-stream protocol at table level only). `recolor` now converts the
  whole arro3 table once via `pa.table(tbl)` and rebuilds from pyarrow columns;
  measured ~1 ms for 50k rows kernel-side in reverse, relative, and both modes.
- **OPEN DEFECTS AND NEXT WORK:** (1) The res-to-zoom ratio "isn't right, doesn't
  look good yet"; the res offset slider is the manual override while it is tuned.
  (2) res 12/13 regional reads are probed but not yet exercised interactively
  (Mississippi deep zoom is the test case). (3) The 2026-08-13 rework (serve-path
  repaint, relative colors, new scale controls, res-offset debounce) passes
  headless but none of it has been flown; the `repaint N ms` readout is there to
  decide whether relative colors is cheap enough to keep. (4) The 2026-08-15
  camera-footprint fold (above) is committed UNFLOWN; the flight is fullscreen,
  pitch 60, full orbit. If those folds are too heavy, the next lever is
  area-based coarsening (steps from the footprint's own area ratio, log base 7)
  in place of the PITCH_COARSE 35 threshold. Full record in
  `docs/mapterhorn-explorer-notes.md`.

None of the notebooks import anything from `archive/`: their only dependencies are the
third-party ones in their PEP 723 headers.

**The PMTiles reader and MVT decode are shared by copy, not by import.** The divisions
notebook's version was ported from the parked terrain notebook, the buildings notebook's
from the divisions one, and the HFP notebook is a whole-file fork of the divisions
notebook (its diff is the raster side only: CRS, scaling, zero handling, ramp). A fix to
the directory walk or the varint machinery in one of them should be carried to the others
by hand.

### What is in `archive/`, and what each one still proves

The five sections below moved here whole on 2026-08-13 (fire-risk buildings, the
HFP pair, flood, canopy);
paths now carry `archive/` in front. They are the newest layer of the archive and
their operational notes still apply wherever the maintained notebooks share code
with them by copy.

`archive/xsql-firerisk-buildings.py` folds CarbonPlan's 30 m CONUS wildfire-risk **Zarr v3
pyramid** to H3 and joins the cells onto **Overture building footprints**, so the map says
which real structures stand on high-risk ground. Full record in
`docs/firerisk-buildings-notes.md`. Four things to know before touching it:

- **Overture's buildings tileset carries attributes ONLY at z14.** Planetiler strips them
  below the top zoom, so at z13 every feature has `@geometry_source`, `@height_source` and
  nothing else. `id` is 100% present at z14 and 0% at z13, measured at four places. `id` is
  the dissolve key and the join key, so a coarser fetch silently returns thousands of
  anonymous polygons: the decode succeeds and nothing errors. The tile zoom is therefore
  pinned at 14 and the camera zoom only decides whether buildings are drawn at all.
- **The polyfill mode is `overlap`, not `center`.** A building is smaller than a res 11
  cell (150-250 m2 against 2,150), so it contains no cell centre and `center` returns
  nothing; `full` wants the cell inside the polygon and is worse. The reason `overlap` was
  rejected for divisions (counties tile the plane, so shared cells double count) does not
  apply, because buildings are disjoint islands.
- **Res 11 is the floor and the raster sets it**, same arithmetic as the NLCD notebooks.
  Res 12 would hold ~0.35 pixels and hole out. The polyfill cannot run finer than the fold,
  since both sides of the equi-join must be the same resolution.
- **Zero cells are KEPT**, the opposite of the deforestation notebook. There zero was ocean;
  here it is ground that will not burn and it is exactly where the buildings are.

`archive/xsql-hfp-divisions.py` is the deforestation notebook's machinery pointed at Vizzuality's
Global 100 m Terrestrial Human Footprint (HFP-100 v1.2, CC-BY 4.0, same source.coop
account, `vizzuality/hfp-100/hfp_<year>_100m_v1-2_cog.tif`, years 2017-2021; `YEAR` in
the constants cell is the seam a year slider would use). Full record in
`docs/hfp-divisions-notes.md`. Five things to know:

- **The COG is World Mollweide (ESRI:54009), not EPSG:4326.** The "no reprojection"
  simplification the deforestation notebook leans on does not hold. Both directions are
  closed-form spherical formulas on R=6378137 in the fold cell: forward (viewport box ->
  pixel window) needs a Newton solve on the parametric angle, inverse (pixel centres ->
  lat/lng for the H3 fold) is three arcsins. No pyproj. Pixels outside the Mollweide
  ellipse invert to |lon| > 180 and are masked before the fold sees them.
- **Values are the index x1000 in uint16, nodata 65535.** The tile reader must use
  `np.ma.filled`, not `np.asarray`, on the masked read: asarray silently drops the mask
  and a nodata coast would average in at score 65.5.
- **Zero cells are KEPT and sit inside the ramp**, not on a separate swatch: 36.7% of
  land scores exactly 0 and that is untouched ground, the bottom of a continuum. Ocean is
  NaN (65.7% of full-res tiles are unstored), so zero and no-data are distinguishable,
  which the deforestation COG never offered. The ramp is log1p over 0-40 on full-range
  cividis (measured: p50 1.0, p75 5.5, p99 23.4).
- **The overview pyramid AVERAGES**, verified the same way as the deforestation COG
  (mean survives an 8x downsample 15.135 -> 15.150 while the max collapses 51.2 -> 45.9),
  and the pyramid geometry is identical (100 m native, ten doublings), so
  `LEVEL_FOR_RES` carries over with one addition: the zoom ladder here is ONE STEP FINER
  than the deforestation notebook from zoom 4 up (`BASE_RES 5`, res 9 reading the
  full-res level at ~10 px per cell), but `MIN_RES` is 4, so fully zoomed out falls to
  res 4 (~70k cells from L6). It briefly ran with a res 5 floor and the ~475k-cell world
  view was visibly slow; that is why the floor is 4. `TILE_BUDGET` stays 512 MB because
  a wide view in the res 5 band still holds ~253 MB of L5 tiles on its own.
- **The viewport size is MEASURED, not assumed, and lonboard cannot tell you it.**
  `view_state` carries longitude/latitude/zoom and nothing about the canvas, so the old
  `VIEW_W`/`VIEW_H` guess (1400x620) was the only source of the fold box's size, and
  fullscreen broke it visibly: cells folded for a 620 px band across a 1500 px screen,
  ragged hex edges top and bottom. The fix lives in the Status widget: every widget
  shares the page document, so it finds the deck canvas (largest canvas on the page),
  watches it with a ResizeObserver plus `resize` and `fullscreenchange` listeners
  (fullscreening an ELEMENT fires no window resize), and syncs `view_wh` to the kernel,
  where `HOLD["wh"]` replaces the constants and a size jump beyond 25 px refolds the
  current view. The constants remain only as the seed for the opening fold and headless
  runs. The deforestation and fire-risk notebooks still carry the guess and the same
  fullscreen defect; port by hand per the shared-by-copy rule. Two browser facts the
  ruler had to learn, each a debugging round trip:
  - **marimo puts cell output in shadow DOM**, so `document.querySelectorAll("canvas")`
    finds nothing even with the map on screen. The search must recurse into every
    `shadowRoot`. A ResizeObserver works fine across the boundary once the canvas is
    found.
  - **The measurement crosses the bridge as a Unicode `"WxH"` string.** A
    `traitlets.List(traitlets.Float())` trait synced from JS never reached the kernel
    under marimo's anywidget bridge; the only trait types proven in these notebooks are
    Unicode (kernel -> browser) and Bool (browser -> kernel), so the ruler uses one of
    those. The on-screen diagnostics for all this (a px readout in the status line, a
    dim browser-side "ruler" line) live next to `set_status` and in `Status._esm`; they
    are currently ENABLED, because the fullscreen defect has been seen again since the
    ruler landed and is not yet closed. Comment them back out once it is.
- **The ranking is a button, not lonboard's draw-box tool.** "rank what's in view" in
  the Controls widget ranks the current view (`view_to_bbox`), so the camera is the only
  statement of intent. The trigger crosses the bridge as a Bool whose CHANGE is the
  click, per the proven-trait-types rule above. lonboard 0.16 renders its bbox-select
  toolbar unconditionally (the Map's `controls` trait governs only
  fullscreen/navigation/scale), so the Controls widget hides it with the same
  recurse-into-shadowRoots walk the ruler uses, on a 1 s interval because the map
  mounts later and can be rebuilt. The division display runs
  region -> county -> **locality** (locality above zoom 9.5, tile floor z10), and the
  ranking ladder starts at locality only when the box's own zoom is in that band, so a
  wide box never gets a towns-only answer.

`archive/xsql-hfp-conus.py` is the one-shot: the HFP fold with everything interactive cut away
(no camera, no divisions, no widgets, no cache), run once over a fixed `BOX` at res 7
from L2 and drawn as one static H3HexagonLayer. It exists for screenshots and as the
smallest runnable statement of the fold. `BOX` is the only knob and it scales hard: the
default lower-48 box is ~120M pixels, ~2 GB of RAM, 1.85M cells; the commented
North-America box is ~760M pixels, 15-20 GB of RAM, 4.88M cells (both measured). The
fold cell is a straightened-out copy of the interactive notebook's read cell, so fixes
to the sparse-tile check, the Mollweide pair or the fold SQL carry across by hand.

`archive/xsql-canopy-3d.py` draws Meta & WRI's High Resolution Canopy Height Maps (~1 m, uint8
metres, CC-BY 4.0, `s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/`) as
an extruded H3HexagonLayer: column height IS mean canopy metres times a stated 3x
exaggeration, colour (matplotlib Greens) repeats it. One dataset, one encoding; it is
the survivor of two parked pairings (see archive). Opens pitched over Prairie Creek
Redwoods. Things to know before touching it:

- **The CHM has NO overview pyramid, and that shapes everything.** 56,147 zoom-9 Web
  Mercator quadkey BigTIFFs, 65,536 px square, deflate + predictor 2 (inverted with a
  wrapping uint8 cumsum), and every strip is ONE ROW of 65,536 px, so any window read
  pays for full-width strips and there is no affordable wide view. The layer therefore
  hides below zoom 13 and folds per viewport above it, res 10 -> 12 on the standard
  ladder formula (read stride 4, relaxing to 2 at res 12). `CANOPY_BUDGET` refuses
  monster reads: dense-forest strips run ~16 KB compressed against a 320 B archive
  average, so the same viewport is 2 MB in most places and ~100 MB over Paradise CA.
- **The msk sidecars are IGNORED, measured, not assumed:** the Paradise mask reads all
  zero across rows carrying real 40 m heights and ~10k tiles have no sidecar, so
  GDAL's 0-is-invalid rule would nuke live data. No-data is an ABSENT quadkey tile
  (ocean, unimaged), which folds no cells; zero canopy is a real measurement and sits
  INSIDE the ramp.
- **The CHM reader is shared by copy** with the two parked canopy forks in `archive/`;
  the full dataset recon (IFD layout, strip sizes, quadkey math, mask finding) is in
  `docs/canopy-firerisk-notes.md`. Vintage is per Maxar acquisition (2018-2020, dates
  in `tiles.geojson`, not yet read); the model saturates in the tallest stands
  (measured max 57 m over real ~100 m redwoods).
- It carries the HFP ruler, a `SessionContext` instead of XarrayContext (no raster
  windowing, so no xarray), and no DuckDB (no polyfill, no dissolve).

`archive/xsql-flood-buildings.py` (EXPERIMENTAL, working but with open defects, see below)
draws FEMA NFHL flood zones from Carl Boettiger's PMTiles build and joins them onto
Overture divisions zoomed out (share of each county/locality inside the 1% floodplain)
and onto individual building footprints past zoom 13 (each coloured by the worst zone
its cells touch). One-line pitch: the fire-risk buildings notebook's question asked of
water instead of fire, with the raster replaced by vector polygons. Things to know:

- **Data:** `cboettig/hazard/flood-hazard.pmtiles` on source.coop (FEMA S_FLD_HAZ_AR,
  5.63M polygons, public domain, z0-13, layer `flood-hazard`, all attributes in the
  tiles at every zoom incl. `fid`, the dissolve key). Geometry is simplified ~10 m
  upstream, stated lossless at H3 res 10. The sibling `sea-level-rise.pmtiles` (NOAA
  5 ft inundation, 147 MB, layer `sea-level-rise`, field `slr_ft`) is the planned
  toggle; `HAZ_PATH`/`HAZ_LAYER` in the constants cell are the seam. Carl's
  precomputed `flood-hazard/hex/` is a real res-10 polyfill (unlike the NWI point
  index) but ships as 14 h0 partitions of ~850 MB each: kept as a correctness
  cross-check, not the live read path.
- **DuckDB is the ONLY engine**, a first for the repo. No raster means no fold, so
  DataFusion/h3ronpy's winning regime never occurs; polyfill, seam dissolve and the
  equi-joins are all duckdb-h3 + spatial. Three PMTiles archives (hazard, divisions,
  buildings) go through ONE parameterised reader instead of the usual copy-per-archive.
- **The joins run on two H3 ladders bridged by `h3_cell_to_parent`** (each cell has
  exactly 7 children, so counts are exact). Divisions polyfill `center` at
  `res_for_zoom`; zones fill `center` ONE step finer (`ZFINE = 1`; +2 exploded over
  coastal Louisiana, half of which is SFHA). Buildings fill `overlap` at res 11
  against zone cells at res 12, `MAX(zc)` = worst zone. `zc >= 2` is the SFHA line.
  Classes: 3 V/VE (orange), 2 A-family (blue), 1 shaded-X 0.2% (slate), 0 D (gray),
  -1 dropped at decode (X minimal, OPEN WATER, AREA NOT INCLUDED; painting "minimal
  hazard" would paint the country). Explicit set membership, NOT startswith("A"):
  FLD_ZONE's own vocabulary includes "AREA NOT INCLUDED".
- **The map cell and the wiring cell are SPLIT**, unlike the older notebooks:
  destroying a lonboard Map terminates deck's MODULE-LEVEL earcut worker pool, after
  which every polygon layer on the page fails to init until the browser reloads. The
  map cell depends only on imports/widget classes/seeds/HOLD and must never re-run;
  the wiring cell re-runs freely, unobserving old handlers via `HOLD["h_*"]` refs
  before re-observing. The camera survives edits (redraw targets `HOLD["vs"]`).
- **A Photon geocoder** (lonboard 0.16 `GeocoderControl`, stdlib urllib on a thread,
  camera-biased). Passing `controls` to `Map` REPLACES the default tuple, so
  fullscreen/navigation/scale are restated alongside it. Photon's `extent` is
  [minLon, maxLat, maxLon, minLat]; lonboard's bbox is (minx, miny, maxx, maxy).
- **OPEN DEFECTS, unresolved at commit time:** (1) re-running after an edit can still
  leave the map blank even after the cell split, mechanism not yet isolated; (2) data
  disappears on zoom-in in some sequences (suspects: the `_instant` tile-zoom
  fidelity check against `HOLD["ztz"]`, or a zone-memo coverage hit serving a stale
  regime). Headless export passes both regimes; the defects are interactive-only.
  Debug against the browser console; the earcut-pool cascade recorded above is what
  a dead deck looks like and is NOT evidence about which layer is at fault.


- `xsql-nlcd-zoom.py` folds and dissolves Annual NLCD entirely in DataFusion + h3ronpy.
- `xsql-duckdb-nlcd-h3.py` keeps the DataFusion fold and moves the dissolve to DuckDB's
  h3 extension. Measured on the same viewport: the fold is 70 ms in DataFusion against
  462 ms in DuckDB, and the dissolve is 75 ms in DuckDB against 928 ms in h3ronpy. The
  reason is which H3 lives underneath: duckdb-h3 wraps Uber's C library, h3ronpy wraps
  h3o. That benchmark is why the deforestation notebook splits the engines the way it does.
- `xsql-nlcd-imagery.py` is the DuckDB notebook with the hexagons switched OFF: only the
  dissolved boundary is drawn, as thin lines over an Esri World Imagery tile layer. The
  point is that the map becomes checkable rather than trustable, since the line either
  follows a real edge on the ground or it does not. It also reads NLCD one overview finer
  from res 7 up, because the boundary is decided by the cells where the class vote is
  closest and those were thinnest on evidence.
- `xsql-duckdb-terrain-h3.py` is the parked NLCD x terrain extrusion. See "Parked
  experiment" below; its PMTiles v3 client is what the divisions reader was ported from.
- `xsql-canopy-firerisk-buildings.py` is the fire-risk buildings notebook plus canopy
  height per building (mean over the footprint's cells widened one ring). Parked on
  MEANING: RPS's LANDFIRE fuel inputs already contain canopy structure, height is the
  wrong fuel axis (the Marshall Fire footprint scores canopy ~0), and the
  structure-survival literature finds spacing and materials dominate. What it still
  proves: the CHM strip reader end to end, and `docs/canopy-firerisk-notes.md` holds
  the whole dataset recon plus two unbuilt ideas (RPS-vs-CHM disagreement layer, the
  CAL FIRE DINS per-structure test).
- `xsql-canopy-deforest.py` is the deforestation notebook with a canopy paint switch
  at deep zoom, two ladders (deforest res 8 cap, canopy res 10-12) bridged by an
  `h3_cell_to_parent` join so each fine cell carries its coarse parent's cleared
  share. Parked as "comparing, not solving"; `docs/canopy-deforest-notes.md` records
  the way back (four-state cleared/regrown/intact map, still-bare permanence ranking,
  acquisition-date gate). Also proves the ruler port onto the deforestation base.
- `xsql-nlcd-sentinel2.py` is an empty placeholder from the abandoned Sentinel-2 render.
- The rest (`xsql-dem-*`, `xsql-naip-*`, `xsql-s1m-*`, `naip.py`, `overture_core.py`,
  `tools/patch_lonboard_surface.py`) are the earlier notebooks and helpers.

Paths in the sections below that name an archived notebook still resolve, with `archive/`
in front.

## Current project

`xsql-mapterhorn-explorer.py`, described above: the worldwide Mapterhorn DEM as extruded H3.
Its open items are in its own section's "OPEN DEFECTS AND NEXT WORK" bullet, and
the 2026-08-13 rework is waiting on an interactive flight. (The canopy notebook,
the previous current project, went to `archive/` on 2026-08-13; the pairing ideas
that were queued for it, terrain base under the columns, imagery underneath, are
recorded in its archived section and in `docs/imagery-and-terrain-notes.md`.)

Also live: `xsql-deforest-divisions.py`. The shape of it, in one line: **free-fly
the planet**, and everywhere the camera lands, the mean share of ground deforested
2002-2022 as H3 hexagons and as a number per administrative division.

- **Raster:** `deforest_100m_cog.tif`, 5.7 GB, EPSG:4326, whole globe, 100 m, from the
  source.coop repository <https://source.coop/vizzuality/lg-land-carbon-data> (Vizzuality
  / LandGriffon, CC-BY 4.0). Values are the **portion of each pixel deforested**, 0-1, so
  `mean()` is valid at every scale and the averaged overview pyramid is legitimate. The
  same repository holds nine other layers of the same shape and CRS: any of them is a
  one-line swap.
- **Boundaries:** Overture divisions, PMTiles, as above.
- **Engines:** obstore streams, DataFusion folds and joins, DuckDB does the two geometry
  steps (polyfill, tile-seam dissolve), lonboard renders.
- **Run:** `uv run marimo edit xsql-deforest-divisions.py --sandbox`

Everything from "Original brief" down is the 3DEP/NLCD lineage this grew out of. The
pipeline described there is now entirely in `archive/`, but the techniques (VRT as
catalog, obstore COG streaming, the H3 UDF, the lonboard gotchas) still apply and are why
those sections are kept.

## Original brief (3DEP; now archived)

A marimo notebook to **free-fly across the USA**: draw a box anywhere on a map, and
the app streams the **USGS 3DEP 10m (1/3 arc-second) seamless DEM** for that AOI
directly from the public `prd-tnm` S3 bucket with **obstore**, converts the elevation
raster to **H3** cells with a **DataFusion UDF**, and renders them as an **extruded
`H3HexagonLayer`** in **lonboard**. No tiling server, no pixels leave object storage
until the AOI asks for them.

**Division of labor:** Python resolves *which* COGs cover the AOI and streams only the
overviews it needs; DataFusion + h3ronpy do the H3 aggregation as a SQL UDF; lonboard
(deck.gl) does the 3D render. Keep the notebook Python/SQL-heavy and the JS thin.

## The pipeline (end to end)

1. **AOI picker** (draw-box). lonboard `Map`, observe `selected_bounds`, push
   `[W, S, E, N]` into `mo.state`. Pattern: the lonboard NYC-taxi marimo example, and
   `deck-terrain-naip-marimo/naip_terrain_viewer.py` (the `picker.observe(... names=
   "selected_bounds")` cell). Free-fly the USA, not a fixed AOI.

2. **Catalog = the VRT, not a STAC API.** USGS publishes a nationwide VRT that lists
   every 1-degree seamless COG on `prd-tnm` with its exact placement, so parsing it
   once turns AOI -> hrefs into a local bbox intersection. No STAC API, no signing.
   URL: `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt`
   Parse each `<ComplexSource>` (SourceFilename minus `/vsicurl/`, DstRect + GeoTransform
   -> degree bbox). Reference: the `dem13_tiles` cell in `naip_terrain_viewer.py`.

3. **Stream the COGs with obstore.** `S3Store(bucket="prd-tnm", region="us-west-2",
   skip_signature=True)`, then `async_geotiff.GeoTIFF.open(path, store=store)`, read an
   overview, `.as_masked()`, honor `nodata`. Reference: the `_read_tile` cell in
   `3dep-seamless-duckdb-h3/s1m_viewer.py`. Read a coarse overview whose resolution
   sits at or below the chosen H3 cell size; do NOT pull full-res pixels.

4. **Raster -> H3, in SQL.** For each valid pixel derive (lat, lng, elevation) from the
   COG geotransform, then aggregate with the DataFusion UDF already in
   `archive/xsql-dem-h3.py`: `h3_latlng_to_cell(lat, lng, res) -> UBIGINT`. Group by cell,
   aggregate elevation (mean/min/max). This is the whole reason the repo exists (the
   `x-sql` / xarray-sql + DataFusion angle).

5. **Render extruded H3.** lonboard `H3HexagonLayer(get_hexagon=table["hex"],
   get_elevation=table["elevation"], extruded=True, high_precision=True)`, elevation
   scale + opacity controls, `DarkMatterNoLabels` basemap. Reference: the layer cell in
   `3dep-seamless-duckdb-h3/naip_usgs_join_h3_1m.py`.

## H3 UDF (already present)

`archive/xsql-dem-h3.py` registers the DataFusion UDF via h3ronpy:

```python
from h3ronpy import cells_to_string
from h3ronpy.arrow.vector import coordinates_to_cells
h3_cell = udf(latlng_to_cell, [pa.float64(), pa.float64(), pa.int32()],
              pa.uint64(), "stable", name="h3_latlng_to_cell")
ctx.register_udf(h3_cell)
```

Confirm the exact h3ronpy import path against the installed version before relying on
it (the module layout has moved between releases).

## Imagery, and why it is a tile layer

`xsql-nlcd-imagery.py` draws its imagery with a **`BitmapTileLayer`** pointed at Esri World
Imagery. That is a retreat, and the reasons are worth knowing before anyone "improves" it.

It first read Earth Genome's Sentinel-2 mosaics off source.coop, which had a real argument
behind it: same year as the land cover, so a disagreement between the line and the ground
could only be classification error. **The data side of that was fully solved** and is
written up in `docs/imagery-and-terrain-notes.md`. The render side never became stable
through two separate architectures. A `BitmapTileLayer` is the one imagery path here that
always worked, because it is what already draws the place labels.

- **URL is `{z}/{y}/{x}`.** Esri puts row before column, the opposite of the Carto labels
  URL two layers above it in the same cell. Swapping them serves imagery from the wrong
  place rather than 404ing, so it looks like a projection bug.
- **The cost, and it is not small:** Esri World Imagery is a mosaic of many sources and
  dates that vary by location, so the same-vintage rule is gone. A boundary can disagree
  with the photograph because the ground genuinely changed, and nothing on screen tells
  you which. Fine for judging whether a forest edge is roughly right. Not evidence about
  a particular year. If that distinction ever matters, the Sentinel-2 data path in the
  notes is the way back.

**Fill is an alpha, not a flag.** `filled` decides whether deck builds a fill sublayer at
all, and flipping it after init does not reliably make one appear. That is the real "the
fill button does nothing" bug, and re-pushing the table does not fix it either. Keep the
layer permanently `filled=True` and switch `get_fill_color` between the class colours and
a transparent constant.

## Parked experiment (read before rebuilding it)

`xsql-duckdb-terrain-h3.py` joins NLCD to Mapterhorn terrain on the H3 cell id and extrudes
the hexagons. The join works and the numbers are right. It was abandoned on looks: height
is a weak encoding for a categorical map, and the extrusion buries the dissolved outlines,
which are the good part of this repo. Full account and the Mapterhorn/PMTiles findings are
in `docs/imagery-and-terrain-notes.md`. Its PMTiles v3 client outlived it: the divisions
notebook's reader is that code, ported.

Other things from that work that cost a session each and apply anywhere in this repo:

- **A `BitmapLayer` with `image=""` aborts deck's entire update pass**, because deck
  initialises all layers in one pass and a throw anywhere kills the batch. The symptom is
  a cascade of assertions naming perfectly healthy layers. An assertion naming a layer is
  weak evidence that the layer is at fault.
- `RasterLayer.from_geotiff` ships with `min_zoom`/`max_zoom` **commented out** in
  lonboard 0.16, and its fetcher indexes `images[len - 1 - z]`, so an out-of-range zoom
  wraps negative onto the full-res image. Pass both through `**kwargs`.
- An async-geotiff `Tile` carries pixels on **`.array`**, not `.data`, and the render
  callback runs where an `AttributeError` is silent.
- `line_width_units` defaults to **metres**; set it to `"pixels"` or width is
  `max(1 metre, line_width_min_pixels)` and can never go below the floor.
- `VIEW_W` is a guess and `VIEW_H` is not. Bitmaps expose that; hexagons hide it.
- Sliders that commit on `input` send one comm message per drag pixel. Fine for a few
  floats; not for anything that re-dissolves. And 12 stops across a 4.5rem track is ~6 px
  per stop, inside the slop of a trackpad drag.

**Directed edges were measured as a replacement for the polygon dissolve and lost**
(17.3 ms / 0.110 MB against 30.2 ms / 0.412 MB even with a despeckle in front). `ST_Dump`
is the prize, not the polygons. Do not re-propose without re-reading those numbers. The
stale comment about outlines "bulging outward" was from the h3ronpy era; `WASH_SQL`
dissolves at native resolution and the outline is exact.

## The fold is the H3 UDF inside DataFusion (hard requirement)

The point of this repo, in Stephen's words (2026-08-17): the H3 fold as an h3ronpy
UDF called INSIDE the DataFusion SQL, `h3_latlng_to_cell(lat, lon, res)` in the
`SELECT ... GROUP BY`, "the gold star of this whole repo, what I've been working on
for weeks". Never move it out of the SQL for a saving. `xsql-hrrr-heat-hex.py`
shipped for one day with the cell ids precomputed in Python and joined in as a
static lookup (carried over from the counties film, where it still is); that was
wrong for this repo and was put back into the query the moment he saw it. Python
may precompute cells for a MASK or a pruning predicate; the fold itself is the UDF.

## Colorblind-safe rendering (hard requirement)

Stephen has trouble seeing RED (protan-type). Never encode anything on a red-green
axis or in red alone; green by itself is fine, and he reads mono-green luminance ramps
(matplotlib Greens) without trouble. Default to **viridis / cividis** or single-hue
luminance ramps (viridis is already the choice in `s1m_viewer.py`) and lean on
**extrusion height** as a redundant, non-color cue.

## Environment

```bash
# Dev (full venv)
uv run marimo edit xsql-mapterhorn-explorer.py

# Shareable sandbox (PEP 723 inline deps in the notebook header)
uv run marimo edit xsql-mapterhorn-explorer.py --sandbox

# Headless smoke test (runs every cell, no browser)
uv run marimo export html xsql-deforest-divisions.py -o /tmp/out.html

# An archived notebook, from the archive's own environment
uv run --project archive marimo edit archive/xsql-nlcd-imagery.py
```

**Two pyprojects, deliberately.** The root `pyproject.toml` is the union of the two
maintained notebooks' PEP 723 headers and nothing more, so it stays honest about what is
actually imported (`async-geotiff` for the COG reader, `pillow` for the terrarium WebP
decode). `archive/pyproject.toml` is the union of every archived
notebook's header (adds `aiohttp`, `arro3-io`, `geoarrow-rust-io`, `geopy`, `morecantile`,
`palettable`, `pillow`, `planetary-computer`, `pyproj`, `pystac-client`, `shapely`) and
is pinned, because a frozen environment is the point of an archive. Keep each notebook's
PEP 723 header in sync with whichever pyproject covers it, so `--sandbox` stays
self-contained either way. Pin the deck.gl-raster / lonboard versions; they move fast.

`duckdb` carries `INSTALL spatial` plus `INSTALL h3`, and in the deforestation notebook it
does exactly two jobs, both geometry: the polygon-to-cells polyfill and the `ST_Union_Agg`
that removes tile seams. It is not a second query engine for the fold; that stays
DataFusion.

`.cache/` is gitignored and disposable: everything in it is fetched at runtime (the 3DEP
VRT, the S1M GeoPackage, the Overture GeoParquet file-bbox indexes) and all of it belongs
to archived notebooks. The deforestation notebook writes nothing to disk, so a fresh clone
has no `.cache/` at all and never grows one.

### Required for every SurfaceLayer notebook

```bash
uv run python archive/tools/patch_lonboard_surface.py   # re-run after ANY install
```

Without it the textured mesh comes back covered in pale quadrilateral facets. lonboard's
`SurfaceLayer` sends no `NORMAL`, and deck's `SimpleMeshLayer` responds to that with
`flatShading: !hasNormals` rather than by skipping lighting: one derived normal per
triangle, lit by the default material. On a colour ramp it passes for texture; on a NAIP
photograph it is a herringbone of translucent facets over the imagery, and **no notebook
parameter can reach it**. The script injects `material: false` into the shipped JS bundle,
so `uv sync`, a lonboard upgrade and `--sandbox` all revert it. Hard-reload the browser
after running it; the widget JS is cached client side and a kernel restart is not enough.
Full account in `docs/xsql-naip-drape-notes.md`.

Layer `parameters` must use luma v9 names: `depthCompare` and `depthWriteEnabled`, not the
WebGL-1 `depthTest`, which deck reads as nothing and silently leaves depth disabled.

## Overture Maps (`archive/overture_core.py`)

Shared helpers for streaming Overture GeoParquet out of `overturemaps-us-west-2` with
obstore. `THEMES` names every theme/type pair, so asking for several at once is a list.

Two things that are not obvious and cost a session each:

- **`GeoParquetDataset.open` refuses the buildings theme.** The geometry column is Polygon
  in some parts and MultiPolygon in others, and a dataset wants one type. Read per file
  (`load_parts`), or take WKB and concatenate (`load_wkb`).
- **The file index is the whole performance story.** A theme is ~512 files of ~500 MB with
  no catalog; `file_index()` reads their GeoParquet footers once (~100 s) and caches the
  bboxes in `.cache/`, after which an AOI reads only the overlapping file (~1.4 s). The
  alternative, a DuckDB `read_parquet` with a bbox predicate, is ~35 s on every query.

`OVERTURE_RELEASE` is pinned and Overture deletes old releases (`2026-01-21.0` is already
gone). `releases()` lists what is live. Building `height` is present on roughly 55-75% of
footprints depending on the city; `num_floors * 3 m` is the only other source.

**Overture also publishes each release as PMTiles**, one archive per theme, in
`overturemaps-extras-us-west-2` under `tiles/<release>/` (the old
`overturemaps-tiles-us-west-2-beta` bucket is dead: `AllAccessDisabled`). Anonymous ranged
GETs, gzipped MVT, z0-12. For any read that is per-viewport rather than per-attribute,
prefer the tiles: they are the vector twin of a COG overview pyramid, and the GeoParquet's
missing spatial layout is exactly what they fix. Subtype minzooms are baked in by the
build (divisions: country z2, region z4, county z8, locality z10, measured not
documented), properties are flattened (`@name` is the primary name), and geometry arrives
tile-clipped, so pieces need a per-id dissolve before they are drawn. The divisions
notebook has the working reader and decoder to copy.

## Reference repos (reuse, do not rebuild)

- `deck-terrain-naip-marimo/naip_terrain_viewer.py` — VRT-as-catalog parse, draw-box
  AOI picker, `selected_bounds` -> `mo.state`.
- `3dep-seamless-duckdb-h3/s1m_viewer.py` — obstore + async_geotiff COG streaming from
  `prd-tnm`, viridis DEM layers.
- `3dep-seamless-duckdb-h3/naip_usgs_join_h3_1m.py` — extruded `H3HexagonLayer`, H3 UDF
  usage, elevation/opacity controls.

## Memory

Per global rule, running notes live in `.claude/memory/MEMORY.md` (gitignored), not the
auto memory path.
