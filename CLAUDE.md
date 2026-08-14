# CLAUDE.md

Guidance for Claude Code working in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory location, colorblind-safe encodings).

## Repository layout

**Two interactive notebooks are the repo**: deforestation divisions and the terrain
3D experiment, plus one maintained one-shot (`xsql-deforest-conus-counties.py`, the
deforestation fold as a static CONUS county choropleth, below). Everything else is in `archive/`, kept for reference and not
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
- **Pitch eats the padding AND the resolution.** `_pad` extends the fold box toward
  the horizon along the bearing (anisotropic, unlike the parked notebook's symmetric
  9x overread), and pitch >= 35 folds one H3 step coarser to pay for it. `_same_view`
  includes pitch/bearing, and coverage is gated by `_cam_ok`, or tilting past the
  folded trapezoid leaves a band of missing cells at the horizon (seen in fullscreen
  over Tibet before the fix).
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
  decide whether relative colors is cheap enough to keep.

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
