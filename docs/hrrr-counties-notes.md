# HRRR temperature by county, animated: feasibility notes (2026-08-15)

Stephen's idea: xarray-sql over dynamical.org's HRRR, folded to H3 res 7 over CONUS,
DuckDB polyfill of Overture counties, temperature per county as an ANIMATED time
series map, tried in marimo first (it may end up a browser app). This is the poking
session, nothing built yet. Everything below is measured unless marked otherwise.

## The data, two archives, both anonymous, both CC-BY 4.0

dynamical.org republishes NOAA HRRR (3 km, CONUS, hourly) as Zarr. Grid is 1799 x 1059
= 1,905,141 pixels, Lambert conformal conic (sphere R=6371229, lat0 38.5, lon0 -97.5),
projected `x`/`y` in metres plus 2-D `latitude`/`longitude` float32 coords (8 MB each)
in the store, so THE H3 FOLD NEEDS NO PYPROJ: lat/lon come off the store. 27 variables
incl. `temperature_2m` (degC), dew point, wind u/v 10 m and 80 m, gust, radiation
fluxes, composite reflectivity, categorical precip types, precipitation. Both archives
carry `spatial_ref` with the full WKT and a GeoTransform.

**48-hour forecast** (the source.coop one Stephen pointed at):
`s3://us-west-2.opendata.source.coop/dynamical/noaa-hrrr-forecast-48-hour/v0.1.0.zarr`,
PLAIN Zarr v3, sharded, opens with obstore + zarr `ObjectStore` exactly like the rest
of the repo (no icechunk dependency, no s3fs). Dims `init_time` (every 6 h since
2018-07-13, 11,821 inits at probe time, latest 12:00 UTC today, so ~5 h behind) x
`lead_time` (49 hourly steps, 0-48 h) x y x x. Chunks `(1, 49, 265, 300)`, shards
`(1, 49, 1060, 1800)`, blosc zstd. Consolidated metadata opens in ~1 s.
Measured: latest init, ALL 49 leads, whole CONUS: **2.2 s, 373 MB float32**. One lead
alone costs the same 2.3 s (the shard is the unit), so take all 49.

**Analysis** (the hourly history; NOT on source.coop, only in the AWS Open Data
bucket): `s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.2.0.icechunk/`, icechunk v2,
`icechunk.s3_storage(..., anonymous=True)` + `xr.open_zarr(session.store,
consolidated=False)`, opens in 1.9 s. Dims time (hourly since 2014-10-01, 104,079
steps, latest 14:00 UTC today, ~3 h behind) x y x x. Chunks `(2160, 45, 45)`, shards
`(2160, 540, 450)`: TIME-OPTIMISED, one chunk is 90 days deep and 45 px square, so
the smallest read of a pixel is its whole 90-day history and one CONUS map costs
960 chunks whatever the time window. Measured: one chunk 1.2 s; 120 chunks 5.4 s
(~22 chunks/s); through xarray-sql lazily (see below) 24 h x CONUS 18.7 s, 240 h
x CONUS 20.5 s. Fetch-bound; the window depth is nearly free up to the chunk depth.
Also on source.coop under `dynamical/`: `noaa-hrrr-forecast-18-hour-virtual`,
`noaa-hrrr-forecast-48-hour-virtual` (icechunk, virtual refs into NOAA GRIB, not
probed), and `noaa-mrms-conus-analysis-hourly` (radar precip, plain zarr).
The "data.dynamical.org access ends 2026-08-31" notice on their pages is about their
old HTTPS host; the S3 paths above are the live ones.

## The fold, and what res 7 means here

Res 7 hexes average 5.16 km2; an HRRR pixel is 9 km2. Measured on one lead:

| res | cells over the grid | px per cell |
|-----|--------------------:|------------:|
| 7   | 1,905,141           | 1.00        |
| 6   |   457,886           | 4.16        |
| 5   |    65,807           | 28.95       |

So at res 7 every pixel lands in its own cell and the "fold" is a relabel, not an
average; ~40% of the res 7 county cells (1,475,007 from the counties one-shot's
polyfill) hold no pixel at all. That is fine for the county mean (each pixel is
counted once, county mean = mean of its pixels) but it means res 7 is finer than the
data. Res 6 (4 px/cell) is the ladder's honest row if the hexagons are ever DRAWN;
for a county choropleth the res only decides which pixels count for which county.

xarray-sql detail: a Dataset with `t(lead, y, x)` and `lat(y, x)`/`lon(y, x)` is
registered as TWO tables (`df.lead_y_x`, `df.y_x`; that is `from_dataset`'s mixed-dims
rule), so the fold is a join on (y, x). Cheaper still, and what the probe settled on:
compute `hex` per pixel ONCE with h3ronpy from the store's lat/lon (1.9M cells,
sub-second), polyfill the counties in DuckDB (`h3_polygon_wkb_to_cells_experimental`,
'center', 2.9 s), join hex -> county id, and keep a static `pix2c(y, x, id)` table of
879,420 pixel rows. Then the per-frame SQL is one join + group by, county x hour,
straight from the cube:

```sql
SELECT lead, id, avg(CAST(t AS DOUBLE)) AS t, count(*) AS px
FROM df JOIN pix2c USING (y, x) WHERE t = t GROUP BY 1, 2
```

Measured, forecast: 49 leads x CONUS -> 152,243 county-hour rows, read 2.5 s +
aggregate 0.7 s. Analysis, lazy (`xr.open_zarr(..., chunks={time:N,y:45,x:45})`,
`from_dataset(chunks=...)`, no dask needed, DataFusion pulls the blocks in parallel): 24 h ->
74,568 rows in 18.7 s; 240 h -> 745,680 rows in 20.5 s; the full 2,160 h chunk
depth (90 days, 4.1 billion pixel-hours through DataFusion) -> 6,711,120 county-hour
rows in 149 s.

Coverage: 3,107 of 3,108 CONUS counties catch at least one pixel; Lexington VA
(6.5 km2) catches none. The VA independent cities sit on 1-2 pixels (Galax, Radford,
Falls Church, Fairfax, Manassas Park at 1). A county on one pixel is that pixel's
temperature; the choropleth is honest about it if `px` rides along.

Counties themselves: the counties one-shot's PMTiles decode + DuckDB dissolve
(1,008 z8 tiles, 6,815 pieces -> 3,108 counties, 7.5 s fetch, 7.7 s with dissolve),
by copy. Running that notebook headless via `app.run()` takes 16.6 s and hands back
the `counties` table; that is how the probe got them.

## The animation: three routes, in marimo

The county-hour table is small (49 frames x 3,108 counties = 152k values; 240 frames
= 746k; a full 90-day hourly = 6.7M values, 27 MB float32). The whole question is
where the frame loop runs.

1. **lonboard PolygonLayer, kernel-driven frames.** One PolygonLayer of the 3,108
   dissolved counties (already built in the one-shot); per frame the kernel sets
   `get_fill_color` to a 3,108 x 4 uint8 column (~12 KB, the `_refill` pattern from
   the divisions notebook), driven by an asyncio task with a play/pause Bool and a
   frame slider in the repo's Controls-style anywidget. Zero new JS, everything is
   proven tech in this repo. Cost: one comm message per frame; expect ~5-10 fps and
   the kernel busy while it plays. Unmeasured; the first flight would measure it.
   Click for a county's series: `Map.on_click` gives lon/lat, DuckDB
   `h3_latlng_to_cell(lat, lon, 7)` against the county cells names the county, and a
   matplotlib line (or `mo.ui.altair_chart`) draws its 49/240 hours. Not plotly:
   not in the env.
2. **A bespoke deck.gl anywidget, browser-driven frames.** Ship the county polygons
   once (GeoJSON or binary) and ALL frames' values once as one Float32Array (600 KB
   for 49 frames, 27 MB for 2,160), and let JS own the clock: requestAnimationFrame,
   ramp in a shader or a per-frame `getFillColor` update from a typed array, a
   scrub slider, and a hover/click series. Smooth (60 fps), scrubbable, and it is
   what "this might be a better browser app" turns into, except it lives inside a
   marimo cell. Cost: real JS, and deck.gl loaded from esm.sh in the widget's ESM
   (fine in local marimo). Bridge caveat from the HFP ruler work: only Unicode and
   Bool traits are PROVEN across marimo's anywidget bridge; a bytes/DataView trait
   kernel -> browser is untested here (anywidget supports it) and base64-in-Unicode
   is the fallback (49 frames trivial, 2,160 frames 36 MB, still workable).
3. **plotly `px.choropleth` with `animation_frame`.** Twenty lines, built-in play
   button and slider, marimo renders it. Known to crawl on thousands of geojson
   shapes per frame (it redraws the shapes each frame) and plotly is not in the env.
   The fallback if 1 and 2 are both too much for a first look, not the plan.

Not viable: lonboard's `DataFilterExtension` trick (all frames as rows, filter on
GPU) because it duplicates the county geometry per frame (49x, 2,160x); a per-frame
colour texture indexed by a uniform would fix that but not through lonboard.

## The choices left open

- Forecast (49 h ahead, 2 s read, refreshes 4x/day) versus analysis (any 90-day
  window of the last 12 years, ~20 s read for up to ~10 days, 149 s for the
  full 90-day chunk depth). Different films: "the next two days" versus
  "the last ten days" or "August 2023's heat dome, hour by hour". The pipeline is
  identical, so a `SOURCE` seam is cheap.
- Hourly frames versus daily means (or daily max) for a long window: 90 frames of
  daily max reads as weather; 2,160 hourly frames mostly show the diurnal pulse.
- Res 7 as asked (relabel; honest per-pixel counting) versus res 6 (a real fold, 4
  px/cell) if hexagons are ever drawn under the counties.
- Fill ramp: temperature wants a diverging or perceptual ramp; per the CB rule
  cividis/viridis for a sequential film, blue-yellow-orange (no red leg) if diverging
  around a fixed pivot; the pivot and the fixed span (frames must share one ramp or
  the animation lies) are decisions.
- Route 1 first as the cheapest measurable thing, or route 2 straight away because
  animation is the point. Stephen's call.

## Build log, route 2 (2026-08-15, later the same day)

Built as `xsql-hrrr-counties.py`; headless passes (store 2 s, counties 7.6 s,
polyfill + lookup 1.5 s, fold 19.9 s, 168 x 3,108 frames). Two lessons from the
first flight and the build:

- **No dask.** `xr.open_zarr(..., chunks=None)` leaves the store lazily indexed;
  xarray-sql cuts it into blocks itself and DataFusion pulls them in parallel:
  20.9 s for 168 h either way. The block must span the whole time window (a
  narrower grid decodes each 2,160 h store chunk once per block: 111.7 s).
- **esm.sh resolves caret ranges to the newest release**, so pinning deck at 9.1.14
  left `geo-layers -> mesh-layers@^9.1.0` on 9.3.x, which imports `phongMaterial`
  from a 9.3 core: `SyntaxError: The requested module '/@deck.gl/core@9.1.14/...'
  does not provide an export named 'phongMaterial'`. Pin every deck package to the
  newest (9.3.10) and pin `?deps=` on every import so the variant hashes agree.
  Verified by crawling the module graph from the notebook's five import URLs:

```python
import re, urllib.request, collections
entries = [...]  # the five import URLs from CountyFilm._esm
seen, q, mods = set(), collections.deque(entries), collections.defaultdict(set)
while q:
    u = q.popleft()
    if u in seen: continue
    seen.add(u)
    s = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})).read().decode()
    for m in re.findall(r'(?:from|import)\s*"(/[^"]+)"', s) + re.findall(r'export \* from "(/[^"]+)"', s):
        pkg = re.match(r'/((?:@[^/]+/)?[^/@?]+)@([^/?]+)', m)
        if pkg and "/es2022/" in m: mods[pkg.group(1)].add(m)
        if "https://esm.sh" + m not in seen: q.append("https://esm.sh" + m)
for k in sorted(mods):
    if k.startswith(("@deck.gl", "@luma.gl")): print(k, len(mods[k]), sorted(mods[k]))
```
  Expect exactly one concrete module per `@deck.gl/*` package and per `@luma.gl/*`
  package (webgl has three files, one package). Result at 9.3.10 + apache-arrow
  18.1.0: 200 modules, all singletons.
- Also seen on that flight, cause not yet isolated: marimo's `Error: Model not found
  for key: <id>` in the console. Likely the widget model dying with the import error;
  if it persists after the pin fix it is a separate defect.

## Second build pass (2026-08-15, evening)

Stephen's asks after the first flight (map rendered, film played, esm.sh pin fixed):
more map controls, live stats, custom date range with limits, click did nothing,
fullscreen for the map (deck's element, not marimo's), and then "the dashboard
should all be on the map with a way to hide, this all full screen".

- HUD overlays inside the deck element (title, legend + buttons, transport, left
  drawer with stats / series / warmest-coolest / settings), ⊟/H hides, ⛶/F is
  `mapEl.requestFullscreen()`. Deck polls canvas size each frame, so fullscreen
  needs no resize handler; the inline 620 px height is overridden by
  `.map:fullscreen { height: 100vh !important }`.
- Stats are per-frame in JS from the Float32 matrix: sort the frame's finite values
  once (cached per frame index) for median, p10/p90, warmest/coolest 8, share over
  the pivot; county mean per frame precomputed once per film for the chart's
  no-selection envelope. hyparquet/hightable were considered and not used: the
  matrix is already a typed array in the page and hightable is a React component.
- Explicit picking on pointerup (`deck.pickObject`, 4 px drag threshold, prefix
  layerIds) replaces deck's `onClick`, which did not fire on the first flight.
- Window form: `mo.ui.date_range` + mode dropdown in a `.batch().form()`, so
  nothing refetches until "load window". Limits hourly 14 days / daily 92 days
  (`mo.stop` with the reason). Fold memoised on the window in `HOLD`.
- Forecast source: lead offsets are rewritten to valid times on `t`, so labels read
  as clock time and the same frames cell serves both sources.
- Headless: 159 h (7 UTC days to the newest hour) in 19.0 s, 494,013 rows.

## Third pass: minimal (2026-08-15, night)

The fuller HUD flew: map and film fine, but the top-right buttons (hide / home /
fullscreen) "freeze the notebook and exit fullscreen", the hide button was not on the
panel it hides, and the drawer was "dizzying". Stephen: "want minimal".

What was cut: the drawer (stats tiles, warmest/coolest lists, settings), the home
button, opacity/lines/labels/loop toggles, the `clicked` trait. What stays: one panel
(title, legend, county mean, clicked county's line, hide toggle ON the panel), the
transport bar (with fps and fullscreen).

Diagnosis of the freeze, best guess: button presses also reached the pointerup pick
handler, selected the county under the button, and `model.set("clicked")` +
`save_changes()` went to the kernel; marimo re-runs cells that reference a displayed
widget whose value changed, and that pulled the fullscreen element out of the DOM
and rebuilt deck. Two fixes regardless of whether that is the whole story: no trait
ever crosses back from the widget, and a press that starts on the HUD is not a pick.
Also fixed: the hide toggle's selector (`.cf .hidden` matched nothing; `.cf.hidden`).
Data volume is not a suspect: the whole film is 2 MB.

**Found it (same night):** the freeze on "hide" was the class name. The toggle put a
class literally called `hidden` on the widget root; marimo's page CSS is Tailwind,
which defines `.hidden { display: none }`, and the widget is not isolated from page
styles, so the whole widget went `display: none`: map gone, fullscreen element gone
(the browser exits fullscreen), notebook "frozen". Both flights that froze had that
class. The `Model not found for key` console line is present from page load and is
unrelated. Fix: every widget class is now `cf-` prefixed and the states are
`cf-collapsed` / `cf-picked`.
