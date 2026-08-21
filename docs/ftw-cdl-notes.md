# Fields of the World x CDL: recon (2026-08-20)

Question: is a CDL notebook with FTW (Fields of the World, `source.coop/ftw/global-data`)
overlaid worth building. Everything below is measured from home unless marked.
Bench script: `xarray-sql-multi-backend-test/bench_ftw_cdl.py` (rc venv).

## What the repo is

CC-BY 4.0, Taylor Geospatial + Microsoft AI for Good (Robinson et al. 2026,
arXiv:2605.11055). PRUE U-Net over Sentinel-2 planting/harvest median composites,
two independent years (2024, 2025), 10 m. Public HTTP base
`https://data.source.coop/ftw/global-data/` is physically
`s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/` (anonymous).
Paths with `admin:country_code=` in them 403 over the HTTPS base; S3 works.

Four collections:

| collection | format | notes |
|---|---|---|
| `predictions/vectors/alpha/results-by-admin-conf/admin:country_code=US/US_<ST>.parquet` | GeoParquet (fiboa), one file per state, 49 US partitions, 22 GB total (CA 629 MB, IA 611, TX 1.4 GB) | columns id, geometry (CRS84), bbox struct, metrics:area/perimeter, determination:datetime (2024-01-01 / 2025-01-01 UTC; both years in ONE file), confidence |
| same dir, `US_<ST>.pmtiles` | tippecanoe z0-13, layers "2024" and "2025", only attr metrics:area, decimated below z13 (`drop-densest-as-needed`), no id | draw-only, no join key |
| `predictions/confidence/...` | 500 m COGs | sparse; see confidence below |
| `predictions/zarr/alpha/global.zarr` | Zarr v3, EPSG:4326, (time 2, band 3, y 1,566,049, x 4,007,517) float32 softmax [non_field, field, boundary], root chunks (1,3,8192,8192) sharded with (1,1,2048,2048) inner zstd; 14 multiscale groups 2x..8192x (4x: 4096 shards, 512 inner) | the thing the vectors are thresholded from at 0.5 |

A "field" is a remote-sensing connected component, not a parcel.

## Confidence is NULL for the whole US

`count(confidence)` is 0 of 1.18M in IA, 1.10M in KS, 514k in WA, and 0 in the
Fresno window of CA. The US is not among the 24 FTW-labelled countries, so the
modeled-confidence layer does not cover it. The per-polygon confidence column, the
"confidence-2024" styles and the `>= 69` filter are all no-ops here. Do not build on it.

## Which data, from where, what type (all of it through DuckDB)

| what | where | type on disk | how DuckDB sees it | role |
|---|---|---|---|---|
| CDL crop type, 2008-2025, 30 m (+ 10 m 2024-25), block-majority pyramid | `s3://us-west-2.opendata.source.coop/chill/usda-cropland-data-layer/v0.1.0.icechunk` (`https://data.source.coop` endpoint), EPSG:5070 | icechunk Zarr v3, uint8 `crop_type(year, y, x)` | `xql.register(con, "cdl_<k>", ds)` on the 0.4.0rc1 xarray-sql DuckDB backend: a pushdown pyarrow dataset, one table per pyramid level, columns year/y/x/crop_type | the crop label and the 18-year history |
| FTW field polygons, 2024 + 2025 | `s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/vectors/alpha/results-by-admin-conf/admin:country_code=US/US_<ST>.parquet`, CRS84 | GeoParquet (fiboa), one file per state, both years in one file | `read_parquet()` through httpfs + spatial, `bbox` struct predicate prunes row groups | the field unit (the frame for the history) |
| FTW per-state PMTiles | same dir, `US_<ST>.pmtiles` | tippecanoe z0-13, no id | not used (draw-only, decimated, no join key) | none |
| FTW softmax probabilities, 2024 + 2025, 10 m, 14 levels | `.../tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr` (+ `/4x` .. `/8192x`), EPSG:4326 | plain Zarr v3 (NOT icechunk), sharded, float32 `variables(time, band, y, x)`, bands non_field_background / field / field_boundaries | `xql.register(con, "ftw_<k>", ds, chunks=inner chunk)` on the same backend, columns time/band/y/x/variables | P(field): the cropland-at-all disagreement |
| FTW 500 m confidence | `.../predictions/confidence/` COGs | COG | not used | none (NULL over the US) |
| Overture divisions (counties) | `overturemaps-extras-us-west-2/tiles/<release>/divisions.pmtiles` | PMTiles MVT | hand-rolled reader, DuckDB dissolve | later, for county stats |

Measured, FTW Zarr through `xql.register` on DuckDB, Fresno 20 x 20 km, 3 bands:
block = the INNER chunk (512 at 4x/16x, 2048 native): 4x **1.2 s**, 16x **0.9 s**,
native **13.9 s**; block = the shard (4096/8192): 19.5 / 17.3 / 62.6 s (the block
expands whole, the x/y predicate cannot prune inside it). Same means as the direct
zarr read to 1e-12. So register fine levels at inner-chunk blocks, never shard-sized.

## Measured, Fresno 20 x 20 km window (-119.9..-119.7, 36.6..36.8)

- FTW fields from `US_CA.parquet` with the bbox-struct predicate: **2.8 s**, 2,110 fields
  (2024) + 2,188 (2025), mean 18 acres, ~1 MB WKB per year. Row groups (48 in CA) are
  roughly spatially sorted, so the predicate prunes. A 60 x 50 km box: 2.7 s, 12.7k
  fields per year, 9 MB per year. State-wide scans are 2-4 s per state for aggregates.
- CDL native 30 m pixels, 2024, same box, `ST_Transform(ST_Point)` to 4326:
  **0.7 s**, 659k points (via `xql.register` on the 0.4.0rc1 backend).
- Majority crop per field (`ST_Contains` join, count per field x crop, row_number):
  **0.1 s**. 2,040 of 2,110 fields catch a pixel. Median purity (share of a field's
  pixels in its top crop) 0.86, mean 0.78. Fresno top: grapes 708, almonds 583,
  grass/pasture 263, developed 111, pistachios 77, alfalfa 57.
- Probability Zarr, same box, 2024, 3 bands: root (10 m) 2226 x 2226 **10.4 s**
  (9 inner chunks x 3 bands x 16 MB decompressed), 4x **1.9 s**, 16x **1.3 s**.
  No NaN in the box. Mean P(field) 0.41.

So the serve cost of a fields-plus-crop view is ~3.5 s cold, dominated by the
parquet fetch; the join is free.

## Ways it could go (not ranked)

1. **The field is the unit, CDL is its label.** One PolygonLayer of FTW fields, fill =
   majority CDL crop (existing palette), per-field purity, acres, and the 2024 vs 2025
   pair. Stays inside the one-layer rule (a second deck layer dies under marimo; the
   fields would REPLACE the pixel squares at fine zooms, not sit over them). Pixel
   squares above the zoom where fields are worth reading, fields below.
2. **Purity / mixed fields.** Fields whose CDL pixels disagree (purity < 0.6) are
   where either model is wrong or the "field" spans two crops: a map of the
   disagreement, and a table of which crop pairs get merged.
3. **Field-level rotation.** The corn/soy self-join already exists per pixel;
   per field it becomes "this field went corn -> soy -> corn" over 18 CDL years
   with the 2024/2025 FTW polygon as the frame. Field size by crop (grapes vs
   wheat vs pasture) is one GROUP BY.
4. **CDL's own 10 m group vs FTW 10 m**: same years (2024, 2025), both 10 m; the
   field boundary from Sentinel-2 against crop pixels from Sentinel-2 + Landsat.
5. **The probability raster instead of the vectors**: P(boundary) at 4x..16x as a
   paint (1-2 s per view), or P(field) as a "is this cropland" mask against CDL's
   crops-only switch (where CDL says crop and FTW says not-field, and vice versa).
   Native 10 m is 10 s per 20 km box from home: not interactive.
6. **FTW 2024 vs 2025 change** (fields that appear/disappear) joined to CDL's
   class change in the same pixels (cropland -> developed was already on the list).

What it does NOT add: boundaries as paint over pixels (Stephen's call on the
removed boundaries mode stands; the pixels are already polygons), and anything
built on confidence.

## The notebook: `xsql-cdl-fields.py` (2026-08-20, built the same day)

Crops notebook fork. Stephen's brief after the recon: a DuckDB demo ("most SQL
is best"), start with the per-field history and the disagreement, add more
later, and make it clear which data comes from where (hence the intro table and
the docstring). Design:

- ONE PolygonLayer (the marimo second-layer rule). Under `FIELD_BOX_DEG2` the
  layer is the FTW fields of the box filled by their majority CDL crop for the
  slider year, stroked 1 px; over it the crops notebook's pixel squares, not
  stroked. `FIELD_ROW_BUDGET` (80k fields) falls back to pixels too.
- Fields per (box, FTW year) are fetched ONCE into `fb_n` on the serve
  connection and the CDL-pixel-centre -> field lookup `lk_n` is built once with
  `ST_Contains` (DuckDB's spatial join); every CDL year is then `cdl_k JOIN lk_n
  USING (y, x)` + `row_number()` majority + purity into `cur`, which feeds both
  the layer (`from_duckdb`, geometry cast `::GEOMETRY`) and the legend (share of
  FIELDS, not pixels). A box already fetched that contains the new one serves it.
- `join_level`: finest CDL level with <= `FIELD_PX_BUDGET` (1.5M) pixel centres
  in the Albers box; with `FIELD_BOX_DEG2` 0.35 that is native 30 m for a county
  view and 60 m at the limit. A first pass with 0.6 deg² reached 120 m, where
  13-acre fields hold 1-3 pixels and purity is trivially 1.00; tightened.
- The analysis is marimo SQL cells on `con` (the serve runs on `mcon`): each one
  statement, `CREATE OR REPLACE TABLE` feeding the next, on the last served box
  from `HOLD["box"]` and the strip's years, re-run by a `mo.ui.run_button`.
  Tables: `fields_view`, `px2field`, `field_years`, then sequences, rotation
  classes, purity bands, least-pure fields, `agreement` (+ 2x2, FTW misses, FTW
  false fields).

Driven (playwright, marimo run --headless, chromium, 1500x950), 2026-08-20:
opening `fields · FTW 2025 · CDL 2025 at 30 m · 4,640 fields · median purity
0.80 · 522 ms` (the map cell pre-builds the tables); year step to 2024 398 ms;
six wheel notches out -> `pixels · 32x · 960 m · 386,816 drawn · 2858 ms`;
seven back in -> `fields · ... 4,045 fields · 440 ms`, screenshot clean at +4 s
and +20 s. Console shows the known benign `assertion failed` per serve. Headless
export: all cells pass, ~50 s cold end to end; `app.run()` 45.6 s.

Opening-box results (Fresno, 0.17 deg², 9,244 fields at the first 60 m pass):
top sequences all-Grapes 408 fields / 18k ac, all-Almonds 132, Grapes x14 ->
Almonds x4 97 fields (the conversions), Grassland 213; rotation classes 53%
perennial planted by 2010, 7% not cropland 2020-25, the rest many-crop row land;
purity (60 m) 47% clean / 14% / 14% / 24% mixed; the least pure fields are
Peaches-Cherries, Peaches-Citrus, Grapes-Developed (FTW merging a yard into a
vineyard). Agreement 2x2 (acres): CDL crop & FTW field 50%, neither 33%, CDL
crop & FTW not field 10%, FTW field & CDL not crop 7%. FTW sees a field on 67% of
Grassland/Pasture acres, 49% of Fallow, 21% of Developed/Open Space; among
crops, Oats 52%, Olives 52%, Winter Wheat 63%, Misc Vegs 54% are the most
missed (small n in this box).

Not built: click-a-field history (geometric picking in JS as the HRRR film does),
the disagreement as paint (a per-field "FTW P(field) mean" column is one more
join), CDL 10 m vs FTW 10 m, 2024 -> 2025 field appearance/disappearance against
CDL class change, county roll-ups.

## Rebuild the same day (Stephen's review)

Review points: no analysis button (it had been moved); no way to show
non-crop data (the polygon layer replaced the pixels); field shapes change over
time, so 2024/2025 polygons cannot frame 2008; FTW is a mask or not; the
P(field) leg was only a table, not on the map; and the dashboard risked being
confusing.
The rebuild is the crops notebook plus ONE control, `FTW: off / mask / fields /
disagreement`: the map is always CDL pixels; mask keeps pixels inside a field
(2024 footprint for older years, stated in the status); fields and disagreement
exist only for 2024-2025 (same-year frame, no anachronism); analyze is back in
the strip and runs under the mask; the SQL cells keep the same-year per-field
tables and the 2x2 and drop the 18-year sequences. Runs from the root project
(root pyproject -> xarray-sql[duckdb]==0.4.0rc1). DuckDB lesson: CASE cannot
return UTINYINT[3]; the disagreement colours come from a VALUES join.

Driven (playwright, root venv, 1500x950): mask open 2x/60 m 108,567 drawn 2.2 s;
off 370,000 drawn 1.8 s; fields 4,374 fields, median purity 0.92, 0.5 s;
disagreement 218,811 drawn 2.4 s (screenshot: orange row-crop blocks CDL calls
crop and FTW misses, grey agreement over the vineyards/orchards, blue speckle);
analyze panel with the six timelapse lines in 2.6 s; year step to 2023 under
disagreement falls back to "mask: FTW 2024 fields" (2.1 s); six wheel notches out
-> 32x/960 m 264,625 drawn with "zoom in for FTW". Zero non-assertion console
errors. Headless export from the root venv passes; deforest passes from the root
on the rc backend too.

## Third shape the same day (the one that stands)

Review of build 2: the fields ARE the mask, and the disagreement had to be
visible WITH the fields. Final controls: the crops notebook's, plus two
checkboxes, `fields` (clip + outlines, one deck layer: outline rows appended to
the pixel table with transparent fill, 4-channel colours, stroked only while
on) and `disagreement` (repaint; 2024-2025 only, greyed out otherwise with the
reason). 2024 and 2025 are served from CDL's 10 m group (his call: FTW's own
resolution); older years 30 m, fields still clip them. The fields-majority fill
mode is gone. Todo: picking (which dataset says what at a point), not now; the route is the
HRRR film's geometric picking in JS, not the bundle patch.

Driven (playwright, root venv, 1500x950), third shape: open `4x · 40 m pixels ·
99,820 drawn · year 2025 · FTW 2025 fields` 6.6 s (cold: the first serve builds
fb_0 + lk_0 on the 10 m group's 40 m level); fields off 417,366 drawn 1.6 s;
disagreement alone 209,271 drawn 2.3 s; disagreement + fields 98,401 drawn
1.8 s (screenshot: grey agreement inside the outlines, blue where FTW draws a
field CDL calls non-crop, orange flecks where P(field) < 0.5 inside a polygon);
analyze panel (6 timelapse lines, 30 m group under the clip) 5.1 s; year 2023:
`2x · 60 m pixels · 44,396 drawn · year 2023 · FTW 2024 fields · disagreement
needs 2024 or 2025`, checkbox disabled; six wheel notches out: 64x/1920 m with
"zoom in for FTW". Zero non-assertion console errors; headless export passes
from the root venv.

## Serve cost round (2026-08-20, evening)

Warm moves still took several seconds. Driven pans with fields on, before:
kernel 2.0-3.4 s per pan when the polygons had to be re-fetched (the fetch was
per serve box), 18 s on a pan that left a 2.5x-padded box (the parquet read is
row-group bound: ~48 row groups of ~13 MB per state file, spatially sorted, so
a wide box reads most of the file). After: padded fetch 1.6x each side capped at
0.4 deg² on a third connection in a thread, concurrent with the CDL centre
scan; lookups per serve box from the cache; grid rows cached; outline rings
simplified to half a pixel. Driven: open 11.1 s (cold: fetch + lookup), pans
1.4 / 2.1 / 2.9 / 1.4 / 4.7 s kernel, tables 12-18 MB each (120-175k squares
at 40 m on the 10 m ladder). Wall clock from drag to paint in the loaded
headless Chrome was 12-18 s: the payload is the floor, not the SQL. Zoom-in
and small pans inside the served box are held (no re-serve). The next lever
is fewer rows on the 10 m ladder (PX_PER / ROW_BUDGET), a resolution trade.

## Bitmap (2026-08-20, night)

The polygon squares bought nothing (not even picking, which needs the same JS
route either way); less data into lonboard means the picture. Same SQL and caches; the serve ends
in a PNG of the view (numpy Albers forward, closed form, mm-exact vs DuckDB;
PIL draws the FTW rings), one BitmapLayer, bounds = the view box. 0.4-1.1 MB
per serve at any resolution (was 12-18 MB of polygons), ROW_BUDGET 3M so the
20 m level serves at the Fresno opening. Two-stage paint on a cold FTW miss.
Driven: pans with fields 0.7-0.9 s kernel / ~2 s wall; disagreement pans
1.0 s; cold region 10-15 s (parquet from home, 3-10 s per new row groups;
warm 0.4 s on the same connection; a wider ring is prefetched after a miss).
Measured pieces: fetch cold 3-10 s / warm 0.4 s, WKB parse 0.0 s, rings
0.1 s, 3.4M centre transforms 0.8 s, ST_Contains into 5k fields 0.5 s. The
lazy `.arrow()` reader was why the "concurrent" fetch was not concurrent.
Screenshots: pixels clipped inside crisp outlines; disagreement + fields shows
the Kings River foothill pasture as large blue (FTW fields, CDL non-crop).

## End of 2026-08-20

Flown (edit and run): moves are fast when the FTW cache hits and ~10 s when it
misses. The 10 s is the FTW parquet read on a miss (row groups of 13 MB,
3-10 s from home), independent of the layer type. Proposed fix, undecided:
download each state file once into the tmp cache in the background when
fields is ticked (CA 629 MB, ~25-30 s once), then all fields moves are local.
Also undecided: polygons back (6b816ac) vs bitmap with ROW_BUDGET 420k, capped
fetch threads, no prefetch. The bitmap serve is not validated interactively;
the driven harness measured kernel and paint times and did not capture the
interactive feel (likely CPU saturation). See CLAUDE.md.

## 2026-08-20, late: no parquet on the serve path (raster clip + PMTiles outlines)

Stephen: polygons vs bitmap had become a yes/no fight; what he wanted was to
find the thing that actually stalls and take it out. Measured that night from
home, same Salinas box then a 20 km pan, then back:

| | parquet (cache_httpfs on) | raster 4x (zarr via xql) | PMTiles z13 | PMTiles z12 |
|---|---|---|---|---|
| cold | 2.4 s, 3.8 MB | 1.1 s | 1.3 s, 0.4 MB, 36 tiles | 0.8 s, 0.5 MB, 12 tiles |
| pan east | 1.6 s | 1.5 s | 0.6 s | 0.6 s |
| back | 0.0 s (disk) | 1.1 s | 0.7 s | 0.7 s |

The link was fast that night (the 10 s parquet stalls are slow-link nights);
by BYTES the order is PMTiles < raster < parquet, and bytes are what a slow
link charges. Decision (Stephen): PMTiles for the outlines, the probability
raster for the clip, parquet only for the SQL cells. The layer stays the
bitmap for this round, polygons from 6b816ac are the revert if it does not
feel right on his screen.

Built:
- `cache_httpfs` (DuckDB community extension) loaded on every connection,
  `HTTPFS_CACHE_DIR` under the OS tmp dir: every byte range fetched over
  httpfs is kept on disk, so any row group read once is local afterwards,
  across connections and restarts. Measured (CA file, Fresno + two pans): 2.3
  / 1.4 / 1.6 s cold, 0.3 / 0.0 / 0.0 s in a fresh process; 19 MB on disk.
  Kept for the SQL cells; the map no longer reads the parquet at all.
- The clip is the P(field) >= 0.5 grid (ftw_4 / ftw_16, the same read
  disagreement makes), read for the padded box and cached; `lk_n(y, x)` is
  the CDL centres that bin into a field cell, built once per (table, level,
  grid box); serve / analyze / timelapse keep their `JOIN lk USING (y, x)`.
- The outlines are the per-state PMTiles (`US_<ST>.pmtiles`, tippecanoe
  z0-13, layers "2024"/"2025"): new cell, the counties film's PMTiles v3 +
  MVT decode by copy, sync obstore in a 16-thread pool, tile zoom =
  floor(camera zoom) capped 13 (~15-40 tiles per view), raw tiles cached on
  disk (`$TMPDIR/x-sql-marimo/ftw-tiles/<ST>/z/x/y.mvt`, empty file = absent),
  decoded polylines in memory. Segments along a tile's clip line (both ends
  outside the tile on the same side) are dropped, so no tile seams; and
  render_view no longer closes polylines (it did, for the parquet rings, and
  a tile-cut piece closed itself with a diagonal across the field).
- Gone: fcon, the fetch pool, the padded parquet fetch, the prefetch, the
  ST_Contains lookup, fields_sql / lookup_sql / rings_from_geojson.
- Semantic change: with fields ON the clip and disagreement's "FTW field"
  are the same grid, so the orange class (CDL crop, no FTW field) only shows
  with fields OFF; the old polygon clip let orange flecks through where the
  polygon and the grid disagreed.

Driven (playwright, headless Chrome, this Mac): fields on cold 5.8-7.1 s
(grid + lookup + 15 tiles z12); pans: 0.6-1.1 s when the grid cache hits,
2.5-3.9 s when it misses (raster read + lookup, not the wire); disagreement +
fields 0.7 s, pans 0.6-2.8 s. Worst case ~4 s against 10+ s before. Not yet
flown by Stephen.

## 2026-08-20, later: the FTW side decided per output pixel; chunk-cached mask

Stephen flew the tile/raster build: judder gone with SWAP_HIDE_S 0.15 (back
with 0), "intermittently non-responsive, sometimes fast, or sluggish". The
driven numbers said the same: hits 0.6-1.1 s, misses 2.5-3.9 s, a miss every
second or third pan. The miss was two SQL passes: the P(field) raster read for
the PADDED box (up to 0.4 deg², ~10x the view, 2-3.5 s) and ST_Transform of
every CDL centre in that box to bin it (1-2 s, millions of points at 20 m);
disagreement paid the transform on every serve.

Rework (in the notebook):
- `render_view` takes CLASS CODES (not RGB) plus a dense boolean of the
  P(field) grid and decides the clip and the disagreement class PER OUTPUT
  PIXEL from the lon/lat it already computes for the warp. No lk tables, no
  ST_Transform passes, no disagreement SQL on the serve; `cur` is the plain
  CDL pixel fetch. Counts for the legend come from the drawn output pixels
  (acres from the output pixel's ground area). The clip edge is the 40 m FTW
  cell, not whole CDL pixels by centre. `_ftw_lookup` (the SQL join) stays
  for analyze / timelapse, built on click.
- The mask is cached BY THE ZARR'S INNER CHUNK (512 px, ~20 km at 40 m):
  `_ftw_mask` reads only the missing chunks of the view in one query over
  their bounding box, keeps them in memory (cap 600) and on disk as packbits
  (`$TMPDIR/x-sql-marimo/ftw-mask/<f>x/<year>/<cx>_<cy>.npy`, 32 KB each),
  and assembles the view's mask from them. Restarts are free.
- `SWAP_HIDE_S` (0.15): opacity 0 across the image/bounds swap, 1 after.
  deck loads a new `image` URL asynchronously but applies `bounds` at once,
  so the old picture sat stretched in the new box: the judder. Lonboard's
  image trait is URL-only, so this is the only lever short of the bundle
  patch or a bespoke deck widget.

Driven (playwright, headless Chrome, this Mac): fields on cold 4.6 s (9
chunks + lookup-free); pans 0.51-0.56 s on a hit, ~2.0 s on a one-chunk miss
(pan4 only; pan2/3/5 that missed before are hits now); disagreement 1.4 s
first, pans 0.4-1.7 s. 9 chunks / 324 KB on disk after the run.

Polygon benchmark (scratch copy `_poly_bench.py`, same pipeline, PNG replaced
by a PolygonLayer table): at the Delta HOME zoom 12 the 420k-row budget picks
4x = 412k squares = 40 MB per frame; fields frame 168k squares + outlines =
22 MB at 3.3 s kernel. `mcon.register` of a pyarrow table from the worker
thread hung the serve (hex WKB in SQL instead); not finished, killed because
it competed with Stephen's flight. The bitmap's payload is 0.1-0.5 MB at the
same views.

## End of 2026-08-20: three states, undecided

Stephen, late: "it looks great, but I don't understand why we had to sacrifice
all the SQL"; the crops notebook is the clean DuckDB demo (fast, simple), this
one compromises DuckDB to chase the map; "I don't want the way it is now, and I
don't want to go back to the way it was before." Nothing decided. The states:

1. **Polygon serve** (`6b816ac`): one SQL query -> `from_duckdb` -> PolygonLayer,
   the FTW clip a `JOIN lk USING (y, x)`, outlines as `UNION ALL` rows. The
   SQL-shaped version. Cost: payload, 12-40 MB per move at the Delta zoom
   (measured today on the new data path: fields frame 22 MB, 3.3 s kernel).
   Levers in SQL if revisited: a coarser rung per zoom, run-length merging of
   same-class pixels along rows (rectangles, not squares).
2. **Bitmap serve** (`main`, `7410439`): one PNG per view; the FTW clip and
   disagreement decided in numpy per output pixel; mask chunk cache;
   `SWAP_HIDE_S` for the judder (deck loads `image` async, applies `bounds` at
   once). Flown: judder gone at 0.15; pans 0.5 s warm / ~2 s on a chunk miss.
   The SQL for clip/disagreement CAN come back in front of this renderer at
   ~0.3-1 s per fresh view (the ST_Transform pass); numpy was an
   over-optimisation, not a requirement.
3. **Tile serve** (branch `fields-tiles`): lonboard RasterLayer, deck asks
   z/x/y, ONE query per batch (per-tile SQL has ~0.2 s fixed overhead on the
   xql table, measured: 24 tiles 5.7 s one by one vs 0.19 s as one query),
   whole view per batch (deck caps in-flight requests at 6), tiles cached,
   remove-then-add the layer on a state change. Everything works EXCEPT the
   colours: lonboard renders each tile through a mesh sub-layer whose
   fragment shader calls `lighting_getLightColor`, ~0.69x on every channel,
   `opacity` ignored; the TMS-less path (`getTileData` returns null without
   tileMatrices) is dead code. Fix = a one-line bundle patch (material off),
   or a bespoke deck.gl anywidget. The deforest notebook's raster is darkened
   the same way (unnoticed under a 0.7 ramp).

The SQL cells under the map redo the joins on the last box; with the map's
own pipeline that is two pipelines for one idea. The shape that would read
as a DuckDB notebook: register -> the joins as SQL cells -> the map as an
output of those tables. Not built.
