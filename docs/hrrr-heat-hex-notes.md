# HRRR heat with a memory: notes

`xsql-hrrr-heat-hex.py`. Built 2026-08-17 from the counties film. The question it
asks is persistence, not level: how the heat sits and drains, hour by hour, and where
the nights do not cool. Stephen's brief: "a time lapse that kinda sticks", using
temperature, humidity, wind and precipitation; hexagons rather than counties; res 6 if
it fits (it did not, see below), a fallback on counties.

## What it is

- **Read**: dynamical.org's HRRR analysis (icechunk, `s3://dynamical-noaa-hrrr`),
  `temperature_2m` + `relative_humidity_2m` always; `precipitation_surface`
  (`READ_RAIN`) and `wind_u_10m` + `wind_v_10m` (`READ_WIND`) as flags. xarray-sql,
  no dask, blocks = the whole time window x one 45x45 store column, as in the
  counties film.
- **Fold**: every pixel labelled with its H3 res 6 cell from the store's 2-D lat/lon;
  `GROUP BY hour, cell` averages ~4 pixels per cell; 210,724 cells over CONUS land.
  (Res 5 was the unit for one round of the build, see below; res 6 is where it
  landed, at Stephen's word.)
- **Land mask**: the counties film's PMTiles counties (same tmp parquet cache),
  polyfilled 'center' at RES in DuckDB: a cell is CONUS land if its centre is in a
  county; that county's name is what a clicked cell reports. Stephen asked why
  DuckDB is there at all for hexes: only for this mask and the names; the fold does
  not touch it. Kept at his word ("thats fine keep it"). Without it the fold would
  cover the whole HRRR domain (ocean, Canada, Mexico; ~65k cells) and lose the
  block pruning below.
- **Heat index**: NWS (Steadman simple formula, Rothfusz regression once the mean of
  it and T reaches 80 F, the two RH adjustments) in numpy on the cell means.
- **The film**: uint8 F x N heat index (0.5 degC steps from -40; 255 = no data),
  plus one packed uint8 for wind (m/s in the high nibble) and rain (0.5 mm/h steps
  in the low), 35 MB per field for a week at res 6 (5 MB at res 5).
- **The accumulator, in the browser**: L[f] = a L[f-1] + (1 - a) max(0, HI[f] - thr),
  a = 2^(-1/half_life), so L is "sustained excess above the threshold" in degC and
  reads on the same scale as the index; rain multiplies L by (1 - flush min(1, mm/2.5))
  that hour, wind scales the excess by max(0, 1 - vent ws/10). Recomputed over the
  whole film on every slider move (35M multiply-adds at res 6, ~0.2 s), stored uint8
  (0.1 degC steps) for paint, the picked cell's line exact. Ramp 0 .. p98 of the load.
- **The widget**: the counties film's skeleton (deck 9.3.10 pinned graph from esm.sh,
  browser clock, minimal HUD on the map, `window` the one trait back) with deck's own
  `H3HexagonLayer` (`highPrecision: true`: deck's own auto rule picks it only at
  res <= 5, but the
  instanced path shares one hexagon shape across the layer and over 60 degrees of
  longitude leaves gaps) in place of the GeoArrow polygons, colours via an accessor
  keyed on `[frame, field, gen]` through `updateTriggers`, and picking by h3-js
  `latLngToCell` (pinned 4.5.0, the version geo-layers 9.3.10 resolves its ^4.4.0 to)
  on the unprojected click. Fields switch with the two buttons or I / L. Every class
  is `hf-` prefixed (the Tailwind `.hidden` lesson).

## The read is the wire (2026-08-17, measured)

Stephen: "this is all too long. internet research for reading async from zarr".
Findings, all measured from his machine:

- Raw range reads from us-west-2 S3 (obstore, `overturemaps-extras` PMTiles object):
  1 x 100 MB 20.9 MB/s; 8 x 25 MB 11.4 MB/s; 32 x 8 MB 16.2 MB/s. The link is ~200
  Mbit and more streams do not help. Async concurrency knobs cannot beat it.
- The store IS sharded (shard 2160 x 540 x 450, inner chunk 2160 x 45 x 45, blosc
  zstd 3, shuffle), so icechunk already range-reads inner chunks; the inner chunk is
  still 2,160 h deep, so a 7-day window fetches the whole current chunk layer.
- dynamical.org: the analysis has no space-optimised variant; the map-optimised
  stores are the 18 h and 48 h forecasts. Nothing cheaper to point at.
- Windowed COG reads (async-geotiff) do the same thing icechunk does here; what a
  COG has and this store does not is a tile that is small in every dimension and an
  overview pyramid.
- **The lever we own: bytes.** Only 523 of the 960 store columns touch CONUS land.
  xarray-sql prunes partitions on dimension predicates (`BETWEEN`/`AND`/`OR`), so a
  `WHERE cube.y BETWEEN .. AND (cube.x BETWEEN .. OR ..)` over 22 block rows skips
  the ocean / Canada / Mexico columns before any byte moves. 2 variables, 7 days,
  res 6: 44.7 s -> 28.5 s. Column names must be qualified (`cube.y`) or DataFusion
  reports an ambiguous reference against pix2h.
- **Cost is per variable and per store chunk, not per day**: 2 vars ~28 s, +14 s
  per further variable (rain 1, wind 2), for any window in the current chunk (447
  of 2,160 h filled on 2026-08-17). A window in a full chunk is ~4x: Jul 6-12 (the
  West dome) measured 129.9 s at res 5. The HUD estimates it for the dates picked
  as 6 s + 0.055 s per filled hour of every chunk the window touches (fits 447 h ->
  31 s, full chunk -> 2 min). Chunk starts: every 2,160 h from 2014-10-01, so
  2026-05-01 and 2026-07-30 this summer.
- deck.gl-raster / zarrita.js: not faster for the read (same bytes over the same
  wire, JS/wasm decode slower than zarr-python); they change the shape (no kernel,
  pixels on the GPU), which is the right tool if a small precomputed cube existed.
  The analysis is icechunk, and no JS icechunk reader is known here.
- Near the data (molab) the wire stops being the floor and decode is; zarrista is
  the lever there (docs/hrrr-counties-notes.md).

## The res 6 / res 5 round trip (2026-08-17)

Same read at any res; what scales is the number of ANSWERS, hours x cells. Res 6
was built first, its 17 GB peak set off the memory hunt below, Stephen called it
"not a good solution" and chose res 5 with counties as the fallback; res 5 was
built, flown and measured; then he set RES back to 6 himself and asked for the
commit, so the res 6 numbers below, with the two DataFusion knobs, are what ships.

| unit | cells | answers / week | DataFusion peak | to browser |
|---|---|---|---|---|
| counties | 3,108 | 522k | small | 2 MB (f32) |
| res 6 | 210,724 | 35.4M | 17 GB default plan; 9.5 GB CollectLeft; ~5 GB + 3 GB spill pool | 35 MB / field |
| res 5 | 30,124 | 5.06M | ~4 GB peak (the read path, see below) | 5 MB / field |

Why 17 GB: DataFusion's default plan re-hashes the cube by (y, x) into 16
partitions for a `Partitioned` hash join (pix2h at ~10 MB is over the 1 MB /
128k-row `hash_join_single_partition_threshold[_rows]` defaults), so a cell's pixels
scatter and every partition's partial aggregate holds its own copy of nearly every
(hour, cell) group. Raising the two thresholds makes it `CollectLeft` (the pixel
lookup broadcast to every block partition, the cube's 523 partitions kept intact),
each block reduces its own cells early: 9.5 GB. A `RuntimeEnvBuilder().with_fair_spill_pool(3 GB)`
on top: 4.6-5.1 GB process peak, no time cost (28-30 s), but a 1.5-2 GB pool spilled
and measured HIGHER (6.1-6.5 GB; spill buffers are outside the pool). A numpy
streaming accumulation (`execute_stream()`, `np.add.at` into F x N sums and counts,
12 B per answer instead of ~150) was drafted and NOT run: Stephen stopped it, and
the SQL fold is the point of the repo. The CollectLeft thresholds AND the 3 GB pool
are in the fold cell; res 5 (one constant) and the counties film are the retreats.

The remaining ~4 GB at res 5 grows linearly THROUGH THE FETCH and plateaus, with 5M
groups, so it is the read path, not the aggregate: DataFusion runs all 523 block
partitions as concurrent tasks (target_partitions does not cap the input tasks of a
RepartitionExec; 4 vs 16 measured the same), each holding its block and in-flight
batches, plus zarr's transient decode of the 17.5 MB inner chunk per variable.
Not chased further; the counties film runs the identical read.

## Flights (2026-08-17, headless Chrome via playwright 1.60 against `marimo run`)

Res 5, last 7 days: boot to 30,124 cells / 159 frames in 33-36 s; both fields paint
(index on the diverging ramp pivoting at the film median, load on inferno 0 ..
p98); the click picks a cell ("cell in Chase County, KS", value, line with the
dashed threshold); the load button's note states the read estimate for the dates
(31 s for Aug 11-17, 2 min for Jul 6-12 or a window straddling Jul 30); no console
errors. `marimo export html`: 33 s wall, exit 0.

Res 6, opening window Jul 23-29 (the Plains dome, a full chunk): 210,724 cells /
168 frames, first frame at 141 s, both fields, picking, no console errors; the load
at 04Z Jul 27 is bright from Kansas to the Ohio valley in the middle of the night,
which is the whole idea on one screen. Stephen ran the notebook locally at res 6
before the commit (the flight harness is a scratch script; playwright must match the
chromium build in ~/Library/Caches/ms-playwright, 1223 -> playwright 1.60).

## The opening window (2026-08-17)

Stephen: "pick a big heatwave from summer 2026 in conus", then "find a better
heatwave from this past summer". Wikipedia's 2026 North American heat wave page
gives three domes: East Jun 28-Jul 5 (Atlantic City 106 F Jul 4), West/central from
Jul 6 (Miles City 115 F, Salt Lake City 109 F on Jul 12), Plains late July (Rapid
City 112 F Jul 26); NOAA: July 2026 the hottest US month on record. A block-sampled
scan of the store (one 45x45 column in every third, Jun 15 to Aug 17, T only) puts
the CONUS-wide peak at Jul 25-27: share of sampled land pixels over 35 degC 0.22-0.24
(West dome Jul 8-12: 0.13-0.14; Aug 6-11: 0.16-0.19), CONUS-mean day peak Jul 27
(25.65 degC). Opening window: 2026-07-23 to 07-29. All three domes are in the full
May-Jul chunk, ~2 min; the East dome is the humid one and the most on-theme for a
heat index film if the cost is ever not the constraint.

## Ideas held back

- Extrusion as the second channel: colour = heat index, height = load, pitched, so
  the columns stay tall while the colour cools each night. Two channels on the same
  hexes is the one honest way to show both at once; two fills is not.
- `downward_long_wave_radiation_flux_surface` as the night-time decay signal (humid
  or cloudy nights radiate back; physically why nights do not cool) in place of a
  fixed half-life; `downward_short_wave_radiation_flux_surface` as the daytime
  forcing instead of a hard threshold. Range not yet looked at.
- Anomaly instead of absolute: the same window in a prior year (two reads).
- Baseline persistence metrics as one-frame maps: hours over threshold, longest run,
  nights that never fell below X, hours-to-recover after the CONUS peak.
- Res 6 over a region (a state or two as `BOX`): 4 px per cell, a tenth of the cells.
