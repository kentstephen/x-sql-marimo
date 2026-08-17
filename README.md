# x-sql-marimo

Fold a raster to H3 in SQL, then join it to something that has edges. Rasters and
boundaries stream straight out of object storage (no tile server, no STAC API, no
pixels leave the bucket until the viewport asks); the fold is a SQL `GROUP BY` over an
H3 UDF; the join is on the cell id; the map is deck.gl.

```bash
# where forest was lost 2002-2022, by administrative division, worldwide
uv run marimo edit xsql-deforest-divisions.py --sandbox

# HRRR 2 m temperature per CONUS county, hour by hour, as a film
uv run marimo edit xsql-hrrr-counties.py --sandbox

# HRRR heat index on H3 hexagons with a browser-side memory (heat load)
uv run marimo edit xsql-hrrr-heat-hex.py --sandbox

# EXPERIMENTAL: worldwide Mapterhorn terrain as extruded H3 columns (open defects)
uv run marimo edit xsql-mapterhorn-explorer.py --sandbox
```

**Two H3 engines, each used for what it wins.** [h3ronpy](https://github.com/nmandery/h3ronpy)
(h3o, whole-column) generates cells: lat/lng to cell over millions of pixels as a
DataFusion UDF, 70 ms against DuckDB's 462 ms per-row UDF on the same 1.58M rows.
[DuckDB `h3`](https://github.com/isaacbrodsky/h3-duckdb) (Uber's C library) does the
other H3 work here: polygon to cells (the polyfill) and, with `spatial`, the seam
dissolve of tile-clipped boundaries, 75 ms against h3ronpy's 928 ms. So every notebook
folds in DataFusion + h3ronpy and does geometry in DuckDB, and the join runs where the
cells already are.

## Deforestation by Overture division

`xsql-deforest-divisions.py`. Free-fly the planet; everywhere the camera lands, the
mean share of ground deforested 2002-2022 per region, county or locality.

**Data.** [vizzuality/lg-land-carbon-data](https://source.coop/vizzuality/lg-land-carbon-data)
on Source Cooperative (Vizzuality for LandGriffon, CC-BY 4.0):
`deforest_100m_cog.tif`, 5.7 GB, EPSG:4326, whole globe, 100 m, ten average-resampled
overviews. A pixel is the **portion of it deforested**, 0 to 1: intensive, so `mean()`
is valid at any scale and the overview pyramid is legitimate. Nine sibling layers
(carbon, cropland expansion, biodiversity intactness, forest landscape integrity) share
the shape and CRS. Boundaries are [Overture Maps](https://overturemaps.org/) divisions
from the release's own PMTiles.

**What is drawn.** The COG itself, as ramp-coloured tiles served from the kernel
(zero and no-data transparent: the map shows where loss is, not where it is absent),
and the divisions as a choropleth of mean share, with a fill slider and outline toggle.
The division numbers come from an H3 fold running underneath, not from the paint.

**Why H3.** Degrees are not equal area (a 100 m pixel at 60 degrees covers half the
ground of one at the equator), so averaging pixels over a wide country overweights its
poleward end. Cells are near-equal-area: weight pixels within a cell by valid count,
weight cells within a division equally, and the fold is an area-weighted mean for
free. Getting this wrong is invisible on screen.

| step | engine |
|---|---|
| stream the COG and the PMTiles | obstore, async-geotiff |
| fold pixels to cells | DataFusion + h3ronpy |
| polyfill divisions, dissolve tile seams | DuckDB `h3` + `spatial` |
| join cells to divisions, rank | DataFusion |
| render | lonboard (deck.gl) |

**The fine views are the cheapest.** Each fold reads only the padded viewport from
the overview that matches the H3 resolution it builds (res 4 from the 6.4 km level,
res 8 from 200 m), so the viewport shrinks faster than the resolution grows. The whole
world at res 4 is 288k cells, 15.7M pixels, 0.8 s read plus 0.3 s fold.

**The COG is sparse.** 73.6% of full-resolution tiles are unstored ocean; a naive read
asks for byte range `0..0` and raises a `ValueError` that reads like a corrupt file.
Reading on the tile grid and consulting `tile_byte_counts` first turns that into a
speedup: an absent tile is NaN with no request.

**PMTiles, not GeoParquet.** Overture's GeoParquet has no spatial ordering, so
geometry (99% of the bytes) cannot be pruned: ~190 MB per file for a Rondônia-sized
viewport. The same release as PMTiles reads ~0.8 MB (regions, cold, 0.74 s against
13.8 s; the whole planet's regions 4.2 s). The v3 reader and MVT decode are hand-rolled
and verified ring-exact against `mapbox-vector-tile`; tile-clipped pieces are dissolved
per `division_id` before drawing. Subtype floors, measured off the tiles: country z2,
region z4, county z8, locality z10.

**Ranking.** "Rank what's in view" turns the join into a table: every division on
screen by mean share, one H3 step finer than the screen, falling back county to region
to country (Overture has counties for 171 of 219 countries). The camera is the only
statement of intent. Views servable from memory (a pan inside the box, a zoom back to a
visited res) answer synchronously; only new bytes go through the debounce.

**Colour.** At res 4, 69.6% of cells are exactly zero and the rest span nine orders
of magnitude, so the ramp is cividis, log10 over 1e-4 to 0.5, and zero gets the dark
end alone: a neutral grey (luminance 0.313) and cividis's 0.1% stop (0.318) were the
same colour, so the floor is lifted to the upper 75% of the ramp. Monotonic in
luminance under a deuteranope simulation; hue carries nothing.

**Sanity.** Rondônia 27.0%, Mato Grosso 21.6%; Iowa counties 0.75-0.95% (cleared in
the 1800s); Congo basin interior 0.8-19.9% with Kisangani highest. A 30-70x split in
the right direction; a smeared join would collapse it.

## HRRR temperature by county, as a film

`xsql-hrrr-counties.py`. [dynamical.org](https://dynamical.org/)'s Zarr build of NOAA's
HRRR analysis read with [xarray-sql](https://github.com/alxmrs/xarray-sql) (DataFusion
over the Zarr cube, no dask), every pixel labelled with its H3 res 7 cell from the
store's own lat/lon, Overture counties polyfilled in DuckDB at the same res, one join
and `GROUP BY` for 2 m temperature per county per hour over a chosen window (UTC
dates, hourly or daily mean/max, from a control on the map). The table is small, so
the whole film goes to the browser once and the browser owns the clock: a bespoke
anywidget on deck.gl + `@geoarrow/deck.gl-layers`, minimal HUD on the map, the map's
own fullscreen; the kernel is idle while it plays and only the load button reaches
back. About thirty seconds to the first frame, nearly all of it the read.

## HRRR heat with a memory

`xsql-hrrr-heat-hex.py`. The same machinery asked how the heat sits, not how hot it
is. Temperature and relative humidity folded to H3 res 6 over CONUS land (210,724
cells, ~4 pixels each), NWS heat index per cell per hour, and in the browser an
accumulator, **heat load**: it rises by the heat index's excess over a threshold and
decays with a half-life (an exponential moving average of the exceedance, in °C), so
places whose nights do not cool stay bright after dark. Half-life, threshold, and if
read a rain flush and a wind vent, are sliders on the map, recomputed over the whole
film in the browser; the map switches between index and load; a click gives a cell's
line. Opens on the late-July 2026 Plains heat dome (~2 min: July is a full 90-day
store chunk); a week in the current chunk is ~30 s, and the panel states the read
for the dates picked. A week at res 6 is 35M hour-cell answers, ~5 GB kernel peak
with the two DataFusion knobs in the fold cell (a broadcast join, a spill pool; 17 GB
without) and 35 MB per field to the browser; res 5 is the one-constant retreat.

**The data.** dynamical.org's `noaa-hrrr-analysis` v0.2.0, NOAA's 3 km hourly CONUS
analysis as an icechunk / Zarr store, CC-BY 4.0, in dynamical's own bucket on the AWS
Open Data programme, **not on Source Cooperative**:

```
s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.2.0.icechunk/
```

anonymous, us-west-2. Chunks are 2,160 hours × 45 × 45 px, so any window fetches every
filled hour of the chunk it falls in; the link, not the code, sets the pace (~21 MB/s
here, single or multi-stream). The heat notebook reads only the store columns that
touch CONUS land (523 of 960; xarray-sql partition pruning). The 48-hour forecast is on
Source Cooperative under `s3://us-west-2.opendata.source.coop/dynamical/` and the
counties film can be pointed at it with `SOURCE = "forecast"`. Notes:
`docs/hrrr-counties-notes.md`, `docs/hrrr-heat-hex-notes.md`.

## The terrain explorer (experimental)

`xsql-mapterhorn-explorer.py` draws [Mapterhorn](https://mapterhorn.com/) terrain
worldwide as extruded H3 columns, folded per viewport from PMTiles DEM archives, with a
colormap panel, an elevation scale and a resolution-offset override. Same chassis,
open defects (the res-to-zoom ladder, deep-zoom regional reads unflown, a raised offset
can refold millions of cells). A demo to poke at, not a tool to lean on.

## Everything else

`archive/` holds what was built on the way here, kept for reference and not
maintained: fire-risk buildings, the human-footprint pair, flood exposure, canopy
height, the Annual NLCD notebooks and their DataFusion-vs-DuckDB benchmark, NLCD
boundaries over imagery, the parked NLCD × terrain extrusion, and the NAIP, 3DEP and
Overture GeoParquet helpers. None of the maintained notebooks imports any of it. They
still run, from a pinned `archive/pyproject.toml`:

```bash
uv run --project archive marimo edit archive/xsql-firerisk-buildings.py
# or self-contained from the notebook's PEP 723 header
uv run marimo edit archive/xsql-firerisk-buildings.py --sandbox
```

`docs/` has the full working record and measurements for each notebook.
