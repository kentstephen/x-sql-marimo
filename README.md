# x-sql-marimo

Fold a raster to H3 in SQL, then join it to something that has edges. One notebook is
the point of the repo: worldwide deforestation, streamed straight out of object storage,
joined to administrative divisions everywhere the camera lands. A second notebook points
the same fold at the weather: HRRR temperature per US county, hour by hour, as a film
that plays in the browser. A third, experimental, applies the chassis to terrain.

```bash
# where forest was lost 2002-2022, by administrative division, worldwide
uv run marimo edit xsql-deforest-divisions.py --sandbox

# HRRR 2 m temperature per CONUS county, hour by hour, animated (dashboard on the map)
uv run marimo edit xsql-hrrr-counties.py --sandbox

# EXPERIMENTAL, use caution: worldwide Mapterhorn terrain as extruded H3 columns.
# Open defects (res-to-zoom tuning, deep-zoom regional reads unflown); expect rough
# edges and occasional refolds that cost real bandwidth.
uv run marimo edit xsql-mapterhorn-explorer.py --sandbox
```

The deforestation notebook streams its raster with
[obstore](https://developmentseed.org/obstore/), folds it into [H3](https://h3geo.org/)
cells in SQL, joins those cells to Overture geometry on the cell id, and draws with
[lonboard](https://developmentseed.org/lonboard/). No tile server, no STAC API, no pixels
leave the bucket until the viewport asks for them. The map paints the COG itself, served
as ramp-coloured tiles from the kernel; H3 is the measurement layer underneath, where the
per-division numbers come from.

## Deforestation by Overture division

A 5.7 GB global COG, read with
[async-geotiff](https://developmentseed.org/async-geotiff/), against administrative
boundaries.

### The data

**[vizzuality/lg-land-carbon-data](https://source.coop/vizzuality/lg-land-carbon-data)**
on Source Cooperative: "Land, carbon and biodiversity data for supply chain impact
calculation", built by [Vizzuality](https://www.vizzuality.com/) for LandGriffon,
CC-BY 4.0.

```
s3://us-west-2.opendata.source.coop/vizzuality/lg-land-carbon-data/deforest_100m_cog.tif
```

EPSG:4326, whole globe, 200,376 x 400,752 float32, 512 px tiles, 10 average-resampled
overviews. The value in a pixel is the **portion of that pixel deforested 2002-2022**, 0
to 1, and that single fact decides most of the design. A portion is intensive, so `mean()`
is valid at any scale, the averaged overview pyramid is legitimate rather than a lie, and
there is no majority vote or mode anywhere in the fold.

The same repository holds nine other layers (carbon, cropland expansion, biodiversity
intactness, forest landscape integrity) at the same shape and CRS, so any of them is a
one-line swap.

Boundaries are [Overture Maps](https://overturemaps.org/) divisions, read from the pinned
release's own PMTiles build.

### What the map draws

Two layers, one ramp:

- **The raster itself**, as a lonboard `RasterLayer`: the COG's own pyramid served from
  the kernel as ramp-coloured PNG tiles, so the paint is pixel-sharp at every zoom rather
  than quantised to cell means. Exact zero and no-data are transparent; this map shows
  where deforestation **is**, not where it is absent.
- **Divisions**, as a choropleth: each region, county or locality coloured by the mean
  share of its ground deforested, with a fill-opacity slider and a toggleable outline.

The division numbers do not come from the paint. They come from an H3 fold that runs
under it, which is the next section.

### Why H3 is not just a demo step

The COG is in degrees, so **its pixels are not equal area**: a 100 m pixel at the equator
covers about twice the ground of one at 60 degrees. Average pixels directly over a country
spanning many latitudes and you overweight its poleward end. H3 cells are near-equal-area,
so folding to H3 and then averaging *cells* equally is an area-weighted mean almost for
free.

Two weightings, each fixing a different bias:

- **within a cell**, weight by valid pixel count, so a coastal cell that is 90% NaN ocean
  does not count as a whole one.
- **within a division**, weight cells equally. This is the area correction, and weighting
  by pixel count here would put back exactly the latitude bias the fold removed.

Getting this wrong is invisible on screen, which is the dangerous part.

### Each engine doing the half it wins

| step | engine | why |
|---|---|---|
| stream the COG and the PMTiles | obstore | unsigned, concurrent, ranged |
| **fold** pixels to H3 cells | DataFusion + h3ronpy | whole-column conversion; DuckDB calls a UDF once per row (70 ms against 462 ms on 1.58M rows) |
| **polyfill** division polygon to cells | DuckDB `h3` | the only engine with one, and it wraps Uber's C library |
| **dissolve** tile-clipped pieces | DuckDB `spatial` | `ST_Union_Agg` per division id |
| **join** cells to divisions | DataFusion | an integer equi-join and a group-by, no geometry involved. The cells are already there |
| render | lonboard | the raster as kernel-served tiles, the divisions as polygons |

Shipping the cells over to DuckDB because DuckDB happens to hold the polygons would have
been backwards. The geometry steps go where the geometry is; the join goes where the data
is.

### The counter-intuitive part

Each fold reads only the padded viewport, from the overview matching the H3 resolution it
is about to build, so the **finest views are the cheapest**: the viewport shrinks faster
than the resolution grows.

| res | cells globally | edge | reads | px/hex |
|---|---:|---:|---|---:|
| 4 | 288,122 | 26.1 km | L6 (6.4 km) | 43 |
| 5 | 2,016,842 | 9.9 km | L5 (3.2 km) | 25 |
| 6 | 14,117,882 | 3.7 km | L3 (800 m) | 56 |
| 7 | 98,825,162 | 1.4 km | L2 (400 m) | 32 |
| 8 | 691,776,122 | 0.5 km | L1 (200 m) | 18 |

Res 4 measures the whole planet comfortably: 288 thousand cells, not 288 million. The
world at res 4 is 15.7M pixels read in 821 ms plus a 282 ms fold, and 8 ms plus 211 ms
the second time off the tile cache.

### The COG is sparse and async-geotiff does not know it

73.6% of full-resolution tiles have offset 0 and length 0, because ocean is simply not
stored. A read that touches one issues the byte range `0..0` and raises

```
TypeError: ValueError: Invalid range requested, start: 0 end: 0
```

which names neither the tile nor the sparseness, so it reads like a corrupt file. Reading
on the COG's own 512 px tile grid and consulting `tile_byte_counts` first turns that from
a crash into a **speedup**: an absent tile is NaN with no request at all. The raster tile
layer consults the same table, so an ocean tile costs nothing there either. This is the
piece most worth stealing from this notebook, and the piece that looks most like
boilerplate.

### Boundaries come from PMTiles, not GeoParquet

Overture's `division_area` GeoParquet has no spatial ordering, so geometry (99.0% of the
bytes) cannot be pruned: a Rondônia-sized viewport decodes about **190 MB** per file to
keep 6,337 rows, and no query makes that smaller. That was measured to the floor before it
was abandoned.

The same release published as PMTiles is the layout problem solved upstream, and it is the
vector twin of the COG's overview pyramid: one 19.5 GB object, anonymous ranged GETs,
Hilbert-ordered gzipped MVT, z0-12. The same viewport reads about **0.8 MB**.

| | PMTiles | GeoParquet |
|---|---:|---:|
| regions, Rondônia-sized viewport, cold | **0.74 s** | 13.8 s |
| regions, whole planet | **4.2 s** | not reachable |
| counties, Iowa view (137 rows) | **0.36 s** | |
| polyfill at res 6 (50,750 cells) | **0.09 s** | |

The PMTiles v3 reader and the MVT decode are hand-rolled on the same varint machinery,
verified ring-exact and property-exact against `mapbox-vector-tile` on ten tiles including
the world tile, Java's coastline and Italy's enclaves. Tile geometry arrives clipped, so
one division comes back as several pieces and they are dissolved per `division_id` before
anything downstream sees them, or the stroke draws tile seams across the map.

### What the map answers

- **Colour is the share of ground deforested**, per pixel on the raster paint, and per
  division once boundaries are on.
- **Press "rank what's in view"** in the controls and the join becomes a number: every
  division on screen, ranked by mean share deforested. It reads one H3 resolution finer
  than the screen, sizes that resolution from the view rather than the current zoom, and
  falls back county to region to country, because Overture has counties for only 171 of
  219 countries. The camera is the only statement of intent; it replaced lonboard's
  draw-box tool, which asked you to describe a region twice.
- **The camera answers from memory first.** `view_state` fires on every frame of a drag,
  and any frame servable from what is already folded (a pan inside the current box, a zoom
  back to a resolution already visited) is answered synchronously in the comm handler. Only
  a view that genuinely needs bytes goes through the debounce.

Subtype floors are baked into the tileset by the build and are honoured here: country z2,
region z4, county z8. Measured off the tiles, not documented anywhere.

### Colour, and why zero gets its own swatch

Folded at res 4 over the world, **69.6% of cells are exactly zero** and the nonzero values
span nine orders of magnitude (p1 7.3e-8, p50 2.1e-3, p99.9 0.45). A linear 0 to 1 ramp
paints a blank world, so the ramp is cividis, log10, over 1e-4 to 0.5.

Zero has to be separated from "almost zero" by **luminance, not hue**, and that was
measured rather than guessed: a flat neutral grey lands at luminance 0.313 and the 0.1%
stop of full-range cividis lands at 0.318, so "none" and "0.1%" came out as the same
colour, which is the worst thing this legend could do given zero is the majority case.
Hue cannot fix it, because the point of cividis is that hue carries nothing. So the ramp
floor is lifted to the upper 75% of cividis and zero takes the dark end alone. On the
raster paint zero is simply transparent; the zero swatch survives in the division fill
and the legend.

| stop | RGB | luminance | deuteranope luminance |
|---|---|---:|---:|
| none | (38,40,44) | 0.156 | 0.150 |
| 0.01% | (67,78,107) | 0.305 | 0.284 |
| 1% | (162,153,116) | 0.597 | 0.614 |
| 10% | (216,196,91) | 0.756 | 0.800 |
| 50%+ | (253,231,55) | 0.874 | 0.934 |

Monotonic in luminance both normally and under a deuteranope simulation, which is the only
thing a sequential ramp has to promise.

### Is it right?

Zonal means checked against geography we can reason about:

| place | reading |
|---|---|
| Congo basin interior (DRC), res 6 | 0.80% to 19.9%, Kisangani highest: an active frontier |
| Iowa, USA, res 6 | 0.011% to 0.296%: cleared in the 1800s, so almost nothing 2002-2022 |
| Rondônia box | Rondônia 27.008%, Mato Grosso 21.571% |
| Iowa box | Wapello 0.953%, Dallas 0.869%, Keokuk 0.754% |
| Congo box | Mbandaka 16.701%, Ngabe 8.926%, Bongandanga 7.650% |

A ~30x to ~70x split in the right direction and roughly the right magnitude. If the join
were smearing neighbours together or dropping the area weighting, that contrast would
collapse.

## HRRR temperature by county, as a film

`xsql-hrrr-counties.py` reads [dynamical.org](https://dynamical.org/)'s Zarr build of
NOAA's HRRR analysis (3 km, hourly, CC-BY 4.0) straight from S3 with
[xarray-sql](https://github.com/alxmrs/xarray-sql), labels every pixel with its H3
res 7 cell from the store's own lat/lon, polyfills Overture counties in DuckDB at the
same res, and one DataFusion join + group by gives 2 m temperature per county per hour
for a chosen window (a submit form: UTC date range, hourly or daily mean / max, with
limits). That table is small, so the whole film goes to the browser once and the
browser owns the clock: a bespoke anywidget on deck.gl + `@geoarrow/deck.gl-layers`
recolours the counties per frame from a Float32Array, and the dashboard (transport,
legend, live per-frame stats, warmest and coolest counties, a clicked county's
series, display settings) is drawn on the map itself, hideable, so the map's own
browser fullscreen carries all of it. The kernel is idle while it plays. Res 7 is
finer than the 3 km pixel, so the fold is a relabel here; the county mean is the
honest mean of its pixels. Notes and measurements: `docs/hrrr-counties-notes.md`.

## The terrain explorer (experimental, use caution)

`xsql-mapterhorn-explorer.py` draws [Mapterhorn](https://mapterhorn.com/) terrain
worldwide as extruded H3 columns: elevation folded per viewport from PMTiles DEM
archives, with a colormap panel, an elevation scale, and a resolution-offset override.
It shares this repo's chassis (the camera machinery, the canvas ruler, the PMTiles
client) but it is an experiment with open defects: the res-to-zoom ladder is still
being tuned, the deepest zoom levels are probed but unflown, and a raised res offset
can make a single view refold millions of cells. Treat it as a demo to poke at, not a
tool to lean on.

## Everything else

`archive/` holds what was built on the way here and is kept for reference, not
maintained: the fire-risk buildings notebook (CarbonPlan wildfire risk joined to
Overture footprints), the human-footprint pair, the flood-exposure experiment, the
canopy-height notebook, the Annual NLCD zoom notebooks and their DataFusion-vs-DuckDB
benchmark, the NLCD boundary over satellite imagery, the parked NLCD x terrain
extrusion, and the NAIP, 3DEP and Overture GeoParquet helpers. None of the maintained
notebooks imports any of it; their only dependencies are the third-party ones in their
PEP 723 headers.

They still run. `archive/pyproject.toml` is the union of every archived notebook's header,
pinned, so the root project can stay in sync with the notebooks that are maintained:

```bash
uv run --project archive marimo edit archive/xsql-firerisk-buildings.py
# or, self-contained from the notebook's own PEP 723 header
uv run marimo edit archive/xsql-firerisk-buildings.py --sandbox
```

`docs/` has the full working record for each of them, including the measurements quoted
above. `docs/deforest-divisions-notes.md` is the one for the deforestation notebook.
