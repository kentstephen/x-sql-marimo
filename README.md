# x-sql-marimo

Fold a raster to H3 in SQL, then join it to something that has edges. Two notebooks do
that with two different pairs, and nothing is read until the camera asks for it.

```bash
# where forest was lost 2002-2022, by administrative division, worldwide
uv run marimo edit xsql-deforest-divisions.py --sandbox

# which buildings stand on high wildfire-risk ground, CONUS
uv run marimo edit xsql-firerisk-buildings.py --sandbox
```

Both stream their raster straight out of object storage with
[obstore](https://developmentseed.org/obstore/), fold it into [H3](https://h3geo.org/) cells
in SQL, join those cells to Overture geometry on the cell id, and draw the result with
[lonboard](https://developmentseed.org/lonboard/). No tile server, no STAC API, no pixels
leave the bucket until the viewport asks for them.

The first is written up in full below, because it is where the machinery was worked out.
[The second](#the-second-notebook-which-structures-are-on-dangerous-ground) reuses it and
only the differences are worth reading.

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
| render | lonboard | |

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

Res 4 draws the whole planet comfortably: 288 thousand cells, not 288 million. The world
at res 4 is 15.7M pixels read in 821 ms plus a 282 ms fold, and 8 ms plus 211 ms the
second time off the tile cache.

### The COG is sparse and async-geotiff does not know it

73.6% of full-resolution tiles have offset 0 and length 0, because ocean is simply not
stored. A read that touches one issues the byte range `0..0` and raises

```
TypeError: ValueError: Invalid range requested, start: 0 end: 0
```

which names neither the tile nor the sparseness, so it reads like a corrupt file. Reading
on the COG's own 512 px tile grid and consulting `tile_byte_counts` first turns that from
a crash into a **speedup**: an absent tile is NaN with no request at all. This is the piece
most worth stealing from this notebook, and the piece that looks most like boilerplate.

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

- **Colour is the mean share deforested** across the cells in view, or across a division
  once boundaries are on.
- **Draw a box** with the ▢ button and the join becomes a number: every division inside
  it, ranked by mean share deforested. It reads one H3 resolution finer than the screen,
  sizes that resolution from the box rather than the current zoom, and falls back county
  to region to country, because Overture has counties for only 171 of 219 countries.
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
floor is lifted to the upper 75% of cividis and zero takes the dark end alone.

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

## The second notebook: which structures are on dangerous ground

```bash
uv run marimo edit xsql-firerisk-buildings.py --sandbox
```

Same machinery, different pair. The raster is
[carbonplan/carbonplan-ocr](https://source.coop/carbonplan/carbonplan-ocr) on Source
Cooperative (CarbonPlan's Open Climate Risk project, CC-BY 4.0), a Zarr v3 multiscale
pyramid covering CONUS at 30 m. The vector side is Overture **building footprints** rather
than divisions.

The value is **RPS, Risk to Potential Structures**: burn probability times the conditional
risk to a structure, were one there. That name is the reason for the join. RPS is computed
without knowing whether anything is actually built, so joining it to real footprints turns
"this ground is dangerous" into "this structure is on dangerous ground".

Nothing in the docs says what `rps` means, so it was settled from the sibling Icechunk
store, which publishes the factors separately: `corr(rps_2011, bp_2011 * crps_scott)` is
1.000000 over 160k pixels, the USFS Wildfire Risk to Communities formula exactly.

**Zarr turns out to be the easier side of the trade.** Three things the COG notebook builds
by hand come free: the pyramid declares `"resampling_method": "mean"` in its metadata
instead of having to be measured; absent chunks read as fill because that is in the spec,
so the sparse-tile crash has no equivalent; and the coordinates are published arrays, so
there is no geotransform to derive. obstore stays the transport either way.

**Buildings have a minimum zoom, and the tiles are always z14.** Overture's tileset carries
footprint geometry from z4, but Planetiler strips the *attributes* off everything below its
top zoom:

| | z13 features | `id` present | z14 features | `id` present |
|---|---:|---:|---:|---:|
| Paradise CA | 1,109 | **0** | 618 | 618 |
| Downtown LA | 4,162 | **0** | 1,875 | 1,875 |

`id` is both the dissolve key and the join key, so a z13 fetch returns thousands of
anonymous polygons and nothing errors: the decode succeeds, the winding is right, the count
is right, and every feature is unusable.

**The polyfill runs in `overlap` mode, where the divisions notebook uses `center`,** and
that inverts for a good reason. A division holds thousands of cells, so `center` assigns
each cell to exactly one division and the map partitions. A building is 150-250 m² against
a 2,150 m² cell: it contains no cell centre at all, and `center` returns nothing for it.
Buildings are disjoint islands rather than a partition, so the double-counting objection
that ruled `overlap` out for counties doesn't apply.

Which leaves the honest limit, and the notebook says it on screen: a house is 10% of a
res-11 cell, so its number is the cell's number. The 30 m raster does not resolve a house;
it resolves the hillside it stands on.

**Is it right?** A building's joined RPS against the raster value at its own centroid, read
straight out of the Zarr window: `corr = 0.9803`, median ratio `1.002` over 3,735 buildings.
That is the check that catches a mis-indexed read or a resolution mismatch, neither of which
changes a single row count.

`docs/firerisk-buildings-notes.md` has the rest, including why res 11 is the floor and what
the Icechunk store would buy.

## Everything else

`archive/` holds what was built on the way to these and is kept for reference, not maintained:
the Annual NLCD zoom notebooks and their DataFusion-vs-DuckDB benchmark, the NLCD boundary
over satellite imagery, the parked NLCD x terrain extrusion, and the NAIP, 3DEP and
Overture GeoParquet helpers. Neither maintained notebook imports any of it; their only
dependencies are the third-party ones in their PEP 723 headers.

They still run. `archive/pyproject.toml` is the union of every archived notebook's header,
pinned, so the root project can stay in sync with the one notebook that is maintained:

```bash
uv run --project archive marimo edit archive/xsql-nlcd-imagery.py
# or, self-contained from the notebook's own PEP 723 header
uv run marimo edit archive/xsql-nlcd-imagery.py --sandbox
```

`docs/` has the full working record for each of them, including the measurements quoted
above. `docs/deforest-divisions-notes.md` is the one for this notebook.
