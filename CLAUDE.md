# CLAUDE.md

Guidance for Claude Code working in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory location, colorblind-safe encodings).

## Repository layout

**Two notebooks are the repo.** Everything else is in `archive/`, kept for reference and
not maintained.

`xsql-firerisk-buildings.py` folds CarbonPlan's 30 m CONUS wildfire-risk **Zarr v3
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

Neither notebook imports anything from `archive/`: their only dependencies are the
third-party ones in their PEP 723 headers.

**The PMTiles reader and MVT decode are shared by copy, not by import.** The divisions
notebook's version was ported from the parked terrain notebook, and the buildings notebook's
from the divisions one. A fix to the directory walk or the varint machinery in one of them
should be carried to the other by hand.

### What is in `archive/`, and what each one still proves

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
- `xsql-nlcd-sentinel2.py` is an empty placeholder from the abandoned Sentinel-2 render.
- The rest (`xsql-dem-*`, `xsql-naip-*`, `xsql-s1m-*`, `naip.py`, `overture_core.py`,
  `tools/patch_lonboard_surface.py`) are the earlier notebooks and helpers.

Paths in the sections below that name an archived notebook still resolve, with `archive/`
in front.

## Current project

`xsql-deforest-divisions.py`, described above. The shape of it, in one line: **free-fly
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

Stephen is red-green colorblind. Never encode elevation (or anything) on a red-green
axis. Default to **viridis / cividis** luminance ramps (viridis is already the choice
in `s1m_viewer.py`) and lean on **extrusion height** as a redundant, non-color cue.

## Environment

```bash
# Dev (full venv)
uv run marimo edit xsql-firerisk-buildings.py

# Shareable sandbox (PEP 723 inline deps in the notebook header)
uv run marimo edit xsql-firerisk-buildings.py --sandbox

# Headless smoke test (runs every cell, no browser)
uv run marimo export html xsql-deforest-divisions.py -o /tmp/out.html

# An archived notebook, from the archive's own environment
uv run --project archive marimo edit archive/xsql-nlcd-imagery.py
```

**Two pyprojects, deliberately.** The root `pyproject.toml` is the union of the two
maintained notebooks' PEP 723 headers and nothing more, so it stays honest about what is
actually imported (`async-geotiff` for the COG reader, `zarr` for the pyramid reader). `archive/pyproject.toml` is the union of every archived
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
