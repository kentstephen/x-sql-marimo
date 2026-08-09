# CLAUDE.md

Guidance for Claude Code working in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory location, colorblind-safe encodings).

## Repository layout

`xsql-nlcd-zoom.py` is the notebook this repo is for. Everything else that was built
along the way lives in `archive/` (earlier notebooks, the Overture and NAIP helpers, the
lonboard patch script) and is kept for reference, not maintained. The published notebook
imports none of it: its only dependencies are the third-party ones in its PEP 723 header.

## Project overview

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

## Colorblind-safe rendering (hard requirement)

Stephen is red-green colorblind. Never encode elevation (or anything) on a red-green
axis. Default to **viridis / cividis** luminance ramps (viridis is already the choice
in `s1m_viewer.py`) and lean on **extrusion height** as a redundant, non-color cue.

## Environment

```bash
# Dev (full venv)
uv run marimo edit <notebook>.py

# Shareable sandbox (PEP 723 inline deps in the notebook header)
uv run marimo edit <notebook>.py --sandbox

# Headless smoke test (runs every cell, no browser)
uv run marimo export html <notebook>.py -o /tmp/out.html
```

Core deps (see `pyproject.toml`): `marimo`, `datafusion`, `h3ronpy`, `pyarrow`,
`xarray-sql`, plus the streaming/render stack to add: `obstore`, `async-geotiff`,
`lonboard`. `duckdb` (with `INSTALL spatial`) is there for exactly one job: reading the
S1M footprint GeoPackage, where `ST_Read` parses the geometry blobs and `ST_Transform`
takes Albers to degrees. It is not a second query engine for the fold; that stays
DataFusion. Keep each notebook's PEP 723 header in sync with `pyproject.toml` so
`--sandbox` stays self-contained. Pin the deck.gl-raster / lonboard versions; they
move fast.

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
