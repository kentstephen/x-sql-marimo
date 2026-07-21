# CLAUDE.md

Guidance for Claude Code working in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory location, colorblind-safe encodings).

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
   `xsql-dem-h3.py`: `h3_latlng_to_cell(lat, lng, res) -> UBIGINT`. Group by cell,
   aggregate elevation (mean/min/max). This is the whole reason the repo exists (the
   `x-sql` / xarray-sql + DataFusion angle).

5. **Render extruded H3.** lonboard `H3HexagonLayer(get_hexagon=table["hex"],
   get_elevation=table["elevation"], extruded=True, high_precision=True)`, elevation
   scale + opacity controls, `DarkMatterNoLabels` basemap. Reference: the layer cell in
   `3dep-seamless-duckdb-h3/naip_usgs_join_h3_1m.py`.

## H3 UDF (already present)

`xsql-dem-h3.py` registers the DataFusion UDF via h3ronpy:

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
`lonboard`. Keep each notebook's PEP 723 header in sync with `pyproject.toml` so
`--sandbox` stays self-contained. Pin the deck.gl-raster / lonboard versions; they
move fast.

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
