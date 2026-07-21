# x-sql-marimo

Free-fly across the USA: draw a box anywhere on a map and see the terrain rise as
extruded H3 hexagons, computed in SQL.

Draw an AOI, and the notebook streams the **USGS 3DEP 10m (1/3 arc-second) seamless
DEM** for that box directly from the public `prd-tnm` S3 bucket with
[obstore](https://developmentseed.org/obstore/), aggregates the elevation raster to
[H3](https://h3geo.org/) cells with a **DataFusion SQL UDF**, and renders them as an
extruded [lonboard](https://developmentseed.org/lonboard/) `H3HexagonLayer`. No tiling
server, no STAC API, no pixels leave object storage until the AOI asks for them.

## Pipeline

1. **Draw box** (lonboard `Map`, `selected_bounds` -> `mo.state`).
2. **Resolve COGs** for the AOI from the nationwide USGS seamless VRT (a local bbox
   intersection, no STAC call).
3. **Stream** the covering COG overviews with `obstore` + `async-geotiff`.
4. **Raster -> H3 in SQL** via the DataFusion UDF `h3_latlng_to_cell(lat, lng, res)`
   (h3ronpy), grouped and aggregated by cell.
5. **Render** extruded H3 hexagons, colored on a colorblind-safe viridis elevation ramp.

## Run

```bash
uv run marimo edit <notebook>.py --sandbox
```

See `CLAUDE.md` for architecture and the reference-repo lineage.
