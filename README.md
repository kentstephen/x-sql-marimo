# x-sql-marimo

Draw a box anywhere in the lower 48 and see the terrain rise as extruded H3 hexagons.

**What runs where (honest version):** most of this is Python, one step is SQL. Streaming
the elevation tiles (obstore + async-geotiff), turning the raster into a table, coloring,
and rendering (numpy + lonboard) are all Python. The one SQL step is the binning: xarray-sql
(DataFusion under the hood) runs a query that folds the pixels into H3 cells and averages
elevation per cell, with H3 wired in as an h3ronpy UDF.

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
