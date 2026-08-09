# x-sql-marimo

Fly across the USA and watch 40 years of land cover under you, as H3 hexagons that get
finer as you zoom in. Nothing is read until the camera asks for it.

```bash
uv run marimo edit xsql-duckdb-nlcd-h3.py --sandbox
```

Annual NLCD is streamed straight out of object storage with
[obstore](https://developmentseed.org/obstore/) and
[async-geotiff](https://developmentseed.org/async-geotiff/), folded into
[H3](https://h3geo.org/) cells in SQL, and drawn with
[lonboard](https://developmentseed.org/lonboard/). No tile server, no STAC API, no pixels
leave the bucket until the viewport asks for them.

## The counter-intuitive part

Each fold reads only the padded viewport, from the overview matching the H3 resolution it
is about to build. So the **finest views are the cheapest**: the viewport shrinks faster
than the resolution grows. Band by band, res 5 at 1920 m reads about 4.1M pixels; res 11
at 30 m reads 72,890.

Res 11 is the floor, and it belongs to the data rather than the code: a res 11 hexagon
holds 2.3 pixels of 30 m NLCD, and res 12 would hold 0.6 and hole out.

## Two engines, each doing the half it wins

The division of labour was benchmarked, not assumed, and it goes both ways. Same
viewport, 1.58M pixels folded to 132,759 cells:

| | DataFusion + h3ronpy | DuckDB |
|---|---:|---:|
| **fold**: pixels to cells, majority class | **70 ms** | 462 ms |
| **dissolve**: cells to region outlines | 928 ms | **75 ms** |

The fold stays in DataFusion because h3ronpy converts a whole column at once, where
DuckDB calls `h3_latlng_to_cell` once per row, 1.58 million times. The dissolve goes to
DuckDB because its `h3` extension wraps Uber's C library, where the cells-to-polygon work
lives; h3ronpy wraps [h3o](https://github.com/HydroniumLabs/h3o), a separate Rust
implementation, so that work never reaches it.

## What the map answers

- **Colour is the majority class** in each cell. Hover for its *purity*: how much of the
  cell actually is that class.
- **Outlines** are dissolved regions, one polygon per run of touching same-class cells,
  built in a single SQL statement.
- **Draw a box** and it is folded across all 40 years of Annual NLCD, 1985 to 2024, at
  the resolution on screen when you drew it. Two answers side by side: **area** per class
  per year, which is a cell question, and **patch count**, which is not. A class can hold
  its share of the box while breaking into more pieces, and only the dissolved polygons
  see that. Kentucky, 1985 to 2024: Pasture/Hay loses 2.9 points of area while going from
  90 patches to 121.

Forty years of a drawn box costs about 300 ms of reads, for the same reason as above: the
box is small and the overview matches the resolution.

## Pipeline

1. **Camera moves.** A debounced coroutine reads the padded viewport, so a drag reads once
   at the end rather than at every position it passed through.
2. **Stream** the overview window with `obstore` + `async-geotiff`, from a 512-tile cache
   keyed per overview level, so panning re-reads only the new strip.
3. **Raster to H3 in SQL** with xarray-sql over DataFusion, `h3_latlng_to_cell` wired in
   as an h3ronpy UDF, taking the modal class per cell.
4. **Dissolve to outlines** in DuckDB: dissolve per class, `ST_Dump` into connected runs,
   drop the speckle.
5. **Render** flat H3 hexagons in NLCD's own palette, swapping traits on live layers so
   the camera never re-runs a marimo cell.

## Colour

The palette is NLCD's own, read out of the COG. The one it replaced was invented here on
a teal-to-brown axis with the forests separated by lightness, which put deciduous forest
at luminance 0.103 in a region where it is 39.7% of the cells: half the map came out near
black. Lightness was the problem, not hue.

## Also here

`xsql-nlcd-zoom.py` is the same notebook with the dissolve left in h3ronpy, kept so the
benchmark above stays runnable rather than being a claim in a commit message.
`archive/` holds the earlier notebooks this grew out of (3DEP elevation, NAIP drapes,
Overture buildings), kept for reference and not maintained.

See `CLAUDE.md` for architecture and `docs/` for the working notes.
