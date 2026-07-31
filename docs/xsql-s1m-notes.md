# S1M notebook notes (`xsql-s1m-h3.py`)

Third notebook in the set. `xsql-dem-rem.py` is 10m seamless, `xsql-dem-1m.py` is the
project-staged 1m product, this one is **S1M**: the USGS seamless 1-metre mosaic, already
cut to one national 10 km grid with no project ambiguity to resolve.

## Shape

Same flow as the other two notebooks: coverage on the picker, draw a box, stream, fold to
H3, render. Two differences from `xsql-dem-1m.py`:

- **No default AOI.** State starts at `None` and `mo.stop` halts at the picker until a box
  is drawn. Nothing streams on load.
- **The picker opens on national coverage**, so you can see where the product exists
  before drawing. That is affordable here and nowhere else: the whole S1M index is one
  ~15 MB file.

Everything downstream (window read, H3 fold, flow join, extruded scene, palette / contrast
/ flow-gain / opacity controls) is carried over from `xsql-dem-1m.py` unchanged.

## The catalog is one file

`s3://prd-tnm/StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg`,
~15 MB, cached to `.cache/`. GeoPackage is SQLite, so **stdlib `sqlite3` reads it**: no
geopandas, no duckdb, no GDAL. AOI resolution is then a local array intersection, no API
call per box.

**The schema moved between April and July 2026.** The older file (used by
`3dep-seamless-duckdb-h3/s1m_viewer.py`) had a single `S1M_Products` table with
`cell_name` / `dataset` / `byte_count`. The current file has two feature tables:

| table | rows (2026-07-31) | meaning |
| --- | --- | --- |
| `current` | 11,717 | the published mosaic, one row per grid cell |
| `historical` | 11,841 | superseded versions |

Columns now: `nwcorner`, `tile`, `production_date`, `pub_date`, `horiz_crs_epsg`,
`vert_crs_epsg`, `tile_version`, `data_source_count`, `z_min`, `z_max`, `dataset_link`,
`spatial_metadata_link`, `metadata_link`. No `byte_count`, so read size is reported as
window pixels instead.

Geometry needs **no WKB parse**. The GeoPackage binary header carries the envelope when
flag bits 1-3 are non-zero (4 doubles at offset 8, `xmin, xmax, ymin, ymax`), and every
S1M footprint *is* its envelope: an axis-aligned 10 km square in EPSG:6350. Corners are
reprojected to lon/lat once, vectorised over the whole product, on the main thread.

`z_min` carries the tile nodata sentinel (-999999) wherever a tile has holes, so the
coverage shading reads `z_max` only.

## Coverage layer: dissolve it, do not draw tiles

Drawn the way `s1m_viewer.py` draws it: `SolidPolygonLayer`, viridis by `z_max` clipped to
the 1st-99th percentile, `opacity=0.35`, `pickable=False`, **no outlines**. At low opacity
with no strokes, neighbouring footprints blend into one continuous field: a dissolved
coverage shape with elevation context.

What not to do, learned by doing it:

- **Outlining each tile** turns the carpet into a grid of 11,717 boxes. That is what makes
  it look wrong; the per-tile fill is fine once the strokes are gone.
- **`pickable=True`** pops a "Feature properties" panel over the map on hover.
- **Per-tile bitmap previews** of the AOI's tiles (coarse overviews as `BitmapLayer` quads)
  read as an uneven patchwork of PNG panels with visible seams and nodata holes. Removed.
  The picker shows coverage and your box, nothing else.
- Palettable has **no Cividis**. Viridis here, matching the rest of the repo.

## Projection

Same constraint and same fix as the 1m notebook: pyproj cannot run inside a DataFusion UDF
(it kills the process from the worker threads), so lon/lat is **fitted** as an order-3
polynomial per tile on the main thread and applied in a pure-numpy UDF.

**Conditioning bug, fixed here, still latent in `xsql-dem-1m.py`.** The design matrix was
built on raw metre offsets. An AOI edge routinely clips a tile into a sliver (a real one:
`(1929996, 2620000, 1930000, 2630000)`, 4 m wide by 10 km tall). At that aspect ratio the
`u^3` column is ~1e-10 of the `v^3` column, `lstsq` drops it as noise, and the fit comes
back **4367 mm** wrong, tripping the 1 mm tolerance. Fix: normalise both axes to [-1, 1]
over the window before fitting (`sx`, `sy` carried in the coefficient tuple and applied in
the UDF). `xsql-dem-1m.py` has the identical unnormalised call and will hit this on any
sliver window.

Fit accuracy on square windows, Albers, order 3:

| tile | fit error |
| --- | --- |
| n2610e1740 | 0.028 mm |
| n0490e1390 | 0.045 mm |
| west edge (-2000000, 3000000) | 0.023 mm |

## Verification (2026-07-31)

- `marimo export html` runs clean and halts at the picker as designed.
- Catalog to H3 exercised headlessly against live S3 on two AOIs: single-tile (292,980 px,
  15,987 cells) and a four-tile seam-straddling box (15,966 cells). Elevation and flow both
  populated.
- Driven interactively in the browser; the sliver-window fit crash above came from that
  run and is fixed. Not re-driven end to end since the `SolidPolygonLayer` change.
