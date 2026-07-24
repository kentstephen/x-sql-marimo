# `xsql-dem-1m.py`: context and open work

Working notes for the 1-meter notebook. The 10m notebook (`xsql-dem-rem.py`) reads the
**seamless** DEM, where one nationwide VRT gives one answer per AOI. The 1m product is
**not seamless**, and almost everything below follows from that.

Last updated 2026-07-24.

---

## MAJOR TODO: stop auto-loading. Make selection visual and explicit.

**This is the next piece of work, and it is a redesign, not a tweak.**

The notebook currently inherits the 10m notebook's reactive flow: change the AOI, and it
immediately resolves tiles and streams them into the H3 pipeline. That is fine at 10m,
where the data volume is small. It is wrong at 1m, for two separate reasons.

### 1. Intersecting tiles are not a mosaic

The current code takes every tile the catalog returns for the AOI, groups by project
directory, and treats each group as a "mosaic". That is an assumption, not a fact. Tiles
that intersect an AOI may not form a usable contiguous surface: they can be different
vintages, different quality levels, have gaps between them, or simply be the ragged edge
of a collection that mostly lies elsewhere. Calling that a mosaic is wrong, and the
notebook's own vocabulary ("mosaic", "complete") currently overstates what it knows.

### 2. The volume does not permit speculative loading

1m is roughly 100x the pixels of 10m for the same ground area. A modest AOI at H3 res 13
already trips the 5,000,000-hexagon guard, and that guard firing is the *expected* case,
not an edge case. Streaming first and asking questions later is the wrong default here.

### What it should do instead

**The notebook has to STOP at the picker map.** Not "load less", not "warn earlier".
Execution halts there and waits for the user. Nothing downstream of the picker runs until
a person has drawn an AOI and chosen tiles.

The required order, with a hard stop between each stage:

1. **Nothing loads until an AOI is drawn.** No default AOI streaming on open. The picker
   comes up empty and waits.
2. **AOI drawn** -> resolve and display candidates only: footprints, tile outlines,
   coarse previews. This stage reads overviews at most. It must never touch full
   resolution.
3. **Tiles selected explicitly.** The user picks the individual tiles, or composes a
   mosaic from them, or picks exactly one tile to look at. Multi-select at the *tile*
   level. Not "whatever the project grouping produced".
4. **Explicit trigger** (run button) -> only now does the scene build and stream.

The failure this prevents: silently pulling ~18 tiles at millions of points each because
they happened to intersect the box. That is not a mosaic, it is everything that touched
the AOI, and at 1m it is an enormous amount of data to move on an accidental drag of the
mouse.

Supporting requirements:

- Surface the cost *before* committing: tile count, estimated pixels, estimated hexagons.
  The user should decide with the number in front of them.
- Stop calling a set of intersecting tiles a "mosaic" unless it has been shown to be
  contiguous. Say what is actually known: N tiles from project X covering M% of the AOI.
- The 5M-hexagon guard stays, but demoted to a last-resort backstop. It is currently doing
  the job that explicit selection should be doing, which is why it fires so routinely.

`mo.stop` at the picker is the marimo-native way to enforce the halt, with the selection
UI and the run button gating everything below it.

---

## Catalogs (two, deliberately not joined)

### TNM Access API (what actually streams)
`https://tnmaccess.nationalmap.gov/api/v1/products`, dataset
`Digital Elevation Model (DEM) 1 meter`. AOI-scoped, keyless, and every item carries its
COG's `prd-tnm` S3 URL plus a lon/lat footprint. Throws intermittent 500s, so the query
retries.

Chosen over the alternatives because there is no nationwide VRT for 1m the way there is
for 1/3 arc-second, and `1m/FullExtentSpatialMetadata/FESM_1m.gpkg` is **1.86 GB**, far
too heavy to pull per AOI.

### 3DEP Elevation Index layer 18 (the discovery overlay)
`https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/18/query`
("1 Meter"). Real project footprint polygons: the same service the National Map downloader
draws. Three hard-won quirks:

- `returnGeometry=true` returns **500** unless you also pass `maxAllowableOffset`. The
  ungeneralised polygons are too large to serialise. `0.0005` degrees (~50 m) is plenty.
- `spatialRel=esriSpatialRelContains` returns **nothing**, even for footprints that
  plainly swallow the AOI. Do not trust server-side relate. Coverage is computed locally
  with `matplotlib.path` using even-odd fill, so interior data gaps correctly do *not*
  count as coverage.
- Its `project` field does **not** always match the staged-products directory name
  (footprint `NH_Connecticut_River_2015` vs TNM `NH_CT_RiverNorthL6_2015`). This is why
  the two catalogs are kept separate rather than joined by name. Joining them needs a
  spatial strategy, not string matching.

### Unexplored
`1m/FullExtentSpatialMetadata/10_km_cell_grid.gpkg` is only **51 MB**. GeoPackage is
SQLite, so stdlib `sqlite3` plus `geoarrow.rust.core.from_wkb` could read it with no
geopandas dependency. Possible replacement for the per-AOI ArcGIS round trip.

### Not an option
The National Map downloader **cannot be embedded**. Both
`apps.nationalmap.gov/downloader/` and `/lidar-explorer/` send
`X-Frame-Options: SAMEORIGIN`, which is enforced browser-side. Even if it rendered, the
iframe is cross-origin with no postMessage API, so nothing could come back out of it.

---

## THE FORKING POINT: where UTM to lon/lat happens

1m tiles are NAD83 UTM, one zone per tile, so something must reproject before H3 can bin
anything.

### pyproj cannot be called from a DataFusion UDF

Verified, not assumed. DataFusion executes UDFs on Rust-spawned worker threads. pyproj's
`Transformer` wraps a PROJ context backed by SQLite, and calling into it from those
threads kills the process from C++ rather than raising a Python exception. In marimo the
symptom is the kernel dying with "failed to connect" and no traceback.

| attempt | result |
| --- | --- |
| shared `Transformer` via `lru_cache` | abort: `SQLite error on SELECT name FROM "geodetic_datum": column index out of range` |
| thread-local `Transformer`s | bus error inside `from_crs` |
| lock around construction only | segfault |
| one global lock around **all** pyproj work | segfault |
| `target_partitions=1` | segfault |

The global lock failing is the decisive result: it rules out a data race. The threads
themselves are the problem, so no amount of serialisation fixes it.

### Nothing in the Rust stack fills the gap

- **geodatafusion**: no `ST_Transform` in the function tables, and no proj crate in
  `Cargo.toml` (it pulls `geo`, `geos`, `geoarrow`). Has Python bindings, so it remains
  interesting for other spatial predicates.
- **geoarrow-rs**: `GeoType` carries CRS *metadata* only (`crs`, `with_crs`). No proj
  crate anywhere in the workspace.
- **async-tiff / async-geotiff**: pixel-to-CRS mapping only (`crs`, `transform`, `xy`,
  `index`). `crs` returns a pyproj object; there is no CRS-to-CRS transform.

### Three options, and the one chosen

1. **Inline polynomial in SQL**: transform in SQL, but ~20 magic float literals per tile
   in the query text. Rejected as unreadable.
2. **Reproject in Python before SQL**: `SELECT lat, lon, elevation FROM dem_n`. Cleanest
   SQL, but the transform sits outside the query. Known-good fallback.
3. **Per-tile closure UDF. CHOSEN.**

Option 3 fits the projection rather than calling it. Inverse transverse Mercator is smooth
over a 10 km tile, so lon and lat are each an order-3 polynomial in `(x - cx, y - cy)`.
pyproj runs **once per tile on the main thread** to produce 20 coefficients; those are
captured in a closure registered as `to_lonlat_<i>`, making the UDF pure numpy with no
native per-thread state. Same reason the h3ronpy UDFs are safe there.

Accuracy, measured against pyproj on an independent dense grid over a full 10 km tile:

| fit | coefficients per axis | max error | mean error |
| --- | --- | --- | --- |
| order 1 (affine) | 3 | 4056 mm | 1205 mm |
| order 2 | 6 | 3.03 mm | 0.81 mm |
| **order 3** | 10 | **0.015 mm** | 0.005 mm |

`fit_lonlat` raises if the fit exceeds 1 mm, so a bad fit fails loudly instead of silently
translating the scene. The resulting SQL reads the way an `ST_Transform` call would:

```sql
SELECT p.lat AS lat, p.lon AS lon, elevation
FROM (SELECT to_lonlat_0(x, y) AS p, elevation
      FROM dem_0 WHERE elevation = elevation)
```

Verified end to end on the same 28,132,105-pixel window that crashed every pyproj attempt:
fit worst case 0.0037 mm, pixel-centre check against pyproj 0.0036 mm max, UDF executed on
**16 threads** without incident, 456,648 cells through the H3 fold and the flow join.

---

## Other implementation notes

- **`h3_grid_disk` is a UDF** (`(cell, k) -> LargeList<UBIGINT>`), so the flow calculation
  is `unnest` + a self-join + `avg()` in SQL rather than a Python loop over a dict of every
  cell. It runs as a **second statement** against a re-registered `scene` table, because
  DataFusion does not materialise CTEs and referencing the scene twice in one query would
  re-run the entire stream-transform-fold pipeline per reference.
- **Footprint geometry to lonboard** goes GeoJSON to WKT to `geoarrow.rust.core.from_wkt`,
  re-parsed into the same type tagged `EPSG:4326` so lonboard does not warn about an
  unverifiable CRS. No geopandas needed.
- **Colours for the footprint overlay** are Okabe-Ito, ordered so the blue / orange / sky /
  pink / yellow block is used first. Hue names *which* collection only; the coverage answer
  lives in the table and in outline weight, never in hue.
- **Compass button**: `NavigationControl(visualize_pitch=True)` makes it call
  `resetNorthPitch()`, so one click returns to north-up **and** flat. The default is
  bearing-only. Applied to both notebooks.
- **Default AOI**: Yazoo, Mississippi Delta
  `(-90.56554, 32.80851, -90.317169, 33.162963)`. Flat floodplain cut by meander scars, so
  elevation alone reads as a flat sheet and the flow offset does the work.
- **H3 resolutions** 11-15, default 13. The 5M-hexagon pre-stream guard is inherited from
  the 10m notebook. See the major TODO above: at 1m this guard fires routinely.

---

## Verification status

- The projection fit, the H3 fold, and the flow join are verified in a standalone script
  against real data on 16 threads.
- **The notebook itself has not been executed since the option-3 edits.** The logic is
  identical to what passed standalone, but that is not the same as the notebook running.
