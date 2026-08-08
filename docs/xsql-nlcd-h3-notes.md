# Annual NLCD in H3: notes

Working notes from a session that got as far as a smoke-testable notebook
(`xsql-nlcd-zoom.py`) and no further. **The notebook does not work.** The findings below
are worth keeping; the notebook is not, in its current state.

## Status: broken

Observed in the browser, not reproduced headlessly:

- **The map disappears.** `marimo export html` exits 0 and serializes an
  `H3HexagonLayer`, so every cell runs and the fold produces real data. Whatever kills it
  is on the widget side, after the data is correct. Prime suspects, untested:
  - The layer is constructed in one cell from `shown` and mutated in another cell from
    the same `shown`. That is a redundant assignment on first run and a possible
    ordering hazard on later runs.
  - res 8 is 10.6M hexagons. deck computes hexagon boundaries client side via h3-js, and
    that may simply be too many. Dropping the top band is the first thing to try.
  - `hold_trait_notifications()` around four traits including `table` may not flush the
    way the surface notebooks' version does.
- **Pitch should be 0.** `view_state={... "pitch": 35}` was a bad default. A land cover
  map is a flat map. If pitch goes to 0, the purity-as-height encoding becomes invisible
  and needs to be replaced by something else (opacity, or a second legend).

Not a bug: **years are never combined.** `year.value` selects exactly one file,
`Annual_NLCD_LndCov_{year}_CU_C1V1.tif`, and each is a standalone CONUS mosaic.

## The data

`s3://us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/`

Lists and reads anonymously with obstore. No requester-pays, no signing, no catalog to
parse: one file per (product, year), so the "which COG covers this AOI" problem that the
DEM notebooks solve with a VRT does not exist here.

- **240 COGs, 506 GB.** 6 products x 40 years (1985-2024): `LndCov` (class raster),
  `FctImp` (% impervious), `ImpDsc`, `LndChg`, `LndCnf` (confidence), `SpcChg`.
- Each file is one CONUS mosaic: 105000 x 160000, 30 m, uint8, nodata 250, DEFLATE,
  512x512 blocks, palette photometric.
- CRS is a custom `AEA WGS84` with **no EPSG code** (`crs.to_epsg()` returns None). It is
  standard NLCD Albers: `+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5
  +datum=WGS84`. `g.crs` is already a pyproj CRS, so `Transformer.from_crs` takes it
  directly.
- 6 overview levels: 60, 120, 240, 480, 960, 1920 m.

**The overviews are class-pure.** Checked at 120 m, 480 m and 1920 m: unique values are
exactly the 16 NLCD classes plus 250. Nearest or mode resampled, no averaged
pseudo-classes. This is what makes reading a coarse level honest instead of a lie, and it
is the single fact the whole design rests on.

NLCD's shipped colormap (`g.colormap.as_dict()`) is a green-forest against red-developed
palette. Treat it as metadata and never draw it. The notebook uses 7 groups on a
teal-to-brown axis with water in blue and the developed ramp carried by luminance.

## The H3 library question

`h3ronpy` 0.22 **already links h3o 0.7.0** (confirmed by pulling crate paths out of
`h3ronpyrs.abi3.so`). h3o is the HydroniumLabs pure-Rust reimplementation, not a binding
to Uber's C. The R package of the same name is JosiahParry's extendr bindings to that
same crate, now maintained under extendr. There is nothing faster to swap in.

There is no `datafusion-h3` crate. The existing UDF is not row-by-row Python: DataFusion
hands it whole batches and it makes one vectorized `coordinates_to_cells` call, so the
per-batch overhead is one `to_numpy()` and one `pa.array()`. Measured, the H3 call is
~0.1 s against a ~2 s fold. A native Rust `ScalarUDF` would be optimising the wrong end.

Module layout moved in 0.22: `h3ronpy.vector`, `h3ronpy.raster`, no `h3ronpy.arrow`, no
`h3ronpy.op`. `h3ronpy.raster.raster_to_dataframe(arr, transform, res, nodata_value=,
compact=)` and `nearest_h3_resolution(shape, transform)` exist and were never tried.

**In the browser:** h3o compiles to WASM by design, but no npm package exists, so it
would mean vendoring your own build. Note that H3 already runs in the browser here:
deck's `H3HexagonLayer` calls h3-js to turn cell ids into boundaries. Moving cell
*assignment* client side would require shipping the raster instead of the much smaller
cell table, so it makes the payload worse. The one place browser-side H3 would pay is
`change_resolution` for zoom-out, which is pure integer math on data already resident.

## lonboard's camera

From the 0.16 bundle (`static/index.js`):

```js
onViewStateChange: He => { $t(HTt(Ce, He.viewState)) }
// $t  = n => { model.set("view_state", n); jye(model) }
// jye = debounce(m => m.save_changes(), 300)
```

- `view_state` is a real two-way trait. `map.observe(fn, names="view_state")` works.
- The comm flush is **debounced 300 ms, trailing only, no maxWait**. Python hears nothing
  during a drag and gets one notification after the camera settles.
- `view_state` reaches deck as `initialViewState`, so the camera is **uncontrolled**.
  Python assignment does not fight a drag, and the camera's echo cannot feed back into
  itself. Only `set_view_state` and the `fly-to` custom message move it.

The trait is not the problem. What breaks it in marimo is the cell graph:

- If the cell that builds the `Map` reads state that the observer writes, every settle
  rebuilds the widget, resets the camera, and fires again.
- Reassigning `deck.layers` per camera event builds a new widget model each time and
  leaks the old one into the browser. A fast pan walks it into a crash. Mutate the
  existing layer's traits instead.
- The fix that matters most: **put the derived job in the state, not the camera**. The
  observer computes the resolution and returns early if unchanged, so a pan or a zoom
  nudge inside a band reaches nothing.

`MapViewState` carries `longitude, latitude, zoom, pitch, bearing, max/min_*` and
**nothing about viewport size**, so the visible bbox is not derivable from the camera. Any
viewport-bbox scheme has to assume a pixel size. This is why the first attempt did not
fill the view.

## Why viewport reads were the wrong idea

Full CONUS, measured:

| level | fold | cells | time | Arrow |
|---|---|---|---|---|
| 960 m | res 5 | 31,629 | 0.9 s | 1 MB |
| 960 m | res 6 | 218,506 | 0.9 s | 4 MB |
| 480 m | res 7 | 1,522,443 | 3.2 s | 30 MB |
| 240 m | res 8 | 10,639,312 | 13.5 s | 213 MB |

The entire country at res 7 is 3.2 s and 30 MB. There is no viewport worth computing.
Tiles, morecantile, per-tile caching and viewport bboxes were all solving a problem that
does not exist at this data size. Read once, query many times.

## xarray-sql earns its place here

Manual flatten (`np.nonzero`, per-pixel interpolation, a materialized pyarrow table)
against `ctx.from_dataset`, same fold, 480 m to res 7:

```
A  manual flatten   3.2s   1,522,443 cells   + 596 MB of Python-side columns
B  from_dataset     1.9s   1,522,443 cells
   hex identical: True    mode identical: True (after the tie-break fix)
```

`from_dataset` takes the raw 2D numpy array, no dask needed. `y` and `x` become columns,
`cls` becomes a column, streamed by chunk. `np.nonzero` becomes `WHERE cls != 250`. The
Albers-to-degrees step becomes `to_lat`/`to_lon` UDFs over the `y`/`x` columns, the same
pattern as `to_lonlat_<i>` in `xsql-s1m-h3.py`.

Reprojection detail: exact pyproj on a 64x64 control grid plus bilinear interpolation is
within ~100 m of a per-pixel transform over CONUS, which is a fraction of a pixel at these
levels. 4096 pyproj calls instead of 35 million.

## Two traps worth remembering

**1. `first_value(... ORDER BY n DESC)` is non-deterministic on ties.** Two folds that
produced byte-identical hex columns disagreed on `mode_cls` until the tie-break was made
explicit:

```sql
first_value(cls ORDER BY n DESC, cls ASC) AS mode_cls
```

This is a categorical-raster problem specifically. A 2-2 split between two classes in a
cell is common, and without the tie-break the map changes between runs.

**2. Parent rollup is not exact.** Folding at res 7 and rolling up to res 6 with
`change_resolution` is 17x faster (0.13 s vs 2.27 s) but gives a different answer:
218,682 cells against the direct fold's 218,628. H3 parent-of-centroid is not exact
containment, so pixels near cell edges land in a different parent than a direct fold puts
them in. If the levels must agree, recompute each one from the raster.

## The empty-cell cliff

Which overview supports which resolution. "Coverage" is cells found against
(CONUS area / average cell area); "1-px cells" is the share of cells holding a single
pixel, which is the signature of a grid too fine for its source.

```
 960m -> res6:    218,506 cells | coverage 102.9% | 1-px   0.1%
 960m -> res7:  1,521,538 cells | coverage 102.4% | 1-px   0.1%
 960m -> res8:  8,584,126 cells | coverage  82.5% | 1-px  97.9%   <- holes
 480m -> res7:  1,522,443 cells | coverage 102.5% | 1-px   0.0%
 480m -> res8: 10,635,430 cells | coverage 102.3% | 1-px   0.1%
 480m -> res9: 35,067,095 cells | coverage  48.2% | 1-px 100.0%   <- holes
 240m -> res8: 10,639,312 cells | coverage 102.3% | 1-px   0.0%
 240m -> res9: 74,415,097 cells | coverage 102.2% | 1-px  24.7%
 240m -> res10:140,284,941 cells| coverage  27.5% | 1-px 100.0%   <- holes
```

The cliff is sharp. Each overview supports exactly one resolution past the obvious
pairing and then collapses. Go finer than the source supports and the map gets holes,
because no pixel centre lands inside those cells. The native 30 m data tops out somewhere
around res 11 (cell edge ~25 m).

H3 average cell areas at CONUS latitude, for reference: res 5 = 266 km2, res 6 = 38.0,
res 7 = 5.44, res 8 = 0.777, res 9 = 0.111, res 10 = 0.0159.

## lonboard API gotchas hit along the way

- `basemap=` wants `MaplibreBasemap(style=CartoBasemap.X)` in 0.16, not the bare enum.
- `async_geotiff` reads want `Window(col_off=, row_off=, width=, height=)`. Passing a
  bare `(r0, r1, c0, c1)` tuple is silently misread as col_off/row_off/width/height and
  starts pulling most of CONUS. This looks exactly like a network hang.
- The layer's `table` trait coerces anything arrow-ish in `__init__` but `validate()` is a
  strict `isinstance(value, arro3.core.Table)`. An assignment that works at construction
  fails afterwards. Convert with `ArroTable.from_arrow(...)`.
- lonboard refuses to serialize a zero-row table, so there is no empty placeholder.
- All columns must agree about chunking. DataFusion returns many chunks and
  numpy-derived columns return one, so `combine_chunks()` first.

## If picked up again

Start from the pieces, not the notebook. In rough order of what would settle the design:

1. Find what kills the map. Try res 6 only, pitch 0, and a single assignment site for the
   layer traits, then add back.
2. Decide what carries purity once the map is flat. Height is gone; opacity is the
   obvious substitute and it stacks with hue without adding a colour axis.
3. `h3ronpy.raster.raster_to_dataframe` was never benchmarked against the SQL fold. It is
   thread-pooled in Rust and goes array to H3 table in one call. It might make the whole
   `to_lat`/`to_lon` UDF layer unnecessary, at the cost of the SQL being the point.
4. The time axis is the interesting thing here and is untouched. 40 years on an identical
   grid means a cell folded once is folded for every year, and `LndChg`/`SpcChg` are
   pre-computed change products. A year slider over stable cells is a different and
   better app than a zoom demo.
