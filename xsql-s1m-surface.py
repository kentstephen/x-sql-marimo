# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "palettable>=3.3",
#     "matplotlib",
#     "arro3-core",
#     "arro3-io",
#     "numpy",
#     "pyproj>=3.7",
# ]
# ///
"""S1M -> H3 -> ONE TEXTURED MESH. A look-at-it demo, not a replacement for the real one.

The problem this exists to answer: `H3HexagonLayer` costs a full extruded prism per cell,
so a scene that is interesting (hundreds of thousands of cells) is a scene that is slow to
fly. Two things measured and killed before this: band compaction (data-dependent and lossy,
and 1 m lidar noise sets the floor) and dissolve-to-polygons (pays only where neighbours are
IDENTICAL, and flow_gain is a deliberately high-frequency per-cell signal, so 109k cells
dissolved to 74k regions).

The move here is different: stop sending the hexagons to the GPU at all.

  * The GEOMETRY becomes one regular triangle mesh over the AOI, with vertex z sampled from
    the H3 field. Its cost is whatever mesh density you pick and is COMPLETELY DECOUPLED
    from the cell count. 200k cells and 2M cells cost exactly the same to draw.
  * The STYLING becomes a texture. Each texel is looked up through `coordinates_to_cells`
    to the H3 cell that contains it and painted with that cell's colour, the same
    elevation + flow_gain composite the real notebook feeds `get_fill_color`. So the
    hexagons are still there, crisp and nearest-neighbour sampled, they just cost one
    image upload instead of N prisms.

The honest trade, stated up front:

  * It is a CONTINUOUS SURFACE, not a field of columns. No vertical walls between cells.
    Hexagons read as flat-shaded tiles painted on terrain, not as prisms. Whether that is
    the same picture to you is exactly what this notebook is for.
  * No picking. `SimpleMeshLayer` has no per-feature hit test (lonboard has no selection
    model for `H3HexagonLayer` either, so nothing is lost today, but this forecloses it).
  * Mesh z is SMOOTH. Set mesh density at or above hex density and you get stepped hex
    plateaus; below it, the surface interpolates between cell heights while the texture
    keeps full hex detail. The slider lets you find where that stops mattering.

SMOOTHING. "Can we smooth the hexes" has an answer that is not "go to a finer resolution"
(each H3 level is 7x the cells). Because the styling is now an image, the shading VALUE can
be blurred in texture space before it is coloured, which softens the plateau walls without
touching mesh height and without refolding anything. The `smooth` slider does that, NaN-aware
so the blur cannot bleed zeros in from outside the scene. At 0 you get hard hex edges.

CONTROLS. The full viz set from `xsql-s1m-h3.py` is here: palette, elevation scale, flow
offset, opacity, reverse, and the contrast range slider under a strip of the live ramp that
doubles as the legend. None of them rebuild the Map. The layer is built once from placeholder
geometry and the update cell at the bottom swaps traits on it, so the view you flew to
survives every adjustment. Only the renderer radio rebuilds.

`SurfaceLayer` is experimental and unexported (it is not in `lonboard.experimental.__init__`,
only `TextLayer` is). It works. One bug is patched below; see the PARQUET PATCH cell.

AOI is fixed to Mount Washington, New Hampshire. No picker: this notebook is about what the
render looks like, and a picker is just a way to get a different scene slowly.

Run:  uv run marimo edit xsql-s1m-surface.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import sqlite3
    import struct
    from io import BytesIO

    import h3ronpy
    import numpy as np
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    from pyproj import Transformer

    from arro3.core import Table
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from lonboard import Map, H3HexagonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard.colormap import apply_continuous_cmap
    from lonboard.controls import FullscreenControl, NavigationControl, ScaleControl

    # SurfaceLayer is real but unexported: import it off the private module.
    from lonboard.experimental._surface import SurfaceLayer

    return (
        BytesIO,
        CartoBasemap,
        FullscreenControl,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        NavigationControl,
        S3Store,
        ScaleControl,
        SurfaceLayer,
        Table,
        Transformer,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        coordinates_to_cells,
        h3ronpy,
        mo,
        np,
        pa,
        pathlib,
        sqlite3,
        struct,
        udf,
        xr,
    )


@app.cell
def _(BytesIO):
    # THE PARQUET PATCH. Without this the kernel SEGFAULTS the moment a SurfaceLayer is
    # constructed, with no traceback, which is a miserable thing to debug from scratch.
    #
    # lonboard ships every synced arrow column to the browser as Parquet, and prefers
    # pyarrow's ParquetWriter over arro3's because pyarrow picks better encodings. The mesh
    # traits are FixedSizeList: tex_coords is 2-wide, positions and triangles are 3-wide.
    # Handing pyarrow 25.0.0 a 3-wide FixedSizeList that arrived over the arro3 C Data
    # Interface crashes inside ParquetWriter.__init__. Measured, narrowly:
    #
    #     FixedSizeList(2 x Float32)  via arro3 -> pyarrow   OK
    #     FixedSizeList(3 x Float32)  via arro3 -> pyarrow   SIGSEGV
    #     FixedSizeList(3 x UInt32)   via arro3 -> pyarrow   SIGSEGV
    #     the same three shapes built natively in pyarrow    all OK
    #     the same three shapes through arro3.io.write_parquet  all OK
    #
    # So it is the handoff, not either library alone, and `positions` cannot avoid being
    # 3-wide. lonboard already has the escape hatch: write_parquet_batch falls back to
    # arro3's own writer if pyarrow is not installed. This forces that branch always.
    # Costs a few percent of file size. Nothing else in the notebook notices.
    import lonboard._serialization as _ser
    from arro3.io import write_parquet as _write_parquet

    def _write_parquet_batch(record_batch):
        if record_batch.num_rows == 0:
            raise ValueError("Batch with 0 rows.")
        bio = BytesIO()
        _write_parquet(
            record_batch,
            bio,
            compression="ZSTD(7)",
            max_row_group_size=record_batch.num_rows,
        )
        return bio.getvalue()

    _ser.write_parquet_batch = _write_parquet_batch
    print("parquet writer forced to arro3 (pyarrow segfaults on FixedSizeList(3))")
    return


@app.cell
def _(mo):
    mo.md("""
    # S1M -> H3 -> one textured mesh

    Same pipeline as `xsql-s1m-h3.py` up to the H3 fold. Then instead of one extruded
    prism per cell, the scene becomes **one triangle mesh** (geometry) plus **one
    image** (styling). Flip the renderer at the bottom to compare against the real
    `H3HexagonLayer` on the identical data.
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, h3ronpy, np, pa, udf):
    # Verbatim from xsql-s1m-h3.py: fit lon/lat per tile as an order-3 polynomial (pyproj
    # cannot run inside a DataFusion UDF; it aborts the process from worker threads) and
    # register the two h3ronpy UDFs.
    PROJ_ORDER = 3

    def _design(u, v, order=PROJ_ORDER):
        cols = [np.ones_like(u)]
        for total in range(1, order + 1):
            for i in range(total + 1):
                cols.append(u ** (total - i) * v**i)
        return np.column_stack(cols)

    def fit_lonlat(crs, bounds, samples=12, check=64, tol_mm=1.0):
        """Fit lon/lat over a tile's extent. Main thread only: this is the pyproj call."""
        left, bottom, right, top = bounds
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
        sx = max((right - left) / 2.0, 1e-9)
        sy = max((top - bottom) / 2.0, 1e-9)

        fx, fy = np.meshgrid(
            np.linspace(left, right, samples), np.linspace(bottom, top, samples)
        )
        flon, flat = inv.transform(fx.ravel(), fy.ravel())
        A = _design((fx.ravel() - cx) / sx, (fy.ravel() - cy) / sy)
        clon = np.linalg.lstsq(A, flon, rcond=None)[0]
        clat = np.linalg.lstsq(A, flat, rcond=None)[0]

        tx, ty = np.meshgrid(
            np.linspace(left, right, check), np.linspace(bottom, top, check)
        )
        tlon, tlat = inv.transform(tx.ravel(), ty.ravel())
        B = _design((tx.ravel() - cx) / sx, (ty.ravel() - cy) / sy)
        err_m = np.hypot(
            (B @ clat - tlat) * 111_320.0,
            (B @ clon - tlon) * 111_320.0 * np.cos(np.radians(tlat)),
        )
        err_mm = float(err_m.max() * 1000.0)
        if not np.isfinite(err_mm) or err_mm > tol_mm:
            raise RuntimeError(
                f"lon/lat fit for {crs} over {bounds} is off by {err_mm:.3f} mm "
                f"(tolerance {tol_mm} mm). Raise PROJ_ORDER or shrink the window."
            )
        return (cx, cy, sx, sy, clon, clat), err_mm

    def make_lonlat_udf(name, fit):
        """One UDF per tile, its fitted coefficients closed over. Pure numpy inside."""
        cx, cy, sx, sy, clon, clat = fit

        def _to_lonlat(x, y):
            A = _design((x.to_numpy() - cx) / sx, (y.to_numpy() - cy) / sy)
            return pa.StructArray.from_arrays(
                [pa.array(A @ clon), pa.array(A @ clat)], names=["lon", "lat"]
            )

        return udf(
            _to_lonlat,
            [pa.float64(), pa.float64()],
            pa.struct([("lon", pa.float64()), ("lat", pa.float64())]),
            "stable",
            name=name,
        )

    def _latlng_to_cell(lat, lng, res):
        return pa.array(
            coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py())
        )

    def _grid_disk(cell, k):
        return pa.array(h3ronpy.grid_disk(cell, k[0].as_py()))

    def make_h3_context():
        ctx = XarrayContext()
        ctx.register_udf(
            udf(
                _latlng_to_cell,
                [pa.float64(), pa.float64(), pa.int32()],
                pa.uint64(),
                "stable",
                name="h3_latlng_to_cell",
            )
        )
        ctx.register_udf(
            udf(
                _grid_disk,
                [pa.uint64(), pa.int32()],
                pa.large_list(pa.uint64()),
                "stable",
                name="h3_grid_disk",
            )
        )
        return ctx

    print("xarray-sql context factory ready")
    return fit_lonlat, make_h3_context, make_lonlat_udf


@app.cell
def _(np, pathlib, sqlite3, struct):
    # Verbatim from xsql-s1m-h3.py: the national S1M index is one ~15 MB GeoPackage, which
    # is SQLite, so stdlib sqlite3 reads it and every AOI is answered from a local file.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"
    GPKG_KEY = "StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg"
    CACHE = pathlib.Path(".cache")

    def fetch_index(refresh=False):
        CACHE.mkdir(exist_ok=True)
        path = CACHE / "S1M_Products.gpkg"
        if refresh or not path.exists():
            import urllib.request

            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(S3_BASE + GPKG_KEY, timeout=300) as r, tmp.open(
                "wb"
            ) as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
        return path

    def _envelope(blob):
        flags = blob[3]
        if (flags >> 1) & 0x07 == 0:
            raise ValueError("S1M footprint has no envelope in its GPB header")
        little = bool(flags & 0x01)
        xmin, xmax, ymin, ymax = struct.unpack_from("<4d" if little else ">4d", blob, 8)
        return xmin, ymin, xmax, ymax

    _path = fetch_index()
    with sqlite3.connect(f"file:{_path}?mode=ro", uri=True) as _con:
        _rows = _con.execute(
            "SELECT geom, tile, production_date, dataset_link FROM current"
        ).fetchall()

    _alb = np.array([_envelope(r[0]) for r in _rows], dtype="float64")
    tiles_all = [
        {
            "tile": r[1],
            "key": r[3].split("amazonaws.com/", 1)[-1],
            "produced": r[2] or "",
            "albers": tuple(_alb[i]),
        }
        for i, r in enumerate(_rows)
    ]
    tiles_albers = _alb

    print(f"S1M index: {len(tiles_all):,} current tiles from {_path}")
    return S3_BASE, tiles_albers, tiles_all


@app.cell
def _():
    # FIXED AOI. Mount Washington, New Hampshire: the summit cone, Tuckerman and Huntington
    # ravines on the east face, and enough of the Cog railway ridge to give the surface
    # something to do. ~7 x 9 km, which is a real scene rather than a toy one.
    #
    # No picker on purpose. This notebook answers "what does the mesh look like", and the
    # picker is only a slower way to get a different scene.
    bbox = [-71.34, 44.23, -71.25, 44.31]
    return (bbox,)


@app.cell
def _(mo):
    h3_res = mo.ui.dropdown(
        options={
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9.4 m hex": 12,
            "res 13 ·  ~3.6 m hex": 13,
            "res 14 ·  ~1.35 m hex (near native)": 14,
        },
        value="res 12 ·  ~9.4 m hex",
        label="H3 resolution",
    )
    h3_res
    return (h3_res,)


@app.cell
def _(Transformer, bbox, h3_res, np, tiles_albers, tiles_all):
    # Verbatim rule from xsql-s1m-h3.py: pick the overview geometrically so EVERY H3 cell is
    # guaranteed at least one pixel centre. p <= sqrt(2) * 0.5373 * sqrt(A), SAFETY 0.6.
    _fwd = Transformer.from_crs("EPSG:4326", "EPSG:6350", always_xy=True)
    _ax, _ay = _fwd.transform(
        [bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]]
    )
    aoi_albers = (min(_ax), min(_ay), max(_ax), max(_ay))
    _w, _s, _e, _n = aoi_albers

    _hit = (
        (tiles_albers[:, 0] < _e)
        & (tiles_albers[:, 2] > _w)
        & (tiles_albers[:, 1] < _n)
        & (tiles_albers[:, 3] > _s)
    )

    H3_CELL_M2 = {11: 2149.6, 12: 307.09, 13: 43.870, 14: 6.2673}
    SAFETY = 0.6
    _target_m = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])
    OVERVIEW_RES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    _fit = [r for r in OVERVIEW_RES if r <= _target_m]
    read_res_m = _fit[-1] if _fit else OVERVIEW_RES[0]

    candidates = [{**tiles_all[int(i)]} for i in np.flatnonzero(_hit)]

    print(
        f"AOI {tuple(bbox)} -> {len(candidates)} S1M tile(s) · "
        f"reading the {read_res_m:g} m overview for H3 res {h3_res.value} "
        f"(~{H3_CELL_M2[h3_res.value] / read_res_m**2:.0f} px per hex)"
    )
    return aoi_albers, candidates, read_res_m


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    S3_BASE,
    Window,
    aoi_albers,
    asyncio,
    candidates,
    fit_lonlat,
    h3_res,
    make_h3_context,
    make_lonlat_udf,
    np,
    pa,
    read_res_m,
    xr,
):
    # THE READ AND THE FOLD, condensed from xsql-s1m-h3.py but not changed: stream each
    # tile's AOI window off the chosen overview, hand the Albers grid to xarray-sql, and let
    # ONE query turn metres into degrees (per-tile fitted UDF) and fold pixels into H3.
    # A second statement adds `flow` = how far each cell sits below its k-ring.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value

    def _window(reader, aoi_proj):
        pw, ps, pe, pn = aoi_proj
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw = max(pw, bw)
        ce = min(pe, be)
        cn = min(pn, bn)
        cs = max(ps, bs)
        if ce <= cw or cn <= cs:
            return None
        col0 = max(0, int((cw - bw) / xres))
        col1 = min(reader.width, int(np.ceil((ce - bw) / xres)))
        row0 = max(0, int((bn - cn) / yres))
        row1 = min(reader.height, int(np.ceil((bn - cs) / yres)))
        if col1 <= col0 or row1 <= row0:
            return None
        return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)

    async def _read_tile(tile):
        g = await GeoTIFF.open(tile["key"], store=_store)
        cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
        fit_lvls = [r for r in cands if r.res[0] <= read_res_m]
        reader = fit_lvls[-1] if fit_lvls else cands[0]
        win = _window(reader, aoi_albers)
        if win is None:
            return None
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)
        if not np.isfinite(elev).any():
            return None

        left, bottom, right, top = r.bounds
        h, w = elev.shape
        y = top - (np.arange(h) + 0.5) * (top - bottom) / h
        x = left + (np.arange(w) + 0.5) * (right - left) / w
        ds = xr.Dataset({"elevation": (("y", "x"), elev)}, coords={"y": y, "x": x})
        fit, err_mm = fit_lonlat(g.crs, (left, bottom, right, top))
        return ds, fit, err_mm, float(reader.res[0])

    print(f"streaming {len(candidates)} S1M COG(s):")
    for _t in candidates:
        print(f"  {S3_BASE}{_t['key']}")

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in candidates]) if d]
    if _datasets:
        _px = sum(int(d[0]["elevation"].size) for d in _datasets)
        print(f"streamed {_px:,} pixels from {len(_datasets)}/{len(candidates)} tile(s)")

        ctx = make_h3_context()
        for _i, (_d, _fit, _, _) in enumerate(_datasets):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"y": 1024})
            ctx.register_udf(make_lonlat_udf(f"to_lonlat_{_i}", _fit))
        _union = " UNION ALL ".join(
            f"SELECT p.lat AS lat, p.lon AS lon, elevation FROM ("
            f"  SELECT to_lonlat_{_i}(x, y) AS p, elevation"
            f"  FROM dem_{_i} WHERE elevation = elevation"
            f")"
            for _i in range(len(_datasets))
        )
        _scene = ctx.sql(
            f"""
            SELECT hex, elevation - MIN(elevation) OVER () AS elevation
            FROM (
                SELECT h3_latlng_to_cell(lat, lon, CAST({_res} AS INT)) AS hex,
                       avg(elevation) AS elevation
                FROM ({_union})
                GROUP BY 1
            )
            """
        ).to_arrow_table()

        ctx.from_arrow(_scene, name="scene")
        h3_table = ctx.sql(
            """
            WITH ring AS (
                SELECT hex, elevation,
                       unnest(h3_grid_disk(hex, CAST(1 AS INT))) AS nb
                FROM scene
            )
            SELECT r.hex AS hex,
                   r.elevation AS elevation,
                   avg(n.elevation) - r.elevation AS flow
            FROM ring r
            JOIN scene n ON r.nb = n.hex
            GROUP BY r.hex, r.elevation
            """
        ).to_arrow_table()
        print(f"H3 res {_res}: {h3_table.num_rows:,} cells")
    else:
        h3_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
                "flow": pa.array([], pa.float64()),
            }
        )
        print("no S1M pixels for this AOI")
    return (h3_table,)




@app.cell
def _():
    # Palette registry: matplotlib + CARTOColors sequential ramps. All luminance-monotonic
    # and free of red/green opposition, so they survive a deuteranope simulation.
    from palettable.matplotlib import Viridis_20, Inferno_20, Magma_20, Plasma_20
    from palettable.cartocolors.sequential import (
        Emrld_7,
        Teal_7,
        BluYl_7,
        Mint_7,
        Sunset_7,
        PurpOr_7,
    )

    PALETTES = {
        "Viridis": Viridis_20,
        "Plasma": Plasma_20,
        "Inferno": Inferno_20,
        "Magma": Magma_20,
        "Emrld": Emrld_7,
        "Teal": Teal_7,
        "BluYl": BluYl_7,
        "Mint": Mint_7,
        "Sunset": Sunset_7,
        "PurpOr": PurpOr_7,
    }
    return (PALETTES,)


@app.cell
def _(PALETTES, mo):
    # The viz controls, ported from xsql-s1m-h3.py. None of them touch the stream or the SQL,
    # and none of them rebuild the Map: the update cell at the bottom swaps traits on the
    # running layer instead, so the view you flew to survives every adjustment.
    palette = mo.ui.dropdown(options=list(PALETTES), value="Emrld", label="Palette")
    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=3.0, debounce=True, label="Elevation scale"
    )
    flow_gain = mo.ui.number(
        start=0.0, stop=50.0, step=0.5, value=8.0, debounce=True, label="Flow offset"
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Opacity"
    )
    reverse_ramp = mo.ui.switch(value=True, label="Reverse ramp")
    wireframe = mo.ui.switch(value=False, label="Wireframe")
    mo.hstack(
        [palette, elevation_scale, flow_gain, fill_opacity, reverse_ramp, wireframe],
        justify="start",
        gap=2,
    )
    return elevation_scale, fill_opacity, flow_gain, palette, reverse_ramp, wireframe


@app.cell
def _(mo):
    # THE TWO COSTS, each on its own control, which is the whole point of this notebook.
    #
    # mesh_density is GEOMETRY. 256 is 66k vertices whether the fold produced 40k cells or
    # 4M. tex_size is STYLING, one image upload.
    #
    # smooth is the answer to "can we smooth the hexes". It is a box blur on the SHADING
    # VALUE, applied in texture space, so it softens the ramp without touching mesh height
    # and without going to a finer H3 resolution (which would be 7x the cells per level).
    # At 0 it is a no-op and you get hard hex edges.
    mesh_density = mo.ui.slider(
        start=32, stop=768, step=32, value=256,
        label="Mesh density (cells/side)", show_value=True,
    )
    tex_size = mo.ui.dropdown(
        options={"1024": 1024, "2048": 2048, "4096": 4096},
        value="2048",
        label="Texture size",
    )
    smooth = mo.ui.slider(
        start=0, stop=12, step=1, value=0, label="Smooth (texels)", show_value=True
    )
    mo.hstack([mesh_density, tex_size, smooth], justify="start", gap=2)
    return mesh_density, smooth, tex_size


@app.cell
def _(h3_table, np):
    # Cell id -> row in h3_table. Sort once, then every sample is a searchsorted. `ok` is
    # False for cells not in the scene at all (AOI corners, nodata holes), which the texture
    # turns transparent and the mesh turns into zero height.
    _hex = np.asarray(h3_table["hex"]).astype("uint64")
    _order = np.argsort(_hex)
    _sorted = _hex[_order]

    def cell_rows(cells):
        """Map an array of H3 cell ids to (row index, found mask)."""
        if _sorted.size == 0:
            return np.zeros(len(cells), "int64"), np.zeros(len(cells), bool)
        pos = np.clip(np.searchsorted(_sorted, cells), 0, _sorted.size - 1)
        ok = _sorted[pos] == cells
        return _order[pos], ok

    return (cell_rows,)


@app.cell
def _(flow_gain, h3_table, np):
    # THE SHADING VALUE, per cell: scene-relative elevation with flow added as an OFFSET so
    # drainage etches into the terrain shading. Gain 0 is pure elevation. Both renderers and
    # the contrast slider read this one array, so they cannot disagree.
    cell_shade = np.asarray(h3_table["elevation"]).astype("float64") + flow_gain.value * (
        np.asarray(h3_table["flow"]).astype("float64")
    )
    return (cell_shade,)


@app.cell
def _(cell_shade, mo, np):
    # Contrast window over the shading value. Its bounds ARE this scene's range, so it resets
    # per AOI and per flow offset. Own cell, depending on cell_shade alone: palette and
    # reverse must never reach it, or picking a palette would rebuild the slider and throw
    # away the window you dragged.
    if cell_shade.size:
        _lo, _hi = float(np.floor(cell_shade.min())), float(np.ceil(cell_shade.max()))
    else:
        _lo, _hi = 0.0, 1.0
    if _hi <= _lo:
        _hi = _lo + 1.0
    contrast = mo.ui.range_slider(
        start=_lo, stop=_hi, value=[_lo, _hi],
        step=max((_hi - _lo) / 200.0, 0.1),
        label="Shading contrast (m)",
        show_value=True, full_width=True, debounce=True,
    )
    return (contrast,)


@app.cell
def _(PALETTES, contrast, mo, palette, reverse_ramp):
    # The slider paints the ramp it controls: same palette, same DIRECTION as the scene, so
    # "reversed" is something you see rather than infer, and the strip doubles as the legend.
    _hex = PALETTES[palette.value].hex_colors
    if reverse_ramp.value:
        _hex = _hex[::-1]
    _strip = mo.Html(
        '<div style="height:14px;width:100%;border-radius:3px;'
        'border:1px solid rgba(128,128,128,0.35);'
        f'background:linear-gradient(to right,{",".join(_hex)});"></div>'
    )
    mo.vstack([_strip, contrast], gap=0)
    return


@app.cell
def _(bbox, cell_rows, coordinates_to_cells, h3_res, np, pa, tex_size):
    # TEXEL -> CELL, computed once and cached here on its own.
    #
    # This is the expensive part of the texture (a coordinates_to_cells call and a
    # searchsorted over every texel: 4.2M of each at 2048), and it depends only on the
    # geometry of the problem, never on the palette or the contrast window. Splitting it out
    # means changing a colour is a colormap over an existing index, not a re-binning.
    #
    # Row 0 of the image is the SOUTH edge, because the mesh's tex_coord v runs 0..1 south to
    # north and WebGL samples v=0 at the first row. If the scene comes out mirrored
    # vertically, this assumption is the thing to flip.
    _T = tex_size.value
    _LON, _LAT = np.meshgrid(
        np.linspace(bbox[0], bbox[2], _T), np.linspace(bbox[1], bbox[3], _T)
    )
    _cells = np.asarray(
        pa.array(
            coordinates_to_cells(_LAT.ravel(), _LON.ravel(), h3_res.value)
        ).to_numpy(zero_copy_only=False)
    ).astype("uint64")
    _rows, _ok = cell_rows(_cells)
    texel_rows = _rows.reshape(_T, _T)
    texel_ok = _ok.reshape(_T, _T)
    print(f"texel index: {_T}x{_T} · {texel_ok.mean() * 100:.1f}% landed on a cell")
    return texel_ok, texel_rows


@app.cell
def _(
    PALETTES,
    apply_continuous_cmap,
    cell_shade,
    contrast,
    np,
    palette,
    reverse_ramp,
    smooth,
    texel_ok,
    texel_rows,
):
    # THE TEXTURE. Paint the shading value into texture space, optionally blur it, then
    # colormap. Blurring the VALUE rather than the finished RGB is what makes "smooth" behave
    # like a coarser fold instead of like a soft-focus filter: the ramp still spans the same
    # contrast window, the hex plateaus just stop having hard walls.
    #
    # NaN-aware on purpose. Blurring straight through the holes would bleed zeros in from
    # outside the scene and draw a dark rind around every edge, so the value and the validity
    # mask are blurred separately and divided (a normalised convolution): edges stay put and
    # only real data contributes.
    _shade = np.where(texel_ok, cell_shade[texel_rows] if cell_shade.size else 0.0, 0.0)
    _mask = texel_ok.astype("float64")

    _k = int(smooth.value)
    if _k > 0:
        def _box(a, r):
            # Separable box blur by cumulative sums: O(n) per axis, no scipy.
            for axis in (0, 1):
                pad = np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)])
                c = np.cumsum(pad, axis=axis)
                lo = np.take(c, np.arange(0, a.shape[axis]), axis=axis)
                hi = np.take(c, np.arange(2 * r + 1, a.shape[axis] + 2 * r + 1), axis=axis)
                a = hi - lo
            return a

        _shade = _box(_shade, _k)
        _mask = _box(_mask, _k)
    _shade = np.divide(_shade, _mask, out=np.zeros_like(_shade), where=_mask > 0)

    _lo, _hi = float(contrast.value[0]), float(contrast.value[1])
    _norm = np.clip((_shade - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
    if reverse_ramp.value:
        _norm = 1.0 - _norm

    _rgb = np.asarray(
        apply_continuous_cmap(_norm.ravel(), PALETTES[palette.value], alpha=1.0)
    )
    # apply_continuous_cmap returns RGB for some palettes and RGBA for others.
    if _rgb.shape[1] == 3:
        _rgb = np.concatenate(
            [_rgb, np.full((_rgb.shape[0], 1), 255, dtype=_rgb.dtype)], axis=1
        )
    texture = _rgb.astype("uint8").reshape(*texel_ok.shape, 4)
    # Holes stay transparent. With smooth > 0 the blur widens the valid region by k texels,
    # so keep the ORIGINAL mask here or the scene grows a soft fringe past its own edge.
    texture[~texel_ok, 3] = 0
    print(f"texture: {texture.shape[1]}x{texture.shape[0]} ({texture.nbytes / 1e6:.1f} MB)")
    return (texture,)


@app.cell
def _(PALETTES, apply_continuous_cmap, cell_shade, contrast, np, palette, reverse_ramp):
    # The same colours per CELL, for the H3HexagonLayer comparison path. Unsmoothed: blurring
    # is a texture-space operation and there is nothing to blur into on a hexagon.
    if cell_shade.size:
        _lo, _hi = float(contrast.value[0]), float(contrast.value[1])
        _norm = np.clip((cell_shade - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
        if reverse_ramp.value:
            _norm = 1.0 - _norm
        _rgb = np.asarray(
            apply_continuous_cmap(_norm, PALETTES[palette.value], alpha=1.0)
        )
        if _rgb.shape[1] == 3:
            _rgb = np.concatenate(
                [_rgb, np.full((_rgb.shape[0], 1), 255, dtype=_rgb.dtype)], axis=1
            )
        cell_colors = _rgb.astype("uint8")
    else:
        cell_colors = np.zeros((0, 4), dtype="uint8")
    return (cell_colors,)


@app.cell
def _(mesh_density, np):
    # MESH TOPOLOGY. Vertex count is (n+1)^2 and triangle count is 2n^2, fixed by the slider
    # and independent of everything else in the notebook. Own cell so moving the elevation
    # scale re-uploads positions without rebuilding indices.
    #
    # tex_coords are the grid in normalised [0,1], which is also how the vertices are laid
    # out, so mesh and texture register with no extra bookkeeping.
    #
    # Vectorised: lonboard's own generate_mesh_grid() writes these indices with a Python
    # double loop, a quarter of a million iterations at density 512.
    _n = mesh_density.value
    _u = np.linspace(0.0, 1.0, _n + 1, dtype="float32")
    _UU, _VV = np.meshgrid(_u, _u)
    tex_coords = np.stack([_UU.ravel(), _VV.ravel()], axis=-1).astype("float32")

    _i = np.arange(_n)
    _r, _c = np.meshgrid(_i, _i, indexing="ij")
    _bl = (_r * (_n + 1) + _c).ravel()
    _br = _bl + 1
    _tl = _bl + (_n + 1)
    _tr = _tl + 1
    triangles = np.empty((_n * _n * 2, 3), dtype="uint32")
    triangles[0::2] = np.stack([_bl, _br, _tl], axis=-1)
    triangles[1::2] = np.stack([_br, _tr, _tl], axis=-1)
    return tex_coords, triangles


@app.cell
def _(
    bbox,
    cell_rows,
    coordinates_to_cells,
    elevation_scale,
    h3_res,
    h3_table,
    np,
    pa,
    tex_coords,
):
    # MESH POSITIONS. Vertex height sampled from the H3 field through the same lookup the
    # texture uses, so colour and relief cannot drift apart. Holes go to zero rather than
    # NaN: a NaN vertex takes its whole triangle fan with it.
    _lon = bbox[0] + tex_coords[:, 0] * (bbox[2] - bbox[0])
    _lat = bbox[1] + tex_coords[:, 1] * (bbox[3] - bbox[1])

    _cells = np.asarray(
        pa.array(coordinates_to_cells(_lat, _lon, h3_res.value)).to_numpy(
            zero_copy_only=False
        )
    ).astype("uint64")
    _rows, _ok = cell_rows(_cells)

    _elev = np.asarray(h3_table["elevation"]).astype("float64")
    _z = np.zeros(len(tex_coords), dtype="float64")
    if _elev.size:
        _z[_ok] = _elev[_rows[_ok]]
    _z *= elevation_scale.value

    # float32 is what the trait casts to anyway: ~1 m of positional quantisation at lat 44,
    # invisible against a 9 km AOI.
    positions = np.stack([_lon, _lat, _z], axis=-1).astype("float32")
    return (positions,)


@app.cell
def _(mo):
    renderer = mo.ui.radio(
        options=["Surface mesh", "H3 hexagons"],
        value="Surface mesh",
        label="Renderer",
        inline=True,
    )
    renderer
    return (renderer,)


@app.cell
def _(h3_table, mesh_density, mo, triangles):
    # The comparison in numbers, before you look at either picture.
    #
    # high_precision=True puts H3HexagonLayer on the PolygonLayer path: each cell is a real
    # extruded prism, a hexagonal top plus six side quads, so roughly 30 vertices of
    # tessellated geometry per cell. The mesh is one draw call whose size you chose.
    _cells = h3_table.num_rows
    _hex_verts = _cells * 30
    _mesh_verts = (mesh_density.value + 1) ** 2
    mo.md(
        f"""
        | | geometry | scales with |
        |---|---|---|
        | **H3 hexagons** | {_cells:,} prisms · ~{_hex_verts / 1e6:.1f}M vertices | the cell count |
        | **Surface mesh** | {len(triangles):,} triangles · {_mesh_verts:,} vertices | the slider, and nothing else |

        Ratio at this scene: **{_hex_verts / max(_mesh_verts, 1):.0f}x** less geometry.
        """
    )
    return


@app.cell
def _(
    CartoBasemap,
    FullscreenControl,
    H3HexagonLayer,
    Map,
    MaplibreBasemap,
    NavigationControl,
    ScaleControl,
    SurfaceLayer,
    Table,
    bbox,
    h3_table,
    np,
    renderer,
):
    # The layer and the Map are built ONCE per renderer choice, from PLACEHOLDER geometry.
    # This cell deliberately references no viz control, so marimo never re-runs it for a
    # palette, contrast, smooth, scale or density change and the view state survives. The
    # update cell below pushes the real arrays in.
    #
    # SurfaceLayer never populates _bbox (it has no geoarrow table to derive one from), so
    # the view state has to be explicit or the Map opens on null island.
    if renderer.value == "Surface mesh":
        scene_layer = SurfaceLayer(
            positions=np.zeros((4, 3), dtype="float32"),
            triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype="uint32"),
            tex_coords=np.zeros((4, 2), dtype="float32"),
            texture=np.zeros((1, 1, 4), dtype="uint8"),
        )
    else:
        _t = Table.from_arrow(h3_table)
        scene_layer = H3HexagonLayer(
            table=_t,
            get_hexagon=_t["hex"],
            get_fill_color=[136, 136, 136],  # placeholder; the update cell paints it
            get_elevation=_t["elevation"],
            high_precision=True,
            extruded=True,
            stroked=False,
        )

    scene = Map(
        layers=[scene_layer],
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 12.5,
            "pitch": 60,
            "bearing": -25,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter),
        controls=[
            FullscreenControl(position="top-right"),
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
        parameters={"depthTest": True, "blend": True},
    )
    scene
    return (scene_layer,)


@app.cell
def _(
    SurfaceLayer,
    cell_colors,
    elevation_scale,
    fill_opacity,
    positions,
    scene_layer,
    tex_coords,
    texture,
    triangles,
    wireframe,
):
    # The only thing the controls do: swap traits on the running layer. No Map rebuild, no
    # re-stream, no re-fold, no re-bin.
    #
    # BATCHED, because positions, tex_coords and triangles have to agree about vertex
    # indices. Moving the mesh density slider changes all three, and if they reach the widget
    # one at a time the frontend briefly holds indices that point past the end of the buffer.
    with scene_layer.hold_trait_notifications():
        if isinstance(scene_layer, SurfaceLayer):
            scene_layer.positions = positions
            scene_layer.tex_coords = tex_coords
            scene_layer.triangles = triangles
            scene_layer.texture = texture
            scene_layer.wireframe = wireframe.value
            scene_layer.opacity = fill_opacity.value
        else:
            scene_layer.get_fill_color = cell_colors
            scene_layer.elevation_scale = elevation_scale.value
            scene_layer.opacity = fill_opacity.value
    return


if __name__ == "__main__":
    app.run()
