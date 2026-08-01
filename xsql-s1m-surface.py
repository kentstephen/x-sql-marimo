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
    from palettable.matplotlib import Viridis_20, Inferno_20, Magma_20, Plasma_20
    from palettable.cartocolors.sequential import BluYl_7, Teal_7, Mint_7, PurpOr_7

    # All luminance-monotonic, no red/green opposition.
    PALETTES = {
        "BluYl": BluYl_7,
        "Viridis": Viridis_20,
        "Plasma": Plasma_20,
        "Inferno": Inferno_20,
        "Magma": Magma_20,
        "Teal": Teal_7,
        "Mint": Mint_7,
        "PurpOr": PurpOr_7,
    }
    return (PALETTES,)


@app.cell
def _(PALETTES, mo):
    palette = mo.ui.dropdown(
        options=list(PALETTES), value="BluYl", label="Palette"
    )
    reverse_ramp = mo.ui.checkbox(value=True, label="Reverse ramp")
    flow_gain = mo.ui.number(
        start=0.0, stop=20.0, step=0.5, value=8.0, label="Flow gain"
    )
    mo.hstack([palette, reverse_ramp, flow_gain], justify="start", gap=2)
    return flow_gain, palette, reverse_ramp


@app.cell
def _(
    PALETTES,
    apply_continuous_cmap,
    flow_gain,
    h3_table,
    np,
    palette,
    reverse_ramp,
):
    # COLOUR, identical in spirit to the real notebook: scene-relative elevation with flow
    # added as an OFFSET so drainage etches into the terrain shading. This array is what
    # `get_fill_color` receives in the H3 path and what gets PAINTED INTO THE TEXTURE in the
    # surface path, so both renderers below are showing the exact same numbers.
    _cmap = PALETTES[palette.value]
    _v = np.asarray(h3_table["elevation"]).astype("float64") + flow_gain.value * np.asarray(
        h3_table["flow"]
    ).astype("float64")
    if _v.size:
        _lo, _hi = float(_v.min()), float(_v.max())
        _norm = np.clip((_v - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
        if reverse_ramp.value:
            _norm = 1.0 - _norm
        _rgb = np.asarray(apply_continuous_cmap(_norm, _cmap, alpha=1.0))
        # apply_continuous_cmap returns RGB for some palettes and RGBA for others. The
        # texture path indexes into a fixed 4-wide array, so normalise here rather than
        # branching downstream.
        if _rgb.shape[1] == 3:
            _rgb = np.concatenate(
                [_rgb, np.full((_rgb.shape[0], 1), 255, dtype=_rgb.dtype)], axis=1
            )
        cell_colors = _rgb.astype("uint8")
    else:
        cell_colors = np.zeros((0, 4), dtype="uint8")
    return (cell_colors,)


@app.cell
def _(h3_table, np):
    # The one lookup both the mesh and the texture go through: an H3 cell id -> its row in
    # h3_table. Sort once, then every sample is a searchsorted. `ok` is False for cells that
    # are not in the scene at all (AOI corners, nodata holes), which the texture turns into
    # transparent texels and the mesh turns into zero height.
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
def _(mo):
    # The two knobs this notebook is actually about.
    #
    # mesh_density is the GEOMETRY cost and it is the whole point: it does not move when the
    # cell count moves. 256 means 257^2 = 66k vertices and 131k triangles no matter whether
    # the H3 fold produced 40k cells or 4M.
    #
    # tex_size is the STYLING cost, one image upload. 2048 is 16 MB of RGBA and shows hex
    # edges cleanly at res 12 over this AOI.
    mesh_density = mo.ui.slider(
        start=32, stop=768, step=32, value=256, label="Mesh density (cells/side)",
        show_value=True,
    )
    tex_size = mo.ui.dropdown(
        options={"512": 512, "1024": 1024, "2048": 2048, "4096": 4096},
        value="2048",
        label="Texture size",
    )
    elevation_scale = mo.ui.number(
        start=0.5, stop=10.0, step=0.5, value=3.0, label="Elevation scale"
    )
    mo.hstack([mesh_density, tex_size, elevation_scale], justify="start", gap=2)
    return elevation_scale, mesh_density, tex_size


@app.cell
def _(
    bbox,
    cell_colors,
    cell_rows,
    coordinates_to_cells,
    h3_res,
    np,
    pa,
    tex_size,
):
    # THE TEXTURE. A regular lon/lat lattice over the AOI, every texel pushed through
    # coordinates_to_cells to find the H3 cell that contains it, then painted with that
    # cell's colour. Nearest-neighbour by construction, so hexagon edges come out crisp
    # rather than blurred: you are seeing the actual cell boundaries, drawn for free.
    #
    # Row 0 of the image is the SOUTH edge, because the mesh's tex_coord v runs 0..1 south
    # to north and WebGL samples v=0 at the first row. If the scene comes out mirrored
    # vertically, that assumption is the thing to flip.
    _T = tex_size.value
    _lon = np.linspace(bbox[0], bbox[2], _T)
    _lat = np.linspace(bbox[1], bbox[3], _T)
    _LON, _LAT = np.meshgrid(_lon, _lat)

    _cells = np.asarray(
        pa.array(
            coordinates_to_cells(_LAT.ravel(), _LON.ravel(), h3_res.value)
        ).to_numpy(zero_copy_only=False)
    ).astype("uint64")
    _rows, _ok = cell_rows(_cells)

    texture = np.zeros((_T, _T, 4), dtype="uint8")
    if cell_colors.shape[0]:
        _flat = texture.reshape(-1, 4)
        _flat[_ok] = cell_colors[_rows[_ok]]
        _flat[~_ok, 3] = 0  # outside the scene -> transparent
    print(
        f"texture: {_T}x{_T} ({texture.nbytes / 1e6:.1f} MB) · "
        f"{_ok.sum() / _ok.size * 100:.1f}% of texels landed on a cell"
    )
    return (texture,)


@app.cell
def _(
    bbox,
    cell_rows,
    coordinates_to_cells,
    elevation_scale,
    h3_res,
    h3_table,
    mesh_density,
    np,
    pa,
):
    # THE MESH. A regular grid over the same AOI, vertex z sampled from the H3 field through
    # the same lookup the texture uses. Vertex count is (n+1)^2 and triangle count is 2n^2,
    # both fixed by the slider and INDEPENDENT of how many cells the fold produced. That
    # independence is the entire argument for this notebook.
    #
    # tex_coords are the grid in normalised [0,1], which is also exactly how the vertices
    # were laid out, so mesh and texture register with no extra bookkeeping.
    _n = mesh_density.value
    _u = np.linspace(0.0, 1.0, _n + 1, dtype="float32")
    _UU, _VV = np.meshgrid(_u, _u)
    tex_coords = np.stack([_UU.ravel(), _VV.ravel()], axis=-1).astype("float32")

    # Triangle indices, vectorised. generate_mesh_grid() in lonboard does this with a Python
    # double loop, which is a quarter of a million iterations at density 512.
    _i = np.arange(_n)
    _r, _c = np.meshgrid(_i, _i, indexing="ij")
    _bl = (_r * (_n + 1) + _c).ravel()
    _br = _bl + 1
    _tl = _bl + (_n + 1)
    _tr = _tl + 1
    triangles = np.empty((_n * _n * 2, 3), dtype="uint32")
    triangles[0::2] = np.stack([_bl, _br, _tl], axis=-1)
    triangles[1::2] = np.stack([_br, _tr, _tl], axis=-1)

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

    # float32 is what the trait casts to anyway. At lat 44 that is ~1 m of positional
    # quantisation on the vertices, invisible against a 9 km AOI.
    positions = np.stack([_lon, _lat, _z], axis=-1).astype("float32")

    print(
        f"mesh: {len(positions):,} vertices · {len(triangles):,} triangles · "
        f"{_ok.sum() / _ok.size * 100:.1f}% of vertices landed on a cell"
    )
    return positions, tex_coords, triangles


@app.cell
def _(mo):
    renderer = mo.ui.radio(
        options=["Surface mesh", "H3 hexagons"],
        value="Surface mesh",
        label="Renderer",
        inline=True,
    )
    wireframe = mo.ui.checkbox(value=False, label="Wireframe (surface only)")
    mo.hstack([renderer, wireframe], justify="start", gap=2)
    return renderer, wireframe


@app.cell
def _(h3_table, mesh_density, mo, triangles):
    # The comparison, in numbers, before you look at either picture.
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
    cell_colors,
    elevation_scale,
    h3_table,
    positions,
    renderer,
    tex_coords,
    texture,
    triangles,
    wireframe,
):
    # Both renderers, same data, same colours. Rebuilt on every control change, which loses
    # view state: fine for a demo whose job is comparison, wrong for the real notebook
    # (which builds the layer once and swaps traits live).
    if renderer.value == "Surface mesh":
        _layer = SurfaceLayer(
            positions=positions,
            triangles=triangles,
            tex_coords=tex_coords,
            texture=texture,
            wireframe=wireframe.value,
        )
    else:
        _t = Table.from_arrow(h3_table)
        _layer = H3HexagonLayer(
            table=_t,
            get_hexagon=_t["hex"],
            get_fill_color=cell_colors,
            get_elevation=_t["elevation"],
            high_precision=True,
            extruded=True,
            stroked=False,
            elevation_scale=elevation_scale.value,
            opacity=1.0,
        )

    # SurfaceLayer never populates _bbox (it has no geoarrow table to derive one from), so
    # the view state has to be explicit or the Map opens on null island.
    scene = Map(
        layers=[_layer],
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
    return


if __name__ == "__main__":
    app.run()
