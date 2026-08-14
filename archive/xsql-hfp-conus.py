# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "numpy==2.5.1",
#     "matplotlib==3.11.1",
# ]
# ///
"""Human footprint 2021 over CONUS, one fold at res 7, one static lonboard map.

This is xsql-hfp-divisions.py with everything interactive cut away: no camera, no
divisions, no widgets, no cache. One bounding box (the lower 48), one read of the L2
overview (400 m, ~32 px per res 7 cell), one DataFusion fold to H3, one H3HexagonLayer.
The reader, the Mollweide formulas, the fold SQL and the ramp are ported from that
notebook by copy, per the repo's shared-by-copy rule: a fix there should be carried
here by hand.

BOX IS THE ONLY KNOB, AND IT SCALES HARD. The default lower-48 box is a ~120M pixel
window: ~2 GB of RAM through the fold, 1.85M cells, under 10 s on a fast connection.
The commented North-America box below it is ~760M pixels: it works (measured: 4.88M
cells, 21.5 s) but peaks around 15-20 GB of RAM and hands lonboard a table 2.5x larger
than anything else in this repo, so treat it as a big-machine poster run, not a default.

Data: Vizzuality / Impact Observatory HFP-100 v1.2, CC-BY 4.0, on source.coop.
Run:  uv run marimo edit xsql-hfp-conus.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import math

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import numpy as np
    import pyarrow as pa
    import xarray as xr
    from arro3.core import Table as ArroTable
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext
    from lonboard import Map, H3HexagonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap

    return (
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        S3Store,
        Window,
        XarrayContext,
        asyncio,
        coordinates_to_cells,
        math,
        matplotlib,
        mo,
        np,
        pa,
        udf,
        xr,
    )


@app.cell
def _(matplotlib, np):
    # ------------------------------------------------------------------ the raster
    SOURCE_BUCKET = "us-west-2.opendata.source.coop"
    YEAR = 2021
    COG = f"vizzuality/hfp-100/hfp_{YEAR}_100m_v1-2_cog.tif"
    TILE = 512
    FETCH_AT_ONCE = 32

    # Res 7 (5.16 km2 cells) reads L2 (400 m, 0.16 km2 pixels): ~32 px per cell, the same
    # row of the LEVEL_FOR_RES ladder the interactive notebook uses. The pyramid AVERAGES
    # (verified there: the mean survives an 8x downsample while the max collapses), which
    # is what makes reading an overview equivalent to reading pixels.
    RES = 7
    LEVEL = 2

    # The lower 48 by bounding box, with a hair of margin: Cape Alava to West Quoddy
    # Head, Key West to the 49th parallel. Canadian and Mexican fringes inside the box
    # fold too; clipping to the border would need the divisions machinery this notebook
    # exists to not have.
    BOX = (-124.8, 24.4, -66.9, 49.5)
    # North America at large, Aleutians to Greenland, for a poster run. See the module
    # docstring before using it: ~760M pixels, 15-20 GB of RAM, 4.88M cells.
    # BOX = (-169.9, 1.9, -26.3, 72.5)

    # ------------------------------------------------------------------ the ramp
    # log1p over 0-40, zero inside the ramp, same stretch as the interactive notebook:
    # the land distribution is bottom-loaded (p50 1.0, p99 23.4) and zero is untouched
    # ground, the bottom of a continuum, not a dropped case. Inferno rather than that
    # notebook's cividis: still monotonic in luminance (the order survives a deuteranope
    # simulation), and its near-black bottom lets wild ground recede into the basemap.
    HI = 40.0
    NAN_RGB = (38, 40, 44)  # NaN only; the fold's `v = v` filter keeps it off screen
    _RAMP = matplotlib.colormaps["inferno"]

    def ramp(v):
        """footprint 0-50 -> uint8 RGB, log1p-stretched; NaN takes the dark swatch."""
        v = np.asarray(v, dtype="float64")
        live = np.isfinite(v)
        t = np.zeros(v.shape)
        if live.any():
            t[live] = np.log1p(np.clip(v[live], 0.0, HI)) / np.log1p(HI)
        out = (_RAMP(t)[..., :3] * 255).astype(np.uint8)
        out[~live] = NAN_RGB
        return out

    return BOX, COG, FETCH_AT_ONCE, LEVEL, RES, SOURCE_BUCKET, TILE, ramp


@app.cell
async def _(
    BOX,
    COG,
    FETCH_AT_ONCE,
    GeoTIFF,
    LEVEL,
    RES,
    S3Store,
    SOURCE_BUCKET,
    TILE,
    Window,
    XarrayContext,
    asyncio,
    coordinates_to_cells,
    math,
    np,
    pa,
    udf,
    xr,
):
    # THE FOLD, ONCE. Ported from xsql-hfp-divisions.py and straightened out: no LRU, no
    # budget, no cache dict, because there is exactly one window and it is read exactly
    # once. What survives the port unchanged is everything that was learned the hard way:
    # the sparse-tile table, np.ma.filled, the Mollweide pair, the `v = v` NaN test.
    import time as _time

    _t0 = _time.perf_counter()
    _store = S3Store(SOURCE_BUCKET, region="us-west-2", skip_signature=True)
    _g = await GeoTIFF.open(COG, store=_store)
    _lv = [_g, *_g.overviews][LEVEL]
    _L, _B, _R, _T = _g.bounds
    _H, _W = _lv.shape
    _px, _py = (_R - _L) / _W, (_T - _B) / _H

    # WHICH TILES EXIST. 65.7% of tiles are unstored ocean (offset 0, length 0) and
    # async-geotiff does not check: a read touching one raises "Invalid range requested,
    # start: 0 end: 0". Consulting tile_byte_counts first turns the crash into a skip.
    _nty, _ntx = -(-_H // TILE), -(-_W // TILE)
    _present = np.asarray(_lv.ifd.tile_byte_counts).reshape(_nty, _ntx) > 0

    # Degree box -> pixel window, through the FORWARD projection. Mollweide meridians
    # curve, so the box's widest x is wherever it comes closest to the equator, not at a
    # corner: project a sampled perimeter and take the envelope.
    _SQ2R = math.sqrt(2.0) * 6378137.0

    def _moll_fwd(lon, lat):
        """Degrees -> Mollweide metres. Newton on 2t + sin 2t = pi sin(lat)."""
        phi = np.radians(np.asarray(lat, dtype=np.float64))
        lam = np.radians(np.asarray(lon, dtype=np.float64))
        th = phi.copy()
        for _ in range(12):  # converges in ~5 everywhere below 89 degrees
            th -= (2 * th + np.sin(2 * th) - np.pi * np.sin(phi)) / np.maximum(
                2 + 2 * np.cos(2 * th), 1e-9
            )
        return (2 * _SQ2R / np.pi) * lam * np.cos(th), _SQ2R * np.sin(th)

    _w, _s, _e, _n = BOX
    _tt = np.linspace(0.0, 1.0, 33)
    _bx, _by = _moll_fwd(
        np.concatenate([_w + (_e - _w) * _tt, np.full(33, _e), _e + (_w - _e) * _tt, np.full(33, _w)]),
        np.concatenate([np.full(33, _s), _s + (_n - _s) * _tt, np.full(33, _n), _n + (_s - _n) * _tt]),
    )
    _col0 = max(0, int((max(_bx.min(), _L) - _L) / _px))
    _col1 = min(_W, int(math.ceil((min(_bx.max(), _R) - _L) / _px)))
    _row0 = max(0, int((_T - min(_by.max(), _T)) / _py))
    _row1 = min(_H, int(math.ceil((_T - max(_by.min(), _B)) / _py)))
    _wpx, _hpx = _col1 - _col0, _row1 - _row0

    _sem = asyncio.Semaphore(FETCH_AT_ONCE)

    async def _tile(ty, tx):
        r0, c0 = ty * TILE, tx * TILE
        h, w = min(TILE, _H - r0), min(TILE, _W - c0)
        if not _present[ty, tx]:
            return ty, tx, None
        async with _sem:
            m = (
                await _lv.read(window=Window(col_off=c0, row_off=r0, width=w, height=h))
            ).as_masked()[0]
        # filled(), NOT np.asarray(): asarray silently drops the mask and a nodata coast
        # would average in at score 65.5. Stored uint16 is the index x1000.
        arr = np.ma.filled(m.astype(np.float32), np.nan)
        arr[arr == 65535.0] = np.nan
        return ty, tx, arr / 1000.0

    _want = [
        (ty, tx)
        for ty in range(_row0 // TILE, (_row1 - 1) // TILE + 1)
        for tx in range(_col0 // TILE, (_col1 - 1) // TILE + 1)
    ]
    arr = np.full((_hpx, _wpx), np.nan, dtype=np.float32)
    fetched = skipped = 0
    for _ty, _tx, _a in await asyncio.gather(*(_tile(*k) for k in _want)):
        if _a is None:
            skipped += 1
            continue
        fetched += 1
        _sr, _sc = _ty * TILE, _tx * TILE
        _r0, _c0 = max(_row0, _sr), max(_col0, _sc)
        _r1 = min(_row1, _sr + _a.shape[0])
        _c1 = min(_col1, _sc + _a.shape[1])
        arr[_r0 - _row0 : _r1 - _row0, _c0 - _col0 : _c1 - _col0] = _a[
            _r0 - _sr : _r1 - _sr, _c0 - _sc : _c1 - _sc
        ]
    _t_read = _time.perf_counter() - _t0

    # Pixel centres -> lat/lng, through the INVERSE projection, closed form. The
    # parametric angle depends only on the row, so lat is one arcsin per ROW. Pixels
    # outside the Mollweide ellipse invert to |lon| > 180 and are masked. The window's
    # width is set at its most equatorial edge, so a tall box's polar rows genuinely
    # leave the ellipse; the CONUS default never does.
    _ym = _T - (_row0 + np.arange(_hpx, dtype=np.float64) + 0.5) * _py
    _xm = _L + (_col0 + np.arange(_wpx, dtype=np.float64) + 0.5) * _px
    _th = np.arcsin(np.clip(_ym / _SQ2R, -1.0, 1.0))
    _lat = np.degrees(np.arcsin(np.clip((2 * _th + np.sin(2 * _th)) / np.pi, -1.0, 1.0)))
    _lon = np.degrees(
        (np.pi * _xm[None, :]) / (2 * _SQ2R * np.maximum(np.cos(_th), 1e-12)[:, None])
    )
    arr[np.abs(_lon) > 180.0] = np.nan

    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(
                coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())
            ),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )
    ctx.from_dataset(
        "df",
        xr.Dataset(
            {
                "v": (("y", "x"), arr),
                "lat": (
                    ("y", "x"),
                    np.ascontiguousarray(np.broadcast_to(_lat[:, None], arr.shape)),
                ),
                "lon": (("y", "x"), _lon),
            },
            coords={"y": np.arange(_hpx), "x": np.arange(_wpx)},
        ),
        chunks={"y": 512},
    )

    # `v = v` IS THE NaN TEST: ocean arrives as NaN, the one value that fails equality
    # with itself. Zero cells are KEPT, untouched land being half of what the map says.
    # px_total is the weight a coastal cell would carry into a join; here it feeds the
    # tooltip only.
    folded = ctx.sql(f"""
        SELECT h3_latlng_to_cell(lat, lon, CAST({RES} AS INT)) AS hex,
               avg(CAST(v AS DOUBLE)) AS hfp,
               count(*)               AS px_total
        FROM df
        WHERE v = v
        GROUP BY 1
    """).to_arrow_table()
    _t_all = _time.perf_counter() - _t0

    fold_stats = (
        f"window {_wpx}x{_hpx} px at L{LEVEL} · {fetched} tiles fetched, "
        f"{skipped} sparse · {folded.num_rows:,} res {RES} cells · "
        f"read {_t_read:.1f}s, total {_t_all:.1f}s"
    )
    return fold_stats, folded


@app.cell
def _(
    ArroTable,
    BitmapTileLayer,
    CartoBasemap,
    H3HexagonLayer,
    Map,
    MaplibreBasemap,
    folded,
    np,
    pa,
    ramp,
):
    # combine_chunks because DataFusion returns many chunks while the numpy-derived
    # colour column is one, and lonboard rejects a table whose columns disagree about
    # chunking.
    _tbl = folded.combine_chunks()
    _hfp = np.asarray(_tbl["hfp"])
    _layer_tbl = ArroTable.from_arrow(
        pa.table(
            {
                "hex": _tbl["hex"],
                "color": pa.FixedSizeListArray.from_arrays(pa.array(ramp(_hfp).ravel()), 3),
                "footprint": pa.array(np.round(_hfp, 2)),
                "pixels": _tbl["px_total"],
            }
        )
    )

    cells = H3HexagonLayer(
        table=_layer_tbl,
        get_hexagon=_layer_tbl["hex"],
        get_fill_color=_layer_tbl["color"],
        extruded=False,
        stroked=False,
        high_precision=True,
        coverage=1,
        opacity=0.7,
        pickable=True,
    )

    # Place labels OVER the cells: the basemap paints under every deck layer, so names on
    # it would sit beneath an opaque hexagon. @2x with tile_size 512 or retina type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
    )

    deck = Map(
        [
        cells, 
        # labels
        ],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state={"longitude": -96.0, "latitude": 38.5, "zoom": 4.0},
        height=700,
        show_tooltip=True,
    )
    deck
    return


@app.cell
def _(fold_stats, mo):
    mo.md(f"""
    `{fold_stats}`
    """)
    return


if __name__ == "__main__":
    app.run()
