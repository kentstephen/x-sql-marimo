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
#     "numpy",
#     "pyproj>=3.7",
# ]
# ///
"""Annual NLCD land cover for all of CONUS, in H3, finer as you zoom in.

Read the raster once, register it with xarray-sql, and every H3 resolution after that is
just another SELECT against the same table. The read is the expensive part and it happens
once per year; the hexes are cheap.

The fold is a mode, not a mean, because land cover is categorical: each cell takes its
most frequent class plus the purity of that mode. Colour is the class, height is purity.

Data: Kyle Barron's mirror of USGS Annual NLCD on source.coop, public and unsigned.

Run:  uv run marimo edit xsql-nlcd-zoom.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pyarrow as pa
    import xarray as xr
    from arro3.core import Table as ArroTable
    from pyproj import Transformer
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from lonboard import Map, H3HexagonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap

    return (
        ArroTable,
        CartoBasemap,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        S3Store,
        Transformer,
        XarrayContext,
        coordinates_to_cells,
        mo,
        np,
        pa,
        udf,
        xr,
    )


@app.cell
def _():
    PREFIX = "kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic"
    NODATA = 250

    # The 480 m overview: fine enough that even res 8 cells all catch pixels (going finer
    # than the source supports is what leaves holes in the map), coarse enough to read
    # the whole country in well under a second.
    LEVEL = 4

    def res_for_zoom(z):
        return 6 if z < 5.0 else (7 if z < 7.0 else 8)

    # 16 NLCD classes in 7 groups on a teal-to-brown axis, water blue, developed carried
    # by luminance. NLCD's own palette is green forest against red developed, which is the
    # one pairing that carries nothing for a deuteranope, so it is never drawn.
    GROUPS = {
        11: ("Water", (8, 48, 107)),
        12: ("Ice", (158, 202, 225)),
        21: ("Developed, open", (215, 215, 215)),
        22: ("Developed, low", (160, 160, 160)),
        23: ("Developed, medium", (99, 99, 99)),
        24: ("Developed, high", (37, 37, 37)),
        31: ("Barren", (222, 217, 204)),
        41: ("Deciduous forest", (1, 102, 94)),
        42: ("Evergreen forest", (0, 60, 48)),
        43: ("Mixed forest", (53, 151, 143)),
        52: ("Shrub", (128, 205, 193)),
        71: ("Herbaceous", (199, 234, 229)),
        81: ("Pasture", (223, 194, 125)),
        82: ("Cropland", (191, 129, 45)),
        90: ("Woody wetland", (67, 147, 195)),
        95: ("Herbaceous wetland", (146, 197, 222)),
    }
    return GROUPS, LEVEL, NODATA, PREFIX, res_for_zoom


@app.cell
def _(mo):
    year = mo.ui.slider(1985, 2024, value=2024, step=1, label="Year", full_width=True)
    return (year,)


@app.cell
def _(mo):
    # The camera writes one integer: the resolution its zoom implies. Panning cannot
    # change it and neither can a zoom nudge inside a band, so the graph stays quiet.
    get_res, set_res = mo.state(6)
    return get_res, set_res


@app.cell
async def _(
    GeoTIFF,
    LEVEL,
    NODATA,
    PREFIX,
    S3Store,
    Transformer,
    XarrayContext,
    coordinates_to_cells,
    np,
    pa,
    udf,
    xr,
    year,
):
    # THE ONLY READ. Once per year, then the array lives here and every resolution is a
    # query against it. Changing zoom never touches object storage.
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _g = await GeoTIFF.open(
        f"{PREFIX}/Annual_NLCD_LndCov_{year.value}_CU_C1V1.tif", store=_store
    )
    _rd = [_g, *_g.overviews][LEVEL]
    _arr = np.asarray((await _rd.read()).as_masked()[0])
    _l, _b, _r, _t = _rd.bounds
    _h, _w = _arr.shape

    # Albers over CONUS is smooth, so exact pyproj on a 64x64 control grid plus bilinear
    # interpolation lands within ~100 m of a per-pixel transform, for 4096 pyproj calls
    # instead of 35 million.
    _inv = Transformer.from_crs(_g.crs, "EPSG:4326", always_xy=True)
    _c = 64
    _gy, _gx = np.linspace(0, _h - 1, _c), np.linspace(0, _w - 1, _c)
    _X, _Y = np.meshgrid(
        _l + (_gx + 0.5) * (_r - _l) / _w, _t - (_gy + 0.5) * (_t - _b) / _h
    )
    _glo, _gla = (a.reshape(_c, _c) for a in _inv.transform(_X.ravel(), _Y.ravel()))

    def _bilinear(grid, rr, cc):
        fy = np.interp(rr, _gy, np.arange(_c))
        fx = np.interp(cc, _gx, np.arange(_c))
        y0 = np.clip(fy.astype(np.int32), 0, _c - 2)
        x0 = np.clip(fx.astype(np.int32), 0, _c - 2)
        dy, dx = fy - y0, fx - x0
        return (
            grid[y0, x0] * (1 - dy) * (1 - dx)
            + grid[y0 + 1, x0] * dy * (1 - dx)
            + grid[y0, x0 + 1] * (1 - dy) * dx
            + grid[y0 + 1, x0 + 1] * dy * dx
        )

    def _to_deg(grid):
        # y, x are Albers metres, the raster's own dims. This is what lets the SQL ask for
        # degrees without anything being flattened in Python first.
        def f(yv, xv):
            rr = (_t - yv.to_numpy()) / ((_t - _b) / _h) - 0.5
            cc = (xv.to_numpy() - _l) / ((_r - _l) / _w) - 0.5
            return pa.array(_bilinear(grid, rr, cc))

        return f

    # The array goes straight in. from_dataset makes y and x columns and cls a column,
    # streamed by chunk, so nothing here builds a 35M-row table in Python.
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
    ctx.register_udf(
        udf(_to_deg(_gla), [pa.float64(), pa.float64()], pa.float64(), "stable", name="to_lat")
    )
    ctx.register_udf(
        udf(_to_deg(_glo), [pa.float64(), pa.float64()], pa.float64(), "stable", name="to_lon")
    )
    ctx.from_dataset(
        "lc",
        xr.Dataset(
            {"cls": (("y", "x"), _arr)},
            coords={
                "y": _t - (np.arange(_h) + 0.5) * (_t - _b) / _h,
                "x": _l + (np.arange(_w) + 0.5) * (_r - _l) / _w,
            },
        ),
        chunks={"y": 512},
    )

    def fold(res):
        # Mode per cell. The `cls ASC` tie-break is not decoration: without it a cell whose
        # top two classes have equal counts picks a different winner run to run.
        return ctx.sql(f"""
            WITH counts AS (
                SELECT h3_latlng_to_cell(to_lat(y, x), to_lon(y, x), CAST({res} AS INT))
                           AS hex,
                       cls, count(*) AS n
                FROM lc WHERE cls != {NODATA}
                GROUP BY 1, 2
            )
            SELECT hex,
                   first_value(cls ORDER BY n DESC, cls ASC) AS mode_cls,
                   sum(n) AS px_total,
                   CAST(max(n) AS DOUBLE) / sum(n) AS purity
            FROM counts GROUP BY hex
        """).to_arrow_table()

    source = f"{_rd.res[0]:.0f} m · {(_arr != NODATA).sum() / 1e6:.0f}M px"
    return fold, source


@app.cell
def _(ArroTable, GROUPS, np, pa):
    _lut = np.full((256, 3), 120, dtype=np.uint8)
    for _c, (_lbl, _rgb) in GROUPS.items():
        _lut[_c] = _rgb
    _names = np.array(
        [GROUPS.get(i, ("", None))[0] for i in range(256)], dtype=object
    )

    def to_layer_table(tbl):
        # combine_chunks because DataFusion returns many chunks and the numpy-derived
        # columns are one; lonboard rejects a table whose columns disagree about chunking.
        # ArroTable because the layer's `table` trait coerces in __init__ but its
        # validate() is a strict isinstance check, so assignment afterwards needs the
        # real type.
        tbl = tbl.combine_chunks()
        cls = np.asarray(tbl["mode_cls"])
        pur = np.asarray(tbl["purity"])
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(_lut[cls].ravel()), 3
                    ),
                    "height": pa.array((pur * 4000.0).astype(np.float32)),
                    "class": pa.array(list(_names[cls])),
                    "purity": tbl["purity"],
                    "pixels": tbl["px_total"],
                }
            )
        )

    return (to_layer_table,)


@app.cell
def _(fold, get_res, to_layer_table):
    # Cached per resolution: zoom back out to a level you have already seen and it is a
    # dict lookup, not a query.
    _cache = {}

    def _get(res):
        if res not in _cache:
            _cache[res] = to_layer_table(fold(res))
        return _cache[res]

    shown = _get(get_res())
    return (shown,)


@app.cell
def _(
    CartoBasemap,
    H3HexagonLayer,
    Map,
    MaplibreBasemap,
    get_res,
    res_for_zoom,
    set_res,
    shown,
):
    # Built once, and NOT downstream of get_res. Rebuilding the Map (or reassigning
    # deck.layers) on a camera event resets the view, fires the observer again, and leaks
    # a widget model per event until the browser gives up. One Map, one layer, traits
    # swapped in place below.
    layer = H3HexagonLayer(
        table=shown,
        get_hexagon=shown["hex"],
        get_fill_color=shown["color"],
        get_elevation=shown["height"],
        extruded=True,
        high_precision=True,
        opacity=0.9,
        pickable=True,
    )
    deck = Map(
        [layer],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state={"longitude": -98.5, "latitude": 39.5, "zoom": 3.8, "pitch": 35},
        height=620,
    )

    def _on_camera(change):
        r = res_for_zoom(change["new"].zoom)
        if r != get_res():
            set_res(r)

    deck.observe(_on_camera, names="view_state")
    return deck, layer


@app.cell
def _(layer, shown):
    with layer.hold_trait_notifications():
        layer.table = shown
        layer.get_hexagon = shown["hex"]
        layer.get_fill_color = shown["color"]
        layer.get_elevation = shown["height"]
    return


@app.cell
def _(GROUPS, deck, get_res, mo, shown, source, year):
    _seen, _sw = set(), []
    for _c, (_lbl, (_r, _g, _b)) in GROUPS.items():
        if _lbl in _seen:
            continue
        _seen.add(_lbl)
        _sw.append(
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
            f'margin-right:.8rem;white-space:nowrap">'
            f'<span style="width:.8rem;height:.8rem;border-radius:2px;'
            f'background:rgb({_r},{_g},{_b});outline:1px solid #8888"></span>{_lbl}</span>'
        )

    # Controls sit in the same stack as the map, right above it.
    mo.vstack(
        [
            year,
            mo.md(
                f"`res {get_res()} · {shown.num_rows:,} cells · {source}` "
                "&nbsp; zoom in for finer cells"
            ),
            deck,
            mo.md(
                "<div style='display:flex;flex-wrap:wrap;font-size:.8rem;line-height:1.7'>"
                + "".join(_sw)
                + "</div>\n\nHeight is **purity**: how much of a cell is its own class. "
                "Short cells are mixed."
            ),
        ],
        gap=0.4,
    )
    return


if __name__ == "__main__":
    app.run()
