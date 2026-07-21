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
#     "geopy==2.5.0",
#     "aiohttp>=3.10",
#     "arro3-core",
#     "numpy",
# ]
# ///
"""Free-fly the USA: draw a box, stream the 10m DEM, aggregate to H3, detrend to a REM.

Same streaming/xarray-sql spine as xsql-dem-h3.py, with one added step: a Relative
Elevation Model. After folding pixels into H3 cells, SQL fits a trend surface (a plane,
by least squares) to the cell elevations and subtracts it, so each cell carries its
height ABOVE the local trend rather than absolute elevation. On a valley or floodplain
the plane approximates the river's longitudinal slope, so the REM reads as height above
water and terraces / paleochannels / meander scars pop out. Colored with CARTOColors
Emrld (a luminance-monotonic green ramp, deuteranope-safe).

Run:  uv run marimo edit xsql-dem-rem.py --sandbox
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import urllib.request
    import xml.etree.ElementTree as ET

    import numpy as np
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    from arro3.core import Table
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import Map, H3HexagonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard.colormap import apply_continuous_cmap
    from lonboard.controls import (
        FullscreenControl,
        GeocoderControl,
        NavigationControl,
        ScaleControl,
    )
    from palettable.cartocolors.sequential import Emrld_7

    return (
        AioHTTPAdapter,
        CartoBasemap,
        ET,
        Emrld_7,
        FullscreenControl,
        GeoTIFF,
        GeocoderControl,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        NavigationControl,
        Photon,
        S3Store,
        ScaleControl,
        Table,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        coordinates_to_cells,
        mo,
        np,
        pa,
        pathlib,
        udf,
        urllib,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # DEM &rarr; H3 &rarr; REM, all in SQL

    **Hold Ctrl/Cmd and drag** on the picker to draw an AOI anywhere in the USA. The
    USGS 3DEP **10m** seamless DEM streams from object storage (`obstore`) into **xarray**
    Datasets, and **xarray-sql** folds the pixels into **H3** cells. Then one more SQL step
    fits a **trend surface** (a plane, by least squares) to the cell elevations and
    subtracts it: the **Relative Elevation Model**. Each hex is colored by height above
    that local trend (CARTOColors **Emrld**), so on a river valley the water surface
    flattens and terraces / old channels stand up. Extrusion still uses true elevation, so
    the landform is real; only the color is detrended.

    REM shows best on floodplains: the default is **Mount Washington** and the
    Presidentials, which reads as height above the ravine floors. Draw over a river reach
    (Connecticut, Androscoggin) to see the classic detrended-valley look.
    """)
    return


@app.cell
def _(ET, pathlib, urllib):
    # Catalog = the VRT, not a STAC API. USGS publishes a nationwide VRT listing every
    # 1-degree seamless COG on prd-tnm with its exact placement, so AOI -> hrefs is a
    # local bbox intersection. Parse each <ComplexSource>: SourceFilename (minus the
    # /vsicurl/ prefix) + DstRect + GeoTransform -> a degree bbox.
    _vrt = pathlib.Path(".cache/USGS_Seamless_DEM_13.vrt")
    _vrt_url = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
        "USGS_Seamless_DEM_13.vrt"
    )
    if not _vrt.exists():
        _vrt.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading 1/3 arc-second seamless VRT index (~830 KB)...")
        urllib.request.urlretrieve(_vrt_url, _vrt)

    _root = ET.parse(_vrt).getroot()
    _gt = [float(v) for v in _root.find("GeoTransform").text.split(",")]
    dem_tiles = []
    for _src in _root.iter("ComplexSource"):
        # The COG path on the bucket, relative to the bucket root.
        _key = _src.find("SourceFilename").text.removeprefix("/vsicurl/")
        _key = _key.split("amazonaws.com/", 1)[-1]
        _rect = _src.find("DstRect")
        _w = _gt[0] + float(_rect.get("xOff")) * _gt[1]
        _n = _gt[3] + float(_rect.get("yOff")) * _gt[5]
        _e = _w + float(_rect.get("xSize")) * _gt[1]
        _s = _n + float(_rect.get("ySize")) * _gt[5]
        dem_tiles.append({"key": _key, "bbox": [_w, _s, _e, _n]})
    print(f"10m seamless tile index: {len(dem_tiles)} COGs")
    return (dem_tiles,)


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # xarray-sql is the point of this notebook: query the DEM raster AS a table with SQL,
    # no manual flattening. XarrayContext IS a DataFusion session with one extra trick,
    # from_dataset(): an xarray Dataset's dimension coords (lat, lon) become columns and
    # its data variables (elevation) become columns, so `SELECT lat, lon, elevation FROM
    # dem` unravels the grid for you. We register the H3 UDF on it (inherited DataFusion
    # method); it returns a UBIGINT (uint64) cell id, exactly what lonboard's H3HexagonLayer
    # consumes with high_precision=True. Bigint in, bigint on the GPU, no string round-trip.
    #
    # Color is deliberately NOT here: this cell + the aggregation below are the expensive
    # ETL that builds the hexagons, and it must NOT re-run when you only change the ramp.
    # Color lives in its own cheap cell downstream, so tuning it never re-streams or
    # re-aggregates. A factory (fresh context per stream) keeps per-tile table names from
    # colliding across re-runs.
    def _latlng_to_cell(lat, lng, res):
        return pa.array(
            coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py())
        )

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
        return ctx

    print("xarray-sql context factory ready; UDF: h3_latlng_to_cell(lat, lon, res) -> UBIGINT")
    return (make_h3_context,)


@app.cell
def _(mo):
    # Reactive AOI. Default: a tight box on Mount Washington and the Presidential Range,
    # White Mountains NH. Small on purpose so the first render is quick; draw a bigger box
    # (or search elsewhere) to fly the rest of the country.
    get_bbox, set_bbox = mo.state((-71.43, 44.165, -71.15, 44.385))
    return get_bbox, set_bbox


@app.cell
def _(mo):
    # Own cell: marimo re-runs any cell that references a UI element, and the picker map
    # must never reference this (a re-run would rebuild the map and drop the drawn AOI).
    # H3 average edge length by resolution: 8~461m, 9~174m, 10~66m, 11~25m, 12~9m. The 10m
    # DEM floors useful detail near res 12; large AOIs at fine res make a lot of cells.
    h3_res = mo.ui.dropdown(
        options={
            "res 8  ·  ~461 m hex": 8,
            "res 9  ·  ~174 m hex": 9,
            "res 10 ·  ~66 m hex": 10,
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9 m hex (near native)": 12,
        },
        value="res 11 ·  ~25 m hex",
        label="H3 resolution",
    )
    h3_res
    return (h3_res,)


@app.cell
def _(
    AioHTTPAdapter,
    CartoBasemap,
    FullscreenControl,
    GeocoderControl,
    Map,
    MaplibreBasemap,
    NavigationControl,
    Photon,
    ScaleControl,
    set_bbox,
):
    # 2D picker. Draw a box (Ctrl/Cmd + drag) -> selected_bounds -> set_bbox. Built once,
    # never references a reactive UI element, so pan/zoom/AOI survive every downstream run.
    #
    # Photon (komoot): free, keyless, OSM-backed geocoder. Search a place to fly there,
    # then draw the box. geopy must run in async mode (AioHTTPAdapter) for lonboard.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        layers=[],
        view_state={"longitude": -71.29, "latitude": 44.275, "zoom": 10, "pitch": 0},
        basemap=MaplibreBasemap(style=CartoBasemap.Positron),
        controls=[
            _geocoder,
            FullscreenControl(position="top-right"),
            NavigationControl(),
            ScaleControl(),
        ],
    )
    picker.observe(
        lambda c: set_bbox(c["new"]) if c["new"] is not None else None,
        names="selected_bounds",
    )
    picker
    return


@app.cell
def _(dem_tiles, get_bbox):
    # Resolve WHICH COGs cover the AOI: a pure local bbox intersection over the VRT index.
    def cog_keys(bbox):
        w, s, e, n = bbox
        return [
            t
            for t in dem_tiles
            if t["bbox"][0] < e and t["bbox"][2] > w
            and t["bbox"][1] < n and t["bbox"][3] > s
        ]

    bbox = list(get_bbox())
    tiles = cog_keys(bbox)
    print(f"AOI {tuple(round(x, 4) for x in bbox)} -> {len(tiles)} COG(s)")
    return bbox, tiles


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    Window,
    asyncio,
    bbox,
    h3_res,
    make_h3_context,
    np,
    pa,
    tiles,
    xr,
):
    # Stream the covering COGs and aggregate to H3 with xarray-sql. For each tile: pick an
    # overview whose ground sampling roughly matches the H3 cell (so we neither oversample
    # nor starve cells), read ONLY the AOI window as an xarray Dataset (elevation over
    # lat/lon coords, nodata -> NaN), and register it as a SQL table. Then ONE SQL statement
    # unravels every tile's grid, unions them, and folds pixels into UBIGINT H3 cells via
    # the UDF. No manual flatten-to-pyarrow: the raster IS the table.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value
    _w, _s, _e, _n = bbox

    # Target ground sampling ~ half the H3 edge, in degrees (1 deg lat ~ 111320 m).
    _edge_m = {8: 461.0, 9: 174.0, 10: 66.0, 11: 25.0, 12: 9.0}[_res]
    _target_deg = (_edge_m / 2.0) / 111320.0
    _PIXEL_BUDGET = 3_000_000  # per-tile window cap; step coarser if exceeded

    def _window(reader, tw, ts, te, tn):
        # AOI clipped to this reader's extent, in pixel coords.
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw = max(_w, bw, tw); ce = min(_e, be, te)
        cn = min(_n, bn, tn); cs = max(_s, bs, ts)
        if ce <= cw or cn <= cs:
            return None
        col0 = int((cw - bw) / xres)
        col1 = int(np.ceil((ce - bw) / xres))
        row0 = int((bn - cn) / yres)
        row1 = int(np.ceil((bn - cs) / yres))
        col0 = max(0, col0); row0 = max(0, row0)
        col1 = min(reader.width, col1); row1 = min(reader.height, row1)
        if col1 <= col0 or row1 <= row0:
            return None
        return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)

    async def _read_tile(tile):
        g = await GeoTIFF.open(tile["key"], store=_store)
        tw, ts, te, tn = tile["bbox"]
        cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
        fit = [r for r in cands if r.res[0] <= _target_deg]
        # Walk from the matched overview toward coarser until the window fits the budget.
        start = cands.index(fit[-1]) if fit else 0
        for reader in cands[start:] if fit else cands:
            win = _window(reader, tw, ts, te, tn)
            if win is None:
                return None
            if win.width * win.height <= _PIXEL_BUDGET or reader is cands[-1]:
                break
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)  # nodata -> NaN
        if not np.isfinite(elev).any():
            return None
        rw, rs, re_, rn = r.bounds
        h, w = elev.shape
        # Pixel-centre coords. lat descends (north-up raster), lon ascends.
        lat = rn - (np.arange(h) + 0.5) * (rn - rs) / h
        lon = rw + (np.arange(w) + 0.5) * (re_ - rw) / w
        return xr.Dataset(
            {"elevation": (("lat", "lon"), elev)},
            coords={"lat": lat, "lon": lon},
        )

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in tiles]) if d]
    if _datasets:
        _px = sum(int(d["elevation"].size) for d in _datasets)
        print(f"streamed {_px:,} pixels as {len(_datasets)} xarray Dataset(s)")

        # Register each tile's grid as a SQL table on the xarray-sql context, then let ONE
        # statement do the work: unravel every grid to (lat, lon, elevation) rows, drop
        # NaN nodata (elevation = elevation is false for NaN), union the tiles, and group
        # by H3 cell. This is the demonstration: SQL straight over xarray, UDF and all.
        ctx = make_h3_context()
        for _i, _d in enumerate(_datasets):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"lat": 1024})
        _union = " UNION ALL ".join(
            f"SELECT lat, lon, elevation FROM dem_{_i} WHERE elevation = elevation"
            for _i in range(len(_datasets))
        )

        # Pass 1: fold pixels into cells, carrying each cell's centroid so the trend
        # surface can be fit against position in the next pass.
        _cells = ctx.sql(
            f"""
            SELECT h3_latlng_to_cell(lat, lon, CAST({_res} AS INT)) AS hex,
                   avg(lat)       AS clat,
                   avg(lon)       AS clon,
                   avg(elevation) AS elevation,
                   count(*)       AS n
            FROM ({_union})
            GROUP BY 1
            """
        ).to_arrow_table()
        ctx.register_record_batches("cells", [_cells.to_batches()])

        # Pass 2: least-squares plane fit  z ~ a + b*(lon-mlon) + c*(lat-mlat). SQL builds
        # the normal-equation sums (centred on the mean position for conditioning); numpy
        # solves the tiny 3x3. This is the REM's trend surface: on a floodplain the plane
        # tracks the river's downstream slope, so subtracting it leaves height above water.
        _m = ctx.sql("SELECT avg(clon) mlon, avg(clat) mlat FROM cells").to_arrow_table()
        _mlon = float(_m["mlon"][0].as_py())
        _mlat = float(_m["mlat"][0].as_py())
        _s = ctx.sql(
            f"""
            SELECT count(*) n,
                   sum(x)   sx,  sum(y)   sy,  sum(z)  sz,
                   sum(x*x) sxx, sum(y*y) syy, sum(x*y) sxy,
                   sum(x*z) sxz, sum(y*z) syz
            FROM (SELECT clon - ({_mlon}) AS x, clat - ({_mlat}) AS y,
                         elevation AS z FROM cells)
            """
        ).to_arrow_table().to_pydict()
        _A = np.array(
            [
                [_s["n"][0],   _s["sx"][0],  _s["sy"][0]],
                [_s["sx"][0],  _s["sxx"][0], _s["sxy"][0]],
                [_s["sy"][0],  _s["sxy"][0], _s["syy"][0]],
            ],
            dtype="float64",
        )
        _rhs = np.array([_s["sz"][0], _s["sxz"][0], _s["syz"][0]], dtype="float64")
        _a, _b, _c = np.linalg.lstsq(_A, _rhs, rcond=None)[0]  # robust if degenerate

        # Pass 3: REM = elevation - plane, per cell, entirely in SQL. This is the end of the
        # ETL: h3_table carries hex + true elevation + rem, and NOTHING about color. Color
        # is a separate downstream cell, so changing the ramp never re-runs this stream/fold.
        h3_table = ctx.sql(
            f"""
            SELECT hex, elevation,
                   elevation - (({_a}) + ({_b}) * (clon - ({_mlon}))
                                       + ({_c}) * (clat - ({_mlat}))) AS rem
            FROM cells
            """
        ).to_arrow_table()
        _slope = (_b * _b + _c * _c) ** 0.5 / 111.32  # m per degree -> m per km
        print(f"H3 res {_res}: {h3_table.num_rows:,} cells; trend slope {_slope:.2f} m/km")
    else:
        h3_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
                "rem": pa.array([], pa.float64()),
            }
        )
        print("no DEM pixels for this AOI")
    return (h3_table,)


@app.cell
def _(Emrld_7, apply_continuous_cmap, h3_table, np):
    # COLOR CELL: separate from the ETL on purpose. Colors each hex by its actual
    # ELEVATION. This depends only on h3_table["elevation"], so it re-runs when the hexagons
    # change (new AOI / resolution) but the expensive stream + H3 fold above never re-runs
    # because of anything color-related. This is where the ramp lives, so a future palette /
    # domain control would re-run only this cheap cell.
    #
    # Emrld is a green ramp monotonic in lightness (deuteranope-safe). Domain is percentile-
    # clamped (2nd..98th) so a lone peak doesn't wash out the ramp. Both ramp directions are
    # precomputed so the Reverse toggle downstream is a live trait swap, not a recompute.
    _elev = np.asarray(h3_table["elevation"]).astype("float64")
    if _elev.size:
        _lo, _hi = (float(v) for v in np.percentile(_elev, [2, 98]))
        _norm = np.clip((_elev - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
        colors_fwd = apply_continuous_cmap(_norm, Emrld_7, alpha=1.0)
        colors_rev = apply_continuous_cmap(1.0 - _norm, Emrld_7, alpha=1.0)
    else:
        colors_fwd = np.zeros((0, 4), dtype="uint8")
        colors_rev = np.zeros((0, 4), dtype="uint8")
    return colors_fwd, colors_rev


@app.cell
def _(
    CartoBasemap,
    FullscreenControl,
    H3HexagonLayer,
    Map,
    MaplibreBasemap,
    NavigationControl,
    ScaleControl,
    Table,
    bbox,
    colors_rev,
    h3_table,
):
    # The output scene: extruded H3 hexagons. Geometry (hex) and height (true elevation) come
    # straight from h3_table as arrow columns; the initial fill is the reversed ramp from the
    # color cell. Extrusion uses TRUE elevation, so the 3D landform is real and only the
    # color is detrended.
    #
    # This cell references NEITHER elevation_scale NOR fill_opacity NOR the reverse toggle.
    # marimo re-runs any cell that reads a UI element, and a re-run here would rebuild the
    # Map. So the layer is built ONCE and the tiny cell below only nudges live traits,
    # including swapping colors_fwd/rev on get_fill_color, which lonboard syncs to the
    # running widget: no Map rebuild, no re-stream, no re-fold.
    scene_table = Table.from_arrow(h3_table)
    h3_layer = H3HexagonLayer(
        table=scene_table,
        get_hexagon=scene_table["hex"],
        get_fill_color=colors_rev,  # reversed default; toggle flips it live below
        get_elevation=scene_table["elevation"],
        high_precision=True,
        extruded=True,
        stroked=False,
        elevation_scale=3.0,  # initial; the number input below nudges this live
        opacity=0.9,          # initial; the number input below nudges this live
    )

    scene = Map(
        layers=[h3_layer],
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 10,
            "pitch": 55,
            "bearing": -20,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter),
        controls=[
            FullscreenControl(position="top-right"),
            NavigationControl(),
            ScaleControl(),
        ],
        parameters={"depthTest": True, "blend": True},
    )
    print(f"scene: {h3_table.num_rows:,} hexes, color from the separate color cell")
    scene
    return (h3_layer,)


@app.cell
def _(mo):
    # Right below the map: float inputs with up/down steppers at 0.1, plus a Reverse toggle
    # for the ramp. mo.ui.number renders native increment arrows. Changing any of these
    # re-runs ONLY the trait-update cell below, never the map cell, so the scene updates in
    # place (lonboard's whole point). Reverse in particular does NOT touch the stream or the
    # SQL: it just swaps a precomputed color array onto the live layer.
    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=3.0, label="Elevation scale"
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=0.9, label="Opacity"
    )
    reverse_ramp = mo.ui.switch(value=True, label="Reverse ramp")
    mo.hstack([elevation_scale, fill_opacity, reverse_ramp], justify="start", gap=2)
    return elevation_scale, fill_opacity, reverse_ramp


@app.cell
def _(
    colors_fwd,
    colors_rev,
    elevation_scale,
    fill_opacity,
    h3_layer,
    reverse_ramp,
):
    # The only thing the controls do: nudge live traits on the running layer. No Map
    # rebuild, no re-stream, no re-fold, no re-color. This is the cell that reads the UI
    # elements, so it is the only one marimo re-runs when they change. Reverse just swaps
    # which precomputed color array feeds get_fill_color, live.
    h3_layer.elevation_scale = elevation_scale.value
    h3_layer.opacity = fill_opacity.value
    h3_layer.get_fill_color = colors_rev if reverse_ramp.value else colors_fwd
    return


if __name__ == "__main__":
    app.run()
