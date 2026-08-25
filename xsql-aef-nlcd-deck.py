# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "geoarrow-rust-core",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
# ]
# ///
"""NLCD backed or not by AlphaEarth, anywhere in CONUS, at any zoom: the deck.gl build.

xsql-aef-nlcd-conus.py with the map as its OWN deck.gl widget instead of lonboard.
The reason is one accessor: deck's H3HexagonLayer takes a single `coverage` for the
whole layer, so the agreement paint (each hexagon scaled by how well AlphaEarth
backs NLCD's word there) had to be drawn as polygons built in the kernel (rings,
earcut in a worker, ~7 vertices x 16 bytes per cell). kepler.gl solved this years
ago with a ColumnLayer subclass that adds ONE instanced attribute,
`instanceCoverage`, and multiplies the column radius by it. That subclass is ~20
lines of this widget's JS, so every paint is the stock H3HexagonLayer: uint64
cell ids, an rgba array and a float32 coverage array cross the bridge, deck
tessellates on the GPU. Layers have real ids (no marimo `undefined` collision),
the NLCD raster is a plain TileLayer the kernel serves over anywidget custom
messages, and a click is deck's pick with an h3-js fallback (the click's lon/lat
to the frame's res; deck's GPU pick returns nothing inside marimo here, as it
did in the HRRR counties film).

The camera-driven fold is unchanged from the lonboard build:
every time the map settles, the ground under it is folded to H3 at the resolution
the zoom deserves, NLCD from its own overview pyramid and AlphaEarth from whichever
of its two source.coop copies can serve that rung:

  res 11    (zoomed in)   tge-labs/aef-mosaic   the 10 m Zarr, native, one window
  res 5-10  (zoomed out)  tge-labs/aef          the per-tile COGs' OVERVIEWS (mean
                                                embeddings at 40..2560 m), many files

Both folds are the h3 UDF in DataFusion; NLCD's majority class and AlphaEarth's
mean vector meet on the cell. Per view: class prototypes, the agreement (sigmoid
over the own-vs-runner-up cosine margin), spherical k-means clusters. The strip
under the map has the four paints as toggles, none required (NLCD raster; agreement:
alpha + coverage; NLCD and AlphaEarth clusters: regular hexagons at coverage
0.8), the pickable legend, a click that lights the
hexagon and tells its story. Prototypes and clusters are PER VIEW: they say what
is typical of a class HERE, and cluster colors are arbitrary per fold.

Measured from home (2026-08-24): a COG opens in 0.8 s, 162 open concurrently in
1.8 s and read their 2560 m overviews in 0.7 s; the ~2,000 files that cover CONUS
are a cold ~30 s at the coarsest rung, then cached (open handles + folded frames).
The mosaic rung is a native 10 m read: ~1-2 s at zoom 12, 10-20 s at zoom 10.

Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind." (CC-BY 4.0.)

Run: uv run marimo edit xsql-aef-nlcd-deck.py   (or --sandbox)
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import json
    import math
    import os
    import tempfile
    import time
    import urllib.parse
    import urllib.request

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xarray as xr
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from pyproj import Transformer

    import io
    from PIL import Image

    return (
        GeoTIFF,
        Image,
        ObjectStore,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        io,
        json,
        math,
        mo,
        np,
        os,
        pa,
        pq,
        tempfile,
        time,
        traitlets,
        udf,
        urllib,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/x-sql-marimo/blob/main/xsql-aef-nlcd-conus.py)

    # NLCD, backed or not by AlphaEarth, across CONUS

    Fly anywhere in the lower 48. When the map settles, the ground in view is folded
    to H3 at the resolution the zoom deserves: **NLCD** (majority class per hexagon,
    from its own overview pyramid) and the **AlphaEarth Foundations embedding** (the
    mean 64-vector per hexagon, from the 10 m mosaic when zoomed in and from the
    per-tile COGs' overviews when zoomed out). Per view, each NLCD class gets a
    *prototype* (the mean of its cells' vectors: what that word looks like to the
    satellites here) and each hexagon an **agreement**: how clearly it sits closer to
    its own class than to the nearest other one.

    - **agreement** paint: NLCD's colors; faint and shrunken where the embedding does
      not back the word.
    - **NLCD** paint: regular hexagons, flat colors.
    - **AlphaEarth** paint: the embedding on its own, k-means clusters, the legend
      saying what each cluster is made of in NLCD terms.

    Click a hexagon for its story; click a legend chip to isolate a class or cluster.
    Prototypes and clusters are recomputed per view (local, honest, colors shift).

    | leg | data | engine |
    |---|---|---|
    | land cover | Annual NLCD, 30 m + pyramid, EPSG:5070 (`kylebarron/usgs-landcover` mirror, COG) | obstore + async-geotiff tiles, DataFusion fold (h3 UDF) |
    | embeddings, zoomed in | `tge-labs/aef-mosaic` (Zarr v3, 10 m, 64 x int8, no pyramid) | obstore + zarr, DataFusion fold (h3 UDF, 64 `avg()`) |
    | embeddings, zoomed out | `tge-labs/aef` COGs' overviews (80..2560 m, per UTM tile) | obstore + async-geotiff, pyproj per tile, one DataFusion fold |
    | score | join on cell, prototypes, sigmoid margin, k-means | numpy; DuckDB for the tables |
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    YEAR_NLCD = 2024
    YEAR_AEF = 2024  # 2017-2025 (NLCD's mirror ends at 2024)

    # The zoom -> H3 ladder (the nlcd-zoom notebook's): BASE_RES at ZOOM0, one step
    # finer every PER_RES zoom units, clamped, then coarsened until the view's
    # expected cell count fits CELL_BUDGET (polygons, not H3HexagonLayer, so the
    # budget is vertices: 150k hexagons is ~1M vertices).
    # BASE_RES 6 (was 7): one step coarser at every zoom, Stephen's call after
    # flying it (the coarse hexagons read better and cost a quarter of the bytes).
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 11
    CELL_BUDGET = 150_000
    # Which NLCD overview each res reads (30 m native, ten doublings).
    NLCD_LEVEL_FOR_RES = {5: 5, 6: 4, 7: 4, 8: 3, 9: 2, 10: 1, 11: 0}
    # Which AlphaEarth source and level each res reads. Mosaic from MOSAIC_MIN_RES
    # up (native 10 m); below that the COG overview index (0 = 20 m, 1 = 40 m,
    # 2 = 80 m, 3 = 160 m, 4 = 320 m, 5 = 640 m, 6 = 1280 m, 7 = 2560 m), picked
    # for ~15-50 overview pixels per cell.
    # res 10 stays on the COGs (40 m): its padded box is ~2,800 km2, ~1.8 GB raw
    # from the mosaic (a minute from home); res 11 (~360 km2, ~230 MB) is the
    # first rung the mosaic serves in ~10 s.
    MOSAIC_MIN_RES = 11
    AEF_LEVEL_FOR_RES = {5: 7, 6: 6, 7: 4, 8: 3, 9: 2, 10: 1}
    AEF_MAX_FILES = 2500  # more files than this and the view gets NLCD only

    # The fold box is the flat camera footprint, padded, from a GUESSED canvas size
    # (the HFP ruler is the port that measures it; not here yet).
    VIEW_W, VIEW_H = 1400, 720
    PAD = 1.3
    SETTLE = 0.35  # seconds the camera must rest before a fold
    # Below this zoom the map is NLCD as a picture (RasterLayer.from_geotiff, the
    # COG's own tiles and colormap, served by the kernel); from it up, the
    # agreement hexagons fold live for the small box in view (Stephen: "show
    # something cheap like the raster, then when you zoom in the agreement hexes").
    HEX_ZOOM = 9.0
    # maplibre layer id deck draws BEFORE (under) in the interleaved basemap: the
    # first label layer of Carto's Positron style (lonboard's viz() uses it too).
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256  # px per NLCD tile the kernel renders for the TileLayer
    HOME = {"longitude": -96.0, "latitude": 38.5, "zoom": 4.0}

    TAU = 0.02
    MIN_CLASS_CELLS = 30
    K_CLUSTERS = 10
    CLUSTER_HEX = ["#0072B2", "#E69F00", "#56B4E9", "#F0E442", "#CC79A7",
                   "#009E73", "#D55E00", "#999999", "#7B4EA3", "#6B3F1D"]
    ALPHA_MIN, ALPHA_MAX = 30, 235
    COV_MIN = 0.30
    # NLCD H3 / AlphaEarth clusters H3: regular hexagons at full coverage (0.8 was
    # tried 2026-08-24 and put back, Stephen's call), a little below the agreement
    # paint's top alpha
    COV_FLAT = 1.00
    ALPHA_FLAT = 190
    # "color by agreement" (the strip's toggle on the agreement paint): the
    # hexagons take a perceptual ramp on the agreement value instead of NLCD's
    # color; cool = disagreement, warm = agreement (Stephen's default), and the
    # highlight-disagreement checkbox reverses it so warm = disagreement. viridis
    # because it has NO RED anywhere: its warm end is yellow, so neither
    # direction of the flip lands on the weak leg (a blue-white-red cool/warm
    # would). cividis is the alternative (same axis, flatter). 32 stops each,
    # matplotlib's tables, interpolated to 256 in the frame cell; no matplotlib
    # import. Coverage scaling stays; alpha is flat (a ramp's dark end fading to
    # nothing would read as no data).
    AGREE_CMAP = "viridis"
    RAMPS = {
        "viridis": "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725",
        "cividis": "00224e00285b002e6a0533711c396f293f6e33446d3c4a6c45506c4d556c555b6d5c616e6467706b6d72727274787877807f78888578908b78979177a09875a89e73b0a571b9ab6dc2b369cbb965d3c05fdcc859e6d051efd748f8df3cfee838",
    }
    ALPHA_RAMP = 225
    DIM_ALPHA = 22
    # Boundaries around clusters of low-agreement cells: the set of cells with
    # agreement below the strip's threshold (EDGE_THR seeds it) is dissolved in
    # DuckDB (h3_cells_to_multi_polygon_wkb, H3's own outer-boundary walk, then
    # ST_Dump into blobs); blobs under EDGE_MIN_CELLS cells (by area against the
    # res's average cell) are speckle and dropped. Drawn as one PathLayer.
    EDGE_THR = 0.5
    EDGE_MIN_CELLS = 7
    # each ring is painted the NLCD color of the blob's majority class (Stephen:
    # "the same color as the NLCD hexes"), at this alpha and width
    EDGE_ALPHA = 235
    EDGE_WIDTH = 2  # px

    NLCD_PREFIX = "kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic"
    NLCD_NODATA = 250
    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-nlcd")

    CLASSES = {
        11: ("Open water", (70, 107, 159)),
        12: ("Perennial ice/snow", (209, 222, 248)),
        21: ("Developed, open space", (222, 197, 197)),
        22: ("Developed, low", (217, 146, 130)),
        23: ("Developed, medium", (235, 0, 0)),
        24: ("Developed, high", (171, 0, 0)),
        31: ("Barren", (179, 172, 159)),
        41: ("Deciduous forest", (104, 171, 95)),
        42: ("Evergreen forest", (28, 95, 44)),
        43: ("Mixed forest", (181, 197, 143)),
        52: ("Shrub/scrub", (204, 184, 121)),
        71: ("Herbaceous", (223, 223, 194)),
        81: ("Pasture/hay", (220, 217, 57)),
        82: ("Cultivated crops", (171, 108, 40)),
        90: ("Woody wetlands", (184, 217, 235)),
        95: ("Emergent wetlands", (108, 159, 184)),
    }
    return (
        AEF_INDEX_URL,
        AEF_LEVEL_FOR_RES,
        AEF_MAX_FILES,
        AEF_NODATA,
        AGREE_CMAP,
        ALPHA_RAMP,
        RAMPS,
        AEF_PREFIX,
        AEF_RES,
        AEF_X0,
        AEF_Y0,
        ALPHA_FLAT,
        ALPHA_MAX,
        ALPHA_MIN,
        BASE_RES,
        CACHE_DIR,
        CELL_BUDGET,
        CLASSES,
        CLUSTER_HEX,
        COV_FLAT,
        COV_MIN,
        DIM_ALPHA,
        EDGE_ALPHA,
        EDGE_MIN_CELLS,
        EDGE_THR,
        EDGE_WIDTH,
        HEX_ZOOM,
        HOME,
        K_CLUSTERS,
        LABELS_SLOT,
        MAX_RES,
        MIN_CLASS_CELLS,
        MIN_RES,
        MOSAIC_MIN_RES,
        NLCD_LEVEL_FOR_RES,
        RASTER_TILE,
        NLCD_NODATA,
        NLCD_PREFIX,
        PAD,
        PER_RES,
        SETTLE,
        TAU,
        VIEW_H,
        VIEW_W,
        YEAR_AEF,
        YEAR_NLCD,
        ZOOM0,
    )


@app.cell
def _(math, np):
    # ---- EPSG:5070 both ways, closed form (verified against pyproj to 3e-10 deg) ----
    _a, _f = 6378137.0, 1 / 298.257222101
    _e2 = 2 * _f - _f * _f
    _e = math.sqrt(_e2)
    _lat0, _lon0, _lat1, _lat2 = map(math.radians, (23.0, -96.0, 29.5, 45.5))

    def _q(p):
        s = np.sin(p)
        return (1 - _e2) * (
            s / (1 - _e2 * s * s) - (1 / (2 * _e)) * np.log((1 - _e * s) / (1 + _e * s))
        )

    def _m(p):
        return math.cos(p) / math.sqrt(1 - _e2 * math.sin(p) ** 2)

    _m1, _m2, _q1, _q2, _q0 = _m(_lat1), _m(_lat2), _q(_lat1), _q(_lat2), _q(_lat0)
    _n = (_m1 * _m1 - _m2 * _m2) / (_q2 - _q1)
    _C = _m1 * _m1 + _n * _q1
    _rho0 = _a * math.sqrt(_C - _n * _q0) / _n

    def albers_fwd(lon, lat):
        lon, lat = np.radians(lon), np.radians(lat)
        rho = _a * np.sqrt(_C - _n * _q(lat)) / _n
        th = _n * (lon - _lon0)
        return rho * np.sin(th), _rho0 - rho * np.cos(th)

    def albers_inv(x, y):
        rho = np.sqrt(x * x + (_rho0 - y) ** 2)
        th = np.arctan2(x, _rho0 - y)
        qq = (_C - rho * rho * _n * _n / (_a * _a)) / _n
        phi = np.arcsin(qq / 2)
        for _ in range(6):
            s = np.sin(phi)
            phi = phi + ((1 - _e2 * s * s) ** 2 / (2 * np.cos(phi))) * (
                qq / (1 - _e2)
                - s / (1 - _e2 * s * s)
                + (1 / (2 * _e)) * np.log((1 - _e * s) / (1 + _e * s))
            )
        return np.degrees(_lon0 + th / _n), np.degrees(phi)

    return albers_fwd, albers_inv


@app.cell
def _(
    BASE_RES,
    CELL_BUDGET,
    MAX_RES,
    MIN_RES,
    PAD,
    PER_RES,
    VIEW_H,
    VIEW_W,
    ZOOM0,
    math,
):
    # ---- the camera -> box and res --------------------------------------------
    _CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) from the view; the widget reports
        its canvas size (`w`, `h`) with every move, the constants are the seed."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (
            vs["longitude"] - half_lon,
            _y_to_lat(yc + half_y),
            vs["longitude"] + half_lon,
            _y_to_lat(yc - half_y),
        )

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (
            max(-179.9, b[0] - dx),
            max(-85.0, b[1] - dy),
            min(179.9, b[2] + dx),
            min(85.0, b[3] + dy),
        )

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box, dres=0):
        """The ladder's res for this zoom (+ the strip's offset), coarsened until the
        box fits CELL_BUDGET."""
        r = max(MIN_RES, min(MAX_RES, BASE_RES + dres + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / _CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return (
            outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3]
        )

    return box_km2, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule). One context, both folds.
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
    return (ctx,)


@app.cell
async def _(
    GeoTIFF,
    Image,
    NLCD_LEVEL_FOR_RES,
    NLCD_NODATA,
    NLCD_PREFIX,
    RASTER_TILE,
    S3Store,
    Transformer,
    Window,
    YEAR_NLCD,
    albers_fwd,
    albers_inv,
    asyncio,
    ctx,
    io,
    math,
    np,
    time,
    xr,
):
    # ---- NLCD: the pyramid reader (the nlcd-zoom notebook's, by copy) + the fold ----
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _g = await GeoTIFF.open(
        f"{NLCD_PREFIX}/Annual_NLCD_LndCov_{YEAR_NLCD}_CU_C1V1.tif", store=_store
    )
    _levels = [_g, *_g.overviews]
    _L, _B, _R, _T = _g.bounds

    # ---- the cheap paint: NLCD as Web Mercator tiles the kernel renders for the
    # widget's TileLayer (deck asks over an anywidget custom message, the kernel
    # answers with a PNG; lonboard's raster layer does the same under the hood).
    # A tile is the COG level nearest the tile's ground resolution, sampled at the
    # 256x256 output pixel centres through the closed-form Albers forward (the
    # COG is EPSG:5070, the tiles are 3857): a nearest-neighbour reprojection in
    # numpy, ~ms per tile. NLCD's own colormap, nodata -> alpha 0.
    _cmap = _g.colormap.as_array()
    _tf84 = Transformer.from_crs(_g.crs, "EPSG:4326", always_xy=True)
    nlcd_bounds = _tf84.transform_bounds(*_g.bounds)  # (W, S, E, N) lon/lat
    _png_cache = {}
    _blank = {"png": None}
    _px0 = (_R - _L) / _levels[0].shape[1]  # native pixel size, m

    def _blank_png():
        if _blank["png"] is None:
            buf = io.BytesIO()
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
            _blank["png"] = buf.getvalue()
        return _blank["png"]

    async def nlcd_tile_png(z, x, y):
        """PNG bytes for Web Mercator tile (z, x, y), RASTER_TILE px square."""
        key = (z, x, y)
        if key in _png_cache:
            return _png_cache[key]
        n = 2 ** z
        lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
        lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        if lon1 < nlcd_bounds[0] or lon0 > nlcd_bounds[2] or lat_n < nlcd_bounds[1] or lat_s > nlcd_bounds[3]:
            return _blank_png()
        T = RASTER_TILE
        js = (np.arange(T) + 0.5) / T
        lons = lon0 + js * (lon1 - lon0)
        my = (y + js) / n
        lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * my))))
        LON, LAT = np.meshgrid(lons, lats)
        ax, ay = albers_fwd(LON.ravel(), LAT.ravel())
        gres = 40075016.686 * math.cos(math.radians((lat_n + lat_s) / 2)) / (n * T)
        li = int(max(0, min(len(_levels) - 1, round(math.log2(max(gres, _px0) / _px0)))))
        H, W = _levels[li].shape
        px = (_R - _L) / W
        cols = np.floor((ax - _L) / px).astype(np.int64)
        rows = np.floor((_T - ay) / px).astype(np.int64)
        ok = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H) & np.isfinite(ax) & np.isfinite(ay)
        if not ok.any():
            _png_cache[key] = _blank_png()
            return _png_cache[key]
        c0, c1 = int(cols[ok].min()), int(cols[ok].max()) + 1
        r0, r1 = int(rows[ok].min()), int(rows[ok].max()) + 1
        arr, _ = await _read_window(li, c0, r0, c1 - c0, r1 - r0)
        out = np.full(T * T, NLCD_NODATA, np.uint8)
        out[ok] = arr[rows[ok] - r0, cols[ok] - c0]
        out = out.reshape(T, T)
        rgba = np.empty((T, T, 4), np.uint8)
        rgba[..., :3] = _cmap[out]
        rgba[..., 3] = np.where(out == NLCD_NODATA, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        _png_cache[key] = buf.getvalue()
        if len(_png_cache) > 4000:
            _png_cache.pop(next(iter(_png_cache)))
        return _png_cache[key]

    TILE = 512
    TILE_BUDGET = 384 * 1024 * 1024
    _tiles = {}
    _held = {"bytes": 0}
    _sem = asyncio.Semaphore(32)

    async def _tile(li, ty, tx):
        rd = _levels[li]
        H, W = rd.shape
        r0, c0 = ty * TILE, tx * TILE
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        async with _sem:
            ra = await rd.read(window=Window(col_off=c0, row_off=r0, width=w, height=h))
        return np.asarray(np.ma.filled(ra.as_masked(), NLCD_NODATA)).reshape(h, w)

    async def _read_window(li, col0, row0, wpx, hpx):
        ty0, ty1 = row0 // TILE, (row0 + hpx - 1) // TILE
        tx0, tx1 = col0 // TILE, (col0 + wpx - 1) // TILE
        want = [(li, ty, tx) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        need = [k for k in want if k not in _tiles]
        fetched = 0
        if need:
            got = await asyncio.gather(*(_tile(*k) for k in need))
            for k, a in zip(need, got):
                _tiles[k] = a
                _held["bytes"] += a.nbytes
                fetched += a.size
            while _held["bytes"] > TILE_BUDGET and len(_tiles) > len(want):
                for k in list(_tiles):
                    if k not in want:
                        _held["bytes"] -= _tiles.pop(k).nbytes
                        break
                else:
                    break
        out = np.full((hpx, wpx), NLCD_NODATA, dtype=np.uint8)
        for k in want:
            _, ty, tx = k
            a = _tiles[k]
            sr, sc = ty * TILE, tx * TILE
            r0, c0 = max(row0, sr), max(col0, sc)
            r1, c1 = min(row0 + hpx, sr + a.shape[0]), min(col0 + wpx, sc + a.shape[1])
            if r1 <= r0 or c1 <= c0:
                continue
            out[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = a[r0 - sr : r1 - sr, c0 - sc : c1 - sc]
        return out, fetched

    async def nlcd_fold(box, res):
        """Majority NLCD class per res cell over the box, from the level the res
        deserves. Returns (arrow table, stats string)."""
        t0 = time.time()
        li = NLCD_LEVEL_FOR_RES[res]
        rd = _levels[li]
        H, W = rd.shape
        px = (_R - _L) / W
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 9), np.full(9, E_), np.linspace(E_, W_, 9), np.full(9, W_)])
        lats = np.concatenate([np.full(9, N_), np.linspace(N_, S_, 9), np.full(9, S_), np.linspace(S_, N_, 9)])
        ax, ay = albers_fwd(lons, lats)
        c0 = max(0, int((ax.min() - _L) / px))
        c1 = min(W, int(math.ceil((ax.max() - _L) / px)))
        r0 = max(0, int((_T - ay.max()) / px))
        r1 = min(H, int(math.ceil((_T - ay.min()) / px)))
        if c1 <= c0 or r1 <= r0:
            return None, "NLCD: box outside CONUS"
        arr, fetched = await _read_window(li, c0, r0, c1 - c0, r1 - r0)
        xs = _L + (np.arange(c0, c1) + 0.5) * px
        ys = _T - (np.arange(r0, r1) + 0.5) * px
        X, Y = np.meshgrid(xs, ys)
        lon, lat = albers_inv(X, Y)
        t1 = time.time()
        try:
            ctx.deregister_table("lc")
        except Exception:
            pass
        ctx.from_dataset(
            "lc",
            xr.Dataset(
                {"cls": (("y", "x"), arr), "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
                coords={"y": ys, "x": xs},
            ),
            chunks={"y": 512},
        )
        out = ctx.sql(f"""
            WITH c AS (
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, cls, count(*) AS n
                FROM lc
                WHERE cls != {NLCD_NODATA}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY 1, 2
            )
            SELECT cell,
                   first_value(cls ORDER BY n DESC, cls ASC) AS cls,
                   sum(n) AS npx,
                   CAST(max(n) AS DOUBLE) / sum(n) AS purity
            FROM c GROUP BY cell
        """).to_arrow_table()
        return out, (
            f"NLCD L{li} {arr.shape[1]:,}x{arr.shape[0]:,} px ({fetched / 1e6:.1f} Mpx fetched) "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return nlcd_bounds, nlcd_fold, nlcd_tile_png


@app.cell
async def _(
    AEF_INDEX_URL,
    AEF_LEVEL_FOR_RES,
    AEF_MAX_FILES,
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_X0,
    AEF_Y0,
    CACHE_DIR,
    GeoTIFF,
    MOSAIC_MIN_RES,
    ObjectStore,
    S3Store,
    Transformer,
    Window,
    YEAR_AEF,
    asyncio,
    ctx,
    duckdb,
    math,
    np,
    os,
    pa,
    pq,
    time,
    xr,
):
    # ---- AlphaEarth: two sources, one fold ------------------------------------
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _mstore = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, prefix=AEF_PREFIX
    )
    _ds = xr.open_zarr(ObjectStore(_mstore, read_only=True), chunks=None, consolidated=False)
    _ti = int(np.where(_ds.time.values == YEAR_AEF)[0][0])

    # The COG index for the year, cached as parquet under tmp (the full index is
    # 302k rows over HTTP, ~10 s; the year's CONUS slice is a few thousand).
    os.makedirs(CACHE_DIR, exist_ok=True)
    _idx_path = os.path.join(CACHE_DIR, f"aef_index_{YEAR_AEF}.parquet")
    if not os.path.exists(_idx_path):
        _c = duckdb.connect()
        _c.execute("INSTALL httpfs; LOAD httpfs")
        _t = _c.execute(f"""
            SELECT path, crs, utm_west, utm_south, utm_east, utm_north,
                   wgs84_west, wgs84_south, wgs84_east, wgs84_north
            FROM read_parquet('{AEF_INDEX_URL}')
            WHERE year = {YEAR_AEF}
              AND wgs84_east > -125.5 AND wgs84_west < -66 AND wgs84_north > 24 AND wgs84_south < 50
        """).arrow().read_all()
        pq.write_table(_t, _idx_path)
        _c.close()
    aef_index = pq.read_table(_idx_path)
    _IDX = {k: aef_index[k].to_numpy() for k in aef_index.column_names if k not in ("path", "crs")}
    _PATHS = aef_index["path"].to_pylist()
    _CRS = aef_index["crs"].to_pylist()

    _open = {}  # path -> GeoTIFF (headers only)
    _sem = asyncio.Semaphore(64)
    _tf_fwd, _tf_inv = {}, {}

    def _tf(crs):
        if crs not in _tf_fwd:
            _tf_fwd[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            _tf_inv[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return _tf_fwd[crs], _tf_inv[crs]

    async def _get(path):
        rel = path.split("source.coop/")[1]
        if rel not in _open:
            async with _sem:
                _open[rel] = await GeoTIFF.open(rel, store=_store)
        return _open[rel]

    async def _read_cog(i, li, box):
        """One file's overview window over the box: (int8 (64, h, w), lon, lat) or None.

        Rows and columns go through the file's AFFINE TRANSFORM, not its bounds:
        these COGs are stored SOUTH-UP (transform e = +10, origin at the south
        edge; `bounds` reports bottom > top), and a north-up assumption mirrors
        every tile within its 82 km (measured 2026-08-24: agreement 86-98% below
        0.5 on the COG rungs, worse than random, against 14% from the mosaic).
        """
        g = await _get(_PATHS[i])
        ov = g.overviews[li]
        H, W = ov.shape
        t = g.transform
        sx, sy = t.a * (g.width / W), t.e * (g.height / H)  # signed overview pixel sizes
        fwd, inv = _tf(_CRS[i])
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 5), np.full(5, E_), np.linspace(E_, W_, 5), np.full(5, W_)])
        lats = np.concatenate([np.full(5, N_), np.linspace(N_, S_, 5), np.full(5, S_), np.linspace(S_, N_, 5)])
        ux, uy = fwd.transform(lons, lats)
        cc = (np.asarray(ux) - t.c) / sx
        rr = (np.asarray(uy) - t.f) / sy
        c0 = max(0, int(np.floor(np.nanmin(cc))))
        c1 = min(W, int(np.ceil(np.nanmax(cc))))
        r0 = max(0, int(np.floor(np.nanmin(rr))))
        r1 = min(H, int(np.ceil(np.nanmax(rr))))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await ov.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, r1 - r0, c1 - c0)
        xs = t.c + (np.arange(c0, c1) + 0.5) * sx
        ys = t.f + (np.arange(r0, r1) + 0.5) * sy
        X, Y = np.meshgrid(xs, ys)
        lon, lat = inv.transform(X, Y)
        return a, lon, lat

    _DEQ = ", ".join(
        f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64)
    )

    def _fold_rows(res, box, cols, lat, lon):
        """cols: int8 (64, n); lat/lon (n,). One 1-D Dataset, one fold."""
        W_, S_, E_, N_ = box
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)}
            | {"lat": (("i",), lat), "lon": (("i",), lon)},
            coords={"i": np.arange(lat.size)},
        )
        try:
            ctx.deregister_table("aef")
        except Exception:
            pass
        ctx.from_dataset("aef", ds1, chunks={"i": 262_144})
        return ctx.sql(f"""
            SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS naef, {_DEQ}
            FROM aef
            WHERE e00 != {AEF_NODATA}
              AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
            GROUP BY cell
        """).to_arrow_table()

    async def aef_fold(box, res):
        """Mean AlphaEarth vector per res cell over the box, from the mosaic (res >=
        MOSAIC_MIN_RES) or the COG overviews. Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        if res >= MOSAIC_MIN_RES:
            x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
            y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
            loop = asyncio.get_running_loop()
            emb = await loop.run_in_executor(
                None, lambda: _ds.embeddings.isel(time=_ti, y=slice(y0, y1), x=slice(x0, x1)).values
            )
            lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
            lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
            LON, LAT = np.meshgrid(lon, lat)
            t1 = time.time()
            out = _fold_rows(res, box, emb.reshape(64, -1), LAT.ravel(), LON.ravel())
            return out, (
                f"AEF mosaic {emb.shape[2]:,}x{emb.shape[1]:,} px ({emb.nbytes / 1e6:.0f} MB) "
                f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
            )
        li = AEF_LEVEL_FOR_RES[res]
        hit = np.where(
            (_IDX["wgs84_east"] > W_) & (_IDX["wgs84_west"] < E_)
            & (_IDX["wgs84_north"] > S_) & (_IDX["wgs84_south"] < N_)
        )[0]
        if len(hit) == 0:
            return None, "AEF: no COG tiles under the view"
        if len(hit) > AEF_MAX_FILES:
            return None, f"AEF: {len(hit):,} tiles under the view (> {AEF_MAX_FILES:,}); zoom in for AlphaEarth"
        parts = await asyncio.gather(*(_read_cog(int(i), li, box) for i in hit))
        parts = [p for p in parts if p is not None]
        if not parts:
            return None, "AEF: nothing read"
        cols = np.concatenate([p[0].reshape(64, -1) for p in parts], axis=1)
        lon = np.concatenate([p[1].ravel() for p in parts])
        lat = np.concatenate([p[2].ravel() for p in parts])
        t1 = time.time()
        out = _fold_rows(res, box, cols, lat, lon)
        return out, (
            f"AEF cog ov{li} ({10 * 2 ** (li + 1)} m) {len(parts):,} files {cols.shape[1] / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return aef_fold, aef_index


@app.cell
def _(
    AGREE_CMAP,
    ALPHA_FLAT,
    ALPHA_MAX,
    ALPHA_MIN,
    ALPHA_RAMP,
    CLASSES,
    CLUSTER_HEX,
    COV_FLAT,
    COV_MIN,
    DIM_ALPHA,
    K_CLUSTERS,
    MIN_CLASS_CELLS,
    RAMPS,
    TAU,
    duckdb,
    io,
    np,
    pa,
    time,
):
    # ---- a FRAME: scores, clusters, coverage and colors for one folded view ------
    # GeoArrow for the boundaries (the counties film's transport): WKB rings ->
    # geoarrow.linestring with INTERLEAVED coords (what @geoarrow/deck.gl-layers
    # reads), through arro3 so the extension metadata survives into the IPC
    # stream (pyarrow's own table constructor drops it, measured)
    import pyarrow.ipc as pa_ipc
    from geoarrow.rust.core import from_wkb as ga_from_wkb, linestring as ga_linestring
    from arro3.core import Array as ArroArray, Table as ArroTable
    # No hexagon geometry here: the widget's H3HexagonLayer draws from the cell
    # ids, and the per-cell coverage is an attribute (the kepler-style column).
    _PAL = np.array([tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in CLUSTER_HEX], np.uint8)
    # the agreement ramp: AGREE_CMAP's stops interpolated to a 256-entry LUT
    _hx = RAMPS[AGREE_CMAP]
    _stops = np.array([[int(_hx[i + j:i + j + 2], 16) for j in (0, 2, 4)] for i in range(0, len(_hx), 6)], np.float64)
    _RAMP = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    RAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in _RAMP[i]) for i in range(0, 256, 17)]  # 16 swatches for the legend
    con = duckdb.connect()
    # h3 + spatial for the low-agreement boundaries (edges_for below); the fold
    # itself stays the h3 UDF in DataFusion
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial")

    def build_frame(nlcd_cells, aef_cells):
        """Join the two folds, score, cluster, build both hexagon tables."""
        import time as _time
        _tt = {"t": _time.time()}
        _lap = {}

        def lap(name):
            now = _time.time()
            _lap[name] = now - _tt["t"]
            _tt["t"] = now

        con.register("nlcd_cells", nlcd_cells)
        if aef_cells is None:
            j = con.execute("SELECT cell, cls, npx, purity FROM nlcd_cells ORDER BY cell").arrow().read_all()
            has_aef = False
        else:
            con.register("aef_cells", aef_cells)
            j = con.execute("SELECT * FROM nlcd_cells JOIN aef_cells USING (cell) ORDER BY cell").arrow().read_all()
            has_aef = True
        n = j.num_rows
        lap("join")
        cls = j["cls"].to_numpy().astype(np.int64)
        if has_aef and n > 0:
            V = np.stack([j[f"e{i:02d}"].to_numpy() for i in range(64)], axis=1).astype(np.float32)
            hom = np.linalg.norm(V, axis=1)
            V = V / np.maximum(hom, 1e-9)[:, None]
            present, counts = np.unique(cls, return_counts=True)
            proto_classes = present[counts >= MIN_CLASS_CELLS]
            if len(proto_classes) >= 2:
                P = np.stack([V[cls == c].mean(0) for c in proto_classes])
                P /= np.linalg.norm(P, axis=1)[:, None]
                cos = V @ P.T
                idx = np.searchsorted(proto_classes, cls)
                has = np.isin(cls, proto_classes)
                idx = np.where(has, idx, 0)
                rows = np.arange(n)
                own = np.where(has, cos[rows, idx], np.nan)
                other = cos.copy()
                other[rows, idx] = -np.inf
                alt_i = other.argmax(1)
                alt = np.where(has, proto_classes[alt_i], -1)
                margin = own - other[rows, alt_i]
                agree = np.where(has, 1.0 / (1.0 + np.exp(-margin / TAU)), np.nan)
            else:
                agree = np.full(n, np.nan)
                alt = np.full(n, -1)
            lap("score")
            # spherical k-means (float32, 12 Lloyd steps: the assignment barely moves after)
            k = min(K_CLUSTERS, n)
            rng = np.random.default_rng(0)
            C = V[rng.integers(n)][None, :]
            for _ in range(1, k):
                d = np.clip(1 - (V @ C.T).max(1), 1e-12, None).astype(np.float64)
                C = np.vstack([C, V[rng.choice(n, p=d / d.sum())]])
            clu = np.zeros(n, np.int64)
            for _ in range(12):
                new = (V @ C.T).argmax(1)
                if (new == clu).all():
                    break
                clu = new
                for kk in range(k):
                    if (clu == kk).any():
                        C[kk] = V[clu == kk].mean(0)
                C /= np.linalg.norm(C, axis=1)[:, None]
            clu = (V @ C.T).argmax(1)
            order = np.argsort(-np.bincount(clu, minlength=k))
            clu = np.argsort(order)[clu]
            lap("kmeans")
        else:
            hom = np.full(n, np.nan)
            agree = np.full(n, np.nan)
            alt = np.full(n, -1)
            clu = np.zeros(n, np.int64)

        cells = pa.table({
            "cell": j["cell"],
            "cls": pa.array(cls.astype(np.uint8)),
            "name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in cls]),
            "cluster": pa.array(clu.astype(np.int16)),
            "purity": j["purity"],
            "homogeneity": pa.array(hom.astype(np.float32)),
            "agree": pa.array(agree.astype(np.float32)),
            "alt_name": pa.array([CLASSES.get(int(c), ("none",))[0] for c in alt]),
        })

        lap("table")
        cov = np.where(np.isnan(agree), 1.0, COV_MIN + (1 - COV_MIN) * np.clip(agree, 0, 1)).astype(np.float32)
        # the flat paints (NLCD H3, clusters H3): every hexagon at COV_FLAT
        cov_flat = np.full(n, COV_FLAT, np.float32)
        # highlight disagreement: coverage inverted too (the least-backed cells
        # full-size and solid, the agreeing ones small and faint), else the two
        # cues point opposite ways and the map reads as pale blobs with bold dots
        cov_inv = np.where(np.isnan(agree), COV_MIN, COV_MIN + (1 - COV_MIN) * (1 - np.clip(agree, 0, 1))).astype(np.float32)
        cellid = cells["cell"].to_numpy().astype(np.uint64)
        lap("hex")
        rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
        alpha_agree = np.where(
            np.isnan(agree), ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.clip(agree, 0, 1)
        ).astype(np.uint8)
        # reversed: the least-backed (smallest) cells solid, the agreeing ones faint
        # (Stephen: "so the smallest coverage cells are noticeable")
        alpha_inv = np.where(
            np.isnan(agree), ALPHA_MIN, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1 - np.clip(agree, 0, 1))
        ).astype(np.uint8)
        rgb_clu = _PAL[clu % len(_PAL)]
        # color by agreement: the ramp on the value (unscored cells grey);
        # `inv` reverses it (warm = disagreement)
        _ai = np.where(np.isnan(agree), 0, np.clip(agree, 0, 1) * 255).round().astype(np.int64)
        _unscored = np.isnan(agree)[:, None]
        rgb_ramp = np.where(_unscored, 128, _RAMP[_ai]).astype(np.uint8)
        rgb_ramp_inv = np.where(_unscored, 128, _RAMP[255 - _ai]).astype(np.uint8)

        def fill(paint, sel, hit=None, inv=False, ramp=False):
            """(N, 4) uint8 rgba for a paint: the widget's getFillColor attribute."""
            if paint == "clusters":
                c, key = rgb_clu, 100 + clu
            elif paint == "agreement" and ramp:
                c, key = (rgb_ramp_inv if inv else rgb_ramp), cls
            else:
                c, key = rgb, cls
            if paint == "agreement":
                a = np.full(len(cls), ALPHA_RAMP, np.uint8) if ramp else (alpha_inv if inv else alpha_agree)
            else:
                a = np.full(len(cls), ALPHA_FLAT, np.uint8)
            if sel:
                a = np.where(np.isin(key, list(sel)), a, DIM_ALPHA).astype(np.uint8)
            rgba = np.ascontiguousarray(np.concatenate([c, a[:, None]], axis=1)).astype(np.uint8)
            if hit is not None:
                rgba[cellid == hit] = (255, 255, 255, 255)
            return rgba

        def coverage(paint, inv=False):
            """(N,) float32: the widget's getCoverage attribute for a paint."""
            if paint == "agreement":
                return cov_inv if inv else cov
            return cov_flat

        lap("colors")
        a_ok = agree[~np.isnan(agree)]
        score = (
            f"{n:,} cells · agreement p50 {np.median(a_ok):.2f} · {(a_ok < 0.5).mean() * 100:.0f}% below 0.5"
            if len(a_ok) else f"{n:,} cells · NLCD only"
        ) + " (" + " ".join(f"{k} {v:.1f}" for k, v in _lap.items()) + ")"
        return {
            "cells": cells, "cellid": cellid, "fill": fill, "coverage": coverage,
            "cls": cls, "clu": clu, "agree": agree, "has_aef": has_aef, "score": score,
        }

    def label_components(a, b, n):
        """Connected components over undirected edges (a, b) among n nodes, in
        numpy: min-label hooking + pointer jumping until stable. 73k low cells /
        91k edges in 0.15 s (361 rounds); the H3 neighbour edges come from DuckDB."""
        lab = np.arange(n)
        while True:
            m = lab.copy()
            np.minimum.at(m, a, lab[b])
            np.minimum.at(m, b, lab[a])
            m = m[m]
            while True:
                mm = m[m]
                if np.array_equal(mm, m):
                    break
                m = mm
            if np.array_equal(m, lab):
                return lab
            lab = m

    def edges_for(frame, thr, min_cells, alpha):
        """Boundaries of the clusters of cells with agreement < thr, one color per
        cluster: the NLCD color of its majority class. DuckDB's h3 extension does
        the geometry (`h3_grid_ring_unsafe` for the neighbour edges,
        `h3_cells_to_multi_polygon_wkb` per blob for the dissolve, the H3
        outer-boundary walk that is 30x faster than ST_Union_Agg of hexagons),
        numpy labels the components in between (the polygon -> cells unnest,
        `h3_polygon_wkb_to_cells`, is O(cells x vertices) and measured 23 s on a
        150k-cell frame with one percolating blob; the labels are 0.15 s). Blob
        area is the sum of `h3_cell_area` (ST_Transform to Albers measured 8.8 s
        on the same frame). Blobs under min_cells cells are dropped. Every ring
        (outer and holes) is one closed LineString; `ipc` is ONE Arrow IPC stream
        of a geoarrow.linestring table (interleaved coords, EPSG:4326, `color`
        rgba uint8[4] per ring, `cls`, `km2`), the layout
        @geoarrow/deck.gl-layers' GeoArrowPathLayer draws directly. Memoised on
        the frame per (thr, min_cells, alpha). Measured at the 150k budget with
        half the cells low: 0.25 s end to end."""
        key = (round(float(thr), 3), int(min_cells), int(alpha))
        cache = frame.setdefault("edges", {})
        if key in cache:
            return cache[key]
        t0 = time.time()
        agree = frame["agree"]
        n_low = int(np.sum(agree < thr))
        out = {"ipc": b"", "n_low": n_low, "blobs": 0, "max_km2": 0.0, "rings": 0, "ms": 0}
        if n_low:
            con.register("edge_cells", frame["cells"])
            low = con.sql(
                "SELECT cell, cls, row_number() OVER (ORDER BY cell) - 1 AS i FROM edge_cells WHERE agree < $thr",
                params={"thr": float(thr)},
            ).arrow().read_all()
            con.register("edge_low", low)
            e = con.sql("""
                WITH nb AS (SELECT i, UNNEST(h3_grid_ring_unsafe(cell, 1)) AS ncell FROM edge_low)
                SELECT nb.i AS a, l2.i AS b FROM nb JOIN edge_low l2 ON nb.ncell = l2.cell WHERE nb.i < l2.i
            """).arrow().read_all()
            lab = label_components(e["a"].to_numpy(), e["b"].to_numpy(), low.num_rows)
            con.register("edge_blob", pa.table({"i": np.arange(low.num_rows), "blob": lab}))
            r = con.sql("""
                WITH g AS (
                  SELECT b.blob, mode(l.cls) AS cls, count(*) AS ncell,
                         sum(h3_cell_area(l.cell, 'km^2')) AS km2,
                         ST_GeomFromWKB(h3_cells_to_multi_polygon_wkb(list(l.cell))) AS geom
                  FROM edge_low l JOIN edge_blob b USING (i)
                  GROUP BY b.blob HAVING count(*) >= $min_cells),
                p AS (SELECT blob, cls, ncell, km2, UNNEST(ST_Dump(geom)).geom AS poly FROM g),
                q AS (SELECT blob, cls, ncell, km2, UNNEST(ST_Dump(ST_Boundary(poly))).geom AS ring FROM p),
                s AS (SELECT count(*) AS blobs, max(km2) AS max_km2 FROM g)
                SELECT q.blob, q.cls, q.ncell, q.km2, ST_AsWKB(q.ring) AS wkb, s.blobs, s.max_km2
                FROM q, s ORDER BY q.ncell DESC
            """, params={"min_cells": int(min_cells)}).arrow().read_all()
            if r.num_rows:
                geom = ga_from_wkb(
                    r["wkb"].combine_chunks().cast(pa.binary()),
                    to_type=ga_linestring("xy", coord_type="interleaved", crs="EPSG:4326"),
                )
                cls = r["cls"].to_numpy().astype(np.int64)
                rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
                rgba = np.concatenate([rgb, np.full((len(cls), 1), alpha, np.uint8)], axis=1).ravel()
                color = pa.FixedSizeListArray.from_arrays(pa.array(rgba, pa.uint8()), 4)
                tbl = pa.table(ArroTable.from_arrays(
                    [ArroArray.from_arrow(geom), ArroArray.from_arrow(color),
                     ArroArray.from_arrow(r["cls"].combine_chunks()), ArroArray.from_arrow(r["km2"].combine_chunks())],
                    names=["geometry", "color", "cls", "km2"],
                )).combine_chunks()
                sink = io.BytesIO()
                with pa_ipc.new_stream(sink, tbl.schema) as w:
                    w.write_table(tbl)
                out.update(ipc=sink.getvalue(), rings=int(r.num_rows),
                           blobs=int(r["blobs"][0].as_py()), max_km2=float(r["max_km2"][0].as_py()))
        out["ms"] = int(1000 * (time.time() - t0))
        cache[key] = out
        return out

    def legend_for(frame, paint, ramp=False, inv=False):
        cls, clu, agree = frame["cls"], frame["clu"], frame["agree"]
        tot = max(1, len(cls))
        items = []
        if paint == "agreement" and ramp and frame["has_aef"]:
            # the ramp bar, cool to warm left to right; the labels say which end
            # is which (the highlight checkbox swaps them, not the bar)
            items.append({
                "ramp": RAMP_HEX, "cmap": AGREE_CMAP,
                "lo": "agreement" if inv else "disagreement",
                "hi": "disagreement" if inv else "agreement",
            })
        if paint == "clusters" and frame["has_aef"]:
            for k in range(int(clu.max()) + 1 if len(clu) else 0):
                m = clu == k
                if not m.any():
                    continue
                cc, cn = np.unique(cls[m], return_counts=True)
                top = sorted(zip(cn, cc), reverse=True)[:3]
                mix = ", ".join(f"{100 * nn / m.sum():.0f}% {CLASSES.get(int(c), ('?',))[0]}" for nn, c in top)
                a = agree[m]
                a = a[~np.isnan(a)]
                items.append({
                    "code": 100 + k, "name": f"cluster {k}", "hex": CLUSTER_HEX[k % len(CLUSTER_HEX)],
                    "pct": round(100 * int(m.sum()) / tot, 1),
                    "p50": f"{np.median(a):.2f}" if len(a) else "none", "note": mix,
                })
        else:
            codes, nn = np.unique(cls, return_counts=True)
            for code, cnt in sorted(zip(codes, nn), key=lambda t: -t[1]):
                if int(code) not in CLASSES:
                    continue
                a = agree[cls == code]
                a = a[~np.isnan(a)]
                items.append({
                    "code": int(code), "name": CLASSES[int(code)][0],
                    "hex": "#%02x%02x%02x" % CLASSES[int(code)][1],
                    "pct": round(100 * int(cnt) / tot, 1),
                    "p50": f"{np.median(a):.2f}" if len(a) else "none",
                    "note": "" if len(a) else "(unscored)",
                })
        return items

    return build_frame, con, edges_for, legend_for


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """The strip under the map (the cdl-ftw-zarr-marimo HudControls skeleton,
        trimmed): paint buttons, pickable legend, panel, status; the one element
        docks into the map's fullscreen. Clicks are the map widget's own (deck
        picking), not captured here."""

        ctl = traitlets.Unicode("").tag(sync=True)
        dres = traitlets.Unicode("0").tag(sync=True)  # kernel -> browser: the offset in force
        thr0 = traitlets.Unicode("0.5").tag(sync=True)  # kernel -> browser: the threshold slider's seed
        status = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;" +
            "font:13px ui-sans-serif,system-ui,sans-serif;padding:.35rem 0 0;" +
            "user-select:none;width:100%";
          const btnCss =
            "font:13px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.2rem .6rem;border-radius:5px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          // The four buttons are VISIBILITY (Stephen): one layer on the map at a time;
          // click another and the map goes to that layer; click the one that is on
          // and it disappears. No stacking. Hiding keeps the fold: the kernel flips
          // the widget's visibility and the frame stays, so coming back is instant.
          let paint = "agreement";
          const sel = new Set();
          let seq = 0;
          let edgesOn = false;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, paint: paint, sel: Array.from(sel), inv: inv.checked, acol: acol,
              edges: edgesOn, thr: parseFloat(thr.value), n: ++seq }, extra || {})));
            model.save_changes();
          };
          const onCss = (b, on) => {
            b.style.borderColor = on ? "#2b6cb0" : "rgba(127,127,127,.45)";
            b.style.fontWeight = on ? "600" : "400";
          };
          const paintBox = document.createElement("span");
          paintBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const pl = document.createElement("span");
          pl.textContent = "layer";
          const mkPaint = (key, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => { paint = paint === key ? null : key; sel.clear(); stylePaint(); send("set"); renderLegend(); };
            return [key, b];
          };
          const paintBtns = [
            mkPaint("raster", "NLCD raster", "NLCD as its own tiles, at any zoom; click again to hide"),
            mkPaint("nlcd", "NLCD H3", "NLCD's majority class per hexagon, flat colors; click again to hide"),
            mkPaint("agreement", "agreement H3", "NLCD's colors; hexagon size and alpha follow how well AlphaEarth backs the class; click again to hide"),
            mkPaint("clusters", "AlphaEarth clusters H3", "the embedding on its own: k-means clusters of the cell vectors, no labels; click again to hide"),
          ];
          const invLab = document.createElement("label");
          invLab.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const inv = document.createElement("input");
          inv.type = "checkbox"; inv.checked = false;
          invLab.appendChild(inv); invLab.appendChild(document.createTextNode("highlight disagreement"));
          invLab.title = "agreement H3: the least-backed (smallest) cells solid, the agreeing ones faint";
          inv.addEventListener("change", () => send("set"));
          // color by agreement: the agreement paint's hexagons on a cool-to-warm
          // ramp (cool = disagreement) instead of NLCD's colors; the highlight
          // checkbox reverses the ramp. Coverage still follows agreement.
          let acol = false;
          const acB = document.createElement("button");
          acB.textContent = "color by agreement"; acB.style.cssText = btnCss;
          acB.title = "agreement H3: color the hexagons by agreement (cool = disagreement, warm = agreement) instead of NLCD's colors; highlight disagreement reverses the ramp";
          const styleAc = () => { onCss(acB, acol); acB.style.opacity = paint === "agreement" ? "1" : ".5"; };
          acB.onclick = () => { acol = !acol; styleAc(); send("set"); };
          const stylePaint = () => { paintBtns.forEach(([k, b]) => onCss(b, k === paint)); styleAc(); };
          stylePaint();
          paintBox.append(pl, ...paintBtns.map(([, b]) => b), invLab, acB);
          // boundaries around the clusters of low-agreement cells (dissolved in
          // the kernel by DuckDB's h3 extension), with the agreement threshold
          // that defines "low": a slider that commits on change (never input:
          // every commit is a dissolve and a send)
          const edgeBox = document.createElement("span");
          edgeBox.style.cssText = "display:inline-flex;gap:.35rem;align-items:center";
          const edB = document.createElement("button");
          edB.textContent = "boundaries"; edB.style.cssText = btnCss;
          edB.title = "outline every cluster of cells whose agreement is below the threshold; click again to hide";
          const thr = document.createElement("input");
          thr.type = "range"; thr.min = "0.05"; thr.max = "0.95"; thr.step = "0.05";
          thr.value = String(model.get("thr0") || "0.5");
          thr.style.cssText = "width:7rem;vertical-align:middle";
          thr.title = "agreement below this is inside a boundary";
          const thrV = document.createElement("span");
          thrV.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.6rem";
          const paintThr = () => { thrV.textContent = "< " + parseFloat(thr.value).toFixed(2); };
          paintThr();
          thr.addEventListener("input", paintThr);
          thr.addEventListener("change", () => { if (edgesOn) send("set"); });
          const styleEd = () => onCss(edB, edgesOn);
          styleEd();
          edB.onclick = () => { edgesOn = !edgesOn; styleEd(); send("set"); };
          edgeBox.append(edB, thr, thrV);
          // res: the offset from the ladder (-2..+2). + refolds the CURRENT view one
          // step finer (zooming in never does on its own); the offset resets when
          // the camera leaves the served box, and the kernel echoes it back.
          const resBox = document.createElement("span");
          resBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const rl = document.createElement("span"); rl.textContent = "res";
          const rv = document.createElement("span");
          rv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums;min-width:1.6rem;text-align:center";
          const mkRes = (d, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => {
              const cur = parseInt(model.get("dres") || "0", 10);
              const nxt = Math.max(-2, Math.min(2, cur + d));
              if (nxt !== cur) send("dres", { dres: nxt });
            };
            return b;
          };
          const rMinus = mkRes(-1, "−", "refold this view one step coarser");
          const rPlus = mkRes(+1, "+", "refold this view one step finer (7x the cells, and the read)");
          const paintR = () => {
            const v = parseInt(model.get("dres") || "0", 10);
            rv.textContent = (v > 0 ? "+" : "") + v;
          };
          model.on("change:dres", paintR);
          paintR();
          resBox.append(rl, rMinus, rv, rPlus);
          const legendBox = document.createElement("div");
          legendBox.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;" +
            "gap:.15rem .7rem;flex:1 1 100%;min-width:14rem;font-size:13px";
          const renderLegend = () => {
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); }
            catch (e) { items = []; }
            legendBox.innerHTML = "";
            if (sel.size) {
              const x = document.createElement("button");
              x.textContent = "× all";
              x.style.cssText =
                "font:11px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;border:1px solid " +
                "#2b6cb0;background:transparent;color:inherit";
              x.onclick = () => { sel.clear(); send("set"); renderLegend(); };
              legendBox.appendChild(x);
            }
            items.forEach((it) => {
              if (it.ramp) {
                // the agreement ramp bar with its end labels
                const r = document.createElement("span");
                r.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;font:12px ui-sans-serif,system-ui,sans-serif";
                r.title = it.cmap + ": color by agreement";
                r.innerHTML =
                  '<span style="opacity:.75">' + it.lo + '</span>' +
                  '<span style="display:inline-block;width:9rem;height:10px;border-radius:2px;' +
                  "background:linear-gradient(to right," + it.ramp.join(",") + ')"></span>' +
                  '<span style="opacity:.75">' + it.hi + '</span>';
                legendBox.appendChild(r);
                return;
              }
              const b = document.createElement("button");
              const on = sel.has(it.code);
              b.style.cssText =
                "display:inline-flex;align-items:center;gap:.3rem;" +
                "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;background:transparent;" +
                "color:inherit;border:1px solid " +
                (on ? "#2b6cb0" : "transparent") + (on ? ";font-weight:600" : "");
              b.title = it.pct + "% of cells · agreement p50 " + it.p50;
              b.innerHTML =
                '<span style="width:10px;height:10px;border-radius:2px;' +
                "background:" + it.hex + ';display:inline-block"></span>' +
                it.name + (it.note ? ' <span style="opacity:.6">' + it.note + "</span>" : "");
              b.onclick = () => {
                if (sel.has(it.code)) sel.delete(it.code); else sel.add(it.code);
                send("set"); renderLegend();
              };
              legendBox.appendChild(b);
            });
          };
          model.on("change:legend", renderLegend);
          renderLegend();
          // analyze what's in view (the crops notebook's button): the kernel fills
          // the panel with the view's summary; × clear empties it
          const anBox = document.createElement("span");
          anBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const anB = document.createElement("button");
          anB.textContent = "analyze what's in view"; anB.style.cssText = btnCss;
          anB.title = "per NLCD class in view: share, area, agreement; and the clusters' make-up";
          anB.onclick = () => send("analyze");
          const clB = document.createElement("button");
          clB.textContent = "× clear"; clB.style.cssText = btnCss;
          clB.onclick = () => send("clear");
          anBox.append(anB, clB);
          let labelsOn = true;
          const lbB = document.createElement("button");
          lbB.textContent = "labels"; lbB.style.cssText = btnCss; lbB.title = "place labels over the map, on or off";
          const styleLb = () => { lbB.style.borderColor = labelsOn ? "#2b6cb0" : "rgba(127,127,127,.45)"; lbB.style.fontWeight = labelsOn ? "600" : "400"; };
          styleLb();
          lbB.onclick = () => { labelsOn = !labelsOn; styleLb(); send("labels", { labels: labelsOn }); };
          anBox.append(lbB);
          // the search field: Photon (cdl-ftw's), Enter geocodes camera-biased on the
          // kernel and the first hit flies the map; the fold follows the moveend
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place…";
          search.title = "Photon geocoder: Enter flies to the first hit";
          search.style.cssText =
            "width:11rem;font:13px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) { e.preventDefault(); send("search", { q: q }); }
          });
          anBox.append(search);
          box.append(paintBox, edgeBox, resBox, anBox, legendBox);
          const panel = document.createElement("div");
          panel.style.cssText = "font:13.5px ui-sans-serif,system-ui,sans-serif;padding:.25rem 0";
          const status = document.createElement("div");
          status.style.cssText =
            "font:13px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.2rem 0;min-height:1.2em;white-space:pre-line";
          const wrap = document.createElement("div");
          wrap.style.cssText = "width:100%;box-sizing:border-box";
          wrap.dataset.aefStrip = "1";
          wrap.append(box, panel, status);
          const killOld = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("[data-aef-strip]").forEach((w) => {
              if (w !== wrap) { w.dataset.dead = "1"; w.remove(); }
            });
            root.querySelectorAll("*").forEach((n) => { if (n.shadowRoot) killOld(n.shadowRoot); });
          };
          killOld(document);
          el.appendChild(wrap);
          const realFs = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement)
              fe = fe.shadowRoot.fullscreenElement;
            return fe;
          };
          const onFs = () => {
            if (wrap.dataset.dead || !el.isConnected) {
              wrap.remove();
              document.removeEventListener("fullscreenchange", onFs);
              return;
            }
            const fe = realFs();
            if (fe && fe !== el && !el.contains(fe)) {
              if (getComputedStyle(fe).position === "static") fe.style.position = "relative";
              wrap.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;z-index:30;" +
                "background:rgba(255,255,255,.94);color:#111;box-sizing:border-box;" +
                "padding:.6rem 1.4rem .7rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "width:100%;box-sizing:border-box";
              el.appendChild(wrap);
            }
          };
          document.addEventListener("fullscreenchange", onFs);
          const paintS = () => { status.textContent = model.get("status") || ""; };
          model.on("change:status", paintS);
          paintS();
          const paintP = () => { panel.innerHTML = model.get("panel") || ""; };
          model.on("change:panel", paintP);
          paintP();
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            wrap.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class DeckMap(anywidget.AnyWidget):
        """The map: maplibre (Carto Positron, interleaved) with deck.gl 9.3.10 from
        esm.sh drawing INSIDE it under the label layers, the HRRR counties film's
        chassis. Two deck layers with real ids: `nlcd`, a TileLayer whose tiles the
        kernel renders on request (anywidget custom messages, PNG bytes back), and
        `hexes`, an H3HexagonLayer subclass with ONE extra instanced attribute,
        `instanceCoverage` (kepler.gl's EnhancedColumnLayer trick: the column
        vertex shader's `column.coverage` becomes `column.coverage *
        instanceCoverage`), so every paint draws from cell ids + rgba + float32
        coverage and nothing is tessellated in the kernel.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `cov` (f32),
        `config` (JSON: height, home, raster mode, labels, hex_zoom, extent).
        Browser -> kernel: `view` (JSON lon/lat/zoom + canvas w/h on every
        moveend) and `pick` (JSON: the clicked cell as a hex string, or null;
        deck's GPU pick when it answers, else h3-js on the click's lon/lat)."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        cov = traitlets.Bytes(b"").tag(sync=True)
        # low-agreement boundaries: one GeoArrow IPC stream (geoarrow.linestring,
        # interleaved coords; the counties film's transport), drawn by a
        # GeoArrowPathLayer under `config.edges`
        edges = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (z, x, y) -> PNG bytes; the wiring sets it
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError:
                self.send({"kind": "tile", "id": content.get("id"), "empty": True})

        async def _tile(self, c):
            try:
                png = await self.tile_fn(int(c["z"]), int(c["x"]), int(c["y"])) if self.tile_fn else None
            except Exception:
                png = None
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        // every deck import pins the same versions AND the same ?deps= per package
        // (esm.sh hashes a module by its deps list), so the whole graph resolves to
        // ONE @deck.gl/core; apache-arrow rides along for the GeoArrow layers. The
        // strings are the HRRR counties film's (crawled: one core, one luma set).
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {ColumnLayer, BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {GeoArrowPathLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/aggregation-layers@9.3.10,@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";

        const STYLES = {
          labels: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
          nolabels: "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
        };

        // kepler.gl's EnhancedColumnLayer, in short: one instanced float per column,
        // multiplied into the two places the vertex shader reads `column.coverage`.
        class CoverageColumnLayer extends ColumnLayer {
          getShaders() {
            const s = super.getShaders();
            s.vs = s.vs
              .replace("in vec3 instancePickingColors;", "in vec3 instancePickingColors;\nin float instanceCoverage;")
              .replaceAll("column.coverage", "(column.coverage * instanceCoverage)");
            return s;
          }
          initializeState() {
            super.initializeState();
            this.getAttributeManager().addInstanced({
              instanceCoverage: {size: 1, accessor: "getCoverage", defaultValue: 1},
            });
          }
        }
        CoverageColumnLayer.layerName = "CoverageColumnLayer";
        CoverageColumnLayer.defaultProps = {...ColumnLayer.defaultProps, getCoverage: {type: "accessor", value: 1}};

        class CoverageH3Layer extends H3HexagonLayer {
          _getForwardProps() {
            const p = super._getForwardProps();
            p.getCoverage = this.props.getCoverage;
            p.updateTriggers.getCoverage = this.props.updateTriggers.getCoverage;
            return p;
          }
        }
        CoverageH3Layer.layerName = "CoverageH3Layer";
        CoverageH3Layer.defaultProps = {...H3HexagonLayer.defaultProps, getCoverage: {type: "accessor", value: 1}};

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        function copyOf(u8) {  // an aligned private copy (DataView slices are not aligned)
          return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
        }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const root = document.createElement("div");
          root.style.cssText = "position:relative;width:100%";
          const mapEl = document.createElement("div");
          mapEl.style.cssText = "width:100%;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
          const note = document.createElement("div");
          note.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;font:11px ui-monospace,Menlo,monospace;" +
            "color:#333;background:rgba(255,255,255,.85);padding:2px 6px;border-radius:3px;pointer-events:none;display:none";
          note.dataset.aefNote = "1";
          root.append(mapEl, note);
          el.append(css, root);
          const say = (t) => { note.textContent = t; note.style.display = t ? "block" : "none"; };

          let hexes = [], N = 0, colors = null, cov = null, seq = 0, map = null, overlay = null;
          let hexIndex = new Map(), res = -1;
          // The bytes are COPIED the moment a trait changes: read later (even 0 ms
          // later, from a timer) the DataView marimo handed over is no longer
          // readable and loadCells silently left N = 0 (measured: the change event
          // saw 46,440 bytes, the deferred read saw nothing, a manual reload worked).
          const raw = {cells: null, colors: null, cov: null, edges: null};
          const grab = (k) => {
            try { const u8 = bytesOf(model.get(k)); raw[k] = u8 && u8.length ? copyOf(u8) : null; }
            catch (e) { raw[k] = null; say("grab " + k + ": " + e.message); }
          };
          let dataObj = null;  // one object per (cells, colors, cov) triple: identity is deck's change signal
          let edgeTable = null;  // the boundaries: an arrow Table with a geoarrow.linestring column
          function loadEdges() {
            const buf = raw.edges;
            if (!buf || !buf.byteLength) { edgeTable = null; return; }
            try { edgeTable = arrow.tableFromIPC(new Uint8Array(buf)); }
            catch (e) { edgeTable = null; say("edges: " + e.message); }
          }

          function loadCells() {
            const buf = raw.cells;
            if (!buf || !buf.byteLength) { hexes = []; N = 0; hexIndex = new Map(); res = -1; return; }
            const ids = new BigUint64Array(buf);
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
          }
          function loadAttrs() {
            const c8 = raw.colors, v8 = raw.cov;
            colors = c8 && c8.byteLength === N * 4 ? new Uint8Array(c8) : null;
            cov = v8 && v8.byteLength === N * 4 ? new Float32Array(v8) : null;
            // OVERFILL: deck's low-precision H3 mode draws every cell with the mesh of
            // the cell at the viewport centre (measured right: 523 m radius at res 8
            // against a 531 m average edge), so cells tile exactly at the centre and
            // drift to a hairline gap towards the edges of a wide view. 2% overlap
            // covers the drift without visible overlap seams at ALPHA_FLAT; more
            // (1.08 was tried) shows dark seams where translucent cells overlap.
            // (The uniform ~7% gaps Stephen saw were COV_FLAT 0.8 in a kernel that
            // had not been reloaded after the revert to 1.0.)
            if (cov) { const f = cfg.overfill || 1.02; for (let i = 0; i < cov.length; i++) cov[i] *= f; }
            dataObj = N && colors && cov ? {
              length: N,
              attributes: {
                getFillColor: {value: colors, size: 4, normalized: true},
                getCoverage: {value: cov, size: 1},
              },
            } : null;
          }

          // NLCD tiles: ask the kernel, get a PNG back on the custom-message channel
          const pending = new Map();
          let tseq = 0;
          model.on("msg:custom", (msg, buffers) => {
            if (msg && msg.kind === "fly" && map) {
              // the geocoder's hit: maplibre flies, moveend sends the view, the kernel folds
              map.flyTo({center: [msg.lon, msg.lat], zoom: msg.zoom, duration: msg.duration || 2000});
              return;
            }
            if (!msg || msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.empty || !buffers || !buffers.length) { p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(p.resolve, () => p.resolve(null));
          });
          const getTileData = ({index, signal}) => new Promise((resolve) => {
            const id = ++tseq;
            pending.set(id, {resolve});
            model.send({kind: "tile", id, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { pending.delete(id); resolve(null); });
          });

          // the raster shows where its switch is on and the hexagons are not drawn
          // (their switch off, or no frame, or below hex_zoom); the hexagons show
          // where their switch is on. Both are `visible` flips: the tiles and the
          // frame stay in the browser, nothing is refetched or refolded.
          const hexesDrawn = () => cfg.show_hexes !== false && !!dataObj && !!map && map.getZoom() >= (cfg.hex_zoom || 9);
          const rasterOn = () => cfg.show_raster !== false && !hexesDrawn();

          function layers() {
            const out = [];
            out.push(new TileLayer({
              id: "nlcd",
              getTileData,
              tileSize: cfg.tile || 256,
              minZoom: 0, maxZoom: 13,
              extent: cfg.extent || null,
              visible: rasterOn(),
              refinementStrategy: "no-overlap",
              beforeId: cfg.labels_slot || "watername_ocean",
              renderSubLayers: (p) => {
                if (!p.data) return null;
                const {west, south, east, north} = p.tile.bbox;
                return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
              },
            }));
            // Two hexagon layers, one on the map at a time. The FLAT paints (NLCD,
            // clusters) need no coverage, so they take the stock H3HexagonLayer on
            // its highPrecision path: every cell's own boundary, tessellated in the
            // browser, no shared mesh, no drift, no gaps (Stephen: "a separate h3
            // hexagon layer for those two"). Agreement keeps the coverage column.
            if (dataObj && cfg.flat) out.push(new H3HexagonLayer({
              id: "hexes-flat",
              data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              visible: cfg.show_hexes !== false,
              pickable: true,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            if (dataObj && !cfg.flat) out.push(new CoverageH3Layer({
              id: "hexes",
              data: dataObj,
              getHexagon: (_, {index}) => hexes[index],
              filled: true, stroked: false, extruded: false,
              highPrecision: false,
              coverage: 1,
              visible: cfg.show_hexes !== false,
              _subLayerProps: {"hexagon-cell": {type: CoverageColumnLayer}},
              pickable: true,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            // boundaries of the low-agreement clusters (dissolved in the kernel by
            // DuckDB's h3 extension): one PathLayer over every closed ring, drawn
            // whatever the paint (they are the hexagons' own frame, so they hide
            // with them below hex_zoom)
            if (cfg.edges && edgeTable && edgeTable.numRows && (!map || map.getZoom() >= (cfg.hex_zoom || 9))) out.push(new GeoArrowPathLayer({
              id: "edges",
              data: edgeTable,
              getPath: edgeTable.getChild("geometry"),
              getColor: edgeTable.getChild("color"),
              widthUnits: "pixels", getWidth: cfg.edge_width || 2, widthMinPixels: 1,
              jointRounded: true,
              _validate: false,
              beforeId: cfg.labels_slot || "watername_ocean",
            }));
            // the picked cell: its own color stays, a gold outline from its boundary
            if (cfg.hit && hexIndex.has(cfg.hit)) {
              let ring = null;
              try { ring = cellToBoundary(cfg.hit, true); } catch (e) { ring = null; }
              if (ring) out.push(new PathLayer({
                id: "picked",
                data: [ring],
                getPath: (d) => d,
                getColor: [255, 200, 40, 255],
                widthUnits: "pixels", getWidth: 3, widthMinPixels: 2,
                beforeId: cfg.labels_slot || "watername_ocean",
              }));
            }
            return out;
          }
          function update() { if (overlay) overlay.setProps({layers: layers()}); }

          function labels(on) {
            if (!map || !map.isStyleLoaded()) return;
            const st = map.getStyle();
            if (!st || !st.layers) return;
            st.layers.forEach((l) => {
              if (l.layout && l.layout["text-field"] !== undefined)
                map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
            });
          }

          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            model.set("view", JSON.stringify({
              longitude: c.lng, latitude: c.lat, zoom: map.getZoom(),
              w: mapEl.clientWidth, h: mapEl.clientHeight, n: ++seq,
            }));
            model.save_changes();
          }

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            map = new maplibregl.Map({
              container: mapEl, style: STYLES.labels,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: true},
            });
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            map.addControl(new maplibregl.FullscreenControl(), "top-right");
            overlay = new MapboxOverlay({
              interleaved: true,
              layers: [],
              onClick: (info) => {
                // deck's GPU pick first; when it returns nothing (it did on every
                // click here, as in the counties film: interleaved inside marimo's
                // shadow DOM), the click's lon/lat -> h3-js cell at the frame's res
                let cell = info && info.layer && (info.layer.id === "hexes" || info.layer.id === "hexes-flat") && info.index >= 0 ? hexes[info.index] : null;
                let how = "gpu";
                if (!cell && info && info.coordinate && res >= 0) {
                  try { const h = latLngToCell(info.coordinate[1], info.coordinate[0], res); if (hexIndex.has(h)) cell = h; how = "h3"; }
                  catch (e) { how = "h3: " + e.message; }
                }
                if (cfg.debug) say("pick " + how + " " + cell);
                model.set("pick", JSON.stringify({cell, lon: info && info.coordinate ? info.coordinate[0] : null,
                  lat: info && info.coordinate ? info.coordinate[1] : null, n: ++seq}));
                model.save_changes();
              },
              onError: (e) => say("deck: " + (e && e.message ? e.message : e)),
            });
            map.addControl(overlay);
            if (cfg.debug) window.__aef = {overlay, map, model, reload, get N() { return N; }, get colors() { return colors; }, get cov() { return cov; }, get dataObj() { return dataObj; }, get edgeTable() { return edgeTable; }, get raw() { return raw; }, get cfg() { return cfg; }};
            map.on("load", () => { labels(cfg.labels !== false); update(); sendView(); });
            map.on("moveend", sendView);
            map.on("zoom", () => update());
            map.on("error", (e) => { if (e && e.error && e.error.message) say("map: " + e.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            document.addEventListener("fullscreenchange", () => { setTimeout(() => { try { map.resize(); } catch (e) {} }, 50); });
          }

          let pendingLoad = null;
          let needCells = false;
          const flush = () => {  // cells/colors/cov land as three trait changes: rebuild once
            pendingLoad = null;
            try { if (needCells) loadCells(); needCells = false; loadAttrs(); loadEdges(); update(); }
            catch (e) { say("load: " + e.message); console.error(e); }
          };
          const reload = () => { needCells = true; if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          const reattr = () => { if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          model.on("change:cells", () => { grab("cells"); reload(); });
          model.on("change:colors", () => { grab("colors"); reattr(); });
          model.on("change:cov", () => { grab("cov"); reattr(); });
          model.on("change:edges", () => { grab("edges"); reattr(); });
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (cfg.height && cfg.height !== was.height && !document.fullscreenElement) mapEl.style.height = cfg.height + "px";
            if (cfg.labels !== was.labels) labels(cfg.labels !== false);
            update();
          });
          try { grab("cells"); grab("colors"); grab("cov"); grab("edges"); loadCells(); loadAttrs(); loadEdges(); boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (DeckMap,)


@app.cell
def _(DeckMap, EDGE_THR, HOME, LABELS_SLOT, RASTER_TILE, json):
    # ---- the map: built ONCE, empty; never re-runs for a parameter -----------------
    deck = DeckMap(config=json.dumps({
        "height": 720, "home": dict(HOME), "show_raster": True, "show_hexes": True, "labels": True,
        "hex_zoom": 9.0, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "task": None, "loop": None,
        "paint": "agreement", "show_raster": True, "show_hexes": True,
        "sel": set(), "hit": None, "memo": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "dres": 0,  # the strip's res offset; a statement about the box it was set on
        "inv": False,  # reversed alpha: disagreeing cells solid
        "acol": False,  # color by agreement (the ramp) instead of NLCD's colors
        "labels": True,
        "edges": False, "thr": EDGE_THR,  # the low-agreement boundaries and their threshold
        "edges_sent": None,  # (frame, thr) the widget holds
    }
    deck  # the cell's LAST statement: what marimo displays
    return HOLD, deck


@app.cell
def _(EDGE_THR, HudControls, mo):
    hud = mo.ui.anywidget(HudControls(thr0=str(EDGE_THR)))
    hud
    return (hud,)


@app.cell
def _(
    CLASSES,
    CLUSTER_HEX,
    EDGE_ALPHA,
    EDGE_MIN_CELLS,
    EDGE_WIDTH,
    HEX_ZOOM,
    HOLD,
    HOME,
    SETTLE,
    VIEW_W,
    aef_fold,
    asyncio,
    build_frame,
    con,
    contains,
    deck,
    edges_for,
    hud,
    json,
    legend_for,
    math,
    nlcd_bounds,
    nlcd_fold,
    nlcd_tile_png,
    np,
    pad_box,
    res_for_view,
    time,
    urllib,
    view_to_bbox,
):
    # ---- wiring: the camera loop and the strip. Re-runs freely (un-observes its
    # old handlers first); the map cell never re-runs.
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    deck.tile_fn = nlcd_tile_png

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        """Push the widget's config (raster mode, labels, extent) as one JSON."""
        c = json.loads(deck.config or "{}")
        c.update(kw)
        deck.config = json.dumps(c)

    def _show():
        """The widget's visibility for the paint: the raster at any zoom under
        `raster`; the hexagons (with the raster where they are not drawn, below
        HEX_ZOOM) under a hexagon paint; nothing under None."""
        HOLD["show_raster"] = HOLD["paint"] is not None
        HOLD["show_hexes"] = HOLD["paint"] not in (None, "raster")
        _cfg(show_raster=HOLD["show_raster"], show_hexes=HOLD["show_hexes"], flat=HOLD["paint"] in ("nlcd", "clusters"))

    _show()
    _cfg(extent=list(nlcd_bounds), hex_zoom=HEX_ZOOM, edges=HOLD["edges"], edge_width=EDGE_WIDTH)

    def _hexes_off(msg):
        """No fold for this camera (below HEX_ZOOM): the frame is dropped and the
        widget draws no hexagon layer. Hiding the hexagons is NOT this: that is a
        visibility flip in the browser and the frame stays."""
        if HOLD["sent"] is not None or HOLD["edges_sent"] is not None:
            with deck.hold_sync():
                deck.cells, deck.colors, deck.cov = b"", b"", b""
                deck.edges = b""
            HOLD["sent"], HOLD["edges_sent"] = None, None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        try:
            hud.widget.legend = "[]"
            hud.widget.panel = ""
        except Exception:
            pass
        _say(msg)

    def _say_dres():
        try:
            hud.widget.dres = str(HOLD["dres"])
        except Exception:
            pass

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    def _paint():
        """Hand the current frame's cells, colors and coverage to the widget."""
        fr = HOLD["frame"]
        if fr is None:
            return
        rgba = fr["fill"](HOLD["paint"], HOLD["sel"], None, HOLD["inv"], HOLD["acol"])
        cov = fr["coverage"](HOLD["paint"], HOLD["inv"])
        # the boundaries: dissolved on demand (memoised on the frame per
        # threshold), sent only when the frame or the threshold changed; while
        # off, the widget keeps what it has and `config.edges` hides it
        ed = None
        if HOLD["edges"]:
            ed = edges_for(fr, HOLD["thr"], EDGE_MIN_CELLS, EDGE_ALPHA)
            HOLD["edge_note"] = (
                f"boundaries: {ed['n_low']:,} cells below {HOLD['thr']:.2f} · {ed['blobs']:,} blobs ≥ {EDGE_MIN_CELLS} cells"
                + (f" · largest {ed['max_km2']:,.1f} km²" if ed["blobs"] else "")
                + f" · {ed['rings']:,} rings · {ed['ms']} ms"
            )
        else:
            HOLD["edge_note"] = ""
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None, edges=HOLD["edges"])
        with deck.hold_sync():
            if HOLD["sent"] is not fr:
                deck.cells = fr["cellid"].astype("<u8").tobytes()
                HOLD["sent"] = fr
            deck.colors = rgba.tobytes()
            deck.cov = cov.astype("<f4").tobytes()
            if ed is not None and HOLD["edges_sent"] != (id(fr), round(HOLD["thr"], 3)):
                deck.edges = ed["ipc"]
                HOLD["edges_sent"] = (id(fr), round(HOLD["thr"], 3))
        try:
            hud.widget.legend = json.dumps(legend_for(fr, HOLD["paint"], HOLD["acol"], HOLD["inv"]))
        except Exception:
            pass

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · NLCD as its own tiles · zoom in past {HEX_ZOOM:g} for the agreement hexes")
            return
        if not HOLD["show_hexes"]:
            # hidden (nothing on, or the raster): no fold for a camera nobody is
            # looking at through the hexes; the held frame stays, and a hexagon
            # paint coming back serves this view
            what = "NLCD raster (its own tiles)" if HOLD["paint"] == "raster" else "nothing on"
            _say(f"zoom {vsd['zoom']:.1f} · {what} · pick a hexagon paint for the fold")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and not force:
            # ZOOMING IN NEVER REFOLDS (Stephen: "you don't necessarily want the res
            # to change right away"): the served hexagons cover the view and the
            # browser scales them. Finer detail is the strip's res + button. Only a
            # camera that LEAVES the served box, or a coarser ladder res (zoomed
            # out, cheap), refolds on its own.
            ladder = res_for_view(vsd, box, HOLD["dres"])
            if ladder >= HOLD["res"]:
                note = f" · finer available (res {ladder}: press res +)" if ladder > HOLD["res"] else ""
                _say(HOLD.get("last_status", "") + " · held" + note)
                return
        if not inside and HOLD["box"] is not None:
            HOLD["dres"] = 0  # a raised offset does not follow you to a new box
            _say_dres()
        res = res_for_view(vsd, box, HOLD["dres"])
        key = (res, tuple(round(v, 3) for v in box))
        t0 = time.time()
        _say(f"res {res} · folding…")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            (nl, s1), (ae, s2) = await asyncio.gather(nlcd_fold(box, res), aef_fold(box, res))
            if nl is None or nl.num_rows == 0:
                _say(f"res {res} · {s1}")
                return
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, nl, ae)
            stats = f"res {res} · {s1} · {s2} · frame {time.time() - t1:.1f} s"
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        HOLD["last_status"] = f"{stats} · {fr['score']} · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
        _say(HOLD["last_status"] + ("\n" + HOLD["edge_note"] if HOLD.get("edge_note") else ""))

    async def refresh(vs):
        """Settle-debounced, coalescing fold (the deforest notebook's loop)."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    continue
                await _serve(vs)
                vs = HOLD["pending"]
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as exc:
            _say(f"failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            deck.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    deck.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    _CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215}

    def _analyze_html(fr):
        """The view's summary as HTML for the strip's panel: every NLCD class (share
        of cells, km2, agreement p50, share below 0.5, usual alternative) and, with
        the embedding in, each cluster's NLCD make-up."""
        cls, clu, agree = fr["cls"], fr["clu"], fr["agree"]
        n = max(1, len(cls))
        km2 = _CELL_KM2.get(HOLD["res"], 0.0)
        a_ok = agree[~np.isnan(agree)]
        head = (
            f"<b>res {HOLD['res']}</b> · {n:,} cells · {n * km2:,.0f} km² · "
            + (f"agreement p50 {np.median(a_ok):.2f}, {(a_ok < 0.5).mean() * 100:.0f}% below 0.5" if len(a_ok) else "NLCD only")
        )
        td = "padding:.1rem .6rem .1rem 0;white-space:nowrap"
        rows = []
        codes, counts = np.unique(cls, return_counts=True)
        con.register("cur_cells", fr["cells"])
        alts = dict(con.execute(
            "SELECT name, mode(alt_name) FILTER (WHERE agree < 0.5) FROM cur_cells GROUP BY name"
        ).fetchall())
        for code, cnt in sorted(zip(codes, counts), key=lambda t: -t[1]):
            if int(code) not in CLASSES:
                continue
            nm, rgb = CLASSES[int(code)]
            a = agree[cls == code]
            a = a[~np.isnan(a)]
            chip = f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:rgb{rgb};margin-right:.35rem;vertical-align:-1px'></span>"
            rows.append(
                f"<tr><td style='{td}'>{chip}{nm}</td><td style='{td};text-align:right'>{100 * cnt / n:.1f}%</td>"
                f"<td style='{td};text-align:right'>{cnt * km2:,.0f} km²</td>"
                + (f"<td style='{td};text-align:right'>{np.median(a):.2f}</td><td style='{td};text-align:right'>{(a < 0.5).mean() * 100:.0f}%</td>"
                   f"<td style='{td};opacity:.75'>{alts.get(nm) or ''}</td>" if len(a) else f"<td style='{td}' colspan=3><span style='opacity:.6'>unscored</span></td>")
                + "</tr>"
            )
        th = "padding:.1rem .6rem .1rem 0;text-align:left;opacity:.6;font-weight:500"
        table = (
            f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr><th style='{th}'>NLCD class</th><th style='{th}'>of cells</th>"
            f"<th style='{th}'>area</th><th style='{th}'>agreement p50</th><th style='{th}'>below 0.5</th><th style='{th}' title='the class whose per-view AlphaEarth prototype the disagreeing cells sit closest to; a suggestion relative to this scene'>AlphaEarth usually suggests</th></tr>"
            + "".join(rows) + "</table>"
        )
        clus = ""
        if fr["has_aef"] and len(clu):
            items = []
            for k in range(int(clu.max()) + 1):
                m = clu == k
                if not m.any():
                    continue
                cc, cn = np.unique(cls[m], return_counts=True)
                top = sorted(zip(cn, cc), reverse=True)[:3]
                mix = ", ".join(f"{100 * nn / m.sum():.0f}% {CLASSES.get(int(c), ('?',))[0]}" for nn, c in top)
                chip = f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{CLUSTER_HEX[k % len(CLUSTER_HEX)]};margin-right:.35rem;vertical-align:-1px'></span>"
                items.append(f"<tr><td style='{td}'>{chip}cluster {k}</td><td style='{td};text-align:right'>{100 * m.sum() / n:.1f}%</td><td style='{td};opacity:.75'>{mix}</td></tr>")
            clus = (
                f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr><th style='{th}'>AlphaEarth cluster</th><th style='{th}'>of cells</th><th style='{th}'>made of (NLCD)</th></tr>"
                + "".join(items) + "</table>"
            )
        return head + table + clus

    def _selection_panel(fr):
        if not HOLD["sel"]:
            return ""
        con.register("cur_cells", fr["cells"])
        if HOLD["paint"] == "clusters":
            rows = con.execute("""
                SELECT 'cluster ' || cluster, count(*), round(median(agree), 2),
                       round(100 * avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END), 0), mode(name)
                FROM cur_cells WHERE cluster IN (SELECT UNNEST(?)) GROUP BY cluster ORDER BY 2 DESC
            """, [[k - 100 for k in HOLD["sel"]]]).fetchall()
            word = "mostly"
        else:
            rows = con.execute("""
                SELECT name, count(*), round(median(agree), 2),
                       round(100 * avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END), 0),
                       mode(alt_name) FILTER (WHERE agree < 0.5)
                FROM cur_cells WHERE cls IN (SELECT UNNEST(?)) GROUP BY name ORDER BY 2 DESC
            """, [list(HOLD["sel"])]).fetchall()
            word = "AlphaEarth usually suggests"
        return " · ".join(
            f"<b>{nm}</b>: {cnt:,} cells"
            + (f", agreement p50 {p50:.2f}, {pct:.0f}% below 0.5" if p50 is not None else "")
            + (f", {word} <i>{alt}</i>" if alt else "")
            for nm, cnt, p50, pct, alt in rows
        )

    def _on_pick(change):
        """deck's own picking: the widget sends the clicked cell (hex string) or null."""
        fr = HOLD["frame"]
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        if fr is None:
            return
        try:
            cellh = p.get("cell")
            if not cellh:
                HOLD["hit"] = None
                hud.widget.panel = _selection_panel(fr)
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT name, agree, alt_name, purity, homogeneity, cluster FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                hud.widget.panel = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                nm, ag, alt, pur, hom, ck = r
                scored = ag is not None and not np.isnan(ag)
                hud.widget.panel = (
                    f"<b>{nm}</b>{where}: cluster {ck}, agreement "
                    + (f"{ag:.2f}" if scored else "unscored")
                    # the runner-up: the class whose PER-VIEW prototype the cell's
                    # vector is closest to, a suggestion relative to the classes in
                    # this scene, not a classification (Stephen: "AEF suggests it
                    # could be this")
                    + (f", AlphaEarth suggests it could be <i>{alt}</i> (relative to this view)" if scored and ag < 0.5 and alt != "none" else "")
                    + f", NLCD purity {pur:.2f}"
                    + (f", homogeneity {hom:.3f}" if hom is not None and not np.isnan(hom) else "")
                )
        except Exception as e:
            hud.widget.panel = f"<span style='opacity:.7'>click: {e}</span>"
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            deck.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    deck.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    # ---- the search field: Photon, camera-biased, one ~0.3 s call on a thread;
    # the hit goes to the widget as a `fly` message (maplibre flyTo), whose
    # moveend sends the view back and the ordinary camera loop folds it. No
    # kernel-side camera state is touched here (cdl-ftw's geocoder, ported).
    def _photon_first(query, vs):
        params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            params["lon"] = round(vs["longitude"], 4)
            params["lat"] = round(vs["latitude"], 4)
        url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "x-sql-marimo aef nlcd deck notebook"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        feats = data.get("features") or []
        if not feats:
            return None
        f = feats[0]
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        name = ", ".join(str(v) for v in (p.get("name"), p.get("city"), p.get("state")) if v) or query
        return name, float(lon), float(lat), p.get("extent")

    async def _search(q):
        vs = _vsd(HOLD.get("vs"))
        try:
            hit = await asyncio.get_running_loop().run_in_executor(None, _photon_first, q, vs)
        except Exception as e:
            _say(f"search error: {type(e).__name__}: {e}")
            return
        if hit is None:
            _say(f"no match: {q}")
            return
        name, lon, lat, ext = hit
        w = vs.get("w") or VIEW_W
        if ext and len(ext) == 4:
            span = max(abs(ext[2] - ext[0]), abs(ext[1] - ext[3]) * 2, 0.01)
            zoom = math.log2(360.0 * (w / 512) / span) - 0.3
        else:
            zoom = 10.0
        zoom = max(3.5, min(13.5, zoom))
        deck.send({"kind": "fly", "lon": lon, "lat": lat, "zoom": zoom, "duration": 2000})
        _say(f"→ {name} · zoom {zoom:.1f}")

    def _on_ctl(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        _was = HOLD["paint"]
        HOLD["paint"] = c.get("paint", "agreement")  # None: the layer that was on was clicked off
        HOLD["sel"] = {int(x) for x in c.get("sel", [])}
        HOLD["inv"] = bool(c.get("inv", False))
        HOLD["acol"] = bool(c.get("acol", False))
        _ed_was = (HOLD["edges"], HOLD["thr"])
        HOLD["edges"] = bool(c.get("edges", False))
        try:
            HOLD["thr"] = min(0.99, max(0.01, float(c.get("thr", HOLD["thr"]))))
        except (TypeError, ValueError):
            pass
        fr = HOLD["frame"]
        if HOLD["paint"] != _was:
            # VISIBILITY. One layer at a time: the raster, one of the hexagon paints,
            # or nothing. A config flip in the browser; the frame is KEPT, so a
            # hexagon paint coming back is a repaint of the held frame (instant) and
            # a fold only if the camera left the box meanwhile.
            _show()
            if HOLD["paint"] in (None, "raster"):
                z = _vsd(HOLD["vs"])["zoom"]
                kept = " · fold kept" if fr is not None else ""
                _say(f"zoom {z:.1f} · " + ("NLCD raster (its own tiles)" if HOLD["paint"] == "raster" else "nothing on") + kept)
                return
            if _was in (None, "raster"):
                vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
                _paint()
                if HOLD["busy"]:
                    HOLD["pending"] = vs
                else:
                    HOLD["task"] = _spawn(_serve(vs))
                return
        if c.get("act") == "dres":
            HOLD["dres"] = max(-2, min(2, int(c.get("dres", 0))))
            _say_dres()
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            if HOLD["busy"]:
                HOLD["pending"] = vs
            else:
                HOLD["task"] = _spawn(_serve(vs, force=True))
            return
        if c.get("act") == "clear":
            hud.widget.panel = ""
            return
        if c.get("act") == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
            return
        if c.get("act") == "analyze":
            hud.widget.panel = _analyze_html(fr) if fr is not None else "<span style='opacity:.7'>no fold in view (zoom in past 9 with the hexagons on)</span>"
            return
        if c.get("act") == "search":
            q = str(c.get("q") or "").strip()
            try:
                if q:
                    _say(f"searching: {q}")
                    HOLD["stask"] = _spawn(_search(q))
            except Exception as e:  # comm-handler exceptions are silent
                _say(f"search error: {type(e).__name__}: {e}")
            return
        if fr is not None:
            hud.widget.panel = _selection_panel(fr)
        _paint()
        if (HOLD["edges"], HOLD["thr"]) != _ed_was and HOLD.get("last_status"):
            _say(HOLD["last_status"] + ("\n" + HOLD["edge_note"] if HOLD.get("edge_note") else ""))

    if HOLD.get("h_ctl") is not None:
        try:
            hud.widget.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    hud.widget.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    # the opening fold (or a repaint if a frame already exists after a re-run)
    if HOLD["frame"] is None:
        HOLD["task"] = _spawn(_serve(HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)))
    else:
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    The tables below are DuckDB over the CURRENT view's cells (press the button
    after the map settles): `cls`, `name`, `cluster`, `purity`, `homogeneity`,
    `agree`, `alt_name`.
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="tables for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no view folded yet*"))
    con.register("view_cells", HOLD["frame"]["cells"])
    per_class = mo.sql(
        """
        SELECT cls, name, count(*) AS cells,
               round(median(agree), 3) AS agree_p50,
               round(avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END) * 100, 1) AS pct_below_half,
               mode(alt_name) FILTER (WHERE agree < 0.5) AS aef_usually_suggests
        FROM view_cells GROUP BY cls, name ORDER BY cells DESC
        """,
        engine=con,
    )
    return (per_class,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None)
    confusion = mo.sql(
        """
        PIVOT (SELECT name, alt_name FROM view_cells WHERE agree < 0.5)
        ON alt_name USING count(*) GROUP BY name ORDER BY name
        """,
        engine=con,
    )
    return (confusion,)


if __name__ == "__main__":
    app.run()
