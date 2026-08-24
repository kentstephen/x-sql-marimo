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
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
# ]
# ///
"""NLCD backed or not by AlphaEarth, anywhere in CONUS, at any zoom.

xsql-aef-nlcd-agreement.py (one box, one fold) turned into a camera-driven fold:
every time the map settles, the ground under it is folded to H3 at the resolution
the zoom deserves, NLCD from its own overview pyramid and AlphaEarth from whichever
of its two source.coop copies can serve that rung:

  res 11    (zoomed in)   tge-labs/aef-mosaic   the 10 m Zarr, native, one window
  res 5-10  (zoomed out)  tge-labs/aef          the per-tile COGs' OVERVIEWS (mean
                                                embeddings at 40..2560 m), many files

Both folds are the h3 UDF in DataFusion; NLCD's majority class and AlphaEarth's
mean vector meet on the cell. Per view: class prototypes, the agreement (sigmoid
over the own-vs-runner-up cosine margin), spherical k-means clusters. The strip
under the map has the three paints (agreement: alpha + coverage; NLCD: regular
hexagons; AlphaEarth: the clusters), the pickable legend, a click that lights the
hexagon and tells its story. Prototypes and clusters are PER VIEW: they say what
is typical of a class HERE, and cluster colours are arbitrary per fold.

Measured from home (2026-08-24): a COG opens in 0.8 s, 162 open concurrently in
1.8 s and read their 2560 m overviews in 0.7 s; the ~2,000 files that cover CONUS
are a cold ~30 s at the coarsest rung, then cached (open handles + folded frames).
The mosaic rung is a native 10 m read: ~1-2 s at zoom 12, 10-20 s at zoom 10.

Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind." (CC-BY 4.0.)

Run: uv run marimo edit xsql-aef-nlcd-conus.py   (or --sandbox)
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
    from h3ronpy.vector import coordinates_to_cells, cells_to_wkb_polygons
    from pyproj import Transformer

    from arro3.core import Table as ArrowTable
    from lonboard import Map, PolygonLayer
    from lonboard.basemap import CartoStyle, MaplibreBasemap

    return (
        ArrowTable,
        CartoStyle,
        GeoTIFF,
        Map,
        MaplibreBasemap,
        ObjectStore,
        PolygonLayer,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        cells_to_wkb_polygons,
        coordinates_to_cells,
        duckdb,
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

    - **agreement** paint: NLCD's colours; faint and shrunken where the embedding does
      not back the word.
    - **NLCD** paint: regular hexagons, flat colours.
    - **AlphaEarth** paint: the embedding on its own, k-means clusters, the legend
      saying what each cluster is made of in NLCD terms.

    Click a hexagon for its story; click a legend chip to isolate a class or cluster.
    Prototypes and clusters are recomputed per view (local, honest, colours shift).

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
    HOME = {"longitude": -96.0, "latitude": 38.5, "zoom": 4.0}

    TAU = 0.02
    MIN_CLASS_CELLS = 30
    K_CLUSTERS = 10
    CLUSTER_HEX = ["#0072B2", "#E69F00", "#56B4E9", "#F0E442", "#CC79A7",
                   "#009E73", "#D55E00", "#999999", "#7B4EA3", "#6B3F1D"]
    ALPHA_MIN, ALPHA_MAX = 30, 235
    COV_MIN = 0.30
    DIM_ALPHA = 22

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
        AEF_PREFIX,
        AEF_RES,
        AEF_X0,
        AEF_Y0,
        ALPHA_MAX,
        ALPHA_MIN,
        BASE_RES,
        CACHE_DIR,
        CELL_BUDGET,
        CLASSES,
        CLUSTER_HEX,
        COV_MIN,
        DIM_ALPHA,
        HOME,
        K_CLUSTERS,
        MAX_RES,
        MIN_CLASS_CELLS,
        MIN_RES,
        MOSAIC_MIN_RES,
        NLCD_LEVEL_FOR_RES,
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
        """The flat camera footprint (W, S, E, N) from view_state and the canvas guess."""
        world = 512 * (2 ** vs["zoom"])
        half_lon = 360.0 * VIEW_W / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), VIEW_H / world / 2
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
    NLCD_LEVEL_FOR_RES,
    NLCD_NODATA,
    NLCD_PREFIX,
    S3Store,
    Window,
    YEAR_NLCD,
    albers_fwd,
    albers_inv,
    asyncio,
    ctx,
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

    return (nlcd_fold,)


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
    ALPHA_MAX,
    ALPHA_MIN,
    CLASSES,
    CLUSTER_HEX,
    COV_MIN,
    DIM_ALPHA,
    K_CLUSTERS,
    MIN_CLASS_CELLS,
    TAU,
    cells_to_wkb_polygons,
    duckdb,
    np,
    pa,
):
    # ---- a FRAME: scores, clusters, hexagons and colours for one folded view ------
    _PAL = np.array([tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in CLUSTER_HEX], np.uint8)
    con = duckdb.connect()

    def _hex_table(cells, xy):
        n = xy.shape[0]
        coords = pa.FixedSizeListArray.from_arrays(pa.array(xy.ravel()), 2)
        rings = pa.ListArray.from_arrays(pa.array(np.arange(0, 7 * n + 1, 7, dtype=np.int32)), coords)
        polys = pa.ListArray.from_arrays(pa.array(np.arange(0, n + 1, dtype=np.int32)), rings)
        geom = pa.field("geometry", polys.type, metadata={"ARROW:extension:name": "geoarrow.polygon", "ARROW:extension:metadata": '{"crs": "OGC:CRS84"}'})
        return pa.Table.from_arrays(
            [polys, cells["cell"]], schema=pa.schema([geom, cells.schema.field("cell")])
        )

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
        blobs = cells_to_wkb_polygons(cells["cell"].combine_chunks()).to_pylist()
        ok = np.array([len(b) == 125 for b in blobs])
        if not ok.all():
            cells = cells.filter(pa.array(ok))
            blobs = [b for b, o in zip(blobs, ok) if o]
            agree, cls, clu, hom = agree[ok], cls[ok], clu[ok], hom[ok]
        raw = np.frombuffer(b"".join(blobs), dtype=np.uint8).reshape(-1, 125)
        xy = np.ascontiguousarray(raw[:, 13:]).view("<f8").reshape(-1, 7, 2)
        ctr = xy[:, :6].mean(1, keepdims=True)
        cov = np.where(np.isnan(agree), 1.0, COV_MIN + (1 - COV_MIN) * np.clip(agree, 0, 1))
        geo = _hex_table(cells, ctr + (xy - ctr) * cov[:, None, None])
        geo_full = _hex_table(cells, xy)
        lap("hex")

        rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in cls], np.uint8)
        alpha_agree = np.where(
            np.isnan(agree), ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.clip(agree, 0, 1)
        ).astype(np.uint8)
        rgb_clu = _PAL[clu % len(_PAL)]
        cellid = cells["cell"].to_numpy()

        def fill(paint, sel, hit=None):
            if paint == "clusters":
                c, key = rgb_clu, 100 + clu
            else:
                c, key = rgb, cls
            a = alpha_agree if paint == "agreement" else np.full(len(cls), ALPHA_MAX, np.uint8)
            if sel:
                a = np.where(np.isin(key, list(sel)), a, DIM_ALPHA).astype(np.uint8)
            rgba = np.concatenate([c, a[:, None]], axis=1)
            if hit is not None:
                rgba[cellid == hit] = (255, 255, 255, 255)
            return pa.FixedSizeListArray.from_arrays(pa.array(rgba.ravel()), 4)

        lap("colours")
        a_ok = agree[~np.isnan(agree)]
        score = (
            f"{n:,} cells · agreement p50 {np.median(a_ok):.2f} · {(a_ok < 0.5).mean() * 100:.0f}% below 0.5"
            if len(a_ok) else f"{n:,} cells · NLCD only"
        ) + " (" + " ".join(f"{k} {v:.1f}" for k, v in _lap.items()) + ")"
        return {
            "cells": cells, "geo": geo, "geo_full": geo_full, "fill": fill,
            "cls": cls, "clu": clu, "agree": agree, "has_aef": has_aef, "score": score,
        }

    def legend_for(frame, paint):
        cls, clu, agree = frame["cls"], frame["clu"], frame["agree"]
        tot = max(1, len(cls))
        items = []
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

    return build_frame, con, legend_for


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """The strip under the map (the cdl-ftw-zarr-marimo HudControls skeleton,
        trimmed): paint buttons, pickable legend, panel, status; the map click through
        `ctl` (canvas pixel + rect); the one element docks into fullscreen."""

        ctl = traitlets.Unicode("").tag(sync=True)
        dres = traitlets.Unicode("0").tag(sync=True)  # kernel -> browser: the offset in force
        status = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.2rem 0 0;" +
            "user-select:none";
          const btnCss =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.1rem .45rem;border-radius:4px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          let paint = "agreement";
          const sel = new Set();
          let seq = 0;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, paint: paint, sel: Array.from(sel), n: ++seq }, extra || {})));
            model.save_changes();
          };
          const paintBox = document.createElement("span");
          paintBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const pl = document.createElement("span");
          pl.textContent = "paint";
          const mkPaint = (key, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => { paint = key; sel.clear(); stylePaint(); send("set"); renderLegend(); };
            return [key, b];
          };
          const paintBtns = [
            mkPaint("agreement", "agreement", "alpha and hexagon size follow agreement"),
            mkPaint("nlcd", "NLCD", "regular hexagons, NLCD's colours, no fade"),
            mkPaint("clusters", "AlphaEarth", "the embedding on its own: k-means clusters of the cell vectors"),
          ];
          const stylePaint = () => {
            paintBtns.forEach(([k, b]) => {
              b.style.borderColor = k === paint ? "#2b6cb0" : "rgba(127,127,127,.45)";
              b.style.fontWeight = k === paint ? "600" : "400";
            });
          };
          stylePaint();
          paintBox.append(pl, ...paintBtns.map(([, b]) => b));
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
            "gap:.1rem .55rem;flex:1;min-width:14rem";
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
          box.append(paintBox, resBox, legendBox);
          const panel = document.createElement("div");
          panel.style.cssText = "font:13px ui-sans-serif,system-ui,sans-serif;padding:.15rem 0";
          const status = document.createElement("div");
          status.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em;white-space:pre-line";
          const wrap = document.createElement("div");
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
                "background:rgba(255,255,255,.94);color:#111;" +
                "padding:.5rem 1.2rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "";
              el.appendChild(wrap);
            }
          };
          document.addEventListener("fullscreenchange", onFs);
          let downAt = null;
          const onDown = (e) => { downAt = [e.clientX, e.clientY]; };
          const onClick = (e) => {
            if (wrap.dataset.dead || !el.isConnected) return;
            const path = e.composedPath ? e.composedPath() : [e.target];
            const cv = path.find((n) => n && n.tagName === "CANVAS");
            if (!cv) return;
            if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
            const r = cv.getBoundingClientRect();
            if (!r.width || !r.height) return;
            send("click", { px: e.clientX - r.left, py: e.clientY - r.top, w: r.width, h: r.height });
          };
          document.addEventListener("pointerdown", onDown, true);
          document.addEventListener("click", onClick, true);
          const paintS = () => { status.textContent = model.get("status") || ""; };
          model.on("change:status", paintS);
          paintS();
          const paintP = () => { panel.innerHTML = model.get("panel") || ""; };
          model.on("change:panel", paintP);
          paintP();
          const hideBbox = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("button[aria-label]").forEach((b) => {
              const a = b.getAttribute("aria-label");
              if (a === "Select BBox" || a === "Cancel drawing" || a === "Clear bounding box") {
                const holder = b.closest("div[style*='absolute']") || b;
                holder.style.display = "none";
              }
            });
            root.querySelectorAll("*").forEach((n) => { if (n.shadowRoot) hideBbox(n.shadowRoot); });
          };
          const bboxTimer = setInterval(() => hideBbox(document), 1000);
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            document.removeEventListener("pointerdown", onDown, true);
            document.removeEventListener("click", onClick, true);
            clearInterval(bboxTimer);
            wrap.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(CartoStyle, HOME, Map, MaplibreBasemap, PolygonLayer, np, pa):
    # ---- the map: built ONCE on a placeholder; never re-runs for a parameter -----
    _xy = np.array([[[-96.001, 38.501], [-95.999, 38.501], [-95.999, 38.499],
                     [-96.001, 38.499], [-96.001, 38.501]]])
    _coords = pa.FixedSizeListArray.from_arrays(pa.array(_xy.ravel()), 2)
    _rings = pa.ListArray.from_arrays(pa.array([0, 5], pa.int32()), _coords)
    _polys = pa.ListArray.from_arrays(pa.array([0, 1], pa.int32()), _rings)
    _geom = pa.field("geometry", _polys.type, metadata={"ARROW:extension:name": "geoarrow.polygon", "ARROW:extension:metadata": '{"crs": "OGC:CRS84"}'})
    layer = PolygonLayer(
        table=pa.Table.from_arrays([_polys], schema=pa.schema([_geom])),
        get_fill_color=pa.FixedSizeListArray.from_arrays(pa.array([0, 0, 0, 0], pa.uint8()), 4),
        filled=True,
        stroked=False,
        pickable=False,
    )
    deck = Map(
        layers=[layer],
        basemap=MaplibreBasemap(style=CartoStyle.Positron),
        view_state=dict(HOME),
        height=720,
    )
    HOLD = {
        "frame": None, "geo_sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "task": None, "loop": None,
        "paint": "agreement", "sel": set(), "hit": None, "memo": {}, "h_cam": None, "h_ctl": None,
        "dres": 0,  # the strip's res offset; a statement about the box it was set on
    }
    deck  # the cell's LAST statement: what marimo displays
    return HOLD, deck, layer


@app.cell
def _(HudControls, mo):
    hud = mo.ui.anywidget(HudControls())
    hud
    return (hud,)


@app.cell
def _(
    ArrowTable,
    HOLD,
    HOME,
    SETTLE,
    aef_fold,
    asyncio,
    build_frame,
    con,
    contains,
    coordinates_to_cells,
    deck,
    hud,
    json,
    layer,
    legend_for,
    math,
    nlcd_fold,
    np,
    pad_box,
    res_for_view,
    time,
    view_to_bbox,
):
    # ---- wiring: the camera loop and the strip. Re-runs freely (un-observes its
    # old handlers first); the map cell never re-runs.
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _say_dres():
        try:
            hud.widget.dres = str(HOLD["dres"])
        except Exception:
            pass

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, dict):
            return {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        return {"longitude": float(vs.longitude), "latitude": float(vs.latitude), "zoom": float(vs.zoom)}

    def _paint():
        """Hand the current frame's hexagons + colours to the one layer."""
        fr = HOLD["frame"]
        if fr is None:
            return
        geo = fr["geo"] if HOLD["paint"] == "agreement" else fr["geo_full"]
        with layer.hold_sync():
            if HOLD["geo_sent"] is not geo:
                layer._rows_per_chunk = max(1, geo.num_rows)
                layer.table = ArrowTable.from_arrow(geo)
                HOLD["geo_sent"] = geo
            layer.get_fill_color = fr["fill"](HOLD["paint"], HOLD["sel"], HOLD["hit"])
        try:
            hud.widget.legend = json.dumps(legend_for(fr, HOLD["paint"]))
        except Exception:
            pass

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
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
        _paint()
        HOLD["last_status"] = f"{stats} · {fr['score']} · {time.time() - t0:.1f} s"
        _say(HOLD["last_status"])

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
        HOLD["vs"] = vs
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            deck.unobserve(HOLD["h_cam"], names="view_state")
        except ValueError:
            pass
    deck.observe(_on_camera, names="view_state")
    HOLD["h_cam"] = _on_camera

    def _unproject(vs, px, py, w, h):
        world = 512 * 2 ** vs["zoom"]
        lon = vs["longitude"] + (px - w / 2) * 360.0 / world
        lat0 = math.radians(vs["latitude"])
        uy = (1 - math.log(math.tan(lat0) + 1 / math.cos(lat0)) / math.pi) / 2
        uy = uy + (py - h / 2) / world
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * uy))))
        return lon, lat

    def _on_ctl(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        HOLD["paint"] = c.get("paint", "agreement")
        HOLD["sel"] = {int(x) for x in c.get("sel", [])}
        fr = HOLD["frame"]
        if c.get("act") == "dres":
            HOLD["dres"] = max(-2, min(2, int(c.get("dres", 0))))
            _say_dres()
            vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
            if HOLD["busy"]:
                HOLD["pending"] = vs
            else:
                HOLD["task"] = _spawn(_serve(vs, force=True))
            return
        if c.get("act") == "click" and fr is not None:
            try:
                vs = _vsd(HOLD["vs"] if HOLD["vs"] is not None else deck.view_state)
                lon, lat = _unproject(vs, float(c["px"]), float(c["py"]), float(c["w"]), float(c["h"]))
                cell = int(coordinates_to_cells(np.array([lat]), np.array([lon]), HOLD["res"])[0].as_py())
                con.register("cur_cells", fr["cells"])
                r = con.execute(
                    "SELECT name, agree, alt_name, purity, homogeneity, cluster FROM cur_cells WHERE cell = ?", [cell]
                ).fetchone()
                if r is None:
                    HOLD["hit"] = None
                    hud.widget.panel = f"<span style='opacity:.7'>({lat:.4f}, {lon:.4f}): no cell here</span>"
                else:
                    HOLD["hit"] = cell if HOLD["hit"] != cell else None
                    nm, ag, alt, pur, hom, ck = r
                    scored = ag is not None and not np.isnan(ag)
                    hud.widget.panel = (
                        f"<b>{nm}</b> at {lat:.4f}, {lon:.4f}: cluster {ck}, agreement "
                        + (f"{ag:.2f}" if scored else "unscored")
                        + (f", looks more like <i>{alt}</i>" if scored and ag < 0.5 and alt != "none" else "")
                        + f", NLCD purity {pur:.2f}"
                        + (f", homogeneity {hom:.3f}" if hom is not None and not np.isnan(hom) else "")
                    )
            except Exception as e:
                hud.widget.panel = f"<span style='opacity:.7'>click: {e}</span>"
        elif fr is not None and HOLD["sel"]:
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
                word = "usually looks like"
            hud.widget.panel = " · ".join(
                f"<b>{nm}</b>: {cnt:,} cells"
                + (f", agreement p50 {p50:.2f}, {pct:.0f}% below 0.5" if p50 is not None else "")
                + (f", {word} <i>{alt}</i>" if alt else "")
                for nm, cnt, p50, pct, alt in rows
            )
        else:
            hud.widget.panel = ""
        _paint()

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
               mode(alt_name) FILTER (WHERE agree < 0.5) AS usual_alternative
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
