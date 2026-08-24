# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo==0.24.0",
#     "datafusion>=54.0.0",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "arro3-core",
#     "numpy",
#     "duckdb>=1.5.5",
# ]
# ///
"""Annual NLCD, coloured as it is, faded and shrunk where AlphaEarth does not back it.

One box, one year, no camera fold (the xsql-deforest-conus-counties.py chassis with
a zoomable map on top). Three reads, two folds, one score:

  NLCD 2024 (30 m, EPSG:5070, the source.coop mirror of USGS's Annual NLCD)
      -> H3 res RES, majority class per cell        (DataFusion, the h3 UDF in the SQL)
  AlphaEarth Foundations 2024 (10 m, EPSG:4326, tge-labs/aef-mosaic, 64 int8 bands)
      -> H3 res RES, mean vector per cell           (same UDF, 64 avg() columns)
  join on cell -> per NLCD class present in the box, a PROTOTYPE (the mean of its
  cells' vectors: what "evergreen forest" looks like to the satellites around here);
  per cell, the cosine to every prototype, and AGREEMENT = sigmoid of (own-class cosine
  minus the best other class's cosine) at temperature TAU. 1 = the embedding is sure
  this is what NLCD says; 0.5 = equidistant from its own class and the nearest other;
  low = the cell looks like the other class's ground.

The map is NLCD in NLCD's own colours. Agreement is drawn as ALPHA and as CELL
COVERAGE (each hexagon scaled about its centre): agreeing ground is solid and full,
disagreeing ground is faint and small, so the basemap shows through where the two
datasets tell different stories. Not extruded. The strip under the map (the
cdl-ftw-zarr-marimo HudControls skeleton) has a fade checkbox and a pickable legend
(click a class to isolate it); clicking a hexagon opens lonboard's feature panel
with the class, its agreement and the class it looks more like. Basemap DarkMatter
(Stephen: low-opacity colours read better on dark).

The score is CONSISTENCY, not correctness: the labels are NLCD's own, so a class
mislabelled the same way everywhere looks perfectly typical. Prototypes are local to
the box; classes with fewer than MIN_CLASS_CELLS cells get no prototype and their
cells are drawn plain (no judgment) and listed as such.

AlphaEarth on this mirror runs 2017-2025; NLCD on its mirror ends at 2024. YEAR_AEF
2025 against YEAR_NLCD 2024 is allowed and then the score also carries a year of
change. Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind." (CC-BY 4.0.)

Run: uv run marimo edit xsql-aef-nlcd-agreement.py   (or --sandbox)
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import json
    import math
    import time

    import numpy as np
    import pyarrow as pa
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
        Window,
        XarrayContext,
        anywidget,
        cells_to_wkb_polygons,
        coordinates_to_cells,
        duckdb,
        json,
        math,
        mo,
        np,
        pa,
        time,
        traitlets,
        udf,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # NLCD, backed or not by AlphaEarth

    **What you are looking at.** Annual NLCD land cover, in its own colours, for one
    box. Every hexagon also carries the mean AlphaEarth embedding of the 10 m pixels
    inside it (64 numbers summarising that ground's whole year as the satellites saw
    it). For each NLCD class in the box, the mean of its cells' vectors is the class's
    *prototype*: what "shrub" looks like to the satellites around here. A cell's
    **agreement** is how clearly it sits closer to its own class's prototype than to the
    nearest other one (a sigmoid over the cosine margin).

    - **Solid, full-size hexagon**: the embedding backs NLCD's word here.
    - **Faint, shrunken hexagon**: the cell looks more like some other class's ground,
      or like nothing in particular. The basemap shows through. These are the leads:
      NLCD's word is thin, and the table under the map says which word the embedding
      would have used instead.

    This is a consistency check, not a truth check: the labels are NLCD's own. What it
    finds is where NLCD's 16 words are stretched over ground that does not look like
    the rest of that word's ground in this box.

    | leg | data | engine |
    |---|---|---|
    | land cover | Annual NLCD 2024, 30 m, EPSG:5070 (`kylebarron/usgs-landcover` mirror, COG) | obstore + async-geotiff read, DataFusion fold (h3 UDF) |
    | embeddings | AlphaEarth Foundations 2024, 10 m, EPSG:4326 (`tge-labs/aef-mosaic`, Zarr v3, 64 x int8) | obstore + zarr read, DataFusion fold (h3 UDF, 64 `avg()`) |
    | score | join on cell, prototypes, cosines, softmax | numpy; tables in DuckDB |
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    # The test box: the Sierra foothills around Folsom Lake / Auburn, CA. Water,
    # four developed classes, shrub, three forests, herbaceous, pasture and the
    # wetlands all present; the shrub/herbaceous/forest confusions NLCD's own
    # accuracy work reports are the ones to look for. ~22 x 28 km.
    BOX = (-121.25, 38.70, -121.00, 38.95)  # W, S, E, N
    # (The opening camera is a literal inside the map cell, so nothing here can
    # re-run it.)

    # H3 resolution of both folds. Res 10 (~1.5 ha, ~17 NLCD px, ~150 AEF px per cell)
    # is where a 30 m majority is a real vote; res 11 (~2.4 NLCD px) turns the NLCD
    # side into a relabel and multiplies cells by 7.
    RES = 10

    YEAR_NLCD = 2024
    YEAR_AEF = 2024  # 2017-2025; 2025 against NLCD 2024 adds a year of change

    # Temperature of the own-vs-runner-up sigmoid over the cosine margin. Prototypes
    # in one box sit ~0.90-0.96 apart and a clearly agreeing cell's own cosine is
    # ~0.03 above its runner-up, so 0.02 puts it near 0.8 and a clear flip near 0.2.
    TAU = 0.02
    # A class needs this many cells before its mean is a prototype.
    MIN_CLASS_CELLS = 30

    # The agreement -> paint mapping. Alpha and coverage both run from their floor at
    # agreement 0 to full at agreement 1.
    ALPHA_MIN, ALPHA_MAX = 30, 235
    COV_MIN = 0.30
    DIM_ALPHA = 22  # cells outside the picked class

    NLCD_PREFIX = "kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic"
    NLCD_NODATA = 250
    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128

    # NLCD's own colormap (read out of the COG in the archived notebook). Kept as
    # is at Stephen's call ("color the way it is"); 23/24 are the official reds.
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
        AEF_NODATA,
        AEF_PREFIX,
        AEF_RES,
        AEF_X0,
        AEF_Y0,
        ALPHA_MAX,
        ALPHA_MIN,
        BOX,
        CLASSES,
        COV_MIN,
        DIM_ALPHA,
        MIN_CLASS_CELLS,
        NLCD_NODATA,
        NLCD_PREFIX,
        RES,
        TAU,
        YEAR_AEF,
        YEAR_NLCD,
    )


@app.cell
def _(math, np):
    # ---- EPSG:5070 (Albers equal area, GRS80) both ways, closed form, no pyproj ----
    # Snyder 14-1..14-11 forward; 14-19..14-21 inverse with the fixed-point phi
    # iteration. Verified against pyproj: inverse within 3e-10 degrees over the box.
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
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule): h3_latlng_to_cell(lat, lon,
    # res) in the GROUP BY of both folds below. One context holds both windows.
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
    BOX,
    GeoTIFF,
    NLCD_NODATA,
    NLCD_PREFIX,
    RES,
    S3Store,
    Window,
    YEAR_NLCD,
    albers_fwd,
    albers_inv,
    ctx,
    math,
    np,
    time,
    xr,
):
    # ---- NLCD: read the box at native 30 m, fold to the majority class per cell ----
    _t0 = time.time()
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _g = await GeoTIFF.open(
        f"{NLCD_PREFIX}/Annual_NLCD_LndCov_{YEAR_NLCD}_CU_C1V1.tif", store=_store
    )
    _L, _B, _R, _T = _g.bounds
    _px = _g.res[0]

    # The Albers window is a DENSIFIED boundary transform (repo rule: a parallel bows
    # in 5070, so a 4-corner min/max clips the box's south or north edge).
    _W, _S, _E, _N = BOX
    _lons = np.concatenate(
        [np.linspace(_W, _E, 9), np.full(9, _E), np.linspace(_E, _W, 9), np.full(9, _W)]
    )
    _lats = np.concatenate(
        [np.full(9, _N), np.linspace(_N, _S, 9), np.full(9, _S), np.linspace(_S, _N, 9)]
    )
    _ax, _ay = albers_fwd(_lons, _lats)
    _c0, _c1 = int((_ax.min() - _L) / _px), int(math.ceil((_ax.max() - _L) / _px))
    _r0, _r1 = int((_T - _ay.max()) / _px), int(math.ceil((_T - _ay.min()) / _px))

    _ra = await _g.read(
        window=Window(col_off=_c0, row_off=_r0, width=_c1 - _c0, height=_r1 - _r0)
    )
    _arr = np.asarray(np.ma.filled(_ra.as_masked(), NLCD_NODATA)).reshape(
        _r1 - _r0, _c1 - _c0
    )
    _t_read = time.time() - _t0

    # Every pixel centre's lat/lon, exactly, as two 2-D data variables (an Albers grid
    # has no 1-D lat/lon coordinate), so the UDF reads them as columns.
    _xs = _L + (np.arange(_c0, _c1) + 0.5) * _px
    _ys = _T - (np.arange(_r0, _r1) + 0.5) * _px
    _X, _Y = np.meshgrid(_xs, _ys)
    _lon, _lat = albers_inv(_X, _Y)

    ctx.from_dataset(
        "lc",
        xr.Dataset(
            {"cls": (("y", "x"), _arr), "lat": (("y", "x"), _lat), "lon": (("y", "x"), _lon)},
            coords={"y": _ys, "x": _xs},
        ),
        chunks={"y": 512},
    )
    _t1 = time.time()
    nlcd_cells = ctx.sql(f"""
        WITH c AS (
            SELECT h3_latlng_to_cell(lat, lon, CAST({RES} AS INT)) AS cell, cls, count(*) AS n
            FROM lc
            WHERE cls != {NLCD_NODATA}
              AND lon >= {_W} AND lon < {_E} AND lat >= {_S} AND lat < {_N}
            GROUP BY 1, 2
        )
        SELECT cell,
               first_value(cls ORDER BY n DESC, cls ASC) AS cls,
               sum(n) AS npx,
               CAST(max(n) AS DOUBLE) / sum(n) AS purity
        FROM c GROUP BY cell
    """).to_arrow_table()
    nlcd_stats = (
        f"NLCD {YEAR_NLCD}: {_arr.shape[1]:,} x {_arr.shape[0]:,} px read {_t_read:.1f} s · "
        f"fold res {RES} {nlcd_cells.num_rows:,} cells {time.time() - _t1:.1f} s"
    )
    print(nlcd_stats)
    return nlcd_cells, nlcd_stats


@app.cell
def _(
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_X0,
    AEF_Y0,
    BOX,
    ObjectStore,
    RES,
    S3Store,
    YEAR_AEF,
    ctx,
    np,
    time,
    xr,
):
    # ---- AlphaEarth: read the box (one year, 64 bands), fold to the mean vector ----
    # The mosaic is one sharded Zarr v3 array (time, band, y, x) on a 4326 grid with
    # no pyramid: a box is always a native 10 m read (~64 B per pixel raw, zstd on
    # the wire). Dequantize int8 -> float per the store's own formula, then 64
    # variables e00..e63 so the fold is one row per pixel and 64 avg() columns.
    _t0 = time.time()
    _store = S3Store(
        "us-west-2.opendata.source.coop",
        region="us-west-2",
        skip_signature=True,
        prefix=AEF_PREFIX,
    )
    _ds = xr.open_zarr(ObjectStore(_store, read_only=True), chunks=None, consolidated=False)
    _ti = int(np.where(_ds.time.values == YEAR_AEF)[0][0])
    _W, _S, _E, _N = BOX
    _x0, _x1 = int((_W - AEF_X0) / AEF_RES), int((_E - AEF_X0) / AEF_RES)
    _y0, _y1 = int((AEF_Y0 - _N) / AEF_RES), int((AEF_Y0 - _S) / AEF_RES)
    _emb = _ds.embeddings.isel(time=_ti, y=slice(_y0, _y1), x=slice(_x0, _x1)).values
    _t_read = time.time() - _t0

    _f = (np.sign(_emb) * (_emb / 127.5) ** 2).astype(np.float32)
    _f[:, _emb[0] == AEF_NODATA] = np.nan
    _lat = AEF_Y0 - (np.arange(_y0, _y1) + 0.5) * AEF_RES
    _lon = AEF_X0 + (np.arange(_x0, _x1) + 0.5) * AEF_RES
    ctx.from_dataset(
        "aef",
        xr.Dataset(
            {f"e{i:02d}": (("y", "x"), _f[i]) for i in range(64)},
            coords={"y": _lat, "x": _lon},
        ),
        chunks={"y": 256},
    )
    _t1 = time.time()
    _cols = ", ".join(f"avg(e{i:02d}) AS e{i:02d}" for i in range(64))
    aef_cells = ctx.sql(f"""
        SELECT h3_latlng_to_cell(y, x, CAST({RES} AS INT)) AS cell, count(*) AS naef, {_cols}
        FROM aef WHERE e00 IS NOT NULL
        GROUP BY cell
    """).to_arrow_table()
    aef_stats = (
        f"AlphaEarth {YEAR_AEF}: {_emb.shape[2]:,} x {_emb.shape[1]:,} px x 64 read {_t_read:.1f} s "
        f"({_emb.nbytes / 1e6:.0f} MB raw) · fold res {RES} {aef_cells.num_rows:,} cells "
        f"{time.time() - _t1:.1f} s"
    )
    print(aef_stats)
    return aef_cells, aef_stats


@app.cell
def _(
    CLASSES,
    MIN_CLASS_CELLS,
    TAU,
    aef_cells,
    duckdb,
    nlcd_cells,
    np,
    pa,
    time,
):
    # ---- join, prototypes, agreement ------------------------------------------
    _t0 = time.time()
    con = duckdb.connect()
    # con.register, not the replacement scan: marimo mangles cell locals (repo lesson).
    con.register("nlcd_cells", nlcd_cells)
    con.register("aef_cells", aef_cells)
    _j = con.execute(
        "SELECT * FROM nlcd_cells JOIN aef_cells USING (cell) ORDER BY cell"
    ).arrow().read_all()

    _V = np.stack([_j[f"e{i:02d}"].to_numpy() for i in range(64)], axis=1)
    hom = np.linalg.norm(_V, axis=1)  # length of the mean of unit vectors: homogeneity
    _V = _V / hom[:, None]
    cls = _j["cls"].to_numpy().astype(np.int64)

    _present, _counts = np.unique(cls, return_counts=True)
    proto_classes = _present[_counts >= MIN_CLASS_CELLS]
    protos = np.stack([_V[cls == c].mean(0) for c in proto_classes])
    protos = protos / np.linalg.norm(protos, axis=1)[:, None]

    # cells x prototypes cosines. Agreement is the two-way contest between the cell's
    # own prototype and the best OTHER prototype: sigmoid of the cosine margin at
    # temperature TAU. Exactly 0.5 where the cell is equidistant, independent of how
    # many classes the box holds (a full softmax would dilute every cell by the
    # class count). alt_p is the other side of the same contest.
    cos = _V @ protos.T
    _idx = np.searchsorted(proto_classes, cls)
    _has = np.isin(cls, proto_classes)
    _idx = np.where(_has, _idx, 0)
    _rows = np.arange(len(cls))
    own_cos = np.where(_has, cos[_rows, _idx], np.nan)
    _cos_other = cos.copy()
    _cos_other[_rows, _idx] = -np.inf
    _alt_i = _cos_other.argmax(1)
    alt = np.where(_has, proto_classes[_alt_i], -1)
    _margin = own_cos - _cos_other[_rows, _alt_i]
    agree = np.where(_has, 1.0 / (1.0 + np.exp(-_margin / TAU)), np.nan)
    alt_p = 1.0 - agree

    cells = pa.table(
        {
            "cell": _j["cell"],
            "cls": pa.array(cls.astype(np.uint8)),
            "name": pa.array([CLASSES.get(int(c), ("?",))[0] for c in cls]),
            "npx": _j["npx"],
            "purity": _j["purity"],
            "naef": _j["naef"],
            "homogeneity": pa.array(hom.astype(np.float32)),
            "own_cos": pa.array(own_cos.astype(np.float32)),
            "agree": pa.array(agree.astype(np.float32)),
            "alt": pa.array(alt.astype(np.int16)),
            "alt_name": pa.array([CLASSES.get(int(c), ("none",))[0] for c in alt]),
            "alt_p": pa.array(alt_p.astype(np.float32)),
        }
    )
    con.register("cells", cells)

    # The prototype-vs-prototype cosine matrix, long form, for the SQL side.
    _pp = protos @ protos.T
    proto_pairs = pa.table(
        {
            "a": pa.array(np.repeat(proto_classes, len(proto_classes)).astype(np.uint8)),
            "b": pa.array(np.tile(proto_classes, len(proto_classes)).astype(np.uint8)),
            "cos": pa.array(_pp.ravel().astype(np.float32)),
        }
    )
    con.register("proto_pairs", proto_pairs)
    con.register(
        "class_names",
        pa.table(
            {
                "cls": pa.array(list(CLASSES), pa.uint8()),
                "name": pa.array([v[0] for v in CLASSES.values()]),
            }
        ),
    )

    _a = agree[_has]
    score_stats = (
        f"join {cells.num_rows:,} cells · {len(proto_classes)} prototypes "
        f"(classes with >= {MIN_CLASS_CELLS} cells; {(~_has).sum():,} cells of rarer classes unscored) · "
        f"agreement p10/p50/p90 {np.percentile(_a, 10):.2f}/{np.percentile(_a, 50):.2f}/"
        f"{np.percentile(_a, 90):.2f} · {(_a < 0.5).mean() * 100:.0f}% below 0.5 · "
        f"{time.time() - _t0:.1f} s"
    )
    print(score_stats)
    return cells, con, score_stats


@app.cell
def _(COV_MIN, cells, cells_to_wkb_polygons, np, pa):
    # ---- hexagons, each scaled about its centre by its agreement (coverage) --------
    # h3ronpy gives WKB; a hexagon's WKB is fixed-size (1 ring, 7 points, 125 bytes),
    # so the whole array parses with one frombuffer. Pentagons (109 bytes) do not
    # occur over land at res 10 in CONUS; assert rather than handle.
    _wkb = cells_to_wkb_polygons(cells["cell"].combine_chunks())
    _blobs = _wkb.to_pylist()
    assert all(len(b) == 125 for b in _blobs), "unexpected WKB layout (pentagon or multi-ring)"
    _raw = np.frombuffer(b"".join(_blobs), dtype=np.uint8).reshape(-1, 125)
    _xy = np.ascontiguousarray(_raw[:, 13:]).view("<f8").reshape(-1, 7, 2)  # lon, lat
    _ctr = _xy[:, :6].mean(1, keepdims=True)

    _agree = cells["agree"].to_numpy(zero_copy_only=False)
    _cov = np.where(np.isnan(_agree), 1.0, COV_MIN + (1 - COV_MIN) * np.clip(_agree, 0, 1))
    hex_xy = _ctr + (_xy - _ctr) * _cov[:, None, None]

    def hex_table(xy):
        """A geoarrow.polygon column (interleaved coords) over the cells table."""
        n = xy.shape[0]
        coords = pa.FixedSizeListArray.from_arrays(pa.array(xy.ravel()), 2)
        rings = pa.ListArray.from_arrays(pa.array(np.arange(0, 7 * n + 1, 7, dtype=np.int32)), coords)
        polys = pa.ListArray.from_arrays(pa.array(np.arange(0, n + 1, dtype=np.int32)), rings)
        geom = pa.field(
            "geometry", polys.type, metadata={"ARROW:extension:name": "geoarrow.polygon"}
        )
        # The other columns are what lonboard's feature panel shows on a click
        # (pickable=True, Stephen: "we can't just rely on the legend"): names, not
        # codes, and the numbers rounded.
        def _r(col, d):
            return pa.array(np.round(cells[col].to_numpy(zero_copy_only=False).astype(float), d))
        props = {
            "class": cells["name"],
            "agreement": _r("agree", 2),
            "looks more like": cells["alt_name"],
            "NLCD purity": _r("purity", 2),
            "homogeneity": _r("homogeneity", 3),
            "cell": cells["cell"],
        }
        arrays = [polys, *props.values()]
        fields = [geom, *(pa.field(k, v.type) for k, v in props.items())]
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields))

    geo = hex_table(hex_xy)
    return (geo,)


@app.cell
def _(ALPHA_MAX, ALPHA_MIN, CLASSES, DIM_ALPHA, cells, np, pa):
    # ---- the fill colours: NLCD's rgb, agreement as alpha --------------------------
    _cls = cells["cls"].to_numpy()
    _rgb = np.array([CLASSES.get(int(c), ("?", (128, 128, 128)))[1] for c in _cls], np.uint8)
    _agree = cells["agree"].to_numpy(zero_copy_only=False)
    _alpha_agree = np.where(
        np.isnan(_agree), ALPHA_MAX, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.clip(_agree, 0, 1)
    ).astype(np.uint8)

    def fill_colors(fade, sel):
        """RGBA per cell: fade=True runs alpha by agreement, False is flat NLCD;
        sel (a set of class codes, empty = all) dims every other class to DIM_ALPHA."""
        a = _alpha_agree if fade else np.full(len(_cls), ALPHA_MAX, np.uint8)
        if sel:
            a = np.where(np.isin(_cls, list(sel)), a, DIM_ALPHA).astype(np.uint8)
        rgba = np.concatenate([_rgb, a[:, None]], axis=1)
        return pa.FixedSizeListArray.from_arrays(pa.array(rgba.ravel()), 4)

    return (fill_colors,)


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """Controls + legend + status UNDER the map: the strip from
        ~/dev/projects/cdl-ftw-zarr-marimo (cdl-ftw.py / aef-similarity.py /
        aef-agreement.py keep one skeleton in sync; this is that skeleton trimmed
        to this notebook). One row: a fade checkbox, the pickable legend (click a
        class to isolate it, multi-select toggles, "× all" resets); a panel line
        (the isolated classes' numbers); a status line (the reads and the score).
        Proven trait types only: `ctl` Unicode JSON browser -> kernel, `status` /
        `legend` / `panel` Unicode kernel -> browser. The ONE strip element is
        re-parented into the fullscreen element as a bottom bar and back, so the
        strip under the map and the strip in fullscreen are the same element with
        the same state. Hides lonboard's bbox toolbar."""

        ctl = traitlets.Unicode("").tag(sync=True)
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
          const mk = (text, on, title) => {
            const l = document.createElement("label");
            l.style.cssText =
              "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
            const i = document.createElement("input");
            i.type = "checkbox"; i.checked = on;
            l.appendChild(i); l.appendChild(document.createTextNode(text));
            if (title) l.title = title;
            return [l, i];
          };
          const [labF, fade] = mk("fade by agreement", true,
            "alpha follows agreement; off = plain NLCD (coverage stays)");
          // the legend: classes in the box, pickable (click isolates, × all resets)
          const sel = new Set();
          const legendBox = document.createElement("div");
          legendBox.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;" +
            "gap:.1rem .55rem;flex:1;min-width:14rem";
          let seq = 0, deb = null;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, fade: fade.checked, sel: Array.from(sel),
              n: ++seq }, extra || {})));
            model.save_changes();
          };
          const commit = () => {
            clearTimeout(deb);
            deb = setTimeout(() => send("set"), 250);
          };
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
                (on ? "#2b6cb0" : "transparent") +
                (on ? ";font-weight:600" : "");
              b.title = it.pct + "% of cells · agreement p50 " + it.p50;
              b.innerHTML =
                '<span style="width:10px;height:10px;border-radius:2px;' +
                "background:" + it.hex + ';display:inline-block"></span>' +
                it.name + (it.note ? ' <span style="opacity:.6">' + it.note + "</span>" : "");
              b.onclick = () => {
                if (sel.has(it.code)) sel.delete(it.code);
                else sel.add(it.code);
                send("set");
                renderLegend();
              };
              legendBox.appendChild(b);
            });
          };
          model.on("change:legend", renderLegend);
          renderLegend();
          box.append(labF, legendBox);
          const panel = document.createElement("div");
          panel.style.cssText =
            "font:13px ui-sans-serif,system-ui,sans-serif;padding:.15rem 0";
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
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) killOld(n.shadowRoot);
            });
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
              if (getComputedStyle(fe).position === "static")
                fe.style.position = "relative";
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
          fade.addEventListener("change", commit);
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
              if (a === "Select BBox" || a === "Cancel drawing" ||
                  a === "Clear bounding box") {
                const holder = b.closest("div[style*='absolute']") || b;
                holder.style.display = "none";
              }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) hideBbox(n.shadowRoot);
            });
          };
          const bboxTimer = setInterval(() => hideBbox(document), 1000);
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            clearInterval(bboxTimer);
            wrap.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(CartoStyle, Map, MaplibreBasemap, PolygonLayer, np, pa):
    # ---- the map: built ONCE, on a placeholder, and NEVER re-run -------------------
    # This cell depends on imports only. Every constant, the folds and the scores
    # reach the map through the wiring cell below (`layer.table`, `get_fill_color`),
    # so editing a parameter re-runs the wiring and never this cell. Destroying a
    # lonboard Map kills deck's earcut worker pool for the whole page (every polygon
    # layer after it fails to init until a reload): the repo's map/wiring split.
    _home = {"longitude": -121.125, "latitude": 38.825, "zoom": 10.8}
    _xy = np.array([[[-121.1251, 38.8251], [-121.1249, 38.8251], [-121.1249, 38.8249],
                     [-121.1251, 38.8249], [-121.1251, 38.8251]]])
    _coords = pa.FixedSizeListArray.from_arrays(pa.array(_xy.ravel()), 2)
    _rings = pa.ListArray.from_arrays(pa.array([0, 5], pa.int32()), _coords)
    _polys = pa.ListArray.from_arrays(pa.array([0, 1], pa.int32()), _rings)
    _geom = pa.field("geometry", _polys.type, metadata={"ARROW:extension:name": "geoarrow.polygon"})
    layer = PolygonLayer(
        table=pa.Table.from_arrays([_polys], schema=pa.schema([_geom])),
        get_fill_color=pa.FixedSizeListArray.from_arrays(pa.array([0, 0, 0, 0], pa.uint8()), 4),
        filled=True,
        stroked=False,
        pickable=True,
    )
    deck = Map(
        layers=[layer],
        basemap=MaplibreBasemap(style=CartoStyle.DarkMatter),
        view_state=_home,
        height=720,
    )
    HOLD = {"geo": None}  # which hexagon table the layer currently holds
    deck  # the cell's LAST statement: what marimo displays
    return HOLD, layer


@app.cell
def _(HudControls, mo):
    hud = mo.ui.anywidget(HudControls())
    hud
    return (hud,)


@app.cell
def _(
    ArrowTable,
    CLASSES,
    HOLD,
    aef_stats,
    cells,
    con,
    fill_colors,
    geo,
    hud,
    json,
    layer,
    nlcd_stats,
    np,
    score_stats,
):
    # ---- wiring: re-runs on every strip commit AND on every parameter edit -------
    # Hands the current hexagons and colours to the one layer (table + colours in
    # one sync so the browser never sees them mismatched) and pushes the strip's
    # legend / status / panel.
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _fade = bool(_c.get("fade", True))
    _sel = {int(x) for x in _c.get("sel", [])}
    with layer.hold_sync():
        if HOLD["geo"] is not geo:
            # The trait takes an arro3 Table on assignment (the constructor converts
            # pyarrow itself; assignment does not). And the constructor fixed
            # `_rows_per_chunk` from the 1-row placeholder: table AND every accessor
            # serialize in chunks of that size, so without resetting it a 39k-row
            # table crosses as 39k one-row chunks and draws nothing. One chunk
            # (the crops notebook's lesson: multi-chunk swaps striped whole bands).
            layer._rows_per_chunk = max(1, geo.num_rows)
            layer.table = ArrowTable.from_arrow(geo)
            HOLD["geo"] = geo
        layer.get_fill_color = fill_colors(_fade, _sel)

    _cls = cells["cls"].to_numpy()
    _ag = cells["agree"].to_numpy(zero_copy_only=False)
    _codes, _n = np.unique(_cls, return_counts=True)
    _tot = int(_n.sum()) or 1
    _legend = []
    for _code, _cnt in sorted(zip(_codes, _n), key=lambda t: -t[1]):
        if int(_code) not in CLASSES:
            continue
        _a = _ag[_cls == _code]
        _a = _a[~np.isnan(_a)]
        _legend.append({
            "code": int(_code),
            "name": CLASSES[int(_code)][0],
            "hex": "#%02x%02x%02x" % CLASSES[int(_code)][1],
            "pct": round(100 * int(_cnt) / _tot, 1),
            "p50": f"{np.median(_a):.2f}" if len(_a) else "none",
            "note": "" if len(_a) else "(unscored)",
        })
    hud.widget.legend = json.dumps(_legend)
    hud.widget.status = "\n".join([nlcd_stats, aef_stats, score_stats])

    if _sel:
        _rows = con.execute(
            """
            SELECT name, count(*) AS cells, round(median(agree), 2) AS p50,
                   round(100 * avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END), 0) AS pct_low,
                   mode(alt_name) FILTER (WHERE agree < 0.5) AS usual_alt
            FROM cells WHERE cls IN (SELECT UNNEST(?)) GROUP BY name ORDER BY cells DESC
            """,
            [list(_sel)],
        ).fetchall()
        hud.widget.panel = " · ".join(
            f"<b>{nm}</b>: {cnt:,} cells, agreement p50 {p50:.2f}, {pct:.0f}% below 0.5"
            + (f", usually looks like <i>{alt}</i>" if alt else "")
            for nm, cnt, p50, pct, alt in _rows
        )
    else:
        hud.widget.panel = ""
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    Every table below is DuckDB over the same `cells` table the map draws: one row
    per hexagon with `cls`, `name`, `purity` (NLCD majority share), `homogeneity`
    (length of the mean unit vector: 1 = the 10 m pixels agree with each other),
    `own_cos`, `agree`, `alt` / `alt_name` / `alt_p` (the best other class and its
    probability). `proto_pairs` holds the prototype-vs-prototype cosines.
    """)
    return


@app.cell
def _(con, mo):
    per_class = mo.sql(
        f"""
        SELECT cls, name,
               count(*)                                   AS cells,
               round(median(agree), 3)                    AS agree_p50,
               round(quantile_cont(agree, 0.10), 3)       AS agree_p10,
               round(avg(CASE WHEN agree < 0.5 THEN 1 ELSE 0 END) * 100, 1) AS pct_below_half,
               round(median(homogeneity), 3)              AS homog_p50,
               mode(alt_name) FILTER (WHERE agree < 0.5)  AS usual_alternative
        FROM cells
        GROUP BY cls, name
        ORDER BY cells DESC
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    # Which word would the embedding have used instead: NLCD class (rows) x the
    # runner-up class (columns), counting only cells below 0.5 agreement.
    confusion = mo.sql(
        f"""
        PIVOT (
            SELECT cls, name, alt FROM cells WHERE agree < 0.5
        )
        ON alt USING count(*)
        GROUP BY cls, name
        ORDER BY cls
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    # How far apart the prototypes are: the pairs the embedding can barely tell apart.
    closest_pairs = mo.sql(
        f"""
        SELECT p.a, na.name AS a_name, p.b, nb.name AS b_name, round(p.cos, 3) AS cos
        FROM proto_pairs p
        JOIN class_names na ON na.cls = p.a
        JOIN class_names nb ON nb.cls = p.b
        WHERE p.a < p.b
        ORDER BY p.cos DESC
        LIMIT 12
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    # The leads: the least-backed cells, with what the embedding would call them.
    leads = mo.sql(
        f"""
        SELECT cell, name, alt_name, round(agree, 3) AS agree, round(alt_p, 3) AS alt_p,
               round(purity, 2) AS purity, round(homogeneity, 3) AS homogeneity
        FROM cells
        WHERE agree IS NOT NULL
        ORDER BY agree ASC
        LIMIT 25
        """,
        engine=con
    )
    return


if __name__ == "__main__":
    app.run()
