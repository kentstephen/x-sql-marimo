# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "duckdb>=1.5.5",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "obstore",
#     "pyarrow>=25.0.0",
#     "numpy",
#     "anywidget>=0.9",
#     "lonboard>=0.16.0",
#     "arro3-core",
#     "pillow",
# ]
# ///
"""USDA Cropland Data Layer with Fields of the World, as DuckDB SQL in marimo.

The crops notebook (xsql-cdl-crops.py) plus two checkboxes. The map is always
CDL pixel squares, 2008-2025, served from the pyramid by the camera as before
("crops only" / everything as there). FTW (Fields of the World: the PRUE
model's field polygons from Sentinel-2 at 10 m, 2024 and 2025, ~1.6 B fields
worldwide, CC-BY 4.0) enters as:

  fields         the pixels are clipped to the inside of the FTW fields and the
                 field outlines are drawn (same deck layer: the outline rows
                 are appended to the pixel table with a transparent fill). On
                 years before 2024 the 2024 footprint is the clip and the status
                 says so.
  disagreement   2024 and 2025 only: the pixels are repainted by whether CDL
                 calls them a crop and whether FTW sees a field there (P(field)
                 >= 0.5 from the probability pyramid): agree (grey), CDL crop but
                 no FTW field (orange), FTW field on CDL non-crop (blue). Works
                 with fields on (outlines over the paint) or off.

FTW's two years are served from CDL's 10 m group (FTW's own resolution, Stephen:
"2024 and 2025 can only be seen in the 10 m, keep it that way"); older years
are the 30 m group, fields still clip them, disagreement is greyed out with the
reason. Field shapes change over time, which is why nothing here claims a
per-field history across years.

WHICH DATA, FROM WHERE, WHAT TYPE (every leg is DuckDB):

  CDL crop_type 2008-2025, 30 m, EPSG:5070 (+ majority pyramid 2x..256x)
    s3://us-west-2.opendata.source.coop/chill/usda-cropland-data-layer/v0.1.0.icechunk
    icechunk Zarr v3, uint8 -> xql.register(con, "cdl_<k>", ds) on the xarray-sql
    0.4.0rc1 DuckDB backend: one table per pyramid level, columns year/y/x/crop_type.
  FTW field polygons 2024 + 2025 (fiboa GeoParquet, one file per US state, both
  years in the file, CRS84)
    s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/vectors/
      alpha/results-by-admin-conf/admin:country_code=US/US_<ST>.parquet
    -> read_parquet() through httpfs + spatial; the `bbox` struct prunes row groups.
  FTW softmax probabilities 2024 + 2025, 10 m, EPSG:4326, 14 multiscale levels
    s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/zarr/
      alpha/global.zarr (+ /4x .. /8192x)
    plain Zarr v3 (not icechunk), float32 variables(time, band, y, x), bands
    non_field_background / field / field_boundaries -> xql.register(con, "ftw_<k>", ds)
    on the same backend, blocks = the INNER chunk (512) so x/y predicates prune.
  Not used: FTW per-state PMTiles (draw-only, decimated, no id) and the 500 m
  confidence COGs (the `confidence` column is NULL for the entire US: the US is
  not one of the 24 labelled countries).

The FTW join is one point-in-polygon pass per (box, year, level): the CDL pixel
centres of the serve level into the field polygons (ST_Contains, DuckDB's
spatial join; 0.1 s for 2k fields x 660k pixels) into a (y, x) -> field lookup,
reused by every serve and every year of the mask. Disagreement bins the CDL
centres into the 40 m (or 160 m) FTW grid by index arithmetic.

Under the map: "analyze what's in view" (the crops notebook's panel: top crops
and the 18-year timelapse of the box, under the mask when it is on), then SQL
cells that show the same joins as plain statements on the box in view.

Measured 2026-08-20 (home link), Fresno 20 x 20 km: fields from US_CA.parquet
2.8 s (2.1k fields); CDL native pixels 0.7 s (659k); majority-crop join 0.1 s;
FTW probabilities through DuckDB at 4x 1.2 s, 16x 0.9 s. Record in
docs/ftw-cdl-notes.md.

Run from the root project (its xarray-sql is the 0.4.0rc1 DuckDB backend since
2026-08-20; this is one backend, not the multi-backend test bed):
  uv run marimo edit xsql-cdl-fields.py
or self-contained:
  uv run marimo edit xsql-cdl-fields.py --sandbox
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import base64
    import io
    import json
    import math
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image, ImageDraw

    import anywidget
    import duckdb
    import icechunk
    import xarray as xr
    import zarr
    import traitlets
    import xarray_sql as xql
    import urllib.parse
    import urllib.request

    from lonboard import BitmapLayer, Map
    from lonboard.basemap import CartoStyle, MaplibreBasemap
    from obstore.store import S3Store

    import marimo as mo

    return (
        BitmapLayer,
        CartoStyle,
        Image,
        ImageDraw,
        Map,
        MaplibreBasemap,
        S3Store,
        ThreadPoolExecutor,
        anywidget,
        asyncio,
        base64,
        duckdb,
        icechunk,
        io,
        json,
        math,
        mo,
        np,
        threading,
        time,
        traitlets,
        urllib,
        xql,
        xr,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Cropland Data Layer with Fields of the World, in SQL

    The [CDL notebook](xsql-cdl-crops.py) plus two checkboxes. The map is the
    **USDA Cropland Data Layer**: what grew on every pixel of CONUS, each year
    2008-2025, served from its pyramid by the camera (2024 and 2025 from the 10 m
    group, older years 30 m). **Fields of the World** (the PRUE model's field
    polygons from Sentinel-2, 10 m, 2024 and 2025) enters as **fields**: the pixels
    clipped to the inside of the fields with the field outlines drawn (the 2024
    footprint on older years); and **disagreement** (2024-2025 only): the pixels
    repainted by whether CDL calls them a crop and whether FTW sees a field there,
    with or without the fields on. Everything is **DuckDB**.

    | data | type on disk | how DuckDB reads it |
    |---|---|---|
    | CDL `crop_type(year, y, x)`, 30 m, EPSG:5070, 2008-2025, + majority pyramid | icechunk Zarr v3, uint8 | `xql.register` (xarray-sql DuckDB backend): tables `cdl_1` (native) .. `cdl_256` |
    | FTW field polygons, one GeoParquet per state, both years in the file | fiboa GeoParquet, CRS84 | `read_parquet(...)` over httpfs, `bbox` struct prunes row groups |
    | FTW softmax P(non-field / field / boundary), 10 m, EPSG:4326, 14 levels | plain Zarr v3, float32 | `xql.register`: tables `ftw_4` (40 m), `ftw_16` (160 m) |

    FTW's `confidence` column is NULL for the whole US (not one of the 24 labelled
    countries), so nothing here uses it. Field shapes change over time: disagreement
    is offered only for the two years FTW drew, and nothing here claims a per-field
    history across years.
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    BUCKET = "chill"
    PREFIX = "usda-cropland-data-layer/v0.1.0.icechunk"
    ENDPOINT = "https://data.source.coop"

    # FTW (Fields of the World), same source.coop bucket, different account
    FTW_BUCKET = "us-west-2.opendata.source.coop"
    FTW_VEC = (
        "tge-labs/ftw-global-data/predictions/vectors/alpha/"
        "results-by-admin-conf/admin:country_code=US/"
    )
    FTW_ZARR = "tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/"
    FTW_LEVELS = [4, 16]          # probability pyramid levels registered (40 m, 160 m)
    FTW_RES = 8.98311982e-05      # degrees per 10 m pixel at the root
    FTW_Y0 = 83.748345            # top edge of the grid
    FTW_YEARS = (2024, 2025)

    LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # 30 m pyramid factor; pixel = 30*k m
    LEVELS10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # the 10m group's own ladder
    # (native + majority pyramid, 2024-2025 only); pixel = 10*k m. FTW's two
    # years are served from the 10 m group, FTW's own resolution (Stephen).
    YEARS = list(range(2008, 2026))
    YEAR0 = 2025                  # CDL year at open
    FIELDS0 = True                # the fields checkbox at open

    ACRES_PER_KM2 = 247.10538

    PX_PER = 1.0                  # level floor: largest k with pixel <= PX_PER screen px
    ROW_BUDGET = 3_000_000        # max pixels per serve (numpy rows, not polygons; the
    # picture is the same size whatever the count); over it, coarsen a level
    OVERSAMPLE = 1.5              # picture px per screen px (deck filters the bitmap
    # linearly; 1.5 keeps level pixels crisp without 4x the bytes)
    FTW_BOX_DEG2 = 0.35           # FTW modes only when the (padded) box is under this
    FTW_PAD = 1.6                 # the FTW fetch covers PAD x the serve box each side
    FTW_FETCH_DEG2 = 0.4          # ... capped at this area. Measured: the state file
    # has ~48 row groups of ~13 MB, spatially sorted, so a wider box touches more
    # of them and the read is NOT flat (2.5x pad: 13 s open, 18 s on a miss);
    # a modest pad makes small pans and zoom-ins hits without that bill
    MARGIN = 0.35                 # fold box slack beyond the viewport
    VIEW_W, VIEW_H = 1400, 700    # the usual guess; no ruler
    HOME = {"longitude": -119.78, "latitude": 36.72, "zoom": 12.0}  # Fresno County

    # disagreement paint (protan-safe: grey / orange / blue, no red-green axis)
    DIS = {
        1: ("agree: CDL crop, FTW field", "#9a9a9a", (154, 154, 154)),
        2: ("CDL crop, no FTW field", "#e07a1e", (224, 122, 30)),
        3: ("FTW field, CDL not crop", "#2b6cdc", (43, 108, 220)),
    }

    HOLD: dict = {}
    return (
        ACRES_PER_KM2,
        BUCKET,
        DIS,
        ENDPOINT,
        FIELDS0,
        FTW_BOX_DEG2,
        FTW_BUCKET,
        FTW_FETCH_DEG2,
        FTW_PAD,
        FTW_LEVELS,
        FTW_RES,
        FTW_VEC,
        FTW_Y0,
        FTW_YEARS,
        FTW_ZARR,
        HOLD,
        HOME,
        LEVELS,
        LEVELS10,
        MARGIN,
        OVERSAMPLE,
        PREFIX,
        PX_PER,
        ROW_BUDGET,
        VIEW_H,
        VIEW_W,
        YEAR0,
    )


@app.cell
def _(
    BUCKET,
    ENDPOINT,
    FTW_BUCKET,
    FTW_LEVELS,
    FTW_ZARR,
    LEVELS,
    LEVELS10,
    PREFIX,
    S3Store,
    duckdb,
    icechunk,
    threading,
    xql,
    xr,
    zarr,
):
    # ---- open both stores, register every level as a DuckDB table -----------
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint_url=ENDPOINT,
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    _repo = icechunk.Repository.open(storage)
    _session = _repo.readonly_session("main")

    def _connect():
        c = duckdb.connect()
        c.sql(
            "INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;"
            " SET s3_region='us-west-2'; SET s3_url_style='path';"
        )
        return c

    con = _connect()
    con_lock = threading.Lock()

    DS = {}
    for _k in LEVELS:
        _grp = "30m" if _k == 1 else f"30m/{_k}x"
        DS[_k] = xr.open_zarr(_session.store, group=_grp, chunks=None)
    # the 10m group (2024-2025): a full mirror of 30m's structure (native +
    # 2x..512x majority pyramid, same extent, same attrs); whole-plane at
    # k >= 128, matching 30m's k >= 32 by plane size
    DS10 = {}
    for _k in LEVELS10:
        _grp = "10m" if _k == 1 else f"10m/{_k}x"
        DS10[_k] = xr.open_zarr(_session.store, group=_grp, chunks=None)

    # ---- FTW probabilities: plain Zarr v3 over obstore, levels 4x and 16x ----
    # Blocks = the level's INNER chunk (512 at these levels), never the shard
    # (4096): a shard-sized block expands whole and the window predicate cannot
    # prune inside it (measured 19.5 s vs 1.2 s for the same 20 km box).
    _ftw_store = zarr.storage.ObjectStore(
        S3Store(bucket=FTW_BUCKET, region="us-west-2", skip_signature=True,
                prefix=FTW_ZARR),
        read_only=True,
    )
    FTW_DS = {}
    for _k in FTW_LEVELS:
        FTW_DS[_k] = xr.open_zarr(_ftw_store, group=f"{_k}x", chunks=None,
                                  consolidated=False)

    def _register(c):
        for _k in LEVELS:
            _ds = DS[_k]
            # whole-plane per year at coarse levels, 2048^2 at fine levels so
            # the x/y predicates prune fragments (crops notebook's layout)
            if _k >= 32:
                _chunks = {"year": 1, "y": _ds.sizes["y"], "x": _ds.sizes["x"]}
            else:
                _chunks = {"year": 1, "y": 2048, "x": 2048}
            xql.register(c, f"cdl_{_k}", _ds, chunks=_chunks)
        for _k in LEVELS10:
            _ds = DS10[_k]
            if _k >= 128:
                _chunks = {"year": 1, "y": _ds.sizes["y"], "x": _ds.sizes["x"]}
            else:
                _chunks = {"year": 1, "y": 2048, "x": 2048}
            xql.register(c, f"cdl10_{_k}", _ds, chunks=_chunks)
        for _k in FTW_LEVELS:
            xql.register(c, f"ftw_{_k}", FTW_DS[_k],
                         chunks={"time": 1, "band": 3, "y": 512, "x": 512})

    _register(con)

    # ---- classes table from the CDL store's own attrs ----------------------
    _at = DS[1]["crop_type"].attrs
    _names, _colors = _at["class_names"], _at["class_colors"]

    def _noncrop(name):
        if name.startswith("Developed"):
            return True
        return name in {
            "Background", "Clouds/No Data", "Water", "Open Water",
            "Perennial Ice/Snow", "Barren", "Forest", "Deciduous Forest",
            "Evergreen Forest", "Mixed Forest", "Shrubland",
            "Grassland/Pasture", "Grass/Pasture", "Woody Wetlands",
            "Herbaceous Wetlands", "Wetlands", "Nonag/Undefined",
        }

    def _rgb(hexs):
        return int(hexs[1:3], 16), int(hexs[3:5], 16), int(hexs[5:7], 16)

    # protan-safe default palette: remap red-dominant classes (red is the weak
    # leg; cotton #FF2525 next to soybean green fails) onto a blue/purple cycle
    _SAFE_CYCLE = ["#3F6BD6", "#8E44AD", "#00B8D4", "#D633C4",
                   "#5C6BC0", "#0091EA", "#7C4DFF", "#6A1B9A"]
    _i = 0
    _rows = []
    for _code in sorted(_names, key=int):
        _nm, _hx = _names[_code], _colors[_code]
        _r, _g, _b = _rgb(_hx)
        _safe = _hx
        if _r >= 170 and _g <= 100 and _b <= 110:
            _safe = _SAFE_CYCLE[_i % len(_SAFE_CYCLE)]
            _i += 1
        _sr, _sg, _sb = _rgb(_safe)
        _rows.append((int(_code), _nm, _hx, _safe, _sr, _sg, _sb, _noncrop(_nm)))

    _CLASSES_DDL = (
        "CREATE TABLE classes(code UTINYINT, name VARCHAR, hex_official VARCHAR,"
        " hex VARCHAR, r UTINYINT, g UTINYINT, b UTINYINT, noncrop BOOLEAN)"
    )
    con.sql(_CLASSES_DDL)
    con.executemany("INSERT INTO classes VALUES (?,?,?,?,?,?,?,?)", _rows)

    # ---- the FTW state partitions: each file's extent from its OWN row-group
    # stats (parquet_metadata over all 60 files, 3.7 s on 2026-08-20; embedded
    # so open costs nothing). The STAC items' bbox is wrong (US_CA reports a box
    # in Montana), so it is not used. Non-CONUS rows (AK, HI, the Canadian and
    # Mexican border fragments) are kept; CDL has no pixels there.
    _STATES = [
        ("AB", -113.4609, 48.8716, -109.9513, 49.1153), ("AK", -179.1069, 51.2673, 178.5722, 71.3595),
        ("AL", -88.4692, 30.2366, -84.9303, 35.0198), ("AR", -94.6086, 32.9912, -89.6512, 36.5159),
        ("AZ", -114.8428, 31.3059, -108.9772, 37.1642), ("BC", -136.9020, 48.9859, -115.0546, 59.6770),
        ("BCN", -116.1912, 32.4933, -114.7463, 32.7504), ("CA", -124.3523, 32.5401, -114.1433, 42.1078),
        ("CHH", -108.7570, 29.0018, -103.3053, 31.7885), ("CO", -109.2281, 36.8565, -101.9949, 41.0663),
        ("COA", -103.3081, 28.9751, -101.2998, 29.6612), ("CT", -73.6412, 41.1286, -71.7891, 42.0537),
        ("DE", -75.7910, 38.4468, -75.0627, 39.8389), ("FL", -87.6050, 24.6337, -80.0375, 31.0112),
        ("GA", -85.6023, 30.3786, -80.8461, 34.9940), ("HI", -171.7315, 18.9141, -154.8429, 25.7605),
        ("IA", -96.6383, 40.3755, -90.1597, 43.5292), ("ID", -117.2075, 41.8476, -111.0439, 49.0006),
        ("IL", -91.5112, 36.9812, -87.4950, 42.5224), ("IN", -88.0956, 37.7752, -84.7778, 41.7762),
        ("KS", -102.1802, 36.8925, -94.5901, 40.0618), ("KY", -89.5650, 36.4889, -82.3231, 39.1427),
        ("LA", -94.0409, 29.1032, -89.1778, 33.0262), ("MA", -73.4565, 41.2416, -69.9653, 42.8878),
        ("MB", -101.3629, 48.9465, -95.3080, 49.0306), ("MD", -79.4903, 37.9769, -75.0799, 39.7319),
        ("ME", -71.0137, 43.1226, -67.0023, 47.4349), ("MI", -90.2135, 41.6930, -82.4660, 47.3937),
        ("MN", -97.2376, 43.4865, -90.0070, 49.3549), ("MO", -95.7638, 35.9749, -89.1052, 40.6164),
        ("MS", -91.6424, 30.2577, -88.1318, 35.0043), ("MT", -116.0404, 44.4582, -103.9331, 49.1742),
        ("NB", -67.7911, 46.1704, -67.7640, 47.0352), ("NC", -84.3100, 33.8565, -75.6323, 36.5740),
        ("ND", -104.0996, 45.8636, -96.5552, 49.0298), ("NE", -104.2059, 39.9444, -95.3097, 43.1038),
        ("NH", -72.5293, 42.6948, -70.7183, 45.1750), ("NJ", -75.5626, 38.9404, -73.9993, 41.3522),
        ("NM", -109.1490, 31.3281, -102.7869, 37.1542), ("NV", -120.1432, 35.0057, -113.7718, 42.1525),
        ("NY", -79.7662, 40.6174, -72.1221, 45.0234), ("OH", -84.8425, 38.4374, -80.5134, 41.9528),
        ("OK", -103.0702, 33.6282, -94.4282, 37.1493), ("OR", -124.5325, 41.7591, -116.5060, 46.1685),
        ("PA", -80.5300, 39.7032, -74.7718, 42.2674), ("QC", -74.4936, 44.9868, -69.0288, 47.4349),
        ("RI", -71.8374, 41.1601, -71.1195, 42.0213), ("SC", -83.2788, 32.0489, -78.6307, 35.1982),
        ("SD", -104.1481, 42.4952, -96.4253, 46.0128), ("SK", -110.0217, 48.8504, -101.3551, 49.1742),
        ("SON", -114.8428, 31.3059, -108.7519, 32.5818), ("TN", -90.3186, 34.9703, -81.7296, 36.6666),
        ("TX", -106.6500, 25.8412, -93.6194, 36.6163), ("UT", -114.2245, 36.8858, -108.9914, 42.1293),
        ("VA", -83.6007, 36.5346, -75.3106, 39.4304), ("VT", -73.4202, 42.7267, -71.5183, 45.0286),
        ("WA", -124.3931, 45.5561, -116.9255, 49.0109), ("WI", -92.8177, 42.4731, -86.8788, 46.9017),
        ("WV", -82.6197, 37.2515, -77.7529, 40.6241), ("WY", -111.1497, 40.8559, -103.8754, 45.1036),
        ("YT", -141.0438, 60.0153, -139.0725, 69.6589),
    ]
    _STATES_DDL = (
        "CREATE TABLE ftw_states(st VARCHAR, xmin DOUBLE, ymin DOUBLE,"
        " xmax DOUBLE, ymax DOUBLE)"
    )
    con.sql(_STATES_DDL)
    con.executemany("INSERT INTO ftw_states VALUES (?,?,?,?,?)", _STATES)

    # A SECOND connection for the map serve path (crops notebook's lesson: marimo
    # SQL cells hold streaming Arrow results open on con, and a serve query
    # interleaving on the same connection raises "Can't 'FetchRaw' from
    # ArrowQueryResult"). mcon carries its own registrations and table copies.
    mcon = _connect()
    _register(mcon)
    mcon.sql(_CLASSES_DDL)
    mcon.executemany("INSERT INTO classes VALUES (?,?,?,?,?,?,?,?)", _rows)
    mcon.sql(_STATES_DDL)
    mcon.executemany("INSERT INTO ftw_states VALUES (?,?,?,?,?)", _STATES)

    # a THIRD connection for the FTW parquet fetch on a thread (httpfs +
    # spatial only, no registrations): it runs concurrently with the CDL
    # centre scan on mcon and hands the polygons back as Arrow (WKB)
    fcon = _connect()

    NONCROP_CODES = sorted(r[0] for r in _rows if r[7])
    return NONCROP_CODES, con, con_lock, fcon, mcon


@app.cell
def _(
    FTW_BUCKET,
    FTW_RES,
    FTW_VEC,
    FTW_Y0,
    Image,
    ImageDraw,
    MARGIN,
    OVERSAMPLE,
    VIEW_H,
    VIEW_W,
    base64,
    io,
    math,
    np,
):
    # ---- the serve helpers: pure functions of (connection, box, ...), shared
    # by the map cell (opening view) and the wiring cell. No HUD dependency, so
    # the map cell never re-runs on a control change.
    def albers_xy(lon, lat):
        """EPSG:5070 forward (Albers equal-area conic on GRS80), closed form in
        numpy; verified against DuckDB ST_Transform to the millimetre at five
        CONUS points (2026-08-20). No pyproj."""
        a = 6378137.0
        e2 = 0.00669438002290
        e = math.sqrt(e2)
        lat0, lon0 = math.radians(23.0), math.radians(-96.0)
        lat1, lat2 = math.radians(29.5), math.radians(45.5)

        def m(p):
            return np.cos(p) / np.sqrt(1 - e2 * np.sin(p) ** 2)

        def q(p):
            sp = np.sin(p)
            return (1 - e2) * (sp / (1 - e2 * sp * sp)
                               - (1 / (2 * e)) * np.log((1 - e * sp) / (1 + e * sp)))

        n = (m(lat1) ** 2 - m(lat2) ** 2) / (q(lat2) - q(lat1))
        C = m(lat1) ** 2 + n * q(lat1)
        rho0 = a * np.sqrt(C - n * q(lat0)) / n
        lon = np.radians(lon)
        lat = np.radians(lat)
        rho = a * np.sqrt(C - n * q(lat)) / n
        th = n * (lon - lon0)
        return rho * np.sin(th), rho0 - rho * np.cos(th)

    def render_view(px_x, px_y, rgb, x0, y0, x1, y1, pix_m, W, S, E, N, rings=None,
                    line=(40, 40, 40, 210), scale=OVERSAMPLE):
        """ONE PNG of the view: the drawn pixels (Albers centres + colours) are
        dropped into a dense grid over the Albers box, every output pixel of
        the lon/lat view box is forward-transformed into that grid (nearest),
        then the FTW rings (already lon/lat) are drawn on top. Returns a data
        URL and the bounds. Transparent where nothing is drawn."""
        w = int(VIEW_W * (1 + MARGIN) * scale)
        h = int(VIEW_H * (1 + MARGIN) * scale)
        # dense Albers grid of the level's pixels
        gw = int(round((x1 - x0) / pix_m)) + 1
        gh = int(round((y1 - y0) / pix_m)) + 1
        grid = np.zeros((gh, gw, 4), dtype=np.uint8)
        if len(px_x):
            ix = np.clip(((px_x - x0) / pix_m).astype(np.int64), 0, gw - 1)
            iy = np.clip(((y1 - px_y) / pix_m).astype(np.int64), 0, gh - 1)
            grid[iy, ix, :3] = rgb
            grid[iy, ix, 3] = 255
        # output lon/lat lattice -> Albers -> grid index
        lons = W + (np.arange(w) + 0.5) * (E - W) / w
        lats = N - (np.arange(h) + 0.5) * (N - S) / h
        LON, LAT = np.meshgrid(lons, lats)
        X, Y = albers_xy(LON, LAT)
        jx = ((X - x0) / pix_m).astype(np.int64)
        jy = ((y1 - Y) / pix_m).astype(np.int64)
        ok = (jx >= 0) & (jx < gw) & (jy >= 0) & (jy < gh)
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[ok] = grid[jy[ok], jx[ok]]
        img = Image.fromarray(out, "RGBA")
        if rings:
            d = ImageDraw.Draw(img)
            sx, sy = w / (E - W), h / (N - S)
            for ring in rings:
                pts = [((lon - W) * sx, (N - lat) * sy) for lon, lat in ring]
                if len(pts) >= 2:
                    d.line(pts + [pts[0]], fill=line, width=1)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=4)
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return url, [W, S, E, N], len(buf.getvalue())

    def rings_from_geojson(rows, json_loads):
        """Every ring (outer and holes) of polygons / multipolygons from
        ST_AsGeoJSON rows, as lists of (lon, lat)."""
        rings = []
        for (gj,) in rows:
            if not gj:
                continue
            g = json_loads(gj)
            if g.get("type") == "Polygon":
                rings.extend(g["coordinates"])
            elif g.get("type") == "MultiPolygon":
                for poly in g["coordinates"]:
                    rings.extend(poly)
        return [[(c[0], c[1]) for c in r] for r in rings if len(r) >= 3]

    def bbox4326(vs):
        span = 360.0 / (512 * 2 ** vs["zoom"])
        dlon = VIEW_W * span * (1 + MARGIN) / 2
        dlat = VIEW_H * span * math.cos(math.radians(vs["latitude"])) * (1 + MARGIN) / 2
        return (vs["longitude"] - dlon, vs["latitude"] - dlat,
                vs["longitude"] + dlon, vs["latitude"] + dlat)

    def to5070(c, lon0, lat0, lon1, lat1):
        # densified box boundary, clamped to the array's Albers bbox (an
        # EPSG:5070 parallel bows; corner-only min clips the Gulf coast)
        _N = 8
        pts = []
        for _i in range(_N + 1):
            _t = _i / _N
            _lon = lon0 + (lon1 - lon0) * _t
            _lat = lat0 + (lat1 - lat0) * _t
            pts += [(_lon, lat0), (_lon, lat1), (lon0, _lat), (lon1, _lat)]
        vals = ", ".join(f"({a}, {b})" for a, b in pts)
        rows = c.sql(
            f"""SELECT ST_X(p), ST_Y(p) FROM (
                  SELECT ST_Transform(ST_Point(lon, lat), 'EPSG:4326', 'EPSG:5070',
                                      always_xy := true) AS p
                  FROM (VALUES {vals}) AS t(lon, lat))"""
        ).fetchall()
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        _X0, _Y0, _X1, _Y1 = -2417835.0, 158265.0, 2387295.0, 3321225.0
        return (max(min(xs), _X0), max(min(ys), _Y0),
                min(max(xs), _X1), min(max(ys), _Y1))

    def ftw_files(c, W, S, E, N):
        """The state parquet files whose extent meets the box."""
        sts = c.sql(
            f"""SELECT st FROM ftw_states
                WHERE xmax > {W} AND xmin < {E} AND ymax > {S} AND ymin < {N}
                ORDER BY st"""
        ).fetchall()
        return [f"s3://{FTW_BUCKET}/{FTW_VEC}US_{r[0]}.parquet" for r in sts]

    def fields_sql(files, W, S, E, N, ftw_year):
        """Fields of the box for one FTW year, with the bbox-struct predicate
        (row groups are spatially sorted, so it prunes). The file's geometry
        type is GEOMETRY('OGC:CRS84'); lonboard's from_duckdb only recognises
        plain GEOMETRY, hence the cast."""
        fl = ", ".join(f"'{f}'" for f in files)
        return f"""
            SELECT id, "metrics:area" AS area_m2, geometry::GEOMETRY AS geometry
            FROM read_parquet([{fl}])
            WHERE bbox.xmin > {W} AND bbox.xmax < {E}
              AND bbox.ymin > {S} AND bbox.ymax < {N}
              AND date_part('year', "determination:datetime" AT TIME ZONE 'UTC')
                  = {ftw_year}
        """

    def lookup_sql(tbl, x0, y0, x1, y1, fields_tbl):
        """(y, x) -> field id for every CDL pixel centre of table `tbl` (a
        cdl_<k> / cdl10_<k> level) inside a field: ONE point-in-polygon pass
        (ST_Contains, DuckDB's spatial join), reused by every serve afterwards
        as a plain hash join."""
        return f"""
            WITH p AS (
                SELECT DISTINCT y, x,
                       ST_Transform(ST_Point(x, y), 'EPSG:5070', 'EPSG:4326',
                                    always_xy := true) AS pt
                FROM {tbl}
                WHERE year = 2025
                  AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1}
            )
            SELECT f.id, p.y, p.x
            FROM {fields_tbl} f JOIN p ON ST_Contains(f.geometry, p.pt)
        """

    def ftw_grid_sql(px_m, year, W, S, E, N):
        """The FTW cells that are field (P(field) >= 0.5) in the 4326 box, as
        grid indexes: ftw_4 (40 m) under CDL 120 m pixels, ftw_16 (160 m) from
        there. Returns (sql, res, factor) so the caller bins CDL centres the
        same way."""
        f = 4 if px_m < 120 else 16
        res = FTW_RES * f
        return (
            f"""SELECT floor((x + 180) / {res})::BIGINT AS ix,
                       floor(({FTW_Y0} - y) / {res})::BIGINT AS iy
                FROM ftw_{f}
                WHERE time = TIMESTAMP '{year}-01-01' AND band = 'field'
                  AND variables >= 0.5
                  AND x BETWEEN {W} AND {E} AND y BETWEEN {S} AND {N}""",
            res, f,
        )

    return (albers_xy, bbox4326, fields_sql, ftw_files, ftw_grid_sql, lookup_sql,
            render_view, rings_from_geojson, to5070)


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """Controls + status + analysis UNDER the map: the crops notebook's
        strip plus two checkboxes, `fields` (clip to the FTW fields, outlines
        drawn) and `disagreement` (repaint by CDL-crop x FTW-field; greyed
        out before 2024). Proven trait types only: `ctl` Unicode browser ->
        kernel (JSON with `act`: "set" | "analyze" | "search"), `status` /
        `panel` / `legend` Unicode kernel -> browser. Commits on `change` +
        250 ms debounce, never `input`."""

        ctl = traitlets.Unicode("").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)

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
          const yl = document.createElement("span");
          yl.textContent = "year";
          const range = document.createElement("input");
          range.type = "range";
          range.min = "2008"; range.max = "2025"; range.step = "1";
          range.value = "2025";
          range.style.cssText = "width:11rem";
          const yv = document.createElement("span");
          yv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums";
          yv.textContent = range.value;
          const arrow = (txt, d) => {
            const a = document.createElement("button");
            a.textContent = txt;
            a.style.cssText = btnCss;
            a.addEventListener("click", () => {
              const v = Math.min(2025, Math.max(2008, +range.value + d));
              if (v === +range.value) return;
              range.value = String(v);
              yv.textContent = range.value;
              syncFtw();
              commit();
            });
            return a;
          };
          const prevB = arrow("◀", -1);
          const nextB = arrow("▶", 1);
          const lab = document.createElement("label");
          lab.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const c = document.createElement("input");
          c.type = "checkbox";
          c.checked = false;
          lab.appendChild(c);
          lab.appendChild(document.createTextNode("crops only"));
          // the two FTW controls: fields (pixels clipped to inside the FTW
          // fields, outlines drawn) and disagreement (pixels repainted by
          // CDL-crop x FTW-field; 2024-2025 only, greyed out otherwise)
          const mk = (text) => {
            const l = document.createElement("label");
            l.style.cssText =
              "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
            const i = document.createElement("input");
            i.type = "checkbox"; i.checked = false;
            l.appendChild(i); l.appendChild(document.createTextNode(text));
            return [l, i];
          };
          const [labF, fld] = mk("fields");
          fld.checked = true;
          const [labD, dis] = mk("disagreement");
          const syncFtw = () => {
            const ok = +range.value >= 2024;
            dis.disabled = !ok;
            labD.style.opacity = ok ? "1" : ".45";
            labD.title = ok ? "" : "disagreement needs 2024 or 2025 (FTW's years)";
          };
          syncFtw();
          const btn = document.createElement("button");
          btn.textContent = "analyze what's in view";
          btn.style.cssText = btnCss;
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place…";
          search.style.cssText =
            "width:11rem;font:12px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) {
              model.set("ctl", JSON.stringify({
                act: "search", q: q, year: +range.value, fields: fld.checked,
                dis: dis.checked, crops: c.checked, sel: Array.from(sel), n: ++seq }));
              model.save_changes();
            }
          });
          // the legend: classes in view, pickable (click isolates, x all resets)
          const sel = new Set();
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
                (on ? "#2b6cb0" : "transparent") +
                (on ? ";font-weight:600" : "");
              b.title = it.pct + "% of view";
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
          box.append(yl, prevB, range, nextB, yv, lab, labF, labD, btn, search, legendBox);
          const status = document.createElement("div");
          status.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const res = document.createElement("div");
          res.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.1rem 0";
          const wrap = document.createElement("div");
          wrap.dataset.cdlStrip = "1";
          wrap.append(box, status, res);
          // one strip only: remove every earlier strip in the page (shadow
          // roots included) before adding ours (crops notebook's lesson)
          const killOld = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("[data-cdl-strip]").forEach((w) => {
              if (w !== wrap) { w.dataset.dead = "1"; w.remove(); }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) killOld(n.shadowRoot);
            });
          };
          killOld(document);
          el.appendChild(wrap);
          // fullscreen: re-parent the strip into the fullscreen element as a
          // docked bottom bar (document.fullscreenElement is the shadow HOST;
          // descend to the real one)
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
          let seq = 0, deb = null;
          const send = (act) => {
            model.set("ctl", JSON.stringify({
              act: act, year: +range.value, fields: fld.checked, dis: dis.checked,
              crops: c.checked, sel: Array.from(sel), n: ++seq }));
            model.save_changes();
          };
          const commit = () => {
            clearTimeout(deb);
            deb = setTimeout(() => send("set"), 250);
          };
          range.addEventListener("input", () => { yv.textContent = range.value; syncFtw(); });
          range.addEventListener("change", commit);
          fld.addEventListener("change", commit);
          dis.addEventListener("change", () => { sel.clear(); commit(); });
          c.addEventListener("change", commit);
          btn.addEventListener("click", () => {
            res.innerHTML = '<span style="opacity:.6">analyzing…</span>';
            send("analyze");
          });
          const paintS = () => { status.textContent = model.get("status") || ""; };
          const paintP = () => {
            const html = model.get("panel") || "";
            res.innerHTML = "";
            if (!html) return;
            const x = document.createElement("button");
            x.textContent = "× clear";
            x.style.cssText =
              "float:right;font:11px ui-sans-serif,system-ui,sans-serif;" +
              "cursor:pointer;padding:.05rem .4rem;border-radius:4px;border:" +
              "1px solid rgba(127,127,127,.45);background:transparent;" +
              "color:inherit;margin-left:.6rem";
            x.addEventListener("click", () => { res.innerHTML = ""; });
            const body = document.createElement("div");
            body.innerHTML = html;
            res.append(x, body);
          };
          model.on("change:status", paintS);
          model.on("change:panel", paintP);
          paintS(); paintP();
          // hide lonboard's draw-box tool (rendered unconditionally in 0.16)
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
def _(
    BitmapLayer,
    CartoStyle,
    HOLD: dict,
    HOME,
    Map,
    MaplibreBasemap,
    YEAR0,
    bbox4326,
    mcon,
    np,
    render_view,
    to5070,
):
    # ---- map cell: builds the Map and the ONE layer, must never re-run. The
    # layer is a BitmapLayer: the serve paints the view as one PNG (pixels
    # from SQL, outlines drawn on top), so the payload is a few hundred KB at
    # any resolution instead of a polygon per pixel (Stephen, 2026-08-20: the
    # squares bought nothing, not even picking). The opening image is HOME
    # (Fresno County) as plain pixels from the 10 m group's 40 m level; the
    # wiring's first forced serve replaces it with the fields-clipped view.
    # NEVER image="" (deck's whole update pass dies, repo lesson): the opening
    # image is a real PNG.
    _W, _S, _E, _N = bbox4326(HOME)
    _x0, _y0, _x1, _y1 = to5070(mcon, _W, _S, _E, _N)
    HOLD["served"] = None
    HOLD.pop("k", None)
    _rows = mcon.sql(
        f"""
        SELECT t.x, t.y, c.r, c.g, c.b
        FROM cdl10_4 t JOIN classes c ON c.code = t.crop_type
        WHERE t.year = {YEAR0} AND t.crop_type NOT IN (0, 81)
          AND t.x BETWEEN {_x0} AND {_x1} AND t.y BETWEEN {_y0} AND {_y1}
        """
    ).fetchnumpy()
    _rgb = np.stack([_rows["r"], _rows["g"], _rows["b"]], axis=1).astype(np.uint8)
    _url, _bounds, _nbytes = render_view(
        np.asarray(_rows["x"], dtype=np.float64), np.asarray(_rows["y"], dtype=np.float64),
        _rgb, _x0, _y0, _x1, _y1, 40.0, _W, _S, _E, _N,
    )
    pixels = BitmapLayer(image=_url, bounds=_bounds, opacity=1.0)

    deck = Map(
        layers=[pixels],
        basemap=MaplibreBasemap(style=CartoStyle.Positron),
        view_state=HOME,
        height=700,
        show_side_panel=False,
    )
    deck
    return deck, pixels


@app.cell
def _(HudControls, mo):
    hud = mo.ui.anywidget(HudControls())
    hud
    return (hud,)


@app.cell
def _(
    ACRES_PER_KM2,
    DIS,
    FIELDS0,
    FTW_BOX_DEG2,
    FTW_Y0,
    FTW_YEARS,
    HOLD: dict,
    HOME,
    LEVELS,
    LEVELS10,
    NONCROP_CODES,
    PX_PER,
    ROW_BUDGET,
    VIEW_W,
    YEAR0,
    FTW_FETCH_DEG2,
    FTW_PAD,
    ThreadPoolExecutor,
    asyncio,
    bbox4326,
    con_lock,
    deck,
    fcon,
    fields_sql,
    ftw_files,
    ftw_grid_sql,
    hud,
    json,
    lookup_sql,
    math,
    mcon,
    np,
    pixels,
    render_view,
    rings_from_geojson,
    time,
    to5070,
    urllib,
):
    # ---- wiring cell: re-runs freely; a HUD commit (ctl) re-runs it ----------
    # No threads and no timers in the serve path (crops notebook's third
    # rework): async settle-debounce on the kernel's loop, busy/pending
    # coalescing, every trait assignment on the loop thread.
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _year = int(_c.get("year", YEAR0))
    _crops_only = bool(_c.get("crops", False))
    _fields = bool(_c.get("fields", FIELDS0))
    _dis_req = bool(_c.get("dis", False))
    _act = _c.get("act", "set")
    _q = str(_c.get("q", "")).strip()
    # FTW drew 2024 and 2025: those two years are served from CDL's 10 m group
    # (FTW's own resolution) and are the only years with disagreement; older
    # years stay 30 m, fields still clip them (today's footprint, stated in
    # the status), and a disagreement request is reported, not honoured
    _ftw_ok = _year in FTW_YEARS
    _dis = _dis_req and _ftw_ok
    _B = 10 if _ftw_ok else 30                   # base pixel, metres
    _LV = LEVELS10 if _ftw_ok else LEVELS        # the group's ladder
    _T = "cdl10_" if _ftw_ok else "cdl_"         # table prefix
    _fyear = _year if _ftw_ok else FTW_YEARS[0]
    _sel = tuple(sorted(int(v) for v in (_c.get("sel") or [])))
    _sel_col = "cls" if _dis else "crop_type"
    _sel_sql = (
        f" AND {_sel_col} IN ({', '.join(str(v) for v in _sel)})" if _sel else ""
    )
    HOLD["year"] = _year
    HOLD["fyear"] = _fyear
    HOLD["fields"] = _fields
    HOLD["dis"] = _dis
    SETTLE = 0.35

    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _vsd(vs):
        if vs is None:
            return None
        if isinstance(vs, dict):
            d = {k: vs.get(k) for k in ("longitude", "latitude", "zoom")}
        else:
            d = {k: getattr(vs, k, None) for k in ("longitude", "latitude", "zoom")}
        return d if None not in d.values() else None

    def _pick_level(vs):
        mpp = 156543.03392 * math.cos(math.radians(vs["latitude"])) / 2 ** vs["zoom"]
        want = max(mpp * PX_PER / _B, 1.0)
        ks = [k for k in _LV if k <= want]
        return ks[-1] if ks else _LV[0]

    def _drop_list():
        return "(0, 81)" if not _crops_only else "(" + ", ".join(
            str(c) for c in sorted({0, 81, *NONCROP_CODES})) + ")"

    def _window(vs):
        """Level + Albers box for a view: floor pick, then the count budget."""
        budget = ROW_BUDGET
        k = _pick_level(vs)
        x0, y0, x1, y1 = to5070(mcon, *bbox4326(vs))
        drop = _drop_list()
        while k < _LV[-1]:
            _est = (x1 - x0) * (y1 - y0) / (_B * k) ** 2
            if _est <= budget:
                break
            if _est > 24 * budget:
                k = _LV[_LV.index(k) + 1]
                continue
            _n = mcon.sql(
                f"""SELECT count(*) FROM {_T}{k}
                    WHERE year = {_year} AND crop_type NOT IN {drop}
                      {_sel_sql if not _dis else ""}
                      AND x BETWEEN {x0} AND {x1}
                      AND y BETWEEN {y0} AND {y1}"""
            ).fetchone()[0]
            if _n > budget:
                k = _LV[_LV.index(k) + 1]
                continue
            break
        return k, x0, y0, x1, y1, drop

    # ---- the FTW tables. Two caches: (1) polygons fetched ONCE for a PADDED
    # box (FTW_PAD x the serve box each side, capped at FTW_FETCH_DEG2; one
    # parquet read costs ~3 s whatever the box, so fetching wide makes pans
    # and zoom-ins cache hits), (2) the pixel -> field lookup per (table,
    # level, serve box), built from the cached polygons. A cold miss fetches
    # the parquet on fcon in a thread WHILE mcon scans the CDL centres. -----
    _pool = HOLD.setdefault("pool", ThreadPoolExecutor(max_workers=2))

    def _padded(W, S, E, N):
        cw, ch = (W + E) / 2, (S + N) / 2
        dw, dh = (E - W) / 2, (N - S) / 2
        f = FTW_PAD
        if (2 * dw * f) * (2 * dh * f) > FTW_FETCH_DEG2:
            f = max(1.0, (FTW_FETCH_DEG2 / (4 * dw * dh)) ** 0.5)
        return cw - dw * f, ch - dh * f, cw + dw * f, ch + dh * f

    def _fetch_fields(files, fyear, W, S, E, N):
        """Blocking parquet read on fcon -> Arrow (geometry as WKB)."""
        _sql = fields_sql(files, W, S, E, N, fyear)
        # read_all(): .arrow() alone is a LAZY reader and the read would happen
        # on the caller's thread at CREATE TABLE time (no concurrency at all)
        return fcon.sql(
            f"SELECT id, area_m2, ST_AsWKB(geometry) AS wkb FROM ({_sql})"
        ).arrow().read_all()

    def _adopt_fetch(fut, fyear, pw, ps, pe, pn):
        """A finished fetch future -> polygon table on mcon + cache entry.
        Runs on the serve thread (mcon is single-threaded under con_lock)."""
        arrow = fut.result()
        n = HOLD.get("fb_n", 0)
        HOLD["fb_n"] = n + 1
        fbt = f"fb_{n}"
        mcon.register(f"_arrow_{n}", arrow)
        mcon.sql(
            f"""CREATE OR REPLACE TABLE {fbt} AS
                SELECT id, area_m2, ST_GeomFromWKB(wkb) AS geometry
                FROM _arrow_{n}"""
        )
        mcon.unregister(f"_arrow_{n}")
        cache = HOLD.setdefault("fcache", {})
        cache[(fyear, round(pw, 3), round(ps, 3), round(pe, 3), round(pn, 3))] = fbt
        if len(cache) > 6:
            _old = next(iter(cache))
            _ot = cache.pop(_old)
            try:
                mcon.sql(f"DROP TABLE IF EXISTS {_ot}")
            except Exception:
                pass
        return fbt, (pw, ps, pe, pn)

    def _fields_cache(W, S, E, N):
        """(polygon table on mcon, its box) covering the box for _fyear: a
        cached padded fetch that contains it, a finished background prefetch
        that does, else a new padded fetch (blocking; the caller runs the
        CDL centre scan on mcon while it is in flight). Also queues a WIDER
        prefetch (2x the padded box) in the pool so the next miss finds the
        row groups warm in fcon's httpfs cache and lands as a table."""
        cache = HOLD.setdefault("fcache", {})
        pf = HOLD.get("prefetch")
        if pf is not None and pf[0].done():
            HOLD["prefetch"] = None
            try:
                _adopt_fetch(*pf)
            except Exception:
                pass
        for (_fy, _w, _s, _e, _n), _t in cache.items():
            if _fy == _fyear and _w <= W and _s <= S and _e >= E and _n >= N:
                return _t, (_w, _s, _e, _n), None
        pw, ps, pe, pn = _padded(W, S, E, N)
        _files = ftw_files(mcon, pw, ps, pe, pn)   # ftw_states lives on mcon
        fut = _pool.submit(_fetch_fields, _files, _fyear, pw, ps, pe, pn)
        return None, (pw, ps, pe, pn), fut

    def _queue_prefetch(W, S, E, N):
        """After a cold miss: fetch a ring twice as wide in the background."""
        if HOLD.get("prefetch") is not None:
            return
        cw, ch = (W + E) / 2, (S + N) / 2
        dw, dh = (E - W), (N - S)   # 2x each side of the padded box
        pw, ps, pe, pn = cw - dw, ch - dh, cw + dw, ch + dh
        cache = HOLD.setdefault("fcache", {})
        if any(_fy == _fyear and _w <= pw and _s <= ps and _e >= pe and _n >= pn
               for (_fy, _w, _s, _e, _n) in cache):
            return
        _files = ftw_files(mcon, pw, ps, pe, pn)
        fut = _pool.submit(_fetch_fields, _files, _fyear, pw, ps, pe, pn)
        HOLD["prefetch"] = (fut, _fyear, pw, ps, pe, pn)

    def _ftw_tables_at(T, k, W, S, E, N):
        """(fields table, lookup table) for table prefix T at level k. The
        lookup is built ONCE over the cached polygon box (not the serve box),
        so pans inside it are hits; keyed per (year, base, level, that box)."""
        B = 10 if T == "cdl10_" else 30
        lk = HOLD.setdefault("fb", {})
        for (_fy, _bb, _k, _w, _s, _e, _n), _v in lk.items():
            if (_fy, _bb, _k) == (_fyear, B, k) and _w <= W and _s <= S and _e >= E and _n >= N:
                return _v
        fbt, (pw, ps, pe, pn), fut = _fields_cache(W, S, E, N)
        x0, y0, x1, y1 = to5070(mcon, pw, ps, pe, pn)
        n = HOLD.get("lk_n", 0)
        HOLD["lk_n"] = n + 1
        # the CDL centres of the padded box, scanned while the parquet read
        # (if any) is in flight on the other connection
        pxt = f"px_{n}"
        mcon.sql(
            f"""CREATE OR REPLACE TABLE {pxt} AS
                SELECT DISTINCT y, x,
                       ST_Transform(ST_Point(x, y), 'EPSG:5070', 'EPSG:4326',
                                    always_xy := true) AS pt
                FROM {T}{k}
                WHERE year = 2025
                  AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1}"""
        )
        cold = fut is not None
        if cold:
            fbt, _ = _adopt_fetch(fut, _fyear, pw, ps, pe, pn)
        lkt = f"lk_{n}"
        mcon.sql(
            f"""CREATE OR REPLACE TABLE {lkt} AS
                SELECT f.id, p.y, p.x
                FROM {fbt} f JOIN {pxt} p ON ST_Contains(f.geometry, p.pt)"""
        )
        mcon.sql(f"DROP TABLE IF EXISTS {pxt}")
        key = (_fyear, B, k, round(pw, 3), round(ps, 3), round(pe, 3), round(pn, 3))
        lk[key] = (fbt, lkt)
        if len(lk) > 8:
            _old = next(iter(lk))
            _ot = lk.pop(_old)
            try:
                mcon.sql(f"DROP TABLE IF EXISTS {_ot[1]}")
            except Exception:
                pass
        if cold:
            _queue_prefetch(pw, ps, pe, pn)
        return lk[key]

    def _ftw_tables(W, S, E, N, k):
        return _ftw_tables_at(_T, k, W, S, E, N)

    def _ftw_grid(px_m, W, S, E, N):
        """The FTW field cells (P >= 0.5) for a PADDED box, cached per (year,
        level, box) as a table of grid indexes; the disagreement query joins
        it. Returns (table, res, factor)."""
        f = 4 if px_m < 120 else 16
        cache = HOLD.setdefault("gcache", {})
        for (_fy, _f, _w, _s, _e, _n), _t in cache.items():
            if (_fy, _f) == (_fyear, f) and _w <= W and _s <= S and _e >= E and _n >= N:
                return _t, ftw_grid_sql(px_m, _fyear, W, S, E, N)[1], f
        pw, ps, pe, pn = _padded(W, S, E, N)
        _gsql, _res, _f = ftw_grid_sql(px_m, _fyear, pw, ps, pe, pn)
        n = HOLD.get("g_n", 0)
        HOLD["g_n"] = n + 1
        gt = f"g_{n}"
        mcon.sql(f"CREATE OR REPLACE TABLE {gt} AS {_gsql}")
        cache[(_fyear, f, round(pw, 3), round(ps, 3), round(pe, 3), round(pn, 3))] = gt
        if len(cache) > 6:
            _old = next(iter(cache))
            _ot = cache.pop(_old)
            try:
                mcon.sql(f"DROP TABLE IF EXISTS {_ot}")
            except Exception:
                pass
        return gt, _res, _f

    _LUT = HOLD.get("lut")
    if _LUT is None:
        _LUT = np.zeros((256, 3), dtype=np.uint8)
        for _code, _r, _g, _b in mcon.sql("SELECT code, r, g, b FROM classes").fetchall():
            _LUT[_code] = (_r, _g, _b)
        HOLD["lut"] = _LUT
    _DIS_LUT = np.zeros((4, 3), dtype=np.uint8)
    for _code, (_nm, _hx, _c3) in DIS.items():
        _DIS_LUT[_code] = _c3

    def _paint(k, x0, y0, x1, y1, W, S, E, N, where, dis, fbt):
        """One PNG from `cur` (x, y, crop_type, cls): colours by class LUT or
        by disagreement class; FTW rings drawn when fbt is given."""
        rows = mcon.sql(
            f"SELECT x, y, {'cls' if dis else 'crop_type'} AS c FROM cur t WHERE TRUE{where}"
        ).fetchnumpy()
        codes = np.asarray(rows["c"])
        rgb = (_DIS_LUT if dis else _LUT)[codes]
        rings = None
        if fbt is not None:
            _tol = _B * k / 2 / 111_000
            rings = rings_from_geojson(mcon.sql(
                f"""SELECT ST_AsGeoJSON(ST_SimplifyPreserveTopology(geometry, {_tol}))
                    FROM {fbt}
                    WHERE ST_XMin(geometry) < {E} AND ST_XMax(geometry) > {W}
                      AND ST_YMin(geometry) < {N} AND ST_YMax(geometry) > {S}"""
            ).fetchall(), json.loads)
        url, bounds, nbytes = render_view(
            np.asarray(rows["x"], dtype=np.float64), np.asarray(rows["y"], dtype=np.float64),
            rgb, x0, y0, x1, y1, float(_B * k), W, S, E, N, rings=rings,
        )
        return url, bounds, len(codes), nbytes

    def _ftw_warm(W, S, E, N, k, dis):
        """True when everything a FTW frame needs is already cached (no
        parquet / grid fetch ahead), so stage 1 can be skipped."""
        fc = HOLD.get("fcache", {})
        poly = any(_fy == _fyear and _w <= W and _s <= S and _e >= E and _n >= N
                   for (_fy, _w, _s, _e, _n) in fc)
        pf = HOLD.get("prefetch")
        if not poly and pf is not None and pf[0].done() and pf[1] == _fyear \
                and pf[2] <= W and pf[3] <= S and pf[4] >= E and pf[5] >= N:
            poly = True
        if not poly and _fields:
            return False
        if dis:
            f = 4 if _B * k < 120 else 16
            gc = HOLD.get("gcache", {})
            if not any((_fy, _f) == (_fyear, f) and _w <= W and _s <= S and _e >= E and _n >= N
                       for (_fy, _f, _w, _s, _e, _n) in gc):
                return False
        return True

    def _stages(vs):
        """Blocking generator: one or two (url, bounds, legend, line) frames
        for a view. SQL -> `cur` (x, y, class) -> one PNG. On a COLD FTW miss
        the plain pixels are painted first and the fields-clipped (and/or
        disagreement) frame follows when the fetch lands: two swaps of the
        same layer, nothing blank in between."""
        with con_lock:
            W, S, E, N = bbox4326(vs)
            k, x0, y0, x1, y1, drop = _window(vs)
            fields, dis = _fields, _dis
            note = ""
            if (fields or dis) and (E - W) * (N - S) > FTW_BOX_DEG2:
                fields, dis, note = False, False, " · zoom in for FTW"
            if _dis_req and not _ftw_ok:
                note += " · disagreement needs 2024 or 2025"
            _served = HOLD.get("served")
            if (
                _served is not None
                and _served[:7] == (fields, dis, k, _year, _fyear, _crops_only, _sel)
                and W >= _served[7] and S >= _served[8]
                and E <= _served[9] and N <= _served[10]
            ):
                return  # held: the picture already covers this view
            key = (fields, dis, k, _year, _fyear, _crops_only, _sel,
                   round(x0, -3), round(y0, -3), round(x1, -3), round(y1, -3))
            memo = HOLD.setdefault("memo", {})
            hit = memo.get(key)
            if hit is not None:
                HOLD["served"] = (fields, dis, k, _year, _fyear, _crops_only, _sel, W, S, E, N)
                HOLD["box"] = (W, S, E, N)
                HOLD["k"] = k
                url, bounds, legend, line = hit
                yield url, bounds, legend, line + note + " · memo"
                return
            _box = f"t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}"
            _sel_px = _sel_sql.replace(_sel_col, f"t.{_sel_col}")
            _t0 = time.time()

            def _crop_sql(join):
                return (f"SELECT t.x, t.y, t.crop_type, t.crop_type::UTINYINT AS cls "
                        f"FROM {_T}{k} t {join} JOIN classes c ON c.code = t.crop_type "
                        f"WHERE t.year = {_year} AND t.crop_type NOT IN {drop} AND {_box}")

            def _crop_legend():
                _lg = mcon.sql(
                    """SELECT c.code, c.name, c.hex, count(*) AS n
                       FROM cur JOIN classes c ON c.code = cur.crop_type
                       GROUP BY 1, 2, 3 ORDER BY n DESC"""
                ).fetchall()
                _tot = sum(r[3] for r in _lg) or 1
                return [{"code": int(r[0]), "name": r[1], "hex": r[2],
                         "pct": round(100 * r[3] / _tot, 1)} for r in _lg[:14]]

            def _line(npx, nbytes, stage_note):
                return (f"{k}x · {_B * k} m pixels · {npx:,} px · {nbytes / 1e3:,.0f} KB png"
                        f" · year {_year}{stage_note} · {int((time.time() - _t0) * 1000)} ms")

            # ---- stage 1: plain pixels, only when FTW is wanted and COLD (the
            # fetch would otherwise leave the old picture up for seconds)
            cold = (fields or dis) and not _ftw_warm(W, S, E, N, k, dis)
            if cold:
                mcon.sql(f"CREATE OR REPLACE TABLE cur AS {_crop_sql('')}")
                url, bounds, npx, nbytes = _paint(k, x0, y0, x1, y1, W, S, E, N, _sel_px if not dis else "", False, None)
                yield url, bounds, _crop_legend(), _line(npx, nbytes, " · fetching FTW…") + note
                _t0 = time.time()
            # ---- stage 2 (or the only stage): the real frame
            _join, _fbt = "", None
            if fields:
                _fbt, _lkt = _ftw_tables(W, S, E, N, k)
                _join = f"JOIN {_lkt} l USING (y, x)"
            if not dis:
                mcon.sql(f"CREATE OR REPLACE TABLE cur AS {_crop_sql(_join)}")
                legend = _crop_legend()
            else:
                _gt, _res, _f = _ftw_grid(_B * k, W, S, E, N)
                mcon.sql(
                    f"""
                    CREATE OR REPLACE TABLE cur AS
                    WITH g AS (SELECT ix, iy FROM {_gt}),
                    p AS (
                        SELECT t.x, t.y, t.crop_type, c.noncrop,
                               ST_Transform(ST_Point(t.x, t.y), 'EPSG:5070', 'EPSG:4326',
                                            always_xy := true) AS pt
                        FROM {_T}{k} t {_join} JOIN classes c ON c.code = t.crop_type
                        WHERE t.year = {_year} AND t.crop_type NOT IN {drop} AND {_box}
                    )
                    SELECT p.x, p.y, p.crop_type,
                           (CASE WHEN NOT p.noncrop AND g.ix IS NOT NULL THEN 1
                                 WHEN NOT p.noncrop THEN 2
                                 WHEN g.ix IS NOT NULL THEN 3 END)::UTINYINT AS cls
                    FROM p LEFT JOIN g
                      ON floor((ST_X(p.pt) + 180) / {_res})::BIGINT = g.ix
                     AND floor(({FTW_Y0} - ST_Y(p.pt)) / {_res})::BIGINT = g.iy
                    """
                )
                mcon.sql("DELETE FROM cur WHERE cls IS NULL")
                _cnt = dict(mcon.sql("SELECT cls, count(*) FROM cur GROUP BY 1").fetchall())
                _tot = sum(_cnt.values()) or 1
                _pxa = (_B * k / 1000) ** 2 * ACRES_PER_KM2
                legend = [
                    {"code": code, "name": nm, "hex": hx,
                     "pct": round(100 * _cnt.get(code, 0) / _tot, 1),
                     "note": f"{_cnt.get(code, 0) * _pxa / 1e3:,.1f}k ac"}
                    for code, (nm, hx, _c3) in DIS.items()
                ]
                HOLD["dis_split"] = {code: 100 * _cnt.get(code, 0) / _tot for code in DIS}
            url, bounds, npx, nbytes = _paint(k, x0, y0, x1, y1, W, S, E, N, _sel_px, dis, _fbt)
            line = _line(npx, nbytes,
                         (f" · FTW {_fyear} fields" if fields else "")
                         + (f" · disagreement vs P(field) at {10 * _f} m" if dis else ""))
            memo[key] = (url, bounds, legend, line)
            if len(memo) > 24:
                memo.pop(next(iter(memo)))
            HOLD["served"] = (fields, dis, k, _year, _fyear, _crops_only, _sel, W, S, E, N)
            HOLD["box"] = (W, S, E, N)
            HOLD["k"] = k
            yield url, bounds, legend, line + note

    async def _refresh(vs, force=False):
        if HOLD.get("busy"):
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                if not force and SETTLE > 0:
                    await asyncio.sleep(SETTLE)
                    if HOLD.get("pending") is not None:
                        vs, HOLD["pending"] = HOLD["pending"], None
                        continue
                painted = False
                for url, bounds, _legend, _line in _stages(vs):
                    if HOLD.get("pending") is not None:
                        HOLD["served"] = None
                        break
                    try:
                        hud.widget.legend = json.dumps(_legend)
                    except Exception:
                        pass
                    with pixels.hold_sync():
                        pixels.image = url
                        pixels.bounds = bounds
                    HOLD["last_line"] = _line
                    _say(_line)
                    painted = True
                    await asyncio.sleep(0)   # let the swap go out before stage 2
                if not painted and HOLD.get("pending") is None:
                    _say((HOLD.get("last_line") or "") + " · held")
                vs, force = HOLD.get("pending"), False
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as _e:
            _say(f"serve error: {type(_e).__name__}: {_e}")
        finally:
            HOLD["busy"] = False
            HOLD["pending"] = None

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            _loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, _loop) if _loop else None

    def _on_vs(change):
        try:
            vs = _vsd(change.new)
            if vs is None:
                return
            HOLD["vs"] = vs
            _say("camera…")
            HOLD["task"] = _spawn(_refresh(vs))
        except Exception as _e:
            _say(f"camera error: {type(_e).__name__}: {_e}")

    _old = HOLD.get("h_vs")
    if _old is not None:
        try:
            deck.unobserve(_old, names="view_state")
        except Exception:
            pass
    deck.observe(_on_vs, names="view_state")
    HOLD["h_vs"] = _on_vs

    # ---- "analyze what's in view": the crops notebook's panel (top crops in
    # the box + the 18-year timelapse), UNDER THE MASK when it is on, plus a
    # mode line (field count and purity / the disagreement split) -------------
    def _analyze_html(vs):
        def _timelapse_svg(top, tl, px_km2, k, tl_ms, masked):
            if not tl:
                return ""
            years = sorted({r[0] for r in tl})
            by = {(r[0], r[1]): r[2] for r in tl}
            series = []
            for nm, hx, _n, code in top:
                vals = [by.get((y, code), 0) * px_km2 * ACRES_PER_KM2 / 1e6 for y in years]
                series.append((nm, hx, vals))
            vmax = max((v for _, _, vals in series for v in vals), default=0) or 1
            Wd, H, L, R, T, B = 640, 150, 62, 150, 8, 18
            def sx(i):
                return L + (Wd - L - R) * (i / max(len(years) - 1, 1))
            def sy(v):
                return T + (H - T - B) * (1 - v / vmax)
            parts = [
                f'<div style="margin-top:6px"><svg viewBox="0 0 {Wd} {H}" '
                f'style="max-width:{Wd}px;width:100%;display:block;font:10px '
                'ui-sans-serif,system-ui,sans-serif">'
            ]
            parts.append(
                f'<line x1="{L}" y1="{sy(0)}" x2="{Wd - R}" y2="{sy(0)}" '
                'stroke="currentColor" stroke-opacity=".25"/>'
                f'<line x1="{L}" y1="{sy(vmax)}" x2="{Wd - R}" y2="{sy(vmax)}" '
                'stroke="currentColor" stroke-opacity=".08"/>'
                f'<text x="{L - 4}" y="{sy(vmax) + 3}" text-anchor="end" '
                f'fill="currentColor" fill-opacity=".6">{vmax:.2f}M ac</text>'
                f'<text x="{L}" y="{H - 4}" fill="currentColor" '
                f'fill-opacity=".6">{years[0]}</text>'
                f'<text x="{sx(len(years) - 1)}" y="{H - 4}" text-anchor="end" '
                f'fill="currentColor" fill-opacity=".6">{years[-1]}</text>'
            )
            _ends = []
            for nm, hx, vals in series:
                pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{hx}" '
                    'stroke-width="2" stroke-linejoin="round"/>'
                )
                _ends.append([sy(vals[-1]), nm, hx])
            _ends.sort()
            for _i in range(1, len(_ends)):
                if _ends[_i][0] - _ends[_i - 1][0] < 11:
                    _ends[_i][0] = _ends[_i - 1][0] + 11
            for _y, nm, hx in _ends:
                parts.append(
                    f'<text x="{Wd - R + 5}" y="{min(max(_y, T + 8), H - B) + 3:.1f}" '
                    f'fill="currentColor" fill-opacity=".85">{nm[:22]}</text>'
                )
            parts.append("</svg>")
            parts.append(
                f'<div style="opacity:.5;font-size:11px">acres by year in view'
                f'{" inside today&#39;s FTW fields" if masked else ""} · '
                f"30 m group at {k}x · timelapse query {tl_ms} ms</div></div>"
            )
            return "".join(parts)

        with con_lock:
            W, S, E, N = bbox4326(vs)
            k, x0, y0, x1, y1, drop = _window(vs)
            fields = _fields and (E - W) * (N - S) <= FTW_BOX_DEG2
            _join = ""
            if fields:
                _fbt, _lkt = _ftw_tables(W, S, E, N, k)
                _join = f"JOIN {_lkt} l USING (y, x)"
            _sel_px = _sel_sql.replace("crop_type", "t.crop_type") if not _dis else ""
            rows = mcon.sql(
                f"""
                SELECT c.name, c.hex, count(*) AS n, c.code
                FROM {_T}{k} t {_join} JOIN classes c ON c.code = t.crop_type
                WHERE t.year = {_year}
                  AND t.crop_type NOT IN {drop}{_sel_px}
                  AND t.x BETWEEN {x0} AND {x1}
                  AND t.y BETWEEN {y0} AND {y1}
                GROUP BY 1, 2, 4 ORDER BY n DESC LIMIT 10
                """
            ).fetchall()
            # the timelapse is 18 years of the 30 m group (the 10 m group has
            # only 2024-2025), at the 30 m level nearest this serve's pixel,
            # under the same fields clip (its own lookup at that level)
            _k30 = max(1, min(256, 2 ** round(math.log2(max(_B * k / 30, 1)))))
            _join30 = ""
            if fields:
                _fbt30, _lkt30 = _ftw_tables(W, S, E, N, _k30) if _B == 30 else \
                    _ftw_tables_at("cdl_", _k30, W, S, E, N)
                _join30 = f"JOIN {_lkt30} l USING (y, x)"
            _t_tl = time.time()
            _tl_codes = [r[3] for r in rows[:6]]
            tl = mcon.sql(
                f"""
                SELECT t.year, t.crop_type, count(*) AS n
                FROM cdl_{_k30} t {_join30}
                WHERE t.crop_type IN ({", ".join(str(c) for c in _tl_codes)})
                  AND t.x BETWEEN {x0} AND {x1}
                  AND t.y BETWEEN {y0} AND {y1}
                GROUP BY 1, 2 ORDER BY 1
                """
            ).fetchall() if _tl_codes else []
            tl_ms = int((time.time() - _t_tl) * 1000)
            extra = ""
            if _dis and HOLD.get("dis_split"):
                extra = "disagreement in view: " + " · ".join(
                    f"{nm} {HOLD['dis_split'].get(code, 0):.0f}%"
                    for code, (nm, _hx, _c3) in DIS.items()
                )
            if fields:
                _nf = mcon.sql(
                    f"""SELECT count(*), sum(area_m2) / 4046.8564 FROM {_fbt} f
                        WHERE ST_XMin(f.geometry) < {E} AND ST_XMax(f.geometry) > {W}
                          AND ST_YMin(f.geometry) < {N} AND ST_YMax(f.geometry) > {S}"""
                ).fetchone()
                extra = (f"{_nf[0]:,} FTW {_fyear} fields · {(_nf[1] or 0) / 1e3:,.1f}k acres"
                         + (" · " + extra if extra else ""))
        mode = "fields" if fields else "off"
        total = sum(r[2] for r in rows) or 1
        px_km2 = (_B * k / 1000) ** 2
        out = [
            f'<span style="opacity:.65;margin-right:.9rem;white-space:nowrap">'
            f"in view · {k}x ({_B * k} m) · year {_year}"
            f"{' · inside FTW ' + str(_fyear) + ' fields' if fields else ''}"
            f" · approx (majority pyramid)</span>"
        ]
        for nm, hx, n, _code in rows:
            macres = n * px_km2 * ACRES_PER_KM2 / 1e6
            amt = f"{macres:.2f} M ac" if macres >= 0.01 else f"{macres * 1000:.1f} k ac"
            out.append(
                '<span style="display:inline-block;margin:2px .9rem 2px 0;white-space:nowrap">'
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                f'background:{hx};margin-right:5px;vertical-align:-1px"></span>{nm} '
                f'<span style="opacity:.8;font-variant-numeric:tabular-nums">'
                f"{amt} · {100 * n / total:.0f}%</span></span>"
            )
        if not rows:
            out.append('<span style="opacity:.6">nothing in view</span>')
            return "".join(out)
        if extra:
            out.append(f'<div style="opacity:.85;margin-top:4px">{extra}</div>')
        out.append(_timelapse_svg(rows[:6], tl, (0.03 * _k30) ** 2, _k30, tl_ms, fields))
        return "".join(out)

    async def _do_analyze():
        try:
            vs = _vsd(HOLD.get("vs")) or dict(HOME)
            html = await asyncio.get_running_loop().run_in_executor(
                None, _analyze_html, vs
            )
            hud.widget.panel = html
        except Exception as _e:
            hud.widget.panel = (
                f'<span style="opacity:.8">analyze error: {type(_e).__name__}: {_e}</span>'
            )

    if _act == "analyze":
        HOLD["atask"] = _spawn(_do_analyze())

    # ---- the search field: Photon, camera-biased, fly_to the first hit ------
    def _photon_first(query, vs):
        _params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            _params["lon"] = round(vs["longitude"], 4)
            _params["lat"] = round(vs["latitude"], 4)
        _url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(_params)
        _req = urllib.request.Request(
            _url, headers={"User-Agent": "x-sql-marimo cdl fields notebook"}
        )
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _data = json.load(_r)
        _feats = _data.get("features") or []
        if not _feats:
            return None
        _f = _feats[0]
        _p = _f.get("properties", {})
        _lon, _lat = _f["geometry"]["coordinates"][:2]
        _name = ", ".join(
            str(v) for v in (_p.get("name"), _p.get("city"), _p.get("state")) if v
        ) or query
        return _name, _lon, _lat, _p.get("extent")

    async def _do_search():
        try:
            _hit = await asyncio.get_running_loop().run_in_executor(
                None, _photon_first, _q, _vsd(HOLD.get("vs"))
            )
            if _hit is None:
                _say(f"no match: {_q}")
                return
            _name, _lon, _lat, _ext = _hit
            if _ext and len(_ext) == 4:
                _span = max(abs(_ext[2] - _ext[0]), abs(_ext[1] - _ext[3]) * 2, 0.01)
                _zoom = math.log2(360.0 * (VIEW_W / 512) / _span) - 0.3
            else:
                _zoom = 10.0
            _zoom = max(3.5, min(13.5, _zoom))
            _vs = {"longitude": _lon, "latitude": _lat, "zoom": _zoom}
            HOLD["vs"] = _vs
            deck.fly_to(longitude=_lon, latitude=_lat, zoom=_zoom, duration=2000)
            _say(f"→ {_name}")
            HOLD["stask"] = _spawn(_refresh(_vs))
        except Exception as _e:
            _say(f"search error: {type(_e).__name__}: {_e}")

    if _act == "search" and _q:
        HOLD["stask0"] = _spawn(_do_search())

    # an analyze click must not repaint the map; only a set commit or the
    # first run serves
    if _act not in ("analyze", "search") or "k" not in HOLD:
        HOLD["task0"] = _spawn(_refresh(_vsd(HOLD.get("vs")) or dict(HOME), force=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The same joins, as plain SQL on the box in view

    The map above runs these on its own connection; here they are as statements
    on `con`, each leaving a table for the next. Press the button after moving
    the map (the strip's year applies; fields and disagreement use 2024 or 2025).
    """)
    return


@app.cell
def _(mo):
    go = mo.ui.run_button(label="run the SQL below on the box in view")
    go
    return (go,)


@app.cell
def _(FTW_YEARS, HOLD: dict, HOME, bbox4326, con, ftw_files, go, mo, to5070):
    # the analysis box: the last served (padded) camera box, at click time; the
    # opening view before any click. `go` is only the trigger.
    _ = go.value
    W, S, E, N = HOLD.get("box") or bbox4326(HOME)
    x0, y0, x1, y1 = to5070(con, W, S, E, N)
    # the cells below are same-year joins, so they read FTW's years from CDL's
    # 10 m group: finest level with <= ~1.5M pixel centres in the box
    CDL_YEAR = int(HOLD.get("year", 2025))
    FTW_YEAR = CDL_YEAR if CDL_YEAR in FTW_YEARS else FTW_YEARS[0]
    T, B = "cdl10_", 10
    K = next((k for k in (1, 2, 4, 8, 16) if (x1 - x0) * (y1 - y0) / (B * k) ** 2 <= 1.5e6), 32)
    PX_KM2 = (B * K / 1000) ** 2
    FILES = ftw_files(con, W, S, E, N)
    FILES_SQL = ", ".join(f"'{f}'" for f in FILES)
    mo.md(
        f"box **{W:.3f}, {S:.3f} → {E:.3f}, {N:.3f}** · CDL {FTW_YEAR} at "
        f"{B * K} m (`{T}{K}`, the 10 m group) · FTW {FTW_YEAR} from "
        f"{', '.join(f.rsplit('/', 1)[-1] for f in FILES)}"
        + (f" · the strip is on {CDL_YEAR}; FTW has no {CDL_YEAR}" if CDL_YEAR != FTW_YEAR else "")
    )
    return E, FILES_SQL, FTW_YEAR, K, N, PX_KM2, S, T, W, x0, x1, y0, y1


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1. The fields: `read_parquet` on the state file(s), pruned by the `bbox` struct
    """)
    return


@app.cell
def _(E, FILES_SQL, FTW_YEAR, N, S, W, con, mo):
    fields_view = mo.sql(
        f"""
        CREATE OR REPLACE TABLE fields_view AS
        SELECT id, "metrics:area" AS area_m2, geometry::GEOMETRY AS geometry
        FROM read_parquet([{FILES_SQL}])
        WHERE bbox.xmin > {W} AND bbox.xmax < {E}
          AND bbox.ymin > {S} AND bbox.ymax < {N}
          AND date_part('year', "determination:datetime" AT TIME ZONE 'UTC') = {FTW_YEAR}
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    fields_summary = mo.sql(
        """
        SELECT count(*) AS fields,
               round(sum(area_m2) / 4046.8564, 0) AS acres,
               round(quantile_cont(area_m2, 0.5) / 4046.8564, 1) AS median_acres,
               round(max(area_m2) / 4046.8564, 0) AS largest_acres
        FROM fields_view
        """,
        engine=con,
    )
    fields_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2. Pixel -> field, once: `ST_Contains` of the CDL pixel centres into the polygons
    """)
    return


@app.cell
def _(K, T, con, mo, x0, x1, y0, y1):
    px2field = mo.sql(
        f"""
        CREATE OR REPLACE TABLE px2field AS
        WITH p AS (
            SELECT DISTINCT y, x,
                   ST_Transform(ST_Point(x, y), 'EPSG:5070', 'EPSG:4326',
                                always_xy := true) AS pt
            FROM {T}{K}
            WHERE year = 2025
              AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1}
        )
        SELECT f.id, p.y, p.x
        FROM fields_view f JOIN p ON ST_Contains(f.geometry, p.pt)
        """,
        engine=con
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3. The crop of every field (same year as the polygons): majority class and purity
    """)
    return


@app.cell
def _(FTW_YEAR, K, T, con, mo, x0, x1, y0, y1):
    field_crop = mo.sql(
        f"""
        CREATE OR REPLACE TABLE field_crop AS
        WITH j AS (
            SELECT l.id, t.crop_type, count(*) AS n
            FROM {T}{K} t JOIN px2field l USING (y, x)
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
            GROUP BY 1, 2
        ),
        m AS (
            SELECT id, crop_type, n,
                   sum(n) OVER (PARTITION BY id) AS tot,
                   row_number() OVER (PARTITION BY id ORDER BY n DESC, crop_type) AS rn
            FROM j
        )
        SELECT id, crop_type, n AS px, tot AS px_total, n / tot AS purity
        FROM m WHERE rn = 1
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    crop_by_field = mo.sql(
        """
        SELECT c.name AS crop, count(*) AS fields,
               round(sum(f.area_m2) / 4046.8564, 0) AS acres,
               round(quantile_cont(f.area_m2, 0.5) / 4046.8564, 1) AS median_field_acres,
               round(avg(fc.purity), 2) AS mean_purity
        FROM field_crop fc JOIN fields_view f USING (id) JOIN classes c ON c.code = fc.crop_type
        GROUP BY 1 ORDER BY fields DESC LIMIT 15
        """,
        engine=con,
    )
    crop_by_field
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4. Purity: the least pure fields with their top two crops (two fields FTW merged, or CDL noise inside one)
    """)
    return


@app.cell
def _(FTW_YEAR, K, T, con, mo, x0, x1, y0, y1):
    mixed_fields = mo.sql(
        f"""
        WITH j AS (
            SELECT l.id, t.crop_type, count(*) AS n
            FROM {T}{K} t JOIN px2field l USING (y, x)
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
            GROUP BY 1, 2
        ),
        r AS (
            SELECT id, crop_type, n,
                   sum(n) OVER (PARTITION BY id) AS tot,
                   row_number() OVER (PARTITION BY id ORDER BY n DESC, crop_type) AS rn
            FROM j
        )
        SELECT a.id,
               round(f.area_m2 / 4046.8564, 1) AS acres,
               ca.name AS top_crop, round(a.n / a.tot, 2) AS share,
               cb.name AS second_crop, round(b.n / b.tot, 2) AS share_2
        FROM r a JOIN r b ON a.id = b.id AND b.rn = 2
        JOIN fields_view f ON f.id = a.id
        JOIN classes ca ON ca.code = a.crop_type
        JOIN classes cb ON cb.code = b.crop_type
        WHERE a.rn = 1 AND a.tot >= 40
        ORDER BY a.n / a.tot ASC LIMIT 20
        """,
        engine=con,
    )
    mixed_fields
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5. Disagreement: CDL crop / not-crop against FTW P(field) >= 0.5 from `ftw_4`

    Each CDL pixel centre is binned into its 40 m FTW cell by index arithmetic.
    Then the crops FTW most often misses, and the non-crop classes FTW calls fields.
    """)
    return


@app.cell
def _(E, FTW_RES, FTW_Y0, FTW_YEAR, K, N, S, T, W, con, mo, x0, x1, y0, y1):
    _res = FTW_RES * 4
    agreement = mo.sql(
        f"""
        CREATE OR REPLACE TABLE agreement AS
        WITH fp AS (
            SELECT floor((x + 180) / {_res})::BIGINT AS ix,
                   floor(({FTW_Y0} - y) / {_res})::BIGINT AS iy,
                   variables >= 0.5 AS is_field
            FROM ftw_4
            WHERE time = TIMESTAMP '{FTW_YEAR}-01-01' AND band = 'field'
              AND x BETWEEN {W} AND {E} AND y BETWEEN {S} AND {N}
        ),
        cp AS (
            SELECT t.crop_type, NOT c.noncrop AS is_crop, c.name,
                   ST_Transform(ST_Point(t.x, t.y), 'EPSG:5070', 'EPSG:4326',
                                always_xy := true) AS pt
            FROM {T}{K} t JOIN classes c ON c.code = t.crop_type
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
        )
        SELECT cp.crop_type, cp.name, cp.is_crop, fp.is_field, count(*) AS px
        FROM cp JOIN fp
          ON floor((ST_X(cp.pt) + 180) / {_res})::BIGINT = fp.ix
         AND floor(({FTW_Y0} - ST_Y(cp.pt)) / {_res})::BIGINT = fp.iy
        GROUP BY 1, 2, 3, 4
        """,
        engine=con,
    )
    return


@app.cell
def _(PX_KM2, con, mo):
    two_by_two = mo.sql(
        f"""
        SELECT CASE WHEN is_crop THEN 'CDL crop' ELSE 'CDL not crop' END AS cdl,
               CASE WHEN is_field THEN 'FTW field' ELSE 'FTW not field' END AS ftw,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(px) / sum(sum(px)) OVER (), 1) AS pct
        FROM agreement GROUP BY 1, 2 ORDER BY 1, 2
        """,
        engine=con,
    )
    two_by_two
    return


@app.cell
def _(PX_KM2, con, mo):
    ftw_misses = mo.sql(
        f"""
        SELECT name AS cdl_crop,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(CASE WHEN is_field THEN px ELSE 0 END) / sum(px), 1)
                   AS pct_ftw_field
        FROM agreement WHERE is_crop
        GROUP BY 1 HAVING sum(px) >= 200
        ORDER BY pct_ftw_field ASC, k_acres DESC LIMIT 15
        """,
        engine=con,
    )
    ftw_false_fields = mo.sql(
        f"""
        SELECT name AS cdl_noncrop,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(CASE WHEN is_field THEN px ELSE 0 END) / sum(px), 1)
                   AS pct_ftw_field
        FROM agreement WHERE NOT is_crop
        GROUP BY 1 HAVING sum(px) >= 200
        ORDER BY pct_ftw_field DESC, k_acres DESC LIMIT 15
        """,
        engine=con,
    )
    mo.hstack([ftw_misses, ftw_false_fields], widths="equal", gap=2)
    return


if __name__ == "__main__":
    app.run()
