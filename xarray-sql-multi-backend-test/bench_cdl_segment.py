"""Go/no-go benchmark for the CDL segmentation plan (docs/cdl-crops-notes.md,
"Segment the pixels with DuckDB", route 1): ST_Union_Agg(ST_MakeEnvelope(...))
per class over a viewport's pixels, at realistic serve sizes.

Real store, real windows. Two sites (Iowa corn/soy = few classes, CA Central
Valley = many), three sizes (~50k / 200k / 400k squares), native 30 m level.
The window is materialized first so the union is timed apart from the fetch;
the fetch time is the serve cost the notebook already pays.

Also timed: ST_Dump blob identity (the heat-domes dome-table move), the single
ST_Transform of each union to 4326, and the isolated-class variant (top-2
classes only, the pickable-legend pairing). Prior to beat: the repo measured
ST_Union_Agg 30x slower than alternatives on HEXAGONS; squares may differ.

Run: uv run --project xarray-sql-multi-backend-test python xarray-sql-multi-backend-test/bench_cdl_segment.py
"""
import time
import duckdb
import icechunk
import xarray as xr
import xarray_sql as xql

BUCKET = "chill"
PREFIX = "usda-cropland-data-layer/v0.1.0.icechunk"
ENDPOINT = "https://data.source.coop"
YEAR = 2025
K = 1                      # native 30 m; union cost scales with rows, not level
HALF = 15 * K

storage = icechunk.s3_storage(
    bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT,
    region="us-east-1", anonymous=True, force_path_style=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, group="30m", chunks=None)

con = duckdb.connect()
con.sql("INSTALL spatial; LOAD spatial;")
xql.register(con, "cdl_1", ds, chunks={"year": 1, "y": 2048, "x": 2048})

SITES = {
    "iowa (few classes)": (42.0, -93.5),
    "central valley (many)": (36.7, -119.8),
}
TARGETS = [50_000, 200_000, 400_000]

def to5070(lat, lon):
    return con.sql(
        f"""SELECT ST_X(p), ST_Y(p) FROM (
              SELECT ST_Transform(ST_Point({lon}, {lat}), 'EPSG:4326',
                     'EPSG:5070', always_xy := true) AS p)"""
    ).fetchone()

for site, (lat, lon) in SITES.items():
    cx, cy = to5070(lat, lon)
    for target in TARGETS:
        # square box sized to the target count of 30*K m pixels
        side = (target ** 0.5) * 30 * K
        x0, x1 = cx - side / 2, cx + side / 2
        y0, y1 = cy - side / 2, cy + side / 2

        t0 = time.perf_counter()
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE win AS
                SELECT x, y, crop_type FROM cdl_{K}
                WHERE year = {YEAR} AND crop_type NOT IN (0, 81)
                  AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1}"""
        )
        t_fetch = time.perf_counter() - t0
        n, ncls = con.sql(
            "SELECT count(*), count(DISTINCT crop_type) FROM win"
        ).fetchone()

        t0 = time.perf_counter()
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE u AS
                SELECT crop_type,
                       ST_Union_Agg(ST_MakeEnvelope(x-{HALF}, y-{HALF},
                                                    x+{HALF}, y+{HALF})) AS g
                FROM win GROUP BY crop_type"""
        )
        t_union = time.perf_counter() - t0

        t0 = time.perf_counter()
        nblob = con.sql(
            "SELECT count(*) FROM (SELECT UNNEST(ST_Dump(g), recursive := true) FROM u)"
        ).fetchone()[0]
        t_dump = time.perf_counter() - t0

        t0 = time.perf_counter()
        nbytes = con.sql(
            """SELECT sum(octet_length(ST_AsWKB(
                 ST_Transform(g, 'EPSG:5070', 'EPSG:4326', always_xy := true))))
               FROM u"""
        ).fetchone()[0]
        t_xf = time.perf_counter() - t0

        # isolated-class variant on the same window (legend pairing): top 2
        t0 = time.perf_counter()
        con.execute(
            """CREATE OR REPLACE TEMP TABLE u2 AS
               SELECT crop_type,
                      ST_Union_Agg(ST_MakeEnvelope(x-15, y-15, x+15, y+15)) AS g
               FROM win
               WHERE crop_type IN (SELECT crop_type FROM win
                                   GROUP BY 1 ORDER BY count(*) DESC LIMIT 2)
               GROUP BY crop_type"""
        )
        t_iso = time.perf_counter() - t0

        print(
            f"{site:22s} target {target//1000:>3}k -> {n:>7,} rows "
            f"{ncls:>2} cls | fetch {t_fetch:5.2f}s  UNION {t_union:6.2f}s  "
            f"dump {t_dump:5.2f}s ({nblob:,} blobs)  xform {t_xf:5.2f}s "
            f"({(nbytes or 0)/1e6:.1f} MB wkb)  iso2 {t_iso:5.2f}s",
            flush=True,
        )
