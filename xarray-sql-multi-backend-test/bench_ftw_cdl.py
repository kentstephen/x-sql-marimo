import time, warnings; warnings.filterwarnings("ignore")
import duckdb, icechunk, xarray as xr, numpy as np, zarr, obstore
from obstore.store import S3Store
import xarray_sql as xql
T0=time.time()
def lap(msg):
    global T0; t=time.time(); print(f"{msg}: {t-T0:.1f}s", flush=True); T0=t

# --- CDL native 30m on duckdb
storage = icechunk.s3_storage(bucket="chill", prefix="usda-cropland-data-layer/v0.1.0.icechunk",
    endpoint_url="https://data.source.coop", region="us-east-1", anonymous=True, force_path_style=True)
sess = icechunk.Repository.open(storage).readonly_session("main")
ds = xr.open_zarr(sess.store, group="30m", chunks=None)
con = duckdb.connect(); con.sql("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; SET s3_url_style='path';")
xql.register(con, "cdl_1", ds, chunks={"year":1,"y":2048,"x":2048})
lap("open cdl")

# window: Fresno 20x20 km in 4326
W,S,E,N = -119.9,36.6,-119.7,36.8
x0,y0,x1,y1 = con.sql(f"SELECT ST_XMin(g),ST_YMin(g),ST_XMax(g),ST_YMax(g) FROM (SELECT ST_Transform(ST_MakeEnvelope({W},{S},{E},{N}),'EPSG:4326','EPSG:5070', always_xy:=true) g)").fetchone()
print("5070 box", x0,y0,x1,y1)

# --- FTW fields
p = "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/predictions/vectors/alpha/results-by-admin-conf/admin:country_code=US/US_CA.parquet"
con.sql(f"""CREATE TABLE fields AS SELECT id, date_part('year', "determination:datetime" AT TIME ZONE 'UTC') AS yr, "metrics:area" area, confidence, geometry g
  FROM read_parquet('{p}') WHERE bbox.xmin>{W} AND bbox.xmax<{E} AND bbox.ymin>{S} AND bbox.ymax<{N}""")
print(con.sql("SELECT yr, count(*), round(avg(area)/4046.86,1) acres, count(confidence) FROM fields GROUP BY 1 ORDER BY 1").fetchall())
lap("ftw fields")

# --- CDL pixels in window, 2024 -> 4326 points
con.sql(f"""CREATE TABLE px AS SELECT ST_Transform(ST_Point(x, y),'EPSG:5070','EPSG:4326', always_xy:=true) pt, crop_type
  FROM cdl_1 WHERE year=2024 AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1} AND crop_type NOT IN (0,81)""")
print("px", con.sql("SELECT count(*) FROM px").fetchone())
lap("cdl pixels 2024 native")

# --- join: majority crop per field (2024 fields)
con.sql("""CREATE TABLE fc AS
  WITH j AS (SELECT f.id, p.crop_type, count(*) n FROM fields f JOIN px p ON ST_Contains(f.g, p.pt) WHERE f.yr=2024 GROUP BY 1,2),
  m AS (SELECT id, crop_type, n, sum(n) OVER (PARTITION BY id) tot, row_number() OVER (PARTITION BY id ORDER BY n DESC) rn FROM j)
  SELECT id, crop_type, n, tot, n/tot purity FROM m WHERE rn=1""")
print(con.sql("SELECT count(*), avg(purity), quantile_cont(purity,0.5) FROM fc").fetchall())
print(con.sql("SELECT crop_type, count(*) FROM fc GROUP BY 1 ORDER BY 2 DESC LIMIT 8").fetchall())
lap("join majority crop per field")

# --- FTW probabilities zarr, same window, root + 4x
base = S3Store(bucket="us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True,
               prefix="tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/")
zs = zarr.storage.ObjectStore(base, read_only=True)
for grp in [".", "4x", "16x"]:
    t=time.time()
    g = zarr.open_group(zs, path=None if grp=="." else grp, mode="r")
    a = g["variables"]
    tr = g.attrs["spatial:transform"] if grp=="." else None
    res = 8.98311982e-05 * (1 if grp=="." else int(grp[:-1]))
    ix0 = int((W+180)/res); ix1=int((E+180)/res); iy0=int((83.748345-N)/res); iy1=int((83.748345-S)/res)
    arr = a[0, :, iy0:iy1, ix0:ix1]  # time 2024, 3 bands
    print(grp, arr.shape, "nan frac", np.isnan(arr).mean().round(3), "field p mean", np.nanmean(arr[1]).round(3), f"{time.time()-t:.1f}s")
lap("zarr reads")
