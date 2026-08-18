"""The fold, same synthetic cube on both engines via the rc's engine-neutral register:
h3_latlng_to_cell(lat, lon, res) in the GROUP BY, h3ronpy UDF on DataFusion vs
duckdb-h3 on DuckDB. Lookup (y, x, lat, lon) joined in, as the heat hex notebook does."""
import time
import numpy as np, pyarrow as pa, xarray as xr, xarray_sql as xql, duckdb, datafusion
from datafusion import udf
from h3ronpy.vector import coordinates_to_cells

T, Y, X, RES = 24, 500, 500, 6
rng = np.random.default_rng(0)
ds = xr.Dataset({"v": (("t", "y", "x"), rng.random((T, Y, X), dtype=np.float32))},
                coords={"t": np.arange(T), "y": np.arange(Y), "x": np.arange(X)})
yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")
lut = pa.table({"y": yy.ravel(), "x": xx.ravel(),
                "lat": np.linspace(25, 50, Y)[yy.ravel()], "lon": np.linspace(-125, -67, X)[xx.ravel()]})
Q = ("SELECT t, h3_latlng_to_cell(lat, lon, {r}) AS cell, avg(v) AS v "
     "FROM cube JOIN lut USING (y, x) GROUP BY 1, 2")
print("rows", T * Y * X, "res", RES)

def h3_udf(lat, lng, res):
    return pa.array(coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py()), type=pa.uint64())
ctx = xql.XarrayContext()
ctx.register_udf(udf(h3_udf, [pa.float64(), pa.float64(), pa.int32()], pa.uint64(), "stable", name="h3_latlng_to_cell"))
ctx.from_arrow(lut, name="lut")
ctx.from_dataset("cube", ds, chunks={"t": T, "y": 45, "x": 45})
for i in range(2):
    t0 = time.perf_counter(); n = ctx.sql(Q.format(r=f"CAST({RES} AS INT)")).collect(); dt = time.perf_counter() - t0
    print(f"datafusion + h3ronpy  run{i}: {dt:6.2f} s  rows out {sum(len(b) for b in n)}")

con = duckdb.connect()
con.execute("INSTALL h3 FROM community; LOAD h3;")
con.register("lut", lut)
xql.register(con, "cube", ds, chunks={"t": T, "y": 45, "x": 45})
for i in range(2):
    t0 = time.perf_counter(); n = con.sql(Q.format(r=RES)).arrow().read_all().num_rows; dt = time.perf_counter() - t0
    print(f"duckdb + duckdb-h3    run{i}: {dt:6.2f} s  rows out {n}")
