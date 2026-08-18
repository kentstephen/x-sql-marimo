"""Full-scan throughput of xarray-sql's DataFusion path on a synthetic HRRR-shaped
cube: (168 h, 500, 500) float32 = 42M rows. Root venv (0.3.x) vs this venv (rc)."""
import time, importlib.metadata as im
import numpy as np, xarray as xr, xarray_sql as xql

T, Y, X = 168, 500, 500
rng = np.random.default_rng(0)
ds = xr.Dataset({"v": (("t", "y", "x"), rng.random((T, Y, X), dtype=np.float32))},
                coords={"t": np.arange(T), "y": np.arange(Y), "x": np.arange(X)})
ctx = xql.XarrayContext()
ctx.from_dataset("cube", ds, chunks={"t": T, "y": 45, "x": 45})   # the notebook's block shape
print("xarray-sql", im.version("xarray-sql"), "rows", T * Y * X)
for q in ["SELECT count(*) FROM cube",
          "SELECT avg(v) FROM cube",
          "SELECT t, avg(v) FROM cube GROUP BY t"]:
    for i in range(2):
        t0 = time.perf_counter(); ctx.sql(q).collect(); dt = time.perf_counter() - t0
        print(f"{q:42s} run{i}: {dt:6.2f} s  {T*Y*X/dt/1e6:6.1f} M rows/s")
