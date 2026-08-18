"""Smoke test: one lazy xarray Dataset registered on DuckDB and on DataFusion via
xarray-sql 0.4.0rc1's engine-neutral `register`, same SQL on both.

    uv run python smoke_test.py
"""
import importlib.metadata as im

import datafusion
import duckdb
import numpy as np
import xarray as xr
import xarray_sql as xql

print("xarray-sql", im.version("xarray-sql"), "| duckdb", duckdb.__version__,
      "| datafusion", datafusion.__version__)

rng = np.random.default_rng(0)
ds = xr.Dataset(
    {"t2m": (("time", "lat", "lon"), rng.random((24, 10, 20)).astype("float32"))},
    coords={"time": np.arange(24), "lat": np.linspace(30, 40, 10),
            "lon": np.linspace(-110, -100, 20)},
).chunk({"time": 12, "lat": 5, "lon": 10})

Q = "SELECT count(*) AS n, avg(t2m) AS mean FROM cube WHERE lat > 35"

con = duckdb.connect()
xql.register(con, "cube", ds)              # pushdown adapter (pyarrow.dataset)
print("duckdb    :", con.sql(Q).fetchall())

ctx = datafusion.SessionContext()
xql.register(ctx, "cube", ds)              # the existing native table provider
print("datafusion:", ctx.sql(Q).to_pylist())

# round trip: any engine's Arrow result back to a labelled Dataset
back = xql.to_dataset(con.sql("SELECT time, lat, lon, t2m FROM cube WHERE lat > 35"),
                      template=ds)
print("round trip:", dict(back.sizes))
