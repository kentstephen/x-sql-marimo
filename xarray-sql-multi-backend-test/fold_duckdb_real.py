"""Viability: the heat hex fold on the REAL cube through the rc's DuckDB backend.
Runs the patched notebook (young chunk, two variables) to get cube_all/pix2h/land_pred
and the DataFusion fold time for comparison, then registers the same window on DuckDB
with xql.register (pushdown dataset) and runs the same fold SQL with duckdb-h3's
h3_latlng_to_cell in the GROUP BY."""
import importlib.util, os, re, sys, time
import numpy as np, pyarrow as pa, duckdb, xarray_sql as xql

OUT = "/private/tmp/claude-501/-Users-stephenk-dev-projects-x-sql-marimo/996c8bf4-050a-4512-a409-57d0c7c02802/scratchpad"
src = open("../xsql-hrrr-heat-hex.py").read()
src = re.sub(r'\n    DAYS = \(.*?\)   # East dome.*', '\n    DAYS = 7', src)
src = src.replace("READ_RAIN = True", "READ_RAIN = False").replace("READ_WIND = True", "READ_WIND = False")
p = os.path.join(OUT, "heathex_patched.py"); open(p, "w").write(src)
spec = importlib.util.spec_from_file_location("heathex", p)
mod = importlib.util.module_from_spec(spec); sys.modules["heathex"] = mod; spec.loader.exec_module(mod)
t0 = time.perf_counter(); outputs, d = mod.app.run(); print(f"notebook {time.perf_counter()-t0:.0f} s"); print(d["fold_stats"])

cube = d["cube_all"].sel(t=slice(d["t0"], d["t1"]))
hours = int(cube.sizes["t"]); RES = int(d["RES"])
con = duckdb.connect()
con.execute("INSTALL h3 FROM community; LOAD h3;")
con.execute("SET threads = 8")
con.register("pix2h", d["pix2h"])
xql.register(con, "cube", cube, chunks={"t": hours, "y": 45, "x": 45})
Q = f"""
SELECT t, h3_latlng_to_cell(lat, lon, {RES}) AS hex,
       CAST(avg(CAST(temperature_2m AS DOUBLE)) AS FLOAT) AS tc,
       CAST(avg(CAST(relative_humidity_2m AS DOUBLE)) AS FLOAT) AS rh
FROM cube JOIN pix2h USING (y, x)
WHERE temperature_2m = temperature_2m AND ({d["land_pred"]})
GROUP BY 1, 2"""
t1 = time.perf_counter()
tbl = con.sql(Q).arrow().read_all()
print(f"duckdb (rc register + duckdb-h3) fold: {time.perf_counter()-t1:.1f} s  {tbl.num_rows:,} rows")
ref = d["cell_hour"]
print("datafusion rows:", ref.num_rows, " duckdb rows:", tbl.num_rows)
# same numbers? compare mean tc over the whole table
print("mean tc datafusion %.4f  duckdb %.4f" % (np.nanmean(ref["tc"].to_numpy()), np.nanmean(tbl["tc"].to_numpy())))
