"""Both engines sequentially in ONE process on the heat domes notebook: does the
second engine's fold ride the chunk cache + mirror (nearly free) and match?

Runs xsql-hrrr-heat-domes.py patched to the East dome week (mirrored on disk),
two variables, ENGINE=datafusion (the default), then registers the same window
on the notebook's own DuckDB connection with xql.register and runs the same fold
SQL with duckdb-h3's h3_latlng_to_cell. Reports fold times, mirror hit/fetch
deltas, row counts and means.
"""
import importlib.util, os, sys, tempfile, time

import numpy as np

OUT = tempfile.mkdtemp(prefix="fold-both-")
src = open(os.path.join(os.path.dirname(__file__), "..", "xsql-hrrr-heat-domes.py")).read()
assert "READ_RAIN = True" in src and "READ_WIND = True" in src
src = src.replace("READ_RAIN = True", "READ_RAIN = False")
src = src.replace("READ_WIND = True", "READ_WIND = False")
assert "\n    DAYS = 7   #" in src
src = src.replace(
    "\n    DAYS = 7   #",
    '\n    DAYS = ("2026-06-28", "2026-07-04")   # East dome (mirrored) #',
)
p = os.path.join(OUT, "heatdomes_patched.py")
open(p, "w").write(src)
spec = importlib.util.spec_from_file_location("heatdomes", p)
mod = importlib.util.module_from_spec(spec)
sys.modules["heatdomes"] = mod
spec.loader.exec_module(mod)

t_run = time.perf_counter()
outputs, d = mod.app.run()
print(f"notebook (datafusion fold) {time.perf_counter() - t_run:.0f} s total")
print("  ", d["fold_stats"])

mirror = d["mirror"]
h0, m0 = (mirror.hits, mirror.misses) if mirror is not None else (0, 0)

cube = d["cube_all"].sel(t=slice(d["t0"], d["t1"]))
hours = int(cube.sizes["t"])
RES = int(d["RES"])
con, xql, pix2h = d["con"], d["xql"], d["pix2h"]

con.register("pix2h", pix2h)
xql.register(con, "cube", cube, chunks={"t": hours, "y": 45, "x": 45})
Q = f"""
SELECT t, h3_latlng_to_cell(lat, lon, {RES}) AS hex,
       CAST(avg(CAST(temperature_2m AS DOUBLE)) AS FLOAT) AS tc,
       CAST(avg(CAST(relative_humidity_2m AS DOUBLE)) AS FLOAT) AS rh
FROM cube JOIN pix2h USING (y, x)
WHERE temperature_2m = temperature_2m AND ({d["land_pred"]})
GROUP BY 1, 2"""
t_dd = time.perf_counter()
tbl = con.sql(Q).arrow().read_all()
dd_s = time.perf_counter() - t_dd
con.unregister("cube")
con.unregister("pix2h")

if mirror is not None:
    print(f"duckdb fold: {dd_s:.1f} s  ·  mirror during duckdb fold: "
          f"{mirror.hits - h0} ranges from disk, {mirror.misses - m0} fetched")
else:
    print(f"duckdb fold: {dd_s:.1f} s (no mirror)")

ref = d["cell_hour"]
print(f"rows  datafusion {ref.num_rows:,}  duckdb {tbl.num_rows:,}")
print("mean tc  datafusion %.4f  duckdb %.4f" % (
    np.nanmean(ref["tc"].to_numpy()), np.nanmean(tbl["tc"].to_numpy())))
print("mean rh  datafusion %.4f  duckdb %.4f" % (
    np.nanmean(ref["rh"].to_numpy()), np.nanmean(tbl["rh"].to_numpy())))
