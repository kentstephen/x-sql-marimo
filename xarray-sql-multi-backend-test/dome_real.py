"""D on the real thing: run xsql-hrrr-heat-hex.py headless (marimo app.run), take its
fold output (cell_hour, cells, hi_q), run the accumulator in numpy, band it, and let
DuckDB dissolve each (frame, band) with h3_cells_to_multi_polygon_wkb + ST_Dump.
Prints per-band blob counts, sizes, WKB bytes, and the biggest domes with their tracks.

    uv run python dome_real.py            (from xarray-sql-multi-backend-test/)
"""
import importlib.util, re, sys, time
import numpy as np, pyarrow as pa, duckdb

t0 = time.perf_counter()
# Patch the notebook's constants in memory: young store chunk (~30 s), two variables.
# Set FULL=1 in the env to run its own DAYS (a dome week, full chunk, ~4.5 min).
import os, types
src = open("../xsql-hrrr-heat-hex.py").read()
if not os.environ.get("FULL"):
    src = re.sub(r'\n    DAYS = \(.*?\)   # East dome.*', '\n    DAYS = 7', src)
    src = src.replace("READ_RAIN = True", "READ_RAIN = False").replace("READ_WIND = True", "READ_WIND = False")
# marimo's cell decorator needs the source on disk (inspect.getsourcelines), so the
# patched copy is written to the scratchpad and imported from there.
OUT = os.environ.get("OUT", "/private/tmp/claude-501/-Users-stephenk-dev-projects-x-sql-marimo/996c8bf4-050a-4512-a409-57d0c7c02802/scratchpad")
_p = os.path.join(OUT, "heathex_patched.py"); open(_p, "w").write(src)
spec = importlib.util.spec_from_file_location("heathex", _p)
mod = importlib.util.module_from_spec(spec); sys.modules["heathex"] = mod; spec.loader.exec_module(mod)
outputs, d = mod.app.run()
print(f"notebook ran in {time.perf_counter()-t0:.0f} s")
print(d["pix_stats"]); print(d["fold_stats"])

cells, hi_q, labels = d["cells"], d["hi_q"], d["frame_labels"]
THR, HALF = float(d["THRESHOLD"]), float(d["HALF_LIFE"])
F, N = hi_q.shape
hi = np.where(hi_q == 255, np.nan, hi_q.astype(np.float32) / 2.0 - 40.0)
print(f"frames {F} ({labels[0]} .. {labels[-1]})  cells {N:,}  threshold {THR}  half-life {HALF} h")

# accumulator, the widget's recurrence: L[f] = a L[f-1] + (1-a) max(0, hi-thr)
a = 2 ** (-1 / HALF)
x = np.nan_to_num(np.maximum(hi - THR, 0.0))
L = np.zeros_like(x)
for f in range(1, F): L[f] = a * L[f-1] + (1-a) * x[f]
BANDS = [1.0, 3.0, 5.0, 10.0]
band = np.zeros_like(L, dtype=np.int8)
for b in BANDS: band[L >= b] = int(b)
print("cell-frames per band:", {b: int((band >= b).sum()) for b in BANDS})
print("peak sustained excess: %.1f degC" % L.max())

fi, ci = np.nonzero(band > 0)
mask = pa.table({"f": fi.astype(np.int32), "cell": cells[ci], "band": band[fi, ci]})
con = duckdb.connect()
con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
con.register("m", mask)
# a cell in band b is also in every lower band: dissolve cumulative sets (>= b)
con.execute("CREATE TABLE lvl AS SELECT unnest([1,3,5,10]) AS b")
t1 = time.perf_counter()
con.execute("""
CREATE TABLE polys AS
SELECT f, b, h3_cells_to_multi_polygon_wkb(list(cell)) AS wkb, count(*) AS ncell
FROM m JOIN lvl ON m.band >= lvl.b GROUP BY f, b""")
tA = time.perf_counter() - t1
r = con.sql("SELECT count(*), sum(octet_length(wkb))/1e6 FROM polys").fetchall()[0]
print(f"A. multipolygons: {tA:.1f} s  {r[0]} (frame,band) rows  {r[1]:.1f} MB wkb")

t1 = time.perf_counter()
con.execute("""
CREATE TABLE blobs AS
WITH b AS (SELECT f, b AS band, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM polys)
SELECT f, band, ST_Area(ST_Transform(g, 'EPSG:4326', 'EPSG:5070', always_xy := true))/1e6 AS km2, ST_X(ST_Centroid(g)) AS cx, ST_Y(ST_Centroid(g)) AS cy, g
FROM b""")
tD = time.perf_counter() - t1
print(f"D. blobs: {tD:.1f} s")
con.execute(f"COPY polys TO '{OUT}/dome_polys.parquet'")
con.execute(f"COPY (SELECT f, band, km2, cx, cy, ST_AsWKB(g) AS wkb FROM blobs) TO '{OUT}/dome_blobs.parquet'")
np.savez_compressed(f"{OUT}/dome_frames.npz", cells=cells, hi_q=hi_q, L=L, labels=np.array(labels))
print("saved to", OUT)
print(con.sql("""
SELECT band, count(*) AS blobs, round(median(km2)) AS med_km2, round(max(km2)) AS max_km2,
       count(*) FILTER (km2 > 10000) AS over_10k_km2
FROM blobs GROUP BY band ORDER BY band""").to_df().to_string(index=False))

# the biggest dome per band, at its peak hour, and where it went
print("\nbiggest blob per band, its peak frame:")
print(con.sql("""
SELECT band, f, round(km2) km2, round(cx,1) lon, round(cy,1) lat
FROM (SELECT *, row_number() OVER (PARTITION BY band ORDER BY km2 DESC) rn FROM blobs) WHERE rn = 1
ORDER BY band""").to_df().to_string(index=False))

# naive track: the >=3 band's largest blob per frame, centroid by frame
print("\nlargest >=3 degC blob per frame (every 12 h): area and centroid")
rows = con.sql("""
SELECT f, round(km2) km2, round(cx,1) lon, round(cy,1) lat
FROM (SELECT *, row_number() OVER (PARTITION BY f ORDER BY km2 DESC) rn FROM blobs WHERE band = 3)
WHERE rn = 1 AND f % 12 = 0 ORDER BY f""").fetchall()
for f, km2, lon, lat in rows: print(f"  {labels[f]}  {km2:>9,.0f} km2  ({lat}, {lon})")

# what the film would ship: WKB per frame at the >=1 band, and the edge alternative
print("\nbytes: median wkb per frame (>=1 band): %.0f kB" % (con.sql(
    "SELECT median(octet_length(wkb))/1e3 FROM polys WHERE b = 1").fetchall()[0][0]))
