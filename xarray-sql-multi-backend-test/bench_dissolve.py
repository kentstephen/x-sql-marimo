"""The design question for the dome outlines: how long does DuckDB take to turn a
sustained-heat mask into polygons for a whole film?

Fake but heat-hex-shaped input: ~210k res 6 cells over CONUS land-ish (a lat/lon box
minus nothing; count is what matters), 168 hourly frames, a heat-index field made of a
few moving/growing gaussian domes plus noise, the accumulator run as a DuckDB window
function, three bands (1, 3, 5 degC sustained excess), then per (frame, band):
  A. h3_cells_to_multi_polygon_wkb   (H3's own outer-boundary walk, one row per group)
  B. ST_Union_Agg(cell boundary) + ST_Dump  (one row per blob, with area etc.)
  C. boundary directed edges (cell in mask, neighbour not)  (what the JS test would do)
"""
import time
import numpy as np, pyarrow as pa, duckdb
from h3ronpy.vector import coordinates_to_cells
from h3ronpy import cells_to_string

RES, F = 6, 168
rng = np.random.default_rng(0)
# cells: sample a CONUS box at ~4 km, unique res 6 cells (~200k)
lat = rng.uniform(25, 49, 1_500_000); lon = rng.uniform(-124, -67, 1_500_000)
cells = np.unique(coordinates_to_cells(lat, lon, RES))
N = len(cells)
# cell centres for the synthetic field
con = duckdb.connect()
con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
con.register("cells0", pa.table({"cell": cells}))
c = con.sql("SELECT cell, h3_cell_to_lat(cell) lat, h3_cell_to_lng(cell) lon FROM cells0").arrow().read_all()
clat, clon = c["lat"].to_numpy(), c["lon"].to_numpy()
print(f"cells {N:,}  frames {F}")

# heat index: base diurnal + 3 domes drifting east and growing, degC
t = np.arange(F)
hi = 22 + 6 * np.sin((t[:, None] - 6) / 24 * 2 * np.pi)
for (la, lo, amp, spd) in [(36, -100, 14, 0.15), (33, -90, 10, 0.1), (42, -112, 12, 0.08)]:
    d2 = ((clat[None, :] - la) / 4) ** 2 + ((clon[None, :] - (lo + spd * t[:, None])) / 6) ** 2
    hi = hi + amp * np.exp(-d2) * (0.5 + 0.5 * t[:, None] / F)
hi = (hi + rng.normal(0, 1, hi.shape)).astype(np.float32)   # F x N

# answer table (frame, cell, hi), what DataFusion's fold hands over
ans = pa.table({"f": np.repeat(t, N).astype(np.int32), "cell": np.tile(cells, F), "hi": hi.ravel()})
con.register("ans", ans)
print(f"answer rows {ans.num_rows:,}")

THR, HALF = 27.0, 12.0
a = 2 ** (-1 / HALF)
t0 = time.perf_counter()
con.execute(f"""
CREATE OR REPLACE TABLE load AS
SELECT f, cell,
       (1-{a}) * pow({a}, f) * sum(greatest(hi-{THR},0) * pow({a}, -f))
           OVER (PARTITION BY cell ORDER BY f) AS L
FROM ans""")
print(f"accumulator (window fn):          {time.perf_counter()-t0:6.2f} s")
# numpy reference for the same recurrence
t0 = time.perf_counter()
L = np.zeros_like(hi); x = np.maximum(hi - THR, 0)
for f in range(1, F): L[f] = a * L[f-1] + (1-a) * x[f]
print(f"accumulator (numpy loop):         {time.perf_counter()-t0:6.2f} s")

t0 = time.perf_counter()
con.execute("""
CREATE OR REPLACE TABLE mask AS
SELECT f, cell, CASE WHEN L >= 5 THEN 5 WHEN L >= 3 THEN 3 WHEN L >= 1 THEN 1 END AS band
FROM load WHERE L >= 1""")
n = con.sql("SELECT count(*), count(DISTINCT (f, band)) FROM mask").fetchall()[0]
print(f"bands (>=1,3,5):                  {time.perf_counter()-t0:6.2f} s   {n[0]:,} cell-frames in {n[1]} (frame,band) groups")

t0 = time.perf_counter()
r = con.sql("""
SELECT f, band, h3_cells_to_multi_polygon_wkb(list(cell)) AS geom
FROM mask GROUP BY f, band""").arrow().read_all()
dt = time.perf_counter() - t0
print(f"A. h3_cells_to_multi_polygon_wkb: {dt:6.2f} s   {r.num_rows} rows  {sum(len(g) for g in r['geom'].to_pylist())/1e6:.1f} MB wkb")

t0 = time.perf_counter()
r = con.sql("""
WITH u AS (
  SELECT f, band, ST_Union_Agg(ST_GeomFromWKB(h3_cell_to_boundary_wkb(cell))) AS g
  FROM mask GROUP BY f, band)
SELECT f, band, UNNEST(ST_Dump(g)).geom AS geom FROM u""").arrow().read_all()
dt = time.perf_counter() - t0
print(f"B. ST_Union_Agg + ST_Dump:        {dt:6.2f} s   {r.num_rows} blob rows")

t0 = time.perf_counter()
r = con.sql("""
WITH e AS (
  SELECT m.f, m.band, UNNEST(h3_origin_to_directed_edges(m.cell)) AS edge
  FROM mask m)
SELECT e.f, e.band, edge
FROM e
LEFT JOIN mask n ON n.f = e.f AND n.band = e.band
                AND n.cell = h3_get_directed_edge_destination(e.edge)
WHERE n.cell IS NULL""").arrow().read_all()
dt = time.perf_counter() - t0
print(f"C. boundary directed edges:       {dt:6.2f} s   {r.num_rows:,} edges")

# one frame only, for the live-follow question
t0 = time.perf_counter()
con.sql("SELECT band, h3_cells_to_multi_polygon_wkb(list(cell)) FROM mask WHERE f = 150 GROUP BY band").arrow().read_all()
print(f"A, single frame (f=150):          {(time.perf_counter()-t0)*1000:6.0f} ms")
t0 = time.perf_counter()
con.sql("""WITH u AS (SELECT band, ST_Union_Agg(ST_GeomFromWKB(h3_cell_to_boundary_wkb(cell))) g FROM mask WHERE f=150 GROUP BY band)
SELECT band, UNNEST(ST_Dump(g)).geom FROM u""").arrow().read_all()
print(f"B, single frame (f=150):          {(time.perf_counter()-t0)*1000:6.0f} ms")

# D. per-blob rows from A's multipolygon (already dissolved by H3): dump + measure
t0 = time.perf_counter()
r = con.sql("""
WITH u AS (SELECT f, band, ST_GeomFromWKB(h3_cells_to_multi_polygon_wkb(list(cell))) AS g
           FROM mask GROUP BY f, band),
     b AS (SELECT f, band, UNNEST(ST_Dump(g)).geom AS geom FROM u)
SELECT f, band, ST_Area_Spheroid(geom)/1e6 AS km2, ST_X(ST_Centroid(geom)) AS cx, ST_Y(ST_Centroid(geom)) AS cy
FROM b""").arrow().read_all()
dt = time.perf_counter() - t0
print(f"D. A + ST_Dump + area/centroid:   {dt:6.2f} s   {r.num_rows:,} blob rows")
