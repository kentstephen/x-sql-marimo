import time, warnings; warnings.filterwarnings("ignore")
import duckdb, xarray as xr, zarr, numpy as np
from obstore.store import S3Store
import xarray_sql as xql
base = S3Store(bucket="us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True,
               prefix="tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/")
zs = zarr.storage.ObjectStore(base, read_only=True)
con = duckdb.connect(); con.sql("INSTALL spatial; LOAD spatial;")
W,S,E,N = -119.9,36.6,-119.7,36.8
for grp in ["4x", "16x", None]:
    t=time.time()
    ds = xr.open_zarr(zs, group=grp, chunks=None, consolidated=False)
    print(grp, dict(ds.sizes), list(ds.data_vars), ds.variables.attrs if hasattr(ds,'variables') and False else "", f"open {time.time()-t:.1f}s")
    k = 1 if grp is None else int(grp[:-1])
    # block = the shard footprint so a window decodes each inner chunk once
    sh = 2048 if grp is None else 512
    name = f"ftw_{k}"
    xql.register(con, name, ds, chunks={"time":1, "band":3, "y":sh, "x":sh})
    t=time.time()
    r = con.sql(f"""SELECT band, count(*), avg(variables), avg(CASE WHEN variables>=0.5 THEN 1 ELSE 0 END)
        FROM {name} WHERE time = TIMESTAMP '2024-01-01' AND x BETWEEN {W} AND {E} AND y BETWEEN {S} AND {N} GROUP BY 1 ORDER BY 1""").fetchall()
    print("  ", r, f"{time.time()-t:.1f}s")
    if grp is None: break
print(con.sql("DESCRIBE ftw_4").fetchall())
