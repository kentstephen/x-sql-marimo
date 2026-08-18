"""Hold the store: xarray-sql fold-shaped scans of the HRRR analysis with icechunk's
chunk-bytes cache on, two 7-day windows in the same 90-day store chunk. The second
window should cost decode only (no wire). Land pruning omitted (whole CONUS box),
one variable, so absolute times are not the notebook's; the ratio is the point."""
import time, icechunk, xarray as xr, xarray_sql as xql, numpy as np
st = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
for gb in (0, 6):
    cfg = icechunk.RepositoryConfig(caching=icechunk.CachingConfig(num_bytes_chunks=gb << 30)) if gb else None
    repo = icechunk.Repository.open(st, config=cfg) if cfg else icechunk.Repository.open(st)
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, chunks=None)[["temperature_2m"]].rename({"time": "t"})
    T = ds.sizes["t"]; c0 = (T // 2160 - 1) * 2160          # a full chunk
    for i, (h0, h1) in enumerate([(c0, c0 + 168), (c0 + 400, c0 + 568)]):
        cube = ds.isel(t=slice(h0, h1), y=slice(0, 1059), x=slice(0, 1799))
        ctx = xql.XarrayContext(); ctx.from_dataset("cube", cube, chunks={"t": 168, "y": 45, "x": 45})
        t0 = time.perf_counter(); r = ctx.sql("SELECT avg(temperature_2m) FROM cube WHERE temperature_2m = temperature_2m").collect(); dt = time.perf_counter() - t0
        print(f"cache {gb} GB · window {i} (168 h, whole CONUS, 1 var): {dt:6.1f} s")
