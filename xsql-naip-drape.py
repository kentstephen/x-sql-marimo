# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "palettable>=3.3",
#     "matplotlib",
#     "arro3-core",
#     "geoarrow-rust-core>=0.6",
#     "arro3-io",
#     "numpy",
#     "pyproj>=3.7",
#     "geopy==2.5.0",
#     "aiohttp>=3.10",
#     "pystac-client>=0.9",
#     "planetary-computer>=1.0",
#     "shapely>=2.1",
# ]
# ///
"""NAIP DRAPED OVER A WIDE SWATH. 10 m DEM -> H3 -> one textured mesh, imagery on top.

This is `xsql-s1m-surface.py` pointed at the thing a textured mesh is actually for. That
notebook proved the escape from `H3HexagonLayer`: geometry is ONE triangle mesh and
styling is ONE image, so neither cost scales with the cell count. Once styling is an
image, it does not have to be a colour ramp. It can be a photograph.

WHAT CHANGED, and each change is because the target is a SWATH, not a summit.

  * THE DEM IS 10 m, NOT 1 m. The 1/3 arc-second seamless product, straight off `prd-tnm`.
    A 30 km swath at 1 m is ~900 million pixels to stream for a mesh that tops out at 2048
    vertices a side, and S1M has real coverage gaps that a wide box will find. The 10 m
    product is nationwide and needs no footprint index at all, which is why the S1M
    coverage carpet is gone from the picker: there is nothing to check.
  * NO ALBERS, NO POLYNOMIAL FIT. S1M tiles are projected, so that notebook had to fit
    lon/lat per tile as an order-3 polynomial (pyproj aborts the process from a DataFusion
    worker thread). The seamless 10 m COGs are EPSG:4269, already geographic, so the grid
    coordinates ARE lon and lat and they go straight into `h3_latlng_to_cell`. About
    eighty lines of machinery deleted rather than ported.
  * THE TEXTURE CAN COME FROM NAIP. One dropdown, three sources, described below.

WHERE NAIP COMES FROM, and it is the one thing here that is not an anonymous S3 read.
All three AWS NAIP buckets are requester-pays and 403 an anonymous request, so they need
an account and a bill. Planetary Computer's STAC signs NAIP hrefs anonymously and free,
so it is the only no-account path and it is what `naip.py` uses. The COGs themselves are
then read with the SAME obstore + async-geotiff reader the DEM uses, over HTTP instead of
S3. The DEM never leaves the USGS bucket.

THE TWO TEXTURE SOURCES:

  * `Palette` is the surface notebook's ramp over scene-relative elevation. (Its flow
    offset is parked along with the k-ring join that produced it.) H3 is the binning.
  * `NAIP` never touches H3. Imagery is resampled straight onto the texel lattice at
    whatever the texture can hold, so it keeps every bit of detail the drape can show, and
    it is the default. Folding imagery into hexagons was tried and removed: a hexagon can
    only ever be an AVERAGE of the pixels under it, so it costs resolution and returns
    nothing, and at res 11 under a 10 m texel it was not even visibly different. H3 still
    bins the DEM.

THE TEXTURE CEILING, AND THE TILE GRID THAT LIFTS IT. One texture over a wide box was the
binding constraint on this drape: 2048 texels over 24 km is ~12 m per texel against NAIP's
0.6 m, so the swath view threw away most of the imagery. It is not a download problem and
no amount of streaming fixes it. A GPU caps a single texture at 8192-16384 texels a side,
and 24 km at 0.6 m would need 40,000, so the image physically cannot be uploaded whole.

So the drape is now N x N `SurfaceLayer`s, `Drape tiles` in the controls. Each tile is its
own mesh and its own texture over 1/N^2 of the ground, which multiplies ground resolution
by N without any single texture growing: 4x4 at 2048 is 8192 texels across the AOI, ~3 m,
four times finer than the largest texture deck will accept. Each tile also reads NAIP at
the overview matching ITS texel size, so the extra resolution is genuinely streamed rather
than interpolated. The tile cell prints metres-per-texel and total VRAM before the reads
start, because every step of the grid is 2x the memory and 4x the download.

WHAT TILING DOES NOT BUY, and it is the reason 4x4 is the last option rather than a
slider. Every tile is resident at once: nothing here knows what is on screen, so the grid
is bounded by total VRAM (4x4 at 4096 is 1.1 GB and will not fit) rather than by the
per-texture cap. Full NAIP resolution over a swath needs the tiles to be VIEW-DRIVEN, i.e.
built and evicted as the camera moves, which is what deck's own `TileLayer` does for
`lonboard.RasterLayer` and what `SurfaceLayer` has no protocol for. See `RasterLayer` in
lonboard 0.16 for the shape that would take: an async `fetch_tile(x, y, z)` plus a
`render_tile` callback, with the COG's overview pyramid used as the zoom pyramid.

TERRAIN IS NOT TILED, ONLY PIXELS ARE. The H3 index, the height field and the hillshade
stay on one global lattice, because they come from a 10 m DEM that a 2048 grid already
out-resolves. So raising the tile count costs no H3 work, and every tile samples the SAME
height field, which is what welds the seams: a boundary vertex has one lon/lat, indexes
one height, and both neighbours place it identically. There is no crack to reconcile
because there is no second opinion about the ground.

THE HILLSHADE DOES NOT TOUCH THE IMAGERY. The surface notebook bakes a synthetic 315/45
sun into its texture because lonboard's SurfaceLayer ships no NORMAL attribute, so deck
cannot light the mesh and an unlit colour ramp does not read as terrain. A NAIP frame
needs none of that: it is a photograph taken in real sunlight and the shadows are already
in the pixels. A second sun on top double-shades it into a glossy shell. So the hillshade
applies to the `Palette` source only, and the drape looks like the raster it is.

CARRIED OVER UNCHANGED from the surface notebook, and see its docstring for why:
`relief_smooth` on the height field (the
fold is piecewise constant, so the hexagonal staircase is in the DATA and no mesh density
fixes it), the NaN-aware blurs, the parquet patch, and the build-once-swap-traits layer.
`wha
Run:  uv run marimo edit xsql-naip-drape.py --sandbox

`naip.py` must sit next to this file: it holds the STAC search and the drape warp.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import sqlite3
    import struct
    import urllib.request
    import xml.etree.ElementTree as ET
    from io import BytesIO

    import h3ronpy
    import numpy as np
    import palettable
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    import geoarrow.rust.core as grc
    from arro3.core import Table
    from pyproj import Transformer
    from obstore.store import S3Store, HTTPStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import BitmapTileLayer, Map, SolidPolygonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard.colormap import apply_continuous_cmap
    from lonboard.controls import (
        FullscreenControl,
        GeocoderControl,
        NavigationControl,
        ScaleControl,
    )

    # SurfaceLayer is real but unexported: import it off the private module.
    from lonboard.experimental._surface import SurfaceLayer

    # The NAIP half of the notebook, kept out of it. STAC search + the drape warp.
    import naip

    return (
        AioHTTPAdapter,
        BitmapTileLayer,
        BytesIO,
        CartoBasemap,
        ET,
        FullscreenControl,
        GeoTIFF,
        GeocoderControl,
        HTTPStore,
        Map,
        MaplibreBasemap,
        NavigationControl,
        Photon,
        S3Store,
        ScaleControl,
        SolidPolygonLayer,
        SurfaceLayer,
        Table,
        Transformer,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        coordinates_to_cells,
        grc,
        h3ronpy,
        mo,
        naip,
        np,
        pa,
        pathlib,
        sqlite3,
        struct,
        udf,
        urllib,
        xr,
    )


@app.cell
def _(BytesIO):
    # THE PARQUET PATCH. Without this the kernel SEGFAULTS the moment a SurfaceLayer is
    # constructed, with no traceback. lonboard ships synced arrow columns to the browser as
    # Parquet and prefers pyarrow's writer; handing pyarrow 25 a 3-wide FixedSizeList that
    # arrived over the arro3 C Data Interface crashes inside ParquetWriter.__init__, and
    # `positions` cannot avoid being 3-wide. lonboard already falls back to arro3's own
    # writer when pyarrow is absent, so force that branch. Full measurement of which shapes
    # crash is in xsql-s1m-surface.py.
    import lonboard._serialization as _ser
    from arro3.io import write_parquet as _write_parquet

    def _write_parquet_batch(record_batch):
        if record_batch.num_rows == 0:
            raise ValueError("Batch with 0 rows.")
        bio = BytesIO()
        _write_parquet(
            record_batch,
            bio,
            compression="ZSTD(7)",
            max_row_group_size=record_batch.num_rows,
        )
        return bio.getvalue()

    _ser.write_parquet_batch = _write_parquet_batch
    print("parquet writer forced to arro3 (pyarrow segfaults on FixedSizeList(3))")
    return


@app.cell
def _(mo):
    mo.md("""
    # NAIP draped over a wide swath

    10 m seamless DEM off the USGS bucket, folded into H3 in SQL, drawn as **one textured
    mesh**. The texture is either a palette ramp or **NAIP aerial imagery**, warped onto
    the same lattice the cells are indexed on. Draw a box (Ctrl/Cmd + drag), then pick a
    texture source under the scene.
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, h3ronpy, np, pa, udf):
    # TWO SOURCES, TWO SHAPES OF QUERY, and the difference is entirely about CRS.
    #
    # The seamless 10 m COGs are EPSG:4269, so their grid coordinates already ARE degrees
    # and `h3_latlng_to_cell(y, x, res)` reads them directly. Nothing below is needed.
    #
    # S1M 1 m COGs are EPSG:6350 Albers, so their coordinates are metres and something has
    # to turn them into degrees per pixel. That cannot be pyproj inside the UDF: called
    # from a DataFusion worker thread it ABORTS THE PROCESS. So fit lon/lat over each
    # tile's extent as an order-3 polynomial on the main thread (where pyproj is fine) and
    # let the UDF evaluate the polynomial in pure numpy. The fit is checked against pyproj
    # on a 64x64 grid and refuses to run if it is off by more than a millimetre.
    #
    # Materialising lon/lat as two float64 arrays instead would also work and would be
    # exact, but at res 14 the read is ~49M pixels and that is another 780 MB resident for
    # a job that is already the memory-heavy one. The polynomial is ~10 flops per pixel
    # and allocates nothing. Verbatim from xsql-s1m-surface.py.
    PROJ_ORDER = 3

    def _design(u, v, order=PROJ_ORDER):
        cols = [np.ones_like(u)]
        for total in range(1, order + 1):
            for i in range(total + 1):
                cols.append(u ** (total - i) * v**i)
        return np.column_stack(cols)

    def fit_lonlat(crs, bounds, samples=12, check=64, tol_mm=1.0):
        """Fit lon/lat over a tile's extent. Main thread only: this is the pyproj call."""
        left, bottom, right, top = bounds
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
        sx = max((right - left) / 2.0, 1e-9)
        sy = max((top - bottom) / 2.0, 1e-9)

        fx, fy = np.meshgrid(
            np.linspace(left, right, samples), np.linspace(bottom, top, samples)
        )
        flon, flat = inv.transform(fx.ravel(), fy.ravel())
        A = _design((fx.ravel() - cx) / sx, (fy.ravel() - cy) / sy)
        clon = np.linalg.lstsq(A, flon, rcond=None)[0]
        clat = np.linalg.lstsq(A, flat, rcond=None)[0]

        tx, ty = np.meshgrid(
            np.linspace(left, right, check), np.linspace(bottom, top, check)
        )
        tlon, tlat = inv.transform(tx.ravel(), ty.ravel())
        B = _design((tx.ravel() - cx) / sx, (ty.ravel() - cy) / sy)
        err_m = np.hypot(
            (B @ clat - tlat) * 111_320.0,
            (B @ clon - tlon) * 111_320.0 * np.cos(np.radians(tlat)),
        )
        err_mm = float(err_m.max() * 1000.0)
        if not np.isfinite(err_mm) or err_mm > tol_mm:
            raise RuntimeError(
                f"lon/lat fit for {crs} over {bounds} is off by {err_mm:.3f} mm "
                f"(tolerance {tol_mm} mm). Raise PROJ_ORDER or shrink the window."
            )
        return (cx, cy, sx, sy, clon, clat), err_mm

    def make_lonlat_udf(name, fit):
        """One UDF per tile, its fitted coefficients closed over. Pure numpy inside."""
        cx, cy, sx, sy, clon, clat = fit

        def _to_lonlat(x, y):
            A = _design((x.to_numpy() - cx) / sx, (y.to_numpy() - cy) / sy)
            return pa.StructArray.from_arrays(
                [pa.array(A @ clon), pa.array(A @ clat)], names=["lon", "lat"]
            )

        return udf(
            _to_lonlat,
            [pa.float64(), pa.float64()],
            pa.struct([("lon", pa.float64()), ("lat", pa.float64())]),
            "stable",
            name=name,
        )

    def _latlng_to_cell(lat, lng, res):
        return pa.array(
            coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py())
        )

    def _grid_disk(cell, k):
        return pa.array(h3ronpy.grid_disk(cell, k[0].as_py()))

    def make_h3_context():
        ctx = XarrayContext()
        ctx.register_udf(
            udf(
                _latlng_to_cell,
                [pa.float64(), pa.float64(), pa.int32()],
                pa.uint64(),
                "stable",
                name="h3_latlng_to_cell",
            )
        )
        ctx.register_udf(
            udf(
                _grid_disk,
                [pa.uint64(), pa.int32()],
                pa.large_list(pa.uint64()),
                "stable",
                name="h3_grid_disk",
            )
        )
        return ctx

    print("xarray-sql context factory ready")
    return fit_lonlat, make_h3_context, make_lonlat_udf


@app.cell
def _(mo):
    # THE DEM SOURCE, and the tradeoff is resolution against extent.
    #
    # 10 m seamless is nationwide with no gaps, so any box in CONUS works, and it is the
    # right choice for a swath: over 20 km the texture is already ~5 m per texel at 4096,
    # so a finer DEM has nowhere to show itself.
    #
    # S1M 1 m is what you switch to when the box is small enough for the detail to
    # survive. It only exists inside published project footprints, so a box outside one
    # gets nothing, and it only actually changes the picture at H3 res 13 or 14: at res 11
    # a hexagon is ~50 m across and the fold averages a 1 m DEM straight back down to it.
    dem_source = mo.ui.dropdown(
        options={
            "10 m seamless (nationwide)": "10m",
            "1 m S1M (project footprints only)": "1m",
        },
        value="10 m seamless (nationwide)",
        label="DEM",
    )
    return (dem_source,)


@app.cell
def _(dem_source, mo):
    # THE FOLD RESOLUTION IS USUALLY WHAT LIMITS THE TERRAIN, not the DEM. A res-11 hexagon
    # is ~50 m corner to corner, so it averages a 1 m DEM and a 10 m DEM to very nearly the
    # same surface. Which is why the options move with the source: there is no point
    # offering res 14 over a 10 m DEM (most cells would hold no pixel centre and the fold
    # comes back full of holes), and no point defaulting a 1 m DEM to res 11 (it throws
    # away the reason you chose it).
    _OPTS = {
        9: "res 9 ·  ~400 m hex",
        10: "res 10 ·  ~150 m hex",
        11: "res 11 ·  ~57 m hex",
        12: "res 12 ·  ~22 m hex",
        13: "res 13 ·  ~8 m hex",
        14: "res 14 ·  ~3 m hex",
    }
    if dem_source.value == "1m":
        _keys, _default = [11, 12, 13, 14], 13
    else:
        _keys, _default = [9, 10, 11, 12], 11
    h3_res = mo.ui.dropdown(
        options={_OPTS[k]: k for k in _keys},
        value=_OPTS[_default],
        label="H3 resolution",
    )
    return (h3_res,)


@app.cell
def _(dem_source, h3_res, mo):
    # BOTH SOURCE CONTROLS, ABOVE THE MAP, because they describe what you are about to
    # fetch rather than how the result looks. Everything that only restyles an existing
    # scene lives under the scene instead.
    mo.vstack(
        [
            mo.hstack([dem_source, h3_res], justify="start", gap=2),
            mo.md(
                "<small>Switching to **1 m** draws the S1M coverage footprints on the map "
                "below: the 1 m product exists only inside them. The 10 m seamless needs "
                "no such check, so the carpet hides itself. Neither control fetches "
                "anything until you draw a box.</small>"
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(ET, pathlib, urllib):
    # THE CATALOG IS THE VRT. USGS publishes one nationwide .vrt that lists every 1-degree
    # seamless COG on prd-tnm with its exact pixel placement, so parsing it once turns
    # "which COGs cover this AOI" into a local bbox intersection. No STAC API, no signing,
    # no index download per query. ~830 KB, cached next to the notebook.
    #
    # Each <ComplexSource> carries the source path plus its DstRect in the mosaic; the
    # mosaic GeoTransform turns that rectangle into degrees.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"
    VRT_URL = (
        S3_BASE + "StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"
    )
    CACHE = pathlib.Path(".cache")

    _vrt = CACHE / "USGS_Seamless_DEM_13.vrt"
    if not _vrt.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("downloading the 1/3 arc-second seamless VRT index (~830 KB)...")
        urllib.request.urlretrieve(VRT_URL, _vrt)

    _root = ET.parse(_vrt).getroot()
    _gt = [float(v) for v in _root.find("GeoTransform").text.split(",")]
    dem_tiles = []
    for _src in _root.iter("ComplexSource"):
        _href = _src.find("SourceFilename").text.removeprefix("/vsicurl/")
        _rect = _src.find("DstRect")
        _w = _gt[0] + float(_rect.get("xOff")) * _gt[1]
        _n = _gt[3] + float(_rect.get("yOff")) * _gt[5]
        _e = _w + float(_rect.get("xSize")) * _gt[1]
        _s = _n + float(_rect.get("ySize")) * _gt[5]
        dem_tiles.append(
            {"key": _href.split("amazonaws.com/", 1)[-1], "bbox": (_w, _s, _e, _n)}
        )

    # Native pixel size in degrees, off the mosaic transform: 1/3 arc-second.
    DEM_DEG = abs(_gt[1])
    print(
        f"10 m seamless index: {len(dem_tiles):,} COGs · "
        f"native {DEM_DEG * 3600:.3f}\" ({DEM_DEG * 111_320:.1f} m)"
    )
    return DEM_DEG, S3_BASE, dem_tiles


@app.cell
def _(Transformer, np, pathlib, sqlite3, struct, urllib):
    # THE S1M CATALOG, for the 1 m source. Same trick as the VRT, different container: the
    # national 1 m index is one ~15 MB GeoPackage, which is SQLite, so stdlib sqlite3 reads
    # it and every AOI is answered from a local file. Verbatim from xsql-s1m-surface.py
    # minus the coverage-layer geometry, which this notebook does not draw.
    #
    # Footprints are in EPSG:6350 Albers. Kept in Albers deliberately: the AOI is projected
    # INTO Albers for the intersection and the read window rather than the tiles being
    # unprojected out of it, because a 10 km Albers square is a rotated quad in degrees and
    # its lon/lat envelope is bigger than the tile.
    GPKG_KEY = "StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg"
    _path = pathlib.Path(".cache") / "S1M_Products.gpkg"
    if not _path.exists():
        _path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the S1M tile index (~15 MB)...")
        urllib.request.urlretrieve(
            "https://prd-tnm.s3.amazonaws.com/" + GPKG_KEY, _path
        )

    def _envelope(blob):
        flags = blob[3]
        if (flags >> 1) & 0x07 == 0:
            raise ValueError("S1M footprint has no envelope in its GPB header")
        little = bool(flags & 0x01)
        xmin, xmax, ymin, ymax = struct.unpack_from("<4d" if little else ">4d", blob, 8)
        return xmin, ymin, xmax, ymax

    with sqlite3.connect(f"file:{_path}?mode=ro", uri=True) as _con:
        _rows = _con.execute(
            "SELECT geom, tile, z_max, dataset_link FROM current"
        ).fetchall()

    s1m_albers = np.array([_envelope(r[0]) for r in _rows], dtype="float64")

    # Reproject every footprint's CORNERS once, vectorised over the whole product. Albers
    # corners, not the lon/lat envelope: a 10 km Albers square is a slightly rotated quad
    # in degrees, and drawing it as an axis-aligned box would smear the national grid into
    # a staircase. Corner order SW, SE, NE, NW.
    _inv = Transformer.from_crs("EPSG:6350", "EPSG:4326", always_xy=True)
    _cx = np.column_stack(
        [s1m_albers[:, 0], s1m_albers[:, 2], s1m_albers[:, 2], s1m_albers[:, 0]]
    )
    _cy = np.column_stack(
        [s1m_albers[:, 1], s1m_albers[:, 1], s1m_albers[:, 3], s1m_albers[:, 3]]
    )
    _lon, _lat = _inv.transform(_cx.ravel(), _cy.ravel())
    _lon, _lat = _lon.reshape(-1, 4), _lat.reshape(-1, 4)

    s1m_tiles = [
        {
            "key": r[3].split("amazonaws.com/", 1)[-1],
            "tile": r[1],
            # z_min carries a nodata sentinel (-999999) wherever a tile has holes, so the
            # coverage shading reads z_max only.
            "z_max": float(r[2]) if r[2] is not None else float("nan"),
            "quad": list(zip(_lon[i], _lat[i])),
        }
        for i, r in enumerate(_rows)
    ]
    # AOI degrees -> Albers metres, needed by both the intersection and the read window.
    to_albers = Transformer.from_crs("EPSG:4326", "EPSG:6350", always_xy=True)
    print(f"S1M 1 m index: {len(s1m_tiles):,} current tiles")
    return s1m_albers, s1m_tiles, to_albers


@app.cell
def _(
    SolidPolygonLayer,
    Table,
    apply_continuous_cmap,
    grc,
    np,
    pa,
    palettable,
    s1m_tiles,
):
    # THE COVERAGE CARPET: the entire 1 m product as geometry, so the question "does S1M
    # exist where I am about to draw" is answered by looking rather than by drawing a box
    # and reading an error. It is built once here and toggled by the DEM control, because
    # the 10 m seamless is nationwide and has no such question to answer.
    #
    # Shaded by z_max on viridis at low opacity and with NO outlines, so neighbouring tiles
    # blend into one continuous field and the carpet reads as a single dissolved coverage
    # shape with some elevation context, rather than as 11,717 separately coloured boxes.
    # Outlining each tile is what made it read as a grid. The coverage answer is the
    # PRESENCE of the shape, never its hue.
    _wkts = pa.array(
        [
            "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in [*t["quad"], t["quad"][0]]) + "))"
            for t in s1m_tiles
        ]
    )
    _geom = grc.from_wkt(_wkts, to_type=grc.from_wkt(_wkts).type.with_crs("EPSG:4326"))

    _zmax = np.array([t["z_max"] for t in s1m_tiles], dtype="float64")
    _zmax = np.where(np.isfinite(_zmax), _zmax, 0.0)
    # Clip to a robust range so a handful of high peaks do not flatten the whole ramp.
    _lo, _hi = float(np.percentile(_zmax, 1)), float(np.percentile(_zmax, 99))
    _norm = np.clip((_zmax - _lo) / max(_hi - _lo, 1.0), 0.0, 1.0)

    coverage_layer = SolidPolygonLayer(
        table=Table.from_arrow(
            pa.table(
                {
                    "tile": pa.array([t["tile"] for t in s1m_tiles]),
                    "max elevation (m)": pa.array(_zmax),
                }
            )
        ).append_column("geometry", _geom),
        get_fill_color=apply_continuous_cmap(
            _norm, palettable.matplotlib.Viridis_20, alpha=150
        ),
        opacity=0.35,
        extruded=False,
        pickable=False,
        visible=False,  # the DEM control turns it on; 10 m needs no coverage check
    )
    print(f"S1M coverage carpet ready ({len(s1m_tiles):,} footprints, hidden by default)")
    return (coverage_layer,)


@app.cell
def _(mo):
    # The Presidential Range, New Hampshire: ~24 x 20 km, the whole ridge from Pinkham
    # Notch to Crawford Notch rather than the single summit cone the surface notebook
    # opens on. A SWATH is the point of this notebook, so it seeds as one.
    #
    # Other good boxes to paste in, all wide and all mountainous:
    #   Sangre de Cristo, NM   [-105.70, 36.50, -105.30, 36.85]
    #   Wind River, WY         [-109.75, 43.05, -109.30, 43.40]
    #   Beartooth, MT          [-109.65, 45.00, -109.20, 45.30]
    get_bbox, set_bbox = mo.state([-71.42, 44.16, -71.15, 44.36])

    # THE FIRST-RUN LATCH. Opening the notebook should stop at the picker: streaming COGs
    # and folding a few hundred thousand cells is not a page-load activity, and certainly
    # not something to redo while you are still choosing a DEM and a resolution.
    #
    # DRAWING THE BOX IS THE RUN. There is no button, because a button would be a second
    # gesture meaning "yes, really" after the one gesture that already said it. The picker
    # exists to be drawn on; doing so is an unambiguous statement of intent, and it is the
    # only action that supplies the one input the pipeline cannot default.
    #
    # It is a LATCH, not a switch: it blocks until the first box and is transparent
    # forever after, so once a scene exists, nudging the resolution just re-runs. Nothing
    # ever closes it.
    get_started, set_started = mo.state(False)
    return get_bbox, get_started, set_bbox, set_started


@app.cell
def _(
    AioHTTPAdapter,
    BitmapTileLayer,
    CartoBasemap,
    FullscreenControl,
    GeocoderControl,
    Map,
    MaplibreBasemap,
    NavigationControl,
    Photon,
    ScaleControl,
    coverage_layer,
    set_bbox,
    set_started,
):
    # THE PICKER, and note what is NOT on it: there is no coverage layer. The S1M notebooks
    # open on a carpet of 1 m footprints because the first question there is "does this
    # product exist here at all". The seamless 10 m answers that question everywhere in the
    # country, so the picker is just a basemap and a box.
    #
    # Both hosts are ArcGIS MapServer, so both are /tile/{z}/{y}/{x}: ROW BEFORE COLUMN. In
    # XYZ order they return tiles from the wrong place rather than 404ing, which reads as a
    # projection bug rather than a typo.
    #
    # Terms: the arcgisonline raster tiles are unauthenticated but Esri scopes them to use
    # with Esri APIs or an API key. The National Map viewer is an Esri JS app so it
    # qualifies and this notebook does not. Kept as the default anyway because it is the
    # basemap the viewer shows and it is LABELLED: you are picking a place, so you need to
    # be able to read which place it is. Every USGS entry below is unencumbered.
    _USGS = "https://basemap.nationalmap.gov/arcgis/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    _ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"

    # label -> (tile template, max zoom the service actually publishes). Declaring max_zoom
    # makes deck overzoom the deepest real tile instead of requesting ones the server does
    # not have; the USGS services stop at 16, Esri's go deeper.
    BASEMAPS = {
        "Esri Topographic (viewer default)": (_ESRI.format("World_Topo_Map"), 19),
        "Esri Terrain": (_ESRI.format("World_Terrain_Base"), 13),
        "USGS Imagery + Topo": (_USGS.format("USGSImageryTopo"), 16),
        "USGS Imagery only": (_USGS.format("USGSImageryOnly"), 16),
        "USGS Topo": (_USGS.format("USGSTopo"), 16),
        "USGS Shaded relief": (_USGS.format("USGSShadedReliefOnly"), 16),
    }
    _default = "Esri Topographic (viewer default)"
    basemap_layer = BitmapTileLayer(
        data=BASEMAPS[_default][0],
        max_zoom=BASEMAPS[_default][1],
        tile_size=256,
        opacity=1.0,
        max_requests=-1,  # HTTP/2, so let the browser pipeline rather than throttling to 6
    )

    # Built once and referencing no reactive UI element, so pan/zoom/AOI survive every
    # downstream run. Nothing ever reassigns .layers.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        # Basemap tiles first so the coverage carpet draws on top of them.
        layers=[basemap_layer, coverage_layer],
        view_state={"longitude": -71.29, "latitude": 44.26, "zoom": 9.5, "pitch": 0},
        basemap=MaplibreBasemap(style=CartoBasemap.Positron),
        controls=[
            _geocoder,
            FullscreenControl(position="top-right"),
            # visualize_pitch makes the compass button call resetNorthPitch(): one click
            # snaps back to north-up AND flat, not just north-up.
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
    )
    def _on_box(change):
        """selected_bounds -> the AOI, and the first box also opens the latch."""
        if change["new"] is None:
            return
        set_bbox(list(change["new"]))
        set_started(True)

    picker.observe(_on_box, names="selected_bounds")
    picker
    return BASEMAPS, basemap_layer


@app.cell
def _(BASEMAPS, mo):
    # Basemap picker for the map above. Its own cell, downstream of the picker, so choosing
    # a basemap never rebuilds the Map and never disturbs a box you have already drawn.
    basemap_choice = mo.ui.dropdown(
        options=list(BASEMAPS), value="Esri Topographic (viewer default)", label="Basemap"
    )
    basemap_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Basemap opacity"
    )
    mo.vstack(
        [
            mo.hstack([basemap_choice, basemap_opacity], justify="start", gap=2),
            mo.md(
                "<small>Basemaps: [USGS The National Map]"
                "(https://basemap.nationalmap.gov/) and Esri. Ctrl/Cmd + drag to draw an "
                "AOI. The 10 m DEM is nationwide, so anywhere in CONUS works.</small>"
            ),
        ],
        gap=0.5,
    )
    return basemap_choice, basemap_opacity


@app.cell
def _(BASEMAPS, basemap_choice, basemap_layer, basemap_opacity):
    # Live trait swap: nudge the running BitmapTileLayer rather than reassigning
    # picker.layers, which would rebuild the deck stack. max_zoom moves with the URL, not
    # after it: leaving a deep Esri max_zoom on a USGS service asks for z17+ tiles that do
    # not exist and the basemap goes blank exactly when you zoom in to place a box.
    _url, _maxz = BASEMAPS[basemap_choice.value]
    basemap_layer.max_zoom = _maxz
    basemap_layer.data = _url
    basemap_layer.opacity = basemap_opacity.value
    return


@app.cell
def _(coverage_layer, dem_source):
    # Live trait swap, same idiom as the basemap: the picker is built once and never
    # reassigns .layers, so showing the carpet is a `visible` flip on the running layer and
    # costs neither a rebuild nor the box you have already drawn.
    coverage_layer.visible = dem_source.value == "1m"
    return


@app.cell
def _(get_bbox):
    bbox = list(get_bbox())
    return (bbox,)


@app.cell
def _(
    DEM_DEG,
    bbox,
    dem_source,
    dem_tiles,
    get_started,
    h3_res,
    mo,
    np,
    s1m_albers,
    s1m_tiles,
    to_albers,
):
    mo.stop(
        not get_started(),
        mo.md(
            "### Draw a box to start\n"
            "Pick a **DEM** and an **H3 resolution** above, then **Ctrl/Cmd + drag** on "
            "the map to draw an AOI. That starts the pipeline; nothing is fetched before "
            "it. After the first box, the controls drive the scene directly."
        ),
    )

    # WHICH COGs, AND WHICH OVERVIEW OF THEM. One geometric rule for both sources: pick the
    # coarsest level whose pixel spacing still guarantees EVERY H3 cell gets at least one
    # pixel centre, p <= sqrt(2) * 0.5373 * sqrt(A), taken at SAFETY 0.6. Reading finer
    # than that is download you throw away in the GROUP BY.
    #
    # What differs is the units the rule comes out in, because the two products are in
    # different CRSs and the reader's `res` is always in the source's own units.
    _w, _s, _e, _n = bbox
    H3_CELL_M2 = {
        9: 105332.5, 10: 15047.5, 11: 2149.6, 12: 307.09, 13: 43.870, 14: 6.2673
    }
    SAFETY = 0.6
    _target_m = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])

    if dem_source.value == "1m":
        # S1M is EPSG:6350 Albers: metres in, metres out, and the AOI is projected into
        # Albers for both the footprint intersection and the read window.
        _ax, _ay = to_albers.transform([_w, _e, _e, _w], [_s, _s, _n, _n])
        aoi_native = (min(_ax), min(_ay), max(_ax), max(_ay))
        _hits = np.flatnonzero(
            (s1m_albers[:, 0] < aoi_native[2])
            & (s1m_albers[:, 2] > aoi_native[0])
            & (s1m_albers[:, 1] < aoi_native[3])
            & (s1m_albers[:, 3] > aoi_native[1])
        )
        candidates = [dict(s1m_tiles[int(i)]) for i in _hits]
        _native_m = 1.0
        _levels = [_native_m * 2**k for k in range(6)]
        _fit = [r for r in _levels if r <= _target_m]
        read_res_m = _fit[-1] if _fit else _levels[0]
        read_res = read_res_m  # source units are metres
        is_geographic = False
    else:
        # The seamless COGs are EPSG:4269, so resolution is in DEGREES and a pixel is not
        # square on the ground: at latitude 44 one is ~10.3 m north-south but ~7.4 m
        # east-west. The north-south spacing is the larger of the two and therefore the
        # binding one, so degrees convert with 111_320 and NO cosine. Using the cosine
        # would overstate how fine the data is and let cells through with no pixel in them.
        aoi_native = (_w, _s, _e, _n)
        candidates = [
            dict(t)
            for t in dem_tiles
            if t["bbox"][0] < _e
            and t["bbox"][2] > _w
            and t["bbox"][1] < _n
            and t["bbox"][3] > _s
        ]
        _native_m = DEM_DEG * 111_320.0
        _levels = [_native_m * 2**k for k in range(6)]
        _fit = [r for r in _levels if r <= _target_m]
        read_res_m = _fit[-1] if _fit else _levels[0]
        read_res = read_res_m / 111_320.0  # source units are degrees
        is_geographic = True

    # Ground size of the AOI, needed by everything downstream that talks in metres.
    _latm = (_s + _n) / 2.0
    aoi_w_m = (_e - _w) * 111_320.0 * np.cos(np.radians(_latm))
    aoi_h_m = (_n - _s) * 111_320.0

    # A 1 m box outside every published footprint returns nothing. That is a coverage fact,
    # not a failure, and it must NOT be an mo.stop: halting here starves every downstream
    # cell, so positions and texture never update and the scene sits on the placeholder it
    # was built with, four vertices and a 1x1 texture. That renders as a blank map with no
    # error, which is the worst way to report "there is no data here". Printing instead
    # leaves the drape to paint on a flat mesh, which still tells you where you are.
    if dem_source.value == "1m" and not candidates:
        print(
            "no S1M 1 m coverage for this AOI: the 1 m product only exists inside "
            "published project footprints. The scene will be FLAT (no terrain). Switch "
            "the DEM control to 10 m seamless, which is nationwide, or move the box."
        )

    # THE ESTIMATE, WHICH IS NOT A GUARD. It used to be an mo.stop at 40M cells, and that
    # was the wrong shape twice over: the number was inherited rather than derived, and a
    # stop halts the whole downstream chain instead of telling you what you are about to
    # spend. It also estimates from AOI area over hex area, so it knows nothing about
    # nodata, water or real coverage and reads high on exactly the boxes worth pushing.
    #
    # What it prints is pixels in and cells out. The expensive statement it used to warn
    # about, the h3_grid_disk ring join at ~168 bytes per cell, is parked (see the fold),
    # so the fold is now a plain GROUP BY and much cheaper than this guard ever assumed.
    # None of it is about what deck can draw: the mesh costs the same whatever the cell
    # count is.
    _est_cells = aoi_w_m * aoi_h_m / H3_CELL_M2[h3_res.value]
    _est_px = aoi_w_m * aoi_h_m / read_res_m**2
    print(
        f"AOI {tuple(round(v, 4) for v in bbox)} · {aoi_w_m / 1000:.1f} x "
        f"{aoi_h_m / 1000:.1f} km -> {len(candidates)} DEM COG(s) · reading the "
        f"{read_res_m:.1f} m level of the {dem_source.value} DEM for H3 res "
        f"{h3_res.value} "
        f"(~{H3_CELL_M2[h3_res.value] / read_res_m**2:.0f} px per hex) · "
        f"~{_est_px / 1e6:.1f}M px in, ~{_est_cells / 1e6:.2f}M cells out"
    )
    return aoi_h_m, aoi_native, aoi_w_m, candidates, is_geographic, read_res


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    S3_BASE,
    Window,
    aoi_native,
    asyncio,
    candidates,
    fit_lonlat,
    h3_res,
    is_geographic,
    make_h3_context,
    make_lonlat_udf,
    np,
    pa,
    read_res,
    xr,
):
    # THE READ AND THE FOLD. Stream each COG's AOI window off the chosen overview, hand the
    # grid to xarray-sql, and let ONE query fold pixels into H3 cells. A second statement
    # The k-ring `flow` statement is parked; see the note at the fold.
    #
    # The ONLY thing the two DEM sources differ by is how a pixel becomes a lat/lon:
    #
    #   10 m seamless (EPSG:4269)  the grid IS degrees, so `y` and `x` go into the UDF as
    #                              they stand. No wrapper, no per-tile UDF, nothing.
    #   1 m S1M (EPSG:6350)        the grid is Albers metres, so each tile gets its own
    #                              fitted `to_lonlat_i` UDF and the fold reads its struct.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value

    def _window(reader, aoi):
        pw, ps, pe, pn = aoi
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw, ce = max(pw, bw), min(pe, be)
        cs, cn = max(ps, bs), min(pn, bn)
        if ce <= cw or cn <= cs:
            return None
        col0 = max(0, int((cw - bw) / xres))
        col1 = min(reader.width, int(np.ceil((ce - bw) / xres)))
        row0 = max(0, int((bn - cn) / yres))
        row1 = min(reader.height, int(np.ceil((bn - cs) / yres)))
        if col1 <= col0 or row1 <= row0:
            return None
        return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)

    async def _read_tile(tile):
        g = await GeoTIFF.open(tile["key"], store=_store)
        cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
        fits = [r for r in cands if r.res[0] <= read_res]
        reader = fits[-1] if fits else cands[0]
        win = _window(reader, aoi_native)
        if win is None:
            return None
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)
        # nodata is -999999 and the overviews carry it as a real value in places, so mask
        # on magnitude too or a single sentinel pixel drags a whole cell's mean to -1e6.
        elev[elev < -1e5] = np.nan
        if not np.isfinite(elev).any():
            return None
        left, bottom, right, top = r.bounds
        h, w = elev.shape
        # Pixel CENTRES, in whatever units this source uses.
        yy = top - (np.arange(h) + 0.5) * (top - bottom) / h
        xx = left + (np.arange(w) + 0.5) * (right - left) / w
        ds = xr.Dataset({"elevation": (("y", "x"), elev)}, coords={"y": yy, "x": xx})
        if is_geographic:
            return ds, None
        # Projected: fit this tile's lon/lat here on the main thread. Fitting on the READ
        # window rather than the whole tile keeps the polynomial's domain small, which is
        # what holds the error under a millimetre at order 3.
        fit, err_mm = fit_lonlat(g.crs, (left, bottom, right, top))
        return ds, fit

    print(f"streaming {len(candidates)} DEM COG(s):")
    for _t in candidates:
        print(f"  {S3_BASE}{_t['key']}")

    _reads = [d for d in await asyncio.gather(*[_read_tile(t) for t in candidates]) if d]
    if _reads:
        _datasets = [d for d, _ in _reads]
        _px = sum(int(d["elevation"].size) for d in _datasets)
        print(f"streamed {_px:,} pixels from {len(_datasets)}/{len(candidates)} tile(s)")

        ctx = make_h3_context()
        for _i, (_d, _fit) in enumerate(_reads):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"y": 1024})
            if _fit is not None:
                ctx.register_udf(make_lonlat_udf(f"to_lonlat_{_i}", _fit))

        if is_geographic:
            # `y` and `x` ARE latitude and longitude, so they go into the UDF as they
            # stand. No aliasing layer: that would rename two columns to themselves.
            _union = " UNION ALL ".join(
                f"SELECT h3_latlng_to_cell(y, x, CAST({_res} AS INT)) AS hex, elevation "
                f"FROM dem_{_i} WHERE elevation = elevation"
                for _i in range(len(_datasets))
            )
        else:
            # Albers metres, so the per-tile UDF runs first and the fold reads lat/lon off
            # the struct it returns. The subquery is load-bearing here: `p` has to exist as
            # a column before h3_latlng_to_cell can take p.lat and p.lon.
            _union = " UNION ALL ".join(
                f"SELECT h3_latlng_to_cell(p.lat, p.lon, CAST({_res} AS INT)) AS hex, "
                f"elevation FROM ("
                f"  SELECT to_lonlat_{_i}(x, y) AS p, elevation"
                f"  FROM dem_{_i} WHERE elevation = elevation"
                f")"
                for _i in range(len(_datasets))
            )
        _scene = ctx.sql(
            f"""
            SELECT hex, elevation - MIN(elevation) OVER () AS elevation
            FROM (
                SELECT hex, avg(elevation) AS elevation
                FROM ({_union})
                GROUP BY 1
            )
            """
        ).to_arrow_table()

        h3_table = _scene
        print(f"H3 res {_res}: {h3_table.num_rows:,} cells")

        # ---- RING JOIN, PARKED WHILE WE TEST WITH IMAGERY -----------------------------
        # `flow` (how far each cell sits below its k-ring) only ever fed the Palette
        # source's flow offset, and the drape does not read it. It is also the single most
        # expensive statement in the notebook: unnest explodes every cell into SEVEN rows
        # before aggregating back down, ~168 bytes each, which is most of the fold's peak
        # memory for a number the imagery never uses. Restore this and the flow_gain
        # control together; nothing else depends on it.
        #
        # ctx.from_arrow(_scene, name="scene")
        # h3_table = ctx.sql(
        #     """
        #     WITH ring AS (
        #         SELECT hex, elevation,
        #                unnest(h3_grid_disk(hex, CAST(1 AS INT))) AS nb
        #         FROM scene
        #     )
        #     SELECT r.hex AS hex,
        #            r.elevation AS elevation,
        #            avg(n.elevation) - r.elevation AS flow
        #     FROM ring r
        #     JOIN scene n ON r.nb = n.hex
        #     GROUP BY r.hex, r.elevation
        #     """
        # ).to_arrow_table()
        # -------------------------------------------------------------------------------
    else:
        h3_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
            }
        )
        print("no DEM pixels for this AOI")
    return (h3_table,)


@app.cell
def _():
    # Palette registry: matplotlib + CARTOColors sequential ramps. All luminance-monotonic
    # and free of red/green opposition, so they survive a deuteranope simulation.
    from palettable.matplotlib import Viridis_20, Inferno_20, Magma_20, Plasma_20
    from palettable.cartocolors.sequential import (
        Emrld_7,
        Teal_7,
        BluYl_7,
        Mint_7,
        Sunset_7,
        PurpOr_7,
    )

    PALETTES = {
        "Viridis": Viridis_20,
        "Plasma": Plasma_20,
        "Inferno": Inferno_20,
        "Magma": Magma_20,
        "Emrld": Emrld_7,
        "Teal": Teal_7,
        "BluYl": BluYl_7,
        "Mint": Mint_7,
        "Sunset": Sunset_7,
        "PurpOr": PurpOr_7,
    }
    return (PALETTES,)


@app.cell
def _(np):
    # Separable box blur over cumulative sums: O(n) per axis whatever the radius, and no
    # scipy. Used on the height field, on the shading value, and on hex-folded imagery.
    def box_sum(a, r):
        for axis in (0, 1):
            pad = np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)])
            c = np.cumsum(pad, axis=axis)
            lo = np.take(c, np.arange(0, a.shape[axis]), axis=axis)
            hi = np.take(c, np.arange(2 * r + 1, a.shape[axis] + 2 * r + 1), axis=axis)
            a = hi - lo
        return a

    def box_mean(value, mask, r):
        """NaN-aware blur: normalised convolution, so holes neither bleed nor darken."""
        if r <= 0:
            return value, mask
        return box_sum(value, r), box_sum(mask, r)

    return (box_mean,)


@app.cell
def _(h3_table, np):
    # Cell id -> row in h3_table. Sort once, then every sample is a searchsorted. `ok` is
    # False for cells not in the scene at all (AOI corners, nodata holes), which the texture
    # turns transparent and the mesh turns into zero height.
    _hex = np.asarray(h3_table["hex"]).astype("uint64")
    _order = np.argsort(_hex)
    _sorted = _hex[_order]

    def cell_rows(cells):
        """Map an array of H3 cell ids to (row index, found mask)."""
        if _sorted.size == 0:
            return np.zeros(len(cells), "int64"), np.zeros(len(cells), bool)
        pos = np.clip(np.searchsorted(_sorted, cells), 0, _sorted.size - 1)
        ok = _sorted[pos] == cells
        return _order[pos], ok

    return (cell_rows,)


@app.cell
def _(h3_table, np):
    # THE SHADING VALUE, per cell. Only the Palette texture source reads it; the NAIP
    # sources get their colour from photographs.
    #
    # While the ring join is parked this is just scene-relative elevation. The flow offset
    # (drainage etched into the terrain colour) came back as a second column and went in
    # here as an additive term:
    #
    #     cell_shade = elevation + flow_gain.value * flow
    #
    cell_elev = np.asarray(h3_table["elevation"]).astype("float64")
    cell_shade = cell_elev
    return cell_elev, cell_shade


@app.cell
def _(bbox, cell_rows, coordinates_to_cells, h3_res, np, pa, tex_size):
    # THE TERRAIN LATTICE: one regular lon/lat grid over the whole AOI, carrying the H3
    # index and therefore the height field, the hillshade and the palette.
    #
    # IT IS NO LONGER THE IMAGERY LATTICE, and that split is what the tiled drape is. The
    # two were one grid while there was one texture, but they want different resolutions:
    # heights come from a 10 m DEM, so at a 24 km AOI a 2048 grid is already finer than the
    # source and going further buys nothing, while imagery is 0.6 m and wants every texel it
    # can get. The tile cell below builds its own, much finer lattice per tile for NAIP, and
    # samples this one for height. Terrain stays global and cheap; only pixels get tiled.
    #
    # This is also the expensive half of everything below (a coordinates_to_cells call plus
    # a searchsorted over every texel: 4.2M of each at 2048) and it depends only on geometry,
    # so changing a colour is a colormap over an existing index, not a re-binning. Keeping
    # it off the tile grid means raising the tile count costs no H3 work at all.
    #
    # Row 0 of the image is the SOUTH edge, because the mesh's tex_coord v runs 0..1 south
    # to north and WebGL samples v=0 at the first row. If the scene comes out mirrored
    # vertically, this assumption is the thing to flip.
    _T = tex_size.value
    _tex_lon, _tex_lat = np.meshgrid(
        np.linspace(bbox[0], bbox[2], _T), np.linspace(bbox[1], bbox[3], _T)
    )
    _cells = np.asarray(
        pa.array(
            coordinates_to_cells(_tex_lat.ravel(), _tex_lon.ravel(), h3_res.value)
        ).to_numpy(zero_copy_only=False)
    ).astype("uint64")
    _rows, _ok = cell_rows(_cells)
    texel_rows = _rows.reshape(_T, _T)
    texel_ok = _ok.reshape(_T, _T)
    print(f"texel index: {_T}x{_T} · {texel_ok.mean() * 100:.1f}% landed on a cell")
    return texel_ok, texel_rows


@app.cell
def _(bbox, drape, get_started, mo, naip):
    # STAC ONLY: which NAIP quads exist and which year to use. No pixels move here, so it
    # re-runs on a new AOI and costs a couple of seconds. Skipped entirely on the Palette
    # source, because there is no reason to hit a STAC API to draw a colour ramp.
    # Latched too, and not only for symmetry: this cell depends on `bbox` alone, so
    # without the gate merely opening the notebook would hit a STAC API before you had
    # chosen anything.
    if not get_started() or drape.value == "Palette":
        naip_quads, naip_info = [], None
    else:
        naip_quads, naip_info = naip.naip_quads(bbox)

    if naip_info and naip_info[0] == "error":
        print(f"NAIP STAC unavailable ({naip_info[1]}) — re-run this cell to retry")
    elif naip_info:
        print(
            f"NAIP {naip_info[0]}: {naip_info[1]} quad(s), {naip_info[2]:.0%} of the AOI"
        )
    elif get_started() and drape.value != "Palette":
        print("no NAIP found for this AOI")

    # A drape that silently falls back to a ramp looks like a bug in the drape. Say it.
    naip_note = (
        mo.md(
            f"**No NAIP for this AOI** ({'STAC error' if naip_info and naip_info[0] == 'error' else 'no coverage'})"
            f" — the scene falls back to the palette. The terrain is unaffected."
        )
        if get_started() and drape.value != "Palette" and not naip_quads
        else None
    )
    naip_note
    return (naip_quads,)


@app.cell
def _(aoi_w_m, bbox, np, tex_size, tile_grid):
    # THE TILE GRID, in geometry only. No pixels, no heights: just which sub-box each tile
    # covers and the lon/lat lattice its texture will be sampled on. Its own cell because
    # every later stage (imagery, texture, mesh) iterates this same list, and they must all
    # agree about tile extents down to the float.
    #
    # TILES SHARE THEIR EDGE SAMPLES. Tile i spans lon [w + i*step, w + (i+1)*step] with
    # S+1 samples INCLUSIVE of both ends, so tile i's last column and tile i+1's first
    # column are the same coordinate carrying the same texel and the same vertex height.
    # That is what keeps the seams invisible: not a tolerance, an identity. Slice the tiles
    # exclusively instead and every boundary becomes a visible crack in the mesh and a
    # one-texel discontinuity in the photograph.
    _N, _S = tile_grid.value, tex_size.value
    _w, _s, _e, _n = bbox
    tiles = []
    for _j in range(_N):
        for _i in range(_N):
            _lo0 = _w + (_e - _w) * _i / _N
            _lo1 = _w + (_e - _w) * (_i + 1) / _N
            _la0 = _s + (_n - _s) * _j / _N
            _la1 = _s + (_n - _s) * (_j + 1) / _N
            tiles.append(
                {
                    "bbox": (_lo0, _la0, _lo1, _la1),
                    "lon": np.linspace(_lo0, _lo1, _S + 1),
                    "lat": np.linspace(_la0, _la1, _S + 1),
                }
            )

    # THE BUDGET, printed before anything is spent. Total texels is (grid * tex)^2 and every
    # one of them is 4 bytes resident on the GPU, so 4x4 at 4096 is 1.1 GB and will not fit
    # in a browser tab. Saying so here is cheaper than finding out by crashing the renderer.
    texel_m = aoi_w_m / (_N * _S)
    _vram = len(tiles) * (_S + 1) ** 2 * 4
    print(
        f"drape grid: {_N}x{_N} tiles of {_S}^2 = {_N * _S} texels across the AOI · "
        f"{texel_m:.2f} m per texel · {_vram / 1e6:.0f} MB of texture"
        + ("  <-- over a typical GPU budget" if _vram > 8e8 else "")
    )
    return texel_m, tiles


@app.cell
async def _(GeoTIFF, HTTPStore, Window, naip, naip_quads, np, texel_m, tiles):
    # THE DRAPE STREAM, now once per tile. Each tile reads only the quads that overlap it,
    # through its own window, at the overview matching ITS texel size. That last part is the
    # whole win: with one texture the target resolution was aoi_width/2048 and the reader
    # picked a coarse overview to match, so the detail was discarded inside async-geotiff
    # before it was ever downloaded. Sixteen tiles ask for a level four times finer, and the
    # same code path delivers it.
    #
    # Still the only network I/O the imagery needs: palette, hillshade, smoothing, elevation
    # scale and the tile textures below all run downstream without re-fetching a photograph.
    #
    # SEQUENTIAL over tiles, concurrent within one. `naip_rgb` already gathers across quads,
    # and a 4x4 grid firing every quad of every tile at once is dozens of concurrent range
    # reads against one host, which Planetary Computer throttles. Tiles are the natural
    # place to let it breathe.
    if naip_quads:
        _opened = await naip.open_quads(naip_quads, GeoTIFF, HTTPStore)
        naip_drape = []
        _read, _cov = 0, []
        for _t in tiles:
            _lon, _lat = np.meshgrid(_t["lon"], _t["lat"])
            _rgb, _c, _info = await naip.naip_rgb(
                naip_quads, _lon, _lat, _t["bbox"], texel_m,
                GeoTIFF, HTTPStore, Window, _opened,
            )
            naip_drape.append((_rgb, _c))
            _read += _info["quads_read"]
            _cov.append(_info["covered"])
        _src = min((r.res[0] for r in _opened.values()), default=float("nan"))
        print(
            f"NAIP drape: {len(tiles)} tile(s), {_read} quad read(s) total · "
            f"{texel_m:.2f} m per texel vs {_src:.1f} m native "
            f"({texel_m / _src:.0f}x coarser) · "
            f"{float(np.mean(_cov)) * 100:.1f}% of the lattice painted"
        )
    else:
        naip_drape = [
            (
                np.zeros((len(_t["lat"]), len(_t["lon"]), 3), dtype="uint8"),
                np.zeros((len(_t["lat"]), len(_t["lon"])), dtype=bool),
            )
            for _t in tiles
        ]
    return (naip_drape,)


@app.cell
def _(
    aoi_h_m,
    aoi_w_m,
    box_mean,
    cell_elev,
    np,
    relief_smooth,
    texel_ok,
    texel_rows,
):
    # THE HEIGHT FIELD, in texture space, and this is where "angular and unnatural" gets
    # fixed. Straight off the fold, height is PIECEWISE CONSTANT: every hexagon is a flat
    # plateau with a vertical step to its neighbour. Sample that densely and you get literal
    # hexagonal stairs; sample it coarsely and you get arbitrary facets. Angular either way,
    # and no mesh density fixes it, because the staircase is in the DATA.
    #
    # So blur the height field itself. relief_smooth is in texels and turns the plateaus
    # into a continuous surface. Deliberately SEPARATE from the colour smooth: this one
    # changes the shape (and therefore the hillshade), that one only changes the ramp.
    _elev = np.where(texel_ok, cell_elev[texel_rows] if cell_elev.size else 0.0, 0.0)
    _mask = texel_ok.astype("float64")
    _v, _m = box_mean(_elev, _mask, int(relief_smooth.value))
    height_tex = np.divide(_v, _m, out=np.zeros_like(_v), where=_m > 0)

    # Ground metres per texel, for the hillshade gradient.
    px_m_x = aoi_w_m / _elev.shape[1]
    px_m_y = aoi_h_m / _elev.shape[0]
    return height_tex, px_m_x, px_m_y


@app.cell
def _(elevation_scale, height_tex, hillshade, np, px_m_x, px_m_y):
    # THE HILLSHADE, in numpy because deck cannot compute it: lonboard's SurfaceLayer ships
    # exactly two mesh attributes, POSITION and TEXCOORD_0, with NO NORMAL, so deck's
    # lighting has nothing to work with and the surface renders effectively unlit. Extruded
    # prisms looked better partly because their vertical walls catch light for free. Adding
    # normals means patching lonboard's JS, so the light is baked into the texture instead.
    #
    # It applies to the PALETTE SOURCE ONLY. NAIP arrives with real sunlight already in it,
    # and a second synthetic sun on top double-shades the photograph into something glossy.
    # See the texture cell.
    #
    # Sun at 315/45, the cartographic convention. Row index increases NORTH (row 0 is the
    # south edge), so the y gradient is already d/d(north). The gradient uses
    # elevation_scale, the SAME exaggeration the mesh uses, so what reads as steep is steep.
    _z = height_tex * max(elevation_scale.value, 1e-6)
    _dzdy, _dzdx = np.gradient(_z, px_m_y, px_m_x)

    _nx, _ny, _nz = -_dzdx, -_dzdy, np.ones_like(_z)
    _norm = np.sqrt(_nx * _nx + _ny * _ny + 1.0)

    _az, _alt = np.radians(315.0), np.radians(45.0)
    _lx = np.cos(_alt) * np.sin(_az)
    _ly = np.cos(_alt) * np.cos(_az)
    _lz = np.sin(_alt)

    _hs = np.clip((_nx * _lx + _ny * _ly + _nz * _lz) / _norm, 0.0, 1.0)

    # Ambient floor so shadowed faces keep their hue instead of going to black, then blend
    # by strength: 0 leaves the colours exactly as they arrived.
    AMBIENT = 0.35
    _f = AMBIENT + (1.0 - AMBIENT) * _hs
    shade_factor = 1.0 + hillshade.value * (_f - 1.0)
    return (shade_factor,)


@app.cell
def _(cell_shade, mo, np):
    # Contrast window over the shading value. Its bounds ARE this scene's range, so it
    # resets per AOI and per flow offset. Own cell, depending on cell_shade alone: palette
    # and reverse must never reach it, or picking a palette would rebuild the slider and
    # throw away the window you dragged. Inert on the NAIP sources.
    if cell_shade.size:
        _lo, _hi = float(np.floor(cell_shade.min())), float(np.ceil(cell_shade.max()))
    else:
        _lo, _hi = 0.0, 1.0
    if _hi <= _lo:
        _hi = _lo + 1.0
    contrast = mo.ui.range_slider(
        start=_lo, stop=_hi, value=[_lo, _hi],
        step=max((_hi - _lo) / 200.0, 0.1),
        label="Shading contrast (m)",
        show_value=True, full_width=True, debounce=True,
    )
    return (contrast,)


@app.cell
def _(bbox, np):
    # NEAREST-SAMPLE a global terrain-lattice array onto one tile's much finer lattice.
    # Separable, so the row and column indices are built once as 1-D and combined with
    # np.ix_ rather than materialising a pair of (S+1)^2 index grids per tile per array.
    def tile_sample(arr, tile):
        _g = arr.shape[0]
        _c = np.clip(
            ((tile["lon"] - bbox[0]) / (bbox[2] - bbox[0]) * (_g - 1)).round(), 0, _g - 1
        ).astype("int64")
        _r = np.clip(
            ((tile["lat"] - bbox[1]) / (bbox[3] - bbox[1]) * (_g - 1)).round(), 0, _g - 1
        ).astype("int64")
        return arr[np.ix_(_r, _c)]

    return (tile_sample,)


@app.cell
def _(
    PALETTES,
    apply_continuous_cmap,
    box_mean,
    cell_shade,
    colour_smooth,
    contrast,
    contrast_value,
    np,
    palette,
    reverse_ramp,
    shade_factor,
    texel_ok,
    texel_rows,
):
    # THE PALETTE, computed ONCE on the terrain lattice rather than once per tile. It is a
    # function of elevation alone, so nothing about it varies across the grid, and every
    # tile just samples the result. The blur especially wants to be global: run per tile it
    # would see each tile's edge as the edge of the data and bend the ramp there.
    #
    # Blur the shading VALUE, not the finished RGB: the ramp still spans the same contrast
    # window, so it behaves like a coarser fold rather than a soft-focus filter, and it
    # happens before the hillshade so softening never flattens relief.
    _ = contrast
    _shade = np.where(texel_ok, cell_shade[texel_rows] if cell_shade.size else 0.0, 0.0)
    _mask = texel_ok.astype("float64")
    _v, _m = box_mean(_shade, _mask, int(colour_smooth.value))
    _shade = np.divide(_v, _m, out=np.zeros_like(_v), where=_m > 0)

    _lo, _hi = float(contrast_value[0]), float(contrast_value[1])
    _norm = np.clip((_shade - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
    if reverse_ramp.value:
        _norm = 1.0 - _norm

    _c = np.asarray(
        apply_continuous_cmap(_norm.ravel(), PALETTES[palette.value], alpha=1.0)
    )
    # The hillshade is baked here, i.e. on the palette and nowhere else. See below.
    palette_rgb = (
        _c[:, :3].astype("float64").reshape(*texel_ok.shape, 3) * shade_factor[..., None]
    )
    return (palette_rgb,)


@app.cell
def _(brightness, drape, naip_drape, np, palette_rgb, texel_ok, tile_sample, tiles):
    # THE TEXTURES, one RGBA image per tile. The loop is the only structural change tiling
    # forces on this cell: the per-source logic below is what it always was, applied N^2
    # times to arrays that each cover 1/N^2 of the ground at N times the resolution.
    #
    # `visible` is what the mesh may show, and it differs by source. The palette covers
    # every cell in the scene; NAIP covers only where a quad was actually read, so an AOI
    # that runs off the edge of the imagery goes transparent there rather than painting
    # black over real terrain. A tile with NO imagery at all falls back to the palette on
    # its own, which is why the test is per tile and not per scene: a grid straddling the
    # edge of a NAIP year gets photograph where there is photograph and ramp elsewhere,
    # rather than all-or-nothing.
    textures = []
    for _k, _t in enumerate(tiles):
        _rgb_src, _cover = naip_drape[_k]
        _ok = tile_sample(texel_ok, _t)
        if drape.value == "NAIP" and _cover.any():
            rgb = _rgb_src.astype("float64")
            visible = _ok & _cover
        else:
            rgb = tile_sample(palette_rgb, _t)
            visible = _ok

        # BRIGHTNESS, on the imagery only, and it is a GAMMA rather than a gain.
        #
        # NAIP over forest is genuinely dark (a Presidentials frame averages RGB
        # 104/120/114) and a plain multiply would push the open ground and the summit rock
        # to pure white before the tree canopy became readable, because the thing you want
        # to see is in the shadows and the thing that clips is not. A gamma lifts the low
        # end and leaves 255 pinned, so detail arrives out of the dark without blowing
        # anything out. 1.0 is the untouched photograph.
        #
        # Deliberately NOT wired to the palette: that ramp is already calibrated by the
        # contrast window and gamma-shifting it would misreport elevation. The test is on
        # `_cover`, not on the dropdown, so a tile that fell back to the ramp is not
        # gamma-shifted while its NAIP neighbours are.
        if _cover.any() and drape.value != "Palette" and brightness.value != 1.0:
            rgb = 255.0 * np.power(np.clip(rgb, 0, 255) / 255.0, 1.0 / brightness.value)

        # THE HILLSHADE IS FOR THE PALETTE ONLY, and it is already baked into palette_rgb
        # upstream rather than applied here.
        #
        # A palette encodes height in hue and nothing else, so it needs a synthetic sun to
        # read as terrain. A NAIP frame is a PHOTOGRAPH: it was taken in real sunlight and
        # it already carries the real shadows, the real aspect shading, and the real time of
        # day. Multiplying a second 315/45 sun over that is double-shading. It lights slopes
        # the actual sun did not light, fights the shadows that are already in the pixels,
        # and the result reads as a glossy shell rather than as ground. Draped imagery
        # should look like the raster it is.
        #
        # Relief still reads on the NAIP sources, from the two places it should: the mesh is
        # genuinely 3D under a pitched camera, and the photograph brought its own light.

        _tex = np.empty((*visible.shape, 4), dtype="uint8")
        _tex[..., :3] = np.clip(rgb, 0, 255).astype("uint8")
        # Cut alpha with the ORIGINAL mask: the blur widens the valid region, and without
        # this the scene grows a soft fringe past its own extent.
        _tex[..., 3] = np.where(visible, 255, 0)
        textures.append(_tex)

    _bytes = sum(t.nbytes for t in textures)
    _opaque = float(np.mean([(t[..., 3] > 0).mean() for t in textures]))
    print(
        f"texture [{drape.value}]: {len(textures)} x {textures[0].shape[1]}"
        f"x{textures[0].shape[0]} ({_bytes / 1e6:.1f} MB total) · "
        f"{_opaque * 100:.1f}% opaque"
    )
    return (textures,)


@app.cell
def _(aoi_w_m, mesh_density, np, tile_grid):
    # MESH TOPOLOGY, built ONCE and shared by every tile. Vertex count is (n+1)^2 and
    # triangle count is 2n^2, fixed by the slider and independent of the cell count: this is
    # the whole reason a swath is affordable. Own cell so moving the elevation scale
    # re-uploads positions without rebuilding indices.
    #
    # The slider is the density ACROSS THE AOI, so it is divided by the grid to hold the
    # triangle budget constant as tiles are added: N^2 tiles of (density/N)^2 quads is the
    # same 2*density^2 triangles however the grid is set. Tiling is for texels, not for
    # geometry, and a 10 m DEM does not get finer because the imagery did.
    #
    # Vectorised: lonboard's own generate_mesh_grid() writes these indices with a Python
    # double loop, a quarter of a million iterations at density 512.
    _n = max(8, mesh_density.value // tile_grid.value)
    _u = np.linspace(0.0, 1.0, _n + 1, dtype="float32")
    _UU, _VV = np.meshgrid(_u, _u)
    tex_coords = np.stack([_UU.ravel(), _VV.ravel()], axis=-1).astype("float32")

    _i = np.arange(_n)
    _r, _c = np.meshgrid(_i, _i, indexing="ij")
    _bl = (_r * (_n + 1) + _c).ravel()
    _br = _bl + 1
    _tl = _bl + (_n + 1)
    _tr = _tl + 1
    triangles = np.empty((_n * _n * 2, 3), dtype="uint32")
    triangles[0::2] = np.stack([_bl, _br, _tl], axis=-1)
    triangles[1::2] = np.stack([_br, _tr, _tl], axis=-1)

    print(
        f"mesh: {tile_grid.value ** 2} x ({len(tex_coords):,} vertices, "
        f"{len(triangles):,} triangles) · "
        f"{aoi_w_m / (_n * tile_grid.value):.1f} m per quad"
    )
    return tex_coords, triangles


@app.cell
def _(bbox, elevation_scale, height_tex, np, tex_coords, tiles):
    # MESH POSITIONS, one array per tile. Height comes from the SMOOTHED texture-space
    # height field, not from a per-vertex cell lookup, which matters twice over: the relief
    # blur has already removed the hexagonal staircase, and sampling the same array the
    # hillshade was computed from means shading and shape cannot drift apart.
    #
    # EVERY TILE SAMPLES THE SAME GLOBAL FIELD, and that is what welds the grid shut. A
    # vertex on the boundary between two tiles is at one lon/lat, so it indexes one entry of
    # height_tex and both tiles place it at exactly the same height. There is no seam to
    # reconcile because there is no second opinion about the ground. Give each tile its own
    # local height field and this is where the cracks would come from.
    _G = height_tex.shape[0]
    positions = []
    for _t in tiles:
        _w, _s, _e, _n = _t["bbox"]
        _lon = _w + tex_coords[:, 0] * (_e - _w)
        _lat = _s + tex_coords[:, 1] * (_n - _s)

        _c = np.clip(
            ((_lon - bbox[0]) / (bbox[2] - bbox[0]) * (_G - 1)).round(), 0, _G - 1
        ).astype("int64")
        _r = np.clip(
            ((_lat - bbox[1]) / (bbox[3] - bbox[1]) * (_G - 1)).round(), 0, _G - 1
        ).astype("int64")
        _z = height_tex[_r, _c] * elevation_scale.value

        # float32 is what the trait casts to anyway: ~1 m of positional quantisation at lat
        # 44, invisible against a 24 km AOI.
        positions.append(np.stack([_lon, _lat, _z], axis=-1).astype("float32"))
    return (positions,)


@app.cell
def _(
    CartoBasemap,
    FullscreenControl,
    Map,
    MaplibreBasemap,
    NavigationControl,
    ScaleControl,
    SurfaceLayer,
    bbox,
    np,
):
    # The layers and the Map are built ONCE, from placeholder geometry. This cell references
    # no control at all, so marimo never re-runs it and the view you flew to survives every
    # adjustment. The update cell at the bottom pushes the real arrays in.
    #
    # A FIXED POOL, sized to the largest grid the dropdown offers, because the Map must not
    # be rebuilt and `Map.layers` must not be reassigned: either one throws away the camera.
    # So all 16 exist from the start and the update cell fills as many as the current grid
    # needs, parking the rest on a degenerate triangle with a transparent texel. An unused
    # layer costs one draw call of nothing.
    #
    # SurfaceLayer never populates _bbox (it has no geoarrow table to derive one from), so
    # the view state has to be explicit or the Map opens on null island. Zoom 10.5 rather
    # than the surface notebook's 12.5: a swath needs to be stood back from.
    MAX_TILES = 16
    surfaces = [
        SurfaceLayer(
            positions=np.zeros((4, 3), dtype="float32"),
            triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype="uint32"),
            tex_coords=np.zeros((4, 2), dtype="float32"),
            texture=np.zeros((1, 1, 4), dtype="uint8"),
        )
        for _ in range(MAX_TILES)
    ]
    scene = Map(
        layers=surfaces,
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 10.5,
            "pitch": 62,
            "bearing": -25,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter),
        controls=[
            FullscreenControl(position="top-right"),
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
        parameters={"depthTest": True, "blend": True},
    )
    scene
    return (surfaces,)


@app.cell
def _(PALETTES, mo):
    # EVERY SCENE CONTROL, in one cell, directly under the map it drives. Three rows: where
    # the colour comes from, what the surface looks like, then what it is. None of them
    # rebuild the Map.
    drape = mo.ui.dropdown(
        options=["NAIP", "Palette"],
        value="NAIP",
        label="Texture source",
    )
    hillshade = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6, label="Hillshade", show_value=True
    )
    brightness = mo.ui.slider(
        start=0.4, stop=3.0, step=0.1, value=1.0, label="Brightness", show_value=True
    )
    colour_smooth = mo.ui.slider(
        start=0, stop=24, step=1, value=2, label="Colour smooth", show_value=True
    )

    palette = mo.ui.dropdown(options=list(PALETTES), value="Emrld", label="Palette")
    reverse_ramp = mo.ui.switch(value=True, label="Reverse")
    # Flow offset parked with the ring join that fed it:
    # flow_gain = mo.ui.number(
    #     start=0.0, stop=50.0, step=0.5, value=8.0, debounce=True, label="Flow offset"
    # )

    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=2.0, debounce=True, label="Elevation scale"
    )
    relief_smooth = mo.ui.slider(
        start=0, stop=24, step=1, value=3, label="Relief smooth", show_value=True
    )
    mesh_density = mo.ui.slider(
        start=64, stop=2048, step=64, value=1024, label="Mesh density", show_value=True
    )
    tex_size = mo.ui.dropdown(
        options={"1024": 1024, "2048": 2048, "4096": 4096},
        value="2048",
        label="Texture / tile",
    )
    # THE TILE GRID, and it is the control that lifts the resolution ceiling. Ground
    # resolution is aoi_width / (grid * texture), so 4x4 at 2048 is 8192 texels across the
    # AOI: four times finer than any single texture can be, because it is sixteen textures.
    # Kept as a small fixed set rather than a slider because every step is 2x the VRAM and
    # 4x the download, and MAX_TILES below is sized to the largest option.
    tile_grid = mo.ui.dropdown(
        options={"1x1": 1, "2x2": 2, "3x3": 3, "4x4": 4}, value="2x2", label="Drape tiles"
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Opacity"
    )
    wireframe = mo.ui.switch(value=False, label="Wireframe")

    mo.vstack(
        [
            mo.hstack(
                [drape, brightness, hillshade, colour_smooth], justify="start", gap=2
            ),
            mo.hstack([palette, reverse_ramp], justify="start", gap=2),
            mo.hstack(
                [elevation_scale, relief_smooth, mesh_density, tex_size, tile_grid,
                 fill_opacity, wireframe],
                justify="start", gap=2,
            ),
        ],
        gap=0.75,
    )
    return (
        brightness,
        colour_smooth,
        drape,
        elevation_scale,
        fill_opacity,
        hillshade,
        mesh_density,
        palette,
        relief_smooth,
        reverse_ramp,
        tex_size,
        tile_grid,
        wireframe,
    )


@app.cell
def _(PALETTES, contrast, drape, mo, palette, reverse_ramp):
    # The slider paints the ramp it controls: same palette, same DIRECTION as the scene, so
    # "reversed" is something you see rather than infer, and the strip doubles as the
    # legend. Both go quiet on the NAIP sources, which have no ramp to explain.
    contrast_value = contrast.value
    if drape.value == "Palette":
        _hex = PALETTES[palette.value].hex_colors
        if reverse_ramp.value:
            _hex = _hex[::-1]
        _out = mo.vstack(
            [
                mo.Html(
                    '<div style="height:14px;width:100%;border-radius:3px;'
                    'border:1px solid rgba(128,128,128,0.35);'
                    f'background:linear-gradient(to right,{",".join(_hex)});"></div>'
                ),
                contrast,
            ],
            gap=0,
        )
    else:
        _out = mo.md(
            "<small>Palette, reverse, contrast and **hillshade** drive the "
            "**Palette** source only: NAIP is a photograph and already carries real "
            "sunlight, so a second synthetic sun would double-shade it. Smoothing and the "
            "mesh controls apply to every source.</small>"
        )
    _out
    return (contrast_value,)


@app.cell
def _(
    fill_opacity,
    np,
    positions,
    surfaces,
    tex_coords,
    textures,
    triangles,
    wireframe,
):
    # The only thing the controls do: swap traits on the running layers. No Map rebuild, no
    # re-stream, no re-fold, no re-bin.
    #
    # BATCHED PER LAYER, because positions, tex_coords and triangles have to agree about
    # vertex indices. Moving the mesh density slider changes all three, and if they reach
    # the widget one at a time the frontend briefly holds indices that point past the end of
    # the buffer.
    _blank_pos = np.zeros((4, 3), dtype="float32")
    _blank_tri = np.array([[0, 1, 2], [1, 3, 2]], dtype="uint32")
    _blank_uv = np.zeros((4, 2), dtype="float32")
    _blank_tex = np.zeros((1, 1, 4), dtype="uint8")

    for _k, _layer in enumerate(surfaces):
        with _layer.hold_trait_notifications():
            if _k < len(positions):
                _layer.positions = positions[_k]
                _layer.tex_coords = tex_coords
                _layer.triangles = triangles
                _layer.texture = textures[_k]
                _layer.wireframe = wireframe.value
                _layer.opacity = fill_opacity.value
            else:
                # Shrinking the grid has to actively blank the surplus layers. Leaving last
                # run's geometry on them draws a coarse copy of the scene underneath the new
                # one, which reads as z-fighting rather than as a stale layer.
                _layer.positions = _blank_pos
                _layer.tex_coords = _blank_uv
                _layer.triangles = _blank_tri
                _layer.texture = _blank_tex
    return


if __name__ == "__main__":
    app.run()
