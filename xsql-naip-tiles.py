# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "duckdb>=1.5.5",
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
"""RESOLUTION IS A SETTING, NOT A CONSEQUENCE OF HOW BIG YOUR BOX IS.

This is `xsql-naip-ndvi.py` with one assumption removed, and every limit that notebook
documented so carefully turns out to have been that assumption talking.

THE ASSUMPTION. That notebook builds ONE regular lattice over the AOI, `tex_size` square,
and defines everything on it: heights, hillshade, the H3 key, NDVI, the textures. Its
spacing is therefore `AOI / tex_size`. So is the mesh quad, `AOI / mesh_density`. So is the
DEM overview it streams. Draw a box twice as wide and every one of those numbers doubles.
Resolution was never a control; it was a function of the box, and the controls could only
choose which constant to divide by.

That is why a 24 km box came back with 1.46 m texels on 23 m triangles and a sawtooth of
triangle silhouettes at a near-ground camera, and why the honest advice had become "draw a
smaller box". The advice was correct and the architecture was the reason.

THE INVARIANT THIS NOTEBOOK IS BUILT TO: **no array here is ever AOI-sized.** Every array
is tile-sized. A tile is a fixed number of texels at a fixed metres-per-texel, the DEM and
NAIP are read PER TILE WINDOW straight out of the COGs, and the AOI decides only how many
tiles exist. Ground resolution is constant. A wider box costs more tiles; it never costs
sharpness.

  `Detail` (m/texel) and `Tile texels` are the two numbers that set resolution, and they
  are in the top row with the DEM source because they decide what gets FETCHED. Box width
  is not in that list any more. That is the whole change.

THE TILE GRID IS GLOBAL, NOT AOI-RELATIVE. Tiles are a quadtree over the degree grid: at
level k a tile is `2^-k` degrees, so every tile boundary is an exact binary fraction and
every lattice coordinate is `(i * T + n) * step` for integers. Two consequences, both load
bearing:

  * Neighbouring tiles compute their shared edge COORDINATE bit-identically, so the vertex
    goes in the same place. The height sampled there is not quite bit-identical, because
    the two tiles read different windows of the COG and `bilinear` divides by window-derived
    numbers: measured 2.4e-4 m on a 2000 m mountain, i.e. float32 eps, four orders below
    the float32 floor the positions are quantised to anyway. The height cell measures this
    on every run and shouts above 1 cm.
  * The grid does not move when you move the box, so a tile means the same ground every
    time and the scheme is ready for a camera-driven version without being re-derived.

THE FOLD NEEDS NO MARGIN, AND THIS IS THE NICE RESULT. `avg`, `min` and `max` are
decomposable, so each tile runs a DataFusion `GROUP BY` over ONLY the pixels it owns (a
half-open clip, so no pixel is counted twice anywhere), and one final `GROUP BY` merges the
partials by cell id:

    -- per tile, over that tile's own DEM pixels and its own NAIP texels
    SELECT hex, sum(elevation), count(*), min(elevation), max(elevation), ... GROUP BY hex
    -- once, over the union of every tile's partials
    SELECT hex, sum(se)/sum(ne) AS elevation, max(mx) - min(mn) AS relief ... GROUP BY hex

That is bit-for-bit the global fold, computed without ever holding the AOI in memory, and
an H3 cell that straddles a tile boundary gets one value rather than two. Partial
aggregation is the reason tiling costs the analysis nothing.

WHAT THE TILING DOES NOT BUY, said plainly. Constant resolution over a bounded AOI is
arithmetic: 50 km at 1 m/texel is 2.5 x 10^9 texels and no architecture puts that on a GPU.
What changes is that the notebook now TELLS you, in tiles and megabytes and triangles,
instead of quietly handing back a blurrier picture. The budget cell prints the whole cost
and stops with the specific knob to turn. Unbounded extent needs view-dependent residency,
which is the same tile machinery with the camera choosing the tile set, and it is the next
thing rather than a different thing.

TWO FLOORS THAT ARE REAL, NEITHER OF THEM THE ARCHITECTURE:

  * `SurfaceLayer.positions` is `list<float32, 3>` and lonboard exposes no coordinate
    origin, so lon/lat vertices snap to a fixed grid: MEASURED at the Wasatch box, 0.64 m
    east-west (longitude -111.7 sits in the [64, 128) binade) and 0.42 m north-south.
    Below about 1.5 m per quad the mesh is precision-limited rather than
    architecture-limited. Tiles still agree exactly at their edges (same lon in, same
    float32 out), so this is a sub-metre irregularity and never a crack. Printed as a NOTE
    when you cross it.
  * NAIP is 0.6 m native and the 10 m seamless DEM is 10 m. Detail finer than the source
    is interpolation, and the read cell says so.

EITHER DEM, STILL A DROPDOWN, and S1M's two old humiliations were both `AOI / tex_size`
talking: "reading the 16 m overview of a 1 m product" and "64 tiles is too many". Read
resolution now comes from `Detail`, so 1 m S1M reads native 1 m at ANY box size, and the
COG count is a streaming cost rather than a rendering limit.

FOUR SURFACES, unchanged in meaning: `NAIP RGB`, `NDVI`, `Elevation`, `Relief`. NDVI is
blue-to-yellow, never red-to-green, which is exactly the pair a deuteranope cannot resolve.

REQUIRED SETUP, and nothing looks right without it:

    uv run python tools/patch_lonboard_surface.py

deck lights a normal-less mesh per triangle; the script turns that off. Re-run after any
install and HARD-RELOAD the browser. There is a check below that shouts if it is missing.

DO NOT RUN THIS ONE WITH `--sandbox`. The patch edits the JS bundle in site-packages of
whatever environment it was run against, which is the project `.venv`. `--sandbox` resolves
the PEP 723 header into a SEPARATE ephemeral environment, so it gets stock lonboard, and the
surface comes back covered in pale facets with no way to reach it from inside the notebook.
The header is kept in sync anyway, because it is what makes the file readable on its own,
but the sandbox is not the way to run a SurfaceLayer notebook.

Run:  uv run marimo edit xsql-naip-tiles.py

`naip.py` must sit next to this file.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import time
    import urllib.request
    import xml.etree.ElementTree as ET
    from datetime import timedelta
    from io import BytesIO

    import duckdb
    import numpy as np
    import palettable
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    import geoarrow.rust.core as grc
    from pyproj import Transformer

    from arro3.core import Table
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
        duckdb,
        grc,
        mo,
        naip,
        np,
        pa,
        palettable,
        pathlib,
        time,
        timedelta,
        udf,
        urllib,
        xr,
    )


@app.cell
def _(BytesIO):
    # THE PARQUET PATCH. Without this the kernel SEGFAULTS the moment a SurfaceLayer is
    # constructed, with no traceback: lonboard prefers pyarrow's Parquet writer to ship
    # synced columns, and pyarrow 25 crashes on a 3-wide FixedSizeList that arrived over
    # the arro3 C Data Interface, which is exactly what `positions` is. Forcing arro3's
    # writer takes the crashing path out of the picture. Measured in xsql-s1m-surface.py.
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
def _(pathlib):
    # THE LIGHTING CHECK, because the failure it catches is silent, looks like six other
    # bugs, and cost an entire session before it was understood. deck lights a normal-less
    # mesh per triangle; `tools/patch_lonboard_surface.py` turns that off. The patch lives
    # in site-packages, so every `uv sync`, every lonboard bump and every `--sandbox` run
    # quietly reverts it. Checking is one read of the bundle.
    import lonboard as _lb

    _bundle = pathlib.Path(_lb.__file__).parent / "static" / "index.js"
    if "material:!1" in _bundle.read_text():
        print("lonboard patched: SurfaceLayer is unlit, textures render as themselves")
    else:
        print(
            "!! LONBOARD IS NOT PATCHED. The mesh will come back covered in pale\n"
            "!! quadrilateral facets: deck is lighting it per triangle because\n"
            "!! SurfaceLayer ships no NORMAL and SimpleMeshLayer answers that with\n"
            "!! flatShading. No control in this notebook can reach it.\n"
            "!!   uv run python tools/patch_lonboard_surface.py\n"
            "!! then restart the kernel AND hard-reload the browser tab."
        )
    return


@app.cell
def _(mo):
    mo.md("""
    # Constant-resolution tiles

    Resolution is set by **Detail** (metres per texel) and **Tile texels**, not by how big
    a box you draw. Tiles are a quadtree over the degree grid, the DEM and NAIP are read
    per tile window straight out of the COGs, and **nothing here is ever AOI-sized**. A
    wider box costs more tiles; it never costs sharpness.

    The budget cell prints tiles, megabytes and triangles before anything streams, and
    stops with the knob to turn if the box asks for more than a GPU will hold. That is the
    trade made visible, which is the thing the previous notebooks could not do.

    Draw a box (Ctrl/Cmd + drag).
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, np, pa, udf):
    # TWO CRSs, AND THE FOLD HAS TO SPEAK BOTH. The seamless 10 m COGs are EPSG:4269, so
    # their grid coordinates already ARE degrees. S1M tiles are EPSG:6350 Albers, and the
    # fold has to reach degrees from metres INSIDE a DataFusion UDF, where pyproj cannot go:
    # called from a worker thread it does not raise, it ABORTS THE PROCESS. So lon/lat is
    # fitted as an order-3 polynomial on the main thread and the UDF closes over the
    # coefficients and is pure numpy.
    #
    # ONE FIT PER SOURCE COG, not per render tile. The polynomial is accurate over the
    # extent it was fitted on, an S1M COG is a 10 km square, and order 3 clears a
    # millimetre over that by a wide margin. Every render tile inside that COG reuses the
    # same fit, which is what keeps a 144-tile scene from doing 144 pyproj fits.
    PROJ_ORDER = 3

    def _design(u, v, order=PROJ_ORDER):
        cols = [np.ones_like(u)]
        for total in range(1, order + 1):
            for i in range(total + 1):
                cols.append(u ** (total - i) * v**i)
        return np.column_stack(cols)

    def fit_lonlat(crs, bounds, samples=12, check=64, tol_mm=1.0):
        """Fit lon/lat over a COG's extent. Main thread only: this is the pyproj call."""
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
        """One UDF per source COG, its fitted coefficients closed over. Pure numpy."""
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

    def make_ctx():
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
        return ctx

    def cells_of(lat, lon, res):
        """Lattice -> cell ids, as a plain uint64 array. Used to paint results back."""
        return np.asarray(
            pa.array(coordinates_to_cells(lat.ravel(), lon.ravel(), res)).to_numpy(
                zero_copy_only=False
            )
        ).astype("uint64")

    print("DataFusion context factory ready (h3_latlng_to_cell, per-COG lon/lat fits)")
    return cells_of, fit_lonlat, make_ctx, make_lonlat_udf


@app.cell
def _(mo):
    # THE RESOLUTION CONTROLS ARE UP HERE WITH THE DEM SOURCE, and that placement is the
    # argument of the notebook. `Detail` and `Tile texels` decide what is FETCHED: the
    # overview level of every COG read, the spacing of every lattice, the ground size of a
    # tile and therefore how many tiles a box needs. In the previous notebooks the number
    # that did all of that was the box width, which is not a control at all.
    #
    # Detail finer than the source is interpolation, not detail: NAIP is 0.6 m native and
    # the 10 m seamless DEM is 10 m, so 0.5 m only means something on S1M lidar under a
    # NAIP campaign that was flown at 0.6 m. The read cells print what they actually got.
    detail = mo.ui.dropdown(
        options={
            "0.5 m / texel": 0.5,
            "1 m / texel": 1.0,
            "2 m / texel": 2.0,
            "4 m / texel": 4.0,
            "8 m / texel": 8.0,
            "16 m / texel": 16.0,
            "32 m / texel": 32.0,
        },
        value="4 m / texel",
        label="Detail",
    )
    # Texels per tile edge. Bigger tiles mean fewer draw calls and fewer COG windows for
    # the same ground, smaller tiles mean finer granularity in what a budget can hold. The
    # product `detail * tile_texels` is the tile's ground size, which the budget cell
    # snaps to the nearest power-of-two fraction of a degree and reports back.
    tile_texels = mo.ui.dropdown(
        options={"256": 256, "512": 512, "1024": 1024},
        value="512",
        label="Tile texels",
    )
    _OPTS = {
        8: "res 8 ·  ~1.2 km hex",
        9: "res 9 ·  ~400 m hex",
        10: "res 10 ·  ~150 m hex",
        11: "res 11 ·  ~57 m hex",
        12: "res 12 ·  ~22 m hex",
        13: "res 13",
    }
    h3_res = mo.ui.dropdown(
        options={v: k for k, v in _OPTS.items()},
        value=_OPTS[10],
        label="H3 resolution",
    )
    naip_season = mo.ui.dropdown(
        options={
            "Any (best mosaic)": "any",
            "Prefer leaf-off": "prefer",
            "Leaf-off only": "off",
        },
        value="Any (best mosaic)",
        label="NAIP season",
    )
    # IMAGERY IS A FETCH SWITCH, which is why it sits with the other fetch controls rather
    # than with `Colour by` under the map. Off means the STAC search does not run and not a
    # single NAIP quad is opened, so the DEM path is the whole cost of a box: useful when
    # the AOI is outside a campaign, when the imagery is slow, or when the terrain is the
    # only thing being looked at. The photograph and NDVI surfaces read as Elevation while
    # it is off, rather than rendering an empty mesh.
    naip_on = mo.ui.switch(value=True, label="NAIP imagery")
    dem_source = mo.ui.dropdown(
        options={
            "10 m seamless (nationwide)": "13",
            "1 m S1M lidar (partial coverage)": "s1m",
        },
        value="10 m seamless (nationwide)",
        label="DEM source",
    )
    mo.vstack(
        [
            mo.hstack(
                [detail, tile_texels, h3_res, dem_source, naip_season, naip_on],
                justify="start",
                gap=2,
            ),
            mo.md(
                "<small>**Detail** and **Tile texels** set resolution; the box only sets "
                "how many tiles. A wider box costs tiles, never sharpness, and the budget "
                "cell says what it costs before anything streams. **1 m S1M** turns on "
                "the coverage carpet in the picker and now reads native 1 m at any box "
                "size, because read resolution comes from Detail rather than from the "
                "AOI. **H3 resolution** is an analysis choice: the shape comes from the "
                "DEM and does not move when you change it. **NAIP imagery** off skips the "
                "STAC search and every quad read, and the imagery surfaces fall back to "
                "Elevation. None of these fetch anything until you draw a box.</small>"
            ),
        ],
        gap=0.5,
    )
    return dem_source, detail, h3_res, naip_on, naip_season, tile_texels


@app.cell
def _(ET, pathlib, urllib):
    # THE CATALOG IS THE VRT. USGS publishes one nationwide .vrt listing every 1-degree
    # seamless COG on prd-tnm with its exact placement, so AOI -> hrefs is a local bbox
    # intersection. ~830 KB, cached next to the notebook, no STAC API and no signing.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"
    CACHE = pathlib.Path(".cache")

    _vrt = CACHE / "USGS_Seamless_DEM_13.vrt"
    if not _vrt.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("downloading the 1/3 arc-second seamless VRT index (~830 KB)...")
        urllib.request.urlretrieve(
            S3_BASE + "StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt", _vrt
        )

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

    DEM_DEG = abs(_gt[1])
    print(
        f"10 m seamless index: {len(dem_tiles):,} COGs · "
        f"native {DEM_DEG * 3600:.3f}\" ({DEM_DEG * 111_320:.1f} m)"
    )
    return CACHE, DEM_DEG, S3_BASE, dem_tiles


@app.cell
def _(CACHE, S3_BASE, duckdb, np, urllib):
    # THE S1M CATALOG, AND IT IS NOT A VRT. The 1 m product publishes no nationwide .vrt:
    # the only national catalog is one ~15 MB GeoPackage of tile footprints. duckdb spatial
    # reads it directly, ST_Read parses the GeoPackage geometry blobs and ST_Transform
    # takes Albers to degrees. The `current` layer is already one row per tile.
    _gpkg = CACHE / "S1M_Products.gpkg"
    if not _gpkg.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("downloading the S1M footprint index (~15 MB, once)...")
        _tmp = _gpkg.with_suffix(".part")
        urllib.request.urlretrieve(
            S3_BASE
            + "StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg",
            _tmp,
        )
        _tmp.replace(_gpkg)

    _con = duckdb.connect()
    _con.sql("INSTALL spatial; LOAD spatial;")

    # CORNERS, NOT A LON/LAT ENVELOPE. A 10 km Albers square is a slightly ROTATED quad in
    # degrees, so drawing it as an axis-aligned box would smear the national grid into a
    # staircase. The Albers envelope is kept for tile selection, which happens in Albers
    # where the boxes really are boxes; the four corners are transformed for drawing.
    _rows = _con.sql(
        f"""
        WITH e AS (
            SELECT tile, dataset_link, production_date, z_max,
                   ST_XMin(geom) AS aw, ST_YMin(geom) AS asouth,
                   ST_XMax(geom) AS ae, ST_YMax(geom) AS an
            FROM ST_Read('{_gpkg}', layer='current')
        ), c AS (
            SELECT *,
                ST_Transform(ST_Point(aw, asouth), 'EPSG:6350', 'EPSG:4326', always_xy := true) AS p_sw,
                ST_Transform(ST_Point(ae, asouth), 'EPSG:6350', 'EPSG:4326', always_xy := true) AS p_se,
                ST_Transform(ST_Point(ae, an), 'EPSG:6350', 'EPSG:4326', always_xy := true) AS p_ne,
                ST_Transform(ST_Point(aw, an), 'EPSG:6350', 'EPSG:4326', always_xy := true) AS p_nw
            FROM e
        )
        SELECT tile, dataset_link, production_date, z_max, aw, asouth, ae, an,
               ST_X(p_sw), ST_Y(p_sw), ST_X(p_se), ST_Y(p_se),
               ST_X(p_ne), ST_Y(p_ne), ST_X(p_nw), ST_Y(p_nw)
        FROM c
        """
    ).fetchall()

    s1m_albers = np.array([r[4:8] for r in _rows], dtype="float64")
    s1m_tiles = [
        {
            "tile": r[0],
            "key": r[1].split("amazonaws.com/", 1)[-1],
            "produced": r[2] or "",
            # z_min carries a nodata sentinel (-999999) wherever a tile has holes, so the
            # coverage shading reads z_max only.
            "z_max": float(r[3]) if r[3] is not None else float("nan"),
            "albers": tuple(r[4:8]),
            "quad": [(r[8], r[9]), (r[10], r[11]), (r[12], r[13]), (r[14], r[15])],
        }
        for r in _rows
    ]
    print(f"S1M index: {len(s1m_tiles):,} current 1 m tiles from {_gpkg}")
    return s1m_albers, s1m_tiles


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
    # THE COVERAGE CARPET: the entire 1 m product as geometry, drawn before anything is
    # picked. It answers the only question that matters before you draw a box on S1M, which
    # is whether S1M exists there at all. Shaded by z_max on viridis at low opacity and with
    # NO outlines, so neighbouring tiles blend into one field rather than 11,749 boxes. The
    # answer is the PRESENCE of the shape, never its hue.
    _wkts = pa.array(
        [
            "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in [*t["quad"], t["quad"][0]]) + "))"
            for t in s1m_tiles
        ]
    )
    _geom = grc.from_wkt(_wkts, to_type=grc.from_wkt(_wkts).type.with_crs("EPSG:4326"))

    _zmax = np.array([t["z_max"] for t in s1m_tiles], dtype="float64")
    _zmax = np.where(np.isfinite(_zmax), _zmax, 0.0)
    _lo, _hi = float(np.percentile(_zmax, 1)), float(np.percentile(_zmax, 99))
    _norm = np.clip((_zmax - _lo) / max(_hi - _lo, 1.0), 0.0, 1.0)

    coverage_layer = SolidPolygonLayer(
        table=Table.from_arrow(
            pa.table(
                {
                    "tile": pa.array([t["tile"] for t in s1m_tiles]),
                    "produced": pa.array([t["produced"] for t in s1m_tiles]),
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
        visible=False,  # flipped on by the DEM source, downstream, without a rebuild
    )
    print(
        f"S1M coverage carpet: {len(s1m_tiles):,} footprints · "
        f"viridis over tile z_max {_lo:.0f} m (dark) to {_hi:.0f} m (bright)"
    )
    return (coverage_layer,)


@app.cell
def _(mo):
    # Big Cottonwood and Little Cottonwood, Wasatch: forested canyons against bare ridges
    # and a city edge. ~14 x 13 km.
    #
    # Others worth pasting in:
    #   Flagstaff + Mt Elden, AZ  [-111.72, 35.15, -111.55, 35.27]
    #   Sangre de Cristo, NM      [-105.70, 36.50, -105.30, 36.85]
    #   Presidentials, NH         [-71.42, 44.16, -71.15, 44.36]
    #   Columbia Gorge, OR        [-121.95, 45.55, -121.75, 45.72]
    #
    # BOX WIDTH IS NO LONGER THE SETTING THAT MATTERS MOST, which is the point of this
    # notebook. It sets the tile COUNT and therefore the bill; it does not set the
    # resolution of anything. Detail does.
    get_bbox, set_bbox = mo.state([-111.79, 40.55, -111.63, 40.66])

    # The first-run latch: opening the notebook should stop at the picker. Drawing a box is
    # the run, because it is the one input nothing can default.
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
    # Both hosts are ArcGIS MapServer, so both are /tile/{z}/{y}/{x}: ROW BEFORE COLUMN. In
    # XYZ order they return tiles from the wrong place rather than 404ing, which reads as a
    # projection bug rather than a typo.
    _USGS = "https://basemap.nationalmap.gov/arcgis/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    _ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"

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
        max_requests=-1,  # HTTP/2, so let the browser pipeline rather than capping at 6
    )

    # Built once and referencing no reactive UI element, so pan/zoom/AOI survive every
    # downstream run. Nothing ever reassigns .layers.
    picker = Map(
        layers=[basemap_layer, coverage_layer],
        view_state={"longitude": -111.71, "latitude": 40.60, "zoom": 10.5, "pitch": 0},
        basemap=MaplibreBasemap(style=CartoBasemap.Positron),
        controls=[
            GeocoderControl.from_geopy(
                Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
            ),
            FullscreenControl(position="top-right"),
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
    # Its own cell, downstream of the picker, so choosing a basemap never rebuilds the Map
    # and never disturbs a box you have already drawn.
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
                "<small>Ctrl/Cmd + drag to draw an AOI. The **10 m** DEM is nationwide, so "
                "anywhere in CONUS works; **1 m S1M** draws its coverage carpet here and "
                "only exists inside it. NAIP is checked before anything is "
                "streamed.</small>"
            ),
        ],
        gap=0.5,
    )
    return basemap_choice, basemap_opacity


@app.cell
def _(
    BASEMAPS,
    basemap_choice,
    basemap_layer,
    basemap_opacity,
    coverage_layer,
    dem_source,
):
    # Live trait swap: nudge the running layer rather than reassigning picker.layers, which
    # would rebuild the deck stack. max_zoom moves with the URL, not after it: leaving a
    # deep Esri max_zoom on a USGS service asks for tiles that do not exist and the basemap
    # goes blank exactly when you zoom in to place a box.
    _url, _maxz = BASEMAPS[basemap_choice.value]
    basemap_layer.max_zoom = _maxz
    basemap_layer.data = _url
    basemap_layer.opacity = basemap_opacity.value

    coverage_layer.visible = dem_source.value == "s1m"
    return


@app.cell
def _(get_bbox):
    bbox = list(get_bbox())
    return (bbox,)


@app.cell
def _(naip_on, surface):
    # THE SURFACE ACTUALLY RENDERED, which is `Colour by` unless imagery is switched off,
    # in which case the two surfaces that ARE imagery become Elevation. Every downstream
    # cell reads `view` rather than `surface.value`, so the switch is honoured in one place
    # instead of eleven, and no cell has to decide separately whether a NaN texture means
    # "no coverage here" or "imagery is off".
    #
    # It lives in its own cell, ahead of the STAC search, because that search ends in an
    # `mo.stop` on low coverage: anything defined beside it would stop with it, and the
    # legend has to keep working when the gate trips.
    view = (
        "Elevation"
        if not naip_on.value and surface.value in ("NDVI", "NAIP RGB")
        else surface.value
    )
    return (view,)


@app.cell
def _(bbox, get_started, mo, naip, naip_season, view):
    # THE IMAGERY QUESTION FIRST, BEFORE ANY PIXEL IS STREAMED. The STAC search is seconds
    # and no pixels; the DEM is minutes and many. On an NDVI surface, imagery is not a
    # decoration but the measurement, so an AOI without it has nothing to show.
    #
    # The campaign is chosen ONCE FOR THE WHOLE AOI and every tile reads through it. That
    # is deliberate: NAIP year selection is a coverage argument over the box, and letting
    # it vary per tile would put a different flight day on either side of a tile edge.
    MIN_COVER = 0.50

    _needs_naip = view in ("NDVI", "NAIP RGB")
    if not get_started() or not _needs_naip:
        naip_quads, naip_info = [], None
    else:
        naip_quads, naip_info = naip.naip_quads(bbox, naip_season.value)

    _bad = naip_info is not None and naip_info[0] in ("error", "none")
    naip_cover = 0.0 if (naip_info is None or _bad) else float(naip_info[2])
    if not _needs_naip or not get_started():
        naip_cover = 1.0  # nothing to gate: elevation and relief need no imagery at all

    if _bad and naip_info[0] == "error":
        print(f"NAIP STAC unavailable ({naip_info[1]}) — re-run this cell to retry")
    elif _bad:
        print(f"NAIP: {naip_info[1]}")
    elif naip_info:
        print(
            f"NAIP {naip_info[0]}: {naip_info[1]} quad(s), {naip_info[2]:.0%} of the AOI"
            f" · {naip_info[3]}"
        )
        if naip_info[4]:
            print(f"  NOTE: {naip_info[4]}")
    elif get_started() and _needs_naip:
        print("no NAIP found for this AOI")

    # THE COVERAGE GATE LIVES HERE rather than with the tile grid, so that it only stops
    # the imagery path. The DEM stream is upstream of nothing in this cell and stays that
    # way: switching Colour by must not re-stream a single COG window.
    mo.stop(
        _needs_naip and get_started() and naip_cover < MIN_COVER,
        mo.md(
            f"### NAIP covers only {naip_cover:.0%} of this box\n"
            f"Below the {MIN_COVER:.0%} floor, so **no imagery has been fetched**. Move "
            f"the box, widen the **NAIP season**, or switch the surface to Elevation or "
            f"Relief, which need no imagery at all."
        ),
    )

    _partial = (
        mo.md(
            f"**NAIP covers {naip_cover:.0%} of this box** — texels outside the imagery "
            f"fall back to the elevation ramp."
        )
        if naip_quads and naip_cover < 0.995
        else None
    )
    _partial
    return (naip_quads,)


@app.cell
def _(bbox, detail, get_started, mo, np, tile_texels):
    # THE TILE GRID, AND IT IS A QUADTREE OVER THE DEGREE GRID RATHER THAN A DIVISION OF THE
    # BOX. At level k a tile is 2^-k degrees, so every boundary is an exact binary fraction
    # and every lattice coordinate is (i*T + n) * step for integers i and n. Two things
    # follow and both are load bearing:
    #
    #   * Neighbouring tiles compute their SHARED EDGE COORDINATE BIT-IDENTICALLY. They
    #     therefore sample the same DEM value and place the vertex in exactly the same
    #     spot. Seams weld by identity, not by tolerance.
    #   * The grid does not move when the box moves, so a tile means the same ground on
    #     every redraw. This is what makes the scheme ready for a camera-driven tile set
    #     without re-deriving any of it.
    #
    # THE HALO is fixed at 24 texels and read on every tile, so the height smoothing and
    # the hillshade gradient at a tile's edge see real neighbouring ground rather than a
    # one-sided difference. Without it every tile boundary would carry a shading seam. It
    # is fixed rather than derived from `smooth` so that moving a scene control never
    # re-streams a COG.
    # THE CAPS, AND THEY ARE DELIBERATELY GENEROUS, because the whole point of this
    # notebook is that big boxes are the requirement. MAX_TILES no longer has anything to do
    # with a layer pool: the Map cell allocates one blank layer and the update cell grows the
    # list to the tile count, so an unrun notebook holds one layer rather than a thousand.
    # What MAX_TILES bounds now is DRAW CALLS, which is a real per-frame cost, and
    # SOFT_TILES warns about it well before the refusal.
    #
    # Sized against a real target: a 24 x 25 km box at ~1.5 m/texel is about 840 MB of
    # texture no matter how it is tiled, because that is just area over texel area. A cap
    # under that would refuse the exact scene this notebook was written to fix.
    MAX_TILES = 1024
    SOFT_TILES = 384
    # Texture is GPU-side and the height field is kernel-side, so they are NOT the same
    # megabyte and are capped separately. The height field pays a halo surcharge that
    # shrinks as tiles get bigger: (T + 49)^2 / (T + 1)^2 is 1.20x at T = 512 but 1.09x at
    # T = 1024, which is the real argument for raising Tile texels on a large scene.
    MAX_TEX_MB = 1600.0
    MAX_HGT_MB = 1800.0
    HALO = 24

    T = tile_texels.value
    _w, _s, _e, _n = bbox
    lat_mid = (_s + _n) / 2.0
    _coslat = float(np.cos(np.radians(lat_mid)))
    aoi_w_m = (_e - _w) * 111_320.0 * _coslat
    aoi_h_m = (_n - _s) * 111_320.0

    # Snap the requested tile size to the nearest power-of-two fraction of a degree. The
    # request is `detail * T` metres; what you get is reported, because rounding to the
    # quadtree can move it by up to a factor of sqrt(2) either way.
    _target_deg = detail.value * T / 111_320.0
    level = int(np.clip(round(np.log2(1.0 / max(_target_deg, 1e-9))), 0, 18))
    tile_deg = 2.0 ** -level
    step = tile_deg / T

    m_texel_ns = tile_deg * 111_320.0 / T
    m_texel_ew = m_texel_ns * _coslat
    m_texel = max(m_texel_ns, m_texel_ew)  # the binding one, for choosing overviews

    _i0, _i1 = int(np.floor(_w / tile_deg)), int(np.ceil(_e / tile_deg))
    _j0, _j1 = int(np.floor(_s / tile_deg)), int(np.ceil(_n / tile_deg))
    nx, ny = max(_i1 - _i0, 1), max(_j1 - _j0, 1)

    _k = np.arange(-HALO, T + 1 + HALO, dtype="int64")
    tiles = [
        {
            "ix": _i,
            "iy": _j,
            "bbox": (
                _i * tile_deg,
                _j * tile_deg,
                (_i + 1) * tile_deg,
                (_j + 1) * tile_deg,
            ),
            # Integer-indexed off a global origin, so tile i's last interior sample and
            # tile i+1's first are the SAME float64, not two roundings of the same idea.
            "lon_h": (_i * T + _k) * step,
            "lat_h": (_j * T + _k) * step,
        }
        for _j in range(_j0, _j1)
        for _i in range(_i0, _i1)
    ]

    n_tiles = len(tiles)
    tex_mb = n_tiles * (T + 1) ** 2 * 4 / 1e6
    hgt_mb = n_tiles * (T + 1 + 2 * HALO) ** 2 * 4 / 1e6

    print(
        f"AOI {tuple(round(v, 4) for v in bbox)} · {aoi_w_m / 1000:.1f} x "
        f"{aoi_h_m / 1000:.1f} km"
    )
    print(
        f"tile grid: level {level} ({tile_deg:.6g}°, "
        f"{tile_deg * 111_320.0 * _coslat / 1000:.2f} x "
        f"{tile_deg * 111_320.0 / 1000:.2f} km) · {nx} x {ny} = {n_tiles} tiles · "
        f"{m_texel_ew:.2f} m/texel E-W, {m_texel_ns:.2f} m/texel N-S "
        f"(asked for {detail.value:g})"
    )
    # THE OVERHANG, SAID OUT LOUD, because it is the one cost of anchoring the grid
    # globally instead of to the box. Tiles are whole or they are not tiles, so a box that
    # does not land on tile boundaries is covered by a slightly larger rectangle. At level
    # 6 over a 24 km box that is about 18%, and it buys the bit-identical edges and the
    # cache-friendly "a tile means the same ground every time" property.
    _grid_w_m = nx * tile_deg * 111_320.0 * _coslat
    _grid_h_m = ny * tile_deg * 111_320.0
    _overhang = (_grid_w_m * _grid_h_m) / max(aoi_w_m * aoi_h_m, 1.0)
    print(
        f"budget: {tex_mb:.0f} MB texture (GPU, cap {MAX_TEX_MB:.0f}) + "
        f"{hgt_mb:.0f} MB height field (kernel, cap {MAX_HGT_MB:.0f}) · "
        f"{n_tiles} draw calls · grid covers "
        f"{_grid_w_m / 1000:.1f} x {_grid_h_m / 1000:.1f} km "
        f"({_overhang:.2f}x the AOI, the cost of snapping to a global grid)"
    )
    # The overhang is worst when tiles are LARGE relative to the box, which is the coarse
    # end of Detail, and it is the one place where this scheme does more work than an
    # AOI-fitted grid would. It shrinks as Detail gets finer, so it is not a trap you fall
    # into on the settings that cost the most.
    if _overhang > 1.5:
        print(
            f"  NOTE: the snapped grid covers {_overhang:.2f}x this AOI, because a tile "
            f"({tile_deg * 111_320.0 * _coslat / 1000:.2f} x "
            f"{tile_deg * 111_320.0 / 1000:.2f} km) is large next to the box. Finer "
            f"Detail means smaller tiles and less overhang, as well as more resolution."
        )
    if SOFT_TILES < n_tiles <= MAX_TILES:
        print(
            f"  NOTE: {n_tiles} tiles is {n_tiles} draw calls per frame. It will render, "
            f"but the camera will feel heavy. Raising Tile texels to {min(T * 2, 1024)} "
            f"quarters the count at the same ground resolution and shrinks the halo "
            f"surcharge on the height field."
        )

    mo.stop(not get_started(), mo.md("**Draw a box on the map to start.**"))
    # THE BUDGET IS A GATE, NOT A SILENT DEGRADATION, and that is the difference from every
    # earlier version of this notebook. Constant resolution over a bounded AOI is
    # arithmetic: what will not fit will not fit. The old architecture answered that by
    # quietly coarsening; this one says which number to move.
    # TILE-COUNT GATE, COMMENTED OUT AT STEPHEN'S REQUEST. It refused a 50.8 x 58.1 km box
    # at 4 m/texel, which needs 1470 tiles against the 1024 cap. Uncomment to restore.
    #
    # MAX_TILES bounds DRAW CALLS, not memory and not a layer pool (the pool is gone; the
    # update cell grows the layer list to the tile count). So going past it is a
    # performance decision rather than a correctness one, and it is the user's to make.
    # mo.stop(
    #     n_tiles > MAX_TILES,
    #     mo.md(
    #         f"### That box needs {n_tiles} tiles (cap {MAX_TILES})\n"
    #         f"Nothing has been streamed. At **{detail.value:g} m/texel** with "
    #         f"**{T}** texels a tile covers "
    #         f"{tile_deg * 111_320.0 * _coslat / 1000:.2f} x "
    #         f"{tile_deg * 111_320.0 / 1000:.2f} km, and this AOI is "
    #         f"{aoi_w_m / 1000:.1f} x {aoi_h_m / 1000:.1f} km.\n\n"
    #         f"Coarsen **Detail** (each step doubles tile ground size and quarters the "
    #         f"count), raise **Tile texels**, or draw a smaller box. Resolution and extent "
    #         f"trade against each other here; the notebook will not pick for you."
    #     ),
    # )
    if n_tiles > MAX_TILES:
        print(
            f"  NOTE: {n_tiles} tiles is over the {MAX_TILES} draw-call cap, whose "
            f"mo.stop is commented out above. Proceeding."
        )
    # mo.stop(
    #     tex_mb > MAX_TEX_MB or hgt_mb > MAX_HGT_MB,
    #     mo.md(
    #         f"### That box wants {tex_mb:.0f} MB of texture and {hgt_mb:.0f} MB of "
    #         f"height field\n"
    #         f"Caps are {MAX_TEX_MB:.0f} MB GPU-side and {MAX_HGT_MB:.0f} MB kernel-side. "
    #         f"Nothing has been streamed.\n\n"
    #         f"Texture size is area over texel area and **no tiling changes it**: at "
    #         f"{m_texel_ns:.2f} m/texel this AOI is {tex_mb:.0f} MB however it is cut up. "
    #         f"So the only lever is **Detail** — each step doubles tile ground size and "
    #         f"quarters both numbers — or a smaller box.\n\n"
    #         f"If it is the *kernel* number that is over, raising **Tile texels** to "
    #         f"{min(T * 2, 1024)} helps on its own: the halo surcharge falls from "
    #         f"{(T + 1 + 2 * HALO) ** 2 / (T + 1) ** 2:.2f}x to "
    #         f"{(min(T * 2, 1024) + 1 + 2 * HALO) ** 2 / (min(T * 2, 1024) + 1) ** 2:.2f}x."
    #     ),
    # )
    # THE NAIP COVERAGE GATE IS NOT HERE, DELIBERATELY. It belongs to the imagery cell,
    # because putting it here would make this cell depend on `surface`, and this cell is
    # what `tiles` comes from: every change of the Colour by dropdown would then rebuild
    # the tile set and re-stream every DEM window. The DEM does not care which surface you
    # are looking at, so it must not be downstream of that control.
    return (
        HALO,
        T,
        level,
        m_texel,
        m_texel_ew,
        m_texel_ns,
        n_tiles,
        tile_deg,
        tiles,
    )


@app.cell
def _(
    DEM_DEG,
    Transformer,
    bbox,
    dem_source,
    dem_tiles,
    h3_res,
    m_texel,
    mo,
    np,
    s1m_albers,
    s1m_tiles,
):
    # WHICH COGs, AND AT WHICH OVERVIEW. Read resolution now comes from `Detail` by way of
    # the tile lattice, NOT from the AOI, which is the single line that used to make a wide
    # box read a coarse product. Two masters still, and it takes the finer:
    #
    #   THE HEIGHT FIELD wants roughly one DEM sample per texel.
    #   THE FOLD wants enough pixel centres per hexagon that avg() means something, at
    #     p <= sqrt(2) * 0.5373 * sqrt(A), taken at SAFETY 0.6.
    #
    # Both are now independent of box width, so a 60 km box and a 6 km box at the same
    # Detail read exactly the same overview level.
    H3_CELL_M2 = {8: 737327.6, 9: 105332.5, 10: 15047.5, 11: 2149.6, 12: 307.09, 13: 43.87}
    SAFETY = 0.6

    _w, _s, _e, _n = bbox
    _for_fold = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])
    _target_m = min(_for_fold, m_texel)

    if dem_source.value == "13":
        # The seamless COGs are EPSG:4269, so their grid IS degrees: tile selection is a
        # bbox intersection in the AOI's own units and no projection is needed anywhere.
        dem_crs = "EPSG:4269"
        candidates = [
            dict(t)
            for t in dem_tiles
            if t["bbox"][0] < _e and t["bbox"][2] > _w
            and t["bbox"][1] < _n and t["bbox"][3] > _s
        ]
        # Resolution is in DEGREES and a pixel is not square on the ground: at latitude 40
        # one is ~10.3 m north-south but ~7.9 m east-west. North-south binds, so degrees
        # convert with 111_320 and NO cosine.
        _native_m = DEM_DEG * 111_320.0
        _levels = [_native_m * 2**k for k in range(6)]
        _fit = [r for r in _levels if r <= _target_m]
        read_res_m = _fit[-1] if _fit else _levels[0]
        read_res = read_res_m / 111_320.0  # source units are degrees
        dem_cover = 1.0  # nationwide: there is nothing to check
        _label = "10 m seamless"
    else:
        # S1M ONLY. Tile selection happens in ALBERS, where the footprints really are
        # axis-aligned boxes. Doing it in degrees would mean intersecting rotated quads.
        _fwd = Transformer.from_crs("EPSG:4326", "EPSG:6350", always_xy=True)
        _ax, _ay = _fwd.transform([_w, _e, _e, _w], [_s, _s, _n, _n])
        dem_crs = "EPSG:6350"
        _pw, _ps, _pe, _pn = min(_ax), min(_ay), max(_ax), max(_ay)

        _hit = (
            (s1m_albers[:, 0] < _pe)
            & (s1m_albers[:, 2] > _pw)
            & (s1m_albers[:, 1] < _pn)
            & (s1m_albers[:, 3] > _ps)
        )
        candidates = [dict(s1m_tiles[int(i)]) for i in np.flatnonzero(_hit)]

        # HOW MUCH OF THE BOX ACTUALLY HAS LIDAR. The national grid does not overlap, so
        # the intersecting footprint areas simply sum.
        _iw = np.clip(np.minimum(s1m_albers[_hit, 2], _pe) - np.maximum(s1m_albers[_hit, 0], _pw), 0, None)
        _ih = np.clip(np.minimum(s1m_albers[_hit, 3], _pn) - np.maximum(s1m_albers[_hit, 1], _ps), 0, None)
        _aoi_area = max((_pe - _pw) * (_pn - _ps), 1e-9)
        dem_cover = float(min(np.sum(_iw * _ih) / _aoi_area, 1.0))

        # S1M overviews are a fixed power-of-two ladder in METRES, and the grid is already
        # projected, so read_res needs no conversion at all.
        _levels = [1.0 * 2**k for k in range(6)]
        _fit = [r for r in _levels if r <= _target_m]
        read_res_m = _fit[-1] if _fit else _levels[0]
        read_res = read_res_m  # source units are metres
        _label = "1 m S1M"

    print(
        f"{len(candidates)} {_label} COG(s) · reading the {read_res_m:g} m level "
        f"(lattice wants {m_texel:.2f} m, fold wants {_for_fold:.1f} m) · "
        f"~{H3_CELL_M2[h3_res.value] / read_res_m**2:.0f} px per hex"
    )
    if dem_source.value == "s1m":
        print(f"  S1M coverage of this box: {dem_cover:.0%}")
        # THE OLD NOTE IS GONE ON PURPOSE. It used to say "you asked for 1 m and are
        # reading the 16 m overview, draw a smaller box", which was the AOI-sized lattice
        # talking. Read resolution is a function of Detail now, so 1 m is 1 m at any width.
        if read_res_m > 1.0:
            print(
                f"  reading the {read_res_m:g} m overview because Detail asks for "
                f"{m_texel:.2f} m/texel. Set Detail to 1 m or finer for native lidar."
            )

    mo.stop(
        dem_source.value == "s1m" and not candidates,
        mo.md(
            "### No 1 m lidar here\n"
            "S1M has no tiles under this box, and **nothing has been streamed**. The "
            "coverage carpet in the picker shows where it exists; draw inside it, or "
            "switch the source back to **10 m seamless**, which is nationwide."
        ),
    )
    mo.stop(
        dem_source.value == "s1m" and dem_cover < 0.25,
        mo.md(
            f"### S1M covers only {dem_cover:.0%} of this box\n"
            f"Below the 25% floor, so **nothing has been streamed**. Draw inside the "
            f"coverage carpet, or use the **10 m seamless** DEM."
        ),
    )
    return candidates, dem_crs, read_res


@app.cell
def _(np):
    # BILINEAR, VALID TO THE READ'S PHYSICAL EDGE rather than to the last pixel CENTRE, and
    # the difference is a visible bug. `fx` is in pixel-centre space, so it runs -0.5 at the
    # left edge of column 0 to w-0.5 at the right edge of column w-1. Testing
    # `0 <= i0 < w-1` throws away the outer HALF PIXEL on all four sides, because there is
    # no second pixel to interpolate against out there.
    #
    # Harmless mid-mosaic, fatal at a seam: two adjacent COGs each discard their own half
    # pixel at the shared edge, so the union carries a ONE PIXEL NaN crack along every tile
    # boundary. No height means transparent texture, which renders as a DASHED dark line
    # (dashed, not solid, because a sub-texel crack only catches a texel some of the time).
    #
    # So the outer half pixel EXTRAPOLATES instead of failing: the index pair is clamped to
    # the last valid pair but `tx`/`ty` are left UNCLIPPED, so they run to -0.5 and 1.5 and
    # the same expression becomes a linear extrapolation off the edge pixels. Clipping them
    # to [0, 1] would also close the crack, but by holding the edge value constant, which
    # measured 5 m of error against a known plane versus 3e-14 for extrapolating. Interior
    # samples are bit-identical either way, because out there the clamps never bind.
    def bilinear(elev, bounds, x, y):
        left, bottom, right, top = bounds
        h, w = elev.shape
        if h < 2 or w < 2:
            return None, np.zeros(x.shape, dtype=bool)
        fx = (x - left) / ((right - left) / w) - 0.5
        fy = (top - y) / ((top - bottom) / h) - 0.5
        ok = (fx >= -0.5) & (fx <= w - 0.5) & (fy >= -0.5) & (fy <= h - 0.5)
        if not ok.any():
            return None, ok
        ic = np.clip(np.floor(fx).astype("int64"), 0, w - 2)
        jc = np.clip(np.floor(fy).astype("int64"), 0, h - 2)
        tx = fx - ic
        ty = fy - jc
        v = (
            elev[jc, ic] * (1 - tx) * (1 - ty)
            + elev[jc, ic + 1] * tx * (1 - ty)
            + elev[jc + 1, ic] * (1 - tx) * ty
            + elev[jc + 1, ic + 1] * tx * ty
        )
        return v, ok

    def window_for(reader, bounds, Window):
        """Source-unit bounds -> a read window, clipped to the reader. None if disjoint."""
        pw, ps, pe, pn = bounds
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

    return bilinear, window_for


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    Transformer,
    Window,
    asyncio,
    bilinear,
    candidates,
    dem_crs,
    dem_source,
    fit_lonlat,
    np,
    read_res,
    tiles,
    window_for,
):
    # THE DEM STREAM, PER TILE WINDOW. This is where the AOI-sized array used to be: the
    # old notebook read every COG's whole AOI intersection into one buffer and interpolated
    # a single lattice out of it, so the buffer grew with the box and the lattice did not.
    # Here each render tile asks each overlapping COG for its own window, at a level set by
    # Detail, and nothing bigger than a tile ever exists.
    #
    # COGs are OPENED once and READ many times. Opening is the part that is pure overhead
    # when repeated (one header fetch and parse per open); a windowed read is not, because
    # each tile genuinely wants different bytes. Same argument as `naip.open_quads`.
    #
    # What comes back per tile is BOTH the halo height field AND the raw source pixels with
    # their bounds, because the two consumers want different things: the mesh interpolates
    # the raster, and the fold aggregates it in its native grid through xarray-sql. Keeping
    # both off one read is what stops H3 resolution from re-streaming a COG.
    # TIMEOUTS AND RETRIES ON THE STORE, for the same reason `naip.py` carries them and with
    # the same numbers. obstore's default `timeout` is 30 s wall clock from request to last
    # byte, which is fine for the small reads it was built for and wrong here: a wide box is
    # hundreds of tile windows sharing one link, so each individual read runs long while the
    # transfer is making steady progress the whole time. Without a retry config a read that
    # trips that deadline takes the whole `gather` down; with the default one it can also sit
    # there re-trying invisibly, which is what a stall looks like from the outside.
    #
    # 3 minutes overall, a short connect timeout so a dead host still fails fast, and a read
    # timeout that RESETS on each chunk received, which is the one that should be catching a
    # genuinely stalled transfer rather than a slow one. `retry_timeout` bounds retries from
    # the first attempt, so it has to exceed `timeout` or a retry can never happen.
    _store = S3Store(
        bucket="prd-tnm",
        region="us-west-2",
        skip_signature=True,
        client_options={
            "timeout": "180s",
            "connect_timeout": "15s",
            "read_timeout": "60s",
        },
        retry_config={"max_retries": 6, "retry_timeout": timedelta(minutes=4)},
    )
    _is_s1m = dem_source.value == "s1m"

    # The OPEN phase gets its own line for the same reason the read phase does. On the 10 m
    # product it is one or two headers and finishes before you read the line; on S1M a wide
    # box is hundreds of 10 km tiles, so this is a real wait that happens BEFORE any window
    # is read, and unannounced it looks exactly like a hang at the start of the cell.
    _keys = list({c["key"]: None for c in candidates})
    print(f"opening {len(_keys)} COG header(s)...")
    _t_open = time.monotonic()
    _opened = dict(
        zip(
            _keys,
            await asyncio.gather(*[GeoTIFF.open(k, store=_store) for k in _keys]),
        )
    )
    print(f"  opened {len(_opened)} in {time.monotonic() - _t_open:.1f}s")

    # Level chosen ONCE per COG, not once per tile: it is a function of Detail alone, so
    # every tile reads the same ladder rung and the mosaic is homogeneous.
    _readers = {}
    _fits = {}
    _t_fit = time.monotonic()
    for _k, _g in _opened.items():
        _cands = sorted([_g, *_g.overviews], key=lambda r: r.res[0])
        _f = [r for r in _cands if r.res[0] <= read_res]
        _readers[_k] = _f[-1] if _f else _cands[0]
        # S1M ONLY: one lon/lat polynomial per SOURCE COG, fitted over its full 10 km
        # extent on the main thread, reused by every render tile inside it. pyproj from a
        # DataFusion worker thread does not raise, it aborts the process.
        if _is_s1m:
            _fits[_k] = fit_lonlat(_g.crs, tuple(_g.bounds))[0]
    if _is_s1m:
        # Serial, main-thread, and one per source COG, so on a wide S1M box this is the
        # other place the cell goes quiet before a single window is read.
        print(f"  fitted {len(_fits)} lon/lat polynomial(s) in {time.monotonic() - _t_fit:.1f}s")

    _to_dem = (
        None
        if dem_crs == "EPSG:4269"
        else Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    )
    _sem = asyncio.Semaphore(8)

    async def _read_tile(t):
        # The tile's halo lattice, carried into the DEM's own coordinates. Bilinear only
        # makes sense on the raster's REGULAR grid, so the lattice is projected rather than
        # the raster dragged into degrees. For the 10 m product this is a no-op.
        _lon2, _lat2 = np.meshgrid(t["lon_h"], t["lat_h"])
        if _to_dem is None:
            gx, gy = _lon2, _lat2
        else:
            _x, _y = _to_dem.transform(_lon2.ravel(), _lat2.ravel())
            gx = np.asarray(_x).reshape(_lon2.shape)
            gy = np.asarray(_y).reshape(_lat2.shape)

        need = (float(gx.min()), float(gy.min()), float(gx.max()), float(gy.max()))
        h = np.full(gx.shape, np.nan, dtype="float32")
        raw = []
        for key in {c["key"] for c in candidates}:
            reader = _readers[key]
            # Two pixels of margin, so the bilinear sample at the very edge of the halo
            # still has neighbours instead of falling off the array. reader.res is in the
            # source's units and so is `need`, so this is unit-agnostic.
            pad = reader.res[0] * 2
            win = window_for(
                reader,
                (need[0] - pad, need[1] - pad, need[2] + pad, need[3] + pad),
                Window,
            )
            if win is None:
                continue
            async with _sem:
                r = await reader.read(window=win)
            elev = np.ma.filled(r.as_masked()[0].astype("float32"), np.nan)
            # nodata is -999999 and the overviews carry it as a real value in places, so
            # mask on magnitude too or one sentinel drags a whole cell's mean to -1e6.
            elev[elev < -1e5] = np.nan
            if not np.isfinite(elev).any():
                continue
            v, ok = bilinear(elev, tuple(r.bounds), gx, gy)
            if v is not None:
                take = ok & np.isfinite(v) & ~np.isfinite(h)
                h[take] = v[take]
            raw.append((elev, tuple(r.bounds), _fits.get(key)))
        return h, raw

    # PROGRESS, BECAUSE A WIDE BOX IS MINUTES AND SILENCE IS INDISTINGUISHABLE FROM A HANG.
    # marimo streams stdout as a cell runs, so a line every tenth of the grid is the whole
    # difference between "this is streaming 400 windows" and "this is stuck". The counter is
    # incremented after the await, so it counts tiles actually finished, and the rate it
    # prints is what tells you whether a long run is progressing or crawling.
    _done = [0]  # a list rather than a counter variable: the cell body is module scope, so
    _t0 = time.monotonic()  # a plain int would need `global`, which marimo's `_` names ban
    _step = max(1, len(tiles) // 10)

    async def _tracked(t):
        _r = await _read_tile(t)
        _done[0] += 1
        if _done[0] % _step == 0 or _done[0] == len(tiles):
            _el = time.monotonic() - _t0
            print(
                f"  DEM {_done[0]}/{len(tiles)} tiles · {_el:.0f}s "
                f"({_done[0] / max(_el, 1e-9):.1f} tiles/s)"
            )
        return _r

    tile_dem = await asyncio.gather(*[_tracked(t) for t in tiles])
    tile_h = [d[0] for d in tile_dem]
    tile_raw = [d[1] for d in tile_dem]

    _px = sum(int(e.size) for parts in tile_raw for e, _, _ in parts)
    _hit = float(np.mean([np.isfinite(h).mean() for h in tile_h]))
    print(
        f"DEM: {len(tiles)} tile(s), {sum(len(p) for p in tile_raw)} window read(s) from "
        f"{len(_keys)} COG(s) · {_px:,} px in · {_hit * 100:.1f}% of the lattice covered"
    )
    return tile_h, tile_raw


@app.cell
def _(np, tile_h):
    # THE RAMP WINDOW AND THE DATUM ARE GLOBAL EVEN THOUGH THE ARRAYS ARE NOT, and getting
    # this wrong is the classic tiled-render bug. If each tile zeroed to its own minimum,
    # neighbouring tiles would sit at different heights and the mesh would step at every
    # boundary; if each tile stretched its own colour ramp, the same elevation would be a
    # different colour on either side of a seam.
    #
    # One pass over the tiles for a min and a max is cheap and is the whole fix. It is a
    # reduction, not a materialisation: no AOI-sized array is built to compute it.
    _mins = [float(np.nanmin(h)) for h in tile_h if np.isfinite(h).any()]
    _maxs = [float(np.nanmax(h)) for h in tile_h if np.isfinite(h).any()]
    z_min = min(_mins) if _mins else 0.0
    z_max = max(_maxs) if _maxs else 1.0
    print(f"height datum: {z_min:.0f} m, range {z_min:.0f} to {z_max:.0f} m (global)")
    return z_max, z_min


@app.cell
async def _(
    GeoTIFF,
    HALO,
    HTTPStore,
    T,
    Window,
    asyncio,
    m_texel,
    naip,
    naip_quads,
    np,
    tiles,
    view,
):
    # THE PHOTOGRAPH, PER TILE, at the tile's own texel size. Every tile reads the quads
    # that overlap it through its own window at the overview matching one texel, which is
    # what makes ground resolution independent of how many tiles there are.
    #
    # Interior lattice only, no halo: a photograph needs no gradient and no blur, so it
    # wants exactly the (T+1)^2 texels the texture will hold.
    #
    # BOUNDED CONCURRENCY OVER TILES, not sequential. The notebook this came from ran the
    # tile loop one at a time, reasoning that "firing every quad of every tile at once is
    # dozens of concurrent range reads against one host, each running past its own timeout
    # while making steady progress". That reasoning is right and it argues for a BOUND, not
    # for a one. At 16 tiles the difference did not matter; at 221 tiles, which this
    # notebook reaches routinely, serial round trips are the whole runtime, because each
    # window is small and the cost is latency rather than bytes.
    #
    # `naip_rgb` already caps itself at 8 concurrent reads internally, so the outer bound
    # multiplies with it: a bound of 4 means at most 32 range reads in flight, of
    # windows that are one tile wide at one overview level.
    _TILE_CONCURRENCY = 8
    _sl = slice(HALO, HALO + T + 1)
    if naip_quads and view == "NAIP RGB":
        _opened = await naip.open_quads(naip_quads, GeoTIFF, HTTPStore)
        _outer = asyncio.Semaphore(_TILE_CONCURRENCY)

        async def _one(t):
            _lon, _lat = np.meshgrid(t["lon_h"][_sl], t["lat_h"][_sl])
            async with _outer:
                return await naip.naip_rgb(
                    naip_quads, _lon, _lat, t["bbox"], m_texel,
                    GeoTIFF, HTTPStore, Window, _opened,
                )

        _out = await asyncio.gather(*[_one(t) for t in tiles])
        photo = [(r[0], r[1]) for r in _out]
        _read = sum(r[2]["quads_read"] for r in _out)
        _cov = [r[2]["covered"] for r in _out]
        _src = min((r.res[0] for r in _opened.values()), default=float("nan"))
        print(
            f"NAIP photo: {len(tiles)} tile(s), {_read} quad read(s) · "
            f"{m_texel:.2f} m/texel vs {_src:.2f} m native "
            f"({m_texel / _src:.1f}x coarser) · "
            f"{float(np.mean(_cov)) * 100:.1f}% painted"
        )
    else:
        # ONE placeholder pair, SHARED by every tile rather than one allocation each. The
        # consumers copy (`astype`) or only test (`.any()`), so nothing writes through these
        # and identity is safe; `writeable = False` is what enforces that rather than a
        # comment. It matters at scale: a 500-tile grid at 513^2 is about 500 MB of zeros
        # that exist only to be read as "no imagery here", and that is memory the DEM path
        # is competing for on exactly the wide boxes where imagery gets switched off.
        _blank_rgb = np.zeros((T + 1, T + 1, 3), dtype="uint8")
        _blank_cov = np.zeros((T + 1, T + 1), dtype=bool)
        _blank_rgb.flags.writeable = False
        _blank_cov.flags.writeable = False
        photo = [(_blank_rgb, _blank_cov) for _ in tiles]
    return (photo,)


@app.cell
async def _(
    GeoTIFF,
    HALO,
    HTTPStore,
    T,
    Window,
    asyncio,
    m_texel,
    naip,
    naip_quads,
    np,
    tiles,
    view,
):
    # NDVI, FOUR BANDS, PER TILE. This read exists to produce a NUMBER per texel rather
    # than a picture, and it is skipped entirely when the surface is the photograph or a
    # pure DEM product.
    #
    # NAIP ships R, G, B AND NIR, and a drape throws the fourth band away because a
    # photograph needs three. That band is the cheapest possible demonstration that a drape
    # can carry DATA: same pixels, same stream, one more band.
    # Same bounded concurrency as the photograph, and for the same reason.
    _TILE_CONCURRENCY = 4
    _sl = slice(HALO, HALO + T + 1)
    if naip_quads and view == "NDVI":
        _opened = await naip.open_quads(naip_quads, GeoTIFF, HTTPStore)
        _outer = asyncio.Semaphore(_TILE_CONCURRENCY)

        async def _one_ndvi(t):
            _lon, _lat = np.meshgrid(t["lon_h"][_sl], t["lat_h"][_sl])
            async with _outer:
                return await naip.naip_rgb(
                    naip_quads, _lon, _lat, t["bbox"], m_texel,
                    GeoTIFF, HTTPStore, Window, _opened, bands=4,
                )

        _out = await asyncio.gather(*[_one_ndvi(t) for t in tiles])
        # NDVI = (NIR - R) / (NIR + R). uint8 in, float out; the denominator is guarded
        # because a black pixel is 0/0 and would be a warning and a NaN.
        ndvi = [
            np.where(
                _c,
                (_px[..., 3].astype("float32") - _px[..., 0].astype("float32"))
                / np.maximum(
                    _px[..., 3].astype("float32") + _px[..., 0].astype("float32"), 1.0
                ),
                np.nan,
            ).astype("float32")
            for _px, _c, _ in _out
        ]
        _read = sum(r[2]["quads_read"] for r in _out)
        _cov = [r[2]["covered"] for r in _out]
        _med = np.nanmedian(np.concatenate([a.ravel() for a in ndvi]))
        print(
            f"NDVI: {len(tiles)} tile(s), {_read} quad read(s) at {m_texel:.2f} m/texel · "
            f"{float(np.mean(_cov)) * 100:.1f}% painted · median {_med:+.2f}"
        )
    else:
        # Same shared placeholder as the photograph, same reason: the fold reads a slice of
        # it and nothing writes to it, so one all-NaN array serves the whole grid.
        _blank_ndvi = np.full((T + 1, T + 1), np.nan, dtype="float32")
        _blank_ndvi.flags.writeable = False
        ndvi = [_blank_ndvi for _ in tiles]
    return (ndvi,)


@app.cell
def _(
    HALO,
    T,
    cells_of,
    h3_res,
    make_ctx,
    make_lonlat_udf,
    ndvi,
    np,
    pa,
    tile_raw,
    tiles,
    view,
    xr,
):
    # THE FOLD, AS A PARTIAL AGGREGATION PER TILE AND ONE MERGE AT THE END. This is the
    # cell that makes tiling free for the analysis, and the argument is one line of algebra:
    # `avg`, `min` and `max` are DECOMPOSABLE. A mean is a sum and a count, and both of
    # those are additive across any partition of the pixels.
    #
    # So each tile runs a GROUP BY over ONLY the pixels it owns, and one final GROUP BY
    # merges the partials by cell id. The result is bit-for-bit what a single global fold
    # would have produced, and an H3 cell straddling a tile boundary gets ONE value rather
    # than one per side. Without this, every tile edge would carry a discontinuity in NDVI
    # and Relief, which on a 144-tile scene is a visible grid.
    #
    # OWNERSHIP IS A HALF-OPEN CLIP, and it has to be, or the partials double-count. DEM
    # windows are read with padding and overlap their neighbours, so a tile takes only the
    # pixels whose centres fall in [w, e) x [s, n). Texels are the same: the lattice sample
    # at u = 1 belongs to the next tile, so the fold takes [:-1, :-1].
    #
    # IT RUNS FOR TWO SURFACES AND NO OTHERS, which is the honest scope of what an
    # aggregation buys here. `Relief` NEEDS it: max - min inside a cell is a statistic over
    # a NEIGHBOURHOOD of pixels, so no resampling at any resolution produces it. `NDVI`
    # uses it to average the NIR band per cell, which is what makes the DEM/NAIP join
    # demonstrable at all. `NAIP RGB` and `Elevation` read no cell value: the photograph
    # never did, and elevation is already a bilinear height field that is strictly better
    # than hexagon means.
    _res = h3_res.value
    _sl = slice(HALO, HALO + T + 1)

    if view not in ("NDVI", "Relief"):
        cell_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
                "relief": pa.array([], pa.float64()),
                "n": pa.array([], pa.int64()),
                "ndvi": pa.array([], pa.float64()),
            }
        )
        tile_cells = [np.zeros(0, dtype="uint64") for _ in tiles]
        print(f"fold skipped: [{view}] reads no cell values")
    else:
        _parts = []
        tile_cells = []
        for _ti, (_t, _raw, _nd) in enumerate(zip(tiles, tile_raw, ndvi)):
            _tw, _ts, _te, _tn = _t["bbox"]
            ctx = make_ctx()

            # THE DEM SIDE goes in as its NATIVE GRID, one relation per window, through
            # xarray-sql. Nothing is reprojected and nothing is rasterised to a common
            # grid: that is the whole point of keeping the raw read alongside the lattice.
            _rels = []
            for _i, (_elev, _bounds, _fit) in enumerate(_raw):
                _left, _bottom, _right, _top = _bounds
                _h, _w = _elev.shape
                _yy = _top - (np.arange(_h) + 0.5) * (_top - _bottom) / _h
                _xx = _left + (np.arange(_w) + 0.5) * (_right - _left) / _w
                ctx.from_dataset(
                    f"dem_{_i}",
                    xr.Dataset(
                        {"elevation": (("y", "x"), _elev)}, coords={"y": _yy, "x": _xx}
                    ),
                    chunks={"y": 1024},
                )
                if _fit is not None:
                    ctx.register_udf(make_lonlat_udf(f"to_lonlat_{_i}", _fit))
                # THE ONE PLACE THE CRS SHOWS UP IN SQL. A 10 m relation's y/x already ARE
                # lat/lon and go straight into the cell id; an S1M relation's are Albers
                # metres and pass through that COG's fitted UDF first. After this line
                # nothing downstream can tell which source it is looking at.
                #
                # The half-open ownership clip rides in the same WHERE, in the source's own
                # units for 10 m and in degrees for S1M (where it has to be applied after
                # the fit, because the pixels are not axis-aligned in degrees).
                if _fit is None:
                    _rels.append(
                        f"SELECT h3_latlng_to_cell(y, x, CAST({_res} AS INT)) AS hex, "
                        f"elevation FROM dem_{_i} "
                        f"WHERE elevation = elevation "
                        f"  AND x >= {_tw} AND x < {_te} "
                        f"  AND y >= {_ts} AND y < {_tn}"
                    )
                else:
                    _rels.append(
                        f"SELECT h3_latlng_to_cell(p.lat, p.lon, CAST({_res} AS INT)) "
                        f"AS hex, elevation FROM ("
                        f"  SELECT to_lonlat_{_i}(x, y) AS p, elevation "
                        f"  FROM dem_{_i} WHERE elevation = elevation"
                        f") WHERE p.lon >= {_tw} AND p.lon < {_te} "
                        f"  AND p.lat >= {_ts} AND p.lat < {_tn}"
                    )

            # THE NAIP SIDE arrives pre-indexed: the lattice cells are computed once here
            # and reused to paint the result back, so `coordinates_to_cells` runs a single
            # time per tile rather than once for the fold and once for the render.
            _lon2, _lat2 = np.meshgrid(_t["lon_h"][_sl], _t["lat_h"][_sl])
            _cells = cells_of(_lat2, _lon2, _res).reshape(_lon2.shape)
            tile_cells.append(_cells)

            _own = _cells[:-1, :-1].ravel()  # half-open: u = 1 and v = 1 are the neighbour's
            _flat = _nd[:-1, :-1].ravel()
            _good = np.isfinite(_flat)
            ctx.from_arrow(
                pa.table({"hex": pa.array(_own[_good]), "ndvi": pa.array(_flat[_good])}),
                name="naip_lattice",
            )

            if not _rels:
                continue
            _parts.append(
                ctx.sql(
                    f"""
                    WITH d AS (
                        SELECT hex,
                               sum(elevation) AS se, count(*) AS ne,
                               min(elevation) AS mn, max(elevation) AS mx
                        FROM ({" UNION ALL ".join(_rels)})
                        GROUP BY 1
                    ),
                    v AS (
                        SELECT hex, sum(ndvi) AS sv, count(*) AS nv
                        FROM naip_lattice GROUP BY 1
                    )
                    SELECT
                        coalesce(d.hex, v.hex) AS hex,
                        coalesce(d.se, 0.0) AS se, coalesce(d.ne, 0) AS ne,
                        d.mn AS mn, d.mx AS mx,
                        coalesce(v.sv, 0.0) AS sv, coalesce(v.nv, 0) AS nv
                    FROM d FULL OUTER JOIN v ON d.hex = v.hex
                    """
                ).to_arrow_table()
            )

        if not _parts:
            cell_table = pa.table(
                {
                    "hex": pa.array([], pa.uint64()),
                    "elevation": pa.array([], pa.float64()),
                    "relief": pa.array([], pa.float64()),
                    "n": pa.array([], pa.int64()),
                    "ndvi": pa.array([], pa.float64()),
                }
            )
        else:
            # THE MERGE. One GROUP BY over the union of every tile's partials, and this is
            # the statement that makes the tiling invisible to the analysis: sums add,
            # counts add, mins take a min and maxes take a max, so the answer equals the
            # global fold exactly rather than approximately.
            _merge = make_ctx()
            _merge.from_arrow(pa.concat_tables(_parts), name="partials")
            cell_table = _merge.sql(
                """
                SELECT hex,
                       sum(se) / nullif(sum(ne), 0) AS elevation,
                       max(mx) - min(mn)            AS relief,
                       sum(ne)                      AS n,
                       sum(sv) / nullif(sum(nv), 0) AS ndvi
                FROM partials
                GROUP BY 1
                """
            ).to_arrow_table()

        if cell_table.num_rows:
            _n = np.asarray(cell_table["n"]).astype("float64")
            _nd_col = np.asarray(cell_table["ndvi"]).astype("float64")
            print(
                f"H3 res {_res}: {cell_table.num_rows:,} cells from {len(_parts)} tile "
                f"partial(s) · {_n.mean():.0f} DEM px/cell (min {_n.min():.0f}) · "
                f"{np.isfinite(_nd_col).mean() * 100:.0f}% of cells carry NDVI"
            )
        else:
            print(f"H3 res {_res}: no cells (nothing overlapped this box)")
    return cell_table, tile_cells


@app.cell
def _(cell_table, np, tile_cells):
    # Cell id -> row, then every texel is a searchsorted against ONE global index. The
    # index is global because the cell table is; only the painting is per tile.
    if cell_table.num_rows == 0:
        def cell_field(name, k):
            return None
    else:
        _hex = np.asarray(cell_table["hex"]).astype("uint64")
        _order = np.argsort(_hex)
        _sorted = _hex[_order]

        _rows, _oks = [], []
        for _c in tile_cells:
            _pos = np.clip(np.searchsorted(_sorted, _c.ravel()), 0, _sorted.size - 1)
            _found = _sorted[_pos] == _c.ravel()
            _rows.append(_order[_pos].reshape(_c.shape))
            _oks.append(_found.reshape(_c.shape))

        def cell_field(name, k):
            """A per-cell column painted onto tile k's lattice. NaN where the cell is missing."""
            _v = np.asarray(cell_table[name]).astype("float64")
            return np.where(_oks[k], _v[_rows[k]], np.nan)

        print(
            f"texel index: {len(tile_cells)} tile(s) · "
            f"{float(np.mean([o.mean() for o in _oks])) * 100:.1f}% on a cell"
        )
    return (cell_field,)


@app.cell
def _(np):
    # Separable box blur over cumulative sums: O(n) per axis whatever the radius, no scipy.
    # NaN-aware by normalised convolution, so holes neither bleed nor darken.
    def box_sum(a, r):
        for axis in (0, 1):
            pad = np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)])
            c = np.cumsum(pad, axis=axis)
            lo = np.take(c, np.arange(0, a.shape[axis]), axis=axis)
            hi = np.take(c, np.arange(2 * r + 1, a.shape[axis] + 2 * r + 1), axis=axis)
            a = hi - lo
        return a

    def box_mean(value, mask, r):
        if r <= 0:
            return value, mask
        return box_sum(value, r), box_sum(mask, r)

    return (box_mean,)


@app.cell
def _(HALO, T, box_mean, np, smooth, tile_h, tiles, z_min):
    # THE HEIGHT FIELD THE MESH SAMPLES, SMOOTHED ON THE HALO AND THEN CROPPED. This is the
    # reason the halo exists: a box blur applied to a tile in isolation sees a hard edge at
    # every boundary and returns a different answer there than its neighbour does, which
    # renders as a ridge along every tile seam. Blurring the halo first and cropping after
    # means the interior samples equal what a single global blur would give, MEASURED to
    # 1.3e-12 m at radius 4 and 5.4e-13 m at radius 16 on a 400 m synthetic surface. Not
    # bit-identical, because box_sum accumulates with cumsum and a different starting
    # offset rounds differently; float noise rather than a seam.
    #
    # `smooth` defaults to ZERO. There are no hexagonal plateaus to sand down here, because
    # the height field never went through H3. The control stays because a 10 m DEM has its
    # own noise (collection seams, void fills) and a wide box sometimes reads better with a
    # touch of blur, but it is an aesthetic choice rather than a repair.
    #
    # The datum is the GLOBAL minimum, subtracted after the blur, so tiles cannot float at
    # different heights.
    # The blur radius cannot exceed the halo it is being computed inside, or the crop stops
    # being equal to a global blur. HALO - 2 leaves the one texel the gradient needs.
    _r = int(min(smooth.value, HALO - 2))
    # TWO CROPS, because two consumers want different margins. `height_m` keeps ONE texel
    # of real neighbouring ground on each side, which is exactly what np.gradient needs to
    # use a central difference at the tile's own edge; `height` is the tile proper.
    _wide = slice(HALO - 1, HALO + T + 2)  # (T + 3)^2, for the hillshade
    _tight = slice(HALO, HALO + T + 1)  # (T + 1)^2, the tile itself

    height_m = []
    height = []
    surface_ok = []
    for _h in tile_h:
        _v = np.where(np.isfinite(_h), _h, 0.0).astype("float64")
        _m = np.isfinite(_h).astype("float64")
        _vs, _ms = box_mean(_v, _m, _r)
        _z = np.divide(_vs, _ms, out=np.zeros_like(_vs), where=_ms > 0) - z_min
        height_m.append(_z[_wide, _wide])
        height.append(_z[_tight, _tight])
        # WHERE THE SURFACE EXISTS AT ALL, which is what every texture's alpha starts from.
        # Not "this texel landed on an H3 cell the fold saw", which would quietly make the
        # PHOTOGRAPH depend on the fold; the honest test is whether the mesh has a height.
        surface_ok.append((_ms > 0)[_tight, _tight])

    # THE SEAM CHECK, MEASURED ON EVERY RUN RATHER THAN ASSERTED IN A COMMENT, and it has
    # already corrected the comment once. The lattice COORDINATE really is bit-identical
    # between neighbours (integers times a power-of-two step, verified at five levels), but
    # the HEIGHT sampled at it is not, and the reason is worth keeping:
    #
    #   Two adjacent tiles read DIFFERENT WINDOWS of the same COG. `bilinear` computes
    #   `(x - left) / ((right - left) / w)`, and `left`, `right` and `w` all come from the
    #   window, so the same ground point goes through a different arithmetic path in each
    #   tile and the two results differ in the last few ulps. MEASURED at 2.4e-4 m on a
    #   2000 m mountain, which is exactly float32 eps at that magnitude.
    #
    # That is four orders of magnitude below the float32 floor the POSITIONS are quantised
    # to anyway (~0.42 m north-south), so it is unrepresentable in the render, let alone
    # visible. The threshold below is set where a seam could actually be seen: 1 cm. A hit
    # means something structural, not rounding.
    #
    # Expect ~1e-4 m. Smoothing adds ~1e-12 m of cumsum rounding on top, which is nothing.
    _byxy = {(t["ix"], t["iy"]): _k for _k, t in enumerate(tiles)}
    _worst = 0.0
    _pairs = 0
    for (_ix, _iy), _k in _byxy.items():
        _r_nb = _byxy.get((_ix + 1, _iy))
        if _r_nb is not None:
            _worst = max(_worst, float(np.max(np.abs(height[_k][:, -1] - height[_r_nb][:, 0]))))
            _pairs += 1
        _u_nb = _byxy.get((_ix, _iy + 1))
        if _u_nb is not None:
            _worst = max(_worst, float(np.max(np.abs(height[_k][-1, :] - height[_u_nb][0, :]))))
            _pairs += 1

    print(
        f"height field: {len(height)} tile(s) at {T + 1}^2 · "
        f"{float(np.mean([m.mean() for m in surface_ok])) * 100:.1f}% covered · "
        f"smooth radius {_r} · seam {_worst:.3g} m over {_pairs} shared edge(s)"
    )
    if _worst > 0.01:
        print(
            f"  !! TILES DISAGREE ABOUT THEIR SHARED EDGE BY {_worst:.3g} m, which is "
            f"above the 1 cm threshold and far above the float32 sampling noise this "
            f"normally shows (~1e-4 m). That renders as a ridge or a crack along the "
            f"grid. Either the lattice is no longer integer-indexed off a global origin, "
            f"or the smooth radius has outgrown HALO."
        )
    return height, height_m, surface_ok


@app.cell
def _(elevation_scale, height_m, hillshade, m_texel_ew, m_texel_ns, np):
    # THE HILLSHADE, PER TILE, COMPUTED ON THE ONE-TEXEL MARGIN AND THEN CROPPED. Same
    # argument as the blur: np.gradient falls back to a ONE-SIDED difference at an array
    # edge, so a tile computing its own gradient in isolation disagrees with its neighbour
    # along the whole boundary and paints a bright or dark line there. With one texel of
    # real neighbouring ground on each side every difference is central and the two tiles
    # agree EXACTLY along the seam (measured 0.0, since a gradient accumulates nothing).
    # The no-margin control measures 0.0075 in shade units at the same edge, which is the
    # visible line this removes.
    #
    # It applies to the DATA surfaces only. A NAIP photograph was taken in real sunlight and
    # already carries the real shadows; a second synthetic sun double-shades it into a
    # glossy shell. Sun at 315/45, the cartographic convention. Row index increases NORTH,
    # so the y gradient is already d/d(north).
    _az, _alt = np.radians(315.0), np.radians(45.0)
    AMBIENT = 0.35

    shade = []
    for _h in height_m:
        _z = _h * max(elevation_scale.value, 1e-6)
        _dzdy, _dzdx = np.gradient(_z, m_texel_ns, m_texel_ew)
        _nx, _ny, _nz = -_dzdx, -_dzdy, np.ones_like(_z)
        _norm = np.sqrt(_nx * _nx + _ny * _ny + 1.0)
        _hs = np.clip(
            (
                _nx * (np.cos(_alt) * np.sin(_az))
                + _ny * (np.cos(_alt) * np.cos(_az))
                + _nz * np.sin(_alt)
            )
            / _norm,
            0.0,
            1.0,
        )
        # Ambient floor so shadowed faces keep their hue instead of going black.
        _f = 1.0 + hillshade.value * (AMBIENT + (1.0 - AMBIENT) * _hs - 1.0)
        shade.append(_f[1:-1, 1:-1])
    return (shade,)


@app.cell
def _():
    # Palette registry: matplotlib + CARTOColors sequential ramps. All luminance-monotonic
    # and free of red/green opposition, so they survive a deuteranope simulation.
    #
    # NDVI defaults to BluYl, blue through to yellow, rather than to the red-to-green ramp
    # every remote-sensing tool ships for it. Red-green is exactly the pair a deuteranope
    # cannot resolve, and blue-yellow is the axis that survives.
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
        "BluYl": BluYl_7,
        "Viridis": Viridis_20,
        "Plasma": Plasma_20,
        "Inferno": Inferno_20,
        "Magma": Magma_20,
        "Emrld": Emrld_7,
        "Teal": Teal_7,
        "Mint": Mint_7,
        "Sunset": Sunset_7,
        "PurpOr": PurpOr_7,
    }
    return (PALETTES,)


@app.cell
def _(cell_table, ndvi_range, np, view, z_max, z_min):
    # THE RAMP WINDOW, DECIDED ONCE FOR THE WHOLE SCENE. Per-tile normalisation is the bug
    # this cell exists to prevent: the same elevation, or the same relief, has to be the
    # same colour in every tile or the seams become the most visible thing in the render.
    #
    # NDVI IS WINDOWED, NOT NORMALISED, and for a second reason on top of that one.
    # Stretching a ramp to a scene's own min and max is what makes NDVI maps incomparable
    # between AOIs and between dates: the same forest gets a different colour depending on
    # what else is in the box. -0.2 to 0.8 covers water through bare rock through dense
    # canopy everywhere on earth, so a colour means the same thing in every scene.
    if view == "NDVI":
        ramp_lo, ramp_hi = float(ndvi_range.value[0]), float(ndvi_range.value[1])
    elif view == "Relief":
        _v = np.asarray(cell_table["relief"]).astype("float64")
        _f = _v[np.isfinite(_v)]
        ramp_lo, ramp_hi = (
            (float(np.percentile(_f, 2)), float(np.percentile(_f, 98)))
            if _f.size
            else (0.0, 1.0)
        )
    else:  # Elevation, and also the fallback under the photograph
        ramp_lo, ramp_hi = 0.0, max(z_max - z_min, 1.0)

    if view == "NAIP RGB":
        print(f"elevation ramp held as the fallback: {ramp_lo:.3g} .. {ramp_hi:.3g} m")
    else:
        print(f"surface [{view}]: ramp {ramp_lo:.3g} .. {ramp_hi:.3g} (global)")
    return ramp_hi, ramp_lo


@app.cell
def _(PALETTES, apply_continuous_cmap, np, palette):
    # THE RAMP, RESOLVED ONCE INTO 256 ROWS. Every colour this notebook paints is a lookup
    # into this table. The saving on the kernel side is small and measured (see the texture
    # cell); the point is that the ramp now EXISTS as 1 KB of palette separate from the
    # per-texel index, which is the shape the browser needs if the index is ever to be what
    # crosses the bridge instead of finished RGBA.
    #
    # It is its own cell so that it depends on `palette` ALONE. Reverse is applied to the
    # normalised value rather than to the table, which keeps this off the path of every
    # control that is not the palette itself.
    ramp_lut = np.asarray(
        apply_continuous_cmap(
            np.linspace(0.0, 1.0, 256), PALETTES[palette.value], alpha=1.0
        )
    )[:, :3].astype("uint8")
    return (ramp_lut,)


@app.cell
def _(
    brightness,
    cell_field,
    height,
    np,
    photo,
    ramp_hi,
    ramp_lo,
    ramp_lut,
    reverse_ramp,
    shade,
    surface_ok,
    view,
):
    # THE TEXTURES, one RGBA image per tile, assembled from whichever source the surface
    # asks for. Everything read here is already tile-shaped; nothing is sampled out of a
    # global array, which is the difference from the notebook this came from (where every
    # tile nearest-sampled one AOI-sized lattice and therefore could not be finer than it).
    #
    # `visible` differs by tile rather than by scene: a grid straddling the edge of a NAIP
    # campaign gets photograph where there is photograph and ramp elsewhere.
    textures = []
    for _k in range(len(height)):
        _rgb_src, _cover = photo[_k]
        _ok = surface_ok[_k]
        if view == "NAIP RGB" and _cover.any():
            rgb = _rgb_src.astype("float64")
            visible = _ok & _cover
            # Gamma, not a gain, and only on the photograph: NAIP over forest is genuinely
            # dark and a plain multiply pushes open ground to white before the canopy
            # becomes readable, because what you want to see is in the shadows and what
            # clips is not. A gamma lifts the low end and leaves 255 pinned.
            if brightness.value != 1.0:
                rgb = 255.0 * np.power(
                    np.clip(rgb, 0, 255) / 255.0, 1.0 / brightness.value
                )
        else:
            if view == "NDVI":
                _v = cell_field("ndvi", _k)
            elif view == "Relief":
                _v = cell_field("relief", _k)
            else:
                # STRAIGHT OFF THE HEIGHT FIELD, not off the fold. Hexagon means of the
                # same pixels the height field already samples bilinearly are strictly
                # worse, and this branch is also what the RGB view falls back to where a
                # tile has no imagery.
                _v = height[_k]
            if _v is None:
                _v = np.full(_ok.shape, np.nan)
            _dok = np.isfinite(_v)
            _norm = np.clip(
                (np.where(_dok, _v, ramp_lo) - ramp_lo) / max(ramp_hi - ramp_lo, 1e-9),
                0.0,
                1.0,
            )
            if reverse_ramp.value:
                _norm = 1.0 - _norm
            # A 256-ENTRY LUT AND ONE `take`. NOT for speed: measured against
            # `apply_continuous_cmap` this is 1.1x, because that call was already 2.2 ms
            # per tile and 0.2 s for the whole grid, i.e. never the thing that made a
            # slider feel slow. It is here because it makes the ramp an explicit TABLE.
            # The expensive half of a colour change is the ~92 MB of RGBA crossing the
            # widget bridge, and the only fix for that is to send this table once and the
            # index per texel, so having the table already separated is the first step.
            # Output matches the colormap call to within 2 counts per channel, which is
            # the 256-level quantisation and is invisible.
            rgb = ramp_lut[np.round(_norm * 255.0).astype("uint8")].astype("float64")
            # The hillshade is for the DATA surfaces only. A photograph brought its own sun.
            rgb = rgb * shade[_k][..., None]
            visible = _ok & _dok

        _tex = np.empty((*visible.shape, 4), dtype="uint8")
        _tex[..., :3] = np.clip(rgb, 0, 255).astype("uint8")
        _tex[..., 3] = np.where(visible, 255, 0)
        textures.append(_tex)

    print(
        f"texture: {len(textures)} x {textures[0].shape[1]}x{textures[0].shape[0]} "
        f"({sum(t.nbytes for t in textures) / 1e6:.1f} MB) · "
        f"{float(np.mean([(t[..., 3] > 0).mean() for t in textures])) * 100:.1f}% opaque"
    )
    return (textures,)


@app.cell
def _(T, bbox, m_texel_ew, m_texel_ns, mo, n_tiles, np, texels_per_quad):
    # MESH TOPOLOGY, built once and shared by every tile, because every tile is the same
    # shape. Vertex count is (n+1)^2 and triangle count is 2n^2 PER TILE, and crucially
    # they do NOT get divided by the tile count: geometry scales with the grid exactly the
    # way texels do.
    #
    # That is the direct fix for the imbalance the previous notebook documented and could
    # not resolve. It computed `mesh_density // tile_grid`, holding the triangle budget
    # constant as tiles were added ("tiling is for texels, not for geometry"), which is how
    # a scene ended up with 1.46 m texels on 23 m triangles. Here the ratio between the two
    # is a control, and it is a ratio rather than two independent numbers precisely so it
    # cannot drift.
    _n = max(1, T // texels_per_quad.value)
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

    _lon0 = (bbox[0] + bbox[2]) / 2.0
    _lat0 = (bbox[1] + bbox[3]) / 2.0
    _coslat0 = float(np.cos(np.radians(_lat0)))
    m_quad = max(m_texel_ew, m_texel_ns) * texels_per_quad.value
    # GEOMETRY HAS ITS OWN BUDGET, and it needs one for the same reason the texture does:
    # decoupling triangles from the tile count is the point of this notebook, so the total
    # now grows with BOTH. At 99 tiles and one vertex per texel that is 52M triangles and
    # 313 MB of positions, which no earlier version of this could reach and which will sit
    # a browser down.
    #
    # It is gated HERE rather than with the tile grid on purpose. This cell depends on
    # Texels/quad, and the tile-grid cell is what `tiles` comes from: gating there would
    # put a scene control upstream of the DEM stream and re-download every window whenever
    # the geometry balance was adjusted.
    MAX_POS_MB = 512.0
    pos_mb = n_tiles * len(tex_coords) * 3 * 4 / 1e6
    print(
        f"mesh: {len(tex_coords):,} vertices and {len(triangles):,} triangles per tile · "
        f"{m_quad:.2f} m/quad · {n_tiles * len(triangles) / 1e6:.1f}M triangles and "
        f"{pos_mb:.0f} MB of positions over {n_tiles} tile(s)"
    )
    mo.stop(
        pos_mb > MAX_POS_MB,
        mo.md(
            f"### That geometry is {pos_mb:.0f} MB (cap {MAX_POS_MB:.0f} MB)\n"
            f"{n_tiles} tiles at **{texels_per_quad.value}** texel(s) per quad is "
            f"{n_tiles * len(triangles) / 1e6:.1f}M triangles. The scene still shows the "
            f"last geometry that fit.\n\n"
            f"Raise **Texels / quad** (each step quarters the triangle count), or coarsen "
            f"**Detail** to need fewer tiles. At {m_quad:.2f} m/quad against a float32 "
            f"position floor of about {abs(float(np.spacing(np.float32((bbox[0] + bbox[2]) / 2.0))) * 111_320.0 * np.cos(np.radians((bbox[1] + bbox[3]) / 2.0))):.2f} m, "
            f"finer quads are mostly buying you rounding anyway."
        ),
    )
    # THE FLOAT32 FLOOR, and it is a property of lonboard rather than of this design.
    # SurfaceLayer.positions is list<float32, 3> and there is no coordinate-origin escape,
    # so a lon/lat vertex snaps to `np.spacing` of its own magnitude: measured 0.64 m
    # east-west and 0.42 m north-south at the Wasatch box. Tiles still agree exactly at
    # their edges (same lon in, same float32 out), so this is a sub-metre irregularity in
    # the lattice and never a crack. The floor moves with longitude, hence the live
    # calculation rather than a constant.
    _floor_ew = float(np.spacing(np.float32(_lon0))) * 111_320.0 * _coslat0
    _floor_ns = float(np.spacing(np.float32(_lat0))) * 111_320.0
    _floor = max(abs(_floor_ew), abs(_floor_ns))
    if m_quad < 3.0 * _floor:
        print(
            f"  NOTE: {m_quad:.2f} m/quad is approaching the float32 position floor "
            f"({abs(_floor_ew):.2f} m E-W, {abs(_floor_ns):.2f} m N-S here). Finer quads "
            f"will not place vertices more precisely; raise Texels/quad and spend the "
            f"budget on texels instead."
        )
    return tex_coords, triangles


@app.cell
def _(HALO, T, elevation_scale, height, np, texels_per_quad, tiles):
    # MESH POSITIONS, one array per tile, off that tile's OWN height field. In the previous
    # notebook every tile sampled one shared global lattice, which is what welded the grid
    # shut and also what capped the height detail at that lattice's spacing however many
    # tiles were added.
    #
    # Here the weld comes from the coordinates instead: tile i's last column and tile i+1's
    # first column are the same float64 lon (both are (i+1)*T*step computed from integers),
    # so both tiles read the same DEM sample through the same bilinear and place the vertex
    # in the same place. Identity, not tolerance, and it survives a redraw of the box.
    _q = texels_per_quad.value
    _n = max(1, T // _q)
    _sl = slice(0, T + 1, _q)

    positions = []
    for _t, _h in zip(tiles, height):
        _lon = _t["lon_h"][HALO : HALO + T + 1][_sl]
        _lat = _t["lat_h"][HALO : HALO + T + 1][_sl]
        _LO, _LA = np.meshgrid(_lon, _lat)
        _z = _h[_sl, _sl] * elevation_scale.value
        positions.append(
            np.stack([_LO.ravel(), _LA.ravel(), _z.ravel()], axis=-1).astype("float32")
        )
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
    # The Map is built ONCE. This cell references no control, so marimo never re-runs it and
    # the view you flew to survives every adjustment. The update cell at the bottom puts the
    # real layers in.
    #
    # NO POOL. The notebooks this came from pre-allocated every layer they could ever need
    # here, at Map construction, because they assumed `Map.layers` could not be reassigned
    # without deck throwing away the camera. That assumption is why a pool existed at all,
    # and it is wrong: in lonboard the view is only recomputed by `add_layer(focus=True)` or
    # `reset_zoom=True`, and plain assignment to `.layers` touches nothing.
    #
    # It also had a cost nobody had priced. THIS CELL DEPENDS ON NO CONTROL, which is the
    # whole point of it, but that also means it does not depend on the first-run latch: it
    # runs the moment the notebook opens, while the pipeline is still parked at the picker.
    # A pool of 1024 therefore built 1024 deck layers and their models before a box was ever
    # drawn, on the same page and the same GPU as the picker you are trying to draw on.
    #
    # So the scene starts with ONE blank layer and the update cell grows the list to the tile
    # count and trims it back. Idle cost is one draw call of nothing.
    def new_surface():
        return SurfaceLayer(
            positions=np.zeros((4, 3), dtype="float32"),
            triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype="uint32"),
            tex_coords=np.zeros((4, 2), dtype="float32"),
            texture=np.zeros((1, 1, 4), dtype="uint8"),
        )

    surfaces = [new_surface()]
    scene = Map(
        layers=surfaces,
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 11.5,
            "pitch": 60,
            "bearing": -25,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        controls=[
            FullscreenControl(position="top-right"),
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
        # DEPTH, IN LUMA v9 NAMES. `depthTest` is the WebGL-1 spelling and deck 9 hands a
        # layer's parameters to luma's render pipeline, which reads `depthCompare` and
        # `depthWriteEnabled`. The old key is not rejected, it is simply never read, so the
        # pipeline keeps its defaults: `depthCompare: "always"` (which luma maps to
        # gl.disable(DEPTH_TEST)) and no depth writes. That is a terrain with NO depth
        # buffer, resolved by submission order, which is right about half the time and
        # therefore reads as an artifact rather than as a broken render.
        parameters={
            "depthTest": True,
            "depthCompare": "less-equal",
            "depthWriteEnabled": True,
            "blend": True,
        },
    )
    scene
    return new_surface, scene, surfaces


@app.cell
def _(PALETTES, mo):
    # EVERY SCENE CONTROL, in one cell under the map it drives. None of them rebuild the
    # Map. Note what is NOT here any more: mesh density, texture size and drape tiles have
    # all moved up to `Detail` and `Tile texels`, because they were resolution controls
    # pretending to be rendering controls, and the box width was doing their job anyway.
    surface = mo.ui.dropdown(
        options=["NAIP RGB", "NDVI", "Elevation", "Relief"],
        value="NAIP RGB",
        label="Colour by",
    )
    palette = mo.ui.dropdown(options=list(PALETTES), value="BluYl", label="Ramp")
    # REVERSE IS A SEPARATE CONTROL FROM THE RAMP, as in every other notebook here, because
    # which end of a luminance ramp should be the high value depends on the surface and not
    # on the palette: bright peaks read as snow on Elevation, while on NDVI the dense canopy
    # is the end you want to stand out. It flips the normalised value, not the colour list,
    # so it costs nothing and the legend strip below flips with it.
    #
    # ON BY DEFAULT, which matches every other notebook in this repo, and on BluYl over
    # elevation it is the right way round: the bright end lands on the peaks.
    reverse_ramp = mo.ui.switch(value=True, label="Reverse ramp")
    # DEBOUNCED, ALL OF THEM, because each of these rebuilds every texture in the grid.
    # Without it a drag fires a rebuild per tick of travel and the notebook spends the whole
    # gesture painting intermediate values nobody asked to see; with it the gesture costs
    # one rebuild, at the value you let go on.
    brightness = mo.ui.slider(
        start=0.4, stop=3.0, step=0.1, value=1.0, label="Brightness", show_value=True,
        debounce=True,
    )
    ndvi_range = mo.ui.range_slider(
        start=-1.0, stop=1.0, step=0.05, value=[-0.2, 0.8],
        label="NDVI window", show_value=True, debounce=True,
    )
    hillshade = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6, label="Hillshade", show_value=True,
        debounce=True,
    )
    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=2.0, debounce=True, label="Elevation scale"
    )
    smooth = mo.ui.slider(
        start=0, stop=16, step=1, value=0, label="Height smooth", show_value=True,
        debounce=True,
    )
    # TEXELS PER QUAD, which is the geometry/texture balance as ONE number rather than two
    # that can drift apart. 1 puts a vertex on every texel; 4 is the default and is about
    # where a photograph still reads as a photograph on a mesh you cannot see. The
    # imbalance that produced the sawtooth silhouettes in the previous notebook was a ratio
    # of 16, arrived at by two controls that did not know about each other.
    texels_per_quad = mo.ui.dropdown(
        options={"1 (finest)": 1, "2": 2, "4": 4, "8": 8, "16": 16},
        value="4",
        label="Texels / quad",
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Opacity"
    )
    wireframe = mo.ui.switch(value=False, label="Wireframe")

    mo.vstack(
        [
            mo.hstack(
                [surface, palette, reverse_ramp, brightness, ndvi_range],
                justify="start", gap=2,
            ),
            mo.hstack(
                [elevation_scale, hillshade, smooth, texels_per_quad, fill_opacity,
                 wireframe],
                justify="start", gap=2,
            ),
        ],
        gap=0.75,
    )
    return (
        brightness,
        elevation_scale,
        fill_opacity,
        hillshade,
        ndvi_range,
        palette,
        reverse_ramp,
        smooth,
        surface,
        texels_per_quad,
        wireframe,
    )


@app.cell
def _(
    PALETTES,
    level,
    m_texel_ew,
    m_texel_ns,
    mo,
    n_tiles,
    ndvi_range,
    palette,
    reverse_ramp,
    tile_deg,
    view,
):
    # The legend paints the ramp it explains and says what the numbers mean, and it carries
    # the tile scheme because that is now the thing worth knowing about a render.
    _hex = PALETTES[palette.value].hex_colors
    if reverse_ramp.value:
        _hex = _hex[::-1]
    _strip = mo.Html(
        '<div style="height:14px;width:100%;border-radius:3px;'
        'border:1px solid rgba(128,128,128,0.35);'
        f'background:linear-gradient(to right,{",".join(_hex)});"></div>'
    )
    _scheme = mo.md(
        f"<small>**{n_tiles} tiles** at quadtree level {level} ({tile_deg:.6g}°), "
        f"{m_texel_ew:.2f} m/texel east-west and {m_texel_ns:.2f} m north-south. These "
        f"numbers do not change when you redraw the box wider; only the tile count "
        f"does.</small>"
    )
    if view == "NDVI":
        _body = mo.md(
            f"<small>**NDVI** = (NIR − Red) / (NIR + Red), from NAIP's fourth band, "
            f"averaged per H3 cell in SQL as a partial aggregate per tile and one merge "
            f"across them, which is exactly the global answer. Window "
            f"**{ndvi_range.value[0]:+.2f} to {ndvi_range.value[1]:+.2f}**, held fixed "
            f"rather than stretched to the scene so a colour means the same thing in "
            f"every AOI. Rough guide: below 0 water and snow, 0 to 0.2 rock, soil and "
            f"pavement, 0.2 to 0.4 grass and sage, 0.4 to 0.8 closed canopy. **Not a "
            f"red-green ramp**: that pair is exactly the one a deuteranope cannot "
            f"resolve.</small>"
        )
        _out = mo.vstack([_strip, _body, _scheme], gap=0.25)
    elif view == "NAIP RGB":
        _out = mo.vstack(
            [
                mo.md(
                    "<small>The photograph, read per tile at the tile's own texel size: "
                    "no hexagons, no hillshade, no ramp. Tiles with no imagery fall back "
                    "to the elevation ramp rather than rendering a hole.</small>"
                ),
                _scheme,
            ],
            gap=0.25,
        )
    else:
        _out = mo.vstack(
            [
                _strip,
                mo.md(
                    f"<small>**{view}**. *Relief* is max − min elevation inside "
                    f"an H3 cell, i.e. roughness at the resolution of the aggregation, "
                    f"which is a statistic about the pixels rather than a property of the "
                    f"surface they were folded into. *Elevation* is the bilinear height "
                    f"field itself, not a fold of it.</small>"
                ),
                _scheme,
            ],
            gap=0.25,
        )
    _out
    return


@app.cell
def _(
    fill_opacity,
    new_surface,
    positions,
    scene,
    surfaces,
    tex_coords,
    textures,
    triangles,
    wireframe,
):
    # THE LAYER LIST IS GROWN TO THE TILE COUNT HERE, not pre-allocated at Map construction.
    # A layer exists because a tile exists, which is the property that stops an unrun
    # notebook from holding a thousand deck layers open while you are using the picker.
    #
    # Reassigning `scene.layers` is safe: lonboard only recomputes the view state from the
    # layers in `add_layer(focus=True)` and `reset_zoom=True`, neither of which is on this
    # path, and `view_state` is an independent trait the frontend owns. It is still done as
    # rarely as possible, i.e. only when the COUNT changes, because the count changing is
    # the one case that cannot be expressed as a trait swap.
    _need = len(positions)
    while len(surfaces) < _need:
        surfaces.append(new_surface())

    # BATCHED PER LAYER, because positions, tex_coords and triangles have to agree about
    # vertex indices. Moving Texels/quad changes all three, and if they reach the frontend
    # one at a time the widget briefly holds indices past the end of a buffer.
    for _k in range(_need):
        _layer = surfaces[_k]
        with _layer.hold_trait_notifications():
            _layer.positions = positions[_k]
            _layer.tex_coords = tex_coords
            _layer.triangles = triangles
            _layer.texture = textures[_k]
            _layer.wireframe = wireframe.value
            _layer.opacity = fill_opacity.value

    # Surplus layers are REMOVED rather than blanked. The pooled version had to actively
    # blank them, because a layer left holding last run's geometry draws a coarse copy
    # underneath the new one; dropping them from the list is both cheaper and harder to get
    # wrong. `surfaces` keeps them so a later, larger tile set can reuse the objects.
    if len(scene.layers) != _need:
        scene.layers = tuple(surfaces[:_need])
    return


if __name__ == "__main__":
    app.run()
