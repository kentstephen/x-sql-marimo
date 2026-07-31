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
#     "geopy==2.5.0",
#     "aiohttp>=3.10",
#     "arro3-core",
#     "numpy",
#     "pyproj>=3.7",
#     "pillow>=11",
#     "geoarrow-rust-core>=0.6",
# ]
# ///
"""S1M (Seamless 1-Metre) 3DEP: see the national coverage, draw a box, fold it into H3.

Three notebooks, three products. `xsql-dem-rem.py` reads the 10m seamless DEM, one
nationwide VRT and one answer per AOI. `xsql-dem-1m.py` reads the project-staged 1m
product, where an AOI can be covered by several overlapping lidar collections of different
vintages and picking between them is most of the work. This one reads S1M: USGS's
seamless 1-metre mosaic, already tiled to a single national grid, one tile per 10 km cell,
no project ambiguity to resolve.

That makes the catalog trivial: the whole product is one ~15 MB index, so the notebook
opens on a map of the ENTIRE S1M coverage and every AOI is answered from a local file with
no network round trip. There is no default AOI. Nothing loads until a box is drawn, and
then the tiles under that box stream into H3.

Catalog: `StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg`
(~15 MB, cached to `.cache/`). A GeoPackage is SQLite, so stdlib `sqlite3` reads it with
no geopandas and no duckdb: the `current` table carries every tile's Albers envelope in
the GeoPackage binary header, its elevation range, and the full S3 URL of its COG.

Tiles are NAD83(2011) Conus Albers (EPSG:6350), not lon/lat, so the grid stays in metres
into the SQL context and a per-tile `to_lonlat_<i>` UDF turns metres into degrees inside
the query. pyproj cannot run in a UDF at all (it aborts the process from DataFusion's
worker threads), so it runs once per tile on the main thread to FIT lon/lat as order-3
polynomials, measured here at ~0.03 mm over a 10 km tile, and the UDF applies those
coefficients with pure numpy. See the UDF cell for the full autopsy.

Run:  uv run marimo edit xsql-s1m-h3.py --sandbox
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import sqlite3
    import struct

    import h3ronpy
    import numpy as np
    import palettable
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    import geoarrow.rust.core as grc
    from pyproj import Transformer

    from arro3.core import Table
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import Map, H3HexagonLayer, SolidPolygonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard.colormap import apply_continuous_cmap
    from lonboard.controls import (
        FullscreenControl,
        GeocoderControl,
        NavigationControl,
        ScaleControl,
    )

    return (
        AioHTTPAdapter,
        CartoBasemap,
        FullscreenControl,
        GeoTIFF,
        GeocoderControl,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        NavigationControl,
        Photon,
        S3Store,
        ScaleControl,
        SolidPolygonLayer,
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
        np,
        pa,
        palettable,
        pathlib,
        sqlite3,
        struct,
        udf,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # S1M 1m seamless DEM to H3

    The map below is **the whole S1M product**: every published 10 km tile of the USGS
    seamless 1-metre DEM, shaded on viridis by its maximum elevation (dark low, bright
    high). Where there is no footprint, there is no S1M yet.

    Nothing loads until you **draw a box (Ctrl/Cmd + drag)** inside the coverage. The tiles
    under it fold into H3 hexagons below. One tile is 10000 x 10000 pixels at full
    resolution, so the read only ever pulls the overview that matches your H3 resolution.
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, h3ronpy, np, pa, udf):
    # Same xarray-sql spine as the other two notebooks: XarrayContext IS a DataFusion
    # session with from_dataset(), so a raster's dims and data variables become columns and
    # `SELECT y, x, elevation FROM dem_0` unravels the grid. The H3 UDF returns a UBIGINT
    # cell id, exactly what H3HexagonLayer wants with high_precision=True.
    #
    # S1M tiles are NAD83(2011) Conus Albers, so something has to reproject before H3 can
    # bin anything. The obvious move is an st_transform UDF calling pyproj. That CANNOT
    # WORK, and the autopsy is worth keeping so nobody tries it again:
    #
    #   DataFusion executes UDFs on Rust-spawned worker threads. pyproj's Transformer wraps
    #   a PROJ context backed by SQLite, and calling into it from those threads kills the
    #   process from C++ rather than raising: first "SQLite error on SELECT name FROM
    #   geodetic_datum: column index out of range", then a bus error inside from_crs once
    #   each thread built its own. Thread-local transformers did not fix it. Serialising
    #   construction did not fix it. A single GLOBAL LOCK around all pyproj work did not fix
    #   it, which is what rules out a data race: the threads themselves are the problem.
    #   target_partitions=1 did not fix it either. In marimo the symptom is the kernel
    #   dying with "failed to connect" and no traceback.
    #
    # So: FIT the projection instead of calling it. Albers is smooth over a 10 km tile, so
    # lon and lat are each an order-3 polynomial in (x - cx, y - cy) to a small fraction of
    # a millimetre. pyproj runs ONCE PER TILE on the main thread to produce 20 coefficients,
    # and those get captured in a closure that DataFusion calls per batch. The UDF is then
    # pure numpy: no PROJ, no native per-thread state, safe on any worker and parallel
    # across all of them. Same reason the h3ronpy UDFs are fine here.
    #
    # h3_grid_disk gives each cell's k-ring as a LIST<UBIGINT>, so the flow calculation is
    # `unnest` + a self-join + avg() instead of a Python loop over a dict of every cell in
    # the scene. h3ronpy returns a LargeList, so that is the declared return type; getting
    # it wrong is a schema error at call time, not a cast.
    PROJ_ORDER = 3  # order 1 ~ metres of error over a tile, 2 ~ mm, 3 ~ 0.03 mm. Measured.

    def _design(u, v, order=PROJ_ORDER):
        # Polynomial design matrix: 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3 ...
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
        # NORMALISE the window to [-1, 1] on each axis before fitting. Raw metres make the
        # design matrix ill-conditioned the moment the window is not roughly square, and an
        # AOI edge routinely clips a tile into a sliver (a real one: 4 m wide by 10 km
        # tall). There the u^3 column is ~1e-10 of the v^3 column, lstsq discards it as
        # noise, and the fit comes back metres wrong. Scaling keeps every column the same
        # order whatever the aspect ratio.
        sx = max((right - left) / 2.0, 1e-9)
        sy = max((top - bottom) / 2.0, 1e-9)

        fx, fy = np.meshgrid(
            np.linspace(left, right, samples), np.linspace(bottom, top, samples)
        )
        flon, flat = inv.transform(fx.ravel(), fy.ravel())
        A = _design((fx.ravel() - cx) / sx, (fy.ravel() - cy) / sy)
        clon = np.linalg.lstsq(A, flon, rcond=None)[0]
        clat = np.linalg.lstsq(A, flat, rcond=None)[0]

        # Score on an INDEPENDENT denser grid and convert the angular residual to ground
        # metres. A silently bad fit would shift the whole scene, so make it fail loudly.
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

    print(
        "xarray-sql context factory ready; UDFs: "
        "h3_latlng_to_cell(lat, lon, res) -> UBIGINT, "
        "h3_grid_disk(cell, k) -> LIST<UBIGINT>, "
        f"to_lonlat_<tile>(x, y) -> STRUCT<lon, lat> (order-{PROJ_ORDER} fit, per tile)"
    )
    return fit_lonlat, make_h3_context, make_lonlat_udf


@app.cell
def _(Transformer, np, pathlib, sqlite3, struct):
    # THE CATALOG, and it is one file. The 10m notebook parses a nationwide VRT; the 1m
    # project notebook has to interrogate the TNM Access API per AOI because the 1m
    # footprint index is 1.8 GB. S1M sits between them: its full-extent spatial metadata is
    # a ~15 MB GeoPackage covering the entire product, small enough to cache locally once
    # and then answer every AOI from memory with no network round trip at all.
    #
    # A GeoPackage is SQLite, so stdlib sqlite3 opens it: no geopandas, no duckdb, no GDAL.
    # Two feature tables, `current` and `historical`; `current` is the published mosaic.
    # Geometry does not even need a WKB parse. The GeoPackage binary header carries the
    # envelope when the flags byte says so (bit 1-3 = 1 means 4 doubles at offset 8), and
    # every S1M footprint IS its envelope: an axis-aligned 10 km Albers square.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"
    GPKG_KEY = "StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg"
    CACHE = pathlib.Path(".cache")

    def fetch_index(refresh=False):
        """Cache the national index locally. One download, then it is a local file."""
        CACHE.mkdir(exist_ok=True)
        path = CACHE / "S1M_Products.gpkg"
        if refresh or not path.exists():
            import urllib.request

            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(S3_BASE + GPKG_KEY, timeout=300) as r, tmp.open(
                "wb"
            ) as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
        return path

    def _envelope(blob):
        # GeoPackage binary: 'GP', version, flags, srs_id, [envelope], WKB.
        flags = blob[3]
        if (flags >> 1) & 0x07 == 0:
            raise ValueError("S1M footprint has no envelope in its GPB header")
        little = bool(flags & 0x01)
        xmin, xmax, ymin, ymax = struct.unpack_from("<4d" if little else ">4d", blob, 8)
        return xmin, ymin, xmax, ymax

    _path = fetch_index()
    with sqlite3.connect(f"file:{_path}?mode=ro", uri=True) as _con:
        _rows = _con.execute(
            """
            SELECT geom, tile, production_date, pub_date, z_min, z_max,
                   horiz_crs_epsg, data_source_count, dataset_link
            FROM current
            """
        ).fetchall()

    # Reproject every footprint's corners ONCE, here on the main thread, vectorised over the
    # whole product. Albers corners, not the lon/lat envelope: a 10 km Albers square is a
    # slightly rotated quad in degrees, and drawing it as an axis-aligned box would smear
    # the national grid into a staircase.
    _alb = np.array([_envelope(r[0]) for r in _rows], dtype="float64")
    _inv = Transformer.from_crs("EPSG:6350", "EPSG:4326", always_xy=True)
    # Corner order: SW, SE, NE, NW.
    _cx = np.column_stack([_alb[:, 0], _alb[:, 2], _alb[:, 2], _alb[:, 0]])
    _cy = np.column_stack([_alb[:, 1], _alb[:, 1], _alb[:, 3], _alb[:, 3]])
    _lon, _lat = _inv.transform(_cx.ravel(), _cy.ravel())
    _lon = _lon.reshape(-1, 4)
    _lat = _lat.reshape(-1, 4)

    # z_min carries the tile's nodata sentinel (-999999) wherever a tile has holes, so the
    # shading below reads z_max only and this keeps the raw value for the table.
    tiles_all = [
        {
            "tile": r[1],
            "key": r[8].split("amazonaws.com/", 1)[-1],
            "url": r[8],
            "produced": r[2] or "",
            "published": r[3] or "",
            "z_min": float(r[4]) if r[4] is not None else float("nan"),
            "z_max": float(r[5]) if r[5] is not None else float("nan"),
            "epsg": int(r[6]) if r[6] else 6350,
            "sources": r[7],
            "albers": tuple(_alb[i]),  # left, bottom, right, top in EPSG:6350 metres
            "quad": list(zip(_lon[i], _lat[i])),  # SW, SE, NE, NW in lon/lat
            "lonlat": (
                float(_lon[i].min()),
                float(_lat[i].min()),
                float(_lon[i].max()),
                float(_lat[i].max()),
            ),
        }
        for i, r in enumerate(_rows)
    ]
    tiles_albers = _alb

    print(
        f"S1M index: {len(tiles_all):,} current tiles from {_path} "
        f"({_path.stat().st_size / 1e6:.1f} MB) · "
        f"published {min(t['published'] for t in tiles_all)} to "
        f"{max(t['published'] for t in tiles_all)}"
    )
    return S3_BASE, tiles_albers, tiles_all


@app.cell
def _(
    SolidPolygonLayer,
    Table,
    apply_continuous_cmap,
    grc,
    np,
    pa,
    palettable,
    tiles_all,
):
    # THE COVERAGE LAYER: the entire product as geometry, drawn before anything is picked.
    # This is the thing the notebook opens on, and it answers the only question that matters
    # before you draw a box: where does S1M exist at all.
    #
    # Shaded by z_max on viridis at low opacity and with NO outlines, exactly the way
    # s1m_viewer.py draws it: neighbouring tiles blend into one continuous field, so the
    # carpet reads as a single dissolved coverage shape with elevation context rather than
    # as 11,717 separately coloured boxes. Outlining each tile is what made it read as a
    # grid. The coverage answer is the PRESENCE of the shape, never its hue.
    _wkts = pa.array(
        [
            "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in [*t["quad"], t["quad"][0]]) + "))"
            for t in tiles_all
        ]
    )
    _geom = grc.from_wkt(_wkts, to_type=grc.from_wkt(_wkts).type.with_crs("EPSG:4326"))

    _zmax = np.array([t["z_max"] for t in tiles_all], dtype="float64")
    _zmax = np.where(np.isfinite(_zmax), _zmax, 0.0)
    # Clip to a robust range so a handful of high peaks do not flatten the whole ramp.
    _lo, _hi = float(np.percentile(_zmax, 1)), float(np.percentile(_zmax, 99))
    _norm = np.clip((_zmax - _lo) / max(_hi - _lo, 1.0), 0.0, 1.0)
    _fill = apply_continuous_cmap(_norm, palettable.matplotlib.Viridis_20, alpha=150)

    coverage_table = Table.from_arrow(
        pa.table(
            {
                "tile": pa.array([t["tile"] for t in tiles_all]),
                "produced": pa.array([t["produced"] for t in tiles_all]),
                "max elevation (m)": pa.array(_zmax),
            }
        )
    ).append_column("geometry", _geom)

    coverage_layer = SolidPolygonLayer(
        table=coverage_table,
        get_fill_color=_fill,
        opacity=0.35,
        extruded=False,
        pickable=False,
    )
    print(
        f"coverage layer: {len(tiles_all):,} S1M footprints · "
        f"viridis over tile z_max {_lo:.0f} m (dark) to {_hi:.0f} m (bright)"
    )
    return (coverage_layer,)


@app.cell
def _(mo):
    # NO DEFAULT AOI. The other notebooks seed a box and stream it on open; at 1m that is
    # the wrong reflex, so this starts at None and every stage below halts until a box has
    # been drawn by hand.
    get_bbox, set_bbox = mo.state(None)
    return get_bbox, set_bbox


@app.cell
def _(mo):
    # Own cell: the picker map must never reference a UI element (a re-run would rebuild the
    # map and drop the drawn AOI). H3 average edge length: 11~25m, 12~9.4m, 13~3.6m,
    # 14~1.35m, 15~0.51m. 1m source means res 14-15 is where you hit native detail, but cell
    # counts explode fast, so 12 is the default.
    h3_res = mo.ui.dropdown(
        options={
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9.4 m hex": 12,
            "res 13 ·  ~3.6 m hex": 13,
            "res 14 ·  ~1.35 m hex (near native)": 14,
            "res 15 ·  ~0.5 m hex (sub-native)": 15,
        },
        value="res 12 ·  ~9.4 m hex",
        label="H3 resolution",
    )
    h3_res
    return (h3_res,)


@app.cell
def _(
    AioHTTPAdapter,
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
):
    # The picker, opened on CONUS with the coverage carpet already on it. Draw a box
    # (Ctrl/Cmd + drag) -> selected_bounds -> set_bbox.
    #
    # Built once and it references no reactive UI element, so pan/zoom/AOI survive every
    # downstream run. coverage_layer is static (it depends only on the cached index), so
    # passing it at construction is safe and nothing ever reassigns .layers: the map shows
    # footprints and your box, nothing else.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        layers=[coverage_layer],
        view_state={"longitude": -96.0, "latitude": 38.5, "zoom": 3.6, "pitch": 0},
        basemap=MaplibreBasemap(style=CartoBasemap.Positron),
        controls=[
            _geocoder,
            FullscreenControl(position="top-right"),
            # visualize_pitch makes the compass button call resetNorthPitch(): one click
            # snaps back to north-up AND flat (pitch 0), not just north-up.
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
    )
    picker.observe(
        lambda c: set_bbox(c["new"]) if c["new"] is not None else None,
        names="selected_bounds",
    )
    picker
    return


@app.cell
def _(Transformer, get_bbox, h3_res, mo, np, tiles_albers, tiles_all):
    # THE HALT. Nothing below runs until an AOI exists.
    _raw = get_bbox()
    mo.stop(
        _raw is None,
        mo.md(
            "### Draw an AOI to continue\n"
            "Hold **Ctrl/Cmd and drag** on the coverage map above. Nothing is fetched "
            "until there is a box."
        ),
    )
    bbox = list(_raw)

    # Candidate resolution is a local array intersection, not an API call: the AOI corners
    # go into Albers once and get tested against the cached envelopes. All four corners,
    # not just SW/NE, because a lon/lat box is a curved quad in Albers and the envelope of
    # two corners would clip the bulge.
    _fwd = Transformer.from_crs("EPSG:4326", "EPSG:6350", always_xy=True)
    _ax, _ay = _fwd.transform(
        [bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]]
    )
    aoi_albers = (min(_ax), min(_ay), max(_ax), max(_ay))
    _w, _s, _e, _n = aoi_albers

    _hit = (
        (tiles_albers[:, 0] < _e)
        & (tiles_albers[:, 2] > _w)
        & (tiles_albers[:, 1] < _n)
        & (tiles_albers[:, 3] > _s)
    )

    # Size each tile's read before it happens, and print it. The read is a window on ONE
    # overview, picked the same way the streaming cell picks it: the coarsest level whose
    # ground sampling still sits at or under half the H3 edge. S1M tiles are uniform
    # (10000 x 10000 at 1 m with the standard power-of-two overview ladder), so the pixel
    # count in the log is what actually gets pulled.
    _edge_m = {11: 25.0, 12: 9.4, 13: 3.6, 14: 1.35, 15: 0.51}[h3_res.value]
    _target_m = _edge_m / 2.0
    OVERVIEW_RES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    _fit = [r for r in OVERVIEW_RES if r <= _target_m]
    read_res_m = _fit[-1] if _fit else OVERVIEW_RES[0]

    candidates = []
    for _i in np.flatnonzero(_hit):
        _t = tiles_all[int(_i)]
        _tl, _tb, _tr, _tt = _t["albers"]
        _ow = min(_e, _tr) - max(_w, _tl)
        _oh = min(_n, _tt) - max(_s, _tb)
        _px = int((_ow / read_res_m) * (_oh / read_res_m))
        candidates.append(
            {
                **_t,
                "overlap_m2": _ow * _oh,
                "window_px": _px,
                "aoi_share": (_ow * _oh) / max((_e - _w) * (_n - _s), 1.0),
            }
        )
    # Biggest slice of the AOI first, so the log reads with the main tile at the top and
    # the ragged edge tiles under it.
    candidates.sort(key=lambda c: -c["overlap_m2"])

    print(
        f"AOI {tuple(round(v, 4) for v in bbox)} -> {len(candidates)} S1M tile(s) · "
        f"reading the {read_res_m:g} m overview for H3 res {h3_res.value}"
    )
    for _c in candidates:
        print(
            f"  {_c['tile']}  {_c['aoi_share'] * 100:5.1f}% of AOI  "
            f"{_c['window_px']:>12,} px  {_c['produced']}"
        )
    return aoi_albers, bbox, candidates, read_res_m


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    S3_BASE,
    Window,
    aoi_albers,
    asyncio,
    bbox,
    candidates,
    fit_lonlat,
    h3_res,
    make_h3_context,
    make_lonlat_udf,
    mo,
    np,
    pa,
    read_res_m,
    xr,
):
    # THE READ, for every tile the box touches.
    #
    # Per tile: pick the overview whose ground sampling matches the H3 cell, read ONLY the
    # AOI window out of it, and hand the grid to the SQL context AS IT LANDS, in Albers
    # metres over dims y/x. The CRS change happens in the QUERY, not here: each tile gets
    # its own fitted to_lonlat_<i> UDF and the SQL converts on the way into H3. The only
    # Python-side projection work is the single fit_lonlat call per tile below, which is the
    # one place pyproj is allowed to run.
    #
    # Guard: refuse to build a scene the browser cannot render. Estimate the cell count from
    # AOI area / H3 cell area BEFORE streaming (an upper bound: assumes full coverage) and
    # stop if it exceeds the cap. At 1m the fine resolutions bite quickly.
    HEX_LIMIT = 5_000_000
    _cell_km2 = {
        11: 0.0021496,
        12: 0.00030712,
        13: 0.0000438710,
        14: 0.0000062673,
        15: 0.0000008953,
    }[h3_res.value]
    _latm = (bbox[1] + bbox[3]) / 2
    _area_km2 = (
        abs(bbox[2] - bbox[0])
        * 111.32
        * np.cos(np.radians(_latm))
        * abs(bbox[3] - bbox[1])
        * 111.32
    )
    _est = _area_km2 / _cell_km2
    mo.stop(
        _est > HEX_LIMIT,
        mo.md(
            f"### Too many hexagons\n"
            f"This AOI at res {h3_res.value} is ~**{_est / 1e6:.1f}M** cells "
            f"(limit **{HEX_LIMIT / 1e6:.0f}M**). Lower the H3 resolution or draw a "
            f"smaller box."
        ),
    )

    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value
    _PIXEL_BUDGET = 3_000_000  # per-tile window cap; step coarser if exceeded

    def _window(reader, aoi_proj):
        # AOI (already in Albers) clipped to the reader's extent, in pixel coords.
        pw, ps, pe, pn = aoi_proj
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw = max(pw, bw); ce = min(pe, be)
        cn = min(pn, bn); cs = max(ps, bs)
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
        fit_lvls = [r for r in cands if r.res[0] <= read_res_m]
        start = cands.index(fit_lvls[-1]) if fit_lvls else 0
        # Walk from the matched overview toward coarser until the window fits the budget.
        for reader in (cands[start:] if fit_lvls else cands):
            win = _window(reader, aoi_albers)
            if win is None:
                return None
            if win.width * win.height <= _PIXEL_BUDGET or reader is cands[-1]:
                break
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)  # nodata -> NaN
        if not np.isfinite(elev).any():
            return None

        left, bottom, right, top = r.bounds
        h, w = elev.shape
        # Pixel-centre coords in Albers metres: y descends (north-up raster), x ascends.
        # They stay metres; the tile's to_lonlat_<i> UDF turns them into degrees in the SQL.
        y = top - (np.arange(h) + 0.5) * (top - bottom) / h
        x = left + (np.arange(w) + 0.5) * (right - left) / w
        ds = xr.Dataset({"elevation": (("y", "x"), elev)}, coords={"y": y, "x": x})

        # The one pyproj call for this tile, here on the main thread: fit lon/lat over the
        # window actually read (not the full tile, so the fit is if anything easier).
        fit, err_mm = fit_lonlat(g.crs, (left, bottom, right, top))
        return ds, g.crs.to_epsg(), fit, err_mm

    # Exactly which objects this scene reads, spelled out in full and BEFORE the reads, so
    # the list is there to compare against even if a fetch fails. obstore addresses the
    # bucket by key; this is the resolvable URL you can paste into a browser or gdalinfo.
    print(f"streaming {len(candidates)} S1M COG(s):")
    for _t in candidates:
        print(f"  {S3_BASE}{_t['key']}")

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in candidates]) if d]
    if _datasets:
        _px = sum(int(d["elevation"].size) for d, _, _, _ in _datasets)
        _worst = max(err for *_, err in _datasets)
        print(
            f"streamed {_px:,} pixels from {len(_datasets)}/{len(candidates)} tile(s) · "
            f"EPSG 6350 · lon/lat fit worst case {_worst:.4f} mm"
        )

        # ONE statement: unravel every tile's grid to (x, y, elevation) rows, drop NaN
        # nodata (elevation = elevation is false for NaN) BEFORE transforming so no pixel
        # is reprojected only to be discarded, turn Albers metres into degrees with the
        # tile's own fitted UDF, union the tiles, fold into H3.
        #
        # to_lonlat_<i> returns a struct, so it is bound once per tile as `p` in the inner
        # subquery and read as p.lon / p.lat in the outer one: one transform per pixel, not
        # one per ordinate.
        ctx = make_h3_context()
        for _i, (_d, _, _fit, _) in enumerate(_datasets):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"y": 1024})
            ctx.register_udf(make_lonlat_udf(f"to_lonlat_{_i}", _fit))
        _union = " UNION ALL ".join(
            f"SELECT p.lat AS lat, p.lon AS lon, elevation FROM ("
            f"  SELECT to_lonlat_{_i}(x, y) AS p, elevation"
            f"  FROM dem_{_i} WHERE elevation = elevation"
            f")"
            for _i in range(len(_datasets))
        )

        # Re-base each cell to the scene: subtract the AOI minimum so the lowest cell sits
        # at 0 and height reads RELATIVE to what's in view, not as height above sea level.
        _scene = ctx.sql(
            f"""
            SELECT hex, elevation - MIN(elevation) OVER () AS elevation
            FROM (
                SELECT h3_latlng_to_cell(lat, lon, CAST({_res} AS INT)) AS hex,
                       avg(elevation) AS elevation
                FROM ({_union})
                GROUP BY 1
            )
            """
        ).to_arrow_table()

        # flow = how far each hex sits below the ground around it: average the cell's k-ring
        # and subtract the cell, so hollows come out positive (bright) and ridges negative
        # (dark). At 1m this picks out ditches and road cuts.
        #
        # SQL, not a Python loop: h3_grid_disk gives each cell its ring as a list, unnest
        # explodes that to one row per neighbour, and a self-join back onto the scene brings
        # each neighbour's elevation in to be averaged. The join is also what handles the
        # scene edge: neighbours that fall outside the AOI simply do not match, so border
        # cells average over the ring they actually have.
        #
        # Two statements rather than one CTE chain on purpose. DataFusion does not
        # materialise CTEs, so referencing the scene twice in a single query would re-run
        # the entire stream-transform-fold pipeline for each reference. Landing it as an
        # arrow table and re-registering it costs one pass and keeps the second query cheap.
        ctx.from_arrow(_scene, name="scene")
        h3_table = ctx.sql(
            """
            WITH ring AS (
                SELECT hex, elevation,
                       unnest(h3_grid_disk(hex, CAST(1 AS INT))) AS nb
                FROM scene
            )
            SELECT r.hex AS hex,
                   r.elevation AS elevation,
                   avg(n.elevation) - r.elevation AS flow
            FROM ring r
            JOIN scene n ON r.nb = n.hex
            GROUP BY r.hex, r.elevation
            """
        ).to_arrow_table()
        print(f"H3 res {_res}: {h3_table.num_rows:,} cells")
    else:
        h3_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
                "flow": pa.array([], pa.float64()),
            }
        )
        print("no S1M pixels for this AOI")
    return (h3_table,)


@app.cell
def _(h3_table):
    # Quick peek: scene-relative elevation vs flow (below-neighbors depth) per hex.
    h3_table.select(["elevation", "flow"]).slice(0, 15)
    return


@app.cell
def _():
    # Palette registry: matplotlib + CARTOColors sequential ramps. All are luminance-
    # monotonic (deuteranope-safe: no red/green opposition). The dropdown at the bottom
    # picks one.
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
def _(
    PALETTES,
    apply_continuous_cmap,
    contrast,
    flow_gain,
    h3_table,
    np,
    palette,
):
    # COLOR CELL: separate from the ETL on purpose. Base is scene-relative ELEVATION; flow
    # is added as an OFFSET (flow_gain * flow) so drainage etches into the elevation shading
    # without losing the overall terrain read. Gain 0 = pure elevation. Depends on h3_table
    # + palette + gain + contrast, so it re-runs on those but never re-streams / re-folds.
    _cmap = PALETTES[palette.value]
    _elev = (
        np.asarray(h3_table["elevation"]).astype("float64")
        + flow_gain.value * np.asarray(h3_table["flow"]).astype("float64")
    )
    if _elev.size:
        _lo, _hi = float(contrast.value[0]), float(contrast.value[1])
        _norm = np.clip((_elev - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
        colors_fwd = apply_continuous_cmap(_norm, _cmap, alpha=1.0)
        colors_rev = apply_continuous_cmap(1.0 - _norm, _cmap, alpha=1.0)
    else:
        colors_fwd = np.zeros((0, 4), dtype="uint8")
        colors_rev = np.zeros((0, 4), dtype="uint8")
    return colors_fwd, colors_rev


@app.cell
def _(
    CartoBasemap,
    FullscreenControl,
    H3HexagonLayer,
    Map,
    MaplibreBasemap,
    NavigationControl,
    ScaleControl,
    Table,
    bbox,
    h3_table,
):
    # The output scene: extruded H3 hexagons. Geometry (hex) and height (scene-relative
    # elevation) come straight from h3_table as arrow columns.
    #
    # This cell references NEITHER the colors NOR the palette NOR any control, so marimo
    # never re-runs it for a control change and the Map is never rebuilt (which would lose
    # view state). The layer is built ONCE with a placeholder fill and the update cell at
    # the bottom paints it live via a get_fill_color trait swap.
    scene_table = Table.from_arrow(h3_table)
    h3_layer = H3HexagonLayer(
        table=scene_table,
        get_hexagon=scene_table["hex"],
        get_fill_color=[136, 136, 136],  # placeholder; update cell paints it live below
        get_elevation=scene_table["elevation"],
        high_precision=True,
        extruded=True,
        stroked=False,
        elevation_scale=3.0,  # initial; the number input below nudges this live
        opacity=0.9,          # initial; the number input below nudges this live
    )

    scene = Map(
        layers=[h3_layer],
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 13,
            "pitch": 55,
            "bearing": -20,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter),
        controls=[
            FullscreenControl(position="top-right"),
            # Compass click -> north-up and flat (resetNorthPitch), the way out of a
            # tilted 3D view without dragging the pitch back by hand.
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
        parameters={"depthTest": True, "blend": True},
    )
    print(f"scene: {h3_table.num_rows:,} hexes")
    scene
    return (h3_layer,)


@app.cell
def _(h3_table, mo, np):
    # Contrast window for the color domain. Its bounds ARE this scene's elevation min..max,
    # so it depends on h3_table and resets to the full range on every new AOI or selection
    # (right behavior: bounds change per scene).
    _elev = np.asarray(h3_table["elevation"]).astype("float64")
    if _elev.size:
        _clo, _chi = float(np.floor(_elev.min())), float(np.ceil(_elev.max()))
    else:
        _clo, _chi = 0.0, 1.0
    if _chi <= _clo:
        _chi = _clo + 1.0
    contrast = mo.ui.range_slider(
        start=_clo,
        stop=_chi,
        value=[_clo, _chi],
        step=max((_chi - _clo) / 200.0, 0.1),
        label="Elevation contrast (m)",
        show_value=True,
        full_width=True,
        debounce=True,  # recolor on release, not every drag tick
    )
    contrast
    return (contrast,)


@app.cell
def _(PALETTES, mo):
    # Right below the map: palette picker + float inputs (0.1 steppers) + toggles. None of
    # them touch the stream, the SQL, or rebuild the map, so the scene updates in place.
    palette = mo.ui.dropdown(
        options=list(PALETTES), value="Emrld", label="Palette"
    )
    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=3.0, debounce=True, label="Elevation scale"
    )
    flow_gain = mo.ui.number(
        start=0.0, stop=50.0, step=0.5, value=8.0, debounce=True, label="Flow offset"
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=0.9, debounce=True, label="Opacity"
    )

    reverse_ramp = mo.ui.switch(value=True, label="Reverse ramp")
    extruded = mo.ui.switch(value=True, label="Extruded")
    mo.hstack(
        [palette, elevation_scale, flow_gain, fill_opacity, reverse_ramp, extruded],
        justify="start", gap=2,
    )
    return (
        elevation_scale,
        extruded,
        fill_opacity,
        flow_gain,
        palette,
        reverse_ramp,
    )


@app.cell
def _(
    colors_fwd,
    colors_rev,
    elevation_scale,
    extruded,
    fill_opacity,
    h3_layer,
    reverse_ramp,
):
    # The only thing the controls do: nudge live traits on the running layer. No Map
    # rebuild, no re-stream, no re-fold, no re-color.
    h3_layer.elevation_scale = elevation_scale.value
    h3_layer.opacity = fill_opacity.value
    h3_layer.get_fill_color = colors_rev if reverse_ramp.value else colors_fwd
    h3_layer.extruded = extruded.value
    return


if __name__ == "__main__":
    app.run()
