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
"""1-meter 3DEP: pick a project mosaic for your AOI, stream it, fold it into H3.

The 10m sibling (xsql-dem-rem.py) reads the SEAMLESS DEM: one nationwide VRT, one
answer per AOI. The 1m product is not seamless. It is staged per lidar PROJECT under
StagedProducts/Elevation/1m/Projects/, projects overlap in space and time, and any AOI
can be covered by several of them at different vintages and quality levels. So this
notebook adds an explorer: query the TNM Access API for the AOI, group the returned
tiles into project mosaics, rank them by how much of the AOI each one actually covers
(complete vs partial), preview the chosen one as viridis bitmaps on the picker map, and
only then stream it into the H3 pipeline.

The other difference from the 10m notebook: 1m tiles are UTM (NAD83, per zone), not
lon/lat. The grid stays UTM into the SQL context and a per-tile `to_lonlat_<i>` UDF turns
metres into degrees inside the query. pyproj cannot run in a UDF at all (it aborts the
process from DataFusion's worker threads), so it runs once per tile on the main thread to
FIT lon/lat as order-3 polynomials, accurate to ~0.015 mm over a 10 km tile, and the UDF
applies those coefficients with pure numpy. See the UDF cell for the full autopsy.

Run:  uv run marimo edit xsql-dem-1m.py --sandbox
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import base64
    import io
    import json
    import urllib.parse
    import urllib.request

    import h3ronpy
    import numpy as np
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    import geoarrow.rust.core as grc
    from matplotlib.path import Path as MplPath
    from PIL import Image
    from pyproj import Transformer

    from arro3.core import Table
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import Map, H3HexagonLayer, BitmapLayer, PolygonLayer
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
        BitmapLayer,
        CartoBasemap,
        FullscreenControl,
        GeoTIFF,
        GeocoderControl,
        H3HexagonLayer,
        Image,
        Map,
        MaplibreBasemap,
        MplPath,
        NavigationControl,
        Photon,
        PolygonLayer,
        S3Store,
        ScaleControl,
        Table,
        Transformer,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        base64,
        coordinates_to_cells,
        grc,
        h3ronpy,
        io,
        json,
        mo,
        np,
        pa,
        udf,
        urllib,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 1m DEM to H3

    Draw a box (Ctrl/Cmd + drag). Unlike the 10m seamless DEM, the 1m product is staged
    **per lidar project**, so an AOI can be covered by several overlapping mosaics of
    different vintages. Pick one from the explorer below, preview it on the map, then it
    streams in and bins into H3 hexagons.

    Coverage is patchy: much of the country has no 1m collection at all, and where it
    exists the project boundaries are irregular. The explorer tells you which mosaics
    **completely** cover the AOI and which only clip a corner of it.
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, h3ronpy, np, pa, udf):
    # Same xarray-sql spine as the 10m notebook: XarrayContext IS a DataFusion session
    # with from_dataset(), so a raster's dims and data variables become columns and
    # `SELECT y, x, elevation FROM dem_0` unravels the grid. The H3 UDF returns a UBIGINT
    # cell id, exactly what H3HexagonLayer wants with high_precision=True.
    #
    # The 1m tiles are NAD83 UTM, one zone per tile, so something has to reproject before
    # H3 can bin anything. The obvious move is an st_transform UDF calling pyproj. That
    # CANNOT WORK, and the autopsy is worth keeping so nobody tries it again:
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
    # So: FIT the projection instead of calling it. Inverse transverse Mercator is smooth
    # over a 10 km tile, so lon and lat are each an order-3 polynomial in (x - cx, y - cy)
    # to well under a millimetre. pyproj runs ONCE PER TILE on the main thread to produce
    # 20 coefficients, and those get captured in a closure that DataFusion calls per batch.
    # The UDF is then pure numpy: no PROJ, no native per-thread state, safe on any worker
    # and parallel across all of them. Same reason the h3ronpy UDFs are fine here.
    #
    # h3_grid_disk gives each cell's k-ring as a LIST<UBIGINT>, so the flow calculation is
    # `unnest` + a self-join + avg() instead of a Python loop over a dict of every cell in
    # the scene. h3ronpy returns a LargeList, so that is the declared return type; getting
    # it wrong is a schema error at call time, not a cast.
    PROJ_ORDER = 3  # 1 ~ 4 m error over a tile, 2 ~ 3 mm, 3 ~ 0.015 mm. Measured, not guessed.

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

        fx, fy = np.meshgrid(
            np.linspace(left, right, samples), np.linspace(bottom, top, samples)
        )
        flon, flat = inv.transform(fx.ravel(), fy.ravel())
        A = _design(fx.ravel() - cx, fy.ravel() - cy)
        clon = np.linalg.lstsq(A, flon, rcond=None)[0]
        clat = np.linalg.lstsq(A, flat, rcond=None)[0]

        # Score on an INDEPENDENT denser grid and convert the angular residual to ground
        # metres. A silently bad fit would shift the whole scene, so make it fail loudly.
        tx, ty = np.meshgrid(
            np.linspace(left, right, check), np.linspace(bottom, top, check)
        )
        tlon, tlat = inv.transform(tx.ravel(), ty.ravel())
        B = _design(tx.ravel() - cx, ty.ravel() - cy)
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
        return (cx, cy, clon, clat), err_mm

    def make_lonlat_udf(name, fit):
        """One UDF per tile, its fitted coefficients closed over. Pure numpy inside."""
        cx, cy, clon, clat = fit

        def _to_lonlat(x, y):
            A = _design(x.to_numpy() - cx, y.to_numpy() - cy)
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
def _(mo):
    # Reactive AOI. Default: the Yazoo, Mississippi Delta. Flat alluvial floodplain cut by
    # meander scars and old channels, which is the case where the flow offset earns its
    # keep: relief is a couple of metres over the whole box, so elevation alone reads as a
    # flat sheet and the drainage only shows up once flow etches it in.
    get_bbox, set_bbox = mo.state((-90.56554, 32.80851, -90.317169, 33.162963))
    return get_bbox, set_bbox


@app.cell
def _(mo):
    # Own cell: the picker map must never reference a UI element (a re-run would rebuild
    # the map and drop the drawn AOI). H3 average edge length: 11~25m, 12~9.4m, 13~3.6m,
    # 14~1.35m, 15~0.51m. 1m source means res 14-15 is where you hit native detail, but
    # cell counts explode fast, so 13 is the default.
    h3_res = mo.ui.dropdown(
        options={
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9.4 m hex": 12,
            "res 13 ·  ~3.6 m hex": 13,
            "res 14 ·  ~1.35 m hex (near native)": 14,
            "res 15 ·  ~0.5 m hex (sub-native)": 15,
        },
        value="res 13 ·  ~3.6 m hex",
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
    set_bbox,
):
    # 2D picker. Draw a box (Ctrl/Cmd + drag) -> selected_bounds -> set_bbox. Built once,
    # never references a reactive UI element, so pan/zoom/AOI survive every downstream run.
    # The selected mosaic's bitmap preview is pushed onto .layers from a cell below, in
    # place, so previewing never rebuilds this map.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        layers=[],
        view_state={"longitude": -71.30, "latitude": 44.275, "zoom": 12, "pitch": 0},
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
    return (picker,)


@app.cell
def _(get_bbox, json, np, urllib):
    # THE CATALOG. The 10m notebook parses one nationwide VRT once; there is no such VRT
    # for 1m, and the nationwide footprint index (1m/FullExtentSpatialMetadata/FESM_1m.gpkg)
    # is 1.8 GB, far too heavy to pull for a single AOI. So the catalog query goes to the
    # TNM Access API instead: keyless, AOI-scoped, and every item already carries the
    # prd-tnm S3 URL of its COG plus a lon/lat footprint.
    #
    # Then the part that matters for 1m: GROUP the tiles by project. Each project is an
    # independent lidar collection with its own vintage, and mosaics overlap, so "which
    # tiles cover my AOI" has several competing answers. Coverage is measured on a 256x256
    # boolean grid over the AOI (tile footprints OR'd together), which is exact enough to
    # separate a mosaic that blankets the box from one that clips its corner.
    TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
    # obstore addresses prd-tnm by bucket key; this is the prefix that turns a key back
    # into the resolvable URL the cells below print for every COG they touch.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"

    def tnm_query(bbox, max_items=1000):
        q = urllib.parse.urlencode(
            {
                "datasets": "Digital Elevation Model (DEM) 1 meter",
                "bbox": ",".join(f"{v:.6f}" for v in bbox),
                "prodFormats": "GeoTIFF",
                "outputFormat": "JSON",
                "max": max_items,
            }
        )
        # The API returns an intermittent 500; retry a few times before giving up.
        last = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(f"{TNM}?{q}", timeout=60) as r:
                    payload = json.load(r)
                if "items" in payload:
                    return payload["items"]
                last = payload.get("message", "no items in response")
            except Exception as exc:  # noqa: BLE001 - surface it in the notebook instead
                last = repr(exc)
        raise RuntimeError(f"TNM Access API failed: {last}")

    def coverage_grid(bbox, footprints, n=256):
        # Fraction of the AOI covered by the union of these footprints.
        w, s, e, nn = bbox
        xs = np.linspace(w, e, n)
        ys = np.linspace(s, nn, n)
        hit = np.zeros((n, n), dtype=bool)
        for fw, fs, fe, fn in footprints:
            hit |= (
                ((ys[:, None] >= fs) & (ys[:, None] <= fn))
                & ((xs[None, :] >= fw) & (xs[None, :] <= fe))
            )
        return float(hit.mean())

    bbox = list(get_bbox())
    _items = tnm_query(bbox)

    _by_project = {}
    for _it in _items:
        _url = (_it.get("urls") or {}).get("TIFF") or _it.get("downloadURL") or ""
        if "/1m/Projects/" not in _url:
            continue
        _key = _url.split("amazonaws.com/", 1)[-1]
        _project = _url.split("/1m/Projects/", 1)[1].split("/", 1)[0]
        _bb = _it["boundingBox"]
        _tile = {
            "key": _key,
            "bbox": [_bb["minX"], _bb["minY"], _bb["maxX"], _bb["maxY"]],
            "title": _it.get("title", ""),
            "published": (_it.get("publicationDate") or "")[:10],
            "bytes": _it.get("sizeInBytes") or 0,
        }
        _by_project.setdefault(_project, []).append(_tile)

    mosaics = []
    for _project, _tiles in _by_project.items():
        _cov = coverage_grid(bbox, [t["bbox"] for t in _tiles])
        mosaics.append(
            {
                "project": _project,
                "tiles": _tiles,
                "coverage": _cov,
                "complete": _cov >= 0.999,
                # Project names carry the collection year; fall back to publication date.
                "published": max((t["published"] for t in _tiles), default=""),
            }
        )
    # Complete mosaics first, then by coverage, then newest: the order you actually want
    # to audition them in.
    mosaics.sort(key=lambda m: (-m["coverage"], m["published"]), reverse=False)
    mosaics.sort(key=lambda m: (m["complete"], m["coverage"], m["published"]), reverse=True)

    print(
        f"AOI {tuple(round(x, 4) for x in bbox)} -> {len(_items)} 1m tile(s) in "
        f"{len(mosaics)} project mosaic(s); "
        f"{sum(1 for m in mosaics if m['complete'])} cover the AOI completely"
    )
    return S3_BASE, bbox, mosaics


@app.cell
def _(mo):
    # How far to look past the AOI for collections. The question this map answers is not
    # only "what covers my box" but "what is NEXT to my box that could complete it", so the
    # footprint query runs on a padded envelope. Pad 0 shows only what touches the AOI.
    pad_deg = mo.ui.dropdown(
        options={
            "AOI only": 0.0,
            "+ 0.1°  (~10 km)": 0.1,
            "+ 0.5°  (~50 km)": 0.5,
            "+ 1.0°  (~110 km)": 1.0,
        },
        value="+ 0.5°  (~50 km)",
        label="Look around the AOI",
    )
    pad_deg
    return (pad_deg,)


@app.cell
def _(MplPath, bbox, json, np, pad_deg, urllib):
    # THE DISCOVERY LAYER: real 3DEP project boundaries, not inferred ones.
    #
    # The tile catalog above can only tell you about tiles that happen to intersect the AOI,
    # so a project's extent came out as the union of those tiles' bounding boxes: a blocky
    # over-estimate that stops at the edge of the box. That is what made a tile seam look
    # like a collection boundary. USGS publishes the actual polygons in the 3DEP Elevation
    # Index (layer 18, "1 Meter"), which is the same service the National Map downloader
    # draws, so ask it instead.
    #
    # Two service quirks, both learned the hard way:
    #   * returnGeometry=true without maxAllowableOffset 500s: the ungeneralised polygons
    #     are too big to serialise. Generalising to ~0.0005 deg (~50 m) is plenty for
    #     judging coverage and keeps the payload small.
    #   * spatialRel=esriSpatialRelContains returns nothing even for footprints that plainly
    #     swallow the AOI, so coverage is computed here instead of trusted to the server.
    INDEX = (
        "https://index.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevationIndex/MapServer/18/query"
    )

    def fetch_footprints(env, offset=0.0005):
        q = urllib.parse.urlencode(
            {
                "geometry": ",".join(f"{v:.6f}" for v in env),
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "project,pub_date",
                "returnGeometry": "true",
                "maxAllowableOffset": offset,
                "f": "geojson",
            }
        )
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(f"{INDEX}?{q}", timeout=90) as r:
                    return json.load(r).get("features", [])
            except Exception as exc:  # noqa: BLE001 - surface it in the notebook instead
                last = repr(exc)
        raise RuntimeError(f"3DEP index service failed: {last}")

    def _mpl_path(geom):
        # GeoJSON polygon -> one compound matplotlib Path. Ring 0 of each polygon is the
        # outer boundary and the rest are holes; Path's even-odd fill handles both, so data
        # gaps inside a collection do not get counted as coverage.
        polys = (
            geom["coordinates"]
            if geom["type"] == "MultiPolygon"
            else [geom["coordinates"]]
        )
        verts, codes = [], []
        for poly in polys:
            for ring in poly:
                pts = [(x, y) for x, y, *_ in ring]
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                verts += pts
                codes += (
                    [MplPath.MOVETO]
                    + [MplPath.LINETO] * (len(pts) - 2)
                    + [MplPath.CLOSEPOLY]
                )
        return MplPath(np.array(verts), codes)

    def aoi_coverage(box, geom, n=200):
        # Fraction of the AOI actually inside the polygon, sampled on an n x n grid.
        w, s, e, nn = box
        gx, gy = np.meshgrid(np.linspace(w, e, n), np.linspace(s, nn, n))
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return float(_mpl_path(geom).contains_points(pts).mean())

    _pad = pad_deg.value
    _env = [bbox[0] - _pad, bbox[1] - _pad, bbox[2] + _pad, bbox[3] + _pad]
    _feats = fetch_footprints(_env)

    collections = []
    for _f in _feats:
        _p = _f["properties"]
        _cov = aoi_coverage(bbox, _f["geometry"])
        collections.append(
            {
                "project": _p.get("project") or "(unnamed)",
                "pub_date": _p.get("pub_date") or "",
                "coverage": _cov,
                "complete": _cov > 0.999,
                "touches_aoi": _cov > 0.0,
                "geometry": _f["geometry"],
            }
        )
    # Complete first, then by how much of the AOI they cover. Nearby-but-not-touching
    # collections sort last: they are the seam candidates, not the answers.
    collections.sort(key=lambda c: (c["complete"], c["coverage"]), reverse=True)

    _n_complete = sum(1 for c in collections if c["complete"])
    _n_partial = sum(1 for c in collections if c["touches_aoi"] and not c["complete"])
    print(
        f"3DEP index: {len(collections)} 1m collection(s) within {_pad}° of the AOI · "
        f"{_n_complete} cover it completely · {_n_partial} cover part of it"
    )
    for _c in collections:
        _label = (
            "COMPLETE" if _c["complete"]
            else f"{_c['coverage'] * 100:5.1f}%" if _c["touches_aoi"]
            else "  nearby"
        )
        print(f"  {_label:>9}  {_c['project']}  ({_c['pub_date']})")
    return (collections,)


@app.cell
def _(collections, mo):
    # The answer table. "complete" means one collection covers the whole AOI on its own, so
    # no seaming needed. Several partials that sum past 100% is the other useful reading:
    # the AOI is coverable, but only by mosaicking collections of different vintages.
    mo.stop(
        not collections,
        mo.md("### No 1m lidar collections anywhere near this AOI"),
    )
    mo.ui.table(
        [
            {
                "collection": c["project"],
                "AOI coverage": (
                    "complete" if c["complete"]
                    else f"{c['coverage'] * 100:.1f}%" if c["touches_aoi"]
                    else "nearby only"
                ),
                "published": c["pub_date"],
            }
            for c in collections
        ],
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(PolygonLayer, Table, collections, grc, pa):
    # Footprints as map geometry. lonboard wants geoarrow, and there is no geopandas here,
    # so the GeoJSON rings go out as WKT and geoarrow-rust parses them back in: two small
    # string passes, no extra dependency.
    #
    # Colour is the Okabe-Ito qualitative set, which is built to stay distinguishable under
    # deuteranopia and protanopia. Ordered so the blue / orange / sky / pink / yellow block
    # gets used first and the green and vermillion sit at the end, where they are least
    # likely to land next to each other. Colour only names WHICH collection; the coverage
    # answer itself is in the table above and in the outline weight below, never in hue.
    OKABE_ITO = [
        (0, 114, 178),    # blue
        (230, 159, 0),    # orange
        (86, 180, 233),   # sky blue
        (204, 121, 167),  # reddish purple
        (240, 228, 66),   # yellow
        (0, 158, 115),    # bluish green
        (213, 94, 0),     # vermillion
    ]

    def _ring(coords):
        return "(" + ", ".join(f"{x} {y}" for x, y, *_ in coords) + ")"

    def _wkt(geom):
        t, c = geom["type"], geom["coordinates"]
        if t == "Polygon":
            return "POLYGON (" + ", ".join(_ring(r) for r in c) + ")"
        return "MULTIPOLYGON (" + ", ".join(
            "(" + ", ".join(_ring(r) for r in poly) + ")" for poly in c
        ) + ")"

    if collections:
        # Parse once to learn the inferred geoarrow type, then re-parse into that same type
        # tagged with EPSG:4326. Without the tag lonboard warns that it cannot verify the
        # CRS and has to assume WGS84; the service already returns lon/lat, so say so.
        _wkts = pa.array([_wkt(c["geometry"]) for c in collections])
        _geom = grc.from_wkt(_wkts, to_type=grc.from_wkt(_wkts).type.with_crs("EPSG:4326"))
        _rgb = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(collections))]
        # Collections that actually touch the AOI are drawn solid and heavier; ones that are
        # merely nearby stay faint, so "covers my box" reads without relying on colour.
        _fill = [
            [*rgb, 70 if c["touches_aoi"] else 25]
            for rgb, c in zip(_rgb, collections)
        ]
        _line = [[*rgb, 255] for rgb in _rgb]
        _width = [40.0 if c["touches_aoi"] else 15.0 for c in collections]

        _tbl = Table.from_arrow(
            pa.table(
                {
                    "collection": pa.array([c["project"] for c in collections]),
                    "published": pa.array([c["pub_date"] for c in collections]),
                    "AOI coverage": pa.array(
                        [
                            "complete" if c["complete"]
                            else f"{c['coverage'] * 100:.1f}%" if c["touches_aoi"]
                            else "nearby only"
                            for c in collections
                        ]
                    ),
                }
            )
        ).append_column("geometry", _geom)

        footprint_layer = PolygonLayer(
            table=_tbl,
            get_fill_color=pa.array(_fill, pa.list_(pa.uint8(), 4)),
            get_line_color=pa.array(_line, pa.list_(pa.uint8(), 4)),
            get_line_width=pa.array(_width, pa.float32()),
            line_width_units="meters",
            line_width_min_pixels=1.5,
            stroked=True,
            filled=True,
            pickable=True,
        )
    else:
        footprint_layer = None
    return (footprint_layer,)


@app.cell
def _(mo, mosaics):
    # The explorer table: every project mosaic that touches the AOI, what fraction of the
    # box it covers, how many tiles that costs, and how big the download would be at full
    # resolution (the notebook only ever pulls overviews, but the number tells you what
    # kind of collection you are looking at).
    mo.stop(
        not mosaics,
        mo.md(
            "### No 1m coverage here\n"
            "The TNM Access API returned no 1-metre DEM tiles for this box. 1m is "
            "project-based, not nationwide. Draw a box somewhere with a lidar "
            "collection, or use the 10m notebook (`xsql-dem-rem.py`) which tiles all of "
            "CONUS."
        ),
    )
    mo.ui.table(
        [
            {
                "mosaic": m["project"],
                "AOI coverage": f"{m['coverage'] * 100:.1f}%",
                "complete": "yes" if m["complete"] else "no",
                "tiles": len(m["tiles"]),
                "full-res size": f"{sum(t['bytes'] for t in m['tiles']) / 1e9:.2f} GB",
                "published": m["published"],
            }
            for m in mosaics
        ],
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(mo):
    # Own cell: marimo forbids reading a UI element's value in the cell that created it,
    # and the dropdown below has to read this one to filter its option list.
    complete_only = mo.ui.switch(value=False, label="Complete coverage only")
    return (complete_only,)


@app.cell
def _(complete_only, mo, mosaics):
    # Mosaic toggle. This cell depends on `mosaics`, so a new AOI rebuilds the dropdown
    # with that AOI's candidates (correct: the option list IS AOI-specific). "Complete
    # only" filters out mosaics that merely clip the box, which is the usual thing you
    # want when the AOI straddles a project boundary.
    mo.stop(not mosaics, mo.md("*No mosaic to pick: this AOI has no 1m coverage.*"))

    _pool = [m for m in mosaics if m["complete"]] if complete_only.value else mosaics
    _pool = _pool or mosaics
    _options = {
        f"{m['project']}  ·  {m['coverage'] * 100:.0f}% of AOI  ·  "
        f"{len(m['tiles'])} tile{'s' if len(m['tiles']) != 1 else ''}"
        f"{'  ·  complete' if m['complete'] else ''}": m
        for m in _pool
    }
    mosaic = mo.ui.dropdown(
        options=_options, value=next(iter(_options)), label="Project mosaic"
    )
    mo.hstack([mosaic, complete_only], justify="start", gap=2)
    return (mosaic,)


@app.cell
async def _(
    BitmapLayer,
    GeoTIFF,
    Image,
    S3Store,
    S3_BASE,
    Transformer,
    asyncio,
    base64,
    io,
    mosaic,
    np,
    palettable_preview,
):
    # PREVIEW: the selected mosaic, drawn on the picker map as bitmaps. Each tile's
    # COARSEST overview (312x312 for a 10k x 10k 1m tile, a few hundred KB) is read,
    # colour-mapped, and handed to a BitmapLayer as a data: URI. This is what makes the
    # mosaic toggle legible: switch projects and you see the footprint, the seams, and the
    # nodata holes before committing to a full stream.
    #
    # Bounds are the tile's four UTM corners reprojected to lon/lat, not the lon/lat
    # envelope, so the quad sits where the pixels actually are instead of being stretched
    # to an axis-aligned box.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _cmap = palettable_preview.mpl_colormap
    PREVIEW_TILE_CAP = 16

    _tiles = mosaic.value["tiles"][:PREVIEW_TILE_CAP]
    print(f"preview reading {len(_tiles)} COG(s):")
    for _t in _tiles:
        print(f"  {S3_BASE}{_t['key']}")

    async def _preview_tile(tile):
        g = await GeoTIFF.open(tile["key"], store=_store)
        ovr = g.overviews[-1] if g.overviews else g
        arr = await ovr.read()
        ma = arr.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)
        if not np.isfinite(elev).any():
            return None
        left, bottom, right, top = arr.bounds
        inv = Transformer.from_crs(g.crs, "EPSG:4326", always_xy=True)
        # deck.gl quad order: bottom-left, bottom-right, top-right, top-left.
        _x = [left, right, right, left]
        _y = [bottom, bottom, top, top]
        lons, lats = inv.transform(_x, _y)
        return elev, list(zip(lons, lats)), (float(np.nanmin(elev)), float(np.nanmax(elev)))

    _reads = [r for r in await asyncio.gather(*[_preview_tile(t) for t in _tiles]) if r]
    if _reads:
        # One shared stretch across the mosaic, so tiles read as a single surface rather
        # than each tile normalising to its own local range.
        _lo = min(r[2][0] for r in _reads)
        _hi = max(r[2][1] for r in _reads)

        def _bitmap(elev, quad):
            norm = np.clip((elev - _lo) / max(_hi - _lo, 1.0), 0.0, 1.0)
            rgba = (_cmap(np.nan_to_num(norm)) * 255).astype(np.uint8)
            rgba[~np.isfinite(elev), 3] = 0  # nodata -> transparent
            buf = io.BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return BitmapLayer(
                image=f"data:image/png;base64,{b64}", bounds=quad, opacity=0.85
            )

        preview_layers = [_bitmap(e, q) for e, q, _ in _reads]
        print(
            f"preview: {mosaic.value['project']} · {len(preview_layers)} tile(s) · "
            f"{_lo:.0f}-{_hi:.0f} m"
        )
    else:
        preview_layers = []
        print("preview: no readable pixels")
    return (preview_layers,)


@app.cell
def _():
    # Preview ramp kept out of the async cell so swapping it never re-reads the overviews.
    import palettable

    palettable_preview = palettable.matplotlib.Viridis_20
    return (palettable_preview,)


@app.cell
def _(footprint_layer, picker, preview_layers):
    # Push onto the running picker map IN PLACE. Assigning .layers is a trait update, not a
    # rebuild, so the drawn AOI and the view state survive every toggle.
    #
    # Pixels underneath, collection boundaries on top: the bitmaps show what the data looks
    # like, the outlines show where each collection stops. Reading them together is the
    # whole point, since a seam in the pixels is only a real edge if a boundary runs there.
    picker.layers = [*preview_layers, *( [footprint_layer] if footprint_layer else [] )]
    return


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    S3_BASE,
    Transformer,
    Window,
    asyncio,
    bbox,
    fit_lonlat,
    h3_res,
    make_h3_context,
    make_lonlat_udf,
    mo,
    mosaic,
    np,
    pa,
    xr,
):
    # Stream the selected mosaic and aggregate to H3 with xarray-sql. Per tile: pick an
    # overview whose ground sampling roughly matches the H3 cell, read ONLY the AOI window,
    # and hand the grid to the context AS IT LANDS, in UTM metres over dims y/x.
    #
    # The one real difference from the 10m notebook is the CRS change, and it happens in
    # the query rather than here. A 10m seamless COG is already lon/lat, so lat and lon are
    # its dimension coordinates and the SQL reads them straight. A 1m tile is NAD83 UTM, so
    # each tile gets its own fitted to_lonlat_<i> UDF and the query converts on the way
    # into H3. The only Python-side projection work is the single fit_lonlat call per tile
    # below, which is the one place pyproj is allowed to run.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value
    _w, _s, _e, _n = bbox
    _tiles = mosaic.value["tiles"]

    # Guard: refuse to build a scene the browser can't render. Estimate the cell count from
    # AOI area / H3 cell area BEFORE streaming (an upper bound: assumes full coverage) and
    # stop if it exceeds the cap. At 1m the fine resolutions bite quickly: res 15 over a
    # 1 km box is already ~1.1M cells.
    HEX_LIMIT = 5_000_000
    _cell_km2 = {
        11: 0.0021496,
        12: 0.00030712,
        13: 0.0000438710,
        14: 0.0000062673,
        15: 0.0000008953,
    }[_res]
    _latm = (_s + _n) / 2
    _area_km2 = (
        abs(_e - _w) * 111.32 * np.cos(np.radians(_latm)) * abs(_n - _s) * 111.32
    )
    _est = _area_km2 / _cell_km2
    mo.stop(
        _est > HEX_LIMIT,
        mo.md(
            f"### Too many hexagons\n"
            f"This AOI at this resolution is ~**{_est / 1e6:.1f}M** cells "
            f"(limit **{HEX_LIMIT / 1e6:.0f}M**). Lower the H3 resolution or draw a "
            f"smaller box."
        ),
    )

    # Target ground sampling ~ half the H3 edge. The CRS is metric, so unlike the 10m
    # notebook this is metres straight through, no degree conversion.
    _edge_m = {11: 25.0, 12: 9.4, 13: 3.6, 14: 1.35, 15: 0.51}[_res]
    _target_m = _edge_m / 2.0
    _PIXEL_BUDGET = 3_000_000  # per-tile window cap; step coarser if exceeded

    def _window(reader, aoi_proj):
        # AOI (already in this tile's CRS) clipped to the reader's extent, in pixel coords.
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
        # AOI corners into the tile's UTM zone, then take the envelope. All four corners,
        # not just SW/NE: the lon/lat box is a curved quad in UTM and the envelope of two
        # corners would clip the bulge.
        fwd = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True)
        _xs, _ys = fwd.transform([_w, _e, _e, _w], [_s, _s, _n, _n])
        aoi_proj = (min(_xs), min(_ys), max(_xs), max(_ys))

        cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
        fit = [r for r in cands if r.res[0] <= _target_m]
        start = cands.index(fit[-1]) if fit else 0
        # Walk from the matched overview toward coarser until the window fits the budget.
        for reader in (cands[start:] if fit else cands):
            win = _window(reader, aoi_proj)
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
        # Pixel-centre coords in UTM metres: y descends (north-up raster), x ascends. They
        # stay metres; the tile's to_lonlat_<i> UDF turns them into degrees in the query.
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
    print(f"streaming {len(_tiles)} COG(s) from {mosaic.value['project']}:")
    for _t in _tiles:
        print(f"  {S3_BASE}{_t['key']}")

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in _tiles]) if d]
    if _datasets:
        _px = sum(int(d["elevation"].size) for d, _, _, _ in _datasets)
        _zones = sorted({epsg for _, epsg, _, _ in _datasets})
        _worst = max(err for *_, err in _datasets)
        print(
            f"streamed {_px:,} pixels from {len(_datasets)}/{len(_tiles)} tile(s) of "
            f"{mosaic.value['project']} · EPSG {', '.join(str(z) for z in _zones)} · "
            f"lon/lat fit worst case {_worst:.4f} mm"
        )

        # ONE statement: unravel every tile's grid to (x, y, elevation) rows, drop NaN
        # nodata (elevation = elevation is false for NaN) BEFORE transforming so no pixel
        # is reprojected only to be discarded, turn UTM metres into degrees with the tile's
        # own fitted UDF, union the tiles, fold into H3.
        #
        # to_lonlat_<i> returns a struct, so it is bound once per tile as `p` in the inner
        # subquery and read as p.lon / p.lat in the outer one: one transform per pixel, not
        # one per ordinate. Each tile gets its OWN UDF carrying its OWN coefficients, so a
        # mosaic spanning two UTM zones reprojects each half out of the zone it came in.
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
        print("no 1m pixels for this AOI / mosaic")
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
    # so it depends on h3_table and resets to the full range on every new AOI or mosaic
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
