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
#     "arro3-io",
#     "numpy",
#     "pyproj>=3.7",
#     "geopy==2.5.0",
#     "aiohttp>=3.10",
#     "geoarrow-rust-core>=0.6",
# ]
# ///
"""S1M -> H3 -> ONE TEXTURED MESH. A lighter way to look at the same fold.

`H3HexagonLayer` costs a full extruded prism per cell, so a scene worth looking at is a scene
that is slow to fly, and at a few hundred thousand cells it will hang a machine. Two escapes
were measured and both died: band compaction (data-dependent, lossy, and 1 m lidar noise sets
the floor) and dissolve-to-polygons (pays only where neighbours are IDENTICAL, and flow_gain
is a deliberately high-frequency per-cell signal, so 109k cells dissolved to 74k regions).

This one stops sending the hexagons to the GPU at all. There is NO hexagon layer in this
notebook, deliberately.

  * GEOMETRY is one regular triangle mesh over the AOI. Its cost is the mesh-density slider
    and nothing else. 200k cells and 2M cells cost exactly the same to draw.
  * STYLING is one texture. Every texel is resolved through `coordinates_to_cells` to the
    cell that contains it and painted from the same elevation + flow_gain composite that
    `get_fill_color` used to receive.

The H3 fold itself is untouched: same catalog, same streaming, same SQL, same ring join. H3
is still how the data is BINNED. It is just no longer how the data is DRAWN.

MAKING IT LOOK LIKE TERRAIN. A textured mesh does not look good by default, and the reasons
are specific and fixable:

  * NO LIGHTING. lonboard's SurfaceLayer ships exactly two mesh attributes, POSITION and
    TEXCOORD_0. There is no NORMAL in the bundle, so deck's lighting has nothing to work with
    and the surface renders effectively unlit. Extruded prisms looked better partly because
    their vertical walls catch light for free. Adding normals would mean patching lonboard's
    JS, so the light is instead BAKED INTO THE TEXTURE: a real hillshade computed in numpy,
    sun at 315/45, multiplied in as pure luminance so hue never shifts and the palette stays
    deuteranope-safe. That is the `hillshade` control and it is the single biggest difference.
  * ANGULARITY, cause one: the mesh was coarser than the data. At density 256 over a 7 km AOI
    each quad is ~28 m while a res-12 hex is 9.4 m, so it sampled one vertex per nine cells
    and spanned the gaps with big flat triangles. The density slider now goes to 2048.
  * ANGULARITY, cause two, and no mesh density fixes it: the height field off the fold is
    PIECEWISE CONSTANT. Every hexagon is a flat plateau with a vertical step to its
    neighbour, so a dense mesh gives literal hexagonal stairs and a coarse one gives
    arbitrary facets. The staircase is in the DATA, not the tessellation. `relief_smooth`
    blurs the height field itself, which is what turns plateaus into terrain.

`colour_smooth` is the separate one: it blurs the shading VALUE before colouring, so the ramp
softens without flattening the relief. Both blurs are NaN-aware (normalised convolution), or
they would bleed zeros in from outside the AOI and draw a dark rind around every edge.

CONTROLS all live in one block directly under the scene. None of them rebuild the Map: the
layer is built once from placeholder geometry and traits are swapped on it, so the view you
flew to survives every adjustment.

PICKER. Opens on the national S1M coverage carpet over the National Map 3D Viewer's own
basemap set, Esri Topographic by default (which is what the viewer actually opens on, and why
its terrain is shaded across Canada and Mexico: the USGS services stop at the border). Both
hosts are ArcGIS MapServer, so both are `/tile/{z}/{y}/{x}`, ROW BEFORE COLUMN. Ctrl/Cmd +
drag to draw an AOI. Mount Washington is the seed, not a fixed AOI.

THE GUARD is on the kernel, not the renderer: the stream, the fold, and the sevenfold ring
join. Nothing here caps what deck can draw, because the cell count never reaches it.

STILL TRUE, and worth knowing: this is a continuous surface, not a field of columns, and
`SimpleMeshLayer` has no per-feature hit test, so there is no picking.

`SurfaceLayer` is experimental and unexported (it is not in `lonboard.experimental.__init__`,
only `TextLayer` is). It works. One bug is patched below; see the PARQUET PATCH cell.

Run:  uv run marimo edit xsql-s1m-surface.py --sandbox
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
    from io import BytesIO

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

    return (
        AioHTTPAdapter,
        BitmapTileLayer,
        BytesIO,
        CartoBasemap,
        FullscreenControl,
        GeoTIFF,
        GeocoderControl,
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
def _(BytesIO):
    # THE PARQUET PATCH. Without this the kernel SEGFAULTS the moment a SurfaceLayer is
    # constructed, with no traceback, which is a miserable thing to debug from scratch.
    #
    # lonboard ships every synced arrow column to the browser as Parquet, and prefers
    # pyarrow's ParquetWriter over arro3's because pyarrow picks better encodings. The mesh
    # traits are FixedSizeList: tex_coords is 2-wide, positions and triangles are 3-wide.
    # Handing pyarrow 25.0.0 a 3-wide FixedSizeList that arrived over the arro3 C Data
    # Interface crashes inside ParquetWriter.__init__. Measured, narrowly:
    #
    #     FixedSizeList(2 x Float32)  via arro3 -> pyarrow   OK
    #     FixedSizeList(3 x Float32)  via arro3 -> pyarrow   SIGSEGV
    #     FixedSizeList(3 x UInt32)   via arro3 -> pyarrow   SIGSEGV
    #     the same three shapes built natively in pyarrow    all OK
    #     the same three shapes through arro3.io.write_parquet  all OK
    #
    # So it is the handoff, not either library alone, and `positions` cannot avoid being
    # 3-wide. lonboard already has the escape hatch: write_parquet_batch falls back to
    # arro3's own writer if pyarrow is not installed. This forces that branch always.
    # Costs a few percent of file size. Nothing else in the notebook notices.
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
    # S1M -> H3 -> one textured mesh

    Same pipeline as `xsql-s1m-h3.py` up to the H3 fold. Then the scene becomes **one
    triangle mesh** (geometry) plus **one image** (styling), neither of which scales with
    the cell count. Draw a box on the coverage map, then drive the surface from the
    controls under the scene.
    """)
    return


@app.cell
def _(Transformer, XarrayContext, coordinates_to_cells, h3ronpy, np, pa, udf):
    # Verbatim from xsql-s1m-h3.py: fit lon/lat per tile as an order-3 polynomial (pyproj
    # cannot run inside a DataFusion UDF; it aborts the process from worker threads) and
    # register the two h3ronpy UDFs.
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
def _(Transformer, np, pathlib, sqlite3, struct):
    # Verbatim from xsql-s1m-h3.py: the national S1M index is one ~15 MB GeoPackage, which
    # is SQLite, so stdlib sqlite3 reads it and every AOI is answered from a local file.
    S3_BASE = "https://prd-tnm.s3.amazonaws.com/"
    GPKG_KEY = "StagedProducts/Elevation/S1M/FullExtentSpatialMetadata/S1M_Products.gpkg"
    CACHE = pathlib.Path(".cache")

    def fetch_index(refresh=False):
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
        flags = blob[3]
        if (flags >> 1) & 0x07 == 0:
            raise ValueError("S1M footprint has no envelope in its GPB header")
        little = bool(flags & 0x01)
        xmin, xmax, ymin, ymax = struct.unpack_from("<4d" if little else ">4d", blob, 8)
        return xmin, ymin, xmax, ymax

    _path = fetch_index()
    with sqlite3.connect(f"file:{_path}?mode=ro", uri=True) as _con:
        _rows = _con.execute(
            "SELECT geom, tile, production_date, z_max, dataset_link FROM current"
        ).fetchall()

    # Reproject every footprint's CORNERS once, vectorised over the whole product. Albers
    # corners, not the lon/lat envelope: a 10 km Albers square is a slightly rotated quad in
    # degrees, and drawing it as an axis-aligned box would smear the national grid into a
    # staircase. Corner order SW, SE, NE, NW.
    _alb = np.array([_envelope(r[0]) for r in _rows], dtype="float64")
    _inv = Transformer.from_crs("EPSG:6350", "EPSG:4326", always_xy=True)
    _cx = np.column_stack([_alb[:, 0], _alb[:, 2], _alb[:, 2], _alb[:, 0]])
    _cy = np.column_stack([_alb[:, 1], _alb[:, 1], _alb[:, 3], _alb[:, 3]])
    _lon, _lat = _inv.transform(_cx.ravel(), _cy.ravel())
    _lon = _lon.reshape(-1, 4)
    _lat = _lat.reshape(-1, 4)

    tiles_all = [
        {
            "tile": r[1],
            "key": r[4].split("amazonaws.com/", 1)[-1],
            "produced": r[2] or "",
            # z_min carries a nodata sentinel (-999999) wherever a tile has holes, so the
            # coverage shading reads z_max only.
            "z_max": float(r[3]) if r[3] is not None else float("nan"),
            "albers": tuple(_alb[i]),
            "quad": list(zip(_lon[i], _lat[i])),  # SW, SE, NE, NW in lon/lat
        }
        for i, r in enumerate(_rows)
    ]
    tiles_albers = _alb

    print(f"S1M index: {len(tiles_all):,} current tiles from {_path}")
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
    # It answers the only question that matters before you draw a box: where does S1M exist
    # at all.
    #
    # Shaded by z_max on viridis at low opacity and with NO outlines, so neighbouring tiles
    # blend into one continuous field and the carpet reads as a single dissolved coverage
    # shape with elevation context, rather than as 11,717 separately coloured boxes.
    # Outlining each tile is what made it read as a grid. The coverage answer is the PRESENCE
    # of the shape, never its hue.
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
    # Mount Washington, New Hampshire as the SEED, not as a fixed AOI: the summit cone,
    # Tuckerman and Huntington ravines on the east face, enough of the Cog ridge to give the
    # surface something to do. ~7 x 9 km.
    #
    # Seeded rather than None (which is what xsql-s1m-h3.py does) because this notebook is
    # about looking at the render, so it should open with something on screen. Draw a box on
    # the picker below to go anywhere else.
    get_bbox, set_bbox = mo.state([-71.34, 44.23, -71.25, 44.31])
    return get_bbox, set_bbox


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
):
    # The National Map 3D Viewer's basemaps, as a deck.gl BitmapTileLayer: one more layer in
    # the same deck stack as the coverage carpet rather than a MapLibre style, so it sits
    # under the footprints and over Positron and can be swapped live.
    #
    # Two different hosts, one path shape. The viewer's DEFAULT is not a USGS service at all,
    # it is Esri's Topographic web map (portalItem 668f436d...), which is why terrain is
    # shaded across Canada and Mexico in it: the basemap.nationalmap.gov services stop at the
    # US border. Its raster form is World_Topo_Map on services.arcgisonline.com.
    #
    # Both hosts are ArcGIS MapServer, so both are /tile/{z}/{y}/{x}: ROW before column. In
    # XYZ order they return tiles from the wrong place rather than 404ing, which reads as a
    # projection bug rather than a typo.
    #
    # Terms: the arcgisonline raster tiles are unauthenticated but Esri scopes them to use
    # with Esri APIs or an API key. The National Map viewer is an Esri JS app so it qualifies
    # and this notebook does not. Kept as the default because it is the basemap the viewer
    # shows, but every USGS entry below is unencumbered and stays sharper over an AOI.
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

    # The picker, opened on CONUS with the coverage carpet already on it. Draw a box
    # (Ctrl/Cmd + drag) -> selected_bounds -> set_bbox.
    #
    # Built once and it references no reactive UI element, so pan/zoom/AOI survive every
    # downstream run. Nothing ever reassigns .layers.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        # Basemap tiles first so the coverage carpet draws on top of them. Positron stays
        # underneath: the National Map services stop at the US border, and without it the
        # rest of the world would be blank.
        layers=[basemap_layer, coverage_layer],
        view_state={"longitude": -96.0, "latitude": 38.5, "zoom": 3.6, "pitch": 0},
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
    picker.observe(
        lambda c: set_bbox(list(c["new"])) if c["new"] is not None else None,
        names="selected_bounds",
    )
    picker
    return BASEMAPS, basemap_layer


@app.cell
def _(BASEMAPS, mo):
    # Basemap picker for the map above. Its own cell, downstream of the picker, so choosing a
    # basemap never rebuilds the Map and never disturbs a box you have already drawn.
    basemap_choice = mo.ui.dropdown(
        options=list(BASEMAPS),
        value="Esri Topographic (viewer default)",
        label="Basemap",
    )
    basemap_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Basemap opacity"
    )
    mo.vstack(
        [
            mo.hstack([basemap_choice, basemap_opacity], justify="start", gap=2),
            mo.md(
                "<small>Basemaps: [USGS The National Map]"
                "(https://basemap.nationalmap.gov/) and Esri. Footprints on top are the "
                "1 m coverage index; Ctrl/Cmd + drag to draw an AOI.</small>"
            ),
        ],
        gap=0.5,
    )
    return basemap_choice, basemap_opacity


@app.cell
def _(BASEMAPS, basemap_choice, basemap_layer, basemap_opacity):
    # Live trait swap, same idiom as the scene layer: nudge the running BitmapTileLayer
    # instead of reassigning picker.layers, which would rebuild the deck stack.
    #
    # max_zoom moves with the URL, not after it. Leaving a deep Esri max_zoom on a USGS
    # service asks for z17+ tiles that do not exist and the basemap goes blank exactly when
    # you zoom in to place a box.
    _url, _maxz = BASEMAPS[basemap_choice.value]
    basemap_layer.max_zoom = _maxz
    basemap_layer.data = _url
    basemap_layer.opacity = basemap_opacity.value
    return


@app.cell
def _(get_bbox):
    bbox = list(get_bbox())
    return (bbox,)


@app.cell
def _(mo):
    h3_res = mo.ui.dropdown(
        options={
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9.4 m hex": 12,
            "res 13 ·  ~3.6 m hex": 13,
            "res 14 ·  ~1.35 m hex (near native)": 14,
        },
        value="res 12 ·  ~9.4 m hex",
        label="H3 resolution",
    )
    h3_res
    return (h3_res,)


@app.cell
def _(Transformer, bbox, h3_res, mo, np, tiles_albers, tiles_all):
    # Verbatim rule from xsql-s1m-h3.py: pick the overview geometrically so EVERY H3 cell is
    # guaranteed at least one pixel centre. p <= sqrt(2) * 0.5373 * sqrt(A), SAFETY 0.6.
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

    H3_CELL_M2 = {11: 2149.6, 12: 307.09, 13: 43.870, 14: 6.2673}
    SAFETY = 0.6
    _target_m = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])
    OVERVIEW_RES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    _fit = [r for r in OVERVIEW_RES if r <= _target_m]
    read_res_m = _fit[-1] if _fit else OVERVIEW_RES[0]

    candidates = [{**tiles_all[int(i)]} for i in np.flatnonzero(_hit)]

    # THE GUARD. It is on the KERNEL, not the renderer: the pixels streamed, the fold over
    # them, and the h3_grid_disk ring join, which explodes each cell into seven rows before
    # aggregating back down. Nothing here is about what deck can draw, because the cell count
    # never reaches the GPU. Estimate prints on every AOI; the stop only fires somewhere you
    # would not want to go by accident.
    CELL_GUARD = 40_000_000
    _est_cells = (_e - _w) * (_n - _s) / H3_CELL_M2[h3_res.value]
    _est_px = (_e - _w) * (_n - _s) / read_res_m**2
    print(
        f"AOI {tuple(round(v, 4) for v in bbox)} -> {len(candidates)} S1M tile(s) · "
        f"reading the {read_res_m:g} m overview for H3 res {h3_res.value} "
        f"(~{H3_CELL_M2[h3_res.value] / read_res_m**2:.0f} px per hex) · "
        f"~{_est_px / 1e6:.1f}M px in, ~{_est_cells / 1e6:.2f}M cells out"
    )
    mo.stop(
        _est_cells > CELL_GUARD,
        mo.md(
            f"### That AOI will not fit in the kernel\n"
            f"~**{_est_cells / 1e6:.0f}M** cells from ~**{_est_px / 1e6:.0f}M** pixels "
            f"(guard **{CELL_GUARD / 1e6:.0f}M**). This is a memory and time limit on the "
            f"stream and fold, **not** a rendering limit. "
            f"Lower the H3 resolution or draw a smaller box."
        ),
    )
    return aoi_albers, candidates, read_res_m


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    S3_BASE,
    Window,
    aoi_albers,
    asyncio,
    candidates,
    fit_lonlat,
    h3_res,
    make_h3_context,
    make_lonlat_udf,
    np,
    pa,
    read_res_m,
    xr,
):
    # THE READ AND THE FOLD, condensed from xsql-s1m-h3.py but not changed: stream each
    # tile's AOI window off the chosen overview, hand the Albers grid to xarray-sql, and let
    # ONE query turn metres into degrees (per-tile fitted UDF) and fold pixels into H3.
    # A second statement adds `flow` = how far each cell sits below its k-ring.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value

    def _window(reader, aoi_proj):
        pw, ps, pe, pn = aoi_proj
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw = max(pw, bw)
        ce = min(pe, be)
        cn = min(pn, bn)
        cs = max(ps, bs)
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
        reader = fit_lvls[-1] if fit_lvls else cands[0]
        win = _window(reader, aoi_albers)
        if win is None:
            return None
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)
        if not np.isfinite(elev).any():
            return None

        left, bottom, right, top = r.bounds
        h, w = elev.shape
        y = top - (np.arange(h) + 0.5) * (top - bottom) / h
        x = left + (np.arange(w) + 0.5) * (right - left) / w
        ds = xr.Dataset({"elevation": (("y", "x"), elev)}, coords={"y": y, "x": x})
        fit, err_mm = fit_lonlat(g.crs, (left, bottom, right, top))
        return ds, fit, err_mm, float(reader.res[0])

    print(f"streaming {len(candidates)} S1M COG(s):")
    for _t in candidates:
        print(f"  {S3_BASE}{_t['key']}")

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in candidates]) if d]
    if _datasets:
        _px = sum(int(d[0]["elevation"].size) for d in _datasets)
        print(f"streamed {_px:,} pixels from {len(_datasets)}/{len(candidates)} tile(s)")

        ctx = make_h3_context()
        for _i, (_d, _fit, _, _) in enumerate(_datasets):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"y": 1024})
            ctx.register_udf(make_lonlat_udf(f"to_lonlat_{_i}", _fit))
        _union = " UNION ALL ".join(
            f"SELECT p.lat AS lat, p.lon AS lon, elevation FROM ("
            f"  SELECT to_lonlat_{_i}(x, y) AS p, elevation"
            f"  FROM dem_{_i} WHERE elevation = elevation"
            f")"
            for _i in range(len(_datasets))
        )
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
    # scipy. Used twice, on the height field and on the shading value, so it lives here.
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
def _(flow_gain, h3_table, np):
    # THE SHADING VALUE, per cell: scene-relative elevation with flow added as an OFFSET so
    # drainage etches into the terrain colour. Gain 0 is pure elevation. The contrast slider
    # and the texture both read this one array, so they cannot disagree.
    cell_shade = np.asarray(h3_table["elevation"]).astype("float64") + flow_gain.value * (
        np.asarray(h3_table["flow"]).astype("float64")
    )
    cell_elev = np.asarray(h3_table["elevation"]).astype("float64")
    return cell_elev, cell_shade


@app.cell
def _(bbox, cell_rows, coordinates_to_cells, h3_res, np, pa, tex_size):
    # TEXEL -> CELL, on its own because it is the expensive half of everything below (a
    # coordinates_to_cells call plus a searchsorted over every texel: 4.2M of each at 2048)
    # and it depends only on the geometry of the problem. Split out this way, changing a
    # colour is a colormap over an existing index rather than a re-binning.
    #
    # Row 0 of the image is the SOUTH edge, because the mesh's tex_coord v runs 0..1 south to
    # north and WebGL samples v=0 at the first row. If the scene comes out mirrored
    # vertically, this assumption is the thing to flip.
    _T = tex_size.value
    _LON, _LAT = np.meshgrid(
        np.linspace(bbox[0], bbox[2], _T), np.linspace(bbox[1], bbox[3], _T)
    )
    _cells = np.asarray(
        pa.array(
            coordinates_to_cells(_LAT.ravel(), _LON.ravel(), h3_res.value)
        ).to_numpy(zero_copy_only=False)
    ).astype("uint64")
    _rows, _ok = cell_rows(_cells)
    texel_rows = _rows.reshape(_T, _T)
    texel_ok = _ok.reshape(_T, _T)
    print(f"texel index: {_T}x{_T} · {texel_ok.mean() * 100:.1f}% landed on a cell")
    return texel_ok, texel_rows


@app.cell
def _(bbox, box_mean, cell_elev, np, relief_smooth, texel_ok, texel_rows):
    # THE HEIGHT FIELD, in texture space, and this is where "angular and unnatural" gets
    # fixed.
    #
    # Straight off the fold, height is PIECEWISE CONSTANT: every hexagon is a flat plateau
    # with a vertical step to its neighbour. Sample that densely and you get literal
    # hexagonal stairs; sample it coarsely and you get arbitrary facets wherever the vertex
    # happened to land. Angular either way, and no mesh density fixes it, because the
    # staircase is in the DATA and not in the tessellation.
    #
    # So blur the height field itself. relief_smooth is in texels and turns the plateaus into
    # a continuous surface. It is deliberately SEPARATE from the colour smooth: this one
    # changes the shape (and therefore the hillshade), that one only changes the ramp.
    _elev = np.where(texel_ok, cell_elev[texel_rows] if cell_elev.size else 0.0, 0.0)
    _mask = texel_ok.astype("float64")
    _v, _m = box_mean(_elev, _mask, int(relief_smooth.value))
    height_tex = np.divide(_v, _m, out=np.zeros_like(_v), where=_m > 0)

    # Ground metres per texel, for the hillshade gradient. Equirectangular over an AOI this
    # small is fine; the error is well under one texel.
    _latm = (bbox[1] + bbox[3]) / 2.0
    px_m_x = abs(bbox[2] - bbox[0]) * 111_320.0 * np.cos(np.radians(_latm)) / _elev.shape[1]
    px_m_y = abs(bbox[3] - bbox[1]) * 111_320.0 / _elev.shape[0]
    return height_tex, px_m_x, px_m_y


@app.cell
def _(elevation_scale, height_tex, hillshade, np, px_m_x, px_m_y):
    # THE HILLSHADE, computed here in numpy because deck cannot compute it.
    #
    # lonboard's SurfaceLayer ships exactly two mesh attributes, POSITION and TEXCOORD_0.
    # There is NO NORMAL attribute in the bundle (verified in lonboard/static/index.js), so
    # deck's lighting has nothing to work with and the surface renders effectively unlit.
    # That is the real reason a mesh looks worse than extruded hexagons: the prisms have
    # vertical walls that catch light and give relief for free, and a flat-lit sheet gets
    # none of it. Adding normals would mean patching the JS, so instead the light is baked
    # into the texture, which is the ordinary fix for this.
    #
    # Standard surface normal against a light vector, sun at 315 degrees / 45 degrees up,
    # the cartographic convention. Row index increases NORTH (row 0 is the south edge), so
    # the y gradient is already d/d(north).
    #
    # The gradient uses elevation_scale, the SAME exaggeration the mesh geometry uses, so
    # what reads as steep is what actually is steep.
    _z = height_tex * max(elevation_scale.value, 1e-6)
    _dzdy, _dzdx = np.gradient(_z, px_m_y, px_m_x)

    _nx, _ny, _nz = -_dzdx, -_dzdy, np.ones_like(_z)
    _norm = np.sqrt(_nx * _nx + _ny * _ny + 1.0)

    _az, _alt = np.radians(315.0), np.radians(45.0)
    _lx = np.cos(_alt) * np.sin(_az)
    _ly = np.cos(_alt) * np.cos(_az)
    _lz = np.sin(_alt)

    _hs = np.clip((_nx * _lx + _ny * _ly + _nz * _lz) / _norm, 0.0, 1.0)

    # Ambient floor so shadowed faces keep their hue instead of going to black, then blend by
    # strength: 0 leaves the colours exactly as the ramp made them.
    AMBIENT = 0.35
    _f = AMBIENT + (1.0 - AMBIENT) * _hs
    shade_factor = 1.0 + hillshade.value * (_f - 1.0)
    return (shade_factor,)


@app.cell
def _(cell_shade, mo, np):
    # Contrast window over the shading value. Its bounds ARE this scene's range, so it resets
    # per AOI and per flow offset. Own cell, depending on cell_shade alone: palette and
    # reverse must never reach it, or picking a palette would rebuild the slider and throw
    # away the window you dragged.
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
    # THE TEXTURE. Paint the shading value into texture space, blur it, colour it, then
    # multiply in the baked hillshade.
    #
    # Blurring the VALUE rather than the finished RGB is what makes colour_smooth behave like
    # a coarser fold instead of like a soft-focus filter: the ramp still spans the same
    # contrast window, the hex plateaus just stop having hard walls. And it happens BEFORE
    # the hillshade multiply, so softening the colours never flattens the relief.
    _ = contrast
    _shade = np.where(texel_ok, cell_shade[texel_rows] if cell_shade.size else 0.0, 0.0)
    _mask = texel_ok.astype("float64")
    _v, _m = box_mean(_shade, _mask, int(colour_smooth.value))
    _shade = np.divide(_v, _m, out=np.zeros_like(_v), where=_m > 0)

    _lo, _hi = float(contrast_value[0]), float(contrast_value[1])
    _norm = np.clip((_shade - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
    if reverse_ramp.value:
        _norm = 1.0 - _norm

    _rgb = np.asarray(
        apply_continuous_cmap(_norm.ravel(), PALETTES[palette.value], alpha=1.0)
    )
    # apply_continuous_cmap returns RGB for some palettes and RGBA for others.
    if _rgb.shape[1] == 3:
        _rgb = np.concatenate(
            [_rgb, np.full((_rgb.shape[0], 1), 255, dtype=_rgb.dtype)], axis=1
        )
    _rgb = _rgb.astype("float64").reshape(*texel_ok.shape, 4)

    # Luminance modulation only: RGB scaled together, alpha untouched. No hue shift, so the
    # palette stays deuteranope-safe and relief arrives as a second, non-colour cue.
    _rgb[..., :3] *= shade_factor[..., None]

    texture = np.clip(_rgb, 0, 255).astype("uint8")
    # Holes stay transparent. The blur widens the valid region, so cut alpha with the
    # ORIGINAL mask or the scene grows a soft fringe past its own extent.
    texture[~texel_ok, 3] = 0
    print(f"texture: {texture.shape[1]}x{texture.shape[0]} ({texture.nbytes / 1e6:.1f} MB)")
    return (texture,)


@app.cell
def _(bbox, mesh_density, np):
    # MESH TOPOLOGY. Vertex count is (n+1)^2 and triangle count is 2n^2, fixed by the slider
    # and independent of the cell count. Own cell so moving the elevation scale re-uploads
    # positions without rebuilding indices.
    #
    # Vectorised: lonboard's own generate_mesh_grid() writes these indices with a Python
    # double loop, a quarter of a million iterations at density 512.
    _n = mesh_density.value
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

    _latm = (bbox[1] + bbox[3]) / 2.0
    _w_m = abs(bbox[2] - bbox[0]) * 111_320.0 * np.cos(np.radians(_latm))
    print(
        f"mesh: {len(tex_coords):,} vertices · {len(triangles):,} triangles · "
        f"{_w_m / _n:.1f} m per quad"
    )
    return tex_coords, triangles


@app.cell
def _(bbox, elevation_scale, height_tex, np, tex_coords):
    # MESH POSITIONS. Height comes from the SMOOTHED texture-space height field, not from a
    # per-vertex cell lookup, which matters twice over: the relief blur has already removed
    # the hexagonal staircase, and sampling the same array the hillshade was computed from
    # means shading and shape cannot drift apart.
    _G = height_tex.shape[0]
    _lon = bbox[0] + tex_coords[:, 0] * (bbox[2] - bbox[0])
    _lat = bbox[1] + tex_coords[:, 1] * (bbox[3] - bbox[1])

    _c = np.clip((tex_coords[:, 0] * (_G - 1)).round().astype("int64"), 0, _G - 1)
    _r = np.clip((tex_coords[:, 1] * (_G - 1)).round().astype("int64"), 0, _G - 1)
    _z = height_tex[_r, _c] * elevation_scale.value

    # float32 is what the trait casts to anyway: ~1 m of positional quantisation at lat 44,
    # invisible against a 9 km AOI.
    positions = np.stack([_lon, _lat, _z], axis=-1).astype("float32")
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
    # The layer and the Map are built ONCE, from placeholder geometry. This cell references
    # no control at all, so marimo never re-runs it and the view you flew to survives every
    # adjustment. The update cell at the bottom pushes the real arrays in.
    #
    # SurfaceLayer never populates _bbox (it has no geoarrow table to derive one from), so
    # the view state has to be explicit or the Map opens on null island.
    surface = SurfaceLayer(
        positions=np.zeros((4, 3), dtype="float32"),
        triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype="uint32"),
        tex_coords=np.zeros((4, 2), dtype="float32"),
        texture=np.zeros((1, 1, 4), dtype="uint8"),
    )
    scene = Map(
        layers=[surface],
        view_state={
            "longitude": (bbox[0] + bbox[2]) / 2,
            "latitude": (bbox[1] + bbox[3]) / 2,
            "zoom": 12.5,
            "pitch": 60,
            "bearing": -25,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.Positron),
        controls=[
            FullscreenControl(position="top-right"),
            NavigationControl(visualize_pitch=True),
            ScaleControl(),
        ],
        parameters={"depthTest": True, "blend": True},
    )
    scene
    return (surface,)


@app.cell
def _(PALETTES, mo):
    # EVERY SCENE CONTROL, in one cell, directly under the map it drives. Two rows: what the
    # surface LOOKS like, then what it IS. None of them rebuild the Map.
    palette = mo.ui.dropdown(options=list(PALETTES), value="Emrld", label="Palette")
    reverse_ramp = mo.ui.switch(value=True, label="Reverse")
    flow_gain = mo.ui.number(
        start=0.0, stop=50.0, step=0.5, value=8.0, debounce=True, label="Flow offset"
    )
    colour_smooth = mo.ui.slider(
        start=0, stop=24, step=1, value=2, label="Colour smooth", show_value=True
    )
    hillshade = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.7, label="Hillshade", show_value=True
    )

    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=3.0, debounce=True, label="Elevation scale"
    )
    relief_smooth = mo.ui.slider(
        start=0, stop=24, step=1, value=4, label="Relief smooth", show_value=True
    )
    mesh_density = mo.ui.slider(
        start=64, stop=2048, step=64, value=1024,
        label="Mesh density", show_value=True,
    )
    tex_size = mo.ui.dropdown(
        options={"1024": 1024, "2048": 2048, "4096": 4096},
        value="2048", label="Texture",
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Opacity"
    )
    wireframe = mo.ui.switch(value=False, label="Wireframe")

    mo.vstack(
        [
            mo.hstack(
                [palette, reverse_ramp, flow_gain, colour_smooth, hillshade],
                justify="start", gap=2,
            ),
            mo.hstack(
                [elevation_scale, relief_smooth, mesh_density, tex_size,
                 fill_opacity, wireframe],
                justify="start", gap=2,
            ),
        ],
        gap=0.75,
    )
    return (
        colour_smooth,
        elevation_scale,
        fill_opacity,
        flow_gain,
        hillshade,
        mesh_density,
        palette,
        relief_smooth,
        reverse_ramp,
        tex_size,
        wireframe,
    )


@app.cell
def _(PALETTES, contrast, mo, palette, reverse_ramp):
    # The slider paints the ramp it controls: same palette, same DIRECTION as the scene, so
    # "reversed" is something you see rather than infer, and the strip doubles as the legend.
    _hex = PALETTES[palette.value].hex_colors
    if reverse_ramp.value:
        _hex = _hex[::-1]
    _strip = mo.Html(
        '<div style="height:14px;width:100%;border-radius:3px;'
        'border:1px solid rgba(128,128,128,0.35);'
        f'background:linear-gradient(to right,{",".join(_hex)});"></div>'
    )
    contrast_value = contrast.value
    mo.vstack([_strip, contrast], gap=0)
    return (contrast_value,)


@app.cell
def _(
    fill_opacity,
    positions,
    surface,
    tex_coords,
    texture,
    triangles,
    wireframe,
):
    # The only thing the controls do: swap traits on the running layer. No Map rebuild, no
    # re-stream, no re-fold, no re-bin.
    #
    # BATCHED, because positions, tex_coords and triangles have to agree about vertex
    # indices. Moving the mesh density slider changes all three, and if they reach the widget
    # one at a time the frontend briefly holds indices that point past the end of the buffer.
    with surface.hold_trait_notifications():
        surface.positions = positions
        surface.tex_coords = tex_coords
        surface.triangles = triangles
        surface.texture = texture
        surface.wireframe = wireframe.value
        surface.opacity = fill_opacity.value
    return


if __name__ == "__main__":
    app.run()
