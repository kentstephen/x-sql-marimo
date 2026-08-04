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
"""THE DEM MAKES THE SHAPE, H3 MAKES THE DATA. Same drape, with the round trip taken out.

This is `xsql-naip-drape.py` after the question that notebook could not answer: what is H3
actually for here. The drape used H3 for the HEIGHTS, and a raster folded into hexagons and
sampled straight back onto a lattice is a round trip. It costs resolution, it returns a
staircase that no mesh density and no resolution fixes, and it buys nothing the raster did
not already have. The terracing on every steep slope of that notebook is that round trip.

**That split is the change.** Everything else here follows from it, including the NDVI
surface, which exists because once H3 is off the geometry path it needs something real to
do. The default view is still the photograph.

So the two paths are split, and each gets the tool that suits it:

  * THE SHAPE COMES FROM THE DEM. Heights are bilinear off the streamed 10 m COGs straight
    onto the texel lattice. Smooth by construction, no plateaus, no `relief_smooth` needed
    to undo a quantisation this notebook never performs. The mesh can be coarser than the
    drape's, because the field it samples is smooth rather than stepped.
  * THE DATA COMES FROM H3, IN SQL, WHERE IT BELONGS. The fold is no longer in service of
    geometry, so it is free to do what a spatial index is for: aggregate, and JOIN.

THE QUERY IS THE POINT OF THE NOTEBOOK:

    WITH dem AS (SELECT h3_latlng_to_cell(y, x, res) AS hex, elevation FROM dem_i ...),
         d   AS (SELECT hex, avg(elevation), max(elevation) - min(elevation) AS relief,
                        count(*) AS n FROM dem GROUP BY 1),
         v   AS (SELECT hex, avg(ndvi) AS ndvi FROM naip_lattice GROUP BY 1)
    SELECT d.hex, d.elevation, d.relief, d.n, v.ndvi FROM d LEFT JOIN v USING (hex)

Two rasters, from two agencies, at two resolutions, in two different CRSs, joined on a
shared cell id in one DataFusion statement. No reprojection, no spatial index, no PostGIS,
no tiling server. That is the thing H3 exists for, and it is a better answer to "why H3"
than a height field ever was. `docs/xsql-s1m-surface-notes.md` says the same about vector
data: buildings polyfilled to cells is the same LEFT JOIN with a different right-hand side.

FOUR SURFACES, AND THE PHOTOGRAPH IS THE DEFAULT. `NAIP RGB` is the drape, unchanged and
now sitting on honest geometry. The other three are what the fold produces, and they are
one dropdown away rather than a different notebook:

  * `NDVI`      vegetation vigour, `(NIR - R) / (NIR + R)`, per cell.
  * `Elevation` the mean the fold computes anyway, which is the drape's old palette.
  * `Relief`    max - min elevation inside a cell: roughness at the resolution of the
                aggregation, free from the same GROUP BY, and a statistic about the pixels
                rather than a property of the surface they were folded into.

NDVI is worth its own paragraph only because of the band. NAIP ships R, G, B AND NIR, and
the drape notebook threw the NIR away because a photograph needs three. That fourth band is
the cheapest possible demonstration that a drape can carry DATA rather than a picture: same
pixels, same stream, one more band, and the surface answers a question instead of showing a
scene. Verified against ground truth while this was built: Big Cottonwood forest reads NDVI
0.39 median and 0.63 at p90, Utah Lake reads negative.

NDVI IS NOT RED-TO-GREEN HERE, and it is worth saying loudly because every NDVI map you
have ever seen is. Red-green is precisely the pair a deuteranope cannot resolve, so the
default ramp is BLUYL, blue through to yellow, off the same registry the other notebooks
use. It is monotonic in luminance, so the signal survives a colour-vision simulation and
hue only labels what brightness already said.

THE HEXAGONS ARE VISIBLE ON THE DATA SURFACES, AND THAT IS THE FEATURE. In the drape the
fold's hexagonal grain was an artifact fighting the photograph. Here it appears only when
you ask for a cell statistic, the shape underneath is smooth either way, and seeing the
cells is seeing the resolution of the analysis. Move `H3 resolution` and watch the
aggregation coarsen against terrain that does not move.

REQUIRED SETUP, and nothing about this notebook looks right without it:

    uv run python tools/patch_lonboard_surface.py

deck's SimpleMeshLayer reads `flatShading: !hasNormals`, and lonboard's SurfaceLayer sends
no NORMAL, so deck lights the mesh per triangle with a sun nobody asked for. On a colour
ramp it passes for texture; on imagery it is a herringbone of pale facets. The script
injects `material: false`. Re-run it after any install and HARD-RELOAD the browser. There
is a check below that shouts if it is missing. Full account in
`docs/xsql-naip-drape-notes.md`.

Run:  uv run marimo edit xsql-naip-ndvi.py --sandbox

`naip.py` must sit next to this file.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import urllib.request
    import xml.etree.ElementTree as ET
    from io import BytesIO

    import numpy as np
    import palettable
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    from obstore.store import S3Store, HTTPStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import BitmapTileLayer, Map
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
        SurfaceLayer,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        coordinates_to_cells,
        mo,
        naip,
        np,
        pa,
        palettable,
        pathlib,
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
    # DEM shape, H3 data

    The 10 m seamless DEM makes the **shape**, bilinear and smooth, with no fold in the
    geometry path: no hexagonal terracing at any resolution. H3 makes the **data**, which
    is what a spatial index is actually for: a DataFusion `GROUP BY` over the DEM pixels
    `LEFT JOIN`ed to a `GROUP BY` over NAIP's near-infrared band, on a shared cell id.

    Draw a box (Ctrl/Cmd + drag). It opens on the **photograph**; NDVI, elevation and
    relief are one dropdown away under the scene.
    """)
    return


@app.cell
def _(XarrayContext, coordinates_to_cells, np, pa, udf):
    # ONE UDF, and this notebook needs no more than that. The seamless 10 m COGs are
    # EPSG:4269, so their grid coordinates already ARE degrees and go straight into
    # `h3_latlng_to_cell(y, x, res)`. The Albers polynomial machinery in the S1M notebooks
    # exists only because 1 m tiles are projected; there is none of it here.
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

    print("DataFusion context factory ready (h3_latlng_to_cell)")
    return cells_of, make_ctx


@app.cell
def _(mo):
    # H3 RESOLUTION IS AN ANALYSIS CHOICE HERE, NOT A TERRAIN ONE, which is the whole point
    # of the split. In the drape it controlled how lumpy the mountain was, so it was really
    # a rendering knob wearing an analysis label. The shape no longer moves when you change
    # it. What changes is how finely the DEM and the NIR band are aggregated, and how many
    # pixels land in each cell, which the fold prints.
    _OPTS = {
        8: "res 8 ·  ~1.2 km hex",
        9: "res 9 ·  ~400 m hex",
        10: "res 10 ·  ~150 m hex",
        11: "res 11 ·  ~57 m hex",
        12: "res 12 ·  ~22 m hex",
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
    # NDVI wants LEAF-ON, which is the exact opposite of what the drape wants, and it is
    # worth being explicit that the same control now means something different. A bare
    # November canopy has no vigour to measure; a July one does.
    mo.vstack(
        [
            mo.hstack([h3_res, naip_season], justify="start", gap=2),
            mo.md(
                "<small>H3 resolution is an **analysis** choice here, not a terrain one: "
                "the shape comes from the DEM and does not move when you change it. Note "
                "that **NDVI wants leaf-on**, the opposite of what the drape wants, "
                "because a bare canopy has no vigour to measure. Neither control fetches "
                "anything until you draw a box.</small>"
            ),
        ],
        gap=0.5,
    )
    return h3_res, naip_season


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
    return DEM_DEG, S3_BASE, dem_tiles


@app.cell
def _(mo):
    # Big Cottonwood and Little Cottonwood, Wasatch: forested canyons against bare ridges
    # and a city edge, which is about the clearest NDVI contrast in the country inside one
    # small box. ~14 x 13 km, small enough that the texture is not the binding constraint.
    #
    # Others worth pasting in:
    #   Sangre de Cristo, NM   [-105.70, 36.50, -105.30, 36.85]
    #   Presidentials, NH      [-71.42, 44.16, -71.15, 44.36]
    #   Columbia Gorge, OR     [-121.95, 45.55, -121.75, 45.72]
    get_bbox, set_bbox = mo.state([-111.79, 40.55, -111.63, 40.66])

    # The first-run latch: opening the notebook should stop at the picker. Drawing a box is
    # the run, because it is the one input nothing can default and an unambiguous statement
    # of intent. It blocks until the first box and is transparent forever after.
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
        layers=[basemap_layer],
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
                "<small>Ctrl/Cmd + drag to draw an AOI. The 10 m DEM is nationwide, so "
                "anywhere in CONUS works; NAIP is checked before anything is "
                "streamed.</small>"
            ),
        ],
        gap=0.5,
    )
    return basemap_choice, basemap_opacity


@app.cell
def _(BASEMAPS, basemap_choice, basemap_layer, basemap_opacity):
    # Live trait swap: nudge the running layer rather than reassigning picker.layers, which
    # would rebuild the deck stack. max_zoom moves with the URL, not after it: leaving a
    # deep Esri max_zoom on a USGS service asks for tiles that do not exist and the basemap
    # goes blank exactly when you zoom in to place a box.
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
def _(bbox, get_started, mo, naip, naip_season, surface):
    # THE IMAGERY QUESTION FIRST, BEFORE ANY PIXEL IS STREAMED. The STAC search is seconds
    # and no pixels; the DEM is minutes and many. On an NDVI surface, imagery is not a
    # decoration but the measurement, so an AOI without it has nothing to show.
    MIN_COVER = 0.50

    _needs_naip = surface.value in ("NDVI", "NAIP RGB")
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

    _partial = (
        mo.md(
            f"**NAIP covers {naip_cover:.0%} of this box** — cells outside the imagery "
            f"get no NDVI and fall back to the elevation ramp."
        )
        if naip_quads and naip_cover < 0.995
        else None
    )
    _partial
    return MIN_COVER, naip_cover, naip_info, naip_quads


@app.cell
def _(
    DEM_DEG,
    MIN_COVER,
    bbox,
    dem_tiles,
    get_started,
    h3_res,
    mo,
    naip_cover,
    naip_info,
    np,
    surface,
    tex_size,
):
    mo.stop(
        not get_started(),
        mo.md(
            "### Draw a box to start\n"
            "Pick an **H3 resolution** above, then **Ctrl/Cmd + drag** on the map to draw "
            "an AOI. That starts the pipeline; nothing is fetched before it."
        ),
    )
    mo.stop(
        surface.value in ("NDVI", "NAIP RGB") and naip_cover < MIN_COVER,
        mo.md(
            f"### Not enough NAIP here\n"
            f"{'The STAC search failed' if naip_info and naip_info[0] == 'error' else f'NAIP covers {naip_cover:.0%} of this box'}"
            f", against a {MIN_COVER:.0%} floor, and **nothing has been streamed**: the "
            f"DEM read sits downstream of this check. Move the box, set **NAIP season** "
            f"to *Any*, or colour by **Elevation** or **Relief**, which need no imagery."
        ),
    )

    # WHICH COGs, AND AT WHAT RESOLUTION. Two consumers now want different things from the
    # same read, so the rule takes the finer of the two:
    #
    #   THE HEIGHT FIELD wants roughly one DEM sample per texel. Reading finer is download
    #   the lattice cannot hold; reading coarser is interpolating detail that was there.
    #   THE FOLD wants enough pixel centres per hexagon that `avg()` means something, at
    #   p <= sqrt(2) * 0.5373 * sqrt(A) taken at SAFETY 0.6.
    #
    # In the drape only the second existed, because the lattice was fed by the fold rather
    # than by the raster. Splitting the paths is what puts the first one back.
    _w, _s, _e, _n = bbox
    H3_CELL_M2 = {8: 737327.6, 9: 105332.5, 10: 15047.5, 11: 2149.6, 12: 307.09}
    SAFETY = 0.6

    _latm = (_s + _n) / 2.0
    aoi_w_m = (_e - _w) * 111_320.0 * np.cos(np.radians(_latm))
    aoi_h_m = (_n - _s) * 111_320.0

    _for_fold = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])
    _for_mesh = max(aoi_w_m, aoi_h_m) / tex_size.value
    _target_m = min(_for_fold, _for_mesh)

    candidates = [
        dict(t)
        for t in dem_tiles
        if t["bbox"][0] < _e and t["bbox"][2] > _w
        and t["bbox"][1] < _n and t["bbox"][3] > _s
    ]

    # The seamless COGs are EPSG:4269, so resolution is in DEGREES and a pixel is not
    # square on the ground: at latitude 40 one is ~10.3 m north-south but ~7.9 m east-west.
    # North-south is the larger and therefore the binding one, so degrees convert with
    # 111_320 and NO cosine. The cosine would overstate how fine the data is.
    _native_m = DEM_DEG * 111_320.0
    _levels = [_native_m * 2**k for k in range(6)]
    _fit = [r for r in _levels if r <= _target_m]
    read_res_m = _fit[-1] if _fit else _levels[0]
    read_res = read_res_m / 111_320.0  # source units are degrees

    print(
        f"AOI {tuple(round(v, 4) for v in bbox)} · {aoi_w_m / 1000:.1f} x "
        f"{aoi_h_m / 1000:.1f} km -> {len(candidates)} DEM COG(s) · reading the "
        f"{read_res_m:.1f} m level "
        f"(mesh wants {_for_mesh:.1f} m, fold wants {_for_fold:.1f} m) · "
        f"~{H3_CELL_M2[h3_res.value] / read_res_m**2:.0f} px per hex"
    )
    return aoi_h_m, aoi_w_m, candidates, read_res


@app.cell
async def _(GeoTIFF, S3Store, S3_BASE, Window, asyncio, bbox, candidates, np, read_res):
    # THE DEM STREAM. Unchanged from the drape except for what comes out: the arrays and
    # their bounds are kept as-is, because BOTH consumers want the raster rather than a
    # fold of it. The height field interpolates them; the SQL folds them.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)

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
        # One pixel of margin on every side, so the bilinear sample at the very edge of the
        # AOI still has four neighbours instead of falling off the array and going NaN.
        pad = reader.res[0] * 2
        win = _window(reader, (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad))
        if win is None:
            return None
        r = await reader.read(window=win)
        elev = np.ma.filled(r.as_masked()[0].astype("float32"), np.nan)
        # nodata is -999999 and the overviews carry it as a real value in places, so mask
        # on magnitude too or a single sentinel drags a whole cell's mean to -1e6.
        elev[elev < -1e5] = np.nan
        if not np.isfinite(elev).any():
            return None
        return elev, tuple(r.bounds)

    print(f"streaming {len(candidates)} DEM COG(s):")
    for _t in candidates:
        print(f"  {S3_BASE}{_t['key']}")

    dem_reads = [d for d in await asyncio.gather(*[_read_tile(t) for t in candidates]) if d]
    _px = sum(int(e.size) for e, _ in dem_reads)
    print(f"streamed {_px:,} DEM pixels from {len(dem_reads)}/{len(candidates)} COG(s)")
    return (dem_reads,)


@app.cell
def _(bbox, np, tex_size):
    # THE LATTICE. One regular lon/lat grid over the AOI, and everything downstream is
    # defined on it: heights, the H3 index, NDVI, the textures. Row 0 is the SOUTH edge,
    # because the mesh's tex_coord v runs 0..1 south to north and WebGL samples v=0 at the
    # first row. If the scene comes out mirrored vertically, this is the assumption to flip.
    _T = tex_size.value
    lat_1d = np.linspace(bbox[1], bbox[3], _T)
    lon_1d = np.linspace(bbox[0], bbox[2], _T)
    tex_lon, tex_lat = np.meshgrid(lon_1d, lat_1d)
    return tex_lat, tex_lon


@app.cell
def _(dem_reads, np, tex_lat, tex_lon):
    # HEIGHTS, BILINEAR OFF THE RASTER. This is the notebook's whole premise in twenty
    # lines: no fold, no hexagons, no `relief_smooth` to undo a quantisation that never
    # happens. The field is as smooth as the DEM is, which at 10 m is smoother than any
    # mesh here can resolve.
    #
    # Bilinear rather than nearest because nearest is what aliases: at a lattice finer than
    # the source, neighbouring texels snap to the same pixel and ridges come back as
    # stair-steps; at a lattice coarser than the source, nearest throws away three of every
    # four samples and picks an arbitrary one. Neither is a rounding detail on a shaded
    # surface, because the eye reads the DERIVATIVE, and nearest quantises the derivative
    # into flats and cliffs.
    #
    # NaN discipline: a bilinear cell touching a nodata pixel is NaN, so nodata dilates by
    # one pixel. That is correct and it is why the read pads the window.
    def _bilinear(elev, bounds, lon, lat):
        left, bottom, right, top = bounds
        h, w = elev.shape
        fx = (lon - left) / ((right - left) / w) - 0.5
        fy = (top - lat) / ((top - bottom) / h) - 0.5
        i0 = np.floor(fx).astype("int64")
        j0 = np.floor(fy).astype("int64")
        ok = (i0 >= 0) & (i0 < w - 1) & (j0 >= 0) & (j0 < h - 1)
        if not ok.any():
            return None, ok
        tx = fx - i0
        ty = fy - j0
        ic = np.clip(i0, 0, w - 2)
        jc = np.clip(j0, 0, h - 2)
        v = (
            elev[jc, ic] * (1 - tx) * (1 - ty)
            + elev[jc, ic + 1] * tx * (1 - ty)
            + elev[jc + 1, ic] * (1 - tx) * ty
            + elev[jc + 1, ic + 1] * tx * ty
        )
        return v, ok

    height_raw = np.full(tex_lon.shape, np.nan)
    for _elev, _bounds in dem_reads:
        _v, _ok = _bilinear(_elev, _bounds, tex_lon, tex_lat)
        if _v is None:
            continue
        _take = _ok & np.isfinite(_v) & ~np.isfinite(height_raw)
        height_raw[_take] = _v[_take]

    _hit = np.isfinite(height_raw)
    print(
        f"height field: {height_raw.shape[0]}x{height_raw.shape[1]} bilinear · "
        f"{_hit.mean() * 100:.1f}% covered · "
        f"{np.nanmin(height_raw):.0f}-{np.nanmax(height_raw):.0f} m"
    )
    return (height_raw,)


@app.cell
async def _(
    GeoTIFF,
    HTTPStore,
    Window,
    aoi_w_m,
    naip,
    naip_quads,
    np,
    surface,
    tex_lat,
    tex_lon,
):
    # NAIP ON THE LATTICE, FOUR BANDS, ONCE. This read exists to produce a NUMBER per texel
    # rather than a picture, so it wants the lattice resolution and nothing finer, and it
    # is skipped entirely when the surface is coloured by elevation or relief.
    #
    # The drape reads per tile at tile resolution because a photograph is worth every texel
    # you can give it. NDVI is not: it is going to be averaged into hexagons a moment from
    # now, and at res 10 a cell is 150 m across, so a 7 m texel is already far finer than
    # the analysis.
    _texel_m = aoi_w_m / tex_lon.shape[1]
    if naip_quads and surface.value == "NDVI":
        _px, ndvi_cover, _info = await naip.naip_rgb(
            naip_quads, tex_lon, tex_lat, [
                float(tex_lon.min()), float(tex_lat.min()),
                float(tex_lon.max()), float(tex_lat.max()),
            ],
            _texel_m, GeoTIFF, HTTPStore, Window, bands=4,
        )
        # NDVI = (NIR - R) / (NIR + R). uint8 in, float out; the denominator is guarded
        # because a black pixel is 0/0 and would otherwise be a warning and a NaN.
        _r = _px[..., 0].astype("float32")
        _nir = _px[..., 3].astype("float32")
        ndvi_tex = np.where(ndvi_cover, (_nir - _r) / np.maximum(_nir + _r, 1.0), np.nan)
        print(
            f"NDVI on the lattice: {_info['quads_read']} quad(s) at "
            f"{_texel_m:.1f} m/texel vs {_info['source_res_m']:.1f} m native · "
            f"{ndvi_cover.mean() * 100:.1f}% painted · "
            f"median {np.nanmedian(ndvi_tex):+.2f}"
        )
    else:
        ndvi_tex = np.full(tex_lon.shape, np.nan, dtype="float32")
        ndvi_cover = np.zeros(tex_lon.shape, dtype=bool)
    return (ndvi_tex,)


@app.cell
def _(
    cells_of,
    dem_reads,
    h3_res,
    make_ctx,
    ndvi_tex,
    np,
    pa,
    tex_lat,
    tex_lon,
    xr,
):
    # THE FOLD, AND THE JOIN. This is why the repository exists.
    #
    # Two rasters that share no CRS, no resolution, no provider and no tiling scheme are
    # aggregated to the SAME KEY and joined on it, in one DataFusion statement. The DEM
    # goes in as its native grid through xarray-sql, one relation per COG, and its
    # coordinates already are degrees. NAIP goes in as the lattice it was sampled onto.
    # H3 makes them the same shape of thing. Nothing is reprojected and nothing is
    # rasterised to a common grid, which is the step this normally costs.
    #
    # `relief` (max - min elevation inside a cell) is free here and worth having: it is
    # roughness at the resolution of the aggregation, it needs no neighbours and therefore
    # no ring join, and it says something the height field cannot, because the height field
    # is a surface and this is a statistic about the pixels under it.
    _res = h3_res.value
    ctx = make_ctx()

    for _i, (_elev, _bounds) in enumerate(dem_reads):
        _left, _bottom, _right, _top = _bounds
        _h, _w = _elev.shape
        _yy = _top - (np.arange(_h) + 0.5) * (_top - _bottom) / _h
        _xx = _left + (np.arange(_w) + 0.5) * (_right - _left) / _w
        ctx.from_dataset(
            f"dem_{_i}",
            xr.Dataset({"elevation": (("y", "x"), _elev)}, coords={"y": _yy, "x": _xx}),
            chunks={"y": 1024},
        )

    # The NAIP side arrives pre-indexed: the lattice cells are computed once here and
    # reused to paint the result back, so the expensive `coordinates_to_cells` call happens
    # a single time rather than once for the fold and once for the render.
    tex_cells = cells_of(tex_lat, tex_lon, _res)
    _flat = ndvi_tex.ravel()
    _good = np.isfinite(_flat)
    ctx.from_arrow(
        pa.table({"hex": pa.array(tex_cells[_good]), "ndvi": pa.array(_flat[_good])}),
        name="naip_lattice",
    )

    _dem_union = " UNION ALL ".join(
        f"SELECT h3_latlng_to_cell(y, x, CAST({_res} AS INT)) AS hex, elevation "
        f"FROM dem_{_i} WHERE elevation = elevation"
        for _i in range(len(dem_reads))
    )
    cell_table = ctx.sql(
        f"""
        WITH d AS (
            SELECT hex,
                   avg(elevation) AS elevation,
                   max(elevation) - min(elevation) AS relief,
                   count(*) AS n
            FROM ({_dem_union})
            GROUP BY 1
        ),
        v AS (
            SELECT hex, avg(ndvi) AS ndvi, count(*) AS n_ndvi
            FROM naip_lattice
            GROUP BY 1
        )
        SELECT d.hex, d.elevation, d.relief, d.n, v.ndvi
        FROM d LEFT JOIN v USING (hex)
        """
    ).to_arrow_table()

    _n = np.asarray(cell_table["n"])
    _nd = np.asarray(cell_table["ndvi"])
    print(
        f"H3 res {_res}: {cell_table.num_rows:,} cells · "
        f"{_n.mean():.1f} DEM px/cell (min {_n.min()}) · "
        f"{np.isfinite(_nd).mean() * 100:.0f}% of cells carry NDVI"
    )
    return cell_table, tex_cells


@app.cell
def _(cell_table, np, tex_cells, tex_lon):
    # Cell id -> row, then every texel is a searchsorted. `ok` is False for cells the fold
    # never saw, which the texture turns transparent. Sorting once is what makes painting
    # 4M texels from a few hundred thousand cells cheap.
    _hex = np.asarray(cell_table["hex"]).astype("uint64")
    _order = np.argsort(_hex)
    _sorted = _hex[_order]

    _pos = np.clip(np.searchsorted(_sorted, tex_cells), 0, max(_sorted.size - 1, 0))
    _found = _sorted[_pos] == tex_cells if _sorted.size else np.zeros(tex_cells.shape, bool)
    texel_rows = _order[_pos].reshape(tex_lon.shape)
    texel_ok = _found.reshape(tex_lon.shape)

    def cell_field(name):
        """A per-cell column, painted onto the lattice. NaN where the cell is missing."""
        _v = np.asarray(cell_table[name]).astype("float64")
        out = np.where(texel_ok, _v[texel_rows], np.nan)
        return out

    print(f"texel index: {texel_ok.shape[0]}^2 · {texel_ok.mean() * 100:.1f}% on a cell")
    return cell_field, texel_ok


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
def _(aoi_h_m, aoi_w_m, box_mean, height_raw, np, smooth):
    # THE HEIGHT FIELD THE MESH ACTUALLY SAMPLES. `smooth` defaults to ZERO here, and that
    # is the visible difference from the drape notebook, where `relief_smooth` defaulted to
    # 3 and existed to sand down hexagonal plateaus. There are no plateaus to sand. The
    # control stays because a 10 m DEM has its own noise (lidar collection seams, void
    # fills) and a wide box sometimes reads better with a touch of blur, but it is now an
    # aesthetic choice rather than a repair.
    _v = np.where(np.isfinite(height_raw), height_raw, 0.0)
    _m = np.isfinite(height_raw).astype("float64")
    _vs, _ms = box_mean(_v, _m, int(smooth.value))
    height_tex = np.divide(_vs, _ms, out=np.zeros_like(_vs), where=_ms > 0)
    height_tex -= height_tex[_ms > 0].min() if (_ms > 0).any() else 0.0

    px_m_x = aoi_w_m / height_tex.shape[1]
    px_m_y = aoi_h_m / height_tex.shape[0]
    return height_tex, px_m_x, px_m_y


@app.cell
def _(elevation_scale, height_tex, hillshade, np, px_m_x, px_m_y):
    # THE HILLSHADE, in numpy, and now for a different reason than the drape had. It used
    # to compensate for a mesh deck could not light. deck CAN light it, per flat triangle,
    # which is exactly the artifact `tools/patch_lonboard_surface.py` turns off. So the
    # only sun in the scene is this one, and it is a real Lambertian shade computed from
    # the height field at the same exaggeration the mesh uses.
    #
    # It applies to the DATA surfaces only. A NAIP photograph was taken in real sunlight and
    # already carries the real shadows; a second synthetic sun double-shades it into a
    # glossy shell. Sun at 315/45, the cartographic convention. Row index increases NORTH,
    # so the y gradient is already d/d(north).
    _z = height_tex * max(elevation_scale.value, 1e-6)
    _dzdy, _dzdx = np.gradient(_z, px_m_y, px_m_x)
    _nx, _ny, _nz = -_dzdx, -_dzdy, np.ones_like(_z)
    _norm = np.sqrt(_nx * _nx + _ny * _ny + 1.0)

    _az, _alt = np.radians(315.0), np.radians(45.0)
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
    AMBIENT = 0.35
    shade_factor = 1.0 + hillshade.value * (AMBIENT + (1.0 - AMBIENT) * _hs - 1.0)
    return (shade_factor,)


@app.cell
def _():
    # Palette registry: matplotlib + CARTOColors sequential ramps. All luminance-monotonic
    # and free of red/green opposition, so they survive a deuteranope simulation. Same
    # registry as the other notebooks in the repo, deliberately: a ramp should mean the
    # same thing across them.
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
def _(bbox, np, tex_size, tile_grid):
    # THE TILE GRID, geometry only. Each tile is its own mesh and its own texture over
    # 1/N^2 of the ground, which multiplies ground resolution by N without any single
    # texture growing past what a GPU will accept.
    #
    # TILES SHARE THEIR EDGE SAMPLES: tile i spans [w + i*step, w + (i+1)*step] with S+1
    # samples INCLUSIVE of both ends, so tile i's last column and tile i+1's first column
    # are the same coordinate carrying the same height. That is what keeps the seams
    # invisible: not a tolerance, an identity.
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
    return (tiles,)


@app.cell
def _(bbox, np):
    # NEAREST-SAMPLE a global lattice array onto one tile's lattice. Separable, so the row
    # and column indices are built once as 1-D and combined with np.ix_ rather than
    # materialising two (S+1)^2 index grids per tile per array.
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
async def _(
    GeoTIFF,
    HTTPStore,
    Window,
    aoi_w_m,
    naip,
    naip_quads,
    np,
    surface,
    tex_size,
    tile_grid,
    tiles,
):
    # THE PHOTOGRAPH, per tile, and only when the surface IS the photograph. Each tile reads
    # the quads that overlap it, through its own window, at the overview matching its texel
    # size, which is the whole reason tiling buys resolution rather than just memory.
    #
    # Sequential over tiles, concurrent within one: `naip_rgb` already gathers across quads
    # and caps itself, and a grid firing every quad of every tile at once is dozens of
    # concurrent range reads against one host.
    _texel_m = aoi_w_m / (tile_grid.value * tex_size.value)
    if naip_quads and surface.value == "NAIP RGB":
        _opened = await naip.open_quads(naip_quads, GeoTIFF, HTTPStore)
        photo = []
        _read, _cov = 0, []
        for _t in tiles:
            _lon, _lat = np.meshgrid(_t["lon"], _t["lat"])
            _rgb, _c, _info = await naip.naip_rgb(
                naip_quads, _lon, _lat, _t["bbox"], _texel_m,
                GeoTIFF, HTTPStore, Window, _opened,
            )
            photo.append((_rgb, _c))
            _read += _info["quads_read"]
            _cov.append(_info["covered"])
        _src = min((r.res[0] for r in _opened.values()), default=float("nan"))
        print(
            f"NAIP photo: {len(tiles)} tile(s), {_read} quad read(s) · "
            f"{_texel_m:.2f} m/texel vs {_src:.1f} m native "
            f"({_texel_m / _src:.0f}x coarser) · "
            f"{float(np.mean(_cov)) * 100:.1f}% painted"
        )
    else:
        photo = [
            (
                np.zeros((len(_t["lat"]), len(_t["lon"]), 3), dtype="uint8"),
                np.zeros((len(_t["lat"]), len(_t["lon"])), dtype=bool),
            )
            for _t in tiles
        ]
    return (photo,)


@app.cell
def _(PALETTES, apply_continuous_cmap, cell_field, np, ndvi_range, palette, surface):
    # THE DATA SURFACE, computed once on the global lattice, because it is a function of
    # the cell values alone and every tile just samples the result.
    #
    # NDVI IS WINDOWED, NOT NORMALISED. Stretching a ramp to a scene's own min and max is
    # what makes NDVI maps incomparable between AOIs and between dates: the same forest
    # gets a different colour depending on what else is in the box. -0.2 to 0.8 covers
    # water through bare rock through dense canopy everywhere on earth, so a colour means
    # the same thing in every scene this notebook draws. The slider moves the window when
    # you want contrast, and it says what it costs.
    if surface.value == "NDVI":
        _v = cell_field("ndvi")
        _lo, _hi = float(ndvi_range.value[0]), float(ndvi_range.value[1])
    elif surface.value == "Relief":
        _v = cell_field("relief")
        _finite = _v[np.isfinite(_v)]
        _lo, _hi = (
            (float(np.percentile(_finite, 2)), float(np.percentile(_finite, 98)))
            if _finite.size
            else (0.0, 1.0)
        )
    else:  # Elevation, and also the fallback under the photograph
        _v = cell_field("elevation")
        _finite = _v[np.isfinite(_v)]
        _lo, _hi = (
            (float(np.nanmin(_finite)), float(np.nanmax(_finite)))
            if _finite.size
            else (0.0, 1.0)
        )

    data_ok = np.isfinite(_v)
    _norm = np.clip((np.where(data_ok, _v, _lo) - _lo) / max(_hi - _lo, 1e-9), 0.0, 1.0)
    data_rgb = np.asarray(
        apply_continuous_cmap(_norm.ravel(), PALETTES[palette.value], alpha=1.0)
    )[:, :3].astype("float64").reshape(*_norm.shape, 3)

    if surface.value == "NAIP RGB":
        # Still computed, because a tile with no imagery falls back to it rather than
        # rendering a hole. Nothing draws it when every tile has a photograph.
        print(f"elevation ramp held as the fallback: {_lo:.3g} .. {_hi:.3g} m")
    else:
        print(f"surface [{surface.value}]: ramp {_lo:.3g} .. {_hi:.3g}")
    return data_ok, data_rgb


@app.cell
def _(
    data_ok,
    data_rgb,
    np,
    photo,
    shade_factor,
    surface,
    texel_ok,
    tile_sample,
    tiles,
):
    # THE TEXTURES, one RGBA image per tile.
    #
    # `visible` differs by source, and per tile rather than per scene: a grid straddling the
    # edge of a NAIP year gets photograph where there is photograph and ramp elsewhere,
    # rather than all-or-nothing.
    textures = []
    for _k, _t in enumerate(tiles):
        _rgb_src, _cover = photo[_k]
        _ok = tile_sample(texel_ok, _t)
        if surface.value == "NAIP RGB" and _cover.any():
            rgb = _rgb_src.astype("float64")
            visible = _ok & _cover
        else:
            rgb = tile_sample(data_rgb, _t)
            # The hillshade is for the DATA surfaces only. A photograph brought its own sun.
            rgb = rgb * tile_sample(shade_factor, _t)[..., None]
            visible = _ok & tile_sample(data_ok, _t)

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
def _(aoi_w_m, mesh_density, np, tile_grid):
    # MESH TOPOLOGY, built once and shared by every tile. Vertex count is (n+1)^2 and
    # triangle count is 2n^2, fixed by the slider and independent of the cell count, which
    # is what makes a wide box affordable at all.
    #
    # The slider is density ACROSS THE AOI, divided by the grid, so the triangle budget is
    # constant as tiles are added: N^2 tiles of (density/N)^2 quads is the same 2*density^2
    # triangles however the grid is set. Tiling is for texels, not for geometry.
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
        f"{len(triangles):,} triangles) · {aoi_w_m / (_n * tile_grid.value):.1f} m/quad"
    )
    return tex_coords, triangles


@app.cell
def _(bbox, elevation_scale, height_tex, np, tex_coords, tiles):
    # MESH POSITIONS, one array per tile, sampling the SAME global height field. That shared
    # field is what welds the grid shut: a vertex on a tile boundary has one lon/lat, indexes
    # one height, and both neighbours place it identically. There is no crack to reconcile
    # because there is no second opinion about the ground.
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
    # no control, so marimo never re-runs it and the view you flew to survives every
    # adjustment. The update cell at the bottom pushes the real arrays in.
    #
    # A FIXED POOL sized to the largest grid the dropdown offers, because `Map.layers` must
    # not be reassigned: that throws away the camera. Unused layers park on a degenerate
    # triangle with a transparent texel and cost one draw call of nothing.
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
            "zoom": 11.5,
            "pitch": 60,
            "bearing": -25,
        },
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter),
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
        # therefore reads as an artifact rather than as a broken render. See
        # docs/xsql-naip-drape-notes.md.
        parameters={
            "depthTest": True,
            "depthCompare": "less-equal",
            "depthWriteEnabled": True,
            "blend": True,
        },
    )
    scene
    return (surfaces,)


@app.cell
def _(PALETTES, mo):
    # EVERY SCENE CONTROL, in one cell under the map it drives. None of them rebuild the
    # Map, and none of them re-stream anything except `surface`, which decides whether NAIP
    # is fetched at all and in which shape.
    surface = mo.ui.dropdown(
        options=["NAIP RGB", "NDVI", "Elevation", "Relief"],
        value="NAIP RGB",
        label="Colour by",
    )
    palette = mo.ui.dropdown(options=list(PALETTES), value="BluYl", label="Ramp")
    ndvi_range = mo.ui.range_slider(
        start=-1.0, stop=1.0, step=0.05, value=[-0.2, 0.8],
        label="NDVI window", show_value=True, debounce=True,
    )
    hillshade = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6, label="Hillshade", show_value=True
    )
    elevation_scale = mo.ui.number(
        start=0.0, stop=50.0, step=0.1, value=1.5, debounce=True, label="Elevation scale"
    )
    # Defaults to ZERO, unlike the drape's `relief_smooth`. There is no hexagonal staircase
    # in this height field to sand down, because the height field never went through H3.
    smooth = mo.ui.slider(
        start=0, stop=16, step=1, value=0, label="Height smooth", show_value=True
    )
    mesh_density = mo.ui.slider(
        start=64, stop=2048, step=64, value=1024, label="Mesh density", show_value=True
    )
    tex_size = mo.ui.dropdown(
        options={"1024": 1024, "2048": 2048, "4096": 4096},
        value="2048",
        label="Texture / tile",
    )
    tile_grid = mo.ui.dropdown(
        options={"1x1": 1, "2x2": 2, "3x3": 3, "4x4": 4}, value="1x1", label="Drape tiles"
    )
    fill_opacity = mo.ui.number(
        start=0.0, stop=1.0, step=0.1, value=1.0, debounce=True, label="Opacity"
    )
    wireframe = mo.ui.switch(value=False, label="Wireframe")

    mo.vstack(
        [
            mo.hstack([surface, palette, ndvi_range], justify="start", gap=2),
            mo.hstack(
                [elevation_scale, hillshade, smooth, mesh_density, tex_size, tile_grid,
                 fill_opacity, wireframe],
                justify="start", gap=2,
            ),
        ],
        gap=0.75,
    )
    return (
        elevation_scale,
        fill_opacity,
        hillshade,
        mesh_density,
        ndvi_range,
        palette,
        smooth,
        surface,
        tex_size,
        tile_grid,
        wireframe,
    )


@app.cell
def _(PALETTES, mo, ndvi_range, palette, surface):
    # The legend paints the ramp it explains, in the direction the scene uses, and says what
    # the numbers mean. NDVI is unitless and its conventional breakpoints are worth stating
    # rather than leaving to the eye.
    _hex = PALETTES[palette.value].hex_colors
    _strip = mo.Html(
        '<div style="height:14px;width:100%;border-radius:3px;'
        'border:1px solid rgba(128,128,128,0.35);'
        f'background:linear-gradient(to right,{",".join(_hex)});"></div>'
    )
    if surface.value == "NDVI":
        _out = mo.vstack(
            [
                _strip,
                mo.md(
                    f"<small>**NDVI** = (NIR − Red) / (NIR + Red), from NAIP's fourth "
                    f"band, averaged per H3 cell in SQL. Window "
                    f"**{ndvi_range.value[0]:+.2f} to {ndvi_range.value[1]:+.2f}**, held "
                    f"fixed rather than stretched to the scene so a colour means the same "
                    f"thing in every AOI. Rough guide: below 0 water and snow, 0 to 0.2 "
                    f"rock, soil and pavement, 0.2 to 0.4 grass and sage, 0.4 to 0.8 "
                    f"closed canopy. **Not a red-green ramp**: that pair is exactly the "
                    f"one a deuteranope cannot resolve.</small>"
                ),
            ],
            gap=0.25,
        )
    elif surface.value == "NAIP RGB":
        _out = mo.md(
            "<small>The photograph, at the tile lattice rather than the cell grid: no "
            "hexagons, no hillshade, no ramp. The H3 fold still ran and the other three "
            "surfaces are one dropdown away.</small>"
        )
    else:
        _out = mo.vstack(
            [
                _strip,
                mo.md(
                    f"<small>**{surface.value}** per H3 cell. *Relief* is "
                    f"max − min elevation inside a cell, i.e. roughness at the resolution "
                    f"of the aggregation, which is a statistic about the pixels rather "
                    f"than a property of the surface they were folded into.</small>"
                ),
            ],
            gap=0.25,
        )
    _out
    return


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
    # The only thing the controls do: swap traits on the running layers. No Map rebuild.
    #
    # BATCHED PER LAYER, because positions, tex_coords and triangles have to agree about
    # vertex indices. Moving the mesh density slider changes all three, and if they reach
    # the widget one at a time the frontend briefly holds indices past the end of a buffer.
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
                # Shrinking the grid has to actively blank the surplus layers, or last
                # run's geometry draws a coarse copy underneath the new one.
                _layer.positions = _blank_pos
                _layer.tex_coords = _blank_uv
                _layer.triangles = _blank_tri
                _layer.texture = _blank_tex
    return


if __name__ == "__main__":
    app.run()
