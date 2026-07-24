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
# ]
# ///
"""Free-fly the USA: draw a box, stream the 10m DEM, aggregate to H3, render extruded.

Same streaming/xarray-sql spine as xsql-dem-h3.py. After folding pixels into H3 cells,
each cell's elevation is re-based to the scene: SQL subtracts the AOI minimum so the
lowest cell sits at 0 and height reads RELATIVE to what's in view, not as absolute height
above sea level. Colored with CARTOColors Emrld (a luminance-monotonic green ramp,
deuteranope-safe), which normalizes over the scene's min -> max.

Run:  uv run marimo edit xsql-dem-rem.py --sandbox
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import pathlib
    import urllib.request
    import xml.etree.ElementTree as ET

    import numpy as np
    import pyarrow as pa
    import xarray as xr
    import marimo as mo

    from arro3.core import Table
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells

    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Photon
    from lonboard import Map, H3HexagonLayer
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
        ET,
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
        Table,
        Window,
        XarrayContext,
        apply_continuous_cmap,
        asyncio,
        coordinates_to_cells,
        mo,
        np,
        pa,
        pathlib,
        udf,
        urllib,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # DEM to H3

    Draw a box (Ctrl/Cmd + drag) anywhere in the lower 48. The 10m elevation streams in,
    bins into H3 hexagons, and renders extruded and colored by elevation.

    How it works: stream the tiles (obstore + async-geotiff), then one SQL query bins the
    pixels into hexagons (xarray-sql / DataFusion, with H3 via an h3ronpy UDF). Coloring
    and drawing are Python (numpy + lonboard). The SQL part is the binning; the rest is
    Python.
    """)
    return


@app.cell
def _(ET, pathlib, urllib):
    # Catalog = the VRT, not a STAC API. USGS publishes a nationwide VRT listing every
    # 1-degree seamless COG on prd-tnm with its exact placement, so AOI -> hrefs is a
    # local bbox intersection. Parse each <ComplexSource>: SourceFilename (minus the
    # /vsicurl/ prefix) + DstRect + GeoTransform -> a degree bbox.
    _vrt = pathlib.Path(".cache/USGS_Seamless_DEM_13.vrt")
    _vrt_url = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
        "USGS_Seamless_DEM_13.vrt"
    )
    if not _vrt.exists():
        _vrt.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading 1/3 arc-second seamless VRT index (~830 KB)...")
        urllib.request.urlretrieve(_vrt_url, _vrt)

    _root = ET.parse(_vrt).getroot()
    _gt = [float(v) for v in _root.find("GeoTransform").text.split(",")]
    dem_tiles = []
    for _src in _root.iter("ComplexSource"):
        # The COG path on the bucket, relative to the bucket root.
        _key = _src.find("SourceFilename").text.removeprefix("/vsicurl/")
        _key = _key.split("amazonaws.com/", 1)[-1]
        _rect = _src.find("DstRect")
        _w = _gt[0] + float(_rect.get("xOff")) * _gt[1]
        _n = _gt[3] + float(_rect.get("yOff")) * _gt[5]
        _e = _w + float(_rect.get("xSize")) * _gt[1]
        _s = _n + float(_rect.get("ySize")) * _gt[5]
        dem_tiles.append({"key": _key, "bbox": [_w, _s, _e, _n]})
    print(f"10m seamless tile index: {len(dem_tiles)} COGs")
    return (dem_tiles,)


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # xarray-sql is the point of this notebook: query the DEM raster AS a table with SQL,
    # no manual flattening. XarrayContext IS a DataFusion session with one extra trick,
    # from_dataset(): an xarray Dataset's dimension coords (lat, lon) become columns and
    # its data variables (elevation) become columns, so `SELECT lat, lon, elevation FROM
    # dem` unravels the grid for you. We register the H3 UDF on it (inherited DataFusion
    # method); it returns a UBIGINT (uint64) cell id, exactly what lonboard's H3HexagonLayer
    # consumes with high_precision=True. Bigint in, bigint on the GPU, no string round-trip.
    #
    # Color is deliberately NOT here: this cell + the aggregation below are the expensive
    # ETL that builds the hexagons, and it must NOT re-run when you only change the ramp.
    # Color lives in its own cheap cell downstream, so tuning it never re-streams or
    # re-aggregates. A factory (fresh context per stream) keeps per-tile table names from
    # colliding across re-runs.
    def _latlng_to_cell(lat, lng, res):
        return pa.array(
            coordinates_to_cells(lat.to_numpy(), lng.to_numpy(), res[0].as_py())
        )

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
        return ctx

    print("xarray-sql context factory ready; UDF: h3_latlng_to_cell(lat, lon, res) -> UBIGINT")
    return (make_h3_context,)


@app.cell
def _(mo):
    # Reactive AOI. Default: a tight box on Mount Washington and the Presidential Range,
    # White Mountains NH. Small on purpose so the first render is quick; draw a bigger box
    # (or search elsewhere) to fly the rest of the country.
    get_bbox, set_bbox = mo.state((-71.43, 44.165, -71.15, 44.385))
    return get_bbox, set_bbox


@app.cell
def _(mo):
    # Own cell: marimo re-runs any cell that references a UI element, and the picker map
    # must never reference this (a re-run would rebuild the map and drop the drawn AOI).
    # H3 average edge length by resolution: 8~461m, 9~174m, 10~66m, 11~25m, 12~9m. The 10m
    # DEM floors useful detail near res 12; large AOIs at fine res make a lot of cells.
    h3_res = mo.ui.dropdown(
        options={
            "res 8  ·  ~461 m hex": 8,
            "res 9  ·  ~174 m hex": 9,
            "res 10 ·  ~66 m hex": 10,
            "res 11 ·  ~25 m hex": 11,
            "res 12 ·  ~9 m hex (near native)": 12,
        },
        value="res 11 ·  ~25 m hex",
        label="H3 resolution",
    )
    h3_res
    return (h3_res,)


@app.cell
def _(mo):
    mo.md(r"""
    *Coverage: all of CONUS. The 10m seamless DEM tiles the entire lower 48, so you
    can draw a box anywhere on the map below.*
    """)
    return


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
    #
    # Photon (komoot): free, keyless, OSM-backed geocoder. Search a place to fly there,
    # then draw the box. geopy must run in async mode (AioHTTPAdapter) for lonboard.
    _geocoder = GeocoderControl.from_geopy(
        Photon(adapter_factory=AioHTTPAdapter, user_agent="x-sql-marimo"),
    )
    picker = Map(
        layers=[],
        view_state={"longitude": -71.29, "latitude": 44.275, "zoom": 10, "pitch": 0},
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
def _(dem_tiles, get_bbox):
    # Resolve WHICH COGs cover the AOI: a pure local bbox intersection over the VRT index.
    def cog_keys(bbox):
        w, s, e, n = bbox
        return [
            t
            for t in dem_tiles
            if t["bbox"][0] < e and t["bbox"][2] > w
            and t["bbox"][1] < n and t["bbox"][3] > s
        ]

    bbox = list(get_bbox())
    tiles = cog_keys(bbox)
    print(f"AOI {tuple(round(x, 4) for x in bbox)} -> {len(tiles)} COG(s)")
    return bbox, tiles


@app.cell
async def _(
    GeoTIFF,
    S3Store,
    Window,
    asyncio,
    bbox,
    h3_res,
    make_h3_context,
    mo,
    np,
    pa,
    tiles,
    xr,
):
    # Stream the covering COGs and aggregate to H3 with xarray-sql. For each tile: pick an
    # overview whose ground sampling roughly matches the H3 cell (so we neither oversample
    # nor starve cells), read ONLY the AOI window as an xarray Dataset (elevation over
    # lat/lon coords, nodata -> NaN), and register it as a SQL table. Then ONE SQL statement
    # unravels every tile's grid, unions them, and folds pixels into UBIGINT H3 cells via
    # the UDF. No manual flatten-to-pyarrow: the raster IS the table.
    _store = S3Store(bucket="prd-tnm", region="us-west-2", skip_signature=True)
    _res = h3_res.value
    _w, _s, _e, _n = bbox

    # Guard: refuse to build a scene the browser can't render. Estimate the cell count from
    # AOI area / H3 cell area BEFORE streaming (an upper bound: assumes full land coverage),
    # and stop if it exceeds the cap. Cheap pre-check, so a too-big AOI fails fast instead of
    # streaming for nothing. Reduce the H3 resolution or draw a smaller box.
    HEX_LIMIT = 5_000_000
    _cell_km2 = {8: 0.7373, 9: 0.10533, 10: 0.015047, 11: 0.0021496, 12: 0.00030712}[_res]
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

    # Target ground sampling ~ half the H3 edge, in degrees (1 deg lat ~ 111320 m).
    _edge_m = {8: 461.0, 9: 174.0, 10: 66.0, 11: 25.0, 12: 9.0}[_res]
    _target_deg = (_edge_m / 2.0) / 111320.0
    _PIXEL_BUDGET = 3_000_000  # per-tile window cap; step coarser if exceeded

    def _window(reader, tw, ts, te, tn):
        # AOI clipped to this reader's extent, in pixel coords.
        bw, bs, be, bn = reader.bounds
        xres = (be - bw) / reader.width
        yres = (bn - bs) / reader.height
        cw = max(_w, bw, tw); ce = min(_e, be, te)
        cn = min(_n, bn, tn); cs = max(_s, bs, ts)
        if ce <= cw or cn <= cs:
            return None
        col0 = int((cw - bw) / xres)
        col1 = int(np.ceil((ce - bw) / xres))
        row0 = int((bn - cn) / yres)
        row1 = int(np.ceil((bn - cs) / yres))
        col0 = max(0, col0); row0 = max(0, row0)
        col1 = min(reader.width, col1); row1 = min(reader.height, row1)
        if col1 <= col0 or row1 <= row0:
            return None
        return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)

    async def _read_tile(tile):
        g = await GeoTIFF.open(tile["key"], store=_store)
        tw, ts, te, tn = tile["bbox"]
        cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
        fit = [r for r in cands if r.res[0] <= _target_deg]
        # Walk from the matched overview toward coarser until the window fits the budget.
        start = cands.index(fit[-1]) if fit else 0
        for reader in cands[start:] if fit else cands:
            win = _window(reader, tw, ts, te, tn)
            if win is None:
                return None
            if win.width * win.height <= _PIXEL_BUDGET or reader is cands[-1]:
                break
        r = await reader.read(window=win)
        ma = r.as_masked()[0]
        elev = np.ma.filled(ma.astype("float32"), np.nan)  # nodata -> NaN
        if not np.isfinite(elev).any():
            return None
        rw, rs, re_, rn = r.bounds
        h, w = elev.shape
        # Pixel-centre coords. lat descends (north-up raster), lon ascends.
        lat = rn - (np.arange(h) + 0.5) * (rn - rs) / h
        lon = rw + (np.arange(w) + 0.5) * (re_ - rw) / w
        return xr.Dataset(
            {"elevation": (("lat", "lon"), elev)},
            coords={"lat": lat, "lon": lon},
        )

    _datasets = [d for d in await asyncio.gather(*[_read_tile(t) for t in tiles]) if d]
    if _datasets:
        _px = sum(int(d["elevation"].size) for d in _datasets)
        print(f"streamed {_px:,} pixels as {len(_datasets)} xarray Dataset(s)")

        # Register each tile's grid as a SQL table on the xarray-sql context, then let ONE
        # statement do the work: unravel every grid to (lat, lon, elevation) rows, drop
        # NaN nodata (elevation = elevation is false for NaN), union the tiles, and group
        # by H3 cell. This is the demonstration: SQL straight over xarray, UDF and all.
        ctx = make_h3_context()
        for _i, _d in enumerate(_datasets):
            ctx.from_dataset(f"dem_{_i}", _d, chunks={"lat": 1024})
        _union = " UNION ALL ".join(
            f"SELECT lat, lon, elevation FROM dem_{_i} WHERE elevation = elevation"
            for _i in range(len(_datasets))
        )

        # Fold pixels into H3 cells, then re-base each cell's elevation to the scene:
        # subtract the AOI minimum so the lowest cell sits at 0 and height reads RELATIVE to
        # what's in view, not as absolute height above sea level. Color normalizes min->max
        # downstream so it's unaffected; this is what makes the extrusion scene-relative.
        h3_table = ctx.sql(
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

        # flow = how far each hex sits below the ground around it. grid_disk gives each cell's
        # ring; average the ring's elevation and subtract the cell, so a low spot where water
        # collects comes out positive (bright) and ridges come out negative (dark).
        import h3ronpy
        _hx = np.asarray(h3_table["hex"])
        _ev = np.asarray(h3_table["elevation"], dtype="float64")
        _lut = dict(zip(_hx.tolist(), _ev.tolist()))
        _disk = h3ronpy.grid_disk(_hx, 1)
        try:
            _rings = _disk.to_pylist()
        except AttributeError:
            _rings = pa.array(_disk).to_pylist()
        _flow = np.empty(len(_hx), dtype="float64")
        for _i, _r in enumerate(_rings):
            _vals = [_lut[c] for c in _r if c in _lut]
            _flow[_i] = sum(_vals) / len(_vals) - _ev[_i]
        h3_table = h3_table.append_column("flow", pa.array(_flow))
        print(f"H3 res {_res}: {h3_table.num_rows:,} cells")
    else:
        h3_table = pa.table(
            {
                "hex": pa.array([], pa.uint64()),
                "elevation": pa.array([], pa.float64()),
                "flow": pa.array([], pa.float64()),
            }
        )
        print("no DEM pixels for this AOI")
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
    # COLOR CELL: separate from the ETL on purpose. Base is scene-relative ELEVATION; flow is
    # added as an OFFSET (flow_gain * flow) so drainage etches into the elevation shading
    # without losing the overall terrain read. Gain 0 = pure elevation. Depends on h3_table +
    # palette + gain + contrast, so it re-runs on those but never re-streams / re-folds.
    #
    # Domain is the CONTRAST WINDOW: a sub-range of the scene's own elevation min..max (never
    # CONUS), so dragging the handles in spends the whole palette on a narrower band. Both
    # directions precomputed for the live Reverse swap.
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
    # This cell references NEITHER the colors NOR the palette NOR any control. marimo re-runs
    # any cell that reads a UI element or a changed value, and a re-run here would rebuild the
    # Map (losing view state). So the layer is built ONCE with a placeholder fill, and the
    # update cell below paints it (and repaints on every palette / reverse change) as a live
    # get_fill_color trait swap: no Map rebuild, no re-stream, no re-fold. Only a new AOI /
    # resolution (which changes h3_table) rebuilds the scene.
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
            "zoom": 10,
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
    # Contrast window for the color domain. Its bounds ARE this scene's elevation min..max, so
    # it depends on h3_table and resets to the full range on every new AOI (right behavior:
    # bounds change per scene). Drag the handles in to spend the whole palette on a narrower
    # band, a live recolor, never a re-stream.
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
        debounce=True,  # recolor on release, not every drag tick (one buffer push, not dozens)
    )
    contrast
    return (contrast,)


@app.cell
def _(PALETTES, mo):
    # Right below the map: palette picker + float inputs (0.1 steppers) + toggles. Changing
    # the palette re-runs the color cell (cheap) then the update cell; scale / opacity /
    # reverse / extruded re-run only the update cell. None of them touch the stream, the SQL,
    # or rebuild the map, so the scene updates in place (lonboard's whole point).
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
    # rebuild, no re-stream, no re-fold, no re-color. This is the cell that reads the UI
    # elements, so it is the only one marimo re-runs when they change. Reverse just swaps
    # which precomputed color array feeds get_fill_color, live; Extruded flips 3D vs flat.
    h3_layer.elevation_scale = elevation_scale.value
    h3_layer.opacity = fill_opacity.value
    h3_layer.get_fill_color = colors_rev if reverse_ramp.value else colors_fwd
    h3_layer.extruded = extruded.value
    return


if __name__ == "__main__":
    app.run()
