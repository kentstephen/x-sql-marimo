# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy",
#     "pyproj>=3.7",
# ]
# ///
"""Annual NLCD land cover in H3, finer as you zoom in, folded for the viewport.

Nothing is read until the camera asks for it. Each fold pulls only the padded viewport,
from the overview that matches the H3 resolution it is about to build, registers that
window with xarray-sql and folds it in SQL. The counter-intuitive part is that the FINEST
views are the cheapest: the viewport shrinks faster than the resolution grows, so res 11
at 30 m reads 72,890 pixels where res 5 at 960 m reads 16.2M.

That is what gets to res 11. Below it the cells would be finer than the imagery: a res 11
hexagon holds 2.3 pixels of 30 m NLCD, and res 12 would hold 0.6 and hole out. The ceiling
is the data's, not the code's.

The fold is a mode, not a mean, because land cover is categorical: each cell takes its
most frequent class, and colour is the class. Flat, not extruded: there is no height in
a land cover map, and hexagon walls only hide the classes behind them.

The camera never re-runs a marimo cell. It schedules a coroutine that reads, folds and
swaps three traits on the one live layer, so panning and zooming stay fluid and the view
is never reset. Same shape as the Jupyter tutorial in `bias-bounty-map-tutorial`, which is
where the pattern is proven.

Data: Kyle Barron's mirror of USGS Annual NLCD on source.coop, public and unsigned.

Run:  uv run marimo edit xsql-nlcd-zoom.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import math

    import anywidget
    import traitlets
    import marimo as mo
    import numpy as np
    import pyarrow as pa
    import xarray as xr
    from arro3.core import Table as ArroTable
    from pyproj import Transformer
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from lonboard import Map, H3HexagonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        infer_rows_per_chunk,
        math,
        mo,
        np,
        pa,
        traitlets,
        udf,
        xr,
    )


@app.cell
def _(anywidget, traitlets):
    class Status(anywidget.AnyWidget):
        """A one-line status readout the camera can write to.

        It has to be a widget rather than `mo.md`, because the only way to update marimo
        output is to re-run the cell that produced it, and the cell holding the map is
        downstream of any state the camera could write: re-running it rebuilds the Map and
        throws the view away. A widget trait syncs straight to the browser instead, so the
        camera can narrate what it is doing without the notebook re-running anything.

        anywidget, not `ipywidgets.HTML`: marimo does not render classic Jupyter widgets
        and puts a "please migrate this widget to anywidget" banner in their place.
        """

        _esm = """
        function render({ model, el }) {
          const line = document.createElement("div");
          line.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const draw = () => { line.innerHTML = model.get("value"); };
          draw();
          model.on("change:value", draw);
          el.appendChild(line);
        }
        export default { render };
        """
        value = traitlets.Unicode("").tag(sync=True)

    return (Status,)


@app.cell
def _(math):
    PREFIX = "kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic"
    NODATA = 250

    # Which overview each H3 resolution reads. The source pyramid is 30 m native and
    # doubles: L0 30 m, L1 60, L2 120, L3 240, L4 480, L5 960.
    #
    # Picked so the mode has enough pixels under it to mean something. px/hex, measured:
    #   res 5  L5   277  ·  res 6  L4  157  ·  res 7  L4  22  ·  res 8  L3  12.5
    #   res 9  L2   7.1  ·  res 10 L1  4.1  ·  res 11 L0  2.3
    #
    # res 11 against 30 m imagery is 2.3 pixels per hexagon, and that is the floor: res 12
    # would be 0.6 and the map would hole out. This is where the data stops, not where the
    # code does.
    LEVEL_FOR_RES = {5: 5, 6: 4, 7: 4, 8: 3, 9: 2, 10: 1, 11: 0}
    MAX_RES = 11

    # The map's pixel size, assumed. It only sets how much of the world the viewport box
    # covers, and PAD is deliberately loose, so being wrong by a few hundred pixels costs
    # a slightly larger query and nothing else.
    VIEW_W, VIEW_H = 1400, 620

    # Fold a box larger than the screen, so a small pan lands inside what is already
    # folded and needs no query at all. This is squared into area, so 1.35 already means
    # folding 1.8x what you can see; 1.8 would mean 3.2x.
    PAD = 1.35

    # Seconds of camera quiet before a fold starts. Every fold is now an object-store read,
    # so rapid back-and-forth should read once at the end, not at every position it passed
    # through. Set to 0 to fold on every event.
    SETTLE = 0.25

    # One H3 resolution per 1.4 zoom levels, because each H3 step is 2.65x linear and
    # log2(2.65) = 1.4. That is what makes the hexagon a constant size ON SCREEN, and a
    # constant cell count falls out of it for free. Bands picked by eye instead are what
    # put 500k cells on screen at one zoom and 3k at the next.
    #
    # BASE_RES at ZOOM0, then finer every PER_RES. BASE_RES 7 puts ~215k cells on screen
    # at every band start (measured 216,896 / 211,810 / 213,219 / 214,909 / 215,932 for
    # res 7, 8, 9, 10, 11); BASE_RES 6 is one step coarser and gives ~31k.
    #
    # math.floor, NOT int(): int truncates toward zero, so every zoom below ZOOM0 would
    # collapse onto BASE_RES and the map would jump 4,626 -> 216,896 cells across a single
    # zoom step at ZOOM0. Flooring lets the ramp continue downward to MIN_RES instead.
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 7
    MIN_RES = 5

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # 16 NLCD classes in 7 groups on a teal-to-brown axis, water blue, developed carried
    # by luminance. NLCD's own palette is green forest against red developed, which is the
    # one pairing that carries nothing for a deuteranope, so it is never drawn.
    GROUPS = {
        11: ("Water", (8, 48, 107)),
        12: ("Ice", (158, 202, 225)),
        21: ("Developed, open", (215, 215, 215)),
        22: ("Developed, low", (160, 160, 160)),
        23: ("Developed, medium", (99, 99, 99)),
        24: ("Developed, high", (37, 37, 37)),
        31: ("Barren", (222, 217, 204)),
        41: ("Deciduous forest", (1, 102, 94)),
        42: ("Evergreen forest", (0, 60, 48)),
        43: ("Mixed forest", (53, 151, 143)),
        52: ("Shrub", (128, 205, 193)),
        71: ("Herbaceous", (199, 234, 229)),
        81: ("Pasture", (223, 194, 125)),
        82: ("Cropland", (191, 129, 45)),
        90: ("Woody wetland", (67, 147, 195)),
        95: ("Herbaceous wetland", (146, 197, 222)),
    }
    # One year, pinned. The slider is out until the camera path is proven: a year change
    # is a fresh read of the whole country, and there is no point putting that behind a
    # control while the thing it feeds is still the open question.
    YEAR = 2024
    return (
        GROUPS,
        LEVEL_FOR_RES,
        NODATA,
        PAD,
        PREFIX,
        SETTLE,
        VIEW_H,
        VIEW_W,
        YEAR,
        res_for_zoom,
    )


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs every
    # downstream cell, and the cell that owns the Map is one of them, so the Map is rebuilt
    # with its opening view_state and the camera snaps back to the middle of the country.
    # A plain dict is invisible to the dataflow graph, so the camera can drive the render
    # without the notebook re-running anything.
    HOLD = {
        "fold": None,  # SQL fold for the loaded year, set by the read cell
        "to_albers": None,  # lon/lat box -> the raster's own CRS, set by the read cell
        "extent": None,  # the raster's Albers bounds, to clamp against
        "res": None,  # H3 resolution currently on screen
        "box": None,  # padded Albers box the current hexes cover
        "busy": False,  # a fold is running
        "pending": None,  # newest camera position seen while busy, folded next
        "jumps": 0,  # camera moves too big to be a drag; see _note_jump
        "source": "",  # year on screen, for the status line
        "loop": None,  # kernel event loop, for scheduling from a non-loop thread
        "task": None,  # strong ref to the in-flight fold; asyncio only holds a weak one
    }
    return (HOLD,)


@app.cell
def _(ArroTable, GROUPS, coordinates_to_cells, np, pa):
    _lut = np.full((256, 3), 120, dtype=np.uint8)
    for _c, (_lbl, _rgb) in GROUPS.items():
        _lut[_c] = _rgb
    _names = np.array([GROUPS.get(i, ("", None))[0] for i in range(256)], dtype=object)

    def to_layer_table(tbl):
        # combine_chunks because DataFusion returns many chunks and the numpy-derived
        # columns are one; lonboard rejects a table whose columns disagree about chunking.
        # ArroTable because the layer's `table` trait coerces in __init__ but its
        # validate() is a strict isinstance check, so assignment afterwards needs the
        # real type.
        tbl = tbl.combine_chunks()
        cls = np.asarray(tbl["mode_cls"])
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(_lut[cls].ravel()), 3
                    ),
                    "class": pa.array(list(_names[cls])),
                    "purity": tbl["purity"],
                    "pixels": tbl["px_total"],
                }
            )
        )

    def seed_table():
        # One hexagon at null island, in the basemap's own dark, so the Map has a valid
        # table at build time. The first camera event replaces it. This is what lets the
        # Map cell depend on nothing: it does not have to wait for the raster read.
        hexes = coordinates_to_cells(np.array([0.0]), np.array([0.0]), 5)
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": pa.array(hexes),
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([13, 17, 23], dtype=np.uint8)), 3
                    ),
                    "class": pa.array([""]),
                    "purity": pa.array([0.0]),
                    "pixels": pa.array([0], type=pa.int64()),
                }
            )
        )

    return seed_table, to_layer_table


@app.cell
def _(
    BitmapTileLayer,
    CartoBasemap,
    H3HexagonLayer,
    HOLD,
    Map,
    MaplibreBasemap,
    PAD,
    SETTLE,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    infer_rows_per_chunk,
    math,
    res_for_zoom,
    seed_table,
    to_layer_table,
):
    # Built exactly once. This cell depends on no control and on no state the camera can
    # write, so nothing in the notebook can re-run it and throw the view away. Everything
    # after this happens by trait assignment on `layer`, which lonboard treats as
    # independent of `view_state`.
    status = Status(value="<b>loading…</b>")

    _seed = seed_table()
    layer = H3HexagonLayer(
        table=_seed,
        get_hexagon=_seed["hex"],
        get_fill_color=_seed["color"],
        extruded=False,
        stroked=False,
        high_precision=True,
        coverage=0.9,
        opacity=0.7,
        pickable=True,
    )
    # Positron's labels, on their own, drawn OVER the hexes. The basemap paints under
    # every deck layer, so place names put on it would sit beneath an opaque hexagon and
    # be lost; as a deck layer above the fill they read on top of it. Carto serves the
    # `positron-labels-only` style as the `light_only_labels` raster set.
    #
    # pickable=False so it never intercepts a hover meant for the cell underneath, and
    # @2x with tile_size 512 because the default 256 would sample retina tiles at half
    # scale and the type would blur.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.9,
        pickable=False,
    )
    deck = Map(
        [layer, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.PositronNoLabels),
        view_state={"longitude": -98.5, "latitude": 39.5, "zoom": 3.8},
        height=VIEW_H,
        # Hover to inspect. show_tooltip defaults to False, which leaves show_side_panel
        # (click) as the only way into a cell's class and purity.
        show_tooltip=True,
    )

    def _lat_to_y(lat):
        s = math.sin(math.radians(max(min(lat, 85.0), -85.0)))
        return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)

    def _y_to_lat(y):
        return math.degrees(2 * math.atan(math.exp((0.5 - y) * 2 * math.pi)) - math.pi / 2)

    def view_to_bbox(vs):
        world = 512 * (2**vs.zoom)
        half_lon = 360.0 * VIEW_W / world / 2
        yc, half_y = _lat_to_y(vs.latitude), VIEW_H / world / 2
        return (
            vs.longitude - half_lon,
            _y_to_lat(yc + half_y),
            vs.longitude + half_lon,
            _y_to_lat(yc - half_y),
        )

    def _pad(b):
        dx, dy = (b[2] - b[0]) * (PAD - 1) / 2, (b[3] - b[1]) * (PAD - 1) / 2
        return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)

    def _covers(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    async def _draw(vs, force):
        """One read + fold, or nothing if this view is already covered."""
        res = res_for_zoom(vs.zoom)
        want = HOLD["to_albers"](_pad(view_to_bbox(vs)))
        if (
            not force
            and res == HOLD["res"]
            and HOLD["box"]
            and _covers(HOLD["box"], HOLD["to_albers"](view_to_bbox(vs)))
        ):
            return
        status.value = f"<b>reading…</b> res {res}"
        raw, m_px, read_px = await HOLD["fold"](res, want)
        if raw is None or raw.num_rows == 0:
            status.value = f"<b>res {res}</b> · nothing here · zoom {vs.zoom:.1f}"
            HOLD["res"], HOLD["box"] = res, want
            return
        tbl = to_layer_table(raw)

        # RECOMPUTE THIS BEFORE EVERY ASSIGNMENT. lonboard infers _rows_per_chunk in
        # __init__ ONLY (layer/_base.py:397) and never again, but every later assignment
        # still rechunks through it (traits/_table.py:106, _h3.py:130, _color.py:140) and
        # writes ONE PARQUET FILE PER CHUNK. Built against a 1-row seed table,
        # infer_rows_per_chunk returns 1, that 1 is latched for the life of the layer, and
        # each fold then serialises one Parquet file PER HEXAGON. Measured over the four
        # folds of a zoom-in: 621.94 MB in 673,581 Parquet files, against 6.89 MB in 12
        # with this line. That is the whole difference between a live map and a machine
        # that has to be restarted.
        layer._rows_per_chunk = infer_rows_per_chunk(tbl)

        # Held together so deck sees one update rather than a frame with a new table
        # against the old hexagon column.
        with layer.hold_trait_notifications():
            layer.table = tbl
            layer.get_hexagon = tbl["hex"]
            layer.get_fill_color = tbl["color"]
        HOLD["res"], HOLD["box"] = res, want
        status.value = (
            f"<b>res {res}</b> · {tbl.num_rows:,} cells · {m_px:.0f} m"
            f" · {read_px / 1e6:.2f}M px read · zoom {vs.zoom:.1f} · {HOLD['source']}"
            + (f" · <b style='color:#E69F00'>{HOLD['jumps']} jumps</b>" if HOLD["jumps"] else "")
        )

    async def refresh(vs, force=False):
        """Fold what the camera is looking at, once it has stopped moving.

        `view_state` fires on every frame of a drag and each fold now costs an object-store
        read, so this does two things. SETTLE debounces: rapid back-and-forth parks the
        newest view and waits for quiet, so a drag reads once at the end instead of at
        every position it passed through. Coalescing then makes sure that whatever piled
        up while a read was in flight collapses to the NEWEST view rather than becoming its
        own fold each: without it a two-second drag queues a hundred folds of stale
        viewports and never catches up. No threads and no timers; the debounce is an await
        on the kernel's own loop, so the map keeps rendering throughout.
        """
        if HOLD["fold"] is None:
            return
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                # Wait for a full SETTLE with no new camera event, taking the newest each
                # time round. force skips it: that is the opening draw, nothing to settle.
                if not force:
                    while SETTLE > 0:
                        await asyncio.sleep(SETTLE)
                        if HOLD["pending"] is None:
                            break
                        vs, HOLD["pending"] = HOLD["pending"], None
                await _draw(vs, force)
                vs, force = HOLD["pending"], False
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as exc:
            # A failure inside a comm handler is otherwise silent.
            status.value = f"<b style='color:#F0E442'>failed:</b> {type(exc).__name__}: {exc}"
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _note_jump(old, new):
        """Catch the camera moving somewhere it was not dragged.

        Instrumentation, not a fix. A drag arrives as many small deltas; a reset or a
        flyTo arrives as ONE big one. Recording that, with where it went, is what
        separates the candidates: a jump landing exactly on the opening view means a
        marimo cell re-ran and rebuilt the Map, anything else means the camera was moved
        from the JS side.
        """
        if old is None or new is None:
            return
        try:
            span = 360.0 * VIEW_W / (512 * 2**old.zoom)
            dz = abs(new.zoom - old.zoom)
            dx = abs(new.longitude - old.longitude) / max(span, 1e-9)
            dy = abs(new.latitude - old.latitude) / max(span, 1e-9)
        except AttributeError:
            return
        if dz > 0.75 or dx > 0.75 or dy > 0.75:
            HOLD["jumps"] += 1
            opening = (
                abs(new.longitude + 98.5) < 0.01
                and abs(new.latitude - 39.5) < 0.01
                and abs(new.zoom - 3.8) < 0.01
            )
            status.value = (
                f"<b style='color:#E69F00'>camera jumped ({HOLD['jumps']}):</b> "
                f"z{old.zoom:.2f}&rarr;{new.zoom:.2f} "
                f"({old.longitude:.3f}, {old.latitude:.3f})&rarr;"
                f"({new.longitude:.3f}, {new.latitude:.3f})"
                + (" <b>= the opening view, so a cell re-ran</b>" if opening else "")
            )

    def _on_camera(change):
        # The observer is sync and the fold is not, so the work is handed to the kernel's
        # event loop. HOLD["task"] keeps a reference: asyncio holds only a weak one, and a
        # bare create_task can be collected mid-flight.
        vs = change["new"]
        _note_jump(change["old"], vs)
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        try:
            # Normal case: the comm handler is already running on the loop.
            HOLD["task"] = asyncio.get_running_loop().create_task(refresh(vs))
        except RuntimeError:
            # Called from some other thread. Needs the loop captured at read time.
            loop = HOLD.get("loop")
            if loop is not None:
                HOLD["task"] = asyncio.run_coroutine_threadsafe(refresh(vs), loop)

    deck.observe(_on_camera, names="view_state")
    return deck, refresh, status


@app.cell
async def _(
    GeoTIFF,
    HOLD,
    LEVEL_FOR_RES,
    NODATA,
    PREFIX,
    S3Store,
    Transformer,
    Window,
    XarrayContext,
    YEAR,
    asyncio,
    coordinates_to_cells,
    deck,
    math,
    np,
    pa,
    refresh,
    udf,
    xr,
):
    # No whole-country read any more. Opening the COG reads headers only; each fold then
    # pulls JUST the padded viewport, from the overview that matches the H3 resolution it
    # is about to build. That inversion is what buys res 11: the viewport shrinks faster
    # than the resolution grows, so the finest reads are the SMALLEST. Measured, band by
    # band: 16.2M pixels at res 5 (whole country, 960 m) down to 72,890 at res 11 (30 m).
    _store = S3Store(
        "us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True
    )
    _g = await GeoTIFF.open(
        f"{PREFIX}/Annual_NLCD_LndCov_{YEAR}_CU_C1V1.tif", store=_store
    )
    _levels = [_g, *_g.overviews]
    _inv = Transformer.from_crs(_g.crs, "EPSG:4326", always_xy=True)
    _fwd = Transformer.from_crs("EPSG:4326", _g.crs, always_xy=True)
    _l, _b, _r, _t = _g.bounds

    def _bilinear(grid, gy, gx, rr, cc):
        _c = len(gy)
        fy = np.interp(rr, gy, np.arange(_c))
        fx = np.interp(cc, gx, np.arange(_c))
        y0 = np.clip(fy.astype(np.int32), 0, _c - 2)
        x0 = np.clip(fx.astype(np.int32), 0, _c - 2)
        dy, dx = fy - y0, fx - x0
        return (
            grid[y0, x0] * (1 - dy) * (1 - dx)
            + grid[y0 + 1, x0] * dy * (1 - dx)
            + grid[y0, x0 + 1] * (1 - dy) * dx
            + grid[y0 + 1, x0 + 1] * dy * dx
        )

    def _to_deg(grid, gy, gx, wt, wl, px_y, px_x):
        # y, x are Albers metres, the window's own dims. This is what lets the SQL ask for
        # degrees without anything being flattened in Python first. Albers over CONUS is
        # smooth, so exact pyproj on a 64x64 control grid plus bilinear interpolation lands
        # within ~100 m of a per-pixel transform, for 4096 pyproj calls instead of millions.
        def f(yv, xv):
            rr = (wt - yv.to_numpy()) / px_y - 0.5
            cc = (xv.to_numpy() - wl) / px_x - 0.5
            return pa.array(_bilinear(grid, gy, gx, rr, cc))

        return f

    def _to_albers(bbox):
        # Albers is curved, so the corners alone understate the box. Walk the edges and
        # take the extremes, then clamp to the raster: a viewport over the ocean must not
        # ask for coordinates the array does not have.
        lo0, la0, lo1, la1 = bbox
        _n = 9
        _e = np.linspace(0, 1, _n)
        lons = np.concatenate(
            [
                lo0 + _e * (lo1 - lo0),
                lo0 + _e * (lo1 - lo0),
                np.full(_n, lo0),
                np.full(_n, lo1),
            ]
        )
        lats = np.concatenate(
            [
                np.full(_n, la0),
                np.full(_n, la1),
                la0 + _e * (la1 - la0),
                la0 + _e * (la1 - la0),
            ]
        )
        xs, ys = _fwd.transform(lons, np.clip(lats, -89.0, 89.0))
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not ok.any():
            return (_l, _b, _r, _t)
        return (
            max(_l, float(xs[ok].min())),
            max(_b, float(ys[ok].min())),
            min(_r, float(xs[ok].max())),
            min(_t, float(ys[ok].max())),
        )

    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(
                coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())
            ),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )

    async def fold(res, box):
        """Read the window for `box` at the overview `res` deserves, then fold it to H3.

        Everything here is per-view: the read, the control grid, the to_lat/to_lon UDFs and
        the registered table. Registering under one fixed name means DataFusion holds one
        window at a time, which is why RSS sits flat at 0.68 GB whatever the zoom.
        """
        rd = _levels[LEVEL_FOR_RES[res]]
        L, B, R, T = rd.bounds
        H, W = rd.shape
        px_x, px_y = (R - L) / W, (T - B) / H

        # Albers box -> whole pixel window, clamped to the raster.
        x0, y0, x1, y1 = box
        col0 = max(0, int((max(x0, L) - L) / px_x))
        col1 = min(W, int(math.ceil((min(x1, R) - L) / px_x)))
        row0 = max(0, int((T - min(y1, T)) / px_y))
        row1 = min(H, int(math.ceil((T - max(y0, B)) / px_y)))
        wpx, hpx = col1 - col0, row1 - row0
        if wpx <= 0 or hpx <= 0:
            return None, rd.res[0], 0

        arr = np.asarray(
            (
                await rd.read(
                    window=Window(col_off=col0, row_off=row0, width=wpx, height=hpx)
                )
            ).as_masked()[0]
        )

        # The window's own corner, not the raster's. Everything downstream is relative to it.
        wl, wt = L + col0 * px_x, T - row0 * px_y
        _c = 64
        gy, gx = np.linspace(0, hpx - 1, _c), np.linspace(0, wpx - 1, _c)
        X, Y = np.meshgrid(wl + (gx + 0.5) * px_x, wt - (gy + 0.5) * px_y)
        glo, gla = (a.reshape(_c, _c) for a in _inv.transform(X.ravel(), Y.ravel()))

        for _name, _grid in (("to_lat", gla), ("to_lon", glo)):
            ctx.register_udf(
                udf(
                    _to_deg(_grid, gy, gx, wt, wl, px_y, px_x),
                    [pa.float64(), pa.float64()],
                    pa.float64(),
                    "stable",
                    name=_name,
                )
            )
        try:
            ctx.deregister_table("lc")
        except Exception:
            pass
        ctx.from_dataset(
            "lc",
            xr.Dataset(
                {"cls": (("y", "x"), arr)},
                coords={
                    "y": wt - (np.arange(hpx) + 0.5) * px_y,
                    "x": wl + (np.arange(wpx) + 0.5) * px_x,
                },
            ),
            chunks={"y": 512},
        )

        # Mode per cell. The `cls ASC` tie-break is not decoration: without it a cell whose
        # top two classes have equal counts picks a different winner run to run.
        out = ctx.sql(f"""
            WITH counts AS (
                SELECT h3_latlng_to_cell(to_lat(y, x), to_lon(y, x), CAST({res} AS INT))
                           AS hex,
                       cls, count(*) AS n
                FROM lc WHERE cls != {NODATA}
                GROUP BY 1, 2
            )
            SELECT hex,
                   first_value(cls ORDER BY n DESC, cls ASC) AS mode_cls,
                   sum(n) AS px_total,
                   CAST(max(n) AS DOUBLE) / sum(n) AS purity
            FROM counts GROUP BY hex
        """).to_arrow_table()
        return out, rd.res[0], wpx * hpx

    # Hand the fold to the camera's world and draw where the camera already is.
    HOLD["fold"] = fold
    HOLD["to_albers"] = _to_albers
    HOLD["extent"] = (_l, _b, _r, _t)
    HOLD["source"] = str(YEAR)
    HOLD["loop"] = asyncio.get_running_loop()
    await refresh(deck.view_state, force=True)
    return


@app.cell
def _(GROUPS, deck, mo, status):
    _seen, _sw = set(), []
    for _c, (_lbl, (_r, _g, _b)) in GROUPS.items():
        if _lbl in _seen:
            continue
        _seen.add(_lbl)
        _sw.append(
            f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
            f'margin-right:.8rem;white-space:nowrap">'
            f'<span style="width:.8rem;height:.8rem;border-radius:2px;'
            f'background:rgb({_r},{_g},{_b});outline:1px solid #8888"></span>{_lbl}</span>'
        )

    mo.vstack(
        [
            status,
            deck,
            mo.md(
                "<div style='display:flex;flex-wrap:wrap;font-size:.8rem;line-height:1.7'>"
                + "".join(_sw)
                + "</div>\n\nColour is the majority class in the cell. Hover for its "
                "**purity**: how much of the cell is actually that class."
            ),
        ],
        gap=0.4,
    )
    return


if __name__ == "__main__":
    app.run()
