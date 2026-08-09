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
    from arro3.core import Array as ArroArray, Table as ArroTable
    from pyproj import Transformer
    from obstore.store import S3Store
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy import grid_disk
    from h3ronpy.vector import cells_to_wkb_polygons, coordinates_to_cells
    from geoarrow.rust.core import from_wkb, multipolygon
    from lonboard import Map, H3HexagonLayer, BitmapTileLayer, PolygonLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroArray,
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        cells_to_wkb_polygons,
        coordinates_to_cells,
        from_wkb,
        grid_disk,
        infer_rows_per_chunk,
        math,
        mo,
        multipolygon,
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
def _(anywidget, traitlets):
    class Controls(anywidget.AnyWidget):
        """Layer switches, in the flow UNDER the map, next to the legend.

        Same constraint as Status, for the same reason: an `mo.ui.checkbox` would make the
        map cell depend on it, and every click would rebuild the Map and reset the camera.
        A widget trait syncs to the kernel, a Python observer assigns straight onto the
        deck layers, and nothing re-runs.

        One wrapping row, so it reads as a strip under the legend rather than a stack, and
        nothing is positioned: it takes its own space in the layout like any other output.
        """

        _esm = """
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.4rem 1.2rem;" +
            "font:12.5px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.35rem 0 .1rem;user-select:none";

          const check = (key, label) => {
            const l = document.createElement("label");
            l.style.cssText =
              "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = model.get(key);
            cb.style.cssText = "margin:0;cursor:pointer";
            cb.addEventListener("change", () => {
              model.set(key, cb.checked);
              model.save_changes();
            });
            model.on("change:" + key, () => { cb.checked = model.get(key); });
            l.appendChild(cb);
            l.appendChild(document.createTextNode(label));
            return l;
          };

          const slider = (key, label) => {
            const w = document.createElement("span");
            w.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const cap = document.createElement("span");
            const draw = () => {
              cap.textContent = label + " " + Number(model.get(key)).toFixed(2);
            };
            cap.style.cssText = "opacity:.7;white-space:nowrap";
            const s = document.createElement("input");
            s.type = "range";
            s.min = "0"; s.max = "1"; s.step = "0.05";
            s.value = model.get(key);
            s.style.cssText = "width:7rem;margin:0;cursor:pointer";
            s.addEventListener("input", () => {
              model.set(key, parseFloat(s.value));
              model.save_changes();
            });
            model.on("change:" + key, () => { s.value = model.get(key); draw(); });
            draw();
            w.appendChild(cap);
            w.appendChild(s);
            return w;
          };

          // Integer slider, and it fires on CHANGE not INPUT: each step re-dissolves
          // the wash, which is ~1 s of work, so it runs when the handle is released.
          const steps = (key, label, lo, hi) => {
            const w = document.createElement("span");
            w.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const cap = document.createElement("span");
            cap.style.cssText = "opacity:.7;white-space:nowrap";
            const draw = () => { cap.textContent = label + " " + model.get(key); };
            const s = document.createElement("input");
            s.type = "range";
            s.min = String(lo); s.max = String(hi); s.step = "1";
            s.value = model.get(key);
            s.style.cssText = "width:7rem;margin:0;cursor:pointer";
            s.addEventListener("input", () => { cap.textContent = label + " " + s.value; });
            s.addEventListener("change", () => {
              model.set(key, parseInt(s.value, 10));
              model.save_changes();
            });
            model.on("change:" + key, () => { s.value = model.get(key); draw(); });
            draw();
            w.appendChild(cap);
            w.appendChild(s);
            return w;
          };

          const head = document.createElement("span");
          head.textContent = "layers";
          head.style.cssText =
            "font:11px ui-monospace,Menlo,monospace;letter-spacing:.06em;" +
            "text-transform:uppercase;opacity:.55";
          box.appendChild(head);
          box.appendChild(check("cells", "H3 cells"));
          box.appendChild(check("cluster_fill", "Cluster fill"));
          box.appendChild(check("cluster_line", "Cluster outline"));
          box.appendChild(slider("cell_opacity", "cell opacity"));
          box.appendChild(slider("cell_coverage", "cell coverage"));
          box.appendChild(slider("cluster_opacity", "cluster opacity"));
          box.appendChild(steps("min_cluster", "min cluster", 1, 300));
          el.appendChild(box);
        }
        export default { render };
        """
        cells = traitlets.Bool(True).tag(sync=True)
        cluster_fill = traitlets.Bool(True).tag(sync=True)
        cluster_line = traitlets.Bool(True).tag(sync=True)
        cell_opacity = traitlets.Float(0.7).tag(sync=True)
        # Hexagon size as a fraction of the cell. 1.0 is edge to edge; below about
        # 0.85 the gaps open up and the basemap reads through the lattice.
        cell_coverage = traitlets.Float(0.9).tag(sync=True)
        cluster_opacity = traitlets.Float(1.0).tag(sync=True)
        # Smallest run of touching like cells that earns a dissolved polygon. THIS is what
        # decides whether the fill-only view covers the map or leaves it nearly empty: at
        # 50, a whole-country fold produced 51 polygons, so turning the hexes off left
        # almost nothing on screen. Low values cover more and cost more, steeply.
        min_cluster = traitlets.Int(20).tag(sync=True)

    return (Controls,)


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

    # CLUSTER WASH. A run of touching cells of the same class, dissolved into one polygon
    # with cells_to_wkb_polygons(link_cells=True) and laid over the hexes as a faint tint in
    # the class's own colour. Both of these are meant to be turned.
    #
    # MIN_CLUSTER is the one that matters. Land cover is mostly speckle, so dissolving
    # everything gives thousands of scraps: at res 8 over a full viewport, measured,
    #   min run     1 ->  43,329 polygons  13.2 MB  22.5 s   (unusable)
    #   min run    20 ->   1,153 polygons   5.2 MB   1.9 s
    #   min run   100 ->     181 polygons   3.6 MB   1.3 s
    #   min run   500 ->      34 polygons   2.5 MB   1.0 s
    #   min run  2000 ->       8 polygons   1.8 MB   0.7 s
    # The polygon count collapses 1,200x while the cells covered only halve, because nearly
    # all of those runs are a handful of cells. 500 leaves the genuinely large regions.
    MIN_CLUSTER = 50
    CLUSTER_OPACITY = 1.0
    CLUSTER_WIDTH = 2  # stroke width in screen pixels
    # 1.0 is the class colour exactly. Lower values darken the edge; below about 0.6 every
    # class collapses toward black and the outlines stop telling each other apart.
    CLUSTER_DARKEN = 2.5

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

    # NLCD'S OWN COLORMAP, read out of the COG itself (`GeoTIFF.colormap.as_dict()`) and
    # written down here so the legend and the fill cannot drift apart.
    #
    # The palette this replaced was invented here, on a teal-to-brown axis, with the three
    # forest classes separated by lightness. That put DECIDUOUS FOREST at luminance 0.103
    # and evergreen at 0.034, and deciduous forest is 39.7% of the cells over the
    # southeast: 52% of the map came out below 0.18 luminance, so the map read as black
    # with everything else apparently outlined in it. NLCD puts deciduous at 0.329.
    # Lightness was the whole problem; the hues were never the point.
    GROUPS = {
        11: ("Water", (70, 107, 159)),
        12: ("Ice/Snow", (209, 222, 248)),
        21: ("Developed, open", (222, 197, 197)),
        22: ("Developed, low", (217, 146, 130)),
        23: ("Developed, medium", (235, 0, 0)),
        24: ("Developed, high", (171, 0, 0)),
        31: ("Barren", (179, 172, 159)),
        41: ("Deciduous forest", (104, 171, 95)),
        42: ("Evergreen forest", (28, 95, 44)),
        43: ("Mixed forest", (181, 197, 143)),
        52: ("Shrub", (204, 184, 121)),
        71: ("Herbaceous", (223, 223, 194)),
        81: ("Pasture/Hay", (220, 217, 57)),
        82: ("Cultivated crops", (171, 108, 40)),
        90: ("Woody wetland", (184, 217, 235)),
        95: ("Herbaceous wetland", (108, 159, 184)),
    }
    # One year, pinned. The slider is out until the camera path is proven: a year change
    # is a fresh read of the whole country, and there is no point putting that behind a
    # control while the thing it feeds is still the open question.
    YEAR = 2024
    return (
        CLUSTER_DARKEN,
        CLUSTER_OPACITY,
        CLUSTER_WIDTH,
        GROUPS,
        LEVEL_FOR_RES,
        MIN_CLUSTER,
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
        "cache": {},  # res -> (box, table): a zoom back out to a level already folded
        "busy": False,  # a fold is running
        "pending": None,  # newest camera position seen while busy, folded next
        "jumps": 0,  # camera moves too big to be a drag; see _note_jump
        "source": "",  # year on screen, for the status line
        "loop": None,  # kernel event loop, for scheduling from a non-loop thread
        "task": None,  # strong ref to the in-flight fold; asyncio only holds a weak one
    }
    return (HOLD,)


@app.cell
def _(
    ArroArray,
    ArroTable,
    GROUPS,
    cells_to_wkb_polygons,
    coordinates_to_cells,
    from_wkb,
    grid_disk,
    multipolygon,
    np,
    pa,
):
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

    def to_cluster_table(tbl, min_cluster, darken=1.0):
        """Runs of touching like cells, dissolved into one polygon each.

        `cells_to_wkb_polygons(..., link_cells=True)` does the dissolve AND the connected
        -component split in one call: neighbours merge, disconnected groups come back as
        separate polygons. The union-find here is only to get run SIZES, so the speckle can
        be dropped before dissolving. That ordering is the whole performance story: 22.5 s
        and 43,329 polygons if everything is dissolved, 1.0 s and 34 at min run 500.

        Returned as arro3 arrays, not pyarrow: pa.array() strips the geoarrow extension
        metadata off the geometry and lonboard's table trait rejects it as "expected
        geometry column in table".
        """
        hx = np.asarray(tbl["hex"])
        cs = np.asarray(tbl["mode_cls"])
        n = len(hx)
        if n == 0:
            return None

        # k=1 ring -> adjacency between cells that are BOTH present and the same class.
        flat = np.asarray(grid_disk(hx, 1, flatten=True))
        per = len(flat) // n
        src = np.repeat(np.arange(n), per)
        order = np.argsort(hx)
        pos = np.clip(np.searchsorted(hx[order], flat), 0, n - 1)
        dst = order[pos]
        keep = hx[dst] == flat
        src, dst = src[keep], dst[keep]
        keep = cs[src] == cs[dst]
        src, dst = src[keep], dst[keep]
        keep = src < dst  # grid_disk emits each adjacency twice; one direction is enough
        src, dst = src[keep], dst[keep]

        parent = np.arange(n)

        def _find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]  # path halving
                a = parent[a]
            return a

        for a, b in zip(src.tolist(), dst.tolist()):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb
        roots = np.fromiter((_find(i) for i in range(n)), np.int64, n)
        _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)

        big = counts[inv] >= min_cluster
        if not big.any():
            return None

        wkbs, colors = [], []
        for c in np.unique(cs[big]):
            sub = hx[big & (cs == c)]
            w = pa.array(cells_to_wkb_polygons(sub, link_cells=True))
            wkbs.append(w)
            colors.append(np.tile(_lut[c], (len(w), 1)))
            # darkened edge, off for now:
            # colors.append(np.tile((_lut[c] * darken).astype(np.uint8), (len(w), 1)))
        # to_type is required, not cosmetic: from_wkb alone yields the generic
        # `geoarrow.geometry` union and lonboard rejects it with "Expected one of
        # geoarrow.polygon, geoarrow.multipolygon". A single polygon happens to downcast
        # on its own, which is why a one-geometry test passed and the real data did not.
        # The crs also stops lonboard warning that it cannot tell whether this is WGS84.
        geom = from_wkb(
            pa.concat_arrays(wkbs), to_type=multipolygon("xy", crs="EPSG:4326")
        )
        rgb = np.concatenate(colors).astype(np.uint8)
        return ArroTable.from_arrays(
            [
                ArroArray.from_arrow(geom),
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(rgb.ravel()), 3)
                ),
            ],
            names=["geometry", "color"],
        )

    def seed_cluster():
        """Two touching cells so the PolygonLayer can be built before any fold has run.
        min_cluster=0 dissolves them into one off-screen polygon; the layer starts
        visible=False and is switched on when a real fold produces some."""
        cells = np.asarray(
            coordinates_to_cells(np.array([0.0, 0.0]), np.array([0.0, 0.0001]), 5)
        )
        return pa.table(
            {"hex": pa.array(np.unique(cells)), "mode_cls": pa.array([41] * len(np.unique(cells)))}
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

    return seed_cluster, seed_table, to_cluster_table, to_layer_table


@app.cell
def _(
    BitmapTileLayer,
    CLUSTER_DARKEN,
    CLUSTER_OPACITY,
    CLUSTER_WIDTH,
    CartoBasemap,
    Controls,
    H3HexagonLayer,
    HOLD,
    MIN_CLUSTER,
    Map,
    MaplibreBasemap,
    PAD,
    PolygonLayer,
    SETTLE,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    infer_rows_per_chunk,
    math,
    res_for_zoom,
    seed_cluster,
    seed_table,
    to_cluster_table,
    to_layer_table,
):
    # Built exactly once. This cell depends on no control and on no state the camera can
    # write, so nothing in the notebook can re-run it and throw the view away. Everything
    # after this happens by trait assignment on `layer`, which lonboard treats as
    # independent of `view_state`.
    status = Status(value="<b>loading…</b>")

    _seed = seed_table()
    h3_layer = H3HexagonLayer(
        table=_seed,
        get_hexagon=_seed["hex"],
        get_fill_color=_seed["color"],
        # get_line_color=_seed["color"],
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
    # The cluster wash: one dissolved polygon per large run of like cells, filled in that
    # class's own colour at CLUSTER_OPACITY over the hexes. Starts invisible and is switched
    # on the first time a fold produces polygons, so it never has to hold a placeholder
    # geometry. pickable=False so it never intercepts a hover meant for a cell.
    # ONE OUTLINE PER CLUSTER, not per cell. The earlier attempt stroked every rim CELL and
    # came out a honeycomb over the whole map; this strokes the DISSOLVED boundary, so a run
    # of 40,000 cells is a single line. Fill starts OFF because the polygon is exactly the
    # cells beneath it, so with the hexes up a fill in the same colour adds nothing; turn
    # the hexes off in the panel and the fill becomes the map: solid regions, no honeycomb.
    _cseed = to_cluster_table(seed_cluster(), 0, CLUSTER_DARKEN)
    clusters = PolygonLayer(
        table=_cseed,
        get_fill_color=_cseed["color"],
        get_line_color=_cseed["color"],
        filled=True,
        stroked=True,
        line_width_min_pixels=CLUSTER_WIDTH,
        opacity=CLUSTER_OPACITY,
        pickable=False,
        visible=False,
    )

    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.9,
        pickable=False,
    )
    deck = Map(
        [
        h3_layer, 
        clusters, 
        labels
        ],
        basemap=MaplibreBasemap(style=CartoBasemap.PositronNoLabels),
        view_state={"longitude": -98.5, "latitude": 39.5, "zoom": 3.8},
        height=VIEW_H,
        # Hover to inspect. show_tooltip defaults to False, which leaves show_side_panel
        # (click) as the only way into a cell's class and purity.
        show_tooltip=True,
    )

    # THE SWITCHES. Every one of these is a trait assignment on a layer that already
    # exists, so a click repaints and nothing in the notebook re-runs.
    #
    # WANT is what the USER asked for; the fold code below owns `clusters.visible` on its
    # own schedule (outlines are hidden the moment the cells they describe go away) and has
    # to ask WANT before turning them back on, or a fold would undo the switch.
    controls = Controls()
    # "clusters": does the user want them. "built": does the layer currently hold polygons
    # dissolved from the cells that are on screen right now. Visible needs both.
    WANT = {"clusters": True, "built": False}

    def apply_controls():
        """Push every control onto the layers, once, at build time.

        THIS IS THE FIX FOR "the fill checkbox does nothing". The layer was constructed
        with filled=True while the checkbox defaulted to False, so the panel and the layer
        disagreed from the first frame: the box was empty, the fill was already on, and
        the first click set filled=True, which changed nothing anyone could see. Nothing
        keeps two hand-written defaults in step, so the widget is now the only place a
        default lives and the layer is told what it says.
        """
        h3_layer.visible = controls.cells
        h3_layer.opacity = controls.cell_opacity
        h3_layer.coverage = controls.cell_coverage
        clusters.filled = controls.cluster_fill
        clusters.stroked = controls.cluster_line
        clusters.opacity = controls.cluster_opacity
        WANT["clusters"] = controls.cluster_fill or controls.cluster_line

    apply_controls()

    def _on_controls(change):
        name, val = change["name"], change["new"]
        if name == "cells":
            h3_layer.visible = val
        elif name == "cell_opacity":
            h3_layer.opacity = val
        elif name == "cell_coverage":
            h3_layer.coverage = val
        elif name == "cluster_opacity":
            clusters.opacity = val
        elif name == "min_cluster":
            rewash()
        else:
            # Fill and outline are the two halves of the cluster layer. With both off there
            # is nothing left to draw, so the layer itself comes off and stays off until one
            # of them is asked for again.
            setattr(clusters, "filled" if name == "cluster_fill" else "stroked", val)
            WANT["clusters"] = controls.cluster_fill or controls.cluster_line
            clusters.visible = WANT["clusters"] and WANT["built"]

    controls.observe(
        _on_controls,
        names=[
            "cells",
            "cluster_fill",
            "cluster_line",
            "cell_opacity",
            "cell_coverage",
            "cluster_opacity",
            "min_cluster",
        ],
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

    def _show(tbl, res, box, note):
        """Put a table on the layer and record what it is. The ONLY place that paints."""
        # RECOMPUTE THIS BEFORE EVERY ASSIGNMENT. lonboard infers _rows_per_chunk in
        # __init__ ONLY (layer/_base.py:397) and never again, but every later assignment
        # still rechunks through it (traits/_table.py:106, _h3.py:130, _color.py:140) and
        # writes ONE PARQUET FILE PER CHUNK. Built against a 1-row seed table,
        # infer_rows_per_chunk returns 1, that 1 is latched for the life of the layer, and
        # each fold then serialises one Parquet file PER HEXAGON. Measured over four folds
        # of a zoom-in: 621.94 MB in 673,581 Parquet files, against 6.89 MB in 12 with this
        # line. That is the difference between a live map and a machine that gets restarted.
        # max(1, ...): infer_rows_per_chunk returns 0 for an empty table, and lonboard
        # asserts max_chunksize > 0 on the way out.
        h3_layer._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # THE BLACK-HEXAGON FIX, and it has to be hold_sync, NOT
        # hold_trait_notifications. hold_trait_notifications batches the traitlets
        # NOTIFICATIONS, but ipywidgets' notify_change calls send_state(key=name) as each
        # one fires, so the browser still received THREE separate comm messages: table,
        # then hexagons, then colours. Between message two and three deck holds the NEW
        # hexagon ids against the OLD, shorter colour buffer, and WebGL reads past the end
        # of a short attribute as zeros. Zero is opaque black. That is the flash of black
        # cells on a resolution change: not nodata, not the transform, just one rendered
        # frame with the colours missing. hold_sync defers the send itself and emits a
        # single message carrying all three, so no frame can see them disagree.
        with h3_layer.hold_sync():
            h3_layer.table = tbl
            h3_layer.get_hexagon = tbl["hex"]
            h3_layer.get_fill_color = tbl["color"]
        # THE OUTLINES DIE WITH THE CELLS THEY DESCRIBED. They are dissolved from one
        # fold's cells, so the moment different cells go up they are stale, and stale
        # outlines do not look stale: they are clean lines in the right colours sitting over
        # the wrong place. Hiding here means the only outlines ever visible are ones built
        # from the cells underneath them.
        clusters.visible = False
        WANT["built"] = False
        HOLD["res"], HOLD["box"] = res, box
        status.value = note

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or clear and refold."""
        res = res_for_zoom(vs.zoom)
        seen = HOLD["to_albers"](view_to_bbox(vs))
        want = HOLD["to_albers"](_pad(view_to_bbox(vs)))

        # Already correct for this view.
        if (
            not force
            and res == HOLD["res"]
            and HOLD["box"]
            and _covers(HOLD["box"], seen)
        ):
            # ...except possibly for the wash. THIS IS WHY THE POLYGONS SOMETIMES DID NOT
            # RENDER. The dissolve is deliberately skipped when a camera event arrives
            # while it is about to run, because a wash for a view that has already moved is
            # wasted work. But nothing ever came back for it: the next _draw found the
            # screen already correct and returned here, so the cells were right and the
            # outlines were simply absent, for as long as you stayed put. It looked random
            # because it depended on whether one more camera event landed inside a
            # particular half-second.
            ensure_wash()
            return

        # A resolution we have folded before, still covering the screen. This is the whole
        # zoom-out case: coming back up to a level already visited is a dict lookup, not a
        # read, so it lands complete and instantly instead of arriving a second later.
        hit = HOLD["cache"].get(res)
        if not force and hit and _covers(hit[0], seen):
            _show(
                hit[1],
                res,
                hit[0],
                f"<b>res {res}</b> · {hit[1].num_rows:,} cells · cached"
                f" · zoom {vs.zoom:.1f} · {HOLD['source']}",
            )
            # The wash was cached with the cells it describes, so a zoom back to a level
            # already visited gets its outlines back instantly. Without this the cluster
            # layer stayed hidden on every cache hit, because _show hides it and only the
            # dissolve path ever turns it on again: zoom out and the outlines were simply
            # gone until something forced a real fold.
            if hit[2] is not None:
                put_clusters(hit[2])
            else:
                # Cached cells, but the wash for them was skipped mid-drag. Dissolve now.
                ensure_wash()
            return

        # NOTHING ON SCREEN MAY OUTLIVE ITS RESOLUTION. Leaving the previous fold up for the
        # duration of the read is what produces the two complaints: zoom IN and the old
        # coarse cells sit there at the wrong size, zoom OUT and the old FINE cells are
        # suddenly sub-pixel, which aliases into a black mush that looks like corruption.
        # Neither is data; both are the last answer overstaying.
        #
        # Cleared with seed_table(), NOT an empty one. A zero-row table blanked the map and
        # then killed deck outright, and it did not come back on a re-run. seed_table is the
        # single off-screen hexagon the layer is CONSTRUCTED with, so it is the one shape
        # known to survive; at null island it is never in a CONUS view.
        if res != HOLD["res"]:
            _show(seed_table(), None, None, f"<b>reading…</b> res {res} · zoom {vs.zoom:.1f}")
        else:
            status.value = f"<b>reading…</b> res {res} · zoom {vs.zoom:.1f}"

        raw, m_px, read_px = await HOLD["fold"](res, want)
        if raw is None or raw.num_rows == 0:
            HOLD["res"], HOLD["box"] = res, want
            status.value = f"<b>res {res}</b> · nothing here · zoom {vs.zoom:.1f}"
            return
        tbl = to_layer_table(raw)
        # box, cells, wash, raw fold. The raw one is kept so the wash threshold can be
        # changed without reading or refolding anything.
        HOLD["cache"][res] = [want, tbl, None, raw]
        _show(
            tbl,
            res,
            want,
            f"<b>res {res}</b> · {tbl.num_rows:,} cells · {m_px:.0f} m"
            f" · {'tiles cached' if read_px == 0 else f'{read_px / 1e6:.2f}M px fetched'}"
            f" · zoom {vs.zoom:.1f} · {HOLD['source']}"
            + (f" · <b style='color:#E69F00'>{HOLD['jumps']} jumps</b>" if HOLD["jumps"] else ""),
        )

        # THE CELLS GO OUT FIRST, ON THEIR OWN. The cluster pass costs about 1.5 s at res 8
        # (0.5 union-find, 1.0 dissolve) and the map is already showing the previous fold
        # for the whole read; making the cells wait on the wash would lengthen exactly the
        # window that reads as a glitch. sleep(0) hands the loop back so the cells actually
        # reach the browser before the dissolve blocks it again.
        await asyncio.sleep(0)
        if HOLD["pending"] is not None:
            return  # camera moved on; this wash would be for a view that is gone
        ctbl = to_cluster_table(raw, controls.min_cluster, CLUSTER_DARKEN)
        if ctbl is None:
            clusters.visible = False
            # Built, in the sense that this fold HAS its answer and the answer is "no run
            # is that large". False here would make ensure_wash redo the dissolve on every
            # settle, for the one view where it is guaranteed to find nothing.
            WANT["built"] = True
            return
        if res in HOLD["cache"]:
            HOLD["cache"][res][2] = ctbl
        put_clusters(ctbl)

    def ensure_wash():
        """Dissolve the wash for the fold on screen, if it does not have one yet.

        Cheap when there is nothing to do, which is the common case: a pan inside the
        padded box calls this on every settle.
        """
        if WANT["built"] or not WANT["clusters"]:
            return
        rewash(quiet=True)

    def rewash(quiet=False):
        """Re-dissolve the CURRENT fold at the new threshold. No read, no refold.

        The raw fold is kept in the cache precisely so this costs only the dissolve: the
        pixels and the H3 aggregation are both already done, and the only thing changing
        is which runs are large enough to draw.
        """
        ent = HOLD["cache"].get(HOLD["res"])
        if not ent or ent[3] is None:
            return
        if not quiet:
            status.value = f"<b>dissolving…</b> min cluster {controls.min_cluster}"
        ctbl = to_cluster_table(ent[3], controls.min_cluster, CLUSTER_DARKEN)
        ent[2] = ctbl
        if ctbl is None:
            clusters.visible = False
            # Nothing here dissolves at this threshold, and that is an ANSWER, not a gap.
            # Marking it built stops ensure_wash retrying the same expensive dissolve on
            # every settle for as long as the camera sits still.
            WANT["built"] = True
            if not quiet:
                status.value = (
                    f"<b>min cluster {controls.min_cluster}</b> · no run that large here"
                )
            return
        put_clusters(ctbl)
        if not quiet:
            status.value = (
                f"<b>min cluster {controls.min_cluster}</b> · {ctbl.num_rows:,} polygons"
            )

    def put_clusters(ctbl):
        """Put a dissolved wash on the cluster layer. The only place that does."""
        clusters._rows_per_chunk = max(1, infer_rows_per_chunk(ctbl))
        # hold_sync for the same reason as the cells: one message, or deck draws the
        # new polygons against the old colour buffer.
        with clusters.hold_sync():
            clusters.table = ctbl
            clusters.get_line_color = ctbl["color"]
            clusters.get_fill_color = ctbl["color"]
            clusters.visible = WANT["clusters"]
        WANT["built"] = True

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
    return controls, deck, refresh, status


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

    # THE ONLY CACHE THAT SHOULD EXIST: NLCD PIXELS, ON A FIXED GRID.
    #
    # Every fold used to issue one ranged read for its exact padded viewport, so panning
    # half a screen re-read the half already in memory, at a different offset, and nothing
    # could ever be reused: no two camera positions produce the same rectangle. Snapping to
    # a fixed TILE grid per overview level is what makes a read SHAREABLE. A pan now touches
    # the tiles it already holds plus a strip of new ones, and a zoom back to a level
    # visited before is free.
    #
    # Measured against the single ranged read it replaces (source.coop, warm process):
    #   L3 viewport 1400x620, cold    118 ms tiled  vs  162 ms direct   (parallelism wins)
    #   L3 viewport, revisited          0 ms tiled  vs  128 ms direct
    #   L5 whole country, cold        463 ms tiled  vs  179 ms direct   (70 tiles, once)
    #   L5 whole country, revisited     2 ms tiled  vs  325 ms direct
    # The one regression is the opening whole-country draw. Everything after it is free,
    # and zoom-out is the motion that used to cost the most.
    #
    # FETCH_AT_ONCE is the number that matters: at 12 the same L5 read took 1,009 ms and
    # the tiling looked like a mistake. Tiles are only faster if they are in flight
    # together.
    TILE = 512
    TILE_BUDGET = 384 * 1024 * 1024  # uint8, so ~1,500 tiles resident
    FETCH_AT_ONCE = 32

    _tiles = {}  # (level, ty, tx) -> uint8 array; insertion order is LRU order
    # A dict, not an int, because a marimo cell body is compiled at MODULE scope: `nonlocal`
    # in a nested def is a SyntaxError there, and a bare rebind would shadow instead of
    # accumulate. The export is what catches this; a plain import never runs the cell.
    _held = {"bytes": 0}
    _sem = asyncio.Semaphore(FETCH_AT_ONCE)

    async def _tile(li, ty, tx):
        rd = _levels[li]
        H, W = rd.shape
        r0, c0 = ty * TILE, tx * TILE
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        async with _sem:
            return np.asarray(
                (
                    await rd.read(
                        window=Window(col_off=c0, row_off=r0, width=w, height=h)
                    )
                ).as_masked()[0]
            )

    async def _read_window(li, col0, row0, wpx, hpx):
        """The window, assembled from cached tiles plus whatever is missing.

        Returns (array, pixels actually fetched), so the status line reports network work
        rather than window size. Verified byte-identical to a single ranged read of the
        same window at L0, L2 and L4.
        """
        ty0, ty1 = row0 // TILE, (row0 + hpx - 1) // TILE
        tx0, tx1 = col0 // TILE, (col0 + wpx - 1) // TILE
        want = [(li, ty, tx) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        need = [k for k in want if k not in _tiles]

        fetched = 0
        if need:
            got = await asyncio.gather(*(_tile(*k) for k in need))
            for k, a in zip(need, got):
                _tiles[k] = a
                _held["bytes"] += a.nbytes
                fetched += a.size
            # Oldest first, and never evict a tile this window is about to read.
            while _held["bytes"] > TILE_BUDGET and len(_tiles) > len(want):
                for k in list(_tiles):
                    if k not in want:
                        _held["bytes"] -= _tiles.pop(k).nbytes
                        break
                else:
                    break

        out = np.full((hpx, wpx), NODATA, dtype=np.uint8)
        for k in want:
            _, ty, tx = k
            a = _tiles[k]
            sr, sc = ty * TILE, tx * TILE
            r0, c0 = max(row0, sr), max(col0, sc)
            r1, c1 = min(row0 + hpx, sr + a.shape[0]), min(col0 + wpx, sc + a.shape[1])
            if r1 <= r0 or c1 <= c0:
                continue
            out[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = a[
                r0 - sr : r1 - sr, c0 - sc : c1 - sc
            ]
        # Touch: anything this window used goes to the young end of the LRU.
        for k in want:
            _tiles[k] = _tiles.pop(k)
        return out, fetched

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
        li = LEVEL_FOR_RES[res]
        rd = _levels[li]
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

        # Tiles, not a bespoke rectangle. `fetched` is what actually crossed the
        # network, which is usually far less than the window.
        arr, fetched = await _read_window(li, col0, row0, wpx, hpx)

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
        return out, rd.res[0], fetched

    # Hand the fold to the camera's world and draw where the camera already is.
    HOLD["fold"] = fold
    HOLD["to_albers"] = _to_albers
    HOLD["extent"] = (_l, _b, _r, _t)
    HOLD["source"] = str(YEAR)
    HOLD["loop"] = asyncio.get_running_loop()
    await refresh(deck.view_state, force=True)
    return


@app.cell
def _(GROUPS, controls, deck, mo, status):
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
            controls,
        ],
        gap=0.4,
    )
    return


if __name__ == "__main__":
    app.run()
