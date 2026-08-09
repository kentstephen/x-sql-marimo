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
#     "numpy==2.5.1",
#     "pyproj>=3.7",
#     "duckdb>=1.5.5",
#     "matplotlib==3.11.1",
#     "pillow>=11",
# ]
# ///
"""NLCD land cover in H3, extruded on Mapterhorn terrain. Two rasters, one cell id.

TWO ENGINES, EACH DOING THE HALF IT WINS. This is the split version of
`xsql-nlcd-zoom.py`, and the division of labour was benchmarked rather than assumed,
because it goes both ways. Same viewport, 1.58M pixels to 132,759 cells:

  the FOLD, pixels -> cells -> majority class   DataFusion + h3ronpy  70 ms
                                                DuckDB               462 ms
  the DISSOLVE, cells -> region outlines        DuckDB                75 ms
                                                h3ronpy              928 ms

The fold stays in DataFusion because h3ronpy converts a whole column at once, where
DuckDB calls h3_latlng_to_cell once per row, 1.58 million times. The dissolve goes to
DuckDB because its h3 extension wraps Uber's C library, where the cells-to-polygon work
lives; h3ronpy wraps h3o, a separate Rust reimplementation, so that work never reaches
it. The union-find that used to prepare cells for the slow dissolve is gone from that
path entirely: WASH_SQL dissolves everything and ST_Dump recovers the connected runs.

Nothing is read until the camera asks for it. Each fold pulls only the padded viewport,
from the overview that matches the H3 resolution it is about to build, registers that
window with xarray-sql and folds it in SQL. The counter-intuitive part is that the FINEST
views are the cheapest: the viewport shrinks faster than the resolution grows, so res 11
at 30 m reads 72,890 pixels where res 5 at 1920 m reads about 4.1M.

That is what gets to res 11. Below it the cells would be finer than the imagery: a res 11
hexagon holds 2.3 pixels of 30 m NLCD, and res 12 would hold 0.6 and hole out. The ceiling
is the data's, not the code's.

The fold is a mode, not a mean, because land cover is categorical: each cell takes its
most frequent class, and colour is the class.

TWO RASTERS, JOINED ON THE CELL ID. This is the split version's other half. NLCD is a
CONUS Albers mosaic at 30 m; Mapterhorn is a global Web Mercator PMTiles pyramid of
terrarium-encoded WebP. They share no CRS, no grid and no resolution. They do share an
H3 cell id, so the join is a LEFT JOIN in DataFusion on a UBIGINT and nothing has to be
warped, resampled or reprojected into anything else first. Class comes from one raster,
height from the other, and the hexagon carries both.

The colour is still the class. The HEIGHT is the terrain, which means the extrusion is
a second variable rather than a restatement of the first: forest climbing a ridge, crops
stopping where the ground tilts. Exaggeration 0 is the flat map this started as, and the
cluster outlines are dissolved on the ground plane, so turning height up eventually
buries them. That trade is the slider's, not a mode.

The Mercator half is the cheap half, which is not obvious. Mapterhorn tile pixels ARE
lat/lon by construction, so the DEM fold registers its window with lat/lon coordinates
and the H3 UDF reads them straight. NLCD needs a 64x64 pyproj control grid and two
interpolating UDFs to say the same thing in Albers.

The camera never re-runs a marimo cell. It schedules a coroutine that reads, folds and
swaps three traits on the one live layer, so panning and zooming stay fluid and the view
is never reset. Same shape as the Jupyter tutorial in `bias-bounty-map-tutorial`, which is
where the pattern is proven.

Data, both public and unsigned on source.coop:
  land cover  Kyle Barron's mirror of USGS Annual NLCD, 30 m Albers, one COG per year
  terrain     Mapterhorn planet.pmtiles, z0-12, 512 px terrarium WebP, 705 GB

Only planet.pmtiles is read. Its z12 is 19.1 m at the equator and 14.6 m at 40N, which
is already finer than the 30 m NLCD this is joined against, so the 457 regional
`6-x-y.pmtiles` archives (z13-18, 26.7 TB) are not needed: they would resolve terrain
the land cover cannot match. z13 is the 10 m level if that ever changes.

Run:  uv run marimo edit xsql-duckdb-terrain-h3.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import io
    import math
    import struct

    import duckdb
    import obstore
    from PIL import Image

    import anywidget
    import traitlets
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import matplotlib.pyplot as plt
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
    from h3ronpy.vector import coordinates_to_cells
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
        Image,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        from_wkb,
        grid_disk,
        gzip,
        infer_rows_per_chunk,
        io,
        math,
        mo,
        multipolygon,
        np,
        obstore,
        pa,
        plt,
        struct,
        traitlets,
        udf,
        xr,
    )


@app.cell
def _(duckdb):
    # ONE JOB: dissolving H3 cells into outlines. The fold stays in DataFusion, and that is
    # measured, not taste. Same viewport, 1.58M pixels to 132,759 cells with a majority
    # class: DataFusion + h3ronpy 70 ms, DuckDB 462 ms, because h3ronpy converts a whole
    # column at once while DuckDB calls h3_latlng_to_cell 1.58M times, once per row.
    #
    # The dissolve is the exact mirror of that, and it is why this connection exists:
    #   h3ronpy, union-find + dissolve of the filtered set   928 ms   (what shipped)
    #   duckdb, same filtered set                             57 ms
    #   duckdb, ALL cells, no union-find at all               75 ms
    #   h3ronpy, ALL cells                                 2,784 ms
    # DuckDB dissolving everything is 12x faster than h3ronpy dissolving the subset that
    # the union-find spent 284 ms preparing for it. So the union-find leaves this path
    # entirely: it only ever existed to keep h3ronpy's dissolve tractable.
    #
    # The reason is the C API. duckdb-h3 wraps Uber's C library, where the cells-to-polygon
    # work lives (and where AJ Friend's 4x rewrite landed, uber/h3 #1113). h3ronpy wraps
    # h3o, a separate Rust reimplementation, so that work never reaches it.
    #
    # Extensions download once into ~/.duckdb and are cached after that.
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


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
            "display:flex;flex-wrap:nowrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.2rem 0 0;user-select:none;overflow:hidden";

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
            w.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
            const cap = document.createElement("span");
            const draw = () => {
              cap.textContent = label + " " + Number(model.get(key)).toFixed(2);
            };
            cap.style.cssText = "opacity:.7;white-space:nowrap";
            const s = document.createElement("input");
            s.type = "range";
            s.min = "0"; s.max = "1"; s.step = "0.05";
            s.value = model.get(key);
            s.style.cssText = "width:5rem;margin:0;cursor:pointer";
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

          // STOPS, NOT A RANGE. min cluster ran 1 to 300 in steps of 1 across a 5rem
          // track: 300 values over 80 pixels, so nearly four values moved under every
          // pixel and landing on one was luck. The slider now indexes a list of stops,
          // which fixes both halves of that. It is coarse where the answer is coarse (the
          // polygon count collapses 1,200x between 1 and 300, so 240 and 250 are the same
          // map) and fine at the bottom where each step visibly changes what survives.
          //
          // Still fires on CHANGE, not INPUT: each stop re-dissolves the wash, which is
          // real work, so it runs when the handle is released and the caption tracks the
          // drag in the meantime.
          const nearest = (stops, v) => {
            let best = 0;
            for (let i = 1; i < stops.length; i++) {
              if (Math.abs(stops[i] - v) < Math.abs(stops[best] - v)) best = i;
            }
            return best;
          };
          // `live` is the difference between the two sliders that use this. Height is a
          // trait assignment on a layer that already holds its data, so it can follow the
          // drag; min cluster re-dissolves the wash, so it waits for the handle to drop
          // and the caption tracks the drag in the meantime.
          const steps = (key, label, stops, width, live) => {
            const w = document.createElement("span");
            w.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const cap = document.createElement("span");
            cap.style.cssText = "opacity:.7;white-space:nowrap";
            const draw = () => { cap.textContent = label + " " + model.get(key); };
            const s = document.createElement("input");
            s.type = "range";
            s.min = "0"; s.max = String(stops.length - 1); s.step = "1";
            s.value = String(nearest(stops, model.get(key)));
            // Wider than the opacity sliders because this one is aimed rather than
            // nudged, and every pixel of track is a stop you can actually land on.
            s.style.cssText = "width:" + width + ";margin:0;cursor:pointer";
            const push = () => {
              model.set(key, stops[parseInt(s.value, 10)]);
              model.save_changes();
            };
            s.addEventListener("input", () => {
              cap.textContent = label + " " + stops[parseInt(s.value, 10)];
              if (live) push();
            });
            s.addEventListener("change", push);
            model.on("change:" + key, () => {
              s.value = String(nearest(stops, model.get(key)));
              draw();
            });
            draw();
            w.appendChild(cap);
            w.appendChild(s);
            return w;
          };
          const CLUSTER_STOPS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300];
          // Vertical exaggeration, as a MULTIPLE of a per-resolution base scale, not as a
          // raw elevation_scale. The base is set in the kernel from the hexagon's own edge
          // length, because the same 1000 m of relief is invisible against 8.5 km hexagons
          // at res 5 and a tower against 25 m ones at res 11. 0 is the flat map.
          const EXAG_STOPS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5];

          // The class filter. Options come from the kernel (`class_options`) so the
          // groupings are written down once, next to the palette they select from,
          // rather than a second time in JS.
          const choose = (key, optsKey, label) => {
            const w = document.createElement("span");
            w.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
            const cap = document.createElement("span");
            cap.textContent = label;
            cap.style.cssText = "opacity:.7;white-space:nowrap";
            const s = document.createElement("select");
            s.style.cssText = "font:inherit;padding:.05rem .2rem;cursor:pointer";
            for (const [val, txt] of model.get(optsKey)) {
              const o = document.createElement("option");
              o.value = val;
              o.textContent = txt;
              s.appendChild(o);
            }
            s.value = model.get(key);
            s.addEventListener("change", () => {
              model.set(key, s.value);
              model.save_changes();
            });
            model.on("change:" + key, () => { s.value = model.get(key); });
            w.appendChild(cap);
            w.appendChild(s);
            return w;
          };

          // GROUPED, NOT SHORTENED. There are two opacities here and they do different
          // things: one dims the hexagons, one dims the cluster polygons. Trimming the
          // labels to fit one line made them both read as "opacity", which is why the
          // group headings carry the noun instead and every control under a heading
          // belongs to it.
          const group = (label) => {
            const g = document.createElement("span");
            g.textContent = label;
            g.style.cssText =
              "font:11px ui-monospace,Menlo,monospace;letter-spacing:.06em;" +
              "text-transform:uppercase;opacity:.5;padding-left:.2rem";
            return g;
          };
          box.appendChild(group("classes"));
          box.appendChild(choose("class_set", "class_options", "show"));
          box.appendChild(group("terrain"));
          box.appendChild(steps("exaggeration", "height", EXAG_STOPS, "5rem", true));
          box.appendChild(group("hexagons"));
          box.appendChild(check("cells", "show"));
          box.appendChild(slider("cell_opacity", "opacity"));
          box.appendChild(slider("cell_coverage", "coverage"));
          box.appendChild(group("clusters"));
          box.appendChild(check("cluster_fill", "fill"));
          box.appendChild(check("cluster_line", "outline"));
          box.appendChild(slider("cluster_opacity", "opacity"));
          box.appendChild(steps("min_cluster", "min cluster", CLUSTER_STOPS, "7rem", false));
          el.appendChild(box);
        }
        export default { render };
        """
        # FOREST, not everything, on purpose. The full palette over a whole state is a
        # picture of the country; one grouping is a question about it, and the outlines
        # only read as regions when they are not every class bordering every other.
        # "all" is in the menu for the picture.
        class_set = traitlets.Unicode("forest").tag(sync=True)
        class_options = traitlets.List([]).tag(sync=True)
        cells = traitlets.Bool(True).tag(sync=True)
        # 1.0, not 0. At the opening pitch the extrusion is visible without hiding
        # anything: a top-down camera sees a hexagon's top face at the same footprint as
        # its flat self, so the cluster outlines underneath survive until you tilt. Tilting
        # is what trades them for relief, and that is the slider's decision to offer.
        exaggeration = traitlets.Float(1.0).tag(sync=True)
        # OFF by default now that the outline is dissolved on parent hexagons. The
        # polygon is coarser than the cells and bulges past them, so filling it paints a
        # class over cells that are not that class. As an outline it reads as "a region is
        # about here"; as a fill it reads as data, and it would be wrong.
        cluster_fill = traitlets.Bool(False).tag(sync=True)
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
        # Counted in whatever `outline coarsen` is set to: cells at 0, parent hexagons
        # above that, where one parent holds ~7 children. So dropping coarsen from 1 to 0
        # means this wants to go up by roughly 7x to keep the same amount of speckle out.
        min_cluster = traitlets.Int(20).tag(sync=True)

    return (Controls,)


@app.cell
def _(math):
    PREFIX = "kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic"
    NODATA = 250

    # Which overview each H3 resolution reads. The source pyramid is 30 m native and
    # doubles SEVEN times: L0 30 m, L1 60, L2 120, L3 240, L4 480, L5 960, L6 1920.
    # L6 is 2500x1640 for the whole conterminous US, ~4 MB of uint8. This comment used to
    # stop at L5 and so did the table below, which is the only reason the top two rows read
    # a level finer than they need.
    #
    # Picked so the mode has enough pixels under it to mean something. px/hex:
    #   res 5  L6   69  ·  res 6  L5  39  ·  res 7  L4  22  ·  res 8  L3  12.5
    #   res 9  L2  7.1  ·  res 10 L1  4.1  ·  res 11 L0  2.3
    # (res 7 down are measured; res 5 and 6 are their measured 277 and 157 divided by the
    # 4x area of one coarser level.)
    #
    # Using L6 at all depends on the pyramid being nearest/mode resampled, since an
    # `average` over class codes is a blend of arbitrary integers. Verified rather than
    # assumed: one 512x512 tile decoded from L6, L5 and L0 contains only legal NLCD codes.
    #
    # res 11 against 30 m imagery is 2.3 pixels per hexagon, and that is the floor: res 12
    # would be 0.6 and the map would hole out. This is where the data stops, not where the
    # code does.
    LEVEL_FOR_RES = {5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1, 11: 0}
    MAX_RES = 11

    # ---------------------------------------------------------------- terrain (mapterhorn)
    PM_BUCKET = "us-west-2.opendata.source.coop"
    PM_PATH = "mapterhorn/mapterhorn/planet.pmtiles"
    PM_TILE = 512  # mapterhorn ships 512 px tiles, so this is the source's own grid
    # Terrarium: elevation = (R*256 + G + B/256) - 32768. Verified against known summits
    # on decode: the Rainier tile tops out at 4391.6 m against a true 4392.
    DEM_FLOOR = -500.0  # below the Dead Sea; anything under this is void, not terrain

    # WHICH MAPTERHORN ZOOM EACH H3 RESOLUTION READS. This is a COST table, not a
    # resolution match, and the two axes pull against each other:
    #
    #   px/hex   DEM pixels landing in one hexagon. Elevation is smooth and this is a
    #            mean, so ~8 samples is already a solid answer; the floor that matters is
    #            not accuracy but COVERAGE, because a cell that catches no pixel has no
    #            height and punches a hole in the extrusion.
    #   tiles    512 px tiles the padded viewport spans, which is what the read costs.
    #            One more zoom level is 4x the tiles.
    #
    # H3 steps by 7x in area and zoom by 4x, so these never line up and px/hex oscillates
    # however the table is written. Aiming at the cheapest level clearing ~8 px/hex, at
    # 40N (hexagon areas are H3's published averages):
    #   res  5 -> z4   18.0 px/hex      res  9 -> z9    7.7 px/hex
    #   res  6 -> z5   10.3 px/hex      res 10 -> z11  17.5 px/hex
    #   res  7 -> z7   23.5 px/hex      res 11 -> z12  10.0 px/hex
    #   res  8 -> z8   13.4 px/hex
    # res 7 and res 10 take the jump because the level below them (5.9 and 4.4 px/hex) is
    # where coverage starts to get patchy.
    #
    # THESE NUMBERS WERE WRONG BY 4x IN AN EARLIER DRAFT, in the safe direction: the table
    # read a level finer than it needed everywhere and quadrupled the tile count at res 5,
    # 6, 8 and 9. Caught by measuring instead of deriving. A res-8 window over the Front
    # Range folds at 57 px/hex against 53.8 predicted, which is the check that matters.
    #
    # z12 is the floor of the planet pyramid, so res 11 ends at 10 px/hex. Going finer
    # needs the regional z13+ archives and there is no point: NLCD is 30 m, so the class
    # would stop resolving well before the terrain did.
    DEM_ZOOM_FOR_RES = {5: 4, 6: 5, 7: 7, 8: 8, 9: 9, 10: 11, 11: 12}

    # Average H3 edge length in metres. Used ONLY to size the extrusion: 1000 m of
    # elevation should read as about half a hexagon's width whatever the resolution, or
    # the same terrain is invisible at res 5 and a skyscraper at res 11.
    EDGE_M = {5: 8544.4, 6: 3229.5, 7: 1220.6, 8: 461.4, 9: 174.4, 10: 65.9, 11: 24.9}

    def elev_base_scale(res):
        return EDGE_M[res] / 2000.0

    # The map's pixel size, assumed. It only sets how much of the world the viewport box
    # covers, and PAD is deliberately loose, so being wrong by a few hundred pixels costs
    # a slightly larger query and nothing else.
    # VIEW_H is also what the map is DRAWN at, so it decides whether the page scrolls.
    # Status line, legend, caption and the control row come to roughly 150 px under it;
    # 620 keeps the whole thing inside a laptop viewport without a scrollbar.
    VIEW_W, VIEW_H = 1400, 620

    # Fold a box larger than the screen, so a small pan lands inside what is already
    # folded and needs no query at all. This is squared into area, so 1.35 already means
    # folding 1.8x what you can see; 1.8 would mean 3.2x.
    PAD = 1.35

    # CLUSTER WASH. A run of touching cells of the same class, dissolved into one polygon
    # (in DuckDB, see WASH_SQL) and laid over the hexes in the class's own colour.
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
    CLUSTER_WIDTH = 3  # stroke width in screen pixels
    # 1.0 is the class colour exactly. Lower values darken the edge; below about 0.6 every
    # class collapses toward black and the outlines stop telling each other apart.
    CLUSTER_DARKEN = 1.0
    # THE OUTLINE IS EXACT, at the resolution on screen. It was not always: coarsening the
    # dissolve onto parent hexagons used to be the only way to keep h3ronpy's super-linear
    # merge tractable (652 ms for one res-8 viewport), and it cost exactness, because a
    # parent counts as included when ANY of its children are and the boundary bulges out by
    # up to one parent hexagon. Moving the dissolve to DuckDB made that trade unnecessary:
    # WASH_SQL dissolves `list(hex)` at native resolution in ~17 ms for the same view.
    #
    # DIRECTED EDGES WERE MEASURED AS THE REPLACEMENT FOR ALL OF THIS, and lost. The idea
    # is sound: emit the shared edge wherever a cell's neighbour is absent or a different
    # class, via h3_cells_to_directed_edge + h3_directed_edge_to_boundary_wkb, and skip
    # polygon topology entirely. Same forest viewport, 24,902 cells:
    #   dissolve + ST_Dump, min 20     17.3 ms     85 rows   0.110 MB
    #   directed edges, no despeckle   38.7 ms  106,422 rows 6.070 MB
    #   union-find despeckle -> edges  30.2 ms      3 rows   0.412 MB
    # Two reasons it loses. ST_Dump is the real prize, not the polygons: splitting the
    # multipolygon into its connected runs is what makes `min cluster` possible at all, and
    # despeckling is where 98% of the payload goes. Edges have no run grouping, so the
    # union-find this path deliberately deleted has to come back, and it is still heavier.
    # Second, edges do not chain: every hexagon rim is its own two-point linestring, so
    # shared vertices are stored twice. ST_LineMerge chains them for 191 ms and saves no
    # bytes. h3ronpy cannot construct a directed edge from a cell pair at all, only render
    # one, so this would have been DuckDB's job either way.

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
    # WHAT THE MAP IS SHOWING. Broad groupings over the 16 NLCD classes, because "all of
    # them at once" is a picture and "one thing at a time" is a question: forest against
    # everything that is not forest is legible in a way the full palette is not, and the
    # dissolved outlines stop being a mosaic of every neighbour and become the shape of
    # one land cover. It is a DISPLAY filter, applied to the fold that is already in hand:
    # switching costs a re-derive of the cells and one dissolve, no read and no refold.
    # None means no filter at all.
    CLASS_SETS = {
        "forest": ("Forest", (41, 42, 43)),
        "developed": ("Developed", (21, 22, 23, 24)),
        "agriculture": ("Agriculture", (81, 82)),
        "water": ("Water & wetland", (11, 12, 90, 95)),
        "open": ("Barren, shrub & grass", (31, 52, 71)),
        "all": ("Everything", None),
    }
    # Lists, not tuples: this crosses to the browser as JSON on a synced trait.
    CLASS_OPTIONS = [[_k, _v[0]] for _k, _v in CLASS_SETS.items()]
    # One year, pinned. The slider is out until the camera path is proven: a year change
    # is a fresh read of the whole country, and there is no point putting that behind a
    # control while the thing it feeds is still the open question.
    YEAR = 2024
    # The whole Annual NLCD series on the mirror. Measured for a drawn box at the zoom's own
    # resolution: 0.45-1.5 s to read all 40 years, ~40 ms to compose them, ~30 ms/year to
    # fold to H3 and 4-10 ms to dissolve a class. The series is cheap because the BOX is
    # small and the overview matches the res; this is the same inversion the camera path
    # runs on. Opening 40 COG headers is the one real cost, 3,979 ms, so it is prefetched.
    YEARS = list(range(1985, 2025))
    return (
        CLASS_OPTIONS,
        CLASS_SETS,
        CLUSTER_DARKEN,
        CLUSTER_OPACITY,
        CLUSTER_WIDTH,
        DEM_FLOOR,
        DEM_ZOOM_FOR_RES,
        GROUPS,
        LEVEL_FOR_RES,
        NODATA,
        PAD,
        PM_BUCKET,
        PM_PATH,
        PM_TILE,
        PREFIX,
        SETTLE,
        VIEW_H,
        VIEW_W,
        YEAR,
        YEARS,
        elev_base_scale,
        res_for_zoom,
    )


@app.cell
def _(mo):
    # The drawn box, and the H3 resolution the map was showing when it was drawn. State,
    # not a plain dict, because unlike the camera this SHOULD re-run something: the cell
    # below it. The Map cell only ever holds the setter, never reads the value, so drawing
    # a box cannot rebuild the map or move the camera.
    get_aoi, set_aoi = mo.state(None)
    return get_aoi, set_aoi


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
    con,
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

    def filter_classes(tbl, keep):
        """Keep only the classes in `keep`, on the RAW fold, before anything is derived.

        Filtering here rather than at the layer is what makes the dissolve agree with the
        cells: the wash is dissolved from whatever rows this returns, so an outline can
        only ever describe cells that are actually on screen. `keep` of None is the
        everything case and costs nothing.
        """
        if keep is None or tbl is None or tbl.num_rows == 0:
            return tbl
        return tbl.filter(pa.array(np.isin(np.asarray(tbl["mode_cls"]), list(keep))))

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
                    "elev": tbl["elev"],
                }
            )
        )

    # THE WHOLE WASH, IN ONE STATEMENT. Dissolve every class, split the result into its
    # connected runs, size each run, drop the speckle. 87 ms for a res-8 viewport, against
    # 928 ms for the union-find-then-dissolve it replaces.
    #
    # ST_Dump is what keeps `min run` meaningful. h3_cells_to_multi_polygon_wkb returns ONE
    # MultiPolygon per class, and its parts are exactly the connected runs, so dumping them
    # recovers the same grouping the union-find used to compute.
    #
    # The size test is in AREA, not cells, because after the dissolve there are no cells
    # left to count. cell_area is calibrated per class from the data in hand (total area
    # over total cells), so it needs no hexagon-area constant and no latitude correction:
    # whatever a cell is worth in this view, that is what it is divided by. Checked against
    # the union-find on the same field: largest run 70,610 by area, 70,581 by counting.
    WASH_SQL = """
        WITH dissolved AS (
            SELECT mode_cls                                 AS cls,
                   count(*)                                 AS n_cells,
                   h3_cells_to_multi_polygon_wkb(list(hex)) AS mp
            FROM cells GROUP BY mode_cls
        ), parts AS (
            SELECT cls, n_cells, UNNEST(ST_Dump(ST_GeomFromWKB(mp))).geom AS geom
            FROM dissolved
        ), sized AS (
            SELECT cls, geom, ST_Area(geom) AS area,
                   sum(ST_Area(geom)) OVER (PARTITION BY cls)
                     / max(n_cells) OVER (PARTITION BY cls) AS cell_area
            FROM parts
        )
        SELECT cls, ST_AsWKB(geom) AS wkb
        FROM sized
        WHERE area >= ? * cell_area
        ORDER BY cls
    """

    def to_cluster_table(tbl, min_cluster, darken=1.0):
        """Runs of touching like cells, dissolved into one polygon each, in DuckDB.

        Returned as arro3 arrays, not pyarrow: pa.array() strips the geoarrow extension
        metadata off the geometry and lonboard's table trait rejects it as "expected
        geometry column in table".
        """
        if tbl.num_rows == 0:
            return None

        # `cells` is the name WASH_SQL selects from, and there is no register() call:
        # DuckDB's replacement scan resolves it straight out of this frame, Arrow buffers
        # and all, even as a local inside a nested function. to_arrow_table, not arrow(),
        # which hands back a RecordBatchReader; and not fetch_arrow_table, which is
        # deprecated as of 1.5.
        cells = tbl  # noqa: F841 - read by the replacement scan, not by Python
        out = con.sql(WASH_SQL, params=[float(min_cluster)]).to_arrow_table()
        if out.num_rows == 0:
            return None

        cls = np.asarray(out["cls"])
        # Straight from the Arrow column. to_pylist() here would materialise every polygon
        # as a Python bytes object on the way past, which is the one place this path could
        # have thrown away what DuckDB just won.
        geom = from_wkb(
            out["wkb"].combine_chunks(), to_type=multipolygon("xy", crs="EPSG:4326")
        )
        rgb = (_lut[cls] * darken).clip(0, 255).astype(np.uint8)
        return ArroTable.from_arrays(
            [
                ArroArray.from_arrow(geom),
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(rgb.ravel()), 3)
                ),
            ],
            names=["geometry", "color"],
        )

    def cluster_runs(hx, cs):
        """Connected runs of touching same-class cells.

        Returns (label per cell, size per label). This is the ONE piece of topology the
        hexagons cannot express on their own: which cells are part of the same thing. The
        wash uses it to drop speckle; the AOI analytics uses it to count patches and
        measure the largest, which is the question a class total cannot answer (two views
        with identical composition can be one block or ten thousand scraps).
        """
        n = len(hx)
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
        return inv, counts

    def patch_stats(hx, cs, min_cluster=1):
        """Per class: cells, number of patches, size of the largest. Objects, not cells."""
        inv, counts = cluster_runs(hx, cs)
        first = np.unique(inv, return_index=True)[1]
        lab_cls = cs[first]  # every cell in a run shares its class
        keep = counts >= min_cluster
        out = {}
        for c in np.unique(cs):
            m = keep & (lab_cls == c)
            out[int(c)] = (
                int(counts[lab_cls == c].sum()),
                int(m.sum()),
                int(counts[m].max()) if m.any() else 0,
            )
        return out

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
                    "elev": pa.array([0.0]),
                }
            )
        )

    return (
        filter_classes,
        patch_stats,
        seed_cluster,
        seed_table,
        to_cluster_table,
        to_layer_table,
    )


@app.cell
def _(
    BitmapTileLayer,
    CLASS_OPTIONS,
    CLASS_SETS,
    CLUSTER_DARKEN,
    CLUSTER_OPACITY,
    CLUSTER_WIDTH,
    CartoBasemap,
    Controls,
    H3HexagonLayer,
    HOLD,
    Map,
    MaplibreBasemap,
    PAD,
    PolygonLayer,
    SETTLE,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    elev_base_scale,
    filter_classes,
    infer_rows_per_chunk,
    math,
    res_for_zoom,
    seed_cluster,
    seed_table,
    set_aoi,
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
        get_elevation=_seed["elev"],
        # Both set for real by apply_elevation() as soon as a resolution is known; the
        # scale is meaningless until then, because it is derived from the hexagon size.
        extruded=False,
        elevation_scale=0.0,
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
        # PITCHED ON OPENING, because an extrusion nobody tilts to see is a flat map that
        # costs a second raster. 40 degrees shows the Rockies standing up from the plains
        # at the opening zoom while still reading as a map rather than a diorama.
        # _pad() knows about pitch; view_to_bbox deliberately does not (see the note there).
        view_state={"longitude": -98.5, "latitude": 39.5, "zoom": 3.8, "pitch": 40},
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
    controls = Controls(class_options=CLASS_OPTIONS)

    # A NEW MAP INHERITS NOTHING ABOUT THE OLD ONE'S SCREEN. HOLD lives in a cell that
    # cannot re-run, which is what lets the camera survive; the cost is that a re-run of
    # THIS cell builds fresh layers and a fresh WANT while HOLD still describes the map
    # that just went away. Every field below is about that dead screen:
    #
    #   busy/pending  a fold from the previous run, still in flight against layers nobody
    #                 can see. Left set, `refresh` parks the forced opening draw in
    #                 `pending` and returns, so no fold ever runs on the new map, nothing
    #                 calls put_clusters, WANT["built"] stays False, and the cluster
    #                 checkboxes are dead: they set a trait on a layer whose `visible` is
    #                 gated on a wash that is never going to be built. THIS is "the fill
    #                 stops working after a re-run".
    #   res/box       "the screen is already correct for this view", which the forced
    #                 opening draw ignores but every later camera event does not.
    #   cache [1][2]  cells and wash tables built for the OLD layers.
    #
    # The raw folds stay: they are the expensive half, they describe the data rather than
    # the screen, and derive() rebuilds cells and wash from them in milliseconds.
    if HOLD["task"] is not None:
        HOLD["task"].cancel()
    HOLD["task"] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["res"], HOLD["box"] = None, None
    for _e in HOLD["cache"].values():
        _e[1], _e[2] = None, None

    def class_entry():
        """The menu's current entry, falling back to the DEFAULT and never to None.

        A `.get(key, (..., None))` here would be a trapdoor: None is the everything case,
        so any key the dict did not recognise would quietly draw all sixteen classes and
        look like the filter had failed rather than like a bad key. Falling back to the
        trait's own default keeps an unknown key visible as the wrong grouping instead of
        as no grouping.
        """
        return CLASS_SETS.get(
            controls.class_set, CLASS_SETS[Controls.class_set.default_value]
        )

    def class_keep():
        """The classes the menu is asking for, or None for everything."""
        return class_entry()[1]

    def class_label():
        return class_entry()[0]

    def count_note(n):
        keep = class_keep()
        return f"{n:,} cells" if keep is None else f"{n:,} {class_label().lower()} cells"

    def derive(raw, wash=True):
        """Raw fold -> (cells table, wash, how many cells survived the filter).

        The one place the class filter is applied, so the cells and the outlines can never
        disagree about what is on screen. An empty result gets the seed table rather than a
        zero-row one: lonboard rechunks on every assignment and an empty table has no
        chunk size to infer, and a hexagon at null island is invisible from anywhere in
        CONUS anyway.

        `wash=False` is the mid-drag case: the dissolve is the expensive half and a wash
        for a view that has already moved is wasted, so the caller skips it and
        ensure_wash picks it up on the next settle.
        """
        sel = filter_classes(raw, class_keep())
        if sel is None or sel.num_rows == 0:
            return seed_table(), None, 0
        return (
            to_layer_table(sel),
            to_cluster_table(sel, controls.min_cluster, CLUSTER_DARKEN) if wash else None,
            sel.num_rows,
        )
    # "clusters": does the user want them. "built": does the layer currently hold polygons
    # dissolved from the cells that are on screen right now. Visible needs both.
    WANT = {"clusters": True, "built": False}

    def apply_elevation():
        """Push the height slider onto the layer, scaled for the resolution on screen.

        elevation_scale is NOT the slider. The slider is a multiple of a base derived from
        the hexagon's own edge length, so that 1000 m of relief reads as about half a
        hexagon width at every resolution. Without that, one setting is a flat smear at
        res 5 and a forest of towers at res 11, and the slider has to be re-aimed on every
        zoom. HOLD["res"] is None before the first fold, in which case there is no hexagon
        size to scale against yet and the opening draw will call this again.
        """
        ex = float(controls.exaggeration)
        res = HOLD["res"]
        h3_layer.extruded = ex > 0
        h3_layer.elevation_scale = 0.0 if res is None else elev_base_scale(res) * ex

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
        apply_elevation()
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
        elif name == "exaggeration":
            apply_elevation()
        elif name == "min_cluster":
            rewash()
        elif name == "class_set":
            refilter()
        else:
            # Fill and outline are the two halves of the cluster layer. With both off there
            # is nothing left to draw, so the layer itself comes off and stays off until one
            # of them is asked for again.
            setattr(clusters, "filled" if name == "cluster_fill" else "stroked", val)
            WANT["clusters"] = controls.cluster_fill or controls.cluster_line
            # ASKING FOR THEM IS ALSO ASKING FOR THEM TO EXIST. This is why "fill does
            # nothing": every path that skips the dissolve leaves WANT["built"] False, and
            # ensure_wash only ran on a camera settle, so a click with no wash in hand set
            # filled=True on a layer that was invisible and stayed invisible until you
            # happened to move the map. With both halves off, ensure_wash refused for the
            # same reason, so turning one back on could never recover on its own.
            if WANT["clusters"]:
                ensure_wash()
            clusters.visible = WANT["clusters"] and WANT["built"]

    controls.observe(
        _on_controls,
        names=[
            "class_set",
            "exaggeration",
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

    def _pad(b, vs=None):
        # PITCH EATS THE PADDING. view_to_bbox is a top-down rectangle, which is exactly
        # right at pitch 0 and increasingly wrong as the camera tilts: a pitched view sees
        # toward the horizon, so the ground it covers is a trapezoid that runs well past
        # the flat box. Extrusion is the reason anyone tilts this map, so the two arrived
        # together. Growing the pad by sin(pitch) over-reads a little to the sides in
        # exchange for not starving the far half of a tilted view; the cap is because the
        # trapezoid diverges as pitch approaches 90 and no finite box would cover it.
        p = float(getattr(vs, "pitch", 0.0) or 0.0) if vs is not None else 0.0
        grow = PAD * (1.0 + 1.5 * math.sin(math.radians(min(p, 60.0))))
        dx, dy = (b[2] - b[0]) * (grow - 1) / 2, (b[3] - b[1]) * (grow - 1) / 2
        return (b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy)

    def _covers(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    def _show(tbl, res, box, note, keep_wash=False):
        """Put a table on the layer and record what it is. The ONLY place that paints.

        `keep_wash` is the difference between a clean swap and a flash. The cluster layer
        is a separate widget, so hiding it here and re-showing it in put_clusters is TWO
        comm messages, and the browser renders between them: one frame of new hexagons with
        no outlines, every single fold. When the caller already holds the new wash there is
        nothing to hide from, and the outlines simply change with the cells.
        """
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
        # Set BEFORE the batch below, because apply_elevation reads it to size the scale.
        HOLD["res"], HOLD["box"] = res, box
        with h3_layer.hold_sync():
            h3_layer.table = tbl
            h3_layer.get_hexagon = tbl["hex"]
            h3_layer.get_fill_color = tbl["color"]
            h3_layer.get_elevation = tbl["elev"]
            # In the SAME message as the table it belongs to. A resolution change moves the
            # hexagon size and the scale that compensates for it together, and if those
            # arrive as two messages one frame renders the new cells at the old
            # exaggeration: at a band boundary that is a 2.65x jump in apparent relief.
            apply_elevation()
        # THE OUTLINES DIE WITH THE CELLS THEY DESCRIBED, unless a replacement is already
        # in hand. They are dissolved from one fold's cells, so once different cells go up
        # they are stale, and stale outlines do not look stale: they are clean lines in the
        # right colours over the wrong place. But if the caller is about to put the new
        # ones on, hiding first only buys a frame of missing outlines.
        if not keep_wash:
            clusters.visible = False
            WANT["built"] = False
        status.value = note

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or clear and refold."""
        res = res_for_zoom(vs.zoom)
        _box_ll = view_to_bbox(vs)
        want_ll = _pad(_box_ll, vs)
        seen = HOLD["to_albers"](_box_ll)
        want = HOLD["to_albers"](want_ll)

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
            # The RAW fold outlives a filter change; the cells and the wash derived from it
            # do not, and refilter drops them everywhere but the level on screen. Coming
            # back to one of those levels re-derives instead of re-reading.
            if hit[1] is None:
                hit[1], hit[2], hit[4] = derive(hit[3])
            _show(
                hit[1],
                res,
                hit[0],
                f"<b>res {res}</b> · {count_note(hit[4])} · cached"
                f" · zoom {vs.zoom:.1f} · {HOLD['source']}",
                keep_wash=hit[2] is not None,
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

        # THE LAST ANSWER STAYS UP UNTIL THERE IS A NEW ONE. Nothing is cleared here: the
        # read happens under the cells that are already on screen, and the swap is a single
        # trait update when the new fold is complete.
        #
        # This reverses an earlier rule ("nothing may outlive its resolution") that blanked
        # to a seed table on every resolution change. Blanking made the wrong-size moment
        # short, but it replaced it with an EMPTY map for the whole read, which is worse:
        # a stale-but-plausible map reads as the map, an empty one reads as broken. The
        # outlines are still hidden, because those genuinely are wrong once the cells move.
        status.value = f"<b>reading…</b> res {res} · zoom {vs.zoom:.1f}"

        raw, m_px, read_px, dem_px = await HOLD["fold"](res, want, want_ll)
        if raw is None or raw.num_rows == 0:
            HOLD["res"], HOLD["box"] = res, want
            status.value = f"<b>res {res}</b> · nothing here · zoom {vs.zoom:.1f}"
            return
        # box, cells, wash, raw fold, cells shown. The raw one is kept so the wash
        # threshold and the class filter can both be changed without reading or refolding
        # anything; the count is kept because the layer may be holding the seed table when
        # the filter matched nothing, and 1 is not the answer.
        tbl, ctbl, n_shown = derive(raw, wash=HOLD["pending"] is None)
        HOLD["cache"][res] = [want, tbl, ctbl, raw, n_shown]
        # THE CELLS AND THE OUTLINES GO OUT TOGETHER, and the wash is built BEFORE either
        # is sent. The old order pushed the cells out first and let the outlines follow,
        # which made sense when the wash cost ~1 s of h3ronpy; in DuckDB it is ~215 ms, and
        # the previous fold now stays on screen for the whole read instead of blanking, so
        # waiting means the OLD map persists 215 ms longer rather than the new one arriving
        # half-dressed. Together with keep_wash below, that is what removes the flash of
        # bare hexagons.
        #
        # The skip survives: if the camera has already moved, this wash is for a view that
        # is gone, and ensure_wash picks it up on the next settle.
        _show(
            tbl,
            res,
            want,
            f"<b>res {res}</b> · {count_note(n_shown)} · {m_px:.0f} m"
            + (
                " · tiles cached"
                if read_px + dem_px == 0
                else f" · {read_px / 1e6:.2f}M lc + {dem_px / 1e6:.2f}M dem px fetched"
            )
            + f" · zoom {vs.zoom:.1f} · {HOLD['source']}"
            + (f" · <b style='color:#E69F00'>{HOLD['jumps']} jumps</b>" if HOLD["jumps"] else ""),
            keep_wash=ctbl is not None,
        )
        if HOLD["pending"] is not None:
            return  # camera moved while the wash was building; that view is gone
        if ctbl is None:
            clusters.visible = False
            # Built, in the sense that this fold HAS its answer and the answer is "no run
            # is that large". False here would make ensure_wash redo the dissolve on every
            # settle, for the one view where it is guaranteed to find nothing.
            WANT["built"] = True
            return
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
        ctbl = to_cluster_table(
            filter_classes(ent[3], class_keep()), controls.min_cluster, CLUSTER_DARKEN
        )
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

    def refilter():
        """Re-derive the CURRENT fold for the class menu. No read, no refold.

        Same bargain as rewash: the raw fold is in hand, so changing what is shown costs
        one filter and one dissolve. Every OTHER cached level has its cells and its wash
        dropped but keeps its raw fold, so zooming back to it re-derives (milliseconds)
        rather than re-reading (a round trip to the bucket) and can never paint the
        previous filter's cells.
        """
        for _r, _e in HOLD["cache"].items():
            if _r != HOLD["res"]:
                _e[1], _e[2] = None, None
        ent = HOLD["cache"].get(HOLD["res"])
        if not ent or ent[3] is None:
            return
        status.value = f"<b>{class_label().lower()}…</b>"
        tbl, ctbl, n = derive(ent[3])
        ent[1], ent[2], ent[4] = tbl, ctbl, n
        _show(
            tbl,
            HOLD["res"],
            HOLD["box"],
            f"<b>{class_label()}</b> · "
            + (f"{count_note(n)}" if n else "none of it in this view")
            + f" · res {HOLD['res']} · {HOLD['source']}",
            keep_wash=ctbl is not None,
        )
        if ctbl is None:
            clusters.visible = False
            WANT["built"] = True
            return
        put_clusters(ctbl)

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

    def _on_draw(change):
        # The box, plus the resolution the map was on when it was drawn. Captured here
        # rather than read later, so the analysis describes the view you drew it over
        # instead of wherever the camera has wandered since.
        b = change["new"]
        if b:
            set_aoi((tuple(float(v) for v in b), res_for_zoom(deck.view_state.zoom)))

    deck.observe(_on_draw, names="selected_bounds")
    deck.observe(_on_camera, names="view_state")
    return controls, deck, refresh, status


@app.cell
async def _(
    DEM_ZOOM_FOR_RES,
    Image,
    PM_BUCKET,
    PM_PATH,
    PM_TILE,
    S3Store,
    asyncio,
    gzip,
    io,
    math,
    np,
    obstore,
    struct,
):
    # THE TERRAIN READER. PMTiles is an XYZ pyramid inside ONE 705 GB object, addressed by
    # ranged GET: header, root directory, leaf directories, then tiles. That is already the
    # shape this notebook wants, which is why it beat every COG option considered. A COG
    # DEM at this coverage is either a single mosaic with a hand-built overview table or
    # ~1,500 one-degree files with no shared pyramid; here the pyramid IS the file, and
    # tile zoom maps onto H3 resolution directly (DEM_ZOOM_FOR_RES).
    #
    # Opening costs three reads: a 127-byte header, the root directory (6,612 B), then one
    # leaf directory per region touched. Directories are gzipped varint deltas, parsed once
    # and cached, so the 21.7 MB of leaf directories in the planet file are never read whole.
    _pm_store = S3Store(PM_BUCKET, region="us-west-2", skip_signature=True)

    async def _pm_range(a, b):
        """Inclusive byte range [a, b]. obstore's `end` is exclusive."""
        return bytes(
            memoryview(
                await obstore.get_range_async(_pm_store, PM_PATH, start=a, end=b + 1)
            )
        )

    def _varint(buf, i):
        r = s = 0
        while True:
            c = buf[i]
            i += 1
            r |= (c & 0x7F) << s
            if not c & 0x80:
                return r, i
            s += 7

    def _parse_dir(buf):
        """A PMTiles v3 directory: four varint columns, tile ids delta-encoded.

        Entries are (tile_id, offset, length, run_length). run_length 0 marks a pointer to
        a LEAF directory rather than to a tile, which is how one file indexes 9.4M tiles
        without reading a 21 MB index on open. A zero OFFSET means "immediately after the
        previous entry", so offsets have to be reconstructed in order, not read.
        """
        n, i = _varint(buf, 0)
        ids, last = [0] * n, 0
        for k in range(n):
            v, i = _varint(buf, i)
            last += v
            ids[k] = last
        runs = [0] * n
        for k in range(n):
            runs[k], i = _varint(buf, i)
        lens = [0] * n
        for k in range(n):
            lens[k], i = _varint(buf, i)
        offs = [0] * n
        for k in range(n):
            v, i = _varint(buf, i)
            offs[k] = (offs[k - 1] + lens[k - 1]) if v == 0 and k > 0 else v - 1
        return list(zip(ids, offs, lens, runs))

    def _tile_id(z, x, y):
        """z/x/y -> PMTiles v3 tile id: Hilbert order within a level, levels stacked.

        Hilbert rather than row-major so tiles that are near each other on the GROUND are
        near each other in the FILE. That is what makes a viewport read touch a few
        contiguous byte ranges instead of one seek per tile.
        """
        acc = sum((1 << t) * (1 << t) for t in range(z))
        n = 1 << z
        d, s = 0, n >> 1
        while s > 0:
            rx = 1 if x & s else 0
            ry = 1 if y & s else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            s >>= 1
        return acc + d

    def _find(entries, tid):
        """Binary search, falling back to the run that COVERS tid.

        The fallback is not an optimisation: directories are run-length encoded, so a tile
        usually has no entry of its own and is covered by an earlier one. Leaf pointers
        (run_length 0) always match this way too.
        """
        lo, hi = 0, len(entries) - 1
        while lo <= hi:
            m = (lo + hi) // 2
            if tid < entries[m][0]:
                hi = m - 1
            elif tid > entries[m][0]:
                lo = m + 1
            else:
                return entries[m]
        if hi >= 0 and (entries[hi][3] == 0 or tid - entries[hi][0] < entries[hi][3]):
            return entries[hi]
        return None

    _hdr = await _pm_range(0, 126)
    assert _hdr[:7] == b"PMTiles" and _hdr[7] == 3, "not a PMTiles v3 archive"
    _rd_off, _rd_len, _, _, _ld_off, _, _td_off, _ = struct.unpack("<8Q", _hdr[8:72])
    PM_MINZ, PM_MAXZ = _hdr[100], _hdr[101]
    # The zoom table is written against this file's pyramid, so it should fail loudly if
    # the two ever disagree rather than silently serve whatever _find lands on: a request
    # above max zoom resolves to SOME entry, so the map would fill with terrain from the
    # wrong scale instead of going blank.
    assert PM_MINZ <= min(DEM_ZOOM_FOR_RES.values()), "DEM_ZOOM_FOR_RES below the pyramid"
    assert max(DEM_ZOOM_FOR_RES.values()) <= PM_MAXZ, (
        f"DEM_ZOOM_FOR_RES wants z{max(DEM_ZOOM_FOR_RES.values())}, "
        f"{PM_PATH} stops at z{PM_MAXZ}"
    )
    _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
    _leaf = {}

    # Same bargain as the NLCD tile cache, for the same reason: a pan re-reads the strip it
    # has not seen and nothing else. The difference is that these tiles are the SOURCE's own
    # 512 px grid rather than one imposed on a COG window, so there is no snapping to do and
    # a tile is either held or it is not. float32 after decode, so ~1 MB resident per tile.
    _dem_tiles = {}
    _dem_held = {"bytes": 0}
    DEM_BUDGET = 512 * 1024 * 1024
    _dem_sem = asyncio.Semaphore(32)

    async def _dem_tile(z, x, y):
        """One tile, walked to through the directories and decoded to float32 metres."""
        tid, ents = _tile_id(z, x, y), _root
        for _ in range(4):  # root + up to three leaf levels
            e = _find(ents, tid)
            if e is None:
                return None
            if e[3] == 0:
                key = (e[1], e[2])
                if key not in _leaf:
                    _leaf[key] = _parse_dir(
                        gzip.decompress(
                            await _pm_range(_ld_off + e[1], _ld_off + e[1] + e[2] - 1)
                        )
                    )
                ents = _leaf[key]
                continue
            async with _dem_sem:
                blob = await _pm_range(_td_off + e[1], _td_off + e[1] + e[2] - 1)
            # Terrarium, straight off the RGB. Measured at 4 ms/tile against ~50 ms to
            # fetch one, so the decode this was feared to cost is not the cost.
            rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB")).astype(np.float32)
            return (rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0) - 32768.0
        return None

    async def _dem_window(z, tx0, ty0, tx1, ty1):
        """A rectangle of tiles, assembled. Returns (array, pixels actually fetched)."""
        want = [(z, tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        need = [k for k in want if k not in _dem_tiles]
        fetched = 0
        if need:
            got = await asyncio.gather(*(_dem_tile(*k) for k in need))
            for k, a in zip(need, got):
                # A missing tile is ocean or off-archive, not an error. NaN so the fold
                # drops it rather than folding a zero into somebody's mean elevation.
                if a is None:
                    a = np.full((PM_TILE, PM_TILE), np.nan, dtype=np.float32)
                _dem_tiles[k] = a
                _dem_held["bytes"] += a.nbytes
                fetched += a.size
            while _dem_held["bytes"] > DEM_BUDGET and len(_dem_tiles) > len(want):
                for k in list(_dem_tiles):
                    if k not in want:
                        _dem_held["bytes"] -= _dem_tiles.pop(k).nbytes
                        break
                else:
                    break
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        out = np.full((ny * PM_TILE, nx * PM_TILE), np.nan, dtype=np.float32)
        for k in want:
            a = _dem_tiles[k]
            r, c = (k[2] - ty0) * PM_TILE, (k[1] - tx0) * PM_TILE
            out[r : r + a.shape[0], c : c + a.shape[1]] = a
        for k in want:
            _dem_tiles[k] = _dem_tiles.pop(k)  # touch: young end of the LRU
        return out, fetched

    # WEB MERCATOR IS CLOSED FORM IN BOTH DIRECTIONS, and that is the whole reason the DEM
    # half of this notebook is shorter than the NLCD half. x is linear in longitude and y
    # is the inverse Gudermannian of latitude, so a tile pixel knows its own lat/lon exactly
    # with no projection library, no 64x64 control grid and no bilinear interpolation. The
    # Albers path upstream needs all three to say the same thing.
    def _px_to_lon(px, z):
        return px / (PM_TILE * (1 << z)) * 360.0 - 180.0

    def _px_to_lat(py, z):
        return np.degrees(
            np.arctan(np.sinh(np.pi * (1.0 - 2.0 * py / (PM_TILE * (1 << z)))))
        )

    def _lon_to_tx(lon, z):
        return (lon + 180.0) / 360.0 * (1 << z)

    def _lat_to_ty(lat, z):
        la = math.radians(max(min(lat, 85.05112), -85.05112))
        return (1.0 - math.log(math.tan(la) + 1.0 / math.cos(la)) / math.pi) / 2.0 * (1 << z)

    async def dem_read(res, box_ll):
        """The DEM for a lon/lat box at the zoom `res` deserves.

        Returns (elev, lats, lons, pixels fetched, metres per pixel). `lats` and `lons` are
        the pixel-centre coordinate VECTORS, which is exactly what xarray-sql wants for a
        dataset's coords, so the H3 UDF downstream reads them with no transform at all.
        """
        z = DEM_ZOOM_FOR_RES[res]
        w, s, e, n = box_ll
        span = 1 << z
        tx0 = max(0, int(math.floor(_lon_to_tx(w, z))))
        tx1 = min(span - 1, int(math.floor(_lon_to_tx(e, z))))
        ty0 = max(0, int(math.floor(_lat_to_ty(n, z))))
        ty1 = min(span - 1, int(math.floor(_lat_to_ty(s, z))))
        if tx1 < tx0 or ty1 < ty0:
            return None, None, None, 0, 0.0
        arr, fetched = await _dem_window(z, tx0, ty0, tx1, ty1)
        lons = _px_to_lon(tx0 * PM_TILE + np.arange(arr.shape[1]) + 0.5, z)
        lats = _px_to_lat(ty0 * PM_TILE + np.arange(arr.shape[0]) + 0.5, z)
        # Trim the tile overhang, so the fold does not carry up to a tile of margin on
        # every side into the group-by.
        ci = np.flatnonzero((lons >= w) & (lons <= e))
        ri = np.flatnonzero((lats >= s) & (lats <= n))
        if ci.size == 0 or ri.size == 0:
            return None, None, None, fetched, 0.0
        arr = arr[ri[0] : ri[-1] + 1, ci[0] : ci[-1] + 1]
        mpp = 78271.517 * math.cos(math.radians((s + n) / 2)) / span
        return arr, lats[ri[0] : ri[-1] + 1], lons[ci[0] : ci[-1] + 1], fetched, mpp

    return PM_MAXZ, PM_MINZ, dem_read


@app.cell
async def _(
    DEM_FLOOR,
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
    YEARS,
    asyncio,
    coordinates_to_cells,
    deck,
    dem_read,
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

    # THE NLCD FOOTPRINT, IN LON/LAT, AND IT IS A COST CONTROL. The DEM is global; the land
    # cover is CONUS, and the JOIN IS LEFT FROM THE LAND COVER, so any terrain read outside
    # that footprint is decoded and then thrown away. It goes unnoticed because it is
    # correct, just wasted. Measured on the opening draw before this clamp existed:
    # 37.75M DEM pixels against 4.10M NLCD, a 9x overread, because the padded box at res 5
    # runs out over the Pacific and up into Canada (and the pitch-aware pad makes it worse,
    # since tilting grows the box in every direction).
    #
    # Albers is curved, so the corners understate the envelope: walk the edges and take
    # the extremes, the same argument as _to_albers going the other way.
    _ee = np.linspace(0, 1, 33)
    _blon, _blat = _inv.transform(
        np.concatenate(
            [_l + _ee * (_r - _l), _l + _ee * (_r - _l), np.full(33, _l), np.full(33, _r)]
        ),
        np.concatenate(
            [np.full(33, _b), np.full(33, _t), _b + _ee * (_t - _b), _b + _ee * (_t - _b)]
        ),
    )
    LL_EXTENT = (
        float(np.nanmin(_blon)),
        float(np.nanmin(_blat)),
        float(np.nanmax(_blon)),
        float(np.nanmax(_blat)),
    )

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

    def _swap(name, tbl):
        """Register `tbl` under `name`, replacing whatever was there.

        One fixed name per role, so DataFusion holds ONE window at a time and RSS stays
        flat whatever the zoom. from_arrow refuses a name that is already taken, and the
        deregister raises when it is not, so both halves are guarded.
        """
        try:
            ctx.deregister_table(name)
        except Exception:
            pass
        ctx.from_arrow(tbl, name=name)

    async def fold(res, box, box_ll):
        """Read the window for `box` at the overview `res` deserves, then fold it to H3.

        `box` is Albers, for NLCD; `box_ll` is the same rectangle in lon/lat, for the DEM.
        Both rasters fold to H3 independently and are joined on the cell id at the end.

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
            return None, rd.res[0], 0, 0

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
        lc_cells = ctx.sql(f"""
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

        # ---- the OTHER raster, onto the same lattice --------------------------------
        # No control grid and no to_lat/to_lon here: mapterhorn is Web Mercator, so the
        # window's own coordinates ARE lat/lon and the H3 UDF reads them directly. This is
        # the entire difference between the two folds, and it is why this half is shorter.
        _w = max(box_ll[0], LL_EXTENT[0])
        _s = max(box_ll[1], LL_EXTENT[1])
        _e2 = min(box_ll[2], LL_EXTENT[2])
        _n = min(box_ll[3], LL_EXTENT[3])
        if _e2 > _w and _n > _s:
            elev, lats, lons, dem_px, _mpp = await dem_read(res, (_w, _s, _e2, _n))
        else:
            elev, lats, lons, dem_px = None, None, None, 0
        dem_cells = None
        if elev is not None:
            _swap_ds = xr.Dataset(
                {"elev": (("lat", "lon"), elev)}, coords={"lat": lats, "lon": lons}
            )
            try:
                ctx.deregister_table("dem")
            except Exception:
                pass
            ctx.from_dataset("dem", _swap_ds, chunks={"lat": 512})
            # `elev > DEM_FLOOR` is doing two jobs: it drops the NaN standing in for tiles
            # the archive has no data for (NaN fails every comparison), and it drops the
            # terrarium void below any real land surface. Ocean at 0 m is kept, because a
            # coastal cell that is genuinely at sea level should extrude to nothing rather
            # than drop out of the join and be COALESCEd to the same nothing by accident.
            dem_cells = ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS hex,
                       avg(elev) AS elev
                FROM dem WHERE elev > {DEM_FLOOR}
                GROUP BY 1
            """).to_arrow_table()

        if dem_cells is None or dem_cells.num_rows == 0:
            return (
                lc_cells.append_column(
                    "elev", pa.array(np.zeros(lc_cells.num_rows, dtype=np.float64))
                ),
                rd.res[0],
                fetched,
                dem_px if elev is not None else 0,
            )

        # THE JOIN, AND IT IS THE POINT. Two rasters that share no CRS, no grid and no
        # resolution, reconciled on a UBIGINT. LEFT from the land cover, because the land
        # cover decides which cells exist: the DEM is global and would otherwise contribute
        # cells over ocean and over Canada that NLCD has nothing to say about.
        #
        # COALESCE to 0 rather than dropping. At 8-24 px/hex a cell missing the DEM
        # entirely is rare, but it is not impossible at a window edge or over a tile the
        # archive has no data for, and a null there would punch a hole in the extrusion
        # rather than flatten one cell. Flat is the honest rendering of "no height known".
        _swap("lc_cells", lc_cells)
        _swap("dem_cells", dem_cells)
        out = ctx.sql("""
            SELECT c.hex, c.mode_cls, c.px_total, c.purity,
                   COALESCE(d.elev, 0.0) AS elev
            FROM lc_cells c
            LEFT JOIN dem_cells d ON d.hex = c.hex
        """).to_arrow_table()
        return out, rd.res[0], fetched, dem_px

    # ---------------------------------------------------------------- the AOI lane
    # Separate from the camera path on purpose. The camera reads ONE year and cares about
    # latency; a drawn box reads FORTY and cares about being complete. They share the store
    # and nothing else, so neither can stall the other.
    # TASKS, not results. Two coroutines asking for the same year at the same moment is
    # the normal case here (the prefetch is still running when the first box is drawn), and
    # a plain `if y not in dict` check would let both of them open it. Awaiting the same
    # task twice is free; opening the same COG twice is a wasted round trip each time.
    _tifs = {}

    def _tif_task(y):
        if y not in _tifs:
            _tifs[y] = asyncio.get_running_loop().create_task(
                GeoTIFF.open(
                    f"{PREFIX}/Annual_NLCD_LndCov_{y}_CU_C1V1.tif", store=_store
                )
            )
        return _tifs[y]

    async def _year_level(y, li):
        g = _g if y == YEAR else await _tif_task(y)
        return g if li == 0 else g.overviews[li - 1]

    async def prefetch_years():
        """Open the 40 COG headers while the map is being panned.

        3,979 ms measured, and it is pure latency: without it the first drawn box wears it.
        Headers only, no pixels.
        """
        await asyncio.gather(*(_year_level(y, 0) for y in YEARS))

    async def aoi_cube(box, res):
        """The AOI window at the level `res` uses, for every year, plus pixel centres.

        Returns (years, cube[year, y, x], lat, lon). The grid is identical across years, so
        the coordinates are computed ONCE and every year folds onto the same H3 cells: that
        is what makes a 40-year patch series comparable year to year rather than a set of
        forty differently-tiled answers.
        """
        li = LEVEL_FOR_RES[res]
        rd = _levels[li]
        L, B, R, T = rd.bounds
        H, W = rd.shape
        px_x, px_y = (R - L) / W, (T - B) / H
        x0, y0, x1, y1 = box
        col0 = max(0, int((max(x0, L) - L) / px_x))
        col1 = min(W, int(math.ceil((min(x1, R) - L) / px_x)))
        row0 = max(0, int((T - min(y1, T)) / px_y))
        row1 = min(H, int(math.ceil((T - max(y0, B)) / px_y)))
        wpx, hpx = col1 - col0, row1 - row0
        if wpx <= 0 or hpx <= 0:
            return None, None, None, None

        sem = asyncio.Semaphore(FETCH_AT_ONCE)

        async def one(y):
            lvl = await _year_level(y, li)
            async with sem:
                return np.asarray(
                    (
                        await lvl.read(
                            window=Window(
                                col_off=col0, row_off=row0, width=wpx, height=hpx
                            )
                        )
                    ).as_masked()[0]
                )

        cube = np.stack(await asyncio.gather(*(one(y) for y in YEARS)))

        wl, wt = L + col0 * px_x, T - row0 * px_y
        gx, gy = np.meshgrid(
            wl + (np.arange(wpx) + 0.5) * px_x, wt - (np.arange(hpx) + 0.5) * px_y
        )
        lon, lat = _inv.transform(gx.ravel(), gy.ravel())
        return YEARS, cube, lat.reshape(hpx, wpx), lon.reshape(hpx, wpx)

    async def aoi_dem(box_ll, res):
        """Mean elevation per H3 cell over the AOI. Returns (sorted cells, elevations).

        ONE read, not forty. The land cover has a year and the terrain does not, so this is
        the cheap half of the AOI: the same elevation column joins against every year, and
        that is exactly what makes "did this class move uphill" answerable at all. Sorted
        on the way out because the caller looks cells up with searchsorted.
        """
        elev, lats, lons, _f, _m = await dem_read(res, box_ll)
        if elev is None:
            return None, None
        lo, la = np.meshgrid(lons, lats)
        ok = np.isfinite(elev) & (elev > DEM_FLOOR)
        if not ok.any():
            return None, None
        cells = np.asarray(coordinates_to_cells(la[ok], lo[ok], res))
        order = np.argsort(cells, kind="stable")
        c, v = cells[order], elev[ok][order].astype(np.float64)
        uniq, start = np.unique(c, return_index=True)
        # Segmented mean without a groupby: reduceat sums each run of equal cell ids.
        return uniq, np.add.reduceat(v, start) / np.diff(np.append(start, len(v)))

    HOLD["aoi_dem"] = aoi_dem
    HOLD["aoi_cube"] = aoi_cube
    # Background, unawaited: the map must not wait on it. asyncio holds only a weak
    # reference to a bare task, hence the slot in HOLD.
    HOLD["prefetch"] = asyncio.get_running_loop().create_task(prefetch_years())

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
            # ABOVE THE MAP, because it is only useful before you reach for the wrong
            # button. Under the map it arrived after the mistake and was competing for the
            # line the legend already owns.
            mo.md(
                "<div style='font-size:.8rem;opacity:.75'>Fullscreen: use marimo's "
                "button, not the button in lonboard, to use the "
                "legend and layer controls.</div>"
            ),
            status,
            deck,
            mo.md(
                "<div style='display:flex;flex-wrap:wrap;font-size:.8rem;line-height:1.7'>"
                + "".join(_sw)
                # TWO LINES, AND THE BREAK IS PLACED, NOT LEFT TO THE WRAP. The console
                # has to fit the window with nothing to scroll, so this text owns exactly
                # two lines: one about colour, one about the box. `<br>` rather than a
                # blank line, which would start a second paragraph and add its margin to
                # a height that has none to give.
                + "</div>\n\nColour is the majority NLCD class in the cell; **height** is "
                "Mapterhorn terrain, joined on the cell id. Hover for both. Drag with "
                "**ctrl** to tilt; **height 0** is the flat map.<br>"
                "The **classes** menu picks what is drawn; it opens on forest. "
                "**Draw a box** (box button, bottom right) for 40 years of analytics below."
            ),
            controls,
        ],
        gap=0.15,
    )
    return


@app.cell
async def _(
    GROUPS,
    HOLD,
    NODATA,
    XarrayContext,
    YEARS,
    coordinates_to_cells,
    get_aoi,
    mo,
    np,
    pa,
    patch_stats,
    plt,
    xr,
):
    # DRAW A BOX AND THIS IS WHAT IT ANSWERS. Two questions, deliberately side by side:
    #
    #   AREA    what the AOI is made of, per year. A CELL question: sum the pixels.
    #   PATCHES how many separate runs that class comes in, and how big the biggest is.
    #           An OBJECT question, and one the cells cannot answer at any resolution.
    #
    # Composition can hold perfectly still while patch count triples. That is a forest
    # being cut into woodlots, and it is invisible in a class total. It is the reason the
    # dissolve exists for anything other than drawing.
    _aoi = get_aoi()
    if _aoi is None:
        _out = mo.md(
            f"**Draw a box on the map** to fold that area across all {len(YEARS)} years "
            f"of Annual NLCD: what it is made of, and how many pieces that comes in."
        )
    else:
        _box_lonlat, _res = _aoi
        _albers = HOLD["to_albers"](_box_lonlat)
        _years, _cube, _lat, _lon = await HOLD["aoi_cube"](_albers, _res)

    if _aoi is not None and _cube is None:
        _out = mo.md("**That box is outside the raster.**")
    elif _aoi is not None:
        # THE FOLD, IN SQL, OVER A CUBE. The year is a dimension, not forty separate
        # queries: this is the xarray-sql premise doing the thing it is for.
        _ctx = XarrayContext()
        _ctx.from_dataset(
            "aoi",
            xr.Dataset(
                {"cls": (("year", "y", "x"), _cube)},
                coords={
                    "year": np.asarray(_years),
                    "y": np.arange(_cube.shape[1]),
                    "x": np.arange(_cube.shape[2]),
                },
            ),
            chunks={"y": 256},
        )
        AOI_SQL = f"""
            SELECT year,
                   cls,
                   count(*)                                        AS px,
                   count(*) * 100.0 / sum(count(*)) OVER (PARTITION BY year) AS pct
            FROM aoi
            WHERE cls != {NODATA}
            GROUP BY year, cls
            ORDER BY year, cls
        """
        _comp = _ctx.sql(AOI_SQL).to_arrow_table().combine_chunks()

        # PATCHES. The H3 cells are computed ONCE, because the pixel grid does not move
        # between years, so a patch count in 1985 and in 2024 is counted over the same
        # lattice and the two are comparable.
        _cells = np.asarray(
            coordinates_to_cells(_lat.ravel(), _lon.ravel(), _res)
        )
        _uniq, _idx = np.unique(_cells, return_inverse=True)

        # TERRAIN, ON THE SAME CELLS. Read once, outside the year loop.
        _dh, _de = await HOLD["aoi_dem"](_box_lonlat, _res)
        _cell_elev = np.full(len(_uniq), np.nan)
        if _dh is not None:
            _p = np.clip(np.searchsorted(_dh, _uniq), 0, len(_dh) - 1)
            _hit = _dh[_p] == _uniq
            _cell_elev[_hit] = _de[_p[_hit]]

        _rows, _erows = [], []
        for _i, _y in enumerate(_years):
            _flat = _cube[_i].ravel()
            _ok = _flat != NODATA
            if not _ok.any():
                continue
            # Majority class per H3 cell, without a groupby: one bincount over
            # cell * 256 + class, then argmax along the class axis.
            _hist = np.bincount(
                _idx[_ok].astype(np.int64) * 256 + _flat[_ok],
                minlength=len(_uniq) * 256,
            ).reshape(len(_uniq), 256)
            _seen = _hist.sum(1) > 0
            _mode = _hist.argmax(1)[_seen].astype(np.int16)
            for _c, (_n, _p, _big) in patch_stats(_uniq[_seen], _mode).items():
                _rows.append((int(_y), _c, _n, _p, _big))
            # THE QUESTION THE SINGLE-RASTER VERSION CANNOT ASK. Where a class SITS, not
            # just how much of it there is. The 95th percentile is the upper edge of a
            # class's range rather than its maximum, which one stray misclassified cell on
            # a summit would otherwise define; for forest that upper edge is a treeline.
            _ez = _cell_elev[_seen]
            for _c in np.unique(_mode):
                _m2 = (_mode == _c) & np.isfinite(_ez)
                if _m2.sum() >= 5:
                    _erows.append(
                        (int(_y), int(_c), float(np.mean(_ez[_m2])),
                         float(np.percentile(_ez[_m2], 95)))
                    )
        _patch = pa.table(
            {
                "year": pa.array([r[0] for r in _rows], pa.int32()),
                "cls": pa.array([r[1] for r in _rows], pa.int16()),
                "cells": pa.array([r[2] for r in _rows], pa.int32()),
                "patches": pa.array([r[3] for r in _rows], pa.int32()),
                "largest": pa.array([r[4] for r in _rows], pa.int32()),
            }
        )
        _elev = pa.table(
            {
                "year": pa.array([r[0] for r in _erows], pa.int32()),
                "cls": pa.array([r[1] for r in _erows], pa.int16()),
                "mean_m": pa.array([r[2] for r in _erows], pa.float64()),
                "p95_m": pa.array([r[3] for r in _erows], pa.float64()),
            }
        )

        # ---- stitch the two into one per-class summary, first year vs last
        _cy = np.asarray(_comp["year"]); _cc = np.asarray(_comp["cls"])
        _cp = np.asarray(_comp["pct"])
        _py = np.asarray(_patch["year"]); _pc = np.asarray(_patch["cls"])
        _pn = np.asarray(_patch["patches"])
        _ey = np.asarray(_elev["year"]); _ec = np.asarray(_elev["cls"])
        _em = np.asarray(_elev["mean_m"]); _ep = np.asarray(_elev["p95_m"])
        _y0, _y1 = int(min(_years)), int(max(_years))

        def _at(ys, cs, vs, y, c):
            m = (ys == y) & (cs == c)
            return float(vs[m][0]) if m.any() else 0.0

        _classes = sorted(
            {int(c) for c in _cc},
            key=lambda c: -_at(_cy, _cc, _cp, _y1, c),
        )
        _rows_md = [
            "| class | "
            f"{_y0} area | {_y1} area | change | {_y0} patches | {_y1} patches | change "
            f"| mean elev | upper edge | shift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _c in _classes:
            _a0, _a1 = _at(_cy, _cc, _cp, _y0, _c), _at(_cy, _cc, _cp, _y1, _c)
            if max(_a0, _a1) < 0.5:
                continue  # under half a percent of the box in both years
            _q0, _q1 = _at(_py, _pc, _pn, _y0, _c), _at(_py, _pc, _pn, _y1, _c)
            _h1 = _at(_ey, _ec, _em, _y1, _c)
            _u1 = _at(_ey, _ec, _ep, _y1, _c)
            _u0 = _at(_ey, _ec, _ep, _y0, _c)
            _rows_md.append(
                f"| {GROUPS.get(_c, (str(_c),))[0]} | {_a0:.1f}% | {_a1:.1f}% | "
                f"{_a1 - _a0:+.1f} | {_q0:.0f} | {_q1:.0f} | {_q1 - _q0:+.0f} | "
                + (
                    f"{_h1:,.0f} m | {_u1:,.0f} m | {_u1 - _u0:+,.0f} m |"
                    if _u0 and _u1
                    else "· | · | · |"
                )
            )

        # ---- small multiples: area and patches per class, on their own panels.
        # One panel per class, NOT sixteen lines in sixteen colours: the classes are told
        # apart by position and label, so nothing here depends on telling two hues apart.
        _show = [c for c in _classes if max(
            _at(_cy, _cc, _cp, _y0, c), _at(_cy, _cc, _cp, _y1, c)) >= 0.5][:8]
        _fig = None
        if _show:
            # Three Okabe-Ito hues, one per ROW, and the row is already named by its
            # ylabel and fixed by its position: the colour is a reminder, never the only
            # way to tell the panels apart. Blue / orange / reddish-purple survive a
            # deuteranope simulation as a set, which no red-green pair does.
            _fig, _axes = plt.subplots(
                3, len(_show), figsize=(2.05 * len(_show), 6.0), sharex=True, squeeze=False
            )
            for _j, _c in enumerate(_show):
                _m = _cc == _c
                _ax = _axes[0][_j]
                _ax.plot(_cy[_m], _cp[_m], lw=1.6, color="#0072B2")
                _ax.set_title(GROUPS.get(_c, (str(_c),))[0], fontsize=8)
                _ax.tick_params(labelsize=7)
                _ax.set_ylim(bottom=0)
                _m = _pc == _c
                _ax = _axes[1][_j]
                _ax.plot(_py[_m], _pn[_m], lw=1.6, color="#E69F00")
                _ax.tick_params(labelsize=7)
                _ax.set_ylim(bottom=0)
                # Elevation does NOT start at zero. The others are counts, where zero is
                # the meaningful floor; this is a height above sea level, and pinning it to
                # zero would flatten a 60 m treeline shift into an invisible wiggle at the
                # top of a 2,000 m axis.
                _m = _ec == _c
                _ax = _axes[2][_j]
                _ax.plot(_ey[_m], _ep[_m], lw=1.6, color="#CC79A7")
                _ax.tick_params(labelsize=7)
            _axes[0][0].set_ylabel("% of AOI", fontsize=8)
            _axes[1][0].set_ylabel("patches", fontsize=8)
            _axes[2][0].set_ylabel("upper edge, m", fontsize=8)
            _fig.tight_layout()

        _out = mo.vstack(
            [
                mo.md(
                    f"### AOI · res {_res} · {_cube.shape[2]}x{_cube.shape[1]} px/year "
                    f"· {len(_years)} years"
                ),
                mo.md("\n".join(_rows_md)),
                _fig if _fig is not None else mo.md(""),
                mo.md(
                    "Top row is **area**, the cell question. Middle row is **patches**, the "
                    "object question: a class can hold its share of the box while breaking "
                    "into more pieces, and only the dissolve sees that. Bottom row is the "
                    "**upper edge**, the 95th percentile of the elevation a class occupies, "
                    "which is the question neither raster can answer alone. For forest it "
                    "is a treeline; for crops it is where the ground gets too steep."
                ),
                mo.accordion({"the SQL": mo.md(f"```sql{AOI_SQL}```")}),
            ],
            gap=0.5,
        )

    _out
    return


if __name__ == "__main__":
    app.run()
