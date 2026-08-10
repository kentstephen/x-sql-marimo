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
#     "geoarrow-rust-core",
#     "geoarrow-rust-io",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
#     "matplotlib==3.11.1",
# ]
# ///
"""Global deforestation 2002-2022, folded to H3 and joined onto Overture divisions.

Vizzuality's `deforest_100m_cog.tif` is one 5.7 GB COG covering the planet at 100 m. Its
value is the PORTION OF EACH CELL deforested between 2002 and 2022: an intensive 0-1
quantity. That single fact decides most of this notebook. A portion can be averaged at any
scale, so `mean()` is valid at every H3 resolution and the COG's averaged overview pyramid
is legitimate rather than a lie. No majority vote, no mode, no class fold.

WHAT EACH ENGINE DOES:

  obstore      streams the COG and the Overture GeoParquet, unsigned. Nothing is cached
               to disk; a viewport reads what it needs and keeps it in memory.
  DataFusion   the fold (pixels -> H3 cells) AND the join (cells -> divisions). The join
               is an integer equi-join on a UBIGINT cell id plus a group-by, which is what
               a query engine is for.
  DuckDB h3    the polyfill only: division polygon -> the cells covering it. The one step
               neither DataFusion nor plain SQL can do.
  lonboard     the render.

WHY H3 IS NOT JUST A DEMO STEP. The COG is EPSG:4326, so its pixels are not equal area: a
100 m pixel at the equator covers about twice the ground of one at 60 degrees. Averaging
pixels directly over a country spanning many latitudes overweights its poleward end. H3
cells are near-equal-area, so folding to H3 and then averaging CELLS equally is an
area-weighted mean almost for free. Pixel count weights WITHIN a cell (a coastal cell is
mostly NaN ocean and should not count as a full one); cells are equal-weighted within a
division. Two weightings, each correcting a different bias.

THE COG IS SPARSE AND async-geotiff DOES NOT KNOW IT. 73.6% of full-resolution tiles have
offset 0 and length 0, because ocean is not stored, and a read touching one issues a byte
range 0..0 and raises `Invalid range requested, start: 0 end: 0`. Reading on the COG's own
512 px tile grid and consulting `ifd.tile_byte_counts` first turns that from a crash into
a speedup: an empty tile is NaN with no request at all.

COLOUR. 69.6% of res-4 cells are exactly zero and the nonzero values span nine orders of
magnitude (p1 7.3e-8, p50 2.1e-3, p99.9 0.45), so a linear 0-1 ramp paints a blank world.
Zero takes its own dark swatch and the rest is log10 over 1e-4 to 0.5, which is p25 to
p99.9. See the ramp cell for why zero is separated by luminance and not by hue.

DRAW A BOX AND THE JOIN BECOMES A NUMBER. The ▢ button at the lower right of the map ranks
every division inside the box you draw by its mean share deforested. It reads one H3
resolution finer than the screen does, sizes that resolution from the BOX rather than the
current zoom, and falls back county -> region -> country, because Overture has counties for
only 171 of 219 countries. This is the one output here that is a figure rather than a colour.

THE CAMERA ANSWERS FROM MEMORY FIRST. `view_state` fires on every frame of a drag, and any
frame that can be served from what is already folded (a pan inside the current box, a zoom
back to a resolution already visited) is answered synchronously in the comm handler. Only a
view that genuinely needs bytes goes through the debounce. See `_instant`.

Data: Vizzuality / LandGriffon, CC-BY 4.0, on source.coop. Boundaries: Overture Maps.
Run:  uv run marimo edit xsql-deforest-divisions.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import json
    import math
    import pathlib

    import anywidget
    import traitlets
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import xarray as xr
    from arro3.core import Array as ArroArray, Table as ArroTable
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from geoarrow.rust.core import from_wkb, multipolygon, to_wkb
    from geoarrow.rust.io import GeoParquetFile
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext
    from lonboard import Map, H3HexagonLayer, PolygonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroArray,
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        GeoParquetFile,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        S3Store,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        from_wkb,
        infer_rows_per_chunk,
        json,
        math,
        matplotlib,
        mo,
        multipolygon,
        np,
        pa,
        pathlib,
        pc,
        pq,
        to_wkb,
        traitlets,
        udf,
        xr,
    )


@app.cell
def _(duckdb):
    # ONE JOB: polygon -> H3 cells. Everything else that could plausibly live here does not.
    #
    # The fold and the join are DataFusion's. The fold because it is a whole-column
    # operation, where h3ronpy converts a column at once and DuckDB would call a UDF per
    # row: 70 ms against 462 ms on 1.58M rows, measured in xsql-duckdb-nlcd-h3.py. The join
    # because it is an ordinary equi-join on an integer, with no geometry in sight.
    #
    # The polyfill is the opposite regime, which is why it is here. There are 219 countries
    # or a few thousand counties, so per-row call overhead is irrelevant and the work is all
    # inside the H3 library: the regime where Uber's C won the dissolve comparison, 75 ms
    # against h3ronpy's 2,784 ms.
    #
    # Extensions download once into ~/.duckdb and are cached after that.
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class Status(anywidget.AnyWidget):
        """A one-line status readout the camera can write to.

        A widget rather than `mo.md`, because the only way to update marimo output is to
        re-run the cell that produced it, and the cell holding the map is downstream of any
        state the camera could write: re-running it rebuilds the Map and throws the view
        away. A widget trait syncs straight to the browser instead.
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

    class Controls(anywidget.AnyWidget):
        """Layer switches, under the map next to the legend.

        Same constraint as Status: an `mo.ui.checkbox` would make the map cell depend on it,
        so every click would rebuild the Map and reset the camera. A widget trait syncs to
        the kernel, a Python observer assigns onto the deck layers, and nothing re-runs.
        """

        _esm = """
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.2rem 0 0;" +
            "user-select:none";
          const check = (key, label) => {
            const l = document.createElement("label");
            l.style.cssText =
              "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
            const c = document.createElement("input");
            c.type = "checkbox";
            c.checked = model.get(key);
            c.onchange = () => { model.set(key, c.checked); model.save_changes(); };
            model.on("change:" + key, () => { c.checked = model.get(key); });
            l.appendChild(c);
            l.appendChild(document.createTextNode(label));
            box.appendChild(l);
          };
          check("show_cells", "hexagons");
          check("show_divisions", "boundaries");
          check("division_fill", "boundary fill");
          el.appendChild(box);
        }
        export default { render };
        """
        show_cells = traitlets.Bool(True).tag(sync=True)
        show_divisions = traitlets.Bool(True).tag(sync=True)
        # ON BY DEFAULT. The join onto Overture is the whole point of this notebook and the
        # choropleth is what it produces, so shipping it behind an unticked box meant the
        # result was invisible unless you went looking for it.
        division_fill = traitlets.Bool(True).tag(sync=True)

    class Panel(anywidget.AnyWidget):
        """A block of HTML the kernel can rewrite, for the drawn-box ranking.

        Status is a one-line strip and this is a table, but the reason for both is the same:
        marimo output only updates by re-running the cell that made it, and the cell that
        made this one owns the Map.
        """

        _esm = """
        function render({ model, el }) {
          const box = document.createElement("div");
          const draw = () => { box.innerHTML = model.get("value"); };
          draw();
          model.on("change:value", draw);
          el.appendChild(box);
        }
        export default { render };
        """
        value = traitlets.Unicode("").tag(sync=True)

    return Controls, Panel, Status


@app.cell
def _(math):
    # ------------------------------------------------------------------ the raster
    SOURCE_BUCKET = "us-west-2.opendata.source.coop"
    COG = "vizzuality/lg-land-carbon-data/deforest_100m_cog.tif"

    # The COG's own tile size, at every level. Reading on this grid is what makes a read
    # shareable between viewports AND what lets the sparse-tile check work, since a tile is
    # the unit that is either present or absent.
    TILE = 512
    FETCH_AT_ONCE = 32  # tiles are only faster than one ranged read if they fly together
    TILE_BUDGET = 256 * 1024 * 1024  # float32, so ~1,000 tiles resident

    # WHICH OVERVIEW EACH H3 RESOLUTION READS. The pyramid is 100 m native and doubles ten
    # times: L0 100 m, L1 200, L2 400, L3 800, L4 1.6 km, L5 3.2, L6 6.4, L7 12.8, L8 25.6,
    # L9 51, L10 102.
    #
    # Chosen so 20-80 pixels sit under every cell: enough for a mean to mean something,
    # without reading pixels the cell will only average away.
    #   res 4 (1,770 km2) / L6 (40.7 km2)  = 43 px
    #   res 5 (  253 km2) / L5 (10.2 km2)  = 25 px
    #   res 6 ( 36.1 km2) / L3 (0.64 km2)  = 56 px
    #   res 7 ( 5.16 km2) / L2 (0.16 km2)  = 32 px
    #   res 8 (0.737 km2) / L1 (0.04 km2)  = 18 px
    #
    # Reading an overview is only equivalent to reading pixels if the pyramid AVERAGES, and
    # that was verified rather than assumed: over one 1-degree box the mean survives a 64x
    # downsample (0.2260 -> 0.2342) while the max collapses (1.0 -> 0.65) and the
    # exact-zero fraction goes 62% -> 0%. That is the signature of average resampling.
    LEVEL_FOR_RES = {4: 6, 5: 5, 6: 3, 7: 2, 8: 1}

    # ------------------------------------------------------------------ the zoom ladder
    # One H3 resolution per 1.4 zoom levels, because each H3 step is 2.65x linear and
    # log2(2.65) = 1.4. That keeps a hexagon a constant size ON SCREEN.
    #
    # math.floor, NOT int(): int truncates toward zero, so every zoom below ZOOM0 would
    # collapse onto BASE_RES instead of continuing down to MIN_RES.
    ZOOM0, PER_RES, BASE_RES = 4.0, 1.4, 4
    MIN_RES, MAX_RES = 4, 8

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # RES 4 COMFORTABLY DRAWS THE WHOLE PLANET: H3 res 4 is 288,122 cells globally, of
    # which ~224k carry data. Measured world fold at res 4: 15.7M pixels read in 821 ms
    # (68 tiles fetched, 10 skipped as sparse), folded in 282 ms.

    # WHICH DIVISION LEVEL IS DRAWN AT WHICH ZOOM, AND WHY THERE IS NONE AT THE TOP.
    #
    # Below DIV_ZOOM there are no boundaries at all: the hexagons carry the map alone. That
    # is a performance decision as much as a design one, and it is the one place where
    # streaming Overture per viewport genuinely does not work.
    #
    # `read_async(bbox=...)` prunes ROW GROUPS, and that is the only pruning available
    # here: the file-level index is useless for divisions because 7 of the 8 files have a
    # bbox wider than 130 degrees, so nearly every viewport hits nearly every file. Row
    # group pruning is excellent when the box is small and buys exactly nothing when the
    # box is the world. So a world view of countries means reading most of 5.5 GB to find
    # 219 rows, and `subtype` is not partitioned, so there is no cheaper way to ask.
    #
    # Zooming in inverts that completely, which is the same inversion the raster read runs
    # on: the tighter the view, the less there is to read. Boundaries arrive when the view
    # is tight enough for them to be both cheap and meaningful.
    #
    # Overture has counties for 171 of 219 countries, so the county band is genuinely empty
    # in places rather than merely sparse.
    DIV_ZOOM = 4.5

    # TODO: a fourth band, `locality`, above roughly zoom 9.5. Overture has 57,072
    # localities in a single division_area file against 11,375 counties, so the row-group
    # pruning has to be doing real work before it is affordable, which it is by that zoom.
    # The open question is not cost but meaning: a locality boundary is a settlement, so
    # most of a drawn box would fall outside every polygon and the ranking would describe
    # the towns rather than the ground. Decide that before adding the band.
    def division_for_zoom(z):
        if z < DIV_ZOOM:
            return None
        if z < 7.0:
            return "region"
        return "county"

    DIVISION_LABEL = {"country": "countries", "region": "regions", "county": "counties"}

    # ------------------------------------------------------------------ boundaries
    OVERTURE_BUCKET = "overturemaps-us-west-2"
    OVERTURE_RELEASE = "2026-07-22.0"
    DIVISION_PREFIX = (
        f"release/{OVERTURE_RELEASE}/theme=divisions/type=division_area"
    )

    # ------------------------------------------------------------------ view
    # The map's pixel size, assumed. It only sets how much of the world the viewport box
    # covers, and PAD is loose, so being wrong by a few hundred pixels costs a slightly
    # larger query and nothing else.
    VIEW_W, VIEW_H = 1400, 620
    PAD = 1.25

    # SETTLE ONLY GUARDS A READ. Every camera event that can be answered from memory (a pan
    # inside the box already folded, a zoom back to a resolution already visited) is now
    # answered synchronously in the comm handler, so this delay is never spent on a view the
    # notebook already knows the answer to. It exists purely so a two-second drag issues one
    # object-store read at the end instead of a hundred along the way.
    SETTLE = 0.15

    # The fill alpha for a division choropleth. Not 255: the hexagons stay legible underneath
    # and the boundary layer reads as a wash over them rather than a lid on them.
    FILL_ALPHA = 165

    # Opens on the tropics, because that is where the data is: the Amazon, the Congo basin
    # and insular southeast Asia are the three places a 2002-2022 deforestation layer has
    # anything dramatic to say, and all three are in view from here.
    HOME = {"longitude": -20.0, "latitude": 0.0, "zoom": 2.4}
    return (
        COG,
        DIVISION_LABEL,
        DIVISION_PREFIX,
        FETCH_AT_ONCE,
        FILL_ALPHA,
        HOME,
        LEVEL_FOR_RES,
        MAX_RES,
        OVERTURE_BUCKET,
        PAD,
        SETTLE,
        SOURCE_BUCKET,
        TILE,
        TILE_BUDGET,
        VIEW_H,
        VIEW_W,
        division_for_zoom,
        res_for_zoom,
    )


@app.cell
def _(matplotlib, np):
    # THE LOG RAMP.
    #
    # 69.6% of res-4 cells are exactly zero and the nonzero part spans nine orders of
    # magnitude, so a linear 0-1 ramp is a blank map. LO..HI is p25..p99.9 of the nonzero
    # values, which spends the whole ramp on the part of the data that varies.
    LO, HI = 1e-4, 0.5

    # THE ZERO SWATCH IS SEPARATED BY LUMINANCE, NOT HUE, AND THAT IS FORCED.
    #
    # The obvious flat neutral grey (78, 80, 84) lands at luminance 0.313, and the 0.1%
    # stop of full-range cividis lands at 0.318. Measured, not guessed: "none" and "0.1%"
    # came out the same colour, which is the worst thing this legend could do given zero is
    # the majority case. Hue cannot fix it, because the entire point of cividis is that hue
    # carries no information. So the ramp's floor is LIFTED off the bottom of cividis and
    # zero takes the dark end alone.
    #
    # FLOOR = 0.25 truncates cividis to its upper 75%, putting the ramp's darkest colour at
    # luminance 0.305 against the zero swatch's 0.156. Nothing is lost: the ramp still
    # spans 0.305 -> 0.874, more luminance range than most sequential maps get.
    #
    # cividis rather than viridis: both are colourblind-safe, but cividis is built for it.
    # It is strictly two-hue (blue -> yellow) and monotonic in luminance, and a deuteranope
    # simulation of these exact stops is monotonic too, so the ORDER survives, which is the
    # only thing a sequential ramp has to promise.
    FLOOR = 0.25
    ZERO_RGB = (38, 40, 44)
    _CIVIDIS = matplotlib.colormaps["cividis"]

    def ramp(v):
        """portion -> uint8 RGB, with exact zero (and NaN) taking the dark swatch."""
        v = np.asarray(v, dtype="float64")
        live = np.isfinite(v) & (v > 0)
        t = np.zeros(v.shape)
        if live.any():
            t[live] = (np.log10(np.clip(v[live], LO, HI)) - np.log10(LO)) / (
                np.log10(HI) - np.log10(LO)
            )
        out = (_CIVIDIS(FLOOR + t * (1 - FLOOR))[..., :3] * 255).astype(np.uint8)
        out[~live] = ZERO_RGB
        return out

    def ramp_rgba(v, alpha):
        """`ramp` with a constant alpha appended, as uint8 RGBA.

        The division fill needs four channels and the hexagons need three, and they must
        agree colour for colour: a division and the cells inside it are the same number
        drawn twice, so any drift between the two ramps would read as a disagreement in the
        data.
        """
        rgb = ramp(v)
        out = np.empty(rgb.shape[:-1] + (4,), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = alpha
        return out

    # 1e-4 is the ramp floor, so anything under it is "below 0.01%", not zero.
    STOPS = [
        (0.0, "none"),
        (1e-4, "0.01%"),
        (1e-3, "0.1%"),
        (1e-2, "1%"),
        (5e-2, "5%"),
        (1e-1, "10%"),
        (2.5e-1, "25%"),
        (5e-1, "50%+"),
    ]
    return STOPS, ramp, ramp_rgba


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs every
    # downstream cell, including the one that owns the Map, so the Map would be rebuilt with
    # its opening view_state and the camera would snap home on every pan. A plain dict is
    # invisible to the dataflow graph.
    HOLD = {
        "fold": None,  # the SQL fold, set by the read cell
        "zonal": None,  # cells -> division means, set by the read cell
        "rank": None,  # drawn box -> divisions ranked, set by the read cell
        "res": None,  # H3 resolution currently on screen
        "box": None,  # padded degree box the current cells cover
        "div": None,  # division subtype currently on screen
        # The box the DIVISIONS cover, tracked apart from the cells' box because they are
        # fetched second and a camera move can land between the two. Without it, a pan that
        # interrupts a fold leaves `div` set while the boundaries on screen belong to the
        # previous place, and the instant path then matches and never refetches them.
        "divbox": None,
        "cache": {},  # res -> [box, layer table, raw fold]
        "divpair": None,  # (fill-on table, fill-off table) currently on the division layer
        # The status line in two halves, so a camera move that reads nothing can still
        # refresh the zoom readout without throwing away what the last read said. Zooming IN
        # always lands inside the box the last read covered, so without this the numbers
        # would freeze exactly when the map feels least responsive.
        "head": "",  # what the cells are
        "tail": "",  # what the divisions are
        "vs": None,  # the last camera acted on, for the echo check
        "busy": False,
        "pending": None,
        "loop": None,
        "task": None,
        "seltask": None,  # the drawn-box ranking, which runs on its own
    }
    return (HOLD,)


@app.cell
def _(ArroArray, ArroTable, FILL_ALPHA, coordinates_to_cells, np, pa, ramp, ramp_rgba):
    def cells_to_layer(tbl):
        """Folded cells -> the arro3 table the H3HexagonLayer draws.

        combine_chunks because DataFusion returns many chunks while the numpy-derived
        colour column is one, and lonboard rejects a table whose columns disagree about
        chunking. ArroTable rather than pyarrow because the layer's `table` trait coerces
        in __init__ but its validate() is a strict isinstance check, so assigning
        afterwards needs the real type.
        """
        tbl = tbl.combine_chunks()
        portion = np.asarray(tbl["portion"])
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(ramp(portion).ravel()), 3
                    ),
                    # Percent, because "0.043 of a cell" is not how anyone reads this, and
                    # the tooltip is the one place the number is stated outright.
                    "deforested %": pa.array(np.round(portion * 100, 4)),
                    "pixels": tbl["px_total"],
                }
            )
        )

    def divisions_to_layer(tbl, from_wkb, multipolygon):
        """Division zonal means -> the TWO tables the PolygonLayer swaps between.

        Two tables, identical except for the alpha in `color`, and that is the fix for the
        dead "boundary fill" checkbox. `filled` is left permanently True (flipping it does
        not reliably build the fill sublayer, per CLAUDE.md), but the previous attempt then
        swapped `get_fill_color` between a TABLE COLUMN and the constant `[0, 0, 0, 0]`, and
        that swap is what never took: deck was being handed two different KINDS of accessor
        for one prop, and the layer only ever picked up whichever it saw first. Here both
        states are the same column of the same schema, and the toggle re-pushes the table,
        which is the one update path this layer has always honoured.

        Geometry comes straight off the Arrow column via from_wkb: to_pylist() here would
        materialise every polygon as a Python bytes object on the way past, and it is built
        once and shared by both tables.
        """
        tbl = tbl.combine_chunks()
        portion = np.asarray(tbl["portion"], dtype="float64")
        geom = ArroArray.from_arrow(
            from_wkb(
                tbl["wkb"].combine_chunks(), to_type=multipolygon("xy", crs="EPSG:4326")
            )
        )
        rest = [
            ArroArray.from_arrow(tbl["name"].combine_chunks()),
            ArroArray.from_arrow(tbl["country"].combine_chunks()),
            ArroArray.from_arrow(pa.array(np.round(portion * 100, 4))),
            ArroArray.from_arrow(tbl["n_cells"].combine_chunks()),
        ]
        names = ["geometry", "color", "name", "country", "deforested %", "cells"]

        def build(alpha):
            col = ArroArray.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(ramp_rgba(portion, alpha).ravel()), 4
                )
            )
            return ArroTable.from_arrays([geom, col, *rest], names=names)

        return build(FILL_ALPHA), build(0)

    def seed_cells():
        """One hexagon at null island so the Map has a valid table at build time.

        This is what lets the Map cell depend on nothing, and therefore never wait for the
        raster read. The first camera event replaces it.
        """
        hexes = coordinates_to_cells(np.array([0.0]), np.array([0.0]), 4)
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": pa.array(hexes),
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([13, 17, 23], dtype=np.uint8)), 3
                    ),
                    "deforested %": pa.array([0.0]),
                    "pixels": pa.array([0], type=pa.int64()),
                }
            )
        )

    def seed_divisions(from_wkb, multipolygon):
        """A one-row polygon table, so the PolygonLayer can be built before any join runs.

        lonboard will not take `table=None`: the layer imports the Arrow C stream in
        __init__ and raises "Expected object with __arrow_c_array__ ..." on anything else.
        So the layer needs real geometry from the start, and it begins invisible until a
        join produces some. The WKB is hand-built rather than borrowed from a geometry
        library, because this notebook has no shapely and one degenerate square at null
        island is not worth adding one for.
        """
        import struct

        # A REAL POLYGON, NOT A DEGENERATE ONE. The first version of this used a 1e-6
        # degree square, and deck's earcut tessellator failed on it, which took down the
        # ENTIRE update pass: deck initialises all layers in one batch, so one throw
        # produced a cascade of "deck.gl: assertion failed" naming BitmapLayer and
        # GeoArrowPolygonLayer, neither of which was at fault, followed by "Cannot schedule
        # pool tasks after terminate()". An assertion naming a layer is weak evidence that
        # the layer is the problem.
        #
        # 0.01 degrees, counter-clockwise, at null island: big enough to tessellate, far
        # enough from anywhere this map opens to be invisible.
        d = 0.01
        ring = [(0.0, 0.0), (d, 0.0), (d, d), (0.0, d), (0.0, 0.0)]
        wkb = struct.pack("<BII", 1, 6, 1)  # little endian, MultiPolygon, 1 polygon
        wkb += struct.pack("<BIII", 1, 3, 1, len(ring))  # Polygon, 1 ring, n points
        for x, y in ring:
            wkb += struct.pack("<dd", x, y)

        geom = from_wkb(
            pa.array([wkb], pa.binary()), to_type=multipolygon("xy", crs="EPSG:4326")
        )
        return ArroTable.from_arrays(
            [
                ArroArray.from_arrow(geom),
                # Four channels, matching what divisions_to_layer produces. A seed whose
                # colour column is three wide would make the first real push a change of
                # accessor WIDTH as well as of data, which is the class of swap that left
                # the fill unpainted before.
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([0, 0, 0, 0], dtype=np.uint8)), 4
                    )
                ),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([0.0])),
                ArroArray.from_arrow(pa.array([0], type=pa.int64())),
            ],
            names=["geometry", "color", "name", "country", "deforested %", "cells"],
        )

    return cells_to_layer, divisions_to_layer, seed_cells, seed_divisions


@app.cell
async def _(
    DIVISION_PREFIX,
    GeoParquetFile,
    OVERTURE_BUCKET,
    S3Store,
    asyncio,
    con,
    json,
    pa,
    pathlib,
    pc,
    pq,
    to_wkb,
):
    # DIVISIONS ARE STREAMED, NOT CACHED TO DISK. obstore opens the GeoParquet directly and
    # `read_async(bbox=...)` pushes the viewport down to row groups, so a zoomed-in view
    # reads a slice rather than a file.
    #
    # WHAT THE FILE INDEX DOES AND DOES NOT BUY HERE. It is 8 footer reads, seconds not the
    # 100 s the buildings index costs, but it prunes almost nothing: 7 of the 8 files have a
    # bbox wider than 130 degrees, so nearly every viewport hits nearly every file. The real
    # pruning is inside read_async, on row groups. The index is kept because skipping even
    # one 600 MB file is worth having and it costs nothing to consult.
    _div_store = S3Store(OVERTURE_BUCKET, region="us-west-2", skip_signature=True)
    _index_cache = pathlib.Path(".cache") / f"overture-index-divisions.json"

    async def _build_index():
        if _index_cache.exists():
            return json.loads(_index_cache.read_text())
        objects = _div_store.list_with_delimiter(DIVISION_PREFIX)["objects"]
        sem = asyncio.Semaphore(16)

        async def one(obj):
            async with sem:
                f = await GeoParquetFile.open_async(obj["path"], store=_div_store)
                try:
                    return obj["path"], list(f.file_bbox())
                except Exception:
                    return obj["path"], None  # no covering bbox: always read it

        idx = list(await asyncio.gather(*[one(o) for o in objects]))
        _index_cache.parent.mkdir(parents=True, exist_ok=True)
        _index_cache.write_text(json.dumps(idx))
        return idx

    DIV_INDEX = await _build_index()

    # WHERE THE TIME ACTUALLY GOES, MEASURED, BECAUSE THE OBVIOUS FIXES DO NOT WORK.
    #
    # One viewport at zoom 5.6 over Rondonia hit 3 of the 8 files and took 18.4 s read
    # serially. Two things that look like they would fix that, and do not:
    #
    #   - COLUMN PROJECTION. Useless: `geometry` is 99.0% of the compressed bytes of a row
    #     group (22.38 MB of 22.6 MB). Dropping `sources`, `names.rules` and the rest saves
    #     nothing, and `read_async` has no projection argument anyway.
    #   - PRUNING ON `subtype`. The row groups are not clustered by it. Their min/max pairs
    #     are ('county','region'), ('country','region'), ('locality','region'), so asking for
    #     'county' keeps nearly every group. This is the same finding as the note that
    #     subtype is not partitioned, now checked at the statistics level too.
    #
    # So the bytes are irreducible for a given box, and the three things that DO help are all
    # about not paying for them twice:
    #
    #   1. READ THE FILES CONCURRENTLY. 18.4 s serial becomes the slowest single file. The
    #      previous loop awaited them one at a time for no reason.
    #   2. KEEP THE OPEN FILE HANDLES. `open_async` is a footer read, ~0.8 s each, and it was
    #      being paid again on every fetch of every file.
    #   3. KEEP THE RESULT ON DISK. A box read once is read once ever, across kernel restarts.
    #      This is the part that decides whether the notebook feels fast, because exploring
    #      revisits the same ground constantly.
    #
    # CACHED BY COVERAGE, NOT BY EXACT BOX. The original memo keyed on the rounded request
    # bbox, so a pan of a few pixels was a cache miss and a fresh 18 s read. A read is grown
    # past what was asked for and kept, and any later box inside it is a lookup.
    #
    # DIV_PAD is on top of the viewport padding, so a cached read covers about 1.75 viewport
    # widths. Larger was tried and is the wrong trade while a cold read costs seconds: at 2.0
    # the first look at anywhere reads 6x the viewport's area, and the cost lands exactly when
    # the user is waiting.
    _div_mem = {}  # subtype -> [[box, table, key], ...], newest last
    DIV_PAD = 1.4
    DIV_KEEP = 8

    # ON-DISK, AND DELIBERATELY NOT A BUILD STEP. Nothing is precomputed and nothing has to
    # be refreshed: this only ever holds boxes that were genuinely asked for. Deleting
    # `.cache/divisions/` costs the next visit to each place and nothing else.
    _DIV_DIR = pathlib.Path(".cache") / "divisions"
    _DIV_LEDGER = _DIV_DIR / "index.json"

    def _load_ledger():
        try:
            return json.loads(_DIV_LEDGER.read_text())
        except Exception:
            return []

    _div_disk = _load_ledger()  # [[subtype, box, filename], ...]

    def _grow(b, f=DIV_PAD):
        w, s, e, n = b
        cx, cy = (w + e) / 2, (s + n) / 2
        hw, hh = (e - w) / 2 * f, (n - s) / 2 * f
        return (
            max(-180.0, cx - hw),
            max(-85.0, cy - hh),
            min(180.0, cx + hw),
            min(85.0, cy + hh),
        )

    def _inside(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    def _hits(bbox):
        w, s, e, n = bbox
        return [
            p
            for p, b in DIV_INDEX
            if b is None or (b[0] < e and b[2] > w and b[1] < n and b[3] > s)
        ]

    def _remember(subtype, box, table, key):
        held = _div_mem.setdefault(subtype, [])
        held.append([box, table, key])
        del held[:-DIV_KEEP]

    _div_files = {}  # path -> open GeoParquetFile; the footer read is ~0.8 s each

    async def _open(path):
        f = _div_files.get(path)
        if f is None:
            f = _div_files[path] = await GeoParquetFile.open_async(path, store=_div_store)
        return f

    async def _one_file(path, subtype, box):
        """One Overture part, filtered to the subtype we want, geometry as WKB.

        WKB rather than GeoArrow because the parts genuinely disagree about geometry type
        (Polygon in some files, MultiPolygon in others), which is the same thing that stops
        GeoParquetDataset opening them at all. A binary column has no opinion, so the parts
        concatenate.
        """
        f = await _open(path)
        data = await f.read_async(bbox=box)
        # A file whose bbox overlaps but whose ROW GROUPS all get pruned away is the normal
        # case here, since the file bboxes are near-global. The result is a table with no
        # chunks at all, and pa.chunked_array([]) raises "cannot construct ChunkedArray from
        # empty vector and omitted type" rather than returning something empty.
        if data.num_rows == 0:
            return None
        # to_wkb must run on the GeoArrow column, BEFORE the PyArrow conversion drops the
        # extension metadata it reads the geometry type from.
        wkb = pa.chunked_array(
            [
                pa.array(c)
                for c in pa.chunked_array(to_wkb(data.column("geometry"))).chunks
            ]
        ).cast(pa.binary())
        t = pa.RecordBatchReader.from_stream(data).read_all()
        # is_land drops the maritime half of division_area. Without it a coastal division's
        # zonal mean is dragged toward zero by open water that was never at risk of being
        # deforested.
        keep = pc.and_(
            pc.equal(t["subtype"], subtype), pc.fill_null(t["is_land"], False)
        )
        t = pa.table(
            {
                "id": t["id"],
                "name": pc.struct_field(t["names"], "primary"),
                "country": t["country"],
                "wkb": wkb,
            }
        ).filter(keep)
        return t if t.num_rows else None

    async def fetch_divisions(subtype, bbox):
        """Overture divisions of one subtype covering bbox, geometry as WKB.

        Returns (table or None, key). The key names the cached read rather than the request,
        so the polyfill can memoise against it: two viewports served by one cached read share
        one set of filled cells.
        """
        for box, tbl, key in _div_mem.get(subtype, []):
            if _inside(box, bbox):
                return tbl, key

        for sub, box, fname in _div_disk:
            if sub == subtype and _inside(box, bbox):
                key = (subtype, tuple(box))
                try:
                    tbl = pq.read_table(_DIV_DIR / fname)
                except Exception:
                    continue  # a half-written file from an interrupted kernel
                tbl = tbl if tbl.num_rows else None
                _remember(subtype, tuple(box), tbl, key)
                return tbl, key

        big = tuple(round(v, 4) for v in _grow(bbox))
        parts = [
            t
            for t in await asyncio.gather(
                *(_one_file(p, subtype, big) for p in _hits(big))
            )
            if t is not None
        ]
        out = pa.concat_tables(parts) if parts else None
        key = (subtype, big)

        fname = f"{subtype}-{abs(hash(key)):016x}.parquet"
        try:
            _DIV_DIR.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                out if out is not None else pa.table({"id": pa.array([], pa.string())}),
                _DIV_DIR / fname,
            )
            _div_disk.append([subtype, list(big), fname])
            _DIV_LEDGER.write_text(json.dumps(_div_disk))
        except Exception:
            pass  # a read-only or full disk costs speed on the next run, not correctness

        _remember(subtype, big, out, key)
        return out, key

    # THE POLYFILL, AND THE ONE THING THAT MAKES IT AWKWARD.
    #
    # h3_polygon_wkb_to_cells_experimental takes a POLYGON and raises
    #   Invalid WKB: expected polygon at 5
    # on a MultiPolygon, which is 148 of 219 countries, 1,193 regions and 3,661 counties. So
    # each division is split with ST_Dump, every part filled, and the parts flattened back
    # into one distinct cell set per division. Cheap, but not optional, and the error names
    # the WKB rather than the geometry type, so it reads like corruption.
    #
    # CONTAINMENT IS 'center', AND THAT IS THE DIFFERENCE BETWEEN A ZONAL MEAN AND A SMEAR.
    # 'overlap' includes every cell that so much as touches the division, so a county a few
    # cells wide would have its mean substantially made of ground outside it, and cells on a
    # shared border would be counted into both neighbours. 'center' puts each cell in
    # exactly one division.
    #
    # The cost is that a division smaller than one cell catches no centre and gets no
    # number: measured at res 4, a Singapore-sized box yields 0 cells under 'center' and 4
    # under 'overlap'. Those are reported rather than guessed at (see the status line).
    # Loosening the rule for small divisions only would mean a handful are measured
    # differently from their neighbours with nothing on screen saying which.
    POLYFILL_SQL = """
        WITH parts AS (
            SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM divs
        ), filled AS (
            SELECT id, UNNEST(
                       h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'center')
                   ) AS hex
            FROM parts
        )
        SELECT DISTINCT id, hex FROM filled
    """

    # Memoised on (cached read, resolution). The polyfill is pure: the same divisions at the
    # same resolution give the same cells forever, and a pan that reuses a cached read now
    # reuses its cells too. Bounded because each entry is a few hundred thousand ids.
    _fill_memo = {}
    FILL_KEEP = 24

    def polyfill(divs_table, key, res):
        """(id, hex) for every division, at one resolution.

        `divs` is the name POLYFILL_SQL selects from, and there is no register() call:
        DuckDB's replacement scan resolves it straight out of this frame, Arrow buffers and
        all, even as a local inside a nested function.
        """
        ck = (key, int(res))
        if ck in _fill_memo:
            return _fill_memo[ck]
        divs = divs_table  # noqa: F841 - read by the replacement scan, not by Python
        # to_arrow_table, NOT .arrow(): as of DuckDB 1.5 that hands back a
        # RecordBatchReader, and the failure surfaces much later as
        # "'pyarrow.lib.RecordBatchReader' object has no attribute 'num_rows'".
        out = con.sql(POLYFILL_SQL, params=[int(res)]).to_arrow_table()
        _fill_memo[ck] = out
        while len(_fill_memo) > FILL_KEEP:
            _fill_memo.pop(next(iter(_fill_memo)))
        return out

    return fetch_divisions, polyfill


@app.cell
def _(
    BitmapTileLayer,
    CartoBasemap,
    Controls,
    DIVISION_LABEL,
    H3HexagonLayer,
    HOLD,
    HOME,
    Map,
    MaplibreBasemap,
    PAD,
    Panel,
    PolygonLayer,
    SETTLE,
    STOPS,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    cells_to_layer,
    division_for_zoom,
    from_wkb,
    infer_rows_per_chunk,
    multipolygon,
    np,
    ramp,
    res_for_zoom,
    seed_cells,
    seed_divisions,
):
    # Built exactly once. This cell depends on no control and on no state the camera can
    # write, so nothing in the notebook can re-run it and throw the view away. Everything
    # after this happens by trait assignment, which lonboard treats as independent of
    # `view_state`.
    status = Status(value="<b>loading…</b>")
    controls = Controls()
    ranking = Panel()

    _seed = seed_cells()
    cells = H3HexagonLayer(
        table=_seed,
        get_hexagon=_seed["hex"],
        get_fill_color=_seed["color"],
        extruded=False,
        stroked=False,
        high_precision=True,
        coverage=1,
        opacity=0.7,
        pickable=True,
    )

    # THE DIVISIONS. Permanently `filled=True`, and `get_fill_color` is ALWAYS a column of
    # the table on the layer, never a constant. Two rules, and the second is the one that
    # was missing.
    #
    # `filled` decides whether deck builds a fill sublayer at all, and flipping it after init
    # does not reliably make one appear: that is the "the fill button does nothing" bug
    # recorded in CLAUDE.md. But following only that rule still left the fill dead, because
    # the toggle swapped `get_fill_color` between a table column and the constant
    # `[0, 0, 0, 0]`, which is a change of accessor KIND rather than of data. The fix is that
    # both states are now the same column of the same schema with a different alpha baked in,
    # and switching between them re-pushes the table. See divisions_to_layer.
    #
    # line_width_units="pixels" explicitly. deck's default is METRES with get_line_width
    # defaulting to 1, so the visible width is max(1 metre in pixels, line_width_min_pixels)
    # and can never go below the floor: two numbers in different units fighting over one
    # line. In pixel units get_line_width is the width, full stop.
    _dseed = seed_divisions(from_wkb, multipolygon)
    divisions = PolygonLayer(
        table=_dseed,
        get_fill_color=_dseed["color"],
        filled=True,
        stroked=True,
        line_width_units="pixels",
        get_line_width=1.0,
        line_width_min_pixels=0,
        line_width_max_pixels=1.5,
        get_line_color=[232, 236, 242, 205],
        opacity=1.0,
        pickable=True,
        visible=False,
    )

    # Place labels drawn OVER the cells. The basemap paints under every deck layer, so
    # names on it would sit beneath an opaque hexagon and be lost. pickable=False so a
    # hover meant for a cell is never intercepted; @2x with tile_size 512 because the
    # default 256 samples retina tiles at half scale and the type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
    )

    deck = Map(
        [cells, divisions, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state=HOME,
        height=VIEW_H,
        show_tooltip=True,
    )

    # A NEW MAP INHERITS NOTHING ABOUT THE OLD ONE'S SCREEN. HOLD lives in a cell that
    # cannot re-run, which is what lets the camera survive; the cost is that a re-run of
    # THIS cell builds fresh layers while HOLD still describes the map that just went away.
    for _t in ("task", "seltask"):
        if HOLD[_t] is not None:
            HOLD[_t].cancel()
        HOLD[_t] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["res"], HOLD["box"], HOLD["div"] = None, None, None
    HOLD["divpair"], HOLD["divbox"], HOLD["vs"] = None, None, None
    HOLD["head"], HOLD["tail"] = "", ""
    HOLD["cache"].clear()

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N], clamped to the world.

        Web Mercator: the horizontal span is a straight function of zoom, and the vertical
        span is that scaled by the aspect ratio and by cos(latitude), because a degree of
        longitude narrows toward the poles.
        """
        import math as _m

        span = 360.0 * VIEW_W / (512 * 2**vs.zoom)
        lat_span = span * (VIEW_H / VIEW_W) * _m.cos(_m.radians(vs.latitude))
        return (
            max(-180.0, vs.longitude - span / 2),
            max(-85.0, vs.latitude - lat_span / 2),
            min(180.0, vs.longitude + span / 2),
            min(85.0, vs.latitude + lat_span / 2),
        )

    def _pad(b):
        w, s, e, n = b
        cx, cy = (w + e) / 2, (s + n) / 2
        hw, hh = (e - w) / 2 * PAD, (n - s) / 2 * PAD
        return (
            max(-180.0, cx - hw),
            max(-85.0, cy - hh),
            min(180.0, cx + hw),
            min(85.0, cy + hh),
        )

    def _covers(box, want):
        return (
            box is not None
            and box[0] <= want[0]
            and box[1] <= want[1]
            and box[2] >= want[2]
            and box[3] >= want[3]
        )

    def _same_view(a, b):
        """The echo check: ignore the event the map emits for a view we set ourselves."""
        return (
            a is not None
            and b is not None
            and round(a.longitude, 6) == round(b.longitude, 6)
            and round(a.latitude, 6) == round(b.latitude, 6)
            and round(a.zoom, 4) == round(b.zoom, 4)
        )

    def set_status(vs):
        """Redraw the status line from what is already known, plus this zoom.

        Kept separate from the read because most camera moves read nothing, and the zoom
        readout still has to move: zooming IN always lands inside the box the last read
        covered, so without this the line would freeze exactly when the map is busiest.
        """
        status.value = f"{HOLD['head']}{HOLD['tail']} · zoom {vs.zoom:.1f}"

    def put_cells(tbl):
        cells._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # hold_sync so deck gets one message. Without it the new hexagons are drawn
        # against the old colour buffer for a frame.
        with cells.hold_sync():
            cells.table = tbl
            cells.get_hexagon = tbl["hex"]
            cells.get_fill_color = tbl["color"]
            cells.visible = controls.show_cells

    def put_divisions(pair):
        """Push whichever of the two colour variants the fill switch is asking for."""
        tbl = pair[0] if controls.division_fill else pair[1]
        divisions._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with divisions.hold_sync():
            divisions.table = tbl
            divisions.get_fill_color = tbl["color"]
            divisions.visible = controls.show_divisions
        HOLD["divpair"] = pair

    def _on_controls(change):
        name = change["name"]
        if name == "show_cells":
            cells.visible = bool(change["new"])
        elif name == "show_divisions":
            divisions.visible = bool(change["new"]) and HOLD["divpair"] is not None
        elif name == "division_fill":
            # A whole re-push, not an accessor assignment. See divisions_to_layer.
            if HOLD["divpair"] is not None:
                put_divisions(HOLD["divpair"])

    controls.observe(_on_controls, names=["show_cells", "show_divisions", "division_fill"])

    def _instant(vs):
        """Everything answerable without a read, done synchronously in the comm handler.

        THIS IS THE ZOOM AND PAN FEEL, and it is where the clunkiness was. `view_state` fires
        on every frame, and every one of those frames used to be handed to an async task that
        slept SETTLE seconds BEFORE it would so much as look at the cache. So a pan inside the
        box already on screen cost a quarter of a second of nothing, and a zoom back out to a
        resolution already folded cost the same again, even though both are dict lookups.
        Answering them here means the map keeps up with the mouse and the debounce is only
        ever spent waiting on bytes that are genuinely missing.
        """
        res, sub = res_for_zoom(vs.zoom), division_for_zoom(vs.zoom)
        seen = view_to_bbox(vs)
        div_ok = sub is None or (sub == HOLD["div"] and _covers(HOLD["divbox"], seen))
        if res == HOLD["res"] and sub == HOLD["div"] and div_ok and _covers(HOLD["box"], seen):
            set_status(vs)
            return True
        # A resolution folded before that still covers the screen. This is the whole
        # zoom-out case: coming back up to a level already visited lands on the frame it is
        # asked for rather than a second later.
        hit = HOLD["cache"].get(res)
        if hit and sub == HOLD["div"] and div_ok and _covers(hit[0], seen):
            put_cells(hit[1])
            HOLD["res"], HOLD["box"] = res, hit[0]
            HOLD["head"] = f"<b>res {res}</b> · {hit[1].num_rows:,} cells · cached"
            set_status(vs)
            return True
        return False

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or read and refold."""
        if not force and _instant(vs):
            return

        res = res_for_zoom(vs.zoom)
        sub = division_for_zoom(vs.zoom)
        want = _pad(view_to_bbox(vs))

        # THE LAST ANSWER STAYS UP UNTIL THERE IS A NEW ONE. Nothing is cleared here: the
        # read happens under the cells already on screen, and the swap is one trait update
        # when the new fold is complete. A stale-but-plausible map reads as the map; an
        # empty one reads as broken.
        HOLD["head"] = f"<b>reading…</b> res {res}"
        set_status(vs)
        raw, fetched, skipped = await HOLD["fold"](res, want)
        if raw is None or raw.num_rows == 0:
            HOLD["res"], HOLD["box"], HOLD["div"] = res, want, sub
            HOLD["head"], HOLD["tail"] = f"<b>res {res}</b> · no data here", ""
            set_status(vs)
            return

        tbl = cells_to_layer(raw)
        HOLD["cache"][res] = [want, tbl, raw]
        put_cells(tbl)
        HOLD["res"], HOLD["box"] = res, want

        HOLD["head"] = (
            f"<b>res {res}</b> · {raw.num_rows:,} cells · "
            f"{'tiles cached' if fetched == 0 else f'{fetched} tiles'}"
            f"{f' · {skipped} sparse' if skipped else ''}"
        )
        set_status(vs)

        # THE DIVISIONS, AFTER THE CELLS. The cells are the expensive read and the thing
        # the user is waiting to see; the zonal join depends on them, so it goes out second
        # rather than holding the whole frame.
        if HOLD["pending"] is not None:
            return  # the camera has already moved; this view is gone
        if sub is None:
            # Zoomed too far out for boundaries. See division_for_zoom: a world bbox
            # prunes no row groups, so this is the one view where streaming Overture would
            # mean reading most of 5.5 GB to find 219 rows.
            divisions.visible = False
            HOLD["divpair"], HOLD["div"], HOLD["divbox"] = None, None, None
            HOLD["tail"] = " · zoom in for boundaries"
            set_status(vs)
            return
        zonal = await HOLD["zonal"](sub, want, res, raw)
        HOLD["div"], HOLD["divbox"] = sub, want
        if zonal is None:
            divisions.visible = False
            HOLD["divpair"], HOLD["tail"] = None, ""
            set_status(vs)
            return
        pair, n_div, n_unmeasured = zonal
        put_divisions(pair)
        HOLD["tail"] = f" · {n_div:,} {DIVISION_LABEL[sub]}" + (
            f" · <b style='color:#E69F00'>{n_unmeasured} too small to measure</b>"
            if n_unmeasured
            else ""
        )
        set_status(vs)

    async def refresh(vs, force=False):
        """Fold what the camera is looking at, once it has stopped moving.

        Each fold is an object-store read, so SETTLE debounces: a drag reads once at the end
        rather than at every position it passed through. Coalescing then collapses whatever
        piled up during a read to the NEWEST view, without which a two-second drag queues a
        hundred folds of stale viewports and never catches up. No threads and no timers; the
        debounce is an await on the kernel's own loop, so the map keeps rendering.
        """
        if HOLD["fold"] is None:
            return
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                if not force and SETTLE > 0:
                    await asyncio.sleep(SETTLE)
                    if HOLD["pending"] is not None:
                        # Still moving. Take the newest view and settle again, which is the
                        # debounce; it ends when a whole SETTLE passes with nothing queued.
                        vs, HOLD["pending"] = HOLD["pending"], None
                        continue
                await _draw(vs, force)
                vs, force = HOLD["pending"], False
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as exc:
            # A failure inside a comm handler is otherwise completely silent.
            HOLD["head"] = (
                f"<b style='color:#F0E442'>failed:</b> {type(exc).__name__}: {exc}"
            )
            HOLD["tail"] = ""
            status.value = HOLD["head"]
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _spawn(coro):
        """Run a coroutine on the kernel's loop, keeping a strong reference to the task.

        asyncio holds only a weak one, so a bare create_task can be collected mid-flight.
        """
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _on_camera(change):
        vs = change["new"]
        if _same_view(vs, HOLD["vs"]):
            return
        HOLD["vs"] = vs
        if HOLD["busy"]:
            # A read is already in flight; let it finish and take this view next, rather
            # than painting a cache hit the in-flight result would immediately overwrite.
            HOLD["pending"] = vs
            return
        if _instant(vs):
            return
        HOLD["task"] = _spawn(refresh(vs))

    deck.observe(_on_camera, names="view_state")

    # ---------------------------------------------------------------- the drawn box
    # Draw a box with the ▢ button at the lower right of the map and the divisions inside it
    # come back ranked, below. This is the one place the join produces a NUMBER rather than a
    # colour, and it is deliberately not tied to the camera: the box is an explicit ask, so
    # it reads one level FINER than the screen and it names the divisions outright.
    RANK_N = 25

    def rank_html(out):
        if out is None:
            return (
                "<div style='font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
                "opacity:.75;padding:.5rem 0'>No division in that box caught a cell centre. "
                "Draw a larger box, or zoom in first.</div>"
            )
        sub, res, tbl, n_small = out
        names = tbl["name"].to_pylist()
        country = tbl["country"].to_pylist()
        n_cells = tbl["n_cells"].to_pylist()
        portion = np.asarray(tbl["portion"], dtype="float64")
        order = np.argsort(-portion)[:RANK_N]
        top = float(portion[order[0]]) if len(order) else 1.0
        rows = []
        for place, i in enumerate(order, 1):
            i = int(i)
            rgb = ",".join(str(int(c)) for c in ramp(np.array([portion[i]]))[0])
            bar = max(2.0, 100.0 * portion[i] / max(top, 1e-12))
            label = names[i] or "(unnamed)"
            rows.append(
                f"<tr>"
                f"<td style='text-align:right;opacity:.5;padding:.12rem .5rem .12rem 0'>{place}</td>"
                f"<td style='padding:.12rem .6rem .12rem 0;white-space:nowrap'>{label}"
                f"<span style='opacity:.45'> · {country[i] or '??'}</span></td>"
                f"<td style='text-align:right;padding:.12rem .6rem .12rem 0;"
                f"font-variant-numeric:tabular-nums'>{portion[i] * 100:.3f}%</td>"
                f"<td style='width:180px;padding:.12rem .6rem .12rem 0'>"
                f"<span style='display:block;height:9px;border-radius:2px;"
                f"width:{bar:.1f}%;background:rgb({rgb})'></span></td>"
                f"<td style='text-align:right;opacity:.45;"
                f"font-variant-numeric:tabular-nums'>{n_cells[i]:,} cells</td>"
                f"</tr>"
            )
        head = (
            f"<b>{len(names):,} {DIVISION_LABEL[sub]} in the box</b>, ranked by mean share "
            f"deforested 2002-2022, measured at H3 res {res}"
            + (
                f" · <span style='color:#E69F00'>{n_small} too small to measure</span>"
                if n_small
                else ""
            )
            + (
                f" · showing the top {RANK_N}"
                if len(names) > RANK_N
                else ""
            )
        )
        return (
            "<div style='font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
            "padding:.5rem 0 .2rem'>"
            f"<div style='opacity:.8;padding-bottom:.35rem'>{head}</div>"
            "<table style='border-collapse:collapse'>" + "".join(rows) + "</table></div>"
        )

    def _on_select(change):
        b = change["new"]
        if not b:
            return  # a fresh Map resets selected_bounds to None
        box = tuple(float(v) for v in b)
        ranking.value = (
            "<div style='font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
            "opacity:.7;padding:.5rem 0'><b>ranking</b> the divisions in that box…</div>"
        )

        async def go():
            try:
                ranking.value = rank_html(await HOLD["rank"](box))
            except Exception as exc:
                ranking.value = (
                    f"<div style='font:12.5px ui-monospace,monospace;padding:.5rem 0'>"
                    f"<b style='color:#F0E442'>ranking failed:</b> "
                    f"{type(exc).__name__}: {exc}</div>"
                )
                raise

        HOLD["seltask"] = _spawn(go())

    deck.observe(_on_select, names="selected_bounds")

    # The legend, built from the same `ramp` the layers use, so a colour on the map and a
    # colour in the key cannot drift apart. The division fill uses the same ramp at a lower
    # alpha, so one key serves both.
    _sw = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
        f"<span style='width:14px;height:14px;border-radius:2px;background:rgb("
        f"{','.join(str(int(c)) for c in ramp(np.array([v]))[0])})"
        f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
        for v, lab in STOPS
    )
    legend = (
        "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
        "display:flex;flex-wrap:wrap;align-items:center;padding:.35rem 0\">"
        "<b style='margin-right:.7rem'>share of cell deforested 2002-2022</b>"
        f"{_sw}</div>"
    )
    return controls, deck, legend, ranking, refresh, status


@app.cell
async def _(
    COG,
    FETCH_AT_ONCE,
    GeoTIFF,
    HOLD,
    HOME,
    LEVEL_FOR_RES,
    MAX_RES,
    S3Store,
    SOURCE_BUCKET,
    TILE,
    TILE_BUDGET,
    VIEW_W,
    Window,
    XarrayContext,
    asyncio,
    coordinates_to_cells,
    divisions_to_layer,
    fetch_divisions,
    from_wkb,
    math,
    multipolygon,
    np,
    pa,
    polyfill,
    refresh,
    res_for_zoom,
    udf,
    xr,
):
    # Opening the COG reads headers only. Each fold then pulls JUST the padded viewport,
    # from the overview matching the H3 resolution it is about to build. That inversion is
    # what makes the whole planet affordable: the viewport shrinks faster than the
    # resolution grows, so the finest reads are the smallest.
    _store = S3Store(SOURCE_BUCKET, region="us-west-2", skip_signature=True)
    _g = await GeoTIFF.open(COG, store=_store)
    _levels = [_g, *_g.overviews]
    _L, _B, _R, _T = _g.bounds

    # WHICH TILES EXIST, PER LEVEL. This is the sparse-COG fix, and it is read straight out
    # of IFDs already in memory, so it costs no network at all.
    #
    # 73.6% of L0 tiles have offset 0 and byte count 0: ocean is simply not stored. async
    # geotiff does not check, so a read touching one asks for byte range 0..0 and raises
    # `TypeError: ValueError: Invalid range requested, start: 0 end: 0`. That error names
    # neither the tile nor the fact that the file is sparse, so it reads like corruption.
    #
    # Consulting the table first turns the crash into a speedup: an absent tile becomes a
    # NaN block with no request. Measured on the opening world view, 10 of 78 tiles skipped.
    _present = []
    for _lv in _levels:
        _ifd = _lv.ifd
        _nty = -(-_lv.shape[0] // TILE)
        _ntx = -(-_lv.shape[1] // TILE)
        _present.append(np.asarray(_ifd.tile_byte_counts).reshape(_nty, _ntx) > 0)

    _tiles = {}  # (level, ty, tx) -> float32 array; insertion order is LRU order
    # A dict, not an int: a marimo cell body is compiled at MODULE scope, so `nonlocal` in
    # a nested def is a SyntaxError there and a bare rebind would shadow instead of
    # accumulate.
    _held = {"bytes": 0}
    _sem = asyncio.Semaphore(FETCH_AT_ONCE)

    async def _tile(li, ty, tx):
        lv = _levels[li]
        H, W = lv.shape
        r0, c0 = ty * TILE, tx * TILE
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        if not _present[li][ty, tx]:
            return np.full((h, w), np.nan, dtype=np.float32), True
        async with _sem:
            arr = np.asarray(
                (
                    await lv.read(
                        window=Window(col_off=c0, row_off=r0, width=w, height=h)
                    )
                ).as_masked()[0]
            ).astype(np.float32)
        return arr, False

    async def _read_window(li, col0, row0, wpx, hpx):
        """The window, assembled from cached tiles plus whatever is missing.

        Snapping to the COG's own tile grid is what makes a read SHAREABLE. One ranged read
        per exact viewport can never be reused, because no two camera positions produce the
        same rectangle; a pan on this grid touches the tiles already held plus a strip of
        new ones, and a zoom back to a level visited before is free.
        """
        ty0, ty1 = row0 // TILE, (row0 + hpx - 1) // TILE
        tx0, tx1 = col0 // TILE, (col0 + wpx - 1) // TILE
        want = [(li, ty, tx) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        need = [k for k in want if k not in _tiles]

        fetched = skipped = 0
        if need:
            got = await asyncio.gather(*(_tile(*k) for k in need))
            for k, (a, was_sparse) in zip(need, got):
                _tiles[k] = a
                _held["bytes"] += a.nbytes
                if was_sparse:
                    skipped += 1
                else:
                    fetched += 1
            # Oldest first, and never evict a tile this window is about to read.
            while _held["bytes"] > TILE_BUDGET and len(_tiles) > len(want):
                for k in list(_tiles):
                    if k not in want:
                        _held["bytes"] -= _tiles.pop(k).nbytes
                        break
                else:
                    break

        out = np.full((hpx, wpx), np.nan, dtype=np.float32)
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
        for k in want:  # touch: anything used goes to the young end of the LRU
            _tiles[k] = _tiles.pop(k)
        return out, fetched, skipped

    # EPSG:4326 IS THE WHOLE SIMPLIFICATION. The NLCD notebooks in this repo carry an Albers
    # control grid, a bilinear interpolator and to_lat/to_lon UDFs purely to get degrees out
    # of projected metres. Here the pixel grid IS degrees, so the y/x coordinates of the
    # registered dataset feed h3_latlng_to_cell directly and all of that machinery is gone.
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

    def _register(name, table):
        """Replace a registered Arrow table, whatever the datafusion version calls it."""
        try:
            ctx.deregister_table(name)
        except Exception:
            pass
        try:
            ctx.from_arrow(table, name=name)
        except Exception:
            # Older datafusion has no from_arrow(name=...); the batches path is stable
            # across every version this notebook has been run against. Catching broadly
            # rather than on AttributeError, because the signature mismatch surfaces as a
            # TypeError, which the narrower catch would let through.
            ctx.register_record_batches(name, [table.to_batches()])

    async def fold(res, box):
        """Read the window for `box` at the overview `res` deserves, then fold it to H3."""
        li = LEVEL_FOR_RES[res]
        rd = _levels[li]
        H, W = rd.shape
        px, py = (_R - _L) / W, (_T - _B) / H

        w, s, e, n = box
        col0 = max(0, int((max(w, _L) - _L) / px))
        col1 = min(W, int(math.ceil((min(e, _R) - _L) / px)))
        row0 = max(0, int((_T - min(n, _T)) / py))
        row1 = min(H, int(math.ceil((_T - max(s, _B)) / py)))
        wpx, hpx = col1 - col0, row1 - row0
        if wpx <= 0 or hpx <= 0:
            return None, 0, 0

        arr, fetched, skipped = await _read_window(li, col0, row0, wpx, hpx)

        # The window's own corner, not the raster's. Everything downstream is relative.
        wl, wt = _L + col0 * px, _T - row0 * py
        try:
            ctx.deregister_table("df")
        except Exception:
            pass
        ctx.from_dataset(
            "df",
            xr.Dataset(
                {"v": (("y", "x"), arr)},
                coords={
                    "y": wt - (np.arange(hpx) + 0.5) * py,
                    "x": wl + (np.arange(wpx) + 0.5) * px,
                },
            ),
            chunks={"y": 512},
        )

        # `v = v` IS THE NaN TEST. This COG declares no nodata value at all, so ocean comes
        # back as NaN rather than a sentinel, and `v != NULL` would not catch it. NaN is the
        # one value that fails equality with itself.
        #
        # px_total is not decoration: it is the weight a cell carries into the zonal join.
        # A coastal cell may be 90% NaN ocean and must not count as a full one.
        # HAVING, NOT WHERE, AND THE DIFFERENCE IS THE WHOLE MEAN.
        #
        # Cells with no deforestation at all are dropped, because they are overwhelmingly
        # ocean and open water: 69.6% of res-4 cells are exactly zero, so drawing them
        # covers the map in dark hexagons that say nothing and cost most of the render.
        #
        # But the filter has to be on the CELL, not the PIXEL. `WHERE v > 0` would exclude
        # zero pixels from the average itself, so a cell that is 90% untouched forest and
        # 10% clearcut would report 100% rather than 10%. Every zero pixel stays in the
        # average; only cells whose average is zero are dropped.
        #
        # THE COST, SAID PLAINLY: land that genuinely lost no forest 2002-2022 disappears
        # too, and looks identical to ocean. This map shows where deforestation IS, not
        # where it is absent.
        return (
            ctx.sql(f"""
                SELECT h3_latlng_to_cell(y, x, CAST({res} AS INT)) AS hex,
                       avg(CAST(v AS DOUBLE)) AS portion,
                       count(*)               AS px_total
                FROM df
                WHERE v = v
                GROUP BY 1
                HAVING avg(CAST(v AS DOUBLE)) > 0
            """).to_arrow_table(),
            fetched,
            skipped,
        )

    # THE ZONAL JOIN, IN DATAFUSION.
    #
    # An equi-join on a UBIGINT plus a group-by. No geometry, no H3 call, nothing a query
    # engine is not already the best tool for. The cells are already in this context, so
    # shipping them to DuckDB because DuckDB happens to hold the polygons would be
    # backwards.
    #
    # avg(portion) EQUAL-WEIGHTS THE CELLS, AND THAT IS THE POINT. H3 cells are
    # near-equal-area, so an unweighted mean over cells is an AREA-weighted mean of the
    # ground. Weighting by px_total here instead would reintroduce exactly the EPSG:4326
    # latitude bias that folding to H3 removed, because a degree box holds more pixels near
    # the equator. The pixel weighting already happened, inside each cell, in `fold`.
    ZONAL_SQL = """
        SELECT d.id           AS id,
               avg(c.portion) AS portion,
               count(*)       AS n_cells
        FROM div_cells d JOIN cells c ON d.hex = c.hex
        GROUP BY d.id
    """

    def join_divisions(meta, key, res, cells_tbl):
        """(divisions with a value, divisions with none) for one subtype at one resolution.

        Synchronous from the register to the result on purpose: `ctx` is one shared context
        and `div_cells` / `cells` are fixed names in it, so anything that awaited in the
        middle could have its tables swapped out from under it by the next camera event.
        With no await between them the event loop cannot interleave and no lock is needed.
        """
        mapping = polyfill(meta, key, res)
        if mapping.num_rows == 0:
            return None, meta.num_rows
        _register("div_cells", mapping)
        _register("cells", cells_tbl)
        joined = ctx.sql(ZONAL_SQL).to_arrow_table().combine_chunks()
        if joined.num_rows == 0:
            return None, meta.num_rows
        # Divisions that caught no cell centre get NO number rather than a guessed one.
        # Always the same cause: smaller than one cell at this resolution. An inner join
        # drops them, which is what makes the choropleth honest and the count reportable.
        out = meta.join(joined, keys="id", join_type="inner")
        return out, max(0, meta.num_rows - out.num_rows)

    async def zonal(subtype, box, res, cells_tbl):
        """Divisions in view, each with its area-weighted mean deforestation.

        Returns (the two colour variants of the layer table, divisions drawn, divisions with
        no number) or None.
        """
        meta, key = await fetch_divisions(subtype, box)
        if meta is None or meta.num_rows == 0:
            return None
        out, n_unmeasured = join_divisions(meta, key, res, cells_tbl)
        if out is None:
            return None
        return (
            divisions_to_layer(out, from_wkb, multipolygon),
            out.num_rows,
            n_unmeasured,
        )

    async def rank(box):
        """Every division inside a drawn box, with its mean, for the ranking below the map.

        Returns (subtype, resolution, table, divisions with no number) or None.

        THREE THINGS THIS DOES NOT SHARE WITH THE CAMERA, AND WHY.
        1. It reads ONE RESOLUTION FINER than the screen would. A drawn box is an explicit
           question about a specific place, so it is worth a read the camera would not
           spend, and the finer the cells the fewer divisions fall through the 'center' rule.
        2. It derives that resolution from the BOX, not from the current zoom, so a small box
           drawn on a wide view still gets measured properly.
        3. It falls back county -> region -> country. Overture has counties for 171 of 219
           countries, so a box over the other 48 would otherwise come back empty rather than
           answering at the finest level that exists there.
        """
        span = max(box[2] - box[0], 1e-9)
        z = math.log2(360.0 * VIEW_W / (512 * span))
        res = min(MAX_RES, res_for_zoom(z) + 1)
        raw, _fetched, _skipped = await fold(res, box)
        if raw is None or raw.num_rows == 0:
            return None
        for sub in ("county", "region", "country"):
            meta, key = await fetch_divisions(sub, box)
            if meta is None or meta.num_rows == 0:
                continue
            out, n_small = join_divisions(meta, key, res, raw)
            if out is not None:
                return sub, res, out, n_small
        return None

    HOLD["fold"] = fold
    HOLD["zonal"] = zonal
    HOLD["rank"] = rank
    HOLD["loop"] = asyncio.get_running_loop()

    # The opening draw. force=True skips the settle: there is nothing to debounce yet.
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]

    await refresh(_VS(), force=True)
    return


@app.cell
def _(controls, deck, legend, mo, ranking, status):
    mo.vstack(
        [
            deck,
            status,
            mo.Html(legend),
            controls,
            mo.md(
                "Deforestation 2002-2022 as a share of each 100 m cell "
                "(Vizzuality / LandGriffon, CC-BY 4.0). Boundaries: Overture Maps. "
                "A division's value is the mean over the H3 cells whose CENTRE falls "
                "inside it, so divisions smaller than one cell at the current resolution "
                "are drawn unfilled rather than given a number. "
                "**Draw a box** with the ▢ button at the lower right of the map to rank "
                "the divisions inside it."
            ),
            ranking,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
