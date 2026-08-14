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
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "lonboard[geotiff]>=0.16.0",
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

  obstore      streams the COG and the Overture divisions PMTiles, unsigned. Nothing is
               cached to disk; a viewport reads what it needs and keeps it in memory.
  DataFusion   the fold (pixels -> H3 cells) AND the join (cells -> divisions). The join
               is an integer equi-join on a UBIGINT cell id plus a group-by, which is what
               a query engine is for.
  DuckDB       the polyfill (division polygon -> the cells covering it) and the tile-seam
               dissolve (clipped pieces of one division -> one MultiPolygon). The two
               geometry steps neither DataFusion nor plain SQL can do.
  lonboard     the render.

WHY PMTILES AND NOT THE GEOPARQUET. Overture lays the division_area files out with no
spatial order, so the geometry (99.0% of the bytes) cannot be pruned: a Rondonia-sized
viewport decodes ~190 MB per file to keep 6,337 rows, and no query makes that smaller.
The same release's divisions.pmtiles is the vector twin of the COG's overview pyramid:
one 19.5 GB object, addressed by ranged GET, Hilbert-ordered tiles, z0-12. The same
viewport reads ~0.8 MB. Tiles are gzipped MVT, decoded here with a hand-rolled protobuf
walk (verified ring-exact against mapbox-vector-tile) because the whole reader is fewer
lines than the dependency. Tile geometry is quantized to ~2.4 m at z12 and clipped to
tile edges; the polyfill is 'center'-ruled at res 4-8 (cells 460 m and up), so the cells
land identically, and the dissolve below removes the clip edges before anything is drawn.

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

THE PAINT IS THE RASTER ITSELF (2026-08-14, not yet flown interactively). A lonboard
RasterLayer serves the COG's own pyramid as PNG tiles coloured by the same ramp, so the
map shows pixels rather than cell means; the H3 hexagon layer is commented out in the map
cell, not deleted. The fold is untouched, because the divisions join and the ranking eat
its cells. Zero and NaN pixels are both transparent, matching the fold's
`HAVING avg(v) > 0`. The layer is built directly rather than via
RasterLayer.from_geotiff, for two reasons recorded at the construction site: from_geotiff's
fetch is not sparse-aware, and its zoom clamp ships commented out.

PRESS THE BUTTON AND THE JOIN BECOMES A NUMBER. "rank what's in view", in the controls
under the map, ranks every division in the current view by its mean share deforested. It
reads one H3 resolution finer than the screen does, sizes that resolution from the view
box rather than the current zoom, and falls back county -> region -> country, because
Overture has counties for only 171 of 219 countries. This is the one output here that is
a figure rather than a colour. It replaced lonboard's draw-box tool, which asked the
user to describe a region twice (camera, then rectangle); the toolbar for that tool is
hidden from the Controls widget, since lonboard 0.16 has no Python-side switch for it.

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
    import gzip
    import io
    import math
    import struct

    import anywidget
    import traitlets
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import duckdb
    import numpy as np
    import obstore
    import pyarrow as pa
    import xarray as xr
    from arro3.core import Array as ArroArray, Table as ArroTable
    from async_geotiff import GeoTIFF, Window
    from async_geotiff.tms import generate_tms
    from datafusion import udf
    from geoarrow.rust.core import from_wkb, multipolygon
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext
    from matplotlib import image as mpl_image
    from lonboard import Map, H3HexagonLayer, PolygonLayer, BitmapTileLayer
    from lonboard import RasterLayer
    from lonboard.raster import EncodedImage
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._geoarrow.ops import Bbox
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroArray,
        ArroTable,
        Bbox,
        BitmapTileLayer,
        CartoBasemap,
        EncodedImage,
        GeoTIFF,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        RasterLayer,
        S3Store,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        from_wkb,
        generate_tms,
        gzip,
        infer_rows_per_chunk,
        io,
        math,
        matplotlib,
        mo,
        mpl_image,
        multipolygon,
        np,
        obstore,
        pa,
        struct,
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
        """A one-line status readout the camera can write to, and the viewport ruler.

        A widget rather than `mo.md`, because the only way to update marimo output is to
        re-run the cell that produced it, and the cell holding the map is downstream of any
        state the camera could write: re-running it rebuilds the Map and throws the view
        away. A widget trait syncs straight to the browser instead.

        THE RULER, AND WHY IT LIVES HERE. lonboard's view_state carries longitude,
        latitude and zoom but NOT the canvas size, so the kernel cannot know how much
        world the screen shows: VIEW_W/VIEW_H were assumed, and going fullscreen made
        that assumption visibly wrong (cells folded for a 620 px band inside a 1400 px
        screen). This widget is always mounted just below the map, and every widget
        shares the page document, so it finds the deck canvas (the largest canvas on the
        page), measures its CSS size, and syncs it up as `view_wh`. Remeasured on window
        resize, on fullscreenchange (fullscreening an ELEMENT resizes no window, so a
        resize listener alone misses it), and via a ResizeObserver on the canvas itself
        for layout changes that are neither. Ported from the HFP notebook, where the
        fullscreen defect was found and the trait-type and shadow-DOM lessons were paid
        for.
        """

        _esm = """
        function render({ model, el }) {
          const line = document.createElement("div");
          line.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          // The browser's OWN reading, drawn from JS with no kernel round trip. When
          // the ruler works, this matches the px readout in the kernel's line above it;
          // when it does not, whichever half is missing names the broken leg. An error
          // in the measuring code lands here too instead of vanishing.
          const probe = document.createElement("div");
          probe.style.cssText =
            "font:10px ui-monospace,SFMono-Regular,Menlo,monospace;opacity:.4";
          const draw = () => { line.innerHTML = model.get("value"); };
          draw();
          model.on("change:value", draw);
          el.appendChild(line);
          // The diagnostic line, off by default. Everything still measures and syncs;
          // this only decides whether the browser-side reading is SHOWN.
          el.appendChild(probe);

          let watched = null;
          const ro = new ResizeObserver(() => kick());
          // marimo puts cell output inside shadow DOM, and document.querySelectorAll
          // does not pierce shadow roots: the deck canvas is on screen and invisible to
          // a plain query (measured: "no canvas found" while the map was clearly
          // there). So the search walks INTO every shadowRoot it passes.
          const collect = (root, out) => {
            root.querySelectorAll("canvas").forEach((c) => out.push(c));
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) collect(n.shadowRoot, out);
            });
          };
          const send = () => {
            try {
              let best = null, area = 0;
              const found = [];
              collect(document, found);
              found.forEach((c) => {
                const a = c.clientWidth * c.clientHeight;
                if (a > area) { area = a; best = c; }
              });
              let w, h, tag;
              if (best) {
                w = best.clientWidth; h = best.clientHeight; tag = "ruler ";
                if (best !== watched) {
                  if (watched) ro.unobserve(watched);
                  ro.observe(best);
                  watched = best;
                }
              } else {
                // No canvas even through the shadow roots: fall back to the window,
                // which OVERSTATES the map and costs a larger read, the cheap
                // direction to be wrong in. A band on screen is the expensive one.
                w = window.innerWidth; h = window.innerHeight; tag = "ruler window ";
              }
              if (w > 0 && h > 0) {
                probe.textContent = tag + w + "x" + h;
                // A string, not a number list: the only trait types this notebook has
                // PROVEN to cross marimo's anywidget bridge are Unicode (value, down)
                // and Bool (the Controls, up). The first ruler used List(Float) and
                // the kernel never heard a word.
                model.set("view_wh", w + "x" + h);
                model.save_changes();
              }
            } catch (err) {
              probe.textContent = "ruler error: " + err;
            }
          };
          let t = null;
          const kick = () => { clearTimeout(t); t = setTimeout(send, 250); };
          window.addEventListener("resize", kick);
          document.addEventListener("fullscreenchange", kick);
          setTimeout(send, 500);
        }
        export default { render };
        """
        value = traitlets.Unicode("").tag(sync=True)
        view_wh = traitlets.Unicode("").tag(sync=True)

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
          check("show_cells", "deforestation");
          check("show_divisions", "boundaries");
          check("division_fill", "boundary fill");

          // The ranking trigger, HERE rather than lonboard's draw-box tool. A Bool
          // toggle, not a counter: Bool is a trait type proven to cross marimo's
          // anywidget bridge browser -> kernel, and the kernel observer fires on any
          // change, so flipping the value is a click.
          const btn = document.createElement("button");
          btn.textContent = "rank what's in view";
          btn.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.15rem .6rem;border-radius:4px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          btn.onclick = () => {
            model.set("rank_view", !model.get("rank_view"));
            model.save_changes();
          };
          box.appendChild(btn);

          // BOUNDARY FILL OPACITY. A stepped slider (0.1-1.0 by 0.1) plus a free
          // number box (any 0-1 float); both write the same Unicode trait, because
          // Unicode is proven to cross marimo's bridge browser -> kernel (the Status
          // ruler's "WxH" string). Commit on 'change', not 'input': every commit
          // re-tints and re-pushes the divisions table, and Safari/Firefox fire
          // 'change' DURING a drag anyway (terrain notebook lesson), so the 0.1
          // steps are the real rate limiter.
          const ow = document.createElement("span");
          ow.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;opacity:.9";
          ow.appendChild(document.createTextNode("fill opacity"));
          const sl = document.createElement("input");
          sl.type = "range";
          sl.min = "0.1"; sl.max = "1"; sl.step = "0.1";
          sl.style.width = "6rem";
          const nb = document.createElement("input");
          nb.type = "number";
          nb.min = "0"; nb.max = "1"; nb.step = "any";
          nb.style.cssText =
            "width:3.6rem;font:inherit;background:transparent;color:inherit;" +
            "border:1px solid rgba(127,127,127,.45);border-radius:4px;" +
            "padding:0 .2rem";
          const seed0 = parseFloat(model.get("fill_alpha"));
          sl.value = nb.value = String(Number.isNaN(seed0) ? 0.65 : seed0);
          const commit = (v) => {
            v = Math.min(1, Math.max(0, v));
            sl.value = String(v);
            nb.value = String(v);
            model.set("fill_alpha", String(v));
            model.save_changes();
          };
          sl.onchange = () => commit(parseFloat(sl.value));
          nb.onchange = () => {
            const v = parseFloat(nb.value);
            if (!Number.isNaN(v)) commit(v);
          };
          ow.appendChild(sl);
          ow.appendChild(nb);
          box.appendChild(ow);
          el.appendChild(box);

          // HIDE LONBOARD'S DRAW-BOX TOOL. Its toolbar is rendered unconditionally in
          // the bundled JS (lonboard 0.16): the Map's `controls` trait governs only
          // fullscreen/navigation/scale, so there is no Python-side switch. The button
          // lives in lonboard's shadow root, hence the same recurse-into-shadowRoots
          // walk the Status ruler uses; an interval rather than a one-shot because the
          // map mounts after this widget and can be rebuilt by a cell re-run.
          const hideBbox = (root) => {
            let hid = false;
            root.querySelectorAll("button[aria-label]").forEach((b) => {
              const a = b.getAttribute("aria-label");
              if (a === "Select BBox" || a === "Cancel drawing" ||
                  a === "Clear bounding box") {
                const holder = b.closest("div[style*='absolute']") || b;
                holder.style.display = "none";
                hid = true;
              }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) hid = hideBbox(n.shadowRoot) || hid;
            });
            return hid;
          };
          setInterval(() => hideBbox(document), 1000);
        }
        export default { render };
        """
        show_cells = traitlets.Bool(True).tag(sync=True)
        show_divisions = traitlets.Bool(True).tag(sync=True)
        # ON BY DEFAULT. The join onto Overture is the whole point of this notebook and the
        # choropleth is what it produces, so shipping it behind an unticked box meant the
        # result was invisible unless you went looking for it.
        division_fill = traitlets.Bool(True).tag(sync=True)
        # Boundary fill opacity as a 0-1 float IN A STRING, per the proven-trait-types
        # rule (Unicode crosses the bridge both ways; numeric traits never made it).
        # "0.65" matches HOLD["fill_alpha"]'s 165 seed.
        fill_alpha = traitlets.Unicode("0.65").tag(sync=True)
        # The ranking trigger. Value is meaningless; a CHANGE is a click.
        rank_view = traitlets.Bool(False).tag(sync=True)

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
    # Below DIV_ZOOM there are no boundaries at all: the hexagons carry the map alone.
    # Under GeoParquet that was forced (a world view of countries meant reading most of
    # 5.5 GB to find 219 rows); under PMTiles a world of countries is 16 tiles at z2 and
    # the constraint is gone. The band is kept as a design choice: at the opening zoom
    # the map is about where deforestation IS, and country outlines over it answer a
    # question nobody has asked yet. Lower DIV_ZOOM if that reading changes.
    #
    # Overture has counties for 171 of 219 countries, so the county band is genuinely empty
    # in places rather than merely sparse.
    DIV_ZOOM = 4.5

    # TODO: a fourth band, `locality`, above roughly zoom 9.5. The tileset carries
    # localities from z10, so under PMTiles the cost question is already answered; what
    # remains is the meaning question. A locality boundary is a settlement, so most of a
    # drawn box would fall outside every polygon and the ranking would describe the towns
    # rather than the ground. Decide that before adding the band.
    def division_for_zoom(z):
        if z < DIV_ZOOM:
            return None
        if z < 7.0:
            return "region"
        return "county"

    DIVISION_LABEL = {"country": "countries", "region": "regions", "county": "counties"}

    # ------------------------------------------------------------------ boundaries
    # Overture's own PMTiles build of the same release the GeoParquet path used to read.
    # One object, anonymous ranged GETs, MVT tiles z0-12.
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"

    # The tile zoom at which each subtype FIRST appears in this tileset. Measured off the
    # tiles themselves (probe: Rondonia, Iowa, Congo, z2-z10), not documented anywhere:
    # Planetiler's minzoom rules are baked into the build. Every subtype persists from its
    # floor up to z12, so these are floors for the zoom picker, not bands.
    SUB_MINZOOM = {"country": 2, "region": 4, "county": 8}

    # ------------------------------------------------------------------ view
    # VIEW_W/VIEW_H and HOME moved INTO the map cell (2026-08-14, the cell split): a
    # constants edit must never re-run the map cell, because destroying the Map kills
    # deck's earcut pool. See the map cell.
    PAD = 1.25

    # SETTLE ONLY GUARDS A READ. Every camera event that can be answered from memory (a pan
    # inside the box already folded, a zoom back to a resolution already visited) is now
    # answered synchronously in the comm handler, so this delay is never spent on a view the
    # notebook already knows the answer to. It exists purely so a two-second drag issues one
    # object-store read at the end instead of a hundred along the way.
    SETTLE = 0.15

    # The fill alpha moved to HOLD["fill_alpha"] (2026-08-14): the Controls slider writes
    # it, and it must live somewhere no cell re-run can reset, which is HOLD's whole job.

    # The stroke alpha. Higher than the fill so the boundary still reads when the fill is
    # toggled off; the RGB underneath is the same ramp either way.
    LINE_ALPHA = 205

    return (
        COG,
        DIVISION_LABEL,
        FETCH_AT_ONCE,
        LEVEL_FOR_RES,
        LINE_ALPHA,
        MAX_RES,
        PAD,
        PM_BUCKET,
        PM_PATH,
        SETTLE,
        SOURCE_BUCKET,
        SUB_MINZOOM,
        TILE,
        TILE_BUDGET,
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
        "wh": (1400.0, 620.0),  # the real canvas size, measured by Status; this is the seed
        # Boundary fill alpha, 0-255. The Controls slider/number box writes it (as a 0-1
        # float, converted in _on_controls); divisions_to_layer reads it for every new
        # pair. 165 is the old FILL_ALPHA constant: not 255, so the deforestation paint
        # stays legible underneath and the fill reads as a wash rather than a lid.
        "fill_alpha": 165,
        "fold": None,  # the SQL fold, set by the read cell
        "zonal": None,  # cells -> division means, set by the read cell
        "rank": None,  # view box -> divisions ranked, set by the read cell
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
def _(
    ArroArray,
    ArroTable,
    HOLD,
    LINE_ALPHA,
    coordinates_to_cells,
    np,
    pa,
    ramp,
    ramp_rgba,
):
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
            ArroArray.from_arrow(tbl["region"].combine_chunks()),
            ArroArray.from_arrow(tbl["country"].combine_chunks()),
            ArroArray.from_arrow(pa.array(np.round(portion * 100, 4))),
            ArroArray.from_arrow(tbl["n_cells"].combine_chunks()),
        ]
        names = [
            "geometry",
            "color",
            "line",
            "name",
            "region",
            "country",
            "deforested %",
            "cells",
        ]

        # The stroke takes the same ramp as the fill, but from its OWN column: the fill
        # toggle works by swapping to a table whose `color` alpha is zero, and a line fed
        # from that column would vanish with it. One line column, shared by both variants,
        # at the stroke's own alpha.
        line = ArroArray.from_arrow(
            pa.FixedSizeListArray.from_arrays(
                pa.array(ramp_rgba(portion, LINE_ALPHA).ravel()), 4
            )
        )

        def build(alpha):
            col = ArroArray.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(ramp_rgba(portion, alpha).ravel()), 4
                )
            )
            return ArroTable.from_arrays([geom, col, line, *rest], names=names)

        # HOLD["fill_alpha"], read at build time: a pair built after the slider moved
        # carries the new alpha without any extra machinery. Pairs built BEFORE the move
        # are re-tinted in place by _refill in the map cell.
        return build(HOLD["fill_alpha"]), build(0)

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
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([0, 0, 0, 0], dtype=np.uint8)), 4
                    )
                ),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([0.0])),
                ArroArray.from_arrow(pa.array([0], type=pa.int64())),
            ],
            names=[
                "geometry",
                "color",
                "line",
                "name",
                "region",
                "country",
                "deforested %",
                "cells",
            ],
        )

    return cells_to_layer, divisions_to_layer, seed_cells, seed_divisions


@app.cell
async def _(
    PM_BUCKET,
    PM_PATH,
    S3Store,
    SUB_MINZOOM,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    struct,
):
    # DIVISIONS COME OUT OF ONE PMTILES OBJECT, BY RANGED GET. The GeoParquet path this
    # replaces was measured to the floor first (see the notes doc): geometry is 99.0% of a
    # row group's bytes, `subtype` statistics prune nothing, and client concurrency was
    # not the bottleneck, so a Rondonia-sized viewport cost ~190 MB of decode per file and
    # no query could make it smaller. The tileset is the same release with the layout
    # problem solved upstream: Hilbert-ordered MVT tiles in one archive, so a viewport is
    # a handful of contiguous ranges. The same viewport reads ~0.8 MB.
    #
    # The reader is the one from xsql-duckdb-terrain-h3.py (Mapterhorn), ported. That
    # notebook is parked on looks; its PMTiles v3 client is the good part. Opening costs
    # two reads (127-byte header, root directory), then one leaf directory per region
    # touched, parsed once and cached.
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

        Entries are (tile_id, offset, length, run_length). run_length 0 marks a pointer
        to a LEAF directory rather than to a tile. A zero OFFSET means "immediately after
        the previous entry", so offsets are reconstructed in order, not read.
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

        Hilbert rather than row-major so tiles near each other on the GROUND are near
        each other in the FILE, which is what makes a viewport a few contiguous ranges.
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

        The fallback is not an optimisation: directories are run-length encoded, so a
        tile usually has no entry of its own and is covered by an earlier one.
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
    PM_MAXZ = _hdr[101]
    assert max(SUB_MINZOOM.values()) <= PM_MAXZ, "SUB_MINZOOM above the pyramid"
    _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
    _leaf = {}

    # ------------------------------------------------------------- the MVT decode
    # Hand-rolled rather than a dependency, deliberately: an MVT is three nested protobuf
    # messages whose only wire types are varint and length-delimited, and the varint
    # machinery already exists two functions up. Verified ring-exact and property-exact
    # against mapbox-vector-tile on ten real tiles (the world tile, Java's coastline,
    # Italy's enclaves) before being trusted with anything.
    def _fields(buf):
        """Iterate (field_number, wire_type, value) over one protobuf message."""
        i, n = 0, len(buf)
        while i < n:
            key, i = _varint(buf, i)
            f, w = key >> 3, key & 0x7
            if w == 0:
                v, i = _varint(buf, i)
            elif w == 2:
                ln, i = _varint(buf, i)
                v = buf[i : i + ln]
                i += ln
            elif w == 5:
                v = buf[i : i + 4]
                i += 4
            elif w == 1:
                v = buf[i : i + 8]
                i += 8
            else:
                raise ValueError(f"wire type {w}")
            yield f, w, v

    def _value(buf):
        """An MVT Value message: exactly one of its fields is set."""
        for f, _w, v in _fields(buf):
            if f == 1:
                return v.decode("utf-8")
            if f == 2:
                return struct.unpack("<f", v)[0]
            if f == 3:
                return struct.unpack("<d", v)[0]
            if f in (4, 5):
                return v
            if f == 6:
                return (v >> 1) ^ -(v & 1)
            if f == 7:
                return bool(v)
        return None

    def _mvt_rings(geom):
        """Packed geometry commands -> rings of (x, y) tile coords, closed."""
        rings, ring = [], None
        x = y = 0
        i, n = 0, len(geom)
        while i < n:
            cmd, i = _varint(geom, i)
            op, count = cmd & 0x7, cmd >> 3
            if op == 1:  # MoveTo: starts a ring
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring = [(x, y)]
                    rings.append(ring)
            elif op == 2:  # LineTo
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring.append((x, y))
            elif op == 7:  # ClosePath: repeat the first point
                ring.append(ring[0])
            else:
                raise ValueError(f"geometry op {op}")
        return rings

    def _area2(ring):
        """Twice the signed shoelace area: >0 marks an exterior ring.

        The spec says exteriors wind clockwise ON SCREEN, and tile y points down, so a
        clockwise-on-screen ring is counterclockwise in plain (x, y) axes and the
        standard shoelace sum comes out positive with no sign flip. Getting the sign
        backwards classifies every ring as a hole and decodes every feature to nothing.
        """
        a = 0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            a += x0 * y1 - x1 * y0
        return a

    def _division_areas(tile_buf):
        """The division_area layer: ([(properties, [(exterior, holes), ...]), ...], extent)."""
        for f, _w, v in _fields(tile_buf):
            if f != 3:  # Tile.layers
                continue
            name, extent = None, 4096
            keys, values, feats = [], [], []
            for lf, _lw, lv in _fields(v):
                if lf == 1:
                    name = lv.decode("utf-8")
                elif lf == 2:
                    feats.append(lv)
                elif lf == 3:
                    keys.append(lv.decode("utf-8"))
                elif lf == 4:
                    values.append(_value(lv))
                elif lf == 5:
                    extent = lv
            if name != "division_area":
                continue
            out = []
            for fv in feats:
                tags, gtype, geom = [], 0, b""
                for ff, _fw, fvv in _fields(fv):
                    if ff == 2:
                        i = 0
                        while i < len(fvv):
                            t, i = _varint(fvv, i)
                            tags.append(t)
                    elif ff == 3:
                        gtype = fvv
                    elif ff == 4:
                        geom = fvv
                if gtype != 3:  # not a polygon feature
                    continue
                props = {
                    keys[tags[i]]: values[tags[i + 1]] for i in range(0, len(tags), 2)
                }
                polys, cur = [], None
                for ring in _mvt_rings(geom):
                    if _area2(ring) > 0:
                        cur = (ring, [])
                        polys.append(cur)
                    elif cur is not None:
                        cur[1].append(ring)
                out.append((props, polys))
            return out, extent
        return [], 4096

    def _feature_wkb(polys, z, x, y, extent):
        """Tile-integer rings -> a lon/lat MultiPolygon WKB.

        Web Mercator is closed form in both directions, so a tile coordinate knows its
        own lon/lat exactly: x is linear in longitude and y is the inverse Gudermannian
        of latitude. The (x, y) row layout of the ring array is already WKB point order,
        so each ring serialises as a length prefix plus the raw float64 bytes.
        """
        n = 1 << z
        parts = []
        for ext, holes in polys:
            rings = []
            for r in (ext, *holes):
                a = np.asarray(r, dtype=np.float64)
                pts = np.empty_like(a)
                pts[:, 0] = (x + a[:, 0] / extent) / n * 360.0 - 180.0
                pts[:, 1] = np.degrees(
                    np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + a[:, 1] / extent) / n)))
                )
                rings.append(struct.pack("<I", len(a)) + pts.tobytes())
            parts.append(struct.pack("<BII", 1, 3, len(rings)) + b"".join(rings))
        return struct.pack("<BII", 1, 6, len(parts)) + b"".join(parts)

    # Decoded per tile and kept, same bargain as the raster tile cache: a pan re-reads
    # the strip it has not seen and nothing else, and a zoom back to a band already
    # visited is free. Entries are a few dozen small dicts each, so the cap is a count.
    _tiles = {}  # (z, x, y) -> [piece, ...]; insertion order is LRU order
    TILE_KEEP = 2048
    _sem = asyncio.Semaphore(32)

    async def _tile_pieces(z, x, y):
        """One tile, walked to through the directories, decoded, filtered to land.

        A piece is one division's presence in one tile: division id, name, country,
        subtype, and the clipped geometry as WKB. The maritime half of division_area is
        dropped here (is_land is always present in this tileset: measured True 431 /
        False 325 over seven tiles, never missing) for the same reason the GeoParquet
        path dropped it: open water was never at risk of being deforested, and it drags
        a coastal division's zonal mean toward zero.
        """
        k = (z, x, y)
        if k in _tiles:
            _tiles[k] = _tiles.pop(k)  # touch: young end of the LRU
            return _tiles[k]
        tid, ents = _tile_id(z, x, y), _root
        blob = None
        for _ in range(4):  # root + up to three leaf levels
            e = _find(ents, tid)
            if e is None:
                break
            if e[3] == 0:
                lk = (e[1], e[2])
                if lk not in _leaf:
                    _leaf[lk] = _parse_dir(
                        gzip.decompress(
                            await _pm_range(_ld_off + e[1], _ld_off + e[1] + e[2] - 1)
                        )
                    )
                ents = _leaf[lk]
                continue
            async with _sem:
                blob = await _pm_range(_td_off + e[1], _td_off + e[1] + e[2] - 1)
            break
        pieces = []
        if blob is not None:
            if blob[:2] == b"\x1f\x8b":  # tile_compression says gzip; trust the bytes
                blob = gzip.decompress(blob)
            feats, extent = _division_areas(blob)
            for props, polys in feats:
                if props.get("is_land") is not True or not polys:
                    continue
                pieces.append(
                    {
                        "sub": props.get("subtype"),
                        # division_id, not id: `id` names this AREA row and a division
                        # can own several, `division_id` names the DIVISION, which is
                        # the thing being coloured and ranked. Same lesson as the
                        # division_boundary experiment, where joining on the wrong one
                        # silently returned zero rows.
                        "id": props.get("division_id") or props.get("id"),
                        "name": props.get("@name"),
                        "country": props.get("country"),
                        # ISO 3166-2 (e.g. "US-KS") on county and region features;
                        # absent on countries, which have nothing above them.
                        "region": props.get("region"),
                        "wkb": _feature_wkb(polys, z, x, y, extent),
                    }
                )
        _tiles[k] = pieces
        while len(_tiles) > TILE_KEEP:
            _tiles.pop(next(iter(_tiles)))
        return pieces

    # ------------------------------------------------------------- which tiles
    def _tz_for(subtype, box):
        """Tile zoom for a box: ~4 tiles across, floored at the subtype's minzoom.

        The floor matters at the top of a band (counties do not exist below z8) and the
        box-derived term everywhere else; both are capped by the pyramid. Finer than
        needed would be more requests for geometry the polyfill cannot see, coarser
        would quantize below the cells: at z8 a tile unit is ~38 m against 460 m cells.
        """
        span = max(box[2] - box[0], 1e-9)
        z = int(math.log2(max(4.0 * 360.0 / span, 1.0)))
        return max(SUB_MINZOOM[subtype], min(PM_MAXZ, z))

    def _mtile(lon, lat, z):
        """lon/lat -> tile x, y at z, clamped to the grid."""
        n = 1 << z
        xx = min(n - 1, max(0, int((lon + 180.0) / 360.0 * n)))
        la = min(85.05, max(-85.05, lat))
        yy = (
            1.0
            - math.log(math.tan(math.radians(la)) + 1.0 / math.cos(math.radians(la)))
            / math.pi
        ) / 2.0
        return xx, min(n - 1, max(0, int(yy * n)))

    def _range_box(z, x0, y0, x1, y1):
        """The lon/lat box a tile range actually covers: the coverage the memo checks."""
        n = 1 << z

        def lat(yy):
            return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yy / n))))

        return (
            x0 / n * 360.0 - 180.0,
            lat(y1 + 1),
            (x1 + 1) / n * 360.0 - 180.0,
            lat(y0),
        )

    # A drawn box has no zoom band protecting it, so a continent-sized box can ask for
    # county tiles by the thousand (the old path's version of this was a 596-second
    # polyfill). Refusing over the cap makes rank() fall back to the next coarser
    # subtype, the same promise it already makes where counties do not exist at all.
    TILE_CAP = 256

    # CACHED BY COVERAGE, NOT BY EXACT BOX, same as before: the box is grown, snapped to
    # the tile grid, and any later box inside that coverage is a lookup. The disk ledger
    # the GeoParquet path carried is gone: it existed because a cold read cost 18 s, and
    # a cold read is now under a second.
    _div_mem = {}  # subtype -> [[coverage box, table, key], ...], newest last
    DIV_PAD = 1.4
    DIV_KEEP = 8

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

    def _remember(subtype, box, table, key):
        held = _div_mem.setdefault(subtype, [])
        held.append([box, table, key])
        del held[:-DIV_KEEP]

    # THE SEAM DISSOLVE. Tile geometry arrives clipped, so one division is several
    # pieces, and the pieces' clip edges are straight lines the stroke would draw across
    # the map. Union-ing the pieces per division removes every interior edge, and the
    # tile buffer (pieces overlap slightly past each tile edge) is what makes the union
    # clean rather than a float-tolerance lottery. Clip edges survive only at the OUTER
    # boundary of the fetched range, which sits a full DIV_PAD beyond the viewport, so a
    # camera inside the coverage box never has one on screen.
    DISSOLVE_SQL = """
        SELECT id,
               any_value(name)    AS name,
               any_value(country) AS country,
               any_value(region)  AS region,
               CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
        FROM pieces
        GROUP BY id
    """

    def _dissolve(pieces_tbl):
        pieces = pieces_tbl  # noqa: F841 - read by DuckDB's replacement scan
        return con.sql(DISSOLVE_SQL).to_arrow_table()

    async def fetch_divisions(subtype, bbox):
        """Overture divisions of one subtype covering bbox, geometry as WKB.

        Returns (table or None, key). The key names the tile range rather than the
        request, so the polyfill can memoise against it: two viewports served by one
        coverage share one set of filled cells.
        """
        for box, tbl, key in _div_mem.get(subtype, []):
            if _inside(box, bbox):
                return tbl, key

        big = _grow(bbox)
        tz = _tz_for(subtype, big)
        x0, y0 = _mtile(big[0], big[3], tz)
        x1, y1 = _mtile(big[2], big[1], tz)
        key = (subtype, tz, x0, y0, x1, y1)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > TILE_CAP:
            # NOT remembered: a smaller box inside this coverage deserves a finer zoom
            # and a real answer, so a refusal must never be served from the memo.
            return None, key

        parts = await asyncio.gather(
            *(
                _tile_pieces(tz, xx, yy)
                for yy in range(y0, y1 + 1)
                for xx in range(x0, x1 + 1)
            )
        )
        rows = [p for tp in parts for p in tp if p["sub"] == subtype]
        cov = _range_box(tz, x0, y0, x1, y1)
        if not rows:
            _remember(subtype, cov, None, key)
            return None, key

        pieces = pa.table(
            {
                "id": pa.array([r["id"] for r in rows]),
                "name": pa.array([r["name"] for r in rows]),
                "country": pa.array([r["country"] for r in rows]),
                # The country prefix is dropped ("US-KS" -> "KS") because country is
                # already its own column everywhere this is shown.
                "region": pa.array(
                    [(r["region"] or "").split("-", 1)[-1] for r in rows]
                ),
                "wkb": pa.array([r["wkb"] for r in rows], pa.binary()),
            }
        )
        out = _dissolve(pieces)
        _remember(subtype, cov, out, key)
        return out, key

    # THE POLYFILL, AND THE ONE THING THAT MAKES IT AWKWARD.
    #
    # h3_polygon_wkb_to_cells_experimental takes a POLYGON and raises
    #   Invalid WKB: expected polygon at 5
    # on a MultiPolygon, which every dissolved division is. So each division is split
    # with ST_Dump, every part filled, and the parts flattened back into one distinct
    # cell set per division. Cheap, but not optional, and the error names the WKB rather
    # than the geometry type, so it reads like corruption.
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
async def _(
    Bbox,
    COG,
    EncodedImage,
    GeoTIFF,
    RasterLayer,
    S3Store,
    SOURCE_BUCKET,
    TILE,
    generate_tms,
    io,
    mpl_image,
    np,
    ramp,
):
    # THE RASTER, DRAWN AS ITSELF. RasterLayer.from_geotiff would almost do this, but two
    # of its choices are wrong for this COG, so the layer is constructed directly with the
    # same private arguments from_geotiff passes (lonboard 0.16, pinned; the repo already
    # reaches for lonboard._serialization, so this is house style):
    #
    #   1. Its fetch is not sparse-aware. 73.6% of full-res tiles are unstored ocean, and
    #      `fetch_tile` on one raises the same `Invalid range requested, start: 0 end: 0`
    #      the fold cell documents. The fetch below consults `ifd.tile_byte_counts` first,
    #      same fix as the fold: an absent tile is None, no request at all.
    #   2. Its min_zoom/max_zoom ship commented out, and the fetcher indexes
    #      `images[len - 1 - z]`, so an overzoomed request wraps negative onto a COARSE
    #      overview: wrong data, silently. max_zoom pins z to the pyramid (11 levels,
    #      z0 = coarsest overview, z10 = the 100 m full res).
    #
    # This is a SECOND header-only open of the COG. It cannot share the fold cell's
    # GeoTIFF: the fold cell depends on `refresh` from the wiring, which depends on the
    # map cell, which needs this layer, so sharing would be a cycle. Headers are ~one
    # ranged read; the pixel caches are separate and that is fine, the browser's tile
    # cache is the one that matters here.
    #
    # The TMS is the COG's own grid in EPSG:4326; deck.gl-raster reprojects client-side,
    # so there is no mercator warp and no resampling in the kernel.
    _rstore = S3Store(SOURCE_BUCKET, region="us-west-2", skip_signature=True)
    _rg = await GeoTIFF.open(COG, store=_rstore)
    _rimgs = [_rg, *_rg.overviews]
    _rpresent = []
    for _rlv in _rimgs:
        _rnty = -(-_rlv.shape[0] // TILE)
        _rntx = -(-_rlv.shape[1] // TILE)
        _rpresent.append(
            np.asarray(_rlv.ifd.tile_byte_counts).reshape(_rnty, _rntx) > 0
        )

    async def _fetch(x, y, z):
        li = len(_rimgs) - 1 - z
        if not 0 <= li < len(_rimgs):
            return None
        p = _rpresent[li]
        if not (0 <= y < p.shape[0] and 0 <= x < p.shape[1]) or not p[y, x]:
            return None
        # boundless=True, AND IT IS THE FIX FOR THE STREAKED WORLD VIEW (2026-08-14,
        # seen on the first flight and misread as a projection bug). boundless=False
        # clips edge tiles to the image, and deck stretches whatever PNG it gets across
        # the FULL tile quad; at coarse levels nearly every tile is an edge tile (the
        # coarsest is one 195x391 image in a 512 px tile), so a zoomed-out view was
        # mostly tiles smeared ~2.6x vertically: horizontal streaks. boundless=True
        # pads to 512x512; the padding arrives as 0.0 (measured, not NaN), which the
        # render below already maps to alpha 0, so it costs nothing visible. If zero
        # ever stops being transparent here, the padding must be masked instead.
        return await _rimgs[li].fetch_tile(x, y, boundless=True)

    def _render(tile):
        # Exceptions in here are swallowed by the layer's task machinery, so keep it
        # simple. `.array`, not `.data` (CLAUDE.md). Alpha 0 for NaN ocean AND for exact
        # zero, matching the fold's `HAVING avg(v) > 0`: this map shows where
        # deforestation IS, and the raster must not disagree with the hexagons it
        # replaces by painting the 69.6% zero majority. matplotlib's PNG writer, so no
        # new dependency.
        if tile is None:
            return None
        v = np.ma.filled(tile.array.as_masked()[0].astype("float64"), np.nan)
        rgba = np.empty(v.shape + (4,), np.uint8)
        rgba[..., :3] = ramp(v)
        rgba[..., 3] = np.where(np.isfinite(v) & (v > 0), 255, 0)
        buf = io.BytesIO()
        mpl_image.imsave(buf, rgba, format="png")
        return EncodedImage(data=buf.getvalue(), media_type="image/png")

    _rb = _rg.bounds
    raster = RasterLayer(
        _tile_matrix_set=generate_tms(_rg),
        _crs=_rg.crs,
        _fetch_tile=_fetch,
        _render_tile=_render,
        _bounds=Bbox(_rb.left, _rb.bottom, _rb.right, _rb.top),
        _center=((_rb.left + _rb.right) / 2, (_rb.bottom + _rb.top) / 2),
        min_zoom=0,
        max_zoom=len(_rimgs) - 1,
        opacity=0.7,
        pickable=False,
        visible=True,
    )
    return (raster,)


@app.cell
def _(
    BitmapTileLayer,
    CartoBasemap,
    Controls,
    HOLD,
    Map,
    MaplibreBasemap,
    Panel,
    PolygonLayer,
    Status,
    from_wkb,
    multipolygon,
    seed_divisions,
):
    # THE MAP CELL, AND WHY IT DEPENDS ON ALMOST NOTHING (split 2026-08-14, the flood
    # notebook's pattern, after Stephen reported losing the boundary fill on re-runs).
    # Destroying a lonboard Map terminates deck's earcut worker pool, which is
    # MODULE-LEVEL in the page: after that every polygon layer on the page fails to
    # initialize ("Cannot schedule pool tasks after terminate()") until the browser
    # reloads. Hexagon/raster/bitmap layers survive, which is why it presented as
    # "lost the fill" specifically. So this cell must never re-run on an ordinary
    # edit: it depends only on imports, widget classes, seeds and HOLD. Everything
    # editable (constants, ramp, handlers, the draw logic) lives in the WIRING cell
    # below, which re-hooks onto these surviving widgets. The RasterLayer is NOT a
    # dependency either; the wiring cell inserts it via `deck.layers`, so a ramp or
    # constants edit rebuilds the raster layer and re-wires without touching the Map.
    #
    # EDITING THIS CELL ITSELF still tears the deck down: restart the kernel AND
    # reload the browser page afterwards.
    #
    # VIEW_W/VIEW_H and HOME live here rather than in the constants cell for the same
    # reason: a constants edit must not reach this cell. The size seeds HOLD["wh"]
    # until the Status ruler reports the real canvas; HOME only seeds a fresh session
    # (and headless runs), because on a wiring re-run the camera survives in
    # HOLD["vs"] with the deck.
    VIEW_W, VIEW_H = 1400, 620
    # Opens on the tropics, because that is where the data is: the Amazon, the Congo
    # basin and insular southeast Asia are the three places a 2002-2022 deforestation
    # layer has anything dramatic to say, and all three are in view from here.
    HOME = {"longitude": -20.0, "latitude": 0.0, "zoom": 2.4}

    status = Status(value="<b>loading…</b>")
    controls = Controls()
    ranking = Panel()

    # H3 VIZ PARKED (2026-08-14). The RasterLayer draws the deforestation paint now, at
    # pixel resolution instead of cell means; the fold still runs untouched because the
    # divisions join eats its cells. To restore the hexagons: uncomment this block
    # (H3HexagonLayer and seed_cells go back in this cell's signature), return `cells`,
    # add it to `deck.layers` in the wiring cell, uncomment put_cells' body there, and
    # repoint the Controls checkbox in _on_controls back at it. Then restart the kernel
    # AND reload the page: this is the map cell.
    # _seed = seed_cells()
    # cells = H3HexagonLayer(
    #     table=_seed,
    #     get_hexagon=_seed["hex"],
    #     get_fill_color=_seed["color"],
    #     extruded=False,
    #     stroked=False,
    #     high_precision=True,
    #     coverage=1.0,
    #     opacity=0.3,
    #     pickable=True,
    # )

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
        get_line_color=_dseed["line"],
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
        opacity=0.6,
        pickable=False,
    )

    # The raster layer is deliberately absent here; the wiring cell inserts it under
    # the divisions via `deck.layers`, so this cell survives raster rebuilds.
    deck = Map(
        [divisions, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state=HOME,
        height=VIEW_H,
        show_tooltip=True,
    )

    # The HOLD state tied to the WIDGET's lifetime rather than the wiring's: the
    # canvas seed (the ruler overwrites it) and the camera. Everything the wiring's
    # caches describe is reset in the wiring cell instead.
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["vs"] = None
    return HOME, controls, deck, divisions, labels, ranking, status


@app.cell
def _(
    ArroTable,
    DIVISION_LABEL,
    HOLD,
    HOME,
    PAD,
    SETTLE,
    STOPS,
    asyncio,
    cells_to_layer,
    controls,
    deck,
    division_for_zoom,
    divisions,
    infer_rows_per_chunk,
    labels,
    np,
    pa,
    ramp,
    ranking,
    raster,
    res_for_zoom,
    status,
):
    # THE WIRING CELL: everything editable about how the map behaves. Re-runs on any
    # edit to the constants, the ramp, the raster layer or the machinery; cancels the
    # old work, unhooks the old observers (via the HOLD["h_*"] refs), and re-hooks
    # onto the SURVIVING deck from the map cell above. The widget is never destroyed,
    # so deck's shared earcut pool stays alive and the camera stays where the user
    # left it. The screen-state keys are reset because the caches they describe were
    # just rebuilt; wh and vs belong to the map cell and are left alone.
    for _t in ("task", "seltask"):
        if HOLD[_t] is not None:
            HOLD[_t].cancel()
        HOLD[_t] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["res"], HOLD["box"], HOLD["div"] = None, None, None
    HOLD["divpair"], HOLD["divbox"] = None, None
    HOLD["head"], HOLD["tail"] = "", ""
    HOLD["cache"].clear()

    # The full layer stack, raster under divisions under labels. Assigned every
    # wiring run: on a raster-cell re-run (a ramp or constants edit) this is what
    # swaps the fresh RasterLayer widget into the surviving Map.
    deck.layers = [raster, divisions, labels]

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N], clamped to the world.

        Web Mercator: the horizontal span is a straight function of zoom, and the vertical
        span is that scaled by the aspect ratio and by cos(latitude), because a degree of
        longitude narrows toward the poles.

        The size comes from HOLD["wh"], which is the MEASURED canvas, not the old
        VIEW_W/VIEW_H guess: view_state has no width or height in it, so the Status
        widget rulers the deck canvas in the browser and syncs it up. Fullscreen was
        where the guess failed visibly.
        """
        import math as _m

        vw, vh = HOLD["wh"]
        span = 360.0 * vw / (512 * 2**vs.zoom)
        lat_span = span * (vh / vw) * _m.cos(_m.radians(vs.latitude))
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

        THE VIEWPORT DIAGNOSTICS ARE ENABLED, matching the HFP notebook, where the
        fullscreen defect has been seen again since the ruler landed and is not yet
        closed. The px readout below is the kernel's half (1400x620 that never moves
        means the browser has not reported in); el.appendChild(probe) in Status._esm is
        the browser's half, no kernel involved. The two disagreeing names the broken
        leg. Comment both out once the defect is closed.
        """
        status.value = (
            f"{HOLD['head']}{HOLD['tail']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

    def put_cells(tbl):
        # H3 VIZ PARKED (2026-08-14): nothing is pushed to the browser. The callers and
        # the cache keep working, so restoring the hexagons is uncommenting.
        return
        # cells._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # hold_sync so deck gets one message. Without it the new hexagons are drawn
        # against the old colour buffer for a frame.
        # with cells.hold_sync():
        #     cells.table = tbl
        #     cells.get_hexagon = tbl["hex"]
        #     cells.get_fill_color = tbl["color"]
        #     cells.visible = controls.show_cells

    def put_divisions(pair):
        """Push whichever of the two colour variants the fill switch is asking for."""
        tbl = pair[0] if controls.division_fill else pair[1]
        divisions._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with divisions.hold_sync():
            divisions.table = tbl
            divisions.get_fill_color = tbl["color"]
            divisions.get_line_color = tbl["line"]
            divisions.visible = controls.show_divisions
            # Picking ignores alpha: the zero-alpha fill still swallows every hover, so
            # with the fill off the layer must stop picking or the cells under it go dead.
            divisions.pickable = bool(controls.division_fill)
        HOLD["divpair"] = pair

    def _refill(tbl, alpha):
        """Re-tint a finished filled-variant divisions table to a new fill alpha.

        Whole-table `pa.table(tbl)`, not per-column: arro3 exposes the C-stream
        protocol at TABLE level only (the terrain notebook's recolor lesson), and the
        geoarrow extension metadata on the geometry column survives the round trip.
        RGB is untouched; only the alpha plane is rewritten, so re-tints are
        idempotent and cached pairs can be re-tinted any number of times.
        """
        pt = pa.table(tbl)
        col = pt["color"].combine_chunks()
        rgba = np.asarray(col.values).reshape(-1, 4).copy()
        rgba[:, 3] = alpha
        new = pa.FixedSizeListArray.from_arrays(pa.array(rgba.ravel()), 4)
        return ArroTable.from_arrow(
            pt.set_column(pt.schema.get_field_index("color"), "color", new)
        )

    def _on_controls(change):
        name = change["name"]
        if name == "show_cells":
            # The checkbox now governs the raster paint (labelled "deforestation" in the
            # Controls widget); it toggled the hexagons before they were parked.
            raster.visible = bool(change["new"])
        elif name == "show_divisions":
            divisions.visible = bool(change["new"]) and HOLD["divpair"] is not None
        elif name == "division_fill":
            # A whole re-push, not an accessor assignment. See divisions_to_layer.
            if HOLD["divpair"] is not None:
                put_divisions(HOLD["divpair"])
        elif name == "fill_alpha":
            # The string is the slider's or the number box's 0-1 float; anything
            # unparseable is ignored rather than crashed on, because this runs inside
            # a comm handler where exceptions are silent (the reverse-cmap defect).
            try:
                _a = float(change["new"])
            except (TypeError, ValueError):
                return
            HOLD["fill_alpha"] = int(round(min(1.0, max(0.0, _a)) * 255))
            if HOLD["divpair"] is not None:
                put_divisions(
                    (
                        _refill(HOLD["divpair"][0], HOLD["fill_alpha"]),
                        HOLD["divpair"][1],
                    )
                )

    # Re-hook, never stack: a wiring re-run must first unhook the handlers the LAST
    # run registered on these surviving widgets, or every edit adds another listener
    # and one click fans out N times. The try/except covers the one case the refs go
    # stale: a map-cell re-run built fresh widgets the old handlers were never on.
    _CTL_NAMES = ["show_cells", "show_divisions", "division_fill", "fill_alpha"]
    if HOLD.get("h_ctl") is not None:
        try:
            controls.unobserve(HOLD["h_ctl"], names=_CTL_NAMES)
        except ValueError:
            pass
    controls.observe(_on_controls, names=_CTL_NAMES)
    HOLD["h_ctl"] = _on_controls

    def _on_wh(change):
        """The canvas changed size: fullscreen, a window resize, a layout shift.

        The measured size replaces the guess, and if the box already folded no longer
        covers what the bigger canvas now shows, the same path a camera move takes will
        refold it. Sub-25 px jitter is ignored: PAD absorbs it, and a ResizeObserver
        will happily fire on a 1 px scrollbar appearing.
        """
        try:
            wh = [float(v) for v in str(change["new"]).split("x")]
        except ValueError:
            return
        if len(wh) != 2 or wh[0] <= 0 or wh[1] <= 0:
            return
        old = HOLD["wh"]
        HOLD["wh"] = (wh[0], wh[1])
        vs = HOLD["vs"]
        if vs is not None:
            set_status(vs)  # the px readout is the ruler's proof of life
        if abs(wh[0] - old[0]) < 25 and abs(wh[1] - old[1]) < 25:
            return
        if vs is None:
            return
        if HOLD["busy"]:
            HOLD["pending"] = vs
        elif not _instant(vs):
            HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_wh") is not None:
        try:
            status.unobserve(HOLD["h_wh"], names="view_wh")
        except ValueError:
            pass
    status.observe(_on_wh, names="view_wh")
    HOLD["h_wh"] = _on_wh

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
            # Zoomed too far out for boundaries, by design rather than by cost now: see
            # division_for_zoom for why the top band stays hexagons-only under PMTiles.
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

    if HOLD.get("h_cam") is not None:
        try:
            deck.unobserve(HOLD["h_cam"], names="view_state")
        except ValueError:
            pass
    deck.observe(_on_camera, names="view_state")
    HOLD["h_cam"] = _on_camera

    # ---------------------------------------------------------------- the ranking
    # Press "rank what's in view" in the controls and the divisions on screen come back
    # ranked, below. This is the one place the join produces a NUMBER rather than a
    # colour. The box is the view itself, but the READ is still an explicit ask: it goes
    # one level FINER than the screen and it names the divisions outright.
    RANK_N = 25

    def rank_html(out):
        if out is None:
            return (
                "<div style='font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
                "opacity:.75;padding:.5rem 0'>No division in view caught a cell centre. "
                "Zoom out for larger divisions, or in for finer cells.</div>"
            )
        sub, res, tbl, n_small = out
        names = tbl["name"].to_pylist()
        country = tbl["country"].to_pylist()
        region = tbl["region"].to_pylist()
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
            where = ", ".join(filter(None, [region[i], country[i] or "??"]))
            rows.append(
                f"<tr>"
                f"<td style='text-align:right;opacity:.5;padding:.12rem .5rem .12rem 0'>{place}</td>"
                f"<td style='padding:.12rem .6rem .12rem 0;white-space:nowrap'>{label}"
                f"<span style='opacity:.45'> · {where}</span></td>"
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
            f"<b>{len(names):,} {DIVISION_LABEL[sub]} in view</b>, ranked by mean share "
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

    def _on_rank_view(change):
        """The ranking, for WHAT IS ON SCREEN. Replaces lonboard's draw-box tool.

        The drawn box asked the user to describe a region twice: once with the camera
        and again with a rectangle. The button keeps the camera as the only statement
        of intent: the ranked box is exactly the view, through the same view_to_bbox
        the fold uses. The Bool's value carries nothing; any change is a click.
        """
        vs = HOLD["vs"]
        if vs is None:
            # No camera event yet: the map still shows HOME, so rank that.
            from types import SimpleNamespace

            vs = SimpleNamespace(**HOME)
        box = view_to_bbox(vs)
        ranking.value = (
            "<div style='font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;"
            "opacity:.7;padding:.5rem 0'><b>ranking</b> the divisions in view…</div>"
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

    if HOLD.get("h_rank") is not None:
        try:
            controls.unobserve(HOLD["h_rank"], names="rank_view")
        except ValueError:
            pass
    controls.observe(_on_rank_view, names="rank_view")
    HOLD["h_rank"] = _on_rank_view

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
        "<b style='margin-right:.7rem'>share of ground deforested 2002-2022</b>"
        f"{_sw}</div>"
    )
    return legend, refresh


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
        """Every division inside the ranked box, with its mean, for the panel below the map.

        Returns (subtype, resolution, table, divisions with no number) or None.

        THREE THINGS THIS DOES NOT SHARE WITH THE CAMERA, AND WHY.
        1. It reads ONE RESOLUTION FINER than the screen would. The button is an explicit
           question about a specific place, so it is worth a read the camera would not
           spend, and the finer the cells the fewer divisions fall through the 'center' rule.
        2. It derives that resolution from the BOX, not from the camera's zoom trait, so
           the two cannot drift.
        3. It falls back county -> region -> country. Overture has counties for 171 of 219
           countries, so a box over the other 48 would otherwise come back empty rather than
           answering at the finest level that exists there.
        """
        span = max(box[2] - box[0], 1e-9)
        z = math.log2(360.0 * HOLD["wh"][0] / (512 * span))
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

    # The opening draw, forced: nothing to debounce yet. On a WIRING re-run the camera
    # survives with the deck, so the redraw targets wherever the user left it, not
    # HOME; HOME only seeds a fresh session (and headless runs).
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]

    await refresh(HOLD["vs"] or _VS(), force=True)
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
                "**Press \"rank what's in view\"** in the controls above to rank the "
                "divisions on screen."
            ),
            ranking,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
