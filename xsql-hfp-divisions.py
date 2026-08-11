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
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
#     "matplotlib==3.11.1",
# ]
# ///
"""Global human footprint 2021, folded to H3 and joined onto Overture divisions.

Vizzuality's HFP-100 (`hfp_2021_100m_v1-2_cog.tif`) is one 14 GB COG covering the planet
at 100 m. Its value is the HUMAN FOOTPRINT INDEX, 0-50: the summed pressure of built land,
cropland, pasture, population, night lights, roads, railways and navigable rivers on each
hectare (stored uint16 x1000, nodata 65535). An index that sums intensities is intensive,
so `mean()` is valid at every H3 resolution, and the COG's overview pyramid AVERAGES
(verified: over one window the mean survives an 8x downsample, 15.135 -> 15.150, while the
max collapses 51.2 -> 45.9). No majority vote, no mode, no class fold.

This is the deforestation notebook's machinery pointed at a different raster, and the two
differences are the whole diff. First, HFP-100 is in WORLD MOLLWEIDE (ESRI:54009), not
EPSG:4326: the viewport box is forward-projected to find the pixel window, and each
pixel centre is inverse-projected to feed the H3 fold. Both directions are closed-form
spherical formulas on R=6378137, a dozen lines of numpy, no pyproj. Second, ZERO IS KEPT:
36.7% of land scores exactly 0 and that is the point, untouched ground, where the
deforestation notebook's zero was ocean and was dropped.

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

WHY H3 STILL MATTERS UNDER AN EQUAL-AREA CRS. Mollweide pixels are a true hectare
everywhere, so the latitude bias the deforestation notebook used H3 to remove does not
exist here. What remains is the join: cells are the unit the divisions machinery fills,
ranks and draws, and pixel count still weights WITHIN a cell (a coastal cell is mostly
NaN ocean and should not count as a full one).

THE COG IS SPARSE AND async-geotiff DOES NOT KNOW IT. 65.7% of full-resolution tiles have
offset 0 and length 0, because ocean is not stored, and a read touching one issues a byte
range 0..0 and raises `Invalid range requested, start: 0 end: 0`. Reading on the COG's own
512 px tile grid and consulting `ifd.tile_byte_counts` first turns that from a crash into
a speedup: an empty tile is NaN with no request at all.

COLOUR. The land distribution is heavily bottom-loaded (p50 1.0, p75 5.5, p99 23.4, max
~51, measured globally at L7), so a linear 0-50 ramp lights only the cities. The ramp is
log1p over 0-40: zero sits at the dark end of cividis as the bottom of a continuum, not a
dropped case. See the ramp cell.

DRAW A BOX AND THE JOIN BECOMES A NUMBER. The ▢ button at the lower right of the map ranks
every division inside the box you draw by its mean footprint. It reads one H3
resolution finer than the screen does, sizes that resolution from the BOX rather than the
current zoom, and falls back county -> region -> country, because Overture has counties for
only 171 of 219 countries. This is the one output here that is a figure rather than a colour.

THE CAMERA ANSWERS FROM MEMORY FIRST. `view_state` fires on every frame of a drag, and any
frame that can be served from what is already folded (a pan inside the current box, a zoom
back to a resolution already visited) is answered synchronously in the comm handler. Only a
view that genuinely needs bytes goes through the debounce. See `_instant`.

Data: Vizzuality / Impact Observatory HFP-100 v1.2, CC-BY 4.0, on source.coop. Years
2017-2021 are published; YEAR below picks one and is the seam a year slider would use.
Boundaries: Overture Maps.
Run:  uv run marimo edit xsql-hfp-divisions.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
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
    from datafusion import udf
    from geoarrow.rust.core import from_wkb, multipolygon
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
        gzip,
        infer_rows_per_chunk,
        math,
        matplotlib,
        mo,
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
        for layout changes that are neither.
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
    # 2017-2021 are published, identical in shape. This constant is the year seam.
    YEAR = 2021
    COG = f"vizzuality/hfp-100/hfp_{YEAR}_100m_v1-2_cog.tif"

    # The COG's own tile size, at every level. Reading on this grid is what makes a read
    # shareable between viewports AND what lets the sparse-tile check work, since a tile is
    # the unit that is either present or absent.
    TILE = 512
    FETCH_AT_ONCE = 32  # tiles are only faster than one ranged read if they fly together
    # 512 MB, not the deforestation notebook's 256: the ladder here starts at res 5, so
    # the opening world view alone holds ~253 L5 tiles (~253 MB, sparse ocean included,
    # since a skipped tile still caches as a NaN block), and a budget the size of one
    # window would evict everything else on every world look.
    TILE_BUDGET = 512 * 1024 * 1024

    # WHICH OVERVIEW EACH H3 RESOLUTION READS. The pyramid is 100 m native and doubles ten
    # times: L0 100 m, L1 200, L2 400, L3 800, L4 1.6 km, L5 3.2, L6 6.4, L7 12.8, L8 25.6,
    # L9 51, L10 102. Same geometry as the deforestation COG, and Mollweide pixels are
    # TRUE areas, so the pixels-per-cell arithmetic below is exact rather than equatorial.
    #
    # Chosen so ~20-80 pixels sit under every cell: enough for a mean to mean something,
    # without reading pixels the cell will only average away. Res 9 is the exception and
    # the ceiling: it reads the FULL-RESOLUTION level at ~10 px per cell, and there is
    # nothing finer to read, so the ladder ends there.
    #   res 4 (1,770 km2) / L6 (40.7 km2)  = 43 px
    #   res 5 (  253 km2) / L5 (10.2 km2)  = 25 px
    #   res 6 ( 36.1 km2) / L3 (0.64 km2)  = 56 px
    #   res 7 ( 5.16 km2) / L2 (0.16 km2)  = 32 px
    #   res 8 (0.737 km2) / L1 (0.04 km2)  = 18 px
    #   res 9 (0.105 km2) / L0 (0.01 km2)  = 10 px
    #
    # Reading an overview is only equivalent to reading pixels if the pyramid AVERAGES, and
    # that was verified rather than assumed, same discipline as the deforestation COG: over
    # one window (0-10E, 45-50N) the mean survives an 8x downsample (L3 15.135 -> L6
    # 15.150) while the max collapses (51.2 -> 45.9). That is the signature of average
    # resampling.
    LEVEL_FOR_RES = {4: 6, 5: 5, 6: 3, 7: 2, 8: 1, 9: 0}

    # ------------------------------------------------------------------ the zoom ladder
    # One H3 resolution per 1.4 zoom levels, because each H3 step is 2.65x linear and
    # log2(2.65) = 1.4. That keeps a hexagon a constant size ON SCREEN.
    #
    # BASE_RES 5, ONE STEP FINER THAN THE DEFORESTATION NOTEBOOK ACROSS THE WHOLE LADDER.
    # Smaller hexagons at every zoom, at a measured cost: the opening world view folds L5
    # instead of L6 (~62M pixels against ~16M) and hands lonboard roughly 280k cells
    # instead of 70k. If the opening view ever feels heavy, this constant is the reason.
    #
    # math.floor, NOT int(): int truncates toward zero, so every zoom below ZOOM0 would
    # collapse onto BASE_RES instead of continuing down to MIN_RES.
    ZOOM0, PER_RES, BASE_RES = 4.0, 1.4, 5
    MIN_RES, MAX_RES = 5, 9

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # WHICH DIVISION LEVEL IS DRAWN AT WHICH ZOOM, AND WHY THERE IS NONE AT THE TOP.
    #
    # Below DIV_ZOOM there are no boundaries at all: the hexagons carry the map alone.
    # Under GeoParquet that was forced (a world view of countries meant reading most of
    # 5.5 GB to find 219 rows); under PMTiles a world of countries is 16 tiles at z2 and
    # the constraint is gone. The band is kept as a design choice: at the opening zoom
    # the map is about where pressure IS, and country outlines over it answer a
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
    # The map's pixel size BEFORE the browser reports the real one. The Status widget
    # measures the deck canvas (view_state has no width/height, so this cannot come from
    # the camera) and overwrites HOLD["wh"]; these constants only cover the opening fold
    # and any headless run, where no browser ever reports in. Fullscreen is the case that
    # made the difference visible: a 620 px assumption inside a 1500 px screen folds a
    # band, not the viewport.
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

    # The stroke alpha. Higher than the fill so the boundary still reads when the fill is
    # toggled off; the RGB underneath is the same ramp either way.
    LINE_ALPHA = 205

    # Opens on the Europe-Africa-India band, where the index shows its whole range: the
    # Sahara near 0, the Nile valley, Europe pushing 30+. Zoom 4, not the old 2.4 world
    # view: the ladder is unchanged (res 5 holds at 4 and everything below), this only
    # opens closer in.
    HOME = {"longitude": 20.0, "latitude": 18.0, "zoom": 4.0}
    return (
        COG,
        DIVISION_LABEL,
        FETCH_AT_ONCE,
        FILL_ALPHA,
        HOME,
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
        VIEW_H,
        VIEW_W,
        division_for_zoom,
        res_for_zoom,
    )


@app.cell
def _(matplotlib, np):
    # THE LOG1P RAMP.
    #
    # The land distribution is bottom-loaded: p50 1.0, p75 5.5, p99 23.4, max ~51,
    # measured globally at L7. A linear 0-50 ramp lights only the cities; a pure log
    # cannot hold zero at all, and 36.7% of land IS exactly zero. log1p does both:
    # t = log(1 + v) / log(1 + HI), so zero sits at t = 0 and the low end (0-5, where
    # most of the world lives) gets most of the ramp.
    #
    # ZERO IS IN THE RAMP, NOT A SEPARATE SWATCH, and that is the opposite of the
    # deforestation notebook's choice, deliberately. There zero was dropped ocean; here
    # zero is untouched land, the bottom of a continuum, and a score of 0 against a score
    # of 0.5 is a smaller distinction than "none against some". The dark swatch is kept
    # only for NaN, which after the fold's `v = v` filter never reaches the screen.
    #
    # HI = 40 rather than 50: p99.9 is 35.0, so the top fifth of the nominal scale holds
    # nothing and would waste ramp on it. Values above HI clip into the top colour.
    #
    # cividis rather than viridis: both are colourblind-safe, but cividis is built for it.
    # It is strictly two-hue (blue -> yellow) and monotonic in luminance, and a deuteranope
    # simulation is monotonic too, so the ORDER survives, which is the only thing a
    # sequential ramp has to promise.
    HI = 40.0
    ZERO_RGB = (38, 40, 44)  # NaN only; see above
    _CIVIDIS = matplotlib.colormaps["cividis"]

    def ramp(v):
        """footprint 0-50 -> uint8 RGB, log1p-stretched; NaN takes the dark swatch."""
        v = np.asarray(v, dtype="float64")
        live = np.isfinite(v)
        t = np.zeros(v.shape)
        if live.any():
            t[live] = np.log1p(np.clip(v[live], 0.0, HI)) / np.log1p(HI)
        out = (_CIVIDIS(t)[..., :3] * 255).astype(np.uint8)
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

    # Stops chosen against the measured percentiles: 0 and 1 bracket the median, 5 sits
    # at p75, 15 near p95, 40+ is the clipped top.
    STOPS = [
        (0.0, "0 · wild"),
        (1.0, "1"),
        (3.0, "3"),
        (7.0, "7"),
        (15.0, "15"),
        (25.0, "25"),
        (40.0, "40+"),
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
def _(
    ArroArray,
    ArroTable,
    FILL_ALPHA,
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
        hfp = np.asarray(tbl["hfp"])
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(ramp(hfp).ravel()), 3
                    ),
                    # The index on its own 0-50 scale; the tooltip is the one place the
                    # number is stated outright.
                    "footprint": pa.array(np.round(hfp, 2)),
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
        hfp = np.asarray(tbl["hfp"], dtype="float64")
        geom = ArroArray.from_arrow(
            from_wkb(
                tbl["wkb"].combine_chunks(), to_type=multipolygon("xy", crs="EPSG:4326")
            )
        )
        rest = [
            ArroArray.from_arrow(tbl["name"].combine_chunks()),
            ArroArray.from_arrow(tbl["region"].combine_chunks()),
            ArroArray.from_arrow(tbl["country"].combine_chunks()),
            ArroArray.from_arrow(pa.array(np.round(hfp, 2))),
            ArroArray.from_arrow(tbl["n_cells"].combine_chunks()),
        ]
        names = [
            "geometry",
            "color",
            "line",
            "name",
            "region",
            "country",
            "footprint",
            "cells",
        ]

        # The stroke takes the same ramp as the fill, but from its OWN column: the fill
        # toggle works by swapping to a table whose `color` alpha is zero, and a line fed
        # from that column would vanish with it. One line column, shared by both variants,
        # at the stroke's own alpha.
        line = ArroArray.from_arrow(
            pa.FixedSizeListArray.from_arrays(
                pa.array(ramp_rgba(hfp, LINE_ALPHA).ravel()), 4
            )
        )

        def build(alpha):
            col = ArroArray.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(ramp_rgba(hfp, alpha).ravel()), 4
                )
            )
            return ArroTable.from_arrays([geom, col, line, *rest], names=names)

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
                    "footprint": pa.array([0.0]),
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
                "footprint",
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
        path dropped it: open water carries no footprint score at all, and it drags
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
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["res"], HOLD["box"], HOLD["div"] = None, None, None
    HOLD["divpair"], HOLD["divbox"], HOLD["vs"] = None, None, None
    HOLD["head"], HOLD["tail"] = "", ""
    HOLD["cache"].clear()

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

        THE VIEWPORT DIAGNOSTICS ARE COMMENTED OUT, NOT DELETED. Debugging the ruler
        meant asking two questions from screenshots: what does the browser measure, and
        does the kernel hear it. To re-enable, append
        f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px" to the line below (the kernel's
        half: 1400x620 that never moves means the browser has not reported in) and
        uncomment el.appendChild(probe) in Status._esm (the browser's half, no kernel
        involved). The two disagreeing names the broken leg.
        """
        status.value = (
            f"{HOLD['head']}{HOLD['tail']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

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
            divisions.get_line_color = tbl["line"]
            divisions.visible = controls.show_divisions
            # Picking ignores alpha: the zero-alpha fill still swallows every hover, so
            # with the fill off the layer must stop picking or the cells under it go dead.
            divisions.pickable = bool(controls.division_fill)
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

    status.observe(_on_wh, names="view_wh")

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
        region = tbl["region"].to_pylist()
        n_cells = tbl["n_cells"].to_pylist()
        hfp = np.asarray(tbl["hfp"], dtype="float64")
        order = np.argsort(-hfp)[:RANK_N]
        top = float(hfp[order[0]]) if len(order) else 1.0
        rows = []
        for place, i in enumerate(order, 1):
            i = int(i)
            rgb = ",".join(str(int(c)) for c in ramp(np.array([hfp[i]]))[0])
            bar = max(2.0, 100.0 * hfp[i] / max(top, 1e-12))
            label = names[i] or "(unnamed)"
            where = ", ".join(filter(None, [region[i], country[i] or "??"]))
            rows.append(
                f"<tr>"
                f"<td style='text-align:right;opacity:.5;padding:.12rem .5rem .12rem 0'>{place}</td>"
                f"<td style='padding:.12rem .6rem .12rem 0;white-space:nowrap'>{label}"
                f"<span style='opacity:.45'> · {where}</span></td>"
                f"<td style='text-align:right;padding:.12rem .6rem .12rem 0;"
                f"font-variant-numeric:tabular-nums'>{hfp[i]:.2f}</td>"
                f"<td style='width:180px;padding:.12rem .6rem .12rem 0'>"
                f"<span style='display:block;height:9px;border-radius:2px;"
                f"width:{bar:.1f}%;background:rgb({rgb})'></span></td>"
                f"<td style='text-align:right;opacity:.45;"
                f"font-variant-numeric:tabular-nums'>{n_cells[i]:,} cells</td>"
                f"</tr>"
            )
        head = (
            f"<b>{len(names):,} {DIVISION_LABEL[sub]} in the box</b>, ranked by mean "
            f"human footprint (0-50), measured at H3 res {res}"
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
        "<b style='margin-right:.7rem'>human footprint 2021 (0 = no pressure, 50 = severe)</b>"
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
    # 65.7% of L0 tiles have offset 0 and byte count 0: ocean is simply not stored. async
    # geotiff does not check, so a read touching one asks for byte range 0..0 and raises
    # `TypeError: ValueError: Invalid range requested, start: 0 end: 0`. That error names
    # neither the tile nor the fact that the file is sparse, so it reads like corruption.
    #
    # Consulting the table first turns the crash into a speedup: an absent tile becomes a
    # NaN block with no request.
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
            m = (
                await lv.read(window=Window(col_off=c0, row_off=r0, width=w, height=h))
            ).as_masked()[0]
        # filled(), NOT np.asarray(): asarray on a masked array silently returns the raw
        # data, so every masked 65535 would survive as a real number and a nodata coast
        # would average in at score 65.5. The stored value is the index x1000 in uint16;
        # dividing here means every tile in the cache is already in index units and
        # nothing downstream knows about the encoding.
        arr = np.ma.filled(m.astype(np.float32), np.nan)
        arr[arr == 65535.0] = np.nan  # belt and braces if the mask ever goes missing
        return arr / 1000.0, False

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

    # THE CRS COMES BACK, AND IT IS TEN LINES, NOT THE NLCD MACHINERY. The deforestation
    # notebook's "EPSG:4326 is the whole simplification" does not hold here: HFP-100 is
    # World Mollweide (ESRI:54009), metres on a sphere of R = 6378137 (verified against
    # the header: the raster is 36,080 km wide, which is 4*sqrt(2)*R plus a pixel of pad).
    # But where the NLCD notebooks needed a control grid and a bilinear interpolator for
    # Albers, Mollweide is closed-form BOTH ways: forward needs a Newton solve for the
    # parametric angle (box -> pixel window, a few dozen points), and the inverse is three
    # lines of arcsin (pixel centres -> lat/lng for the fold). No pyproj.
    _SQ2R = math.sqrt(2.0) * 6378137.0

    def _moll_fwd(lon, lat):
        """Degrees -> Mollweide metres. Newton on 2t + sin 2t = pi sin(lat)."""
        phi = np.radians(np.asarray(lat, dtype=np.float64))
        lam = np.radians(np.asarray(lon, dtype=np.float64))
        th = phi.copy()
        for _ in range(12):  # converges in ~5 everywhere below 89 degrees
            th -= (2 * th + np.sin(2 * th) - np.pi * np.sin(phi)) / np.maximum(
                2 + 2 * np.cos(2 * th), 1e-9
            )
        return (2 * _SQ2R / np.pi) * lam * np.cos(th), _SQ2R * np.sin(th)

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

        # Degree box -> pixel window, through the FORWARD projection. Mollweide meridians
        # curve, so a lat/lng box does not map to a rectangle: the widest x is wherever
        # the box comes closest to the equator, not at a corner. Projecting a sampled
        # perimeter and taking the envelope handles that without case analysis.
        w, s, e, n = box
        _t = np.linspace(0.0, 1.0, 33)
        _bx, _by = _moll_fwd(
            np.concatenate([w + (e - w) * _t, np.full(33, e), e + (w - e) * _t, np.full(33, w)]),
            np.concatenate([np.full(33, s), s + (n - s) * _t, np.full(33, n), n + (s - n) * _t]),
        )
        col0 = max(0, int((max(_bx.min(), _L) - _L) / px))
        col1 = min(W, int(math.ceil((min(_bx.max(), _R) - _L) / px)))
        row0 = max(0, int((_T - min(_by.max(), _T)) / py))
        row1 = min(H, int(math.ceil((_T - max(_by.min(), _B)) / py)))
        wpx, hpx = col1 - col0, row1 - row0
        if wpx <= 0 or hpx <= 0:
            return None, 0, 0

        arr, fetched, skipped = await _read_window(li, col0, row0, wpx, hpx)

        # Pixel centres -> lat/lng, through the INVERSE projection, which is closed form.
        # The parametric angle depends only on the row, so lat is one arcsin per ROW and
        # only lon is a full 2D array. Pixels outside the Mollweide ellipse (the dark
        # corners of the projection plane) invert to |lon| > 180; they are unstored
        # nodata anyway, and masking them keeps the fold from ever seeing a fake
        # coordinate.
        _ym = _T - (row0 + np.arange(hpx, dtype=np.float64) + 0.5) * py
        _xm = _L + (col0 + np.arange(wpx, dtype=np.float64) + 0.5) * px
        _th = np.arcsin(np.clip(_ym / _SQ2R, -1.0, 1.0))
        _lat = np.degrees(
            np.arcsin(np.clip((2 * _th + np.sin(2 * _th)) / np.pi, -1.0, 1.0))
        )
        _lon = np.degrees(
            (np.pi * _xm[None, :]) / (2 * _SQ2R * np.maximum(np.cos(_th), 1e-12)[:, None])
        )
        arr[np.abs(_lon) > 180.0] = np.nan
        try:
            ctx.deregister_table("df")
        except Exception:
            pass
        ctx.from_dataset(
            "df",
            xr.Dataset(
                {
                    "v": (("y", "x"), arr),
                    "lat": (("y", "x"), np.ascontiguousarray(np.broadcast_to(_lat[:, None], arr.shape))),
                    "lon": (("y", "x"), _lon),
                },
                coords={"y": np.arange(hpx), "x": np.arange(wpx)},
            ),
            chunks={"y": 512},
        )

        # `v = v` IS THE NaN TEST. Ocean is unstored or nodata and both arrive as NaN,
        # which `v != NULL` would not catch; NaN is the one value that fails equality
        # with itself.
        #
        # px_total is not decoration: it is the weight a cell carries into the zonal join.
        # A coastal cell may be 90% NaN ocean and must not count as a full one.
        #
        # NO `HAVING avg > 0`, AND THAT IS THE OPPOSITE OF THE DEFORESTATION NOTEBOOK.
        # There zero cells were overwhelmingly ocean and were dropped; here ocean is
        # already NaN and a zero cell is UNTOUCHED LAND, which is half of what this map
        # has to say. The Sahara at 0 next to the Nile valley at 30 is the picture.
        return (
            ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS hex,
                       avg(CAST(v AS DOUBLE)) AS hfp,
                       count(*)               AS px_total
                FROM df
                WHERE v = v
                GROUP BY 1
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
    # avg(hfp) EQUAL-WEIGHTS THE CELLS. H3 cells are near-equal-area and so are Mollweide
    # pixels, so unweighted-over-cells and pixel-weighted agree here to within coastal
    # NaN handling; equal weighting is kept because it matches what the choropleth colour
    # already shows. The pixel weighting already happened, inside each cell, in `fold`.
    ZONAL_SQL = """
        SELECT d.id       AS id,
               avg(c.hfp) AS hfp,
               count(*)   AS n_cells
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
        """Divisions in view, each with its mean human footprint.

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
                "Human footprint 2021, the 0-50 pressure index summed from built land, "
                "crops, pasture, population, night lights, roads, rails and rivers "
                "(HFP-100 v1.2, Vizzuality / Impact Observatory, CC-BY 4.0). "
                "Boundaries: Overture Maps. "
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
