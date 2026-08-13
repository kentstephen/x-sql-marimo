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
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "matplotlib==3.11.1",
#     "pillow>=11",
# ]
# ///
"""Mapterhorn terrain, extruded: free-fly the whole planet as H3 columns.

ONE DATASET, ONE ENCODING, WORLDWIDE. The Mapterhorn planet.pmtiles pyramid (z0-12,
512 px terrarium WebP, 705 GB, source.coop) folded to H3 per viewport and drawn as an
extruded H3HexagonLayer. Column height is height above the lowest ground in view, so
the mesh sits on the basemap instead of floating at the local sea-level offset; colour
(matplotlib magma over a fixed 0-5,000 m hypsometric ramp) carries the true elevation,
so the map still works from straight above and a colour means the same altitude in
Nepal and in Kansas. The tooltip carries the true metres.

THIS IS THE DEM HALF OF THE PARKED xsql-duckdb-terrain-h3.py, STANDING ALONE. That
notebook joined NLCD onto this terrain and was parked on looks: height was a weak
encoding for a categorical map. Here height IS the measurement, the same argument that
made xsql-canopy-3d.py the survivor of its own pairings. The PMTiles reader and the
terrarium decode are shared by copy with the parked notebook (repo rule: fixes carry
by hand); the camera machinery and the viewport ruler are the canopy notebook's.

THE ELEVATION SCALE IS A NUMBER, NOT A FITTED MULTIPLE, BY CHOICE. It opens at
SCALE_FIXED (20x) at every zoom, so the exaggeration is a stated constant you can
reason about. The one button in the panel switches to AUTO: the scale is refit on
every fold so all the relief in view stands about 1.5 hexagon edges tall (the parked
notebook's fitting rule), which keeps a hillside and a continent equally legible;
pressing it again resets to the fixed 20x. The caption always says what is applied.

OCEAN FOLDS OUT, AND THAT IS MEASURED, NOT ASSUMED. The archive has no bathymetry:
open-ocean tiles are simply ABSENT from about z6 up (decoded as NaN, folds no cells)
and read ~0 m at the coarse zooms where they exist (a mid-Pacific z4 tile measures
99.7% of pixels at |elev| <= 1 m). Pixels within OCEAN_EPS of zero are therefore
dropped before the fold, so the sea is basemap, not a carpet of black hexagons. Real
below-sea land survives: Death Valley at -86 m and the Dead Sea shore at -430 m are
far outside the epsilon and far above DEM_FLOOR.

Data: Mapterhorn planet.pmtiles, us-west-2.opendata.source.coop/mapterhorn/mapterhorn,
      terrarium encoding (R*256 + G + B/256) - 32768, verified on decode (the Rainier
      tile tops at 4,391.6 m against a true 4,392).
Run:  uv run marimo edit xsql-terrain-3d.py --sandbox
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
    import numpy as np
    import obstore
    import pyarrow as pa
    import xarray as xr
    from arro3.core import Table as ArroTable
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from PIL import Image
    from lonboard import Map, H3HexagonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        H3HexagonLayer,
        Image,
        Map,
        MaplibreBasemap,
        S3Store,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        gzip,
        infer_rows_per_chunk,
        io,
        math,
        matplotlib,
        mo,
        np,
        obstore,
        pa,
        struct,
        traitlets,
        udf,
        xr,
    )


@app.cell
def _(anywidget, traitlets):
    class Status(anywidget.AnyWidget):
        """A one-line status readout the camera can write to, and the viewport ruler.

        A widget rather than `mo.md`, because the only way to update marimo output is to
        re-run the cell that produced it, and the cell holding the map is downstream of any
        state the camera could write: re-running it rebuilds the Map and throws the view
        away. A widget trait syncs straight to the browser instead.

        THE RULER, PORTED FROM THE HFP NOTEBOOK BY WAY OF CANOPY. lonboard's view_state
        carries longitude, latitude and zoom but NOT the canvas size, so the kernel
        cannot know how much world the screen shows: VIEW_W/VIEW_H were assumed, and
        going fullscreen made that assumption visibly wrong. This widget is always
        mounted just below the map, and every widget shares the page document, so it
        finds the deck canvas (the largest canvas on the page), measures its CSS size,
        and syncs it up as `view_wh`. Remeasured on window resize, on fullscreenchange
        (fullscreening an ELEMENT resizes no window, so a resize listener alone misses
        it), and via a ResizeObserver on the canvas itself.
        """

        _esm = """
        function render({ model, el }) {
          const line = document.createElement("div");
          line.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const probe = document.createElement("div");
          probe.style.cssText =
            "font:10px ui-monospace,SFMono-Regular,Menlo,monospace;opacity:.4";
          const draw = () => { line.innerHTML = model.get("value"); };
          draw();
          model.on("change:value", draw);
          el.appendChild(line);
          el.appendChild(probe);

          let watched = null;
          const ro = new ResizeObserver(() => kick());
          // marimo puts cell output inside shadow DOM, and document.querySelectorAll
          // does not pierce shadow roots, so the search walks INTO every shadowRoot.
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
                // direction to be wrong in.
                w = window.innerWidth; h = window.innerHeight; tag = "ruler window ";
              }
              if (w > 0 && h > 0) {
                probe.textContent = tag + w + "x" + h;
                // A string, not a number list: the only trait types this repo has
                // PROVEN to cross marimo's anywidget bridge are Unicode and Bool.
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

    return (Status,)


@app.cell
def _(anywidget, traitlets):
    class Controls(anywidget.AnyWidget):
        """The elevation-scale strip under the map: a stepped slider and the auto button.

        A widget rather than mo.ui, for the usual reason: an mo.ui control would make
        the map cell depend on it, and every click would rebuild the Map and reset the
        camera. Traits sync to the kernel, a Python observer assigns straight onto the
        live layer, and nothing re-runs.

        TRAIT TYPES FOLLOW THE PROVEN LIST. The slider's value crosses the bridge as a
        Unicode string and the button as a Bool, because Unicode and Bool are the only
        trait types these notebooks have proven across marimo's anywidget bridge
        (a List(Float) synced from JS never arrived; see the HFP notes).
        """

        _esm = """
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:nowrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.25rem 0 0;user-select:none;overflow:hidden";

          // STOPS, NOT A RANGE, per the parked terrain notebook's lesson: a long
          // linear range over a short track puts several values under every pixel.
          // The stops are coarse at the top where 150 and 160 are the same map and
          // fine at the bottom where each step visibly changes the relief.
          const STOPS = [0, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200];
          const nearest = (v) => {
            let best = 0;
            for (let i = 1; i < STOPS.length; i++) {
              if (Math.abs(STOPS[i] - v) < Math.abs(STOPS[best] - v)) best = i;
            }
            return best;
          };

          const cap = document.createElement("span");
          cap.style.cssText = "opacity:.7;white-space:nowrap";
          const note = () => {
            const n = model.get("note");
            return n ? "  (" + n + ")" : "";
          };
          const val = () => Number(model.get("scale"));
          const draw = () => { cap.textContent = "scale " + val() + note(); };

          const s = document.createElement("input");
          s.type = "range";
          s.min = "0"; s.max = String(STOPS.length - 1); s.step = "1";
          s.value = String(nearest(val()));
          s.style.cssText = "width:9rem;margin:0;cursor:pointer";
          // LIVE, on input: the scale is one trait assignment on a layer that
          // already holds its data, so it can follow the drag.
          s.addEventListener("input", () => {
            model.set("scale", String(STOPS[parseInt(s.value, 10)]));
            model.save_changes();
          });
          model.on("change:scale", () => { s.value = String(nearest(val())); draw(); });
          model.on("change:note", draw);

          // RES +/-: nudge the zoom ladder a step or two either way. Fires on
          // CHANGE, not input, because every stop is a refold (a real read), unlike
          // the scale slider which is one trait assignment. The caption tracks the
          // drag in the meantime.
          const ROFFS = [-2, -1, 0, 1, 2];
          const rcap = document.createElement("span");
          rcap.style.cssText = "opacity:.7;white-space:nowrap";
          const roff = () => Number(model.get("res_off"));
          const rdraw = (v) =>
            { rcap.textContent = "res offset " + (v > 0 ? "+" + v : v); };
          const rs = document.createElement("input");
          rs.type = "range";
          rs.min = "0"; rs.max = String(ROFFS.length - 1); rs.step = "1";
          rs.value = String(ROFFS.indexOf(roff()) < 0 ? 2 : ROFFS.indexOf(roff()));
          rs.style.cssText = "width:5rem;margin:0;cursor:pointer";
          rs.addEventListener("input", () => rdraw(ROFFS[parseInt(rs.value, 10)]));
          rs.addEventListener("change", () => {
            model.set("res_off", String(ROFFS[parseInt(rs.value, 10)]));
            model.save_changes();
          });
          model.on("change:res_off", () => {
            const i = ROFFS.indexOf(roff());
            rs.value = String(i < 0 ? 2 : i);
            rdraw(roff());
          });
          rdraw(roff());

          // ONE BUTTON, TWO JOBS. Off -> on arms the auto fit; on -> off is the
          // reset, and the reset also puts the slider back to the fixed 20 HERE, in
          // the same click, so the kernel never has to guess whether a scale change
          // came from the slider or from the reset.
          const btn = document.createElement("button");
          btn.style.cssText =
            "font:inherit;padding:.15rem .6rem;cursor:pointer;border-radius:4px;" +
            "border:1px solid #8886;background:transparent;color:inherit";
          const label = () => {
            btn.textContent = model.get("auto_scale")
              ? "reset scale to " + model.get("fixed")
              : "auto scale (fit view)";
          };
          btn.addEventListener("click", () => {
            if (model.get("auto_scale")) {
              model.set("auto_scale", false);
              model.set("scale", model.get("fixed"));
            } else {
              model.set("auto_scale", true);
            }
            model.save_changes();
          });
          model.on("change:auto_scale", () => { label(); draw(); });
          label();
          draw();

          // Plain pressed-state toggles: the background says which way the switch
          // sits, the kernel does the rest by trait assignment.
          const toggle = (key, text) => {
            const t = document.createElement("button");
            t.style.cssText =
              "font:inherit;padding:.15rem .6rem;cursor:pointer;border-radius:4px;" +
              "border:1px solid #8886;background:transparent;color:inherit";
            const tDraw = () => {
              t.textContent = text;
              t.style.background = model.get(key) ? "#8884" : "transparent";
            };
            t.addEventListener("click", () => {
              model.set(key, !model.get(key));
              model.save_changes();
            });
            model.on("change:" + key, tDraw);
            tDraw();
            return t;
          };

          box.appendChild(cap);
          box.appendChild(s);
          box.appendChild(btn);
          box.appendChild(rcap);
          box.appendChild(rs);
          // FLAT is a view switch, not a scale of 0: the slider and the auto state
          // keep their values, so leaving flat restores exactly the relief you had.
          box.appendChild(toggle("flat", "flat"));
          // Place names are a separate Carto tile layer OVER the columns (the
          // basemap itself is label-free), so they read through tall terrain
          // strangely; off by default, one click to bring back.
          box.appendChild(toggle("labels", "labels"));
          // THE RAMP FLIP: the kernel recolours the tables in hand and the legend
          // widget follows.
          box.appendChild(toggle("reverse", "reverse cmap"));
          el.appendChild(box);
        }
        export default { render };
        """
        # The applied vertical exaggeration, as a string (see the trait-types note).
        # The kernel parses it; the default here IS the "start at 20 for all" rule.
        scale = traitlets.Unicode("20").tag(sync=True)
        # True while the fit-to-view rule owns the scale. The kernel refits on every
        # fold while this is on; the slider value is ignored until the reset.
        auto_scale = traitlets.Bool(False).tag(sync=True)
        # Kernel -> panel only: what is actually applied right now ("auto 214x" while
        # fitting, blank while fixed, where the slider's own number is the truth).
        note = traitlets.Unicode("").tag(sync=True)
        # The reset target, stated once in the kernel and read by the button label.
        fixed = traitlets.Unicode("20").tag(sync=True)
        # True while the ramp runs dark-high (magma_r). Bool, per the proven list.
        reverse = traitlets.Bool(False).tag(sync=True)
        # True while the extrusion is switched off entirely. The scale and auto
        # state survive underneath it, so flat is reversible in one click.
        flat = traitlets.Bool(False).tag(sync=True)
        # The zoom ladder nudge, -2..+2, as a string per the proven-trait-types
        # rule. 0 is the ladder as computed; each step is one H3 resolution.
        res_off = traitlets.Unicode("0").tag(sync=True)
        # Place-name tiles over the columns. OFF by default: they float above the
        # extrusion and read strangely against tall terrain.
        labels = traitlets.Bool(False).tag(sync=True)

    class HtmlLine(anywidget.AnyWidget):
        """One line of kernel-writable HTML, for the legend.

        The legend has to be a widget for the same reason the status line is: the
        reverse button repaints the map by trait assignment, and an mo.Html legend
        built at layout time would keep showing the ramp the map no longer wears.
        """

        _esm = """
        function render({ model, el }) {
          const div = document.createElement("div");
          const draw = () => { div.innerHTML = model.get("value"); };
          draw();
          model.on("change:value", draw);
          el.appendChild(div);
        }
        export default { render };
        """
        value = traitlets.Unicode("").tag(sync=True)

    return Controls, HtmlLine


@app.cell
def _(math):
    # ------------------------------------------------------------------ the archive
    PM_BUCKET = "us-west-2.opendata.source.coop"
    PM_PATH = "mapterhorn/mapterhorn/planet.pmtiles"
    PM_TILE = 512  # mapterhorn ships 512 px tiles, so this is the source's own grid
    # Terrarium: elevation = (R*256 + G + B/256) - 32768. Verified against known
    # summits on decode: the Rainier tile tops out at 4,391.6 m against a true 4,392.
    DEM_FLOOR = -500.0  # below the Dead Sea; anything under this is void, not terrain
    # OCEAN, MEASURED (probe recorded in the docstring): no bathymetry, sea reads ~0 m
    # where a tile exists at all, and real below-sea land is tens to hundreds of
    # metres negative. Pixels within this band of zero are dropped before the fold.
    OCEAN_EPS = 1.0

    # WHICH MAPTERHORN ZOOM EACH H3 RESOLUTION READS. A COST table, not a resolution
    # match, carried from the parked terrain notebook (rows 5-11 measured there: a
    # res-8 window over the Front Range folds at 57 px/hex against 53.8 predicted).
    # Rows 2-4 extend it worldwide by the same arithmetic: one H3 step is 7x in area,
    # one zoom step is 4x in pixels, so px/hex multiplies by 7 per res step down and
    # divides by 4 per zoom step down. From res 5 -> z4 at ~18 px/hex:
    #   res 4 -> z3   ~31 px/hex   (z2 would be 7.9, patchy coverage)
    #   res 3 -> z1   ~14 px/hex
    #   res 2 -> z0   ~24 px/hex
    # Tile counts stay trivial down there (z3 is 64 tiles for the whole planet), so
    # the generous px/hex costs nothing that matters.
    #
    # RES 12 AND 13 READ THE REGIONAL ARCHIVES, not the planet file, which stops at
    # z12 (and is why res 11 was the ceiling at first). Measured before wiring:
    # `mapterhorn/mapterhorn/6-{x}-{y}.pmtiles`, one per z6 tile with land under it
    # (457 of 4,096; an ocean key is an ABSENT OBJECT, not an empty archive), z13-18
    # (z17 over flat country), same 512 px terrarium WebP, Mont Blanc reads 4,778.8 m
    # at z15. Same 7x/4x arithmetic picks the rungs: res 12 -> z14 (~23 px/hex; z13
    # would be 5.7, patchy) and res 13 -> z15 (~13 px/hex).
    DEM_ZOOM_FOR_RES = {
        2: 0, 3: 1, 4: 3, 5: 4, 6: 5, 7: 7, 8: 8, 9: 9, 10: 11, 11: 12,
        12: 14, 13: 15,
    }
    MAX_RES = 13

    # Average H3 edge length in metres, used ONLY by the auto fit. res 5-11 are the
    # parked notebook's; 2-4 are H3's published averages for the coarser rings.
    EDGE_M = {
        2: 158244.7,
        3: 59810.9,
        4: 22606.4,
        5: 8544.4,
        6: 3229.5,
        7: 1220.6,
        8: 461.4,
        9: 174.4,
        10: 65.9,
        11: 24.9,
        12: 9.4,
        13: 3.6,
    }

    # THE AUTO FIT, carried whole from the parked notebook: all the relief in view
    # stands about 1.5 hexagon edges tall, which is ~25 px on screen at any zoom,
    # because res_for_zoom keeps a hexagon at a roughly constant pixel size. A view
    # with under MIN_RELIEF_M of spread is treated as flat rather than dividing the
    # scale toward infinity.
    TARGET_EDGES = 1.5
    MIN_RELIEF_M = 25.0

    def elev_base_scale(res, relief_m):
        """Metres of drawn height per metre of elevation, fitted to the view."""
        return EDGE_M[res] * TARGET_EDGES / max(relief_m, MIN_RELIEF_M)

    # THE OPENING SCALE, AND THE RESET TARGET. 20x for all zooms: a stated constant
    # rather than a moving fit, so two screenshots at different zooms are comparable.
    # The Controls button trades it for the fit and back.
    SCALE_FIXED = 20.0

    # ------------------------------------------------------------------ the ladder
    # One H3 resolution per 1.4 zoom levels (each H3 step is 2.65x linear and
    # log2(2.65) = 1.4), the constant-cells-on-screen rule every notebook here uses.
    # BASE_RES 7 at ZOOM0 6.2 is the parked notebook's band unchanged. MIN_RES 4 is
    # Stephen's call (res 2-3 read too chunky at world view): ~84k land cells for the
    # whole planet from z3 tiles, the same order as the HFP world floor that measured
    # fine. The ceiling reaches res 12 at zoom ~13.2 and res 13 at ~14.6 (unpitched).
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 7
    MIN_RES = 4

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # PITCH EATS THE RESOLUTION AS WELL AS THE PADDING. A pitched camera sees toward
    # the horizon, so the ground on screen is a trapezoid far larger than the flat
    # box, and folding all of it at the flat view's resolution is what made the
    # parked notebook overread 9x. Past PITCH_COARSE the fold steps ONE H3 coarser:
    # the extra ground is in the far field where a cell is subpixel anyway, and one
    # step is 7x fewer cells, which is what pays for the horizon-ward padding in
    # _pad. Below the threshold a tilt is mostly cosmetic and the flat ladder holds.
    PITCH_COARSE = 35.0

    def res_for_view(z, pitch, off=0):
        """The ladder's answer for this camera: zoom band, plus the panel's res
        offset, minus the pitch step. `off` is the res +/- slider, clamped so it can
        never ask below MIN_RES or above the data's own ceiling."""
        r = max(MIN_RES, min(MAX_RES, res_for_zoom(z) + off))
        return max(MIN_RES, r - 1) if pitch >= PITCH_COARSE else r

    # ------------------------------------------------------------------ view
    # The map's pixel size, as a SEED. The Status widget rulers the real deck canvas
    # and overwrites HOLD["wh"]; these constants only cover the opening fold and
    # headless runs, where no browser ever reports in.
    VIEW_W, VIEW_H = 1400, 620

    # The SYMMETRIC part of the padding only. The canopy notebook's flat PAD 1.5
    # stood in for pitch here at first, and a fullscreen pitched view over Tibet
    # showed exactly what the parked notebook's "PITCH EATS THE PADDING" note
    # predicted: a band of missing cells along the horizon, because the trapezoid a
    # tilted camera sees runs far past the flat box. The pitch handling now lives in
    # _pad explicitly (horizon-ward extension along the bearing) and in res_for_view
    # (one step coarser when tilted), so the symmetric pad drops back to 1.35.
    PAD = 1.35

    SETTLE = 0.2

    # The Alps, pitched, mid-band: enough relief that the opening 20x reads as
    # mountains immediately, and a familiar range to sanity-check against.
    HOME = {
        "longitude": 8.4,
        "latitude": 46.3,
        "zoom": 6.5,
        "pitch": 50,
        "bearing": 0,
    }
    return (
        DEM_FLOOR,
        DEM_ZOOM_FOR_RES,
        HOME,
        OCEAN_EPS,
        PAD,
        PM_BUCKET,
        PM_PATH,
        PM_TILE,
        SCALE_FIXED,
        SETTLE,
        VIEW_H,
        VIEW_W,
        elev_base_scale,
        res_for_view,
    )


@app.cell
def _(matplotlib, np):
    # THE RAMP: magma over a FIXED hypsometric range, not a per-view stretch. A colour
    # means the same altitude everywhere on the planet, so the legend is honest and
    # panning does not repaint ground that did not move; the per-view adaptation lives
    # entirely in the extrusion (relief above the lowest cell in view). sqrt rather
    # than linear because most land sits under 1,000 m and a linear 0-5,000 ramp would
    # spend half its colours on ground almost nobody lives above.
    #
    # Magma is luminance-monotonic (near-black low to pale high) and carries no
    # red-vs-green pair, so it reads correctly for Stephen's protan vision, same
    # argument as the inferno ramps in the fire-risk and HFP notebooks. The near-black
    # bottom recedes into the dark basemap, which those notebooks measured as a
    # feature: lowlands fade back, ranges stand forward.
    ELEV_HI = 5000.0
    # THE ONE PLACE THE COLORMAP IS NAMED. Any matplotlib cmap drops in here ("name"):
    # everything downstream is agnostic, because recolor() repaints the tables in hand
    # from their own elev_m column and the legend rebuilds from ramp_elev itself, and
    # the reverse button composes with any choice since matplotlib registers a "_r"
    # twin for every registered map. Mutable so the panel can flip it without
    # re-running any cell; ramp_elev reads it at call time. Two cautions if swapping:
    # keep to luminance-monotonic maps (viridis, cividis, inferno, mono-hue ramps;
    # never a red-green diverging pair), and remember the dark basemap pairs with a
    # dark-low ramp, so a pale-low map will glow at sea level.
    # "gen" counts repaints: every ramp change bumps it, painted tables record the
    # gen they wore, and stale ones are recoloured LAZILY when they are next served.
    # That is the fix for the reverse button being painfully slow (it used to
    # eagerly rebuild and resend every cached resolution) and for the flip not
    # sticking (a fold in flight across the click used to land wearing the old
    # ramp and stay cached that way).
    RAMP = {"name": "gist_heat", "rev": False, "gen": 0}

    def ramp_elev(v):
        """Mean elevation metres -> uint8 RGB, sqrt-stretched RAMP over 0-5,000 m.

        Below-sea land (Death Valley, the Dead Sea shore) clamps to the bottom of the
        ramp: it is the lowest ground there is, and the tooltip carries the signed
        number. RAMP["rev"] serves the _r twin instead: pale lowlands, dark peaks.
        """
        cmap = matplotlib.colormaps[RAMP["name"] + ("_r" if RAMP["rev"] else "")]
        v = np.asarray(v, dtype="float64")
        t = np.sqrt(np.clip(np.nan_to_num(v) / ELEV_HI, 0.0, 1.0))
        return (cmap(t)[..., :3] * 255).astype(np.uint8)

    ELEV_STOPS = [
        (0.0, "0 m"),
        (250.0, "250"),
        (1000.0, "1,000"),
        (2500.0, "2,500"),
        (5000.0, "5,000+"),
    ]

    def legend_html():
        """The legend, from the same ramp the layer uses, rebuilt on every flip so a
        colour on the map and a colour in the key cannot drift apart."""
        sw = "".join(
            f"<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
            f"<span style='width:14px;height:14px;border-radius:2px;background:rgb("
            f"{','.join(str(int(c)) for c in ramp_elev(np.array([v]))[0])})"
            f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
            for v, lab in ELEV_STOPS
        )
        return (
            "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
            "display:flex;flex-wrap:wrap;align-items:center;padding:.35rem 0\">"
            "<b style='margin-right:.7rem'>elevation</b>"
            f"{sw}</div>"
        )

    return ELEV_STOPS, RAMP, legend_html, ramp_elev


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs
    # every downstream cell, including the one that owns the Map, so the Map would be
    # rebuilt with its opening view_state and the camera would snap home on every pan.
    # A plain dict is invisible to the dataflow graph.
    HOLD = {
        "fold": None,  # box, res -> layer table + note, set by the read cell
        "res": None,  # H3 resolution currently on screen
        "box": None,  # padded degree box the current cells cover
        "relief": 0.0,  # metres, 2nd to 98th percentile of the cells in view
        "cache": {},  # res -> [box, layer table, relief]
        "cam": (0.0, 0.0),  # (pitch, bearing) the current fold was padded for
        "wh": None,  # measured canvas (w, h); the ruler writes it
        "head": "",
        "vs": None,  # the last camera acted on, for the echo check
        "busy": False,
        "pending": None,
        "loop": None,
        "task": None,
    }
    return (HOLD,)


@app.cell
def _(ArroTable, HOLD, coordinates_to_cells, np, pa, ramp_elev):
    def cells_to_layer(tbl):
        """Folded elevation cells -> the arro3 table the extruded layer draws.

        HEIGHT IS RELIEF, NOT ELEVATION, same rule as the parked notebook: `elev` is
        metres above sea level and stays that way for the tooltip and the colour, but
        extruding it would float a Colorado view 2 km off the basemap and multiply
        the offset along with the shape. The columns stand on the 2nd percentile of
        the view, so the mesh sits on the ground and the whole scale goes into the
        part that differs. The 2nd percentile rather than the minimum, so one stray
        cell cannot define the floor for a mountain range.
        """
        tbl = tbl.combine_chunks()
        elev = np.asarray(tbl["elev"], dtype="float64")
        if elev.size:
            floor, ceil = np.percentile(elev, [2.0, 98.0])
        else:
            floor, ceil = 0.0, 0.0
        relief = np.clip(elev - floor, 0.0, None)
        return (
            ArroTable.from_arrow(
                pa.table(
                    {
                        "hex": tbl["hex"],
                        "color": pa.FixedSizeListArray.from_arrays(
                            pa.array(ramp_elev(elev).ravel()), 3
                        ),
                        "height": pa.array(relief),
                        "elev_m": pa.array(np.round(elev, 1)),
                        "pixels": tbl["px"],
                    }
                )
            ),
            float(max(ceil - floor, 0.0)),
        )

    def recolor(tbl):
        """The same table wearing the ramp's CURRENT colours, from its own elev_m.

        This is what makes the reverse button free: the fold and the read are both
        untouched, only the colour column is rebuilt. elev_m is rounded to 0.1 m,
        which moves a colour by well under one ramp step.
        """
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(
                            ramp_elev(np.asarray(tbl["elev_m"], dtype="float64")).ravel()
                        ),
                        3,
                    ),
                    "height": tbl["height"],
                    "elev_m": tbl["elev_m"],
                    "pixels": tbl["pixels"],
                }
            )
        )

    def seed_cells():
        """One flat hexagon at null island so the Map has a valid table at build time.

        This is what lets the Map cell depend on nothing, and therefore never wait
        for the first read. The opening draw replaces it.
        """
        hexes = coordinates_to_cells(np.array([0.0]), np.array([0.0]), 4)
        HOLD["relief"] = 0.0
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": pa.array(hexes),
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([13, 17, 23], dtype=np.uint8)), 3
                    ),
                    "height": pa.array([0.0]),
                    "elev_m": pa.array([0.0]),
                    "pixels": pa.array([0], type=pa.int64()),
                }
            )
        )

    return cells_to_layer, recolor, seed_cells


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
    # THE TERRAIN READER, shared by copy with archive/xsql-duckdb-terrain-h3.py (repo
    # rule: fixes to the directory walk or the varint machinery carry by hand).
    # PMTiles is an XYZ pyramid inside ONE 705 GB object, addressed by ranged GET:
    # header, root directory, leaf directories, then tiles. Opening costs three reads;
    # directories are gzipped varint deltas, parsed once and cached, so the 21.7 MB of
    # leaf directories in the planet file are never read whole.
    _pm_store = S3Store(PM_BUCKET, region="us-west-2", skip_signature=True)

    async def _pm_range(path, a, b):
        """Inclusive byte range [a, b]. obstore's `end` is exclusive."""
        return bytes(
            memoryview(
                await obstore.get_range_async(_pm_store, path, start=a, end=b + 1)
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

        Entries are (tile_id, offset, length, run_length). run_length 0 marks a
        pointer to a LEAF directory rather than to a tile. A zero OFFSET means
        "immediately after the previous entry", so offsets are reconstructed in
        order, not read.
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

        Hilbert rather than row-major so tiles that are near each other on the
        GROUND are near each other in the FILE.
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

        The fallback is not an optimisation: directories are run-length encoded, so
        a tile usually has no entry of its own and is covered by an earlier one.
        Leaf pointers (run_length 0) always match this way too.
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

    async def _pm_open(path):
        """Header plus root directory of ONE archive, or None where the object is
        absent. Absence is a NORMAL answer for the regional set: there is one
        archive per z6 tile with land under it (457 of 4,096, measured), and an
        ocean key is a missing object, not an empty file."""
        try:
            hdr = await _pm_range(path, 0, 126)
        except Exception:
            return None
        assert hdr[:7] == b"PMTiles" and hdr[7] == 3, f"{path}: not a PMTiles v3 archive"
        rd_off, rd_len, _, _, ld_off, _, td_off, _ = struct.unpack("<8Q", hdr[8:72])
        root = _parse_dir(
            gzip.decompress(await _pm_range(path, rd_off, rd_off + rd_len - 1))
        )
        return {
            "path": path,
            "root": root,
            "leaf": {},
            "ld": ld_off,
            "td": td_off,
            "minz": hdr[100],
            "maxz": hdr[101],
        }

    _planet = await _pm_open(PM_PATH)
    assert _planet is not None, f"{PM_PATH} not found"
    _PM_DIR = PM_PATH.rsplit("/", 1)[0]
    # The zoom table is written against these pyramids, so it should fail loudly if
    # they ever disagree rather than silently serve terrain from the wrong scale: a
    # request above an archive's max zoom resolves to SOME entry via the covering
    # run. Rows at or below the planet's max come from the planet file; deeper rows
    # come from the regional archives, whose measured band is z13-18 (z17 in flat
    # country, hence the per-tile maxz guard in _dem_tile).
    assert _planet["minz"] <= min(DEM_ZOOM_FOR_RES.values()), (
        "DEM_ZOOM_FOR_RES below the pyramid"
    )
    assert max(DEM_ZOOM_FOR_RES.values()) <= 18, "DEM_ZOOM_FOR_RES beyond the regionals"

    # (x6, y6) -> the TASK opening that regional archive. Tasks rather than results,
    # the firerisk lesson: a viewport's tiles arrive together, so several would
    # otherwise race the same open; awaiting one task twice is free.
    _regional = {}

    def _regional_task(x6, y6):
        key = (x6, y6)
        if key not in _regional:
            _regional[key] = asyncio.get_running_loop().create_task(
                _pm_open(f"{_PM_DIR}/6-{x6}-{y6}.pmtiles")
            )
        return _regional[key]

    # Decoded-tile LRU on the source's own 512 px grid: a pan re-reads the strip it
    # has not seen and nothing else. float32 after decode, so ~1 MB resident per tile.
    _dem_tiles = {}
    _dem_held = {"bytes": 0}
    DEM_BUDGET = 512 * 1024 * 1024
    _dem_sem = asyncio.Semaphore(32)

    async def _dem_tile(z, x, y):
        """One tile, walked to through its archive's directories, as float32 metres.

        The planet file serves everything up to its own max zoom (12); deeper
        requests route to the regional archive owning the z6 tile above this one.
        No archive, or a request past this archive's own floor, is no data.
        """
        if z <= _planet["maxz"]:
            a = _planet
        else:
            a = await _regional_task(x >> (z - 6), y >> (z - 6))
            if a is None or z > a["maxz"]:
                return None
        tid, ents = _tile_id(z, x, y), a["root"]
        for _ in range(4):  # root + up to three leaf levels
            e = _find(ents, tid)
            if e is None:
                return None
            if e[3] == 0:
                key = (e[1], e[2])
                if key not in a["leaf"]:
                    a["leaf"][key] = _parse_dir(
                        gzip.decompress(
                            await _pm_range(
                                a["path"], a["ld"] + e[1], a["ld"] + e[1] + e[2] - 1
                            )
                        )
                    )
                ents = a["leaf"][key]
                continue
            async with _dem_sem:
                blob = await _pm_range(
                    a["path"], a["td"] + e[1], a["td"] + e[1] + e[2] - 1
                )
            # Terrarium, straight off the RGB. Measured at 4 ms/tile against ~50 ms
            # to fetch one, so the decode is not the cost.
            rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB")).astype(
                np.float32
            )
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
                # A missing tile is ocean or off-archive, not an error. NaN so the
                # fold drops it rather than folding a zero into somebody's mean.
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

    # Web Mercator is closed form in both directions: a tile pixel knows its own
    # lat/lon exactly with no projection library, which is why this whole reader is
    # shorter than any reprojecting fold in the repo.
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
        return (
            (1.0 - math.log(math.tan(la) + 1.0 / math.cos(la)) / math.pi) / 2.0 * (1 << z)
        )

    async def dem_read(res, box_ll):
        """The DEM for a lon/lat box at the zoom `res` deserves.

        Returns (elev, lats, lons, pixels fetched, metres per pixel). `lats` and
        `lons` are the pixel-centre coordinate VECTORS, exactly what xarray-sql wants
        for a dataset's coords, so the H3 UDF downstream reads them with no transform.
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
        # Trim the tile overhang, so the fold does not carry up to a tile of margin
        # on every side into the group-by.
        ci = np.flatnonzero((lons >= w) & (lons <= e))
        ri = np.flatnonzero((lats >= s) & (lats <= n))
        if ci.size == 0 or ri.size == 0:
            return None, None, None, fetched, 0.0
        arr = arr[ri[0] : ri[-1] + 1, ci[0] : ci[-1] + 1]
        mpp = 78271.517 * math.cos(math.radians((s + n) / 2)) / span
        return arr, lats[ri[0] : ri[-1] + 1], lons[ci[0] : ci[-1] + 1], fetched, mpp

    return (dem_read,)


@app.cell
def _(
    BitmapTileLayer,
    CartoBasemap,
    Controls,
    ELEV_STOPS,
    H3HexagonLayer,
    HOLD,
    HOME,
    HtmlLine,
    Map,
    MaplibreBasemap,
    PAD,
    RAMP,
    SCALE_FIXED,
    SETTLE,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    elev_base_scale,
    infer_rows_per_chunk,
    legend_html,
    np,
    ramp_elev,
    recolor,
    res_for_view,
    seed_cells,
):
    # Built exactly once. This cell depends on no state the camera can write, so
    # nothing in the notebook can re-run it and throw the view away. Everything after
    # this happens by trait assignment on the live layer.
    status = Status(value="<b>loading…</b>")
    controls = Controls(fixed=f"{SCALE_FIXED:g}", scale=f"{SCALE_FIXED:g}")
    # A fresh Controls opens unreversed, so the ramp state follows it: a re-run of
    # this cell clears the cache below, and the next fold paints from a RAMP that
    # agrees with the panel.
    RAMP["rev"] = controls.reverse
    legend = HtmlLine(value=legend_html())

    _seed = seed_cells()
    cells = H3HexagonLayer(
        table=_seed,
        get_hexagon=_seed["hex"],
        get_fill_color=_seed["color"],
        get_elevation=_seed["height"],
        extruded=True,
        elevation_scale=SCALE_FIXED,
        stroked=False,
        high_precision=True,
        coverage=1,
        opacity=1.0,
        pickable=True,
    )

    # Place labels drawn OVER the cells, as their own layer, because the basemap is
    # the label-free DarkMatter and anything painted on it would vanish under the
    # opaque columns. The cost of that architecture is labels floating above tall
    # terrain, so the layer answers to the panel's `labels` toggle and starts OFF.
    # pickable=False so a hover meant for a column is never intercepted; @2x with
    # tile_size 512 because the default 256 samples retina tiles at half scale and
    # the type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
        visible=False,
    )

    deck = Map(
        [cells, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state=HOME,
        height=VIEW_H,
        show_tooltip=True,
    )

    # A NEW MAP INHERITS NOTHING ABOUT THE OLD ONE'S SCREEN.
    if HOLD["task"] is not None:
        HOLD["task"].cancel()
    HOLD["task"] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["res"], HOLD["box"], HOLD["vs"] = None, None, None
    HOLD["head"], HOLD["relief"], HOLD["cam"] = "", 0.0, (0.0, 0.0)
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["cache"].clear()

    def apply_scale():
        """Put the panel's scale on the layer. The ONLY place elevation_scale is set.

        Fixed mode is the slider's number, applied as-is at every zoom (the opening
        state: 20). Auto mode refits on every fold so the relief in view stands
        about 1.5 hexagon edges tall whatever the resolution; before the first fold
        there is nothing to fit to and the fixed number stands in. The note trait
        tells the caption which number is actually applied.

        OPEN ISSUE (2026-08-13, noted on request, not fixed): Stephen reports auto
        "just flattens the hexagons". It is doing what it says, and that is the
        problem: the fit targets ~25 px of total relief (TARGET_EDGES 1.5), which
        next to the 20x default reads as flat, e.g. an Alps view at res 7 fits to
        ~0.6x. If auto is to feel useful the target needs retuning (a larger
        TARGET_EDGES, or fitting toward the fixed scale as a ceiling).

        FLAT overrides everything: extrusion off, scale and auto state preserved
        underneath, so the button is a view switch rather than a destructive reset.
        """
        if controls.auto_scale and HOLD["res"] is not None:
            scale = elev_base_scale(HOLD["res"], HOLD["relief"])
            controls.note = f"auto {scale:,.0f}x" if scale >= 1 else f"auto {scale:.2f}x"
        else:
            try:
                scale = float(controls.scale)
            except ValueError:
                scale = SCALE_FIXED
            controls.note = "fitting on next fold" if controls.auto_scale else ""
        cells.elevation_scale = scale
        cells.extruded = (not controls.flat) and scale > 0

    def _ensure_paint(ent):
        """Recolour a cache entry if the ramp moved since it was painted. Lazy on
        purpose: the flip pays for the ONE table being served, and every other
        resolution pays when (if) it is next shown."""
        if ent[4] != RAMP["gen"]:
            ent[1] = recolor(ent[1])
            ent[4] = RAMP["gen"]

    def _on_controls(change):
        if change["name"] == "reverse":
            # The flip repaints, it never refetches, and it resends COLOURS ONLY:
            # the rows on screen are exactly the served entry's rows, so assigning
            # get_fill_color ships one small accessor buffer instead of the whole
            # table (hex ids, heights, tooltip columns) that made this painfully
            # slow. Other cached resolutions are marked stale by the gen bump and
            # recoloured when next served; a view that has not folded yet has
            # nothing to repaint and picks the ramp up on its first fold.
            RAMP["rev"] = controls.reverse
            RAMP["gen"] += 1
            legend.value = legend_html()
            hit = HOLD["cache"].get(HOLD["res"])
            if hit is not None:
                _ensure_paint(hit)
                cells.get_fill_color = hit[1]["color"]
            return
        if change["name"] == "labels":
            labels.visible = controls.labels
            return
        if change["name"] == "res_off":
            # A ladder nudge is a new resolution for the same camera: cache hit if
            # that rung was visited, else a fold.
            vs = HOLD["vs"]
            if vs is None:
                return
            if HOLD["busy"]:
                HOLD["pending"] = vs
            elif not _instant(vs):
                HOLD["task"] = _spawn(refresh(vs))
            return
        # A slider drag while auto owns the scale is a manual override: auto yields.
        # The reset button never lands here as a bare scale change, because its JS
        # flips auto_scale in the same message.
        if change["name"] == "scale" and controls.auto_scale:
            controls.auto_scale = False
        apply_scale()

    controls.observe(
        _on_controls,
        names=["scale", "auto_scale", "reverse", "flat", "res_off", "labels"],
    )
    labels.visible = controls.labels

    def _res_off():
        try:
            return int(controls.res_off)
        except ValueError:
            return 0

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N], clamped to the world.

        Web Mercator flat-view arithmetic; the pitch is absorbed by the widened PAD.
        The size comes from HOLD["wh"], the MEASURED canvas.
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

    def _cam(vs):
        """(pitch, bearing) with the Nones and missing attrs flattened to 0."""
        return (
            float(getattr(vs, "pitch", 0.0) or 0.0),
            float(getattr(vs, "bearing", 0.0) or 0.0),
        )

    def _cam_ok(fold_cam, now_cam):
        """Is a fold padded for `fold_cam` still honest under `now_cam`?

        A shallow camera is covered by any fold (the flat box is inside every padded
        box). A deep tilt needs the fold's horizon extension to have been at least
        as deep, and pointed the same way: the extension runs along the bearing, so
        orbiting far enough swings the far field off the folded box.
        """
        if now_cam[0] <= 20.0:
            return True
        if now_cam[0] > fold_cam[0] + 5.0:
            return False
        d = abs(now_cam[1] - fold_cam[1]) % 360.0
        return min(d, 360.0 - d) <= 15.0

    def _pad(b, vs=None):
        """PITCH EATS THE PADDING, so the pad follows the camera.

        The parked terrain notebook learned this and grew the box symmetrically; the
        measured cost there (a 9x overread at the opening view) is why this version
        grows it WHERE THE CAMERA LOOKS instead. The top of a pitched screen shows
        ground far beyond the flat box along the bearing, so the box is extended
        that way by up to 1.5 view-heights, and widened at half that rate because
        the far field is also wider in world terms than the near field. The cap at
        60 degrees is because the trapezoid diverges as pitch approaches 90; deck's
        default max pitch is 60. res_for_view coarsens the fold in the same regime,
        which is what pays for the extra ground.
        """
        w, s, e, n = b
        sw_, sh_ = e - w, n - s
        w -= sw_ * (PAD - 1) / 2
        e += sw_ * (PAD - 1) / 2
        s -= sh_ * (PAD - 1) / 2
        n += sh_ * (PAD - 1) / 2
        if vs is not None:
            import math as _m

            p, brg = _cam(vs)
            if p > 0:
                ext = 1.5 * _m.sin(_m.radians(min(p, 60.0)))
                dlon, dlat = _m.sin(_m.radians(brg)), _m.cos(_m.radians(brg))
                if dlat >= 0:
                    n += dlat * ext * sh_
                else:
                    s += dlat * ext * sh_
                if dlon >= 0:
                    e += dlon * ext * sw_
                else:
                    w += dlon * ext * sw_
                w -= 0.25 * ext * sw_
                e += 0.25 * ext * sw_
                s -= 0.25 * ext * sh_
                n += 0.25 * ext * sh_
        return (max(-180.0, w), max(-85.0, s), min(180.0, e), min(85.0, n))

    def _covers(box, want):
        return (
            box is not None
            and box[0] <= want[0]
            and box[1] <= want[1]
            and box[2] >= want[2]
            and box[3] >= want[3]
        )

    def _same_view(a, b):
        """The echo check: ignore the event the map emits for a view we set ourselves.

        Pitch and bearing are IN the comparison here, unlike the canopy notebook it
        came from, and that is the other half of the horizon fix: the pad and the
        resolution both follow the camera now, so an orbit or a tilt has to reach
        _instant, where _cam_ok decides whether the fold in hand still covers it.
        Orbit frames that stay inside the tolerances cost one cheap check each.
        """
        return (
            a is not None
            and b is not None
            and round(a.longitude, 6) == round(b.longitude, 6)
            and round(a.latitude, 6) == round(b.latitude, 6)
            and round(a.zoom, 4) == round(b.zoom, 4)
            and _cam(a) == _cam(b)
        )

    def set_status(vs):
        """Redraw the status line from what is already known, plus this zoom.

        The px readout is the kernel's half of the ruler diagnostics; the probe line
        in Status._esm is the browser's half. Both stay on while the fullscreen
        defect recorded in CLAUDE.md is open.
        """
        status.value = (
            f"{HOLD['head']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

    def put_cells(tbl, relief):
        HOLD["relief"] = relief
        cells._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # hold_sync so deck gets one message: hexagons, colours, heights AND the
        # scale land as one update. The scale belongs in the batch because a
        # resolution change under auto moves the hexagon size and the scale that
        # compensates together; two messages would render one frame of new cells at
        # the old exaggeration, a 2.65x jump in apparent relief at a band boundary.
        with cells.hold_sync():
            cells.table = tbl
            cells.get_hexagon = tbl["hex"]
            cells.get_fill_color = tbl["color"]
            cells.get_elevation = tbl["height"]
            cells.visible = True
            apply_scale()

    def _instant(vs):
        """Everything answerable without a read, done synchronously in the comm handler."""
        cam = _cam(vs)
        res = res_for_view(vs.zoom, cam[0], _res_off())
        seen = view_to_bbox(vs)
        if (
            res == HOLD["res"]
            and _covers(HOLD["box"], seen)
            and _cam_ok(HOLD["cam"], cam)
        ):
            set_status(vs)
            return True
        hit = HOLD["cache"].get(res)
        if hit and _covers(hit[0], seen) and _cam_ok(hit[3], cam):
            _ensure_paint(hit)
            put_cells(hit[1], hit[2])
            HOLD["res"], HOLD["box"], HOLD["cam"] = res, hit[0], hit[3]
            HOLD["head"] = f"<b>res {res}</b> · {hit[1].num_rows:,} cells · cached"
            set_status(vs)
            return True
        return False

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or read and refold."""
        if not force and _instant(vs):
            return
        cam = _cam(vs)
        res = res_for_view(vs.zoom, cam[0], _res_off())
        want = _pad(view_to_bbox(vs), vs)
        # THE LAST ANSWER STAYS UP UNTIL THERE IS A NEW ONE: the read happens under
        # the columns already on screen and the swap is one trait update.
        HOLD["head"] = f"<b>reading…</b> res {res}"
        set_status(vs)
        # The gen the fold will paint with, captured BEFORE the await: a reverse
        # click landing mid-read leaves this fold stale, and _ensure_paint catches
        # it below instead of the old ramp going up and staying cached.
        gen = RAMP["gen"]
        layer, relief, note = await HOLD["fold"](want, res)
        if layer is None:
            cells.visible = False
            HOLD["res"], HOLD["box"], HOLD["cam"] = res, want, cam
            HOLD["head"] = f"<b>res {res}</b> · {note}"
            set_status(vs)
            return
        ent = [want, layer, relief, cam, gen]
        HOLD["cache"][res] = ent
        _ensure_paint(ent)
        put_cells(ent[1], relief)
        HOLD["res"], HOLD["box"], HOLD["cam"] = res, want, cam
        HOLD["head"] = f"<b>res {res}</b> · {layer.num_rows:,} cells · {note}"
        set_status(vs)

    async def refresh(vs, force=False):
        """Fold what the camera is looking at, once it has stopped moving.

        SETTLE debounces so a drag reads once at the end; coalescing collapses
        whatever piled up during a read to the NEWEST view. No threads and no
        timers; the debounce is an await on the kernel's own loop.
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
            status.value = HOLD["head"]
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _spawn(coro):
        """Run a coroutine on the kernel's loop, keeping a strong reference to the task."""
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
            HOLD["pending"] = vs
            return
        if _instant(vs):
            return
        HOLD["task"] = _spawn(refresh(vs))

    deck.observe(_on_camera, names="view_state")

    def _on_wh(change):
        """The canvas changed size: fullscreen, a window resize, a layout shift."""
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

    # The legend, built from the same ramp the layer uses, so a colour on the map and
    # a colour in the key cannot drift apart.
    _sw = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
        f"<span style='width:14px;height:14px;border-radius:2px;background:rgb("
        f"{','.join(str(int(c)) for c in ramp_elev(np.array([v]))[0])})"
        f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
        for v, lab in ELEV_STOPS
    )
    legend = (
        "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
        "display:flex;flex-wrap:wrap;align-items:center;padding:.35rem 0\">"
        "<b style='margin-right:.7rem'>elevation</b>"
        f"{_sw}</div>"
    )
    return controls, deck, legend, refresh, status


@app.cell
async def _(
    DEM_FLOOR,
    HOLD,
    HOME,
    OCEAN_EPS,
    XarrayContext,
    asyncio,
    cells_to_layer,
    coordinates_to_cells,
    dem_read,
    pa,
    refresh,
    udf,
    xr,
):
    # THE FOLD. dem_read hands back the window as a 2D array with pixel-centre
    # coordinate vectors, so it registers with xarray-sql directly and the H3 UDF
    # reads lat/lon with no transform: Mapterhorn tile pixels ARE lat/lon by
    # construction, the whole reason the Mercator half of the parked notebook was
    # the short half.
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

    async def fold_terrain(box, res):
        """Terrain layer table for `box` at `res`: (table, relief metres, note).

        The WHERE is doing three jobs, all measured (see the docstring): NaN from
        absent tiles fails every comparison and drops; the terrarium void below any
        real land surface drops at DEM_FLOOR; and the |elev| <= OCEAN_EPS band drops
        the sea, which this archive stores as ~0 where it stores it at all. What
        survives is land, so a coastal cell's mean is a mean over its land pixels.
        """
        elev, lats, lons, fetched, mpp = await dem_read(res, box)
        if elev is None:
            return None, 0.0, "off the grid"
        try:
            ctx.deregister_table("dem")
        except Exception:
            pass
        # One fixed name, one window at a time, so RSS stays flat whatever the zoom.
        # No await between the register and the query: an interleaved camera event
        # must not swap the table mid-fold.
        ctx.from_dataset(
            "dem",
            xr.Dataset({"elev": (("lat", "lon"), elev)}, coords={"lat": lats, "lon": lons}),
            chunks={"lat": 512},
        )
        raw = ctx.sql(f"""
            SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS hex,
                   avg(elev) AS elev,
                   count(*) AS px
            FROM dem
            WHERE elev > {DEM_FLOOR} AND abs(elev) > {OCEAN_EPS}
            GROUP BY 1
        """).to_arrow_table()
        if raw.num_rows == 0:
            return None, 0.0, "open water"
        layer, relief = cells_to_layer(raw)
        note = (
            f"{mpp:,.0f} m px"
            + (" · tiles cached" if fetched == 0 else f" · {fetched / 1e6:.2f}M px fetched")
        )
        return layer, relief, note

    HOLD["fold"] = fold_terrain
    HOLD["loop"] = asyncio.get_running_loop()

    # The opening draw. force=True skips the settle: there is nothing to debounce yet.
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]
        # The opening camera is pitched, so the opening fold must be padded and
        # coarsened for that pitch or the horizon opens with the missing band.
        pitch = HOME["pitch"]
        bearing = HOME["bearing"]

    await refresh(_VS(), force=True)
    return


@app.cell
def _(controls, deck, legend, mo, status):
    mo.vstack(
        [
            deck,
            status,
            mo.Html(legend),
            controls,
            mo.md(
                "Mean elevation per H3 cell, worldwide: Mapterhorn terrain "
                "(planet.pmtiles, z0-12, terrarium WebP), folded per viewport at the "
                "resolution the zoom deserves. Column height is height above the "
                "lowest ground in view times the **scale**; colour is true elevation "
                "on a fixed 0-5,000 m ramp, and the tooltip carries the metres. "
                "Hold Ctrl (or right-drag) to tilt and orbit; a tilted view folds "
                "one H3 step coarser and pads toward the horizon, so the far field "
                "stays filled. **scale** applies a constant vertical exaggeration "
                "at every zoom (opens at 20x); **auto scale** refits on every move "
                "so the relief in view always stands about 25 px tall, and the same "
                "button resets to the fixed 20x. **flat** switches the extrusion "
                "off without touching the scale; **labels** brings the place names "
                "back over the columns; **res** nudges the hexagon resolution up to "
                "two steps either way (res 12-13 read Mapterhorn's regional z13+ "
                "archives); **reverse cmap** flips the ramp. The "
                "sea folds out: this archive has no bathymetry, so ocean is basemap, "
                "and below-sea land (Death Valley, the Dead Sea shore) keeps its "
                "signed metres."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
