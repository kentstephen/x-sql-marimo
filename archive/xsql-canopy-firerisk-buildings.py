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
#     "zarr>=3.1",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
#     "matplotlib==3.11.1",
# ]
# ///
"""Wildfire risk x canopy fuel, folded to H3 and joined onto Overture buildings.

CarbonPlan's Open Climate Risk fire-risk pyramid is a Zarr v3 multiscale store covering
CONUS at 30 m. Its value is RPS, Risk to Potential Structures, the USFS Wildfire Risk to
Communities metric: burn probability times conditional risk to a structure, were one
there. Verified on the tensor store rather than assumed, where the factors are published
separately: corr(rps_2011, bp_2011 * crps_scott) = 1.000000 over 160k pixels.

THE NAME IS THE POINT. RPS is computed for POTENTIAL structures, so it says nothing about
whether anything is actually there. Joining it to Overture's building footprints is what
turns it into a statement about real ones: not "this ground is dangerous" but "this
structure is on dangerous ground".

WHAT EACH ENGINE DOES:

  obstore      streams the Zarr chunks and the Overture PMTiles, unsigned. Nothing cached
               to disk; a viewport reads what it needs and keeps it in memory.
  DataFusion   the fold (pixels -> H3 cells) AND the join (cells -> buildings). The join
               is an integer equi-join on a UBIGINT cell id plus a group-by.
  DuckDB       the polyfill (building footprint -> the cells covering it) and the
               tile-seam dissolve (clipped pieces of one building -> one MultiPolygon).
  lonboard     the render.

ZARR INSTEAD OF A COG, AND IT IS THE EASIER SIDE OF THE TRADE. Three things the
deforestation notebook had to build by hand come free here:

  - The pyramid is DECLARED. The root metadata carries a `multiscales` convention block
    where every one of the 12 levels says `"resampling_method": "mean"`, 2x per level. The
    COG's averaging had to be reverse-engineered by measuring. Checked anyway: one
    1-degree Sierra box reads mean 0.4579 / 0.4587 / 0.4558 at L2 / L4 / L6.
  - Sparseness is in the SPEC. Chunks outside CONUS are absent (a HEAD on
    `0/rps_2011/c/0/0` is a 404), and "absent chunk means fill value" is something every
    Zarr reader implements. The COG's `tile_byte_counts` check and its
    `Invalid range requested, start: 0 end: 0` crash have no equivalent here.
  - The coordinates are IN the store, as 1D float64 degrees. No geotransform arithmetic,
    no affine, and as in the deforestation notebook no reprojection: EPSG:4326 means the
    y/x of the registered dataset feed h3_latlng_to_cell directly.

RES 11 IS THE FLOOR AND THE RASTER SETS IT. A pixel is 0.000308 degrees, which is 770 to
1,070 m2 depending on latitude; a res 11 cell is 2,150 m2, so it holds 2.0-2.8 pixels. Res
12 is 307 m2 and would hold about 0.35, meaning 60-70% of cells would contain no pixel
centre and the layer would hole out. Same arithmetic and same answer as the NLCD notebooks.

WHY THE POLYFILL IS 'overlap' HERE AND 'center' THERE. A division holds thousands of
cells, so 'center' assigns each cell to exactly one division and the map partitions. A
building is the other regime: 150-250 m2 against a 2,150 m2 cell, so it catches no cell
centre and 'center' returns NOTHING for it. 'full' is worse: it wants the CELL inside the
POLYGON. Measured on the small-box case in the deforestation notes, center 0, full 0,
overlap 4. The objection to 'overlap' there does not transfer: counties tile the plane, so
a shared cell double counts, while buildings are disjoint islands and two houses in one
cell genuinely share its value.

THE LIMIT, STATED PLAINLY: a 200 m2 house is 10% of a res 11 cell, so its number is the
cell's number, 90% of which is ground the house does not sit on. That is the 30 m raster,
not the join. Nothing here resolves a single house. Where 'overlap' earns its keep is the
large tail, a 20,000 m2 warehouse covering ~9 cells and getting the mean over its actual
footprint.

BUILDINGS HAVE A MIN ZOOM, AND THE TILES ARE ALWAYS z14. Overture's buildings tileset is
z0-14 and the `building` layer claims minzoom 4, but that is geometry only: Planetiler
strips the ATTRIBUTES off everything below the top zoom. At z13 and under, every feature
carries `@geometry_source` and `@height_source` and nothing else, so `id` is present on
100% of z14 features and 0% of z13 features at all four places measured. `id` is both the
dissolve key and the join key, so a coarser fetch returns thousands of anonymous polygons
and the decode gives no hint that anything is wrong. The camera's zoom therefore decides
only WHETHER buildings are drawn, never which tiles are read.

THE CANOPY SIDE, AND WHY ITS READER IS HAND-ROLLED. Meta/WRI's High Resolution Canopy
Height Maps (dataforgood-fb-data, CC-BY 4.0) are 56,147 zoom-9 Web Mercator quadkey
tiles, each a BigTIFF of 65,536 x 65,536 uint8 metres at ~1.19 m, deflate with the
horizontal-differencing predictor, and NO overview pyramid: every strip is ONE ROW of
65,536 pixels, so any window read pays for full-width strips. That rules out the
free-fly fold the rasters above get from their pyramids, so canopy is read the way the
buildings are: only in the buildings band, per viewport, strided. Each building carries
the mean canopy height over its res 11 cells widened one ring (~80 m), a
defensible-space fuel proxy, and a control switches whether the footprints are painted
by fire risk or by that fuel number. The msk sidecars were measured ALL ZERO over a
tile with real 40 m heights (Paradise), and ~10k tiles have none, so they are ignored:
the only no-data signal is an absent quadkey tile, and it surfaces as NULL, never as 0.

Data:  CarbonPlan Open Climate Risk, CC-BY 4.0, on source.coop. Buildings: Overture Maps.
       Canopy: Meta & WRI High Resolution Canopy Height Maps, CC-BY 4.0, AWS Open Data.
Run:   uv run marimo edit xsql-canopy-firerisk-buildings.py --sandbox
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
    import zlib

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
    import zarr
    from arro3.core import Array as ArroArray, Table as ArroTable
    from datafusion import udf
    from geoarrow.rust.core import from_wkb, multipolygon
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    from xarray_sql import XarrayContext
    from lonboard import Map, H3HexagonLayer, PolygonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroArray,
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        ObjectStore,
        PolygonLayer,
        S3Store,
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
        zarr,
        zlib,
    )


@app.cell
def _(duckdb):
    # ONE JOB, TWO STATEMENTS: footprint -> H3 cells, and clipped pieces -> one footprint.
    # Both are geometry, which is the whole test for what belongs here.
    #
    # The fold and the join stay in DataFusion. The fold because it is a whole-column
    # operation where h3ronpy converts a column at once and DuckDB would call a UDF per
    # row (70 ms against 462 ms on 1.58M rows, measured in xsql-duckdb-nlcd-h3.py). The
    # join because it is an equi-join on an integer with no geometry in sight.
    #
    # The polyfill is the opposite regime: a few thousand tiny polygons, so per-row call
    # overhead is irrelevant and the work is all inside the H3 library, where Uber's C won
    # the dissolve comparison 75 ms against h3ronpy's 2,784 ms.
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

        THE RULER, PORTED FROM THE HFP NOTEBOOK. lonboard's view_state carries longitude,
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
          // The diagnostic line, on while the fullscreen defect stays open in the HFP
          // notebook. Everything still measures and syncs without it; this only decides
          // whether the browser-side reading is SHOWN.
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
                // A string, not a number list: the only trait types this repo has
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
        """Layer switches and the scenario picker, under the map next to the legend.

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
          check("show_buildings", "buildings");

          const wrap = document.createElement("label");
          wrap.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          wrap.appendChild(document.createTextNode("scenario"));
          const sel = document.createElement("select");
          sel.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.05rem .2rem";
          for (const [v, t] of [["2011", "2011 (present)"], ["2047", "2047 (mid-century)"]]) {
            const o = document.createElement("option");
            o.value = v; o.textContent = t;
            sel.appendChild(o);
          }
          sel.value = model.get("year");
          sel.onchange = () => { model.set("year", sel.value); model.save_changes(); };
          model.on("change:year", () => { sel.value = model.get("year"); });
          wrap.appendChild(sel);
          box.appendChild(wrap);

          const pwrap = document.createElement("label");
          pwrap.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          pwrap.appendChild(document.createTextNode("colour"));
          const psel = document.createElement("select");
          psel.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.05rem .2rem";
          for (const [v, t] of [["rps", "fire risk"], ["canopy", "canopy height"]]) {
            const o = document.createElement("option");
            o.value = v; o.textContent = t;
            psel.appendChild(o);
          }
          psel.value = model.get("paint");
          psel.onchange = () => { model.set("paint", psel.value); model.save_changes(); };
          model.on("change:paint", () => { psel.value = model.get("paint"); });
          pwrap.appendChild(psel);
          box.appendChild(pwrap);

          el.appendChild(box);
        }
        export default { render };
        """
        show_cells = traitlets.Bool(True).tag(sync=True)
        show_buildings = traitlets.Bool(True).tag(sync=True)
        year = traitlets.Unicode("2011").tag(sync=True)
        # Which number the building fill encodes. Unicode because it is one of the two
        # trait types proven to cross marimo's anywidget bridge (see the ruler above).
        paint = traitlets.Unicode("rps").tag(sync=True)

    return Controls, Status


@app.cell
def _(math):
    # ------------------------------------------------------------------ the raster
    SOURCE_BUCKET = "us-west-2.opendata.source.coop"
    ZARR_ROOT = (
        "carbonplan/carbonplan-ocr/output/fire-risk/pyramid/production/v1.1.0/pyramid.zarr"
    )

    # Two scenarios, present and mid-century, same grid. The version is in the PATH
    # (production/v1.1.0), which is the pin: the same job OVERTURE_RELEASE does below.
    VARIABLES = {"2011": "rps_2011", "2047": "rps_2047"}

    CHUNK_BUDGET = 256 * 1024 * 1024  # float32 blocks, so ~500 chunks resident at L0
    FETCH_AT_ONCE = 32

    # WHICH LEVEL EACH H3 RESOLUTION READS. The pyramid is 12 levels, 2x each, L0 at
    # 0.000308 degrees: L0 ~34 m, L1 68, L2 137, L3 274, L4 547, L5 1.1 km, L6 2.2,
    # L7 4.4, L8 8.8, L9 17.5, L10 35, L11 70.
    #
    # Chosen so 20-80 pixels sit under every cell, the same rule the deforestation
    # notebook uses: enough for a mean to mean something, without reading pixels the cell
    # will only average away.
    #   res  4 (1,770 km2) / L8 (76.6 km2)   = 23 px
    #   res  5 (  253 km2) / L6 (4.79 km2)   = 53 px
    #   res  6 ( 36.1 km2) / L5 (1.20 km2)   = 30 px
    #   res  7 ( 5.16 km2) / L3 (0.075 km2)  = 69 px
    #   res  8 (0.737 km2) / L2 (0.019 km2)  = 39 px
    #   res  9 (0.105 km2) / L1 (0.0046 km2) = 23 px
    #   res 10 (15,047 m2) / L0 (~900 m2)    = 17 px
    #   res 11 ( 2,150 m2) / L0 (~900 m2)    = 2.4 px   <- the floor, see below
    LEVEL_FOR_RES = {4: 8, 5: 6, 6: 5, 7: 3, 8: 2, 9: 1, 10: 0, 11: 0}

    # ------------------------------------------------------------------ the zoom ladder
    # One H3 resolution per 1.4 zoom levels, because each H3 step is 2.65x linear and
    # log2(2.65) = 1.4. That keeps a hexagon a constant size ON SCREEN.
    #
    # math.floor, NOT int(): int truncates toward zero, so every zoom below ZOOM0 would
    # collapse onto BASE_RES instead of continuing down to MIN_RES.
    ZOOM0, PER_RES, BASE_RES = 3.0, 1.4, 4

    # MAX_RES 11 IS THE DATA'S FLOOR, NOT A RENDER CHOICE. A pixel is 0.000308 degrees,
    # 34 m north-south and 22-31 m east-west across CONUS, so 770-1,070 m2. A res 11 cell
    # is 2,150 m2 and holds 2.0-2.8 of them. Res 12 is 307 m2 and would hold ~0.35, so
    # 60-70% of cells would catch no pixel centre and the layer would hole out. Zooming
    # past z12.8 keeps res 11 and simply draws the cells larger.
    MIN_RES, MAX_RES = 4, 11

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # ------------------------------------------------------------------ the buildings
    # Overture's own PMTiles build. One 179 GB object, anonymous ranged GETs, gzipped MVT,
    # z0-14. Nine times the divisions archive and read exactly the same way.
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/buildings.pmtiles"
    PM_LAYER = "building"

    # BELOW THIS ZOOM THERE ARE NO BUILDINGS. The tileset carries footprints down to z4,
    # but thinned hard: over downtown LA, z13 holds 4,162 features against z14's 1,875 on a
    # QUARTER of the ground, so z14 is about 2x denser. A thinned sample of footprints, each
    # 1-2 px across, is a texture that reads as data and is not.
    BLD_ZOOM = 13.0

    # THE TILE ZOOM IS ALWAYS 14, AND THAT IS NOT A QUALITY PREFERENCE. Planetiler strips
    # the attributes off this layer below its top zoom: at z13 and under, every feature
    # carries `@geometry_source` and `@height_source` and NOTHING else. Measured at four
    # places, `id` is present on 100% of z14 features and 0% of z13 features:
    #
    #   Paradise CA   z13  1,109 feats, id on     0    z14    618 feats, id on   618
    #   Superior CO   z13    531 feats, id on     0    z14    601 feats, id on   601
    #   Downtown LA   z13  4,162 feats, id on     0    z14  1,875 feats, id on 1,875
    #   Malibu CA     z13    389 feats, id on     0    z14    348 feats, id on   348
    #
    # `id` is the dissolve key and the join key, so a z13 fetch yields geometry that cannot
    # be healed across tile seams and cannot be joined to anything. The failure is silent:
    # the decode succeeds, the features are all there, and every one of them is anonymous.
    #
    # `class` is optional even at z14 (Paradise and Malibu have none, Superior and LA do),
    # which is why the class column falls back through `subtype` to empty.
    BLD_TILE_Z = 14

    # A viewport at z13 is ~8 tiles and at z14 ~9. The cap is loose enough never to bite in
    # normal panning and tight enough that a wide view cannot ask for a thousand tiles.
    TILE_CAP = 64

    # Measured tile sizes, for whoever wonders whether this is affordable: gzipped, at z14,
    # 33 KB Malibu, 45 KB Paradise, 64 KB Superior CO, 154 KB Santa Rosa, 177 KB downtown
    # LA. A viewport is a few hundred KB.

    # ------------------------------------------------------------------ the canopy
    # Meta/WRI High Resolution Canopy Height Maps: one BigTIFF per zoom-9 Web Mercator
    # quadkey tile, 65,536 px square at ~1.19 m, uint8 metres, deflate + predictor 2,
    # 1-row strips, NO overview pyramid. Read only in the buildings band, per viewport.
    CHM_BUCKET = "dataforgood-fb-data"
    CHM_BASE = "forests/v1/alsgedi_global_v6_float"
    CHM_Z = 9
    CHM_TILE = 65536

    # EVERY 4TH PIXEL IN BOTH AXES. A res 11 cell holds ~1,500 native pixels and a mean
    # needs a few dozen; stride 4 leaves ~94 per cell and cuts the decompress bill 4x,
    # because the strips are rows, so a row skipped is a strip never inflated. The FETCH
    # is not cut: KB-sized strips coalesce into one contiguous span regardless.
    CAN_STRIDE = 4

    # The strip span one view may fetch. Paradise sits in a 1.0 GB tile (~16 KB per row
    # against a 320 B average over all 56k tiles), and a z13 viewport there spans
    # ~75 MB; anything past this refuses with a note rather than stalling the map.
    CANOPY_BUDGET = 160 * 1024 * 1024

    # ------------------------------------------------------------------ view
    # The map's pixel size, as a SEED. The Status widget rulers the real deck canvas
    # (ported from the HFP notebook) and overwrites HOLD["wh"]; these constants only
    # cover the opening fold and headless runs, where no browser ever reports in.
    VIEW_W, VIEW_H = 1400, 620
    PAD = 1.25

    # SETTLE ONLY GUARDS A READ. Every camera event answerable from memory is handled
    # synchronously in the comm handler, so this delay is never spent on a view the
    # notebook already knows. It exists so a two-second drag issues one read at the end.
    SETTLE = 0.15

    # Buildings are the subject here, not a wash over the hexagons, so they are nearly
    # opaque. The hexagons underneath drop to a lower opacity when buildings are drawn.
    BLD_ALPHA = 235
    BLD_LINE_ALPHA = 255

    # Opens on Paradise, California, at the buildings threshold. The town burned in the
    # 2018 Camp Fire and has been rebuilding since, so the footprints on screen are largely
    # structures placed on ground this layer scores. That is the notebook's argument in one
    # view: RPS is computed for POTENTIAL structures, and here are the real ones.
    HOME = {"longitude": -121.6219, "latitude": 39.7596, "zoom": 13.6}
    return (
        BLD_ALPHA,
        BLD_LINE_ALPHA,
        BLD_TILE_Z,
        BLD_ZOOM,
        CHUNK_BUDGET,
        FETCH_AT_ONCE,
        HOME,
        LEVEL_FOR_RES,
        PAD,
        PM_BUCKET,
        PM_LAYER,
        PM_PATH,
        SETTLE,
        SOURCE_BUCKET,
        TILE_CAP,
        CAN_STRIDE,
        CANOPY_BUDGET,
        CHM_BASE,
        CHM_BUCKET,
        CHM_TILE,
        CHM_Z,
        VARIABLES,
        VIEW_H,
        VIEW_W,
        ZARR_ROOT,
        res_for_zoom,
    )


@app.cell
def _(matplotlib, np):
    # THE LOG RAMP.
    #
    # RPS spans four orders of magnitude and is heavily skewed. Measured over CONUS at L6,
    # 2.19M cells: p25 0.0029, p50 0.023, p75 0.105, p90 0.356, p95 0.643, p99 1.91,
    # p99.9 4.91, max 11.5. A linear ramp would spend 95% of its range on the top 1% of
    # the data and paint the rest flat.
    #
    # LO..HI is p25..p99.9, which is the same rule the deforestation notebook uses: the
    # ramp is spent on the part of the data that varies.
    LO, HI = 3e-3, 5.0

    # ZERO IS A REAL ANSWER HERE, AND IT IS KEPT. 2.8-3.6% of cells CONUS-wide are exactly
    # zero, rising to 10-22% inside a city: water, dense urban core, irrigated cropland.
    # Ground that will not burn.
    #
    # The deforestation notebook DROPS its zero cells, because there zero was overwhelmingly
    # ocean and drawing it covered the map in hexagons that said nothing. The opposite is
    # true here. Zero-risk ground is exactly where buildings are densest, and a building
    # standing on it has a meaningful answer: not at risk. Dropping those cells would punch
    # holes through every city and strand the buildings inside them.
    #
    # THE ZERO SWATCH IS SEPARATED BY LUMINANCE, NOT HUE, AND THAT IS FORCED. A flat neutral
    # grey (78, 80, 84) lands at luminance 0.313 and the low stop of full-range cividis
    # lands at 0.318: measured, not guessed, and "none" and "lowest" came out the same
    # colour. Hue cannot fix it, because the entire point of cividis is that hue carries no
    # information. So the ramp's floor is LIFTED off the bottom of cividis and zero takes
    # the dark end alone.
    #
    # cividis rather than viridis: both are colourblind-safe, but cividis is built for it.
    # Strictly two-hue (blue -> yellow) and monotonic in luminance, and a deuteranope
    # simulation of these stops is monotonic too, which is the only thing a sequential ramp
    # has to promise.
    FLOOR = 0.25
    ZERO_RGB = (38, 40, 44)
    _CIVIDIS = matplotlib.colormaps["cividis"]

    def ramp(v):
        """RPS -> uint8 RGB, with exact zero (and NaN) taking the dark swatch."""
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

        The buildings need four channels and the hexagons need three, and they must agree
        colour for colour: a building and the cells under it are the same number drawn
        twice, so any drift between the two ramps would read as a disagreement in the data.
        """
        rgb = ramp(v)
        out = np.empty(rgb.shape[:-1] + (4,), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = alpha
        return out

    STOPS = [
        (0.0, "none"),
        (3e-3, "0.003"),
        (1e-2, "0.01"),
        (3e-2, "0.03"),
        (1e-1, "0.1"),
        (3e-1, "0.3"),
        (1.0, "1.0"),
        (5.0, "5+"),
    ]

    # THE CANOPY RAMP FOLLOWS THE HFP LESSON, NOT THE RPS ONE: ZERO IS INSIDE THE RAMP.
    # Zero canopy is a real measurement (pavement, bare ground, grass), the bottom of a
    # continuum, exactly like HFP's 36.7% of land scoring 0. The separate grey swatch
    # here means NO DATA: the building's ring fell where no quadkey tile exists. Linear
    # rather than log, because heights are metres on a human scale and 0-25 m is where
    # the mass sits (Paradise box, measured: mean 1.5 m, max 40 m). Both ramps are now
    # cividis, so the legend and tooltip, not hue, say which mode the fill is in; the
    # fire ramp keeps its lifted floor and this one keeps zero inside the ramp.
    CAN_HI = 25.0
    NOCAN_RGB = (96, 94, 102)
    _CAN_CMAP = matplotlib.colormaps["cividis"]

    def ramp_canopy(v):
        """Mean canopy metres -> uint8 RGB; NaN (no CHM tile) takes the grey swatch."""
        v = np.asarray(v, dtype="float64")
        live = np.isfinite(v)
        t = np.zeros(v.shape)
        t[live] = np.clip(v[live] / CAN_HI, 0.0, 1.0)
        out = (_CAN_CMAP(t)[..., :3] * 255).astype(np.uint8)
        out[~live] = NOCAN_RGB
        return out

    def ramp_canopy_rgba(v, alpha):
        rgb = ramp_canopy(v)
        out = np.empty(rgb.shape[:-1] + (4,), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = alpha
        return out

    CAN_STOPS = [
        (0.0, "0 m"),
        (2.0, "2"),
        (5.0, "5"),
        (10.0, "10"),
        (15.0, "15"),
        (20.0, "20"),
        (25.0, "25+"),
    ]
    return CAN_STOPS, NOCAN_RGB, STOPS, ramp, ramp_canopy, ramp_canopy_rgba, ramp_rgba


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs every
    # downstream cell, including the one that owns the Map, so the Map would be rebuilt with
    # its opening view_state and the camera would snap home on every pan. A plain dict is
    # invisible to the dataflow graph.
    HOLD = {
        "fold": None,  # the SQL fold, set by the read cell
        "join": None,  # cells -> buildings, set by the read cell
        "res": None,  # H3 resolution currently on screen
        "box": None,  # padded degree box the current cells cover
        "var": None,  # scenario the cells on screen were folded from
        "bld": False,  # whether buildings are currently drawn
        # The box the BUILDINGS cover, tracked apart from the cells' box because they are
        # fetched second and a camera move can land between the two. Without it, a pan that
        # interrupts a fold leaves the buildings on screen belonging to the previous place
        # while the instant path happily matches and never refetches them.
        "bldbox": None,
        "cache": {},  # (var, res) -> [box, layer table, raw fold]
        "bldtbl": None,  # the table currently on the buildings layer
        "bldraw": None,  # the joined table BEHIND that layer, kept for repainting
        "wh": None,  # measured canvas (w, h); the ruler writes it, view_to_bbox reads it
        # The status line in two halves, so a camera move that reads nothing can still
        # refresh the zoom readout without throwing away what the last read said.
        "head": "",
        "tail": "",
        "vs": None,  # the last camera acted on, for the echo check
        "busy": False,
        "pending": None,
        "loop": None,
        "task": None,
    }
    return (HOLD,)


@app.cell
def _(
    ArroArray,
    ArroTable,
    BLD_ALPHA,
    BLD_LINE_ALPHA,
    coordinates_to_cells,
    np,
    pa,
    ramp,
    ramp_canopy_rgba,
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
        rps = np.asarray(tbl["rps"])
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(ramp(rps).ravel()), 3
                    ),
                    "RPS": pa.array(np.round(rps, 5)),
                    "pixels": tbl["px_total"],
                }
            )
        )

    def buildings_to_layer(tbl, paint, from_wkb, multipolygon):
        """Buildings with joined RPS AND canopy -> the arro3 table the PolygonLayer draws.

        `paint` picks which number the fill encodes; both ride along as tooltip columns,
        so flipping the switch is a repaint of the same joined table, never a refetch.

        Geometry comes straight off the Arrow column via from_wkb: to_pylist() here would
        materialise every polygon as a Python bytes object on the way past.
        """
        tbl = tbl.combine_chunks()
        rps = np.asarray(tbl["rps"], dtype="float64")
        # to_numpy(zero_copy_only=False), not np.asarray: canopy is NULLABLE (a building
        # whose ring touched no CHM tile) and the nulls must come out as NaN, which is
        # what the ramp's grey no-data swatch keys on.
        can = np.asarray(
            tbl["canopy"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype="float64",
        )
        if paint == "canopy":
            fill, line = ramp_canopy_rgba(can, BLD_ALPHA), ramp_canopy_rgba(
                can, BLD_LINE_ALPHA
            )
        else:
            fill, line = ramp_rgba(rps, BLD_ALPHA), ramp_rgba(rps, BLD_LINE_ALPHA)
        geom = ArroArray.from_arrow(
            from_wkb(
                tbl["wkb"].combine_chunks(), to_type=multipolygon("xy", crs="EPSG:4326")
            )
        )
        return ArroTable.from_arrays(
            [
                geom,
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(fill.ravel()), 4)
                ),
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(line.ravel()), 4)
                ),
                ArroArray.from_arrow(tbl["name"].combine_chunks()),
                ArroArray.from_arrow(tbl["class"].combine_chunks()),
                ArroArray.from_arrow(pa.array(np.round(rps, 5))),
                # from_pandas=True turns NaN into NULL, so the tooltip reads "null"
                # where there is no CHM tile rather than a fake 0 m.
                ArroArray.from_arrow(pa.array(np.round(can, 1), from_pandas=True)),
                ArroArray.from_arrow(tbl["n_cells"].combine_chunks()),
            ],
            names=["geometry", "color", "line", "name", "class", "RPS", "canopy_m", "cells"],
        )

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
                    "RPS": pa.array([0.0]),
                    "pixels": pa.array([0], type=pa.int64()),
                }
            )
        )

    def seed_buildings(from_wkb, multipolygon):
        """A one-row polygon table, so the PolygonLayer can be built before any join runs.

        lonboard will not take `table=None`: the layer imports the Arrow C stream in
        __init__ and raises "Expected object with __arrow_c_array__ ..." on anything else.

        A REAL POLYGON, NOT A DEGENERATE ONE. A 1e-6 degree square fails deck's earcut
        tessellator, and that takes down the ENTIRE update pass, because deck initialises
        all layers in one batch and one throw kills the batch. The symptom is a cascade of
        assertions naming perfectly healthy layers. 0.01 degrees at null island: big enough
        to tessellate, far enough from anywhere this map opens to be invisible.
        """
        import struct

        d = 0.01
        ring = [(0.0, 0.0), (d, 0.0), (d, d), (0.0, d), (0.0, 0.0)]
        wkb = struct.pack("<BII", 1, 6, 1)  # little endian, MultiPolygon, 1 polygon
        wkb += struct.pack("<BIII", 1, 3, 1, len(ring))  # Polygon, 1 ring, n points
        for x, y in ring:
            wkb += struct.pack("<dd", x, y)

        geom = from_wkb(
            pa.array([wkb], pa.binary()), to_type=multipolygon("xy", crs="EPSG:4326")
        )
        zero4 = pa.FixedSizeListArray.from_arrays(
            pa.array(np.array([0, 0, 0, 0], dtype=np.uint8)), 4
        )
        return ArroTable.from_arrays(
            [
                ArroArray.from_arrow(geom),
                ArroArray.from_arrow(zero4),
                ArroArray.from_arrow(zero4),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([0.0])),
                ArroArray.from_arrow(pa.array([0.0])),
                ArroArray.from_arrow(pa.array([0], type=pa.int64())),
            ],
            names=["geometry", "color", "line", "name", "class", "RPS", "canopy_m", "cells"],
        )

    return buildings_to_layer, cells_to_layer, seed_buildings, seed_cells


@app.cell
async def _(
    BLD_TILE_Z,
    PM_BUCKET,
    PM_LAYER,
    PM_PATH,
    S3Store,
    TILE_CAP,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    struct,
):
    # BUILDINGS COME OUT OF ONE PMTILES OBJECT, BY RANGED GET. The reader is the one the
    # divisions notebook uses, itself ported from xsql-duckdb-terrain-h3.py, unchanged
    # except for which layer it decodes. Opening costs two reads (127-byte header, root
    # directory), then one leaf directory per region touched, parsed once and cached.
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
        """z/x/y -> PMTiles v3 tile id: Hilbert order within a level, levels stacked."""
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
    assert BLD_TILE_Z <= PM_MAXZ, "BLD_TILE_Z above the pyramid"
    _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
    _leaf = {}

    # ------------------------------------------------------------- the MVT decode
    # Hand-rolled rather than a dependency: an MVT is three nested protobuf messages whose
    # only wire types are varint and length-delimited, and the varint machinery already
    # exists two functions up. Verified ring-exact and property-exact against
    # mapbox-vector-tile on ten real tiles before being trusted with anything.
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
        clockwise-on-screen ring is counterclockwise in plain (x, y) axes and the standard
        shoelace sum comes out positive with no sign flip. Getting the sign backwards
        classifies every ring as a hole and decodes every feature to nothing.
        """
        a = 0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            a += x0 * y1 - x1 * y0
        return a

    def _buildings(tile_buf):
        """The building layer: ([(properties, [(exterior, holes), ...]), ...], extent)."""
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
            if name != PM_LAYER:
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

        Web Mercator is closed form in both directions, so a tile coordinate knows its own
        lon/lat exactly: x is linear in longitude and y is the inverse Gudermannian of
        latitude. The (x, y) row layout of the ring array is already WKB point order, so
        each ring serialises as a length prefix plus the raw float64 bytes.
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

    _tiles = {}  # (z, x, y) -> [piece, ...]; insertion order is LRU order
    TILE_KEEP = 512
    _sem = asyncio.Semaphore(24)

    async def _tile_pieces(z, x, y):
        """One tile, walked to through the directories, decoded.

        A piece is one building's presence in one tile: id, name, class and the clipped
        geometry as WKB. Underground structures are dropped, because a fire-risk score for
        a subway box is meaningless and they would draw as footprints like any other.
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
            if e[3] == 0:  # a pointer to a leaf directory
                lk = (e[1], e[2])
                if lk not in _leaf:
                    async with _sem:
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
        if blob:
            buf = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
            feats, extent = _buildings(buf)
            for props, polys in feats:
                if not polys or props.get("is_underground"):
                    continue
                bid = props.get("id")
                if bid is None:
                    continue
                pieces.append(
                    (
                        bid,
                        props.get("@name") or "",
                        props.get("class") or props.get("subtype") or "",
                        _feature_wkb(polys, z, x, y, extent),
                    )
                )
        _tiles[k] = pieces
        while len(_tiles) > TILE_KEEP:
            _tiles.pop(next(iter(_tiles)))
        return pieces

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

    def _tz_for(_z):
        """Always z14, whatever the camera is doing. See BLD_TILE_Z for why.

        Below z14 the layer has no attributes at all, so a coarser fetch returns anonymous
        geometry that cannot be dissolved or joined. Above z14 there is nothing coarser to
        overzoom from, and the geometry is quantised to ~2.4 m there anyway, far finer than
        a res 11 cell.
        """
        return BLD_TILE_Z

    # CACHED BY COVERAGE, NOT BY EXACT BOX: the box is grown, snapped to the tile grid, and
    # any later box inside that coverage is a lookup rather than a fetch.
    _bld_mem = []  # [[coverage box, tz, table, key], ...], newest last
    BLD_PAD = 1.3
    BLD_KEEP = 6

    def _grow(b, f=BLD_PAD):
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

    # THE TILE-SEAM DISSOLVE. Tile geometry arrives CLIPPED, so one building that straddles
    # a tile edge comes back as two pieces, and drawing them leaves a hairline where the
    # stroke runs along the seam. Worse for the polyfill: two fragments of one footprint
    # would be filled independently and the id would carry a mean over both fragment
    # coverages rather than over the building.
    #
    # ST_Union_Agg per id fixes both, and the tile buffer makes the union clean. Clip edges
    # survive only at the outer boundary of the fetched range, a full BLD_PAD beyond the
    # viewport.
    # cx/cy are the dissolved footprint's centroid, and they exist only to decide which
    # buildings are INSIDE the folded box. The fetch is snapped out to whole z14 tiles and
    # grown by BLD_PAD so that panning is free, which makes it reliably wider than the
    # raster window; without this filter every building in that margin joins to nothing and
    # gets counted as missing data, which it is not.
    DISSOLVE_SQL = """
        WITH u AS (
            SELECT id,
                   any_value(name)  AS name,
                   any_value(class) AS class,
                   ST_Union_Agg(ST_GeomFromWKB(wkb)) AS g
            FROM pieces
            GROUP BY id
        )
        SELECT id, name, class,
               CAST(ST_AsWKB(g) AS BLOB) AS wkb,
               ST_X(ST_Centroid(g))      AS cx,
               ST_Y(ST_Centroid(g))      AS cy
        FROM u
    """

    async def fetch_buildings(bbox, z):
        """Buildings covering `bbox`, dissolved per id. (table, key, note)."""
        tz = _tz_for(z)
        want = _grow(bbox)
        for ent in reversed(_bld_mem):
            if ent[1] == tz and _inside(ent[0], bbox):
                _bld_mem.remove(ent)
                _bld_mem.append(ent)
                return ent[2], ent[3], "cached"

        x0, y0 = _mtile(want[0], want[3], tz)
        x1, y1 = _mtile(want[2], want[1], tz)
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_tiles > TILE_CAP:
            # A wide view asking for buildings. Refused instantly rather than fetched
            # slowly; the zoom band above already makes this nearly unreachable.
            return None, None, "capped"

        got = await asyncio.gather(
            *(
                _tile_pieces(tz, tx, ty)
                for ty in range(y0, y1 + 1)
                for tx in range(x0, x1 + 1)
            )
        )
        pieces = [p for tile in got for p in tile]
        if not pieces:
            return None, None, "empty"

        tbl = pa.table(
            {
                "id": pa.array([p[0] for p in pieces], pa.string()),
                "name": pa.array([p[1] for p in pieces], pa.string()),
                "class": pa.array([p[2] for p in pieces], pa.string()),
                "wkb": pa.array([p[3] for p in pieces], pa.binary()),
            }
        )
        # `pieces` is the name DISSOLVE_SQL selects from: DuckDB's replacement scan
        # resolves it straight out of this frame, Arrow buffers and all.
        pieces = tbl  # noqa: F841 - read by the replacement scan, not by Python
        out = con.sql(DISSOLVE_SQL).to_arrow_table()

        cover = _range_box(tz, x0, y0, x1, y1)
        key = (tz, x0, y0, x1, y1)
        _bld_mem.append([cover, tz, out, key])
        while len(_bld_mem) > BLD_KEEP:
            _bld_mem.pop(0)
        return out, key, f"{n_tiles} tiles"

    # THE POLYFILL, AND WHY THE MODE IS 'overlap'.
    #
    # h3_polygon_wkb_to_cells_experimental takes a POLYGON and rejects MultiPolygon with a
    # message that blames the WKB ("Invalid WKB: expected polygon at 5"), and _feature_wkb
    # always emits MultiPolygon, so ST_Dump is not optional.
    #
    # 'center' is what the divisions notebook uses and it would return NOTHING here. It
    # keeps a cell when the CELL'S CENTRE falls inside the polygon, which is the right rule
    # when the polygon holds thousands of cells and the map has to partition. A building is
    # the opposite regime: 150-250 m2 against a 2,150 m2 cell, so it contains no cell centre
    # at all. 'full' is worse still, wanting the whole CELL inside the POLYGON.
    #
    # The objection to 'overlap' recorded for divisions does not transfer. There it was
    # rejected because counties TILE THE PLANE, so a cell on a shared border counts into
    # both neighbours and a narrow county fills up with ground outside itself. Buildings are
    # disjoint islands: two houses sharing a cell genuinely share its value, and there is no
    # partition to violate.
    POLYFILL_SQL = """
        WITH parts AS (
            SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM blds
        ), filled AS (
            SELECT id, UNNEST(
                       h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'overlap')
                   ) AS hex
            FROM parts
        )
        SELECT DISTINCT id, hex FROM filled
    """

    # Memoised on (cached read, resolution). The polyfill is pure: the same buildings at the
    # same resolution give the same cells forever, so a pan that reuses a cached fetch
    # reuses its cells too.
    _fill_memo = {}
    FILL_KEEP = 12

    def polyfill(blds_table, key, res):
        """(id, hex) for every building, at one resolution."""
        ck = (key, int(res))
        if ck in _fill_memo:
            return _fill_memo[ck]
        blds = blds_table  # noqa: F841 - read by the replacement scan, not by Python
        # to_arrow_table, NOT .arrow(): as of DuckDB 1.5 that hands back a
        # RecordBatchReader, and the failure surfaces much later as
        # "'pyarrow.lib.RecordBatchReader' object has no attribute 'num_rows'".
        out = con.sql(POLYFILL_SQL, params=[int(res)]).to_arrow_table()
        _fill_memo[ck] = out
        while len(_fill_memo) > FILL_KEEP:
            _fill_memo.pop(next(iter(_fill_memo)))
        return out

    # THE FUEL RING. The polyfill's cells are the ground a footprint STANDS ON; the fuel
    # question is what stands AROUND it. h3_grid_disk k=1 widens every covered cell by
    # its six neighbours, ~80 m radius at res 11, which is defensible-space distance.
    # The CANOPY join runs on this mapping; the fire join keeps the tight one, because
    # RPS under the footprint and fuel around it are different questions.
    NEAR_SQL = """
        SELECT DISTINCT id, UNNEST(h3_grid_disk(hex, 1)) AS hex FROM fill
    """
    _near_memo = {}

    def near_cells(mapping, key, res):
        """(id, hex) for every building, widened one ring. Memoised like the polyfill."""
        ck = (key, int(res))
        if ck in _near_memo:
            return _near_memo[ck]
        fill = mapping  # noqa: F841 - read by the replacement scan, not by Python
        out = con.sql(NEAR_SQL).to_arrow_table()
        _near_memo[ck] = out
        while len(_near_memo) > FILL_KEEP:
            _near_memo.pop(next(iter(_near_memo)))
        return out

    return fetch_buildings, near_cells, polyfill


@app.cell
def _(
    CAN_STRIDE,
    CANOPY_BUDGET,
    CHM_BASE,
    CHM_BUCKET,
    CHM_TILE,
    CHM_Z,
    S3Store,
    asyncio,
    math,
    np,
    obstore,
    pa,
    struct,
    zlib,
):
    # THE CANOPY READER. Hand-rolled for the same reason the PMTiles reader is: the
    # object being read is simple and hostile in one specific way no library flag fixes.
    # Each CHM tile is a BigTIFF of 65,536 ONE-ROW strips, deflate with predictor 2, no
    # overview pyramid, so the only defensible read is a strided window at deep zoom.
    #
    # THE MSK SIDECARS ARE IGNORED, AND THAT IS MEASURED, NOT ASSUMED. GDAL mask
    # semantics say 0 = invalid, yet the Paradise tile's mask reads ALL ZERO across rows
    # carrying real 40 m heights, and ~10k of the 56k chm tiles have no sidecar at all.
    # A mask that flags live data invalid is worse than none. No-data is therefore an
    # ABSENT TILE (ocean, unimaged), which surfaces as NULL canopy on a building,
    # never as height 0.
    _chm_store = S3Store(CHM_BUCKET, region="us-east-1", skip_signature=True)
    _csem = asyncio.Semaphore(12)

    async def _chm_range(path, a, b):
        """Inclusive byte range [a, b], like the PMTiles reader's."""
        async with _csem:
            return bytes(
                memoryview(
                    await obstore.get_range_async(_chm_store, path, start=a, end=b + 1)
                )
            )

    def _quadkey(tx, ty):
        """Tile x, y at CHM_Z -> the base-4 quadkey string the objects are named by."""
        return "".join(
            str((((ty >> (CHM_Z - 1 - i)) & 1) << 1) | ((tx >> (CHM_Z - 1 - i)) & 1))
            for i in range(CHM_Z)
        )

    _ifds = {}  # quadkey -> {tag: value}, or None where the tile does not exist

    async def _chm_ifd(qk):
        """The tags that matter from one tile's IFD, fetched once, kept forever.

        BigTIFF: 16-byte header with a u64 IFD offset, then a u64 entry count and
        20-byte entries. Everything this reader needs (the strip offset and byte-count
        ARRAY offsets, compression, predictor) fits or points within the u64 value
        slot, so entries are read as (tag, value) and the types are ignored.
        """
        if qk in _ifds:
            return _ifds[qk]
        path = f"{CHM_BASE}/chm/{qk}.tif"
        try:
            head = await _chm_range(path, 0, 15)
        except Exception:
            # obstore surfaces a missing key as a generic error, and an absent quadkey
            # is a NORMAL answer here (ocean, unimaged): cached so it is asked once.
            _ifds[qk] = None
            return None
        assert head[:2] == b"II" and struct.unpack("<H", head[2:4])[0] == 43, (
            "not a little-endian BigTIFF"
        )
        off = struct.unpack("<Q", head[8:16])[0]
        buf = await _chm_range(path, off, off + 8 + 20 * 24 - 1)
        n = struct.unpack("<Q", buf[:8])[0]
        tags = {}
        for k in range(min(n, 24)):
            tag, _typ, _cnt = struct.unpack_from("<HHQ", buf, 8 + 20 * k)
            (tags[tag],) = struct.unpack_from("<Q", buf, 8 + 20 * k + 12)
        assert tags.get(259) == 8, "not deflate"
        assert tags.get(256) == CHM_TILE and tags.get(278, 1) == 1, "not 1-row strips"
        _ifds[qk] = tags
        return tags

    class _TooBig(Exception):
        pass

    async def _tile_window(qk, lr, lc):
        """Strided local rows x columns of one tile, as float32 metres, or None.

        The fetch is the whole strip span in ~8 MB pieces. The strips wanted are a
        stride-CAN_STRIDE comb over that span, but each is KB-sized, so any range
        coalescing refetches the gaps anyway and one contiguous span is the honest
        request. The stride pays off in the DECODE, where three strips in four are
        never inflated, and a strip that IS inflated is decompressed full width
        whatever the column window wants: that is what 1-row strips cost.
        """
        t = await _chm_ifd(qk)
        if t is None:
            return None, 0.0
        path = f"{CHM_BASE}/chm/{qk}.tif"
        r0, r1 = int(lr[0]), int(lr[-1]) + 1
        oo, cc = await asyncio.gather(
            _chm_range(path, t[273] + 8 * r0, t[273] + 8 * r1 - 1),
            _chm_range(path, t[279] + 8 * r0, t[279] + 8 * r1 - 1),
        )
        oo = np.frombuffer(oo, "<u8")
        cc = np.frombuffer(cc, "<u8")
        lo, hi = int(oo[0]), int(oo[-1] + cc[-1])
        if hi - lo > CANOPY_BUDGET:
            raise _TooBig(f"{(hi - lo) / 1e6:.0f} MB")
        piece = 8 << 20
        parts = await asyncio.gather(
            *(_chm_range(path, a, min(a + piece, hi) - 1) for a in range(lo, hi, piece))
        )
        blob = b"".join(parts)
        out = np.empty((lr.size, lc.size), np.float32)
        pred = t.get(317, 1)
        for j, r in enumerate(lr):
            i = int(r) - r0
            raw = zlib.decompress(blob[int(oo[i]) - lo : int(oo[i] + cc[i]) - lo])
            row = np.frombuffer(raw, np.uint8)
            if pred == 2:
                # Predictor 2 is horizontal differencing; uint8 cumsum wraps mod 256,
                # which is exactly its inverse.
                row = np.cumsum(row, dtype=np.uint8)
            out[j] = row[lc]
        return out, (hi - lo) / 1e6

    async def read_canopy(box):
        """CHM pixels covering `box`, strided, as (lat, lng, v) Arrow rows plus a note.

        Web Mercator both ways is closed form (the HFP notebook needed a Newton solve
        for Mollweide; this is the easy CRS): rows map to latitude through the inverse
        Gudermannian and columns to longitude linearly. Pixel CENTRES, hence the +0.5,
        same as every fold in this repo.
        """
        w, s, e, n = box
        N = CHM_TILE << CHM_Z

        def _gy(la):
            la = min(85.05, max(-85.05, la))
            yf = (
                1.0
                - math.log(
                    math.tan(math.radians(la)) + 1.0 / math.cos(math.radians(la))
                )
                / math.pi
            ) / 2.0
            return min(N - 1, max(0, int(yf * N)))

        gx0 = max(0, int((w + 180.0) / 360.0 * N))
        gx1 = min(N, int(math.ceil((e + 180.0) / 360.0 * N)))
        gy0, gy1 = _gy(n), _gy(s) + 1
        if gx1 <= gx0 or gy1 <= gy0:
            return None, "off-grid"
        rows = np.arange(gy0, gy1, CAN_STRIDE)
        cols = np.arange(gx0, gx1, CAN_STRIDE)
        tys = range(gy0 // CHM_TILE, (gy1 - 1) // CHM_TILE + 1)
        txs = range(gx0 // CHM_TILE, (gx1 - 1) // CHM_TILE + 1)
        if len(tys) * len(txs) > 4:
            # Unreachable from the buildings band (a z13 view is ~13 km against a 78 km
            # tile), so this only guards a future caller with a wider ambition.
            return None, "view too wide"

        vals = np.full((rows.size, cols.size), np.nan, np.float32)
        mb, hit = 0.0, 0
        try:
            for ty in tys:
                rsel = np.nonzero(rows // CHM_TILE == ty)[0]
                if rsel.size == 0:
                    continue
                lr = rows[rsel] - ty * CHM_TILE
                for tx in txs:
                    csel = np.nonzero(cols // CHM_TILE == tx)[0]
                    if csel.size == 0:
                        continue
                    lc = cols[csel] - tx * CHM_TILE
                    data, nb = await _tile_window(_quadkey(tx, ty), lr, lc)
                    if data is None:
                        continue
                    vals[np.ix_(rsel, csel)] = data
                    mb += nb
                    hit += 1
        except _TooBig as exc:
            return None, f"skipped ({exc})"
        if hit == 0:
            return None, "no CHM tile"

        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (rows + 0.5) / N))))
        lng = (cols + 0.5) / N * 360.0 - 180.0
        vv = vals.ravel()
        keep = np.isfinite(vv)
        tbl = pa.table(
            {
                "lat": np.repeat(lat, cols.size)[keep],
                "lng": np.tile(lng, rows.size)[keep],
                "v": vv[keep].astype("float64"),
            }
        )
        return tbl, f"{mb:.0f} MB"

    return (read_canopy,)


@app.cell
def _(
    BLD_ZOOM,
    BitmapTileLayer,
    CAN_STOPS,
    CartoBasemap,
    Controls,
    H3HexagonLayer,
    HOLD,
    HOME,
    Map,
    MaplibreBasemap,
    PAD,
    PolygonLayer,
    SETTLE,
    STOPS,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    buildings_to_layer,
    cells_to_layer,
    from_wkb,
    infer_rows_per_chunk,
    multipolygon,
    np,
    ramp,
    ramp_canopy,
    res_for_zoom,
    seed_buildings,
    seed_cells,
):
    # Built exactly once. This cell depends on no control and on no state the camera can
    # write, so nothing in the notebook can re-run it and throw the view away. Everything
    # after this happens by trait assignment, which lonboard treats as independent of
    # `view_state`.
    status = Status(value="<b>loading…</b>")
    controls = Controls()

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

    # THE BUILDINGS. Flat, and the colour is the only variable. Extrusion was considered and
    # dropped: `height` is present on 98% of the footprints in these tiles, but the parked
    # terrain notebook's lesson holds, an extruded layer buries what is underneath it, and
    # here that is the hexagons the buildings are being compared against.
    #
    # line_width_units="pixels" explicitly. deck's default is METRES with get_line_width
    # defaulting to 1, so the visible width is max(1 metre in pixels, line_width_min_pixels)
    # and can never go below the floor: two numbers in different units fighting over one
    # line. At these zooms a 1 m stroke would swallow the fill of a small house.
    _bseed = seed_buildings(from_wkb, multipolygon)
    buildings = PolygonLayer(
        table=_bseed,
        get_fill_color=_bseed["color"],
        filled=True,
        stroked=True,
        line_width_units="pixels",
        get_line_width=0.5,
        line_width_min_pixels=0,
        line_width_max_pixels=1.0,
        get_line_color=_bseed["line"],
        opacity=1.0,
        pickable=True,
        visible=False,
    )

    # Place labels drawn OVER the cells. The basemap paints under every deck layer, so names
    # on it would sit beneath an opaque hexagon and be lost. pickable=False so a hover meant
    # for a building is never intercepted; @2x with tile_size 512 because the default 256
    # samples retina tiles at half scale and the type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
    )

    deck = Map(
        [cells, buildings, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state=HOME,
        height=VIEW_H,
        show_tooltip=True,
    )

    # A NEW MAP INHERITS NOTHING ABOUT THE OLD ONE'S SCREEN. HOLD lives in a cell that
    # cannot re-run, which is what lets the camera survive; the cost is that a re-run of
    # THIS cell builds fresh layers while HOLD still describes the map that just went away.
    if HOLD["task"] is not None:
        HOLD["task"].cancel()
    HOLD["task"] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["res"], HOLD["box"], HOLD["var"] = None, None, None
    HOLD["bld"], HOLD["bldbox"], HOLD["bldtbl"], HOLD["vs"] = False, None, None, None
    HOLD["bldraw"] = None
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["head"], HOLD["tail"] = "", ""
    HOLD["cache"].clear()

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N], clamped to the world.

        Web Mercator: the horizontal span is a straight function of zoom, and the vertical
        span is that scaled by the aspect ratio and by cos(latitude), because a degree of
        longitude narrows toward the poles.

        The size comes from HOLD["wh"], the MEASURED canvas, not the VIEW_W/VIEW_H
        guess: view_state has no width or height in it, so the Status widget rulers the
        deck canvas in the browser and syncs it up. Fullscreen is where the guess
        failed visibly in the HFP notebook, and this fork ports its fix.
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

        The px readout is the kernel's half of the ruler diagnostics (the browser's
        half is the probe line in Status._esm); both stay ON while the fullscreen
        defect logged in the HFP notes remains open.
        """
        status.value = (
            f"{HOLD['head']}{HOLD['tail']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

    def put_cells(tbl):
        cells._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # hold_sync so deck gets one message. Without it the new hexagons are drawn against
        # the old colour buffer for a frame.
        with cells.hold_sync():
            cells.table = tbl
            cells.get_hexagon = tbl["hex"]
            cells.get_fill_color = tbl["color"]
            cells.visible = controls.show_cells

    def put_buildings(tbl):
        buildings._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with buildings.hold_sync():
            buildings.table = tbl
            buildings.get_fill_color = tbl["color"]
            buildings.get_line_color = tbl["line"]
            buildings.visible = controls.show_buildings
        HOLD["bldtbl"] = tbl
        # The hexagons step back when buildings are on top of them. Not hidden: the
        # comparison between a footprint and the ground under it is the whole point, and
        # it only works if both are visible.
        cells.opacity = 0.35

    def hide_buildings():
        buildings.visible = False
        HOLD["bldtbl"], HOLD["bld"], HOLD["bldbox"] = None, False, None
        cells.opacity = 0.7

    def _on_controls(change):
        name = change["name"]
        if name == "show_cells":
            cells.visible = bool(change["new"])
        elif name == "show_buildings":
            buildings.visible = bool(change["new"]) and HOLD["bldtbl"] is not None
        elif name == "year":
            # A different scenario is a different raster, so nothing already folded
            # applies. Everything is dropped and the current view is redrawn from scratch.
            HOLD["cache"].clear()
            HOLD["res"], HOLD["box"], HOLD["var"] = None, None, None
            HOLD["bldbox"] = None
            if HOLD["vs"] is not None:
                HOLD["task"] = _spawn(refresh(HOLD["vs"], force=True))
        elif name == "paint":
            # A repaint of the joined table already in hand: no fetch, no fold, no join.
            if HOLD["bldraw"] is not None and HOLD["bldtbl"] is not None:
                put_buildings(
                    buildings_to_layer(
                        HOLD["bldraw"], controls.paint, from_wkb, multipolygon
                    )
                )

    controls.observe(
        _on_controls, names=["show_cells", "show_buildings", "year", "paint"]
    )

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

        THIS IS THE ZOOM AND PAN FEEL. `view_state` fires on every frame, and handing every
        one of those frames to a task that sleeps SETTLE seconds BEFORE it looks at the
        cache costs a quarter second of nothing for a pan inside the box already on screen.
        Answering those here means the map keeps up with the mouse and the debounce is only
        ever spent waiting on bytes that are genuinely missing.
        """
        var = controls.year
        res = res_for_zoom(vs.zoom)
        want_bld = vs.zoom >= BLD_ZOOM
        seen = view_to_bbox(vs)
        bld_ok = (not want_bld and not HOLD["bld"]) or (
            want_bld and HOLD["bld"] and _covers(HOLD["bldbox"], seen)
        )
        if (
            var == HOLD["var"]
            and res == HOLD["res"]
            and bld_ok
            and _covers(HOLD["box"], seen)
        ):
            set_status(vs)
            return True
        # A resolution folded before that still covers the screen. This is the whole
        # zoom-out case: coming back up to a level already visited lands on the frame it is
        # asked for rather than a second later.
        hit = HOLD["cache"].get((var, res))
        if hit and bld_ok and _covers(hit[0], seen):
            put_cells(hit[1])
            HOLD["res"], HOLD["box"], HOLD["var"] = res, hit[0], var
            HOLD["head"] = f"<b>res {res}</b> · {hit[1].num_rows:,} cells · cached"
            set_status(vs)
            return True
        return False

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or read and refold."""
        if not force and _instant(vs):
            return

        var = controls.year
        res = res_for_zoom(vs.zoom)
        want = _pad(view_to_bbox(vs))
        want_bld = vs.zoom >= BLD_ZOOM

        # THE LAST ANSWER STAYS UP UNTIL THERE IS A NEW ONE. Nothing is cleared here: the
        # read happens under the cells already on screen, and the swap is one trait update
        # when the new fold is complete. A stale-but-plausible map reads as the map; an
        # empty one reads as broken.
        HOLD["head"] = f"<b>reading…</b> res {res}"
        set_status(vs)
        raw, blocks, empty = await HOLD["fold"](res, want, var)
        if raw is None or raw.num_rows == 0:
            HOLD["res"], HOLD["box"], HOLD["var"] = res, want, var
            HOLD["head"] = f"<b>res {res}</b> · no data here (CONUS only)"
            HOLD["tail"] = ""
            hide_buildings()
            set_status(vs)
            return

        tbl = cells_to_layer(raw)
        HOLD["cache"][(var, res)] = [want, tbl, raw]
        put_cells(tbl)
        HOLD["res"], HOLD["box"], HOLD["var"] = res, want, var

        HOLD["head"] = (
            f"<b>res {res}</b> · {raw.num_rows:,} cells · "
            f"{'chunks cached' if blocks == 0 else f'{blocks} chunks'}"
            f"{f' · {empty} empty' if empty else ''}"
        )
        set_status(vs)

        # THE BUILDINGS, AFTER THE CELLS. The fold is the read the user is waiting on and
        # the join depends on it, so the buildings go out second rather than holding the
        # whole frame.
        if HOLD["pending"] is not None:
            return  # the camera has already moved; this view is gone
        if not want_bld:
            hide_buildings()
            HOLD["tail"] = f" · zoom to {BLD_ZOOM:g} for buildings"
            set_status(vs)
            return

        joined = await HOLD["join"](want, vs.zoom, res, raw)
        if joined is None:
            hide_buildings()
            HOLD["tail"] = " · no buildings here"
            set_status(vs)
            return
        bt, n_bld, n_missed, note = joined
        put_buildings(bt)
        HOLD["bld"], HOLD["bldbox"] = True, want
        HOLD["tail"] = f" · {n_bld:,} buildings ({note})" + (
            f" · <b style='color:#E69F00'>{n_missed} off-grid</b>" if n_missed else ""
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

    # The legend, built from the same `ramp` the layers use, so a colour on the map and a
    # colour in the key cannot drift apart. The buildings use the same ramp at a higher
    # alpha, so one key serves both.
    _sw = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
        f"<span style='width:14px;height:14px;border-radius:2px;background:rgb("
        f"{','.join(str(int(c)) for c in ramp(np.array([v]))[0])})"
        f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
        for v, lab in STOPS
    )
    _cw = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
        f"<span style='width:14px;height:14px;border-radius:2px;background:rgb("
        f"{','.join(str(int(c)) for c in ramp_canopy(np.array([v]))[0])})"
        f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
        for v, lab in CAN_STOPS
    ) + (
        "<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
        "<span style='width:14px;height:14px;border-radius:2px;background:rgb("
        f"{','.join(str(int(c)) for c in ramp_canopy(np.array([float('nan')]))[0])})"
        ";outline:1px solid rgba(255,255,255,.18)'></span>no CHM tile</span>"
    )
    legend = (
        "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
        "display:flex;flex-wrap:wrap;align-items:center;padding:.35rem 0\">"
        "<b style='margin-right:.7rem'>risk to potential structures (RPS)</b>"
        f"{_sw}</div>"
        "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
        "display:flex;flex-wrap:wrap;align-items:center;padding:.1rem 0 .35rem\">"
        "<b style='margin-right:.7rem'>canopy height, mean within ~80 m</b>"
        f"{_cw}</div>"
    )
    return controls, deck, legend, refresh, status


@app.cell
async def _(
    CHUNK_BUDGET,
    FETCH_AT_ONCE,
    HOLD,
    HOME,
    LEVEL_FOR_RES,
    ObjectStore,
    S3Store,
    SOURCE_BUCKET,
    VARIABLES,
    XarrayContext,
    ZARR_ROOT,
    asyncio,
    buildings_to_layer,
    controls,
    coordinates_to_cells,
    fetch_buildings,
    from_wkb,
    multipolygon,
    near_cells,
    np,
    pa,
    polyfill,
    read_canopy,
    refresh,
    udf,
    xr,
    zarr,
):
    # obstore is the transport under Zarr too: zarr 3 takes an ObjectStore wrapper directly,
    # so the same unsigned S3 client serves the raster and the tiles.
    _zstore = ObjectStore(
        S3Store(SOURCE_BUCKET, region="us-west-2", skip_signature=True), read_only=True
    )

    _arrays = {}

    def _arr(lvl, var):
        k = (lvl, var)
        if k not in _arrays:
            _arrays[k] = zarr.open_array(
                _zstore, path=f"{ZARR_ROOT}/{lvl}/{var}", mode="r"
            )
        return _arrays[k]

    _coord_memo = {}

    async def _coords(lvl):
        """The level's 1D latitude and longitude, in degrees, read once and kept.

        THE COORDINATES ARE IN THE STORE. The COG path had to derive lon/lat from a
        geotransform; here they are published arrays, so there is no affine to get wrong and
        no assumption that the grid is exactly regular. Both ascend. At L0 the pair is
        2.5 MB, which is a one-time cost at the finest level only.
        """
        if lvl not in _coord_memo:
            lat = zarr.open_array(_zstore, path=f"{ZARR_ROOT}/{lvl}/latitude", mode="r")
            lon = zarr.open_array(_zstore, path=f"{ZARR_ROOT}/{lvl}/longitude", mode="r")
            a, b = await asyncio.gather(
                asyncio.to_thread(lambda: np.asarray(lat[:], dtype="float64")),
                asyncio.to_thread(lambda: np.asarray(lon[:], dtype="float64")),
            )
            _coord_memo[lvl] = (a, b)
        return _coord_memo[lvl]

    _blocks = {}  # (lvl, var, cy, cx) -> float32 array; insertion order is LRU order
    # A dict, not an int: a marimo cell body is compiled at MODULE scope, so `nonlocal` in a
    # nested def is a SyntaxError there and a bare rebind would shadow instead of accumulate.
    _held = {"bytes": 0}
    _rsem = asyncio.Semaphore(FETCH_AT_ONCE)

    async def _block(lvl, var, cy, cx):
        """One chunk of one level, on the store's OWN chunk grid.

        Reading on the chunk grid is what makes a read SHAREABLE. One ranged read per exact
        viewport can never be reused, because no two camera positions produce the same
        rectangle; a pan on this grid touches the chunks already held plus a strip of new
        ones, and a zoom back to a level visited before is free.

        Absent chunks need no special case. Ocean and everything outside CONUS is simply not
        stored, and "a missing chunk reads as fill_value" is in the Zarr spec, so this comes
        back as NaN. The COG path had to consult `tile_byte_counts` by hand to avoid a byte
        range of 0..0, which raised an error naming neither the tile nor the sparseness.
        """
        a = _arr(lvl, var)
        ch, cw = a.chunks
        r0, c0 = cy * ch, cx * cw
        r1, c1 = min(a.shape[0], r0 + ch), min(a.shape[1], c0 + cw)
        async with _rsem:
            blk = await asyncio.to_thread(
                lambda: np.asarray(a[r0:r1, c0:c1], dtype=np.float32)
            )
        return blk

    async def _read_window(lvl, var, row0, col0, hpx, wpx):
        """The window, assembled from cached chunks plus whatever is missing."""
        a = _arr(lvl, var)
        ch, cw = a.chunks
        cy0, cy1 = row0 // ch, (row0 + hpx - 1) // ch
        cx0, cx1 = col0 // cw, (col0 + wpx - 1) // cw
        want = [
            (lvl, var, cy, cx)
            for cy in range(cy0, cy1 + 1)
            for cx in range(cx0, cx1 + 1)
        ]
        need = [k for k in want if k not in _blocks]

        if need:
            got = await asyncio.gather(*(_block(*k) for k in need))
            for k, blk in zip(need, got):
                _blocks[k] = blk
                _held["bytes"] += blk.nbytes
            # Oldest first, and never evict a chunk this window is about to read.
            while _held["bytes"] > CHUNK_BUDGET and len(_blocks) > len(want):
                for k in list(_blocks):
                    if k not in want:
                        _held["bytes"] -= _blocks.pop(k).nbytes
                        break
                else:
                    break

        out = np.full((hpx, wpx), np.nan, dtype=np.float32)
        empty = 0
        for k in want:
            _, _, cy, cx = k
            blk = _blocks[k]
            if not np.isfinite(blk).any():
                empty += 1
            sr, sc = cy * ch, cx * cw
            r0, c0 = max(row0, sr), max(col0, sc)
            r1 = min(row0 + hpx, sr + blk.shape[0])
            c1 = min(col0 + wpx, sc + blk.shape[1])
            if r1 <= r0 or c1 <= c0:
                continue
            out[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = blk[
                r0 - sr : r1 - sr, c0 - sc : c1 - sc
            ]
        for k in want:  # touch: anything used goes to the young end of the LRU
            _blocks[k] = _blocks.pop(k)
        return out, len(need), empty

    # EPSG:4326 IS THE WHOLE SIMPLIFICATION. The NLCD notebooks carry an Albers control grid,
    # a bilinear interpolator and to_lat/to_lon UDFs purely to get degrees out of projected
    # metres. Here the pixel grid IS degrees, so the y/x coordinates of the registered
    # dataset feed h3_latlng_to_cell directly and all of that machinery is gone.
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

    async def fold(res, box, year):
        """Read the window for `box` at the level `res` deserves, then fold it to H3."""
        var = VARIABLES[year]
        lvl = LEVEL_FOR_RES[res]
        lat, lon = await _coords(lvl)

        w, s, e, n = box
        row0 = int(np.clip(np.searchsorted(lat, s, "left"), 0, lat.size))
        row1 = int(np.clip(np.searchsorted(lat, n, "right"), 0, lat.size))
        col0 = int(np.clip(np.searchsorted(lon, w, "left"), 0, lon.size))
        col1 = int(np.clip(np.searchsorted(lon, e, "right"), 0, lon.size))
        hpx, wpx = row1 - row0, col1 - col0
        if hpx <= 0 or wpx <= 0:
            return None, 0, 0  # the viewport is off CONUS entirely

        arr, blocks, empty = await _read_window(lvl, var, row0, col0, hpx, wpx)

        try:
            ctx.deregister_table("df")
        except Exception:
            pass
        ctx.from_dataset(
            "df",
            xr.Dataset(
                {"v": (("y", "x"), arr)},
                coords={"y": lat[row0:row1], "x": lon[col0:col1]},
            ),
            chunks={"y": 512},
        )

        # `v = v` IS THE NaN TEST. There is no nodata sentinel: everything outside CONUS is
        # NaN, which is about half of the store's bounding rectangle, and `v != NULL` would
        # not catch it. NaN is the one value that fails equality with itself.
        #
        # px_total is not decoration: it is the weight a cell carries into the join. A cell
        # on the CONUS edge may be 90% NaN and must not count as a full one.
        #
        # NO `HAVING`, AND THAT IS A DELIBERATE DIFFERENCE FROM THE DEFORESTATION NOTEBOOK.
        # There, cells averaging zero are dropped because they are overwhelmingly ocean.
        # Here zero means ground that will not burn (water, dense urban core, irrigated
        # cropland), it is 3% of CONUS and up to 22% inside a city, and it is exactly where
        # the buildings are. Dropping it would punch holes through every city and strand the
        # footprints inside them with nothing to join to.
        return (
            ctx.sql(f"""
                SELECT h3_latlng_to_cell(y, x, CAST({res} AS INT)) AS hex,
                       avg(CAST(v AS DOUBLE)) AS rps,
                       count(*)               AS px_total
                FROM df
                WHERE v = v
                GROUP BY 1
            """).to_arrow_table(),
            blocks,
            empty,
        )

    # THE CANOPY FOLD. The reader hands back strided (lat, lng, metres) rows; the fold
    # is the same UDF group-by as the raster fold above, minus xarray, because the
    # pixels arrive already flattened. Memoised by coverage like the buildings fetch.
    _CAN_EMPTY = pa.table(
        {"hex": pa.array([], pa.uint64()), "canopy": pa.array([], pa.float64())}
    )
    _can_mem = []  # [[box, res, folded table, note], ...], newest last
    CAN_KEEP = 6

    def _can_covers(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    async def fold_canopy(box, res):
        """Mean canopy metres per H3 cell over `box`, or (None, why not)."""
        for ent in reversed(_can_mem):
            if ent[1] == res and _can_covers(ent[0], box):
                _can_mem.remove(ent)
                _can_mem.append(ent)
                return ent[2], "cached"
        pix, note = await read_canopy(box)
        if pix is None:
            return None, note
        # No await between the register and the query, same rule as the join below.
        _register("canpix", pix)
        tbl = ctx.sql(f"""
            SELECT h3_latlng_to_cell(lat, lng, CAST({res} AS INT)) AS hex,
                   avg(v) AS canopy
            FROM canpix
            GROUP BY 1
        """).to_arrow_table()
        _can_mem.append([box, res, tbl, note])
        while len(_can_mem) > CAN_KEEP:
            _can_mem.pop(0)
        return tbl, note

    # THE JOIN, IN DATAFUSION.
    #
    # An equi-join on a UBIGINT plus a group-by. No geometry, no H3 call, nothing a query
    # engine is not already the best tool for. The cells are already in this context, so
    # shipping them to DuckDB because DuckDB happens to hold the polygons would be backwards.
    #
    # avg over the covered cells, not max. A building spanning several cells gets the mean of
    # the ground under it, which is consistent with how every other number in this repo is
    # aggregated. `max(c.rps)` is a defensible alternative for a RISK layer, since the worst
    # ground a structure sits on is arguably what matters, and it is a one-word change here.
    #
    # The canopy leg is a LEFT join on the WIDENED mapping (the k=1 ring): fire risk is
    # the ground under the footprint, fuel is what stands around it. LEFT, because "no
    # CHM tile" must come through as NULL on a building that still has a fire number.
    JOIN_SQL = """
        WITH fire AS (
            SELECT b.id       AS id,
                   avg(c.rps) AS rps,
                   count(*)   AS n_cells
            FROM bld_cells b JOIN cells c ON b.hex = c.hex
            GROUP BY b.id
        ), fuel AS (
            SELECT n.id          AS id,
                   avg(k.canopy) AS canopy
            FROM bld_near n JOIN canopy k ON n.hex = k.hex
            GROUP BY n.id
        )
        SELECT fire.id, fire.rps, fire.n_cells, fuel.canopy
        FROM fire LEFT JOIN fuel ON fire.id = fuel.id
    """

    async def join_buildings(box, zoom, res, cells_tbl):
        """Buildings in view, each carrying the mean RPS of the cells it covers."""
        meta, key, note = await fetch_buildings(box, zoom)
        if meta is None or meta.num_rows == 0:
            return None

        # Trim to the box the cells actually cover. The fetch is deliberately wider (whole
        # z14 tiles, grown by BLD_PAD, so a pan is free) and everything in that margin has
        # no cell to join to. Dropping it here rather than letting the inner join do it is
        # what keeps the "off-grid" count meaning what it says: raster with no value under
        # a building, not a building outside the read.
        cx = np.asarray(meta["cx"])
        cy = np.asarray(meta["cy"])
        vis = meta.filter(
            pa.array((cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3]))
        )
        if vis.num_rows == 0:
            return None

        # The canopy fold happens BEFORE the synchronous register block below, because
        # it awaits the network; its own register/query pair is internally synchronous.
        canopy_tbl, can_note = await fold_canopy(box, res)

        # The polyfill still runs over the FULL cached fetch, not the trim: it is memoised
        # on (fetch, resolution) and stays reusable across pans that way, and the join
        # below only ever sees the ids in `vis`.
        mapping = polyfill(meta, key, res)
        if mapping.num_rows == 0:
            return None
        near = near_cells(mapping, key, res)
        # Synchronous from the register to the result on purpose: `ctx` is one shared
        # context and these are fixed names in it, so anything that awaited in the
        # middle could have its tables swapped out from under it by the next camera
        # event. With no await between them the event loop cannot interleave.
        _register("bld_cells", mapping)
        _register("bld_near", near)
        _register("cells", cells_tbl)
        _register("canopy", canopy_tbl if canopy_tbl is not None else _CAN_EMPTY)
        joined = ctx.sql(JOIN_SQL).to_arrow_table().combine_chunks()
        if joined.num_rows == 0:
            return None

        # Buildings that matched no cell get NO number rather than a guessed one. With
        # 'overlap' every footprint covers at least one cell, so the only way to land here
        # is ground the raster does not cover: outside CONUS, or a cell whose pixels were
        # all NaN. Counted rather than silently dropped.
        out = vis.join(joined, keys="id", join_type="inner")
        if out.num_rows == 0:
            return None
        HOLD["bldraw"] = out  # the paint switch repaints from this without refetching
        return (
            buildings_to_layer(out, controls.paint, from_wkb, multipolygon),
            out.num_rows,
            max(0, vis.num_rows - out.num_rows),
            f"{note} · canopy {can_note}",
        )

    HOLD["fold"] = fold
    HOLD["join"] = join_buildings
    HOLD["loop"] = asyncio.get_running_loop()

    # The opening draw. force=True skips the settle: there is nothing to debounce yet.
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]

    await refresh(_VS(), force=True)
    return


@app.cell
def _(BLD_ZOOM, controls, deck, legend, mo, status):
    mo.vstack(
        [
            deck,
            status,
            mo.Html(legend),
            controls,
            mo.md(
                "**RPS, risk to potential structures**: burn probability times the "
                "conditional risk to a structure were one there (CarbonPlan Open Climate "
                "Risk, CC-BY 4.0, CONUS at 30 m). Footprints: Overture Maps. "
                "**Canopy**: Meta & WRI High Resolution Canopy Height Maps, ~1 m "
                "(CC-BY 4.0); each building's number is the mean canopy height over its "
                "res 11 cells widened one ring, roughly everything within 80 m, which is "
                "defensible-space distance. The `colour` switch repaints the footprints "
                "by either number; the hexagons always carry fire risk. "
                f"Buildings appear above zoom {BLD_ZOOM:g}; below that the hexagons carry "
                "the map. At res 11 a cell is 2,150 m² against a 150-250 m² house, so a "
                "single house takes the value of the ground around it: the rasters do "
                "not resolve a house, they resolve the hillside it stands on. Canopy "
                "vintage varies per Maxar acquisition (2018-2020 mostly), so a cleared "
                "lot can still wear last year's trees."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
