# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "geoarrow-rust-core",
#     "obstore>=0.9.2",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
#     "matplotlib==3.11.1",
# ]
# ///
"""FEMA flood zones joined onto Overture divisions and buildings. All vector, one engine.

The hazard is FEMA's National Flood Hazard Layer (S_FLD_HAZ_AR): 5.63M flood-zone
polygons compiled from every effective FIRM, served as one 1.74 GB PMTiles archive on
source.coop (cboettig/hazard, public domain, z0-13, geometry simplified to ~10 m
upstream). Zoomed out, every county or locality carries the SHARE of its ground inside
the 1%-annual-chance floodplain; past zoom 13 the map switches to individual Overture
building footprints, each coloured by the worst flood zone it touches.

THIS IS THE FIRST NOTEBOOK HERE WITH NO RASTER, AND THAT COLLAPSES THE ENGINE SPLIT.
The repo's fold benchmark (xsql-duckdb-nlcd-h3.py) has two regimes: whole-column pixel
folds, where DataFusion + h3ronpy win 70 ms to 462, and polygon geometry ops, where
DuckDB's C-backed H3 wins 75 ms to 928. Every H3 cell in this notebook is born from a
polygon polyfill (zones, divisions, buildings), which is all the second regime, and the
remaining equi-joins are viewport-sized, so DuckDB does everything: polyfill, tile-seam
dissolve, and the joins. No DataFusion, no h3ronpy, no xarray.

THREE PMTILES ARCHIVES, ONE READER. flood-hazard.pmtiles (source.coop), plus Overture's
divisions.pmtiles and buildings.pmtiles, all PMTiles v3 with gzipped MVT, all read by
ranged GET through the same directory walk and hand-rolled protobuf decode the HFP and
fire-risk notebooks proved (ring-exact against mapbox-vector-tile). Tile-clipped pieces
are dissolved per feature id before drawing or filling, same as always.

THE JOIN RUNS ON TWO H3 LADDERS BRIDGED BY PARENTAGE, because FEMA zones are ribbons.
A floodway is often narrower than the cell that a wide view can afford, so polyfilling
zones at the division ladder's own resolution would 'center'-miss most of them. Zones
are therefore filled ONE resolution finer than the divisions (7x the samples) and
rolled up through h3_cell_to_parent, the bridge the parked canopy-deforest experiment
proved. Each H3 cell has exactly seven children, so the share arithmetic is exact in
cell counts. Buildings get the same trick at the bottom of the ladder: footprint cells
at res 11 (overlap mode, the fire-risk rule: a building contains no cell centre),
zones at res 12 (center), worst zone wins.

WHAT IS AND IS NOT DRAWN. V/VE (coastal 1%, wave action), the A family (1% annual
chance), and the shaded-X 0.2% band are drawn; D (undetermined) is drawn dim. X minimal
hazard, OPEN WATER and AREA NOT INCLUDED are dropped at decode: painting "minimal
hazard" would paint most of the country and say nothing. A building outside every
drawn zone is slate, which on this map is a statement, not an absence.

COLOUR. Zone class is ordinal, so it gets an ordered palette with the red axis unused:
orange for V/VE against blues for the rest, the classic protan-safe pairing. The
division share ramp is cividis. Nothing on this map distinguishes red from green.

NOAA SEAM. cboettig/hazard also carries sea-level-rise.pmtiles (NOAA 5 ft inundation,
147 MB, layer "sea-level-rise", fields state/region/slr_ft). HAZ_PATH and HAZ_LAYER
in the constants cell are the seam; a toggle would open a second archive alongside
this one and swap which table feeds the zone layer and the joins.

Data: FEMA NFHL via source.coop cboettig/hazard (US public domain); boundaries and
buildings: Overture Maps. Geometry in the hazard tiles is simplified to ~10 m, which
the publisher states is lossless at H3 res 10; the finest polyfill here is res 12,
inside a factor of two of that, and the zone assignment is stated per building as
"the worst zone its cells touch", not a legal determination.
Run:  uv run marimo edit xsql-flood-buildings.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import json
    import math
    import struct
    import urllib.parse
    import urllib.request

    import anywidget
    import traitlets
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import duckdb
    import numpy as np
    import obstore
    import pyarrow as pa
    from arro3.core import Array as ArroArray, Table as ArroTable
    from geoarrow.rust.core import from_wkb, multipolygon
    from obstore.store import S3Store
    from lonboard import Map, PolygonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard.controls import (
        FullscreenControl,
        GeocoderControl,
        NavigationControl,
        ScaleControl,
    )
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroArray,
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        FullscreenControl,
        GeocoderControl,
        Map,
        MaplibreBasemap,
        NavigationControl,
        PolygonLayer,
        S3Store,
        ScaleControl,
        anywidget,
        asyncio,
        duckdb,
        from_wkb,
        gzip,
        infer_rows_per_chunk,
        json,
        math,
        matplotlib,
        mo,
        multipolygon,
        np,
        obstore,
        pa,
        struct,
        traitlets,
        urllib,
    )


@app.cell
def _(duckdb):
    # THE ONLY QUERY ENGINE IN THIS NOTEBOOK. Every prior notebook split fold and
    # geometry between DataFusion and DuckDB; with no raster there is no fold, and the
    # polyfill, the dissolve and the equi-joins are all the regime DuckDB won in the
    # archived benchmark. Extensions download once into ~/.duckdb and are cached.
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class Status(anywidget.AnyWidget):
        """A one-line status readout the camera can write to, and the viewport ruler.

        A widget rather than `mo.md`, because the only way to update marimo output is to
        re-run the cell that produced it, and the cell holding the map is downstream of
        any state the camera could write: re-running it rebuilds the Map and throws the
        view away. A widget trait syncs straight to the browser instead.

        THE RULER is the HFP notebook's, verbatim: lonboard's view_state carries no
        canvas size, so this widget finds the deck canvas (largest canvas on the page,
        searched THROUGH shadow roots, because marimo puts cell output in shadow DOM),
        watches it with a ResizeObserver plus resize and fullscreenchange listeners,
        and syncs "WxH" up as a Unicode trait, the one browser-to-kernel trait type
        proven to cross marimo's anywidget bridge.
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
                w = window.innerWidth; h = window.innerHeight; tag = "ruler window ";
              }
              if (w > 0 && h > 0) {
                probe.textContent = tag + w + "x" + h;
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

        Same constraint as Status: an `mo.ui.checkbox` would make the map cell depend
        on it, so every click would rebuild the Map and reset the camera. A widget
        trait syncs to the kernel, a Python observer assigns onto the deck layers,
        and nothing re-runs.
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
          check("show_zones", "flood zones");
          check("show_divisions", "boundaries");
          check("division_fill", "boundary fill");
          el.appendChild(box);

          // HIDE LONBOARD'S DRAW-BOX TOOL, same as the HFP notebook: the toolbar is
          // rendered unconditionally in lonboard 0.16's bundle and the Map's
          // `controls` trait cannot reach it, so it is hidden from here with the same
          // recurse-into-shadowRoots walk the ruler uses, on an interval because the
          // map mounts later and can be rebuilt.
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
        show_zones = traitlets.Bool(True).tag(sync=True)
        show_divisions = traitlets.Bool(True).tag(sync=True)
        # OFF BY DEFAULT, the opposite of the HFP notebook, deliberately: the zone
        # polygons are the ground truth here and a cividis wash over them would bury
        # them. The share still shows in the stroke and the tooltip.
        division_fill = traitlets.Bool(False).tag(sync=True)

    return Controls, Status


@app.cell
def _(math):
    # ------------------------------------------------------------------ the hazard
    HAZ_BUCKET = "us-west-2.opendata.source.coop"
    # THE NOAA SEAM. sea-level-rise.pmtiles (layer "sea-level-rise", field slr_ft) is
    # the same archive format in the same account; a second Archive opened on it plus a
    # zone_class for inundation is the whole diff a toggle would need.
    HAZ_PATH = "cboettig/hazard/flood-hazard.pmtiles"
    HAZ_LAYER = "flood-hazard"

    # ~4 tiles across the padded box, like the divisions ladder. The archive tops out
    # at z13; a viewport that would need more tiles than the cap COARSENS instead of
    # refusing, because unlike a drawn-box ranking the zones must always draw.
    ZONE_TILE_CAP = 256

    # ------------------------------------------------------------------ the zoom ladder
    # One H3 resolution per 1.4 zoom levels (each H3 step is 2.65x linear,
    # log2(2.65) = 1.4), floored with math.floor, not int(): int truncates toward zero
    # and would collapse everything below ZOOM0 onto BASE_RES.
    ZOOM0, PER_RES, BASE_RES = 4.0, 1.4, 5
    MIN_RES, MAX_RES = 4, 10

    def res_for_zoom(z):
        return max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES)))

    # ZONES ARE FILLED ONE RESOLUTION FINER THAN THE DIVISIONS AND ROLLED UP THROUGH
    # h3_cell_to_parent (each cell has exactly 7 children, so counts are exact). One
    # step, not two: FEMA country like coastal Louisiana is ~half SFHA, so at +2 a
    # region-band view polyfills millions of zone cells; at +1 the same view is ~200k
    # and a floodway ribbon is still sampled 7x finer than the division grid.
    ZFINE = 1
    ZFINE_MAX = 11

    # WHICH DIVISION LEVEL IS DRAWN AT WHICH ZOOM. Same bands as the HFP notebook;
    # above BLD_ZOOM the buildings take over and divisions leave the screen entirely.
    DIV_ZOOM = 4.5

    def division_for_zoom(z):
        if z < DIV_ZOOM:
            return None
        if z < 7.0:
            return "region"
        if z < 9.5:
            return "county"
        return "locality"

    DIVISION_LABEL = {
        "country": "countries",
        "region": "regions",
        "county": "counties",
        "locality": "localities",
    }

    # ------------------------------------------------------------------ boundaries
    OVERTURE_RELEASE = "2026-07-22.0"
    OVT_BUCKET = "overturemaps-extras-us-west-2"
    DIV_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    # Tile zooms where each division subtype first exists, measured off the tiles in
    # the HFP notebook (Planetiler bakes the minzooms into the build, undocumented).
    SUB_MINZOOM = {"country": 2, "region": 4, "county": 8, "locality": 10}
    DIV_TILE_CAP = 256

    # ------------------------------------------------------------------ buildings
    # All four facts below are measurements from xsql-firerisk-buildings.py, not
    # choices: attributes exist ONLY at z14 (id on 100% of z14 features and 0% of z13,
    # measured at four places, so a coarser fetch returns anonymous geometry that can
    # be neither dissolved nor joined); below z13 the layer is a thinned sample that
    # reads as texture; res 11 overlap is the polyfill rule for footprints smaller
    # than any cell; and a z13-14 viewport is well under 64 tiles.
    BLD_PATH = f"tiles/{OVERTURE_RELEASE}/buildings.pmtiles"
    BLD_LAYER = "building"
    BLD_ZOOM = 10.0
    BLD_TILE_Z = 14
    BLD_TILE_CAP = 64
    BLD_RES = 11
    # The zone side of the building join: res 12 cells (307 m2, ~19 m across) so a
    # narrow floodway still catches centres, rolled up to the building's res 11.
    ZBLD_RES = 12

    # ------------------------------------------------------------------ view
    # VIEW_W/VIEW_H and HOME live in the MAP cell, not here: the map cell must not
    # depend on this one, so that editing a constant never rebuilds the deck widget.
    # SETTLE only guards a read; camera moves answerable from memory are answered
    # synchronously in the comm handler (see _instant).
    SETTLE = 0.15

    # Division choropleth alphas, same meaning as the HFP notebook: fill is a wash,
    # stroke survives the fill toggle.
    FILL_ALPHA = 150
    LINE_ALPHA = 205
    # Zone fill alpha: strong enough to read as ground, weak enough that the dark
    # basemap's streets stay legible through it.
    ZONE_ALPHA = 120
    return (
        BLD_LAYER,
        BLD_PATH,
        BLD_RES,
        BLD_TILE_CAP,
        BLD_TILE_Z,
        BLD_ZOOM,
        DIVISION_LABEL,
        DIV_PATH,
        DIV_TILE_CAP,
        FILL_ALPHA,
        HAZ_BUCKET,
        HAZ_LAYER,
        HAZ_PATH,
        LINE_ALPHA,
        OVT_BUCKET,
        SETTLE,
        SUB_MINZOOM,
        ZBLD_RES,
        ZFINE,
        ZFINE_MAX,
        ZONE_ALPHA,
        ZONE_TILE_CAP,
        division_for_zoom,
        res_for_zoom,
    )


@app.cell
def _(matplotlib, np):
    # ------------------------------------------------------------------ zone classes
    # FLD_ZONE is a taxonomy, not an axis, so it is folded to an ordinal class first:
    #   3  V, VE          coastal 1% annual chance, wave action
    #   2  A family       1% annual chance (the SFHA: A, AE, AH, AO, A99, AR)
    #   1  shaded X       0.2% annual chance ("500-year"), marked in ZONE_SUBTY
    #   0  D              possible but undetermined, no study
    #  -1  everything else: X minimal hazard, OPEN WATER, AREA NOT INCLUDED. Dropped
    #      at decode; drawing "minimal hazard" would paint most of the country.
    # Membership is an explicit set, NOT startswith("A"): FLD_ZONE's own vocabulary
    # includes "AREA NOT INCLUDED".
    _SFHA_A = {"A", "AE", "AH", "AO", "A99", "AR"}

    def zone_class(zone, subty):
        z = (zone or "").strip().upper()
        if z in ("V", "VE"):
            return 3
        if z in _SFHA_A:
            return 2
        if z == "X" and "0.2" in (subty or ""):
            return 1
        if z == "D":
            return 0
        return -1

    # Ordinal palette with the red axis unused: orange against blues, the protan-safe
    # pairing, severity also carried by saturation. Index is zc + 1, so -1 (outside,
    # buildings only) is the first row.
    ZONE_LUT = np.array(
        [
            [64, 70, 82],  # -1 outside every drawn zone (buildings only)
            [118, 118, 118],  # 0 D undetermined
            [116, 141, 166],  # 1 shaded X, 0.2% annual chance
            [62, 146, 204],  # 2 A family, 1% annual chance
            [230, 159, 0],  # 3 V/VE coastal 1%
        ],
        dtype=np.uint8,
    )
    ZONE_LABEL = {
        -1: "outside mapped flood zones",
        0: "undetermined (D)",
        1: "0.2% annual chance",
        2: "1% annual chance",
        3: "coastal 1% (V/VE)",
    }

    # The division share ramp: cividis, linear over 0-1. Shares cluster low outside
    # the coasts, but the interesting places (parishes, coastal counties) genuinely
    # span the range, and a log ramp here would just relabel the legend.
    _cmap = matplotlib.colormaps["cividis"]

    def share_rgba(v, alpha):
        c = (_cmap(np.clip(np.asarray(v, dtype="float64"), 0.0, 1.0))[:, :3] * 255.0)
        out = np.empty((len(c), 4), dtype=np.uint8)
        out[:, :3] = c.astype(np.uint8)
        out[:, 3] = alpha
        return out

    def _chip(rgb, label):
        return (
            f"<span style='display:inline-flex;align-items:center;gap:.3rem;"
            f"margin-right:.9rem'><span style='width:.85em;height:.85em;"
            f"border-radius:2px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>"
            f"{label}</span>"
        )

    legend = (
        "<div style='font:12px ui-sans-serif,system-ui,sans-serif;padding:.25rem 0'>"
        + _chip(ZONE_LUT[4], "coastal 1% (V/VE)")
        + _chip(ZONE_LUT[3], "1% annual chance")
        + _chip(ZONE_LUT[2], "0.2% annual chance")
        + _chip(ZONE_LUT[1], "undetermined (D)")
        + _chip(ZONE_LUT[0], "building outside mapped zones")
        + "<span style='opacity:.65'>boundary stroke: share of ground in the 1% "
        "floodplain, dark (0) to yellow (1)</span></div>"
    )
    return ZONE_LABEL, ZONE_LUT, legend, share_rgba, zone_class


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs
    # every downstream cell, including the one that owns the Map, so the camera would
    # snap home on every pan. A plain dict is invisible to the dataflow graph.
    HOLD = {
        "wh": (1400.0, 620.0),  # measured canvas; the constants are only the seed
        "vs": None,  # last camera acted on, for the echo check
        # A rebuild happened and the browser has not reported in yet; the first ruler
        # report or camera event after it triggers a full re-send (see repaint).
        "fresh": True,
        "busy": False,
        "pending": None,
        "loop": None,
        "task": None,
        "draw": None,  # the machinery cell's _draw, wired at the end
        # what is on screen, for _instant
        "regime": None,  # "div", "bld" or None (zones only)
        "zbox": None,  # coverage of the zones on screen
        "ztz": None,  # tile zoom the zones were fetched at
        "res": None,  # division polyfill resolution on screen
        "div": None,  # division subtype on screen
        "divbox": None,
        "divpair": None,  # (fill-on, fill-off) tables on the division layer
        "bldbox": None,
        "head": "",
        "tail": "",
    }
    return (HOLD,)


@app.cell
def _(
    ArroArray,
    ArroTable,
    FILL_ALPHA,
    LINE_ALPHA,
    ZONE_ALPHA,
    ZONE_LABEL,
    ZONE_LUT,
    from_wkb,
    multipolygon,
    np,
    pa,
    share_rgba,
    struct,
):
    _MP = multipolygon("xy", crs="EPSG:4326")

    def _geom(tbl):
        """WKB column -> geoarrow multipolygon, straight off the Arrow buffers."""
        return ArroArray.from_arrow(
            from_wkb(tbl["wkb"].combine_chunks(), to_type=_MP)
        )

    def zones_to_layer(tbl):
        """Dissolved zone polygons -> the table the zone PolygonLayer draws."""
        tbl = tbl.combine_chunks()
        zc = np.asarray(tbl["zc"], dtype="int64")
        rgba = np.empty((len(zc), 4), dtype=np.uint8)
        rgba[:, :3] = ZONE_LUT[zc + 1]
        rgba[:, 3] = ZONE_ALPHA
        bfe = np.asarray(tbl["bfe"], dtype="float64")
        return ArroTable.from_arrays(
            [
                _geom(tbl),
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(rgba.ravel()), 4)
                ),
                ArroArray.from_arrow(tbl["zone"].combine_chunks()),
                ArroArray.from_arrow(tbl["subty"].combine_chunks()),
                # STATIC_BFE uses -9999 for N/A; shown only where it is a number.
                ArroArray.from_arrow(
                    pa.array(
                        [f"{v:.0f} ft" if v > -9000 else "" for v in bfe]
                    )
                ),
            ],
            names=["geometry", "color", "zone", "detail", "base_flood_elev"],
        )

    def buildings_to_layer(tbl):
        """Joined buildings -> the table the building PolygonLayer draws."""
        tbl = tbl.combine_chunks()
        zc = np.asarray(tbl["zc"], dtype="int64")
        rgba = np.empty((len(zc), 4), dtype=np.uint8)
        rgba[:, :3] = ZONE_LUT[zc + 1]
        rgba[:, 3] = 235
        return ArroTable.from_arrays(
            [
                _geom(tbl),
                ArroArray.from_arrow(
                    pa.FixedSizeListArray.from_arrays(pa.array(rgba.ravel()), 4)
                ),
                ArroArray.from_arrow(pa.array([ZONE_LABEL[int(v)] for v in zc])),
                ArroArray.from_arrow(tbl["name"].combine_chunks()),
                ArroArray.from_arrow(tbl["class"].combine_chunks()),
            ],
            names=["geometry", "color", "flood_zone", "name", "class"],
        )

    def divisions_pair(tbl):
        """Division shares -> the TWO tables the fill toggle swaps between.

        Two tables, identical except the fill alpha, because `filled` cannot be
        flipped after init and `get_fill_color` must always be a column of the same
        schema: the pair of lessons the HFP notebook's dead fill button paid for.
        The stroke has its OWN column so it survives the zero-alpha fill.
        """
        tbl = tbl.combine_chunks()
        share = np.asarray(tbl["share"], dtype="float64")
        geom = _geom(tbl)
        line = ArroArray.from_arrow(
            pa.FixedSizeListArray.from_arrays(
                pa.array(share_rgba(share, LINE_ALPHA).ravel()), 4
            )
        )
        rest = [
            ArroArray.from_arrow(tbl["name"].combine_chunks()),
            ArroArray.from_arrow(tbl["region"].combine_chunks()),
            ArroArray.from_arrow(pa.array(np.round(share * 100.0, 1))),
            ArroArray.from_arrow(tbl["n_cells"].combine_chunks()),
        ]
        names = ["geometry", "color", "line", "name", "region", "pct_sfha", "cells"]

        def build(alpha):
            col = ArroArray.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(share_rgba(share, alpha).ravel()), 4
                )
            )
            return ArroTable.from_arrays([geom, col, line, *rest], names=names)

        return build(FILL_ALPHA), build(0)

    return buildings_to_layer, divisions_pair, zones_to_layer


@app.cell
def _(ArroArray, ArroTable, from_wkb, multipolygon, np, pa, struct):
    # ------------------------------------------------------------------ seeds
    # lonboard will not take table=None, and a degenerate polygon kills deck's whole
    # update pass (the earcut cascade recorded in CLAUDE.md), so every layer starts
    # with one real 0.01 degree square at null island, invisible from anywhere this
    # map opens, with EXACTLY the schema the real pushes will carry: a seed whose
    # colour is a different width would make the first push a change of accessor
    # width, the class of swap that leaves fills unpainted.
    #
    # IN THEIR OWN CELL, apart from the layer builders, deliberately: the seeds feed
    # the MAP cell, which must depend on as little as possible so that ordinary edits
    # (palette, constants, machinery) never rebuild the deck widget. The builders
    # depend on the palette; the seeds are all zero-alpha and depend on nothing.
    _MP = multipolygon("xy", crs="EPSG:4326")

    def _seed_wkb():
        d = 0.01
        ring = [(0.0, 0.0), (d, 0.0), (d, d), (0.0, d), (0.0, 0.0)]
        wkb = struct.pack("<BII", 1, 6, 1)
        wkb += struct.pack("<BIII", 1, 3, 1, len(ring))
        for x, y in ring:
            wkb += struct.pack("<dd", x, y)
        return wkb

    def _seed_geom():
        return ArroArray.from_arrow(
            from_wkb(pa.array([_seed_wkb()], pa.binary()), to_type=_MP)
        )

    def _rgba0():
        return ArroArray.from_arrow(
            pa.FixedSizeListArray.from_arrays(
                pa.array(np.zeros(4, dtype=np.uint8)), 4
            )
        )

    def seed_zones():
        return ArroTable.from_arrays(
            [
                _seed_geom(),
                _rgba0(),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
            ],
            names=["geometry", "color", "zone", "detail", "base_flood_elev"],
        )

    def seed_buildings():
        return ArroTable.from_arrays(
            [
                _seed_geom(),
                _rgba0(),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
            ],
            names=["geometry", "color", "flood_zone", "name", "class"],
        )

    def seed_divisions():
        return ArroTable.from_arrays(
            [
                _seed_geom(),
                _rgba0(),
                _rgba0(),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([""])),
                ArroArray.from_arrow(pa.array([0.0])),
                ArroArray.from_arrow(pa.array([0], type=pa.int64())),
            ],
            names=["geometry", "color", "line", "name", "region", "pct_sfha", "cells"],
        )

    return seed_buildings, seed_divisions, seed_zones


@app.cell
async def _(
    BLD_LAYER,
    BLD_PATH,
    BLD_RES,
    BLD_TILE_CAP,
    BLD_TILE_Z,
    DIV_PATH,
    DIV_TILE_CAP,
    HAZ_BUCKET,
    HAZ_LAYER,
    HAZ_PATH,
    OVT_BUCKET,
    S3Store,
    SUB_MINZOOM,
    ZBLD_RES,
    ZFINE,
    ZFINE_MAX,
    ZONE_TILE_CAP,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    struct,
    zone_class,
):
    # THREE PMTILES ARCHIVES THROUGH ONE READER. The client is the one ported from
    # xsql-duckdb-terrain-h3.py into the divisions and fire-risk notebooks, here
    # parameterised over the archive instead of copied per archive. Opening costs two
    # reads each (127-byte header, root directory); leaf directories are parsed once
    # and cached per archive.
    _haz_store = S3Store(HAZ_BUCKET, region="us-west-2", skip_signature=True)
    _ovt_store = S3Store(OVT_BUCKET, region="us-west-2", skip_signature=True)
    _sem = asyncio.Semaphore(32)

    async def _range(ar, a, b):
        """Inclusive byte range [a, b]. obstore's `end` is exclusive."""
        return bytes(
            memoryview(
                await obstore.get_range_async(
                    ar["store"], ar["path"], start=a, end=b + 1
                )
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

        run_length 0 marks a pointer to a LEAF directory. A zero OFFSET means
        "immediately after the previous entry", so offsets are reconstructed.
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
        """Binary search, falling back to the run that COVERS tid."""
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

    async def _open(store, path):
        ar = {"store": store, "path": path}
        hdr = await _range(ar, 0, 126)
        assert hdr[:7] == b"PMTiles" and hdr[7] == 3, f"not PMTiles v3: {path}"
        rd_off, rd_len, _, _, ld_off, _, td_off, _ = struct.unpack("<8Q", hdr[8:72])
        ar["ld"], ar["td"], ar["maxz"] = ld_off, td_off, hdr[101]
        ar["root"] = _parse_dir(gzip.decompress(await _range(ar, rd_off, rd_off + rd_len - 1)))
        ar["leaf"] = {}
        return ar

    HAZ = await _open(_haz_store, HAZ_PATH)
    DIV = await _open(_ovt_store, DIV_PATH)
    BLD = await _open(_ovt_store, BLD_PATH)

    async def _tile_buf(ar, z, x, y):
        """One tile's decompressed bytes, walked to through the directories."""
        tid, ents = _tile_id(z, x, y), ar["root"]
        for _ in range(4):  # root + up to three leaf levels
            e = _find(ents, tid)
            if e is None:
                return None
            if e[3] == 0:
                lk = (e[1], e[2])
                if lk not in ar["leaf"]:
                    async with _sem:
                        ar["leaf"][lk] = _parse_dir(
                            gzip.decompress(
                                await _range(
                                    ar, ar["ld"] + e[1], ar["ld"] + e[1] + e[2] - 1
                                )
                            )
                        )
                ents = ar["leaf"][lk]
                continue
            async with _sem:
                blob = await _range(ar, ar["td"] + e[1], ar["td"] + e[1] + e[2] - 1)
            return gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
        return None

    # ------------------------------------------------------------- the MVT decode
    # Hand-rolled, verified ring-exact and property-exact against mapbox-vector-tile
    # in the divisions notebook before being trusted with anything.
    def _fields(buf):
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
        rings, ring = [], None
        x = y = 0
        i, n = 0, len(geom)
        while i < n:
            cmd, i = _varint(geom, i)
            op, count = cmd & 0x7, cmd >> 3
            if op == 1:
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring = [(x, y)]
                    rings.append(ring)
            elif op == 2:
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring.append((x, y))
            elif op == 7:
                ring.append(ring[0])
            else:
                raise ValueError(f"geometry op {op}")
        return rings

    def _area2(ring):
        """Twice the signed shoelace area: >0 marks an exterior ring (tile y is down)."""
        a = 0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            a += x0 * y1 - x1 * y0
        return a

    def _layer_feats(buf, want):
        """One named layer of one tile: ([(props, [(exterior, holes), ...]), ...], extent)."""
        for f, _w, v in _fields(buf):
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
            if name != want:
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
        """Tile-integer rings -> a lon/lat MultiPolygon WKB (closed-form Web Mercator)."""
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

    # ------------------------------------------------------------- tile geometry
    def _mtile(lon, lat, z):
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
        n = 1 << z

        def lat(yy):
            return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yy / n))))

        return (
            x0 / n * 360.0 - 180.0,
            lat(y1 + 1),
            (x1 + 1) / n * 360.0 - 180.0,
            lat(y0),
        )

    def _grow(b, f):
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

    # Decoded tiles kept per archive, LRU by insertion order, capped by count.
    _zone_tiles, _div_tiles, _bld_tiles = {}, {}, {}
    TILE_KEEP = 2048

    def _lru(cache, k, v):
        cache[k] = v
        while len(cache) > TILE_KEEP:
            cache.pop(next(iter(cache)))

    # ================================================================= FLOOD ZONES
    async def _zone_pieces(z, x, y):
        """One hazard tile: [(fid, zone, subty, zc, bfe, wkb), ...].

        Class -1 (X minimal, OPEN WATER, AREA NOT INCLUDED) is dropped HERE, so the
        dissolve, the polyfill and the render never see it. fid is the dissolve key;
        tippecanoe's coalescing keeps one survivor's attributes, which is the same
        bargain the zones already made by being generalised tiles at all.
        """
        k = (z, x, y)
        if k in _zone_tiles:
            _zone_tiles[k] = _zone_tiles.pop(k)
            return _zone_tiles[k]
        buf = await _tile_buf(HAZ, z, x, y)
        pieces = []
        if buf is not None:
            feats, extent = _layer_feats(buf, HAZ_LAYER)
            for props, polys in feats:
                if not polys:
                    continue
                fid = props.get("fid")
                if fid is None:
                    continue
                zc = zone_class(props.get("FLD_ZONE"), props.get("ZONE_SUBTY"))
                if zc < 0:
                    continue
                pieces.append(
                    (
                        int(fid),
                        props.get("FLD_ZONE") or "",
                        props.get("ZONE_SUBTY") or "",
                        zc,
                        float(props.get("STATIC_BFE") or -9999.0),
                        _feature_wkb(polys, z, x, y, extent),
                    )
                )
        _lru(_zone_tiles, k, pieces)
        return pieces

    # THE SEAM DISSOLVE, same reason as every notebook: tile geometry arrives clipped
    # and buffered, so one zone polygon is several overlapping pieces, and an alpha
    # fill double-paints every overlap into a visible grid. Union per fid heals it.
    ZONE_DISSOLVE_SQL = """
        SELECT fid,
               any_value(zone)  AS zone,
               any_value(subty) AS subty,
               any_value(zc)    AS zc,
               any_value(bfe)   AS bfe,
               ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb)))::BLOB AS wkb
        FROM pieces
        GROUP BY fid
    """

    _zone_mem = []  # [[coverage box, tz, table, key], ...], newest last
    ZONE_PAD = 1.35
    ZONE_KEEP = 8

    def _tz_zones(box):
        """~4 tiles across the box, capped by the pyramid, COARSENED under the cap."""
        span = max(box[2] - box[0], 1e-9)
        return max(0, min(HAZ["maxz"], int(math.log2(max(4.0 * 360.0 / span, 1.0)))))

    async def fetch_zones(bbox):
        """Flood zones covering bbox, dissolved per fid.

        Returns (table or None, key, coverage box, note). Cached by coverage; a hit
        must have been fetched at LEAST as fine as this box wants, or a zoom-in would
        be served generalised geometry from three levels up.
        """
        big = _grow(bbox, ZONE_PAD)
        tz = _tz_zones(big)
        for ent in reversed(_zone_mem):
            if ent[1] >= tz and _inside(ent[0], bbox):
                _zone_mem.remove(ent)
                _zone_mem.append(ent)
                return ent[2], ent[3], ent[0], "cached"

        x0, y0 = _mtile(big[0], big[3], tz)
        x1, y1 = _mtile(big[2], big[1], tz)
        while (x1 - x0 + 1) * (y1 - y0 + 1) > ZONE_TILE_CAP and tz > 0:
            tz -= 1
            x0, y0 = _mtile(big[0], big[3], tz)
            x1, y1 = _mtile(big[2], big[1], tz)

        got = await asyncio.gather(
            *(
                _zone_pieces(tz, tx, ty)
                for ty in range(y0, y1 + 1)
                for tx in range(x0, x1 + 1)
            )
        )
        rows = [p for tile in got for p in tile]
        cov = _range_box(tz, x0, y0, x1, y1)
        key = ("z", tz, x0, y0, x1, y1)
        if not rows:
            _zone_mem.append([cov, tz, None, key])
            del _zone_mem[:-ZONE_KEEP]
            return None, key, cov, "no zones"

        pieces = pa.table(
            {
                "fid": pa.array([r[0] for r in rows], pa.int64()),
                "zone": pa.array([r[1] for r in rows], pa.string()),
                "subty": pa.array([r[2] for r in rows], pa.string()),
                "zc": pa.array([r[3] for r in rows], pa.int32()),
                "bfe": pa.array([r[4] for r in rows], pa.float64()),
                "wkb": pa.array([r[5] for r in rows], pa.binary()),
            }
        )
        out = con.sql(ZONE_DISSOLVE_SQL).to_arrow_table()
        _zone_mem.append([cov, tz, out, key])
        del _zone_mem[:-ZONE_KEEP]
        n = (x1 - x0 + 1) * (y1 - y0 + 1)
        return out, key, cov, f"{n} tiles @ z{tz}"

    # The zone polyfill: 'center', because zones tile the studied ground without
    # overlapping, the same partition argument as divisions. The under-count on
    # ribbons narrower than a cell is what the finer-ladder bridge in the joins
    # compensates for.
    ZONE_FILL_SQL = """
        WITH parts AS (
            SELECT zc, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM zones
        ), filled AS (
            SELECT zc, UNNEST(
                       h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'center')
                   ) AS hex
            FROM parts
        )
        SELECT hex, MAX(zc) AS zc FROM filled GROUP BY hex
    """
    _EMPTY_ZCELLS = pa.table(
        {"hex": pa.array([], pa.uint64()), "zc": pa.array([], pa.int32())}
    )
    _zfill_memo = {}
    ZFILL_KEEP = 16

    def zone_cells(ztbl, zkey, res):
        """(hex, zc) for the fetched zones at one resolution, memoised on (key, res)."""
        if ztbl is None:
            return _EMPTY_ZCELLS
        ck = (zkey, int(res))
        if ck in _zfill_memo:
            return _zfill_memo[ck]
        zones = ztbl  # noqa: F841 - read by DuckDB's replacement scan
        out = con.sql(ZONE_FILL_SQL, params=[int(res)]).to_arrow_table()
        _zfill_memo[ck] = out
        while len(_zfill_memo) > ZFILL_KEEP:
            _zfill_memo.pop(next(iter(_zfill_memo)))
        return out

    # ================================================================= DIVISIONS
    async def _div_pieces(z, x, y):
        """One divisions tile, filtered to land, all subtypes kept."""
        k = (z, x, y)
        if k in _div_tiles:
            _div_tiles[k] = _div_tiles.pop(k)
            return _div_tiles[k]
        buf = await _tile_buf(DIV, z, x, y)
        pieces = []
        if buf is not None:
            feats, extent = _layer_feats(buf, "division_area")
            for props, polys in feats:
                if props.get("is_land") is not True or not polys:
                    continue
                pieces.append(
                    {
                        "sub": props.get("subtype"),
                        # division_id, not id: `id` names the AREA row, and a
                        # division can own several. Same lesson as the HFP notebook.
                        "id": props.get("division_id") or props.get("id"),
                        "name": props.get("@name"),
                        "region": props.get("region"),
                        "wkb": _feature_wkb(polys, z, x, y, extent),
                    }
                )
        _lru(_div_tiles, k, pieces)
        return pieces

    DIV_DISSOLVE_SQL = """
        SELECT id,
               any_value(name)   AS name,
               any_value(region) AS region,
               CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
        FROM pieces
        GROUP BY id
    """

    _div_mem = {}  # subtype -> [[coverage box, table, key], ...]
    DIV_PAD = 1.4
    DIV_KEEP = 8

    def _tz_divisions(subtype, box):
        span = max(box[2] - box[0], 1e-9)
        z = int(math.log2(max(4.0 * 360.0 / span, 1.0)))
        return max(SUB_MINZOOM[subtype], min(DIV["maxz"], z))

    async def fetch_divisions(subtype, bbox):
        """Overture divisions of one subtype covering bbox. (table, key, coverage)."""
        for box, tbl, key in _div_mem.get(subtype, []):
            if _inside(box, bbox):
                return tbl, key, box

        big = _grow(bbox, DIV_PAD)
        tz = _tz_divisions(subtype, big)
        x0, y0 = _mtile(big[0], big[3], tz)
        x1, y1 = _mtile(big[2], big[1], tz)
        key = (subtype, tz, x0, y0, x1, y1)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > DIV_TILE_CAP:
            return None, key, None

        got = await asyncio.gather(
            *(
                _div_pieces(tz, tx, ty)
                for ty in range(y0, y1 + 1)
                for tx in range(x0, x1 + 1)
            )
        )
        rows = [p for tp in got for p in tp if p["sub"] == subtype]
        cov = _range_box(tz, x0, y0, x1, y1)
        held = _div_mem.setdefault(subtype, [])
        if not rows:
            held.append([cov, None, key])
            del held[:-DIV_KEEP]
            return None, key, cov

        pieces = pa.table(
            {
                "id": pa.array([r["id"] for r in rows]),
                "name": pa.array([r["name"] for r in rows]),
                # "US-LA" -> "LA"; the country prefix says nothing new on a US map.
                "region": pa.array(
                    [(r["region"] or "").split("-", 1)[-1] for r in rows]
                ),
                "wkb": pa.array([r["wkb"] for r in rows], pa.binary()),
            }
        )
        out = con.sql(DIV_DISSOLVE_SQL).to_arrow_table()
        held.append([cov, out, key])
        del held[:-DIV_KEEP]
        return out, key, cov

    # 'center' for divisions, the partition rule: every cell in exactly one division.
    DIV_FILL_SQL = """
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
    _dfill_memo = {}
    DFILL_KEEP = 16

    def _polyfill_divisions(divs_tbl, key, res):
        ck = (key, int(res))
        if ck in _dfill_memo:
            return _dfill_memo[ck]
        divs = divs_tbl  # noqa: F841 - read by the replacement scan
        out = con.sql(DIV_FILL_SQL, params=[int(res)]).to_arrow_table()
        _dfill_memo[ck] = out
        while len(_dfill_memo) > DFILL_KEEP:
            _dfill_memo.pop(next(iter(_dfill_memo)))
        return out

    # THE DIVISION JOIN, ACROSS THE TWO LADDERS. Divisions are filled at the display
    # resolution r; zones one step finer (ZFINE), rolled up to r through
    # h3_cell_to_parent. Every res r cell has exactly 7^ZFINE children, so the share
    # is exact in cell counts: fine SFHA children over (division cells x 7^ZFINE).
    # zc >= 2 is the SFHA line: the A family and V/VE, not the 0.2% band, not D.
    DIV_JOIN_SQL = """
        WITH zp AS (
            SELECT h3_cell_to_parent(hex, ?) AS phex,
                   COUNT(*) FILTER (WHERE zc >= 2) AS n_sfha
            FROM zcells
            GROUP BY 1
        ), s AS (
            SELECT d.id,
                   COUNT(*) AS n_cells,
                   COALESCE(SUM(z.n_sfha), 0) AS sfha_fine
            FROM dcells d
            LEFT JOIN zp z ON d.hex = z.phex
            GROUP BY d.id
        )
        SELECT m.id, m.name, m.region, s.n_cells,
               LEAST(1.0, CAST(s.sfha_fine AS DOUBLE) / (s.n_cells * POWER(7, ?))) AS share,
               m.wkb
        FROM divs m
        JOIN s USING (id)
    """

    def join_divisions(divs_tbl, dkey, ztbl, zkey, res):
        """Share of each division's cells inside the SFHA. (table, n unmeasured)."""
        dcells_t = _polyfill_divisions(divs_tbl, dkey, res)
        if dcells_t.num_rows == 0:
            return None, divs_tbl.num_rows
        rz = min(int(res) + ZFINE, ZFINE_MAX)
        zcells_t = zone_cells(ztbl, zkey, rz)
        dcells, zcells, divs = dcells_t, zcells_t, divs_tbl  # noqa: F841
        out = con.sql(
            DIV_JOIN_SQL, params=[int(res), int(rz - res)]
        ).to_arrow_table()
        return out, divs_tbl.num_rows - out.num_rows

    # ================================================================= BUILDINGS
    async def _bld_pieces(z, x, y):
        """One buildings tile at z14: [(id, name, class, wkb), ...], no underground."""
        k = (z, x, y)
        if k in _bld_tiles:
            _bld_tiles[k] = _bld_tiles.pop(k)
            return _bld_tiles[k]
        buf = await _tile_buf(BLD, z, x, y)
        pieces = []
        if buf is not None:
            feats, extent = _layer_feats(buf, BLD_LAYER)
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
        _lru(_bld_tiles, k, pieces)
        return pieces

    BLD_DISSOLVE_SQL = """
        SELECT id,
               any_value(name)  AS name,
               any_value(class) AS class,
               CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
        FROM pieces
        GROUP BY id
    """

    _bld_mem = []  # [[coverage box, table, key], ...]
    BLD_PAD = 1.3
    BLD_KEEP = 6

    async def fetch_buildings(bbox):
        """Buildings covering bbox, dissolved per id. (table, key, coverage, note)."""
        for ent in reversed(_bld_mem):
            if _inside(ent[0], bbox):
                _bld_mem.remove(ent)
                _bld_mem.append(ent)
                return ent[1], ent[2], ent[0], "cached"

        want = _grow(bbox, BLD_PAD)
        x0, y0 = _mtile(want[0], want[3], BLD_TILE_Z)
        x1, y1 = _mtile(want[2], want[1], BLD_TILE_Z)
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_tiles > BLD_TILE_CAP:
            return None, None, None, "capped"

        got = await asyncio.gather(
            *(
                _bld_pieces(BLD_TILE_Z, tx, ty)
                for ty in range(y0, y1 + 1)
                for tx in range(x0, x1 + 1)
            )
        )
        rows = [p for tile in got for p in tile]
        cov = _range_box(BLD_TILE_Z, x0, y0, x1, y1)
        key = ("b", x0, y0, x1, y1)
        if not rows:
            return None, key, cov, "empty"

        pieces = pa.table(
            {
                "id": pa.array([p[0] for p in rows], pa.string()),
                "name": pa.array([p[1] for p in rows], pa.string()),
                "class": pa.array([p[2] for p in rows], pa.string()),
                "wkb": pa.array([p[3] for p in rows], pa.binary()),
            }
        )
        out = con.sql(BLD_DISSOLVE_SQL).to_arrow_table()
        _bld_mem.append([cov, out, key])
        while len(_bld_mem) > BLD_KEEP:
            _bld_mem.pop(0)
        return out, key, cov, f"{n_tiles} tiles"

    # 'overlap' for buildings, the fire-risk rule: a footprint smaller than any cell
    # contains no cell centre, and buildings are disjoint islands with no partition
    # to violate.
    BLD_FILL_SQL = """
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
    _bfill_memo = {}
    BFILL_KEEP = 12

    def _polyfill_buildings(blds_tbl, key, res):
        ck = (key, int(res))
        if ck in _bfill_memo:
            return _bfill_memo[ck]
        blds = blds_tbl  # noqa: F841 - read by the replacement scan
        out = con.sql(BLD_FILL_SQL, params=[int(res)]).to_arrow_table()
        _bfill_memo[ck] = out
        while len(_bfill_memo) > BFILL_KEEP:
            _bfill_memo.pop(next(iter(_bfill_memo)))
        return out

    # THE BUILDING JOIN. Building cells at res 11 (overlap), zone cells at res 12
    # (center), parented up to 11, MAX(zc) so the WORST zone a footprint touches is
    # the one it wears. No zone cell under any of a building's cells means -1:
    # outside every drawn zone, which on this map is a colour, not an absence.
    BLD_JOIN_SQL = """
        WITH zp AS (
            SELECT h3_cell_to_parent(hex, ?) AS phex, MAX(zc) AS zc
            FROM zcells
            GROUP BY 1
        ), bz AS (
            SELECT b.id, MAX(z.zc) AS zc
            FROM bcells b
            LEFT JOIN zp z ON b.hex = z.phex
            GROUP BY b.id
        )
        SELECT m.id, m.name, m.class, COALESCE(bz.zc, -1) AS zc, m.wkb
        FROM blds m
        JOIN bz USING (id)
    """

    def join_buildings(blds_tbl, bkey, ztbl, zkey):
        """Every building with the worst flood-zone class its cells touch."""
        bcells_t = _polyfill_buildings(blds_tbl, bkey, BLD_RES)
        zcells_t = zone_cells(ztbl, zkey, ZBLD_RES)
        bcells, zcells, blds = bcells_t, zcells_t, blds_tbl  # noqa: F841
        return con.sql(BLD_JOIN_SQL, params=[BLD_RES]).to_arrow_table()

    return (
        fetch_buildings,
        fetch_divisions,
        fetch_zones,
        join_buildings,
        join_divisions,
    )


@app.cell
def _(
    BitmapTileLayer,
    CartoBasemap,
    Controls,
    FullscreenControl,
    GeocoderControl,
    HOLD,
    Map,
    MaplibreBasemap,
    NavigationControl,
    PolygonLayer,
    ScaleControl,
    Status,
    asyncio,
    json,
    seed_buildings,
    seed_divisions,
    seed_zones,
    urllib,
):
    # THE MAP CELL, AND WHY IT DEPENDS ON ALMOST NOTHING. Destroying a lonboard Map
    # terminates deck's earcut worker pool, which is MODULE-LEVEL in the page: after
    # that every polygon layer on the page fails to initialize ("Cannot schedule pool
    # tasks after terminate()"), one throw kills the whole layer batch, and the map is
    # bare basemap until the browser page reloads. So this cell must never re-run on
    # an ordinary edit. It depends only on the imports, the widget classes, the seeds
    # and HOLD (none of which change during normal work); everything editable (the
    # palette, the constants, the machinery, the draw logic) lives downstream in the
    # WIRING cell, which re-hooks onto these surviving widgets.
    #
    # EDITING THIS CELL ITSELF still tears the deck down: restart the kernel AND
    # reload the browser page afterwards.
    #
    # HOME and the canvas-size seed live here rather than in the constants cell for
    # the same reason: a constants edit must not reach this cell.
    VIEW_W, VIEW_H = 1400, 620
    # Opens over New Orleans in the county band: parishes, the full zone alphabet
    # (V/VE on the coast, AE basins, shaded X), and enough dry ground for contrast.
    HOME = {"longitude": -90.05, "latitude": 29.85, "zoom": 9.2}

    status = Status(value="<b>loading…</b>")
    controls = Controls()

    _zseed = seed_zones()
    zones = PolygonLayer(
        table=_zseed,
        get_fill_color=_zseed["color"],
        filled=True,
        stroked=False,
        opacity=1.0,
        pickable=True,
    )

    _dseed = seed_divisions()
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
        pickable=False,
        visible=False,
    )

    _bseed = seed_buildings()
    buildings = PolygonLayer(
        table=_bseed,
        get_fill_color=_bseed["color"],
        filled=True,
        stroked=False,
        opacity=1.0,
        pickable=True,
        visible=False,
    )

    # Place labels OVER the data layers; @2x with tile_size 512 or retina type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
    )

    # THE GEOCODER. lonboard 0.16's GeocoderControl is a search box whose queries come
    # back to the kernel; the handler answers with GeoJSON and the frontend flies the
    # camera there, which lands in _on_camera like any other move, so a search result
    # triggers the same fetch-and-join a pan would. Photon (komoot's public geocoder)
    # already speaks GeoJSON Point features, so no geopy: one urllib GET on a thread
    # (stdlib has no async HTTP and this notebook is not adding aiohttp for a search
    # box). Results are BIASED toward the current camera via Photon's lon/lat params,
    # so "springfield" finds the one nearest the map, not the biggest one on Earth.
    def _photon_feature(f):
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        name = p.get("name") or ""
        place = ", ".join(
            str(v)
            for v in (p.get("name"), p.get("city"), p.get("state"), p.get("country"))
            if v
        )
        out = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": (lon, lat)},
            "text": name or place,
            "place_name": place or name,
            "place_type": [p.get("type") or "place"],
            "center": (lon, lat),
        }
        # Photon's extent is [minLon, maxLat, maxLon, minLat]; lonboard's bbox is
        # (minx, miny, maxx, maxy). Getting this wrong flies to an inside-out box.
        ext = p.get("extent")
        if ext and len(ext) == 4:
            out["bbox"] = (ext[0], ext[3], ext[2], ext[1])
        return out

    async def _photon(query):
        vs = HOLD["vs"]
        params = {"q": query, "limit": 5, "lang": "en"}
        if vs is not None:
            params["lon"] = round(vs.longitude, 4)
            params["lat"] = round(vs.latitude, 4)
        url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)

        def hit():
            req = urllib.request.Request(
                url, headers={"User-Agent": "x-sql-marimo flood notebook"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)

        data = await asyncio.to_thread(hit)
        return {
            "type": "FeatureCollection",
            "features": [_photon_feature(f) for f in data.get("features", [])],
        }

    geocoder = GeocoderControl(client=_photon, position="top-left")

    deck = Map(
        [zones, divisions, buildings, labels],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state=HOME,
        height=VIEW_H,
        show_tooltip=True,
        # Passing `controls` REPLACES the default tuple, so the defaults
        # (fullscreen, navigation, scale) are restated alongside the geocoder.
        controls=[
            FullscreenControl(),
            NavigationControl(),
            ScaleControl(),
            geocoder,
        ],
    )

    # The HOLD state tied to the WIDGET's lifetime rather than the wiring's: the
    # canvas seed (the ruler overwrites it), the camera, and the mount flag.
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["vs"] = None
    HOLD["fresh"] = True
    return HOME, buildings, controls, deck, divisions, status, zones


@app.cell
def _(
    BLD_ZOOM,
    DIVISION_LABEL,
    HOLD,
    SETTLE,
    asyncio,
    buildings,
    buildings_to_layer,
    controls,
    deck,
    division_for_zoom,
    divisions,
    divisions_pair,
    fetch_buildings,
    fetch_divisions,
    fetch_zones,
    infer_rows_per_chunk,
    join_buildings,
    join_divisions,
    math,
    res_for_zoom,
    status,
    zones,
    zones_to_layer,
):
    # THE WIRING CELL: everything editable about how the map behaves. Re-runs on any
    # edit to the constants, palette, builders or machinery, cancels the old work,
    # unhooks the old observers, and re-hooks onto the SURVIVING deck from the map
    # cell above; the widget is never destroyed, so deck's shared earcut pool stays
    # alive and the camera stays where the user left it. The screen-state keys are
    # reset because the machinery's caches were just rebuilt; wh, vs and fresh belong
    # to the map cell and are left alone.
    if HOLD["task"] is not None:
        HOLD["task"].cancel()
    HOLD["task"] = None
    HOLD["busy"], HOLD["pending"] = False, None
    HOLD["regime"], HOLD["zbox"], HOLD["ztz"] = None, None, None
    HOLD["res"], HOLD["div"], HOLD["divbox"] = None, None, None
    HOLD["divpair"], HOLD["bldbox"] = None, None
    HOLD["head"], HOLD["tail"] = "", ""

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N] from the MEASURED canvas (HOLD['wh'])."""
        vw, vh = HOLD["wh"]
        span = 360.0 * vw / (512 * 2**vs.zoom)
        lat_span = span * (vh / vw) * math.cos(math.radians(vs.latitude))
        return (
            max(-180.0, vs.longitude - span / 2),
            max(-85.0, vs.latitude - lat_span / 2),
            min(180.0, vs.longitude + span / 2),
            min(85.0, vs.latitude + lat_span / 2),
        )

    def _pad(b):
        w, s, e, n = b
        cx, cy = (w + e) / 2, (s + n) / 2
        hw, hh = (e - w) / 2 * 1.25, (n - s) / 2 * 1.25
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
        """The echo check: ignore the event the map emits for a view we set."""
        return (
            a is not None
            and b is not None
            and round(a.longitude, 6) == round(b.longitude, 6)
            and round(a.latitude, 6) == round(b.latitude, 6)
            and round(a.zoom, 4) == round(b.zoom, 4)
        )

    def set_status(vs):
        status.value = (
            f"{HOLD['head']}{HOLD['tail']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

    def put_zones(tbl):
        zones._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with zones.hold_sync():
            zones.table = tbl
            zones.get_fill_color = tbl["color"]
            zones.visible = controls.show_zones

    def put_divisions(pair):
        tbl = pair[0] if controls.division_fill else pair[1]
        divisions._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with divisions.hold_sync():
            divisions.table = tbl
            divisions.get_fill_color = tbl["color"]
            divisions.get_line_color = tbl["line"]
            divisions.visible = controls.show_divisions
            # Picking ignores alpha: a zero-alpha fill still swallows hovers meant
            # for the zones underneath, so stroke-only divisions must not pick.
            divisions.pickable = bool(controls.division_fill)
        HOLD["divpair"] = pair

    def put_buildings(tbl):
        buildings._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        with buildings.hold_sync():
            buildings.table = tbl
            buildings.get_fill_color = tbl["color"]
            buildings.visible = True

    def repaint():
        """Re-send every layer's full widget state to the browser.

        Guards the MOUNT RACE: pushes made before the browser has mounted the
        widgets can be lost, so the first ruler report or camera event after a
        fresh session (both fire only once the widgets exist on the page) re-sends
        everything. send_state() rather than re-assignment, because traitlets
        suppresses notifications for unchanged values. The deck itself is not
        re-sent, so a repaint can never snap the camera. (The blank-map-on-re-run
        defect turned out to be a different mechanism entirely, deck's module-level
        earcut pool dying with the old widget, and is fixed by the map/wiring cell
        split above; this stays for the mount race alone.)
        """
        for lyr in (zones, divisions, buildings):
            lyr.send_state()

    def _on_controls(change):
        name = change["name"]
        if name == "show_zones":
            zones.visible = bool(change["new"])
        elif name == "show_divisions":
            divisions.visible = (
                bool(change["new"])
                and HOLD["divpair"] is not None
                and HOLD["regime"] == "div"
            )
        elif name == "division_fill":
            if HOLD["divpair"] is not None:
                put_divisions(HOLD["divpair"])

    # OLD OBSERVERS COME OFF FIRST. The widgets survive this cell's re-run, so a
    # plain observe() would stack a new handler beside the old one and every event
    # would fire both, the old one into dead closures. HOLD carries the refs.
    _CTL_NAMES = ["show_zones", "show_divisions", "division_fill"]
    if HOLD.get("h_ctl") is not None:
        controls.unobserve(HOLD["h_ctl"], names=_CTL_NAMES)
    controls.observe(_on_controls, names=_CTL_NAMES)
    HOLD["h_ctl"] = _on_controls

    def _on_wh(change):
        """The canvas changed size: fullscreen, a resize, a layout shift."""
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
            set_status(vs)
        if HOLD["fresh"]:
            HOLD["fresh"] = False
            repaint()
        if abs(wh[0] - old[0]) < 25 and abs(wh[1] - old[1]) < 25:
            return
        if vs is None:
            return
        if HOLD["busy"]:
            HOLD["pending"] = vs
        elif not _instant(vs):
            HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_wh") is not None:
        status.unobserve(HOLD["h_wh"], names="view_wh")
    status.observe(_on_wh, names="view_wh")
    HOLD["h_wh"] = _on_wh

    def _instant(vs):
        """Everything answerable without a read, synchronously in the comm handler.

        The zones must cover the view AND have been fetched at the tile zoom this
        view wants (a zoom-in served from three levels up would draw generalised
        blobs); on top of that the current regime's own coverage must hold.
        """
        seen = view_to_bbox(vs)
        if not _covers(HOLD["zbox"], seen):
            return False
        span = max(seen[2] - seen[0], 1e-9)
        want_tz = int(math.log2(max(4.0 * 360.0 / (span * 1.35), 1.0)))
        if HOLD["ztz"] is not None and min(want_tz, 13) > HOLD["ztz"]:
            return False
        if vs.zoom >= BLD_ZOOM:
            ok = HOLD["regime"] == "bld" and _covers(HOLD["bldbox"], seen)
        else:
            sub = division_for_zoom(vs.zoom)
            if sub is None:
                ok = HOLD["regime"] is None
            else:
                ok = (
                    HOLD["regime"] == "div"
                    and sub == HOLD["div"]
                    and HOLD["res"] == res_for_zoom(vs.zoom)
                    and _covers(HOLD["divbox"], seen)
                )
        if ok:
            set_status(vs)
            return True
        return False

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view. The last answer stays up
        until there is a new one: nothing is cleared before the swap."""
        if not force and _instant(vs):
            return

        want = _pad(view_to_bbox(vs))
        HOLD["head"] = "<b>reading…</b>"
        set_status(vs)

        ztbl, zkey, zcov, znote = await fetch_zones(want)
        if ztbl is not None:
            put_zones(zones_to_layer(ztbl))
            HOLD["head"] = f"<b>{ztbl.num_rows:,} flood polygons</b> · {znote}"
        else:
            zones.visible = False
            HOLD["head"] = "<b>no mapped flood zones here</b>"
        HOLD["zbox"], HOLD["ztz"] = zcov, zkey[1]
        set_status(vs)
        if HOLD["pending"] is not None:
            return  # the camera has already moved; this view is gone

        if vs.zoom >= BLD_ZOOM:
            divisions.visible = False
            HOLD["div"], HOLD["divpair"], HOLD["divbox"] = None, None, None
            btbl, bkey, bcov, bnote = await fetch_buildings(want)
            if btbl is None:
                buildings.visible = False
                HOLD["regime"], HOLD["bldbox"] = "bld", bcov
                HOLD["tail"] = f" · buildings: {bnote}"
                set_status(vs)
                return
            out = join_buildings(btbl, bkey, ztbl, zkey)
            put_buildings(buildings_to_layer(out))
            HOLD["regime"], HOLD["bldbox"] = "bld", bcov
            HOLD["tail"] = f" · {out.num_rows:,} buildings ({bnote})"
            set_status(vs)
            return

        buildings.visible = False
        HOLD["bldbox"] = None
        sub = division_for_zoom(vs.zoom)
        if sub is None:
            divisions.visible = False
            HOLD["regime"], HOLD["div"], HOLD["divpair"] = None, None, None
            HOLD["tail"] = " · zoom in for boundaries"
            set_status(vs)
            return

        res = res_for_zoom(vs.zoom)
        dtbl, dkey, dcov = await fetch_divisions(sub, want)
        if dtbl is None:
            divisions.visible = False
            HOLD["regime"], HOLD["div"], HOLD["divbox"] = "div", sub, dcov
            HOLD["res"], HOLD["divpair"] = res, None
            HOLD["tail"] = f" · no {DIVISION_LABEL[sub]} here"
            set_status(vs)
            return
        out, n_small = join_divisions(dtbl, dkey, ztbl, zkey, res)
        HOLD["regime"], HOLD["div"], HOLD["divbox"] = "div", sub, dcov
        HOLD["res"] = res
        if out is None:
            divisions.visible = False
            HOLD["divpair"] = None
            HOLD["tail"] = f" · {DIVISION_LABEL[sub]} too small to measure at res {res}"
            set_status(vs)
            return
        put_divisions(divisions_pair(out))
        HOLD["tail"] = f" · {out.num_rows:,} {DIVISION_LABEL[sub]} · res {res}" + (
            f" · <b style='color:#E69F00'>{n_small} too small to measure</b>"
            if n_small
            else ""
        )
        set_status(vs)

    async def refresh(vs, force=False):
        """Read what the camera is looking at, once it has stopped moving.

        SETTLE debounces; coalescing collapses whatever piled up during a read to
        the NEWEST view. No threads and no timers: the debounce is an await on the
        kernel's own loop, so the map keeps rendering.
        """
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
            HOLD["tail"] = ""
            status.value = HOLD["head"]
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _spawn(coro):
        """Run a coroutine on the kernel's loop, keeping a strong task reference:
        asyncio holds only a weak one and a bare create_task can be collected."""
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _on_camera(change):
        vs = change["new"]
        if HOLD["fresh"]:
            # The browser is provably mounted now; see repaint. Runs even for an
            # echo of the view we set, which is exactly what the first event after
            # a rebuild is.
            HOLD["fresh"] = False
            repaint()
        if _same_view(vs, HOLD["vs"]):
            return
        HOLD["vs"] = vs
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        if _instant(vs):
            return
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        deck.unobserve(HOLD["h_cam"], names="view_state")
    deck.observe(_on_camera, names="view_state")
    HOLD["h_cam"] = _on_camera
    return (refresh,)


@app.cell
async def _(HOLD, HOME, asyncio, refresh):
    HOLD["loop"] = asyncio.get_running_loop()

    # The opening draw, forced: nothing to debounce yet. On a WIRING re-run the
    # camera survives with the deck, so the redraw targets wherever the user left
    # it, not HOME; HOME only seeds a fresh session (and headless runs).
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]

    await refresh(HOLD["vs"] or _VS(), force=True)
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
                "FEMA National Flood Hazard Layer (effective FIRMs, US public "
                "domain) via source.coop `cboettig/hazard`; boundaries and building "
                "footprints: Overture Maps. Zoomed out, each boundary's stroke (and "
                "fill, if toggled) carries the share of its ground inside the "
                "1%-annual-chance floodplain, measured on H3 cells with zones "
                "sampled one resolution finer. Past zoom 13 the map switches to "
                "individual buildings, each coloured by the WORST zone its res 11 "
                "cells touch; slate means outside every drawn zone. X minimal "
                "hazard, open water and unstudied areas are not drawn. This is a "
                "screening view of generalised map data, not a flood determination "
                "for any property."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
