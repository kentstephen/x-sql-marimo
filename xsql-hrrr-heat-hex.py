# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.2",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
# ]
# ///
"""HRRR heat, per H3 res 6 cell over CONUS, as a film with a memory.

The counties film (xsql-hrrr-counties.py) asks how hot each county is, hour by hour.
This one asks how the heat SITS: dynamical.org's HRRR analysis (3 km, hourly, CC-BY
4.0) is read straight from its Zarr with xarray-sql, 2 m temperature and relative
humidity folded to H3 res 6 (~4 pixels per cell, 210,724 cells over CONUS land)
and turned into NWS heat index per cell per hour. The whole film crosses to the browser once, and the browser runs an
ACCUMULATOR over it: each hour a cell's heat load rises by how far the heat index sits
above a threshold and decays with a half-life; nights that do not cool leave the load
up, so a place with three hot days and warm nights ends bright while a place with the
same afternoon highs and cool nights stays dim. That memory (half-life, threshold,
and, when read, a rain flush and a wind vent) is four sliders in the HUD, recomputed
in the browser in ~0.2 s, and the map switches between the instantaneous heat index
and the accumulated load. Same widget skeleton as the counties film (deck.gl from
esm.sh, browser-owned clock, minimal HUD on the map, the HUD's window loader as the
one thing that reaches back), with an H3HexagonLayer in place of the GeoArrow
polygons and geometric picking replaced by h3-js latLngToCell.

WHAT IT COSTS. The read is the floor and it is the wire, not the code: the analysis
store's chunks are 2,160 hours deep, so any window inside the current 90 days fetches
the same bytes per variable (~0.5 GB today, more as the quarter fills), and this link
runs ~21 MB/s single- or multi-stream (measured; async concurrency changes nothing).
Two levers exist and both are used: (1) only the 523 of 960 store columns that touch
CONUS land are read, via xarray-sql's partition pruning on a y/x block predicate
(2 variables: 44.7 s -> 28.5 s measured), and (2) fewer variables. Heat index needs
two (T, RH): ~28 s. Rain (precipitation rate, the accumulator's flush) is one more
(~14 s), wind (u, v; the vent) two more; both are flags in the constants cell, off
by default. Near the data (molab) the wire stops being the floor and decode is;
zarrista's Rust decode would be the lever there (see docs/hrrr-counties-notes.md).

MEMORY, AND WHY THE FOLD CELL SETS DATAFUSION KNOBS. The read is the same at any
res (every land pixel is fetched either way); what scales is the number of ANSWERS,
hours x cells, which DataFusion holds as a hash aggregate until the last block lands
and the browser holds as the film. Res 6 is 35M answers for a week: 17 GB peak with
DataFusion's default plan (the join re-hashes the cube by pixel and every partition's
partial aggregate holds its own copy of nearly every group), 9.5 GB once the join
broadcasts the pixel lookup instead (CollectLeft, two thresholds), ~5 GB under a 3 GB
spill pool, all at the same ~28 s; both are set in the fold cell. Res 5 (30,124
cells, 5M answers, well under a GB, 5 MB per field) is the one-constant retreat, the
counties film with this accumulator the fallback after that. Bytes to the browser:
uint8 fields (heat index in 0.5 degC steps; wind/rain packed in one byte), 35 MB per
field for a week at res 6, the load computed and held browser-side. Hourly frames
only, up to 14 days.

DuckDB does geometry (county dissolve, polyfill; counties are the CONUS land mask
and give a clicked cell its county name), DataFusion does the fold, numpy the heat
index. con.register throughout (marimo mangles underscore locals, so DuckDB's
replacement scan cannot see them). Design notes in docs/hrrr-heat-hex-notes.md.
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

    import anywidget
    import duckdb
    import marimo as mo
    import numpy as np
    import obstore
    import pyarrow as pa
    import pyarrow.parquet as pq
    import traitlets
    import xarray as xr
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext

    return (
        S3Store,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        gzip,
        json,
        math,
        mo,
        np,
        obstore,
        pa,
        pq,
        struct,
        traitlets,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/x-sql-marimo/blob/main/xsql-hrrr-heat-hex.py)

    # HRRR heat with a memory

    Hourly heat index for the lower 48 on H3 hexagons, played as a film, with a
    switch to a second field computed live in the browser: **heat load**, an
    accumulator that rises while the heat index sits above a threshold and decays
    with a half-life you set. It is the same weather, but the map remembers: a place
    whose nights do not cool stays bright after the sun goes down, and a place with
    the same afternoon highs and cool nights goes dark. Set the half-life, the
    threshold, and (if read) how much rain flushes and wind vents the load; press
    play.

    **Where the numbers come from.** [dynamical.org](https://dynamical.org/)'s Zarr
    build of NOAA's HRRR analysis (3 km, hourly, CONUS, CC-BY 4.0), read anonymously
    from the AWS Open Data bucket `s3://dynamical-noaa-hrrr`: 2 m temperature and
    relative humidity, optionally precipitation rate and 10 m wind. Counties are
    Overture Maps divisions from Overture's PMTiles; here they are the land mask (and
    the name a clicked cell reports), not the unit of the map. Nothing is precomputed.

    **What the notebook does with them.** The store is queried with
    [xarray-sql](https://github.com/alxmrs/xarray-sql), which lets DataFusion treat the
    Zarr cube as a table and prune the store columns that never touch CONUS land.
    Every pixel is labelled with its H3 res 6 cell straight from the store's own
    latitude/longitude; one `GROUP BY` averages the ~4 pixels in each cell per hour;
    numpy turns temperature and humidity into the NWS heat index. The film crosses to
    the browser once as bytes; the accumulator, the clock, the picking and the chart
    all run there, and the only thing that reaches back to Python is the load button.

    **What it costs.** The read: the archive is chunked for time series (each 45 × 45
    pixel column is 2,160 hours = 90 days deep), so a window fetches every filled hour
    of the chunk it falls in, whatever its length, and the link, not the code, sets
    the pace. A week in the current, part-filled chunk is about thirty seconds for
    two variables on a ~200 Mbit link; the opening window (the late-July 2026 heat
    dome) sits in a full chunk and is about two minutes; the panel states the estimate
    for whatever dates you pick. Rain and wind are off by default because each
    variable is another chunk read; flip `READ_RAIN` / `READ_WIND` in the constants
    cell to have them.
    """)
    return


@app.cell
def _():
    # ------------------------------------------------------------------ the weather
    # dynamical.org's hourly HRRR analysis, icechunk v2 in the AWS Open Data bucket
    # (not on source.coop), time-optimised chunks (2160 h x 45 x 45 px, sharded
    # 2160 x 540 x 450). The 48 h forecast SOURCE of the counties film is not carried:
    # a film with a memory wants a long past, not 48 h ahead.
    ANALYSIS_BUCKET = "dynamical-noaa-hrrr"
    ANALYSIS_PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
    # Heat index needs temperature_2m and relative_humidity_2m (always read). Each
    # further variable is another whole 90-day chunk layer per column: ~14 s on a
    # ~21 MB/s link after the land pruning, and it grows through the quarter as the
    # current chunk fills. Rain is the accumulator's flush, wind its vent.
    READ_RAIN = False  # precipitation_surface (kg m-2 s-1, an hourly-mean RATE -> mm/h)
    READ_WIND = False  # wind_u_10m + wind_v_10m -> speed (m/s); two variables
    # Opening window: an int is the last DAYS UTC days ending at the newest hour; a
    # ("YYYY-MM-DD", "YYYY-MM-DD") tuple is a fixed window. THE COST DEPENDS ON WHICH
    # STORE CHUNK THE DATES FALL IN, not on how many days: chunks are 90 days deep and
    # start on 2026-05-01 and 2026-07-30 (every 2,160 h from 2014-10-01), and a
    # window fetches every filled hour of the chunk it touches. Measured at res 5,
    # two variables, land-pruned: a week in the current chunk (447 h filled on
    # 2026-08-17) 27 s; a week in the full July chunk 130 s. Summer 2026's heat
    # domes (per Wikipedia and NOAA: East Jun 28-Jul 5, West/central from Jul 6 with
    # all-time records on Jul 12, Plains Jul 24-28; July 2026 the hottest US month
    # on record) all sit in the full chunk, so as presets they cost ~2 min:
    #   DAYS = ("2026-07-06", "2026-07-12")   # West dome, Salt Lake City 109 F Jul 12
    #   DAYS = ("2026-06-29", "2026-07-05")   # East dome, Atlantic City 106 F Jul 4
    #   DAYS = 7                              # the last week, ~30 s while the chunk is young
    # The opening window is the late-July dome: in a block-sampled scan of the store
    # (one 45x45 column in every third, Jun 15 to Aug 17) it is the summer's CONUS-wide
    # peak, the largest share of land pixels over 35 degC (0.22-0.24 on Jul 25-27
    # against 0.13 for the West dome's Jul 8-12) and the highest CONUS-mean day (Jul 27).
    DAYS = ("2026-07-23", "2026-07-29")   # Plains dome, Rapid City 112 F Jul 26; ~2 min
    HOURLY_MAX_DAYS = 14  # 336 frames x 210k cells = 71 MB per field across the bridge

    # ------------------------------------------------------------------ the fold
    # Res 6: 4.2 HRRR pixels per cell (measured), 210,724 cells over CONUS land, all
    # of them catch a pixel; 35M (hour, cell) answers for a week, which is what sets
    # the kernel's memory (~5 GB peak with the fold cell's DataFusion knobs; 17 GB
    # without them) and the bytes to the browser (35 MB per field). The read is
    # identical at any res (same pixels). Res 5 (30,124 cells, 5M answers, well
    # under a GB, 5 MB per field, ~250 km2 hexes) is the one-constant retreat, and
    # was flown; the counties film with this accumulator is the fallback after that.
    # Res 7 is the pixel itself (a relabel) and 1.47M cells: no film fits.
    RES = 6

    # ------------------------------------------------------------------ the land mask
    # Overture's PMTiles build of the pinned release, same object, box, zoom and
    # CONUS filter as the counties film; the counties are the mask (a cell is CONUS
    # land if its centre falls in a county) and the click readout's name.
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    COUNTY_Z = 8
    BOX = (-124.8, 24.4, -66.9, 49.5)
    NOT_CONUS = {"AK", "HI"}
    import tempfile as _tempfile

    # Same cache file as the counties film (tmp, not a project .cache; None disables).
    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"

    # ------------------------------------------------------------------ the film
    # Heat index: diverging blue <-> yellow/orange (protan-safe: no red leg, no
    # red-vs-green pair), pale at the pivot (the film's median), span to the wider of
    # p2/p98 unless PIVOT/SPAN pin it. Heat load: one-signed, so a luminance ramp,
    # dark to bright (matplotlib inferno's stops; on the colorblind-safe shortlist),
    # scaled in the browser to the p98 of the load it just computed.
    INDEX_STOPS = ["#08306b", "#2f79b5", "#9ecae1", "#f2f0e6", "#fee391", "#fdb034", "#d94801"]
    LOAD_STOPS = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]
    PIVOT = None  # degC heat index, or None for the film median
    SPAN = None  # degC either side, or None for the p2/p98 rule
    # Accumulator defaults (all live sliders in the HUD): the load rises by the heat
    # index's excess over THRESHOLD (27 degC is where the NWS caution band starts)
    # and decays with HALF_LIFE hours; RAIN_FLUSH and WIND_VENT are 0..1 weights,
    # meaningful only when the field was read.
    THRESHOLD = 27.0
    HALF_LIFE = 12.0
    RAIN_FLUSH = 0.5
    WIND_VENT = 0.3
    FPS = 8
    MAP_HEIGHT = 620
    return (
        ANALYSIS_BUCKET,
        ANALYSIS_PREFIX,
        BOX,
        CACHE_DIR,
        COUNTY_Z,
        DAYS,
        FPS,
        HALF_LIFE,
        HOURLY_MAX_DAYS,
        INDEX_STOPS,
        LOAD_STOPS,
        MAP_HEIGHT,
        NOT_CONUS,
        OVERTURE_RELEASE,
        PIVOT,
        PM_BUCKET,
        PM_PATH,
        RAIN_FLUSH,
        READ_RAIN,
        READ_WIND,
        RES,
        SPAN,
        THRESHOLD,
        WIND_VENT,
    )


@app.cell
def _(duckdb):
    # DuckDB does the geometry (tile-seam dissolve, polyfill); DataFusion does the fold.
    con = duckdb.connect()
    con.sql("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class HexFilm(anywidget.AnyWidget):
        """deck.gl H3HexagonLayer, browser-side clock AND accumulator, minimal HUD.

        Kernel -> browser: `cells` (uint64 LE H3 ids, N, sorted), `cidx` (uint16 county
        index per cell), `names` (JSON list of "County, ST"), `frames` (uint8 F x N,
        frame-major: heat index in 0.5 degC steps offset -40, 255 = no data), `wx`
        (uint8 F x N or empty: wind m/s rounded in the high nibble, rain in 0.5 mm/h
        steps in the low nibble) and `config` (JSON: labels, index ramp lo/mid/hi and
        stops, load stops, accumulator defaults, has_rain/has_wind, fps, height,
        title, subtitle, and `win`, the store's day span, the served window and the
        limit). Browser -> kernel: `window` only, the HUD's date range + load button,
        as one Unicode JSON trait, exactly as in the counties film.

        The ACCUMULATOR runs here, over the frame matrix, whenever a slider moves:
        L[f] = a * L[f-1] + (1 - a) * max(0, HI[f] - threshold), a = 2^(-1/half_life),
        so L is "sustained excess above the threshold" in degC and comparable to the
        index; rain multiplies L by (1 - flush * min(1, mm / 2.5)) that hour, wind
        scales the excess by max(0, 1 - vent * ws / 10). 35M multiply-adds, ~0.2 s,
        stored as uint8 (0.1 degC steps) for paint; the picked cell's line is
        recomputed exactly. The map paints either field; the ramp for the load is
        0 .. p98 of the load just computed.

        Rendering: deck's H3HexagonLayer with `highPrecision: true` (one polygon per
        cell, which deck would pick on its own at res <= 5; the instanced path shares
        one hexagon shape across the layer and over 60 degrees of longitude leaves
        visible gaps), colours via an accessor
        keyed on [frame, field, gen] through updateTriggers, so a frame step re-uploads
        one attribute. Picking is h3-js `latLngToCell` on the unprojected click (deck's
        GPU picking returned null inside marimo's shadow DOM on the counties flights).

        esm.sh pins: every deck package at 9.3.10 with `?deps` so all resolve to ONE
        core (see the counties film's docstring and docs/hrrr-counties-notes.md for
        the crawl); h3-js pinned to 4.5.0, the version the pinned geo-layers resolves
        its ^4.4.0 to (checked 2026-08-17), so the picker and the layer share it.
        """

        _esm = r"""
        import {Deck} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .hf { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .hf * { box-sizing: border-box; }
          .hf .hf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .hf .hf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .hf .hf-map:fullscreen { height: 100vh !important; width: 100vw; }
          .hf .hf-hud { position: absolute; z-index: 5; }
          .hf .hf-hud.hf-tl { top: .6rem; left: .6rem; width: 22rem; max-width: calc(100% - 1.2rem); }
          .hf .hf-hud.hf-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .hf .hf-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .hf .hf-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }
          .hf .hf-ttl { font-weight: 600; }
          .hf .hf-sub { color: var(--dim); display: block; margin-top: .1rem; }
          .hf .hf-fields { display: flex; gap: .3rem; margin-top: .5rem; }
          .hf .hf-fields button { flex: 1; }
          .hf .hf-fields button.hf-on { background: #3a3f2a; border-color: var(--accent); color: #fff; }
          .hf .hf-legend { display: flex; align-items: center; gap: .45rem; margin-top: .45rem; }
          .hf .hf-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .hf .hf-row { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin-top: .4rem; }
          .hf .hf-row .hf-v { font-size: 16px; }
          .hf .hf-row .hf-k { color: var(--dim); }
          .hf .hf-cell { margin-top: .35rem; display: none; }
          .hf.hf-picked .hf-cell { display: block; }
          .hf .hf-chart { display: block; width: 100%; height: 96px; margin-top: .3rem; cursor: crosshair; }
          .hf.hf-collapsed .hf-body { display: none; }
          .hf .hf-toggle, .hf .hf-clear { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .hf .hf-toggle:hover, .hf .hf-clear:hover { color: var(--ink); }
          .hf .hf-params { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .hf .hf-p { display: grid; grid-template-columns: 6.2rem 1fr 3.4rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .hf .hf-p label { color: var(--dim); }
          .hf .hf-p.hf-off { display: none; }
          .hf .hf-transport { display: flex; align-items: center; gap: .55rem; }
          .hf .hf-stamp { font-size: 15px; min-width: 11.5rem; }
          .hf .hf-stamp small { display: block; font-size: 10px; color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
          .hf .hf-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .hf .hf-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .hf .hf-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .hf input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .hf button.hf-b, .hf select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .hf button.hf-b:hover, .hf select:hover { background: #2b323b; }
          .hf button:focus-visible, .hf select:focus-visible, .hf input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .hf .hf-dim { color: var(--dim); }
          .hf .hf-win { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem .4rem; margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .hf .hf-win input[type=date] { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .15rem .3rem; font: inherit; color-scheme: dark; min-width: 0; }
          .hf .hf-win .hf-note { flex-basis: 100%; }
          .hf .hf-win .hf-note.hf-bad { color: var(--accent); }
          .hf .hf-win button.hf-load:disabled { opacity: .55; cursor: default; }
          .hf .hf-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; }
          @media (max-width: 720px) { .hf .hf-stamp { min-width: 0; } .hf .hf-hud.hf-tl { width: calc(100% - 1.2rem); } }
        `;

        function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
        function buildLut(stops) {
          const rgb = stops.map(hexToRgb), lut = new Uint8Array(256 * 3);
          for (let i = 0; i < 256; i++) {
            const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
            for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
          }
          return lut;
        }
        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset ?? 0, v.byteLength);
          return null;
        }
        const HI_OF = q => q / 2 - 40;  // uint8 -> degC

        function render({model, el}) {
          el.innerHTML = "";
          const root = document.createElement("div"); root.className = "hf";
          root.innerHTML = `<style>${CSS}</style>
            <div class="hf-map">
              <div class="hf-hud hf-tl"><div class="hf-card hf-panel">
                <div class="hf-head"><span><span class="hf-ttl"></span><span class="hf-sub"></span></span><button class="hf-toggle" title="hide / show (H)">hide</button></div>
                <div class="hf-body">
                  <div class="hf-fields"><button class="hf-b hf-fi hf-on" data-field="index" title="NWS heat index this hour (I)">heat index</button><button class="hf-b hf-fi" data-field="load" title="accumulated heat load (L)">heat load</button></div>
                  <div class="hf-legend"><span class="hf-num hf-lo"></span><div class="hf-grad"></div><span class="hf-num hf-hi"></span></div>
                  <div class="hf-row"><span class="hf-k hf-meank">CONUS mean</span><span class="hf-num hf-v hf-mean">–</span></div>
                  <div class="hf-cell">
                    <div class="hf-row"><span class="hf-k hf-cname">–</span><span><span class="hf-num hf-v hf-cval">–</span> <button class="hf-clear" title="clear">×</button></span></div>
                    <canvas class="hf-chart" height="96"></canvas>
                  </div>
                  <div class="hf-params">
                    <div class="hf-p"><label>half-life</label><input type="range" class="hf-half" min="1" max="72" step="1"><span class="hf-num hf-halfv"></span></div>
                    <div class="hf-p"><label>threshold</label><input type="range" class="hf-thr" min="15" max="40" step="0.5"><span class="hf-num hf-thrv"></span></div>
                    <div class="hf-p hf-p-rain"><label>rain flush</label><input type="range" class="hf-rain" min="0" max="1" step="0.05"><span class="hf-num hf-rainv"></span></div>
                    <div class="hf-p hf-p-wind"><label>wind vent</label><input type="range" class="hf-wind" min="0" max="1" step="0.05"><span class="hf-num hf-windv"></span></div>
                    <div class="hf-dim hf-pnote"></div>
                  </div>
                  <div class="hf-win">
                    <input type="date" class="hf-d0" title="first UTC day, inclusive" aria-label="window start (UTC day)"><span class="hf-dim">to</span><input type="date" class="hf-d1" title="last UTC day, inclusive" aria-label="window end (UTC day)">
                    <button class="hf-b hf-load" title="fold this window (the kernel refetches; the read estimate is beside the dates)">load</button>
                    <span class="hf-dim hf-note"></span>
                  </div>
                  <div class="hf-dim hf-hint">click a cell for its value and line · space plays · ← → step · I / L switch field</div>
                </div>
              </div></div>
              <span class="hf-ruler hf-num"></span>
              <div class="hf-hud hf-bl"><div class="hf-card hf-transport">
                <button class="hf-b hf-prev" title="step back (←)">‹</button>
                <button class="hf-b hf-play" title="play / pause (space)">▶</button>
                <button class="hf-b hf-next" title="step forward (→)">›</button>
                <div class="hf-track"><div class="hf-ticks"></div><input class="hf-frame" type="range" min="0" max="0" value="0" step="1" aria-label="frame"></div>
                <div class="hf-stamp hf-num"><small class="hf-stampk">hour (UTC)</small><span class="hf-stampv">–</span></div>
                <select class="hf-fps" title="frames per second"><option>2</option><option>4</option><option>6</option><option>8</option><option>12</option><option>24</option></select>
                <button class="hf-b hf-full" title="fullscreen (F)">⛶</button>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".hf-map"), playBtn = q(".hf-play"), slider = q(".hf-frame"), ticks = q(".hf-ticks"),
                stampV = q(".hf-stampv"), fpsSel = q(".hf-fps"), grad = q(".hf-grad"),
                loEl = q(".hf-lo"), hiEl = q(".hf-hi"), chart = q(".hf-chart"), ruler = q(".hf-ruler"),
                ttl = q(".hf-ttl"), sub = q(".hf-sub"), meanEl = q(".hf-mean"), meanK = q(".hf-meank"), cname = q(".hf-cname"), cval = q(".hf-cval"),
                d0In = q(".hf-d0"), d1In = q(".hf-d1"), loadBtn = q(".hf-load"), noteEl = q(".hf-note"),
                halfIn = q(".hf-half"), thrIn = q(".hf-thr"), rainIn = q(".hf-rain"), windIn = q(".hf-wind"),
                halfV = q(".hf-halfv"), thrV = q(".hf-thrv"), rainV = q(".hf-rainv"), windV = q(".hf-windv"), pnote = q(".hf-pnote");

          let hexes = [], hexIndex = new Map(), cidx = null, names = [], N = 0, F = 0;
          let frames = null, wx = null, load = null, loadHi = 1;
          let cfg = {}, frame = 0, playing = false, timer = null, deck = null, selected = -1;
          let field = "index", gen = 0, lutI = null, lutL = null;
          let meansI = null, meansL = null;
          const HOME = {longitude: -96.5, latitude: 38.3, zoom: 3.8, minZoom: 2, maxZoom: 11};

          const fmtC = v => Number.isFinite(v) ? v.toFixed(1) + "°C" : "no data";
          const hiAt = (f, i) => { const qv = frames[f * N + i]; return qv === 255 ? NaN : HI_OF(qv); };
          const loadAt = (f, i) => { const qv = load[f * N + i]; return qv === 255 ? NaN : qv / 10; };
          const valAt = (f, i) => field === "load" ? loadAt(f, i) : hiAt(f, i);
          const params = () => ({
            half: parseFloat(halfIn.value) || 12, thr: parseFloat(thrIn.value) || 27,
            rain: cfg.has_rain ? (parseFloat(rainIn.value) || 0) : 0, wind: cfg.has_wind ? (parseFloat(windIn.value) || 0) : 0,
          });

          // THE ACCUMULATOR, over the whole film, into uint8 (0.1 degC steps, 255 = no data).
          function computeLoad() {
            if (!frames) return;
            const {half, thr, rain, wind} = params();
            const a = Math.pow(2, -1 / half), b = 1 - a;
            load = load && load.length === F * N ? load : new Uint8Array(F * N);
            const prev = new Float32Array(N), hist = new Uint32Array(256);
            meansL = new Float32Array(F);
            for (let f = 0; f < F; f++) {
              const base = f * N; let s = 0, n = 0;
              for (let i = 0; i < N; i++) {
                const k = base + i, qv = frames[k];
                if (qv === 255) { load[k] = 255; continue; }
                let ex = HI_OF(qv) - thr; if (ex < 0) ex = 0;
                if (wind) { const ws = wx[k] >> 4; ex *= Math.max(0, 1 - wind * ws / 10); }
                let L = a * prev[i] + b * ex;
                if (rain) { const mm = (wx[k] & 15) / 2; if (mm > 0) L *= 1 - rain * Math.min(1, mm / 2.5); }
                prev[i] = L;
                let lq = Math.round(L * 10); if (lq > 254) lq = 254;
                load[k] = lq; hist[lq]++; s += L; n++;
              }
              meansL[f] = n ? s / n : NaN;
            }
            // ramp top: p98 of the non-zero load, at least 1 degC
            let tot = 0; for (let i = 1; i < 255; i++) tot += hist[i];
            let acc = 0, top = 10;
            for (let i = 1; i < 255; i++) { acc += hist[i]; if (acc >= tot * 0.98) { top = i; break; } }
            loadHi = Math.max(1, top / 10);
            gen++;
          }
          function paramLabels() {
            const p = params();
            halfV.textContent = p.half + " h"; thrV.textContent = p.thr.toFixed(1) + "°C";
            rainV.textContent = p.rain.toFixed(2); windV.textContent = p.wind.toFixed(2);
            pnote.textContent = `load = sustained heat index above ${p.thr.toFixed(1)}°C, half-life ${p.half} h` +
              (cfg.has_rain ? "" : " · rain not read") + (cfg.has_wind ? "" : " · wind not read");
          }
          let ptimer = null;
          const onParam = () => { paramLabels(); if (ptimer) clearTimeout(ptimer); ptimer = setTimeout(() => { computeLoad(); legend(); update(); }, 120); };
          halfIn.oninput = thrIn.oninput = rainIn.oninput = windIn.oninput = onParam;

          function loadCells() {
            const u8 = bytesOf(model.get("cells"));
            if (!u8 || !u8.length) return;
            const ids = new BigUint64Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            const c8 = bytesOf(model.get("cidx"));
            cidx = c8 && c8.length ? new Uint16Array(c8.buffer.slice(c8.byteOffset, c8.byteOffset + c8.byteLength)) : null;
            try { names = JSON.parse(model.get("names") || "[]"); } catch (e) { names = []; }
          }
          function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 620) + "px";
            lutI = buildLut(cfg.index_stops || ["#08306b", "#f2f0e6", "#d94801"]);
            lutL = buildLut(cfg.load_stops || ["#000004", "#fcffa4"]);
            const u8 = bytesOf(model.get("frames"));
            if (!u8 || !u8.length || !N) { frames = null; F = 0; return; }
            frames = new Uint8Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            F = Math.floor(frames.length / N);
            const w8 = bytesOf(model.get("wx"));
            wx = w8 && w8.length === frames.length ? new Uint8Array(w8.buffer.slice(w8.byteOffset, w8.byteOffset + w8.byteLength)) : new Uint8Array(frames.length);
            meansI = new Float32Array(F);
            for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let i = 0; i < N; i++) { const qv = frames[f * N + i]; if (qv !== 255) { s += HI_OF(qv); n++; } } meansI[f] = n ? s / n : NaN; }
            slider.max = String(Math.max(0, F - 1));
            if (frame >= F) frame = 0;
            fpsSel.value = String(cfg.fps || 8);
            if (!halfIn.dataset.set) {  // seed the sliders once from the kernel's defaults
              halfIn.value = cfg.half_life ?? 12; thrIn.value = cfg.threshold ?? 27;
              rainIn.value = cfg.rain_flush ?? 0.5; windIn.value = cfg.wind_vent ?? 0.3; halfIn.dataset.set = "1";
            }
            q(".hf-p-rain").classList.toggle("hf-off", !cfg.has_rain);
            q(".hf-p-wind").classList.toggle("hf-off", !cfg.has_wind);
            paramLabels();
            computeLoad();
            ttl.textContent = cfg.title || ""; sub.textContent = cfg.subtitle || "";
            legend();
            syncWindow();
            const labels = cfg.labels || [];
            let html = "";
            for (let f = 1; f < labels.length; f++) {
              const d0 = labels[f - 1].slice(0, 10), d1 = labels[f].slice(0, 10);
              if (d0 !== d1) html += `<i style="left:${(f / (F - 1) * 100).toFixed(2)}%"></i>`;
            }
            ticks.innerHTML = F > 1 ? html : "";
          }
          function legend() {
            const lut = field === "load" ? lutL : lutI;
            const stops = [];
            for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            if (field === "load") { loEl.textContent = "0"; hiEl.textContent = "+" + loadHi.toFixed(1) + "°C"; meanK.textContent = "CONUS mean load"; }
            else { loEl.textContent = fmtC(cfg.lo); hiEl.textContent = fmtC(cfg.hi); meanK.textContent = "CONUS mean heat index"; }
            root.querySelectorAll(".hf-fi").forEach(b => b.classList.toggle("hf-on", b.dataset.field === field));
          }
          const setField = f => { field = f; legend(); update(); };
          root.querySelectorAll(".hf-fi").forEach(b => { b.onclick = () => setField(b.dataset.field); });

          // colour accessor: one attribute re-upload per frame step (updateTriggers)
          function fillColor(d, {index, target}) {
            const k = frame * N + index;
            if (field === "load") {
              const qv = load ? load[k] : 255;
              if (qv === 255) { target[0] = 40; target[1] = 44; target[2] = 50; target[3] = 60; return target; }
              let t = (qv / 10) / loadHi; if (t > 1) t = 1;
              const j = Math.round(t * 255) * 3;
              target[0] = lutL[j]; target[1] = lutL[j + 1]; target[2] = lutL[j + 2]; target[3] = 60 + Math.round(175 * t);
              return target;
            }
            const qv = frames[k];
            if (qv === 255) { target[0] = 40; target[1] = 44; target[2] = 50; target[3] = 60; return target; }
            let t = (HI_OF(qv) - cfg.lo) / (cfg.hi - cfg.lo); t = t < 0 ? 0 : t > 1 ? 1 : t;
            const j = Math.round(t * 255) * 3;
            target[0] = lutI[j]; target[1] = lutI[j + 1]; target[2] = lutI[j + 2]; target[3] = 225;
            return target;
          }
          const tiles = (id, url, opacity) => new TileLayer({
            id, data: url, tileSize: 256, minZoom: 0, maxZoom: 19, opacity, pickable: false,
            renderSubLayers: p => {
              const {west, south, east, north} = p.tile.bbox;
              return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
            },
          });
          function layers() {
            const out = [tiles("base", "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png", 1.0)];
            if (N && frames) {
              out.push(new H3HexagonLayer({
                id: "cells",
                data: hexes,
                getHexagon: d => d,
                getFillColor: fillColor,
                updateTriggers: {getFillColor: [frame, field, gen]},
                filled: true, stroked: false, extruded: false, coverage: 1,
                highPrecision: true,
                pickable: false,
              }));
              if (selected >= 0) out.push(new H3HexagonLayer({
                id: "picked",
                data: [hexes[selected]],
                getHexagon: d => d,
                filled: false, stroked: true, extruded: false, highPrecision: true,
                getLineColor: [230, 193, 74, 255], lineWidthUnits: "pixels", getLineWidth: 2, lineWidthMinPixels: 2,
                pickable: false,
              }));
            }
            out.push(tiles("labels", "https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png", 0.6));
            return out;
          }

          function stats() {
            const m = field === "load" ? meansL : meansI;
            meanEl.textContent = m && F ? (field === "load" ? "+" : "") + fmtC(m[frame]) : "–";
            if (selected >= 0) cval.textContent = (field === "load" ? "+" : "") + fmtC(valAt(frame, selected));
          }
          function drawChart() {
            if (selected < 0 || !frames || F < 2) return;
            const w = chart.clientWidth || 300, h = chart.height;
            if (chart.width !== w) chart.width = w;
            const g = chart.getContext("2d");
            g.clearRect(0, 0, w, h);
            const L = 44, R = 4, T = 6, B = 14;
            const X = f => L + (w - L - R) * f / (F - 1);
            let lo = Infinity, hi = -Infinity;
            for (let f = 0; f < F; f++) { const v = valAt(f, selected); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
            if (!Number.isFinite(lo)) return;
            if (field === "load") lo = 0;
            if (hi - lo < 1) { hi += .5; lo = field === "load" ? 0 : lo - .5; }
            const Y = v => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(lo)); g.lineTo(w - R, Y(lo)); g.moveTo(L, Y(hi)); g.lineTo(w - R, Y(hi)); g.stroke();
            if (field === "index") {  // the threshold, as a dashed line
              const thr = params().thr;
              if (thr > lo && thr < hi) { g.setLineDash([3, 3]); g.strokeStyle = "#8b929c"; g.beginPath(); g.moveTo(L, Y(thr)); g.lineTo(w - R, Y(thr)); g.stroke(); g.setLineDash([]); }
            }
            g.fillStyle = "#8b929c"; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "right";
            g.fillText(fmtC(hi), L - 4, Y(hi) + 4); g.fillText(fmtC(lo), L - 4, Y(lo) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, h - 3);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - R, h - 3);
            g.strokeStyle = "#e6c14a"; g.lineWidth = 1.5; g.beginPath();
            let pen = false;
            for (let f = 0; f < F; f++) { const v = valAt(f, selected); if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
            g.stroke();
            g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(frame), T); g.lineTo(X(frame), h - B); g.stroke();
            const cv = valAt(frame, selected);
            if (Number.isFinite(cv)) { g.fillStyle = "#ffffff"; g.beginPath(); g.arc(X(frame), Y(cv), 3, 0, 6.283); g.fill(); }
          }
          chart.addEventListener("click", ev => {
            if (F < 2) return;
            const r = chart.getBoundingClientRect(), L = 44, R = 4;
            const t = ((ev.clientX - r.left) - L) / (r.width - L - R);
            frame = Math.max(0, Math.min(F - 1, Math.round(t * (F - 1)))); update();
          });

          // THE WINDOW CONTROL: the one thing that crosses back (see the counties film).
          let loading = false;
          const dayCount = () => {
            const a = Date.parse(d0In.value), b = Date.parse(d1In.value);
            return Number.isFinite(a) && Number.isFinite(b) ? Math.round(Math.abs(b - a) / 864e5) + 1 : 0;
          };
          // the read's cost for the dates picked: every 90-day store chunk the window
          // touches is fetched to its filled depth (see the kernel's window cell)
          function costOf() {
            const w = cfg.win || {};
            if (!w.store_start || !w.store_hours) return w.cost || "30 s";
            const a = Date.parse(d0In.value), b = Date.parse(d1In.value), s0 = Date.parse(w.store_start);
            if (!Number.isFinite(a) || !Number.isFinite(b)) return w.cost || "30 s";
            const h0 = Math.floor((Math.min(a, b) - s0) / 36e5), h1 = Math.min(Math.floor((Math.max(a, b) - s0) / 36e5) + 23, w.store_hours - 1);
            const ch = w.chunk_h || 2160; let filled = 0;
            for (let c = Math.floor(h0 / ch); c <= Math.floor(h1 / ch); c++) filled += Math.min(ch, w.store_hours - c * ch);
            const s = Math.round(6 + 0.055 * filled);
            return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;
          }
          function checkWindow() {
            const w = cfg.win || {};
            const n = dayCount(), lim = w.hourly_max || 14;
            let bad = "";
            if (!n) bad = "pick both days";
            else if (n > lim) bad = `${n} days is over the ${lim}-day limit`;
            noteEl.classList.toggle("hf-bad", !!bad);
            if (loading) noteEl.textContent = `loading ${n} days · about ${costOf()}…`;
            else noteEl.textContent = bad || `${n} UTC days · ${n * 24} hourly frames · read about ${costOf()} · limit ${lim} d`;
            loadBtn.disabled = loading || !!bad;
            return !bad;
          }
          function syncWindow() {
            const w = cfg.win;
            if (!w) return;
            d0In.min = d1In.min = w.first || ""; d0In.max = d1In.max = w.last || "";
            if (w.d0) d0In.value = w.d0;
            if (w.d1) d1In.value = w.d1;
            loading = false; loadBtn.textContent = "load";
            checkWindow();
          }
          d0In.onchange = d1In.onchange = checkWindow;
          loadBtn.onclick = () => {
            if (!checkWindow()) return;
            let d0 = d0In.value, d1 = d1In.value;
            if (d1 < d0) [d0, d1] = [d1, d0];
            loading = true; loadBtn.textContent = "loading"; frame = 0; checkWindow();
            model.set("window", JSON.stringify({d0, d1}));
            model.save_changes();
          };
          checkWindow();

          function select(i) {
            if (!(i >= 0 && i < N)) return;
            selected = i;
            root.classList.add("hf-picked");
            const c = cidx && cidx[i] !== 65535 ? names[cidx[i]] : null;
            cname.textContent = c ? `cell in ${c}` : `cell ${hexes[i]}`;
            update();
          }
          q(".hf-clear").onclick = () => { selected = -1; root.classList.remove("hf-picked"); update(); };
          function update() {
            if (!deck) return;
            deck.setProps({layers: layers()});
            slider.value = String(frame);
            stampV.textContent = (cfg.labels && cfg.labels[frame]) ? cfg.labels[frame] : `frame ${frame}`;
            stats(); drawChart();
          }
          function setPlaying(p) {
            playing = p; playBtn.textContent = p ? "❚❚" : "▶";
            if (timer) { clearInterval(timer); timer = null; }
            if (p && F > 1) timer = setInterval(() => { frame = (frame + 1) % F; update(); }, 1000 / (parseFloat(fpsSel.value) || 8));
          }
          const step = d => { if (F) { frame = (frame + d + F) % F; update(); } };
          playBtn.onclick = () => setPlaying(!playing);
          q(".hf-prev").onclick = () => step(-1);
          q(".hf-next").onclick = () => step(1);
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          fpsSel.onchange = () => { if (playing) setPlaying(true); };
          const toggle = q(".hf-toggle");
          // "hf-collapsed", never "hidden": marimo's Tailwind owns `.hidden` (see the counties film)
          toggle.onclick = () => { root.classList.toggle("hf-collapsed"); toggle.textContent = root.classList.contains("hf-collapsed") ? "show" : "hide"; };
          q(".hf-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen?.(); };
          mapEl.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 620) + "px"; });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "BUTTON") return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".hf-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { toggle.click(); }
            else if (ev.key === "i" || ev.key === "I") { setField("index"); }
            else if (ev.key === "l" || ev.key === "L") { setField("load"); }
          });

          const rulerText = () => `${N.toLocaleString()} cells · ${F} frames`;
          function boot() {
            loadCells(); loadFrames();
            deck = new Deck({
              parent: mapEl,
              initialViewState: HOME,
              controller: true,
              layers: layers(),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); },
            });
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".hf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
            mapEl.addEventListener("pointerup", ev => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !deck) return;
              const r = mapEl.getBoundingClientRect();
              let ll = null;
              try { ll = deck.getViewports()[0].unproject([ev.clientX - r.left, ev.clientY - r.top]); }
              catch (e) { ruler.textContent = "unproject: " + e.message; return; }
              let i = -1;
              try { i = hexIndex.get(latLngToCell(ll[1], ll[0], cfg.res || 6)) ?? -1; } catch (e) { i = -1; }
              if (i >= 0 && i !== selected) select(i);
              else { selected = -1; root.classList.remove("hf-picked"); update(); }
            }, true);
            ruler.textContent = rulerText();
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:cells", () => { loadCells(); loadFrames(); ruler.textContent = rulerText(); update(); });
          model.on("change:frames", () => { loadFrames(); ruler.textContent = rulerText(); update(); });
          model.on("change:config", () => { loadFrames(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (deck) deck.finalize(); };
        }
        export default {render};
        """
        cells = traitlets.Bytes(b"").tag(sync=True)
        cidx = traitlets.Bytes(b"").tag(sync=True)
        names = traitlets.Unicode("[]").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        wx = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        # browser -> kernel, the one thing that crosses back: {"d0","d1"} JSON from the
        # HUD's load button ("" until the first load, meaning the default window).
        window = traitlets.Unicode("").tag(sync=True)

    return (HexFilm,)


@app.cell
async def _(
    BOX,
    CACHE_DIR,
    COUNTY_Z,
    NOT_CONUS,
    OVERTURE_RELEASE,
    PM_BUCKET,
    PM_PATH,
    S3Store,
    asyncio,
    con,
    gzip,
    math,
    np,
    obstore,
    pa,
    pq,
    struct,
):
    # THE COUNTIES, OUT OF ONE PMTILES OBJECT BY RANGED GET. The client and the MVT
    # decode are the interactive notebook's, ported by copy and trimmed of the LRU and
    # the coverage memo: everything here is fetched exactly once. The decode was
    # verified ring-exact against mapbox-vector-tile there before being trusted.
    import time as _ctime

    _ct0 = _ctime.perf_counter()
    # DISK CACHE in the OS temp dir: the dissolved counties never change for a pinned
    # Overture release and BOX, and this fetch + dissolve is ~7.3 s of the ~30 s before
    # the map. First run writes the parquet, every run after reads it (0.0 s).
    # CACHE_DIR = None turns it off.
    import pathlib as _pl

    _cache = (
        _pl.Path(CACHE_DIR) / f"counties-{OVERTURE_RELEASE}-z{COUNTY_Z}-{'-'.join(str(b) for b in BOX)}.parquet"
        if CACHE_DIR
        else None
    )
    _rows, _x0, _y0, _x1, _y1, _t_fetch = [], 0, 0, -1, -1, 0.0
    if _cache is not None and _cache.exists():
        counties = pq.read_table(_cache)
        _how = f"from {_cache}"
    else:
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
            """A PMTiles v3 directory: four varint columns, tile ids delta-encoded."""
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

        _hdr = await _pm_range(0, 126)
        assert _hdr[:7] == b"PMTiles" and _hdr[7] == 3, "not a PMTiles v3 archive"
        _rd_off, _rd_len, _, _, _ld_off, _, _td_off, _ = struct.unpack("<8Q", _hdr[8:72])
        assert COUNTY_Z <= _hdr[101], "COUNTY_Z above the pyramid"
        _root = _parse_dir(gzip.decompress(await _pm_range(_rd_off, _rd_off + _rd_len - 1)))
        _leaf = {}

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
            """Twice the signed shoelace area: >0 marks an exterior ring (tile y is down)."""
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
            """Tile-integer rings -> a lon/lat MultiPolygon WKB, closed-form Web Mercator."""
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

        _sem = asyncio.Semaphore(32)

        async def _tile_pieces(z, x, y):
            """One tile, walked to through the directories, decoded, filtered to CONUS counties.

            A piece is one county's presence in one tile. The filter runs at decode: county
            subtype only, land only (is_land is always present in this tileset, measured in
            the interactive notebook), country US, region not in NOT_CONUS. `division_id`
            rather than `id`, because `id` names the AREA row and a division can own
            several; joining on the wrong one silently returns zero rows.
            """
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
                    if props.get("subtype") != "county":
                        continue
                    if props.get("is_land") is not True or not polys:
                        continue
                    if props.get("country") != "US":
                        continue
                    region = (props.get("region") or "").split("-", 1)[-1]
                    if region in NOT_CONUS:
                        continue
                    pieces.append(
                        {
                            "id": props.get("division_id") or props.get("id"),
                            "name": props.get("@name"),
                            "region": region,
                            "wkb": _feature_wkb(polys, z, x, y, extent),
                        }
                    )
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

        _x0, _y0 = _mtile(BOX[0], BOX[3], COUNTY_Z)
        _x1, _y1 = _mtile(BOX[2], BOX[1], COUNTY_Z)
        _parts = await asyncio.gather(
            *(
                _tile_pieces(COUNTY_Z, xx, yy)
                for yy in range(_y0, _y1 + 1)
                for xx in range(_x0, _x1 + 1)
            )
        )
        _rows = [p for tp in _parts for p in tp]
        _t_fetch = _ctime.perf_counter() - _ct0

        # THE SEAM DISSOLVE. Tile geometry arrives clipped, so one county is several pieces
        # and the clip edges are straight lines the stroke would draw across the map.
        # Union-ing per division removes every interior edge; the tile buffer (pieces
        # overlap slightly past each tile edge) is what makes the union clean.
        #
        # con.register, NOT the replacement scan the interactive notebook leans on: marimo
        # mangles underscore-prefixed cell locals to make them cell-private, so the frame
        # name never matches the SQL name and DuckDB reports the table as missing.
        _pieces = pa.table(
            {
                "id": pa.array([r["id"] for r in _rows]),
                "name": pa.array([r["name"] for r in _rows]),
                "region": pa.array([r["region"] for r in _rows]),
                "wkb": pa.array([r["wkb"] for r in _rows], pa.binary()),
            }
        )
        con.register("pm_pieces", _pieces)
        counties = con.sql("""
            SELECT id,
                   any_value(name)   AS name,
                   any_value(region) AS region,
                   CAST(ST_AsWKB(ST_Union_Agg(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb
            FROM pm_pieces
            GROUP BY id
        """).to_arrow_table()
        con.unregister("pm_pieces")
        _how = "fetched"
        if _cache is not None:
            _cache.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(counties, _cache)

    county_stats = (
        f"{counties.num_rows:,} counties {_how} · "
        + (
            f"{(_x1 - _x0 + 1) * (_y1 - _y0 + 1)} tiles at z{COUNTY_Z} · {len(_rows):,} pieces · fetch {_t_fetch:.1f}s, with dissolve "
            if _rows
            else ""
        )
        + f"{_ctime.perf_counter() - _ct0:.1f}s"
    )
    return counties, county_stats


@app.cell
def _(ANALYSIS_BUCKET, ANALYSIS_PREFIX, READ_RAIN, READ_WIND, np, xr):
    # THE STORE, opened once; only metadata and the 2-D lat/lon are read here. NO DASK:
    # chunks=None leaves it lazily indexed and xarray-sql cuts it into blocks itself.
    import time as _stime

    import icechunk

    _st0 = _stime.perf_counter()
    _storage = icechunk.s3_storage(
        bucket=ANALYSIS_BUCKET, prefix=ANALYSIS_PREFIX, region="us-west-2", anonymous=True
    )
    _sess = icechunk.Repository.open(_storage).readonly_session("main")
    _ds = xr.open_zarr(_sess.store, consolidated=False, chunks=None)
    VARS = ["temperature_2m", "relative_humidity_2m"]
    if READ_RAIN:
        VARS.append("precipitation_surface")
    if READ_WIND:
        VARS += ["wind_u_10m", "wind_v_10m"]
    cube_all = _ds[VARS].rename({"time": "t"})
    all_times = cube_all["t"].values.astype("datetime64[m]")
    lat = _ds["latitude"].values.astype("float64")
    lon = _ds["longitude"].values.astype("float64")
    grid_y = _ds["y"].values
    grid_x = _ds["x"].values
    source_note = "HRRR analysis"
    store_stats = (
        f"{source_note} · {len(VARS)} variables · grid {lat.shape[1]}x{lat.shape[0]} px · hourly "
        f"{np.datetime_as_string(all_times[0])} to {np.datetime_as_string(all_times[-1])} UTC "
        f"({all_times.size:,} steps) · open {_stime.perf_counter() - _st0:.1f}s"
    )
    return (
        VARS,
        all_times,
        cube_all,
        grid_x,
        grid_y,
        lat,
        lon,
        source_note,
        store_stats,
    )


@app.cell
def _(DAYS, HOURLY_MAX_DAYS, all_times, film, json, mo, np):
    # THE WINDOW: the HUD's `window` trait once "load" has been pressed, else the
    # opening default. Read off the widget, not film.value (that packs every synced
    # trait, the frame bytes included). Over the limit stops with the reason.
    import datetime as _wdt

    _last = all_times[-1].astype("datetime64[D]").astype(_wdt.date)
    _first = all_times[0].astype("datetime64[D]").astype(_wdt.date)
    _req = {}
    try:
        _req = json.loads(film.widget.window or "{}")
    except ValueError:
        _req = {}
    if _req.get("d0") and _req.get("d1"):
        _d0 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d0"])))
        _d1 = max(_first, min(_last, _wdt.date.fromisoformat(_req["d1"])))
    elif isinstance(DAYS, tuple):
        _d0, _d1 = (max(_first, min(_last, _wdt.date.fromisoformat(d))) for d in DAYS)
    else:
        _d0, _d1 = _last - _wdt.timedelta(days=DAYS - 1), _last
    if _d1 < _d0:
        _d0, _d1 = _d1, _d0
    n_days = (_d1 - _d0).days + 1
    # The read's cost, stated to the HUD: every store chunk the window touches is
    # fetched to its filled depth (2,160 h when full), whatever the window length.
    # 6 s + 0.055 s per filled hour fits both measurements (447 h -> 30 s, a full
    # chunk -> 125 s) on a ~21 MB/s link with two variables; the JS shows it.
    _h0 = int((np.datetime64(_d0.isoformat()) - all_times[0].astype("datetime64[D]")) // np.timedelta64(1, "h"))
    _h1 = int((np.datetime64(_d1.isoformat()) - all_times[0].astype("datetime64[D]")) // np.timedelta64(1, "h")) + 23
    _filled = sum(
        min(2160, all_times.size - _c * 2160) for _c in range(_h0 // 2160, min(_h1, all_times.size - 1) // 2160 + 1)
    )
    read_cost_s = int(round(6 + 0.055 * _filled))
    win_cfg = {
        "first": _first.isoformat(),
        "last": _last.isoformat(),
        "d0": _d0.isoformat(),
        "d1": _d1.isoformat(),
        "hourly_max": HOURLY_MAX_DAYS,
        "cost": f"{read_cost_s} s" if read_cost_s < 90 else f"{read_cost_s / 60:.0f} min",
        "chunk_h": 2160,
        "store_start": all_times[0].astype("datetime64[D]").astype(_wdt.date).isoformat(),
        "store_hours": int(all_times.size),
    }
    mo.stop(
        n_days > HOURLY_MAX_DAYS,
        mo.md(f"**{n_days} days is over the {HOURLY_MAX_DAYS}-day limit.** Shorten the window."),
    )
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[ns]")
    t1 = min(
        (np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[ns]"),
        all_times[-1].astype("datetime64[ns]"),
    )
    window_note = (
        f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to "
        f"{np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    )
    return n_days, t0, t1, win_cfg, window_note


@app.cell
def _(
    RES,
    con,
    coordinates_to_cells,
    counties,
    grid_x,
    grid_y,
    lat,
    lon,
    np,
    pa,
):
    # PIXEL -> CELL, ONCE, AND THE LAND MASK. Cell per pixel from the store's own
    # lat/lon; CONUS land = the res 6 cells whose centre falls in a county (DuckDB
    # polyfill, 'center' rule, so each cell has exactly one county, which is the click
    # readout's name). Only pixels in land cells enter the fold. The store is 960
    # blocks of 45x45 px; the ones that touch no land cell are named in a y/x range
    # predicate (one term per block row, runs of touching block columns) that
    # xarray-sql pushes down as partition pruning, so those columns are never fetched
    # (measured: 523 of 960 blocks read, 2 variables 44.7 s -> 28.5 s).
    import time as _ptime

    _pt0 = _ptime.perf_counter()
    con.register("conus_divs", counties)
    _mapping = con.sql(
        """
        WITH parts AS (
            SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM conus_divs
        ), filled AS (
            SELECT id, UNNEST(
                       h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), ?, 'center')
                   ) AS hex
            FROM parts
        )
        SELECT hex, any_value(id) AS id FROM filled GROUP BY hex ORDER BY hex
        """,
        params=[int(RES)],
    ).to_arrow_table()
    con.unregister("conus_divs")
    _t_fill = _ptime.perf_counter() - _pt0

    _ny, _nx = lat.shape
    _hex = np.asarray(coordinates_to_cells(lat.ravel(), lon.ravel(), int(RES)))
    cells = _mapping["hex"].to_numpy().astype(np.uint64)  # sorted: the film's row order
    _land = np.isin(_hex, cells).reshape(_ny, _nx)
    _flat = _land.ravel()
    pix2h = pa.table(
        {
            "y": pa.array(np.repeat(grid_y, _nx)[_flat]),
            "x": pa.array(np.tile(grid_x, _ny)[_flat]),
            "hex": pa.array(_hex[_flat]),
        }
    )
    # county index per cell (uint16, 65535 = none), names in county-table order
    _cid = counties["id"].to_pylist()
    _pos = {i: k for k, i in enumerate(_cid)}
    cell_county = np.fromiter((_pos.get(i, 65535) for i in _mapping["id"].to_pylist()), dtype=np.uint16, count=cells.size)
    county_names = [f"{n}, {r}" for n, r in zip(counties["name"].to_pylist(), counties["region"].to_pylist())]

    # the land-block predicate for partition pruning
    _B = 45
    _by, _bx = -(-_ny // _B), -(-_nx // _B)
    _terms, n_land_blocks = [], 0
    for _j in range(_by):
        _cols = [_i for _i in range(_bx) if _land[_j * _B:(_j + 1) * _B, _i * _B:(_i + 1) * _B].any()]
        if not _cols:
            continue
        _runs, _s, _prev = [], _cols[0], _cols[0]
        for _i in _cols[1:]:
            if _i != _prev + 1:
                _runs.append((_s, _prev))
                _s = _i
            _prev = _i
        _runs.append((_s, _prev))
        _ys = (float(grid_y[_j * _B]), float(grid_y[min((_j + 1) * _B, _ny) - 1]))
        _xr = []
        for _a, _b in _runs:
            _xs = (float(grid_x[_a * _B]), float(grid_x[min((_b + 1) * _B, _nx) - 1]))
            _xr.append(f"cube.x BETWEEN {min(_xs)} AND {max(_xs)}")
            n_land_blocks += _b - _a + 1
        _terms.append(f"(cube.y BETWEEN {min(_ys)} AND {max(_ys)} AND ({' OR '.join(_xr)}))")
    land_pred = " OR ".join(_terms)
    pix_stats = (
        f"{cells.size:,} res {RES} land cells (polyfill {_t_fill:.1f}s) · "
        f"{pix2h.num_rows:,} of {_hex.size:,} pixels on CONUS land · "
        f"{n_land_blocks} of {_by * _bx} store blocks touch land · {_ptime.perf_counter() - _pt0:.1f}s"
    )
    return cell_county, cells, county_names, land_pred, pix2h, pix_stats


@app.cell
def _():
    # Kernel-side memo across window loads: the last folded window and its table.
    HOLD = {"key": None, "cell_hour": None, "stats": ""}
    return (HOLD,)


@app.cell
def _(
    HOLD,
    READ_RAIN,
    READ_WIND,
    VARS,
    XarrayContext,
    cube_all,
    land_pred,
    pix2h,
    t0,
    t1,
):
    # THE FOLD, ONE STATEMENT, STRAIGHT OFF THE CUBE: the join to pix2h on the grid
    # coordinates is the H3 fold (each pixel carries its cell), the group by averages
    # the ~4 pixels per cell per hour, and the land predicate prunes the blocks that
    # touch no land before any byte is fetched. Blocks span the whole time window
    # (one block per store column, so each 2,160 h store chunk is decoded once). Wind
    # speed is averaged per pixel (sqrt(u^2+v^2)), not from the mean vector; rain rate
    # x 3600 is mm in the hour. Floats, not doubles, in the output: 35M rows at res 6.
    #
    # Memoised on the window: re-submitting the same dates never refetches.
    import time as _ftime

    _key = (tuple(VARS), str(t0), str(t1))
    if HOLD["key"] == _key and HOLD["cell_hour"] is not None:
        cell_hour = HOLD["cell_hour"]
        fold_stats = HOLD["stats"] + " (memo)"
    else:
        _ft0 = _ftime.perf_counter()
        _cube = cube_all.sel(t=slice(t0, t1))
        _hours = int(_cube.sizes["t"])
        # A 3 GB fair spill pool: the final aggregate holds one entry per (hour, cell)
        # answer until the last block lands (35M at res 6), and over the pool it spills
        # to the temp dir instead of growing; measured ~5 GB process peak against 9.5
        # without, same ~28 s. (A smaller pool spilled more and measured HIGHER; spill
        # buffers live outside it.)
        from datafusion import RuntimeEnvBuilder as _RTB, SessionConfig as _SC

        ctx = XarrayContext(_SC(), _RTB().with_fair_spill_pool(3 << 30))
        # Broadcast the pixel lookup to every block partition (CollectLeft) instead of
        # re-hashing the cube by (y, x): with the default Partitioned join a cell's
        # pixels scatter across partitions and every partition's partial aggregate
        # holds its own copy of nearly every (hour, cell) group (measured at res 6:
        # 17 GB against 9.5 GB). pix2h is ~10 MB, above the 1 MB / 128k-row defaults.
        ctx.sql("SET datafusion.optimizer.hash_join_single_partition_threshold = 268435456")
        ctx.sql("SET datafusion.optimizer.hash_join_single_partition_threshold_rows = 16777216")
        ctx.from_arrow(pix2h, name="pix2h")
        ctx.from_dataset("cube", _cube, chunks={"t": _hours, "y": 45, "x": 45})
        _cols = [
            "CAST(avg(CAST(temperature_2m AS DOUBLE)) AS FLOAT) AS tc",
            "CAST(avg(CAST(relative_humidity_2m AS DOUBLE)) AS FLOAT) AS rh",
        ]
        if READ_RAIN:
            _cols.append("CAST(avg(CAST(precipitation_surface AS DOUBLE)) * 3600 AS FLOAT) AS mm")
        if READ_WIND:
            _cols.append(
                "CAST(avg(sqrt(CAST(wind_u_10m AS DOUBLE) * wind_u_10m + CAST(wind_v_10m AS DOUBLE) * wind_v_10m)) AS FLOAT) AS ws"
            )
        cell_hour = ctx.sql(f"""
            SELECT t, hex, {", ".join(_cols)}
            FROM cube JOIN pix2h USING (y, x)
            WHERE temperature_2m = temperature_2m AND ({land_pred})
            GROUP BY 1, 2
        """).to_arrow_table()
        fold_stats = (
            f"{_hours} hours · {len(VARS)} variables · {cell_hour.num_rows:,} cell-hour rows · "
            f"fold {_ftime.perf_counter() - _ft0:.1f}s"
        )
        HOLD["key"], HOLD["cell_hour"], HOLD["stats"] = _key, cell_hour, fold_stats
    return cell_hour, fold_stats


@app.cell
def _(PIVOT, SPAN, cell_hour, cells, np):
    # THE FRAME MATRICES: F hours x N cells, in `cells` order (sorted ids; the widget
    # indexes by row). Heat index from the cell-mean temperature and humidity (NWS:
    # Steadman's simple formula, the Rothfusz regression once the mean of it and T
    # reaches 80 F, with the two RH adjustments), quantised to uint8 in 0.5 degC steps
    # from -40 (255 = no data). Wind and rain, if read, packed in one byte: wind m/s
    # rounded (0..15) in the high nibble, rain in 0.5 mm/h steps (0..7.5) in the low.
    # One ramp for the film's heat index: pivot at the median, span to the wider of
    # p2/p98, unless PIVOT/SPAN pin them.
    def _heat_index_c(tc, rh):
        T = tc * 9.0 / 5.0 + 32.0
        hi = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + rh * 0.094)
        m = (hi + T) / 2.0 >= 80.0
        T2, R2 = T[m], rh[m]
        h = (
            -42.379 + 2.04901523 * T2 + 10.14333127 * R2 - 0.22475541 * T2 * R2
            - 0.00683783 * T2 * T2 - 0.05481717 * R2 * R2 + 0.00122874 * T2 * T2 * R2
            + 0.00085282 * T2 * R2 * R2 - 0.00000199 * T2 * T2 * R2 * R2
        )
        a1 = (R2 < 13) & (T2 >= 80) & (T2 <= 112)
        h[a1] -= ((13 - R2[a1]) / 4.0) * np.sqrt((17 - np.abs(T2[a1] - 95.0)) / 17.0)
        a2 = (R2 > 85) & (T2 >= 80) & (T2 <= 87)
        h[a2] += ((R2[a2] - 85.0) / 10.0) * ((87.0 - T2[a2]) / 5.0)
        hi[m] = h
        return (hi - 32.0) * 5.0 / 9.0

    _t = cell_hour["t"].to_numpy()
    _fkeys = np.unique(_t)
    _fi = np.searchsorted(_fkeys, _t)
    _ci = np.searchsorted(cells, cell_hour["hex"].to_numpy().astype(np.uint64))
    F, N = _fkeys.size, cells.size
    _tc = np.full((F, N), np.nan, dtype=np.float32)
    _rh = np.full((F, N), np.nan, dtype=np.float32)
    _tc[_fi, _ci] = cell_hour["tc"].to_numpy()
    _rh[_fi, _ci] = cell_hour["rh"].to_numpy()
    _hi = _heat_index_c(_tc.astype(np.float64), _rh.astype(np.float64)).astype(np.float32)
    _ok = np.isfinite(_hi)
    hi_q = np.full((F, N), 255, dtype=np.uint8)
    hi_q[_ok] = np.clip(np.rint((_hi[_ok] + 40.0) * 2.0), 0, 254).astype(np.uint8)
    wx_q = np.zeros((F, N), dtype=np.uint8)
    has_rain = "mm" in cell_hour.column_names
    has_wind = "ws" in cell_hour.column_names
    if has_rain:
        _mm = np.zeros((F, N), dtype=np.float32)
        _mm[_fi, _ci] = cell_hour["mm"].to_numpy()
        wx_q |= np.clip(np.rint(np.nan_to_num(_mm) * 2.0), 0, 15).astype(np.uint8)
    if has_wind:
        _ws = np.zeros((F, N), dtype=np.float32)
        _ws[_fi, _ci] = cell_hour["ws"].to_numpy()
        wx_q |= (np.clip(np.rint(np.nan_to_num(_ws)), 0, 15).astype(np.uint8) << 4)
    frame_labels = [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in _fkeys]

    _vals = _hi[_ok]
    _mid = float(np.median(_vals)) if PIVOT is None else float(PIVOT)
    _span = (
        float(max(_mid - np.percentile(_vals, 2), np.percentile(_vals, 98) - _mid))
        if SPAN is None
        else float(SPAN)
    )
    ramp_lo, ramp_mid, ramp_hi = _mid - _span, _mid, _mid + _span
    frame_stats = (
        f"{F} frames x {N:,} cells · heat index {np.nanmin(_hi):.1f} to {np.nanmax(_hi):.1f} °C · "
        f"ramp {ramp_lo:.1f} / {ramp_mid:.1f} / {ramp_hi:.1f} · "
        f"{hi_q.nbytes / 1e6:.0f} MB per field to the browser"
    )
    return (
        frame_labels,
        frame_stats,
        has_rain,
        has_wind,
        hi_q,
        ramp_hi,
        ramp_lo,
        ramp_mid,
        wx_q,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Using the map.** Space plays, arrows step, drag the slider to scrub, `I` / `L`
    (or the two buttons) switch between the heat index and the heat load, `H` hides
    the panel, `F` or ⛶ goes fullscreen. Click a cell for its value and its line over
    the window; click it again, or empty ground, to clear. The four sliders are the
    accumulator: half-life (how long the load takes to halve once the heat is gone),
    threshold (heat index above which load builds; the dashed line on a cell's index
    chart), rain flush and wind vent (only when those fields were read). Every move
    recomputes the whole film in the browser. The window takes UTC days, inclusive,
    up to 14; load refetches and refolds, about thirty seconds.
    """)
    return


@app.cell
def _(HexFilm, cell_county, cells, county_names, json, mo):
    # THE WIDGET, BUILT ONCE with the cell ids and nothing else. Frames and config are
    # set from the wiring cell below, so a window change never rebuilds the map.
    film = mo.ui.anywidget(
        HexFilm(
            cells=cells.astype("<u8").tobytes(),
            cidx=cell_county.astype("<u2").tobytes(),
            names=json.dumps(county_names),
        )
    )
    film
    return (film,)


@app.cell
def _(
    FPS,
    HALF_LIFE,
    INDEX_STOPS,
    LOAD_STOPS,
    MAP_HEIGHT,
    RAIN_FLUSH,
    RES,
    THRESHOLD,
    WIND_VENT,
    film,
    fold_stats,
    frame_labels,
    frame_stats,
    has_rain,
    has_wind,
    hi_q,
    json,
    n_days,
    ramp_hi,
    ramp_lo,
    ramp_mid,
    source_note,
    win_cfg,
    window_note,
    wx_q,
):
    # THE WIRING: re-runs on every window change and only pushes JSON + bytes at the
    # existing widget. Config, then wx, then frames: the JS recomputes on frames.
    film.config = json.dumps(
        {
            "labels": frame_labels,
            "lo": ramp_lo,
            "mid": ramp_mid,
            "hi": ramp_hi,
            "index_stops": INDEX_STOPS,
            "load_stops": LOAD_STOPS,
            "threshold": THRESHOLD,
            "half_life": HALF_LIFE,
            "rain_flush": RAIN_FLUSH,
            "wind_vent": WIND_VENT,
            "has_rain": has_rain,
            "has_wind": has_wind,
            "res": RES,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "title": f"heat index · {source_note}",
            "subtitle": f"{window_note} · {n_days} days · hourly",
            "meta": f"{fold_stats} · {frame_stats}",
            "win": win_cfg,
            "autoplay": False,
        }
    )
    film.wx = wx_q.tobytes() if (has_rain or has_wind) else b""
    film.frames = hi_q.tobytes()
    return


@app.cell
def _(county_stats, fold_stats, frame_stats, mo, pix_stats, store_stats):
    mo.md(
        "<br>".join(
            f"<span style='color:#8b929c;font-size:.85em'>{s}</span>"
            for s in (store_stats, county_stats, pix_stats, fold_stats, frame_stats)
        )
    )
    return


if __name__ == "__main__":
    app.run()
