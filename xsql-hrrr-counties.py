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
#     "arro3-core",
#     "geoarrow-rust-core",
#     "obstore>=0.9.2",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "duckdb>=1.5.5",
# ]
# ///
"""HRRR 2 m temperature per CONUS county, hour by hour, as an animated choropleth.

The pipeline is the deforestation county one-shot's, pointed at weather: dynamical.org's
HRRR analysis (3 km, hourly, CC-BY 4.0) read straight from its Zarr with xarray-sql,
every pixel labelled with its H3 res 7 cell from the store's own 2-D lat/lon (no
pyproj), Overture counties out of the divisions PMTiles, dissolved and polyfilled in
DuckDB at the same res, and one DataFusion join + group by giving temperature per
county per hour for the last DAYS days. That table is small (3,108 counties x hours),
so the WHOLE FILM is shipped to the browser once and the browser owns the clock: a
bespoke anywidget with deck.gl + @geoarrow/deck.gl-layers (the same layers lonboard
renders with) draws the counties from one GeoArrow IPC table and recolours them per
frame from a Float32Array. The HUD is minimal and ON THE MAP (one hideable panel:
title, legend, county mean, a clicked county's line; one transport bar), so the deck
element's own browser fullscreen carries it. Nothing crosses back to the kernel while
it plays; a submit form above the map (UTC date range + hourly / daily mean / daily
max, with per-mode limits) is the only thing that reaches back for data.

RES 7 IS FINER THAN THE DATA. A res 7 hex averages 5.16 km2 and an HRRR pixel is 9 km2,
so the "fold" is a relabel (measured: 1,905,141 cells for 1,905,141 pixels, 1.00 px
per cell) and ~40% of the res 7 county cells hold no pixel; the county mean is still
an honest mean of its pixels, each counted once. Res 6 (4.2 px per cell) is the fold
that actually averages, and is the fallback if the polyfill or the join get slow.
3,107 of 3,108 counties catch at least one pixel; Lexington VA (6.5 km2) does not and
is drawn hollow.

THE ANALYSIS STORE IS TIME-OPTIMISED (chunks 2,160 hours deep x 45 px square), so
any CONUS window up to 90 days costs about the same fetch: 24 h and 240 h both
measured near 20 s through xarray-sql; the full 2,160 h chunk depth is 149 s. DAYS
scales the frame count, not the read, until it crosses a 90-day chunk boundary.
Days are UTC days. The 48-hour forecast on source.coop is the other SOURCE (plain
Zarr, all 49 leads x CONUS in 2.2 s); the pipeline is identical from the fold on.

WHAT IT COSTS TO RUN, AND WHY. About thirty seconds to the first frame from a cold
start: the store opens in ~3 s, the counties take ~7 s (1,008 ranged GETs against the
PMTiles object plus the MVT decode; the dissolve is 0.1 s), the pixel -> county lookup
~2 s, and the fold ~20 s. THE COUNTIES ARE CACHED ON DISK as one parquet in the OS temp
dir (see CACHE_DIR): they never change for a pinned Overture release, so every run
after the first reads them in 0.0 s and the system cleans the file up on its own
schedule. Nothing else is cached. The fold cannot be made faster from here: the
archive is time-optimised (each 45 x 45 px chunk is 2,160 hours deep), so any window
downloads the whole current 90-day layer, ~0.44 GB today and more as it fills, and
that is bandwidth-bound wherever you run it (dynamical.org, source.coop and the AWS
Open Data bucket are the same us-west-2 objects). A precomputed county-hour cube would
make any window sub-second and reach back to 2014, and the fold here is its recipe;
for a demonstration, computing it on the fly is the point, and thirty seconds is fine.

Two engines, same split as the rest of the repo: DuckDB does geometry (dissolve,
polyfill, the daily roll-up), DataFusion does the fold and the join, and DuckDB's
replacement scan is NOT used from cell bodies (marimo mangles underscore locals, so
`con.register` throughout). Full record and the render-route discussion in
docs/hrrr-counties-notes.md.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import io
    import json
    import math
    import struct

    import anywidget
    import duckdb
    import marimo as mo
    import numpy as np
    import obstore
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq
    import traitlets
    import xarray as xr
    from arro3.core import Array as ArroArray, Table as ArroTable
    from geoarrow.rust.core import from_wkb, multipolygon
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from xarray_sql import XarrayContext

    return (
        ArroArray,
        ArroTable,
        S3Store,
        XarrayContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        duckdb,
        from_wkb,
        gzip,
        io,
        json,
        math,
        mo,
        multipolygon,
        np,
        obstore,
        pa,
        pa_ipc,
        pq,
        struct,
        traitlets,
        xr,
    )


@app.cell
def _():
    # ------------------------------------------------------------------ the weather
    # "analysis": dynamical.org's hourly HRRR analysis, icechunk v2 in the AWS Open Data
    #   bucket (not on source.coop), time-optimised chunks (2160 h x 45 x 45 px).
    # "forecast": the 48-hour forecast on source.coop, plain Zarr v3, one init at a time
    #   (chunks 1 init x 49 leads x 265 x 300), the newest init is used.
    SOURCE = "analysis"
    VAR = "temperature_2m"  # degC in the store; any (time, y, x) variable works
    UNITS = "°C"
    DAYS = 7  # opening window: the last DAYS days ending at the newest hour
    # Window form limits per frame mode. Hourly frames are the film; 14 days is 336
    # frames. Daily modes roll hourly up in DuckDB, so their limit is the READ: a
    # 90-day window is one full store chunk deep (2,160 h, measured 149 s).
    HOURLY_MAX_DAYS = 14
    DAILY_MAX_DAYS = 92

    ANALYSIS_BUCKET = "dynamical-noaa-hrrr"
    ANALYSIS_PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
    FORECAST_BUCKET = "us-west-2.opendata.source.coop"
    FORECAST_PREFIX = "dynamical/noaa-hrrr-forecast-48-hour/v0.1.0.zarr"

    # ------------------------------------------------------------------ the fold
    # Res 7 as asked: finer than the 3 km pixel (1.00 px per cell, measured), so this
    # is a relabel and the polyfill decides which pixel belongs to which county. Res 6
    # holds 4.2 px per cell and is the first thing to try if anything here is slow.
    RES = 7

    # ------------------------------------------------------------------ boundaries
    # Overture's PMTiles build of the pinned release; the same object, box, zoom and
    # CONUS filter as xsql-deforest-conus-counties.py (counties first appear at z8).
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    COUNTY_Z = 8
    BOX = (-124.8, 24.4, -66.9, 49.5)
    NOT_CONUS = {"AK", "HI"}
    # Disk cache for the dissolved counties, which never change between runs: ~7.3 s
    # of the ~30 s before the map (1,008 ranged GETs + the Python MVT decode; the
    # dissolve itself is 0.1 s). In the OS temp dir, so the system cleans it up
    # (Stephen: tmp, not .cache). None turns it off.
    import tempfile as _tempfile

    CACHE_DIR = str(_tempfile.gettempdir()) + "/x-sql-marimo"

    # ------------------------------------------------------------------ the film
    # Diverging ramp on a blue <-> yellow/orange axis (protan-safe: no red leg, no
    # red-vs-green pair), pale at the pivot. One ramp for the whole film, or the
    # animation lies: the pivot is the median of every value in the slice and the span
    # is symmetric to the wider of p2/p98, unless PIVOT/SPAN are set to numbers.
    RAMP_STOPS = ["#08306b", "#2f79b5", "#9ecae1", "#f2f0e6", "#fee391", "#fdb034", "#d94801"]
    PIVOT = None  # degC, or None for the slice median
    SPAN = None  # degC either side of the pivot, or None for the p2/p98 rule
    FPS = 8
    MAP_HEIGHT = 620
    return (
        ANALYSIS_BUCKET,
        ANALYSIS_PREFIX,
        BOX,
        CACHE_DIR,
        COUNTY_Z,
        DAILY_MAX_DAYS,
        DAYS,
        FORECAST_BUCKET,
        FORECAST_PREFIX,
        FPS,
        HOURLY_MAX_DAYS,
        MAP_HEIGHT,
        NOT_CONUS,
        OVERTURE_RELEASE,
        PIVOT,
        PM_BUCKET,
        PM_PATH,
        RAMP_STOPS,
        RES,
        SOURCE,
        SPAN,
        UNITS,
        VAR,
    )


@app.cell
def _(duckdb):
    # DuckDB does the geometry (tile-seam dissolve, polyfill) and the daily roll-up;
    # DataFusion does the fold and the join, per the engine benchmark in
    # xsql-duckdb-nlcd-h3.py.
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _(anywidget, traitlets):
    class CountyFilm(anywidget.AnyWidget):
        """deck.gl + @geoarrow/deck.gl-layers, browser-side clock, minimal HUD.

        Kernel -> browser only: `counties` (one GeoArrow IPC stream: geometry, name,
        state, interleaved coords, the layout the JS layers want), `frames` (Float32Array
        of F x N values, frame-major, NaN = no data) and `config` (JSON: labels, ramp
        bounds, stops, fps, units, height, title). NOTHING crosses back: the first HUD
        synced the clicked county to a trait and marimo answered a widget value change
        by re-running the cells that reference the widget, which pulled the fullscreen
        element out of the DOM and rebuilt deck (Stephen: buttons "freeze the notebook
        and exit fullscreen"). Bytes traits kernel -> browser are how lonboard ships
        its tables, so that direction is proven.

        The HUD is inside the map element so the ELEMENT's own browser fullscreen (⛶
        or F, `mapEl.requestFullscreen()`, not marimo's) carries it: one panel top-left
        (title, legend, county mean for the frame, and a clicked county's line only
        after a click) with its own hide toggle, and the transport across the bottom
        (step / play / step, slider with UTC-day ticks, timestamp, fps, fullscreen).
        Space plays, arrows step, H hides. Deck polls its canvas size every frame, so
        the fullscreen resize needs no handler.

        Clicks are picked EXPLICITLY on pointerup with `deck.pickObject` (a press that
        starts on the HUD or moves more than 4 px is not a click) rather than through
        deck's onClick, which did nothing on the first flight inside marimo's shadow DOM.

        Every esm.sh import pins its `?deps` so that all of them resolve to ONE
        @deck.gl/core module (esm.sh hashes the variant by the deps list; two cores
        would mean two luma devices and layers that fail to init), and EVERY deck
        package is pinned to the same version, the newest, because esm.sh resolves the
        packages' own caret ranges (geo-layers -> mesh-layers@^9.1.0) to the newest
        release: pinned at 9.1.14 the first flight died on mesh-layers 9.3 asking core
        9.1 for `phongMaterial`. The whole module graph was crawled (200 modules) and
        holds exactly one core, one luma set, one geo-layers; re-crawl if any version
        moves (docs/hrrr-counties-notes.md has the crawler).
        """

        _esm = r"""
        import {Deck} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {GeoArrowPolygonLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/aggregation-layers@9.3.10,@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";

        const CSS = `
          .cf { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .cf * { box-sizing: border-box; }
          .cf .cf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .cf .cf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .cf .cf-map:fullscreen { height: 100vh !important; width: 100vw; }
          .cf .cf-hud { position: absolute; z-index: 5; }
          .cf .cf-hud.cf-tl { top: .6rem; left: .6rem; width: 21rem; max-width: calc(100% - 1.2rem); }
          .cf .cf-hud.cf-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .cf .cf-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .cf .cf-panel .cf-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }
          .cf .cf-panel .cf-ttl { font-weight: 600; }
          .cf .cf-panel .cf-sub { color: var(--dim); display: block; margin-top: .1rem; }
          .cf .cf-legend { display: flex; align-items: center; gap: .45rem; margin-top: .5rem; }
          .cf .cf-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .cf .cf-row { display: flex; justify-content: space-between; align-items: baseline; gap: .6rem; margin-top: .45rem; }
          .cf .cf-row .cf-v { font-size: 16px; }
          .cf .cf-row .cf-k { color: var(--dim); }
          .cf .cf-county { margin-top: .4rem; display: none; }
          .cf.cf-picked .cf-county { display: block; }
          .cf .cf-chart { display: block; width: 100%; height: 96px; margin-top: .3rem; cursor: crosshair; }
          .cf.cf-collapsed .cf-body { display: none; }
          .cf .cf-toggle, .cf .cf-clear { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .cf .cf-toggle:hover, .cf .cf-clear:hover { color: var(--ink); }
          .cf .cf-transport { display: flex; align-items: center; gap: .55rem; }
          .cf .cf-stamp { font-size: 15px; min-width: 11.5rem; }
          .cf .cf-stamp small { display: block; font-size: 10px; color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
          .cf .cf-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .cf .cf-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .cf .cf-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .cf input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .cf button.cf-b, .cf select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .cf button.cf-b:hover, .cf select:hover { background: #2b323b; }
          .cf button:focus-visible, .cf select:focus-visible, .cf input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .cf .cf-dim { color: var(--dim); }
          .cf .cf-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; }
          @media (max-width: 720px) { .cf .cf-stamp { min-width: 0; } .cf .cf-hud.cf-tl { width: calc(100% - 1.2rem); } }
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
        const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

        function render({model, el}) {
          el.innerHTML = "";
          const root = document.createElement("div"); root.className = "cf";
          root.innerHTML = `<style>${CSS}</style>
            <div class="cf-map">
              <div class="cf-hud cf-tl"><div class="cf-card cf-panel">
                <div class="cf-head"><span><span class="cf-ttl"></span><span class="cf-sub"></span></span><button class="cf-toggle" title="hide / show (H)">hide</button></div>
                <div class="cf-body">
                  <div class="cf-legend"><span class="cf-num cf-lo"></span><div class="cf-grad"></div><span class="cf-num cf-hi"></span></div>
                  <div class="cf-row"><span class="cf-k">county mean</span><span class="cf-num cf-v cf-mean">–</span></div>
                  <div class="cf-county">
                    <div class="cf-row"><span class="cf-k cf-cname">–</span><span><span class="cf-num cf-v cf-cval">–</span> <button class="cf-clear" title="clear">×</button></span></div>
                    <canvas class="cf-chart" height="96"></canvas>
                  </div>
                  <div class="cf-dim cf-hint">click a county for its value and line · space plays · ← → step</div>
                </div>
              </div></div>
              <span class="cf-ruler cf-num"></span>
              <div class="cf-hud cf-bl"><div class="cf-card cf-transport">
                <button class="cf-b cf-prev" title="step back (←)">‹</button>
                <button class="cf-b cf-play" title="play / pause (space)">▶</button>
                <button class="cf-b cf-next" title="step forward (→)">›</button>
                <div class="cf-track"><div class="cf-ticks"></div><input class="cf-frame" type="range" min="0" max="0" value="0" step="1" aria-label="frame"></div>
                <div class="cf-stamp cf-num"><small class="cf-stampk">frame</small><span class="cf-stampv">–</span></div>
                <select class="cf-fps" title="frames per second"><option>2</option><option>4</option><option>6</option><option>8</option><option>12</option><option>24</option></select>
                <button class="cf-b cf-full" title="fullscreen (F)">⛶</button>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".cf-map"), playBtn = q(".cf-play"), slider = q(".cf-frame"), ticks = q(".cf-ticks"),
                stampV = q(".cf-stampv"), stampK = q(".cf-stampk"), fpsSel = q(".cf-fps"), grad = q(".cf-grad"),
                loEl = q(".cf-lo"), hiEl = q(".cf-hi"), chart = q(".cf-chart"), ruler = q(".cf-ruler"),
                ttl = q(".cf-ttl"), sub = q(".cf-sub"), meanEl = q(".cf-mean"), cname = q(".cf-cname"), cval = q(".cf-cval");

          let table = null, N = 0, F = 0, frames = null, colors = null, names = [], states = [];
          let cfg = {}, frame = 0, playing = false, timer = null, deck = null, selected = -1, lut = null;
          let order = null;  // per-frame cache: sorted indices for the current frame
          let orderFrame = -1;
          let means = null;  // county mean per frame, computed once per film
          const HOME = {longitude: -96.5, latitude: 38.3, zoom: 3.8, minZoom: 2, maxZoom: 11};

          const fmt = v => Number.isFinite(v) ? v.toFixed(1) + (cfg.units || "") : "no data";
          const val = (f, i) => frames ? frames[f * N + i] : NaN;
          const rgbAt = v => {
            let t = (v - cfg.lo) / (cfg.hi - cfg.lo); t = t < 0 ? 0 : t > 1 ? 1 : t;
            const i = Math.round(t * 255) * 3; return `rgb(${lut[i]},${lut[i+1]},${lut[i+2]})`;
          };

          // Geometry index for picking IN JS: deck's GPU picking returned null for every
          // click on the flights here (ruler read "pick: none (null)"), so a click is
          // unprojected to lon/lat and tested against the county rings the browser
          // already holds: bbox reject, then even-odd over every ring of every polygon
          // (holes fall out of even-odd). ~ms for 3,108 multipolygons.
          let geo = null;
          function indexGeometry() {
            const d = table.getChild("geometry").data[0];       // multipolygon: list<polygon>
            const polyD = d.children[0];                        // polygon: list<ring>
            const ringD = polyD.children[0];                    // ring: list<coord>
            const coordD = ringD.children[0];                   // coord: fixed_size_list<f64, 2>
            const xy = coordD.children[0].values;               // interleaved x y
            const mpOff = d.valueOffsets, polyOff = polyD.valueOffsets, ringOff = ringD.valueOffsets;
            const bbox = new Float64Array(N * 4), polys = new Array(N);
            for (let i = 0; i < N; i++) {
              let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
              const ps = [];
              for (let p = mpOff[d.offset + i]; p < mpOff[d.offset + i + 1]; p++) {
                const rings = [];
                for (let r = polyOff[p]; r < polyOff[p + 1]; r++) {
                  const s = ringOff[r], e = ringOff[r + 1];
                  rings.push([s, e]);
                  for (let c = s; c < e; c++) { const x = xy[2 * c], y = xy[2 * c + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
                }
                ps.push(rings);
              }
              bbox[4 * i] = x0; bbox[4 * i + 1] = y0; bbox[4 * i + 2] = x1; bbox[4 * i + 3] = y1;
              polys[i] = ps;
            }
            geo = {xy, bbox, polys};
          }
          function countyAt(lng, lat) {
            if (!geo) return -1;
            const {xy, bbox, polys} = geo;
            for (let i = 0; i < N; i++) {
              if (lng < bbox[4 * i] || lng > bbox[4 * i + 2] || lat < bbox[4 * i + 1] || lat > bbox[4 * i + 3]) continue;
              for (const rings of polys[i]) {
                let inside = false;
                for (const [s, e] of rings) {
                  for (let a = s, b = e - 1; a < e; b = a++) {
                    const xa = xy[2 * a], ya = xy[2 * a + 1], xb = xy[2 * b], yb = xy[2 * b + 1];
                    if ((ya > lat) !== (yb > lat) && lng < (xb - xa) * (lat - ya) / (yb - ya) + xa) inside = !inside;
                  }
                }
                if (inside) return i;
              }
            }
            return -1;
          }
          function loadTable() {
            const u8 = bytesOf(model.get("counties"));
            if (!u8 || !u8.length) return;
            table = arrow.tableFromIPC(u8);
            N = table.numRows;
            names = table.getChild("name").toArray();
            states = table.getChild("state").toArray();
            try { indexGeometry(); } catch (e) { geo = null; ruler.textContent = "geometry index: " + e.message; }
          }
          function recolor() {
            if (!frames) return;
            const a = 235;
            colors = colors && colors.length === F * N * 4 ? colors : new Uint8Array(F * N * 4);
            const lo = cfg.lo, hi = cfg.hi;
            for (let k = 0; k < F * N; k++) {
              const v = frames[k], o = k * 4;
              if (!Number.isFinite(v)) { colors[o] = 40; colors[o + 1] = 44; colors[o + 2] = 50; colors[o + 3] = 70; continue; }
              let t = (v - lo) / (hi - lo); t = t < 0 ? 0 : t > 1 ? 1 : t;
              const i = Math.round(t * 255) * 3;
              colors[o] = lut[i]; colors[o + 1] = lut[i + 1]; colors[o + 2] = lut[i + 2]; colors[o + 3] = a;
            }
          }
          function loadFrames() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 620) + "px";
            lut = buildLut(cfg.stops || ["#08306b", "#f2f0e6", "#d94801"]);
            const u8 = bytesOf(model.get("frames"));
            if (!u8 || !u8.length || !N) { frames = null; F = 0; return; }
            // copy: the DataView's buffer offset is not guaranteed 4-byte aligned
            frames = new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
            F = Math.floor(frames.length / N);
            recolor();
            means = new Float32Array(F);
            for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let i = 0; i < N; i++) { const v = frames[f * N + i]; if (Number.isFinite(v)) { s += v; n++; } } means[f] = n ? s / n : NaN; }
            slider.max = String(Math.max(0, F - 1));
            if (frame >= F) frame = 0;
            orderFrame = -1;
            fpsSel.value = String(cfg.fps || 8);
            const stops = [];
            for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            loEl.textContent = fmt(cfg.lo); hiEl.textContent = fmt(cfg.hi);
            ttl.textContent = cfg.title || ""; sub.textContent = cfg.subtitle || "";
            stampK.textContent = cfg.frame_kind || "frame";
            // day ticks under the slider: one per label whose date part changes
            const labels = cfg.labels || [];
            let html = "";
            for (let f = 1; f < labels.length; f++) {
              const d0 = labels[f - 1].slice(0, 10), d1 = labels[f].slice(0, 10);
              if (d0 !== d1) html += `<i class="cf-day" style="left:${(f / (F - 1) * 100).toFixed(2)}%"></i>`;
            }
            ticks.innerHTML = F > 1 && labels.length > 30 ? html : "";  // hourly films only; a daily film's frames ARE the days
          }

          function colorVector(f) {
            const sub = colors.subarray(f * N * 4, (f + 1) * N * 4);
            const child = arrow.makeData({type: new arrow.Uint8(), data: sub});
            const data = arrow.makeData({type: new arrow.FixedSizeList(4, new arrow.Field("c", new arrow.Uint8(), false)), length: N, nullCount: 0, child});
            return arrow.makeVector(data);
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
            if (table && colors) {
              out.push(new GeoArrowPolygonLayer({
                id: "counties",
                data: table,
                getPolygon: table.getChild("geometry"),
                getFillColor: colorVector(frame),
                filled: true,
                stroked: false,
                pickable: false,
                _validate: false,
              }));
              if (selected >= 0 && selected < N && geo) {
                // the picked county, outlined: its rings as plain paths (a one-row
                // GeoArrow layer via table.slice drew EVERY county: the layer reads the
                // full offsets under a sliced table)
                const paths = [];
                for (const rings of geo.polys[selected]) for (const [st, en] of rings) paths.push(geo.xy.subarray(2 * st, 2 * en));
                out.push(new PathLayer({
                  id: "picked",
                  data: paths,
                  getPath: d => d,
                  positionFormat: "XY",
                  getColor: [230, 193, 74, 255],
                  getWidth: 2,
                  widthUnits: "pixels",
                  widthMinPixels: 2,
                  pickable: false,
                }));
              }
            }
            out.push(tiles("labels", "https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png", 0.6));
            return out;
          }

          function stats() {
            meanEl.textContent = means && F ? fmt(means[frame]) : "–";
            if (selected >= 0) cval.textContent = fmt(val(frame, selected));
          }

          function drawChart() {
            if (selected < 0 || !frames || F < 2) return;
            const w = chart.clientWidth || 300, h = chart.height;
            if (chart.width !== w) chart.width = w;
            const g = chart.getContext("2d");
            g.clearRect(0, 0, w, h);
            const L = 40, R = 4, T = 6, B = 14;
            const X = f => L + (w - L - R) * f / (F - 1);
            let lo = Infinity, hi = -Infinity;
            for (let f = 0; f < F; f++) { const v = val(f, selected); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
            if (!Number.isFinite(lo)) return;
            if (hi - lo < 1) { hi += .5; lo -= .5; }
            const Y = v => T + (h - T - B) * (1 - (v - lo) / (hi - lo));
            axes(g, w, h, L, R, T, B, lo, hi, Y);
            g.strokeStyle = "#e6c14a"; g.lineWidth = 1.5; g.beginPath();
            let pen = false;
            for (let f = 0; f < F; f++) { const v = val(f, selected); if (!Number.isFinite(v)) { pen = false; continue; } pen ? g.lineTo(X(f), Y(v)) : g.moveTo(X(f), Y(v)); pen = true; }
            g.stroke();
            cursor(g, X(frame), T, h - B);
            const cv = val(frame, selected);
            if (Number.isFinite(cv)) { g.fillStyle = "#ffffff"; g.beginPath(); g.arc(X(frame), Y(cv), 3, 0, 6.283); g.fill(); }
          }
          function axes(g, w, h, L, R, T, B, lo, hi, Y) {
            g.strokeStyle = "#262c35"; g.lineWidth = 1;
            g.beginPath(); g.moveTo(L, Y(lo)); g.lineTo(w - R, Y(lo)); g.moveTo(L, Y(hi)); g.lineTo(w - R, Y(hi)); g.stroke();
            g.fillStyle = "#8b929c"; g.font = "11px ui-monospace, Menlo, monospace"; g.textAlign = "right";
            g.fillText(fmt(hi), L - 4, Y(hi) + 4); g.fillText(fmt(lo), L - 4, Y(lo) + 4);
            g.font = "10px system-ui, sans-serif"; g.textAlign = "left"; g.fillText((cfg.labels?.[0] || "").slice(0, 10), L, h - 3);
            g.textAlign = "right"; g.fillText((cfg.labels?.[F - 1] || "").slice(0, 10), w - R, h - 3);
          }
          function cursor(g, x, top, bottom) { g.strokeStyle = "rgba(230,193,74,.55)"; g.lineWidth = 1; g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); }
          chart.addEventListener("click", ev => {
            if (F < 2) return;
            const r = chart.getBoundingClientRect(), L = 40, R = 4;
            const t = ((ev.clientX - r.left) - L) / (r.width - L - R);
            frame = Math.max(0, Math.min(F - 1, Math.round(t * (F - 1)))); update();
          });

          function select(i) {
            if (!(i >= 0 && i < N)) return;
            selected = i;
            root.classList.add("cf-picked");
            cname.textContent = `${names[i]}, ${states[i]}`;
            update();
          }
          q(".cf-clear").onclick = () => { selected = -1; root.classList.remove("cf-picked"); update(); };
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
            if (p && F > 1) timer = setInterval(() => {
              frame = (frame + 1) % F; update();
            }, 1000 / (parseFloat(fpsSel.value) || 8));
          }
          const step = d => { if (F) { frame = (frame + d + F) % F; update(); } };
          playBtn.onclick = () => setPlaying(!playing);
          q(".cf-prev").onclick = () => step(-1);
          q(".cf-next").onclick = () => step(1);
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          fpsSel.onchange = () => { if (playing) setPlaying(true); };
          const toggle = q(".cf-toggle");
          // "cf-collapsed", not "hidden": marimo's page CSS (Tailwind) owns `.hidden { display: none }`
          // and the widget shares the page's stylesheet, so a root class named "hidden" blanked the
          // whole widget, kicked the browser out of fullscreen and read as a frozen notebook.
          toggle.onclick = () => { root.classList.toggle("cf-collapsed"); toggle.textContent = root.classList.contains("cf-collapsed") ? "show" : "hide"; };
          q(".cf-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen?.(); };
          mapEl.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) mapEl.style.height = (cfg.height || 620) + "px"; });
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "BUTTON") return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); step(1); }
            else if (ev.key === "f" || ev.key === "F") { q(".cf-full").click(); }
            else if (ev.key === "h" || ev.key === "H") { toggle.click(); }
          });

          function boot() {
            loadTable(); loadFrames();
            deck = new Deck({
              parent: mapEl,
              initialViewState: HOME,
              controller: true,
              layers: layers(),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); },
            });
            // explicit pick on pointerup: deck's onClick did nothing on the first flight
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".cf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
            mapEl.addEventListener("pointerup", ev => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !deck) return;
              const r = mapEl.getBoundingClientRect();
              let ll = null;
              try { ll = deck.getViewports()[0].unproject([ev.clientX - r.left, ev.clientY - r.top]); }
              catch (e) { ruler.textContent = "unproject: " + e.message; return; }
              const i = countyAt(ll[0], ll[1]);
              if (i >= 0 && i !== selected) select(i);
              else { selected = -1; root.classList.remove("cf-picked"); update(); }  // click off a county, or the picked one again, clears
            }, true);
            ruler.textContent = `${N.toLocaleString()} counties · ${F} frames`;
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:counties", () => { loadTable(); loadFrames(); ruler.textContent = `${N.toLocaleString()} counties · ${F} frames`; update(); });
          model.on("change:frames", () => { loadFrames(); ruler.textContent = `${N.toLocaleString()} counties · ${F} frames`; update(); });
          model.on("change:config", () => { loadFrames(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (deck) deck.finalize(); };
        }
        export default {render};
        """
        counties = traitlets.Bytes(b"").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)

    return (CountyFilm,)


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
def _(
    ANALYSIS_BUCKET,
    ANALYSIS_PREFIX,
    FORECAST_BUCKET,
    FORECAST_PREFIX,
    S3Store,
    SOURCE,
    VAR,
    np,
    xr,
):
    # THE STORE, opened once. Nothing but metadata and the 2-D lat/lon (8 MB each) is
    # read here; the window is cut in the fold cell. The analysis is icechunk (its own
    # reader over S3, anonymous); the forecast is plain Zarr v3 through obstore, like
    # every other read in this repo. NO DASK: chunks=None leaves the store lazily
    # indexed and xarray-sql cuts it into blocks itself (see the fold cell).
    import time as _stime

    _st0 = _stime.perf_counter()
    if SOURCE == "analysis":
        import icechunk

        _storage = icechunk.s3_storage(
            bucket=ANALYSIS_BUCKET, prefix=ANALYSIS_PREFIX, region="us-west-2", anonymous=True
        )
        _sess = icechunk.Repository.open(_storage).readonly_session("main")
        _ds = xr.open_zarr(_sess.store, consolidated=False, chunks=None)
        cube_all = _ds[[VAR]].rename({"time": "t"})
        source_note = "HRRR analysis"
    else:
        from zarr.storage import ObjectStore as _ZStore

        _zs = _ZStore(
            S3Store(FORECAST_BUCKET, prefix=FORECAST_PREFIX, region="us-west-2", skip_signature=True),
            read_only=True,
        )
        _ds = xr.open_zarr(_zs, consolidated=True, chunks=None)
        _init = _ds["init_time"].values[-1]
        cube_all = (
            _ds[[VAR]].isel(init_time=-1).drop_vars("init_time", errors="ignore").rename({"lead_time": "t"})
        )
        # lead offsets -> valid times, so the frame labels read as clock time
        cube_all = cube_all.assign_coords(t=(_init + _ds["lead_time"].values).astype("datetime64[ns]"))
        source_note = f"HRRR 48 h forecast, init {np.datetime_as_string(_init.astype('datetime64[m]'))}Z"

    all_times = cube_all["t"].values.astype("datetime64[m]")
    lat = _ds["latitude"].values.astype("float64")
    lon = _ds["longitude"].values.astype("float64")
    grid_y = _ds["y"].values
    grid_x = _ds["x"].values
    store_stats = (
        f"{source_note} · grid {lat.shape[1]}x{lat.shape[0]} px · hourly {np.datetime_as_string(all_times[0])} "
        f"to {np.datetime_as_string(all_times[-1])} UTC ({all_times.size:,} steps) · open {_stime.perf_counter() - _st0:.1f}s"
    )
    return all_times, cube_all, grid_x, grid_y, lat, lon, source_note, store_stats


@app.cell
def _(DAILY_MAX_DAYS, DAYS, HOURLY_MAX_DAYS, SOURCE, all_times, mo, np, window_form):
    # THE WINDOW, resolved: the form's value once submitted, the last DAYS days before
    # that. Over the limit stops here with the reason instead of clamping silently.
    import datetime as _wdt

    _last = all_times[-1].astype("datetime64[D]").astype(_wdt.date)
    if window_form.value is None:
        _d0, _d1 = _last - _wdt.timedelta(days=DAYS - 1), _last
        slice_mode = "hourly"
    else:
        _d0, _d1 = window_form.value["dates"]
        slice_mode = window_form.value["mode"]
    if _d1 < _d0:
        _d0, _d1 = _d1, _d0
    n_days = (_d1 - _d0).days + 1
    _limit = HOURLY_MAX_DAYS if slice_mode == "hourly" else DAILY_MAX_DAYS
    mo.stop(
        n_days > _limit,
        mo.md(
            f"**{n_days} days is over the {_limit}-day limit for {slice_mode.replace('_', ' ')}.** "
            f"Shorten the window or switch to a daily mode (limit {DAILY_MAX_DAYS} days)."
        ),
    )
    t0 = np.datetime64(_d0.isoformat()).astype("datetime64[ns]")
    t1 = min(
        (np.datetime64(_d1.isoformat()) + np.timedelta64(23, "h")).astype("datetime64[ns]"),
        all_times[-1].astype("datetime64[ns]"),
    )
    if SOURCE != "analysis":
        # the forecast is one init's 49 leads; the window form does not apply
        t0, t1 = all_times[0].astype("datetime64[ns]"), all_times[-1].astype("datetime64[ns]")
    window_note = (
        f"{np.datetime_as_string(t0, unit='m').replace('T', ' ')}Z to "
        f"{np.datetime_as_string(t1, unit='m').replace('T', ' ')}Z"
    )
    return n_days, slice_mode, t0, t1, window_note


@app.cell
def _(RES, con, coordinates_to_cells, counties, grid_x, grid_y, lat, lon, np, pa):
    # PIXEL -> CELL -> COUNTY, ONCE. The grid never moves, so this static lookup is what
    # every frame joins against. Cell per pixel from the store's own lat/lon (h3ronpy,
    # sub-second for 1.9M points); county cells by DuckDB polyfill, 'center' rule, so
    # each cell (and so each pixel) is in exactly one county and the join is the CONUS
    # clip. h3_polygon_wkb_to_cells_experimental takes a POLYGON and raises on a
    # MultiPolygon, which every dissolved county is, so ST_Dump splits first.
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
        SELECT DISTINCT id, hex FROM filled
        """,
        params=[int(RES)],
    ).to_arrow_table()
    con.unregister("conus_divs")
    _t_fill = _ptime.perf_counter() - _pt0

    _hex = np.asarray(coordinates_to_cells(lat.ravel(), lon.ravel(), int(RES)))
    _pix = pa.table(
        {
            "y": pa.array(np.repeat(grid_y, grid_x.size)),
            "x": pa.array(np.tile(grid_x, grid_y.size)),
            "hex": pa.array(_hex),
        }
    )
    con.register("pix", _pix)
    con.register("county_cells", _mapping)
    pix2c = con.sql(
        "SELECT p.y, p.x, m.id FROM pix p JOIN county_cells m USING (hex)"
    ).to_arrow_table()
    n_with_pixel = con.sql("SELECT count(DISTINCT id) FROM pix p JOIN county_cells m USING (hex)").fetchone()[0]
    con.unregister("pix")
    con.unregister("county_cells")
    pix_stats = (
        f"{_mapping.num_rows:,} res {RES} county cells (polyfill {_t_fill:.1f}s) · "
        f"{pix2c.num_rows:,} of {_hex.size:,} pixels in a county · "
        f"{n_with_pixel:,} of {counties.num_rows:,} counties catch a pixel · "
        f"{_ptime.perf_counter() - _pt0:.1f}s"
    )
    return pix2c, pix_stats


@app.cell
def _():
    # Kernel-side memo across form submits: the last folded window and its table.
    HOLD = {"key": None, "county_hour": None, "stats": ""}
    return (HOLD,)


@app.cell
def _(HOLD, SOURCE, VAR, XarrayContext, cube_all, pix2c, t0, t1):
    # THE FOLD AND THE JOIN, ONE STATEMENT, STRAIGHT OFF THE CUBE. xarray-sql exposes the
    # lazy (t, y, x) window as a table; the join to pix2c on the grid coordinates is the
    # H3 fold (each pixel already carries its cell, and its cell carries its county) and
    # the group by is the zonal mean per county per hour. The block is the WHOLE time
    # window x one 45x45 store column, so each 2,160 h store chunk is decoded once (a
    # narrower block grid decoded them once per block: 111.7 s against 20.9 s for the
    # same 168 h). DataFusion pulls the columns in parallel; the analysis is fetch-bound
    # (~20 s for a CONUS window up to ~10 days) and the aggregate is nearly free.
    #
    # Memoised on the window: re-submitting the same dates, or switching hourly/daily,
    # never refetches.
    import time as _ftime

    _key = (SOURCE, str(t0), str(t1))
    if HOLD["key"] == _key and HOLD["county_hour"] is not None:
        county_hour = HOLD["county_hour"]
        fold_stats = HOLD["stats"] + " (memo)"
    else:
        _ft0 = _ftime.perf_counter()
        _cube = cube_all.sel(t=slice(t0, t1))
        _hours = int(_cube.sizes["t"])
        _chunks = {"t": _hours, "y": 45, "x": 45} if SOURCE == "analysis" else {"t": _hours, "y": 265, "x": 300}
        ctx = XarrayContext()
        ctx.from_arrow(pix2c, name="pix2c")
        ctx.from_dataset("cube", _cube, chunks=_chunks)
        county_hour = ctx.sql(f"""
            SELECT t, id, avg(CAST({VAR} AS DOUBLE)) AS v, count(*) AS px
            FROM cube JOIN pix2c USING (y, x)
            WHERE {VAR} = {VAR}
            GROUP BY 1, 2
        """).to_arrow_table()
        fold_stats = (
            f"{_hours} hours · {county_hour.num_rows:,} county-hour rows · "
            f"fold + join {_ftime.perf_counter() - _ft0:.1f}s"
        )
        HOLD["key"], HOLD["county_hour"], HOLD["stats"] = _key, county_hour, fold_stats
    return county_hour, fold_stats


@app.cell
def _(PIVOT, SPAN, con, counties, county_hour, np, slice_mode):
    # THE FRAME MATRIX: F frames x N counties, float32, NaN where a county has no
    # pixel, in the counties table's row order (the widget indexes by row). Hourly is
    # the table as it comes; the daily slices roll it up in DuckDB on UTC days (the
    # first and last day of a window are partial if the window is, labelled as days
    # all the same). One ramp for the whole film: pivot at the median, span to the
    # wider of p2/p98, unless PIVOT/SPAN pin them.
    con.register("ch", county_hour)
    if slice_mode == "hourly":
        _tbl = con.sql("SELECT t AS f, id, v FROM ch ORDER BY f").to_arrow_table()
        _labels = [np.datetime_as_string(t, unit="m").replace("T", " ") + "Z" for t in np.unique(_tbl["f"].to_numpy())]
        frame_kind = "hour (UTC)"
    else:
        _agg = "avg" if slice_mode == "daily_mean" else "max"
        _tbl = con.sql(
            f"SELECT date_trunc('day', t) AS f, id, {_agg}(v) AS v FROM ch GROUP BY 1, 2 ORDER BY f"
        ).to_arrow_table()
        _labels = [np.datetime_as_string(t, unit="D") for t in np.unique(_tbl["f"].to_numpy())]
        frame_kind = "UTC day, " + ("mean" if _agg == "avg" else "max")
    con.unregister("ch")

    _fkeys = np.unique(_tbl["f"].to_numpy())
    _fi = np.searchsorted(_fkeys, _tbl["f"].to_numpy())
    _ids = counties["id"].to_pylist()
    _pos = {i: k for k, i in enumerate(_ids)}
    _ci = np.fromiter((_pos[i] for i in _tbl["id"].to_pylist()), dtype=np.int64, count=_tbl.num_rows)
    frames = np.full((_fkeys.size, len(_ids)), np.nan, dtype=np.float32)
    frames[_fi, _ci] = _tbl["v"].to_numpy().astype(np.float32)
    frame_labels = _labels

    _vals = frames[np.isfinite(frames)]
    _mid = float(np.median(_vals)) if PIVOT is None else float(PIVOT)
    _span = (
        float(max(_mid - np.percentile(_vals, 2), np.percentile(_vals, 98) - _mid))
        if SPAN is None
        else float(SPAN)
    )
    ramp_lo, ramp_mid, ramp_hi = _mid - _span, _mid, _mid + _span
    frame_stats = (
        f"{frames.shape[0]} frames x {frames.shape[1]} counties ({slice_mode.replace('_', ' ')}) · "
        f"ramp {ramp_lo:.1f} / {ramp_mid:.1f} / {ramp_hi:.1f}"
    )
    return frame_kind, frame_labels, frame_stats, frames, ramp_hi, ramp_lo, ramp_mid


@app.cell
def _(DAILY_MAX_DAYS, DAYS, HOURLY_MAX_DAYS, all_times, mo):
    # THE WINDOW FORM. A form, not live controls: every submit is a fold (~20 s for the
    # analysis), so nothing runs until "load window". Dates are UTC days, inclusive; the
    # end day is clipped to the newest hour in the store. Limits per mode keep the
    # frame count and the read honest: hourly is capped at HOURLY_MAX_DAYS (336 frames
    # at 14), daily at DAILY_MAX_DAYS (a 90-day window is one full store chunk deep,
    # measured 149 s).
    import datetime as _dt

    _last = all_times[-1].astype("datetime64[D]").astype(_dt.date)
    _first = all_times[0].astype("datetime64[D]").astype(_dt.date)
    window_form = (
        mo.md("{dates} &nbsp;&nbsp; {mode}")
        .batch(
            dates=mo.ui.date_range(
                start=_first,
                stop=_last,
                value=(_last - _dt.timedelta(days=DAYS - 1), _last),
                label="window (UTC days, inclusive)",
            ),
            mode=mo.ui.dropdown(
                options={
                    f"hourly, as it comes (max {HOURLY_MAX_DAYS} days)": "hourly",
                    f"daily mean (max {DAILY_MAX_DAYS} days)": "daily_mean",
                    f"daily max (max {DAILY_MAX_DAYS} days)": "daily_max",
                },
                value=f"hourly, as it comes (max {HOURLY_MAX_DAYS} days)",
                label="frames",
            ),
        )
        .form(submit_button_label="load window", bordered=False)
    )
    window_form
    return (window_form,)


@app.cell
def _(
    ArroArray,
    ArroTable,
    CountyFilm,
    con,
    counties,
    from_wkb,
    io,
    multipolygon,
    pa,
    pa_ipc,
):
    # THE WIDGET, BUILT ONCE with the county geometry and nothing else. The frames and
    # the config are set from the wiring cell below, so a window or slice change never
    # rebuilds the map (the deck instance and its GPU buffers survive; only the colour
    # attribute is re-uploaded, browser side).
    #
    # Geometry: dissolved MultiPolygons (ST_Multi so from_wkb sees one type), converted
    # to GeoArrow with INTERLEAVED coords, which is what @geoarrow/deck.gl-layers reads
    # (lonboard converts to interleaved before serialising for the same reason), then
    # one Arrow IPC stream with the geoarrow.multipolygon extension metadata on the
    # field, one record batch (the colour vector must match the batch structure).
    con.register("cnt", counties)
    _c = con.sql(
        "SELECT id, name, region, CAST(ST_AsWKB(ST_Multi(ST_GeomFromWKB(wkb))) AS BLOB) AS wkb FROM cnt"
    ).to_arrow_table()
    con.unregister("cnt")
    assert _c["id"].to_pylist() == counties["id"].to_pylist()
    _geom = ArroArray.from_arrow(
        from_wkb(
            _c["wkb"].combine_chunks(),
            to_type=multipolygon("xy", coord_type="interleaved", crs="EPSG:4326"),
        )
    )
    _tbl = pa.table(
        ArroTable.from_arrays(
            [
                _geom,
                ArroArray.from_arrow(_c["name"].combine_chunks()),
                ArroArray.from_arrow(_c["region"].combine_chunks()),
            ],
            names=["geometry", "name", "state"],
        )
    ).combine_chunks()
    _sink = io.BytesIO()
    with pa_ipc.new_stream(_sink, _tbl.schema) as _w:
        _w.write_table(_tbl)
    film = CountyFilm(counties=_sink.getvalue())
    film
    return (film,)


@app.cell
def _(
    FPS,
    MAP_HEIGHT,
    RAMP_STOPS,
    UNITS,
    VAR,
    film,
    fold_stats,
    frame_kind,
    frame_labels,
    frame_stats,
    frames,
    json,
    n_days,
    ramp_hi,
    ramp_lo,
    ramp_mid,
    source_note,
    window_note,
):
    # THE WIRING: re-runs on every window/slice change and only pushes JSON + bytes at
    # the existing widget. Config first, then frames: the JS rebuilds its colour table
    # on the frames change and reads the config it already has.
    film.config = json.dumps(
        {
            "labels": frame_labels,
            "lo": ramp_lo,
            "mid": ramp_mid,
            "hi": ramp_hi,
            "stops": RAMP_STOPS,
            "units": UNITS,
            "fps": FPS,
            "height": MAP_HEIGHT,
            "title": f"{VAR.replace('_', ' ')} · {source_note}",
            "subtitle": f"{window_note} · {n_days} days · {frame_kind}",
            "frame_kind": frame_kind,
            "meta": f"{fold_stats} · {frame_stats}",
            "autoplay": False,
        }
    )
    film.frames = frames.tobytes()
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
