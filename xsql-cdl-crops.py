# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "duckdb>=1.5.5",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "pyarrow>=25.0.0",
#     "numpy",
#     "anywidget>=0.9",
#     "lonboard>=0.16.0",
#     "altair>=5.4",
# ]
# ///
"""USDA Cropland Data Layer, queried as SQL in marimo, drawn by lonboard from DuckDB.

The store is chill/usda-cropland-data-layer on source.coop: the official CDL rasters
(2008-2025, 30 m, EPSG:5070, uint8 class codes, bit-identical to NASS's GeoTIFFs)
reformatted into one icechunk repo with a block-majority multiscale pyramid
(30m/2x .. 30m/256x) and the class names/colors embedded in the array attrs.

This notebook is DUCKDB-ONLY, a first for the repo alongside the archived flood
notebook: the cube is categorical on a fixed Albers grid, so there is no mean-fold
and no H3, and the h3ronpy-in-DataFusion UDF has no role to play (Stephen,
2026-08-20: "we dont necessarily need to use h3"). xarray-sql 0.4.0rc1's
`xql.register(con, name, ds, chunks=...)` exposes every pyramid level as a DuckDB
table, and from there the notebook body is marimo SQL cells: crop area time series,
a crop rotation matrix (PIVOT), pixel counting as acreage. The map is pixel squares
built IN SQL (ST_MakeEnvelope in EPSG:5070, ST_Transform to 4326, colors joined
from a classes table built off the store's own attrs) and handed to
lonboard's PolygonLayer.from_duckdb; the camera picks the pyramid level.

Measured 2026-08-20 (home link): CONUS class histogram for one year at 64x 0.4 s;
full 18-year CONUS scan at 64x 1.3 s; corn/soy rotation self-join CONUS 0.6 s;
native 30 m 20x20 km window across 3 years 0.9 s (pushdown against the 304 GB
array); CONUS cropland pixel squares at 64x (320k polygons) 0.5 s SQL + 0.7 s layer.

Palette: the official NASS class colors ship in the attrs and are kept in the
classes table, but Cotton is #FF2525 pure red beside Soybeans #256F00 green, a
protan-fail pair, so the DEFAULT palette remaps the red-dominant classes onto a
blue/purple/cyan cycle and keeps everything else official. `hex_official` is the
untouched column if anyone wants NASS's exact look.

Area numbers from pyramid levels are approximate: the pyramid is block-majority,
so dominant classes overcount (corn at 64x reads ~119M acres against ~90M planted).
Trends and transitions are honest at a fixed level; absolute acreage wants the
native array over a window.

Run (rc venv, same as heat domes):
  uv run --project xarray-sql-multi-backend-test marimo edit xsql-cdl-crops.py
or self-contained:
  uv run marimo edit xsql-cdl-crops.py --sandbox
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import json
    import math
    import threading
    import time

    import altair as alt
    import anywidget
    import duckdb
    import icechunk
    import xarray as xr
    import traitlets
    import xarray_sql as xql
    import urllib.parse
    import urllib.request

    from lonboard import Map, PolygonLayer
    from lonboard.basemap import CartoStyle, MaplibreBasemap

    import marimo as mo

    return (
        CartoStyle,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        alt,
        anywidget,
        asyncio,
        duckdb,
        icechunk,
        json,
        math,
        mo,
        threading,
        time,
        traitlets,
        urllib,
        xql,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # USDA Cropland Data Layer, in SQL

    Every query below runs on **DuckDB**, against the icechunk store's pyramid
    levels registered as tables by **xarray-sql** (`cdl_1` is the native 30 m
    array, `cdl_64` is the 1.92 km majority overview, and so on). The map is a
    SQL query too: pixel squares from `ST_MakeEnvelope`, reprojected and
    colored in the query, drawn by lonboard straight from the relation. Zoom
    in and the camera walks down the pyramid toward the native 30 m pixels.

    Data: USDA NASS CDL 2008-2025, US public domain, via
    [source.coop/chill/usda-cropland-data-layer](https://source.coop/chill/usda-cropland-data-layer).
    30 m for 2024+ is NASS's own resampling of the native 10 m product.
    Pyramid counts are block-majority approximations; dominant crops read high.
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    BUCKET = "chill"
    PREFIX = "usda-cropland-data-layer/v0.1.0.icechunk"
    ENDPOINT = "https://data.source.coop"

    LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # pyramid factor; pixel = 30*k m
    YEARS = list(range(2008, 2026))
    YEAR0 = 2025

    ANALYSIS_K = 64          # level for the CONUS-wide SQL cells (1.3 s full scan)
    TOP_N = 10               # crops in the time series chart
    ROT_N = 10               # crops in the rotation matrix
    PIX_KM2 = (0.03 * ANALYSIS_K) ** 2   # km^2 per pixel at ANALYSIS_K
    ACRES_PER_KM2 = 247.10538

    PX_PER = 1.0             # level floor: largest k with pixel <= PX_PER screen px
    ROW_BUDGET = 420_000     # max pixel squares per serve; over it, coarsen a level
    MARGIN = 0.35            # fold box slack beyond the viewport (pan/zoom headroom)
    VIEW_W, VIEW_H = 1400, 700   # the usual guess; no ruler in v1
    HOME = {"longitude": -96.9, "latitude": 38.8, "zoom": 3.6}  # all of CONUS in view

    HOLD: dict = {}
    return (
        ACRES_PER_KM2,
        ANALYSIS_K,
        BUCKET,
        ENDPOINT,
        HOLD,
        HOME,
        LEVELS,
        MARGIN,
        PIX_KM2,
        PREFIX,
        PX_PER,
        ROT_N,
        ROW_BUDGET,
        TOP_N,
        VIEW_H,
        VIEW_W,
        YEAR0,
        YEARS,
    )


@app.cell
def _(
    BUCKET,
    ENDPOINT,
    LEVELS,
    PREFIX,
    duckdb,
    icechunk,
    threading,
    xql,
    xr,
):
    # ---- open the store, register every pyramid level as a DuckDB table ----
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint_url=ENDPOINT,
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    _repo = icechunk.Repository.open(storage)
    _session = _repo.readonly_session("main")

    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    # one connection for everything (cursors do not see xql's registrations);
    # the wiring serializes on this lock
    con_lock = threading.Lock()

    DS = {}
    for _k in LEVELS:
        _grp = "30m" if _k == 1 else f"30m/{_k}x"
        _ds = xr.open_zarr(_session.store, group=_grp, chunks=None)
        DS[_k] = _ds
        # block layout: whole-plane per year at coarse levels (we scan them whole),
        # 2048^2 tiles at fine levels so x/y predicates prune fragments
        if _k >= 32:
            _chunks = {"year": 1, "y": _ds.sizes["y"], "x": _ds.sizes["x"]}
        else:
            _chunks = {"year": 1, "y": 2048, "x": 2048}
        xql.register(con, f"cdl_{_k}", _ds, chunks=_chunks)

    # ---- classes table from the store's own attrs ---------------------------
    _at = DS[1]["crop_type"].attrs
    _names, _colors = _at["class_names"], _at["class_colors"]

    def _noncrop(name):
        if name.startswith("Developed"):
            return True
        return name in {
            "Background", "Clouds/No Data", "Water", "Open Water",
            "Perennial Ice/Snow", "Barren", "Forest", "Deciduous Forest",
            "Evergreen Forest", "Mixed Forest", "Shrubland",
            "Grassland/Pasture", "Grass/Pasture", "Woody Wetlands",
            "Herbaceous Wetlands", "Wetlands", "Nonag/Undefined",
        }

    def _rgb(hexs):
        return int(hexs[1:3], 16), int(hexs[3:5], 16), int(hexs[5:7], 16)

    # protan-safe default palette: remap red-dominant classes (red is the weak
    # leg; cotton #FF2525 next to soybean green fails) onto a blue/purple cycle
    _SAFE_CYCLE = ["#3F6BD6", "#8E44AD", "#00B8D4", "#D633C4",
                   "#5C6BC0", "#0091EA", "#7C4DFF", "#6A1B9A"]
    _i = 0
    _rows = []
    for _code in sorted(_names, key=int):
        _nm, _hx = _names[_code], _colors[_code]
        _r, _g, _b = _rgb(_hx)
        _safe = _hx
        if _r >= 170 and _g <= 100 and _b <= 110:
            _safe = _SAFE_CYCLE[_i % len(_SAFE_CYCLE)]
            _i += 1
        _sr, _sg, _sb = _rgb(_safe)
        _rows.append((int(_code), _nm, _hx, _safe, _sr, _sg, _sb, _noncrop(_nm)))

    con.sql(
        "CREATE TABLE classes(code UTINYINT, name VARCHAR, hex_official VARCHAR,"
        " hex VARCHAR, r UTINYINT, g UTINYINT, b UTINYINT, noncrop BOOLEAN)"
    )
    con.executemany("INSERT INTO classes VALUES (?,?,?,?,?,?,?,?)", _rows)

    # A SECOND connection for the map serve path. One duckdb connection cannot
    # serve two interleaved consumers: marimo's SQL cells hold streaming Arrow
    # results open while the serve's count query fetches, and duckdb raises
    # "Can't 'FetchRaw' from ArrowQueryResult" (seen on the first load, the
    # initial serve racing the analytics cells). mcon carries its own
    # registrations and classes copy; con_lock serializes serve vs analyze on it.
    mcon = duckdb.connect()
    mcon.sql("LOAD spatial;")
    for _k in LEVELS:
        _ds = DS[_k]
        if _k >= 32:
            _chunks = {"year": 1, "y": _ds.sizes["y"], "x": _ds.sizes["x"]}
        else:
            _chunks = {"year": 1, "y": 2048, "x": 2048}
        xql.register(mcon, f"cdl_{_k}", _ds, chunks=_chunks)
    mcon.sql(
        "CREATE TABLE classes(code UTINYINT, name VARCHAR, hex_official VARCHAR,"
        " hex VARCHAR, r UTINYINT, g UTINYINT, b UTINYINT, noncrop BOOLEAN)"
    )
    mcon.executemany("INSERT INTO classes VALUES (?,?,?,?,?,?,?,?)", _rows)

    # crop ranking for the analytics cells (2025 CONUS pixel counts at 64x)
    con.sql(
        """
        CREATE TABLE crop_rank AS
        SELECT c.code, c.name, count(*) AS n
        FROM cdl_64 t JOIN classes c ON c.code = t.crop_type
        WHERE t.year = 2025 AND NOT c.noncrop
        GROUP BY 1, 2 ORDER BY n DESC
        """
    )

    NONCROP_CODES = sorted(r[0] for r in _rows if r[7])
    return NONCROP_CODES, con, con_lock, mcon


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Crop area over 18 years

    One scan of `cdl_64` groups every year at once (1.3 s for all of CONUS).
    Acres are pixel counts times the level's pixel area.
    """)
    return


@app.cell
def _(ACRES_PER_KM2, ANALYSIS_K, PIX_KM2, TOP_N, con, mo):
    area_by_year = mo.sql(
        f"""
        WITH counts AS (
            SELECT year, crop_type, count(*) AS n
            FROM cdl_{ANALYSIS_K}
            WHERE crop_type IN (SELECT code FROM crop_rank LIMIT {TOP_N})
            GROUP BY 1, 2
        )
        SELECT year, c.name AS crop,
               round(n * {PIX_KM2} * {ACRES_PER_KM2} / 1e6, 2) AS m_acres
        FROM counts JOIN classes c ON c.code = crop_type
        ORDER BY year, m_acres DESC
        """,
        engine=con
    )
    return (area_by_year,)


@app.cell
def _(TOP_N, alt, area_by_year, con, mo):
    _pal = con.sql(
        f"SELECT c.name, c.hex FROM crop_rank r JOIN classes c USING (code) LIMIT {TOP_N}"
    ).fetchall()
    _domain = [p[0] for p in _pal]
    _range = [p[1] for p in _pal]
    area_chart = mo.ui.altair_chart(
        alt.Chart(area_by_year.arrow().read_all())
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title=None),
            y=alt.Y("m_acres:Q", title="million acres (approx, majority pyramid)"),
            color=alt.Color(
                "crop:N",
                scale=alt.Scale(domain=_domain, range=_range),
                legend=alt.Legend(title=None),
            ),
            tooltip=["crop:N", "year:O", "m_acres:Q"],
        )
        .properties(height=320)
    )
    area_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Crop rotation

    Same pixels, two years, one self-join on `(y, x)`, pivoted. Rows are the
    first year's crop, columns the second's; cells are million acres. The
    corn/soy off-diagonal dominating its diagonal is the rotation itself.
    """)
    return


@app.cell
def _(YEARS, mo):
    year_a = mo.ui.dropdown([str(y) for y in YEARS], value="2024", label="from")
    year_b = mo.ui.dropdown([str(y) for y in YEARS], value="2025", label="to")
    mo.hstack([year_a, year_b], justify="start", gap=2)
    return year_a, year_b


@app.cell
def _(ACRES_PER_KM2, ANALYSIS_K, PIX_KM2, ROT_N, con, mo, year_a, year_b):
    rotation = mo.sql(
        f"""
        WITH top AS (SELECT code, name FROM crop_rank LIMIT {ROT_N}),
        a AS (
            SELECT y, x, crop_type AS ca FROM cdl_{ANALYSIS_K}
            WHERE year = {int(year_a.value)}
              AND crop_type IN (SELECT code FROM top)
        ),
        b AS (
            SELECT y, x, crop_type AS cb FROM cdl_{ANALYSIS_K}
            WHERE year = {int(year_b.value)}
              AND crop_type IN (SELECT code FROM top)
        ),
        m AS (
            SELECT ta.name AS from_crop, tb.name AS to_crop,
                   round(count(*) * {PIX_KM2} * {ACRES_PER_KM2} / 1e6, 2) AS m_acres
            FROM a JOIN b USING (y, x)
            JOIN top ta ON ta.code = ca
            JOIN top tb ON tb.code = cb
            GROUP BY 1, 2
        )
        PIVOT m ON to_crop USING sum(m_acres) GROUP BY from_crop ORDER BY from_crop
        """,
        engine=con,
    )
    rotation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The map

    Pixel squares straight out of DuckDB. The camera picks the pyramid level
    (about 1.5 data pixels per screen pixel, coarsened if the box would
    exceed the row budget); pan or zoom and the view refolds after the
    camera settles. At deep zoom you are reading the native 30 m array.
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """Controls + status + analysis, UNDER the map as cell output, in the
        deforestation notebook's idiom (Stephen, 2026-08-20: "buttons and
        analysis go on the white space below where the map is, like the other
        notebooks"; a floating on-map panel was built first and rejected).

        Styling copied from deforest's Controls/Status widgets: a 12px
        ui-sans-serif flex row, transparent bordered button, color:inherit
        (no dark panel; the page theme paints it), and a 12.5px ui-monospace
        status line. Proven trait types only: `ctl` Unicode browser -> kernel
        (JSON with `act`: "set" | "analyze"), `status`/`panel` Unicode kernel
        -> browser. A widget rather than mo.ui, so nothing re-runs the map
        cell (deforest's reason). Commits on `change` + 250 ms debounce,
        never `input`; the native range input keeps years comma-free.
        """

        ctl = traitlets.Unicode("").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.2rem 0 0;" +
            "user-select:none";
          const yl = document.createElement("span");
          yl.textContent = "year";
          const range = document.createElement("input");
          range.type = "range";
          range.min = "2008"; range.max = "2025"; range.step = "1";
          range.value = "2025";
          range.style.cssText = "width:11rem";
          const yv = document.createElement("span");
          yv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums";
          yv.textContent = range.value;
          const arrow = (txt, d) => {
            const a = document.createElement("button");
            a.textContent = txt;
            a.style.cssText =
              "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
              "padding:.1rem .45rem;border-radius:4px;border:1px solid " +
              "rgba(127,127,127,.45);background:transparent;color:inherit";
            a.addEventListener("click", () => {
              const v = Math.min(2025, Math.max(2008, +range.value + d));
              if (v === +range.value) return;
              range.value = String(v);
              yv.textContent = range.value;
              commit();
            });
            return a;
          };
          const prevB = arrow("\u25c0", -1);
          const nextB = arrow("\u25b6", 1);
          const lab = document.createElement("label");
          lab.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const c = document.createElement("input");
          c.type = "checkbox";
          c.checked = false;
          lab.appendChild(c);
          lab.appendChild(document.createTextNode("crops only"));
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place\u2026";
          search.style.cssText =
            "width:11rem;font:12px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) {
              model.set("ctl", JSON.stringify({
                act: "search", q: q, year: +range.value,
                crops: c.checked, n: ++seq }));
              model.save_changes();
            }
          });
          const btn = document.createElement("button");
          btn.textContent = "analyze what's in view";
          btn.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.15rem .6rem;border-radius:4px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          box.append(yl, prevB, range, nextB, yv, lab, btn, search);
          const status = document.createElement("div");
          status.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const res = document.createElement("div");
          res.style.cssText =
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.1rem 0";
          const wrap = document.createElement("div");
          wrap.append(box, status, res);
          el.appendChild(wrap);
          // FULLSCREEN: lonboard's own control fullscreens ITS widget element,
          // and the strip must ride along into the white area under the map
          // (Stephen's ask, third layout iteration). Same node, re-parented on
          // fullscreenchange; handlers and traits keep working either way.
          // document.fullscreenElement reports the SHADOW HOST when the real
          // fullscreen element lives in shadow DOM (it does: marimo wraps cell
          // output in shadow roots); descend via shadowRoot.fullscreenElement
          // to the element that actually went fullscreen and append there.
          const realFs = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement)
              fe = fe.shadowRoot.fullscreenElement;
            return fe;
          };
          const onFs = () => {
            const fe = realFs();
            if (fe && fe !== el && !el.contains(fe)) {
              // docked bar at the bottom edge of the fullscreen element: when
              // lonboard stretches the canvas to full height there is no white
              // area left, so the strip brings its own (measured: appended
              // in-flow it lands below the fold and never shows)
              if (getComputedStyle(fe).position === "static")
                fe.style.position = "relative";
              wrap.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;z-index:30;" +
                "background:rgba(255,255,255,.94);color:#111;" +
                "padding:.5rem 1.2rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "";
              el.appendChild(wrap);
            }
          };
          document.addEventListener("fullscreenchange", onFs);
          let seq = 0, deb = null;
          const send = (act) => {
            model.set("ctl", JSON.stringify({
              act: act, year: +range.value, crops: c.checked, n: ++seq }));
            model.save_changes();
          };
          const commit = () => {
            clearTimeout(deb);
            deb = setTimeout(() => send("set"), 250);
          };
          range.addEventListener("input", () => { yv.textContent = range.value; });
          range.addEventListener("change", commit);
          c.addEventListener("change", commit);
          btn.addEventListener("click", () => {
            res.innerHTML = '<span style="opacity:.6">analyzing…</span>';
            send("analyze");
          });
          const paintS = () => { status.textContent = model.get("status") || ""; };
          const paintP = () => {
            const html = model.get("panel") || "";
            res.innerHTML = "";
            if (!html) return;
            const x = document.createElement("button");
            x.textContent = "\u00d7 clear";
            x.style.cssText =
              "float:right;font:11px ui-sans-serif,system-ui,sans-serif;" +
              "cursor:pointer;padding:.05rem .4rem;border-radius:4px;border:" +
              "1px solid rgba(127,127,127,.45);background:transparent;" +
              "color:inherit;margin-left:.6rem";
            x.addEventListener("click", () => { res.innerHTML = ""; });
            const body = document.createElement("div");
            body.innerHTML = html;
            res.append(x, body);
          };
          model.on("change:status", paintS);
          model.on("change:panel", paintP);
          paintS(); paintP();
          // HIDE LONBOARD'S DRAW-BOX TOOL (deforest's fix, ported): the
          // toolbar is rendered unconditionally in lonboard 0.16 and the
          // Map's `controls` trait cannot reach it; recurse the shadow roots
          // on an interval because the map mounts later and can be rebuilt.
          const hideBbox = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("button[aria-label]").forEach((b) => {
              const a = b.getAttribute("aria-label");
              if (a === "Select BBox" || a === "Cancel drawing" ||
                  a === "Clear bounding box") {
                const holder = b.closest("div[style*='absolute']") || b;
                holder.style.display = "none";
              }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) hideBbox(n.shadowRoot);
            });
          };
          const bboxTimer = setInterval(() => hideBbox(document), 1000);
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            clearInterval(bboxTimer);
            wrap.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(
    CartoStyle,
    HOME,
    Map,
    MaplibreBasemap,
    PolygonLayer,
    YEAR0,
    mcon,
):
    # ---- map cell: builds the Map and the ONE pixels layer, must never
    # re-run (repo rule). Labels come from the BASEMAP (Positron WITH labels,
    # Stephen's call after the second-layer id collision killed every overlay
    # route): they render under the pixels, visible over water/background and
    # wherever the fill leaves gaps. The serve only updates this layer's
    # traits; deck.layers is never reassigned. The opening view is served
    # HERE, straight from DuckDB: geometry + a UTINYINT[3] color column.
    _drop = "(" + ", ".join(str(c) for c in sorted({0, 81})) + ")"
    _tmp = PolygonLayer.from_duckdb(
        mcon.sql(
            f"""
            SELECT ST_Transform(
                     ST_MakeEnvelope(x - 3840, y - 3840, x + 3840, y + 3840),
                     'EPSG:5070', 'EPSG:4326', always_xy := true) AS geometry,
                   [c.r, c.g, c.b]::UTINYINT[3] AS color,
                   t.crop_type
            FROM cdl_256 t JOIN classes c ON c.code = t.crop_type
            WHERE t.year = {YEAR0} AND t.crop_type NOT IN {_drop}
            """
        ),
        con=mcon,
        crs="EPSG:4326",
    )
    # SINGLE-CHUNK from the very first table (the serve keeps it so): a layer
    # mounted multi-chunk then trait-updated single-chunk stripes (drive7)
    _t0 = _tmp.table.rechunk(max_chunksize=max(1, _tmp.table.num_rows))
    pixels = PolygonLayer(
        table=_t0,
        _rows_per_chunk=max(1, _t0.num_rows),
        stroked=False,
        get_fill_color=_t0["color"],
    )

    deck = Map(
        layers=[pixels],
        basemap=MaplibreBasemap(style=CartoStyle.Positron),
        view_state=HOME,
        height=700,
        show_side_panel=False,
    )
    deck
    return deck, pixels


@app.cell
def _(HudControls, mo):
    hud = mo.ui.anywidget(HudControls())
    hud
    return (hud,)


@app.cell
def _(
    ACRES_PER_KM2,
    HOLD,
    HOME,
    LEVELS,
    MARGIN,
    NONCROP_CODES,
    PX_PER,
    PolygonLayer,
    ROW_BUDGET,
    VIEW_H,
    VIEW_W,
    YEAR0,
    asyncio,
    con_lock,
    deck,
    hud,
    json,
    math,
    mcon,
    pixels,
    time,
    urllib,
):
    # ---- wiring cell: re-runs freely; a HUD commit (ctl) re-runs it ----------
    # NO THREADS AND NO TIMERS in the serve path (2026-08-20, second flight):
    # a threading.Timer serve ran fine under `marimo run` + playwright but under
    # `marimo edit` on Stephen's machine the camera reached the kernel (a toggle
    # commit served the zoomed view correctly) while the timer-thread serve never
    # painted. Deforest's machinery is the fix, copied: an async settle-debounce
    # awaited on the kernel's own loop, busy/pending coalescing, run_in_executor
    # for the blocking DuckDB work, and every trait assignment on the loop thread.
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _year = int(_c.get("year", YEAR0))
    _crops_only = bool(_c.get("crops", False))
    _act = _c.get("act", "set")
    _q = str(_c.get("q", "")).strip()
    SETTLE = 0.35

    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass


    def _say(msg):
        # comm-handler exceptions are silent (repo lesson); everything the
        # observer path does reports here, errors included
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _vsd(vs):
        if vs is None:
            return None
        if isinstance(vs, dict):
            d = {k: vs.get(k) for k in ("longitude", "latitude", "zoom")}
        else:
            d = {k: getattr(vs, k, None) for k in ("longitude", "latitude", "zoom")}
        return d if None not in d.values() else None

    def _pick_level(vs):
        # floor rule: the finest level whose pixel is still >= PX_PER screen px
        mpp = 156543.03392 * math.cos(math.radians(vs["latitude"])) / 2 ** vs["zoom"]
        want = max(mpp * PX_PER / 30.0, 1.0)
        ks = [k for k in LEVELS if k <= want]
        return ks[-1] if ks else LEVELS[0]

    def _bbox4326(vs):
        span = 360.0 / (512 * 2 ** vs["zoom"])
        dlon = VIEW_W * span * (1 + MARGIN) / 2
        dlat = VIEW_H * span * math.cos(math.radians(vs["latitude"])) * (1 + MARGIN) / 2
        return (
            vs["longitude"] - dlon,
            vs["latitude"] - dlat,
            vs["longitude"] + dlon,
            vs["latitude"] + dlat,
        )

    def _to5070(lon0, lat0, lon1, lat1):
        # DENSIFIED box boundary, not just corners: an EPSG:5070 parallel bows,
        # lowest y AT the central meridian (-96, over Texas), so a 4-corner min
        # clips south TX / the Gulf coast in an arc at CONUS-wide zooms (seen
        # on the opening view; zooming into LA/TX "fixed" it because the
        # corners came close). Sample every edge, then clamp to the array's
        # own Albers bbox: far-out corners land outside the projection's
        # validity and give wild coordinates.
        _N = 8
        pts = []
        for _i in range(_N + 1):
            _t = _i / _N
            _lon = lon0 + (lon1 - lon0) * _t
            _lat = lat0 + (lat1 - lat0) * _t
            pts += [(_lon, lat0), (_lon, lat1), (lon0, _lat), (lon1, _lat)]
        vals = ", ".join(f"({a}, {b})" for a, b in pts)
        rows = mcon.sql(
            f"""
            SELECT ST_X(p), ST_Y(p) FROM (
              SELECT ST_Transform(ST_Point(lon, lat), 'EPSG:4326', 'EPSG:5070',
                                  always_xy := true) AS p
              FROM (VALUES {vals}) AS t(lon, lat))
            """
        ).fetchall()
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        _X0, _Y0, _X1, _Y1 = -2417835.0, 158265.0, 2387295.0, 3321225.0
        return (
            max(min(xs), _X0), max(min(ys), _Y0),
            min(max(xs), _X1), min(max(ys), _Y1),
        )

    def _drop_list():
        return "(0, 81)" if not _crops_only else "(" + ", ".join(
            str(c) for c in sorted({0, 81, *NONCROP_CODES})) + ")"

    def _window(vs):
        """Level + Albers box for a view: floor pick, then the count-based budget."""
        k = _pick_level(vs)
        x0, y0, x1, y1 = _to5070(*_bbox4326(vs))
        drop = _drop_list()
        # row budget: the box's cell count is the UPPER BOUND of drawn rows, so
        # when it already fits the budget no count query is needed at all (every
        # deep zoom skips it; that latency was part of "res change is slow").
        # Only a box that COULD exceed the budget pays for a real count, which
        # serves a level or two finer than geometry alone (background dominates:
        # CONUS crops at 128x is 68k rows against a 1.2M-cell box).
        while k < LEVELS[-1]:
            _est = (x1 - x0) * (y1 - y0) / (30 * k) ** 2
            if _est <= ROW_BUDGET:
                break
            if _est > 24 * ROW_BUDGET:
                k = LEVELS[LEVELS.index(k) + 1]
                continue
            _n = mcon.sql(
                f"""SELECT count(*) FROM cdl_{k}
                    WHERE year = {_year} AND crop_type NOT IN {drop}
                      AND x BETWEEN {x0} AND {x1}
                      AND y BETWEEN {y0} AND {y1}"""
            ).fetchone()[0]
            if _n > ROW_BUDGET:
                k = LEVELS[LEVELS.index(k) + 1]
                continue
            break
        return k, x0, y0, x1, y1, drop

    def _frame(vs):
        """Blocking: one PolygonLayer straight from DuckDB for a view. Runs in
        the executor. The color is typed IN SQL (UTINYINT[3] -> arrow
        FixedSizeList), so `get_fill_color` is the table's own column and the
        layer is `con -> from_duckdb`, nothing in between (Stephen's call,
        2026-08-20: "go from con directly to the lonboard layer")."""
        with con_lock:
            k, x0, y0, x1, y1, drop = _window(vs)
            _served = HOLD.get("served")
            if (
                _served is not None
                and _served[:3] == (k, _year, _crops_only)
                and x0 >= _served[3] and y0 >= _served[4]
                and x1 <= _served[5] and y1 <= _served[6]
            ):
                return None  # held: deck already shows every pixel this view has
            key = (k, _year, _crops_only,
                   round(x0, -3), round(y0, -3), round(x1, -3), round(y1, -3))
            memo = HOLD.setdefault("memo", {})
            tbl = memo.get(key)
            if tbl is None:
                half = 15 * k
                rel = mcon.sql(
                    f"""
                    SELECT ST_Transform(
                             ST_MakeEnvelope(x - {half}, y - {half}, x + {half}, y + {half}),
                             'EPSG:5070', 'EPSG:4326', always_xy := true) AS geometry,
                           [c.r, c.g, c.b]::UTINYINT[3] AS color,
                           t.crop_type
                    FROM cdl_{k} t JOIN classes c ON c.code = t.crop_type
                    WHERE t.year = {_year}
                      AND t.crop_type NOT IN {drop}
                      AND t.x BETWEEN {x0} AND {x1}
                      AND t.y BETWEEN {y0} AND {y1}
                    """
                )
                # from_duckdb does ALL the conversion work in its __init__
                # (WKB parse, interleave, reproject); the widget it creates is
                # a throwaway and only its table is kept, rechunked to ONE
                # chunk so the persistent layer's chunk-sublayer count never
                # changes across serves (multi-chunk swaps striped, drive7)
                _tmp = PolygonLayer.from_duckdb(rel, con=mcon, crs="EPSG:4326")
                tbl = _tmp.table.rechunk(
                    max_chunksize=max(1, _tmp.table.num_rows)
                )
                memo[key] = tbl
                if len(memo) > 24:
                    memo.pop(next(iter(memo)))
            HOLD["served"] = (k, _year, _crops_only, x0, y0, x1, y1)
            return tbl, k

    async def _refresh(vs, force=False):
        """Serve once the camera settles; coalesce whatever piled up meanwhile."""
        if HOLD.get("busy"):
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                if not force and SETTLE > 0:
                    await asyncio.sleep(SETTLE)
                    if HOLD.get("pending") is not None:
                        vs, HOLD["pending"] = HOLD["pending"], None
                        continue
                _t0 = time.time()
                # ON THE LOOP, not an executor: layer widgets created in an
                # executor thread never reach the browser (empty map, healthy
                # kernel); deforest's wiring creates its replacement layers in
                # comm handlers, i.e. on the loop, and that path is proven.
                _out = _frame(vs)
                if _out is None:
                    _say((HOLD.get("last_line") or "") + " · held")
                else:
                    tbl, k = _out
                    n = tbl.num_rows
                    pixels._rows_per_chunk = max(1, n)
                    with pixels.hold_sync():
                        pixels.table = tbl
                        pixels.get_fill_color = tbl["color"]
                    HOLD["k"] = k
                    _ms = int((time.time() - _t0) * 1000)
                    _line = f"{k}x · {30 * k} m pixels · {n:,} drawn · {_ms} ms · year {_year}"
                    HOLD["last_line"] = _line
                    _say(_line)
                vs, force = HOLD.get("pending"), False
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as _e:
            _say(f"serve error: {type(_e).__name__}: {_e}")
        finally:
            HOLD["busy"] = False
            HOLD["pending"] = None

    def _spawn(coro):
        # strong ref kept by the caller; asyncio holds only a weak one
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            _loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, _loop) if _loop else None

    def _on_vs(change):
        try:
            vs = _vsd(change.new)
            if vs is None:
                return
            HOLD["vs"] = vs
            _say("camera…")
            HOLD["task"] = _spawn(_refresh(vs))
        except Exception as _e:
            _say(f"camera error: {type(_e).__name__}: {_e}")

    _old = HOLD.get("h_vs")
    if _old is not None:
        try:
            deck.unobserve(_old, names="view_state")
        except Exception:
            pass
    deck.observe(_on_vs, names="view_state")
    HOLD["h_vs"] = _on_vs


    # ---- "analyze what's in view": top crops for the camera box -> the panel --
    def _analyze_html(vs):
        def _timelapse_svg(top, tl, px_km2, k, tl_ms):
            """Inline SVG line chart, no libraries: M acres per year in the box for
            the top crops in view. Lines wear the class colors the chips above
            already use; each line is direct-labeled at its right end."""
            if not tl:
                return ""
            years = sorted({r[0] for r in tl})
            by = {(r[0], r[1]): r[2] for r in tl}
            series = []
            for nm, hx, _n, code in top:
                vals = [by.get((y, code), 0) * px_km2 * ACRES_PER_KM2 / 1e6 for y in years]
                series.append((nm, hx, vals))
            vmax = max((v for _, _, vals in series for v in vals), default=0) or 1
            W, H, L, R, T, B = 640, 150, 62, 150, 8, 18
            def sx(i):
                return L + (W - L - R) * (i / max(len(years) - 1, 1))
            def sy(v):
                return T + (H - T - B) * (1 - v / vmax)
            parts = [
                f'<div style="margin-top:6px"><svg viewBox="0 0 {W} {H}" '
                f'style="max-width:{W}px;width:100%;display:block;font:10px '
                'ui-sans-serif,system-ui,sans-serif">'
            ]
            # recessive axis: baseline, max gridline, first/last year ticks
            parts.append(
                f'<line x1="{L}" y1="{sy(0)}" x2="{W - R}" y2="{sy(0)}" '
                'stroke="currentColor" stroke-opacity=".25"/>'
                f'<line x1="{L}" y1="{sy(vmax)}" x2="{W - R}" y2="{sy(vmax)}" '
                'stroke="currentColor" stroke-opacity=".08"/>'
                f'<text x="{L - 4}" y="{sy(vmax) + 3}" text-anchor="end" '
                f'fill="currentColor" fill-opacity=".6">{vmax:.1f}M ac</text>'
                f'<text x="{L}" y="{H - 4}" fill="currentColor" '
                f'fill-opacity=".6">{years[0]}</text>'
                f'<text x="{sx(len(years) - 1)}" y="{H - 4}" text-anchor="end" '
                f'fill="currentColor" fill-opacity=".6">{years[-1]}</text>'
            )
            # lines + right-edge name labels, nudged apart when they collide
            _ends = []
            for nm, hx, vals in series:
                pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
                parts.append(
                    f'<polyline points="{pts}" fill="none" stroke="{hx}" '
                    'stroke-width="2" stroke-linejoin="round"/>'
                )
                _ends.append([sy(vals[-1]), nm, hx])
            _ends.sort()
            for _i in range(1, len(_ends)):
                if _ends[_i][0] - _ends[_i - 1][0] < 11:
                    _ends[_i][0] = _ends[_i - 1][0] + 11
            for _y, nm, hx in _ends:
                parts.append(
                    f'<text x="{W - R + 5}" y="{min(max(_y, T + 8), H - B) + 3:.1f}" '
                    f'fill="currentColor" fill-opacity=".85">{nm[:22]}</text>'
                )
            parts.append("</svg>")
            parts.append(
                f'<div style="opacity:.5;font-size:11px">acres by year in view · '
                f"{k}x · timelapse query {tl_ms} ms</div></div>"
            )
            return "".join(parts)

        with con_lock:
            k, x0, y0, x1, y1, drop = _window(vs)
            rows = mcon.sql(
                f"""
                SELECT c.name, c.hex, count(*) AS n, c.code
                FROM cdl_{k} t JOIN classes c ON c.code = t.crop_type
                WHERE t.year = {_year}
                  AND t.crop_type NOT IN {drop}
                  AND t.x BETWEEN {x0} AND {x1}
                  AND t.y BETWEEN {y0} AND {y1}
                GROUP BY 1, 2, 4 ORDER BY n DESC LIMIT 10
                """
            ).fetchall()
            # timelapse: the same box, every year, top 6 crops in view
            _t_tl = time.time()
            _tl_codes = [r[3] for r in rows[:6]]
            tl = mcon.sql(
                f"""
                SELECT year, crop_type, count(*) AS n
                FROM cdl_{k}
                WHERE crop_type IN ({", ".join(str(c) for c in _tl_codes)})
                  AND x BETWEEN {x0} AND {x1}
                  AND y BETWEEN {y0} AND {y1}
                GROUP BY 1, 2 ORDER BY 1
                """
            ).fetchall() if _tl_codes else []
            tl_ms = int((time.time() - _t_tl) * 1000)
        total = sum(r[2] for r in rows) or 1
        px_km2 = (0.03 * k) ** 2
        out = [
            f'<span style="opacity:.65;margin-right:.9rem;white-space:nowrap">'
            f"in view · {k}x · year {_year} · approx (majority pyramid)</span>"
        ]
        for nm, hx, n, _code in rows:
            macres = n * px_km2 * ACRES_PER_KM2 / 1e6
            amt = f"{macres:.2f} M ac" if macres >= 0.01 else f"{macres * 1000:.1f} k ac"
            out.append(
                '<span style="display:inline-block;margin:2px .9rem 2px 0;white-space:nowrap">'
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                f'background:{hx};margin-right:5px;vertical-align:-1px"></span>{nm} '
                f'<span style="opacity:.8;font-variant-numeric:tabular-nums">'
                f"{amt} · {100 * n / total:.0f}%</span></span>"
            )
        if not rows:
            out.append('<span style="opacity:.6">nothing in view</span>')
            return "".join(out)
        # ---- the timelapse: acres by year for the box, one line per crop ----
        out.append(_timelapse_svg(rows[:6], tl, px_km2, k, tl_ms))
        return "".join(out)

    async def _do_analyze():
        try:
            vs = _vsd(HOLD.get("vs")) or dict(HOME)
            html = await asyncio.get_running_loop().run_in_executor(
                None, _analyze_html, vs
            )
            hud.widget.panel = html
        except Exception as _e:
            hud.widget.panel = (
                f'<span style="opacity:.8">analyze error: {type(_e).__name__}: {_e}</span>'
            )

    if _act == "analyze":
        HOLD["atask"] = _spawn(_do_analyze())

    # ---- the search field: Photon (komoot), flood's client moved into the
    # strip (Stephen's call over an on-map GeocoderControl). One urllib GET on
    # a thread, biased toward the camera; the first hit flies the camera (its
    # extent picks the zoom) and the refold follows like any camera move.
    def _photon_first(query, vs):
        _params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            _params["lon"] = round(vs["longitude"], 4)
            _params["lat"] = round(vs["latitude"], 4)
        _url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(_params)
        _req = urllib.request.Request(
            _url, headers={"User-Agent": "x-sql-marimo cdl notebook"}
        )
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _data = json.load(_r)
        _feats = _data.get("features") or []
        if not _feats:
            return None
        _f = _feats[0]
        _p = _f.get("properties", {})
        _lon, _lat = _f["geometry"]["coordinates"][:2]
        _name = ", ".join(
            str(v)
            for v in (_p.get("name"), _p.get("city"), _p.get("state"))
            if v
        ) or query
        # Photon's extent is [minLon, maxLat, maxLon, minLat]
        _ext = _p.get("extent")
        return _name, _lon, _lat, _ext

    async def _do_search():
        try:
            _hit = await asyncio.get_running_loop().run_in_executor(
                None, _photon_first, _q, _vsd(HOLD.get("vs"))
            )
            if _hit is None:
                _say(f"no match: {_q}")
                return
            _name, _lon, _lat, _ext = _hit
            if _ext and len(_ext) == 4:
                _span = max(abs(_ext[2] - _ext[0]),
                            abs(_ext[1] - _ext[3]) * 2, 0.01)
                _zoom = math.log2(360.0 * (VIEW_W / 512) / _span) - 0.3
            else:
                _zoom = 10.0
            _zoom = max(3.5, min(13.5, _zoom))
            _vs = {"longitude": _lon, "latitude": _lat, "zoom": _zoom}
            HOLD["vs"] = _vs
            # assigning view_state kernel-side is ignored; fly_to is the API
            deck.fly_to(longitude=_lon, latitude=_lat, zoom=_zoom,
                        duration=2000)
            _say(f"\u2192 {_name}")
            HOLD["stask"] = _spawn(_refresh(_vs))
        except Exception as _e:
            _say(f"search error: {type(_e).__name__}: {_e}")

    if _act == "search" and _q:
        HOLD["stask0"] = _spawn(_do_search())

    # an analyze click must not repaint the map: the layer is already right, and
    # replacing it makes deck re-triangulate 400k quads (a visible blank). Only a
    # set commit (year/crops) or the first run serves.
    if _act != "analyze" or "k" not in HOLD:
        HOLD["task0"] = _spawn(_refresh(_vsd(HOLD.get("vs")) or dict(HOME), force=True))
    return


@app.cell(hide_code=True)
def _(TOP_N, con, mo):
    _rows = con.sql(
        f"""
        SELECT c.name, c.hex FROM crop_rank r JOIN classes c USING (code)
        LIMIT {TOP_N + 5}
        """
    ).fetchall()
    _chips = " ".join(
        f'<span style="display:inline-block;margin:2px 8px 2px 0;white-space:nowrap">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;'
        f'background:{hx};vertical-align:-1px"></span> {nm}</span>'
        for nm, hx in _rows
    )
    mo.Html(f'<div style="font-size:12px;opacity:.85">{_chips}</div>')
    return


if __name__ == "__main__":
    app.run()
