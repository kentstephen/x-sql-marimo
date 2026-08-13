# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "arro3-core",
#     "obstore>=0.9.2",
#     "lonboard>=0.16.0",
#     "anywidget>=0.9",
#     "numpy==2.5.1",
#     "matplotlib==3.11.1",
# ]
# ///
"""Canopy height, extruded: fly low over the forest and see how tall it stands.

ONE DATASET, ONE ENCODING. Meta/WRI's High Resolution Canopy Height Maps (~1 m, uint8
metres, CC-BY 4.0) folded to H3 per viewport and drawn as an extruded H3HexagonLayer:
the height of a column IS the mean canopy height under it, in metres, times a stated
exaggeration. No joins, no boundaries, no second raster, no toggle. The two attempts
that led here (canopy x fire risk, canopy x deforestation, both in archive/) died of
meaning more than mechanics; this one only claims what it draws.

WHY THERE IS NO WORLD VIEW. The CHM has no overview pyramid: 56,147 zoom-9 Web Mercator
quadkey tiles, each a BigTIFF of 65,536 ONE-ROW strips (deflate, predictor 2), so every
read is full-res and pays for full-width strips. The free-fly fold the other notebooks
get from their pyramids does not exist here. The camera is free, but below CANOPY_ZOOM
the layer hides and the status line says why; inside the band every settle folds the
viewport at a resolution the zoom deserves (res 10 at z13 up to res 12 at z15.2).

EXTRUSION IS THE HONEST ENCODING HERE, unlike the parked NLCD x terrain experiment that
CLAUDE.md warns about. There, height was a decoration on a categorical map and buried
the outlines that mattered. Here height is the measured quantity itself: a redwood
stand at 80 m and a clearcut at 0 are the datum, drawn as the thing they are. Colour
(matplotlib Greens, pale -> deep with height) repeats the same number, so the map
survives being viewed from straight above.

THE MSK SIDECARS ARE IGNORED, MEASURED, NOT ASSUMED: GDAL semantics say 0 = invalid,
yet the Paradise CA mask reads ALL ZERO across rows carrying real 40 m heights, and
~10k tiles have no sidecar. No-data is an ABSENT quadkey tile (ocean, unimaged), which
folds no cells at all. Zero canopy is a real measurement and sits INSIDE the ramp.
Vintage is per Maxar acquisition (2018-2020 mostly).

Opens over Prairie Creek Redwoods State Park, because if a canopy height map has one
thing to say, it should say it about the tallest trees on Earth.

Data: Meta & WRI High Resolution Canopy Height Maps, CC-BY 4.0, AWS Open Data
      (s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/).
Run:  uv run marimo edit xsql-canopy-3d.py --sandbox
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import math
    import struct
    import zlib

    import anywidget
    import traitlets
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")  # no GUI backend in a kernel
    import numpy as np
    import obstore
    import pyarrow as pa
    from arro3.core import Table as ArroTable
    from datafusion import SessionContext, udf
    from h3ronpy.vector import coordinates_to_cells
    from obstore.store import S3Store
    from lonboard import Map, H3HexagonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap
    from lonboard._serialization import infer_rows_per_chunk

    return (
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        H3HexagonLayer,
        Map,
        MaplibreBasemap,
        S3Store,
        SessionContext,
        anywidget,
        asyncio,
        coordinates_to_cells,
        infer_rows_per_chunk,
        math,
        matplotlib,
        mo,
        np,
        obstore,
        pa,
        struct,
        traitlets,
        udf,
        zlib,
    )


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
        that assumption visibly wrong. This widget is always mounted just below the map,
        and every widget shares the page document, so it finds the deck canvas (the
        largest canvas on the page), measures its CSS size, and syncs it up as `view_wh`.
        Remeasured on window resize, on fullscreenchange (fullscreening an ELEMENT
        resizes no window, so a resize listener alone misses it), and via a
        ResizeObserver on the canvas itself for layout changes that are neither.
        """

        _esm = """
        function render({ model, el }) {
          const line = document.createElement("div");
          line.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          // The browser's OWN reading, drawn from JS with no kernel round trip. When
          // the ruler works, this matches the px readout in the kernel's line above it;
          // when it does not, whichever half is missing names the broken leg.
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
def _(math):
    # ------------------------------------------------------------------ the canopy
    # One BigTIFF per zoom-9 Web Mercator quadkey tile, 65,536 px square at ~1.19 m,
    # uint8 metres, deflate + predictor 2, 1-row strips, NO overview pyramid.
    CHM_BUCKET = "dataforgood-fb-data"
    CHM_BASE = "forests/v1/alsgedi_global_v6_float"
    CHM_Z = 9
    CHM_TILE = 65536

    # EVERY 4TH PIXEL IN BOTH AXES. The mean over a cell needs a few dozen pixels, not
    # hundreds; the stride pays in the DECODE (a row skipped is a strip never inflated),
    # not the fetch, because KB-sized strips coalesce into one span regardless.
    CAN_STRIDE = 4

    # The strip span one view may fetch. Rows over dense forest run ~16 KB compressed
    # (Paradise CA measured; 320 B is the archive-wide average), so a z13 viewport can
    # span tens of MB; past this the read refuses with a note instead of stalling.
    CANOPY_BUDGET = 160 * 1024 * 1024

    # ------------------------------------------------------------------ the zoom band
    # Same ladder formula as every notebook in this repo (one H3 res per 1.4 zoom
    # levels), but the band only OPENS at CANOPY_ZOOM, because with no pyramid a wide
    # view is unaffordable, and it runs finer than any 100 m raster could: res 10 at
    # z13, res 11 at z14.4, res 12 at z15.2 (~216 native px per cell; the read stride
    # relaxes to 2 there so a cell still averages ~54).
    ZOOM0, PER_RES, BASE_RES = 4.0, 1.4, 4
    CANOPY_ZOOM = 13.0
    CAN_MAX_RES = 15

    def can_res_for_zoom(z):
        return min(CAN_MAX_RES, BASE_RES + math.floor((z - ZOOM0) / PER_RES))

    # ------------------------------------------------------------------ the extrusion
    # get_elevation is the mean canopy in METRES; this is the one lie the map tells,
    # and it is stated: 3x, because a 25 m canopy on a 100 m-wide res-10 cell viewed
    # at pitch 55 reads as texture, not height. The tooltip carries the true metres.
    EXAG = 3.0

    # ------------------------------------------------------------------ view
    # The map's pixel size, as a SEED. The Status widget rulers the real deck canvas
    # and overwrites HOLD["wh"]; these constants only cover the opening fold and
    # headless runs, where no browser ever reports in.
    VIEW_W, VIEW_H = 1400, 620

    # PAD 1.5, not the repo's usual 1.25: the camera opens PITCHED, and a pitched view
    # sees more ground toward the horizon than the flat-view arithmetic in view_to_bbox
    # accounts for. The extra margin is the cheap fix; the alternative is projecting
    # the full frustum, which nothing else here needs.
    PAD = 1.5

    SETTLE = 0.15

    # Prairie Creek Redwoods State Park, pitched so the columns read as columns. The
    # tallest trees on Earth are the argument for this notebook in one view.
    HOME = {
        "longitude": -124.03,
        "latitude": 41.40,
        "zoom": 13.8,
        "pitch": 55,
        "bearing": 0,
    }
    return (
        CANOPY_BUDGET,
        CANOPY_ZOOM,
        CAN_STRIDE,
        CHM_BASE,
        CHM_BUCKET,
        CHM_TILE,
        CHM_Z,
        EXAG,
        HOME,
        PAD,
        SETTLE,
        VIEW_H,
        VIEW_W,
        can_res_for_zoom,
    )


@app.cell
def _(matplotlib, np):
    # THE RAMP: matplotlib Greens, pale -> deep with height, ZERO INSIDE THE RAMP (the
    # HFP lesson: 0 m is a real measurement, the bottom of a continuum, and it is
    # exactly what pavement, grassland and a fresh clearcut look like). There is no
    # no-data swatch: an absent CHM tile folds no cells, so the hexagon is missing.
    #
    # Greens are Stephen's pick and read fine for his eyes: his colour issue is RED
    # (protan-type), a mono-green luminance ramp orders normally, and no red-green
    # pair appears anywhere. Truncated so bare ground is pale green, not white glare.
    # Colour REPEATS the extrusion height, so the map still works from straight above.
    CAN_HI = 40.0
    _CAN_CMAP = matplotlib.colormaps["Greens"]

    def ramp_canopy(v):
        """Mean canopy metres -> uint8 RGB, linear over 0-40 m, pale -> deep green.

        0-40 rather than the 0-25 the parked notebooks used, because this one opens on
        redwoods: the interesting variation here is at the tall end, and a ramp that
        saturates at 25 m would paint the entire old-growth stand one colour.
        """
        v = np.asarray(v, dtype="float64")
        t = np.clip(np.nan_to_num(v) / CAN_HI, 0.0, 1.0)
        return (_CAN_CMAP(0.15 + t * 0.80)[..., :3] * 255).astype(np.uint8)

    CAN_STOPS = [
        (0.0, "0 m"),
        (5.0, "5"),
        (10.0, "10"),
        (20.0, "20"),
        (30.0, "30"),
        (40.0, "40+"),
    ]
    return CAN_STOPS, ramp_canopy


@app.cell
def _():
    # Callback memory. NOT mo.state: writing mo.state from a camera observer re-runs
    # every downstream cell, including the one that owns the Map, so the Map would be
    # rebuilt with its opening view_state and the camera would snap home on every pan.
    # A plain dict is invisible to the dataflow graph.
    HOLD = {
        "fold": None,  # box, res -> layer table, set by the read cell
        "res": None,  # H3 resolution currently on screen
        "box": None,  # padded degree box the current cells cover
        "mode": "hidden",  # "shown" above the band with cells up, "hidden" below it
        "cache": {},  # res -> [box, layer table]
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
def _(ArroTable, coordinates_to_cells, np, pa, ramp_canopy):
    def cells_to_layer(tbl):
        """Folded canopy cells -> the arro3 table the extruded H3HexagonLayer draws.

        `height` is the raw metres; the layer's elevation_scale applies the stated
        exaggeration, so the number in the table stays true and the tooltip can show
        it unmodified as canopy_m.
        """
        tbl = tbl.combine_chunks()
        can = np.asarray(tbl["canopy"], dtype="float64")
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": tbl["hex"],
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(ramp_canopy(can).ravel()), 3
                    ),
                    "height": pa.array(can),
                    "canopy_m": pa.array(np.round(can, 1)),
                    "pixels": tbl["px"],
                }
            )
        )

    def seed_cells():
        """One flat hexagon at null island so the Map has a valid table at build time.

        This is what lets the Map cell depend on nothing, and therefore never wait for
        the first read. The opening draw replaces it.
        """
        hexes = coordinates_to_cells(np.array([0.0]), np.array([0.0]), 4)
        return ArroTable.from_arrow(
            pa.table(
                {
                    "hex": pa.array(hexes),
                    "color": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.array([13, 17, 23], dtype=np.uint8)), 3
                    ),
                    "height": pa.array([0.0]),
                    "canopy_m": pa.array([0.0]),
                    "pixels": pa.array([0], type=pa.int64()),
                }
            )
        )

    return cells_to_layer, seed_cells


@app.cell
def _(
    CANOPY_BUDGET,
    CAN_STRIDE,
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
    # Shared by copy with the two parked forks in archive/; recon record in
    # docs/canopy-firerisk-notes.md.
    #
    # THE MSK SIDECARS ARE IGNORED, AND THAT IS MEASURED, NOT ASSUMED. GDAL mask
    # semantics say 0 = invalid, yet the Paradise CA tile's mask reads ALL ZERO across
    # rows carrying real 40 m heights, and ~10k of the 56k chm tiles have no sidecar at
    # all. A mask that flags live data invalid is worse than none. No-data is therefore
    # an ABSENT TILE (ocean, unimaged), which folds no cells at all.
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
        strided comb over that span, but each is KB-sized, so any range coalescing
        refetches the gaps anyway and one contiguous span is the honest request. The
        stride pays off in the DECODE, where most strips are never inflated, and a
        strip that IS inflated is decompressed full width whatever the column window
        wants: that is what 1-row strips cost.
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

    async def read_canopy(box, stride=CAN_STRIDE):
        """CHM pixels covering `box`, strided, as (lat, lng, v) Arrow rows plus a note.

        `stride` is per call because the ladder's top rung needs a finer comb: res 12
        cells hold ~13 pixels under the default stride, ~54 under stride 2.

        Web Mercator both ways is closed form: rows map to latitude through the
        inverse Gudermannian and columns to longitude linearly. Pixel CENTRES, hence
        the +0.5, same as every fold in this repo.
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
        rows = np.arange(gy0, gy1, stride)
        cols = np.arange(gx0, gx1, stride)
        tys = range(gy0 // CHM_TILE, (gy1 - 1) // CHM_TILE + 1)
        txs = range(gx0 // CHM_TILE, (gx1 - 1) // CHM_TILE + 1)
        if len(tys) * len(txs) > 4:
            # Unreachable from the band (a z13 view is ~13 km against a 78 km tile),
            # so this only guards a future caller with a wider ambition.
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
    BitmapTileLayer,
    CANOPY_ZOOM,
    CAN_STOPS,
    CartoBasemap,
    EXAG,
    H3HexagonLayer,
    HOLD,
    HOME,
    Map,
    MaplibreBasemap,
    PAD,
    SETTLE,
    Status,
    VIEW_H,
    VIEW_W,
    asyncio,
    can_res_for_zoom,
    infer_rows_per_chunk,
    np,
    ramp_canopy,
    seed_cells,
):
    # Built exactly once. This cell depends on no control and on no state the camera
    # can write, so nothing in the notebook can re-run it and throw the view away.
    status = Status(value="<b>loading…</b>")

    _seed = seed_cells()
    cells = H3HexagonLayer(
        table=_seed,
        get_hexagon=_seed["hex"],
        get_fill_color=_seed["color"],
        get_elevation=_seed["height"],
        extruded=True,
        elevation_scale=EXAG,
        stroked=False,
        high_precision=True,
        coverage=1,
        opacity=1.0,
        pickable=True,
    )

    # Place labels drawn OVER the cells; pickable=False so a hover meant for a column
    # is never intercepted; @2x with tile_size 512 because the default 256 samples
    # retina tiles at half scale and the type blurs.
    labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.8,
        pickable=False,
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
    HOLD["mode"], HOLD["head"] = "hidden", ""
    HOLD["wh"] = (float(VIEW_W), float(VIEW_H))
    HOLD["cache"].clear()

    def view_to_bbox(vs):
        """Camera -> [W, S, E, N], clamped to the world.

        Web Mercator flat-view arithmetic; the pitch is absorbed by the widened PAD
        (see the constant). The size comes from HOLD["wh"], the MEASURED canvas.
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
        """The echo check: ignore the event the map emits for a view we set ourselves.

        Longitude, latitude and zoom only, ON PURPOSE: orbiting (pitch or bearing)
        moves neither the fold box nor the resolution, so those events should cost
        nothing, and they fall out here as "same view".
        """
        return (
            a is not None
            and b is not None
            and round(a.longitude, 6) == round(b.longitude, 6)
            and round(a.latitude, 6) == round(b.latitude, 6)
            and round(a.zoom, 4) == round(b.zoom, 4)
        )

    def set_status(vs):
        """Redraw the status line from what is already known, plus this zoom.

        The px readout is the kernel's half of the ruler diagnostics; the probe line
        in Status._esm is the browser's half.
        """
        status.value = (
            f"{HOLD['head']} · zoom {vs.zoom:.1f}"
            f" · {HOLD['wh'][0]:.0f}x{HOLD['wh'][1]:.0f}px"
        )

    def put_cells(tbl):
        cells._rows_per_chunk = max(1, infer_rows_per_chunk(tbl))
        # hold_sync so deck gets one message: hexagons, colours and heights land as
        # one update rather than a frame of new columns wearing old paint.
        with cells.hold_sync():
            cells.table = tbl
            cells.get_hexagon = tbl["hex"]
            cells.get_fill_color = tbl["color"]
            cells.get_elevation = tbl["height"]
            cells.visible = True

    def _instant(vs):
        """Everything answerable without a read, done synchronously in the comm handler."""
        if vs.zoom < CANOPY_ZOOM:
            # Below the band there is nothing this notebook can afford to draw, and
            # stale forest floating over a five-state view would be worse than none.
            if HOLD["mode"] != "hidden":
                cells.visible = False
                HOLD["mode"] = "hidden"
            HOLD["head"] = f"<b>zoom to {CANOPY_ZOOM:g}+ for canopy</b>"
            set_status(vs)
            return True
        res = can_res_for_zoom(vs.zoom)
        seen = view_to_bbox(vs)
        if HOLD["mode"] == "shown" and res == HOLD["res"] and _covers(HOLD["box"], seen):
            set_status(vs)
            return True
        hit = HOLD["cache"].get(res)
        if hit and _covers(hit[0], seen):
            put_cells(hit[1])
            HOLD["mode"] = "shown"
            HOLD["res"], HOLD["box"] = res, hit[0]
            HOLD["head"] = f"<b>res {res}</b> · {hit[1].num_rows:,} cells · cached"
            set_status(vs)
            return True
        return False

    async def _draw(vs, force):
        """Make the screen authoritative for THIS view: cache hit, or read and refold."""
        if not force and _instant(vs):
            return
        res = can_res_for_zoom(vs.zoom)
        want = _pad(view_to_bbox(vs))
        # THE LAST ANSWER STAYS UP UNTIL THERE IS A NEW ONE: the read happens under
        # the columns already on screen and the swap is one trait update.
        HOLD["head"] = f"<b>reading…</b> res {res}"
        set_status(vs)
        layer, note = await HOLD["fold"](want, res)
        if layer is None:
            cells.visible = False
            HOLD["mode"] = "hidden"
            HOLD["res"], HOLD["box"] = res, want
            HOLD["head"] = f"<b>res {res}</b> · {note}"
            set_status(vs)
            return
        HOLD["cache"][res] = [want, layer]
        put_cells(layer)
        HOLD["mode"] = "shown"
        HOLD["res"], HOLD["box"] = res, want
        HOLD["head"] = f"<b>res {res}</b> · {layer.num_rows:,} cells · {note}"
        set_status(vs)

    async def refresh(vs, force=False):
        """Fold what the camera is looking at, once it has stopped moving.

        SETTLE debounces so a drag reads once at the end; coalescing collapses
        whatever piled up during a read to the NEWEST view. No threads and no timers;
        the debounce is an await on the kernel's own loop.
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
        f"{','.join(str(int(c)) for c in ramp_canopy(np.array([v]))[0])})"
        f";outline:1px solid rgba(255,255,255,.18)'></span>{lab}</span>"
        for v, lab in CAN_STOPS
    )
    legend = (
        "<div style=\"font:12px ui-sans-serif,system-ui,sans-serif;"
        "display:flex;flex-wrap:wrap;align-items:center;padding:.35rem 0\">"
        f"<b style='margin-right:.7rem'>canopy height (column = {EXAG:g}x metres)</b>"
        f"{_sw}</div>"
    )
    return deck, legend, refresh, status


@app.cell
async def _(
    HOLD,
    HOME,
    SessionContext,
    asyncio,
    cells_to_layer,
    coordinates_to_cells,
    pa,
    read_canopy,
    refresh,
    udf,
):
    # THE FOLD. The reader hands back strided (lat, lng, metres) rows already
    # flattened, so this is the repo's standard UDF group-by with no xarray in sight:
    # a plain SessionContext, one registered function, one GROUP BY.
    ctx = SessionContext()
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
            # across every version this repo has been run against.
            ctx.register_record_batches(name, [table.to_batches()])

    _can_mem = []  # [[box, res, layer table], ...], newest last
    CAN_KEEP = 6

    def _mem_covers(outer, inner):
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    async def fold_canopy(box, res):
        """Canopy layer table for `box` at `res`, or (None, why not)."""
        for ent in reversed(_can_mem):
            if ent[1] == res and _mem_covers(ent[0], box):
                _can_mem.remove(ent)
                _can_mem.append(ent)
                return ent[2], "cached"
        # The ladder's top rung reads a finer comb; see CAN_MAX_RES for the arithmetic.
        pix, note = await (read_canopy(box, 2) if res >= 12 else read_canopy(box))
        if pix is None:
            return None, note
        # No await between the register and the query: `canpix` is a fixed name in a
        # shared context, and an interleaved camera event must not swap it mid-fold.
        _register("canpix", pix)
        raw = ctx.sql(f"""
            SELECT h3_latlng_to_cell(lat, lng, CAST({res} AS INT)) AS hex,
                   avg(v) AS canopy,
                   count(*) AS px
            FROM canpix
            GROUP BY 1
        """).to_arrow_table()
        if raw.num_rows == 0:
            return None, "no canopy cells"
        layer = cells_to_layer(raw)
        _can_mem.append([box, res, layer])
        while len(_can_mem) > CAN_KEEP:
            _can_mem.pop(0)
        return layer, note

    HOLD["fold"] = fold_canopy
    HOLD["loop"] = asyncio.get_running_loop()

    # The opening draw. force=True skips the settle: there is nothing to debounce yet.
    class _VS:
        longitude = HOME["longitude"]
        latitude = HOME["latitude"]
        zoom = HOME["zoom"]

    await refresh(_VS(), force=True)
    return


@app.cell
def _(CANOPY_ZOOM, deck, legend, mo, status):
    mo.vstack(
        [
            deck,
            status,
            mo.Html(legend),
            mo.md(
                "Mean canopy height per H3 cell, drawn as columns: Meta & WRI High "
                "Resolution Canopy Height Maps, ~1 m (CC-BY 4.0). Column height is the "
                "measured metres times the stated exaggeration; the tooltip carries "
                "the true number. Hold Ctrl (or right-drag) to tilt and orbit. The "
                f"layer draws above zoom {CANOPY_ZOOM:g} only, because the dataset "
                "has no overview pyramid and a wide view would read gigabytes. Zero "
                "is a real measurement (pavement, grass, a fresh clearcut) and sits "
                "at the pale end of the ramp; where no tile exists (ocean, unimaged) "
                "no hexagon is drawn at all. Vintage is per Maxar acquisition, "
                "2018-2020 mostly."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
