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
"""Deforestation 2002-2022 per CONUS county, one fold at res 7, one static map.

This is xsql-deforest-divisions.py with everything interactive cut away, on the
xsql-hfp-conus.py chassis: no camera, no widgets, no cache, no zoom ladder. One
bounding box (the lower 48), one read of the deforestation COG's L2 overview (400 m,
~32 px per res 7 cell, the same LEVEL_FOR_RES row the interactive notebook uses), one
DataFusion fold to H3 res 7, one fetch of Overture's county polygons out of the
divisions PMTiles, one DuckDB dissolve + polyfill, one zonal join, one PolygonLayer.
The COG reader, the PMTiles client, the MVT decode, the dissolve and polyfill SQL and
the ramp are all ported from that notebook by copy, per the repo's shared-by-copy
rule: a fix there should be carried here by hand.

THE CLIP TO CONUS IS THE JOIN ITSELF. The fold box hangs over Canada, Mexico and open
water, but the polyfill is 'center'-ruled against county polygons filtered to
country = US minus Alaska and Hawaii, so a cell counts only if its centre lands inside
a CONUS county, and nothing but county polygons is drawn. No separate country-polygon
clip step exists because none is needed.

ZERO CELLS ARE KEPT, and that is a deliberate departure from the interactive
notebook's fold. Its `HAVING avg(v) > 0` is a render economy: zero cells are
overwhelmingly ocean, and drawing them buries the map in hexagons that say nothing.
Here no hexagons are drawn and ocean is already excluded twice over (NaN fails the
`v = v` test; a cell centred on water is in no county), so what zero cells remain are
UNTOUCHED LAND, and a county's mean share deforested must include the ground that
lost nothing or it is a different, inflated number. The residual bias is coastal:
open water the COG stores as 0 rather than leaving unstored still averages into a
shoreline cell and drags it low. That is the same pixel-level tradeoff the
interactive notebook accepts inside its nonzero cells.

Counties smaller than one res 7 cell (5.16 km2; a handful of Virginia's independent
cities) can catch no cell centre and are dropped by the inner join rather than given
a guessed number; the count is reported in the stats line.

Data: Vizzuality / LandGriffon deforest_100m_cog.tif, CC-BY 4.0, on source.coop.
Boundaries: Overture Maps divisions PMTiles.
Run:  uv run marimo edit xsql-deforest-conus-counties.py --sandbox
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import math
    import struct

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
    from lonboard import Map, PolygonLayer, BitmapTileLayer
    from lonboard.basemap import CartoBasemap, MaplibreBasemap

    return (
        ArroArray,
        ArroTable,
        BitmapTileLayer,
        CartoBasemap,
        GeoTIFF,
        Map,
        MaplibreBasemap,
        PolygonLayer,
        S3Store,
        Window,
        XarrayContext,
        asyncio,
        coordinates_to_cells,
        duckdb,
        from_wkb,
        gzip,
        math,
        matplotlib,
        mo,
        multipolygon,
        np,
        obstore,
        pa,
        struct,
        udf,
        xr,
    )


@app.cell
def _(duckdb):
    # ONE JOB, same split as the interactive notebook: DuckDB does the two geometry steps
    # (the polyfill and the tile-seam dissolve), DataFusion does the fold and the
    # equi-join. The engine benchmark behind the split is in xsql-duckdb-nlcd-h3.py.
    con = duckdb.connect()
    con.execute("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
    return (con,)


@app.cell
def _():
    # ------------------------------------------------------------------ the raster
    SOURCE_BUCKET = "us-west-2.opendata.source.coop"
    COG = "vizzuality/lg-land-carbon-data/deforest_100m_cog.tif"
    TILE = 512
    FETCH_AT_ONCE = 32

    # Res 7 (5.16 km2 cells) reads L2 (400 m, 0.16 km2 pixels): ~32 px per cell, the same
    # row of the interactive notebook's LEVEL_FOR_RES ladder. Reading an overview is only
    # equivalent to reading pixels because the pyramid AVERAGES, verified there (the mean
    # survives a 64x downsample while the max collapses).
    RES = 7
    LEVEL = 2

    # The lower 48 by bounding box, same box as xsql-hfp-conus.py: Cape Alava to West
    # Quoddy Head, Key West to the 49th parallel. Unlike that notebook the fringes of
    # Canada and Mexico inside the box do NOT survive to the map: the county join is the
    # clip. The box also contains every CONUS county whole, so the tile-range edge never
    # clips a county that will be drawn.
    BOX = (-124.8, 24.4, -66.9, 49.5)

    # CONUS = country US minus these region codes (ISO 3166-2 with the "US-" stripped,
    # as fetch stores them). Territories (PR, VI, GU, ...) fall outside BOX on their own.
    NOT_CONUS = {"AK", "HI"}

    # ------------------------------------------------------------------ boundaries
    # Overture's PMTiles build of the pinned release, same object the interactive
    # notebook reads. Counties first appear at tile zoom 8 (measured there, baked in by
    # Planetiler), so z8 is both the coarsest and the cheapest zoom that has them; at z8
    # a tile unit is ~38 m against 1,400 m cell edges, so quantization is nowhere near
    # the polyfill. CONUS at z8 is ~1,000 tiles, fetched once, concurrently.
    OVERTURE_RELEASE = "2026-07-22.0"
    PM_BUCKET = "overturemaps-extras-us-west-2"
    PM_PATH = f"tiles/{OVERTURE_RELEASE}/divisions.pmtiles"
    COUNTY_Z = 8

    # ------------------------------------------------------------------ the paint
    FILL_ALPHA = 235
    return (
        BOX,
        COG,
        COUNTY_Z,
        FETCH_AT_ONCE,
        FILL_ALPHA,
        LEVEL,
        NOT_CONUS,
        PM_BUCKET,
        PM_PATH,
        RES,
        SOURCE_BUCKET,
        TILE,
    )


@app.cell
def _(matplotlib, np):
    # THE LOG RAMP, ported whole from the interactive notebook. County means over CONUS
    # are bottom-loaded the same way the global cells are, so the same stretch applies:
    # zero takes its own dark swatch separated by LUMINANCE (the flat grey collided with
    # the 0.1% stop at luminance 0.313 vs 0.318, measured there), and the live range is
    # log10 over LO..HI on cividis truncated to its upper 75%. cividis because it is
    # strictly two-hue blue -> yellow and monotonic in luminance, so the order survives a
    # colour-vision simulation, which is the only promise a sequential ramp has to keep.
    LO, HI = 1e-4, 0.5
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
        """`ramp` with a constant alpha appended, as uint8 RGBA."""
        rgb = ramp(v)
        out = np.empty(rgb.shape[:-1] + (4,), dtype=np.uint8)
        out[..., :3] = rgb
        out[..., 3] = alpha
        return out

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
async def _(
    BOX,
    COG,
    FETCH_AT_ONCE,
    GeoTIFF,
    LEVEL,
    RES,
    S3Store,
    SOURCE_BUCKET,
    TILE,
    Window,
    XarrayContext,
    asyncio,
    coordinates_to_cells,
    math,
    np,
    pa,
    udf,
    xr,
):
    # THE FOLD, ONCE. Ported from the interactive notebook and straightened out: no LRU,
    # no budget, no cache dict, because there is exactly one window and it is read exactly
    # once. What survives the port unchanged is what was learned the hard way there: the
    # sparse-tile table (73.6% of full-res tiles are unstored ocean, and a read touching
    # one raises "Invalid range requested, start: 0 end: 0"), and the `v = v` NaN test
    # (the COG declares no nodata, so ocean arrives as NaN).
    #
    # EPSG:4326 is the whole simplification: the pixel grid IS degrees, so the dataset's
    # y/x coordinates feed h3_latlng_to_cell directly. No projection machinery.
    import time as _time

    _t0 = _time.perf_counter()
    _store = S3Store(SOURCE_BUCKET, region="us-west-2", skip_signature=True)
    _g = await GeoTIFF.open(COG, store=_store)
    _lv = [_g, *_g.overviews][LEVEL]
    _L, _B, _R, _T = _g.bounds
    _H, _W = _lv.shape
    _px, _py = (_R - _L) / _W, (_T - _B) / _H
    _present = (
        np.asarray(_lv.ifd.tile_byte_counts).reshape(-(-_H // TILE), -(-_W // TILE)) > 0
    )

    _w, _s, _e, _n = BOX
    _col0 = max(0, int((max(_w, _L) - _L) / _px))
    _col1 = min(_W, int(math.ceil((min(_e, _R) - _L) / _px)))
    _row0 = max(0, int((_T - min(_n, _T)) / _py))
    _row1 = min(_H, int(math.ceil((_T - max(_s, _B)) / _py)))
    _wpx, _hpx = _col1 - _col0, _row1 - _row0

    _sem = asyncio.Semaphore(FETCH_AT_ONCE)

    async def _tile(ty, tx):
        r0, c0 = ty * TILE, tx * TILE
        h, w = min(TILE, _H - r0), min(TILE, _W - c0)
        if not _present[ty, tx]:
            return ty, tx, None
        async with _sem:
            m = (
                await _lv.read(window=Window(col_off=c0, row_off=r0, width=w, height=h))
            ).as_masked()[0]
        return ty, tx, np.asarray(m).astype(np.float32)

    _want = [
        (ty, tx)
        for ty in range(_row0 // TILE, (_row1 - 1) // TILE + 1)
        for tx in range(_col0 // TILE, (_col1 - 1) // TILE + 1)
    ]
    _arr = np.full((_hpx, _wpx), np.nan, dtype=np.float32)
    _fetched = _skipped = 0
    for _ty, _tx, _a in await asyncio.gather(*(_tile(*k) for k in _want)):
        if _a is None:
            _skipped += 1
            continue
        _fetched += 1
        _sr, _sc = _ty * TILE, _tx * TILE
        _r0, _c0 = max(_row0, _sr), max(_col0, _sc)
        _r1 = min(_row1, _sr + _a.shape[0])
        _c1 = min(_col1, _sc + _a.shape[1])
        _arr[_r0 - _row0 : _r1 - _row0, _c0 - _col0 : _c1 - _col0] = _a[
            _r0 - _sr : _r1 - _sr, _c0 - _sc : _c1 - _sc
        ]
    _t_read = _time.perf_counter() - _t0

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
    ctx.from_dataset(
        "df",
        xr.Dataset(
            {"v": (("y", "x"), _arr)},
            coords={
                "y": _T - (_row0 + np.arange(_hpx) + 0.5) * _py,
                "x": _L + (_col0 + np.arange(_wpx) + 0.5) * _px,
            },
        ),
        chunks={"y": 512},
    )

    # NO `HAVING avg(v) > 0`, unlike the interactive fold, and the module docstring says
    # why at length: zero cells here are untouched land, and a county mean without them
    # is a different number. `v = v` still drops NaN ocean; px_total still weights the
    # pixels within a cell (a shoreline cell that is 90% NaN must not count as full).
    folded = ctx.sql(f"""
        SELECT h3_latlng_to_cell(y, x, CAST({RES} AS INT)) AS hex,
               avg(CAST(v AS DOUBLE)) AS portion,
               count(*)               AS px_total
        FROM df
        WHERE v = v
        GROUP BY 1
    """).to_arrow_table()
    _t_all = _time.perf_counter() - _t0

    fold_stats = (
        f"window {_wpx}x{_hpx} px at L{LEVEL} · {_fetched} tiles fetched, "
        f"{_skipped} sparse · {folded.num_rows:,} res {RES} cells · "
        f"read {_t_read:.1f}s, fold total {_t_all:.1f}s"
    )
    return ctx, fold_stats, folded


@app.cell
async def _(
    BOX,
    COUNTY_Z,
    NOT_CONUS,
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
    struct,
):
    # THE COUNTIES, OUT OF ONE PMTILES OBJECT BY RANGED GET. The client and the MVT
    # decode are the interactive notebook's, ported by copy and trimmed of the LRU and
    # the coverage memo: everything here is fetched exactly once. The decode was
    # verified ring-exact against mapbox-vector-tile there before being trusted.
    import time as _ctime

    _ct0 = _ctime.perf_counter()
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

    county_stats = (
        f"{(_x1 - _x0 + 1) * (_y1 - _y0 + 1)} tiles at z{COUNTY_Z} · "
        f"{len(_rows):,} pieces -> {counties.num_rows:,} counties · "
        f"fetch {_t_fetch:.1f}s, with dissolve {_ctime.perf_counter() - _ct0:.1f}s"
    )
    return counties, county_stats


@app.cell
def _(RES, con, counties, ctx, folded):
    # THE POLYFILL AND THE ZONAL JOIN, one pass each.
    #
    # h3_polygon_wkb_to_cells_experimental takes a POLYGON and raises on a MultiPolygon,
    # which every dissolved county is, so ST_Dump splits first. Containment is 'center':
    # each cell lands in exactly one county, which is what makes the mean a zonal mean
    # rather than a smear, and what makes the join the CONUS clip (a cell centred in
    # Canada, Mexico or open water is in no county and drops out here).
    #
    # avg(portion) EQUAL-WEIGHTS THE CELLS, and that is the point: H3 cells are
    # near-equal-area, so an unweighted mean over cells is an area-weighted mean of the
    # ground. The pixel weighting already happened inside each cell, in the fold.
    import time as _jtime

    _jt0 = _jtime.perf_counter()
    # con.register, not the replacement scan: see the dissolve cell.
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

    for _name, _tbl in (("div_cells", _mapping), ("cells", folded)):
        try:
            ctx.deregister_table(_name)
        except Exception:
            pass
        try:
            ctx.from_arrow(_tbl, name=_name)
        except Exception:
            # Older datafusion has no from_arrow(name=...); the batches path is stable.
            ctx.register_record_batches(_name, [_tbl.to_batches()])

    _zonal = (
        ctx.sql("""
        SELECT d.id           AS id,
               avg(c.portion) AS portion,
               count(*)       AS n_cells
        FROM div_cells d JOIN cells c ON d.hex = c.hex
        GROUP BY d.id
    """)
        .to_arrow_table()
        .combine_chunks()
    )

    # Counties that caught no cell centre get NO number rather than a guessed one: the
    # inner join drops them, which is what keeps the choropleth honest.
    joined = counties.join(_zonal, keys="id", join_type="inner").combine_chunks()
    n_unmeasured = counties.num_rows - joined.num_rows
    join_stats = (
        f"{_mapping.num_rows:,} filled cells · {joined.num_rows:,} counties measured, "
        f"{n_unmeasured} too small for a res {RES} centre · "
        f"join {_jtime.perf_counter() - _jt0:.1f}s"
    )
    return join_stats, joined


@app.cell
def _(
    ArroArray,
    ArroTable,
    BitmapTileLayer,
    CartoBasemap,
    FILL_ALPHA,
    Map,
    MaplibreBasemap,
    PolygonLayer,
    from_wkb,
    joined,
    multipolygon,
    np,
    pa,
    ramp_rgba,
):
    # JUST THE COUNTIES: no hexagons, no raster. Fill and a thin neutral stroke, so the
    # boundaries read as lines rather than a second encoding; the stroke is separated
    # from the fill by luminance alone, per the repo's colour rules.
    #
    # line_width_units="pixels" explicitly, or deck's metre default makes the width
    # max(1 metre, line_width_min_pixels) and it can never go below the floor.
    _tbl = joined
    _portion = np.asarray(_tbl["portion"], dtype="float64")
    _geom = ArroArray.from_arrow(
        from_wkb(_tbl["wkb"].combine_chunks(), to_type=multipolygon("xy", crs="EPSG:4326"))
    )
    _layer_tbl = ArroTable.from_arrays(
        [
            _geom,
            ArroArray.from_arrow(
                pa.FixedSizeListArray.from_arrays(
                    pa.array(ramp_rgba(_portion, FILL_ALPHA).ravel()), 4
                )
            ),
            ArroArray.from_arrow(_tbl["name"].combine_chunks()),
            ArroArray.from_arrow(_tbl["region"].combine_chunks()),
            ArroArray.from_arrow(pa.array(np.round(_portion * 100, 4))),
            ArroArray.from_arrow(_tbl["n_cells"].combine_chunks()),
        ],
        names=["geometry", "color", "name", "state", "deforested %", "cells"],
    )

    _counties_layer = PolygonLayer(
        table=_layer_tbl,
        get_fill_color=_layer_tbl["color"],
        filled=True,
        stroked=False,
        line_width_units="pixels",
        get_line_width=0.5,
        line_width_min_pixels=0,
        line_width_max_pixels=1.0,
        get_line_color=[15, 17, 21, 170],
        opacity=1.0,
        pickable=True,
    )

    # Place labels OVER the fills: the basemap paints under every deck layer. @2x with
    # tile_size 512 or retina type blurs.
    _labels = BitmapTileLayer(
        data="https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png",
        tile_size=512,
        max_zoom=19,
        min_zoom=0,
        opacity=0.6,
        pickable=False,
    )

    deck = Map(
        [
            _counties_layer,
            # _labels
        ],
        basemap=MaplibreBasemap(style=CartoBasemap.DarkMatterNoLabels),
        view_state={"longitude": -96.0, "latitude": 38.5, "zoom": 4.0},
        height=700,
        show_tooltip=True,
    )
    deck
    return


@app.cell
def _(STOPS, county_stats, fold_stats, join_stats, mo, np, ramp):
    _sw = "".join(
        "<span style='display:inline-flex;align-items:center;gap:.3rem;"
        "margin-right:.9rem'><span style='width:.8rem;height:.8rem;border-radius:2px;"
        f"background:rgb({','.join(str(int(c)) for c in ramp(np.array([v]))[0])})'>"
        f"</span>{label}</span>"
        for v, label in STOPS
    )
    mo.vstack(
        [
            mo.Html(
                "<div style='font:12px ui-sans-serif,system-ui,sans-serif;"
                f"padding:.2rem 0'>mean share deforested 2002-2022 &nbsp; {_sw}</div>"
            ),
            mo.md(
                f"`{fold_stats}`  \n`{county_stats}`  \n`{join_stats}`  \n"
                "Each county's value is the mean over the res 7 H3 cells whose centre "
                "falls inside it; cells are near-equal-area, so the mean is "
                "area-weighted. Counties smaller than one cell are dropped, not "
                "guessed. Raster: Vizzuality / LandGriffon (CC-BY 4.0). Boundaries: "
                "Overture Maps."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
