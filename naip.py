"""NAIP discovery and drape, as a module so the notebook stays about the fold.

Two calls, and they are deliberately split at the network boundary:

  * `naip_quads(bbox)` asks a STAC API which COGs exist and picks a year. No pixels.
  * `naip_rgb(quads, lon, lat, ...)` streams those COGs and inverse-warps them onto a
    lon/lat grid the CALLER supplies.

Handing the grid in rather than computing one here is the whole point of the second
signature: the notebook already builds a texel lattice to index H3 cells with, so passing
that same lattice makes imagery and shading aligned by construction rather than by two
independent derivations that agree until one of them is edited.

WHY NAIP COMES FROM PLANETARY COMPUTER AND NOT S3. Everything else this repo touches is
an anonymous read off `prd-tnm`. NAIP is not available that way: all three AWS buckets
(`naip-analytic`, `naip-source`, `naip-visualization`) are requester-pays and return 403
to an anonymous request, so they need an AWS account and billing. Planetary Computer's
STAC signs NAIP hrefs anonymously and for free, so it is the only no-account path. The
DEM in the drape notebook still comes straight from the USGS bucket.

The COGs are read with the same obstore + async-geotiff reader the DEM uses, just over
HTTP instead of S3: `HTTPStore.from_url(signed_href)` plus `GeoTIFF.open("", store=...)`.
No rioxarray, no GDAL.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np
from pyproj import Transformer


NAIP_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def naip_quads(bbox, datetime_range="2010-01-01/2026-12-31", max_items=600):
    """The most complete NAIP set for `bbox`, as (items, info).

    NAIP is reflown state by state, so a bbox usually intersects several years and the
    naive "newest" answer is a mosaic stitched from flights months or years apart, with a
    visible seam down every quad boundary. So score each year on two things instead:
    whether all its quads share ONE capture date, and how much of the AOI it covers.
    A single-date year is preferred even if it is older, because a seam reads as a bug in
    the drape while an older photo just reads as an older photo.

    `info` is (year, n_quads, coverage_fraction), or ("error", message) if the STAC call
    failed, or None if the AOI has no NAIP at all. Returning the failure rather than
    raising keeps a network hiccup from taking down a cell that has a DEM to draw.
    """
    import planetary_computer
    import pystac_client
    from shapely import union_all
    from shapely.geometry import box as sbox, shape as sshape

    cat = pystac_client.Client.open(NAIP_STAC, modifier=planetary_computer.sign_inplace)
    # No `sortby`: Planetary Computer sorts server-side and it is slow enough over a
    # 15-year window to trip the request timeout. The year grouping below sorts locally,
    # so the server only has to do the cheap bbox + datetime filter.
    try:
        items = cat.search(
            collections=["naip"], bbox=list(bbox), datetime=datetime_range,
            max_items=max_items,
        ).item_collection()
    except Exception as e:  # pystac_client.APIError, transport errors, timeouts
        return [], ("error", f"{type(e).__name__}: {e}")
    if not items:
        return [], None

    # One item per geographic quad per year: PC carries reprocessed duplicates of the
    # same flight, and loading both costs a full quad of pixels to paint the same ground.
    year_quads = defaultdict(dict)
    for it in items:
        year_quads[it.datetime.year].setdefault(tuple(round(x, 4) for x in it.bbox), it)

    aoi = sbox(*bbox)

    def coverage(quads):
        shapes = union_all([sshape(it.geometry) for it in quads])
        return shapes.intersection(aoi).area / aoi.area if aoi.area else 0.0

    scored = []
    for year, qs in sorted(year_quads.items(), reverse=True):
        quads = list(qs.values())
        one_date = len({it.datetime.strftime("%Y-%m-%d") for it in quads}) == 1
        scored.append((year, quads, one_date, coverage(quads)))

    single = [(y, q, c) for (y, q, s, c) in scored if s]
    if single:
        year, quads, cov = max(single, key=lambda r: r[2])
    else:
        year, quads, _, cov = max(scored, key=lambda r: r[3])
    return quads, (year, len(quads), cov)


def _window_for(reader, bounds_proj, Window):
    """Pixel window on `reader` covering a bbox given in the reader's own CRS."""
    pw, ps, pe, pn = bounds_proj
    bw, bs, be, bn = reader.bounds
    xres = (be - bw) / reader.width
    yres = (bn - bs) / reader.height
    cw, ce = max(pw, bw), min(pe, be)
    cs, cn = max(ps, bs), min(pn, bn)
    if ce <= cw or cn <= cs:
        return None
    col0 = max(0, int((cw - bw) / xres))
    col1 = min(reader.width, int(np.ceil((ce - bw) / xres)))
    row0 = max(0, int((bn - cn) / yres))
    row1 = min(reader.height, int(np.ceil((bn - cs) / yres)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


async def open_quads(quads, GeoTIFF, HTTPStore):
    """Open every quad's COG once, as {href: GeoTIFF}, for reuse across many reads.

    A tiled drape calls `naip_rgb` once PER TILE, so without this the same COG header is
    re-fetched and re-parsed grid_n^2 times. Opening is the part that is pure overhead
    when repeated; the windowed reads are not, because each tile genuinely wants its own
    window and, at a finer texel size, its own overview level.
    """
    hrefs = list({q.assets["image"].href: None for q in quads})
    opened = await asyncio.gather(
        *[GeoTIFF.open("", store=HTTPStore.from_url(h)) for h in hrefs]
    )
    return dict(zip(hrefs, opened))


async def _read_quad(item, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened=None):
    """Stream one NAIP quad's AOI window at an overview matched to `target_res_m`."""
    href = item.assets["image"].href  # already signed by sign_inplace
    g = (opened or {}).get(href) or await GeoTIFF.open("", store=HTTPStore.from_url(href))

    # Coarsest level still at least as fine as one texel. NAIP is 0.6 m native with
    # overviews to /64, and reading a level finer than the texture can show is pure
    # download: at 2048 texels over 20 km a texel is ~10 m, which is the /16 overview.
    cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
    fits = [r for r in cands if r.res[0] <= target_res_m]
    reader = fits[-1] if fits else cands[0]

    # The AOI arrives in degrees; the quad is in its own UTM zone. `proj:epsg` is
    # frequently absent on PC NAIP items, so take the CRS off the COG itself.
    fwd = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True)
    xs, ys = fwd.transform(
        [bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]]
    )
    win = _window_for(reader, (min(xs), min(ys), max(xs), max(ys)), Window)
    if win is None:
        return None

    r = await reader.read(window=win)
    ma = r.as_masked()
    data = np.ma.getdata(ma)[:3]  # NAIP is R, G, B, NIR; the drape wants the first three
    mask = np.ma.getmaskarray(ma)
    mask = mask[:3].any(axis=0) if mask.ndim == 3 else np.zeros(data.shape[1:], bool)
    # A quad's collar is padded with pure black rather than with a nodata value, so
    # without this every quad edge drapes as a hard black band across the terrain.
    mask |= (data[0] == 0) & (data[1] == 0) & (data[2] == 0)
    if mask.all():
        return None
    return data.astype("uint8"), mask, tuple(r.bounds), str(g.crs), float(reader.res[0])


async def naip_rgb(
    quads, lon, lat, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened=None
):
    """Stream `quads` and inverse-warp them onto the caller's lon/lat lattice.

    Returns (rgb uint8 (H, W, 3), covered bool (H, W), info dict). `lon` and `lat` are
    2-D arrays of the same shape: whatever grid the caller wants painted, in whatever row
    order it uses. Nothing here assumes north-up, so the notebook's south-first texture
    convention needs no flip on either side.

    INVERSE warp, i.e. ask "which source pixel is under this texel" rather than "where
    does this source pixel land". Forward-scattering leaves holes wherever the source is
    coarser than the destination and needs a fill pass; inverse-sampling cannot, because
    every destination cell is written exactly once. Nearest neighbour: NAIP is already
    being read at roughly one source pixel per texel, so interpolation would only blur.

    The 4326 -> UTM transform is done ONCE PER CRS, not once per quad. An AOI wide enough
    to want this notebook can span 40 quads, and re-projecting a 2048-square lattice 40
    times costs more than reading the imagery does.
    """
    reads = await asyncio.gather(
        *[
            _read_quad(q, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened)
            for q in quads
        ]
    )
    tiles = [t for t in reads if t is not None]
    info = {
        "quads_read": len(tiles),
        "quads_found": len(quads),
        "source_res_m": min((t[4] for t in tiles), default=float("nan")),
    }
    rgb = np.zeros((*lon.shape, 3), dtype="uint8")
    covered = np.zeros(lon.shape, dtype=bool)
    if not tiles:
        return rgb, covered, info

    by_crs = defaultdict(list)
    for t in tiles:
        by_crs[t[3]].append(t)

    for crs, group in by_crs.items():
        X, Y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
        for data, mask, bounds, _, _ in group:
            h, w = mask.shape
            left, bottom, right, top = bounds
            ci = np.floor((X - left) / ((right - left) / w)).astype("int64")
            ri = np.floor((top - Y) / ((top - bottom) / h)).astype("int64")
            inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h) & ~covered
            cic, ric = np.clip(ci, 0, w - 1), np.clip(ri, 0, h - 1)
            good = inside & ~mask[ric, cic]
            if not good.any():
                continue
            for band in range(3):
                rgb[..., band][good] = data[band][ric, cic][good]
            covered |= good

    info["covered"] = float(covered.mean())
    return rgb, covered, info
