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
import time
from collections import defaultdict, namedtuple
from datetime import timedelta
from functools import lru_cache

import numpy as np
from pyproj import Transformer


NAIP_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# How much AOI coverage a single-date year may give up to win on the seam, and how much a
# leaf-off year may give up to win on the season. Both are the same guard against picking
# a sliver: a year that mosaics beautifully over 10% of the box is not the best answer to
# "drape this box".
COV_TOL = 0.02
LEAF_OFF_COV_RATIO = 0.9

# How many candidate campaigns get re-searched in full when the archive query hits its cap.
REFETCH_YEARS = 3

# Latitude above which a December-March flight should be assumed to have snow in it.
SNOW_LAT = 38.0

# THE TWO TIMEOUTS, and they are different problems with the same word.
#
# obstore's HTTP client gives an ENTIRE request 30 seconds by default, connect through
# last byte of body. That is a sane default for the small range reads it was built for and
# far too tight here: one tile of one NAIP quad at a fine overview is tens of megabytes,
# and a wide box asks for dozens of those at once, so they share the link and each one
# individually runs past 30 s. The read then fails as a timeout even though the transfer
# was making progress the whole time, which is why this shows up on big AOIs and big tile
# grids and never on a small box. So: 3 minutes overall, a short connect timeout so a dead
# host still fails fast, and a read timeout that resets on each chunk received, which is
# the one that should be catching a genuinely stalled transfer.
#
# `retry_timeout` bounds retries from the FIRST attempt, so it has to exceed `timeout` or a
# retry can never happen. Kept under 5 minutes because the hrefs are SAS-signed and retries
# do not re-sign.
HTTP_CLIENT_OPTS = {
    "timeout": "180s",
    "connect_timeout": "15s",
    "read_timeout": "60s",
}
HTTP_RETRY = {"max_retries": 6, "retry_timeout": timedelta(minutes=4)}

# The STAC API is the other one, and it is not a bandwidth problem: the search is small
# JSON, but pystac_client ships NO timeout by default, so a stalled connection hangs the
# cell forever rather than failing. (connect, read) in seconds, plus a retry with backoff
# around the whole search, because a transient 429 or 503 from a public API should cost a
# pause rather than the pipeline: with the coverage gate downstream, a failed search now
# stops the DEM read too.
STAC_TIMEOUT = (10, 90)
STAC_TRIES = 3

# Items per page. PC defaults to a small page, so a 3,400-item archive query over a wide
# box was 35 round trips; at 1,000 it is 4, and measurably faster end to end.
STAC_PAGE = 1000

# How much coarser than one texel an overview may be and still be preferred.
#
# MEASURED, and it is the only change here that moved the clock. A 4x4 grid at a 16.3 m
# texel over a 0.4 x 0.3 degree box: 45.9 Mpix in 5.5 s at 1.0, 11.5 Mpix in 2.4 s at 1.4,
# with AOI coverage identical to three decimal places. See the note in `_read_quad`.
#
# It is NOT free, and the cost is worth stating: at 1.4 the chosen overview can be coarser
# than the texel, so some texels repeat a source pixel where at 1.0 they would each get
# their own. On nearest-neighbour sampling of a photograph that is a slight softening, not
# an artifact. Drop to 1.0 if a drape ever needs to be pixel-faithful to the source.
OVERVIEW_TOL = 1.4

# SIMULTANEOUS QUAD READS, and the thing to know is WHICH REGIME YOU ARE IN, because the
# right answer inverts between them and the notebook can be in either.
#
#   BANDWIDTH-BOUND: few large tiles, coarse grid, so each window is tens of megapixels.
#   The link is saturated, extra transfers buy no throughput, and all they do is divide
#   the same bandwidth into more streams that each run longer against a per-request
#   timeout. Here the bound wants to be LOW.
#
#   LATENCY-BOUND: many small tiles, which is what this notebook reaches routinely at 200+
#   tiles. Each window is a fraction of a megapixel and the cost is the round trip, not
#   the bytes. The link sits idle between requests and the bound wants to be HIGH.
#
# The default assumes the latency-bound case, because that is the one a tiled drape is FOR
# and the one where a wrong setting costs minutes rather than seconds. `read_stats_report`
# tells you which regime a given run was in: short per-read times with a low tiles/s means
# latency-bound and under-parallelised, long per-read times near the timeout mean the
# opposite. Tune from that rather than from this comment.
MAX_CONCURRENT_READS = 8

# ---------------------------------------------------------------------------
# AM I BEING THROTTLED?
#
# The honest answer is that obstore will not tell you. `retry_config` handles 429 and 503
# INSIDE the client: a read that was rate-limited four times and succeeded on the fifth
# returns exactly like a read that succeeded first try, only slower. Nothing is raised,
# nothing is logged, and the only trace left is wall clock. So throttling here has to be
# inferred from timing, and confirmed out of band with a raw request that does not retry.
#
# Two signals, and they answer different questions:
#
#   `read_stats_report()` is the PASSIVE one. Every windowed read records when it started,
#   how long it took and how many pixels came back. Rate limiting is not uniform slowness,
#   it is slowness that GETS WORSE as a token bucket drains, so the report compares the
#   median read time of the first third of the run against the last third. A link that is
#   merely narrow is flat; a bucket that is emptying ramps. That ratio is the tell.
#
#   `probe_throttle(href)` is the ACTIVE one, and it is the one that gives a number rather
#   than an inference. It issues a single small range GET with NO retry layer, so a 429 or
#   503 arrives as a status code instead of being absorbed. Azure Blob, which is what
#   Planetary Computer hands out, puts the reason in `x-ms-error-code` (`ServerBusy`,
#   `TooManyRequests`) and the wait in `Retry-After`.
#   THE DRIFT METRIC IS NORMALISED BY CONCURRENCY, and it is wrong without that. A read
#   competing with 40 others is slower than the same read alone, and a `gather` DRAINS:
#   the reads that start last have the fewest competitors left. So raw elapsed time always
#   falls across a run, every run looks like it is speeding up, and a genuine token bucket
#   emptying underneath would be hidden by the drain. Dividing each read's elapsed time by
#   the number in flight when it started gives a per-read cost that is comparable across
#   the run, which is the only version of this number that can detect anything.
_READS: list[tuple[float, float, int, str, int]] = []  # (t0, elapsed, npix, kind, inflight)
_READ_EPOCH = time.monotonic()
_INFLIGHT = [0]

# A read slower than this, at the small windows a tiled drape asks for, is not bandwidth.
SLOW_READ_S = 20.0
# First-third to last-third median ratio above which the run is degrading, not just slow.
RAMP_RATIO = 1.8


def reset_read_stats():
    """Clear the read log. Call once before a fetch you want to measure on its own."""
    global _READ_EPOCH
    _READS.clear()
    _READ_EPOCH = time.monotonic()
    _INFLIGHT[0] = 0


def classify_error(exc):
    """Bucket an obstore/async-geotiff failure into something actionable.

    Matching on the STRING is deliberate. obstore surfaces transport failures through a
    small number of exception types that carry the HTTP status in their message rather
    than as an attribute, and the type names have moved between releases, so a string
    test survives an upgrade where an isinstance chain quietly stops matching and calls
    every throttle an "other".
    """
    s = f"{type(exc).__name__}: {exc}".lower()
    if "429" in s or "too many requests" in s or "slowdown" in s or "serverbusy" in s:
        return "throttled"
    if "503" in s or "service unavailable" in s:
        return "unavailable"
    if "timed out" in s or "timeout" in s or "deadline" in s:
        return "timeout"
    if "403" in s or "401" in s or "authentic" in s or "signature" in s:
        return "auth"
    return "other"


def _record_read(t0, npix, kind, inflight):
    _READS.append((t0 - _READ_EPOCH, time.monotonic() - t0, npix, kind, inflight))


def read_stats_report():
    """Summarise the read log as (text, dict). Empty log gives (None, {})."""
    if not _READS:
        return None, {}
    rows = sorted(_READS)
    n = len(rows)
    ok = [r for r in rows if r[3] == "ok"]
    fails = defaultdict(int)
    for r in rows:
        if r[3] != "ok":
            fails[r[3]] += 1

    def _med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    span = max(r[0] + r[1] for r in rows)
    elapsed = [r[1] for r in ok]
    # Cost per read with the crowd divided out. See the note on `_READS`: raw elapsed
    # falls across every run because the queue drains, so only this is comparable.
    unit = [r[1] / max(r[4], 1) for r in ok]
    third = max(1, len(ok) // 3)
    early, late = _med(unit[:third]), _med(unit[-third:])
    ramp = late / early if early and early == early and early > 0 else float("nan")
    slow = sum(1 for e in elapsed if e > SLOW_READ_S)
    mpix = sum(r[2] for r in ok) / 1e6
    conc = _med([r[4] for r in ok])
    mpix_read = mpix / max(len(ok), 1)

    d = {
        "reads": n, "ok": len(ok), "failures": dict(fails), "wall_s": span,
        "median_s": _med(elapsed), "max_s": max(elapsed, default=float("nan")),
        "early_unit_s": early, "late_unit_s": late, "ramp": ramp,
        "median_inflight": conc, "mpix_per_read": mpix_read,
        "slow_reads": slow, "megapixels": mpix,
        "mpix_per_s": mpix / span if span else float("nan"),
    }

    # WHICH REGIME, which is the number that decides whether to raise or lower the
    # concurrency bounds. Small windows mean the round trip dominates and the link is
    # idle between requests; large ones mean the bytes dominate and it is not.
    d["regime"] = (
        "latency-bound (small windows: raise concurrency if tiles/s is low)"
        if mpix_read < 2.0
        else "bandwidth-bound (large windows: lower concurrency if reads near the timeout)"
    )

    # The verdict, in the order a reader should care about it. A hard 429 that survived
    # six retries outranks any amount of inference from timing.
    if fails.get("throttled"):
        d["verdict"] = "THROTTLED (server said so, after retries were exhausted)"
    elif fails.get("unavailable") or fails.get("timeout"):
        d["verdict"] = "failing on timeouts/503s, which is what sustained throttling looks like once retries run out"
    elif ramp == ramp and ramp > RAMP_RATIO and late > 5.0:
        d["verdict"] = f"LIKELY THROTTLED: reads got {ramp:.1f}x slower across the run"
    elif slow > n * 0.25:
        d["verdict"] = "slow but flat, which reads as a narrow link rather than a rate limit"
    else:
        d["verdict"] = "no sign of throttling"

    lines = [
        f"reads: {len(ok)}/{n} ok"
        + (f" · failures: {dict(fails)}" if fails else "")
        + f" · {span:.1f}s wall",
        f"per read: {d['median_s']:.1f}s median, {d['max_s']:.1f}s max, "
        f"{slow} over {SLOW_READ_S:.0f}s",
        f"drift: {early:.2f}s -> {late:.2f}s per read per unit of concurrency "
        f"({ramp:.1f}x, median {conc:.0f} in flight)",
        f"throughput: {mpix:.0f} Mpix in {span:.1f}s ({d['mpix_per_s']:.1f} Mpix/s decoded), "
        f"{mpix_read:.2f} Mpix/read",
        f"regime: {d['regime']}",
        f"verdict: {d['verdict']}",
    ]
    return "\n".join(lines), d


async def probe_throttle(href, nbytes=65536, tries=3, pause=1.0):
    """Hit one signed href with a small range GET and NO retries; report what comes back.

    This is the check that cannot be fooled by a retry layer. `tries` requests go out
    back to back, because a rate limiter that has room for one request will often refuse
    the third, and a single 200 proves nothing about a burst.

    ASYNC, AND NOT AS A STYLE CHOICE. The first version was a plain function, and calling
    it from a marimo cell froze the kernel: `urlopen` and `sleep` are blocking, so three
    attempts hold the event loop for at least the pause and for up to 90 s if the requests
    hang. A frozen kernel cannot service the websocket or accept a parameter change, which
    presents as the notebook being stuck rather than as this probe being slow. The blocking
    calls go to a worker thread; only the awaits happen here.

    Returns a list of dicts, one per attempt: status, elapsed_s, retry_after, and the
    provider's own error code where it gives one. urllib rather than httpx so this stays
    dependency-free and usable from any of these notebooks.
    """
    import urllib.error
    import urllib.request

    def _once(req):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
                return r.status, dict(r.headers), None
        except urllib.error.HTTPError as e:  # 429 and 503 land here, un-retried
            return e.code, dict(e.headers or {}), None
        except Exception as e:
            return None, {}, f"{type(e).__name__}: {e}"

    out = []
    for i in range(tries):
        req = urllib.request.Request(
            href, headers={"Range": f"bytes=0-{nbytes - 1}"}, method="GET"
        )
        t0 = time.monotonic()
        status, headers, err = await asyncio.to_thread(_once, req)
        h = {k.lower(): v for k, v in headers.items()}
        rec = {
            "attempt": i + 1,
            "status": status,
            "elapsed_s": time.monotonic() - t0,
            "retry_after": h.get("retry-after"),
            "provider_code": h.get("x-ms-error-code") or h.get("x-amz-error-code"),
        }
        if err:
            rec["error"] = err
        rec["throttled"] = status in (429, 503) or bool(rec["retry_after"])
        out.append(rec)
        if i + 1 < tries:
            await asyncio.sleep(pause)
    return out

_Year = namedtuple("_Year", "year quads one_date cov leaf_off dates")


def leaf_on_window(lat):
    """(start, end) day-of-year of the LEAF-ON season at this latitude.

    A drape over forest shows canopy, not ground: in a leaf-on frame the terrain under
    the trees is simply not in the photograph. Leaf-off imagery shows the bare ground,
    the road cuts, the drainage and the rock, which is usually what you drew the box for.

    NAIP is specified as growing-season imagery, so leaf-off frames exist only where a
    state's flight window opened before leaf-out or closed after leaf-drop. That window
    moves with latitude, hence the linear ramp: ~Apr 15 to ~Nov 1 at latitude 35, pulling
    in about nine days per degree north, which puts New England at roughly Apr 23 to
    Oct 24. Clamped at both ends so the far south and the far north stay sane.

    This is a phenology RULE OF THUMB, not a measurement. Elevation, aspect and a cold
    spring all move real leaf-out by a couple of weeks, so a frame near either edge of
    the window may be partly leafed. The dates are reported so you can judge.
    """
    lat = abs(float(lat))
    start = min(max(105.0 + 0.9 * (lat - 35.0), 60.0), 160.0)
    end = min(max(305.0 - 0.9 * (lat - 35.0), 250.0), 340.0)
    return start, end


def naip_quads(bbox, season="any", datetime_range="2010-01-01/2026-12-31", max_items=4000):
    """The most complete NAIP set for `bbox`, as (items, info).

    NAIP is reflown state by state, so a bbox usually intersects several years and the
    naive "newest" answer is a mosaic stitched from flights months or years apart, with a
    visible seam down every quad boundary. So score each year on two things instead:
    whether all its quads share ONE capture date, and how much of the AOI it covers.
    A single-date year is preferred even if it is older, because a seam reads as a bug in
    the drape while an older photo just reads as an older photo.

    COVERAGE OUTRANKS THE SEAM, and this is the correction that made wide boxes work.
    Preferring one date UNCONDITIONALLY is fine while every year covers the whole AOI,
    which is true of a box inside a single state's flight, and false the moment the box
    spans two. A state line is exactly where one year has several dates (two flights) and
    a neighbouring year has one date over 10% of the AOI, so the old rule reached past a
    100%-coverage year for a sliver, and the drape came back nearly empty. Worse, the rule
    SELECTED for the failure: the fewer quads a year has, the likelier they share a date.
    So a single-date year now has to be within `COV_TOL` of the best coverage on offer to
    win. Below that the seam is the lesser problem.

    `season` picks WHAT IS ON THE GROUND rather than when it was flown:

      "any"     the rule above, unchanged. Whatever year mosaics best.
      "prefer"  restrict to years flown entirely outside the leaf-on window, and fall
                back to "any" if this AOI has none. Bare ground where it exists.
      "off"     the same restriction with NO fallback: an AOI with only growing-season
                NAIP returns nothing and the caller drapes something else. Use it when a
                leaf-on frame would be worse than no frame.

    Leaf-off is required of EVERY quad in a year, not most of them, for the same reason
    one capture date is preferred: half a mosaic in leaf and half out is a seam across
    the drape, and a seasonal seam is a far louder one than a date seam.

    `info` is (year, n_quads, coverage_fraction, season_label, warning_or_None), or
    ("error", message) if the STAC call failed, ("none", message) if `season="off"` found
    nothing, or None if the AOI has no NAIP at all. Returning the failure rather than
    raising keeps a network hiccup from taking down a cell that has a DEM to draw.
    """
    import planetary_computer
    import pystac_client
    from shapely import union_all
    from shapely.geometry import box as sbox, shape as sshape

    cat = pystac_client.Client.open(
        NAIP_STAC, modifier=planetary_computer.sign_inplace, timeout=STAC_TIMEOUT
    )

    # No `sortby`: Planetary Computer sorts server-side and it is slow enough over a
    # 15-year window to trip the request timeout. The year grouping below sorts locally,
    # so the server only has to do the cheap bbox + datetime filter.
    #
    # Retried with backoff rather than failed on: this is a public API with rate limits,
    # the query is idempotent, and the whole pipeline is now downstream of the answer.
    def _search(dt, cap):
        for attempt in range(STAC_TRIES):
            try:
                return cat.search(
                    collections=["naip"], bbox=list(bbox), datetime=dt,
                    max_items=cap, limit=STAC_PAGE,
                ).item_collection()
            except Exception:
                if attempt == STAC_TRIES - 1:
                    raise
                time.sleep(2**attempt)

    try:
        items = _search(datetime_range, max_items)
    except Exception as e:  # pystac_client.APIError, transport errors, timeouts
        return [], ("error", f"{type(e).__name__}: {e}")
    if not items:
        return [], None

    # GROUPED BY CAMPAIGN, NOT BY CALENDAR YEAR, and this one was quietly halving coverage
    # on real boxes. A NAIP campaign is a state contract that can run past New Year: the
    # Vermont 2018 flight has quads dated 2018-09-14 through 2019-01-14, and grouping on
    # `datetime.year` split it into a 56% "2019" and a 46% "2018", so a box over the Green
    # Mountains was offered two half-mosaics of the same flight and the winner covered
    # barely half the AOI. `naip:year` is the campaign the quad belongs to, which is the
    # thing that actually mosaics, so group on that and the same flight comes back whole at
    # 100%. Falls back to the calendar year if the property is ever missing.
    #
    # One item per geographic quad per campaign: PC carries reprocessed duplicates of the
    # same flight, and loading both costs a full quad of pixels to paint the same ground.
    def _group(its):
        by_year = defaultdict(dict)
        for it in its:
            year = int(it.properties.get("naip:year") or it.datetime.year)
            by_year[year].setdefault(tuple(round(x, 4) for x in it.bbox), it)
        return by_year

    year_quads = _group(items)
    aoi = sbox(*bbox)

    # TRUNCATION IS NOT A REPORTING PROBLEM, IT IS A HOLE IN THE DRAPE. Hitting `max_items`
    # does not just understate coverage; the quads past the cap are never streamed, so the
    # imagery is genuinely missing over part of the AOI. The API returns roughly
    # newest-first with no order guarantee, so the cap slices a year in half rather than
    # dropping whole years, and a 2-degree box used to arrive with two thirds of its newest
    # year and a 64% coverage figure that was accurate about the wrong set.
    #
    # The cap still has to exist (a 4-degree box is tens of thousands of 0.6 m quads and
    # nobody is draping that), so instead: notice it, and after a year is chosen re-search
    # THAT YEAR ALONE, where the same cap goes a great deal further. The choice was still
    # made on partial data, hence the warning.
    truncated = len(items) >= max_items

    def coverage(quads):
        shapes = union_all([sshape(it.geometry) for it in quads])
        return shapes.intersection(aoi).area / aoi.area if aoi.area else 0.0

    # The window is a function of latitude, so take it at the AOI centre. A box wide
    # enough for the two edges to disagree about leaf-out is wider than this notebook
    # can texture anyway.
    _doy0, _doy1 = leaf_on_window((bbox[1] + bbox[3]) / 2.0)

    def is_off(it):
        doy = it.datetime.timetuple().tm_yday
        return doy < _doy0 or doy > _doy1

    def score(year, qs):
        quads = list(qs.values())
        dates = sorted({it.datetime.strftime("%Y-%m-%d") for it in quads})
        return _Year(
            year=year,
            quads=quads,
            one_date=len(dates) == 1,
            cov=coverage(quads),
            leaf_off=all(is_off(it) for it in quads),
            dates=dates,
        )

    def score_all():
        return [score(y, qs) for y, qs in sorted(year_quads.items(), reverse=True)]

    scored = score_all()

    # REPAIR THE CANDIDATES BEFORE CHOOSING BETWEEN THEM. Repairing only the winner would
    # compare a truncated year against a truncated year and then carefully complete the
    # one that happened to win, which is the wrong order: coverage is the criterion, and
    # under truncation coverage is exactly the number that is wrong. So re-search the top
    # few campaigns in full first, then pick. REFETCH_YEARS is small because each one is
    # another full-archive request on a box that is already the slow case, and the years
    # below it were not going to win on coverage even repaired.
    #
    # The re-search runs to mid-NEXT-year for the same reason the grouping does: a campaign
    # that ends in January would come back missing its own tail from a calendar-year query.
    if truncated:
        for r in sorted(scored, key=lambda r: -r.cov)[:REFETCH_YEARS]:
            try:
                full = _group(_search(f"{r.year}-01-01/{r.year + 1}-06-30", max_items))
            except Exception:
                continue  # the truncated view of this campaign stands
            if len(full.get(r.year, {})) > len(r.quads):
                year_quads[r.year] = full[r.year]
        scored = score_all()

    best_cov = max(r.cov for r in scored)

    # THE LEAF-OFF POOL IS SUBJECT TO THE SAME COVERAGE RULE, and for the same reason: a
    # January flight over one corner of the box is not a better answer than a full summer
    # mosaic, it is a mostly-empty scene. So "prefer" only switches pools when leaf-off
    # reaches LEAF_OFF_COV_RATIO of the best coverage available; below that it says so and
    # falls back. "off" is the explicit instruction to accept whatever leaf-off exists,
    # coverage included, so it takes the pool unconditionally and the caller drapes the
    # palette everywhere the imagery does not reach.
    # A CAMPAIGN CAN BE HALF LEAF-OFF, so the leaf-off candidate is the leaf-off SUBSET of
    # each campaign rather than the campaign entire. Vermont 2018 is the case that forced
    # this: it flew the state in September and October, ran out of season, and came back on
    # January 2nd to finish, so the campaign as a whole is neither leaf-on nor leaf-off but
    # its January half is 52 quads of clean bare ground over 56% of a Green Mountains box.
    # Requiring the whole campaign threw that away and reported "no leaf-off here".
    #
    # The subset is scored like any other candidate, so it still has to win on coverage,
    # and it is single-season by construction, which is the property the seam rule cares
    # about. What it is NOT is a whole-AOI answer: the rest of the box gets the palette.
    pool, warns = scored, []
    if season in ("off", "prefer"):
        leafless = []
        for _y, _qs in sorted(year_quads.items(), reverse=True):
            _sub = {k: v for k, v in _qs.items() if is_off(v)}
            if _sub:
                leafless.append(score(_y, _sub))
        if leafless and (
            season == "off"
            or max(r.cov for r in leafless) >= LEAF_OFF_COV_RATIO * best_cov
        ):
            pool = leafless
        elif season == "off":
            return [], (
                "none",
                f"no leaf-off NAIP here: every flight in the archive falls inside the "
                f"leaf-on window (day {_doy0:.0f}-{_doy1:.0f} at this latitude)",
            )
        elif leafless:
            warns.append(
                f"leaf-off exists here but only over "
                f"{max(r.cov for r in leafless):.0%} of the AOI against {best_cov:.0%} "
                f"leaf-on; using leaf-on. Pick 'Leaf-off only' to take it anyway"
            )

    # Coverage first, the seam second: single date only wins inside COV_TOL of the best on
    # offer. Ties break to the newest campaign.
    _best = max(r.cov for r in pool)
    near = [r for r in pool if r.cov >= _best - COV_TOL]
    single = [r for r in near if r.one_date]
    pick = max(single or near, key=lambda r: (r.cov, r.year))
    quads, cov, dates = pick.quads, pick.cov, pick.dates

    # LEAF-OFF IN THE NORTH IS OFTEN SNOW-OFF-LEAF. Winter NAIP is real and there is more
    # of it than the program's growing-season spec suggests, but it is not evenly spread:
    # by volume it is Florida, Texas, Georgia and Louisiana, where December is simply a
    # convenient clear-sky month, and where it does reach the north it is a state finishing
    # a campaign in deep winter (Vermont and New Hampshire, January 2019). Those northern
    # frames are exactly the ones a mountain box wants and exactly the ones likely to be
    # under snow, with a sun 20 degrees up throwing shadows across every north slope. That
    # is not a reason to refuse them, and snow-off-leaf still shows the ground shape better
    # than canopy does, but it should not be a surprise either.
    if pick.leaf_off and abs((bbox[1] + bbox[3]) / 2.0) >= SNOW_LAT:
        _deep = [
            d for d in dates
            if int(d[5:7]) in (12, 1, 2, 3)
        ]
        if _deep:
            warns.append(
                f"deep-winter imagery above {SNOW_LAT:.0f}N ({', '.join(_deep[:3])}): "
                f"expect snow cover and long low-sun shadows as well as bare branches"
            )

    # What the repair above cannot fix: campaigns the cap cut off entirely were never
    # scored, so an older one that would have won was never in the running. Say so rather
    # than imply the answer is complete. It matters most for leaf-off, which is a small
    # minority of the archive and therefore the first thing a cap hides.
    if truncated:
        warns.insert(
            0,
            f"the archive search hit its {max_items}-item cap, so campaigns before "
            f"{min(year_quads)} were never seen and the choice was made from a partial "
            f"list. Narrow the box or the date range to be sure of it",
        )

    span = dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"
    label = f"{'leaf-off' if pick.leaf_off else 'leaf-on'} · {span}"
    return quads, (pick.year, len(quads), cov, label, " · ".join(warns) or None)


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


@lru_cache(maxsize=64)
def _to_crs(crs):
    """4326 -> `crs`, built once. `Transformer.from_crs` is not cheap, and a tiled drape
    used to call it once per quad PER TILE: 200 tiles over 40 quads is 8,000 PROJ pipeline
    constructions to produce at most a handful of distinct UTM zones."""
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def quads_for_bbox(quads, bbox, pad=0.0):
    """The subset of `quads` whose own footprint intersects `bbox`.

    A tiled drape hands EVERY quad in the AOI to EVERY tile, so a 16-tile grid over 48
    quads makes 768 `_read_quad` calls to issue 120 reads.

    THIS IS NOT A SPEEDUP, and it was added believing it was one. Measured on that grid it
    changed the wall clock by nothing (5.6 s against 5.5 s) and the read count by nothing,
    because `_window_for` already rejects a non-overlapping quad before touching the
    network: the 648 wasted calls were pure local arithmetic and cost about a tenth of a
    second. Kept because it makes the miss explicit and keeps `quads_tried` honest in the
    info dict, not because it buys time. The bytes are the thing; see `OVERVIEW_TOL`.
    """
    w, s, e, n = bbox
    out = []
    for q in quads:
        qw, qs, qe, qn = q.bbox
        if qe + pad >= w and qw - pad <= e and qn + pad >= s and qs - pad <= n:
            out.append(q)
    return out


def _store_for(HTTPStore, href):
    """One store per href, with the timeouts and retries the defaults do not give."""
    return HTTPStore.from_url(
        href, client_options=HTTP_CLIENT_OPTS, retry_config=HTTP_RETRY
    )


async def open_quads(quads, GeoTIFF, HTTPStore):
    """Open every quad's COG once, as {href: GeoTIFF}, for reuse across many reads.

    A tiled drape calls `naip_rgb` once PER TILE, so without this the same COG header is
    re-fetched and re-parsed grid_n^2 times. Opening is the part that is pure overhead
    when repeated; the windowed reads are not, because each tile genuinely wants its own
    window and, at a finer texel size, its own overview level.
    """
    hrefs = list({q.assets["image"].href: None for q in quads})
    t0 = time.monotonic()
    done = [0]

    # A LINE EVERY TENTH, NOT ONE PER HEADER. Per-header was the first version and it is a
    # websocket flood: marimo ships every print to the frontend, a wide box is 50+ quads,
    # and both the photograph and the NDVI cell re-run this on every parameter change. The
    # frontend spends its time rendering progress about work that has already finished.
    step = max(1, len(hrefs) // 10)

    async def _open(h):
        g = await GeoTIFF.open("", store=_store_for(HTTPStore, h))
        done[0] += 1
        if done[0] % step == 0 or done[0] == len(hrefs):
            print(
                f"  opened {done[0]}/{len(hrefs)} COG header(s) · "
                f"{time.monotonic() - t0:.0f}s",
                flush=True,
            )
        return g

    opened = await asyncio.gather(*[_open(h) for h in hrefs])
    return dict(zip(hrefs, opened))


async def _read_quad(
    item, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened=None, bands=3
):
    """Stream one NAIP quad's AOI window at an overview matched to `target_res_m`.

    `bands=3` is R, G, B, the drape. `bands=4` adds NIR, which is the band NAIP carries
    and a drape throws away: it is what makes NDVI possible without a second sensor.
    """
    href = item.assets["image"].href  # already signed by sign_inplace
    g = (opened or {}).get(href) or await GeoTIFF.open(
        "", store=_store_for(HTTPStore, href)
    )

    # Coarsest level still at least as fine as one texel. NAIP is 0.6 m native with
    # overviews to /64, and reading a level finer than the texture can show is pure
    # download: at 2048 texels over 20 km a texel is ~10 m, which is the /16 overview.
    #
    # THE TOLERANCE IS WHERE THE BYTES ARE, and it is worth more than any concurrency
    # setting. Overviews are powers of two, so a strict `res <= texel` test lands on a
    # level that is up to 2x finer than the texture can show in EACH axis, i.e. up to 4x
    # the pixels for a texture that cannot display one of them. A texel of 19 m with
    # levels at 9.6 and 19.2 is the worst case and it is not a rare one. Allowing a level
    # a little coarser than the texel collapses that: the warp below is nearest-neighbour
    # onto the caller's lattice, so a source pixel 1.4x the texel is sampled the same way
    # a source pixel 0.9x the texel is, and the difference does not survive to the screen.
    cands = sorted([g, *g.overviews], key=lambda r: r.res[0])
    fits = [r for r in cands if r.res[0] <= target_res_m * OVERVIEW_TOL]
    reader = fits[-1] if fits else cands[0]

    # The AOI arrives in degrees; the quad is in its own UTM zone. `proj:epsg` is
    # frequently absent on PC NAIP items, so take the CRS off the COG itself.
    fwd = _to_crs(str(g.crs))
    xs, ys = fwd.transform(
        [bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3]]
    )
    win = _window_for(reader, (min(xs), min(ys), max(xs), max(ys)), Window)
    if win is None:
        return None

    # The only line in this function that touches the network, so it is the only one
    # worth timing. Failures are classified and re-raised: swallowing a 429 here would
    # turn "you are being rate limited" into "that quad had no data", which is the exact
    # confusion this instrumentation exists to end.
    t0 = time.monotonic()
    _INFLIGHT[0] += 1
    _n = _INFLIGHT[0]
    try:
        r = await reader.read(window=win)
    except Exception as e:
        _INFLIGHT[0] -= 1
        _record_read(t0, 0, classify_error(e), _n)
        raise
    _INFLIGHT[0] -= 1
    _record_read(t0, int(win.width) * int(win.height) * bands, "ok", _n)
    ma = r.as_masked()
    data = np.ma.getdata(ma)[:bands]  # NAIP band order is R, G, B, NIR
    mask = np.ma.getmaskarray(ma)
    mask = mask[:bands].any(axis=0) if mask.ndim == 3 else np.zeros(data.shape[1:], bool)
    # A quad's collar is padded with pure black rather than with a nodata value, so
    # without this every quad edge drapes as a hard black band across the terrain.
    mask |= (data[0] == 0) & (data[1] == 0) & (data[2] == 0)
    if mask.all():
        return None
    # item.bbox is the quad's FULL lon/lat extent, collar included, which is what the
    # mosaic rule scores against. `r.bounds` is only the window that was read.
    return (
        data.astype("uint8"), mask, tuple(r.bounds), str(g.crs), float(reader.res[0]),
        tuple(item.bbox),
    )


async def naip_rgb(
    quads, lon, lat, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened=None, bands=3
):
    """Stream `quads` and inverse-warp them onto the caller's lon/lat lattice.

    Returns (pixels uint8 (H, W, `bands`), covered bool (H, W), info dict), where `bands`
    is 3 for the R, G, B drape and 4 to keep NAIP's NIR band as well, which is what an
    index like NDVI needs and what a photograph alone cannot give you. `lon` and `lat` are
    2-D arrays of the same shape: whatever grid the caller wants painted, in whatever row
    order it uses. Nothing here assumes north-up, so the notebook's south-first texture
    convention needs no flip on either side.

    INVERSE warp, i.e. ask "which source pixel is under this texel" rather than "where
    does this source pixel land". Forward-scattering leaves holes wherever the source is
    coarser than the destination and needs a fill pass; inverse-sampling cannot, because
    every destination cell is written exactly once. Nearest neighbour: NAIP is already
    being read at roughly one source pixel per texel, so interpolation would only blur.

    WHERE QUADS OVERLAP, THE DEEPEST ONE WINS, and this is what removes the pale rectangles
    that used to sit all over a mountain drape. NAIP quarter-quads are not edge-matched
    tiles: each one carries a COLLAR of 400-500 m that its neighbour also covers, and a
    campaign is flown over weeks, so the two copies of that ground are different days with
    different sun, haze and snow. Filling first-quad-wins therefore painted every collar as
    a rectangle of one flight day surrounded by another, ~400 m across and rectangular
    because a collar is. It read as a rendering artifact and it was a mosaicking one.

    So each texel takes the quad it is FURTHEST INSIDE, scored on normalised distance to
    that quad's own edges. Seams land on the midlines between quad centres instead of on
    the collar boundaries, which is the standard Voronoi-ish mosaic rule: still a seam
    where two days meet, but a single line through it rather than a 400 m patch, and it
    runs where the two images are each at their most reliable rather than at their edges.

    The 4326 -> UTM transform is done ONCE PER CRS, not once per quad. An AOI wide enough
    to want this notebook can span 40 quads, and re-projecting a 2048-square lattice 40
    times costs more than reading the imagery does.
    """
    # CAPPED CONCURRENCY, which is a timeout fix and not a politeness one. Every read has
    # its own wall clock, so firing forty at once does not make them forty times slower in
    # parallel, it makes each one forty times slower against a timeout that does not know
    # about the others. A cap keeps each transfer wide enough to finish inside its budget,
    # and the total is no slower because the link was the constraint either way.
    sem = asyncio.Semaphore(MAX_CONCURRENT_READS)

    # Drop the quads this bbox cannot touch BEFORE opening a store or projecting a corner.
    # See `quads_for_bbox`: on a tiled drape this is most of them, every time.
    hits = quads_for_bbox(quads, bbox)

    async def read_one(q):
        async with sem:
            return await _read_quad(
                q, bbox, target_res_m, GeoTIFF, HTTPStore, Window, opened, bands
            )

    # PHASE TIMING, because "the fetch is slow" has two completely different fixes. Time
    # in `read` is the network and answers to concurrency and overview level; time in the
    # warp is numpy on a 513-square lattice per quad and answers to nothing but doing less
    # of it. Without the split, a CPU-bound warp looks exactly like a throttled link.
    _t_read = time.monotonic()
    reads = await asyncio.gather(*[read_one(q) for q in hits])
    _t_read = time.monotonic() - _t_read
    _t_warp = time.monotonic()

    tiles = [t for t in reads if t is not None]
    info = {
        "quads_read": len(tiles),
        "quads_found": len(quads),
        "quads_tried": len(hits),
        "read_s": _t_read,
        "warp_s": 0.0,
        "source_res_m": min((t[4] for t in tiles), default=float("nan")),
    }
    rgb = np.zeros((*lon.shape, bands), dtype="uint8")
    covered = np.zeros(lon.shape, dtype=bool)
    if not tiles:
        return rgb, covered, info

    # How far inside its own quad each texel sits, 0 at the quad edge and 0.5 dead centre,
    # normalised so quads of different sizes compare. Every texel keeps the best score it
    # has seen, so a later quad only overwrites where it has the better claim.
    depth = np.full(lon.shape, -np.inf)

    by_crs = defaultdict(list)
    for t in tiles:
        by_crs[t[3]].append(t)

    for crs, group in by_crs.items():
        X, Y = _to_crs(crs).transform(lon, lat)
        for data, mask, bounds, _, _, qbbox in group:
            h, w = mask.shape
            left, bottom, right, top = bounds
            ci = np.floor((X - left) / ((right - left) / w)).astype("int64")
            ri = np.floor((top - Y) / ((top - bottom) / h)).astype("int64")
            inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            cic, ric = np.clip(ci, 0, w - 1), np.clip(ri, 0, h - 1)

            qw, qs, qe, qn = qbbox
            d = np.minimum(
                np.minimum(lon - qw, qe - lon) / max(qe - qw, 1e-12),
                np.minimum(lat - qs, qn - lat) / max(qn - qs, 1e-12),
            )
            good = inside & ~mask[ric, cic] & (d > depth)
            if not good.any():
                continue
            for band in range(bands):
                rgb[..., band][good] = data[band][ric, cic][good]
            depth[good] = d[good]
            covered |= good

    info["covered"] = float(covered.mean())
    info["warp_s"] = time.monotonic() - _t_warp
    return rgb, covered, info
