# Human footprint x Overture divisions: what was built and what was learned

`xsql-hfp-divisions.py`. Vizzuality's Global 100 m Terrestrial Human Footprint folded to
H3 in DataFusion, joined onto Overture division boundaries on the H3 cell id. A
whole-file fork of the deforestation notebook; the diff is the raster side only, and
these notes cover that diff plus what has happened since. `xsql-hfp-conus.py`, the
static one-shot cut from this notebook, is recorded at the bottom.

Status: **runs, opening view fast at res 4, one open defect.** The fullscreen refold
has been seen misfolding again since the viewport ruler landed, so the on-screen
diagnostics are currently enabled (see "The fullscreen defect, reopened").

## The data

`s3://us-west-2.opendata.source.coop/vizzuality/hfp-100/hfp_<year>_100m_v1-2_cog.tif`

| | |
|---|---|
| CRS | **World Mollweide (ESRI:54009)**, metres, R = 6378137 |
| size | ~14 GB per year, 512 px tiles, DEFLATE |
| overviews | 10 doublings, average-resampled (verified, see below) |
| values | human footprint index 0-50, stored **uint16 x1000**, nodata 65535 |
| years | 2017-2021 published; `YEAR` in the constants cell is the seam |
| sparse | 65.7% of full-resolution tiles unstored (ocean) |

The index sums the pressures of built land, cropland, pasture, population, night
lights, roads, railways and navigable rivers per hectare. A sum of intensities is
intensive, so `mean()` is valid at every scale, same argument as the deforestation
portion and the reason the averaged pyramid is legitimate.

The pyramid was verified to average rather than assumed to: over one window (0-10E,
45-50N) the mean survives an 8x downsample (L3 15.135 -> L6 15.150) while the max
collapses (51.2 -> 45.9). Same discipline, same signature as the deforestation COG.

### The uint16 encoding has one trap

The tile reader must use `np.ma.filled` on the masked read, never `np.asarray`:
asarray silently drops the mask and returns raw data, so every masked 65535 survives
as a real number and a nodata coastline averages in at score 65.5 on a 0-50 index.
The division by 1000 happens in the tile reader too, so every cached tile is already
in index units and nothing downstream knows about the encoding.

## Mollweide, closed form, no pyproj

The deforestation notebook's "EPSG:4326 is the whole simplification" does not hold
here, but Mollweide needs ten lines, not the NLCD control-grid machinery:

- **Forward** (viewport box -> pixel window): Newton on `2t + sin 2t = pi sin(lat)`,
  converges in ~5 iterations below 89 degrees. Meridians curve, so a lat/lng box does
  not map to a rectangle; the widest x is wherever the box comes closest to the
  equator. Projecting a 33-point sampled perimeter and taking the envelope handles
  that without case analysis.
- **Inverse** (pixel centres -> lat/lng for the fold): three arcsins. The parametric
  angle depends only on the row, so lat is one arcsin per row and only lon is a full
  2D array.
- Pixels outside the Mollweide ellipse invert to |lon| > 180 and are masked before
  the fold sees them. They are unstored nodata anyway.

Mollweide pixels are true equal-area, so the latitude bias H3 removed for the
deforestation COG does not exist here. H3 still matters for the join (cells are the
unit the divisions machinery fills, ranks and draws) and for within-cell weighting
(a coastal cell is mostly NaN and must not count as a full one).

## Zero is in the ramp, and that is the opposite choice

36.7% of land scores exactly 0, and that is untouched ground, the bottom of a
continuum, not dropped ocean. Ocean is NaN here (unstored), so zero and no-data are
distinguishable, which the deforestation COG never offered. The dark swatch is kept
only for NaN, which the fold's `v = v` filter keeps off screen.

The land distribution, measured globally at L7: p50 1.0, p75 5.5, p99 23.4, max ~51.
Bottom-loaded, so the ramp is log1p over 0-40 (p99.9 is 35.0; the top fifth of a
nominal 0-50 scale holds nothing). Cividis, full range: strictly two-hue, monotonic
in luminance normally and under a deuteranope simulation.

## The zoom ladder, and the res 4 floor

`BASE_RES 5`: one step finer than the deforestation notebook from zoom 4 up, so every
working view gets smaller hexagons. `LEVEL_FOR_RES` carries over from the
deforestation notebook (identical pyramid geometry) with `4: 6` added.

**`MIN_RES` is 4, and it was 5 first.** With a res 5 floor the world view folded
~475k cells from L5 (~62M pixels) and was visibly slow to draw and to hover. An
attempt to fix that by opening the camera at zoom 4 was the wrong lever (it changed
where the map starts, not what the world view costs) and was reverted. Dropping the
floor to res 4 was the right one: fully zoomed out now folds **70,371 cells from L6,
31 tiles, 1 sparse** (measured headless), and every rung from res 5 up is unchanged.
`TILE_BUDGET` stays 512 MB because a wide view in the res 5 band still holds ~253 MB
of L5 tiles on its own.

## The viewport is measured, not assumed

lonboard's `view_state` carries longitude, latitude and zoom, and nothing about the
canvas, so `VIEW_W`/`VIEW_H` (1400x620) were the only source of the fold box's size,
and fullscreen broke it visibly: cells folded for a 620 px band inside a 1500 px
screen. The fix is a ruler in the Status widget: every widget shares the page
document, so it finds the deck canvas (largest canvas on the page), watches it with a
ResizeObserver plus `resize` and `fullscreenchange` listeners (fullscreening an
ELEMENT fires no window resize), and syncs the size to the kernel, where `HOLD["wh"]`
replaces the constants and a size jump beyond 25 px refolds the current view.

Two browser facts, each a debugging round trip:

- **marimo puts cell output in shadow DOM**, so `document.querySelectorAll("canvas")`
  finds nothing with the map plainly on screen. The search must recurse into every
  `shadowRoot`. A ResizeObserver works fine across the boundary once the canvas is
  found.
- **The measurement crosses the bridge as a Unicode `"WxH"` string.** A
  `traitlets.List(traitlets.Float())` synced from JS never reached the kernel under
  marimo's anywidget bridge; the only trait types proven in these notebooks are
  Unicode (kernel -> browser) and Bool (browser -> kernel).

### The fullscreen defect, reopened

After the ruler landed, a fullscreen view again showed cells folded for a band
narrower than the screen, this time with the unfolded gap on the left and right
rather than top and bottom, which points at a stale or undersized width. The
two-sided diagnostics are ENABLED while this is open:

- the kernel's half: the status line ends with ` · WxHpx` from `HOLD["wh"]`. If it
  reads 1400x620 and never moves after fullscreen, the browser's measurement is not
  reaching the kernel.
- the browser's half: a dim `ruler WxH` line under the status line, drawn from JS
  with no kernel round trip. If it is wrong or missing, the measuring code itself is
  the broken leg.

If both readouts agree with the true size and the bands persist, the size is fine
and the refold did not fire: suspect the 25 px jitter gate or `HOLD["vs"]` being
None when `_on_wh` ran. Comment the diagnostics back out when this closes.

## The one-shot: xsql-hfp-conus.py

The fold with everything interactive cut away: no camera, no divisions, no widgets,
no cache. One `BOX`, one read of L2 (400 m, ~32 px per res 7 cell), one fold, one
static H3HexagonLayer. It exists for screenshots and as the smallest runnable
statement of the fold. Its fold cell is a straightened-out copy of this notebook's
read cell, so fixes to the sparse-tile check, the Mollweide pair or the fold SQL
carry across by hand.

`BOX` is the only knob and it scales hard, both measured:

| box | window | tiles | cells | time | RAM |
|---|---|---|---|---|---|
| lower 48 (default) | 16,708 x 7,099 px | 301 fetched, 209 sparse | 1,846,654 | 8.9 s | ~2 GB |
| North America (commented) | 39,481 x 19,360 px | 974 fetched, 2,028 sparse | 4,879,108 | 21.5 s | 15-20 GB |

The North America box hands lonboard a table 2.5x larger than anything else in the
repo draws: a poster run for a big machine, not a default.

Its ramp is **inferno**, not cividis, deliberately: still monotonic in luminance (the
order survives a deuteranope simulation), and its near-black bottom lets untouched
ground recede into the dark basemap, which suits a poster frame where the footprint
should read as light. The cost is that the NaN swatch (38, 40, 44) sits close to the
ramp's own bottom; on this map NaN never draws, so it costs nothing here. The
interactive notebook keeps cividis, where zero-vs-low distinctions are the point.
