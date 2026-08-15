# Mapterhorn explorer: the worldwide DEM as extruded H3

Running record for `xsql-mapterhorn-explorer.py`. The dataset recon (Mapterhorn
PMTiles layout, terrarium decode, the parked NLCD x terrain pairing it grew out of)
is in `docs/imagery-and-terrain-notes.md` under "Mapterhorn terrain, for the record";
this file starts where that one stops: the standalone explorer, its camera
machinery, and what each flight taught.

## Shape of it

Mapterhorn terrain (`planet.pmtiles` z0-12, regional `6-{x}-{y}.pmtiles` z13-18)
read as terrarium WebP tiles, mosaicked per viewport, folded to H3 with xarray-sql
plus the h3 UDF (`avg(elev)` per cell), drawn as one extruded H3HexagonLayer. No
DuckDB, no pyproj. Column height is metres above the lowest ground in view times a
stated scale (opens 20x); colour is true elevation on a fixed ramp unless relative
colors is on. The chassis is the canopy notebook's: ruler Status widget, camera
observers, `_instant`/`refresh` with a settle debounce and coalescing.

The ladder is BASE_RES 7 / ZOOM0 6.2 / PER_RES 1.4, MIN_RES 4 (the world), MAX_RES
13, a res offset slider of +/- 2, and one H3 step coarser past PITCH_COARSE 35
degrees. res 12 and 13 route to the regional archives (z14 and z15 reads).

## The camera footprint (2026-08-15): fold what the camera sees, exactly

The complaint: tilt or orbit the map, or go fullscreen, and the far field is not
filled. Cells stop in a band along the horizon and the far corners are empty.

Three generations of answer, each measured on the same geometry:

1. **Symmetric pad** (canopy's PAD 1.5). Fullscreen pitched over Tibet showed the
   missing band predicted by the parked notebook's "pitch eats the padding" note.
2. **Heuristic horizon extension** in `_pad`: 1.5 view-heights times sin(pitch)
   along the bearing, widened 0.25 of that, symmetric PAD dropped to 1.35, plus
   `_cam_ok` tolerances (5 degrees of pitch, 15 of bearing) deciding when a fold
   in hand still counted. Better, still short.
3. **Exact ray-cast** of deck's own camera, now `view_to_bbox`. `_cam_ok` deleted;
   plain box coverage of the footprint is the whole test.

Why 2 was short, in numbers. Deck's MapView is a pinhole camera 1.5 screen-heights
from the focal point (the map centre on the ground), so its half-fov is
atan(0.5/1.5) = 18.4 degrees. Ray-casting the four screen corners onto the ground
plane, in units of the flat view (screen widths across, screen heights along the
bearing, relative to the centre):

| pitch | across (widths) | along (heights) | area vs flat |
|------:|-----------------|-----------------|-------------:|
| 0     | -0.50 .. 0.50   | -0.50 .. 0.50   | 1.0 |
| 20    | -0.57 .. 0.57   | -0.47 .. 0.61   | 1.2 |
| 35    | -0.65 .. 0.65   | -0.49 .. 0.80   | 1.7 |
| 45    | -0.75 .. 0.75   | -0.53 .. 1.06   | 2.4 |
| 60    | -1.18 .. 1.18   | -0.63 .. 2.37   | 7.1 |

At pitch 60 (deck's default maximum) the top of the screen is 2.37 view-heights
past the centre and the far edge is 2.36 screen-widths wide. The heuristic reached
about 1.98 view-heights (0.675 flat-plus-PAD edge, plus 1.5 x sin 60 = 1.30) and
about 1.0 widths, so roughly the last 0.4 view-heights of the horizon and the
outer 0.6 widths of the far corners were never folded. Fullscreen made both
larger in ground terms because the footprint scales with the measured canvas.

The bearing rotates the trapezoid, so its axis-aligned bounding box grows again
off the cardinal directions: pitch 60 at bearing 45 is ~9x the flat area against
7.1x at bearing 0. That is the cost of folding boxes rather than trapezoids and it
is accepted; the alternative (a bearing-independent disc) was arithmetic'd at
~2.2x the bbox and not built.

The one-step pitch coarsening past 35 degrees is what pays for it: one H3 step is
7x fewer cells per unit area, and pitch 60 is ~7x the area, so the cell count on
a tilted screen stays about level with the flat one. Between 35 and 60 the fold
is cheaper than flat; below 35 the flat ladder holds and the footprint is at most
1.7x.

Details of the ray-cast worth keeping straight:

- Frame is x = screen-right, y = screen-up on the ground, z = up. Camera at
  (0, -d sin p, d cos p) with d = 1.5 x canvas height in px; ray through screen
  point (u, v) has direction (u, v cos p + d sin p, v sin p - d cos p); ground
  hit at t = cam_z / -dz.
- `T_MAX = 6` caps a ray that grazes or misses the ground. The top corners miss
  it past pitch 71.6 degrees (tan p = 3); deck's default max pitch is 60, where
  the top corners sit at t = 2.37. If pitch is ever unlocked past 60 by a bundle
  patch (lonboard 0.16 has no controller knob for it), the cap is what keeps the
  fold finite; the footprint will still be enormous.
- Bearing rotates screen-up clockwise from north: east = x cos b + y sin b,
  north = -x sin b + y cos b. Bearing 90 puts east at the top of the screen and
  south on the right, matching maplibre.
- Latitude goes through Mercator properly (ln tan) rather than the old
  span x cos(lat) approximation; the opening world fold dropped from 83,889 to
  82,387 cells from that alone.
- The canvas size is `HOLD["wh"]`, the ruler's measurement, so fullscreen
  changes the footprint and `_on_wh` refolds when the size jumps by 25 px.

Headless: opens res 4 · 82,387 cells · 12.58M px. Unflown at the time of writing;
the flight to run is fullscreen, pitch to 60, orbit a full turn, and look for the
horizon band.

## Still open

- The res-to-zoom ratio "isn't right, doesn't look good yet"; the res offset
  slider is the manual override while it is tuned. Sina Kashuk's Fused ladder
  (int(3 + z/1.5)) is one step finer than ours through z2.6-7.5 and one coarser
  past z10.4; tried in a reverted session, not flown.
- res 12/13 regional reads are probed, not exercised interactively.
- Whether relative colors is cheap enough to keep is decided by the
  `repaint N ms` readout, not yet read on a real flight.
- Pitch past 60 needs a deck bundle patch (patch_lonboard_surface.py style).
- Whether the footprint's cell counts at pitch 60 fullscreen are acceptable, or
  whether the pitch coarsening should become area-based (steps from the
  footprint's own area ratio, log base 7) instead of a 35 degree threshold.
