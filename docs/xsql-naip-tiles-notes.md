# xsql-naip-tiles.py: constant-resolution tiles

New notebook, built from `xsql-naip-ndvi.py` by removing one assumption. That notebook is
untouched and still works.

## The assumption, and why every documented ceiling was it

`xsql-naip-ndvi.py:1094` is `_T = tex_size.value`: one lattice over the AOI, `tex_size`
square, carrying heights, hillshade, the H3 key, NDVI and the textures. Its spacing is
`AOI / tex_size`. So is the mesh quad (`AOI / mesh_density`) and so is the DEM overview it
streams. **Resolution was a function of box width.** The controls could only pick which
constant to divide by, which is why the honest advice had become "draw a smaller box".

The three ceilings in the previous handoff are one ceiling stated three times.

## The invariant

**No array is AOI-sized. Every array is tile-sized.** A tile is a fixed number of texels at
a fixed metres-per-texel; the DEM and NAIP are read per tile window out of the COGs; the
AOI decides only how many tiles exist. `Detail` (m/texel) and `Tile texels` replace mesh
density, texture size and drape tiles, and they sit in the top row with the DEM source
because they decide what gets fetched.

## The tile grid is a quadtree over degrees

Level k means `2^-k` degrees, and a lattice coordinate is `(i * T + n) * step` for integers
`i`, `n` with `step` a power of two. Verified bit-identical shared edges at levels 4, 6, 8,
11 and 14 across CONUS tile indices.

Two properties follow: neighbours agree on their shared coordinate exactly, and the grid
does not move when the box moves, so a tile means the same ground on every redraw. The
second is what makes this ready for a camera-driven tile set without re-deriving anything.

**Cost: overhang.** Tiles are whole, so a box that does not land on tile boundaries is
covered by a larger rectangle. Measured 1.18x at level 6 over a 24 km box, 2.00x at level 4
over a 13.5 km box. Worst when tiles are large relative to the AOI, i.e. at coarse Detail,
which is the cheap end anyway. Printed on every run, with a NOTE above 1.5x.

## The fold needs no margin: partial aggregation

`avg`, `min` and `max` are decomposable. Each tile runs a `GROUP BY` over only the pixels it
owns (half-open clip in `[w, e) x [s, n)`, so nothing is double counted) and one final
`GROUP BY` merges the partials by cell id. An H3 cell straddling a tile boundary gets ONE
value, not one per side.

Verified against a single global fold over 200k pixels partitioned into 88 tiles:

| column | max abs difference |
|---|---|
| `avg(elevation)` | 9.09e-13 |
| `max - min` (relief) | 0 |
| `count` | 0 |
| `avg(ndvi)` | 2.22e-16 |

This is why tiling costs the analysis nothing, and it is a better answer than the margin
trick it replaced.

## The halo, and what it buys

Every tile reads `HALO = 24` texels beyond its own edge. Smoothing and the hillshade
gradient run on the halo and are cropped after, so the interior equals a single global
operation.

* Blur: equals a global blur to **1.3e-12 m** at radius 4, **5.4e-13 m** at radius 16 (on a
  400 m synthetic surface). Not bit-exact because `box_sum` accumulates with `cumsum`.
* Hillshade: neighbours agree **exactly** (0.0), because a gradient accumulates nothing.
  The no-margin control measures **0.0075 in shade units** at the same edge, which is the
  bright line the halo removes.

HALO is fixed rather than derived from `smooth`, so moving a scene control never re-streams
a COG.

## The seam check corrected its own comment

The height cell measures the worst disagreement between adjacent tiles along shared edges,
every run. It fired at **2.44e-4 m** on the 24 km target scene, and the explanation in the
docstring was wrong: the lattice COORDINATE is bit-identical, but the HEIGHT is not,
because two tiles read different WINDOWS of the same COG and `bilinear` divides by
window-derived numbers. 2.4e-4 m is float32 eps at 2000 m, four orders below the float32
floor the positions are quantised to anyway. Threshold is now 1 cm, where a seam could
actually be seen. On real data at 10 m and at S1M 1 m it reads **0 m**.

## Two real floors, neither of them the architecture

* **float32 positions.** `SurfaceLayer.positions` is `list<float32, 3>` and lonboard exposes
  no coordinate origin. Measured at the Wasatch box: **0.64 m E-W** (longitude -111.7 is in
  the [64, 128) binade) and **0.42 m N-S**. Below about 1.5 m/quad the mesh is
  precision-limited. Printed as a NOTE, calculated live because the floor moves with
  longitude.
* **Texture is area over texel area.** 24 x 25 km at 1.7 m/texel is ~1.08 GB and **no tiling
  changes that number**. Only view-dependent residency does.

## Measured

| scene | tiles | m/texel | m/quad | texture | wall |
|---|---|---|---|---|---|
| 13.5 x 12.2 km, Detail 16 | 9 | 13.59 | 54.4 | 9.5 MB | ~25 s |
| 13.5 x 12.2 km, Detail 4 (default) | 88 | 3.40 | 13.6 | 92.6 MB | ~60 s |
| 42.3 x 40.1 km, Detail 8 | 221 | 6.79 | 27.2 | 232.6 MB | 90 s |
| 23.7 x 25.0 km, Detail 2, T=1024 | 304 | 1.70 | 6.79 | 1277.6 MB | 3 m 54 s |
| 4.5 x 4.5 km, S1M, Detail 2, Relief | 42 | 1.70 | 6.79 | 44.2 MB | ~90 s |

All 100% covered, 100% painted, 100% opaque.

**The headline, proven directly:** 13.5 km and 42.3 km boxes at the same Detail both return
**13.59 m/texel** and both read the same 10.3 m overview. 3x the width, identical
resolution, 9 tiles versus 63.

**Against the scene that produced the sawtooth** (23.9 x 25.0 km, reported 1.46 m/texel on
23.3 m quads, texture/geometry ratio 16): the new one gives 1.70 m/texel on **6.79 m quads**,
a ratio of 4. That ratio is now a single control (`Texels / quad`) rather than two numbers
that did not know about each other.

## S1M's two old humiliations were the same bug

"Reading the 16 m overview of a 1 m product" and "64 tiles is too many" were both
`AOI / tex_size` talking. Read resolution comes from Detail now, so **1 m S1M reads native
1 m at any box size**. Verified over Asheville: 4.5 km box, 2 COGs, 100% coverage, native
1 m, 42 tiles, seam 0 m over 71 edges, fold merging 42 partials at 12,924 DEM px/cell.

## Things that bit during the build

* **`TILE_CONCURRENCY` in two cells.** marimo requires globals to be unique across cells;
  the fix is an underscore prefix, which makes a name cell-local. My own DAG checker missed
  it because it only reads args and returns. marimo's error caught it.
* **The NAIP loop was sequential over tiles**, inherited from a notebook where 16 tiles was
  the maximum. At 221 tiles serial round trips are the entire runtime, because each window
  is small and the cost is latency. Now bounded concurrency (4 tiles x `naip_rgb`'s own cap
  of 8 = at most 32 reads in flight). The original comment argued for a bound and
  implemented a one.
* **Geometry had no budget.** Decoupling triangles from the tile count is the point of the
  notebook, so the total grows with both: 99 tiles at one vertex per texel is 52M triangles
  and 313 MB of positions. Gated in the mesh cell rather than the tile-grid cell, so that a
  scene control does not sit upstream of the DEM stream.
* **The caps were arbitrary and too tight.** 144 tiles refused a 42 km box at 8 m/texel that
  needs 221 tiles and 511 MB. Constructing 1024 SurfaceLayers measures 0.51 s, so the pool
  is not the cost; draw calls are, which is now a soft NOTE above 384 rather than a refusal.
* **Dependency direction matters more here than it did.** The NAIP coverage gate had to move
  out of the tile-grid cell, because that cell produces `tiles`: leaving it there would have
  made every change of `Colour by` rebuild the tile set and re-stream every DEM window.

## Still unbuilt, and now the obvious next thing

**View-dependent residency.** The tile scheme is already global and camera-independent, so
this is a different tile *selection*, not a different architecture: pick the level from
camera altitude, load what is visible, evict what is not. That is the only thing that lifts
the texture ceiling, because `area / texel²` does not care how the ground is cut up.

The blocker is known and is not technical mystery: lonboard's `onViewStateChange` writes
`view_state` to the Python model **raw, every frame of a drag, with no throttle**, which is
what has burned this project before. The pattern that works is an observer writing to a
plain dict and a debounce timer promoting it to `mo.state` once the camera has been still
for ~400 ms and has actually moved, with nothing heavy depending on the raw trait.
