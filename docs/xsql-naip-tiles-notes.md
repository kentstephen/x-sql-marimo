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

## The layer pool was a workaround for a constraint that does not exist

Reported as "just the picker is really laggy, shouldn't be 1.1 GB in the browser before
run". Two things were true at once and only one of them was mine.

**The picker was never changed.** Diffed cell by cell against `xsql-naip-ndvi.py`: the
picker `Map` and the coverage layer differ by COMMENT TEXT ONLY, zero code difference. I
had asserted the pool was the cause before measuring anything, which was wrong of me to
state as fact.

**What actually sits in that browser tab is two maps.** The scene `Map` cell deliberately
depends on no control, which is what stops a slider from resetting the camera, but it
therefore does not depend on the first-run latch either, and `mo.stop` only halts
DESCENDANTS of the cell it is in. So the scene map is constructed at notebook open, while
the pipeline is still parked at the picker, and it was constructing a fixed pool of 1024
SurfaceLayers to have them ready. I had raised that from 144 to 1024 to fit the 304-tile
scene, which made a pre-existing cost 7x worse.

**The pool existed because of an assumption that is false.** Every notebook in this repo
carries the comment "`Map.layers` must not be reassigned: that throws away the camera", and
that is why they pre-allocate. lonboard only recomputes the view from the layers inside
`add_layer(focus=True)` and `reset_zoom=True`; `view_state` is an independent trait the
frontend owns. Verified: fly the camera, grow the list 1 -> 88, shrink it 88 -> 9, and
`view_state` is unchanged both times.

So there is no pool. The Map is built with ONE blank layer and the update cell grows the
list to the tile count and trims it back, removing surplus layers rather than blanking
them. Idle cost is one draw call of nothing. `MAX_TILES` now bounds draw calls rather than
a pool, and no longer has to equal anything in the Map cell.

Worth carrying to the other notebooks: `xsql-naip-drape.py`, `xsql-naip-ndvi.py` and
`xsql-s1m-surface.py` all pre-allocate on the same false premise, though at 16 layers
rather than 1024, so it costs them much less.

Two process notes, both earned:

* **State a diagnosis as a hypothesis until it is measured.** "That's the 1.1 GB" was a
  guess dressed as a finding, and Stephen was right to stop it.
* **"Why would that even run?" is usually a better question than "how big should it be?"**
  The answer here was that the thing should not have been running at all, which no amount
  of tuning the constant would have reached.

## OPEN, NOT DECIDED: what H3 is still doing here

Raised at the end of the session and left unresolved. Nothing below has been changed.

**H3 was scoped, not dropped.** The gate is `surface.value not in ("NDVI", "Relief")` at
`xsql-naip-tiles.py:1511`; on `NAIP RGB` and `Elevation` the fold is skipped and says so.
The default surface is `NAIP RGB`, so on the view the notebook opens into, H3 does nothing.

**There is one real leak onto the RGB path, and it is a bug.** In the DEM-selection cell:

```python
_for_fold = SAFETY * np.sqrt(H3_CELL_M2[h3_res.value])
_target_m = min(_for_fold, m_texel)
```

This runs unconditionally. So on `NAIP RGB` the `H3 resolution` dropdown still narrows which
overview gets streamed, for a fold that never runs: at res 13 it pulls the DEM read down to
4 m on a scene that never reads a cell value. Leftover from when the fold fed the geometry.
The fix is to let `_for_fold` into the minimum only when the fold will actually run. NOT YET
APPLIED.

**Stephen's verdict on NDVI: not worth doing.** Second time he has said this (the previous
session's notes record the same). The known cause is that the NDVI surface paints
`cell_field("ndvi", _k)` at `:1941`, i.e. res-10 hexagon averages, while the sharp per-texel
NDVI array computed one cell upstream is right there. It is a one-line change and has never
been made, so "NDVI looks awful" has never actually been tested against the sharp version.

**Relief is the only surface where the fold buys something no resampling can produce**:
`max - min` inside a cell is a statistic over a neighbourhood of pixels.

Three directions, none chosen:

1. Fix the leak and make NDVI sharp. H3 then stands behind Relief alone.
2. Cut H3 out of this notebook. Relief becomes a local min/max window over the height field,
   which drops DataFusion, h3ronpy and xarray-sql from the render path entirely. The
   notebook stops advertising a spatial index it barely uses.
3. Give H3 the job the notes have argued for across three sessions and never built: the
   VECTOR JOIN. Overture buildings on S3 as GeoParquet, polyfilled with
   `h3ronpy.vector.geometry_to_cells`, `LEFT JOIN`ed to the terrain aggregate on cell id.
   That is a thing a raster pipeline genuinely cannot do, unlike everything H3 is currently
   being asked to do here.

Worth noting that 2 and 3 are not opposites: cutting H3 out of the RENDER path and using it
for a join are the same argument, which is that a spatial index should be doing joins rather
than resampling rasters.

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

## Imagery is a fetch switch, not a colour one

`NAIP imagery` sits in the top row with `Detail`, `Tile texels`, `H3 resolution` and
`DEM source`, because turning it off is a decision about what gets STREAMED. Off means the
STAC search never runs, no quad is opened, and the fold is skipped.

One derived `view` carries it: `Colour by` unless imagery is off, in which case the two
imagery surfaces read as `Elevation`. Every consumer reads `view` rather than
`surface.value`, so the switch is honoured in one place instead of eleven, and no cell has
to decide separately whether a NaN texture means "no coverage here" or "imagery is off".

It lives in its own cell AHEAD of the coverage gate, because that gate ends in `mo.stop`
and anything defined beside it stops with it. The legend has to keep working when the gate
trips.

**The placeholder arrays were not free.** The no-imagery branches allocated one zero RGB
array, one coverage mask and one all-NaN NDVI grid PER TILE. At 88 tiles and 513² that is
about 180 MB whose only content is "no imagery here"; at 500 tiles it is a gigabyte,
competing with the DEM on exactly the wide boxes where imagery gets switched off. Now one
shared read-only array each, with `writeable = False` enforcing that nothing writes through
it.

## The DEM stream had no timeouts and no voice

Two separate bugs with the same symptom, both only visible on a wide box.

`S3Store` was constructed with no `client_options` and no `retry_config`, unlike the NAIP
path in `naip.py`, which carries both and documents why. obstore's default `timeout` is 30 s
wall clock from request to last byte. That is right for the small reads it was built for and
wrong here: hundreds of tile windows share one link, so each read runs long while the
transfer is making steady progress the whole time. Now 180s/15s/60s with 6 retries, the same
numbers and the same reasoning as the imagery path.

The cell's only `print` came AFTER `asyncio.gather`, so a stall anywhere produced total
silence, which is indistinguishable from a hang. It now prints the open phase, the S1M
lon/lat fits (serial, main thread, one per COG, the other place a wide box goes quiet) and
tiles/s as it goes. Measured 19 to 24 tiles/s against `prd-tnm` for the 88-tile Wasatch box.

## The interactivity ceiling is the widget bridge, and one attempt at it failed

Measured, so that nobody re-derives it:

| | cost |
|---|---|
| `apply_continuous_cmap` over the whole 88-tile grid | 0.20 s |
| the same via a 256-entry LUT and one `take` | 0.19 s, i.e. 1.1x |
| RGBA for that grid across the widget bridge | ~92 MB |

**The kernel was never the problem.** Every colour control was resending the picture. That
is what makes the notebook stop feeling interactive on a wide box.

### What worked

`Reverse ramp` is back next to `Ramp`. Brightness, Hillshade and Height smooth are
debounced, since each rebuilds every texture in the grid and a drag was costing one rebuild
per tick of travel.

### What failed, and the number that kills it

Attempted and reverted (`2d62772`, reverted in `31a3eda`): send two bytes per texel, index
plus hillshade, with the ramp as a separate 256 x 4 `ramp_lut` trait, and expand to RGBA in
the browser via a patched `prepareTexture`.

Everything about it worked except the thing that mattered:

* Payload 46.3 MB where it was 92.6 MB.
* The patched JS verified under node against the exact string the patch script writes: index
  0 transparent, shade multiply correct, RGBA passthrough intact, missing table renders
  transparent rather than noise. Expansion matched the old kernel-side colouring to 3 counts
  per channel, which is the 255-level quantisation.
* Python side verified: `ramp_lut` is a synced key serialising as a 1024-byte buffer, and
  the `{height, width, data}` dict goes round `TextureTrait`'s 3-or-4-channel check because
  that trait passes such a dict through as already-validated state.

**And it ran the tab out of memory on an AOI that had always fit.**

    RangeError: Array buffer allocation failed
      at new Uint8ClampedArray
      at QS.prepareTexture

| | browser RAM |
|---|---|
| RGBA payload | 92 MB, and `prepareTexture` made a VIEW over it, allocating nothing |
| indexed payload | 46 MB, PLUS 92 MB of expansion, all 88 tiles alive in one render pass |

The bridge was halved and browser memory went up 1.5x. `new Uint8ClampedArray(buffer,
offset, length)` is a view; `new Uint8ClampedArray(n * 4)` is an allocation, and there is one
per tile per render. marimo's own renderer was failing `fromBase64` alongside deck, which
says the whole tab was out of headroom rather than deck specifically.

**Price browser memory, not just wire bytes.** CPU-side expansion in the browser always
holds both representations, so it cannot win this trade at any tile count.

### The version that is actually better on every axis

Upload the index as an `r8unorm` texture and do the palette lookup in the fragment shader.
Nothing is expanded on the CPU, so there is no second copy and no per-render allocation:
browser RAM 92 -> 46 MB, GPU 92 -> 46 MB, wire halved, palette change still 1 KB.

The cost is a real shader hook. `SurfaceLayer` renders through `SimpleMeshLayer`, which
takes no modules, so this means subclassing inside the minified bundle rather than splicing
a string like the two existing patches. The bundle does ship deck.gl-raster with a
`renderPipeline` of shader modules, which is the mechanism worth copying. Prove the memory
claim on a small grid before pointing it at 88 tiles.
