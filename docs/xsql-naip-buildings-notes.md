# Buildings in the surface: what was measured

Running account for `xsql-naip-buildings.py`, which is `xsql-naip-tiles.py` with Overture
building footprints folded into the height field. Written down because three of these cost
real time to find and none of them are guessable from the symptom.

## Reading Overture is a catalog problem, not a bandwidth problem

A theme is ~512 GeoParquet files of about 500 MB on `overturemaps-us-west-2`, and Overture
publishes nothing that says which file holds which ground. Three ways in, measured on the
same Sedona box:

| approach | first run | per AOI after |
| --- | --- | --- |
| `GeoParquetDataset.open(all 512)` | fails | fails |
| DuckDB `read_parquet` + bbox predicate | 35 s | 35 s |
| cached file-bbox index, then read the hits | ~100 s | **1.4 s** |

The dataset open does not fail on size, it fails on **type**: the geometry column is
Polygon in some parts and MultiPolygon in others, and a GeoParquet dataset insists on one.
Per-file reads sidestep it entirely, and `load_wkb` concatenates through WKB, which is a
binary column with no opinion about geometry type.

The index is 512 GeoParquet footer reads at concurrency 32, cached as JSON in `.cache/`
keyed by release and theme. It never changes under a pinned release. `overture_core.py`
does this generically over `THEMES`, so places or transportation cost the same 100 s once
and the same second afterwards.

Release ids are **deleted**, not archived: `2026-01-21.0` was already gone when this was
written, hence `releases()` and a pinned `OVERTURE_RELEASE`.

## Height coverage is partial and it varies by city

| AOI | footprints | with `height` | with `num_floors` |
| --- | --- | --- | --- |
| Sedona, AZ | 9,837 | 72% | 0.3% |
| Downtown Salt Lake, UT | 3,102 | 58% | 13% |

So the ladder is `height`, then `num_floors * 3 m`, then a control that may be zero
meaning "do not invent this one". The fetch cell prints the split every run. Sedona is
low-rise: median 4.7 m, tallest 10.8 m.

## A roof is flat and the ground under it is not

Adding `height` to whatever the terrain does inside a footprint gives a building with a
sloping roof sunk into a pit on the uphill side. Each footprint instead gets **one base**,
sampled at its centroid from the tile that owns the centroid, and every texel inside the
polygon is raised to `base + height` with a `maximum` so overlapping footprints keep the
taller roof.

One base per BUILDING rather than per TILE is a correctness rule, not an aesthetic one. A
footprint crossing a tile boundary is stamped into both tiles; per-tile bases would step
the roof by the terrain difference exactly along the seam. The existing seam check picks
this up for free and measures **0 m over 31 shared edges** with buildings on. If the base
were ever sampled per tile that number would jump to metres.

Stamping is per polygon over its own index window (`searchsorted` into the tile lattice,
then `shapely.contains_xy` on a ~30x30 grid against prepared geometry). A full-lattice
test would be `(T + 49)^2` points per building, which is minutes.

The height field is left **uncropped** through the smoothing cell now, because both of
those things need the halo: a footprint on a tile edge has to be stamped into the
neighbour's halo too, or the blur and the gradient see a wall on one side and bare ground
on the other.

## deck fails silently, well below the cap

The first defaults here were 1 m/texel with a vertex on every texel, which is what
buildings want in isolation. On a 5 x 4.5 km box that is 192 tiles, **606 MB of positions
and 100.7M triangles**, and what happens is:

- the basemap renders
- the camera works
- no Python error, no `mo.stop`, every cell prints normally
- the surface simply never appears
- the only console output is an unrelated `BitmapLayer({id: 'undefined'})` assertion from
  the picker map, which sends you looking in the wrong place entirely

The 512 MB `MAX_POS_MB` gate did not catch it because 606 MB was reached only in the
configuration nobody ran, and everything between about 128 MB and the cap is in the region
where deck may or may not draw. So there is now a **soft warning at 128 MB** that names the
symptom, and the defaults start where a town certainly renders:

| Detail | Texels / quad | tiles | positions | triangles | draws |
| --- | --- | --- | --- | --- | --- |
| 4 m | 2 | 15 | 12 MB | 2.0M | yes |
| 2 m | 2 | 63 | 50 MB | 8.3M | yes |
| 1 m | 1 | 192 | 606 MB | 100.7M | **no** |

The cost is quadratic in both directions at once, which is why stepping Detail down one
notch and Texels/quad down one notch together is a 16x move, not a 2x one.

## What this does not do

No per-building picking or attributes: the buildings ARE the terrain by the time deck sees
them, so there is nothing to click. That is the argument for the other implementation, an
extruded `PolygonLayer` over the surface, which would need the same per-building base
elevation computed here and would z-fight wherever the two disagree, but would carry names,
classes and heights into a tooltip. Both are legitimate; this notebook is the first one.

Footprints smaller than the lattice are counted (`too small for the lattice`) rather than
dropped quietly, and the mesh samples the height field every `Texels / quad` texels, so a
house can be in the shading and not in the geometry. At 4 m/texel on Sedona that was 184 of
5,162.
