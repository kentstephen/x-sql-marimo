# Imagery and terrain: what was tried, and what survived

Two attempts to put a second raster under the NLCD boundaries. One shipped in a reduced
form, one is parked. The findings are kept because most of them cost a session each, and
because the next person to have either idea should read the measurements first.

## Verdict first

**Extruded terrain (`xsql-duckdb-terrain-h3.py`): parked, on looks.** NLCD class joined to
Mapterhorn terrain on the H3 cell id, hexagons extruded by elevation. The join works, the
numbers are right, the extrusion is real. It reads as a novelty. Height is a weak encoding
for a categorical map, the relief is subtle at any exaggeration that does not also look
silly, and the extruded hexagons bury the dissolved outlines that are the good part of this
repo. Do not revisit without a better reason than "the data supports it".

**Imagery underlay (`xsql-nlcd-imagery.py`): shipped, but not with the imagery it wanted.**
The idea is sound and much better than the extrusion: thin vector lines over a photograph
is a combination that has always looked good, and it makes the map self-verifying, because
the boundary either follows a real edge on the ground or it does not.

It was built on Earth Genome's Sentinel-2 mosaics, which had a real argument behind them:
same year as the land cover, so a disagreement could only be classification error. **That
data side is fully solved and is written up below.** The RENDER side never became stable
through two architectures:

- `RasterLayer.from_geotiff` is the right architecture (browser-side range reads per
  visible tile, no server, no pixels through the kernel) and did not survive here.
- A Python-side `BitmapLayer` works but pushes ~950 KB of base64 per view through the
  comm channel, which will never feel smooth on a live camera.

**What shipped instead is a `BitmapTileLayer` on Esri World Imagery**, which is the one
imagery path in this notebook that always worked, because it is what already draws the
place labels. The cost is the same-vintage rule: Esri is a mosaic of many sources and
dates varying by location, so a boundary can disagree with the photograph because the
ground genuinely changed, and nothing on screen distinguishes that from classification
error. Good enough to judge whether a forest edge is roughly right; not evidence about a
particular year.

**Everything below is the way back**, if same-vintage imagery ever matters enough to spend
another session on the render side. Start by deciding whether the imagery has to follow the
camera at all: a frozen single scene sidesteps the whole problem.

## Sentinel-2 / Earth Genome: the data side (solved, and unused)

**Catalog is a STAC API, not a bucket listing.** `https://stac.earthgenome.org/search`,
collection `sentinel2-yearly-mosaics`, POST a bbox, get items whose `TCI` href points at
`data.source.coop/earthgenome/earthindeximagery/...`. No VRT parse, no MGRS index, no
`file_index()` cache. One round trip.

**The `datetime` filter does not constrain these items.** Asking for 2024 still returns
the 2023 mosaic for the same footprint. Enforce the year on the item id, which carries the
window explicitly: `13TDE_2024-01-01_2025-01-01`. Getting this wrong silently mixes
vintages across adjacent footprints, and a boundary judged against the wrong year's
photograph is worse than no map.

**Use `TCI`, not `B04`/`B03`/`B02`.** TCI is a precomposed 3-band uint8 visualisation
product: no stretch to choose, no per-tile brightness patchwork. It also sidesteps the
seam bug diagnosed in `sentinel-2-cog-deckgl-raster/docs/SEAMS.md`, where compositing
three separately-tiled single-band COGs leaves their tile grids disagreeing at sub-pixel
level and paints a faint grid over the whole scene. **TCI's fill is (0,0,0)**, so black is
nodata and must go transparent or every footprint paints an apron over its neighbours.

**The COGs are EPSG:3857 and already cut to the WebMercatorQuad grid.** This is the big
one and it was a surprise: no reprojection, no resampling, no warp anywhere in the path.
The overview pyramid *is* the tile pyramid. Verified from exact bounds (computing origins
from rounded bounds gives false "offset" readings):

```
L0    9.555 m -> z13  ALIGNED     L3   76.437 m -> z10  ALIGNED
L1   19.109 m -> z12  ALIGNED     L4  152.874 m -> z9   ALIGNED
L2   38.219 m -> z11  ALIGNED     L5  305.748 m -> z8   offset
```

So choosing a level for an H3 resolution is a table lookup and the window is integer
arithmetic. Imagery exists from about map zoom 8 to 13, i.e. H3 res 8-11. Outside that
range there is nothing to draw and the basemap shows through; that is the pyramid ending,
not a bug, and it should be stated on the page or it reads as one.

**One MGRS footprint is ~147 km across**, which covers any single view. Spanning several
means several layers, which is where the render trouble started.

## lonboard and deck: everything that cost a session

**`Tile.array`, not `Tile.data`.** An async-geotiff `Tile` carries a `RasterArray` on
`.array`; `.data` does not exist, and `reshape_as_image` refuses the `RasterArray` itself,
so it is `tile.array.data`. The render callback runs inside a comm handler where an
`AttributeError` is **silent**: every tile comes back empty, the layer reports as built,
and the map renders with no imagery and no error anywhere.

**`RasterLayer.from_geotiff` ships with its zoom bounds commented out.** In lonboard
0.16's own source:

```python
# min_zoom=0,
# max_zoom=len(tms.tileMatrices) - 1,
```

Its tile fetcher is `images[len(images) - 1 - z]`, so any `z` past the pyramid depth goes
**negative and wraps onto the full-resolution image**. deck then asks for a 15360x15360
read through a 512 px tile slot. Both bounds pass through `**kwargs`; set them. Set
`extent` too, or every layer claims the whole world and requests tiles for ground it has
no pixels for.

**WebP, not PNG, for tile encoding.** These are photographs. Measured on one z3 tile:
PNG 792 KB, WebP q88 83 KB, both about 21 ms to encode. Ten times the payload for nothing.

**A `BitmapLayer` with `image=""` poisons the entire deck update pass.** `image` is a URL
handed to the browser's image loader, and `""` is not one. deck initialises every layer in
ONE pass, so that throw aborts the batch and takes healthy layers down with it. The
symptom is a cascade of assertion failures naming layers that are fine
(`GeoArrowPolygonLayer`, `TileLayer`, `RasterTileLayer`), plus imagery that draws on the
first frame and dies on the next update. Use a 1x1 transparent PNG data URI as the
placeholder. **This mechanism is the likeliest explanation for both failed render paths,**
and it means an assertion naming a layer is weak evidence that the layer is at fault.

**No lonboard layer exposes an `id` trait.** Every layer reaches deck as `id: 'undefined'`.
That is noise in error text, not a cause; layers of different types coexist fine. Whether
two layers of the SAME type collide was tested and is *not* the failure here.

**`line_width_units` defaults to metres.** With `get_line_width` defaulting to 1, the
visible width is `max(1 metre in pixels, line_width_min_pixels)`: two numbers in different
units fighting over one line, where only the floor ever moves and nothing can take the
line *below* it. Set `line_width_units="pixels"` and `get_line_width` is the width
outright, sub-pixel included. Set `line_width_max_pixels` at the same time or a stale cap
silently clamps anything you ask for above it.

**`VIEW_W` is a guess; `VIEW_H` is enforced.** `height=VIEW_H` is real, nothing constrains
the width, so on a wide window the map is 1800-2200 px and `view_to_bbox` returns a box
narrower than the frame. Hexagons hide this (ragged edge, cells just stop); a bitmap does
not, and the symptom is "the imagery does not fill the frame" while the read, the bounds
and the encode are all exactly right. Over-read the imagery box independently of the
fold's pad.

**Sliders that commit on `input` send one comm message per drag pixel.** Fine for a few
floats on one layer. Not fine for anything that re-dissolves or re-pushes a table, which
should commit on `change`. Also: 12 stops across a 4.5rem track is ~6 px per stop, inside
the slop of a trackpad drag, and the control feels like it is fighting you. Give an aimed
slider 9rem.

## Directed edges instead of the polygon dissolve: measured, and it loses

Recorded here because the idea keeps coming back. Same forest viewport, 24,902 cells:

| approach | time | rows | payload |
|---|---:|---:|---:|
| dissolve + `ST_Dump`, min 20 | 17.3 ms | 85 | **0.110 MB** |
| directed edges, no despeckle | 38.7 ms | 106,422 | 6.07 MB |
| union-find despeckle then edges | 30.2 ms | 3 | 0.412 MB |

`ST_Dump` is the prize, not the polygons: splitting the multipolygon into connected runs
is what makes `min cluster` possible, and despeckling is where 98% of the payload goes.
Edges have no run grouping, so the union-find that was deliberately deleted has to come
back, and it is still heavier. Edges also do not chain, so shared vertices are stored
twice; `ST_LineMerge` chains them for 191 ms and saves no bytes. h3ronpy can *render* a
directed edge (`directededges_to_wkb_linestrings`) but cannot *construct* one from a cell
pair, so this was always DuckDB's job.

Related: the long comment about dissolving on parent hexagons and the outline "bulging
outward" was stale from the h3ronpy era. `WASH_SQL` dissolves at native resolution. The
outline is exact.

## Mapterhorn terrain, for the record

Kept because the reader works and the data is good, even though the notebook is not.

- `mapterhorn/mapterhorn/planet.pmtiles` on `us-west-2.opendata.source.coop`. z0-12,
  705 GB, 512 px terrarium WebP: `(R*256 + G + B/256) - 32768`. Verified on decode, the
  Mt Rainier tile tops at 4391.6 m against a true 4392.
- 457 regional `6-{x}-{y}.pmtiles` (z13-18, 26.7 TB) whose max zoom is effectively a
  source-resolution map. z13 is the 10 m level. Not needed against 30 m NLCD.
- PMTiles needs its own reader (~120 lines): header, gzipped varint root directory, leaf
  walk, Hilbert tile ids. Directory entries are run-length encoded, so a tile usually has
  **no entry of its own** and the binary search must fall back to the covering run.
- WebP decode is not the bottleneck: 4 ms/tile, 128 ms for a 30-tile viewport.
- **Clamp any global raster read to the other raster's footprint.** The DEM is global and
  the join is LEFT from CONUS land cover, so anything outside is decoded and discarded. It
  goes unnoticed because it is *correct*, just wasted: 37.75M DEM px against 4.10M NLCD on
  the opening draw, cut to 3.15M once clamped and the zoom table fixed.
- **Derive px/hex, then go and measure it.** The first zoom table was 4x too fine
  everywhere, in the safe direction, so nothing looked wrong while it quadrupled the tile
  count at four of seven resolutions. Caught only because a real fold measured 57 px/hex
  where the comment predicted 13.
- `async-pmtiles` + `RasterLayer.from_pmtiles` exists and would have replaced the hand
  written reader. Not tried.
