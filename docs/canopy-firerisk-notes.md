# xsql-canopy-firerisk-buildings: the canopy x fire-risk x buildings pairing

**STATUS: PARKED**, in `archive/xsql-canopy-firerisk-buildings.py`. The pipeline works
end to end, but the pairing failed on meaning, not mechanics: RPS's fuel inputs
(LANDFIRE) already include canopy structure, height is the wrong axis of fuel (the
Marshall Fire footprint scores canopy ~0), and the structure-survival literature
(Knapp 2021 on the Camp Fire, Syphard & Keeley) finds spacing and materials dominate
while canopy effects are weak and direction-unstable. Painting a building by nearby
canopy height asserts "houses in the woods are at higher risk", which the evidence
does not support. What survives: the CHM strip reader (ported to
`xsql-canopy-deforest.py`, see `docs/canopy-deforest-notes.md`), the recon below, and
two unbuilt ideas recorded at the bottom (the RPS-vs-CHM disagreement layer, the DINS
per-structure test).

A whole-file fork of `xsql-firerisk-buildings.py` that adds Meta & WRI's High
Resolution Canopy Height Maps as a second per-building number: mean canopy height over
the building's res 11 cells widened one k=1 ring (~80 m), a defensible-space fuel
proxy. A `colour` control repaints the footprints by fire risk or by canopy; both ride
in the tooltip. The hexagons always carry fire risk. It also ports the HFP notebook's
measured-viewport ruler, so this fork does not have the fullscreen defect the parent
still carries.

## The dataset, measured, not read off the registry page

`s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/`, us-east-1, anonymous,
CC-BY 4.0.

- 56,147 tiles in `chm/`, named by 9-digit **zoom-9 Web Mercator quadkey**
  (`001311332.tif` decodes to tile x=126 y=39, verified against the tile's own
  ModelTiepoint). Each is 65,536 x 65,536 px at ~1.19 m, EPSG:3857, uint8 metres
  (`UNITS: METERS`), no nodata value.
- **BigTIFF, deflate, predictor 2 (horizontal differencing), 1-row strips.** Every
  strip is one full 65,536 px row; there is no internal tiling and there are **no
  overviews**. StripOffsets/StripByteCounts are LONG8 arrays (512 KB each per tile),
  so a window read fetches only the slice of those arrays it needs.
- Strip sizes swing two orders of magnitude with the ground: 320 B average over the
  whole archive (21 MB median tile), ~16 KB per row over Paradise CA (a 1.0 GB tile,
  dense conifer). Decoding is `zlib.decompress` then `np.cumsum(..., dtype=np.uint8)`,
  because uint8 cumsum wraps mod 256, which is exactly the predictor's inverse.
- `tiles.geojson` (15 MB) is the catalog with per-tile acquisition dates;
  `CHM_acquisition_date.tif` carries the same per pixel. Vintage is per-Maxar-strip,
  2018-2020 mostly: the same-year argument from the Sentinel-2 notes does NOT hold,
  and a cleared lot can still wear last year's trees.

## The msk sidecars are unusable, and that is measured

GDAL mask semantics say 0 = invalid. The Paradise tile's mask
(`msk/023010031.tif.msk`, classic TIFF, uint8, same strip layout) reads **all zero**
on rows that carry real 40 m heights, sampled at six rows across the tile. And only
46,448 masks exist against 56,147 chm tiles. A mask that flags live data invalid is
worse than none, so the notebook ignores them entirely. No-data therefore means **an
absent quadkey tile** (ocean, unimaged), which is cached per quadkey after one failed
HEAD-equivalent and surfaces as NULL canopy on the building, never as height 0.

## Why there is no free-fly canopy layer

Every other raster in this repo rides an averaged overview pyramid: `LEVEL_FOR_RES`
picks a level whose pixels sit just under the cell size the camera implies. The CHM
has no pyramid, so every read is full-res, and the 1-row strips mean every decoded
strip is a full 78 km row whatever the column window wants. A zoomed-out fold would
read gigabytes per pan. So the canopy is read the way the buildings are: only in the
buildings band (>= z13), per viewport, strided.

- **CAN_STRIDE 4 in both axes.** A res 11 cell holds ~1,500 native pixels and a mean
  needs a few dozen; stride 4 leaves ~94. The stride pays in the DECODE (three strips
  in four never inflated), not the fetch: the wanted strips are a KB-sized comb over
  the span and any range coalescing refetches the gaps, so the reader fetches the
  contiguous span in 8 MB pieces. Fetching exactly the comb as ~1,000 individual
  ranged GETs was considered and is a wash on time at this strip size; it would cut
  bytes ~4x if it ever matters.
- Measured, opening Paradise view (z13.6, 1400x620 assumed canvas): 99 MB span,
  ~4,100 rows, one tile. That is close to the worst case in CONUS; a median tile's
  viewport is ~2 MB. `CANOPY_BUDGET` (160 MB) refuses beyond that with a status note
  rather than stalling the map.
- Mercator both ways is closed form (rows: inverse Gudermannian; columns: linear), so
  after the Mollweide Newton solve in the HFP notebook this is the easy CRS. Pixel
  centres, +0.5, as everywhere in the repo.

## The join

Two mappings per building, one join statement:

- `bld_cells` (the parent's overlap polyfill) x fire cells: RPS under the footprint.
- `bld_near` = `h3_grid_disk(hex, 1)` over `bld_cells`, DISTINCT per id, in DuckDB
  (memoised like the polyfill) x canopy cells: fuel within ~80 m. LEFT join, because
  "no CHM tile" must arrive as NULL on a building that still has a fire number.

The canopy fold itself is the repo's standard UDF group-by in DataFusion, minus
xarray: the reader hands back already-flattened (lat, lng, metres) Arrow rows. Both
folds are at res 11, forced anyway because both sides of an equi-join must share a
resolution and the polyfill is pinned by the fire raster's floor. The canopy data
would support res 13 (~30 px per cell); that headroom is unused in this pairing.

## Render decisions

- **Zero canopy is INSIDE the ramp** (the HFP lesson, opposite of the RPS ramp's
  lifted floor): 0 m is a real measurement, the bottom of a continuum. The grey
  swatch means no data (no CHM tile).
- Linear 0-25 m on cividis. The burn ramp is cividis too (switched from inferno on
  request), so the legend and tooltip, not hue, say which mode the fill is in; the
  burn ramp keeps its lifted floor and dark zero swatch, the canopy ramp keeps zero
  inside the ramp. Both survive a deuteranope simulation; nothing is red-vs-green.
- The paint switch repaints from the joined table held in `HOLD["bldraw"]`: no fetch,
  no fold, no join.

## What was ported from HFP, verbatim

The Status ruler (shadow-DOM-recursive canvas search, ResizeObserver +
fullscreenchange, `"WxH"` Unicode trait), `view_to_bbox` on `HOLD["wh"]`, the 25 px
jitter gate, and the px readout in the status line (diagnostics left ON while the
fullscreen defect logged in the HFP notes stays open). The parent fire-risk notebook
still carries the guess.

## Ideas considered and where they stand

- **Repack the CHM as a Zarr/Icechunk pyramid** to get the free-fly fold back: right
  answer for a zoomed-out canopy layer, but it is a preprocessing job over 1.2 TB
  (global) or ~10s of GB (a state), and it trades away "no pixels move until the
  camera asks". Not needed for the per-building pairing; the strip reader is.
- **Zoomed-out building-density heatmap**: blocked on the same fact as the parent
  notebook's z14 pin (attributes and honest counts exist only at z14), so it is an
  offline pre-fold over Overture GeoParquet, not a viewport read. Pairs naturally
  with the pyramid idea above.
