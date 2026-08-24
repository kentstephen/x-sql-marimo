# AlphaEarth x Annual NLCD: the agreement map (`xsql-aef-nlcd-agreement.py`)

Started 2026-08-24. Stephen's framing: in the CDL + FTW work, CDL was the "language
key" that named the embedding clusters; NLCD is coarser (16 words at 30 m) so the
embeddings cannot be named by it, and the value they add is showing where NLCD's
words are stretched. First build: one year, one box, NLCD coloured as it is,
disagreement as alpha + cell coverage (his spec: "cell opacity or alpha and cell
coverage indicate agreement or not, not extruded").

## Data

| leg | where | shape | read |
|---|---|---|---|
| Annual NLCD | `s3://us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_<year>_CU_C1V1.tif` | COG, 30 m, EPSG:5070, 1985-2024 (NO 2025 on the mirror) | obstore + async-geotiff window |
| AlphaEarth mosaic | `s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/` | Zarr v3 (zarrs 0.23), one array `embeddings (time=9, band=64, y=1,859,584, x=4,009,984)` int8, nodata -128, EPSG:4326, 8.983e-5 deg (~10 m), 2017-2025, shards (1,64,4096,4096), inner chunks (1,64,256,256), zstd 3, NO PYRAMID | obstore `S3Store(prefix=...)` -> `zarr.storage.ObjectStore` -> `xr.open_zarr(chunks=None, consolidated=False)` (s3fs is not installed and not wanted) |
| AlphaEarth COGs | `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/<year>/<utm zone>/...tiff` | 8192² x 64 int8 per file, UTM per zone, WITH overviews (mean, renormalised), `aef_index.parquet` | not used yet; the wide-view route if one is wanted |

Dequantize per the store's own attrs: `(x / 127.5) ** 2 * sign(x)`. Verified: the
dequantized vectors are unit length (norm p50 1.000, p98 1.004), so averaging them
per cell and renormalising is the centroid direction and the norm of the mean is a
free homogeneity number.

Attribution required: "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind." CC-BY 4.0.

## Pipeline (measured 2026-08-24, from home)

Box: Folsom Lake / Auburn foothills, (-121.25, 38.70, -121.00, 38.95), ~22 x 28 km.

- NLCD 2024 window 937 x 1,090 px: read 0.7-2.3 s. Pixel centres -> lat/lon by a
  closed-form inverse Albers (Snyder, GRS80, fixed-point phi; max error vs pyproj
  3e-10 deg), passed as 2-D `lat`/`lon` data variables so the UDF reads columns.
  Fold res 10 (majority class + purity): 0.0-0.1 s, 38,944 cells.
- AlphaEarth 2024 window 2,783² x 64 = 496 MB raw: read 13-19 s (the wire). 64
  variables `e00..e63`, fold res 10 with 64 `avg()` columns: 0.4 s, 39,001 cells.
- Join + prototypes + scores: 0.1 s.

Everything after the reads is sub-second; the AEF read is the whole cost, and it is
a native read because the mosaic has no pyramid.

## The score

Per class with >= `MIN_CLASS_CELLS` (30) cells, the prototype is the mean of its
cells' unit vectors. Per cell: cosine to every prototype; agreement = sigmoid of
(own cosine minus best other cosine) / `TAU` (0.02). 0.5 = equidistant. A full
softmax was tried first and dropped: it dilutes by the class count.

Findings in this box (12 prototypes):

- Prototypes sit close together: developed-open vs herbaceous 0.96, deciduous vs
  evergreen 0.96, deciduous vs shrub 0.96, developed-open vs developed-low 0.96,
  water is the only outlier (0.3-0.65 to everything else).
- 37% of cells are closer to another class's prototype than their own. The flips
  are between NLCD's adjacent words: developed open space -> low developed /
  herbaceous / woody wetland / shrub (50% of its cells flip); deciduous forest ->
  shrub / woody wetlands / evergreen (51%); shrub -> deciduous / herbaceous (41%);
  herbaceous -> developed open / woody wetlands / shrub / pasture (42%). Water
  (13%), pasture (9%), developed high (9%), woody wetlands (12%) hold.
- The map reads accordingly: water, evergreen, developed medium/high solid;
  developed open space (the impervious-fraction definition, which looks like
  whatever surrounds it) is the most faded class.

## Encoding

NLCD's own colormap (Stephen: "color the way it is"; 23/24 are the official reds).
Alpha `ALPHA_MIN..ALPHA_MAX` (30..235) and hexagon coverage `COV_MIN..1` (0.30..1)
both linear in agreement. Coverage is per cell: deck's H3HexagonLayer has one
`coverage` for the layer, so the hexagons are built as a geoarrow.polygon column
(h3ronpy WKB, fixed 125-byte layout parsed with one frombuffer, vertices scaled
about the centre) on a PolygonLayer. Class dropdown dims the others to alpha 22;
"plain NLCD" mode turns the fade off. The map cell builds the layer once and the
wiring cell assigns `layer.get_fill_color` (repo rule: never rebuild the Map).

Verified in a driven headless Chrome (marimo run + playwright, 1500x1000): home
view and a 4-notch zoom both paint, no console errors.

## Open

- Prototypes are local to the box (right for "typical of its class HERE", but
  scores shift if the box moves). A camera-driven version would either refold per
  view or hold prototypes from a fixed wider window.
- Rare classes (mixed forest 7 cells, crops 5, emergent wetland 2 here) get no
  prototype; drawn plain.
- Res 11 multiplies cells by 7 and makes the NLCD side a relabel (~2.4 px/cell).
- `YEAR_AEF = 2025` against NLCD 2024 runs and adds a year of change to the score.
- Wide views: the COG mirror's overviews.
- Later, per Stephen: the historical component (AEF 2017-2025 vs NLCD's annual
  stack), once one year makes sense.

## TODO (Stephen, 2026-08-24)

- **Earth Genome Sentinel-2 mosaics (source.coop) under the hexagons**, so a patch
  of faint, shrunken cells can be inspected from the same notebook. Same-year
  imagery as the embeddings would make the disagreement checkable on the ground.
  The DATA side is already solved in `docs/imagery-and-terrain-notes.md` (the
  nlcd-imagery notebook read those mosaics before retreating to an Esri
  BitmapTileLayer because the lonboard render side never became stable). Open:
  whether lonboard 0.16's RasterLayer / BitmapTileLayer can serve them now; the
  deforest notebook's kernel-side tile callbacks are the likely route. Under the
  map/wiring split, an imagery layer would be a SECOND deck layer, which the
  cdl-crops saga says fails under marimo when one of the two updates; a static
  imagery tile layer under a static hexagon layer may survive (two static layers
  did).
- Variable zoom for CONUS (Stephen: "if we can continue and do variable zoom for
  conus i like this"). Constraint: the mosaic has no pyramid; the COG mirror
  does. Measured 2026-08-24: one COG opens in 0.8 s; overview 4 (512 px, 160 m,
  64 bands) reads in 0.54 s, overview 3 (1024 px, 80 m) 2.1 s, overview 2 (2048
  px, 40 m) 6.0 s; 20 files open concurrently in 0.9 s; 20 x overview 5 (256 px,
  320 m) in 1.1 s; 20 x overview 3 in 18 s (bandwidth-bound, as always from
  home). The 2024 index lists 1,990 files touching the CONUS bbox, one UTM zone
  each (EPSG:326xx), so a coarse CONUS view is ~2,000 ranged reads of tiny
  overviews (feasible in batches), and each view needs a UTM -> lat/lon inverse
  per file (closed-form transverse Mercator, or pyproj). Ladder sketch: res <= 8
  from COG overviews (the overview level chosen like NLCD's LEVEL_FOR_RES), res
  >= 9 from the mosaic (native, the current read), NLCD from its own pyramid at
  every rung, prototypes per view (or per state, held).

## 2026-08-24, second round (after Stephen's first run)

- `pickable=True` (his edit); the geometry table's columns are now the readable
  ones for lonboard's feature panel: class, agreement, looks more like, NLCD
  purity, homogeneity, cell.
- The marimo radio/dropdown are gone; the strip is the cdl-ftw-zarr-marimo
  `HudControls` skeleton (fade checkbox, pickable legend with per-class share and
  agreement p50 in the tooltip, panel line for the isolated classes, status with
  the three read/fold/score lines). Driven: 15 chips, click "Shrub/scrub" -> the
  panel reads "Shrub/scrub: 5,171 cells, agreement p50 0.63, 41% below 0.5,
  usually looks like Deciduous forest"; fade off -> flat NLCD.
- Map cell decoupled from every parameter (placeholder layer + literal camera;
  wiring assigns table + colours). Two lonboard lessons on the way: assignment
  wants an arro3 Table, and `_rows_per_chunk` must be reset to the new row count
  or the swap serializes as one-row chunks and draws nothing, with no error
  anywhere (kernel or console).
- Basemap DarkMatter with labels (Stephen: faint colours read better on dark).
- marimo 0.24.0 across the repo; the crops black-map defect did not reproduce.
- Third pass: paint buttons `agreement` / `NLCD` (regular hexagons: `geo_full`
  swapped in, same row count, `_rows_per_chunk` reset each swap); deck picking
  replaced by the strip's geometric click (worked once on his screen, then
  never; `pickable=False`); Positron again. Driven: chip isolate, "× all", NLCD
  paint, and a map click resolving to "Evergreen forest at 38.8361, -121.0647:
  agreement 0.98, NLCD purity 1.00, homogeneity 0.979".
- Stephen's next idea (open): a THIRD tier, the embedding on its own. Options
  laid out: nearest-prototype class in NLCD's palette; k-means clusters with
  a categorical palette and NLCD-composition legend; PCA-1 / similarity ramp.
- Fourth pass: the third paint, `AlphaEarth` = spherical k-means (K_CLUSTERS 10,
  k-means++ seed, numpy, ~0.2 s) over the cell vectors, Okabe-Ito+2 palette,
  legend chips = clusters with their NLCD make-up. In the foothills box the
  clusters read as: 0 = low/medium developed; 1-3 = open-space developed mixed
  with herbaceous (the developed-open-space confusion, seen from the other
  side); 4 = evergreen (+deciduous); 5 = herbaceous; 6 = shrub/deciduous mix;
  7 = evergreen/shrub mix; 8 = water. The seeding needs `1 - cos` clipped at
  zero (float32 cosines pass 1). Driven: paint switch, chips, click story with
  the cluster id.

## `xsql-aef-nlcd-conus.py` (2026-08-24): the camera-driven version

Stephen: "go ahead and build conus". Design decisions taken without asking (he
said no questions needed): two AEF sources by rung (mosaic res 10-11, COG
overviews res 5-9), a file budget above which the view is NLCD only, prototypes
and clusters per view, pyproj for the UTM tiles.

The COG path: index parquet (year slice cached under tmp, 1,993 CONUS files for
2024), per-tile open (cached; 64 concurrent), per-tile overview window from the
box transformed into the tile's UTM, pixel centres back to lat/lon with pyproj,
all tiles' pixels stacked into ONE 1-D Dataset (int8 x 64 + lat/lon) and folded
by ONE DataFusion query (dequantize in SQL with `signum`; DataFusion has no
`sign`). Overview per res: 9 -> 80 m, 8 -> 160 m, 7 -> 320 m, 6 -> 1280 m,
5 -> 2560 m.

Driven (headless Chrome, from home):

| view | res | NLCD | AEF | total |
|---|---|---|---|---|
| CONUS, zoom 4 (open) | 5 | L5 16.4 Mpx, 2.0 s | 1,993 files @ 2560 m, 2.0 Mpx, 20.6 s | 21 s, 31.5k cells |
| +3 notches | 5 | held | held | 0 |
| +6 | 6 | L4 10.5 Mpx 1.4 s | 293 files @ 1280 m, 4.3 s | 4.6 s, 35.7k cells |
| +9 (Colorado) | 7 | L4 (cached tiles) 0.2 s | 84 files @ 320 m, 4.5 Mpx, 8.8 s | 9.7 s, 58k cells |

Agreement at CONUS p50 0.81 (44% below 0.5); over Colorado at res 7 p50 0.35,
56% below 0.5: at 320 m herbaceous, shrub and the forests are one continuum to
the embedding and NLCD's cuts through it are the faded ground.

Open: the mosaic rungs (res 10-11) are not yet driven here (same code as the
one-shot); VIEW_W/H is a guess (the ruler port); cluster colours shift per view
(holding prototypes/clusters from a wider fold is the next knob); the SQL cells
under the map run on the current frame behind a run button.

### The south-up COGs (same day)

The first deep zoom read agreement 86-98% below 0.5 at res 9-10 (worse than
random: with labels uncorrelated to the vectors the best OTHER prototype beats
the own one most of the time) while res 11 from the mosaic read 14%. The AEF
COGs' transform is `| 10 0 418080 | 0 +10 5406720 |`: y pixel size POSITIVE,
origin at the SOUTH edge, `bounds` reporting bottom > top. The window and the
pixel centres now go through the affine transform. Re-driven: res 5 30% below
0.5 (p50 0.99), res 6 18%, res 7 19%, res 8 17%, res 9 15%, res 10 18%, res
11 14%. Also `MOSAIC_MIN_RES` 11 (res 10 from the 40 m overview; its padded box
would be ~1.8 GB raw from the mosaic). Full rung table: open 21 s; res 6 3.3 s;
res 7 8.6 s; res 8 12.3 s (107k cells); res 9 6.9 s; res 10 7.3 s; res 11
6.6 s (mosaic, 202 MB, 118k cells).
