# Fire risk x Overture buildings: what was built and what was learned

`xsql-firerisk-buildings.py`. CarbonPlan's 30 m CONUS wildfire risk pyramid folded to H3 in
DataFusion, joined onto Overture building footprints on the H3 cell id.

Status: **runs, validated against the raster it reads, headless export clean.** Built as a
copy of `xsql-deforest-divisions.py` with the raster and the vector layer swapped. No drawn
box yet: that was deliberately deferred.

## The data

`s3://us-west-2.opendata.source.coop/carbonplan/carbonplan-ocr/output/fire-risk/pyramid/production/v1.1.0/pyramid.zarr`

Source Cooperative, [carbonplan/carbonplan-ocr](https://source.coop/carbonplan/carbonplan-ocr),
CarbonPlan's Open Climate Risk project, CC-BY 4.0.

| | |
|---|---|
| format | **Zarr v3 multiscale**, 12 levels, 2x each |
| CRS | EPSG:4326, rectilinear, 1D `latitude`/`longitude` in degrees |
| L0 | 97,579 x 208,881, chunks 362 x 362, float32, fill NaN |
| resolution | 0.000308 deg, ~34 m N-S and 22-31 m E-W across CONUS |
| coverage | CONUS only, bbox 22.7-51.8 N / -128 to -65; **48.7% of the rectangle is NaN** |
| variables | `rps_2011`, `rps_2047` |

### `rps` is Risk to Potential Structures, and it was verified not guessed

The pyramid publishes no `long_name` or `units`. The name was settled from the sibling
**Icechunk** store at `.../fire-risk/tensor/production/v1.1.0/ocr.icechunk`, which carries
the model decomposition: `bp_2011`, `bp_2047`, `bp_2011_riley`, `bp_2047_riley`,
`crps_scott`, `rps_scott`, `rps_2011`, `rps_2047`.

```
corr(rps_2011, bp_2011 * crps_scott) = 1.000000    median ratio 1.0000   (160k px)
```

So RPS = BP x cRPS, the USFS Wildfire Risk to Communities formula (Scott et al.), and all
three are intensive, which is what makes `mean()` valid at every scale. Sample medians:
BP 0.0059, cRPS 34, RPS 0.164.

**A trap the decomposition creates:** mean(RPS) is not mean(BP) x mean(cRPS). Aggregate RPS
itself. Averaging the factors and multiplying gives a different, plausible-looking, wrong
number.

### Distribution, which sets the ramp

CONUS at L6, 2.19M cells: zero 3.6%, p25 0.0029, p50 0.023, p75 0.105, p90 0.356,
p95 0.643, p99 1.91, p99.9 4.91, max 11.5. Ramp is cividis, log10, `LO..HI = 3e-3..5.0`,
which is p25..p99.9.

Zero rises to 10-22% inside a city (water, dense urban core, irrigated cropland). Unlike
the deforestation notebook, **zero cells are kept**: there zero was ocean and dropping it
cleaned up the map, here zero is ground that will not burn and it is exactly where the
buildings are. Dropping it would punch holes through every city and strand the footprints
inside them with nothing to join to.

## Zarr against the COG: three things that come free

1. **The pyramid is declared.** Root metadata carries a `multiscales` convention block
   where every level states `"resampling_method": "mean"`. The Vizzuality COG's averaging
   had to be reverse-engineered by measuring. Checked anyway, one 1-degree Sierra box:
   mean 0.4579 / 0.4587 / 0.4558 at L2 / L4 / L6.
2. **Sparseness is in the spec.** Chunks outside CONUS are absent (`0/rps_2011/c/0/0` is a
   404) and "absent chunk reads as fill_value" is something every reader implements. The
   COG's `tile_byte_counts` check and its `Invalid range requested, start: 0 end: 0` crash
   have no equivalent.
3. **Coordinates are published arrays**, so there is no geotransform to derive and no
   assumption that the grid is regular. Both axes ascend. At L0 the pair is 2.5 MB, read
   once per level and kept.

obstore stays the transport: zarr 3 takes `zarr.storage.ObjectStore(S3Store(...))` directly.

## Resolution ladder

`res_for_zoom`: one H3 resolution per 1.4 zoom levels, `ZOOM0 = 3.0`, res 4 to 11.

| res | cells | reads | px/hex |
|---|---|---|---|
| 4 | 1,770 km2 | L8 (8.8 km) | 23 |
| 5 | 253 km2 | L6 (2.2 km) | 53 |
| 6 | 36.1 km2 | L5 (1.1 km) | 30 |
| 7 | 5.16 km2 | L3 (274 m) | 69 |
| 8 | 0.737 km2 | L2 (137 m) | 39 |
| 9 | 0.105 km2 | L1 (68 m) | 23 |
| 10 | 15,047 m2 | L0 (34 m) | 17 |
| 11 | 2,150 m2 | L0 (34 m) | 2.4 |

**Res 11 is the floor and the raster sets it.** A pixel is 770-1,070 m2 depending on
latitude; a res 12 cell is 307 m2 and would hold ~0.35 pixels, so 60-70% of cells would
catch no pixel centre and the layer would hole out. Same arithmetic as the NLCD floor
(2.3 px at res 11, 0.6 at res 12). Zooming past z12.8 keeps res 11 and draws it bigger.

Asked directly whether res 12 becomes available once the join is a polyfill: no. Both sides
of an equi-join must be the same resolution, so the polyfill cannot run finer than the
fold, and the fold is what holes out.

## THE ONE THAT COST THE SESSION: attributes exist only at z14

`buildings.pmtiles` is 179 GB, z0-14, layers `building` (declared minzoom 4) and
`building_part` (minzoom 8). The declared minzoom is about GEOMETRY. **Planetiler strips
the attributes off everything below the top zoom.** At z13 and under, every feature carries
`@geometry_source` and `@height_source` and nothing else.

| place | z13 feats | id present | z14 feats | id present |
|---|---:|---:|---:|---:|
| Paradise CA | 1,109 | **0** | 618 | 618 |
| Superior CO | 531 | **0** | 601 | 601 |
| Downtown LA | 4,162 | **0** | 1,875 | 1,875 |
| Malibu CA | 389 | **0** | 348 | 348 |

`id` is both the dissolve key and the join key, so a z13 fetch returns thousands of
anonymous polygons. **The failure is silent**: the directory walk succeeds, the MVT decodes,
the ring winding is correct, the feature count is right, and every feature is unusable. The
first symptom was `fetch_buildings` returning "empty" with no error anywhere, and the
suspect list ran through winding, layer naming and the tile grid before the properties.

So the camera zoom decides only WHETHER buildings are drawn. The tile zoom is always 14.

`class` is optional even at z14 (Paradise and Malibu have none, Superior and LA do), hence
the fallback through `subtype` to empty string.

Tile sizes, gzipped, z14: 33 KB Malibu, 45 KB Paradise, 64 KB Superior, 154 KB Santa Rosa,
177 KB downtown LA. A viewport at z13.6 is 28 tiles.

## The polyfill mode is `overlap`, and `center` would have returned nothing

The divisions notebook uses `center`, which keeps a cell when the CELL'S CENTRE is inside
the polygon. That is right when the polygon holds thousands of cells and the map has to
partition. A building is the opposite regime: 150-250 m2 against a 2,150 m2 cell, so it
contains no cell centre at all and `center` returns an empty set. `full` is worse, wanting
the whole cell inside the polygon.

The measured mode table in `deforest-divisions-notes.md` already said so: on a
Singapore-sized box, center 0, full 0, overlap 4, overlap_bbox 5.

**Why the objection to `overlap` does not transfer.** It was rejected for divisions because
counties tile the plane, so a cell on a shared border counts into both neighbours and a
narrow county fills with ground outside itself. Buildings are disjoint islands: two houses
sharing a cell genuinely share its value, and there is no partition to violate.

Measured: **1.87 cells per building** at res 11 over Paradise, so the typical house
straddles a cell boundary and gets the mean of two.

**The limit, stated in the notebook too:** a 200 m2 house is 10% of a res 11 cell, so its
number is the cell's number, 90% of which is ground the house does not stand on. That is
the 30 m raster, not the join. Where `overlap` earns its keep is the large tail, a
20,000 m2 warehouse covering ~9 cells and getting the mean over its actual footprint.

`avg` over the covered cells, not `max`. For a risk layer `max` is defensible (the worst
ground a structure sits on) and it is a one-word change; the line is commented so the
choice is visible rather than accidental.

`ST_Dump` is still required: `h3_polygon_wkb_to_cells_experimental` rejects MultiPolygon
and `_feature_wkb` always emits one.

## The fetch box is wider than the fold box, and that is not missing data

Buildings are fetched over whole z14 tiles grown by `BLD_PAD`, so the coverage is reliably
wider than the raster window. The first working run reported **2,570 of 6,378 buildings
"off-grid"**, which was pure accounting: those footprints simply sat outside the folded box.

Fixed by carrying the dissolved footprint's centroid (`cx`, `cy`) out of the dissolve and
trimming to the fold box before the join. The polyfill still runs over the full cached
fetch, so it stays memoisable across pans. Off-grid went 2,570 -> **1**, and now means what
it says: raster with no value under a building.

## Validation

`tools/itest_firerisk.py` extracts the notebook's cells by AST and drives a real run with
no browser or kernel. The load-bearing check: a building's joined RPS against the raster
value at its own centroid, read straight out of the L0 Zarr window.

```bash
uv run python tools/itest_firerisk.py
```

```
3,735 buildings compared to the pixel under their centroid
corr(joined RPS, centroid pixel) = 0.9803    median ratio = 1.002
```

That is what catches a mis-indexed window or an H3 resolution mismatch, neither of which
changes any row count and both of which look completely normal on screen.

Other checks that pass: res 11 at z13.6 and res 9 at z10, buildings off below `BLD_ZOOM`,
every footprint gets at least one cell, and 2047 median RPS (0.0101) above 2011 (0.0085) at
Paradise.

Opening view is Paradise, CA at z13.6. Median RPS under its buildings is 0.0085 against a
p50 of 0.057 for the surrounding L1 window: structures in town sit on lower-modelled-risk
ground than the forest around them, which is what a burn-probability model does with dense
development and is worth knowing before quoting a number.

## Open: reported glitchy in the browser, symptom not captured

The headless test passes and the first interactive run drew correctly, but Stephen reported
the map "kinda glitchy" and the session ended before the symptom was described. **Ask before
guessing.** Reading the controller afterwards found three real defects, any of which could
be it. None is exercised by `tools/itest_firerisk.py`, which drives `refresh` directly and
never goes through the comm handler.

1. **The scenario dropdown is dead until the map is moved.** The opening draw calls
   `refresh(_VS(), force=True)` with a throwaway view object and never assigns `HOLD["vs"]`,
   which is only set in `_on_camera`. `_on_controls` guards on `HOLD["vs"] is not None`, so
   on a fresh load switching 2011 -> 2047 clears the cache and then does nothing, leaving
   the dropdown disagreeing with the map.

2. **Ground with no buildings re-folds the raster on every camera event.** `hide_buildings`
   sets `HOLD["bld"] = False`, and `_instant`'s `bld_ok` needs `HOLD["bld"]` true whenever
   `want_bld` is true. Above `BLD_ZOOM` in a place Overture has no footprints, that can
   never be satisfied, so `_instant` always returns False and every frame of a pan runs a
   full `_draw`. Needs a third state distinguishing "not fetched" from "fetched, none here".

3. **Outrunning the buildings box re-reads the raster unnecessarily.** `_instant` is
   all-or-nothing, so losing buildings coverage forces a full re-fold even when the cell
   cache still covers the view. The cells are the expensive half; the two steps should be
   able to invalidate independently.

Ruled out: `TILE_CAP`. Tile counts across latitudes are 45 at z13.0, 18 at z13.6 and 15 at
z14 against a cap of 64, so it never bites at a reachable zoom.

## Not built

- **The drawn box.** Deferred deliberately. The interesting version is a count of structures
  by risk band ("how many sit on top-decile ground"), which is the output that turns risk to
  POTENTIAL structures into a statement about real ones.
- **Extrusion.** `height` is present on 98% of z14 footprints, higher than the 55-75% the
  GeoParquet notes record. Dropped on the terrain notebook's lesson: an extruded layer
  buries what is underneath, and here that is the hexagons being compared against.
- **The Icechunk store.** Only needed for `bp_*` and `crps_scott`, which the pyramid does
  not publish. It would separate a county that burns often and mildly from one that burns
  rarely and catastrophically. Its chunks are 6000 x 4500 (108 MB each) with no pyramid, so
  the same 1-degree box costs 12.7 s against the pyramid's 0.2 s at L4: an archive, not a
  viewport read.
