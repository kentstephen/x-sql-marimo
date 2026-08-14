# Deforestation x Overture divisions: what was built and what was learned

`xsql-deforest-divisions.py`. Vizzuality's global 100 m deforestation COG folded to H3 in
DataFusion, joined onto Overture division boundaries on the H3 cell id.

Status: **runs, numbers validated, render fixed, speed measured to the floor.** Session 1 is
below; session 2 (fill fix, camera fix, the Overture measurements, the `division_boundary`
test, the drawn-box ranking) starts at "Session 2" further down and supersedes open items 1
and 2. One live bug: the drawn-box ranking polyfills whole divisions at res 8, 596 s.

## The data

`s3://us-west-2.opendata.source.coop/vizzuality/lg-land-carbon-data/deforest_100m_cog.tif`

| | |
|---|---|
| CRS | EPSG:4326, full globe, -180/-90 to 180/90 |
| size | 5.7 GB, 200,376 x 400,752, float32, DEFLATE |
| tiles | 512 px, at every level |
| overviews | 10, average-resampled (verified, see below) |
| nodata | **none declared**; ocean comes back NaN |
| units | **portion of each cell deforested 2002-2022**, 0-1 |

The units decide most of the design. A portion is intensive, so `mean()` is valid at any
scale and no majority vote or mode is needed. That is why this is simpler than the NLCD
notebooks despite covering the whole planet.

The same bucket holds nine other layers (carbon, cropland expansion, BII, FLII). Same
shape, same CRS, so any of them is a one-line swap.

### The overviews really do average

Verified rather than assumed, over one 1-degree box in Rondonia:

| level | mean | max | exact-zero fraction |
|---|---|---|---|
| L0 (100 m) | 0.2260 | 1.0000 | 62.3% |
| L2 (400 m) | 0.2279 | 1.0000 | 41.6% |
| L4 (1.6 km) | 0.2312 | 0.9806 | 16.3% |
| L6 (6.4 km) | 0.2342 | 0.6472 | 0.0% |

Mean survives a 64x downsample; max and the zero fraction collapse. That is the signature
of `average` resampling, and it is what makes reading an overview equivalent to reading
pixels. If it had been `nearest`, the pyramid would be a subsample and the whole
level-per-resolution table below would be unsound.

### The distribution is why colour is hard

Folded at res 4 over the world: 224,238 cells, **69.6% exactly zero**. The nonzero part:

```
p1  7.3e-08   p25 8.7e-05   p75 1.5e-02   p95 9.5e-02   p99.9 4.5e-01
p5  8.5e-07   p50 2.1e-03   p90 5.3e-02   p99 2.4e-01   max   7.6e-01
```

Nine orders of magnitude. A linear 0-1 ramp paints a blank world.

## THE COG IS SPARSE, and async-geotiff does not know it

The single biggest gotcha. Ocean is simply not stored:

| level | tiles | zero-length |
|---|---|---|
| L0 | 306,936 | 225,967 (73.6%) |
| L1 | 76,832 | 54,518 (71.0%) |
| L3 | 4,802 | 2,945 (61.3%) |
| L6 | 91 | 10 (11.0%) |
| L7+ | 28 | 0 |

`async_geotiff` does not check `tile_byte_counts`, so a read touching an absent tile issues
a byte range `0..0` and raises:

```
TypeError: ValueError: Invalid range requested, start: 0 end: 0
```

That error names neither the tile nor the sparseness, so it reads like a corrupt file. It
is why the first world-view fold failed outright.

**Fix, and it is a speedup not a workaround:** read on the COG's own 512 px tile grid and
consult `ifd.tile_byte_counts` (already in memory, reshaped to the tile grid) before
requesting. An absent tile becomes a NaN block with **no network request**. Three quarters
of the planet costs nothing. Measured on the opening world view: 68 tiles fetched, 10
skipped.

This generalises to any sparse COG and nothing else in this repo had hit it, because NLCD
is dense over CONUS.

## Why H3 is not just a demo step here

I had this backwards for most of the session and it is worth writing down.

The COG is EPSG:4326, so **its pixels are not equal area**: a 100 m pixel at the equator
covers about twice the ground of one at 60 degrees. Averaging pixels directly over a
country spanning many latitudes overweights its poleward end. H3 cells are near-equal-area,
so folding to H3 and then averaging **cells** equally is an area-weighted mean, almost for
free.

So there are two different weightings doing two different jobs:

- **within a cell**: weight by valid pixel count (`px_total`). A coastal cell that is 90%
  NaN ocean must not count as a full one.
- **within a division**: weight cells **equally**. This is the area correction. Weighting
  by `px_total` here would reintroduce exactly the latitude bias that folding to H3 removed.

Getting this wrong is invisible on screen, which is the dangerous part.

## Resolution ladder

`res_for_zoom`: one H3 resolution per 1.4 zoom levels (each H3 step is 2.65x linear,
log2(2.65) = 1.4), floor not int. res 4 to 8.

**res 4 draws the whole planet comfortably**: H3 res 4 is **288,122 cells globally**, not
288 million. I got this wrong by a factor of 1000 mid-session and nearly designed a
pointless gate around it.

| res | cells globally | edge | area | reads |
|---|---|---|---|---|
| 4 | 288,122 | 26.1 km | 1,770 km2 | L6 (6.4 km) = 43 px/hex |
| 5 | 2,016,842 | 9.9 km | 253 km2 | L5 (3.2 km) = 25 px/hex |
| 6 | 14,117,882 | 3.7 km | 36.1 km2 | L3 (800 m) = 56 px/hex |
| 7 | 98,825,162 | 1.4 km | 5.16 km2 | L2 (400 m) = 32 px/hex |
| 8 | 691,776,122 | 0.5 km | 0.737 km2 | L1 (200 m) = 18 px/hex |

Measured: world at res 4, 15.7M px read in 821 ms + 282 ms fold -> 224,238 cells. Same
view again off the tile cache: 8 ms + 211 ms.

## Overture divisions: streaming, and where it breaks

`theme=divisions/type=division_area`, 5.5 GB in **8 files**.

**File-level pruning is useless here.** 7 of the 8 files have a bbox wider than 130
degrees, so nearly every viewport hits nearly every file. The only real pruning is row
groups, inside `GeoParquetFile.read_async(bbox=...)`, which works well when the box is
small and buys nothing when the box is the world.

Consequence: a world view of countries means reading most of 5.5 GB to find 219 rows, and
`subtype` is not partitioned so there is no cheaper way to ask. **Resolution: no boundaries
below zoom 4.5.** Regions 4.5-7, counties above 7. Zooming in inverts the cost the same way
the raster read does.

Row counts in one file: locality 57,072 / neighborhood 53,621 / county 11,375 / localadmin
1,858 / region 873 / country 67. Globally, three subtypes of interest: 219 countries,
3,919 regions, 38,909 counties. **171 of 219 countries have counties**, so the county band
is genuinely empty in places.

`is_land` matters: division_area carries maritime polygons, and polyfilling a country's
EEZ drags its zonal mean toward zero over water that was never at risk.

A local simplified cache was built and then removed (not asked for). For the record it was
62 s to build and 75 MB with tolerances of 0.01 / 0.005 / 0.002 degrees per subtype, a 94%
cut from 1,217 MB. If streaming ever proves too slow, that is the fallback and the numbers
are here.

## The polyfill

`h3_polygon_wkb_to_cells_experimental(wkb, res, mode)` in duckdb-h3.

**It takes a Polygon and rejects MultiPolygon**, with a message that blames the WKB:

```
Invalid Input Error: Invalid WKB: expected polygon at 5
```

That is 148 of 219 countries, 1,193 regions, 3,661 counties. So `ST_Dump` each division,
fill every part, flatten back to one distinct cell set per division. Not optional.

`ST_MakeValid` can return a GeometryCollection, which the polyfill also rejects; use
`ST_CollectionExtract(g, 3)` if you ever simplify.

**Containment modes are `center`, `full`, `overlap`, `overlap_bbox`** (not `overlapping`).
Measured at res 4:

| mode | 1-degree box | Singapore-sized box |
|---|---|---|
| center | 7 | **0** |
| full | 2 | **0** |
| overlap | 15 | 4 |
| overlap_bbox | 22 | 5 |

`center` is used, deliberately. `overlap` includes every cell that so much as touches a
division, so a county a few cells wide would have its mean substantially made of ground
outside it, and cells on a shared border would be counted into both neighbours. `center`
puts each cell in exactly one division.

The cost: a division smaller than one cell catches no centre and gets no number. Those are
dropped by the inner join and counted, so the status line can say how many rather than
letting them silently vanish from a choropleth that otherwise looks complete.

## Engine split

| step | engine | why |
|---|---|---|
| stream COG + GeoParquet | obstore | unsigned, concurrent, already the repo idiom |
| fold pixels -> H3 | DataFusion + h3ronpy | whole-column; DuckDB would call a UDF per row (70 ms vs 462 ms on 1.58M rows) |
| polygon -> cells | DuckDB h3 | only engine with a polyfill; few rows so per-row overhead is irrelevant and Uber's C wins |
| **join cells -> divisions** | **DataFusion** | integer equi-join + group-by. No geometry. The cells are already in that context |
| render | lonboard | |

The join being in DataFusion was Stephen's call and it is right. Shipping the cells to
DuckDB because DuckDB happens to hold the polygons would have been backwards.

EPSG:4326 is a large simplification against the NLCD notebooks: no Albers control grid, no
bilinear interpolator, no `to_lat`/`to_lon` UDFs. The pixel grid IS degrees, so `y`/`x` feed
`h3_latlng_to_cell` directly.

## Colour

cividis, log10 over 1e-4 .. 0.5 (which is p25 .. p99.9 of the nonzero values).

**The zero swatch has to be separated by luminance, not hue**, and this was measured, not
guessed. A flat neutral grey (78,80,84) lands at luminance 0.313 and the 0.1% stop of
full-range cividis lands at 0.318: "none" and "0.1%" came out as the same colour, which is
the worst thing this legend could do given zero was the majority case. Hue cannot fix it
because the point of cividis is that hue carries nothing. So the ramp floor is lifted off
the bottom of cividis (`FLOOR = 0.25`, upper 75%) and zero takes the dark end alone:

| stop | RGB | luminance | deuteranope luminance |
|---|---|---|---|
| none | (38,40,44) | 0.156 | 0.150 |
| 0.01% | (67,78,107) | 0.305 | 0.284 |
| 1% | (162,153,116) | 0.597 | 0.614 |
| 10% | (216,196,91) | 0.756 | 0.800 |
| 50%+ | (253,231,55) | 0.874 | 0.934 |

Monotonic in luminance both normally and under a deuteranope simulation, which is the only
thing a sequential ramp has to promise.

## Validation

Zonal means at res 6, checked against geography we can reason about:

| place | range | reading |
|---|---|---|
| Congo basin interior (DRC) | 0.80% - 19.9% | active frontier, Kisangani highest |
| Iowa, USA | 0.011% - 0.296% | cleared in the 1800s, so ~nothing 2002-2022 |

A ~70x split in the right direction and roughly the right magnitude. If the join were
smearing neighbours together or dropping the weighting, that contrast would collapse.

## Open

1. ~~Boundaries never coloured, fill checkbox dead.~~ **FIXED in session 2.** Neither guess
   below was the cause; it was swapping `get_fill_color` between a column and a constant. See
   "Open item 1" in session 2. The guesses are kept because being wrong about them is the
   useful part: the `filled=True` rule from CLAUDE.md was already followed and was never it.
2. ~~Not snappy.~~ **FIXED in session 2**, and the suspicion below was half right. The
   divisions memo did re-read on every pan (fixed by caching on coverage), but the bigger
   cause was the debounce running BEFORE the cache lookup. Cold Overture reads remain slow
   and that is now measured to the floor rather than suspected.
3. **Zero cells were just dropped** (`HAVING avg(v) > 0`) to kill the ocean hexagons and
   cut the render. Correct arithmetic (the filter is on the cell mean, not the pixel, so
   zero pixels still count in the average) but it means land with genuinely no
   deforestation now looks identical to ocean. The legend still shows a "none" swatch that
   can no longer occur. **Unverified since the change.**
4. **Rondonia returned zero counties** in a standalone harness on a wide box while the
   notebook found 3 on a narrow one. Never resolved. Could be a subtype/`is_land` quirk in
   Brazil, could be a harness bug. If Rondonia comes up bare when zoomed in, start here.
5. The seed polygon was degenerate (1e-6 square) and blew up deck's earcut, taking the
   whole update pass with it and producing a cascade of assertions naming innocent layers.
   Fixed to a 0.01-degree square. Worth remembering as another instance of the rule already
   in CLAUDE.md: an assertion naming a layer is weak evidence that the layer is at fault.

---

# Session 2: the render bugs fixed, and Overture measured properly

Open items 1 and 2 from the list above. Both were worked; one is closed, one turned out not
to be a bug at all.

## Open item 1, the dead fill: CLOSED, and the cause was not either candidate

The note above guessed `division_fill` defaulting to `False`, or a stale `HOLD["divtable"]`.
Neither. The real cause:

**`get_fill_color` was being swapped between a TABLE COLUMN and the constant `[0,0,0,0]`.**
That is a change of accessor KIND, not of data, and deck keeps whichever it saw first. The
`filled=True` rule from CLAUDE.md was already being followed correctly and was never the
problem, which is why following it harder did nothing.

Fix: `divisions_to_layer` now returns TWO tables, identical except for the alpha baked into
an RGBA colour column, and the toggle re-pushes the table. Both states are the same column
of the same schema, so deck never sees an accessor change. The seed polygon's colour column
was also widened from 3 to 4 so the first real push is not a change of accessor WIDTH either.
Fill now defaults ON, because the choropleth is the point of the join.

Verified by driving the widget, not by reading the code:

```
fill ON  rgba = [233, 210, 77, 165]
fill OFF rgba = [233, 210, 77,   0]
fill ON  rgba = [233, 210, 77, 165]
```

**Generalised rule, worth carrying to any lonboard layer:** never swap an accessor between a
column and a constant. Pick one and vary the data inside it.

## Open item 2, "not snappy": it was the debounce, and it was structural

`view_state` fires on every frame of a drag. Every one of those frames was handed to an async
task that slept `SETTLE` (0.25 s) BEFORE it would so much as look at the cache. So a pan
inside the box already on screen, and a zoom back to a resolution already folded, each cost a
quarter second of nothing, even though both are dict lookups.

Fix, taken from `bias-bounty-map-tutorial` (`set_status` / `on_camera` / no timers): answer
everything answerable from memory SYNCHRONOUSLY in the comm handler (`_instant`), and let the
debounce guard only a genuine object-store read. Plus an echo check so the map's own emitted
view is ignored, and a head/tail split on the status line so the zoom readout moves on frames
that read nothing.

Measured through the real traitlets observer:

| camera move | observer returns in | read scheduled |
|---|---|---|
| same view (echo) | 0.06 ms | no |
| tiny pan, inside the box | 0.05 ms | no |
| zoom back out to a cached res | 0.04 ms | yes (subtype changed) |
| jump somewhere new | 0.11 ms | yes |

**`HOLD["divbox"]` is load-bearing and was a real bug.** The divisions are fetched after the
cells, so a camera move landing between the two used to leave `HOLD["div"]` set while the
boundaries on screen belonged to the previous place; the instant path then matched and never
refetched them. The box the DIVISIONS cover has to be tracked apart from the box the CELLS
cover.

## Overture: measured to the floor, and there is no client-side fix

This is the important part of the session, and two of my own claims were wrong along the way.

### What does not help

- **Column projection.** `geometry` is **99.0%** of a row group's compressed bytes (22.38 MB
  of 22.6 MB). Everything else together is noise. `read_async` has no projection argument
  anyway.
- **Pruning on `subtype`.** Checked at the statistics level, not just the partitioning level.
  Row-group min/max pairs are `('county','region')`, `('country','region')`,
  `('locality','region')`, `('county','neighborhood')`. They span the alphabet, so asking for
  counties still reads country polygons. **Overture owns a level-of-detail hierarchy
  (country -> region -> county -> locality -> neighborhood) and does not lay the files out by
  it.** That single fact is most of why this is slow, and it would cost Overture nothing to
  fix.
- **More client concurrency.** Measured directly against raw ranged gets of the same bytes:

  | | time | throughput |
  |---|---|---|
  | `read_async` | 8.16 s | 23.3 MB/s |
  | raw ranged gets, concurrency 1 | 6.80 s | 27.9 MB/s |
  | raw ranged gets, concurrency 8 | 8.04 s | 23.6 MB/s |
  | raw ranged gets, concurrency 32 | 5.83 s | 32.6 MB/s |

  The client is already getting what the connection gets from that bucket. **I claimed a
  "3-5x on the table" from within-file concurrency. That was wrong.** The original note's
  claim, that no query makes this faster, was right.

### The floor, stated as arithmetic

One `division_area` file over a Brazil-sized box (-70,-20,-37,0): bbox pruning cuts **64 row
groups to 10**, an 84% cut, and those 10 still weigh **190 MB**, because getting 6,337
matching rows means touching 21,782 rows of geometry. 190 MB at ~28 MB/s is 7 seconds.

That is not a bug. Overture division_area costs what its geometry weighs, and the only lever
is touching less of it.

### What does help

- **Reading the 8 files concurrently.** The old loop awaited them one at a time for no
  reason. Same Brazil box: **34.7 s -> 13.8 s**. Notebook viewport at zoom 5.6: **24.5 s ->
  12.0 s**; at zoom 7.2: **13.6 s -> 5.3 s**.
- **Keeping the open `GeoParquetFile` handles.** `open_async` is a footer read, ~0.8 s per
  file, and it was being paid again on every fetch.
- Caching by COVERAGE rather than by exact bbox (the old memo re-read on a pan of a few
  pixels), plus an on-disk copy in `.cache/divisions/`. **Stephen's objection to the disk
  cache is correct and recorded here deliberately: it does not make the notebook fast, it
  makes it fast the second time.** The cold number is the honest one.

## `division_boundary` instead of `division_area`: Stephen's idea, tested

`division_boundary` is **0.51 GB in 1 file against 4.47 GB in 8**, and **1.8 KB/row against
11 KB/row**. Over the Brazil box it decodes to **7.0 MB against 187.6 MB**. If a division
polygon could be rebuilt from its boundary segments, `division_area` would be unnecessary.

**The join key is `division_id`, not `id`.** `division_boundary.division_ids` references
`type=division` entities; `division_area` carries a separate `division_id` column for exactly
this. Joining on `division_area.id` silently returns zero rows.

**The reassembly is lossless where it works.** `ST_BuildArea(ST_Node(ST_Union_Agg(g)))` over
a state's segments:

| state | true km2 | rebuilt km2 | ratio |
|---|---|---|---|
| Goiás | 230,870 | 230,870 | **1.000** |
| Tocantins | 188,814 | 188,814 | **1.000** |
| Distrito Federal | 4,050 | 4,050 | **1.000** |
| Pará, Amazonas, Bahia, Rondônia, ... | | 0 | open GEOMETRYCOLLECTION |

Exactly, not approximately. The mechanism is sound; the data is incomplete. Two causes:

1. **No coastline.** Overture publishes no coastal line as a division boundary, so every
   coastal division has a hole where the sea is. The 7 `is_land=false, is_territorial=true`
   rows in the box are 19-33 km stubs, not coastline.
2. **International borders are tagged at country level.** Rondônia got 3 segments and Roraima
   1, because the border with Bolivia or Venezuela has `subtype='country'` and its
   `division_ids` names Brazil and Bolivia, not the state.

Polygonizing the WHOLE noded network in the box (all subtypes together, not per division)
does not rescue it: 24 faces totalling 510,194 km² against Brazil's ~8.5M. Only the three
interior divisions above close.

**Verdict: `division_boundary` cannot replace `division_area` for the polyfill.** It works
for 3 of 23 Brazilian states, and the pattern (interior only) generalises.

**But it is the right source for the DRAWING half.** A border is a line, and you do not need
a closed polygon to draw one. 27x less data for the same box, no polygonization, and the
reassembly failure is irrelevant to it. That splits the two jobs currently fused into
`division_area`: lines for the render, polygons for the statistics.

**Untested idea, recorded because it may make the coastline problem disappear for the
STATISTICS:** the raster is NaN over ocean and `HAVING avg(v) > 0` already drops empty cells,
so a coastal division closed crudely on the seaward side would contribute nothing extra to
its own mean. The polygon only has to be correct on land. If that holds, `division_boundary`
plus any crude seaward closure is sufficient for zonal means, though not for a drawn outline.

## The drawn box: built, works, and has one bad bug

`selected_bounds` -> rank every division in the box by mean share deforested, rendered under
the map in an anywidget `Panel`. Reads one H3 resolution finer than the screen, sizes that
resolution from the BOX rather than the current zoom, and falls back county -> region ->
country (Overture has counties for only 171 of 219 countries).

Numbers are defensible:

| box | result |
|---|---|
| Rondônia | Rondônia 27.008%, Mato Grosso 21.571% |
| Iowa | Wapello 0.953%, Dallas 0.869%, Keokuk 0.754% |
| Congo basin | Mbandaka 16.701%, Ngabe 8.926%, Bongandanga 7.650% |

A ~30x split in the right direction, consistent with the res-6 validation above.

**BUG, NOT FIXED: the first box took 596 seconds.** `rank` bumps to res 8 for a small box and
then polyfills the ENTIRE geometry of Rondônia and Mato Grosso (~1.5M cells) when only a
2-degree window was drawn. The polyfill is never clipped to the box. Fix is either to clip
the division geometry with `ST_Intersection` before filling, or to cap the resolution by
division size. Clipping also changes the SEMANTICS (mean within the box vs mean of the whole
division) and that choice has not been made.

Also reproduced open item 4 from session 1: **that Brazil box returned zero counties** and
fell back to regions, while the Congo box got 46 counties fine. Still unexplained.

## Where this might go next: drop Overture, bring NLCD back

Stephen's idea at the end of the session, deliberately not acted on. Recording it because
the argument for it is strong and not obvious.

**Nearly every open problem above comes from the polygon side.** The `ST_Dump` workaround
for MultiPolygon, the `center` vs `overlap` decision, the "too small to measure" case, the
row-group pruning that collapses at world zoom, and the dead fill checkbox are all
Overture. A raster-to-raster join on the H3 cell id has none of them: both sides fold to
cells and the join is the SAME integer equi-join in DataFusion, minus the polyfill
entirely. Open items 1, 2 and 4 all disappear; only the zero-cell question (3) survives.

Two readings, and they are not the same notebook:

- **NLCD as a second layer joined to deforestation on the cell id.** "What land cover is
  being lost." No boundaries at all. The catch is coverage: the deforest layer is global
  and NLCD is CONUS, so the answer only exists for one country, and the USA is close to the
  least interesting place on earth for a 2002-2022 deforestation layer (see the Iowa
  numbers above: 0.011-0.296%). Worth checking whether the interesting CONUS signal is the
  Pacific Northwest and the southeast timber belt before committing.
- **NLCD as the raster, with this notebook's machinery underneath it.** Keeps the sparse
  tile reader, the log ramp, the zoom ladder and the DataFusion join; drops the global data.

**The piece most worth carrying over either way is the sparse-tile check**, and it is the
piece that looks most like boilerplate. Any COG that stores no ocean will otherwise fail
with `Invalid range requested, start: 0 end: 0`, which names neither the tile nor the
sparseness.

## Session 3 (2026-08-10): the GeoParquet path is gone, divisions come from PMTiles

The conclusion of "measured to the floor" above was that no query makes the GeoParquet
read faster because the layout is the problem. The fix was never client-side: Overture
publishes the same release as a PMTiles build, and that is the layout problem solved
upstream. `overturemaps-extras-us-west-2/tiles/2026-07-22.0/divisions.pmtiles`: one
19.5 GB object, anonymous ranged GETs, Hilbert-ordered gzipped MVT tiles, z0-12. (The
old `overturemaps-tiles-us-west-2-beta` bucket is dead: `AllAccessDisabled`.)

The reader is the PMTiles v3 client from `xsql-duckdb-terrain-h3.py`, ported nearly
verbatim; that notebook is parked on looks but its reader was the good part. The MVT
decode is hand-rolled on the same varint machinery (the one real bug on the way in:
exterior-ring winding. Tile y points down, so a spec-clockwise exterior is
counterclockwise in plain axes and the standard shoelace sum is already positive.
Getting the sign backwards classifies every ring as a hole and decodes every feature
to nothing). Verified ring-exact and property-exact against mapbox-vector-tile on ten
tiles including the world tile, Java's coastline and Italy's enclaves.

Measured, same Rondonia-sized viewport as the floor measurements:

    regions, cold          0.74 s   (GeoParquet: 13.8 s with all three fixes in)
    regions, warm          0 ms     (memo; the disk ledger is deleted, not ported)
    counties, Iowa view    0.36 s   (137 rows)
    polyfill res 6         0.09 s   (50,750 cells; Rondonia 6,617 vs ~6,600 expected
                                     from exact geometry, so the simplified tile
                                     geometry costs nothing at these resolutions)
    regions, whole planet  4.2 s    (3,912 rows; not reachable at all before)

Facts about the tileset, measured off the tiles (none of it is documented):

- Subtype minzooms are baked in by Planetiler: country z2, region z4, county z8,
  locality z10. Everything persists from its floor to z12. `SUB_MINZOOM` in the
  notebook records this and floors the zoom picker.
- `division_area` carries `division_id`, `subtype`, `@name` (plain primary name, no
  JSON parse), `country`, `is_land`. Nothing needed is missing, and `is_land` is
  always present (measured 431 True / 325 False over seven tiles): the maritime half
  IS in the tileset and still needs the filter.
- Join on `division_id`, not `id`: `id` names the area row, several per division.
  Same lesson as the `division_boundary` experiment.

Two structural notes:

- Tile geometry arrives clipped, so one division is several pieces. Pieces are
  dissolved per id in DuckDB (`ST_Union_Agg`) before anything downstream sees them,
  or the stroke draws tile-edge lines across the map. The tile buffer makes the union
  clean. Clip edges survive only at the outer boundary of the fetched range, a full
  DIV_PAD beyond the viewport.
- **The 596-second rank() bug is closed as a side effect.** The zonal join was always
  an inner join against in-view cells, so the whole-division polyfill never changed
  the NUMBERS, only the cost; tile-clipped geometry cuts the cost without touching
  the semantics. The remaining guard is TILE_CAP=256: a continent-sized box asking
  for counties is refused instantly and rank() falls back to regions, the same
  promise it already makes where counties do not exist.

Deleted along with the read path: the file-bbox index, the on-disk division cache and
ledger (existed because a cold read cost 18 s; it now costs under a second), and the
`geoarrow-rust-io` dependency. `.cache/divisions/` and
`.cache/overture-index-divisions.json` are orphans and can be deleted.

Integration test: `itest_divisions.py` (scratchpad, session-local) extracts the
divisions cell from the notebook by AST and drives fetch -> dissolve -> polyfill ->
cap-refusal against the live archive. Headless export also clean.

Session 2's open item 4 (the Brazil box fell back from counties to regions,
unexplained) is CLOSED: a county fetch over Rondonia returns exactly one county,
Itenez, which is in BOLIVIA, across the border. Brazil has no county-subtype
divisions in Overture at all; its municipalities are not mapped to that subtype. The
fallback was the notebook behaving correctly on data that is genuinely absent, which
is the same story as the 48 countries with no counties anywhere.

## The paint is the raster now (2026-08-14): RasterLayer, hexagons parked

Stephen's call: keep H3 for the divisions join, draw the COG itself. The hexagon layer
is commented out in the map cell (construction, put_cells body, its slot in the Map
list); the fold, the cache, the zonal join and the ranking are untouched, because the
cells are their input. The Controls checkbox that toggled the hexagons now toggles the
raster and is labelled "deforestation".

The layer is lonboard 0.16's RasterLayer: kernel-side fetch and render callbacks, tiles
served to deck as PNGs over the anywidget bridge, one TMS generated from the COG's own
pyramid (async_geotiff.tms.generate_tms, EPSG:4326; deck.gl-raster reprojects
client-side, so no Mollweide/mercator work in the kernel). Eleven levels, z0 the
coarsest overview, z10 the 100 m full res.

Built DIRECTLY with the private constructor arguments from_geotiff passes, not via
from_geotiff, for two reasons measured against the installed 0.16.0 source:

- from_geotiff's fetch calls `image.fetch_tile` blind, and 73.6% of full-res tiles are
  unstored ocean: same `Invalid range requested, start: 0 end: 0` crash the fold cell
  documents. The layer's fetch consults `ifd.tile_byte_counts` first (the fold's own
  fix); an absent tile returns None and the render callback passes the None through, no
  request issued.
- from_geotiff ships `min_zoom`/`max_zoom` commented out while its fetcher indexes
  `images[len - 1 - z]`, so overzoom wraps NEGATIVE onto a coarse overview: wrong data,
  silently. `max_zoom=10` pins it.

Render: `tile.array.as_masked()[0]` (`.array`, not `.data`), the shared `ramp` for RGB,
alpha 0 for NaN AND exact zero. Zero-transparent is a semantic decision, not a
convenience: the fold's `HAVING avg(v) > 0` already drops zero cells, so the map's
promise ("shows where deforestation IS") survives the paint swap; painting the 69.6%
zero majority with the legend's dark swatch would have covered the ocean too, because
the averaged overviews turn unstored ocean into stored 0.0 at coarse levels (measured:
the coarsest level is one tile, only 6.2% NaN). PNG encode is matplotlib's writer
(`matplotlib.image.imsave`), so no new imaging dependency; morecantile IS new
(`lonboard[geotiff]` in the header and root pyproject), pulled in by generate_tms.

The RasterLayer cell opens the COG a SECOND time (headers only). It cannot share the
fold cell's instance: fold depends on `refresh`, refresh on the map cell, the map cell
on this layer; sharing would be a dependency cycle. The pixel caches are therefore
separate too, which is fine, the browser's tile cache is the one doing the work for the
raster.

Status: headless export passes; NOT yet flown interactively. Watch for: the anywidget
bridge under marimo carrying the layer's custom tile messages (the repo has only proven
Unicode and Bool traits across that bridge, and this path uses on_msg dispatch), and
per-tile render latency in the status-line feel (one PNG encode per tile, ~250 ms
measured cold for a 195x391 tile including the S3 read).

## First raster flight (2026-08-14): the streaks are not a projection bug

Stephen's screenshot zoomed out over North America showed horizontal smears across land
and ocean and read as a CRS problem. It is tile geometry, not projection: the fetch used
`boundless=False`, which CLIPS edge tiles to the image bounds, and deck stretches
whatever PNG it receives across the full tile quad. At coarse levels nearly every tile
is an edge tile; the coarsest is one 195x391 image declared as a 512 px tile, so it
drew stretched ~2.6x vertically, and parents shown while children load did the same.
Fix: `boundless=True`. The padding arrives as 0.0 (measured on the coarsest tile: NaN
count identical clipped vs padded, so padding is stored zero, not mask), and the render
maps zero to alpha 0 anyway, so the padding is invisible. If zero ever stops being
transparent in this render, the padding must be masked explicitly. Fix passes headless,
not yet reflown.

## Boundary fill opacity control (2026-08-14)

A stepped slider (0.1-1.0 by 0.1, Stephen's spec) plus a free number box (any 0-1
float) in the Controls widget, both writing one Unicode trait (`fill_alpha`), because
Unicode is the trait type proven to cross marimo's bridge browser -> kernel (the Status
ruler's "WxH" string). Commit on `change`, not `input`; the 0.1 steps rate-limit the
re-pushes and Safari/Firefox fire `change` during drags regardless.

Kernel side: the alpha moved from the FILL_ALPHA constant (deleted) into
`HOLD["fill_alpha"]` (0-255, seed 165), so no cell re-run resets it.
`divisions_to_layer` reads it when building each new pair; the pair already on screen
is re-tinted by `_refill` in the map cell: whole-table `pa.table(tbl)` (arro3 exposes
the C stream at table level only, the terrain recolor lesson), rewrite the alpha plane
of `color`, rebuild via `set_column`. The geoarrow extension metadata on the geometry
column survives the round trip; verified against PolygonLayer's table validation
headless, including a second re-tint (idempotent, RGB untouched). The handler parses
inside a try: it runs in a comm handler where exceptions are silent.

## Open: re-running cells loses the boundary fill until reload (reported 2026-08-14)

Stephen reports edits/re-runs can leave the boundary fill gone until a restart. This is
the flood notebook's recorded failure mechanism: destroying a lonboard Map terminates
deck's MODULE-LEVEL earcut worker pool, after which every polygon layer on the page
fails to init until the browser reloads; hexagon and bitmap layers survive, which is
why it presents as "lost the fill" specifically. This notebook has ONE giant map cell
(widgets + layers + Map + all handlers + refresh), so ANY edit to a handler re-runs it
and rebuilds the Map. The fix on file is the flood notebook's cell split: a map cell
that never re-runs (imports/widget classes/seeds/HOLD only) and a wiring cell that
re-runs freely, un-observing old handlers via HOLD refs. Not yet applied here; it is a
structural refactor of the big cell, not a one-liner.

Update, same day: the split IS applied, per Stephen's go-ahead. Shape of it, matching
the flood notebook: the map cell holds widgets, seeds, the two always-alive layers
(divisions, labels) and the Map, plus HOLD["wh"]/HOLD["vs"] only; VIEW_W/VIEW_H and
HOME moved into it so a constants edit cannot reach it. The wiring cell cancels old
tasks, resets the screen-state HOLD keys, assigns `deck.layers = [raster, divisions,
labels]` (the raster is NOT a Map() argument, so raster rebuilds from ramp/constants
edits swap in through the surviving Map), defines every handler, and re-hooks with
unobserve-then-observe through HOLD["h_ctl"/"h_wh"/"h_cam"/"h_rank"], try/except
ValueError for the map-cell-rebuilt case. The fold cell's opening draw now targets
HOLD["vs"] or HOME, so a wiring re-run redraws where the user left the camera.
`Map.layers` trait reassignment verified headless on 0.16. Headless export passes;
the split has not been flown.

## Open: raster stretch at low zooms, on zoom out (second flight, 2026-08-14)

After the boundless fix the world paint reads correctly, but one smeared band survived
over the Gulf at Cuba's latitude, and Stephen reports the stretch happens ON ZOOM OUT,
at low zooms, and that he has seen this class of artifact before OUTSIDE this repo
(possibly NaN-related, his suggestion: "maybe not a number"). One candidate change is
in: an absent (sparse-skipped) tile now renders as a real 8x8 fully-transparent PNG
instead of None, because deck's TileLayer keeps stretched neighbours-in-zoom on screen
until replacement content actually arrives, and a None never arrives. UNPROVEN against
the zoom-out case; if the band persists, next suspects are how deck treats the custom
4326 TMS when underzoomed (`extent` is not set on the layer; from_pmtiles sets it,
from_geotiff does not) and NaN handling in the coarse overviews. lonboard 0.16 exposes
no refinement-strategy trait to turn the stretching off directly.
