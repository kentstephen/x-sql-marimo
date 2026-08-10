# Deforestation x Overture divisions: what was built and what was learned

`xsql-deforest-divisions.py`. Vizzuality's global 100 m deforestation COG folded to H3 in
DataFusion, joined onto Overture division boundaries on the H3 cell id.

Status: **prototype, runs, numbers validated, render not finished.** See "Open" at the end.

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

1. **Boundaries are never coloured by their value, and the "boundary fill" checkbox does
   nothing.** This is the biggest gap against what was asked for: the choropleth, which was
   the whole point of joining to Overture, does not render. Confirmed by use, not just by
   reading the code.

   Two candidate causes and they are not exclusive:
   - `division_fill` defaults to `False`, so the layer is built with
     `get_fill_color=[0,0,0,0]` and only ever gets colour if the box is ticked. It should
     default ON.
   - Ticking it still does nothing, which is the more serious half and matches the bug
     already recorded in CLAUDE.md almost exactly. The layer IS permanently `filled=True`
     with only `get_fill_color` swapping, which is the documented fix, so that rule is
     already followed and is not the explanation. Suspect instead that `_on_controls`
     assigns `divisions.get_fill_color = tbl["color"]` where `tbl` is `HOLD["divtable"]`,
     and either that is stale/None, or deck needs the TABLE re-pushed for the fill sublayer
     to pick up a new accessor (`put_divisions` does re-push, `_on_controls` does not).
     Try making `_on_controls` call `put_divisions(HOLD["divtable"])` rather than assigning
     the accessor alone.

   Do NOT "fix" this by flipping `filled` itself. That is the trap CLAUDE.md warns about.
2. **Not snappy; data does not load half the time.** Not diagnosed. Suspect the divisions
   fetch on every camera settle (memoised only by exact rounded bbox, so a pan re-reads),
   and the res-6/7/8 bands reading L3/L2/L1 over a padded box.
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
