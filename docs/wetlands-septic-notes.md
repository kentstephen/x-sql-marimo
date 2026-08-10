# Wetlands and septic siting: the first pairing with a real question behind it

Nothing built yet. The diagnosis of why the land cover notebooks stalled is in
`flood-exposure-notes.md` and is not repeated here; the short version is that NLCD arrives
pre-thresholded, so the dissolve redraws a decision made before the notebook loaded, and
no second dataset has yet come out of a join as a number.

This one is different on both counts.

## The question

You cannot put a septic system in a wetland, and you cannot put one within a setback
distance of a wetland either. The setback varies by state and is usually somewhere in the
50 to 100 ft range. That is an actual constraint somebody is paid to check, on a specific
parcel, before money moves.

So the output is a count and an area, not a picture: how much of this ground is
disqualified, and which buildings or parcels sit inside the disqualified zone.

## Use `cboettig/wetlands`, not `giswqs/nwi`

Both mirror the FWS National Wetlands Inventory. They are not close.

`cboettig/wetlands` (updated June 2026) publishes NWI as **two assets that share a key**,
and that pairing is the whole architecture:

| | `nwi-v2/hex/` | `nwi-v2.parquet` |
|---|---|---|
| what | H3 cell index, no geometry | full geometry, 76 GB |
| layout | Hive, `h0=<res0 cell>/data_0.parquet` | single file, 379 row groups |
| cells | `h8`, `h9`, `h10` as UBIGINT | none |
| CRS | n/a | OGC:CRS84 (lon/lat) |
| pruning | partition on the res-0 cell | `state_code` only |

Both carry `_cng_fid`, `ATTRIBUTE`, `WETLAND_TYPE`, `ACRES`, `Shape_Length`, `Shape_Area`,
`state_code`. **`_cng_fid` is the join**, and it is what makes the two files one system.

Also in the repo: Ramsar (global, GeoParquet + PMTiles + hex), GLWD (global COG + hex), and
`nwi-v2.pmtiles` at 13 GB.

### The hex side

One row per feature per res-10 cell, partitioned by res-0 cell. Verified on the Alaska
partition `h0=576988517884755967`: 8.81 MB, 1,205,233 rows.

- **The bbox problem is gone.** An AOI maps to one or two `h0` partitions and Hive pruning
  does the rest. No index to build, no rewrite, no `.cache/`.
- **The polyfill is done**, which was the main compute in the original plan.
- **`h10` is UBIGINT**, the same type `h3_latlng_to_cell` returns, so Overture building
  centroids join to it with no conversion at all.
- **No geometry ships**, so the boundary has to be dissolved from cells. That is exactly
  what `WASH_SQL` already does, and it means the outline on screen is this repo's product
  rather than a re-render of the FWS polygons.

### The geometry side

Verified: 38,065,251 features, 379 row groups averaging ~100k rows and ~200 MB. GeoParquet
**1.0.0**, WKB, `covering: None`. There is a `bbox` in the `geo` metadata but it is the
file-level extent, one box for the whole file, **not** a per-row bbox column, so there is
no spatial pruning.

What does prune: `state_code` statistics are present and **331 of 379 row groups hold a
single state**. Unverified but likely worth checking: whether `_cng_fid` is ordered enough
for its statistics to prune within a state.

## The two-stage pipeline, and why res 10 stops being a problem

H3 res 10 is an edge of about 66 m and an area of about 15,000 m². A 100 ft setback is
30 m, well under half a cell, so **`grid_disk` cannot express the setback**: one ring is
roughly 430 ft. `cell_to_children` down to res 12 would give res-12 precision on a res-10
boundary, which is false precision rather than precision. An earlier draft of this note
proposed exactly that; it does not work.

The two assets solve it instead:

1. **Screen on the hex index.** AOI to `h0` partition to `h10` cells. Counts, area, the
   dissolve, the Overture join, the live map. Cheap, no geometry moved. Answers "is this
   parcel clear, nowhere near clear, or does it need a closer look".
2. **Measure on the geometry.** Take the handful of `_cng_fid` values that actually matter
   and pull only those from the 76 GB file with `state_code = 'XX' AND _cng_fid IN (...)`.
   State prunes the row groups, the fid list prunes the rows. Exact distance from the real
   polygon, at whatever setback the state requires.

Screening and delineation-grade measurement in one pipeline, and the expensive asset is
only ever touched for a few dozen features.

## Why the threshold is genuinely the user's

**`ATTRIBUTE` is the Cowardin code, not a label.** `PFO1A` reads as Palustrine / Forested /
Broad-leaved deciduous / Temporarily flooded, and the trailing letter is the **water
regime**, which is the part that bears on whether ground will drain a leach field. In the
giswqs DC file there were 114 distinct `ATTRIBUTE` values against 5 distinct
`WETLAND_TYPE` values, so the plain-English column throws almost all of it away.

That is what NLCD could not do: the full code is handed over raw, and the line between
"wet enough to disqualify" and "fine" is drawn by the jurisdiction, differently in
different states. The threshold is contested and it belongs to the user.

## Join mechanics

**Do not polyfill the buildings.** At res 10 a hexagon is ~66 m across and a house is 10 to
15 m, so a polyfill returns zero cells for most footprints and the count silently
undercounts. Centroid through the existing `h3_latlng_to_cell` UDF at res 10 and join
straight to `h10`. Coarser than ideal for the building side, but it is the resolution the
wetland index is published at, and stage 2 is where precision comes from.

Overture gotchas from `CLAUDE.md` all apply: buildings needs `load_parts` or `load_wkb`
because the geometry column is Polygon in some files and MultiPolygon in others, and
`file_index()` gets an AOI read to ~1.4 s once footers are cached. Height is on 55-75% of
footprints; count and footprint area are honest, volume needs a caveat.

`WASH_SQL` still has to stop dropping its cell ids. It runs `ST_Dump(ST_GeomFromWKB(mp))`
and throws the cells away, which is why the imagery notebook derives its cell count from
area instead of counting.

## Honesty requirement, on the map itself

NWI is a photo-interpreted inventory and FWS states explicitly that it is **not a
regulatory wetland delineation**. Boundaries are drawn from imagery at a mapping scale, not
walked, and the absence of a wetland on NWI is not evidence there is none.

Put that on the map rather than in a footnote. It makes the tool more useful, not less: it
says where a delineation is needed, it does not replace one. That is the same claim the
imagery notebook makes about its boundary, and it is the reason this repo's maps are worth
looking at. It also happens to be the honest description of the two-stage pipeline above.

## The other mirror, for the record

`giswqs/nwi` (November 2023): 51 files, 81.4 GB, one Parquet per state, MN largest at
6.2 GB and DC smallest at 2.6 MB. Schema without any cell columns. GeoParquet
`1.0.0-beta.1`, WKB, no `covering`, which is a date rather than an oversight since bbox
covering only arrived in GeoParquet 1.1 the year after. Useful properties it has and
`cboettig` does not: **CRS is EPSG:5070**, the same Albers as the NLCD CU COGs, and there
are sibling `riparian/` and `historic_wetlands/` prefixes.

## Follow-ons, once the base exists

- `historic_wetlands/` on the giswqs mirror is the honest version of the change-over-time
  idea Annual NLCD could not support, because the comparison is between two deliberate
  inventories rather than two runs of a classifier that disagrees with itself.
- `riparian/` is a second overlay with its own regulatory weight in western states.
- Ramsar and GLWD in the `cboettig` repo are the same pipeline run globally, if the US
  framing ever feels too narrow.
- Parcels instead of buildings would be the real product, but parcel data is
  county-by-county and mostly not open. Buildings are the available proxy.
