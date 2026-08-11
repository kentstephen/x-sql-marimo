# GLWD: what is actually in the bucket

Global Lakes and Wetlands Database, in `cboettig/wetlands` on source.coop
(bucket `us-west-2.opendata.source.coop`, prefix `cboettig/wetlands/glwd/`).
Listed and read unsigned, nothing built yet.

## The assets

```
   0.241 GB  glwd/glwd-main-class-cog.tif      1 file
   0.393 GB  glwd/glwd-area-cog.tif            1 file
   0.324 GB  glwd/hex                         98 parquet
   1.694 GB  glwd/area-hex                   118 parquet
   2.443 GB  glwd/class-area-hex           2,327 parquet
   0.023 GB  glwd/hex-fractions                3 parquet
   0.731 GB  glwd/class-area-sparse           33 tif
             glwd/category_codes.csv
             glwd/stac-collection.json
```

The whole global wetland classification is one 241 MB COG. No VRT, no catalog,
no tile index.

## Schema, hex side

Verified on `h0=576495936675512319`: 485,077 rows, 1.1 MB.

```
Z    DOUBLE     h0   BIGINT
h5   UBIGINT    h6   UBIGINT    h7   UBIGINT    h8   UBIGINT
```

`Z` is the class code. `category_codes.csv` is the legend: 0 no data, 1-7 open
water (freshwater lake, saline lake, reservoir, large river, large estuarine
river, other permanent, small streams), 8-9 lacustrine, 10+ riverine by
flooding regime.

## Three things worth remembering

**Stops at res 8.** NWI goes to res 10. Res 8 is a ~460 m edge, so GLWD cannot
answer a septic setback question. Different tool, different question.

**h5/h6/h7 ship as columns**, so the zoom ladder is precomputed. A zoom step is
a different column, not a re-fold.

**Partitioning is on h0 only, which runs backwards for a zoom-driven map.**
Close in, one 4.6 MB file. Zoomed to the globe, all 98 files, 324 MB, to draw
cells you immediately coarsen. The coarse columns live inside the files you
have to read to get them. The NLCD path gets cheaper as you zoom out; this gets
more expensive.

That last point is the argument for using `glwd-main-class-cog.tif` instead.
A COG has real overviews, so zooming out reads less, which is the shape the
existing notebooks already handle.

## Legend palette

The class descriptions name their colours and run dark green / light green /
teal / pale teal across lacustrine and riverine. Remap onto luminance before
using. Open water 1-7 is blues and is fine as shipped.
