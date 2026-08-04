# xsql-naip-drape.py: session notes

Companion to `docs/xsql-s1m-surface-notes.md`. That notebook proved the textured mesh; this
one points it at NAIP and finds out what breaks when the texture is a photograph.

## The bug that mattered, and the four wrong answers before it

The drape came back covered in **pale translucent quadrilaterals**, worst on slopes facing
the camera, in a herringbone at close range. Ruled out in this order, each with evidence:

1. **H3 resolution.** Res 11 to 12 changed nothing. The artifact is not the fold.
2. **NAIP quad collars.** Real and worth fixing (below), but the diff between the old and
   new mosaic rule is a clean grid of 400 m strips at quad boundaries, and the artifact
   was neither that shape nor that size.
3. **Texture content.** Rebuilt the texture headlessly at the exact AOI scale, then read
   the textures the running notebook itself had produced. Clean at 1:1. Whatever it was,
   it was not in the image being uploaded.
4. **Depth and mipmaps.** Both were genuinely misconfigured (below) and neither changed
   the picture.

**It was deck lighting the photograph.** `SimpleMeshLayer` sets
`flatShading: !this.state.hasNormals`, and lonboard's `SurfaceLayer` sends only `POSITION`
and `TEXCOORD_0`. A mesh with no normals therefore does not render unlit, which is what
`xsql-s1m-surface-notes.md` claimed and what the whole hillshade design assumed. It renders
**flat shaded**: one derived normal per triangle, lit by the default material. One
brightness per triangle, disagreeing across each quad's diagonal, over a photograph that
already contains the real sun.

Fix: `tools/patch_lonboard_surface.py`, which injects `material: false` (and
`textureParameters: {maxAnisotropy: 16}`) into the props lonboard passes to
SimpleMeshLayer. It edits the shipped JS bundle, so **it must be re-run after any install**
and the browser must be hard-reloaded, not just the kernel. Both belong upstream as traits
on `SurfaceLayer`; when they land, delete the script.

The lesson worth carrying: an artifact that survives every parameter in the notebook is
probably not in the notebook. Dump the texture to a PNG early. It splits the search space
in half and it is two lines.

## Depth parameters were in the wrong dialect

`parameters={"depthTest": True, "blend": True}` is the WebGL-1 spelling. deck 9 hands a
layer's `parameters` to luma's render pipeline, whose depth keys are `depthCompare` and
`depthWriteEnabled`. `depthTest` is not rejected, it is simply not read, so the pipeline
fell back to `depthCompare: "always"` (which luma maps to `gl.disable(DEPTH_TEST)`) and
`depthWriteEnabled: false`: no depth buffer at all, scene resolved by submission order.

A heightfield in painter's order is right about half the time, which is why this never
looked like an outright bug. Triangles go out south row to north row, so a camera looking
north draws the far ridges last and paints them over the near ones. Turn to look south and
the same code silently draws the correct picture.

Fixed here; **still wrong in `xsql-s1m-surface.py`, `xsql-s1m-h3.py`, `xsql-dem-1m.py`,
`xsql-dem-h3.py` and `xsql-dem-rem.py`**, which all carry the same two-key dict.

## NAIP selection, which was quietly picking bad mosaics

Three separate defects, all of which showed up as "a big box returns almost no imagery":

1. **The single-date preference had no coverage floor.** Preferring a year whose quads
   share one capture date is right when every year covers the whole AOI, and wrong the
   moment a box spans two states. It also *selected for* the failure: the fewer quads a
   year has, the likelier they share a date. A 1-degree box took an 11% flight over a 100%
   one. A single-date year now has to be within `COV_TOL` of the best coverage available.
2. **Grouping by `datetime.year` split campaigns.** A NAIP campaign is a state contract
   that can cross New Year: Vermont's 2018 campaign flew September to October and finished
   on 2019-01-14. Grouped by calendar year, one 100% mosaic became a 56% "2019" and a 46%
   "2018". Group on `naip:year`.
3. **`max_items` truncated mid-campaign**, which is a hole in the drape and not just a bad
   number, because the quads past the cap are never streamed. Cap raised, top candidates
   re-searched in full when it is hit, and what re-searching cannot fix (campaigns cut off
   entirely) is reported.

Measured over the Presidentials, boxes from 0.3 to 1.5 degrees now all return 100%
coverage. Before: 0.6 gave 50%, 1.0 gave 11%.

## The mosaic rule: deepest quad wins

NAIP quarter-quads are not edge-matched. Each carries a **400-500 m collar** its neighbour
also covers, and a campaign is flown over weeks, so the two copies of that ground are
different days with different sun, haze and snow. Filling first-quad-wins painted every
collar as a rectangle of one flight day surrounded by another.

Each texel now takes the quad it is furthest *inside*, scored on normalised distance to
that quad's own edges, so seams land on the midlines between quad centres. Verified on a
Wasatch box: 7.7% of texels changed hands and the map of which ones is exactly the collar
grid.

## Leaf-off

`NAIP season` (Any / Prefer leaf-off / Leaf-off only), in the top row with the DEM and the
H3 resolution because it decides what gets fetched. Leaf-on imagery drapes the canopy, so
the ground the DEM describes is hidden under it.

The leaf-on window is a latitude ramp in `naip.leaf_on_window`, ~Apr 15 to ~Nov 1 at
latitude 35, pulling in about nine days per degree north. A campaign can be half leaf-off,
so the candidate is the leaf-off **subset** of a campaign, which is what finds Vermont's
January quads and the Presidentials' late October.

**Winter NAIP on Planetary Computer, by volume:** Florida 3,235 items, Texas 810, New
Mexico 606, Georgia 404, Louisiana 264, then a thin northern tail of Vermont 144 and New
Hampshire 38, all January 2019. The south is clear-sky convenience; the northern tail is a
state finishing a campaign in deep winter, which is what a mountain box wants and also
what is likely under snow with a 20-degree sun. A leaf-off pick above 38N says so.

## Timeouts

obstore gives an entire HTTP request **30 seconds by default**, connect through last byte.
One tile of one quad at a fine overview is tens of megabytes and a wide box fires dozens at
once, so they share the link and each runs past 30 s while making steady progress. Now 3
minutes overall, a short connect timeout, a per-chunk read timeout, and **concurrency
capped at 8**: forty simultaneous reads are not forty times faster, they are each forty
times slower against a clock that knows nothing about the other thirty-nine.

pystac_client ships **no timeout at all**, so a stalled search hung the cell forever. Now
`(10, 90)`, three attempts with backoff, and a 1000-item page size (a 3,400-item query was
35 round trips, now 4).

## The coverage gate

The STAC search now sits under the AOI and above the COG reads. Below 50% NAIP coverage the
pipeline stops with a message instead of streaming a DEM, folding H3 and building a mesh to
discover there is nothing to paint on it. The cheap question goes first. Cost: the scene
keeps the last box that got through, so the message says plainly that nothing was fetched.

## The ceiling, restated with numbers

A 121 km box at 2x2 tiles of 2048 is 29.5 m per texel against NAIP's 0.6 m: a 2048² texture
being asked to hold roughly 200,000² pixels. Every remaining artifact at that width is that
ratio. The drape is a fixed-AOI technique with everything resident and nothing
view-dependent, and it is good at 10-25 km and structurally incapable above ~40 km. Wide
swaths need view-driven tiles, which is the `RasterLayer`/`TileLayer` note in the notebook
docstring.

## DONE, as `xsql-naip-ndvi.py`: DEM to heights, H3 to analysis

Decided at the end of this session, and it follows from the `xsql-s1m-surface-notes.md`
observation that a raster folded to hexes and sampled back to a lattice is a round trip.
The terracing on steep ground is that round trip, and it does not go away at any resolution
because the staircase is in the data.

So: **heights bilinear from the streamed DEM**, no fold in the geometry path, and H3 keeps
the job the index actually exists for. In this notebook that means the SQL aggregation and
what gets painted on the mesh; the join to vector data (Overture buildings, see the surface
notes) is still the strongest case for it and is still unbuilt.

First thing to paint that way: **NDVI**, from the NIR band `naip.py` used to discard.
`(NIR - R) / (NIR + R)`, folded per cell in SQL, on a colourblind-safe ramp (NOT the
conventional red-to-green, which is exactly the pair to avoid). That gives H3 something to
do that a photograph cannot, and gives the drape a reason to be a data product rather than
a picture.

Built, and the shape it took:

- **Heights are bilinear off the streamed COGs**, straight onto the texel lattice. No fold
  in the geometry path, so no plateaus, so `smooth` defaults to **0** rather than to the
  drape's 3. Nearest would have aliased instead, because a shaded surface is read through
  its derivative and nearest quantises the derivative into flats and cliffs.
- **The read resolution now serves two masters**, and takes the finer: roughly one DEM
  sample per texel for the height field, and enough pixel centres per hexagon for `avg()`
  to mean something. In the drape only the second existed, because the lattice was fed by
  the fold rather than by the raster.
- **The query is the notebook.** Two rasters, two agencies, two resolutions, two CRSs,
  aggregated to the same key and `LEFT JOIN`ed on it in one DataFusion statement. The DEM
  side goes in through xarray-sql as its native grid, one relation per COG; the NAIP side
  goes in as the lattice it was sampled onto. Nothing is reprojected to a common grid,
  which is the step this normally costs.
- **`relief`** (max − min elevation inside a cell) comes free from the same GROUP BY, needs
  no ring join, and says something the height field structurally cannot: it is a statistic
  about the pixels rather than a property of the surface they were folded into.
- **NAIP is read once, in the shape the surface needs.** NDVI reads 4 bands onto the global
  lattice, because it is about to be averaged into hexagons and does not want tile
  resolution. The photograph reads 3 bands per tile at tile resolution, because it does.
- **`naip.py` grew a `bands` argument**, defaulting to 3. Band order is R, G, B, NIR;
  verified against ground truth, with Big Cottonwood forest at NDVI 0.39 median and 0.63
  p90 and Utah Lake negative.
- **The hexagons are visible, and that is now the point.** The shape is smooth and the
  cells are the data, so seeing them is seeing the resolution of the analysis.
- **The notebook checks the lonboard patch itself** and prints a loud warning if the bundle
  has been reverted, because that failure is silent and looks like six other bugs.

Still unbuilt, and still the strongest case for the index: the **vector** join. Overture
buildings on S3 as GeoParquet, polyfilled with `h3ronpy.vector.geometry_to_cells`, is the
same `LEFT JOIN` with a different right-hand side.

## Verdict on `xsql-naip-ndvi.py`, after a false alarm

It works. Flagstaff and Mount Elden at ~15 km came back as a photograph on smooth terrain:
no hexagonal terracing at any H3 resolution, no facets, imagery sharp.

**The interim "this did not work" was box width, not method.** The mesh cell prints metres
per quad, and it read **98.7 m/quad**, which is 1024 quads across a **101 km** box: the
swath left over from the drape session. Those were mesh triangles, exactly as large as they
looked. The texture was 24.7 m per texel at the same width, which was the pixelation. Both
scale linearly with the box and nothing else in the notebook does, so the DEM-heights
change was never implicated.

The lesson is procedural rather than technical. Four theories went out before that number
was asked for, and the number was already printed in the notebook's own output the whole
time. **Read the diagnostics the notebook prints before forming a theory about the render.**
Same lesson as dumping the texture to a PNG, one layer up.

### Settings for a ~100 km swath, if the box has to stay wide

| Control | Set to | Result at 101 km |
|---|---|---|
| Mesh density | 2048 (max) | 49 m quads |
| Drape tiles | 4x4 | 8192 texels across the AOI |
| Texture / tile | 2048 | 12.3 m per texel, 269 MB of texture |
| Elevation scale | 1.0-1.2 | exaggeration amplifies facets |
| Height smooth | 1-2 | softens crests so 49 m quads read less sharply |
| H3 resolution | 10 | 150 m cells, well matched to a 12 m texel |

`Texture / tile` 4096 with 4x4 is 1.07 GB and will not fit. 3x3 at 4096 is 8.2 m per texel
at 604 MB and is the better trade if the GPU is generous.

At swath width **the data surfaces hold up and the photograph does not**: a res-10 cell is
150 m and a 12 m texel resolves it twelve times over, while NAIP at 0.6 m is starved by a
factor of twenty. If a 100 km box exists to show a pattern across a whole range, NDVI and
Relief are the surfaces that will show it. The photograph wants 15-25 km.

## Session: S1M restored, the fold scoped, and a seam that was mine

### The DEM source was never Stephen's call to lose

The drape hard-wired 10 m and argued the case in its own docstring as though it were
settled. It was not: nobody asked for S1M to be dropped, and swapping a notebook's data
source is not a routine judgment call. `xsql-naip-ndvi.py` now carries a **DEM source**
dropdown, `10 m seamless (nationwide)` default and `1 m S1M lidar (partial coverage)`
alongside it. Three things follow from S1M and each is marked `S1M ONLY` in the code:

* **No VRT.** The only national catalog is `S1M_Products.gpkg`, ~15 MB, read with duckdb
  spatial. The `current` layer is one row per tile (11,749 rows, 11,749 distinct tiles),
  so the `ROW_NUMBER() OVER (PARTITION BY cell_name ...)` version dedupe in
  `3dep-seamless-duckdb-h3/s1m_viewer.py` is not needed against this vintage.
* **The coverage carpet** rides in the picker's layer stack from the start with
  `visible=False`, flipped by the DEM source in the same live-trait-swap cell as the
  basemap. The Map is still built once, so a drawn box survives the toggle.
* **Albers.** Tile selection happens in EPSG:6350 where footprints really are boxes; the
  lattice is projected into metres for the height sample; and the fold reaches degrees
  through the per-tile order-3 polynomial, because pyproj from a DataFusion worker thread
  aborts the process. Measured fit error on real tiles: 0.0000 to 0.0003 mm against a 1 mm
  tolerance.

Guards: no tiles, coverage under 25%, and more than 64 tiles. The last is a statement
about box size, since S1M is a 10 km grid.

### duckdb over stdlib sqlite3, and why

A GeoPackage is SQLite, so `sqlite3` reads it with zero new dependencies, which is what
`xsql-s1m-h3.py` and `xsql-s1m-surface.py` do (hand-unpacking the GPB envelope with
`struct`, reprojecting with pyproj outside SQL). Stephen asked for duckdb, correctly: the
reference repo is duckdb throughout, `ST_Read` removes the hand-rolled blob parsing, and
`ST_Transform` removes the out-of-band reprojection. Cost is `INSTALL spatial` fetching a
binary extension on first run. pyproj stays regardless, because the polynomial fit lives
inside the UDF where `ST_Transform` cannot reach.

### What H3 is actually for here, established by being asked four times

Stephen pushed until the answer was honest, and the honest answer is narrow:

| Surface | Needs the fold? |
|---|---|
| `NAIP RGB` | No. Never touched it. |
| `Elevation` | No. The bilinear height field is strictly better than hexagon means. |
| `NDVI` | Yes, to average the NIR band per cell and make the join demonstrable. |
| `Relief` | **Yes, genuinely.** `max - min` inside a cell is a statistic over a
neighbourhood of pixels, so no resampling at any resolution produces it. |

So the fold is now gated to NDVI and Relief and prints `fold skipped: [<surface>] reads no
cell values` otherwise. Three ties to the RGB path were cut:

1. The texture alpha started from `texel_ok` ("this texel landed on a cell the fold saw"),
   which made the PHOTOGRAPH depend on the fold. It now starts from `surface_ok`, i.e.
   whether the mesh has a height there, which is what the geometry is built from.
2. The elevation ramp (and the fallback under the photograph) read `cell_field("elevation")`.
   It now reads `height_raw`. Visibly better: the Cottonwood range widened from
   `1.49e3 .. 3.47e3` to `1.48e3 .. 3.5e3`, because averaging into hexagons was clipping
   the summit and the valley floor.
3. `coordinates_to_cells` over every texel is skipped along with the fold.

**The speed claim was overstated and is worth recording as such.** Skipping the fold saves
about a second out of twenty-three on a Cottonwood-sized box (medians over three runs each:
24.1s with, 23.1s without). It was never the bottleneck. The value of the change is
structural and the elevation ramp getting better, not throughput. Measure before agreeing
that something is slow.

### The seam was in the sampler, not in a product called Seamless

Reported as "tile gaps": a dashed dark cross over an Asheville S1M drape, with
`98.7% covered` on the height field and `98.7% opaque` on the texture.

`_bilinear` computed `fx` in pixel-centre space, so it runs -0.5 at the left edge of column
0 to `w-0.5` at the right edge of column `w-1`, then tested `0 <= i0 < w-1`. That discards
the outer HALF PIXEL on all four sides, because there is no second pixel to interpolate
against out there. Harmless mid-mosaic, fatal at a seam: two adjacent COGs each discard
their own half pixel at the shared boundary, so the union carries a ONE PIXEL NaN crack
along every tile edge. No height means transparent texture, hence the dashes.

The fix clamps the index pair to the last valid pair but leaves `tx`/`ty` UNCLIPPED, so
they run to -0.5 and 1.5 and the same expression becomes a linear extrapolation off the
edge pixels. Verified on synthetic edge-matched tiles whose truth is a plane:

| sampler | NaN | max error vs truth |
|---|---|---|
| old | 10.0% | 3e-14 (where it answers at all) |
| clipped `tx` to [0,1] | 0% | **5.0 m** (half-pixel flat step at every seam) |
| unclipped (shipped) | 0% | 3e-14 |

Interior samples are bit-identical to the old code, because out there the clamps never bind.

On Stephen's exact AOI (63 S1M tiles, Asheville) the height field went **98.7% -> 99.0%**,
and the texture with it. The recovered 0.3% is the ~11 internal tile seams at roughly one
texel each.

**Why the dashes were dashed, which is the diagnostic worth keeping:** the crack is one
read pixel (16 m) wide against 31.8 m texel spacing, so each seam line only catches a texel
some of the time. A sub-texel crack sampled by a coarser lattice renders as a dashed line,
never a solid one. If a seam artifact is dashed, suspect a sub-texel gap.

The remaining **1.0% is real**: `S1M coverage of this box: 99%`. The height field now agrees
with the coverage carpet to the decimal. No sampler change fills ground USGS never flew.

### Two false starts on that diagnosis

* First theory was NAIP mosaic holes. Wrong: `100.0% painted` ruled it out immediately, and
  the number was already in the output.
* Second attempt to reproduce used the 10 m product with 4 COGs and came back 100% opaque
  with BOTH samplers, because at that read resolution the crack was narrower than a texel
  and missed the lattice entirely. **Reproduce on the user's actual source and box, not on
  a convenient one.** The AOI print line carries the bbox and the source; ask for it first.

### S1M at swath width is not S1M

63 COG reads, 18.8M pixels, and the notebook reads the **16 m overview** of a 1 m product,
which is the 10 m product with coverage gaps and a much larger bill. Read resolution is set
by the lattice (AOI / texture), so native 1 m needs a box of roughly `texture * 2 m`:
~4 km at 2048, ~8 km at 4096. The tile-selection cell now prints this as a NOTE with the
box size and the threshold whenever it reads coarser than 2 m, rather than leaving it to be
discovered.

### Still unbuilt, and now the obvious next thing

The 10 m seamless product is nationwide and covers exactly the ground S1M is missing.
Filling S1M gaps from the 10 m DEM is a small change to the stream cell, since both already
flow through one reader, and it would make S1M usable on boxes that clip a coverage edge
instead of gating them at 25%.
