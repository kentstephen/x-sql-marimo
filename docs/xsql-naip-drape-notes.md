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

## NEXT: DEM to heights, H3 to analysis

Decided at the end of this session, and it follows from the `xsql-s1m-surface-notes.md`
observation that a raster folded to hexes and sampled back to a lattice is a round trip.
The terracing on steep ground is that round trip, and it does not go away at any resolution
because the staircase is in the data.

So: **heights bilinear from the streamed DEM**, no fold in the geometry path, and H3 keeps
the job the index actually exists for. In this notebook that means the SQL aggregation and
what gets painted on the mesh; the join to vector data (Overture buildings, see the surface
notes) is still the strongest case for it and is still unbuilt.

First thing to paint that way: **NDVI**, from the NIR band `naip.py` currently discards.
`(NIR - R) / (NIR + R)`, folded per cell in SQL, on a colourblind-safe ramp (NOT the
conventional red-to-green, which is exactly the pair to avoid). That gives H3 something to
do that a photograph cannot, and gives the drape a reason to be a data product rather than
a picture.
