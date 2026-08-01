# xsql-s1m-surface.py: state and next steps

Handoff notes. Last updated 2026-08-01, end of the surface-layer session.

## Where things stand

On `main`, pushed to origin. `935850e` is the tip.

| commit | what |
|---|---|
| `78f384f` | new notebook: H3 fold rendered as one textured mesh instead of N prisms |
| `50382b8` | viz controls ported from `xsql-s1m-h3.py`, plus texture-space smoothing |
| `0853045` | coverage picker + ArcGIS/USGS basemap set, lifted from `7e2783d` |
| `dd7bfd0` | hillshade, relief smoothing, hexagon layer removed, controls consolidated |
| `935850e` | marimo formatting pass |

Branch `surface-array` holds the parked array-direct experiment; see
`docs/xsql-s1m-array-plan.md`.

```bash
uv run marimo edit xsql-s1m-surface.py --sandbox
```

## What the notebook is

Identical to `xsql-s1m-h3.py` through the H3 fold: same GeoPackage catalog, same obstore
streaming, same per-tile fitted lon/lat UDF, same `h3_latlng_to_cell` fold, same
`h3_grid_disk` ring join for flow. **H3 still bins the data. It no longer draws it.**

Then, instead of `H3HexagonLayer`:

- **geometry** is one regular triangle mesh, cost set by the `mesh_density` slider and
  nothing else. 200k cells and 2M cells draw identically.
- **styling** is one texture, every texel resolved through `coordinates_to_cells` to its
  cell and painted from the elevation + `flow_gain` composite.

Mt Washington at res 12: 295,840 cells, 1,050,625 mesh vertices at 7.0 m per quad, 2048²
texture. The hexagon layer would have been ~8.9M vertices of tessellated prism.

There is deliberately **no hexagon renderer** in this notebook. An earlier version had a
radio to switch back and it nearly took a machine down; that was a mistake and it is gone.

## The three things that made the mesh look bad, and what fixed them

1. **No lighting.** lonboard's `SurfaceLayer` ships exactly two mesh attributes, `POSITION`
   and `TEXCOORD_0`. There is **no `NORMAL`** in the JS bundle (verified in
   `lonboard/static/index.js`), so deck's lighting has nothing to work with and the surface
   renders effectively unlit. Extruded prisms looked smoother partly because their vertical
   walls catch light for free.
   Fix: hillshade computed in numpy and **baked into the texture**. Surface normal against a
   light vector, sun 315°/45°, ambient floor 0.35, multiplied in as pure luminance so hue
   never shifts and the ramp stays deuteranope-safe. Uses the same `elevation_scale` as the
   geometry. Adding real normals would mean patching lonboard's JS.
2. **Mesh coarser than the data.** Density 256 over a 7 km AOI is 28 m quads against 9.4 m
   hexes: one vertex per nine cells, flat triangles spanning the gaps. Slider now reaches
   2048, defaults to 1024, and the cell prints metres per quad.
3. **Piecewise-constant height field**, and no mesh density fixes this one. Every hexagon is
   a flat plateau with a vertical step to its neighbour, so dense sampling gives literal
   hexagonal stairs and coarse sampling gives arbitrary facets. The staircase is in the DATA.
   Fix: `relief_smooth` blurs the height field itself, and mesh z samples that blurred field
   rather than doing a per-vertex cell lookup, so shading and shape cannot drift apart.

`colour_smooth` is separate and blurs the shading VALUE before colouring, so the ramp softens
without flattening relief. Both blurs are NaN-aware normalised convolutions, or they bleed
zeros in from outside the AOI and draw a dark rind on every edge.

Defaults: hillshade 0.7, relief smooth 4, colour smooth 2, density 1024.

## Landmines worth not rediscovering

- **The parquet segfault.** Constructing any `SurfaceLayer` kills the kernel with no
  traceback. pyarrow 25.0.0 crashes in `ParquetWriter.__init__` on a 3-wide `FixedSizeList`
  arriving over the arro3 C Data Interface, and `positions` cannot be anything but 3-wide.
  2-wide is fine; the same shapes built natively in pyarrow are fine; the same shapes through
  `arro3.io.write_parquet` are fine. So it is the handoff. The PARQUET PATCH cell forces
  lonboard onto its own arro3 writer, which is code it already carries as a no-pyarrow
  fallback. **Do not remove that cell.**
- `SurfaceLayer` never populates `_bbox` (returns inf), so `Map` needs an explicit
  `view_state` or it opens on null island.
- `positions`, `tex_coords` and `triangles` must be swapped under
  `hold_trait_notifications()`. Mesh density changes all three, and unbatched the frontend
  briefly holds indices pointing past the end of the buffer.
- `apply_continuous_cmap` returns RGB for some palettes and RGBA for others. Pad to 4 wide.
- Basemaps: both hosts are ArcGIS MapServer, so `/tile/{z}/{y}/{x}`, **row before column**.
  XYZ order returns tiles from the wrong place rather than 404ing. `max_zoom` must travel
  with the URL: USGS stops at 16, Esri goes deeper.
- The guard is on the **kernel** (stream, fold, sevenfold ring join), not the renderer.
  Nothing caps what deck draws because the cell count never reaches it.

## Unverified, needs eyes

- **Is the colouring flipped north-south?** In plain terms: the texture might be applied
  upside down, so the colours from the north end of the AOI land on the south end and vice
  versa. The relief would still look right, only the colour would be wrong, which is why it
  is easy to miss. Image row 0 is built as SOUTH, on the assumption that mesh `tex_coord` v
  runs 0..1 south to north and WebGL samples v=0 at the first row. Never confirmed visually.
  If it is wrong, flip `_lat` in the texel-index cell. Stephen looked and it seemed fine, so
  this is low priority.
- Whether the current defaults actually read as terrain. Stephen's last look was before the
  hillshade and relief-smooth landed.
- Stephen changed the texture size while looking; the committed default may not be what he
  settled on.

## NEXT: joining vector data (buildings) on H3

Stephen's question, and the answer is yes. Worth saying plainly: **H3 as a join key between
heterogeneous datasets is the canonical use for it, and it is a better answer to "why H3 at
all" than flow isotropy was.** Earlier in the session the honest ranking of what H3 bought
this pipeline was thin, because a raster folded to hexes and sampled back to a lattice is a
round trip. A raster-derived table joined to a vector-derived table on a shared cell id is
not a round trip; it is the thing the index exists for.

The SQL shape is the payoff:

```sql
SELECT t.hex, t.elevation, t.flow, b.height, b.n_buildings
FROM h3_table t
LEFT JOIN building_cells b USING (hex)
```

One join in the same DataFusion context that already did the fold, between a streamed raster
and a streamed vector file.

### Getting footprints into cells

`h3ronpy.vector` has `geometry_to_cells` and `wkb_to_cells` (confirmed present in the
installed 0.22), which is the polyfill. Register the result as an arrow table and it joins
like anything else.

Source worth looking at first: **Overture Maps buildings**. It is GeoParquet on S3, carries
`height` and `num_floors`, and has bbox struct columns for predicate pushdown, so an AOI
filter is `WHERE bbox.xmin < E AND bbox.xmax > W AND ...` pushed into the scan. That fits the
repo's whole thesis (stream from object storage, no API, no tile server) better than
Microsoft Building Footprints or an OSM extract would.

### Three ways to render it, not yet chosen

1. **Into the texture.** Polyfill to cells, join, fold the building attribute into the
   shading composite. Buildings appear as coloured cells. Costs nothing to draw and needs no
   new layer. Loses footprint shape at coarse resolution.
2. **Into the height field.** Same join, but add building height to the mesh z. Buildings
   extrude the terrain itself. Same zero render cost. Reads as lumps rather than structures
   unless the resolution is fine.
3. **As a separate extruded polygon layer on top.** `SolidPolygonLayer` with
   `extruded=True`, `get_elevation` = building height, base = terrain height looked up
   through H3 at the footprint centroid. Keeps footprint geometry crisp, which polyfill
   destroys. **Prism cost is not a problem here**: the thing that hangs a machine is 300k
   hexagons, not 3k buildings. This uses H3 purely as a join key and not as geometry, which
   is arguably the best of both.

### The resolution tension, again

Res 12 hexes are 9.4 m across. A typical house is 10-15 m, so a building is one or two cells
and its shape is gone. Footprint shape needs res 13 (3.6 m) or res 14 (1.35 m), which is 7x
and 49x the cells. The mesh does not care, but the fold and the texture do, and the texture
caps out: at 2048 over a 7 km AOI a texel is 3.5 m, so res 13 is about one texel and res 14
is sub-texel and invisible.

Which points at option 3 for anything where the building should look like a building, and
options 1-2 for anything where buildings are an aggregate statistic (density, mean height,
built fraction per cell). That choice is Stephen's and has not been made.

## NEXT ORDER OF BUSINESS: drape NAIP over a wide swath

Not today. Stated goal: take NAIP imagery and drape it over the terrain surface across a
**wide swath**, wherever NAIP is available, rather than a small AOI.

Areas he wants to look at:

- the White Mountains (New Hampshire, i.e. the current Mt Washington neighbourhood)
- other mountainous country: New Mexico, Wyoming, Montana

Why this fits the surface layer specifically: draping imagery is exactly what a textured mesh
is FOR. The texture stops being a colour ramp over an H3 fold and becomes actual photography,
and the mesh underneath stays the same fixed cost regardless of swath size. This is the
use case the render path was heading toward without anyone saying so.

Things that will come up, noted now so they are not surprises:

- **Wide swath vs texture ceiling.** The texture caps out (2048 over 7 km is already 3.5 m
  per texel). A wide swath at NAIP's 0.6-1 m native resolution cannot fit in one texture, so
  this probably wants either several SurfaceLayers tiled side by side, or an accepted coarse
  ground sample, or the `RasterLayer` tile-callback path that was found and parked earlier in
  the session (see MEMORY: it takes arbitrary Python `_fetch_tile` / `_render_tile` and the
  frontend drives it from the viewport).
- NAIP lives on `prd-tnm` too, and there is prior art: `deck-terrain-naip-marimo/` and
  `3dep-seamless-duckdb-h3/naip_usgs_join_h3_1m.py` in the reference repos.
- NAIP and S1M do not share a grid or a vintage, so the two have to be co-registered. The
  existing per-tile fitted lon/lat UDF handles the projection side.

## GIVE UP: per-feature picking

Recorded as abandoned, not deferred. There is no click-a-thing-and-see-its-values in this
notebook and there is not going to be one.

- `SimpleMeshLayer` has no per-feature hit test, so the surface cannot be picked at all.
- lonboard has no selection model for `H3HexagonLayer` either, so nothing was lost by
  switching renderers.
- Earlier in the project two selection UIs were built and both were rejected as unusable
  (`mo.ui.table` multi-select over a tile list, and py-maplibregl click-to-toggle
  footprints). The conclusion then was that a linked list + map + cart is an APPLICATION,
  not a notebook, and Stephen did not want to build that.

Inverse lookup (screen point -> lon/lat -> `latLngToCell` -> the table) remains technically
possible if this is ever revived, but it is not on the list.
