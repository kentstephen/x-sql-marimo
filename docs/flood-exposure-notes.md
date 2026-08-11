# Flood exposure: the next notebook, and why the land cover ones stalled

Nothing built yet. This is a direction, written down because the diagnosis took a
conversation to reach and would otherwise be re-derived.

The diagnosis below is shared with `wetlands-septic-notes.md`, which came out of the same
conversation and is further along: NWI plus Overture buildings answers a question somebody
is actually paid to check, and is the one to build first.

## Why NLCD kept feeling flat

The stack is fine. The dataset was the problem, and in a specific way:

**NLCD arrives pre-thresholded.** Somebody at USGS already decided this pixel is forest,
so the dissolve redraws a boundary that was fixed before the notebook loaded it. There is
nothing for the user to be doing. The stack's distinguishing move is that you pick a cut
on a continuous field and objects appear *because of your choice*, live, at the resolution
you are looking at. A categorical raster gives that away for free, which is why every
version reads as a rendering demo rather than a tool.

DEM has the same problem inverted: the threshold is the user's, but nobody argues about
"land above 300 m", so moving it means nothing.

**The second dataset has so far only supplied appearance.** The terrain notebook joins
NLCD to Mapterhorn on the cell id and the join is correct, but terrain became extrusion
height. NAIP became texture. Nothing has yet come out of a join as a *number*. That, not
the join itself, is what has been missing: dissolve a contour, aggregate something else
across the cells inside it, report a quantity.

So the target is: continuous field, threshold genuinely contested, and a second dataset
inside the polygon that turns into a figure somebody wants.

## The pairing

Flood extent plus Overture buildings. The number at the end is buildings inside the water,
which is a question with money attached rather than a picture.

**NHD is not a flood dataset.** It is where water normally is: flowlines and waterbodies.
Buildings joined to NHD gives proximity, and proximity is not risk. NHD enters as a
component, not as the answer.

Two routes, and they are different projects.

### Take the polygon as given: FEMA NFHL

The National Flood Hazard Layer is the 100-year and 500-year floodplain, public, vector,
and already the operational object: it is what sets insurance requirements. Join buildings
to it and you get counts by flood zone immediately, because the hard part is handed over.

The catch is that this is a pre-thresholded input again, with one real difference from
NLCD: that line is contested. It is widely known to be under-mapped and out of date, so
"what changes if the line moves 30 cm" is a live argument rather than a setting.

### Compute the field: HAND, from the DEM already being streamed

Height Above Nearest Drainage. Take the 3DEP DEM already pulled from `prd-tnm`, take NHD
flowlines as the drainage network, and for each cell compute elevation above the stream it
drains to. Threshold at a stage in metres and that is an inundation extent. NOAA's
National Water Model uses the same approach operationally.

This is the version where the existing pipeline earns its place: continuous field, the
threshold is the user's and genuinely arguable, the polygon exists because of their choice,
and dragging stage from 1 m to 5 m makes the building count climb. That is the demo.

Unverified, and worth checking before committing:

- How NHD is best consumed here. There may or may not be a clean cloud-native mirror; the
  HUC-based downloads are the fallback and are not small.
- Whether the flow-direction work HAND needs is tractable per-AOI or wants a precomputed
  layer. This is the piece most likely to sink the route.

## Join mechanics

**Do not polyfill the buildings.** At res 11 a hexagon is about 25 m across and a house is
10 to 15 m, so a polyfill returns zero cells for most footprints and the count silently
undercounts. Take the centroid through the existing `h3_latlng_to_cell` UDF at a fine res
(12 or 13), then `cell_to_parent` up to whatever res the flood layer is folded at. Hexify
once, join at any resolution, no re-read. Polyfill only earns its cost for area-weighted
exposure, meaning partial inundation of a footprint, and that is a later refinement.

The Overture gotchas are already in `CLAUDE.md` and all apply: the buildings theme needs
`load_parts` or `load_wkb` because the geometry column is Polygon in some files and
MultiPolygon in others, and `file_index()` gets an AOI read down to ~1.4 s once the
GeoParquet footers are cached. Height is present on 55-75% of footprints, so exposure is
honest by count and by footprint area, and needs a caveat by volume.

The dissolve also has to stop throwing its cells away. `WASH_SQL` runs
`ST_Dump(ST_GeomFromWKB(mp))` and drops the cell ids, which is why the imagery notebook
has to derive its cell count from area rather than count it. Carrying the ids through is
what turns a polygon from a picture into a query handle, and every join above needs it.

## Ideas raised and not taken

Recorded so they are not re-proposed as if new.

- **Year slider over Annual NLCD.** The AOI lane already reads all 40 years for a drawn
  box; only the map is pinned at `YEAR = 2024`. Pixels are cheap (40 years at res 11 is
  ~2.9 MB, against a 384 MB tile budget) and the tile cache takes a year in its key as a
  one-line change. The wire is the real cost: 40 years of dissolved polygons is 40x the
  geometry at ~0.412 MB a year, so it wants dissolve-on-release, one year at a time.
- **Two-year difference map**, gained and lost bands filled between the two extents.
  Rejected on two grounds: NLCD reclassifies year to year for reasons that are not the
  ground changing, and differencing does not cancel that noise, it doubles it. And most
  of the country did not change, so the map is blank except where it is wrong.
- **Collapse time instead of scrubbing**: persistence (how many of 40 years a cell held
  the class) or year-of-last-change, on a luminance ramp. Costs one layer on the wire
  regardless of how many years went in. Still the best version of the time idea if it
  ever comes back.
- **Tracking one polygon across years** by seed cell, reporting splits and merges. The
  most interesting and the most exposed to the same reclassification noise.
- **Alpha to encode change.** Alpha runs one direction and change runs two, so alpha alone
  makes a patch that halved look like one that doubled. Works as magnitude with hue
  carrying direction (blue grew, orange shrank), or alone for something genuinely
  one-directional like persistence. Also unreliable over Esri imagery, where the backdrop
  swings from dark forest to bright field and the same alpha reads as two intensities.
  Alpha wants the dark basemap.

## Scope

This is a new notebook, not a feature added to `xsql-nlcd-imagery.py`. Each notebook in
this repo is one argument, and the imagery one's argument is whether the line matches the
ground. `VIEW_H = 620` was picked so the map, status line, legend, caption and controls fit
a laptop without scrolling; there is no room for another control row that does not come out
of the map.
