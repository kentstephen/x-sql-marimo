# Plan: hexagons as a finder over imagery (deforest x Sentinel-2)

Status: not started. This is the agreed next experiment for `xsql-deforest-divisions.py`,
written down before any code. Read `docs/imagery-and-terrain-notes.md` first; every
render trap named there applies, and the Sentinel-2 data path below is already solved
there in full.

## The idea

The hexagons alone are not that helpful to look at: they say where deforestation is, but
a cividis blob is not evidence and there is nothing under it to go look at. So invert
their job. The hexagons become a FINDER: a threshold slider isolates the cells with the
most deforestation, you fly to one, and then you hide the hexagons (or the threshold
does it for you) and look at the actual ground in high-res imagery. Sentinel-2 (Earth
Genome yearly mosaics) is the default pairing; NAIP is the same idea where CONUS-only
and ~1 m matters.

The pairing fixes both halves. The hexagons get something to point at; the imagery gets
an index, because a global mosaic with nothing over it gives no reason to look anywhere
in particular.

## Why the finder framing changes the old render problem

The imagery render side failed twice (see the notes doc), but both architectures were
built for imagery that follows a live camera. A finder mostly avoids that case:

- Wide view: hexagons only. The Earth Genome pyramid has nothing to draw above map zoom
  ~8 anyway, so nothing is lost.
- By the time a hot cell is picked and dived on, the camera has stopped. The view sits
  inside one MGRS footprint (~147 km across, covers any single view), which is exactly
  the regime where the render trouble had not yet started.
- The zoom bands line up on their own: imagery exists over map zoom 8-13 (H3 res 8-11),
  and the deforest ladder caps at res 8, so the handoff (hexagons carry the wide view,
  imagery carries the deep one) has a natural seam at z8 rather than an arbitrary one.
  State on the page that imagery starts at z8 and ends at z13, or the pyramid ending
  reads as a bug.

## The threshold slider

Nearly free in the current architecture: the fold table is already in `HOLD["cache"]`,
so a threshold is a filter plus a `put_cells` re-push. No raster read, no dissolve.

Two readings of what "threshold" shows, genuinely different on screen; decide by looking:

1. **Filter cells out entirely.** Below-threshold ground opens windows straight onto the
   imagery; the map becomes a mask over the photograph. "Hide hexagons" is then just
   threshold at max (the existing checkbox stays as the blunt version).
2. **Keep every cell, drop below-threshold alpha to near zero.** The field stays legible
   as context while the hot cells burn through.

Mechanics, all from lessons already paid for:

- Commit on `change`, not `input`: each commit re-pushes a table.
- 9rem of track minimum; 6 px per stop is inside trackpad slop.
- The value crosses the anywidget bridge as Unicode (the proven trait types are Unicode
  down and Bool up; List(Float) never arrived).
- The slider lives in the Controls widget, not `mo.ui`, or every drag rebuilds the Map.

## Vintage: mutated, not solved

Deforestation 2002-2022 has no single year for the imagery to match. A 2024 mosaic shows
the AFTER: a hot cell should look like bare ground, pasture, or young regrowth, and 2003
clearing may be fully regrown or converted. Two ways to lean:

1. **Accept it.** The finder says "look here"; the mosaic says "this is what it looks
   like now." Honest, useful, zero extra machinery.
2. **Exploit the yearly mosaics.** A year flip (say 2018 vs 2024) over a hot cell shows
   change directly, inside the Sentinel era. The finder points, the flip proves. This
   would be the first thing in the repo where imagery is evidence rather than backdrop.
   The `datetime` filter does not constrain the STAC items; enforce the year on the item
   id (`13TDE_2024-01-01_2025-01-01`), per the notes doc.

## Render paths, in the order to try them

Stephen's read, and it is fair: the earlier failures do not prove the layer cannot work,
because the notes' own diagnosis is that the `BitmapLayer image=""` poisoned-update-pass
mechanism likely took down BOTH architectures. Neither was tested clean.

1. **`RasterLayer.from_geotiff`, retried with the traps closed.** Right architecture
   (browser-side range reads per visible tile, no pixels through the kernel). Known
   traps, all recorded: zoom bounds and `extent` through `**kwargs` (they ship commented
   out and the fetcher wraps negative), `Tile.array` not `.data` (the AttributeError is
   silent in the comm handler), transparent placeholder never `image=""`, TCI black fill
   made transparent, `tile_size` vs retina.
2. **A local tile endpoint feeding `BitmapTileLayer`.** The COGs are EPSG:3857 and
   already cut to the WebMercatorQuad grid, so an in-kernel HTTP handler serving
   `{z}/{x}/{y}` is integer window reads plus WebP encode (83 KB/tile measured, 21 ms).
   Marries the one layer that has always worked to the solved data path, at the cost of
   running a little server next to the kernel.
3. **One-shot `BitmapLayer` per settled view.** ~950 KB of base64 per push was
   disqualifying for a live camera; in a finder you push only when the dive settles.
   Simplest possible fallback if 1 and 2 both die.

Use `TCI`, never composite `B04/B03/B02` (the sub-pixel seam bug in
`sentinel-2-cog-deckgl-raster/docs/SEAMS.md`).

## The open question: does this belong in the browser instead?

Worth knowing before sinking another session into lonboard's render path: would this
work better as a browser app on deck.gl-raster (Development Seed), with the kernel gone?

What moves cleanly:

- The raster read side. deck.gl-raster's whole point is browser-side COG windowing and
  GPU band math, and the Earth Genome COGs are the friendliest possible case (3857,
  WebMercatorQuad-aligned, precomposed TCI).
- The camera/settle/cache machinery is all reimplementable state; nothing about it is
  Python-shaped.

What is fussy, and why this stays a question rather than a plan:

- **The fold needs H3 in the browser, and h3o IS the fold engine already.** h3ronpy is
  a wrapper around h3o (Rust), and h3o is why the fold wins its benchmark (70 ms vs
  462 ms; the archive comparison is which-H3-lives-underneath, h3o vs Uber's C). Rust
  compiles to wasm, so the natural port is h3o-to-wasm: the same speed demon, one
  toolchain step away. Check whether a published h3o wasm package exists before
  building one. h3-js (the emscripten transpile of Uber's C, scalar calls in a loop)
  is the fallback, and it is the losing library from that benchmark, so measure it
  rather than assume it is fine: a viewport is order 10^5-10^6 pixels at these zooms.
- **Getting folded cells into deck as geoarrow.** The Python side hands lonboard real
  Arrow tables; in the browser the equivalent is building geoarrow buffers by hand or
  taking deck.gl's H3HexagonLayer with string/bigint cell ids and eating the conversion.
  Fussy, not impossible.
- **The divisions machinery would need DuckDB-WASM** with the spatial and h3 community
  extensions. The h3 extension working under wasm is CONFIRMED: Isaac Brodsky (the
  extension's author) has said it does. So the divisions join survives a browser port,
  and DuckDB-WASM could even cover the fold-side H3 call if the JS loop measures slow,
  though that trades the per-row-call overhead the Python split exists to avoid.

Cheap probe: look for a published h3o wasm build, then time it (or h3-js as fallback)
folding a synthetic 1M-pixel viewport, then decide. The marimo version does not wait on
this; path 1 above is a one-session experiment either way.
