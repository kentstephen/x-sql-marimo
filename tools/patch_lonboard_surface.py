"""Unlight lonboard's SurfaceLayer, and give its texture anisotropic filtering.

Re-run after any install: this edits the shipped JS bundle in site-packages, so `uv sync`,
a lonboard upgrade or a `--sandbox` run all revert it.

    uv run python tools/patch_lonboard_surface.py
    uv run python tools/patch_lonboard_surface.py --revert

WHY. lonboard's SurfaceLayer renders through deck's SimpleMeshLayer and passes it two mesh
attributes, POSITION and TEXCOORD_0. No NORMAL. The natural assumption, and the one the
drape notebooks were written under, is that a mesh without normals cannot be lit, so the
surface renders as pure texture. That is not what deck does. SimpleMeshLayer reads:

    flatShading: !this.state.hasNormals

so a mesh with no normals turns FLAT SHADING ON, and the shader derives a face normal per
triangle from screen-space derivatives and lights it with the default material
(`material: true`). Every triangle therefore gets exactly one brightness, chosen by its
orientation against deck's default light.

On a drape that is ruinous. A photograph already contains the real sun, and deck adds a
second one that is quantised per triangle: the surface breaks into pale and dark facets
the size of a mesh quad, the two triangles of each quad disagree, and the result reads as
a herringbone of translucent quadrilaterals laid over the imagery. It looks like a texture
bug, a mipmap bug or a depth bug, and it is none of those. It survives every parameter in
the notebook because it is not in the notebook.

  * `material: false` disables the lighting module, so colour is the texture and nothing
    else, which is what a draped photograph wants.
  * `maxAnisotropy: 16` is unrelated and cheap: a drape is a texture at a grazing angle,
    the sampler default is `maxAnisotropy: 1`, and without it the GPU picks a mip level
    from the worst axis and blurs both. This mostly shows on wide AOIs where texels are
    tens of metres.

Both belong upstream as traits on SurfaceLayer (`material`, `texture_parameters`), at
which point this file can be deleted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import lonboard

# The tail of lonboard's SurfaceLayer `layerProps()`, minified. Matched literally, so a
# lonboard release that reshapes it makes this script say so rather than guess.
ANCHOR = "_instanced:!1,getPosition:[0,0,0],getColor:[255,255,255]}}render()"
PROPS = "material:!1,textureParameters:{maxAnisotropy:16},"
PATCHED = ANCHOR.replace("_instanced:!1,", "_instanced:!1," + PROPS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    bundle = pathlib.Path(lonboard.__file__).parent / "static" / "index.js"
    backup = bundle.with_suffix(".js.orig")

    # Always start from pristine when a backup exists, so re-running after an earlier
    # version of this patch replaces it instead of stacking on top of it.
    src = backup.read_text() if backup.exists() else bundle.read_text()

    if args.revert:
        if not backup.exists():
            print("no .orig backup to revert to")
            return 1
        bundle.write_text(src)
        print(f"reverted {bundle}")
        return 0

    hits = src.count(ANCHOR)
    if hits != 1:
        print(
            f"expected exactly 1 match for the SurfaceLayer anchor, found {hits}.\n"
            f"lonboard {lonboard.__version__} probably reshaped its bundle; re-derive "
            f"the anchor from `layerProps()` in {bundle}."
        )
        return 1

    if not backup.exists():
        backup.write_text(src)
    bundle.write_text(src.replace(ANCHOR, PATCHED))
    print(
        f"patched {bundle}\n"
        f"  material: false            -> no deck lighting, colour is the texture\n"
        f"  textureParameters: {{maxAnisotropy: 16}}\n"
        f"  backup at {backup}\n\n"
        f"Restart the marimo kernel AND hard-reload the tab (Cmd+Shift+R): the widget JS "
        f"is cached client side, so a kernel restart alone will not pick this up."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
