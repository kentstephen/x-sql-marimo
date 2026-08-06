"""Unlight lonboard's SurfaceLayer, give its texture anisotropic filtering, and teach it
to take one byte per texel plus a palette instead of finished RGBA.

Re-run after any install: this edits the shipped JS bundle in site-packages, so `uv sync`,
a lonboard upgrade or a `--sandbox` run all revert it.

    uv run python tools/patch_lonboard_surface.py
    uv run python tools/patch_lonboard_surface.py --revert

PATCH 1: material and sampler.

lonboard's SurfaceLayer renders through deck's SimpleMeshLayer and passes it two mesh
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

PATCH 2: indexed textures, i.e. ship the number rather than the colour.

A data surface (elevation, relief, NDVI) is ONE NUMBER per texel that the kernel converts
to RGBA before sending. That means every colour control resends the whole picture: on an
88-tile grid at 513 texels a side, moving `Ramp` or `Reverse ramp` pushes about 92 MB
across the widget bridge to change a 1 KB lookup table. That is the whole reason the
notebook stops feeling interactive on a wide box, and it is not the kernel: the colormap
call itself measures 0.2 s for the entire grid.

So the payload becomes TWO BYTES per texel and the palette travels separately:

    byte 0   palette index, 0 reserved for "no data" (fully transparent)
    byte 1   shade, the hillshade multiplier as 0..255

    ramp_lut a 256 x 4 uint8 table, synced as its own trait, ~1 KB

The browser expands index and shade into RGBA. The GPU still receives ordinary RGBA, so
nothing about the render path, the shader, the mipmaps or the sampler changes: this is a
transport change and only a transport change. What it buys:

  * The height field and its shading cost 2 bytes per texel rather than 4.
  * A PALETTE change costs 1 KB and no texture transfer at all, because the index bytes the
    browser already holds are unchanged. Reverse is a reversal of the table, not of the
    data, for exactly this reason.

`prepareTexture` decides by payload size: `width * height * 2` is indexed, anything else
is the RGBA path it always had, which is what the NAIP photograph still uses. An indexed
payload with no `ramp_lut` on the model renders fully transparent rather than as noise, so
a notebook that forgets to send the table shows nothing instead of showing garbage.

The `ramp_lut` trait is added by the NOTEBOOK, on a SurfaceLayer subclass, not here: any
trait tagged `sync=True` reaches the JS model, and lonboard's base class already listens
for `change` on the whole model, so a new trait redraws without a new listener.

Patch 1 belongs upstream as traits on SurfaceLayer (`material`, `texture_parameters`).
Patch 2 belongs upstream as a texture format the trait understands. At that point this
file can be deleted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import lonboard

# The tail of lonboard's SurfaceLayer `layerProps()`, minified. Matched literally, so a
# lonboard release that reshapes it makes this script say so rather than guess.
ANCHOR_PROPS = "_instanced:!1,getPosition:[0,0,0],getColor:[255,255,255]}}render()"
PROPS = "material:!1,textureParameters:{maxAnisotropy:16},"
PATCHED_PROPS = ANCHOR_PROPS.replace("_instanced:!1,", "_instanced:!1," + PROPS)

# SurfaceLayer.prepareTexture(), minified, verbatim from the pristine bundle.
ANCHOR_TEX = (
    'prepareTexture(){if(!Ht(this.texture))return;if(typeof this.texture=="string")'
    "return this.texture;let t=new Uint8ClampedArray(this.texture.data.buffer,"
    "this.texture.data.byteOffset,this.texture.data.byteLength);"
    "return new ImageData(t,this.texture.width,this.texture.height)}"
)

# Same head, then: two bytes per texel means indexed, and expand through the table. Index 0
# is left as the zeroed output, i.e. transparent, which is also what an absent or short
# table produces. `x<<2` is the row offset into a 256 x 4 uint8 LUT.
PATCHED_TEX = (
    'prepareTexture(){if(!Ht(this.texture))return;if(typeof this.texture=="string")'
    "return this.texture;let t=new Uint8ClampedArray(this.texture.data.buffer,"
    "this.texture.data.byteOffset,this.texture.data.byteLength),"
    "w=this.texture.width,h=this.texture.height,n=w*h;"
    "if(t.length===n*2){"
    'let L=this.model.get("ramp_lut"),'
    "lut=L?new Uint8Array(L.buffer||L,L.byteOffset||0,L.byteLength||L.length):null,"
    "out=new Uint8ClampedArray(n*4);"
    "if(!lut||lut.length<1024)return new ImageData(out,w,h);"
    "for(let i=0,j=0,k=0;i<n;i++,j+=2,k+=4){"
    "let x=t[j];if(x===0)continue;"
    "let o=x<<2,f=t[j+1]/255;"
    "out[k]=lut[o]*f;out[k+1]=lut[o+1]*f;out[k+2]=lut[o+2]*f;out[k+3]=lut[o+3]}"
    "return new ImageData(out,w,h)}"
    "return new ImageData(t,w,h)}"
)

PATCHES = [
    ("SurfaceLayer.layerProps", ANCHOR_PROPS, PATCHED_PROPS),
    ("SurfaceLayer.prepareTexture", ANCHOR_TEX, PATCHED_TEX),
]


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

    # Every anchor is checked BEFORE anything is written, so a lonboard release that moved
    # one of them leaves the bundle untouched rather than half patched.
    for name, anchor, _ in PATCHES:
        hits = src.count(anchor)
        if hits != 1:
            print(
                f"expected exactly 1 match for the {name} anchor, found {hits}.\n"
                f"lonboard {lonboard.__version__} probably reshaped its bundle; re-derive "
                f"the anchor from {bundle}."
            )
            return 1

    out = src
    for _, anchor, patched in PATCHES:
        out = out.replace(anchor, patched)

    if not backup.exists():
        backup.write_text(src)
    bundle.write_text(out)
    print(
        f"patched {bundle}\n"
        f"  material: false            -> no deck lighting, colour is the texture\n"
        f"  textureParameters: {{maxAnisotropy: 16}}\n"
        f"  prepareTexture             -> index + shade + ramp_lut, expanded in the browser\n"
        f"  backup at {backup}\n\n"
        f"Restart the marimo kernel AND hard-reload the tab (Cmd+Shift+R): the widget JS "
        f"is cached client side, so a kernel restart alone will not pick this up."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
