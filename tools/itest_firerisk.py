"""Drive xsql-firerisk-buildings.py's cells in order and assert the pipeline is right.

    uv run python tools/itest_firerisk.py


A marimo notebook is semantically a flat script whose cells are written in dependency
order, so exec'ing each cell body into one shared namespace reproduces a real run without
a browser, a kernel or a comm channel. Same trick the divisions integration test used.

The load-bearing assertion is the last one: a building's joined RPS is compared against
the raster value at its own centroid, read straight out of the Zarr window. That is what
catches a mis-indexed read or an H3 resolution mismatch, neither of which changes the row
counts and both of which would look completely normal on screen.
"""
import ast
import asyncio
import pathlib
import math
import sys
import textwrap
import time

NB = str(pathlib.Path(__file__).resolve().parent.parent / "xsql-firerisk-buildings.py")


def cells(path):
    tree = ast.parse(open(path).read())
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            isinstance(d, ast.Attribute) and d.attr == "cell" for d in node.decorator_list
        ):
            continue
        body = [b for b in node.body if not isinstance(b, ast.Return)]
        out.append(
            (
                isinstance(node, ast.AsyncFunctionDef),
                "\n".join(ast.unparse(b) for b in body),
            )
        )
    return out


def viewport(ns, lon, lat, zoom):
    """The padded box the notebook's own camera maths would produce."""
    span = 360.0 * ns["VIEW_W"] / (512 * 2**zoom)
    ls = span * (ns["VIEW_H"] / ns["VIEW_W"]) * math.cos(math.radians(lat))
    bb = (lon - span / 2, lat - ls / 2, lon + span / 2, lat + ls / 2)
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    hw = (bb[2] - bb[0]) / 2 * ns["PAD"]
    hh = (bb[3] - bb[1]) / 2 * ns["PAD"]
    return (cx - hw, cy - hh, cx + hw, cy + hh)


async def main():
    ns = {"__name__": "nb"}
    for i, (is_async, src) in enumerate(cells(NB)):
        t = time.perf_counter()
        if is_async:
            # An async wrapper keeps the body's names local, so they are promoted into the
            # shared namespace explicitly. In a real marimo run the body is compiled at
            # module scope and this happens for free.
            wrapper = (
                "async def __cell():\n"
                + textwrap.indent(src, "    ")
                + "\n    globals().update(locals())\n"
            )
            exec(compile(wrapper, f"cell{i}", "exec"), ns)
            await ns["__cell"]()
        else:
            exec(compile(src, f"cell{i}", "exec"), ns)
        print(f"  cell {i:>2} {'async' if is_async else '     '} ok {(time.perf_counter()-t)*1000:8.0f} ms")

    np, H = ns["np"], ns["HOLD"]
    PARADISE = (-121.6219, 39.7596, 13.6)

    print("\n=== opening draw: Paradise, CA z13.6 ===")
    print("  ", ns["status"].value)
    assert H["res"] == 11, f"expected res 11 at z13.6, got {H['res']}"
    assert H["bldtbl"] is not None, "no buildings drawn"
    n_drawn = H["bldtbl"].num_rows
    assert n_drawn > 500, f"only {n_drawn} buildings drawn"
    rps11 = np.asarray(H["bldtbl"]["RPS"])
    print(f"   {n_drawn:,} buildings; RPS min {rps11.min():.4f} "
          f"med {np.median(rps11):.4f} max {rps11.max():.4f}")

    print("\n=== the join against the raster it came from ===")
    box = viewport(ns, *PARADISE)
    meta, key, note = await ns["fetch_buildings"](box, PARADISE[2])
    cx, cy = np.asarray(meta["cx"]), np.asarray(meta["cy"])
    inside = (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])
    print(f"   fetched {meta.num_rows:,} ({note}), {int(inside.sum()):,} inside the fold box")

    mapping = ns["polyfill"](meta, key, 11)
    per_bld = mapping.num_rows / meta.num_rows
    print(f"   polyfill res 11: {mapping.num_rows:,} (id, cell) pairs, "
          f"{per_bld:.2f} cells per building")
    assert per_bld >= 1.0, "overlap must give every footprint at least one cell"

    raw = H["cache"][("2011", 11)][2]
    ns["_register"]("bld_cells", mapping)
    ns["_register"]("cells", raw)
    joined = ns["ctx"].sql(ns["JOIN_SQL"]).to_arrow_table().combine_chunks()

    # The same L0 window the fold read, indexed directly. If the fold's row/col maths or
    # the H3 resolution were wrong, the join would still return this many rows and the
    # values would be somebody else's ground.
    lat, lon = await ns["_coords"](0)
    r0 = int(np.clip(np.searchsorted(lat, box[1], "left"), 0, lat.size))
    r1 = int(np.clip(np.searchsorted(lat, box[3], "right"), 0, lat.size))
    c0 = int(np.clip(np.searchsorted(lon, box[0], "left"), 0, lon.size))
    c1 = int(np.clip(np.searchsorted(lon, box[2], "right"), 0, lon.size))
    win, _, _ = await ns["_read_window"](0, "rps_2011", r0, c0, r1 - r0, c1 - c0)

    ids = {v: i for i, v in enumerate(meta["id"].to_pylist())}
    jid = joined["id"].to_pylist()
    jrps = np.asarray(joined["rps"])
    a, b = [], []
    for k, bid in enumerate(jid):
        m = ids.get(bid)
        if m is None or not inside[m]:
            continue
        ri = int(np.searchsorted(lat, cy[m]) - r0)
        ci = int(np.searchsorted(lon, cx[m]) - c0)
        if 0 <= ri < win.shape[0] and 0 <= ci < win.shape[1] and np.isfinite(win[ri, ci]):
            a.append(jrps[k])
            b.append(float(win[ri, ci]))
    a, b = np.array(a), np.array(b)
    r = float(np.corrcoef(a, b)[0, 1])
    # nanmedian: a zero-risk pixel under a building makes the ratio undefined, and those
    # are common in a town. The correlation above is the real test; this only catches a
    # systematic scale error.
    med_ratio = float(np.nanmedian(a / np.where(b == 0, np.nan, b)))
    print(f"   {a.size:,} buildings compared to the pixel under their centroid")
    print(f"   corr(joined RPS, centroid pixel) = {r:.4f}   median ratio = {med_ratio:.3f}")
    assert r > 0.9, f"join does not track the raster: corr {r:.3f}"
    assert 0.5 < med_ratio < 2.0, f"join is biased against the raster: ratio {med_ratio:.3f}"

    print("\n=== zoom out to z10: buildings off, coarser res ===")

    class VS:
        longitude, latitude, zoom = -121.6219, 39.7596, 10.0

    await ns["refresh"](VS(), force=True)
    print("  ", ns["status"].value)
    assert H["res"] == 9, f"expected res 9 at z10, got {H['res']}"
    assert H["bldtbl"] is None, "buildings should be off below BLD_ZOOM"

    print("\n=== 2047 scenario, back at Paradise ===")
    ns["controls"].year = "2047"

    class VS2:
        longitude, latitude, zoom = PARADISE

    await ns["refresh"](VS2(), force=True)
    print("  ", ns["status"].value)
    r47 = np.asarray(H["bldtbl"]["RPS"])
    print(f"   median RPS 2047 {np.median(r47):.4f} against 2011 {np.median(rps11):.4f}")
    assert np.median(r47) > np.median(rps11), "2047 should not be lower here"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
