"""Shared Overture Maps data functions for marimo notebooks."""

import asyncio
import json
import pathlib
import warnings

import numpy as np
import pyarrow as pa
from geoarrow.rust.core import get_type_id, to_wkb
from geoarrow.rust.io import GeoParquetDataset, GeoParquetFile
from lonboard import PathLayer, PolygonLayer, ScatterplotLayer
from obstore.store import S3Store

warnings.filterwarnings("ignore", message="No CRS exists on data")
warnings.filterwarnings("ignore", message="Successfully reconstructed a store")

OVERTURE_BUCKET = "overturemaps-us-west-2"
OVERTURE_RELEASE = "2026-07-22.0"
OVERTURE_S3 = f"s3://{OVERTURE_BUCKET}/release/{OVERTURE_RELEASE}/"

# Every theme/type pair is the same shape of thing: a hive prefix under a release. Named
# here so a notebook asks for "buildings" rather than pasting a path, and so asking for
# several at once is a list comprehension.
THEMES = {
    "buildings": "theme=buildings/type=building",
    "building_parts": "theme=buildings/type=building_part",
    "places": "theme=places/type=place",
    "segments": "theme=transportation/type=segment",
    "connectors": "theme=transportation/type=connector",
    "infrastructure": "theme=base/type=infrastructure",
    "land": "theme=base/type=land",
    "land_cover": "theme=base/type=land_cover",
    "land_use": "theme=base/type=land_use",
    "water": "theme=base/type=water",
    "addresses": "theme=addresses/type=address",
    "divisions": "theme=divisions/type=division_area",
}


def releases():
    """Every published release id, oldest first."""
    store = S3Store.from_url(
        f"s3://{OVERTURE_BUCKET}/release/", region="us-west-2", skip_signature=True
    )
    return sorted(p.rstrip("/").split("/")[-1] for p in store.list_with_delimiter("")["common_prefixes"])


def get_store(release=OVERTURE_RELEASE):
    """Create an anonymous S3Store rooted at one Overture release."""
    return S3Store.from_url(
        f"s3://{OVERTURE_BUCKET}/release/{release}/", region="us-west-2", skip_signature=True
    )


def resolve(theme):
    """A THEMES key or a raw `theme=.../type=...` prefix -> the prefix."""
    return THEMES.get(theme, theme)


def index_path(theme, cache_dir=".cache", release=OVERTURE_RELEASE):
    """Where the cached file index for a theme lives."""
    name = resolve(theme).replace("/", "-")
    return pathlib.Path(cache_dir) / f"overture-index-{release}-{name}.json"


async def file_index(store, theme, cache_dir=".cache", release=OVERTURE_RELEASE):
    """[(path, [w, s, e, n])] for every parquet file under a theme/type prefix.

    This is the part that has to be cached. Overture ships a theme as ~512 files of half
    a gigabyte, and the file-level bbox lives in the GeoParquet footer, so building the
    index means 512 footer reads: ~100 s the first time. It is per RELEASE and per THEME,
    it never changes under a fixed release id, and with it in hand an AOI read touches
    only the one or two files that actually overlap (~1 s). Without it, the alternatives
    measured are a 35 s DuckDB scan per query, or GeoParquetDataset.open, which refuses
    these files outright: the geometry column is Polygon in some parts and MultiPolygon in
    others, and a dataset wants one type.
    """
    path = resolve(theme)
    cache = index_path(theme, cache_dir, release)
    if cache.exists():
        return [(p, b) for p, b in json.loads(cache.read_text())]

    objects = store.list_with_delimiter(path)["objects"]
    sem = asyncio.Semaphore(32)

    async def _one(obj):
        async with sem:
            f = await GeoParquetFile.open_async(obj["path"], store=store)
            try:
                bounds = list(f.file_bbox())
            except Exception:
                bounds = None  # no covering bbox: has to be read every time
            return obj["path"], bounds

    index = list(await asyncio.gather(*[_one(o) for o in objects]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    return index


def index_hits(index, bbox):
    """The files in an index whose own bbox overlaps this AOI."""
    w, s, e, n = bbox
    return [
        p
        for p, b in index
        if b is None or (b[0] < e and b[2] > w and b[1] < n and b[3] > s)
    ]


async def load_parts(store, theme, bbox, index):
    """[arro3 table] for one AOI, one per intersecting file, geometry left as GeoArrow.

    Kept per file rather than concatenated because the parts genuinely disagree about
    geometry type (Polygon vs MultiPolygon), which is the same thing that stops
    GeoParquetDataset from opening them. lonboard is happy to take one layer per part.
    """
    out = []
    for path in index_hits(index, bbox):
        f = await GeoParquetFile.open_async(path, store=store)
        out.append(await f.read_async(bbox=bbox))
    return out


async def load_wkb(store, theme, bbox, index, columns=None):
    """One PyArrow table for an AOI with `geometry` as WKB, concatenated across files.

    WKB is what makes the concat legal (a binary column has no opinion about Polygon vs
    MultiPolygon) and it is also what shapely wants, so anything that reads geometry in
    numpy rather than handing it to deck should come through here.
    """
    out = []
    for data in await load_parts(store, theme, bbox, index):
        # to_wkb has to run on the GeoArrow column, before the PyArrow conversion drops
        # the extension metadata it reads the geometry type from.
        wkb = pa.chunked_array(
            [pa.array(c) for c in pa.chunked_array(to_wkb(data.column("geometry"))).chunks]
        ).cast(pa.binary())
        table = pa.RecordBatchReader.from_stream(data).read_all()
        keep = [c for c in (columns or table.schema.names) if c != "geometry"]
        out.append(
            pa.table({**{c: table.column(c) for c in keep if c in table.schema.names},
                      "geometry": wkb})
        )
    if not out:
        return None
    return pa.concat_tables(out, promote_options="permissive")


def load_geoarrow(store, path, bbox):
    """Load GeoParquet data for a path and bbox, return raw GeoArrow data.

    Preserves extension metadata needed by geoarrow-rust-core (get_type_id, etc.).
    Convert to PyArrow table with: pa.RecordBatchReader.from_stream(data).read_all()
    """
    objects = store.list_with_delimiter(path)["objects"]
    dataset = GeoParquetDataset.open(objects, store=store)
    return dataset.read(bbox=bbox)


def load_data(store, path, bbox):
    """Load GeoParquet data for a path and bbox, return a PyArrow table."""
    return pa.RecordBatchReader.from_stream(load_geoarrow(store, path, bbox)).read_all()


def filter_by_class(table, class_value):
    """Filter a PyArrow table by class column value."""
    classes = table.column("class").to_pylist()
    mask = [c == class_value for c in classes]
    return table.filter(mask)


def filter_to_lines(table):
    """Filter a table to only line geometries (LineString, MultiLineString + Z/M variants)."""
    type_ids = []
    for chunk in get_type_id(table.column("geometry")):
        type_ids.extend(chunk.to_pylist())
    idx = [i for i, t in enumerate(type_ids) if t in LINE_IDS]
    return table.take(idx)


def _get_voltage(tags):
    """Extract voltage from an Overture source_tags map entry."""
    if tags is None:
        return 0
    for key, val in tags:
        if key == "voltage":
            try:
                # Handle ranges like "115000;230000": take max
                return max(int(v) for v in str(val).replace(";", ",").split(",") if v.strip().isdigit())
            except (ValueError, TypeError):
                return 0
    return 0


def load_power_lines(store, bbox, min_voltage=115000):
    """Load major power lines from Overture infrastructure.

    Filters to class=power_line, line geometries only, voltage >= min_voltage.
    Keeps native GeoArrow geometry so lonboard can consume directly.
    """
    data = load_geoarrow(store, "theme=base/type=infrastructure", bbox)

    # Extract type IDs while we have GeoArrow metadata
    type_ids = []
    for chunk in get_type_id(data.column("geometry")):
        type_ids.extend(chunk.to_pylist())

    # Filter to line geometries on the arro3 table (preserves GeoArrow geometry)
    line_idx = [i for i, t in enumerate(type_ids) if t in LINE_IDS]
    data = data.take(line_idx)

    # Need PyArrow for class/voltage filtering, but keep arro3 table for geometry
    pa_table = pa.RecordBatchReader.from_stream(data).read_all()

    # Filter to power_line class
    classes = pa_table.column("class").to_pylist()
    keep = [i for i, c in enumerate(classes) if c == "power_line"]
    data = data.take(keep)
    pa_table = pa_table.take(keep)

    # Filter by voltage from source_tags
    source_tags = pa_table.column("source_tags").to_pylist()
    voltages = [_get_voltage(tags) for tags in source_tags]
    keep = [i for i, v in enumerate(voltages) if v >= min_voltage]
    return data.take(keep)


# Geometry type ID sets (base + Z/M/ZM variants)
POLY_IDS = {3, 6, 13, 16, 23, 26, 33, 36}
POINT_IDS = {1, 4, 11, 14, 21, 24, 31, 34}
LINE_IDS = {2, 5, 12, 15, 22, 25, 32, 35}


def build_layers(data, cfg):
    """Build lonboard layers from a GeoArrow table + a LAYER_OPTIONS config dict.

    Splits mixed geometries by type, applies cmap or flat color, merges
    per-geometry-type kwargs from cfg["polygon"], cfg["point"], cfg["line"].

    Accepts raw arro3 GeoArrow data (preferred) or PyArrow tables.
    get_type_id must run on raw data BEFORE PyArrow conversion (metadata is lost).
    """
    # Extract type IDs from raw GeoArrow data before any conversion
    type_ids = []
    for chunk in get_type_id(data.column("geometry")):
        type_ids.extend(chunk.to_pylist())

    if isinstance(data, pa.RecordBatch):
        table = pa.Table.from_batches([data])
    elif not isinstance(data, pa.Table):
        table = pa.RecordBatchReader.from_stream(data).read_all()
    else:
        table = data

    # Per-row colors from cmap, or flat fallback
    if "cmap" in cfg:
        cmap = cfg["cmap"]
        classes = table.column(cmap["column"]).to_pylist()
        all_colors = [cmap["colors"].get(c, cmap["default"]) + [cmap["alpha"]] for c in classes]
    else:
        all_colors = None
    flat = cfg.get("fill_color", [70, 130, 180, 160])

    layers = []
    for type_set, key, LayerCls, color_prop in [
        (POLY_IDS, "polygon", PolygonLayer, "get_fill_color"),
        (POINT_IDS, "point", ScatterplotLayer, "get_fill_color"),
        (LINE_IDS, "line", PathLayer, "get_color"),
    ]:
        idx = [i for i, t in enumerate(type_ids) if t in type_set]
        if not idx:
            continue
        subset = table.take(idx)
        colors = np.array([all_colors[i] for i in idx], dtype=np.uint8) if all_colors else flat
        opts = cfg.get(key, {})
        layers.append(LayerCls(subset, **{color_prop: colors}, auto_highlight=True, **opts))
    return layers
