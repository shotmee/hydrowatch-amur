"""Post-processing: minimum mapping unit, hydrological plausibility,
connectivity to the drainage network.

These three filters target exactly the false-positive sources the case
brief calls out (radar shadow, smooth dry surfaces, isolated "lakes" far
from any channel) without ever touching the reference masks.
"""
from __future__ import annotations

import json

import numpy as np
import pyproj
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, label

from .grid import Grid


def _reproject_coords(coords, transformer: pyproj.Transformer):
    """Recursively reprojects a GeoJSON `coordinates` array of arbitrary
    nesting depth (Point -> ... -> MultiPolygon)."""
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (int, float)):
        x, y = transformer.transform(coords[0], coords[1])
        return [x, y]
    return [_reproject_coords(c, transformer) for c in coords]


def _reproject_geojson_features(geojson_path: str, dst_crs: str) -> list[dict]:
    """OSM exports ship in EPSG:4326; the pair grid is a local UTM zone
    (EPSG:32652). Rasterizing without reprojecting silently produces an
    all-zero mask (degrees vs. metres never overlap) — caught in
    development by a coverage sanity-check on a known river AOI."""
    with open(geojson_path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    src_crs = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326") if gj.get("crs") else "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    features = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        reproj_geom = dict(geom)
        reproj_geom["coordinates"] = _reproject_coords(geom["coordinates"], transformer)
        features.append(reproj_geom)
    return features


def remove_small_objects(mask: np.ndarray, min_pixels: int = 25) -> np.ndarray:
    """Drop connected components smaller than the competition's stated
    minimum mapping unit (25 px = 0.25 ha at 10 m)."""
    labeled, n = label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_pixels
    keep[0] = False  # background
    return keep[labeled]


def apply_hydro_filter(mask: np.ndarray, implausible: np.ndarray) -> np.ndarray:
    return mask & ~implausible


def rasterize_hydrography(geojson_path: str, grid: Grid, buffer_px: int = 3) -> np.ndarray:
    """Rasterize the OSM hydrography network onto the pair grid and dilate
    it by a few pixels so a flood blob only has to *touch* a channel, not
    be pixel-perfect aligned with a coarse vector line.
    """
    geoms = _reproject_geojson_features(geojson_path, grid.crs)
    shapes = [(geom, 1) for geom in geoms]
    if not shapes:
        return np.zeros((grid.height, grid.width), dtype=bool)
    burned = rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    if buffer_px > 0:
        burned = binary_dilation(burned, iterations=buffer_px)
    return burned


def two_tier_connectivity_filter(
    mask: np.ndarray,
    strong_seed: np.ndarray,
    weak_seed: np.ndarray,
    weak_min_pixels: int,
) -> np.ndarray:
    """Connectivity where the required evidence depends on the anchor.

    A flood component touching a mapped channel (`strong_seed`: OSM
    hydrography + permanent water) is accepted at the normal minimum
    mapping unit. A component anchored only to `weak_seed` — JRC's "this
    pixel has been water at some point in the archive" — has to be at least
    `weak_min_pixels` to count.

    The two anchors are not equally informative and the measurement says so:
    adding the weak anchor lifted Q_flood from 0.208 to 0.362 but dropped
    Spec_base from 0.951 to 0.738, because on a dry control pair it lets
    scattered noise survive. Size is the cheapest discriminator available —
    real flooding arrives as large coherent patches, phantom flood does not.
    """
    labeled, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask

    strong_ids = set(np.unique(labeled[strong_seed & mask]))
    strong_ids.discard(0)

    weak_ids = set(np.unique(labeled[weak_seed & mask]))
    weak_ids.discard(0)
    weak_only = weak_ids - strong_ids

    keep = np.zeros(count + 1, dtype=bool)
    for component_id in strong_ids:
        keep[component_id] = True

    if weak_only:
        sizes = np.bincount(labeled.ravel(), minlength=count + 1)
        for component_id in weak_only:
            keep[component_id] = sizes[component_id] >= weak_min_pixels

    keep[0] = False
    return keep[labeled]


def connectivity_filter(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep only connected components of `mask` that touch `seed`
    (permanent water and/or the buffered hydrography network). Drops
    isolated "lakes" on watersheds — a textbook false positive per the
    case brief's hydrological-context section.
    """
    labeled, n = label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask
    touching_ids = set(np.unique(labeled[seed & mask]))
    touching_ids.discard(0)
    if not touching_ids:
        return np.zeros_like(mask)
    keep = np.isin(labeled, list(touching_ids))
    return keep
