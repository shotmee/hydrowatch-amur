"""Raster masks -> vector contours (GeoJSON).

Required by the rubric: «выгрузка векторных контуров водной поверхности и
зоны затопления в формате GeoJSON или Shapefile с корректной системой
координат и атрибутами: идентификатор контура, тип, площадь».

Contours are written in EPSG:4326 (the GeoJSON default per RFC 7946, and
what every web map expects) while areas are computed on the source UTM grid,
where a pixel is exactly 100 m² — computing area after reprojection to
degrees would be wrong.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyproj
from rasterio.features import shapes

from .grid import Grid


def _reproject_ring(ring, transformer):
    return [list(transformer.transform(x, y)) for x, y in ring]


def mask_to_features(
    mask: np.ndarray,
    grid: Grid,
    layer: str,
    min_pixels: int = 25,
    to_wgs84: bool = True,
) -> list[dict]:
    """One GeoJSON Feature per connected polygon, with id/type/area attrs."""
    transformer = (
        pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        if to_wgs84
        else None
    )
    pixel_area_ha = grid.pixel_area_ha

    features = []
    for geom, value in shapes(
        mask.astype(np.uint8), mask=mask.astype(bool), transform=grid.transform
    ):
        if value != 1:
            continue
        rings = geom["coordinates"]
        # Shoelace on the exterior ring minus holes, in projected metres.
        area_m2 = _ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:])
        area_ha = abs(area_m2) / 10_000.0
        if area_ha < min_pixels * pixel_area_ha:
            continue

        out_geom = geom
        if transformer is not None:
            out_geom = {
                "type": geom["type"],
                "coordinates": [_reproject_ring(r, transformer) for r in rings],
            }

        features.append(
            {
                "type": "Feature",
                "geometry": out_geom,
                "properties": {
                    "id": f"{layer}_{len(features):04d}",
                    "type": layer,
                    "area_ha": round(area_ha, 4),
                    "area_km2": round(area_ha / 100.0, 6),
                },
            }
        )
    return features


def _ring_area(ring) -> float:
    coords = np.asarray(ring, dtype=np.float64)
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def write_geojson(
    layers: dict[str, np.ndarray],
    grid: Grid,
    out_path: str,
    min_pixels: int = 25,
) -> dict:
    """`layers` maps a layer name (water_pre / water_peak / flood / receded)
    to its boolean mask. All layers go into one FeatureCollection, each
    feature tagged with its `type`, so a map client can filter/toggle."""
    features: list[dict] = []
    for name, mask in layers.items():
        features.extend(mask_to_features(mask, grid, name, min_pixels=min_pixels))

    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, ensure_ascii=False)
    return collection
