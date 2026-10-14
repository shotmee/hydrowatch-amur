"""Writes submission.csv and the per-pair GeoTIFF flood masks the platform
expects (see hydrowatch_amur/README.md, "Что нужно сдать" / "Формат
сабмита")."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict

import numpy as np
import rasterio

from .grid import Grid


@dataclass
class PairResult:
    pair_id: str
    flood_ha: float
    water_pre_ha: float
    water_peak_ha: float


def write_submission_csv(rows: list[PairResult], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"])
        for r in rows:
            writer.writerow(
                [r.pair_id, f"{r.flood_ha:.2f}", f"{r.water_pre_ha:.2f}", f"{r.water_peak_ha:.2f}"]
            )


def write_flood_mask_tif(mask: np.ndarray, grid: Grid, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "nodata": None,
    }
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(mask.astype(np.uint8), 1)


def validate_pair_result(
    flood_ha: float, water_pre_ha: float, water_peak_ha: float, aoi_ha: float
) -> list[str]:
    """Mirrors the platform's automatic checks so failures surface locally,
    not after upload."""
    problems = []
    for name, val in (("flood_ha", flood_ha), ("water_pre_ha", water_pre_ha), ("water_peak_ha", water_peak_ha)):
        if val < 0:
            problems.append(f"{name} is negative ({val})")
        if val > aoi_ha:
            problems.append(f"{name} exceeds AOI area ({val} > {aoi_ha})")
    if flood_ha > water_peak_ha + 1e-6:
        problems.append(f"flood_ha ({flood_ha}) exceeds water_peak_ha ({water_peak_ha})")
    return problems
