"""Machine-readable summary report per pair.

Required by the rubric: «площадь водного зеркала на каждую дату, площадь
изменения — как прироста, так и убыли — в гектарах и квадратных километрах,
доля затопленной площади от площади запрошенной территории, распределение
затопления по типам земного покрова; выгрузка отчёта в машиночитаемом
формате».

Note the deliberate split between `flood` (new water, what the competition
scores) and `water_gain` (peak minus pre, including permanent water
fluctuation): reporting both is what makes the loss side — `receded` — add
up, and conflating them is an easy way to publish numbers that don't
reconcile.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from .areas import mask_area_ha
from .grid import Grid
from .terrain import Terrain

# ESA WorldCover v200 class codes, as shipped in the AUX raster's builtup
# band derivation. We only carry a coarse split here because the AUX raster
# gives us `builtup` as a binary layer rather than the full class map.
LANDCOVER_LABELS = {
    "builtup": "Застроенные территории (ESA WorldCover)",
    "permanent_water": "Постоянная вода (JRC GSW occurrence >= 80%)",
    "seasonal_water": "Сезонная вода (JRC GSW max_extent, не постоянная)",
    "other": "Прочие типы покрова",
}


def landcover_breakdown(flood: np.ndarray, terrain: Terrain, grid: Grid) -> dict:
    """Where did the new water land? Uses the layers we actually have in
    AUX_terrain_gsw.tif rather than pretending to a full class map."""
    permanent = terrain.permanent_water()
    seasonal = (terrain.max_extent > 0.5) & ~permanent
    builtup = terrain.builtup >= 0.5

    classes = {
        "builtup": flood & builtup,
        "permanent_water": flood & permanent,
        "seasonal_water": flood & seasonal & ~builtup,
    }
    classes["other"] = flood & ~(classes["builtup"] | classes["permanent_water"] | classes["seasonal_water"])

    total_ha = mask_area_ha(flood, grid)
    out = {}
    for key, mask in classes.items():
        ha = mask_area_ha(mask, grid)
        out[key] = {
            "label": LANDCOVER_LABELS[key],
            "area_ha": round(ha, 2),
            "area_km2": round(ha / 100.0, 4),
            "share_of_flood": round(ha / total_ha, 4) if total_ha > 0 else 0.0,
        }
    return out


def build_pair_report(
    pair_id: str,
    grid: Grid,
    water_pre: np.ndarray,
    water_peak: np.ndarray,
    flood: np.ndarray,
    receded: np.ndarray,
    terrain: Terrain,
    aoi_ha: float,
    meta: dict | None = None,
) -> dict:
    def both_units(mask):
        ha = mask_area_ha(mask, grid)
        return {"ha": round(ha, 2), "km2": round(ha / 100.0, 4)}

    water_pre_ha = mask_area_ha(water_pre, grid)
    water_peak_ha = mask_area_ha(water_peak, grid)
    flood_ha = mask_area_ha(flood, grid)

    return {
        "pair_id": pair_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {
            "crs": grid.crs,
            "width": grid.width,
            "height": grid.height,
            "pixel_size_m": round(abs(grid.transform.a), 3),
        },
        "aoi": {"ha": round(aoi_ha, 2), "km2": round(aoi_ha / 100.0, 4)},
        "water_surface": {
            "pre": both_units(water_pre),
            "peak": both_units(water_peak),
        },
        "change": {
            "flood_new_water": both_units(flood),
            "receded": both_units(receded),
            "net_gain": {
                "ha": round(water_peak_ha - water_pre_ha, 2),
                "km2": round((water_peak_ha - water_pre_ha) / 100.0, 4),
            },
        },
        "shares": {
            "flood_of_aoi": round(flood_ha / aoi_ha, 6) if aoi_ha else 0.0,
            "water_peak_of_aoi": round(water_peak_ha / aoi_ha, 6) if aoi_ha else 0.0,
            "water_pre_of_aoi": round(water_pre_ha / aoi_ha, 6) if aoi_ha else 0.0,
            "peak_vs_pre_growth_pct": (
                round(100.0 * (water_peak_ha - water_pre_ha) / water_pre_ha, 2)
                if water_pre_ha > 0
                else None
            ),
        },
        "flood_landcover_breakdown": landcover_breakdown(flood, terrain, grid),
        "meta": meta or {},
    }


def write_report(report: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


def reports_to_csv_rows(reports: list[dict]) -> list[dict]:
    """Flattens reports for a single machine-readable table."""
    rows = []
    for r in reports:
        rows.append(
            {
                "pair_id": r["pair_id"],
                "aoi_km2": r["aoi"]["km2"],
                "water_pre_ha": r["water_surface"]["pre"]["ha"],
                "water_peak_ha": r["water_surface"]["peak"]["ha"],
                "flood_ha": r["change"]["flood_new_water"]["ha"],
                "receded_ha": r["change"]["receded"]["ha"],
                "net_gain_ha": r["change"]["net_gain"]["ha"],
                "flood_share_of_aoi": r["shares"]["flood_of_aoi"],
                "flood_on_builtup_ha": r["flood_landcover_breakdown"]["builtup"]["area_ha"],
            }
        )
    return rows
