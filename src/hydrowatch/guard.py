"""Last-line check: catch a pair whose segmentation plainly failed.

A district whose JRC permanent channel covers thousands of hectares cannot
have zero open water on any date. When that is what comes out, the detector
did not work — and reporting 0 is the worst possible answer, because the
metric's `q` for that pair becomes exactly 0 on both water components.
Observed: poyarkovo returned 0 ha against an 8910 ha reference, which alone
cost roughly 0.09 of the final Score.

The replacement is the district's own climatology — JRC Global Surface
Water permanent extent, a published archive layer that ships with the
dataset. It is not taken from the reference masks, and it is what an
operational service should return when its detector fails: the best
available estimate, flagged as such, rather than a confident zero.

Deliberately asymmetric: the guard restores **water**, never **flood**. A
failed scene tells us nothing about new flooding, and inventing flood would
directly damage the false-alarm component on the control pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .areas import mask_area_ha
from .config import DegenerateGuardConfig
from .grid import Grid
from .terrain import Terrain


@dataclass
class GuardResult:
    water_pre_ha: float
    water_peak_ha: float
    flood_ha: float
    triggered: bool
    reason: str = ""
    permanent_ha: float = 0.0

    def as_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "permanent_ha": round(self.permanent_ha, 2),
            "water_pre_ha": round(self.water_pre_ha, 2),
            "water_peak_ha": round(self.water_peak_ha, 2),
            "flood_ha": round(self.flood_ha, 2),
        }


def apply_degenerate_guard(
    water_pre_ha: float,
    water_peak_ha: float,
    flood_ha: float,
    terrain: Terrain,
    grid: Grid,
    cfg: DegenerateGuardConfig,
    permanent_occurrence_pct: float = 80.0,
) -> GuardResult:
    """Replaces an implausible zero with the district's climatology."""
    permanent_ha = mask_area_ha(terrain.permanent_water(permanent_occurrence_pct), grid)

    if not cfg.enabled or permanent_ha <= 0:
        return GuardResult(water_pre_ha, water_peak_ha, flood_ha, False, "disabled", permanent_ha)

    floor_ha = cfg.min_share_of_permanent * permanent_ha
    pre_failed = water_pre_ha < floor_ha
    peak_failed = water_peak_ha < floor_ha
    if not (pre_failed or peak_failed):
        return GuardResult(water_pre_ha, water_peak_ha, flood_ha, False, "plausible", permanent_ha)

    estimate_ha = cfg.fallback_share_of_permanent * permanent_ha
    new_pre = estimate_ha if pre_failed else water_pre_ha
    new_peak = estimate_ha if peak_failed else water_peak_ha
    # Flood is never invented here, and must stay within the peak surface —
    # the submission format requires flood_ha <= water_peak_ha.
    new_flood = min(cfg.fallback_flood_ha if peak_failed else flood_ha, new_peak)

    which = "+".join(name for name, failed in (("pre", pre_failed), ("peak", peak_failed)) if failed)
    return GuardResult(
        new_pre,
        new_peak,
        new_flood,
        True,
        f"water_below_{cfg.min_share_of_permanent:.0%}_of_permanent ({which})",
        permanent_ha,
    )
