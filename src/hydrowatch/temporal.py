"""Per-AOI threshold prior built from extra Sentinel-1 acquisitions.

**What the extra data is for.** The organizers ship scene *passports*, not
imagery — every pixel is fetched from Earth Engine by us. So nothing stops
us from fetching far more than the 22 scenes the 11 scored pairs name:
`scene_catalog.csv` alone lists 148 Sentinel-1 acquisitions across the five
districts, and the same AOIs can be sampled on any other date.

The metric is still computed on those 11 fixed pairs, so extra scenes
cannot change a prediction directly. What they *can* do is tell us what a
threshold for this district normally looks like — and that turns out to be
the difference between a working pair and a dead one. Observed thresholds
for konstantinovka across scored pairs:

    межень 2018      -13.4 dB
    паводок 2019     -15.4 dB
    паводок 2021 пик -22.0 dB  <- relaxed fallback fired, mask came out empty

The third value is not a measurement, it is a failure: no tile passed, the
threshold collapsed to the dry end of the window and the pair returned
essentially nothing against a non-trivial reference. A district-level prior
makes that outcome impossible: a threshold that disagrees with every other
observation of the same district gets pulled back toward them.

This is deliberately a *shrinkage* prior, not a replacement. When the
per-scene estimate is well supported (many qualifying tiles, tight spread)
it is kept as is; the prior only takes over as the scene's own evidence
weakens. A district whose river genuinely changes character between seasons
is therefore not forced onto a single number.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class AoiThresholdPrior:
    """Distribution of plausible sigma0 thresholds for one district."""

    aoi_id: str
    thresholds_db: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.thresholds_db)

    @property
    def median_db(self) -> float | None:
        return statistics.median(self.thresholds_db) if self.thresholds_db else None

    @property
    def spread_db(self) -> float:
        """Interquartile range; 0 when there is too little to spread."""
        if len(self.thresholds_db) < 4:
            return 0.0
        ordered = sorted(self.thresholds_db)
        q1 = statistics.median(ordered[: len(ordered) // 2])
        q3 = statistics.median(ordered[(len(ordered) + 1) // 2 :])
        return float(q3 - q1)

    def is_outlier(self, threshold_db: float, tolerance_db: float) -> bool:
        """Outside the district's own experience by more than `tolerance`."""
        centre = self.median_db
        if centre is None:
            return False
        allowed = max(tolerance_db, 2.0 * self.spread_db)
        return abs(threshold_db - centre) > allowed


def build_prior(observations: dict[str, list[float]]) -> dict[str, AoiThresholdPrior]:
    """`observations` maps aoi_id -> thresholds measured on extra scenes."""
    return {
        aoi_id: AoiThresholdPrior(aoi_id=aoi_id, thresholds_db=[t for t in values if t == t])
        for aoi_id, values in observations.items()
    }


def regularize_threshold(
    threshold_db: float,
    diagnostics: dict,
    prior: AoiThresholdPrior | None,
    min_observations: int = 3,
    outlier_tolerance_db: float = 3.0,
    fallback_weight: float = 1.0,
    weak_evidence_tiles: int = 5,
    weak_evidence_weight: float = 0.5,
) -> tuple[float, dict]:
    """Shrinks a per-scene threshold toward its district's prior.

    Three regimes, by how much the scene itself supports its own answer:

    - **relaxed fallback fired** — the scene produced no qualifying tile at
      all, so its threshold carries no evidence. Take the prior outright
      (`fallback_weight`, default 1.0).
    - **few qualifying tiles** — some evidence, not much. Blend.
    - **threshold is an outlier** for this district despite decent support —
      blend, because one scene disagreeing with every other observation of
      the same river is more likely wrong than revelatory.

    Otherwise the scene's own estimate is returned untouched.
    """
    info = {"prior_applied": False}
    if prior is None or prior.count < min_observations or prior.median_db is None:
        info["prior_reason"] = "no_prior"
        return threshold_db, info

    centre = prior.median_db
    tiles_kept = int(diagnostics.get("tiles_kept", 0) or 0)
    used_fallback = bool(diagnostics.get("fallback"))

    if used_fallback:
        weight, reason = fallback_weight, "relaxed_fallback"
    elif tiles_kept < weak_evidence_tiles:
        weight, reason = weak_evidence_weight, "few_tiles"
    elif prior.is_outlier(threshold_db, outlier_tolerance_db):
        weight, reason = weak_evidence_weight, "outlier_vs_district"
    else:
        info["prior_reason"] = "scene_evidence_sufficient"
        return threshold_db, info

    adjusted = (1.0 - weight) * threshold_db + weight * centre
    info.update(
        {
            "prior_applied": True,
            "prior_reason": reason,
            "prior_median_db": round(centre, 3),
            "prior_spread_db": round(prior.spread_db, 3),
            "prior_observations": prior.count,
            "threshold_before_prior_db": round(threshold_db, 3),
            "threshold_after_prior_db": round(adjusted, 3),
        }
    )
    return adjusted, info


def prior_summary(priors: dict[str, AoiThresholdPrior]) -> list[dict]:
    return [
        {
            "aoi_id": p.aoi_id,
            "observations": p.count,
            "median_db": round(p.median_db, 2) if p.median_db is not None else None,
            "iqr_db": round(p.spread_db, 2),
        }
        for p in priors.values()
    ]
