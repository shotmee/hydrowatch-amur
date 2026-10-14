"""Re-implementation of the organizers' scoring formula, for **local
self-validation only** (case brief, "Метрика оценки").

IMPORTANT: this module reads areas computed from the reference masks purely
to score a *candidate submission* during development. It must never be
imported by the inference path that produces submission.csv — doing so
would be exactly the "подмена расчёта эталоном" violation that the rules
disqualify.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FLOOD_THRESHOLD_HA = 50.0
WATER_THRESHOLD_HA = 200.0
SPEC_BASE_TOLERANCE = 0.005  # 0.5% of AOI area

WEIGHTS = {"flood": 0.45, "water_peak": 0.25, "water_pre": 0.15, "spec_base": 0.15}


def q_value(x_pred: float, x_ref: float, threshold: float) -> float:
    denom = max(x_ref, threshold)
    return max(0.0, 1.0 - abs(x_pred - x_ref) / denom)


@dataclass
class ScoreReport:
    score: float
    q_flood: float
    q_water_peak: float
    q_water_pre: float
    spec_base: float
    per_pair: pd.DataFrame


def score_submission(
    submission: pd.DataFrame,
    reference: pd.DataFrame,
    event_kind: pd.Series | dict,
    aoi_ha: pd.Series | dict,
) -> ScoreReport:
    """
    Parameters
    ----------
    submission, reference : DataFrame indexed or columned by `pair_id`,
        with `flood_ha`, `water_pre_ha`, `water_peak_ha`.
    event_kind : mapping pair_id -> "baseline" | anything else (event pair).
    aoi_ha : mapping pair_id -> AOI area in hectares (needed for Spec_base).
    """
    sub = submission.set_index("pair_id")
    ref = reference.set_index("pair_id")
    kind = pd.Series(event_kind)
    aoi = pd.Series(aoi_ha)

    rows = []
    for pair_id in ref.index:
        s = sub.loc[pair_id] if pair_id in sub.index else pd.Series(
            {"flood_ha": 0.0, "water_pre_ha": 0.0, "water_peak_ha": 0.0}
        )
        r = ref.loc[pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "kind": kind.get(pair_id, "event"),
                "flood_pred": s["flood_ha"],
                "flood_ref": r["flood_ha"],
                "water_pre_pred": s["water_pre_ha"],
                "water_pre_ref": r["water_pre_ha"],
                "water_peak_pred": s["water_peak_ha"],
                "water_peak_ref": r["water_peak_ha"],
                "aoi_ha": aoi.get(pair_id, float("nan")),
            }
        )
    df = pd.DataFrame(rows)

    events = df[df["kind"] != "baseline"].copy()
    baselines = df[df["kind"] == "baseline"].copy()

    events["q_flood"] = events.apply(
        lambda row: q_value(row.flood_pred, row.flood_ref, FLOOD_THRESHOLD_HA), axis=1
    )
    events["q_water_peak"] = events.apply(
        lambda row: q_value(row.water_peak_pred, row.water_peak_ref, WATER_THRESHOLD_HA), axis=1
    )
    events["q_water_pre"] = events.apply(
        lambda row: q_value(row.water_pre_pred, row.water_pre_ref, WATER_THRESHOLD_HA), axis=1
    )

    q_flood = events["q_flood"].mean()
    q_water_peak = events["q_water_peak"].mean()
    q_water_pre = events["q_water_pre"].mean()

    if len(baselines):
        baselines["false_alarm_share"] = (
            (baselines["flood_pred"] - baselines["flood_ref"]).clip(lower=0.0) / baselines["aoi_ha"]
        )
        baselines["spec"] = 1.0 - (baselines["false_alarm_share"] / SPEC_BASE_TOLERANCE).clip(upper=1.0)
        spec_base = baselines["spec"].mean()
    else:
        spec_base = float("nan")

    score = (
        WEIGHTS["flood"] * q_flood
        + WEIGHTS["water_peak"] * q_water_peak
        + WEIGHTS["water_pre"] * q_water_pre
        + WEIGHTS["spec_base"] * spec_base
    )

    per_pair = pd.concat([events, baselines], axis=0, ignore_index=True)
    return ScoreReport(
        score=score,
        q_flood=q_flood,
        q_water_peak=q_water_peak,
        q_water_pre=q_water_pre,
        spec_base=spec_base,
        per_pair=per_pair,
    )
