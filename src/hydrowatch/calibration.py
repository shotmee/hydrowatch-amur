"""Configuration sweep with leave-one-AOI-out validation.

The case brief explicitly permits this: «Маски выданы для обучения, подбора
порогов и самопроверки». What it forbids is copying reference areas into
the submission. So hyper-parameters may be *tuned* against the reference —
but only honestly, and two properties are enforced here in code rather than
promised in prose:

1. **Folds are districts, not pairs.** 11 pairs come from 5 AOIs; two pairs
   over the same district share terrain, permanent water and geometry.
   Tuning on Blagoveshchensk-2019 and testing on Blagoveshchensk-2021 is
   memorisation, not validation.
2. **The selected configuration is scored out-of-fold.** For each held-out
   district, the winner is chosen on the *other* districts and then applied
   to the held-out one. The gap between the in-sample best and this
   out-of-fold estimate is the honest measure of how much of the gain is
   real — and it is reported, not hidden.

The sweep runs on cached imagery, so a variant costs only re-segmentation,
not another Earth Engine round trip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .areas import mask_area_ha
from .config import Config
from .metrics import score_submission
from .pipeline import segment_pair


@dataclass
class PairInputs:
    """Everything needed to re-segment one pair without touching the network."""

    pair_id: str
    grid_source_path: str
    aux_terrain_path: str
    pre: object  # DateArrays
    peak: object  # DateArrays


def run_variant(
    variant: Config,
    inputs: list[PairInputs],
    hydrography_path: str,
) -> pd.DataFrame:
    """Segments every cached pair with one configuration."""
    rows = []
    for item in inputs:
        masks = segment_pair(
            item.pair_id,
            item.grid_source_path,
            item.aux_terrain_path,
            hydrography_path,
            item.pre,
            item.peak,
            variant,
        )
        rows.append(
            {
                "pair_id": item.pair_id,
                "flood_ha": mask_area_ha(masks.flood, masks.grid),
                "water_pre_ha": mask_area_ha(masks.water_pre, masks.grid),
                "water_peak_ha": mask_area_ha(masks.water_peak, masks.grid),
            }
        )
    return pd.DataFrame(rows)


def score_variant(submission, reference, kind, aoi_ha) -> dict:
    ref = reference[reference["pair_id"].isin(submission["pair_id"])]
    report = score_submission(submission, ref, kind, aoi_ha)
    return {
        "Score": report.score,
        "Q_flood": report.q_flood,
        "Q_water_peak": report.q_water_peak,
        "Q_water_pre": report.q_water_pre,
        "Spec_base": report.spec_base,
    }


def sweep(
    variants: dict[str, Config],
    inputs: list[PairInputs],
    hydrography_path: str,
    reference: pd.DataFrame,
    kind: dict,
    aoi_ha: dict,
    aoi_of_pair: dict,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (in-sample table, out-of-fold table, honest best variant name).

    The in-sample table is what a naive sweep would report. The out-of-fold
    table is what you should actually believe.
    """
    predictions: dict[str, pd.DataFrame] = {}
    rows = []
    for idx, (name, variant) in enumerate(variants.items(), start=1):
        preds = run_variant(variant, inputs, hydrography_path)
        predictions[name] = preds
        rows.append({"variant": name, **score_variant(preds, reference, kind, aoi_ha)})
        if progress is not None:
            progress(idx, len(variants), name)
    in_sample = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    districts = sorted({aoi_of_pair[p] for p in aoi_of_pair if p in set(reference["pair_id"])})
    fold_rows = []
    for held_out in districts:
        held_pairs = [p for p, a in aoi_of_pair.items() if a == held_out]
        train_pairs = [p for p, a in aoi_of_pair.items() if a != held_out]
        if not held_pairs or not train_pairs:
            continue

        # Choose the winner WITHOUT looking at the held-out district.
        best_name, best_train = None, -np.inf
        for name, preds in predictions.items():
            tr = preds[preds["pair_id"].isin(train_pairs)]
            if tr.empty:
                continue
            s = score_variant(tr, reference, kind, aoi_ha)["Score"]
            if s == s and s > best_train:  # s == s guards NaN
                best_train, best_name = s, name
        if best_name is None:
            continue

        held = predictions[best_name][predictions[best_name]["pair_id"].isin(held_pairs)]
        held_scores = score_variant(held, reference, kind, aoi_ha) if not held.empty else {}
        fold_rows.append(
            {
                "held_out_aoi": held_out,
                "pairs": len(held),
                "chosen_on_train": best_name,
                "train_Score": round(best_train, 4),
                "heldout_Score": round(held_scores.get("Score", float("nan")), 4),
                "heldout_Q_flood": round(held_scores.get("Q_flood", float("nan")), 4),
                "heldout_Q_water_peak": round(held_scores.get("Q_water_peak", float("nan")), 4),
            }
        )
    out_of_fold = pd.DataFrame(fold_rows)

    # The honest pick: the variant chosen most often across folds; ties go
    # to the better in-sample Score.
    if not out_of_fold.empty:
        counts = out_of_fold["chosen_on_train"].value_counts()
        top = counts[counts == counts.max()].index.tolist()
        order = {n: i for i, n in enumerate(in_sample["variant"])}
        best = sorted(top, key=lambda n: order.get(n, 999))[0]
    else:
        best = in_sample.iloc[0]["variant"]

    return in_sample, out_of_fold, best
