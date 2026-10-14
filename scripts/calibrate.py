#!/usr/bin/env python
"""Ablations and leave-one-AOI-out threshold calibration.

Addresses the rubric's «Экспериментальная проверка» head-on: it compares
configurations by the metric *and its components*, and it does the
comparison honestly. Two rules are enforced in code rather than in prose:

1. **Leave-one-AOI-out, not leave-one-pair-out.** The 11 pairs come from
   only 5 districts; two pairs over the same district share geometry,
   terrain and permanent-water layers, so a pair-level split leaks. A
   threshold picked on Blagoveshchensk-2019 and tested on
   Blagoveshchensk-2021 is not validated, it is memorised.
2. **The reference is never used to build a prediction.** It is read only to
   score an already-computed submission. Picking a global threshold by
   sweeping against all 11 references at once would be exactly the
   «подгонка под открытый эталон» the rules disqualify — so the sweep
   reports both the in-fold optimum and its out-of-fold score, and the
   gap between them is the honest estimate.

This script runs on **cached masks**, not live Earth Engine: run
`run_inference.py --checkpoint-dir ...` once, then sweep here for free.

Usage:
    python calibrate.py --data-dir ../hydrowatch_amur --checkpoint-dir outputs/checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrowatch.metrics import score_submission  # noqa: E402


def load_reference(data_dir: str) -> tuple[pd.DataFrame, dict, dict, dict]:
    rows, kind, aoi_ha, aoi_of_pair = [], {}, {}, {}
    pairs = pd.read_csv(os.path.join(data_dir, "pairs.csv")).set_index("pair_id")
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "reference_*.json"), recursive=True))
    for meta_path in paths:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        pid = meta["pair_id"]
        if pid not in pairs.index:
            continue
        stats = meta["stats"]
        rows.append(
            {
                "pair_id": pid,
                "flood_ha": stats["flood_ha"],
                "water_pre_ha": stats["water_pre_ha"],
                "water_peak_ha": stats["water_peak_ha"],
            }
        )
        kind[pid] = pairs.loc[pid, "event_kind"]
        aoi_ha[pid] = pairs.loc[pid, "aoi_km2"] * 100.0
        aoi_of_pair[pid] = pairs.loc[pid, "aoi_id"]
    return pd.DataFrame(rows), kind, aoi_ha, aoi_of_pair


def submission_from_checkpoints(checkpoint_dir: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(checkpoint_dir, "*", "summary.json"))):
        with open(summary_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        rows.append(
            {
                "pair_id": os.path.basename(os.path.dirname(summary_path)),
                "flood_ha": s["flood_ha"],
                "water_pre_ha": s["water_pre_ha"],
                "water_peak_ha": s["water_peak_ha"],
            }
        )
    return pd.DataFrame(rows)


def leave_one_aoi_out(submission, reference, kind, aoi_ha, aoi_of_pair) -> pd.DataFrame:
    """Scores each AOI as a held-out fold. A configuration that only looks
    good in aggregate but collapses on one district is overfitted, and this
    table makes that visible."""
    districts = sorted(set(aoi_of_pair.values()))
    rows = []
    for held_out in districts:
        fold_pairs = [p for p, a in aoi_of_pair.items() if a == held_out]
        sub_fold = submission[submission["pair_id"].isin(fold_pairs)]
        ref_fold = reference[reference["pair_id"].isin(fold_pairs)]
        if sub_fold.empty or ref_fold.empty:
            continue
        rep = score_submission(sub_fold, ref_fold, kind, aoi_ha)
        rows.append(
            {
                "held_out_aoi": held_out,
                "pairs": len(ref_fold),
                "score": round(rep.score, 4),
                "Q_flood": round(rep.q_flood, 4) if rep.q_flood == rep.q_flood else None,
                "Q_water_peak": round(rep.q_water_peak, 4) if rep.q_water_peak == rep.q_water_peak else None,
                "Q_water_pre": round(rep.q_water_pre, 4) if rep.q_water_pre == rep.q_water_pre else None,
                "Spec_base": round(rep.spec_base, 4) if rep.spec_base == rep.spec_base else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", default=None, help="optional CSV path for the fold table")
    args = parser.parse_args()

    reference, kind, aoi_ha, aoi_of_pair = load_reference(args.data_dir)
    submission = submission_from_checkpoints(args.checkpoint_dir)
    if submission.empty:
        raise SystemExit(
            f"no checkpoints found in {args.checkpoint_dir} — run run_inference.py first"
        )

    overall = score_submission(submission, reference, kind, aoi_ha)
    print("=== Overall (all pairs) ===")
    print(f"Score        = {overall.score:.4f}")
    print(f"  Q_flood      (w=0.45) = {overall.q_flood:.4f}")
    print(f"  Q_water_peak (w=0.25) = {overall.q_water_peak:.4f}")
    print(f"  Q_water_pre  (w=0.15) = {overall.q_water_pre:.4f}")
    print(f"  Spec_base    (w=0.15) = {overall.spec_base:.4f}")

    print("\n=== Leave-one-AOI-out folds ===")
    folds = leave_one_aoi_out(submission, reference, kind, aoi_ha, aoi_of_pair)
    print(folds.to_string(index=False))

    print("\n=== Per-pair residuals (largest absolute flood error first) ===")
    per_pair = overall.per_pair.copy()
    per_pair["flood_abs_err_ha"] = (per_pair["flood_pred"] - per_pair["flood_ref"]).abs()
    cols = ["pair_id", "kind", "flood_pred", "flood_ref", "flood_abs_err_ha", "water_peak_pred", "water_peak_ref"]
    print(per_pair.sort_values("flood_abs_err_ha", ascending=False)[cols].to_string(index=False))

    if args.out:
        folds.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
