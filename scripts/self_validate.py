#!/usr/bin/env python
"""Dev-only: score a submission.csv against the shipped reference masks
using the organizers' exact formula (hydrowatch.metrics).

This never feeds back into inference — it's purely so the team can see the
Score locally before uploading, per the case brief's "Маски выданы для
обучения, подбора порогов и самопроверки."

Usage:
    python self_validate.py --data-dir /path/to/hydrowatch_amur \
                             --submission /path/to/outputs/submission.csv
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


def load_reference_table(data_dir: str) -> tuple[pd.DataFrame, dict, dict]:
    rows = []
    kind = {}
    aoi_ha = {}
    pairs = pd.read_csv(os.path.join(data_dir, "pairs.csv"))
    pairs = pairs.set_index("pair_id")

    # recursive: tolerates the nested-folder quirk some dataset re-uploads
    # introduce (e.g. reference_masks/reference_masks/... on Kaggle, see
    # hydrowatch.paths docstring)
    json_paths = sorted(glob.glob(os.path.join(data_dir, "**", "reference_*.json"), recursive=True))
    for meta_path in json_paths:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        pair_id = meta["pair_id"]
        stats = meta["stats"]
        rows.append(
            {
                "pair_id": pair_id,
                "flood_ha": stats["flood_ha"],
                "water_pre_ha": stats["water_pre_ha"],
                "water_peak_ha": stats["water_peak_ha"],
            }
        )
        kind[pair_id] = pairs.loc[pair_id, "event_kind"]
        aoi_ha[pair_id] = pairs.loc[pair_id, "aoi_km2"] * 100.0

    return pd.DataFrame(rows), kind, aoi_ha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    reference, kind, aoi_ha = load_reference_table(args.data_dir)
    submission = pd.read_csv(args.submission)

    report = score_submission(submission, reference, kind, aoi_ha)

    print(f"Score        = {report.score:.4f}")
    print(f"  Q_flood      (w=0.45) = {report.q_flood:.4f}")
    print(f"  Q_water_peak (w=0.25) = {report.q_water_peak:.4f}")
    print(f"  Q_water_pre  (w=0.15) = {report.q_water_pre:.4f}")
    print(f"  Spec_base    (w=0.15) = {report.spec_base:.4f}")
    print()
    print(report.per_pair.to_string(index=False))


if __name__ == "__main__":
    main()
