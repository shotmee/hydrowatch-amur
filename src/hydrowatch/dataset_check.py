"""Dataset-completeness diagnostics.

Mirrors of `hydrowatch_amur/` (re-uploaded to Kaggle, copied to Drive,
zipped and re-extracted) can silently lose files, or gain extra nesting —
e.g. Kaggle's web uploader nests a dragged-in top-level folder under
itself (`rasters/rasters/...`), while files dropped straight into the
dataset root (`pairs.csv`) stay unnested. `hydrowatch.paths.resolve` looks
past that; this module uses it to report, per pair, exactly what is and
isn't findable — and hands back the *resolved* paths so callers don't have
to re-derive them (and re-fail) with a naive `os.path.join`.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

from .paths import try_resolve


@dataclass
class PairAvailability:
    pair_id: str
    rasters_dir: str | None
    aux_terrain_path: str | None
    s1_pre_path: str | None
    s1_peak_path: str | None
    reference_mask_path: str | None

    @property
    def ready_for_grid(self) -> bool:
        """Minimum needed to read the target grid + terrain layers."""
        return self.reference_mask_path is not None and self.aux_terrain_path is not None

    @property
    def ready_for_inference(self) -> bool:
        return self.ready_for_grid and self.s1_pre_path is not None and self.s1_peak_path is not None

    @property
    def missing(self) -> list[str]:
        problems = []
        if self.rasters_dir is None:
            problems.append("rasters_dir")
        if self.aux_terrain_path is None:
            problems.append("AUX_terrain_gsw.tif")
        if self.s1_pre_path is None:
            problems.append("S1_pre_*.json")
        if self.s1_peak_path is None:
            problems.append("S1_peak_*.json")
        if self.reference_mask_path is None:
            problems.append("reference_masks/*.tif")
        return problems


def check_pair(data_dir: str, row: pd.Series) -> PairAvailability:
    rasters_dir = try_resolve(data_dir, row["rasters_dir"])
    aux_path = None
    s1_pre_path = None
    s1_peak_path = None
    if rasters_dir is not None:
        aux_candidate = os.path.join(rasters_dir, "AUX_terrain_gsw.tif")
        aux_path = aux_candidate if os.path.isfile(aux_candidate) else None
        pre_matches = glob.glob(os.path.join(rasters_dir, "S1_pre_*.json"))
        s1_pre_path = pre_matches[0] if pre_matches else None
        peak_matches = glob.glob(os.path.join(rasters_dir, "S1_peak_*.json"))
        s1_peak_path = peak_matches[0] if peak_matches else None

    reference_mask_path = try_resolve(data_dir, row["reference_mask"])

    return PairAvailability(
        pair_id=row["pair_id"],
        rasters_dir=rasters_dir,
        aux_terrain_path=aux_path,
        s1_pre_path=s1_pre_path,
        s1_peak_path=s1_peak_path,
        reference_mask_path=reference_mask_path,
    )


def check_dataset_detailed(data_dir: str, pairs: pd.DataFrame) -> dict[str, PairAvailability]:
    """pair_id -> PairAvailability, resolved paths included. Callers that
    go on to actually run inference should reuse these paths rather than
    re-deriving (and re-failing) them with a naive `os.path.join`."""
    return {row["pair_id"]: check_pair(data_dir, row) for _, row in pairs.iterrows()}


def check_dataset(data_dir: str, pairs: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per pair with availability flags — run this right
    after pointing DATA_DIR at a (possibly partial or oddly-nested)
    dataset copy."""
    availabilities = check_dataset_detailed(data_dir, pairs)
    rows = [
        {
            "pair_id": avail.pair_id,
            "ready_for_inference": avail.ready_for_inference,
            "missing": ", ".join(avail.missing) if avail.missing else "",
        }
        for avail in availabilities.values()
    ]
    return pd.DataFrame(rows)
