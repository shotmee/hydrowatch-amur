"""Module 2 — spatio-temporal change between the "pre" and "peak" masks.

Class definitions follow the case brief's «Что считается водой» verbatim:
    flood    = water_peak & ~water_pre & ~permanent
    receded  = water_pre & ~water_peak & ~permanent
`permanent` (JRC GSW occurrence >= 80%) is excluded from both so an
already-wet river channel never counts as "new" flooding.

On top of that set logic the brief documents one more condition used when
the reference itself was built: «требование падения обратного рассеяния на
пике не менее чем на 3 дБ». Independent per-date thresholding cannot see
that — a pixel can land just either side of two separately-fitted
thresholds and be declared "new water" on noise alone. Requiring a real,
physical drop in sigma0 between the two dates is a bi-temporal consistency
check, and it is the cheapest available suppressor of the false positives
that `Spec_base` punishes on the dry baseline pairs.
"""
from __future__ import annotations

import numpy as np

from .config import ChangeConfig


def backscatter_drop_db(vv_pre_db: np.ndarray, vv_peak_db: np.ndarray) -> np.ndarray:
    """Positive where backscatter fell between the two dates.

    In dB the log-ratio of the linear-power images is just their difference,
    so this is the classic SAR log-ratio change image without extra work.
    """
    return vv_pre_db - vv_peak_db


def compute_change(
    water_pre: np.ndarray,
    water_peak: np.ndarray,
    permanent: np.ndarray,
    vv_pre_db: np.ndarray | None = None,
    vv_peak_db: np.ndarray | None = None,
    cfg: ChangeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (flood, receded)."""
    cfg = cfg or ChangeConfig()

    flood = water_peak & ~water_pre & ~permanent
    receded = water_pre & ~water_peak & ~permanent

    if cfg.require_backscatter_drop and vv_pre_db is not None and vv_peak_db is not None:
        drop = backscatter_drop_db(vv_pre_db, vv_peak_db)
        flood &= drop >= cfg.min_drop_db
        # Symmetrically, water that receded should show backscatter *rising*.
        receded &= drop <= -cfg.min_drop_db

    return flood, receded
