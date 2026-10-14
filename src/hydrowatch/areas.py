"""Pixel-count to hectare conversion — single source of truth so the
number written to submission.csv always matches the number you'd get by
re-counting the saved GeoTIFF mask (the <=2% consistency check in the
grading rubric)."""
from __future__ import annotations

import numpy as np

from .grid import Grid


def mask_area_ha(mask: np.ndarray, grid: Grid) -> float:
    return float(np.count_nonzero(mask)) * grid.pixel_area_ha
