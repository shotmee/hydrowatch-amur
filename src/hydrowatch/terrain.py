"""Terrain / hydrological context layers from AUX_terrain_gsw.tif.

Bands (fixed order, see hydrowatch_amur/README.md):
    slope, hand, occurrence, seasonality, max_extent, builtup
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .grid import Grid

BAND_NAMES = ["slope", "hand", "occurrence", "seasonality", "max_extent", "builtup"]


@dataclass
class Terrain:
    slope: np.ndarray
    hand: np.ndarray
    occurrence: np.ndarray
    seasonality: np.ndarray
    max_extent: np.ndarray
    builtup: np.ndarray

    @classmethod
    def load_native(cls, path: str) -> "Terrain":
        """Loads at AUX_terrain_gsw.tif's native resolution (30 m). Only
        useful for inspection/EDA — segmentation needs the 10 m pair grid,
        see `load_resampled`."""
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32)
        return cls(*[arr[i] for i in range(6)])

    @classmethod
    def load_resampled(cls, path: str, grid: Grid) -> "Terrain":
        """Reprojects every band from AUX_terrain_gsw.tif's native 30 m
        grid onto the pair's 10 m target grid. The two grids share an
        origin but are not an exact integer multiple of each other in
        every AOI, so this uses a real reprojection (bilinear) rather than
        a pixel-repeat upsample."""
        with rasterio.open(path) as ds:
            src_arr = ds.read().astype(np.float32)
            src_transform = ds.transform
            src_crs = ds.crs

        dst = np.empty((6, grid.height, grid.width), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.bilinear,
        )
        return cls(*[dst[i] for i in range(6)])

    def permanent_water(self, occurrence_thresh: float = 80.0) -> np.ndarray:
        """JRC GSW occurrence >= 80% — the competition's definition of `permanent`."""
        return self.occurrence >= occurrence_thresh

    def implausible_flood(
        self,
        slope_thresh_deg: float = 8.0,
        hand_thresh_m: float = 25.0,
        builtup_thresh: float = 0.5,
    ) -> np.ndarray:
        """Pixels where new open-water flooding is physically implausible.

        This is a *soft* hydrological filter, not a hard truth: HAND breaks
        down in deltas/flat floodplains and is inapplicable on coasts, below
        dams and for local pluvial ponding (see case brief, "HAND: где
        признак работает и где нет"). It is used only to suppress isolated
        false positives, never to grow the flood mask.
        """
        return (
            (self.slope > slope_thresh_deg)
            | (self.hand > hand_thresh_m)
            | (self.builtup >= builtup_thresh)
        )
