"""Target-grid utilities.

The competition fixes the output grid per pair: 10 m, EPSG:32652, 3000-4500
px per side. The case brief is explicit that this grid is *defined by* the
reference masks ("Сетку задают эталонные маски"), and the submission format
requires the predicted mask to match it exactly. `AUX_terrain_gsw.tif`
cannot substitute for it — it ships at the native 30 m Copernicus DEM
resolution, not 10 m (verified: e.g. blagoveshchensk/flood_2021_06_amur is
1493x1228 @ 30 m vs. the reference mask's 4479x3682 @ 10 m, and the two
grids don't even divide evenly, so a bilinear reprojection — see
``terrain.resample_to_grid`` — is required, not a naive 3x repeat).

We therefore read *only the raster header* (crs, transform, width, height)
from ``reference_masks/*.tif`` — never ``.read()`` on it, never open the
accompanying ``.json`` stats. Header geometry carries no information about
which pixels are water; it is the documented, sanctioned way to obtain the
grid, not a backdoor into the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import rasterio
from rasterio.transform import Affine


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int

    @property
    def pixel_area_ha(self) -> float:
        return abs(self.transform.a * self.transform.e) / 10_000.0

    @property
    def crs_transform(self) -> list[float]:
        """Six-element affine in the order GEE's ``crsTransform`` expects."""
        t = self.transform
        return [t.a, t.b, t.c, t.d, t.e, t.f]

    @property
    def dimensions(self) -> str:
        return f"{self.width}x{self.height}"

    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        x0, y0 = t.c, t.f
        x1 = x0 + t.a * self.width
        y1 = y0 + t.e * self.height
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def read_grid(reference_mask_path: str) -> Grid:
    """Header-only read: crs/transform/shape, never pixel values."""
    with rasterio.open(reference_mask_path) as ds:
        return Grid(crs=str(ds.crs), transform=ds.transform, width=ds.width, height=ds.height)
