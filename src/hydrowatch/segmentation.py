"""Single-date water segmentation from SAR and optical bands.

**Why the threshold is not a plain global Otsu.** The case brief states the
core difficulty outright: «доля затопления составляет доли процента от
площади района». A global histogram over such a scene is effectively
unimodal — the water mode is buried in the land mode's tail, and Otsu, which
maximises between-class variance over the *whole* image, drifts into the
noise. The split-based approach (Martinis et al. 2009; Chini et al. 2017,
"A Hierarchical Split-Based Approach for Parametric Thresholding of SAR
Images"; the same idea ships as `bmaxOtsu` in Servir-Mekong's HYDRAFloods,
MIT) fixes this: tile the scene, keep only tiles whose own histogram is
genuinely bimodal (i.e. that straddle a shoreline and therefore contain both
classes), and fit the threshold on those tiles' pooled pixels.

Kittler-Illingworth minimum-error thresholding is offered alongside Otsu
because Otsu is biased when the two classes differ wildly in prior
probability — exactly our regime. Blending the two (`sar.ki_weight`) is a
cheap ensemble.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .config import OpticalConfig, SarConfig


def _median_2d(values: np.ndarray, size: int) -> np.ndarray:
    """Square median filter, using OpenCV when it can and SciPy otherwise.

    This is the single hottest operation in the pipeline: profiling one
    4479x3682 pair put 12.8 s of a 16.1 s segmentation inside SciPy's rank
    filter — 80% of the run. OpenCV's `medianBlur` computes the **identical**
    result for float32 at kernel 3 or 5 (verified: max |difference| = 0 over
    the interior on 16.5 Mpx) and is ~82x faster.

    Only the border differs, because the two libraries pad differently, so
    SciPy remains the reference implementation and is used whenever OpenCV
    cannot handle the request.
    """
    if size in (3, 5) and values.dtype == np.float32:
        try:
            import cv2

            return cv2.medianBlur(values, size)
        except ImportError:
            pass
    return median_filter(values, size=size)


def speckle_filter(band_db: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter, NaN-aware.

    A median is rank-based, so applying it in dB or in linear intensity
    gives the same answer — unlike a *mean*, which must be taken on
    intensity because averaging decibels is a geometric mean of power and is
    biased low. NaNs (masked pixels outside the scene footprint) are held
    out during filtering and restored afterwards, otherwise the filter would
    smear them across the whole neighbourhood.
    """
    values = band_db if band_db.dtype == np.float32 else band_db.astype(np.float32)
    nodata = ~np.isfinite(values)
    if not nodata.any():
        return _median_2d(values, size)

    filled = values.copy()
    valid = values[~nodata]
    filled[nodata] = float(np.median(valid)) if valid.size else 0.0
    out = _median_2d(filled, size)
    out[nodata] = np.nan
    return out


def histogram_stats(values: np.ndarray, bins: int = 128) -> dict | None:
    """Otsu threshold plus the two formal bimodality tests.

    Follows the organizers' own teaching notebook (method from Cao et al.,
    Water 11(4):786, 2019):

    - **BC** = (skew² + 1) / kurtosis — above 5/9 ≈ 0.555 indicates two modes.
    - **Ashman D** = √2·|μ₁−μ₂| / √(σ₁²+σ₂²) — mode separation relative to
      mode width; above ~3 the modes are genuinely distinct.
    - **eta** = σ²_B / σ²_T — reported for diagnostics only. It is *not* a
      usable gate: on a land-only tile it reads ~0.64 while the threshold it
      produces (-8 dB) is meaningless.
    """
    values = values[np.isfinite(values)]
    if values.size < 500:
        return None

    hist, centers = _histogram(values, bins, None)
    n = hist.sum()
    if n <= 0:
        return None

    mu = (centers * hist).sum() / n
    var = (hist * (centers - mu) ** 2).sum() / n
    sd = np.sqrt(max(var, 1e-12))
    skew = (hist * ((centers - mu) / sd) ** 3).sum() / n
    kurt = (hist * ((centers - mu) / sd) ** 4).sum() / n
    bc = float((skew**2 + 1) / kurt) if kurt > 0 else 0.0

    threshold, eta = otsu_from_histogram(hist, centers)

    left = centers <= threshold

    def _mean_sd(sel):
        w, x = hist[sel], centers[sel]
        total = w.sum()
        if total <= 0:
            return 0.0, 0.0
        m = (x * w).sum() / total
        return m, float(np.sqrt(max((w * (x - m) ** 2).sum() / total, 1e-12)))

    m_water, s_water = _mean_sd(left)
    m_land, s_land = _mean_sd(~left)
    ashman_d = float(
        np.sqrt(2) * abs(m_land - m_water) / np.sqrt(s_water**2 + s_land**2 + 1e-12)
    )

    return {
        "threshold": float(threshold),
        "bc": bc,
        "ashman_d": ashman_d,
        "eta": float(eta),
        "water_fraction": float(hist[left].sum() / n),
        "dark_mode_mean": float(m_water),
        "n": float(n),
    }


def _histogram(values: np.ndarray, bins: int, value_range: tuple[float, float] | None):
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return hist.astype(np.float64), centers


def otsu_from_histogram(hist: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
    """Returns (threshold, bimodality) where bimodality is the share of total
    variance explained by the best split — Otsu's own objective, normalised.
    A value near 1 means the histogram really is two separated modes; near 0
    means a single blob and the threshold is meaningless."""
    total = hist.sum()
    if total <= 0:
        return float("nan"), 0.0

    sum_all = (hist * centers).sum()
    weight_bg = np.cumsum(hist)
    sum_bg = np.cumsum(hist * centers)
    weight_fg = total - weight_bg

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(weight_bg > 0, sum_bg / weight_bg, 0.0)
        mean_fg = np.where(weight_fg > 0, (sum_all - sum_bg) / weight_fg, 0.0)
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    best = int(np.nanargmax(between))
    mean_all = sum_all / total
    total_variance = (hist * (centers - mean_all) ** 2).sum()
    bimodality = float(between[best] / (total * total_variance)) if total_variance > 0 else 0.0
    return float(centers[best]), bimodality


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("otsu_threshold: empty input")
    hist, centers = _histogram(values, bins, None)
    threshold, _ = otsu_from_histogram(hist, centers)
    return threshold


def kittler_illingworth_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Minimum-error thresholding (Kittler & Illingworth, 1986).

    Models each class as a Gaussian and minimises the expected
    misclassification cost, rather than maximising between-class variance.
    Less biased than Otsu when one class occupies a tiny fraction of the
    scene — the documented failure mode for flood extent.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("kittler_illingworth_threshold: empty input")
    hist, centers = _histogram(values, bins, None)
    total = hist.sum()
    if total <= 0:
        return float("nan")
    p = hist / total

    w1 = np.cumsum(p)
    w2 = 1.0 - w1
    m1 = np.cumsum(p * centers)
    m2 = m1[-1] - m1

    with np.errstate(divide="ignore", invalid="ignore"):
        mu1 = np.where(w1 > 0, m1 / w1, 0.0)
        mu2 = np.where(w2 > 0, m2 / w2, 0.0)
        s1 = np.cumsum(p * centers**2)
        s2 = s1[-1] - s1
        var1 = np.where(w1 > 0, s1 / w1 - mu1**2, 0.0)
        var2 = np.where(w2 > 0, s2 / w2 - mu2**2, 0.0)
        var1 = np.clip(var1, 1e-12, None)
        var2 = np.clip(var2, 1e-12, None)
        # Kittler-Illingworth criterion J(T), minimised.
        j = (
            1.0
            + 2.0 * (w1 * np.log(np.sqrt(var1)) + w2 * np.log(np.sqrt(var2)))
            - 2.0 * (w1 * np.log(np.clip(w1, 1e-12, None)) + w2 * np.log(np.clip(w2, 1e-12, None)))
        )

    valid = (w1 > 1e-6) & (w2 > 1e-6) & np.isfinite(j)
    if not valid.any():
        return otsu_threshold(values, bins=bins)
    j_masked = np.where(valid, j, np.inf)
    return float(centers[int(np.argmin(j_masked))])


def split_based_threshold(band_db: np.ndarray, cfg: SarConfig) -> tuple[float, dict]:
    """Per-tile threshold selection, then the median of the survivors.

    Three filters decide whether a tile gets a vote, and each one exists
    because of a measured failure mode:

    1. **Bimodality** — the tile must actually straddle a shoreline. A tile
       of uniform land has no valley to find and would contribute noise.
    2. **Plausibility** — the tile's own threshold must fall inside the
       open-water sigma0 window. A tile split between bright and
       medium-bright land produces a threshold far above it.
    3. **Dark-mode brightness** — the tile's darker class must be as dark as
       real open water. Without this, the brief's «гладкая сухая
       поверхность» (asphalt, runway) passes filters 1-2 and pulls the
       threshold up; measured on a synthetic dry scene, that invents 6.2% of
       the AOI as water, which alone wipes out Spec_base (tolerance 0.5%).

    Taking the **median** rather than pooling histograms makes the estimate
    robust to a handful of surviving bad tiles, and to the incidence-angle
    drift across the swath that biases any single global fit.
    """
    lo, hi = cfg.valid_range_db
    h, w = band_db.shape
    tile = cfg.tile_px
    bins = cfg.histogram_bins
    frac_lo, frac_hi = cfg.tile_water_fraction_range

    candidates: list[dict] = []  # every tile with usable statistics
    rejected = {
        "not_bimodal": 0,
        "degenerate_fraction": 0,
        "implausible_threshold": 0,
        "dark_mode_too_bright": 0,
    }

    for row in range(0, h, tile):
        for col in range(0, w, tile):
            patch = band_db[row : row + tile, col : col + tile].ravel()
            stats = histogram_stats(patch, bins=bins)
            if stats is None:
                continue

            t_otsu = stats["threshold"]
            stats["threshold_blended"] = float(t_otsu)

            passes = True
            if stats["bc"] < cfg.bimodality_coefficient_min or stats["ashman_d"] < cfg.ashman_d_min:
                rejected["not_bimodal"] += 1
                passes = False
            elif not (frac_lo <= stats["water_fraction"] <= frac_hi):
                rejected["degenerate_fraction"] += 1
                passes = False
            elif not (lo <= t_otsu <= hi):
                rejected["implausible_threshold"] += 1
                passes = False
            elif stats["dark_mode_mean"] > cfg.water_dark_mode_max_db:
                rejected["dark_mode_too_bright"] += 1
                passes = False
            stats["passes"] = passes
            if passes and cfg.ki_weight > 0:
                # Kittler-Illingworth is the expensive part, so it runs only
                # on tiles that already survived the gates — same result,
                # a fraction of the work, which is what makes small tiles
                # (and therefore narrow rivers) affordable.
                t_ki = kittler_illingworth_threshold(patch[np.isfinite(patch)], bins=bins)
                if np.isfinite(t_ki) and lo <= t_ki <= hi:
                    stats["threshold_blended"] = float(
                        (1.0 - cfg.ki_weight) * t_otsu + cfg.ki_weight * t_ki
                    )
            candidates.append(stats)

    accepted = [c for c in candidates if c["passes"]]
    diagnostics = {
        "method": "split",
        "tiles_scored": len(candidates),
        "tiles_kept": len(accepted),
        "best_ashman_d": round(max((c["ashman_d"] for c in candidates), default=0.0), 3),
        "best_bc": round(max((c["bc"] for c in candidates), default=0.0), 3),
        "rejected": rejected,
    }

    if len(accepted) < cfg.min_bimodal_tiles:
        # Relax rather than declare the scene waterless. Returning the dry
        # end of the window here is catastrophic on a real river AOI: in the
        # first scored run it produced 0 ha against an 8910 ha reference for
        # poyarkovo, zeroing both Q components for that pair. The organizers'
        # own notebook relaxes the same way — take the tiles with the widest
        # mode separation, but keep the physical plausibility clamp.
        # Only the *statistical* gates (BC, Ashman D, fraction) are relaxed.
        # The *physical* ones are not: the threshold must still sit in the
        # open-water window and the dark mode must still be as dark as
        # water. Relaxing those too would trade the poyarkovo failure for a
        # worse one — inventing ~6% of a dry control AOI as water, which
        # alone zeroes Spec_base.
        plausible = [
            c
            for c in candidates
            if lo <= c["threshold_blended"] <= hi
            and c["dark_mode_mean"] <= cfg.water_dark_mode_max_db
        ]
        if not plausible:
            diagnostics["fallback"] = "no_physically_plausible_tiles"
            diagnostics["threshold_db"] = float(lo)
            return float(lo), diagnostics
        accepted = sorted(plausible, key=lambda c: c["ashman_d"], reverse=True)[
            : cfg.relaxed_fallback_tiles
        ]
        diagnostics["fallback"] = "relaxed_top_ashman_d"

    thresholds = [c["threshold_blended"] for c in accepted]
    threshold = float(np.clip(np.median(thresholds), lo, hi))
    diagnostics["threshold_db"] = round(threshold, 3)
    diagnostics["tiles_used"] = len(thresholds)
    diagnostics["tile_threshold_spread_db"] = round(
        float(np.percentile(thresholds, 75) - np.percentile(thresholds, 25)), 3
    ) if len(thresholds) > 1 else 0.0
    return threshold, diagnostics


def sar_threshold(band_db: np.ndarray, cfg: SarConfig) -> tuple[float, dict]:
    lo, hi = cfg.valid_range_db
    finite = band_db[np.isfinite(band_db)]

    if cfg.threshold_method == "split":
        return split_based_threshold(band_db, cfg)

    if cfg.threshold_method == "global_in_range":
        # v0 behaviour, retained so the ablation table can quantify it.
        sample = finite[(finite >= lo) & (finite <= hi)]
        if sample.size < 100:
            return float(lo), {"method": "global_in_range", "fallback": "too_few_pixels"}
        threshold = float(np.clip(otsu_threshold(sample), lo, hi))
        return threshold, {"method": "global_in_range", "threshold_db": threshold}

    if finite.size < 100:
        return float(lo), {"method": "global", "fallback": "too_few_pixels"}
    threshold = float(np.clip(otsu_threshold(finite), lo, hi))
    return threshold, {"method": "global", "threshold_db": threshold}


def sar_water_probability(
    vv_db: np.ndarray, threshold_db: float, soft_margin_db: float
) -> np.ndarray:
    """Soft membership in [0,1] instead of a hard cut.

    Dempster-Shafer fusion needs a graded confidence: a pixel 6 dB below the
    threshold is unambiguously water, one 0.2 dB below it is a coin flip.
    A logistic ramp of width `soft_margin_db` around the threshold encodes
    exactly that, and collapses back to the hard mask at margin -> 0.
    """
    if soft_margin_db <= 0:
        return (vv_db <= threshold_db).astype(np.float32)
    z = (threshold_db - vv_db) / soft_margin_db
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def sar_water_mask(
    vv_db: np.ndarray,
    cfg: SarConfig | None = None,
    return_diagnostics: bool = False,
):
    """Water = calm, specular, low-backscatter surface (case brief, «Как
    радиолокатор видит воду»)."""
    cfg = cfg or SarConfig()
    filtered = speckle_filter(vv_db, size=cfg.speckle_filter_size)
    threshold, diagnostics = sar_threshold(filtered, cfg)
    mask = filtered <= threshold
    if return_diagnostics:
        return mask, diagnostics
    return mask


def aweish(b2: np.ndarray, b3: np.ndarray, b8: np.ndarray, b11: np.ndarray, b12: np.ndarray):
    """Automated Water Extraction Index, shadow variant (Feyisa et al. 2014).

    Included specifically for the failure mode the brief calls out: turbid
    flood water raises NIR reflectance and drags NDWI to zero or below
    exactly when the service is most needed. AWEIsh leans on SWIR and the
    blue band instead, so it degrades far more gracefully.
    """
    return b2 + 2.5 * b3 - 1.5 * (b8 + b11) - 0.25 * b12


def optical_water_probability(
    optical: dict,
    cfg: OpticalConfig,
    soft_margin: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (probability, valid_mask) or (None, None) when this date has
    no usable optical scene at all — the SAR-only case the rubric asks about
    explicitly."""
    if not optical:
        return None, None

    mndwi = optical.get("mndwi")
    if mndwi is None:
        return None, None

    threshold = cfg.mndwi_thresh
    if cfg.adaptive_threshold:
        valid_vals = mndwi[np.isfinite(mndwi)]
        if valid_vals.size >= 1000:
            candidate = otsu_threshold(valid_vals)
            # Keep the adaptive cut anchored near the physical zero-crossing;
            # a scene with no water at all would otherwise split noise.
            if -0.3 <= candidate <= 0.6:
                threshold = float(candidate)

    if soft_margin > 0:
        z = (mndwi - threshold) / soft_margin
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    else:
        prob = (mndwi > threshold).astype(np.float32)

    # Vegetation veto: a high NDVI pixel is not open water regardless of the
    # water index (the brief excludes flooded vegetation from the target).
    ndvi = optical.get("ndvi")
    if ndvi is not None:
        prob = np.where(ndvi > cfg.ndvi_thresh, prob * 0.25, prob)

    if cfg.use_aweish and optical.get("aweish") is not None:
        agrees = optical["aweish"] > cfg.aweish_thresh
        # AWEIsh acts as a corroborating vote for turbid water that MNDWI
        # alone would miss, never as a veto on its own.
        prob = np.clip(np.where(agrees, np.maximum(prob, 0.6), prob), 0.0, 1.0)

    valid = optical.get("valid")
    valid_mask = valid if valid is not None else np.ones(mndwi.shape, dtype=bool)
    return prob.astype(np.float32), valid_mask


def optical_water_mask(
    mndwi: np.ndarray,
    ndwi: np.ndarray,
    ndvi: np.ndarray,
    valid_pixels: np.ndarray | None = None,
    mndwi_thresh: float = 0.1,
    ndwi_thresh: float = 0.15,
    ndvi_thresh: float = 0.2,
) -> np.ndarray:
    """Fixed-threshold optical mask (the v0 rule, kept for ablations)."""
    mask = (mndwi > mndwi_thresh) & (ndwi > ndwi_thresh) & (ndvi <= ndvi_thresh)
    if valid_pixels is not None:
        mask &= valid_pixels
    return mask


def fuse_water_masks(sar_mask: np.ndarray, optical_mask: np.ndarray | None) -> np.ndarray:
    """Plain union (the v0 rule, kept for ablations)."""
    if optical_mask is None:
        return sar_mask
    return sar_mask | optical_mask
