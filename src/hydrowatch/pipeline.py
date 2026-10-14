"""Per-pair orchestration.

Deliberately split in two:

- ``segment_pair`` — pure numpy in, masks/areas out. No network, no GEE.
  This is what the unit tests exercise, and it is where nearly all of the
  scientific content of the solution lives.
- ``fetch_pair_arrays`` — the only function that talks to Earth Engine.
  Swap it out (mock it, replace it with a cached-tile reader, etc.)
  without touching the segmentation logic at all.

Every tunable lives in `config.Config`; nothing is hardcoded here.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from .change import compute_change
from .config import Config
from .fusion import fuse_evidence
from .grid import Grid, read_grid
from scipy.ndimage import binary_dilation

from .postprocess import (
    apply_hydro_filter,
    connectivity_filter,
    rasterize_hydrography,
    two_tier_connectivity_filter,
    remove_small_objects,
)
from .segmentation import (
    fuse_water_masks,
    optical_water_mask,
    optical_water_probability,
    sar_threshold,
    sar_water_probability,
    speckle_filter,
)
from .temporal import AoiThresholdPrior, regularize_threshold
from .terrain import Terrain

# Bumped whenever a change can alter a mask. Mask checkpoints carry a hash
# of (config + this string), so a re-run after an algorithm change cannot
# silently replay stale results — which is exactly what happened once.
PIPELINE_VERSION = "2026.07-prior-and-guard"


@dataclass
class DateArrays:
    """One observation date. `optical` is None when the pair has no usable
    Sentinel-2 scene for this window — the SAR-only path."""

    vv_db: np.ndarray
    optical: dict | None = None  # {"mndwi","ndwi","ndvi","aweish","valid"}


@dataclass
class PairMasks:
    water_pre: np.ndarray
    water_peak: np.ndarray
    flood: np.ndarray
    receded: np.ndarray
    grid: Grid
    terrain: Terrain
    diagnostics: dict = field(default_factory=dict)


def segment_date(
    date: DateArrays, cfg: Config, prior: "AoiThresholdPrior | None" = None
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Returns (water_mask, diagnostics, speckle_filtered_vv_db).

    The filtered VV is handed back because the bi-temporal backscatter-drop
    test in `change.compute_change` must run on the same smoothed signal the
    threshold was fitted to, not on the raw noisy input.

    `prior` is the district's threshold distribution measured on extra
    scenes; it only intervenes when this scene's own evidence is thin.
    """
    filtered = speckle_filter(date.vv_db, size=cfg.sar.speckle_filter_size)
    threshold, diag = sar_threshold(filtered, cfg.sar)

    if cfg.temporal.use_threshold_prior and prior is not None:
        threshold, prior_info = regularize_threshold(
            threshold,
            diag,
            prior,
            min_observations=cfg.temporal.min_observations,
            outlier_tolerance_db=cfg.temporal.outlier_tolerance_db,
            fallback_weight=cfg.temporal.fallback_weight,
            weak_evidence_tiles=cfg.temporal.weak_evidence_tiles,
            weak_evidence_weight=cfg.temporal.weak_evidence_weight,
        )
        diag.update(prior_info)

    sar_mask = filtered <= threshold

    if cfg.fusion.method in ("or", "consensus", "strict_consensus", "sar_only"):
        optical_mask = None
        optical_valid = None
        if date.optical and cfg.fusion.method != "sar_only":
            optical_mask = optical_water_mask(
                date.optical["mndwi"],
                date.optical["ndwi"],
                date.optical["ndvi"],
                valid_pixels=None,  # validity handled explicitly below
                mndwi_thresh=cfg.optical.mndwi_thresh,
                ndwi_thresh=cfg.optical.ndwi_thresh,
                ndvi_thresh=cfg.optical.ndvi_thresh,
            )
            optical_valid = date.optical.get("valid")

        if cfg.fusion.method == "sar_only":
            water = sar_mask
        elif cfg.fusion.method == "or":
            if optical_mask is not None and optical_valid is not None:
                optical_mask = optical_mask & optical_valid
            water = fuse_water_masks(sar_mask, optical_mask)
        elif cfg.fusion.method == "consensus":
            # Both sensors must agree wherever the optical scene actually
            # sees the ground. Where it does not (no scene at all, or
            # cloud/shadow at this pixel) the SAR verdict stands alone —
            # the same precedence the reference labelling used.
            if optical_mask is None:
                water = sar_mask
            else:
                seen = optical_valid if optical_valid is not None else np.ones_like(sar_mask)
                water = np.where(seen, sar_mask & optical_mask, sar_mask)
        else:
            # strict_consensus: on a date that HAS an optical scene, a pixel
            # is water only where the optics actually saw the ground and
            # agreed. Cloud means "not confirmed", not "trust the radar".
            # This is the harsher reading of the reference recipe, and it is
            # what the near-empty reference masks on the cloudy 2021 peak
            # dates look like (konstantinovka: 30 ha of water_pre in a
            # district whose permanent channel alone is 5436 ha).
            if optical_mask is None:
                water = sar_mask
            else:
                seen = optical_valid if optical_valid is not None else np.ones_like(sar_mask)
                water = sar_mask & optical_mask & seen
        diag["fusion"] = cfg.fusion.method
        diag["optical_available"] = optical_mask is not None
        if optical_valid is not None:
            diag["optical_valid_share"] = round(float(optical_valid.mean()), 4)
        return water, diag, filtered

    sar_prob = sar_water_probability(filtered, threshold, cfg.fusion.sar_soft_margin_db)
    opt_prob, opt_valid = optical_water_probability(
        date.optical, cfg.optical, cfg.fusion.optical_soft_margin
    )
    water, _belief = fuse_evidence(sar_prob, opt_prob, opt_valid, cfg.fusion)

    diag["fusion"] = "dempster_shafer"
    diag["optical_available"] = opt_prob is not None
    if opt_valid is not None:
        diag["optical_valid_share"] = round(float(opt_valid.mean()), 4)
    return water, diag, filtered


def segment_pair(
    pair_id: str,
    grid_source_path: str,
    aux_terrain_path: str,
    hydrography_path: str,
    pre: DateArrays,
    peak: DateArrays,
    cfg: Config | None = None,
    prior: "AoiThresholdPrior | None" = None,
) -> PairMasks:
    """`grid_source_path` is a reference_masks/*.tif path — its raster
    *header* (crs/transform/shape) defines the 10 m grid every mask must
    share; its pixel values are never read (see grid.py docstring)."""
    cfg = cfg or Config()
    pp = cfg.postprocess

    grid = read_grid(grid_source_path)
    terrain = Terrain.load_resampled(aux_terrain_path, grid)
    permanent = terrain.permanent_water(pp.permanent_occurrence_pct)
    implausible = terrain.implausible_flood(
        slope_thresh_deg=pp.slope_max_deg,
        hand_thresh_m=pp.hand_max_m,
        builtup_thresh=pp.builtup_thresh,
    )

    water_pre, diag_pre, vv_pre_filt = segment_date(pre, cfg, prior)
    water_peak, diag_peak, vv_peak_filt = segment_date(peak, cfg, prior)

    if pp.restrict_water_pre_to_max_extent:
        envelope = terrain.max_extent > 0.5
        if pp.max_extent_dilation_px > 0:
            envelope = binary_dilation(envelope, iterations=pp.max_extent_dilation_px)
        diag_pre["water_pre_before_envelope_px"] = int(water_pre.sum())
        water_pre = water_pre & envelope
        diag_pre["water_pre_after_envelope_px"] = int(water_pre.sum())

    flood, receded = compute_change(
        water_pre, water_peak, permanent, vv_pre_filt, vv_peak_filt, cfg.change
    )

    flood = apply_hydro_filter(flood, implausible)
    flood = remove_small_objects(flood, pp.min_mapping_unit_px)
    if pp.connectivity_filter:
        strong_seed = (
            rasterize_hydrography(hydrography_path, grid, buffer_px=pp.hydrography_buffer_px)
            | permanent
        )
        if pp.connectivity_use_max_extent:
            # JRC "maximum observed water extent" is a weaker anchor than a
            # mapped channel: it recovers real flood that never touches an
            # OSM centreline, but on a dry control pair it also keeps
            # scattered noise alive. Hence the second tier — see
            # `two_tier_connectivity_filter`.
            weak_seed = (terrain.max_extent > 0.5) & ~strong_seed
            flood = two_tier_connectivity_filter(
                flood, strong_seed, weak_seed, pp.max_extent_only_min_px
            )
        else:
            flood = connectivity_filter(flood, strong_seed)

    water_pre = remove_small_objects(water_pre, pp.min_mapping_unit_px)
    water_peak = remove_small_objects(water_peak, pp.min_mapping_unit_px)
    receded = remove_small_objects(receded, pp.min_mapping_unit_px)

    return PairMasks(
        water_pre=water_pre,
        water_peak=water_peak,
        flood=flood,
        receded=receded,
        grid=grid,
        terrain=terrain,
        diagnostics={"pre": diag_pre, "peak": diag_peak},
    )


def quantize_like_cache(date: DateArrays) -> DateArrays:
    """Round freshly fetched imagery to the precision the caches store.

    Both the array cache and the offline bundle keep float16. Without this,
    the *first* run segments the raw float32 arrays while every later run —
    and every reviewer running from the bundle — segments the float16 round
    trip, so the same command gives areas that differ by ~0.001%. That is
    far inside the 0.005 metric tolerance, but "voluntarily not bit-exact"
    is a strange thing to claim under a reproducibility criterion. Applying
    the rounding at the single point where imagery enters the pipeline makes
    run 1, run 2 and the offline reproduction identical by construction.
    """
    vv = date.vv_db.astype(np.float16).astype(np.float32)
    optical = None
    if date.optical:
        optical = {}
        for key, value in date.optical.items():
            if value is None:
                optical[key] = None
            elif key == "valid":
                optical[key] = value  # already boolean
            else:
                optical[key] = value.astype(np.float16).astype(np.float32)
    return DateArrays(vv_db=vv, optical=optical)


def find_passport(rasters_dir: str, prefix: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(rasters_dir, f"{prefix}_*.json")))
    return matches[0] if matches else None


def fetch_pair_arrays(
    rasters_dir: str,
    grid: Grid,
    window: str,
    cfg: Config | None = None,
    progress=None,
) -> DateArrays:
    """Live Earth Engine fetch for one date (`window` = "pre" | "peak").

    The grid alone determines both the scene-selection footprint and the
    output pixels — see `gee_export.grid_geometry`. Requires
    `gee_export.init_ee()` first.
    """
    from . import gee_export as ge

    cfg = cfg or Config()

    s1_json = find_passport(rasters_dir, f"S1_{window}")
    if s1_json is None:
        raise FileNotFoundError(f"no S1 passport for window={window} in {rasters_dir}")

    geom = ge.grid_geometry(grid)

    s1_passport = ge.ScenePassport.from_json(s1_json)
    s1_image = ge.build_s1_image(s1_passport, geom)
    vv_db = ge.fetch_as_arrays(s1_image, grid, ["VV"], progress=progress)["VV"]

    optical = None
    s2_json = find_passport(rasters_dir, f"SENTINEL2_{window}")
    if s2_json is not None:
        s2_passport = ge.ScenePassport.from_json(s2_json)
        s2_image = ge.build_s2_image(s2_passport, geom, with_aweish=cfg.optical.use_aweish)
        bands = ["NDWI", "MNDWI", "NDVI", "valid"]
        if cfg.optical.use_aweish:
            bands.append("AWEISH")
        arrays = ge.fetch_as_arrays(s2_image, grid, bands, progress=progress)
        optical = {
            "ndwi": arrays["NDWI"],
            "mndwi": arrays["MNDWI"],
            "ndvi": arrays["NDVI"],
            "valid": arrays["valid"] > 0.5,
            "aweish": arrays.get("AWEISH"),
        }

    # Единственная точка входа снимков в пайплайн — здесь же и приводим их к
    # точности кэша, иначе первый прогон не совпадёт с повторным.
    return quantize_like_cache(DateArrays(vv_db=vv_db, optical=optical))
