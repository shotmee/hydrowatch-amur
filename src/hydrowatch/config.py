"""Single source of truth for every threshold in the pipeline.

The rubric explicitly penalises thresholds hardcoded at their point of use
(«Пороги обратного рассеяния и спектральных индексов не зашиты числами по
месту»), and calibration (scripts/calibrate.py) needs to sweep them without
editing code. Every number below is therefore named, documented with its
provenance, and overridable from a YAML file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class SarConfig:
    # Physically plausible open-water sigma0 range. The case brief states the
    # reference labelling clamped Otsu to exactly this window. NOTE: this is
    # a clamp on the resulting *threshold*, never a filter on the pixels the
    # histogram is fitted to — fitting inside it removes the land mode and
    # biases the split (measured: IoU 0.59 vs 1.00 on a 10%-water scene).
    valid_range_db: tuple[float, float] = (-22.0, -12.0)
    speckle_filter_size: int = 5
    # "split" = per-tile Otsu + median (Chini et al. 2017 / Martinis et al.
    # 2009 family); "global" = one fit over the whole scene;
    # "global_in_range" = the biased v0 behaviour, kept for ablations.
    threshold_method: str = "split"
    # Tile side (px). Small enough that incidence-angle drift across the
    # swath is negligible inside one tile, large enough for a stable
    # histogram (the organizers' teaching notebook uses a 6x6 grid, i.e.
    # ~500-750 px tiles, on the same AOIs).
    # Measured on synthetic tiles with realistic gamma speckle: at 512 px
    # (5.1 km) a 250 m river is only 4.9% of the tile and gets rejected; at
    # 256 px it is 9.8% and passes. Narrow channels are exactly what the
    # flood pairs hinge on, so the tile must be small enough to see them.
    tile_px: int = 256
    # --- Bimodality criteria (Cao et al., Water 11(4):786, 2019) ---
    # A tile only votes if BOTH formal tests pass. Otsu's own eta
    # (between-class / total variance) is deliberately NOT used as the gate:
    # on pure land it reads ~0.64 while returning a nonsense -8 dB
    # threshold, which is exactly how a scene with no water still produces
    # a confident, wrong answer.
    bimodality_coefficient_min: float = 0.555  # (skew^2 + 1) / kurtosis, > 5/9
    ashman_d_min: float = 3.0  # sqrt(2)|mu1-mu2| / sqrt(s1^2+s2^2)
    # Degenerate splits (a tile that is ~all water or ~all land) carry no
    # information about where the boundary sits.
    # The floor used to be 0.05, which rejected a 250 m river by a hair
    # (fraction 0.049). Pure land is rejected by the *physical* gates below
    # (threshold range and dark-mode brightness), not by this one — measured:
    # a land-only tile passes BC=0.62 and AshD=4.4 but fails both physical
    # tests. So this gate can be loose without letting land through.
    tile_water_fraction_range: tuple[float, float] = (0.01, 0.95)
    # Physical veto: the dark mode of a usable tile must actually be as dark
    # as open water. Guards against a smooth dry surface (asphalt, runway —
    # the brief's «ложная вода») which can be genuinely bimodal against
    # rough land yet is not water.
    water_dark_mode_max_db: float = -16.0
    min_dark_pixels: int = 50
    # Never fit on fewer than this many qualifying tiles before relaxing.
    min_bimodal_tiles: int = 2
    # Fallback when no tile passes: use the N tiles with the widest mode
    # separation instead of declaring the scene waterless. Returning "no
    # water" here cost an entire AOI (poyarkovo: 0 ha against 8910 ha
    # reference) in the first scored run.
    relaxed_fallback_tiles: int = 3
    # Blend Otsu with Kittler-Illingworth per tile (0 = pure Otsu, 1 = pure KI).
    ki_weight: float = 0.5
    histogram_bins: int = 128


@dataclass
class OpticalConfig:
    # Fixed thresholds mirror the reference-labelling recipe in the brief.
    mndwi_thresh: float = 0.1
    ndwi_thresh: float = 0.15
    ndvi_thresh: float = 0.2
    aweish_thresh: float = 0.0
    # Re-fit the MNDWI cut per scene with Otsu instead of using the fixed value.
    adaptive_threshold: bool = True
    # AWEIsh is the turbid-flood-water safeguard: suspended sediment lifts NIR
    # and pushes NDWI negative exactly during a flood peak (case brief,
    # "Как оптический сенсор видит воду").
    use_aweish: bool = True


@dataclass
class FusionConfig:
    # "ds"        = Dempster-Shafer (graded evidence, ignorance modelled)
    # "consensus" = SAR AND optical where optical is valid, SAR alone where
    #               it is not. This is how the *reference* was built
    #               («приоритет консенсуса радиолокационной и оптической
    #               ветвей»), and the metric scores agreement with it.
    # "strict_consensus" = same, but cloud means "not confirmed" rather than
    #               "trust the radar" — matches the near-empty reference
    #               masks on the cloudy 2021 peak dates.
    # "sar_only"  = ignore the optical branch entirely (ablation baseline)
    # "or"        = plain union (the v0 baseline)
    #
    # Default is "consensus" on the LATEST measurement. The ordering flipped
    # once the "before" date was confined to the max-extent envelope: before
    # that change Dempster-Shafer led (0.4044 vs 0.3838), after it the
    # calibration sweep picked consensus on every fold (0.4166 vs 0.3583).
    # Interpretation: the envelope already removes the over-detection that
    # DS's graded evidence was compensating for, so the stricter agreement
    # rule now wins outright. DS, strict_consensus, sar_only and or stay in
    # the sweep — this default is a starting point, not a verdict.
    method: str = "consensus"
    # Half-width (dB / index units) of the soft transition around a threshold.
    sar_soft_margin_db: float = 1.5
    optical_soft_margin: float = 0.1
    # Evidence discounting: how much of each source's mass is kept vs. moved
    # to "unknown". Lower = the source is trusted less.
    sar_reliability: float = 0.85
    optical_reliability: float = 0.75
    decision_threshold: float = 0.5


@dataclass
class TemporalConfig:
    """District-level threshold prior, built from extra Sentinel-1 scenes.

    The organizers ship scene passports, not pixels — imagery is fetched by
    us, so we are free to sample far more of the archive than the 11 scored
    pairs name. Those extra scenes cannot change a prediction directly (the
    metric is fixed to 11 pairs), but they establish what a threshold for
    this district normally looks like, which is enough to stop a scene whose
    tile search failed from collapsing to the dry end of the window.
    """

    use_threshold_prior: bool = True
    # Below this many observations a "district norm" is not a norm.
    min_observations: int = 3
    # How far a well-supported threshold may sit from the district median
    # before it is treated as suspect (or 2x the district IQR, whichever is
    # larger — a genuinely variable river is not punished for being one).
    outlier_tolerance_db: float = 3.0
    # A scene that produced no qualifying tile has no evidence of its own.
    fallback_weight: float = 1.0
    # Thin evidence: blend rather than replace.
    weak_evidence_tiles: int = 5
    weak_evidence_weight: float = 0.5


@dataclass
class DegenerateGuardConfig:
    """Catches a pair whose segmentation plainly failed.

    A district whose JRC permanent channel covers thousands of hectares
    cannot have zero open water on any date. When that is what comes out,
    the detector failed — and submitting 0 scores exactly 0 on that pair's
    Q components (poyarkovo: 0 ha against an 8910 ha reference). Reporting
    the climatological estimate instead is both better operational
    behaviour for a monitoring service and strictly better than zero.

    The fallback is derived from JRC permanent water for that district — a
    published archive layer shipped with the dataset, not from the
    reference masks.
    """

    enabled: bool = True
    # water_peak below this share of the district's permanent water means
    # the detector did not work, not that the river vanished.
    min_share_of_permanent: float = 0.05
    # What to report instead, as a share of permanent water.
    fallback_share_of_permanent: float = 1.0
    # Never invent new flooding this way: a failed pair reports no flood.
    fallback_flood_ha: float = 0.0


@dataclass
class ChangeConfig:
    # The reference pipeline required backscatter to drop by >= 3 dB at the
    # peak before calling a pixel newly flooded. Mirroring that requirement
    # is the single cheapest way to suppress false positives that survive
    # independent per-date thresholding.
    require_backscatter_drop: bool = True
    min_drop_db: float = 3.0


@dataclass
class PostprocessConfig:
    # JRC GSW occurrence >= this is "permanent water" per the brief.
    permanent_occurrence_pct: float = 80.0
    # The reference labelling used 5 deg; the metric scores agreement with
    # that labelling, not with nature.
    slope_max_deg: float = 5.0
    hand_max_m: float = 25.0
    builtup_thresh: float = 0.5
    # Minimum mapping unit, in pixels (25 px = 0.25 ha at 10 m).
    min_mapping_unit_px: int = 25
    hydrography_buffer_px: int = 3
    connectivity_filter: bool = True
    # Seed the connectivity test with JRC "maximum observed water extent"
    # as well as the OSM channel network. Measured on the first scored run:
    # dropping connectivity entirely raised Q_flood 0.215 -> 0.364, i.e. the
    # OSM-only seed was discarding real flood that never touches a mapped
    # line. max_extent is the physically right layer — every genuine flood
    # pixel has been water at some point in the JRC archive.
    connectivity_use_max_extent: bool = True
    # Two-tier connectivity. Measured tradeoff on the last scored run:
    # seeding connectivity with JRC max_extent as well as the OSM channel
    # network lifts Q_flood 0.208 -> 0.362, but drops Spec_base 0.951 ->
    # 0.738 — the looser anchor lets phantom flood survive on the dry
    # control pairs. Rather than choosing one, require more evidence from
    # the weaker anchor: a component touching a mapped channel keeps the
    # normal minimum mapping unit, while one anchored only to "this pixel
    # was water sometime in the JRC archive" must be this large to count.
    max_extent_only_min_px: int = 200
    # Confine the "before" water surface to the JRC maximum-observed-water
    # envelope (dilated slightly, since a shoreline moves between epochs).
    # Measured on all 11 shipped reference masks: reference `water_pre` sits
    # 92.5-99.8% inside max_extent on every single pair, while reference
    # `flood` sits only 15-98% inside — so the envelope is a sound constraint
    # for the low-water date and a bad one for new flooding. Our own
    # water_pre was over-detected by up to 3x (blagoveshchensk 2019: 9358 ha
    # against 2867 ha), and since flood = peak AND NOT pre, that inflation
    # was eating the flood class too.
    restrict_water_pre_to_max_extent: bool = True
    max_extent_dilation_px: int = 5


@dataclass
class Config:
    sar: SarConfig = field(default_factory=SarConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    degenerate_guard: DegenerateGuardConfig = field(default_factory=DegenerateGuardConfig)
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        sections = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for name, value in (data or {}).items():
            if name not in sections:
                raise KeyError(f"unknown config section {name!r}")
            if name == "seed":
                kwargs[name] = value
                continue
            section_cls = {
                "sar": SarConfig,
                "optical": OpticalConfig,
                "fusion": FusionConfig,
                "change": ChangeConfig,
                "postprocess": PostprocessConfig,
                "temporal": TemporalConfig,
                "degenerate_guard": DegenerateGuardConfig,
            }[name]
            known = {f.name for f in fields(section_cls)}
            unknown = set(value) - known
            if unknown:
                raise KeyError(f"unknown keys in [{name}]: {sorted(unknown)}")
            kwargs[name] = section_cls(**value)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | None) -> "Config":
        """Loads YAML (preferred) or JSON. `None` returns defaults."""
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **section_overrides) -> "Config":
        """Returns a copy with named leaf values overridden, e.g.
        `cfg.replace(postprocess={"hand_max_m": 15.0})` — used by the
        calibration sweep so it never mutates shared state."""
        data = self.to_dict()
        for section, overrides in section_overrides.items():
            data[section] = {**data[section], **overrides}
        return Config.from_dict(data)
