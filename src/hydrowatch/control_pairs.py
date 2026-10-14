"""Extra control pairs mined from the shipped scene catalogue.

The rules permit gathering additional training data from open EO sources.
A trained model is not what this solution has, so the useful reading of
that permission is: **more data to calibrate and validate against**. The
obstacle is that extra events come without reference masks, so they cannot
be scored.

There is one family of extra pairs that needs no reference mask at all.
`scene_catalog.csv` lists 3-6 Sentinel-1 acquisitions inside each `pre`,
`peak` and `post` window. Take two acquisitions from the *same* window: no
hydrological event separates them, so the correct answer for new flooding
is known a priori — essentially zero. That is a free label, derived from
chronology rather than from the organizers' masks.

Why it matters here: `Spec_base` (weight 0.15) is measured on just three
низкая-вода pairs, and it is currently the weakest component (0.74 after a
baseline AOI produced 497 ha of phantom flood). Mining the catalogue yields
**39 orbit-matched control pairs against those 3**, which lets calibration
actually see false-alarm behaviour instead of inferring it from three
samples.

Only orbit-matched pairs are offered by default: the case brief is explicit
that «пары "до/после" с разных орбит геометрически несопоставимы», and a
radar-shadow pattern that moves between orbits would manufacture exactly
the false change these pairs are meant to measure.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ControlPair:
    """Two acquisitions of one AOI from inside one window.

    `expected_flood_ha` is 0 by construction: nothing happened in between.
    """

    aoi_id: str
    event_id: str
    window: str
    date_a: str
    date_b: str
    relative_orbit: float | None

    @property
    def control_id(self) -> str:
        return f"control__{self.aoi_id}__{self.event_id}__{self.window}__{self.date_a}__{self.date_b}"

    @property
    def expected_flood_ha(self) -> float:
        return 0.0


def enumerate_control_pairs(
    scene_catalog: pd.DataFrame,
    same_orbit_only: bool = True,
    windows: tuple[str, ...] = ("pre", "post"),
    max_per_group: int | None = None,
) -> list[ControlPair]:
    """All same-window acquisition pairs, newest-first within each group.

    `peak` is excluded by default: inside a peak window the water level is
    genuinely moving (the brief notes the shoreline shifts within hours), so
    "nothing happened between these two dates" is not safe there. `pre` and
    `post` sit on the flat parts of the hydrograph.
    """
    s1 = scene_catalog[scene_catalog["sensor"] == "sentinel1"]
    pairs: list[ControlPair] = []

    for (aoi_id, event_id, window), group in s1.groupby(["aoi_id", "event_id", "window"]):
        if window not in windows:
            continue
        dates = sorted(group["date"].unique())
        made = 0
        for date_a, date_b in itertools.combinations(dates, 2):
            orbits_a = set(group.loc[group["date"] == date_a, "relativeOrbitNumber_start"].dropna())
            orbits_b = set(group.loc[group["date"] == date_b, "relativeOrbitNumber_start"].dropna())
            shared = orbits_a & orbits_b
            if same_orbit_only and not shared:
                continue
            if max_per_group is not None and made >= max_per_group:
                break
            pairs.append(
                ControlPair(
                    aoi_id=aoi_id,
                    event_id=event_id,
                    window=window,
                    date_a=date_a,
                    date_b=date_b,
                    relative_orbit=min(shared) if shared else None,
                )
            )
            made += 1
    return pairs


def passport_for(pair: ControlPair, orbit_pass: str | None):
    """Scene passport for one side of a control pair.

    The orbit must come from the catalogue row for *that date*, not from the
    AOI's scored pair: a control acquisition is frequently on a different
    relative orbit, and reusing the scored pair's orbit makes the Earth
    Engine filter match nothing at all.
    """
    from .gee_export import ScenePassport

    return ScenePassport(
        sensor="sentinel1",
        scene_date=pair.date_a,
        orbit_pass=orbit_pass,
        relative_orbit=int(pair.relative_orbit) if pair.relative_orbit is not None else None,
    )


def false_alarm_rate(flood_ha: float, aoi_ha: float) -> float:
    """Share of the district wrongly called newly flooded.

    Same quantity the official `Spec_base` uses, so a configuration tuned on
    these mined pairs is being tuned on the metric's own definition rather
    than on a proxy.
    """
    if aoi_ha <= 0:
        return 0.0
    return max(0.0, flood_ha) / aoi_ha


def spec_score(false_alarm_shares, tolerance: float = 0.005) -> float:
    """`Spec_base` formula from the brief, applied to mined control pairs."""
    shares = list(false_alarm_shares)
    if not shares:
        return float("nan")
    return sum(1.0 - min(1.0, s / tolerance) for s in shares) / len(shares)


def summarize(pairs: list[ControlPair]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "control_id": p.control_id,
                "aoi_id": p.aoi_id,
                "event_id": p.event_id,
                "window": p.window,
                "date_a": p.date_a,
                "date_b": p.date_b,
                "relative_orbit": p.relative_orbit,
            }
            for p in pairs
        ]
    )
