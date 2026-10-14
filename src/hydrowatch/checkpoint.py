"""Resumable runs, with two caches that expire on different terms.

A full pass is network-bound (hundreds of Earth Engine tile requests) and
Colab sessions die, so results are checkpointed to a directory that can
live on Google Drive. The two kinds of cached work must NOT share an
expiry rule:

- **Downloaded imagery** (`arrays_<window>_<FETCH_VERSION>.npz`): expensive
  (network) and independent of how we later segment it, so it survives
  configuration changes — but NOT changes to the download logic itself.
- **Masks and summaries**: cheap to recompute (~10 s per pair) and entirely
  determined by the algorithm and its thresholds. Keyed on a fingerprint of
  the configuration plus `PIPELINE_VERSION`.

Both keys were learned the hard way, one run each. Mask checkpoints were
first keyed on `pair_id` alone, so after the algorithm was fixed a re-run
replayed every stale result and reproduced a byte-identical score. Then the
array cache had no key at all, so a fix to scene coverage ran against
imagery downloaded before that fix existed and changed nothing — poyarkovo
stayed at 0% coverage and 0 ha. A cache that cannot say *which code wrote
it* is a trap, and it is a trap that costs a whole scored run to notice.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import rasterio

from .grid import Grid
from .submission import write_flood_mask_tif

MASK_LAYERS = ("flood", "water_pre", "water_peak", "receded")


def config_fingerprint(
    config_dict: dict, pipeline_version: str, fetch_version: str | None = None
) -> str:
    """Short, stable hash of everything that can change a mask.

    `fetch_version` belongs here even though it describes the *download*
    rather than the segmentation: a mask is a function of the imagery it was
    computed from. Leaving it out meant a valid mask checkpoint
    short-circuited the fetch entirely, so a fix to scene coverage could
    never take effect — poyarkovo stayed at 0 ha across two scored runs for
    exactly this reason.
    """
    payload = json.dumps(
        {"config": config_dict, "pipeline": pipeline_version, "fetch": fetch_version},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pair_dir(checkpoint_dir: str, pair_id: str) -> str:
    return os.path.join(checkpoint_dir, pair_id)


def is_complete(checkpoint_dir: str, pair_id: str, fingerprint: str | None = None) -> bool:
    """A pair counts as done only when every mask and its summary exist AND
    the summary was written by the same configuration you are about to run.

    Passing `fingerprint=None` keeps the old, unsafe behaviour and is only
    for reading back a run whose config you no longer have.
    """
    d = pair_dir(checkpoint_dir, pair_id)
    summary_path = os.path.join(d, "summary.json")
    if not os.path.isfile(summary_path):
        return False
    if not all(os.path.isfile(os.path.join(d, f"{layer}.tif")) for layer in MASK_LAYERS):
        return False
    if fingerprint is None:
        return True
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            stored = json.load(fh).get("fingerprint")
    except (OSError, json.JSONDecodeError):
        return False
    return stored == fingerprint


def stale_pairs(checkpoint_dir: str, pair_ids, fingerprint: str) -> list[str]:
    """Pairs that have a checkpoint, but from a different configuration —
    i.e. exactly the ones a naive resume would silently replay."""
    return [
        p
        for p in pair_ids
        if is_complete(checkpoint_dir, p, None) and not is_complete(checkpoint_dir, p, fingerprint)
    ]


def save_pair(
    checkpoint_dir: str,
    pair_id: str,
    masks: dict[str, np.ndarray],
    grid: Grid,
    summary: dict,
    fingerprint: str | None = None,
) -> str:
    d = pair_dir(checkpoint_dir, pair_id)
    os.makedirs(d, exist_ok=True)
    for layer in MASK_LAYERS:
        if layer in masks:
            write_flood_mask_tif(masks[layer], grid, os.path.join(d, f"{layer}.tif"))
    # summary.json is written last: its presence is the "this pair finished"
    # marker that is_complete() keys on.
    payload = dict(summary)
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    tmp = os.path.join(d, "summary.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(d, "summary.json"))
    return d


def load_pair(checkpoint_dir: str, pair_id: str) -> tuple[dict[str, np.ndarray], dict]:
    d = pair_dir(checkpoint_dir, pair_id)
    masks = {}
    for layer in MASK_LAYERS:
        path = os.path.join(d, f"{layer}.tif")
        if os.path.isfile(path):
            with rasterio.open(path) as ds:
                masks[layer] = ds.read(1).astype(bool)
    with open(os.path.join(d, "summary.json"), "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    return masks, summary


def array_cache_path(checkpoint_dir: str, pair_id: str, window: str, fetch_version: str | None = None) -> str:
    """The cache file carries the fetch version in its name.

    Imagery is expensive and algorithm-independent, so it is kept across
    runs — but only while the *download* logic is unchanged. When band
    selection, scene choice, nodata handling or coverage gap-filling change,
    the old arrays are a different product and must not be reused. Leaving
    that out once cost a whole AOI: the poyarkovo coverage fix ran against a
    cache written before it existed, and the pair stayed at 0% coverage.
    """
    suffix = f"_{fetch_version}" if fetch_version else ""
    return os.path.join(pair_dir(checkpoint_dir, pair_id), f"arrays_{window}{suffix}.npz")


def save_arrays(
    checkpoint_dir: str, pair_id: str, window: str, arrays: dict, fetch_version: str | None = None
) -> None:
    """float16 keeps ~0.01 dB precision on sigma0 and ~5e-4 on indices —
    far below the noise floor, and halves the footprint."""
    path = array_cache_path(checkpoint_dir, pair_id, window, fetch_version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for key, value in arrays.items():
        if value is None:
            continue
        payload[key] = value.astype(np.float16) if value.dtype.kind == "f" else value
    np.savez_compressed(path, **payload)


def load_arrays(
    checkpoint_dir: str, pair_id: str, window: str, fetch_version: str | None = None
) -> dict | None:
    path = array_cache_path(checkpoint_dir, pair_id, window, fetch_version)
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}
