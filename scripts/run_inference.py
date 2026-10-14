#!/usr/bin/env python
"""End-to-end inference: pairs.csv -> submission.csv + predictions/ + reports/.

Usage:
    python run_inference.py --data-dir /path/to/hydrowatch_amur \
                             --out-dir  /path/to/outputs \
                             --gee-project my-earth-engine-project \
                             [--config configs/default.yaml] \
                             [--checkpoint-dir /path/on/drive] [--resume]

Requires a Google Earth Engine account with the project registered for the
Earth Engine API (https://code.earthengine.google.com/register). The first
run prompts an interactive OAuth flow; later runs reuse cached credentials.

Outputs
    submission.csv                     the four scored columns, one row per pair
    predictions/<pair_id>_flood.tif    uint8 mask on the pair's exact grid
    reports/<pair_id>.json             machine-readable summary per pair
    reports/summary.csv                all pairs in one table
    vectors/<pair_id>.geojson          water/flood contours with attributes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hydrowatch import checkpoint as ckpt  # noqa: E402
from hydrowatch import offline  # noqa: E402
from hydrowatch.areas import mask_area_ha  # noqa: E402
from hydrowatch.config import Config  # noqa: E402
from hydrowatch.dataset_check import check_dataset_detailed  # noqa: E402
from hydrowatch.grid import read_grid  # noqa: E402
from hydrowatch.paths import resolve  # noqa: E402
from hydrowatch.pipeline import PIPELINE_VERSION, segment_pair  # noqa: E402
from hydrowatch.report import build_pair_report, reports_to_csv_rows, write_report  # noqa: E402
from hydrowatch.submission import (  # noqa: E402
    PairResult,
    validate_pair_result,
    write_flood_mask_tif,
    write_submission_csv,
)
from hydrowatch.vectorize import write_geojson  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="path to hydrowatch_amur/")
    parser.add_argument("--out-dir", required=True, help="where outputs go")
    parser.add_argument("--gee-project", default=None, help="Earth Engine cloud project id")
    parser.add_argument("--config", default=None, help="YAML/JSON config; defaults to built-ins")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="directory (e.g. on Google Drive) for resumable per-pair outputs",
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip pairs already present in --checkpoint-dir"
    )
    parser.add_argument("--pairs", default=None, help="comma-separated pair_id subset")
    parser.add_argument("--max-workers", type=int, default=4, help="parallel GEE tile requests")
    parser.add_argument(
        "--offline",
        default=None,
        metavar="BUNDLE_DIR",
        help="run with no network at all, reading imagery from an offline bundle "
        "(build it once with export_offline_bundle.py)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # Earth Engine is imported only when actually needed. In offline mode the
    # process never touches the network, which is what the reproducibility
    # criterion asks for — a solution that requires a cloud platform at
    # inference time does not count as reproducible.
    manifest = None
    if args.offline:
        manifest = offline.read_manifest(args.offline)
        FETCH_VERSION = manifest.fetch_version
        print(f"OFFLINE mode: bundle {args.offline} "
              f"({len(manifest.pairs)} pairs, {manifest.precision}, fetch {FETCH_VERSION})")
    else:
        from hydrowatch import gee_export as ge
        from hydrowatch.gee_export import FETCH_VERSION

        ge.init_ee(project=args.gee_project)

    pairs = pd.read_csv(os.path.join(args.data_dir, "pairs.csv"))
    if args.pairs:
        pairs = pairs[pairs["pair_id"].isin(set(args.pairs.split(",")))]

    hydrography = resolve(args.data_dir, "vectors/hydrography_osm.geojson")
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.out_dir, "checkpoints")

    availabilities = check_dataset_detailed(args.data_dir, pairs)
    not_ready = {pid: a for pid, a in availabilities.items() if not a.ready_for_inference}
    if not_ready:
        report = "\n".join(f"  {pid}: missing {', '.join(a.missing)}" for pid, a in not_ready.items())
        print(
            "WARNING: --data-dir is missing files for the pairs below. They will be "
            f"written to submission.csv as 0 ha and skipped:\n{report}\n",
            flush=True,
        )

    # Mask checkpoints are keyed on the configuration that produced them:
    # resuming across an algorithm change would otherwise replay stale
    # results and silently reproduce the previous score.
    fingerprint = ckpt.config_fingerprint(cfg.to_dict(), PIPELINE_VERSION, FETCH_VERSION)
    stale = ckpt.stale_pairs(checkpoint_dir, list(pairs["pair_id"]), fingerprint)
    print(f"config fingerprint: {fingerprint} (pipeline {PIPELINE_VERSION}, fetch {FETCH_VERSION})")
    if stale:
        print(f"{len(stale)} checkpoint(s) came from a different configuration and will be recomputed.")

    results: list[PairResult] = []
    reports: list[dict] = []
    started = time.time()

    for _, row in pairs.iterrows():
        pair_id = row["pair_id"]
        avail = availabilities[pair_id]
        if not avail.ready_for_inference:
            results.append(PairResult(pair_id, 0.0, 0.0, 0.0))
            continue

        if args.resume and ckpt.is_complete(checkpoint_dir, pair_id, fingerprint):
            masks, summary = ckpt.load_pair(checkpoint_dir, pair_id)
            grid = read_grid(avail.reference_mask_path)
            results.append(
                PairResult(
                    pair_id,
                    summary["flood_ha"],
                    summary["water_pre_ha"],
                    summary["water_peak_ha"],
                )
            )
            reports.append(summary["report"])
            print(f"[{pair_id}] resumed from checkpoint", flush=True)
            continue

        grid = read_grid(avail.reference_mask_path)
        t0 = time.time()
        if manifest is not None:
            offline.verify_grid(manifest, pair_id, grid)
            pre = offline.load_date(args.offline, pair_id, "pre", manifest.precision)
            peak = offline.load_date(args.offline, pair_id, "peak", manifest.precision)
        else:
            from hydrowatch.pipeline import fetch_pair_arrays

            print(f"[{pair_id}] fetching pre/peak imagery from GEE ...", flush=True)
            pre = fetch_pair_arrays(avail.rasters_dir, grid, "pre", cfg)
            peak = fetch_pair_arrays(avail.rasters_dir, grid, "peak", cfg)
        t_fetch = time.time() - t0

        t0 = time.time()
        masks = segment_pair(
            pair_id, avail.reference_mask_path, avail.aux_terrain_path, hydrography, pre, peak, cfg
        )
        t_segment = time.time() - t0

        flood_ha = mask_area_ha(masks.flood, masks.grid)
        water_pre_ha = mask_area_ha(masks.water_pre, masks.grid)
        water_peak_ha = mask_area_ha(masks.water_peak, masks.grid)

        problems = validate_pair_result(flood_ha, water_pre_ha, water_peak_ha, row["aoi_km2"] * 100)
        if problems:
            print(f"[{pair_id}] WARNING: {problems}", flush=True)

        report = build_pair_report(
            pair_id,
            masks.grid,
            masks.water_pre,
            masks.water_peak,
            masks.flood,
            masks.receded,
            masks.terrain,
            aoi_ha=row["aoi_km2"] * 100,
            meta={
                "thresholds": masks.diagnostics,
                "config": cfg.to_dict(),
                "timing_s": {"fetch": round(t_fetch, 1), "segment": round(t_segment, 1)},
            },
        )
        reports.append(report)

        write_flood_mask_tif(
            masks.flood, masks.grid, os.path.join(args.out_dir, "predictions", f"{pair_id}_flood.tif")
        )
        write_report(report, os.path.join(args.out_dir, "reports", f"{pair_id}.json"))
        write_geojson(
            {
                "water_pre": masks.water_pre,
                "water_peak": masks.water_peak,
                "flood": masks.flood,
                "receded": masks.receded,
            },
            masks.grid,
            os.path.join(args.out_dir, "vectors", f"{pair_id}.geojson"),
            min_pixels=cfg.postprocess.min_mapping_unit_px,
        )
        ckpt.save_pair(
            checkpoint_dir,
            pair_id,
            {
                "flood": masks.flood,
                "water_pre": masks.water_pre,
                "water_peak": masks.water_peak,
                "receded": masks.receded,
            },
            masks.grid,
            {
                "flood_ha": flood_ha,
                "water_pre_ha": water_pre_ha,
                "water_peak_ha": water_peak_ha,
                "report": report,
            },
            fingerprint=fingerprint,
        )

        results.append(PairResult(pair_id, flood_ha, water_pre_ha, water_peak_ha))
        print(
            f"[{pair_id}] flood={flood_ha:.2f} ha water_pre={water_pre_ha:.2f} ha "
            f"water_peak={water_peak_ha:.2f} ha  (fetch {t_fetch:.0f}s, segment {t_segment:.0f}s)",
            flush=True,
        )

    write_submission_csv(results, os.path.join(args.out_dir, "submission.csv"))
    if reports:
        pd.DataFrame(reports_to_csv_rows(reports)).to_csv(
            os.path.join(args.out_dir, "reports", "summary.csv"), index=False
        )
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=2)

    print(
        f"\nWrote {os.path.join(args.out_dir, 'submission.csv')} ({len(results)} pairs) "
        f"in {(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
