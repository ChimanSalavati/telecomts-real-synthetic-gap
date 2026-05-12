"""Shadow-mode usage statistics: batch-audit every per-flow CSV in a workspace.

Invokes ``telecomts-audit --synthetic-only-from-flag`` over every CSV listed
in a config file, records per-CSV verdict + wall-clock, and aggregates into
a single ``shadow_mode_usage.json``.

This is the script that produced
``evidence/industrial/shadow_mode_usage.json`` for the CIKM 2026 paper's
§5.2 shadow-mode statistics.

Run::

    python evidence/industrial/scripts/batch_shadow_audit.py \\
        --workspace /path/to/operator/workspace \\
        --csv-list configs/csvs.txt \\
        --output evidence/industrial/shadow_mode_usage.json

Where ``configs/csvs.txt`` contains one CSV path per line, resolved
relative to ``--workspace``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from telecomts_gap.cli import _DEFAULT_IGNORE, _attach_synthetic_only_from_flag
from telecomts_gap.origin_audit import origin_audit


def _load_csv_list(workspace: Path, csv_list_path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in csv_list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = (workspace / line).resolve() if not Path(line).is_absolute() else Path(line)
        paths.append(p)
    return paths


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Batch-audit a list of per-flow CSVs and emit aggregate "
            "shadow-mode usage statistics as a single JSON."
        )
    )
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument(
        "--csv-list",
        type=Path,
        required=True,
        help="Path to a text file with one CSV path per line, relative to --workspace.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/industrial/shadow_mode_usage.json"),
    )
    p.add_argument("--n-perm", type=int, default=200)
    args = p.parse_args()

    csv_paths = _load_csv_list(args.workspace, args.csv_list)

    per_csv = []
    total_windows = 0
    total_anomalous = 0
    total_normal = 0
    total_audit_wall_s = 0.0
    total_interval_seconds = 0.0
    verdict_counts = {
        "pass": 0,
        "gap_detected": 0,
        "origin_incomplete_synthetic_only": 0,
        "origin_incomplete_real_only": 0,
        "no_anomalies_present": 0,
    }
    detectors_blocked = 0

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[shadow] SKIP missing {csv_path}")
            continue

        print(f"\n[shadow] auditing {csv_path}")
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        df_aug = _attach_synthetic_only_from_flag(df)
        feat_cols = [
            c
            for c in df_aug.columns
            if c != "anomaly_origin"
            and c not in _DEFAULT_IGNORE
            and pd.api.types.is_numeric_dtype(df_aug[c])
        ]
        n_rows = len(df_aug)
        n_anom = int((df_aug.get("is_anomalous", 0) == 1).sum())
        n_norm = n_rows - n_anom
        interval_sum = 0.0
        if "interval_seconds" in df_aug.columns:
            interval_sum = float(df_aug["interval_seconds"].fillna(1.0).sum())

        result = origin_audit(
            df_aug,
            origin_col="anomaly_origin",
            feature_cols=feat_cols,
            do_mmd=True,
            do_bh=True,
            n_perm=args.n_perm,
            seed=0,
        )
        wall = time.perf_counter() - t0

        per_csv.append(
            {
                "csv": csv_path.name,
                "rows": n_rows,
                "n_normal": n_norm,
                "n_anomalous": n_anom,
                "n_features_after_ignore": len(feat_cols),
                "interval_seconds_total": interval_sum,
                "wall_seconds": round(wall, 3),
                "verdict": result.verdict.value,
                "c2st_accuracy": result.c2st_accuracy,
                "c2st_auroc": result.c2st_auroc,
                "mmd_norm": result.mmd_norm,
                "mmd_p_value": result.mmd_p_value,
                "bh_significant_features": result.bh_significant_features,
            }
        )
        verdict_counts[result.verdict.value] = (
            verdict_counts.get(result.verdict.value, 0) + 1
        )
        if result.verdict.value.startswith("origin_incomplete") or result.verdict.value == "gap_detected":
            detectors_blocked += 1
        total_windows += n_rows
        total_anomalous += n_anom
        total_normal += n_norm
        total_audit_wall_s += wall
        total_interval_seconds += interval_sum

        print(
            f"  verdict={result.verdict.value}  rows={n_rows}  "
            f"anom={n_anom}  norm={n_norm}  wall={wall:.2f}s"
        )

    summary = {
        "total_csvs_audited": len(per_csv),
        "total_windows_audited": total_windows,
        "total_anomalous_windows": total_anomalous,
        "total_normal_windows": total_normal,
        "total_timing_flow_seconds": total_interval_seconds,
        "total_timing_flow_hours": round(total_interval_seconds / 3600.0, 3),
        "total_audit_wall_seconds": round(total_audit_wall_s, 2),
        "total_audit_wall_minutes": round(total_audit_wall_s / 60.0, 2),
        "median_audit_wall_seconds": (
            float(np.median([r["wall_seconds"] for r in per_csv])) if per_csv else None
        ),
        "p95_audit_wall_seconds": (
            float(np.percentile([r["wall_seconds"] for r in per_csv], 95))
            if per_csv
            else None
        ),
        "verdict_distribution": verdict_counts,
        "detectors_held_in_shadow": detectors_blocked,
        "per_csv": per_csv,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n[shadow] wrote {args.output}")

    print("\n=== SHADOW-MODE USAGE SUMMARY ===")
    for k, v in summary.items():
        if k in {"per_csv", "verdict_distribution"}:
            continue
        print(f"  {k:36s} = {v}")
    print(f"  verdict_distribution                 = {verdict_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
