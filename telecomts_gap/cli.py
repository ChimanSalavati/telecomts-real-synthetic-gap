"""``telecomts-audit`` console-script entry point.

Examples
--------

Audit a CSV that already has an ``anomaly_origin`` column::

    telecomts-audit \\
        --csv data/benchmark_with_origin.csv \\
        --origin-col anomaly_origin \\
        --output results/audit_verdict.json

For CSVs that ship a binary ``is_anomalous`` flag and no controlled-real
labels (the common case for benchmarks that combine real Normal traffic
with synthetic injection), build the origin column automatically::

    telecomts-audit \\
        --csv data/benchmark_anomaly_1s.csv \\
        --synthetic-only-from-flag \\
        --output results/audit_verdict.json

This returns ``origin_incomplete_synthetic_only`` and exits with code 1 so
the surrounding CI pipeline can gate on it.

Print the operator checklist as plain text (same text as ``CHECKLIST.md``)::

    telecomts-audit --print-checklist

The CLI never reaches out to any external service; it is a pure-function
entry point so it can be unit-tested and called from any container.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ._checklist import get_checklist
from .origin_audit import origin_audit


# Columns the audit ignores even if numeric: bookkeeping, timestamps, ids,
# and any label-derived column. We never want to "audit" the label itself.
# This default list is a superset of common per-flow-CSV layouts; users
# whose schema differs can pass ``--ignore-cols ...`` to extend it.
_DEFAULT_IGNORE = {
    "window_id",
    "interval_start_epoch",
    "interval_end_epoch",
    "interval_seconds",
    "first_packet_time",
    "last_packet_time",
    "active_duration",
    "extraction_timestamp",
    "is_anomalous",
    "anomalous_packet_count",
    "anomalous_packet_ratio",
    "anomalous_bytes_total",
    "anomalous_bytes_ratio",
    "distinct_anomaly_class_count",
    "anomalous_packet_count_class_1",
    "anomalous_packet_count_class_2",
    "anomalous_packet_count_class_3",
    "anomalous_packet_count_class_4",
    "anomalous_packet_count_class_5",
    "anomalous_packet_count_class_6",
    "anomalous_packet_count_class_7",
    "anomalous_packet_count_class_8",
}


def _attach_synthetic_only_from_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Build a synthetic-only ``anomaly_origin`` column from an
    ``is_anomalous`` binary flag.

    Used for CSVs that ship real Normal traffic plus injected synthetic
    anomalies with no controlled-real labels. Normal-only baseline
    captures may leave ``is_anomalous`` empty / NaN; those are all-normal
    by definition.
    """
    if "anomaly_origin" in df.columns:
        return df
    if "is_anomalous" not in df.columns:
        raise KeyError(
            "--synthetic-only-from-flag expects an ``is_anomalous`` "
            f"column; got {list(df.columns)[:20]}..."
        )
    flag = pd.to_numeric(df["is_anomalous"], errors="coerce").fillna(0).astype(int)
    origin = pd.Series("normal", index=df.index, dtype="object")
    origin.loc[flag == 1] = "synthetic"
    return df.assign(anomaly_origin=origin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="telecomts-audit",
        description=(
            "Origin-aware benchmark audit for telecom anomaly-detection "
            "pipelines. Returns a verdict the operator gate consumes."
        ),
    )
    parser.add_argument(
        "--print-checklist",
        action="store_true",
        help="Print the operator pre-deployment checklist and exit.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV.")
    parser.add_argument(
        "--origin-col",
        default="anomaly_origin",
        help="Column name with the anomaly-origin label.",
    )
    parser.add_argument(
        "--controlled-real-label",
        default="controlled_real",
        help="Label used in --origin-col for controlled-real anomalies.",
    )
    parser.add_argument(
        "--synthetic-label",
        default="synthetic",
        help="Label used in --origin-col for perturbation-synthetic anomalies.",
    )
    parser.add_argument(
        "--synthetic-only-from-flag",
        dest="synthetic_only_from_flag",
        action="store_true",
        help=(
            "Build the anomaly_origin column automatically from a binary "
            "``is_anomalous`` flag. Use for CSVs that ship real Normal "
            "traffic plus injected synthetic anomalies with no "
            "controlled-real labels."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON verdict here."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--no-mmd", action="store_true")
    parser.add_argument("--no-bh", action="store_true")
    parser.add_argument(
        "--feature-prefix-allow",
        nargs="*",
        default=None,
        help=(
            "Optional whitelist of column prefixes to treat as numeric "
            "features. If unset, every numeric column not in the default "
            "ignore-list is used."
        ),
    )
    parser.add_argument(
        "--ignore-cols",
        nargs="*",
        default=None,
        help="Extra column names to add to the default ignore-list.",
    )

    args = parser.parse_args(argv)

    if args.print_checklist:
        sys.stdout.write(get_checklist())
        return 0

    if args.csv is None:
        parser.error("--csv is required unless --print-checklist is set")

    df = pd.read_csv(args.csv)
    if args.synthetic_only_from_flag:
        df = _attach_synthetic_only_from_flag(df)

    ignore_set = set(_DEFAULT_IGNORE)
    if args.ignore_cols:
        ignore_set.update(args.ignore_cols)

    feature_cols: list[str]
    if args.feature_prefix_allow:
        feature_cols = [
            c
            for c in df.columns
            if any(c.startswith(p) for p in args.feature_prefix_allow)
            and c not in ignore_set
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        feature_cols = [
            c
            for c in df.columns
            if c != args.origin_col
            and c not in ignore_set
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    result = origin_audit(
        df,
        origin_col=args.origin_col,
        feature_cols=feature_cols,
        controlled_real_label=args.controlled_real_label,
        synthetic_label=args.synthetic_label,
        do_mmd=not args.no_mmd,
        do_bh=not args.no_bh,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    payload: dict[str, Any] = {
        "csv": str(args.csv),
        "origin_col": args.origin_col,
        "feature_count": len(feature_cols),
        "result": result.to_dict(),
    }
    out_str = json.dumps(payload, indent=2, default=str)
    print(out_str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_str + "\n")
    # Exit code: 0 = pass, 1 = gap detected / origin incomplete (so CI can gate).
    return 0 if result.verdict.value == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
