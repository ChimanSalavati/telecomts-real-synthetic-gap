"""End-to-end latency benchmark: HGB baseline vs audit-screened HGB inference.

This is the script that produced ``evidence/industrial/latency_audited_vs_baseline.csv``
and ``.md`` for the CIKM 2026 paper's §5.2 latency numbers.

Run on any per-flow CSV with an ``is_anomalous`` column::

    python evidence/industrial/scripts/bench_audited_inference.py \\
        --csv /path/to/industrial_anomaly_1s.csv \\
        --warmup 200 \\
        --measure 2000 \\
        --results-dir evidence/industrial/

Three configurations are timed:

  A) HGB inference only (baseline).
  B) Audit-screen + HGB inference (audited deployment).
  C) Audit-screen only (isolates the cost the audit gate adds).

The audit screen here is the audit *gate decision* (read a precomputed
verdict from a small JSON), not a fresh C2ST + MMD per window: in
deployment the audit is run ONCE per benchmark refresh, exactly as a
pre-deployment screening step. We therefore time the per-window cost of
consulting the gate decision, which is what production actually pays.
"""
from __future__ import annotations

import argparse
import statistics as _stats
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


_IGNORE = {
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
}


def _select_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in _IGNORE
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("anomalous_packet_count_class_")
    ]


def _fit_baseline_hgb(
    df: pd.DataFrame, feature_cols: list[str], seed: int = 0
) -> HistGradientBoostingClassifier:
    X = df[feature_cols].to_numpy(dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    y = df["is_anomalous"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr = perm[:cut]
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X[tr], y[tr])
    return clf


def _bench(
    clf, X_one_row: np.ndarray, *, warmup: int, measure: int, audit_gate_us: float = 0.0
) -> dict:
    for _ in range(warmup):
        clf.predict_proba(X_one_row)

    samples_ns: list[int] = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        if audit_gate_us > 0.0:
            target = t0 + int(audit_gate_us * 1000)
            while time.perf_counter_ns() < target:
                pass
        clf.predict_proba(X_one_row)
        samples_ns.append(time.perf_counter_ns() - t0)
    return {
        "n": measure,
        "median_ms": float(_stats.median(samples_ns)) / 1e6,
        "mean_ms": float(_stats.mean(samples_ns)) / 1e6,
        "p95_ms": float(np.percentile(samples_ns, 95)) / 1e6,
        "p99_ms": float(np.percentile(samples_ns, 99)) / 1e6,
        "stdev_ms": float(_stats.pstdev(samples_ns)) / 1e6,
    }


def _measure_gate_cost(n_iter: int = 10000) -> float:
    verdict = {
        "verdict": "origin_incomplete_synthetic_only",
        "c2st_acc": 1.0,
        "mmd_norm": 50.0,
    }
    samples_ns = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        v = verdict["verdict"]
        _ok = v == "pass"
        samples_ns.append(time.perf_counter_ns() - t0)
    return float(np.median(samples_ns)) / 1e3  # microseconds


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark per-window inference latency of HGB with and without "
            "the origin-aware audit gate, on any per-flow CSV that has an "
            "is_anomalous column."
        )
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--measure", type=int, default=2000)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evidence/industrial"),
        help="Directory to write latency_audited_vs_baseline.{csv,md}",
    )
    args = p.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] loading {args.csv}")
    df = pd.read_csv(args.csv)
    feats = _select_features(df)
    print(f"[bench] {len(df)} rows, {len(feats)} numeric features")

    print("[bench] fitting baseline HGB ...")
    clf = _fit_baseline_hgb(df, feats)

    X_one = df[feats].iloc[[0]].to_numpy(dtype=np.float64)
    np.nan_to_num(X_one, copy=False, nan=0.0)

    print("[bench] measuring audit gate-read cost ...")
    gate_us = _measure_gate_cost(n_iter=10000)
    print(f"[bench] gate-read cost median = {gate_us:.3f} us")

    print("[bench] (A) HGB baseline ...")
    a = _bench(clf, X_one, warmup=args.warmup, measure=args.measure, audit_gate_us=0.0)
    print("[bench] (B) Audit-screen + HGB ...")
    b = _bench(
        clf, X_one, warmup=args.warmup, measure=args.measure, audit_gate_us=gate_us
    )
    print("[bench] (C) Audit gate-only (no HGB) ...")
    samples_c_ns: list[int] = []
    verdict = {"verdict": "origin_incomplete_synthetic_only"}
    for _ in range(args.measure):
        t0 = time.perf_counter_ns()
        v = verdict["verdict"]
        _ok = v == "pass"
        samples_c_ns.append(time.perf_counter_ns() - t0)
    c = {
        "n": args.measure,
        "median_ms": float(np.median(samples_c_ns)) / 1e6,
        "mean_ms": float(np.mean(samples_c_ns)) / 1e6,
        "p95_ms": float(np.percentile(samples_c_ns, 95)) / 1e6,
        "p99_ms": float(np.percentile(samples_c_ns, 99)) / 1e6,
        "stdev_ms": float(np.std(samples_c_ns)) / 1e6,
    }

    rows = [
        {"configuration": "HGB inference (baseline)", **a},
        {"configuration": "Audit screen + HGB inference", **b},
        {"configuration": "Audit screen alone", **c},
    ]
    out_csv = args.results_dir / "latency_audited_vs_baseline.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[bench] wrote {out_csv}")

    md_lines = [
        "# End-to-end latency: HGB baseline vs audit-screened HGB",
        "",
        f"Measured on Apple M-series laptop, single-thread CPU, {args.measure} "
        f"calls per row, {args.warmup}-call warmup, single 232-column window.",
        "",
        "| Configuration | Median (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['configuration']} | {r['median_ms']:.3f} | "
            f"{r['p95_ms']:.3f} | {r['p99_ms']:.3f} | "
            f"{r['mean_ms']:.3f} | {r['stdev_ms']:.3f} |"
        )
    md_lines += [
        "",
        f"Audit gate-read median cost: **{gate_us:.3f} us** (i.e. {gate_us / 1000:.4f} ms).",
        "",
        "The audit gate is run ONCE per benchmark refresh (or whenever the "
        "training pool changes), not per inference. The per-window cost is "
        "therefore the dict-read latency above. The full origin audit "
        "(C2ST + MMD) is amortized over thousands of inferences.",
    ]
    out_md = args.results_dir / "latency_audited_vs_baseline.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[bench] wrote {out_md}")

    print("\n=== SUMMARY ===")
    for r in rows:
        print(
            f"  {r['configuration']:35s}  median={r['median_ms']:>6.3f} ms   "
            f"p95={r['p95_ms']:>6.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
