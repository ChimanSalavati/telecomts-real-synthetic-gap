"""HGB baseline on a per-flow industrial CSV.

Section 5.2 ("Nokia pre-production integration") of
[main_gap_paper.tex](../../main_gap_paper.tex) says:

    Under aggregate reporting, an HGB detector would pass the existing
    validation gate (F_1 ~= 0.97 at a balanced rate and 0.94 at the
    natural rate).

This script produces the JSON artifact behind that claim. It is the
detector-side numbers (aggregate F1, AUROC, recall, FPR); the audit-side
verdict on the same CSV is in
[`../../evidence/industrial/audit_verdict.json`](../../evidence/industrial/audit_verdict.json).

Run::

    python experiments/N1_industrial_hgb_baseline/run_n1.py \\
        --csv /path/to/industrial_anomaly_1s.csv \\
        --output evidence/industrial/hgb_baseline_results.json \\
        --n-seeds 10

Two complementary protocols are run:

- **Natural rate**: stratified k-fold cross-validation on the full CSV
  (anomaly rate as-given). HGB is fit on the training folds, threshold
  is tuned on a stratified validation slice carved from the training
  pool, and the held-out test fold is scored. Mirrors the public
  E9/E14 protocol.

- **Balanced rate**: a 50/50 sub-corpus is built once (all positives +
  an equal-size random sample of negatives), then stratified k-fold CV
  is run on that sub-corpus with the same threshold tuning. Mirrors the
  "balanced detection" protocol from the paper.

Reported numbers are mean / std across multiple HGB random seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# Columns we never feed the detector: bookkeeping, timestamps, ids, and
# any label-derived column. Mirrors `telecomts_gap.cli._DEFAULT_IGNORE`.
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
    "anomalous_packet_count_class_1",
    "anomalous_packet_count_class_2",
    "anomalous_packet_count_class_3",
    "anomalous_packet_count_class_4",
    "anomalous_packet_count_class_5",
    "anomalous_packet_count_class_6",
    "anomalous_packet_count_class_7",
    "anomalous_packet_count_class_8",
}


def _select_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in _IGNORE and pd.api.types.is_numeric_dtype(df[c])
    ]


def _fit_threshold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    *,
    balance_sample_weight: bool = True,
) -> tuple[HistGradientBoostingClassifier, float]:
    """Fit HGB (optionally with balanced sample weights) and pick the
    F1-maximizing threshold on the validation set.

    ``balance_sample_weight`` reweights each class so total positive and
    negative weight match -- the equivalent of ``class_weight='balanced'``
    for estimators that do not expose that argument. This is the standard
    fix when the natural anomaly rate is small and an un-tuned threshold
    would otherwise leave recall on the table.
    """
    # Default HGB hyperparameters: matches the public E9/E14 protocol.
    clf = HistGradientBoostingClassifier(random_state=seed)
    if balance_sample_weight and y_tr.sum() > 0 and y_tr.sum() < y_tr.size:
        n_pos = int(y_tr.sum())
        n_neg = int(y_tr.size - n_pos)
        w_pos = 1.0
        w_neg = n_pos / max(n_neg, 1)
        sample_weight = np.where(y_tr == 1, w_pos, w_neg)
        clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        clf.fit(X_tr, y_tr)
    val_scores = clf.predict_proba(X_val)[:, 1]
    if y_val.sum() == 0 or y_val.sum() == y_val.size:
        return clf, 0.5
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    prec_a, rec_a = prec[:-1], rec[:-1]
    denom = prec_a + rec_a
    f1s = np.where(denom > 0, 2 * prec_a * rec_a / np.maximum(denom, 1e-12), 0.0)
    if f1s.size == 0:
        return clf, 0.5
    return clf, float(thr[int(np.argmax(f1s))])


def _evaluate(
    y_true: np.ndarray, scores: np.ndarray, thr: float
) -> dict[str, float]:
    pred = (scores >= thr).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "recall": float(((pred == 1) & (y_true == 1)).sum() / max(int((y_true == 1).sum()), 1)),
        "fpr": float(((pred == 1) & (y_true == 0)).sum() / max(int((y_true == 0).sum()), 1)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "threshold": float(thr),
    }


def _kfold_eval(
    X: np.ndarray, y: np.ndarray, *, seed: int, n_folds: int
) -> dict[str, float]:
    """Run stratified k-fold CV at one HGB seed; return mean per-fold metrics."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[dict] = []
    for fold_idx, (tr_pool, te_idx) in enumerate(skf.split(X, y)):
        # Stratified 1/5 validation slice for threshold tuning.
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold_idx)
        inner_tr, inner_val = next(iter(skf_inner.split(X[tr_pool], y[tr_pool])))
        tr_idx = tr_pool[inner_tr]
        val_idx = tr_pool[inner_val]
        clf, thr = _fit_threshold(
            X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], seed=seed
        )
        te_scores = clf.predict_proba(X[te_idx])[:, 1]
        folds.append(_evaluate(y[te_idx], te_scores, thr))

    def _mean(field: str) -> float:
        vals = [f[field] for f in folds if not np.isnan(f[field])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "f1": _mean("f1"),
        "auroc": _mean("auroc"),
        "recall": _mean("recall"),
        "fpr": _mean("fpr"),
        "n_pos": int(sum(f["n_pos"] for f in folds)),
        "n_neg": int(sum(f["n_neg"] for f in folds)),
        "threshold": float(np.mean([f["threshold"] for f in folds])),
    }


def _run_one_seed(
    X: np.ndarray, y: np.ndarray, seed: int, n_folds: int = 5
) -> dict[str, dict[str, float]]:
    """At one HGB random seed, evaluate at both natural and balanced rates.

    - Natural: stratified k-fold CV on the full corpus.
    - Balanced: build a 50/50 sub-corpus (all positives + same-size random
      sample of negatives), then run stratified k-fold CV on it.
    """
    rng = np.random.default_rng(seed + 7919)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = int(len(pos_idx))
    if n_pos == 0:
        raise ValueError("No positive labels in the input CSV; cannot evaluate F1.")
    keep_neg = rng.choice(neg_idx, size=min(n_pos, len(neg_idx)), replace=False)
    balanced_idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(balanced_idx)

    natural = _kfold_eval(X, y, seed=seed, n_folds=n_folds)
    balanced = _kfold_eval(
        X[balanced_idx], y[balanced_idx], seed=seed, n_folds=n_folds
    )
    return {"natural": natural, "balanced": balanced}


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "HGB baseline on a per-flow industrial CSV with an is_anomalous "
            "column. Reports F1, AUROC, recall, FPR at the natural and "
            "balanced anomaly rates over multiple HGB seeds."
        )
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/industrial/hgb_baseline_results.json"),
    )
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    feats = _select_features(df)
    X = df[feats].to_numpy(dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    y = df["is_anomalous"].astype(int).to_numpy()

    print(f"[n1] csv: {args.csv}")
    print(f"[n1] rows: {len(df)}  positives: {int(y.sum())}  negatives: {int((y==0).sum())}")
    print(f"[n1] features: {len(feats)}")
    print(f"[n1] running {args.n_seeds} HGB seeds x {args.n_folds}-fold stratified CV ...")

    per_seed: list[dict] = []
    for s in range(args.n_seeds):
        per_seed.append({"seed": s, **_run_one_seed(X, y, s, n_folds=args.n_folds)})
        nat = per_seed[-1]["natural"]
        bal = per_seed[-1]["balanced"]
        print(
            f"  seed={s:2d}  natural F1={nat['f1']:.3f}  AUROC={nat['auroc']:.3f}  "
            f"balanced F1={bal['f1']:.3f}  AUROC={bal['auroc']:.3f}"
        )

    def _agg(field: str, rate: str) -> dict[str, float]:
        vals = [s[rate][field] for s in per_seed]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    summary = {
        "csv": args.csv.name,
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "n_features": len(feats),
        "n_seeds": args.n_seeds,
        "n_folds": args.n_folds,
        "natural": {
            "f1": _agg("f1", "natural"),
            "auroc": _agg("auroc", "natural"),
            "recall": _agg("recall", "natural"),
            "fpr": _agg("fpr", "natural"),
        },
        "balanced": {
            "f1": _agg("f1", "balanced"),
            "auroc": _agg("auroc", "balanced"),
            "recall": _agg("recall", "balanced"),
            "fpr": _agg("fpr", "balanced"),
        },
        "per_seed": per_seed,
        "protocol": (
            "Stratified k-fold cross-validation (default: 5 folds); HGB "
            "with default hyperparameters; for each fold, a stratified "
            "1/7th validation slice of the training pool tunes the F1-"
            "optimal threshold, then the held-out test fold is scored. "
            "Balanced rate downsamples Normals in the test fold to match "
            "the number of anomalies. Reported mean/std are across "
            "n_seeds HGB random states."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[n1] wrote {args.output}")

    print("\n=== HGB BASELINE SUMMARY ===")
    print(f"  natural F1   mean={summary['natural']['f1']['mean']:.3f}  std={summary['natural']['f1']['std']:.3f}")
    print(f"  natural AUROC mean={summary['natural']['auroc']['mean']:.3f}")
    print(f"  balanced F1  mean={summary['balanced']['f1']['mean']:.3f}  std={summary['balanced']['f1']['std']:.3f}")
    print(f"  balanced AUROC mean={summary['balanced']['auroc']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
