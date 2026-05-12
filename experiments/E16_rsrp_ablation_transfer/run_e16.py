"""E16 — Anti-confound transfer ablation.

Tests whether the synthetic-only transfer failure on real Jamming is a
trivial RSRP-feature shortcut or a deeper operating-regime coverage gap.

For each of three feature subsets:
  * all_240  : full engineered feature set (baseline)
  * no_rsrp  : the 240 features with all 15 RSRP-derived features removed
  * no_level : the 240 features with the four absolute-level statistics
               (mean, median, min, max) removed across every KPI -> 176 feats

we run HistGradientBoostingClassifier on the fullscale split (seed=42 for
the split itself), with two training-pool compositions:

  * synth-only  : train_norm + train_synth_pool          (f = 0 calibration)
  * f10         : train_norm + train_synth_pool + f=0.10 of train_real_pool

10 model seeds per (subset, condition).  Threshold selection on a
deterministic 10% val carve-out (matches E4).  Output: one CSV with
per-seed metrics + one aggregated mean+/-std summary.

Drop-in for the §4.2 anti-confound table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))

from _shared.data_utils import (  # noqa: E402
    KPI_NAMES,
    exp_output_dir,
    feature_names,
    get_or_build_corpus_features,
    load_corpus,
    make_fullscale_split,
)

RESULTS = exp_output_dir("E16", "results")


def _positive_proba(clf, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        classes = list(getattr(clf, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]
    return np.zeros(proba.shape[0], dtype=float)


def fit_score_threshold(
    F: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int = 0,
) -> tuple[HistGradientBoostingClassifier, float]:
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(F[train_idx], y[train_idx])
    val_scores = _positive_proba(clf, F[val_idx])
    y_val = y[val_idx]
    if np.unique(y[train_idx]).size < 2 or y_val.sum() == 0 or y_val.sum() == y_val.size:
        return clf, 0.5
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    prec_a, rec_a = prec[:-1], rec[:-1]
    denom = prec_a + rec_a
    f1s = np.where(denom > 0, 2 * prec_a * rec_a / np.maximum(denom, 1e-12), 0.0)
    if f1s.size == 0:
        return clf, 0.5
    return clf, float(thr[int(np.argmax(f1s))])


def evaluate(
    clf,
    threshold: float,
    F: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    test_scores = _positive_proba(clf, F[test_idx])
    y_test = y[test_idx]
    pred = (test_scores >= threshold).astype(int)
    p = float(precision_score(y_test, pred, zero_division=0))
    r = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    if 0 < y_test.sum() < y_test.size:
        auroc = float(roc_auc_score(y_test, test_scores))
        avg_prec = float(average_precision_score(y_test, test_scores))
    else:
        auroc = float("nan")
        avg_prec = float("nan")
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "auroc": auroc,
        "avg_precision": avg_prec,
        "normal_fpr": float((test_scores[masks["normals"]] >= threshold).mean())
        if masks["normals"].any()
        else float("nan"),
        "real_jamming_detection_rate": float((test_scores[masks["real"]] >= threshold).mean())
        if masks["real"].any()
        else float("nan"),
        "synthetic_detection_rate": float((test_scores[masks["synthetic"]] >= threshold).mean())
        if masks["synthetic"].any()
        else float("nan"),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    print("[E16] loading corpus + features ...")
    corpus = load_corpus(verbose=True)
    F_full, feat_names = get_or_build_corpus_features(verbose=True)
    F_full = np.asarray(F_full, dtype=np.float32)
    assert F_full.shape[0] == corpus.n
    assert F_full.shape[1] == 240
    print(f"[E16] feature matrix: {F_full.shape}")

    # Feature subsets
    rsrp_idx = np.array([i for i, n in enumerate(feat_names) if n.startswith("RSRP__")])
    level_stats = ("mean", "median", "min", "max")
    level_idx = np.array(
        [i for i, n in enumerate(feat_names) if n.split("__", 1)[1] in level_stats]
    )
    print(
        f"[E16] subsets: all=240, no_rsrp drops {rsrp_idx.size} feats, "
        f"no_level drops {level_idx.size} feats"
    )

    SUBSETS: dict[str, np.ndarray] = {
        "all_240": np.arange(240),
        "no_rsrp": np.setdiff1d(np.arange(240), rsrp_idx),
        "no_level": np.setdiff1d(np.arange(240), level_idx),
    }

    # Fullscale split (matches E4)
    split = make_fullscale_split(corpus, seed=42)
    train_idx_all = np.asarray(split["train"])
    test_idx = np.asarray(split["test"])
    y_train_all = corpus.y[train_idx_all]
    train_remaining_pos, val_pos = train_test_split(
        np.arange(train_idx_all.size),
        test_size=0.10,
        stratify=y_train_all,
        random_state=0,
    )
    train_remaining = train_idx_all[train_remaining_pos]
    val_idx = train_idx_all[val_pos]

    y_remain = corpus.y[train_remaining]
    origin_remain = corpus.anomaly_origin[train_remaining]
    train_norm = train_remaining[y_remain == 0]
    train_real_pool = train_remaining[origin_remain == "real"]
    train_synth_pool = train_remaining[origin_remain == "synthetic"]
    print(
        f"[E16] pools: norm={train_norm.size}, "
        f"synth_pool={train_synth_pool.size}, real_pool={train_real_pool.size}"
    )

    y_test = corpus.y[test_idx]
    origin_test = corpus.anomaly_origin[test_idx]
    masks = {
        "normals": (y_test == 0),
        "real": (origin_test == "real"),
        "synthetic": (origin_test == "synthetic"),
    }
    print(
        f"[E16] test groups: norm={int(masks['normals'].sum())}, "
        f"real={int(masks['real'].sum())}, synth={int(masks['synthetic'].sum())}"
    )

    from _shared.data_utils import default_seeds  # noqa: E402
    SEEDS = list(range(default_seeds(10)))
    n_real_pool = int(train_real_pool.size)
    n_added_f10 = max(0, int(round(0.10 * n_real_pool)))
    print(f"[E16] f=10% calibration windows: {n_added_f10}")

    conditions = [
        ("synth_only", 0),
        ("f10_calibrated", n_added_f10),
    ]

    rows = []
    for subset_name, col_idx in SUBSETS.items():
        F_sub = F_full[:, col_idx]
        print(f"\n[E16] subset {subset_name}  shape={F_sub.shape}")
        for cond_name, n_added in conditions:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                if n_added > 0:
                    real_subset = rng.choice(train_real_pool, size=n_added, replace=False)
                else:
                    real_subset = np.empty(0, dtype=train_real_pool.dtype)
                train_idx = np.concatenate([train_norm, train_synth_pool, real_subset])
                clf, thr = fit_score_threshold(
                    F_sub, corpus.y, train_idx, val_idx, seed=seed
                )
                metrics = evaluate(clf, thr, F_sub, corpus.y, test_idx, masks)
                row = {
                    "subset": subset_name,
                    "n_features": int(col_idx.size),
                    "condition": cond_name,
                    "n_added_real": int(n_added),
                    "seed": int(seed),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"  {subset_name:10s} {cond_name:15s} seed={seed} "
                    f"F1={metrics['f1']:.3f} "
                    f"AUC={metrics['auroc']:.3f} "
                    f"real={metrics['real_jamming_detection_rate']:.3f} "
                    f"syn={metrics['synthetic_detection_rate']:.3f} "
                    f"fpr={metrics['normal_fpr']:.3f}"
                )

    per_seed = pd.DataFrame(rows)
    per_seed_csv = RESULTS / "E16_per_seed.csv"
    per_seed.to_csv(per_seed_csv, index=False)
    print(f"\n[E16] wrote {per_seed_csv}")

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "auroc",
        "avg_precision",
        "normal_fpr",
        "real_jamming_detection_rate",
        "synthetic_detection_rate",
    ]
    agg = (
        per_seed.groupby(["subset", "n_features", "condition", "n_added_real"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg_csv = RESULTS / "E16_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"[E16] wrote {agg_csv}")

    # Compact human-readable JSON for the paper
    compact = []
    for _, row in agg.iterrows():
        compact.append(
            {
                "subset": row["subset"],
                "n_features": int(row["n_features"]),
                "condition": row["condition"],
                "n_added_real": int(row["n_added_real"]),
                "f1": f"{row['f1_mean']:.3f}+-{row['f1_std']:.3f}",
                "auroc": f"{row['auroc_mean']:.3f}+-{row['auroc_std']:.3f}",
                "real_recall": f"{row['real_jamming_detection_rate_mean']:.3f}+-"
                f"{row['real_jamming_detection_rate_std']:.3f}",
                "synth_recall": f"{row['synthetic_detection_rate_mean']:.3f}+-"
                f"{row['synthetic_detection_rate_std']:.3f}",
                "normal_fpr": f"{row['normal_fpr_mean']:.3f}+-{row['normal_fpr_std']:.3f}",
            }
        )
    compact_path = RESULTS / "E16_compact.json"
    with compact_path.open("w") as f:
        json.dump(compact, f, indent=2)
    print(f"[E16] wrote {compact_path}")
    print(f"[E16] DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
