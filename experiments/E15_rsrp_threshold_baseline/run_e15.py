"""E15 (M3): Trivial single-feature RSRP threshold baseline + PHY-only HGB on Full-6.4k/synth.

Addresses reviewer concern M3: given that real Jamming has Cohen's d ~3 on RSRP
and KS = 1.0, what does a univariate threshold on RSRP do?

Decision rules evaluated on the held-out fullscale test set:
  1. mean(RSRP_window) < tau, for tau in {-100, -95, -90, -85} dBm
  2. min(RSRP_window) < tau, for the same grid
  3. PHY-only HGB: HistGradientBoosting trained on the 30 PHY features
     (15 stats x {RSRP, UL_SNR}), trained synthetic-only and tested on all
     test windows -- isolates whether RSRP-driven engineered features alone
     are enough.

The same fullscale split builder used by E3 / E9 is reused so the test set is
identical, allowing the new rows to be appended directly to Table 1 in the paper.

Run::
    python experiments/E15_rsrp_threshold_baseline/run_e15.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_helpers import setup_paths  # noqa: E402
setup_paths()

from _shared.data_utils import (  # noqa: E402
    load_corpus,
    get_or_build_corpus_features,
    feature_names,
    feature_indices_for_kpis,
    make_fullscale_split,
    KPI_NAMES,
    kpi_indices,
    exp_output_dir,
)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

RESULTS = exp_output_dir("E15", "results")
OUT_CSV = RESULTS / "E15_rsrp_threshold_summary.csv"

from _shared.data_utils import default_seeds  # noqa: E402

THRESHOLDS_DBM = [-100, -95, -90, -85]
SEEDS = list(range(default_seeds(10)))


def threshold_rule_metrics(scores: np.ndarray, mask_pos: np.ndarray, mask_norm: np.ndarray,
                           y: np.ndarray, real_mask: np.ndarray, synth_mask: np.ndarray,
                           threshold_negate: bool, threshold: float) -> dict:
    """Return F1 + per-origin recalls for a univariate thresholding rule.

    threshold_negate: True means anomaly := score < threshold (RSRP rule);
                      False means anomaly := score >= threshold.
    """
    pred = (scores < threshold).astype(int) if threshold_negate else (scores >= threshold).astype(int)
    f1 = f1_score(y, pred, zero_division=0) if y.sum() else float("nan")
    real_recall  = float(pred[real_mask].mean())   if real_mask.sum() else float("nan")
    synth_recall = float(pred[synth_mask].mean())  if synth_mask.sum() else float("nan")
    normal_fpr   = float(pred[mask_norm].mean())   if mask_norm.sum() else float("nan")
    return {
        "f1"          : float(f1),
        "real_recall" : real_recall,
        "synth_recall": synth_recall,
        "normal_fpr"  : normal_fpr,
    }


def main() -> None:
    print("Loading corpus and engineered features (cached) ...")
    corpus = load_corpus(verbose=True)
    F_full, feat_names = get_or_build_corpus_features(verbose=True)
    assert F_full.shape == (corpus.n, 240)

    rsrp_idx = KPI_NAMES.index("RSRP")
    win_mean_rsrp = corpus.X[:, :, rsrp_idx].mean(axis=1)
    win_min_rsrp  = corpus.X[:, :, rsrp_idx].min(axis=1)

    sp = make_fullscale_split(corpus, seed=42)
    test_idx = sp["test"]
    y_test = corpus.y[test_idx]
    real_mask  = (corpus.anomaly_origin[test_idx] == "real")
    synth_mask = (corpus.anomaly_origin[test_idx] == "synthetic")
    norm_mask  = (corpus.y[test_idx] == 0)

    rows = []

    # 1. Univariate mean(RSRP) < tau
    print("\n[1] mean(RSRP) < tau ...")
    for tau in THRESHOLDS_DBM:
        scores = win_mean_rsrp[test_idx]
        m = threshold_rule_metrics(scores, None, norm_mask, y_test, real_mask, synth_mask,
                                    threshold_negate=True, threshold=tau)
        # AUROC for the underlying signed score: lower RSRP = more anomalous
        try:
            auroc = float(roc_auc_score(y_test, -scores))
        except Exception:
            auroc = float("nan")
        rows.append({"detector": f"mean(RSRP) < {tau}", **m, "auroc": auroc, "seed": -1})
        print(f"   tau = {tau:>4d}: F1 = {m['f1']:.3f}, real = {m['real_recall']:.3f}, "
              f"synth = {m['synth_recall']:.3f}, FPR = {m['normal_fpr']:.3f}, AUROC = {auroc:.3f}")

    # 2. Univariate min(RSRP) < tau
    print("\n[2] min(RSRP) < tau ...")
    for tau in THRESHOLDS_DBM:
        scores = win_min_rsrp[test_idx]
        m = threshold_rule_metrics(scores, None, norm_mask, y_test, real_mask, synth_mask,
                                    threshold_negate=True, threshold=tau)
        try:
            auroc = float(roc_auc_score(y_test, -scores))
        except Exception:
            auroc = float("nan")
        rows.append({"detector": f"min(RSRP) < {tau}", **m, "auroc": auroc, "seed": -1})
        print(f"   tau = {tau:>4d}: F1 = {m['f1']:.3f}, real = {m['real_recall']:.3f}, "
              f"synth = {m['synth_recall']:.3f}, FPR = {m['normal_fpr']:.3f}, AUROC = {auroc:.3f}")

    # 3. PHY-only HGB trained synthetic-only (same regime as the failing E9 column)
    phy_idx = feature_indices_for_kpis(feature_names(), ["RSRP", "UL_SNR"])
    print(f"\n[3] PHY-only HGB (synthetic-only training, {phy_idx.size} features) ...")
    train_pool = sp["train"]
    # Drop real-Jamming windows from training to mimic the Full-6.4k/synth regime in Table 1.
    keep = (corpus.anomaly_origin[train_pool] != "real")
    train_idx = train_pool[keep]
    for seed in SEEDS:
        y_train = corpus.y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        # Stratified validation carve-out for threshold selection (same protocol as E9, M4 doc).
        train_inner, val_idx = train_test_split(
            train_idx, test_size=0.10, stratify=y_train, random_state=seed,
        )
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        clf.fit(F_full[train_inner][:, phy_idx], corpus.y[train_inner])
        val_p = clf.predict_proba(F_full[val_idx][:, phy_idx])[:, 1]
        test_p = clf.predict_proba(F_full[test_idx][:, phy_idx])[:, 1]
        # Best-F1 threshold on val.
        thr_grid = np.linspace(0.01, 0.99, 99)
        f1_val = [f1_score(corpus.y[val_idx], (val_p >= t).astype(int), zero_division=0) for t in thr_grid]
        thr = float(thr_grid[int(np.argmax(f1_val))])
        pred = (test_p >= thr).astype(int)
        f1   = f1_score(y_test, pred, zero_division=0) if y_test.sum() else float("nan")
        try:
            auroc = float(roc_auc_score(y_test, test_p))
        except Exception:
            auroc = float("nan")
        rows.append({
            "detector"    : f"HGB on PHY-only ({phy_idx.size} feats), synth-only train",
            "seed"        : seed,
            "threshold"   : thr,
            "f1"          : float(f1),
            "real_recall" : float(pred[real_mask].mean())   if real_mask.sum() else float("nan"),
            "synth_recall": float(pred[synth_mask].mean())  if synth_mask.sum() else float("nan"),
            "normal_fpr"  : float(pred[norm_mask].mean())   if norm_mask.sum() else float("nan"),
            "auroc"       : auroc,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\n--- Aggregated summary (PHY-HGB averaged over seeds) ---")
    out_rows = []
    for det, g in df.groupby("detector", sort=False):
        if g["seed"].iloc[0] == -1:
            r = g.iloc[0].to_dict()
            out_rows.append({
                "detector"    : det,
                "n_seeds"     : 1,
                "f1_mean"     : r["f1"],     "f1_std"     : 0.0,
                "real_mean"   : r["real_recall"], "real_std"   : 0.0,
                "synth_mean"  : r["synth_recall"], "synth_std"  : 0.0,
                "normal_fpr_mean": r["normal_fpr"], "normal_fpr_std": 0.0,
                "auroc_mean"  : r["auroc"],  "auroc_std"  : 0.0,
            })
        else:
            out_rows.append({
                "detector"     : det,
                "n_seeds"      : int(g["seed"].nunique()),
                "f1_mean"      : float(g["f1"].mean()),         "f1_std"      : float(g["f1"].std()),
                "real_mean"    : float(g["real_recall"].mean()), "real_std"   : float(g["real_recall"].std()),
                "synth_mean"   : float(g["synth_recall"].mean()),"synth_std"  : float(g["synth_recall"].std()),
                "normal_fpr_mean": float(g["normal_fpr"].mean()),"normal_fpr_std": float(g["normal_fpr"].std()),
                "auroc_mean"   : float(g["auroc"].mean()),       "auroc_std"  : float(g["auroc"].std()),
            })
    summary = pd.DataFrame(out_rows)
    summary_path = RESULTS / "E15_rsrp_threshold_aggregated.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
