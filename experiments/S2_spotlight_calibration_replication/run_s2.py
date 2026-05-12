"""S2: SpotLight cross-corpus replication of the TelecomTS TSTR + calibration recipe.

Mirrors the TelecomTS protocol (E3/E4/E10) on the independent SpotLight Open RAN
corpus by mapping its labelled categories to the same controlled-real-RF vs.
perturbation-synthetic contrast we audit on TelecomTS:

  - RADIO   <--> TelecomTS Jamming (controlled-real RF, USRP at 70-75 dB gain)
  - PDCP/MAC/NETWORK <--> the ten TelecomTS perturbation-synthetic types
  - NORMAL  <--> baseline traffic

Protocol per seed in {0..9}:
  1. Take the union of the released train+val splits as the training pool, and
     the released test split as the held-out test set (so the 35 RADIO test
     windows are never seen in training).
  2. Build a (mean, std, min, max) per-channel feature matrix on the 452 KPI
     channels = 1808 features per window.
  3. Synthetic-only baseline (f=0): train HGB on Normal + non-RADIO anomalies
     only, test on the full test set; report binary F1, Real (RADIO) recall,
     Synth (non-RADIO) recall, Normal FPR, AUROC.
  4. Calibration sweep: for each f in {0.01, 0.05, 0.10, 0.25, 0.50, 1.00},
     add the first n_added = floor(f * 206) train-eligible RADIO windows back
     to training (the train+val RADIO pool has 206 windows), retrain, and
     re-evaluate on the same fixed test set.

Outputs:
    results/S2_spotlight_calibration_per_seed.csv
    results/S2_spotlight_calibration_summary.csv
    figures/S2_spotlight_calibration_curve.pdf

Run:
    python experiments/S2_spotlight_calibration_replication/run_s2.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_helpers import setup_paths, configure_matplotlib  # noqa: E402
setup_paths()
configure_matplotlib()

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from _shared.data_utils import exp_output_dir  # noqa: E402

RESULTS = exp_output_dir("S2", "results")
FIGURES = exp_output_dir("S2", "figures")
PER_SEED_CSV = RESULTS / "S2_spotlight_calibration_per_seed.csv"
SUMMARY_CSV  = RESULTS / "S2_spotlight_calibration_summary.csv"
FIG_PDF      = FIGURES / "S2_spotlight_calibration_curve.pdf"

SPOT_DIR = (EXP_ROOT / ".." / ".." / "evaluation_ver2" / "SpotLight" / "data").resolve()
SPOT_VARIANT = "paper5ue_single"
FRACTIONS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SEEDS = list(range(10))


def _load_npz(split: str):
    p = SPOT_DIR / f"SpotLight_{SPOT_VARIANT}_{split}.npz"
    return np.load(p, allow_pickle=True)


def select_threshold(val_p: np.ndarray, y_val: np.ndarray) -> float:
    if y_val.sum() == 0 or (1 - y_val).sum() == 0:
        return 0.5
    grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = f1_score(y_val, (val_p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def main() -> None:
    print(f"Loading SpotLight from {SPOT_DIR} ...")
    train = _load_npz("train")
    val   = _load_npz("val")
    test  = _load_npz("test")

    feature_cols = list(train["feature_cols"])
    n_features = len(feature_cols)
    print(f"  channels: {n_features}")

    # Pool train + val into a single training pool; keep test held out.
    X_pool = np.concatenate([train["X"], val["X"]], axis=0)
    y_pool = np.concatenate([train["y"], val["y"]], axis=0).astype(int)
    types_pool = np.concatenate([train["anomaly_types"], val["anomaly_types"]], axis=0)
    X_test = test["X"]
    y_test = test["y"].astype(int)
    types_test = test["anomaly_types"]
    print(f"  pool windows: {X_pool.shape[0]}  (anomalies = {int(y_pool.sum())})")
    print(f"  test windows: {X_test.shape[0]}  (anomalies = {int(y_test.sum())})")
    print(f"  test composition by category: {dict(zip(*np.unique(types_test, return_counts=True)))}")

    # Build feature matrices: (mean, std, min, max) per channel.
    def featurize(X):
        return np.concatenate([X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1)], axis=1)

    F_pool = featurize(X_pool).astype(np.float64)
    F_test = featurize(X_test).astype(np.float64)
    F_pool = np.nan_to_num(F_pool, nan=0.0, posinf=0.0, neginf=0.0)
    F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  feature dim: {F_pool.shape[1]}")

    # Pool indices by category.
    is_radio_pool = (types_pool == "RADIO")
    is_other_anom_pool = (y_pool == 1) & ~is_radio_pool
    is_normal_pool = (y_pool == 0)
    radio_pool_idx = np.where(is_radio_pool)[0]
    other_anom_pool_idx = np.where(is_other_anom_pool)[0]
    normal_pool_idx = np.where(is_normal_pool)[0]
    print(f"  pool RADIO     : {radio_pool_idx.size}")
    print(f"  pool non-RADIO : {other_anom_pool_idx.size}")
    print(f"  pool Normal    : {normal_pool_idx.size}")

    # Test masks.
    test_radio_mask = (types_test == "RADIO")
    test_synth_mask = (y_test == 1) & ~test_radio_mask
    test_norm_mask  = (y_test == 0)

    rows = []
    for f in FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            radio_perm = radio_pool_idx[rng.permutation(radio_pool_idx.size)]
            n_inject = int(round(f * radio_pool_idx.size))
            inject_idx = radio_perm[:n_inject]
            base_idx = np.concatenate([normal_pool_idx, other_anom_pool_idx])
            train_idx = np.concatenate([base_idx, inject_idx])
            rng.shuffle(train_idx)
            y_train = y_pool[train_idx]
            if len(np.unique(y_train)) < 2:
                continue
            # Stratified val carve-out for threshold selection.
            train_inner_idx, val_inner_idx = train_test_split(
                train_idx, test_size=0.10, stratify=y_train, random_state=seed,
            )
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
            clf.fit(F_pool[train_inner_idx], y_pool[train_inner_idx])
            val_p  = clf.predict_proba(F_pool[val_inner_idx])[:, 1]
            test_p = clf.predict_proba(F_test)[:, 1]
            thr = select_threshold(val_p, y_pool[val_inner_idx])
            pred = (test_p >= thr).astype(int)
            f1   = f1_score(y_test, pred, zero_division=0) if y_test.sum() else float("nan")
            try:
                auroc = float(roc_auc_score(y_test, test_p))
            except Exception:
                auroc = float("nan")
            real_recall  = float(pred[test_radio_mask].mean())  if test_radio_mask.sum() else float("nan")
            synth_recall = float(pred[test_synth_mask].mean())  if test_synth_mask.sum() else float("nan")
            normal_fpr   = float(pred[test_norm_mask].mean())   if test_norm_mask.sum() else float("nan")
            rows.append({
                "fraction": f,
                "n_injected_radio": int(n_inject),
                "seed": seed,
                "n_train": int(train_inner_idx.size),
                "threshold": float(thr),
                "f1": float(f1),
                "auroc": auroc,
                "real_recall_radio": real_recall,
                "synth_recall_non_radio": synth_recall,
                "normal_fpr": normal_fpr,
            })
            print(f"  f = {f:>5.2f}  seed = {seed}  -- "
                  f"F1 = {f1:.3f}  AUROC = {auroc:.3f}  RADIO = {real_recall:.3f}  "
                  f"non-RADIO = {synth_recall:.3f}  FPR = {normal_fpr:.3f}")

    per_seed = pd.DataFrame.from_records(rows)
    per_seed.to_csv(PER_SEED_CSV, index=False)
    print(f"\nWrote {PER_SEED_CSV}")

    # Bootstrap CIs over seeds (2000 resamples).
    rng = np.random.default_rng(42)
    summary_rows = []
    for f, g in per_seed.groupby("fraction"):
        rates = g["real_recall_radio"].to_numpy(dtype=float)
        n = rates.size
        boots = np.empty(2000)
        for b in range(2000):
            boots[b] = rng.choice(rates, size=n, replace=True).mean()
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        summary_rows.append({
            "fraction"            : float(f),
            "n_injected_radio_avg": float(g["n_injected_radio"].mean()),
            "n_seeds"             : int(n),
            "real_recall_mean"    : float(rates.mean()),
            "real_recall_std"     : float(rates.std(ddof=0)),
            "real_recall_min"     : float(rates.min()),
            "real_recall_ci_lo"   : float(ci_lo),
            "real_recall_ci_hi"   : float(ci_hi),
            "synth_recall_mean"   : float(g["synth_recall_non_radio"].mean()),
            "synth_recall_std"    : float(g["synth_recall_non_radio"].std(ddof=0)),
            "normal_fpr_mean"     : float(g["normal_fpr"].mean()),
            "f1_mean"             : float(g["f1"].mean()),
            "auroc_mean"          : float(g["auroc"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("fraction")
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}\n")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))

    # Figure: side-by-side plots of RADIO recall and non-RADIO recall vs f.
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    fracs = summary["fraction"].to_numpy() * 100
    real_mean = summary["real_recall_mean"].to_numpy() * 100
    real_ci_lo = summary["real_recall_ci_lo"].to_numpy() * 100
    real_ci_hi = summary["real_recall_ci_hi"].to_numpy() * 100
    synth_mean = summary["synth_recall_mean"].to_numpy() * 100
    fpr_mean = summary["normal_fpr_mean"].to_numpy() * 100
    ax.fill_between(fracs, real_ci_lo, real_ci_hi, color="#d62728", alpha=0.18,
                    label="RADIO 95% bootstrap CI")
    ax.plot(fracs, real_mean, "o-", color="#d62728", lw=1.4, ms=4,
            label="RADIO recall (controlled-real RF)")
    ax.plot(fracs, synth_mean, "s--", color="#1f77b4", lw=1.2, ms=4,
            label="non-RADIO recall (synthetic)")
    ax.plot(fracs, fpr_mean, "x:", color="#444", lw=1.0, ms=5,
            label="Normal FPR")
    ax.set_xlabel("% of train-eligible RADIO pool added to training")
    ax.set_ylabel("Rate (%)")
    ax.set_title("SpotLight: cross-corpus calibration replication")
    ax.set_ylim(-2, 102)
    ax.set_xticks([0, 5, 10, 25, 50, 100])
    ax.grid(linestyle=":", alpha=0.45)
    ax.legend(fontsize=7, loc="center right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"\nWrote {FIG_PDF}")


if __name__ == "__main__":
    main()
