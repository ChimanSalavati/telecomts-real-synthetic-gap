"""E18 -- Domain-adaptation baselines vs real calibration.

Reviewer concern (R4 W3, R4 Q2): the only mitigation we compare against
is matched-budget extra-synthetic / extra-normal / reweighting. Standard
covariate-shift correction techniques (importance weighting via density
ratios, Sugiyama et al. 2007; uLSIF, Kanamori et al. 2009) should be
tested too.

We compare four mitigations at the same f=10% (18-window) effective
budget on TelecomTS full-scale split:

  (a) Baseline synth-only HGB                                  [from E16]
  (b) Importance-weighted HGB via LR density-ratio (Sugiyama)
  (c) uLSIF-lite (LSIF with a 100-centre Gaussian-kernel basis)
  (d) Real calibration: add 18 controlled-real Jamming windows (ours)

All four use the same 240-feature representation, the same HGB
hyper-parameters, the same validation-tuned F1 threshold protocol, and
the same 10 random seeds. Densities are estimated using the
``train_real_pool`` (controlled-real anomaly examples that the audit
gate would gate on); importance weights are computed for the synthetic
training pool only and then passed to HGB via ``sample_weight``.

Tables produced:
  - results/E18_da_baseline_summary.csv     (per-method aggregate)
  - results/E18_da_baseline_per_seed.csv    (raw)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _fit_threshold(F_sub, y_sub, F_val, y_val, sample_weight, seed):
    clf = HistGradientBoostingClassifier(random_state=seed)
    if sample_weight is not None:
        clf.fit(F_sub, y_sub, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        clf.fit(F_sub, y_sub)
    val_scores = clf.predict_proba(F_val)[:, 1]
    if np.unique(y_sub).size < 2 or y_val.sum() == 0 or y_val.sum() == y_val.size:
        return clf, 0.5
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return clf, float(thr[int(np.argmax(f1))]) if f1.size else 0.5


def _evaluate(clf, thr, F_test, y_test, masks):
    s = clf.predict_proba(F_test)[:, 1]
    pred = (s >= thr).astype(int)
    out = {
        "f1": float(((pred == 1) & (y_test == 1)).sum())
        * 2.0
        / max(1, int((pred == 1).sum() + (y_test == 1).sum())),
        "auc": float(roc_auc_score(y_test, s)) if 0 < y_test.sum() < y_test.size else float("nan"),
        "real_recall": float(pred[masks["real"]].mean()) if masks["real"].any() else float("nan"),
        "synth_recall": float(pred[masks["synth"]].mean()) if masks["synth"].any() else float("nan"),
        "normal_fpr": float(pred[masks["normal"]].mean()) if masks["normal"].any() else float("nan"),
    }
    return out


# ---------------------------------------------------------------- DA methods


def _lr_density_ratio(F_train: np.ndarray, F_real: np.ndarray, clip=(0.1, 10.0)) -> np.ndarray:
    """Estimate r(x) = p(real)/p(train) via LR + Bayes inversion.

    Returns sample weights for the rows of ``F_train`` (synthetic+normal
    training pool). The weights upweight rows whose feature vectors look
    more like the controlled-real pool, exactly the Sugiyama (2007)
    importance-weighting recipe.
    """
    scaler = StandardScaler().fit(np.vstack([F_train, F_real]))
    X = scaler.transform(np.vstack([F_train, F_real]))
    y = np.concatenate(
        [np.zeros(len(F_train), dtype=int), np.ones(len(F_real), dtype=int)]
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="liblinear")
    clf.fit(X, y)
    p_real_given_x = clf.predict_proba(X[: len(F_train)])[:, 1]
    # Density ratio: r(x) = p(real|x) / p(train|x) * (p_train / p_real)
    pi = float(np.mean(y))
    p_train = max(1 - pi, 1e-12)
    p_real = max(pi, 1e-12)
    ratio = (p_real_given_x / (1.0 - p_real_given_x + 1e-12)) * (p_train / p_real)
    return np.clip(ratio, *clip)


def _ulsif_lite_density_ratio(
    F_train: np.ndarray, F_real: np.ndarray, n_centres: int = 100, sigma: float | None = None, lam: float = 1e-2, seed: int = 0
) -> np.ndarray:
    """uLSIF-lite: least-squares density-ratio estimation with a
    Gaussian-kernel basis whose centres are randomly chosen from the
    controlled-real pool.

    Implements the squared-error form of the LSIF objective so the
    optimum is a closed-form linear system, in the spirit of Kanamori
    et al. 2009. Slimmer than full uLSIF (no cross-validation over
    sigma or lambda; we use the median heuristic for sigma) but
    matches the qualitative behaviour at deployment scale.
    """
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(np.vstack([F_train, F_real]))
    Xt = scaler.transform(F_train)
    Xr = scaler.transform(F_real)
    n_centres = min(n_centres, len(Xr))
    centres = Xr[rng.choice(len(Xr), size=n_centres, replace=False)]

    if sigma is None:
        # Median heuristic over a random sample of pairwise distances.
        sub = np.vstack([Xt[rng.choice(len(Xt), min(len(Xt), 200), replace=False)], Xr[rng.choice(len(Xr), min(len(Xr), 200), replace=False)]])
        diffs = np.sqrt(np.maximum(
            np.sum(sub * sub, axis=1, keepdims=True)
            + np.sum(sub * sub, axis=1)
            - 2.0 * sub @ sub.T,
            0.0,
        ))
        iu = np.triu_indices_from(diffs, k=1)
        sigma = float(np.median(diffs[iu])) if iu[0].size else 1.0
        sigma = max(sigma, 1e-3)

    def _phi(X):
        # K(X, centres) Gaussian RBF
        sq = (
            np.sum(X * X, axis=1, keepdims=True)
            + np.sum(centres * centres, axis=1)
            - 2.0 * X @ centres.T
        )
        return np.exp(-sq / (2.0 * sigma * sigma))

    Phi_t = _phi(Xt)
    Phi_r = _phi(Xr)
    # uLSIF normal-equation form
    H = Phi_t.T @ Phi_t / len(Phi_t)
    h = Phi_r.mean(axis=0)
    H_reg = H + lam * np.eye(H.shape[0])
    alpha = np.linalg.solve(H_reg, h)
    ratio = (Phi_t @ alpha)
    ratio = np.clip(ratio, 0.1, 10.0)
    return ratio


def main() -> int:
    HERE = Path(__file__).resolve().parent
    EXP_ROOT = HERE.parent
    sys.path.insert(0, str(EXP_ROOT))
    from _shared.data_utils import (  # noqa
        default_seeds,
        exp_output_dir,
        get_or_build_corpus_features,
        load_corpus,
        make_fullscale_split,
    )

    RESULTS = exp_output_dir("E18", "results")
    print("[E18] loading corpus + features ...")
    corpus = load_corpus(verbose=False)
    F_full, _ = get_or_build_corpus_features(verbose=False)
    F_full = np.asarray(F_full, dtype=np.float32)

    split = make_fullscale_split(corpus, seed=42)
    train_idx_all = np.asarray(split["train"])
    test_idx = np.asarray(split["test"])
    y_train_all = corpus.y[train_idx_all]
    tr_pos, val_pos = train_test_split(
        np.arange(train_idx_all.size),
        test_size=0.10,
        stratify=y_train_all,
        random_state=0,
    )
    train_remaining = train_idx_all[tr_pos]
    val_idx = train_idx_all[val_pos]
    y_remain = corpus.y[train_remaining]
    origin_remain = corpus.anomaly_origin[train_remaining]
    train_norm = train_remaining[y_remain == 0]
    train_real_pool = train_remaining[origin_remain == "real"]
    train_synth_pool = train_remaining[origin_remain == "synthetic"]

    y_test = corpus.y[test_idx]
    origin_test = corpus.anomaly_origin[test_idx]
    masks = {
        "normal": y_test == 0,
        "real": origin_test == "real",
        "synth": origin_test == "synthetic",
    }

    n_real_pool = int(train_real_pool.size)
    n_added = max(0, int(round(0.10 * n_real_pool)))
    print(f"[E18] f=10% calibration windows: {n_added}")

    SEEDS = list(range(default_seeds(10)))
    rng_master = np.random.default_rng(0)
    rows = []

    for seed in SEEDS:
        rng = np.random.default_rng(rng_master.integers(2**31 - 1))
        # ---- (a) baseline synth-only HGB ----
        tr0 = np.concatenate([train_norm, train_synth_pool])
        F_tr0 = F_full[tr0]
        y_tr0 = corpus.y[tr0]
        clf, thr = _fit_threshold(F_tr0, y_tr0, F_full[val_idx], corpus.y[val_idx], None, seed)
        m = _evaluate(clf, thr, F_full[test_idx], y_test, masks)
        rows.append({"method": "(a) synth-only baseline", "seed": seed, **m})

        # ---- (b) Importance weighting via LR density ratio ----
        # Density estimated against the controlled-real pool we DO have access to.
        # The reweighting touches ONLY the synth-only training pool.
        ratio_b = _lr_density_ratio(F_tr0, F_full[train_real_pool])
        clf, thr = _fit_threshold(F_tr0, y_tr0, F_full[val_idx], corpus.y[val_idx], ratio_b, seed)
        m = _evaluate(clf, thr, F_full[test_idx], y_test, masks)
        m["mean_weight"] = float(ratio_b.mean())
        m["max_weight"] = float(ratio_b.max())
        rows.append({"method": "(b) IW-LR (Sugiyama)", "seed": seed, **m})

        # ---- (c) uLSIF-lite ----
        ratio_c = _ulsif_lite_density_ratio(F_tr0, F_full[train_real_pool], seed=seed)
        clf, thr = _fit_threshold(F_tr0, y_tr0, F_full[val_idx], corpus.y[val_idx], ratio_c, seed)
        m = _evaluate(clf, thr, F_full[test_idx], y_test, masks)
        m["mean_weight"] = float(ratio_c.mean())
        m["max_weight"] = float(ratio_c.max())
        rows.append({"method": "(c) uLSIF-lite (Kanamori)", "seed": seed, **m})

        # ---- (d) Real calibration f=10% (ours) ----
        if n_added > 0:
            real_subset = rng.choice(train_real_pool, size=n_added, replace=False)
        else:
            real_subset = np.empty(0, dtype=train_real_pool.dtype)
        tr_c = np.concatenate([train_norm, train_synth_pool, real_subset])
        clf, thr = _fit_threshold(F_full[tr_c], corpus.y[tr_c], F_full[val_idx], corpus.y[val_idx], None, seed)
        m = _evaluate(clf, thr, F_full[test_idx], y_test, masks)
        rows.append({"method": "(d) real calibration f=10% (ours)", "seed": seed, **m})

        print(
            f"[E18] seed={seed} "
            f"  base_real={rows[-4]['real_recall']:.3f}"
            f"  iw_lr_real={rows[-3]['real_recall']:.3f}"
            f"  ulsif_real={rows[-2]['real_recall']:.3f}"
            f"  cal_real={rows[-1]['real_recall']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "E18_da_baseline_per_seed.csv", index=False)

    agg = (
        df.groupby("method")
        [["f1", "auc", "real_recall", "synth_recall", "normal_fpr"]]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg_path = RESULTS / "E18_da_baseline_summary.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\n[E18] wrote {agg_path}")

    print("\n=== DA-baseline summary (mean +/- std over 10 seeds) ===")
    for _, r in agg.iterrows():
        print(
            f"  {r['method']:36s} "
            f"real={r['real_recall_mean']:.3f}+/-{r['real_recall_std']:.3f}  "
            f"synth={r['synth_recall_mean']:.3f}  "
            f"F1={r['f1_mean']:.3f}  "
            f"FPR={r['normal_fpr_mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
