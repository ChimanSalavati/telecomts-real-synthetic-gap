#!/usr/bin/env python3
# AUTO-GENERATED from S1_spotlight_origin_distributional_check.ipynb by pipeline/convert_notebooks.py -- do not edit by hand.
# This is the executable Python conversion of the original Jupyter notebook.
# Standalone:  python experiments/<dir>/run_s1.py
# Via runner:  python main.py --experiment S1
"""S1: converted notebook runner (offline-aware, centralized outputs)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_EXP_ROOT = Path(__file__).resolve().parent.parent  # experiments/
_REPO_ROOT = _EXP_ROOT.parent                        # repo root (telecomts_gap/)
for _p in (_EXP_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("MPLBACKEND", "Agg")

from _shared.data_utils import exp_output_dir  # noqa: E402


if __name__ == "__main__":
    # Redirect the notebook's cwd-relative ./results, ./figures, ./manifests
    # into the centralized artifacts/S1/ tree.
    os.chdir(exp_output_dir("S1", ""))

    # Make _shared importable, set plot defaults, suppress noisy warnings.
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path('..').resolve()))
    from _shared.notebook_helpers import setup_paths, configure_matplotlib
    EXPERIMENTS_ROOT = setup_paths()
    configure_matplotlib()

    import json
    import time
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    warnings.filterwarnings('ignore')

    from scipy.stats import ks_2samp, wasserstein_distance
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics.pairwise import rbf_kernel

    from _shared.data_utils import cohens_d, benjamini_hochberg

    HERE = Path('.').resolve()
    RESULTS = HERE / 'results'
    FIGURES = HERE / 'figures'
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # SpotLight processed NPZ files (written by evaluation_ver2/SpotLight/prepare_spotlight_dataset.py)
    SPOT_DIR = (EXPERIMENTS_ROOT / '..' / '..' / 'evaluation_ver2' / 'SpotLight' / 'data').resolve()
    SPOT_VARIANT = 'paper5ue_single'   # 5-UE single-anomaly subset (excludes MIXED)
    SEED = 42
    print('SpotLight data dir :', SPOT_DIR)
    print('Working dir        :', HERE)

    # Concatenate train+val+test so we have the full anomaly population for the audit.
    # (The split was originally for downstream detector training; for a distributional
    # audit we want as many anomaly windows as possible.)
    def _load_split(split):
        p = SPOT_DIR / f'SpotLight_{SPOT_VARIANT}_{split}.npz'
        return np.load(p, allow_pickle=True)

    splits = {s: _load_split(s) for s in ('train', 'val', 'test')}
    feature_cols = list(splits['train']['feature_cols'])
    n_features = len(feature_cols)

    X_all = np.concatenate([splits[s]['X'] for s in ('train', 'val', 'test')], axis=0)
    y_all = np.concatenate([splits[s]['y'] for s in ('train', 'val', 'test')], axis=0)
    types_all = np.concatenate([splits[s]['anomaly_types'] for s in ('train', 'val', 'test')], axis=0)

    n_windows, win_len, n_chan = X_all.shape
    assert n_chan == n_features, (n_chan, n_features)
    print(f'Total SpotLight windows: {n_windows}  (window={win_len}*100ms = {win_len*0.1:.1f}s, channels={n_chan})')
    print('Anomaly types:')
    for t, c in zip(*np.unique(types_all, return_counts=True)):
        print(f'  {t:8s}: {c:4d} windows  (y=1 in this group: {int(y_all[types_all==t].sum())})')

    rf_mask    = (types_all == 'RADIO')
    nonrf_mask = np.isin(types_all, ['PDCP', 'MAC', 'NETWORK'])

    n_rf, n_nonrf = int(rf_mask.sum()), int(nonrf_mask.sum())
    print(f'RF (RADIO)              windows: {n_rf}')
    print(f'NonRF (PDCP/MAC/NETWORK) windows: {n_nonrf}')
    assert n_rf > 0 and n_nonrf > 0, 'origin groups must be non-empty'

    # Per-window mean of each channel: (n_windows, n_channels)
    W_mean = X_all.mean(axis=1)
    # Richer (mean, std, min, max) summary per channel: (n_windows, 4*n_channels)
    W_std  = X_all.std(axis=1)
    W_min  = X_all.min(axis=1)
    W_max  = X_all.max(axis=1)
    W_full = np.concatenate([W_mean, W_std, W_min, W_max], axis=1).astype(np.float64)

    # Drop columns that are constant across the union of RF + NonRF (they cannot contribute
    # to a two-sample test and would yield NaN Cohen's d).
    union_idx = np.where(rf_mask | nonrf_mask)[0]
    col_var = W_mean[union_idx].var(axis=0)
    keep_chan = np.where(col_var > 1e-12)[0]
    print(f'Channels with variance > 0 on RF+NonRF union: {keep_chan.size}/{n_features}')

    rng = np.random.default_rng(SEED)

    rf_idx    = np.where(rf_mask)[0]
    nonrf_idx = np.where(nonrf_mask)[0]

    records = []
    for j in keep_chan:
        a = W_mean[rf_idx,    j].astype(np.float64)
        b = W_mean[nonrf_idx, j].astype(np.float64)
        if np.std(a) == 0 and np.std(b) == 0:
            continue
        ks_stat, ks_p = ks_2samp(a, b)
        d           = cohens_d(a, b)
        # normalised wasserstein: divide by combined std so it's comparable across KPIs.
        pooled_std = np.std(np.concatenate([a, b])) + 1e-12
        wd = wasserstein_distance(a, b) / pooled_std
        records.append({
            'channel'    : feature_cols[j],
            'ks_stat'    : float(ks_stat),
            'ks_pvalue'  : float(ks_p),
            'cohens_d'   : float(d),
            'abs_d'      : float(abs(d)),
            'wasserstein_norm': float(wd),
            'mean_RF'    : float(a.mean()),
            'mean_NonRF' : float(b.mean()),
        })
    per_kpi = pd.DataFrame.from_records(records).sort_values('abs_d', ascending=False).reset_index(drop=True)

    # BH multiple-testing correction over channels.
    per_kpi['ks_significant_bh'] = benjamini_hochberg(per_kpi['ks_pvalue'].to_numpy(), alpha=0.05)
    print(f'Channels tested: {len(per_kpi)}')
    print(f'KS-significant after BH (alpha=0.05): {int(per_kpi["ks_significant_bh"].sum())}')
    print()
    print('Top 15 channels by |Cohen\'s d|:')
    print(per_kpi[['channel', 'ks_stat', 'cohens_d', 'wasserstein_norm', 'mean_RF', 'mean_NonRF']].head(15).to_string(index=False))

    # Save per-KPI effect-size table.
    per_kpi_path = RESULTS / 'S1_per_kpi_effect_sizes.csv'
    per_kpi.to_csv(per_kpi_path, index=False)
    print(f'Wrote {per_kpi_path}')

    # Highlight the SINR family explicitly (the SpotLight analogue of TelecomTS RSRP).
    sinr_rows = per_kpi[per_kpi['channel'].str.contains('sinr', case=False, na=False)]
    print()
    print(f'Signal-quality (SINR) channels among tested ({len(sinr_rows)}):')
    if len(sinr_rows):
        print(sinr_rows[['channel', 'ks_stat', 'cohens_d', 'wasserstein_norm']].to_string(index=False))
    else:
        print('  (none survived the variance filter)')

    # Build the (n_windows, 4*K) feature matrix on the kept channels only.
    feat_mask = np.concatenate([
        np.isin(np.arange(n_features), keep_chan),
        np.isin(np.arange(n_features), keep_chan),
        np.isin(np.arange(n_features), keep_chan),
        np.isin(np.arange(n_features), keep_chan),
    ])
    F = W_full[:, feat_mask]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    union_idx = np.where(rf_mask | nonrf_mask)[0]
    F_union = F[union_idx]
    y_union = rf_mask[union_idx].astype(int)   # 1 = RF, 0 = NonRF
    print(f'C2ST design matrix: F_union.shape={F_union.shape}, positives (RF)={int(y_union.sum())}, negatives (NonRF)={int((1-y_union).sum())}')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(F_union, y_union)):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED + fold)
        clf.fit(F_union[tr_idx], y_union[tr_idx])
        proba = clf.predict_proba(F_union[te_idx])[:, 1]
        yhat  = (proba >= 0.5).astype(int)
        fold_metrics.append({
            'fold'    : fold,
            'accuracy': float(accuracy_score(y_union[te_idx], yhat)),
            'auroc'   : float(roc_auc_score(y_union[te_idx], proba)),
        })

    c2st_df = pd.DataFrame(fold_metrics)
    print()
    print(c2st_df.to_string(index=False))
    print()
    print(f'C2ST mean accuracy = {c2st_df["accuracy"].mean():.3f}  (chance = 0.5)')
    print(f'C2ST mean AUROC    = {c2st_df["auroc"].mean():.3f}')

    # Standardise features for kernel-bandwidth stability.
    F_z = StandardScaler().fit_transform(F_union)

    def median_heuristic_gamma(X, max_n=400, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        n = X.shape[0]
        sub = rng.choice(n, size=min(max_n, n), replace=False)
        Xs = X[sub]
        # pairwise squared Euclidean distances
        sq = np.sum(Xs ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2.0 * (Xs @ Xs.T)
        d2 = d2[np.triu_indices_from(d2, k=1)]
        med = np.median(d2[d2 > 0])
        return float(1.0 / (med + 1e-12))

    def mmd2_rbf(X, Y, gamma):
        Kxx = rbf_kernel(X, X, gamma=gamma)
        Kyy = rbf_kernel(Y, Y, gamma=gamma)
        Kxy = rbf_kernel(X, Y, gamma=gamma)
        nx, ny = X.shape[0], Y.shape[0]
        # Unbiased estimator (drop diagonal of Kxx, Kyy)
        sxx = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
        syy = (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1))
        sxy = Kxy.mean()
        return float(sxx + syy - 2.0 * sxy)

    rng_mmd = np.random.default_rng(SEED)
    gamma = median_heuristic_gamma(F_z, max_n=400, rng=rng_mmd)
    F_rf, F_non = F_z[y_union == 1], F_z[y_union == 0]
    mmd_obs = mmd2_rbf(F_rf, F_non, gamma)

    # Permutation null
    N_PERM = 1000
    nx = F_rf.shape[0]
    F_pool = np.concatenate([F_rf, F_non], axis=0)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = rng_mmd.permutation(F_pool.shape[0])
        Xa = F_pool[perm[:nx]]
        Xb = F_pool[perm[nx:]]
        null[k] = mmd2_rbf(Xa, Xb, gamma)
    null_mean = float(null.mean())
    ratio = float(mmd_obs / max(null_mean, 1e-12))
    p_perm = float((1 + np.sum(null >= mmd_obs)) / (1 + N_PERM))

    print(f'Median-heuristic gamma     : {gamma:.4g}')
    print(f'Observed MMD^2             : {mmd_obs:.6g}')
    print(f'Permutation-null mean MMD^2: {null_mean:.6g}')
    print(f'Ratio observed / null      : {ratio:.2f}x')
    print(f'Permutation p-value (1000) : {p_perm:.4f}')

    summary = {
        'corpus'                : 'SpotLight (paper5ue_single)',
        'n_RF_radio_windows'    : int(rf_mask.sum()),
        'n_NonRF_software_windows': int(nonrf_mask.sum()),
        'n_channels_total'      : int(n_features),
        'n_channels_with_variance': int(len(keep_chan)),
        'n_ks_significant_bh'   : int(per_kpi['ks_significant_bh'].sum()),
        'top_channel_by_abs_d'  : str(per_kpi.iloc[0]['channel']),
        'top_abs_d'             : float(per_kpi.iloc[0]['abs_d']),
        'top_ks_stat'           : float(per_kpi.iloc[0]['ks_stat']),
        'top_wasserstein_norm'  : float(per_kpi.iloc[0]['wasserstein_norm']),
        'sinr_top_abs_d'        : float(sinr_rows['cohens_d'].abs().max()) if len(sinr_rows) else float('nan'),
        'sinr_top_ks_stat'      : float(sinr_rows['ks_stat'].max()) if len(sinr_rows) else float('nan'),
        'c2st_mean_accuracy'    : float(c2st_df['accuracy'].mean()),
        'c2st_std_accuracy'     : float(c2st_df['accuracy'].std()),
        'c2st_mean_auroc'       : float(c2st_df['auroc'].mean()),
        'c2st_std_auroc'        : float(c2st_df['auroc'].std()),
        'mmd_observed'          : mmd_obs,
        'mmd_null_mean'         : null_mean,
        'mmd_ratio'             : ratio,
        'mmd_perm_pvalue'       : p_perm,
        'mmd_n_permutations'    : int(N_PERM),
        'seed'                  : int(SEED),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = RESULTS / 'S1_two_sample_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Wrote {summary_path}')
    print()
    print(json.dumps(summary, indent=2))

    TOP_N = 15
    top = per_kpi.head(TOP_N).iloc[::-1].reset_index(drop=True)  # reverse for horizontal bar plot

    fig, ax = plt.subplots(figsize=(5.0, 0.32 * TOP_N + 0.6))
    ypos = np.arange(len(top))
    bars = ax.barh(ypos, top['cohens_d'], color='#3a7ca5', edgecolor='#222', linewidth=0.5)
    # Highlight SINR-family bars in a different colour.
    for i, name in enumerate(top['channel']):
        if 'sinr' in name.lower():
            bars[i].set_color('#d62728')
    ax.set_yticks(ypos)
    ax.set_yticklabels(top['channel'], fontsize=7)
    ax.set_xlabel("Cohen's d  (RADIO  vs  PDCP/MAC/NETWORK)")
    ax.set_title('SpotLight: top per-channel effect sizes between\ncontrolled-real RF (RADIO) and perturbation-synthetic origins')
    ax.axvline(0, color='k', linewidth=0.6)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig_path = FIGURES / 'S1_top_kpi_effect_sizes.pdf'
    fig.savefig(fig_path)
    plt.show()
    print(f'Wrote {fig_path}')

    sinr_max_d  = float(sinr_rows['cohens_d'].abs().max()) if len(sinr_rows) else float('nan')
    sinr_max_ks = float(sinr_rows['ks_stat'].max()) if len(sinr_rows) else float('nan')
    top_channel = str(per_kpi.iloc[0]['channel'])
    top_abs_d   = float(per_kpi.iloc[0]['abs_d'])
    n_sig_bh    = int(per_kpi['ks_significant_bh'].sum())

    print(f'''On SpotLight ({int(rf_mask.sum())} RADIO windows vs {int(nonrf_mask.sum())} PDCP/MAC/NETWORK windows),
    {n_sig_bh}/{len(per_kpi)} channels show a KS-significant difference between the controlled-real
    RF anomaly origin and the perturbation-synthetic origins after Benjamini-Hochberg correction.
    The dominant separator is "{top_channel}" with |Cohen's d| = {top_abs_d:.2f}; the SINR-family channels
    (the SpotLight analogue of TelecomTS RSRP) reach |Cohen's d| up to {sinr_max_d:.2f} and KS up to {sinr_max_ks:.2f}.
    A 5-fold HGB classifier two-sample test reaches accuracy = {summary["c2st_mean_accuracy"]:.3f} (chance = 0.5)
    and AUROC = {summary["c2st_mean_auroc"]:.3f}; the RBF-MMD test rejects equality of the two distributions
    with the observed MMD running {summary["mmd_ratio"]:.1f}x the permutation-null mean (p = {summary["mmd_perm_pvalue"]:.4f}, 1000 perms).
    This replicates, on an independent 5G testbed, the operating-regime separation we observe between
    real Jamming and synthetic anomalies on TelecomTS, and so supports the interpretation that the gap is
    intrinsic to RF vs non-RF anomaly origins rather than a TelecomTS-specific generator artifact.''')
