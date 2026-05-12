#!/usr/bin/env python3
# AUTO-GENERATED from E9b_leave_one_anomaly_out_audit.ipynb by pipeline/convert_notebooks.py -- do not edit by hand.
# This is the executable Python conversion of the original Jupyter notebook.
# Standalone:  python experiments/<dir>/run_e9b.py
# Via runner:  python main.py --experiment E9b
"""E9b: converted notebook runner (offline-aware, centralized outputs)."""
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
    # into the centralized artifacts/E9b/ tree.
    os.chdir(exp_output_dir("E9b", ""))

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

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    from _shared.data_utils import (
        load_corpus,
        get_or_build_corpus_features,
        ANOMALY_TYPES,
    )

    HERE = Path('.').resolve()
    RESULTS = HERE / 'results'
    FIGURES = HERE / 'figures'
    TABLES = HERE / 'tables'
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    print('Working dir:', HERE)

    corpus = load_corpus(verbose=True)
    F_full, feat_names = get_or_build_corpus_features(verbose=True)
    assert F_full.shape == (corpus.n, 240)
    print(f'corpus.n     = {corpus.n}')
    print(f'F_full.shape = {F_full.shape}')

    # Per-type counts to size the experiment.
    counts = pd.Series(corpus.anomaly_type[corpus.y == 1]).value_counts().reindex(ANOMALY_TYPES, fill_value=0)
    print()
    print('Per-anomaly-type counts:')
    print(counts.to_string())
    print(f'\nTotal anomalies: {int(counts.sum())} ; total normals: {int((corpus.y == 0).sum())}')

    def build_loo_split(held_out_type: str, seed: int):
        rng = np.random.default_rng(seed)
        is_T     = (corpus.anomaly_type == held_out_type) & (corpus.y == 1)
        is_other = (corpus.y == 1) & ~is_T
        is_norm  = (corpus.y == 0)

        T_idx     = np.where(is_T)[0];     rng.shuffle(T_idx)
        other_idx = np.where(is_other)[0]; rng.shuffle(other_idx)
        norm_idx  = np.where(is_norm)[0];  rng.shuffle(norm_idx)

        n_T = T_idx.size
        n_T_test       = max(int(round(0.8 * n_T)), 1)
        T_test         = T_idx[:n_T_test]
        T_train_pool   = T_idx[n_T_test:]   # not used by the default LOO regime

        n_other = other_idx.size
        n_other_train = max(int(round(0.8 * n_other)), 1)
        other_train   = other_idx[:n_other_train]
        other_test    = other_idx[n_other_train:]

        # Balance: roughly 50% positive rate on train; balance test normals to total anomaly count.
        n_norm_train = min(n_other_train, norm_idx.size)
        train_norm   = norm_idx[:n_norm_train]
        n_norm_test  = min(n_T_test + other_test.size, norm_idx.size - n_norm_train)
        test_norm    = norm_idx[n_norm_train:n_norm_train + n_norm_test]

        # Default LOO regime: train = normals + (anomalies != T)
        train_pool = np.concatenate([train_norm, other_train])
        rng.shuffle(train_pool)
        y_pool     = corpus.y[train_pool]
        # Stratified val carve-out (15%)
        train_idx, val_idx = train_test_split(
            train_pool, test_size=0.15, stratify=y_pool, random_state=seed,
        )
        test_idx = np.concatenate([test_norm, T_test, other_test])
        rng.shuffle(test_idx)
        return {
            'train': train_idx, 'val': val_idx, 'test': test_idx,
            'T_train_pool': T_train_pool, 'T_test': T_test, 'other_test': other_test,
        }


    def select_threshold(val_p, y_val):
        if y_val.sum() == 0 or (1 - y_val).sum() == 0:
            return 0.5
        grid = np.linspace(0.01, 0.99, 99)
        best_thr, best_f1 = 0.5, -1.0
        for thr in grid:
            f1 = f1_score(y_val, (val_p >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr


    def fit_hgb(X_tr, y_tr, X_val, X_test, seed):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]

    SEEDS = list(range(10))
    rows = []
    t0 = time.time()
    for T in ANOMALY_TYPES:
        for seed in SEEDS:
            sp = build_loo_split(T, seed)
            train_idx, val_idx, test_idx = sp['train'], sp['val'], sp['test']
            y_tr = corpus.y[train_idx]; y_val = corpus.y[val_idx]; y_test = corpus.y[test_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            val_p, test_p = fit_hgb(F_full[train_idx], y_tr, F_full[val_idx], F_full[test_idx], seed)
            thr = select_threshold(val_p, y_val)
            pred = (test_p >= thr).astype(int)
            # Subgroup masks within test
            t_types = corpus.anomaly_type[test_idx]
            is_T_test     = (t_types == T) & (y_test == 1)
            is_other_test = (y_test == 1) & ~is_T_test
            is_norm_test  = (y_test == 0)
            # Held-out-T recall (the headline metric)
            held_recall = float((test_p[is_T_test] >= thr).mean()) if is_T_test.sum() else float('nan')
            # Average recall over the OTHER trained-on types' positives
            trained_recall = float((test_p[is_other_test] >= thr).mean()) if is_other_test.sum() else float('nan')
            normal_fpr   = float((test_p[is_norm_test] >= thr).mean()) if is_norm_test.sum() else float('nan')
            f1_val = f1_score(y_test, pred, zero_division=0) if y_test.sum() else float('nan')
            try:
                auroc = float(roc_auc_score(y_test, test_p))
            except Exception:
                auroc = float('nan')
            rows.append({
                'held_out_type': T, 'seed': seed,
                'n_train': int(train_idx.size), 'n_val': int(val_idx.size), 'n_test': int(test_idx.size),
                'n_T_test': int(is_T_test.sum()), 'n_other_test': int(is_other_test.sum()),
                'n_norm_test': int(is_norm_test.sum()),
                'threshold': thr,
                'holdout_recall': held_recall,
                'trained_recall_avg': trained_recall,
                'normal_fpr': normal_fpr,
                'f1': float(f1_val), 'auroc': auroc,
            })
    elapsed = time.time() - t0
    print(f'Done LOO sweep: {len(rows)} rows in {elapsed:.1f}s')

    per_seed = pd.DataFrame.from_records(rows)
    per_seed_path = RESULTS / 'E9b_leave_one_out_per_seed.csv'
    per_seed.to_csv(per_seed_path, index=False)
    print(f'Wrote {per_seed_path}')

    def boot_ci_mean(values, n_boot=2000, seed=0, alpha=0.05):
        rng = np.random.default_rng(seed)
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            return float('nan'), float('nan'), float('nan')
        boots = np.empty(n_boot)
        for i in range(n_boot):
            boots[i] = rng.choice(v, size=v.size, replace=True).mean()
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(v.mean()), float(lo), float(hi)


    summary_rows = []
    for T, g in per_seed.groupby('held_out_type'):
        held_mean, held_lo, held_hi = boot_ci_mean(g['holdout_recall'], seed=hash(T) % (2**31))
        summary_rows.append({
            'held_out_type'      : T,
            'n_seeds'            : int(g['seed'].nunique()),
            'n_T_test_avg'       : float(g['n_T_test'].mean()),
            'holdout_recall_mean': held_mean,
            'holdout_recall_std' : float(g['holdout_recall'].std()),
            'holdout_recall_ci_lo': held_lo,
            'holdout_recall_ci_hi': held_hi,
            'trained_recall_avg' : float(g['trained_recall_avg'].mean()),
            'normal_fpr'         : float(g['normal_fpr'].mean()),
            'f1_mean'            : float(g['f1'].mean()),
            'f1_std'             : float(g['f1'].std()),
            'auroc_mean'         : float(g['auroc'].mean()),
            'auroc_std'          : float(g['auroc'].std()),
        })
    # Reindex to canonical ANOMALY_TYPES order
    summary = pd.DataFrame(summary_rows).set_index('held_out_type').reindex(list(ANOMALY_TYPES)).reset_index()
    summary_path = RESULTS / 'E9b_leave_one_out_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f'\nWrote {summary_path}')

    # Build a horizontal bar chart of held-out recall and trained recall side by side.
    SHORT = {
        'Jamming'                                  : 'Jam',
        'High Network Congestion (Gradual Buildup)': 'HC-G',
        'High Network Congestion (Sudden Spike)'   : 'HC-S',
        'Co-Channel Interference (Severe)'         : 'CCI-S',
        'Co-Channel Interference (Mild)'           : 'CCI-M',
        'Faulty RF Filters (Temporal)'             : 'RF-T',
        'Doppler Shift (Severe)'                   : 'Dop',
        'Resource Allocation Bugs'                 : 'RA-B',
        'Antenna Failure'                          : 'Ant',
        'Faulty Handover Algorithm (Too Frequent)' : 'FH',
        'Buffer Overflow (Gradual Buildup)'        : 'BO',
    }
    labels = [SHORT[t] for t in summary['held_out_type']]
    held = summary['holdout_recall_mean'].values * 100
    held_lo = summary['holdout_recall_ci_lo'].values * 100
    held_hi = summary['holdout_recall_ci_hi'].values * 100
    trained = summary['trained_recall_avg'].values * 100
    yerr = np.vstack([held - held_lo, held_hi - held])

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ypos = np.arange(len(labels))
    w = 0.4
    bars_h = ax.barh(ypos - w/2, held, height=w, xerr=yerr,
                     color='#d62728', edgecolor='#222', linewidth=0.5,
                     label='Held-out type recall (the LOO target)',
                     error_kw=dict(ecolor='#222', lw=0.7, capsize=2))
    bars_t = ax.barh(ypos + w/2, trained, height=w,
                     color='#2ca02c', edgecolor='#222', linewidth=0.5,
                     label='Avg. recall on trained-on types')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Recall (%)')
    ax.set_xlim(0, 100)
    ax.set_title('TelecomTS leave-one-anomaly-out audit\n(HGB on 240 KPI summaries; 10 seeds; 95% bootstrap CI on held-out recall)')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig_path = FIGURES / 'E9b_leave_one_out_recall.pdf'
    fig.savefig(fig_path)
    plt.show()
    print(f'Wrote {fig_path}')

    CALIB_TYPES   = ['Jamming', 'Co-Channel Interference (Severe)', 'Doppler Shift (Severe)']
    CALIB_FRACS   = [0.0, 0.05, 0.10, 0.25]
    CALIB_SEEDS   = list(range(10))

    calib_rows = []
    t0 = time.time()
    for T in CALIB_TYPES:
        for f in CALIB_FRACS:
            for seed in CALIB_SEEDS:
                sp = build_loo_split(T, seed)
                train_idx = sp['train']; val_idx = sp['val']; test_idx = sp['test']
                T_train_pool = sp['T_train_pool']
                n_inject = int(round(f * T_train_pool.size))
                inj = T_train_pool[:n_inject]   # already shuffled by seed inside build_loo_split
                train_idx_aug = np.concatenate([train_idx, inj])
                y_tr = corpus.y[train_idx_aug]; y_val = corpus.y[val_idx]; y_test = corpus.y[test_idx]
                if len(np.unique(y_tr)) < 2:
                    continue
                val_p, test_p = fit_hgb(
                    F_full[train_idx_aug], y_tr, F_full[val_idx], F_full[test_idx], seed,
                )
                thr = select_threshold(val_p, y_val)
                t_types = corpus.anomaly_type[test_idx]
                is_T_test     = (t_types == T) & (y_test == 1)
                is_other_test = (y_test == 1) & ~is_T_test
                is_norm_test  = (y_test == 0)
                held_recall = float((test_p[is_T_test] >= thr).mean()) if is_T_test.sum() else float('nan')
                trained_recall = float((test_p[is_other_test] >= thr).mean()) if is_other_test.sum() else float('nan')
                normal_fpr   = float((test_p[is_norm_test] >= thr).mean()) if is_norm_test.sum() else float('nan')
                try:
                    auroc = float(roc_auc_score(y_test, test_p))
                except Exception:
                    auroc = float('nan')
                calib_rows.append({
                    'held_out_type': T, 'fraction': f, 'seed': seed,
                    'n_inject': n_inject, 'n_T_train_pool': int(T_train_pool.size),
                    'holdout_recall': held_recall,
                    'trained_recall_avg': trained_recall,
                    'normal_fpr': normal_fpr,
                    'auroc': auroc,
                    'threshold': thr,
                })
    elapsed = time.time() - t0
    print(f'Done calibration sweep: {len(calib_rows)} rows in {elapsed:.1f}s')

    calib = pd.DataFrame.from_records(calib_rows)
    calib_per_seed_path = RESULTS / 'E9b_calibration_per_seed.csv'
    calib.to_csv(calib_per_seed_path, index=False)
    print(f'Wrote {calib_per_seed_path}')

    calib_summary = calib.groupby(['held_out_type', 'fraction']).agg(
        n_seeds         =('seed', 'nunique'),
        holdout_mean    =('holdout_recall', 'mean'),
        holdout_std     =('holdout_recall', 'std'),
        trained_mean    =('trained_recall_avg', 'mean'),
        normal_fpr_mean =('normal_fpr', 'mean'),
        auroc_mean      =('auroc', 'mean'),
    ).reset_index()
    calib_summary_path = RESULTS / 'E9b_calibration_summary.csv'
    calib_summary.to_csv(calib_summary_path, index=False)
    print()
    print(calib_summary.to_string(index=False))
    print(f'\nWrote {calib_summary_path}')

    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    markers = {'Jamming': 'o', 'Co-Channel Interference (Severe)': 's', 'Doppler Shift (Severe)': '^'}
    colours = {'Jamming': '#d62728', 'Co-Channel Interference (Severe)': '#1f77b4', 'Doppler Shift (Severe)': '#2ca02c'}
    for T, g in calib_summary.groupby('held_out_type'):
        g = g.sort_values('fraction')
        ax.errorbar(g['fraction'] * 100, g['holdout_mean'] * 100,
                    yerr=g['holdout_std'] * 100,
                    marker=markers.get(T, 'o'), color=colours.get(T, 'k'),
                    linewidth=1.2, capsize=2, label=SHORT.get(T, T))
    ax.set_xlabel('% of held-out-type training pool added to training')
    ax.set_ylabel('Held-out-type recall (%)')
    ax.set_ylim(-2, 102)
    ax.set_xticks([0, 5, 10, 25])
    ax.grid(linestyle=':', alpha=0.5)
    ax.legend(title='Held-out type', fontsize=7, title_fontsize=7, loc='lower right')
    ax.set_title('Calibration mini-sweep across three held-out anomaly types')
    fig.tight_layout()
    calib_fig_path = FIGURES / 'E9b_calibration_minisweep.pdf'
    fig.savefig(calib_fig_path)
    plt.show()
    print(f'Wrote {calib_fig_path}')

    # Headline LOO statistics
    mean_holdout = float(summary['holdout_recall_mean'].mean()) * 100
    worst_type = summary.loc[summary['holdout_recall_mean'].idxmin(), 'held_out_type']
    worst_val  = float(summary['holdout_recall_mean'].min()) * 100
    best_type  = summary.loc[summary['holdout_recall_mean'].idxmax(), 'held_out_type']
    best_val   = float(summary['holdout_recall_mean'].max()) * 100
    mean_trained = float(summary['trained_recall_avg'].mean()) * 100
    mean_normal_fpr = float(summary['normal_fpr'].mean()) * 100

    # Calibration headline: f=0 vs f=10 for each calib type
    calib_pivot = calib_summary.pivot(index='held_out_type', columns='fraction', values='holdout_mean')
    print('Calibration recall (mean over 10 seeds), columns are f:')
    print((calib_pivot * 100).round(1).to_string())
    print()

    print(f'''Across the 11 TelecomTS anomaly types, leaving any single type out of training collapses
    its held-out recall to {mean_holdout:.1f}% on average (mean across types; best={SHORT.get(best_type,best_type)} at {best_val:.1f}%, worst={SHORT.get(worst_type,worst_type)} at {worst_val:.1f}%),
    while average recall on the trained-on types stays at {mean_trained:.1f}% and the normal false-positive
    rate at {mean_normal_fpr:.1f}%. The transfer failure documented in E9 for Jamming is therefore not specific
    to Jamming: it is a property of the audit protocol that surfaces whenever any anomaly origin is
    held out of training. The calibration mini-sweep on three representative held-out types (Jam, CCI-S, Dop)
    shows the same ~10%-budget repair pattern: held-out-type recall rises from near zero at f=0% toward
    saturation by f=10%-25%, replicating the E10 result on real Jamming.''')
