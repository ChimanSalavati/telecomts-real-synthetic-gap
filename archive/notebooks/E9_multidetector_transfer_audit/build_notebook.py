"""Builds the E9 multi-detector transfer audit notebook.

Run::

    python experiments/E9_multidetector_transfer_audit/build_notebook.py

This regenerates ``E9_multidetector_transfer_audit.ipynb`` next to this file.

The notebook reuses E3's three settings (controlled-500, balanced detection,
full-scale TSTR) but evaluates seven tabular detectors instead of one.
Output files include a wide two-column-spanning LaTeX table for the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_builder import NotebookBuilder  # noqa: E402


def build() -> Path:
    nb = NotebookBuilder("E9: Multi-detector real-synthetic transfer audit")

    nb.md(r"""# E9 — Multi-detector real-synthetic transfer audit

Goal: re-run E3's three real-vs-synthetic transfer settings with **seven
different tabular detectors** instead of one. The point is to show that the
transfer failure (synthetic-only training collapses on real Jamming) is
**detector-independent**: every reasonable tabular detector reproduces it.

This notebook is reviewer defence. A common reviewer concern would be
"maybe HGB is not strong enough; an XGBoost / LightGBM / MLP would solve
it". E9 answers that directly with mean ± std over 10 seeds for each
detector x (setting, regime) combination.

## Settings (same splits as E3)
- Controlled-500 (50 normals + 25 real Jamming + 25 synthetic anomalies in test).
- Balanced-detection (494 test windows, 50% anomaly rate).
- Full-scale TSTR (6{,}400 test windows, natural anomaly rate).

## Regimes
- ``all_origins``: train on normals + real Jamming + synthetic anomalies.
- ``synthetic_only``: drop real Jamming from train + val.
- ``real_only``: drop synthetic from train + val (controlled-500 only;
  the larger splits do not have enough real Jamming to train alone).

## Detectors
- HistGradientBoosting (HGB)
- RandomForest (RF, 200 trees)
- LogisticRegression on standardized features (LR)
- k-Nearest Neighbours (k=5, on standardized features)
- MLPClassifier (sklearn, hidden=(64, 32))
- XGBoost (300 trees, max_depth=6, learning_rate=0.1)
- LightGBM (300 trees, max_depth=-1, learning_rate=0.1)

## Metrics per (setting, regime, detector, seed)
- Aggregate F1
- AUROC (overall test)
- Average precision
- Real-Jamming detection rate (recall on real subset)
- Synthetic-anomaly detection rate (recall on synthetic subset)
- Normal false-positive rate
""")

    nb.md("## 0. Bootstrap")

    nb.code("""# Make _shared importable, set plot defaults, suppress noisy warnings.
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
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

# Optional installs.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print('xgboost not available:', e)
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception as e:
    print('lightgbm not available:', e)
    HAS_LGBM = False

from _shared.data_utils import (
    load_corpus,
    get_or_build_corpus_features,
    make_controlled_500_split,
    make_balanced_detection_split,
    make_fullscale_split,
    REAL_ANOMALY_TYPES,
)

HERE = Path('.').resolve()
RESULTS = HERE / 'results'
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
RESULTS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
print('Working dir :', HERE)
print('Has XGBoost :', HAS_XGB)
print('Has LightGBM:', HAS_LGBM)
""")

    nb.md("## 1. Load corpus, features, and the three splits")

    nb.code("""corpus = load_corpus(verbose=True)
F_full, feat_names = get_or_build_corpus_features(verbose=True)
print('corpus N    :', corpus.n)
print('F_full shape:', F_full.shape)
assert F_full.shape == (corpus.n, 240)

real_origin = (corpus.anomaly_origin == 'real')
synth_origin = (corpus.anomaly_origin == 'synthetic')
normal_mask = (corpus.y == 0)

SPLITS = {
    'controlled_500': make_controlled_500_split(corpus, seed=42),
    'balanced_detection': make_balanced_detection_split(corpus, seed=42),
    'fullscale': make_fullscale_split(corpus, seed=42),
}
for name, parts in SPLITS.items():
    sizes = {k: int(v.size) for k, v in parts.items()}
    print(f'{name:>22s} sizes={sizes}')
""")

    nb.md("## 2. Detector factory and threshold helper")

    nb.code("""def make_detector(name: str, seed: int):
    \"\"\"Return (estimator, needs_scaling) for the given detector name.\"\"\"
    if name == 'HGB':
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=None, random_state=seed,
        ), False
    if name == 'RF':
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
        ), False
    if name == 'LR':
        return LogisticRegression(max_iter=2000, random_state=seed), True
    if name == 'kNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True
    if name == 'MLP':
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed,
            early_stopping=True, n_iter_no_change=10,
        ), True
    if name == 'XGB':
        if not HAS_XGB:
            return None, False
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
            verbosity=0, random_state=seed, n_jobs=-1,
        ), False
    if name == 'LGBM':
        if not HAS_LGBM:
            return None, False
        return LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, verbose=-1,
            random_state=seed, n_jobs=-1,
        ), False
    raise ValueError(f'Unknown detector {name}')


DETECTORS = ['HGB', 'RF', 'LR', 'kNN', 'MLP']
if HAS_XGB:  DETECTORS.append('XGB')
if HAS_LGBM: DETECTORS.append('LGBM')
print('Detectors :', DETECTORS)


def fit_predict_proba(name: str, X_train, y_train, X_val, X_test, seed: int):
    est, needs_scaling = make_detector(name, seed)
    if needs_scaling:
        scaler = StandardScaler().fit(X_train)
        X_train_t = scaler.transform(X_train)
        X_val_t = scaler.transform(X_val)
        X_test_t = scaler.transform(X_test)
    else:
        X_train_t, X_val_t, X_test_t = X_train, X_val, X_test
    # XGB requires class 0/1 already; we already pass int labels.
    est.fit(X_train_t, y_train)
    val_p = est.predict_proba(X_val_t)[:, 1]
    test_p = est.predict_proba(X_test_t)[:, 1]
    return val_p, test_p


def select_threshold(val_p: np.ndarray, y_val: np.ndarray) -> float:
    \"\"\"Pick the threshold that maximises F1 on the val set.\"\"\"
    if y_val.sum() == 0 or (1 - y_val).sum() == 0:
        return 0.5
    grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (val_p >= thr).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float('nan')


def _safe_ap(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float('nan')


def evaluate_subgroup_detection(test_p: np.ndarray, y_test: np.ndarray,
                                 thr: float, mask: np.ndarray) -> float:
    \"\"\"Detection rate among the subgroup defined by ``mask`` over test.\"\"\"
    if mask.sum() == 0:
        return float('nan')
    return float((test_p[mask] >= thr).mean())
""")

    nb.md("""## 3. Per-(setting, regime) train/val/test index builder

We reuse E3's logic: from the canonical split, optionally drop one
anomaly origin from train+val (to construct ``synthetic_only`` /
``real_only`` regimes), but never modify the test set.
""")

    nb.code("""def build_split_pool(setting: str, seed: int):
    \"\"\"Return (train_idx, val_idx, test_idx) for the given setting.

    For controlled_500 we re-sample with the per-seed split builder so the 10
    seeds have varying compositions. For balanced_detection and fullscale we
    keep the canonical split (seed=42) fixed and only vary the model seed.
    \"\"\"
    if setting == 'controlled_500':
        sp = make_controlled_500_split(corpus, seed=seed)
        return sp['train'], sp['val'], sp['test']
    elif setting == 'balanced_detection':
        sp = SPLITS['balanced_detection']
        return sp['train'], sp['val'], sp['test']
    elif setting == 'fullscale':
        sp = SPLITS['fullscale']
        train_pool = sp['train']
        # Carve a stratified val from train_pool (10%) deterministically per seed.
        y_train_pool = corpus.y[train_pool]
        train_idx, val_idx = train_test_split(
            train_pool, test_size=0.10, stratify=y_train_pool, random_state=seed,
        )
        return train_idx, val_idx, sp['test']
    raise ValueError(setting)


def apply_regime(train_idx, val_idx, regime: str):
    \"\"\"Drop one anomaly origin from train+val (test is left unchanged).\"\"\"
    def _filter(idx):
        if regime == 'all_origins':
            return idx
        keep = []
        for i in idx:
            if corpus.y[i] == 0:
                keep.append(i)
                continue
            if regime == 'synthetic_only' and corpus.anomaly_origin[i] == 'synthetic':
                keep.append(i)
            elif regime == 'real_only' and corpus.anomaly_origin[i] == 'real':
                keep.append(i)
        return np.array(keep, dtype=idx.dtype)
    return _filter(train_idx), _filter(val_idx)
""")

    nb.md("## 4. Main multi-detector loop")

    nb.code("""SEEDS = list(range(10))

REGIMES_BY_SETTING = {
    'controlled_500':       ['all_origins', 'synthetic_only', 'real_only'],
    'balanced_detection':   ['synthetic_only'],
    'fullscale':            ['synthetic_only'],
}

per_seed_rows = []
total = sum(len(v) * len(SEEDS) * len(DETECTORS) for v in REGIMES_BY_SETTING.values())
done = 0
print(f'Total runs to do: {total}')
t_global = time.time()

for setting, regimes in REGIMES_BY_SETTING.items():
    for regime in regimes:
        for seed in SEEDS:
            train_idx, val_idx, test_idx = build_split_pool(setting, seed=seed)
            train_idx, val_idx = apply_regime(train_idx, val_idx, regime)
            # Skip if a regime makes training infeasible
            y_train = corpus.y[train_idx]
            y_val = corpus.y[val_idx]
            y_test = corpus.y[test_idx]
            test_real_mask = (corpus.anomaly_origin[test_idx] == 'real')
            test_synth_mask = (corpus.anomaly_origin[test_idx] == 'synthetic')
            test_normal_mask = (corpus.y[test_idx] == 0)
            X_train_f = F_full[train_idx]
            X_val_f = F_full[val_idx]
            X_test_f = F_full[test_idx]

            for det_name in DETECTORS:
                t0 = time.time()
                if len(np.unique(y_train)) < 2:
                    # Cannot train: regime removed all of one class.
                    per_seed_rows.append({
                        'setting': setting, 'regime': regime, 'detector': det_name,
                        'seed': seed,
                        'n_train': int(train_idx.size), 'n_val': int(val_idx.size),
                        'n_test': int(test_idx.size),
                        'n_real_test': int(test_real_mask.sum()),
                        'n_synth_test': int(test_synth_mask.sum()),
                        'threshold': float('nan'), 'f1': float('nan'),
                        'precision': float('nan'), 'recall': float('nan'),
                        'auroc': float('nan'), 'avg_precision': float('nan'),
                        'real_det': float('nan'), 'synth_det': float('nan'),
                        'normal_fpr': float('nan'),
                        'fit_time_sec': 0.0,
                        'note': 'single_class_train',
                    })
                    done += 1
                    continue
                try:
                    val_p, test_p = fit_predict_proba(
                        det_name, X_train_f, y_train, X_val_f, X_test_f, seed=seed,
                    )
                except Exception as e:
                    per_seed_rows.append({
                        'setting': setting, 'regime': regime, 'detector': det_name,
                        'seed': seed,
                        'n_train': int(train_idx.size), 'n_val': int(val_idx.size),
                        'n_test': int(test_idx.size),
                        'n_real_test': int(test_real_mask.sum()),
                        'n_synth_test': int(test_synth_mask.sum()),
                        'threshold': float('nan'), 'f1': float('nan'),
                        'precision': float('nan'), 'recall': float('nan'),
                        'auroc': float('nan'), 'avg_precision': float('nan'),
                        'real_det': float('nan'), 'synth_det': float('nan'),
                        'normal_fpr': float('nan'),
                        'fit_time_sec': 0.0,
                        'note': f'error:{type(e).__name__}',
                    })
                    done += 1
                    continue
                thr = select_threshold(val_p, y_val)
                pred = (test_p >= thr).astype(int)
                if y_test.sum() > 0:
                    f1_val = f1_score(y_test, pred, zero_division=0)
                    pr_val = precision_score(y_test, pred, zero_division=0)
                    rc_val = recall_score(y_test, pred, zero_division=0)
                else:
                    f1_val = pr_val = rc_val = float('nan')
                auroc = _safe_auc(y_test, test_p)
                ap = _safe_ap(y_test, test_p)
                # Subgroup detection rates (recall within positives only, per origin).
                real_det = evaluate_subgroup_detection(test_p, y_test, thr, test_real_mask)
                synth_det = evaluate_subgroup_detection(test_p, y_test, thr, test_synth_mask)
                # Normal FPR
                if test_normal_mask.sum() > 0:
                    normal_fpr = float((test_p[test_normal_mask] >= thr).mean())
                else:
                    normal_fpr = float('nan')
                per_seed_rows.append({
                    'setting': setting, 'regime': regime, 'detector': det_name,
                    'seed': seed,
                    'n_train': int(train_idx.size), 'n_val': int(val_idx.size),
                    'n_test': int(test_idx.size),
                    'n_real_test': int(test_real_mask.sum()),
                    'n_synth_test': int(test_synth_mask.sum()),
                    'threshold': float(thr), 'f1': float(f1_val),
                    'precision': float(pr_val), 'recall': float(rc_val),
                    'auroc': float(auroc), 'avg_precision': float(ap),
                    'real_det': float(real_det), 'synth_det': float(synth_det),
                    'normal_fpr': float(normal_fpr),
                    'fit_time_sec': float(time.time() - t0),
                    'note': 'ok',
                })
                done += 1
            print(f'  setting={setting:>20s} regime={regime:>16s} seed={seed} '
                  f'({done}/{total} runs done; elapsed={time.time()-t_global:.0f}s)')

per_seed_df = pd.DataFrame(per_seed_rows)
per_seed_df.to_csv(RESULTS / 'E9_transfer_per_seed.csv', index=False)
print('\\nSaved per-seed CSV. Shape:', per_seed_df.shape)
per_seed_df.head(8)
""")

    nb.md("## 5. Aggregate across seeds")

    nb.code("""metric_cols = ['f1', 'precision', 'recall', 'auroc', 'avg_precision',
               'real_det', 'synth_det', 'normal_fpr', 'fit_time_sec']
agg_funcs = ['mean', 'std']
group_keys = ['setting', 'regime', 'detector']

summary = (
    per_seed_df.groupby(group_keys)[metric_cols]
    .agg(agg_funcs)
    .reset_index()
)
# Flatten column names: ('f1', 'mean') -> 'f1_mean'
summary.columns = group_keys + [f'{m}_{s}' for m, s in summary.columns[len(group_keys):]]
summary['n_seeds'] = per_seed_df.groupby(group_keys).size().values
summary.to_csv(RESULTS / 'E9_transfer_summary.csv', index=False)
print('Saved summary CSV.')
print(summary[['setting','regime','detector','f1_mean','real_det_mean','synth_det_mean','auroc_mean','normal_fpr_mean']].to_string(index=False))
""")

    nb.md("""## 6. Wide LaTeX table for the paper

Layout: rows = detectors, columns = (setting, regime) x metric.
Spans both columns of the ACM template via ``table*``.
""")

    nb.code(r"""# Wide LaTeX table generation. We use raw strings throughout so backslashes
# are not over-escaped; the only escape we need is `\\` for LaTeX row endings,
# which we get with `r'\\'`.
DETECTOR_ORDER = DETECTORS

COLUMN_BLOCKS = [
    ('controlled_500',     'all_origins',    'Ctrl-500 / all'),
    ('controlled_500',     'synthetic_only', 'Ctrl-500 / synth'),
    ('balanced_detection', 'synthetic_only', 'Bal-494 / synth'),
    ('fullscale',          'synthetic_only', 'Full-6.4k / synth'),
]
METRICS_IN_TABLE = [
    ('f1',        'F1',    'f1'),    # show as 0.NN +/- std
    ('auroc',     'AUC',   'f1'),    # show as 0.NN +/- std
    ('real_det',  'Real',  'pct'),   # show as NN\% (+/- std)
    ('synth_det', 'Synth', 'pct'),   # show as NN\% (+/- std)
]


def _fmt_pair(mean, std, kind):
    # Always show mean +/- std so the table stays visually consistent.
    # Stds that are exactly 0.0 are still rendered (as +/- 0.00 or 0%) so the
    # reader sees that we ran 10 seeds.
    if pd.isna(mean):
        return '--'
    if kind == 'f1':
        s = 0.0 if pd.isna(std) else float(std)
        return f'{mean:.2f}{{\\tiny$\\pm${s:.2f}}}'
    elif kind == 'pct':
        s = 0.0 if pd.isna(std) else float(std)
        return f'{100*mean:.0f}{{\\tiny$\\pm${100*s:.0f}}}\\%'
    return f'{mean:.3f}'


n_blocks = len(COLUMN_BLOCKS)
n_metrics = len(METRICS_IN_TABLE)
n_cols = n_blocks * n_metrics
col_spec = 'l' + ''.join(['c'] * n_cols)

EOL = r'\\'  # LaTeX row terminator (two literal backslashes)

# Header row 1: block labels spanning n_metrics each.
hdr_block_cells = ['Detector']
for _, _, label in COLUMN_BLOCKS:
    hdr_block_cells.append(rf'\multicolumn{{{n_metrics}}}{{c}}{{{label}}}')
hdr_block_line = ' & '.join(hdr_block_cells) + ' ' + EOL

# Cmidrules under the block headers.
cmidrules = []
for k in range(n_blocks):
    a = 2 + k * n_metrics
    b = a + n_metrics - 1
    cmidrules.append(rf'\cmidrule(lr){{{a}-{b}}}')
cmidrule_line = ''.join(cmidrules)

# Header row 2: metric names.
hdr_metric_cells = [' ']
for _ in COLUMN_BLOCKS:
    for _, lbl, _ in METRICS_IN_TABLE:
        hdr_metric_cells.append(lbl)
hdr_metric_line = ' & '.join(hdr_metric_cells) + ' ' + EOL

# Body rows.
body_rows = []
summary_indexed = summary.set_index(group_keys)
for det in DETECTOR_ORDER:
    row_cells = [det]
    for setting, regime, _ in COLUMN_BLOCKS:
        for col, _, kind in METRICS_IN_TABLE:
            try:
                row = summary_indexed.loc[(setting, regime, det)]
                mean = row[f'{col}_mean']; std = row[f'{col}_std']
            except KeyError:
                mean = float('nan'); std = float('nan')
            row_cells.append(_fmt_pair(mean, std, kind))
    body_rows.append(' & '.join(row_cells) + ' ' + EOL)

caption = (
    'Multi-detector real--synthetic transfer audit on TelecomTS. '
    'Each cell reports mean$\\pm$std over 10 seeds. '
    '``F1\'\' is aggregate binary F1 at the validation-tuned threshold; '
    '``AUC\'\' is threshold-free area under the ROC curve on the test set; '
    '``Real\'\'~/~``Synth\'\' are detection rates (recalls) within the real-Jamming and '
    'synthetic-anomaly subgroups of the test set. '
    'Columns are grouped by training setting and training regime. '
    'Across every detector tested, synthetic-only training collapses on real Jamming at scale '
    'while preserving synthetic-anomaly recall and high overall AUC.'
)

lines = []
lines.append(r'\begin{table*}[t]')
lines.append(r'  \caption{' + caption + r'}')
lines.append(r'  \label{tab:e9_transfer}')
lines.append(r'  \footnotesize')
lines.append(r'  \setlength{\tabcolsep}{1.5pt}')
lines.append(r'  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}' + col_spec + r'@{}}')
lines.append(r'    \toprule')
lines.append('    ' + hdr_block_line)
lines.append('    ' + cmidrule_line)
lines.append('    ' + hdr_metric_line)
lines.append(r'    \midrule')
for r in body_rows:
    lines.append('    ' + r)
lines.append(r'    \bottomrule')
lines.append(r'  \end{tabular*}')
lines.append(r'\end{table*}')

table_tex = '\n'.join(lines) + '\n'
out = TABLES / 'E9_transfer_table.tex'
out.write_text(table_tex)
print('Wrote', out)
print('\n----- LaTeX preview -----')
print(table_tex[:2200])
""")

    nb.md("""## 7. Inline Markdown summary

A compact text-mode rendering of the same table for quick inspection.
""")

    nb.code("""def fmt_pct(mean, std):
    if pd.isna(mean): return '--'
    if pd.isna(std) or std < 1e-9: return f'{100*mean:.1f}'
    return f'{100*mean:.1f}±{100*std:.1f}'

def fmt_f1(mean, std):
    if pd.isna(mean): return '--'
    if pd.isna(std) or std < 1e-9: return f'{mean:.3f}'
    return f'{mean:.3f}±{std:.3f}'

print(f"{'Detector':>8s} | " +
      ' | '.join([f'{lbl:>26s}' for _, _, lbl in COLUMN_BLOCKS]))
for det in DETECTOR_ORDER:
    cells = []
    for setting, regime, _ in COLUMN_BLOCKS:
        try:
            row = summary_indexed.loc[(setting, regime, det)]
            cells.append(
                f"F1 {fmt_f1(row['f1_mean'], row['f1_std'])} R {fmt_pct(row['real_det_mean'], row['real_det_std'])} S {fmt_pct(row['synth_det_mean'], row['synth_det_std'])}"
            )
        except KeyError:
            cells.append('--')
    print(f"{det:>8s} | " + ' | '.join(c.ljust(26) for c in cells))
""")

    out = HERE / "E9_multidetector_transfer_audit.ipynb"
    return nb.write(out)


if __name__ == "__main__":
    p = build()
    print(f"Wrote notebook to {p}")
