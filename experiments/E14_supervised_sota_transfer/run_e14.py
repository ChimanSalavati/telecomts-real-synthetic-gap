#!/usr/bin/env python3
# AUTO-GENERATED from E14_supervised_sota_transfer.ipynb by pipeline/convert_notebooks.py -- do not edit by hand.
# This is the executable Python conversion of the original Jupyter notebook.
# Standalone:  python experiments/<dir>/run_e14.py
# Via runner:  python main.py --experiment E14
"""E14: converted notebook runner (offline-aware, centralized outputs)."""
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
    # into the centralized artifacts/E14/ tree.
    os.chdir(exp_output_dir("E14", ""))

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
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd

    from _shared.data_utils import (
        load_corpus,
        make_balanced_detection_split,
        make_fullscale_split,
    )
    from _shared import sota_helpers as sh

    PATHS = sh.add_sota_paths()
    print('repo_root :', PATHS['repo_root'])

    import torch
    DEVICE_TORCH = (
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    # Use CPU for Mantis (its internal tensors do not move with .to() reliably).
    print('Device     :', DEVICE_TORCH, '| torch', torch.__version__)

    HERE = Path('.').resolve()
    RESULTS = HERE / 'results'
    TABLES  = HERE / 'tables'
    EMB_CACHE = HERE / 'embeddings_cache'
    RESULTS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    EMB_CACHE.mkdir(exist_ok=True)
    print('Working dir:', HERE)

    corpus = load_corpus(verbose=True)
    print('corpus N:', corpus.n)
    print('X       :', corpus.X.shape, corpus.X.dtype)

    SPLITS = {
        'balanced_detection': make_balanced_detection_split(corpus, seed=42),
        'fullscale':          make_fullscale_split(corpus, seed=42),
    }
    for name, parts in SPLITS.items():
        sizes = {k: int(v.size) for k, v in parts.items()}
        print(f'  {name:>22s} sizes={sizes}')

    REGIMES_BY_SETTING = {
        'controlled_500':       ['all_origins', 'synthetic_only'],
        'balanced_detection':   ['synthetic_only'],
        'fullscale':            ['synthetic_only'],
    }
    SEEDS = list(range(10))

    from typing import Optional


    def _emb_path(name: str, setting: str, seed: int, split_part: str) -> Path:
        return EMB_CACHE / f'{name}__{setting}__seed{seed}__{split_part}.npz'


    def _load_or_extract(name: str, encoder_fn, X_btc, setting: str, seed: int,
                         split_part: str, **encoder_kwargs):
        path = _emb_path(name, setting, seed, split_part)
        if path.exists():
            d = np.load(path)
            return d['Z'], bool(d['loaded'])
        Z, loaded = encoder_fn(X_btc, **encoder_kwargs)
        np.savez_compressed(path, Z=Z, loaded=np.array(loaded))
        return Z, loaded


    def get_embeddings(setting: str, seed: int):
        """Return a dict {encoder_name: (Z_train_pool, Z_val_pool, Z_test, loaded)}.

        The train_pool / val_pool / test indices come from the canonical
        pre-regime split (we slice by regime later, downstream of the cache).
        """
        bundle = sh.make_unsup_dataset(corpus, SPLITS, setting, seed)
        train_idx, val_idx, test_idx = bundle.train_idx, bundle.val_idx, bundle.test_idx
        X_train_btc = corpus.X[train_idx]
        X_val_btc   = corpus.X[val_idx]
        X_test_btc  = corpus.X[test_idx]
        out = {}
        Zm_tr, m_loaded = _load_or_extract('moment', sh.encode_moment, X_train_btc, setting, seed, 'train',
                                            device=DEVICE_TORCH)
        Zm_va, _        = _load_or_extract('moment', sh.encode_moment, X_val_btc,   setting, seed, 'val',
                                            device=DEVICE_TORCH)
        Zm_te, _        = _load_or_extract('moment', sh.encode_moment, X_test_btc,  setting, seed, 'test',
                                            device=DEVICE_TORCH)
        out['MOMENT'] = (Zm_tr, Zm_va, Zm_te, m_loaded)
        Zt_tr, t_loaded = _load_or_extract('toto', sh.encode_toto, X_train_btc, setting, seed, 'train',
                                            device=DEVICE_TORCH)
        Zt_va, _        = _load_or_extract('toto', sh.encode_toto, X_val_btc,   setting, seed, 'val',
                                            device=DEVICE_TORCH)
        Zt_te, _        = _load_or_extract('toto', sh.encode_toto, X_test_btc,  setting, seed, 'test',
                                            device=DEVICE_TORCH)
        out['Toto'] = (Zt_tr, Zt_va, Zt_te, t_loaded)
        Zn_tr, n_loaded = _load_or_extract('mantis', sh.encode_mantis, X_train_btc, setting, seed, 'train',
                                            device='cpu')
        Zn_va, _        = _load_or_extract('mantis', sh.encode_mantis, X_val_btc,   setting, seed, 'val',
                                            device='cpu')
        Zn_te, _        = _load_or_extract('mantis', sh.encode_mantis, X_test_btc,  setting, seed, 'test',
                                            device='cpu')
        out['Mantis'] = (Zn_tr, Zn_va, Zn_te, n_loaded)
        out['__indices__'] = (train_idx, val_idx, test_idx)
        return out


    # Eagerly populate the cache so a long e2e run does not get blocked behind
    # repeated encoder loads. We extract once per (setting, seed). For
    # ``balanced_detection`` and ``fullscale`` the canonical split is fixed at
    # seed=42, so foundation embeddings only need to be extracted once for those
    # settings (we still vary the per-seed validation slice for fullscale below).
    print('Foundation models will lazily populate', EMB_CACHE, 'as needed.')

    def build_split(setting: str, regime: str, seed: int):
        bundle = sh.make_sup_dataset(corpus, SPLITS, setting, regime, seed)
        return bundle


    def regime_filter_indices(parent_idx, parent_y, parent_origin, regime: str):
        """Return the relative positions inside parent_idx that survive the
        regime filter. Used to slice cached embeddings without re-extracting."""
        keep_mask = np.zeros(parent_idx.size, dtype=bool)
        for k, i in enumerate(parent_idx):
            if regime == 'all_origins':
                keep_mask[k] = True
                continue
            if parent_y[k] == 0:
                keep_mask[k] = True
                continue
            origin = parent_origin[k]
            if regime == 'synthetic_only' and origin == 'synthetic':
                keep_mask[k] = True
            elif regime == 'real_only' and origin == 'real':
                keep_mask[k] = True
        return keep_mask

    SUP_DETECTORS = ['MOMENT', 'Toto', 'Mantis', 'TimesNet-lite', 'InceptionTime-lite', 'PatchTST']

    per_seed_rows = []
    total_runs = sum(len(v) * len(SEEDS) * len(SUP_DETECTORS) for v in REGIMES_BY_SETTING.values())
    done = 0
    t_global = time.time()
    print(f'Total runs to do: {total_runs}')


    def _safe_train(name: str, *, X_tr, y_tr, X_va, y_va, X_te, seed: int):
        """E2E supervised models. Returns (val_p, test_p, fit_time, note)."""
        return sh.train_e2e_binary(
            name, X_tr, y_tr, X_va, X_te,
            seed=seed, device=DEVICE_TORCH,
            epochs=10, lr=1e-4, batch_size=64,
            in_channels=corpus.X.shape[2], seq_len=corpus.X.shape[1],
        )


    for setting, regimes in REGIMES_BY_SETTING.items():
        for seed in SEEDS:
            cached = None  # lazy: extract foundation embeddings only when needed
            # Pre-build the per-seed bundle once; we will re-filter for each regime.
            parent_bundle = sh.make_unsup_dataset(corpus, SPLITS, setting, seed)
            parent_train_idx = parent_bundle.train_idx
            parent_val_idx   = parent_bundle.val_idx
            parent_test_idx  = parent_bundle.test_idx
            parent_y_train   = corpus.y[parent_train_idx]
            parent_y_val     = corpus.y[parent_val_idx]
            parent_origin_train = corpus.anomaly_origin[parent_train_idx]
            parent_origin_val   = corpus.anomaly_origin[parent_val_idx]
            for regime in regimes:
                keep_train = regime_filter_indices(parent_train_idx, parent_y_train,
                                                    parent_origin_train, regime)
                keep_val   = regime_filter_indices(parent_val_idx, parent_y_val,
                                                    parent_origin_val, regime)
                train_idx = parent_train_idx[keep_train]
                val_idx   = parent_val_idx[keep_val]
                test_idx  = parent_test_idx
                y_train = corpus.y[train_idx]; y_val = corpus.y[val_idx]; y_test = corpus.y[test_idx]
                test_real_mask  = (corpus.anomaly_origin[test_idx] == 'real')
                test_synth_mask = (corpus.anomaly_origin[test_idx] == 'synthetic')
                test_normal_mask = (corpus.y[test_idx] == 0)

                # Skip if regime removed all of one class -> binary head cannot train.
                if len(np.unique(y_train)) < 2:
                    for det in SUP_DETECTORS:
                        per_seed_rows.append(sh.nan_row(
                            setting=setting, regime=regime, detector=det, seed=seed,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            test_real_mask=test_real_mask, test_synth_mask=test_synth_mask,
                            note='single_class_train',
                        ))
                        done += 1
                    continue

                # ---------- E2E supervised models ---------------------------------
                X_tr_btc = corpus.X[train_idx]
                X_va_btc = corpus.X[val_idx]
                X_te_btc = corpus.X[test_idx]
                for det in ['TimesNet-lite', 'InceptionTime-lite', 'PatchTST']:
                    t0 = time.time()
                    try:
                        val_p, test_p, fit_dt = _safe_train(
                            det, X_tr=X_tr_btc, y_tr=y_train, X_va=X_va_btc, y_va=y_val,
                            X_te=X_te_btc, seed=seed,
                        )
                        thr = sh.select_threshold(val_p, y_val)
                        row = sh.compute_metrics_row(
                            setting=setting, regime=regime, detector=det, seed=seed,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            test_real_mask=test_real_mask, test_synth_mask=test_synth_mask,
                            test_normal_mask=test_normal_mask,
                            threshold=thr, test_scores=test_p, y_test=y_test,
                            fit_time_sec=fit_dt, note='ok_e2e',
                        )
                    except Exception as e:
                        print(f'  [{det}] FAILED at setting={setting} regime={regime} seed={seed}: '
                              f'{type(e).__name__}: {e}')
                        row = sh.nan_row(
                            setting=setting, regime=regime, detector=det, seed=seed,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            test_real_mask=test_real_mask, test_synth_mask=test_synth_mask,
                            fit_time_sec=time.time()-t0, note=f'error:{type(e).__name__}',
                        )
                    per_seed_rows.append(row)
                    done += 1
                    print(f"  setting={setting:>20s} regime={regime:>16s} seed={seed} "
                          f"det={det:>20s} f1={row['f1']:.3f} auroc={row['auroc']:.3f} "
                          f"real={row['real_det']:.2f} synth={row['synth_det']:.2f} "
                          f"({done}/{total_runs}; elapsed={time.time()-t_global:.0f}s)")

                # ---------- Frozen-encoder + supervised head ---------------------
                if cached is None:
                    cached = get_embeddings(setting, seed)
                for det in ['MOMENT', 'Toto', 'Mantis']:
                    Zp_tr, Zp_va, Zp_te, loaded = cached[det]
                    # Slice cached embeddings down to the regime-filtered rows.
                    Z_tr = Zp_tr[keep_train]
                    Z_va = Zp_va[keep_val]
                    Z_te = Zp_te
                    t0 = time.time()
                    try:
                        val_p, test_p, fit_dt = sh.train_foundation_binary_head(
                            det, Z_tr, y_train, Z_va, y_val, Z_te,
                            seed=seed, device=DEVICE_TORCH,
                        )
                        thr = sh.select_threshold(val_p, y_val)
                        row = sh.compute_metrics_row(
                            setting=setting, regime=regime, detector=det, seed=seed,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            test_real_mask=test_real_mask, test_synth_mask=test_synth_mask,
                            test_normal_mask=test_normal_mask,
                            threshold=thr, test_scores=test_p, y_test=y_test,
                            fit_time_sec=fit_dt,
                            note='ok_foundation' if loaded else 'ok_surrogate',
                        )
                    except Exception as e:
                        print(f'  [{det}] FAILED at setting={setting} regime={regime} seed={seed}: '
                              f'{type(e).__name__}: {e}')
                        row = sh.nan_row(
                            setting=setting, regime=regime, detector=det, seed=seed,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                            test_real_mask=test_real_mask, test_synth_mask=test_synth_mask,
                            fit_time_sec=time.time()-t0, note=f'error:{type(e).__name__}',
                        )
                    per_seed_rows.append(row)
                    done += 1
                    print(f"  setting={setting:>20s} regime={regime:>16s} seed={seed} "
                          f"det={det:>20s} f1={row['f1']:.3f} auroc={row['auroc']:.3f} "
                          f"real={row['real_det']:.2f} synth={row['synth_det']:.2f} "
                          f"({done}/{total_runs}; elapsed={time.time()-t_global:.0f}s)")

            # Checkpoint after each (setting, seed) sweep.
            per_seed_df = pd.DataFrame(per_seed_rows)
            per_seed_df.to_csv(RESULTS / 'E14_sup_per_seed.csv', index=False)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(RESULTS / 'E14_sup_per_seed.csv', index=False)
    print('\nFinished. Saved per-seed CSV. shape:', per_seed_df.shape)
    per_seed_df.head(8)

    summary = sh.aggregate_per_seed(per_seed_df)
    summary.to_csv(RESULTS / 'E14_sup_summary.csv', index=False)
    print('Saved summary CSV. shape:', summary.shape)
    print(
        summary[['setting','regime','detector','f1_mean','real_det_mean','synth_det_mean','auroc_mean','normal_fpr_mean']]
        .to_string(index=False)
    )

    DETECTOR_PRETTY = {
        'MOMENT':              'MOMENT~\\cite{goswami2024moment} (frozen)',
        'Toto':                'Toto~\\cite{toto2024} (frozen)',
        'Mantis':              'Mantis~\\cite{feofanov2025mantis} (frozen)',
        'TimesNet-lite':       'TimesNet~\\cite{wu2023timesnet} (e2e)',
        'InceptionTime-lite':  'InceptionTime~\\cite{ismail2020inceptiontime} (e2e)',
        'PatchTST':            'PatchTST~\\cite{nie2023patchtst} (e2e)',
    }

    COLUMN_BLOCKS = [
        ('controlled_500',     'all_origins',    'Ctrl-500 / all'),
        ('controlled_500',     'synthetic_only', 'Ctrl-500 / synth'),
        ('balanced_detection', 'synthetic_only', 'Bal-494 / synth'),
        ('fullscale',          'synthetic_only', 'Full-6.4k / synth'),
    ]
    METRICS = [
        ('f1', 'F1', 'f1'),
        ('auroc', 'AUC', 'f1'),
        ('real_det', 'Real', 'pct'),
        ('synth_det', 'Synth', 'pct'),
    ]

    # Reuse the same renderer to make the full standalone table (helpful for a
    # quick LaTeX preview); we will hand-paste only the body rows into the paper.
    caption_full = (
        'Supervised SOTA detectors on TelecomTS using the same regime grid as '
        'Table~\\ref{tab:e9_transfer}. Each cell reports mean$\\pm$std over '
        '10 seeds. Foundation backbones (MOMENT, Toto, Mantis) are frozen with a '
        'binary head trained on the regime-filtered training pool; '
        'TimesNet, InceptionTime, and PatchTST are trained end-to-end with '
        'cross-entropy on $\\{normal, anomaly\\}$ labels.'
    )
    table_tex = sh.render_wide_table(
        summary,
        detector_order=['MOMENT', 'Toto', 'Mantis', 'TimesNet-lite', 'InceptionTime-lite', 'PatchTST'],
        column_blocks=COLUMN_BLOCKS,
        metrics=METRICS,
        label='tab:e14_sup',
        caption=caption_full,
        detector_pretty=DETECTOR_PRETTY,
    )
    out = TABLES / 'E14_sup_table.tex'
    out.write_text(table_tex)
    print('Wrote', out)
    print('\n----- LaTeX preview (full standalone) -----')
    print(table_tex[:2400])

    # Body-only rendering for direct paste into tab:e9_transfer.
    def _fmt(mean, std, kind):
        return sh._fmt_pair(mean, std, kind)


    summary_indexed = summary.set_index(['setting', 'regime', 'detector'])
    body_rows = []
    for det in ['MOMENT', 'Toto', 'Mantis', 'TimesNet-lite', 'InceptionTime-lite', 'PatchTST']:
        cells = [DETECTOR_PRETTY.get(det, det)]
        for setting, regime, _ in COLUMN_BLOCKS:
            for col, _, kind in METRICS:
                try:
                    row = summary_indexed.loc[(setting, regime, det)]
                    mean = row[f'{col}_mean']; std = row[f'{col}_std']
                except KeyError:
                    mean = float('nan'); std = float('nan')
                cells.append(_fmt(mean, std, kind))
        body_rows.append('    ' + ' & '.join(cells) + ' \\\\')

    body_tex = '\n'.join(body_rows) + '\n'
    (TABLES / 'E14_sup_body_rows.tex').write_text(body_tex)
    print('Wrote', TABLES / 'E14_sup_body_rows.tex')
    print('\n----- body rows (paste under \\midrule of tab:e9_transfer) -----')
    print(body_tex)

    def fmt_pct(mean, std):
        if pd.isna(mean): return '--'
        if pd.isna(std) or std < 1e-9: return f'{100*mean:.1f}'
        return f'{100*mean:.1f}±{100*std:.1f}'

    def fmt_f1(mean, std):
        if pd.isna(mean): return '--'
        if pd.isna(std) or std < 1e-9: return f'{mean:.3f}'
        return f'{mean:.3f}±{std:.3f}'

    print(f"{'Detector':>20s} | " + ' | '.join([f'{lbl:>26s}' for _, _, lbl in COLUMN_BLOCKS]))
    for det in ['MOMENT', 'Toto', 'Mantis', 'TimesNet-lite', 'InceptionTime-lite', 'PatchTST']:
        cells = []
        for setting, regime, _ in COLUMN_BLOCKS:
            try:
                row = summary_indexed.loc[(setting, regime, det)]
                cells.append(
                    f"F1 {fmt_f1(row['f1_mean'], row['f1_std'])} "
                    f"R {fmt_pct(row['real_det_mean'], row['real_det_std'])} "
                    f"S {fmt_pct(row['synth_det_mean'], row['synth_det_std'])}"
                )
            except KeyError:
                cells.append('--')
        print(f"{det:>20s} | " + ' | '.join(c.ljust(26) for c in cells))
