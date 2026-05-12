"""Shared helpers for the E13 (unsupervised SOTA) and E14 (supervised SOTA)
transfer-audit experiments.

This module assumes:
- ``evaluation_ver2/sota_models/`` is importable as ``sota_models`` after
  injecting that directory onto ``sys.path`` (handled by ``add_sota_paths``).
- ``spotlight_baseline`` is importable as a top-level package from the repo
  root (also handled by ``add_sota_paths``).
- ``sota_clones/tslib/`` is on the path so PatchTST and its sibling layers can
  be imported (handled by ``add_sota_paths``).

All dataset-builders return raw windows ``(N, 128, 16)``, not the engineered
240-D summaries. The supervised metric helpers compute the same per-cell
quantities as E9 (F1, AUC, real-Jamming detection rate, synthetic-anomaly
detection rate, normal FPR) so the resulting CSVs concatenate cleanly with
``E9_transfer_per_seed.csv``.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Path setup (idempotent)
# ---------------------------------------------------------------------------
def _install_reformer_pytorch_shim() -> None:
    """sota_clones/tslib/layers/SelfAttention_Family.py imports
    ``reformer_pytorch.LSHSelfAttention`` at module import time even though
    PatchTST and most other tslib models do not actually use LSH attention.

    We register a placeholder module under that name so the import succeeds.
    Anything that genuinely tries to instantiate LSHSelfAttention will get a
    NotImplementedError at first use, which is the right failure mode.
    """
    if "reformer_pytorch" in sys.modules:
        return
    import types
    mod = types.ModuleType("reformer_pytorch")

    class _LSHShim:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "reformer_pytorch.LSHSelfAttention is not installed; this "
                "tslib model variant is unsupported in the SOTA helpers."
            )

    mod.LSHSelfAttention = _LSHShim
    sys.modules["reformer_pytorch"] = mod


def add_sota_paths() -> dict[str, Path]:
    """Inject the repo-root SOTA folders onto ``sys.path``.

    The experiments live under
    ``experiments/<EX>/``; the repo root is two levels up.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent  # _shared -> experiments -> paper repo -> repo root
    paths = {
        "repo_root": repo_root,
        "evaluation_ver2": repo_root / "evaluation_ver2",
        "sota_models": repo_root / "evaluation_ver2",  # parent of sota_models pkg
        "spotlight_baseline": repo_root,  # parent of spotlight_baseline pkg
        "tslib": repo_root / "sota_clones" / "tslib",
        "tslib_layers": repo_root / "sota_clones" / "tslib" / "layers",
    }
    # sys.path: repo_root for spotlight_baseline; evaluation_ver2 for sota_models
    for key in ("repo_root", "sota_models", "tslib", "tslib_layers"):
        p = str(paths[key])
        if p not in sys.path:
            sys.path.insert(0, p)
    _install_reformer_pytorch_shim()
    return paths


# ---------------------------------------------------------------------------
# Split builders (raw windows + binary labels)
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Bundle returned by ``make_unsup_dataset`` / ``make_sup_dataset``."""

    setting: str
    regime: str
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray  # (N_train, 128, 16)
    y_train: np.ndarray  # (N_train,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_real_mask: np.ndarray  # bool, real-Jamming positives in test
    test_synth_mask: np.ndarray  # bool, synthetic-origin positives in test
    test_normal_mask: np.ndarray  # bool, normal-class samples in test


def _build_split_pool(corpus, splits, setting: str, seed: int):
    """Replica of E9's ``build_split_pool`` so we can reuse the same indices."""
    from _shared.data_utils import (
        make_controlled_500_split,
        make_balanced_detection_split,
        make_fullscale_split,
    )
    if setting == "controlled_500":
        sp = make_controlled_500_split(corpus, seed=seed)
        return sp["train"], sp["val"], sp["test"]
    if setting == "balanced_detection":
        sp = splits["balanced_detection"]
        return sp["train"], sp["val"], sp["test"]
    if setting == "fullscale":
        sp = splits["fullscale"]
        train_pool = sp["train"]
        y_train_pool = corpus.y[train_pool]
        train_idx, val_idx = train_test_split(
            train_pool, test_size=0.10, stratify=y_train_pool, random_state=seed,
        )
        return train_idx, val_idx, sp["test"]
    raise ValueError(setting)


def _apply_regime(corpus, train_idx, val_idx, regime: str):
    """Drop one anomaly origin from train+val (test is left unchanged).

    Mirrors E9's ``apply_regime``.
    """
    def _filter(idx):
        if regime == "all_origins":
            return idx
        keep = []
        for i in idx:
            if corpus.y[i] == 0:
                keep.append(i)
                continue
            origin = corpus.anomaly_origin[i]
            if regime == "synthetic_only" and origin == "synthetic":
                keep.append(i)
            elif regime == "real_only" and origin == "real":
                keep.append(i)
        return np.array(keep, dtype=idx.dtype)
    return _filter(train_idx), _filter(val_idx)


def _materialize(corpus, train_idx, val_idx, test_idx) -> dict:
    return {
        "X_train": corpus.X[train_idx],
        "y_train": corpus.y[train_idx],
        "X_val": corpus.X[val_idx],
        "y_val": corpus.y[val_idx],
        "X_test": corpus.X[test_idx],
        "y_test": corpus.y[test_idx],
        "test_real_mask": (corpus.anomaly_origin[test_idx] == "real"),
        "test_synth_mask": (corpus.anomaly_origin[test_idx] == "synthetic"),
        "test_normal_mask": (corpus.y[test_idx] == 0),
    }


def make_unsup_dataset(
    corpus, splits, setting: str, seed: int,
) -> SplitData:
    """Build a (train_normal, val, test) bundle for unsupervised SOTA.

    The ``y_train`` returned here is the binary anomaly label; downstream code
    in E13 selects the normal-only subset before training. Test indices come
    from the corresponding canonical split so E13 results are directly
    comparable to E9.
    """
    train_idx, val_idx, test_idx = _build_split_pool(corpus, splits, setting, seed)
    bundle = _materialize(corpus, train_idx, val_idx, test_idx)
    return SplitData(
        setting=setting,
        regime="unsup_normal_only",
        seed=seed,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        **bundle,
    )


def make_sup_dataset(
    corpus, splits, setting: str, regime: str, seed: int,
) -> SplitData:
    """Build a (train, val, test) bundle for supervised SOTA detectors using
    the same regime grid as E9 (``all_origins``, ``synthetic_only``,
    ``real_only``).
    """
    train_idx, val_idx, test_idx = _build_split_pool(corpus, splits, setting, seed)
    train_idx, val_idx = _apply_regime(corpus, train_idx, val_idx, regime)
    bundle = _materialize(corpus, train_idx, val_idx, test_idx)
    return SplitData(
        setting=setting,
        regime=regime,
        seed=seed,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        **bundle,
    )


# ---------------------------------------------------------------------------
# Threshold + metric helpers (same conventions as E9)
# ---------------------------------------------------------------------------
def select_threshold(val_p: np.ndarray, y_val: np.ndarray) -> float:
    """Pick the threshold maximising F1 on the val set; fallback to 0.5."""
    val_p = np.asarray(val_p, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.int64)
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


def select_threshold_continuous(val_scores: np.ndarray, y_val: np.ndarray,
                                 n_grid: int = 201,
                                 min_predicted_positive: float = 0.02,
                                 max_predicted_positive: float = 0.98) -> float:
    """For continuous (non-probability) anomaly scores: scan the score range.

    Excludes degenerate thresholds that predict <2% or >98% positives so a
    completely-uninformative score does not look like a perfect detector.
    """
    val_scores = np.asarray(val_scores, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.int64)
    if val_scores.size == 0 or len(np.unique(y_val)) < 2:
        return float(np.median(val_scores)) if val_scores.size else 0.0
    lo, hi = float(np.min(val_scores)), float(np.max(val_scores))
    if lo == hi:
        return lo
    grid = np.linspace(lo, hi, n_grid)
    best_thr, best_f1 = float(np.median(val_scores)), -1.0
    for thr in grid:
        pred = (val_scores >= thr).astype(int)
        frac_pos = pred.mean()
        if frac_pos < min_predicted_positive or frac_pos > max_predicted_positive:
            continue
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _safe_auc(y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def _safe_ap(y_true, scores) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def compute_metrics_row(
    *,
    setting: str,
    regime: str,
    detector: str,
    seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    test_real_mask: np.ndarray,
    test_synth_mask: np.ndarray,
    test_normal_mask: np.ndarray,
    threshold: float,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    fit_time_sec: float,
    note: str = "ok",
    extra: Optional[dict] = None,
) -> dict:
    """Compute the per-seed row in E9's exact schema (extended with ``note``)."""
    test_scores = np.asarray(test_scores, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    pred = (test_scores >= threshold).astype(int)
    if y_test.sum() > 0:
        f1_val = float(f1_score(y_test, pred, zero_division=0))
        pr_val = float(precision_score(y_test, pred, zero_division=0))
        rc_val = float(recall_score(y_test, pred, zero_division=0))
    else:
        f1_val = pr_val = rc_val = float("nan")
    auroc = _safe_auc(y_test, test_scores)
    ap = _safe_ap(y_test, test_scores)
    real_det = float((test_scores[test_real_mask] >= threshold).mean()) \
        if test_real_mask.sum() > 0 else float("nan")
    synth_det = float((test_scores[test_synth_mask] >= threshold).mean()) \
        if test_synth_mask.sum() > 0 else float("nan")
    normal_fpr = float((test_scores[test_normal_mask] >= threshold).mean()) \
        if test_normal_mask.sum() > 0 else float("nan")
    row = {
        "setting": setting,
        "regime": regime,
        "detector": detector,
        "seed": seed,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "n_real_test": int(test_real_mask.sum()),
        "n_synth_test": int(test_synth_mask.sum()),
        "threshold": float(threshold),
        "f1": f1_val,
        "precision": pr_val,
        "recall": rc_val,
        "auroc": auroc,
        "avg_precision": ap,
        "real_det": real_det,
        "synth_det": synth_det,
        "normal_fpr": normal_fpr,
        "fit_time_sec": float(fit_time_sec),
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def nan_row(
    *,
    setting: str,
    regime: str,
    detector: str,
    seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    test_real_mask: np.ndarray,
    test_synth_mask: np.ndarray,
    fit_time_sec: float = 0.0,
    note: str = "skipped",
) -> dict:
    return {
        "setting": setting,
        "regime": regime,
        "detector": detector,
        "seed": seed,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "n_real_test": int(test_real_mask.sum()),
        "n_synth_test": int(test_synth_mask.sum()),
        "threshold": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
        "avg_precision": float("nan"),
        "real_det": float("nan"),
        "synth_det": float("nan"),
        "normal_fpr": float("nan"),
        "fit_time_sec": float(fit_time_sec),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Aggregation + LaTeX helpers
# ---------------------------------------------------------------------------
def aggregate_per_seed(
    per_seed_df: pd.DataFrame,
    group_keys: list[str] = ("setting", "regime", "detector"),
) -> pd.DataFrame:
    """Same aggregation as E9: mean+std for the metric columns + n_seeds."""
    metric_cols = [
        "f1", "precision", "recall", "auroc", "avg_precision",
        "real_det", "synth_det", "normal_fpr", "fit_time_sec",
    ]
    keys = list(group_keys)
    summary = (
        per_seed_df.groupby(keys)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = keys + [f"{m}_{s}" for m, s in summary.columns[len(keys):]]
    summary["n_seeds"] = per_seed_df.groupby(keys).size().values
    return summary


def _fmt_pair(mean, std, kind: str) -> str:
    """E9-compatible formatter; std is rendered in tiny font."""
    if pd.isna(mean):
        return "--"
    if kind == "f1":
        s = 0.0 if pd.isna(std) else float(std)
        return f"{mean:.2f}{{\\tiny$\\pm${s:.2f}}}"
    if kind == "pct":
        s = 0.0 if pd.isna(std) else float(std)
        return f"{100*mean:.0f}{{\\tiny$\\pm${100*s:.0f}}}\\%"
    return f"{mean:.3f}"


def render_wide_table(
    summary: pd.DataFrame,
    detector_order: list[str],
    column_blocks: list[tuple[str, str, str]],
    *,
    metrics: list[tuple[str, str, str]] = None,
    label: str = "tab:sota",
    caption: str = "",
    group_keys: list[str] = ("setting", "regime", "detector"),
    detector_pretty: Optional[dict] = None,
) -> str:
    """Build a two-column-spanning LaTeX table identical in style to E9."""
    if metrics is None:
        metrics = [
            ("f1", "F1", "f1"),
            ("auroc", "AUC", "f1"),
            ("real_det", "Real", "pct"),
            ("synth_det", "Synth", "pct"),
        ]
    n_blocks = len(column_blocks)
    n_metrics = len(metrics)
    n_cols = n_blocks * n_metrics
    col_spec = "l" + "".join(["c"] * n_cols)
    EOL = r"\\"

    summary_indexed = summary.set_index(list(group_keys))

    hdr_block_cells = ["Detector"]
    for _, _, label_text in column_blocks:
        hdr_block_cells.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{label_text}}}")
    hdr_block_line = " & ".join(hdr_block_cells) + " " + EOL

    cmidrules = []
    for k in range(n_blocks):
        a = 2 + k * n_metrics
        b = a + n_metrics - 1
        cmidrules.append(rf"\cmidrule(lr){{{a}-{b}}}")
    cmidrule_line = "".join(cmidrules)

    hdr_metric_cells = [" "]
    for _ in column_blocks:
        for _, lbl, _ in metrics:
            hdr_metric_cells.append(lbl)
    hdr_metric_line = " & ".join(hdr_metric_cells) + " " + EOL

    body_rows = []
    pretty = detector_pretty or {}
    for det in detector_order:
        cells = [pretty.get(det, det)]
        for setting, regime, _ in column_blocks:
            for col, _, kind in metrics:
                try:
                    row = summary_indexed.loc[(setting, regime, det)]
                    mean = row[f"{col}_mean"]
                    std = row[f"{col}_std"]
                except KeyError:
                    mean = float("nan"); std = float("nan")
                cells.append(_fmt_pair(mean, std, kind))
        body_rows.append(" & ".join(cells) + " " + EOL)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \caption{" + caption + r"}")
    lines.append(r"  \label{" + label + r"}")
    lines.append(r"  \footnotesize")
    lines.append(r"  \setlength{\tabcolsep}{1.5pt}")
    lines.append(r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + col_spec + r"@{}}")
    lines.append(r"    \toprule")
    lines.append("    " + hdr_block_line)
    lines.append("    " + cmidrule_line)
    lines.append("    " + hdr_metric_line)
    lines.append(r"    \midrule")
    for r in body_rows:
        lines.append("    " + r)
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular*}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Unsupervised SOTA scoring (5 vendored models from evaluation_ver2/sota_models)
# ---------------------------------------------------------------------------
def score_anomaly(
    method: str,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
    overrides: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train ``method`` on normal-only train, score val and test, return
    ``(val_scores, test_scores, fit_time_sec)``.

    ``method`` is one of ``DCdetector``, ``TimesNet``, ``ModernTCN``,
    ``MEMTO``, ``D3R``.
    ``X_*`` are float arrays shaped ``(N, T, C)``.
    ``overrides`` is a per-method kwargs dict (e.g., epochs/batch_size) merged
    over the defaults baked into the helper.
    """
    add_sota_paths()
    from sota_models.training import (  # type: ignore
        train_dcdetector, train_timesnet, train_moderntcn, train_memto, train_d3r,
    )
    from sota_models.common import reduce_window_score  # type: ignore

    fn = {
        "DCdetector": train_dcdetector,
        "TimesNet":   train_timesnet,
        "ModernTCN":  train_moderntcn,
        "MEMTO":      train_memto,
        "D3R":        train_d3r,
    }[method]

    X_train_normal = np.ascontiguousarray(X_train_normal, dtype=np.float32)
    X_score = np.concatenate(
        [np.ascontiguousarray(X_val, dtype=np.float32),
         np.ascontiguousarray(X_test, dtype=np.float32)],
        axis=0,
    )
    win_size = X_train_normal.shape[1]
    enc_in = X_train_normal.shape[2]

    kw = dict(seed=seed, device=device, verbose=verbose)
    if overrides:
        kw.update(overrides)
    t0 = time.time()
    point_scores = fn(X_train_normal, X_score, win_size, enc_in, **kw)
    fit_time = time.time() - t0

    val_scores = reduce_window_score(point_scores[: X_val.shape[0]], "mean")
    test_scores = reduce_window_score(point_scores[X_val.shape[0]:], "mean")
    return val_scores, test_scores, fit_time


# ---------------------------------------------------------------------------
# SpotLight wrapper
# ---------------------------------------------------------------------------
def score_spotlight(
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 42,
    device: str = "cpu",
    profile: str = "full",
    checkpoint_dir: Optional[str] = None,
    n_jvgan_samples: int = 100,
    n_csdi_samples: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train SpotLight on normal-only train; score val + test.

    ``profile`` controls JVGAN / MRPI epoch counts:
    - ``diagnostic``: 5 / 5 (smoke tests only)
    - ``medium``: 40 / 20
    - ``faithful`` / ``full`` (default): 100 / 200 (paper protocol)

    Returns ``(val_scores, test_scores, fit_time_sec)`` where scores are the
    ``window_score`` confirmed-fraction in [0, 1] returned by SpotLight.
    """
    add_sota_paths()
    import torch as _torch
    _torch.manual_seed(seed)
    np.random.seed(seed)

    from spotlight_baseline.pipeline import (  # type: ignore
        SpotLightPipeline, preprocess_windows,
    )

    profiles = {
        "diagnostic": dict(jvgan_epochs=5, mrpi_epochs=5),
        "medium":     dict(jvgan_epochs=40, mrpi_epochs=20),
        "faithful":   dict(jvgan_epochs=100, mrpi_epochs=200),
        "full":       dict(jvgan_epochs=100, mrpi_epochs=200),
    }
    cfg = profiles.get(profile, profiles["full"])

    X_train_normal = np.ascontiguousarray(X_train_normal, dtype=np.float32)
    X_val = np.ascontiguousarray(X_val, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    T = X_train_normal.shape[1]
    K = X_train_normal.shape[2]

    X_train_pp, scaler = preprocess_windows(X_train_normal)
    X_val_pp, _ = preprocess_windows(X_val, scaler=scaler)
    X_test_pp, _ = preprocess_windows(X_test, scaler=scaler)

    pipe = SpotLightPipeline(n_kpis=K, seq_len=T, device=device)
    t0 = time.time()
    pipe.fit(
        X_train_pp,
        jvgan_epochs=cfg["jvgan_epochs"],
        mrpi_epochs=cfg["mrpi_epochs"],
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )
    val_out = pipe.predict(
        X_val_pp,
        n_jvgan_samples=n_jvgan_samples,
        n_csdi_samples=n_csdi_samples,
        run_causal=False,
        verbose=verbose,
    )
    test_out = pipe.predict(
        X_test_pp,
        n_jvgan_samples=n_jvgan_samples,
        n_csdi_samples=n_csdi_samples,
        run_causal=False,
        verbose=verbose,
    )
    fit_time = time.time() - t0
    return (
        val_out["window_score"].astype(np.float64),
        test_out["window_score"].astype(np.float64),
        fit_time,
    )


# ---------------------------------------------------------------------------
# Foundation-model supervised heads
# ---------------------------------------------------------------------------
def _device_for_torch(prefer_mps: bool = True) -> str:
    """Pick a torch device string (mps -> cuda -> cpu)."""
    import torch as _torch
    if prefer_mps and _torch.backends.mps.is_available():
        return "mps"
    if _torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _setup_ssl_for_huggingface() -> None:
    """Best-effort SSL fix for macOS so HuggingFace downloads do not fail."""
    try:
        import ssl, certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass


def _bce_loss_with_pos_weight(y_train: np.ndarray, device: str):
    import torch as _torch
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - y_train.sum()), 1)
    pw = _torch.tensor(neg / pos, dtype=_torch.float32, device=device)
    return _torch.nn.BCEWithLogitsLoss(pos_weight=pw)


def _train_head_on_emb(
    train_emb: np.ndarray,
    y_train: np.ndarray,
    val_emb: np.ndarray,
    y_val: np.ndarray,
    test_emb: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    device: str,
    head: str = "linear",  # 'linear' | 'mlp'
    hidden: int = 256,
    dropout: float = 0.2,
    batch_size: int = 64,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a binary BCE head on (train_emb, y_train); return (val_p, test_p).

    ``head='mlp'`` is used for Toto (matches Foundation_Models_TelecomTS.ipynb).
    """
    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader, TensorDataset

    _torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = train_emb.shape[1]

    if head == "mlp":
        model = _nn.Sequential(
            _nn.Linear(in_dim, hidden),
            _nn.GELU(),
            _nn.Dropout(dropout),
            _nn.Linear(hidden, 1),
        )
    else:
        model = _nn.Linear(in_dim, 1)
    model = model.to(device)

    crit = _bce_loss_with_pos_weight(y_train, device)
    opt = _torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    Xt = _torch.from_numpy(train_emb).float()
    yt = _torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with _torch.no_grad():
        val_logits = model(_torch.from_numpy(val_emb).float().to(device)).squeeze(-1).cpu().numpy()
        test_logits = model(_torch.from_numpy(test_emb).float().to(device)).squeeze(-1).cpu().numpy()
    val_p = 1.0 / (1.0 + np.exp(-val_logits))
    test_p = 1.0 / (1.0 + np.exp(-test_logits))
    return val_p, test_p


# ----- MOMENT --------------------------------------------------------------
def encode_moment(
    X_btc: np.ndarray,
    *,
    weights_dir: Optional[Path] = None,
    n_channels: int = 16,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, bool]:
    """Compute frozen MOMENT classification logits as a 2-D feature, used as
    embeddings for the binary head. Falls back to deterministic surrogate
    embeddings if MOMENT cannot be loaded.

    Returns ``(features, loaded)`` with ``features`` shape ``(N, 2)`` for the
    real backbone (MOMENT outputs class logits we feed forward to a 2->1 head).
    """
    add_sota_paths()
    if device is None:
        device = _device_for_torch()
    _setup_ssl_for_huggingface()
    try:
        from momentfm import MOMENTPipeline  # type: ignore
        import torch as _torch
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "MOMENT-1-large"
        src = str(weights_dir) if weights_dir.is_dir() else "AutonLab/MOMENT-1-large"
        pipe = MOMENTPipeline.from_pretrained(
            src,
            model_kwargs={
                "task_name": "classification",
                "n_channels": n_channels,
                "num_class": 2,
            },
        )
        pipe.init()
        for n, p in pipe.named_parameters():
            if "head" not in n.lower() and "classifier" not in n.lower():
                p.requires_grad = False
        pipe = pipe.to(device).eval()
        X_bct = X_btc.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
        Xt = _torch.from_numpy(X_bct)
        feats = []
        with _torch.no_grad():
            for j in range(0, Xt.shape[0], batch_size):
                xb = Xt[j:j+batch_size].to(device)
                out = pipe(x_enc=xb)
                feats.append(out.logits.detach().cpu().float().numpy())
        return np.concatenate(feats, axis=0), True
    except Exception as e:
        print(f"[encode_moment] WARNING: real MOMENT load failed ({e}); using surrogate.")
        return _surrogate_embedding(X_btc, dim=64, key="moment"), False


def _minimal_gluonts_torch_for_toto():
    import gluonts, importlib.util, os, sys, types  # type: ignore
    root = os.path.join(os.path.dirname(gluonts.__file__), "torch")
    aff_path = os.path.join(root, "distributions", "affine_transformed.py")
    spec = importlib.util.spec_from_file_location(
        "gluonts.torch.distributions.affine_transformed", aff_path,
    )
    aff_mod = importlib.util.module_from_spec(spec)
    sys.modules["gluonts.torch.distributions.affine_transformed"] = aff_mod
    spec.loader.exec_module(aff_mod)
    distpkg = types.ModuleType("gluonts.torch.distributions")
    distpkg.AffineTransformed = aff_mod.AffineTransformed
    sys.modules["gluonts.torch.distributions"] = distpkg
    from torch.distributions import StudentT as TorchStudentT
    st_mod = types.ModuleType("gluonts.torch.distributions.studentT")
    class StudentT(TorchStudentT): ...
    st_mod.StudentT = StudentT
    sys.modules["gluonts.torch.distributions.studentT"] = st_mod
    gt = types.ModuleType("gluonts.torch")
    gt.__path__ = [root]
    sys.modules["gluonts.torch"] = gt


def _prepare_gluonts_for_toto():
    import sys
    for k in list(sys.modules):
        if k == "gluonts.torch" or k.startswith("gluonts.torch."):
            del sys.modules[k]
    try:
        from gluonts.torch.distributions import AffineTransformed  # noqa: F401
        from gluonts.torch.distributions.studentT import StudentT  # noqa: F401
    except Exception:
        _minimal_gluonts_torch_for_toto()


def encode_toto(
    X_btc: np.ndarray,
    *,
    weights_dir: Optional[Path] = None,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> tuple[np.ndarray, bool]:
    """Frozen Toto backbone embeddings (mean-pooled to 768 dim).

    Falls back to deterministic surrogate embeddings if the package can not
    be loaded. The Foundation_Models_TelecomTS shim handles GluonTS.
    """
    add_sota_paths()
    if device is None:
        device = _device_for_torch()
    _setup_ssl_for_huggingface()
    try:
        _prepare_gluonts_for_toto()
        from toto.model.toto import Toto  # type: ignore
        import torch as _torch
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Toto-Open-Base-1.0"
        src = str(weights_dir) if weights_dir.is_dir() else "Datadog/Toto-Open-Base-1.0"
        toto = Toto.from_pretrained(src)
        backbone = toto.model.to(device).eval()
        X_bct = X_btc.transpose(0, 2, 1).astype(np.float32)
        N, V, T = X_bct.shape
        TOTO_PATCH = 64
        target_t = ((T + TOTO_PATCH - 1) // TOTO_PATCH) * TOTO_PATCH
        Xt = _torch.from_numpy(X_bct)
        embs = []
        with _torch.no_grad():
            for j in range(0, N, batch_size):
                xb = Xt[j:j+batch_size].to(device)
                b, v, t = xb.shape
                if t < target_t:
                    pad = _torch.zeros(b, v, target_t - t, dtype=xb.dtype, device=device)
                    xb = _torch.cat([xb, pad], dim=2)
                pad_mask = _torch.ones(b, v, target_t, dtype=_torch.bool, device=device)
                if t < target_t:
                    pad_mask[:, :, t:] = False
                id_mask = _torch.zeros(b, v, target_t, dtype=_torch.float32, device=device)
                flat, _, _ = backbone.backbone(xb, pad_mask, id_mask)
                emb = flat.mean(dim=(1, 2)).detach().cpu().float().numpy()
                embs.append(emb)
        return np.concatenate(embs, axis=0), True
    except Exception as e:
        print(f"[encode_toto] WARNING: real Toto load failed ({e}); using surrogate.")
        return _surrogate_embedding(X_btc, dim=768, key="toto"), False


def encode_mantis(
    X_btc: np.ndarray,
    *,
    weights_dir: Optional[Path] = None,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[np.ndarray, bool]:
    """Frozen Mantis-8M per-channel embeddings (concatenated to 16*256 = 4096).

    Falls back to deterministic surrogate embeddings if Mantis cannot be loaded.
    """
    add_sota_paths()
    _setup_ssl_for_huggingface()
    try:
        from mantis.architecture import Mantis8M  # type: ignore
        import torch as _torch
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "Mantis-8M"
        src = str(weights_dir) if weights_dir.is_dir() else "paris-noah/Mantis-8M"
        net = Mantis8M(device="cpu")
        net = net.from_pretrained(src)
        net.eval()
        MANTIS_SEQ_LEN = 512
        X_bct = X_btc.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
        N, C, T = X_bct.shape
        if T > MANTIS_SEQ_LEN:
            X_bct = X_bct[:, :, :MANTIS_SEQ_LEN]
            T = MANTIS_SEQ_LEN
        pad_amt = MANTIS_SEQ_LEN - T
        Xt = _torch.from_numpy(X_bct)
        embs = []
        with _torch.no_grad():
            for j in range(0, N, batch_size):
                xb = Xt[j:j+batch_size]
                b = xb.shape[0]
                xb_uni = xb.reshape(b * C, 1, T)
                if pad_amt > 0:
                    pad = _torch.zeros(b * C, 1, pad_amt, dtype=xb_uni.dtype)
                    xb_uni = _torch.cat([xb_uni, pad], dim=2)
                emb = net(xb_uni)  # (b*C, 256)
                emb = emb.reshape(b, -1).detach().cpu().float().numpy()
                embs.append(emb)
        return np.concatenate(embs, axis=0), True
    except Exception as e:
        print(f"[encode_mantis] WARNING: real Mantis load failed ({e}); using surrogate.")
        return _surrogate_embedding(X_btc, dim=4096, key="mantis"), False


def _surrogate_embedding(X_btc: np.ndarray, *, dim: int, key: str) -> np.ndarray:
    """Deterministic surrogate embedding: project the per-window mean+std+last
    statistics into a fixed random matrix so that experiments still produce a
    valid (but obviously weak) result row when the real foundation model
    weights are unavailable.
    """
    import hashlib
    N, T, C = X_btc.shape
    # 16 channels x 4 statistics = 64-dim summary
    means = X_btc.mean(axis=1)        # (N, C)
    stds = X_btc.std(axis=1)          # (N, C)
    last = X_btc[:, -1, :]            # (N, C)
    first = X_btc[:, 0, :]            # (N, C)
    summary = np.concatenate([means, stds, last, first], axis=1)  # (N, 4C)
    # Stable seed for the random projection so it is reproducible.
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    rng = np.random.default_rng(h)
    W = rng.standard_normal(size=(summary.shape[1], dim)).astype(np.float32)
    return summary.astype(np.float32) @ W


def train_foundation_binary_head(
    model_name: str,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    *,
    seed: int,
    device: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train the binary head for a frozen foundation encoder. Returns
    ``(val_p, test_p, fit_time_sec)``.

    Defaults follow Foundation_Models_TelecomTS.ipynb:
    - MOMENT: 30 epochs, lr=5e-4, head='linear' (the wrapper is a Linear(2,1)).
    - Toto:   30 epochs, lr=5e-4, head='mlp' (768->256->1, dropout 0.2).
    - Mantis: 10 epochs, lr=1e-4, head='linear'.
    """
    if device is None:
        device = _device_for_torch()
    defaults = {
        "MOMENT": dict(epochs=30, lr=5e-4, head="linear"),
        "Toto":   dict(epochs=30, lr=5e-4, head="mlp", hidden=256, dropout=0.2),
        "Mantis": dict(epochs=10, lr=1e-4, head="linear"),
    }
    cfg = defaults.get(model_name, dict(epochs=10, lr=1e-4, head="linear"))
    if overrides:
        cfg.update(overrides)

    t0 = time.time()
    val_p, test_p = _train_head_on_emb(
        Z_train, y_train, Z_val, y_val, Z_test,
        seed=seed, device=device,
        epochs=cfg.get("epochs", 10),
        lr=cfg.get("lr", 1e-4),
        head=cfg.get("head", "linear"),
        hidden=cfg.get("hidden", 256),
        dropout=cfg.get("dropout", 0.2),
    )
    return val_p, test_p, time.time() - t0


# ---------------------------------------------------------------------------
# Lightweight end-to-end supervised models (TimesNet-lite, InceptionTime-lite,
# PatchTST). All take ``(N, 128, 16)`` raw windows.
# ---------------------------------------------------------------------------
def _build_e2e_model(name: str, *, in_channels: int = 16, n_classes: int = 2,
                     seq_len: int = 128):
    """Construct a fresh end-to-end PyTorch model on CPU. Imports are lazy."""
    import torch.nn as _nn
    import torch as _torch
    if name == "TimesNet-lite":
        model = _TimesNetLite(in_channels=in_channels, n_classes=n_classes)
    elif name == "InceptionTime-lite":
        model = _InceptionTimeLite(in_channels=in_channels, n_classes=n_classes)
    elif name == "PatchTST":
        model = build_patchtst_classifier(seq_len=seq_len, n_channels=in_channels,
                                          n_classes=n_classes)
    else:
        raise ValueError(name)
    return model


def train_e2e_binary(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    device: Optional[str] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    in_channels: int = 16,
    seq_len: int = 128,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train a small e2e PyTorch classifier with CrossEntropyLoss on (0,1).
    Returns ``(val_p, test_p, fit_time_sec)`` where p is the class-1 softmax.
    """
    import torch as _torch
    import torch.nn as _nn
    if device is None:
        device = _device_for_torch()
    _torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_e2e_model(name, in_channels=in_channels, n_classes=2,
                             seq_len=seq_len).to(device)
    opt = _torch.optim.Adam(model.parameters(), lr=lr)
    crit = _nn.CrossEntropyLoss()

    Xtr = _torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    ytr = _torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    Xva = _torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
    Xte = _torch.from_numpy(np.ascontiguousarray(X_test, dtype=np.float32))
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i+batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    fit_time = time.time() - t0

    model.eval()
    val_p, test_p = [], []
    with _torch.no_grad():
        for i in range(0, Xva.shape[0], batch_size):
            xb = Xva[i:i+batch_size].to(device)
            p = _torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
            val_p.append(p)
        for i in range(0, Xte.shape[0], batch_size):
            xb = Xte[i:i+batch_size].to(device)
            p = _torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
            test_p.append(p)
    return np.concatenate(val_p), np.concatenate(test_p), fit_time


# Inlined copies of the E8 model classes so we do not have to import the
# generated notebook. Kept identical except for default ``n_classes=2``.
class _TimesBlockLite:
    pass  # placeholder; real class below


def _make_lite_classes():
    """Return the TimesNet-lite and InceptionTime-lite classes lazily so
    importing this module does not require torch.
    """
    import torch.nn as _nn
    import torch as _torch

    class TimesBlockLite(_nn.Module):
        def __init__(self, ch: int, hidden: int = 128):
            super().__init__()
            self.dw = _nn.Conv1d(ch, ch, kernel_size=5, padding=2, groups=ch)
            self.pw1 = _nn.Conv1d(ch, hidden, kernel_size=1)
            self.pw2 = _nn.Conv1d(hidden, ch, kernel_size=1)
            self.norm = _nn.GroupNorm(8 if ch >= 8 else 1, ch)
            self.act = _nn.GELU()

        def forward(self, x):
            h = self.dw(x); h = self.pw1(h); h = self.act(h); h = self.pw2(h)
            return self.act(self.norm(x + h))

    class TimesNetLite(_nn.Module):
        def __init__(self, in_channels: int = 16, n_classes: int = 2, ch: int = 64):
            super().__init__()
            self.stem = _nn.Conv1d(in_channels, ch, kernel_size=1)
            self.blocks = _nn.Sequential(TimesBlockLite(ch), TimesBlockLite(ch))
            self.gap = _nn.AdaptiveAvgPool1d(1)
            self.head = _nn.Linear(ch, n_classes)

        def forward(self, x_btc):
            x = x_btc.transpose(1, 2).contiguous()
            x = self.stem(x); x = self.blocks(x)
            x = self.gap(x).squeeze(-1)
            return self.head(x)

    class InceptionModule(_nn.Module):
        def __init__(self, in_ch: int, n_filters: int = 32,
                     kernel_sizes=(9, 19, 39), bottleneck_ch: int = 32):
            super().__init__()
            use_bottleneck = in_ch > 1 and bottleneck_ch is not None
            self.bottleneck = (_nn.Conv1d(in_ch, bottleneck_ch, 1, bias=False)
                               if use_bottleneck else _nn.Identity())
            ch = bottleneck_ch if use_bottleneck else in_ch
            self.convs = _nn.ModuleList([
                _nn.Conv1d(ch, n_filters, kernel_size=k, padding=k // 2, bias=False)
                for k in kernel_sizes
            ])
            self.maxpool_branch = _nn.Sequential(
                _nn.MaxPool1d(3, stride=1, padding=1),
                _nn.Conv1d(in_ch, n_filters, 1, bias=False),
            )
            out_ch = n_filters * (len(kernel_sizes) + 1)
            self.bn = _nn.BatchNorm1d(out_ch)
            self.act = _nn.ReLU(inplace=True)

        def forward(self, x):
            z = self.bottleneck(x)
            outs = [c(z) for c in self.convs]
            outs.append(self.maxpool_branch(x))
            return self.act(self.bn(_torch.cat(outs, dim=1)))

    class InceptionTimeLite(_nn.Module):
        def __init__(self, in_channels: int = 16, n_classes: int = 2, n_filters: int = 32):
            super().__init__()
            m1 = InceptionModule(in_channels, n_filters=n_filters)
            out1 = n_filters * 4
            m2 = InceptionModule(out1, n_filters=n_filters)
            out2 = n_filters * 4
            m3 = InceptionModule(out2, n_filters=n_filters)
            out3 = n_filters * 4
            self.modules_list = _nn.ModuleList([m1, m2, m3])
            self.gap = _nn.AdaptiveAvgPool1d(1)
            self.head = _nn.Linear(out3, n_classes)

        def forward(self, x_btc):
            x = x_btc.transpose(1, 2).contiguous()
            for m in self.modules_list:
                x = m(x)
            x = self.gap(x).squeeze(-1)
            return self.head(x)

    return TimesNetLite, InceptionTimeLite


# Module-level wrappers so ``_build_e2e_model`` can call them without
# constructing classes inside every call.
def _TimesNetLite(in_channels: int = 16, n_classes: int = 2, ch: int = 64):
    TNL, _ = _make_lite_classes()
    return TNL(in_channels=in_channels, n_classes=n_classes, ch=ch)


def _InceptionTimeLite(in_channels: int = 16, n_classes: int = 2, n_filters: int = 32):
    _, ITL = _make_lite_classes()
    return ITL(in_channels=in_channels, n_classes=n_classes, n_filters=n_filters)


# ---------------------------------------------------------------------------
# PatchTST classification glue (uses sota_clones/tslib)
# ---------------------------------------------------------------------------
@dataclass
class _PatchTSTConfig:
    task_name: str = "classification"
    seq_len: int = 128
    pred_len: int = 0
    enc_in: int = 16
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    factor: int = 1
    activation: str = "gelu"
    num_class: int = 2


def build_patchtst_classifier(*, seq_len: int = 128, n_channels: int = 16,
                              n_classes: int = 2):
    """Construct a PatchTST classifier from sota_clones/tslib.

    Wraps the tslib ``Model`` class with a small ``forward(x)`` shim so it
    matches the (B, T, C) -> (B, n_classes) calling convention used elsewhere
    in our project.
    """
    add_sota_paths()
    import torch.nn as _nn
    # Importing PatchTST loads ``layers.Transformer_EncDec`` etc. from
    # ``sota_clones/tslib/layers`` (added to sys.path by add_sota_paths).
    from models.PatchTST import Model as TSLibPatchTST  # type: ignore

    cfg = _PatchTSTConfig(seq_len=seq_len, enc_in=n_channels, num_class=n_classes)
    inner = TSLibPatchTST(cfg)

    class _PatchTSTBinaryClassifier(_nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x_btc):
            # tslib expects classification(x_enc, x_mark_enc); we pass None.
            return self.m(x_btc, None, None, None)

    return _PatchTSTBinaryClassifier(inner)


__all__ = [
    "SplitData",
    "add_sota_paths",
    "make_unsup_dataset",
    "make_sup_dataset",
    "select_threshold",
    "select_threshold_continuous",
    "compute_metrics_row",
    "nan_row",
    "aggregate_per_seed",
    "render_wide_table",
    "score_anomaly",
    "score_spotlight",
    "encode_moment",
    "encode_toto",
    "encode_mantis",
    "train_foundation_binary_head",
    "train_e2e_binary",
    "build_patchtst_classifier",
]
