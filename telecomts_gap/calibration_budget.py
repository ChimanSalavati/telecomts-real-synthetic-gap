"""Per-corpus controlled-real calibration-budget estimation.

Given a synthetic-only training pool, a (small) controlled-real pool, and a
held-out controlled-real test set, sweep the fraction ``f`` of the
controlled-real pool added to training and find the smallest ``f`` such
that controlled-real recall on the test set reaches the operator's target.

This is the deployment-time complement of :func:`origin_audit`: once the
audit flags a gap, this function tells the operator how many
controlled-real labels they need to collect before the existing detector
is safe to promote.

Mirrors the CIKM 2026 paper's E4 / E16 protocol (HGB on numeric features,
validation-tuned F1 threshold, multi-seed mean) so the deployed numbers
and the in-paper numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve


DEFAULT_SWEEP = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)


@dataclass
class CalibrationCurvePoint:
    """One point on the calibration sweep."""

    fraction: float
    n_added_real: int
    real_recall_mean: float
    real_recall_std: float
    synth_recall_mean: float
    normal_fpr_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationBudgetResult:
    """Result returned by :func:`calibration_budget`."""

    target_recall: float
    sweep: list[CalibrationCurvePoint]
    recommended_fraction: float | None
    recommended_n_added: int | None
    train_real_pool_size: int
    test_real_pool_size: int
    n_seeds: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sweep"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in d["sweep"]]
        return d


def _fit_threshold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[HistGradientBoostingClassifier, float]:
    """Fit HGB and pick the F1-maximizing threshold on the validation set."""
    clf = HistGradientBoostingClassifier(random_state=seed)
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


def calibration_budget(
    train_normal: np.ndarray,
    train_synth: np.ndarray,
    train_real_pool: np.ndarray,
    test_normal: np.ndarray,
    test_real: np.ndarray,
    test_synth: np.ndarray | None = None,
    *,
    target_recall: float = 0.90,
    sweep: tuple[float, ...] = DEFAULT_SWEEP,
    n_seeds: int = 10,
    val_fraction: float = 0.1,
) -> CalibrationBudgetResult:
    """Estimate the smallest controlled-real calibration budget that achieves
    ``target_recall`` on the held-out controlled-real test set.

    Parameters
    ----------
    train_normal, train_synth, train_real_pool, test_normal, test_real, test_synth
        Numpy arrays of shape ``(n_samples, n_features)``. Test sets are held
        out and never used for training. ``train_real_pool`` is the
        controlled-real pool that calibration draws from.
    target_recall
        Operator's required recall on the controlled-real test set.
    sweep
        Calibration fractions of ``train_real_pool`` to test. Defaults to
        the paper's sweep ``{0, 5, 10, 25, 50, 100}%``.
    n_seeds
        Number of HGB random seeds to average over.
    val_fraction
        Fraction of the training pool held out as a per-seed validation
        slice for threshold tuning.

    Returns
    -------
    CalibrationBudgetResult
        Per-fraction recall + the smallest fraction whose mean recall first
        reaches ``target_recall``.
    """
    rng_master = np.random.default_rng(0)
    real_pool_size = int(train_real_pool.shape[0])
    test_real_size = int(test_real.shape[0])

    base_X = np.vstack([train_normal, train_synth])
    base_y = np.concatenate(
        [np.zeros(len(train_normal), dtype=int), np.ones(len(train_synth), dtype=int)]
    )

    if test_synth is None:
        test_synth = np.empty((0, base_X.shape[1]), dtype=base_X.dtype)
    test_X = np.vstack([test_normal, test_real, test_synth])
    real_mask = np.concatenate(
        [
            np.zeros(len(test_normal), dtype=bool),
            np.ones(len(test_real), dtype=bool),
            np.zeros(len(test_synth), dtype=bool),
        ]
    )
    synth_mask = np.concatenate(
        [
            np.zeros(len(test_normal), dtype=bool),
            np.zeros(len(test_real), dtype=bool),
            np.ones(len(test_synth), dtype=bool),
        ]
    )
    normal_mask = np.concatenate(
        [
            np.ones(len(test_normal), dtype=bool),
            np.zeros(len(test_real), dtype=bool),
            np.zeros(len(test_synth), dtype=bool),
        ]
    )

    sweep_points: list[CalibrationCurvePoint] = []
    recommended_f: float | None = None
    recommended_n: int | None = None

    for f in sweep:
        n_add = max(0, int(round(f * real_pool_size)))
        real_recalls: list[float] = []
        synth_recalls: list[float] = []
        normal_fprs: list[float] = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(rng_master.integers(2**31 - 1))
            if n_add > 0 and real_pool_size > 0:
                idx = rng.choice(
                    real_pool_size, size=min(n_add, real_pool_size), replace=False
                )
                added = train_real_pool[idx]
            else:
                added = np.empty((0, base_X.shape[1]), dtype=base_X.dtype)

            X_tr_full = np.vstack([base_X, added])
            y_tr_full = np.concatenate(
                [base_y, np.ones(added.shape[0], dtype=int)]
            )
            n = X_tr_full.shape[0]
            perm = rng.permutation(n)
            n_val = max(2, int(round(val_fraction * n)))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            clf, thr = _fit_threshold(
                X_tr_full[tr_idx],
                y_tr_full[tr_idx],
                X_tr_full[val_idx],
                y_tr_full[val_idx],
                seed=seed,
            )
            scores = clf.predict_proba(test_X)[:, 1]
            pred = (scores >= thr).astype(int)
            if real_mask.sum():
                real_recalls.append(float(pred[real_mask].mean()))
            if synth_mask.sum():
                synth_recalls.append(float(pred[synth_mask].mean()))
            if normal_mask.sum():
                normal_fprs.append(float(pred[normal_mask].mean()))

        point = CalibrationCurvePoint(
            fraction=float(f),
            n_added_real=int(n_add),
            real_recall_mean=float(np.mean(real_recalls)) if real_recalls else float("nan"),
            real_recall_std=float(np.std(real_recalls)) if real_recalls else float("nan"),
            synth_recall_mean=float(np.mean(synth_recalls)) if synth_recalls else float("nan"),
            normal_fpr_mean=float(np.mean(normal_fprs)) if normal_fprs else float("nan"),
        )
        sweep_points.append(point)
        if recommended_f is None and point.real_recall_mean >= target_recall:
            recommended_f = float(f)
            recommended_n = int(n_add)

    notes = (
        f"Sweep {sweep} over {n_seeds} seeds. "
        f"Smallest fraction reaching target_recall={target_recall}: "
        f"{recommended_f}"
    )
    return CalibrationBudgetResult(
        target_recall=target_recall,
        sweep=sweep_points,
        recommended_fraction=recommended_f,
        recommended_n_added=recommended_n,
        train_real_pool_size=real_pool_size,
        test_real_pool_size=test_real_size,
        n_seeds=n_seeds,
        notes=notes,
    )
