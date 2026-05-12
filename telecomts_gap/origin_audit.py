"""Origin-aware audit: C2ST + RBF-MMD between two anomaly origins.

Given two anomaly windows pools -- one labelled ``controlled_real`` and one
labelled ``synthetic`` -- return a verdict on whether they occupy the same
KPI operating regime. If they do not, a detector trained on one is unlikely
to transfer to the other; this is the failure mode the CIKM 2026 paper
documents.

When the pipeline ships only one anomaly origin (the common case for many
public and operator benchmarks: real Normal traffic + synthetic injection,
no controlled-real fault labels), the verdict is
``ORIGIN_INCOMPLETE_SYNTHETIC_ONLY`` and the operator gate refuses to
certify the benchmark for operational model selection.

References
----------
Salavati et al., "TelecomAudit: Origin-Aware Benchmark Auditing and
Calibration for 5G Anomaly Detection", CIKM 2026.
Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests", ICLR 2017.
Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Verdict(str, enum.Enum):
    """Audit verdict returned to the operator gate."""

    PASS = "pass"
    """Controlled-real and synthetic anomalies occupy the same operating
    regime; the benchmark is safe for operational model selection."""

    GAP_DETECTED = "gap_detected"
    """The two origins are statistically distinguishable. Calibration is
    required before the detector can be promoted."""

    ORIGIN_INCOMPLETE_SYNTHETIC_ONLY = "origin_incomplete_synthetic_only"
    """The benchmark contains only synthetic anomalies; no controlled-real
    origin is available to compare against. The audit refuses to certify
    the benchmark for operational model selection until a controlled-real
    source is added OR operator tickets validate it in shadow mode."""

    ORIGIN_INCOMPLETE_REAL_ONLY = "origin_incomplete_real_only"
    """Benchmark contains only controlled-real anomalies; no synthetic
    perturbations exist to enable train-on-synthetic/test-on-real audits.
    Returned only for completeness; rare in practice."""

    NO_ANOMALIES_PRESENT = "no_anomalies_present"
    """The benchmark CSV contains only Normal traffic and no labelled
    anomalies of any origin. There is nothing for the audit to
    distinguish; the audit returns this verdict so downstream
    pipelines can treat such CSVs as baseline-traffic captures rather
    than benchmarks."""


@dataclass
class OriginAuditResult:
    """Result returned by :func:`origin_audit`.

    All numeric fields are ``None`` when the verdict is one of the
    ``ORIGIN_INCOMPLETE_*`` or ``NO_ANOMALIES_PRESENT`` cases.
    """

    n_controlled_real: int
    n_synthetic: int
    c2st_accuracy: float | None
    c2st_auroc: float | None
    mmd_norm: float | None
    mmd_p_value: float | None
    bh_significant_features: int | None
    n_features_tested: int
    verdict: Verdict
    notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# C2ST -----------------------------------------------------------------------


def _c2st_5fold_hgb(
    X_real: np.ndarray, X_synth: np.ndarray, *, seed: int = 0
) -> tuple[float, float]:
    """5-fold stratified Classifier-Two-Sample Test using HGB.

    Returns ``(accuracy, AUROC)``. Accuracy close to 0.5 means the two
    populations are indistinguishable; accuracy close to 1.0 means they
    occupy disjoint regions of feature space.
    """
    X = np.vstack([X_real, X_synth])
    y = np.concatenate(
        [np.zeros(len(X_real), dtype=int), np.ones(len(X_synth), dtype=int)]
    )
    if min(len(X_real), len(X_synth)) < 5:
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        return float(accuracy_score(y, (proba >= 0.5).astype(int))), float(
            roc_auc_score(y, proba)
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs: list[float] = []
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X[tr], y[tr])
        proba_te = clf.predict_proba(X[te])[:, 1]
        accs.append(float(accuracy_score(y[te], (proba_te >= 0.5).astype(int))))
        aucs.append(float(roc_auc_score(y[te], proba_te)))
    return float(np.mean(accs)), float(np.mean(aucs))


# MMD ------------------------------------------------------------------------


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Biased estimator of squared MMD with an RBF kernel."""
    XX = _rbf_kernel(X, X, sigma)
    YY = _rbf_kernel(Y, Y, sigma)
    XY = _rbf_kernel(X, Y, sigma)
    return float(XX.mean() + YY.mean() - 2.0 * XY.mean())


def _rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True)
    sq = AA + BB.T - 2.0 * A @ B.T
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-sq / (2.0 * sigma * sigma + 1e-12))


def _median_heuristic_sigma(
    X: np.ndarray, Y: np.ndarray, rng: np.random.Generator
) -> float:
    """Median pairwise distance over a subsample, used as RBF bandwidth."""
    pool = np.vstack([X, Y])
    n = pool.shape[0]
    k = min(n, 200)
    idx = rng.choice(n, size=k, replace=False)
    sub = pool[idx]
    diffs = np.sqrt(
        np.maximum(
            np.sum(sub * sub, axis=1, keepdims=True)
            + np.sum(sub * sub, axis=1)
            - 2.0 * sub @ sub.T,
            0.0,
        )
    )
    iu = np.triu_indices_from(diffs, k=1)
    med = float(np.median(diffs[iu])) if iu[0].size else 1.0
    return max(med, 1e-6)


def _mmd_with_permutation_null(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Observed MMD^2 divided by mean of the permutation null, plus a p-value.

    The ``mmd_norm`` form (observed / null-mean) matches the CIKM 2026 paper
    so the in-paper numbers and the deployed pipeline numbers are
    directly comparable.
    """
    rng = np.random.default_rng(seed)
    sigma = _median_heuristic_sigma(X_real, X_synth, rng)
    observed = _rbf_mmd2(X_real, X_synth, sigma)

    pool = np.vstack([X_real, X_synth])
    n_r = len(X_real)
    nulls = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(pool.shape[0])
        Xa = pool[perm[:n_r]]
        Xb = pool[perm[n_r:]]
        nulls[i] = _rbf_mmd2(Xa, Xb, sigma)
    null_mean = float(nulls.mean()) + 1e-12
    p_value = float((nulls >= observed).sum() + 1) / float(n_perm + 1)
    return observed / null_mean, p_value


# Per-feature significance ---------------------------------------------------


def _bh_significant_count(
    X_real: np.ndarray, X_synth: np.ndarray, alpha: float = 0.05
) -> int:
    """Per-feature Kolmogorov-Smirnov + Benjamini-Hochberg FDR correction.

    Returns the count of features for which we reject H0 (same
    distribution) at FDR <= alpha.
    """
    from scipy import stats

    n_feat = X_real.shape[1]
    p_values = np.empty(n_feat, dtype=float)
    for j in range(n_feat):
        a = X_real[:, j]
        b = X_synth[:, j]
        if np.var(a) == 0 and np.var(b) == 0:
            p_values[j] = 1.0
            continue
        _, p = stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
        p_values[j] = float(p) if np.isfinite(p) else 1.0

    order = np.argsort(p_values)
    ranked = p_values[order]
    thresh = np.arange(1, n_feat + 1) / n_feat * alpha
    passed = ranked <= thresh
    if not passed.any():
        return 0
    cutoff = int(np.max(np.where(passed)[0]))
    return int(cutoff + 1)


# Public API -----------------------------------------------------------------


def origin_audit(
    df: pd.DataFrame,
    *,
    origin_col: str,
    feature_cols: list[str] | None = None,
    controlled_real_label: str | None = None,
    synthetic_label: str | None = None,
    do_mmd: bool = True,
    do_bh: bool = True,
    n_perm: int = 200,
    seed: int = 0,
) -> OriginAuditResult:
    """Audit whether ``controlled_real`` and ``synthetic`` anomaly pools
    occupy the same KPI operating regime.

    Parameters
    ----------
    df
        Dataframe of anomaly windows. May include Normal rows; only rows
        with an origin label in ``{controlled_real_label, synthetic_label}``
        are used for the C2ST + MMD + BH-FDR analysis.
    origin_col
        Name of the column in ``df`` that distinguishes anomaly origins.
        Operators typically add an ``anomaly_origin`` column upstream
        of the audit; alternatively, use :func:`telecomts_gap.cli` with
        ``--synthetic-only-from-flag`` to build one automatically from a
        binary ``is_anomalous`` flag.
    feature_cols
        Numeric KPI feature columns. If ``None``, all numeric columns of
        ``df`` other than ``origin_col`` are used.
    controlled_real_label, synthetic_label
        Strings used in ``origin_col`` to identify the two pools.
        Defaults: ``"controlled_real"`` and ``"synthetic"``.

    Returns
    -------
    OriginAuditResult
        C2ST accuracy/AUROC, MMD norm + p-value, count of BH-significant
        features, and a :class:`Verdict` the operator gate consumes.

    Notes
    -----
    If only one origin is populated, the audit refuses to certify and
    emits ``ORIGIN_INCOMPLETE_SYNTHETIC_ONLY`` (or its real-only twin).
    This is the operator-facing verdict for benchmarks that ship real
    Normal traffic plus only synthetic anomaly injection.
    """
    cr_label = controlled_real_label or "controlled_real"
    syn_label = synthetic_label or "synthetic"

    if origin_col not in df.columns:
        raise KeyError(
            f"origin_col={origin_col!r} not in dataframe columns: "
            f"{list(df.columns)[:10]}..."
        )

    sel_cr = df[origin_col].astype(str) == cr_label
    sel_syn = df[origin_col].astype(str) == syn_label
    n_cr = int(sel_cr.sum())
    n_syn = int(sel_syn.sum())

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c != origin_col and pd.api.types.is_numeric_dtype(df[c])
        ]
    feature_cols = list(feature_cols)
    n_features = len(feature_cols)

    if n_cr == 0 and n_syn > 0:
        return OriginAuditResult(
            n_controlled_real=0,
            n_synthetic=n_syn,
            c2st_accuracy=None,
            c2st_auroc=None,
            mmd_norm=None,
            mmd_p_value=None,
            bh_significant_features=None,
            n_features_tested=n_features,
            verdict=Verdict.ORIGIN_INCOMPLETE_SYNTHETIC_ONLY,
            notes=(
                "Benchmark contains synthetic anomalies only "
                f"(n={n_syn}); no controlled-real anomaly origin is "
                "labelled. The audit refuses to certify this benchmark "
                "for operational model selection. Operator gate: do not "
                "promote until (i) a controlled-real anomaly source is "
                "added, or (ii) operator-validated real tickets confirm "
                "precision/recall in shadow mode."
            ),
        )

    if n_syn == 0 and n_cr > 0:
        return OriginAuditResult(
            n_controlled_real=n_cr,
            n_synthetic=0,
            c2st_accuracy=None,
            c2st_auroc=None,
            mmd_norm=None,
            mmd_p_value=None,
            bh_significant_features=None,
            n_features_tested=n_features,
            verdict=Verdict.ORIGIN_INCOMPLETE_REAL_ONLY,
            notes=(
                f"Benchmark contains controlled-real anomalies only "
                f"(n={n_cr}); no synthetic perturbations exist."
            ),
        )

    if n_cr == 0 and n_syn == 0:
        return OriginAuditResult(
            n_controlled_real=0,
            n_synthetic=0,
            c2st_accuracy=None,
            c2st_auroc=None,
            mmd_norm=None,
            mmd_p_value=None,
            bh_significant_features=None,
            n_features_tested=n_features,
            verdict=Verdict.NO_ANOMALIES_PRESENT,
            notes=(
                "The CSV contains only Normal traffic and no labelled "
                "anomalies; the audit has nothing to distinguish. "
                "Treat as a baseline-traffic capture, not a benchmark."
            ),
        )

    # Two populations are present: run C2ST + optionally MMD + BH-FDR.
    # ``copy=True`` because pandas can hand us read-only numpy views.
    X_cr = np.array(df.loc[sel_cr, feature_cols].to_numpy(dtype=np.float64), copy=True)
    X_syn = np.array(df.loc[sel_syn, feature_cols].to_numpy(dtype=np.float64), copy=True)

    # Drop columns that are all-NaN in either pool, then fill remaining NaNs
    # with column medians computed over the union. Handles operator CSVs
    # where some flow features are populated only in attacks or only in
    # Normal windows.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        keep = ~(
            np.all(np.isnan(X_cr), axis=0) | np.all(np.isnan(X_syn), axis=0)
        )
    if not np.all(keep):
        X_cr = X_cr[:, keep]
        X_syn = X_syn[:, keep]
        n_features = int(keep.sum())
    union = np.vstack([X_cr, X_syn])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(union, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    if np.any(np.isnan(X_cr)):
        nan_mask = np.isnan(X_cr)
        X_cr[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    if np.any(np.isnan(X_syn)):
        nan_mask = np.isnan(X_syn)
        X_syn[nan_mask] = np.take(medians, np.where(nan_mask)[1])

    c2st_acc, c2st_auc = _c2st_5fold_hgb(X_cr, X_syn, seed=seed)

    mmd_norm: float | None
    mmd_p: float | None
    if do_mmd:
        mmd_norm, mmd_p = _mmd_with_permutation_null(
            X_cr, X_syn, n_perm=n_perm, seed=seed
        )
    else:
        mmd_norm, mmd_p = None, None

    bh_sig: int | None
    if do_bh:
        bh_sig = _bh_significant_count(X_cr, X_syn)
    else:
        bh_sig = None

    if c2st_acc < 0.6 and (mmd_norm is None or mmd_norm < 3.0):
        verdict = Verdict.PASS
        notes = (
            f"C2ST accuracy {c2st_acc:.3f} is near chance and MMD norm "
            f"{mmd_norm}; controlled-real and synthetic origins occupy "
            "the same KPI operating regime."
        )
    else:
        verdict = Verdict.GAP_DETECTED
        notes = (
            f"C2ST accuracy {c2st_acc:.3f} (AUROC {c2st_auc:.3f}) and "
            f"MMD norm {mmd_norm} indicate the two origins are "
            "statistically distinguishable; calibration with "
            "controlled-real windows is required before the detector "
            "can be promoted (see calibration_budget)."
        )

    return OriginAuditResult(
        n_controlled_real=n_cr,
        n_synthetic=n_syn,
        c2st_accuracy=c2st_acc,
        c2st_auroc=c2st_auc,
        mmd_norm=mmd_norm,
        mmd_p_value=mmd_p,
        bh_significant_features=bh_sig,
        n_features_tested=n_features,
        verdict=verdict,
        notes=notes,
    )
