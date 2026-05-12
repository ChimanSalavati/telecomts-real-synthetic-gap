"""Unit tests for the origin-aware audit package.

All fixtures are synthetic Gaussian data; no real-world telemetry is
required. These tests are also the package's correctness contract: the
verdict-string values they assert are the same strings the deployed CI
gate emits in the JSON payload.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from telecomts_gap import Verdict, calibration_budget, origin_audit


def _make_two_pop_df(
    *, n_real: int, n_synth: int, n_feat: int, gap: float, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X_real = rng.normal(loc=gap, scale=1.0, size=(n_real, n_feat))
    X_syn = rng.normal(loc=0.0, scale=1.0, size=(n_synth, n_feat))
    feats = pd.DataFrame(
        np.vstack([X_real, X_syn]), columns=[f"f{j}" for j in range(n_feat)]
    )
    feats["anomaly_origin"] = ["controlled_real"] * n_real + ["synthetic"] * n_synth
    return feats


def test_audit_pass_when_same_distribution():
    df = _make_two_pop_df(n_real=200, n_synth=200, n_feat=10, gap=0.0, seed=0)
    r = origin_audit(df, origin_col="anomaly_origin", do_mmd=True, n_perm=80)
    assert r.verdict == Verdict.PASS
    assert r.c2st_accuracy is not None
    assert 0.4 < r.c2st_accuracy < 0.7  # near chance


def test_audit_gap_when_clearly_different():
    df = _make_two_pop_df(n_real=200, n_synth=200, n_feat=10, gap=3.0, seed=0)
    r = origin_audit(df, origin_col="anomaly_origin", do_mmd=True, n_perm=80)
    assert r.verdict == Verdict.GAP_DETECTED
    assert r.c2st_accuracy is not None and r.c2st_accuracy > 0.8


def test_audit_origin_incomplete_synthetic_only():
    df = _make_two_pop_df(n_real=0, n_synth=300, n_feat=10, gap=0.0, seed=0)
    r = origin_audit(df, origin_col="anomaly_origin")
    assert r.verdict == Verdict.ORIGIN_INCOMPLETE_SYNTHETIC_ONLY
    assert r.n_controlled_real == 0
    assert r.n_synthetic == 300
    assert r.c2st_accuracy is None


def test_audit_no_anomalies_present():
    df = pd.DataFrame(
        {
            "f0": np.zeros(50),
            "f1": np.ones(50),
            "anomaly_origin": ["normal"] * 50,
        }
    )
    r = origin_audit(df, origin_col="anomaly_origin")
    assert r.verdict == Verdict.NO_ANOMALIES_PRESENT


def test_calibration_budget_recovers_target():
    """Two-feature regime mismatch: synthetic and controlled-real shift
    DIFFERENT features, so synth-only training cannot transfer until at
    least one controlled-real window is calibrated in.
    """
    rng = np.random.default_rng(0)
    n_norm, n_synth, n_real_pool = 600, 200, 60
    n_test_real, n_test_synth, n_test_norm = 30, 30, 200
    n_feat = 10

    train_norm = rng.normal(0, 1, (n_norm, n_feat))
    train_synth = rng.normal(0, 1, (n_synth, n_feat))
    train_synth[:, 0] += 4.0
    real_pool = rng.normal(0, 1, (n_real_pool, n_feat))
    real_pool[:, 1] += 4.0
    test_norm = rng.normal(0, 1, (n_test_norm, n_feat))
    test_real = rng.normal(0, 1, (n_test_real, n_feat))
    test_real[:, 1] += 4.0
    test_synth = rng.normal(0, 1, (n_test_synth, n_feat))
    test_synth[:, 0] += 4.0

    res = calibration_budget(
        train_norm,
        train_synth,
        real_pool,
        test_norm,
        test_real,
        test_synth,
        target_recall=0.7,
        sweep=(0.0, 0.1, 0.25, 1.0),
        n_seeds=3,
    )
    p0 = res.sweep[0]
    pf = res.sweep[-1]
    assert p0.real_recall_mean < pf.real_recall_mean, (
        f"Calibration did nothing: f=0 recall={p0.real_recall_mean}, "
        f"f=1 recall={pf.real_recall_mean}"
    )
    assert res.recommended_fraction is not None
