"""Smoke tests for the ``telecomts-audit`` console script."""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from telecomts_gap.cli import main


def _write_two_pop_csv(tmp_path, *, gap: float, n: int = 200, n_feat: int = 8):
    rng = np.random.default_rng(0)
    X_real = rng.normal(loc=gap, scale=1.0, size=(n, n_feat))
    X_syn = rng.normal(loc=0.0, scale=1.0, size=(n, n_feat))
    df = pd.DataFrame(
        np.vstack([X_real, X_syn]), columns=[f"f{j}" for j in range(n_feat)]
    )
    df["anomaly_origin"] = ["controlled_real"] * n + ["synthetic"] * n
    path = tmp_path / "benchmark.csv"
    df.to_csv(path, index=False)
    return path


def _write_synthetic_only_csv(tmp_path, *, n_norm: int = 200, n_anom: int = 50):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(0, 1, (n_norm + n_anom, 6)),
        columns=[f"f{j}" for j in range(6)],
    )
    df["is_anomalous"] = [0] * n_norm + [1] * n_anom
    path = tmp_path / "synth_only.csv"
    df.to_csv(path, index=False)
    return path


def test_cli_print_checklist():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--print-checklist"])
    assert rc == 0
    out = buf.getvalue()
    assert "Pre-deployment synthetic-benchmark audit checklist" in out
    assert "controlled_real" in out
    assert "ORIGIN_INCOMPLETE_SYNTHETIC_ONLY" in out


def test_cli_emits_gap_detected_when_distributions_differ(tmp_path):
    csv = _write_two_pop_csv(tmp_path, gap=3.0)
    out_path = tmp_path / "verdict.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--csv",
                str(csv),
                "--output",
                str(out_path),
                "--n-perm",
                "60",
            ]
        )
    payload = json.loads(out_path.read_text())
    assert payload["result"]["verdict"] == "gap_detected"
    assert rc == 1  # CI gate fires


def test_cli_emits_origin_incomplete_synthetic_only(tmp_path):
    csv = _write_synthetic_only_csv(tmp_path)
    out_path = tmp_path / "verdict.json"
    rc = main(
        [
            "--csv",
            str(csv),
            "--synthetic-only-from-flag",
            "--output",
            str(out_path),
            "--no-mmd",
            "--no-bh",
        ]
    )
    payload = json.loads(out_path.read_text())
    assert payload["result"]["verdict"] == "origin_incomplete_synthetic_only"
    assert rc == 1


def test_cli_no_mmd_no_bh_short_path(tmp_path):
    """Smoke test for the ``--no-mmd`` and ``--no-bh`` fast-path flags."""
    csv = _write_synthetic_only_csv(tmp_path)
    rc = main(
        [
            "--csv",
            str(csv),
            "--synthetic-only-from-flag",
            "--no-mmd",
            "--no-bh",
        ]
    )
    assert rc == 1
