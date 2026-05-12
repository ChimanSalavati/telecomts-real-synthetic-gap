#!/usr/bin/env python3
# AUTO-GENERATED from E20_audit_demo.ipynb by pipeline/convert_notebooks.py -- do not edit by hand.
# This is the executable Python conversion of the original Jupyter notebook.
# Standalone:  python experiments/<dir>/run_e20.py
# Via runner:  python main.py --experiment E20
"""E20: converted notebook runner (offline-aware, centralized outputs)."""
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
    # into the centralized artifacts/E20/ tree.
    os.chdir(exp_output_dir("E20", ""))

    import json
    import numpy as np
    import pandas as pd

    from telecomts_gap import (
        Verdict,
        calibration_budget,
        origin_audit,
    )

    print("telecomts_gap import OK, verdict values:", [v.value for v in Verdict])

    def make_df(n_real: int, n_synth: int, n_feat: int, gap: float, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        X_real = rng.normal(loc=gap, scale=1.0, size=(n_real, n_feat))
        X_syn = rng.normal(loc=0.0, scale=1.0, size=(n_synth, n_feat))
        df = pd.DataFrame(
            np.vstack([X_real, X_syn]), columns=[f"kpi_{j}" for j in range(n_feat)]
        )
        df["anomaly_origin"] = ["controlled_real"] * n_real + ["synthetic"] * n_synth
        return df


    df_pass = make_df(n_real=200, n_synth=200, n_feat=10, gap=0.0)
    res_pass = origin_audit(df_pass, origin_col="anomaly_origin", n_perm=120)
    print(json.dumps(res_pass.to_dict(), indent=2))

    df_gap = make_df(n_real=200, n_synth=200, n_feat=10, gap=3.0)
    res_gap = origin_audit(df_gap, origin_col="anomaly_origin", n_perm=120)
    print(json.dumps(res_gap.to_dict(), indent=2))

    df_synth_only = make_df(n_real=0, n_synth=300, n_feat=10, gap=0.0)
    res_incomplete = origin_audit(df_synth_only, origin_col="anomaly_origin")
    print(json.dumps(res_incomplete.to_dict(), indent=2))
    assert res_incomplete.verdict is Verdict.ORIGIN_INCOMPLETE_SYNTHETIC_ONLY

    rng = np.random.default_rng(0)
    n_feat = 10

    train_norm = rng.normal(0, 1, (600, n_feat))
    train_synth = rng.normal(0, 1, (200, n_feat))
    train_synth[:, 0] += 4.0
    real_pool = rng.normal(0, 1, (60, n_feat))
    real_pool[:, 1] += 4.0
    test_norm = rng.normal(0, 1, (200, n_feat))
    test_real = rng.normal(0, 1, (30, n_feat))
    test_real[:, 1] += 4.0
    test_synth = rng.normal(0, 1, (30, n_feat))
    test_synth[:, 0] += 4.0

    res = calibration_budget(
        train_norm,
        train_synth,
        real_pool,
        test_norm,
        test_real,
        test_synth,
        target_recall=0.7,
        sweep=(0.0, 0.05, 0.10, 0.25, 0.50, 1.00),
        n_seeds=5,
    )

    rows = [
        {
            "fraction": p.fraction,
            "n_added_real": p.n_added_real,
            "real_recall_mean": round(p.real_recall_mean, 3),
            "synth_recall_mean": round(p.synth_recall_mean, 3),
            "normal_fpr_mean": round(p.normal_fpr_mean, 3),
        }
        for p in res.sweep
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(f"Recommended fraction f for target_recall=0.7: {res.recommended_fraction}")
    print(f"That corresponds to {res.recommended_n_added} controlled-real windows added to training.")

    import tempfile
    from pathlib import Path

    from telecomts_gap.cli import main as cli_main

    tmp = Path(tempfile.mkdtemp())
    df_gap.to_csv(tmp / "benchmark.csv", index=False)
    out = tmp / "verdict.json"
    rc = cli_main(
        [
            "--csv",
            str(tmp / "benchmark.csv"),
            "--output",
            str(out),
            "--n-perm",
            "80",
        ]
    )
    print(f"CLI exit code (0=pass, 1=gate fires): {rc}")
    print("verdict JSON:", json.loads(out.read_text())["result"]["verdict"])

    try:
        from fastapi.testclient import TestClient

        from telecomts_gap.api import app

        client = TestClient(app)
        print("GET /audit/health ->", client.get("/audit/health").json())

        df_gap.to_csv(tmp / "benchmark.csv", index=False)
        with open(tmp / "benchmark.csv", "rb") as f:
            r = client.post(
                "/audit/origin",
                files={"file": ("benchmark.csv", f, "text/csv")},
                data={"synthetic_only_from_flag": "false", "n_perm": "80"},
            )
        print("POST /audit/origin ->", r.json()["result"]["verdict"])
    except ImportError:
        print("FastAPI extra not installed -- run: pip install -e .[api]")
