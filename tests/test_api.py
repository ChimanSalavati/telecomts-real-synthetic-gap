"""Smoke tests for the FastAPI router exposed by ``telecomts_gap.api``.

If FastAPI is not installed, the tests are skipped.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


def _have_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_fastapi(),
    reason="FastAPI not installed (install with: pip install -e .[api])",
)


def _make_csv_bytes(*, gap: float, n: int = 150) -> bytes:
    rng = np.random.default_rng(0)
    X_real = rng.normal(loc=gap, scale=1.0, size=(n, 6))
    X_syn = rng.normal(loc=0.0, scale=1.0, size=(n, 6))
    df = pd.DataFrame(np.vstack([X_real, X_syn]), columns=[f"f{j}" for j in range(6)])
    df["anomaly_origin"] = ["controlled_real"] * n + ["synthetic"] * n
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from telecomts_gap.api import app

    client = TestClient(app)
    r = client.get("/audit/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "telecomts_gap" in body["service"]


def test_audit_origin_endpoint_gap_detected():
    from fastapi.testclient import TestClient

    from telecomts_gap.api import app

    client = TestClient(app)
    csv_bytes = _make_csv_bytes(gap=3.0)
    r = client.post(
        "/audit/origin",
        files={"file": ("benchmark.csv", csv_bytes, "text/csv")},
        data={"synthetic_only_from_flag": "false", "n_perm": "60"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["result"]["verdict"] == "gap_detected"
