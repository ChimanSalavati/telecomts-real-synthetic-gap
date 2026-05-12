"""FastAPI route for the origin-aware audit.

Mounted via :mod:`telecomts_gap.api` at ``/audit``:

- ``POST /audit/origin`` (multipart): upload a CSV; receive a JSON audit
  verdict identical to the ``telecomts-audit`` CLI output.
- ``GET  /audit/health``: liveness probe for the audit subsystem.

The endpoint stays single-shot (synchronous) for benchmarks up to a few
thousand windows. For streaming integrations, batch CSVs upstream and
call this endpoint once per benchmark; the audit itself is cheap
relative to data movement.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..cli import _attach_synthetic_only_from_flag, _DEFAULT_IGNORE
from ..origin_audit import origin_audit


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "telecomts_gap.audit"}


@router.post("/origin")
async def audit_origin(
    file: UploadFile = File(...),
    origin_col: str = Form("anomaly_origin"),
    controlled_real_label: str = Form("controlled_real"),
    synthetic_label: str = Form("synthetic"),
    synthetic_only_from_flag: bool = Form(True),
    n_perm: int = Form(200),
    do_mmd: bool = Form(True),
    do_bh: bool = Form(True),
    seed: int = Form(0),
) -> dict[str, Any]:
    """Run an origin-aware audit on an uploaded CSV.

    The CSV should have one row per anomaly-detection window plus an
    ``anomaly_origin`` column distinguishing ``controlled_real`` from
    ``synthetic`` rows. For CSVs that ship only an ``is_anomalous`` flag
    (real Normal traffic + synthetic injection), set
    ``synthetic_only_from_flag=true`` and the endpoint will build the
    origin column itself.
    """
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {exc}") from exc

    if synthetic_only_from_flag:
        try:
            df = _attach_synthetic_only_from_flag(df)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    feature_cols = [
        c
        for c in df.columns
        if c != origin_col
        and c not in _DEFAULT_IGNORE
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    result = origin_audit(
        df,
        origin_col=origin_col,
        feature_cols=feature_cols,
        controlled_real_label=controlled_real_label,
        synthetic_label=synthetic_label,
        do_mmd=do_mmd,
        do_bh=do_bh,
        n_perm=n_perm,
        seed=seed,
    )
    return {
        "csv_filename": file.filename,
        "origin_col": origin_col,
        "feature_count": len(feature_cols),
        "result": result.to_dict(),
    }
