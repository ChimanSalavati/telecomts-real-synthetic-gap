"""HTTP-facing wrapper around the audit module.

The submodule exposes a FastAPI ``APIRouter`` that any existing service
can mount to expose the audit endpoints:

.. code-block:: python

    from fastapi import FastAPI
    from telecomts_gap.api import router as audit_router

    app = FastAPI()
    app.include_router(audit_router, prefix="/audit", tags=["audit"])

The router is decoupled from the inference path so shadow-mode evaluation
of candidate benchmark CSVs can run independently of the existing
detector serving path.

A pre-built FastAPI app for quick demonstration is also available as
``telecomts_gap.api.app``::

    uvicorn telecomts_gap.api:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

from .audit_endpoint import router

try:  # pragma: no cover - the API extra may not be installed
    from fastapi import FastAPI

    app = FastAPI(
        title="telecomts-gap audit",
        version="0.1.0",
        description=(
            "Origin-aware benchmark audit for telecom anomaly-detection "
            "pipelines. See https://github.com/ChimanSalavati/"
            "telecomts-real-synthetic-gap"
        ),
    )
    app.include_router(router, prefix="/audit", tags=["audit"])
except Exception:  # pragma: no cover
    app = None  # type: ignore[assignment]


__all__ = ["router", "app"]
