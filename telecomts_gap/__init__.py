"""Origin-aware benchmark auditing for telecom anomaly-detection pipelines.

This package is the public, MIT-licensed reference implementation of the
audit toolkit described in:

    Salavati et al., "TelecomAudit: Origin-Aware Benchmark Auditing and
    Calibration for 5G Anomaly Detection", CIKM 2026 Applied Research
    Track.

It provides a deployment-time check that a candidate detector's training
benchmark is not silently synthetic-only, i.e. that synthetic anomalies
adequately cover the KPI operating regime occupied by controlled-real
faults. When the audit detects a gap, ``calibration_budget`` estimates
how many controlled-real labels the operator needs to collect before the
existing detector is safe to promote.

Public surface
--------------

- ``origin_audit``       -- C2ST + RBF-MMD between two anomaly origins.
- ``calibration_budget`` -- smallest controlled-real fraction that closes
                            the per-origin recall gap.
- ``Verdict``            -- enum of audit outcomes used by the operator gate.

CLI
---

The console script ``telecomts-audit`` (registered via ``pyproject.toml``)
wraps these two functions for batch use as a pre-deploy CI gate.

HTTP
----

An optional FastAPI router lives in :mod:`telecomts_gap.api`; mount it on
any existing FastAPI app to expose ``POST /audit/origin`` and
``GET /audit/health`` for shadow-mode integration.
"""

from .origin_audit import Verdict, OriginAuditResult, origin_audit
from .calibration_budget import (
    CalibrationBudgetResult,
    CalibrationCurvePoint,
    calibration_budget,
)

__version__ = "0.1.0"

__all__ = [
    "origin_audit",
    "calibration_budget",
    "Verdict",
    "OriginAuditResult",
    "CalibrationBudgetResult",
    "CalibrationCurvePoint",
    "__version__",
]
