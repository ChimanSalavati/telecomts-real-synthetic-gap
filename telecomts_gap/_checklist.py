"""Operator-facing pre-deployment audit checklist.

A single source of truth for both ``telecomts-audit --print-checklist`` and
the top-level ``CHECKLIST.md`` artifact. The text follows the operator
workflow described in Section 4.2 of the CIKM 2026 paper.
"""
from __future__ import annotations

CHECKLIST_TEXT = """\
Pre-deployment synthetic-benchmark audit checklist
==================================================

Operator workflow for deciding whether a synthetic telecom-anomaly
benchmark can support model-promotion decisions. Each step has a
concrete deliverable. The audit refuses to certify the benchmark
unless steps 1 through 5 produce evidence and step 6 returns a PASS
or a verdict the operator chooses to act on.

1. Label anomaly windows by origin
   -- For every anomalous window, attach an ``anomaly_origin`` field
      whose value is one of: ``controlled_real``, ``synthetic``,
      ``field_real``.
   -- ``controlled_real`` covers anomalies injected through physical
      RF or hardware mechanisms and observed through a real RAN
      stack in a testbed environment.
   -- ``synthetic`` covers KPI-perturbed, generated, or simulator-
      injected anomalies that do not exercise the physical layer.
   -- ``field_real`` covers passively observed outages from deployed
      operational networks.

2. Report recall by origin, not only aggregate F1
   -- Compute origin-conditioned recall for every candidate detector.
   -- Aggregate F1 alone hides operating-regime coverage failures and
      is not sufficient certification evidence for deployment.

3. Test controlled-real vs synthetic anomaly distributions
   -- Run a Classifier Two-Sample Test (C2ST) and an RBF-MMD with a
      permutation null between the two pools' KPI-summary vectors.
   -- ``telecomts_gap.origin_audit`` returns both.

4. Train on synthetic, test on controlled-real at the natural anomaly rate
   -- The deployment-relevant TSTR protocol. If recall on controlled-real
      collapses while aggregate F1 stays high, the benchmark cannot
      certify the detector.

5. Estimate the smallest controlled-real calibration budget
   -- Sweep the fraction ``f`` of controlled-real windows added to
      training. Find the smallest ``f`` that recovers operator-target
      recall on a held-out controlled-real test set.
   -- ``telecomts_gap.calibration_budget`` returns the per-fraction
      recall curve and the recommended budget.

6. Operator verdict
   -- ``PASS``: standard model selection is permitted.
   -- ``GAP_DETECTED``: controlled-real calibration is required;
      re-audit after collecting at least the recommended budget.
   -- ``ORIGIN_INCOMPLETE_SYNTHETIC_ONLY``: benchmark-based
      certification is blocked until either a controlled-real source
      is added or operator-confirmed real-fault evidence is available.

Required user inputs
--------------------
- KPI windows in a tabular form, plus the per-window ``anomaly_origin``
  label.
- Optional deployment-context metadata for split-sensitivity audits.

Output
------
- A single JSON verdict (printed by ``telecomts-audit`` or returned by
  the ``POST /audit/origin`` route) containing per-origin recall, the
  C2ST/MMD gap statistic, the recommended calibration budget, and the
  one-line operator verdict above.
"""


def get_checklist() -> str:
    """Return the operator checklist as plain text."""
    return CHECKLIST_TEXT
