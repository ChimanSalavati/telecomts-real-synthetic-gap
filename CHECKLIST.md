# Pre-deployment synthetic-benchmark audit checklist

Operator workflow for deciding whether a synthetic telecom-anomaly
benchmark can support model-promotion decisions. Each step has a
concrete deliverable. The audit refuses to certify the benchmark
unless steps 1 through 5 produce evidence and step 6 returns a
`PASS` or a verdict the operator chooses to act on.

This file is the printed form of the operator workflow described in
Section 4.2 of [main_gap_paper.tex](main_gap_paper.tex) (line ~510). It
is also reachable from the installed package via:

```bash
telecomts-audit --print-checklist
```

so the checklist ships both as a static artifact and as live output of
the released CLI.

---

## 1. Label anomaly windows by origin

- For every anomalous window, attach an `anomaly_origin` field whose
  value is one of: `controlled_real`, `synthetic`, `field_real`.
- **`controlled_real`** covers anomalies injected through physical
  RF or hardware mechanisms and observed through a real RAN stack in
  a testbed environment (e.g. TelecomTS Jamming, SpotLight Radio).
- **`synthetic`** covers KPI-perturbed, generated, or simulator-
  injected anomalies that do not exercise the physical layer.
- **`field_real`** covers passively observed outages from deployed
  operational networks.

## 2. Report recall by origin, not only aggregate F1

- Compute origin-conditioned recall for every candidate detector.
- Aggregate F1 alone hides operating-regime coverage failures and is
  not sufficient certification evidence for deployment.

## 3. Test controlled-real vs synthetic anomaly distributions

- Run a Classifier Two-Sample Test (C2ST) and an RBF-MMD with a
  permutation null between the two pools' KPI-summary vectors.
- `telecomts_gap.origin_audit` returns both, plus per-feature
  KS + Benjamini-Hochberg significance counts.

## 4. Train on synthetic, test on controlled-real at the natural anomaly rate

- The deployment-relevant Train-on-Synthetic / Test-on-Real protocol.
  If recall on controlled-real collapses while aggregate F1 stays
  high, the benchmark cannot certify the detector.

## 5. Estimate the smallest controlled-real calibration budget

- Sweep the fraction $f$ of controlled-real windows added to
  training. Find the smallest $f$ that recovers operator-target
  recall on a held-out controlled-real test set.
- `telecomts_gap.calibration_budget` returns the per-fraction recall
  curve and the recommended budget.

## 6. Operator verdict

| Verdict | Action |
|---|---|
| `PASS` | Standard model selection is permitted. |
| `GAP_DETECTED` | Controlled-real calibration is required; re-audit after collecting at least the recommended budget. |
| `ORIGIN_INCOMPLETE_SYNTHETIC_ONLY` | Benchmark-based certification is blocked until either a controlled-real source is added or operator-confirmed real-fault evidence is available. |
| `ORIGIN_INCOMPLETE_REAL_ONLY` | Rare in practice; benchmark contains controlled-real anomalies but no synthetic perturbations. |
| `NO_ANOMALIES_PRESENT` | The CSV contains only Normal traffic; treat as baseline-traffic capture, not a benchmark. |

---

## Required user inputs

- KPI windows in a tabular form, plus the per-window `anomaly_origin`
  label.
- Optional deployment-context metadata for split-sensitivity audits.

## Output

A single JSON verdict (printed by `telecomts-audit` or returned by
the `POST /audit/origin` route) containing per-origin recall, the
C2ST/MMD gap statistic, the recommended calibration budget, and the
one-line operator verdict above.

## Minimal example (synthetic-only benchmark)

```bash
telecomts-audit \
  --csv /path/to/benchmark.csv \
  --synthetic-only-from-flag \
  --output /path/to/audit_verdict.json
# Exit code 1 with verdict == "origin_incomplete_synthetic_only" gates the CI promotion.
```

## Minimal example (Python API)

```python
import pandas as pd
from telecomts_gap import calibration_budget, origin_audit

df = pd.read_csv("benchmark.csv")
result = origin_audit(df, origin_col="anomaly_origin")
print(result.verdict, result.notes)

if result.verdict.value == "gap_detected":
    budget = calibration_budget(
        train_normal, train_synth, train_real_pool,
        test_normal, test_real, test_synth=test_synth,
        target_recall=0.9,
    )
    print("Recommended controlled-real budget:", budget.recommended_fraction)
```
