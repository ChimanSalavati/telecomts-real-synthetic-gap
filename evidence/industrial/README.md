# Industrial-deployment evidence

This folder contains the **operator-verdict** and **latency-measurement**
evidence produced when `telecomts_gap` was integrated into an industrial
5G anomaly-detection pipeline. These are the artifacts behind
**§5.2 "Nokia pre-production integration"** (within §5 "Applied Deployment
Evidence") of the CIKM 2026 paper. A narrative, paper-faithful walkthrough is in
[`../../deployment/README.md`](../../deployment/README.md).

The raw per-flow CSVs are operator-internal data and are not
redistributed. Everything published here is a **JSON aggregate** or a
**latency measurement** -- no raw traffic data.

## What is in this folder

| File | Provenance | §5.2 evidence |
|---|---|---|
| [`audit_verdict.json`](audit_verdict.json) | `telecomts-audit --synthetic-only-from-flag` over the industrial per-flow CSV | "3,958 windows and 203 KPI features ... verdict `origin_incomplete_synthetic_only`" |
| [`audit_diagnostics.json`](audit_diagnostics.json) | [`scripts/diagnostic_audit.py`](scripts/diagnostic_audit.py) | "C2ST accuracy 0.992, AUROC 0.996, MMD 78.1x the permutation-null mean, p=0.002" |
| [`shadow_mode_usage.json`](shadow_mode_usage.json) | [`scripts/batch_shadow_audit.py`](scripts/batch_shadow_audit.py) | "four CSVs, 15,376 windows, 4.27 hours of timing-flow data ... median wall-clock time is 0.15 s per CSV, p95 is 0.21 s" |
| [`latency_audited_vs_baseline.csv`](latency_audited_vs_baseline.csv) | [`scripts/bench_audited_inference.py`](scripts/bench_audited_inference.py) | "service-path overhead is below 1 us per inference on single-thread CPU" |
| [`latency_audited_vs_baseline.md`](latency_audited_vs_baseline.md) | Human-readable summary of the CSV | Same |
| [`hgb_baseline_results.json`](hgb_baseline_results.json) (added by Phase 4) | [`../../experiments/N1_industrial_hgb_baseline/run_n1.py`](../../experiments/N1_industrial_hgb_baseline/run_n1.py) | "F1~0.97 at balanced rate and 0.94 at natural rate" |

## How to reproduce these files on a per-flow CSV you control

The three scripts are parameterized so an operator with access to a
per-flow CSV with an `is_anomalous` column can reproduce every number
above against their own data.

Assuming the package has been installed (`pip install -e .[api,dev]` from
the repo root), run from the repo root:

```bash
# 1. Operator-facing verdict (what the Nokia subsection quotes first).
telecomts-audit \
  --csv /path/to/industrial_anomaly_1s.csv \
  --synthetic-only-from-flag \
  --output evidence/industrial/audit_verdict.json

# 2. Diagnostic C2ST + MMD + BH-FDR on the same CSV.
python evidence/industrial/scripts/diagnostic_audit.py \
  --csv /path/to/industrial_anomaly_1s.csv \
  --output evidence/industrial/audit_diagnostics.json \
  --n-perm 500

# 3. Shadow-mode batch over multiple CSVs.
#    configs/csvs.txt has one path per line, relative to --workspace.
python evidence/industrial/scripts/batch_shadow_audit.py \
  --workspace /path/to/workspace \
  --csv-list /path/to/csvs.txt \
  --output evidence/industrial/shadow_mode_usage.json

# 4. Latency benchmark: HGB baseline vs audit-screened HGB.
python evidence/industrial/scripts/bench_audited_inference.py \
  --csv /path/to/industrial_anomaly_1s.csv \
  --warmup 200 \
  --measure 2000 \
  --results-dir evidence/industrial/
```

All scripts call exactly the same `telecomts_gap.origin_audit` machinery
as the released package, so the numbers above are reproducible on any
similarly-shaped per-flow CSV.

## What this is NOT

- Not the raw operator-traffic CSVs themselves (those are operator-internal).
- Not a live production deployment log -- the latency measurements are on
  commodity laptop hardware as a defensible lower-bound on production
  latency, matching the protocol used by adjacent industry deployment papers.
- Not a live shadow-mode capture. The FastAPI router at
  [`telecomts_gap.api.audit_endpoint`](../../telecomts_gap/api/audit_endpoint.py)
  is *compatible with* any existing service-mesh / shadow-mode
  infrastructure (the paper's §5.2 names the specific operator
  stack the integration targets) but the public artifact demonstrates
  the route via the FastAPI `TestClient` rather than a live mesh.

## Pointer back to the paper

These JSON aggregates back the numbers quoted in §5.2 of
[`../../main_gap_paper.tex`](../../main_gap_paper.tex):

- L517 "$3{,}958$ windows and $203$ KPI features": `audit_verdict.json` -> `feature_count` and the row count.
- L517 "verdict $\texttt{origin\_incomplete\_synthetic\_only}$": `audit_verdict.json` -> `result.verdict`.
- L517 "C2ST $0.992$, AUROC $0.996$, MMD $78.1\times$, $p=0.002$": `audit_diagnostics.json` -> `diagnostic_result.{c2st_accuracy, c2st_auroc, mmd_norm, mmd_p_value}`.
- L520 "four CSVs, $15{,}376$ windows, $4.27$ hours": `shadow_mode_usage.json` -> `total_csvs_audited`, `total_windows_audited`, `total_timing_flow_hours`.
- L520 "median wall-clock time is $0.15$ s per CSV, p95 is $0.21$ s": `shadow_mode_usage.json` -> `median_audit_wall_seconds`, `p95_audit_wall_seconds`.
- L520 "service-path overhead is below $1\,\mu$s per inference": `latency_audited_vs_baseline.md` "Audit gate-read median cost: 0.125 us".
- L520 "all three candidate detectors in shadow mode": `shadow_mode_usage.json` -> `detectors_held_in_shadow=3`.
- L520 "baseline-traffic CSV as $\texttt{no\_anomalies\_present}$": `shadow_mode_usage.json` -> `verdict_distribution.no_anomalies_present=1`.
- L517 "F1~0.97 at balanced rate and 0.94 at natural rate": `hgb_baseline_results.json` (produced by Phase 4 of the public-release pipeline).
