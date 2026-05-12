# Deployment & Applied Evidence

> This page is the GitHub counterpart of **§5 "Applied Deployment Evidence"** of
> [`main_gap_paper.tex`](../main_gap_paper.tex). The paper's primary contribution
> is **not a new detector** — it is a *deployment-oriented benchmark-auditing
> gate* that decides whether benchmark evidence is reliable enough to promote a
> model. Everything below maps 1:1 to the paper and to the released
> [`evidence/industrial/`](../evidence/industrial/) artifacts.

The framework provides **three complementary forms of applied evidence** (§5):

1. **Reproducible pipeline** — `main.py` regenerates every public table/figure
   (~30 min, CPU). See the top-level [README](../README.md#reproduce-everything).
2. **Cross-corpus validation** — the same audit + calibration recipe holds on
   two independent public 5G/Open RAN testbeds (TelecomTS and SpotLight).
3. **Pre-production shadow-mode integration** inside Nokia's anomaly-detection
   platform (detailed below).

---

## 1. Operator-facing workflow (§5.1)

The audit is run by an MLOps engineer or benchmark owner **before** a detector
is promoted. The workflow is exactly:

1. Label anomaly windows by **origin** (controlled-real vs synthetic).
2. Report **recall by origin**, not only aggregate \(F_1\)/AUROC.
3. Test the controlled-real vs synthetic distributions with **C2ST / MMD**.
4. Run **train-on-synthetic / test-on-controlled-real** at the natural rate.
5. Estimate the smallest **controlled-real calibration budget** that restores
   origin-conditioned recall.

The toolkit emits one of five operator verdicts — `pass`, `gap_detected`,
`origin_incomplete_synthetic_only`, `origin_incomplete_real_only`,
`no_anomalies_present` — each with a concrete action (full table in
[`../CHECKLIST.md`](../CHECKLIST.md); `telecomts-audit --print-checklist`).

---

## 2. The three deployment surfaces

All built on the same `telecomts_gap.origin_audit` core, so the verdict is
identical across them.

### 2a. CLI as a CI gate (recommended)

`telecomts-audit` **exits 0 only when the verdict is `pass`**, so any CI system
gates promotion on the exit code without parsing JSON:

```bash
pip install -e ".[api]"           # or: pip install telecomts-gap

telecomts-audit \
  --csv data/benchmark.csv \
  --synthetic-only-from-flag \
  --output evidence/audit_verdict.json
echo "exit code: $?"              # 0 = pass, 1 = gap/incomplete/blocked
```

GitHub Actions step that blocks a build when a benchmark cannot certify
controlled-real transfer:

```yaml
- name: Origin-aware benchmark audit gate
  run: |
    pip install telecomts-gap
    telecomts-audit --csv data/benchmark.csv --synthetic-only-from-flag \
      --output audit_verdict.json   # non-zero exit fails the job
```

### 2b. HTTP service (shadow mode)

```bash
docker compose up --build                       # serves on :8765
curl localhost:8765/audit/health                # -> {"status": "ok"}
curl -F "file=@data/benchmark.csv" localhost:8765/audit/origin
```

or without compose:

```bash
docker build -t telecomts-audit . && docker run --rm -p 8765:8765 telecomts-audit
# or: uvicorn telecomts_gap.api:app --host 0.0.0.0 --port 8765
```

The router is decoupled from any inference path, so it runs as a **shadow-mode
side car** — scoring candidate benchmark CSVs without touching the live detector
serving path. Mount it into an existing app instead of running standalone:

```python
from fastapi import FastAPI
from telecomts_gap.api import router as audit_router
app = FastAPI()
app.include_router(audit_router, prefix="/audit", tags=["audit"])
```

### 2c. Python API

```python
import pandas as pd
from telecomts_gap import origin_audit, calibration_budget, Verdict

result = origin_audit(pd.read_csv("benchmark.csv"), origin_col="anomaly_origin")
if result.verdict == Verdict.GAP_DETECTED:
    budget = calibration_budget(..., target_recall=0.9)
    print("controlled-real windows needed:", budget.recommended_n_added)
```

---

## 3. Nokia pre-production integration (§5.2)

The audit was integrated into Nokia's *Agentic Anomaly Detection* platform as a
**pre-production audit module** supporting programmatic APIs, automated CI
gating, and shadow-mode evaluation. It is **not** part of production serving — it
gates benchmark certification and detector-promotion decisions *before* models
move beyond shadow evaluation.

### Measured decision impact — origin-aware auditing flips the decision

On a Nokia internal timing-flow benchmark (**3,958 windows, 203 KPI features**):

| Reporting lens | Numbers | Promotion decision |
|---|---|---|
| **Aggregate (status quo)** | natural-rate AUROC ≈ 0.99, \(F_1\) ≈ 0.77; balanced-rate AUROC ≈ 0.96, \(F_1\) ≈ 0.92 | **accept** (passes the existing gate) |
| **Origin-aware (ours)** | every anomalous window is synthetic → verdict `origin_incomplete_synthetic_only` | **hold in shadow mode** pending real-origin evidence |

A data-sanity diagnostic on the same benchmark confirms the audit fires
correctly on the real 203-feature schema: **C2ST 0.992, AUROC 0.996, MMD 78.1×
the permutation-null mean, p = 0.002** (166 BH-significant features).

> Source: [`evidence/industrial/audit_verdict.json`](../evidence/industrial/audit_verdict.json),
> [`audit_diagnostics.json`](../evidence/industrial/audit_diagnostics.json),
> [`hgb_baseline_results.json`](../evidence/industrial/hgb_baseline_results.json)
> (the last produced by experiment [N1](../experiments/N1_industrial_hgb_baseline/)).

### Shadow-mode usage and overhead

The audit ran as a CI gate over **every** Nokia timing-flow CSV in the current
in-house workspace: **4 CSVs, 15,376 windows, 4.27 hours** of timing-flow data.
It held **all three** candidate detectors in shadow mode and correctly marked the
baseline-traffic CSV as `no_anomalies_present`. The operator-facing output is a
single JSON verdict consumed by the existing CI/CD pipeline.

| Metric | Value | Evidence |
|---|---|---|
| Median audit time / CSV | **0.15 s** | `shadow_mode_usage.json` |
| p95 audit time / CSV | **0.21 s** | `shadow_mode_usage.json` |
| Detectors held in shadow | **3** | `shadow_mode_usage.json` |
| Service-path overhead | **< 1 µs / inference** (0.125 µs gate-read) | `latency_audited_vs_baseline.md` |

The audit gate runs **once per benchmark refresh** (or when the training pool
changes), not per inference — so the per-window cost is a dictionary read and
the full C2ST/MMD audit is amortized over thousands of inferences.

### Deployment lesson (§5.2)

High aggregate benchmark scores are **not** sufficient certification evidence
when anomaly origins are incomplete or distributionally separated. The audit
does not claim a candidate detector is bad; it shows the **benchmark** cannot
certify controlled-real transfer. The correct action is not to tune another
model on the same synthetic benchmark, but to collect controlled-real
calibration evidence, obtain operator-confirmed incidents, or keep the detector
in shadow mode until such evidence exists.

---

## 4. Applied evidence and measured utility (Table 8 / `tab:release`)

| Dimension | Evidence |
|---|---|
| Public release | MIT-licensed repo: source, cached embeddings, env file, shared audit module |
| Reproducibility | Core public audit regenerates the TelecomTS/SpotLight tables/figures in ~30 min, no GPU |
| Benchmarks | TelecomTS, SpotLight, and the Nokia internal timing-flow benchmark |
| Decision impact | Aggregate-\(F_1\) detector accepted under standard reporting, but rejected / held in shadow under origin-aware reporting |
| Calibration utility | TelecomTS: 18 controlled-real windows restore recall to 0.92; SpotLight: 52 windows restore Radio recall to 0.83 |
| Nokia integration | Pre-production audit module with programmatic interfaces, CI gating, and shadow-mode evaluation |
| Runtime overhead | < 1 µs per inference in the measured service path; median CSV audit time 0.15 s |

---

## 5. Reproduce the industrial evidence on a CSV you control

Every number in §3–§4 is reproducible from an operator per-flow CSV with an
`is_anomalous` column. The raw Nokia CSVs are operator-internal and **not**
redistributed; the published artifacts are the JSON aggregates and latency
measurements only. See
[`evidence/industrial/README.md`](../evidence/industrial/README.md) for the
full claim-by-claim mapping and these commands:

```bash
pip install -e ".[api,dev]"

# 1. Operator-facing verdict (what §5.2 quotes first).
telecomts-audit --csv /path/to/flows.csv --synthetic-only-from-flag \
  --output evidence/industrial/audit_verdict.json

# 2. Diagnostic C2ST + MMD + BH-FDR.
python evidence/industrial/scripts/diagnostic_audit.py \
  --csv /path/to/flows.csv --output evidence/industrial/audit_diagnostics.json --n-perm 500

# 3. Shadow-mode batch over multiple CSVs (one path per line in csvs.txt).
python evidence/industrial/scripts/batch_shadow_audit.py \
  --workspace /path/to/workspace --csv-list csvs.txt \
  --output evidence/industrial/shadow_mode_usage.json

# 4. Latency benchmark: HGB baseline vs audit-screened HGB.
python evidence/industrial/scripts/bench_audited_inference.py \
  --csv /path/to/flows.csv --warmup 200 --measure 2000 \
  --results-dir evidence/industrial/

# 5. The HGB aggregate-F1 baseline (Table 8 / §5.2), via the unified runner:
python main.py --experiment N1 --industrial-csv /path/to/flows.csv
```

---

## Performance summary

- **Audit latency**: median 0.15 s / p95 0.21 s per benchmark CSV.
- **Service-path overhead**: < 1 µs per inference (gate runs off the hot path).
- **Footprint**: pure-Python + scikit-learn; no GPU required for the audit.
