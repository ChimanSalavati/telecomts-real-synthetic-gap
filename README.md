# TelecomAudit: Origin-Aware Benchmark Auditing and Calibration for 5G Anomaly Detection

[![CI](https://github.com/ChimanSalavati/telecomts-real-synthetic-gap/actions/workflows/ci.yml/badge.svg)](https://github.com/ChimanSalavati/telecomts-real-synthetic-gap/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Website](https://img.shields.io/badge/website-online-2ea44f.svg)](https://chimansalavati.github.io/telecomts-real-synthetic-gap/)

> **Project website:** **[chimansalavati.github.io/telecomts-real-synthetic-gap](https://chimansalavati.github.io/telecomts-real-synthetic-gap/)** — abstract, headline results, figure gallery, deployment, and reproduction at a glance.

Official, reproducible artifact for [**`main_gap_paper.tex`**](main_gap_paper.tex)
(CIKM 2026 Applied Research Track).

**TelecomAudit** is an *origin-aware benchmark-auditing* framework — a
deployment-oriented evaluation workflow that conditions evaluation on anomaly
**origin** (controlled-real vs. synthetic) to decide whether a benchmark is
reliable enough to promote an anomaly detector. We apply the audit to **two
independent public 5G/Open RAN testbed benchmarks, TelecomTS and SpotLight**,
and validate it as a **pre-production shadow-mode CI gate inside Nokia's
anomaly-detection pipeline**.

This repository contains:

1. [`telecomts_gap/`](telecomts_gap/) — an installable, MIT-licensed Python
   package implementing the audit toolkit (`origin_audit`, `calibration_budget`,
   the `telecomts-audit` CLI, and a `POST /audit/origin` FastAPI route).
2. [`main.py`](main.py) + [`pipeline/`](pipeline/) — a single configurable
   runner that reproduces every table and figure in the paper.
3. [`experiments/`](experiments/) — the per-experiment Python runners and the
   shared corpus/feature/split utilities in [`experiments/_shared/`](experiments/_shared/).
4. [`evidence/industrial/`](evidence/industrial/) — sanitized operator-verdict
   and latency artifacts backing the Nokia pre-production integration (§5.2).

> **TL;DR.** Synthetic-only training preserves synthetic-anomaly recall while
> collapsing on controlled-real Jamming at the natural anomaly rate (recall ~0%
> on the full 6,400-window TelecomTS test set), and standard covariate-shift
> correction does not repair the gap. Adding **18 controlled-real Jamming
> windows** lifts the same detector from ~0 to 0.93 controlled-real recall
> ([E4](experiments/E4_real_calibration_learning_curve/)). On SpotLight, 52
> controlled-real Radio windows recover Radio recall to 0.83
> ([S2](experiments/S2_spotlight_calibration_replication/)). The findings
> replicate across detector families and survive RSRP / level / context-matched
> ablations ([E16](experiments/E16_rsrp_ablation_transfer/)).

---

## Quick start

```bash
git clone https://github.com/ChimanSalavati/telecomts-real-synthetic-gap.git
cd telecomts-real-synthetic-gap

python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"          # toolkit + tests + plotting

# 1. Unit tests for the audit toolkit.
pytest tests/ -q

# 2. See what you can run.
python main.py --list

# 3. Offline smoke: every runnable experiment on a synthetic corpus
#    (no downloads, no GPU, ~90 s).
python main.py --all --preset smoke

# 4. Try the released CLI gate.
telecomts-audit --print-checklist
```

The TelecomTS corpus (~32,000 windows × 128 timesteps × 16 KPIs) is fetched
from [`AliMaatouk/TelecomTS`](https://huggingface.co/datasets/AliMaatouk/TelecomTS)
on first real run and cached under `experiments/_shared/cache/`. SpotLight and
the operator industrial CSV are external (see
[Data availability](#data-availability)).

---

## Running experiments

Everything is driven by a single entry point, [`main.py`](main.py), with a
centralized configuration (presets in [`pipeline/config.py`](pipeline/config.py)).
**Override precedence is `defaults < preset < CLI flag`**, so any individual knob
can be overridden on top of a preset.

```bash
# One experiment, paper settings (real corpus):
python main.py --experiment E16 --preset paper

# A whole benchmark, including the GPU deep block:
python main.py --benchmark telecomts --preset paper --with-gpu

# Everything, offline synthetic smoke:
python main.py --all --preset smoke

# Override knobs on top of a preset:
python main.py --experiment E4 --preset paper --output-dir /scratch/run1
python main.py --benchmark industrial --industrial-csv data/flows.csv
python main.py --experiment E16 --preset smoke --seeds 5 --synthetic-n 3000
```

| Selection | Flag | Example |
|---|---|---|
| Experiment id(s) | `--experiment` / `-e` | `-e E4 -e E16` |
| Benchmark group | `--benchmark` / `-b` | `-b telecomts` |
| Everything | `--all` | `--all` |
| List & exit | `--list` | `--list` |

Benchmarks: `telecomts`, `spotlight`, `industrial`, `demo`.

Presets:

| Preset | Corpus | Seeds | GPU | External data |
|---|---|---|---|---|
| `paper` | real TelecomTS/SpotLight | 10 | opt-in (`--with-gpu`) | yes |
| `quick` | real, public CPU only | 10 | no | skipped |
| `smoke` | offline **synthetic** | 2 | no | skipped |

All outputs land in the centralized [`artifacts/`](artifacts/) tree
(`artifacts/<EXP>/{results,figures,tables}`) with a machine-readable
`artifacts/run_summary.json`. Redirect with `--output-dir` or
`TELECOMTS_GAP_OUTPUT_DIR`.

### Reproduce everything

```bash
bash scripts/reproduce_all.sh                 # CPU public set, real corpora
bash scripts/reproduce_all.sh --with-gpu      # add E14 / S3 deep blocks
bash scripts/reproduce_all.sh --smoke         # offline synthetic
```

---

## Deployment

TelecomAudit is a deployment-oriented benchmark-auditing gate, not a new
detector. It is exercised end-to-end in Section 5 of the paper ("Applied
Deployment Evidence"): an operator workflow, three integration surfaces, and a
pre-production shadow-mode integration inside Nokia's anomaly-detection pipeline.
The full, paper-faithful walkthrough is in
[`deployment/README.md`](deployment/README.md), backed by the released artifacts
in [`evidence/industrial/`](evidence/industrial/).

**Three deployment surfaces, one core** (`telecomts_gap.origin_audit`):

```bash
# (1) CLI CI gate — exits 0 only on `pass`, so CI blocks promotion on the verdict.
telecomts-audit --csv data/benchmark.csv --synthetic-only-from-flag \
  --output evidence/audit_verdict.json        # non-zero exit => do not promote

# (2) Shadow-mode HTTP service (Docker):
docker compose up --build
curl localhost:8765/audit/health
curl -F "file=@data/benchmark.csv" localhost:8765/audit/origin

# (3) Python API: origin_audit(...) / calibration_budget(...)
```

**Measured Nokia pre-production impact (§5.2).** On a 3,958-window / 203-feature
internal timing-flow benchmark, origin-aware auditing *changes the deployment
decision*:

| Reporting lens | Numbers | Decision |
|---|---|---|
| Aggregate (status quo) | AUROC ≈ 0.99, F1 ≈ 0.77 (natural); F1 ≈ 0.92 (balanced) | **accept** |
| Origin-aware (ours) | all anomalies synthetic → `origin_incomplete_synthetic_only` | **hold in shadow** |

Run as a CI gate over 4 CSVs / 15,376 windows / 4.27 h, it held all 3 candidate
detectors in shadow, flagged the baseline-traffic CSV as `no_anomalies_present`,
at **median 0.15 s / CSV** and **< 1 µs / inference** service-path overhead. All
numbers are reproducible from your own per-flow CSV — see
[`deployment/README.md`](deployment/README.md) §5 and
[`evidence/industrial/`](evidence/industrial/).

---

## The audit toolkit

The released, MIT-licensed [`telecomts_gap`](telecomts_gap/) package implements
the audit described in the paper, with a Python API, a CLI, and a FastAPI route
all built on the same core.

```python
import pandas as pd
from telecomts_gap import origin_audit, calibration_budget, Verdict

df = pd.read_csv("benchmark.csv")          # needs an anomaly_origin column
result = origin_audit(df, origin_col="anomaly_origin")
print(result.verdict, result.notes)

if result.verdict == Verdict.GAP_DETECTED:
    budget = calibration_budget(
        train_normal, train_synth, train_real_pool,
        test_normal, test_real, test_synth=test_synth,
        target_recall=0.9,
    )
    print("controlled-real budget:", budget.recommended_n_added, "windows")
```

The same call is reachable from the `telecomts-audit` CLI (exit code 0 only for
`pass`, so CI can gate on it) and the `POST /audit/origin` route. The five
operator verdicts and their actions are in [`CHECKLIST.md`](CHECKLIST.md)
(`telecomts-audit --print-checklist`). An end-to-end demo of all three surfaces
is [E20](experiments/E20_audit_demo/) (`python main.py --experiment E20`).

---

## Paper artifact map

Table/figure numbers follow the order they appear in `main_gap_paper.tex`.

| Paper artifact | Experiment | Key output |
|---|---|---|
| **Table 1** — TelecomTS audit partitions | [E1](experiments/E1_dataset_split_leakage_audit/) | `E1_dataset_split_leakage_audit.csv` |
| **Table 2** — Detector robustness (TelecomTS) | [E9](experiments/E9_multidetector_transfer_audit/) + [E14](experiments/E14_supervised_sota_transfer/) | `E9_transfer_summary.csv`, `E14_sup_summary.csv` |
| **Table 3** — Feature-ablation analysis | [E16](experiments/E16_rsrp_ablation_transfer/) | `E16_summary.csv` |
| **Table 4** — Calibration vs covariate-shift baselines | [E18](experiments/E18_importance_weighting_baseline/) | `E18_da_baseline_summary.csv` |
| **Table 5** — Controlled-real calibration sweep (CI) | [E4](experiments/E4_real_calibration_learning_curve/) | `E4_real_calibration_with_ci.csv` |
| **Table 6** — SpotLight calibration sweep | [S2](experiments/S2_spotlight_calibration_replication/) | `S2_spotlight_calibration_summary.csv` |
| **Table 7** — SpotLight detector robustness | [S3](experiments/S3_spotlight_multidetector_transfer/) | `S3_summary.csv` |
| **Table 8** — Applied evidence / measured utility | [evidence/industrial/](evidence/industrial/) | `*.json` |
| **Figure 1** — Per-anomaly-type RSRP distributions | [E17](experiments/E17_rsrp_per_type_histograms/) | `E17_rsrp_per_type_distribution.pdf` |
| **Figure 2** — Calibration sweep with controls | [E4](experiments/E4_real_calibration_learning_curve/) | `E4_paper_optionA_with_controls.pdf` |
| **Figure 3** — Leave-one-anomaly-out + mini-sweep | [E9b](experiments/E9b_leave_one_anomaly_out_audit/) | `E9b_leave_one_out_recall.pdf`, `E9b_calibration_minisweep.pdf` |
| §4.2 RSRP / Cohen's d / C2ST / MMD numbers | [E2](experiments/E2_distribution_gap_robustness/) | `E2_*.csv` |
| §4.2 RSRP-threshold baseline | [E15](experiments/E15_rsrp_threshold_baseline/) | `E15_rsrp_threshold_summary.csv` |
| §4.4 SpotLight distributional check | [S1](experiments/S1_spotlight_origin_distributional_check/) | `S1_per_kpi_effect_sizes.csv` |
| §5.2 Nokia industrial aggregates | [evidence/industrial/](evidence/industrial/) | `audit_verdict.json`, `audit_diagnostics.json`, ... |
| §5.2 HGB F1 baseline | [N1](experiments/N1_industrial_hgb_baseline/) | `evidence/industrial/hgb_baseline_results.json` |

The committed, paper-backing reference outputs live under
`experiments/<EXP>/results/` and the top-level [`figures/`](figures/) used by the
LaTeX source; fresh `main.py` runs reproduce them into `artifacts/`. Each
experiment runs as a plain Python script; the original notebooks are archived in
[`archive/notebooks/`](archive/notebooks/) and regenerated with
`python pipeline/convert_notebooks.py`.

### The 15 experiments

| Id | Benchmark | Paper artifact | Purpose |
|---|---|---|---|
| E1 | telecomts | Table 1 | Splits, anomaly counts, leakage checks |
| E2 | telecomts | §4.2 | Per-KPI tests, C2ST, MMD, context-matched robustness |
| E4 | telecomts | Table 5, Figure 2 | Controlled-real calibration sweep + matched-budget controls |
| E9 | telecomts | Table 2 (tabular) | Six tabular detectors × evaluation regimes |
| E9b | telecomts | Figure 3 | Leave-one-anomaly-out + per-type calibration mini-sweep |
| E14 | telecomts (GPU) | Table 2 (deep) | Toto, Mantis, TimesNet, InceptionTime, PatchTST |
| E15 | telecomts | §4.2 | RSRP-threshold + PHY-only baselines |
| E16 | telecomts | Table 3 | No-RSRP / no-level feature ablations |
| E17 | telecomts | Figure 1 | Per-anomaly-type RSRP distributions |
| E18 | telecomts | Table 4 | IW-LR (Sugiyama 2007) and uLSIF-lite (Kanamori 2009) baselines |
| E20 | demo | §5 toolkit demo | End-to-end `telecomts_gap` demo |
| N1 | industrial | §5.2 HGB F1 | HGB baseline on an operator per-flow CSV |
| S1 | spotlight | §4.4 | SpotLight per-channel Cohen's d between origins |
| S2 | spotlight | Table 6 | SpotLight calibration sweep |
| S3 | spotlight (GPU) | Table 7 | SpotLight multidetector audit |

---

## Data availability

- **TelecomTS** — public; auto-downloaded from HuggingFace on first real run.
- **SpotLight** (S1/S2/S3) — public Open RAN release; place the `.npz` splits
  where the SpotLight runners expect them (see each `run_s*.py`).
- **Operator industrial CSV** (N1) — Nokia-internal and **not** redistributed;
  the published artifacts are the verdict aggregates and latency measurements in
  `evidence/industrial/`. Pass your own CSV with `--industrial-csv`.

GPU experiments (E14, S3) are skipped unless `--with-gpu` is given.

---

## Reproducibility

`scripts/reproduce_all.sh` runs the public CPU set end-to-end (~30 min, no GPU)
and writes `artifacts/run_summary.json`. See
[`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md) for measured wall-clocks.

---

## Citation

```bibtex
@inproceedings{salavati2026telecomaudit,
  title     = {TelecomAudit: Origin-Aware Benchmark Auditing and Calibration for 5G Anomaly Detection},
  author    = {Salavati, Chiman and Wu, Liang and Wan, Kelly and Darbari, Mayank and Hong, Liangjie},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- The TelecomTS corpus by Feng et al. ([arXiv:2510.06063](https://arxiv.org/abs/2510.06063)).
- The SpotLight Open RAN corpus by Sun et al. (MobiCom 2024).
- Frozen-encoder foundation models: Toto, Mantis, MOMENT.
