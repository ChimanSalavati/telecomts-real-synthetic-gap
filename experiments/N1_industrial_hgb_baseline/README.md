# N1 -- Industrial HGB baseline

HGB aggregate-F1 baseline measurement script behind §5.2 ("Nokia
pre-production integration") of
[`../../main_gap_paper.tex`](../../main_gap_paper.tex). Trains a
`HistGradientBoostingClassifier` on a per-flow CSV that has an
`is_anomalous` flag and reports F1 / AUROC / recall / FPR at both the
natural and balanced anomaly rates via stratified k-fold cross-validation.

## Where the output goes

This experiment intentionally has **no local `results/` folder**. Its
output is written to
[`../../evidence/industrial/hgb_baseline_results.json`](../../evidence/industrial/hgb_baseline_results.json)
so it sits next to the other §5.2 evidence files (`audit_verdict.json`,
`audit_diagnostics.json`, `shadow_mode_usage.json`,
`latency_audited_vs_baseline.{csv,md}`).

The script is **not** part of `scripts/reproduce_all.sh` because it
requires an operator-internal per-flow CSV that is not redistributed
under this repository's MIT license. Reviewers with their own per-flow
CSV (with an `is_anomalous` column) can rerun the script in a few
seconds and overwrite the published JSON with their measurement.

## Reproduce

Through the unified runner (recommended):

```bash
python main.py --experiment N1 --industrial-csv /path/to/industrial_anomaly_1s.csv
```

Or directly:

```bash
python experiments/N1_industrial_hgb_baseline/run_n1.py \
    --csv /path/to/industrial_anomaly_1s.csv \
    --output evidence/industrial/hgb_baseline_results.json \
    --n-seeds 10 \
    --n-folds 5
```

See [`run_n1.py`](run_n1.py) for the full protocol description.
