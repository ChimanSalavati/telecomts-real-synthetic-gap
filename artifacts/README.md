# `artifacts/` — centralized run outputs

Every experiment launched through the unified runner

```bash
python main.py --experiment E4 --preset paper
```

writes **all** of its outputs under this directory, namespaced by experiment id:

```
artifacts/
  E4/
    results/   # CSV / JSON tables
    figures/   # PDF / PNG figures
    tables/    # generated LaTeX table fragments (when applicable)
  E16/
    results/
  ...
```

The output root is configurable with `--output-dir` (or the
`TELECOMTS_GAP_OUTPUT_DIR` environment variable), so you can point a batch run
at scratch storage without touching the repo.

The generated payloads (CSV/PDF/JSON) are intentionally **git-ignored** so a
clean checkout stays small and reruns never dirty the working tree. The
committed, paper-backing reference outputs live next to each experiment under
`experiments/<EXP>/results/` and in the top-level `figures/` directory used by
`main_gap_paper.tex`. Re-running an experiment here lets you diff your fresh
numbers against those committed references.
