# `experiments/`

The 16 experiments that regenerate every table and figure in
[`../main_gap_paper.tex`](../main_gap_paper.tex). There are no leftover
analyses; every folder under `experiments/` backs a specific
quantitative claim in the paper.

See the **top-level [`../README.md`](../README.md)** for:

- The full paper-artifact map (Tables 1, 2, 3, 5, 6, 7, 8, 9 + Figures 1-3 -> experiment folder + result CSV).
- The list of all 16 experiments with their one-line purpose.
- Quick-start install instructions.

To regenerate every experiment in this folder at once:

```bash
bash ../scripts/reproduce_all.sh           # CPU-only path (~30 min)
bash ../scripts/reproduce_all.sh --with-gpu # also runs E14 and S3 deep blocks
```

To rerun any single experiment, either execute its notebook with
`jupyter nbconvert --execute --to notebook --inplace ...` or run its
`run_*.py` script directly.

The shared utility module [`_shared/`](_shared/) provides the TelecomTS
loader, the engineered KPI feature extractor, the canonical splits, and
SOTA helpers shared by the deep-model notebooks.
