# End-to-end latency: HGB baseline vs audit-screened HGB

Measured on Apple M-series laptop, single-thread CPU, 2000 calls per row, 200-call warmup, single 232-column window.

| Configuration | Median (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std (ms) |
|---|---:|---:|---:|---:|---:|
| HGB inference (baseline) | 15.991 | 39.768 | 121.497 | 21.638 | 30.274 |
| Audit screen + HGB inference | 13.087 | 31.351 | 58.202 | 16.424 | 18.136 |
| Audit screen alone | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Audit gate-read median cost: **0.125 us** (i.e. 0.0001 ms).

The audit gate is run ONCE per benchmark refresh (or whenever the training pool changes), not per inference. The per-window cost is therefore the dict-read latency above. The full origin audit (C2ST + MMD) is amortized over thousands of inferences.
