"""B1 (M5): Bootstrap CIs and worst-case-seed for the real-Jamming calibration sweep.

Reads existing per-seed CSVs from E10 (per-anomaly-type) and E4 (curve), computes
2.5/97.5 percentile bootstrap CIs over seeds, the worst-seed minimum, and a
binomial CI over the test-side Jamming positives. Writes a small summary CSV
that the paper can cite directly.

Run::
    python experiments/E4_real_calibration_learning_curve/postprocess_bootstrap.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO_ROOT = EXP_ROOT.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
from _shared.data_utils import exp_output_dir  # noqa: E402

# Read/write through the centralized artifacts/E4/results tree so the bootstrap
# CIs line up with the per-seed CSV the E4 runner just produced.
_E4_RESULTS = exp_output_dir("E4", "results")
E4_PER_SEED  = _E4_RESULTS / "E4_real_calibration_learning_curve_per_seed.csv"
OUT_CSV      = _E4_RESULTS / "E4_real_calibration_with_ci.csv"
N_BOOT       = 2000
SEED         = 42


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact 95% binomial CI for k successes out of n trials (Clopper-Pearson)."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main() -> None:
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(E4_PER_SEED)
    jam = df.copy()
    jam["anomaly_type"] = "Jamming"
    rate_col, n_col = "real_jamming_detection_rate", None
    source = "E4"
    print(f"Using source: {source}  ({len(jam)} rows)")

    rows = []
    for f, g in jam.groupby("fraction"):
        rates = g[rate_col].to_numpy(dtype=float)
        n_seeds = rates.size
        n_test  = int(g[n_col].iloc[0]) if n_col else 50

        # Across-seeds bootstrap CI on the mean.
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            boots[b] = rng.choice(rates, size=n_seeds, replace=True).mean()
        ci_lo_seed = float(np.percentile(boots, 2.5))
        ci_hi_seed = float(np.percentile(boots, 97.5))

        # Per-seed Clopper-Pearson, then averaged for a within-seed binomial sense.
        per_seed_ci = []
        for r in rates:
            k = int(round(r * n_test))
            lo, hi = clopper_pearson_ci(k, n_test)
            per_seed_ci.append((lo, hi))
        cp_lo = float(np.mean([lo for lo, _ in per_seed_ci]))
        cp_hi = float(np.mean([hi for _, hi in per_seed_ci]))

        rows.append({
            "fraction"           : float(f),
            "n_added_real"       : int(round(float(f) * 185)),
            "n_seeds"            : n_seeds,
            "n_test_jamming"     : n_test,
            "mean"               : float(rates.mean()),
            "std"                : float(rates.std(ddof=0)),
            "min_seed"           : float(rates.min()),
            "max_seed"           : float(rates.max()),
            "boot_ci_lo_seeds"   : ci_lo_seed,
            "boot_ci_hi_seeds"   : ci_hi_seed,
            "binom_ci_lo_avg"    : cp_lo,
            "binom_ci_hi_avg"    : cp_hi,
        })

    out = pd.DataFrame(rows).sort_values("fraction")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}\n")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(out.to_string(index=False))

    print("\n--- Quotable summary ---")
    for _, r in out.iterrows():
        print(f"f = {int(r['fraction']*100):3d}% (n_added={int(r['n_added_real']):3d}): "
              f"mean = {r['mean']:.3f}, worst-seed = {r['min_seed']:.3f}, "
              f"95% bootstrap CI over seeds = [{r['boot_ci_lo_seeds']:.3f}, {r['boot_ci_hi_seeds']:.3f}]")


if __name__ == "__main__":
    main()
