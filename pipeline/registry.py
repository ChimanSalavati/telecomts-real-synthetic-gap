"""Registry of every experiment behind ``main_gap_paper.tex``.

Each :class:`ExperimentSpec` records where the experiment's runnable Python
lives, which paper artifact it backs, and the resource/data requirements the
runner uses to decide whether it can run under a given preset.

The notebook-derived runners (``run_<exp>.py``) are produced from the original
Jupyter notebooks by ``pipeline/convert_notebooks.py``; the script-based ones
(E15-E18, S2, S3, N1) were always plain Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    title: str
    benchmark: str               # telecomts | spotlight | industrial | demo
    folder: str                  # directory under experiments/
    script: str                  # runnable .py inside that folder
    paper_artifact: str          # which table/figure/section it produces
    source: str                  # "notebook" (converted) or "script"
    requires_gpu: bool = False
    requires_external_data: bool = False
    smoke_runnable: bool = True   # can it run on the offline synthetic corpus?

    @property
    def script_path(self) -> Path:
        return EXPERIMENTS_DIR / self.folder / self.script


# Order follows the paper's narrative (setup -> diagnosis -> repair -> applied).
_SPECS: list[ExperimentSpec] = [
    ExperimentSpec(
        "E1", "Dataset, split, and leakage audit", "telecomts",
        "E1_dataset_split_leakage_audit", "run_e1.py",
        "Table 1 (audit partitions)", "notebook",
        # E1 asserts the exact real-corpus counts, so it only runs on real data.
        smoke_runnable=False,
    ),
    ExperimentSpec(
        "E2", "Distribution-gap robustness (KS / C2ST / MMD)", "telecomts",
        "E2_distribution_gap_robustness", "run_e2.py",
        "Sec 3.4 numbers (RSRP, Cohen's d, C2ST, MMD)", "notebook",
    ),
    ExperimentSpec(
        "E4", "Controlled-real calibration learning curve", "telecomts",
        "E4_real_calibration_learning_curve", "run_e4.py",
        "Table (calibration sweep CI) + calibration figure", "notebook",
    ),
    ExperimentSpec(
        "E9", "Multi-detector transfer audit (tabular)", "telecomts",
        "E9_multidetector_transfer_audit", "run_e9.py",
        "Table 2 (tabular block)", "notebook",
        # Six detectors x several splits; too heavy for the offline smoke path.
        smoke_runnable=False,
    ),
    ExperimentSpec(
        "E9b", "Leave-one-anomaly-out + calibration mini-sweep", "telecomts",
        "E9b_leave_one_anomaly_out_audit", "run_e9b.py",
        "Leave-one-out figure", "notebook",
        smoke_runnable=False,
    ),
    ExperimentSpec(
        "E14", "Supervised SOTA transfer (Toto/Mantis/TimesNet/...)", "telecomts",
        "E14_supervised_sota_transfer", "run_e14.py",
        "Table 2 (deep block)", "notebook",
        requires_gpu=True, smoke_runnable=False,
    ),
    ExperimentSpec(
        "E15", "RSRP-threshold + PHY-only baseline", "telecomts",
        "E15_rsrp_threshold_baseline", "run_e15.py",
        "Sec 3.4 RSRP-threshold baseline", "script",
    ),
    ExperimentSpec(
        "E16", "RSRP / level feature-ablation transfer", "telecomts",
        "E16_rsrp_ablation_transfer", "run_e16.py",
        "Table 3 (feature ablation)", "script",
    ),
    ExperimentSpec(
        "E17", "Per-anomaly-type RSRP distributions", "telecomts",
        "E17_rsrp_per_type_histograms", "run_e17.py",
        "Figure (per-type RSRP)", "script",
    ),
    ExperimentSpec(
        "E18", "Importance-weighting / covariate-shift baselines", "telecomts",
        "E18_importance_weighting_baseline", "run_e18.py",
        "Table (covariate-shift baselines)", "script",
    ),
    ExperimentSpec(
        "E20", "End-to-end telecomts_gap audit demo", "demo",
        "E20_audit_demo", "run_e20.py",
        "Sec 4 toolkit demo", "notebook",
    ),
    ExperimentSpec(
        "N1", "Industrial HGB baseline", "industrial",
        "N1_industrial_hgb_baseline", "run_n1.py",
        "Sec 4.3 HGB F1 baseline", "script",
        requires_external_data=True,
    ),
    ExperimentSpec(
        "S1", "SpotLight origin distributional check", "spotlight",
        "S1_spotlight_origin_distributional_check", "run_s1.py",
        "Sec cross-corpus SpotLight distributional", "notebook",
        requires_external_data=True, smoke_runnable=False,
    ),
    ExperimentSpec(
        "S2", "SpotLight calibration replication", "spotlight",
        "S2_spotlight_calibration_replication", "run_s2.py",
        "Table (SpotLight calibration)", "script",
        requires_external_data=True, smoke_runnable=False,
    ),
    ExperimentSpec(
        "S3", "SpotLight multi-detector transfer", "spotlight",
        "S3_spotlight_multidetector_transfer", "run_s3.py",
        "Table (SpotLight multidetector)", "script",
        requires_gpu=True, requires_external_data=True, smoke_runnable=False,
    ),
]

EXPERIMENTS: dict[str, ExperimentSpec] = {s.id: s for s in _SPECS}

# Benchmark groups expand to ordered experiment ids.
BENCHMARKS: dict[str, list[str]] = {
    "telecomts": [s.id for s in _SPECS if s.benchmark == "telecomts"],
    "spotlight": [s.id for s in _SPECS if s.benchmark == "spotlight"],
    "industrial": [s.id for s in _SPECS if s.benchmark == "industrial"],
    "demo": [s.id for s in _SPECS if s.benchmark == "demo"],
}


def resolve_experiments(
    *,
    experiments: list[str] | None = None,
    benchmarks: list[str] | None = None,
    run_all: bool = False,
) -> list[ExperimentSpec]:
    """Resolve a request into an ordered, de-duplicated list of specs."""
    ids: list[str] = []
    if run_all:
        ids = list(EXPERIMENTS)
    else:
        for b in benchmarks or []:
            if b not in BENCHMARKS:
                raise ValueError(f"Unknown benchmark '{b}'. Choose from {sorted(BENCHMARKS)}.")
            ids.extend(BENCHMARKS[b])
        for e in experiments or []:
            key = e.upper() if e.upper() in EXPERIMENTS else e
            if key not in EXPERIMENTS:
                raise ValueError(f"Unknown experiment '{e}'. Choose from {sorted(EXPERIMENTS)}.")
            ids.append(key)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    ordered: list[ExperimentSpec] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append(EXPERIMENTS[i])
    return ordered
