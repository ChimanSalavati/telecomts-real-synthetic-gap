"""Execute experiments described by the registry under a :class:`RunConfig`.

Each experiment runs in its own child process (``python experiments/.../run_*.py``)
so heavy, import-time side effects (matplotlib, torch, gluonts) never bleed
between experiments. Configuration is passed through environment variables that
``_shared/data_utils.py`` and the experiment scripts already understand, and
every experiment writes into the centralized ``artifacts/<EXP>/`` tree.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig, REPO_ROOT
from .registry import ExperimentSpec, resolve_experiments

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class ExperimentResult:
    id: str
    title: str
    status: str          # "ok" | "failed" | "skipped"
    reason: str = ""     # populated for skipped/failed
    seconds: float = 0.0
    output_dir: str = ""
    returncode: int | None = None


# --------------------------------------------------------------------------- #
# Skip logic
# --------------------------------------------------------------------------- #
def _skip_reason(spec: ExperimentSpec, cfg: RunConfig) -> str | None:
    if spec.requires_gpu and not cfg.with_gpu:
        return "GPU experiment (pass --with-gpu to run)"
    if cfg.synthetic and not spec.smoke_runnable:
        return "not runnable on the offline synthetic corpus"
    if spec.requires_external_data:
        has_csv = spec.id == "N1" and (cfg.industrial_csv is not None or cfg.synthetic)
        if cfg.skip_external_data and not has_csv:
            return "needs an external dataset not bundled in the repo"
    return None


# --------------------------------------------------------------------------- #
# N1 needs a CSV; synthesize a tiny one for the offline smoke path.
# --------------------------------------------------------------------------- #
def _synthesize_industrial_csv(path: Path, n: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n_pos = max(20, n // 10)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    # A handful of numeric "KPI" columns; anomalies shift two of them.
    df = pd.DataFrame(
        {
            "window_id": np.arange(n),
            "is_anomalous": y,
            "feat_a": rng.normal(0, 1, n) + 2.5 * y,
            "feat_b": rng.normal(5, 2, n) - 1.5 * y,
            "feat_c": rng.normal(-3, 1.5, n),
            "feat_d": rng.gamma(2.0, 1.0, n) + 0.8 * y,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Build the command line for one experiment.
# --------------------------------------------------------------------------- #
def _build_command(spec: ExperimentSpec, cfg: RunConfig, out_dir: Path) -> list[str]:
    cmd = [sys.executable, str(spec.script_path)]
    if spec.id == "N1":
        if cfg.industrial_csv is not None:
            csv = Path(cfg.industrial_csv)
        else:  # synthetic smoke path
            csv = _synthesize_industrial_csv(out_dir / "N1" / "_synthetic_input.csv")
        cmd += [
            "--csv", str(csv),
            "--output", str(out_dir / "N1" / "results" / "hgb_baseline_results.json"),
        ]
        if cfg.synthetic:
            cmd += ["--n-seeds", "2", "--n-folds", "3"]
    return cmd


def run_experiment(spec: ExperimentSpec, cfg: RunConfig) -> ExperimentResult:
    out_dir = cfg.resolved_output_dir
    skip = _skip_reason(spec, cfg)
    if skip is not None:
        return ExperimentResult(spec.id, spec.title, "skipped", reason=skip,
                                output_dir=str(out_dir / spec.id))

    if not spec.script_path.exists():
        return ExperimentResult(spec.id, spec.title, "failed",
                                reason=f"runner script missing: {spec.script_path}",
                                output_dir=str(out_dir / spec.id))

    env = {**os.environ, **cfg.env()}
    cmd = _build_command(spec, cfg, out_dir)

    print(f"\n=== [{spec.id}] {spec.title} ===")
    print(f"    {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=not cfg.verbose,
        text=True,
    )
    secs = time.time() - t0
    if proc.returncode == 0:
        return ExperimentResult(spec.id, spec.title, "ok", seconds=secs,
                                output_dir=str(out_dir / spec.id), returncode=0)
    tail = ""
    if not cfg.verbose and proc.stderr:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
    return ExperimentResult(spec.id, spec.title, "failed", reason=tail or "non-zero exit",
                            seconds=secs, output_dir=str(out_dir / spec.id),
                            returncode=proc.returncode)


def run(cfg: RunConfig) -> list[ExperimentResult]:
    """Run all experiments selected by ``cfg`` and return their results."""
    specs = resolve_experiments(
        experiments=cfg.experiments, benchmarks=cfg.benchmarks, run_all=cfg.run_all
    )
    if not specs:
        raise SystemExit(
            "Nothing to run. Pass --experiment <ID>, --benchmark <NAME>, or --all."
        )

    out_dir = cfg.resolved_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] preset={cfg.preset}  synthetic={cfg.synthetic}  "
          f"with_gpu={cfg.with_gpu}  output_dir={out_dir}")
    print(f"[pipeline] {len(specs)} experiment(s): {', '.join(s.id for s in specs)}")

    results: list[ExperimentResult] = []
    for spec in specs:
        res = run_experiment(spec, cfg)
        results.append(res)
        if res.status == "failed" and cfg.fail_fast:
            print(f"[pipeline] fail-fast: aborting after {spec.id}")
            break

    _print_summary(results)
    _write_summary(results, out_dir)
    return results


def _print_summary(results: list[ExperimentResult]) -> None:
    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    width = max((len(r.id) for r in results), default=2)
    for r in results:
        mark = {"ok": "OK ", "failed": "FAIL", "skipped": "skip"}[r.status]
        extra = f"  ({r.reason.splitlines()[0]})" if r.reason else ""
        secs = f"{r.seconds:6.1f}s" if r.seconds else "      -"
        print(f"  [{mark}] {r.id:<{width}}  {secs}  {r.title}{extra}")
    n_ok = sum(r.status == "ok" for r in results)
    n_fail = sum(r.status == "failed" for r in results)
    n_skip = sum(r.status == "skipped" for r in results)
    print("-" * 64)
    print(f"  ok={n_ok}  failed={n_fail}  skipped={n_skip}")


def _write_summary(results: list[ExperimentResult], out_dir: Path) -> None:
    path = out_dir / "run_summary.json"
    payload = {
        "results": [asdict(r) for r in results],
        "ok": sum(r.status == "ok" for r in results),
        "failed": sum(r.status == "failed" for r in results),
        "skipped": sum(r.status == "skipped" for r in results),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[pipeline] wrote {path}")
