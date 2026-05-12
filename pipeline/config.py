"""Centralized run configuration with named presets.

A single :class:`RunConfig` controls every knob the pipeline exposes. Values
are resolved with the precedence::

    dataclass defaults  <  named preset  <  explicit CLI flags

so ``--preset smoke --seeds 3`` starts from the smoke preset and then overrides
just the seed count. The CLI layer in ``main.py`` is a thin wrapper that builds
a preset and applies any explicitly-passed flags via :meth:`RunConfig.override`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Repo root = parent of this package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts"


@dataclass
class RunConfig:
    """All tunable parameters for a pipeline run.

    Attributes
    ----------
    experiments:
        Explicit experiment ids to run (e.g. ``["E4", "E16"]``). Empty means
        "use ``benchmarks`` / ``run_all`` instead".
    benchmarks:
        Benchmark groups to expand into experiment ids (e.g. ``["telecomts"]``).
    run_all:
        Run every registered experiment.
    preset:
        Name of the preset this config was built from (informational).
    output_dir:
        Root directory for all experiment outputs. ``None`` -> ``artifacts/``.
    synthetic:
        Use the offline synthetic corpus instead of downloading real data.
    synthetic_n:
        Window count for the synthetic corpus (smoke runs stay small).
    seeds:
        Optional override for the per-experiment seed count. ``None`` keeps
        each experiment's published default (10).
    with_gpu:
        Allow GPU-only experiments (E14, S3) to run.
    skip_external_data:
        Skip experiments that need datasets not bundled in the repo
        (SpotLight, the operator industrial CSV) unless a path is supplied.
    industrial_csv:
        Path to the per-flow industrial CSV for N1. ``None`` lets the runner
        synthesize a tiny stand-in when ``synthetic`` is set.
    fail_fast:
        Abort the whole run on the first experiment failure.
    verbose:
        Stream child-process output to the console.
    """

    experiments: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    run_all: bool = False

    preset: str = "paper"
    output_dir: Path | None = None

    synthetic: bool = False
    synthetic_n: int = 1500
    seeds: int | None = None
    with_gpu: bool = False
    skip_external_data: bool = False
    industrial_csv: Path | None = None

    fail_fast: bool = False
    verbose: bool = True

    # ---------------------------------------------------------------- helpers
    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else DEFAULT_OUTPUT_DIR

    def override(self, **kwargs: Any) -> "RunConfig":
        """Return a copy with the given (non-None) fields replaced.

        ``None`` values are ignored so callers can pass through unset CLI
        options without clobbering preset values.
        """
        updates = {k: v for k, v in kwargs.items() if v is not None}
        unknown = set(updates) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"Unknown RunConfig fields: {sorted(unknown)}")
        return dataclasses.replace(self, **updates)

    def env(self) -> dict[str, str]:
        """Environment variables consumed by ``_shared/data_utils.py`` and the
        experiment scripts, so child processes inherit this configuration."""
        env = {
            "TELECOMTS_GAP_OUTPUT_DIR": str(self.resolved_output_dir),
            "TELECOMTS_GAP_SYNTHETIC": "1" if self.synthetic else "0",
            "TELECOMTS_GAP_SYNTHETIC_N": str(self.synthetic_n),
            "MPLBACKEND": "Agg",
        }
        if self.seeds is not None:
            env["TELECOMTS_GAP_SEEDS"] = str(self.seeds)
        return env


# --------------------------------------------------------------------------- #
# Named presets. Each is a dict of RunConfig overrides applied on top of the
# dataclass defaults.
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, Any]] = {
    # Full paper reproduction on the real corpora. GPU blocks (E14, S3) still
    # require --with-gpu; everything else is CPU-only (~30 min).
    "paper": {
        "synthetic": False,
        "with_gpu": False,
        "skip_external_data": False,
    },
    # Same as paper but explicitly skips anything needing external datasets or a
    # GPU -- a quick "does the CPU/public path still run" pass on real data.
    "quick": {
        "synthetic": False,
        "with_gpu": False,
        "skip_external_data": True,
    },
    # Offline CI smoke: tiny synthetic corpus, few seeds, no downloads, no GPU.
    # Only the smoke-runnable experiments execute; the rest are reported skipped.
    "smoke": {
        "synthetic": True,
        "synthetic_n": 700,
        "seeds": 2,
        "with_gpu": False,
        "skip_external_data": True,
    },
}


def from_preset(name: str, **overrides: Any) -> RunConfig:
    """Build a :class:`RunConfig` from a named preset plus explicit overrides."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from {sorted(PRESETS)}.")
    cfg = RunConfig(preset=name, **PRESETS[name])
    return cfg.override(**overrides) if overrides else cfg
