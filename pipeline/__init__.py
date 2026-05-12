"""Unified experiment pipeline for the TelecomAudit (CIKM 2026) artifact.

This package turns the collection of per-experiment notebooks and scripts that
back ``main_gap_paper.tex`` into a single, configurable entry point::

    python main.py --experiment E4 --preset paper
    python main.py --benchmark telecomts --preset paper
    python main.py --all --preset smoke

Everything is driven by a centralized :class:`pipeline.config.RunConfig` with
named presets (``paper`` / ``quick`` / ``smoke``) and full command-line
override, and every experiment writes to a single ``artifacts/`` output tree.
"""
from __future__ import annotations

from .config import RunConfig, PRESETS
from .registry import EXPERIMENTS, BENCHMARKS, ExperimentSpec, resolve_experiments

__all__ = [
    "RunConfig",
    "PRESETS",
    "EXPERIMENTS",
    "BENCHMARKS",
    "ExperimentSpec",
    "resolve_experiments",
]
