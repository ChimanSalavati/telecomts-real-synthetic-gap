#!/usr/bin/env python3
"""TelecomAudit — unified experiment runner.

A single entry point that reproduces any subset of the experiments behind
``main_gap_paper.tex``. Select experiments by id, by benchmark, or run them all;
pick a preset; override any individual knob. Every run writes into the
centralized ``artifacts/`` tree.

Examples
--------
List everything that can be run::

    python main.py --list

Reproduce one experiment on the real corpus (paper settings)::

    python main.py --experiment E16 --preset paper

Reproduce a whole benchmark, including the GPU deep block::

    python main.py --benchmark telecomts --preset paper --with-gpu

Run the full offline smoke set (tiny synthetic corpus, no downloads, no GPU)::

    python main.py --all --preset smoke

Override knobs on top of a preset (precedence: defaults < preset < CLI)::

    python main.py --experiment E4 --preset paper --output-dir /scratch/run1
    python main.py --benchmark industrial --industrial-csv data/flows.csv

Configuration reference: see ``pipeline/config.py`` (RunConfig + PRESETS).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import PRESETS, from_preset
from pipeline.registry import BENCHMARKS, EXPERIMENTS
from pipeline.runner import run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sel = p.add_argument_group("selection (combine freely)")
    sel.add_argument("--experiment", "-e", action="append", default=[], metavar="ID",
                     help="experiment id, e.g. E4 (repeatable)")
    sel.add_argument("--benchmark", "-b", action="append", default=[], metavar="NAME",
                     choices=sorted(BENCHMARKS),
                     help=f"benchmark group: {', '.join(sorted(BENCHMARKS))} (repeatable)")
    sel.add_argument("--all", dest="run_all", action="store_true",
                     help="run every registered experiment")
    sel.add_argument("--list", action="store_true",
                     help="list all experiments and benchmarks, then exit")

    cfg = p.add_argument_group("configuration (override the preset)")
    cfg.add_argument("--preset", default="paper", choices=sorted(PRESETS),
                     help="named configuration preset (default: paper)")
    cfg.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                     help="root output directory (default: ./artifacts)")
    cfg.add_argument("--synthetic", dest="synthetic", action="store_true", default=None,
                     help="force the offline synthetic corpus")
    cfg.add_argument("--no-synthetic", dest="synthetic", action="store_false", default=None,
                     help="force the real corpus (override a synthetic preset)")
    cfg.add_argument("--synthetic-n", type=int, default=None, metavar="N",
                     help="window count for the synthetic corpus")
    cfg.add_argument("--seeds", type=int, default=None, metavar="N",
                     help="override the per-experiment seed count")
    cfg.add_argument("--with-gpu", dest="with_gpu", action="store_true", default=None,
                     help="allow GPU-only experiments (E14, S3)")
    cfg.add_argument("--skip-external-data", dest="skip_external_data",
                     action="store_true", default=None,
                     help="skip experiments needing datasets not bundled in the repo")
    cfg.add_argument("--industrial-csv", type=Path, default=None, metavar="CSV",
                     help="per-flow industrial CSV for the N1 experiment")
    cfg.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=None,
                     help="abort on the first experiment failure")
    cfg.add_argument("--quiet", dest="verbose", action="store_false", default=None,
                     help="suppress child-process output (only show the summary)")
    return p


def _print_listing() -> None:
    print("Benchmarks (use with --benchmark):")
    for name, ids in BENCHMARKS.items():
        print(f"  {name:<11s} -> {', '.join(ids)}")
    print("\nExperiments (use with --experiment):")
    width = max(len(i) for i in EXPERIMENTS)
    for spec in EXPERIMENTS.values():
        flags = []
        if spec.requires_gpu:
            flags.append("gpu")
        if spec.requires_external_data:
            flags.append("external-data")
        if not spec.smoke_runnable:
            flags.append("no-smoke")
        tag = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {spec.id:<{width}}  {spec.title}")
        print(f"  {'':<{width}}  -> {spec.paper_artifact}{tag}")
    print("\nPresets (use with --preset): " + ", ".join(sorted(PRESETS)))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        _print_listing()
        return 0

    cfg = from_preset(args.preset).override(
        experiments=args.experiment or None,
        benchmarks=args.benchmark or None,
        run_all=args.run_all or None,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        synthetic_n=args.synthetic_n,
        seeds=args.seeds,
        with_gpu=args.with_gpu,
        skip_external_data=args.skip_external_data,
        industrial_csv=args.industrial_csv,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )

    results = run(cfg)
    # Non-zero exit if anything failed, so CI / shell callers can gate on it.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
