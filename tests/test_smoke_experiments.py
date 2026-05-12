"""Offline smoke tests for the unified experiment pipeline.

These tests never touch the network or a GPU. They:

1. Assert every registered experiment has an existing, byte-compilable runner
   (this is the "all notebook-equivalent runs" coverage -- every converted
   ``run_<exp>.py`` is checked, even the ones too heavy to execute in CI).
2. Assert the converted runners are in sync with the archived notebooks.
3. Actually execute a fast subset of experiments end-to-end on the offline
   synthetic corpus and assert they produce artifacts and report success.
"""
from __future__ import annotations

import json
import py_compile

import pytest

from pipeline.config import from_preset
from pipeline.registry import EXPERIMENTS
from pipeline import convert_notebooks
from pipeline import runner

# A representative, fast subset that exercises both script-based experiments and
# at least one converted notebook (E20) end-to-end on the synthetic corpus.
FAST_SMOKE_IDS = ["E15", "E16", "E18", "E20", "N1"]


def test_every_experiment_has_a_compilable_runner():
    for spec in EXPERIMENTS.values():
        assert spec.script_path.exists(), f"missing runner for {spec.id}: {spec.script_path}"
        # Byte-compile so syntax errors in any converted notebook are caught.
        py_compile.compile(str(spec.script_path), doraise=True)


def test_converted_runners_in_sync_with_notebooks():
    problems = convert_notebooks.convert_all(check=True)
    assert problems == 0, (
        "converted run_<exp>.py files are out of sync with the archived notebooks; "
        "run `python pipeline/convert_notebooks.py`"
    )


@pytest.mark.parametrize("exp_id", FAST_SMOKE_IDS)
def test_fast_experiment_runs_offline(exp_id, tmp_path):
    cfg = from_preset("smoke").override(
        experiments=[exp_id],
        output_dir=tmp_path,
        verbose=False,
    )
    results = runner.run(cfg)
    assert len(results) == 1
    res = results[0]
    assert res.status == "ok", f"{exp_id} did not succeed: {res.reason}"

    # The run summary is written successfully.
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["failed"] == 0

    # Experiments that emit tables/figures must produce at least one file. E20
    # is an interactive console demo of the toolkit and writes only to stdout.
    if exp_id != "E20":
        produced = list((tmp_path / exp_id).rglob("*"))
        assert any(p.is_file() for p in produced), f"{exp_id} produced no artifacts"
