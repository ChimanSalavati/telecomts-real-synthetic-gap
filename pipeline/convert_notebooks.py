"""Convert the experiment Jupyter notebooks into executable Python runners.

For every ``experiments/<EXP>/<name>.ipynb`` this writes a sibling
``experiments/<EXP>/run_<exp>.py`` containing the notebook's code cells, wrapped
so that:

* ``_shared`` is importable regardless of the working directory, and
* the notebook's cwd-relative ``./results`` / ``./figures`` / ``./manifests``
  land in the centralized ``artifacts/<EXP>/`` tree (honoring
  ``TELECOMTS_GAP_OUTPUT_DIR``).

The generated files are committed so the repo ships runnable Python, not just
notebooks. Regenerate after editing a notebook with::

    python pipeline/convert_notebooks.py            # convert all
    python pipeline/convert_notebooks.py --check     # verify they're in sync

Markdown cells and IPython magics (``%``/``!`` lines) are dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
# Original notebooks are archived here once converted; the converter still
# regenerates the runners from them so they remain the source of truth.
ARCHIVE_DIR = REPO_ROOT / "archive" / "notebooks"

# Folders whose notebooks we convert, mapped to the output runner filename.
NOTEBOOK_EXPERIMENTS: dict[str, str] = {
    "E1_dataset_split_leakage_audit": "E1",
    "E2_distribution_gap_robustness": "E2",
    "E4_real_calibration_learning_curve": "E4",
    "E9_multidetector_transfer_audit": "E9",
    "E9b_leave_one_anomaly_out_audit": "E9b",
    "E14_supervised_sota_transfer": "E14",
    "E20_audit_demo": "E20",
    "S1_spotlight_origin_distributional_check": "S1",
}


def _extract_code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    cells: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        # Drop IPython magics and shell escapes that aren't valid plain Python.
        kept = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))]
        code = "\n".join(kept).rstrip()
        if code.strip():
            cells.append(code)
    return cells


def _header(nb_name: str, exp_id: str) -> str:
    return (
        "#!/usr/bin/env python3\n"
        f"# AUTO-GENERATED from {nb_name} by pipeline/convert_notebooks.py -- do not edit by hand.\n"
        "# This is the executable Python conversion of the original Jupyter notebook.\n"
        f"# Standalone:  python experiments/<dir>/run_{exp_id.lower()}.py\n"
        f"# Via runner:  python main.py --experiment {exp_id}\n"
        f'"""{exp_id}: converted notebook runner (offline-aware, centralized outputs)."""\n'
        "from __future__ import annotations\n\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "_EXP_ROOT = Path(__file__).resolve().parent.parent  # experiments/\n"
        "_REPO_ROOT = _EXP_ROOT.parent                        # repo root (telecomts_gap/)\n"
        "for _p in (_EXP_ROOT, _REPO_ROOT):\n"
        "    if str(_p) not in sys.path:\n"
        "        sys.path.insert(0, str(_p))\n"
        'os.environ.setdefault("MPLBACKEND", "Agg")\n\n'
        "from _shared.data_utils import exp_output_dir  # noqa: E402\n"
    )


def _indent(code: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join((pad + ln) if ln.strip() else "" for ln in code.splitlines())


def render(nb_path: Path, exp_id: str) -> str:
    cells = _extract_code_cells(nb_path)
    body = "\n\n".join(cells)
    guard = (
        '\n\nif __name__ == "__main__":\n'
        "    # Redirect the notebook's cwd-relative ./results, ./figures, ./manifests\n"
        f"    # into the centralized artifacts/{exp_id}/ tree.\n"
        f'    os.chdir(exp_output_dir("{exp_id}", ""))\n\n'
    )
    return _header(nb_path.name, exp_id) + guard + _indent(body) + "\n"


def convert_all(check: bool = False) -> int:
    problems = 0
    for folder, exp_id in NOTEBOOK_EXPERIMENTS.items():
        exp_dir = EXPERIMENTS_DIR / folder
        # Notebook lives either alongside the experiment (pre-archive) or under
        # archive/notebooks/<folder>/ (post-archive). Either works as source.
        nbs = sorted(exp_dir.glob("*.ipynb")) or sorted((ARCHIVE_DIR / folder).glob("*.ipynb"))
        if not nbs:
            target = exp_dir / f"run_{exp_id.lower()}.py"
            if not target.exists():
                print(f"[convert] WARNING: no notebook and no runner for {folder}")
                problems += 1
            continue
        nb_path = nbs[0]
        target = exp_dir / f"run_{exp_id.lower()}.py"
        rendered = render(nb_path, exp_id)
        if check:
            current = target.read_text() if target.exists() else ""
            if current != rendered:
                print(f"[convert] OUT OF SYNC: {target.relative_to(REPO_ROOT)}")
                problems += 1
            continue
        target.write_text(rendered)
        print(f"[convert] wrote {target.relative_to(REPO_ROOT)} ({len(rendered.splitlines())} lines)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify generated runners match the notebooks; non-zero exit if not")
    args = ap.parse_args()
    problems = convert_all(check=args.check)
    if args.check and problems:
        print(f"[convert] {problems} runner(s) out of sync; run 'python pipeline/convert_notebooks.py'")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
