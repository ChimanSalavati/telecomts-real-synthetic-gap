"""Boilerplate helpers used at the top of each experiment notebook.

Keeping this in a single module so notebooks can be regenerated/edited
without diverging on basic concerns (path resolution, plotting style, etc.).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


def setup_paths() -> Path:
    """Add the experiments root to ``sys.path`` so notebooks can do ``from _shared import ...``.

    Returns the experiments root directory.
    """
    here = Path(__file__).resolve()
    experiments_root = here.parent.parent  # _shared/.. => experiments/
    if str(experiments_root) not in sys.path:
        sys.path.insert(0, str(experiments_root))
    # Quiet a few noisy libraries.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return experiments_root


def configure_matplotlib() -> None:
    import matplotlib
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


__all__ = ["setup_paths", "configure_matplotlib"]
