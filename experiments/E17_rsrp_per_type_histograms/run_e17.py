"""E17 (M2): Per-anomaly-type RSRP distribution audit on TelecomTS.

Tests the M2 alternative interpretation that the real-vs-synthetic gap is partly
a generator-operating-point artifact. For each of the 11 anomaly types (and
Normal), compute per-window mean RSRP and report median, IQR, and the fraction
of windows with mean RSRP below -100 dBm (the "near-edge-of-coverage" cutoff).

Run::
    python experiments/E17_rsrp_per_type_histograms/run_e17.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_helpers import setup_paths, configure_matplotlib  # noqa: E402
setup_paths()
configure_matplotlib()
from _shared.data_utils import load_corpus, KPI_NAMES, exp_output_dir  # noqa: E402

RESULTS = exp_output_dir("E17", "results")
FIGURES = exp_output_dir("E17", "figures")
OUT_CSV = RESULTS / "E17_rsrp_per_type_summary.csv"
OUT_PDF = FIGURES / "E17_rsrp_per_type_distribution.pdf"

# Cell-edge cutoff for the "low-RSRP regime" diagnostic (per the M2 reviewer note).
LOW_RSRP_CUTOFF_DBM = -100.0


def main() -> None:
    print("Loading TelecomTS corpus ...")
    corpus = load_corpus(verbose=True)
    rsrp_idx = KPI_NAMES.index("RSRP")
    win_rsrp = corpus.X[:, :, rsrp_idx]   # (N, 128) per-window RSRP
    win_mean = win_rsrp.mean(axis=1)      # per-window MEAN RSRP

    # Build a per-window category label.
    cats = np.array(corpus.anomaly_type, dtype=object)
    cats = np.where(corpus.y == 0, "Normal", cats)

    rows = []
    for cat in (["Normal"] + sorted(set(cats[cats != "Normal"]))):
        m = (cats == cat)
        n = int(m.sum())
        if n == 0:
            continue
        v = win_mean[m]
        rows.append({
            "category"            : cat,
            "n_windows"           : n,
            "mean_rsrp_dbm"       : float(v.mean()),
            "median_rsrp_dbm"     : float(np.median(v)),
            "q25_rsrp_dbm"        : float(np.percentile(v, 25)),
            "q75_rsrp_dbm"        : float(np.percentile(v, 75)),
            "min_rsrp_dbm"        : float(v.min()),
            "max_rsrp_dbm"        : float(v.max()),
            f"frac_mean_below_{int(LOW_RSRP_CUTOFF_DBM)}dbm": float((v < LOW_RSRP_CUTOFF_DBM).mean()),
        })
    df = pd.DataFrame(rows).sort_values("median_rsrp_dbm").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(df.to_string(index=False))

    SHORT = {
        "Normal": "Normal",
        "Jamming": "Jam (real)",
        "High Network Congestion (Gradual Buildup)": "HC-G",
        "High Network Congestion (Sudden Spike)": "HC-S",
        "Co-Channel Interference (Severe)": "CCI-S",
        "Co-Channel Interference (Mild)": "CCI-M",
        "Faulty RF Filters (Temporal)": "RF-T",
        "Doppler Shift (Severe)": "Dop",
        "Resource Allocation Bugs": "RA-B",
        "Antenna Failure": "Ant",
        "Faulty Handover Algorithm (Too Frequent)": "FH",
        "Buffer Overflow (Gradual Buildup)": "BO",
    }
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    cats_for_plot = list(df["category"])
    data = [win_mean[cats == c] for c in cats_for_plot]
    pos  = list(range(len(cats_for_plot)))
    parts = ax.violinplot(data, positions=pos, vert=False, widths=0.85,
                          showmeans=False, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], cats_for_plot):
        body.set_facecolor("#d62728" if c == "Jamming" else ("#7fb3d5" if c != "Normal" else "#bdbdbd"))
        body.set_edgecolor("#222")
        body.set_alpha(0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels([SHORT.get(c, c) for c in cats_for_plot], fontsize=8)
    ax.axvline(LOW_RSRP_CUTOFF_DBM, ls="--", color="#444", lw=0.7,
               label=f"{int(LOW_RSRP_CUTOFF_DBM)} dBm cell-edge cutoff")
    ax.set_xlabel("per-window mean RSRP (dBm)")
    ax.set_title("TelecomTS per-anomaly-type RSRP distributions\n(real Jamming highlighted)")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
