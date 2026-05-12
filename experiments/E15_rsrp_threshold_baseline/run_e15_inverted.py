"""E15-extension: 'right-direction' RSRP rules (mean(RSRP) > tau).

Real Jamming sits at high RSRP (~ -76 dBm) while synthetic anomalies and Normal
windows both sit at low RSRP. So the 'wrong-direction' rule (mean(RSRP) < tau)
in run_e15.py inherits the synthetic-only learning bias and gets 0% real recall.
This script tests the 'right-direction' rule (mean(RSRP) > tau) to show that
ONE bit of real-anomaly knowledge -- knowing the right sign on the dominant KPI --
recovers most of the Jamming detection that no amount of synthetic-only training
achieves.

Run:
    python experiments/E15_rsrp_threshold_baseline/run_e15_inverted.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_helpers import setup_paths  # noqa: E402
setup_paths()
from _shared.data_utils import load_corpus, KPI_NAMES, make_fullscale_split  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

OUT_CSV = HERE / "results" / "E15_rsrp_threshold_inverted.csv"
THRESHOLDS_DBM = [-85, -90]  # right-direction rules: anomaly := mean(RSRP) > tau


def main() -> None:
    corpus = load_corpus(verbose=True)
    rsrp_idx = KPI_NAMES.index("RSRP")
    win_mean_rsrp = corpus.X[:, :, rsrp_idx].mean(axis=1)
    sp = make_fullscale_split(corpus, seed=42)
    test_idx = sp["test"]
    y_test = corpus.y[test_idx]
    real_mask  = (corpus.anomaly_origin[test_idx] == "real")
    synth_mask = (corpus.anomaly_origin[test_idx] == "synthetic")
    norm_mask  = (corpus.y[test_idx] == 0)

    rows = []
    for tau in THRESHOLDS_DBM:
        scores = win_mean_rsrp[test_idx]
        pred = (scores > tau).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0) if y_test.sum() else float("nan")
        try:
            auroc = float(roc_auc_score(y_test, scores))   # higher RSRP = more anomalous
        except Exception:
            auroc = float("nan")
        real_recall  = float(pred[real_mask].mean())   if real_mask.sum() else float("nan")
        synth_recall = float(pred[synth_mask].mean())  if synth_mask.sum() else float("nan")
        normal_fpr   = float(pred[norm_mask].mean())   if norm_mask.sum() else float("nan")
        rows.append({
            "rule": f"mean(RSRP) > {tau}",
            "f1": float(f1),
            "auroc": auroc,
            "real_recall": real_recall,
            "synth_recall": synth_recall,
            "normal_fpr": normal_fpr,
        })
        print(f"  mean(RSRP) > {tau}: F1 = {f1:.3f}, real = {real_recall:.3f}, "
              f"synth = {synth_recall:.3f}, FPR = {normal_fpr:.3f}, AUROC = {auroc:.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
