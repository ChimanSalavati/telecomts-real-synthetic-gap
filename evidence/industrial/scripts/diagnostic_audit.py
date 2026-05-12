"""Diagnostic audit run on a per-flow CSV.

Complements ``telecomts-audit --synthetic-only-from-flag`` (which returns
the operator-facing ``origin_incomplete_synthetic_only`` verdict): here we
additionally compute C2ST, MMD, and BH-significant feature count between
Normal traffic and the synthetic anomaly pool, demonstrating that the
audit's diagnostic machinery fires correctly on a real per-flow schema.

Produces ``evidence/industrial/audit_diagnostics.json``.

Run::

    python evidence/industrial/scripts/diagnostic_audit.py \\
        --csv /path/to/industrial_anomaly_1s.csv \\
        --output evidence/industrial/audit_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from telecomts_gap.cli import _DEFAULT_IGNORE
from telecomts_gap.origin_audit import origin_audit


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Diagnostic origin-audit on a per-flow CSV with an is_anomalous "
            "flag. Treats Normal rows as controlled-real for the purpose "
            "of exercising C2ST + MMD + BH-FDR on the deployed schema."
        )
    )
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/industrial/audit_diagnostics.json"),
    )
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[diag] loaded {len(df)} rows from {args.csv}")
    n_normal = int((df["is_anomalous"] == 0).sum())
    n_anom = int((df["is_anomalous"] == 1).sum())
    print(f"[diag] {n_normal} normal, {n_anom} anomalous (synthetic-injection family)")

    # Treat Normal rows as controlled_real for the diagnostic; this exercises
    # the audit machinery on the deployed schema. The operator-facing verdict
    # remains ORIGIN_INCOMPLETE_SYNTHETIC_ONLY via the primary CLI invocation.
    df_diag = df.copy()
    df_diag["anomaly_origin"] = np.where(
        df_diag["is_anomalous"].astype(int) == 1, "synthetic", "controlled_real"
    )

    feature_cols = [
        c
        for c in df_diag.columns
        if c != "anomaly_origin"
        and c not in _DEFAULT_IGNORE
        and pd.api.types.is_numeric_dtype(df_diag[c])
    ]
    print(f"[diag] {len(feature_cols)} numeric KPI features tested")

    result = origin_audit(
        df_diag,
        origin_col="anomaly_origin",
        feature_cols=feature_cols,
        do_mmd=True,
        do_bh=True,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    payload = {
        "csv": str(args.csv),
        "rows": int(len(df)),
        "n_normal": n_normal,
        "n_synthetic": n_anom,
        "n_features": len(feature_cols),
        "diagnostic_result": result.to_dict(),
        "note": (
            "DIAGNOSTIC ONLY. The industrial benchmark used for this "
            "measurement does not contain controlled-real anomaly labels; "
            "this run uses Normal traffic in place of controlled-real to "
            "confirm the audit machinery fires correctly on the deployed "
            "per-flow schema. The operator-facing verdict is the primary "
            "``telecomts-audit --synthetic-only-from-flag`` run, which "
            "returns ``origin_incomplete_synthetic_only`` and refuses to "
            "certify the benchmark for operational model selection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[diag] wrote {args.output}")

    print("\n=== DIAGNOSTIC AUDIT VERDICT ===")
    r = result.to_dict()
    for k, v in r.items():
        if k != "notes":
            print(f"  {k:30s} = {v}")
    print("  notes:")
    print("    " + r["notes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
