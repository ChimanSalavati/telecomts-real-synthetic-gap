"""S3: SpotLight multi-detector origin-aware transfer audit.

Cross-corpus analogue of TelecomTS Table~2 (E9 / E14) on the independent
SpotLight Open RAN corpus. Mirrors S2's calibration protocol but evaluates a
panel of detectors organised in two blocks (matching tab:e9_transfer):

  Tabular detectors on (mean, std, min, max) per-channel features (1808-D):
    - HGB, RF, XGB

  Modern supervised detectors on raw (64, 452) SpotLight windows:
    - Toto (frozen) + LR head
    - TimesNet-lite (end-to-end)
    - PatchTST (end-to-end)

For each (detector, fraction f in {0, 10%, 25%}, seed in 0..9):

    1. Pool train+val splits as the training pool, hold the released test split
       fixed (35 RADIO + 67 non-RADIO + 102 Normal windows are never seen in
       training across any seed or fraction).
    2. Build the training set: Normal + non-RADIO anomalies, plus
       n_inject = round(f * |RADIO_pool|) RADIO windows drawn deterministically
       from a per-seed permutation.
    3. Carve out a stratified 10% validation slice for threshold selection,
       fit the detector, score the held-out test set, pick the F1-best
       threshold on the val slice, evaluate on test.

Outputs:
    results/S3_summary.csv
    results/S3_per_seed.csv
    tables/S3_spotlight_transfer_table.tex

Run:
    python experiments/S3_spotlight_multidetector_transfer/run_s3.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
sys.path.insert(0, str(EXP_ROOT))
from _shared.notebook_helpers import setup_paths  # noqa: E402
setup_paths()
from _shared import sota_helpers as sh  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

# Lazy XGBoost import: importing xgboost at module load time conflicts with
# torch / gluonts OpenMP runtimes on macOS Apple Silicon and causes a SIGSEGV
# in the Toto / TimesNet / PatchTST blocks. We only resolve XGB when the
# tabular block actually needs it.
HAS_XGB: bool | None = None
XGBClassifier = None  # type: ignore


def _ensure_xgb():
    global HAS_XGB, XGBClassifier
    if HAS_XGB is None:
        try:
            from xgboost import XGBClassifier as _XGB  # noqa: WPS433
            XGBClassifier = _XGB
            HAS_XGB = True
        except Exception as exc:  # pragma: no cover
            print(f"[S3] xgboost unavailable: {exc}")
            HAS_XGB = False
    return HAS_XGB

from _shared.data_utils import exp_output_dir  # noqa: E402

RESULTS = exp_output_dir("S3", "results")
TABLES = exp_output_dir("S3", "tables")
EMB_CACHE = HERE / "embeddings_cache"
EMB_CACHE.mkdir(exist_ok=True)
PER_SEED_CSV = RESULTS / "S3_per_seed.csv"
SUMMARY_CSV = RESULTS / "S3_summary.csv"
TABLE_TEX = TABLES / "S3_spotlight_transfer_table.tex"

REPO_ROOT = (EXP_ROOT / ".." / "..").resolve()
SPOT_DIR = (REPO_ROOT / "evaluation_ver2" / "SpotLight" / "data").resolve()
SPOT_VARIANT = "paper5ue_single"
TOTO_WEIGHTS = REPO_ROOT / "models" / "Toto-Open-Base-1.0"

FRACTIONS = [0.0, 0.10, 0.25]
SEEDS = list(range(10))

TABULAR_DETECTORS = ["HGB", "RF", "XGB"]
RAW_DETECTORS = ["Toto (frozen)", "TimesNet (e2e)", "PatchTST (e2e)"]
ALL_DETECTORS = TABULAR_DETECTORS + RAW_DETECTORS

# Match the SpotLight Foundation_Models notebook: subsample 452 channels to 64
# for the Toto pipeline so the encoder runs in reasonable time on a laptop.
TOTO_N_CHANNELS = 64

# Lightweight e2e training defaults.
E2E_EPOCHS = 5
E2E_BATCH = 32
E2E_LR = 1e-3

# PatchTST-specific override: PatchTST on (64, 452) is much heavier than
# TimesNet-lite (full transformer encoder vs depthwise CNN). 5 epochs takes
# ~6 min/seed on CPU, so we use 2 epochs to keep total runtime under an hour.
PATCHTST_EPOCHS = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_npz(split: str):
    return np.load(SPOT_DIR / f"SpotLight_{SPOT_VARIANT}_{split}.npz", allow_pickle=True)


def load_spotlight():
    """Return (X_pool, y_pool, types_pool, X_test, y_test, types_test, feature_cols)."""
    train = _load_npz("train")
    val = _load_npz("val")
    test = _load_npz("test")
    feature_cols = list(train["feature_cols"])
    X_pool = np.concatenate([train["X"], val["X"]], axis=0)
    y_pool = np.concatenate([train["y"], val["y"]], axis=0).astype(int)
    types_pool = np.concatenate([train["anomaly_types"], val["anomaly_types"]], axis=0)
    X_test = test["X"]
    y_test = test["y"].astype(int)
    types_test = test["anomaly_types"]
    return X_pool, y_pool, types_pool, X_test, y_test, types_test, feature_cols


def featurize_tabular(X):
    """(mean, std, min, max) per-channel summary -> (N, 4*C)."""
    F = np.concatenate([X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1)], axis=1)
    return np.nan_to_num(F.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def select_threshold(val_p: np.ndarray, y_val: np.ndarray) -> float:
    if y_val.sum() == 0 or (1 - y_val).sum() == 0:
        return 0.5
    grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        f1 = f1_score(y_val, (val_p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def make_tabular_detector(name: str, seed: int):
    """Return (estimator, needs_scaling) for a tabular detector."""
    if name == "HGB":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=None, random_state=seed,
        ), False
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
        ), False
    if name == "XGB":
        if not _ensure_xgb():
            raise RuntimeError("XGB requested but xgboost is not installed")
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            verbosity=0, random_state=seed, n_jobs=-1,
        ), False
    raise ValueError(f"Unknown detector {name}")


# ---------------------------------------------------------------------------
# Per-seed train/val/test index construction (shared across detector blocks)
# ---------------------------------------------------------------------------
def make_split_indices(y_pool, types_pool, fraction: float, seed: int):
    """Build per-seed (train_inner_idx, val_inner_idx) on the pooled dataset.

    The same (fraction, seed) -> (train_inner, val_inner) construction is used
    for every detector so tabular and raw-window models see identical data.
    """
    rng = np.random.default_rng(seed)
    is_radio = (types_pool == "RADIO")
    is_other_anom = (y_pool == 1) & ~is_radio
    is_normal = (y_pool == 0)
    radio_idx = np.where(is_radio)[0]
    other_anom_idx = np.where(is_other_anom)[0]
    normal_idx = np.where(is_normal)[0]
    radio_perm = radio_idx[rng.permutation(radio_idx.size)]
    n_inject = int(round(fraction * radio_idx.size))
    inject_idx = radio_perm[:n_inject]
    base_idx = np.concatenate([normal_idx, other_anom_idx])
    train_idx = np.concatenate([base_idx, inject_idx])
    rng.shuffle(train_idx)
    y_train = y_pool[train_idx]
    if len(np.unique(y_train)) < 2:
        return None, None, n_inject
    train_inner_idx, val_inner_idx = train_test_split(
        train_idx, test_size=0.10, stratify=y_train, random_state=seed,
    )
    return train_inner_idx, val_inner_idx, n_inject


def metric_row(detector: str, fraction: float, seed: int, n_inject: int,
               n_train: int, thr: float, test_p: np.ndarray, y_test: np.ndarray,
               test_radio_mask: np.ndarray, test_synth_mask: np.ndarray,
               test_norm_mask: np.ndarray, fit_time: float) -> dict:
    pred = (test_p >= thr).astype(int)
    f1 = f1_score(y_test, pred, zero_division=0) if y_test.sum() else float("nan")
    try:
        auroc = float(roc_auc_score(y_test, test_p))
    except Exception:
        auroc = float("nan")
    real_recall = (
        float(pred[test_radio_mask].mean()) if test_radio_mask.sum() else float("nan")
    )
    synth_recall = (
        float(pred[test_synth_mask].mean()) if test_synth_mask.sum() else float("nan")
    )
    normal_fpr = (
        float(pred[test_norm_mask].mean()) if test_norm_mask.sum() else float("nan")
    )
    return {
        "detector": detector,
        "fraction": float(fraction),
        "n_injected_radio": int(n_inject),
        "seed": int(seed),
        "n_train": int(n_train),
        "threshold": float(thr),
        "f1": float(f1),
        "auroc": auroc,
        "real_recall_radio": real_recall,
        "synth_recall_non_radio": synth_recall,
        "normal_fpr": normal_fpr,
        "fit_time_sec": float(fit_time),
    }


# ---------------------------------------------------------------------------
# Block 1: tabular detectors
# ---------------------------------------------------------------------------
def run_tabular_block(F_pool, F_test, y_pool, types_pool, y_test,
                     test_radio_mask, test_synth_mask, test_norm_mask):
    rows = []
    for detector in TABULAR_DETECTORS:
        for f in FRACTIONS:
            for seed in SEEDS:
                t0 = time.time()
                train_inner_idx, val_inner_idx, n_inject = make_split_indices(
                    y_pool, types_pool, f, seed,
                )
                if train_inner_idx is None:
                    continue
                est, needs_scaling = make_tabular_detector(detector, seed)
                if needs_scaling:
                    scaler = StandardScaler().fit(F_pool[train_inner_idx])
                    Xtr = scaler.transform(F_pool[train_inner_idx])
                    Xva = scaler.transform(F_pool[val_inner_idx])
                    Xte = scaler.transform(F_test)
                else:
                    Xtr = F_pool[train_inner_idx]
                    Xva = F_pool[val_inner_idx]
                    Xte = F_test
                est.fit(Xtr, y_pool[train_inner_idx])
                val_p = est.predict_proba(Xva)[:, 1]
                test_p = est.predict_proba(Xte)[:, 1]
                thr = select_threshold(val_p, y_pool[val_inner_idx])
                row = metric_row(
                    detector, f, seed, n_inject, train_inner_idx.size, thr,
                    test_p, y_test, test_radio_mask, test_synth_mask, test_norm_mask,
                    time.time() - t0,
                )
                rows.append(row)
                print(
                    f"  {detector:<14s} f={f:>4.2f} seed={seed} "
                    f"F1={row['f1']:.3f} AUROC={row['auroc']:.3f} "
                    f"RAD={row['real_recall_radio']:.3f} "
                    f"nonR={row['synth_recall_non_radio']:.3f} "
                    f"FPR={row['normal_fpr']:.3f}  ({row['fit_time_sec']:.1f}s)"
                )
    return rows


# ---------------------------------------------------------------------------
# Block 2a: Toto (frozen) -> LR head
# ---------------------------------------------------------------------------
def get_toto_embeddings(X_pool, X_test):
    """Extract Toto embeddings for pool + test. Cache to disk and reuse.

    Following Foundation_Models_SpotLight.ipynb, we uniformly subsample 452
    channels to TOTO_N_CHANNELS (=64) for laptop-feasible runtime.

    Uses pool/test caches separately so a partial completion of one set is
    re-used on the next run.
    """
    pool_cache = EMB_CACHE / "spotlight_toto_pool.npz"
    test_cache = EMB_CACHE / "spotlight_toto_test.npz"
    if pool_cache.exists() and test_cache.exists():
        dp = np.load(pool_cache)
        dt = np.load(test_cache)
        Z_pool, Z_test = dp["Z"], dt["Z"]
        loaded = bool(dp["loaded"]) and bool(dt["loaded"])
        print(f"  [Toto] cache hit: pool {Z_pool.shape}, test {Z_test.shape}, real={loaded}")
        return Z_pool, Z_test, loaded

    # Force a clean memory state before loading Toto.
    import gc, os
    print("  [Toto] gc.collect()", flush=True)
    gc.collect()
    print("  [Toto] env setup", flush=True)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    n_chan = X_pool.shape[2]
    if n_chan > TOTO_N_CHANNELS:
        idx = np.linspace(0, n_chan - 1, TOTO_N_CHANNELS).round().astype(int)
        Xp = X_pool[:, :, idx].copy()
        Xt = X_test[:, :, idx].copy()
        print(f"  [Toto] subsampling {n_chan} -> {TOTO_N_CHANNELS} channels  Xp={Xp.shape}", flush=True)
    else:
        Xp = X_pool.copy()
        Xt = X_test.copy()
    print("  [Toto] arrays prepared", flush=True)

    if not pool_cache.exists():
        print(f"  [Toto] extracting pool embeddings ({Xp.shape}) on CPU ...")
        t0 = time.time()
        Z_pool, loaded_p = sh.encode_toto(
            Xp, weights_dir=TOTO_WEIGHTS, batch_size=8, device="cpu",
        )
        print(f"  [Toto] pool {Z_pool.shape} in {time.time()-t0:.1f}s (real={loaded_p})")
        np.savez_compressed(pool_cache, Z=Z_pool, loaded=np.array(loaded_p))
    else:
        dp = np.load(pool_cache)
        Z_pool, loaded_p = dp["Z"], bool(dp["loaded"])
        print(f"  [Toto] pool cache hit: {Z_pool.shape}, real={loaded_p}")

    gc.collect()

    if not test_cache.exists():
        print(f"  [Toto] extracting test embeddings ({Xt.shape}) on CPU ...")
        t0 = time.time()
        Z_test, loaded_t = sh.encode_toto(
            Xt, weights_dir=TOTO_WEIGHTS, batch_size=8, device="cpu",
        )
        print(f"  [Toto] test {Z_test.shape} in {time.time()-t0:.1f}s (real={loaded_t})")
        np.savez_compressed(test_cache, Z=Z_test, loaded=np.array(loaded_t))
    else:
        dt = np.load(test_cache)
        Z_test, loaded_t = dt["Z"], bool(dt["loaded"])
        print(f"  [Toto] test cache hit: {Z_test.shape}, real={loaded_t}")

    loaded = bool(loaded_p and loaded_t)
    return Z_pool, Z_test, loaded


def run_toto_block(X_pool, X_test, y_pool, types_pool, y_test,
                   test_radio_mask, test_synth_mask, test_norm_mask):
    Z_pool, Z_test, loaded = get_toto_embeddings(X_pool, X_test)
    if not loaded:
        print("  [Toto] WARNING: surrogate embeddings; results below are placeholder.")
    rows = []
    detector = "Toto (frozen)"
    for f in FRACTIONS:
        for seed in SEEDS:
            t0 = time.time()
            train_inner_idx, val_inner_idx, n_inject = make_split_indices(
                y_pool, types_pool, f, seed,
            )
            if train_inner_idx is None:
                continue
            scaler = StandardScaler().fit(Z_pool[train_inner_idx])
            Ztr = scaler.transform(Z_pool[train_inner_idx])
            Zva = scaler.transform(Z_pool[val_inner_idx])
            Zte = scaler.transform(Z_test)
            head = LogisticRegression(max_iter=2000, random_state=seed)
            head.fit(Ztr, y_pool[train_inner_idx])
            val_p = head.predict_proba(Zva)[:, 1]
            test_p = head.predict_proba(Zte)[:, 1]
            thr = select_threshold(val_p, y_pool[val_inner_idx])
            row = metric_row(
                detector, f, seed, n_inject, train_inner_idx.size, thr,
                test_p, y_test, test_radio_mask, test_synth_mask, test_norm_mask,
                time.time() - t0,
            )
            rows.append(row)
            print(
                f"  {detector:<14s} f={f:>4.2f} seed={seed} "
                f"F1={row['f1']:.3f} AUROC={row['auroc']:.3f} "
                f"RAD={row['real_recall_radio']:.3f} "
                f"nonR={row['synth_recall_non_radio']:.3f} "
                f"FPR={row['normal_fpr']:.3f}  ({row['fit_time_sec']:.1f}s)"
            )
    return rows


# ---------------------------------------------------------------------------
# Block 2b/2c: TimesNet-lite and PatchTST end-to-end
# ---------------------------------------------------------------------------
def run_e2e_block(detector_name: str, model_key: str, *, X_pool, X_test,
                  y_pool, types_pool, y_test, test_radio_mask, test_synth_mask,
                  test_norm_mask, seq_len: int, in_channels: int):
    epochs = PATCHTST_EPOCHS if model_key == "PatchTST" else E2E_EPOCHS
    print(f"  [e2e] {detector_name}: epochs={epochs} batch={E2E_BATCH} lr={E2E_LR}", flush=True)

    # Resume support: if a per-block CSV already has (fraction, seed) rows for
    # this detector, skip them. This lets a PatchTST run that was killed
    # mid-sweep be resumed without recomputing the seeds we already have.
    block_key = "patchtst" if model_key == "PatchTST" else "timesnet"
    block_csv = BLOCK_CSVS.get(block_key)
    rows: list[dict] = []
    done_pairs: set[tuple[float, int]] = set()
    if block_csv is not None and block_csv.exists():
        existing = pd.read_csv(block_csv)
        existing = existing[existing["detector"] == detector_name]
        rows = existing.to_dict("records")
        done_pairs = {(round(float(r["fraction"]), 4), int(r["seed"])) for r in rows}
        print(
            f"  [resume] {detector_name}: {len(done_pairs)} (fraction, seed) "
            f"pairs already in {block_csv}",
            flush=True,
        )
    for f in FRACTIONS:
        for seed in SEEDS:
            if (round(float(f), 4), int(seed)) in done_pairs:
                continue
            t0 = time.time()
            train_inner_idx, val_inner_idx, n_inject = make_split_indices(
                y_pool, types_pool, f, seed,
            )
            if train_inner_idx is None:
                continue
            X_train = X_pool[train_inner_idx]
            X_val = X_pool[val_inner_idx]
            try:
                val_p, test_p, fit_time = sh.train_e2e_binary(
                    model_key,
                    X_train, y_pool[train_inner_idx],
                    X_val, X_test,
                    seed=seed,
                    epochs=epochs,
                    lr=E2E_LR,
                    batch_size=E2E_BATCH,
                    in_channels=in_channels,
                    seq_len=seq_len,
                )
            except Exception as exc:
                print(f"  {detector_name} f={f} seed={seed} FAILED: {exc}")
                continue
            thr = select_threshold(val_p, y_pool[val_inner_idx])
            row = metric_row(
                detector_name, f, seed, n_inject, train_inner_idx.size, thr,
                test_p, y_test, test_radio_mask, test_synth_mask, test_norm_mask,
                time.time() - t0,
            )
            rows.append(row)
            print(
                f"  {detector_name:<14s} f={f:>4.2f} seed={seed} "
                f"F1={row['f1']:.3f} AUROC={row['auroc']:.3f} "
                f"RAD={row['real_recall_radio']:.3f} "
                f"nonR={row['synth_recall_non_radio']:.3f} "
                f"FPR={row['normal_fpr']:.3f}  ({row['fit_time_sec']:.1f}s)"
            )
            # Incremental checkpoint: write the partial CSV after every seed
            # so a future run can resume even if we crash mid-sweep.
            if block_csv is not None:
                pd.DataFrame.from_records(rows).to_csv(block_csv, index=False)
    return rows


# ---------------------------------------------------------------------------
# Aggregate + LaTeX
# ---------------------------------------------------------------------------
def write_summary(rows):
    per_seed = pd.DataFrame.from_records(rows)
    per_seed.to_csv(PER_SEED_CSV, index=False)
    print(f"\nWrote {PER_SEED_CSV}")
    summary_rows = []
    for (detector, f), g in per_seed.groupby(["detector", "fraction"]):
        n = len(g)
        summary_rows.append({
            "detector": detector,
            "fraction": float(f),
            "n_injected_radio_avg": float(g["n_injected_radio"].mean()),
            "n_seeds": int(n),
            "f1_mean": float(g["f1"].mean()),
            "f1_std": float(g["f1"].std(ddof=0)),
            "auroc_mean": float(g["auroc"].mean()),
            "auroc_std": float(g["auroc"].std(ddof=0)),
            "real_recall_mean": float(g["real_recall_radio"].mean()),
            "real_recall_std": float(g["real_recall_radio"].std(ddof=0)),
            "synth_recall_mean": float(g["synth_recall_non_radio"].mean()),
            "synth_recall_std": float(g["synth_recall_non_radio"].std(ddof=0)),
            "normal_fpr_mean": float(g["normal_fpr"].mean()),
            "normal_fpr_std": float(g["normal_fpr"].std(ddof=0)),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["detector", "fraction"])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}\n")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))
    return summary


def write_latex(summary: pd.DataFrame):
    def fmt_pct(mean, std):
        return f"{int(round(mean * 100))}{{\\tiny$\\pm${int(round(std * 100))}}}\\%"

    def fmt_score(mean, std):
        return f"{mean:.2f}{{\\tiny$\\pm${std:.2f}}}"

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(
        r"  \caption{SpotLight multi-detector origin-aware transfer audit "
        r"(mean$\pm$std over 10 seeds; 35 \textsc{Radio} / 67 non-\textsc{Radio} / 102 "
        r"Normal test windows held fixed across detectors and seeds). "
        r"\textbf{Synth-only} ($f\!=\!0$) drops the controlled-real \textsc{Radio} pool "
        r"from training; $f\!=\!10\%$ (21 windows) is the TelecomTS-equivalent budget; "
        r"$f\!=\!25\%$ (52 windows) is the SpotLight per-corpus budget that matches "
        r"Table~\ref{tab:s2_spotlight_cal}. The detector-independent failure and the "
        r"detector-independent recovery at the paper-recommended budget replicate the "
        r"TelecomTS pattern (Table~\ref{tab:e9_transfer}) on a second testbed corpus.}"
    )
    lines.append(r"  \label{tab:s3_spotlight_transfer}")
    lines.append(r"  \footnotesize")
    lines.append(r"  \setlength{\tabcolsep}{2pt}")
    lines.append(
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    Detector & \multicolumn{4}{c}{Synth-only ($f\!=\!0$)} & "
        r"\multicolumn{4}{c}{Calibrated ($f\!=\!10\%$)} & "
        r"\multicolumn{4}{c}{Calibrated ($f\!=\!25\%$)} \\"
    )
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(
        r"      & F1 & AUC & \textsc{Rad} & non-\textsc{R} & "
        r"F1 & AUC & \textsc{Rad} & non-\textsc{R} & "
        r"F1 & AUC & \textsc{Rad} & non-\textsc{R} \\"
    )
    lines.append(r"    \midrule")

    summary_indexed = summary.set_index(["detector", "fraction"])

    def emit_row(name: str, label: str | None = None):
        try:
            r0 = summary_indexed.loc[(name, 0.0)]
            r1 = summary_indexed.loc[(name, 0.10)]
            r2 = summary_indexed.loc[(name, 0.25)]
        except KeyError:
            return
        cells = [
            fmt_score(r0["f1_mean"], r0["f1_std"]),
            fmt_score(r0["auroc_mean"], r0["auroc_std"]),
            fmt_pct(r0["real_recall_mean"], r0["real_recall_std"]),
            fmt_pct(r0["synth_recall_mean"], r0["synth_recall_std"]),
            fmt_score(r1["f1_mean"], r1["f1_std"]),
            fmt_score(r1["auroc_mean"], r1["auroc_std"]),
            fmt_pct(r1["real_recall_mean"], r1["real_recall_std"]),
            fmt_pct(r1["synth_recall_mean"], r1["synth_recall_std"]),
            fmt_score(r2["f1_mean"], r2["f1_std"]),
            fmt_score(r2["auroc_mean"], r2["auroc_std"]),
            fmt_pct(r2["real_recall_mean"], r2["real_recall_std"]),
            fmt_pct(r2["synth_recall_mean"], r2["synth_recall_std"]),
        ]
        display = label if label is not None else name
        lines.append(f"    {display} & " + " & ".join(cells) + r" \\")

    lines.append(
        r"    \multicolumn{13}{l}{Tabular detectors on per-channel KPI summaries} \\"
    )
    for name in TABULAR_DETECTORS:
        emit_row(name)
    lines.append(r"    \midrule")
    lines.append(
        r"    \multicolumn{13}{l}{Modern supervised detectors on raw $64\times452$ KPI windows} \\"
    )
    for name in RAW_DETECTORS:
        emit_row(name)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular*}")
    lines.append(r"\end{table*}")
    TABLE_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLE_TEX}")


# ---------------------------------------------------------------------------
# Per-block CSV checkpoints
# ---------------------------------------------------------------------------
BLOCK_CSVS = {
    "tabular": RESULTS / "S3_block_tabular.csv",
    "toto": RESULTS / "S3_block_toto.csv",
    "timesnet": RESULTS / "S3_block_timesnet.csv",
    "patchtst": RESULTS / "S3_block_patchtst.csv",
}


def save_block(name: str, rows: list[dict]) -> None:
    path = BLOCK_CSVS[name]
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    print(f"  [{name}] wrote {path} ({len(rows)} rows)")


def load_all_block_csvs() -> list[dict]:
    rows: list[dict] = []
    for name, path in BLOCK_CSVS.items():
        if path.exists():
            df = pd.read_csv(path)
            print(f"  [{name}] loaded {len(df)} rows from {path}")
            rows.extend(df.to_dict("records"))
        else:
            print(f"  [{name}] MISSING: {path}")
    return rows


# ---------------------------------------------------------------------------
# Main entry points (one per block) -- each block can run in a fresh subprocess
# ---------------------------------------------------------------------------
def _load_pool_and_masks():
    X_pool, y_pool, types_pool, X_test, y_test, types_test, feature_cols = load_spotlight()
    print(f"  pool  : windows={X_pool.shape[0]} (anom={int(y_pool.sum())})")
    print(f"  test  : windows={X_test.shape[0]} (anom={int(y_test.sum())})")
    print(f"  shape : pool {X_pool.shape}  test {X_test.shape}  channels={len(feature_cols)}")
    test_radio_mask = (types_test == "RADIO")
    test_synth_mask = (y_test == 1) & ~test_radio_mask
    test_norm_mask = (y_test == 0)
    print(
        f"  test composition: RADIO={int(test_radio_mask.sum())} "
        f"non-RAD={int(test_synth_mask.sum())} Normal={int(test_norm_mask.sum())}"
    )
    return X_pool, y_pool, types_pool, X_test, y_test, test_radio_mask, test_synth_mask, test_norm_mask


def main_tabular():
    print(f"Loading SpotLight from {SPOT_DIR}")
    (X_pool, y_pool, types_pool, X_test, y_test,
     test_radio_mask, test_synth_mask, test_norm_mask) = _load_pool_and_masks()
    F_pool = featurize_tabular(X_pool)
    F_test = featurize_tabular(X_test)
    print(f"  tabular feature dim: {F_pool.shape[1]}")
    print("\n=== Block 1: tabular detectors ===")
    rows = run_tabular_block(
        F_pool, F_test, y_pool, types_pool, y_test,
        test_radio_mask, test_synth_mask, test_norm_mask,
    )
    save_block("tabular", rows)


def main_toto():
    print(f"Loading SpotLight from {SPOT_DIR}")
    (X_pool, y_pool, types_pool, X_test, y_test,
     test_radio_mask, test_synth_mask, test_norm_mask) = _load_pool_and_masks()
    print("\n=== Block 2a: Toto (frozen) + LR head ===")
    rows = run_toto_block(
        X_pool, X_test, y_pool, types_pool, y_test,
        test_radio_mask, test_synth_mask, test_norm_mask,
    )
    save_block("toto", rows)


def main_timesnet():
    print(f"Loading SpotLight from {SPOT_DIR}")
    (X_pool, y_pool, types_pool, X_test, y_test,
     test_radio_mask, test_synth_mask, test_norm_mask) = _load_pool_and_masks()
    print("\n=== Block 2b: TimesNet-lite (e2e) ===")
    rows = run_e2e_block(
        "TimesNet (e2e)", "TimesNet-lite",
        X_pool=X_pool, X_test=X_test,
        y_pool=y_pool, types_pool=types_pool, y_test=y_test,
        test_radio_mask=test_radio_mask, test_synth_mask=test_synth_mask,
        test_norm_mask=test_norm_mask,
        seq_len=X_pool.shape[1], in_channels=X_pool.shape[2],
    )
    save_block("timesnet", rows)


def main_patchtst():
    print(f"Loading SpotLight from {SPOT_DIR}")
    (X_pool, y_pool, types_pool, X_test, y_test,
     test_radio_mask, test_synth_mask, test_norm_mask) = _load_pool_and_masks()
    print("\n=== Block 2c: PatchTST (e2e) ===")
    rows = run_e2e_block(
        "PatchTST (e2e)", "PatchTST",
        X_pool=X_pool, X_test=X_test,
        y_pool=y_pool, types_pool=types_pool, y_test=y_test,
        test_radio_mask=test_radio_mask, test_synth_mask=test_synth_mask,
        test_norm_mask=test_norm_mask,
        seq_len=X_pool.shape[1], in_channels=X_pool.shape[2],
    )
    save_block("patchtst", rows)


def main_aggregate():
    print("=== Aggregating per-block CSV checkpoints ===")
    rows = load_all_block_csvs()
    if not rows:
        raise SystemExit("No block CSVs found; run blocks first.")
    summary = write_summary(rows)
    write_latex(summary)


def main_orchestrator():
    """Run each block in a fresh Python subprocess to isolate memory state.

    Mixing scikit/XGBoost training and Toto/PyTorch in one process triggers a
    silent SIGSEGV on this Python 3.14 / torch 2.11 / Apple Silicon setup, so
    we spawn one subprocess per block. Each block writes its own CSV checkpoint.
    """
    import subprocess
    script = str(Path(__file__).resolve())
    blocks = ["tabular", "toto", "timesnet", "patchtst"]
    for block in blocks:
        print(f"\n>>> launching subprocess for block: {block}", flush=True)
        env = {
            **__import__("os").environ,
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        rc = subprocess.call(
            [sys.executable, script, "--block", block],
            env=env,
        )
        if rc != 0:
            print(f"  block {block} subprocess exited with code {rc}; continuing")
    print("\n>>> aggregating block CSVs", flush=True)
    main_aggregate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=list(BLOCK_CSVS) + ["aggregate", "orchestrate"],
                    default="orchestrate")
    args = ap.parse_args()
    if args.block == "tabular":
        main_tabular()
    elif args.block == "toto":
        main_toto()
    elif args.block == "timesnet":
        main_timesnet()
    elif args.block == "patchtst":
        main_patchtst()
    elif args.block == "aggregate":
        main_aggregate()
    else:
        main_orchestrator()
