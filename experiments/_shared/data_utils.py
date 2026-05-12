"""Shared TelecomTS data utilities for the CIKM2026 paper experiments (E1-E8).

This module centralises:
  - Loading the full TelecomTS corpus from HuggingFace (with on-disk pickle cache).
  - Canonical KPI ordering, anomaly-type taxonomy, and real/synthetic mapping.
  - Window-level metadata extraction (context labels, anomaly origin, anomaly type).
  - Engineered KPI-summary feature extraction used by E2-E8.
  - Reproducible split builders for the five splits referenced by the paper.

All paths are resolved relative to this file so notebooks can be run from any cwd.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Canonical KPI order (matches data/TelecomTS_train.npz feature_cols and
# split/TelecomTS_*.npz feature_cols).
KPI_NAMES: tuple[str, ...] = (
    "RSRP",
    "DL_BLER",
    "DL_MCS",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_SNR",
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "UL_NumberOfPackets",
    "DL_NumberOfPackets",
)

# KPI groupings used by the per-group ablation in E7.
KPI_GROUPS: dict[str, tuple[str, ...]] = {
    "PHY": ("RSRP", "UL_SNR"),
    "Error": ("DL_BLER", "UL_BLER"),
    "MCS": ("DL_MCS", "UL_MCS"),
    "Scheduler": (
        "PRB_Utilization_DL",
        "PRB_Utilization_UL",
        "PRBs_DL_Current",
        "PRBs_UL_Current",
        "UL_NPRB",
        "Estimated_UL_Buffer",
    ),
    "Traffic": (
        "TX_Bytes",
        "RX_Bytes",
        "DL_NumberOfPackets",
        "UL_NumberOfPackets",
    ),
}

# Anomaly types as they appear in the TelecomTS HuggingFace release.
ANOMALY_TYPES: tuple[str, ...] = (
    "Jamming",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
    "Co-Channel Interference (Severe)",
    "Co-Channel Interference (Mild)",
    "Faulty RF Filters (Temporal)",
    "Doppler Shift (Severe)",
    "Resource Allocation Bugs",
    "Antenna Failure",
    "Faulty Handover Algorithm (Too Frequent)",
    "Buffer Overflow (Gradual Buildup)",
)
N_ANOMALY_TYPES = 11

REAL_ANOMALY_TYPES: tuple[str, ...] = ("Jamming",)
SYNTHETIC_ANOMALY_TYPES: tuple[str, ...] = tuple(
    t for t in ANOMALY_TYPES if t not in REAL_ANOMALY_TYPES
)

CONTEXT_FIELDS: tuple[str, ...] = ("zone", "application", "mobility", "congestion")

# Zones encoded in the released TelecomTS labels.
#
# IMPORTANT — collection-protocol findings (verified against TelecomTS Appendix B
# and Table 7):
#   * Fixed zones are A (0-3 m), B (3-6 m), C (>6 m).
#   * Mobile sessions cannot be assigned to a fixed zone; the released labels
#     encode them as ``zone == "In motion"``. This is *equivalent* to
#     ``mobility == "Yes"`` (perfect 1:1 correspondence in the corpus).
#   * Real Jamming was collected ONLY in Zone A (jammer near the RU). All 279
#     real-Jamming windows have ``zone == "A"``. Synthetic anomalies span
#     Zones A/B/C.
#   * "In motion" sessions contain only normal observations — no anomalies of
#     any kind were collected while the device was moving.
#
# The helpers ``zone_a_indices``, ``mobile_indices``, and ``restrict_to_zone_a``
# below let downstream notebooks slice cleanly along these axes.
ZONES_FIXED: tuple[str, ...] = ("A", "B", "C")
ZONE_IN_MOTION: str = "In motion"

# Tokens to scrub for E5 masked text variants.
ANOMALY_NAME_TOKENS: tuple[str, ...] = (
    "Jamming", "Antenna Failure", "Buffer Overflow",
    "Co-Channel Interference", "Co-channel interference",
    "Doppler Shift", "Doppler shift",
    "Faulty Handover", "faulty handover",
    "Faulty RF Filters", "faulty RF filters",
    "High Network Congestion", "Resource Allocation Bug",
)
KPI_NAME_TOKENS: tuple[str, ...] = (
    "RSRP", "SNR", "BLER", "MCS", "PRB", "buffer",
    "bytes", "packets", "throughput", "congestion", "handover",
)
CONTEXT_NAME_TOKENS: tuple[str, ...] = (
    "Zone A", "Zone B", "Zone C", "zone A", "zone B", "zone C",
    "YouTube", "Twitch", "File", "file transfer", "file download",
)
HEAVY_CUE_TOKENS: tuple[str, ...] = (
    "anomaly", "anomalous", "abnormal", "fault", "interference",
    "degradation", "drop", "spike", "outage", "failure",
)

# -----------------------------------------------------------------------------
# Paths and cache locations
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = _THIS_DIR.parent
REPO_ROOT = EXPERIMENTS_ROOT.parent
REPO_ROOT = REPO_ROOT.parent

CACHE_DIR = EXPERIMENTS_ROOT / "_shared" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_CACHE = CACHE_DIR / "telecomts_corpus_v1.pkl"
FEATURE_CACHE = CACHE_DIR / "telecomts_kpi_summary_features_v1.npz"


# -----------------------------------------------------------------------------
# Centralized output directory + offline (synthetic) mode
# -----------------------------------------------------------------------------
#
# Both knobs are environment-driven so they work uniformly whether an
# experiment is launched through ``python main.py ...`` (the unified runner) or
# invoked as a standalone script. Neither changes behaviour for the real,
# full-scale 32k-window corpus unless the corresponding variable is set.

# Where every experiment writes its CSV / PDF / JSON outputs. The runner sets
# this to ``<repo>/artifacts``; when unset we fall back to the same default so
# standalone script runs still land in the centralized tree.
_OUTPUT_DIR_ENV = "TELECOMTS_GAP_OUTPUT_DIR"
_DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts"

# When set to a truthy value, ``load_corpus`` returns a small, deterministic
# *synthetic* corpus instead of downloading TelecomTS from HuggingFace. This is
# what makes the smoke tests run offline in seconds with no GPU and no network.
_SYNTHETIC_ENV = "TELECOMTS_GAP_SYNTHETIC"
_SYNTHETIC_N_ENV = "TELECOMTS_GAP_SYNTHETIC_N"


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"} if value else False


def synthetic_mode() -> bool:
    """True when offline synthetic-corpus mode is requested via the environment."""
    return _truthy(os.environ.get(_SYNTHETIC_ENV))


# Per-experiment seed count. The paper uses 10; the smoke preset shrinks this
# via TELECOMTS_GAP_SEEDS so the offline CI run finishes in seconds.
_SEEDS_ENV = "TELECOMTS_GAP_SEEDS"


def default_seeds(default: int = 10) -> int:
    """Number of random seeds an experiment should sweep (env-overridable)."""
    raw = os.environ.get(_SEEDS_ENV)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def output_root() -> Path:
    """Root directory for all experiment outputs (honours ``TELECOMTS_GAP_OUTPUT_DIR``)."""
    raw = os.environ.get(_OUTPUT_DIR_ENV)
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_OUTPUT_ROOT


def exp_output_dir(exp_id: str, kind: str = "results") -> Path:
    """Return (and create) the centralized output directory for an experiment.

    ``exp_output_dir("E16", "results")`` -> ``<output_root>/E16/results``.
    Passing ``kind=""`` returns ``<output_root>/E16`` itself. Every experiment
    in the repo writes through this helper so a single ``--output-dir`` flag
    redirects the whole pipeline.
    """
    base = output_root() / exp_id
    out = base / kind if kind else base
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_synthetic_corpus(n: int | None = None, seed: int = 0) -> "TelecomCorpus":
    """Build a tiny, structured stand-in for the TelecomTS corpus.

    The synthetic corpus reproduces the *qualitative* structure the audit keys
    on -- controlled-real (Jamming) windows sit at high RSRP (~-76 dBm) while
    Normal and synthetic-anomaly windows sit at low RSRP (~-109 dBm) -- so the
    origin-aware transfer gap and its calibration repair are exercised end to
    end without any download. It is **only** used for offline smoke tests and
    is never a substitute for the real corpus in the paper numbers.
    """
    if n is None:
        n = int(os.environ.get(_SYNTHETIC_N_ENV, "1500"))
    n = max(400, int(n))
    rng = np.random.default_rng(seed)

    # Composition: ~70% normal, ~10% controlled-real Jamming, ~20% synthetic.
    n_real = max(60, int(round(0.10 * n)))
    n_synth = max(120, int(round(0.20 * n)))
    n_norm = n - n_real - n_synth
    origins = np.array(["normal"] * n_norm + ["real"] * n_real + ["synthetic"] * n_synth, dtype=object)
    perm = rng.permutation(n)
    origins = origins[perm]

    y = (origins != "normal").astype(np.int64)
    anomaly_type = np.where(
        origins == "real",
        "Jamming",
        np.where(origins == "synthetic", rng.choice(list(SYNTHETIC_ANOMALY_TYPES), size=n), ""),
    ).astype(object)

    # Per-window RSRP regime by origin (the dominant separating axis on TelecomTS).
    rsrp_centre = np.where(origins == "real", -76.0, -109.0).astype(np.float64)
    X = rng.normal(0.0, 1.0, size=(n, 128, len(KPI_NAMES))).astype(np.float32)
    # Channel 0 == RSRP: centre it on the per-origin regime and add an anomaly bump.
    X[:, :, 0] = (rsrp_centre[:, None] + rng.normal(0.0, 3.0, size=(n, 128))).astype(np.float32)
    anom_bump = (y == 1)[:, None].astype(np.float32)
    X[:, :, 3] += 2.0 * anom_bump  # UL_BLER rises during anomalies
    X[:, :, 6] -= 1.5 * anom_bump  # UL_SNR degrades during anomalies

    # Context metadata that mirrors the collection-protocol findings:
    #   real Jamming lives only in Zone A; synthetic spans A/B/C; some normals
    #   are "In motion" (mobile) and carry no anomalies.
    zone = np.empty(n, dtype=object)
    mobility = np.empty(n, dtype=object)
    application = rng.choice(["YouTube", "Twitch", "FileTransfer"], size=n).astype(object)
    congestion = rng.choice(["Low", "Medium", "High"], size=n).astype(object)
    for i in range(n):
        if origins[i] == "real":
            zone[i], mobility[i] = "A", "No"
        elif origins[i] == "synthetic":
            zone[i], mobility[i] = rng.choice(["A", "B", "C"]), "No"
        else:
            if rng.random() < 0.15:
                zone[i], mobility[i] = ZONE_IN_MOTION, "Yes"
            else:
                zone[i], mobility[i] = rng.choice(["A", "B", "C"]), "No"

    empty = np.array([""] * n, dtype=object)
    return TelecomCorpus(
        sample_id=np.arange(n, dtype=np.int64),
        X=X,
        y=y,
        anomaly_type=anomaly_type,
        anomaly_origin=origins,
        description=empty.copy(),
        troubleshooting=empty.copy(),
        zone=zone,
        application=application,
        mobility=mobility,
        congestion=congestion,
        feature_cols=np.array(KPI_NAMES),
    )


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

@dataclass
class TelecomCorpus:
    """Holds the full TelecomTS corpus in a numpy-friendly layout."""

    sample_id: np.ndarray  # int64, shape (N,)
    X: np.ndarray  # float32, shape (N, 128, 16)
    y: np.ndarray  # int64, shape (N,) binary anomaly label
    anomaly_type: np.ndarray  # object/str, shape (N,), "" for normals
    anomaly_origin: np.ndarray  # object/str, shape (N,) in {"normal","real","synthetic"}
    description: np.ndarray  # object/str, shape (N,)
    troubleshooting: np.ndarray  # object/str, shape (N,)
    zone: np.ndarray  # object/str
    application: np.ndarray  # object/str
    mobility: np.ndarray  # object/str
    congestion: np.ndarray  # object/str
    feature_cols: np.ndarray  # str array of len 16

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def context_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": self.sample_id,
                "y": self.y,
                "anomaly_type": self.anomaly_type,
                "anomaly_origin": self.anomaly_origin,
                "zone": self.zone,
                "application": self.application,
                "mobility": self.mobility,
                "congestion": self.congestion,
            }
        )


def _row_to_X(row: dict, kpi_order: Sequence[str]) -> np.ndarray:
    arr = np.column_stack([row["KPIs"][k] for k in kpi_order]).astype(np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.any(np.isnan(col)):
            arr[:, j] = np.where(np.isnan(col), np.nanmedian(col), col)
    return arr.astype(np.float32)


def load_corpus(force_reload: bool = False, verbose: bool = True) -> TelecomCorpus:
    """Load the full TelecomTS corpus from HuggingFace (cached locally as pickle).

    The on-disk cache lives at ``_shared/cache/telecomts_corpus_v1.pkl`` and is
    safe to reuse across notebooks. Pass ``force_reload=True`` to ignore it.

    When ``TELECOMTS_GAP_SYNTHETIC`` is set in the environment, a small offline
    synthetic corpus is returned instead (used by the smoke tests).
    """
    if synthetic_mode():
        if verbose:
            print("[data_utils] SYNTHETIC mode: returning offline synthetic corpus "
                  "(set TELECOMTS_GAP_SYNTHETIC=0 to use the real TelecomTS corpus)")
        return make_synthetic_corpus()
    if (not force_reload) and CORPUS_CACHE.exists():
        if verbose:
            print(f"[data_utils] Loading cached corpus from {CORPUS_CACHE}")
        with CORPUS_CACHE.open("rb") as f:
            return pickle.load(f)

    from datasets import load_dataset  # imported lazily

    if verbose:
        print("[data_utils] Loading TelecomTS from HuggingFace (this is slow on first run)...")
    ds = load_dataset("AliMaatouk/TelecomTS", split="train")
    if verbose:
        print(f"[data_utils] Got {len(ds)} rows; building arrays...")

    n = len(ds)
    X = np.empty((n, 128, len(KPI_NAMES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    sample_id = np.arange(n, dtype=np.int64)
    anomaly_type = np.empty(n, dtype=object)
    anomaly_origin = np.empty(n, dtype=object)
    description = np.empty(n, dtype=object)
    troubleshooting = np.empty(n, dtype=object)
    zone = np.empty(n, dtype=object)
    application = np.empty(n, dtype=object)
    mobility = np.empty(n, dtype=object)
    congestion = np.empty(n, dtype=object)

    for i, row in enumerate(ds):
        X[i] = _row_to_X(row, KPI_NAMES)
        labels = row.get("labels", {}) or {}
        anomalies = row.get("anomalies", {}) or {}
        present = labels.get("anomaly_present", "No") == "Yes"
        y[i] = 1 if present else 0
        atype = (anomalies.get("type") or "").strip()
        anomaly_type[i] = atype if present else ""
        if not present:
            anomaly_origin[i] = "normal"
        elif atype in REAL_ANOMALY_TYPES:
            anomaly_origin[i] = "real"
        else:
            anomaly_origin[i] = "synthetic"
        description[i] = row.get("description", "") or ""
        troubleshooting[i] = (anomalies.get("troubleshooting_tickets") or "")
        zone[i] = labels.get("zone", "")
        application[i] = labels.get("application", "")
        mobility[i] = labels.get("mobility", "")
        congestion[i] = labels.get("congestion", "")
        if verbose and i and (i % 4000 == 0):
            print(f"[data_utils]   processed {i}/{n}")

    corpus = TelecomCorpus(
        sample_id=sample_id,
        X=X,
        y=y,
        anomaly_type=anomaly_type,
        anomaly_origin=anomaly_origin,
        description=description,
        troubleshooting=troubleshooting,
        zone=zone,
        application=application,
        mobility=mobility,
        congestion=congestion,
        feature_cols=np.array(KPI_NAMES),
    )

    if verbose:
        print(f"[data_utils] Caching corpus to {CORPUS_CACHE}")
    with CORPUS_CACHE.open("wb") as f:
        pickle.dump(corpus, f)
    return corpus


# -----------------------------------------------------------------------------
# Engineered KPI-summary features (used by E2, E3, E4, E5, E6, E7, E8 ablations)
# -----------------------------------------------------------------------------

# Fixed feature names so every notebook agrees.
_PER_KPI_STAT_NAMES = (
    "mean", "std", "min", "max", "median", "iqr",
    "trend",
    "max_abs_diff", "mean_abs_diff", "energy_diff",
    "acf_lag1", "acf_lag5", "acf_lag10",
    "dom_freq", "spec_entropy",
)


def _safe_acf(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= x.size:
        return 0.0
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom <= 0:
        return 0.0
    return float(np.dot(x0[:-lag], x0[lag:]) / denom)


def _per_window_kpi_features(window: np.ndarray) -> np.ndarray:
    """Return a flat feature vector of length len(KPI_NAMES) * 15 = 240."""
    n_kpi = window.shape[1]
    out = np.empty(n_kpi * len(_PER_KPI_STAT_NAMES), dtype=np.float64)
    t = np.arange(window.shape[0], dtype=np.float64)
    for j in range(n_kpi):
        x = window[:, j].astype(np.float64)
        diff = np.diff(x)
        if x.size > 1:
            slope, _ = np.polyfit(t, x, 1) if np.std(x) > 0 else (0.0, 0.0)
        else:
            slope = 0.0
        # Spectral
        f = np.fft.rfft(x - x.mean())
        ps = (f.conj() * f).real
        if ps.sum() > 0:
            dom = int(np.argmax(ps))
            p = ps / ps.sum()
            p = p[p > 0]
            ent = float(-np.sum(p * np.log(p)))
        else:
            dom = 0
            ent = 0.0
        feats = (
            float(x.mean()),
            float(x.std()),
            float(x.min()),
            float(x.max()),
            float(np.median(x)),
            float(np.subtract(*np.percentile(x, [75, 25]))),
            float(slope),
            float(np.max(np.abs(diff))) if diff.size else 0.0,
            float(np.mean(np.abs(diff))) if diff.size else 0.0,
            float(np.sum(diff ** 2)) if diff.size else 0.0,
            _safe_acf(x, 1),
            _safe_acf(x, 5),
            _safe_acf(x, 10),
            float(dom),
            ent,
        )
        out[j * len(_PER_KPI_STAT_NAMES):(j + 1) * len(_PER_KPI_STAT_NAMES)] = feats
    return out


def feature_names(kpi_order: Sequence[str] | None = None) -> list[str]:
    """Names of the 240 engineered features, ordered KPI-major then stat-minor."""
    if kpi_order is None:
        kpi_order = KPI_NAMES
    return [f"{k}__{s}" for k in kpi_order for s in _PER_KPI_STAT_NAMES]


def kpi_for_feature(feature_name: str) -> str:
    """Recover the KPI name from a feature name produced by ``feature_names``."""
    return feature_name.split("__", 1)[0]


def extract_kpi_summary_features(
    X: np.ndarray,
    use_cache: bool = False,
    cache_key: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract engineered features for an array of windows.

    Parameters
    ----------
    X
        ``(N, 128, 16)`` float array.
    use_cache
        Only honoured when ``cache_key`` is provided. Caches features per
        ``cache_key`` under ``_shared/cache``.
    cache_key
        String identifying the source set (e.g. ``"corpus_full"``).
    """
    cache_path = None
    if use_cache and cache_key:
        h = hashlib.md5(cache_key.encode()).hexdigest()[:10]
        cache_path = CACHE_DIR / f"feats_{h}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as d:
                return d["F"], list(d["names"])
    F = np.empty((X.shape[0], len(KPI_NAMES) * len(_PER_KPI_STAT_NAMES)), dtype=np.float64)
    for i in range(X.shape[0]):
        F[i] = _per_window_kpi_features(X[i])
    names = feature_names(KPI_NAMES)
    if cache_path is not None:
        np.savez_compressed(cache_path, F=F, names=np.array(names))
    return F, names


def get_or_build_corpus_features(verbose: bool = True) -> tuple[np.ndarray, list[str]]:
    """Convenience: features for the entire corpus, cached on disk.

    In synthetic (offline) mode the features are computed directly from the
    synthetic corpus and never touch the on-disk cache.
    """
    if synthetic_mode():
        corpus = load_corpus(verbose=verbose)
        if verbose:
            print(f"[data_utils] SYNTHETIC mode: extracting features for {corpus.n} windows ...")
        F, names = extract_kpi_summary_features(corpus.X)
        return F.astype(np.float32), names
    if FEATURE_CACHE.exists():
        if verbose:
            print(f"[data_utils] Loading cached corpus features from {FEATURE_CACHE}")
        with np.load(FEATURE_CACHE, allow_pickle=True) as d:
            return d["F"], list(d["names"])
    corpus = load_corpus(verbose=verbose)
    if verbose:
        print(f"[data_utils] Extracting engineered features for {corpus.n} windows...")
    F, names = extract_kpi_summary_features(corpus.X)
    np.savez_compressed(FEATURE_CACHE, F=F.astype(np.float32), names=np.array(names))
    return F.astype(np.float32), names


# -----------------------------------------------------------------------------
# Splits used by the paper
# -----------------------------------------------------------------------------


def _stratified_pick(
    rng: np.random.Generator,
    pool: np.ndarray,
    n: int,
) -> np.ndarray:
    """Pick ``n`` elements uniformly without replacement."""
    if n > pool.size:
        raise ValueError(f"Need {n} samples but pool only has {pool.size}")
    return rng.choice(pool, size=n, replace=False)


def make_small_natural_split(corpus: TelecomCorpus, seed: int = 42) -> dict[str, np.ndarray]:
    """640/160/200 windows, ~5% anomaly rate (matches existing split/ folder)."""
    rng = np.random.default_rng(seed)
    norm_pool = np.where(corpus.y == 0)[0]
    anom_pool = np.where(corpus.y == 1)[0]
    n_total, n_anom = 1000, 50
    n_norm = n_total - n_anom
    norm_idx = _stratified_pick(rng, norm_pool, n_norm)
    anom_idx = _stratified_pick(rng, anom_pool, n_anom)
    n_norm_train = int(n_norm * 0.8)
    n_anom_train = int(n_anom * 0.8)
    train_norm, test_norm = norm_idx[:n_norm_train], norm_idx[n_norm_train:]
    train_anom, test_anom = anom_idx[:n_anom_train], anom_idx[n_anom_train:]
    pool_train = np.concatenate([train_norm, train_anom])
    rng.shuffle(pool_train)
    n_val = int(pool_train.size * 0.2)
    val_idx = pool_train[:n_val]
    train_idx = pool_train[n_val:]
    test_idx = np.concatenate([test_norm, test_anom])
    rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_balanced_detection_split(corpus: TelecomCorpus, seed: int = 42) -> dict[str, np.ndarray]:
    """1580/396/494 windows with 50% anomaly rate (uses all 1235 anomalies)."""
    rng = np.random.default_rng(seed)
    norm_pool = np.where(corpus.y == 0)[0]
    anom_pool = np.where(corpus.y == 1)[0]
    n_anom = anom_pool.size
    rng.shuffle(anom_pool)
    n_anom_test = 247  # mirror split_rca counts
    n_anom_val = 198
    n_anom_train = n_anom - n_anom_test - n_anom_val  # 790
    test_anom = anom_pool[:n_anom_test]
    val_anom = anom_pool[n_anom_test:n_anom_test + n_anom_val]
    train_anom = anom_pool[n_anom_test + n_anom_val:]
    norm_pick = _stratified_pick(rng, norm_pool, n_anom)
    rng.shuffle(norm_pick)
    test_norm = norm_pick[:n_anom_test]
    val_norm = norm_pick[n_anom_test:n_anom_test + n_anom_val]
    train_norm = norm_pick[n_anom_test + n_anom_val:]
    train_idx = np.concatenate([train_norm, train_anom])
    val_idx = np.concatenate([val_norm, val_anom])
    test_idx = np.concatenate([test_norm, test_anom])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_controlled_500_split(corpus: TelecomCorpus, seed: int = 42) -> dict[str, np.ndarray]:
    """300/100/100 windows with explicit 50 normal + 25 real Jamming + 25 synthetic test set."""
    rng = np.random.default_rng(seed)
    norm_pool = np.where(corpus.y == 0)[0]
    real_pool = np.where(corpus.anomaly_origin == "real")[0]
    synth_pool = np.where(corpus.anomaly_origin == "synthetic")[0]
    rng.shuffle(norm_pool); rng.shuffle(real_pool); rng.shuffle(synth_pool)

    # Test: 50 normal + 25 real + 25 synth
    test_norm = norm_pool[:50]
    test_real = real_pool[:25]
    test_synth = synth_pool[:25]
    # Val: 50 normal + 25 real + 25 synth (next chunk)
    val_norm = norm_pool[50:100]
    val_real = real_pool[25:50]
    val_synth = synth_pool[25:50]
    # Train: 150 normal + 75 real + 75 synth
    train_norm = norm_pool[100:250]
    train_real = real_pool[50:125]
    train_synth = synth_pool[50:125]

    train_idx = np.concatenate([train_norm, train_real, train_synth])
    val_idx = np.concatenate([val_norm, val_real, val_synth])
    test_idx = np.concatenate([test_norm, test_real, test_synth])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_fullscale_split(corpus: TelecomCorpus, seed: int = 42) -> dict[str, np.ndarray]:
    """25,600/6,400 windows with natural anomaly rates (~3.9%)."""
    rng = np.random.default_rng(seed)
    norm_pool = np.where(corpus.y == 0)[0]
    anom_pool = np.where(corpus.y == 1)[0]
    rng.shuffle(norm_pool); rng.shuffle(anom_pool)

    # On the real 32k corpus this reproduces the paper's exact 25,600 / 6,400
    # partition. For a smaller (e.g. synthetic offline) corpus we fall back to
    # a proportional 80/20 split so the same code path stays runnable.
    if corpus.n >= 32000:
        n_total = 32000
        n_train = 25600
        n_test = 6400
    else:
        n_total = corpus.n
        n_train = int(round(corpus.n * 0.8))
        n_test = corpus.n - n_train
    rate = float(anom_pool.size) / float(corpus.n)
    n_anom_train = int(round(n_train * rate))
    n_anom_test = anom_pool.size - n_anom_train  # use up all anomalies
    n_anom_test = max(min(n_anom_test, n_test), 1)
    train_anom = anom_pool[:n_anom_train]
    test_anom = anom_pool[n_anom_train:n_anom_train + n_anom_test]
    train_norm = norm_pool[:n_train - n_anom_train]
    test_norm = norm_pool[n_train - n_anom_train:n_train - n_anom_train + (n_test - n_anom_test)]
    train_idx = np.concatenate([train_norm, train_anom])
    test_idx = np.concatenate([test_norm, test_anom])
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return {"train": train_idx, "test": test_idx}


def make_rca_balanced_split(
    corpus: TelecomCorpus,
    seed: int = 42,
    n_test_target: int = 247,
) -> dict[str, np.ndarray]:
    """988/247 anomaly-only split, stratified by anomaly type (11-way).

    Uses the largest-remainder (Hamilton) allocation so the resulting
    train/test sizes hit ``n_total - n_test_target`` and ``n_test_target``
    exactly, regardless of per-class rounding residuals. Each present
    anomaly type contributes at least one test sample.
    """
    rng = np.random.default_rng(seed)
    anom_pool = np.where(corpus.y == 1)[0]
    types_all = corpus.anomaly_type[anom_pool]
    n_total = anom_pool.size
    if n_test_target > n_total:
        raise ValueError(f"n_test_target={n_test_target} exceeds anomaly count {n_total}")

    # 1. Per-type indices (shuffled deterministically by ``seed``).
    per_type: list[tuple[str, np.ndarray]] = []
    for t in ANOMALY_TYPES:
        mask = (types_all == t)
        idx = anom_pool[mask]
        if idx.size == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        per_type.append((t, idx))

    # 2. Largest-remainder allocation.
    sizes = np.array([idx.size for _, idx in per_type], dtype=np.int64)
    raw = sizes * (n_test_target / n_total)
    floor = np.floor(raw).astype(np.int64)
    remainders = raw - floor
    # Guarantee each present type gets at least one test sample.
    floor = np.maximum(floor, 1)
    # The +1 lift may push the sum above the target; trim by removing
    # singletons from the largest classes (those that have room) until
    # the total matches.
    while floor.sum() > n_test_target:
        # find a class with floor>1 that we can shrink, prefer the one with
        # smallest remainder so we hurt rounding fairness least.
        candidates = np.where(floor > 1)[0]
        if candidates.size == 0:
            break
        choose = candidates[np.argmin(remainders[candidates])]
        floor[choose] -= 1
    # Distribute any remaining shortfall using largest remainders, never
    # exceeding the available pool size for that class.
    while floor.sum() < n_test_target:
        order = np.argsort(-remainders)
        for i in order:
            if floor[i] < sizes[i]:
                floor[i] += 1
                remainders[i] = -np.inf  # don't pick this one again
                break
        else:
            break

    train_chunks, test_chunks = [], []
    for (_, idx), n_test in zip(per_type, floor):
        n_test = int(n_test)
        test_chunks.append(idx[:n_test])
        train_chunks.append(idx[n_test:])
    train_idx = np.concatenate(train_chunks)
    test_idx = np.concatenate(test_chunks)
    assert test_idx.size == n_test_target, (test_idx.size, n_test_target)
    assert train_idx.size == n_total - n_test_target
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return {"train": train_idx, "test": test_idx}


def all_splits(corpus: TelecomCorpus, seed: int = 42) -> dict[str, dict[str, np.ndarray]]:
    return {
        "small_natural": make_small_natural_split(corpus, seed),
        "balanced_detection": make_balanced_detection_split(corpus, seed),
        "controlled_500": make_controlled_500_split(corpus, seed),
        "fullscale": make_fullscale_split(corpus, seed),
        "split_rca_balanced": make_rca_balanced_split(corpus, seed),
    }


# -----------------------------------------------------------------------------
# Text masking utilities (E5)
# -----------------------------------------------------------------------------

def _mask_tokens(text: str, tokens: Sequence[str], placeholder: str = "[MASK]") -> str:
    if not text:
        return text
    out = text
    for tok in tokens:
        if not tok:
            continue
        out = re.sub(re.escape(tok), placeholder, out, flags=re.IGNORECASE)
    return out


def make_masked_text(text: str, level: str) -> str:
    """Return a masked version of ``text``.

    Levels:
      raw: unchanged.
      masked_type: anomaly-name tokens removed.
      masked_kpi: anomaly + KPI names removed.
      heavily_masked: anomaly + KPI + context + cue tokens removed.
    """
    if level == "raw":
        return text
    out = text
    if level in {"masked_type", "masked_kpi", "heavily_masked"}:
        out = _mask_tokens(out, ANOMALY_NAME_TOKENS)
    if level in {"masked_kpi", "heavily_masked"}:
        out = _mask_tokens(out, KPI_NAME_TOKENS)
    if level == "heavily_masked":
        out = _mask_tokens(out, CONTEXT_NAME_TOKENS)
        out = _mask_tokens(out, HEAVY_CUE_TOKENS)
    return out


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def kpi_indices(kpis: Sequence[str]) -> np.ndarray:
    """Return the integer column indices for a list of KPI names (in KPI_NAMES order)."""
    name_to_idx = {k: i for i, k in enumerate(KPI_NAMES)}
    return np.array([name_to_idx[k] for k in kpis], dtype=int)


def feature_indices_for_kpis(names: Sequence[str], kpis: Sequence[str]) -> np.ndarray:
    """Return indices into ``names`` (engineered feature names) that derive from ``kpis``."""
    kset = set(kpis)
    return np.array([i for i, nm in enumerate(names) if kpi_for_feature(nm) in kset], dtype=int)


def per_kpi_stat_means(corpus: TelecomCorpus) -> pd.DataFrame:
    """Return a (N, 16) DataFrame of per-window KPI means; useful for E2 quick tests."""
    means = corpus.X.mean(axis=1)  # (N, 16)
    return pd.DataFrame(means, columns=list(corpus.feature_cols))


def bootstrap_mean_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    obs = float(a.mean() - b.mean())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, a.size, a.size)
        ib = rng.integers(0, b.size, b.size)
        boot[i] = a[ia].mean() - b[ib].mean()
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return obs, lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = np.arange(1, n + 1) / n * alpha
    passed = ranked <= thresh
    if not passed.any():
        cutoff = -1
    else:
        cutoff = int(np.max(np.where(passed)[0]))
    out = np.zeros(n, dtype=bool)
    if cutoff >= 0:
        out[order[: cutoff + 1]] = True
    return out


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else float("nan")


# -----------------------------------------------------------------------------
# Zone / mobility selectors (encode the collection-protocol findings)
# -----------------------------------------------------------------------------

def zone_a_indices(corpus: TelecomCorpus) -> np.ndarray:
    """Indices of all windows collected in fixed Zone A (close to the RU).

    Real Jamming lives entirely in Zone A by collection design; restricting
    real-vs-synthetic comparisons to this subset removes the geographic
    confound documented in TelecomTS Table 7.
    """
    return np.where(corpus.zone == "A")[0]


def fixed_zone_indices(corpus: TelecomCorpus) -> np.ndarray:
    """Indices of windows that have a fixed zone label (A, B, or C).

    Excludes the 927 ``zone == "In motion"`` windows, which by collection
    protocol are mobile sessions with no fixed zone.
    """
    mask = np.isin(corpus.zone, ZONES_FIXED)
    return np.where(mask)[0]


def mobile_indices(corpus: TelecomCorpus) -> np.ndarray:
    """Indices of mobile-session windows (``mobility == "Yes"`` ↔ ``zone == "In motion"``)."""
    mask = (corpus.mobility == "Yes") | (corpus.zone == ZONE_IN_MOTION)
    return np.where(mask)[0]


def restrict_to_zone_a(corpus: TelecomCorpus, idx: np.ndarray) -> np.ndarray:
    """Filter ``idx`` to only those samples in Zone A."""
    return idx[corpus.zone[idx] == "A"]


def context_collection_findings(corpus: TelecomCorpus) -> dict:
    """Return a dict of collection-protocol findings the audit should surface.

    Useful for both the E1 audit and any wrap-up Markdown in later notebooks.
    """
    df = corpus.context_df()
    zone_counts = df["zone"].value_counts(dropna=False).to_dict()
    zone_x_origin = (
        df.assign(_=1)
        .pivot_table(index="zone", columns="anomaly_origin", values="_",
                     aggfunc="sum", fill_value=0)
        .to_dict("index")
    )
    zone_x_mobility_disjoint = bool(
        ((df["zone"] == ZONE_IN_MOTION) == (df["mobility"] == "Yes")).all()
    )
    real_only_in_zone_a = bool(
        ((df["anomaly_origin"] == "real") & (df["zone"] == "A")).sum()
        == (df["anomaly_origin"] == "real").sum()
    )
    no_anom_in_motion = bool(
        ((df["zone"] == ZONE_IN_MOTION) & (df["y"] == 1)).sum() == 0
    )
    return {
        "zone_counts": {str(k): int(v) for k, v in zone_counts.items()},
        "zone_by_anomaly_origin": {
            str(z): {str(o): int(c) for o, c in row.items()}
            for z, row in zone_x_origin.items()
        },
        "zone_in_motion_iff_mobility_yes": zone_x_mobility_disjoint,
        "real_jamming_only_in_zone_A": real_only_in_zone_a,
        "no_anomalies_during_mobility": no_anom_in_motion,
    }


def detection_rate(scores: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Fraction of mask==True samples whose score >= threshold."""
    sel = scores[mask]
    if sel.size == 0:
        return float("nan")
    return float((sel >= threshold).mean())


__all__ = [
    "KPI_NAMES",
    "KPI_GROUPS",
    "ANOMALY_TYPES",
    "REAL_ANOMALY_TYPES",
    "SYNTHETIC_ANOMALY_TYPES",
    "CONTEXT_FIELDS",
    "ANOMALY_NAME_TOKENS",
    "KPI_NAME_TOKENS",
    "CONTEXT_NAME_TOKENS",
    "HEAVY_CUE_TOKENS",
    "TelecomCorpus",
    "load_corpus",
    "extract_kpi_summary_features",
    "get_or_build_corpus_features",
    "feature_names",
    "kpi_for_feature",
    "kpi_indices",
    "feature_indices_for_kpis",
    "per_kpi_stat_means",
    "bootstrap_mean_diff_ci",
    "cohens_d",
    "benjamini_hochberg",
    "make_masked_text",
    "make_small_natural_split",
    "make_balanced_detection_split",
    "make_controlled_500_split",
    "make_fullscale_split",
    "make_rca_balanced_split",
    "all_splits",
    "EXPERIMENTS_ROOT",
    "REPO_ROOT",
    "CACHE_DIR",
    "CORPUS_CACHE",
    "FEATURE_CACHE",
    "exp_output_dir",
    "output_root",
    "synthetic_mode",
    "default_seeds",
    "make_synthetic_corpus",
    "detection_rate",
    "safe_div",
    "ZONES_FIXED",
    "ZONE_IN_MOTION",
    "zone_a_indices",
    "fixed_zone_indices",
    "mobile_indices",
    "restrict_to_zone_a",
    "context_collection_findings",
]
