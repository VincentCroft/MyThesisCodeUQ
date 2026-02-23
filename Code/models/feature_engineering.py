"""
PMU Fault Classifier — Feature Engineering
Protocol v1.2 Compatible

Converts raw PMU phasor (MAG, ANG) to rectangular (Re, Im),
normalizes DFDT/FREQ, builds sliding-window sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# ── Protocol constants ─────────────────────────────────────
PROTOCOL_COLS = [
    "DFDT",
    "FREQ",
    "IA_MAG",
    "IA_ANG",
    "IB_MAG",
    "IB_ANG",
    "IC_MAG",
    "IC_ANG",
    "VA_MAG",
    "VA_ANG",
    "VB_MAG",
    "VB_ANG",
    "VC_MAG",
    "VC_ANG",
    "TIMESTAMP",
    "ERROR_CODE",
]

PHASOR_PAIRS = [
    ("IA_MAG", "IA_ANG"),
    ("IB_MAG", "IB_ANG"),
    ("IC_MAG", "IC_ANG"),
    ("VA_MAG", "VA_ANG"),
    ("VB_MAG", "VB_ANG"),
    ("VC_MAG", "VC_ANG"),
]

# CLASS LABEL MAPPING  (must match configs/train_config.yaml)
CLASS_NAMES = ["NORMAL", "SLG_FAULT", "LL_FAULT", "THREE_PHASE_FAULT"]
CLASS_CODES = {
    0: "NORMAL",
    201: "SLG_FAULT",
    202: "LL_FAULT",
    204: "THREE_PHASE_FAULT",
}
LABEL_MAP = {
    0: 0,  # NORMAL
    201: 1,  # SLG_FAULT
    202: 2,  # LL_FAULT
    204: 3,  # THREE_PHASE_FAULT
}

# Normalization statistics (will be fitted on training data)
FEATURE_STATS: Optional[dict] = None


# ── Low-level helpers ──────────────────────────────────────


def polar_to_rect(
    mag: np.ndarray, ang_deg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert phasor polar -> rectangular.  NaN propagates correctly."""
    ang_rad = np.deg2rad(ang_deg)
    re = mag * np.cos(ang_rad)
    im = mag * np.sin(ang_rad)
    return re, im


def build_feature_matrix(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Build a 2-D feature matrix [N_rows × N_features] from a protocol DataFrame.

    Features (order):
      DFDT, FREQ,
      IA_Re, IA_Im, IB_Re, IB_Im, IC_Re, IC_Im,
      VA_Re, VA_Im, VB_Re, VB_Im, VC_Re, VC_Im
    """
    feat_cols_raw = cfg["features"]["raw_cols"]  # 14 measurement cols
    use_rect = cfg["features"]["use_polar_to_rect"]

    # ── 1. Scalar features ──
    scalars = df[["DFDT", "FREQ"]].values.astype(np.float32)  # [N, 2]

    # ── 2. Phasor features ──
    phasor_parts = []
    for mag_col, ang_col in PHASOR_PAIRS:
        mag = df[mag_col].values.astype(np.float32)
        ang = df[ang_col].values.astype(np.float32)
        if use_rect:
            re, im = polar_to_rect(mag, ang)
            phasor_parts.extend([re, im])
        else:
            phasor_parts.extend([mag, ang])

    phasors = np.stack(phasor_parts, axis=1)  # [N, 12]

    features = np.concatenate([scalars, phasors], axis=1)  # [N, 14]
    return features


def impute_nan(features: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Column-wise NaN imputation.  strategy: 'mean' | 'zero'."""
    out = features.copy()
    for col in range(out.shape[1]):
        col_data = out[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            if strategy == "mean":
                fill = np.nanmean(col_data) if not np.all(mask) else 0.0
            else:
                fill = 0.0
            out[mask, col] = fill
    return out


# ── Normalization ──────────────────────────────────────────


class FeatureNormalizer:
    """Z-score normalizer fitted per-feature on training data."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first."
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: Path) -> "FeatureNormalizer":
        data = np.load(path)
        n = cls()
        n.mean_ = data["mean"]
        n.std_ = data["std"]
        return n


# ── Sliding window ─────────────────────────────────────────


def sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping windows.
    Each window label = majority vote of contained row labels.
    Returns X [N_windows, window, F], y [N_windows]
    """
    xs, ys = [], []
    n = len(features)
    for start in range(0, n - window + 1, step):
        end = start + window
        xs.append(features[start:end])
        window_labels = labels[start:end]
        # Majority vote
        counts = np.bincount(window_labels, minlength=len(CLASS_NAMES))
        ys.append(int(np.argmax(counts)))
    if not xs:
        return np.empty((0, window, features.shape[1]), dtype=np.float32), np.empty(
            0, dtype=np.int64
        )
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


# ── File loading ───────────────────────────────────────────


def load_csv_file(
    path: Path,
    drop_unusable: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load one protocol CSV.
    Returns None if empty after filtering.
    """
    try:
        df = pd.read_csv(path, keep_default_na=True)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return None

    # Drop UNUSABLE (S >= 3, i.e. ERROR_CODE >= 300)
    if drop_unusable:
        df = df[df["ERROR_CODE"] < 300].reset_index(drop=True)

    if len(df) == 0:
        return None
    return df


def load_class_data(
    data_root: Path,
    class_name: str,
    cfg: dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all CSV files for one class folder.
    Returns (raw_features [N,F], labels [N]) or None.
    """
    folder = data_root / class_name
    if not folder.exists():
        warnings.warn(f"Class folder not found: {folder}")
        return None

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        warnings.warn(f"No CSV files in: {folder}")
        return None

    all_feat, all_lab = [], []
    for fp in csv_files:
        df = load_csv_file(fp, drop_unusable=cfg["data"]["drop_unusable"])
        if df is None:
            continue

        # Map ERROR_CODE to class label
        # Use the dominant ERROR_CODE to determine file-level class
        dominant_code = int(df["ERROR_CODE"].mode().iloc[0])
        label = LABEL_MAP.get(dominant_code)
        if label is None:
            # Try row-level mapping
            row_labels = df["ERROR_CODE"].map(LABEL_MAP).dropna()
            if len(row_labels) == 0:
                continue
            labels_arr = row_labels.values.astype(np.int64)
        else:
            labels_arr = np.full(len(df), label, dtype=np.int64)

        feat = build_feature_matrix(df, cfg)
        feat = impute_nan(feat)

        all_feat.append(feat)
        all_lab.append(labels_arr)

    if not all_feat:
        return None

    return np.concatenate(all_feat, axis=0), np.concatenate(all_lab, axis=0)


def load_inference_csv(
    path: Path,
    normalizer: FeatureNormalizer,
    cfg: dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load a single CSV for inference.
    Returns (windows [N_win, W, F], original_df).
    Rows with ERROR_CODE >= 300 are dropped but original_df is kept for display.
    """
    df_orig = pd.read_csv(path, keep_default_na=True)
    df = df_orig.copy()

    if "ERROR_CODE" in df.columns:
        df = df[df["ERROR_CODE"] < 300].reset_index(drop=True)

    feat = build_feature_matrix(df, cfg)
    feat = impute_nan(feat)
    feat = normalizer.transform(feat)

    window = cfg["data"]["window_size"]
    step = cfg["data"]["step_size"]

    # Create dummy labels for sliding_windows call
    dummy_labels = np.zeros(len(feat), dtype=np.int64)
    X, _ = sliding_windows(feat, dummy_labels, window, step)

    return X, df_orig
