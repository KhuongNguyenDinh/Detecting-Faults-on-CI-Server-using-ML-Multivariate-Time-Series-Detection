"""Multivariate time-series preprocessing and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_numeric_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Extract a clean (T, D) float64 matrix, column names, and timestamp strings from a DataFrame.

    Missing values are forward-filled, back-filled, then median-imputed.
    """
    work = df.copy()

    # Identify timestamp column or index
    if "time" in work.columns:
        timestamps = work["time"].astype(str).tolist()
    elif "timestamp" in work.columns:
        timestamps = work["timestamp"].astype(str).tolist()
    elif isinstance(work.index, pd.DatetimeIndex):
        timestamps = work.index.astype(str).tolist()
    else:
        timestamps = [str(i) for i in range(len(work))]

    if feature_cols is None:
        exclude = {"time", "timestamp", "label", "is_fault", "fault_type"}
        cols = [
            c
            for c in work.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(work[c])
        ]
    else:
        cols = [c for c in feature_cols if c in work.columns]

    if not cols:
        raise ValueError("No numeric telemetry columns found in dataset")

    num_df = work[cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.ffill().bfill()
    num_df = num_df.fillna(num_df.median(numeric_only=True)).fillna(0.0)

    return num_df.to_numpy(dtype=np.float64), [str(c) for c in cols], timestamps


def build_multivariate_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    window: int = 12,
) -> pd.DataFrame:
    """
    Build rolling-window multivariate features (mean, std, diff, EWMA residual, z-score)
    for each numeric telemetry metric.
    """
    X, cols, _ = extract_numeric_matrix(df, feature_cols=feature_cols)
    base = pd.DataFrame(X, columns=cols)
    out = pd.DataFrame(index=base.index)

    for col in cols:
        s = base[col]
        roll_mean = s.rolling(window=window, min_periods=1).mean()
        roll_std = s.rolling(window=window, min_periods=1).std(ddof=0).replace(0.0, 1e-6)
        ewma = s.ewm(span=max(2, window // 2), adjust=False).mean()

        out[col] = s
        out[f"{col}__roll_mean"] = roll_mean
        out[f"{col}__roll_std"] = roll_std
        out[f"{col}__diff1"] = s.diff().fillna(0.0)
        out[f"{col}__ewma_resid"] = s - ewma
        out[f"{col}__zscore"] = (s - roll_mean) / roll_std

    return out
