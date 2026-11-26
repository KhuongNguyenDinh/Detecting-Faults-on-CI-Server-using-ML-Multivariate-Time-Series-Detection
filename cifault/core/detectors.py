"""Multivariate time-series fault & anomaly detectors with root-cause attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from cifault.core.features import extract_numeric_matrix

ALL_DETECTORS = [
    "ensemble",
    "robust_mad",
    "ewma_residual",
    "mahalanobis",
    "pca_reconstruction",
]


@dataclass
class FaultEvent:
    """A detected fault point with severity and root-cause metric attribution."""

    index: int
    timestamp: str
    score: float
    severity: str
    root_cause_metric: str
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionReport:
    """Complete multivariate fault detection report."""

    detector: str
    threshold: float
    total_points: int
    fault_count: int
    fault_rate_pct: float
    metrics_analyzed: list[str]
    scores: list[float]
    is_fault: list[int]
    events: list[FaultEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "threshold": self.threshold,
            "total_points": self.total_points,
            "fault_count": self.fault_count,
            "fault_rate_pct": self.fault_rate_pct,
            "metrics_analyzed": self.metrics_analyzed,
            "scores": self.scores,
            "is_fault": self.is_fault,
            "events": [e.to_dict() for e in self.events],
        }


def _robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-column median, MAD-based scale, and robustly standardized matrix."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    # Fallback to std if MAD is zero (e.g., discrete/stepped metrics)
    std = np.std(X, axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, np.where(std > 1e-9, std, 1.0))
    Z = (X - med) / scale
    return med, scale, Z


def _scores_robust_mad(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, Z = _robust_scale(X)
    per_metric = np.abs(Z)
    # Normalize via soft-clipping to [0, 1] around z=3.5
    raw = np.max(per_metric, axis=1)
    norm = raw / (raw + 3.5)
    return norm, per_metric


def _scores_ewma_residual(X: np.ndarray, alpha: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    T, D = X.shape
    ewma = np.zeros_like(X)
    ewma[0] = X[0]
    for t in range(1, T):
        ewma[t] = alpha * X[t - 1] + (1.0 - alpha) * ewma[t - 1]
    resid = np.abs(X - ewma)
    _, scale, _ = _robust_scale(resid)
    # Scale residuals relative to baseline metric variability
    _, base_scale, _ = _robust_scale(X)
    per_metric = resid / np.maximum(base_scale, scale)
    raw = np.max(per_metric, axis=1)
    norm = raw / (raw + 3.0)
    return norm, per_metric


def _scores_mahalanobis(X: np.ndarray, reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    T, D = X.shape
    _, _, Z = _robust_scale(X)
    cov = np.cov(Z, rowvar=False) if T > 1 else np.eye(D)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + reg * np.eye(D)
    inv_cov = np.linalg.pinv(cov_reg)
    # Squared Mahalanobis distance per row: z^T Sigma^-1 z
    left = Z @ inv_cov
    d2 = np.maximum(0.0, np.sum(left * Z, axis=1))
    dist = np.sqrt(d2)
    # Normalize relative to expected sqrt(D)
    ref = np.sqrt(D)
    norm = dist / (dist + ref)
    per_metric = np.abs(Z)
    return norm, per_metric


def _scores_pca_reconstruction(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, D = X.shape
    _, _, Z = _robust_scale(X)
    if D <= 1 or T <= 2:
        per_metric = np.abs(Z)
        raw = np.max(per_metric, axis=1)
        return raw / (raw + 3.5), per_metric

    # SVD of standardized matrix; keep top-k components explaining ~85% variance (at most D-1)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    var = s**2
    cum = np.cumsum(var) / max(1e-9, np.sum(var))
    k = int(np.searchsorted(cum, 0.85) + 1)
    k = min(max(1, k), D - 1)
    Vk = vt[:k]
    Z_hat = (Z @ Vk.T) @ Vk
    err = np.abs(Z - Z_hat)
    spe = np.sqrt(np.sum(err**2, axis=1))
    ref = np.sqrt(max(1, D - k))
    norm = spe / (spe + ref)
    return norm, err


def detect_faults(
    df: pd.DataFrame,
    detector: str = "ensemble",
    threshold: float = 0.52,
    feature_cols: list[str] | None = None,
) -> DetectionReport:
    """
    Run multivariate time-series fault detection on a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame containing numeric telemetry columns and optional `time`/`timestamp`.
    detector:
        One of `ensemble`, `robust_mad`, `ewma_residual`, `mahalanobis`, `pca_reconstruction`.
    threshold:
        Normalized anomaly score threshold in (0, 1). Default 0.52.
    feature_cols:
        Optional subset of numeric columns to analyze.
    """
    det = detector.strip().lower()
    if det not in ALL_DETECTORS:
        raise ValueError(f"Unknown detector '{detector}'. Valid: {ALL_DETECTORS}")

    X, cols, timestamps = extract_numeric_matrix(df, feature_cols=feature_cols)

    if det == "robust_mad":
        scores, contrib_mat = _scores_robust_mad(X)
    elif det == "ewma_residual":
        scores, contrib_mat = _scores_ewma_residual(X)
    elif det == "mahalanobis":
        scores, contrib_mat = _scores_mahalanobis(X)
    elif det == "pca_reconstruction":
        scores, contrib_mat = _scores_pca_reconstruction(X)
    else:
        s_mad, c_mad = _scores_robust_mad(X)
        s_ewma, c_ewma = _scores_ewma_residual(X)
        s_maha, _ = _scores_mahalanobis(X)
        s_pca, _ = _scores_pca_reconstruction(X)
        peak = np.maximum.reduce([s_mad, s_maha, s_pca])
        scores = 0.55 * peak + 0.25 * s_mad + 0.15 * s_maha + 0.05 * s_ewma
        contrib_mat = 0.7 * c_mad + 0.3 * c_ewma

    flags = (scores >= threshold).astype(int)
    events: list[FaultEvent] = []
    for idx in np.where(flags == 1)[0]:
        sc = float(scores[idx])
        severity = "CRITICAL" if sc >= 0.72 else ("WARNING" if sc >= 0.58 else "INFO")
        row_contrib = contrib_mat[idx]
        total_c = float(np.sum(row_contrib)) or 1.0
        contrib_dict = {
            cols[j]: round(float(row_contrib[j] / total_c) * 100.0, 1)
            for j in range(len(cols))
        }
        root_metric = max(contrib_dict, key=contrib_dict.get)
        events.append(
            FaultEvent(
                index=int(idx),
                timestamp=timestamps[idx],
                score=round(sc, 4),
                severity=severity,
                root_cause_metric=root_metric,
                contributions=contrib_dict,
            )
        )

    total = len(scores)
    fault_count = int(np.sum(flags))
    return DetectionReport(
        detector=det,
        threshold=threshold,
        total_points=total,
        fault_count=fault_count,
        fault_rate_pct=round((fault_count / total) * 100.0, 2) if total else 0.0,
        metrics_analyzed=cols,
        scores=[round(float(s), 4) for s in scores],
        is_fault=[int(f) for f in flags],
        events=events,
    )
