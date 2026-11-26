"""Core multivariate time-series feature engineering, detection, and simulation."""

from cifault.core.detectors import (
    ALL_DETECTORS,
    DetectionReport,
    FaultEvent,
    detect_faults,
)
from cifault.core.features import build_multivariate_features, extract_numeric_matrix
from cifault.core.simulator import generate_ci_telemetry

__all__ = [
    "ALL_DETECTORS",
    "DetectionReport",
    "FaultEvent",
    "build_multivariate_features",
    "detect_faults",
    "extract_numeric_matrix",
    "generate_ci_telemetry",
]
