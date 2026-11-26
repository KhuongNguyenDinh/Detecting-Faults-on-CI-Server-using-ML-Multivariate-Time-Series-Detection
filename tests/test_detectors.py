"""Unit tests for multivariate feature engineering and fault detectors."""

from cifault.core.detectors import ALL_DETECTORS, detect_faults
from cifault.core.features import build_multivariate_features
from cifault.core.simulator import generate_ci_telemetry


def test_multivariate_feature_engineering():
    df = generate_ci_telemetry(n_points=120, seed=1)
    feats = build_multivariate_features(df, window=10)
    assert len(feats) == 120
    assert "cpu_usage__roll_mean" in feats.columns
    assert "memory_usage__ewma_resid" in feats.columns
    assert not feats.isna().any().any()


def test_all_detectors_and_root_cause_attribution():
    df = generate_ci_telemetry(n_points=600, seed=42)
    for det in ALL_DETECTORS:
        report = detect_faults(df, detector=det, threshold=0.52)
        assert report.total_points == 600
        assert report.fault_count > 0
        assert len(report.scores) == 600
        assert all(0.0 <= s <= 1.0 for s in report.scores)
        assert report.events[0].root_cause_metric in report.metrics_analyzed
        assert abs(sum(report.events[0].contributions.values()) - 100.0) < 1.0
