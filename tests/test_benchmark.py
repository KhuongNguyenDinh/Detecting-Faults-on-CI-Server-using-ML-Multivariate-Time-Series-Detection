"""Regression tests for the continuous-improvement fault benchmark suite."""

from cifault.eval.benchmark import run_benchmark_suite


def test_benchmark_meets_f1_quality_floor():
    summary = run_benchmark_suite(record=False)
    assert summary["best_f1"] >= 0.70
    assert len(summary["results"]) == 5
