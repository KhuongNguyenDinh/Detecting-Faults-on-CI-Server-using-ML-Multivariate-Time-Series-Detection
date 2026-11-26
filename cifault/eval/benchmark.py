"""Automated multivariate fault detection benchmark & continuous-improvement tracker."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cifault.core.detectors import ALL_DETECTORS, detect_faults
from cifault.core.simulator import generate_ci_telemetry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_HISTORY_PATH = ROOT_DIR / "benchmarks" / "history.jsonl"


def evaluate_detector(
    detector: str,
    threshold: float = 0.52,
    seeds: tuple[int, ...] = (42, 7, 99),
    n_points: int = 600,
) -> dict[str, Any]:
    """Evaluate a detector across multiple labeled multivariate CI telemetry runs."""
    tp = fp = fn = tn = 0
    total_ms = 0.0

    for seed in seeds:
        df = generate_ci_telemetry(n_points=n_points, seed=seed)
        y_true = df["is_fault"].to_numpy(dtype=int)

        t0 = time.perf_counter()
        report = detect_faults(df, detector=detector, threshold=threshold)
        total_ms += (time.perf_counter() - t0) * 1000.0

        y_pred = np.asarray(report.is_fault, dtype=int)
        tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        tn += int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "detector": detector,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "runtime_ms": round(total_ms / len(seeds), 3),
    }


def run_benchmark_suite(record: bool = False) -> dict[str, Any]:
    """Benchmark all detectors and optionally record results to benchmarks/history.jsonl."""
    results = [evaluate_detector(det) for det in ALL_DETECTORS]
    best = max(results, key=lambda r: (r["f1"], -r["fpr"], -r["runtime_ms"]))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_detector": best["detector"],
        "best_f1": best["f1"],
        "results": results,
    }

    if record:
        BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BENCHMARK_HISTORY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")

    return summary


if __name__ == "__main__":
    print(json.dumps(run_benchmark_suite(record=True), indent=2))
