"""Command-line interface for CI-FaultGuard."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import uvicorn

from cifault.core.detectors import detect_faults
from cifault.eval.benchmark import run_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cifault", description="CI-FaultGuard CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the CI-FaultGuard dashboard and REST API")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8001)
    serve_p.add_argument("--reload", action="store_true")

    det_p = sub.add_parser("detect", help="Run fault detection on a CSV file")
    det_p.add_argument("file", help="Path to CSV telemetry file")
    det_p.add_argument("--detector", default="ensemble")
    det_p.add_argument("--threshold", type=float, default=0.52)

    bench_p = sub.add_parser("benchmark", help="Run continuous-improvement fault benchmark")
    bench_p.add_argument("--record", action="store_true", help="Append results to benchmarks/history.jsonl")

    args = parser.parse_args(argv)

    if args.command == "serve":
        uvicorn.run("cifault.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "detect":
        df = pd.read_csv(args.file)
        report = detect_faults(df, detector=args.detector, threshold=args.threshold)
        print(
            json.dumps(
                {
                    "detector": report.detector,
                    "threshold": report.threshold,
                    "total_points": report.total_points,
                    "fault_count": report.fault_count,
                    "fault_rate_pct": report.fault_rate_pct,
                    "metrics_analyzed": report.metrics_analyzed,
                    "sample_events": [e.to_dict() for e in report.events[:10]],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark":
        summary = run_benchmark_suite(record=args.record)
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
