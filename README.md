# CI-FaultGuard v2.0 — Multivariate Time-Series Fault Detection for CI Servers

**CI-FaultGuard** is an ML-powered multivariate time-series fault & anomaly detection system for CI/CD servers — modernized from the original ARIMA + Random Forest prototype into a fast, pure-NumPy multivariate detection engine, FastAPI REST API, interactive live dashboard, and automated Continuous Improvement benchmark loop.

---

## Key Capabilities

- **5 Multivariate Fault Detectors**:
  - `ensemble` (Best: **F1 = 0.9542**, Precision = 1.0, FPR = 0.0)
  - `mahalanobis` (Multivariate covariance distance — **F1 = 0.9216**)
  - `robust_mad` (Median + MAD Z-score — **F1 = 0.9167**)
  - `pca_reconstruction` (SVD Squared Prediction Error — **F1 = 0.8167**)
  - `ewma_residual` (Exponentially Weighted Moving Average residual)
- **Root-Cause Metric Attribution**: Every detected fault event includes severity (`INFO`, `WARNING`, `CRITICAL`), the primary `root_cause_metric`, and percentage contribution breakdown across all metrics.
- **Interactive Dashboard & REST API**: Upload any CSV, analyze the included `mem_use_20_Jan_NodeX9WS.csv` CI node trace, or simulate 4 labeled multivariate fault scenarios (Memory Leak, CPU Spike, Disk Saturation, Correlated Degradation).
- **Prometheus Integration**: Clean `/api/v1/query_range` client in `cifault/core/prometheus_client.py` configured via `PROMETHEUS_URL` and `PROMETHEUS_TOKEN`.
- **Continuous Improvement Loop**: Automated benchmark suite (`cifault benchmark --record`) logged to `benchmarks/history.jsonl` and gated by GitHub Actions CI.

---

## Quick Start

```bash
pip install -e ".[dev]"

# Launch interactive dashboard & OpenAPI docs (/docs)
python -m cifault.cli serve --host 127.0.0.1 --port 8001

# Detect faults on the included CI node memory trace
python -m cifault.cli detect mem_use_20_Jan_NodeX9WS.csv

# Run and record continuous-improvement benchmarks
python -m cifault.cli benchmark --record

# Run tests
pytest -v
```

---

## Documentation

- [Architecture & Prototype Modernization](docs/ARCHITECTURE.md)
- [Multivariate Detection Algorithms](docs/ALGORITHMS.md)
- [Continuous Improvement System](docs/CONTINUOUS_IMPROVEMENT.md)

---

## Research Paper & Citation

This project is based on the published Springer book chapter:

> *Multi-disciplinary Trends in Artificial Intelligence* (Proceedings of the 18th International Conference, MIWAI 2025), Chapter 32  
> Springer Nature, DOI: `10.1007/978-981-95-4957-3_32`  
> Paper link: [SpringerLink](https://link.springer.com/chapter/10.1007/978-981-95-4957-3_32)

```bibtex
@incollection{nguyen2025cifault,
  booktitle = {Multi-disciplinary Trends in Artificial Intelligence (MIWAI 2025)},
  publisher = {Springer Nature},
  year      = {2025},
  doi       = {10.1007/978-981-95-4957-3_32},
  url       = {https://link.springer.com/chapter/10.1007/978-981-95-4957-3_32}
}
```
