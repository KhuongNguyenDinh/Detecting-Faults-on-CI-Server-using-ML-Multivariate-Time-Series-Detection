# Architecture & Modernization Notes

## 1. Package Layout

- `cifault/core/features.py` — Numeric matrix extraction, imputation, and rolling-window multivariate feature engineering (rolling mean/std, 1st diff, EWMA residual, z-score).
- `cifault/core/detectors.py` — 5 multivariate fault detectors (`ensemble`, `robust_mad`, `ewma_residual`, `mahalanobis`, `pca_reconstruction`) with severity classification and root-cause attribution.
- `cifault/core/simulator.py` — Synthetic multivariate CI server telemetry generator (`cpu_usage`, `memory_usage`, `disk_io_wait`, `network_rx_kb`, `build_queue_len`) with 4 labeled fault scenarios.
- `cifault/core/prometheus_client.py` — Environment-variable-configured Prometheus `query_range` client.
- `cifault/api/app.py` & `cifault/web/templates/index.html` — FastAPI REST API and interactive HTML5 Canvas dashboard.
- `cifault/eval/benchmark.py` — Multi-seed evaluation harness recording Precision, Recall, F1, FPR, and Runtime to `benchmarks/history.jsonl`.

## 2. Fixes Over Original Scripts

| Original File | Issue | Resolution in `cifault/` |
| :--- | :--- | :--- |
| `arirf.py` | Re-fitted `ARIMA(1,1,1)` 4 times; empty slice `fit_arima(...)[len(data):]`; univariate only. | Vectorized NumPy multivariate detectors running in ~2–6 ms with full root-cause attribution. |
| `export_data.py` | Top-level `return 1` / `return 0` outside a function (`SyntaxError`); empty `prometheus_url`. | Replaced with `fetch_prometheus_range()` using `PROMETHEUS_URL` and `PROMETHEUS_TOKEN`. |
| `requirements.txt` | Duplicate `pandas`/`numpy` lines and Python 2 `urllib2_file`. | Clean `pyproject.toml` + `requirements.txt`. |
