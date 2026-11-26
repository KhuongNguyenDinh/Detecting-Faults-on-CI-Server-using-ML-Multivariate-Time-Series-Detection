# Continuous Improvement System

`cifault/eval/benchmark.py` evaluates every detector across multiple seeds of `generate_ci_telemetry` (Memory Leak, CPU Spike, Disk Saturation, Correlated Degradation) and logs runs to `benchmarks/history.jsonl`.

## Measured Improvement Progression

Using the benchmark loop directly uncovered normalization bottlenecks in `mahalanobis`, `pca_reconstruction`, and `ensemble` weighting:

| Detector | Initial F1 | Tuned F1 | Precision | Recall | FPR | Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ensemble` | `0.1683` | **0.9542** | `1.0000` | `0.9124` | `0.0000` | `5.55` |
| `mahalanobis` | `0.0859` | **0.9216** | `0.9139` | `0.9295` | `0.0308` | `3.49` |
| `robust_mad` | `0.9167` | **0.9167** | `1.0000` | `0.8462` | `0.0000` | `3.01` |
| `pca_reconstruction` | `0.2239` | **0.8167** | `1.0000` | `0.6902` | `0.0000` | `2.52` |
| `ewma_residual` | `0.1076` | `0.1076` | `0.4085` | `0.0620` | `0.0315` | `2.75` |

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest -v` (enforcing `best_f1 >= 0.70`) and `python -m cifault.cli benchmark` on every push and pull request.
