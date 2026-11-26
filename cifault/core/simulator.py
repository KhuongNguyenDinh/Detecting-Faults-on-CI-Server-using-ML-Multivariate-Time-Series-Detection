"""Synthetic multivariate CI server telemetry generator with labeled fault injections."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ci_telemetry(
    n_points: int = 600,
    seed: int = 42,
    freq_seconds: int = 10,
) -> pd.DataFrame:
    """
    Generate realistic multivariate CI server telemetry with 4 labeled fault scenarios:
      1. Memory Leak (gradual ramp + elevated GC)
      2. CPU Spike / Infinite Loop (sharp sustained spike)
      3. Disk I/O Saturation (high io_wait + queue backup)
      4. Correlated Multivariate Degradation (simultaneous moderate shift across metrics)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64)

    # Normal periodic CI workload + correlated noise
    diurnal = np.sin(2 * np.pi * t / max(60, n_points // 3))
    cpu = 32.0 + 8.0 * diurnal + rng.normal(0, 2.2, n_points)
    mem = 42.0 + 3.0 * diurnal + 0.15 * (cpu - 32.0) + rng.normal(0, 1.5, n_points)
    disk_io = 14.0 + 4.0 * np.maximum(0, diurnal) + rng.normal(0, 1.2, n_points)
    net_rx = 180.0 + 35.0 * diurnal + rng.normal(0, 10.0, n_points)
    build_queue = np.clip(2.0 + 1.5 * diurnal + rng.normal(0, 0.6, n_points), 0, None)

    is_fault = np.zeros(n_points, dtype=int)
    fault_type = np.array(["normal"] * n_points, dtype=object)

    if n_points >= 200:
        # 1. Memory Leak: [120, 160)
        s1, e1 = int(n_points * 0.20), int(n_points * 0.27)
        ramp = np.linspace(4.0, 38.0, e1 - s1)
        mem[s1:e1] += ramp
        cpu[s1:e1] += 6.0
        is_fault[s1:e1] = 1
        fault_type[s1:e1] = "memory_leak"

        # 2. CPU Spike: [240, 275)
        s2, e2 = int(n_points * 0.40), int(n_points * 0.46)
        cpu[s2:e2] += 48.0
        build_queue[s2:e2] += 8.0
        is_fault[s2:e2] = 1
        fault_type[s2:e2] = "cpu_spike"

        # 3. Disk I/O Saturation: [360, 395)
        s3, e3 = int(n_points * 0.60), int(n_points * 0.66)
        disk_io[s3:e3] += 42.0
        build_queue[s3:e3] += 6.0
        is_fault[s3:e3] = 1
        fault_type[s3:e3] = "disk_saturation"

        # 4. Correlated Multivariate Degradation: [480, 520)
        s4, e4 = int(n_points * 0.80), int(n_points * 0.87)
        cpu[s4:e4] += 18.0
        mem[s4:e4] += 16.0
        disk_io[s4:e4] += 14.0
        net_rx[s4:e4] += 95.0
        build_queue[s4:e4] += 5.0
        is_fault[s4:e4] = 1
        fault_type[s4:e4] = "correlated_degradation"

    timestamps = pd.date_range("2024-01-20 06:00:00", periods=n_points, freq=f"{freq_seconds}s")
    return pd.DataFrame(
        {
            "time": timestamps.astype(str),
            "cpu_usage": np.round(np.clip(cpu, 0, 100), 3),
            "memory_usage": np.round(np.clip(mem, 0, 100), 3),
            "disk_io_wait": np.round(np.clip(disk_io, 0, 100), 3),
            "network_rx_kb": np.round(np.clip(net_rx, 0, None), 3),
            "build_queue_len": np.round(np.clip(build_queue, 0, None), 3),
            "is_fault": is_fault,
            "fault_type": fault_type,
        }
    )
