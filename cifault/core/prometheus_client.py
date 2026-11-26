"""Prometheus HTTP API client for fetching CI server multivariate time-series metrics."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
import requests


def fetch_prometheus_range(
    queries: dict[str, str],
    base_url: str | None = None,
    token: str | None = None,
    hours: int = 6,
    step: str = "15s",
) -> pd.DataFrame:
    """
    Query multiple PromQL expressions via `/api/v1/query_range` and align them into a DataFrame.

    Reads `PROMETHEUS_URL` and `PROMETHEUS_TOKEN` from environment variables if not provided.
    """
    url = (base_url or os.environ.get("PROMETHEUS_URL", "")).rstrip("/")
    if not url:
        raise ValueError("PROMETHEUS_URL is not configured")

    auth_token = token or os.environ.get("PROMETHEUS_TOKEN", "")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)

    series_map: dict[str, pd.Series] = {}
    for metric_name, promql in queries.items():
        resp = requests.get(
            f"{url}/api/v1/query_range",
            headers=headers,
            params={
                "query": promql,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "step": step,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        results: Sequence[dict] = payload.get("data", {}).get("result", [])
        if not results:
            continue
        values = results[0].get("values", [])
        idx = pd.to_datetime([float(v[0]) for v in values], unit="s", utc=True)
        vals = [float(v[1]) for v in values]
        series_map[metric_name] = pd.Series(vals, index=idx, name=metric_name)

    if not series_map:
        return pd.DataFrame()

    df = pd.DataFrame(series_map).sort_index()
    df.insert(0, "time", df.index.astype(str))
    return df.reset_index(drop=True)
