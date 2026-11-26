"""Integration tests for FastAPI endpoints and dashboard UI."""

from fastapi.testclient import TestClient

from cifault.api.app import create_app

client = TestClient(create_app())


def test_dashboard_and_health():
    r = client.get("/")
    assert r.status_code == 200
    assert "CI-FaultGuard" in r.text

    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"


def test_simulate_and_sample_detection():
    sim = client.post("/api/simulate", json={"n_points": 300, "detector": "ensemble", "threshold": 0.52})
    assert sim.status_code == 200
    payload = sim.json()
    assert payload["report"]["total_points"] == 300
    assert payload["report"]["fault_count"] > 0

    sample = client.post(
        "/api/detect/sample",
        json={"filename": "mem_use_20_Jan_NodeX9WS.csv", "detector": "robust_mad", "threshold": 0.52},
    )
    assert sample.status_code == 200
    assert sample.json()["report"]["total_points"] > 1000


def test_upload_and_benchmarks():
    csv_bytes = b"time,cpu_usage,memory_usage\n2024-01-01 00:00:00,20,30\n2024-01-01 00:00:10,21,31\n2024-01-01 00:00:20,98,95\n"
    up = client.post(
        "/api/detect/upload?detector=ensemble&threshold=0.45",
        files={"file": ("node.csv", csv_bytes, "text/csv")},
    )
    assert up.status_code == 200
    assert up.json()["report"]["total_points"] == 3

    bench = client.get("/api/benchmarks")
    assert bench.status_code == 200
    assert "runs" in bench.json()
