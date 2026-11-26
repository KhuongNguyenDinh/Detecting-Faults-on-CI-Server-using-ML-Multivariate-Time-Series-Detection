"""FastAPI backend for CI Server Multivariate Fault Detection Studio."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from cifault.core.detectors import ALL_DETECTORS, detect_faults
from cifault.core.simulator import generate_ci_telemetry

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
BENCHMARK_HISTORY_PATH = ROOT_DIR / "benchmarks" / "history.jsonl"


class SimulateDetectRequest(BaseModel):
    n_points: int = 600
    seed: int = 42
    detector: str = "ensemble"
    threshold: float = 0.52


class SampleDetectRequest(BaseModel):
    filename: str = "mem_use_20_Jan_NodeX9WS.csv"
    detector: str = "ensemble"
    threshold: float = 0.52


def create_app() -> FastAPI:
    app = FastAPI(
        title="CI-FaultGuard v2.0",
        description="Multivariate Time-Series Fault Detection for CI Servers",
        version="2.0.0",
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"detectors": ALL_DETECTORS},
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "2.0.0", "detectors": ALL_DETECTORS}

    @app.get("/api/samples")
    async def list_samples() -> dict[str, Any]:
        csvs = sorted([p.name for p in ROOT_DIR.glob("*.csv")])
        return {"samples": csvs}

    @app.post("/api/simulate")
    async def simulate_and_detect(req: SimulateDetectRequest) -> dict[str, Any]:
        df = generate_ci_telemetry(n_points=req.n_points, seed=req.seed)
        report = detect_faults(df, detector=req.detector, threshold=req.threshold)
        return {
            "source": f"synthetic_ci_telemetry(n={req.n_points}, seed={req.seed})",
            "timestamps": df["time"].tolist(),
            "series": {
                col: df[col].tolist()
                for col in report.metrics_analyzed
            },
            "ground_truth": df["is_fault"].tolist(),
            "fault_types": df["fault_type"].tolist(),
            "report": report.to_dict(),
        }

    @app.post("/api/detect/sample")
    async def detect_sample(req: SampleDetectRequest) -> dict[str, Any]:
        target = (ROOT_DIR / req.filename).resolve()
        if not target.is_file() or target.parent != ROOT_DIR.resolve():
            raise HTTPException(status_code=404, detail=f"Sample '{req.filename}' not found")
        df = pd.read_csv(target)
        report = detect_faults(df, detector=req.detector, threshold=req.threshold)
        ts = df["time"].astype(str).tolist() if "time" in df.columns else [str(i) for i in range(len(df))]
        return {
            "source": req.filename,
            "timestamps": ts,
            "series": {col: df[col].tolist() for col in report.metrics_analyzed},
            "report": report.to_dict(),
        }

    @app.post("/api/detect/upload")
    async def detect_upload(
        file: UploadFile = File(...),
        detector: str = Query("ensemble"),
        threshold: float = Query(0.52),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        raw = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
            report = detect_faults(df, detector=detector, threshold=threshold)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Detection failed: {exc}") from exc
        ts = df["time"].astype(str).tolist() if "time" in df.columns else [str(i) for i in range(len(df))]
        return {
            "source": file.filename,
            "timestamps": ts,
            "series": {col: df[col].tolist() for col in report.metrics_analyzed},
            "report": report.to_dict(),
        }

    @app.get("/api/benchmarks")
    async def get_benchmarks() -> dict[str, Any]:
        if not BENCHMARK_HISTORY_PATH.exists():
            return {"runs": []}
        runs = [
            json.loads(line)
            for line in BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {"runs": runs[-20:]}

    return app


app = create_app()
