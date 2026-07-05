from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .service import WorkbenchService


app = FastAPI(title="CameraE2E Workbench API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4175",
        "http://127.0.0.1:4175",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = WorkbenchService()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "camerae2e-workbench"}


@app.get("/api/presets")
def presets() -> dict:
    return service.presets()


@app.get("/api/assets/status")
def assets_status() -> dict:
    return service.assets_status()


@app.post("/api/simulate")
def simulate(payload: dict) -> dict:
    try:
        return service.simulate(payload)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/optimize")
def optimize(payload: dict) -> dict:
    try:
        return service.optimize(payload)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/dataset/export")
def dataset_export(payload: dict) -> dict:
    try:
        return service.dataset_export(payload)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/report")
def report(payload: dict | None = None) -> dict:
    try:
        return service.report(payload or {})
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

