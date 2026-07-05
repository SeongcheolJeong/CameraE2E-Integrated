from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .schemas import (
    AnalysisRequest,
    AnalysisSummaryResponse,
    DraftAutosaveRequest,
    DraftAutosaveResponse,
    DraftRestoreRequest,
    ExampleCheckRequest,
    ExampleCheckResponse,
    ModelOpenRequest,
    ModelResponse,
    ModelSaveRequest,
    ModelSettingsPatchRequest,
    NewModelRequest,
    PatentOpenRequest,
    QuickOptimizeRequest,
    QuickOptimizeResponse,
    SensorPatchRequest,
    SurfaceCreateRequest,
    SurfacePatchRequest,
    SystemPatchRequest,
    ToleranceSweepRequest,
    ToleranceSweepResponse,
)
from .service import RayOpticsStore


app = FastAPI(title="RayOptics Web Workbench", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RayOpticsStore()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "rayoptics": store.rayoptics_version}


@app.get("/api/examples")
def examples() -> dict[str, list[dict[str, str]]]:
    return {"examples": store.list_examples()}


@app.get("/api/patents/status")
def patent_status() -> dict:
    return store.patent_db_status()


@app.get("/api/patents/companies")
def patent_companies() -> dict[str, list[dict]]:
    return {"companies": store.list_patent_companies()}


@app.get("/api/patents/search")
def patent_search(
    company: str | None = None,
    query: str | None = None,
    status: str = "camerae2e_ready",
    limit: int = 80,
) -> dict[str, list[dict]]:
    return {"results": store.search_lens_patents(company=company, query=query, status=status, limit=limit)}


@app.post("/api/patents/open", response_model=ModelResponse)
def patent_open(payload: PatentOpenRequest) -> ModelResponse:
    try:
        return store.open_lens_patent(payload.simulation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/examples/check", response_model=ExampleCheckResponse)
def check_examples(payload: ExampleCheckRequest | None = None) -> ExampleCheckResponse:
    try:
        return store.check_examples(payload or ExampleCheckRequest())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/new", response_model=ModelResponse)
def new_model(payload: NewModelRequest) -> ModelResponse:
    return store.new_model(payload)


@app.post("/api/models/open", response_model=ModelResponse)
def open_model(payload: ModelOpenRequest) -> ModelResponse:
    try:
        return store.open_model(Path(payload.path).expanduser())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/save", response_model=ModelResponse)
def save_model(payload: ModelSaveRequest) -> ModelResponse:
    try:
        return store.save_model(
            payload.model_id,
            Path(payload.path).expanduser() if payload.path else None,
            overwrite=payload.overwrite,
            workbench=payload.workbench,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/draft", response_model=DraftAutosaveResponse)
def autosave_draft(model_id: str, payload: DraftAutosaveRequest | None = None) -> DraftAutosaveResponse:
    try:
        return store.autosave_draft(model_id, workbench=payload.workbench if payload else None)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/drafts/restore", response_model=ModelResponse)
def restore_draft(payload: DraftRestoreRequest) -> ModelResponse:
    try:
        return store.restore_draft(Path(payload.path).expanduser())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str) -> ModelResponse:
    try:
        return store.response(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.patch("/api/models/{model_id}/settings", response_model=ModelResponse)
def patch_model_settings(model_id: str, payload: ModelSettingsPatchRequest) -> ModelResponse:
    try:
        return store.patch_settings(model_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/models/{model_id}/surfaces/{surface_index}", response_model=ModelResponse)
def patch_surface(model_id: str, surface_index: int, payload: SurfacePatchRequest) -> ModelResponse:
    try:
        return store.patch_surface(model_id, surface_index, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/models/{model_id}/system", response_model=ModelResponse)
def patch_system(model_id: str, payload: SystemPatchRequest) -> ModelResponse:
    try:
        return store.patch_system(model_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/models/{model_id}/sensor", response_model=AnalysisSummaryResponse)
def patch_sensor(model_id: str, payload: SensorPatchRequest) -> AnalysisSummaryResponse:
    try:
        return store.patch_sensor(model_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/surfaces", response_model=ModelResponse)
def create_surface(model_id: str, payload: SurfaceCreateRequest) -> ModelResponse:
    try:
        return store.create_surface(model_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/models/{model_id}/surfaces/{surface_index}", response_model=ModelResponse)
def delete_surface(model_id: str, surface_index: int) -> ModelResponse:
    try:
        return store.delete_surface(model_id, surface_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/update", response_model=ModelResponse)
def update_model(model_id: str) -> ModelResponse:
    try:
        return store.update_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.post("/api/models/{model_id}/optimize/quick", response_model=QuickOptimizeResponse)
def quick_optimize(model_id: str, payload: QuickOptimizeRequest | None = None) -> QuickOptimizeResponse:
    try:
        return store.quick_optimize(model_id, payload or QuickOptimizeRequest())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/analysis/tolerance-sweep", response_model=ToleranceSweepResponse)
def tolerance_sweep(model_id: str, payload: ToleranceSweepRequest | None = None) -> ToleranceSweepResponse:
    try:
        return store.tolerance_sweep(model_id, payload or ToleranceSweepRequest())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/{model_id}/undo", response_model=ModelResponse)
def undo_model(model_id: str) -> ModelResponse:
    try:
        return store.undo(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.post("/api/models/{model_id}/redo", response_model=ModelResponse)
def redo_model(model_id: str) -> ModelResponse:
    try:
        return store.redo(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.get("/api/models/{model_id}/layout")
def layout(model_id: str) -> dict:
    try:
        return store.layout(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.get("/api/models/{model_id}/analysis/first-order")
def first_order(model_id: str) -> dict:
    try:
        return store.first_order(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.get("/api/models/{model_id}/analysis/summary", response_model=AnalysisSummaryResponse)
def analysis_summary(model_id: str) -> AnalysisSummaryResponse:
    try:
        return store.analysis_summary(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@app.post("/api/models/{model_id}/analysis/ray-fan")
def ray_fan(model_id: str, payload: AnalysisRequest) -> Response:
    return _analysis_svg(model_id, "ray-fan", payload)


@app.post("/api/models/{model_id}/analysis/opd-fan")
def opd_fan(model_id: str, payload: AnalysisRequest) -> Response:
    return _analysis_svg(model_id, "opd-fan", payload)


@app.post("/api/models/{model_id}/analysis/spot")
def spot(model_id: str, payload: AnalysisRequest) -> Response:
    return _analysis_svg(model_id, "spot", payload)


@app.post("/api/models/{model_id}/analysis/wavefront")
def wavefront(model_id: str, payload: AnalysisRequest) -> Response:
    return _analysis_svg(model_id, "wavefront", payload)


@app.post("/api/models/{model_id}/analysis/field-curves")
def field_curves(model_id: str, payload: AnalysisRequest) -> Response:
    return _analysis_svg(model_id, "field-curves", payload)


def _analysis_svg(model_id: str, kind: str, payload: AnalysisRequest) -> Response:
    try:
        svg = store.analysis_svg(model_id, kind, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=svg, media_type="image/svg+xml")
