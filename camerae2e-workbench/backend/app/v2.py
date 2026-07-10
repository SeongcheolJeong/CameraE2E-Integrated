from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from camerae2e_v2 import CameraE2EService, StudyCreate


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateProjectBody(APIModel):
    name: str = Field(default="ADAS Camera Study", min_length=1)
    preset: str = "adas"


class JobBody(APIModel):
    kind: str
    request: dict[str, Any] = Field(default_factory=dict)


class OperationBody(APIModel):
    request: dict[str, Any] = Field(default_factory=dict)


def create_v2_router(service: CameraE2EService) -> APIRouter:
    router = APIRouter(prefix="/api/v2", tags=["CameraE2E v2"])

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "camerae2e-v2", "persistence": "sqlite"}

    @router.post("/bootstrap")
    def bootstrap() -> dict[str, Any]:
        return service.bootstrap()

    @router.get("/projects")
    def list_projects() -> list[dict[str, Any]]:
        return service.list_projects()

    @router.post("/projects", status_code=status.HTTP_201_CREATED)
    def create_project(body: CreateProjectBody) -> dict[str, Any]:
        try:
            project = service.create_project(body.name, preset=body.preset)
            return service.project_payload(project)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            return service.project_payload(service.projects.open(project_id))
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/assets/status")
    def assets_status(project_id: str) -> dict[str, Any]:
        try:
            return service.asset_status(project_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/assets")
    def camera_assets(project_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        try:
            return service.list_camera_assets(project_id, kind=kind)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/studies", status_code=status.HTTP_201_CREATED)
    def create_study(project_id: str, spec: StudyCreate) -> dict[str, Any]:
        try:
            return service.create_study(project_id, spec)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/studies/{study_id}")
    def get_study(project_id: str, study_id: str) -> dict[str, Any]:
        try:
            return service.get_study(project_id, study_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.put("/projects/{project_id}/studies/{study_id}")
    def update_study(project_id: str, study_id: str, spec: StudyCreate) -> dict[str, Any]:
        try:
            return service.update_study(project_id, study_id, spec)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/jobs")
    def list_jobs(
        project_id: str,
        study_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            return service.list_jobs(project_id, study_id=study_id, limit=min(max(limit, 1), 1000))
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(project_id: str, body: JobBody, response: Response) -> dict[str, Any]:
        try:
            study_id = body.request.get("study_id")
            job = service.submit_job(project_id, study_id, body.kind, body.request)
            response.headers["Location"] = f"/api/v2/projects/{project_id}/jobs/{job.id}"
            return {
                "job": job.model_dump(mode="json"),
                "poll_url": response.headers["Location"],
            }
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/jobs/{job_id}")
    def get_job(project_id: str, job_id: str) -> dict[str, Any]:
        try:
            return service.get_job(project_id, job_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/jobs/{job_id}/cancel")
    def cancel_job(project_id: str, job_id: str) -> dict[str, Any]:
        try:
            return service.cancel_job(project_id, job_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def submit_operation(
        project_id: str,
        study_id: str,
        kind: str,
        body: OperationBody,
        response: Response,
    ) -> dict[str, Any]:
        try:
            job = service.submit_job(project_id, study_id, kind, body.request)
            response.status_code = status.HTTP_202_ACCEPTED
            response.headers["Location"] = f"/api/v2/projects/{project_id}/jobs/{job.id}"
            return {
                "job": job.model_dump(mode="json"),
                "poll_url": response.headers["Location"],
            }
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/studies/{study_id}/evaluate")
    def evaluate(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "evaluate", body, response)

    @router.get("/projects/{project_id}/studies/{study_id}/benchmark/status")
    def benchmark_status(project_id: str, study_id: str) -> dict[str, Any]:
        try:
            return service.benchmark_status(project_id, study_id)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/projects/{project_id}/studies/{study_id}/benchmark/preflight")
    def benchmark_preflight(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "benchmark_preflight", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/benchmark/run")
    def benchmark_run(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "benchmark_run", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/benchmark/train-detector")
    def train_detector(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "train_detector", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/requirements/evaluate")
    def requirements_evaluate(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "requirements_evaluate", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/explore")
    def explore(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "sensitivity", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/optimize")
    def optimize(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "optimize", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/candidates/validate")
    def validate_candidate(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "validate_candidate", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/candidates/{case_id}/promote")
    def promote_candidate(
        project_id: str,
        study_id: str,
        case_id: str,
        body: OperationBody,
        response: Response,
    ) -> dict[str, Any]:
        request = {**body.request, "case_id": case_id}
        return submit_operation(
            project_id,
            study_id,
            "validate_candidate",
            OperationBody(request=request),
            response,
        )

    @router.post("/projects/{project_id}/studies/{study_id}/datasets")
    def dataset_export(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "dataset_export", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/calibrations")
    def calibrate(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "calibrate", body, response)

    @router.post("/projects/{project_id}/studies/{study_id}/reports")
    def report(
        project_id: str, study_id: str, body: OperationBody, response: Response
    ) -> dict[str, Any]:
        return submit_operation(project_id, study_id, "report", body, response)

    @router.get("/projects/{project_id}/artifacts/{artifact_hash}")
    def artifact(project_id: str, artifact_hash: str) -> FileResponse:
        try:
            path, media_type = service.artifact_path(project_id, artifact_hash)
            return FileResponse(path, media_type=media_type, filename=path.name)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/projects/{project_id}/studies/{study_id}/scenes/{scene_id}/preview")
    def scene_preview(project_id: str, study_id: str, scene_id: str) -> FileResponse:
        try:
            path, media_type = service.scene_preview_path(project_id, study_id, scene_id)
            return FileResponse(path, media_type=media_type)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
