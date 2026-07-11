"""Application services and persistent local job orchestration."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from .benchmark import (
    benchmark_inventory,
    benchmark_scenes,
    resolve_benchmark_root,
)
from .benchmark import benchmark_manifest as build_benchmark_manifest
from .calibration import calibration_pack_status, fit_calibration
from .catalog import seed_builtin_camera_assets
from .components import ComponentCatalogService
from .dataset_sources import kitti_dataset_inventory, kitti_dataset_scenes
from .engine import CameraEngine
from .evaluation import StudyEvaluator, load_adas_label_payload
from .geometry import transform_label_payload
from .models import (
    AssetKind,
    CalibrationRequest,
    CameraAssetRecord,
    DatasetEstimateRequest,
    DatasetExportRequest,
    DatasetInventoryRequest,
    FidelityLevel,
    GeometryTransform,
    JobRecord,
    JobStatus,
    ModuleBaselineRequest,
    ModuleCompareRequest,
    ModuleEvaluationRequest,
    ReadinessTier,
    ReportRequest,
    SceneCase,
    StudyCreate,
    StudyRecord,
    utc_now,
)
from .project import Project, ProjectManager
from .provenance import canonical_hash, runtime_provenance, sha256_file
from .solvers import candidate_escalation_plan, solver_adapter


class CameraE2EService:
    """Single application boundary used by CLI, API, and React."""

    def __init__(
        self,
        projects_root: str | Path | None = None,
        *,
        max_workers: int = 2,
    ) -> None:
        self.projects = ProjectManager(projects_root)
        self.engine = CameraEngine()
        self.evaluator = StudyEvaluator(self.engine)
        self.components = ComponentCatalogService()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="camerae2e-v2"
        )
        self._futures: dict[str, Future[Any]] = {}
        self._future_lock = threading.RLock()

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def bootstrap(self) -> dict[str, Any]:
        projects = self.projects.list()
        if not projects:
            project = self.create_project("ADAS Camera Study", preset="adas")
        else:
            project = projects[0]
            if not project.store.list_studies():
                project.create_study(self.default_study_spec("adas"))
            seed_builtin_camera_assets(project)
        return self.project_payload(project)

    def create_project(self, name: str, *, preset: str = "adas") -> Project:
        project = self.projects.create(name)
        project.create_study(self.default_study_spec(preset))
        seed_builtin_camera_assets(project)
        return project

    def default_study_spec(self, preset: str = "adas") -> StudyCreate:
        spec = StudyCreate()
        if preset != "adas":
            return spec
        scene = self._discover_kitti_scene()
        model = self._discover_yolo_model()
        updates: dict[str, Any] = {"scenes": [scene] if scene else spec.scenes}
        if scene and scene.image_path:
            image = np.asarray(iio.imread(Path(scene.image_path).expanduser()))
            rows = int(np.ceil(image.shape[0] / 2.0) * 2)
            cols = int(np.floor(image.shape[1] / 2.0) * 2)
            updates["baseline"] = spec.baseline.model_copy(
                update={
                    "sensor": spec.baseline.sensor.model_copy(
                        update={
                            "rows": rows,
                            "cols": cols,
                            "qe_profile": "onsemi_ar0132at",
                        }
                    )
                }
            )
            updates["benchmark"] = spec.benchmark.model_copy(
                update={"source_root": str(Path(scene.image_path).parents[2])}
            )
        if model and scene and scene.label_path:
            updates.update(
                {
                    "target_profile": "adas_yolo_perception",
                    "perception_model_path": str(model),
                }
            )
        return spec.model_copy(update=updates)

    def list_projects(self) -> list[dict[str, Any]]:
        return [self.project_payload(project) for project in self.projects.list()]

    def project_payload(self, project: Project) -> dict[str, Any]:
        jobs = project.store.list_jobs(limit=20)
        return {
            "info": project.info.model_dump(mode="json"),
            "studies": [item.model_dump(mode="json") for item in project.store.list_studies()],
            "job_summary": {
                "total": len(jobs),
                "running": sum(item.status == JobStatus.RUNNING for item in jobs),
                "failed": sum(item.status == JobStatus.FAILED for item in jobs),
            },
            "artifact_summary": {
                "count": len(project.store.list_artifacts()),
                "validation": project.artifacts.validate(),
            },
            "camera_asset_summary": {
                "count": len(project.store.list_camera_assets()),
            },
        }

    def create_study(self, project_id: str, spec: StudyCreate) -> dict[str, Any]:
        project = self.projects.open(project_id)
        return project.create_study(spec).model_dump(mode="json")

    def update_study(self, project_id: str, study_id: str, spec: StudyCreate) -> dict[str, Any]:
        project = self.projects.open(project_id)
        return project.update_study(study_id, spec).model_dump(mode="json")

    def get_study(self, project_id: str, study_id: str) -> dict[str, Any]:
        return self.projects.open(project_id).store.get_study(study_id).model_dump(mode="json")

    def benchmark_status(self, project_id: str, study_id: str) -> dict[str, Any]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        latest: dict[str, Any] = {}
        for job in project.store.list_jobs(study_id=study_id, limit=500):
            if job.status == JobStatus.SUCCEEDED and job.kind not in latest:
                latest[job.kind] = {
                    "job_id": job.id,
                    "finished_at": job.finished_at,
                    "result": job.result,
                }
        preflight = latest.get("benchmark_preflight", {}).get("result")
        if preflight is None:
            preflight = (latest.get("optimize", {}).get("result") or {}).get("preflight")
        return {
            "schema_version": "camerae2e_benchmark_status_v2",
            "study_id": study_id,
            "inventory": benchmark_inventory(study),
            "preflight": preflight,
            "optimization_ready": bool(preflight and preflight.get("ready")),
            "latest_benchmark": latest.get("benchmark_run"),
            "latest_requirements": latest.get("requirements_evaluate"),
            "latest_optimization": latest.get("optimize"),
        }

    def benchmark_manifest(
        self,
        project_id: str,
        study_id: str,
        *,
        scene_count: int | None = None,
    ) -> dict[str, Any]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        count = int(scene_count or study.spec.benchmark.quick_scene_count)
        scenes = benchmark_scenes(study, count)
        if len(scenes) < count:
            raise ValueError(
                f"Benchmark manifest requested {count} scenes but only {len(scenes)} are available"
            )
        return build_benchmark_manifest(study, scenes)

    def dataset_inventory(
        self,
        project_id: str,
        study_id: str,
        request: DatasetInventoryRequest,
    ) -> dict[str, Any]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        root = self._dataset_source_root(study, request.source_root)
        return kitti_dataset_inventory(root, splits=request.source_splits)

    def dataset_estimate(
        self,
        project_id: str,
        study_id: str,
        request: DatasetEstimateRequest,
    ) -> dict[str, Any]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        if request.source_adapter == "kitti":
            root = self._dataset_source_root(study, request.source_root)
            inventory = kitti_dataset_inventory(root, splits=request.source_splits)
            available = int(inventory["available_scene_count"])
        else:
            inventory = {
                "schema_version": "camerae2e_dataset_inventory_v1",
                "adapter": "study",
                "available_scene_count": len(study.spec.scenes),
            }
            available = len(study.spec.scenes)
        scene_count = min(int(request.scene_count or available), available)
        candidates = self._dataset_candidates(project, study_id, request)
        sample_count = (
            scene_count
            * len(candidates)
            * len(request.exposure_variants_ev)
            * request.noise_repeats
        )
        rows = study.spec.baseline.sensor.rows
        cols = study.spec.baseline.sensor.cols
        bytes_per_sample = rows * cols * (4 + 3) + 4096
        return {
            "schema_version": "camerae2e_dataset_estimate_v1",
            "inventory": inventory,
            "scene_count": scene_count,
            "camera_count": len(candidates),
            "exposure_variant_count": len(request.exposure_variants_ev),
            "noise_repeats": request.noise_repeats,
            "sample_count": sample_count,
            "estimated_bytes": sample_count * bytes_per_sample,
            "estimated_gib": sample_count * bytes_per_sample / (1024**3),
            "resolution_policy": request.resolution_policy,
            "truth_boundary": (
                "Storage is estimated from configured simulation readout and may differ after "
                "compression. It is not an execution-time estimate."
            ),
        }

    def submit_job(
        self,
        project_id: str,
        study_id: str | None,
        kind: str,
        request: dict[str, Any] | None = None,
    ) -> JobRecord:
        project = self.projects.open(project_id)
        if study_id is not None:
            project.store.get_study(study_id)
        job = JobRecord(
            project_id=project.info.id,
            study_id=study_id,
            kind=kind,
            request=dict(request or {}),
        )
        project.store.put_job(job)
        future = self._executor.submit(self._execute_job, project.root, job.id)
        with self._future_lock:
            self._futures[job.id] = future
        future.add_done_callback(lambda _: self._forget_future(job.id))
        return job

    def execute_job_now(
        self,
        project_id: str,
        study_id: str | None,
        kind: str,
        request: dict[str, Any] | None = None,
    ) -> JobRecord:
        project = self.projects.open(project_id)
        job = JobRecord(
            project_id=project.info.id,
            study_id=study_id,
            kind=kind,
            request=dict(request or {}),
        )
        project.store.put_job(job)
        self._execute_job(project.root, job.id)
        return project.store.get_job(job.id)

    def get_job(self, project_id: str, job_id: str) -> dict[str, Any]:
        project = self.projects.open(project_id)
        job = project.store.get_job(job_id)
        payload = job.model_dump(mode="json")
        payload["artifacts"] = project.store.job_artifacts(job_id)
        return payload

    def list_jobs(
        self, project_id: str, *, study_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        project = self.projects.open(project_id)
        return [
            item.model_dump(mode="json")
            for item in project.store.list_jobs(study_id=study_id, limit=limit)
        ]

    def cancel_job(self, project_id: str, job_id: str) -> dict[str, Any]:
        project = self.projects.open(project_id)
        job = project.store.get_job(job_id)
        with self._future_lock:
            future = self._futures.get(job_id)
        if future is not None and future.cancel():
            job.status = JobStatus.CANCELLED
            job.finished_at = utc_now()
            job.log.append("Cancelled before local execution started.")
            project.store.put_job(job)
        elif job.status == JobStatus.RUNNING:
            job.request["cancel_requested"] = True
            job.log.append(
                "Cancellation requested; the current native solver call must finish first."
            )
            project.store.put_job(job)
        return job.model_dump(mode="json")

    def artifact_path(self, project_id: str, artifact_hash: str) -> tuple[Path, str]:
        project = self.projects.open(project_id)
        record = project.store.get_artifact(artifact_hash)
        return project.artifacts.resolve(artifact_hash), record.media_type

    def scene_preview_path(self, project_id: str, study_id: str, scene_id: str) -> tuple[Path, str]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        scene = next((item for item in study.spec.scenes if item.id == scene_id), None)
        if scene is None:
            raise KeyError(f"Unknown scene: {scene_id}")
        if not scene.image_path:
            raise FileNotFoundError("This synthetic scene has no source preview file")
        path = Path(scene.image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(path.suffix.lower(), "application/octet-stream")
        return path, media_type

    def asset_status(self, project_id: str | None = None) -> dict[str, Any]:
        payload = self.engine.asset_status()
        if project_id:
            project = self.projects.open(project_id)
            payload["project_artifacts"] = project.artifacts.validate()
            payload["calibration_pack"] = calibration_pack_status(project)
        payload["solver_adapters"] = {
            family: {
                "available": True,
                "execution": "explicit_validation_job_only",
            }
            for family in ("rayoptics", "fdtd", "tcad")
        }
        if project_id:
            studies = self.projects.open(project_id).store.list_studies()
            if studies:
                study = studies[0]
                payload["benchmark"] = benchmark_inventory(study)
                payload["perception"] = {
                    "model_path": study.spec.perception_model_path,
                    "model_ready": bool(
                        study.spec.perception_model_path
                        and Path(study.spec.perception_model_path).expanduser().is_file()
                    ),
                    "label_ready": bool(
                        study.spec.scenes
                        and study.spec.scenes[0].label_path
                        and Path(study.spec.scenes[0].label_path).expanduser().is_file()
                    ),
                }
        return payload

    def list_camera_assets(
        self, project_id: str, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        project = self.projects.open(project_id)
        return [
            item.model_dump(mode="json") for item in project.store.list_camera_assets(kind=kind)
        ]

    def search_lenses(self, **filters: Any) -> dict[str, Any]:
        return self.components.search_lenses(**filters)

    def lens_detail(self, simulation_id: str) -> dict[str, Any]:
        return self.components.lens_detail(simulation_id)

    def search_sensors(self, **filters: Any) -> dict[str, Any]:
        return self.components.search_sensors(**filters)

    def sensor_detail(self, sensor_id: str) -> dict[str, Any]:
        return self.components.sensor_detail(sensor_id)

    def evaluate_component_modules(self, request: ModuleEvaluationRequest) -> dict[str, Any]:
        return self.components.evaluate_modules(request.requirements, request.candidates)

    def apply_component_module(
        self,
        project_id: str,
        study_id: str,
        request: ModuleBaselineRequest,
    ) -> dict[str, Any]:
        project = self.projects.open(project_id)
        study = project.store.get_study(study_id)
        module, evaluation = self.components.build_module(
            request.selection, study.spec.baseline, study.spec.requirements
        )
        if evaluation["status"] == "incompatible" and not request.allow_incompatible:
            raise ValueError(
                "Component module is incompatible: " + ", ".join(evaluation["failed_gate_ids"])
            )
        descriptor = {
            "schema_version": "camerae2e_component_module_descriptor_v1",
            "selection": request.selection.model_dump(mode="json"),
            "module": module.model_dump(mode="json"),
            "compatibility": evaluation,
            "truth_boundary": evaluation["truth_boundary"],
        }
        artifact = project.artifacts.put_json(
            descriptor,
            artifact_type="camera_module_descriptor",
            fidelity_level=(
                FidelityLevel.LUT
                if module.metadata.get("use_geometric_psf")
                else FidelityLevel.ANALYTIC
            ),
            readiness_tier=ReadinessTier.PROXY,
            source="camerae2e_v2.component_catalog",
            validation={
                "compatibility_status": evaluation["status"],
                "failed_gate_ids": evaluation["failed_gate_ids"],
            },
        )
        asset = CameraAssetRecord(
            kind=AssetKind.CAMERA_PRESET,
            name=module.name,
            artifact_hashes=[artifact.hash],
            fidelity_level=(
                FidelityLevel.LUT
                if module.metadata.get("use_geometric_psf")
                else FidelityLevel.ANALYTIC
            ),
            readiness_tier=ReadinessTier.PROXY,
            parameters=module.model_dump(mode="json"),
            valid_domain={"requirements": study.spec.requirements.model_dump(mode="json")},
            source="camerae2e_v2.component_catalog",
            validation={"compatibility": evaluation},
        )
        project.store.put_camera_asset(asset)
        updated = project.update_study(study.id, study.spec.model_copy(update={"baseline": module}))
        return {
            "schema_version": "camerae2e_component_module_application_v1",
            "study": updated.model_dump(mode="json"),
            "module": module.model_dump(mode="json"),
            "compatibility": evaluation,
            "asset": asset.model_dump(mode="json"),
            "artifact": artifact.model_dump(mode="json"),
        }

    def _execute_job(self, project_root: Path, job_id: str) -> None:
        project = Project.open(project_root)
        job = project.store.get_job(job_id)
        if job.status == JobStatus.CANCELLED:
            return
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.progress = 0.02
        job.log.append(f"Started {job.kind}.")
        project.store.put_job(job)
        try:
            result = self._dispatch(project, job)
            job = project.store.get_job(job_id)
            if bool(job.request.get("cancel_requested", False)):
                job.status = JobStatus.CANCELLED
                job.progress = 1.0
                job.finished_at = utc_now()
                job.log.append("Cancelled after the current native operation returned.")
                project.store.put_job(job)
                return
            job.result = result
            job.status = JobStatus.SUCCEEDED
            job.progress = 1.0
            job.finished_at = utc_now()
            job.log.append(f"Completed {job.kind}.")
            project.store.put_job(job)
        except Exception as exc:
            job = project.store.get_job(job_id)
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = utc_now()
            job.log.append(job.error)
            project.store.put_job(job)

    def _dispatch(self, project: Project, job: JobRecord) -> dict[str, Any]:
        handlers = {
            "evaluate": self._run_evaluate,
            "benchmark_preflight": self._run_benchmark_preflight,
            "benchmark_run": self._run_benchmark,
            "requirements_evaluate": self._run_requirements,
            "train_detector": self._run_train_detector,
            "sensitivity": self._run_sensitivity,
            "optimize": self._run_optimize,
            "validate_candidate": self._run_validate_candidate,
            "dataset_export": self._run_dataset_export,
            "calibrate": self._run_calibration,
            "report": self._run_report,
            "compare_modules": self._run_compare_modules,
        }
        if job.kind not in handlers:
            raise ValueError(f"Unsupported CameraE2E v2 job kind: {job.kind}")
        return handlers[job.kind](project, job)

    def _run_compare_modules(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        request = ModuleCompareRequest.model_validate(job.request)
        scene_index = 0
        if request.scene_id is not None:
            scene_index = next(
                (
                    index
                    for index, scene in enumerate(study.spec.scenes)
                    if scene.id == request.scene_id
                ),
                -1,
            )
            if scene_index < 0:
                raise KeyError(f"Unknown scene: {request.scene_id}")
        compatibility = self.components.evaluate_modules(
            study.spec.requirements,
            request.candidates,
            baseline=study.spec.baseline,
        )
        if not request.allow_incompatible:
            blocked = [
                item for item in compatibility["candidates"] if item["status"] == "incompatible"
            ]
            if blocked:
                reasons = sorted(
                    {gate_id for item in blocked for gate_id in item["failed_gate_ids"]}
                )
                raise ValueError("Incompatible module comparison: " + ", ".join(reasons))
        input_hashes = self._register_study_inputs(project, job, study)
        evaluations = []
        for index, selection in enumerate(request.candidates):
            module, module_evaluation = self.components.build_module(
                selection, study.spec.baseline, study.spec.requirements
            )
            comparison_spec = study.spec.model_copy(update={"baseline": module})
            comparison_study = study.model_copy(update={"spec": comparison_spec})
            result = self.evaluator.evaluate_baseline(
                comparison_study,
                scene_index=scene_index,
                include_arrays=True,
            )
            persisted = self._persist_evaluation(
                project,
                job,
                result,
                input_hashes=input_hashes,
            )
            evaluations.append(
                {
                    "selection": selection.model_dump(mode="json"),
                    "compatibility": module_evaluation,
                    "evaluation": persisted,
                }
            )
            current = project.store.get_job(job.id)
            current.progress = 0.05 + 0.85 * (index + 1) / len(request.candidates)
            project.store.put_job(current)
        summary = {
            "schema_version": "camerae2e_component_module_comparison_v1",
            "study_id": study.id,
            "scene_id": study.spec.scenes[scene_index].id,
            "compatibility": compatibility,
            "evaluations": evaluations,
            "truth_boundary": (
                "All modules were rerun on the same scene and seed. Native sensor geometry "
                "drives engineering gates while image arrays use the recorded downsampled "
                "research readout. No automatic winner is asserted."
            ),
        }
        artifact = project.artifacts.put_json(
            summary,
            artifact_type="camera_module_comparison",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.PROXY,
            source="camerae2e_v2.component_catalog.compare",
            dependencies=input_hashes,
        )
        project.store.link_job_artifact(job.id, artifact.hash, "module_comparison")
        return {**summary, "artifact": artifact.model_dump(mode="json")}

    def _run_train_detector(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        if not bool(job.request.get("execute", False)):
            raise ValueError("Detector training requires execute=true")
        inventory = benchmark_inventory(study)
        root_value = inventory.get("root")
        if not root_value:
            raise ValueError("KITTI benchmark root is unavailable")
        root = Path(str(root_value)).resolve()
        if not (root / "images/train").is_dir() or not (root / "images/val").is_dir():
            raise ValueError("KITTI train/validation image directories are required")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Detector training requires the pyisetcam[yolo] dependencies"
            ) from exc
        epochs = min(max(int(job.request.get("epochs", 30)), 1), 100)
        image_size = min(max(int(job.request.get("image_size", 640)), 320), 1280)
        device = str(job.request.get("device", "auto"))
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        run_root = project.root / "runs" / f"detector_{job.id}"
        run_root.mkdir(parents=True, exist_ok=True)
        data_yaml = run_root / "kitti_absolute.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {root}",
                    "train: images/train",
                    "val: images/val",
                    "names:",
                    "  0: car",
                    "  1: van",
                    "  2: truck",
                    "  3: pedestrian",
                    "  4: Person_sitting",
                    "  5: cyclist",
                    "  6: tram",
                    "  7: misc",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        base_model = str(
            job.request.get("base_model") or study.spec.perception_model_path or "yolo11n.pt"
        )
        model = YOLO(base_model)

        def update_training_progress(trainer: Any) -> None:
            current = project.store.get_job(job.id)
            completed = min(int(getattr(trainer, "epoch", -1)) + 1, epochs)
            current.progress = min(0.85, 0.05 + (0.75 * completed / max(epochs, 1)))
            metrics = dict(getattr(trainer, "metrics", {}) or {})
            current.log.append(
                "Epoch "
                f"{completed}/{epochs}: "
                f"mAP50={float(metrics.get('metrics/mAP50(B)', 0.0)):.4f}, "
                f"mAP50-95={float(metrics.get('metrics/mAP50-95(B)', 0.0)):.4f}, "
                f"recall={float(metrics.get('metrics/recall(B)', 0.0)):.4f}"
            )
            if bool(current.request.get("cancel_requested", False)):
                trainer.stop = True
                current.log.append("Stopping detector training after the current epoch.")
            project.store.put_job(current)

        model.add_callback("on_fit_epoch_end", update_training_progress)
        train_result = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=image_size,
            device=device,
            project=str(run_root),
            name="kitti_yolo11n",
            seed=study.spec.seed,
            deterministic=True,
            patience=8,
            workers=2,
            verbose=False,
        )
        save_dir = Path(str(train_result.save_dir))
        best_path = save_dir / "weights" / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Ultralytics training did not create {best_path}")
        artifact = project.artifacts.put_file(
            best_path,
            artifact_type="adas_yolo_model",
            media_type="application/octet-stream",
            fidelity_level=FidelityLevel.ANALYTIC,
            readiness_tier=ReadinessTier.AVAILABLE,
            source="camerae2e_v2.detector_training",
            validation={"epochs": epochs, "dataset_root": str(root)},
        )
        project.store.link_job_artifact(job.id, artifact.hash, "perception_model")
        candidate_spec = study.spec.model_copy(update={"perception_model_path": str(best_path)})
        candidate_study = study.model_copy(update={"spec": candidate_spec})
        preflight = self.evaluator.preflight(
            candidate_study, scene_count=study.spec.benchmark.quick_scene_count
        )
        detector_ready = all(
            item.get("pass", False)
            for item in preflight.get("checks", [])
            if not str(item.get("id", "")).startswith("fidelity_")
        )
        activated = bool(detector_ready and job.request.get("activate", True))
        if activated:
            project.update_study(study.id, candidate_spec)
        return {
            "schema_version": "camerae2e_detector_training_v2",
            "model_path": str(best_path),
            "model_artifact": artifact.model_dump(mode="json"),
            "epochs": epochs,
            "device": device,
            "preflight": preflight,
            "detector_ready": detector_ready,
            "activated": activated,
            "status": "ready" if preflight.get("ready") else "training_completed_preflight_failed",
        }

    def _run_benchmark_preflight(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        result = self.evaluator.preflight(
            study,
            scene_count=job.request.get("scene_count"),
        )
        artifact = project.artifacts.put_json(
            result,
            artifact_type="benchmark_preflight",
            fidelity_level=FidelityLevel.ANALYTIC,
            readiness_tier=(
                ReadinessTier.VALIDATED if result.get("ready") else ReadinessTier.AVAILABLE
            ),
            source="camerae2e_v2.benchmark.preflight",
            dependencies=input_hashes,
            validation={"ready": bool(result.get("ready"))},
        )
        project.store.link_job_artifact(job.id, artifact.hash, "benchmark_preflight")
        return {**result, "artifact": artifact.model_dump(mode="json")}

    def _run_benchmark(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        result = self.evaluator.benchmark(
            study,
            scene_count=job.request.get("scene_count"),
            parameters=dict(job.request.get("parameters", {})),
            include_robustness=bool(job.request.get("include_robustness", True)),
        )
        if job.request.get("compare_fidelity", False):
            result["cross_fidelity"] = self.evaluator.compare_fidelities(
                study,
                scene_count=job.request.get("fidelity_scene_count"),
            )
        artifact = project.artifacts.put_json(
            result,
            artifact_type="benchmark_report",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.benchmark.run",
            dependencies=input_hashes,
        )
        project.store.link_job_artifact(job.id, artifact.hash, "benchmark")
        return {**result, "artifact": artifact.model_dump(mode="json")}

    def _run_requirements(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        evaluation = self.evaluator.evaluate_baseline(study, include_arrays=False)
        result = {
            "schema_version": "camerae2e_requirement_evaluation_v2",
            "study_id": study.id,
            "requirements": study.spec.requirements.model_dump(mode="json"),
            "result": evaluation.get("requirement_gates", {}),
            "geometry": evaluation.get("geometry"),
        }
        artifact = project.artifacts.put_json(
            result,
            artifact_type="requirement_gate_report",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.requirements",
            dependencies=input_hashes,
        )
        project.store.link_job_artifact(job.id, artifact.hash, "requirements")
        return {**result, "artifact": artifact.model_dump(mode="json")}

    def _run_evaluate(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        requested = job.request.get("fidelity")
        fidelity = FidelityLevel(requested) if requested else None
        result = self.evaluator.evaluate_baseline(study, fidelity=fidelity, include_arrays=True)
        return self._persist_evaluation(project, job, result, input_hashes=input_hashes)

    def _run_sensitivity(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        result = self.evaluator.sensitivity(study)
        artifact = project.artifacts.put_json(
            result,
            artifact_type="sensitivity_report",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.study.sensitivity",
            dependencies=input_hashes,
        )
        project.store.link_job_artifact(job.id, artifact.hash, "sensitivity")
        return {**result, "artifact": artifact.model_dump(mode="json")}

    def _run_optimize(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        result = self.evaluator.optimize(study)
        artifact = project.artifacts.put_json(
            result,
            artifact_type="optimization_report",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.study.optimize",
            dependencies=input_hashes,
        )
        project.store.link_job_artifact(job.id, artifact.hash, "optimization")
        result["artifact"] = artifact.model_dump(mode="json")
        if result.get("best_case"):
            result["escalation"] = candidate_escalation_plan(result["best_case"])
        return result

    def _run_validate_candidate(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        requested_case_id = job.request.get("case_id")
        candidate = dict(
            job.request.get("candidate")
            or (
                self._candidate_by_case_id(project, study.id, str(requested_case_id))
                if requested_case_id
                else self._latest_best_candidate(project, study.id)
            )
        )
        requested_families = job.request.get("families")
        escalation = candidate_escalation_plan(candidate)
        families = list(
            requested_families or [item["family"] for item in escalation["recommendations"]]
        )
        execute = bool(job.request.get("execute", False))
        acknowledge = bool(job.request.get("acknowledge_expensive", False))
        results = []
        optimization_hash = self._latest_artifact_hash_by_role(
            project, study.id, "optimize", "optimization"
        )
        for family in families:
            adapter = solver_adapter(str(family))
            prepared = adapter.prepare(
                project,
                candidate=candidate,
                job_id=job.id,
                dependencies=(() if optimization_hash is None else (optimization_hash,)),
            )
            project.store.link_job_artifact(job.id, prepared["artifact"]["hash"], f"{family}_input")
            submission = adapter.submit(
                prepared,
                execute=execute,
                acknowledge_expensive=acknowledge,
                timeout_s=study.spec.fidelity_policy.solver_timeout_s,
            )
            collected = adapter.collect(project, prepared, submission)
            project.store.link_job_artifact(
                job.id, collected["artifact"]["hash"], f"{family}_result"
            )
            results.append(collected)
        validations_ok = bool(results) and all(
            bool(item.get("validation", {}).get("ok", False)) for item in results
        )
        executed = any(bool(item.get("status") == "completed") for item in results)
        evidence_state = (
            "solver_evidenced"
            if validations_ok and executed
            else "lut_validated"
            if validations_ok
            else "analytic_screened"
        )
        evidence_hashes = [
            str(item.get("artifact", {}).get("hash"))
            for item in results
            if item.get("artifact", {}).get("hash")
        ]
        benchmark_rerun: dict[str, Any] = {
            "status": "blocked_no_consumable_solver_lut",
            "required_scene_count": study.spec.benchmark.final_scene_count,
            "message": (
                "A solver manifest alone cannot be attached as a camera LUT. A validated "
                "PSF/QE/collection artifact is required before automatic benchmark rerun."
            ),
        }
        fdtd_lut = next(
            (
                str(output["path"])
                for item in results
                if item.get("family") == "fdtd" and item.get("status") == "completed"
                for output in item.get("consumable_outputs", [])
                if output.get("role") == "camera_lut_json"
                and Path(str(output.get("path", ""))).is_file()
            ),
            None,
        )
        if fdtd_lut and bool(job.request.get("rerun_benchmark", True)):
            metadata = {**study.spec.baseline.metadata, "fdtd_lut_path": fdtd_lut}
            baseline = study.spec.baseline.model_copy(update={"metadata": metadata})
            policy = study.spec.fidelity_policy.model_copy(
                update={"search_level": FidelityLevel.LUT}
            )
            promoted_spec = study.spec.model_copy(
                update={"baseline": baseline, "fidelity_policy": policy}
            )
            promoted_study = study.model_copy(update={"spec": promoted_spec})
            rerun = self.evaluator.benchmark(
                promoted_study,
                scene_count=study.spec.benchmark.final_scene_count,
                parameters=dict(candidate.get("parameters", {})),
                include_robustness=True,
            )
            rerun_artifact = project.artifacts.put_json(
                rerun,
                artifact_type="candidate_promoted_benchmark",
                fidelity_level=FidelityLevel.LUT,
                readiness_tier=ReadinessTier.PROXY,
                source="camerae2e_v2.candidate.promotion",
                dependencies=evidence_hashes,
            )
            project.store.link_job_artifact(job.id, rerun_artifact.hash, "promoted_benchmark")
            benchmark_rerun = {
                "status": "completed",
                "lut_path": fdtd_lut,
                "scene_count": study.spec.benchmark.final_scene_count,
                "result": rerun,
                "artifact": rerun_artifact.model_dump(mode="json"),
            }
        promotion = {
            "case_id": str(candidate.get("case_id", f"case_{candidate.get('case_index', 0):04d}")),
            "previous_state": str(candidate.get("evidence_state", "analytic_screened")),
            "state": evidence_state,
            "next_state": "calibration_required",
            "evidence_artifacts": evidence_hashes,
            "benchmark_rerun": benchmark_rerun,
            "calibration_blocked": True,
            "message": (
                "Candidate evidence was advanced without promoting it to calibrated or sign-off."
            ),
        }
        return {
            "schema_version": "camerae2e_candidate_validation_v2",
            "candidate": candidate,
            "execute": execute,
            "escalation": escalation,
            "solver_results": results,
            "promotion": promotion,
            "truth_boundary": (
                "Solver completion is evidence generation, not automatic calibration or "
                "camera-module sign-off."
            ),
        }

    def _run_dataset_export(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        input_hashes = self._register_study_inputs(project, job, study)
        request = DatasetExportRequest.model_validate(job.request)
        candidates = self._dataset_candidates(project, study.id, request)
        optimization_hash = (
            None
            if request.selection == "baseline"
            else self._latest_artifact_hash_by_role(project, study.id, "optimize", "optimization")
        )
        if optimization_hash:
            input_hashes.append(optimization_hash)
        default_scene_count = (
            study.spec.benchmark.final_scene_count
            if study.spec.target_profile == "adas_yolo_perception"
            else len(study.spec.scenes)
        )
        scene_count = int(request.scene_count or default_scene_count)
        if request.source_adapter == "kitti":
            source_root = self._dataset_source_root(study, request.source_root)
            scenes = kitti_dataset_scenes(
                source_root,
                splits=request.source_splits,
                count=scene_count,
            )
            source_inventory = kitti_dataset_inventory(
                source_root,
                splits=request.source_splits,
            )
        else:
            source_root = None
            scenes = (
                benchmark_scenes(study, scene_count)
                if study.spec.target_profile == "adas_yolo_perception"
                else list(study.spec.scenes[:scene_count])
            )
            source_inventory = {
                "schema_version": "camerae2e_dataset_inventory_v1",
                "adapter": "study",
                "available_scene_count": len(study.spec.scenes),
            }
        if len(scenes) < scene_count:
            raise ValueError(
                f"Dataset export requested {scene_count} scenes but only "
                f"{len(scenes)} are available"
            )
        export_recipe = {
            "study_id": study.id,
            "study_revision": study.revision,
            "seed": study.spec.seed,
            "request": request.model_dump(mode="json"),
            "candidates": [dict(item.get("parameters", item)) for item in candidates],
            "scenes": [
                {
                    "id": scene.id,
                    "source_sha256": (
                        self._file_sha256(Path(scene.image_path).expanduser())
                        if scene.image_path
                        else canonical_hash(scene.model_dump(mode="json"))
                    ),
                    "label_sha256": (
                        self._file_sha256(Path(scene.label_path).expanduser())
                        if scene.label_path
                        else None
                    ),
                }
                for scene in scenes
            ],
        }
        export_recipe_hash = canonical_hash(export_recipe)
        directory_name = (
            f"dataset_{export_recipe_hash[:20]}" if request.resume else f"dataset_{job.id}"
        )
        output_dir = project.root / "exports" / directory_name
        for directory in ("raw", "raw_uint16", "rgb", "labels", "raw_tiff", "stages"):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint.jsonl"
        metadata_rows = (
            self._load_dataset_checkpoint(checkpoint_path, output_dir) if request.resume else []
        )
        resumed_sample_count = len(metadata_rows)
        completed_samples = {str(row["sample_id"]) for row in metadata_rows}
        split_counts = {"train": 0, "validation": 0, "test": 0}
        for row in metadata_rows:
            split_name = str(row.get("split", "train"))
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
        for candidate_index, candidate in enumerate(candidates):
            parameters = dict(candidate.get("parameters", candidate))
            case_id = str(candidate.get("case_id", f"candidate_{candidate_index:03d}"))
            camera_profile = self.engine.apply_module_overrides(
                study.spec.baseline,
                parameters,
            ).model_dump(mode="json")
            camera_profile_hash = canonical_hash(camera_profile)
            for scene_index, scene_case in enumerate(scenes):
                for exposure_index, exposure_ev in enumerate(request.exposure_variants_ev):
                    for noise_repeat in range(request.noise_repeats):
                        sample_parameters = dict(parameters)
                        base_exposure_s = float(
                            sample_parameters.get(
                                "sensor.integration_time",
                                study.spec.baseline.sensor.exposure_ms * 1e-3,
                            )
                        )
                        sample_parameters["sensor.integration_time"] = base_exposure_s * (
                            2.0 ** float(exposure_ev)
                        )
                        if request.resolution_policy == "source_bounded":
                            source_rows, source_cols = self._scene_image_size(scene_case)
                            target_rows = min(study.spec.baseline.sensor.rows, source_rows)
                            target_cols = min(study.spec.baseline.sensor.cols, source_cols)
                            cfa_alignment = max(
                                2, int(study.spec.baseline.sensor.binning_factor) * 2
                            )
                            sample_parameters["sensor.rows"] = max(
                                cfa_alignment,
                                target_rows - (target_rows % cfa_alignment),
                            )
                            sample_parameters["sensor.cols"] = max(
                                cfa_alignment,
                                target_cols - (target_cols % cfa_alignment),
                            )
                        contract = {
                            "candidate_id": case_id,
                            "candidate_index": candidate_index,
                            "scene_id": scene_case.id,
                            "source_hash": (
                                self._file_sha256(Path(scene_case.image_path).expanduser())
                                if scene_case.image_path
                                else canonical_hash(scene_case.model_dump(mode="json"))
                            ),
                            "parameters": sample_parameters,
                            "exposure_ev": exposure_ev,
                            "noise_repeat": noise_repeat,
                            "resolution_policy": request.resolution_policy,
                            "fidelity": (
                                request.fidelity_level or study.spec.fidelity_policy.search_level
                            ).value,
                        }
                        sample_id = f"sample_{canonical_hash(contract)[:20]}"
                        if sample_id in completed_samples:
                            continue
                        seed = (
                            study.spec.seed
                            + candidate_index * 1_000_000
                            + scene_index * 1000
                            + exposure_index * 100
                            + noise_repeat
                        )
                        result = self.engine.evaluate(
                            study.spec.baseline,
                            scene_case,
                            fidelity=request.fidelity_level
                            or study.spec.fidelity_policy.search_level,
                            policy=study.spec.fidelity_policy,
                            seed=seed,
                            parameter_overrides=sample_parameters,
                            include_arrays=True,
                        )
                        raw = self._result_stage_array(result, "sensor_raw", "sensor_digital")
                        sensor_digital = self._result_stage_array(result, "sensor_digital")
                        rgb = self._result_stage_array(result, "ip_srgb", "ip_result")
                        if raw is None or rgb is None:
                            raise ValueError(f"Simulation did not produce RAW/RGB for {sample_id}")
                        raw_export = np.asarray(raw, dtype=np.float32)
                        digital_export = np.asarray(
                            raw if sensor_digital is None else sensor_digital,
                            dtype=np.float32,
                        )
                        module_payload = dict(result.get("module", {}))
                        sensor_payload = dict(module_payload.get("sensor", {}))
                        bit_depth = int(sensor_payload.get("bit_depth", 12))
                        white_level = (1 << bit_depth) - 1
                        raw_path = output_dir / "raw" / f"{sample_id}.npz"
                        np.savez_compressed(
                            raw_path,
                            raw=raw_export,
                            sensor_digital=digital_export,
                            black_level=np.asarray([0], dtype=np.uint32),
                            white_level=np.asarray([white_level], dtype=np.uint32),
                            bit_depth=np.asarray([bit_depth], dtype=np.uint8),
                            cfa_pattern=np.asarray(
                                [str(sensor_payload.get("cfa_preset", "unknown"))]
                            ),
                        )
                        raw_uint16_path = None
                        if request.include_raw_uint16:
                            raw_uint16_path = output_dir / "raw_uint16" / f"{sample_id}.npy"
                            normalized = np.clip(digital_export, 0.0, 1.0)
                            np.save(
                                raw_uint16_path,
                                np.round(normalized * white_level).astype(np.uint16),
                                allow_pickle=False,
                            )
                        rgb_path = output_dir / "rgb" / f"{sample_id}.png"
                        rgb_bytes = self._png_bytes(rgb)
                        if rgb_bytes is None:
                            raise ValueError(f"Could not encode RGB preview for {sample_id}")
                        rgb_path.write_bytes(rgb_bytes)
                        label_payload: dict[str, Any] = {
                            "schema_version": "camerae2e_dataset_labels_v3",
                            "objects": [],
                            "source": None,
                        }
                        if scene_case.label_path:
                            source_labels = load_adas_label_payload(
                                scene_case.label_path,
                                image_size=self._scene_image_size(scene_case),
                            )
                            transform_payload = (result.get("geometry") or {}).get("transform")
                            label_payload = (
                                transform_label_payload(
                                    source_labels,
                                    GeometryTransform.model_validate(transform_payload),
                                )
                                if transform_payload
                                else source_labels
                            )
                            label_payload["schema_version"] = "camerae2e_dataset_labels_v3"
                        label_path = output_dir / "labels" / f"{sample_id}.json"
                        label_path.write_text(
                            json.dumps(label_payload, indent=2, sort_keys=True), encoding="utf-8"
                        )
                        if request.include_tiff:
                            tiff_path = output_dir / "raw_tiff" / f"{sample_id}.tiff"
                            iio.imwrite(
                                tiff_path,
                                np.round(np.clip(digital_export, 0.0, 1.0) * 65535.0).astype(
                                    np.uint16
                                ),
                            )
                        if request.include_stage_outputs:
                            for stage_name, stage in result.get("stages", {}).items():
                                if "array" not in stage:
                                    continue
                                np.savez_compressed(
                                    output_dir / "stages" / f"{sample_id}_{stage_name}.npz",
                                    array=np.asarray(stage["array"], dtype=np.float32),
                                )
                        split = self._dataset_split(scene_case)
                        split_counts[split] = split_counts.get(split, 0) + 1
                        source_rows, source_cols = self._scene_image_size(scene_case)
                        requested_rows = int(sensor_payload.get("rows", raw_export.shape[0]))
                        requested_cols = int(sensor_payload.get("cols", raw_export.shape[1]))
                        row = {
                            "sample_id": sample_id,
                            "candidate_id": case_id,
                            "scene_id": scene_case.id,
                            "group_id": scene_case.metadata.get("group_id", scene_case.id),
                            "split": split,
                            "seed": seed,
                            "exposure_ev": float(exposure_ev),
                            "noise_repeat": noise_repeat,
                            "source_kind": scene_case.source_kind,
                            "source_ref": {
                                "dataset": scene_case.metadata.get("dataset"),
                                "frame_id": scene_case.metadata.get("frame_id", scene_case.id),
                            },
                            "source_hash": (
                                self._file_sha256(Path(scene_case.image_path).expanduser())
                                if scene_case.image_path
                                else None
                            ),
                            "label_source_hash": (
                                self._file_sha256(Path(scene_case.label_path).expanduser())
                                if scene_case.label_path
                                else None
                            ),
                            "raw": str(raw_path.relative_to(output_dir)),
                            "raw_uint16": (
                                None
                                if raw_uint16_path is None
                                else str(raw_uint16_path.relative_to(output_dir))
                            ),
                            "rgb": str(rgb_path.relative_to(output_dir)),
                            "labels": str(label_path.relative_to(output_dir)),
                            "raw_shape": list(raw_export.shape),
                            "raw_dtype": str(raw_export.dtype),
                            "raw_sha256": sha256_file(raw_path),
                            "rgb_sha256": sha256_file(rgb_path),
                            "labels_sha256": sha256_file(label_path),
                            "label_object_count": len(label_payload.get("objects", [])),
                            "cfa_preset": sensor_payload.get("cfa_preset"),
                            "bit_depth": bit_depth,
                            "black_level": 0,
                            "white_level": white_level,
                            "units": "simulator_sensor_response",
                            "camera_config": module_payload,
                            "camera_profile": camera_profile,
                            "camera_profile_hash": camera_profile_hash,
                            "sample_contract_hash": canonical_hash(contract),
                            "resolution_policy": request.resolution_policy,
                            "source_effective_resolution_rc": [source_rows, source_cols],
                            "target_readout_rc": [requested_rows, requested_cols],
                            "upsampled_scene_proxy": bool(
                                request.resolution_policy == "target_readout_proxy"
                                and (requested_rows > source_rows or requested_cols > source_cols)
                            ),
                            "source_intrinsics": scene_case.metadata.get("source_intrinsics"),
                            "geometry": result.get("geometry"),
                            "fidelity": result.get("fidelity"),
                            "metrics": result.get("metrics"),
                            "truth_boundary": result.get("truth_boundary"),
                        }
                        metadata_rows.append(row)
                        completed_samples.add(sample_id)
                        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                            checkpoint.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        metadata_path = output_dir / "metadata.jsonl"
        metadata_rows.sort(key=lambda item: str(item["sample_id"]))
        metadata_path.write_text(
            "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in metadata_rows),
            encoding="utf-8",
        )
        selected_manifest = build_benchmark_manifest(study, scenes)
        candidate_contract = [
            {
                "case_id": str(item.get("case_id", f"candidate_{index:03d}")),
                "parameters": dict(item.get("parameters", item)),
                "configuration_hash": canonical_hash(dict(item.get("parameters", item))),
            }
            for index, item in enumerate(candidates)
        ]
        formats = ["raw_npz", "rgb_png", "labels_json", "metadata_jsonl"]
        if request.include_raw_uint16:
            formats.append("raw_uint16_npy")
        if request.include_tiff:
            formats.append("raw_tiff")
        if request.include_stage_outputs:
            formats.append("stage_npz")
        effective_fidelity = request.fidelity_level or study.spec.fidelity_policy.search_level
        manifest = {
            "schema_version": "camerae2e_raw_dataset_v3",
            "study_id": study.id,
            "selection": request.selection,
            "candidate_count": len(candidates),
            "scene_count": len(scenes),
            "case_count": len(metadata_rows),
            "seed": study.spec.seed,
            "benchmark_manifest": selected_manifest,
            "source_adapter": request.source_adapter,
            "source_root": None if source_root is None else str(source_root),
            "source_splits": request.source_splits,
            "source_inventory": source_inventory,
            "source_optimization_artifact": optimization_hash,
            "candidate_contract": candidate_contract,
            "camera_profile_summaries": self._dataset_profile_summaries(metadata_rows),
            "runtime": runtime_provenance(),
            "fidelity_policy": study.spec.fidelity_policy.model_dump(mode="json"),
            "effective_fidelity": effective_fidelity.value,
            "formats": formats,
            "split_counts": split_counts,
            "source_kinds": sorted({scene.source_kind for scene in scenes}),
            "label_policy": "caller_or_dataset_provided_only_no_automatic_inference",
            "dng_included": False,
            "raw_contract": {
                "container": "npz",
                "array_keys": [
                    "raw",
                    "sensor_digital",
                    "black_level",
                    "white_level",
                    "bit_depth",
                    "cfa_pattern",
                ],
                "dtype": "float32",
                "units": "simulator_sensor_response",
                "cfa_and_bit_depth_recorded_per_sample": True,
            },
            "recipe": {
                "hash": export_recipe_hash,
                "resolution_policy": request.resolution_policy,
                "exposure_variants_ev": request.exposure_variants_ev,
                "noise_repeats": request.noise_repeats,
                "resume": request.resume,
                "display_scene_assumption": {
                    "color_space": "sRGB",
                    "white_point": "D65",
                    "mean_luminance_cd_m2": 80.0,
                },
            },
            "integrity": {
                "metadata_jsonl_sha256": sha256_file(metadata_path),
                "sample_checksums_recorded": True,
            },
            "truth_boundary": (
                "RAW values are simulated from the recorded camera configuration. Source KITTI "
                "RGB frames are display-derived scene proxies and are not original KITTI RAW. "
                "target_readout_proxy may resize beyond source information but never adds "
                "measured spatial detail."
            ),
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        validation = self._validate_dataset_export(output_dir, manifest, metadata_rows)
        artifacts = []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            artifact = project.artifacts.put_file(
                path,
                artifact_type=self._dataset_artifact_type(path),
                media_type=self._media_type(path),
                fidelity_level=effective_fidelity,
                readiness_tier=ReadinessTier.VALIDATED,
                source="camerae2e_v2.dataset_export",
                dependencies=input_hashes,
                validation=validation if path.name == "manifest.json" else {},
                metadata={"dataset_relative_path": str(path.relative_to(output_dir))},
            )
            project.store.link_job_artifact(
                job.id, artifact.hash, self._dataset_artifact_type(path)
            )
            artifacts.append(artifact.model_dump(mode="json"))
        return {
            "schema_version": "camerae2e_dataset_export_v3",
            "dataset_root": str(output_dir),
            "selection": request.selection,
            "case_count": manifest.get("case_count", 0),
            "candidate_count": len(candidates),
            "scene_count": len(scenes),
            "source_adapter": request.source_adapter,
            "resolution_policy": request.resolution_policy,
            "resumed_sample_count": resumed_sample_count,
            "export_recipe_hash": export_recipe_hash,
            "camera_profile_summaries": manifest["camera_profile_summaries"],
            "manifest": manifest,
            "validation": validation,
            "artifacts": artifacts,
            "truth_boundary": (
                f"{manifest.get('truth_boundary', '')} Dataset split is deterministic by scene "
                "group id to prevent candidate variants from leaking across splits."
            ),
        }

    def _run_calibration(self, project: Project, job: JobRecord) -> dict[str, Any]:
        request = CalibrationRequest.model_validate(job.request)
        result = fit_calibration(project, request)
        project.store.link_job_artifact(job.id, result["artifact"]["hash"], "calibration_fit")
        return result

    def _run_report(self, project: Project, job: JobRecord) -> dict[str, Any]:
        study = self._study(project, job)
        request = ReportRequest.model_validate(job.request)
        jobs = [
            item
            for item in project.store.list_jobs(study_id=study.id, limit=500)
            if item.id != job.id and item.status == JobStatus.SUCCEEDED
        ]
        latest: dict[str, Any] = {}
        for item in jobs:
            latest.setdefault(item.kind, item.result)
        calibration_pack = calibration_pack_status(project)
        use_limit = self._report_use_limit(study, calibration_pack)
        payload = {
            "schema_version": "camerae2e_decision_report_v2",
            "title": request.title or f"{study.spec.name} Decision Report",
            "project": project.info.model_dump(mode="json"),
            "study": study.model_dump(mode="json"),
            "requirements": study.spec.requirements.model_dump(mode="json"),
            "baseline": latest.get("evaluate"),
            "module_comparison": latest.get("compare_modules"),
            "benchmark_preflight": latest.get("benchmark_preflight"),
            "benchmark": latest.get("benchmark_run"),
            "requirement_evaluation": latest.get("requirements_evaluate"),
            "sensitivity": latest.get("sensitivity"),
            "optimization": self._limit_candidates(
                latest.get("optimize"), request.include_candidates
            ),
            "candidate_validation": latest.get("validate_candidate"),
            "dataset": latest.get("dataset_export"),
            "calibration": latest.get("calibrate"),
            "calibration_pack": calibration_pack,
            "use_limit": use_limit,
            "fidelity_policy": study.spec.fidelity_policy.model_dump(mode="json"),
            "artifact_validation": project.artifacts.validate(),
            "reproduce": {
                "project": str(project.root),
                "study_id": study.id,
                "command": f"camerae2e run {project.root} {study.id} report",
            },
            "claim_boundary": (
                f"Use limit: {use_limit}. This decision report does not assert product sign-off. "
                "L3 claims require measured calibration evidence for every contributing stage."
            ),
        }
        report_dependencies = sorted(
            {
                item["artifact"]["hash"]
                for previous in jobs
                for item in project.store.job_artifacts(previous.id)
            }
        )
        json_artifact = project.artifacts.put_json(
            payload,
            artifact_type="decision_report_json",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.report",
            dependencies=report_dependencies,
        )
        html_content = self._render_report_html(payload).encode("utf-8")
        html_artifact = project.artifacts.put_bytes(
            html_content,
            artifact_type="decision_report_html",
            media_type="text/html",
            fidelity_level=study.spec.fidelity_policy.search_level,
            readiness_tier=ReadinessTier.VALIDATED,
            source="camerae2e_v2.report",
            dependencies=[json_artifact.hash],
        )
        project.store.link_job_artifact(job.id, json_artifact.hash, "report_json")
        project.store.link_job_artifact(job.id, html_artifact.hash, "report_html")
        report_dir = project.root / "reports" / job.id
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        (report_dir / "report.html").write_bytes(html_content)
        return {
            "schema_version": payload["schema_version"],
            "json_artifact": json_artifact.model_dump(mode="json"),
            "html_artifact": html_artifact.model_dump(mode="json"),
            "report_dir": str(report_dir),
            "claim_boundary": payload["claim_boundary"],
        }

    def _persist_evaluation(
        self,
        project: Project,
        job: JobRecord,
        result: dict[str, Any],
        *,
        input_hashes: list[str],
    ) -> dict[str, Any]:
        stage_artifacts = []
        preview_hash = None
        preview_hashes: dict[str, str] = {}
        for name, stage in result.get("stages", {}).items():
            if name not in {
                "sensor_raw",
                "sensor_digital",
                "ip_result",
                "ip_srgb",
                "ideal_recapture",
                "detector_overlay",
            }:
                continue
            if "array" not in stage:
                continue
            array = np.asarray(stage["array"])
            if array.size == 0:
                continue
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            artifact = project.artifacts.put_bytes(
                buffer.getvalue(),
                artifact_type=f"stage_{name}",
                media_type="application/x-npy",
                fidelity_level=FidelityLevel(result["fidelity"]["effective"]),
                readiness_tier=ReadinessTier(result["fidelity"]["readiness_tier"]),
                source="camerae2e_v2.engine",
                dependencies=input_hashes,
                metadata={"stage": name, "shape": list(array.shape), "dtype": str(array.dtype)},
            )
            project.store.link_job_artifact(job.id, artifact.hash, name)
            stage_artifacts.append(artifact.model_dump(mode="json"))
            if name in {"ip_srgb", "ip_result", "ideal_recapture", "detector_overlay"}:
                preview = self._png_bytes(array)
                if preview:
                    png = project.artifacts.put_bytes(
                        preview,
                        artifact_type="evaluation_preview",
                        media_type="image/png",
                        fidelity_level=FidelityLevel(result["fidelity"]["effective"]),
                        readiness_tier=ReadinessTier(result["fidelity"]["readiness_tier"]),
                        source="camerae2e_v2.engine.preview",
                        dependencies=[artifact.hash],
                    )
                    project.store.link_job_artifact(job.id, png.hash, "preview")
                    preview_hashes[name] = png.hash
                    if preview_hash is None and name in {"ip_srgb", "ip_result"}:
                        preview_hash = png.hash
        clean = {key: value for key, value in result.items() if key != "stages"}
        clean["stage_artifacts"] = stage_artifacts
        clean["preview_artifact_hash"] = preview_hash
        clean["preview_artifact_hashes"] = preview_hashes
        report = project.artifacts.put_json(
            clean,
            artifact_type="evaluation_report",
            fidelity_level=FidelityLevel(result["fidelity"]["effective"]),
            readiness_tier=ReadinessTier(result["fidelity"]["readiness_tier"]),
            source="camerae2e_v2.study.evaluate",
            dependencies=input_hashes + [item["hash"] for item in stage_artifacts],
        )
        project.store.link_job_artifact(job.id, report.hash, "evaluation")
        clean["artifact"] = report.model_dump(mode="json")
        return clean

    @staticmethod
    def _study(project: Project, job: JobRecord) -> Any:
        if job.study_id is None:
            raise ValueError(f"{job.kind} requires study_id")
        return project.store.get_study(job.study_id)

    @staticmethod
    def _latest_best_candidate(project: Project, study_id: str) -> dict[str, Any]:
        for job in project.store.list_jobs(study_id=study_id, limit=500):
            if job.kind == "optimize" and job.status == JobStatus.SUCCEEDED and job.result:
                candidate = job.result.get("best_case")
                if candidate:
                    return dict(candidate)
        raise ValueError("Run optimization before validating a candidate")

    @staticmethod
    def _candidate_by_case_id(project: Project, study_id: str, case_id: str) -> dict[str, Any]:
        for job in project.store.list_jobs(study_id=study_id, limit=500):
            if job.kind != "optimize" or job.status != JobStatus.SUCCEEDED or not job.result:
                continue
            for candidate in job.result.get("cases", []):
                if str(candidate.get("case_id")) == case_id:
                    return dict(candidate)
        raise ValueError(f"Unknown optimized candidate: {case_id}")

    @staticmethod
    def _latest_artifact_hash_by_role(
        project: Project,
        study_id: str,
        job_kind: str,
        role: str,
    ) -> str | None:
        for previous in project.store.list_jobs(study_id=study_id, limit=500):
            if previous.kind != job_kind or previous.status != JobStatus.SUCCEEDED:
                continue
            for item in project.store.job_artifacts(previous.id):
                if item["role"] == role:
                    return str(item["artifact"]["hash"])
        return None

    @staticmethod
    def _register_study_inputs(project: Project, job: JobRecord, study: Any) -> list[str]:
        hashes: list[str] = []
        scene = study.spec.scenes[0]
        source_tier = {
            "physical": ReadinessTier.AVAILABLE,
            "measured_proxy": ReadinessTier.AVAILABLE,
            "display_rgb_proxy": ReadinessTier.PROXY,
            "synthetic": ReadinessTier.VALIDATED,
        }[scene.source_kind]
        inputs = [
            (
                scene.image_path,
                "scene_source",
                CameraE2EService._media_type(
                    Path(scene.image_path) if scene.image_path else Path("scene.bin")
                ),
                source_tier,
            ),
            (scene.label_path, "scene_labels", "text/plain", ReadinessTier.AVAILABLE),
            (
                study.spec.perception_model_path,
                "perception_model",
                "application/octet-stream",
                ReadinessTier.AVAILABLE,
            ),
        ]
        for raw_path, artifact_type, media_type, readiness in inputs:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                continue
            artifact = project.artifacts.put_file(
                path,
                artifact_type=artifact_type,
                media_type=media_type,
                fidelity_level=FidelityLevel.ANALYTIC,
                readiness_tier=readiness,
                source="camerae2e_v2.study_input",
                metadata={"study_id": study.id, "scene_id": scene.id},
            )
            project.store.link_job_artifact(job.id, artifact.hash, artifact_type)
            hashes.append(artifact.hash)
        return hashes

    @staticmethod
    def _dataset_candidates(
        project: Project, study_id: str, request: DatasetExportRequest
    ) -> list[dict[str, Any]]:
        if request.selection == "baseline":
            study = project.store.get_study(study_id)
            allowed = {"baseline", study.spec.baseline.id}
            unknown = set(request.camera_ids) - allowed
            if unknown:
                raise ValueError(f"Unknown baseline camera_ids: {sorted(unknown)}")
            return [
                {"case_id": study.spec.baseline.id, "parameters": {}}
                for _ in range(request.case_count)
            ]
        optimization = None
        for job in project.store.list_jobs(study_id=study_id, limit=500):
            if job.kind == "optimize" and job.status == JobStatus.SUCCEEDED:
                optimization = job.result
                break
        if not optimization:
            raise ValueError("Run optimization before exporting selected candidates")
        if request.selection == "best":
            source = [optimization.get("best_case")]
        elif request.selection == "pareto":
            source = optimization.get("pareto_front", [])
        else:
            source = optimization.get("top_cases", [])
        candidates = [dict(item) for item in source if item]
        if request.camera_ids:
            requested = set(request.camera_ids)
            candidates = [item for item in candidates if str(item.get("case_id")) in requested]
            resolved = {str(item.get("case_id")) for item in candidates}
            if missing := requested - resolved:
                raise ValueError(f"Unknown optimization camera_ids: {sorted(missing)}")
        if not candidates:
            raise ValueError(f"Optimization has no candidates for selection={request.selection}")
        return (candidates * request.case_count)[: request.case_count]

    @staticmethod
    def _scene_image_size(scene: SceneCase) -> tuple[int, int]:
        if scene.image_path and Path(scene.image_path).expanduser().is_file():
            image = iio.imread(Path(scene.image_path).expanduser())
            return int(image.shape[0]), int(image.shape[1])
        return 96, 320

    @staticmethod
    def _result_stage_array(result: dict[str, Any], *names: str) -> np.ndarray | None:
        for name in names:
            stage = result.get("stages", {}).get(name, {})
            if isinstance(stage, dict) and "array" in stage:
                array = np.asarray(stage["array"])
                if array.size:
                    return array
        return None

    @staticmethod
    def _dataset_split(scene: SceneCase) -> str:
        declared = str(scene.metadata.get("split", "")).strip().lower()
        if declared in {"validation", "val"}:
            return "validation"
        if declared in {"train", "test"}:
            return declared
        group_id = str(scene.metadata.get("group_id", scene.id))
        bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "validation"
        return "test"

    @staticmethod
    def _dataset_source_root(study: StudyRecord, requested: str | None) -> Path:
        root: Path | None
        if requested:
            root = Path(requested).expanduser().resolve()
        else:
            root = resolve_benchmark_root(study)
        if root is None or not root.is_dir():
            raise ValueError(
                "KITTI source root is unavailable; set source_root or benchmark.source_root"
            )
        return root

    @staticmethod
    def _load_dataset_checkpoint(path: Path, root: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            required = [root / str(row.get(key, "")) for key in ("raw", "rgb", "labels")]
            if all(item.is_file() for item in required):
                rows.append(row)
        return rows

    @staticmethod
    def _dataset_profile_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("camera_profile_hash", "unknown"))
            groups.setdefault(key, []).append(row)
        summaries = []
        for profile_hash, samples in sorted(groups.items()):
            def metric(
                group: str,
                name: str,
                sample_rows: list[dict[str, Any]] = samples,
            ) -> float | None:
                values = [
                    item.get("metrics", {}).get(group, {}).get(name)
                    for item in sample_rows
                ]
                numeric = [float(value) for value in values if isinstance(value, (int, float))]
                return float(np.mean(numeric)) if numeric else None

            summaries.append(
                {
                    "camera_profile_hash": profile_hash,
                    "camera_name": samples[0].get("camera_config", {}).get("name"),
                    "sample_count": len(samples),
                    "mean_rgb_high_clip_fraction": metric(
                        "artifact", "rgb_high_clip_fraction"
                    ),
                    "mean_raw_std": metric("artifact", "raw_std"),
                    "mean_rgb_signal": metric("color", "rgb_mean"),
                    "label_object_count": sum(
                        int(item.get("label_object_count", 0)) for item in samples
                    ),
                    "fidelity": samples[0].get("fidelity"),
                }
            )
        return summaries

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return sha256_file(path)

    @staticmethod
    def _validate_dataset_export(
        root: Path,
        manifest: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        group_splits: dict[str, str] = {}
        source_splits: dict[str, str] = {}
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                issues.append({"kind": "duplicate_sample_id", "sample_id": sample_id})
            sample_ids.add(sample_id)
            for key in ("raw", "rgb", "labels"):
                if not (root / str(row[key])).is_file():
                    issues.append(
                        {"kind": "missing_sample_file", "sample_id": sample_id, "role": key}
                    )
            raw_path = root / str(row["raw"])
            rgb_path = root / str(row["rgb"])
            label_path = root / str(row["labels"])
            if raw_path.is_file():
                try:
                    with np.load(raw_path, allow_pickle=False) as archive:
                        if "raw" not in archive:
                            raise KeyError("raw")
                        raw = np.asarray(archive["raw"])
                    if raw.dtype != np.float32:
                        issues.append(
                            {
                                "kind": "raw_dtype_mismatch",
                                "sample_id": sample_id,
                                "expected": "float32",
                                "actual": str(raw.dtype),
                            }
                        )
                    if list(raw.shape) != list(row.get("raw_shape", [])):
                        issues.append(
                            {
                                "kind": "raw_shape_mismatch",
                                "sample_id": sample_id,
                                "expected": row.get("raw_shape"),
                                "actual": list(raw.shape),
                            }
                        )
                    if not np.all(np.isfinite(raw)):
                        issues.append({"kind": "raw_nonfinite", "sample_id": sample_id})
                except (OSError, ValueError, KeyError) as exc:
                    issues.append(
                        {
                            "kind": "raw_read_error",
                            "sample_id": sample_id,
                            "detail": str(exc),
                        }
                    )
            if rgb_path.is_file():
                try:
                    rgb = np.asarray(iio.imread(rgb_path))
                    if list(rgb.shape[:2]) != list(row.get("raw_shape", []))[:2]:
                        issues.append(
                            {
                                "kind": "rgb_raw_alignment",
                                "sample_id": sample_id,
                                "rgb_shape": list(rgb.shape),
                                "raw_shape": row.get("raw_shape"),
                            }
                        )
                except (OSError, ValueError) as exc:
                    issues.append(
                        {"kind": "rgb_read_error", "sample_id": sample_id, "detail": str(exc)}
                    )
            if label_path.is_file():
                try:
                    labels = json.loads(label_path.read_text(encoding="utf-8"))
                    CameraE2EService._validate_export_labels(
                        labels,
                        tuple(int(value) for value in row.get("raw_shape", [])[:2]),
                        sample_id,
                        issues,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(
                        {
                            "kind": "label_read_error",
                            "sample_id": sample_id,
                            "detail": str(exc),
                        }
                    )
            for role, path, expected in (
                ("raw", raw_path, row.get("raw_sha256")),
                ("rgb", rgb_path, row.get("rgb_sha256")),
                ("labels", label_path, row.get("labels_sha256")),
            ):
                if path.is_file() and expected and sha256_file(path) != expected:
                    issues.append(
                        {"kind": "checksum_mismatch", "sample_id": sample_id, "role": role}
                    )
            group = str(row["group_id"])
            previous = group_splits.setdefault(group, str(row["split"]))
            if previous != row["split"]:
                issues.append(
                    {
                        "kind": "split_leakage",
                        "group_id": group,
                        "splits": [previous, row["split"]],
                    }
                )
            source_hash = row.get("source_hash")
            if source_hash:
                source_previous = source_splits.setdefault(str(source_hash), str(row["split"]))
                if source_previous != row["split"]:
                    issues.append(
                        {
                            "kind": "source_hash_split_leakage",
                            "source_hash": source_hash,
                            "splits": [source_previous, row["split"]],
                        }
                    )
        if len(rows) != int(manifest.get("case_count", -1)):
            issues.append(
                {
                    "kind": "manifest_count_mismatch",
                    "manifest": manifest.get("case_count"),
                    "actual": len(rows),
                }
            )
        metadata_path = root / "metadata.jsonl"
        expected_metadata_hash = (manifest.get("integrity") or {}).get("metadata_jsonl_sha256")
        if (
            metadata_path.is_file()
            and expected_metadata_hash
            and sha256_file(metadata_path) != expected_metadata_hash
        ):
            issues.append({"kind": "metadata_checksum_mismatch"})
        recorded_manifest_hash = manifest.get("manifest_hash")
        manifest_without_hash = {
            key: value for key, value in manifest.items() if key != "manifest_hash"
        }
        if recorded_manifest_hash != canonical_hash(manifest_without_hash):
            issues.append({"kind": "manifest_hash_mismatch"})
        return {
            "ok": not issues,
            "sample_count": len(rows),
            "group_count": len(group_splits),
            "source_count": len(source_splits),
            "issue_count": len(issues),
            "issues": issues,
        }

    @staticmethod
    def _validate_export_labels(
        payload: dict[str, Any],
        image_size_rc: tuple[int, ...],
        sample_id: str,
        issues: list[dict[str, Any]],
    ) -> None:
        if not isinstance(payload, dict):
            issues.append({"kind": "label_payload_invalid", "sample_id": sample_id})
            return
        if len(image_size_rc) != 2:
            issues.append({"kind": "label_image_shape_missing", "sample_id": sample_id})
            return
        rows, cols = image_size_rc
        recorded_size = payload.get("image_size_rc")
        if recorded_size is not None and list(recorded_size) != [rows, cols]:
            issues.append(
                {
                    "kind": "label_image_size_mismatch",
                    "sample_id": sample_id,
                    "expected": [rows, cols],
                    "actual": recorded_size,
                }
            )
        for index, item in enumerate(payload.get("objects", [])):
            bbox = item.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues.append(
                    {"kind": "label_bbox_invalid", "sample_id": sample_id, "index": index}
                )
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if not (0.0 <= x1 < x2 <= cols and 0.0 <= y1 < y2 <= rows):
                issues.append(
                    {
                        "kind": "label_bbox_out_of_bounds",
                        "sample_id": sample_id,
                        "index": index,
                        "bbox_xyxy": bbox,
                        "image_size_rc": [rows, cols],
                    }
                )

    @staticmethod
    def _discover_kitti_scene() -> SceneCase | None:
        repo = Path(__file__).resolve().parents[2]
        configured = os.environ.get("CAMERAE2E_KITTI_ROOT", "").strip()
        roots = [repo / "camerae2e-workbench/data/kitti"]
        if configured:
            roots.insert(0, Path(configured).expanduser())
        for root in roots:
            if not root.is_dir():
                continue
            image_root = root / "images" / "train"
            label_root = root / "labels" / "train"
            for image in sorted(image_root.glob("*.png")):
                label = label_root / f"{image.stem}.txt"
                if label.is_file():
                    return SceneCase(
                        name=f"KITTI {image.stem}",
                        source_kind="display_rgb_proxy",
                        scene_type="rgb_file",
                        image_path=str(image),
                        label_path=str(label),
                        metadata={
                            "dataset": "KITTI",
                            "truth_boundary": (
                                "RGB-to-scene display proxy; original scene spectra and sensor RAW "
                                "are not recovered."
                            ),
                        },
                    )
        return None

    @staticmethod
    def _discover_yolo_model() -> Path | None:
        repo = Path(__file__).resolve().parents[2]
        configured = os.environ.get("CAMERAE2E_YOLO_MODEL", "").strip()
        candidates = [
            repo / "camerae2e-workbench/runs/yolo_training/checkpoints/"
            "kitti_yolo11n_epoch4_best.pt",
            repo / "camerae2e-workbench/runs/yolo_training/"
            "kitti_yolo11n_e1_workbench/weights/best.pt",
        ]
        if configured:
            candidates.insert(0, Path(configured).expanduser())
        return next((path.resolve() for path in candidates if path.is_file()), None)

    @staticmethod
    def _png_bytes(array: np.ndarray) -> bytes | None:
        image = np.asarray(array, dtype=float)
        if image.size == 0:
            return None
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3:
            return None
        image = image[..., :3]
        image = np.nan_to_num(image)
        if image.max() > 1.0 or image.min() < 0.0:
            span = image.max() - image.min()
            image = (image - image.min()) / span if span > 0 else np.zeros_like(image)
        buffer = io.BytesIO()
        iio.imwrite(buffer, np.clip(image * 255.0, 0, 255).astype(np.uint8), extension=".png")
        return buffer.getvalue()

    @staticmethod
    def preview_data_url(content: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(content).decode("ascii")

    @staticmethod
    def _dataset_artifact_type(path: Path) -> str:
        if path.name == "manifest.json":
            return "dataset_manifest"
        if path.name == "metadata.jsonl":
            return "dataset_metadata"
        if path.parent.name == "raw":
            return "dataset_raw"
        if path.parent.name == "rgb":
            return "dataset_rgb"
        if path.parent.name == "labels":
            return "dataset_labels"
        return "dataset_file"

    @staticmethod
    def _media_type(path: Path) -> str:
        return {
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".npz": "application/x-npz",
            ".npy": "application/x-npy",
            ".png": "image/png",
            ".tiff": "image/tiff",
            ".html": "text/html",
        }.get(path.suffix.lower(), "application/octet-stream")

    @staticmethod
    def _limit_candidates(result: dict[str, Any] | None, count: int) -> dict[str, Any] | None:
        if result is None:
            return None
        payload = dict(result)
        payload["top_cases"] = list(payload.get("top_cases", []))[:count]
        payload.pop("cases", None)
        return payload

    @staticmethod
    def _report_use_limit(study: Any, calibration_pack: dict[str, Any]) -> str:
        if (
            study.spec.fidelity_policy.search_level == FidelityLevel.CALIBRATED
            and calibration_pack.get("complete")
        ):
            return "validated"
        if study.spec.fidelity_policy.search_level != FidelityLevel.ANALYTIC:
            return "calibration_required"
        return "research_only"

    @staticmethod
    def _render_report_html(payload: dict[str, Any]) -> str:
        def number(value: Any, digits: int = 4) -> str:
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "--"

        optimization = payload.get("optimization") or {}
        candidates = optimization.get("top_cases", [])
        rows = "".join(
            "<tr>"
            f"<td>{item.get('case_index')}</td>"
            f"<td>{float(item.get('target_score', 0.0)):.5f}</td>"
            f"<td>{html.escape(CameraE2EService._format_score_interval(item))}</td>"
            f"<td>{float((item.get('perception_metrics') or {}).get('map50_95', 0.0)):.4f}</td>"
            f"<td>{float((item.get('perception_metrics') or {}).get('recall50', 0.0)):.4f}</td>"
            f"<td>{int(item.get('scene_count', 0))}</td>"
            f"<td>{html.escape(json.dumps(item.get('parameters', {}), sort_keys=True))}</td>"
            f"<td>{'yes' if item.get('feasible') else 'no'}</td>"
            f"<td>{html.escape(str(item.get('evidence_state', 'analytic_screened')))}</td>"
            "</tr>"
            for item in candidates
        )
        module_comparison = payload.get("module_comparison") or {}

        def module_row(index: int, item: dict[str, Any]) -> str:
            selection = item.get("selection") or {}
            compatibility = item.get("compatibility") or {}
            evaluation = item.get("evaluation") or {}
            score = evaluation.get("evaluation") or {}
            artifact_metrics = (evaluation.get("metrics") or {}).get("artifact") or {}
            fidelity = (evaluation.get("fidelity") or {}).get("effective", "--")
            hfov = (compatibility.get("derived") or {}).get("hfov_deg")
            return (
                "<tr>"
                f"<td>{index + 1}</td>"
                f"<td>{html.escape(str(selection.get('lens_id', '--')))}</td>"
                f"<td>{html.escape(str(selection.get('sensor_id', '--')))}</td>"
                f"<td>{html.escape(str(compatibility.get('status', '--')))}</td>"
                f"<td>{number(hfov, 2)}</td>"
                f"<td>{number(score.get('target_score'))}</td>"
                f"<td>{number(artifact_metrics.get('rgb_high_clip_fraction'))}</td>"
                f"<td>{html.escape(str(fidelity))}</td>"
                "</tr>"
            )

        module_rows = "".join(
            module_row(index, item)
            for index, item in enumerate(module_comparison.get("evaluations", []))
        )
        module_boundary = html.escape(str(module_comparison.get("truth_boundary", "Not run")))
        requirement_json = html.escape(json.dumps(payload.get("requirements", {}), indent=2))
        requirement_result = html.escape(
            json.dumps(payload.get("requirement_evaluation", {}), indent=2)
        )
        preflight_json = html.escape(json.dumps(payload.get("benchmark_preflight", {}), indent=2))
        dataset_json = html.escape(json.dumps(payload.get("dataset", {}), indent=2))
        validation_json = html.escape(json.dumps(payload.get("candidate_validation", {}), indent=2))
        boundary = html.escape(str(payload.get("claim_boundary", "")))
        decision_json = html.escape(
            json.dumps((payload.get("optimization") or {}).get("decision", {}), indent=2)
        )
        calibration_json = html.escape(json.dumps(payload.get("calibration_pack", {}), indent=2))
        title = html.escape(str(payload["title"]))
        reproduce = html.escape(json.dumps(payload.get("reproduce", {}), indent=2))
        style = (
            "<style>"
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "margin:32px;color:#18222d}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccd5df;padding:8px;text-align:left;vertical-align:top}"
            "th{background:#eef3f5}"
            "code,pre{background:#f5f7f8;padding:8px;white-space:pre-wrap}"
            ".boundary{border-left:4px solid #b7791f;padding:12px;background:#fff8e7}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}"
            "</style>"
        )
        return (
            '<!doctype html><html><head><meta charset="utf-8">'
            f"<title>{title}</title>{style}</head><body>"
            f'<h1>{title}</h1><p class="boundary">{boundary}</p>'
            f'<div class="grid"><section><h2>Decision status</h2><pre>{decision_json}</pre>'
            f"</section><section><h2>Calibration pack</h2><pre>{calibration_json}</pre>"
            "</section></div>"
            f'<div class="grid"><section><h2>Requirements</h2><pre>{requirement_json}</pre>'
            f"</section><section><h2>Requirement gates</h2><pre>{requirement_result}</pre>"
            f"</section></div><h2>ADAS benchmark preflight</h2><pre>{preflight_json}</pre>"
            f'<h2>Camera module comparison</h2><p class="boundary">{module_boundary}</p>'
            "<table><thead><tr><th>Module</th><th>Lens</th><th>Sensor</th>"
            "<th>Compatibility</th><th>HFOV</th><th>Target</th><th>Clip</th>"
            f"<th>Fidelity</th></tr></thead><tbody>{module_rows}</tbody></table>"
            "<h2>Top candidates</h2><table><thead><tr>"
            "<th>Case</th><th>Score</th><th>Confidence interval</th><th>mAP</th>"
            "<th>Recall</th><th>Scenes</th>"
            "<th>Parameters</th><th>Feasible</th><th>Evidence</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            f'<div class="grid"><section><h2>Candidate evidence</h2><pre>{validation_json}</pre>'
            f"</section><section><h2>RAW dataset</h2><pre>{dataset_json}</pre></section></div>"
            f"<h2>Reproduce</h2><pre>{reproduce}</pre></body></html>"
        )

    @staticmethod
    def _format_score_interval(item: dict[str, Any]) -> str:
        uncertainty = item.get("uncertainty") or {}
        low = uncertainty.get("ci_low")
        high = uncertainty.get("ci_high")
        if low is None or high is None:
            return "unavailable"
        return f"[{float(low):.5f}, {float(high):.5f}]"

    def _forget_future(self, job_id: str) -> None:
        with self._future_lock:
            self._futures.pop(job_id, None)
