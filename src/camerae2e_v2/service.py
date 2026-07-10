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

from .benchmark import benchmark_inventory, benchmark_scenes
from .calibration import fit_calibration
from .catalog import seed_builtin_camera_assets
from .engine import CameraEngine
from .evaluation import StudyEvaluator, load_adas_label_payload
from .geometry import transform_label_payload
from .models import (
    CalibrationRequest,
    DatasetExportRequest,
    FidelityLevel,
    GeometryTransform,
    JobRecord,
    JobStatus,
    ReadinessTier,
    ReportRequest,
    SceneCase,
    StudyCreate,
    utc_now,
)
from .project import Project, ProjectManager
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
        }
        if job.kind not in handlers:
            raise ValueError(f"Unsupported CameraE2E v2 job kind: {job.kind}")
        return handlers[job.kind](project, job)

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
            job.request.get("base_model")
            or study.spec.perception_model_path
            or "yolo11n.pt"
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
        scene_count = int(
            request.scene_count
            or (
                study.spec.benchmark.final_scene_count
                if study.spec.target_profile == "adas_yolo_perception"
                else len(study.spec.scenes)
            )
        )
        scenes = (
            benchmark_scenes(study, scene_count)
            if study.spec.target_profile == "adas_yolo_perception"
            else list(study.spec.scenes[:scene_count])
        )
        if len(scenes) < scene_count:
            raise ValueError(
                f"Dataset export requested {scene_count} scenes but only "
                f"{len(scenes)} are available"
            )
        output_dir = project.root / "exports" / f"dataset_{job.id}"
        for directory in ("raw", "rgb", "labels", "raw_tiff", "stages"):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
        metadata_rows: list[dict[str, Any]] = []
        split_counts = {"train": 0, "validation": 0, "test": 0}
        for candidate_index, candidate in enumerate(candidates):
            parameters = dict(candidate.get("parameters", candidate))
            case_id = str(candidate.get("case_id", f"candidate_{candidate_index:03d}"))
            for scene_index, scene_case in enumerate(scenes):
                sample_id = f"{case_id}_{scene_case.id}"
                seed = study.spec.seed + candidate_index * 100000 + scene_index
                result = self.engine.evaluate(
                    study.spec.baseline,
                    scene_case,
                    fidelity=study.spec.fidelity_policy.search_level,
                    policy=study.spec.fidelity_policy,
                    seed=seed,
                    parameter_overrides=parameters,
                    include_arrays=True,
                )
                raw = self._result_stage_array(result, "sensor_raw", "sensor_digital")
                rgb = self._result_stage_array(result, "ip_srgb", "ip_result")
                if raw is None or rgb is None:
                    raise ValueError(f"Simulation did not produce RAW/RGB for {sample_id}")
                raw_export = np.asarray(raw, dtype=np.float32)
                raw_path = output_dir / "raw" / f"{sample_id}.npz"
                np.savez_compressed(raw_path, raw=raw_export)
                rgb_path = output_dir / "rgb" / f"{sample_id}.png"
                rgb_bytes = self._png_bytes(rgb)
                if rgb_bytes is None:
                    raise ValueError(f"Could not encode RGB preview for {sample_id}")
                rgb_path.write_bytes(rgb_bytes)
                label_payload: dict[str, Any] = {
                    "schema_version": "camerae2e_dataset_labels_v2",
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
                label_path = output_dir / "labels" / f"{sample_id}.json"
                label_path.write_text(
                    json.dumps(label_payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                if request.include_tiff:
                    tiff_path = output_dir / "raw_tiff" / f"{sample_id}.tiff"
                    normalized = np.asarray(raw_export, dtype=float)
                    peak = max(float(np.max(normalized)), 1e-12)
                    iio.imwrite(
                        tiff_path,
                        np.clip(normalized / peak * 65535.0, 0, 65535).astype(np.uint16),
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
                split_counts[split] += 1
                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": case_id,
                        "scene_id": scene_case.id,
                        "group_id": scene_case.metadata.get("group_id", scene_case.id),
                        "split": split,
                        "seed": seed,
                        "source_kind": scene_case.source_kind,
                        "source_image": scene_case.image_path,
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
                        "rgb": str(rgb_path.relative_to(output_dir)),
                        "labels": str(label_path.relative_to(output_dir)),
                        "raw_shape": list(raw_export.shape),
                        "raw_dtype": str(raw_export.dtype),
                        "camera_config": result.get("module"),
                        "geometry": result.get("geometry"),
                        "fidelity": result.get("fidelity"),
                        "metrics": result.get("metrics"),
                        "truth_boundary": result.get("truth_boundary"),
                    }
                )
        metadata_path = output_dir / "metadata.jsonl"
        metadata_path.write_text(
            "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in metadata_rows),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "camerae2e_raw_dataset_v2",
            "study_id": study.id,
            "selection": request.selection,
            "candidate_count": len(candidates),
            "scene_count": len(scenes),
            "case_count": len(metadata_rows),
            "seed": study.spec.seed,
            "formats": ["raw_npz", "rgb_png", "labels_json", "metadata_jsonl"],
            "split_counts": split_counts,
            "source_kinds": sorted({scene.source_kind for scene in scenes}),
            "label_policy": "caller_or_dataset_provided_only_no_automatic_inference",
            "dng_included": False,
            "truth_boundary": (
                "RAW values are simulated from the recorded camera configuration. Source KITTI "
                "RGB frames are display-derived scene proxies and are not original KITTI RAW."
            ),
        }
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
                fidelity_level=study.spec.fidelity_policy.search_level,
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
            "schema_version": "camerae2e_dataset_export_v2",
            "dataset_root": str(output_dir),
            "selection": request.selection,
            "case_count": manifest.get("case_count", 0),
            "candidate_count": len(candidates),
            "scene_count": len(scenes),
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
        payload = {
            "schema_version": "camerae2e_decision_report_v2",
            "title": request.title or f"{study.spec.name} Decision Report",
            "project": project.info.model_dump(mode="json"),
            "study": study.model_dump(mode="json"),
            "requirements": study.spec.requirements.model_dump(mode="json"),
            "baseline": latest.get("evaluate"),
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
            "fidelity_policy": study.spec.fidelity_policy.model_dump(mode="json"),
            "artifact_validation": project.artifacts.validate(),
            "reproduce": {
                "project": str(project.root),
                "study_id": study.id,
                "command": f"camerae2e run {project.root} {study.id} report",
            },
            "claim_boundary": (
                "This research decision report does not assert product sign-off. L3 claims "
                "require measured calibration evidence for every contributing stage."
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
            return [{} for _ in range(request.case_count)]
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
        group_id = str(scene.metadata.get("group_id", scene.id))
        bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 80:
            return "train"
        if bucket < 90:
            return "validation"
        return "test"

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_dataset_export(
        root: Path,
        manifest: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        group_splits: dict[str, str] = {}
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
        if len(rows) != int(manifest.get("case_count", -1)):
            issues.append(
                {
                    "kind": "manifest_count_mismatch",
                    "manifest": manifest.get("case_count"),
                    "actual": len(rows),
                }
            )
        return {
            "ok": not issues,
            "sample_count": len(rows),
            "group_count": len(group_splits),
            "issue_count": len(issues),
            "issues": issues,
        }

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
    def _render_report_html(payload: dict[str, Any]) -> str:
        optimization = payload.get("optimization") or {}
        candidates = optimization.get("top_cases", [])
        rows = "".join(
            "<tr>"
            f"<td>{item.get('case_index')}</td>"
            f"<td>{float(item.get('target_score', 0.0)):.5f}</td>"
            f"<td>{float((item.get('perception_metrics') or {}).get('map50_95', 0.0)):.4f}</td>"
            f"<td>{float((item.get('perception_metrics') or {}).get('recall50', 0.0)):.4f}</td>"
            f"<td>{int(item.get('scene_count', 0))}</td>"
            f"<td>{html.escape(json.dumps(item.get('parameters', {}), sort_keys=True))}</td>"
            f"<td>{'yes' if item.get('feasible') else 'no'}</td>"
            f"<td>{html.escape(str(item.get('evidence_state', 'analytic_screened')))}</td>"
            "</tr>"
            for item in candidates
        )
        requirement_json = html.escape(json.dumps(payload.get("requirements", {}), indent=2))
        requirement_result = html.escape(
            json.dumps(payload.get("requirement_evaluation", {}), indent=2)
        )
        preflight_json = html.escape(
            json.dumps(payload.get("benchmark_preflight", {}), indent=2)
        )
        dataset_json = html.escape(json.dumps(payload.get("dataset", {}), indent=2))
        validation_json = html.escape(
            json.dumps(payload.get("candidate_validation", {}), indent=2)
        )
        boundary = html.escape(str(payload.get("claim_boundary", "")))
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
            f'<div class="grid"><section><h2>Requirements</h2><pre>{requirement_json}</pre>'
            f"</section><section><h2>Requirement gates</h2><pre>{requirement_result}</pre>"
            f"</section></div><h2>ADAS benchmark preflight</h2><pre>{preflight_json}</pre>"
            "<h2>Top candidates</h2><table><thead><tr>"
            "<th>Case</th><th>Score</th><th>mAP</th><th>Recall</th><th>Scenes</th>"
            "<th>Parameters</th><th>Feasible</th><th>Evidence</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            f'<div class="grid"><section><h2>Candidate evidence</h2><pre>{validation_json}</pre>'
            f"</section><section><h2>RAW dataset</h2><pre>{dataset_json}</pre></section></div>"
            f"<h2>Reproduce</h2><pre>{reproduce}</pre></body></html>"
        )

    def _forget_future(self, job_id: str) -> None:
        with self._future_lock:
            self._futures.pop(job_id, None)
