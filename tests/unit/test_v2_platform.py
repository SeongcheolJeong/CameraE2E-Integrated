from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from pydantic import ValidationError

from camerae2e_v2.calibration import fit_calibration
from camerae2e_v2.color import fit_constrained_ccm
from camerae2e_v2.engine import CameraEngine
from camerae2e_v2.evaluation import StudyEvaluator, load_adas_label_payload
from camerae2e_v2.geometry import derive_geometry_transform
from camerae2e_v2.migration import migrate_v1_settings
from camerae2e_v2.models import (
    AssetKind,
    CalibrationRequest,
    CameraAssetRecord,
    DesignVariable,
    FidelityLevel,
    JobRecord,
    JobStatus,
    StudyCreate,
)
from camerae2e_v2.project import Project, ProjectManager
from camerae2e_v2.service import CameraE2EService
from pyisetcam import TaskBoundingBox


def test_v2_schema_rejects_invalid_units_and_domains() -> None:
    with pytest.raises(ValidationError):
        DesignVariable(path="sensor.pixel_size", minimum=4.0, maximum=2.0)
    with pytest.raises(ValidationError):
        StudyCreate(search_budget=0)


def test_v2_project_persists_study_and_recovers_running_job(tmp_path: Path) -> None:
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create("Persistent Study")
    study = project.create_study(StudyCreate())
    running = JobRecord(
        project_id=project.info.id,
        study_id=study.id,
        kind="evaluate",
        status=JobStatus.RUNNING,
    )
    project.store.put_job(running)

    reopened_manager = ProjectManager(tmp_path / "projects")
    reopened = reopened_manager.open(project.info.id)

    assert reopened.store.get_study(study.id).spec.name == study.spec.name
    recovered = reopened.store.get_job(running.id)
    assert recovered.status == JobStatus.INTERRUPTED
    assert recovered.finished_at is not None


def test_v2_artifact_registry_is_content_addressed_and_detects_missing_file(
    tmp_path: Path,
) -> None:
    project = Project.create(tmp_path / "project", name="Artifacts")
    first = project.artifacts.put_json(
        {"value": 1},
        artifact_type="test",
        fidelity_level=FidelityLevel.ANALYTIC,
        readiness_tier="validated",
        source="test",
    )
    second = project.artifacts.put_json(
        {"value": 1},
        artifact_type="test",
        fidelity_level=FidelityLevel.ANALYTIC,
        readiness_tier="validated",
        source="test",
    )
    assert first.hash == second.hash
    dependency = project.artifacts.put_json(
        {"input": 1},
        artifact_type="input",
        fidelity_level=FidelityLevel.ANALYTIC,
        readiness_tier="validated",
        source="test",
    )
    merged = project.artifacts.put_json(
        {"value": 1},
        artifact_type="test",
        fidelity_level=FidelityLevel.ANALYTIC,
        readiness_tier="validated",
        source="test-second-lineage",
        dependencies=[dependency.hash],
    )
    assert merged.dependencies == [dependency.hash]
    assert "test-second-lineage" in merged.metadata["alternate_sources"]
    asset = CameraAssetRecord(
        kind=AssetKind.OPTICAL_LUT,
        name="Test LUT",
        artifact_hashes=[first.hash],
        fidelity_level=FidelityLevel.LUT,
        readiness_tier="proxy",
        source="test",
    )
    project.store.put_camera_asset(asset)
    assert project.store.list_camera_assets(kind="optical_lut")[0].id == asset.id
    project.artifacts.resolve(first.hash).unlink()
    validation = project.artifacts.validate()
    assert not validation["ok"]
    assert validation["issues"][0]["kind"] == "missing_file"


def test_v2_engine_runs_real_pipeline_and_persists_preview(tmp_path: Path) -> None:
    service = CameraE2EService(tmp_path / "projects", max_workers=1)
    project = service.create_project("Baseline", preset="general")
    study = project.store.list_studies()[0]
    assert len(project.store.list_camera_assets()) == 6

    job = service.execute_job_now(project.info.id, study.id, "evaluate", {})

    assert job.status == JobStatus.SUCCEEDED
    assert job.result is not None
    assert job.result["schema_version"] == "camerae2e_evaluation_v2"
    assert job.result["preview_artifact_hash"]
    assert job.result["metrics"]["color"]["rgb_mean"] > 0
    assert project.artifacts.validate()["ok"]
    service.shutdown()


def test_v2_perception_target_fails_without_model_instead_of_using_proxy(
    tmp_path: Path,
) -> None:
    service = CameraE2EService(tmp_path / "projects", max_workers=1)
    project = service.create_project("No Detector", preset="general")
    study = project.store.list_studies()[0]
    spec = study.spec.model_copy(
        update={"target_profile": "adas_yolo_perception", "perception_model_path": None}
    )
    study = project.update_study(study.id, spec)

    job = service.execute_job_now(project.info.id, study.id, "evaluate", {})

    assert job.status == JobStatus.FAILED
    assert "requires an existing KITTI/ADAS YOLO model" in str(job.error)
    assert job.result is None
    service.shutdown()


def test_v2_label_loader_supports_yolo_and_kitti_rows(tmp_path: Path) -> None:
    yolo = tmp_path / "yolo.txt"
    yolo.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    kitti = tmp_path / "kitti.txt"
    kitti.write_text(
        "Car 0.00 0 0.00 10 20 50 80 1.5 1.6 4.0 0 0 20 0\n",
        encoding="utf-8",
    )

    yolo_payload = load_adas_label_payload(yolo, image_size=(100, 200))
    kitti_payload = load_adas_label_payload(kitti, image_size=(100, 200))

    assert yolo_payload["source_format"] == "yolo_xywhn"
    assert yolo_payload["objects"][0]["bbox_xyxy"] == pytest.approx([80, 30, 120, 70])
    assert kitti_payload["source_format"] == "kitti_object"
    assert kitti_payload["objects"][0]["label"] == "Car"


def test_v2_trade_study_dataset_solver_evidence_and_report(tmp_path: Path) -> None:
    service = CameraE2EService(tmp_path / "projects", max_workers=1)
    project = service.create_project("Trade Study", preset="general")
    study = project.store.list_studies()[0]
    study = project.update_study(
        study.id,
        study.spec.model_copy(
            update={
                "search_budget": 2,
                "design_variables": study.spec.design_variables[:2],
            }
        ),
    )

    optimization = service.execute_job_now(project.info.id, study.id, "optimize", {})
    validation = service.execute_job_now(
        project.info.id,
        study.id,
        "validate_candidate",
        {"execute": False, "families": ["rayoptics"]},
    )
    dataset = service.execute_job_now(
        project.info.id,
        study.id,
        "dataset_export",
        {"selection": "best", "case_count": 1},
    )
    report = service.execute_job_now(project.info.id, study.id, "report", {})

    assert optimization.status == JobStatus.SUCCEEDED
    assert optimization.result and optimization.result["case_count"] == 2
    assert validation.status == JobStatus.SUCCEEDED
    assert validation.result and validation.result["solver_results"][0]["family"] == "rayoptics"
    assert dataset.status == JobStatus.SUCCEEDED
    assert dataset.result and dataset.result["validation"]["ok"]
    raw_files = sorted(Path(dataset.result["dataset_root"]).glob("raw/*.npz"))
    assert raw_files
    with np.load(raw_files[0]) as payload:
        assert payload["raw"].dtype == np.float32
    assert report.status == JobStatus.SUCCEEDED
    assert report.result and report.result["html_artifact"]["hash"]
    service.shutdown()


def test_v2_calibration_fit_is_scoped_and_lineaged(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", name="Calibration")
    simulated_path = tmp_path / "simulated.npy"
    measured_path = tmp_path / "measured.npy"
    simulated = np.linspace(0.0, 1.0, 10)
    np.save(simulated_path, simulated)
    np.save(measured_path, 1.2 * simulated + 0.03)

    result = fit_calibration(
        project,
        CalibrationRequest(
            kind="qe",
            measured_path=str(measured_path),
            simulated_path=str(simulated_path),
        ),
    )

    assert result["model"]["gain"] == pytest.approx(1.2)
    assert result["model"]["offset"] == pytest.approx(0.03)
    assert result["residual"]["rmse"] < 1e-12
    assert len(result["artifact"]["dependencies"]) == 2
    assert "does not promote" in result["promotion_scope"]


def test_v2_migrates_v1_settings_without_inventing_perception_score(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", name="Migration")
    source = tmp_path / "settings.json"
    source.write_text(
        json.dumps(
            {
                "settings": {
                    "goal": "ADAS RAW Factory",
                    "pixelSizeUm": 2.8,
                    "fNumber": 2.0,
                    "targetProfile": "adas_yolo_perception",
                    "yoloModelPath": str(tmp_path / "missing.pt"),
                }
            }
        ),
        encoding="utf-8",
    )

    result = migrate_v1_settings(project, source)

    assert result["study"]["spec"]["baseline"]["sensor"]["pixel_size_um"] == 2.8
    assert result["study"]["spec"]["target_profile"] == "raw_quality"
    assert result["perception_preserved"] is False
    assert result["warnings"]


def test_v2_free_ccm_requires_calibration_evidence() -> None:
    spec = StudyCreate()
    module = spec.baseline.model_copy(
        update={
            "isp": spec.baseline.isp.model_copy(
                update={"ccm_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
            )
        }
    )
    engine = CameraEngine()
    decision = engine.router.resolve(FidelityLevel.ANALYTIC, spec.fidelity_policy, module)
    with pytest.raises(ValueError, match="color calibration evidence"):
        engine.to_scenario(module, decision)


def test_v2_engine_enforces_sensor_geometry_and_seed(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.full((33, 65, 3), 128, dtype=np.uint8)
    iio.imwrite(image_path, image)
    spec = StudyCreate()
    module = spec.baseline.model_copy(
        update={
            "sensor": spec.baseline.sensor.model_copy(
                update={"rows": 32, "cols": 64, "qe_profile": "sony_imx363"}
            )
        }
    )
    scene = spec.scenes[0].model_copy(
        update={
            "scene_type": "rgb_file",
            "source_kind": "display_rgb_proxy",
            "image_path": str(image_path),
        }
    )
    engine = CameraEngine()
    first = engine.evaluate(
        module,
        scene,
        fidelity=FidelityLevel.ANALYTIC,
        policy=spec.fidelity_policy,
        seed=11,
    )
    repeated = engine.evaluate(
        module,
        scene,
        fidelity=FidelityLevel.ANALYTIC,
        policy=spec.fidelity_policy,
        seed=11,
    )
    changed = engine.evaluate(
        module,
        scene,
        fidelity=FidelityLevel.ANALYTIC,
        policy=spec.fidelity_policy,
        seed=12,
    )

    transform = first["geometry"]["transform"]
    assert transform["active_sensor_size_rc"] == [32, 64]
    assert transform["output_size_rc"] == [32, 64]
    assert first["geometry"]["metrics"]["hfov_error_deg"] < 1e-9
    assert first["scenario"]["camerae2e_v2"]["qe_profile"] == "sony_imx363"
    first_raw = first["stages"]["sensor_raw"]["array"]
    repeated_raw = repeated["stages"]["sensor_raw"]["array"]
    changed_raw = changed["stages"]["sensor_raw"]["array"]
    assert np.array_equal(first_raw, repeated_raw)
    assert not np.array_equal(first_raw, changed_raw)


class _NormalizedBoxDetector:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    def detect(self, image: np.ndarray, _config: object) -> list[TaskBoundingBox]:
        if self.empty:
            return []
        rows, cols = image.shape[:2]
        return [
            TaskBoundingBox(
                (0.4 * cols, 0.3 * rows, 0.6 * cols, 0.7 * rows),
                label="car",
                score=0.99,
            )
        ]


class _ColorBiasedEngine:
    def evaluate(
        self,
        module: object,
        scene: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        image = np.zeros((40, 80, 3), dtype=float)
        image[..., 1] = 0.9
        transform = derive_geometry_transform(
            module,
            scene,
            output_size_rc=(40, 80),
            readout_size_rc=(40, 80),
        )
        return {
            "stages": {"ip_srgb": {"array": image}},
            "geometry": {"transform": transform.model_dump(mode="json")},
            "color_diagnostics": {
                "ideal_output_ssim": 0.8,
                "relative_channel_gain_imbalance": 0.7,
            },
            "fidelity": {"effective": "L1_lut", "readiness_tier": "proxy"},
        }


def _perception_study(tmp_path: Path) -> tuple[Project, object, Path]:
    root = tmp_path / "kitti"
    image_root = root / "images" / "train"
    label_root = root / "labels" / "train"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    image_path = image_root / "000001.png"
    iio.imwrite(image_path, np.full((40, 80, 3), 128, dtype=np.uint8))
    label_path = label_root / "000001.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.4\n7 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"fixture")
    project = Project.create(tmp_path / "project", name="Perception")
    base = StudyCreate()
    scene = base.scenes[0].model_copy(
        update={
            "id": "kitti_000001",
            "scene_type": "rgb_file",
            "source_kind": "display_rgb_proxy",
            "image_path": str(image_path),
            "label_path": str(label_path),
        }
    )
    benchmark = base.benchmark.model_copy(
        update={
            "source_root": str(root),
            "quick_scene_count": 1,
            "final_scene_count": 1,
            "halving_scene_counts": (1,),
            "fidelity_ssim_min": 0.0,
        }
    )
    spec = base.model_copy(
        update={
            "target_profile": "adas_yolo_perception",
            "perception_model_path": str(model_path),
            "scenes": [scene],
            "benchmark": benchmark,
        }
    )
    study = project.create_study(spec)
    return project, study, model_path


def test_v2_benchmark_preflight_filters_non_adas_and_detects_degenerate_target(
    tmp_path: Path,
) -> None:
    _project, study, model_path = _perception_study(tmp_path)
    key = str(model_path.resolve())
    evaluator = StudyEvaluator()
    evaluator._detectors[key] = _NormalizedBoxDetector()

    ready = evaluator.preflight(study, scene_count=1)

    assert ready["ready"]
    assert ready["source_baseline"]["ground_truth_count"] == 1
    assert ready["ideal_recapture"]["max_bbox_transform_error_px"] <= 0.5

    blocked = StudyEvaluator()
    blocked._detectors[key] = _NormalizedBoxDetector(empty=True)
    result = blocked.preflight(study, scene_count=1)
    assert not result["ready"]
    assert result["status"] == "objective_degenerate"
    assert "source_map50" in {item["id"] for item in result["checks"] if not item["pass"]}


def test_v2_preflight_blocks_color_biased_camera_fidelity(tmp_path: Path) -> None:
    _project, study, model_path = _perception_study(tmp_path)
    key = str(model_path.resolve())
    evaluator = StudyEvaluator(engine=_ColorBiasedEngine())
    evaluator._detectors[key] = _NormalizedBoxDetector()

    result = evaluator.preflight(study, scene_count=1)

    assert not result["ready"]
    assert result["status"] == "fidelity_invalid"
    assert result["camera_output"]["recall_retention"] == pytest.approx(1.0)
    assert result["camera_output"]["relative_channel_gain_imbalance"] == pytest.approx(0.7)
    assert "fidelity_color_balance" in {
        item["id"] for item in result["checks"] if not item["pass"]
    }
    assert result["training"] == {}


def test_v2_highlight_gate_does_not_treat_black_pixels_as_saturation(
    tmp_path: Path,
) -> None:
    _project, study, model_path = _perception_study(tmp_path)
    evaluator = StudyEvaluator()
    evaluator._detectors[str(model_path.resolve())] = _NormalizedBoxDetector()
    image = np.full((40, 80, 3), 0.5, dtype=float)
    image[:, :20] = 0.0
    result = {
        "stages": {
            "ip_srgb": {"array": image},
            "sensor_raw": {"array": np.linspace(0.1, 0.9, 40 * 80).reshape(40, 80)},
        },
        "metrics": {"artifact": {}, "color": {}},
    }

    score = evaluator.score_result(study, result, scene=study.spec.scenes[0])
    gates = {item["id"]: item for item in score["hard_gates"]}

    assert gates["rgb_high_clip_fraction"]["pass"]
    assert gates["rgb_high_clip_fraction"]["value"] == 0.0
    assert gates["rgb_low_clip_diagnostic"]["value"] == pytest.approx(0.25)
    assert score["feasible"]


class _HalvingEvaluator(StudyEvaluator):
    def preflight(self, _study: object, *, scene_count: int | None = None) -> dict[str, object]:
        return {"ready": True, "scene_count": scene_count or 1, "checks": []}

    def _evaluate_candidate_scenes(
        self,
        _study: object,
        parameters: dict[str, object],
        scenes: list[object],
        *,
        case_index: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        scores = {0: {1: 0.9, 2: 0.6, 3: 0.5}, 1: {1: 0.8}}
        score = scores[case_index][len(scenes)]
        return {
            "case_id": f"case_{case_index:04d}",
            "case_index": case_index,
            "parameters": parameters,
            "target_score": score,
            "feasible": True,
            "score_components": {},
            "constraint_results": [],
            "hard_gates": [],
            "perception_metrics": {"map50_95": score, "recall50": score, "prediction_count": 1},
            "metrics": {"artifact": {"rgb_high_clip_fraction": 0.0}},
            "scene_count": len(scenes),
            "scene_metrics": [],
            "robustness_cases": [],
            "evidence_state": "analytic_screened",
            "fidelity": {"effective": "L0_analytic"},
            "truth_boundary": "test",
        }


def test_v2_successive_halving_selects_best_only_from_finalists(tmp_path: Path) -> None:
    project, source_study, _model_path = _perception_study(tmp_path)
    root = Path(source_study.spec.benchmark.source_root or "")
    for index in (2, 3):
        iio.imwrite(
            root / "images/train" / f"{index:06d}.png",
            np.full((40, 80, 3), 128, dtype=np.uint8),
        )
        (root / "labels/train" / f"{index:06d}.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
        )
    benchmark = source_study.spec.benchmark.model_copy(
        update={
            "quick_scene_count": 1,
            "final_scene_count": 3,
            "halving_scene_counts": (1, 2),
            "final_candidate_count": 1,
        }
    )
    spec = source_study.spec.model_copy(
        update={
            "benchmark": benchmark,
            "search_budget": 2,
            "search_method": "grid",
            "design_variables": [
                DesignVariable(path="sensor.integration_time", unit="s", values=[0.002, 0.004])
            ],
        }
    )
    study = project.create_study(spec)

    result = _HalvingEvaluator().optimize(study)

    assert result["best_case"]["case_id"] == "case_0000"
    assert result["best_case"]["scene_count"] == 3
    assert result["top_cases"][0]["finalist"] is True
    assert result["top_cases"][1]["finalist"] is False
    assert result["pareto_front"][0]["case_id"] == "case_0000"


def test_v2_constrained_ccm_uses_holdout_and_preserves_neutral() -> None:
    rng = np.random.default_rng(9)
    sensor_rgb = rng.uniform(0.05, 0.9, size=(48, 3))
    expected = np.array([[1.08, -0.04, -0.04], [-0.03, 1.06, -0.03], [-0.02, -0.05, 1.07]])
    target_rgb = sensor_rgb @ expected

    result = fit_constrained_ccm(sensor_rgb, target_rgb, seed=3)

    matrix = np.asarray(result["matrix"])
    assert result["validated"]
    assert result["holdout"]["rmse"] < result["identity_holdout"]["rmse"]
    assert np.max(np.abs(np.sum(matrix, axis=0) - 1.0)) <= 0.15
