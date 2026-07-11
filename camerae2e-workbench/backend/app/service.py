from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyisetcam import (  # noqa: E402
    AssetStore,
    TaskBoundingBox,
    camerae2e_adas_camera_spec,
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_from_optimization,
    camerae2e_dataset_validate,
    camerae2e_db_manifest,
    camerae2e_db_validate,
    camerae2e_faca_report,
    camerae2e_kitti_yolo_labels,
    camerae2e_optimize_camera_parameters,
    camerae2e_parameter_candidate_plan,
    camerae2e_parameter_space_catalog,
    camerae2e_run_scenario,
    detection_metrics,
    fdtd_sensor_default_lut_path,
    mean_average_precision,
    task_model_from_config,
    task_perception_config,
    tcad_sensor_db_load,
    tcad_sensor_default_paths,
)
from pyisetcam.dataset import _adas_scene_from_rgb, _synthetic_adas_rgb_and_labels  # noqa: E402

_ADAS_CLASS_GROUPS = {
    "bus": "vehicle",
    "car": "vehicle",
    "person": "pedestrian",
    "van": "vehicle",
    "truck": "vehicle",
    "tram": "vehicle",
    "pedestrian": "pedestrian",
    "person_sitting": "pedestrian",
    "person sitting": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "cyclist": "cyclist",
}
_ADAS_GROUPS = ("vehicle", "pedestrian", "cyclist")
_MAP50_95_THRESHOLDS = tuple(float(f"{value:.2f}") for value in np.arange(0.50, 1.0, 0.05))
_PERCEPTION_SCORE_WEIGHTS = {
    "map50_95": 0.45,
    "recall50": 0.20,
    "small_object_recall50": 0.10,
    "localization_iou": 0.10,
    "robustness": 0.10,
    "image_quality_support": 0.05,
}
_WORKBENCH_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_KITTI_ROOT = Path(
    os.environ.get("CAMERAE2E_KITTI_ROOT", str(_WORKBENCH_ROOT / "data/kitti"))
).expanduser()
_LOCAL_KITTI_YOLO_CHECKPOINTS = (
    _WORKBENCH_ROOT / "runs/yolo_training/checkpoints/kitti_yolo11n_epoch4_best.pt",
    _WORKBENCH_ROOT / "runs/yolo_training/kitti_yolo11n_e1_workbench/weights/best.pt",
)
_LOCAL_COCO_YOLO_FALLBACK = _WORKBENCH_ROOT / "yolo11n.pt"
_KITTI_ID_TO_LABEL = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc",
    -1: "DontCare",
}


class WorkbenchService:
    """Thin local-first wrapper around the real CameraE2E APIs."""

    def __init__(self) -> None:
        self.run_root = REPO_ROOT / "camerae2e-workbench" / "runs"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.latest_scenario_result: dict[str, Any] | None = None
        self.latest_scenario_report: dict[str, Any] | None = None
        self.latest_optimization: dict[str, Any] | None = None
        self.latest_dataset: dict[str, Any] | None = None
        self.latest_report: dict[str, Any] | None = None
        self._tcad_db: Any | None = None

    def presets(self) -> dict[str, Any]:
        default_settings = self.default_settings()
        catalog = camerae2e_parameter_space_catalog()
        return {
            "schema_version": "camerae2e_workbench_presets_v1",
            "default_goal": "ADAS RAW Factory",
            "default_settings": default_settings,
            "goals": [
                {
                    "id": "adas_raw_factory",
                    "name": "ADAS RAW Factory",
                    "description": "KITTI-style ADAS camera tuning plus RAW NPZ export.",
                    "objective": "adas_yolo_perception",
                    "dataset_mode": "best_candidate",
                },
                {
                    "id": "mobile_low_light",
                    "name": "Mobile Low Light",
                    "description": "Exposure/noise/blur tuning for compact mobile modules.",
                    "objective": "snr_color",
                    "dataset_mode": "top_candidates",
                },
                {
                    "id": "high_resolution",
                    "name": "High Resolution",
                    "description": "Pixel pitch, CFA, OCL, and PSF-radius tradeoff exploration.",
                    "objective": "artifact_limited_sharpness_proxy",
                    "dataset_mode": "pareto",
                },
            ],
            "camera_presets": [
                self._camera_preset_payload("kitti_yolo_demo", "KITTI-style ADAS"),
                self._camera_preset_payload("wide_fov_adas_demo", "Wide FOV ADAS"),
                self._camera_preset_payload("narrow_fov_adas_demo", "Narrow FOV ADAS"),
            ],
            "optimization": {
                "default_target": "adas_yolo_perception",
                "default_preset": "adas_camera",
                "default_method": "latin_hypercube",
                "default_max_candidates": 24,
                "objective_presets": [
                    {
                        "id": "adas_yolo_perception",
                        "name": "ADAS YOLO Perception",
                        "requires": ["kitti_trained_yolo_model", "kitti_style_labels"],
                        "weights": _PERCEPTION_SCORE_WEIGHTS,
                    },
                    {
                        "id": "raw_quality_proxy",
                        "name": "RAW Quality Proxy",
                        "metrics": _balanced_objective(),
                    },
                    {
                        "id": "low_clip_color",
                        "name": "Color With Clip Guard",
                        "metrics": [
                            {
                                "metric": "metrics.color.rgb_mean",
                                "direction": "maximize",
                                "weight": 1.0,
                            },
                            {
                                "metric": "metrics.artifact.rgb_clip_fraction",
                                "direction": "minimize",
                                "weight": 1.0,
                            },
                        ],
                    },
                ],
                "registered_axes": catalog["axes"],
                "presets": catalog["presets"],
            },
            "truth_boundary": (
                "Workbench runs real CameraE2E simulation APIs, but current assets are "
                "research/proxy/calibration-required unless their DB registry tier says otherwise."
            ),
        }

    def default_settings(self) -> dict[str, Any]:
        spec = camerae2e_adas_camera_spec("kitti_yolo_demo")
        local_kitti = _discover_local_kitti_pair()
        return {
            "goal": "adas_raw_factory",
            "cameraPreset": "kitti_yolo_demo",
            "sceneType": "adas_kitti_synthetic",
            "seed": 42,
            "outputDir": str(self.run_root / "latest"),
            "fovDeg": round(float(spec["optics"]["hfov_deg"]), 2),
            "pixelSizeUm": 3.75,
            "cfaPreset": "bayer_rgb",
            "oclMode": "centered",
            "oclGroupShape": "2x2",
            "oclEqualization": 0.5,
            "exposureMs": 4.0,
            "analogGain": 1.0,
            "fNumber": 1.8,
            "lensPsfRadiusUm": 2.0,
            "fdtdEnabled": False,
            "fdtdMode": "qe+field",
            "fdtdCrosstalkStrength": 0.5,
            "tcadEnabled": False,
            "hwIspEnabled": False,
            "hwIspFrames": 2,
            "binningFactor": 1,
            "demosaicMethod": "bilinear",
            "sensorConversionMethod": "mcc_optimized",
            "targetProfile": "adas_yolo_perception",
            "optimizationPreset": "adas_camera",
            "optimizationMethod": "latin_hypercube",
            "maxCandidates": 24,
            "yoloModelPath": _resolve_yolo_model_path({}) or "",
            "yoloDevice": os.environ.get("CAMERAE2E_YOLO_DEVICE", "cpu"),
            "yoloScoreThreshold": 0.25,
            "labelSource": "local_kitti_yolo" if local_kitti else "synthetic_kitti",
            "kittiImagePath": str(local_kitti["image"]) if local_kitti else "",
            "kittiLabelPath": str(local_kitti["label"]) if local_kitti else "",
            "robustnessCases": 1,
            "datasetCaseCount": 2,
            "datasetSelection": "best",
            "includeTiff": False,
            "includeStageOutputs": False,
        }

    def assets_status(self) -> dict[str, Any]:
        manifest = camerae2e_db_manifest()
        validation = camerae2e_db_validate(strict=False)
        entries = {str(entry["name"]): entry for entry in manifest.get("entries", [])}
        fdtd_path = fdtd_sensor_default_lut_path()
        tcad_paths = tcad_sensor_default_paths()
        lens_entry = entries.get("lens_patents_active") or entries.get("rayoptics_lens_db_v9", {})
        yolo_status = _yolo_model_status(self.default_settings())
        label_status = _kitti_label_status(self.default_settings())
        return {
            "schema_version": "camerae2e_workbench_assets_status_v1",
            "ok": bool(validation.get("ok", False)),
            "validation": _safe_json(validation),
            "assets": {
                "yolo_model": {
                    **yolo_status,
                    "readiness_tier": yolo_status.get("readiness_tier", "external_required"),
                    "badge": yolo_status.get("badge", "KITTI YOLO"),
                    "truth_boundary": yolo_status.get(
                        "truth_boundary",
                        (
                            "ADAS perception optimization requires a caller-supplied "
                            "KITTI-trained YOLO model. CameraE2E does not synthesize "
                            "detector accuracy."
                        ),
                    ),
                },
                "kitti_labels": {
                    **label_status,
                    "readiness_tier": "available" if label_status["available"] else "missing",
                    "badge": "KITTI Labels",
                    "truth_boundary": (
                        "Labels are caller-provided or synthetic KITTI-style demo labels; "
                        "arbitrary scene labels are not inferred."
                    ),
                },
                "analytic": {
                    "available": True,
                    "readiness_tier": "validated",
                    "badge": "Analytic",
                    "truth_boundary": "Fast analytic CameraE2E scene/optics/sensor/IP path.",
                },
                "fdtd_lut": {
                    "available": bool(fdtd_path and Path(fdtd_path).exists()),
                    "readiness_tier": _entry_tier(entries, "fdtd_sensor_lut_active", "proxy"),
                    "badge": "FDTD LUT",
                    "path": None if fdtd_path is None else str(fdtd_path),
                    "stale_reason": _entry_stale(entries, "fdtd_sensor_lut_active"),
                    "truth_boundary": "LUT-backed optical-response proxy, not product sign-off.",
                },
                "rayoptics": {
                    "available": bool(lens_entry.get("path") and Path(lens_entry["path"]).exists()),
                    "readiness_tier": str(lens_entry.get("readiness_tier", "proxy")),
                    "badge": "RayOptics Geometric",
                    "path": lens_entry.get("path"),
                    "truth_boundary": (
                        "Bundled lens DB and geometric PSF metadata; diffraction is "
                        "a separate comparison."
                    ),
                },
                "tcad": {
                    "available": bool(
                        Path(tcad_paths["generation_map_path"]).exists()
                        and Path(tcad_paths["root"]).exists()
                    ),
                    "readiness_tier": _entry_tier(
                        entries, "tcad_sensor_db_active", "calibration_required"
                    ),
                    "badge": "TCAD Requires Calibration",
                    "root": str(tcad_paths["root"]),
                    "generation_map_path": str(tcad_paths["generation_map_path"]),
                    "stale_reason": _entry_stale(entries, "tcad_sensor_db_active"),
                    "truth_boundary": (
                        "Collection-response framework; calibration and lineage are not closed."
                    ),
                },
                "hw_isp": {
                    "available": True,
                    "readiness_tier": _entry_tier(entries, "hwisp_parameter_profiles", "proxy"),
                    "badge": "HW ISP Profile",
                    "truth_boundary": "Simulator profile, not measured latency trace sign-off.",
                },
            },
            "fidelity_badges": self._fidelity_badges(),
        }

    def optimization_targets(self) -> dict[str, Any]:
        settings = self.default_settings()
        perception_ready = self._perception_readiness(settings)
        return {
            "schema_version": "camerae2e_workbench_optimization_targets_v1",
            "default_target": "adas_yolo_perception",
            "targets": [
                {
                    "id": "adas_yolo_perception",
                    "name": "ADAS YOLO Perception",
                    "default": True,
                    "available": bool(perception_ready["ok"]),
                    "status": perception_ready["status"],
                    "required_assets": perception_ready["required_assets"],
                    "weights": _PERCEPTION_SCORE_WEIGHTS,
                    "hard_gates": [
                        "simulation_success",
                        "detector_configured",
                        "labels_present",
                        "rgb_high_clip_fraction <= 0.01",
                        "0.02 <= rgb_mean <= 0.95",
                        "fov_within_preset_bounds",
                    ],
                },
                {
                    "id": "raw_quality_proxy",
                    "name": "RAW Quality Proxy",
                    "default": False,
                    "available": True,
                    "status": "available",
                    "required_assets": [],
                    "metrics": _balanced_objective(),
                },
            ],
        }

    def scene_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._merged_settings(payload.get("settings", payload))
        scene_payload = self._scene_preview_payload(settings)
        return {
            "schema_version": "camerae2e_workbench_scene_preview_v1",
            "settings": _public_settings(settings),
            **scene_payload,
        }

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._merged_settings(payload.get("settings", payload))
        started = time.perf_counter()
        scene_source = self._simulation_scene_source(settings)
        scenario = self._scenario_from_settings(
            settings,
            scene_override=scene_source.get("scene"),
        )
        result = camerae2e_run_scenario(
            scenario,
            seed=int(settings["seed"]),
            include_arrays=True,
        )
        report = camerae2e_faca_report(result)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        preview = _preview_from_result(result)
        response = {
            "schema_version": "camerae2e_workbench_simulation_result_v1",
            "run_id": f"sim-{int(time.time())}",
            "elapsed_ms": elapsed_ms,
            "settings": _public_settings(settings),
            "scene_source": _safe_json(
                {key: value for key, value in scene_source.items() if key != "scene"}
            ),
            "scenario": _safe_json(report.get("scenario", {})),
            "stage_summaries": _stage_summaries(report),
            "metrics": _metric_cards(report.get("metrics", {})),
            "raw_metrics": _safe_json(report.get("metrics", {})),
            "parameter_lineage": _slim_lineage(report.get("parameter_lineage", [])),
            "artifact_lineage": _artifact_summary(report.get("artifact_lineage", {})),
            "preview_png": preview,
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": self._truth_boundary(settings),
        }
        self.latest_scenario_result = result
        self.latest_scenario_report = response
        self._write_json(self.run_root / "latest" / "simulation_result.json", response)
        return response

    def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._merged_settings(payload.get("settings", payload))
        max_cases = int(payload.get("maxCandidates", settings.get("maxCandidates", 24)))
        max_cases = max(1, min(max_cases, 96))
        target_profile = str(
            payload.get("targetProfile", settings.get("targetProfile", "adas_yolo_perception"))
        )
        if target_profile == "adas_yolo_perception":
            return self._optimize_adas_yolo_perception(payload, settings, max_cases=max_cases)
        if target_profile not in {"raw_quality_proxy", "balanced_faca"}:
            raise ValueError(
                "targetProfile must be one of: adas_yolo_perception, raw_quality_proxy."
            )
        started = time.perf_counter()
        scenario = self._scenario_from_settings(settings, include_hw_isp=False)
        axes = self._optimization_axes(settings)
        result = camerae2e_optimize_camera_parameters(
            scenario,
            preset=str(settings.get("optimizationPreset", "raw_factory")),
            parameter_space=axes,
            objective=_balanced_objective(),
            method=str(settings.get("optimizationMethod", "latin_hypercube")),
            max_cases=max_cases,
            seed=int(settings["seed"]),
            top_k=8,
            include_arrays=True,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = {
            "schema_version": "camerae2e_workbench_optimization_result_v1",
            "ok": True,
            "target_profile": "raw_quality_proxy",
            "target_status": "available",
            "run_id": f"opt-{int(time.time())}",
            "elapsed_ms": elapsed_ms,
            "settings": _public_settings(settings),
            "method": result.get("method"),
            "search_method": result.get("search_method"),
            "case_count": result.get("case_count"),
            "feasible_count": result.get("feasible_count"),
            "pareto_case_count": result.get("pareto_case_count"),
            "candidate_plan": _safe_json(result.get("candidate_plan", {})),
            "objective": _safe_json(result.get("objective", {})),
            "parameter_space": _safe_json(result.get("parameter_space", {})),
            "best_case": _case_card(result.get("best_case")),
            "top_cases": [_case_card(item) for item in result.get("top_cases", [])],
            "pareto_points": _pareto_points(
                result.get("pareto_front", []),
                result.get("top_cases", []),
            ),
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": self._truth_boundary(settings),
        }
        self.latest_optimization = result
        self._write_json(self.run_root / "latest" / "optimization_result.json", response)
        return response

    def _optimize_adas_yolo_perception(
        self, payload: dict[str, Any], settings: dict[str, Any], *, max_cases: int
    ) -> dict[str, Any]:
        started = time.perf_counter()
        readiness = self._perception_readiness(settings)
        if not readiness["ok"]:
            return self._optimization_unavailable_response(
                settings,
                status=str(readiness["status"]),
                detail=str(readiness["detail"]),
                started=started,
            )
        try:
            detector = self._make_yolo_detector(settings)
        except (ImportError, ValueError, FileNotFoundError) as exc:
            return self._optimization_unavailable_response(
                settings,
                status="detector_backend_unavailable",
                detail=str(exc),
                started=started,
            )

        adas_scene = self._adas_scene_and_labels(settings)
        if not adas_scene["labels"].get("objects"):
            return self._optimization_unavailable_response(
                settings,
                status="labels_missing",
                detail="KITTI-style labels are required for ADAS YOLO Perception scoring.",
                started=started,
            )

        scenario = self._scenario_from_settings(
            settings,
            include_hw_isp=False,
            scene_override=adas_scene["scene"],
        )
        axes = self._optimization_axes(settings)
        method = str(settings.get("optimizationMethod", "latin_hypercube"))
        seed = int(settings["seed"])
        candidate_plan = camerae2e_parameter_candidate_plan(
            axes,
            method=method,
            max_cases=max_cases,
            seed=seed,
            base_scenario=None,
        )
        cases = [
            self._evaluate_adas_yolo_candidate(
                dict(axis_values),
                case_index=index,
                base_scenario=scenario,
                labels=adas_scene["labels"],
                detector=detector,
                settings=settings,
                seed=seed,
            )
            for index, axis_values in enumerate(candidate_plan.get("candidates", []))
        ]
        ranked = _rank_perception_cases(cases)
        feasible_cases = [case for case in ranked if case.get("feasible", False)]
        pareto = _perception_pareto_cases(feasible_cases)
        result = {
            "schema_version": "camerae2e_parameter_optimization_v1",
            "ok": True,
            "target_profile": "adas_yolo_perception",
            "target_status": "available",
            "method": method,
            "search_method": candidate_plan.get("method", method),
            "seed": seed,
            "case_count": len(cases),
            "feasible_count": len(feasible_cases),
            "pareto_case_count": len(pareto),
            "candidate_plan": _safe_json(candidate_plan),
            "objective": {
                "id": "adas_yolo_perception",
                "name": "ADAS YOLO Perception Score",
                "weights": _PERCEPTION_SCORE_WEIGHTS,
                "model_path": _resolve_yolo_model_path(settings),
                "model_status": readiness["status"],
                "label_source": adas_scene["label_source"],
            },
            "constraints": [
                {"metric": "metrics.artifact.rgb_high_clip_fraction", "max": 0.01},
                {"metric": "metrics.color.rgb_mean", "min": 0.02, "max": 0.95},
            ],
            "parameter_space": _safe_json(axes),
            "cases": cases,
            "best_case": feasible_cases[0] if feasible_cases else (ranked[0] if ranked else None),
            "top_cases": ranked[:8],
            "pareto_front": pareto,
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": _detector_truth_boundary(readiness),
        }
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = {
            "schema_version": "camerae2e_workbench_optimization_result_v1",
            "ok": True,
            "target_profile": "adas_yolo_perception",
            "target_status": readiness["status"],
            "run_id": f"opt-{int(time.time())}",
            "elapsed_ms": elapsed_ms,
            "settings": _public_settings(settings),
            "method": result.get("method"),
            "search_method": result.get("search_method"),
            "case_count": result.get("case_count"),
            "feasible_count": result.get("feasible_count"),
            "pareto_case_count": result.get("pareto_case_count"),
            "candidate_plan": _safe_json(candidate_plan),
            "objective": _safe_json(result.get("objective", {})),
            "parameter_space": _safe_json(axes),
            "best_case": _case_card(result.get("best_case")),
            "top_cases": [_case_card(item) for item in result.get("top_cases", [])],
            "pareto_points": _perception_pareto_points(pareto or ranked[:8]),
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": result["truth_boundary"],
        }
        self.latest_optimization = result
        self._write_json(self.run_root / "latest" / "optimization_result.json", response)
        return response

    def _optimization_unavailable_response(
        self,
        settings: dict[str, Any],
        *,
        status: str,
        detail: str,
        started: float,
    ) -> dict[str, Any]:
        response = {
            "schema_version": "camerae2e_workbench_optimization_result_v1",
            "ok": False,
            "target_profile": "adas_yolo_perception",
            "target_status": status,
            "run_id": f"opt-{int(time.time())}",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "settings": _public_settings(settings),
            "method": str(settings.get("optimizationMethod", "latin_hypercube")),
            "search_method": None,
            "case_count": 0,
            "feasible_count": 0,
            "pareto_case_count": 0,
            "candidate_plan": {},
            "objective": {
                "id": "adas_yolo_perception",
                "name": "ADAS YOLO Perception Score",
                "weights": _PERCEPTION_SCORE_WEIGHTS,
                "unavailable_reason": status,
                "detail": detail,
            },
            "parameter_space": {},
            "best_case": None,
            "top_cases": [],
            "pareto_points": [],
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": (
                "Perception Target Score is not computed without a configured "
                "KITTI-trained YOLO model and KITTI-style labels. No proxy score is "
                "substituted as perception performance."
            ),
        }
        self.latest_optimization = None
        self._write_json(self.run_root / "latest" / "optimization_result.json", response)
        return response

    def _perception_readiness(self, settings: dict[str, Any]) -> dict[str, Any]:
        yolo_status = _yolo_model_status(settings)
        label_status = _kitti_label_status(settings)
        required_assets = [
            {
                "id": "kitti_trained_yolo_model",
                "available": bool(yolo_status["available"]),
                "status": yolo_status["status"],
                "path": yolo_status.get("path"),
                "source": yolo_status.get("source"),
                "badge": yolo_status.get("badge"),
                "readiness_tier": yolo_status.get("readiness_tier"),
            },
            {
                "id": "kitti_style_labels",
                "available": bool(label_status["available"]),
                "status": label_status["status"],
                "source": label_status.get("source"),
            },
        ]
        if not yolo_status["configured"]:
            status = "detector_not_configured"
            detail = "Set CAMERAE2E_YOLO_MODEL or yoloModelPath to a KITTI-trained YOLO checkpoint."
        elif not yolo_status["path_exists"]:
            status = "detector_not_configured"
            detail = f"YOLO checkpoint does not exist: {yolo_status.get('path')}"
        elif not label_status["available"]:
            status = "labels_missing"
            detail = "Set labelSource/kittiLabelPath or use synthetic_kitti labels."
        elif not yolo_status["dependency_available"]:
            status = "detector_backend_unavailable"
            detail = "Python package 'ultralytics' is required for YOLO-first optimization."
        elif yolo_status.get("source") == "local_kitti_finetune":
            status = "available"
            detail = (
                "Using local YOLO11n checkpoint fine-tuned on KITTI. It is enough "
                "to unblock real Workbench perception scoring, but not a final "
                "detector-quality claim."
            )
        elif yolo_status.get("source") == "autodiscovered_coco_demo":
            status = "available_demo_fallback"
            detail = (
                "Using autodiscovered COCO-pretrained YOLO demo fallback. It enables "
                "real detector scoring, but it is not a KITTI-trained detector."
            )
        else:
            status = "available"
            detail = "KITTI-trained YOLO model and KITTI-style labels are configured."
        return {
            "ok": status in {"available", "available_demo_fallback"},
            "status": status,
            "detail": detail,
            "required_assets": required_assets,
        }

    def _make_yolo_detector(self, settings: dict[str, Any]) -> Any:
        model_path = _resolve_yolo_model_path(settings)
        if model_path is None:
            raise FileNotFoundError(
                "Set CAMERAE2E_YOLO_MODEL or yoloModelPath to a KITTI-trained YOLO checkpoint."
            )
        if not Path(model_path).expanduser().exists():
            raise FileNotFoundError(f"YOLO checkpoint does not exist: {model_path}")
        model_source = _yolo_model_source(model_path, settings)
        model_name = (
            "coco_yolo_demo"
            if model_source == "autodiscovered_coco_demo"
            else "kitti_trained_yolo"
        )
        return task_model_from_config(
            {
                "name": model_name,
                "backend": "ultralytics_yolo",
                "task": "detection",
                "model_id": str(model_path),
                "device": str(settings.get("yoloDevice", "cpu")),
                "score_threshold": float(settings.get("yoloScoreThreshold", 0.25)),
                "options": {"inference": {"verbose": False}},
            }
        )

    def _adas_scene_and_labels(self, settings: dict[str, Any]) -> dict[str, Any]:
        spec = _adas_spec_from_settings(settings)
        store = AssetStore.default()
        image_path = str(settings.get("kittiImagePath", "") or "").strip()
        label_path = str(settings.get("kittiLabelPath", "") or "").strip()
        label_source = str(settings.get("labelSource", "synthetic_kitti") or "synthetic_kitti")
        label_key = label_source.strip().lower().replace("-", "_")
        if label_key in {"none", "missing", "not_provided"}:
            rows, cols = _image_size_from_spec(spec)
            rgb, _ = _synthetic_adas_rgb_and_labels(rows, cols, index=0)
            labels = {
                "objects": [],
                "masks": [],
                "source": "not_provided",
                "image_size_rc": [rows, cols],
            }
            return {
                "scene": _adas_scene_from_rgb(rgb, spec, store, "synthetic_kitti_unlabeled"),
                "labels": labels,
                "label_source": "not_provided",
            }

        if image_path:
            rgb = np.asarray(iio.imread(Path(image_path).expanduser()))
            if rgb.ndim < 2:
                raise ValueError(f"KITTI image must be 2D or 3D: {image_path}")
            rows, cols = int(rgb.shape[0]), int(rgb.shape[1])
            effective_label_path = _resolve_label_path(label_path, image_path)
            labels = _load_adas_labels(effective_label_path, image_size=(rows, cols))
            return {
                "scene": _adas_scene_from_rgb(rgb, spec, store, Path(image_path).stem),
                "labels": labels,
                "label_source": str(labels.get("source", "labels_missing")),
            }

        rows, cols = _image_size_from_spec(spec)
        rgb, labels = _synthetic_adas_rgb_and_labels(rows, cols, index=0)
        if label_path:
            labels = _load_adas_labels(label_path, image_size=(rows, cols))
        return {
            "scene": _adas_scene_from_rgb(rgb, spec, store, "synthetic_kitti_optimization"),
            "labels": labels,
            "label_source": str(label_path or "synthetic_kitti"),
        }

    def _simulation_scene_source(self, settings: dict[str, Any]) -> dict[str, Any]:
        spec = _adas_spec_from_settings(settings)
        image_path = str(settings.get("kittiImagePath", "") or "").strip()
        if image_path:
            path = Path(image_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"KITTI image does not exist: {path}")
            rgb = np.asarray(iio.imread(path))
            if rgb.ndim < 2:
                raise ValueError(f"KITTI image must be 2D or 3D: {path}")
            return {
                "scene": _adas_scene_from_rgb(rgb, spec, AssetStore.default(), path.stem),
                "source": "kitti_file",
                "source_path": str(path),
                "label_source": str(settings.get("kittiLabelPath", "") or "") or None,
                "truth_boundary": (
                    "Simulation uses the caller-provided KITTI RGB as a scene proxy and "
                    "re-captures it through the configured CameraE2E path. It is not KITTI RAW."
                ),
            }

        scene_key = str(settings.get("sceneType", "")).strip().lower().replace("-", "_")
        if scene_key in {"adas_kitti_synthetic", "adas synthetic", "synthetic_kitti", ""}:
            rows, cols = _image_size_from_spec(spec)
            rgb, labels = _synthetic_adas_rgb_and_labels(
                rows,
                cols,
                index=int(settings["seed"]) % 3,
            )
            return {
                "scene": _adas_scene_from_rgb(
                    rgb,
                    spec,
                    AssetStore.default(),
                    "synthetic_kitti_simulation",
                ),
                "source": "synthetic_kitti",
                "source_path": None,
                "label_source": labels.get("source", "synthetic_kitti_style"),
                "truth_boundary": (
                    "Simulation uses the deterministic synthetic ADAS RGB scene because no "
                    "local KITTI image is configured."
                ),
            }

        return {
            "scene": None,
            "source": str(settings.get("sceneType", "macbeth")),
            "source_path": None,
            "label_source": None,
            "truth_boundary": "Simulation uses the selected built-in ISETCam scene.",
        }

    def _evaluate_adas_yolo_candidate(
        self,
        axis_values: dict[str, Any],
        *,
        case_index: int,
        base_scenario: dict[str, Any],
        labels: dict[str, Any],
        detector: Any,
        settings: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        scenario = deepcopy(base_scenario)
        scenario["name"] = f"{base_scenario.get('name', 'workbench_adas_yolo')}_opt{case_index:04d}"
        for key, value in axis_values.items():
            _assign_workbench_parameter(scenario, str(key), value)
        try:
            result = camerae2e_run_scenario(
                scenario,
                seed=int(seed) + int(case_index),
                include_arrays=True,
            )
            report = camerae2e_faca_report(result)
            score_payload = _score_adas_yolo_result(
                result,
                report,
                labels=labels,
                detector=detector,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001 - candidate failures should not abort the whole search.
            report = {"metrics": {}, "scenario": _safe_json(scenario)}
            score_payload = _failed_perception_score(str(exc))

        score = float(score_payload["target_score"]) if score_payload["feasible"] else -math.inf
        objective_values = dict(score_payload["objective_values"])
        return {
            "case_index": int(case_index),
            "seed": int(seed) + int(case_index),
            "parameters": _safe_json(axis_values),
            "score": score,
            "target_score": None if not math.isfinite(score) else score,
            "feasible": bool(score_payload["feasible"]),
            "objective_values": _safe_json(objective_values),
            "objective_utilities": _safe_json(objective_values),
            "constraint_results": _safe_json(score_payload["hard_gates"]),
            "hard_gates": _safe_json(score_payload["hard_gates"]),
            "score_components": _safe_json(score_payload["score_components"]),
            "perception_metrics": _safe_json(score_payload["perception_metrics"]),
            "raw_quality_proxy": _safe_json(score_payload["raw_quality_proxy"]),
            "scenario": scenario,
            "report": report,
        }

    def dataset_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._merged_settings(payload.get("settings", payload))
        selection = str(payload.get("selection", settings.get("datasetSelection", "best")))
        max_cases = int(payload.get("caseCount", settings.get("datasetCaseCount", 2)))
        output_dir = Path(
            payload.get("outputDir")
            or Path(settings["outputDir"]) / f"dataset_{selection}_{int(time.time())}"
        ).expanduser()
        if self.latest_optimization is None:
            manifest = camerae2e_dataset_export_adas_kitti_demo(
                output_dir,
                case_count=max_cases,
                spec=camerae2e_adas_camera_spec(
                    str(settings.get("cameraPreset", "kitti_yolo_demo"))
                ),
                seed=int(settings["seed"]),
                include_rgb=True,
                include_tiff=bool(settings.get("includeTiff", False)),
                include_stage_outputs=bool(settings.get("includeStageOutputs", False)),
            )
            source = "adas_kitti_demo"
        else:
            manifest = camerae2e_dataset_export_from_optimization(
                output_dir,
                self.latest_optimization,
                selection=selection,
                max_cases=max_cases,
                seed=int(settings["seed"]),
                include_rgb=True,
                include_tiff=bool(settings.get("includeTiff", False)),
                include_stage_outputs=bool(settings.get("includeStageOutputs", False)),
            )
            source = "latest_optimization"
        validation = camerae2e_dataset_validate(manifest, strict=False)
        response = {
            "schema_version": "camerae2e_workbench_dataset_export_result_v1",
            "source": source,
            "ok": bool(validation.get("ok", False)),
            "dataset_root": manifest.get("dataset_root"),
            "manifest_path": str(Path(manifest["dataset_root"]) / "manifest.json"),
            "case_count": manifest.get("case_count"),
            "records": [_dataset_record_card(item) for item in manifest.get("records", [])],
            "validation": _safe_json(validation),
            "format": manifest.get("format", {}),
            "truth_boundary": manifest.get("truth_boundary"),
        }
        self.latest_dataset = response
        self._write_json(self.run_root / "latest" / "dataset_export_result.json", response)
        return response

    def report(self, payload: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(
            payload.get("outputDir") or self.run_root / "latest" / f"report_{int(time.time())}"
        ).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "camerae2e_workbench_report_v1",
            "generated_at_unix": int(time.time()),
            "simulation": self.latest_scenario_report,
            "optimization": (
                None
                if self.latest_optimization is None
                else {
                    "case_count": self.latest_optimization.get("case_count"),
                    "feasible_count": self.latest_optimization.get("feasible_count"),
                    "best_case": _case_card(self.latest_optimization.get("best_case")),
                    "top_cases": [
                        _case_card(item) for item in self.latest_optimization.get("top_cases", [])
                    ],
                }
            ),
            "dataset": self.latest_dataset,
            "assets": self.assets_status(),
            "fidelity_boundary": (
                "This report is for research-grade optimization and RAW data generation. "
                "It does not contain product sign-off claims for proxy/calibration-required assets."
            ),
        }
        json_path = output_dir / "camerae2e-workbench-report.json"
        html_path = output_dir / "camerae2e-workbench-report.html"
        json_path.write_text(
            json.dumps(_safe_json(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        html_path.write_text(_html_report(report), encoding="utf-8")
        response = {
            "schema_version": "camerae2e_workbench_report_result_v1",
            "json_path": str(json_path),
            "html_path": str(html_path),
            "report": _safe_json(report),
        }
        self.latest_report = response
        return response

    def _scene_preview_payload(self, settings: dict[str, Any]) -> dict[str, Any]:
        image_path = str(settings.get("kittiImagePath", "") or "").strip()
        if image_path:
            path = Path(image_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"KITTI image does not exist: {path}")
            image = np.asarray(iio.imread(path))
            labels = _load_adas_labels(
                _resolve_label_path(str(settings.get("kittiLabelPath", "") or ""), str(path)),
                image_size=(int(image.shape[0]), int(image.shape[1])),
            )
            return {
                "scene_preview_png": _preview_from_array(image),
                "source": "kitti_file",
                "source_path": str(path),
                "label_source": labels.get("source"),
                "object_count": len(labels.get("objects", [])),
                "truth_boundary": (
                    "Scene preview uses the caller-provided KITTI RGB frame without "
                    "label overlays. Labels remain available for scoring/export, but "
                    "this is the detector input scene proxy, not KITTI RAW sensor data."
                ),
            }

        scene_key = str(settings.get("sceneType", "")).strip().lower().replace("-", "_")
        if scene_key in {"adas_kitti_synthetic", "adas synthetic", "synthetic_kitti", ""}:
            spec = camerae2e_adas_camera_spec(str(settings.get("cameraPreset", "kitti_yolo_demo")))
            rows, cols = _image_size_from_spec(spec)
            image, labels = _synthetic_adas_rgb_and_labels(
                rows,
                cols,
                index=int(settings["seed"]) % 3,
            )
            return {
                "scene_preview_png": _preview_from_array(image),
                "source": "synthetic_kitti",
                "source_path": None,
                "label_source": labels.get("source", "synthetic_kitti_style"),
                "object_count": len(labels.get("objects", [])),
                "truth_boundary": (
                    "No local KITTI RGB frame is configured, so this preview uses the "
                    "deterministic synthetic KITTI-style ADAS scene. It is not real KITTI data."
                ),
            }

        return {
            "scene_preview_png": None,
            "source": "unavailable",
            "source_path": None,
            "label_source": None,
            "object_count": 0,
            "truth_boundary": (
                "Scene preview is currently available for KITTI file or synthetic KITTI "
                "ADAS scene modes."
            ),
        }

    def _scenario_from_settings(
        self,
        settings: dict[str, Any],
        *,
        include_hw_isp: bool | None = None,
        scene_override: Any | None = None,
    ) -> dict[str, Any]:
        pixel_pitch_m = float(settings["pixelSizeUm"]) * 1.0e-6
        focal_length_m = _focal_length_from_hfov(
            hfov_deg=float(settings["fovDeg"]),
            pixel_pitch_m=pixel_pitch_m,
        )
        scene = (
            scene_override
            if scene_override is not None
            else _scene_payload(str(settings.get("sceneType", "macbeth")))
        )
        sensor_payload: dict[str, Any] = {
            "noise_flag": 2,
            "integration_time": float(settings["exposureMs"]) / 1000.0,
            "analog_gain": float(settings.get("analogGain", 1.0)),
            "pixel_size": pixel_pitch_m,
            "cfa_preset": str(settings["cfaPreset"]),
            "ocl_group_shape": str(settings["oclGroupShape"]),
            "ocl_group_equalization": float(settings["oclEqualization"]),
            "binning_factor": int(settings.get("binningFactor", 1)),
        }
        ocl_mode = str(settings["oclMode"]).strip().lower()
        if ocl_mode not in {"off", "none", "disabled", "false", "0"}:
            sensor_payload["ocl_vignetting"] = str(settings["oclMode"])

        scenario: dict[str, Any] = {
            "name": f"workbench_{settings.get('goal', 'adas_raw_factory')}",
            "scene": scene,
            "sensor": sensor_payload,
            "parameters": {
                "optics.fnumber": float(settings["fNumber"]),
                "optics.focal_length": focal_length_m,
                "optics.si_psf_radius_um": float(settings["lensPsfRadiusUm"]),
                "ip.demosaic_method": str(settings.get("demosaicMethod", "bilinear")),
                "ip.sensor_conversion_method": str(
                    settings.get("sensorConversionMethod", "mcc_optimized")
                ),
            },
            "workbench": {
                "fov_deg": float(settings["fovDeg"]),
                "camera_preset": str(settings.get("cameraPreset", "kitti_yolo_demo")),
            },
        }
        if bool(settings.get("fdtdEnabled", False)):
            lut = fdtd_sensor_default_lut_path()
            if lut is None or not Path(lut).exists():
                raise FileNotFoundError("FDTD LUT is enabled but no bundled LUT is available.")
            scenario["fdtd"] = {
                "lut": str(lut),
                "mode": str(settings.get("fdtdMode", "qe+field")),
                "crosstalk_strength": float(settings.get("fdtdCrosstalkStrength", 0.5)),
            }
        if bool(settings.get("tcadEnabled", False)):
            scenario["tcad"] = {
                "db": self._load_tcad_db(),
                "collection_mode": "collection",
            }
        use_hw = (
            bool(settings.get("hwIspEnabled", False))
            if include_hw_isp is None
            else include_hw_isp
        )
        if use_hw:
            scenario["hw_isp"] = {
                "enabled": True,
                "nframes": int(settings.get("hwIspFrames", 2)),
                "ae_apply_delay_frames": 1,
                "awb_apply_delay_frames": 1,
            }
        return scenario

    def _optimization_axes(self, settings: dict[str, Any]) -> dict[str, list[Any]]:
        pixel_um = float(settings["pixelSizeUm"])
        exposure_s = float(settings["exposureMs"]) / 1000.0
        psf = float(settings["lensPsfRadiusUm"])
        axes: dict[str, list[Any]] = {
            "sensor.integration_time": _unique_sorted(
                [exposure_s * 0.5, exposure_s, exposure_s * 1.5]
            ),
            "sensor.analog_gain": _unique_sorted(
                [1.0, float(settings.get("analogGain", 1.0)), 2.0]
            ),
            "sensor.pixel_size": _unique_sorted(
                [(pixel_um * 0.8) * 1e-6, pixel_um * 1e-6, (pixel_um * 1.2) * 1e-6]
            ),
            "sensor.cfa_preset": ["bayer_rgb", "quad_bayer_rgb"],
            "sensor.ocl_vignetting": ["centered", "optimal"],
            "sensor.ocl_group_shape": ["1x1", "2x2"],
            "sensor.ocl_group_equalization": [0.0, 0.5, 1.0],
            "ip.sensor_conversion_method": ["mcc_optimized", "esser_optimized"],
            "optics.fnumber": _unique_sorted(
                [
                    max(1.2, float(settings["fNumber"]) - 0.4),
                    float(settings["fNumber"]),
                    float(settings["fNumber"]) + 0.6,
                ]
            ),
            "optics.si_psf_radius_um": _unique_sorted([max(0.5, psf * 0.5), psf, psf * 1.5]),
        }
        if int(settings.get("binningFactor", 1)) > 1:
            axes["sensor.binning_factor"] = [1, 2]
        if bool(settings.get("fdtdEnabled", False)):
            axes["fdtd.mode"] = [str(settings.get("fdtdMode", "qe+field"))]
            axes["fdtd.crosstalk_strength"] = [
                0.0,
                float(settings.get("fdtdCrosstalkStrength", 0.5)),
            ]
        return axes

    def _load_tcad_db(self) -> Any:
        if self._tcad_db is None:
            self._tcad_db = tcad_sensor_db_load(root=tcad_sensor_default_paths()["root"])
        return self._tcad_db

    def _fidelity_badges(self, settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assets = self.assets_status()["assets"] if settings is not None else None
        if assets is None:
            manifest = camerae2e_db_manifest()
            entries = {str(entry["name"]): entry for entry in manifest.get("entries", [])}
            fdtd_path = fdtd_sensor_default_lut_path()
            assets = {
                "analytic": {"available": True, "readiness_tier": "validated"},
                "fdtd_lut": {
                    "available": bool(fdtd_path and Path(fdtd_path).exists()),
                    "readiness_tier": _entry_tier(entries, "fdtd_sensor_lut_active", "proxy"),
                },
                "rayoptics": {
                    "available": True,
                    "readiness_tier": _entry_tier(entries, "lens_patents_active", "proxy"),
                },
                "tcad": {
                    "available": Path(tcad_sensor_default_paths()["generation_map_path"]).exists(),
                    "readiness_tier": _entry_tier(
                        entries, "tcad_sensor_db_active", "calibration_required"
                    ),
                },
            }
        active_fdtd = bool(settings and settings.get("fdtdEnabled", False))
        active_tcad = bool(settings and settings.get("tcadEnabled", False))
        return [
            {
                "label": "Analytic",
                "active": True,
                "available": True,
                "tier": "validated",
                "tone": "green",
            },
            {
                "label": "FDTD LUT",
                "active": active_fdtd,
                "available": bool(assets["fdtd_lut"]["available"]),
                "tier": assets["fdtd_lut"]["readiness_tier"],
                "tone": "blue",
            },
            {
                "label": "RayOptics Geometric",
                "active": True,
                "available": bool(assets["rayoptics"]["available"]),
                "tier": assets["rayoptics"]["readiness_tier"],
                "tone": "teal",
            },
            {
                "label": "TCAD Requires Calibration",
                "active": active_tcad,
                "available": bool(assets["tcad"]["available"]),
                "tier": assets["tcad"]["readiness_tier"],
                "tone": "amber",
            },
        ]

    def _truth_boundary(self, settings: dict[str, Any]) -> str:
        parts = ["Analytic CameraE2E path is active."]
        if settings.get("fdtdEnabled"):
            parts.append("FDTD LUT is used as LUT-backed optical-response evidence.")
        if settings.get("tcadEnabled"):
            parts.append("TCAD collection response is calibration-required and lineage-limited.")
        parts.append("No output is labeled as product sign-off.")
        return " ".join(parts)

    def _merged_settings(self, raw: dict[str, Any]) -> dict[str, Any]:
        settings = self.default_settings()
        settings.update({key: value for key, value in raw.items() if value is not None})
        return settings

    def _camera_preset_payload(self, preset: str, name: str) -> dict[str, Any]:
        spec = camerae2e_adas_camera_spec(preset)
        return {
            "id": preset,
            "name": name,
            "hfov_deg": spec["optics"]["hfov_deg"],
            "pixel_pitch_um": spec["sensor"]["pixel_pitch_m"] * 1e6,
            "f_number": spec["optics"]["f_number"],
            "readiness_tier": spec["readiness_tier"],
            "truth_boundary": spec["truth_boundary"],
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def _entry_tier(entries: dict[str, Any], name: str, fallback: str) -> str:
    return str(entries.get(name, {}).get("readiness_tier", fallback))


def _entry_stale(entries: dict[str, Any], name: str) -> str | None:
    value = entries.get(name, {}).get("stale_reason")
    return None if value in {"", None} else str(value)


def _scene_payload(scene_type: str) -> Any:
    key = scene_type.strip().lower().replace("_", " ")
    if key in {"adas kitti synthetic", "adas synthetic", "synthetic kitti"}:
        spec = camerae2e_adas_camera_spec("kitti_yolo_demo")
        rows, cols = _image_size_from_spec(spec)
        rgb, _ = _synthetic_adas_rgb_and_labels(rows, cols, index=0)
        return _adas_scene_from_rgb(rgb, spec, AssetStore.default(), "synthetic_kitti_preview")
    if key == "slanted bar":
        return {"type": "slanted bar", "args": [64]}
    if key == "uniform":
        return {"type": "uniform ee", "args": [8]}
    return {"type": "macbeth"}


def _adas_spec_from_settings(settings: dict[str, Any]) -> dict[str, Any]:
    spec = deepcopy(
        camerae2e_adas_camera_spec(str(settings.get("cameraPreset", "kitti_yolo_demo")))
    )
    spec.setdefault("optics", {})["hfov_deg"] = float(settings.get("fovDeg", 81.4))
    spec.setdefault("sensor", {})["pixel_pitch_m"] = float(settings.get("pixelSizeUm", 3.75)) * 1e-6
    return spec


def _focal_length_from_hfov(*, hfov_deg: float, pixel_pitch_m: float, cols: int = 1242) -> float:
    half_width_m = (float(cols) * float(pixel_pitch_m)) / 2.0
    return float(half_width_m / math.tan(math.radians(float(hfov_deg)) / 2.0))


def _preview_from_result(result: dict[str, Any]) -> str | None:
    stages = dict(result.get("stages", {}))
    for key in ("ip_srgb", "ip_result", "sensor_raw"):
        stage = dict(stages.get(key, {}))
        if "array" not in stage:
            continue
        image = np.asarray(stage["array"], dtype=float)
        if image.size == 0:
            continue
        preview = _preview_from_array(image)
        if preview:
            return preview
    return None


def _preview_from_array(image: Any) -> str | None:
    finite = np.asarray(image, dtype=float)
    if finite.size == 0:
        return None
    if finite.ndim == 2:
        finite = np.repeat(finite[..., None], 3, axis=2)
    if finite.ndim != 3:
        return None
    if finite.shape[2] > 3:
        finite = finite[..., :3]
    finite = np.nan_to_num(finite, copy=False)
    if finite.max() > finite.min():
        if finite.max() > 1.0 or finite.min() < 0.0:
            finite = (finite - finite.min()) / (finite.max() - finite.min())
    finite = np.clip(finite, 0.0, 1.0)
    buffer = io.BytesIO()
    iio.imwrite(buffer, (finite * 255.0).astype(np.uint8), extension=".png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _stage_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    payload = []
    for name, summary in dict(report.get("stage_summaries", {})).items():
        item = dict(summary)
        payload.append(
            {
                "name": name,
                "available": bool(item.get("available", False)),
                "shape": item.get("shape", []),
                "dtype": item.get("dtype"),
                "min": _round_float(item.get("min")),
                "max": _round_float(item.get("max")),
                "mean": _round_float(item.get("mean")),
                "std": _round_float(item.get("std")),
            }
        )
    return payload


def _metric_cards(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    color = dict(metrics.get("color", {}))
    artifact = dict(metrics.get("artifact", {}))
    control = dict(metrics.get("control", {}))
    return [
        {
            "id": "rgb_mean",
            "label": "RGB Mean",
            "value": _round_float(color.get("rgb_mean")),
            "unit": "linear",
            "area": "Color",
        },
        {
            "id": "raw_std",
            "label": "RAW Std",
            "value": _round_float(artifact.get("raw_std")),
            "unit": "V/equiv",
            "area": "Artifact",
        },
        {
            "id": "rgb_clip_fraction",
            "label": "Clip Fraction",
            "value": _round_float(artifact.get("rgb_clip_fraction")),
            "unit": "ratio",
            "area": "Artifact",
        },
        {
            "id": "frame_count",
            "label": "HW Frames",
            "value": control.get("frame_count") if control else None,
            "unit": "frames",
            "area": "Control",
        },
    ]


def _case_card(case: dict[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    values = dict(case.get("objective_values", {}))
    params = dict(case.get("parameters", {}))
    return {
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "score": _round_float(case.get("score")),
        "target_score": _round_float(case.get("target_score", case.get("score"))),
        "feasible": bool(case.get("feasible", True)),
        "parameters": _compact_parameters(params),
        "objective_values": {key: _round_float(value) for key, value in values.items()},
        "constraint_results": case.get("constraint_results", []),
        "hard_gates": _safe_json(case.get("hard_gates", [])),
        "score_components": _safe_json(case.get("score_components", {})),
        "perception_metrics": _safe_json(case.get("perception_metrics", {})),
        "raw_quality_proxy": _safe_json(case.get("raw_quality_proxy", {})),
    }


def _pareto_points(
    front: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = front or fallback
    points = []
    for case in cases:
        values = dict(case.get("objective_values", {}))
        points.append(
            {
                "case_index": case.get("case_index"),
                "score": _round_float(case.get("score")),
                "x": _round_float(values.get("metrics.artifact.raw_std")),
                "y": _round_float(values.get("metrics.color.rgb_mean")),
                "clip": _round_float(values.get("metrics.artifact.rgb_clip_fraction")),
            }
        )
    return points


def _compact_parameters(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        label = str(key)
        if label == "sensor.pixel_size":
            out[label] = _round_float(float(value) * 1e6)
        elif label == "sensor.integration_time":
            out[label] = _round_float(float(value) * 1000.0)
        elif label == "ip.sensor_conversion_method":
            out[label] = str(value).replace("_", " ")
        elif isinstance(value, (int, float)):
            out[label] = _round_float(value)
        else:
            out[label] = value
    return out


def _slim_lineage(lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for item in lineage:
        items.append(
            {
                "path": item.get("path"),
                "parameter": item.get("parameter"),
                "status": item.get("status"),
                "requested": _summarize_value(item.get("requested")),
                "after": _summarize_value(item.get("after")),
            }
        )
    return items


def _summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "source_path" in value:
            return {"source_path": str(value["source_path"])}
        keys = list(value.keys())
        if len(keys) > 8:
            return {"type": "mapping", "keys": keys[:8], "key_count": len(keys)}
        return {key: _summarize_value(value[key]) for key in keys}
    if isinstance(value, list):
        if len(value) > 8:
            return {"type": "list", "length": len(value), "head": value[:4]}
        return [_summarize_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    return _safe_json(value)


def _artifact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    validation = dict(payload.get("validation", {}))
    summary = dict(payload.get("summary", {}))
    return {
        "db_ok": validation.get("ok"),
        "warnings": validation.get("warnings", [])[:4],
        "stale_dependency_count": validation.get("stale_dependency_count", 0),
        "summary": summary,
    }


def _dataset_record_card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "split": record.get("split"),
        "raw": record.get("raw"),
        "rgb": record.get("rgb"),
        "labels": record.get("labels"),
        "raw_shape": record.get("raw_shape"),
        "raw_dtype": record.get("raw_dtype"),
        "raw_sha256": record.get("raw_sha256"),
        "label_summary": record.get("label_summary"),
    }


def _balanced_objective() -> list[dict[str, Any]]:
    return [
        {
            "metric": "metrics.color.rgb_mean",
            "direction": "maximize",
            "weight": 1.0,
        },
        {
            "metric": "metrics.artifact.rgb_clip_fraction",
            "direction": "minimize",
            "weight": 0.8,
        },
        {
            "metric": "metrics.artifact.raw_std",
            "direction": "target",
            "target": 0.02,
            "weight": 0.35,
        },
    ]


def _score_adas_yolo_result(
    result: dict[str, Any],
    report: dict[str, Any],
    *,
    labels: dict[str, Any],
    detector: Any,
    settings: dict[str, Any],
) -> dict[str, Any]:
    image = _perception_image_from_result(result)
    if image is None:
        return _failed_perception_score("No ISP RGB image is available for detector input.")
    raw = _stage_array(result, "sensor_raw")
    gt_boxes = _labels_to_adas_boxes(labels, image_shape=image.shape[:2])
    if not gt_boxes:
        return _failed_perception_score("No KITTI-style ground-truth boxes are available.")

    cfg = task_perception_config(
        iou_threshold=0.50,
        score_threshold=float(settings.get("yoloScoreThreshold", 0.25)),
        map_iou_thresholds=_MAP50_95_THRESHOLDS,
    )
    raw_predictions = (
        detector.detect(image, cfg) if hasattr(detector, "detect") else detector(image)
    )
    predictions = _normalize_adas_predictions(raw_predictions)
    map_payload = mean_average_precision(
        predictions,
        gt_boxes,
        cfg,
        iou_thresholds=_MAP50_95_THRESHOLDS,
    )
    det50 = detection_metrics(predictions, gt_boxes, cfg, iou_threshold=0.50)
    small_gt = _small_object_boxes(gt_boxes, image.shape[:2])
    small_det50 = (
        detection_metrics(predictions, small_gt, cfg, iou_threshold=0.50)
        if small_gt
        else det50
    )

    metrics = dict(report.get("metrics", {}))
    color = dict(metrics.get("color", {}))
    rgb_mean = _finite_float(color.get("rgb_mean"))
    clip_fraction = _high_clip_fraction(image)
    quality = _image_quality_support(image, raw, clip_fraction)
    fov = _finite_float(settings.get("fovDeg"))
    preset_fov = _finite_float(
        camerae2e_adas_camera_spec(str(settings.get("cameraPreset", "kitti_yolo_demo"))).get(
            "optics", {}
        ).get("hfov_deg")
    )
    fov_tolerance = max(20.0, float(preset_fov or 0.0) * 0.35)

    hard_gates = [
        _gate("simulation_success", True, True, "scenario produced detector input"),
        _gate("detector_configured", True, True, "YOLO detector returned predictions"),
        _gate("labels_present", len(gt_boxes), 1, "KITTI-style ground truth boxes", op=">="),
        _gate(
            "rgb_high_clip_fraction",
            clip_fraction,
            0.01,
            "high-end RGB saturation must stay <= 1%",
            op="<=",
        ),
        _gate("rgb_mean_min", rgb_mean, 0.02, "mean signal must not be underexposed", op=">="),
        _gate("rgb_mean_max", rgb_mean, 0.95, "mean signal must not be overexposed", op="<="),
        _gate(
            "fov_within_preset_bounds",
            None if fov is None or preset_fov is None else abs(fov - preset_fov),
            fov_tolerance,
            "FOV stays within ADAS preset bounds",
            op="<=",
        ),
    ]
    feasible = all(item["pass"] for item in hard_gates)

    map50_95 = float(map_payload.get("map", 0.0))
    recall50 = _class_balanced_recall(det50)
    small_recall50 = float(small_det50.get("recall", 0.0))
    localization_iou = float(det50.get("mean_matched_iou", 0.0))
    robustness = min(map50_95, recall50)
    support = float(quality["support_score"])
    components = {
        "map50_95": _component(map50_95, _PERCEPTION_SCORE_WEIGHTS["map50_95"]),
        "recall50": _component(recall50, _PERCEPTION_SCORE_WEIGHTS["recall50"]),
        "small_object_recall50": _component(
            small_recall50, _PERCEPTION_SCORE_WEIGHTS["small_object_recall50"]
        ),
        "localization_iou": _component(
            localization_iou, _PERCEPTION_SCORE_WEIGHTS["localization_iou"]
        ),
        "robustness": {
            **_component(robustness, _PERCEPTION_SCORE_WEIGHTS["robustness"]),
            "case_count": int(settings.get("robustnessCases", 1)),
            "status": (
                "nominal_only"
                if int(settings.get("robustnessCases", 1)) <= 1
                else "nominal_proxy"
            ),
        },
        "image_quality_support": _component(
            support, _PERCEPTION_SCORE_WEIGHTS["image_quality_support"]
        ),
    }
    target_score = sum(float(item["contribution"]) for item in components.values())
    if not feasible:
        target_score = 0.0

    objective_values = {
        "perception.target_score": target_score,
        "perception.map50_95": map50_95,
        "perception.map50": float(map_payload.get("ap50", 0.0)),
        "perception.recall50": recall50,
        "perception.small_object_recall50": small_recall50,
        "perception.mean_iou": localization_iou,
        "perception.robustness": robustness,
        "quality.support_score": support,
        "metrics.artifact.rgb_clip_fraction": clip_fraction,
    }
    return {
        "target_score": float(target_score),
        "feasible": bool(feasible),
        "hard_gates": hard_gates,
        "score_components": components,
        "objective_values": objective_values,
        "perception_metrics": {
            "map50_95": map50_95,
            "map": map_payload,
            "detection50": _detection_metric_summary(det50),
            "small_object_detection50": _detection_metric_summary(small_det50),
            "class_balanced_recall50": recall50,
            "prediction_count": len(predictions),
            "ground_truth_count": len(gt_boxes),
            "small_object_count": len(small_gt),
            "classes": list(_ADAS_GROUPS),
        },
        "raw_quality_proxy": quality,
    }


def _failed_perception_score(reason: str) -> dict[str, Any]:
    return {
        "target_score": 0.0,
        "feasible": False,
        "hard_gates": [_gate("simulation_success", False, True, reason)],
        "score_components": {
            key: _component(0.0, weight) for key, weight in _PERCEPTION_SCORE_WEIGHTS.items()
        },
        "objective_values": {
            "perception.target_score": 0.0,
            "perception.map50_95": 0.0,
            "perception.recall50": 0.0,
            "perception.small_object_recall50": 0.0,
            "perception.mean_iou": 0.0,
            "perception.robustness": 0.0,
            "quality.support_score": 0.0,
            "metrics.artifact.rgb_clip_fraction": None,
        },
        "perception_metrics": {"status": "failed", "detail": reason},
        "raw_quality_proxy": {"support_score": 0.0, "status": "unavailable"},
    }


def _component(value: float, weight: float) -> dict[str, float]:
    clipped = float(np.clip(float(value), 0.0, 1.0))
    return {
        "value": clipped,
        "weight": float(weight),
        "contribution": float(clipped * float(weight)),
    }


def _gate(
    metric: str,
    value: Any,
    threshold: Any,
    message: str,
    *,
    op: str = "==",
) -> dict[str, Any]:
    passed = False
    number = _finite_float(value)
    threshold_number = _finite_float(threshold)
    if op == "==":
        passed = bool(value == threshold)
    elif number is not None and threshold_number is not None:
        if op == "<=":
            passed = number <= threshold_number
        elif op == ">=":
            passed = number >= threshold_number
        elif op == "<":
            passed = number < threshold_number
        elif op == ">":
            passed = number > threshold_number
    return {
        "metric": metric,
        "value": _safe_json(value),
        "threshold": _safe_json(threshold),
        "op": op,
        "pass": bool(passed),
        "message": message,
    }


def _perception_image_from_result(result: dict[str, Any]) -> np.ndarray | None:
    for key in ("ip_srgb", "ip_result"):
        image = _stage_array(result, key)
        if image is None or image.size == 0:
            continue
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim == 3:
            return _normalize_image(image)
    return None


def _stage_array(result: dict[str, Any], stage_name: str) -> np.ndarray | None:
    stage = dict(dict(result.get("stages", {})).get(stage_name, {}))
    if "array" not in stage:
        return None
    return np.asarray(stage["array"], dtype=float)


def _normalize_image(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    arr = np.nan_to_num(arr, copy=False)
    if arr.size and (float(np.nanmax(arr)) > 1.0 or float(np.nanmin(arr)) < 0.0):
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0).astype(float, copy=False)


def _discover_local_kitti_pair() -> dict[str, Path] | None:
    for split in ("train", "val"):
        image_dir = _LOCAL_KITTI_ROOT / "images" / split
        label_dir = _LOCAL_KITTI_ROOT / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image in sorted(image_dir.glob("*.png")):
            label = label_dir / f"{image.stem}.txt"
            if label.exists():
                return {"image": image, "label": label}
    return None


def _resolve_label_path(label_path: str | Path | None, image_path: str | Path | None = None) -> str:
    raw = str(label_path or "").strip()
    if raw:
        return raw
    if image_path in {None, ""}:
        return ""
    image = Path(str(image_path)).expanduser()
    parts = list(image.parts)
    for split in ("train", "val"):
        candidate = _LOCAL_KITTI_ROOT / "labels" / split / f"{image.stem}.txt"
        if candidate.exists():
            return str(candidate)
    if "images" in parts:
        index = parts.index("images")
        candidate_parts = parts[:index] + ["labels"] + parts[index + 1 :]
        candidate = Path(*candidate_parts).with_suffix(".txt")
        if candidate.exists():
            return str(candidate)
    return ""


def _load_adas_labels(
    label_path: str | Path | None,
    *,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    path_text = str(label_path or "").strip()
    if not path_text:
        return {
            "schema_version": "camerae2e_workbench_labels_v1",
            "source": "labels_missing",
            "image_size_rc": [int(image_size[0]), int(image_size[1])],
            "objects": [],
            "masks": [],
        }
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"KITTI label does not exist: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if _looks_like_yolo_labels(lines):
        return _yolo_label_file_to_payload(lines, source=str(path), image_size=image_size)
    return camerae2e_kitti_yolo_labels(path, image_size=image_size)


def _looks_like_yolo_labels(lines: list[str]) -> bool:
    if not lines:
        return False
    parts = lines[0].split()
    if len(parts) != 5:
        return False
    try:
        [float(item) for item in parts]
    except ValueError:
        return False
    return True


def _yolo_label_file_to_payload(
    lines: list[str],
    *,
    source: str,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    rows, cols = int(image_size[0]), int(image_size[1])
    objects = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, width, height = [float(item) for item in parts[1:]]
        x1 = (cx - width / 2.0) * cols
        y1 = (cy - height / 2.0) * rows
        x2 = (cx + width / 2.0) * cols
        y2 = (cy + height / 2.0) * rows
        label = _KITTI_ID_TO_LABEL.get(class_id, str(class_id))
        objects.append(
            {
                "label": label,
                "class_id": class_id,
                "bbox_xyxy": [
                    float(np.clip(x1, 0.0, cols)),
                    float(np.clip(y1, 0.0, rows)),
                    float(np.clip(x2, 0.0, cols)),
                    float(np.clip(y2, 0.0, rows)),
                ],
                "yolo_xywhn": [cx, cy, width, height],
                "source_format": "yolo_normalized_xywh",
            }
        )
    return {
        "schema_version": "camerae2e_workbench_yolo_labels_v1",
        "source": source,
        "image_size_rc": [rows, cols],
        "class_map": {label: class_id for class_id, label in _KITTI_ID_TO_LABEL.items()},
        "objects": objects,
        "masks": [],
    }


def _labels_to_adas_boxes(
    labels: dict[str, Any],
    *,
    image_shape: tuple[int, int],
) -> list[TaskBoundingBox]:
    source_rows, source_cols = _image_size_rc(labels.get("image_size_rc", image_shape))
    target_rows, target_cols = image_shape
    scale_x = float(target_cols) / max(float(source_cols), 1.0)
    scale_y = float(target_rows) / max(float(source_rows), 1.0)
    boxes: list[TaskBoundingBox] = []
    for item in labels.get("objects", []):
        group = _adas_group(item.get("label", item.get("class_name", "object")))
        if group is None:
            continue
        bbox = item.get("bbox_xyxy", item.get("xyxy", item.get("bbox")))
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        boxes.append(
            TaskBoundingBox(
                (
                    np.clip(x1 * scale_x, 0.0, target_cols),
                    np.clip(y1 * scale_y, 0.0, target_rows),
                    np.clip(x2 * scale_x, 0.0, target_cols),
                    np.clip(y2 * scale_y, 0.0, target_rows),
                ),
                label=group,
                score=1.0,
                metadata={"source_label": item.get("label"), "source": labels.get("source")},
            )
        )
    return boxes


def _normalize_adas_predictions(predictions: Any) -> list[TaskBoundingBox]:
    boxes = []
    for item in predictions or []:
        box = _as_task_box(item)
        if box is None:
            continue
        group = _adas_group(box.label)
        if group is None:
            continue
        boxes.append(
            TaskBoundingBox(
                box.xyxy,
                label=group,
                score=box.score,
                metadata={"source_label": box.label, **dict(box.metadata)},
            )
        )
    return boxes


def _as_task_box(item: Any) -> TaskBoundingBox | None:
    if isinstance(item, TaskBoundingBox):
        return item
    if isinstance(item, dict):
        bbox = item.get("xyxy", item.get("bbox_xyxy", item.get("bbox")))
        if bbox is None or len(bbox) != 4:
            return None
        return TaskBoundingBox(
            tuple(float(value) for value in bbox),
            label=str(item.get("label", item.get("class_name", "object"))),
            score=float(item.get("score", 1.0)),
            metadata={
                key: value
                for key, value in item.items()
                if key not in {"xyxy", "bbox_xyxy", "bbox", "label", "score"}
            },
        )
    if hasattr(item, "xyxy"):
        return TaskBoundingBox(
            tuple(float(value) for value in item.xyxy),
            label=str(getattr(item, "label", "object")),
            score=float(getattr(item, "score", 1.0)),
        )
    return None


def _adas_group(label: Any) -> str | None:
    key = str(label).strip().lower().replace("_", " ")
    compact = key.replace(" ", "_")
    return _ADAS_CLASS_GROUPS.get(key) or _ADAS_CLASS_GROUPS.get(compact)


def _small_object_boxes(
    boxes: list[TaskBoundingBox],
    image_shape: tuple[int, int],
) -> list[TaskBoundingBox]:
    rows, cols = image_shape
    area_threshold = max(64.0, float(rows * cols) * 0.025)
    return [box for box in boxes if box.area <= area_threshold]


def _class_balanced_recall(metrics: dict[str, Any]) -> float:
    per_label = dict(metrics.get("per_label", {}))
    recalls = [
        float(item.get("recall", 0.0))
        for item in per_label.values()
        if isinstance(item, dict)
    ]
    if recalls:
        return float(np.mean(recalls, dtype=float))
    return float(metrics.get("recall", 0.0))


def _detection_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": _round_float(metrics.get("precision")),
        "recall": _round_float(metrics.get("recall")),
        "f1": _round_float(metrics.get("f1")),
        "average_precision": _round_float(metrics.get("average_precision")),
        "mean_matched_iou": _round_float(metrics.get("mean_matched_iou")),
        "true_positive": metrics.get("true_positive", 0),
        "false_positive": metrics.get("false_positive", 0),
        "false_negative": metrics.get("false_negative", 0),
    }


def _image_quality_support(
    image: np.ndarray,
    raw: np.ndarray | None,
    clip_fraction: float | None,
) -> dict[str, Any]:
    rgb = _normalize_image(image)
    gray = np.mean(rgb, axis=2) if rgb.ndim == 3 else rgb
    gy, gx = np.gradient(gray.astype(float))
    sharpness = float(np.clip(np.mean(np.hypot(gx, gy)) * 24.0, 0.0, 1.0))
    contrast = float(np.clip(float(np.std(gray)) / 0.25, 0.0, 1.0))
    if raw is not None and raw.size:
        raw_arr = np.asarray(raw, dtype=float)
        raw_snr = float(np.mean(raw_arr) / max(float(np.std(raw_arr)), 1e-9))
    else:
        raw_snr = 0.0
    snr_score = float(np.clip(raw_snr / 12.0, 0.0, 1.0))
    low_clip_score = 1.0 - float(np.clip((clip_fraction or 0.0) / 0.01, 0.0, 1.0))
    support = float(
        np.clip(
            0.35 * sharpness + 0.25 * contrast + 0.25 * snr_score + 0.15 * low_clip_score,
            0.0,
            1.0,
        )
    )
    return {
        "support_score": support,
        "sharpness_proxy": sharpness,
        "local_contrast_proxy": contrast,
        "raw_snr_proxy": snr_score,
        "low_clip_score": low_clip_score,
    }


def _high_clip_fraction(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= 0.995))


def _resolve_yolo_model_path(settings: dict[str, Any]) -> str | None:
    raw = str(settings.get("yoloModelPath") or os.environ.get("CAMERAE2E_YOLO_MODEL", "")).strip()
    if raw:
        return raw
    disabled = str(os.environ.get("CAMERAE2E_DISABLE_YOLO_AUTODISCOVER", "")).strip()
    if disabled in {"1", "true", "yes"}:
        return None
    discovered = _discover_yolo_model_path()
    return str(discovered) if discovered is not None else None


def _discover_yolo_model_path() -> Path | None:
    for candidate in _LOCAL_KITTI_YOLO_CHECKPOINTS:
        if candidate.exists():
            return candidate
    if _LOCAL_COCO_YOLO_FALLBACK.exists():
        return _LOCAL_COCO_YOLO_FALLBACK
    return None


def _yolo_model_status(settings: dict[str, Any]) -> dict[str, Any]:
    model_path = _resolve_yolo_model_path(settings)
    path_exists = bool(model_path and Path(model_path).expanduser().exists())
    dependency_available = importlib.util.find_spec("ultralytics") is not None
    configured = bool(model_path)
    source = _yolo_model_source(model_path, settings)
    if not configured:
        status = "detector_not_configured"
    elif not path_exists:
        status = "model_path_missing"
    elif not dependency_available:
        status = "ultralytics_missing"
    elif source == "local_kitti_finetune":
        status = "available"
    elif source == "autodiscovered_coco_demo":
        status = "available_demo_fallback"
    else:
        status = "available"
    if source == "local_kitti_finetune":
        readiness_tier = "trained_local_smoke"
        badge = "KITTI YOLO"
        truth_boundary = (
            "Autodiscovered local YOLO11n checkpoint fine-tuned on KITTI for "
            "Workbench scoring. It is suitable for real detector smoke scoring, "
            "but longer training and held-out evaluation are required before "
            "treating it as detector-performance evidence."
        )
    elif source == "autodiscovered_coco_demo":
        readiness_tier = "demo_fallback"
        badge = "COCO YOLO Demo"
        truth_boundary = (
            "Autodiscovered COCO-pretrained YOLO fallback. This is a real detector "
            "run for demo scoring, but it is not a KITTI-trained ADAS detector."
        )
    else:
        readiness_tier = "external_configured" if configured else "external_required"
        badge = "KITTI YOLO"
        truth_boundary = (
            "ADAS perception optimization requires a caller-supplied KITTI-trained "
            "YOLO model. CameraE2E does not synthesize detector accuracy."
        )
    return {
        "available": configured and path_exists and dependency_available,
        "configured": configured,
        "path_exists": path_exists,
        "dependency_available": dependency_available,
        "path": model_path,
        "source": source,
        "status": status,
        "readiness_tier": readiness_tier,
        "badge": badge,
        "truth_boundary": truth_boundary,
    }


def _yolo_model_source(model_path: str | None, settings: dict[str, Any]) -> str | None:
    if not model_path:
        return None
    try:
        resolved = Path(model_path).expanduser().resolve()
        for candidate in _LOCAL_KITTI_YOLO_CHECKPOINTS:
            if candidate.exists() and resolved == candidate.resolve():
                return "local_kitti_finetune"
        if Path(model_path).expanduser().resolve() == _LOCAL_COCO_YOLO_FALLBACK.resolve():
            return "autodiscovered_coco_demo"
    except FileNotFoundError:
        return "configured"
    explicit = str(
        settings.get("yoloModelPath") or os.environ.get("CAMERAE2E_YOLO_MODEL", "")
    ).strip()
    if explicit:
        return "configured"
    return "autodiscovered"


def _detector_truth_boundary(readiness: dict[str, Any]) -> str:
    if readiness.get("status") == "available_demo_fallback":
        return (
            "ADAS YOLO Perception Score is currently using a COCO-pretrained YOLO "
            "demo fallback with KITTI-style labels. It is a real detector metric for "
            "Workbench bring-up, but it must not be interpreted as KITTI-trained "
            "ADAS detector performance or product sign-off."
        )
    source = None
    for asset in readiness.get("required_assets", []):
        if asset.get("id") == "kitti_trained_yolo_model":
            source = asset.get("source")
            break
    if source == "local_kitti_finetune":
        return (
            "ADAS YOLO Perception Score is using a local YOLO11n checkpoint "
            "fine-tuned on KITTI for one fast Workbench pass. It is real detector "
            "scoring, but still a smoke/fine-tune asset rather than final detector "
            "performance evidence or product sign-off."
        )
    return (
        "ADAS YOLO Perception Score uses an external KITTI-trained detector and "
        "KITTI-style labels. It is a task-performance optimization signal, not "
        "product sign-off or camera calibration evidence."
    )


def _kitti_label_status(settings: dict[str, Any]) -> dict[str, Any]:
    label_source = str(settings.get("labelSource", "synthetic_kitti") or "synthetic_kitti")
    label_path = str(settings.get("kittiLabelPath", "") or "").strip()
    key = label_source.strip().lower().replace("-", "_")
    if key in {"none", "missing", "not_provided"}:
        return {
            "available": False,
            "configured": False,
            "source": label_source,
            "status": "labels_missing",
        }
    if label_path:
        exists = Path(label_path).expanduser().exists()
        return {
            "available": exists,
            "configured": True,
            "source": label_path,
            "status": "available" if exists else "label_path_missing",
        }
    return {
        "available": key in {"synthetic_kitti", "demo", "default"},
        "configured": True,
        "source": label_source,
        "status": (
            "available" if key in {"synthetic_kitti", "demo", "default"} else "labels_missing"
        ),
    }


def _image_size_from_spec(spec: dict[str, Any]) -> tuple[int, int]:
    return _image_size_rc(spec.get("demo", {}).get("image_size_rc", [96, 320]))


def _image_size_rc(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return 1, 1


def _assign_workbench_parameter(scenario: dict[str, Any], path: str, value: Any) -> None:
    if path.startswith("sensor."):
        scenario.setdefault("sensor", {})[path.split(".", 1)[1]] = value
    elif path.startswith("fdtd."):
        scenario.setdefault("fdtd", {})[path.split(".", 1)[1]] = value
    elif path.startswith("tcad."):
        scenario.setdefault("tcad", {})[path.split(".", 1)[1]] = value
    elif path.startswith("hw_isp."):
        scenario.setdefault("hw_isp", {})[path.split(".", 1)[1]] = value
    else:
        scenario.setdefault("parameters", {})[path] = value


def _rank_perception_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda item: (
            bool(item.get("feasible", False)),
            float(item.get("score", -math.inf)),
            -int(item.get("case_index", 0)),
        ),
        reverse=True,
    )


def _perception_pareto_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in cases:
        dominated = False
        for other in cases:
            if other is candidate:
                continue
            if _perception_dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return _rank_perception_cases(frontier)


def _perception_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = dict(left.get("objective_values", {}))
    right_values = dict(right.get("objective_values", {}))
    left_score = _finite_float(left_values.get("perception.target_score"))
    right_score = _finite_float(right_values.get("perception.target_score"))
    left_clip = _finite_float(left_values.get("metrics.artifact.rgb_clip_fraction"))
    right_clip = _finite_float(right_values.get("metrics.artifact.rgb_clip_fraction"))
    if None in {left_score, right_score, left_clip, right_clip}:
        return False
    return bool(
        left_score >= right_score
        and left_clip <= right_clip
        and (left_score > right_score or left_clip < right_clip)
    )


def _perception_pareto_points(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for case in cases:
        values = dict(case.get("objective_values", {}))
        points.append(
            {
                "case_index": case.get("case_index"),
                "score": _round_float(case.get("score")),
                "x": _round_float(values.get("perception.target_score")),
                "y": _round_float(values.get("metrics.artifact.rgb_clip_fraction")),
                "clip": _round_float(values.get("metrics.artifact.rgb_clip_fraction")),
                "map50_95": _round_float(values.get("perception.map50_95")),
                "recall50": _round_float(values.get("perception.recall50")),
            }
        )
    return points


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unique_sorted(values: list[float]) -> list[float]:
    rounded = sorted({round(float(item), 12) for item in values})
    return [float(item) for item in rounded]


def _public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _safe_json(value)
        for key, value in settings.items()
        if key not in {"tcadDb", "fdtdLut"}
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": _round_float(np.nanmin(value)) if value.size else None,
            "max": _round_float(np.nanmax(value)) if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _round_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(f"{number:.6g}")


def _html_report(report: dict[str, Any]) -> str:
    simulation = report.get("simulation") or {}
    metrics = simulation.get("metrics", []) if isinstance(simulation, dict) else []
    metric_rows = "\n".join(
        f"<tr><td>{item.get('label')}</td><td>{item.get('value')}</td><td>{item.get('area')}</td></tr>"
        for item in metrics
    )
    dataset = report.get("dataset") or {}
    dataset_root = dataset.get("dataset_root") if isinstance(dataset, dict) else None
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CameraE2E Workbench Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #15202b;
    }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    td, th {{ border: 1px solid #d7dee8; padding: 8px; text-align: left; }}
    .boundary {{
      background: #fff7e6;
      border: 1px solid #e5b35b;
      padding: 12px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>CameraE2E Workbench Report</h1>
  <p class="boundary">{report.get("fidelity_boundary")}</p>
  <h2>Simulation Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Value</th><th>Area</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Optimization</h2>
  <pre>{json.dumps(_safe_json(report.get("optimization")), indent=2)}</pre>
  <h2>Dataset</h2>
  <p>{dataset_root or "No dataset exported yet."}</p>
</body>
</html>
"""
