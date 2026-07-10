"""Objective, constraint, sensitivity, and perception evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from pyisetcam import (
    TaskBoundingBox,
    camerae2e_parameter_candidate_plan,
    detection_metrics,
    mean_average_precision,
    render_detection_overlay,
    task_model_from_config,
    task_perception_config,
)

from .benchmark import ADAS_KITTI_CLASS_IDS, benchmark_inventory, benchmark_scenes
from .engine import CameraEngine
from .geometry import (
    aligned_sensor_size,
    derive_geometry_transform,
    global_ssim,
    ideal_recapture,
    scene_image_size,
    transform_label_payload,
)
from .models import (
    BenchmarkPreflightResult,
    Constraint,
    DesignVariable,
    FidelityLevel,
    GeometryTransform,
    Objective,
    RobustnessCase,
    SceneCase,
    StudyRecord,
)
from .requirements import evaluate_requirement_gates


class PerceptionConfigurationError(ValueError):
    pass


def numeric_path(payload: dict[str, Any], path: str) -> float:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric path is unavailable: {path}")
        current = current[part]
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"Metric is not finite: {path}")
    return value


def score_objectives(
    report: dict[str, Any], objectives: list[Objective]
) -> tuple[float, dict[str, Any]]:
    components: dict[str, Any] = {}
    total = 0.0
    for objective in objectives:
        value = numeric_path(report, objective.metric)
        if objective.direction == "minimize":
            utility = -value
        elif objective.direction == "target":
            assert objective.target is not None
            utility = -abs(value - float(objective.target))
        else:
            utility = value
        contribution = objective.weight * utility
        total += contribution
        components[objective.id] = {
            "metric": objective.metric,
            "value": value,
            "direction": objective.direction,
            "weight": objective.weight,
            "utility": utility,
            "contribution": contribution,
        }
    return float(total), components


def evaluate_constraints(
    report: dict[str, Any], constraints: list[Constraint]
) -> tuple[bool, list[dict[str, Any]]]:
    results = []
    for constraint in constraints:
        value = numeric_path(report, constraint.metric)
        passed = {
            "<=": value <= constraint.value,
            ">=": value >= constraint.value,
            "<": value < constraint.value,
            ">": value > constraint.value,
            "==": math.isclose(value, constraint.value),
        }[constraint.operator]
        results.append(
            {
                "id": constraint.id,
                "metric": constraint.metric,
                "value": value,
                "operator": constraint.operator,
                "threshold": constraint.value,
                "hard": constraint.hard,
                "pass": bool(passed),
            }
        )
    return all(item["pass"] or not item["hard"] for item in results), results


class StudyEvaluator:
    def __init__(self, engine: CameraEngine | None = None) -> None:
        self.engine = engine or CameraEngine()
        self._detectors: dict[str, Any] = {}

    def evaluate_baseline(
        self,
        study: StudyRecord,
        *,
        scene_index: int = 0,
        fidelity: FidelityLevel | None = None,
        include_arrays: bool = True,
    ) -> dict[str, Any]:
        spec = study.spec
        result = self.engine.evaluate(
            spec.baseline,
            spec.scenes[scene_index],
            fidelity=fidelity or spec.fidelity_policy.search_level,
            policy=spec.fidelity_policy,
            seed=spec.seed,
            include_arrays=include_arrays,
        )
        score = self.score_result(study, result, scene=spec.scenes[scene_index])
        overlay = score.pop("_overlay", None)
        if overlay is not None and include_arrays:
            result.setdefault("stages", {})["detector_overlay"] = {
                "array": overlay,
                "shape": list(overlay.shape),
                "dtype": str(overlay.dtype),
            }
        requirements = evaluate_requirement_gates(study, result).model_dump(mode="json")
        score = self._apply_requirement_result(score, requirements)
        return {**result, "evaluation": score, "requirement_gates": requirements}

    def preflight(
        self,
        study: StudyRecord,
        *,
        scene_count: int | None = None,
    ) -> dict[str, Any]:
        spec = study.spec
        requested_count = int(scene_count or spec.benchmark.quick_scene_count)
        inventory = benchmark_inventory(study)
        scenes = benchmark_scenes(study, requested_count)
        checks: list[dict[str, Any]] = []

        def check(check_id: str, passed: bool, value: Any, limit: Any, message: str) -> None:
            checks.append(
                {
                    "id": check_id,
                    "pass": bool(passed),
                    "value": value,
                    "limit": limit,
                    "message": message,
                }
            )

        model_path = spec.perception_model_path
        model_ready = bool(model_path and Path(model_path).expanduser().is_file())
        check(
            "detector_model",
            model_ready,
            model_path,
            "existing KITTI/ADAS YOLO model",
            "Configure perception_model_path with an existing model file.",
        )
        check(
            "benchmark_scene_count",
            len(scenes) >= requested_count,
            len(scenes),
            requested_count,
            "The benchmark must contain the requested number of image/label pairs.",
        )
        labels_ready = bool(scenes) and all(
            scene.label_path and Path(scene.label_path).expanduser().is_file() for scene in scenes
        )
        check(
            "benchmark_labels",
            labels_ready,
            sum(
                bool(scene.label_path and Path(scene.label_path).expanduser().is_file())
                for scene in scenes
            ),
            len(scenes),
            "Every selected benchmark frame requires labels.",
        )
        if not model_ready or not labels_ready or len(scenes) < requested_count:
            result = BenchmarkPreflightResult(
                ready=False,
                status="detector_not_configured" if not model_ready else "benchmark_incomplete",
                scene_count=len(scenes),
                model_path=model_path,
                checks=checks,
                training=self._training_recommendation(study),
            )
            return {**result.model_dump(mode="json"), "inventory": inventory}

        try:
            detector = self._detector(str(model_path))
        except Exception as exc:
            check("detector_load", False, type(exc).__name__, "load succeeds", str(exc))
            result = BenchmarkPreflightResult(
                ready=False,
                status="detector_load_failed",
                scene_count=len(scenes),
                model_path=model_path,
                checks=checks,
                training=self._training_recommendation(study),
            )
            return {**result.model_dump(mode="json"), "inventory": inventory}

        source_rows: list[dict[str, float]] = []
        ideal_rows: list[dict[str, float]] = []
        ssim_values: list[float] = []
        transform_errors: list[float] = []
        camera_rows: list[dict[str, float]] = []
        camera_reference_rows: list[dict[str, float]] = []
        camera_ssim_values: list[float] = []
        camera_color_values: list[float] = []
        camera_fidelity: dict[str, Any] = {}
        camera_execution_error: str | None = None
        active_rows, active_cols = aligned_sensor_size(spec.baseline)
        output_size = (
            max(1, active_rows // spec.baseline.sensor.binning_factor),
            max(1, active_cols // spec.baseline.sensor.binning_factor),
        )
        fidelity_scene_count = min(spec.benchmark.fidelity_scene_count, len(scenes))
        for scene_index, scene in enumerate(scenes):
            image = self._read_scene_rgb(scene)
            source_labels = load_adas_label_payload(
                str(scene.label_path), image_size=image.shape[:2]
            )
            source_gt = self._ground_truth(source_labels)
            source_metric, _source_predictions = self._perception_metrics(
                detector, image, source_gt
            )
            source_rows.append(source_metric)
            transform = derive_geometry_transform(
                spec.baseline,
                scene,
                output_size_rc=output_size,
                readout_size_rc=output_size,
            )
            ideal_image = ideal_recapture(image, output_size)
            ideal_labels = transform_label_payload(source_labels, transform)
            ideal_gt = self._ground_truth(ideal_labels)
            ideal_metric, _ideal_predictions = self._perception_metrics(
                detector, ideal_image, ideal_gt
            )
            ideal_rows.append(ideal_metric)
            ssim_values.append(global_ssim(image, ideal_image))
            transform_errors.append(self._bbox_transform_error(source_gt, ideal_gt, transform))
            if scene_index >= fidelity_scene_count or camera_execution_error is not None:
                continue
            try:
                camera_result = self.engine.evaluate(
                    spec.baseline,
                    scene,
                    fidelity=spec.fidelity_policy.search_level,
                    policy=spec.fidelity_policy,
                    seed=spec.seed + scene_index,
                    include_arrays=True,
                )
                camera_image = self._stage_image(camera_result, "ip_srgb", "ip_result")
                if camera_image is None:
                    raise ValueError("Camera evaluation did not produce an ISP RGB stage")
                camera_transform_payload = (camera_result.get("geometry") or {}).get(
                    "transform"
                )
                if not camera_transform_payload:
                    raise ValueError("Camera evaluation did not produce a geometry transform")
                camera_labels = transform_label_payload(
                    source_labels,
                    GeometryTransform.model_validate(camera_transform_payload),
                )
                camera_metric, _camera_predictions = self._perception_metrics(
                    detector, camera_image, self._ground_truth(camera_labels)
                )
                diagnostics = camera_result.get("color_diagnostics") or {}
                camera_rows.append(camera_metric)
                camera_reference_rows.append(ideal_metric)
                camera_ssim_values.append(
                    float(diagnostics.get("ideal_output_ssim", 0.0))
                )
                camera_color_values.append(
                    float(
                        diagnostics.get(
                            "relative_channel_gain_imbalance",
                            diagnostics.get("output_channel_imbalance", float("inf")),
                        )
                    )
                )
                camera_fidelity = dict(camera_result.get("fidelity") or {})
            except Exception as exc:
                camera_execution_error = f"{type(exc).__name__}: {exc}"

        source = self._aggregate_perception(source_rows)
        ideal_summary = self._aggregate_perception(ideal_rows)
        source_recall = float(source.get("recall50", 0.0))
        ideal_recall = float(ideal_summary.get("recall50", 0.0))
        retention = ideal_recall / max(source_recall, 1e-12)
        ideal_summary.update(
            {
                "recall_retention": retention,
                "ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
                "max_bbox_transform_error_px": max(transform_errors, default=0.0),
            }
        )
        benchmark = spec.benchmark
        camera_summary = self._aggregate_perception(camera_rows)
        camera_reference = self._aggregate_perception(camera_reference_rows)
        camera_reference_recall = float(camera_reference.get("recall50", 0.0))
        camera_recall = float(camera_summary.get("recall50", 0.0))
        camera_retention = (
            camera_recall / camera_reference_recall
            if camera_reference_recall > 0.0
            else 0.0
        )
        camera_ssim = float(np.mean(camera_ssim_values)) if camera_ssim_values else None
        camera_color_imbalance = (
            float(np.max(camera_color_values)) if camera_color_values else None
        )
        camera_output: dict[str, Any] = {
            **camera_summary,
            "scene_count": len(camera_rows),
            "reference_recall50": camera_reference_recall,
            "recall_retention": camera_retention,
            "ideal_output_ssim": camera_ssim,
            "relative_channel_gain_imbalance": camera_color_imbalance,
            "fidelity": camera_fidelity,
            "execution_error": camera_execution_error,
            "remediation": (
                "Use L0_analytic for search or attach a color/response-calibrated LUT."
                if spec.fidelity_policy.search_level != FidelityLevel.ANALYTIC
                else (
                    "Inspect exposure, radiometry, QE, and ISP settings before optimization."
                )
            ),
        }
        check(
            "source_map50",
            source["map50"] >= benchmark.detector_map50_min,
            source["map50"],
            benchmark.detector_map50_min,
            "The source detector must establish a useful KITTI baseline.",
        )
        check(
            "source_map50_95",
            source["map50_95"] >= benchmark.detector_map50_95_min,
            source["map50_95"],
            benchmark.detector_map50_95_min,
            "The detector localization baseline is below the optimization threshold.",
        )
        check(
            "source_recall",
            source["recall50"] >= benchmark.detector_recall_min,
            source["recall50"],
            benchmark.detector_recall_min,
            "The source detector recall is too low to rank camera candidates.",
        )
        check(
            "ideal_recall_retention",
            retention >= benchmark.ideal_recall_retention_min,
            retention,
            benchmark.ideal_recall_retention_min,
            "Geometry-only resizing must preserve source detector recall.",
        )
        check(
            "ideal_ssim",
            ideal_summary["ssim"] >= benchmark.ideal_ssim_min,
            ideal_summary["ssim"],
            benchmark.ideal_ssim_min,
            "The ideal recapture path must preserve source image structure.",
        )
        check(
            "bbox_geometry",
            ideal_summary["max_bbox_transform_error_px"] <= 0.5,
            ideal_summary["max_bbox_transform_error_px"],
            0.5,
            "Bounding-box coordinate transforms must be sub-pixel accurate.",
        )
        check(
            "fidelity_execution",
            camera_execution_error is None and len(camera_rows) == fidelity_scene_count,
            camera_execution_error or len(camera_rows),
            fidelity_scene_count,
            "The selected search fidelity must execute on the camera sanity subset.",
        )
        check(
            "fidelity_detector_retention",
            camera_execution_error is None
            and camera_retention >= benchmark.fidelity_recall_retention_min,
            camera_retention,
            benchmark.fidelity_recall_retention_min,
            "Camera output must retain enough detector signal to rank candidates.",
        )
        check(
            "fidelity_color_balance",
            camera_execution_error is None
            and camera_color_imbalance is not None
            and camera_color_imbalance <= benchmark.fidelity_color_imbalance_max,
            camera_color_imbalance,
            benchmark.fidelity_color_imbalance_max,
            "Camera channel gains are inconsistent with the ideal scene reference.",
        )
        check(
            "fidelity_ssim",
            camera_execution_error is None
            and camera_ssim is not None
            and camera_ssim >= benchmark.fidelity_ssim_min,
            camera_ssim,
            benchmark.fidelity_ssim_min,
            "Camera output structure is too degraded for a rankable objective.",
        )
        ready = all(item["pass"] for item in checks)
        failed_ids = {item["id"] for item in checks if not item["pass"]}
        detector_failed = bool(
            failed_ids
            & {
                "detector_model",
                "detector_load",
                "benchmark_scene_count",
                "benchmark_labels",
                "source_map50",
                "source_map50_95",
                "source_recall",
            }
        )
        result = BenchmarkPreflightResult(
            ready=ready,
            status=(
                "ready"
                if ready
                else (
                    "objective_degenerate"
                    if detector_failed
                    else (
                        "fidelity_invalid"
                        if any(item.startswith("fidelity_") for item in failed_ids)
                        else "objective_degenerate"
                    )
                )
            ),
            scene_count=len(scenes),
            model_path=str(Path(str(model_path)).expanduser().resolve()),
            checks=checks,
            source_baseline=source,
            ideal_recapture=ideal_summary,
            camera_output=camera_output,
            training=self._training_recommendation(study) if detector_failed else {},
        )
        return {**result.model_dump(mode="json"), "inventory": inventory}

    def benchmark(
        self,
        study: StudyRecord,
        *,
        scene_count: int | None = None,
        parameters: dict[str, Any] | None = None,
        include_robustness: bool = True,
    ) -> dict[str, Any]:
        count = int(scene_count or study.spec.benchmark.quick_scene_count)
        scenes = benchmark_scenes(study, count)
        if len(scenes) < count:
            raise ValueError(
                f"Benchmark requested {count} scenes but only {len(scenes)} are available"
            )
        case = self._evaluate_candidate_scenes(
            study,
            dict(parameters or {}),
            scenes,
            case_index=0,
            cache={},
            include_robustness=include_robustness,
        )
        return {
            "schema_version": "camerae2e_benchmark_run_v2",
            "study_id": study.id,
            "scene_count": count,
            "candidate": case,
            "truth_boundary": (
                "Metrics aggregate only the recorded benchmark scenes and fidelity. "
                "KITTI RGB inputs remain display-derived scene proxies."
            ),
        }

    def sensitivity(self, study: StudyRecord) -> dict[str, Any]:
        spec = study.spec
        if spec.target_profile == "adas_yolo_perception":
            self._require_preflight(study)
        count = min(8, spec.benchmark.quick_scene_count)
        scenes = benchmark_scenes(study, count)
        cache: dict[tuple[int, str], dict[str, Any]] = {}
        baseline = self._evaluate_candidate_scenes(
            study, {}, scenes, case_index=-1, cache=cache, include_robustness=False
        )
        baseline_score = float(baseline["target_score"])
        axes: list[dict[str, Any]] = []
        for variable in spec.design_variables:
            if not variable.enabled:
                continue
            cases = []
            for index, value in enumerate(variable_values(variable)):
                scored = self._evaluate_candidate_scenes(
                    study,
                    {variable.path: value},
                    scenes,
                    case_index=(len(axes) + 1) * 1000 + index,
                    cache=cache,
                    include_robustness=False,
                )
                cases.append(
                    {
                        "value": value,
                        "target_score": scored["target_score"],
                        "feasible": scored["feasible"],
                        "metrics": scored["metrics"],
                    }
                )
            values = [float(item["target_score"]) for item in cases]
            span = max(values) - min(values) if values else 0.0
            axes.append(
                {
                    "path": variable.path,
                    "unit": variable.unit,
                    "readiness_tier": variable.readiness_tier.value,
                    "score_span": span,
                    "relative_influence": span / max(abs(baseline_score), 1e-12),
                    "cases": cases,
                }
            )
        ranked = sorted(axes, key=lambda item: item["score_span"], reverse=True)
        return {
            "schema_version": "camerae2e_sensitivity_v2",
            "study_id": study.id,
            "baseline_score": baseline_score,
            "axis_count": len(ranked),
            "scene_count": len(scenes),
            "inactive_axes": [item["path"] for item in ranked if item["score_span"] <= 1e-12],
            "axes": ranked,
        }

    def optimize(self, study: StudyRecord) -> dict[str, Any]:
        spec = study.spec
        preflight = None
        if spec.target_profile == "adas_yolo_perception":
            preflight = self._require_preflight(study)
        parameter_space = {
            variable.path: variable_values(variable)
            for variable in spec.design_variables
            if variable.enabled
        }
        candidate_plan = camerae2e_parameter_candidate_plan(
            parameter_space,
            method=spec.search_method,
            max_cases=spec.search_budget,
            seed=spec.seed,
        )
        candidates = [dict(item) for item in candidate_plan.get("candidates", [])]
        cache: dict[tuple[int, str], dict[str, Any]] = {}
        latest_cases: dict[int, dict[str, Any]] = {}
        active = list(range(len(candidates)))
        stages: list[dict[str, Any]] = []
        if spec.target_profile == "adas_yolo_perception":
            scene_counts = list(spec.benchmark.halving_scene_counts)
            if spec.benchmark.final_scene_count not in scene_counts:
                scene_counts.append(spec.benchmark.final_scene_count)
        else:
            scene_counts = [len(spec.scenes)]
        for stage_index, scene_count in enumerate(scene_counts):
            scenes = (
                benchmark_scenes(study, scene_count)
                if spec.target_profile == "adas_yolo_perception"
                else list(spec.scenes)
            )
            if len(scenes) < scene_count:
                raise ValueError(
                    f"Optimization stage requires {scene_count} benchmark scenes; "
                    f"only {len(scenes)} are available"
                )
            final_stage = stage_index == len(scene_counts) - 1
            stage_cases = []
            for case_index in active:
                scored = self._evaluate_candidate_scenes(
                    study,
                    candidates[case_index],
                    scenes,
                    case_index=case_index,
                    cache=cache,
                    include_robustness=final_stage,
                )
                latest_cases[case_index] = scored
                stage_cases.append(scored)
            ranked_stage = self._rank_cases(stage_cases)
            if final_stage:
                keep = min(spec.benchmark.final_candidate_count, len(ranked_stage))
            else:
                keep = max(
                    spec.benchmark.final_candidate_count,
                    int(math.ceil(len(ranked_stage) / 2.0)),
                )
            active = [int(item["case_index"]) for item in ranked_stage[:keep]]
            stages.append(
                {
                    "stage": stage_index + 1,
                    "scene_count": len(scenes),
                    "evaluated_candidate_count": len(stage_cases),
                    "promoted_candidate_ids": [item["case_id"] for item in ranked_stage[:keep]],
                }
            )
        finalist_indices = set(active)
        cases = [
            {**latest_cases[index], "finalist": index in finalist_indices}
            for index in sorted(latest_cases)
        ]
        finalists = [item for item in cases if item["finalist"]]
        non_finalists = sorted(
            (item for item in cases if not item["finalist"]),
            key=lambda item: (
                int(item.get("scene_count", 0)),
                bool(item["feasible"]),
                float(item["target_score"]),
            ),
            reverse=True,
        )
        ranked_finalists = self._rank_cases(finalists)
        ranked = [*ranked_finalists, *non_finalists]
        feasible = [item for item in ranked if item["feasible"]]
        feasible_finalists = [item for item in ranked_finalists if item["feasible"]]
        if spec.target_profile == "adas_yolo_perception" and cases:
            detector_signal = [
                float((item.get("perception_metrics") or {}).get("map50_95", 0.0))
                + float((item.get("perception_metrics") or {}).get("recall50", 0.0))
                + float((item.get("perception_metrics") or {}).get("prediction_count", 0.0))
                for item in cases
            ]
            if max(detector_signal, default=0.0) <= 0.0:
                raise PerceptionConfigurationError(
                    "objective_degenerate: all candidates produced zero detector signal; "
                    "no best camera candidate was created"
                )
        pareto = self._pareto(feasible_finalists, spec.objectives)
        return {
            "schema_version": "camerae2e_optimization_v2",
            "study_id": study.id,
            "target_profile": spec.target_profile,
            "search_method": candidate_plan.get("method"),
            "seed": spec.seed,
            "parameter_space": parameter_space,
            "candidate_plan": {
                key: value for key, value in candidate_plan.items() if key != "candidates"
            },
            "preflight": preflight,
            "successive_halving": stages,
            "case_count": len(cases),
            "feasible_count": len(feasible),
            "finalist_count": len(finalists),
            "final_feasible_count": len(feasible_finalists),
            "best_case": (
                feasible_finalists[0]
                if feasible_finalists
                else (
                    ranked_finalists[0]
                    if ranked_finalists and spec.target_profile == "raw_quality"
                    else None
                )
            ),
            "top_cases": ranked[:8],
            "pareto_front": pareto,
            "cases": cases,
            "truth_boundary": (
                "Ranking is valid only for the recorded scenes, objectives, model, and fidelity. "
                "Proxy parameters do not become calibrated evidence through optimization."
            ),
        }

    def score_result(
        self,
        study: StudyRecord,
        result: dict[str, Any],
        *,
        scene: SceneCase | None = None,
        robustness_value: float | None = None,
    ) -> dict[str, Any]:
        if study.spec.target_profile == "adas_yolo_perception":
            return self._score_perception(
                study,
                result,
                scene=scene or study.spec.scenes[0],
                robustness_value=robustness_value,
            )
        report = {"metrics": result["metrics"]}
        target_score, components = score_objectives(report, study.spec.objectives)
        feasible, constraints = evaluate_constraints(report, study.spec.constraints)
        return {
            "target_score": target_score if feasible else 0.0,
            "feasible": feasible,
            "score_components": components,
            "constraint_results": constraints,
            "target_status": "available",
        }

    def _score_perception(
        self,
        study: StudyRecord,
        result: dict[str, Any],
        *,
        scene: SceneCase,
        robustness_value: float | None = None,
    ) -> dict[str, Any]:
        model_path = study.spec.perception_model_path
        if not model_path or not Path(model_path).expanduser().is_file():
            raise PerceptionConfigurationError(
                "ADAS perception optimization requires an existing KITTI/ADAS YOLO model path"
            )
        if not scene.label_path or not Path(scene.label_path).expanduser().is_file():
            raise PerceptionConfigurationError(
                "ADAS perception optimization requires a KITTI label file for the scene"
            )
        image = self._stage_image(result, "ip_srgb", "ip_result")
        if image is None:
            raise ValueError("No ISP RGB stage is available for perception evaluation")
        labels = load_adas_label_payload(scene.label_path, image_size=scene_image_size(scene))
        transform_payload = (result.get("geometry") or {}).get("transform")
        if transform_payload:
            labels = transform_label_payload(
                labels, GeometryTransform.model_validate(transform_payload)
            )
        ground_truth = self._ground_truth(labels)
        if not ground_truth:
            raise PerceptionConfigurationError(
                "KITTI label file contains no ADAS ground-truth boxes"
            )
        detector = self._detector(model_path)
        perception, predictions = self._perception_metrics(detector, image, ground_truth)
        map_value = perception["map50_95"]
        recall = perception["recall50"]
        small_recall = perception["small_object_recall50"]
        localization = perception["mean_iou"]
        robustness = min(map_value, recall) if robustness_value is None else float(robustness_value)
        rgb_mean = float(np.mean(image))
        low_clip = float(np.mean(image <= 0.0))
        high_clip = float(np.mean(image >= 1.0))
        raw = self._stage_image(result, "sensor_raw")
        raw_snr = 0.0 if raw is None else float(np.mean(raw) / max(np.std(raw), 1e-12))
        support = float(
            np.clip(
                0.5 * min(raw_snr / 20.0, 1.0) + 0.5 * (1.0 - high_clip),
                0.0,
                1.0,
            )
        )
        weights = {
            "map50_95": 0.45,
            "recall50": 0.20,
            "small_object_recall50": 0.10,
            "localization_iou": 0.10,
            "robustness": 0.10,
            "image_quality_support": 0.05,
        }
        values = {
            "map50_95": map_value,
            "recall50": recall,
            "small_object_recall50": small_recall,
            "localization_iou": localization,
            "robustness": robustness,
            "image_quality_support": support,
        }
        geometry_ok = bool(
            (result.get("geometry") or {})
            .get("metrics", {})
            .get("output_shape_matches_contract", True)
        )
        gates = [
            {"id": "labels_present", "pass": bool(ground_truth), "value": len(ground_truth)},
            {
                "id": "rgb_high_clip_fraction",
                "pass": high_clip <= 0.01,
                "value": high_clip,
                "limit": 0.01,
            },
            {
                "id": "rgb_low_clip_diagnostic",
                "pass": True,
                "value": low_clip,
                "hard": False,
            },
            {"id": "rgb_mean_min", "pass": rgb_mean >= 0.02, "value": rgb_mean, "limit": 0.02},
            {"id": "rgb_mean_max", "pass": rgb_mean <= 0.95, "value": rgb_mean, "limit": 0.95},
            {"id": "geometry_contract", "pass": geometry_ok, "value": geometry_ok},
        ]
        feasible = all(item["pass"] for item in gates)
        score = sum(weights[key] * values[key] for key in values)
        constraint_report = {"metrics": result["metrics"]}
        constraint_feasible, constraints = evaluate_constraints(
            constraint_report, study.spec.constraints
        )
        feasible = feasible and constraint_feasible
        return {
            "target_score": score if feasible else 0.0,
            "feasible": feasible,
            "score_components": {
                key: {
                    "value": values[key],
                    "weight": weights[key],
                    "contribution": weights[key] * values[key],
                }
                for key in values
            },
            "constraint_results": constraints,
            "hard_gates": gates,
            "perception_metrics": {
                **perception,
                "model_path": str(Path(model_path).expanduser().resolve()),
            },
            "target_status": "available",
            "_overlay": render_detection_overlay(image, predictions, ground_truth),
        }

    def _evaluate_candidate_scenes(
        self,
        study: StudyRecord,
        parameters: dict[str, Any],
        scenes: list[SceneCase],
        *,
        case_index: int,
        cache: dict[tuple[int, str], dict[str, Any]],
        include_robustness: bool,
    ) -> dict[str, Any]:
        records = []
        for scene_index, scene in enumerate(scenes):
            key = (case_index, scene.id)
            if key not in cache:
                result = self.engine.evaluate(
                    study.spec.baseline,
                    scene,
                    fidelity=study.spec.fidelity_policy.search_level,
                    policy=study.spec.fidelity_policy,
                    seed=study.spec.seed + (max(case_index, 0) * 10000) + scene_index,
                    parameter_overrides=parameters,
                    include_arrays=True,
                )
                scored = self.score_result(study, result, scene=scene)
                scored.pop("_overlay", None)
                requirement = evaluate_requirement_gates(study, result).model_dump(mode="json")
                scored = self._apply_requirement_result(scored, requirement)
                cache[key] = {
                    "scene_id": scene.id,
                    "scene_name": scene.name,
                    "target_score": scored["target_score"],
                    "feasible": scored["feasible"],
                    "score_components": scored["score_components"],
                    "constraint_results": scored["constraint_results"],
                    "hard_gates": scored.get("hard_gates", []),
                    "perception_metrics": scored.get("perception_metrics"),
                    "metrics": result["metrics"],
                    "fidelity": result["fidelity"],
                    "requirement_gates": requirement,
                    "truth_boundary": result["truth_boundary"],
                }
            records.append(cache[key])
        robustness = None
        robustness_cases: list[dict[str, Any]] = []
        if include_robustness and study.spec.target_profile == "adas_yolo_perception":
            robustness, robustness_cases = self._robustness_score(
                study, parameters, scenes[:1], case_index=case_index
            )
        aggregate = self._aggregate_candidate_records(records, robustness=robustness)
        return {
            "case_id": f"case_{case_index:04d}" if case_index >= 0 else "baseline",
            "case_index": case_index,
            "parameters": parameters,
            **aggregate,
            "scene_count": len(records),
            "scene_metrics": records,
            "robustness_cases": robustness_cases,
            "evidence_state": "analytic_screened",
            "fidelity": records[0]["fidelity"] if records else {},
            "truth_boundary": records[0]["truth_boundary"] if records else "",
        }

    def _robustness_score(
        self,
        study: StudyRecord,
        parameters: dict[str, Any],
        scenes: list[SceneCase],
        *,
        case_index: int,
    ) -> tuple[float, list[dict[str, Any]]]:
        values = []
        cases = []
        for robust_index, robust_case in enumerate(study.spec.benchmark.robustness_cases):
            for scene in scenes:
                overrides, perturbed_scene = self._robustness_inputs(
                    study, parameters, scene, robust_case
                )
                result = self.engine.evaluate(
                    study.spec.baseline,
                    perturbed_scene,
                    fidelity=study.spec.fidelity_policy.search_level,
                    policy=study.spec.fidelity_policy,
                    seed=study.spec.seed + max(case_index, 0) * 10000 + 9000 + robust_index,
                    parameter_overrides=overrides,
                    include_arrays=True,
                )
                scored = self.score_result(study, result, scene=perturbed_scene)
                scored.pop("_overlay", None)
                perception = scored.get("perception_metrics") or {}
                value = min(
                    float(perception.get("map50_95", 0.0)),
                    float(perception.get("recall50", 0.0)),
                )
                values.append(value)
                cases.append(
                    {
                        "name": robust_case.name,
                        "kind": robust_case.kind,
                        "amount": robust_case.amount,
                        "scene_id": scene.id,
                        "score": value,
                        "perception_metrics": perception,
                        "simulation": "full_camera_pipeline",
                    }
                )
        return (float(np.mean(values)) if values else 0.0), cases

    @staticmethod
    def _robustness_inputs(
        study: StudyRecord,
        parameters: dict[str, Any],
        scene: SceneCase,
        robust_case: RobustnessCase,
    ) -> tuple[dict[str, Any], SceneCase]:
        overrides = dict(parameters)
        perturbed = scene
        if robust_case.kind == "exposure_ev":
            overrides["sensor.integration_time"] = (
                study.spec.baseline.sensor.exposure_ms * 1e-3 * (2.0**robust_case.amount)
            )
        elif robust_case.kind == "psf_scale":
            overrides["optics.si_psf_radius_um"] = (
                study.spec.baseline.lens.psf_radius_um * robust_case.amount
            )
        elif robust_case.kind == "ocl_equalization":
            overrides["sensor.ocl_group_equalization"] = robust_case.amount
        elif robust_case.kind == "low_light":
            perturbed = scene.model_copy(
                update={
                    "mean_luminance_cd_m2": max(
                        0.01, scene.mean_luminance_cd_m2 * robust_case.amount
                    )
                }
            )
        return overrides, perturbed

    @staticmethod
    def _aggregate_candidate_records(
        records: list[dict[str, Any]], *, robustness: float | None
    ) -> dict[str, Any]:
        if not records:
            return {
                "target_score": 0.0,
                "feasible": False,
                "score_components": {},
                "constraint_results": [],
                "hard_gates": [],
                "perception_metrics": None,
                "metrics": {},
            }
        component_names = sorted(
            {name for item in records for name in item.get("score_components", {})}
        )
        components = {}
        for name in component_names:
            rows = [
                item["score_components"][name]
                for item in records
                if name in item["score_components"]
            ]
            value = float(np.mean([float(row.get("value", 0.0)) for row in rows]))
            if name == "robustness" and robustness is not None:
                value = float(robustness)
            weight = float(rows[0].get("weight", 1.0)) if rows else 1.0
            components[name] = {
                "value": value,
                "weight": weight,
                "contribution": value * weight,
            }
        score = float(sum(item["contribution"] for item in components.values()))
        perception_rows = [
            item["perception_metrics"] for item in records if item.get("perception_metrics")
        ]
        perception = None
        if perception_rows:
            numeric_keys = {
                key
                for row in perception_rows
                for key, value in row.items()
                if isinstance(value, (int, float))
            }
            perception = {
                key: float(np.mean([float(row.get(key, 0.0)) for row in perception_rows]))
                for key in sorted(numeric_keys)
            }
            perception["prediction_count"] = int(
                sum(int(row.get("prediction_count", 0)) for row in perception_rows)
            )
            perception["ground_truth_count"] = int(
                sum(int(row.get("ground_truth_count", 0)) for row in perception_rows)
            )
        feasible = all(bool(item["feasible"]) for item in records)
        metric_groups: dict[str, dict[str, float]] = {}
        for group in {key for item in records for key in item.get("metrics", {})}:
            keys = {
                key
                for item in records
                for key, value in item.get("metrics", {}).get(group, {}).items()
                if isinstance(value, (int, float))
            }
            metric_groups[group] = {
                key: float(
                    np.mean(
                        [
                            float(item["metrics"][group][key])
                            for item in records
                            if isinstance(
                                item.get("metrics", {}).get(group, {}).get(key), (int, float)
                            )
                        ]
                    )
                )
                for key in keys
            }
        return {
            "target_score": score if feasible else 0.0,
            "feasible": feasible,
            "score_components": components,
            "constraint_results": [item for row in records for item in row["constraint_results"]],
            "hard_gates": [item for row in records for item in row.get("hard_gates", [])],
            "perception_metrics": perception,
            "metrics": metric_groups,
            "requirement_summary": {
                "all_scenes_feasible": all(
                    row.get("requirement_gates", {}).get("feasible", False) for row in records
                ),
                "failed_gate_ids": sorted(
                    {
                        gate["id"]
                        for row in records
                        for gate in row.get("requirement_gates", {}).get("gates", [])
                        if gate.get("hard") and gate.get("pass") is False
                    }
                ),
            },
        }

    @staticmethod
    def _apply_requirement_result(
        scored: dict[str, Any], requirement: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(scored)
        result["requirement_gates"] = requirement
        result["feasible"] = bool(
            result.get("feasible", False) and requirement.get("feasible", False)
        )
        if not result["feasible"]:
            result["target_score"] = 0.0
        return result

    def _require_preflight(self, study: StudyRecord) -> dict[str, Any]:
        preflight = self.preflight(study)
        if not preflight.get("ready", False):
            failed = [item["id"] for item in preflight.get("checks", []) if not item.get("pass")]
            raise PerceptionConfigurationError(
                "objective_degenerate: ADAS benchmark preflight failed: " + ", ".join(failed)
            )
        return preflight

    @staticmethod
    def _rank_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            cases,
            key=lambda item: (
                bool(item["feasible"]),
                float(item["target_score"]),
                -int(item["case_index"]),
            ),
            reverse=True,
        )

    @staticmethod
    def _training_recommendation(study: StudyRecord) -> dict[str, Any]:
        inventory = benchmark_inventory(study)
        root = inventory.get("root")
        return {
            "required": True,
            "epochs": 30,
            "base_model": study.spec.perception_model_path or "yolo11n.pt",
            "dataset_yaml": None if root is None else str(Path(str(root)) / "data.yaml"),
            "device": "mps_or_cpu",
            "acceptance": {"map50": 0.50, "map50_95": 0.30, "recall50": 0.50},
            "note": "Training is explicit because it can consume substantial local compute.",
        }

    @staticmethod
    def _read_scene_rgb(scene: SceneCase) -> np.ndarray:
        if not scene.image_path:
            raise ValueError(f"Scene {scene.id} has no RGB image path")
        image = np.asarray(iio.imread(Path(scene.image_path).expanduser()), dtype=float)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3]
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0)

    @classmethod
    def _ground_truth(cls, labels: dict[str, Any]) -> list[TaskBoundingBox]:
        ground_truth = []
        for item in labels.get("objects", []):
            class_id = int(item.get("class_id", -1))
            mapped = cls._adas_label(str(item.get("label", "")))
            if class_id not in ADAS_KITTI_CLASS_IDS or mapped not in {"car", "person", "bicycle"}:
                continue
            coords = [float(value) for value in item["bbox_xyxy"]]
            ground_truth.append(
                TaskBoundingBox((coords[0], coords[1], coords[2], coords[3]), label=mapped)
            )
        return ground_truth

    @classmethod
    def _perception_metrics(
        cls, detector: Any, image: np.ndarray, ground_truth: list[TaskBoundingBox]
    ) -> tuple[dict[str, float], list[TaskBoundingBox]]:
        config = task_perception_config(
            iou_threshold=0.5,
            score_threshold=0.25,
            map_iou_thresholds=tuple(np.arange(0.5, 1.0, 0.05)),
        )
        predictions = [
            TaskBoundingBox(box.xyxy, label=cls._adas_label(box.label), score=box.score)
            for box in detector.detect(image, config)
            if cls._adas_label(box.label) in {"car", "person", "bicycle"}
        ]
        map_result = mean_average_precision(predictions, ground_truth, config)
        detection = detection_metrics(predictions, ground_truth, config, iou_threshold=0.5)
        small_gt = [box for box in ground_truth if box.area <= 32.0 * 32.0]
        small = (
            detection_metrics(predictions, small_gt, config, iou_threshold=0.5)
            if small_gt
            else detection
        )
        return (
            {
                "map50_95": float(map_result.get("map", 0.0)),
                "map50": float(map_result.get("ap50", 0.0)),
                "recall50": cls._balanced_recall(detection),
                "small_object_recall50": float(small.get("recall", 0.0)),
                "mean_iou": float(detection.get("mean_matched_iou", 0.0)),
                "prediction_count": float(len(predictions)),
                "ground_truth_count": float(len(ground_truth)),
            },
            predictions,
        )

    @staticmethod
    def _aggregate_perception(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = sorted({key for row in rows for key in row})
        result = {
            key: float(np.mean([row.get(key, 0.0) for row in rows])) if rows else 0.0
            for key in keys
        }
        result["prediction_count"] = float(sum(row.get("prediction_count", 0.0) for row in rows))
        result["ground_truth_count"] = float(
            sum(row.get("ground_truth_count", 0.0) for row in rows)
        )
        return result

    @staticmethod
    def _bbox_transform_error(
        source: list[TaskBoundingBox],
        target: list[TaskBoundingBox],
        transform: Any,
    ) -> float:
        errors: list[float] = []
        for source_box, target_box in zip(source, target, strict=False):
            expected = transform.transform_bbox(list(source_box.xyxy))
            errors.extend(
                abs(float(a) - float(b)) for a, b in zip(expected, target_box.xyxy, strict=True)
            )
        return max(errors, default=0.0)

    def _detector(self, model_path: str) -> Any:
        key = str(Path(model_path).expanduser().resolve())
        if key not in self._detectors:
            self._detectors[key] = task_model_from_config(
                {
                    "name": "camerae2e_v2_adas_yolo",
                    "backend": "ultralytics_yolo",
                    "task": "detection",
                    "model_id": key,
                    "device": "cpu",
                    "score_threshold": 0.25,
                }
            )
        return self._detectors[key]

    @staticmethod
    def _stage_image(result: dict[str, Any], *names: str) -> np.ndarray | None:
        for name in names:
            stage = result.get("stages", {}).get(name, {})
            if "array" in stage:
                array = np.asarray(stage["array"], dtype=float)
                if array.size:
                    return array
        return None

    @staticmethod
    def _adas_label(label: str) -> str:
        key = label.strip().lower()
        if key in {"car", "van", "truck", "tram", "bus"}:
            return "car"
        if key in {"pedestrian", "person_sitting", "person"}:
            return "person"
        if key in {"cyclist", "bicycle", "bike"}:
            return "bicycle"
        return key

    @staticmethod
    def _balanced_recall(payload: dict[str, Any]) -> float:
        values = [float(item.get("recall", 0.0)) for item in payload.get("per_label", {}).values()]
        return float(np.mean(values)) if values else float(payload.get("recall", 0.0))

    @staticmethod
    def _pareto(cases: list[dict[str, Any]], objectives: list[Objective]) -> list[dict[str, Any]]:
        if not cases:
            return []
        if any(case.get("perception_metrics") for case in cases):
            front = []
            for index, case in enumerate(cases):
                score = float(case.get("target_score", 0.0))
                risk = float(
                    case.get("metrics", {})
                    .get("artifact", {})
                    .get(
                        "rgb_high_clip_fraction",
                        case.get("metrics", {})
                        .get("artifact", {})
                        .get("rgb_clip_fraction", 1.0),
                    )
                )
                dominated = any(
                    other_index != index
                    and float(other.get("target_score", 0.0)) >= score
                    and float(
                        other.get("metrics", {})
                        .get("artifact", {})
                        .get(
                            "rgb_high_clip_fraction",
                            other.get("metrics", {})
                            .get("artifact", {})
                            .get("rgb_clip_fraction", 1.0),
                        )
                    )
                    <= risk
                    and (
                        float(other.get("target_score", 0.0)) > score
                        or float(
                            other.get("metrics", {})
                            .get("artifact", {})
                            .get(
                                "rgb_high_clip_fraction",
                                other.get("metrics", {})
                                .get("artifact", {})
                                .get("rgb_clip_fraction", 1.0),
                            )
                        )
                        < risk
                    )
                    for other_index, other in enumerate(cases)
                )
                if not dominated:
                    front.append(case)
            return front
        metric_directions = {objective.metric: objective.direction for objective in objectives}

        def vector(case: dict[str, Any]) -> dict[str, float]:
            report = {"metrics": case["metrics"]}
            values = {}
            for metric, direction in metric_directions.items():
                value = numeric_path(report, metric)
                values[metric] = -value if direction == "minimize" else value
            return values

        front = []
        vectors = [vector(case) for case in cases]
        for index, case in enumerate(cases):
            dominated = False
            for other_index, _other in enumerate(cases):
                if index == other_index:
                    continue
                left = vectors[other_index]
                right = vectors[index]
                if all(left[key] >= right[key] for key in right) and any(
                    left[key] > right[key] for key in right
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(case)
        return front


def variable_values(variable: DesignVariable) -> list[Any]:
    if variable.values is not None:
        return list(variable.values)
    assert variable.minimum is not None and variable.maximum is not None
    return np.linspace(variable.minimum, variable.maximum, variable.steps).tolist()


_KITTI_CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc",
}


def load_adas_label_payload(
    source: str | Path,
    *,
    image_size: tuple[int, int] | list[int],
) -> dict[str, Any]:
    """Load either original KITTI rows or Ultralytics YOLO-normalized KITTI rows."""

    path = Path(source).expanduser().resolve()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, cols = int(image_size[0]), int(image_size[1])
    if not lines:
        return {
            "schema_version": "camerae2e_adas_labels_v2",
            "source": str(path),
            "source_format": "empty",
            "image_size_rc": [rows, cols],
            "objects": [],
        }
    first = lines[0].split()
    if len(first) >= 8:
        from pyisetcam import camerae2e_kitti_yolo_labels

        payload = camerae2e_kitti_yolo_labels(path, image_size=(rows, cols))
        payload["schema_version"] = "camerae2e_adas_labels_v2"
        payload["source_format"] = "kitti_object"
        return payload
    objects = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label row must be KITTI object or YOLO class/xywhn format: {line!r}")
        class_id = int(float(parts[0]))
        cx, cy, width, height = [float(value) for value in parts[1:]]
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            raise ValueError(f"YOLO label contains non-finite coordinates: {line!r}")
        x1 = (cx - width / 2.0) * cols
        y1 = (cy - height / 2.0) * rows
        x2 = (cx + width / 2.0) * cols
        y2 = (cy + height / 2.0) * rows
        objects.append(
            {
                "label": _KITTI_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "yolo_xywhn": [cx, cy, width, height],
                "yolo": {"line": line},
            }
        )
    return {
        "schema_version": "camerae2e_adas_labels_v2",
        "source": str(path),
        "source_format": "yolo_xywhn",
        "image_size_rc": [rows, cols],
        "class_map": _KITTI_CLASS_NAMES,
        "objects": objects,
        "masks": [],
    }
