from __future__ import annotations

import base64
import io
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyisetcam import (  # noqa: E402
    camerae2e_adas_camera_spec,
    camerae2e_dataset_export_adas_kitti_demo,
    camerae2e_dataset_export_from_optimization,
    camerae2e_dataset_validate,
    camerae2e_db_manifest,
    camerae2e_db_validate,
    camerae2e_faca_report,
    camerae2e_optimization_report,
    camerae2e_optimize_camera_parameters,
    camerae2e_parameter_space_catalog,
    camerae2e_run_scenario,
    fdtd_sensor_default_lut_path,
    tcad_sensor_db_load,
    tcad_sensor_default_paths,
)


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
                    "objective": "balanced_faca",
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
                "default_preset": "raw_factory",
                "default_method": "latin_hypercube",
                "default_max_candidates": 24,
                "objective_presets": [
                    {
                        "id": "balanced_faca",
                        "name": "Balanced FACA",
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
        return {
            "goal": "adas_raw_factory",
            "cameraPreset": "kitti_yolo_demo",
            "sceneType": "macbeth",
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
            "optimizationPreset": "raw_factory",
            "optimizationMethod": "latin_hypercube",
            "maxCandidates": 24,
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
        return {
            "schema_version": "camerae2e_workbench_assets_status_v1",
            "ok": bool(validation.get("ok", False)),
            "validation": _safe_json(validation),
            "assets": {
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
                        "Bundled lens DB and geometric PSF metadata; diffraction is a separate comparison."
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
                    "truth_boundary": "Collection-response framework; calibration and lineage are not closed.",
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

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._merged_settings(payload.get("settings", payload))
        started = time.perf_counter()
        scenario = self._scenario_from_settings(settings)
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
        report = camerae2e_optimization_report(result)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = {
            "schema_version": "camerae2e_workbench_optimization_result_v1",
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
            "pareto_points": _pareto_points(result.get("pareto_front", []), result.get("top_cases", [])),
            "fidelity_badges": self._fidelity_badges(settings),
            "truth_boundary": self._truth_boundary(settings),
        }
        self.latest_optimization = result
        self._write_json(self.run_root / "latest" / "optimization_result.json", response)
        return response

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
                spec=camerae2e_adas_camera_spec(str(settings.get("cameraPreset", "kitti_yolo_demo"))),
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
        json_path.write_text(json.dumps(_safe_json(report), indent=2, sort_keys=True), encoding="utf-8")
        html_path.write_text(_html_report(report), encoding="utf-8")
        response = {
            "schema_version": "camerae2e_workbench_report_result_v1",
            "json_path": str(json_path),
            "html_path": str(html_path),
            "report": _safe_json(report),
        }
        self.latest_report = response
        return response

    def _scenario_from_settings(
        self, settings: dict[str, Any], *, include_hw_isp: bool | None = None
    ) -> dict[str, Any]:
        pixel_pitch_m = float(settings["pixelSizeUm"]) * 1.0e-6
        focal_length_m = _focal_length_from_hfov(
            hfov_deg=float(settings["fovDeg"]),
            pixel_pitch_m=pixel_pitch_m,
        )
        scene = _scene_payload(str(settings.get("sceneType", "macbeth")))
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
        use_hw = bool(settings.get("hwIspEnabled", False)) if include_hw_isp is None else include_hw_isp
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
            "sensor.integration_time": _unique_sorted([exposure_s * 0.5, exposure_s, exposure_s * 1.5]),
            "sensor.analog_gain": _unique_sorted([1.0, float(settings.get("analogGain", 1.0)), 2.0]),
            "sensor.pixel_size": _unique_sorted(
                [(pixel_um * 0.8) * 1e-6, pixel_um * 1e-6, (pixel_um * 1.2) * 1e-6]
            ),
            "sensor.cfa_preset": ["bayer_rgb", "quad_bayer_rgb"],
            "sensor.ocl_vignetting": ["centered", "optimal"],
            "sensor.ocl_group_shape": ["1x1", "2x2"],
            "sensor.ocl_group_equalization": [0.0, 0.5, 1.0],
            "optics.fnumber": _unique_sorted([max(1.2, float(settings["fNumber"]) - 0.4), float(settings["fNumber"]), float(settings["fNumber"]) + 0.6]),
            "optics.si_psf_radius_um": _unique_sorted([max(0.5, psf * 0.5), psf, psf * 1.5]),
        }
        if int(settings.get("binningFactor", 1)) > 1:
            axes["sensor.binning_factor"] = [1, 2]
        if bool(settings.get("fdtdEnabled", False)):
            axes["fdtd.mode"] = [str(settings.get("fdtdMode", "qe+field"))]
            axes["fdtd.crosstalk_strength"] = [0.0, float(settings.get("fdtdCrosstalkStrength", 0.5))]
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


def _scene_payload(scene_type: str) -> dict[str, Any]:
    key = scene_type.strip().lower().replace("_", " ")
    if key == "slanted bar":
        return {"type": "slanted bar", "args": [64]}
    if key == "uniform":
        return {"type": "uniform ee", "args": [8]}
    return {"type": "macbeth"}


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
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3:
            continue
        finite = np.nan_to_num(image, copy=False)
        if finite.max() > finite.min():
            if finite.max() > 1.0 or finite.min() < 0.0:
                finite = (finite - finite.min()) / (finite.max() - finite.min())
        finite = np.clip(finite, 0.0, 1.0)
        buffer = io.BytesIO()
        iio.imwrite(buffer, (finite * 255.0).astype(np.uint8), extension=".png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return None


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
        "feasible": bool(case.get("feasible", True)),
        "parameters": _compact_parameters(params),
        "objective_values": {key: _round_float(value) for key, value in values.items()},
        "constraint_results": case.get("constraint_results", []),
    }


def _pareto_points(front: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #15202b; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    td, th {{ border: 1px solid #d7dee8; padding: 8px; text-align: left; }}
    .boundary {{ background: #fff7e6; border: 1px solid #e5b35b; padding: 12px; border-radius: 6px; }}
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
