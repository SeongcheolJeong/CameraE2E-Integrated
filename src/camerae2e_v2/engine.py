"""Stateless CameraE2E stage adapter and fidelity routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from pyisetcam import (
    camerae2e_db_manifest,
    camerae2e_db_validate,
    camerae2e_faca_report,
    camerae2e_run_scenario,
    fdtd_sensor_default_lut_path,
    scene_from_file,
    sensor_create,
    sensor_get,
)

from .geometry import (
    derive_geometry_transform,
    geometry_contract_metrics,
    global_ssim,
    ideal_recapture,
)
from .models import (
    CameraModule,
    FidelityLevel,
    FidelityPolicy,
    ReadinessTier,
    SceneCase,
)


@dataclass(frozen=True)
class FidelityDecision:
    requested: FidelityLevel
    effective: FidelityLevel
    backend: str
    readiness_tier: ReadinessTier
    assets: tuple[str, ...]
    warnings: tuple[str, ...]
    truth_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested.value,
            "effective": self.effective.value,
            "backend": self.backend,
            "readiness_tier": self.readiness_tier.value,
            "assets": list(self.assets),
            "warnings": list(self.warnings),
            "truth_boundary": self.truth_boundary,
        }


class FidelityRouter:
    """Choose the strongest usable backend without inflating claims."""

    def resolve(
        self,
        requested: FidelityLevel,
        policy: FidelityPolicy,
        module: CameraModule,
    ) -> FidelityDecision:
        if requested == FidelityLevel.ANALYTIC:
            return FidelityDecision(
                requested=requested,
                effective=FidelityLevel.ANALYTIC,
                backend="pyisetcam_analytic",
                readiness_tier=ReadinessTier.VALIDATED,
                assets=(),
                warnings=(),
                truth_boundary=(
                    "Fast analytic CameraE2E model for research ranking. It is not a "
                    "measured camera-module sign-off result."
                ),
            )

        if requested == FidelityLevel.LUT:
            configured = module.metadata.get("fdtd_lut_path")
            lut = (
                Path(str(configured)).expanduser() if configured else fdtd_sensor_default_lut_path()
            )
            if lut is not None and Path(lut).is_file():
                validation = camerae2e_db_validate(strict=False)
                warnings = tuple(
                    str(item.get("message", item.get("kind", "registry warning")))
                    for item in validation.get("warnings", [])
                    if str(item.get("entry", "")) == "fdtd_sensor_lut_active"
                )
                if policy.require_fresh_lineage and any(
                    item.get("kind") == "stale_dependency"
                    for item in validation.get("warnings", [])
                    if str(item.get("entry", "")) == "fdtd_sensor_lut_active"
                ):
                    if not policy.allow_proxy:
                        raise ValueError(
                            "FDTD LUT has stale lineage and proxy fallback is disabled"
                        )
                return FidelityDecision(
                    requested=requested,
                    effective=FidelityLevel.LUT,
                    backend="pyisetcam_fdtd_lut",
                    readiness_tier=ReadinessTier.PROXY,
                    assets=(str(Path(lut).resolve()),),
                    warnings=warnings,
                    truth_boundary=(
                        "FDTD LUT-backed optical response is active. The bundled LUT is "
                        "research/proxy evidence unless calibrated lineage is attached."
                    ),
                )
            if policy.allow_proxy:
                return FidelityDecision(
                    requested=requested,
                    effective=FidelityLevel.ANALYTIC,
                    backend="pyisetcam_analytic",
                    readiness_tier=ReadinessTier.PROXY,
                    assets=(),
                    warnings=("Requested LUT was unavailable; analytic fallback was recorded.",),
                    truth_boundary="Analytic fallback; no LUT-backed claim is permitted.",
                )
            raise FileNotFoundError("No FDTD LUT is available for L1_lut evaluation")

        if requested == FidelityLevel.SOLVER:
            solver_artifact = module.metadata.get("solver_lut_path")
            if solver_artifact and Path(str(solver_artifact)).expanduser().is_file():
                return FidelityDecision(
                    requested=requested,
                    effective=FidelityLevel.SOLVER,
                    backend="solver_generated_lut",
                    readiness_tier=ReadinessTier.CALIBRATION_REQUIRED,
                    assets=(str(Path(str(solver_artifact)).expanduser().resolve()),),
                    warnings=(),
                    truth_boundary=(
                        "A solver-generated artifact is attached. Quantitative accuracy still "
                        "requires convergence and measured calibration evidence."
                    ),
                )
            raise ValueError(
                "L2_solver evaluation requires a completed solver artifact. "
                "Submit candidate validation first."
            )

        evidence = module.metadata.get("calibration_evidence")
        if not evidence or not Path(str(evidence)).expanduser().is_file():
            raise ValueError("L3_calibrated evaluation requires calibration evidence")
        return FidelityDecision(
            requested=requested,
            effective=FidelityLevel.CALIBRATED,
            backend="calibrated_camera_model",
            readiness_tier=ReadinessTier.CALIBRATED,
            assets=(str(Path(str(evidence)).expanduser().resolve()),),
            warnings=(),
            truth_boundary="Measured calibration evidence is attached to this model snapshot.",
        )


class CameraEngine:
    """Convert v2 domain objects into one real pyisetcam execution."""

    def __init__(self, router: FidelityRouter | None = None) -> None:
        self.router = router or FidelityRouter()

    def evaluate(
        self,
        module: CameraModule,
        scene: SceneCase,
        *,
        fidelity: FidelityLevel,
        policy: FidelityPolicy,
        seed: int,
        parameter_overrides: dict[str, Any] | None = None,
        include_arrays: bool = True,
    ) -> dict[str, Any]:
        effective_module = self.apply_module_overrides(module, parameter_overrides or {})
        decision = self.router.resolve(fidelity, policy, effective_module)
        scenario = self.to_scenario(
            effective_module, decision, parameter_overrides=parameter_overrides
        )
        scenario.setdefault("sensor", {})["noise_seed"] = int(seed)
        scene_object = self.resolve_scene(scene)
        result = camerae2e_run_scenario(
            scenario,
            scene=scene_object,
            seed=int(seed),
            include_arrays=include_arrays,
        )
        report = camerae2e_faca_report(result)
        stages = dict(result.get("stages", {})) if include_arrays else {}
        output = self._stage_array(stages, "ip_srgb", "ip_result")
        raw = self._stage_array(stages, "sensor_raw", "sensor_digital")
        metrics = dict(report.get("metrics", {}))
        if output is not None:
            artifact_metrics = dict(metrics.get("artifact", {}))
            artifact_metrics["rgb_low_clip_fraction"] = float(np.mean(output <= 0.0))
            artifact_metrics["rgb_high_clip_fraction"] = float(np.mean(output >= 1.0))
            metrics["artifact"] = artifact_metrics
        geometry = None
        color_diagnostics: dict[str, Any] = {}
        if output is not None:
            readout_size = output.shape[:2] if raw is None else raw.shape[:2]
            transform = derive_geometry_transform(
                effective_module,
                scene,
                output_size_rc=output.shape[:2],
                readout_size_rc=readout_size,
            )
            geometry = {
                "transform": transform.model_dump(mode="json"),
                "metrics": geometry_contract_metrics(effective_module, transform),
            }
            if scene.scene_type == "rgb_file" and scene.image_path:
                source = self._read_rgb(Path(scene.image_path).expanduser().resolve())
                ideal = ideal_recapture(source, transform.output_size_rc)
                color_diagnostics = self._color_diagnostics(source, ideal, output)
                if include_arrays:
                    stages["ideal_recapture"] = {
                        "array": ideal,
                        "shape": list(ideal.shape),
                        "dtype": str(ideal.dtype),
                        "truth_boundary": "geometry_only_reference_not_camera_simulation",
                    }
        return {
            "schema_version": "camerae2e_evaluation_v2",
            "seed": int(seed),
            "module": effective_module.model_dump(mode="json"),
            "scene": scene.model_dump(mode="json"),
            "fidelity": decision.to_dict(),
            "scenario": report.get("scenario", {}),
            "metrics": metrics,
            "stage_summaries": report.get("stage_summaries", {}),
            "parameter_lineage": report.get("parameter_lineage", []),
            "artifact_lineage": report.get("artifact_lineage", {}),
            "geometry": geometry,
            "color_diagnostics": color_diagnostics,
            "stages": stages,
            "truth_boundary": self._truth_boundary(scene, decision),
        }

    @staticmethod
    def apply_module_overrides(module: CameraModule, overrides: dict[str, Any]) -> CameraModule:
        lens_updates: dict[str, Any] = {}
        sensor_updates: dict[str, Any] = {}
        isp_updates: dict[str, Any] = {}
        mapping: dict[str, tuple[dict[str, Any], str, float]] = {
            "sensor.integration_time": (sensor_updates, "exposure_ms", 1000.0),
            "sensor.pixel_size": (sensor_updates, "pixel_size_um", 1e6),
            "sensor.cfa_preset": (sensor_updates, "cfa_preset", 1.0),
            "sensor.binning_factor": (sensor_updates, "binning_factor", 1.0),
            "sensor.rows": (sensor_updates, "rows", 1.0),
            "sensor.cols": (sensor_updates, "cols", 1.0),
            "sensor.ocl_group_equalization": (sensor_updates, "ocl_equalization", 1.0),
            "sensor.analog_gain": (sensor_updates, "analog_gain", 1.0),
            "optics.fnumber": (lens_updates, "f_number", 1.0),
            "optics.focal_length": (lens_updates, "focal_length_mm", 1000.0),
            "optics.si_psf_radius_um": (lens_updates, "psf_radius_um", 1.0),
            "ip.sensor_conversion_method": (isp_updates, "ccm_method", 1.0),
            "ip.sensor_conversion_matrix": (isp_updates, "ccm_matrix", 1.0),
            "ip.demosaic_method": (isp_updates, "demosaic_method", 1.0),
        }
        for path, value in overrides.items():
            target = mapping.get(path)
            if target is None:
                continue
            bucket, field, scale = target
            if isinstance(value, (int, float)) and scale != 1.0:
                bucket[field] = float(value) * scale
            else:
                bucket[field] = value
        return module.model_copy(
            update={
                "lens": module.lens.model_copy(update=lens_updates),
                "sensor": module.sensor.model_copy(update=sensor_updates),
                "isp": module.isp.model_copy(update=isp_updates),
            }
        )

    def to_scenario(
        self,
        module: CameraModule,
        decision: FidelityDecision,
        *,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if module.isp.ccm_matrix is not None:
            calibration = module.metadata.get("ccm_calibration_evidence")
            if not calibration:
                raise ValueError(
                    "Free 3x3 CCM requires color calibration evidence; "
                    "use a validated CCM method otherwise"
                )
        sensor = module.sensor
        lens = module.lens
        focal_m = (
            float(lens.focal_length_mm) * 1e-3
            if lens.focal_length_mm is not None
            else self._focal_length_from_hfov(
                hfov_deg=lens.hfov_deg,
                pixel_pitch_m=sensor.pixel_size_um * 1e-6,
                cols=sensor.cols,
            )
        )
        sensor_payload: dict[str, Any] = {
            "noise_flag": 2,
            "rows": sensor.rows,
            "cols": sensor.cols,
            "integration_time": sensor.exposure_ms * 1e-3,
            "analog_gain": sensor.analog_gain,
            "pixel_size": sensor.pixel_size_um * 1e-6,
            "cfa_preset": sensor.cfa_preset,
            "ocl_group_shape": sensor.ocl_group_shape,
            "ocl_group_equalization": sensor.ocl_equalization,
            "binning_factor": sensor.binning_factor,
        }
        sensor_payload.update(self._qe_profile_overrides(sensor.qe_profile))
        default_sensor = sensor_create()
        if isinstance(default_sensor, list):
            default_sensor = default_sensor[0]
        conversion_gain = float(sensor_get(default_sensor, "pixel conversion gain"))
        if sensor.read_noise_e is not None:
            sensor_payload["pixel_read_noise_v"] = sensor.read_noise_e * conversion_gain
        if sensor.full_well_e is not None:
            sensor_payload["pixel_voltage_swing"] = sensor.full_well_e * conversion_gain
        if sensor.ocl_mode.strip().lower() not in {"off", "none", "disabled"}:
            sensor_payload["ocl_vignetting"] = sensor.ocl_mode
        scenario: dict[str, Any] = {
            "name": f"camerae2e_v2_{module.id}",
            "sensor": sensor_payload,
            "parameters": {
                "optics.fnumber": lens.f_number,
                "optics.focal_length": focal_m,
                "optics.si_psf_radius_um": lens.psf_radius_um,
                "ip.demosaic_method": module.isp.demosaic_method,
                "ip.sensor_conversion_method": module.isp.ccm_method,
            },
            "geometry": {
                "scene_hfov_deg": lens.hfov_deg,
                "sensor_resize": False,
                "requested_sensor_size_rc": [sensor.rows, sensor.cols],
            },
            "camerae2e_v2": {
                "module_id": module.id,
                "hfov_deg": lens.hfov_deg,
                "fidelity": decision.to_dict(),
                "qe_profile": sensor.qe_profile,
            },
        }
        if module.isp.ccm_matrix is not None:
            scenario["parameters"]["ip.sensor_conversion_matrix"] = np.asarray(
                module.isp.ccm_matrix, dtype=float
            )
        if decision.effective in {FidelityLevel.LUT, FidelityLevel.SOLVER}:
            scenario["fdtd"] = {
                "lut": decision.assets[0],
                "mode": str(module.metadata.get("fdtd_mode", "qe+field")),
                "crosstalk_strength": float(module.metadata.get("fdtd_crosstalk_strength", 0.5)),
            }
        if module.hw_isp_enabled:
            scenario["hw_isp"] = {"enabled": True, "nframes": module.hw_isp_frames}
        for path, value in dict(parameter_overrides or {}).items():
            self._assign_parameter(scenario, path, value)
        return scenario

    def resolve_scene(self, scene: SceneCase) -> Any:
        if scene.scene_type == "rgb_file":
            path = Path(str(scene.image_path)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            image = np.asarray(iio.imread(path))
            return scene_from_file(image, "rgb", mean_luminance=scene.mean_luminance_cd_m2)
        if scene.scene_type == "multispectral_file":
            path = Path(str(scene.image_path)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return scene_from_file(path, "multispectral", mean_luminance=scene.mean_luminance_cd_m2)
        if scene.scene_type == "slanted_bar":
            return {"type": "slanted bar", "args": [64]}
        if scene.scene_type == "uniform":
            return {"type": "uniform ee", "args": [16]}
        return {"type": "macbeth"}

    def asset_status(self) -> dict[str, Any]:
        manifest = camerae2e_db_manifest()
        validation = camerae2e_db_validate(strict=False)
        return {
            "schema_version": "camerae2e_asset_status_v2",
            "manifest_summary": manifest.get("summary", {}),
            "validation": validation,
            "fidelity_levels": [level.value for level in FidelityLevel],
            "qe_profiles": {
                "isetcam_default_rgb": {"available": True, "readiness": "validated"},
                "sony_imx363": {"available": True, "readiness": "proxy"},
                "onsemi_ar0132at": {"available": True, "readiness": "proxy"},
            },
        }

    @staticmethod
    def _stage_array(stages: dict[str, Any], *names: str) -> np.ndarray | None:
        for name in names:
            stage = stages.get(name, {})
            if isinstance(stage, dict) and "array" in stage:
                array = np.asarray(stage["array"], dtype=float)
                if array.size:
                    return array
        return None

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = np.asarray(iio.imread(path), dtype=float)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3]
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0)

    @classmethod
    def _color_diagnostics(
        cls, source: np.ndarray, ideal: np.ndarray, output: np.ndarray
    ) -> dict[str, Any]:
        rendered = np.asarray(output, dtype=float)
        if rendered.ndim == 2:
            rendered = np.repeat(rendered[..., None], 3, axis=2)
        rendered = np.clip(rendered[..., :3], 0.0, 1.0)
        source_mean = np.mean(source[..., :3], axis=(0, 1))
        output_mean = np.mean(rendered, axis=(0, 1))
        ideal_mean = np.mean(ideal, axis=(0, 1))
        relative_channel_gain = output_mean / np.maximum(ideal_mean, 1e-12)
        return {
            "source_mean_rgb": source_mean.tolist(),
            "ideal_mean_rgb": ideal_mean.tolist(),
            "output_mean_rgb": output_mean.tolist(),
            "output_channel_imbalance": float(
                np.std(output_mean) / max(float(np.mean(output_mean)), 1e-12)
            ),
            "relative_channel_gain": relative_channel_gain.tolist(),
            "relative_channel_gain_imbalance": float(
                np.std(relative_channel_gain)
                / max(float(np.mean(relative_channel_gain)), 1e-12)
            ),
            "ideal_output_ssim": global_ssim(ideal, rendered),
            "mean_absolute_color_error": float(np.mean(np.abs(ideal - rendered))),
            "qe_profile_applied": True,
        }

    @staticmethod
    def _qe_profile_overrides(profile: str) -> dict[str, Any]:
        normalized = profile.strip().lower().replace("-", "_")
        if normalized in {"", "default", "isetcam_default_rgb"}:
            return {}
        sensor_names = {
            "sony_imx363": "IMX363",
            "imx363": "IMX363",
            "onsemi_ar0132at": "ar0132at",
            "ar0132at": "ar0132at",
        }
        if normalized in sensor_names:
            source = sensor_create(sensor_names[normalized])
            if isinstance(source, list):
                source = source[0]
            target = sensor_create()
            if isinstance(target, list):
                target = target[0]
            source_wave = np.asarray(sensor_get(source, "wave"), dtype=float)
            target_wave = np.asarray(sensor_get(target, "wave"), dtype=float)
            source_qe = np.asarray(sensor_get(source, "spectral qe"), dtype=float)
            interpolated = np.column_stack(
                [
                    np.interp(target_wave, source_wave, source_qe[:, index], left=0.0, right=0.0)
                    for index in range(source_qe.shape[1])
                ]
            )
            return {
                "filter_spectra": interpolated,
                "filter_names": list(sensor_get(source, "filter names")),
                "pixel_spectral_qe": np.ones(target_wave.size, dtype=float),
                "ir_filter": np.ones(target_wave.size, dtype=float),
            }
        path = Path(profile).expanduser()
        if path.is_file() and path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if "filter_spectra" not in payload:
                    raise ValueError("Custom QE NPZ requires filter_spectra")
                result: dict[str, Any] = {
                    "filter_spectra": np.asarray(payload["filter_spectra"], dtype=float)
                }
                if "pixel_qe" in payload:
                    result["pixel_spectral_qe"] = np.asarray(payload["pixel_qe"], dtype=float)
                if "ir_filter" in payload:
                    result["ir_filter"] = np.asarray(payload["ir_filter"], dtype=float)
                return result
        raise ValueError(
            f"Unknown QE profile {profile!r}. Use isetcam_default_rgb, sony_imx363, "
            "onsemi_ar0132at, or a validated NPZ profile."
        )

    @staticmethod
    def _focal_length_from_hfov(*, hfov_deg: float, pixel_pitch_m: float, cols: int) -> float:
        half_width_m = float(cols) * float(pixel_pitch_m) / 2.0
        return float(half_width_m / math.tan(math.radians(float(hfov_deg)) / 2.0))

    @staticmethod
    def _assign_parameter(scenario: dict[str, Any], path: str, value: Any) -> None:
        parts = str(path).split(".", 1)
        if len(parts) == 2 and parts[0] in {"sensor", "fdtd", "tcad", "hw_isp"}:
            scenario.setdefault(parts[0], {})[parts[1]] = value
        else:
            scenario.setdefault("parameters", {})[path] = value

    @staticmethod
    def _truth_boundary(scene: SceneCase, decision: FidelityDecision) -> str:
        scene_note = {
            "physical": "The scene carries physical/spectral provenance.",
            "measured_proxy": "The scene is a measured proxy with caller calibration metadata.",
            "display_rgb_proxy": (
                "The source RGB frame is a display-derived re-capture proxy; lost spectral and "
                "radiometric information is not recovered."
            ),
            "synthetic": "The scene is synthetic and suitable for controlled research comparisons.",
        }[scene.source_kind]
        return f"{decision.truth_boundary} {scene_note}"
