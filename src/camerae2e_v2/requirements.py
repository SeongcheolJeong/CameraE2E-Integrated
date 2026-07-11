"""Engineering requirement gates derived from one camera evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import CameraModule, RequirementGateResult, StudyRecord


def evaluate_requirement_gates(
    study: StudyRecord,
    result: dict[str, Any],
) -> RequirementGateResult:
    spec = study.spec
    module = CameraModule.model_validate(result.get("module", spec.baseline.model_dump()))
    requirements = spec.requirements
    geometry = dict((result.get("geometry") or {}).get("metrics", {}))
    transform = dict((result.get("geometry") or {}).get("transform", {}))
    output_rows, output_cols = _size(
        transform.get("output_size_rc"), (module.sensor.rows, module.sensor.cols)
    )
    active_rows, active_cols = _size(
        transform.get("active_sensor_size_rc"), (module.sensor.rows, module.sensor.cols)
    )
    native_rows = module.sensor.native_rows or active_rows
    native_cols = module.sensor.native_cols or active_cols
    pixel_pitch_mm = module.sensor.pixel_size_um * 1e-3
    effective_pitch_mm = pixel_pitch_mm * module.sensor.binning_factor
    sensor_width_mm = float(geometry.get("sensor_width_mm", native_cols * pixel_pitch_mm))
    sensor_height_mm = float(geometry.get("sensor_height_mm", native_rows * pixel_pitch_mm))
    focal_mm = geometry.get("focal_length_mm")
    if focal_mm is None:
        focal_mm = sensor_width_mm / (2.0 * math.tan(math.radians(module.lens.hfov_deg) / 2.0))
    focal_mm = float(focal_mm)
    derived_hfov = float(geometry.get("derived_hfov_deg", module.lens.hfov_deg))
    sensor_diagonal = math.hypot(sensor_width_mm, sensor_height_mm)
    object_pixels = (
        focal_mm
        * requirements.reference_object_height.value
        / max(requirements.detection_range.value, 1e-12)
        / max(effective_pitch_mm, 1e-12)
    )
    airy_diameter_um = 2.44 * requirements.reference_wavelength_nm * 1e-3 * module.lens.f_number
    airy_diameter_pixels = airy_diameter_um / max(
        module.sensor.pixel_size_um * module.sensor.binning_factor, 1e-12
    )
    fps = module.sensor.frame_rate_fps
    pixel_rate_mpix_s = native_rows * native_cols * fps / 1e6
    rolling_shutter_ms = native_rows * module.sensor.row_time_us / 1000.0
    isp_latency_ms = module.hw_isp_frames * 1000.0 / fps if module.hw_isp_enabled else 0.0
    total_latency_ms = rolling_shutter_ms + isp_latency_ms
    artifact_metrics = result.get("metrics", {}).get("artifact", {})
    clip = float(
        artifact_metrics.get(
            "rgb_high_clip_fraction",
            artifact_metrics.get("rgb_clip_fraction", 0.0),
        )
        or 0.0
    )
    raw = _stage_array(result, "sensor_raw")
    snr_db = None
    if raw is not None and raw.size and np.std(raw) > 0.0:
        snr_db = 20.0 * math.log10(max(float(np.mean(raw)), 1e-12) / float(np.std(raw)))

    gates: list[dict[str, Any]] = []

    def gate(
        gate_id: str,
        value: Any,
        limit: Any,
        passed: bool | None,
        *,
        unit: str,
        hard: bool = True,
        note: str = "",
    ) -> None:
        gates.append(
            {
                "id": gate_id,
                "value": value,
                "limit": limit,
                "unit": unit,
                "pass": passed,
                "hard": hard,
                "status": "not_evaluable" if passed is None else ("pass" if passed else "fail"),
                "note": note,
            }
        )

    fov_tolerance = max(1.0, requirements.hfov.value * 0.02)
    gate(
        "hfov_geometry",
        derived_hfov,
        f"{requirements.hfov.value} +/- {fov_tolerance}",
        abs(derived_hfov - requirements.hfov.value) <= fov_tolerance,
        unit="deg",
    )
    gate(
        "object_pixels_at_range",
        object_pixels,
        f">= {requirements.min_object_height.value}",
        object_pixels >= requirements.min_object_height.value,
        unit="px",
    )
    if requirements.max_sensor_diagonal is not None:
        gate(
            "sensor_diagonal",
            sensor_diagonal,
            f"<= {requirements.max_sensor_diagonal.value}",
            sensor_diagonal <= requirements.max_sensor_diagonal.value,
            unit="mm",
        )
    gate(
        "diffraction_sampling",
        airy_diameter_pixels,
        "<= 2.0",
        airy_diameter_pixels <= 2.0,
        unit="readout px",
        note="Airy diameter at the configured reference wavelength.",
    )
    gate(
        "pixel_bandwidth",
        pixel_rate_mpix_s,
        f"<= {requirements.max_pixel_rate.value}",
        pixel_rate_mpix_s <= requirements.max_pixel_rate.value,
        unit="Mpixel/s",
    )
    gate(
        "frame_rate",
        fps,
        f">= {requirements.min_frame_rate.value}",
        fps >= requirements.min_frame_rate.value,
        unit="fps",
    )
    gate(
        "rolling_shutter",
        rolling_shutter_ms,
        f"<= {requirements.max_rolling_shutter.value}",
        rolling_shutter_ms <= requirements.max_rolling_shutter.value,
        unit="ms",
    )
    gate(
        "total_latency",
        total_latency_ms,
        f"<= {requirements.max_latency.value}",
        total_latency_ms <= requirements.max_latency.value,
        unit="ms",
    )
    gate(
        "rgb_clipping",
        clip,
        f"<= {requirements.max_clip_fraction}",
        clip <= requirements.max_clip_fraction,
        unit="fraction",
        note="Highlight saturation only; black-level pixels are reported separately.",
    )
    snr_is_quantitative = (
        module.sensor.read_noise_e is not None and module.sensor.full_well_e is not None
    )
    gate(
        "sensor_snr",
        snr_db,
        f">= {requirements.min_snr_db}",
        None if snr_db is None or not snr_is_quantitative else snr_db >= requirements.min_snr_db,
        unit="dB",
        hard=snr_is_quantitative,
        note=(
            "Quantitative SNR requires configured read noise and full-well evidence."
            if not snr_is_quantitative
            else "Computed from the recorded RAW stage."
        ),
    )
    gate(
        "full_well_evidence",
        module.sensor.full_well_e,
        "configured",
        module.sensor.full_well_e is not None,
        unit="electron",
        hard=False,
        note="Missing full-well evidence prevents calibrated sensor claims.",
    )

    feasible = all(item["pass"] is not False for item in gates if item["hard"])
    return RequirementGateResult(
        feasible=feasible,
        gates=gates,
        derived={
            "output_size_rc": [output_rows, output_cols],
            "active_sensor_size_rc": [active_rows, active_cols],
            "native_sensor_size_rc": [native_rows, native_cols],
            "sensor_diagonal_mm": sensor_diagonal,
            "focal_length_mm": focal_mm,
            "object_pixels_at_range": object_pixels,
            "airy_diameter_um": airy_diameter_um,
            "airy_diameter_pixels": airy_diameter_pixels,
            "pixel_rate_mpix_s": pixel_rate_mpix_s,
            "rolling_shutter_ms": rolling_shutter_ms,
            "isp_latency_ms": isp_latency_ms,
            "total_latency_ms": total_latency_ms,
            "snr_db": snr_db,
        },
    )


def _size(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return fallback


def _stage_array(result: dict[str, Any], name: str) -> np.ndarray | None:
    stage = result.get("stages", {}).get(name, {})
    if isinstance(stage, dict) and "array" in stage:
        array = np.asarray(stage["array"], dtype=float)
        return array if array.size else None
    return None
