"""Explicit scene, sensor, readout, and detector-coordinate contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import zoom  # type: ignore[import-untyped]

from .models import CameraModule, GeometryTransform, SceneCase


def cfa_block_shape(cfa_preset: str) -> tuple[int, int]:
    normalized = cfa_preset.strip().lower().replace("-", "_")
    if "quad" in normalized:
        return 4, 4
    return 2, 2


def scene_image_size(scene: SceneCase) -> tuple[int, int]:
    if scene.image_path:
        path = Path(scene.image_path).expanduser().resolve()
        if path.is_file():
            image = iio.imread(path)
            return int(image.shape[0]), int(image.shape[1])
    size = scene.metadata.get("image_size_rc")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 96, 320


def aligned_sensor_size(module: CameraModule) -> tuple[int, int]:
    block_rows, block_cols = cfa_block_shape(module.sensor.cfa_preset)
    rows = max(block_rows, (module.sensor.rows // block_rows) * block_rows)
    cols = max(block_cols, (module.sensor.cols // block_cols) * block_cols)
    return rows, cols


def derive_geometry_transform(
    module: CameraModule,
    scene: SceneCase,
    *,
    output_size_rc: tuple[int, int] | list[int],
    readout_size_rc: tuple[int, int] | list[int] | None = None,
) -> GeometryTransform:
    source_rows, source_cols = scene_image_size(scene)
    output_rows, output_cols = int(output_size_rc[0]), int(output_size_rc[1])
    active_rows, active_cols = aligned_sensor_size(module)
    readout = readout_size_rc or output_size_rc
    readout_rows, readout_cols = int(readout[0]), int(readout[1])
    requested = (module.sensor.rows, module.sensor.cols)
    warning = None
    if (active_rows, active_cols) != requested:
        warning = (
            f"Requested sensor size {requested} was aligned to CFA block "
            f"{cfa_block_shape(module.sensor.cfa_preset)} as {(active_rows, active_cols)}."
        )
    return GeometryTransform(
        source_size_rc=(source_rows, source_cols),
        requested_sensor_size_rc=requested,
        active_sensor_size_rc=(active_rows, active_cols),
        readout_size_rc=(readout_rows, readout_cols),
        output_size_rc=(output_rows, output_cols),
        cfa_block_rc=cfa_block_shape(module.sensor.cfa_preset),
        binning_factor=module.sensor.binning_factor,
        scale_xy=(output_cols / max(source_cols, 1), output_rows / max(source_rows, 1)),
        hfov_deg=module.lens.hfov_deg,
        alignment_warning=warning,
    )


def transform_label_payload(
    payload: dict[str, Any], transform: GeometryTransform
) -> dict[str, Any]:
    result = {**payload, "image_size_rc": list(transform.output_size_rc)}
    objects = []
    rows, cols = transform.output_size_rc
    for item in payload.get("objects", []):
        transformed = dict(item)
        bbox = transform.transform_bbox(item["bbox_xyxy"])
        transformed["bbox_xyxy"] = bbox
        x1, y1, x2, y2 = bbox
        transformed["yolo_xywhn"] = [
            ((x1 + x2) / 2.0) / max(cols, 1),
            ((y1 + y2) / 2.0) / max(rows, 1),
            (x2 - x1) / max(cols, 1),
            (y2 - y1) / max(rows, 1),
        ]
        objects.append(transformed)
    result["objects"] = objects
    result["geometry_transform"] = transform.model_dump(mode="json")
    return result


def ideal_recapture(image: np.ndarray, output_size_rc: tuple[int, int]) -> np.ndarray:
    """Bilinear geometry-only reference used to diagnose the camera path."""

    source = np.asarray(image, dtype=float)
    if source.ndim == 2:
        source = np.repeat(source[..., None], 3, axis=2)
    source = source[..., :3]
    if source.max(initial=0.0) > 1.0:
        source = source / 255.0
    rows, cols = int(output_size_rc[0]), int(output_size_rc[1])
    factors = (rows / max(source.shape[0], 1), cols / max(source.shape[1], 1), 1.0)
    resized = zoom(source, factors, order=1, prefilter=False)
    if resized.shape[0] != rows or resized.shape[1] != cols:
        fixed = np.zeros((rows, cols, 3), dtype=float)
        copy_rows = min(rows, resized.shape[0])
        copy_cols = min(cols, resized.shape[1])
        fixed[:copy_rows, :copy_cols] = resized[:copy_rows, :copy_cols, :3]
        resized = fixed
    return np.clip(resized, 0.0, 1.0)


def global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Deterministic three-channel SSIM diagnostic without optional dependencies."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.shape != y.shape:
        y = ideal_recapture(y, x.shape[:2])
    values = []
    c1 = 0.01**2
    c2 = 0.03**2
    for channel in range(min(x.shape[2], y.shape[2], 3)):
        a = x[..., channel]
        b = y[..., channel]
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        var_a = float(np.var(a))
        var_b = float(np.var(b))
        covariance = float(np.mean((a - mean_a) * (b - mean_b)))
        numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
        denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
        values.append(numerator / max(denominator, 1e-12))
    return float(np.clip(np.mean(values), -1.0, 1.0)) if values else 0.0


def geometry_contract_metrics(module: CameraModule, transform: GeometryTransform) -> dict[str, Any]:
    native_rows = module.sensor.native_rows or transform.active_sensor_size_rc[0]
    native_cols = module.sensor.native_cols or transform.active_sensor_size_rc[1]
    sensor_width_mm = native_cols * module.sensor.pixel_size_um * 1e-3
    sensor_height_mm = native_rows * module.sensor.pixel_size_um * 1e-3
    simulation_pitch_um = (
        module.sensor.simulation_pixel_size_um or module.sensor.pixel_size_um
    )
    simulation_width_mm = transform.active_sensor_size_rc[1] * simulation_pitch_um * 1e-3
    simulation_height_mm = transform.active_sensor_size_rc[0] * simulation_pitch_um * 1e-3
    focal_mm = module.lens.focal_length_mm
    if focal_mm is None:
        focal_mm = sensor_width_mm / (2.0 * math.tan(math.radians(module.lens.hfov_deg) / 2.0))
    derived_hfov = math.degrees(2.0 * math.atan2(sensor_width_mm / 2.0, focal_mm))
    return {
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "sensor_diagonal_mm": math.hypot(sensor_width_mm, sensor_height_mm),
        "simulation_sensor_width_mm": simulation_width_mm,
        "simulation_sensor_height_mm": simulation_height_mm,
        "simulation_extent_error_fraction": max(
            abs(simulation_width_mm - sensor_width_mm) / max(sensor_width_mm, 1e-12),
            abs(simulation_height_mm - sensor_height_mm) / max(sensor_height_mm, 1e-12),
        ),
        "native_sensor_size_rc": [native_rows, native_cols],
        "simulation_readout_size_rc": list(transform.active_sensor_size_rc),
        "sensor_geometry_source": module.sensor.geometry_source,
        "focal_length_mm": focal_mm,
        "configured_hfov_deg": module.lens.hfov_deg,
        "derived_hfov_deg": derived_hfov,
        "hfov_error_deg": abs(derived_hfov - module.lens.hfov_deg),
        "output_shape_matches_contract": (
            tuple(transform.output_size_rc) == tuple(transform.readout_size_rc)
        ),
        "simulation_is_downsampled": (native_rows, native_cols)
        != tuple(transform.active_sensor_size_rc),
    }
