#!/usr/bin/env python3
"""Generate split-PD and OCL supercell optical-response LUTs.

The output is intended for camera-system simulation: each sweep point produces a
response vector over sub-PD or pixel collection regions. Units are microns.
This is an optical proxy only; it does not include TCAD carrier collection.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from meep_microlens_array_3d import MicrolensArrayGeometry
from sensor_stack_config import (
    DEFAULT_STACK_CONFIG,
    VALID_SHIELD_MODES,
    geometry_from_config,
    load_stack_config,
    material_role_for_color,
    medium_for_role,
    metal_for_stack,
    nk_from_material,
    shield_active_half_width_um,
    shield_blocks_local_point,
    shield_config_for_stack,
    stack_metadata,
)


ROOT = Path(__file__).resolve().parent
CFA_PATTERNS = ("uniform", "bayer", "quad", "nona")
CFA_COLORS = ("red", "green", "blue")
COLOR_CHANNELS = (*CFA_COLORS, "clear")


@dataclass(frozen=True)
class SweepCase:
    name: str
    cra_x_deg: float
    cra_z_deg: float
    field_x_norm: float
    field_z_norm: float
    lens_shift_x_um: float
    lens_shift_z_um: float
    aperture_shift_x_um: float | None = None
    aperture_shift_z_um: float | None = None

    @property
    def aperture_shift_x(self) -> float:
        return self.lens_shift_x_um if self.aperture_shift_x_um is None else self.aperture_shift_x_um

    @property
    def aperture_shift_z(self) -> float:
        return self.lens_shift_z_um if self.aperture_shift_z_um is None else self.aperture_shift_z_um


@dataclass(frozen=True)
class Region:
    region_id: str
    kind: str
    ix: int
    iz: int
    x_um: float
    z_um: float
    sx_um: float
    sz_um: float


@dataclass(frozen=True)
class OclLens:
    lens_id: str
    kind: str
    ix: int
    iz: int
    w: int
    h: int
    x_um: float
    z_um: float
    aperture_radius_um: float
    shift_x_um: float = 0.0
    shift_z_um: float = 0.0
    polygon_um: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class OclSagProfile:
    profile_type: str = "sphere"
    curvature_radius_um: float | None = None
    conic_k: float = 0.0
    a4: float = 0.0
    a6: float = 0.0
    a8: float = 0.0
    normalize_edge: bool = True


@dataclass(frozen=True)
class OclSurfaceMap:
    x_um: tuple[float, ...]
    z_um: tuple[float, ...]
    height_um: tuple[tuple[float, ...], ...]
    source: str = "inline"


@dataclass(frozen=True)
class CfaPolygon:
    polygon_id: str
    color: str
    ix: int | None
    iz: int | None
    polygon_um: tuple[tuple[float, float], ...]
    shift_x_um: float = 0.0
    shift_z_um: float = 0.0
    source: str = "inline"


@dataclass(frozen=True)
class CfaPolygonSet:
    polygons: tuple[CfaPolygon, ...] = ()
    background: str = "nearest"
    source: str = "inline"


@dataclass(frozen=True)
class PupilRay:
    ray_index: int
    pupil_u: float
    pupil_v: float
    weight: float
    cra_x_deg: float
    cra_z_deg: float


ROW_FIELD_ORDER = [
    "schema",
    "mode",
    "color_channel",
    "wavelength_nm",
    "case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "aperture_shift_x_um",
    "aperture_shift_z_um",
    "cfa_pattern",
    "cfa_shift_red_x_um",
    "cfa_shift_green_x_um",
    "cfa_shift_blue_x_um",
    "cfa_shift_red_z_um",
    "cfa_shift_green_z_um",
    "cfa_shift_blue_z_um",
    "cfa_polygon_count",
    "cfa_polygon_background",
    "ocl_layout_name",
    "ocl_lens_count",
    "collection_mode",
    "target_lens_id",
    "source_aperture_lens_id",
    "source_aperture_enabled",
    "shield_mode",
    "shield_mask_edge_width_um",
    "region_id",
    "region_kind",
    "region_ix",
    "region_iz",
    "region_x_um",
    "region_z_um",
    "region_sx_um",
    "region_sz_um",
    "response_model",
    "response",
    "total_si_absorption_fraction_estimate",
    "signed_flux_si_absorption_fraction_diagnostic",
    "volume_absorption_region_fraction",
    "volume_absorption_region_raw",
    "volume_absorption_total_raw",
    "volume_absorption_voxel_um3",
    "volume_absorption_scale_to_flux",
    "volume_absorption_region_fraction_sum",
    "focal_region_fraction",
    "incident_monitor_net_power_normalized",
    "full_si_top_net_power_normalized",
    "full_si_bottom_net_power_normalized",
    "region_top_net_power_normalized",
    "region_bottom_net_power_normalized",
    "regional_flux_response_diagnostic",
    "focal_centroid_x_um",
    "focal_centroid_z_um",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
    "silicon_n",
    "silicon_k",
    "silicon_eps_imag",
    "pupil_integrated",
    "pupil_ray_count",
    "pupil_ray_index",
    "pupil_u",
    "pupil_v",
    "pupil_weight",
    "ray_cra_x_deg",
    "ray_cra_z_deg",
    "split_mode",
    "normalized_region_response_to_first_same_region",
]


SUMMARY_FIELD_ORDER = [
    "schema",
    "mode",
    "color_channel",
    "wavelength_nm",
    "case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "cfa_pattern",
    "cfa_shift_red_x_um",
    "cfa_shift_green_x_um",
    "cfa_shift_blue_x_um",
    "cfa_shift_red_z_um",
    "cfa_shift_green_z_um",
    "cfa_shift_blue_z_um",
    "cfa_polygon_count",
    "cfa_polygon_background",
    "ocl_layout_name",
    "ocl_lens_count",
    "collection_mode",
    "target_lens_id",
    "source_aperture_lens_id",
    "source_aperture_enabled",
    "shield_mode",
    "shield_mask_edge_width_um",
    "pupil_integrated",
    "pupil_ray_count",
    "total_response",
    "normalized_total_response_to_first",
    "max_region_response",
    "min_region_response",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "incident_monitor_net_power_normalized",
    "total_si_absorption_fraction_estimate",
    "signed_flux_si_absorption_fraction_diagnostic",
    "volume_absorption_total_raw",
    "volume_absorption_scale_to_flux",
    "volume_absorption_region_fraction_sum",
    "focal_centroid_x_um",
    "focal_centroid_z_um",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
    "grid_dx_um",
    "si_n_at_wavelength",
    "si_internal_wavelength_um",
    "si_internal_wavelength_pixels",
    "passivation_thickness_pixels",
    "lens_edge_gap_pixels",
    "cfa_thickness_pixels",
    "minimum_critical_feature_pixels",
    "min_feature_pixels_required",
    "min_si_wavelength_pixels_required",
    "recommended_si_wavelength_resolution_px_per_um",
    "recommended_feature_resolution_px_per_um",
    "recommended_min_resolution_px_per_um",
    "si_wavelength_gate_pass",
    "critical_feature_gate_pass",
    "grid_resolution_gate_pass",
    "grid_resolution_notes",
]


TCAD_PROFILE_FIELD_ORDER = [
    "schema",
    "mode",
    "color_channel",
    "wavelength_nm",
    "case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "pupil_integrated",
    "pupil_ray_count",
    "pupil_ray_index",
    "pupil_u",
    "pupil_v",
    "pupil_weight",
    "ray_cra_x_deg",
    "ray_cra_z_deg",
    "depth_um_from_si_top",
    "y_um",
    "absorption_fraction_per_um",
    "absorption_fraction_per_cm",
    "generation_cm3_s",
    "generation_normalized",
    "incident_photon_flux_cm2_s",
]


def fieldnames_for(rows: list[dict], preferred: list[str]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows))
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def write_csv_atomic(path: Path, rows: list[dict], preferred: list[str]) -> Path | None:
    if not rows:
        return None
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(rows, preferred))
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
    return path


def apply_response_normalization(
    summaries: list[dict],
    all_rows: list[dict],
    grouped_rows: list[list[dict]],
) -> None:
    first_total = summaries[0]["total_response"] if summaries else 0.0
    first_by_region = (
        {row["region_id"]: row["response"] for row in grouped_rows[0]}
        if grouped_rows
        else {}
    )
    for summary in summaries:
        summary["normalized_total_response_to_first"] = (
            summary["total_response"] / first_total if first_total else float("nan")
        )
    for row in all_rows:
        row["normalized_region_response_to_first_same_region"] = (
            row["response"] / first_by_region[row["region_id"]]
            if row["region_id"] in first_by_region and first_by_region[row["region_id"]]
            else float("nan")
        )


def write_partial_outputs(
    output_dir: Path,
    all_rows: list[dict],
    all_ray_rows: list[dict],
    all_tcad_profile_rows: list[dict],
    all_tcad_ray_profile_rows: list[dict],
    summaries: list[dict],
) -> None:
    """Persist restart/debug evidence after each completed sweep point."""
    partial_paths = {
        "summary_csv": write_csv_atomic(
            output_dir / "camera_lut_summary_partial.csv",
            summaries,
            SUMMARY_FIELD_ORDER,
        ),
        "long_csv": write_csv_atomic(
            output_dir / "camera_lut_long_partial.csv",
            all_rows,
            ROW_FIELD_ORDER,
        ),
        "pupil_ray_csv": write_csv_atomic(
            output_dir / "camera_lut_pupil_rays_partial.csv",
            all_ray_rows,
            ROW_FIELD_ORDER,
        ),
        "tcad_generation_profile_1d_csv": write_csv_atomic(
            output_dir / "tcad_generation_profile_1d_partial.csv",
            all_tcad_profile_rows,
            TCAD_PROFILE_FIELD_ORDER,
        ),
        "tcad_generation_profile_1d_pupil_ray_csv": write_csv_atomic(
            output_dir / "tcad_generation_profile_1d_pupil_rays_partial.csv",
            all_tcad_ray_profile_rows,
            TCAD_PROFILE_FIELD_ORDER,
        ),
    }
    checkpoint = {
        "schema": "camera_supercell_optical_lut_checkpoint_v1",
        "completed_sweep_points": len(summaries),
        "last_sweep_point": summaries[-1] if summaries else None,
        "partial_paths": {
            key: str(value) if value is not None else None
            for key, value in partial_paths.items()
        },
    }
    checkpoint_path = output_dir / "camera_lut_checkpoint.json"
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def parse_wavelengths(raw: str) -> list[float]:
    wavelengths = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not wavelengths:
        raise ValueError("At least one wavelength is required")
    return wavelengths


def parse_cases(raw: str) -> list[SweepCase]:
    """Parse cases.

    Preferred format:
      name:cra_x:cra_z:field_x:field_z:lens_shift_x:lens_shift_z[:ap_shift_x[:ap_shift_z]]

    Compatibility shorthand:
      name:cra_x:lens_shift_x
    """
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 3:
            cases.append(
                SweepCase(
                    name=parts[0],
                    cra_x_deg=float(parts[1]),
                    cra_z_deg=0.0,
                    field_x_norm=0.0,
                    field_z_norm=0.0,
                    lens_shift_x_um=float(parts[2]),
                    lens_shift_z_um=0.0,
                )
            )
            continue
        if len(parts) < 7:
            raise ValueError(
                "Case must be name:cra_x:cra_z:field_x:field_z:lens_shift_x:lens_shift_z"
                "[:ap_shift_x[:ap_shift_z]]"
            )
        cases.append(
            SweepCase(
                name=parts[0],
                cra_x_deg=float(parts[1]),
                cra_z_deg=float(parts[2]),
                field_x_norm=float(parts[3]),
                field_z_norm=float(parts[4]),
                lens_shift_x_um=float(parts[5]),
                lens_shift_z_um=float(parts[6]),
                aperture_shift_x_um=float(parts[7]) if len(parts) >= 8 and parts[7] else None,
                aperture_shift_z_um=float(parts[8]) if len(parts) >= 9 and parts[8] else None,
            )
        )
    if not cases:
        raise ValueError("No sweep cases parsed")
    return cases


def mode_shape(mode: str, layout_nx: int | None = None, layout_nz: int | None = None) -> tuple[int, int]:
    if mode == "split-pd-1x1":
        return 1, 1
    if mode == "ocl-2x2":
        return 2, 2
    if mode == "ocl-3x3":
        return 3, 3
    if mode == "ocl-layout":
        if layout_nx is None or layout_nz is None:
            raise ValueError("ocl-layout mode requires --layout-nx and --layout-nz")
        if layout_nx < 1 or layout_nz < 1 or layout_nx > 12 or layout_nz > 12:
            raise ValueError("layout dimensions must be between 1 and 12 pixels")
        return layout_nx, layout_nz
    raise ValueError(f"Unsupported mode: {mode}")


def pixel_centers(count: int, pitch: float) -> list[float]:
    offset = 0.5 * (count - 1)
    return [(i - offset) * pitch for i in range(count)]


def periodic_delta(value: float, center: float, period: float) -> float:
    return ((value - center + 0.5 * period) % period) - 0.5 * period


def center_for_pixel_span(centers: list[float], start: int, width: int) -> float:
    return 0.5 * (centers[start] + centers[start + width - 1])


def parse_ocl_layout(raw: str | None, nx: int, nz: int, geom: MicrolensArrayGeometry) -> list[OclLens]:
    if not raw:
        return []
    x_centers = pixel_centers(nx, geom.pitch)
    z_centers = pixel_centers(nz, geom.pitch)
    lenses: list[OclLens] = []
    occupied: dict[tuple[int, int], str] = {}
    for item in raw.split(","):
        clean = item.strip()
        if not clean:
            continue
        parts = clean.split(":")
        if len(parts) < 5:
            raise ValueError("OCL layout item must be id:ix:iz:w:h[:shift_x[:shift_z]]")
        lens_id = parts[0]
        if not lens_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid OCL lens id: {lens_id!r}")
        ix = int(parts[1])
        iz = int(parts[2])
        w = int(parts[3])
        h = int(parts[4])
        shift_x = float(parts[5]) if len(parts) >= 6 and parts[5] else 0.0
        shift_z = float(parts[6]) if len(parts) >= 7 and parts[6] else 0.0
        if ix < 0 or iz < 0 or w <= 0 or h <= 0 or ix + w > nx or iz + h > nz:
            raise ValueError(f"OCL lens {lens_id!r} is outside the {nx}x{nz} layout")
        if abs(shift_x) > geom.pitch or abs(shift_z) > geom.pitch:
            raise ValueError(f"OCL lens {lens_id!r} shift is larger than one pixel pitch")
        for px in range(ix, ix + w):
            for pz in range(iz, iz + h):
                key = (px, pz)
                if key in occupied:
                    raise ValueError(
                        f"OCL layout overlap at pixel ({px},{pz}) between {occupied[key]!r} and {lens_id!r}"
                    )
                occupied[key] = lens_id
        aperture_radius = 0.5 * min(w, h) * geom.pitch - geom.lens_edge_gap
        if aperture_radius <= 0:
            raise ValueError(f"OCL lens {lens_id!r} aperture is non-positive")
        lenses.append(
            OclLens(
                lens_id=lens_id,
                kind=f"{w}x{h}",
                ix=ix,
                iz=iz,
                w=w,
                h=h,
                x_um=center_for_pixel_span(x_centers, ix, w),
                z_um=center_for_pixel_span(z_centers, iz, h),
                aperture_radius_um=aperture_radius,
                shift_x_um=shift_x,
                shift_z_um=shift_z,
            )
        )
    if not lenses:
        raise ValueError("OCL layout string did not contain any lenses")
    return lenses


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    area = 0.0
    for index, (x0, z0) in enumerate(points):
        x1, z1 = points[(index + 1) % len(points)]
        area += x0 * z1 - x1 * z0
    return 0.5 * area


def point_on_segment(
    x: float,
    z: float,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    tolerance: float = 1e-9,
) -> bool:
    cross = (x - x0) * (z1 - z0) - (z - z0) * (x1 - x0)
    if abs(cross) > tolerance:
        return False
    dot = (x - x0) * (x1 - x0) + (z - z0) * (z1 - z0)
    if dot < -tolerance:
        return False
    length2 = (x1 - x0) ** 2 + (z1 - z0) ** 2
    return dot <= length2 + tolerance


def point_in_polygon(x: float, z: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    count = len(polygon)
    for index in range(count):
        x0, z0 = polygon[index]
        x1, z1 = polygon[(index + 1) % count]
        if point_on_segment(x, z, x0, z0, x1, z1):
            return True
        crosses = (z0 > z) != (z1 > z)
        if crosses:
            x_intersect = (x1 - x0) * (z - z0) / (z1 - z0) + x0
            if x < x_intersect:
                inside = not inside
    return inside


def _resolve_json_import_path(reference: str, option_name: str) -> Path:
    path_text = reference.strip()
    if not path_text:
        raise ValueError(f"{option_name} import path is empty")
    candidate = Path(path_text)
    path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{option_name} import path must stay under {ROOT}") from error
    if path.suffix.lower() != ".json":
        raise ValueError(f"{option_name} import path must be a JSON file")
    if not path.is_file():
        raise ValueError(f"{option_name} import file does not exist: {path}")
    return path


def json_argument_source(raw: str | None, option_name: str) -> str | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if not text.startswith("@"):
        return "inline"
    path = _resolve_json_import_path(text[1:], option_name)
    return f"@{path.relative_to(ROOT).as_posix()}"


def json_argument_payload(
    raw: str | None,
    *,
    option_name: str,
    root_keys: tuple[str, ...],
) -> tuple[Any | None, str | None]:
    if not raw:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    if text.startswith("@"):
        path = _resolve_json_import_path(text[1:], option_name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{option_name} import file is not valid JSON: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{option_name} import file must contain a JSON object")
        for key in root_keys:
            if key in payload:
                return payload[key], f"@{path.relative_to(ROOT).as_posix()}"
        geometry = payload.get("geometry")
        if isinstance(geometry, dict):
            for key in root_keys:
                if key in geometry:
                    return geometry[key], f"@{path.relative_to(ROOT).as_posix()}"
        return payload, f"@{path.relative_to(ROOT).as_posix()}"
    try:
        return json.loads(text), "inline"
    except json.JSONDecodeError as error:
        raise ValueError(f"{option_name} must be a JSON object or @path/to/file.json import") from error


def parse_ocl_polygon_map(
    raw: str | None,
    geom: MicrolensArrayGeometry,
) -> dict[str, tuple[tuple[float, float], ...]]:
    if not raw:
        return {}
    payload, _source = json_argument_payload(
        raw,
        option_name="--ocl-polygons",
        root_keys=("ocl_polygons",),
    )
    if not isinstance(payload, dict):
        raise ValueError("--ocl-polygons must be a JSON object")
    max_extent = max(12.0 * geom.pitch, 1.0)
    parsed: dict[str, tuple[tuple[float, float], ...]] = {}
    for lens_id, raw_points in payload.items():
        if not isinstance(lens_id, str) or not lens_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid OCL polygon lens id: {lens_id!r}")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            raise ValueError(f"OCL polygon {lens_id!r} must contain at least three [x,z] points")
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"OCL polygon {lens_id!r} points must be [x,z] pairs")
            x = float(point[0])
            z = float(point[1])
            if not math.isfinite(x) or not math.isfinite(z):
                raise ValueError(f"OCL polygon {lens_id!r} contains a non-finite point")
            if abs(x) > max_extent or abs(z) > max_extent:
                raise ValueError(f"OCL polygon {lens_id!r} point exceeds allowed local extent")
            points.append((x, z))
        polygon = tuple(points)
        if abs(polygon_area(polygon)) < 1e-6:
            raise ValueError(f"OCL polygon {lens_id!r} area is too small")
        parsed[lens_id] = polygon
    return parsed


def apply_ocl_polygons(
    lenses: list[OclLens],
    raw: str | None,
    geom: MicrolensArrayGeometry,
) -> list[OclLens]:
    polygon_map = parse_ocl_polygon_map(raw, geom)
    if not polygon_map:
        return lenses
    if not lenses:
        raise ValueError("--ocl-polygons currently requires --ocl-layout so lens ids can be resolved")
    known = {lens.lens_id for lens in lenses}
    matched = known & set(polygon_map)
    if not matched:
        imported_ids = ", ".join(sorted(polygon_map))
        raise ValueError(f"--ocl-polygons did not match any lens in --ocl-layout; imported ids: {imported_ids}")
    updated = []
    for lens in lenses:
        polygon = polygon_map.get(lens.lens_id)
        if polygon is None:
            updated.append(lens)
            continue
        radius = max(math.hypot(x, z) for x, z in polygon)
        updated.append(replace(lens, aperture_radius_um=max(radius, 0.05), polygon_um=polygon))
    return updated


def choose_target_ocl_lens(lenses: list[OclLens], target_lens_id: str | None = None) -> OclLens | None:
    if not lenses:
        return None
    if target_lens_id:
        for lens in lenses:
            if lens.lens_id == target_lens_id:
                return lens
        raise ValueError(f"target OCL lens {target_lens_id!r} was not found in --ocl-layout")
    return min(lenses, key=lambda lens: lens.x_um * lens.x_um + lens.z_um * lens.z_um)


def _safe_inline_id(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    if not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid {label}: {text!r}")
    return text


def _parse_local_polygon_points(
    raw_points: Any,
    *,
    label: str,
    max_extent: float,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise ValueError(f"{label} must contain at least three [x,z] points")
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} points must be [x,z] pairs")
        x = float(point[0])
        z = float(point[1])
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError(f"{label} contains a non-finite point")
        if abs(x) > max_extent or abs(z) > max_extent:
            raise ValueError(f"{label} point exceeds allowed local extent")
        points.append((x, z))
    polygon = tuple(points)
    if abs(polygon_area(polygon)) < 1e-6:
        raise ValueError(f"{label} area is too small")
    return polygon


def _parse_cfa_polygon_spec(
    polygon_id: str,
    raw_spec: Any,
    *,
    default_color: str | None,
    default_ix: int | None,
    default_iz: int | None,
    max_extent: float,
    default_source: str,
) -> CfaPolygon:
    if isinstance(raw_spec, dict):
        points = raw_spec.get("points", raw_spec.get("polygon_um", raw_spec.get("polygon")))
        color = str(raw_spec.get("color", default_color or "")).lower()
        ix_value = raw_spec.get("ix", default_ix)
        iz_value = raw_spec.get("iz", default_iz)
        shift_x = float(raw_spec.get("shift_x_um", raw_spec.get("x_shift_um", 0.0)) or 0.0)
        shift_z = float(raw_spec.get("shift_z_um", raw_spec.get("z_shift_um", 0.0)) or 0.0)
        source = str(raw_spec.get("source") or default_source)[:160]
    else:
        points = raw_spec
        color = str(default_color or "").lower()
        ix_value = default_ix
        iz_value = default_iz
        shift_x = 0.0
        shift_z = 0.0
        source = default_source
    if color not in CFA_COLORS:
        raise ValueError(f"CFA polygon {polygon_id!r} color must be one of {CFA_COLORS}")
    ix = None if ix_value in {None, ""} else int(ix_value)
    iz = None if iz_value in {None, ""} else int(iz_value)
    if ix is not None and abs(ix) > 64:
        raise ValueError(f"CFA polygon {polygon_id!r} ix is out of range")
    if iz is not None and abs(iz) > 64:
        raise ValueError(f"CFA polygon {polygon_id!r} iz is out of range")
    if not math.isfinite(shift_x) or not math.isfinite(shift_z):
        raise ValueError(f"CFA polygon {polygon_id!r} shift must be finite")
    if abs(shift_x) > max_extent or abs(shift_z) > max_extent:
        raise ValueError(f"CFA polygon {polygon_id!r} shift exceeds allowed local extent")
    return CfaPolygon(
        polygon_id=polygon_id,
        color=color,
        ix=ix,
        iz=iz,
        polygon_um=_parse_local_polygon_points(points, label=f"CFA polygon {polygon_id!r}", max_extent=max_extent),
        shift_x_um=shift_x,
        shift_z_um=shift_z,
        source=source,
    )


def parse_cfa_polygons(raw: str | None, geom: MicrolensArrayGeometry) -> CfaPolygonSet:
    """Parse local CFA mask polygons.

    Supported JSON forms:
      {"red": [[x,z], ...], "green": ..., "blue": ..., "background": "passivation"}
      {"colors": {"red": {"points": [[x,z], ...], "shift_x_um": 0.01}}, "cells": [...]}

    Points are local to the CFA tile center after the global per-color CFA shift.
    When polygons are supplied, unmatched CFA-layer points default to passivation
    unless background is explicitly set to "nearest" or "air".
    """
    if not raw:
        return CfaPolygonSet()
    payload, source = json_argument_payload(
        raw,
        option_name="--cfa-polygons",
        root_keys=("cfa_polygons",),
    )
    if not isinstance(payload, dict):
        raise ValueError("--cfa-polygons must be a JSON object")
    polygon_source = source or "inline"
    background = str(payload.get("background", "passivation")).lower()
    if background not in {"nearest", "passivation", "air"}:
        raise ValueError("--cfa-polygons background must be nearest, passivation, or air")
    max_extent = max(12.0 * geom.pitch, 1.0)
    polygons: list[CfaPolygon] = []
    for color in CFA_COLORS:
        if color in payload:
            polygons.append(
                _parse_cfa_polygon_spec(
                    color,
                    payload[color],
                    default_color=color,
                    default_ix=None,
                    default_iz=None,
                    max_extent=max_extent,
                    default_source=polygon_source,
                )
            )
    color_specs = payload.get("colors", {})
    if color_specs:
        if not isinstance(color_specs, dict):
            raise ValueError("--cfa-polygons.colors must be an object")
        for color, spec in color_specs.items():
            color_key = str(color).lower()
            polygons.append(
                _parse_cfa_polygon_spec(
                    f"color_{color_key}",
                    spec,
                    default_color=color_key,
                    default_ix=None,
                    default_iz=None,
                    max_extent=max_extent,
                    default_source=polygon_source,
                )
            )
    cell_specs = payload.get("cells", [])
    if cell_specs:
        if not isinstance(cell_specs, list):
            raise ValueError("--cfa-polygons.cells must be a list")
        for index, spec in enumerate(cell_specs):
            if not isinstance(spec, dict):
                raise ValueError("--cfa-polygons.cells entries must be objects")
            polygon_id = str(spec.get("id") or f"cell_{index}")
            _safe_inline_id(polygon_id, "CFA cell polygon id")
            polygons.append(
                _parse_cfa_polygon_spec(
                    polygon_id,
                    spec,
                    default_color=None,
                    default_ix=None,
                    default_iz=None,
                    max_extent=max_extent,
                    default_source=polygon_source,
                )
            )
    if not polygons:
        raise ValueError("--cfa-polygons did not contain any red/green/blue/colors/cells polygon")
    if len(polygons) > 96:
        raise ValueError("--cfa-polygons contains too many polygons for the local Python material function")
    return CfaPolygonSet(polygons=tuple(polygons), background=background, source=polygon_source)


def cfa_polygon_candidates(mask: CfaPolygonSet, ix: int, iz: int, color: str) -> list[CfaPolygon]:
    cell_specific = [
        item
        for item in mask.polygons
        if item.ix is not None and item.iz is not None and item.ix == ix and item.iz == iz
    ]
    color_specific = [
        item
        for item in mask.polygons
        if item.ix is None and item.iz is None and item.color == color
    ]
    return cell_specific + color_specific


def cfa_polygon_color_at(
    x: float,
    z: float,
    x_centers: list[float],
    z_centers: list[float],
    cell_x: float,
    cell_z: float,
    color_for_index: Callable[[int, int], str],
    cfa_shifts: dict[str, tuple[float, float]],
    mask: CfaPolygonSet,
    *,
    periodic: bool,
) -> str | None:
    if not mask.polygons:
        return None
    for ix, xc in enumerate(x_centers):
        for iz, zc in enumerate(z_centers):
            base_color = color_for_index(ix, iz)
            shift_x, shift_z = cfa_shift_for_color(base_color, cfa_shifts)
            for item in cfa_polygon_candidates(mask, ix, iz, base_color):
                center_x = xc + shift_x + item.shift_x_um
                center_z = zc + shift_z + item.shift_z_um
                dx = periodic_delta(x, center_x, cell_x) if periodic else x - center_x
                dz = periodic_delta(z, center_z, cell_z) if periodic else z - center_z
                if point_in_polygon(dx, dz, item.polygon_um):
                    return item.color
    return None


def lens_sphere_radius(aperture_radius_um: float, lens_height_um: float) -> float:
    if lens_height_um <= 0:
        raise ValueError("lens height must be positive")
    return (aperture_radius_um * aperture_radius_um + lens_height_um * lens_height_um) / (2 * lens_height_um)


def parse_ocl_sag_profiles(raw: str | None) -> dict[str, OclSagProfile]:
    if not raw:
        return {}
    payload, _source = json_argument_payload(
        raw,
        option_name="--ocl-sag",
        root_keys=("ocl_sag", "ocl_sag_profiles"),
    )
    if not isinstance(payload, dict):
        raise ValueError("--ocl-sag must be a JSON object")
    profiles: dict[str, OclSagProfile] = {}
    for lens_id, spec in payload.items():
        key = str(lens_id)
        if key != "default" and not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid --ocl-sag key: {key!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"--ocl-sag[{key!r}] must be an object")
        profile_type = str(spec.get("type", spec.get("profile_type", "asphere"))).lower()
        if profile_type not in {"sphere", "asphere"}:
            raise ValueError(f"Unsupported OCL sag profile type: {profile_type!r}")
        radius_value = spec.get("curvature_radius_um", spec.get("radius_um"))
        radius = None if radius_value in {None, ""} else float(radius_value)
        if radius is not None and (not math.isfinite(radius) or radius <= 0):
            raise ValueError(f"--ocl-sag[{key!r}].curvature_radius_um must be positive")
        profiles[key] = OclSagProfile(
            profile_type=profile_type,
            curvature_radius_um=radius,
            conic_k=float(spec.get("conic_k", spec.get("k", 0.0)) or 0.0),
            a4=float(spec.get("a4", 0.0) or 0.0),
            a6=float(spec.get("a6", 0.0) or 0.0),
            a8=float(spec.get("a8", 0.0) or 0.0),
            normalize_edge=bool(spec.get("normalize_edge", True)),
        )
    return profiles


def strictly_increasing(values: tuple[float, ...]) -> bool:
    return all(values[index] < values[index + 1] for index in range(len(values) - 1))


def parse_ocl_surface_maps(raw: str | None, geom: MicrolensArrayGeometry) -> dict[str, OclSurfaceMap]:
    if not raw:
        return {}
    payload, source = json_argument_payload(
        raw,
        option_name="--ocl-surface-map",
        root_keys=("ocl_surface_map", "ocl_surface_maps", "surface_maps"),
    )
    if not isinstance(payload, dict):
        raise ValueError("--ocl-surface-map must be a JSON object")
    maps: dict[str, OclSurfaceMap] = {}
    for lens_id, spec in payload.items():
        key = str(lens_id)
        if key != "default" and not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid --ocl-surface-map key: {key!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"--ocl-surface-map[{key!r}] must be an object")
        try:
            x_values = tuple(float(value) for value in spec["x_um"])
            z_values = tuple(float(value) for value in spec["z_um"])
            height_rows = tuple(tuple(float(value) for value in row) for row in spec["height_um"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"--ocl-surface-map[{key!r}] must contain x_um, z_um, and height_um numeric arrays"
            ) from error
        if len(x_values) < 2 or len(z_values) < 2:
            raise ValueError(f"--ocl-surface-map[{key!r}] must contain at least a 2x2 grid")
        if len(x_values) > 65 or len(z_values) > 65:
            raise ValueError(f"--ocl-surface-map[{key!r}] grid is too large for the local Python material function")
        if not strictly_increasing(x_values) or not strictly_increasing(z_values):
            raise ValueError(f"--ocl-surface-map[{key!r}] x_um and z_um must be strictly increasing")
        if len(height_rows) != len(z_values) or any(len(row) != len(x_values) for row in height_rows):
            raise ValueError(f"--ocl-surface-map[{key!r}] height_um shape must be len(z_um) x len(x_um)")
        finite_values = [*x_values, *z_values, *(value for row in height_rows for value in row)]
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError(f"--ocl-surface-map[{key!r}] contains a non-finite value")
        if min(value for row in height_rows for value in row) < -1e-9:
            raise ValueError(f"--ocl-surface-map[{key!r}] height_um must be non-negative")
        max_height = max(value for row in height_rows for value in row)
        if max_height > geom.lens_height + 1e-9:
            raise ValueError(
                f"--ocl-surface-map[{key!r}] max height {max_height:g} exceeds geometry lens_height {geom.lens_height:g}; "
                "increase geometry_um.lens_height or scale the measured surface."
            )
        maps[key] = OclSurfaceMap(
            x_um=x_values,
            z_um=z_values,
            height_um=height_rows,
            source=str(spec.get("source") or source or "inline")[:160],
        )
    return maps


def sag_profile_for_lens(
    lens_id: str | None,
    profiles: dict[str, OclSagProfile],
) -> OclSagProfile:
    if lens_id and lens_id in profiles:
        return profiles[lens_id]
    return profiles.get("default", OclSagProfile())


def surface_map_for_lens(
    lens_id: str | None,
    maps: dict[str, OclSurfaceMap],
) -> OclSurfaceMap | None:
    if lens_id and lens_id in maps:
        return maps[lens_id]
    return maps.get("default")


def bilinear_surface_height_um(dx: float, dz: float, surface_map: OclSurfaceMap) -> float:
    xs = surface_map.x_um
    zs = surface_map.z_um
    if dx < xs[0] or dx > xs[-1] or dz < zs[0] or dz > zs[-1]:
        return 0.0
    ix = int(np.searchsorted(xs, dx, side="right") - 1)
    iz = int(np.searchsorted(zs, dz, side="right") - 1)
    ix = min(max(ix, 0), len(xs) - 2)
    iz = min(max(iz, 0), len(zs) - 2)
    x0, x1 = xs[ix], xs[ix + 1]
    z0, z1 = zs[iz], zs[iz + 1]
    tx = 0.0 if x1 == x0 else (dx - x0) / (x1 - x0)
    tz = 0.0 if z1 == z0 else (dz - z0) / (z1 - z0)
    h00 = surface_map.height_um[iz][ix]
    h10 = surface_map.height_um[iz][ix + 1]
    h01 = surface_map.height_um[iz + 1][ix]
    h11 = surface_map.height_um[iz + 1][ix + 1]
    return max(
        (1.0 - tx) * (1.0 - tz) * h00
        + tx * (1.0 - tz) * h10
        + (1.0 - tx) * tz * h01
        + tx * tz * h11,
        0.0,
    )


def asphere_drop_um(radius_um: float, profile: OclSagProfile, aperture_radius_um: float, lens_height_um: float) -> float:
    if radius_um <= 0:
        return 0.0
    curvature_radius = profile.curvature_radius_um or lens_sphere_radius(aperture_radius_um, lens_height_um)
    rho = min(radius_um, aperture_radius_um)
    radical = 1.0 - (1.0 + profile.conic_k) * (rho * rho) / (curvature_radius * curvature_radius)
    if radical < 0.0:
        radical = 0.0
    denominator = curvature_radius * (1.0 + math.sqrt(radical))
    base = rho * rho / denominator if denominator > 0 else 0.0
    return base + profile.a4 * rho**4 + profile.a6 * rho**6 + profile.a8 * rho**8


def spherical_sag_height_um(radius_um: float, aperture_radius_um: float, lens_height_um: float) -> float:
    if radius_um > aperture_radius_um or lens_height_um <= 0:
        return 0.0
    sphere_radius = lens_sphere_radius(aperture_radius_um, lens_height_um)
    return max(lens_height_um - sphere_radius + math.sqrt(max(sphere_radius * sphere_radius - radius_um * radius_um, 0.0)), 0.0)


def sag_height_um(
    radius_um: float,
    aperture_radius_um: float,
    lens_height_um: float,
    profile: OclSagProfile,
) -> float:
    if radius_um > aperture_radius_um or lens_height_um <= 0:
        return 0.0
    if profile.profile_type == "sphere":
        return spherical_sag_height_um(radius_um, aperture_radius_um, lens_height_um)
    center_drop = asphere_drop_um(0.0, profile, aperture_radius_um, lens_height_um)
    edge_drop = asphere_drop_um(aperture_radius_um, profile, aperture_radius_um, lens_height_um)
    local_drop = asphere_drop_um(radius_um, profile, aperture_radius_um, lens_height_um)
    if profile.normalize_edge:
        denom = edge_drop - center_drop
        if abs(denom) < 1e-12:
            return spherical_sag_height_um(radius_um, aperture_radius_um, lens_height_um)
        normalized = (local_drop - center_drop) / denom
        return max(lens_height_um * (1.0 - normalized), 0.0)
    return max(lens_height_um - (local_drop - center_drop), 0.0)


def inside_lens_volume(
    dx: float,
    dz: float,
    y: float,
    aperture_radius_um: float,
    geom: MicrolensArrayGeometry,
    profile: OclSagProfile,
    surface_map: OclSurfaceMap | None = None,
) -> bool:
    if surface_map is not None:
        height = bilinear_surface_height_um(dx, dz, surface_map)
    else:
        radius = math.hypot(dx, dz)
        height = sag_height_um(radius, aperture_radius_um, geom.lens_height, profile)
    if height <= 0.0:
        return False
    return geom.lens_bottom <= y <= geom.lens_bottom + height


def bayer_color(ix: int, iz: int) -> str:
    if iz % 2 == 0:
        return "red" if ix % 2 == 0 else "green"
    return "green" if ix % 2 == 0 else "blue"


def cfa_color_for_cell(ix: int, iz: int, pattern: str, fallback_color: str) -> str:
    if pattern == "uniform":
        return fallback_color
    if pattern == "bayer":
        return bayer_color(ix, iz)
    if pattern == "quad":
        return bayer_color(ix // 2, iz // 2)
    if pattern == "nona":
        return bayer_color(ix // 3, iz // 3)
    raise ValueError(f"Unsupported CFA pattern: {pattern}")


def cfa_shift_for_color(
    color: str,
    shifts: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    return shifts.get(color, (0.0, 0.0))


def transverse_k(frequency: float, cra_x_deg: float, cra_z_deg: float) -> tuple[float, float]:
    sx = math.sin(math.radians(cra_x_deg))
    sz = math.sin(math.radians(cra_z_deg))
    if sx * sx + sz * sz >= 1:
        raise ValueError("CRA x/z components exceed a propagating ray")
    return frequency * sx, frequency * sz


def source_phase(
    kx: float,
    kz: float,
    *,
    aperture_lens: OclLens | None = None,
    cell_x: float = 0.0,
    cell_z: float = 0.0,
    case: SweepCase | None = None,
):
    def amp(point: mp.Vector3):
        if aperture_lens is not None and case is not None:
            center_x = aperture_lens.x_um + aperture_lens.shift_x_um + case.aperture_shift_x
            center_z = aperture_lens.z_um + aperture_lens.shift_z_um + case.aperture_shift_z
            dx = periodic_delta(point.x, center_x, cell_x)
            dz = periodic_delta(point.z, center_z, cell_z)
            if aperture_lens.polygon_um is not None:
                if not point_in_polygon(dx, dz, aperture_lens.polygon_um):
                    return 0.0
            elif dx * dx + dz * dz > aperture_lens.aperture_radius_um**2:
                return 0.0
        return np.exp(1j * 2 * np.pi * (kx * point.x + kz * point.z))

    return amp


def make_supercell_material_function(
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    silicon: mp.Medium,
    cfa_media: dict[str, mp.Medium],
    passivation: mp.Medium,
    lens: mp.Medium,
    metal: mp.Medium,
    case: SweepCase,
    shield: dict[str, Any],
    cfa_pattern: str,
    color_channel: str,
    cfa_shifts: dict[str, tuple[float, float]],
    cfa_polygons: CfaPolygonSet,
    ocl_lenses: list[OclLens],
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
):
    cell_x = nx * geom.pitch
    cell_z = nz * geom.pitch
    x_centers = pixel_centers(nx, geom.pitch)
    z_centers = pixel_centers(nz, geom.pitch)
    shield_mode = shield["mode"]
    shield_active_half_width = (
        geom.active_half_width
        if shield_mode == "off"
        else shield_active_half_width_um(geom, shield)
    )

    def inside_shifted_square(x: float, z: float, sx: float, sz: float, half_width: float) -> bool:
        for xc in x_centers:
            dx = periodic_delta(x, xc + sx, cell_x)
            if abs(dx) > half_width:
                continue
            for zc in z_centers:
                dz = periodic_delta(z, zc + sz, cell_z)
                if abs(dz) <= half_width:
                    return True
        return False

    def nearest_shifted_pixel_local(
        x: float,
        z: float,
        sx: float,
        sz: float,
    ) -> tuple[float, float, int]:
        best: tuple[float, float, float, int] | None = None
        for ix, xc in enumerate(x_centers):
            dx = periodic_delta(x, xc + sx, cell_x)
            for zc in z_centers:
                dz = periodic_delta(z, zc + sz, cell_z)
                distance2 = dx * dx + dz * dz
                if best is None or distance2 < best[0]:
                    best = (distance2, dx, dz, ix)
        if best is None:
            raise RuntimeError("No pixel centers are available for shield mask evaluation")
        _, dx, dz, ix = best
        return dx, dz, ix

    def nearest_cfa_color(x: float, z: float) -> str:
        best: tuple[float, str] | None = None
        for ix, xc in enumerate(x_centers):
            for iz, zc in enumerate(z_centers):
                if (
                    (cfa_pattern == "bayer" and nx == 1 and nz == 1)
                    or (cfa_pattern == "quad" and nx <= 2 and nz <= 2)
                    or (cfa_pattern == "nona" and nx <= 3 and nz <= 3)
                ):
                    color = color_channel
                else:
                    color = cfa_color_for_cell(ix, iz, cfa_pattern, color_channel)
                shift_x, shift_z = cfa_shift_for_color(color, cfa_shifts)
                dx = periodic_delta(x, xc + shift_x, cell_x)
                dz = periodic_delta(z, zc + shift_z, cell_z)
                distance2 = dx * dx + dz * dz
                if best is None or distance2 < best[0]:
                    best = (distance2, color)
        if best is None:
            return color_channel
        return best[1]

    def cfa_color_for_index(ix: int, iz: int) -> str:
        if (
            (cfa_pattern == "bayer" and nx == 1 and nz == 1)
            or (cfa_pattern == "quad" and nx <= 2 and nz <= 2)
            or (cfa_pattern == "nona" and nx <= 3 and nz <= 3)
        ):
            return color_channel
        return cfa_color_for_cell(ix, iz, cfa_pattern, color_channel)

    def cfa_material_at(x: float, z: float):
        color = cfa_polygon_color_at(
            x,
            z,
            x_centers,
            z_centers,
            cell_x,
            cell_z,
            cfa_color_for_index,
            cfa_shifts,
            cfa_polygons,
            periodic=True,
        )
        if color:
            return cfa_media[color]
        if cfa_polygons.polygons:
            if cfa_polygons.background == "passivation":
                return passivation
            if cfa_polygons.background == "air":
                return mp.air
        return cfa_media[nearest_cfa_color(x, z)]

    def inside_shifted_lens(x: float, y: float, z: float) -> bool:
        if ocl_lenses:
            for item in ocl_lenses:
                dx = periodic_delta(x, item.x_um + item.shift_x_um + case.lens_shift_x_um, cell_x)
                dz = periodic_delta(z, item.z_um + item.shift_z_um + case.lens_shift_z_um, cell_z)
                if item.polygon_um is not None and not point_in_polygon(dx, dz, item.polygon_um):
                    continue
                profile = sag_profile_for_lens(item.lens_id, ocl_sag_profiles)
                surface_map = surface_map_for_lens(item.lens_id, ocl_surface_maps)
                if inside_lens_volume(dx, dz, y, item.aperture_radius_um, geom, profile, surface_map):
                    return True
            return False
        profile = sag_profile_for_lens(None, ocl_sag_profiles)
        surface_map = surface_map_for_lens(None, ocl_surface_maps)
        for xc in x_centers:
            dx = periodic_delta(x, xc + case.lens_shift_x_um, cell_x)
            for zc in z_centers:
                dz = periodic_delta(z, zc + case.lens_shift_z_um, cell_z)
                if inside_lens_volume(dx, dz, y, geom.lens_aperture_radius, geom, profile, surface_map):
                    return True
        return False

    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        z = point.z

        if geom.si_bottom <= y < geom.si_top:
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            cfa = cfa_material_at(x, z)
            if shield_mode == "off":
                return cfa
            if shield_mode == "edge":
                if inside_shifted_square(
                    x,
                    z,
                    case.aperture_shift_x,
                    case.aperture_shift_z,
                    shield_active_half_width,
                ):
                    return cfa
                return metal
            dx, dz, pair_index = nearest_shifted_pixel_local(
                x,
                z,
                case.aperture_shift_x,
                case.aperture_shift_z,
            )
            if shield_blocks_local_point(
                dx,
                dz,
                shield_active_half_width,
                shield_mode,
                pair_index=pair_index,
            ):
                return metal
            return cfa
        if geom.lens_bottom <= y <= geom.lens_top and inside_shifted_lens(x, y, z):
            return lens
        return mp.air

    return material


def split_regions(
    geom: MicrolensArrayGeometry,
    split_mode: str,
    split_gap_um: float,
    center_x_um: float = 0.0,
    center_z_um: float = 0.0,
    half_width_x_um: float | None = None,
    half_width_z_um: float | None = None,
    region_prefix: str = "pd",
) -> list[Region]:
    ax = geom.active_half_width if half_width_x_um is None else half_width_x_um
    az = geom.active_half_width if half_width_z_um is None else half_width_z_um
    gap = split_gap_um
    if gap >= 2 * min(ax, az):
        raise ValueError("split_gap_um is too large for the active area")

    def span(lo: float, hi: float) -> tuple[float, float]:
        return 0.5 * (lo + hi), hi - lo

    if split_mode == "dual-x":
        left_center, left_size = span(-ax, -0.5 * gap)
        right_center, right_size = span(0.5 * gap, ax)
        return [
            Region(f"{region_prefix}_left", "subpd", -1, 0, center_x_um + left_center, center_z_um, left_size, 2 * az),
            Region(f"{region_prefix}_right", "subpd", 1, 0, center_x_um + right_center, center_z_um, right_size, 2 * az),
        ]
    if split_mode == "dual-z":
        bottom_center, bottom_size = span(-az, -0.5 * gap)
        top_center, top_size = span(0.5 * gap, az)
        return [
            Region(f"{region_prefix}_bottom", "subpd", 0, -1, center_x_um, center_z_um + bottom_center, 2 * ax, bottom_size),
            Region(f"{region_prefix}_top", "subpd", 0, 1, center_x_um, center_z_um + top_center, 2 * ax, top_size),
        ]
    if split_mode == "quad":
        left_center, x_size = span(-ax, -0.5 * gap)
        right_center, _ = span(0.5 * gap, ax)
        bottom_center, z_size = span(-az, -0.5 * gap)
        top_center, _ = span(0.5 * gap, az)
        return [
            Region(f"{region_prefix}_q00_left_bottom", "subpd", -1, -1, center_x_um + left_center, center_z_um + bottom_center, x_size, z_size),
            Region(f"{region_prefix}_q10_right_bottom", "subpd", 1, -1, center_x_um + right_center, center_z_um + bottom_center, x_size, z_size),
            Region(f"{region_prefix}_q01_left_top", "subpd", -1, 1, center_x_um + left_center, center_z_um + top_center, x_size, z_size),
            Region(f"{region_prefix}_q11_right_top", "subpd", 1, 1, center_x_um + right_center, center_z_um + top_center, x_size, z_size),
        ]
    raise ValueError(f"Unsupported split mode: {split_mode}")


def pixel_regions(geom: MicrolensArrayGeometry, nx: int, nz: int) -> list[Region]:
    regions = []
    for iz, zc in enumerate(pixel_centers(nz, geom.pitch)):
        for ix, xc in enumerate(pixel_centers(nx, geom.pitch)):
            regions.append(
                Region(
                    region_id=f"pix_x{ix}_z{iz}",
                    kind="pixel",
                    ix=ix,
                    iz=iz,
                    x_um=xc,
                    z_um=zc,
                    sx_um=geom.pitch,
                    sz_um=geom.pitch,
                )
            )
    return regions


def collection_regions(
    mode: str,
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    split_mode: str,
    split_gap_um: float,
    collection_mode: str,
    ocl_lenses: list[OclLens],
    target_lens_id: str | None,
) -> list[Region]:
    effective_mode = "split-pd" if collection_mode == "auto" and mode == "split-pd-1x1" else collection_mode
    if effective_mode == "split-pd":
        if mode == "ocl-layout":
            target = choose_target_ocl_lens(ocl_lenses, target_lens_id)
            if target is None:
                raise ValueError("collection-mode split-pd with ocl-layout requires --ocl-layout")
            return split_regions(
                geom,
                split_mode,
                split_gap_um,
                center_x_um=target.x_um,
                center_z_um=target.z_um,
                half_width_x_um=0.5 * target.w * geom.pitch,
                half_width_z_um=0.5 * target.h * geom.pitch,
                region_prefix=f"{target.lens_id}_pd",
            )
        return split_regions(geom, split_mode, split_gap_um)
    if effective_mode not in {"auto", "pixel"}:
        raise ValueError(f"Unsupported collection mode: {collection_mode}")
    return pixel_regions(geom, nx, nz)


def stack_materials(
    stack_config: dict[str, Any],
    color_channel: str,
    wavelength_um: float,
    frequency: float,
) -> tuple[mp.Medium, dict[str, mp.Medium], mp.Medium, mp.Medium, mp.Medium, dict[str, Any]]:
    silicon, si_spec = medium_for_role(stack_config, "silicon", wavelength_um, frequency)
    passivation, passivation_spec = medium_for_role(
        stack_config, "passivation", wavelength_um, frequency
    )
    cfa_media: dict[str, mp.Medium] = {}
    cfa_specs: dict[str, Any] = {}
    for color in CFA_COLORS:
        role = material_role_for_color(color)
        cfa_media[color], cfa_specs[role] = medium_for_role(stack_config, role, wavelength_um, frequency)
    cfa_media["clear"] = passivation
    cfa_specs["cfa_clear"] = {
        **passivation_spec,
        "usage": "clear/monochrome transparent CFA proxy using passivation medium",
    }
    lens, lens_spec = medium_for_role(stack_config, "lens", wavelength_um, frequency)
    metal, metal_spec = metal_for_stack(stack_config)
    material_info = {
        "silicon": si_spec,
        **cfa_specs,
        "passivation": passivation_spec,
        "lens": lens_spec,
        "metal": metal_spec,
    }
    return silicon, cfa_media, passivation, lens, metal, material_info


def build_simulation(
    geom: MicrolensArrayGeometry,
    mode: str,
    nx: int,
    nz: int,
    case: SweepCase,
    wavelength_um: float,
    resolution: int,
    regions: list[Region],
    stack_config: dict[str, Any],
    color_channel: str,
    cfa_pattern: str,
    cfa_shifts: dict[str, tuple[float, float]],
    cfa_polygons: CfaPolygonSet,
    ocl_lenses: list[OclLens],
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    include_stack: bool,
    include_focal_map: bool,
    include_absorption_volume: bool,
    source_aperture_lens_id: str | None,
):
    frequency = 1 / wavelength_um
    kx, kz = transverse_k(frequency, case.cra_x_deg, case.cra_z_deg)
    cell_size = mp.Vector3(nx * geom.pitch, geom.cell_y, nz * geom.pitch)
    source_lens = choose_target_ocl_lens(ocl_lenses, source_aperture_lens_id) if source_aperture_lens_id else None
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ez,
        center=mp.Vector3(0, geom.source_y, 0),
        size=mp.Vector3(cell_size.x, 0, cell_size.z),
        amp_func=source_phase(kx, kz, aperture_lens=source_lens, cell_x=cell_size.x, cell_z=cell_size.z, case=case),
    )

    kwargs = {}
    extra_materials = []
    if include_stack:
        silicon, cfa_media, passivation, lens, metal, _ = stack_materials(
            stack_config, color_channel, wavelength_um, frequency
        )
        shield = shield_config_for_stack(stack_config)
        kwargs["material_function"] = make_supercell_material_function(
            geom,
            nx,
            nz,
            silicon,
            cfa_media,
            passivation,
            lens,
            metal,
            case,
            shield,
            cfa_pattern,
            color_channel,
            cfa_shifts,
            cfa_polygons,
            ocl_lenses,
            ocl_sag_profiles,
            ocl_surface_maps,
        )
        extra_materials = [silicon, *cfa_media.values(), passivation, lens]
        if shield["enabled"]:
            extra_materials.append(metal)

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml, direction=mp.Y)],
        sources=[source],
        resolution=resolution,
        k_point=mp.Vector3(kx, 0, kz),
        force_complex_fields=True,
        default_material=mp.air,
        extra_materials=extra_materials,
        **kwargs,
    )

    incident_flux = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mp.Vector3(0, geom.incident_monitor_y, 0), size=mp.Vector3(cell_size.x, 0, cell_size.z)),
    )

    full_top_flux = None
    full_bottom_flux = None
    top_fluxes = []
    bottom_fluxes = []
    focal_fields = None
    absorption_fields = None
    if include_stack:
        full_top_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_top_monitor_y, 0),
                size=mp.Vector3(cell_size.x, 0, cell_size.z),
            ),
        )
        full_bottom_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_bottom_monitor_y, 0),
                size=mp.Vector3(cell_size.x, 0, cell_size.z),
            ),
        )
        for region in regions:
            top_fluxes.append(
                sim.add_flux(
                    frequency,
                    0,
                    1,
                    mp.FluxRegion(
                        center=mp.Vector3(region.x_um, geom.si_top_monitor_y, region.z_um),
                        size=mp.Vector3(region.sx_um, 0, region.sz_um),
                    ),
                )
            )
            bottom_fluxes.append(
                sim.add_flux(
                    frequency,
                    0,
                    1,
                    mp.FluxRegion(
                        center=mp.Vector3(region.x_um, geom.si_bottom_monitor_y, region.z_um),
                        size=mp.Vector3(region.sx_um, 0, region.sz_um),
                    ),
                )
            )
        if include_focal_map:
            focal_fields = sim.add_dft_fields(
                [mp.Ez],
                frequency,
                0,
                1,
                center=mp.Vector3(0, geom.focal_plane_y, 0),
                size=mp.Vector3(cell_size.x, 0, cell_size.z),
            )
        if include_absorption_volume:
            absorption_fields = sim.add_dft_fields(
                [mp.Ex, mp.Ey, mp.Ez],
                frequency,
                0,
                1,
                center=mp.Vector3(0, 0.5 * (geom.si_top + geom.si_bottom), 0),
                size=mp.Vector3(cell_size.x, geom.si_thickness, cell_size.z),
            )

    return (
        sim,
        incident_flux,
        full_top_flux,
        full_bottom_flux,
        top_fluxes,
        bottom_fluxes,
        focal_fields,
        absorption_fields,
    )


def focal_region_fraction(
    focal_map: np.ndarray,
    nx: int,
    nz: int,
    geom: MicrolensArrayGeometry,
    region: Region,
) -> float:
    total = float(np.sum(focal_map))
    if total <= 0:
        return 0.0
    x_values = np.linspace(-0.5 * nx * geom.pitch, 0.5 * nx * geom.pitch, focal_map.shape[0])
    z_values = np.linspace(-0.5 * nz * geom.pitch, 0.5 * nz * geom.pitch, focal_map.shape[1])
    x_grid, z_grid = np.meshgrid(x_values, z_values, indexing="ij")
    mask = (
        (np.abs(x_grid - region.x_um) <= 0.5 * region.sx_um)
        & (np.abs(z_grid - region.z_um) <= 0.5 * region.sz_um)
    )
    return float(np.sum(focal_map[mask]) / total)


def focal_map_metrics(
    focal_map: np.ndarray | None,
    nx: int,
    nz: int,
    geom: MicrolensArrayGeometry,
    case: SweepCase,
    target_lens: OclLens | None,
) -> dict[str, float]:
    zero_metrics = {
        "focal_centroid_x_um": 0.0,
        "focal_centroid_z_um": 0.0,
        "focal_centroid_shift_x_um": 0.0,
        "focal_centroid_shift_z_um": 0.0,
        "focal_rms_radius_um": 0.0,
        "focal_target_fraction": 0.0,
    }
    if focal_map is None:
        return zero_metrics
    total = float(np.sum(focal_map))
    if total <= 0:
        return zero_metrics
    x_values = np.linspace(-0.5 * nx * geom.pitch, 0.5 * nx * geom.pitch, focal_map.shape[0])
    z_values = np.linspace(-0.5 * nz * geom.pitch, 0.5 * nz * geom.pitch, focal_map.shape[1])
    x_grid, z_grid = np.meshgrid(x_values, z_values, indexing="ij")
    cx = float(np.sum(focal_map * x_grid) / total)
    cz = float(np.sum(focal_map * z_grid) / total)
    rms = float(np.sqrt(np.sum(focal_map * ((x_grid - cx) ** 2 + (z_grid - cz) ** 2)) / total))
    if target_lens is None:
        target_x = case.lens_shift_x_um
        target_z = case.lens_shift_z_um
        target_mask = (
            (np.abs(x_grid - target_x) <= 0.5 * geom.pitch)
            & (np.abs(z_grid - target_z) <= 0.5 * geom.pitch)
        )
    else:
        target_x = target_lens.x_um + target_lens.shift_x_um + case.lens_shift_x_um
        target_z = target_lens.z_um + target_lens.shift_z_um + case.lens_shift_z_um
        periodic_x = np.vectorize(periodic_delta)(x_grid, target_x, nx * geom.pitch)
        periodic_z = np.vectorize(periodic_delta)(z_grid, target_z, nz * geom.pitch)
        if target_lens.polygon_um is not None:
            contains = np.vectorize(lambda x, z: point_in_polygon(float(x), float(z), target_lens.polygon_um))
            target_mask = contains(periodic_x, periodic_z)
        else:
            target_mask = periodic_x * periodic_x + periodic_z * periodic_z <= target_lens.aperture_radius_um**2
    return {
        "focal_centroid_x_um": cx,
        "focal_centroid_z_um": cz,
        "focal_centroid_shift_x_um": cx - target_x,
        "focal_centroid_shift_z_um": cz - target_z,
        "focal_rms_radius_um": rms,
        "focal_target_fraction": float(np.sum(focal_map[target_mask]) / total),
    }


def axis_centers(span_um: float, count: int) -> np.ndarray:
    step = span_um / count
    return np.linspace(-0.5 * span_um + 0.5 * step, 0.5 * span_um - 0.5 * step, count)


def grid_rounding_axis(axis: str, requested_um: float, resolution: int) -> dict[str, float | int | bool | str]:
    requested_pixels = requested_um * resolution
    rounded_pixels = int(round(requested_pixels))
    effective_um = rounded_pixels / resolution
    error_um = effective_um - requested_um
    return {
        "axis": axis,
        "requested_um": requested_um,
        "requested_pixels": requested_pixels,
        "rounded_pixels": rounded_pixels,
        "effective_um": effective_um,
        "rounding_error_um": error_um,
        "rounding_error_nm": error_um * 1000.0,
        "integer_grid": abs(requested_pixels - rounded_pixels) < 1.0e-9,
    }


def grid_rounding_metadata(
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    resolution: int,
) -> dict[str, Any]:
    axes = [
        grid_rounding_axis("x", nx * geom.pitch, resolution),
        grid_rounding_axis("y", geom.cell_y, resolution),
        grid_rounding_axis("z", nz * geom.pitch, resolution),
    ]
    lateral_axes = [axis for axis in axes if axis["axis"] in {"x", "z"}]
    return {
        "resolution_px_per_um": resolution,
        "axes": axes,
        "all_axes_integer_grid": all(bool(axis["integer_grid"]) for axis in axes),
        "lateral_period_axes_integer_grid": all(bool(axis["integer_grid"]) for axis in lateral_axes),
        "max_abs_rounding_error_nm": max(abs(float(axis["rounding_error_nm"])) for axis in axes),
        "note": "Meep rounds simulation-cell dimensions to an integer number of pixels. Nonzero rounding changes the effective simulated dimensions at that resolution.",
    }


def grid_resolution_metadata(
    geom: MicrolensArrayGeometry,
    stack_config: dict[str, Any],
    wavelength_nm: float,
    resolution: int,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
) -> dict[str, Any]:
    wavelength_um = wavelength_nm / 1000.0
    si_n, _, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    si_internal_wavelength_um = wavelength_um / si_n
    feature_sizes_um = {
        "passivation_thickness_pixels": geom.passivation_thickness,
        "lens_edge_gap_pixels": geom.lens_edge_gap,
        "cfa_thickness_pixels": geom.cfa_thickness,
    }
    feature_pixels = {
        name: size_um * resolution
        for name, size_um in feature_sizes_um.items()
    }
    positive_critical_features = [
        feature_pixels["passivation_thickness_pixels"],
        *([feature_pixels["lens_edge_gap_pixels"]] if geom.lens_edge_gap > 0.0 else []),
    ]
    minimum_critical_feature_pixels = min(positive_critical_features) if positive_critical_features else 0.0
    si_internal_wavelength_pixels = si_internal_wavelength_um * resolution
    recommended_si_resolution = math.ceil(min_si_wavelength_pixels / si_internal_wavelength_um)
    positive_feature_um = [
        geom.passivation_thickness,
        *([geom.lens_edge_gap] if geom.lens_edge_gap > 0.0 else []),
    ]
    min_feature_um = min(positive_feature_um) if positive_feature_um else 0.0
    recommended_feature_resolution = math.ceil(min_feature_pixels / min_feature_um)
    recommended_resolution = max(recommended_si_resolution, recommended_feature_resolution)
    si_wavelength_gate_pass = si_internal_wavelength_pixels >= min_si_wavelength_pixels
    critical_feature_gate_pass = minimum_critical_feature_pixels >= min_feature_pixels
    notes = []
    if not si_wavelength_gate_pass:
        notes.append(
            f"Si internal wavelength is only {si_internal_wavelength_pixels:.2f} grid pixels; "
            f"need >= {min_si_wavelength_pixels:g}."
        )
    if not critical_feature_gate_pass:
        notes.append(
            f"Minimum critical optical feature is only {minimum_critical_feature_pixels:.2f} grid pixels; "
            f"need >= {min_feature_pixels:g}."
        )
    return {
        "grid_dx_um": 1.0 / resolution,
        "si_n_at_wavelength": si_n,
        "si_internal_wavelength_um": si_internal_wavelength_um,
        "si_internal_wavelength_pixels": si_internal_wavelength_pixels,
        **feature_pixels,
        "minimum_critical_feature_pixels": minimum_critical_feature_pixels,
        "min_feature_pixels_required": min_feature_pixels,
        "min_si_wavelength_pixels_required": min_si_wavelength_pixels,
        "recommended_si_wavelength_resolution_px_per_um": recommended_si_resolution,
        "recommended_feature_resolution_px_per_um": recommended_feature_resolution,
        "recommended_min_resolution_px_per_um": recommended_resolution,
        "si_wavelength_gate_pass": si_wavelength_gate_pass,
        "critical_feature_gate_pass": critical_feature_gate_pass,
        "grid_resolution_gate_pass": si_wavelength_gate_pass and critical_feature_gate_pass,
        "grid_resolution_notes": " ".join(notes) if notes else "Grid resolution meets configured feature and Si wavelength gates.",
    }


def si_volume_absorption_density(
    sim: mp.Simulation,
    absorption_fields,
    stack_config: dict[str, Any],
    wavelength_um: float,
) -> tuple[np.ndarray, dict[str, float]]:
    ex = np.asarray(sim.get_dft_array(absorption_fields, mp.Ex, 0))
    ey = np.asarray(sim.get_dft_array(absorption_fields, mp.Ey, 0))
    ez = np.asarray(sim.get_dft_array(absorption_fields, mp.Ez, 0))
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    if intensity.ndim != 3:
        raise RuntimeError(f"Expected 3D Si absorption field, got shape {intensity.shape}")
    si_n, si_k, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    eps_imag = 2.0 * si_n * si_k
    density = eps_imag * intensity
    return density, {
        "silicon_n": si_n,
        "silicon_k": si_k,
        "silicon_eps_imag": eps_imag,
    }


def volume_absorption_integrals(
    density: np.ndarray,
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    regions: list[Region],
) -> tuple[float, float, dict[str, float]]:
    cell_x = nx * geom.pitch
    cell_z = nz * geom.pitch
    dx = cell_x / density.shape[0]
    dy = geom.si_thickness / density.shape[1]
    dz = cell_z / density.shape[2]
    dvol = dx * dy * dz
    total_raw = float(np.sum(density) * dvol)
    x_values = axis_centers(cell_x, density.shape[0])
    z_values = axis_centers(cell_z, density.shape[2])
    region_raw: dict[str, float] = {}
    for region in regions:
        x_mask = np.abs(x_values - region.x_um) <= 0.5 * region.sx_um
        z_mask = np.abs(z_values - region.z_um) <= 0.5 * region.sz_um
        mask_xz = x_mask[:, None] & z_mask[None, :]
        region_raw[region.region_id] = float(np.sum(density * mask_xz[:, None, :]) * dvol)
    return total_raw, dvol, region_raw


def tcad_generation_profile_1d(
    density: np.ndarray,
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    volume_scale_to_flux: float,
    incident_photon_flux_cm2_s: float,
) -> list[dict]:
    cell_x = nx * geom.pitch
    cell_z = nz * geom.pitch
    dx = cell_x / density.shape[0]
    dz = cell_z / density.shape[2]
    y_rel = axis_centers(geom.si_thickness, density.shape[1])
    y_mid = 0.5 * (geom.si_top + geom.si_bottom)
    y_abs = y_mid + y_rel
    depth_um = geom.si_top - y_abs

    # density * dx * dz is absorption fraction per micron of Si depth.
    absorption_fraction_per_um = (
        np.sum(density, axis=(0, 2)) * dx * dz * volume_scale_to_flux
    )
    absorption_fraction_per_cm = absorption_fraction_per_um * 1.0e4
    generation_cm3_s = incident_photon_flux_cm2_s * absorption_fraction_per_cm
    peak_generation = max(float(np.max(generation_cm3_s)), 1e-300)

    order = np.argsort(depth_um)
    rows = []
    for index in order:
        rows.append(
            {
                "depth_um_from_si_top": float(depth_um[index]),
                "y_um": float(y_abs[index]),
                "absorption_fraction_per_um": float(absorption_fraction_per_um[index]),
                "absorption_fraction_per_cm": float(absorption_fraction_per_cm[index]),
                "generation_cm3_s": float(generation_cm3_s[index]),
                "generation_normalized": float(generation_cm3_s[index] / peak_generation),
                "incident_photon_flux_cm2_s": float(incident_photon_flux_cm2_s),
            }
        )
    return rows


def tcad_generation_map_x_depth(
    density: np.ndarray,
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    volume_scale_to_flux: float,
    incident_photon_flux_cm2_s: float,
) -> dict[str, np.ndarray]:
    """Return a z-collapsed FDTD generation map for 2D TCAD.

    The calibrated density integrates to an absorption fraction over the full
    optical supercell. For a 2D x-depth electrical cross-section, we average over
    the out-of-plane z direction and keep the lateral x dependence.
    """
    cell_x = nx * geom.pitch
    cell_z = nz * geom.pitch
    dz = cell_z / density.shape[2]
    x_um = axis_centers(cell_x, density.shape[0])
    y_rel = axis_centers(geom.si_thickness, density.shape[1])
    y_mid = 0.5 * (geom.si_top + geom.si_bottom)
    y_abs = y_mid + y_rel
    depth_um = geom.si_top - y_abs

    scaled_density = density * volume_scale_to_flux
    # Sum over z, then scale by the full x span to obtain the z-averaged
    # volumetric generation rate used by the 2D DEVSIM cross-section.
    absorption_fraction_per_um2 = np.sum(scaled_density, axis=2) * dz
    generation_cm3_s = (
        incident_photon_flux_cm2_s
        * cell_x
        * 1.0e4
        * absorption_fraction_per_um2
    )

    order = np.argsort(depth_um)
    return {
        "x_um": x_um,
        "depth_um_from_si_top": depth_um[order],
        "y_um": y_abs[order],
        "generation_cm3_s": generation_cm3_s[:, order],
        "absorption_fraction_per_um2": absorption_fraction_per_um2[:, order],
    }


def tcad_generation_volume_3d(
    density: np.ndarray,
    geom: MicrolensArrayGeometry,
    nx: int,
    nz: int,
    volume_scale_to_flux: float,
    incident_photon_flux_cm2_s: float,
) -> dict[str, np.ndarray]:
    cell_x = nx * geom.pitch
    cell_z = nz * geom.pitch
    x_um = axis_centers(cell_x, density.shape[0])
    z_um = axis_centers(cell_z, density.shape[2])
    y_rel = axis_centers(geom.si_thickness, density.shape[1])
    y_mid = 0.5 * (geom.si_top + geom.si_bottom)
    y_abs = y_mid + y_rel
    depth_um = geom.si_top - y_abs
    order = np.argsort(depth_um)

    scaled_density = density * volume_scale_to_flux
    generation_cm3_s = (
        incident_photon_flux_cm2_s
        * cell_x
        * cell_z
        * 1.0e4
        * scaled_density
    )
    return {
        "x_um": x_um,
        "depth_um_from_si_top": depth_um[order],
        "y_um": y_abs[order],
        "z_um": z_um,
        "generation_cm3_s": generation_cm3_s[:, order, :],
        "absorption_fraction_per_um3": scaled_density[:, order, :],
    }


def run_termination(
    geom: MicrolensArrayGeometry,
    after_source_time: float,
    decay_by: float,
    decay_check_time: float,
):
    if decay_by <= 0:
        return after_source_time
    return mp.stop_when_fields_decayed(
        decay_check_time,
        mp.Ez,
        mp.Vector3(0, geom.si_bottom_monitor_y, 0),
        decay_by,
    )


def summary_from_rows(
    mode: str,
    wavelength_nm: float,
    case: SweepCase,
    rows: list[dict],
    pupil_integrated: bool,
    pupil_ray_count: int,
) -> dict:
    response_array = np.asarray([row["response"] for row in rows], dtype=float)
    summary = {
        "schema": "camera_supercell_optical_lut_v2",
        "mode": mode,
        "color_channel": rows[0].get("color_channel") if rows else "",
        "wavelength_nm": wavelength_nm,
        "case": case.name,
        "field_x_norm": case.field_x_norm,
        "field_z_norm": case.field_z_norm,
        "cra_x_deg": case.cra_x_deg,
        "cra_z_deg": case.cra_z_deg,
        "lens_shift_x_um": case.lens_shift_x_um,
        "lens_shift_z_um": case.lens_shift_z_um,
        "pupil_integrated": pupil_integrated,
        "pupil_ray_count": pupil_ray_count,
        "total_response": float(np.sum(response_array)),
        "max_region_response": float(np.max(response_array)),
        "min_region_response": float(np.min(response_array)),
    }
    if rows:
        for key in (
            "cfa_pattern",
            "cfa_shift_red_x_um",
            "cfa_shift_green_x_um",
            "cfa_shift_blue_x_um",
            "cfa_shift_red_z_um",
            "cfa_shift_green_z_um",
            "cfa_shift_blue_z_um",
            "cfa_polygon_count",
            "cfa_polygon_background",
            "ocl_layout_name",
            "ocl_lens_count",
            "collection_mode",
            "target_lens_id",
            "source_aperture_lens_id",
            "source_aperture_enabled",
            "split_mode",
            "shield_mode",
            "shield_mask_edge_width_um",
            "incident_monitor_net_power_normalized",
            "total_si_absorption_fraction_estimate",
            "signed_flux_si_absorption_fraction_diagnostic",
            "volume_absorption_total_raw",
            "volume_absorption_scale_to_flux",
            "volume_absorption_region_fraction_sum",
            "focal_centroid_x_um",
            "focal_centroid_z_um",
            "focal_centroid_shift_x_um",
            "focal_centroid_shift_z_um",
            "focal_rms_radius_um",
            "focal_target_fraction",
        ):
            if key in rows[0]:
                summary[key] = rows[0][key]
    if any(row.get("region_kind") == "subpd" for row in rows):
        by_id = {row["region_id"]: row["response"] for row in rows}
        if "pd_left" in by_id and "pd_right" in by_id:
            denom = by_id["pd_left"] + by_id["pd_right"]
            summary["split_phase_x_proxy"] = (by_id["pd_right"] - by_id["pd_left"]) / denom if denom else 0.0
        if "pd_bottom" in by_id and "pd_top" in by_id:
            denom = by_id["pd_bottom"] + by_id["pd_top"]
            summary["split_phase_z_proxy"] = (by_id["pd_top"] - by_id["pd_bottom"]) / denom if denom else 0.0
        quad_left = sum(value for key, value in by_id.items() if "left" in key)
        quad_right = sum(value for key, value in by_id.items() if "right" in key)
        quad_bottom = sum(value for key, value in by_id.items() if "bottom" in key)
        quad_top = sum(value for key, value in by_id.items() if "top" in key)
        if quad_left + quad_right:
            summary["split_phase_x_proxy"] = (quad_right - quad_left) / (quad_right + quad_left)
        if quad_bottom + quad_top:
            summary["split_phase_z_proxy"] = (quad_top - quad_bottom) / (quad_top + quad_bottom)
    return summary


def run_one(
    geom: MicrolensArrayGeometry,
    mode: str,
    nx: int,
    nz: int,
    case: SweepCase,
    wavelength_nm: float,
    resolution: int,
    after_source_time: float,
    decay_by: float,
    decay_check_time: float,
    regions: list[Region],
    stack_config: dict[str, Any],
    color_channel: str,
    cfa_pattern: str,
    cfa_shifts: dict[str, tuple[float, float]],
    cfa_polygons: CfaPolygonSet,
    ocl_lenses: list[OclLens],
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    ocl_layout_name: str,
    collection_mode: str,
    target_lens_id: str | None,
    split_mode: str,
    incident_photon_flux_cm2_s: float,
    source_aperture_lens_id: str | None,
):
    wavelength_um = wavelength_nm / 1000
    shield = shield_config_for_stack(stack_config)
    effective_source_aperture_lens_id = (
        target_lens_id if source_aperture_lens_id == "target" else source_aperture_lens_id
    )
    ref_sim, ref_incident, _, _, _, _, _, _ = build_simulation(
        geom,
        mode,
        nx,
        nz,
        case,
        wavelength_um,
        resolution,
        regions,
        stack_config,
        color_channel,
        cfa_pattern,
        cfa_shifts,
        cfa_polygons,
        ocl_lenses,
        ocl_sag_profiles,
        ocl_surface_maps,
        include_stack=False,
        include_focal_map=False,
        include_absorption_volume=False,
        source_aperture_lens_id=effective_source_aperture_lens_id,
    )
    ref_sim.run(
        until_after_sources=run_termination(
            geom, after_source_time, decay_by, decay_check_time
        )
    )
    incident_flux = mp.get_fluxes(ref_incident)[0]
    downward_sign = 1 if incident_flux >= 0 else -1
    incident_power = abs(incident_flux)

    sim, full_incident, full_top_flux, full_bottom_flux, top_fluxes, bottom_fluxes, focal_fields, absorption_fields = build_simulation(
        geom,
        mode,
        nx,
        nz,
        case,
        wavelength_um,
        resolution,
        regions,
        stack_config,
        color_channel,
        cfa_pattern,
        cfa_shifts,
        cfa_polygons,
        ocl_lenses,
        ocl_sag_profiles,
        ocl_surface_maps,
        include_stack=True,
        include_focal_map=True,
        include_absorption_volume=True,
        source_aperture_lens_id=effective_source_aperture_lens_id,
    )
    sim.run(
        until_after_sources=run_termination(
            geom, after_source_time, decay_by, decay_check_time
        )
    )

    incident_full_raw = mp.get_fluxes(full_incident)[0]
    full_top_raw = mp.get_fluxes(full_top_flux)[0]
    full_bottom_raw = mp.get_fluxes(full_bottom_flux)[0]
    incident_full = incident_full_raw * downward_sign
    full_top_power = full_top_raw * downward_sign
    full_bottom_power = full_bottom_raw * downward_sign
    signed_total_si_absorption = (full_bottom_raw - full_top_raw) / incident_power
    total_si_absorption = abs(signed_total_si_absorption)
    focal_map = None
    if focal_fields is not None:
        focal = np.abs(np.squeeze(sim.get_dft_array(focal_fields, mp.Ez, 0))) ** 2
        focal_map = focal / max(float(np.max(focal)), 1e-30)
    target_lens = choose_target_ocl_lens(ocl_lenses, target_lens_id) if ocl_lenses else None
    focal_metrics = focal_map_metrics(focal_map, nx, nz, geom, case, target_lens)

    absorption_density, si_optical = si_volume_absorption_density(
        sim, absorption_fields, stack_config, wavelength_um
    )
    volume_total_raw, volume_dv, region_absorption_raw = volume_absorption_integrals(
        absorption_density, geom, nx, nz, regions
    )
    volume_scale_to_flux = total_si_absorption / volume_total_raw if volume_total_raw else 0.0
    region_fraction_sum = (
        sum(region_absorption_raw.values()) / volume_total_raw if volume_total_raw else 0.0
    )
    tcad_profile = tcad_generation_profile_1d(
        absorption_density,
        geom,
        nx,
        nz,
        volume_scale_to_flux,
        incident_photon_flux_cm2_s,
    )
    tcad_map_2d = tcad_generation_map_x_depth(
        absorption_density,
        geom,
        nx,
        nz,
        volume_scale_to_flux,
        incident_photon_flux_cm2_s,
    )
    tcad_volume_3d = tcad_generation_volume_3d(
        absorption_density,
        geom,
        nx,
        nz,
        volume_scale_to_flux,
        incident_photon_flux_cm2_s,
    )

    responses = []
    rows = []
    for region, top, bottom in zip(regions, top_fluxes, bottom_fluxes):
        top_raw = mp.get_fluxes(top)[0]
        bottom_raw = mp.get_fluxes(bottom)[0]
        top_power = top_raw * downward_sign
        bottom_power = bottom_raw * downward_sign
        regional_flux_response = (bottom_raw - top_raw) / incident_power
        focal_fraction = (
            focal_region_fraction(focal_map, nx, nz, geom, region)
            if focal_map is not None
            else 0.0
        )
        region_raw = region_absorption_raw[region.region_id]
        volume_fraction = region_raw / volume_total_raw if volume_total_raw else 0.0
        response = volume_scale_to_flux * region_raw
        responses.append(response)
        rows.append(
            {
                "schema": "camera_supercell_optical_lut_v2",
                "mode": mode,
                "color_channel": color_channel,
                "wavelength_nm": wavelength_nm,
                "case": case.name,
                "field_x_norm": case.field_x_norm,
                "field_z_norm": case.field_z_norm,
                "cra_x_deg": case.cra_x_deg,
                "cra_z_deg": case.cra_z_deg,
                "lens_shift_x_um": case.lens_shift_x_um,
                "lens_shift_z_um": case.lens_shift_z_um,
                "aperture_shift_x_um": case.aperture_shift_x,
                "aperture_shift_z_um": case.aperture_shift_z,
                "cfa_pattern": cfa_pattern,
                "cfa_shift_red_x_um": cfa_shifts["red"][0],
                "cfa_shift_green_x_um": cfa_shifts["green"][0],
                "cfa_shift_blue_x_um": cfa_shifts["blue"][0],
                "cfa_shift_red_z_um": cfa_shifts["red"][1],
                "cfa_shift_green_z_um": cfa_shifts["green"][1],
                "cfa_shift_blue_z_um": cfa_shifts["blue"][1],
                "cfa_polygon_count": len(cfa_polygons.polygons),
                "cfa_polygon_background": cfa_polygons.background,
                "ocl_layout_name": ocl_layout_name,
                "ocl_lens_count": len(ocl_lenses),
                "collection_mode": collection_mode,
                "target_lens_id": target_lens_id or "",
                "source_aperture_lens_id": effective_source_aperture_lens_id or "",
                "source_aperture_enabled": bool(effective_source_aperture_lens_id),
                "split_mode": split_mode if collection_mode == "split-pd" or mode == "split-pd-1x1" else "",
                "shield_mode": shield["mode"],
                "shield_mask_edge_width_um": shield["mask_edge_width_um"],
                "region_id": region.region_id,
                "region_kind": region.kind,
                "region_ix": region.ix,
                "region_iz": region.iz,
                "region_x_um": region.x_um,
                "region_z_um": region.z_um,
                "region_sx_um": region.sx_um,
                "region_sz_um": region.sz_um,
                "incident_monitor_net_power_normalized": incident_full / incident_power,
                "full_si_top_net_power_normalized": full_top_power / incident_power,
                "full_si_bottom_net_power_normalized": full_bottom_power / incident_power,
                "total_si_absorption_fraction_estimate": total_si_absorption,
                "signed_flux_si_absorption_fraction_diagnostic": signed_total_si_absorption,
                "focal_region_fraction": focal_fraction,
                "volume_absorption_region_fraction": volume_fraction,
                "volume_absorption_region_raw": region_raw,
                "volume_absorption_total_raw": volume_total_raw,
                "volume_absorption_voxel_um3": volume_dv,
                "volume_absorption_scale_to_flux": volume_scale_to_flux,
                "volume_absorption_region_fraction_sum": region_fraction_sum,
                **focal_metrics,
                "silicon_n": si_optical["silicon_n"],
                "silicon_k": si_optical["silicon_k"],
                "silicon_eps_imag": si_optical["silicon_eps_imag"],
                "region_top_net_power_normalized": top_power / incident_power,
                "region_bottom_net_power_normalized": bottom_power / incident_power,
                "regional_flux_response_diagnostic": regional_flux_response,
                "response_model": "si_volume_absorption_flux_calibrated_v1",
                "response": response,
            }
        )

    summary = summary_from_rows(
        mode,
        wavelength_nm,
        case,
        rows,
        pupil_integrated=False,
        pupil_ray_count=1,
    )

    return rows, summary, focal_map, tcad_profile, tcad_map_2d, tcad_volume_3d


def response_matrix(mode: str, nx: int, nz: int, regions: list[Region], rows: list[dict]) -> np.ndarray:
    if mode == "split-pd-1x1":
        xs = sorted({region.ix for region in regions})
        zs = sorted({region.iz for region in regions})
        matrix = np.zeros((len(zs), len(xs)))
        x_index = {value: idx for idx, value in enumerate(xs)}
        z_index = {value: idx for idx, value in enumerate(zs)}
        for region, row in zip(regions, rows):
            matrix[z_index[region.iz], x_index[region.ix]] = row["response"]
        return matrix

    matrix = np.zeros((nz, nx))
    for row in rows:
        matrix[int(row["region_iz"]), int(row["region_ix"])] = row["response"]
    return matrix


def save_response_plot(
    output_dir: Path,
    mode: str,
    nx: int,
    nz: int,
    regions: list[Region],
    grouped_rows: list[list[dict]],
    summaries: list[dict],
) -> None:
    nplots = len(grouped_rows)
    fig, axes = plt.subplots(1, nplots, figsize=(4.2 * nplots, 4), constrained_layout=True)
    if nplots == 1:
        axes = [axes]
    vmax = max(max(row["response"] for row in rows) for rows in grouped_rows)
    for axis, rows, summary in zip(axes, grouped_rows, summaries):
        matrix = response_matrix(mode, nx, nz, regions, rows)
        image = axis.imshow(matrix, origin="lower", cmap="magma", vmin=0, vmax=max(vmax, 1e-30))
        axis.set_title(
            f"{summary['case']}\n{summary['wavelength_nm']:g} nm, CRA x/z="
            f"{summary['cra_x_deg']:g}/{summary['cra_z_deg']:g}"
        )
        axis.set_xlabel("region x index")
        axis.set_ylabel("region z index")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "response_maps.png", dpi=180)
    plt.close(fig)


def save_focal_plot(
    output_dir: Path,
    nx: int,
    nz: int,
    geom: MicrolensArrayGeometry,
    focal_maps: list[np.ndarray],
    summaries: list[dict],
) -> None:
    if not focal_maps:
        return
    nplots = len(focal_maps)
    fig, axes = plt.subplots(1, nplots, figsize=(4.2 * nplots, 4), constrained_layout=True)
    if nplots == 1:
        axes = [axes]
    extent = [
        -0.5 * nx * geom.pitch,
        0.5 * nx * geom.pitch,
        -0.5 * nz * geom.pitch,
        0.5 * nz * geom.pitch,
    ]
    for axis, fmap, summary in zip(axes, focal_maps, summaries):
        image = axis.imshow(fmap.T, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(f"{summary['case']}\n{summary['wavelength_nm']:g} nm")
        axis.set_xlabel("x (um)")
        axis.set_ylabel("z (um)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "focal_maps.png", dpi=180)
    plt.close(fig)


def default_cases_for_mode(mode: str) -> str:
    if mode == "split-pd-1x1":
        return "center:0:0:0:0:0:0,edge20x:20:0:1:0:0:0"
    return "center:0:0:0:0:0:0,edge20x_uncomp:20:0:1:0:0:0,edge20x_ocl:20:0:1:0:-0.18:0"


def build_pupil_rays(case: SweepCase, f_number: float | None, samples: int) -> list[PupilRay]:
    if samples <= 1 or f_number is None or f_number <= 0:
        return [PupilRay(0, 0.0, 0.0, 1.0, case.cra_x_deg, case.cra_z_deg)]

    numerical_aperture = 1.0 / (2.0 * f_number)
    base_sx = math.sin(math.radians(case.cra_x_deg))
    base_sz = math.sin(math.radians(case.cra_z_deg))
    coords = np.linspace(-1.0, 1.0, samples)
    rays = []
    for u in coords:
        for v in coords:
            if u * u + v * v > 1.0 + 1e-12:
                continue
            sx = base_sx + float(u) * numerical_aperture
            sz = base_sz + float(v) * numerical_aperture
            if sx * sx + sz * sz >= 0.999:
                continue
            rays.append(
                PupilRay(
                    ray_index=len(rays),
                    pupil_u=float(u),
                    pupil_v=float(v),
                    weight=1.0,
                    cra_x_deg=math.degrees(math.asin(sx)),
                    cra_z_deg=math.degrees(math.asin(sz)),
                )
            )
    if not rays:
        raise ValueError("No propagating pupil rays generated")
    weight = 1.0 / len(rays)
    return [replace(ray, weight=weight) for ray in rays]


def add_ray_metadata(rows: list[dict], ray: PupilRay, ray_count: int, integrated: bool) -> list[dict]:
    updated = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "pupil_integrated": integrated,
                "pupil_ray_count": ray_count,
                "pupil_ray_index": ray.ray_index,
                "pupil_u": ray.pupil_u,
                "pupil_v": ray.pupil_v,
                "pupil_weight": ray.weight,
                "ray_cra_x_deg": ray.cra_x_deg,
                "ray_cra_z_deg": ray.cra_z_deg,
            }
        )
        updated.append(item)
    return updated


def weighted_value(values: list[Any], weights: list[float]) -> Any:
    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
        return float(sum(float(value) * weight for value, weight in zip(values, weights)))
    return values[0]


def aggregate_pupil_rows(
    mode: str,
    wavelength_nm: float,
    base_case: SweepCase,
    ray_rows: list[tuple[PupilRay, list[dict]]],
) -> tuple[list[dict], dict]:
    weights = [ray.weight for ray, _ in ray_rows]
    region_count = len(ray_rows[0][1])
    aggregated_rows = []
    for region_index in range(region_count):
        keys = set().union(*(rows[region_index].keys() for _, rows in ray_rows))
        item = {}
        for key in keys:
            values = [rows[region_index].get(key) for _, rows in ray_rows]
            if any(value is None for value in values):
                item[key] = values[0]
            else:
                item[key] = weighted_value(values, weights)
        item.update(
            {
                "case": base_case.name,
                "cra_x_deg": base_case.cra_x_deg,
                "cra_z_deg": base_case.cra_z_deg,
                "pupil_integrated": True,
                "pupil_ray_count": len(ray_rows),
                "pupil_ray_index": -1,
                "pupil_u": 0.0,
                "pupil_v": 0.0,
                "pupil_weight": 1.0,
                "ray_cra_x_deg": float("nan"),
                "ray_cra_z_deg": float("nan"),
            }
        )
        aggregated_rows.append(item)
    summary = summary_from_rows(
        mode,
        wavelength_nm,
        base_case,
        aggregated_rows,
        pupil_integrated=True,
        pupil_ray_count=len(ray_rows),
    )
    return aggregated_rows, summary


def weighted_focal_map(ray_focals: list[tuple[PupilRay, np.ndarray | None]]) -> np.ndarray | None:
    available = [(ray, fmap) for ray, fmap in ray_focals if fmap is not None]
    if not available:
        return None
    total = np.zeros_like(available[0][1], dtype=float)
    for ray, fmap in available:
        total += ray.weight * fmap
    peak = float(np.max(total))
    return total / peak if peak > 0 else total


def add_profile_metadata(
    profile_rows: list[dict],
    mode: str,
    color_channel: str,
    wavelength_nm: float,
    case: SweepCase,
    ray: PupilRay,
    ray_count: int,
    integrated: bool,
) -> list[dict]:
    updated = []
    for row in profile_rows:
        item = dict(row)
        item.update(
            {
                "schema": "tcad_generation_profile_1d_v1",
                "mode": mode,
                "color_channel": color_channel,
                "wavelength_nm": wavelength_nm,
                "case": case.name,
                "field_x_norm": case.field_x_norm,
                "field_z_norm": case.field_z_norm,
                "cra_x_deg": case.cra_x_deg,
                "cra_z_deg": case.cra_z_deg,
                "pupil_integrated": integrated,
                "pupil_ray_count": ray_count,
                "pupil_ray_index": ray.ray_index,
                "pupil_u": ray.pupil_u,
                "pupil_v": ray.pupil_v,
                "pupil_weight": ray.weight,
                "ray_cra_x_deg": ray.cra_x_deg,
                "ray_cra_z_deg": ray.cra_z_deg,
            }
        )
        updated.append(item)
    return updated


def aggregate_pupil_profiles(
    base_case: SweepCase,
    ray_profiles: list[tuple[PupilRay, list[dict]]],
) -> list[dict]:
    if not ray_profiles:
        return []
    depth_count = len(ray_profiles[0][1])
    aggregated = []
    for depth_index in range(depth_count):
        base = dict(ray_profiles[0][1][depth_index])
        weights = [ray.weight for ray, _ in ray_profiles]
        for key in (
            "absorption_fraction_per_um",
            "absorption_fraction_per_cm",
            "generation_cm3_s",
        ):
            base[key] = float(
                sum(rows[depth_index][key] * weight for (ray, rows), weight in zip(ray_profiles, weights))
            )
        base.update(
            {
                "case": base_case.name,
                "cra_x_deg": base_case.cra_x_deg,
                "cra_z_deg": base_case.cra_z_deg,
                "pupil_integrated": True,
                "pupil_ray_count": len(ray_profiles),
                "pupil_ray_index": -1,
                "pupil_u": 0.0,
                "pupil_v": 0.0,
                "pupil_weight": 1.0,
                "ray_cra_x_deg": float("nan"),
                "ray_cra_z_deg": float("nan"),
            }
        )
        aggregated.append(base)
    peak_generation = max(max(row["generation_cm3_s"] for row in aggregated), 1e-300)
    for row in aggregated:
        row["generation_normalized"] = row["generation_cm3_s"] / peak_generation
    return aggregated


def map_metadata(
    mode: str,
    color_channel: str,
    wavelength_nm: float,
    case: SweepCase,
    ray: PupilRay,
    ray_count: int,
    integrated: bool,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "color_channel": color_channel,
        "wavelength_nm": wavelength_nm,
        "case": case.name,
        "field_x_norm": case.field_x_norm,
        "field_z_norm": case.field_z_norm,
        "cra_x_deg": case.cra_x_deg,
        "cra_z_deg": case.cra_z_deg,
        "pupil_integrated": integrated,
        "pupil_ray_count": ray_count,
        "pupil_ray_index": ray.ray_index,
        "pupil_u": ray.pupil_u,
        "pupil_v": ray.pupil_v,
        "pupil_weight": ray.weight,
        "ray_cra_x_deg": ray.cra_x_deg,
        "ray_cra_z_deg": ray.cra_z_deg,
    }


def with_map_metadata(tcad_map: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = dict(tcad_map)
    item.update(metadata)
    return item


def aggregate_pupil_maps(
    base_case: SweepCase,
    ray_maps: list[tuple[PupilRay, dict[str, Any]]],
) -> dict[str, Any]:
    if not ray_maps:
        return {}
    base = dict(ray_maps[0][1])
    weights = [ray.weight for ray, _ in ray_maps]
    for key in (
        "generation_cm3_s",
        "absorption_fraction_per_um2",
        "absorption_fraction_per_um3",
    ):
        if key in base:
            base[key] = sum(
                np.asarray(item[key]) * weight
                for (ray, item), weight in zip(ray_maps, weights)
            )
    base.update(
        {
            "case": base_case.name,
            "cra_x_deg": base_case.cra_x_deg,
            "cra_z_deg": base_case.cra_z_deg,
            "pupil_integrated": True,
            "pupil_ray_count": len(ray_maps),
            "pupil_ray_index": -1,
            "pupil_u": 0.0,
            "pupil_v": 0.0,
            "pupil_weight": 1.0,
            "ray_cra_x_deg": float("nan"),
            "ray_cra_z_deg": float("nan"),
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("split-pd-1x1", "ocl-2x2", "ocl-3x3", "ocl-layout"),
        required=True,
    )
    parser.add_argument(
        "--split-mode",
        choices=("dual-x", "dual-z", "quad"),
        default="quad",
    )
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument(
        "--collection-mode",
        choices=("auto", "pixel", "split-pd"),
        default="auto",
        help="Collection region primitive. Use split-pd with ocl-layout to compute QPD/dual split response under advanced OCL geometry.",
    )
    parser.add_argument("--layout-nx", type=int, default=None)
    parser.add_argument("--layout-nz", type=int, default=None)
    parser.add_argument(
        "--ocl-layout",
        default=None,
        help="Mixed/shared OCL descriptor: id:ix:iz:w:h[:shift_x[:shift_z]],...",
    )
    parser.add_argument(
        "--ocl-polygons",
        default=None,
        help="JSON map or @file.json import of lens id to local-um polygon points.",
    )
    parser.add_argument(
        "--ocl-sag",
        default=None,
        help="JSON map or @file.json import of 'default' or lens id to sag profile.",
    )
    parser.add_argument(
        "--ocl-surface-map",
        default=None,
        help="JSON map or @file.json import of 'default' or lens id to measured/freeform height map: x_um, z_um, height_um.",
    )
    parser.add_argument("--ocl-layout-name", default=None)
    parser.add_argument("--target-lens-id", default=None)
    parser.add_argument(
        "--source-aperture-lens-id",
        default=None,
        help=(
            "Restrict source amplitude to one OCL aperture for target-kernel crosstalk proxy. "
            "Use 'target' to reuse --target-lens-id."
        ),
    )
    parser.add_argument("--wavelengths-nm", default="550")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument(
        "--min-feature-pixels",
        type=float,
        default=2.0,
        help="Minimum grid pixels required across critical optical features for quantitative use.",
    )
    parser.add_argument(
        "--min-si-wavelength-pixels",
        type=float,
        default=8.0,
        help="Minimum grid pixels per Si internal wavelength for quantitative use.",
    )
    parser.add_argument("--after-source-time", type=float, default=25.0)
    parser.add_argument(
        "--decay-by",
        type=float,
        default=0.0,
        help="If >0, use Meep stop_when_fields_decayed instead of fixed after-source time.",
    )
    parser.add_argument(
        "--decay-check-time",
        type=float,
        default=50.0,
        help="Time window for stop_when_fields_decayed when --decay-by is enabled.",
    )
    parser.add_argument("--pml-um", type=float, default=None)
    parser.add_argument(
        "--grid-snap-y",
        choices=("off", "nearest", "ceil", "floor"),
        default="off",
        help="Adjust only bottom air padding so the y cell length is an integer number of grid pixels.",
    )
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument(
        "--shield-mode",
        choices=tuple(sorted(VALID_SHIELD_MODES)),
        default=None,
        help="Override stack shield.mode; default uses the stack config, normally off for imaging pixels.",
    )
    parser.add_argument(
        "--shield-mask-edge-width-um",
        type=float,
        default=None,
        help="Override stack shield.mask_edge_width_um used by edge aperture variants.",
    )
    parser.add_argument("--color-channel", choices=COLOR_CHANNELS, default="green")
    parser.add_argument(
        "--cfa-pattern",
        choices=CFA_PATTERNS,
        default="uniform",
        help="CFA layout primitive. uniform preserves the legacy single-color CFA layer.",
    )
    parser.add_argument("--cfa-shift-red-x-um", type=float, default=0.0)
    parser.add_argument("--cfa-shift-green-x-um", type=float, default=0.0)
    parser.add_argument("--cfa-shift-blue-x-um", type=float, default=0.0)
    parser.add_argument("--cfa-shift-red-z-um", type=float, default=0.0)
    parser.add_argument("--cfa-shift-green-z-um", type=float, default=0.0)
    parser.add_argument("--cfa-shift-blue-z-um", type=float, default=0.0)
    parser.add_argument(
        "--cfa-polygons",
        default=None,
        help="JSON or @file.json CFA mask primitive: color/cell local-um polygons plus optional background nearest/passivation/air.",
    )
    parser.add_argument(
        "--f-number",
        type=float,
        default=0.0,
        help="Lens f-number for finite-pupil integration. Use 0 to run chief-ray only.",
    )
    parser.add_argument(
        "--pupil-samples",
        type=int,
        default=1,
        help="Samples per pupil diameter. 1 disables cone integration.",
    )
    parser.add_argument(
        "--incident-photon-flux-cm2-s",
        type=float,
        default=1.0e20,
        help="Incident photon flux used to scale TCAD 1D generation export.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    stack_config = load_stack_config(args.stack_config)
    if args.shield_mode is not None or args.shield_mask_edge_width_um is not None:
        stack_config = dict(stack_config)
        stack_config["shield"] = dict(stack_config.get("shield", {}))
        if args.shield_mode is not None:
            stack_config["shield"]["mode"] = args.shield_mode
        if args.shield_mask_edge_width_um is not None:
            stack_config["shield"]["mask_edge_width_um"] = args.shield_mask_edge_width_um
    shield = shield_config_for_stack(stack_config)
    unsnapped_geom = geometry_from_config(stack_config, pml_um=args.pml_um)
    geom = geometry_from_config(
        stack_config,
        pml_um=args.pml_um,
        grid_snap_y_resolution=args.resolution,
        grid_snap_y_mode=args.grid_snap_y,
    )
    nx, nz = mode_shape(args.mode, args.layout_nx, args.layout_nz)
    wavelengths = parse_wavelengths(args.wavelengths_nm)
    cases = parse_cases(args.cases or default_cases_for_mode(args.mode))
    cfa_shifts = {
        "red": (args.cfa_shift_red_x_um, args.cfa_shift_red_z_um),
        "green": (args.cfa_shift_green_x_um, args.cfa_shift_green_z_um),
        "blue": (args.cfa_shift_blue_x_um, args.cfa_shift_blue_z_um),
    }
    ocl_lenses = apply_ocl_polygons(parse_ocl_layout(args.ocl_layout, nx, nz, geom), args.ocl_polygons, geom)
    ocl_sag_profiles = parse_ocl_sag_profiles(args.ocl_sag)
    ocl_surface_maps = parse_ocl_surface_maps(args.ocl_surface_map, geom)
    cfa_polygons = parse_cfa_polygons(args.cfa_polygons, geom)
    if args.mode == "ocl-layout" and not ocl_lenses:
        raise ValueError("ocl-layout mode requires --ocl-layout")
    regions = collection_regions(
        args.mode,
        geom,
        nx,
        nz,
        args.split_mode,
        args.split_gap_um,
        args.collection_mode,
        ocl_lenses,
        args.target_lens_id,
    )
    ocl_layout_name = args.ocl_layout_name or ("legacy_pixel_lens" if not ocl_lenses else "custom_ocl_layout")
    output_dir = args.output_dir or ROOT / "runs" / f"supercell_lut_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Meep {mp.__version__}")
    print(
        f"mode={args.mode}, cell={nx}x{nz}, wavelengths={wavelengths}, "
        f"cases={[case.name for case in cases]}, resolution={args.resolution}, "
        f"color={args.color_channel}, pml={geom.pml:g} um, "
        f"shield={shield['mode']}, cfa_pattern={args.cfa_pattern}, "
        f"ocl_layout={ocl_layout_name} ({len(ocl_lenses)} lenses)"
    )

    all_rows = []
    all_ray_rows = []
    all_tcad_profile_rows = []
    all_tcad_ray_profile_rows = []
    summaries = []
    grouped_rows = []
    grouped_tcad_profiles = []
    grouped_tcad_maps_2d = []
    grouped_tcad_volumes_3d = []
    focal_maps = []
    for wavelength_nm in wavelengths:
        si_n, si_k, _ = nk_from_material(stack_config, "silicon", wavelength_nm / 1000)
        print(f"wavelength={wavelength_nm:g} nm, Si n={si_n:.4f}, k={si_k:.5f}")
        for case in cases:
            print(
                f"running {case.name}: CRA=({case.cra_x_deg:g},{case.cra_z_deg:g}) deg, "
                f"lens_shift=({case.lens_shift_x_um:g},{case.lens_shift_z_um:g}) um"
            )
            pupil_rays = build_pupil_rays(
                case,
                args.f_number if args.f_number > 0 else None,
                args.pupil_samples,
            )
            ray_rows = []
            ray_focals = []
            ray_profiles = []
            ray_maps_2d = []
            ray_volumes_3d = []
            for ray in pupil_rays:
                ray_case = replace(
                    case,
                    cra_x_deg=ray.cra_x_deg,
                    cra_z_deg=ray.cra_z_deg,
                )
                if len(pupil_rays) > 1:
                    print(
                        f"  pupil ray {ray.ray_index + 1}/{len(pupil_rays)}: "
                        f"u/v=({ray.pupil_u:g},{ray.pupil_v:g}), "
                        f"CRA=({ray.cra_x_deg:.3f},{ray.cra_z_deg:.3f}) deg"
                    )
                rows, _, focal_map, tcad_profile, tcad_map_2d, tcad_volume_3d = run_one(
                    geom,
                    args.mode,
                    nx,
                    nz,
                    ray_case,
                    wavelength_nm,
                    args.resolution,
                    args.after_source_time,
                    args.decay_by,
                    args.decay_check_time,
                    regions,
                    stack_config,
                    args.color_channel,
                    args.cfa_pattern,
                    cfa_shifts,
                    cfa_polygons,
                    ocl_lenses,
                    ocl_sag_profiles,
                    ocl_surface_maps,
                    ocl_layout_name,
                    args.collection_mode,
                    args.target_lens_id,
                    args.split_mode,
                    args.incident_photon_flux_cm2_s,
                    args.source_aperture_lens_id,
                )
                rows = add_ray_metadata(
                    rows,
                    ray,
                    len(pupil_rays),
                    integrated=len(pupil_rays) > 1,
                )
                tcad_profile = add_profile_metadata(
                    tcad_profile,
                    args.mode,
                    args.color_channel,
                    wavelength_nm,
                    case,
                    ray,
                    len(pupil_rays),
                    integrated=len(pupil_rays) > 1,
                )
                metadata = map_metadata(
                    args.mode,
                    args.color_channel,
                    wavelength_nm,
                    case,
                    ray,
                    len(pupil_rays),
                    integrated=len(pupil_rays) > 1,
                )
                tcad_map_2d = with_map_metadata(tcad_map_2d, metadata)
                tcad_volume_3d = with_map_metadata(tcad_volume_3d, metadata)
                ray_rows.append((ray, rows))
                ray_focals.append((ray, focal_map))
                ray_profiles.append((ray, tcad_profile))
                ray_maps_2d.append((ray, tcad_map_2d))
                ray_volumes_3d.append((ray, tcad_volume_3d))
                all_ray_rows.extend(rows)
                all_tcad_ray_profile_rows.extend(tcad_profile)

            if len(pupil_rays) > 1:
                rows, summary = aggregate_pupil_rows(
                    args.mode, wavelength_nm, case, ray_rows
                )
                focal_map = weighted_focal_map(ray_focals)
                tcad_profile = aggregate_pupil_profiles(case, ray_profiles)
                tcad_map_2d = aggregate_pupil_maps(case, ray_maps_2d)
                tcad_volume_3d = aggregate_pupil_maps(case, ray_volumes_3d)
            else:
                rows = ray_rows[0][1]
                summary = summary_from_rows(
                    args.mode,
                    wavelength_nm,
                    case,
                    rows,
                    pupil_integrated=False,
                    pupil_ray_count=1,
                )
                focal_map = ray_focals[0][1]
                tcad_profile = ray_profiles[0][1]
                tcad_map_2d = ray_maps_2d[0][1]
                tcad_volume_3d = ray_volumes_3d[0][1]

            summary.update(
                grid_resolution_metadata(
                    geom,
                    stack_config,
                    wavelength_nm,
                    args.resolution,
                    args.min_feature_pixels,
                    args.min_si_wavelength_pixels,
                )
            )
            for row in rows:
                row["split_mode"] = args.split_mode if args.collection_mode == "split-pd" or args.mode == "split-pd-1x1" else ""
            all_rows.extend(rows)
            all_tcad_profile_rows.extend(tcad_profile)
            summaries.append(summary)
            grouped_rows.append(rows)
            grouped_tcad_profiles.append(tcad_profile)
            grouped_tcad_maps_2d.append(tcad_map_2d)
            grouped_tcad_volumes_3d.append(tcad_volume_3d)
            if focal_map is not None:
                focal_maps.append(focal_map)
            apply_response_normalization(summaries, all_rows, grouped_rows)
            write_partial_outputs(
                output_dir,
                all_rows,
                all_ray_rows,
                all_tcad_profile_rows,
                all_tcad_ray_profile_rows,
                summaries,
            )

    apply_response_normalization(summaries, all_rows, grouped_rows)

    long_csv = output_dir / "camera_lut_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(all_rows, ROW_FIELD_ORDER))
        writer.writeheader()
        writer.writerows(all_rows)

    ray_csv = output_dir / "camera_lut_pupil_rays.csv"
    if all_ray_rows:
        for row in all_ray_rows:
            row["split_mode"] = args.split_mode if args.collection_mode == "split-pd" or args.mode == "split-pd-1x1" else ""
        with ray_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames_for(all_ray_rows, ROW_FIELD_ORDER))
            writer.writeheader()
            writer.writerows(all_ray_rows)

    tcad_profile_csv = output_dir / "tcad_generation_profile_1d.csv"
    if all_tcad_profile_rows:
        with tcad_profile_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames_for(all_tcad_profile_rows, TCAD_PROFILE_FIELD_ORDER),
            )
            writer.writeheader()
            writer.writerows(all_tcad_profile_rows)

    tcad_ray_profile_csv = output_dir / "tcad_generation_profile_1d_pupil_rays.csv"
    if all_tcad_ray_profile_rows:
        with tcad_ray_profile_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames_for(all_tcad_ray_profile_rows, TCAD_PROFILE_FIELD_ORDER),
            )
            writer.writeheader()
            writer.writerows(all_tcad_ray_profile_rows)

    summary_csv = output_dir / "camera_lut_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(summaries, SUMMARY_FIELD_ORDER))
        writer.writeheader()
        writer.writerows(summaries)

    response_tensor = np.asarray(
        [[row["response"] for row in rows] for rows in grouped_rows],
        dtype=float,
    )
    np.savez(
        output_dir / "camera_lut.npz",
        response=response_tensor,
        case=np.asarray([summary["case"] for summary in summaries]),
        wavelength_nm=np.asarray([summary["wavelength_nm"] for summary in summaries], dtype=float),
        field_x_norm=np.asarray([summary["field_x_norm"] for summary in summaries], dtype=float),
        field_z_norm=np.asarray([summary["field_z_norm"] for summary in summaries], dtype=float),
        cra_x_deg=np.asarray([summary["cra_x_deg"] for summary in summaries], dtype=float),
        cra_z_deg=np.asarray([summary["cra_z_deg"] for summary in summaries], dtype=float),
        region_id=np.asarray([region.region_id for region in regions]),
        color_channel=np.asarray([args.color_channel]),
        shield_mode=np.asarray([shield["mode"]]),
        shield_mask_edge_width_um=np.asarray([shield["mask_edge_width_um"]], dtype=float),
    )

    if grouped_tcad_profiles:
        generation_tensor = np.asarray(
            [[row["generation_cm3_s"] for row in rows] for rows in grouped_tcad_profiles],
            dtype=float,
        )
        absorption_per_um_tensor = np.asarray(
            [[row["absorption_fraction_per_um"] for row in rows] for rows in grouped_tcad_profiles],
            dtype=float,
        )
        depth_um = np.asarray(
            [row["depth_um_from_si_top"] for row in grouped_tcad_profiles[0]],
            dtype=float,
        )
        np.savez(
            output_dir / "tcad_generation_profile_1d.npz",
            generation_cm3_s=generation_tensor,
            absorption_fraction_per_um=absorption_per_um_tensor,
            depth_um_from_si_top=depth_um,
            case=np.asarray([summary["case"] for summary in summaries]),
            wavelength_nm=np.asarray([summary["wavelength_nm"] for summary in summaries], dtype=float),
            field_x_norm=np.asarray([summary["field_x_norm"] for summary in summaries], dtype=float),
            field_z_norm=np.asarray([summary["field_z_norm"] for summary in summaries], dtype=float),
            cra_x_deg=np.asarray([summary["cra_x_deg"] for summary in summaries], dtype=float),
            cra_z_deg=np.asarray([summary["cra_z_deg"] for summary in summaries], dtype=float),
            incident_photon_flux_cm2_s=np.asarray([args.incident_photon_flux_cm2_s], dtype=float),
            shield_mode=np.asarray([shield["mode"]]),
            shield_mask_edge_width_um=np.asarray([shield["mask_edge_width_um"]], dtype=float),
        )

    if grouped_tcad_maps_2d:
        np.savez_compressed(
            output_dir / "tcad_generation_map_2d.npz",
            schema=np.asarray(["tcad_generation_map_2d_x_depth_v1"]),
            generation_cm3_s=np.asarray(
                [item["generation_cm3_s"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            absorption_fraction_per_um2=np.asarray(
                [item["absorption_fraction_per_um2"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            x_um=np.asarray(grouped_tcad_maps_2d[0]["x_um"], dtype=float),
            depth_um_from_si_top=np.asarray(
                grouped_tcad_maps_2d[0]["depth_um_from_si_top"],
                dtype=float,
            ),
            y_um=np.asarray(grouped_tcad_maps_2d[0]["y_um"], dtype=float),
            case=np.asarray([item["case"] for item in grouped_tcad_maps_2d]),
            wavelength_nm=np.asarray(
                [item["wavelength_nm"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            field_x_norm=np.asarray(
                [item["field_x_norm"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            field_z_norm=np.asarray(
                [item["field_z_norm"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            cra_x_deg=np.asarray(
                [item["cra_x_deg"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            cra_z_deg=np.asarray(
                [item["cra_z_deg"] for item in grouped_tcad_maps_2d],
                dtype=float,
            ),
            color_channel=np.asarray([args.color_channel]),
            shield_mode=np.asarray([shield["mode"]]),
            shield_mask_edge_width_um=np.asarray(
                [shield["mask_edge_width_um"]],
                dtype=float,
            ),
            incident_photon_flux_cm2_s=np.asarray(
                [args.incident_photon_flux_cm2_s],
                dtype=float,
            ),
            out_of_plane_axis=np.asarray(["z"]),
            method=np.asarray(
                [
                    "z-averaged calibrated Si volume absorption density for 2D x-depth DEVSIM import"
                ]
            ),
        )

    if grouped_tcad_volumes_3d:
        np.savez_compressed(
            output_dir / "tcad_generation_volume_3d.npz",
            schema=np.asarray(["tcad_generation_volume_3d_v1"]),
            generation_cm3_s=np.asarray(
                [item["generation_cm3_s"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            absorption_fraction_per_um3=np.asarray(
                [item["absorption_fraction_per_um3"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            x_um=np.asarray(grouped_tcad_volumes_3d[0]["x_um"], dtype=float),
            depth_um_from_si_top=np.asarray(
                grouped_tcad_volumes_3d[0]["depth_um_from_si_top"],
                dtype=float,
            ),
            y_um=np.asarray(grouped_tcad_volumes_3d[0]["y_um"], dtype=float),
            z_um=np.asarray(grouped_tcad_volumes_3d[0]["z_um"], dtype=float),
            case=np.asarray([item["case"] for item in grouped_tcad_volumes_3d]),
            wavelength_nm=np.asarray(
                [item["wavelength_nm"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            field_x_norm=np.asarray(
                [item["field_x_norm"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            field_z_norm=np.asarray(
                [item["field_z_norm"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            cra_x_deg=np.asarray(
                [item["cra_x_deg"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            cra_z_deg=np.asarray(
                [item["cra_z_deg"] for item in grouped_tcad_volumes_3d],
                dtype=float,
            ),
            color_channel=np.asarray([args.color_channel]),
            shield_mode=np.asarray([shield["mode"]]),
            shield_mask_edge_width_um=np.asarray(
                [shield["mask_edge_width_um"]],
                dtype=float,
            ),
            incident_photon_flux_cm2_s=np.asarray(
                [args.incident_photon_flux_cm2_s],
                dtype=float,
            ),
            method=np.asarray(
                [
                    "calibrated Si volume absorption density for future 3D TCAD import"
                ]
            ),
        )

    metadata = {
        "schema": "camera_supercell_optical_lut_v2",
        "response_model": "si_volume_absorption_flux_calibrated_v1",
        "mode": args.mode,
        "collection_mode": args.collection_mode,
        "split_mode": args.split_mode if args.collection_mode == "split-pd" or args.mode == "split-pd-1x1" else None,
        "target_lens_id": args.target_lens_id,
        "source_aperture_lens_id": args.source_aperture_lens_id,
        "source_aperture_enabled": bool(args.source_aperture_lens_id),
        "cell_pixels": {"x": nx, "z": nz},
        "shield": shield,
        "ocl_layout": {
            "name": ocl_layout_name,
            "descriptor": args.ocl_layout,
            "polygons_descriptor": args.ocl_polygons,
            "polygons_source": json_argument_source(args.ocl_polygons, "--ocl-polygons"),
            "sag_descriptor": args.ocl_sag,
            "sag_source": json_argument_source(args.ocl_sag, "--ocl-sag"),
            "sag_profiles": {key: asdict(value) for key, value in ocl_sag_profiles.items()},
            "surface_map_descriptor": args.ocl_surface_map,
            "surface_map_source": json_argument_source(args.ocl_surface_map, "--ocl-surface-map"),
            "surface_maps": {
                key: {
                    "source": item.source,
                    "x_count": len(item.x_um),
                    "z_count": len(item.z_um),
                    "x_range_um": [item.x_um[0], item.x_um[-1]],
                    "z_range_um": [item.z_um[0], item.z_um[-1]],
                    "height_max_um": max(value for row in item.height_um for value in row),
                }
                for key, item in ocl_surface_maps.items()
            },
            "lenses": [asdict(item) for item in ocl_lenses],
            "legacy_pixel_lens": not bool(ocl_lenses),
        },
        "cfa": {
            "pattern": args.cfa_pattern,
            "color_channel": args.color_channel,
            "shifts_um": {
                "red": {"x": cfa_shifts["red"][0], "z": cfa_shifts["red"][1]},
                "green": {"x": cfa_shifts["green"][0], "z": cfa_shifts["green"][1]},
                "blue": {"x": cfa_shifts["blue"][0], "z": cfa_shifts["blue"][1]},
            },
            "polygons_descriptor": args.cfa_polygons,
            "polygons_source": cfa_polygons.source,
            "polygon_background": cfa_polygons.background,
            "polygons": [asdict(item) for item in cfa_polygons.polygons],
        },
        "sensor_stack": stack_metadata(stack_config, geom, args.color_channel),
        "geometry": asdict(geom),
        "geometry_unsnapped": asdict(unsnapped_geom),
        "grid_snap_y": {
            "mode": args.grid_snap_y,
            "applied": args.grid_snap_y != "off",
            "resolution_px_per_um": args.resolution,
            "original_cell_y_um": unsnapped_geom.cell_y,
            "snapped_cell_y_um": geom.cell_y,
            "bottom_air_delta_um": geom.bottom_air - unsnapped_geom.bottom_air,
            "active_stack_layer_thicknesses_preserved": True,
        },
        "grid_rounding": grid_rounding_metadata(geom, nx, nz, args.resolution),
        "resolution_px_per_um": args.resolution,
        "after_source_time": args.after_source_time,
        "termination": {
            "mode": "field_decay" if args.decay_by > 0 else "fixed_after_source_time",
            "decay_by": args.decay_by if args.decay_by > 0 else None,
            "decay_check_time": args.decay_check_time if args.decay_by > 0 else None,
            "decay_field_component": "Ez",
            "decay_probe_y_um": geom.si_bottom_monitor_y if args.decay_by > 0 else None,
        },
        "pml_um": geom.pml,
        "pupil": {
            "f_number": args.f_number if args.f_number > 0 else None,
            "samples_per_diameter": args.pupil_samples,
            "integration": "uniform_disk_grid" if args.pupil_samples > 1 and args.f_number > 0 else "chief_ray_only",
        },
        "tcad_generation_export": {
            "schema": "tcad_generation_profile_1d_v1",
            "incident_photon_flux_cm2_s": args.incident_photon_flux_cm2_s,
            "depth_axis": "depth_um_from_si_top",
            "generation_units": "cm^-3 s^-1",
            "method": "collapse calibrated Si volume absorption density over x/z",
            "map_2d_schema": "tcad_generation_map_2d_x_depth_v1",
            "map_2d_method": "collapse calibrated Si volume absorption density over z, preserving x/depth",
            "volume_3d_schema": "tcad_generation_volume_3d_v1",
        },
        "wavelengths_nm": wavelengths,
        "cases": [asdict(case) for case in cases],
        "regions": [asdict(region) for region in regions],
        "summaries": summaries,
        "long_csv": str(long_csv),
        "pupil_ray_csv": str(ray_csv) if all_ray_rows else None,
        "tcad_generation_profile_1d_csv": str(tcad_profile_csv) if all_tcad_profile_rows else None,
        "tcad_generation_profile_1d_pupil_ray_csv": str(tcad_ray_profile_csv) if all_tcad_ray_profile_rows else None,
        "tcad_generation_profile_1d_npz": str(output_dir / "tcad_generation_profile_1d.npz") if grouped_tcad_profiles else None,
        "tcad_generation_map_2d_npz": str(output_dir / "tcad_generation_map_2d.npz") if grouped_tcad_maps_2d else None,
        "tcad_generation_volume_3d_npz": str(output_dir / "tcad_generation_volume_3d.npz") if grouped_tcad_volumes_3d else None,
        "summary_csv": str(summary_csv),
        "npz": str(output_dir / "camera_lut.npz"),
        "notes": [
            "Primary response uses Si-volume absorption distribution integrated over each collection region.",
            "The absolute scale is calibrated to the full Si top/bottom flux absorption estimate.",
            "TCAD 1D generation export collapses the Si volume absorption over x/z and scales it with incident_photon_flux_cm2_s.",
            "TCAD 2D generation export preserves x/depth variation and collapses only the out-of-plane z direction.",
            "TCAD 3D generation export preserves x/depth/z variation for future 3D electrical import.",
            "Split-PD phase proxies are optical-only PDAF signals, not charge-collection TCAD results.",
            "OCL shifts are explicit lens/aperture offsets in microns; replace defaults with product geometry.",
            "Default imaging-pixel runs use shield.mode=off; metal optical masks are only present for edge or PDAF shield modes.",
            "Finite-pupil integration samples a uniform disk in direction-cosine space using the supplied f-number.",
            "Use resolution and time convergence before quantitative camera-system use.",
        ],
    }
    json_path = output_dir / "camera_lut.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_response_plot(output_dir, args.mode, nx, nz, regions, grouped_rows, summaries)
    save_focal_plot(output_dir, nx, nz, geom, focal_maps, summaries)

    print(json.dumps(summaries, indent=2))
    print(f"wrote {long_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
