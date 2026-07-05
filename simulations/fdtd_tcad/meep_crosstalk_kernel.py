#!/usr/bin/env python3
"""Full-array FDTD crosstalk kernel solver for image-sensor OCL supercells.

This is intentionally separate from the periodic supercell LUT runner. Crosstalk
needs a finite OCL neighborhood: illuminate the center OCL/output cell and
integrate silicon absorption in surrounding output cells and physical PD pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from meep_microlens_array_3d import MicrolensArrayGeometry
from sensor_stack_config import (
    DEFAULT_STACK_CONFIG,
    geometry_from_config,
    load_stack_config,
    material_role_for_color,
    medium_for_role,
    nk_from_material,
    shield_config_for_stack,
)
from meep_supercell_lut import (
    CFA_COLORS,
    COLOR_CHANNELS,
    OclLens,
    OclSagProfile,
    OclSurfaceMap,
    CfaPolygonSet,
    apply_ocl_polygons,
    cfa_color_for_cell,
    cfa_polygon_color_at,
    cfa_shift_for_color,
    inside_lens_volume,
    json_argument_source,
    parse_cfa_polygons,
    parse_ocl_layout,
    parse_ocl_sag_profiles,
    parse_ocl_surface_maps,
    point_in_polygon,
    sag_profile_for_lens,
    surface_map_for_lens,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TCAD_PROFILE = ROOT / "measured_profiles" / "reference_cmos_ppd_1p4um" / "profile.json"
INTEGER_CSV_FIELDS = {
    "layout_size",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "output_cell_count",
    "raw_pd_count",
    "resolution_px_per_um",
    "output_dx",
    "output_dz",
    "raw_pd_ix",
    "raw_pd_iz",
    "layout_nx",
    "layout_nz",
    "ocl_lens_count",
    "cfa_polygon_count",
    "lens_ix",
    "lens_iz",
    "lens_w",
    "lens_h",
}
FLOAT_CSV_FIELDS = {
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "region_x_um",
    "region_z_um",
    "response_fraction",
    "center_response_fraction",
    "output_crosstalk_fraction",
    "border_response_fraction",
    "outside_output_kernel_fraction",
    "truncation_response_fraction",
    "support_edge_response_fraction",
    "strongest_neighbor_fraction",
    "total_integrated_response_fraction",
    "total_absorption_raw",
    "source_sigma_um",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "output_dx_um",
    "output_dz_um",
    "optical_dti_width_um",
    "optical_dti_depth_um",
    "grid_dx_um",
    "si_n_at_wavelength",
    "si_internal_wavelength_um",
    "si_internal_wavelength_pixels",
    "dti_width_pixels",
    "passivation_thickness_pixels",
    "lens_height_pixels",
    "lens_edge_gap_pixels",
    "cfa_thickness_pixels",
    "minimum_critical_feature_pixels",
    "min_feature_pixels_required",
    "min_si_wavelength_pixels_required",
}
BOOL_CSV_FIELDS = {
    "measured_accuracy_blocked",
    "optical_dti_enabled",
    "optical_dti_measured",
    "si_wavelength_gate_pass",
    "critical_feature_gate_pass",
    "grid_resolution_gate_pass",
}
MERGE_KEY_FIELDS = (
    "schema",
    "mode",
    "layout_label",
    "kernel_scope",
    "case",
    "wavelength_nm",
    "resolution_px_per_um",
    "color_channel",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "output_dx",
    "output_dz",
    "raw_pd_ix",
    "raw_pd_iz",
)


@dataclass(frozen=True)
class CrosstalkCase:
    name: str
    cra_x_deg: float
    cra_z_deg: float
    field_x_norm: float = 0.0
    field_z_norm: float = 0.0
    lens_shift_x_um: float = 0.0
    lens_shift_z_um: float = 0.0


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
class OpticalDtiConfig:
    enabled: bool
    width_um: float
    depth_um: float
    material_role: str
    source: str
    measured: bool


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or value == "":
            result[key] = value
        elif key in INTEGER_CSV_FIELDS:
            result[key] = int(float(value))
        elif key in FLOAT_CSV_FIELDS:
            result[key] = float(value)
        elif key in BOOL_CSV_FIELDS:
            result[key] = str(value).strip().lower() in {"1", "true", "yes", "y"}
        else:
            result[key] = value
    return result


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [coerce_csv_row(row) for row in csv.DictReader(handle)]


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in [*existing, *incoming]:
        key = tuple(str(row.get(field, "")) for field in MERGE_KEY_FIELDS)
        merged[key] = row
    return list(merged.values())


def unique_values(rows: list[dict[str, Any]], field: str, caster=str) -> list[Any]:
    values = {caster(row[field]) for row in rows if field in row and row[field] != ""}
    return sorted(values)


def cases_from_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("case", ""))
        if not name or name in cases:
            continue
        cases[name] = {
            "name": name,
            "cra_x_deg": float(row.get("cra_x_deg", 0.0) or 0.0),
            "cra_z_deg": float(row.get("cra_z_deg", 0.0) or 0.0),
        }
    return list(cases.values())


def grid_resolution_metadata(
    geom: MicrolensArrayGeometry,
    dti: OpticalDtiConfig,
    stack_config: dict[str, Any],
    wavelength_nm: float,
    resolution: int,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
) -> dict[str, Any]:
    wavelength_um = wavelength_nm / 1000.0
    si_n, _, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    si_internal_wavelength = wavelength_um / si_n
    feature_pixels = {
        "dti_width_pixels": dti.width_um * resolution if dti.enabled else float("inf"),
        "passivation_thickness_pixels": geom.passivation_thickness * resolution,
        "lens_height_pixels": geom.lens_height * resolution,
        "lens_edge_gap_pixels": geom.lens_edge_gap * resolution,
        "cfa_thickness_pixels": geom.cfa_thickness * resolution,
    }
    critical_features = [
        feature_pixels["passivation_thickness_pixels"],
        feature_pixels["lens_edge_gap_pixels"],
    ]
    if dti.enabled:
        critical_features.append(feature_pixels["dti_width_pixels"])
    minimum_critical_feature_pixels = min(critical_features)
    si_internal_wavelength_pixels = si_internal_wavelength * resolution
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
        "si_internal_wavelength_um": si_internal_wavelength,
        "si_internal_wavelength_pixels": si_internal_wavelength_pixels,
        **feature_pixels,
        "minimum_critical_feature_pixels": minimum_critical_feature_pixels,
        "min_feature_pixels_required": min_feature_pixels,
        "min_si_wavelength_pixels_required": min_si_wavelength_pixels,
        "si_wavelength_gate_pass": si_wavelength_gate_pass,
        "critical_feature_gate_pass": critical_feature_gate_pass,
        "grid_resolution_gate_pass": si_wavelength_gate_pass and critical_feature_gate_pass,
        "grid_resolution_notes": " ".join(notes) if notes else "Grid resolution meets configured feature and Si wavelength gates.",
    }


def enrich_summaries_with_grid_metadata(
    summaries: list[dict[str, Any]],
    geom: MicrolensArrayGeometry,
    dti: OpticalDtiConfig,
    stack_config: dict[str, Any],
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
) -> list[dict[str, Any]]:
    enriched = []
    for row in summaries:
        updated = dict(row)
        metadata = grid_resolution_metadata(
            geom,
            dti,
            stack_config,
            float(updated["wavelength_nm"]),
            int(updated["resolution_px_per_um"]),
            min_feature_pixels,
            min_si_wavelength_pixels,
        )
        updated.update(metadata)
        enriched.append(updated)
    return enriched


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_cases(raw: str) -> list[CrosstalkCase]:
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 3:
            cases.append(CrosstalkCase(parts[0], float(parts[1]), 0.0, float(parts[2]), 0.0))
            continue
        if len(parts) < 5:
            raise ValueError("Case must be name:cra_x:cra_z:field_x:field_z[:lens_shift_x[:lens_shift_z]]")
        cases.append(
            CrosstalkCase(
                name=parts[0],
                cra_x_deg=float(parts[1]),
                cra_z_deg=float(parts[2]),
                field_x_norm=float(parts[3]),
                field_z_norm=float(parts[4]),
                lens_shift_x_um=float(parts[5]) if len(parts) > 5 and parts[5] else 0.0,
                lens_shift_z_um=float(parts[6]) if len(parts) > 6 and parts[6] else 0.0,
            )
        )
    if not cases:
        raise ValueError("No cases parsed")
    return cases


def mode_layout_size(mode: str) -> int:
    if mode == "split-pd-1x1":
        return 1
    if mode == "ocl-2x2":
        return 2
    if mode == "ocl-3x3":
        return 3
    raise ValueError(f"Unsupported mode: {mode}")


def layout_label(mode: str) -> str:
    return {
        "split-pd-1x1": "1x1 Bayer",
        "ocl-2x2": "2x2 Quad",
        "ocl-3x3": "3x3 Nona",
    }[mode]


def centered_positions(count: int, pitch: float) -> list[float]:
    half = 0.5 * (count - 1)
    return [(index - half) * pitch for index in range(count)]


def transverse_k(frequency: float, cra_x_deg: float, cra_z_deg: float) -> tuple[float, float]:
    sx = math.sin(math.radians(cra_x_deg))
    sz = math.sin(math.radians(cra_z_deg))
    if sx * sx + sz * sz >= 1.0:
        raise ValueError("CRA x/z components exceed a propagating ray")
    return frequency * sx, frequency * sz


def source_amplitude(
    kx: float,
    kz: float,
    profile: str,
    sigma_um: float,
    center_x_um: float = 0.0,
    center_z_um: float = 0.0,
):
    def amp(point: mp.Vector3):
        phase = np.exp(1j * 2 * np.pi * (kx * point.x + kz * point.z))
        if profile == "gaussian":
            dx = point.x - center_x_um
            dz = point.z - center_z_um
            envelope = math.exp(-0.5 * (dx * dx + dz * dz) / (sigma_um * sigma_um))
            return envelope * phase
        return phase

    return amp


def lens_radius_for_layout(geom: MicrolensArrayGeometry, layout_size: int) -> float:
    return max(0.5 * layout_size * geom.pitch - geom.lens_edge_gap, 0.05)


def lens_sphere_radius(aperture_radius: float, lens_height: float) -> float:
    return (aperture_radius * aperture_radius + lens_height * lens_height) / (2 * lens_height)


def optical_dti_from_profile(path: Path | None, geom: MicrolensArrayGeometry) -> OpticalDtiConfig:
    if not path:
        return OpticalDtiConfig(False, 0.0, 0.0, "passivation", "", False)
    profile = load_json(path)
    profile_geometry = profile.get("geometry", {})
    width_um = float(profile_geometry.get("dti_width_um", 0.0) or 0.0)
    depth_candidates = []
    if profile_geometry.get("dti_depth_um") is not None:
        depth_candidates.append(float(profile_geometry["dti_depth_um"]))
    for item in profile.get("implants", []):
        if "dti" in str(item.get("name", "")).lower() and item.get("depth_max_um") is not None:
            depth_candidates.append(float(item["depth_max_um"]))
    if not depth_candidates and profile_geometry.get("depth_um") is not None:
        depth_candidates.append(float(profile_geometry["depth_um"]))
    depth_um = min(max(depth_candidates or [0.0]), geom.si_thickness)
    enabled = width_um > 0.0 and depth_um > 0.0
    measured = bool(profile.get("calibration_status", {}).get("is_measured", False)) or bool(profile.get("measured", False))
    return OpticalDtiConfig(
        enabled=enabled,
        width_um=width_um,
        depth_um=depth_um,
        material_role="passivation",
        source=str(path),
        measured=measured,
    )


def distance_to_pitch_boundary(value_um: float, pitch_um: float) -> float:
    local = (((value_um / pitch_um) + 0.5) % 1.0 - 0.5) * pitch_um
    return 0.5 * pitch_um - abs(local)


def in_dti_trench(point: mp.Vector3, geom: MicrolensArrayGeometry, dti: OpticalDtiConfig, half_array: float) -> bool:
    if not dti.enabled:
        return False
    if abs(point.x) > half_array or abs(point.z) > half_array:
        return False
    depth_from_si_top = geom.si_top - point.y
    if depth_from_si_top < 0.0 or depth_from_si_top > dti.depth_um:
        return False
    half_width = 0.5 * dti.width_um
    x_boundary_distance = distance_to_pitch_boundary(point.x, geom.pitch)
    z_boundary_distance = distance_to_pitch_boundary(point.z, geom.pitch)
    return x_boundary_distance <= half_width or z_boundary_distance <= half_width


def bayer_color_for_output(offset_x: int, offset_z: int, target: str) -> str:
    target = target.lower()
    if target == "clear":
        return "clear"
    if target == "red":
        colors = {(0, 0): "red", (1, 0): "green", (0, 1): "green", (1, 1): "blue"}
    elif target == "blue":
        colors = {(0, 0): "blue", (1, 0): "green", (0, 1): "green", (1, 1): "red"}
    else:
        colors = {(0, 0): "green", (1, 0): "red", (0, 1): "blue", (1, 1): "green"}
    return colors[(offset_x & 1, offset_z & 1)]


def nearest_index(value: float, centers: list[float], pitch: float) -> int | None:
    if not centers:
        return None
    distances = [abs(value - center) for center in centers]
    index = int(np.argmin(distances))
    if distances[index] <= 0.5 * pitch + 1e-9:
        return index
    return None


def build_regions(geom: MicrolensArrayGeometry, layout_size: int, neighborhood: int) -> tuple[list[Region], list[Region]]:
    supercell_pitch = layout_size * geom.pitch
    output_regions = []
    raw_pd_regions = []
    output_centers = centered_positions(neighborhood, supercell_pitch)
    pd_count = neighborhood * layout_size
    pd_centers = centered_positions(pd_count, geom.pitch)

    half = neighborhood // 2
    for iz, zc in enumerate(output_centers):
        for ix, xc in enumerate(output_centers):
            dx = ix - half
            dz = iz - half
            output_regions.append(
                Region(
                    region_id=f"out_dx{dx:+d}_dz{dz:+d}",
                    kind="output_cell",
                    ix=dx,
                    iz=dz,
                    x_um=xc,
                    z_um=zc,
                    sx_um=supercell_pitch,
                    sz_um=supercell_pitch,
                )
            )

    pd_half = pd_count // 2
    for iz, zc in enumerate(pd_centers):
        for ix, xc in enumerate(pd_centers):
            raw_pd_regions.append(
                Region(
                    region_id=f"pd_ix{ix - pd_half:+d}_iz{iz - pd_half:+d}",
                    kind="raw_pd",
                    ix=ix - pd_half,
                    iz=iz - pd_half,
                    x_um=xc,
                    z_um=zc,
                    sx_um=geom.pitch,
                    sz_um=geom.pitch,
                )
            )
    return output_regions, raw_pd_regions


def make_material_function(
    geom: MicrolensArrayGeometry,
    layout_size: int,
    simulation_neighborhood: int,
    case: CrosstalkCase,
    color_channel: str,
    silicon: mp.Medium,
    cfa_media: dict[str, mp.Medium],
    passivation: mp.Medium,
    lens: mp.Medium,
    dti_medium: mp.Medium,
    dti: OpticalDtiConfig,
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    cfa_polygons: CfaPolygonSet,
):
    supercell_pitch = layout_size * geom.pitch
    output_centers = centered_positions(simulation_neighborhood, supercell_pitch)
    lens_radius = lens_radius_for_layout(geom, layout_size)
    half_array = 0.5 * simulation_neighborhood * supercell_pitch
    sag_profile = sag_profile_for_lens(None, ocl_sag_profiles)
    surface_map = surface_map_for_lens(None, ocl_surface_maps)

    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        z = point.z
        if abs(x) > half_array or abs(z) > half_array:
            return mp.air
        if geom.si_bottom <= y < geom.si_top:
            if in_dti_trench(point, geom, dti, half_array):
                return dti_medium
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            ix = nearest_index(x, output_centers, supercell_pitch)
            iz = nearest_index(z, output_centers, supercell_pitch)
            if ix is None or iz is None:
                return mp.air
            offset_x = ix - simulation_neighborhood // 2
            offset_z = iz - simulation_neighborhood // 2
            color = cfa_polygon_color_at(
                x,
                z,
                output_centers,
                output_centers,
                simulation_neighborhood * supercell_pitch,
                simulation_neighborhood * supercell_pitch,
                lambda cell_ix, cell_iz: bayer_color_for_output(
                    cell_ix - simulation_neighborhood // 2,
                    cell_iz - simulation_neighborhood // 2,
                    color_channel,
                ),
                {c: (0.0, 0.0) for c in ("red", "green", "blue")},
                cfa_polygons,
                periodic=False,
            )
            if color:
                return cfa_media[color]
            if cfa_polygons.polygons:
                if cfa_polygons.background == "passivation":
                    return passivation
                if cfa_polygons.background == "air":
                    return mp.air
            return cfa_media[bayer_color_for_output(offset_x, offset_z, color_channel)]
        if geom.lens_bottom <= y <= geom.lens_top:
            for xc in output_centers:
                dx = x - (xc + case.lens_shift_x_um)
                if abs(dx) > lens_radius:
                    continue
                for zc in output_centers:
                    dz = z - (zc + case.lens_shift_z_um)
                    if inside_lens_volume(dx, dz, y, lens_radius, geom, sag_profile, surface_map):
                        return lens
        return mp.air

    return material


def inside_spherical_cap(
    dx: float,
    dz: float,
    y: float,
    aperture_radius: float,
    geom: MicrolensArrayGeometry,
) -> bool:
    r2 = dx * dx + dz * dz
    if r2 > aperture_radius * aperture_radius or geom.lens_height <= 0:
        return False
    sphere_radius = lens_sphere_radius(aperture_radius, geom.lens_height)
    sphere_center_y = geom.lens_top - sphere_radius
    return r2 + (y - sphere_center_y) ** 2 <= sphere_radius * sphere_radius


def choose_target_lens(lenses: list[OclLens], target_lens_id: str | None) -> OclLens:
    if not lenses:
        raise ValueError("ocl-layout requires at least one OCL lens")
    if target_lens_id:
        for lens in lenses:
            if lens.lens_id == target_lens_id:
                return lens
        raise ValueError(f"target OCL lens {target_lens_id!r} was not found in --ocl-layout")
    return min(lenses, key=lambda lens: lens.x_um * lens.x_um + lens.z_um * lens.z_um)


def make_layout_material_function(
    geom: MicrolensArrayGeometry,
    layout_nx: int,
    layout_nz: int,
    lenses: list[OclLens],
    case: CrosstalkCase,
    color_channel: str,
    cfa_pattern: str,
    cfa_shifts: dict[str, tuple[float, float]],
    silicon: mp.Medium,
    cfa_media: dict[str, mp.Medium],
    passivation: mp.Medium,
    lens_medium: mp.Medium,
    dti_medium: mp.Medium,
    dti: OpticalDtiConfig,
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    cfa_polygons: CfaPolygonSet,
):
    half_x = 0.5 * layout_nx * geom.pitch
    half_z = 0.5 * layout_nz * geom.pitch
    x_centers = centered_positions(layout_nx, geom.pitch)
    z_centers = centered_positions(layout_nz, geom.pitch)
    half_array_for_dti = max(half_x, half_z)

    def nearest_cfa_color(x: float, z: float) -> str:
        best: tuple[float, str] | None = None
        for ix, xc in enumerate(x_centers):
            for iz, zc in enumerate(z_centers):
                color = cfa_color_for_cell(ix, iz, cfa_pattern, color_channel)
                shift_x, shift_z = cfa_shift_for_color(color, cfa_shifts)
                dx = x - (xc + shift_x)
                dz = z - (zc + shift_z)
                distance2 = dx * dx + dz * dz
                if best is None or distance2 < best[0]:
                    best = (distance2, color)
        return best[1] if best else color_channel

    def cfa_material_at(x: float, z: float):
        color = cfa_polygon_color_at(
            x,
            z,
            x_centers,
            z_centers,
            layout_nx * geom.pitch,
            layout_nz * geom.pitch,
            lambda ix, iz: cfa_color_for_cell(ix, iz, cfa_pattern, color_channel),
            cfa_shifts,
            cfa_polygons,
            periodic=False,
        )
        if color:
            return cfa_media[color]
        if cfa_polygons.polygons:
            if cfa_polygons.background == "passivation":
                return passivation
            if cfa_polygons.background == "air":
                return mp.air
        return cfa_media[nearest_cfa_color(x, z)]

    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        z = point.z
        if abs(x) > half_x or abs(z) > half_z:
            return mp.air
        if geom.si_bottom <= y < geom.si_top:
            if in_dti_trench(point, geom, dti, half_array_for_dti):
                return dti_medium
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            return cfa_material_at(x, z)
        if geom.lens_bottom <= y <= geom.lens_top:
            for item in lenses:
                lens_x = item.x_um + item.shift_x_um + case.lens_shift_x_um
                lens_z = item.z_um + item.shift_z_um + case.lens_shift_z_um
                dx = x - lens_x
                dz = z - lens_z
                if item.polygon_um is not None and not point_in_polygon(dx, dz, item.polygon_um):
                    continue
                profile = sag_profile_for_lens(item.lens_id, ocl_sag_profiles)
                surface_map = surface_map_for_lens(item.lens_id, ocl_surface_maps)
                if inside_lens_volume(dx, dz, y, item.aperture_radius_um, geom, profile, surface_map):
                    return lens_medium
        return mp.air

    return material


def build_layout_regions(
    geom: MicrolensArrayGeometry,
    layout_nx: int,
    layout_nz: int,
    lenses: list[OclLens],
    target: OclLens,
) -> tuple[list[Region], list[Region]]:
    output_regions = []
    raw_pd_regions = []
    x_centers = centered_positions(layout_nx, geom.pitch)
    z_centers = centered_positions(layout_nz, geom.pitch)
    for lens in lenses:
        output_regions.append(
            Region(
                region_id=f"ocl_{lens.lens_id}",
                kind="ocl_output",
                ix=lens.ix - target.ix,
                iz=lens.iz - target.iz,
                x_um=lens.x_um,
                z_um=lens.z_um,
                sx_um=lens.w * geom.pitch,
                sz_um=lens.h * geom.pitch,
            )
        )
    for iz, zc in enumerate(z_centers):
        for ix, xc in enumerate(x_centers):
            raw_pd_regions.append(
                Region(
                    region_id=f"pd_ix{ix}_iz{iz}",
                    kind="raw_pd",
                    ix=ix,
                    iz=iz,
                    x_um=xc,
                    z_um=zc,
                    sx_um=geom.pitch,
                    sz_um=geom.pitch,
                )
            )
    return output_regions, raw_pd_regions


def axis_centers(span_um: float, count: int) -> np.ndarray:
    step = span_um / count
    return np.linspace(-0.5 * span_um + 0.5 * step, 0.5 * span_um - 0.5 * step, count)


def integrate_regions(density: np.ndarray, geom: MicrolensArrayGeometry, span_x: float, span_z: float, regions: list[Region]) -> dict[str, float]:
    dx = span_x / density.shape[0]
    dy = geom.si_thickness / density.shape[1]
    dz = span_z / density.shape[2]
    dvol = dx * dy * dz
    x_values = axis_centers(span_x, density.shape[0])
    z_values = axis_centers(span_z, density.shape[2])
    values = {}
    for region in regions:
        x_mask = np.abs(x_values - region.x_um) <= 0.5 * region.sx_um
        z_mask = np.abs(z_values - region.z_um) <= 0.5 * region.sz_um
        mask_xz = x_mask[:, None] & z_mask[None, :]
        values[region.region_id] = float(np.sum(density * mask_xz[:, None, :]) * dvol)
    return values


def run_one(
    geom: MicrolensArrayGeometry,
    stack_config: dict[str, Any],
    mode: str,
    neighborhood: int,
    case: CrosstalkCase,
    wavelength_nm: float,
    resolution: int,
    after_source_time: float,
    color_channel: str,
    source_scale: float,
    source_profile: str,
    source_sigma_scale: float,
    dti: OpticalDtiConfig,
    guard_cells: int,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    cfa_polygons: CfaPolygonSet,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], np.ndarray]:
    layout_size = mode_layout_size(mode)
    simulation_neighborhood = neighborhood + 2 * guard_cells
    if simulation_neighborhood < neighborhood:
        raise ValueError("simulation neighborhood must be >= output neighborhood")
    if simulation_neighborhood % 2 != 1:
        raise ValueError("simulation neighborhood must be odd")
    wavelength_um = wavelength_nm / 1000.0
    frequency = 1.0 / wavelength_um
    kx, kz = transverse_k(frequency, case.cra_x_deg, case.cra_z_deg)
    supercell_pitch = layout_size * geom.pitch
    output_span = neighborhood * supercell_pitch
    simulation_span = simulation_neighborhood * supercell_pitch
    cell_size = mp.Vector3(simulation_span + 2 * geom.pml, geom.cell_y, simulation_span + 2 * geom.pml)
    source_size = source_scale * supercell_pitch
    source_sigma = max(source_sigma_scale * supercell_pitch, 1e-6)

    silicon, silicon_spec = medium_for_role(stack_config, "silicon", wavelength_um, frequency)
    passivation, passivation_spec = medium_for_role(stack_config, "passivation", wavelength_um, frequency)
    lens, lens_spec = medium_for_role(stack_config, "lens", wavelength_um, frequency)
    cfa_media = {}
    cfa_specs = {}
    for color in CFA_COLORS:
        medium, spec = medium_for_role(stack_config, material_role_for_color(color), wavelength_um, frequency)
        cfa_media[color] = medium
        cfa_specs[color] = spec
    cfa_media["clear"] = passivation
    cfa_specs["clear"] = {
        **passivation_spec,
        "usage": "clear/monochrome transparent CFA proxy using passivation medium",
    }
    si_n, si_k, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    eps_imag = 2.0 * si_n * si_k

    material_function = make_material_function(
        geom,
        layout_size,
        simulation_neighborhood,
        case,
        color_channel,
        silicon,
        cfa_media,
        passivation,
        lens,
        passivation,
        dti,
        ocl_sag_profiles,
        ocl_surface_maps,
        cfa_polygons,
    )
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ez,
        center=mp.Vector3(0, geom.source_y, 0),
        size=mp.Vector3(source_size, 0, source_size),
        amp_func=source_amplitude(kx, kz, source_profile, source_sigma),
    )
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml)],
        sources=[source],
        resolution=resolution,
        force_complex_fields=True,
        default_material=mp.air,
        extra_materials=[silicon, passivation, lens, *cfa_media.values()],
        material_function=material_function,
    )
    fields = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        frequency,
        0,
        1,
        center=mp.Vector3(0, 0.5 * (geom.si_top + geom.si_bottom), 0),
        size=mp.Vector3(simulation_span, geom.si_thickness, simulation_span),
    )
    sim.run(until_after_sources=after_source_time)
    ex = np.asarray(sim.get_dft_array(fields, mp.Ex, 0))
    ey = np.asarray(sim.get_dft_array(fields, mp.Ey, 0))
    ez = np.asarray(sim.get_dft_array(fields, mp.Ez, 0))
    density = eps_imag * (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    total_raw = float(
        np.sum(density)
        * (simulation_span / density.shape[0])
        * (geom.si_thickness / density.shape[1])
        * (simulation_span / density.shape[2])
    )

    output_regions, raw_pd_regions = build_regions(geom, layout_size, neighborhood)
    output_raw = integrate_regions(density, geom, simulation_span, simulation_span, output_regions)
    pd_raw = integrate_regions(density, geom, simulation_span, simulation_span, raw_pd_regions)
    normalize = total_raw if total_raw > 0 else 1.0

    output_rows = []
    output_kernel = np.zeros((neighborhood, neighborhood), dtype=float)
    for region in output_regions:
        value = output_raw[region.region_id] / normalize
        output_kernel[region.iz + neighborhood // 2, region.ix + neighborhood // 2] = value
        color = bayer_color_for_output(region.ix, region.iz, color_channel)
        output_rows.append(
            {
                "schema": "camera_crosstalk_full_array_fdtd_v1",
                "mode": mode,
                "layout_label": layout_label(mode),
                "layout_size": layout_size,
                "neighborhood": neighborhood,
                "simulation_neighborhood": simulation_neighborhood,
                "guard_cells": guard_cells,
                "kernel_scope": "binned_output",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "cra_x_deg": case.cra_x_deg,
                "cra_z_deg": case.cra_z_deg,
                "output_dx": region.ix,
                "output_dz": region.iz,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": region.z_um,
                "response_fraction": value,
                "color": color,
                "color_relation": "target_color" if region.ix == 0 and region.iz == 0 else ("same_color" if color == color_channel else "cross_color"),
                "source_model": "finite_array_center_ocl_impulse_fdtd",
            }
        )
    raw_rows = []
    for region in raw_pd_regions:
        raw_rows.append(
            {
                "schema": "camera_crosstalk_full_array_fdtd_v1",
                "mode": mode,
                "layout_label": layout_label(mode),
                "layout_size": layout_size,
                "neighborhood": neighborhood,
                "simulation_neighborhood": simulation_neighborhood,
                "guard_cells": guard_cells,
                "kernel_scope": "raw_pd",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "raw_pd_ix": region.ix,
                "raw_pd_iz": region.iz,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": region.z_um,
                "response_fraction": pd_raw[region.region_id] / normalize,
                "source_model": "finite_array_center_ocl_impulse_fdtd",
            }
        )

    center = float(output_kernel[neighborhood // 2, neighborhood // 2])
    border = float(
        np.sum(output_kernel[0, :])
        + np.sum(output_kernel[-1, :])
        + np.sum(output_kernel[1:-1, 0])
        + np.sum(output_kernel[1:-1, -1])
    )
    off_center = float(np.sum(output_kernel) - center)
    outside = max(0.0, float(1.0 - np.sum(output_kernel)))
    neighbor_kernel = output_kernel.copy()
    neighbor_kernel[neighborhood // 2, neighborhood // 2] = 0.0
    summary = {
        "schema": "camera_crosstalk_full_array_fdtd_v1",
        "mode": mode,
        "layout_label": layout_label(mode),
        "layout_size": layout_size,
        "neighborhood": neighborhood,
        "simulation_neighborhood": simulation_neighborhood,
        "guard_cells": guard_cells,
        "output_cell_count": neighborhood * neighborhood,
        "raw_pd_kernel_shape": f"{neighborhood * layout_size}x{neighborhood * layout_size}",
        "raw_pd_count": (neighborhood * layout_size) ** 2,
        "case": case.name,
        "wavelength_nm": wavelength_nm,
        "resolution_px_per_um": resolution,
        "color_channel": color_channel,
        "cfa_polygon_count": len(cfa_polygons.polygons),
        "cfa_polygon_background": cfa_polygons.background,
        "cra_x_deg": case.cra_x_deg,
        "cra_z_deg": case.cra_z_deg,
        "field_x_norm": case.field_x_norm,
        "field_z_norm": case.field_z_norm,
        "lens_shift_x_um": case.lens_shift_x_um,
        "lens_shift_z_um": case.lens_shift_z_um,
        "center_response_fraction": center,
        "output_crosstalk_fraction": off_center,
        "border_response_fraction": border,
        "outside_output_kernel_fraction": outside,
        "truncation_response_fraction": outside,
        "support_edge_response_fraction": border,
        "strongest_neighbor_fraction": float(np.max(neighbor_kernel)),
        "total_integrated_response_fraction": float(np.sum(output_kernel)),
        "total_absorption_raw": total_raw,
        "source_model": "finite_array_center_ocl_impulse_fdtd",
        "source_profile": source_profile,
        "source_sigma_um": source_sigma,
        "optical_dti_enabled": dti.enabled,
        "optical_dti_width_um": dti.width_um,
        "optical_dti_depth_um": dti.depth_um,
        "optical_dti_measured": dti.measured,
        "accuracy_status": "full_array_fdtd_not_measured_stack",
        "measured_accuracy_blocked": True,
        "notes": "Finite-array FDTD crosstalk kernel. Product accuracy still requires measured stack geometry and measured n,k.",
        "materials": {
            "silicon": silicon_spec,
            "passivation": passivation_spec,
            "lens": lens_spec,
            "cfa": cfa_specs,
        },
        "optical_dti": asdict(dti),
    }
    summary.update(
        grid_resolution_metadata(
            geom,
            dti,
            stack_config,
            wavelength_nm,
            resolution,
            min_feature_pixels,
            min_si_wavelength_pixels,
        )
    )
    return output_rows, raw_rows, summary, output_kernel


def run_one_layout(
    geom: MicrolensArrayGeometry,
    stack_config: dict[str, Any],
    layout_nx: int,
    layout_nz: int,
    lenses: list[OclLens],
    target_lens_id: str | None,
    layout_name: str,
    neighborhood: int,
    case: CrosstalkCase,
    wavelength_nm: float,
    resolution: int,
    after_source_time: float,
    color_channel: str,
    cfa_pattern: str,
    cfa_shifts: dict[str, tuple[float, float]],
    ocl_sag_profiles: dict[str, OclSagProfile],
    ocl_surface_maps: dict[str, OclSurfaceMap],
    source_scale: float,
    source_profile: str,
    source_sigma_scale: float,
    dti: OpticalDtiConfig,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
    cfa_polygons: CfaPolygonSet,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], np.ndarray]:
    target = choose_target_lens(lenses, target_lens_id)
    wavelength_um = wavelength_nm / 1000.0
    frequency = 1.0 / wavelength_um
    kx, kz = transverse_k(frequency, case.cra_x_deg, case.cra_z_deg)
    layout_span_x = layout_nx * geom.pitch
    layout_span_z = layout_nz * geom.pitch
    cell_size = mp.Vector3(layout_span_x + 2 * geom.pml, geom.cell_y, layout_span_z + 2 * geom.pml)
    source_size_x = source_scale * target.w * geom.pitch
    source_size_z = source_scale * target.h * geom.pitch
    source_sigma = max(source_sigma_scale * max(target.w, target.h) * geom.pitch, 1e-6)
    source_center_x = target.x_um + target.shift_x_um + case.lens_shift_x_um
    source_center_z = target.z_um + target.shift_z_um + case.lens_shift_z_um

    silicon, silicon_spec = medium_for_role(stack_config, "silicon", wavelength_um, frequency)
    passivation, passivation_spec = medium_for_role(stack_config, "passivation", wavelength_um, frequency)
    lens, lens_spec = medium_for_role(stack_config, "lens", wavelength_um, frequency)
    cfa_media = {}
    cfa_specs = {}
    for color in CFA_COLORS:
        medium, spec = medium_for_role(stack_config, material_role_for_color(color), wavelength_um, frequency)
        cfa_media[color] = medium
        cfa_specs[color] = spec
    cfa_media["clear"] = passivation
    cfa_specs["clear"] = {
        **passivation_spec,
        "usage": "clear/monochrome transparent CFA proxy using passivation medium",
    }
    si_n, si_k, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    eps_imag = 2.0 * si_n * si_k

    material_function = make_layout_material_function(
        geom,
        layout_nx,
        layout_nz,
        lenses,
        case,
        color_channel,
        cfa_pattern,
        cfa_shifts,
        silicon,
        cfa_media,
        passivation,
        lens,
        passivation,
        dti,
        ocl_sag_profiles,
        ocl_surface_maps,
        cfa_polygons,
    )
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ez,
        center=mp.Vector3(source_center_x, geom.source_y, source_center_z),
        size=mp.Vector3(source_size_x, 0, source_size_z),
        amp_func=source_amplitude(kx, kz, source_profile, source_sigma, source_center_x, source_center_z),
    )
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml)],
        sources=[source],
        resolution=resolution,
        force_complex_fields=True,
        default_material=mp.air,
        extra_materials=[silicon, passivation, lens, *cfa_media.values()],
        material_function=material_function,
    )
    fields = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],
        frequency,
        0,
        1,
        center=mp.Vector3(0, 0.5 * (geom.si_top + geom.si_bottom), 0),
        size=mp.Vector3(layout_span_x, geom.si_thickness, layout_span_z),
    )
    sim.run(until_after_sources=after_source_time)
    ex = np.asarray(sim.get_dft_array(fields, mp.Ex, 0))
    ey = np.asarray(sim.get_dft_array(fields, mp.Ey, 0))
    ez = np.asarray(sim.get_dft_array(fields, mp.Ez, 0))
    density = eps_imag * (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
    total_raw = float(
        np.sum(density)
        * (layout_span_x / density.shape[0])
        * (geom.si_thickness / density.shape[1])
        * (layout_span_z / density.shape[2])
    )
    normalize = total_raw if total_raw > 0 else 1.0

    output_regions, raw_pd_regions = build_layout_regions(geom, layout_nx, layout_nz, lenses, target)
    output_raw = integrate_regions(density, geom, layout_span_x, layout_span_z, output_regions)
    pd_raw = integrate_regions(density, geom, layout_span_x, layout_span_z, raw_pd_regions)

    layout_label_value = layout_name or f"{layout_nx}x{layout_nz} mixed OCL"
    lens_by_region = {f"ocl_{lens.lens_id}": lens for lens in lenses}
    output_values: dict[str, float] = {}
    output_kernel = np.zeros((layout_nz, layout_nx), dtype=float)
    output_rows = []
    for region in output_regions:
        lens_item = lens_by_region[region.region_id]
        value = output_raw[region.region_id] / normalize
        output_values[lens_item.lens_id] = value
        for iz in range(lens_item.iz, lens_item.iz + lens_item.h):
            for ix in range(lens_item.ix, lens_item.ix + lens_item.w):
                output_kernel[iz, ix] = value / max(lens_item.w * lens_item.h, 1)
        dx_um = lens_item.x_um - target.x_um
        dz_um = lens_item.z_um - target.z_um
        output_rows.append(
            {
                "schema": "camera_crosstalk_full_array_fdtd_v1",
                "mode": "ocl-layout",
                "layout_label": layout_label_value,
                "layout_size": max(layout_nx, layout_nz),
                "layout_nx": layout_nx,
                "layout_nz": layout_nz,
                "ocl_lens_count": len(lenses),
                "target_lens_id": target.lens_id,
                "ocl_lens_id": lens_item.lens_id,
                "ocl_lens_kind": lens_item.kind,
                "lens_ix": lens_item.ix,
                "lens_iz": lens_item.iz,
                "lens_w": lens_item.w,
                "lens_h": lens_item.h,
                "lens_shift_x_um": lens_item.shift_x_um,
                "lens_shift_z_um": lens_item.shift_z_um,
                "neighborhood": neighborhood,
                "simulation_neighborhood": neighborhood,
                "guard_cells": 0,
                "kernel_scope": "mixed_ocl_output",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "cra_x_deg": case.cra_x_deg,
                "cra_z_deg": case.cra_z_deg,
                "output_dx": lens_item.ix - target.ix,
                "output_dz": lens_item.iz - target.iz,
                "output_dx_um": dx_um,
                "output_dz_um": dz_um,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": region.z_um,
                "response_fraction": value,
                "color": "mixed" if lens_item.w * lens_item.h > 1 else cfa_color_for_cell(lens_item.ix, lens_item.iz, cfa_pattern, color_channel),
                "color_relation": "target_lens" if lens_item.lens_id == target.lens_id else "neighbor_lens",
                "source_model": "finite_mixed_ocl_layout_impulse_fdtd",
            }
        )

    raw_rows = []
    for region in raw_pd_regions:
        color = cfa_color_for_cell(region.ix, region.iz, cfa_pattern, color_channel)
        raw_rows.append(
            {
                "schema": "camera_crosstalk_full_array_fdtd_v1",
                "mode": "ocl-layout",
                "layout_label": layout_label_value,
                "layout_size": max(layout_nx, layout_nz),
                "layout_nx": layout_nx,
                "layout_nz": layout_nz,
                "ocl_lens_count": len(lenses),
                "target_lens_id": target.lens_id,
                "neighborhood": neighborhood,
                "simulation_neighborhood": neighborhood,
                "guard_cells": 0,
                "kernel_scope": "raw_pd",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "raw_pd_ix": region.ix,
                "raw_pd_iz": region.iz,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": region.z_um,
                "response_fraction": pd_raw[region.region_id] / normalize,
                "color": color,
                "source_model": "finite_mixed_ocl_layout_impulse_fdtd",
            }
        )

    target_value = float(output_values.get(target.lens_id, 0.0))
    off_target = float(sum(value for lens_id, value in output_values.items() if lens_id != target.lens_id))
    edge_neighbor_value = 0.0
    strongest_neighbor = 0.0
    for lens_item in lenses:
        value = float(output_values.get(lens_item.lens_id, 0.0))
        touches_edge = (
            lens_item.ix == 0
            or lens_item.iz == 0
            or lens_item.ix + lens_item.w == layout_nx
            or lens_item.iz + lens_item.h == layout_nz
        )
        if touches_edge and lens_item.lens_id != target.lens_id:
            edge_neighbor_value += value
        if lens_item.lens_id != target.lens_id:
            strongest_neighbor = max(strongest_neighbor, value)
    outside = max(0.0, float(1.0 - sum(output_values.values())))
    truncation_proxy = max(outside, edge_neighbor_value)

    summary = {
        "schema": "camera_crosstalk_full_array_fdtd_v1",
        "mode": "ocl-layout",
        "layout_label": layout_label_value,
        "layout_size": max(layout_nx, layout_nz),
        "layout_nx": layout_nx,
        "layout_nz": layout_nz,
        "ocl_lens_count": len(lenses),
        "target_lens_id": target.lens_id,
        "neighborhood": neighborhood,
        "simulation_neighborhood": neighborhood,
        "guard_cells": 0,
        "output_cell_count": len(lenses),
        "raw_pd_kernel_shape": f"{layout_nx}x{layout_nz}",
        "raw_pd_count": layout_nx * layout_nz,
        "case": case.name,
        "wavelength_nm": wavelength_nm,
        "resolution_px_per_um": resolution,
        "color_channel": color_channel,
        "cfa_pattern": cfa_pattern,
        "cfa_polygon_count": len(cfa_polygons.polygons),
        "cfa_polygon_background": cfa_polygons.background,
        "cra_x_deg": case.cra_x_deg,
        "cra_z_deg": case.cra_z_deg,
        "field_x_norm": case.field_x_norm,
        "field_z_norm": case.field_z_norm,
        "lens_shift_x_um": case.lens_shift_x_um,
        "lens_shift_z_um": case.lens_shift_z_um,
        "center_response_fraction": target_value,
        "output_crosstalk_fraction": off_target,
        "border_response_fraction": edge_neighbor_value,
        "outside_output_kernel_fraction": outside,
        "truncation_response_fraction": truncation_proxy,
        "support_edge_response_fraction": edge_neighbor_value,
        "strongest_neighbor_fraction": strongest_neighbor,
        "total_integrated_response_fraction": float(sum(output_values.values())),
        "total_absorption_raw": total_raw,
        "source_model": "finite_mixed_ocl_layout_impulse_fdtd",
        "source_profile": source_profile,
        "source_sigma_um": source_sigma,
        "optical_dti_enabled": dti.enabled,
        "optical_dti_width_um": dti.width_um,
        "optical_dti_depth_um": dti.depth_um,
        "optical_dti_measured": dti.measured,
        "accuracy_status": "mixed_ocl_fdtd_not_measured_stack",
        "measured_accuracy_blocked": True,
        "notes": (
            "Finite mixed-OCL layout FDTD. Edge response is treated as the truncation proxy because "
            "the local layout does not model the infinite continuation outside the boundary window."
        ),
        "materials": {
            "silicon": silicon_spec,
            "passivation": passivation_spec,
            "lens": lens_spec,
            "cfa": cfa_specs,
        },
        "optical_dti": asdict(dti),
        "ocl_layout": [asdict(lens_item) for lens_item in lenses],
    }
    summary.update(
        grid_resolution_metadata(
            geom,
            dti,
            stack_config,
            wavelength_nm,
            resolution,
            min_feature_pixels,
            min_si_wavelength_pixels,
        )
    )
    return output_rows, raw_rows, summary, output_kernel


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(path: Path, kernels: list[tuple[dict[str, Any], np.ndarray]]) -> None:
    if not kernels:
        return
    cols = min(3, len(kernels))
    rows = math.ceil(len(kernels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    vmax = max(float(np.max(kernel)) for _, kernel in kernels)
    for axis, (summary, kernel) in zip(axes, kernels):
        image = axis.imshow(kernel, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(
            f"{summary['layout_label']} {summary['case']}\n"
            f"N={summary['neighborhood']}, XT={100*summary['output_crosstalk_fraction']:.2f}%, "
            f"out={100*summary.get('outside_output_kernel_fraction', 0.0):.2f}%"
        )
        axis.set_xticks(range(kernel.shape[1]))
        axis.set_yticks(range(kernel.shape[0]))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    for axis in axes[len(kernels) :]:
        axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_kernels_from_rows(
    summaries: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], np.ndarray]]:
    if not summaries or not output_rows:
        return []
    max_neighborhood = max(int(row["neighborhood"]) for row in summaries)
    max_resolution = max(int(row["resolution_px_per_um"]) for row in summaries)
    selected = [
        row
        for row in summaries
        if int(row["neighborhood"]) == max_neighborhood and int(row["resolution_px_per_um"]) == max_resolution
    ]
    kernels = []
    for summary in selected:
        neighborhood = int(summary["neighborhood"])
        half = neighborhood // 2
        kernel = np.zeros((neighborhood, neighborhood), dtype=float)
        for row in output_rows:
            if (
                str(row.get("mode")) == str(summary.get("mode"))
                and str(row.get("case")) == str(summary.get("case"))
                and float(row.get("wavelength_nm", 0.0)) == float(summary.get("wavelength_nm", 0.0))
                and int(row.get("neighborhood", 0)) == neighborhood
                and int(row.get("simulation_neighborhood", 0)) == int(summary.get("simulation_neighborhood", 0))
                and int(row.get("resolution_px_per_um", 0)) == max_resolution
            ):
                kernel[int(row["output_dz"]) + half, int(row["output_dx"]) + half] = float(row["response_fraction"])
        kernels.append((summary, kernel))
    return kernels


def convergence_status(
    summaries: list[dict[str, Any]],
    truncation_threshold: float,
    delta_threshold: float,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in summaries:
        groups.setdefault((row["mode"], row["case"], row["wavelength_nm"]), []).append(row)
    checks = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda item: (item["neighborhood"], item["resolution_px_per_um"]))
        largest_neighborhood = max(int(row["neighborhood"]) for row in rows)
        largest_rows = [row for row in rows if int(row["neighborhood"]) == largest_neighborhood]
        largest_rows = sorted(largest_rows, key=lambda item: item["resolution_px_per_um"])
        largest = largest_rows[-1]
        truncation_metric = float(
            largest.get(
                "truncation_response_fraction",
                largest.get("outside_output_kernel_fraction", largest.get("border_response_fraction", 1.0)),
            )
        )
        truncation_pass = truncation_metric <= truncation_threshold
        checks.append(
            {
                "name": f"{key[0]} {key[1]} {key[2]:g}nm kernel truncation",
                "status": "PASS" if truncation_pass else "FAIL",
                "metric": truncation_metric,
                "threshold": truncation_threshold,
                "neighborhood": largest["neighborhood"],
                "simulation_neighborhood": largest.get("simulation_neighborhood"),
                "resolution_px_per_um": largest["resolution_px_per_um"],
                "support_edge_response_fraction": largest.get("support_edge_response_fraction", largest.get("border_response_fraction")),
            }
        )
        si_pixels = float(largest.get("si_internal_wavelength_pixels", 0.0) or 0.0)
        checks.append(
            {
                "name": f"{key[0]} {key[1]} {key[2]:g}nm Si wavelength grid",
                "status": "PASS" if si_pixels >= min_si_wavelength_pixels else "FAIL",
                "metric": si_pixels,
                "threshold": min_si_wavelength_pixels,
                "neighborhood": largest["neighborhood"],
                "resolution_px_per_um": largest["resolution_px_per_um"],
                "details": largest.get("grid_resolution_notes", ""),
            }
        )
        feature_pixels = float(largest.get("minimum_critical_feature_pixels", 0.0) or 0.0)
        checks.append(
            {
                "name": f"{key[0]} {key[1]} {key[2]:g}nm critical feature grid",
                "status": "PASS" if feature_pixels >= min_feature_pixels else "FAIL",
                "metric": feature_pixels,
                "threshold": min_feature_pixels,
                "neighborhood": largest["neighborhood"],
                "resolution_px_per_um": largest["resolution_px_per_um"],
                "details": largest.get("grid_resolution_notes", ""),
            }
        )
        if len(largest_rows) > 1:
            previous = largest_rows[-2]
            relative_metrics = ("center_response_fraction", "output_crosstalk_fraction")
            deltas = {}
            for metric_name in relative_metrics:
                current_value = float(largest.get(metric_name, 0.0) or 0.0)
                previous_value = float(previous.get(metric_name, 0.0) or 0.0)
                denom = max(abs(current_value), 1e-12)
                deltas[metric_name] = abs(current_value - previous_value) / denom
            current_truncation = float(largest.get("truncation_response_fraction", 0.0) or 0.0)
            previous_truncation = float(previous.get("truncation_response_fraction", 0.0) or 0.0)
            deltas["truncation_response_fraction_abs_to_threshold"] = (
                abs(current_truncation - previous_truncation) / max(truncation_threshold, 1e-12)
            )
            rel_delta = max(deltas.values())
            checks.append(
                {
                    "name": f"{key[0]} {key[1]} {key[2]:g}nm kernel convergence",
                    "status": "PASS" if rel_delta <= delta_threshold else "FAIL",
                    "metric": rel_delta,
                    "threshold": delta_threshold,
                    "neighborhood": largest["neighborhood"],
                    "resolution_px_per_um": largest["resolution_px_per_um"],
                    "previous_resolution_px_per_um": previous["resolution_px_per_um"],
                    "metric_deltas": deltas,
                }
            )
        else:
            checks.append(
                {
                    "name": f"{key[0]} {key[1]} {key[2]:g}nm kernel convergence",
                    "status": "WARN",
                    "metric": None,
                    "threshold": delta_threshold,
                    "neighborhood": largest["neighborhood"],
                    "resolution_px_per_um": largest["resolution_px_per_um"],
                    "details": "Only one resolution point was run for the largest exported kernel support.",
                }
            )
    status = "PASS" if checks and all(check["status"] == "PASS" for check in checks) else "FAIL"
    if checks and any(check["status"] == "WARN" for check in checks) and not any(check["status"] == "FAIL" for check in checks):
        status = "WARN"
    return {
        "schema": "crosstalk_convergence_report_v1",
        "status": status,
        "truncation_threshold": truncation_threshold,
        "delta_threshold": delta_threshold,
        "min_feature_pixels": min_feature_pixels,
        "min_si_wavelength_pixels": min_si_wavelength_pixels,
        "checks": checks,
    }


def cfa_shift_args_to_dict(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return {
        "red": (args.cfa_shift_red_x_um, args.cfa_shift_red_z_um),
        "green": (args.cfa_shift_green_x_um, args.cfa_shift_green_z_um),
        "blue": (args.cfa_shift_blue_x_um, args.cfa_shift_blue_z_um),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="split-pd-1x1,ocl-2x2,ocl-3x3")
    parser.add_argument("--neighborhoods", default="3")
    parser.add_argument("--resolutions", default="4")
    parser.add_argument("--wavelengths-nm", default="550")
    parser.add_argument("--cases", default="center:0:0:0:0,edge20x:20:0:1:0")
    parser.add_argument("--color-channel", choices=COLOR_CHANNELS, default="green")
    parser.add_argument("--cfa-pattern", choices=("uniform", "bayer", "quad", "nona"), default="uniform")
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
    parser.add_argument("--layout-nx", type=int, default=None)
    parser.add_argument("--layout-nz", type=int, default=None)
    parser.add_argument("--ocl-layout", default=None)
    parser.add_argument(
        "--ocl-polygons",
        default=None,
        help="JSON map or @file.json import of lens id to local-um polygon points; requires --ocl-layout.",
    )
    parser.add_argument(
        "--ocl-sag",
        default=None,
        help="JSON map or @file.json import of 'default' or lens id to sag profile; supports sphere and asphere.",
    )
    parser.add_argument(
        "--ocl-surface-map",
        default=None,
        help="JSON map or @file.json import of 'default' or lens id to measured/freeform height map: x_um, z_um, height_um.",
    )
    parser.add_argument("--ocl-layout-name", default=None)
    parser.add_argument("--target-lens-id", default=None)
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--tcad-profile", type=Path, default=DEFAULT_TCAD_PROFILE)
    parser.add_argument("--pml-um", type=float, default=0.45)
    parser.add_argument("--after-source-time", type=float, default=16.0)
    parser.add_argument("--source-scale", type=float, default=0.92)
    parser.add_argument("--source-profile", choices=("gaussian", "rect"), default="gaussian")
    parser.add_argument("--source-sigma-scale", type=float, default=0.34)
    parser.add_argument("--guard-cells", type=int, default=1)
    parser.add_argument("--truncation-threshold", "--border-threshold", dest="truncation_threshold", type=float, default=0.015)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--min-feature-pixels", type=float, default=2.0)
    parser.add_argument("--min-si-wavelength-pixels", type=float, default=8.0)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/crosstalk_kernel_reference")
    args = parser.parse_args()

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    neighborhoods = parse_csv_ints(args.neighborhoods)
    resolutions = parse_csv_ints(args.resolutions)
    wavelengths = parse_csv_floats(args.wavelengths_nm)
    cases = parse_cases(args.cases)
    stack_config = load_stack_config(args.stack_config)
    shield = shield_config_for_stack(stack_config)
    if shield["enabled"]:
        raise ValueError("Full-array crosstalk kernel currently supports baseline imaging pixels with shield.mode=off")
    geom = geometry_from_config(stack_config, pml_um=args.pml_um)
    dti = optical_dti_from_profile(args.tcad_profile, geom)
    cfa_shifts = cfa_shift_args_to_dict(args)
    cfa_polygons = parse_cfa_polygons(args.cfa_polygons, geom)
    ocl_sag_profiles = parse_ocl_sag_profiles(args.ocl_sag)
    ocl_surface_maps = parse_ocl_surface_maps(args.ocl_surface_map, geom)
    ocl_lenses: list[OclLens] = []
    if "ocl-layout" in modes:
        if args.layout_nx is None or args.layout_nz is None:
            raise ValueError("ocl-layout mode requires --layout-nx and --layout-nz")
        if args.layout_nx < 1 or args.layout_nz < 1 or args.layout_nx > 12 or args.layout_nz > 12:
            raise ValueError("layout dimensions must be between 1 and 12 pixels")
        if not args.ocl_layout:
            raise ValueError("ocl-layout mode requires --ocl-layout")
        ocl_lenses = apply_ocl_polygons(
            parse_ocl_layout(args.ocl_layout, args.layout_nx, args.layout_nz, geom),
            args.ocl_polygons,
            geom,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    plot_kernels: list[tuple[dict[str, Any], np.ndarray]] = []
    for mode in modes:
        for neighborhood in neighborhoods:
            if mode != "ocl-layout" and neighborhood % 2 != 1:
                raise ValueError("neighborhoods must be odd")
            for resolution in resolutions:
                for wavelength_nm in wavelengths:
                    for case in cases:
                        print(
                            f"running mode={mode}, neighborhood={neighborhood}, res={resolution}, "
                            f"wavelength={wavelength_nm:g}nm, case={case.name}"
                        )
                        if mode == "ocl-layout":
                            out, raw, summary, kernel = run_one_layout(
                                geom,
                                stack_config,
                                int(args.layout_nx),
                                int(args.layout_nz),
                                ocl_lenses,
                                args.target_lens_id,
                                args.ocl_layout_name or "",
                                neighborhood,
                                case,
                                wavelength_nm,
                                resolution,
                                args.after_source_time,
                                args.color_channel,
                                args.cfa_pattern,
                                cfa_shifts,
                                ocl_sag_profiles,
                                ocl_surface_maps,
                                args.source_scale,
                                args.source_profile,
                                args.source_sigma_scale,
                                dti,
                                args.min_feature_pixels,
                                args.min_si_wavelength_pixels,
                                cfa_polygons,
                            )
                        else:
                            out, raw, summary, kernel = run_one(
                                geom,
                                stack_config,
                                mode,
                                neighborhood,
                                case,
                                wavelength_nm,
                                resolution,
                                args.after_source_time,
                                args.color_channel,
                                args.source_scale,
                                args.source_profile,
                                args.source_sigma_scale,
                                dti,
                                args.guard_cells,
                                args.min_feature_pixels,
                                args.min_si_wavelength_pixels,
                                ocl_sag_profiles,
                                ocl_surface_maps,
                                cfa_polygons,
                            )
                        output_rows.extend(out)
                        raw_rows.extend(raw)
                        summaries.append(summary)
                        if neighborhood == max(neighborhoods) and resolution == max(resolutions):
                            plot_kernels.append((summary, kernel))

    output_csv = args.output_dir / "crosstalk_output_kernel.csv"
    raw_csv = args.output_dir / "crosstalk_raw_pd_kernel.csv"
    summary_csv = args.output_dir / "crosstalk_kernel_summary.csv"
    heatmap_png = args.output_dir / "crosstalk_kernel_heatmap.png"
    if args.merge_existing:
        output_rows = merge_rows(read_csv_rows(output_csv), output_rows)
        raw_rows = merge_rows(read_csv_rows(raw_csv), raw_rows)
        summaries = merge_rows(read_csv_rows(summary_csv), summaries)
        summaries = enrich_summaries_with_grid_metadata(
            summaries,
            geom,
            dti,
            stack_config,
            args.min_feature_pixels,
            args.min_si_wavelength_pixels,
        )
    if args.merge_existing:
        plot_kernels = plot_kernels_from_rows(summaries, output_rows)
    write_csv(output_csv, output_rows)
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summaries)
    save_heatmap(heatmap_png, plot_kernels)
    convergence = convergence_status(
        summaries,
        args.truncation_threshold,
        args.delta_threshold,
        args.min_feature_pixels,
        args.min_si_wavelength_pixels,
    )
    convergence_path = args.output_dir / "crosstalk_convergence.json"
    convergence_path.write_text(json.dumps(convergence, indent=2), encoding="utf-8")

    manifest = {
        "schema": "camera_crosstalk_full_array_fdtd_v1",
        "solver": "Meep finite-array FDTD",
        "source_model": "center OCL finite-aperture impulse",
        "accuracy_status": "full_array_fdtd_not_measured_stack",
        "measured_accuracy_blocked": True,
        "convergence_status": convergence["status"],
        "configuration": {
            "modes": unique_values(summaries, "mode", str),
            "neighborhoods": unique_values(summaries, "neighborhood", int),
            "simulation_neighborhoods": unique_values(summaries, "simulation_neighborhood", int),
            "resolutions_px_per_um": unique_values(summaries, "resolution_px_per_um", int),
            "wavelengths_nm": unique_values(summaries, "wavelength_nm", float),
            "cases": cases_from_summaries(summaries),
            "color_channel": args.color_channel,
            "cfa_pattern": args.cfa_pattern,
            "cfa_shifts_um": {
                color: {"x": shift[0], "z": shift[1]} for color, shift in cfa_shifts.items()
            },
            "cfa_polygons": args.cfa_polygons,
            "cfa_polygons_source": cfa_polygons.source,
            "cfa_polygon_background": cfa_polygons.background,
            "cfa_polygon_count": len(cfa_polygons.polygons),
            "layout_nx": args.layout_nx,
            "layout_nz": args.layout_nz,
            "ocl_layout_name": args.ocl_layout_name,
            "ocl_polygons": args.ocl_polygons,
            "ocl_polygons_source": json_argument_source(args.ocl_polygons, "--ocl-polygons"),
            "ocl_sag": args.ocl_sag,
            "ocl_sag_source": json_argument_source(args.ocl_sag, "--ocl-sag"),
            "ocl_sag_profiles": {key: asdict(value) for key, value in ocl_sag_profiles.items()},
            "ocl_surface_map": args.ocl_surface_map,
            "ocl_surface_map_source": json_argument_source(args.ocl_surface_map, "--ocl-surface-map"),
            "ocl_surface_maps": {
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
            "target_lens_id": args.target_lens_id,
            "ocl_layout": [asdict(lens_item) for lens_item in ocl_lenses],
            "guard_cells": args.guard_cells,
            "stack_config": str(args.stack_config),
            "tcad_profile": str(args.tcad_profile) if args.tcad_profile else "",
            "geometry_um": asdict(geom),
            "optical_dti": asdict(dti),
            "source_scale": args.source_scale,
            "source_profile": args.source_profile,
            "source_sigma_scale": args.source_sigma_scale,
            "after_source_time": args.after_source_time,
            "min_feature_pixels": args.min_feature_pixels,
            "min_si_wavelength_pixels": args.min_si_wavelength_pixels,
            "last_run_request": {
                "modes": modes,
                "neighborhoods": neighborhoods,
                "resolutions_px_per_um": resolutions,
                "wavelengths_nm": wavelengths,
                "cases": [asdict(case) for case in cases],
                "merge_existing": args.merge_existing,
            },
        },
        "scope": {
            "primary_kernel": "binned output-cell crosstalk",
            "diagnostic_kernel": "raw physical-PD crosstalk",
            "required_product_accuracy_inputs": [
                "measured stack geometry",
                "measured wavelength-dependent n,k",
                "measured color-filter transmission",
                "electrical collection calibration for converting optical absorption to electrons",
            ],
        },
        "summaries": summaries,
        "convergence": convergence,
        "outputs": {
            "output_kernel_csv": str(output_csv),
            "raw_pd_kernel_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
            "heatmap_png": str(heatmap_png),
            "convergence_json": str(convergence_path),
        },
    }
    manifest_path = args.output_dir / "crosstalk_kernel.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output_dir": str(args.output_dir),
                "summary_count": len(summaries),
                "convergence_status": convergence["status"],
                "outputs": manifest["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
