#!/usr/bin/env python3
"""Generate reusable parametric CAD templates for image-sensor pixels.

The intent is to stop scattering proxy assumptions through solver commands.
Each template writes an explicit parameter file plus CAD artifacts that can be
opened in FreeCAD or other STEP/BREP readers, and a footprint JSON that the
existing FDTD import path can consume.

This is not measured product geometry. It is a reusable parametric source of
truth for common structures until measured CAD/profilometry is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CFA_COLORS = ("red", "green", "blue")


@dataclass(frozen=True)
class OclBlock:
    lens_id: str
    ix: int
    iz: int
    sx: int
    sz: int
    height_um: float | None = None


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    label: str
    nx: int
    nz: int
    cfa_pattern: str
    ocl_blocks: tuple[OclBlock, ...]
    split_mode: str = "none"
    shield_mode: str = "off"
    pitch_um: float = 1.4
    si_thickness_um: float = 3.0
    passivation_thickness_um: float = 0.08
    cfa_thickness_um: float = 0.8
    lens_height_um: float = 0.657
    lens_edge_gap_um: float = 0.08
    cfa_gap_um: float = 0.02
    dti_width_um: float = 0.06
    dti_depth_um: float = 3.0
    pd_depth_min_um: float = 0.32
    pd_depth_max_um: float = 1.28
    pd_margin_um: float = 0.26
    notes: tuple[str, ...] = field(default_factory=tuple)


def load_gmsh():
    try:
        import gmsh  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "gmsh is required. Use the local TCAD environment: "
            "`.tcad-env/bin/python pixel_cad_template_library.py`."
        ) from error
    return gmsh


def bayer_color(ix: int, iz: int) -> str:
    even_x = ix % 2 == 0
    even_z = iz % 2 == 0
    if even_x and even_z:
        return "red"
    if (not even_x) and (not even_z):
        return "blue"
    return "green"


def cfa_color(pattern: str, ix: int, iz: int) -> str:
    if pattern == "uniform_green":
        return "green"
    if pattern == "bayer":
        return bayer_color(ix, iz)
    if pattern == "quad":
        return bayer_color(ix // 2, iz // 2)
    if pattern == "nona":
        return bayer_color(ix // 3, iz // 3)
    raise ValueError(f"Unsupported CFA pattern: {pattern}")


def rect_points(width_um: float, height_um: float) -> list[list[float]]:
    hx = 0.5 * width_um
    hz = 0.5 * height_um
    return [[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]]


def block_lens_height(spec: TemplateSpec, block: OclBlock) -> float:
    return spec.lens_height_um if block.height_um is None else float(block.height_um)


def spherical_edge_slope(aperture_width_um: float, lens_height_um: float) -> float:
    aperture_radius = 0.5 * aperture_width_um
    radius = (aperture_radius * aperture_radius + lens_height_um * lens_height_um) / max(2.0 * lens_height_um, 1.0e-9)
    root = max(radius * radius - aperture_radius * aperture_radius, 1.0e-18)
    return aperture_radius / math.sqrt(root)


def spherical_height_for_edge_slope(aperture_width_um: float, target_slope: float) -> float:
    aperture_radius = 0.5 * aperture_width_um
    if target_slope <= 0.0:
        return 0.0
    return aperture_radius * (math.sqrt(1.0 + target_slope * target_slope) - 1.0) / target_slope


def cell_center(spec: TemplateSpec, ix: int, iz: int) -> tuple[float, float]:
    total_x = spec.nx * spec.pitch_um
    total_z = spec.nz * spec.pitch_um
    return (
        (ix + 0.5) * spec.pitch_um - 0.5 * total_x,
        (iz + 0.5) * spec.pitch_um - 0.5 * total_z,
    )


def block_center(spec: TemplateSpec, block: OclBlock) -> tuple[float, float]:
    x0, z0 = cell_center(spec, block.ix, block.iz)
    x1, z1 = cell_center(spec, block.ix + block.sx - 1, block.iz + block.sz - 1)
    return 0.5 * (x0 + x1), 0.5 * (z0 + z1)


def pd_volume_multiplier(split_mode: str) -> int:
    if split_mode in {"dual_x", "dual-x", "dual_z", "dual-z"}:
        return 2
    if split_mode == "quad":
        return 4
    return 1


def expected_shield_volume_count(shield_mode: str) -> int:
    if shield_mode == "off":
        return 0
    if shield_mode == "pdaf_pair":
        return 2
    return 1


def build_footprint_payload(spec: TemplateSpec) -> dict[str, Any]:
    cfa_cells = []
    tile_width = max(spec.pitch_um - spec.cfa_gap_um, spec.pitch_um * 0.25)
    tile_points = rect_points(tile_width, tile_width)
    for iz in range(spec.nz):
        for ix in range(spec.nx):
            color = cfa_color(spec.cfa_pattern, ix, iz)
            cfa_cells.append(
                {
                    "id": f"{color}_{ix}_{iz}",
                    "color": color,
                    "ix": ix,
                    "iz": iz,
                    "points": tile_points,
                    "source": f"parametric CAD template {spec.template_id}",
                }
            )

    ocl_polygons = {}
    ocl_lens_parameters = {}
    for block in spec.ocl_blocks:
        width = max(block.sx * spec.pitch_um - spec.lens_edge_gap_um, spec.pitch_um * 0.25)
        height = max(block.sz * spec.pitch_um - spec.lens_edge_gap_um, spec.pitch_um * 0.25)
        ocl_polygons[block.lens_id] = rect_points(width, height)
        ocl_lens_parameters[block.lens_id] = {
            "height_um": block_lens_height(spec, block),
            "surface": "spherical_cap",
            "edge_slope_abs": spherical_edge_slope(min(width, height), block_lens_height(spec, block)),
        }

    return {
        "schema": "pixel_geometry_import_v1",
        "units": "um",
        "source": f"parametric CAD template {spec.template_id}",
        "cad_template": {
            "template_id": spec.template_id,
            "label": spec.label,
            "source_truth_level": "parametric_template_not_measured",
            "freecad_openable_artifacts": ["model.step", "model.brep"],
        },
        "notes": [
            "Coordinates map x/z footprints in microns.",
            "This is explicit parametric geometry, not measured process geometry.",
            "Use measured OCL profilometry and n,k data before claiming product LUT accuracy.",
        ],
        "ocl_polygons": ocl_polygons,
        "ocl_lens_parameters": ocl_lens_parameters,
        "cfa_polygons": {
            "background": "passivation",
            "cells": cfa_cells,
        },
    }


def solver_cfa_mapping(cfa_pattern: str) -> dict[str, str]:
    if cfa_pattern.startswith("uniform_"):
        color = cfa_pattern.split("_", 1)[1]
        if color in CFA_COLORS:
            return {
                "raw_cfa_pattern": cfa_pattern,
                "solver_cfa_pattern": "uniform",
                "solver_color_channel": color,
            }
    return {
        "raw_cfa_pattern": cfa_pattern,
        "solver_cfa_pattern": cfa_pattern,
        "solver_color_channel": "green",
    }


def assumption_ledger(spec: TemplateSpec) -> dict[str, Any]:
    total_pixels = spec.nx * spec.nz
    total_x_um = spec.nx * spec.pitch_um
    total_z_um = spec.nz * spec.pitch_um
    cfa_mapping = solver_cfa_mapping(spec.cfa_pattern)
    assumptions = [
        {
            "category": "source",
            "item": "geometry_source",
            "value": "parametric_template",
            "truth_level": "not_measured",
            "impact": "Template dimensions are explicit and repeatable, but not product mask/profilometry data.",
        },
        {
            "category": "optical",
            "item": "ocl_surface",
            "value": "spherical_cap_from_height_and_aperture",
            "truth_level": "analytic_proxy",
            "impact": "Real reflow/asphere/freeform microlens profile can shift focus and CRA response.",
        },
        {
            "category": "optical",
            "item": "cfa_footprint",
            "value": f"{spec.cfa_pattern} rectangular tiles with {spec.cfa_gap_um:.3f} um gap",
            "truth_level": "parametric_proxy",
            "impact": "Real CFA corner rounding, overlay, taper, and process bias are not represented.",
        },
        {
            "category": "device",
            "item": "photodiode_geometry",
            "value": f"box PD depth {spec.pd_depth_min_um:.3f}-{spec.pd_depth_max_um:.3f} um",
            "truth_level": "proxy",
            "impact": "Implant profile, pinned layer, transfer gate coupling, and junction curvature are not encoded in CAD.",
        },
        {
            "category": "isolation",
            "item": "dti_geometry",
            "value": f"{spec.dti_width_um:.3f} um width, {spec.dti_depth_um:.3f} um depth",
            "truth_level": "proxy",
            "impact": "Real DTI/BDTI material stack, taper, liner, stress, and interface traps are not calibrated.",
        },
        {
            "category": "material",
            "item": "materials",
            "value": "geometry_only",
            "truth_level": "not_in_cad",
            "impact": "Measured n,k, absorption, mobility, recombination, and trap parameters must be loaded separately.",
        },
        {
            "category": "solver",
            "item": "tcad_bridge",
            "value": "2d_parameter_derived_mesh",
            "truth_level": "smoke_bridge",
            "impact": "Useful for wiring and trend checks, not a full 3D calibrated product TCAD mesh.",
        },
    ]
    if spec.shield_mode != "off":
        assumptions.append(
            {
                "category": "pdaf",
                "item": "shield",
                "value": spec.shield_mode,
                "truth_level": "proxy",
                "impact": "PDAF shield is a simplified metal aperture, not a process-calibrated optical/electrical stack.",
            }
        )
    measured_blockers = [
        "measured mask/GDS or silicon SEM-derived dimensions",
        "measured OCL surface/profilometry for each field location",
        "measured CFA/passivation/Si thickness and n,k tables",
        "implant profiles for pinned PD, TG, FD, wells, and isolation",
        "interface trap, mobility, recombination, and contact calibration targets",
        "solver convergence pass at quantitative resolution",
    ]
    return {
        "schema": "cad_template_assumption_ledger_v1",
        "template_id": spec.template_id,
        "label": spec.label,
        "source_truth_level": "parametric_template_not_measured",
        "product_accuracy_ready": False,
        "dimension_summary_um": {
            "pitch": spec.pitch_um,
            "domain_x": total_x_um,
            "domain_z": total_z_um,
            "si_thickness": spec.si_thickness_um,
            "passivation_thickness": spec.passivation_thickness_um,
            "cfa_thickness": spec.cfa_thickness_um,
            "lens_height": spec.lens_height_um,
        },
        "topology": {
            "pixels": total_pixels,
            "nx": spec.nx,
            "nz": spec.nz,
            "ocl_blocks": [asdict(block) for block in spec.ocl_blocks],
            "split_mode": spec.split_mode,
            "shield_mode": spec.shield_mode,
        },
        "solver_mapping": {
            **cfa_mapping,
            "fdtd_footprint_source": "geometry_import.json",
            "cad_review_source": ["model.step", "model.brep"],
            "cad_mesh_review": "model.msh",
            "tcad_bridge": "tcad_bridge_2d/split_pixel_2d.msh when generated",
        },
        "assumptions": assumptions,
        "measured_blockers": measured_blockers,
        "review_checklist": [
            "Open model.step/model.brep in FreeCAD or another CAD viewer.",
            "Check OCL footprint and CFA color grouping against the intended pixel topology.",
            "Confirm DTI/PD/shield geometry is appropriate for the simulation question.",
            "Use geometry_import.json for FDTD footprint import only after review.",
            "Do not use smoke or proxy results as a product LUT without measured data and convergence pass.",
        ],
        "notes": list(spec.notes),
    }


def add_box(gmsh: Any, name: str, x: float, y: float, z: float, dx: float, dy: float, dz: float) -> int:
    tag = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
    gmsh.model.setEntityName(3, tag, name)
    return tag


def add_spherical_cap(
    gmsh: Any,
    name: str,
    cx: float,
    base_y: float,
    cz: float,
    width: float,
    depth: float,
    height: float,
) -> list[int]:
    aperture_radius = 0.5 * min(width, depth)
    radius = (aperture_radius * aperture_radius + height * height) / max(2.0 * height, 1.0e-9)
    sphere_center_y = base_y + height - radius
    sphere = gmsh.model.occ.addSphere(cx, sphere_center_y, cz, radius)
    clip = gmsh.model.occ.addBox(cx - 0.5 * width, base_y, cz - 0.5 * depth, width, height, depth)
    result, _ = gmsh.model.occ.intersect([(3, sphere)], [(3, clip)], removeObject=True, removeTool=True)
    tags = [tag for dim, tag in result if dim == 3]
    for index, tag in enumerate(tags):
        gmsh.model.setEntityName(3, tag, f"{name}_{index}" if len(tags) > 1 else name)
    return tags


def add_photodiode_volumes(
    gmsh: Any,
    volumes: dict[str, list[int]],
    spec: TemplateSpec,
    cx: float,
    cz: float,
    ix: int,
    iz: int,
) -> None:
    pd_width = max(spec.pitch_um - 2.0 * spec.pd_margin_um, spec.pitch_um * 0.25)
    pd_y = -spec.pd_depth_max_um
    pd_height = max(spec.pd_depth_max_um - spec.pd_depth_min_um, 0.05)
    gap = min(0.04, pd_width * 0.16)
    split_mode = spec.split_mode.replace("-", "_")

    if split_mode == "quad":
        half = 0.5 * (pd_width - gap)
        for dx_index, x_sign in enumerate((-1, 1)):
            for dz_index, z_sign in enumerate((-1, 1)):
                qx = cx + x_sign * 0.25 * (pd_width + gap)
                qz = cz + z_sign * 0.25 * (pd_width + gap)
                volumes["photodiode"].append(
                    add_box(
                        gmsh,
                        f"pd_q{dx_index}{dz_index}_{ix}_{iz}",
                        qx - 0.5 * half,
                        pd_y,
                        qz - 0.5 * half,
                        half,
                        pd_height,
                        half,
                    )
                )
        return

    if split_mode == "dual_x":
        half = 0.5 * (pd_width - gap)
        for side, x_sign in (("left", -1), ("right", 1)):
            qx = cx + x_sign * 0.25 * (pd_width + gap)
            volumes["photodiode"].append(
                add_box(
                    gmsh,
                    f"pd_{side}_{ix}_{iz}",
                    qx - 0.5 * half,
                    pd_y,
                    cz - 0.5 * pd_width,
                    half,
                    pd_height,
                    pd_width,
                )
            )
        return

    if split_mode == "dual_z":
        half = 0.5 * (pd_width - gap)
        for side, z_sign in (("top", 1), ("bottom", -1)):
            qz = cz + z_sign * 0.25 * (pd_width + gap)
            volumes["photodiode"].append(
                add_box(
                    gmsh,
                    f"pd_{side}_{ix}_{iz}",
                    cx - 0.5 * pd_width,
                    pd_y,
                    qz - 0.5 * half,
                    pd_width,
                    pd_height,
                    half,
                )
            )
        return

    volumes["photodiode"].append(
        add_box(
            gmsh,
            f"pd_{ix}_{iz}",
            cx - 0.5 * pd_width,
            pd_y,
            cz - 0.5 * pd_width,
            pd_width,
            pd_height,
            pd_width,
        )
    )


def add_shield_volumes(gmsh: Any, volumes: dict[str, list[int]], spec: TemplateSpec, total_x: float, total_z: float) -> None:
    if spec.shield_mode == "off":
        return
    shield_y = spec.passivation_thickness_um + 0.02
    shield_thickness = 0.06
    mode = spec.shield_mode
    if mode in {"pdaf_left", "pdaf_right", "pdaf_pair"}:
        aperture = min(max(spec.pitch_um * 0.55, 0.3), total_x * 0.55)
        left_width = max(0.5 * (total_x - aperture), 0.02)
        right_width = left_width
        if mode in {"pdaf_right", "pdaf_pair"}:
            volumes["shield"].append(
                add_box(
                    gmsh,
                    "pdaf_shield_left_blocker",
                    -0.5 * total_x,
                    shield_y,
                    -0.5 * total_z,
                    left_width,
                    shield_thickness,
                    total_z,
                )
            )
        if mode in {"pdaf_left", "pdaf_pair"}:
            volumes["shield"].append(
                add_box(
                    gmsh,
                    "pdaf_shield_right_blocker",
                    0.5 * total_x - right_width,
                    shield_y,
                    -0.5 * total_z,
                    right_width,
                    shield_thickness,
                    total_z,
                )
            )
        return

    aperture = max(spec.pitch_um * 0.9, 0.3)
    plate = add_box(gmsh, "pdaf_shield_plate", -0.5 * total_x, shield_y, -0.5 * total_z, total_x, shield_thickness, total_z)
    aperture_box = gmsh.model.occ.addBox(-0.5 * aperture, shield_y - 0.01, -0.5 * aperture, aperture, shield_thickness + 0.02, aperture)
    cut, _ = gmsh.model.occ.cut([(3, plate)], [(3, aperture_box)], removeObject=True, removeTool=True)
    volumes["shield"].extend(tag for dim, tag in cut if dim == 3)


def add_template_geometry(gmsh: Any, spec: TemplateSpec) -> dict[str, list[int]]:
    total_x = spec.nx * spec.pitch_um
    total_z = spec.nz * spec.pitch_um
    stack_top = spec.passivation_thickness_um + spec.cfa_thickness_um
    volumes: dict[str, list[int]] = {
        "silicon": [],
        "passivation": [],
        "cfa": [],
        "ocl": [],
        "dti": [],
        "photodiode": [],
        "shield": [],
    }

    volumes["silicon"].append(add_box(gmsh, "silicon", -0.5 * total_x, -spec.si_thickness_um, -0.5 * total_z, total_x, spec.si_thickness_um, total_z))
    volumes["passivation"].append(add_box(gmsh, "passivation", -0.5 * total_x, 0.0, -0.5 * total_z, total_x, spec.passivation_thickness_um, total_z))

    tile_width = max(spec.pitch_um - spec.cfa_gap_um, spec.pitch_um * 0.25)
    for iz in range(spec.nz):
        for ix in range(spec.nx):
            cx, cz = cell_center(spec, ix, iz)
            color = cfa_color(spec.cfa_pattern, ix, iz)
            volumes["cfa"].append(
                add_box(
                    gmsh,
                    f"cfa_{color}_{ix}_{iz}",
                    cx - 0.5 * tile_width,
                    spec.passivation_thickness_um,
                    cz - 0.5 * tile_width,
                    tile_width,
                    spec.cfa_thickness_um,
                    tile_width,
                )
            )

            add_photodiode_volumes(gmsh, volumes, spec, cx, cz, ix, iz)

    dti_depth = min(spec.dti_depth_um, spec.si_thickness_um)
    for ix in range(1, spec.nx):
        x = ix * spec.pitch_um - 0.5 * total_x - 0.5 * spec.dti_width_um
        volumes["dti"].append(add_box(gmsh, f"dti_x_{ix}", x, -dti_depth, -0.5 * total_z, spec.dti_width_um, dti_depth, total_z))
    for iz in range(1, spec.nz):
        z = iz * spec.pitch_um - 0.5 * total_z - 0.5 * spec.dti_width_um
        volumes["dti"].append(add_box(gmsh, f"dti_z_{iz}", -0.5 * total_x, -dti_depth, z, total_x, dti_depth, spec.dti_width_um))

    add_shield_volumes(gmsh, volumes, spec, total_x, total_z)

    for block in spec.ocl_blocks:
        cx, cz = block_center(spec, block)
        width = max(block.sx * spec.pitch_um - spec.lens_edge_gap_um, spec.pitch_um * 0.25)
        depth = max(block.sz * spec.pitch_um - spec.lens_edge_gap_um, spec.pitch_um * 0.25)
        volumes["ocl"].extend(
            add_spherical_cap(
                gmsh,
                f"ocl_{block.lens_id}",
                cx,
                stack_top,
                cz,
                width,
                depth,
                block_lens_height(spec, block),
            )
        )

    gmsh.model.occ.synchronize()
    for role, tags in volumes.items():
        if tags:
            group = gmsh.model.addPhysicalGroup(3, tags)
            gmsh.model.setPhysicalName(3, group, role)
    return volumes


def write_preview_svg(spec: TemplateSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width_px = 820
    height_px = 560
    margin = 48
    total_x = spec.nx * spec.pitch_um
    total_z = spec.nz * spec.pitch_um
    scale = min((width_px - 2 * margin) / total_x, (height_px - 2 * margin) / total_z)

    def map_point(x: float, z: float) -> tuple[float, float]:
        return margin + (x + 0.5 * total_x) * scale, margin + (0.5 * total_z - z) * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">',
        '<rect width="100%" height="100%" rx="10" fill="#07131f"/>',
        f'<text x="28" y="34" fill="#e2e8f0" font-family="Inter, Arial" font-size="18" font-weight="700">{spec.label}</text>',
        f'<text x="28" y="56" fill="#94a3b8" font-family="Inter, Arial" font-size="12">{spec.nx} x {spec.nz} pixels, {spec.cfa_pattern}, split={spec.split_mode}</text>',
    ]
    color_fill = {"red": "#ef4444", "green": "#22c55e", "blue": "#3b82f6"}
    tile = spec.pitch_um - spec.cfa_gap_um
    for iz in range(spec.nz):
        for ix in range(spec.nx):
            cx, cz = cell_center(spec, ix, iz)
            x0, y0 = map_point(cx - 0.5 * tile, cz + 0.5 * tile)
            x1, y1 = map_point(cx + 0.5 * tile, cz - 0.5 * tile)
            color = color_fill[cfa_color(spec.cfa_pattern, ix, iz)]
            lines.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" rx="3" fill="{color}" opacity="0.62" stroke="#dbeafe" stroke-width="0.8"/>')
    for block in spec.ocl_blocks:
        cx, cz = block_center(spec, block)
        w = block.sx * spec.pitch_um - spec.lens_edge_gap_um
        h = block.sz * spec.pitch_um - spec.lens_edge_gap_um
        x0, y0 = map_point(cx - 0.5 * w, cz + 0.5 * h)
        x1, y1 = map_point(cx + 0.5 * w, cz - 0.5 * h)
        lines.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" rx="14" fill="none" stroke="#38bdf8" stroke-width="3"/>')
        lines.append(f'<text x="{0.5 * (x0 + x1):.2f}" y="{0.5 * (y0 + y1):.2f}" text-anchor="middle" fill="#e0f2fe" font-family="Inter, Arial" font-size="11" font-weight="700">{block.lens_id}</text>')
    lines.append('<text x="28" y="532" fill="#94a3b8" font-family="Inter, Arial" font-size="12">Footprint preview from parametric CAD source; open model.step/model.brep in FreeCAD for 3D review.</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def template_specs() -> dict[str, TemplateSpec]:
    boundary_pitch = 1.4
    boundary_lens_height = 0.657
    mixed_edge_slope = spherical_edge_slope(3.0 * boundary_pitch, boundary_lens_height)
    mixed_height_1x1 = spherical_height_for_edge_slope(1.0 * boundary_pitch, mixed_edge_slope)
    mixed_height_2x2 = spherical_height_for_edge_slope(2.0 * boundary_pitch, mixed_edge_slope)
    return {
        "bayer_1x1_3x3": TemplateSpec(
            template_id="bayer_1x1_3x3",
            label="Bayer 1x1 OCL, 3x3 neighborhood",
            nx=3,
            nz=3,
            cfa_pattern="bayer",
            ocl_blocks=tuple(OclBlock(f"ocl_{ix}_{iz}", ix, iz, 1, 1) for iz in range(3) for ix in range(3)),
            notes=("Baseline 1x1 image-pixel neighborhood for crosstalk and CRA anchors.",),
        ),
        "quad_2x2_ocl": TemplateSpec(
            template_id="quad_2x2_ocl",
            label="Quad Bayer 2x2 OCL supercell",
            nx=4,
            nz=4,
            cfa_pattern="quad",
            ocl_blocks=tuple(OclBlock(f"quad_{ix}_{iz}", ix, iz, 2, 2) for iz in (0, 2) for ix in (0, 2)),
            notes=("Common 2x2 same-color binning/OCL template.",),
        ),
        "quad_2x2_ocl_3x3_neighborhood": TemplateSpec(
            template_id="quad_2x2_ocl_3x3_neighborhood",
            label="Quad Bayer 2x2 OCL, 3x3 group neighborhood",
            nx=6,
            nz=6,
            cfa_pattern="quad",
            ocl_blocks=tuple(OclBlock(f"quad_{ix}_{iz}", ix, iz, 2, 2) for iz in (0, 2, 4) for ix in (0, 2, 4)),
            notes=(
                "Minimum 3x3 OCL-group neighborhood for central 2x2 OCL crosstalk kernel checks.",
                "Use a larger 5x5 OCL-group domain for long-range or high-CRA leakage truncation studies.",
            ),
        ),
        "quad_2x2_ocl_5x5_crosstalk": TemplateSpec(
            template_id="quad_2x2_ocl_5x5_crosstalk",
            label="Quad Bayer 2x2 OCL, 5x5 crosstalk domain",
            nx=10,
            nz=10,
            cfa_pattern="quad",
            ocl_blocks=tuple(OclBlock(f"quad_{ix}_{iz}", ix, iz, 2, 2) for iz in (0, 2, 4, 6, 8) for ix in (0, 2, 4, 6, 8)),
            lens_edge_gap_um=0.0,
            cfa_gap_um=0.0,
            notes=(
                "Practical 5x5 OCL-group domain for crosstalk kernel truncation checks around the central 2x2 OCL group.",
                "Boundary review variant uses gapless OCL/CFA footprints so C0 continuity can be evaluated directly.",
                "Use this larger domain for high-CRA or long-range leakage studies before camera-system kernel export.",
            ),
        ),
        "nona_3x3_ocl": TemplateSpec(
            template_id="nona_3x3_ocl",
            label="Nona 3x3 OCL supercell",
            nx=6,
            nz=6,
            cfa_pattern="nona",
            ocl_blocks=tuple(OclBlock(f"nona_{ix}_{iz}", ix, iz, 3, 3) for iz in (0, 3) for ix in (0, 3)),
            lens_edge_gap_um=0.0,
            cfa_gap_um=0.0,
            notes=(
                "3x3 grouped-pixel template for binning uniformity and edge CRA checks.",
                "Boundary review variant uses gapless OCL/CFA footprints so C0 continuity can be evaluated directly.",
            ),
        ),
        "qpd_split_pd_2x2": TemplateSpec(
            template_id="qpd_split_pd_2x2",
            label="QPD 2x2 split photodiode pixel",
            nx=2,
            nz=2,
            cfa_pattern="uniform_green",
            ocl_blocks=(OclBlock("qpd_2x2_ocl", 0, 0, 2, 2),),
            split_mode="quad",
            shield_mode="pdaf_pair",
            notes=("QPD/split-PD CAD template with quadrant photodiode volumes.",),
        ),
        "qpd_split_pd_no_shield_2x2": TemplateSpec(
            template_id="qpd_split_pd_no_shield_2x2",
            label="QPD 2x2 split photodiode, no shield control",
            nx=2,
            nz=2,
            cfa_pattern="uniform_green",
            ocl_blocks=(OclBlock("qpd_2x2_ocl", 0, 0, 2, 2),),
            split_mode="quad",
            shield_mode="off",
            notes=("QPD control template for separating split-PD geometry from PDAF metal-shield penalty.",),
        ),
        "dual_pd_x_1x1": TemplateSpec(
            template_id="dual_pd_x_1x1",
            label="Dual-PD X split 1x1 pixel",
            nx=1,
            nz=1,
            cfa_pattern="uniform_green",
            ocl_blocks=(OclBlock("dual_x_ocl", 0, 0, 1, 1),),
            split_mode="dual-x",
            shield_mode="off",
            notes=("Left/right split photodiode template for x-axis phase response and CRA-x checks.",),
        ),
        "dual_pd_z_1x1": TemplateSpec(
            template_id="dual_pd_z_1x1",
            label="Dual-PD Z split 1x1 pixel",
            nx=1,
            nz=1,
            cfa_pattern="uniform_green",
            ocl_blocks=(OclBlock("dual_z_ocl", 0, 0, 1, 1),),
            split_mode="dual-z",
            shield_mode="off",
            notes=("Top/bottom split photodiode template for z-axis phase response and CRA-z checks.",),
        ),
        "pdaf_dual_x_shield_pair": TemplateSpec(
            template_id="pdaf_dual_x_shield_pair",
            label="Dual-PD X PDAF shield pair",
            nx=2,
            nz=1,
            cfa_pattern="uniform_green",
            ocl_blocks=(OclBlock("pdaf_left_ocl", 0, 0, 1, 1), OclBlock("pdaf_right_ocl", 1, 0, 1, 1)),
            split_mode="dual-x",
            shield_mode="pdaf_pair",
            notes=("Paired left/right PDAF shield template for comparing phase signal and image-QE penalty.",),
        ),
        "mixed_1x1_2x2_3x3_boundary": TemplateSpec(
            template_id="mixed_1x1_2x2_3x3_boundary",
            label="Mixed 1x1 / 2x2 / 3x3 OCL boundary",
            nx=5,
            nz=3,
            cfa_pattern="nona",
            ocl_blocks=(
                OclBlock("nona_left", 0, 0, 3, 3, boundary_lens_height),
                OclBlock("quad_right", 3, 0, 2, 2, mixed_height_2x2),
                OclBlock("bayer_r0", 3, 2, 1, 1, mixed_height_1x1),
                OclBlock("bayer_r1", 4, 2, 1, 1, mixed_height_1x1),
            ),
            lens_edge_gap_um=0.0,
            cfa_gap_um=0.0,
            notes=(
                "Layout transition template for mixed-OCL boundary leakage and remosaic risk.",
                "Boundary review variant uses gapless OCL/CFA footprints and slope-matched spherical caps across 1x1/2x2/3x3 OCL transitions.",
            ),
        ),
    }


def write_template(spec: TemplateSpec, output_dir: Path, *, mesh: bool) -> dict[str, Any]:
    gmsh = load_gmsh()
    template_dir = output_dir / spec.template_id
    template_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.model.add(spec.template_id)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        volumes = add_template_geometry(gmsh, spec)
        step_path = template_dir / "model.step"
        brep_path = template_dir / "model.brep"
        gmsh.write(str(brep_path))
        gmsh.write(str(step_path))
        mesh_path = template_dir / "model.msh"
        if mesh:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.10)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.35)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(mesh_path))
    finally:
        gmsh.finalize()

    footprint_path = template_dir / "geometry_import.json"
    footprint_path.write_text(json.dumps(build_footprint_payload(spec), indent=2), encoding="utf-8")
    params_path = template_dir / "template_parameters.json"
    params_path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    preview_path = template_dir / "footprint_preview.svg"
    write_preview_svg(spec, preview_path)
    ledger_path = template_dir / "assumption_ledger.json"
    ledger_path.write_text(json.dumps(assumption_ledger(spec), indent=2), encoding="utf-8")

    return {
        "template_id": spec.template_id,
        "label": spec.label,
        "status": "generated",
        "source_truth_level": "parametric_template_not_measured",
        "freecad_openable": True,
        "files": {
            "step": str(step_path),
            "brep": str(brep_path),
            "mesh": str(mesh_path) if mesh and mesh_path.exists() else None,
            "geometry_import": str(footprint_path),
            "parameters": str(params_path),
            "assumption_ledger": str(ledger_path),
            "footprint_preview": str(preview_path),
        },
        "counts": {role: len(tags) for role, tags in volumes.items()},
        "notes": list(spec.notes)
        + [
            "Open model.step or model.brep in FreeCAD for 3D review.",
            "This template centralizes assumptions but does not replace measured process geometry.",
        ],
    }


def mesh_physical_names(mesh_path: Path) -> list[str]:
    if not mesh_path.exists():
        return []
    lines = mesh_path.read_text(errors="ignore").splitlines()
    if "$PhysicalNames" not in lines:
        return []
    index = lines.index("$PhysicalNames")
    try:
        count = int(lines[index + 1])
    except (ValueError, IndexError):
        return []
    names = []
    for line in lines[index + 2 : index + 2 + count]:
        if '"' not in line:
            continue
        names.append(line.split('"', 2)[1])
    return names


def fdtd_smoke_summary(output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "qpd_split_pd_2x2" / "fdtd_smoke" / "camera_lut_summary.csv"
    if not summary_path.exists():
        return None
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"path": str(summary_path), "row_count": 0}
    first = rows[0]
    return {
        "path": str(summary_path),
        "row_count": len(rows),
        "total_response": float(first["total_response"]) if first.get("total_response") else None,
        "grid_resolution_gate_pass": first.get("grid_resolution_gate_pass"),
        "expected_gate_status": "CHECK for low-resolution smoke",
    }


def polygon_bbox(points: list[Any]) -> dict[str, float | None]:
    clean: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                clean.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if not clean:
        return {
            "xmin": None,
            "xmax": None,
            "zmin": None,
            "zmax": None,
            "xlen": None,
            "zlen": None,
        }
    xs = [point[0] for point in clean]
    zs = [point[1] for point in clean]
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "zmin": min(zs),
        "zmax": max(zs),
        "xlen": max(xs) - min(xs),
        "zlen": max(zs) - min(zs),
    }


def close_um(actual: Any, expected: float, tolerance_um: float = 1.0e-6) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance_um
    except (TypeError, ValueError):
        return False


def rule(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def validate_template_design_rules(record: dict[str, Any], payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    template_id = str(record.get("template_id") or "")
    counts = record.get("counts", {}) if isinstance(record.get("counts"), dict) else {}
    nx = int(parameters.get("nx") or 0)
    nz = int(parameters.get("nz") or 0)
    pitch_um = float(parameters.get("pitch_um") or 0.0)
    cfa_gap_um = float(parameters.get("cfa_gap_um") or 0.0)
    lens_edge_gap_um = float(parameters.get("lens_edge_gap_um") or 0.0)
    si_thickness_um = float(parameters.get("si_thickness_um") or 0.0)
    dti_depth_um = float(parameters.get("dti_depth_um") or 0.0)
    pd_min_um = float(parameters.get("pd_depth_min_um") or 0.0)
    pd_max_um = float(parameters.get("pd_depth_max_um") or 0.0)
    split_mode = str(parameters.get("split_mode") or "none")
    shield_mode = str(parameters.get("shield_mode") or "off")
    cfa_pattern = str(parameters.get("cfa_pattern") or "")
    total_pixels = nx * nz

    payload_template = payload.get("cad_template", {}) if isinstance(payload.get("cad_template"), dict) else {}
    rules.append(
        rule(
            "schema_and_template_id",
            payload.get("schema") == "pixel_geometry_import_v1"
            and payload.get("units") == "um"
            and payload_template.get("template_id") == template_id
            and parameters.get("template_id") == template_id,
            {
                "payload_schema": payload.get("schema"),
                "payload_units": payload.get("units"),
                "payload_template_id": payload_template.get("template_id"),
                "parameter_template_id": parameters.get("template_id"),
                "record_template_id": template_id,
            },
        )
    )

    positive_fields = [
        "nx",
        "nz",
        "pitch_um",
        "si_thickness_um",
        "passivation_thickness_um",
        "cfa_thickness_um",
        "lens_height_um",
        "dti_width_um",
        "dti_depth_um",
        "pd_depth_min_um",
        "pd_depth_max_um",
        "pd_margin_um",
    ]
    invalid_positive = [
        field
        for field in positive_fields
        if (float(parameters.get(field) or 0.0) <= 0.0)
    ]
    dimension_passed = (
        not invalid_positive
        and cfa_gap_um >= 0.0
        and lens_edge_gap_um >= 0.0
        and cfa_gap_um < pitch_um
        and lens_edge_gap_um < max(nx, nz) * pitch_um
        and 0.0 < pd_min_um < pd_max_um <= si_thickness_um
        and 0.0 < dti_depth_um <= si_thickness_um
    )
    rules.append(
        rule(
            "dimension_ranges",
            dimension_passed,
            {
                "invalid_positive_fields": invalid_positive,
                "cfa_gap_um": cfa_gap_um,
                "lens_edge_gap_um": lens_edge_gap_um,
                "pd_depth_min_um": pd_min_um,
                "pd_depth_max_um": pd_max_um,
                "si_thickness_um": si_thickness_um,
                "dti_depth_um": dti_depth_um,
            },
        )
    )

    cfa_payload = payload.get("cfa_polygons", {}) if isinstance(payload.get("cfa_polygons"), dict) else {}
    cfa_cells = cfa_payload.get("cells", []) if isinstance(cfa_payload.get("cells"), list) else []
    cfa_positions: dict[tuple[int, int], dict[str, Any]] = {}
    duplicate_cfa: list[tuple[int, int]] = []
    color_mismatches: list[dict[str, Any]] = []
    invalid_cfa_bbox: list[dict[str, Any]] = []
    expected_cfa_width = max(pitch_um - cfa_gap_um, pitch_um * 0.25)
    for cell in cfa_cells:
        if not isinstance(cell, dict):
            continue
        ix = int(cell.get("ix", -1))
        iz = int(cell.get("iz", -1))
        key = (ix, iz)
        if key in cfa_positions:
            duplicate_cfa.append(key)
        cfa_positions[key] = cell
        expected_color = cfa_color(cfa_pattern, ix, iz) if 0 <= ix < nx and 0 <= iz < nz else None
        if expected_color and cell.get("color") != expected_color:
            color_mismatches.append({"ix": ix, "iz": iz, "actual": cell.get("color"), "expected": expected_color})
        bbox = polygon_bbox(cell.get("points", []))
        if not (
            close_um(bbox["xlen"], expected_cfa_width)
            and close_um(bbox["zlen"], expected_cfa_width)
            and close_um(bbox["xmin"], -0.5 * expected_cfa_width)
            and close_um(bbox["zmin"], -0.5 * expected_cfa_width)
        ):
            invalid_cfa_bbox.append({"ix": ix, "iz": iz, "bbox": bbox, "expected_width_um": expected_cfa_width})
    expected_positions = {(ix, iz) for iz in range(nz) for ix in range(nx)}
    cfa_rule_passed = (
        len(cfa_cells) == total_pixels
        and set(cfa_positions) == expected_positions
        and not duplicate_cfa
        and not color_mismatches
        and not invalid_cfa_bbox
    )
    rules.append(
        rule(
            "cfa_grid_and_pattern",
            cfa_rule_passed,
            {
                "cell_count": len(cfa_cells),
                "expected_cell_count": total_pixels,
                "missing_positions": sorted(expected_positions.difference(cfa_positions)),
                "extra_positions": sorted(set(cfa_positions).difference(expected_positions)),
                "duplicates": duplicate_cfa,
                "color_mismatches": color_mismatches,
                "invalid_bbox_count": len(invalid_cfa_bbox),
                "invalid_bbox_examples": invalid_cfa_bbox[:5],
            },
        )
    )

    ocl_blocks = parameters.get("ocl_blocks", []) if isinstance(parameters.get("ocl_blocks"), list) else []
    ocl_polygons = payload.get("ocl_polygons", {}) if isinstance(payload.get("ocl_polygons"), dict) else {}
    covered: dict[tuple[int, int], str] = {}
    overlap: list[dict[str, Any]] = []
    block_errors: list[dict[str, Any]] = []
    invalid_ocl_bbox: list[dict[str, Any]] = []
    for block in ocl_blocks:
        if not isinstance(block, dict):
            block_errors.append({"block": block, "error": "not_object"})
            continue
        lens_id = str(block.get("lens_id") or "")
        ix = int(block.get("ix", -1))
        iz = int(block.get("iz", -1))
        sx = int(block.get("sx", 0))
        sz = int(block.get("sz", 0))
        in_bounds = ix >= 0 and iz >= 0 and sx > 0 and sz > 0 and ix + sx <= nx and iz + sz <= nz
        if not in_bounds:
            block_errors.append({"lens_id": lens_id, "ix": ix, "iz": iz, "sx": sx, "sz": sz, "error": "out_of_bounds"})
            continue
        for cell in [(cx, cz) for cz in range(iz, iz + sz) for cx in range(ix, ix + sx)]:
            if cell in covered:
                overlap.append({"cell": cell, "first": covered[cell], "second": lens_id})
            covered[cell] = lens_id
        expected_width = max(sx * pitch_um - lens_edge_gap_um, pitch_um * 0.25)
        expected_height = max(sz * pitch_um - lens_edge_gap_um, pitch_um * 0.25)
        bbox = polygon_bbox(ocl_polygons.get(lens_id, []))
        if not (
            close_um(bbox["xlen"], expected_width)
            and close_um(bbox["zlen"], expected_height)
            and close_um(bbox["xmin"], -0.5 * expected_width)
            and close_um(bbox["zmin"], -0.5 * expected_height)
        ):
            invalid_ocl_bbox.append(
                {
                    "lens_id": lens_id,
                    "bbox": bbox,
                    "expected_width_um": expected_width,
                    "expected_height_um": expected_height,
                }
            )
    expected_lens_ids = {str(block.get("lens_id")) for block in ocl_blocks if isinstance(block, dict)}
    ocl_rule_passed = (
        set(ocl_polygons) == expected_lens_ids
        and set(covered) == expected_positions
        and not overlap
        and not block_errors
        and not invalid_ocl_bbox
    )
    rules.append(
        rule(
            "ocl_block_coverage",
            ocl_rule_passed,
            {
                "lens_count": len(ocl_polygons),
                "expected_lens_count": len(expected_lens_ids),
                "missing_lens_ids": sorted(expected_lens_ids.difference(ocl_polygons)),
                "extra_lens_ids": sorted(set(ocl_polygons).difference(expected_lens_ids)),
                "missing_cells": sorted(expected_positions.difference(covered)),
                "overlap": overlap,
                "block_errors": block_errors,
                "invalid_bbox_count": len(invalid_ocl_bbox),
                "invalid_bbox_examples": invalid_ocl_bbox[:5],
            },
        )
    )

    expected_counts = {
        "cfa": total_pixels,
        "ocl": len(ocl_blocks),
        "dti": max(nx - 1, 0) + max(nz - 1, 0),
        "photodiode": total_pixels * pd_volume_multiplier(split_mode),
        "shield": expected_shield_volume_count(shield_mode),
    }
    count_mismatches = {
        key: {"actual": counts.get(key), "expected": expected}
        for key, expected in expected_counts.items()
        if counts.get(key) != expected
    }
    rules.append(
        rule(
            "cad_volume_counts",
            not count_mismatches,
            {
                "counts": counts,
                "expected_counts": expected_counts,
                "mismatches": count_mismatches,
                "split_mode": split_mode,
                "shield_mode": shield_mode,
            },
        )
    )

    fail_count = sum(1 for item in rules if item["status"] != "PASS")
    return {
        "schema": "cad_template_design_rule_validation_v1",
        "status": "PASS" if fail_count == 0 else "FAIL",
        "fail_count": fail_count,
        "rule_count": len(rules),
        "rules": rules,
    }


def validate_library(records: list[dict[str, Any]], output_dir: Path, *, mesh_expected: bool) -> dict[str, Any]:
    template_reports = []
    status = "PASS"
    for record in records:
        files = record["files"]
        geometry_path = Path(files["geometry_import"])
        payload = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.exists() else {}
        parameters_path = Path(files["parameters"]) if files.get("parameters") else None
        parameters = json.loads(parameters_path.read_text(encoding="utf-8")) if parameters_path and parameters_path.exists() else {}
        cfa_cells = payload.get("cfa_polygons", {}).get("cells", []) if isinstance(payload.get("cfa_polygons"), dict) else []
        mesh_path = Path(files["mesh"]) if files.get("mesh") else None
        record_mesh_expected = mesh_expected and mesh_path is not None
        physical_names = mesh_physical_names(mesh_path) if mesh_path else []
        counts = record["counts"] if isinstance(record.get("counts"), dict) else {}
        required_groups = {"silicon", "passivation", "cfa", "ocl", "photodiode"}
        if int(counts.get("dti") or 0) > 0:
            required_groups.add("dti")
        if int(counts.get("shield") or 0) > 0:
            required_groups.add("shield")
        missing_groups = sorted(required_groups.difference(physical_names)) if record_mesh_expected else []
        design_rule_validation = validate_template_design_rules(record, payload, parameters)
        report = {
            "template_id": record["template_id"],
            "step_exists": Path(files["step"]).exists(),
            "brep_exists": Path(files["brep"]).exists(),
            "mesh_exists": bool(mesh_path and mesh_path.exists()),
            "mesh_path": str(mesh_path) if mesh_path else None,
            "mesh_size_bytes": mesh_path.stat().st_size if mesh_path and mesh_path.exists() else None,
            "mesh_physical_groups": physical_names,
            "mesh_required_groups_missing": missing_groups,
            "geometry_import_exists": geometry_path.exists(),
            "assumption_ledger_exists": bool(files.get("assumption_ledger") and Path(files["assumption_ledger"]).exists()),
            "preview_exists": Path(files["footprint_preview"]).exists(),
            "counts": record["counts"],
            "ocl_polygon_count": len(payload.get("ocl_polygons", {})) if isinstance(payload.get("ocl_polygons"), dict) else 0,
            "cfa_cell_count": len(cfa_cells),
            "cfa_colors": sorted({str(cell.get("color")) for cell in cfa_cells if isinstance(cell, dict)}),
            "design_rule_status": design_rule_validation["status"],
            "design_rule_fail_count": design_rule_validation["fail_count"],
            "design_rule_validation": design_rule_validation,
        }
        if not all(
            [
                report["step_exists"],
                report["brep_exists"],
                report["geometry_import_exists"],
                report["assumption_ledger_exists"],
                report["preview_exists"],
                report["ocl_polygon_count"] > 0,
                report["cfa_cell_count"] > 0,
                report["design_rule_status"] == "PASS",
                (not record_mesh_expected or (report["mesh_exists"] and not missing_groups)),
            ]
        ):
            status = "FAIL"
        template_reports.append(report)
    smoke = fdtd_smoke_summary(output_dir)
    return {
        "schema": "pixel_cad_template_library_validation_v1",
        "status": status,
        "template_count": len(template_reports),
        "mesh_expected": mesh_expected,
        "templates": template_reports,
        "fdtd_smoke": smoke,
        "notes": [
            "STEP/BREP are FreeCAD-openable CAD artifacts generated with Gmsh/OpenCASCADE.",
            "model.msh files are coarse 3D CAD review meshes with physical volume groups, not calibrated DEVSIM electrical meshes.",
            "geometry_import.json is the FDTD footprint source used by the Workbench CAD template solver path.",
            "Smoke grid gate can be CHECK/FAIL because low-resolution runs are wiring checks.",
        ],
    }


def append_or_replace_records(existing: list[Any], generated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [item for item in existing if isinstance(item, dict)]
    by_id = {record.get("template_id"): index for index, record in enumerate(records)}
    for record in generated:
        template_id = record.get("template_id")
        if template_id in by_id:
            records[by_id[template_id]] = record
        else:
            by_id[template_id] = len(records)
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "pixel_cad_template_library_reference")
    parser.add_argument("--template", action="append", default=[], help="Template id to generate. Repeatable. Defaults to all.")
    parser.add_argument("--mesh", action="store_true", help="Also write a coarse 3D MSH preview mesh.")
    parser.add_argument("--append", action="store_true", help="Merge generated templates into the existing manifest instead of replacing it.")
    parser.add_argument("--validate-only", action="store_true", help="Refresh validation report from the existing manifest without regenerating CAD artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        manifest_path = args.output_dir / "template_library_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("templates", [])
        if not isinstance(records, list):
            raise SystemExit("Manifest templates field is invalid")
        mesh_expected = args.mesh or any(
            isinstance(record, dict)
            and isinstance(record.get("files"), dict)
            and bool(record["files"].get("mesh"))
            for record in records
        )
        validation = validate_library([record for record in records if isinstance(record, dict)], args.output_dir, mesh_expected=mesh_expected)
        validation_path = args.output_dir / "cad_template_validation_report.json"
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps({"validation": str(validation_path), **validation}, indent=2))
        return

    specs = template_specs()
    selected = args.template or sorted(specs)
    missing = [item for item in selected if item not in specs]
    if missing:
        raise SystemExit(f"Unknown template id(s): {', '.join(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_records = [write_template(specs[item], args.output_dir, mesh=args.mesh) for item in selected]
    manifest_path = args.output_dir / "template_library_manifest.json"
    if args.append and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = append_or_replace_records(previous_manifest.get("templates", []), generated_records)
    else:
        records = generated_records
    manifest = {
        "schema": "pixel_cad_template_library_manifest_v1",
        "output_dir": str(args.output_dir),
        "template_count": len(records),
        "generated_with": "Gmsh/OpenCASCADE",
        "freecad_role": "Open generated STEP/BREP files for 3D review; FreeCAD is not required for headless generation.",
        "mask_role": "Use geometry_import.json or downstream GDS export for solver footprints.",
        "mesh_role": "Optional model.msh files are coarse 3D CAD review meshes with physical volume groups; they are not calibrated DEVSIM electrical meshes.",
        "accuracy_status": "parametric_templates_not_measured",
        "templates": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = validate_library(records, args.output_dir, mesh_expected=args.mesh)
    validation_path = args.output_dir / "cad_template_validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "validation_report": str(validation_path),
                "template_count": len(records),
                "status": validation["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI should produce useful local diagnostics.
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
