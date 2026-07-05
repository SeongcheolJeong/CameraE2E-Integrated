#!/usr/bin/env python3
"""Evaluate mixed-OCL boundary continuity, mesh quality, and response drivers.

This report is deliberately separate from the optical KPI sweep. The CRA sweep
proves that a topology was simulated; this script checks whether the CAD/FDTD
boundary representation is continuous enough to trust boundary-specific trends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_ROOT = ROOT / "runs" / "pixel_cad_template_library_reference"
DEFAULT_CRA_OUTPUT = ROOT / "runs" / "reference_sensor_cra_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_CRA_OUTPUT / "boundary_continuity"
DEFAULT_TEMPLATES = "mixed_1x1_2x2_3x3_boundary,nona_3x3_ocl,quad_2x2_ocl_5x5_crosstalk"
RESPONSE_METRICS = (
    "relative_qe_to_center",
    "neighbor_leakage_fraction",
    "response_centroid_shift_x_um",
    "focal_centroid_shift_x_um",
)


@dataclass(frozen=True)
class Box:
    item_id: str
    layer: str
    xmin: float
    xmax: float
    zmin: float
    zmax: float
    ix: int | None = None
    iz: int | None = None
    sx: int = 1
    sz: int = 1
    height_um: float | None = None

    @property
    def xlen(self) -> float:
        return self.xmax - self.xmin

    @property
    def zlen(self) -> float:
        return self.zmax - self.zmin


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def number(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def compact(value: Any, digits: int = 4) -> str:
    numeric = number(value)
    if numeric is None:
        return "" if value is None else str(value)
    if numeric != 0 and abs(numeric) < 0.001:
        return f"{numeric:.{digits}e}"
    return f"{numeric:.{digits}g}"


def cell_center(params: dict[str, Any], ix: int, iz: int) -> tuple[float, float]:
    pitch = float(params["pitch_um"])
    total_x = int(params["nx"]) * pitch
    total_z = int(params["nz"]) * pitch
    return (ix + 0.5) * pitch - 0.5 * total_x, (iz + 0.5) * pitch - 0.5 * total_z


def block_center(params: dict[str, Any], block: dict[str, Any]) -> tuple[float, float]:
    pitch = float(params["pitch_um"])
    total_x = int(params["nx"]) * pitch
    total_z = int(params["nz"]) * pitch
    return (float(block["ix"]) + 0.5 * float(block["sx"])) * pitch - 0.5 * total_x, (
        float(block["iz"]) + 0.5 * float(block["sz"])
    ) * pitch - 0.5 * total_z


def bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    zs = [float(point[1]) for point in points]
    return min(xs), max(xs), min(zs), max(zs)


def ocl_boxes(params: dict[str, Any], geometry: dict[str, Any]) -> list[Box]:
    polygons = geometry.get("ocl_polygons", {})
    blocks = params.get("ocl_blocks", [])
    output = []
    for block in blocks:
        lens_id = str(block["lens_id"])
        local = polygons.get(lens_id)
        if not local:
            continue
        cx, cz = block_center(params, block)
        points = [[cx + float(x), cz + float(z)] for x, z in local]
        xmin, xmax, zmin, zmax = bbox(points)
        output.append(
            Box(
                lens_id,
                "ocl",
                xmin,
                xmax,
                zmin,
                zmax,
                ix=int(block["ix"]),
                iz=int(block["iz"]),
                sx=int(block["sx"]),
                sz=int(block["sz"]),
                height_um=number(block.get("height_um")),
            )
        )
    return output


def cfa_boxes(params: dict[str, Any], geometry: dict[str, Any]) -> list[Box]:
    cells = geometry.get("cfa_polygons", {}).get("cells", [])
    output = []
    for cell in cells:
        ix = int(cell["ix"])
        iz = int(cell["iz"])
        cx, cz = cell_center(params, ix, iz)
        points = [[cx + float(x), cz + float(z)] for x, z in cell.get("points", [])]
        xmin, xmax, zmin, zmax = bbox(points)
        output.append(Box(str(cell.get("id")), "cfa", xmin, xmax, zmin, zmax, ix=ix, iz=iz))
    return output


def pd_boxes(params: dict[str, Any]) -> list[Box]:
    pitch = float(params["pitch_um"])
    pd_width = max(pitch - 2.0 * float(params["pd_margin_um"]), pitch * 0.25)
    output = []
    for iz in range(int(params["nz"])):
        for ix in range(int(params["nx"])):
            cx, cz = cell_center(params, ix, iz)
            output.append(
                Box(
                    f"pd_{ix}_{iz}",
                    "pd",
                    cx - 0.5 * pd_width,
                    cx + 0.5 * pd_width,
                    cz - 0.5 * pd_width,
                    cz + 0.5 * pd_width,
                    ix=ix,
                    iz=iz,
                )
            )
    return output


def dti_boxes(params: dict[str, Any]) -> list[Box]:
    pitch = float(params["pitch_um"])
    total_x = int(params["nx"]) * pitch
    total_z = int(params["nz"]) * pitch
    width = float(params["dti_width_um"])
    output = []
    for ix in range(1, int(params["nx"])):
        x = ix * pitch - 0.5 * total_x
        output.append(Box(f"dti_x_{ix}", "dti", x - 0.5 * width, x + 0.5 * width, -0.5 * total_z, 0.5 * total_z))
    for iz in range(1, int(params["nz"])):
        z = iz * pitch - 0.5 * total_z
        output.append(Box(f"dti_z_{iz}", "dti", -0.5 * total_x, 0.5 * total_x, z - 0.5 * width, z + 0.5 * width))
    return output


def adjacency_pairs(boxes: list[Box]) -> list[tuple[Box, Box, str]]:
    pairs = []
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            if first.ix is None or first.iz is None or second.ix is None or second.iz is None:
                continue
            if first.ix + first.sx == second.ix or second.ix + second.sx == first.ix:
                if max(first.iz, second.iz) < min(first.iz + first.sz, second.iz + second.sz):
                    left, right = (first, second) if first.xmin < second.xmin else (second, first)
                    pairs.append((left, right, "x"))
            if first.iz + first.sz == second.iz or second.iz + second.sz == first.iz:
                if max(first.ix, second.ix) < min(first.ix + first.sx, second.ix + second.sx):
                    bottom, top = (first, second) if first.zmin < second.zmin else (second, first)
                    pairs.append((bottom, top, "z"))
    return pairs


def signed_gap(first: Box, second: Box, axis: str) -> float:
    if axis == "x":
        return second.xmin - first.xmax
    return second.zmin - first.zmax


def rectangle_distance(first: Box, second: Box) -> tuple[float, float, float, float]:
    dx = max(second.xmin - first.xmax, first.xmin - second.xmax, 0.0)
    dz = max(second.zmin - first.zmax, first.zmin - second.zmax, 0.0)
    overlap_x = max(0.0, min(first.xmax, second.xmax) - max(first.xmin, second.xmin))
    overlap_z = max(0.0, min(first.zmax, second.zmax) - max(first.zmin, second.zmin))
    return math.hypot(dx, dz), dx, dz, overlap_x * overlap_z


def lens_surface_metrics(params: dict[str, Any], box: Box, axis: str) -> dict[str, float]:
    height = box.height_um if box.height_um is not None else float(params["lens_height_um"])
    aperture_radius = 0.5 * min(box.xlen, box.zlen)
    radius = (aperture_radius * aperture_radius + height * height) / max(2.0 * height, 1.0e-9)
    edge_r = 0.5 * (box.xlen if axis == "x" else box.zlen)
    edge_r = min(edge_r, aperture_radius * 0.999999)
    root = max(radius * radius - edge_r * edge_r, 1.0e-18)
    slope = abs(edge_r / math.sqrt(root))
    sphere_center_y = height - radius
    edge_height = sphere_center_y + math.sqrt(root)
    return {"edge_slope_abs": slope, "edge_height_um": edge_height, "sphere_radius_um": radius}


def c0_c1_ocl_report(params: dict[str, Any], boxes: list[Box], tolerance_um: float, slope_tolerance: float) -> dict[str, Any]:
    expected_gap = float(params["lens_edge_gap_um"])
    rows = []
    for first, second, axis in adjacency_pairs(boxes):
        gap = signed_gap(first, second, axis)
        first_surface = lens_surface_metrics(params, first, axis)
        second_surface = lens_surface_metrics(params, second, axis)
        design_error = abs(gap - expected_gap)
        slope_delta = abs(first_surface["edge_slope_abs"] - second_surface["edge_slope_abs"])
        height_delta = abs(first_surface["edge_height_um"] - second_surface["edge_height_um"])
        rows.append(
            {
                "first": first.item_id,
                "second": second.item_id,
                "axis": axis,
                "signed_gap_um": gap,
                "physical_overlap_um": max(0.0, -gap),
                "expected_gap_um": expected_gap,
                "c0_design_error_um": design_error,
                "c0_physical_continuity_status": "PASS" if abs(gap) <= tolerance_um else "CHECK",
                "c0_design_status": "PASS" if design_error <= tolerance_um and gap >= -tolerance_um else "FAIL",
                "edge_height_delta_um": height_delta,
                "edge_slope_first": first_surface["edge_slope_abs"],
                "edge_slope_second": second_surface["edge_slope_abs"],
                "c1_slope_delta": slope_delta,
                "c1_status": "PASS" if abs(gap) <= tolerance_um and slope_delta <= slope_tolerance else "CHECK",
            }
        )
    return {
        "adjacency_count": len(rows),
        "max_physical_gap_um": max((row["signed_gap_um"] for row in rows), default=None),
        "max_overlap_um": max((row["physical_overlap_um"] for row in rows), default=None),
        "max_c0_design_error_um": max((row["c0_design_error_um"] for row in rows), default=None),
        "max_edge_height_delta_um": max((row["edge_height_delta_um"] for row in rows), default=None),
        "max_c1_slope_delta": max((row["c1_slope_delta"] for row in rows), default=None),
        "rows": rows,
    }


def layer_gap_report(layer: str, boxes: list[Box], expected_gap_um: float, tolerance_um: float) -> dict[str, Any]:
    rows = []
    for first, second, axis in adjacency_pairs(boxes):
        gap = signed_gap(first, second, axis)
        rows.append(
            {
                "layer": layer,
                "first": first.item_id,
                "second": second.item_id,
                "axis": axis,
                "signed_gap_um": gap,
                "overlap_um": max(0.0, -gap),
                "expected_gap_um": expected_gap_um,
                "design_error_um": abs(gap - expected_gap_um),
                "status": "PASS" if abs(gap - expected_gap_um) <= tolerance_um and gap >= -tolerance_um else "FAIL",
            }
        )
    return {
        "layer": layer,
        "pair_count": len(rows),
        "min_signed_gap_um": min((row["signed_gap_um"] for row in rows), default=None),
        "max_overlap_um": max((row["overlap_um"] for row in rows), default=None),
        "max_design_error_um": max((row["design_error_um"] for row in rows), default=None),
        "rows": rows,
    }


def pd_dti_clearance_report(pd: list[Box], dti: list[Box]) -> dict[str, Any]:
    rows = []
    for pd_box in pd:
        candidates = []
        for dti_box in dti:
            dist, dx, dz, overlap_area = rectangle_distance(pd_box, dti_box)
            if dx == 0.0 or dz == 0.0:
                candidates.append((dist, dx, dz, overlap_area, dti_box))
        if not candidates:
            continue
        dist, dx, dz, overlap_area, nearest = min(candidates, key=lambda item: item[0])
        rows.append(
            {
                "pd": pd_box.item_id,
                "nearest_dti": nearest.item_id,
                "clearance_um": dist,
                "dx_um": dx,
                "dz_um": dz,
                "overlap_area_um2": overlap_area,
                "status": "PASS" if overlap_area == 0.0 and dist >= 0.0 else "FAIL",
            }
        )
    return {
        "pair_count": len(rows),
        "min_clearance_um": min((row["clearance_um"] for row in rows), default=None),
        "max_overlap_area_um2": max((row["overlap_area_um2"] for row in rows), default=None),
        "rows": rows,
    }


def dti_overlap_report(dti: list[Box]) -> dict[str, Any]:
    rows = []
    for index, first in enumerate(dti):
        for second in dti[index + 1 :]:
            _, _, _, area = rectangle_distance(first, second)
            if area > 0:
                rows.append({"first": first.item_id, "second": second.item_id, "overlap_area_um2": area})
    return {"overlap_count": len(rows), "max_overlap_area_um2": max((row["overlap_area_um2"] for row in rows), default=0.0), "rows": rows}


def all_boundary_coordinates(params: dict[str, Any], boxes_by_layer: dict[str, list[Box]]) -> list[float]:
    coords = []
    for boxes in boxes_by_layer.values():
        for box in boxes:
            coords.extend([box.xmin, box.xmax, box.zmin, box.zmax])
    max_lens_height = max(
        [box.height_um for box in boxes_by_layer.get("ocl", []) if box.height_um is not None]
        or [float(params["lens_height_um"])]
    )
    coords.extend(
        [
            -float(params["si_thickness_um"]),
            -float(params["pd_depth_min_um"]),
            -float(params["pd_depth_max_um"]),
            0.0,
            float(params["passivation_thickness_um"]),
            float(params["passivation_thickness_um"]) + float(params["cfa_thickness_um"]),
            float(params["passivation_thickness_um"]) + float(params["cfa_thickness_um"]) + max_lens_height,
        ]
    )
    return coords


def grid_snapping_report(params: dict[str, Any], grid_dx_um: float | None, boxes_by_layer: dict[str, list[Box]]) -> dict[str, Any]:
    if not grid_dx_um:
        return {"status": "MISSING", "reason": "grid_dx_um not found in CRA output"}
    errors = [abs(coord - round(coord / grid_dx_um) * grid_dx_um) for coord in all_boundary_coordinates(params, boxes_by_layer)]
    feature_um = {
        "lens_edge_gap_pixels": float(params["lens_edge_gap_um"]),
        "cfa_gap_pixels": float(params["cfa_gap_um"]),
        "dti_width_pixels": float(params["dti_width_um"]),
        "pd_margin_pixels": float(params["pd_margin_um"]),
    }
    feature_pixels = {
        name: value / grid_dx_um
        for name, value in feature_um.items()
        if value > 0.0
    }
    min_feature_pixels = min(feature_pixels.values()) if feature_pixels else None
    return {
        "status": "PASS" if min_feature_pixels is not None and min_feature_pixels >= 2.0 else "CHECK",
        "grid_dx_um": grid_dx_um,
        "max_boundary_snap_error_um": max(errors) if errors else None,
        "mean_boundary_snap_error_um": mean(errors),
        "max_boundary_snap_error_pixels": max(errors) / grid_dx_um if errors else None,
        "feature_pixels": feature_pixels,
        "min_feature_pixels": min_feature_pixels,
        "note": "FDTD uses Cartesian/Yee sampling; CAD boundaries are staircased unless subpixel smoothing or higher resolution is used.",
    }


def grid_dx_from_cra(cra_output: Path, sensor_id: str) -> float | None:
    summary = cra_output / "simulations" / sensor_id / "green" / "camera_lut_summary.csv"
    rows = csv_rows(summary)
    if not rows:
        return None
    return number(rows[0].get("grid_dx_um"))


def parse_msh_nodes(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[list[int]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nodes: dict[int, tuple[float, float, float]] = {}
    tetra_elements: list[list[int]] = []
    index = 0
    element_node_counts = {4: 4, 11: 10}
    while index < len(lines):
        line = lines[index].strip()
        if line == "$Nodes":
            index += 1
            block_count, _, _, _ = [int(float(item)) for item in lines[index].split()[:4]]
            index += 1
            for _ in range(block_count):
                _, _, parametric, block_nodes = [int(float(item)) for item in lines[index].split()[:4]]
                index += 1
                tags = [int(float(lines[index + offset].strip())) for offset in range(block_nodes)]
                index += block_nodes
                for tag in tags:
                    values = [float(item) for item in lines[index].split()]
                    index += 1
                    nodes[tag] = (values[0], values[1], values[2])
                    if parametric:
                        index += 0
        elif line == "$Elements":
            index += 1
            block_count, _, _, _ = [int(float(item)) for item in lines[index].split()[:4]]
            index += 1
            for _ in range(block_count):
                _, _, element_type, block_elements = [int(float(item)) for item in lines[index].split()[:4]]
                index += 1
                node_count = element_node_counts.get(element_type)
                for _ in range(block_elements):
                    parts = [int(float(item)) for item in lines[index].split()]
                    index += 1
                    if node_count == 4:
                        tetra_elements.append(parts[1:5])
                    elif node_count == 10:
                        tetra_elements.append(parts[1:5])
        else:
            index += 1
    return nodes, tetra_elements


def tetra_quality(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    edges = [dist(points[i], points[j]) for i in range(4) for j in range(i + 1, 4)]
    min_edge = min(edges)
    max_edge = max(edges)
    volume = abs(dot(sub(points[1], points[0]), cross(sub(points[2], points[0]), sub(points[3], points[0])))) / 6.0
    quality = 6.0 * math.sqrt(2.0) * volume / max(max_edge**3, 1.0e-18)
    aspect = max_edge / max(min_edge, 1.0e-18)
    return min_edge, quality, aspect


def percentile(values: list[float], fraction: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    index = min(len(finite) - 1, max(0, round((len(finite) - 1) * fraction)))
    return finite[index]


def mesh_quality_report(path: Path, params: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "MISSING",
            "mesh_path": repo_rel(path),
            "note": "No model.msh was generated for this template; FreeCAD/STEP geometry exists but mesh quality is not evaluated.",
        }
    nodes, tetra = parse_msh_nodes(path)
    min_edges = []
    qualities = []
    aspects = []
    for element in tetra:
        try:
            points = [nodes[tag] for tag in element]
        except KeyError:
            continue
        min_edge, quality, aspect = tetra_quality(points)
        min_edges.append(min_edge)
        qualities.append(quality)
        aspects.append(aspect)
    y_planes = {
        "si_passivation_y0": 0.0,
        "passivation_cfa": float(params["passivation_thickness_um"]),
        "cfa_ocl_base": float(params["passivation_thickness_um"]) + float(params["cfa_thickness_um"]),
        "pd_top": -float(params["pd_depth_min_um"]),
        "pd_bottom": -float(params["pd_depth_max_um"]),
    }
    tolerance = max(min(min_edges) if min_edges else 0.0, 0.02)
    plane_counts = {
        name: sum(1 for _, y, _ in nodes.values() if abs(y - plane) <= tolerance)
        for name, plane in y_planes.items()
    }
    min_quality = min(qualities) if qualities else None
    quality_p001 = percentile(qualities, 0.001)
    quality_p01 = percentile(qualities, 0.01)
    aspect_p99 = percentile(aspects, 0.99)
    sliver_count_q003 = sum(1 for value in qualities if value < 0.03)
    interface_status = "PASS" if all(count > 0 for count in plane_counts.values()) else "CHECK"
    mesh_status = (
        "PASS"
        if quality_p001 is not None
        and quality_p001 >= 0.03
        and (aspect_p99 is None or aspect_p99 <= 10.0)
        and interface_status == "PASS"
        else "CHECK"
    )
    return {
        "status": mesh_status,
        "mesh_path": repo_rel(path),
        "node_count": len(nodes),
        "tetra_count": len(tetra),
        "min_element_edge_um": min(min_edges) if min_edges else None,
        "mean_element_edge_um": mean(min_edges),
        "min_tetra_quality_proxy": min_quality,
        "tetra_quality_p001": quality_p001,
        "tetra_quality_p01": quality_p01,
        "mean_tetra_quality_proxy": mean(qualities),
        "max_aspect_ratio_proxy": max(aspects) if aspects else None,
        "aspect_ratio_p99": aspect_p99,
        "sliver_tetra_count_quality_lt_0p03": sliver_count_q003,
        "sliver_tetra_fraction_quality_lt_0p03": sliver_count_q003 / len(qualities) if qualities else None,
        "interface_plane_node_counts": plane_counts,
        "interface_conformity_status": interface_status,
        "note": "model.msh is a coarse CAD review mesh, not the Meep Yee grid and not a calibrated DEVSIM mesh.",
    }


def template_report(
    template_id: str,
    template_root: Path,
    cra_output: Path,
    tolerance_um: float,
    slope_tolerance: float,
    grid_dx_override_um: float | None = None,
) -> dict[str, Any]:
    root = template_root / template_id
    params = read_json(root / "template_parameters.json")
    geometry = read_json(root / "geometry_import.json")
    ocl = ocl_boxes(params, geometry)
    cfa = cfa_boxes(params, geometry)
    pd = pd_boxes(params)
    dti = dti_boxes(params)
    boxes_by_layer = {"ocl": ocl, "cfa": cfa, "pd": pd, "dti": dti}
    measured_grid_dx = grid_dx_from_cra(cra_output, f"topology_{template_id}")
    grid_dx = grid_dx_override_um or measured_grid_dx
    grid_report = grid_snapping_report(params, grid_dx, boxes_by_layer)
    grid_report["source"] = "boundary_grid_override" if grid_dx_override_um else "cra_output"
    grid_report["cra_output_grid_dx_um"] = measured_grid_dx
    return {
        "template_id": template_id,
        "geometry_source": repo_rel(root / "geometry_import.json"),
        "template_parameters": repo_rel(root / "template_parameters.json"),
        "c0_c1_ocl": c0_c1_ocl_report(params, ocl, tolerance_um, slope_tolerance),
        "interface_gap_overlap": {
            "ocl_ocl": layer_gap_report("ocl", ocl, float(params["lens_edge_gap_um"]), tolerance_um),
            "cfa_cfa": layer_gap_report("cfa", cfa, float(params["cfa_gap_um"]), tolerance_um),
            "pd_pd": layer_gap_report("pd", pd, 2.0 * float(params["pd_margin_um"]), tolerance_um),
            "pd_dti_clearance": pd_dti_clearance_report(pd, dti),
            "dti_dti_overlap": dti_overlap_report(dti),
        },
        "fdtd_grid_snapping": grid_report,
        "mesh_quality": mesh_quality_report(root / "model.msh", params),
    }


def kpi_rows(path: Path) -> list[dict[str, Any]]:
    rows = csv_rows(path)
    return [row for row in rows if row.get("source_template_id")]


def average_metric(rows: list[dict[str, Any]], template: str, channel: str, field: str, angle: float, metric: str) -> float | None:
    values = [
        value
        for row in rows
        if row.get("source_template_id") == template
        and row.get("channel") == channel
        and row.get("field") == field
        and number(row.get("nominal_cra_deg")) == angle
        for value in [number(row.get(metric))]
        if value is not None
    ]
    return mean(values)


def response_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mixed = "mixed_1x1_2x2_3x3_boundary"
    controls = ["nona_3x3_ocl", "quad_2x2_ocl_5x5_crosstalk"]
    channels = sorted({row.get("channel") for row in rows if row.get("source_template_id") == mixed})
    fields = sorted({row.get("field") for row in rows if row.get("source_template_id") == mixed})
    angles = sorted({number(row.get("nominal_cra_deg")) for row in rows if row.get("source_template_id") == mixed and number(row.get("nominal_cra_deg")) is not None})
    output_rows = []
    for channel in channels:
        for field in fields:
            for metric in RESPONSE_METRICS:
                mixed_0 = average_metric(rows, mixed, channel, field, 0.0, metric)
                control_0 = mean(
                    [
                        value
                        for template in controls
                        for value in [average_metric(rows, template, channel, field, 0.0, metric)]
                        if value is not None
                    ]
                )
                if mixed_0 is None or control_0 is None:
                    continue
                for angle in angles:
                    mixed_angle = average_metric(rows, mixed, channel, field, angle, metric)
                    control_angle = mean(
                        [
                            value
                            for template in controls
                            for value in [average_metric(rows, template, channel, field, angle, metric)]
                            if value is not None
                        ]
                    )
                    if mixed_angle is None or control_angle is None:
                        continue
                    geometry_component = mixed_0 - control_0
                    mixed_cra_delta = mixed_angle - mixed_0
                    control_cra_delta = control_angle - control_0
                    extra_boundary_cra_component = mixed_cra_delta - control_cra_delta
                    geom_abs = abs(geometry_component)
                    cra_abs = abs(extra_boundary_cra_component)
                    if geom_abs > 1.25 * cra_abs:
                        driver = "geometry_discontinuity"
                    elif cra_abs > 1.25 * geom_abs:
                        driver = "cra_interaction"
                    else:
                        driver = "mixed"
                    output_rows.append(
                        {
                            "channel": channel,
                            "field": field,
                            "metric": metric,
                            "nominal_cra_deg": angle,
                            "mixed_value": mixed_angle,
                            "homogeneous_control_avg": control_angle,
                            "geometry_component_at_0deg": geometry_component,
                            "mixed_cra_delta_from_0deg": mixed_cra_delta,
                            "homogeneous_cra_delta_from_0deg": control_cra_delta,
                            "extra_boundary_cra_component": extra_boundary_cra_component,
                            "dominant_driver": driver,
                        }
                    )
    counter = Counter(row["dominant_driver"] for row in output_rows if row["nominal_cra_deg"] != 0.0)
    return {
        "schema": "mixed_boundary_response_decomposition_v1",
        "mixed_template": mixed,
        "homogeneous_controls": controls,
        "row_count": len(output_rows),
        "dominant_driver_counts_nonzero_cra": dict(counter),
        "max_abs_geometry_component": max((abs(row["geometry_component_at_0deg"]) for row in output_rows), default=None),
        "max_abs_extra_boundary_cra_component": max((abs(row["extra_boundary_cra_component"]) for row in output_rows if row["nominal_cra_deg"] != 0.0), default=None),
        "rows": output_rows,
    }


def required_resolution_px_per_um(template: dict[str, Any], min_feature_pixels: float = 2.0) -> float | None:
    grid = template["fdtd_grid_snapping"]
    features = grid.get("feature_pixels", {})
    grid_dx = grid.get("grid_dx_um")
    if not isinstance(features, dict) or not grid_dx:
        return None
    feature_sizes = []
    for pixels in features.values():
        value = number(pixels)
        if value and value > 0:
            feature_sizes.append(value * float(grid_dx))
    if not feature_sizes:
        return None
    return min_feature_pixels / min(feature_sizes)


def template_readiness(template: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(settings["tolerance_um"])
    slope_tolerance = float(settings["slope_tolerance"])
    c0 = template["c0_c1_ocl"]
    grid = template["fdtd_grid_snapping"]
    mesh = template["mesh_quality"]
    interfaces = template["interface_gap_overlap"]
    blockers: list[str] = []
    checks: list[str] = []

    if (c0.get("max_c0_design_error_um") or 0.0) > tolerance:
        blockers.append("OCL design gap/overlap does not match configured lens_edge_gap_um.")
    if (c0.get("max_physical_gap_um") or 0.0) > tolerance:
        checks.append("OCL C0 physical surface is intentionally gapped; do not call it continuous.")
    if (c0.get("max_c1_slope_delta") or 0.0) > slope_tolerance:
        checks.append("OCL C1 slope is discontinuous at at least one adjacent boundary.")
    if interfaces["cfa_cfa"].get("max_design_error_um") and interfaces["cfa_cfa"]["max_design_error_um"] > tolerance:
        blockers.append("CFA tile gap/overlap deviates from configured cfa_gap_um.")
    if interfaces["pd_dti_clearance"].get("max_overlap_area_um2", 0.0) > 0.0:
        blockers.append("PD overlaps DTI in the x/z footprint.")
    non_blocking_notes: list[str] = []
    if interfaces["dti_dti_overlap"].get("overlap_count", 0) > 0:
        non_blocking_notes.append("DTI bars overlap at grid intersections in the simplified crossbar model; this is tracked but does not block OCL boundary readiness.")
    if grid.get("status") != "PASS":
        checks.append("FDTD grid undersamples boundary features; staircase error dominates boundary accuracy.")
    if mesh.get("status") == "MISSING":
        checks.append("No Gmsh/FreeCAD review mesh exists for this template.")
    elif mesh.get("status") != "PASS":
        checks.append("Coarse CAD review mesh quality is below the current pass threshold.")

    if blockers:
        status = "FAIL"
        use_scope = "do_not_use"
    elif checks:
        status = "CHECK"
        use_scope = "research_trend_only"
    else:
        status = "PASS"
        use_scope = "boundary_design_review"

    recommendations = []
    required_resolution = required_resolution_px_per_um(template)
    if required_resolution:
        recommendations.append(
            f"Use resolution >= {required_resolution:.1f} px/um to put the smallest configured boundary feature at >= 2 pixels."
        )
    if (c0.get("max_c1_slope_delta") or 0.0) > slope_tolerance:
        recommendations.append("Replace abrupt adjacent OCL caps with a shared freeform/sag surface or add measured OCL surface maps.")
    if mesh.get("status") == "MISSING":
        recommendations.append("Generate model.msh for CAD review mesh quality and interface-plane coverage.")
    if mesh.get("status") == "CHECK":
        recommendations.append("Regenerate CAD review mesh with smaller/finer element size near OCL/CFA/DTI boundaries.")
    if grid.get("status") != "PASS":
        recommendations.append("Run targeted boundary convergence before interpreting mixed-boundary optical asymmetry quantitatively.")
    return {
        "template_id": template["template_id"],
        "status": status,
        "use_scope": use_scope,
        "blockers": blockers,
        "checks": checks,
        "non_blocking_notes": non_blocking_notes,
        "recommended_resolution_px_per_um": required_resolution,
        "recommendations": recommendations,
    }


def readiness_report(report: dict[str, Any]) -> dict[str, Any]:
    templates = [template_readiness(template, report["settings"]) for template in report["templates"]]
    if any(item["status"] == "FAIL" for item in templates):
        status = "FAIL"
    elif any(item["status"] == "CHECK" for item in templates):
        status = "CHECK"
    else:
        status = "PASS"
    response_counts = report["response_decomposition"].get("dominant_driver_counts_nonzero_cra", {})
    return {
        "schema": "boundary_readiness_gate_v1",
        "status": status,
        "use_scope": "research_trend_only" if status != "PASS" else "boundary_design_review",
        "template_gates": templates,
        "response_driver_counts_nonzero_cra": response_counts,
        "camera_lut_boundary_gate": "CHECK" if status != "PASS" else "PASS",
        "notes": [
            "A PASS topology coverage result only means the case was simulated; this gate evaluates boundary-specific trust.",
            "Grid/mesh CHECK results keep the boundary data in research/trend scope even when CAD design-rule gaps match.",
        ],
    }


def summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for template in report["templates"]:
        c0 = template["c0_c1_ocl"]
        grid = template["fdtd_grid_snapping"]
        mesh = template["mesh_quality"]
        interfaces = template["interface_gap_overlap"]
        rows.append(
            {
                "template_id": template["template_id"],
                "readiness_status": report.get("readiness", {}).get("template_gates_by_id", {}).get(template["template_id"], {}).get("status"),
                "use_scope": report.get("readiness", {}).get("template_gates_by_id", {}).get(template["template_id"], {}).get("use_scope"),
                "ocl_adjacency_count": c0["adjacency_count"],
                "ocl_max_physical_gap_um": c0["max_physical_gap_um"],
                "ocl_max_c0_design_error_um": c0["max_c0_design_error_um"],
                "ocl_max_c1_slope_delta": c0["max_c1_slope_delta"],
                "cfa_max_design_error_um": interfaces["cfa_cfa"]["max_design_error_um"],
                "pd_dti_min_clearance_um": interfaces["pd_dti_clearance"]["min_clearance_um"],
                "dti_overlap_count": interfaces["dti_dti_overlap"]["overlap_count"],
                "grid_dx_um": grid.get("grid_dx_um"),
                "grid_max_snap_error_um": grid.get("max_boundary_snap_error_um"),
                "grid_min_feature_pixels": grid.get("min_feature_pixels"),
                "mesh_status": mesh.get("status"),
                "mesh_min_edge_um": mesh.get("min_element_edge_um"),
                "mesh_min_quality": mesh.get("min_tetra_quality_proxy"),
                "mesh_max_aspect_ratio": mesh.get("max_aspect_ratio_proxy"),
            }
        )
    return rows


def readiness_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "template_id": gate["template_id"],
            "status": gate["status"],
            "use_scope": gate["use_scope"],
            "recommended_resolution_px_per_um": gate["recommended_resolution_px_per_um"],
            "blockers": "; ".join(gate["blockers"]),
            "checks": "; ".join(gate["checks"]),
            "non_blocking_notes": "; ".join(gate.get("non_blocking_notes", [])),
            "recommendations": "; ".join(gate["recommendations"]),
        }
        for gate in report.get("readiness", {}).get("template_gates", [])
    ]


def table_html(rows: list[dict[str, Any]], limit: int = 100) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{escape(compact(row.get(column)))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, report: dict[str, Any]) -> None:
    summary = summary_rows(report)
    readiness = readiness_rows(report)
    response_rows = report["response_decomposition"]["rows"]
    status_rows = []
    for template in report["templates"]:
        status_rows.append(
            {
                "template_id": template["template_id"],
                "c0_physical_continuity": "CHECK" if (template["c0_c1_ocl"].get("max_physical_gap_um") or 0) > report["settings"]["tolerance_um"] else "PASS",
                "c0_design_error": "PASS" if (template["c0_c1_ocl"].get("max_c0_design_error_um") or 0) <= report["settings"]["tolerance_um"] else "FAIL",
                "c1_slope": "CHECK" if (template["c0_c1_ocl"].get("max_c1_slope_delta") or 0) > report["settings"]["slope_tolerance"] else "PASS",
                "grid_snapping": template["fdtd_grid_snapping"].get("status"),
                "mesh_quality": template["mesh_quality"].get("status"),
            }
        )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boundary Continuity Analysis</title>
  <style>
    body {{ margin:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#071017; color:#e7f4f8; }}
    h1,h2 {{ margin:0 0 10px; }}
    p {{ color:#9ab6c2; line-height:1.55; }}
    section {{ margin-top:18px; border:1px solid #24495a; border-radius:12px; padding:16px; background:#0d1b24; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border-bottom:1px solid #24495a; padding:7px; text-align:left; vertical-align:top; }}
    th {{ color:#54d7ee; }}
    code {{ color:#dcefff; }}
  </style>
</head>
<body>
  <h1>Boundary Continuity Analysis</h1>
  <p>This report separates CAD design-rule continuity, physical C0/C1 continuity, FDTD grid snapping risk, coarse Gmsh mesh quality, and mixed-boundary optical response drivers.</p>
  <section>
    <h2>Readiness Gate</h2>
    <p>Overall boundary gate: <code>{escape(str(report.get('readiness', {}).get('status')))}</code>. Use scope: <code>{escape(str(report.get('readiness', {}).get('use_scope')))}</code>.</p>
    {table_html(readiness)}
  </section>
  <section>
    <h2>Status</h2>
    {table_html(status_rows)}
  </section>
  <section>
    <h2>Summary Metrics</h2>
    {table_html(summary)}
  </section>
  <section>
    <h2>Mixed Boundary Response Decomposition</h2>
    <p>Geometry component is the mixed-boundary offset at CRA 0 deg versus homogeneous Nona/Quad controls. Extra CRA component is the remaining CRA-dependent mixed-boundary delta after subtracting the homogeneous CRA delta.</p>
    {table_html(response_rows, limit=180)}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    template_ids = [item.strip() for item in args.templates.split(",") if item.strip()]
    grid_dx_override_um = 1.0 / args.boundary_grid_resolution_px_per_um if args.boundary_grid_resolution_px_per_um else None
    templates = [
        template_report(
            template_id,
            args.template_root,
            args.cra_output,
            args.tolerance_um,
            args.slope_tolerance,
            grid_dx_override_um,
        )
        for template_id in template_ids
    ]
    response = response_decomposition(kpi_rows(args.cra_output / "cra_kpi_summary.csv"))
    report = {
        "schema": "boundary_continuity_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "tolerance_um": args.tolerance_um,
            "slope_tolerance": args.slope_tolerance,
            "template_root": repo_rel(args.template_root),
            "cra_output": repo_rel(args.cra_output),
            "output_dir": repo_rel(args.output_dir),
            "boundary_grid_resolution_px_per_um": args.boundary_grid_resolution_px_per_um,
            "boundary_grid_dx_override_um": grid_dx_override_um,
        },
        "templates": templates,
        "response_decomposition": response,
        "notes": [
            "C0 design error is measured against configured lens_edge_gap_um/cfa_gap_um, while physical C0 continuity requires near-zero signed gap.",
            "C1 slope is a spherical-cap edge-slope proxy from the parametric CAD model, not a measured freeform lens surface.",
            "FDTD snapping error is a Cartesian grid sampling proxy; Meep does not use the Gmsh/FreeCAD mesh.",
            "When boundary_grid_resolution_px_per_um is supplied, grid readiness is evaluated against that target grid while retaining cra_output_grid_dx_um for the actually available CRA run.",
            "Mesh quality is evaluated on optional coarse model.msh CAD review meshes, not on a solver-native FDTD mesh.",
        ],
    }
    readiness = readiness_report(report)
    readiness["template_gates_by_id"] = {gate["template_id"]: gate for gate in readiness["template_gates"]}
    report["readiness"] = readiness
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "boundary_continuity_report.json", report)
    write_csv(args.output_dir / "boundary_continuity_summary.csv", summary_rows(report))
    write_csv(args.output_dir / "boundary_readiness_gate.csv", readiness_rows(report))
    write_csv(args.output_dir / "mixed_boundary_response_decomposition.csv", response["rows"])
    write_html(args.output_dir / "index.html", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--cra-output", type=Path, default=DEFAULT_CRA_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--templates", default=DEFAULT_TEMPLATES)
    parser.add_argument("--tolerance-um", type=float, default=0.001)
    parser.add_argument("--slope-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--boundary-grid-resolution-px-per-um",
        type=float,
        default=None,
        help="Evaluate boundary grid readiness at this target FDTD resolution while keeping CRA output grid_dx as provenance.",
    )
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "template_count": len(report["templates"]),
                "response_rows": report["response_decomposition"]["row_count"],
                "output_dir": report["settings"].get("output_dir"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
