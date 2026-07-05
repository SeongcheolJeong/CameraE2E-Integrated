#!/usr/bin/env python3
"""Build practical G*W coupling reports for split-PD image-sensor runs.

This is a reduction model around the existing Meep/Gmsh/DEVSIM outputs:

    optical generation G(x, depth) * electrical collection weighting W

Two weighting variants are exported:

    W_proxy: analytic geometry/doping collection proxy
    W_mesh : FEM Laplace terminal weighting potential on the Gmsh mesh
    W_devsim_laplace: DEVSIM-native pure Laplace terminal weighting potential

Neither variant is a calibrated DEVSIM adjoint/Green's-function solve. The
report compares both reductions against native DEVSIM cathode electron-current signal deltas so
the approximation error is visible.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parent
Q_E = 1.602176634e-19


@dataclass(frozen=True)
class GmshMesh2D:
    nodes: dict[int, tuple[float, float]]
    triangles: list[tuple[int, int, int]]
    boundary_edges: dict[str, list[tuple[int, int]]]
    physical_names: dict[tuple[int, int], str]


@dataclass(frozen=True)
class CouplingConfig:
    generation_map_npz: Path
    split_summary: list[Path]
    output_dir: Path
    split_summary_manifest: Path | None = None
    devsim_weighting_csv: Path | None = None
    devsim_dd_probe_csv: Path | None = None
    optical_convergence_report: Path | None = None
    reference_case: str = "center"
    wavelength_nm: float = 550.0
    split_transition_um: float = 3.0
    split_center_offset_um: float = 0.07
    edge_rolloff_um: float = 0.025
    depth_rolloff_um: float = 0.08
    doping_transition_cm3: float = 2.0e15
    pixel_pitch_um: float = 1.4
    dd_probe_interpolation: str = "idw"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_summaries_from_manifest(path: Path) -> list[Path]:
    data = read_json(path)
    values = data.get("split_summaries")
    if values is None:
        outputs = data.get("outputs", {})
        list_path = outputs.get("split_summary_json")
        if list_path:
            list_file = Path(str(list_path))
            if not list_file.is_absolute():
                root_relative = (ROOT / list_file).resolve()
                list_file = root_relative if root_relative.exists() else (path.parent / list_file).resolve()
            values = read_json(list_file)
    if not isinstance(values, list):
        raise RuntimeError(f"split summary manifest does not contain a split_summaries list: {path}")
    summaries: list[Path] = []
    for value in values:
        summary_path = Path(str(value))
        if not summary_path.is_absolute():
            root_relative = (ROOT / summary_path).resolve()
            summary_path = root_relative if root_relative.exists() else (path.parent / summary_path).resolve()
        summaries.append(summary_path)
    if not summaries:
        raise RuntimeError(f"split summary manifest contains no completed summaries: {path}")
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _axis_float_key(value: Any, *, digits: int = 9) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return round(result, digits)


def response_axis_key(row: dict[str, Any]) -> tuple[str, float | None, float | None, float | None, float | None, float | None]:
    return (
        str(row.get("case", "")),
        _axis_float_key(row.get("wavelength_nm")),
        _axis_float_key(row.get("field_x_norm")),
        _axis_float_key(row.get("field_z_norm")),
        _axis_float_key(row.get("cra_x_deg")),
        _axis_float_key(row.get("cra_z_deg")),
    )


def response_axis_label(row: dict[str, Any]) -> str:
    case, wavelength, field_x, field_z, cra_x, cra_z = response_axis_key(row)
    parts = [case or "case"]
    if wavelength is not None:
        parts.append(f"{wavelength:g}nm")
    if field_x is not None:
        parts.append(f"fx{field_x:g}")
    if field_z is not None:
        parts.append(f"fz{field_z:g}")
    if cra_x is not None:
        parts.append(f"crax{cra_x:g}")
    if cra_z is not None:
        parts.append(f"craz{cra_z:g}")
    return "@".join(parts)


def response_tensor_from_long_rows(
    summary_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    region_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    long_by_axis_region: dict[
        tuple[
            tuple[str, float | None, float | None, float | None, float | None, float | None],
            str,
        ],
        dict[str, Any],
    ] = {}
    for row in long_rows:
        long_by_axis_region[(response_axis_key(row), str(row.get("region_id", "")))] = row

    case_keys = [response_axis_label(row) for row in summary_rows]
    tensor_rows = []
    for summary in summary_rows:
        axis_key = response_axis_key(summary)
        tensor_row = []
        for region_id in region_ids:
            long_row = long_by_axis_region.get((axis_key, region_id))
            if long_row is None:
                raise RuntimeError(
                    f"Missing long-row response for {response_axis_label(summary)} / {region_id}"
                )
            tensor_row.append(long_row["response_a_per_cm"])
        tensor_rows.append(tensor_row)
    return np.asarray(tensor_rows, dtype=float), case_keys


def _section(lines: list[str], name: str) -> tuple[int, int]:
    start_token = f"${name}"
    end_token = f"$End{name}"
    start = lines.index(start_token) + 1
    end = lines.index(end_token)
    return start, end


def parse_gmsh_msh22(path: Path) -> GmshMesh2D:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    physical_names: dict[tuple[int, int], str] = {}
    if "$PhysicalNames" in lines:
        phys_start, phys_end = _section(lines, "PhysicalNames")
        phys_count = int(lines[phys_start])
        for line in lines[phys_start + 1 : phys_start + 1 + phys_count]:
            dim_text, tag_text, raw_name = line.split(maxsplit=2)
            physical_names[(int(dim_text), int(tag_text))] = raw_name.strip('"')

    node_start, node_end = _section(lines, "Nodes")
    node_count = int(lines[node_start])
    nodes: dict[int, tuple[float, float]] = {}
    for line in lines[node_start + 1 : node_start + 1 + node_count]:
        parts = line.split()
        node_id = int(parts[0])
        nodes[node_id] = (float(parts[1]), float(parts[2]))

    elem_start, elem_end = _section(lines, "Elements")
    elem_count = int(lines[elem_start])
    triangles: list[tuple[int, int, int]] = []
    boundary_edges: dict[str, list[tuple[int, int]]] = {}
    for line in lines[elem_start + 1 : elem_start + 1 + elem_count]:
        parts = line.split()
        elem_type = int(parts[1])
        num_tags = int(parts[2])
        tags = [int(value) for value in parts[3 : 3 + num_tags]]
        node_ids = [int(value) for value in parts[3 + num_tags :]]
        if elem_type == 2 and len(node_ids) == 3:
            triangles.append((node_ids[0], node_ids[1], node_ids[2]))
        elif elem_type == 1 and len(node_ids) == 2:
            physical_name = physical_names.get((1, tags[0]), f"physical_{tags[0]}" if tags else "unlabeled")
            boundary_edges.setdefault(physical_name, []).append((node_ids[0], node_ids[1]))
    if not triangles:
        raise RuntimeError(f"no triangle elements found in {path}")
    return GmshMesh2D(
        nodes=nodes,
        triangles=triangles,
        boundary_edges=boundary_edges,
        physical_names=physical_names,
    )


def coordinate_key(x_cm: float, y_cm: float) -> tuple[float, float]:
    return (round(x_cm, 14), round(y_cm, 14))


def load_node_profile(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    rows = read_csv_rows(path)
    result: dict[tuple[float, float], dict[str, float]] = {}
    for row in rows:
        x_cm = float(row["x_cm"])
        y_cm = float(row["y_cm"])
        result[coordinate_key(x_cm, y_cm)] = {
            "x_cm": x_cm,
            "y_cm": y_cm,
            "x_um": float(row["x_um"]),
            "y_um": float(row["y_um"]),
            "Potential": float(row["Potential"]),
            "Electrons": float(row["Electrons"]),
            "Holes": float(row["Holes"]),
            "NetDoping": float(row["NetDoping"]),
            "FixedChargeDoping": float(row["FixedChargeDoping"]),
            "OpticalGenerationRate": float(row["OpticalGenerationRate"]),
            "ElectricField_proxy_v_per_cm": float(row["ElectricField_proxy_v_per_cm"]),
        }
    return result


def load_devsim_weighting(
    path: Path,
    mesh: GmshMesh2D,
    node_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    by_xy: dict[tuple[float, float], dict[str, str]] = {}
    for row in rows:
        by_xy[coordinate_key(float(row["x_cm"]), float(row["y_cm"]))] = row
    left: list[float] = []
    right: list[float] = []
    total: list[float] = []
    missing: list[int] = []
    for node_id in node_ids:
        row = by_xy.get(coordinate_key(*mesh.nodes[node_id]))
        if row is None:
            missing.append(node_id)
            left.append(math.nan)
            right.append(math.nan)
            total.append(math.nan)
            continue
        left_value = float(row["w_cathode_left_devsim_laplace"])
        right_value = float(row["w_cathode_right_devsim_laplace"])
        left.append(left_value)
        right.append(right_value)
        total.append(float(row.get("w_total_devsim_laplace", left_value + right_value)))
    if missing:
        raise RuntimeError(f"{len(missing)} mesh nodes are missing from DEVSIM weighting CSV {path}")
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float), np.asarray(total, dtype=float)


def load_devsim_dd_probe_weighting(
    path: Path,
    x_um: np.ndarray,
    depth_um: np.ndarray,
    case: str = "",
    k_nearest: int = 8,
    interpolation: str = "idw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    case_rows = [row for row in rows if row.get("case", "") == case]
    if case_rows:
        rows = case_rows
    elif any("case" in row and row.get("case") for row in rows):
        raise RuntimeError(f"DD probe CSV {path} does not contain rows for case={case}")
    points: list[tuple[float, float]] = []
    left_values: list[float] = []
    right_values: list[float] = []
    for row in rows:
        try:
            left = float(row["w_left_devsim_dd_probe"])
            right = float(row["w_right_devsim_dd_probe"])
            x_value = float(row["x_um"])
            depth_value = float(row["depth_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid DD probe row in {path}: {row}") from exc
        if not all(math.isfinite(value) for value in (left, right, x_value, depth_value)):
            continue
        points.append((x_value, depth_value))
        left_values.append(left)
        right_values.append(right)
    if not points:
        raise RuntimeError(f"no finite DD probe response rows in {path}")
    probe_points = np.asarray(points, dtype=float)
    left_array = np.asarray(left_values, dtype=float)
    right_array = np.asarray(right_values, dtype=float)
    query = np.column_stack([x_um, depth_um])
    x_axis = np.asarray(sorted({point[0] for point in points}), dtype=float)
    depth_axis = np.asarray(sorted({point[1] for point in points}), dtype=float)
    if interpolation == "bilinear" and len(points) == len(x_axis) * len(depth_axis):
        left_grid = np.full((len(x_axis), len(depth_axis)), math.nan, dtype=float)
        right_grid = np.full((len(x_axis), len(depth_axis)), math.nan, dtype=float)
        x_index = {value: index for index, value in enumerate(x_axis)}
        depth_index = {value: index for index, value in enumerate(depth_axis)}
        for (x_value, depth_value), left_value, right_value in zip(points, left_values, right_values):
            left_grid[x_index[x_value], depth_index[depth_value]] = left_value
            right_grid[x_index[x_value], depth_index[depth_value]] = right_value
        if np.all(np.isfinite(left_grid)) and np.all(np.isfinite(right_grid)):
            clipped_x = np.clip(x_um, x_axis[0], x_axis[-1])
            clipped_depth = np.clip(depth_um, depth_axis[0], depth_axis[-1])
            left_out = bilinear_rectilinear(x_axis, depth_axis, left_grid, clipped_x, clipped_depth)
            right_out = bilinear_rectilinear(x_axis, depth_axis, right_grid, clipped_x, clipped_depth)
            return left_out, right_out, np.abs(left_out) + np.abs(right_out)
    elif interpolation not in {"idw", "bilinear"}:
        raise ValueError(f"unsupported DD-probe interpolation mode: {interpolation}")
    k = max(1, min(k_nearest, len(probe_points)))
    left_out = np.zeros(len(query), dtype=float)
    right_out = np.zeros(len(query), dtype=float)
    for index, point in enumerate(query):
        d2 = np.sum((probe_points - point) ** 2, axis=1)
        nearest = np.argpartition(d2, k - 1)[:k]
        exact = nearest[np.argmin(d2[nearest])]
        if d2[exact] < 1.0e-18:
            left_out[index] = left_array[exact]
            right_out[index] = right_array[exact]
            continue
        weights = 1.0 / np.maximum(d2[nearest], 1.0e-12)
        weights /= np.sum(weights)
        left_out[index] = float(np.sum(left_array[nearest] * weights))
        right_out[index] = float(np.sum(right_array[nearest] * weights))
    return left_out, right_out, np.abs(left_out) + np.abs(right_out)


def bilinear_rectilinear(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    x_order = np.argsort(x_axis)
    y_order = np.argsort(y_axis)
    x_axis = x_axis[x_order]
    y_axis = y_axis[y_order]
    values = values[np.ix_(x_order, y_order)]

    result = np.zeros_like(x_query, dtype=float)
    inside = (
        (x_query >= x_axis[0])
        & (x_query <= x_axis[-1])
        & (y_query >= y_axis[0])
        & (y_query <= y_axis[-1])
    )
    if not np.any(inside):
        return result

    x = x_query[inside]
    y = y_query[inside]
    ix1 = np.searchsorted(x_axis, x, side="right")
    iy1 = np.searchsorted(y_axis, y, side="right")
    ix1 = np.clip(ix1, 1, len(x_axis) - 1)
    iy1 = np.clip(iy1, 1, len(y_axis) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = y_axis[iy0]
    y1 = y_axis[iy1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))

    v00 = values[ix0, iy0]
    v10 = values[ix1, iy0]
    v01 = values[ix0, iy1]
    v11 = values[ix1, iy1]
    result[inside] = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return result


def load_generation_case(
    path: Path,
    case: str,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        cases = np.asarray(data["case"]).astype(str)
        wavelengths = np.asarray(data["wavelength_nm"], dtype=float)
        candidates = (cases == case) & np.isclose(wavelengths, wavelength_nm, rtol=0.0, atol=1.0e-9)
        indices = np.flatnonzero(candidates)
        if indices.size == 0:
            raise RuntimeError(f"no generation map entry for case={case}, wavelength_nm={wavelength_nm}")
        index = int(indices[0])
        x_um = np.asarray(data["x_um"], dtype=float)
        depth_um = np.asarray(data["depth_um_from_si_top"], dtype=float)
        generation = np.asarray(data["generation_cm3_s"][index], dtype=float)
        metadata = {
            "case": str(cases[index]),
            "wavelength_nm": float(wavelengths[index]),
            "cra_x_deg": float(np.asarray(data["cra_x_deg"], dtype=float)[index]),
            "cra_z_deg": float(np.asarray(data["cra_z_deg"], dtype=float)[index]),
            "field_x_norm": float(np.asarray(data["field_x_norm"], dtype=float)[index]),
            "field_z_norm": float(np.asarray(data["field_z_norm"], dtype=float)[index]),
            "incident_photon_flux_cm2_s": float(np.asarray(data["incident_photon_flux_cm2_s"], dtype=float)[0]),
            "color_channel": str(np.asarray(data["color_channel"]).astype(str)[0]) if "color_channel" in data else "",
            "schema": str(np.asarray(data["schema"]).astype(str)[0]),
        }
    return x_um, depth_um, generation, metadata


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def collection_weights(
    x_um: np.ndarray,
    depth_um: np.ndarray,
    net_doping_cm3: np.ndarray,
    geometry: dict[str, Any],
    config: CouplingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width_um = float(geometry.get("width_um", 1.4))
    depth_max_um = float(geometry.get("depth_um", np.max(depth_um)))
    dti_width_um = float(geometry.get("dti_width_um", 0.06))
    pinning_depth_um = float(geometry.get("pinning_depth_um", 0.08))
    split_gap_um = float(geometry.get("split_gap_um", 0.04))

    half_width = 0.5 * width_um
    active_x_min = -half_width + dti_width_um
    active_x_max = half_width - dti_width_um
    edge_sigma = max(config.edge_rolloff_um, 1.0e-6)
    split_sigma = max(config.split_transition_um, 1.0e-6)
    depth_sigma = max(config.depth_rolloff_um, 1.0e-6)

    edge_window = stable_sigmoid((x_um - active_x_min) / edge_sigma) * stable_sigmoid(
        (active_x_max - x_um) / edge_sigma
    )
    depth_window = stable_sigmoid((depth_um - pinning_depth_um) / depth_sigma) * stable_sigmoid(
        ((depth_max_um - 0.05) - depth_um) / (1.5 * depth_sigma)
    )
    storage_score = stable_sigmoid(net_doping_cm3 / max(config.doping_transition_cm3, 1.0))
    center_sigma = max(0.5 * split_gap_um, 0.02)
    center_barrier_loss = 1.0 - 0.65 * np.exp(-0.5 * (x_um / center_sigma) ** 2)

    collection = edge_window * depth_window * (0.25 + 0.75 * storage_score) * center_barrier_loss
    collection = np.clip(collection, 0.0, 1.0)
    p_right = stable_sigmoid((x_um - config.split_center_offset_um) / split_sigma)
    p_left = 1.0 - p_right
    return collection * p_left, collection * p_right, collection


def triangle_area_cm2(points: list[tuple[float, float]]) -> float:
    (x0, y0), (x1, y1), (x2, y2) = points
    return 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def boundary_node_set(mesh: GmshMesh2D, contact: str) -> set[int]:
    nodes: set[int] = set()
    for n0, n1 in mesh.boundary_edges.get(contact, []):
        nodes.add(n0)
        nodes.add(n1)
    return nodes


def solve_laplace_weighting(
    mesh: GmshMesh2D,
    node_ids: list[int],
    terminal: str,
) -> np.ndarray:
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    stiffness = lil_matrix((len(node_ids), len(node_ids)), dtype=float)

    for triangle in mesh.triangles:
        points = [mesh.nodes[node_id] for node_id in triangle]
        area = triangle_area_cm2(points)
        if area <= 0.0:
            continue
        (x0, y0), (x1, y1), (x2, y2) = points
        b = np.asarray([y1 - y2, y2 - y0, y0 - y1], dtype=float)
        c = np.asarray([x2 - x1, x0 - x2, x1 - x0], dtype=float)
        local = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for local_i, node_i in enumerate(triangle):
            row = id_to_index[node_i]
            for local_j, node_j in enumerate(triangle):
                stiffness[row, id_to_index[node_j]] += local[local_i, local_j]

    terminal_nodes = boundary_node_set(mesh, terminal)
    other_contacts = {"anode", "cathode_left", "cathode_right"} - {terminal}
    grounded_nodes: set[int] = set()
    for contact in other_contacts:
        grounded_nodes.update(boundary_node_set(mesh, contact))

    boundary_values: dict[int, float] = {node_id: 1.0 for node_id in terminal_nodes}
    boundary_values.update({node_id: 0.0 for node_id in grounded_nodes if node_id not in boundary_values})
    if not terminal_nodes:
        raise RuntimeError(f"mesh has no boundary nodes for terminal {terminal}")

    boundary_indices = np.asarray(sorted(id_to_index[node_id] for node_id in boundary_values), dtype=int)
    all_indices = np.arange(len(node_ids), dtype=int)
    free_indices = np.setdiff1d(all_indices, boundary_indices)
    solution = np.zeros(len(node_ids), dtype=float)
    for node_id, value in boundary_values.items():
        solution[id_to_index[node_id]] = value
    if free_indices.size:
        matrix = stiffness.tocsr()
        rhs = -matrix[free_indices][:, boundary_indices].dot(solution[boundary_indices])
        if not np.all(np.isfinite(rhs)):
            raise RuntimeError(f"non-finite Laplace RHS while solving {terminal}")
        solution[free_indices] = spsolve(matrix[free_indices][:, free_indices], rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(f"non-finite Laplace solution while solving {terminal}")
    return np.clip(solution, 0.0, 1.0)


def integrate_weighting(
    mesh: GmshMesh2D,
    id_to_index: dict[int, int],
    generation_nodes: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
) -> tuple[float, float]:
    integral_left = 0.0
    integral_right = 0.0
    for triangle in mesh.triangles:
        idx = [id_to_index[node_id] for node_id in triangle]
        area = triangle_area_cm2([mesh.nodes[node_id] for node_id in triangle])
        g = float(np.mean(generation_nodes[idx]))
        left_w = float(np.mean(w_left[idx]))
        right_w = float(np.mean(w_right[idx]))
        integral_left += g * left_w * area
        integral_right += g * right_w * area
    return Q_E * integral_left, Q_E * integral_right


def split_phase(left: float, right: float) -> float:
    denom = abs(left) + abs(right)
    return (right - left) / denom if denom else 0.0


def case_from_summary(summary: dict[str, Any], path: Path) -> str:
    return str(summary.get("config", {}).get("generation_profile_case") or path.parent.name)


def actual_currents(summary: dict[str, Any]) -> tuple[float, float, float]:
    left = float(summary.get("left_photo_delta_a_per_cm", 0.0))
    right = float(summary.get("right_photo_delta_a_per_cm", 0.0))
    return left, right, abs(left) + abs(right)


def relative_error(predicted: float, actual: float) -> float:
    if actual == 0.0:
        return 0.0 if predicted == 0.0 else math.inf
    return (predicted - actual) / actual


def evaluate_case(
    summary_path: Path,
    config: CouplingConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(summary_path)
    case = case_from_summary(summary, summary_path)
    node_profile_path = Path(summary.get("outputs", {}).get("node_profile_2d_csv", summary_path.parent / "node_profile_2d.csv"))
    gmsh_mesh_path = Path(summary.get("gmsh_mesh") or summary.get("config", {}).get("gmsh_mesh", ""))
    if not gmsh_mesh_path.is_absolute():
        gmsh_mesh_path = (ROOT / gmsh_mesh_path).resolve()
    if not node_profile_path.is_absolute():
        node_profile_path = (ROOT / node_profile_path).resolve()

    mesh = parse_gmsh_msh22(gmsh_mesh_path)
    nodes = mesh.nodes
    triangles = mesh.triangles
    profile_by_xy = load_node_profile(node_profile_path)
    geometry = summary.get("config", {})
    measured_profile_path = geometry.get("measured_profile", "")
    if measured_profile_path:
        profile_path = Path(measured_profile_path)
        if not profile_path.is_absolute():
            profile_path = (ROOT / profile_path).resolve()
        if profile_path.exists():
            geometry = read_json(profile_path).get("geometry", geometry)

    node_ids = sorted(nodes)
    x_cm = np.asarray([nodes[node_id][0] for node_id in node_ids], dtype=float)
    y_cm = np.asarray([nodes[node_id][1] for node_id in node_ids], dtype=float)
    x_um = x_cm * 1.0e4
    depth_um = y_cm * 1.0e4
    profile_rows = []
    missing = []
    for node_id in node_ids:
        row = profile_by_xy.get(coordinate_key(*nodes[node_id]))
        if row is None:
            missing.append(node_id)
            row = {
                "NetDoping": 0.0,
                "OpticalGenerationRate": 0.0,
                "Potential": 0.0,
                "ElectricField_proxy_v_per_cm": 0.0,
            }
        profile_rows.append(row)
    if missing:
        raise RuntimeError(f"{len(missing)} mesh nodes are missing from {node_profile_path}")

    net_doping = np.asarray([row["NetDoping"] for row in profile_rows], dtype=float)
    generation_profile = np.asarray([row["OpticalGenerationRate"] for row in profile_rows], dtype=float)
    x_axis, depth_axis, generation_map, generation_meta = load_generation_case(
        config.generation_map_npz,
        case,
        config.wavelength_nm,
    )
    generation_nodes = bilinear_rectilinear(x_axis, depth_axis, generation_map, x_um, depth_um)
    w_left, w_right, w_total = collection_weights(x_um, depth_um, net_doping, geometry, config)
    w_mesh_left = solve_laplace_weighting(mesh, node_ids, "cathode_left")
    w_mesh_right = solve_laplace_weighting(mesh, node_ids, "cathode_right")
    w_mesh_total = np.clip(w_mesh_left + w_mesh_right, 0.0, 1.0)
    if config.devsim_weighting_csv:
        weighting_csv = config.devsim_weighting_csv
        if not weighting_csv.is_absolute():
            weighting_csv = (ROOT / weighting_csv).resolve()
        w_devsim_left, w_devsim_right, w_devsim_total = load_devsim_weighting(weighting_csv, mesh, node_ids)
    else:
        w_devsim_left = np.full(len(node_ids), math.nan, dtype=float)
        w_devsim_right = np.full(len(node_ids), math.nan, dtype=float)
        w_devsim_total = np.full(len(node_ids), math.nan, dtype=float)
    has_devsim_weighting = bool(np.all(np.isfinite(w_devsim_left)) and np.all(np.isfinite(w_devsim_right)))
    if config.devsim_dd_probe_csv:
        dd_probe_csv = config.devsim_dd_probe_csv
        if not dd_probe_csv.is_absolute():
            dd_probe_csv = (ROOT / dd_probe_csv).resolve()
        w_dd_probe_left, w_dd_probe_right, w_dd_probe_total = load_devsim_dd_probe_weighting(
            dd_probe_csv,
            x_um,
            depth_um,
            case=case,
            interpolation=config.dd_probe_interpolation,
        )
    else:
        w_dd_probe_left = np.full(len(node_ids), math.nan, dtype=float)
        w_dd_probe_right = np.full(len(node_ids), math.nan, dtype=float)
        w_dd_probe_total = np.full(len(node_ids), math.nan, dtype=float)
    has_devsim_dd_probe = bool(
        np.all(np.isfinite(w_dd_probe_left)) and np.all(np.isfinite(w_dd_probe_right))
    )

    node_area = np.zeros(len(node_ids), dtype=float)
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    integral_generation = 0.0
    integral_left = 0.0
    integral_right = 0.0
    for triangle in triangles:
        idx = [id_to_index[node_id] for node_id in triangle]
        area = triangle_area_cm2([nodes[node_id] for node_id in triangle])
        for index in idx:
            node_area[index] += area / 3.0
        g = float(np.mean(generation_nodes[idx]))
        left_w = float(np.mean(w_left[idx]))
        right_w = float(np.mean(w_right[idx]))
        integral_generation += g * area
        integral_left += g * left_w * area
        integral_right += g * right_w * area

    left_raw = Q_E * integral_left
    right_raw = Q_E * integral_right
    mesh_left_raw, mesh_right_raw = integrate_weighting(mesh, id_to_index, generation_nodes, w_mesh_left, w_mesh_right)
    if has_devsim_weighting:
        devsim_left_raw, devsim_right_raw = integrate_weighting(
            mesh,
            id_to_index,
            generation_nodes,
            w_devsim_left,
            w_devsim_right,
        )
    else:
        devsim_left_raw = math.nan
        devsim_right_raw = math.nan
    if has_devsim_dd_probe:
        dd_probe_left_raw, dd_probe_right_raw = integrate_weighting(
            mesh,
            id_to_index,
            generation_nodes,
            w_dd_probe_left,
            w_dd_probe_right,
        )
    else:
        dd_probe_left_raw = math.nan
        dd_probe_right_raw = math.nan
    total_raw = abs(left_raw) + abs(right_raw)
    mesh_total_raw = abs(mesh_left_raw) + abs(mesh_right_raw)
    devsim_total_raw = abs(devsim_left_raw) + abs(devsim_right_raw) if has_devsim_weighting else math.nan
    dd_probe_total_raw = abs(dd_probe_left_raw) + abs(dd_probe_right_raw) if has_devsim_dd_probe else math.nan
    generated_current_a_per_cm = Q_E * integral_generation
    actual_left, actual_right, actual_total = actual_currents(summary)
    phase_raw = split_phase(left_raw, right_raw)
    mesh_phase_raw = split_phase(mesh_left_raw, mesh_right_raw)
    devsim_phase_raw = split_phase(devsim_left_raw, devsim_right_raw) if has_devsim_weighting else math.nan
    dd_probe_phase_raw = split_phase(dd_probe_left_raw, dd_probe_right_raw) if has_devsim_dd_probe else math.nan
    actual_phase = float(summary.get("photo_split_phase_x_proxy", 0.0))
    terminal_fit_scale = actual_total / total_raw if total_raw else math.nan
    mesh_terminal_fit_scale = actual_total / mesh_total_raw if mesh_total_raw else math.nan
    devsim_laplace_terminal_fit_scale = (
        actual_total / devsim_total_raw if has_devsim_weighting and devsim_total_raw else math.nan
    )
    dd_probe_terminal_fit_scale = (
        actual_total / dd_probe_total_raw if has_devsim_dd_probe and dd_probe_total_raw else math.nan
    )
    generation_diff = generation_nodes - generation_profile

    case_output = {
        "run_id": summary_path.parent.name,
        "case": case,
        "summary_json": str(summary_path),
        "node_profile_csv": str(node_profile_path),
        "gmsh_mesh": str(gmsh_mesh_path),
        "mesh_node_count": len(nodes),
        "triangle_count": len(triangles),
        "boundary_node_counts": {
            "anode": len(boundary_node_set(mesh, "anode")),
            "cathode_left": len(boundary_node_set(mesh, "cathode_left")),
            "cathode_right": len(boundary_node_set(mesh, "cathode_right")),
        },
        "wavelength_nm": config.wavelength_nm,
        "color_channel": generation_meta["color_channel"],
        "field_x_norm": generation_meta["field_x_norm"],
        "field_z_norm": generation_meta["field_z_norm"],
        "cra_x_deg": generation_meta["cra_x_deg"],
        "cra_z_deg": generation_meta["cra_z_deg"],
        "incident_photon_flux_cm2_s": generation_meta["incident_photon_flux_cm2_s"],
        "generated_current_a_per_cm": generated_current_a_per_cm,
        "gw_left_raw_a_per_cm": left_raw,
        "gw_right_raw_a_per_cm": right_raw,
        "gw_total_raw_a_per_cm": total_raw,
        "gw_split_phase_raw": phase_raw,
        "gw_mesh_left_raw_a_per_cm": mesh_left_raw,
        "gw_mesh_right_raw_a_per_cm": mesh_right_raw,
        "gw_mesh_total_raw_a_per_cm": mesh_total_raw,
        "gw_mesh_split_phase_raw": mesh_phase_raw,
        "gw_devsim_laplace_left_raw_a_per_cm": devsim_left_raw,
        "gw_devsim_laplace_right_raw_a_per_cm": devsim_right_raw,
        "gw_devsim_laplace_total_raw_a_per_cm": devsim_total_raw,
        "gw_devsim_laplace_split_phase_raw": devsim_phase_raw,
        "gw_devsim_dd_probe_left_raw_a_per_cm": dd_probe_left_raw,
        "gw_devsim_dd_probe_right_raw_a_per_cm": dd_probe_right_raw,
        "gw_devsim_dd_probe_total_raw_a_per_cm": dd_probe_total_raw,
        "gw_devsim_dd_probe_split_phase_raw": dd_probe_phase_raw,
        "native_left_delta_a_per_cm": actual_left,
        "native_right_delta_a_per_cm": actual_right,
        "native_total_abs_delta_a_per_cm": actual_total,
        "native_split_phase_x_proxy": actual_phase,
        "native_signal_carrier": summary.get("photo_signal_carrier", "electron"),
        "terminal_fit_scale_to_native": terminal_fit_scale,
        "mesh_terminal_fit_scale_to_native": mesh_terminal_fit_scale,
        "devsim_laplace_terminal_fit_scale_to_native": devsim_laplace_terminal_fit_scale,
        "devsim_dd_probe_terminal_fit_scale_to_native": dd_probe_terminal_fit_scale,
        "mean_w_left": float(np.mean(w_left)),
        "mean_w_right": float(np.mean(w_right)),
        "mean_w_total": float(np.mean(w_total)),
        "mean_w_mesh_left": float(np.mean(w_mesh_left)),
        "mean_w_mesh_right": float(np.mean(w_mesh_right)),
        "mean_w_mesh_total": float(np.mean(w_mesh_total)),
        "mean_w_devsim_laplace_left": float(np.mean(w_devsim_left)) if has_devsim_weighting else math.nan,
        "mean_w_devsim_laplace_right": float(np.mean(w_devsim_right)) if has_devsim_weighting else math.nan,
        "mean_w_devsim_laplace_total": float(np.mean(w_devsim_total)) if has_devsim_weighting else math.nan,
        "mean_w_devsim_dd_probe_left": float(np.mean(w_dd_probe_left)) if has_devsim_dd_probe else math.nan,
        "mean_w_devsim_dd_probe_right": float(np.mean(w_dd_probe_right)) if has_devsim_dd_probe else math.nan,
        "mean_w_devsim_dd_probe_total": float(np.mean(w_dd_probe_total)) if has_devsim_dd_probe else math.nan,
        "area_weighted_mean_w_total": float(np.sum(w_total * node_area) / np.sum(node_area)),
        "area_weighted_mean_w_mesh_total": float(np.sum(w_mesh_total * node_area) / np.sum(node_area)),
        "area_weighted_mean_w_devsim_laplace_total": float(np.sum(w_devsim_total * node_area) / np.sum(node_area))
        if has_devsim_weighting
        else math.nan,
        "area_weighted_mean_w_devsim_dd_probe_total": float(
            np.sum(w_dd_probe_total * node_area) / np.sum(node_area)
        )
        if has_devsim_dd_probe
        else math.nan,
        "generation_profile_max_abs_diff_cm3_s": float(np.max(np.abs(generation_diff))),
        "generation_profile_rms_diff_cm3_s": float(np.sqrt(np.mean(generation_diff**2))),
    }

    node_rows: list[dict[str, Any]] = []
    for index, node_id in enumerate(node_ids):
        node_rows.append(
            {
                "node_id": node_id,
                "x_um": x_um[index],
                "depth_um": depth_um[index],
                "cell_area_cm2_proxy": node_area[index],
                "generation_cm3_s": generation_nodes[index],
                "generation_profile_cm3_s": generation_profile[index],
                "w_left_proxy": w_left[index],
                "w_right_proxy": w_right[index],
                "w_total_proxy": w_total[index],
                "w_left_mesh": w_mesh_left[index],
                "w_right_mesh": w_mesh_right[index],
                "w_total_mesh": w_mesh_total[index],
                "w_left_devsim_laplace": w_devsim_left[index],
                "w_right_devsim_laplace": w_devsim_right[index],
                "w_total_devsim_laplace": w_devsim_total[index],
                "w_left_devsim_dd_probe": w_dd_probe_left[index],
                "w_right_devsim_dd_probe": w_dd_probe_right[index],
                "w_total_devsim_dd_probe": w_dd_probe_total[index],
                "gw_left_carriers_per_cm_s_proxy": generation_nodes[index] * w_left[index] * node_area[index],
                "gw_right_carriers_per_cm_s_proxy": generation_nodes[index] * w_right[index] * node_area[index],
                "gw_left_carriers_per_cm_s_mesh": generation_nodes[index] * w_mesh_left[index] * node_area[index],
                "gw_right_carriers_per_cm_s_mesh": generation_nodes[index] * w_mesh_right[index] * node_area[index],
                "gw_left_carriers_per_cm_s_devsim_laplace": generation_nodes[index]
                * w_devsim_left[index]
                * node_area[index]
                if has_devsim_weighting
                else math.nan,
                "gw_right_carriers_per_cm_s_devsim_laplace": generation_nodes[index]
                * w_devsim_right[index]
                * node_area[index]
                if has_devsim_weighting
                else math.nan,
                "gw_left_carriers_per_cm_s_devsim_dd_probe": generation_nodes[index]
                * w_dd_probe_left[index]
                * node_area[index]
                if has_devsim_dd_probe
                else math.nan,
                "gw_right_carriers_per_cm_s_devsim_dd_probe": generation_nodes[index]
                * w_dd_probe_right[index]
                * node_area[index]
                if has_devsim_dd_probe
                else math.nan,
                "net_doping_cm3": net_doping[index],
                "potential_v": profile_rows[index]["Potential"],
                "electric_field_proxy_v_per_cm": profile_rows[index]["ElectricField_proxy_v_per_cm"],
            }
        )
    return case_output, node_rows


def apply_reference_scale(rows: list[dict[str, Any]], reference_case: str) -> None:
    reference = next((row for row in rows if row["case"] == reference_case), rows[0] if rows else None)
    if not reference:
        return
    scale = float(reference["terminal_fit_scale_to_native"])
    mesh_scale = float(reference["mesh_terminal_fit_scale_to_native"])
    devsim_laplace_scale = float(reference.get("devsim_laplace_terminal_fit_scale_to_native", math.nan))
    dd_probe_scale = float(reference.get("devsim_dd_probe_terminal_fit_scale_to_native", math.nan))
    for row in rows:
        row["reference_case"] = reference["case"]
        row["reference_terminal_scale"] = scale
        row["mesh_reference_terminal_scale"] = mesh_scale
        row["devsim_laplace_reference_terminal_scale"] = devsim_laplace_scale
        row["devsim_dd_probe_reference_terminal_scale"] = dd_probe_scale
        row["gw_left_reference_scaled_a_per_cm"] = row["gw_left_raw_a_per_cm"] * scale
        row["gw_right_reference_scaled_a_per_cm"] = row["gw_right_raw_a_per_cm"] * scale
        row["gw_total_reference_scaled_a_per_cm"] = row["gw_total_raw_a_per_cm"] * scale
        row["gw_total_reference_scaled_rel_error"] = relative_error(
            row["gw_total_reference_scaled_a_per_cm"],
            row["native_total_abs_delta_a_per_cm"],
        )
        row["gw_split_phase_error"] = row["gw_split_phase_raw"] - row["native_split_phase_x_proxy"]
        row["gw_mesh_left_reference_scaled_a_per_cm"] = row["gw_mesh_left_raw_a_per_cm"] * mesh_scale
        row["gw_mesh_right_reference_scaled_a_per_cm"] = row["gw_mesh_right_raw_a_per_cm"] * mesh_scale
        row["gw_mesh_total_reference_scaled_a_per_cm"] = row["gw_mesh_total_raw_a_per_cm"] * mesh_scale
        row["gw_mesh_total_reference_scaled_rel_error"] = relative_error(
            row["gw_mesh_total_reference_scaled_a_per_cm"],
            row["native_total_abs_delta_a_per_cm"],
        )
        row["gw_mesh_split_phase_error"] = row["gw_mesh_split_phase_raw"] - row["native_split_phase_x_proxy"]
        if math.isfinite(devsim_laplace_scale) and math.isfinite(row["gw_devsim_laplace_total_raw_a_per_cm"]):
            row["gw_devsim_laplace_left_reference_scaled_a_per_cm"] = (
                row["gw_devsim_laplace_left_raw_a_per_cm"] * devsim_laplace_scale
            )
            row["gw_devsim_laplace_right_reference_scaled_a_per_cm"] = (
                row["gw_devsim_laplace_right_raw_a_per_cm"] * devsim_laplace_scale
            )
            row["gw_devsim_laplace_total_reference_scaled_a_per_cm"] = (
                row["gw_devsim_laplace_total_raw_a_per_cm"] * devsim_laplace_scale
            )
            row["gw_devsim_laplace_total_reference_scaled_rel_error"] = relative_error(
                row["gw_devsim_laplace_total_reference_scaled_a_per_cm"],
                row["native_total_abs_delta_a_per_cm"],
            )
            row["gw_devsim_laplace_split_phase_error"] = (
                row["gw_devsim_laplace_split_phase_raw"] - row["native_split_phase_x_proxy"]
            )
        else:
            row["gw_devsim_laplace_left_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_laplace_right_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_laplace_total_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_laplace_total_reference_scaled_rel_error"] = math.nan
            row["gw_devsim_laplace_split_phase_error"] = math.nan
        if math.isfinite(dd_probe_scale) and math.isfinite(row["gw_devsim_dd_probe_total_raw_a_per_cm"]):
            row["gw_devsim_dd_probe_left_reference_scaled_a_per_cm"] = (
                row["gw_devsim_dd_probe_left_raw_a_per_cm"] * dd_probe_scale
            )
            row["gw_devsim_dd_probe_right_reference_scaled_a_per_cm"] = (
                row["gw_devsim_dd_probe_right_raw_a_per_cm"] * dd_probe_scale
            )
            row["gw_devsim_dd_probe_total_reference_scaled_a_per_cm"] = (
                row["gw_devsim_dd_probe_total_raw_a_per_cm"] * dd_probe_scale
            )
            row["gw_devsim_dd_probe_total_reference_scaled_rel_error"] = relative_error(
                row["gw_devsim_dd_probe_total_reference_scaled_a_per_cm"],
                row["native_total_abs_delta_a_per_cm"],
            )
            row["gw_devsim_dd_probe_split_phase_error"] = (
                row["gw_devsim_dd_probe_split_phase_raw"] - row["native_split_phase_x_proxy"]
            )
        else:
            row["gw_devsim_dd_probe_left_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_dd_probe_right_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_dd_probe_total_reference_scaled_a_per_cm"] = math.nan
            row["gw_devsim_dd_probe_total_reference_scaled_rel_error"] = math.nan
            row["gw_devsim_dd_probe_split_phase_error"] = math.nan


def save_plots(output_dir: Path, rows: list[dict[str, Any]], node_outputs: dict[str, list[dict[str, Any]]]) -> None:
    has_devsim_laplace = any(
        math.isfinite(float(row.get("gw_devsim_laplace_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    has_devsim_dd_probe = any(
        math.isfinite(float(row.get("gw_devsim_dd_probe_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    cases = [row["case"] for row in rows]
    x = np.arange(len(rows))
    actual = [row["native_total_abs_delta_a_per_cm"] for row in rows]
    scaled = [row["gw_total_reference_scaled_a_per_cm"] for row in rows]
    mesh_scaled = [row["gw_mesh_total_reference_scaled_a_per_cm"] for row in rows]
    if has_devsim_laplace:
        devsim_scaled = [row["gw_devsim_laplace_total_reference_scaled_a_per_cm"] for row in rows]
        axes[0].bar(x - 0.30, actual, width=0.20, label="native DEVSIM")
        axes[0].bar(x - 0.10, scaled, width=0.20, label="G*W_proxy ref-scaled")
        axes[0].bar(x + 0.10, mesh_scaled, width=0.20, label="G*W_mesh ref-scaled")
        axes[0].bar(x + 0.30, devsim_scaled, width=0.20, label="G*W_devsim_laplace ref-scaled")
    else:
        axes[0].bar(x - 0.25, actual, width=0.25, label="native DEVSIM")
        axes[0].bar(x, scaled, width=0.25, label="G*W_proxy ref-scaled")
        axes[0].bar(x + 0.25, mesh_scaled, width=0.25, label="G*W_mesh ref-scaled")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cases)
    axes[0].set_ylabel("A/cm")
    axes[0].set_title("Total collected-current proxy")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    actual_phase = [row["native_split_phase_x_proxy"] for row in rows]
    raw_phase = [row["gw_split_phase_raw"] for row in rows]
    mesh_phase = [row["gw_mesh_split_phase_raw"] for row in rows]
    if has_devsim_laplace:
        devsim_phase = [row["gw_devsim_laplace_split_phase_raw"] for row in rows]
        axes[1].bar(x - 0.30, actual_phase, width=0.20, label="native DEVSIM")
        axes[1].bar(x - 0.10, raw_phase, width=0.20, label="G*W_proxy")
        axes[1].bar(x + 0.10, mesh_phase, width=0.20, label="G*W_mesh")
        axes[1].bar(x + 0.30, devsim_phase, width=0.20, label="G*W_devsim_laplace")
    else:
        axes[1].bar(x - 0.25, actual_phase, width=0.25, label="native DEVSIM")
        axes[1].bar(x, raw_phase, width=0.25, label="G*W_proxy")
        axes[1].bar(x + 0.25, mesh_phase, width=0.25, label="G*W_mesh")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cases)
    axes[1].set_title("Split phase")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.savefig(output_dir / "gw_coupling_response.png", dpi=180)
    plt.close(fig)

    plot_count = 5 if has_devsim_laplace else 4
    fig, axes = plt.subplots(
        len(rows),
        plot_count,
        figsize=(3.5 * plot_count, 3.6 * max(1, len(rows))),
        constrained_layout=True,
    )
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(rows):
        case = row["case"]
        nodes = node_outputs[case]
        xs = np.asarray([node["x_um"] for node in nodes], dtype=float)
        ys = np.asarray([node["depth_um"] for node in nodes], dtype=float)
        generation = np.asarray([node["generation_cm3_s"] for node in nodes], dtype=float)
        w_delta = np.asarray([node["w_right_proxy"] - node["w_left_proxy"] for node in nodes], dtype=float)
        w_mesh_delta = np.asarray([node["w_right_mesh"] - node["w_left_mesh"] for node in nodes], dtype=float)
        w_devsim_delta = np.asarray(
            [
                node["w_right_devsim_laplace"] - node["w_left_devsim_laplace"]
                for node in nodes
            ],
            dtype=float,
        )
        gw_mesh = np.asarray(
            [
                node["generation_cm3_s"] * (node["w_left_mesh"] + node["w_right_mesh"])
                for node in nodes
            ],
            dtype=float,
        )
        plots = [
            (np.log10(np.maximum(generation, 1.0)), "log10 G"),
            (w_delta, "W_proxy right-left"),
            (w_mesh_delta, "W_mesh right-left"),
        ]
        if has_devsim_laplace and np.all(np.isfinite(w_devsim_delta)):
            plots.append((w_devsim_delta, "W_devsim_laplace right-left"))
        plots.append((np.log10(np.maximum(gw_mesh, 1.0)), "log10 G*W_mesh"))
        for axis, (values, title) in zip(axes[row_index], plots):
            sc = axis.scatter(xs, ys, c=values, s=10, cmap="viridis")
            axis.set_title(f"{case}: {title}")
            axis.set_xlabel("x (um)")
            axis.set_ylabel("depth (um)")
            axis.invert_yaxis()
            fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "gw_coupling_maps.png", dpi=180)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    has_devsim_laplace = any(
        math.isfinite(float(row.get("gw_devsim_laplace_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    has_devsim_dd_probe = any(
        math.isfinite(float(row.get("gw_devsim_dd_probe_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    header = (
        "| Case | CRA X (deg) | Native total (A/cm) | W_proxy scaled (A/cm) | W_proxy rel err | "
        "W_mesh scaled (A/cm) | W_mesh rel err |"
    )
    rule = "|---|---:|---:|---:|---:|---:|---:|"
    if has_devsim_laplace:
        header += " W_devsim_laplace scaled (A/cm) | W_devsim_laplace rel err |"
        rule += "---:|---:|"
    if has_devsim_dd_probe:
        header += " W_devsim_dd_probe scaled (A/cm) | W_devsim_dd_probe rel err |"
        rule += "---:|---:|"
    header += " Native split | W_proxy split | W_mesh split |"
    rule += "---:|---:|---:|"
    if has_devsim_laplace:
        header += " W_devsim_laplace split |"
        rule += "---:|"
    if has_devsim_dd_probe:
        header += " W_devsim_dd_probe split |"
        rule += "---:|"

    lines = [
        "# G*W Coupling Report",
        "",
        "This report integrates Meep optical generation with a geometry/doping-based collection weighting proxy.",
        "It also solves a Gmsh-mesh FEM Laplace terminal weighting potential.",
        "When provided, it includes a DEVSIM-native pure-Laplace terminal weighting potential.",
        "When provided, it includes a sparse DEVSIM drift-diffusion local-generation probe response.",
        "No weighting in this report is a calibrated drift-diffusion adjoint collection solve.",
        "",
        header,
        rule,
    ]
    for row in rows:
        line = (
            "| {case} | {cra_x_deg:.3g} | {native_total_abs_delta_a_per_cm:.6e} | "
            "{gw_total_reference_scaled_a_per_cm:.6e} | {gw_total_reference_scaled_rel_error:.6g} | "
            "{gw_mesh_total_reference_scaled_a_per_cm:.6e} | {gw_mesh_total_reference_scaled_rel_error:.6g} | ".format(**row)
        )
        if has_devsim_laplace:
            line += (
                "{gw_devsim_laplace_total_reference_scaled_a_per_cm:.6e} | "
                "{gw_devsim_laplace_total_reference_scaled_rel_error:.6g} | ".format(**row)
            )
        if has_devsim_dd_probe:
            line += (
                "{gw_devsim_dd_probe_total_reference_scaled_a_per_cm:.6e} | "
                "{gw_devsim_dd_probe_total_reference_scaled_rel_error:.6g} | ".format(**row)
            )
        line += (
            "{native_split_phase_x_proxy:.6g} | {gw_split_phase_raw:.6g} | "
            "{gw_mesh_split_phase_raw:.6g} |".format(**row)
        )
        if has_devsim_laplace:
            line += " {gw_devsim_laplace_split_phase_raw:.6g} |".format(**row)
        if has_devsim_dd_probe:
            line += " {gw_devsim_dd_probe_split_phase_raw:.6g} |".format(**row)
        lines.append(line)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `W_proxy` is the analytic geometry/doping collection proxy.",
            "- `W_mesh` is a FEM Laplace terminal weighting potential solved on the Gmsh triangle mesh with contact Dirichlet boundaries.",
            "- `W_devsim_laplace` is the same pure-Laplace terminal weighting concept solved through DEVSIM and exported as a solver-native dataset.",
            "- `W_devsim_dd_probe` is interpolated from direct DEVSIM drift-diffusion local-generation probe solves; it includes the configured transport, SRH/trap, and bias operating point but is sparse and uncalibrated.",
            "- `Ref-scaled` applies one scalar fit from the reference case to expose whether each weighting transfers to other CRA cases.",
            "- If this report is used for a camera-system LUT, keep the label `proxy` until a measured stack, measured n,k, and calibrated electrical weighting model are available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if math.isfinite(value):
                return f"{value:.6e}"
            return str(value)
        return html.escape(str(value))

    has_devsim_laplace = any(
        math.isfinite(float(row.get("gw_devsim_laplace_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    has_devsim_dd_probe = any(
        math.isfinite(float(row.get("gw_devsim_dd_probe_total_reference_scaled_a_per_cm", math.nan)))
        for row in rows
    )
    header_cells = (
        "<th>Case</th><th>CRA X</th><th>Native total</th><th>W_proxy scaled</th>"
        "<th>W_proxy err</th><th>W_mesh scaled</th><th>W_mesh err</th>"
    )
    if has_devsim_laplace:
        header_cells += "<th>W_devsim_laplace scaled</th><th>W_devsim_laplace err</th>"
    if has_devsim_dd_probe:
        header_cells += "<th>W_devsim_dd_probe scaled</th><th>W_devsim_dd_probe err</th>"
    header_cells += "<th>Native split</th><th>W_proxy split</th><th>W_mesh split</th>"
    if has_devsim_laplace:
        header_cells += "<th>W_devsim_laplace split</th>"
    if has_devsim_dd_probe:
        header_cells += "<th>W_devsim_dd_probe split</th>"

    table_items = []
    for row in rows:
        cells = (
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{row['cra_x_deg']:.3g}</td>"
            f"<td>{fmt(row['native_total_abs_delta_a_per_cm'])}</td>"
            f"<td>{fmt(row['gw_total_reference_scaled_a_per_cm'])}</td>"
            f"<td>{fmt(row['gw_total_reference_scaled_rel_error'])}</td>"
            f"<td>{fmt(row['gw_mesh_total_reference_scaled_a_per_cm'])}</td>"
            f"<td>{fmt(row['gw_mesh_total_reference_scaled_rel_error'])}</td>"
        )
        if has_devsim_laplace:
            cells += (
                f"<td>{fmt(row['gw_devsim_laplace_total_reference_scaled_a_per_cm'])}</td>"
                f"<td>{fmt(row['gw_devsim_laplace_total_reference_scaled_rel_error'])}</td>"
            )
        if has_devsim_dd_probe:
            cells += (
                f"<td>{fmt(row['gw_devsim_dd_probe_total_reference_scaled_a_per_cm'])}</td>"
                f"<td>{fmt(row['gw_devsim_dd_probe_total_reference_scaled_rel_error'])}</td>"
            )
        cells += (
            f"<td>{fmt(row['native_split_phase_x_proxy'])}</td>"
            f"<td>{fmt(row['gw_split_phase_raw'])}</td>"
            f"<td>{fmt(row['gw_mesh_split_phase_raw'])}</td>"
        )
        if has_devsim_laplace:
            cells += f"<td>{fmt(row['gw_devsim_laplace_split_phase_raw'])}</td>"
        if has_devsim_dd_probe:
            cells += f"<td>{fmt(row['gw_devsim_dd_probe_split_phase_raw'])}</td>"
        table_items.append(f"<tr>{cells}</tr>")
    table_rows = "\n".join(table_items)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G*W Coupling Report</title>
<style>
body{{margin:20px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f8fafc}}
h1{{font-size:22px;margin:0 0 8px}}
p{{max-width:980px;line-height:1.45}}
.warn{{border-left:4px solid #b45309;background:#fff7ed;padding:10px 12px;margin:12px 0;max-width:980px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}
th,td{{border:1px solid #d7dde5;padding:6px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #d7dde5;background:white;margin:12px 0}}
code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>G*W Coupling Report</h1>
<p>This integrates Meep optical generation <code>G(x,depth)</code> with electrical
weighting variants on the Gmsh triangle mesh: analytic <code>W_proxy</code>, FEM
Laplace terminal weighting <code>W_mesh</code>, and when available DEVSIM-native
pure-Laplace terminal weighting <code>W_devsim_laplace</code> or sparse
drift-diffusion local-generation probes <code>W_devsim_dd_probe</code>.</p>
<div class="warn">Important: this is not a calibrated drift-diffusion adjoint weighting-function solve.
Do not label the output as product-accurate LUT data.</div>
<img src="gw_coupling_response.png" alt="G*W response comparison">
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<img src="gw_coupling_maps.png" alt="G*W maps">
</body>
</html>
""",
        encoding="utf-8",
    )


def lut_method_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "response_method": "native_devsim",
            "weighting_model": "native_terminal_current_delta",
            "left_response_a_per_cm": row["native_left_delta_a_per_cm"],
            "right_response_a_per_cm": row["native_right_delta_a_per_cm"],
            "total_response_a_per_cm": row["native_total_abs_delta_a_per_cm"],
            "split_phase_x": row["native_split_phase_x_proxy"],
            "total_reference_scaled_rel_error": 0.0,
            "split_phase_error_to_native": 0.0,
        },
        {
            "response_method": "gw_proxy_ref_scaled",
            "weighting_model": "analytic_geometry_doping_w_proxy",
            "left_response_a_per_cm": row["gw_left_reference_scaled_a_per_cm"],
            "right_response_a_per_cm": row["gw_right_reference_scaled_a_per_cm"],
            "total_response_a_per_cm": row["gw_total_reference_scaled_a_per_cm"],
            "split_phase_x": row["gw_split_phase_raw"],
            "total_reference_scaled_rel_error": row["gw_total_reference_scaled_rel_error"],
            "split_phase_error_to_native": row["gw_split_phase_error"],
        },
        {
            "response_method": "gw_mesh_ref_scaled",
            "weighting_model": "fem_laplace_terminal_w_mesh",
            "left_response_a_per_cm": row["gw_mesh_left_reference_scaled_a_per_cm"],
            "right_response_a_per_cm": row["gw_mesh_right_reference_scaled_a_per_cm"],
            "total_response_a_per_cm": row["gw_mesh_total_reference_scaled_a_per_cm"],
            "split_phase_x": row["gw_mesh_split_phase_raw"],
            "total_reference_scaled_rel_error": row["gw_mesh_total_reference_scaled_rel_error"],
            "split_phase_error_to_native": row["gw_mesh_split_phase_error"],
        },
    ]
    if math.isfinite(float(row.get("gw_devsim_laplace_total_reference_scaled_a_per_cm", math.nan))):
        rows.append(
            {
                "response_method": "gw_devsim_laplace_ref_scaled",
                "weighting_model": "devsim_native_laplace_terminal_w",
                "left_response_a_per_cm": row["gw_devsim_laplace_left_reference_scaled_a_per_cm"],
                "right_response_a_per_cm": row["gw_devsim_laplace_right_reference_scaled_a_per_cm"],
                "total_response_a_per_cm": row["gw_devsim_laplace_total_reference_scaled_a_per_cm"],
                "split_phase_x": row["gw_devsim_laplace_split_phase_raw"],
                "total_reference_scaled_rel_error": row["gw_devsim_laplace_total_reference_scaled_rel_error"],
                "split_phase_error_to_native": row["gw_devsim_laplace_split_phase_error"],
            }
        )
    if math.isfinite(float(row.get("gw_devsim_dd_probe_total_reference_scaled_a_per_cm", math.nan))):
        rows.append(
            {
                "response_method": "gw_devsim_dd_probe_ref_scaled",
                "weighting_model": "devsim_drift_diffusion_local_generation_probe_w",
                "left_response_a_per_cm": row["gw_devsim_dd_probe_left_reference_scaled_a_per_cm"],
                "right_response_a_per_cm": row["gw_devsim_dd_probe_right_reference_scaled_a_per_cm"],
                "total_response_a_per_cm": row["gw_devsim_dd_probe_total_reference_scaled_a_per_cm"],
                "split_phase_x": row["gw_devsim_dd_probe_split_phase_raw"],
                "total_reference_scaled_rel_error": row["gw_devsim_dd_probe_total_reference_scaled_rel_error"],
                "split_phase_error_to_native": row["gw_devsim_dd_probe_split_phase_error"],
            }
        )
    return rows


def eqe_proxy(current_a_per_cm: float, photon_flux_cm2_s: float, pitch_um: float) -> float:
    pitch_cm = pitch_um * 1.0e-4
    denom = Q_E * photon_flux_cm2_s * pitch_cm
    return current_a_per_cm / denom if denom else math.nan


def write_camera_lut_html(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['case']))}</td>"
        f"<td>{html.escape(str(row['response_method']))}</td>"
        f"<td>{float(row['cra_x_deg']):.3g}</td>"
        f"<td>{float(row['total_response_a_per_cm']):.6e}</td>"
        f"<td>{float(row['normalized_total_response_to_reference']):.6g}</td>"
        f"<td>{float(row['eqe_proxy_total']):.6g}</td>"
        f"<td>{float(row['split_phase_x']):.6g}</td>"
        f"<td>{float(row['total_reference_scaled_rel_error']):.6g}</td>"
        "</tr>"
        for row in summary_rows
    )
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera-System G*W LUT</title>
<style>
body{{margin:20px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f8fafc}}
h1{{font-size:22px;margin:0 0 8px}}
p{{max-width:980px;line-height:1.45}}
.warn{{border-left:4px solid #b45309;background:#fff7ed;padding:10px 12px;margin:12px 0;max-width:980px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}
th,td{{border:1px solid #d7dde5;padding:6px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>Camera-System G*W LUT</h1>
<p>This export reshapes native DEVSIM cathode electron-current signal deltas, <code>G*W_proxy</code>,
<code>G*W_mesh</code>, and when available <code>G*W_devsim_laplace</code> into
camera-system lookup-table columns.</p>
<div class="warn">Proxy only: no measured stack, measured n,k, or calibrated drift-diffusion collection targets are included.</div>
<table>
<thead><tr><th>Case</th><th>Method</th><th>CRA X</th><th>Total A/cm</th><th>Norm total</th><th>EQE proxy</th><th>Split phase</th><th>Total err</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_native_devsim_response_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
) -> dict[str, str]:
    native_summary_rows = [
        row for row in summary_rows if row.get("response_method") == "native_devsim"
    ]
    native_long_rows = [
        row for row in long_rows if row.get("response_method") == "native_devsim"
    ]
    summary_csv = output_dir / "camera_system_native_devsim_summary.csv"
    long_csv = output_dir / "camera_system_native_devsim_long.csv"
    json_path = output_dir / "camera_system_native_devsim_response.json"
    npz_path = output_dir / "camera_system_native_devsim_response.npz"
    write_csv(summary_csv, native_summary_rows)
    write_csv(long_csv, native_long_rows)

    cases = [row["case"] for row in native_summary_rows]
    region_ids = ["pd_left", "pd_right"]
    response_tensor, case_keys = response_tensor_from_long_rows(
        native_summary_rows,
        native_long_rows,
        region_ids,
    )
    np.savez(
        npz_path,
        response_a_per_cm=response_tensor,
        case=np.asarray(cases),
        case_key=np.asarray(case_keys),
        region_id=np.asarray(region_ids),
        response_method=np.asarray(["native_devsim"]),
    )
    manifest = {
        "schema": "camera_system_native_devsim_response_v1",
        "artifact_role": "direct_solver_response",
        "primary_response_method": "native_devsim",
        "solver": "DEVSIM drift-diffusion",
        "input_optical_generation": "FDTD G(x,depth) imported directly into DEVSIM",
        "product_lut_ready": False,
        "product_lut_block_reason": (
            "Direct solver responses are available, but product LUT export still requires measured stack geometry, "
            "measured optical n,k, calibrated electrical targets, and convergence gates."
        ),
        "summary_csv": str(summary_csv),
        "long_csv": str(long_csv),
        "npz": str(npz_path),
        "case_count": len(cases),
        "region_ids": region_ids,
        "tensor_axes": {
            "response_a_per_cm": ["case_index", "region_id"],
            "case": cases,
            "case_key": case_keys,
            "region_id": region_ids,
        },
        "tensor_shape": list(response_tensor.shape),
        "summary_rows": native_summary_rows,
        "long_rows": native_long_rows,
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "summary_csv": str(summary_csv),
        "long_csv": str(long_csv),
        "json": str(json_path),
        "npz": str(npz_path),
    }


def write_camera_system_research_lut_html(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['case']))}</td>"
        f"<td>{float(row['cra_x_deg']):.3g}</td>"
        f"<td>{float(row['total_response_a_per_cm']):.6e}</td>"
        f"<td>{float(row['normalized_total_response_to_reference']):.6g}</td>"
        f"<td>{float(row['split_phase_x']):.6g}</td>"
        f"<td>{float(row['eqe_proxy_total']):.6g}</td>"
        "</tr>"
        for row in rows
    )
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera-System Research LUT</title>
<style>
body{{margin:24px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f8fafc}}
.box{{max-width:980px;background:white;border:1px solid #cbd5e1;border-radius:8px;padding:18px 20px;box-shadow:0 8px 30px rgba(15,23,42,.06)}}
h1{{font-size:22px;margin:0 0 8px;color:#0f172a}}
p{{line-height:1.45;color:#475569}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:white;margin-top:14px}}
th,td{{border:1px solid #d7dde5;padding:6px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
code{{background:#eef2f7;padding:1px 4px;border-radius:4px}}
</style>
</head>
<body>
<div class="box">
<h1>Camera-System Research LUT</h1>
<p>This artifact uses <code>native_devsim</code> direct terminal-current responses from the full imported FDTD generation map.
It is intended for camera-system trend and sensitivity simulation, not product signoff.</p>
<table>
<thead><tr><th>Case</th><th>CRA X</th><th>Total A/cm</th><th>Norm total</th><th>Split phase</th><th>EQE proxy</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_camera_system_research_lut_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    optical_convergence_report: Path | None = None,
) -> dict[str, str]:
    native_summary_rows = [
        row for row in summary_rows if row.get("response_method") == "native_devsim"
    ]
    native_long_rows = [
        row for row in long_rows if row.get("response_method") == "native_devsim"
    ]
    summary_csv = output_dir / "camera_system_research_lut_summary.csv"
    long_csv = output_dir / "camera_system_research_lut_long.csv"
    json_path = output_dir / "camera_system_research_lut.json"
    npz_path = output_dir / "camera_system_research_lut.npz"
    html_path = output_dir / "camera_system_research_lut_report.html"
    write_csv(summary_csv, native_summary_rows)
    write_csv(long_csv, native_long_rows)

    cases = [row["case"] for row in native_summary_rows]
    region_ids = ["pd_left", "pd_right"]
    response_tensor, case_keys = response_tensor_from_long_rows(
        native_summary_rows,
        native_long_rows,
        region_ids,
    )
    np.savez(
        npz_path,
        response_a_per_cm=response_tensor,
        case=np.asarray(cases),
        case_key=np.asarray(case_keys),
        region_id=np.asarray(region_ids),
        response_method=np.asarray(["native_devsim"]),
        cra_x_deg=np.asarray([row["cra_x_deg"] for row in native_summary_rows], dtype=float),
        cra_z_deg=np.asarray([row["cra_z_deg"] for row in native_summary_rows], dtype=float),
        field_x_norm=np.asarray([row["field_x_norm"] for row in native_summary_rows], dtype=float),
        field_z_norm=np.asarray([row["field_z_norm"] for row in native_summary_rows], dtype=float),
        wavelength_nm=np.asarray([row["wavelength_nm"] for row in native_summary_rows], dtype=float),
        normalized_total_response_to_reference=np.asarray(
            [row["normalized_total_response_to_reference"] for row in native_summary_rows],
            dtype=float,
        ),
        split_phase_x=np.asarray([row["split_phase_x"] for row in native_summary_rows], dtype=float),
    )
    convergence_data: dict[str, Any] = {}
    convergence_path = ""
    if optical_convergence_report:
        resolved_convergence = (
            optical_convergence_report
            if optical_convergence_report.is_absolute()
            else (ROOT / optical_convergence_report).resolve()
        )
        convergence_path = str(resolved_convergence)
        if resolved_convergence.exists():
            convergence_data = read_json(resolved_convergence)
    numerical_convergence_passed = (
        bool(convergence_data.get("passed")) if convergence_data else None
    )
    full_numerical_convergence_passed = (
        bool(convergence_data.get("full_numerical_convergence_pass"))
        if convergence_data
        else None
    )
    if full_numerical_convergence_passed is True:
        research_lut_status = "READY_FULL_CONVERGENCE_PASS"
    elif numerical_convergence_passed is True:
        research_lut_status = "READY_PARTIAL_CONVERGENCE_PASS"
    elif numerical_convergence_passed is None:
        research_lut_status = "READY_CONVERGENCE_NOT_PROVEN"
    else:
        research_lut_status = "SMOKE_CONVERGENCE_FAIL"
    manifest = {
        "schema": "camera_system_research_lut_v1",
        "artifact_role": "research_lut",
        "research_lut_ready": True,
        "research_lut_status": research_lut_status,
        "product_lut_ready": False,
        "accuracy_class": "direct_solver_uncalibrated_reference",
        "primary_response_method": "native_devsim",
        "solver": "DEVSIM drift-diffusion",
        "input_optical_generation": "FDTD G(x,depth) imported directly into DEVSIM",
        "numerical_convergence": {
            "optical_convergence_report": convergence_path,
            "passed": numerical_convergence_passed,
            "spatial_convergence_pass": convergence_data.get("spatial_convergence_pass"),
            "time_convergence_pass": convergence_data.get("time_convergence_pass"),
            "pml_convergence_pass": convergence_data.get("pml_convergence_pass"),
            "full_numerical_convergence_pass": convergence_data.get(
                "full_numerical_convergence_pass"
            ),
            "varied_axes": convergence_data.get("varied_axes"),
            "unproven_axes": convergence_data.get("unproven_axes"),
            "failed_axes": convergence_data.get("failed_axes"),
            "axis_convergence": convergence_data.get("axis_convergence"),
            "unique_resolution_count": convergence_data.get("unique_resolution_count"),
            "unique_after_source_time_count": convergence_data.get(
                "unique_after_source_time_count"
            ),
            "unique_pml_count": convergence_data.get("unique_pml_count"),
            "max_total_response_rel_delta_to_reference": convergence_data.get(
                "max_total_response_rel_delta_to_reference"
            ),
            "max_split_phase_x_abs_delta_to_reference": convergence_data.get(
                "max_split_phase_x_abs_delta_to_reference"
            ),
            "negative_signed_flux_count": convergence_data.get("negative_signed_flux_count"),
            "note": (
                "Research LUT tensor is exported for trend simulation, but it must not be used "
                "as an accuracy LUT unless convergence and measured-data gates pass."
            ),
        },
        "summary_csv": str(summary_csv),
        "long_csv": str(long_csv),
        "npz": str(npz_path),
        "html": str(html_path),
        "case_count": len(cases),
        "region_ids": region_ids,
        "tensor_axes": {
            "response_a_per_cm": ["case_index", "region_id"],
            "case": cases,
            "case_key": case_keys,
            "region_id": region_ids,
        },
        "tensor_shape": list(response_tensor.shape),
        "summary_rows": native_summary_rows,
        "long_rows": native_long_rows,
        "acceptance_contract": [
            "Use native_devsim rows as the camera-system response source.",
            "Do not replace these rows with G*W surrogate rows unless the surrogate error gates pass against native_devsim.",
            "Do not label this artifact as a product accuracy LUT until measured stack, measured n,k, calibrated transport targets, and convergence gates pass.",
        ],
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_camera_system_research_lut_html(html_path, native_summary_rows)
    return {
        "summary_csv": str(summary_csv),
        "long_csv": str(long_csv),
        "json": str(json_path),
        "npz": str(npz_path),
        "html": str(html_path),
    }


def write_product_lut_blocked_html(path: Path, diagnostic_json: Path, removed_files: list[str]) -> None:
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera-System Product LUT Blocked</title>
<style>
body{{margin:24px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#fff7ed}}
.box{{max-width:920px;background:white;border:1px solid #f59e0b;border-left:6px solid #dc2626;border-radius:8px;padding:18px 20px;box-shadow:0 8px 30px rgba(15,23,42,.08)}}
h1{{font-size:22px;margin:0 0 10px;color:#991b1b}}
p,li{{line-height:1.45}}
code{{background:#f1f5f9;padding:2px 5px;border-radius:4px}}
.muted{{color:#64748b}}
</style>
</head>
<body>
<div class="box">
  <h1>Product LUT export is blocked</h1>
  <p>This artifact intentionally contains no camera-system product LUT tensor.
  The available G*W response data is diagnostic only and has been written to
  <code>{html.escape(str(diagnostic_json))}</code>.</p>
  <p>A product LUT can be emitted only after measured stack geometry, measured
  optical n,k, calibrated collection/transport targets, and numerical convergence
  gates all pass.</p>
  <p class="muted">Removed stale product-like files: {html.escape(", ".join(removed_files) or "none")}.</p>
</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_camera_system_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    pixel_pitch_um: float = 1.4,
    optical_convergence_report: Path | None = None,
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    reference_by_method: dict[str, dict[str, Any]] = {}
    reference_region_by_method: dict[tuple[str, str], float] = {}

    for case_row in rows:
        for method_row in lut_method_rows(case_row):
            response_method = method_row["response_method"]
            if response_method not in reference_by_method or case_row["case"] == case_row["reference_case"]:
                reference_by_method[response_method] = {**case_row, **method_row}

    for case_row in rows:
        photon_flux = float(case_row["incident_photon_flux_cm2_s"])
        for method_row in lut_method_rows(case_row):
            response_method = method_row["response_method"]
            ref = reference_by_method[response_method]
            total = float(method_row["total_response_a_per_cm"])
            left = float(method_row["left_response_a_per_cm"])
            right = float(method_row["right_response_a_per_cm"])
            summary = {
                "schema": "camera_system_diagnostic_response_v1",
                "artifact_role": "diagnostic_response",
                "mode": "split-pd-1x1",
                "split_mode": "dual-x",
                "color_channel": case_row.get("color_channel", ""),
                "wavelength_nm": case_row["wavelength_nm"],
                "case": case_row["case"],
                "field_x_norm": case_row["field_x_norm"],
                "field_z_norm": case_row["field_z_norm"],
                "cra_x_deg": case_row["cra_x_deg"],
                "cra_z_deg": case_row["cra_z_deg"],
                "response_method": response_method,
                "weighting_model": method_row["weighting_model"],
                "reference_case": case_row["reference_case"],
                "total_response_a_per_cm": total,
                "left_response_a_per_cm": left,
                "right_response_a_per_cm": right,
                "normalized_total_response_to_reference": total / ref["total_response_a_per_cm"]
                if ref["total_response_a_per_cm"]
                else math.nan,
                "max_region_response_a_per_cm": max(left, right),
                "min_region_response_a_per_cm": min(left, right),
                "split_phase_x": method_row["split_phase_x"],
                "incident_photon_flux_cm2_s": photon_flux,
                "pixel_pitch_um": pixel_pitch_um,
                "eqe_proxy_total": eqe_proxy(total, photon_flux, pixel_pitch_um),
                "eqe_proxy_left": eqe_proxy(left, photon_flux, pixel_pitch_um),
                "eqe_proxy_right": eqe_proxy(right, photon_flux, pixel_pitch_um),
                "total_reference_scaled_rel_error": method_row["total_reference_scaled_rel_error"],
                "split_phase_error_to_native": method_row["split_phase_error_to_native"],
                "product_lut_ready": False,
                "product_lut_block_reason": "diagnostic G*W response; measured stack/n,k, calibrated transport, and convergence gates are required",
            }
            summary_rows.append(summary)

            for region_id, region_ix, response in (
                ("pd_left", -1, left),
                ("pd_right", 1, right),
            ):
                key = (response_method, region_id)
                if key not in reference_region_by_method or case_row["case"] == case_row["reference_case"]:
                    reference_region_by_method[key] = response
                ref_region = reference_region_by_method[key]
                long_rows.append(
                    {
                        "schema": "camera_system_diagnostic_response_v1",
                        "artifact_role": "diagnostic_response",
                        "mode": "split-pd-1x1",
                        "split_mode": "dual-x",
                        "color_channel": case_row.get("color_channel", ""),
                        "wavelength_nm": case_row["wavelength_nm"],
                        "case": case_row["case"],
                        "field_x_norm": case_row["field_x_norm"],
                        "field_z_norm": case_row["field_z_norm"],
                        "cra_x_deg": case_row["cra_x_deg"],
                        "cra_z_deg": case_row["cra_z_deg"],
                        "response_method": response_method,
                        "weighting_model": method_row["weighting_model"],
                        "region_id": region_id,
                        "region_kind": "subpd",
                        "region_ix": region_ix,
                        "region_iz": 0,
                        "response_a_per_cm": response,
                        "normalized_region_response_to_reference_same_region": response / ref_region
                        if ref_region
                        else math.nan,
                        "eqe_proxy_region": eqe_proxy(response, photon_flux, pixel_pitch_um),
                        "total_response_a_per_cm": total,
                        "split_phase_x": method_row["split_phase_x"],
                        "product_lut_ready": False,
                        "product_lut_block_reason": "diagnostic G*W response; measured stack/n,k, calibrated transport, and convergence gates are required",
                    }
                )

    summary_csv = output_dir / "camera_system_diagnostic_summary.csv"
    long_csv = output_dir / "camera_system_diagnostic_long.csv"
    json_path = output_dir / "camera_system_diagnostic.json"
    npz_path = output_dir / "camera_system_diagnostic.npz"
    html_path = output_dir / "camera_system_diagnostic_report.html"
    write_csv(summary_csv, summary_rows)
    write_csv(long_csv, long_rows)
    write_camera_lut_html(html_path, summary_rows)
    native_outputs = write_native_devsim_response_outputs(output_dir, summary_rows, long_rows)
    research_lut_outputs = write_camera_system_research_lut_outputs(
        output_dir,
        summary_rows,
        long_rows,
        optical_convergence_report,
    )

    methods = sorted({row["response_method"] for row in summary_rows})
    cases = [row["case"] for row in rows]
    response_tensor = np.asarray(
        [
            [
                [
                    next(
                        row
                        for row in long_rows
                        if row["response_method"] == method
                        and row["case"] == case
                        and row["region_id"] == region_id
                    )["response_a_per_cm"]
                    for region_id in ("pd_left", "pd_right")
                ]
                for case in cases
            ]
            for method in methods
        ],
        dtype=float,
    )
    np.savez(
        npz_path,
        response_a_per_cm=response_tensor,
        response_method=np.asarray(methods),
        case=np.asarray(cases),
        region_id=np.asarray(["pd_left", "pd_right"]),
    )

    manifest = {
        "schema": "camera_system_diagnostic_response_v1",
        "artifact_role": "diagnostic_response",
        "primary_response_method": "native_devsim",
        "primary_response_artifact": native_outputs["json"],
        "research_lut_artifact": research_lut_outputs["json"],
        "product_lut_ready": False,
        "product_lut_status": "BLOCKED",
        "summary_csv": str(summary_csv),
        "long_csv": str(long_csv),
        "json": str(json_path),
        "npz": str(npz_path),
        "html": str(html_path),
        "response_methods": methods,
        "case_count": len(cases),
        "region_ids": ["pd_left", "pd_right"],
        "tensor_axes": {
            "response_a_per_cm": ["response_method", "case", "region_id"],
            "response_method": methods,
            "case": cases,
            "region_id": ["pd_left", "pd_right"],
        },
        "tensor_shape": list(response_tensor.shape),
        "summary_rows": summary_rows,
        "long_rows": long_rows,
        "notes": [
            "native_devsim is the primary direct-solver response path: the FDTD G(x,depth) map is imported into DEVSIM and cathode electron-current deltas are measured directly.",
            "G*W rows are surrogate reductions for speed and should be accepted only when their error gates pass against native_devsim.",
            "EQE values are proxy normalizations using incident photon flux and pixel pitch.",
            "W_devsim_laplace, when present, is a pure Laplace terminal weighting potential from DEVSIM, not a calibrated collection probability.",
            "W_devsim_dd_probe, when present, is interpolated from direct drift-diffusion local-generation probe solves; it is solver-derived but sparse and uncalibrated.",
            "This is not emitted as a product LUT because measured stack, measured n,k, calibrated collection targets, and convergence gates are not all available.",
        ],
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stale_product_files = [
        output_dir / "camera_system_lut_summary.csv",
        output_dir / "camera_system_lut_long.csv",
        output_dir / "camera_system_lut.npz",
    ]
    removed_files: list[str] = []
    for stale_path in stale_product_files:
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
            removed_files.append(str(stale_path))

    product_json = output_dir / "camera_system_lut.json"
    product_html = output_dir / "camera_system_lut_report.html"
    product_block = {
        "schema": "camera_system_product_lut_blocked_v1",
        "artifact_role": "product_lut_block",
        "product_lut_ready": False,
        "status": "BLOCKED",
        "reason": (
            "Product camera-system LUT export is blocked until measured stack geometry, measured optical n,k, "
            "calibrated electrical collection/transport targets, and convergence gates all pass."
        ),
        "diagnostic_response_json": str(json_path),
        "diagnostic_response_report": str(html_path),
        "removed_stale_product_like_files": removed_files,
    }
    product_json.write_text(json.dumps(product_block, indent=2), encoding="utf-8")
    write_product_lut_blocked_html(product_html, json_path, removed_files)
    return {
        "diagnostic_response": {
            "summary_csv": str(summary_csv),
            "long_csv": str(long_csv),
            "json": str(json_path),
            "npz": str(npz_path),
            "html": str(html_path),
        },
        "native_devsim_response": native_outputs,
        "research_lut": research_lut_outputs,
        "product_lut": {
            "json": str(product_json),
            "html": str(product_html),
        },
        "primary_response_method": "native_devsim",
        "primary_response_artifact": native_outputs["json"],
        "research_lut_artifact": research_lut_outputs["json"],
        "response_methods": methods,
    }


def run(config: CouplingConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = list(config.split_summary)
    if config.split_summary_manifest:
        manifest_path = config.split_summary_manifest
        if not manifest_path.is_absolute():
            manifest_path = (ROOT / manifest_path).resolve()
        summary_paths.extend(split_summaries_from_manifest(manifest_path))
    deduped_summary_paths: list[Path] = []
    seen_summaries: set[str] = set()
    for summary_path in summary_paths:
        resolved = summary_path if summary_path.is_absolute() else (ROOT / summary_path).resolve()
        key = str(resolved)
        if key in seen_summaries:
            continue
        seen_summaries.add(key)
        deduped_summary_paths.append(resolved)
    if not deduped_summary_paths:
        raise RuntimeError("no DEVSIM split summaries were supplied")
    rows: list[dict[str, Any]] = []
    node_outputs: dict[str, list[dict[str, Any]]] = {}
    node_csvs: dict[str, str] = {}
    for summary_path in deduped_summary_paths:
        row, nodes = evaluate_case(summary_path, config)
        rows.append(row)
        node_outputs[row["case"]] = nodes
        node_csv = config.output_dir / f"gw_coupling_nodes_{row['case']}.csv"
        write_csv(node_csv, nodes)
        node_csvs[row["case"]] = str(node_csv)

    rows.sort(key=lambda row: (row["case"] != config.reference_case, row["case"]))
    apply_reference_scale(rows, config.reference_case)
    summary_csv = config.output_dir / "gw_coupling_summary.csv"
    write_csv(summary_csv, rows)
    save_plots(config.output_dir, rows, node_outputs)
    markdown_path = config.output_dir / "gw_coupling_report.md"
    html_path = config.output_dir / "gw_coupling_report.html"
    write_markdown(markdown_path, rows)
    write_html(html_path, rows)
    camera_system_outputs = write_camera_system_outputs(
        config.output_dir,
        rows,
        config.pixel_pitch_um,
        config.optical_convergence_report,
    )
    devsim_weighting_outputs: dict[str, str] = {}
    if config.devsim_weighting_csv:
        weighting_csv = config.devsim_weighting_csv
        if not weighting_csv.is_absolute():
            weighting_csv = (ROOT / weighting_csv).resolve()
        devsim_weighting_outputs["csv"] = str(weighting_csv)
        summary_path = weighting_csv.parent / "weighting_potential_2d_summary.json"
        plot_path = weighting_csv.parent / "weighting_potential_2d.png"
        tecplot_path = weighting_csv.parent / "weighting_potential_2d.dat"
        if summary_path.exists():
            devsim_weighting_outputs["summary_json"] = str(summary_path)
        if plot_path.exists():
            devsim_weighting_outputs["plot_png"] = str(plot_path)
        if tecplot_path.exists():
            devsim_weighting_outputs["tecplot"] = str(tecplot_path)
    devsim_dd_probe_outputs: dict[str, str] = {}
    if config.devsim_dd_probe_csv:
        dd_probe_csv = config.devsim_dd_probe_csv
        if not dd_probe_csv.is_absolute():
            dd_probe_csv = (ROOT / dd_probe_csv).resolve()
        devsim_dd_probe_outputs["csv"] = str(dd_probe_csv)
        summary_path = dd_probe_csv.parent / "dd_probe_response_2d_summary.json"
        if summary_path.exists():
            devsim_dd_probe_outputs["summary_json"] = str(summary_path)

    manifest = {
        "schema": "tcad_gw_coupling_report_v1",
        "method": "w_proxy_fem_mesh_devsim_laplace_and_optional_dd_probe_v1",
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "split_summary": [str(path) for path in deduped_summary_paths],
            "split_summary_manifest": str(config.split_summary_manifest) if config.split_summary_manifest else "",
            "output_dir": str(config.output_dir),
            "devsim_weighting_csv": str(config.devsim_weighting_csv) if config.devsim_weighting_csv else "",
            "devsim_dd_probe_csv": str(config.devsim_dd_probe_csv) if config.devsim_dd_probe_csv else "",
            "optical_convergence_report": str(config.optical_convergence_report)
            if config.optical_convergence_report
            else "",
        },
        "case_count": len(rows),
        "cases": rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "report_markdown": str(markdown_path),
            "report_html": str(html_path),
            "response_png": str(config.output_dir / "gw_coupling_response.png"),
            "maps_png": str(config.output_dir / "gw_coupling_maps.png"),
            "node_csvs": node_csvs,
            "camera_system_diagnostic": camera_system_outputs["diagnostic_response"],
            "native_devsim_response": camera_system_outputs["native_devsim_response"],
            "camera_system_research_lut": camera_system_outputs["research_lut"],
            "camera_system_lut": camera_system_outputs["product_lut"],
            "devsim_weighting": devsim_weighting_outputs,
            "devsim_dd_probe": devsim_dd_probe_outputs,
        },
        "primary_response_method": camera_system_outputs["primary_response_method"],
        "primary_response_artifact": camera_system_outputs["primary_response_artifact"],
        "research_lut_artifact": camera_system_outputs["research_lut_artifact"],
        "response_methods": camera_system_outputs["response_methods"],
        "product_lut_ready": False,
        "product_lut_status": "BLOCKED",
        "product_lut_block_reason": (
            "Only diagnostic G*W response artifacts are exported. Product LUT export requires measured stack/n,k, "
            "calibrated electrical targets, and passing numerical convergence gates."
        ),
        "limitations": [
            "native_devsim rows are direct DEVSIM drift-diffusion terminal-current responses to the imported FDTD generation map and are the primary accuracy-oriented path in this open-source flow.",
            "W_proxy is analytic and geometry/doping based.",
            "W_mesh is a FEM Laplace terminal weighting potential on the Gmsh mesh, not a drift-diffusion collection probability.",
            "W_devsim_laplace is a DEVSIM-native pure Laplace terminal weighting potential when a weighting CSV is supplied.",
            "W_devsim_dd_probe is a sparse local-generation drift-diffusion probe response when a DD probe CSV is supplied.",
            "No weighting in this report is a calibrated drift-diffusion adjoint collection probability.",
            "The reference scale is fitted to native DEVSIM cathode electron-current signal deltas, not measured sensor data.",
            "Use this report for design trend exploration only until measured process stack, measured n,k, and calibrated transport targets are available.",
        ],
    }
    manifest_path = config.output_dir / "gw_coupling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-map-npz",
        type=Path,
        default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
    )
    parser.add_argument(
        "--split-summary",
        type=Path,
        nargs="+",
        default=None,
    )
    parser.add_argument("--split-summary-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/tcad_gw_coupling_reference")
    default_devsim_weighting = ROOT / "runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv"
    parser.add_argument(
        "--devsim-weighting-csv",
        type=Path,
        default=default_devsim_weighting if default_devsim_weighting.exists() else None,
    )
    parser.add_argument(
        "--devsim-dd-probe-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--optical-convergence-report",
        type=Path,
        default=None,
        help="Optional Meep/FDTD convergence report to embed in research-LUT metadata.",
    )
    parser.add_argument("--reference-case", default="center")
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--split-transition-um", type=float, default=3.0)
    parser.add_argument("--split-center-offset-um", type=float, default=0.07)
    parser.add_argument("--edge-rolloff-um", type=float, default=0.025)
    parser.add_argument("--depth-rolloff-um", type=float, default=0.08)
    parser.add_argument("--doping-transition-cm3", type=float, default=2.0e15)
    parser.add_argument("--pixel-pitch-um", type=float, default=1.4)
    parser.add_argument("--dd-probe-interpolation", choices=("idw", "bilinear"), default="idw")
    args = parser.parse_args()
    split_summary = args.split_summary
    if split_summary is None and args.split_summary_manifest is None:
        split_summary = [
            ROOT / "runs/devsim_split_pd_2d_reference_profile_center_gmsh_native/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native/summary.json",
        ]
    run(
        CouplingConfig(
            generation_map_npz=args.generation_map_npz,
            split_summary=split_summary or [],
            output_dir=args.output_dir,
            split_summary_manifest=args.split_summary_manifest,
            devsim_weighting_csv=args.devsim_weighting_csv,
            devsim_dd_probe_csv=args.devsim_dd_probe_csv,
            optical_convergence_report=args.optical_convergence_report,
            reference_case=args.reference_case,
            wavelength_nm=args.wavelength_nm,
            split_transition_um=args.split_transition_um,
            split_center_offset_um=args.split_center_offset_um,
            edge_rolloff_um=args.edge_rolloff_um,
            depth_rolloff_um=args.depth_rolloff_um,
            doping_transition_cm3=args.doping_transition_cm3,
            pixel_pitch_um=args.pixel_pitch_um,
            dd_probe_interpolation=args.dd_probe_interpolation,
        )
    )


if __name__ == "__main__":
    main()
