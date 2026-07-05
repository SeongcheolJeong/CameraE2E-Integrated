#!/usr/bin/env python3
"""Export TCAD/FDTD artifacts to VTK and generate lightweight design viewers.

The script deliberately avoids heavy visualization dependencies. It writes
ParaView-friendly `.vtu`/`.vtk` files and self-contained HTML viewers that can
be opened directly from the workspace.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from measured_tcad_profile import load_measured_profile


ROOT = Path(__file__).resolve().parent

VTK_TRIANGLE = 5
VTK_TETRA = 10


@dataclass
class MeshData:
    points: np.ndarray
    cells: list[list[int]]
    vtk_type: int
    point_data: dict[str, np.ndarray]
    cell_data: dict[str, np.ndarray]
    source: str


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    clean = clean.strip("_")
    return clean or "value"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    if not rows:
        return
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_array(values: list[Any]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def parse_cellcentered_indices(zone_line: str) -> set[int]:
    match = re.search(r"VARLOCATION\s*=\s*\(\[([^\]]+)\]\s*=\s*CELLCENTERED\)", zone_line)
    if not match:
        return set()
    result: set[int] = set()
    for part in match.group(1).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = item.split("-", 1)
            result.update(range(int(start), int(stop) + 1))
        else:
            result.add(int(item))
    return result


def parse_tecplot_block(path: Path) -> MeshData:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    variables_line = next(line for line in lines if line.startswith("VARIABLES"))
    variables = re.findall(r'"([^"]+)"', variables_line)
    zone_index = next(index for index, line in enumerate(lines) if line.startswith("ZONE"))
    zone_line = lines[zone_index]
    node_count = int(re.search(r"NODES\s*=\s*(\d+)", zone_line).group(1))
    element_count = int(re.search(r"ELEMENTS\s*=\s*(\d+)", zone_line).group(1))
    zone_type = re.search(r"ZONETYPE\s*=\s*([A-Z0-9]+)", zone_line).group(1)
    if zone_type == "FETRIANGLE":
        nodes_per_cell = 3
        vtk_type = VTK_TRIANGLE
    elif zone_type == "FETETRAHEDRON":
        nodes_per_cell = 4
        vtk_type = VTK_TETRA
    else:
        raise ValueError(f"Unsupported Tecplot zone type {zone_type} in {path}")
    cellcentered = parse_cellcentered_indices(zone_line)
    tokens = " ".join(lines[zone_index + 1 :]).split()
    cursor = 0
    point_data_raw: dict[str, np.ndarray] = {}
    cell_data_raw: dict[str, np.ndarray] = {}
    for index, name in enumerate(variables, start=1):
        count = element_count if index in cellcentered else node_count
        values = np.asarray(tokens[cursor : cursor + count], dtype=float)
        cursor += count
        if index in cellcentered:
            cell_data_raw[sanitize_name(name)] = values
        else:
            point_data_raw[sanitize_name(name)] = values
    conn_tokens = tokens[cursor : cursor + element_count * nodes_per_cell]
    if len(conn_tokens) < element_count * nodes_per_cell:
        raise ValueError(f"{path}: not enough connectivity tokens")
    cells = []
    for offset in range(0, element_count * nodes_per_cell, nodes_per_cell):
        cells.append([int(float(value)) - 1 for value in conn_tokens[offset : offset + nodes_per_cell]])

    x = point_data_raw.pop("x")
    y = point_data_raw.pop("y")
    z = point_data_raw.pop("z", np.zeros_like(x))
    points = np.column_stack([x, y, z])
    keep_names = {
        "Acceptors",
        "Donors",
        "FixedChargeDoping",
        "NetDoping",
        "Potential",
        "Electrons",
        "Holes",
        "IntrinsicElectrons",
        "IntrinsicHoles",
        "ElectronGeneration",
        "HoleGeneration",
        "OpticalGenerationRate",
        "USRH",
        "NCharge",
        "PCharge",
        "IntrinsicCharge",
        "PotentialNodeCharge",
        "AtContactNode",
        "NodeVolume",
        "SurfaceArea",
        "anodenodemodel",
        "cathode_leftnodemodel",
        "cathode_rightnodemodel",
        "coordinate_index",
        "node_index",
    }
    point_data = {
        name: values
        for name, values in point_data_raw.items()
        if name in keep_names and len(values) == node_count
    }
    return MeshData(points, cells, vtk_type, point_data, cell_data_raw, str(path))


def vtk_data_array(name: str, values: np.ndarray, indent: str = "        ") -> str:
    flat = np.asarray(values, dtype=float).ravel()
    chunks = []
    for start in range(0, flat.size, 8):
        chunks.append(" ".join(f"{value:.16g}" for value in flat[start : start + 8]))
    body = ("\n" + indent + "  ").join(chunks)
    return (
        f'{indent}<DataArray type="Float64" Name="{html.escape(sanitize_name(name))}" '
        f'format="ascii">\n{indent}  {body}\n{indent}</DataArray>'
    )


def write_vtu(path: Path, mesh: MeshData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = mesh.points
    connectivity = np.asarray([node for cell in mesh.cells for node in cell], dtype=int)
    offsets = np.cumsum([len(cell) for cell in mesh.cells], dtype=int)
    types = np.full(len(mesh.cells), mesh.vtk_type, dtype=int)
    point_lines = [
        " ".join(f"{value:.16g}" for value in row)
        for row in points
    ]
    parts = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(mesh.cells)}">',
        "      <PointData>",
    ]
    for name, values in mesh.point_data.items():
        if len(values) == len(points):
            parts.append(vtk_data_array(name, values))
    parts.extend(
        [
            "      </PointData>",
            "      <CellData>",
        ]
    )
    for name, values in mesh.cell_data.items():
        if len(values) == len(mesh.cells):
            parts.append(vtk_data_array(name, values))
    parts.extend(
        [
            "      </CellData>",
            "      <Points>",
            '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
            "          " + "\n          ".join(point_lines),
            "        </DataArray>",
            "      </Points>",
            "      <Cells>",
            '        <DataArray type="Int32" Name="connectivity" format="ascii">',
            "          " + " ".join(str(int(value)) for value in connectivity),
            "        </DataArray>",
            '        <DataArray type="Int32" Name="offsets" format="ascii">',
            "          " + " ".join(str(int(value)) for value in offsets),
            "        </DataArray>",
            '        <DataArray type="UInt8" Name="types" format="ascii">',
            "          " + " ".join(str(int(value)) for value in types),
            "        </DataArray>",
            "      </Cells>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_legacy_vtk(path: Path, mesh: MeshData, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_size = sum(len(cell) + 1 for cell in mesh.cells)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {len(mesh.points)} float",
    ]
    lines.extend(" ".join(f"{value:.16g}" for value in point) for point in mesh.points)
    lines.append(f"CELLS {len(mesh.cells)} {cell_size}")
    lines.extend(
        f"{len(cell)} " + " ".join(str(int(node)) for node in cell)
        for cell in mesh.cells
    )
    lines.append(f"CELL_TYPES {len(mesh.cells)}")
    lines.extend(str(mesh.vtk_type) for _ in mesh.cells)
    lines.append(f"POINT_DATA {len(mesh.points)}")
    for name, values in mesh.point_data.items():
        if len(values) != len(mesh.points):
            continue
        clean = sanitize_name(name)
        lines.append(f"SCALARS {clean} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{float(value):.16g}" for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_profile_to_mesh(path: Path) -> MeshData:
    rows = read_csv_rows(path)
    x = float_array([row["x_cm"] for row in rows])
    y = float_array([row["y_cm"] for row in rows])
    points2 = np.column_stack([x, y])
    rounded = np.round(points2, decimals=14)
    unique_map: dict[tuple[float, float], int] = {}
    unique_indices: list[int] = []
    for index, point in enumerate(rounded):
        key = (float(point[0]), float(point[1]))
        if key not in unique_map:
            unique_map[key] = len(unique_indices)
            unique_indices.append(index)
    unique_points2 = points2[unique_indices]
    tri = Delaunay(unique_points2)
    points = np.column_stack([unique_points2[:, 0], unique_points2[:, 1], np.zeros(len(unique_points2))])
    numeric_keys = []
    for key in rows[0].keys():
        try:
            float(rows[0][key])
            numeric_keys.append(key)
        except Exception:
            pass
    point_data: dict[str, np.ndarray] = {}
    for key in numeric_keys:
        values = float_array([rows[index][key] for index in unique_indices])
        if key not in {"x_cm", "y_cm"}:
            point_data[sanitize_name(key)] = values
    return MeshData(points, [list(map(int, cell)) for cell in tri.simplices], VTK_TRIANGLE, point_data, {}, str(path))


def interp2d_rectilinear(
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
    ix1 = np.clip(np.searchsorted(x_axis, x, side="right"), 1, len(x_axis) - 1)
    iy1 = np.clip(np.searchsorted(y_axis, y, side="right"), 1, len(y_axis) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = y_axis[iy0]
    y1 = y_axis[iy1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=x1 != x0)
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=y1 != y0)
    v00 = values[ix0, iy0]
    v10 = values[ix1, iy0]
    v01 = values[ix0, iy1]
    v11 = values[ix1, iy1]
    result[inside] = (
        (1 - tx) * (1 - ty) * v00
        + tx * (1 - ty) * v10
        + (1 - tx) * ty * v01
        + tx * ty * v11
    )
    return result


def interp3d_rectilinear(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    z_query: np.ndarray,
) -> np.ndarray:
    x_order = np.argsort(x_axis)
    y_order = np.argsort(y_axis)
    z_order = np.argsort(z_axis)
    x_axis = x_axis[x_order]
    y_axis = y_axis[y_order]
    z_axis = z_axis[z_order]
    values = values[np.ix_(x_order, y_order, z_order)]
    result = np.zeros_like(x_query, dtype=float)
    inside = (
        (x_query >= x_axis[0])
        & (x_query <= x_axis[-1])
        & (y_query >= y_axis[0])
        & (y_query <= y_axis[-1])
        & (z_query >= z_axis[0])
        & (z_query <= z_axis[-1])
    )
    if not np.any(inside):
        return result
    x = x_query[inside]
    y = y_query[inside]
    z = z_query[inside]
    ix1 = np.clip(np.searchsorted(x_axis, x, side="right"), 1, len(x_axis) - 1)
    iy1 = np.clip(np.searchsorted(y_axis, y, side="right"), 1, len(y_axis) - 1)
    iz1 = np.clip(np.searchsorted(z_axis, z, side="right"), 1, len(z_axis) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    iz0 = iz1 - 1
    tx = np.divide(x - x_axis[ix0], x_axis[ix1] - x_axis[ix0], out=np.zeros_like(x), where=x_axis[ix1] != x_axis[ix0])
    ty = np.divide(y - y_axis[iy0], y_axis[iy1] - y_axis[iy0], out=np.zeros_like(y), where=y_axis[iy1] != y_axis[iy0])
    tz = np.divide(z - z_axis[iz0], z_axis[iz1] - z_axis[iz0], out=np.zeros_like(z), where=z_axis[iz1] != z_axis[iz0])
    c000 = values[ix0, iy0, iz0]
    c100 = values[ix1, iy0, iz0]
    c010 = values[ix0, iy1, iz0]
    c110 = values[ix1, iy1, iz0]
    c001 = values[ix0, iy0, iz1]
    c101 = values[ix1, iy0, iz1]
    c011 = values[ix0, iy1, iz1]
    c111 = values[ix1, iy1, iz1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    result[inside] = c0 * (1 - tz) + c1 * tz
    return result


def add_generation_to_mesh(mesh: MeshData, map_npz: Path | None, volume_npz: Path | None) -> None:
    if mesh.points.shape[1] < 3:
        return
    x_um = mesh.points[:, 0] * 1.0e4
    depth_um = mesh.points[:, 1] * 1.0e4
    z_um = mesh.points[:, 2] * 1.0e4
    if map_npz and map_npz.exists() and mesh.vtk_type == VTK_TRIANGLE:
        with np.load(map_npz, allow_pickle=False) as data:
            x_axis = np.asarray(data["x_um"], dtype=float)
            y_axis = np.asarray(data["depth_um_from_si_top"], dtype=float)
            cases = np.asarray(data["case"]).astype(str)
            generation = np.asarray(data["generation_cm3_s"], dtype=float)
            for index, case in enumerate(cases):
                mesh.point_data[f"OpticalGeneration_{case}"] = interp2d_rectilinear(
                    x_axis, y_axis, generation[index], x_um, depth_um
                )
    if volume_npz and volume_npz.exists() and mesh.vtk_type == VTK_TETRA:
        with np.load(volume_npz, allow_pickle=False) as data:
            x_axis = np.asarray(data["x_um"], dtype=float)
            y_axis = np.asarray(data["depth_um_from_si_top"], dtype=float)
            z_axis = np.asarray(data["z_um"], dtype=float)
            cases = np.asarray(data["case"]).astype(str)
            generation = np.asarray(data["generation_cm3_s"], dtype=float)
            for index, case in enumerate(cases):
                mesh.point_data[f"OpticalGeneration_{case}"] = interp3d_rectilinear(
                    x_axis, y_axis, z_axis, generation[index], x_um, depth_um, z_um
                )


def summary_row_from_split(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    left = float(summary["left_photo_delta_a_per_cm"])
    right = float(summary["right_photo_delta_a_per_cm"])
    total = left + right
    return {
        "run_id": summary_path.parent.name,
        "summary_json": str(summary_path),
        "case": config.get("generation_profile_case", ""),
        "wavelength_nm": config.get("generation_profile_wavelength_nm", ""),
        "electrical_model": summary.get("electrical_model", ""),
        "generation_source": summary.get("generation_source", ""),
        "photo_signal_carrier": summary.get("photo_signal_carrier", "electron"),
        "generation_map_scale": config.get("generation_map_scale", ""),
        "left_photo_delta_a_per_cm": left,
        "right_photo_delta_a_per_cm": right,
        "total_photo_delta_a_per_cm": total,
        "photo_split_phase_x_proxy": float(summary["photo_split_phase_x_proxy"]),
        "terminal_balance_illuminated_a_per_cm": float(summary["terminal_current_balance_illuminated_a_per_cm"]),
        "node_count": summary.get("node_count", ""),
    }


def discover_split_summaries(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.glob("**/summary.json")))
    unique = []
    seen = set()
    for path in result:
        real = str(path.resolve())
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def split_summary_sort_key(path: Path, by_case_only: bool) -> tuple[Any, ...]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    case = str(config.get("generation_profile_case", path.parent.name))
    if by_case_only:
        return (case,)
    scale = float(config.get("generation_map_scale") or 1.0)
    return (case, f"{scale:.12g}")


def split_summary_has_native_mesh(path: Path) -> bool:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    tecplot = summary.get("outputs", {}).get("device_tecplot")
    return bool(tecplot and Path(tecplot).exists())


def prefer_native_split_summaries(paths: list[Path], by_case_only: bool = False) -> list[Path]:
    preferred: dict[tuple[Any, ...], Path] = {}
    order: list[tuple[Any, ...]] = []
    for path in paths:
        key = split_summary_sort_key(path, by_case_only=by_case_only)
        if key not in preferred:
            preferred[key] = path
            order.append(key)
            continue
        old_native = split_summary_has_native_mesh(preferred[key])
        new_native = split_summary_has_native_mesh(path)
        if new_native and not old_native:
            preferred[key] = path
    return [preferred[key] for key in order]


def add_cra_metadata(rows: list[dict[str, Any]], generation_map_npz: Path | None) -> None:
    case_to_cra: dict[str, dict[str, float]] = {}
    if generation_map_npz and generation_map_npz.exists():
        with np.load(generation_map_npz, allow_pickle=False) as data:
            cases = np.asarray(data["case"]).astype(str)
            for index, case in enumerate(cases):
                case_to_cra[case] = {
                    "cra_x_deg": float(np.asarray(data["cra_x_deg"], dtype=float)[index]),
                    "cra_z_deg": float(np.asarray(data["cra_z_deg"], dtype=float)[index]),
                    "field_x_norm": float(np.asarray(data["field_x_norm"], dtype=float)[index]),
                    "field_z_norm": float(np.asarray(data["field_z_norm"], dtype=float)[index]),
                }
    for row in rows:
        row.update(case_to_cra.get(str(row.get("case", "")), {}))


def write_sweep_report(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "parameter_sweep_comparison.csv"
    write_csv_rows(
        csv_path,
        rows,
        [
            "run_id",
            "case",
            "cra_x_deg",
            "cra_z_deg",
            "generation_map_scale",
            "left_photo_delta_a_per_cm",
            "right_photo_delta_a_per_cm",
            "total_photo_delta_a_per_cm",
            "photo_split_phase_x_proxy",
            "terminal_balance_illuminated_a_per_cm",
            "summary_json",
        ],
    )
    png_path = report_dir / "parameter_sweep_comparison.png"
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        cases = sorted({str(row["case"]) for row in rows})
        for case in cases:
            case_rows = [row for row in rows if str(row["case"]) == case]
            case_rows.sort(key=lambda row: float(row.get("generation_map_scale") or 1.0))
            x = [float(row.get("generation_map_scale") or 1.0) for row in case_rows]
            total = [float(row["total_photo_delta_a_per_cm"]) for row in case_rows]
            phase = [float(row["photo_split_phase_x_proxy"]) for row in case_rows]
            axes[0].plot(x, total, marker="o", label=case)
            axes[1].plot(x, phase, marker="o", label=case)
        axes[0].set_xlabel("generation_map_scale")
        axes[0].set_ylabel("total photo delta (A/cm)")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("generation_map_scale")
        axes[1].set_ylabel("split phase x proxy")
        axes[1].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].legend()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    md_path = report_dir / "parameter_sweep_comparison.md"
    lines = [
        "# Parameter Sweep Comparison",
        "",
        "This report compares center/edge CRA response, split phase, and terminal balance.",
        "",
        f"- Row count: `{len(rows)}`",
        f"- CSV: `{csv_path}`",
        f"- Plot: `{png_path}`",
        "",
        "| Run | Case | CRA X | Scale | Total Photo Delta (A/cm) | Split Phase X | Terminal Balance (A/cm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['case']} | {float(row.get('cra_x_deg', 0.0)):.3g} | "
            f"{float(row.get('generation_map_scale') or 1.0):.6g} | "
            f"{float(row['total_photo_delta_a_per_cm']):.6e} | "
            f"{float(row['photo_split_phase_x_proxy']):.6g} | "
            f"{float(row['terminal_balance_illuminated_a_per_cm']):.3e} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_path = report_dir / "parameter_sweep_comparison.html"
    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key in [
                "run_id",
                "case",
                "cra_x_deg",
                "generation_map_scale",
                "total_photo_delta_a_per_cm",
                "photo_split_phase_x_proxy",
                "terminal_balance_illuminated_a_per_cm",
            ]
        )
        + "</tr>"
        for row in rows
    )
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TCAD Parameter Sweep</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d5d8dc;padding:6px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #d5d8dc}}
</style></head><body>
<h1>TCAD Parameter Sweep Comparison</h1>
<p>Center/edge CRA response, split phase, and terminal balance.</p>
<img src="parameter_sweep_comparison.png" alt="parameter sweep plot">
<table><thead><tr><th>Run</th><th>Case</th><th>CRA X</th><th>Scale</th><th>Total Photo Delta</th><th>Split Phase X</th><th>Terminal Balance</th></tr></thead><tbody>
{table_rows}
</tbody></table>
</body></html>
""",
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "markdown": str(md_path), "html": str(html_path), "png": str(png_path)}


def box_display_name(name: str) -> str:
    lower = name.lower()
    if lower == "microlens":
        return "microlens"
    if lower == "cfa":
        return "CFA"
    if lower == "passivation":
        return "passivation"
    if lower == "si":
        return "Si"
    if "floating" in lower or "fd" in lower:
        return "FD n+"
    if "transfer" in lower or "tg" in lower:
        return "TG barrier"
    if "left" in lower and "dti" in lower:
        return "L DTI"
    if "right" in lower and "dti" in lower:
        return "R DTI"
    if "split" in lower:
        return "split iso"
    if "pinning" in lower:
        return "pinning p+"
    compact = name.replace("_reference_stable", "").replace("_reference", "").replace("_", " ")
    return compact if len(compact) <= 18 else compact[:17] + "."


def profile_geometry_boxes(profile_path: Path, stack_config_path: Path) -> list[dict[str, Any]]:
    profile = load_measured_profile(profile_path)
    stack = json.loads(stack_config_path.read_text(encoding="utf-8"))
    geom = profile.geometry
    width = float(geom["width_um"])
    depth = float(geom["depth_um"])
    z_width = float(geom.get("z_width_um", width))
    half_x = 0.5 * width
    half_z = 0.5 * z_width
    stack_geom = stack.get("geometry_um", {})
    passivation = float(stack_geom.get("passivation_thickness", 0.08))
    cfa = float(stack_geom.get("cfa_thickness", 0.8))
    lens = float(stack_geom.get("lens_height", 0.657))
    boxes: list[dict[str, Any]] = [
        {"name": "microlens", "kind": "optical", "x0": -half_x, "x1": half_x, "y0": -(passivation + cfa + lens), "y1": -(passivation + cfa), "z0": -half_z, "z1": half_z, "color": "#66c2a5"},
        {"name": "CFA", "kind": "optical", "x0": -half_x, "x1": half_x, "y0": -(passivation + cfa), "y1": -passivation, "z0": -half_z, "z1": half_z, "color": "#8da0cb"},
        {"name": "passivation", "kind": "optical", "x0": -half_x, "x1": half_x, "y0": -passivation, "y1": 0.0, "z0": -half_z, "z1": half_z, "color": "#e5c494"},
        {"name": "Si", "kind": "silicon", "x0": -half_x, "x1": half_x, "y0": 0.0, "y1": depth, "z0": -half_z, "z1": half_z, "color": "#b3b3b3"},
    ]
    for implant in profile.implants:
        name = str(implant.get("name", "implant"))
        lower = name.lower()
        if "dti" in lower or "split" in lower or "pinning" in lower:
            boxes.append(
                {
                    "name": name,
                    "kind": "implant",
                    "x0": float(implant["x_min_um"]),
                    "x1": float(implant["x_max_um"]),
                    "y0": float(implant["depth_min_um"]),
                    "y1": float(implant["depth_max_um"]),
                    "z0": float(implant.get("z_min_um", -half_z)),
                    "z1": float(implant.get("z_max_um", half_z)),
                    "color": "#fc8d62" if "dti" in lower else "#ffd92f",
                }
            )
    for feature in profile.electrical_features:
        if feature.get("type") == "doping_box":
            role = str(feature.get("role", feature.get("name", ""))).lower()
            boxes.append(
                {
                    "name": str(feature.get("name", "feature")),
                    "kind": "electrical_feature",
                    "x0": float(feature["x_min_um"]),
                    "x1": float(feature["x_max_um"]),
                    "y0": float(feature["depth_min_um"]),
                    "y1": float(feature["depth_max_um"]),
                    "z0": float(feature.get("z_min_um", -half_z)),
                    "z1": float(feature.get("z_max_um", half_z)),
                    "color": "#e78ac3" if "floating" in role else "#a6d854",
                }
            )
    for box in boxes:
        box["display_name"] = box_display_name(str(box["name"]))
    return boxes


def build_2d_case_data(summary_paths: list[Path]) -> dict[str, Any]:
    cases = {}
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        case = summary.get("config", {}).get("generation_profile_case", summary_path.parent.name)
        node_path = summary_path.parent / "node_profile_2d.csv"
        rows = read_csv_rows(node_path)
        points = []
        for row in rows:
            generation = max(float(row.get("OpticalGenerationRate", 0.0)), 1.0)
            points.append(
                {
                    "x": float(row["x_um"]),
                    "y": float(row["y_um"]),
                    "NetDoping": float(row["NetDoping"]),
                    "Potential": float(row["Potential"]),
                    "FixedChargeDoping": float(row.get("FixedChargeDoping", 0.0)),
                    "OpticalGenerationRate": float(row.get("OpticalGenerationRate", 0.0)),
                    "log10OpticalGenerationRate": math.log10(generation),
                    "ElectricField_proxy_v_per_cm": float(row.get("ElectricField_proxy_v_per_cm", 0.0)),
                }
            )
        cases[case] = {
            "summary": summary_row_from_split(summary_path),
            "points": points,
        }
    return {"cases": cases}


def write_2d_viewer(path: Path, data: dict[str, Any], boxes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"data": data, "boxes": boxes})
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TCAD 2D Cross Section Viewer</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#17202a;background:#f7f9fb}}
header{{height:52px;display:flex;align-items:center;gap:16px;padding:0 18px;background:white;border-bottom:1px solid #d7dde5}}
main{{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,360px);height:calc(100vh - 53px)}}
canvas{{width:100%;height:100%;display:block;background:white}}
aside{{border-left:1px solid #d7dde5;background:#fbfcfd;padding:16px;overflow-y:auto;overflow-x:hidden;min-width:0}}
label{{display:block;font-size:12px;font-weight:600;margin-top:12px;color:#4d5b6a}}
select{{width:100%;padding:8px;margin-top:4px}}
.metric{{display:grid;grid-template-columns:1fr auto;gap:8px;margin:7px 0;font-size:13px}}
.metric span,.metric b{{min-width:0;overflow-wrap:anywhere}}
#legend{{display:grid;gap:6px}}
.legendItem{{display:flex;align-items:flex-start;gap:6px;min-width:0;font-size:13px;line-height:1.25;overflow-wrap:anywhere}}
.swatch{{display:inline-block;flex:0 0 auto;width:10px;height:10px;margin-top:3px;border:1px solid #3333}}
</style></head><body>
<header><strong>TCAD 2D Cross Section</strong><span>NetDoping / Potential / Optical Generation / Current Split Overlay</span></header>
<main><canvas id="canvas"></canvas><aside>
<label>Case</label><select id="case"></select>
<label>Field</label><select id="field">
<option>NetDoping</option><option>Potential</option><option>log10OpticalGenerationRate</option><option>FixedChargeDoping</option><option>ElectricField_proxy_v_per_cm</option>
</select>
<div id="metrics"></div>
<h3>Geometry Overlay</h3>
<div id="legend"></div>
</aside></main>
<script>
const payload = {payload};
const cases = Object.keys(payload.data.cases);
const caseSelect = document.getElementById('case');
const fieldSelect = document.getElementById('field');
for (const c of cases) {{ const o=document.createElement('option'); o.value=c; o.textContent=c; caseSelect.appendChild(o); }}
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
function resize() {{ canvas.width = canvas.clientWidth * devicePixelRatio; canvas.height = canvas.clientHeight * devicePixelRatio; draw(); }}
function color(t, diverge=false) {{
  t = Math.max(0, Math.min(1, t));
  if (diverge) {{
    if (t < .5) {{ const u=t*2; return `rgb(${{Math.round(30+220*u)}},${{Math.round(80+170*u)}},255)`; }}
    const u=(t-.5)*2; return `rgb(255,${{Math.round(250-180*u)}},${{Math.round(250-220*u)}})`;
  }}
  const r=Math.round(45+210*t), g=Math.round(70+110*Math.sin(t*Math.PI)), b=Math.round(170-140*t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function draw() {{
  if (!canvas.width) return;
  const caseName = caseSelect.value || cases[0];
  const field = fieldSelect.value;
  const points = payload.data.cases[caseName].points;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad=60*devicePixelRatio, w=canvas.width-2*pad, h=canvas.height-2*pad;
  const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=0, ymax=Math.max(...ys);
  const values=points.map(p=>p[field]);
  let vmin=Math.min(...values), vmax=Math.max(...values);
  const diverge = field.includes('Doping') || field.includes('ElectricField');
  if (diverge) {{ const m=Math.max(Math.abs(vmin), Math.abs(vmax)); vmin=-m; vmax=m; }}
  function sx(x) {{ return pad + (x-xmin)/(xmax-xmin)*w; }}
  function sy(y) {{ return pad + (y-ymin)/(ymax-ymin)*h; }}
  ctx.strokeStyle='#1f2933'; ctx.lineWidth=1*devicePixelRatio; ctx.strokeRect(pad,pad,w,h);
  for (const box of payload.boxes) {{
    if (box.y1 < 0) continue;
    ctx.fillStyle = box.color + '33'; ctx.strokeStyle = box.color; ctx.lineWidth=1.5*devicePixelRatio;
    const x=sx(box.x0), y=sy(box.y0), bw=sx(box.x1)-sx(box.x0), bh=sy(box.y1)-sy(box.y0);
    ctx.fillRect(x,y,bw,bh); ctx.strokeRect(x,y,bw,bh);
  }}
  for (const p of points) {{
    const t=(p[field]-vmin)/(vmax-vmin || 1);
    ctx.fillStyle=color(t, diverge);
    ctx.fillRect(sx(p.x)-2*devicePixelRatio, sy(p.y)-2*devicePixelRatio, 4*devicePixelRatio, 4*devicePixelRatio);
  }}
  ctx.fillStyle='#17202a'; ctx.font=`${{12*devicePixelRatio}}px sans-serif`;
  ctx.fillText('x (um)', pad+w-38*devicePixelRatio, pad+h+32*devicePixelRatio);
  ctx.save(); ctx.translate(22*devicePixelRatio,pad+80*devicePixelRatio); ctx.rotate(-Math.PI/2); ctx.fillText('depth from Si top (um)',0,0); ctx.restore();
  const summary=payload.data.cases[caseName].summary;
  const maxBar=90*devicePixelRatio;
  const left=Math.abs(summary.left_photo_delta_a_per_cm), right=Math.abs(summary.right_photo_delta_a_per_cm);
  const scale=maxBar/Math.max(left,right,1e-30);
  ctx.fillText('photo current split', pad+w-150*devicePixelRatio, pad+22*devicePixelRatio);
  ctx.fillStyle='#4e79a7'; ctx.fillRect(pad+w-145*devicePixelRatio,pad+34*devicePixelRatio,left*scale,10*devicePixelRatio);
  ctx.fillStyle='#f28e2b'; ctx.fillRect(pad+w-145*devicePixelRatio,pad+50*devicePixelRatio,right*scale,10*devicePixelRatio);
}}
function updateMetrics() {{
  const s = payload.data.cases[caseSelect.value || cases[0]].summary;
  document.getElementById('metrics').innerHTML = `
    <h3>Current Metrics</h3>
    <div class="metric"><span>Left photo delta</span><b>${{Number(s.left_photo_delta_a_per_cm).toExponential(4)}} A/cm</b></div>
    <div class="metric"><span>Right photo delta</span><b>${{Number(s.right_photo_delta_a_per_cm).toExponential(4)}} A/cm</b></div>
    <div class="metric"><span>Total photo delta</span><b>${{Number(s.total_photo_delta_a_per_cm).toExponential(4)}} A/cm</b></div>
    <div class="metric"><span>Split phase x</span><b>${{Number(s.photo_split_phase_x_proxy).toPrecision(5)}}</b></div>
    <div class="metric"><span>Terminal balance</span><b>${{Number(s.terminal_balance_illuminated_a_per_cm).toExponential(3)}} A/cm</b></div>`;
}}
document.getElementById('legend').innerHTML = payload.boxes.map(b=>`<div class="legendItem" title="${{b.name}}"><span class="swatch" style="background:${{b.color}}"></span><span>${{b.name}}</span></div>`).join('');
caseSelect.onchange=()=>{{updateMetrics();draw();}}; fieldSelect.onchange=draw; window.onresize=resize; updateMetrics(); resize();
</script></body></html>
""",
        encoding="utf-8",
    )


def generation_slice_data(volume_npz: Path) -> dict[str, Any]:
    with np.load(volume_npz, allow_pickle=False) as data:
        x = np.asarray(data["x_um"], dtype=float)
        y = np.asarray(data["depth_um_from_si_top"], dtype=float)
        z = np.asarray(data["z_um"], dtype=float)
        cases = np.asarray(data["case"]).astype(str)
        generation = np.asarray(data["generation_cm3_s"], dtype=float)
        z_index = int(np.argmin(np.abs(z)))
        case_payload = {}
        for index, case in enumerate(cases):
            points = []
            slice_values = generation[index, :, :, z_index]
            for ix, xv in enumerate(x):
                for iy, yv in enumerate(y):
                    value = float(slice_values[ix, iy])
                    points.append({"x": float(xv), "y": float(yv), "z": float(z[z_index]), "value": value, "log10": math.log10(max(value, 1.0))})
            case_payload[case] = points
        return {"cases": case_payload, "z_slice_um": float(z[z_index])}


def write_3d_viewer(path: Path, boxes: list[dict[str, Any]], slice_data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"boxes": boxes, "slice": slice_data})
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TCAD 3D Design Viewer</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#17202a;background:#f7f9fb}}
header{{height:52px;display:flex;align-items:center;gap:16px;padding:0 18px;background:white;border-bottom:1px solid #d7dde5}}
main{{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,360px);height:calc(100vh - 53px)}}
canvas{{width:100%;height:100%;display:block;background:#ffffff}}
aside{{border-left:1px solid #d7dde5;background:#fbfcfd;padding:16px;overflow-y:auto;overflow-x:hidden;min-width:0}}
label{{display:block;font-size:12px;font-weight:600;margin-top:12px;color:#4d5b6a}}
select,input{{width:100%;margin-top:4px}}
#legend{{display:grid;gap:6px}}
.legendItem{{display:flex;align-items:flex-start;gap:6px;min-width:0;font-size:13px;line-height:1.25;overflow-wrap:anywhere}}
.swatch{{display:inline-block;flex:0 0 auto;width:10px;height:10px;margin-top:3px;border:1px solid #3333}}
</style></head><body>
<header><strong>TCAD 3D Geometry + Field Slice</strong><span>Microlens / CFA / Si / DTI / TG / FD + OpticalGeneration slice</span></header>
<main><canvas id="canvas"></canvas><aside>
<label>Case</label><select id="case"></select>
<label>Scale</label><input id="scale" type="range" min="70" max="180" value="120">
<p>Drag on the canvas to rotate. The colored slice is log10 optical generation at z≈0.</p>
<h3>Geometry</h3><div id="legend"></div>
</aside></main>
<script>
const payload={payload};
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const caseSelect=document.getElementById('case'), scaleInput=document.getElementById('scale');
const cases=Object.keys(payload.slice.cases); for (const c of cases){{const o=document.createElement('option');o.value=c;o.textContent=c;caseSelect.appendChild(o);}}
let ax=-0.7, az=0.72, dragging=false, last=null;
function resize(){{canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;draw();}}
function color(t){{t=Math.max(0,Math.min(1,t));return `rgb(${{Math.round(40+215*t)}},${{Math.round(80+120*Math.sin(Math.PI*t))}},${{Math.round(190-160*t)}})`;}}
function project(p){{
  let x=p.x, y=p.y, z=p.z;
  let cz=Math.cos(az), sz=Math.sin(az), cx=Math.cos(ax), sx=Math.sin(ax);
  let x1=x*cz-z*sz, z1=x*sz+z*cz;
  let y1=y*cx-z1*sx, z2=y*sx+z1*cx;
  const s=Number(scaleInput.value)*devicePixelRatio;
  return [canvas.width/2+x1*s, canvas.height/2+(y1-0.9)*s, z2];
}}
function edge(a,b,color){{const A=project(a),B=project(b);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();}}
function shouldLabelBox(b){{return ['microlens','CFA','Si','TG barrier','FD n+','L DTI','R DTI'].includes(b.display_name);}}
function drawBox(b){{const c=b.color;ctx.lineWidth=1.4*devicePixelRatio;const x0=b.x0,x1=b.x1,y0=b.y0,y1=b.y1,z0=b.z0,z1=b.z1;
const pts=[{{x:x0,y:y0,z:z0}},{{x:x1,y:y0,z:z0}},{{x:x1,y:y1,z:z0}},{{x:x0,y:y1,z:z0}},{{x:x0,y:y0,z:z1}},{{x:x1,y:y0,z:z1}},{{x:x1,y:y1,z:z1}},{{x:x0,y:y1,z:z1}}];
[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(e=>edge(pts[e[0]],pts[e[1]],c));
if (shouldLabelBox(b)) {{const m=project({{x:(x0+x1)/2,y:(y0+y1)/2,z:(z0+z1)/2}});ctx.fillStyle=c;ctx.fillText(b.display_name,m[0]+3,m[1]-3);}}}}
function draw(){{
 ctx.clearRect(0,0,canvas.width,canvas.height); ctx.font=`${{11*devicePixelRatio}}px sans-serif`;
 const pts=payload.slice.cases[caseSelect.value||cases[0]]; const vals=pts.map(p=>p.log10); const vmin=Math.min(...vals), vmax=Math.max(...vals);
 const sorted=[...pts].sort((a,b)=>project({{x:a.x,y:a.y,z:a.z}})[2]-project({{x:b.x,y:b.y,z:b.z}})[2]);
 for (const p of sorted){{const q=project(p);ctx.fillStyle=color((p.log10-vmin)/(vmax-vmin||1));ctx.fillRect(q[0]-3*devicePixelRatio,q[1]-3*devicePixelRatio,6*devicePixelRatio,6*devicePixelRatio);}}
 for (const b of payload.boxes) drawBox(b);
 ctx.fillStyle='#17202a';ctx.fillText(`slice z = ${{payload.slice.z_slice_um.toFixed(3)}} um`,18*devicePixelRatio,24*devicePixelRatio);
}}
canvas.onmousedown=e=>{{dragging=true;last=[e.clientX,e.clientY];}}; window.onmouseup=()=>dragging=false; window.onmousemove=e=>{{if(!dragging)return;az+=(e.clientX-last[0])*0.01;ax+=(e.clientY-last[1])*0.01;last=[e.clientX,e.clientY];draw();}};
caseSelect.onchange=draw;scaleInput.oninput=draw;window.onresize=resize;
document.getElementById('legend').innerHTML=payload.boxes.map(b=>`<div class="legendItem" title="${{b.name}}"><span class="swatch" style="background:${{b.color}}"></span><span>${{b.name}}</span></div>`).join('');
resize();
</script></body></html>
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    vtk_dir = output_dir / "vtk"
    viewer_dir = output_dir / "viewers"
    output_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    split_summaries = discover_split_summaries(args.split_summary)
    split_summaries = [
        path for path in split_summaries if (path.parent / "node_profile_2d.csv").exists()
    ]
    split_summaries = prefer_native_split_summaries(split_summaries, by_case_only=True)
    for summary_path in split_summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        case = summary.get("config", {}).get("generation_profile_case", summary_path.parent.name)
        run_id = summary_path.parent.name
        tecplot = summary.get("outputs", {}).get("device_tecplot")
        if tecplot and Path(tecplot).exists():
            mesh = parse_tecplot_block(Path(tecplot))
            stem = sanitize_name(f"{run_id}_{case}_native_split2d")
        else:
            mesh = split_profile_to_mesh(summary_path.parent / "node_profile_2d.csv")
            stem = sanitize_name(f"{run_id}_{case}_delaunay_split2d")
        vtu = vtk_dir / f"{stem}.vtu"
        vtk = vtk_dir / f"{stem}.vtk"
        write_vtu(vtu, mesh)
        write_legacy_vtk(vtk, mesh, stem)
        exports[f"{stem}_vtu"] = str(vtu)
        exports[f"{stem}_vtk"] = str(vtk)

    for label, tecplot, map_path, volume_path in (
        ("gmsh_reference_2d_devsim", args.gmsh_2d_tecplot, args.generation_map_npz, None),
        ("gmsh_reference_3d_devsim", args.gmsh_3d_tecplot, None, args.generation_volume_npz),
    ):
        if tecplot and tecplot.exists():
            mesh = parse_tecplot_block(tecplot)
            add_generation_to_mesh(mesh, map_path, volume_path)
            vtu = vtk_dir / f"{label}.vtu"
            vtk = vtk_dir / f"{label}.vtk"
            write_vtu(vtu, mesh)
            write_legacy_vtk(vtk, mesh, label)
            exports[f"{label}_vtu"] = str(vtu)
            exports[f"{label}_vtk"] = str(vtk)

    boxes = profile_geometry_boxes(args.profile, args.stack_config)
    if split_summaries:
        write_2d_viewer(viewer_dir / "cross_section_2d.html", build_2d_case_data(split_summaries), boxes)
        exports["cross_section_2d_html"] = str(viewer_dir / "cross_section_2d.html")
    if args.generation_volume_npz.exists():
        write_3d_viewer(viewer_dir / "geometry_3d.html", boxes, generation_slice_data(args.generation_volume_npz))
        exports["geometry_3d_html"] = str(viewer_dir / "geometry_3d.html")

    report_summaries = discover_split_summaries(args.report_summary)
    report_summaries = prefer_native_split_summaries(report_summaries, by_case_only=False)
    rows = [summary_row_from_split(path) for path in report_summaries]
    add_cra_metadata(rows, args.generation_map_npz)
    rows.sort(key=lambda row: (str(row.get("case", "")), float(row.get("generation_map_scale") or 1.0), str(row.get("run_id", ""))))
    report_exports = write_sweep_report(output_dir, rows)
    exports.update({f"report_{key}": value for key, value in report_exports.items()})

    manifest = {
        "schema": "tcad_design_viewer_manifest_v1",
        "profile": str(args.profile),
        "stack_config": str(args.stack_config),
        "output_dir": str(output_dir),
        "split_summary_count": len(split_summaries),
        "parameter_report_row_count": len(rows),
        "exports": exports,
        "notes": [
            "VTK/VTU files are dependency-free ASCII exports for ParaView or compatible viewers.",
            "Split-PD VTK/VTU exports use DEVSIM device_tecplot solver connectivity when the source summary provides it; older CSV-only summaries fall back to Delaunay visualization meshes.",
            "HTML viewers are self-contained browser views for design iteration, not calibrated accuracy evidence.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=ROOT / "measured_profiles/reference_cmos_ppd_1p4um/profile.json")
    parser.add_argument("--stack-config", type=Path, default=ROOT / "configs/sensor_stack_proxy_1p4um.json")
    parser.add_argument("--generation-map-npz", type=Path, default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz")
    parser.add_argument("--generation-volume-npz", type=Path, default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_volume_3d.npz")
    parser.add_argument("--gmsh-2d-tecplot", type=Path, default=ROOT / "runs/devsim_gmsh_reference_import_2d/gmsh_pixel_2d_potential.dat")
    parser.add_argument("--gmsh-3d-tecplot", type=Path, default=ROOT / "runs/devsim_gmsh_reference_import_3d/gmsh_pixel_3d_potential.dat")
    parser.add_argument(
        "--split-summary",
        type=Path,
        action="append",
        default=[
            ROOT / "runs/devsim_split_pd_2d_reference_profile_center_smoke/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_edge20x_smoke/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_center_gmsh_native/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native/summary.json",
        ],
    )
    parser.add_argument(
        "--report-summary",
        type=Path,
        action="append",
        default=[
            ROOT / "runs/devsim_split_pd_2d_reference_profile_center_smoke/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_edge20x_smoke/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_center_gmsh_native/summary.json",
            ROOT / "runs/devsim_split_pd_2d_reference_profile_edge20x_gmsh_native/summary.json",
            ROOT / "runs/tcad_calibration_reference_profile/evals",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/tcad_design_viewer_reference")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
