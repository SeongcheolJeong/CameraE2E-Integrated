#!/usr/bin/env python3
"""Build and optionally run local high-resolution FDTD boundary patches.

Meep uses a uniform Cartesian Yee grid, so this is not local adaptive mesh
refinement inside a full simulation. The intent is to crop small OCL/CFA
boundary neighborhoods from the CAD-derived template geometry, run them at a
higher grid resolution than the full CRA sweep, and use the result as boundary
convergence evidence or a correction input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from boundary_continuity_analysis import (
    ROOT,
    DEFAULT_TEMPLATE_ROOT,
    Box,
    adjacency_pairs,
    compact,
    ocl_boxes,
    read_json,
    repo_rel,
    signed_gap,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_DIR = ROOT / "runs" / "boundary_patch_analysis"
DEFAULT_TEMPLATES = "mixed_1x1_2x2_3x3_boundary,nona_3x3_ocl,quad_2x2_ocl_5x5_crosstalk"
DEFAULT_STACK_CONFIG = ROOT / "configs" / "sensor_stack_proxy_1p4um.json"
DEFAULT_MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def polygon_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    zs = [float(point[1]) for point in points]
    return min(xs), max(xs), min(zs), max(zs)


def point_in_polygon(x: float, z: float, points: list[list[float]]) -> bool:
    inside = False
    count = len(points)
    for index in range(count):
        x0, z0 = float(points[index][0]), float(points[index][1])
        x1, z1 = float(points[(index + 1) % count][0]), float(points[(index + 1) % count][1])
        if abs((x - x0) * (z1 - z0) - (z - z0) * (x1 - x0)) < 1.0e-12:
            dot = (x - x0) * (x1 - x0) + (z - z0) * (z1 - z0)
            length2 = (x1 - x0) ** 2 + (z1 - z0) ** 2
            if -1.0e-12 <= dot <= length2 + 1.0e-12:
                return True
        crosses = (z0 > z) != (z1 > z)
        if crosses:
            x_intersect = (x1 - x0) * (z - z0) / (z1 - z0) + x0
            if x < x_intersect:
                inside = not inside
    return inside


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("count must be at least 2")
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def lens_height_um(block: dict[str, Any], geometry: dict[str, Any], params: dict[str, Any]) -> float:
    if block.get("height_um") not in (None, ""):
        return float(block["height_um"])
    lens_id = str(block["lens_id"])
    lens_params = geometry.get("ocl_lens_parameters", {}).get(lens_id, {})
    if lens_params.get("height_um") not in (None, ""):
        return float(lens_params["height_um"])
    return float(params["lens_height_um"])


def spherical_surface_map(
    points: list[list[float]],
    height_um: float,
    *,
    grid_count: int,
) -> dict[str, Any]:
    xmin, xmax, zmin, zmax = polygon_bbox(points)
    x_values = linspace(xmin, xmax, grid_count)
    z_values = linspace(zmin, zmax, grid_count)
    aperture_radius = 0.5 * min(xmax - xmin, zmax - zmin)
    if height_um <= 0.0 or aperture_radius <= 0.0:
        rows = [[0.0 for _ in x_values] for _ in z_values]
    else:
        sphere_radius = (aperture_radius * aperture_radius + height_um * height_um) / (2.0 * height_um)
        rows = []
        for z in z_values:
            row = []
            for x in x_values:
                radius = math.hypot(x, z)
                if radius > aperture_radius or not point_in_polygon(x, z, points):
                    row.append(0.0)
                    continue
                row.append(
                    max(
                        height_um
                        - sphere_radius
                        + math.sqrt(max(sphere_radius * sphere_radius - radius * radius, 0.0)),
                        0.0,
                    )
                )
            rows.append(row)
    return {
        "x_um": x_values,
        "z_um": z_values,
        "height_um": rows,
        "source": "boundary_patch_spherical_cap_from_cad_template",
    }


def block_map(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(block["lens_id"]): block for block in params.get("ocl_blocks", [])}


def contained_blocks(
    params: dict[str, Any],
    ix0: int,
    ix1: int,
    iz0: int,
    iz1: int,
) -> list[dict[str, Any]]:
    output = []
    for block in params.get("ocl_blocks", []):
        bix0 = int(block["ix"])
        biz0 = int(block["iz"])
        bix1 = bix0 + int(block["sx"])
        biz1 = biz0 + int(block["sz"])
        if bix0 >= ix0 and bix1 <= ix1 and biz0 >= iz0 and biz1 <= iz1:
            output.append(dict(block))
    return output


def local_cfa_cells(geometry: dict[str, Any], ix0: int, ix1: int, iz0: int, iz1: int) -> list[dict[str, Any]]:
    cells = []
    for cell in geometry.get("cfa_polygons", {}).get("cells", []):
        ix = int(cell["ix"])
        iz = int(cell["iz"])
        if ix0 <= ix < ix1 and iz0 <= iz < iz1:
            local = dict(cell)
            local["ix"] = ix - ix0
            local["iz"] = iz - iz0
            local["id"] = safe_id(f"{cell.get('id', 'cell')}_local_{local['ix']}_{local['iz']}")
            local["source"] = "boundary patch crop from " + str(cell.get("source", "template"))
            cells.append(local)
    return cells


def layout_descriptor(blocks: list[dict[str, Any]]) -> str:
    return ",".join(
        f"{block['lens_id']}:{int(block['ix'])}:{int(block['iz'])}:{int(block['sx'])}:{int(block['sz'])}"
        for block in blocks
    )


def stack_geometry(path: Path) -> dict[str, float]:
    payload = read_json(path)
    geometry = payload.get("geometry_um", {})
    defaults = {
        "pitch": 1.4,
        "pml": 0.45,
        "air_top": 0.55,
        "lens_height": 0.657,
        "cfa_thickness": 0.8,
        "passivation_thickness": 0.08,
        "si_thickness": 2.8,
        "bottom_air": 0.25,
    }
    return {key: float(geometry.get(key, value)) for key, value in defaults.items()}


def snapped_cell_y_um(cell_y_um: float, resolution: int, mode: str) -> float:
    if mode == "nearest":
        return round(cell_y_um * resolution) / resolution
    if mode == "ceil":
        return math.ceil(cell_y_um * resolution - 1.0e-12) / resolution
    if mode == "floor":
        return math.floor(cell_y_um * resolution + 1.0e-12) / resolution
    return cell_y_um


def yee_estimate(nx: int, nz: int, resolution: int, stack_config: Path, pml_um: float, grid_snap_y: str) -> dict[str, Any]:
    geometry = stack_geometry(stack_config)
    geometry["pml"] = pml_um
    cell_y = (
        2.0 * geometry["pml"]
        + geometry["air_top"]
        + geometry["lens_height"]
        + geometry["cfa_thickness"]
        + geometry["passivation_thickness"]
        + geometry["si_thickness"]
        + geometry["bottom_air"]
    )
    cell_y = snapped_cell_y_um(cell_y, resolution, grid_snap_y)
    x_pixels = max(1, round(nx * geometry["pitch"] * resolution))
    y_pixels = max(1, round(cell_y * resolution))
    z_pixels = max(1, round(nz * geometry["pitch"] * resolution))
    cells = x_pixels * y_pixels * z_pixels
    return {
        "layout_nx": nx,
        "layout_nz": nz,
        "resolution_px_per_um": resolution,
        "cell_y_um": cell_y,
        "x_pixels": x_pixels,
        "y_pixels": y_pixels,
        "z_pixels": z_pixels,
        "estimated_yee_cells": cells,
        "estimated_yee_cells_million": cells / 1.0e6,
    }


def build_command(
    *,
    meep_python: Path,
    patch: dict[str, Any],
    geometry_path: Path,
    output_dir: Path,
    resolution: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(meep_python),
        "meep_supercell_lut.py",
        "--mode",
        "ocl-layout",
        "--layout-nx",
        str(patch["layout_nx"]),
        "--layout-nz",
        str(patch["layout_nz"]),
        "--ocl-layout",
        patch["ocl_layout"],
        "--ocl-layout-name",
        patch["patch_id"],
        "--ocl-polygons",
        "@" + repo_rel(geometry_path),
        "--ocl-surface-map",
        "@" + repo_rel(geometry_path),
        "--cfa-polygons",
        "@" + repo_rel(geometry_path),
        "--target-lens-id",
        patch["target_lens_id"],
        "--source-aperture-lens-id",
        "target",
        "--collection-mode",
        "pixel",
        "--cfa-pattern",
        str(patch["cfa_pattern"]),
        "--color-channel",
        args.color_channel,
        "--wavelengths-nm",
        args.wavelengths_nm,
        "--cases",
        args.cases,
        "--resolution",
        str(resolution),
        "--after-source-time",
        str(args.after_source_time),
        "--pml-um",
        str(args.pml_um),
        "--grid-snap-y",
        args.grid_snap_y,
        "--stack-config",
        repo_rel(args.stack_config),
        "--output-dir",
        repo_rel(output_dir),
    ]


def parse_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "camera_lut_summary.csv"
    long_path = output_dir / "camera_lut_long.csv"
    if not summary_path.exists():
        return {"summary_status": "MISSING", "summary_csv": repo_rel(summary_path)}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with long_path.open(newline="", encoding="utf-8") as handle:
        long_rows = list(csv.DictReader(handle)) if long_path.exists() else []
    responses = []
    for row in rows:
        try:
            responses.append(float(row.get("total_response", "nan")))
        except ValueError:
            pass
    grid_pass = [row.get("grid_resolution_gate_pass") == "True" for row in rows]
    source_enabled = [row.get("source_aperture_enabled") == "True" for row in rows]
    return {
        "summary_status": "PASS" if rows else "EMPTY",
        "summary_csv": repo_rel(summary_path),
        "long_csv": repo_rel(long_path) if long_path.exists() else "",
        "response_maps": repo_rel(output_dir / "response_maps.png") if (output_dir / "response_maps.png").exists() else "",
        "focal_maps": repo_rel(output_dir / "focal_maps.png") if (output_dir / "focal_maps.png").exists() else "",
        "case_count": len(rows),
        "region_row_count": len(long_rows),
        "grid_gate_pass": all(grid_pass) if grid_pass else False,
        "source_aperture_enabled": all(source_enabled) if source_enabled else False,
        "mean_total_response": sum(responses) / len(responses) if responses else None,
        "min_total_response": min(responses) if responses else None,
        "max_total_response": max(responses) if responses else None,
        "first_row": rows[0] if rows else {},
    }


def run_command(command: list[str], cwd: Path, output_dir: Path, timeout_s: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False)
        elapsed = time.time() - started
        (output_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
        parsed = parse_summary(output_dir)
        if result.returncode == 0:
            run_status = "PASS"
        elif result.returncode < 0:
            run_status = "INTERRUPTED"
        else:
            run_status = "FAIL"
        return {
            "run_status": run_status,
            "returncode": result.returncode,
            "elapsed_s": elapsed,
            "stdout_log": repo_rel(output_dir / "stdout.log"),
            "stderr_log": repo_rel(output_dir / "stderr.log"),
            **parsed,
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.time() - started
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(stderr + f"\nTIMEOUT after {timeout_s} s\n", encoding="utf-8")
        return {
            "run_status": "TIMEOUT",
            "returncode": None,
            "elapsed_s": elapsed,
            "stdout_log": repo_rel(output_dir / "stdout.log"),
            "stderr_log": repo_rel(output_dir / "stderr.log"),
            **parse_summary(output_dir),
        }


def make_patch(
    template_id: str,
    params: dict[str, Any],
    geometry: dict[str, Any],
    first: Box,
    second: Box,
    axis: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    blocks_by_id = block_map(params)
    first_block = blocks_by_id[first.item_id]
    second_block = blocks_by_id[second.item_id]
    ix0 = min(int(first_block["ix"]), int(second_block["ix"]))
    iz0 = min(int(first_block["iz"]), int(second_block["iz"]))
    ix1 = max(int(first_block["ix"]) + int(first_block["sx"]), int(second_block["ix"]) + int(second_block["sx"]))
    iz1 = max(int(first_block["iz"]) + int(first_block["sz"]), int(second_block["iz"]) + int(second_block["sz"]))
    included = contained_blocks(params, ix0, ix1, iz0, iz1)
    local_blocks = []
    for block in included:
        local = dict(block)
        local["ix"] = int(local["ix"]) - ix0
        local["iz"] = int(local["iz"]) - iz0
        local_blocks.append(local)
    local_blocks.sort(key=lambda item: (int(item["iz"]), int(item["ix"]), str(item["lens_id"])))

    ocl_polygons = {
        str(block["lens_id"]): geometry.get("ocl_polygons", {})[str(block["lens_id"])]
        for block in local_blocks
        if str(block["lens_id"]) in geometry.get("ocl_polygons", {})
    }
    surface_maps = {}
    for block in local_blocks:
        lens_id = str(block["lens_id"])
        if lens_id not in ocl_polygons:
            continue
        surface_maps[lens_id] = spherical_surface_map(
            ocl_polygons[lens_id],
            lens_height_um(block, geometry, params),
            grid_count=args.surface_map_grid,
        )

    cfa_cells = local_cfa_cells(geometry, ix0, ix1, iz0, iz1)
    patch_id = safe_id(f"{template_id}__{first.item_id}__{second.item_id}__{axis}")
    first_kind = f"{first.sx}x{first.sz}"
    second_kind = f"{second.sx}x{second.sz}"
    height_delta = abs(lens_height_um(first_block, geometry, params) - lens_height_um(second_block, geometry, params))
    mixed_transition = first_kind != second_kind or height_delta > 1.0e-9
    layout_nx = ix1 - ix0
    layout_nz = iz1 - iz0
    target_lens_id = first.item_id if lens_height_um(first_block, geometry, params) >= lens_height_um(second_block, geometry, params) else second.item_id
    patch_geometry = {
        "schema": "boundary_patch_geometry_import_v1",
        "units": "um",
        "source": {
            "template_id": template_id,
            "template_geometry": repo_rel(args.template_root / template_id / "geometry_import.json"),
            "template_parameters": repo_rel(args.template_root / template_id / "template_parameters.json"),
        },
        "patch": {
            "patch_id": patch_id,
            "pair": [first.item_id, second.item_id],
            "axis": axis,
            "original_pixel_bbox": {"ix0": ix0, "ix1": ix1, "iz0": iz0, "iz1": iz1},
            "layout_nx": layout_nx,
            "layout_nz": layout_nz,
            "included_lens_ids": [str(block["lens_id"]) for block in local_blocks],
            "target_lens_id": target_lens_id,
            "mixed_transition": mixed_transition,
            "signed_gap_um": signed_gap(first, second, axis),
        },
        "notes": [
            "This patch preserves CAD-derived local OCL and CFA polygons for a cropped boundary neighborhood.",
            "OCL surface maps are generated as spherical caps using per-lens CAD heights so mixed 1x1/2x2/3x3 height ratios are preserved.",
            "This is a local high-resolution FDTD patch, not a full-domain adaptive mesh refinement run.",
        ],
        "ocl_polygons": ocl_polygons,
        "ocl_surface_maps": surface_maps,
        "cfa_polygons": {
            "background": geometry.get("cfa_polygons", {}).get("background", "passivation"),
            "cells": cfa_cells,
        },
    }
    return {
        "patch_id": patch_id,
        "crop_key": f"{template_id}:{ix0}:{ix1}:{iz0}:{iz1}:{target_lens_id}",
        "covered_pairs": [f"{first.item_id}->{second.item_id}:{axis}"],
        "template_id": template_id,
        "pair_first": first.item_id,
        "pair_second": second.item_id,
        "axis": axis,
        "signed_gap_um": signed_gap(first, second, axis),
        "first_kind": first_kind,
        "second_kind": second_kind,
        "first_height_um": lens_height_um(first_block, geometry, params),
        "second_height_um": lens_height_um(second_block, geometry, params),
        "height_delta_um": height_delta,
        "mixed_transition": mixed_transition,
        "layout_nx": layout_nx,
        "layout_nz": layout_nz,
        "local_lens_count": len(local_blocks),
        "local_cfa_cell_count": len(cfa_cells),
        "target_lens_id": target_lens_id,
        "cfa_pattern": params.get("cfa_pattern", "uniform"),
        "ocl_layout": layout_descriptor(local_blocks),
        "patch_geometry": patch_geometry,
    }


def collect_patches(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_patches = []
    for template_id in [item.strip() for item in args.templates.split(",") if item.strip()]:
        root = args.template_root / template_id
        params = read_json(root / "template_parameters.json")
        geometry = read_json(root / "geometry_import.json")
        boxes = ocl_boxes(params, geometry)
        for first, second, axis in adjacency_pairs(boxes):
            raw_patches.append(make_patch(template_id, params, geometry, first, second, axis, args))
    patches_by_crop: dict[str, dict[str, Any]] = {}
    for patch in raw_patches:
        key = patch["crop_key"]
        if key in patches_by_crop:
            patches_by_crop[key]["covered_pairs"].extend(patch["covered_pairs"])
            continue
        patches_by_crop[key] = patch
    patches = list(patches_by_crop.values())
    for patch in patches:
        patch["low_resolution_estimate"] = yee_estimate(
            patch["layout_nx"],
            patch["layout_nz"],
            args.low_resolution,
            args.stack_config,
            args.pml_um,
            args.grid_snap_y,
        )
        patch["high_resolution_estimate"] = yee_estimate(
            patch["layout_nx"],
            patch["layout_nz"],
            args.high_resolution,
            args.stack_config,
            args.pml_um,
            args.grid_snap_y,
        )
    patches.sort(
        key=lambda item: (
            not item["mixed_transition"] if args.prefer_mixed else False,
            item["high_resolution_estimate"]["estimated_yee_cells_million"],
            item["template_id"],
            item["patch_id"],
        )
    )
    return patches[: args.max_patches] if args.max_patches else patches


def patch_summary_row(patch: dict[str, Any]) -> dict[str, Any]:
    low_run = patch.get("low_run", {})
    high_run = patch.get("high_run", {})
    return {
        "patch_id": patch["patch_id"],
        "template_id": patch["template_id"],
        "pair": f"{patch['pair_first']} -> {patch['pair_second']}",
        "covered_pairs": "; ".join(patch.get("covered_pairs", [])),
        "axis": patch["axis"],
        "mixed_transition": patch["mixed_transition"],
        "layout": f"{patch['layout_nx']}x{patch['layout_nz']}",
        "local_lens_count": patch["local_lens_count"],
        "target_lens_id": patch["target_lens_id"],
        "signed_gap_um": patch["signed_gap_um"],
        "height_delta_um": patch["height_delta_um"],
        "low_res": patch["low_resolution_estimate"]["resolution_px_per_um"],
        "low_cells_m": patch["low_resolution_estimate"]["estimated_yee_cells_million"],
        "low_run": low_run.get("run_status", patch.get("low_status", "not run")),
        "low_grid_gate": low_run.get("grid_gate_pass", ""),
        "high_res": patch["high_resolution_estimate"]["resolution_px_per_um"],
        "high_cells_m": patch["high_resolution_estimate"]["estimated_yee_cells_million"],
        "high_run": high_run.get("run_status", patch.get("high_status", "not run")),
        "high_grid_gate": high_run.get("grid_gate_pass", ""),
        "geometry": patch.get("geometry_path", ""),
        "manifest": patch.get("manifest_path", ""),
    }


def table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            text = compact(value)
            if column in {"geometry", "manifest"} and text:
                cells.append(f"<td><a href=\"../../{escape(text)}\">{escape(Path(text).name)}</a></td>")
            else:
                cells.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, report: dict[str, Any]) -> None:
    rows = report["summary_rows"]
    command_rows = []
    for patch in report["patches"]:
        for tier in ("low", "high"):
            command_rows.append(
                {
                    "patch_id": patch["patch_id"],
                    "tier": tier,
                    "status": patch.get(f"{tier}_status", "not run"),
                    "command": " ".join(patch.get(f"{tier}_command", [])),
                }
            )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boundary Patch FDTD Analysis</title>
  <style>
    body {{ margin:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#061017; color:#e8f6fa; }}
    h1,h2 {{ margin:0 0 10px; }}
    p,li {{ color:#a9c2cc; line-height:1.55; }}
    section {{ margin-top:18px; border:1px solid #24495a; border-radius:12px; padding:16px; background:#0d1b24; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border-bottom:1px solid #24495a; padding:7px; text-align:left; vertical-align:top; }}
    th {{ color:#58d9ee; }}
    code {{ color:#f7db70; }}
    a {{ color:#66d9ef; }}
    .pill {{ display:inline-block; border:1px solid #35758a; border-radius:999px; padding:4px 8px; margin-right:6px; }}
  </style>
</head>
<body>
  <h1>Boundary Patch FDTD Analysis</h1>
  <p>This report creates local high-resolution Meep patches around CAD-derived OCL/CFA boundaries. It is not adaptive mesh refinement in a full-domain run; it is a cropped FDTD experiment used for boundary convergence evidence and correction development.</p>
  <section>
    <h2>Run Settings</h2>
    <p>
      <span class="pill">low {escape(str(report['settings']['low_resolution']))} px/um</span>
      <span class="pill">high {escape(str(report['settings']['high_resolution']))} px/um</span>
      <span class="pill">max high cells {escape(str(report['settings']['max_yee_cells_million']))}M</span>
      <span class="pill">patches {escape(str(len(report['patches'])))}</span>
    </p>
    <p>Use scope: <code>boundary convergence / correction input</code>. Full camera-system LUT still needs full-domain CRA/crosstalk sweeps plus measured stack/material calibration.</p>
  </section>
  <section>
    <h2>Patch Summary</h2>
    {table_html(rows)}
  </section>
  <section>
    <h2>Commands</h2>
    {table_html(command_rows)}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patches = collect_patches(args)
    for patch_index, patch in enumerate(patches):
        patch_dir = args.output_dir / patch["patch_id"]
        patch_dir.mkdir(parents=True, exist_ok=True)
        geometry_path = patch_dir / "geometry_import_patch.json"
        write_json(geometry_path, patch["patch_geometry"])
        patch["geometry_path"] = repo_rel(geometry_path)

        low_dir = patch_dir / f"meep_res_{args.low_resolution}"
        high_dir = patch_dir / f"meep_res_{args.high_resolution}"
        low_command = build_command(
            meep_python=args.meep_python,
            patch=patch,
            geometry_path=geometry_path,
            output_dir=low_dir,
            resolution=args.low_resolution,
            args=args,
        )
        high_command = build_command(
            meep_python=args.meep_python,
            patch=patch,
            geometry_path=geometry_path,
            output_dir=high_dir,
            resolution=args.high_resolution,
            args=args,
        )
        patch["low_command"] = low_command
        patch["high_command"] = high_command
        patch["low_status"] = "not requested"
        patch["high_status"] = "not requested"

        run_this_patch = args.run_patch_limit <= 0 or patch_index < args.run_patch_limit
        if args.run_low and run_this_patch:
            patch["low_status"] = "running"
            patch["low_run"] = run_command(low_command, ROOT, low_dir, args.timeout_s)
            patch["low_status"] = patch["low_run"]["run_status"]
        elif args.run_low:
            patch["low_status"] = "SKIPPED_BY_RUN_PATCH_LIMIT"
        if args.run_high and run_this_patch:
            estimate_m = patch["high_resolution_estimate"]["estimated_yee_cells_million"]
            if estimate_m > args.max_yee_cells_million:
                patch["high_status"] = "SKIPPED_TOO_LARGE"
                patch["high_skip_reason"] = (
                    f"Estimated {estimate_m:.1f}M Yee cells exceeds limit {args.max_yee_cells_million:.1f}M."
                )
            else:
                patch["high_status"] = "running"
                patch["high_run"] = run_command(high_command, ROOT, high_dir, args.timeout_s)
                patch["high_status"] = patch["high_run"]["run_status"]
        elif args.run_high:
            patch["high_status"] = "SKIPPED_BY_RUN_PATCH_LIMIT"

        manifest = {
            "schema": "boundary_patch_manifest_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patch": {key: value for key, value in patch.items() if key != "patch_geometry"},
        }
        manifest_path = patch_dir / "patch_manifest.json"
        write_json(manifest_path, manifest)
        patch["manifest_path"] = repo_rel(manifest_path)

    summary_rows = [patch_summary_row(patch) for patch in patches]
    report = {
        "schema": "boundary_patch_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "template_root": repo_rel(args.template_root),
            "templates": args.templates,
            "output_dir": repo_rel(args.output_dir),
            "stack_config": repo_rel(args.stack_config),
            "low_resolution": args.low_resolution,
            "high_resolution": args.high_resolution,
            "max_yee_cells_million": args.max_yee_cells_million,
            "run_low": args.run_low,
            "run_high": args.run_high,
            "run_patch_limit": args.run_patch_limit,
            "cases": args.cases,
            "wavelengths_nm": args.wavelengths_nm,
            "color_channel": args.color_channel,
        },
        "summary_rows": summary_rows,
        "patches": [{key: value for key, value in patch.items() if key != "patch_geometry"} for patch in patches],
        "limitations": [
            "Meep does not use local adaptive mesh refinement here; each patch is a separate uniform-grid FDTD run.",
            "Patch results do not replace the full OCL-neighborhood crosstalk kernel because boundary conditions and long-range leakage are cropped.",
            "Use patch deltas as convergence evidence or correction inputs, then validate against a full-domain sweep when feasible.",
            "Measured stack geometry and measured n,k are still required for product LUT accuracy.",
        ],
    }
    write_json(args.output_dir / "boundary_patch_report.json", report)
    write_csv(args.output_dir / "boundary_patch_summary.csv", summary_rows)
    write_html(args.output_dir / "index.html", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--templates", default=DEFAULT_TEMPLATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--meep-python", type=Path, default=DEFAULT_MEEP_PYTHON)
    parser.add_argument("--max-patches", type=int, default=8)
    parser.add_argument("--prefer-mixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-resolution", type=int, default=12)
    parser.add_argument("--high-resolution", type=int, default=60)
    parser.add_argument("--max-yee-cells-million", type=float, default=80.0)
    parser.add_argument("--surface-map-grid", type=int, default=33)
    parser.add_argument("--run-low", action="store_true")
    parser.add_argument("--run-high", action="store_true")
    parser.add_argument("--run-patch-limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--color-channel", choices=("red", "green", "blue"), default="green")
    parser.add_argument("--wavelengths-nm", default="550")
    parser.add_argument("--cases", default="center_0:0:0:0:0:0:0,cra20_x:20:0:1:0:0:0")
    parser.add_argument("--after-source-time", type=float, default=0.3)
    parser.add_argument("--pml-um", type=float, default=0.45)
    parser.add_argument("--grid-snap-y", choices=("off", "nearest", "ceil", "floor"), default="nearest")
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    output_dir = ROOT / report["settings"]["output_dir"]
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "patch_count": len(report["patches"]),
                "output_dir": report["settings"]["output_dir"],
                "summary_csv": repo_rel(output_dir / "boundary_patch_summary.csv"),
                "html": repo_rel(output_dir / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
