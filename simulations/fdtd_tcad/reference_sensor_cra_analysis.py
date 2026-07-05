#!/usr/bin/env python3
"""Run CRA/field/color analysis from reference sensor CAD templates.

This script builds on ``reference_sensor_template_pipeline.py`` outputs. It
reuses each sensor's generated CAD/FDTD command, sweeps field position, CRA, and
RGB wavelength/channel cases, then extracts camera-system-facing KPI tables:
relative QE proxy, edge response, crosstalk kernel proxy, color balance, OCL
group/binning uniformity, split-PD phase proxy, and grid gate status.

The outputs are trend/research artifacts. They are not product-accuracy LUTs
until measured stack/material data and convergence gates pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_CATALOG = ROOT / "runs" / "reference_sensor_template_analysis" / "analysis_catalog.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "reference_sensor_cra_analysis"
DEFAULT_SUPPLEMENTAL_TEMPLATE_ROOT = ROOT / "runs" / "pixel_cad_template_library_reference"
DEFAULT_SUPPLEMENTAL_TEMPLATES = "nona_3x3_ocl,mixed_1x1_2x2_3x3_boundary,quad_2x2_ocl_5x5_crosstalk"
MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"
CHANNELS = {
    "red": 650.0,
    "green": 550.0,
    "blue": 450.0,
}
FIELD_ORDER = ("center", "x-edge", "z-edge", "diagonal")
FIELD_DISPLAY = {
    "center": "center",
    "x-edge": "x-edge",
    "z-edge": "y-edge (sim z)",
    "diagonal": "diagonal",
}
CHANNEL_COLORS = {
    "red": "#ef4444",
    "green": "#22c55e",
    "blue": "#3b82f6",
}
FIELD_COLORS = {
    "center": "#54d7ee",
    "x-edge": "#facc15",
    "z-edge": "#fb7185",
    "diagonal": "#a78bfa",
}
CHART_COLORS = ("#54d7ee", "#facc15", "#22c55e", "#fb7185", "#a78bfa", "#60a5fa", "#f97316", "#14b8a6")


@dataclass(frozen=True)
class CraFieldCase:
    name: str
    field: str
    nominal_cra_deg: float
    cra_x_deg: float
    cra_z_deg: float
    field_x_norm: float
    field_z_norm: float
    lens_shift_x_um: float = 0.0
    lens_shift_z_um: float = 0.0

    def solver_token(self) -> str:
        return (
            f"{self.name}:{self.cra_x_deg:.8g}:{self.cra_z_deg:.8g}:"
            f"{self.field_x_norm:.8g}:{self.field_z_norm:.8g}:"
            f"{self.lens_shift_x_um:.8g}:{self.lens_shift_z_um:.8g}"
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def field_display(field: Any) -> str:
    return FIELD_DISPLAY.get(str(field), str(field))


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return parse_float(value)


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def grouped_mean(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    filters: dict[str, Any] | None = None,
    group_key: str = "nominal_cra_deg",
) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if filters and any(row.get(key) != value for key, value in filters.items()):
            continue
        x_value = numeric(row.get(group_key))
        y_value = numeric(row.get(metric))
        if x_value is None or y_value is None:
            continue
        grouped[x_value].append(y_value)
    points = []
    for x_value in sorted(grouped):
        y_value = mean(grouped[x_value])
        if y_value is not None:
            points.append((x_value, y_value))
    return points


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def default_cases(angles: list[float]) -> list[CraFieldCase]:
    cases: list[CraFieldCase] = []
    for angle in angles:
        cases.append(CraFieldCase(f"center_{angle:g}", "center", angle, angle, 0.0, 0.0, 0.0))
        cases.append(CraFieldCase(f"xedge_{angle:g}", "x-edge", angle, angle, 0.0, 1.0, 0.0))
        cases.append(CraFieldCase(f"zedge_{angle:g}", "z-edge", angle, 0.0, angle, 0.0, 1.0))
        if angle == 0:
            comp = 0.0
        else:
            comp = math.degrees(math.asin(math.sin(math.radians(angle)) / math.sqrt(2.0)))
        cases.append(CraFieldCase(f"diag_{angle:g}", "diagonal", angle, comp, comp, 1.0, 1.0))
    return cases


def cases_string(cases: list[CraFieldCase]) -> str:
    return ",".join(case.solver_token() for case in cases)


def field_case_lookup(cases: list[CraFieldCase]) -> dict[str, CraFieldCase]:
    return {case.name: case for case in cases}


def ocl_layout_from_template_params(params: dict[str, Any]) -> str:
    blocks = params.get("ocl_blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"template {params.get('template_id')} has no ocl_blocks")
    return ",".join(
        f"{block['lens_id']}:{block['ix']}:{block['iz']}:{block['sx']}:{block['sz']}"
        for block in blocks
    )


def central_lens_id_from_template_params(params: dict[str, Any]) -> str:
    blocks = params.get("ocl_blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"template {params.get('template_id')} has no ocl_blocks")
    nx = float(params["nx"])
    nz = float(params["nz"])
    center_x = nx / 2.0
    center_z = nz / 2.0
    central = min(
        blocks,
        key=lambda block: (
            abs((float(block["ix"]) + float(block["sx"]) / 2.0) - center_x)
            + abs((float(block["iz"]) + float(block["sz"]) / 2.0) - center_z),
            str(block["lens_id"]),
        ),
    )
    return str(central["lens_id"])


def cfa_pattern_for_template(params: dict[str, Any]) -> tuple[str, str]:
    pattern = str(params.get("cfa_pattern") or "uniform")
    if pattern.startswith("uniform_"):
        return "uniform", pattern.split("_", 1)[1]
    return pattern, "green"


def supplemental_template_record(template_id: str, template_root: Path) -> dict[str, Any]:
    template_dir = template_root / template_id
    params_path = template_dir / "template_parameters.json"
    geometry_path = template_dir / "geometry_import.json"
    if not params_path.exists():
        raise FileNotFoundError(f"missing supplemental template parameters: {params_path}")
    if not geometry_path.exists():
        raise FileNotFoundError(f"missing supplemental geometry import: {geometry_path}")
    params = read_json(params_path)
    geometry = read_json(geometry_path)
    cfa_polygon_count = len(geometry.get("cfa_polygons", {}).get("cells", []))
    cfa_pattern, color_channel = cfa_pattern_for_template(params)
    split_mode = str(params.get("split_mode") or "none").replace("_", "-")
    command = [
        str(MEEP_PYTHON),
        "meep_supercell_lut.py",
        "--mode",
        "ocl-layout",
        "--layout-nx",
        str(params["nx"]),
        "--layout-nz",
        str(params["nz"]),
        "--ocl-layout",
        ocl_layout_from_template_params(params),
        "--ocl-polygons",
        f"@{repo_rel(geometry_path)}",
        "--ocl-layout-name",
        str(template_id)[:80],
        "--target-lens-id",
        central_lens_id_from_template_params(params),
        "--cfa-pattern",
        cfa_pattern,
        "--color-channel",
        color_channel,
        "--wavelengths-nm",
        "550",
        "--cases",
        "center:0:0:0:0:0:0",
        "--resolution",
        "4",
        "--after-source-time",
        "0.3",
        "--pml-um",
        "0.45",
        "--grid-snap-y",
        "nearest",
        "--output-dir",
        str(ROOT / "runs" / "reference_sensor_cra_analysis" / "supplemental" / template_id),
    ]
    if cfa_polygon_count <= 96:
        command.extend(["--cfa-polygons", f"@{repo_rel(geometry_path)}"])
    if split_mode in {"dual-x", "dual-z", "quad"}:
        command.extend(["--split-mode", split_mode, "--collection-mode", "split-pd"])
    else:
        command.extend(["--collection-mode", "pixel"])
    shield_mode = str(params.get("shield_mode") or "off")
    if shield_mode in {"edge", "off", "pdaf_left", "pdaf_right", "pdaf_pair"}:
        command.extend(["--shield-mode", shield_mode])
    return {
        "sensor_id": f"topology_{template_id}",
        "code": template_id,
        "manufacturer": "CAD template",
        "device_name": str(params.get("label") or template_id),
        "source_template_id": template_id,
        "simulation": {"command": command},
        "notes": [
            "Supplemental topology coverage record generated from the reusable CAD template library.",
            "This is proxy parametric CAD, not a measured product sensor stack.",
        ],
    }


def supplemental_template_records(template_ids: str, template_root: Path) -> list[dict[str, Any]]:
    return [
        supplemental_template_record(template_id, template_root)
        for template_id in [item.strip() for item in template_ids.split(",") if item.strip()]
    ]


def replace_arg(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    if flag in command:
        index = command.index(flag)
        if index + 1 >= len(command):
            raise ValueError(f"Command flag {flag} has no value")
        command[index + 1] = value
    else:
        command.extend([flag, value])
    return command


def command_arg(command: list[str], flag: str, default: str = "") -> str:
    if flag not in command:
        return default
    index = command.index(flag)
    if index + 1 >= len(command):
        return default
    return command[index + 1]


def run_solver_command(command: list[str], cwd: Path, output_dir: Path, timeout_s: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "solver_stdout.log"
    stderr_path = output_dir / "solver_stderr.log"
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        status = "PASS" if result.returncode == 0 else "FAIL"
        exit_code: int | str = result.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        status = "TIMEOUT"
        exit_code = "timeout"
        error = str(exc)
    return {
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": round(time.time() - started, 3),
        "command": command,
        "stdout": repo_rel(stdout_path),
        "stderr": repo_rel(stderr_path),
        "error": error,
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bayer_color(ix: int, iz: int) -> str:
    if iz % 2 == 0:
        return "red" if ix % 2 == 0 else "green"
    return "green" if ix % 2 == 0 else "blue"


def cfa_color(ix: int, iz: int, pattern: str, fallback: str) -> str:
    if pattern == "uniform":
        return fallback
    if pattern == "bayer":
        return bayer_color(ix, iz)
    if pattern == "quad":
        return bayer_color(ix // 2, iz // 2)
    if pattern == "nona":
        return bayer_color(ix // 3, iz // 3)
    return fallback


def target_pixels(lut: dict[str, Any], nx: int, nz: int) -> set[tuple[int, int]]:
    target_lens_id = str(lut.get("target_lens_id") or "")
    for lens in lut.get("ocl_layout", {}).get("lenses", []):
        if str(lens.get("lens_id")) != target_lens_id:
            continue
        ix0 = parse_int(lens.get("ix"))
        iz0 = parse_int(lens.get("iz"))
        sx = max(1, parse_int(lens.get("w"), 1))
        sz = max(1, parse_int(lens.get("h"), 1))
        return {(ix, iz) for iz in range(iz0, iz0 + sz) for ix in range(ix0, ix0 + sx)}
    return {(nx // 2, nz // 2)}


def group_rows_by_case(rows: list[dict[str, str]]) -> dict[tuple[str, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("case", ""), parse_float(row.get("wavelength_nm"), 0.0) or 0.0)].append(row)
    return grouped


def response_matrix(rows: list[dict[str, str]]) -> tuple[list[list[float]], dict[tuple[int, int], dict[str, Any]]]:
    pixels: dict[tuple[int, int], dict[str, Any]] = {}
    max_ix = 0
    max_iz = 0
    for row in rows:
        if row.get("region_kind") != "pixel":
            continue
        ix = parse_int(row.get("region_ix"))
        iz = parse_int(row.get("region_iz"))
        response = parse_float(row.get("response"), 0.0) or 0.0
        pixels[(ix, iz)] = {
            "response": response,
            "x_um": parse_float(row.get("region_x_um"), 0.0) or 0.0,
            "z_um": parse_float(row.get("region_z_um"), 0.0) or 0.0,
        }
        max_ix = max(max_ix, ix)
        max_iz = max(max_iz, iz)
    matrix = [[0.0 for _ in range(max_ix + 1)] for _ in range(max_iz + 1)]
    for (ix, iz), item in pixels.items():
        matrix[iz][ix] = item["response"]
    return matrix, pixels


def split_pd_metrics(rows: list[dict[str, str]], summary: dict[str, str]) -> dict[str, Any]:
    values = {row.get("region_id", ""): parse_float(row.get("response"), 0.0) or 0.0 for row in rows}
    total = sum(values.values())
    nonzero = [value for value in values.values() if value >= 0]
    result = {
        "split_region_count": len(values),
        "split_total_response": total,
        "split_balance_error": ((max(nonzero) - min(nonzero)) / (sum(nonzero) / len(nonzero))) if nonzero and sum(nonzero) else None,
        "split_phase_x_proxy": parse_float(summary.get("split_phase_x_proxy")),
        "split_phase_z_proxy": parse_float(summary.get("split_phase_z_proxy")),
        "split_responses": values,
    }
    return result


def pixel_kpis(
    rows: list[dict[str, str]],
    summary: dict[str, str],
    lut: dict[str, Any],
    channel: str,
    case: CraFieldCase,
) -> dict[str, Any]:
    matrix, pixels = response_matrix(rows)
    if not pixels:
        return split_pd_metrics(rows, summary)
    nx = parse_int(lut.get("cell_pixels", {}).get("x"), max(ix for ix, _ in pixels) + 1)
    nz = parse_int(lut.get("cell_pixels", {}).get("z"), max(iz for _, iz in pixels) + 1)
    cfa_pattern = str(summary.get("cfa_pattern") or lut.get("cfa", {}).get("pattern") or "uniform")
    targets = target_pixels(lut, nx, nz)
    total = sum(item["response"] for item in pixels.values())
    target_response = sum(pixels.get(pixel, {}).get("response", 0.0) for pixel in targets)
    outside_response = max(total - target_response, 0.0)
    same_color_outside = 0.0
    diff_color_outside = 0.0
    for (ix, iz), item in pixels.items():
        if (ix, iz) in targets:
            continue
        if cfa_color(ix, iz, cfa_pattern, channel) == channel:
            same_color_outside += item["response"]
        else:
            diff_color_outside += item["response"]
    target_items = [pixels[pixel]["response"] for pixel in targets if pixel in pixels]
    target_mean = sum(target_items) / len(target_items) if target_items else 0.0
    target_std = (
        math.sqrt(sum((value - target_mean) ** 2 for value in target_items) / len(target_items))
        if target_items
        else 0.0
    )
    if total:
        centroid_x = sum(item["response"] * item["x_um"] for item in pixels.values()) / total
        centroid_z = sum(item["response"] * item["z_um"] for item in pixels.values()) / total
    else:
        centroid_x = 0.0
        centroid_z = 0.0
    target_centers = [pixels[pixel] for pixel in targets if pixel in pixels]
    if target_centers:
        target_center_x = sum(item["x_um"] for item in target_centers) / len(target_centers)
        target_center_z = sum(item["z_um"] for item in target_centers) / len(target_centers)
    else:
        target_center_x = 0.0
        target_center_z = 0.0
    normalized_kernel = [[value / total if total else 0.0 for value in row] for row in matrix]
    return {
        "pixel_region_count": len(pixels),
        "target_pixel_count": len(targets),
        "target_pixels": sorted([list(pixel) for pixel in targets]),
        "target_response": target_response,
        "target_fraction": target_response / total if total else None,
        "outside_fraction": outside_response / total if total else None,
        "neighbor_leakage_fraction": outside_response / total if total else None,
        "same_color_crosstalk_fraction": same_color_outside / total if total else None,
        "different_color_crosstalk_fraction": diff_color_outside / total if total else None,
        "target_uniformity_cv": target_std / target_mean if target_mean else None,
        "response_centroid_x_um": centroid_x,
        "response_centroid_z_um": centroid_z,
        "response_centroid_shift_x_um": centroid_x - target_center_x,
        "response_centroid_shift_z_um": centroid_z - target_center_z,
        "dti_hit_risk_proxy": outside_response / total if total else None,
        "response_matrix": matrix,
        "crosstalk_kernel": normalized_kernel,
    }


def process_channel_output(sensor: dict[str, Any], channel: str, output_dir: Path, cases: list[CraFieldCase]) -> dict[str, Any]:
    lut_path = output_dir / "camera_lut.json"
    summary_path = output_dir / "camera_lut_summary.csv"
    long_path = output_dir / "camera_lut_long.csv"
    if not lut_path.exists():
        return {"status": "MISSING", "kpis": [], "artifacts": {}}
    lut = read_json(lut_path)
    summaries = csv_rows(summary_path)
    long_rows = csv_rows(long_path)
    grouped_long = group_rows_by_case(long_rows)
    case_map = field_case_lookup(cases)
    center_by_channel = {}
    for summary in summaries:
        if summary.get("case") == "center_0":
            center_by_channel[channel] = parse_float(summary.get("total_response"), 0.0) or 0.0
    center_response = center_by_channel.get(channel, 0.0)
    kpis = []
    for summary in summaries:
        case_name = summary.get("case", "")
        wavelength_nm = parse_float(summary.get("wavelength_nm"), CHANNELS[channel]) or CHANNELS[channel]
        case = case_map.get(case_name)
        if case is None:
            case = CraFieldCase(
                case_name,
                "unknown",
                parse_float(summary.get("cra_x_deg"), 0.0) or 0.0,
                parse_float(summary.get("cra_x_deg"), 0.0) or 0.0,
                parse_float(summary.get("cra_z_deg"), 0.0) or 0.0,
                parse_float(summary.get("field_x_norm"), 0.0) or 0.0,
                parse_float(summary.get("field_z_norm"), 0.0) or 0.0,
            )
        rows = grouped_long.get((case_name, wavelength_nm), [])
        total_response = parse_float(summary.get("total_response"), 0.0) or 0.0
        kpi = {
            "sensor_id": sensor["sensor_id"],
            "code": sensor.get("code"),
            "manufacturer": sensor.get("manufacturer"),
            "device_name": sensor.get("device_name"),
            "source_template_id": sensor.get("source_template_id"),
            "channel": channel,
            "wavelength_nm": wavelength_nm,
            "case": case_name,
            "field": case.field,
            "nominal_cra_deg": case.nominal_cra_deg,
            "cra_x_deg": parse_float(summary.get("cra_x_deg"), case.cra_x_deg),
            "cra_z_deg": parse_float(summary.get("cra_z_deg"), case.cra_z_deg),
            "field_x_norm": parse_float(summary.get("field_x_norm"), case.field_x_norm),
            "field_z_norm": parse_float(summary.get("field_z_norm"), case.field_z_norm),
            "total_response": total_response,
            "relative_qe_to_center": total_response / center_response if center_response else None,
            "edge_to_center_response_ratio": total_response / center_response if center_response and case.field != "center" else 1.0,
            "grid_resolution_gate_pass": summary.get("grid_resolution_gate_pass") == "True",
            "si_wavelength_gate_pass": summary.get("si_wavelength_gate_pass") == "True",
            "critical_feature_gate_pass": summary.get("critical_feature_gate_pass") == "True",
            "recommended_min_resolution_px_per_um": parse_float(summary.get("recommended_min_resolution_px_per_um")),
            "grid_resolution_notes": summary.get("grid_resolution_notes", ""),
            "source_aperture_lens_id": summary.get("source_aperture_lens_id", ""),
            "source_aperture_enabled": summary.get("source_aperture_enabled") == "True",
            "focal_centroid_x_um": parse_float(summary.get("focal_centroid_x_um")),
            "focal_centroid_z_um": parse_float(summary.get("focal_centroid_z_um")),
            "focal_centroid_shift_x_um": parse_float(summary.get("focal_centroid_shift_x_um")),
            "focal_centroid_shift_z_um": parse_float(summary.get("focal_centroid_shift_z_um")),
            "focal_rms_radius_um": parse_float(summary.get("focal_rms_radius_um")),
            "focal_target_fraction": parse_float(summary.get("focal_target_fraction")),
        }
        kpi.update(pixel_kpis(rows, summary, lut, channel, case))
        if kpi.get("focal_target_fraction") is not None:
            kpi["focal_dti_risk_proxy"] = 1.0 - kpi["focal_target_fraction"]
        kpis.append(kpi)
    return {
        "status": "PASS" if kpis else "EMPTY",
        "kpis": kpis,
        "artifacts": {
            "camera_lut_json": repo_rel(lut_path),
            "summary_csv": repo_rel(summary_path),
            "long_csv": repo_rel(long_path),
            "response_maps": repo_rel(output_dir / "response_maps.png") if (output_dir / "response_maps.png").exists() else "",
            "focal_maps": repo_rel(output_dir / "focal_maps.png") if (output_dir / "focal_maps.png").exists() else "",
        },
    }


def color_balance_rows(kpis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in kpis:
        grouped[(row["sensor_id"], row["case"])][row["channel"]] = row
    output = []
    for (sensor_id, case), by_channel in grouped.items():
        green = by_channel.get("green", {}).get("total_response")
        red = by_channel.get("red", {}).get("total_response")
        blue = by_channel.get("blue", {}).get("total_response")
        example = next(iter(by_channel.values()))
        output.append(
            {
                "sensor_id": sensor_id,
                "case": case,
                "field": example.get("field"),
                "nominal_cra_deg": example.get("nominal_cra_deg"),
                "red_to_green": red / green if red is not None and green else None,
                "blue_to_green": blue / green if blue is not None and green else None,
                "color_shading_index": (max(red or 0, green or 0, blue or 0) - min(red or 0, green or 0, blue or 0)) / green if green else None,
            }
        )
    return output


def response_curve_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sensor_id": row.get("sensor_id"),
            "source_template_id": row.get("source_template_id"),
            "channel": row.get("channel"),
            "field": row.get("field"),
            "field_display": field_display(row.get("field")),
            "nominal_cra_deg": row.get("nominal_cra_deg"),
            "case": row.get("case"),
            "total_response": row.get("total_response"),
            "relative_qe_to_center": row.get("relative_qe_to_center"),
            "edge_to_center_response_ratio": row.get("edge_to_center_response_ratio"),
            "grid_resolution_gate_pass": row.get("grid_resolution_gate_pass"),
        }
        for row in flat_rows
    ]


def field_map_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sensor_id": row.get("sensor_id"),
            "source_template_id": row.get("source_template_id"),
            "channel": row.get("channel"),
            "field": row.get("field"),
            "field_display": field_display(row.get("field")),
            "field_x_norm": row.get("field_x_norm"),
            "field_z_norm": row.get("field_z_norm"),
            "nominal_cra_deg": row.get("nominal_cra_deg"),
            "relative_qe_to_center": row.get("relative_qe_to_center"),
            "neighbor_leakage_fraction": row.get("neighbor_leakage_fraction"),
        }
        for row in flat_rows
    ]


def focus_shift_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sensor_id": row.get("sensor_id"),
            "source_template_id": row.get("source_template_id"),
            "channel": row.get("channel"),
            "case": row.get("case"),
            "field": row.get("field"),
            "field_display": field_display(row.get("field")),
            "nominal_cra_deg": row.get("nominal_cra_deg"),
            "focal_centroid_shift_x_um": row.get("focal_centroid_shift_x_um"),
            "focal_centroid_shift_z_um": row.get("focal_centroid_shift_z_um"),
            "focal_rms_radius_um": row.get("focal_rms_radius_um"),
            "focal_target_fraction": row.get("focal_target_fraction"),
            "focal_dti_risk_proxy": row.get("focal_dti_risk_proxy"),
            "response_centroid_shift_x_um": row.get("response_centroid_shift_x_um"),
            "response_centroid_shift_z_um": row.get("response_centroid_shift_z_um"),
            "dti_hit_risk_proxy": row.get("dti_hit_risk_proxy"),
        }
        for row in flat_rows
    ]


def pdaf_split_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sensor_id": row.get("sensor_id"),
            "source_template_id": row.get("source_template_id"),
            "channel": row.get("channel"),
            "case": row.get("case"),
            "field": row.get("field"),
            "field_display": field_display(row.get("field")),
            "nominal_cra_deg": row.get("nominal_cra_deg"),
            "split_phase_x_proxy": row.get("split_phase_x_proxy"),
            "split_phase_z_proxy": row.get("split_phase_z_proxy"),
            "split_balance_error": row.get("split_balance_error"),
        }
        for row in flat_rows
        if row.get("split_phase_x_proxy") is not None
        or row.get("split_phase_z_proxy") is not None
        or row.get("split_balance_error") is not None
    ]


def binning_rows(flat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sensor_id": row.get("sensor_id"),
            "source_template_id": row.get("source_template_id"),
            "channel": row.get("channel"),
            "case": row.get("case"),
            "field": row.get("field"),
            "field_display": field_display(row.get("field")),
            "nominal_cra_deg": row.get("nominal_cra_deg"),
            "target_fraction": row.get("target_fraction"),
            "target_uniformity_cv": row.get("target_uniformity_cv"),
            "neighbor_leakage_fraction": row.get("neighbor_leakage_fraction"),
            "same_color_crosstalk_fraction": row.get("same_color_crosstalk_fraction"),
            "different_color_crosstalk_fraction": row.get("different_color_crosstalk_fraction"),
        }
        for row in flat_rows
    ]


def center_crop_kernel(kernel: list[list[float]], size: int) -> list[list[float]] | None:
    if not kernel or not kernel[0]:
        return None
    rows = len(kernel)
    cols = len(kernel[0])
    if rows < size or cols < size:
        return None
    row0 = max(0, (rows - size) // 2)
    col0 = max(0, (cols - size) // 2)
    return [list(row[col0 : col0 + size]) for row in kernel[row0 : row0 + size]]


def kernel_shape(kernel: list[list[float]] | None) -> str:
    if not kernel:
        return "0x0"
    return f"{len(kernel)}x{len(kernel[0]) if kernel[0] else 0}"


def crosstalk_kernel_records(kpis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in kpis:
        kernel = row.get("crosstalk_kernel")
        if not kernel:
            continue
        crop_3x3 = center_crop_kernel(kernel, 3)
        crop_5x5 = center_crop_kernel(kernel, 5)
        crop_7x7 = center_crop_kernel(kernel, 7)
        records.append(
            {
                "sensor_id": row.get("sensor_id"),
                "source_template_id": row.get("source_template_id"),
                "channel": row.get("channel"),
                "case": row.get("case"),
                "field": row.get("field"),
                "field_display": field_display(row.get("field")),
                "nominal_cra_deg": row.get("nominal_cra_deg"),
                "target_pixels": row.get("target_pixels"),
                "target_fraction": row.get("target_fraction"),
                "neighbor_leakage_fraction": row.get("neighbor_leakage_fraction"),
                "same_color_crosstalk_fraction": row.get("same_color_crosstalk_fraction"),
                "different_color_crosstalk_fraction": row.get("different_color_crosstalk_fraction"),
                "kernel_shape": kernel_shape(kernel),
                "kernel": kernel,
                "kernel_crop_3x3": crop_3x3,
                "kernel_crop_5x5": crop_5x5,
                "kernel_crop_7x7": crop_7x7,
                "available_kernel_crops": [
                    shape
                    for shape, crop in (("3x3", crop_3x3), ("5x5", crop_5x5), ("7x7", crop_7x7))
                    if crop is not None
                ],
            }
        )
    return records


def topology_coverage_report(flat_rows: list[dict[str, Any]], kernel_records: list[dict[str, Any]]) -> dict[str, Any]:
    templates = sorted({str(row.get("source_template_id")) for row in flat_rows if row.get("source_template_id")})
    fields = sorted({str(row.get("field")) for row in flat_rows if row.get("field")})
    channels = sorted({str(row.get("channel")) for row in flat_rows if row.get("channel")})
    angles = sorted({numeric(row.get("nominal_cra_deg")) for row in flat_rows if numeric(row.get("nominal_cra_deg")) is not None})
    kernel_shapes = sorted(
        {
            str(record.get("kernel_shape") or kernel_shape(record.get("kernel")))
            for record in kernel_records
        }
    )
    kernel_crops = sorted(
        {
            str(crop)
            for record in kernel_records
            for crop in record.get("available_kernel_crops", [])
        }
    )
    template_counts = defaultdict(int)
    for row in flat_rows:
        template_counts[str(row.get("source_template_id"))] += 1
    required = [
        {
            "id": "bayer_1x1",
            "required_for": "1x1 Bayer baseline and 3x3 nearest-neighbor crosstalk",
            "status": "PASS" if any("bayer" in item for item in templates) else "MISSING",
        },
        {
            "id": "quad_2x2_ocl",
            "required_for": "2x2 OCL binning response and grouped-pixel crosstalk",
            "status": "PASS" if any("quad_2x2" in item for item in templates) else "MISSING",
        },
        {
            "id": "qpd_split",
            "required_for": "PDAF / split-pixel CRA response",
            "status": "PASS" if any("qpd" in item or "split" in item for item in templates) else "MISSING",
        },
        {
            "id": "nona_3x3_ocl",
            "required_for": "3x3 OCL / Nona binning response",
            "status": "PASS" if any("nona" in item or "3x3_ocl" in item for item in templates) else "MISSING",
        },
        {
            "id": "mixed_ocl_boundary",
            "required_for": "1x1/2x2/3x3 OCL transition leakage",
            "status": "PASS" if any("mixed" in item for item in templates) else "MISSING",
        },
        {
            "id": "5x5_or_7x7_kernel",
            "required_for": "long-range high-CRA crosstalk truncation checks",
            "status": (
                "PASS"
                if any(crop in {"5x5", "7x7"} for crop in kernel_crops)
                or any("5x5" in item or "7x7" in item for item in templates)
                else "MISSING"
            ),
        },
    ]
    return {
        "schema": "reference_sensor_cra_coverage_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fields": [{"id": field, "display": field_display(field)} for field in fields],
        "channels": channels,
        "cra_angles_deg": angles,
        "source_template_ids": templates,
        "source_template_row_counts": dict(sorted(template_counts.items())),
        "crosstalk_kernel_shapes": kernel_shapes,
        "available_kernel_crops": kernel_crops,
        "required_topology_status": required,
        "notes": [
            "The optical propagation axis is Meep y; sensor lateral field y is represented as sim z-edge in these outputs.",
            "A missing required topology means the code may support the primitive, but this reference run did not include that simulation coverage.",
        ],
    }


def write_derived_outputs(output_dir: Path, flat_rows: list[dict[str, Any]], kpis: list[dict[str, Any]]) -> dict[str, str]:
    response_curve_path = output_dir / "cra_response_curve.csv"
    field_map_path = output_dir / "cra_field_map.csv"
    focus_path = output_dir / "cra_focus_shift.csv"
    pdaf_path = output_dir / "cra_pdaf_split.csv"
    binning_path = output_dir / "cra_binning_response.csv"
    kernels_path = output_dir / "cra_crosstalk_kernels.json"
    coverage_path = output_dir / "cra_coverage_report.json"
    kernels = crosstalk_kernel_records(kpis)
    write_csv(response_curve_path, response_curve_rows(flat_rows))
    write_csv(field_map_path, field_map_rows(flat_rows))
    write_csv(focus_path, focus_shift_rows(flat_rows))
    write_csv(pdaf_path, pdaf_split_rows(flat_rows))
    write_csv(binning_path, binning_rows(flat_rows))
    write_json(
        kernels_path,
        {
            "schema": "reference_sensor_cra_crosstalk_kernels_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "records": kernels,
        },
    )
    write_json(coverage_path, topology_coverage_report(flat_rows, kernels))
    return {
        "response_curve_csv": repo_rel(response_curve_path),
        "field_map_csv": repo_rel(field_map_path),
        "focus_shift_csv": repo_rel(focus_path),
        "pdaf_split_csv": repo_rel(pdaf_path),
        "binning_response_csv": repo_rel(binning_path),
        "crosstalk_kernels_json": repo_rel(kernels_path),
        "coverage_report_json": repo_rel(coverage_path),
    }


def compact_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if value != 0 and abs(value) < 0.001:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}g}"
    return str(value)


def flatten_kpi_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "sensor_id",
        "code",
        "manufacturer",
        "device_name",
        "source_template_id",
        "channel",
        "wavelength_nm",
        "case",
        "field",
        "nominal_cra_deg",
        "cra_x_deg",
        "cra_z_deg",
        "field_x_norm",
        "field_z_norm",
        "total_response",
        "relative_qe_to_center",
        "edge_to_center_response_ratio",
        "target_fraction",
        "neighbor_leakage_fraction",
        "same_color_crosstalk_fraction",
        "different_color_crosstalk_fraction",
        "target_uniformity_cv",
        "response_centroid_shift_x_um",
        "response_centroid_shift_z_um",
        "dti_hit_risk_proxy",
        "focal_centroid_x_um",
        "focal_centroid_z_um",
        "focal_centroid_shift_x_um",
        "focal_centroid_shift_z_um",
        "focal_rms_radius_um",
        "focal_target_fraction",
        "focal_dti_risk_proxy",
        "split_phase_x_proxy",
        "split_phase_z_proxy",
        "split_balance_error",
        "grid_resolution_gate_pass",
        "recommended_min_resolution_px_per_um",
        "source_aperture_lens_id",
        "source_aperture_enabled",
    ]
    return {field: row.get(field) for field in fields}


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


def sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def sqlite_column_type(rows: list[dict[str, Any]], column: str) -> str:
    values = [sqlite_value(row.get(column)) for row in rows if row.get(column) is not None]
    if not values:
        return "TEXT"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "REAL"
    return "TEXT"


def sqlite_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def write_sqlite_rows(conn: sqlite3.Connection, table_name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    column_defs = ", ".join(f"{sqlite_ident(column)} {sqlite_column_type(rows, column)}" for column in columns)
    conn.execute(f"CREATE TABLE {sqlite_ident(table_name)} ({column_defs})")
    placeholders = ", ".join("?" for _ in columns)
    for row in rows:
        conn.execute(
            f"INSERT INTO {sqlite_ident(table_name)} VALUES ({placeholders})",
            tuple(sqlite_value(row.get(column)) for column in columns),
        )


def write_sqlite(
    path: Path,
    catalog: dict[str, Any],
    flat_rows: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    kpis: list[dict[str, Any]],
) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE cra_kpi (
                sensor_id TEXT,
                code TEXT,
                manufacturer TEXT,
                device_name TEXT,
                source_template_id TEXT,
                channel TEXT,
                wavelength_nm REAL,
                case_name TEXT,
                field TEXT,
                field_x_norm REAL,
                field_z_norm REAL,
                nominal_cra_deg REAL,
                cra_x_deg REAL,
                cra_z_deg REAL,
                total_response REAL,
                relative_qe_to_center REAL,
                edge_to_center_response_ratio REAL,
                target_fraction REAL,
                neighbor_leakage_fraction REAL,
                same_color_crosstalk_fraction REAL,
                different_color_crosstalk_fraction REAL,
                target_uniformity_cv REAL,
                response_centroid_shift_x_um REAL,
                response_centroid_shift_z_um REAL,
                dti_hit_risk_proxy REAL,
                focal_centroid_x_um REAL,
                focal_centroid_z_um REAL,
                focal_centroid_shift_x_um REAL,
                focal_centroid_shift_z_um REAL,
                focal_rms_radius_um REAL,
                focal_target_fraction REAL,
                focal_dti_risk_proxy REAL,
                split_phase_x_proxy REAL,
                split_phase_z_proxy REAL,
                split_balance_error REAL,
                grid_resolution_gate_pass INTEGER,
                recommended_min_resolution_px_per_um REAL,
                source_aperture_lens_id TEXT,
                source_aperture_enabled INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE color_balance (
                sensor_id TEXT,
                case_name TEXT,
                field TEXT,
                nominal_cra_deg REAL,
                red_to_green REAL,
                blue_to_green REAL,
                color_shading_index REAL
            )
            """
        )
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema", catalog["schema"]))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("generated_at", catalog["generated_at"]))
        for row in flat_rows:
            conn.execute(
                """
                INSERT INTO cra_kpi VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    row.get("sensor_id"),
                    row.get("code"),
                    row.get("manufacturer"),
                    row.get("device_name"),
                    row.get("source_template_id"),
                    row.get("channel"),
                    row.get("wavelength_nm"),
                    row.get("case"),
                    row.get("field"),
                    row.get("field_x_norm"),
                    row.get("field_z_norm"),
                    row.get("nominal_cra_deg"),
                    row.get("cra_x_deg"),
                    row.get("cra_z_deg"),
                    row.get("total_response"),
                    row.get("relative_qe_to_center"),
                    row.get("edge_to_center_response_ratio"),
                    row.get("target_fraction"),
                    row.get("neighbor_leakage_fraction"),
                    row.get("same_color_crosstalk_fraction"),
                    row.get("different_color_crosstalk_fraction"),
                    row.get("target_uniformity_cv"),
                    row.get("response_centroid_shift_x_um"),
                    row.get("response_centroid_shift_z_um"),
                    row.get("dti_hit_risk_proxy"),
                    row.get("focal_centroid_x_um"),
                    row.get("focal_centroid_z_um"),
                    row.get("focal_centroid_shift_x_um"),
                    row.get("focal_centroid_shift_z_um"),
                    row.get("focal_rms_radius_um"),
                    row.get("focal_target_fraction"),
                    row.get("focal_dti_risk_proxy"),
                    row.get("split_phase_x_proxy"),
                    row.get("split_phase_z_proxy"),
                    row.get("split_balance_error"),
                    1 if row.get("grid_resolution_gate_pass") else 0,
                    row.get("recommended_min_resolution_px_per_um"),
                    row.get("source_aperture_lens_id"),
                    1 if row.get("source_aperture_enabled") else 0,
                ),
            )
        for row in balance_rows:
            conn.execute(
                "INSERT INTO color_balance VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("sensor_id"),
                    row.get("case"),
                    row.get("field"),
                    row.get("nominal_cra_deg"),
                    row.get("red_to_green"),
                    row.get("blue_to_green"),
                    row.get("color_shading_index"),
                ),
            )
        conn.execute("CREATE INDEX idx_cra_kpi_sensor ON cra_kpi(sensor_id)")
        conn.execute("CREATE INDEX idx_cra_kpi_case ON cra_kpi(case_name)")
        conn.execute("CREATE INDEX idx_cra_kpi_channel ON cra_kpi(channel)")
        kernel_rows = [
            {
                **{key: value for key, value in record.items() if key != "kernel"},
                "kernel_rows": len(record.get("kernel") or []),
                "kernel_cols": len((record.get("kernel") or [[]])[0]) if record.get("kernel") else 0,
                "kernel_json": record.get("kernel"),
            }
            for record in crosstalk_kernel_records(kpis)
        ]
        coverage = topology_coverage_report(flat_rows, crosstalk_kernel_records(kpis))
        coverage_rows = [
            {
                "coverage_id": item["id"],
                "required_for": item["required_for"],
                "status": item["status"],
            }
            for item in coverage["required_topology_status"]
        ]
        write_sqlite_rows(conn, "cra_response_curve", response_curve_rows(flat_rows))
        write_sqlite_rows(conn, "cra_field_map", field_map_rows(flat_rows))
        write_sqlite_rows(conn, "cra_focus_shift", focus_shift_rows(flat_rows))
        write_sqlite_rows(conn, "cra_pdaf_split", pdaf_split_rows(flat_rows))
        write_sqlite_rows(conn, "cra_binning_response", binning_rows(flat_rows))
        write_sqlite_rows(conn, "cra_crosstalk_kernel", kernel_rows)
        write_sqlite_rows(conn, "cra_topology_coverage", coverage_rows)


def mini_heatmap(kernel: list[list[float]] | None) -> str:
    if not kernel:
        return ""
    max_value = max((value for row in kernel for value in row), default=0.0)
    cells = []
    for row in kernel:
        for value in row:
            alpha = 0.08 + 0.85 * (value / max_value if max_value else 0.0)
            cells.append(
                f'<span style="background:rgba(84,215,238,{alpha:.3f})" title="{value:.3e}">{compact_float(value, 2)}</span>'
            )
    columns = len(kernel[0]) if kernel else 1
    return f'<div class="kernel" style="grid-template-columns:repeat({columns}, minmax(34px,1fr))">{"".join(cells)}</div>'


def svg_line_chart(
    title: str,
    series: list[dict[str, Any]],
    *,
    y_label: str,
    width: int = 860,
    height: int = 310,
) -> str:
    cleaned = []
    for item in series:
        points = [(float(x), float(y)) for x, y in item.get("points", []) if math.isfinite(float(x)) and math.isfinite(float(y))]
        if points:
            cleaned.append({**item, "points": sorted(points)})
    if not cleaned:
        return f'<div class="empty-chart">{escape(title)}: no data</div>'
    all_x = [x for item in cleaned for x, _ in item["points"]]
    all_y = [y for item in cleaned for _, y in item["points"]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= abs(y_min) * 0.05 + 0.05
        y_max += abs(y_max) * 0.05 + 0.05
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad
    left, right, top, bottom = 58, 18, 30, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    lines = [
        f'<svg class="svg-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text x="{left}" y="18" class="chart-title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4
        y_coord = sy(y_value)
        lines.append(f'<line x1="{left}" y1="{y_coord:.2f}" x2="{left + plot_w}" y2="{y_coord:.2f}" class="grid-line"/>')
        lines.append(f'<text x="{left - 8}" y="{y_coord + 4:.2f}" text-anchor="end" class="axis-label">{compact_float(y_value, 3)}</text>')
    for x_value in sorted(set(all_x)):
        x_coord = sx(x_value)
        lines.append(f'<text x="{x_coord:.2f}" y="{top + plot_h + 22}" text-anchor="middle" class="axis-label">{compact_float(x_value, 3)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis-label">CRA angle (deg)</text>')
    lines.append(f'<text x="14" y="{top + plot_h / 2:.2f}" transform="rotate(-90 14 {top + plot_h / 2:.2f})" text-anchor="middle" class="axis-label">{escape(y_label)}</text>')
    legend_x = left + 8
    legend_y = top + 18
    for index, item in enumerate(cleaned):
        color = item.get("color") or CHART_COLORS[index % len(CHART_COLORS)]
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in item["points"])
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
        for x_value, y_value in item["points"]:
            lines.append(f'<circle cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="3" fill="{color}"/>')
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y + index * 18}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 10}" y="{legend_y + 4 + index * 18}" class="legend-label">{escape(str(item.get("label", "")))}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def field_response_heatmap(flat_rows: list[dict[str, Any]], *, channel: str = "green") -> str:
    buckets: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in flat_rows:
        if row.get("channel") != channel:
            continue
        angle = numeric(row.get("nominal_cra_deg"))
        value = numeric(row.get("relative_qe_to_center"))
        field = str(row.get("field"))
        if angle is None or value is None:
            continue
        buckets[(angle, field)].append(value)
    averaged = {(angle, field): mean(values) for (angle, field), values in buckets.items()}
    values = [value for value in averaged.values() if value is not None]
    if not values:
        return '<div class="empty-chart">No field response map data</div>'
    v_min, v_max = min(values), max(values)
    if v_min == v_max:
        v_min -= 0.05
        v_max += 0.05
    cards = []
    for angle in sorted({angle for angle, _ in averaged}):
        cells = []
        for field in FIELD_ORDER:
            value = averaged.get((angle, field))
            if value is None:
                color = "rgba(84, 215, 238, 0.05)"
                label = ""
            else:
                normalized = (value - v_min) / (v_max - v_min)
                color = f"rgba(84, 215, 238, {0.18 + 0.72 * normalized:.3f})"
                label = compact_float(value, 3)
            cells.append(
                f'<div class="field-cell" style="background:{color}"><span>{escape(field_display(field))}</span><strong>{label}</strong></div>'
            )
        cards.append(
            f'<div class="field-card"><h3>{compact_float(angle, 3)} deg</h3><div class="field-grid">{"".join(cells)}</div></div>'
        )
    return f'<div class="field-heatmaps">{"".join(cards)}</div>'


def kernel_gallery(kpis: list[dict[str, Any]]) -> str:
    records = crosstalk_kernel_records(kpis)
    if not records:
        return '<div class="empty-chart">No crosstalk kernel data</div>'
    selected = []
    template_ids = sorted(
        {str(record.get("source_template_id")) for record in records},
        key=lambda template_id: (0 if "5x5" in template_id or "7x7" in template_id else 1, template_id),
    )
    for template_id in template_ids:
        template_records = [record for record in records if record.get("source_template_id") == template_id and record.get("channel") == "green"]
        for angle in (0.0, 10.0, 20.0, 30.0):
            preferred_field = "center" if angle == 0.0 else "x-edge"
            match = next(
                (
                    record
                    for record in template_records
                    if numeric(record.get("nominal_cra_deg")) == angle and record.get("field") == preferred_field
                ),
                None,
            )
            if match is None:
                match = next((record for record in template_records if numeric(record.get("nominal_cra_deg")) == angle), None)
            if match:
                selected.append(match)
        if len(selected) >= 12:
            break
    cards = []
    for record in selected[:12]:
        kernel = record.get("kernel_crop_7x7") or record.get("kernel_crop_5x5") or record.get("kernel_crop_3x3") or record.get("kernel")
        shape = kernel_shape(kernel)
        full_shape = record.get("kernel_shape") or kernel_shape(record.get("kernel"))
        cards.append(
            '<div class="kernel-card">'
            f'<h3>{escape(str(record.get("source_template_id")))} · {escape(str(record.get("case")))}</h3>'
            f'<p>{escape(str(record.get("channel")))} · {escape(field_display(record.get("field")))} · {compact_float(record.get("nominal_cra_deg"))} deg · displayed {shape} crop from {full_shape} raw</p>'
            f'{mini_heatmap(kernel)}'
            f'<p>leakage {compact_float(record.get("neighbor_leakage_fraction"))}, same-color {compact_float(record.get("same_color_crosstalk_fraction"))}, diff-color {compact_float(record.get("different_color_crosstalk_fraction"))}</p>'
            '</div>'
        )
    return f'<div class="kernel-gallery">{"".join(cards)}</div>'


def coverage_table_html(coverage: dict[str, Any]) -> str:
    rows = []
    for item in coverage.get("required_topology_status", []):
        status = str(item.get("status"))
        cls = "status-pass" if status == "PASS" else "status-check"
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('id')))}</td>"
            f"<td>{escape(str(item.get('required_for')))}</td>"
            f'<td><span class="{cls}">{escape(status)}</span></td>'
            "</tr>"
        )
    return "<table><thead><tr><th>Coverage item</th><th>Why it matters</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def write_html(path: Path, catalog: dict[str, Any], flat_rows: list[dict[str, Any]], balance_rows: list[dict[str, Any]]) -> None:
    sensors = catalog["records"]
    all_kpis = [row for record in sensors for row in record.get("kpis", [])]
    kernel_records = crosstalk_kernel_records(all_kpis)
    coverage = topology_coverage_report(flat_rows, kernel_records)
    green_field_chart = svg_line_chart(
        "Green relative QE by field",
        [
            {
                "label": field_display(field),
                "color": FIELD_COLORS.get(field),
                "points": grouped_mean(flat_rows, "relative_qe_to_center", filters={"channel": "green", "field": field}),
            }
            for field in FIELD_ORDER
        ],
        y_label="relative QE",
    )
    rgb_center_chart = svg_line_chart(
        "RGB center response vs CRA",
        [
            {
                "label": channel,
                "color": CHANNEL_COLORS[channel],
                "points": grouped_mean(flat_rows, "relative_qe_to_center", filters={"channel": channel, "field": "center"}),
            }
            for channel in ("red", "green", "blue")
        ],
        y_label="relative QE",
    )
    leakage_chart = svg_line_chart(
        "Green neighbor leakage by field",
        [
            {
                "label": field_display(field),
                "color": FIELD_COLORS.get(field),
                "points": grouped_mean(flat_rows, "neighbor_leakage_fraction", filters={"channel": "green", "field": field}),
            }
            for field in FIELD_ORDER
        ],
        y_label="leakage fraction",
    )
    color_chart = svg_line_chart(
        "Color shading balance",
        [
            {"label": "R/G", "color": CHANNEL_COLORS["red"], "points": grouped_mean(balance_rows, "red_to_green")},
            {"label": "B/G", "color": CHANNEL_COLORS["blue"], "points": grouped_mean(balance_rows, "blue_to_green")},
            {"label": "CSI", "color": "#facc15", "points": grouped_mean(balance_rows, "color_shading_index")},
        ],
        y_label="ratio / index",
    )
    focus_metric_rows = []
    for row in flat_rows:
        x_shift = numeric(row.get("focal_centroid_shift_x_um"))
        z_shift = numeric(row.get("focal_centroid_shift_z_um"))
        if x_shift is None or z_shift is None:
            continue
        focus_metric_rows.append({**row, "focal_shift_magnitude_um": math.sqrt(x_shift * x_shift + z_shift * z_shift)})
    focus_chart = svg_line_chart(
        "OCL focal shift magnitude and target loss",
        [
            {
                "label": "focus shift um",
                "color": "#54d7ee",
                "points": grouped_mean(focus_metric_rows, "focal_shift_magnitude_um", filters={"channel": "green"}),
            },
            {
                "label": "DTI risk proxy",
                "color": "#fb7185",
                "points": grouped_mean(flat_rows, "focal_dti_risk_proxy", filters={"channel": "green"}),
            },
        ],
        y_label="um / fraction",
    )
    pdaf_chart = svg_line_chart(
        "PDAF split response vs CRA",
        [
            {"label": "phase x", "color": "#54d7ee", "points": grouped_mean(flat_rows, "split_phase_x_proxy", filters={"channel": "green"})},
            {"label": "phase z", "color": "#a78bfa", "points": grouped_mean(flat_rows, "split_phase_z_proxy", filters={"channel": "green"})},
            {"label": "balance error", "color": "#facc15", "points": grouped_mean(flat_rows, "split_balance_error", filters={"channel": "green"})},
        ],
        y_label="phase proxy / balance",
    )
    binning_chart = svg_line_chart(
        "Binning uniformity CV by topology",
        [
            {
                "label": template_id,
                "color": CHART_COLORS[index % len(CHART_COLORS)],
                "points": grouped_mean(flat_rows, "target_uniformity_cv", filters={"source_template_id": template_id, "channel": "green"}),
            }
            for index, template_id in enumerate(sorted({str(row.get("source_template_id")) for row in flat_rows if row.get("source_template_id")}))
        ],
        y_label="uniformity CV",
    )
    overview_rows = []
    for record in sensors:
        kpis = record["kpis"]
        center_green = next((row for row in kpis if row["channel"] == "green" and row["case"] == "center_0"), {})
        worst_edge = min(
            (row for row in kpis if row.get("relative_qe_to_center") is not None),
            key=lambda row: row.get("relative_qe_to_center") or 999,
            default={},
        )
        max_leak = max(
            (row for row in kpis if row.get("neighbor_leakage_fraction") is not None),
            key=lambda row: row.get("neighbor_leakage_fraction") or -1,
            default={},
        )
        overview_rows.append(
            "<tr>"
            f"<td>{escape(record['manufacturer'])}<br><code>{escape(record['code'])}</code></td>"
            f"<td>{escape(record['device_name'])}</td>"
            f"<td>{escape(record['source_template_id'])}</td>"
            f"<td>{compact_float(center_green.get('total_response'))}</td>"
            f"<td>{escape(str(worst_edge.get('case', '')))}<br>{compact_float(worst_edge.get('relative_qe_to_center'))}</td>"
            f"<td>{escape(str(max_leak.get('case', '')))}<br>{compact_float(max_leak.get('neighbor_leakage_fraction'))}</td>"
            f"<td>{record['run_status']}</td>"
            "</tr>"
        )
    balance_table = []
    for row in balance_rows[:80]:
        balance_table.append(
            "<tr>"
            f"<td>{escape(str(row['sensor_id']))}</td><td>{escape(str(row['case']))}</td>"
            f"<td>{compact_float(row.get('red_to_green'))}</td><td>{compact_float(row.get('blue_to_green'))}</td>"
            f"<td>{compact_float(row.get('color_shading_index'))}</td>"
            "</tr>"
        )
    curve_rows = []
    for row in flat_rows[:160]:
        curve_rows.append(
            "<tr>"
            f"<td>{escape(str(row['sensor_id']))}</td><td>{escape(str(row['channel']))}</td>"
            f"<td>{escape(str(row['case']))}</td><td>{compact_float(row.get('relative_qe_to_center'))}</td>"
            f"<td>{compact_float(row.get('neighbor_leakage_fraction'))}</td>"
            f"<td>{compact_float(row.get('response_centroid_shift_x_um'))}, {compact_float(row.get('response_centroid_shift_z_um'))}</td>"
            f"<td>{'PASS' if row.get('grid_resolution_gate_pass') else 'CHECK'}</td>"
            "</tr>"
        )
    focus_rows = []
    for row in flat_rows[:160]:
        focus_rows.append(
            "<tr>"
            f"<td>{escape(str(row['sensor_id']))}</td><td>{escape(str(row['channel']))}</td><td>{escape(str(row['case']))}</td>"
            f"<td>{compact_float(row.get('focal_centroid_shift_x_um'))}, {compact_float(row.get('focal_centroid_shift_z_um'))}</td>"
            f"<td>{compact_float(row.get('focal_rms_radius_um'))}</td>"
            f"<td>{compact_float(row.get('focal_target_fraction'))}</td>"
            f"<td>{compact_float(row.get('focal_dti_risk_proxy'))}</td>"
            "</tr>"
        )
    pdaf_rows = []
    for row in [item for item in flat_rows if item.get("split_phase_x_proxy") is not None or item.get("split_phase_z_proxy") is not None][:160]:
        pdaf_rows.append(
            "<tr>"
            f"<td>{escape(str(row['sensor_id']))}</td><td>{escape(str(row['channel']))}</td><td>{escape(str(row['case']))}</td>"
            f"<td>{compact_float(row.get('split_phase_x_proxy'))}</td>"
            f"<td>{compact_float(row.get('split_phase_z_proxy'))}</td>"
            f"<td>{compact_float(row.get('split_balance_error'))}</td>"
            "</tr>"
        )
    artifact_rows = "".join(
        f"<li><code>{escape(str(path_value))}</code></li>"
        for path_value in catalog.get("artifacts", {}).values()
    )
    boundary_report = path.parent / "boundary_continuity" / "index.html"
    if boundary_report.exists():
        artifact_rows += '<li><code>boundary_continuity/index.html</code> - boundary continuity, mesh quality, and response decomposition</li>'
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reference Sensor CRA Analysis</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#071017; color:#e7f4f8; }}
    h1, h2 {{ margin: 0 0 10px; }}
    p {{ color:#9ab6c2; line-height:1.55; }}
    section {{ margin-top:18px; border:1px solid #24495a; border-radius:12px; padding:16px; background:#0d1b24; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid #24495a; padding:8px; text-align:left; vertical-align:top; }}
    th {{ color:#54d7ee; }}
    code {{ color:#dcefff; }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }}
    .card {{ border:1px solid #24495a; border-radius:10px; padding:12px; background:#081720; }}
    .card span {{ display:block; color:#9ab6c2; font-size:12px; }}
    .card strong {{ display:block; font-size:22px; margin-top:4px; }}
    .kernel {{ display:grid; gap:2px; max-width:320px; }}
    .kernel span {{ display:block; min-height:26px; padding:5px; color:#061018; font-size:11px; text-align:center; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:12px 0 16px; }}
    .chart-panel, .kernel-card, .field-card {{ border:1px solid #24495a; border-radius:10px; background:#081720; padding:12px; }}
    .chart-panel h3, .kernel-card h3, .field-card h3 {{ margin:0 0 8px; font-size:14px; color:#dcefff; }}
    .svg-chart {{ width:100%; height:auto; display:block; background:#071017; border-radius:8px; }}
    .chart-title {{ fill:#dcefff; font-size:15px; font-weight:700; }}
    .axis {{ stroke:#7392a1; stroke-width:1.2; }}
    .grid-line {{ stroke:#1d3643; stroke-width:1; }}
    .axis-label, .legend-label {{ fill:#9ab6c2; font-size:11px; }}
    .field-heatmaps {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .field-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; }}
    .field-cell {{ min-height:76px; border:1px solid #24495a; border-radius:8px; padding:8px; color:#061018; }}
    .field-cell span {{ display:block; font-size:11px; color:#dcefff; text-shadow:0 1px 1px #061018; }}
    .field-cell strong {{ display:block; margin-top:8px; font-size:20px; color:#fff; text-shadow:0 1px 1px #061018; }}
    .kernel-gallery {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .kernel-card p {{ margin:6px 0; font-size:12px; }}
    .status-pass, .status-check {{ display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:12px; }}
    .status-pass {{ color:#bbf7d0; background:#064e3b; border:1px solid #22c55e; }}
    .status-check {{ color:#fde68a; background:#3f2f05; border:1px solid #eab308; }}
    .empty-chart {{ border:1px solid #24495a; border-radius:10px; padding:16px; color:#9ab6c2; background:#081720; }}
    .tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }}
    .tabs button {{ border:1px solid #2f6174; border-radius:8px; background:#081720; color:#dcefff; padding:9px 12px; cursor:pointer; }}
    .tabs button.active {{ border-color:#54d7ee; background:#123140; color:#54d7ee; }}
    .tab-pane {{ display:none; }}
    .tab-pane.active {{ display:block; }}
    @media (max-width: 1100px) {{ .chart-grid, .kernel-gallery, .field-heatmaps {{ grid-template-columns:1fr; }} }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns:1fr 1fr; }} }}
  </style>
  <script>
    function showTab(id) {{
      document.querySelectorAll('.tab-pane').forEach(function(el) {{ el.classList.remove('active'); }});
      document.querySelectorAll('.tabs button').forEach(function(el) {{ el.classList.remove('active'); }});
      document.getElementById(id).classList.add('active');
      document.querySelector('[data-tab="' + id + '"]').classList.add('active');
    }}
  </script>
</head>
<body>
  <h1>Reference Sensor CRA Analysis</h1>
  <p>Field position × CRA × RGB wavelength/channel sweep for camera-system response LUTs. Results are low-resolution trend artifacts, not product-ready QE or crosstalk calibration.</p>
  <div class="cards">
    <div class="card"><span>Sensors</span><strong>{catalog['sensor_count']}</strong></div>
    <div class="card"><span>Channels</span><strong>{len(catalog['channels'])}</strong></div>
    <div class="card"><span>Cases / channel</span><strong>{len(catalog['cases'])}</strong></div>
    <div class="card"><span>KPI rows</span><strong>{len(flat_rows)}</strong></div>
    <div class="card"><span>Grid gates</span><strong>{escape(str(catalog['grid_gate_counts']))}</strong></div>
  </div>
  <div class="tabs">
    <button class="active" data-tab="overview" onclick="showTab('overview')">Overview</button>
    <button data-tab="cra-response" onclick="showTab('cra-response')">CRA Response</button>
    <button data-tab="field-map" onclick="showTab('field-map')">Field Map</button>
    <button data-tab="crosstalk" onclick="showTab('crosstalk')">Crosstalk</button>
    <button data-tab="color" onclick="showTab('color')">Color Shading</button>
    <button data-tab="focus" onclick="showTab('focus')">OCL Focus</button>
    <button data-tab="pdaf" onclick="showTab('pdaf')">PDAF</button>
    <button data-tab="export" onclick="showTab('export')">Camera LUT Export</button>
  </div>
  <section id="overview" class="tab-pane active">
    <h2>Overview</h2>
    <p>Coverage: fields {escape(', '.join(field_display(item) for item in sorted({row.get('field') for row in flat_rows})))}, CRA {escape(', '.join(compact_float(item) for item in sorted({row.get('nominal_cra_deg') for row in flat_rows})))} deg, channels {escape(', '.join(catalog['channels']))}. Grid gates remain trend-level unless they pass convergence.</p>
    <table><thead><tr><th>Sensor</th><th>Device</th><th>Template</th><th>Center G response</th><th>Worst relative QE</th><th>Max leakage</th><th>Status</th></tr></thead><tbody>{''.join(overview_rows)}</tbody></table>
    <h2>Topology Coverage</h2>
    {coverage_table_html(coverage)}
  </section>
  <section id="cra-response" class="tab-pane">
    <h2>CRA Response</h2>
    <p>Angle response is averaged across the reference sensor set. Use <code>cra_response_curve.csv</code> or SQLite table <code>cra_response_curve</code> for per-sensor rows.</p>
    <div class="chart-grid">
      <div class="chart-panel">{green_field_chart}</div>
      <div class="chart-panel">{rgb_center_chart}</div>
      <div class="chart-panel">{leakage_chart}</div>
      <div class="chart-panel">{binning_chart}</div>
    </div>
    <table><thead><tr><th>Sensor</th><th>Channel</th><th>Case</th><th>Relative QE</th><th>Leakage</th><th>Response centroid shift x,z um</th><th>Gate</th></tr></thead><tbody>{''.join(curve_rows)}</tbody></table>
  </section>
  <section id="field-map" class="tab-pane">
    <h2>Field Response Map</h2>
    <p>Green-channel field map uses center, x-edge, y-edge represented by sim z-edge, and diagonal. Use <code>cra_field_map.csv</code> or SQLite table <code>cra_field_map</code> for camera-system LUT export.</p>
    {field_response_heatmap(flat_rows, channel='green')}
  </section>
  <section id="crosstalk" class="tab-pane">
    <h2>Crosstalk Kernel vs CRA</h2>
    <p>Target OCL aperture illumination is enabled, so the kernel is a target-aperture optical crosstalk proxy rather than measured electrical crosstalk. Kernel size follows the simulated topology domain in this run.</p>
    {kernel_gallery(all_kpis)}
  </section>
  <section id="color" class="tab-pane">
    <h2>Color Shading / Channel Balance</h2>
    <div class="chart-grid"><div class="chart-panel">{color_chart}</div></div>
    <table><thead><tr><th>Sensor</th><th>Case</th><th>R/G</th><th>B/G</th><th>Color shading index</th></tr></thead><tbody>{''.join(balance_table)}</tbody></table>
  </section>
  <section id="focus" class="tab-pane">
    <h2>OCL Focus Shift / DTI Risk</h2>
    <div class="chart-grid"><div class="chart-panel">{focus_chart}</div></div>
    <table><thead><tr><th>Sensor</th><th>Channel</th><th>Case</th><th>Focal shift x,z um</th><th>Focal RMS um</th><th>Target fraction</th><th>DTI risk proxy</th></tr></thead><tbody>{''.join(focus_rows)}</tbody></table>
  </section>
  <section id="pdaf" class="tab-pane">
    <h2>PDAF / Split Pixel CRA Response</h2>
    <div class="chart-grid"><div class="chart-panel">{pdaf_chart}</div></div>
    <table><thead><tr><th>Sensor</th><th>Channel</th><th>Case</th><th>Phase X proxy</th><th>Phase Z proxy</th><th>Balance error</th></tr></thead><tbody>{''.join(pdaf_rows)}</tbody></table>
  </section>
  <section id="export" class="tab-pane">
    <h2>Camera LUT Export</h2>
    <p><code>cra_camera_lut_export.json</code>, <code>cra_analysis.sqlite</code>, and the derived CSV files below are generated for downstream camera-system simulation. SQLite includes analysis tables for response curve, field map, focus shift, PDAF, binning, crosstalk kernel, and topology coverage.</p>
    <ul>{artifact_rows}</ul>
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_channel_command(
    base_command: list[str],
    output_dir: Path,
    cases: list[CraFieldCase],
    channel: str,
    resolution: int,
    after_source_time: float,
    source_aperture_lens_id: str,
) -> list[str]:
    command = replace_arg(base_command, "--output-dir", str(output_dir))
    command = replace_arg(command, "--cases", cases_string(cases))
    command = replace_arg(command, "--wavelengths-nm", str(CHANNELS[channel]))
    command = replace_arg(command, "--color-channel", channel)
    command = replace_arg(command, "--resolution", str(resolution))
    command = replace_arg(command, "--after-source-time", str(after_source_time))
    if source_aperture_lens_id:
        command = replace_arg(command, "--source-aperture-lens-id", source_aperture_lens_id)
    return command


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    template_catalog = read_json(args.template_catalog)
    sensors = template_catalog["records"][: args.max_sensors]
    supplemental_records = supplemental_template_records(args.supplemental_templates, args.supplemental_template_root)
    sensors = [*sensors, *supplemental_records]
    cases = default_cases([float(value) for value in args.cra_angles.split(",") if value.strip()])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    all_kpis = []
    for sensor_index, sensor in enumerate(sensors, 1):
        sensor_id = sensor["sensor_id"]
        print(f"{sensor_index}/{len(sensors)} {sensor_id}")
        sensor_record = {
            "sensor_id": sensor_id,
            "code": sensor.get("code"),
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "source_template_id": sensor.get("source_template_id", ""),
            "channels": {},
            "kpis": [],
            "run_status": "PASS",
        }
        base_command = sensor.get("simulation", {}).get("command")
        if not isinstance(base_command, list):
            sensor_record["run_status"] = "NO_BASE_COMMAND"
            records.append(sensor_record)
            continue
        for channel in args.channels.split(","):
            channel = channel.strip()
            if channel not in CHANNELS:
                raise ValueError(f"Unsupported channel {channel!r}")
            channel_dir = output_dir / "simulations" / sensor_id / channel
            command = build_channel_command(
                base_command,
                channel_dir,
                cases,
                channel,
                args.resolution,
                args.after_source_time,
                args.source_aperture_lens_id,
            )
            if args.reuse_existing and (channel_dir / "camera_lut.json").exists():
                run_result = {"status": "REUSED", "command": command, "elapsed_s": 0.0, "exit_code": 0}
            else:
                run_result = run_solver_command(command, ROOT, channel_dir, args.timeout_s)
            processed = process_channel_output(sensor, channel, channel_dir, cases)
            channel_status = "PASS" if run_result["status"] in {"PASS", "REUSED"} and processed["status"] == "PASS" else run_result["status"]
            if channel_status not in {"PASS", "REUSED"}:
                sensor_record["run_status"] = "CHECK"
            sensor_record["channels"][channel] = {
                "status": channel_status,
                "run": run_result,
                "processed": {key: value for key, value in processed.items() if key != "kpis"},
            }
            sensor_record["kpis"].extend(processed["kpis"])
            all_kpis.extend(processed["kpis"])
        records.append(sensor_record)
    flat_rows = [flatten_kpi_row(row) for row in all_kpis]
    balance = color_balance_rows(all_kpis)
    derived_artifacts = write_derived_outputs(output_dir, flat_rows, all_kpis)
    grid_counts = defaultdict(int)
    for row in flat_rows:
        grid_counts["PASS" if row.get("grid_resolution_gate_pass") else "CHECK"] += 1
    catalog = {
        "schema": "reference_sensor_cra_analysis_db_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_template_catalog": repo_rel(args.template_catalog),
        "output_dir": repo_rel(output_dir),
        "sensor_count": len(records),
        "reference_sensor_count": min(args.max_sensors, len(template_catalog["records"])),
        "supplemental_template_count": len(supplemental_records),
        "channels": [channel.strip() for channel in args.channels.split(",") if channel.strip()],
        "wavelengths_nm": {channel: CHANNELS[channel] for channel in args.channels.split(",") if channel.strip()},
        "cases": [case.__dict__ for case in cases],
        "runtime": {
            "resolution": args.resolution,
            "after_source_time": args.after_source_time,
            "tier": args.tier,
            "source_aperture_lens_id": args.source_aperture_lens_id,
            "supplemental_templates": [record["source_template_id"] for record in supplemental_records],
        },
        "grid_gate_counts": dict(sorted(grid_counts.items())),
        "artifacts": derived_artifacts,
        "records": records,
        "notes": [
            "relative_qe_to_center is an optical response ratio to center_0 within the same sensor/channel.",
            "crosstalk_kernel is normalized pixel-region Si absorption response, not measured electrical crosstalk.",
            "response_centroid_shift is computed from pixel response distribution and is a focus/DTI-risk proxy.",
            "Current low-resolution trend tier is not product LUT ready unless grid gates pass with measured stack/material data.",
        ],
    }
    write_json(output_dir / "cra_analysis_catalog.json", catalog)
    write_csv(output_dir / "cra_kpi_summary.csv", flat_rows)
    write_csv(output_dir / "cra_color_balance.csv", balance)
    write_json(
        output_dir / "cra_camera_lut_export.json",
        {
            "schema": "camera_system_sensor_cra_lut_export_v1",
            "generated_at": catalog["generated_at"],
            "tier": args.tier,
            "kpi_rows": flat_rows,
            "color_balance": balance,
            "accuracy_status": "research_trend_only",
        },
    )
    write_sqlite(output_dir / "cra_analysis.sqlite", catalog, flat_rows, balance, all_kpis)
    write_html(output_dir / "index.html", catalog, flat_rows, balance)
    return catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-catalog", type=Path, default=DEFAULT_TEMPLATE_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sensors", type=int, default=8)
    parser.add_argument(
        "--supplemental-template-root",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_TEMPLATE_ROOT,
        help="CAD template library root used for topology coverage supplements.",
    )
    parser.add_argument(
        "--supplemental-templates",
        default=DEFAULT_SUPPLEMENTAL_TEMPLATES,
        help="Comma-separated CAD template ids appended after the reference sensor records. Use an empty string to disable.",
    )
    parser.add_argument("--channels", default="red,green,blue")
    parser.add_argument("--cra-angles", default="0,10,20,30")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--after-source-time", type=float, default=0.3)
    parser.add_argument(
        "--source-aperture-lens-id",
        default="target",
        help="Use 'target' to restrict illumination to --target-lens-id OCL aperture for crosstalk-kernel proxy.",
    )
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--tier", choices=("smoke", "trend", "quantitative"), default="trend")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    catalog = run_analysis(parse_args())
    print(
        json.dumps(
            {
                "schema": catalog["schema"],
                "sensor_count": catalog["sensor_count"],
                "grid_gate_counts": catalog["grid_gate_counts"],
                "output_dir": catalog["output_dir"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
