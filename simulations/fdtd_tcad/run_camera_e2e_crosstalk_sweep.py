#!/usr/bin/env python3
"""Run CameraE2E crosstalk FDTD jobs from the sensor LUT package."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_TCAD_PROFILE_DIR = ROOT / "image_sensor_db" / "generated_tcad_profiles"
DEFAULT_MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"

MANIFEST_COLUMNS = [
    "job_id",
    "tier",
    "slug",
    "code",
    "mode",
    "color_channel",
    "cfa_pattern",
    "wavelengths_nm",
    "field_cases",
    "sweep_point_count",
    "neighborhood",
    "required_neighborhood",
    "simulation_neighborhood",
    "required_simulation_neighborhood",
    "guard_cells",
    "required_guard_cells",
    "boundary_domain_gate",
    "boundary_domain_notes",
    "resolution_px_per_um",
    "estimated_voxels",
    "resource_gate",
    "resource_notes",
    "after_source_time",
    "stack_config",
    "tcad_profile",
    "output_dir",
    "status",
    "returncode",
    "duration_s",
    "stdout",
    "stderr",
    "command",
]

SUMMARY_COLUMNS = [
    "job_id",
    "tier",
    "slug",
    "code",
    "mode",
    "color_channel",
    "cfa_pattern",
    "wavelength_nm",
    "case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "neighborhood",
    "simulation_neighborhood",
    "required_neighborhood",
    "required_simulation_neighborhood",
    "guard_cells",
    "required_guard_cells",
    "boundary_domain_gate",
    "boundary_domain_notes",
    "raw_pd_kernel_shape",
    "center_response_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "grid_resolution_gate_pass",
    "estimated_voxels",
    "resource_gate",
    "resource_notes",
    "convergence_status",
    "solver_gate",
    "product_lut_ready",
    "source_summary_csv",
    "source_kernel_json",
]


TIER_CONFIG = {
    "smoke": {
        "resolution": 4,
        "after_source_time": 8.0,
        "wavelengths_nm": "550",
        "field_cases": ("center",),
        "colors": ("green",),
        "neighborhood_override": 3,
        "guard_override": 0,
        "quantitative": False,
    },
    "trend": {
        "resolution": 20,
        "after_source_time": 18.0,
        "wavelengths_nm": "550",
        "field_cases": ("center", "x_plus_edge", "z_plus_edge", "diag_plus_plus"),
        "colors": ("red", "green", "blue"),
        "neighborhood_override": 0,
        "guard_override": 1,
        "quantitative": False,
    },
    "quantitative": {
        "resolution": 80,
        "after_source_time": 60.0,
        "wavelengths_nm": "450,550,620",
        "field_cases": (
            "center",
            "x_minus_edge",
            "x_plus_edge",
            "z_minus_edge",
            "z_plus_edge",
            "diag_minus_minus",
            "diag_minus_plus",
            "diag_plus_minus",
            "diag_plus_plus",
        ),
        "colors": ("red", "green", "blue"),
        "neighborhood_override": 0,
        "guard_override": -1,
        "quantitative": True,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_resolution_sweep(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if any(value <= 0 for value in values):
        raise ValueError("--resolution-sweep values must be positive integers")
    return values


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if limit is not None and len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def mode_for_ocl(ocl_mode: str) -> str:
    if ocl_mode == "quad_2x2":
        return "ocl-2x2"
    if ocl_mode == "nona_3x3":
        return "ocl-3x3"
    return "split-pd-1x1"


def layout_size_for_mode(mode: str) -> int:
    if mode == "ocl-2x2":
        return 2
    if mode == "ocl-3x3":
        return 3
    return 1


def stack_geometry(path: Path) -> dict[str, float]:
    stack = read_json(path)
    geometry = stack.get("geometry_um", {})
    pitch = finite_float(geometry.get("pitch"), 1.4)
    pml = finite_float(geometry.get("pml"), 0.45)
    cell_y = (
        2.0 * pml
        + finite_float(geometry.get("air_top"), 0.55)
        + finite_float(geometry.get("lens_height"), 0.35)
        + finite_float(geometry.get("cfa_thickness"), 0.45)
        + finite_float(geometry.get("passivation_thickness"), 0.15)
        + finite_float(geometry.get("si_thickness"), 2.0)
        + finite_float(geometry.get("bottom_air"), 0.25)
    )
    return {"pitch": pitch, "pml": pml, "cell_y": cell_y}


def estimated_crosstalk_voxels(
    stack_config: Path,
    mode: str,
    neighborhood: int,
    guard_cells: int,
    resolution: int,
) -> int:
    geometry = stack_geometry(stack_config)
    simulation_neighborhood = neighborhood + 2 * guard_cells
    layout_size = layout_size_for_mode(mode)
    simulation_span = simulation_neighborhood * layout_size * geometry["pitch"]
    cell_x = simulation_span + 2.0 * geometry["pml"]
    cell_z = cell_x
    return int(math.ceil(cell_x * resolution) * math.ceil(geometry["cell_y"] * resolution) * math.ceil(cell_z * resolution))


def boundary_domain_status(
    *,
    neighborhood: int,
    guard_cells: int,
    required_neighborhood: int,
    required_simulation_neighborhood: int,
    required_guard_cells: int,
) -> tuple[str, str]:
    simulation_neighborhood = neighborhood + 2 * guard_cells
    blockers = []
    if required_neighborhood > 0 and neighborhood < required_neighborhood:
        blockers.append(f"neighborhood {neighborhood} < required {required_neighborhood}")
    if required_guard_cells > 0 and guard_cells < required_guard_cells:
        blockers.append(f"guard {guard_cells} < required {required_guard_cells}")
    if required_simulation_neighborhood > 0 and simulation_neighborhood < required_simulation_neighborhood:
        blockers.append(f"simulation_neighborhood {simulation_neighborhood} < required {required_simulation_neighborhood}")
    if blockers:
        return "FAIL", "; ".join(blockers)
    return "PASS", "actual finite-array domain satisfies the required crosstalk boundary domain"


def cfa_pattern_for(catalog_pattern: str, ocl_mode: str) -> str:
    text = str(catalog_pattern or "").lower()
    if "mono" in text:
        return "uniform"
    if "nona" in text or ocl_mode == "nona_3x3":
        return "nona"
    if "quad" in text or ocl_mode == "quad_2x2":
        return "quad"
    if "bayer" in text:
        return "bayer"
    return "uniform"


def case_string(rows: list[dict[str, str]], requested_cases: tuple[str, ...]) -> str:
    by_case = {row["field_case"]: row for row in rows}
    items = []
    for name in requested_cases:
        row = by_case.get(name)
        if row is None:
            continue
        items.append(
            ":".join(
                [
                    name,
                    str(finite_float(row.get("cra_x_deg"), 0.0)),
                    str(finite_float(row.get("cra_z_deg"), 0.0)),
                    str(finite_float(row.get("field_x_norm"), 0.0)),
                    str(finite_float(row.get("field_z_norm"), 0.0)),
                    str(finite_float(row.get("lens_shift_x_um"), 0.0)),
                    str(finite_float(row.get("lens_shift_z_um"), 0.0)),
                ]
            )
        )
    if not items:
        raise ValueError("no matching field cases found")
    return ",".join(items)


def case_params(cases: str) -> dict[str, dict[str, str]]:
    output = {}
    for item in [part for part in cases.split(",") if part.strip()]:
        parts = item.split(":")
        if len(parts) < 7:
            continue
        output[parts[0]] = {
            "cra_x_deg": parts[1],
            "cra_z_deg": parts[2],
            "field_x_norm": parts[3],
            "field_z_norm": parts[4],
            "lens_shift_x_um": parts[5],
            "lens_shift_z_um": parts[6],
        }
    return output


def csv_count(text: Any) -> int:
    return len([item for item in str(text or "").split(",") if item.strip()])


def selected_slugs(all_slugs: list[str], raw: str) -> list[str]:
    if not raw:
        return all_slugs
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    known = set(all_slugs)
    missing = [item for item in requested if item not in known]
    if missing:
        raise ValueError("unknown sensor slugs: " + ", ".join(missing))
    return requested


def crosstalk_requirements(package_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(package_dir / "camera_e2e_required_runs.csv")
    output = {}
    for row in rows:
        if row.get("requirement_id") == "fdtd_crosstalk_kernel_convergence":
            output[row["slug"]] = row
    return output


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    package_dir = args.package_dir
    sensor_rows = read_csv(package_dir / "camera_e2e_sensor_index.csv")
    field_rows = read_csv(package_dir / "camera_e2e_field_design_cases.csv")
    requirements = crosstalk_requirements(package_dir)
    fields_by_slug: dict[str, list[dict[str, str]]] = {}
    for row in field_rows:
        fields_by_slug.setdefault(row["slug"], []).append(row)

    tier = dict(TIER_CONFIG[args.tier])
    if args.resolution:
        tier["resolution"] = args.resolution
    if args.after_source_time:
        tier["after_source_time"] = args.after_source_time
    if args.wavelengths_nm:
        tier["wavelengths_nm"] = args.wavelengths_nm
    colors = tuple(item.strip() for item in (args.colors or ",".join(tier["colors"])).split(",") if item.strip())
    requested_case_names = tuple(
        item.strip() for item in (args.field_cases or ",".join(tier["field_cases"])).split(",") if item.strip()
    )
    slugs = selected_slugs([row["slug"] for row in sensor_rows], args.slugs)
    jobs = []
    for sensor in sensor_rows:
        slug = sensor["slug"]
        if slug not in slugs:
            continue
        stack_config = args.stack_dir / f"{slug}.json"
        tcad_profile = args.tcad_profile_dir / slug / "profile.json"
        if not stack_config.exists():
            continue
        mode = mode_for_ocl(sensor.get("ocl_mode_guess", ""))
        cfa_pattern = cfa_pattern_for(sensor.get("cfa_pattern", ""), sensor.get("ocl_mode_guess", ""))
        req = requirements.get(slug, {})
        sensor_resolution = int(finite_float(req.get("target_resolution_px_per_um"), tier["resolution"]))
        if args.resolution_sweep:
            sensor_resolutions = parse_resolution_sweep(args.resolution_sweep)
        elif args.resolution:
            sensor_resolutions = [int(args.resolution)]
        else:
            sensor_resolutions = [sensor_resolution]
        sensor_resolution = max(sensor_resolutions)
        sensor_resolution_text = ",".join(str(value) for value in sensor_resolutions)
        neighborhood = int_value(args.neighborhood or tier["neighborhood_override"], 0)
        if neighborhood <= 0:
            neighborhood = int_value(req.get("required_neighborhood"), 3)
        guard_cells = int_value(args.guard_cells if args.guard_cells >= 0 else tier["guard_override"], -1)
        if guard_cells < 0:
            guard_cells = int_value(req.get("guard_cells"), 1)
        required_neighborhood = int_value(req.get("required_neighborhood"), neighborhood)
        required_simulation_neighborhood = int_value(
            req.get("required_simulation_neighborhood"),
            required_neighborhood + 2 * int_value(req.get("guard_cells"), guard_cells),
        )
        required_guard_cells = int_value(req.get("guard_cells"), guard_cells)
        simulation_neighborhood = neighborhood + 2 * guard_cells
        boundary_gate, boundary_notes = boundary_domain_status(
            neighborhood=neighborhood,
            guard_cells=guard_cells,
            required_neighborhood=required_neighborhood,
            required_simulation_neighborhood=required_simulation_neighborhood,
            required_guard_cells=required_guard_cells,
        )
        estimated_voxels = estimated_crosstalk_voxels(stack_config, mode, neighborhood, guard_cells, sensor_resolution)
        resource_gate = "PASS"
        resource_notes = "estimated voxel count is within local runner limit"
        skip_resource = False
        if args.max_local_voxels > 0 and estimated_voxels > args.max_local_voxels:
            resource_gate = "CHECK"
            resource_notes = (
                f"estimated {estimated_voxels} voxels exceeds local limit {args.max_local_voxels}; "
                "use a batch/cluster runner or pass --max-local-voxels 0 to force"
            )
            skip_resource = True
        cases = case_string(fields_by_slug.get(slug, []), requested_case_names)
        params_by_case = case_params(cases)
        sweep_point_count = csv_count(cases) * csv_count(tier["wavelengths_nm"])
        for color in colors:
            job_id = f"{args.tier}_{slug}_{color}"
            output_dir = args.output_dir / slug / color
            command = [
                str(args.meep_python),
                "meep_crosstalk_kernel.py",
                "--modes",
                mode,
                "--neighborhoods",
                str(neighborhood),
                "--resolutions",
                sensor_resolution_text,
                "--wavelengths-nm",
                str(tier["wavelengths_nm"]),
                "--cases",
                cases,
                "--guard-cells",
                str(guard_cells),
                "--after-source-time",
                str(tier["after_source_time"]),
                "--stack-config",
                str(stack_config),
                "--tcad-profile",
                str(tcad_profile),
                "--color-channel",
                color,
                "--cfa-pattern",
                cfa_pattern,
                "--output-dir",
                str(output_dir),
            ]
            jobs.append(
                {
                    "job_id": job_id,
                    "tier": args.tier,
                    "slug": slug,
                    "code": sensor.get("code", ""),
                    "mode": mode,
                    "color_channel": color,
                    "cfa_pattern": cfa_pattern,
                    "wavelengths_nm": tier["wavelengths_nm"],
                    "field_cases": ",".join(requested_case_names),
                    "sweep_point_count": sweep_point_count,
                    "neighborhood": neighborhood,
                    "required_neighborhood": required_neighborhood,
                    "simulation_neighborhood": simulation_neighborhood,
                    "required_simulation_neighborhood": required_simulation_neighborhood,
                    "guard_cells": guard_cells,
                    "required_guard_cells": required_guard_cells,
                    "boundary_domain_gate": boundary_gate,
                    "boundary_domain_notes": boundary_notes,
                    "resolution_px_per_um": sensor_resolution_text,
                    "estimated_voxels": estimated_voxels,
                    "resource_gate": resource_gate,
                    "resource_notes": resource_notes,
                    "after_source_time": tier["after_source_time"],
                    "stack_config": repo_rel(stack_config),
                    "tcad_profile": repo_rel(tcad_profile) if tcad_profile.exists() else "",
                    "output_dir": repo_rel(output_dir),
                    "status": "PLANNED",
                    "returncode": "",
                    "duration_s": "",
                    "stdout": repo_rel(output_dir / "stdout.log"),
                    "stderr": repo_rel(output_dir / "stderr.log"),
                    "command": shlex.join(command),
                    "_command_list": command,
                    "_output_dir": output_dir,
                    "_skip_resource": skip_resource,
                    "_case_params": params_by_case,
                }
            )
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    return jobs


def solver_gate(row: dict[str, Any], convergence_status: str, tier: str) -> str:
    if tier == "smoke":
        return "CHECK"
    if str(row.get("boundary_domain_gate", "")).upper() == "FAIL":
        return "FAIL"
    if not boolish(row.get("grid_resolution_gate_pass")):
        return "FAIL"
    if str(convergence_status).upper() != "PASS":
        return "FAIL"
    truncation = finite_float(row.get("truncation_response_fraction"), math.nan)
    if math.isfinite(truncation) and truncation > 0.015:
        return "FAIL"
    return "PASS" if tier == "quantitative" else "CHECK"


def aggregate_summary(job: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = Path(job["_output_dir"])
    summary_path = output_dir / "crosstalk_kernel_summary.csv"
    kernel_path = output_dir / "crosstalk_kernel.json"
    convergence_status = ""
    if kernel_path.exists():
        payload = read_json(kernel_path)
        convergence_status = str(payload.get("convergence_status") or payload.get("convergence", {}).get("status", ""))
    if not summary_path.exists():
        return []
    rows = read_csv(summary_path)
    output = []
    for row in rows:
        params = job.get("_case_params", {}).get(row.get("case", ""), {})
        boundary_gate = str(job.get("boundary_domain_gate", ""))
        row_for_gate = dict(row)
        row_for_gate["boundary_domain_gate"] = boundary_gate
        gate = solver_gate(row_for_gate, convergence_status, str(job["tier"]))
        output.append(
            {
                "job_id": job["job_id"],
                "tier": job["tier"],
                "slug": job["slug"],
                "code": job["code"],
                "mode": job["mode"],
                "color_channel": job["color_channel"],
                "cfa_pattern": job["cfa_pattern"],
                "wavelength_nm": row.get("wavelength_nm", ""),
                "case": row.get("case", ""),
                "field_x_norm": row.get("field_x_norm", params.get("field_x_norm", "")),
                "field_z_norm": row.get("field_z_norm", params.get("field_z_norm", "")),
                "cra_x_deg": row.get("cra_x_deg", params.get("cra_x_deg", "")),
                "cra_z_deg": row.get("cra_z_deg", params.get("cra_z_deg", "")),
                "lens_shift_x_um": row.get("lens_shift_x_um", params.get("lens_shift_x_um", "")),
                "lens_shift_z_um": row.get("lens_shift_z_um", params.get("lens_shift_z_um", "")),
                "neighborhood": row.get("neighborhood", ""),
                "simulation_neighborhood": row.get("simulation_neighborhood", ""),
                "required_neighborhood": job.get("required_neighborhood", ""),
                "required_simulation_neighborhood": job.get("required_simulation_neighborhood", ""),
                "guard_cells": row.get("guard_cells", ""),
                "required_guard_cells": job.get("required_guard_cells", ""),
                "boundary_domain_gate": boundary_gate,
                "boundary_domain_notes": job.get("boundary_domain_notes", ""),
                "raw_pd_kernel_shape": row.get("raw_pd_kernel_shape", ""),
                "center_response_fraction": row.get("center_response_fraction", ""),
                "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": row.get("strongest_neighbor_fraction", ""),
                "truncation_response_fraction": row.get("truncation_response_fraction", ""),
                "grid_resolution_gate_pass": row.get("grid_resolution_gate_pass", ""),
                "convergence_status": convergence_status,
                "solver_gate": gate,
                "product_lut_ready": False,
                "source_summary_csv": repo_rel(summary_path),
                "source_kernel_json": repo_rel(kernel_path),
            }
        )
    return output


def resource_limit_summary(job: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cases = [item.strip() for item in str(job.get("field_cases", "")).split(",") if item.strip()]
    wavelengths = [item.strip() for item in str(job.get("wavelengths_nm", "")).split(",") if item.strip()]
    for wavelength in wavelengths:
        for case in cases:
            params = job.get("_case_params", {}).get(case, {})
            rows.append(
                {
                    "job_id": job["job_id"],
                    "tier": job["tier"],
                    "slug": job["slug"],
                    "code": job["code"],
                    "mode": job["mode"],
                    "color_channel": job["color_channel"],
                    "cfa_pattern": job["cfa_pattern"],
                    "wavelength_nm": wavelength,
                    "case": case,
                    "field_x_norm": params.get("field_x_norm", ""),
                    "field_z_norm": params.get("field_z_norm", ""),
                    "cra_x_deg": params.get("cra_x_deg", ""),
                    "cra_z_deg": params.get("cra_z_deg", ""),
                    "lens_shift_x_um": params.get("lens_shift_x_um", ""),
                    "lens_shift_z_um": params.get("lens_shift_z_um", ""),
                    "neighborhood": job.get("neighborhood", ""),
                    "simulation_neighborhood": int(job.get("neighborhood", 0)) + 2 * int(job.get("guard_cells", 0)),
                    "required_neighborhood": job.get("required_neighborhood", ""),
                    "required_simulation_neighborhood": job.get("required_simulation_neighborhood", ""),
                    "guard_cells": job.get("guard_cells", ""),
                    "required_guard_cells": job.get("required_guard_cells", ""),
                    "boundary_domain_gate": job.get("boundary_domain_gate", ""),
                    "boundary_domain_notes": job.get("boundary_domain_notes", ""),
                    "raw_pd_kernel_shape": "",
                    "center_response_fraction": "",
                    "output_crosstalk_fraction": "",
                    "strongest_neighbor_fraction": "",
                    "truncation_response_fraction": "",
                    "grid_resolution_gate_pass": "",
                    "estimated_voxels": job.get("estimated_voxels", ""),
                    "resource_gate": job.get("resource_gate", ""),
                    "resource_notes": job.get("resource_notes", ""),
                    "convergence_status": "RESOURCE_LIMIT",
                    "solver_gate": "CHECK",
                    "product_lut_ready": False,
                    "source_summary_csv": "",
                    "source_kernel_json": "",
                }
            )
    return rows


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_summary_rows: list[dict[str, Any]] = []
    for job in jobs:
        output_dir = Path(job["_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            job["status"] = "DRY_RUN"
            continue
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        if job.get("_skip_resource"):
            job["status"] = "SKIPPED_RESOURCE"
            job["returncode"] = ""
            job["duration_s"] = 0.0
            stdout_path.write_text(str(job.get("resource_notes", "")) + "\n", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            all_summary_rows.extend(resource_limit_summary(job))
            continue
        started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                result = subprocess.run(
                    job["_command_list"],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.timeout_s,
                    check=False,
                )
                returncode: int | str = result.returncode
                status = "DONE" if result.returncode == 0 else "FAIL"
            except subprocess.TimeoutExpired as exc:
                returncode = "TIMEOUT"
                status = "TIMEOUT"
                stderr.write(f"Timed out after {args.timeout_s} seconds: {shlex.join(job['_command_list'])}\n")
                if exc.stdout:
                    stdout.write(str(exc.stdout))
                if exc.stderr:
                    stderr.write(str(exc.stderr))
        job["duration_s"] = round(time.time() - started, 3)
        job["returncode"] = returncode
        job["status"] = status
        all_summary_rows.extend(aggregate_summary(job))
    public_jobs = []
    for job in jobs:
        clean = dict(job)
        clean.pop("_command_list", None)
        clean.pop("_output_dir", None)
        public_jobs.append(clean)
    return public_jobs, all_summary_rows


def write_html_report(
    output_dir: Path,
    report: dict[str, Any],
    jobs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    gate_counts: dict[str, int] = {}
    for row in summary_rows:
        gate = str(row.get("solver_gate", ""))
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Crosstalk Sweep</title>
  <style>
    :root {{ color-scheme: dark; --bg:#081118; --panel:#0f1c27; --line:#254255; --text:#e8f5ff; --muted:#9db6c8; --cyan:#55e4ff; --yellow:#ffd95f; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width:1360px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    h2 {{ margin:26px 0 10px; color:var(--cyan); font-size:19px; }}
    p {{ color:var(--muted); line-height:1.55; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .metric {{ font-size:26px; font-weight:800; }}
    .label {{ color:var(--muted); font-size:13px; }}
    .note {{ border-left:3px solid var(--yellow); padding-left:12px; color:var(--text); }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:10px; }}
    th, td {{ border:1px solid var(--line); padding:7px 8px; vertical-align:top; text-align:left; }}
    th {{ color:var(--cyan); background:#102633; position:sticky; top:0; }}
    code {{ color:#d8f8ff; }}
  </style>
</head>
<body>
<main>
  <h1>CameraE2E Crosstalk Sweep</h1>
  <p>Generated: <code>{html_cell(report.get("generated_at", ""))}</code></p>
  <div class="grid">
    <div class="card"><div class="metric">{report.get("job_count", 0)}</div><div class="label">jobs</div></div>
    <div class="card"><div class="metric">{report.get("completed_job_count", 0)}</div><div class="label">completed</div></div>
    <div class="card"><div class="metric">{report.get("summary_row_count", 0)}</div><div class="label">summary rows</div></div>
    <div class="card"><div class="metric">{report.get("total_sweep_point_count", 0)}</div><div class="label">sweep points</div></div>
  </div>
  <p>Solver gates: <code>{html_cell(gate_counts)}</code></p>
  <p class="note">This report is crosstalk execution evidence. Smoke/trend tiers are not product-ready CameraE2E kernels.</p>
  <h2>Manifest</h2>
  {html_table(jobs, MANIFEST_COLUMNS)}
  <h2>Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS, limit=80)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def update_package_links(
    package_dir: Path,
    output_dir: Path,
    report: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    pointer = {
        "schema": "camera_e2e_crosstalk_sweep_latest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "output_dir": repo_rel(output_dir),
        "html_report": repo_rel(output_dir / "index.html"),
    }
    write_json(package_dir / "camera_e2e_crosstalk_sweep_latest.json", pointer)
    package_path = package_dir / "camera_e2e_lut_package.json"
    if package_path.exists():
        package = read_json(package_path)
        package.setdefault("outputs", {})["latest_crosstalk_sweep_report"] = repo_rel(
            package_dir / "camera_e2e_crosstalk_sweep_latest.json"
        )
        package.setdefault("outputs", {})["latest_crosstalk_sweep_html"] = repo_rel(output_dir / "index.html")
        package["latest_crosstalk_sweep"] = {
            "tier": report.get("tier"),
            "summary_row_count": report.get("summary_row_count"),
            "completed_job_count": report.get("completed_job_count"),
            "product_lut_ready": False,
        }
        write_json(package_path, package)
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_slug.setdefault(str(row.get("slug", "")), []).append(row)
    for slug, rows in by_slug.items():
        lut_path = package_dir / "sensors" / slug / "camera_e2e_lut.json"
        if not lut_path.exists():
            continue
        lut = read_json(lut_path)
        gate_counts: dict[str, int] = {}
        for row in rows:
            gate = str(row.get("solver_gate", ""))
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
        lut["crosstalk_sweep_evidence"] = {
            "schema": "camera_e2e_crosstalk_sweep_evidence_v1",
            "tier": report.get("tier"),
            "summary_row_count": len(rows),
            "solver_gate_counts": dict(sorted(gate_counts.items())),
            "summary_csv": report.get("outputs", {}).get("summary_csv", ""),
            "html_report": repo_rel(output_dir / "index.html"),
            "product_lut_ready": False,
        }
        write_json(lut_path, lut)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--tcad-profile-dir", type=Path, default=DEFAULT_TCAD_PROFILE_DIR)
    parser.add_argument("--meep-python", type=Path, default=DEFAULT_MEEP_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tier", choices=tuple(TIER_CONFIG), default="smoke")
    parser.add_argument("--slugs", default="")
    parser.add_argument("--colors", default="")
    parser.add_argument("--field-cases", default="")
    parser.add_argument("--wavelengths-nm", default="")
    parser.add_argument("--resolution", type=int, default=0)
    parser.add_argument(
        "--resolution-sweep",
        default="",
        help="Comma-separated Meep resolutions for one convergence job, for example 67,72.",
    )
    parser.add_argument("--after-source-time", type=float, default=0.0)
    parser.add_argument("--neighborhood", type=int, default=0)
    parser.add_argument("--guard-cells", type=int, default=-1)
    parser.add_argument(
        "--max-local-voxels",
        type=int,
        default=50_000_000,
        help="Skip crosstalk jobs above this estimated voxel count; use 0 to force execution.",
    )
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.package_dir / f"crosstalk_sweep_{args.tier}"
    args.output_dir = args.output_dir.resolve()
    jobs = build_jobs(args)
    public_jobs, summary_rows = run_jobs(args, jobs)
    write_csv(args.output_dir / "crosstalk_sweep_manifest.csv", public_jobs, MANIFEST_COLUMNS)
    write_csv(args.output_dir / "crosstalk_sweep_summary.csv", summary_rows, SUMMARY_COLUMNS)
    report = {
        "schema": "camera_e2e_crosstalk_sweep_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "dry_run": args.dry_run,
        "job_count": len(public_jobs),
        "completed_job_count": sum(1 for row in public_jobs if row.get("status") == "DONE"),
        "failed_job_count": sum(1 for row in public_jobs if row.get("status") in {"FAIL", "TIMEOUT"}),
        "timeout_job_count": sum(1 for row in public_jobs if row.get("status") == "TIMEOUT"),
        "summary_row_count": len(summary_rows),
        "total_sweep_point_count": sum(int(row.get("sweep_point_count") or 0) for row in public_jobs),
        "product_lut_ready": False,
        "outputs": {
            "manifest_csv": repo_rel(args.output_dir / "crosstalk_sweep_manifest.csv"),
            "summary_csv": repo_rel(args.output_dir / "crosstalk_sweep_summary.csv"),
        },
        "notes": [
            "Smoke and trend tiers are execution/trend evidence only.",
            "Quantitative crosstalk requires convergence PASS plus measured stack/material before CameraE2E product use.",
        ],
    }
    write_json(args.output_dir / "crosstalk_sweep_report.json", report)
    write_html_report(args.output_dir, report, public_jobs, summary_rows)
    if not args.dry_run:
        update_package_links(args.package_dir, args.output_dir, report, summary_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
