#!/usr/bin/env python3
"""Run CameraE2E field-response FDTD jobs from the sensor LUT package.

This is the execution bridge between the gate-driven package produced by
build_camera_e2e_sensor_luts.py and meep_supercell_lut.py. It keeps smoke,
trend, and quantitative runs separate so downstream CameraE2E code cannot
mistake a low-resolution execution check for a product-quality LUT.
"""

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
DEFAULT_MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"

MANIFEST_COLUMNS = [
    "job_id",
    "tier",
    "slug",
    "code",
    "mode",
    "collection_mode",
    "color_channel",
    "cfa_pattern",
    "wavelengths_nm",
    "field_cases",
    "sweep_point_count",
    "resolution_px_per_um",
    "after_source_time",
    "grid_snap_y",
    "stack_config",
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
    "collection_mode",
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
    "total_response",
    "normalized_total_response_to_first",
    "max_region_response",
    "min_region_response",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "signed_flux_si_absorption_fraction_diagnostic",
    "grid_dx_um",
    "si_internal_wavelength_pixels",
    "minimum_critical_feature_pixels",
    "grid_resolution_gate_pass",
    "solver_gate",
    "convergence_status",
    "resource_gate",
    "resource_notes",
    "product_lut_ready",
    "source_summary_csv",
]


TIER_CONFIG = {
    "smoke": {
        "resolution": 4,
        "after_source_time": 8.0,
        "wavelengths_nm": "550",
        "field_cases": ("center", "x_plus_edge"),
        "colors": ("green",),
        "grid_snap_y": "nearest",
        "quantitative": False,
    },
    "trend": {
        "resolution": 20,
        "after_source_time": 25.0,
        "wavelengths_nm": "450,550,620",
        "field_cases": ("center", "x_plus_edge", "z_plus_edge", "diag_plus_plus"),
        "colors": ("red", "green", "blue"),
        "grid_snap_y": "nearest",
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
        "grid_snap_y": "nearest",
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def mode_for_ocl(ocl_mode: str) -> str:
    if ocl_mode == "quad_2x2":
        return "ocl-2x2"
    if ocl_mode == "nona_3x3":
        return "ocl-3x3"
    return "split-pd-1x1"


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


def field_requirements(package_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(package_dir / "camera_e2e_required_runs.csv")
    output = {}
    for row in rows:
        if row.get("requirement_id") == "fdtd_cra_rgb_field_sweep":
            output[row["slug"]] = row
    return output


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    package_dir = args.package_dir
    sensor_rows = read_csv(package_dir / "camera_e2e_sensor_index.csv")
    field_rows = read_csv(package_dir / "camera_e2e_field_design_cases.csv")
    requirements = field_requirements(package_dir)
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
        if not stack_config.exists():
            continue
        req = requirements.get(slug, {})
        sensor_resolution = int(finite_float(req.get("target_resolution_px_per_um"), tier["resolution"]))
        if args.resolution:
            sensor_resolution = int(args.resolution)
        mode = mode_for_ocl(sensor.get("ocl_mode_guess", ""))
        collection_mode = "pixel"
        cfa_pattern = cfa_pattern_for(sensor.get("cfa_pattern", ""), sensor.get("ocl_mode_guess", ""))
        cases = case_string(fields_by_slug.get(slug, []), requested_case_names)
        sweep_point_count = csv_count(cases) * csv_count(tier["wavelengths_nm"])
        for color in colors:
            job_id = f"{args.tier}_{slug}_{color}"
            output_dir = args.output_dir / slug / color
            command = [
                str(args.meep_python),
                "meep_supercell_lut.py",
                "--mode",
                mode,
                "--collection-mode",
                collection_mode,
                "--wavelengths-nm",
                str(tier["wavelengths_nm"]),
                "--cases",
                cases,
                "--resolution",
                str(sensor_resolution),
                "--after-source-time",
                str(tier["after_source_time"]),
                "--grid-snap-y",
                str(tier["grid_snap_y"]),
                "--stack-config",
                str(stack_config),
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
                    "collection_mode": collection_mode,
                    "color_channel": color,
                    "cfa_pattern": cfa_pattern,
                    "wavelengths_nm": tier["wavelengths_nm"],
                    "field_cases": ",".join(requested_case_names),
                    "sweep_point_count": sweep_point_count,
                    "resolution_px_per_um": sensor_resolution,
                    "after_source_time": tier["after_source_time"],
                    "grid_snap_y": tier["grid_snap_y"],
                    "stack_config": repo_rel(stack_config),
                    "output_dir": repo_rel(output_dir),
                    "status": "PLANNED",
                    "returncode": "",
                    "duration_s": "",
                    "stdout": repo_rel(output_dir / "stdout.log"),
                    "stderr": repo_rel(output_dir / "stderr.log"),
                    "command": shlex.join(command),
                    "_command_list": command,
                    "_output_dir": output_dir,
                }
            )
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    return jobs


def solver_gate(row: dict[str, Any], tier: str) -> str:
    if not boolish(row.get("grid_resolution_gate_pass")):
        return "CHECK" if tier == "smoke" else "FAIL"
    signed_flux = finite_float(row.get("signed_flux_si_absorption_fraction_diagnostic"), math.nan)
    if math.isfinite(signed_flux) and signed_flux < 0:
        return "CHECK" if tier == "smoke" else "FAIL"
    return "PASS" if tier == "quantitative" else "CHECK"


def aggregate_summary(job: dict[str, Any]) -> list[dict[str, Any]]:
    summary_path = Path(job["_output_dir"]) / "camera_lut_summary.csv"
    if not summary_path.exists():
        return []
    rows = read_csv(summary_path)
    output = []
    for row in rows:
        gate = solver_gate(row, str(job["tier"]))
        output.append(
            {
                "job_id": job["job_id"],
                "tier": job["tier"],
                "slug": job["slug"],
                "code": job["code"],
                "mode": job["mode"],
                "collection_mode": job["collection_mode"],
                "color_channel": job["color_channel"],
                "cfa_pattern": job["cfa_pattern"],
                "wavelength_nm": row.get("wavelength_nm", ""),
                "case": row.get("case", ""),
                "field_x_norm": row.get("field_x_norm", ""),
                "field_z_norm": row.get("field_z_norm", ""),
                "cra_x_deg": row.get("cra_x_deg", ""),
                "cra_z_deg": row.get("cra_z_deg", ""),
                "lens_shift_x_um": row.get("lens_shift_x_um", ""),
                "lens_shift_z_um": row.get("lens_shift_z_um", ""),
                "total_response": row.get("total_response", ""),
                "normalized_total_response_to_first": row.get("normalized_total_response_to_first", ""),
                "max_region_response": row.get("max_region_response", ""),
                "min_region_response": row.get("min_region_response", ""),
                "split_phase_x_proxy": row.get("split_phase_x_proxy", ""),
                "split_phase_z_proxy": row.get("split_phase_z_proxy", ""),
                "signed_flux_si_absorption_fraction_diagnostic": row.get(
                    "signed_flux_si_absorption_fraction_diagnostic", ""
                ),
                "grid_dx_um": row.get("grid_dx_um", ""),
                "si_internal_wavelength_pixels": row.get("si_internal_wavelength_pixels", ""),
                "minimum_critical_feature_pixels": row.get("minimum_critical_feature_pixels", ""),
                "grid_resolution_gate_pass": row.get("grid_resolution_gate_pass", ""),
                "solver_gate": gate,
                "convergence_status": row.get("convergence_status", ""),
                "resource_gate": row.get("resource_gate", ""),
                "resource_notes": row.get("resource_notes", ""),
                "product_lut_ready": False,
                "source_summary_csv": repo_rel(summary_path),
            }
        )
    return output


def timeout_summary(job: dict[str, Any], timeout_s: float) -> list[dict[str, Any]]:
    wavelengths = [item.strip() for item in str(job.get("wavelengths_nm", "")).split(",") if item.strip()]
    cases = [item.strip() for item in str(job.get("field_cases", "")).split(",") if item.strip()]
    notes = (
        f"Timed out after {timeout_s} seconds before producing a field summary; "
        "run on batch/HPC, reduce the point set, or lower resolution for non-quantitative trend work."
    )
    rows = []
    for wavelength in wavelengths:
        for case in cases:
            rows.append(
                {
                    "job_id": job["job_id"],
                    "tier": job["tier"],
                    "slug": job["slug"],
                    "code": job["code"],
                    "mode": job["mode"],
                    "collection_mode": job["collection_mode"],
                    "color_channel": job["color_channel"],
                    "cfa_pattern": job["cfa_pattern"],
                    "wavelength_nm": wavelength,
                    "case": case,
                    "field_x_norm": "",
                    "field_z_norm": "",
                    "cra_x_deg": "",
                    "cra_z_deg": "",
                    "lens_shift_x_um": "",
                    "lens_shift_z_um": "",
                    "total_response": "",
                    "normalized_total_response_to_first": "",
                    "max_region_response": "",
                    "min_region_response": "",
                    "split_phase_x_proxy": "",
                    "split_phase_z_proxy": "",
                    "signed_flux_si_absorption_fraction_diagnostic": "",
                    "grid_dx_um": "",
                    "si_internal_wavelength_pixels": "",
                    "minimum_critical_feature_pixels": "",
                    "grid_resolution_gate_pass": "",
                    "solver_gate": "CHECK",
                    "convergence_status": "RESOURCE_LIMIT",
                    "resource_gate": "RESOURCE_LIMIT",
                    "resource_notes": notes,
                    "product_lut_ready": False,
                    "source_summary_csv": "",
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
        if status == "TIMEOUT":
            all_summary_rows.extend(timeout_summary(job, args.timeout_s))
        else:
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
  <title>CameraE2E FDTD Field Sweep</title>
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
  <h1>CameraE2E FDTD Field Sweep</h1>
  <p>Generated: <code>{html_cell(report.get("generated_at", ""))}</code></p>
  <div class="grid">
    <div class="card"><div class="metric">{report.get("job_count", 0)}</div><div class="label">jobs</div></div>
    <div class="card"><div class="metric">{report.get("completed_job_count", 0)}</div><div class="label">completed</div></div>
    <div class="card"><div class="metric">{report.get("summary_row_count", 0)}</div><div class="label">summary rows</div></div>
    <div class="card"><div class="metric">{report.get("total_sweep_point_count", 0)}</div><div class="label">sweep points</div></div>
  </div>
  <p>Solver gates: <code>{html_cell(gate_counts)}</code></p>
  <p class="note">This report is execution evidence. Smoke and trend tiers are not product-ready CameraE2E LUTs.</p>
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
        "schema": "camera_e2e_fdtd_field_sweep_latest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": report,
        "output_dir": repo_rel(output_dir),
        "html_report": repo_rel(output_dir / "index.html"),
    }
    write_json(package_dir / "camera_e2e_fdtd_field_sweep_latest.json", pointer)
    package_path = package_dir / "camera_e2e_lut_package.json"
    if package_path.exists():
        package = read_json(package_path)
        package.setdefault("outputs", {})["latest_fdtd_field_sweep_report"] = repo_rel(
            package_dir / "camera_e2e_fdtd_field_sweep_latest.json"
        )
        package.setdefault("outputs", {})["latest_fdtd_field_sweep_html"] = repo_rel(output_dir / "index.html")
        package["latest_fdtd_field_sweep"] = {
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
        lut["fdtd_field_sweep_evidence"] = {
            "schema": "camera_e2e_fdtd_field_sweep_evidence_v1",
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
    parser.add_argument("--meep-python", type=Path, default=DEFAULT_MEEP_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tier", choices=tuple(TIER_CONFIG), default="smoke")
    parser.add_argument("--slugs", default="")
    parser.add_argument("--colors", default="")
    parser.add_argument("--field-cases", default="")
    parser.add_argument("--wavelengths-nm", default="")
    parser.add_argument("--resolution", type=int, default=0)
    parser.add_argument("--after-source-time", type=float, default=0.0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.package_dir / f"fdtd_field_sweep_{args.tier}"
    args.output_dir = args.output_dir.resolve()
    jobs = build_jobs(args)
    public_jobs, summary_rows = run_jobs(args, jobs)
    write_csv(args.output_dir / "fdtd_field_sweep_manifest.csv", public_jobs, MANIFEST_COLUMNS)
    write_csv(args.output_dir / "fdtd_field_sweep_summary.csv", summary_rows, SUMMARY_COLUMNS)
    report = {
        "schema": "camera_e2e_fdtd_field_sweep_report_v1",
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
            "manifest_csv": repo_rel(args.output_dir / "fdtd_field_sweep_manifest.csv"),
            "summary_csv": repo_rel(args.output_dir / "fdtd_field_sweep_summary.csv"),
        },
        "notes": [
            "Smoke and trend tiers are execution/trend evidence only.",
            "Quantitative tier still requires measured stack/material and convergence gates before CameraE2E product use.",
        ],
    }
    write_json(args.output_dir / "fdtd_field_sweep_report.json", report)
    write_html_report(args.output_dir, report, public_jobs, summary_rows)
    if not args.dry_run:
        update_package_links(args.package_dir, args.output_dir, report, summary_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
