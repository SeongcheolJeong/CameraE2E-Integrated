#!/usr/bin/env python3
"""Merge quantitative point-run evidence into CameraE2E package progress files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"

MERGED_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "solver",
    "color",
    "field_case",
    "wavelength_nm",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "mode",
    "cfa_pattern",
    "target_resolution_px_per_um",
    "actual_resolution_px_per_um",
    "total_response",
    "normalized_total_response_to_first",
    "signed_flux_si_absorption_fraction_diagnostic",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "focal_centroid_x_um",
    "focal_centroid_z_um",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "raw_pd_kernel_shape",
    "center_response_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "solver_gate",
    "grid_resolution_gate_pass",
    "convergence_status",
    "resource_gate",
    "estimated_voxels",
    "resource_notes",
    "product_lut_ready",
    "source_summary_csv",
    "source_detail_csv",
    "source_kernel_json",
    "source_report_json",
]

FIELD_LUT_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "color",
    "field_case",
    "wavelength_nm",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "total_response",
    "normalized_total_response_to_first",
    "signed_flux_si_absorption_fraction_diagnostic",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "focal_centroid_x_um",
    "focal_centroid_z_um",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
    "target_resolution_px_per_um",
    "actual_resolution_px_per_um",
    "solver_gate",
    "grid_resolution_gate_pass",
    "source_summary_csv",
    "source_detail_csv",
]

CROSSTALK_LUT_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "color",
    "field_case",
    "wavelength_nm",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "raw_pd_kernel_shape",
    "center_response_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "target_resolution_px_per_um",
    "actual_resolution_px_per_um",
    "estimated_voxels",
    "resource_gate",
    "resource_notes",
    "convergence_status",
    "solver_gate",
    "grid_resolution_gate_pass",
    "source_summary_csv",
    "source_kernel_json",
]

COVERAGE_COLUMNS = [
    "slug",
    "code",
    "solver",
    "required_points",
    "attempted_points",
    "completed_points",
    "resource_limited_points",
    "pass_points",
    "check_points",
    "fail_points",
    "coverage_fraction",
    "attempted_fraction",
    "gate",
    "product_lut_ready",
    "summary_csv",
    "notes",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def queue_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("slug", "")),
        str(row.get("solver", "")),
        str(row.get("color", "")),
        str(row.get("field_case", "")),
        str(row.get("wavelength_nm", "")),
        str(row.get("target_resolution_px_per_um", "")),
    )


def queue_id(slug: str, solver: str, color: str, field_case: str, wavelength_nm: str) -> str:
    return f"{slug}_{solver}_{color}_{field_case}_{wavelength_nm}"


def root_path(path_text: Any) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def first_csv_row(path_text: Any) -> dict[str, str]:
    path = root_path(path_text)
    if not path or not path.exists():
        return {}
    rows = read_csv(path)
    return rows[0] if rows else {}


def split_csv_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def timeout_summary_rows(solver: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_rel = report.get("outputs", {}).get("manifest_csv", "")
    manifest_path = ROOT / manifest_rel if manifest_rel else None
    if not manifest_path or not manifest_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for job in read_csv(manifest_path):
        if str(job.get("status", "")).upper() != "TIMEOUT":
            continue
        notes = (
            f"Timed out after {job.get('duration_s') or 'unknown'} seconds before producing quantitative KPI rows; "
            "run this point on a batch/HPC environment or reduce resolution only for non-product trend work."
        )
        for wavelength in split_csv_values(job.get("wavelengths_nm")):
            for case in split_csv_values(job.get("field_cases")):
                row = {
                    "job_id": job.get("job_id", ""),
                    "tier": job.get("tier", report.get("tier", "")),
                    "slug": job.get("slug", ""),
                    "code": job.get("code", ""),
                    "mode": job.get("mode", ""),
                    "collection_mode": job.get("collection_mode", ""),
                    "color_channel": job.get("color_channel", ""),
                    "cfa_pattern": job.get("cfa_pattern", ""),
                    "wavelength_nm": wavelength,
                    "case": case,
                    "field_x_norm": "",
                    "field_z_norm": "",
                    "cra_x_deg": "",
                    "cra_z_deg": "",
                    "lens_shift_x_um": "",
                    "lens_shift_z_um": "",
                    "grid_resolution_gate_pass": "",
                    "solver_gate": "CHECK",
                    "convergence_status": "RESOURCE_LIMIT",
                    "resource_gate": "RESOURCE_LIMIT",
                    "resource_notes": notes,
                    "product_lut_ready": False,
                    "source_summary_csv": "",
                    "source_kernel_json": "",
                }
                if solver == "crosstalk":
                    row.update(
                        {
                            "neighborhood": job.get("neighborhood", ""),
                            "simulation_neighborhood": job.get("simulation_neighborhood", ""),
                            "guard_cells": job.get("guard_cells", ""),
                            "estimated_voxels": job.get("estimated_voxels", ""),
                        }
                    )
                rows.append(row)
    return rows


def report_summaries(package_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    summaries: list[tuple[str, Path, dict[str, Any]]] = []
    patterns = [
        ("field", "fdtd_field_sweep_quantitative*/fdtd_field_sweep_report.json"),
        ("field", "quantitative_point_runs/**/fdtd_field_sweep_report.json"),
        ("crosstalk", "crosstalk_sweep_quantitative*/crosstalk_sweep_report.json"),
        ("crosstalk", "quantitative_point_runs/**/crosstalk_sweep_report.json"),
    ]
    seen: set[Path] = set()
    for solver, pattern in patterns:
        for report_path in sorted(package_dir.glob(pattern)):
            report_path = report_path.resolve()
            if report_path in seen:
                continue
            seen.add(report_path)
            report = read_json(report_path)
            if report.get("dry_run") is True:
                continue
            if str(report.get("tier", "")).lower() != "quantitative":
                continue
            summary_rel = report.get("outputs", {}).get("summary_csv", "")
            summary_path = ROOT / summary_rel if summary_rel else report_path.with_name(
                "fdtd_field_sweep_summary.csv" if solver == "field" else "crosstalk_sweep_summary.csv"
            )
            if summary_path.exists():
                summaries.append((solver, report_path, report))
    return summaries


def merged_rows(package_dir: Path) -> list[dict[str, Any]]:
    queue_rows = read_csv(package_dir / "camera_e2e_quantitative_point_queue.csv")
    queue_by_point: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in queue_rows:
        key = (
            row["slug"],
            row["solver"],
            row["color"],
            row["field_case"],
            row["wavelength_nm"],
        )
        queue_by_point[key] = row

    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for solver, report_path, report in report_summaries(package_dir):
        summary_rel = report.get("outputs", {}).get("summary_csv", "")
        summary_path = ROOT / summary_rel if summary_rel else report_path.with_name(
            "fdtd_field_sweep_summary.csv" if solver == "field" else "crosstalk_sweep_summary.csv"
        )
        summary_rows = read_csv(summary_path)
        if not summary_rows and int(report.get("timeout_job_count") or 0) > 0:
            summary_rows = timeout_summary_rows(solver, report)
        for row in summary_rows:
            detail_row = first_csv_row(row.get("source_summary_csv"))
            slug = row.get("slug", "")
            color = row.get("color_channel", "")
            field_case = row.get("case", "")
            wavelength = str(int(finite_float(row.get("wavelength_nm"), 0.0)))
            q = queue_by_point.get((slug, solver, color, field_case, wavelength), {})
            target_resolution = q.get("target_resolution_px_per_um", "")
            actual_resolution = row.get("resolution_px_per_um") or row.get("grid_dx_um", "")
            if solver == "field" and actual_resolution and "." in str(actual_resolution):
                dx = finite_float(actual_resolution)
                actual_resolution = f"{round(1.0 / dx):d}" if dx > 0 else ""
            output = {
                "queue_id": q.get("queue_id") or queue_id(slug, solver, color, field_case, wavelength),
                "slug": slug,
                "code": row.get("code", ""),
                "solver": solver,
                "color": color,
                "field_case": field_case,
                "wavelength_nm": wavelength,
                "field_x_norm": row.get("field_x_norm", detail_row.get("field_x_norm", "")),
                "field_z_norm": row.get("field_z_norm", detail_row.get("field_z_norm", "")),
                "cra_x_deg": row.get("cra_x_deg", detail_row.get("cra_x_deg", "")),
                "cra_z_deg": row.get("cra_z_deg", detail_row.get("cra_z_deg", "")),
                "lens_shift_x_um": row.get("lens_shift_x_um", detail_row.get("lens_shift_x_um", "")),
                "lens_shift_z_um": row.get("lens_shift_z_um", detail_row.get("lens_shift_z_um", "")),
                "mode": row.get("mode", detail_row.get("mode", "")),
                "cfa_pattern": row.get("cfa_pattern", detail_row.get("cfa_pattern", "")),
                "target_resolution_px_per_um": target_resolution,
                "actual_resolution_px_per_um": actual_resolution,
                "total_response": row.get("total_response", detail_row.get("total_response", "")),
                "normalized_total_response_to_first": row.get(
                    "normalized_total_response_to_first",
                    detail_row.get("normalized_total_response_to_first", ""),
                ),
                "signed_flux_si_absorption_fraction_diagnostic": row.get(
                    "signed_flux_si_absorption_fraction_diagnostic",
                    detail_row.get("signed_flux_si_absorption_fraction_diagnostic", ""),
                ),
                "split_phase_x_proxy": row.get("split_phase_x_proxy", detail_row.get("split_phase_x_proxy", "")),
                "split_phase_z_proxy": row.get("split_phase_z_proxy", detail_row.get("split_phase_z_proxy", "")),
                "focal_centroid_x_um": detail_row.get("focal_centroid_x_um", ""),
                "focal_centroid_z_um": detail_row.get("focal_centroid_z_um", ""),
                "focal_centroid_shift_x_um": detail_row.get("focal_centroid_shift_x_um", ""),
                "focal_centroid_shift_z_um": detail_row.get("focal_centroid_shift_z_um", ""),
                "focal_rms_radius_um": detail_row.get("focal_rms_radius_um", ""),
                "focal_target_fraction": detail_row.get("focal_target_fraction", ""),
                "neighborhood": row.get("neighborhood", ""),
                "simulation_neighborhood": row.get("simulation_neighborhood", ""),
                "guard_cells": row.get("guard_cells", ""),
                "raw_pd_kernel_shape": row.get("raw_pd_kernel_shape", ""),
                "center_response_fraction": row.get("center_response_fraction", ""),
                "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": row.get("strongest_neighbor_fraction", ""),
                "truncation_response_fraction": row.get("truncation_response_fraction", ""),
                "solver_gate": str(row.get("solver_gate", "")).upper(),
                "grid_resolution_gate_pass": row.get("grid_resolution_gate_pass", ""),
                "convergence_status": row.get("convergence_status", ""),
                "resource_gate": row.get("resource_gate", ""),
                "estimated_voxels": row.get("estimated_voxels", ""),
                "resource_notes": row.get("resource_notes", ""),
                "product_lut_ready": row.get("product_lut_ready", ""),
                "source_summary_csv": repo_rel(summary_path),
                "source_detail_csv": row.get("source_summary_csv", ""),
                "source_kernel_json": row.get("source_kernel_json", ""),
                "source_report_json": repo_rel(report_path),
            }
            # Keep the latest discovered row for each point. Deterministic path sorting
            # makes explicit rerun folders override earlier dry/pilot names when names sort later.
            merged[(slug, solver, color, field_case, wavelength)] = output
    return list(merged.values())


def coverage_rows(package_dir: Path, merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = read_csv(package_dir / "camera_e2e_quantitative_point_queue.csv")
    required: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in queue:
        required.setdefault((row["slug"], row["solver"]), []).append(row)
    completed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in merged:
        completed.setdefault((row["slug"], row["solver"]), []).append(row)

    rows = []
    for key in sorted(required):
        slug, solver = key
        required_rows = required[key]
        attempted_rows = completed.get(key, [])
        resource_limited_rows = [
            row
            for row in attempted_rows
            if str(row.get("convergence_status", "")).upper() == "RESOURCE_LIMIT"
            or str(row.get("resource_gate", "")).upper() == "RESOURCE_LIMIT"
        ]
        completed_rows = [row for row in attempted_rows if row not in resource_limited_rows]
        gates = [str(row.get("solver_gate", "")).upper() for row in attempted_rows]
        pass_count = gates.count("PASS")
        check_count = gates.count("CHECK")
        fail_count = gates.count("FAIL")
        if fail_count:
            gate = "FAIL"
            ready = False
            notes = "At least one quantitative point failed."
        elif resource_limited_rows:
            gate = "CHECK"
            ready = False
            notes = "At least one point is resource-limited and needs a batch/cluster run."
        elif len(completed_rows) < len(required_rows):
            gate = "CHECK" if completed_rows else "MISSING"
            ready = False
            notes = "Quantitative coverage is incomplete."
        elif check_count:
            gate = "CHECK"
            ready = False
            notes = "All points completed but at least one point is CHECK."
        else:
            gate = "PASS"
            ready = True
            notes = "All required quantitative points completed with PASS."
        rows.append(
            {
                "slug": slug,
                "code": required_rows[0].get("code", ""),
                "solver": solver,
                "required_points": len(required_rows),
                "attempted_points": len(attempted_rows),
                "completed_points": len(completed_rows),
                "resource_limited_points": len(resource_limited_rows),
                "pass_points": pass_count,
                "check_points": check_count,
                "fail_points": fail_count,
                "coverage_fraction": f"{(len(completed_rows) / max(1, len(required_rows))):.6f}",
                "attempted_fraction": f"{(len(attempted_rows) / max(1, len(required_rows))):.6f}",
                "gate": gate,
                "product_lut_ready": ready,
                "summary_csv": repo_rel(package_dir / "camera_e2e_quantitative_merged_summary.csv"),
                "notes": notes,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    merged = merged_rows(package_dir)
    coverage = coverage_rows(package_dir, merged)
    field_lut = [row for row in merged if row.get("solver") == "field"]
    crosstalk_lut = [row for row in merged if row.get("solver") == "crosstalk"]
    write_csv(package_dir / "camera_e2e_quantitative_merged_summary.csv", merged, MERGED_COLUMNS)
    write_csv(package_dir / "camera_e2e_quantitative_field_lut.csv", field_lut, FIELD_LUT_COLUMNS)
    write_csv(package_dir / "camera_e2e_quantitative_crosstalk_lut.csv", crosstalk_lut, CROSSTALK_LUT_COLUMNS)
    write_csv(package_dir / "camera_e2e_quantitative_coverage.csv", coverage, COVERAGE_COLUMNS)
    payload = {
        "schema": "camera_e2e_quantitative_progress_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merged_summary_csv": repo_rel(package_dir / "camera_e2e_quantitative_merged_summary.csv"),
        "field_lut_csv": repo_rel(package_dir / "camera_e2e_quantitative_field_lut.csv"),
        "crosstalk_lut_csv": repo_rel(package_dir / "camera_e2e_quantitative_crosstalk_lut.csv"),
        "coverage_csv": repo_rel(package_dir / "camera_e2e_quantitative_coverage.csv"),
        "merged_point_count": len(merged),
        "field_lut_row_count": len(field_lut),
        "crosstalk_lut_row_count": len(crosstalk_lut),
        "coverage_rows": coverage,
    }
    write_json(package_dir / "camera_e2e_quantitative_progress.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
