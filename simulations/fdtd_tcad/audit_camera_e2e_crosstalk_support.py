#!/usr/bin/env python3
"""Audit finite-array crosstalk kernel support for CameraE2E handoff.

This report compares low-resolution finite-array support pilots such as 3x3,
5x5, 7x7, ... output kernels. It is not a product-accuracy certificate. The
purpose is to prevent a too-small crosstalk kernel from being promoted to
CameraE2E product use when pilot runs already show large truncation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_camera_e2e_crosstalk_sweep import estimated_crosstalk_voxels, mode_for_ocl


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_crosstalk_support_audit"

PILOT_DIRS = {
    3: "crosstalk_boundary_pass_res20_pilot",
    5: "crosstalk_boundary_n5_res20_pilot",
    7: "crosstalk_boundary_n7_res20_pilot",
    9: "crosstalk_boundary_n9_res20_pilot",
    11: "crosstalk_boundary_n11_res20_pilot",
    13: "crosstalk_boundary_n13_res20_pilot",
    15: "crosstalk_boundary_n15_res20_pilot",
}

PILOT_COLUMNS = [
    "slug",
    "code",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "resolution_px_per_um",
    "estimated_voxels",
    "duration_s",
    "center_response_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "truncation_gate",
    "grid_resolution_gate_pass",
    "boundary_domain_gate",
    "convergence_status",
    "solver_gate",
    "product_use_gate",
    "stack_geometry_gate",
    "stack_geometry_notes",
    "summary_csv",
    "kernel_json",
]

CANDIDATE_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "requirement_id",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "mode",
    "neighborhood",
    "simulation_neighborhood",
    "guard_cells",
    "resolution_px_per_um",
    "estimated_voxels",
    "estimated_memory_class",
    "local_feasibility",
    "recommended_min_neighborhood",
    "support_evidence_gate",
    "candidate_support_role",
    "candidate_priority",
    "support_candidate_scope",
    "command",
]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "pilot_resolution_px_per_um",
    "best_pilot_neighborhood",
    "best_pilot_truncation_fraction",
    "truncation_threshold",
    "support_recommendation",
    "product_crosstalk_gate",
    "primary_blockers",
    "next_action",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def crosstalk_requirements_by_slug(package_dir: Path) -> dict[str, dict[str, str]]:
    requirements: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(package_dir / "camera_e2e_required_runs.csv"):
        if row.get("requirement_id") == "fdtd_crosstalk_kernel_convergence":
            requirements[row.get("slug", "")] = row
    return requirements


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in indexed:
            indexed[value] = row
    return indexed


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def abs_from_repo(path: str | Path | None) -> Path:
    if not path:
        return Path("")
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def current_stack_geometry(slug: str) -> dict[str, Any]:
    stack = read_json(ROOT / "image_sensor_db" / "generated_stack_configs" / f"{slug}.json")
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    return geometry if isinstance(geometry, dict) else {}


def kernel_run_geometry(kernel_json: str) -> dict[str, Any]:
    payload = read_json(abs_from_repo(kernel_json))
    config = payload.get("configuration", {}) if isinstance(payload.get("configuration"), dict) else {}
    geometry = config.get("geometry_um", {}) if isinstance(config.get("geometry_um"), dict) else {}
    return geometry if isinstance(geometry, dict) else {}


def stack_geometry_gate(slug: str, kernel_json: str) -> tuple[str, str]:
    run_geometry = kernel_run_geometry(kernel_json)
    if not run_geometry:
        return "UNKNOWN", "kernel JSON missing run geometry; rerun support pilot after current stack update"
    current = current_stack_geometry(slug)
    if not current:
        return "UNKNOWN", "current stack geometry missing"
    keys = ["pitch", "lens_height", "cfa_thickness", "passivation_thickness", "si_thickness"]
    mismatches: list[str] = []
    for key in keys:
        old = float_value(run_geometry.get(key), float("nan"))
        new = float_value(current.get(key), float("nan"))
        if not (old == old and new == new):
            continue
        if abs(old - new) > 1e-6:
            mismatches.append(f"{key}:run={old:g},current={new:g}")
    if mismatches:
        return "STALE", "; ".join(mismatches)
    return "PASS", "run geometry matches current stack geometry"


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def memory_class(voxels: int) -> str:
    if voxels <= 50_000_000:
        return "local_candidate"
    if voxels <= 250_000_000:
        return "large_workstation_or_small_cluster"
    if voxels <= 1_000_000_000:
        return "cluster_required"
    return "hpc_or_domain_decomposition_required"


def local_feasibility(voxels: int) -> str:
    if voxels <= 50_000_000:
        return "RUNNABLE_LOCAL_CHECK"
    if voxels <= 250_000_000:
        return "NOT_LOCAL_DEFAULT_USE_BATCH"
    return "NOT_LOCAL_REQUIRES_HPC_OR_REFORMULATION"


def support_key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row.get("slug", "")),
        str(row.get("color_channel", "") or row.get("color", "")),
        round(float_value(row.get("wavelength_nm"), -1.0), 6),
        str(row.get("field_case", "") or row.get("case", "")),
    )


def build_support_requirements(pilot_rows: list[dict[str, Any]], threshold: float) -> dict[tuple[str, str, float, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for row in pilot_rows:
        grouped.setdefault(support_key(row), []).append(row)
    requirements: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int_value(item.get("neighborhood")))
        best = rows[-1]
        valid_rows = [
            row
            for row in rows
            if str(row.get("boundary_domain_gate", "")).upper() == "PASS"
            and str(row.get("stack_geometry_gate", "")).upper() == "PASS"
        ]
        passing = [
            row
            for row in valid_rows
            if float_value(row.get("truncation_response_fraction"), 1.0) <= threshold
        ]
        if passing:
            recommended = int_value(passing[0].get("neighborhood"))
            gate = "LOW_RES_TRUNCATION_SUPPORT_ESTABLISHED"
        elif valid_rows:
            best_valid = valid_rows[-1]
            recommended = int_value(best_valid.get("neighborhood")) + 2
            gate = "LOW_RES_SUPPORT_STILL_INSUFFICIENT"
        else:
            recommended = int_value(best.get("neighborhood"), 15)
            gate = "LOW_RES_SUPPORT_STILL_INSUFFICIENT"
        best_evidence = valid_rows[-1] if valid_rows else best
        requirements[key] = {
            "recommended_min_neighborhood": recommended,
            "support_evidence_gate": gate,
            "best_pilot_neighborhood": int_value(best_evidence.get("neighborhood")),
            "best_pilot_truncation_fraction": best_evidence.get("truncation_response_fraction", ""),
        }
    return requirements


def candidate_role(neighborhood: int, support: dict[str, Any]) -> tuple[str, str]:
    recommended = int_value(support.get("recommended_min_neighborhood"), 0)
    if recommended <= 0:
        return "UNVERIFIED_SUPPORT_SIZE", "SUPPORT_DISCOVERY_REQUIRED"
    if neighborhood < recommended:
        return "BELOW_RECOMMENDED_SUPPORT", "SKIP_UNLESS_DIAGNOSTIC"
    if neighborhood == recommended:
        return "RECOMMENDED_MINIMUM_SUPPORT", "PRIMARY_PRODUCT_CANDIDATE"
    return "ABOVE_RECOMMENDED_SUPPORT", "SECONDARY_MARGIN_CANDIDATE"


def pilot_rows_from_summary(summary_path: Path, threshold: float, fallback_neighborhood: int | None = None) -> list[dict[str, Any]]:
    manifest_path = summary_path.with_name("crosstalk_sweep_manifest.csv")
    summary_rows = read_csv_rows(summary_path)
    manifest_rows = read_csv_rows(manifest_path)
    manifest = manifest_rows[0] if manifest_rows else {}
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        truncation = float_value(row.get("truncation_response_fraction"))
        grid_pass = str(row.get("grid_resolution_gate_pass", "")).strip().lower() in {"1", "true", "yes", "pass"}
        convergence = str(row.get("convergence_status", ""))
        solver_gate = str(row.get("solver_gate", ""))
        neighborhood = int_value(row.get("neighborhood"), fallback_neighborhood or 0)
        kernel_json = row.get("source_kernel_json", "")
        geometry_gate, geometry_notes = stack_geometry_gate(row.get("slug", ""), kernel_json)
        rows.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "color_channel": row.get("color_channel", ""),
                "wavelength_nm": row.get("wavelength_nm", ""),
                "field_case": row.get("case", ""),
                "neighborhood": neighborhood,
                "simulation_neighborhood": row.get("simulation_neighborhood", ""),
                "guard_cells": row.get("guard_cells", ""),
                "resolution_px_per_um": row.get("resolution_px_per_um", "") or manifest.get("resolution_px_per_um", ""),
                "estimated_voxels": row.get("estimated_voxels", "") or manifest.get("estimated_voxels", ""),
                "duration_s": row.get("duration_s", "") or manifest.get("duration_s", ""),
                "center_response_fraction": row.get("center_response_fraction", ""),
                "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": row.get("strongest_neighbor_fraction", ""),
                "truncation_response_fraction": row.get("truncation_response_fraction", ""),
                "truncation_gate": "PASS" if truncation <= threshold else "FAIL",
                "grid_resolution_gate_pass": grid_pass,
                "boundary_domain_gate": row.get("boundary_domain_gate", ""),
                "convergence_status": convergence,
                "solver_gate": solver_gate,
                "product_use_gate": "PASS" if grid_pass and convergence == "PASS" and solver_gate == "PASS" else "FAIL",
                "stack_geometry_gate": geometry_gate,
                "stack_geometry_notes": geometry_notes,
                "summary_csv": repo_rel(summary_path),
                "kernel_json": kernel_json,
            }
        )
    return rows


def load_pilot_rows(package_dir: Path, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for neighborhood, dirname in PILOT_DIRS.items():
        summary_path = (package_dir / dirname / "crosstalk_sweep_summary.csv").resolve()
        if summary_path in seen:
            continue
        seen.add(summary_path)
        rows.extend(pilot_rows_from_summary(summary_path, threshold, neighborhood))
    for summary_path in sorted((package_dir / "crosstalk_support_discovery").glob("**/crosstalk_sweep_summary.csv")):
        summary_path = summary_path.resolve()
        if summary_path in seen:
            continue
        seen.add(summary_path)
        rows.extend(pilot_rows_from_summary(summary_path, threshold))
    return rows


def build_candidate_rows(package_dir: Path, pilot_rows: list[dict[str, Any]], product_resolution: int, threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    support_requirements = build_support_requirements(pilot_rows, threshold=threshold)
    requirements = crosstalk_requirements_by_slug(package_dir)
    queue_rows = [
        row
        for row in read_csv_rows(package_dir / "camera_e2e_quantitative_point_queue.csv")
        if str(row.get("solver", "")).strip().lower() == "crosstalk"
    ]
    sensor_by_slug = index_by(
        read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv"),
        "slug",
    )
    if not queue_rows:
        queue_rows = [
            {
                "queue_id": f"{row.get('slug', '')}_crosstalk_{row.get('color_channel', '')}_{row.get('field_case', '')}_{row.get('wavelength_nm', '')}",
                "slug": str(row.get("slug", "")),
                "code": str(row.get("code", "")),
                "requirement_id": "finite_array_support_pilot_followup",
                "color": str(row.get("color_channel", "green")),
                "field_case": str(row.get("field_case", "center")),
                "wavelength_nm": str(row.get("wavelength_nm", "550")).replace(".0", ""),
                "target_resolution_px_per_um": str(product_resolution),
            }
            for row in pilot_rows
            if row.get("slug")
        ]

    for queue_row in queue_rows:
        slug = str(queue_row.get("slug", ""))
        stack_config = ROOT / "image_sensor_db" / "generated_stack_configs" / f"{slug}.json"
        if not stack_config.exists():
            continue
        sensor = sensor_by_slug.get(slug, {})
        mode = mode_for_ocl(sensor.get("ocl_mode_guess", ""))
        queue_resolution = int_value(queue_row.get("target_resolution_px_per_um"), product_resolution)
        color = str(queue_row.get("color", "") or queue_row.get("color_channel", "") or "green")
        field_case = str(queue_row.get("field_case", "") or "center")
        wavelength = str(queue_row.get("wavelength_nm", "") or "550")
        requirement = requirements.get(slug, {})
        required_guard = max(1, int_value(requirement.get("guard_cells"), 1))
        support = support_requirements.get((slug, color, round(float_value(wavelength, -1.0), 6), field_case), {})
        recommended = int_value(support.get("recommended_min_neighborhood"), 0)
        neighborhoods = [3, 5, 7, 9, 11, 13, 15]
        if recommended > 15 and recommended % 2 == 1:
            neighborhoods.append(recommended)
        for neighborhood in sorted(set(neighborhoods)):
            guard = required_guard
            voxels = estimated_crosstalk_voxels(stack_config, mode, neighborhood, guard, queue_resolution)
            role, priority = candidate_role(neighborhood, support)
            command = (
                "python3 run_camera_e2e_crosstalk_sweep.py "
                f"--tier quantitative --slugs {slug} --colors {color} --field-cases {field_case} --wavelengths-nm {wavelength} "
                f"--resolution {queue_resolution} --neighborhood {neighborhood} --guard-cells {guard} "
                "--max-local-voxels 0 "
                f"--output-dir runs/camera_e2e_sensor_lut_package/quantitative_point_runs/{slug}/crosstalk/{color}/{field_case}/{wavelength}nm/support_n{neighborhood}_g{guard}_res{queue_resolution}"
            )
            candidates.append(
                {
                    "queue_id": queue_row.get("queue_id", ""),
                    "slug": slug,
                    "code": queue_row.get("code", sensor.get("code", "")),
                    "requirement_id": queue_row.get("requirement_id", "fdtd_crosstalk_kernel_convergence"),
                    "color_channel": color,
                    "wavelength_nm": wavelength,
                    "field_case": field_case,
                    "mode": mode,
                    "neighborhood": neighborhood,
                    "simulation_neighborhood": neighborhood + 2 * guard,
                    "guard_cells": guard,
                    "resolution_px_per_um": queue_resolution,
                    "estimated_voxels": voxels,
                    "estimated_memory_class": memory_class(voxels),
                    "local_feasibility": local_feasibility(voxels),
                    "recommended_min_neighborhood": support.get("recommended_min_neighborhood", ""),
                    "support_evidence_gate": support.get("support_evidence_gate", "NO_SUPPORT_EVIDENCE"),
                    "candidate_support_role": role,
                    "candidate_priority": priority,
                    "support_candidate_scope": "full_quantitative_crosstalk_queue",
                    "command": command,
                }
            )
    return candidates


def build_sensor_rows(pilot_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in pilot_rows:
        key = (
            str(row.get("slug", "")),
            str(row.get("code", "")),
            str(row.get("color_channel", "")),
            str(row.get("wavelength_nm", "")),
            str(row.get("field_case", "")),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (slug, code, color, wavelength, field_case), items in sorted(groups.items()):
        items = sorted(items, key=lambda row: int_value(row.get("neighborhood")))
        best = items[-1]
        valid_items = [
            row
            for row in items
            if str(row.get("boundary_domain_gate", "")).upper() == "PASS"
            and str(row.get("stack_geometry_gate", "")).upper() == "PASS"
        ]
        passing = [
            row
            for row in valid_items
            if float_value(row.get("truncation_response_fraction"), 1.0) <= threshold
        ]
        if passing:
            support = int_value(passing[0].get("neighborhood"))
            recommendation = f"low-res pilot first meets truncation at {support}x{support}; confirm at product resolution"
            blocker_prefix = ""
        elif valid_items:
            best_valid = valid_items[-1]
            support = int_value(best_valid.get("neighborhood")) + 2
            recommendation = (
                f"boundary-valid low-res pilots up to {best_valid.get('neighborhood')}x{best_valid.get('neighborhood')} "
                f"still exceed truncation threshold; try at least {support}x{support} support"
            )
            blocker_prefix = ""
        else:
            support = int_value(best.get("neighborhood"), 15)
            stale_count = sum(1 for row in items if str(row.get("stack_geometry_gate", "")).upper() == "STALE")
            unknown_count = sum(1 for row in items if str(row.get("stack_geometry_gate", "")).upper() == "UNKNOWN")
            if stale_count:
                support_note = f"{stale_count} low-res pilot row(s) are stale after current stack/CFA geometry update; "
                blocker_prefix = "stale support pilot after stack/CFA geometry update; "
            elif unknown_count:
                support_note = f"{unknown_count} low-res pilot row(s) have unknown stack geometry provenance; "
                blocker_prefix = "support pilot stack geometry provenance missing; "
            else:
                support_note = ""
                blocker_prefix = "boundary-valid support pilot missing; "
            recommendation = (
                f"{support_note}no boundary-valid/current-stack low-res pilot exists yet; rerun at {support}x{support} with required guard cells "
                "before treating truncation as support evidence"
            )
        rows.append(
            {
                "slug": slug,
                "code": code,
                "color_channel": color,
                "wavelength_nm": wavelength,
                "field_case": field_case,
                "pilot_resolution_px_per_um": (valid_items[-1] if valid_items else best).get("resolution_px_per_um", ""),
                "best_pilot_neighborhood": (valid_items[-1] if valid_items else best).get("neighborhood", ""),
                "best_pilot_truncation_fraction": (valid_items[-1] if valid_items else best).get("truncation_response_fraction", ""),
                "truncation_threshold": threshold,
                "support_recommendation": recommendation,
                "product_crosstalk_gate": "FAIL",
                "primary_blockers": (
                    blocker_prefix
                    + "low-res grid gate fails; no high-resolution convergence pass; "
                    "measured stack/n,k still missing"
                ),
                "next_action": (
                    "run expanded-support product-resolution crosstalk on batch/HPC or develop a validated "
                    "domain-decomposition/surrogate workflow calibrated to high-res finite-array points"
                ),
            }
        )
    return rows


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], pilot_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1400px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px;color:#65e7ff}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#9fe8ff}
"""
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Crosstalk Support Audit</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Crosstalk Support Audit</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Low-resolution support pilots are used only to detect kernel-support risk; they are not product crosstalk evidence.</p>
<div class="grid">
<div class="card"><div class="metric fail">{html_cell(payload.get("product_crosstalk_ready", False))}</div><div class="muted">product crosstalk ready</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("pilot_row_count", 0))}</div><div class="muted">pilot rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("min_truncation_fraction", ""))}</div><div class="muted">best truncation</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("truncation_threshold", ""))}</div><div class="muted">threshold</div></div>
</div>
<h2>Sensor Recommendation</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Pilot Support Sweep</h2>{html_table(pilot_rows, PILOT_COLUMNS)}
<h2>Product Candidate Commands</h2>{html_table(candidate_rows, CANDIDATE_COLUMNS)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--truncation-threshold", type=float, default=0.015)
    parser.add_argument("--product-resolution", type=int, default=84)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    pilot_rows = load_pilot_rows(package_dir, args.truncation_threshold)
    candidate_rows = build_candidate_rows(package_dir, pilot_rows, args.product_resolution, args.truncation_threshold)
    sensor_rows = build_sensor_rows(pilot_rows, args.truncation_threshold)
    min_trunc = min((float_value(row.get("truncation_response_fraction"), 1.0) for row in pilot_rows), default=1.0)
    stale_pilot_count = sum(1 for row in pilot_rows if str(row.get("stack_geometry_gate", "")).upper() == "STALE")
    unknown_geometry_pilot_count = sum(1 for row in pilot_rows if str(row.get("stack_geometry_gate", "")).upper() == "UNKNOWN")

    outputs = {
        "json": repo_rel(output_dir / "camera_e2e_crosstalk_support_audit.json"),
        "sensor_csv": repo_rel(output_dir / "camera_e2e_crosstalk_support_by_sensor.csv"),
        "pilot_csv": repo_rel(output_dir / "camera_e2e_crosstalk_support_pilots.csv"),
        "candidate_csv": repo_rel(output_dir / "camera_e2e_crosstalk_product_candidates.csv"),
        "html": repo_rel(output_dir / "index.html"),
    }
    payload = {
        "schema": "camera_e2e_crosstalk_support_audit_v1",
        "artifact_role": "crosstalk_support_and_feasibility_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "truncation_threshold": args.truncation_threshold,
        "product_resolution_px_per_um": args.product_resolution,
        "pilot_row_count": len(pilot_rows),
        "stale_pilot_row_count": stale_pilot_count,
        "unknown_geometry_pilot_row_count": unknown_geometry_pilot_count,
        "candidate_row_count": len(candidate_rows),
        "product_candidate_count": len(candidate_rows),
        "sensor_row_count": len(sensor_rows),
        "min_truncation_fraction": f"{min_trunc:.6f}",
        "product_crosstalk_ready": False,
        "validation": {
            "schema": "camera_e2e_crosstalk_support_audit_validation_v1",
            "pass": True,
            "status": "RESEARCH_SUPPORT_AUDIT_READY_PRODUCT_BLOCKED",
            "issues": [],
        },
        "outputs": outputs,
        "sensor_rows": sensor_rows,
    }
    write_csv(output_dir / "camera_e2e_crosstalk_support_by_sensor.csv", sensor_rows, SENSOR_COLUMNS)
    write_csv(output_dir / "camera_e2e_crosstalk_support_pilots.csv", pilot_rows, PILOT_COLUMNS)
    write_csv(output_dir / "camera_e2e_crosstalk_product_candidates.csv", candidate_rows, CANDIDATE_COLUMNS)
    write_json(output_dir / "camera_e2e_crosstalk_support_audit.json", payload)
    write_html(output_dir / "index.html", payload, sensor_rows, pilot_rows, candidate_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
