#!/usr/bin/env python3
"""Plan the next concrete work needed to close CameraE2E LUT readiness gaps.

The plan is generated from current readiness, coverage, and point-queue
artifacts. It separates non-runnable measured-data blockers from runnable solver
jobs and emits exact commands for the next prioritized solver batches.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_closure_plan"

PLAN_COLUMNS = [
    "plan_id",
    "priority",
    "track",
    "runnable",
    "blocking_gate",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "solver",
    "queue_id",
    "color",
    "field_case",
    "wavelength_nm",
    "target_resolution_px_per_um",
    "estimated_hours",
    "expected_success_gate",
    "command",
    "expected_artifact",
    "why_this_next",
    "notes",
]

BATCH_COLUMNS = [
    "batch_id",
    "priority",
    "track",
    "slug",
    "solver",
    "queue_id_count",
    "estimated_hours_sum",
    "command",
    "queue_ids",
    "notes",
]

CHECK_COLUMNS = [
    "check_id",
    "pass",
    "status",
    "evidence",
    "required_action",
]

FIELD_CASE_PRIORITY = {
    "center": 0,
    "x_minus_edge": 1,
    "x_plus_edge": 1,
    "z_minus_edge": 2,
    "z_plus_edge": 2,
    "diag_minus_minus": 3,
    "diag_minus_plus": 3,
    "diag_plus_minus": 3,
    "diag_plus_plus": 3,
}
COLOR_PRIORITY = {"green": 0, "red": 1, "blue": 2}
WAVELENGTH_PRIORITY = {"550": 0, "620": 1, "450": 2}
TRACK_PRIORITY = {
    "measured_input": 0,
    "measured_calibration_input": 1,
    "solver_crosstalk_product_primary": 2,
    "solver_crosstalk_support_discovery": 3,
    "solver_resource_limited_batch": 4,
    "solver_quantitative": 5,
}
ISSUE_PRIORITY = {
    "cra_input_not_measured": "P0",
    "stack_material_not_measured": "P0",
    "field_fdtd_coverage_incomplete": "P0",
    "finite_array_crosstalk_not_pass": "P0",
    "measured_pixel_electrical_calibration_missing": "P0",
    "measured_readout_raw_calibration_missing": "P0",
    "measured_module_coupling_calibration_missing": "P0",
    "compact_crosstalk_surrogate": "P1",
    "combined_query_rows_excluded": "P1",
}

COVERAGE_CLOSURE_DOMAINS = {"Pixel / Electrical", "Readout / RAW", "Module Coupling"}


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


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def priority_rank(value: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}.get(str(value), 9)


def queue_sort_key(row: dict[str, str], *, pass_by_slug: dict[str, int], failed: set[str]) -> tuple[Any, ...]:
    queue_id = row.get("queue_id", "")
    solver = row.get("solver", "")
    slug = row.get("slug", "")
    field_case = row.get("field_case", "")
    color = row.get("color", "")
    wavelength = str(row.get("wavelength_nm", ""))
    partial_credit = 0 if pass_by_slug.get(slug, 0) > 0 else 1
    failed_first = 0 if queue_id in failed else 1
    solver_rank = 0 if solver == "field" else 1
    return (
        partial_credit,
        failed_first,
        solver_rank,
        FIELD_CASE_PRIORITY.get(field_case, 8),
        WAVELENGTH_PRIORITY.get(wavelength, 8),
        COLOR_PRIORITY.get(color, 8),
        finite_float(row.get("estimated_hours"), 9999.0),
        slug,
        queue_id,
    )


def merged_status(package_dir: Path) -> tuple[set[str], set[str], dict[str, int]]:
    merged = read_csv(package_dir / "camera_e2e_quantitative_merged_summary.csv")
    completed: set[str] = set()
    failed: set[str] = set()
    pass_by_slug: dict[str, int] = defaultdict(int)
    for row in merged:
        queue_id = row.get("queue_id", "")
        gate = str(row.get("solver_gate", "")).upper()
        if not queue_id:
            continue
        completed.add(queue_id)
        if gate == "FAIL":
            failed.add(queue_id)
        if gate == "PASS":
            pass_by_slug[row.get("slug", "")] += 1
    return completed, failed, pass_by_slug


def output_artifact_for(row: dict[str, str]) -> str:
    output_dir = ""
    command = row.get("command", "")
    parts = command.split()
    if "--output-dir" in parts:
        index = parts.index("--output-dir")
        if index + 1 < len(parts):
            output_dir = parts[index + 1]
    if not output_dir:
        return ""
    filename = "crosstalk_sweep_report.json" if row.get("solver") == "crosstalk" else "fdtd_field_sweep_report.json"
    return f"{output_dir.rstrip('/')}/{filename}"


def issue_plan_rows(
    issues: list[dict[str, str]],
    sensors_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in issues:
        code = issue.get("issue_code", "")
        if code not in {"cra_input_not_measured", "stack_material_not_measured"}:
            continue
        slug = issue.get("slug", "")
        sensor = sensors_by_slug.get(slug, {})
        if code == "cra_input_not_measured":
            command = "Populate image_sensor_db/camera_module_field_map.csv, then run: python3 validate_camera_module_field_map.py --field-map-csv image_sensor_db/camera_module_field_map.csv"
            artifact = "image_sensor_db/camera_module_field_map.csv"
            notes = "Needs module raytrace, lab calibration, or measured CRA/OCL shift map; TechInsights teardown alone is not enough."
        else:
            command = "Import measured stack geometry and n,k tables, then rebuild: python3 build_camera_e2e_sensor_luts.py --major-only"
            artifact = "measured stack/n,k source tables"
            notes = "Needs measured layer thickness and wavelength-dependent material data."
        rows.append(
            {
                "plan_id": f"{slug}_{code}",
                "priority": ISSUE_PRIORITY.get(code, "P1"),
                "track": "measured_input",
                "runnable": False,
                "blocking_gate": code,
                "slug": slug,
                "code": issue.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "solver": "data_import",
                "queue_id": "",
                "color": "",
                "field_case": "",
                "wavelength_nm": "",
                "target_resolution_px_per_um": "",
                "estimated_hours": "",
                "expected_success_gate": "CRA_INPUT_PASS" if code == "cra_input_not_measured" else "MEASURED_STACK_PASS",
                "command": command,
                "expected_artifact": artifact,
                "why_this_next": issue.get("detail", ""),
                "notes": notes,
            }
        )
    return rows


def coverage_blocker_code(domain: str) -> str:
    if domain == "Pixel / Electrical":
        return "measured_pixel_electrical_calibration_missing"
    if domain == "Readout / RAW":
        return "measured_readout_raw_calibration_missing"
    if domain == "Module Coupling":
        return "measured_module_coupling_calibration_missing"
    return "measured_calibration_missing"


def coverage_closure_command(domain: str, requirement_id: str) -> tuple[str, str, str]:
    if domain == "Pixel / Electrical":
        return (
            "Import measured electrical calibration tables for conversion gain, FWC, dark current, DSNU/PRNU, temporal noise, and charge-collection/TCAD calibration; then rerun: python3 export_camera_e2e_electrical_readout_tables.py --package-dir runs/camera_e2e_sensor_lut_package",
            "measured electrical/noise/charge-collection calibration tables",
            "Needs measured pixel electrical calibration; prior seed values only support research CameraE2E plumbing.",
        )
    if domain == "Readout / RAW":
        return (
            "Import measured readout/RAW calibration tables for analog/digital gain, black level, ADC clipping/quantization, row/column FPN, rolling-shutter timing, readout direction, defect/hot-pixel statistics, and binning/remosaic modes; then rerun: python3 export_camera_e2e_electrical_readout_tables.py --package-dir runs/camera_e2e_sensor_lut_package",
            "measured readout/raw mode calibration tables",
            "Needs measured readout and RAW-mode calibration; current rows are configurable prior seeds.",
        )
    return (
        "Import module raytrace or measured module calibration tables for field CRA, sensor position/tilt/decenter, vignetting/shading, and wavelength-dependent pupil behavior; then rerun: python3 export_camera_e2e_module_coupling.py --package-dir runs/camera_e2e_sensor_lut_package",
        "measured module raytrace/CRA/vignetting/pupil calibration tables",
        "Needs module-level measured or raytraced calibration, not teardown-derived priors.",
    )


def coverage_calibration_plan_rows(
    coverage_rows: list[dict[str, str]],
    sensors_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coverage in coverage_rows:
        domain = coverage.get("domain", "")
        if domain not in COVERAGE_CLOSURE_DOMAINS:
            continue
        if coverage.get("product_gate", "") not in {"FAIL", "MISSING"}:
            continue
        slug = coverage.get("slug", "")
        requirement_id = coverage.get("requirement_id", "")
        if not slug or not requirement_id:
            continue
        sensor = sensors_by_slug.get(slug, {})
        issue_code = coverage_blocker_code(domain)
        command, artifact, notes = coverage_closure_command(domain, requirement_id)
        rows.append(
            {
                "plan_id": f"{slug}_{requirement_id}_product_calibration",
                "priority": ISSUE_PRIORITY.get(issue_code, "P0"),
                "track": "measured_calibration_input",
                "runnable": False,
                "blocking_gate": issue_code,
                "slug": slug,
                "code": coverage.get("code", ""),
                "manufacturer": sensor.get("manufacturer", coverage.get("manufacturer", "")),
                "device_name": sensor.get("device_name", coverage.get("device_name", "")),
                "solver": "data_import",
                "queue_id": "",
                "color": "",
                "field_case": "",
                "wavelength_nm": "",
                "target_resolution_px_per_um": "",
                "estimated_hours": "",
                "expected_success_gate": "PRODUCT_CALIBRATION_PASS",
                "command": command,
                "expected_artifact": artifact,
                "why_this_next": f"{domain} / {coverage.get('requirement', requirement_id)}: {coverage.get('primary_blocker', '')}",
                "notes": f"{notes} requirement_id={requirement_id}; research_status={coverage.get('research_status', '')}; product_gate={coverage.get('product_gate', '')}",
            }
        )
    return rows


def solver_plan_rows(
    package_dir: Path,
    queue_rows: list[dict[str, str]],
    sensors_by_slug: dict[str, dict[str, str]],
    *,
    max_solver_points: int,
    include_failed: bool,
    prefer_slug: str,
) -> list[dict[str, Any]]:
    completed, failed, pass_by_slug = merged_status(package_dir)
    candidates = []
    for row in queue_rows:
        queue_id = row.get("queue_id", "")
        if not queue_id:
            continue
        if queue_id in completed and (queue_id not in failed or not include_failed):
            continue
        if prefer_slug and row.get("slug") != prefer_slug:
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: queue_sort_key(row, pass_by_slug=pass_by_slug, failed=failed))
    output = []
    for row in candidates[:max_solver_points]:
        slug = row.get("slug", "")
        sensor = sensors_by_slug.get(slug, {})
        solver = row.get("solver", "")
        queue_id = row.get("queue_id", "")
        is_retry = queue_id in failed
        why = "retry failed quantitative point" if is_retry else "close missing quantitative coverage point"
        if pass_by_slug.get(slug, 0) > 0:
            why += "; sensor already has PASS anchors, so incremental coverage improves interpolation evidence"
        output.append(
            {
                "plan_id": queue_id,
                "priority": "P0",
                "track": "solver_quantitative",
                "runnable": True,
                "blocking_gate": "field_fdtd_coverage_incomplete" if solver == "field" else "finite_array_crosstalk_not_pass",
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "solver": solver,
                "queue_id": queue_id,
                "color": row.get("color", ""),
                "field_case": row.get("field_case", ""),
                "wavelength_nm": row.get("wavelength_nm", ""),
                "target_resolution_px_per_um": row.get("target_resolution_px_per_um", ""),
                "estimated_hours": row.get("estimated_hours", ""),
                "expected_success_gate": "FDTD_FIELD_PASS" if solver == "field" else "FDTD_CROSSTALK_PASS",
                "command": row.get("command", ""),
                "expected_artifact": output_artifact_for(row),
                "why_this_next": why,
                "notes": "Run through queue runner for logging/resume, then merge and rebuild package.",
            }
        )
    return output


def resource_limited_plan_rows(
    resource_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    sensors_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    queue_by_id = {row.get("queue_id", ""): row for row in queue_rows if row.get("queue_id", "")}
    rows: list[dict[str, Any]] = []
    for resource in resource_rows:
        queue_id = resource.get("queue_id", "")
        queue = queue_by_id.get(queue_id, {})
        slug = resource.get("slug", "") or queue.get("slug", "")
        sensor = sensors_by_slug.get(slug, {})
        command = resource.get("batch_command", "") or queue.get("command", "")
        rows.append(
            {
                "plan_id": f"{queue_id}_batch_resource_limited",
                "priority": "P0",
                "track": "solver_resource_limited_batch",
                "runnable": True,
                "blocking_gate": "finite_array_crosstalk_not_pass",
                "slug": slug,
                "code": resource.get("code", "") or queue.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "solver": resource.get("solver", "") or queue.get("solver", ""),
                "queue_id": queue_id,
                "color": resource.get("color", "") or queue.get("color", ""),
                "field_case": resource.get("field_case", "") or queue.get("field_case", ""),
                "wavelength_nm": resource.get("wavelength_nm", "") or queue.get("wavelength_nm", ""),
                "target_resolution_px_per_um": resource.get("target_resolution_px_per_um", "") or queue.get("target_resolution_px_per_um", ""),
                "estimated_hours": queue.get("estimated_hours", ""),
                "expected_success_gate": "FDTD_CROSSTALK_PASS",
                "command": command,
                "expected_artifact": output_artifact_for(queue),
                "why_this_next": (
                    "finite-array crosstalk is resource-limited locally; run this point on a batch/cluster runner "
                    "before treating optical crosstalk kernels as solver-backed"
                ),
                "notes": (
                    f"estimated_voxels={resource.get('estimated_voxels', '')}; "
                    f"fdtd_domain_factor={resource.get('fdtd_domain_factor', '')}; "
                    "rerun merge_camera_e2e_quantitative_points.py and the package pipeline after completion"
                ),
            }
        )
    return rows


def crosstalk_priority_plan_rows(
    priority_rows: list[dict[str, str]],
    sensors_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for priority in priority_rows:
        slug = priority.get("slug", "")
        command = priority.get("command", "")
        if not slug or not command:
            continue
        sensor = sensors_by_slug.get(slug, {})
        action_type = priority.get("action_type", "")
        is_product_primary = action_type == "product_resolution_crosstalk_primary"
        priority_class = priority.get("priority_class", "")
        priority_label = priority_class.split("_", 1)[0] if priority_class else ("P0" if is_product_primary else "P1")
        track = "solver_crosstalk_product_primary" if is_product_primary else "solver_crosstalk_support_discovery"
        plan_suffix = "product_primary" if is_product_primary else "support_discovery"
        rows.append(
            {
                "plan_id": f"{priority.get('slug', '')}_{priority.get('color_channel', '')}_{priority.get('field_case', '')}_{priority.get('wavelength_nm', '')}_crosstalk_{plan_suffix}",
                "priority_rank": priority.get("priority_rank", ""),
                "priority": priority_label,
                "track": track,
                "runnable": True,
                "blocking_gate": "finite_array_crosstalk_not_pass",
                "slug": slug,
                "code": priority.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "solver": "crosstalk",
                "queue_id": priority.get("queue_id", ""),
                "color": priority.get("color_channel", ""),
                "field_case": priority.get("field_case", ""),
                "wavelength_nm": priority.get("wavelength_nm", ""),
                "target_resolution_px_per_um": priority.get("resolution_px_per_um", ""),
                "estimated_hours": "",
                "expected_success_gate": "FDTD_CROSSTALK_PASS" if is_product_primary else "LOW_RES_SUPPORT_ESTABLISHED",
                "command": command,
                "expected_artifact": priority.get("expected_artifact", ""),
                "why_this_next": priority.get("why_this_next", ""),
                "notes": (
                    f"support-aware crosstalk priority; recommended_neighborhood={priority.get('recommended_neighborhood', '')}; "
                    f"estimated_voxels={priority.get('estimated_voxels', '')}; "
                    f"local_feasibility={priority.get('local_feasibility', '')}; "
                    f"support_evidence_gate={priority.get('support_evidence_gate', '')}; "
                    f"candidate_support_role={priority.get('candidate_support_role', '')}; "
                    f"candidate_priority={priority.get('candidate_priority', '')}"
                ),
            }
        )
    return rows


def batch_rows(plan_rows: list[dict[str, Any]], package_dir: Path, max_batches: int) -> list[dict[str, Any]]:
    runnable = [row for row in plan_rows if str(row.get("runnable")) in {"True", "true", "1"} or row.get("runnable") is True]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runnable:
        groups[(row.get("slug", ""), row.get("solver", ""), row.get("track", ""))].append(row)
    output = []
    def group_sort_key(item: tuple[tuple[str, str, str], list[dict[str, Any]]]) -> tuple[Any, ...]:
        (slug, solver, track), rows = item
        best_priority = min((priority_rank(str(row.get("priority", "P2"))) for row in rows), default=9)
        best_rank = min((finite_float(row.get("priority_rank"), 9999.0) for row in rows), default=9999.0)
        return (best_priority, TRACK_PRIORITY.get(track, 9), best_rank, slug, solver)

    for (slug, solver, track), rows in sorted(groups.items(), key=group_sort_key)[:max_batches]:
        queue_ids = [str(row.get("queue_id", "")) for row in rows if row.get("queue_id")]
        estimated = sum(finite_float(row.get("estimated_hours"), 0.0) for row in rows)
        if track in {"solver_resource_limited_batch", "solver_crosstalk_product_primary", "solver_crosstalk_support_discovery"} and len(rows) == 1:
            command = str(rows[0].get("command", ""))
        elif track == "solver_resource_limited_batch":
            command = "Run each resource-limited command from camera_e2e_closure_plan.csv for this slug/solver; direct commands preserve --max-local-voxels 0."
        elif track in {"solver_crosstalk_product_primary", "solver_crosstalk_support_discovery"}:
            command = "Run each support-aware crosstalk command from camera_e2e_closure_plan.csv for this slug/solver/track."
        else:
            command = (
                "python3 run_camera_e2e_quantitative_queue.py "
                f"--package-dir {package_dir} --queue-ids {','.join(queue_ids)} "
                "--max-points 999 --timeout-s 86400"
            )
        if track == "solver_resource_limited_batch":
            batch_label = "resource_limited"
        elif track == "solver_crosstalk_product_primary":
            batch_label = "product_primary"
        elif track == "solver_crosstalk_support_discovery":
            batch_label = "support_discovery"
        else:
            batch_label = "quantitative"
        if track in {"solver_resource_limited_batch", "solver_crosstalk_product_primary", "solver_crosstalk_support_discovery"}:
            notes = "Direct crosstalk command; rerun support/mesh audits and the package pipeline after completion."
        else:
            notes = "Batch command executes selected point-sized queue items with resume logging."
        output.append(
            {
                "batch_id": f"{slug}_{solver}_{batch_label}_batch",
                "priority": min((row.get("priority", "P2") for row in rows), key=priority_rank) if rows else "P2",
                "track": {
                    "solver_resource_limited_batch": "solver_resource_limited_batch",
                    "solver_crosstalk_product_primary": "solver_crosstalk_product_primary_batch",
                    "solver_crosstalk_support_discovery": "solver_crosstalk_support_discovery_batch",
                }.get(track, "solver_quantitative_batch"),
                "slug": slug,
                "solver": solver,
                "queue_id_count": len(queue_ids),
                "estimated_hours_sum": f"{estimated:.2f}",
                "command": command,
                "queue_ids": ",".join(queue_ids),
                "notes": notes,
            }
        )
    return output


def validate_plan(
    *,
    rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    expected_measured_input_count: int,
    expected_measured_calibration_count: int,
    expected_resource_limited_count: int,
    expected_crosstalk_priority_count: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    plan_ids = [str(row.get("plan_id", "")) for row in rows if str(row.get("plan_id", ""))]
    duplicate_plan_ids = sorted(plan_id for plan_id, count in Counter(plan_ids).items() if count > 1)
    measured_rows = [row for row in rows if row.get("track") == "measured_input"]
    measured_calibration_rows = [row for row in rows if row.get("track") == "measured_calibration_input"]
    resource_rows = [row for row in rows if row.get("track") == "solver_resource_limited_batch"]
    crosstalk_priority_rows = [
        row
        for row in rows
        if row.get("track") in {"solver_crosstalk_product_primary", "solver_crosstalk_support_discovery"}
    ]
    runnable_rows = [row for row in rows if row.get("runnable") is True]
    runnable_missing_commands = [row.get("plan_id", "") for row in runnable_rows if not str(row.get("command", "")).strip()]
    resource_missing_force = [
        row.get("plan_id", "")
        for row in resource_rows
        if "--max-local-voxels 0" not in str(row.get("command", ""))
    ]
    resource_missing_voxels = [
        row.get("plan_id", "")
        for row in resource_rows
        if "estimated_voxels=" not in str(row.get("notes", ""))
    ]

    checks.append(
        check_row(
            "plan_rows_present",
            len(rows) > 0,
            "PASS" if rows else "FAIL",
            {"plan_row_count": len(rows)},
            "Regenerate closure plan after readiness and queue exports.",
        )
    )
    checks.append(
        check_row(
            "plan_ids_unique",
            not duplicate_plan_ids,
            "PASS" if not duplicate_plan_ids else "FAIL",
            {"duplicate_plan_ids": duplicate_plan_ids},
            "Fix duplicate plan_id generation.",
        )
    )
    checks.append(
        check_row(
            "measured_input_blockers_covered",
            len(measured_rows) == expected_measured_input_count,
            "PASS" if len(measured_rows) == expected_measured_input_count else "FAIL",
            {"expected": expected_measured_input_count, "actual": len(measured_rows)},
            "Include every measured CRA and measured stack/n,k blocker in the closure plan.",
        )
    )
    checks.append(
        check_row(
            "measured_calibration_blockers_covered",
            len(measured_calibration_rows) == expected_measured_calibration_count,
            "PASS" if len(measured_calibration_rows) == expected_measured_calibration_count else "FAIL",
            {"expected": expected_measured_calibration_count, "actual": len(measured_calibration_rows)},
            "Include every product-blocked electrical/readout/module calibration requirement from the coverage matrix.",
        )
    )
    checks.append(
        check_row(
            "resource_limited_crosstalk_covered",
            True if expected_crosstalk_priority_count else len(resource_rows) == expected_resource_limited_count,
            "PASS" if (True if expected_crosstalk_priority_count else len(resource_rows) == expected_resource_limited_count) else "FAIL",
            {
                "expected_legacy_resource_limited": expected_resource_limited_count,
                "actual_legacy_resource_limited": len(resource_rows),
                "support_aware_priority_rows": len(crosstalk_priority_rows),
            },
            "Use support-aware crosstalk priority rows when available; otherwise include every resource-limited finite-array crosstalk row.",
        )
    )
    checks.append(
        check_row(
            "support_aware_crosstalk_priority_covered",
            len(crosstalk_priority_rows) == expected_crosstalk_priority_count,
            "PASS" if len(crosstalk_priority_rows) == expected_crosstalk_priority_count else "FAIL",
            {"expected": expected_crosstalk_priority_count, "actual": len(crosstalk_priority_rows)},
            "Include every row from camera_e2e_crosstalk_batch_priority.csv in the closure plan.",
        )
    )
    checks.append(
        check_row(
            "runnable_commands_present",
            not runnable_missing_commands,
            "PASS" if not runnable_missing_commands else "FAIL",
            {"missing_command_plan_ids": runnable_missing_commands},
            "Every runnable solver row needs an executable command.",
        )
    )
    checks.append(
        check_row(
            "resource_limited_commands_force_batch",
            not resource_missing_force,
            "PASS" if not resource_missing_force else "FAIL",
            {"missing_force_batch_plan_ids": resource_missing_force},
            "Resource-limited crosstalk commands must preserve --max-local-voxels 0.",
        )
    )
    checks.append(
        check_row(
            "resource_limited_rows_carry_voxels",
            not resource_missing_voxels,
            "PASS" if not resource_missing_voxels else "FAIL",
            {"missing_voxel_note_plan_ids": resource_missing_voxels},
            "Resource-limited crosstalk rows must carry estimated voxel count for scheduling.",
        )
    )
    checks.append(
        check_row(
            "batches_present_for_runnable_rows",
            bool(batches) if runnable_rows else True,
            "PASS" if (bool(batches) if runnable_rows else True) else "FAIL",
            {"runnable_row_count": len(runnable_rows), "batch_row_count": len(batches)},
            "Generate closure batch rows for runnable solver work.",
        )
    )
    failures = [row for row in checks if str(row.get("status")) == "FAIL"]
    return {
        "schema": "camera_e2e_closure_plan_validation_v1",
        "pass": not failures,
        "status": "CLOSURE_PLAN_READY_PRODUCT_BLOCKED" if not failures else "FAIL",
        "issue_count": len(failures),
        "error_count": len(failures),
        "warning_count": 0,
        "issues": failures,
        "checks": checks,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 200) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    more = f"<p class=\"muted\">Showing {min(limit, len(rows))} of {len(rows)} rows.</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}.bad{color:#ff8b8b}.warn{color:#ffd36e}
"""
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Closure Plan</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Closure Plan</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This plan turns readiness blockers into measured-data tasks and runnable solver batches.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("validation", {}).get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("plan_row_count", 0))}</div><div class="muted">plan rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("runnable_solver_row_count", 0))}</div><div class="muted">runnable solver rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("measured_input_row_count", 0))}</div><div class="muted">measured input blockers</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("measured_calibration_input_row_count", 0))}</div><div class="muted">electrical/readout/module calibration blockers</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("crosstalk_product_primary_solver_row_count", 0))}</div><div class="muted">crosstalk product primary rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("crosstalk_support_discovery_solver_row_count", 0))}</div><div class="muted">crosstalk support discovery rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("resource_limited_solver_row_count", 0))}</div><div class="muted">resource-limited solver rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("estimated_solver_hours", 0))}</div><div class="muted">selected solver hours</div></div>
  </div>
  <h2>Validation Checks</h2>
  {html_table(payload.get("validation", {}).get("issues", []) or [], CHECK_COLUMNS, limit=80) if payload.get("validation", {}).get("issues") else '<p class="muted">All closure plan checks passed.</p>'}
  <h2>Batches</h2>
  {html_table(payload.get("batches", []), BATCH_COLUMNS, limit=80)}
  <h2>Plan Rows</h2>
  {html_table(payload.get("rows", []), PLAN_COLUMNS, limit=200)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_package_links(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_closure_plan_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_closure_plan_csv"] = payload["outputs"]["plan_csv"]
    outputs["camera_e2e_closure_batches_csv"] = payload["outputs"]["batch_csv"]
    outputs["camera_e2e_closure_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_closure_plan_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_closure_plan"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "plan_row_count": payload["plan_row_count"],
        "runnable_solver_row_count": payload["runnable_solver_row_count"],
        "measured_input_row_count": payload["measured_input_row_count"],
        "measured_calibration_input_row_count": payload.get("measured_calibration_input_row_count", 0),
        "resource_limited_solver_row_count": payload.get("resource_limited_solver_row_count", 0),
        "crosstalk_priority_solver_row_count": payload.get("crosstalk_priority_solver_row_count", 0),
        "crosstalk_product_primary_solver_row_count": payload.get("crosstalk_product_primary_solver_row_count", 0),
        "crosstalk_support_discovery_solver_row_count": payload.get("crosstalk_support_discovery_solver_row_count", 0),
        "estimated_solver_hours": payload["estimated_solver_hours"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = read_csv(package_dir / "camera_e2e_sensor_index.csv")
    sensors_by_slug = {row.get("slug", ""): row for row in sensors}
    issues = read_csv(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_issues.csv")
    coverage = read_csv(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    queue = read_csv(package_dir / "camera_e2e_quantitative_point_queue.csv")
    resource_limited = read_csv(package_dir / "camera_e2e_resource_limited_batch_plan.csv")
    crosstalk_priority = read_csv(package_dir / "camera_e2e_crosstalk_batch_priority" / "camera_e2e_crosstalk_batch_priority.csv")
    if args.prefer_slug:
        issues = [issue for issue in issues if issue.get("slug") == args.prefer_slug]
        coverage = [row for row in coverage if row.get("slug") == args.prefer_slug]
        resource_limited = [row for row in resource_limited if row.get("slug") == args.prefer_slug]
        crosstalk_priority = [row for row in crosstalk_priority if row.get("slug") == args.prefer_slug]

    measured_rows = issue_plan_rows(issues, sensors_by_slug)
    measured_calibration_rows = coverage_calibration_plan_rows(coverage, sensors_by_slug)
    crosstalk_priority_rows = crosstalk_priority_plan_rows(crosstalk_priority, sensors_by_slug)
    rows = list(measured_rows)
    rows.extend(measured_calibration_rows)
    if crosstalk_priority_rows:
        rows.extend(crosstalk_priority_rows)
    else:
        rows.extend(resource_limited_plan_rows(resource_limited, queue, sensors_by_slug))
    rows.extend(
        solver_plan_rows(
            package_dir,
            queue,
            sensors_by_slug,
            max_solver_points=args.max_solver_points,
            include_failed=args.include_failed,
            prefer_slug=args.prefer_slug,
        )
    )
    rows.sort(
        key=lambda row: (
            priority_rank(str(row.get("priority", "P2"))),
            TRACK_PRIORITY.get(str(row.get("track", "")), 9),
            str(row.get("slug", "")),
            str(row.get("solver", "")),
            finite_float(row.get("estimated_hours"), 9999.0),
        )
    )
    batches = batch_rows(rows, package_dir, args.max_batches)
    estimated_solver_hours = sum(finite_float(row.get("estimated_hours"), 0.0) for row in rows if row.get("runnable") is True)
    validation = validate_plan(
        rows=rows,
        batches=batches,
        expected_measured_input_count=len(measured_rows),
        expected_measured_calibration_count=len(measured_calibration_rows),
        expected_resource_limited_count=0 if crosstalk_priority_rows else len(resource_limited),
        expected_crosstalk_priority_count=len(crosstalk_priority_rows),
    )

    plan_csv = output_dir / "camera_e2e_closure_plan.csv"
    batch_csv = output_dir / "camera_e2e_closure_batches.csv"
    checks_csv = output_dir / "camera_e2e_closure_checks.csv"
    report_json = output_dir / "camera_e2e_closure_plan.json"
    html_path = output_dir / "index.html"
    write_csv(plan_csv, rows, PLAN_COLUMNS)
    write_csv(batch_csv, batches, BATCH_COLUMNS)
    write_csv(checks_csv, validation["checks"], CHECK_COLUMNS)
    payload = {
        "schema": "camera_e2e_closure_plan_v1",
        "artifact_role": "camera_e2e_product_gate_closure_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "prefer_slug": args.prefer_slug,
        "max_solver_points": args.max_solver_points,
        "plan_row_count": len(rows),
        "runnable_solver_row_count": sum(1 for row in rows if row.get("runnable") is True),
        "measured_input_row_count": sum(1 for row in rows if row.get("track") == "measured_input"),
        "measured_calibration_input_row_count": sum(1 for row in rows if row.get("track") == "measured_calibration_input"),
        "resource_limited_solver_row_count": sum(1 for row in rows if row.get("track") == "solver_resource_limited_batch"),
        "crosstalk_priority_solver_row_count": len(crosstalk_priority_rows),
        "crosstalk_product_primary_solver_row_count": sum(1 for row in rows if row.get("track") == "solver_crosstalk_product_primary"),
        "crosstalk_support_discovery_solver_row_count": sum(1 for row in rows if row.get("track") == "solver_crosstalk_support_discovery"),
        "estimated_solver_hours": round(estimated_solver_hours, 3),
        "track_counts": dict(Counter(row.get("track", "") for row in rows)),
        "priority_counts": dict(Counter(row.get("priority", "") for row in rows)),
        "validation": {key: value for key, value in validation.items() if key != "checks"},
        "rows": rows,
        "batches": batches,
        "outputs": {
            "json": repo_rel(report_json),
            "plan_csv": repo_rel(plan_csv),
            "batch_csv": repo_rel(batch_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(report_json, payload)
    write_html(html_path, payload)
    update_package_links(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-solver-points", type=int, default=24)
    parser.add_argument("--max-batches", type=int, default=12)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--prefer-slug", default="")
    return parser


def main() -> None:
    payload = plan(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "plan_row_count": payload["plan_row_count"],
                "runnable_solver_row_count": payload["runnable_solver_row_count"],
                "measured_input_row_count": payload["measured_input_row_count"],
                "measured_calibration_input_row_count": payload["measured_calibration_input_row_count"],
                "resource_limited_solver_row_count": payload["resource_limited_solver_row_count"],
                "crosstalk_priority_solver_row_count": payload["crosstalk_priority_solver_row_count"],
                "crosstalk_product_primary_solver_row_count": payload["crosstalk_product_primary_solver_row_count"],
                "crosstalk_support_discovery_solver_row_count": payload["crosstalk_support_discovery_solver_row_count"],
                "estimated_solver_hours": payload["estimated_solver_hours"],
                "track_counts": payload["track_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
