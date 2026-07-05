#!/usr/bin/env python3
"""Export runnable field/QE execution scripts for CameraE2E LUT closure.

The quantitative point queue already contains exact Meep field commands. This
exporter turns that queue into an integration-facing execution pack so the next
solver work is explicit: center spectral/color anchors first, then green CRA
field anchors, then the remaining full RGB x wavelength x field closure.

It does not run solvers and does not promote any row to product accuracy.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_field_execution_pack"
DEFAULT_FIELD_TIMEOUT_S = 7200.0

JOB_COLUMNS = [
    "job_index",
    "execution_group",
    "priority_rank",
    "priority_class",
    "action_type",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "queue_id",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "target_resolution_px_per_um",
    "estimated_hours",
    "estimated_volume_factor",
    "fdtd_domain_factor",
    "local_feasibility",
    "current_solver_gate",
    "current_actual_resolution_px_per_um",
    "mesh_confidence_class",
    "field_pass_points",
    "field_required_points",
    "expected_impact",
    "product_use_gate",
    "expected_artifact",
    "command",
]

SCRIPT_COLUMNS = [
    "script_id",
    "role",
    "path",
    "job_count",
    "execution_policy",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in indexed:
            indexed[value] = row
    return indexed


def command_with_timeout(command: str, timeout_s: float) -> str:
    parts = shlex.split(command)
    if not parts or "--timeout-s" in parts:
        return command
    return shlex.join([*parts, "--timeout-s", str(float(timeout_s))])


def expected_artifact_for_command(command: str) -> str:
    parts = shlex.split(command)
    if "--output-dir" not in parts:
        return ""
    index = parts.index("--output-dir")
    if index + 1 >= len(parts):
        return ""
    return f"{parts[index + 1].rstrip('/')}/fdtd_field_sweep_report.json"


def local_feasibility(row: dict[str, str]) -> str:
    hours = float_value(row.get("estimated_hours"), 999.0)
    if hours <= 0.5:
        return "LOCAL_LONG_SINGLE_POINT"
    if hours <= 1.0:
        return "LOCAL_LONG_OR_BATCH"
    return "BATCH_RECOMMENDED"


def current_by_queue_id(merged_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in merged_rows:
        if row.get("solver") != "field":
            continue
        queue_id = row.get("queue_id", "")
        if queue_id:
            output[queue_id] = row
    return output


def classify_job(row: dict[str, str], current: dict[str, str]) -> tuple[str, str, str, str]:
    gate = str(current.get("solver_gate", "")).upper()
    field_case = row.get("field_case", "")
    color = row.get("color", "")
    wavelength = str(row.get("wavelength_nm", ""))
    if gate == "FAIL":
        return (
            "P0_RERUN_FAILED_FIELD_POINT",
            "rerun_failed_field_point",
            "failed_or_stale_field_rerun",
            "Rerun a failed or stale quantitative field point with the current target resolution and stack inputs.",
        )
    if field_case == "center" and wavelength == "550":
        return (
            "P1_CENTER_550_COLOR_ANCHOR",
            "center_550_color_anchor",
            "center_spectral_color_anchor",
            "Adds a center 550 nm channel anchor for color response and QE normalization.",
        )
    if field_case == "center":
        return (
            "P2_CENTER_SPECTRAL_ANCHOR",
            "center_spectral_anchor",
            "center_spectral_color_anchor",
            "Completes center spectral response for RGB/mono color sensitivity curves.",
        )
    if color == "green" and wavelength == "550":
        return (
            "P3_GREEN_CRA_FIELD_ANCHOR",
            "green_550_cra_field_anchor",
            "green_cra_field_anchor",
            "Adds the highest-value CRA/shading field anchor before full RGB field closure.",
        )
    return (
        "P4_FULL_FIELD_COLOR_COMPLETION",
        "full_field_color_completion",
        "full_field_color_completion",
        "Fills the remaining RGB x wavelength x field grid for quantitative CameraE2E coverage.",
    )


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    priority_order = {
        "P0_RERUN_FAILED_FIELD_POINT": 0,
        "P1_CENTER_550_COLOR_ANCHOR": 1,
        "P2_CENTER_SPECTRAL_ANCHOR": 2,
        "P3_GREEN_CRA_FIELD_ANCHOR": 3,
        "P4_FULL_FIELD_COLOR_COMPLETION": 4,
    }
    sensor_order = {
        "MEDIUM_RESEARCH_FIELD_TREND": 0,
        "LOW_RESEARCH_ANCHOR": 1,
        "LOW_RESEARCH_WITH_FAILED_POINT": 2,
        "STRUCTURAL_PRIOR_ONLY": 3,
    }
    field_order = {
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
    color_order = {"green": 0, "red": 1, "blue": 2, "clear": 3}
    wavelength_order = {"550": 0, "450": 1, "620": 2}
    return (
        priority_order.get(str(row.get("priority_class")), 9),
        sensor_order.get(str(row.get("mesh_confidence_class")), 9),
        float_value(row.get("estimated_hours"), 999.0),
        field_order.get(str(row.get("field_case")), 9),
        color_order.get(str(row.get("color_channel")), 9),
        wavelength_order.get(str(row.get("wavelength_nm")), 9),
        row.get("slug", ""),
    )


def build_job_rows(package_dir: Path, *, field_timeout_s: float) -> list[dict[str, Any]]:
    queue_rows = [row for row in read_csv_rows(package_dir / "camera_e2e_quantitative_point_queue.csv") if row.get("solver") == "field"]
    merged_by_queue = current_by_queue_id(read_csv_rows(package_dir / "camera_e2e_quantitative_merged_summary.csv"))
    sensor_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_sensor_index.csv"), "slug")
    mesh_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv"), "slug")

    rows: list[dict[str, Any]] = []
    for queue in queue_rows:
        queue_id = queue.get("queue_id", "")
        current = merged_by_queue.get(queue_id, {})
        if str(current.get("solver_gate", "")).upper() == "PASS":
            continue
        priority_class, action_type, execution_group, impact = classify_job(queue, current)
        sensor = sensor_by_slug.get(queue.get("slug", ""), {})
        mesh = mesh_by_slug.get(queue.get("slug", ""), {})
        command = command_with_timeout(queue.get("command", ""), field_timeout_s)
        rows.append(
            {
                "job_index": 0,
                "execution_group": execution_group,
                "priority_rank": 0,
                "priority_class": priority_class,
                "action_type": action_type,
                "slug": queue.get("slug", ""),
                "code": queue.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "queue_id": queue_id,
                "color_channel": queue.get("color", ""),
                "wavelength_nm": queue.get("wavelength_nm", ""),
                "field_case": queue.get("field_case", ""),
                "target_resolution_px_per_um": queue.get("target_resolution_px_per_um", ""),
                "estimated_hours": queue.get("estimated_hours", ""),
                "estimated_volume_factor": queue.get("estimated_volume_factor", ""),
                "fdtd_domain_factor": queue.get("fdtd_domain_factor", ""),
                "local_feasibility": local_feasibility(queue),
                "current_solver_gate": current.get("solver_gate", "MISSING"),
                "current_actual_resolution_px_per_um": current.get("actual_resolution_px_per_um", ""),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
                "field_pass_points": mesh.get("field_pass_points", ""),
                "field_required_points": mesh.get("field_required_points", ""),
                "expected_impact": impact,
                "product_use_gate": "FAIL",
                "expected_artifact": expected_artifact_for_command(command),
                "command": command,
            }
        )
    rows = sorted(rows, key=sort_key)
    for index, row in enumerate(rows, start=1):
        row["job_index"] = index
        row["priority_rank"] = index
    return rows


def write_script(path: Path, rows: list[dict[str, Any]], *, title: str, policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# {title}",
        f"# {policy}",
        f"cd {ROOT}",
        "",
    ]
    for row in rows:
        lines.append(
            "echo "
            + repr(
                f"[{row.get('priority_rank')}] {row.get('slug')} {row.get('color_channel')} "
                f"{row.get('field_case')} {row.get('wavelength_nm')}nm"
            )
        )
        lines.append(str(row.get("command", "")))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def write_refresh_script(path: Path, package_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        f"python3 merge_camera_e2e_quantitative_points.py --package-dir {package_dir}",
        f"python3 build_camera_e2e_sensor_luts.py --major-only --output-dir {package_dir}",
        f"python3 run_camera_e2e_package_pipeline.py --include-failed --skip-rebuild",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


def script_rows_for(output_dir: Path, groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "script_id": "center_spectral_color_anchor",
            "role": "center spectral/color field anchors",
            "path": repo_rel(output_dir / "run_center_spectral_color_anchors.sh"),
            "job_count": len(groups["center_spectral_color_anchor"]),
            "execution_policy": "Run first to improve per-sensor color response and center QE anchors.",
        },
        {
            "script_id": "green_cra_field_anchor",
            "role": "green 550 nm CRA field anchors",
            "path": repo_rel(output_dir / "run_green_cra_field_anchors.sh"),
            "job_count": len(groups["green_cra_field_anchor"]),
            "execution_policy": "Run after center anchors to improve CRA/shading field-map confidence.",
        },
        {
            "script_id": "failed_or_stale_field_rerun",
            "role": "failed or stale field-point reruns",
            "path": repo_rel(output_dir / "run_failed_or_stale_field_reruns.sh"),
            "job_count": len(groups["failed_or_stale_field_rerun"]),
            "execution_policy": "Run selectively; these are already attempted points that need current-resolution rerun or inspection.",
        },
        {
            "script_id": "all_field_quantitative_remaining",
            "role": "all remaining quantitative field points",
            "path": repo_rel(output_dir / "run_all_field_quantitative_remaining.sh"),
            "job_count": sum(len(rows) for rows in groups.values()),
            "execution_policy": "Long batch script for full field/QE closure; use batch/HPC or overnight local scheduling.",
        },
        {
            "script_id": "refresh_after_field_jobs",
            "role": "package refresh after field solver jobs",
            "path": repo_rel(output_dir / "refresh_after_field_jobs.sh"),
            "job_count": 0,
            "execution_policy": "Run after selected field jobs complete.",
        },
    ]


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], jobs: list[dict[str, Any]], scripts: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}.muted{color:#9db6c8}.warn{color:#ffd36e}.fail{color:#ff8b8b}.pass{color:#7dff9c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#d8fbff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Field Execution Pack</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Field Execution Pack</h1>
<p class="muted">Generated <code>{html_cell(payload.get("generated_at", ""))}</code>. These scripts schedule quantitative Meep field/QE points; product use remains blocked until all selected points pass mesh/convergence and measured stack/material gates.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">status</div></div>
<div class="card"><div class="metric">{payload.get("job_count", 0)}</div><div class="muted">remaining/stale field jobs</div></div>
<div class="card"><div class="metric">{payload.get("center_anchor_job_count", 0)}</div><div class="muted">center spectral/color anchors</div></div>
<div class="card"><div class="metric">{payload.get("green_cra_anchor_job_count", 0)}</div><div class="muted">green CRA anchors</div></div>
<div class="card"><div class="metric warn">{payload.get("failed_or_stale_job_count", 0)}</div><div class="muted">failed/stale reruns</div></div>
</div>
<p>Priority classes: <code>{html_cell(payload.get("priority_class_counts", {}))}</code></p>
<p>Local feasibility: <code>{html_cell(payload.get("local_feasibility_counts", {}))}</code></p>
<h2>Scripts</h2>
{html_table(scripts, SCRIPT_COLUMNS)}
<h2>Jobs</h2>
{html_table(jobs, JOB_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def export_pack(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    jobs = build_job_rows(package_dir, field_timeout_s=args.field_timeout_s)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        groups[str(row.get("execution_group", ""))].append(row)

    write_script(
        output_dir / "run_center_spectral_color_anchors.sh",
        groups["center_spectral_color_anchor"],
        title="CameraE2E center spectral/color field anchors",
        policy="Quantitative Meep field runs; each point can take tens of minutes locally.",
    )
    write_script(
        output_dir / "run_green_cra_field_anchors.sh",
        groups["green_cra_field_anchor"],
        title="CameraE2E green 550 nm CRA field anchors",
        policy="Run after center anchors to improve CRA/shading response maps.",
    )
    write_script(
        output_dir / "run_failed_or_stale_field_reruns.sh",
        groups["failed_or_stale_field_rerun"],
        title="CameraE2E failed/stale field reruns",
        policy="Inspect or rerun failed quantitative field points with the current package.",
    )
    write_script(
        output_dir / "run_all_field_quantitative_remaining.sh",
        jobs,
        title="CameraE2E all remaining quantitative field points",
        policy="Long-running closure script; use controlled batch scheduling.",
    )
    write_refresh_script(output_dir / "refresh_after_field_jobs.sh", package_dir)
    scripts = script_rows_for(output_dir, groups)

    validation_pass = len(jobs) > 0
    validation = {
        "schema": "camera_e2e_field_execution_pack_validation_v1",
        "pass": validation_pass,
        "status": "FIELD_EXECUTION_PACK_READY_PRODUCT_BLOCKED" if validation_pass else "FIELD_EXECUTION_PACK_EMPTY",
        "issue_count": 0 if validation_pass else 1,
        "error_count": 0 if validation_pass else 1,
        "warning_count": 0,
        "issues": []
        if validation_pass
        else [
            {
                "level": "error",
                "code": "no_remaining_field_jobs",
                "message": "No remaining or stale field jobs were found in the quantitative queue.",
            }
        ],
    }
    payload = {
        "schema": "camera_e2e_field_execution_pack_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "field_timeout_s": args.field_timeout_s,
        "job_count": len(jobs),
        "center_anchor_job_count": len(groups["center_spectral_color_anchor"]),
        "green_cra_anchor_job_count": len(groups["green_cra_field_anchor"]),
        "failed_or_stale_job_count": len(groups["failed_or_stale_field_rerun"]),
        "full_field_completion_job_count": len(groups["full_field_color_completion"]),
        "priority_class_counts": dict(Counter(row.get("priority_class", "") for row in jobs)),
        "local_feasibility_counts": dict(Counter(row.get("local_feasibility", "") for row in jobs)),
        "product_use_gate": "FAIL",
        "validation": validation,
        "outputs": {
            "json": repo_rel(output_dir / "camera_e2e_field_execution_pack.json"),
            "jobs_csv": repo_rel(output_dir / "camera_e2e_field_execution_jobs.csv"),
            "scripts_csv": repo_rel(output_dir / "camera_e2e_field_execution_scripts.csv"),
            "html": repo_rel(output_dir / "index.html"),
            "center_spectral_anchor_script": repo_rel(output_dir / "run_center_spectral_color_anchors.sh"),
            "green_cra_anchor_script": repo_rel(output_dir / "run_green_cra_field_anchors.sh"),
            "failed_or_stale_rerun_script": repo_rel(output_dir / "run_failed_or_stale_field_reruns.sh"),
            "all_field_quantitative_script": repo_rel(output_dir / "run_all_field_quantitative_remaining.sh"),
            "refresh_script": repo_rel(output_dir / "refresh_after_field_jobs.sh"),
        },
    }
    write_csv(output_dir / "camera_e2e_field_execution_jobs.csv", jobs, JOB_COLUMNS)
    write_csv(output_dir / "camera_e2e_field_execution_scripts.csv", scripts, SCRIPT_COLUMNS)
    write_json(output_dir / "camera_e2e_field_execution_pack.json", payload)
    write_html(output_dir / "index.html", payload, jobs, scripts)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-timeout-s", type=float, default=DEFAULT_FIELD_TIMEOUT_S)
    payload = export_pack(parser.parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
