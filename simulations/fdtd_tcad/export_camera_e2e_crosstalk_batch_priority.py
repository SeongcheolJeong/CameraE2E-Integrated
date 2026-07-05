#!/usr/bin/env python3
"""Export support-aware finite-array crosstalk batch priorities.

The crosstalk support audit lists many generated candidate commands. This
export collapses them into one actionable row per crosstalk condition:

- if low-resolution support evidence exists, schedule the recommended product
  resolution support size;
- if support evidence is missing, schedule a low-resolution support discovery
  run before any product crosstalk job.

It does not certify product accuracy. Product use still requires the selected
batch job to finish with mesh/convergence PASS and measured stack/material data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_camera_e2e_crosstalk_sweep import estimated_crosstalk_voxels


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_crosstalk_batch_priority"

PRIORITY_COLUMNS = [
    "priority_rank",
    "priority_class",
    "action_type",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "queue_id",
    "requirement_id",
    "color_channel",
    "wavelength_nm",
    "field_case",
    "mode",
    "recommended_neighborhood",
    "resolution_px_per_um",
    "estimated_voxels",
    "estimated_memory_class",
    "local_feasibility",
    "support_evidence_gate",
    "candidate_support_role",
    "candidate_priority",
    "command",
    "expected_artifact",
    "why_this_next",
    "product_use_gate",
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


def crosstalk_requirements_by_slug(package_dir: Path) -> dict[str, dict[str, str]]:
    requirements: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(package_dir / "camera_e2e_required_runs.csv"):
        if row.get("requirement_id") == "fdtd_crosstalk_kernel_convergence":
            requirements[row.get("slug", "")] = row
    return requirements


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 160) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in indexed:
            indexed[value] = row
    return indexed


def candidate_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("slug", ""),
        row.get("color_channel", ""),
        row.get("field_case", ""),
        str(row.get("wavelength_nm", "")),
    )


def expected_artifact_for_command(command: str, filename: str) -> str:
    parts = command.split()
    if "--output-dir" not in parts:
        return ""
    idx = parts.index("--output-dir")
    if idx + 1 >= len(parts):
        return ""
    return f"{parts[idx + 1].rstrip('/')}/{filename}"


def discovery_command(
    row: dict[str, str],
    package_dir: Path,
    *,
    neighborhood_override: int | None = None,
) -> tuple[str, str, int, int, int, str, str]:
    slug = row.get("slug", "")
    color = row.get("color_channel", "")
    field_case = row.get("field_case", "")
    wavelength = row.get("wavelength_nm", "")
    mode = row.get("mode", "")
    neighborhood = neighborhood_override or 15
    resolution = 20
    requirements = crosstalk_requirements_by_slug(package_dir)
    guard = max(1, int_value(requirements.get(slug, {}).get("guard_cells"), int_value(row.get("guard_cells"), 1)))
    stack_config = ROOT / "image_sensor_db" / "generated_stack_configs" / f"{slug}.json"
    voxels = estimated_crosstalk_voxels(stack_config, mode, neighborhood, guard, resolution) if stack_config.exists() else 0
    command = (
        "python3 run_camera_e2e_crosstalk_sweep.py "
        f"--tier trend --slugs {slug} --colors {color} --field-cases {field_case} --wavelengths-nm {wavelength} "
        f"--resolution {resolution} --neighborhood {neighborhood} --guard-cells {guard} "
        f"--output-dir runs/camera_e2e_sensor_lut_package/crosstalk_support_discovery/{slug}/{color}/{field_case}/{wavelength}nm_n{neighborhood}_g{guard}_res{resolution}"
    )
    return command, "crosstalk_sweep_report.json", neighborhood, resolution, voxels, memory_class(voxels), local_feasibility(voxels)


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    priority_order = {
        "P0_PRODUCT_PRIMARY_HPC": 0,
        "P1_SUPPORT_DISCOVERY_CENTER_550": 1,
        "P2_SUPPORT_DISCOVERY_FIELD_OR_COLOR": 2,
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
    color_order = {"green": 0, "clear": 0, "red": 1, "blue": 2}
    return (
        priority_order.get(str(row.get("priority_class", "")), 9),
        field_order.get(str(row.get("field_case", "")), 9),
        0 if str(row.get("wavelength_nm", "")) == "550" else 1,
        color_order.get(str(row.get("color_channel", "")), 9),
        float_value(row.get("estimated_voxels"), 1e18),
        row.get("slug", ""),
    )


def build_priority_rows(package_dir: Path) -> list[dict[str, Any]]:
    candidate_rows = read_csv_rows(package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_product_candidates.csv")
    sensor_rows = read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv")
    sensor_by_slug = index_by(sensor_rows, "slug")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in candidate_rows:
        grouped.setdefault(candidate_key(row), []).append(row)

    priority_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int_value(row.get("neighborhood")))
        primary = next((row for row in rows if row.get("candidate_priority") == "PRIMARY_PRODUCT_CANDIDATE"), {})
        representative = primary or rows[0]
        sensor = sensor_by_slug.get(representative.get("slug", ""), {})
        if primary and primary.get("support_evidence_gate") != "LOW_RES_SUPPORT_STILL_INSUFFICIENT":
            command = primary.get("command", "")
            action_type = "product_resolution_crosstalk_primary"
            priority_class = "P0_PRODUCT_PRIMARY_HPC"
            expected_artifact = expected_artifact_for_command(command, "crosstalk_sweep_report.json")
            why = (
                "Low-resolution support audit found the minimum support that passes truncation. "
                "Run this product-resolution finite-array job on HPC/domain-decomposition before using product crosstalk LUT."
            )
            selected = primary
            recommended_neighborhood = selected.get("recommended_min_neighborhood", "")
            resolution = selected.get("resolution_px_per_um", "")
            estimated_voxels = selected.get("estimated_voxels", "")
            estimated_memory_class = selected.get("estimated_memory_class", "")
            feasibility = selected.get("local_feasibility", "")
        else:
            recommended_from_support = int_value(representative.get("recommended_min_neighborhood"), 0)
            discovery_neighborhood = recommended_from_support if recommended_from_support > 15 else 15
            command, filename, recommended_neighborhood, resolution, estimated_voxels, estimated_memory_class, feasibility = discovery_command(
                representative,
                package_dir,
                neighborhood_override=discovery_neighborhood,
            )
            action_type = "low_resolution_support_discovery"
            priority_class = (
                "P1_SUPPORT_DISCOVERY_CENTER_550"
                if representative.get("field_case") == "center" and str(representative.get("wavelength_nm")) == "550"
                else "P2_SUPPORT_DISCOVERY_FIELD_OR_COLOR"
            )
            expected_artifact = expected_artifact_for_command(command, filename)
            why = (
                "Support-size evidence is missing or still insufficient for this exact color/field/wavelength. "
                f"Run low-resolution {recommended_neighborhood}x{recommended_neighborhood} support discovery before any product-resolution crosstalk job."
            )
            selected = representative
        priority_rows.append(
            {
                "priority_rank": 0,
                "priority_class": priority_class,
                "action_type": action_type,
                "slug": selected.get("slug", ""),
                "code": selected.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "queue_id": selected.get("queue_id", ""),
                "requirement_id": selected.get("requirement_id", ""),
                "color_channel": selected.get("color_channel", ""),
                "wavelength_nm": selected.get("wavelength_nm", ""),
                "field_case": selected.get("field_case", ""),
                "mode": selected.get("mode", ""),
                "recommended_neighborhood": selected.get("recommended_min_neighborhood", "") or recommended_neighborhood,
                "resolution_px_per_um": resolution,
                "estimated_voxels": estimated_voxels,
                "estimated_memory_class": estimated_memory_class,
                "local_feasibility": feasibility,
                "support_evidence_gate": selected.get("support_evidence_gate", "NO_SUPPORT_EVIDENCE"),
                "candidate_support_role": (
                    "NEXT_EXPANDED_SUPPORT_DISCOVERY"
                    if selected.get("support_evidence_gate") == "LOW_RES_SUPPORT_STILL_INSUFFICIENT"
                    else selected.get("candidate_support_role", "UNVERIFIED_SUPPORT_SIZE")
                ),
                "candidate_priority": (
                    "SUPPORT_DISCOVERY_REQUIRED"
                    if action_type == "low_resolution_support_discovery"
                    else selected.get("candidate_priority", "SUPPORT_DISCOVERY_REQUIRED")
                ),
                "command": command,
                "expected_artifact": expected_artifact,
                "why_this_next": why,
                "product_use_gate": "FAIL",
            }
        )
    priority_rows.sort(key=row_sort_key)
    for index, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = index
    return priority_rows


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Crosstalk Batch Priority</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Crosstalk Batch Priority</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Product use remains blocked until selected jobs pass product mesh/convergence gates.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("priority_row_count", 0))}</div><div class="muted">priority rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("product_primary_row_count", 0))}</div><div class="muted">product primary rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("support_discovery_row_count", 0))}</div><div class="muted">support discovery rows</div></div>
</div>
<h2>Priority Rows</h2>{html_table(rows, PRIORITY_COLUMNS)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_crosstalk_batch_priority_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_crosstalk_batch_priority_csv"] = payload["outputs"]["csv"]
    outputs["camera_e2e_crosstalk_batch_priority_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_crosstalk_batch_priority"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "priority_row_count": payload["priority_row_count"],
        "product_primary_row_count": payload["product_primary_row_count"],
        "support_discovery_row_count": payload["support_discovery_row_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_json, package)


def export_priority(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = build_priority_rows(package_dir)
    class_counts = dict(Counter(row.get("priority_class", "") for row in rows))
    action_counts = dict(Counter(row.get("action_type", "") for row in rows))
    issues: list[dict[str, Any]] = []
    if not rows:
        issues.append({"severity": "error", "code": "no_priority_rows"})
    if not any(row.get("priority_class") == "P0_PRODUCT_PRIMARY_HPC" for row in rows):
        issues.append({"severity": "warning", "code": "no_product_primary_candidate", "message": "No support-established product primary crosstalk candidate is available yet."})
    csv_path = output_dir / "camera_e2e_crosstalk_batch_priority.csv"
    json_path = output_dir / "camera_e2e_crosstalk_batch_priority.json"
    html_path = output_dir / "index.html"
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    payload = {
        "schema": "camera_e2e_crosstalk_batch_priority_v1",
        "artifact_role": "support_aware_crosstalk_batch_scheduler",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "priority_row_count": len(rows),
        "product_primary_row_count": sum(1 for row in rows if row.get("action_type") == "product_resolution_crosstalk_primary"),
        "support_discovery_row_count": sum(1 for row in rows if row.get("action_type") == "low_resolution_support_discovery"),
        "priority_class_counts": class_counts,
        "action_type_counts": action_counts,
        "product_use_gate": "FAIL",
        "validation": {
            "schema": "camera_e2e_crosstalk_batch_priority_validation_v1",
            "pass": error_count == 0,
            "status": "CROSSTALK_BATCH_PRIORITY_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL",
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "issues": issues,
        },
        "outputs": {
            "json": repo_rel(json_path),
            "csv": repo_rel(csv_path),
            "html": repo_rel(html_path),
        },
    }
    write_csv(csv_path, rows, PRIORITY_COLUMNS)
    write_json(json_path, payload)
    write_html(html_path, payload, rows)
    update_package(package_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    payload = export_priority(parser.parse_args())
    print(json.dumps({k: payload.get(k) for k in ("schema", "validation", "priority_row_count", "product_primary_row_count", "support_discovery_row_count", "outputs")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
