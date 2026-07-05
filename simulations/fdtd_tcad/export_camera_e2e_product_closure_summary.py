#!/usr/bin/env python3
"""Export a sensor-level CameraE2E product-closure summary.

The closure plan is intentionally detailed. This exporter creates the compact
view a CameraE2E or sensor-design reviewer needs first: per sensor, what can be
used now, what remains product-blocked, and which measured-data or solver
action should happen next.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_product_closure_summary"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "product_closure_status",
    "camera_e2e_use_scope",
    "camera_e2e_allowed_use",
    "product_ready",
    "trust_class",
    "mesh_confidence_class",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "coverage_requirement_count",
    "coverage_product_fail_count",
    "coverage_product_missing_count",
    "coverage_product_na_count",
    "measured_input_blocker_count",
    "measured_calibration_blocker_count",
    "field_solver_row_count",
    "crosstalk_product_primary_row_count",
    "crosstalk_support_discovery_row_count",
    "crosstalk_product_primary_hpc_job_count",
    "crosstalk_support_local_candidate_count",
    "crosstalk_support_batch_or_reformulation_count",
    "first_action_class",
    "first_action_type",
    "first_action_source",
    "first_action_local_feasibility",
    "first_action_command_or_input",
    "first_action_expected_artifact",
    "first_action_why",
    "primary_blockers",
]

DOMAIN_COLUMNS = [
    "slug",
    "domain",
    "trust_class",
    "camera_e2e_allowed_use",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "requirement_count",
    "product_gate_counts",
    "primary_blockers",
    "recommended_next_action",
]

CHECK_COLUMNS = [
    "check_id",
    "pass",
    "status",
    "evidence",
    "required_action",
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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def track_count(rows: list[dict[str, str]], track: str) -> int:
    return sum(1 for row in rows if row.get("track") == track)


def solver_count(rows: list[dict[str, str]], track: str, solver: str = "") -> int:
    return sum(1 for row in rows if row.get("track") == track and (not solver or row.get("solver") == solver))


def first_action(actions: list[dict[str, str]], closure_rows: list[dict[str, str]], sensor: dict[str, str]) -> dict[str, str]:
    if actions:
        return actions[0]
    for row in closure_rows:
        if row.get("priority") == "P0":
            return {
                "action_class": "CLOSURE_PLAN",
                "action_type": row.get("track", ""),
                "source": row.get("blocking_gate", ""),
                "local_feasibility": "EXTERNAL_DATA_REQUIRED" if row.get("runnable") == "False" else "RUNNABLE",
                "command_or_input": row.get("command", ""),
                "expected_artifact": row.get("expected_artifact", ""),
                "why": row.get("why_this_next", ""),
            }
    return {
        "action_class": "NONE",
        "action_type": "",
        "source": "",
        "local_feasibility": "",
        "command_or_input": "",
        "expected_artifact": "",
        "why": sensor.get("required_before_product_use", ""),
    }


def closure_status(sensor: dict[str, str]) -> str:
    if boolish(sensor.get("product_ready")):
        return "PRODUCT_READY"
    scope = sensor.get("camera_e2e_use_scope", "")
    if scope == "CAMERA_E2E_RESEARCH_TREND_ONLY":
        return "RESEARCH_TREND_LOADABLE_PRODUCT_BLOCKED"
    if scope == "CAMERA_E2E_SINGLE_ANCHOR_OR_SMOKE_ONLY":
        return "SINGLE_ANCHOR_LOADABLE_PRODUCT_BLOCKED"
    if "CFA_UNKNOWN" in scope:
        return "SCHEMA_PRIOR_CFA_UNKNOWN_PRODUCT_BLOCKED"
    return "SCHEMA_OR_PRIOR_LOADABLE_PRODUCT_BLOCKED"


def build_sensor_rows(package_dir: Path) -> list[dict[str, Any]]:
    use_scope_rows = read_csv_rows(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_sensor.csv")
    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_summary.csv")
    closure_rows = read_csv_rows(package_dir / "camera_e2e_closure_plan" / "camera_e2e_closure_plan.csv")
    action_rows = read_csv_rows(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_next_actions.csv")
    execution_rows = read_csv_rows(package_dir / "camera_e2e_crosstalk_execution_pack" / "camera_e2e_crosstalk_execution_jobs.csv")

    coverage_by_slug = index_by(coverage_rows, "slug")
    closure_by_slug = group_rows(closure_rows, "slug")
    actions_by_slug = group_rows(action_rows, "slug")
    execution_by_slug = group_rows(execution_rows, "slug")

    rows: list[dict[str, Any]] = []
    for sensor in use_scope_rows:
        slug = sensor.get("slug", "")
        coverage = coverage_by_slug.get(slug, {})
        closure = closure_by_slug.get(slug, [])
        actions = actions_by_slug.get(slug, [])
        execution = execution_by_slug.get(slug, [])
        first = first_action(actions, closure, sensor)
        execution_group_counts = Counter(row.get("execution_group", "") for row in execution)
        rows.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "product_closure_status": closure_status(sensor),
                "camera_e2e_use_scope": sensor.get("camera_e2e_use_scope", ""),
                "camera_e2e_allowed_use": sensor.get("camera_e2e_allowed_use", ""),
                "product_ready": sensor.get("product_ready", ""),
                "trust_class": sensor.get("trust_class", ""),
                "mesh_confidence_class": sensor.get("mesh_confidence_class", ""),
                "field_mesh_pass_points": sensor.get("field_mesh_pass_points", ""),
                "field_mesh_required_points": sensor.get("field_mesh_required_points", ""),
                "crosstalk_mesh_pass_points": sensor.get("crosstalk_mesh_pass_points", ""),
                "crosstalk_mesh_required_points": sensor.get("crosstalk_mesh_required_points", ""),
                "coverage_requirement_count": coverage.get("requirement_count", ""),
                "coverage_product_fail_count": coverage.get("product_fail_count", ""),
                "coverage_product_missing_count": coverage.get("product_missing_count", ""),
                "coverage_product_na_count": coverage.get("product_na_count", ""),
                "measured_input_blocker_count": track_count(closure, "measured_input"),
                "measured_calibration_blocker_count": track_count(closure, "measured_calibration_input"),
                "field_solver_row_count": solver_count(closure, "solver_quantitative", "field"),
                "crosstalk_product_primary_row_count": track_count(closure, "solver_crosstalk_product_primary"),
                "crosstalk_support_discovery_row_count": track_count(closure, "solver_crosstalk_support_discovery"),
                "crosstalk_product_primary_hpc_job_count": execution_group_counts.get("product_primary_hpc", 0),
                "crosstalk_support_local_candidate_count": execution_group_counts.get("support_discovery_local_candidate", 0),
                "crosstalk_support_batch_or_reformulation_count": execution_group_counts.get("support_discovery_batch_or_reformulation", 0),
                "first_action_class": first.get("action_class", ""),
                "first_action_type": first.get("action_type", ""),
                "first_action_source": first.get("source", ""),
                "first_action_local_feasibility": first.get("local_feasibility", ""),
                "first_action_command_or_input": first.get("command_or_input", ""),
                "first_action_expected_artifact": first.get("expected_artifact", ""),
                "first_action_why": first.get("why", ""),
                "primary_blockers": sensor.get("primary_blockers", coverage.get("primary_blockers", "")),
            }
        )
    return rows


def build_domain_rows(package_dir: Path) -> list[dict[str, Any]]:
    trust_domain_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv")
    return [{column: row.get(column, "") for column in DOMAIN_COLUMNS} for row in trust_domain_rows]


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def validate_summary(sensor_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [
        check_row(
            "sensors_present",
            bool(sensor_rows),
            "PASS" if sensor_rows else "FAIL",
            {"sensor_count": len(sensor_rows)},
            "Generate use-scope and closure inputs before product closure summary.",
        ),
        check_row(
            "product_gates_blocked",
            all(not boolish(row.get("product_ready")) for row in sensor_rows),
            "PASS",
            {"product_ready_count": sum(1 for row in sensor_rows if boolish(row.get("product_ready")))},
            "Keep product gates closed until measured inputs and solver convergence pass.",
        ),
        check_row(
            "closure_actions_present",
            all(str(row.get("first_action_class", "")) not in {"", "NONE"} for row in sensor_rows),
            "PASS" if all(str(row.get("first_action_class", "")) not in {"", "NONE"} for row in sensor_rows) else "FAIL",
            {"missing_action_slugs": [row.get("slug") for row in sensor_rows if str(row.get("first_action_class", "")) in {"", "NONE"}]},
            "Every sensor should expose at least one next product-closure action.",
        ),
        check_row(
            "domain_rows_present",
            bool(domain_rows),
            "PASS" if domain_rows else "FAIL",
            {"domain_row_count": len(domain_rows)},
            "Generate LUT trust assessment domain rows.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    return {
        "schema": "camera_e2e_product_closure_summary_validation_v1",
        "pass": error_count == 0,
        "status": "PRODUCT_CLOSURE_SUMMARY_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL",
        "issue_count": error_count,
        "error_count": error_count,
        "warning_count": 0,
        "issues": [row for row in checks if not boolish(row.get("pass"))],
        "checks": checks,
    }


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan='{len(columns)}'>... {len(rows) - limit} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}.muted{color:#9db6c8}.warn{color:#ffd36e}.ok{color:#7dff9c}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#d8fbff}
"""
    validation = payload.get("validation", {})
    status_class = "ok" if validation.get("pass") else "warn"
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Product Closure Summary</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Product Closure Summary</h1>
<p class="muted">Generated <code>{html.escape(str(payload.get("generated_at", "")))}</code>. This is a product-readiness worklist summary, not an accuracy certificate.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html.escape(str(validation.get("status", "")))}</div><div class="muted">status</div></div>
<div class="card"><div class="metric">{payload.get("sensor_count", 0)}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric warn">{payload.get("product_ready_count", 0)}</div><div class="muted">product-ready sensors</div></div>
<div class="card"><div class="metric">{payload.get("domain_row_count", 0)}</div><div class="muted">domain rows</div></div>
</div>
<h2>Sensor Closure Rows</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Domain Trust Rows</h2>{html_table(domain_rows, DOMAIN_COLUMNS)}
</main></body></html>
"""
    path.write_text(html_text, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_product_closure_summary_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_product_closure_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_product_closure_by_domain_csv"] = payload["outputs"]["domain_csv"]
    outputs["camera_e2e_product_closure_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_product_closure_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_product_closure_summary"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "domain_row_count": payload["domain_row_count"],
        "product_ready_count": payload["product_ready_count"],
        "product_closure_status_counts": payload.get("product_closure_status_counts", {}),
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_summary(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_rows = build_sensor_rows(package_dir)
    domain_rows = build_domain_rows(package_dir)
    validation = validate_summary(sensor_rows, domain_rows)

    sensor_csv = output_dir / "camera_e2e_product_closure_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_product_closure_by_domain.csv"
    checks_csv = output_dir / "camera_e2e_product_closure_checks.csv"
    json_path = output_dir / "camera_e2e_product_closure_summary.json"
    html_path = output_dir / "index.html"
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_csv(checks_csv, validation.get("checks", []), CHECK_COLUMNS)
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    status_counts = dict(Counter(str(row.get("product_closure_status", "")) for row in sensor_rows))
    payload = {
        "schema": "camera_e2e_product_closure_summary_v1",
        "artifact_role": "camera_e2e_product_readiness_closure_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "domain_row_count": len(domain_rows),
        "product_ready_count": product_ready_count,
        "product_closure_status_counts": status_counts,
        "validation": validation,
        "policy": "Research loading is allowed by upstream gates. Product use remains blocked until every product closure row reaches product-ready evidence.",
        "outputs": {
            "json": repo_rel(json_path),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(json_path, payload)
    write_html(html_path, payload, sensor_rows, domain_rows)
    update_package(package_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    payload = export_summary(parser.parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "domain_row_count": payload["domain_row_count"],
                "product_ready_count": payload["product_ready_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
