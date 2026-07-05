#!/usr/bin/env python3
"""Rebuild and validate the CameraE2E runtime package without launching long solvers.

This runner stitches together the existing package builders, exporters, queries,
readiness audit, runtime bundle export, and closure-plan generation. It is meant
to be the repeatable handoff command for camera-system integration:

- research-mode runtime query must be structurally valid and allowed;
- product-mode strict query must fail closed until product gates are truly PASS;
- closure plan must keep measured-data blockers and runnable solver batches
  explicit.

It does not run quantitative Meep/DEVSIM jobs. Use
run_camera_e2e_quantitative_queue.py for the long solver queue, then rerun this
pipeline to refresh all downstream artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_pipeline_validation"
DEFAULT_PREFER_SLUG = "dep_2505_802_smartsens_sc550xs"

STEP_COLUMNS = [
    "step",
    "returncode",
    "expected_returncodes",
    "pass",
    "duration_s",
    "command",
    "stdout_tail",
    "stderr_tail",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def tail_text(value: str, max_chars: int = 4000) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.pass{color:#7dff9c}.fail{color:#ff8b8b}.warn{color:#ffd36e}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    issues_html = (
        html_table(validation.get("issues", []), ["level", "code", "message"])
        if validation.get("issues")
        else '<p class="pass">No structural pipeline issues.</p>'
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Pipeline Validation</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Pipeline Validation</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Long solver jobs are not run by this pipeline.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">pipeline status</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("step_count", 0))}</div><div class="muted">steps</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("runtime_row_count", 0))}</div><div class="muted">runtime rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("kernel_row_count", 0))}</div><div class="muted">kernel rows</div></div>
  </div>
  <h2>Gate Summary</h2>
  <ul>
    <li>Runtime research smoke allowed rows: <code>{html_cell(payload.get("runtime_research_allowed_query_count", payload.get("research_allowed_query_count", "")))}</code></li>
    <li>Product strict probe: <code>{html_cell(payload.get("product_strict_status", ""))}</code></li>
    <li>Product LUT ready: <code>{html_cell(payload.get("product_lut_ready", ""))}</code></li>
    <li>Module coupling field rows: <code>{html_cell(payload.get("module_coupling_field_row_count", ""))}</code></li>
    <li>Material rows: <code>{html_cell(payload.get("material_row_count", ""))}</code></li>
    <li>Material gates: <code>{html_cell(payload.get("material_gate_counts", ""))}</code></li>
    <li>CFA DB status: <code>{html_cell(payload.get("cfa_db_status", ""))}</code></li>
    <li>CFA DB transmission rows: <code>{html_cell(payload.get("cfa_db_transmission_row_count", ""))}</code></li>
    <li>Electrical/noise rows: <code>{html_cell(payload.get("electrical_row_count", ""))}</code></li>
    <li>Readout gain rows: <code>{html_cell(payload.get("readout_row_count", ""))}</code></li>
    <li>Binning/remosaic rows: <code>{html_cell(payload.get("binning_row_count", ""))}</code></li>
    <li>Module coupling gates: <code>{html_cell(payload.get("module_coupling_gate_counts", ""))}</code></li>
    <li>Module coupling research gates: <code>{html_cell(payload.get("module_coupling_research_gate_counts", ""))}</code></li>
    <li>Module coupling product gates: <code>{html_cell(payload.get("module_coupling_product_gate_counts", ""))}</code></li>
    <li>Requirement coverage rows: <code>{html_cell(payload.get("coverage_row_count", ""))}</code></li>
    <li>Requirement coverage status: <code>{html_cell(payload.get("coverage_status", ""))}</code></li>
    <li>Coverage product-ready sensors: <code>{html_cell(payload.get("coverage_product_ready_count", ""))}</code></li>
    <li>Consumer bundle status: <code>{html_cell(payload.get("consumer_bundle_status", ""))}</code></li>
    <li>Consumer bundle sensors: <code>{html_cell(payload.get("consumer_bundle_sensor_count", ""))}</code></li>
    <li>Consumer bundle product-ready sensors: <code>{html_cell(payload.get("consumer_bundle_product_ready_count", ""))}</code></li>
    <li>Uncertainty budget: <code>{html_cell(payload.get("uncertainty_budget_status", ""))}</code>, domain rows <code>{html_cell(payload.get("uncertainty_budget_domain_row_count", ""))}</code></li>
    <li>Response trace: <code>{html_cell(payload.get("response_trace_status", ""))}</code>, rows <code>{html_cell(payload.get("response_trace_row_count", ""))}</code></li>
    <li>Response examples: <code>{html_cell(payload.get("response_example_status", ""))}</code>, rows <code>{html_cell(payload.get("response_example_row_count", ""))}</code></li>
    <li>Method provenance: <code>{html_cell(payload.get("method_provenance_status", ""))}</code>, requirement rows <code>{html_cell(payload.get("method_provenance_row_count", ""))}</code></li>
    <li>Source integrity: <code>{html_cell(payload.get("source_integrity_status", ""))}</code>, requirement rows <code>{html_cell(payload.get("source_integrity_row_count", ""))}</code></li>
    <li>Use-scope summary status: <code>{html_cell(payload.get("use_scope_status", ""))}</code></li>
    <li>Use-scope product-ready sensors: <code>{html_cell(payload.get("use_scope_product_ready_count", ""))}</code></li>
    <li>Use-scope classes: <code>{html_cell(payload.get("use_scope_counts", ""))}</code></li>
    <li>Flat sensor bundle status: <code>{html_cell(payload.get("flat_sensor_bundle_status", ""))}</code></li>
    <li>Flat sensor bundle sensors: <code>{html_cell(payload.get("flat_sensor_bundle_sensor_count", ""))}</code></li>
    <li>Flat sensor embedded rows: <code>{html_cell(payload.get("flat_sensor_bundle_total_embedded_row_count", ""))}</code></li>
    <li>Import contract: <code>{html_cell(payload.get("import_contract_status", ""))}</code>, pointers <code>{html_cell(payload.get("import_contract_pointer_resolved_count", ""))}</code>/<code>{html_cell(payload.get("import_contract_requirement_row_count", ""))}</code></li>
    <li>Canonical payload: <code>{html_cell(payload.get("canonical_payload_status", ""))}</code>, pointers <code>{html_cell(payload.get("canonical_payload_pointer_resolved_count", ""))}</code>/<code>{html_cell(payload.get("canonical_payload_requirement_row_count", ""))}</code></li>
    <li>Flat sensor research query: <code>{html_cell(payload.get("flat_sensor_query_status", ""))}</code>, allowed <code>{html_cell(payload.get("flat_sensor_query_allowed_count", ""))}</code>/<code>{html_cell(payload.get("flat_sensor_query_row_count", ""))}</code></li>
    <li>Flat sensor product query: <code>{html_cell(payload.get("flat_sensor_product_query_status", ""))}</code>, allowed <code>{html_cell(payload.get("flat_sensor_product_query_allowed_count", ""))}</code></li>
    <li>Analysis report: <code>{html_cell(payload.get("analysis_report_status", ""))}</code>, channel rows <code>{html_cell(payload.get("analysis_report_channel_row_count", ""))}</code>, CHECK rows <code>{html_cell(payload.get("analysis_report_check_channel_row_count", ""))}</code></li>
    <li>Sensor deliverable summary: <code>{html_cell(payload.get("sensor_deliverable_summary_status", ""))}</code>, sensors <code>{html_cell(payload.get("sensor_deliverable_summary_sensor_count", ""))}</code>, product-ready <code>{html_cell(payload.get("sensor_deliverable_summary_product_ready_count", ""))}</code></li>
    <li>Consumer research query status: <code>{html_cell(payload.get("consumer_query_status", ""))}</code></li>
    <li>Consumer research query rows: <code>{html_cell(payload.get("consumer_query_row_count", ""))}</code></li>
    <li>Consumer research allowed rows: <code>{html_cell(payload.get("consumer_query_allowed_count", ""))}</code></li>
    <li>Consumer product query status: <code>{html_cell(payload.get("consumer_product_query_status", ""))}</code></li>
    <li>Consumer product allowed rows: <code>{html_cell(payload.get("consumer_product_query_allowed_count", ""))}</code></li>
    <li>Mesh confidence status: <code>{html_cell(payload.get("mesh_confidence_status", ""))}</code></li>
    <li>Mesh field PASS points: <code>{html_cell(payload.get("mesh_confidence_field_pass_total", ""))}/{html_cell(payload.get("mesh_confidence_field_required_total", ""))}</code></li>
    <li>Mesh crosstalk PASS points: <code>{html_cell(payload.get("mesh_confidence_crosstalk_pass_total", ""))}/{html_cell(payload.get("mesh_confidence_crosstalk_required_total", ""))}</code></li>
    <li>Mesh confidence classes: <code>{html_cell(payload.get("mesh_confidence_class_counts", ""))}</code></li>
    <li>Field execution pack: <code>{html_cell(payload.get("field_execution_pack_status", ""))}</code>, jobs <code>{html_cell(payload.get("field_execution_pack_job_count", ""))}</code>, center anchors <code>{html_cell(payload.get("field_execution_pack_center_anchor_count", ""))}</code>, green CRA anchors <code>{html_cell(payload.get("field_execution_pack_green_cra_anchor_count", ""))}</code></li>
    <li>Crosstalk support status: <code>{html_cell(payload.get("crosstalk_support_status", ""))}</code></li>
    <li>Crosstalk support pilot rows: <code>{html_cell(payload.get("crosstalk_support_pilot_row_count", ""))}</code></li>
    <li>Crosstalk support best truncation: <code>{html_cell(payload.get("crosstalk_support_min_truncation_fraction", ""))}</code></li>
    <li>Crosstalk product-ready: <code>{html_cell(payload.get("crosstalk_support_product_ready", ""))}</code></li>
    <li>Crosstalk batch priority status: <code>{html_cell(payload.get("crosstalk_batch_priority_status", ""))}</code></li>
    <li>Crosstalk product primary candidates: <code>{html_cell(payload.get("crosstalk_batch_priority_product_primary_count", ""))}</code></li>
    <li>Crosstalk support discovery jobs: <code>{html_cell(payload.get("crosstalk_batch_priority_support_discovery_count", ""))}</code></li>
    <li>Crosstalk execution pack: <code>{html_cell(payload.get("crosstalk_execution_pack_status", ""))}</code>, product primary scripts <code>{html_cell(payload.get("crosstalk_execution_pack_product_primary_count", ""))}</code></li>
    <li>Handoff status: <code>{html_cell(payload.get("handoff_status", ""))}</code></li>
    <li>Handoff artifacts: <code>{html_cell(payload.get("handoff_artifact_count", ""))}</code></li>
    <li>Handoff loader status: <code>{html_cell(payload.get("handoff_loader_status", ""))}</code></li>
    <li>Handoff loader issues: <code>{html_cell(payload.get("handoff_loader_issue_count", ""))}</code></li>
    <li>Objective acceptance status: <code>{html_cell(payload.get("objective_acceptance_status", ""))}</code></li>
    <li>Objective acceptance sensors: <code>{html_cell(payload.get("objective_acceptance_sensor_count", ""))}</code></li>
    <li>Objective acceptance product-ready sensors: <code>{html_cell(payload.get("objective_acceptance_product_ready_count", ""))}</code></li>
    <li>Objective research allowed rows: <code>{html_cell(payload.get("objective_acceptance_consumer_allowed_count", ""))}</code></li>
    <li>Objective product allowed rows: <code>{html_cell(payload.get("objective_acceptance_product_allowed_count", ""))}</code></li>
    <li>Closure crosstalk priority rows: <code>{html_cell(payload.get("closure_plan_crosstalk_priority_solver_row_count", ""))}</code></li>
    <li>Closure crosstalk product primary rows: <code>{html_cell(payload.get("closure_plan_crosstalk_product_primary_solver_row_count", ""))}</code></li>
    <li>Closure crosstalk support discovery rows: <code>{html_cell(payload.get("closure_plan_crosstalk_support_discovery_solver_row_count", ""))}</code></li>
    <li>Product closure summary: <code>{html_cell(payload.get("product_closure_summary_status", ""))}</code>, sensors <code>{html_cell(payload.get("product_closure_summary_sensor_count", ""))}</code></li>
    <li>Usage policy: <code>{html_cell(payload.get("usage_policy_status", ""))}</code>, strict product rows <code>{html_cell(payload.get("usage_policy_strict_product_filter_row_count", ""))}</code></li>
    <li>Adapter examples: <code>{html_cell(payload.get("adapter_examples_status", ""))}</code>, files <code>{html_cell(payload.get("adapter_examples_file_count", ""))}</code></li>
    <li>Adapter smoke: <code>{html_cell(payload.get("adapter_smoke_status", ""))}</code>, research allowed <code>{html_cell(payload.get("adapter_smoke_total_research_allowed_query_count", ""))}</code>, product allowed <code>{html_cell(payload.get("adapter_smoke_total_product_allowed_query_count", ""))}</code></li>
    <li>Objective trace: <code>{html_cell(payload.get("objective_trace_status", ""))}</code>, requirement summaries <code>{html_cell(payload.get("objective_trace_requirement_summary_row_count", ""))}</code>, rows <code>{html_cell(payload.get("objective_trace_row_count", ""))}</code></li>
    <li>Readiness audit: <code>{html_cell(payload.get("readiness_summary", ""))}</code></li>
  </ul>
  <h2>Issues</h2>
  {issues_html}
  <h2>Steps</h2>
  {html_table(payload.get("steps", []), STEP_COLUMNS)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def run_step(name: str, command: list[str], expected_returncodes: set[int] | None = None) -> dict[str, Any]:
    expected = expected_returncodes or {0}
    start = time.monotonic()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    duration = time.monotonic() - start
    return {
        "step": name,
        "returncode": proc.returncode,
        "expected_returncodes": ",".join(str(item) for item in sorted(expected)),
        "pass": proc.returncode in expected,
        "duration_s": f"{duration:.2f}",
        "command": " ".join(command),
        "stdout_tail": tail_text(proc.stdout),
        "stderr_tail": tail_text(proc.stderr),
    }


def python_command(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def add_field_map_args(command: list[str], field_map_csv: Path | None) -> list[str]:
    if field_map_csv:
        return [*command, "--field-map-csv", str(field_map_csv)]
    return command


def update_package_links(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_pipeline_validation_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_pipeline_validation_csv"] = payload["outputs"]["steps_csv"]
    outputs["camera_e2e_pipeline_validation_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_pipeline_validation"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "status": payload["validation"]["status"],
        "product_lut_ready": payload["product_lut_ready"],
        "runtime_row_count": payload["runtime_row_count"],
        "kernel_row_count": payload["kernel_row_count"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def validate_pipeline(
    *,
    steps: list[dict[str, Any]],
    readiness: dict[str, Any],
    runtime_bundle: dict[str, Any],
    material_tables: dict[str, Any],
    cfa_provenance: dict[str, Any],
    cfa_db_tables: dict[str, Any],
    electrical_readout: dict[str, Any],
    module_coupling: dict[str, Any],
    coverage_matrix: dict[str, Any],
    capability_profile: dict[str, Any],
    trust_assessment: dict[str, Any],
    uncertainty_budget: dict[str, Any],
    response_trace: dict[str, Any],
    response_example: dict[str, Any],
    method_provenance: dict[str, Any],
    source_integrity: dict[str, Any],
    consumer_bundle: dict[str, Any],
    use_scope_summary: dict[str, Any],
    flat_sensor_bundle: dict[str, Any],
    import_contract: dict[str, Any],
    canonical_payload: dict[str, Any],
    flat_sensor_query: dict[str, Any],
    flat_sensor_product_query: dict[str, Any],
    analysis_report: dict[str, Any],
    deliverable_summary: dict[str, Any],
    consumer_query: dict[str, Any],
    consumer_product_query: dict[str, Any],
    mesh_confidence: dict[str, Any],
    field_execution_pack: dict[str, Any],
    crosstalk_support: dict[str, Any],
    crosstalk_batch_priority: dict[str, Any],
    crosstalk_execution_pack: dict[str, Any],
    product_closure_summary: dict[str, Any],
    usage_policy: dict[str, Any],
    adapter_examples: dict[str, Any],
    adapter_smoke: dict[str, Any],
    objective_trace: dict[str, Any],
    handoff: dict[str, Any],
    handoff_loader: dict[str, Any],
    objective_acceptance: dict[str, Any],
    research_query: dict[str, Any],
    product_query: dict[str, Any],
    closure_plan: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    for step in steps:
        if not boolish(step.get("pass")):
            issues.append(
                {
                    "level": "error",
                    "code": "step_failed",
                    "message": f"{step.get('step')} returned {step.get('returncode')} outside {step.get('expected_returncodes')}",
                }
            )
    if readiness.get("schema") != "camera_e2e_lut_readiness_audit_v1":
        issues.append({"level": "error", "code": "readiness_missing", "message": "Readiness audit report is missing or invalid."})
    elif not bool(readiness.get("research_ingest_valid")):
        issues.append({"level": "error", "code": "research_readiness_failed", "message": "Research ingest gate is not valid."})

    bundle_validation = runtime_bundle.get("validation", {})
    if runtime_bundle.get("schema") != "camera_e2e_runtime_bundle_v1" or not bool(bundle_validation.get("pass")):
        issues.append({"level": "error", "code": "runtime_bundle_invalid", "message": "Runtime bundle validation did not pass."})
    if not runtime_bundle.get("runtime_row_count") or not runtime_bundle.get("kernel_row_count"):
        issues.append({"level": "error", "code": "runtime_bundle_empty", "message": "Runtime bundle has no runtime or kernel rows."})

    material_validation = material_tables.get("validation", {})
    if material_tables.get("schema") != "camera_e2e_material_tables_export_v1" or not bool(material_validation.get("pass")):
        issues.append({"level": "error", "code": "material_tables_invalid", "message": "Material n,k table export is missing or structurally invalid."})
    if int(material_tables.get("material_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "material_tables_empty", "message": "Material n,k table has no rows."})

    cfa_validation = cfa_provenance.get("validation", {})
    if cfa_provenance.get("schema") != "camera_e2e_cfa_provenance_audit_v1" or not bool(cfa_validation.get("pass")):
        issues.append({"level": "error", "code": "cfa_provenance_invalid", "message": "CFA provenance audit is missing or structurally invalid."})
    if int(cfa_provenance.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "cfa_provenance_empty", "message": "CFA provenance audit has no sensor rows."})
    if int(cfa_provenance.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "cfa_provenance_unexpected_product_ready", "message": "CFA provenance audit marked product-ready sensors while product gates are blocked."})

    cfa_db_validation = cfa_db_tables.get("validation", {})
    if cfa_db_tables.get("schema") != "camera_e2e_cfa_db_tables_v1" or not bool(cfa_db_validation.get("pass")):
        issues.append({"level": "error", "code": "cfa_db_tables_invalid", "message": "Dedicated CFA DB table export is missing or structurally invalid."})
    if int(cfa_db_tables.get("sensor_count", 0) or 0) <= 0 or int(cfa_db_tables.get("transmission_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "cfa_db_tables_empty", "message": "CFA DB table export has no sensor or transmission rows."})
    if int(cfa_db_tables.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "cfa_db_tables_unexpected_product_ready", "message": "CFA DB table export marked product-ready rows while measured CFA data is incomplete."})

    electrical_validation = electrical_readout.get("validation", {})
    if electrical_readout.get("schema") != "camera_e2e_electrical_readout_tables_export_v1" or not bool(electrical_validation.get("pass")):
        issues.append({"level": "error", "code": "electrical_readout_tables_invalid", "message": "Electrical/readout table export is missing or structurally invalid."})
    if int(electrical_readout.get("electrical_row_count", 0) or 0) <= 0 or int(electrical_readout.get("readout_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "electrical_readout_tables_empty", "message": "Electrical/readout tables have no rows."})

    module_validation = module_coupling.get("validation", {})
    if module_coupling.get("schema") != "camera_e2e_module_coupling_export_v1" or not bool(module_validation.get("pass")):
        issues.append({"level": "error", "code": "module_coupling_invalid", "message": "Module coupling LUT export is missing or structurally invalid."})
    if int(module_coupling.get("field_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "module_coupling_empty", "message": "Module coupling LUT has no field rows."})

    coverage_validation = coverage_matrix.get("validation", {})
    if coverage_matrix.get("schema") != "camera_e2e_coverage_matrix_export_v1" or not bool(coverage_validation.get("pass")):
        issues.append({"level": "error", "code": "coverage_matrix_invalid", "message": "CameraE2E requirement coverage matrix is missing or structurally invalid."})
    if int(coverage_matrix.get("coverage_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "coverage_matrix_empty", "message": "CameraE2E requirement coverage matrix has no rows."})

    capability_validation = capability_profile.get("validation", {})
    if capability_profile.get("schema") != "camera_e2e_capability_profile_v1" or not bool(capability_validation.get("pass")):
        issues.append({"level": "error", "code": "capability_profile_invalid", "message": "CameraE2E capability profile is missing or structurally invalid."})
    if int(capability_profile.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "capability_profile_empty", "message": "CameraE2E capability profile has no sensor rows."})
    if int(capability_profile.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "capability_profile_unexpected_product_ready", "message": "Capability profile marked product-ready sensors while product gates are blocked."})

    trust_validation = trust_assessment.get("validation", {})
    if trust_assessment.get("schema") != "camera_e2e_lut_trust_assessment_v1" or not bool(trust_validation.get("pass")):
        issues.append({"level": "error", "code": "trust_assessment_invalid", "message": "LUT trust assessment is missing or structurally invalid."})
    if int(trust_assessment.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "trust_assessment_empty", "message": "LUT trust assessment has no sensor rows."})
    if int(trust_assessment.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "trust_assessment_unexpected_product_ready", "message": "Trust assessment marked product-ready sensors while product gates are blocked."})

    uncertainty_validation = uncertainty_budget.get("validation", {})
    if uncertainty_budget.get("schema") != "camera_e2e_uncertainty_budget_v1" or not bool(uncertainty_validation.get("pass")):
        issues.append({"level": "error", "code": "uncertainty_budget_invalid", "message": "CameraE2E uncertainty budget is missing or structurally invalid."})
    if int(uncertainty_budget.get("sensor_count", 0) or 0) <= 0 or int(uncertainty_budget.get("domain_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "uncertainty_budget_empty", "message": "CameraE2E uncertainty budget has no sensor or domain rows."})
    if int(uncertainty_budget.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "uncertainty_budget_unexpected_product_ready", "message": "Uncertainty budget marked product-ready rows while product gates are blocked."})

    response_trace_validation = response_trace.get("validation", {})
    if response_trace.get("schema") != "camera_e2e_response_trace_v1" or not bool(response_trace_validation.get("pass")):
        issues.append({"level": "error", "code": "response_trace_invalid", "message": "CameraE2E response trace is missing or structurally invalid."})
    if int(response_trace.get("sensor_count", 0) or 0) <= 0 or int(response_trace.get("trace_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "response_trace_empty", "message": "CameraE2E response trace has no sensor or trace rows."})
    if int(response_trace.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "response_trace_unexpected_product_ready", "message": "Response trace marked product-ready rows while product gates are blocked."})

    response_example_validation = response_example.get("validation", {})
    if response_example.get("schema") != "camera_e2e_response_example_v1" or not bool(response_example_validation.get("pass")):
        issues.append({"level": "error", "code": "response_example_invalid", "message": "CameraE2E response example is missing or structurally invalid."})
    if int(response_example.get("sensor_count", 0) or 0) <= 0 or int(response_example.get("example_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "response_example_empty", "message": "CameraE2E response example has no sensor or example rows."})
    if int(response_example.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "response_example_unexpected_product_ready", "message": "Response example marked product-ready rows while product gates are blocked."})

    method_provenance_validation = method_provenance.get("validation", {})
    if method_provenance.get("schema") != "camera_e2e_method_provenance_v1" or not bool(method_provenance_validation.get("pass")):
        issues.append({"level": "error", "code": "method_provenance_invalid", "message": "CameraE2E method provenance matrix is missing or structurally invalid."})
    if int(method_provenance.get("sensor_count", 0) or 0) <= 0 or int(method_provenance.get("matrix_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "method_provenance_empty", "message": "CameraE2E method provenance matrix has no sensor or requirement rows."})
    if int(method_provenance.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "method_provenance_unexpected_product_ready", "message": "Method provenance marked product-ready sensors while product gates are blocked."})

    source_integrity_validation = source_integrity.get("validation", {})
    if source_integrity.get("schema") != "camera_e2e_lut_source_integrity_v1" or not bool(source_integrity_validation.get("pass")):
        issues.append({"level": "error", "code": "source_integrity_invalid", "message": "CameraE2E LUT source-integrity matrix is missing or structurally invalid."})
    if int(source_integrity.get("sensor_count", 0) or 0) <= 0 or int(source_integrity.get("matrix_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "source_integrity_empty", "message": "CameraE2E LUT source-integrity matrix has no sensor or requirement rows."})
    if int(source_integrity.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "source_integrity_unexpected_product_ready", "message": "Source-integrity matrix marked product-ready sensors while product gates are blocked."})

    consumer_validation = consumer_bundle.get("validation", {})
    if consumer_bundle.get("schema") != "camera_e2e_consumer_bundle_v1" or not bool(consumer_validation.get("pass")):
        issues.append({"level": "error", "code": "consumer_bundle_invalid", "message": "CameraE2E consumer bundle is missing or structurally invalid."})
    if int(consumer_bundle.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "consumer_bundle_empty", "message": "CameraE2E consumer bundle has no sensor manifests."})
    consumer_requirement_count = int(consumer_bundle.get("requirement_count", 0) or 0)
    objective_requirement_count = int(objective_trace.get("requirement_summary_row_count", 0) or objective_trace.get("requirement_count_per_sensor", 0) or 0)
    requirement_load_map = consumer_bundle.get("requirement_load_map", [])
    if consumer_requirement_count <= 0 or not isinstance(requirement_load_map, list) or len(requirement_load_map) != consumer_requirement_count:
        issues.append({"level": "error", "code": "consumer_requirement_load_map_invalid", "message": "Consumer bundle does not expose a complete requirement_load_map."})
    if objective_requirement_count and consumer_requirement_count != objective_requirement_count:
        issues.append({"level": "error", "code": "consumer_requirement_count_mismatch", "message": "Consumer requirement count does not match objective trace requirement count."})
    consumer_join_keys = consumer_bundle.get("join_keys", {})
    if not isinstance(consumer_join_keys, dict) or not consumer_join_keys.get("objective_fulfillment") or not consumer_join_keys.get("source_integrity"):
        issues.append({"level": "error", "code": "consumer_requirement_join_keys_missing", "message": "Consumer bundle is missing objective/source-integrity join keys."})
    consumer_source_tables = consumer_bundle.get("source_tables", {})
    if not isinstance(consumer_source_tables, dict) or not consumer_source_tables.get("source_integrity_matrix") or not consumer_source_tables.get("coverage_matrix"):
        issues.append({"level": "error", "code": "consumer_requirement_source_tables_missing", "message": "Consumer bundle is missing requirement-level source table paths."})

    use_scope_validation = use_scope_summary.get("validation", {})
    if use_scope_summary.get("schema") != "camera_e2e_use_scope_summary_v1" or not bool(use_scope_validation.get("pass")):
        issues.append({"level": "error", "code": "use_scope_summary_invalid", "message": "CameraE2E use-scope summary is missing or structurally invalid."})
    if int(use_scope_summary.get("sensor_count", 0) or 0) <= 0 or int(use_scope_summary.get("domain_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "use_scope_summary_empty", "message": "CameraE2E use-scope summary has no sensor or domain rows."})
    if int(use_scope_summary.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "use_scope_unexpected_product_ready", "message": "Use-scope summary marked product-ready sensors while product gates are blocked."})

    flat_validation = flat_sensor_bundle.get("validation", {})
    if flat_sensor_bundle.get("schema") != "camera_e2e_flat_sensor_bundle_v1" or not bool(flat_validation.get("pass")):
        issues.append({"level": "error", "code": "flat_sensor_bundle_invalid", "message": "Flat per-sensor CameraE2E bundle is missing or structurally invalid."})
    if int(flat_sensor_bundle.get("sensor_count", 0) or 0) <= 0 or int(flat_sensor_bundle.get("total_embedded_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "flat_sensor_bundle_empty", "message": "Flat per-sensor CameraE2E bundle has no sensor models or embedded rows."})
    if int(flat_sensor_bundle.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "flat_sensor_bundle_unexpected_product_ready", "message": "Flat sensor bundle marked product-ready sensors while product gates are blocked."})

    import_contract_validation = import_contract.get("validation", {})
    if import_contract.get("schema") != "camera_e2e_import_contract_v1" or not bool(import_contract_validation.get("pass")):
        issues.append({"level": "error", "code": "import_contract_invalid", "message": "CameraE2E import contract is missing or structurally invalid."})
    if int(import_contract.get("sensor_count", 0) or 0) != int(flat_sensor_bundle.get("sensor_count", 0) or 0):
        issues.append({"level": "error", "code": "import_contract_sensor_count_mismatch", "message": "Import contract sensor count does not match flat sensor bundle."})
    if int(import_contract.get("requirement_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "import_contract_empty", "message": "Import contract has no requirement rows."})
    if int(import_contract.get("pointer_resolved_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        issues.append({"level": "error", "code": "import_contract_pointer_unresolved", "message": "Import contract has unresolved objective pointers."})
    if int(import_contract.get("research_allowed_requirement_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        issues.append({"level": "error", "code": "import_contract_research_not_loadable", "message": "Import contract has research-blocked objective rows."})
    if int(import_contract.get("product_allowed_requirement_count", 0) or 0) != 0 or int(import_contract.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "import_contract_unexpected_product_rows", "message": "Import contract exposed product-ready rows while product gates are blocked."})

    canonical_validation = canonical_payload.get("validation", {})
    if canonical_payload.get("schema") != "camera_e2e_canonical_payload_v1" or not bool(canonical_validation.get("pass")):
        issues.append({"level": "error", "code": "canonical_payload_invalid", "message": "CameraE2E canonical payload is missing or structurally invalid."})
    if int(canonical_payload.get("sensor_count", 0) or 0) != int(flat_sensor_bundle.get("sensor_count", 0) or 0):
        issues.append({"level": "error", "code": "canonical_payload_sensor_count_mismatch", "message": "Canonical payload sensor count does not match flat sensor bundle."})
    if int(canonical_payload.get("requirement_row_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        issues.append({"level": "error", "code": "canonical_payload_requirement_count_mismatch", "message": "Canonical payload requirement count does not match import contract."})
    if int(canonical_payload.get("pointer_resolved_count", 0) or 0) != int(canonical_payload.get("requirement_row_count", 0) or 0):
        issues.append({"level": "error", "code": "canonical_payload_pointer_unresolved", "message": "Canonical payload has unresolved import pointers."})
    if int(canonical_payload.get("product_allowed_requirement_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "canonical_payload_unexpected_product_rows", "message": "Canonical payload exposed product-ready requirement rows while product gates are blocked."})

    flat_query_validation = flat_sensor_query.get("validation", {})
    if flat_sensor_query.get("schema") != "camera_e2e_flat_sensor_query_v1" or not bool(flat_query_validation.get("pass")):
        issues.append({"level": "error", "code": "flat_sensor_query_invalid", "message": "Flat sensor research query is missing or invalid."})
    if int(flat_sensor_query.get("allowed_query_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "flat_sensor_query_blocked", "message": "Flat sensor research query returned no allowed rows."})
    if int(flat_sensor_query.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "flat_sensor_query_unexpected_product_ready", "message": "Flat sensor research query marked product-ready rows while product gates are blocked."})

    flat_product_validation = flat_sensor_product_query.get("validation", {})
    if flat_sensor_product_query.get("schema") != "camera_e2e_flat_sensor_query_v1" or not bool(flat_product_validation.get("pass")):
        issues.append({"level": "error", "code": "flat_sensor_product_query_invalid", "message": "Flat sensor product query probe is missing or invalid."})
    if int(flat_sensor_product_query.get("allowed_query_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "flat_sensor_product_query_unexpectedly_allowed", "message": "Flat sensor product query allowed rows while product gates are blocked."})

    analysis_validation = analysis_report.get("validation", {})
    if analysis_report.get("schema") != "camera_e2e_analysis_report_v1" or not bool(analysis_validation.get("pass")):
        issues.append({"level": "error", "code": "analysis_report_invalid", "message": "CameraE2E analysis report is missing or invalid."})
    if int(analysis_report.get("sensor_count", 0) or 0) <= 0 or int(analysis_report.get("channel_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "analysis_report_empty", "message": "CameraE2E analysis report has no sensor/channel rows."})
    if int(analysis_report.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "analysis_report_unexpected_product_ready", "message": "CameraE2E analysis report marked product-ready sensors while product gates are blocked."})

    deliverable_validation = deliverable_summary.get("validation", {})
    if deliverable_summary.get("schema") != "camera_e2e_sensor_deliverable_summary_v1" or not bool(deliverable_validation.get("pass")):
        issues.append({"level": "error", "code": "sensor_deliverable_summary_invalid", "message": "Sensor deliverable summary is missing or invalid."})
    if int(deliverable_summary.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "sensor_deliverable_summary_empty", "message": "Sensor deliverable summary has no sensor rows."})
    if int(deliverable_summary.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "sensor_deliverable_summary_unexpected_product_ready", "message": "Sensor deliverable summary marked product-ready rows while product gates are blocked."})

    consumer_query_validation = consumer_query.get("validation", {})
    if consumer_query.get("schema") != "camera_e2e_consumer_query_v1" or not bool(consumer_query_validation.get("pass")):
        issues.append({"level": "error", "code": "consumer_query_invalid", "message": "CameraE2E consumer research query is missing or invalid."})
    if int(consumer_query.get("allowed_query_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "consumer_query_blocked", "message": "CameraE2E consumer research query returned no allowed rows."})

    consumer_product_validation = consumer_product_query.get("validation", {})
    if consumer_product_query.get("schema") != "camera_e2e_consumer_query_v1" or not bool(consumer_product_validation.get("pass")):
        issues.append({"level": "error", "code": "consumer_product_query_invalid", "message": "CameraE2E consumer product query probe is missing or invalid."})
    if int(consumer_product_query.get("allowed_query_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "consumer_product_query_unexpectedly_allowed", "message": "CameraE2E consumer product query allowed rows while product gates are blocked."})

    mesh_validation = mesh_confidence.get("validation", {})
    if mesh_confidence.get("schema") != "camera_e2e_mesh_confidence_audit_v1" or not bool(mesh_validation.get("pass")):
        issues.append({"level": "error", "code": "mesh_confidence_invalid", "message": "CameraE2E mesh confidence audit is missing or structurally invalid."})
    if int(mesh_confidence.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "mesh_confidence_empty", "message": "CameraE2E mesh confidence audit validated no sensors."})
    if int(mesh_confidence.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "mesh_confidence_unexpected_product_ready", "message": "Mesh confidence audit marked product-ready sensors while product gates are blocked."})

    field_execution_validation = field_execution_pack.get("validation", {})
    if field_execution_pack.get("schema") != "camera_e2e_field_execution_pack_v1" or not bool(field_execution_validation.get("pass")):
        issues.append({"level": "error", "code": "field_execution_pack_invalid", "message": "Field/QE execution pack is missing or structurally invalid."})
    if int(field_execution_pack.get("job_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "field_execution_pack_empty", "message": "Field/QE execution pack has no remaining or stale quantitative jobs."})
    if int(field_execution_pack.get("center_anchor_job_count", 0) or 0) <= 0:
        issues.append({"level": "warning", "code": "field_execution_pack_no_center_anchors", "message": "No center spectral/color anchor jobs remain; confirm full field coverage separately."})
    if str(field_execution_pack.get("product_use_gate", "")).upper() != "FAIL":
        issues.append({"level": "error", "code": "field_execution_pack_unexpected_product_gate", "message": "Field/QE execution pack must remain product-blocked."})

    crosstalk_support_validation = crosstalk_support.get("validation", {})
    if crosstalk_support.get("schema") != "camera_e2e_crosstalk_support_audit_v1" or not bool(crosstalk_support_validation.get("pass")):
        issues.append({"level": "error", "code": "crosstalk_support_invalid", "message": "Finite-array crosstalk support audit is missing or structurally invalid."})
    if int(crosstalk_support.get("pilot_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "crosstalk_support_empty", "message": "Crosstalk support audit found no finite-array support pilot rows."})
    if bool(crosstalk_support.get("product_crosstalk_ready")):
        issues.append({"level": "error", "code": "crosstalk_support_unexpected_product_ready", "message": "Crosstalk support audit marked product-ready while product crosstalk mesh pass is incomplete."})

    crosstalk_batch_validation = crosstalk_batch_priority.get("validation", {})
    if crosstalk_batch_priority.get("schema") != "camera_e2e_crosstalk_batch_priority_v1" or not bool(crosstalk_batch_validation.get("pass")):
        issues.append({"level": "error", "code": "crosstalk_batch_priority_invalid", "message": "Support-aware crosstalk batch priority artifact is missing or structurally invalid."})
    if int(crosstalk_batch_priority.get("priority_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "crosstalk_batch_priority_empty", "message": "Crosstalk batch priority artifact has no actionable rows."})
    if int(crosstalk_batch_priority.get("product_primary_row_count", 0) or 0) <= 0:
        issues.append({"level": "warning", "code": "crosstalk_batch_priority_no_primary", "message": "No product-resolution crosstalk primary candidate is available from support-established pilots."})
    if str(crosstalk_batch_priority.get("product_use_gate", "")).upper() != "FAIL":
        issues.append({"level": "error", "code": "crosstalk_batch_priority_unexpected_product_gate", "message": "Crosstalk batch priority must remain product-blocked until selected jobs pass product gates."})

    crosstalk_execution_validation = crosstalk_execution_pack.get("validation", {})
    if crosstalk_execution_pack.get("schema") != "camera_e2e_crosstalk_execution_pack_v1" or not bool(crosstalk_execution_validation.get("pass")):
        issues.append({"level": "error", "code": "crosstalk_execution_pack_invalid", "message": "Crosstalk execution pack is missing or structurally invalid."})
    if int(crosstalk_execution_pack.get("product_primary_job_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "crosstalk_execution_pack_no_product_primary", "message": "Crosstalk execution pack has no product-primary jobs."})
    if int(crosstalk_execution_pack.get("local_probe_evidence_count", 0) or 0) <= 0:
        issues.append({"level": "warning", "code": "crosstalk_execution_pack_no_local_probe_evidence", "message": "No local crosstalk probe evidence is attached to the execution pack."})
    if str(crosstalk_execution_pack.get("product_use_gate", "")).upper() != "FAIL":
        issues.append({"level": "error", "code": "crosstalk_execution_pack_unexpected_product_gate", "message": "Crosstalk execution pack must remain product-blocked."})

    product_closure_validation = product_closure_summary.get("validation", {})
    if product_closure_summary.get("schema") != "camera_e2e_product_closure_summary_v1" or not bool(product_closure_validation.get("pass")):
        issues.append({"level": "error", "code": "product_closure_summary_invalid", "message": "Product-closure summary is missing or structurally invalid."})
    if int(product_closure_summary.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "product_closure_summary_empty", "message": "Product-closure summary has no sensor rows."})
    if int(product_closure_summary.get("product_ready_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "product_closure_summary_unexpected_product_ready", "message": "Product-closure summary marked product-ready sensors while product gates are blocked."})

    usage_policy_validation = usage_policy.get("validation", {})
    if usage_policy.get("schema") != "camera_e2e_usage_policy_v1" or not bool(usage_policy_validation.get("pass")):
        issues.append({"level": "error", "code": "usage_policy_invalid", "message": "CameraE2E usage policy is missing or structurally invalid."})
    if int(usage_policy.get("sensor_policy_row_count", 0) or 0) <= 0 or int(usage_policy.get("domain_policy_row_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "usage_policy_empty", "message": "CameraE2E usage policy has no sensor or domain rows."})
    if int(usage_policy.get("strict_product_filter_row_count", 0) or 0) != 0 or int(usage_policy.get("product_ingest_allowed_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "usage_policy_unexpected_product_rows", "message": "Usage policy exposed product-ingest rows while product gates are blocked."})

    adapter_validation = adapter_examples.get("validation", {})
    if adapter_examples.get("schema") != "camera_e2e_adapter_examples_v1" or not bool(adapter_validation.get("pass")):
        issues.append({"level": "error", "code": "adapter_examples_invalid", "message": "CameraE2E adapter examples are missing or structurally invalid."})
    if int(adapter_examples.get("sensor_count", 0) or 0) <= 0 or int(adapter_examples.get("example_file_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "adapter_examples_empty", "message": "CameraE2E adapter examples have no per-sensor JSON files."})
    if int(adapter_examples.get("product_allowed_query_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "adapter_examples_unexpected_product_queries", "message": "Adapter examples allowed product queries while product gates are blocked."})

    adapter_smoke_validation = adapter_smoke.get("validation", {})
    if adapter_smoke.get("schema") != "camera_e2e_adapter_smoke_v1" or not bool(adapter_smoke_validation.get("pass")):
        issues.append({"level": "error", "code": "adapter_smoke_invalid", "message": "CameraE2E adapter smoke is missing or structurally invalid."})
    if int(adapter_smoke.get("sensor_count", 0) or 0) <= 0 or int(adapter_smoke.get("total_research_allowed_query_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "adapter_smoke_no_research_queries", "message": "Adapter smoke did not prove research-mode query loading."})
    if int(adapter_smoke.get("total_product_allowed_query_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "adapter_smoke_unexpected_product_queries", "message": "Adapter smoke allowed product queries while product gates are blocked."})

    objective_trace_validation = objective_trace.get("validation", {})
    if objective_trace.get("schema") != "camera_e2e_objective_trace_v1" or not bool(objective_trace_validation.get("pass")):
        issues.append({"level": "error", "code": "objective_trace_invalid", "message": "CameraE2E objective trace is missing or structurally invalid."})
    if int(objective_trace.get("trace_row_count", 0) or 0) <= 0 or int(objective_trace.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "objective_trace_empty", "message": "CameraE2E objective trace has no requirement or sensor rows."})
    if int(objective_trace.get("requirement_summary_row_count", 0) or 0) != int(objective_trace.get("requirement_count_per_sensor", 0) or 0):
        issues.append({"level": "error", "code": "objective_trace_requirement_summary_mismatch", "message": "CameraE2E objective requirement summary row count does not match the objective requirement count."})
    if int(objective_trace.get("flat_pointer_fail_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "objective_trace_flat_pointer_fail", "message": "One or more CameraE2E objective flat_json_pointer values do not resolve inside per-sensor flat JSON files."})
    if int(objective_trace.get("product_ready_count", 0) or 0) != 0 or int(objective_trace.get("adapter_product_allowed_query_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "objective_trace_unexpected_product_ready", "message": "Objective trace exposed product-ready or product-allowed rows while product gates are blocked."})

    handoff_validation = handoff.get("validation", {})
    if handoff.get("schema") != "camera_e2e_handoff_manifest_v1" or not bool(handoff_validation.get("pass")):
        issues.append({"level": "error", "code": "handoff_manifest_invalid", "message": "CameraE2E handoff manifest is missing or structurally invalid."})
    if int(handoff.get("sensor_count", 0) or 0) <= 0 or int(handoff.get("artifact_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "handoff_manifest_empty", "message": "CameraE2E handoff manifest has no sensor or artifact rows."})

    handoff_loader_validation = handoff_loader.get("validation", {})
    if handoff_loader.get("schema") != "camera_e2e_handoff_loader_validation_v1" or not bool(handoff_loader_validation.get("pass")):
        issues.append({"level": "error", "code": "handoff_loader_invalid", "message": "CameraE2E handoff loader validation did not pass."})
    if int(handoff_loader.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "handoff_loader_empty", "message": "CameraE2E handoff loader validated no sensors."})

    acceptance_validation = objective_acceptance.get("validation", {})
    if objective_acceptance.get("schema") != "camera_e2e_objective_acceptance_audit_v1" or not bool(acceptance_validation.get("pass")):
        issues.append({"level": "error", "code": "objective_acceptance_invalid", "message": "CameraE2E objective acceptance audit did not pass."})
    if int(objective_acceptance.get("sensor_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "objective_acceptance_empty", "message": "CameraE2E objective acceptance audit validated no sensors."})
    if int(objective_acceptance.get("consumer_query_allowed_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "objective_research_queries_blocked", "message": "Objective acceptance audit found no allowed research consumer queries."})
    if int(objective_acceptance.get("product_query_allowed_count", 0) or 0) != 0:
        issues.append({"level": "error", "code": "objective_product_queries_unexpectedly_allowed", "message": "Objective acceptance audit found product-mode consumer queries allowed while product gates are blocked."})

    research_validation = research_query.get("validation", {})
    if not bool(research_validation.get("pass")):
        issues.append({"level": "error", "code": "research_query_invalid", "message": "Research runtime query validation did not pass."})
    if int(research_query.get("allowed_query_count", 0) or 0) <= 0:
        issues.append({"level": "error", "code": "research_query_blocked", "message": "Research runtime query returned no allowed rows."})

    product_validation = product_query.get("validation", {})
    if not bool(product_validation.get("pass")):
        issues.append({"level": "error", "code": "product_probe_invalid", "message": "Product strict probe output is structurally invalid."})
    if int(product_query.get("allowed_query_count", 0) or 0) != 0:
        issues.append(
            {
                "level": "error",
                "code": "product_probe_unexpectedly_allowed",
                "message": "Product strict probe allowed rows even though product gates are not closed.",
            }
        )

    if closure_plan.get("schema") != "camera_e2e_closure_plan_v1":
        issues.append({"level": "warning", "code": "closure_plan_missing", "message": "Closure plan was not generated."})
    elif not bool(closure_plan.get("validation", {}).get("pass")):
        issues.append({"level": "error", "code": "closure_plan_invalid", "message": "Closure plan validation failed."})
    elif int(closure_plan.get("plan_row_count", 0) or 0) <= 0:
        issues.append({"level": "warning", "code": "closure_plan_empty", "message": "Closure plan has no rows."})
    elif int(closure_plan.get("measured_calibration_input_row_count", 0) or 0) <= 0:
        issues.append(
            {
                "level": "warning",
                "code": "closure_plan_missing_calibration_blockers",
                "message": "Coverage matrix contains electrical/readout/module product blockers, but the closure plan has no measured calibration input rows.",
            }
        )
    elif (
        int(mesh_confidence.get("crosstalk_pass_total", 0) or 0) < int(mesh_confidence.get("crosstalk_required_total", 0) or 0)
        and int(closure_plan.get("resource_limited_solver_row_count", 0) or 0) <= 0
        and int(closure_plan.get("crosstalk_priority_solver_row_count", 0) or 0) <= 0
    ):
        issues.append(
            {
                "level": "warning",
                "code": "closure_plan_missing_crosstalk_work",
                "message": "Finite-array crosstalk is incomplete, but closure plan has no support-aware or resource-limited crosstalk rows.",
            }
        )

    product_ready = bool(runtime_bundle.get("product_lut_ready")) and bool(readiness.get("product_lut_ready"))
    if product_ready:
        status = "PRODUCT_READY"
    elif not issues:
        status = "RESEARCH_VALID_PRODUCT_BLOCKED"
    else:
        status = "FAIL"
    return {"schema": "camera_e2e_pipeline_validation_v1", "pass": not any(i["level"] == "error" for i in issues), "status": status, "issues": issues}


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    ingest_dir = package_dir / "camera_e2e_ingest_export"
    compact_dir = package_dir / "camera_e2e_compact_crosstalk_lut"
    combined_dir = package_dir / "camera_e2e_combined_query_pipeline_5x5_no_fail"
    readiness_dir = package_dir / "camera_e2e_readiness_audit"
    runtime_dir = package_dir / "camera_e2e_runtime_bundle"
    prior_seed_dir = package_dir / "camera_e2e_prior_seed_models"
    cfa_sync_dir = package_dir / "fdtd_stack_cfa_sync"
    electrical_readout_dir = package_dir / "camera_e2e_electrical_readout_tables"
    sensor_models_dir = package_dir / "camera_e2e_sensor_models"
    color_response_dir = package_dir / "camera_e2e_color_response"
    material_tables_dir = package_dir / "camera_e2e_material_tables"
    cfa_provenance_dir = package_dir / "camera_e2e_cfa_provenance"
    cfa_db_tables_dir = package_dir / "camera_e2e_cfa_db_tables"
    module_coupling_dir = package_dir / "camera_e2e_module_coupling"
    coverage_dir = package_dir / "camera_e2e_coverage_matrix"
    capability_profile_dir = package_dir / "camera_e2e_capability_profile"
    trust_assessment_dir = package_dir / "camera_e2e_lut_trust_assessment"
    uncertainty_budget_dir = package_dir / "camera_e2e_uncertainty_budget"
    response_trace_dir = package_dir / "camera_e2e_response_trace"
    response_example_dir = package_dir / "camera_e2e_response_example"
    method_provenance_dir = package_dir / "camera_e2e_method_provenance"
    source_integrity_dir = package_dir / "camera_e2e_lut_source_integrity"
    consumer_bundle_dir = package_dir / "camera_e2e_consumer_bundle"
    use_scope_dir = package_dir / "camera_e2e_use_scope_summary"
    flat_sensor_bundle_dir = package_dir / "camera_e2e_flat_sensor_bundle"
    import_contract_dir = package_dir / "camera_e2e_import_contract"
    canonical_payload_dir = package_dir / "camera_e2e_canonical_payload"
    flat_sensor_query_dir = package_dir / "camera_e2e_flat_sensor_query"
    flat_sensor_product_query_dir = package_dir / "camera_e2e_flat_sensor_query_product_probe"
    analysis_report_dir = package_dir / "camera_e2e_analysis_report"
    deliverable_summary_dir = package_dir / "camera_e2e_sensor_deliverable_summary"
    consumer_query_dir = package_dir / "camera_e2e_consumer_query"
    consumer_product_query_dir = package_dir / "camera_e2e_consumer_query_product_probe"
    mesh_confidence_dir = package_dir / "camera_e2e_mesh_confidence"
    field_execution_pack_dir = package_dir / "camera_e2e_field_execution_pack"
    crosstalk_support_dir = package_dir / "camera_e2e_crosstalk_support_audit"
    crosstalk_batch_priority_dir = package_dir / "camera_e2e_crosstalk_batch_priority"
    crosstalk_execution_pack_dir = package_dir / "camera_e2e_crosstalk_execution_pack"
    product_closure_summary_dir = package_dir / "camera_e2e_product_closure_summary"
    usage_policy_dir = package_dir / "camera_e2e_usage_policy"
    adapter_examples_dir = package_dir / "camera_e2e_adapter_examples"
    adapter_smoke_dir = package_dir / "camera_e2e_adapter_smoke"
    objective_trace_dir = package_dir / "camera_e2e_objective_trace"
    handoff_dir = package_dir / "camera_e2e_handoff_manifest"
    handoff_loader_dir = package_dir / "camera_e2e_handoff_loader_validation"
    objective_acceptance_dir = package_dir / "camera_e2e_objective_acceptance"
    sensor_probe_dir = package_dir / "camera_e2e_sensor_probe"
    sensor_probe_all_dir = package_dir / "camera_e2e_sensor_probe_all_sensors"
    product_probe_dir = output_dir / "runtime_query_product_strict_probe"
    research_query_dir = output_dir / "runtime_query_research_smoke"
    closure_dir = package_dir / f"camera_e2e_closure_plan_{args.prefer_slug}" if args.prefer_slug else package_dir / "camera_e2e_closure_plan"

    field_map_csv = args.field_map_csv.resolve() if args.field_map_csv else None
    steps: list[dict[str, Any]] = []

    steps.append(
        run_step(
            "sync_fdtd_stack_cfa_from_optical_db",
            python_command(
                "sync_fdtd_stack_cfa_from_optical_db.py",
                "--output-dir",
                str(cfa_sync_dir),
            ),
        )
    )

    if not args.skip_rebuild:
        build_base = python_command(
            "build_camera_e2e_sensor_luts.py",
            "--major-only",
            "--output-dir",
            str(package_dir),
        )
        steps.append(run_step("build_package_initial", add_field_map_args(build_base, field_map_csv)))
        steps.append(run_step("merge_quantitative_points", python_command("merge_camera_e2e_quantitative_points.py", "--package-dir", str(package_dir))))
        build_merged = python_command(
            "build_camera_e2e_sensor_luts.py",
            "--major-only",
            "--output-dir",
            str(package_dir),
        )
        steps.append(run_step("build_package_after_merge", add_field_map_args(build_merged, field_map_csv)))

    export_command = python_command(
        "export_camera_e2e_ingest_luts.py",
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(ingest_dir),
    )
    if args.slugs:
        export_command.extend(["--slugs", args.slugs])
    steps.append(run_step("export_ingest_luts", export_command))
    steps.append(
        run_step(
            "build_compact_crosstalk",
            python_command(
                "build_camera_e2e_compact_crosstalk_lut.py",
                "--field-lut",
                str(ingest_dir / "camera_e2e_field_response_lut.csv"),
                "--output-dir",
                str(compact_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "query_combined_5x5_no_fail",
            python_command(
                "query_camera_e2e_combined_lut.py",
                "--field-lut-json",
                str(ingest_dir / "camera_e2e_field_response_lut.json"),
                "--compact-crosstalk-csv",
                str(compact_dir / "camera_e2e_compact_crosstalk_kernel_lut.csv"),
                f"--field-x={args.combined_field_x}",
                f"--field-z={args.combined_field_z}",
                f"--wavelength-nm={args.combined_wavelength_nm}",
                "--output-dir",
                str(combined_dir),
                "--exclude-fail",
            ),
        )
    )
    steps.append(run_step("audit_readiness", python_command("audit_camera_e2e_lut_readiness.py", "--package-dir", str(package_dir), "--output-dir", str(readiness_dir))))
    steps.append(run_step("export_runtime_bundle", python_command("export_camera_e2e_runtime_bundle.py", "--package-dir", str(package_dir), "--output-dir", str(runtime_dir))))
    steps.append(
        run_step(
            "build_prior_seed_models",
            python_command(
                "build_camera_e2e_prior_seed_models.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(prior_seed_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_electrical_readout_tables",
            python_command(
                "export_camera_e2e_electrical_readout_tables.py",
                "--package-dir",
                str(package_dir),
                "--prior-dir",
                str(prior_seed_dir),
                "--output-dir",
                str(electrical_readout_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_sensor_models_base",
            python_command(
                "export_camera_e2e_sensor_models.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(sensor_models_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_color_response",
            python_command(
                "export_camera_e2e_color_response.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(color_response_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_material_tables",
            python_command(
                "export_camera_e2e_material_tables.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(material_tables_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "audit_cfa_provenance",
            python_command(
                "audit_camera_e2e_cfa_provenance.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(cfa_provenance_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_cfa_db_tables",
            python_command(
                "export_camera_e2e_cfa_db_tables.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(cfa_db_tables_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_sensor_models_handoff",
            python_command(
                "export_camera_e2e_sensor_models.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(sensor_models_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_module_coupling",
            python_command(
                "export_camera_e2e_module_coupling.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(module_coupling_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "simulate_sensor_probe_research",
            python_command(
                "simulate_camera_e2e_sensor_probe.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(sensor_probe_dir),
                "--slugs",
                args.query_slugs,
                "--field-x=0",
                "--field-z=0",
                f"--wavelength-nm={args.query_wavelength_nm}",
                "--mode",
                "research",
                "--incident-photons",
                str(args.probe_incident_photons),
                "--exposure-s",
                str(args.probe_exposure_s),
                "--temperature-c",
                str(args.probe_temperature_c),
            ),
        )
    )
    steps.append(
        run_step(
            "simulate_sensor_probe_all_sensors",
            python_command(
                "simulate_camera_e2e_sensor_probe.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(sensor_probe_all_dir),
                "--slugs",
                "all",
                "--field-x=-1,0,1",
                "--field-z=-1,0,1",
                "--wavelength-nm",
                "all",
                "--mode",
                "research",
                "--incident-photons",
                str(args.probe_incident_photons),
                "--exposure-s",
                str(args.probe_exposure_s),
                "--temperature-c",
                str(args.probe_temperature_c),
            ),
        )
    )
    steps.append(
        run_step(
            "export_coverage_matrix",
            python_command(
                "export_camera_e2e_coverage_matrix.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(coverage_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "audit_mesh_confidence",
            python_command(
                "audit_camera_e2e_mesh_confidence.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(mesh_confidence_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_field_execution_pack",
            python_command(
                "export_camera_e2e_field_execution_pack.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(field_execution_pack_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "audit_crosstalk_support",
            python_command(
                "audit_camera_e2e_crosstalk_support.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(crosstalk_support_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_crosstalk_batch_priority",
            python_command(
                "export_camera_e2e_crosstalk_batch_priority.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(crosstalk_batch_priority_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_crosstalk_execution_pack",
            python_command(
                "export_camera_e2e_crosstalk_execution_pack.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(crosstalk_execution_pack_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_capability_profile",
            python_command(
                "export_camera_e2e_capability_profile.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(capability_profile_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_lut_trust_assessment",
            python_command(
                "export_camera_e2e_lut_trust_assessment.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(trust_assessment_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_uncertainty_budget",
            python_command(
                "export_camera_e2e_uncertainty_budget.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(uncertainty_budget_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_response_trace",
            python_command(
                "export_camera_e2e_response_trace.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(response_trace_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_response_example",
            python_command(
                "export_camera_e2e_response_example.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(response_example_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_method_provenance",
            python_command(
                "export_camera_e2e_method_provenance.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(method_provenance_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_source_integrity",
            python_command(
                "export_camera_e2e_lut_source_integrity.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(source_integrity_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_consumer_bundle",
            python_command(
                "export_camera_e2e_consumer_bundle.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(consumer_bundle_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_use_scope_summary",
            python_command(
                "export_camera_e2e_use_scope_summary.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(use_scope_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_flat_sensor_bundle",
            python_command(
                "export_camera_e2e_flat_sensor_bundle.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(flat_sensor_bundle_dir),
            ),
        )
    )
    consumer_bundle_json = consumer_bundle_dir / "camera_e2e_consumer_bundle.json"
    flat_sensor_bundle_json = flat_sensor_bundle_dir / "camera_e2e_flat_sensor_bundle.json"
    import_contract_json = import_contract_dir / "camera_e2e_import_contract.json"
    steps.append(
        run_step(
            "export_import_contract",
            python_command(
                "export_camera_e2e_import_contract.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(import_contract_dir),
                "--flat-bundle-json",
                str(flat_sensor_bundle_json),
            ),
        )
    )
    steps.append(
        run_step(
            "export_canonical_payload",
            python_command(
                "export_camera_e2e_canonical_payload.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(canonical_payload_dir),
                "--flat-bundle-json",
                str(flat_sensor_bundle_json),
                "--import-contract-json",
                str(import_contract_json),
            ),
        )
    )
    steps.append(
        run_step(
            "flat_sensor_query_research",
            python_command(
                "query_camera_e2e_flat_sensor_bundle.py",
                "--flat-bundle-json",
                str(flat_sensor_bundle_json),
                "--output-dir",
                str(flat_sensor_query_dir),
                "--slugs",
                "all",
                "--field-x=-1,0,1",
                "--field-z=-1,0,1",
                "--wavelength-nm",
                "all",
                "--mode",
                "research",
            ),
        )
    )
    steps.append(
        run_step(
            "flat_sensor_query_product_block_probe",
            python_command(
                "query_camera_e2e_flat_sensor_bundle.py",
                "--flat-bundle-json",
                str(flat_sensor_bundle_json),
                "--output-dir",
                str(flat_sensor_product_query_dir),
                "--slugs",
                "all",
                "--field-x=-1,0,1",
                "--field-z=-1,0,1",
                "--wavelength-nm",
                "all",
                "--mode",
                "product",
            ),
        )
    )
    steps.append(
        run_step(
            "export_analysis_report",
            python_command(
                "export_camera_e2e_analysis_report.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(analysis_report_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_sensor_deliverable_summary",
            python_command(
                "export_camera_e2e_sensor_deliverable_summary.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(deliverable_summary_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "consumer_query_research_smoke",
            python_command(
                "query_camera_e2e_consumer_bundle.py",
                "--consumer-bundle-json",
                str(consumer_bundle_json),
                "--output-dir",
                str(consumer_query_dir),
                "--slugs",
                "all",
                "--field-x",
                "0",
                "--field-z",
                "0",
                "--wavelength-nm",
                "550",
                "--mode",
                "research",
            ),
        )
    )
    steps.append(
        run_step(
            "consumer_query_product_block_probe",
            python_command(
                "query_camera_e2e_consumer_bundle.py",
                "--consumer-bundle-json",
                str(consumer_bundle_json),
                "--output-dir",
                str(consumer_product_query_dir),
                "--slugs",
                "all",
                "--field-x",
                "0",
                "--field-z",
                "0",
                "--wavelength-nm",
                "550",
                "--mode",
                "product",
            ),
        )
    )
    bundle_json = runtime_dir / "camera_e2e_runtime_bundle.json"
    steps.append(
        run_step(
            "runtime_query_product_strict_probe",
            python_command(
                "query_camera_e2e_runtime_bundle.py",
                "--bundle-json",
                str(bundle_json),
                "--output-dir",
                str(product_probe_dir),
                "--slugs",
                args.query_slugs,
                "--field-x=0",
                "--field-z=0",
                f"--wavelength-nm={args.query_wavelength_nm}",
                "--mode",
                "product",
                "--strict",
            ),
            expected_returncodes={2} if args.expect_product_blocked else {0},
        )
    )
    steps.append(
        run_step(
            "runtime_query_research_smoke",
            python_command(
                "query_camera_e2e_runtime_bundle.py",
                "--bundle-json",
                str(bundle_json),
                "--output-dir",
                str(research_query_dir),
                "--slugs",
                args.query_slugs,
                f"--field-x={args.query_field_x}",
                f"--field-z={args.query_field_z}",
                f"--wavelength-nm={args.query_wavelength_nm}",
                "--mode",
                "research",
            ),
        )
    )
    closure_command = python_command(
        "plan_camera_e2e_closure.py",
        "--package-dir",
        str(package_dir),
        "--output-dir",
        str(closure_dir),
        "--max-solver-points",
        str(args.max_solver_points),
    )
    if args.include_failed:
        closure_command.append("--include-failed")
    if args.prefer_slug:
        closure_command.extend(["--prefer-slug", args.prefer_slug])
    steps.append(run_step("plan_closure", closure_command))
    steps.append(
        run_step(
            "export_product_closure_summary",
            python_command(
                "export_camera_e2e_product_closure_summary.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(product_closure_summary_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_usage_policy",
            python_command(
                "export_camera_e2e_usage_policy.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(usage_policy_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_adapter_examples",
            python_command(
                "export_camera_e2e_adapter_examples.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(adapter_examples_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "run_adapter_smoke",
            python_command(
                "run_camera_e2e_adapter_smoke.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(adapter_smoke_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_objective_trace",
            python_command(
                "export_camera_e2e_objective_trace.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(objective_trace_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "export_handoff_manifest",
            python_command(
                "export_camera_e2e_handoff_manifest.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(handoff_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "validate_handoff_loader",
            python_command(
                "validate_camera_e2e_handoff_loader.py",
                "--package-dir",
                str(package_dir),
                "--handoff-dir",
                str(handoff_dir),
                "--output-dir",
                str(handoff_loader_dir),
            ),
        )
    )
    steps.append(
        run_step(
            "audit_objective_acceptance",
            python_command(
                "audit_camera_e2e_objective_acceptance.py",
                "--package-dir",
                str(package_dir),
                "--output-dir",
                str(objective_acceptance_dir),
            ),
        )
    )

    readiness = read_json(readiness_dir / "camera_e2e_lut_readiness_report.json")
    runtime_bundle = read_json(bundle_json)
    prior_seed_models = read_json(prior_seed_dir / "camera_e2e_prior_seed_models.json")
    cfa_sync = read_json(cfa_sync_dir / "fdtd_stack_cfa_sync.json")
    electrical_readout = read_json(electrical_readout_dir / "camera_e2e_electrical_readout_tables.json")
    sensor_models = read_json(sensor_models_dir / "camera_e2e_sensor_models.json")
    color_response = read_json(color_response_dir / "camera_e2e_color_response.json")
    material_tables = read_json(material_tables_dir / "camera_e2e_material_tables.json")
    cfa_provenance = read_json(cfa_provenance_dir / "camera_e2e_cfa_provenance.json")
    cfa_db_tables = read_json(cfa_db_tables_dir / "camera_e2e_cfa_db_tables.json")
    module_coupling = read_json(module_coupling_dir / "camera_e2e_module_coupling.json")
    coverage_matrix = read_json(coverage_dir / "camera_e2e_coverage_matrix.json")
    capability_profile = read_json(capability_profile_dir / "camera_e2e_capability_profile.json")
    trust_assessment = read_json(trust_assessment_dir / "camera_e2e_lut_trust_assessment.json")
    uncertainty_budget = read_json(uncertainty_budget_dir / "camera_e2e_uncertainty_budget.json")
    response_trace = read_json(response_trace_dir / "camera_e2e_response_trace.json")
    response_example = read_json(response_example_dir / "camera_e2e_response_example.json")
    method_provenance = read_json(method_provenance_dir / "camera_e2e_method_provenance.json")
    source_integrity = read_json(source_integrity_dir / "camera_e2e_lut_source_integrity.json")
    consumer_bundle = read_json(consumer_bundle_dir / "camera_e2e_consumer_bundle.json")
    use_scope_summary = read_json(use_scope_dir / "camera_e2e_use_scope_summary.json")
    flat_sensor_bundle = read_json(flat_sensor_bundle_dir / "camera_e2e_flat_sensor_bundle.json")
    import_contract = read_json(import_contract_dir / "camera_e2e_import_contract.json")
    canonical_payload = read_json(canonical_payload_dir / "camera_e2e_canonical_payload.json")
    flat_sensor_query = read_json(flat_sensor_query_dir / "camera_e2e_flat_sensor_query.json")
    flat_sensor_product_query = read_json(flat_sensor_product_query_dir / "camera_e2e_flat_sensor_query.json")
    analysis_report = read_json(analysis_report_dir / "camera_e2e_analysis_report.json")
    deliverable_summary = read_json(deliverable_summary_dir / "camera_e2e_sensor_deliverable_summary.json")
    consumer_query = read_json(consumer_query_dir / "camera_e2e_consumer_query.json")
    consumer_product_query = read_json(consumer_product_query_dir / "camera_e2e_consumer_query.json")
    mesh_confidence = read_json(mesh_confidence_dir / "camera_e2e_mesh_confidence.json")
    field_execution_pack = read_json(field_execution_pack_dir / "camera_e2e_field_execution_pack.json")
    crosstalk_support = read_json(crosstalk_support_dir / "camera_e2e_crosstalk_support_audit.json")
    crosstalk_batch_priority = read_json(crosstalk_batch_priority_dir / "camera_e2e_crosstalk_batch_priority.json")
    crosstalk_execution_pack = read_json(crosstalk_execution_pack_dir / "camera_e2e_crosstalk_execution_pack.json")
    product_closure_summary = read_json(product_closure_summary_dir / "camera_e2e_product_closure_summary.json")
    usage_policy = read_json(usage_policy_dir / "camera_e2e_usage_policy.json")
    adapter_examples = read_json(adapter_examples_dir / "camera_e2e_adapter_examples.json")
    adapter_smoke = read_json(adapter_smoke_dir / "camera_e2e_adapter_smoke.json")
    objective_trace = read_json(objective_trace_dir / "camera_e2e_objective_trace.json")
    handoff = read_json(handoff_dir / "camera_e2e_handoff_manifest.json")
    handoff_loader = read_json(handoff_loader_dir / "camera_e2e_handoff_loader_validation.json")
    objective_acceptance = read_json(objective_acceptance_dir / "camera_e2e_objective_acceptance.json")
    sensor_probe = read_json(sensor_probe_dir / "camera_e2e_sensor_probe.json")
    sensor_probe_all = read_json(sensor_probe_all_dir / "camera_e2e_sensor_probe.json")
    research_query = read_json(research_query_dir / "camera_e2e_runtime_query.json")
    product_query = read_json(product_probe_dir / "camera_e2e_runtime_query.json")
    closure_plan = read_json(closure_dir / "camera_e2e_closure_plan.json")
    validation = validate_pipeline(
        steps=steps,
        readiness=readiness,
        runtime_bundle=runtime_bundle,
        material_tables=material_tables,
        cfa_provenance=cfa_provenance,
        cfa_db_tables=cfa_db_tables,
        electrical_readout=electrical_readout,
        module_coupling=module_coupling,
        coverage_matrix=coverage_matrix,
        capability_profile=capability_profile,
        trust_assessment=trust_assessment,
        uncertainty_budget=uncertainty_budget,
        response_trace=response_trace,
        response_example=response_example,
        method_provenance=method_provenance,
        source_integrity=source_integrity,
        consumer_bundle=consumer_bundle,
        use_scope_summary=use_scope_summary,
        flat_sensor_bundle=flat_sensor_bundle,
        import_contract=import_contract,
        canonical_payload=canonical_payload,
        flat_sensor_query=flat_sensor_query,
        flat_sensor_product_query=flat_sensor_product_query,
        analysis_report=analysis_report,
        deliverable_summary=deliverable_summary,
        consumer_query=consumer_query,
        consumer_product_query=consumer_product_query,
        mesh_confidence=mesh_confidence,
        field_execution_pack=field_execution_pack,
        crosstalk_support=crosstalk_support,
        crosstalk_batch_priority=crosstalk_batch_priority,
        crosstalk_execution_pack=crosstalk_execution_pack,
        product_closure_summary=product_closure_summary,
        usage_policy=usage_policy,
        adapter_examples=adapter_examples,
        adapter_smoke=adapter_smoke,
        objective_trace=objective_trace,
        handoff=handoff,
        handoff_loader=handoff_loader,
        objective_acceptance=objective_acceptance,
        research_query=research_query,
        product_query=product_query,
        closure_plan=closure_plan,
    )

    report_json = output_dir / "camera_e2e_pipeline_validation.json"
    steps_csv = output_dir / "camera_e2e_pipeline_steps.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_pipeline_run_v1",
        "artifact_role": "camera_e2e_rebuild_and_gate_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "step_count": len(steps),
        "validation": validation,
        "product_lut_ready": bool(runtime_bundle.get("product_lut_ready")) and bool(readiness.get("product_lut_ready")),
        "readiness_summary": {
            "research_ingest_valid": readiness.get("research_ingest_valid"),
            "research_gate_counts": readiness.get("research_gate_counts"),
            "production_gate_counts": readiness.get("production_gate_counts"),
        },
        "runtime_row_count": runtime_bundle.get("runtime_row_count", 0),
        "kernel_row_count": runtime_bundle.get("kernel_row_count", 0),
        "prior_seed_model_count": prior_seed_models.get("sensor_count", 0),
        "fdtd_stack_cfa_sync_status": cfa_sync.get("validation", {}).get("status", ""),
        "fdtd_stack_cfa_sync_updated_count": cfa_sync.get("updated_count", 0),
        "fdtd_stack_cfa_sync_unchanged_count": cfa_sync.get("unchanged_count", 0),
        "fdtd_stack_cfa_sync_skip_count": cfa_sync.get("skip_count", 0),
        "fdtd_stack_cfa_sync_fail_count": cfa_sync.get("fail_count", 0),
        "electrical_row_count": electrical_readout.get("electrical_row_count", 0),
        "readout_row_count": electrical_readout.get("readout_row_count", 0),
        "binning_row_count": electrical_readout.get("binning_row_count", 0),
        "electrical_readout_gate_counts": electrical_readout.get("gate_counts", {}),
        "sensor_model_count": sensor_models.get("sensor_count", 0),
        "sensor_model_product_ready_count": sensor_models.get("product_ready_count", 0),
        "color_response_spectral_row_count": color_response.get("spectral_row_count", 0),
        "color_response_matrix_row_count": color_response.get("matrix_row_count", 0),
        "material_row_count": material_tables.get("material_row_count", 0),
        "material_summary_row_count": material_tables.get("summary_row_count", 0),
        "material_gate_counts": material_tables.get("gate_counts", {}),
        "cfa_provenance_status": cfa_provenance.get("status", ""),
        "cfa_provenance_class_counts": cfa_provenance.get("class_counts", {}),
        "cfa_assumption_gate_counts": cfa_provenance.get("assumption_gate_counts", {}),
        "cfa_generic_rgb_fallback_unknown_pattern_count": cfa_provenance.get("generic_rgb_fallback_unknown_pattern_count", 0),
        "cfa_db_status": cfa_db_tables.get("validation", {}).get("status", ""),
        "cfa_db_sensor_count": cfa_db_tables.get("sensor_count", 0),
        "cfa_db_transmission_row_count": cfa_db_tables.get("transmission_row_count", 0),
        "cfa_db_generic_rgb_fallback_unknown_pattern_count": cfa_db_tables.get("generic_rgb_fallback_unknown_pattern_count", 0),
        "cfa_db_provenance_class_counts": cfa_db_tables.get("cfa_provenance_class_counts", {}),
        "cfa_db_assumption_gate_counts": cfa_db_tables.get("cfa_assumption_gate_counts", {}),
        "module_coupling_field_row_count": module_coupling.get("field_row_count", 0),
        "module_coupling_summary_row_count": module_coupling.get("summary_row_count", 0),
        "module_coupling_gate_counts": module_coupling.get("gate_counts", {}),
        "module_coupling_research_gate_counts": module_coupling.get("research_gate_counts", {}),
        "module_coupling_product_gate_counts": module_coupling.get("product_gate_counts", {}),
        "coverage_row_count": coverage_matrix.get("coverage_row_count", 0),
        "coverage_status": coverage_matrix.get("validation", {}).get("status", ""),
        "coverage_product_ready_count": coverage_matrix.get("product_ready_count", 0),
        "capability_profile_status": capability_profile.get("validation", {}).get("status", ""),
        "capability_profile_scope_counts": capability_profile.get("scope_counts", {}),
        "capability_profile_product_ready_count": capability_profile.get("product_ready_count", 0),
        "lut_trust_status": trust_assessment.get("validation", {}).get("status", ""),
        "lut_trust_sensor_count": trust_assessment.get("sensor_count", 0),
        "lut_trust_mean_evidence_confidence_score_0_100": trust_assessment.get("mean_evidence_confidence_score_0_100", 0),
        "lut_trust_class_counts": trust_assessment.get("trust_class_counts", {}),
        "lut_trust_product_ready_count": trust_assessment.get("product_ready_count", 0),
        "uncertainty_budget_status": uncertainty_budget.get("validation", {}).get("status", ""),
        "uncertainty_budget_sensor_count": uncertainty_budget.get("sensor_count", 0),
        "uncertainty_budget_domain_row_count": uncertainty_budget.get("domain_row_count", 0),
        "uncertainty_budget_product_ready_count": uncertainty_budget.get("product_ready_count", 0),
        "response_trace_status": response_trace.get("validation", {}).get("status", ""),
        "response_trace_sensor_count": response_trace.get("sensor_count", 0),
        "response_trace_row_count": response_trace.get("trace_row_count", 0),
        "response_trace_product_ready_count": response_trace.get("product_ready_count", 0),
        "response_example_status": response_example.get("validation", {}).get("status", ""),
        "response_example_sensor_count": response_example.get("sensor_count", 0),
        "response_example_row_count": response_example.get("example_row_count", 0),
        "response_example_product_ready_count": response_example.get("product_ready_count", 0),
        "method_provenance_status": method_provenance.get("validation", {}).get("status", ""),
        "method_provenance_sensor_count": method_provenance.get("sensor_count", 0),
        "method_provenance_row_count": method_provenance.get("matrix_row_count", 0),
        "method_provenance_product_ready_count": method_provenance.get("product_ready_count", 0),
        "method_provenance_source_class_counts": method_provenance.get("source_class_counts", {}),
        "source_integrity_status": source_integrity.get("validation", {}).get("status", ""),
        "source_integrity_sensor_count": source_integrity.get("sensor_count", 0),
        "source_integrity_row_count": source_integrity.get("matrix_row_count", 0),
        "source_integrity_product_ready_count": source_integrity.get("product_ready_count", 0),
        "source_integrity_source_class_counts": source_integrity.get("source_class_counts", {}),
        "source_integrity_gate_counts": source_integrity.get("source_integrity_gate_counts", {}),
        "consumer_bundle_status": consumer_bundle.get("validation", {}).get("status", ""),
        "consumer_bundle_sensor_count": consumer_bundle.get("sensor_count", 0),
        "consumer_bundle_requirement_count": consumer_bundle.get("requirement_count", 0),
        "consumer_bundle_requirement_load_map_count": len(consumer_bundle.get("requirement_load_map", [])) if isinstance(consumer_bundle.get("requirement_load_map"), list) else 0,
        "consumer_bundle_product_ready_count": consumer_bundle.get("product_ready_count", 0),
        "use_scope_status": use_scope_summary.get("validation", {}).get("status", ""),
        "use_scope_sensor_count": use_scope_summary.get("sensor_count", 0),
        "use_scope_domain_row_count": use_scope_summary.get("domain_row_count", 0),
        "use_scope_next_action_row_count": use_scope_summary.get("next_action_row_count", 0),
        "use_scope_product_ready_count": use_scope_summary.get("product_ready_count", 0),
        "use_scope_counts": use_scope_summary.get("use_scope_counts", {}),
        "use_scope_domain_scope_counts": use_scope_summary.get("domain_scope_counts", {}),
        "flat_sensor_bundle_status": flat_sensor_bundle.get("validation", {}).get("status", ""),
        "flat_sensor_bundle_sensor_count": flat_sensor_bundle.get("sensor_count", 0),
        "flat_sensor_bundle_product_ready_count": flat_sensor_bundle.get("product_ready_count", 0),
        "flat_sensor_bundle_total_embedded_row_count": flat_sensor_bundle.get("total_embedded_row_count", 0),
        "import_contract_status": import_contract.get("validation", {}).get("status", ""),
        "import_contract_sensor_count": import_contract.get("sensor_count", 0),
        "import_contract_requirement_row_count": import_contract.get("requirement_row_count", 0),
        "import_contract_pointer_resolved_count": import_contract.get("pointer_resolved_count", 0),
        "import_contract_product_allowed_requirement_count": import_contract.get("product_allowed_requirement_count", 0),
        "canonical_payload_status": canonical_payload.get("validation", {}).get("status", ""),
        "canonical_payload_sensor_count": canonical_payload.get("sensor_count", 0),
        "canonical_payload_requirement_row_count": canonical_payload.get("requirement_row_count", 0),
        "canonical_payload_pointer_resolved_count": canonical_payload.get("pointer_resolved_count", 0),
        "canonical_payload_product_allowed_requirement_count": canonical_payload.get("product_allowed_requirement_count", 0),
        "flat_sensor_query_status": flat_sensor_query.get("validation", {}).get("status", ""),
        "flat_sensor_query_row_count": flat_sensor_query.get("query_row_count", 0),
        "flat_sensor_query_allowed_count": flat_sensor_query.get("allowed_query_count", 0),
        "flat_sensor_product_query_status": flat_sensor_product_query.get("validation", {}).get("status", ""),
        "flat_sensor_product_query_allowed_count": flat_sensor_product_query.get("allowed_query_count", 0),
        "analysis_report_status": analysis_report.get("validation", {}).get("status", ""),
        "analysis_report_sensor_count": analysis_report.get("sensor_count", 0),
        "analysis_report_channel_row_count": analysis_report.get("channel_row_count", 0),
        "analysis_report_check_channel_row_count": analysis_report.get("check_channel_row_count", 0),
        "sensor_deliverable_summary_status": deliverable_summary.get("validation", {}).get("status", ""),
        "sensor_deliverable_summary_sensor_count": deliverable_summary.get("sensor_count", 0),
        "sensor_deliverable_summary_product_ready_count": deliverable_summary.get("product_ready_count", 0),
        "sensor_deliverable_summary_gate_counts": deliverable_summary.get("deliverable_gate_counts", {}),
        "sensor_deliverable_summary_use_scope_counts": deliverable_summary.get("use_scope_counts", {}),
        "consumer_query_status": consumer_query.get("validation", {}).get("status", ""),
        "consumer_query_row_count": consumer_query.get("query_row_count", 0),
        "consumer_query_allowed_count": consumer_query.get("allowed_query_count", 0),
        "consumer_product_query_status": consumer_product_query.get("validation", {}).get("status", ""),
        "consumer_product_query_allowed_count": consumer_product_query.get("allowed_query_count", 0),
        "mesh_confidence_status": mesh_confidence.get("status", ""),
        "mesh_confidence_field_pass_total": mesh_confidence.get("field_pass_total", 0),
        "mesh_confidence_field_required_total": mesh_confidence.get("field_required_total", 0),
        "mesh_confidence_crosstalk_pass_total": mesh_confidence.get("crosstalk_pass_total", 0),
        "mesh_confidence_crosstalk_required_total": mesh_confidence.get("crosstalk_required_total", 0),
        "mesh_confidence_class_counts": mesh_confidence.get("confidence_class_counts", {}),
        "mesh_confidence_product_ready_count": mesh_confidence.get("product_ready_count", 0),
        "field_execution_pack_status": field_execution_pack.get("validation", {}).get("status", ""),
        "field_execution_pack_job_count": field_execution_pack.get("job_count", 0),
        "field_execution_pack_center_anchor_count": field_execution_pack.get("center_anchor_job_count", 0),
        "field_execution_pack_green_cra_anchor_count": field_execution_pack.get("green_cra_anchor_job_count", 0),
        "field_execution_pack_failed_or_stale_count": field_execution_pack.get("failed_or_stale_job_count", 0),
        "crosstalk_support_status": crosstalk_support.get("validation", {}).get("status", ""),
        "crosstalk_support_pilot_row_count": crosstalk_support.get("pilot_row_count", 0),
        "crosstalk_support_min_truncation_fraction": crosstalk_support.get("min_truncation_fraction", ""),
        "crosstalk_support_product_ready": crosstalk_support.get("product_crosstalk_ready", False),
        "crosstalk_batch_priority_status": crosstalk_batch_priority.get("validation", {}).get("status", ""),
        "crosstalk_batch_priority_row_count": crosstalk_batch_priority.get("priority_row_count", 0),
        "crosstalk_batch_priority_product_primary_count": crosstalk_batch_priority.get("product_primary_row_count", 0),
        "crosstalk_batch_priority_support_discovery_count": crosstalk_batch_priority.get("support_discovery_row_count", 0),
        "crosstalk_execution_pack_status": crosstalk_execution_pack.get("validation", {}).get("status", ""),
        "crosstalk_execution_pack_job_count": crosstalk_execution_pack.get("job_count", 0),
        "crosstalk_execution_pack_product_primary_count": crosstalk_execution_pack.get("product_primary_job_count", 0),
        "crosstalk_execution_pack_support_local_candidate_count": crosstalk_execution_pack.get("support_local_candidate_count", 0),
        "crosstalk_execution_pack_support_batch_or_reformulation_count": crosstalk_execution_pack.get("support_batch_or_reformulation_count", 0),
        "crosstalk_execution_pack_local_probe_evidence_count": crosstalk_execution_pack.get("local_probe_evidence_count", 0),
        "product_closure_summary_status": product_closure_summary.get("validation", {}).get("status", ""),
        "product_closure_summary_sensor_count": product_closure_summary.get("sensor_count", 0),
        "product_closure_summary_domain_row_count": product_closure_summary.get("domain_row_count", 0),
        "product_closure_summary_product_ready_count": product_closure_summary.get("product_ready_count", 0),
        "usage_policy_status": usage_policy.get("validation", {}).get("status", ""),
        "usage_policy_sensor_policy_row_count": usage_policy.get("sensor_policy_row_count", 0),
        "usage_policy_domain_policy_row_count": usage_policy.get("domain_policy_row_count", 0),
        "usage_policy_runtime_filter_row_count": usage_policy.get("runtime_filter_row_count", 0),
        "usage_policy_strict_product_filter_row_count": usage_policy.get("strict_product_filter_row_count", 0),
        "usage_policy_product_ingest_allowed_count": usage_policy.get("product_ingest_allowed_count", 0),
        "usage_policy_profile_counts": usage_policy.get("profile_counts", {}),
        "adapter_examples_status": adapter_examples.get("validation", {}).get("status", ""),
        "adapter_examples_sensor_count": adapter_examples.get("sensor_count", 0),
        "adapter_examples_file_count": adapter_examples.get("example_file_count", 0),
        "adapter_examples_product_allowed_query_count": adapter_examples.get("product_allowed_query_count", 0),
        "adapter_examples_profile_counts": adapter_examples.get("profile_counts", {}),
        "adapter_smoke_status": adapter_smoke.get("validation", {}).get("status", ""),
        "adapter_smoke_sensor_count": adapter_smoke.get("sensor_count", 0),
        "adapter_smoke_total_research_allowed_query_count": adapter_smoke.get("total_research_allowed_query_count", 0),
        "adapter_smoke_total_product_allowed_query_count": adapter_smoke.get("total_product_allowed_query_count", 0),
        "adapter_smoke_profile_counts": adapter_smoke.get("profile_counts", {}),
        "objective_trace_status": objective_trace.get("validation", {}).get("status", ""),
        "objective_trace_sensor_count": objective_trace.get("sensor_count", 0),
        "objective_trace_requirement_count_per_sensor": objective_trace.get("requirement_count_per_sensor", 0),
        "objective_trace_requirement_summary_row_count": objective_trace.get("requirement_summary_row_count", 0),
        "objective_trace_row_count": objective_trace.get("trace_row_count", 0),
        "objective_trace_gate_counts": objective_trace.get("trace_gate_counts", {}),
        "objective_trace_flat_pointer_gate_counts": objective_trace.get("flat_pointer_gate_counts", {}),
        "objective_trace_flat_pointer_fail_count": objective_trace.get("flat_pointer_fail_count", 0),
        "objective_trace_product_ready_count": objective_trace.get("product_ready_count", 0),
        "objective_trace_adapter_product_allowed_query_count": objective_trace.get("adapter_product_allowed_query_count", 0),
        "handoff_status": handoff.get("validation", {}).get("status", ""),
        "handoff_artifact_count": handoff.get("artifact_count", 0),
        "handoff_loader_status": handoff_loader.get("validation", {}).get("status", ""),
        "handoff_loader_sensor_count": handoff_loader.get("sensor_count", 0),
        "handoff_loader_issue_count": handoff_loader.get("validation", {}).get("issue_count", 0),
        "objective_acceptance_status": objective_acceptance.get("validation", {}).get("status", ""),
        "objective_acceptance_sensor_count": objective_acceptance.get("sensor_count", 0),
        "objective_acceptance_requirement_count_per_sensor": objective_acceptance.get("requirement_count_per_sensor", 0),
        "objective_acceptance_product_ready_count": objective_acceptance.get("product_ready_count", 0),
        "objective_acceptance_consumer_allowed_count": objective_acceptance.get("consumer_query_allowed_count", 0),
        "objective_acceptance_product_allowed_count": objective_acceptance.get("product_query_allowed_count", 0),
        "closure_plan_row_count": closure_plan.get("plan_row_count", 0),
        "closure_plan_runnable_solver_row_count": closure_plan.get("runnable_solver_row_count", 0),
        "closure_plan_measured_input_row_count": closure_plan.get("measured_input_row_count", 0),
        "closure_plan_measured_calibration_input_row_count": closure_plan.get("measured_calibration_input_row_count", 0),
        "closure_plan_resource_limited_solver_row_count": closure_plan.get("resource_limited_solver_row_count", 0),
        "closure_plan_crosstalk_priority_solver_row_count": closure_plan.get("crosstalk_priority_solver_row_count", 0),
        "closure_plan_crosstalk_product_primary_solver_row_count": closure_plan.get("crosstalk_product_primary_solver_row_count", 0),
        "closure_plan_crosstalk_support_discovery_solver_row_count": closure_plan.get("crosstalk_support_discovery_solver_row_count", 0),
        "closure_plan_estimated_solver_hours": closure_plan.get("estimated_solver_hours", 0),
        "closure_plan_status": closure_plan.get("validation", {}).get("status", ""),
        "sensor_probe_row_count": sensor_probe.get("validation", {}).get("probe_row_count", 0),
        "sensor_probe_allowed_count": sensor_probe.get("validation", {}).get("allowed_probe_count", 0),
        "all_sensor_probe_row_count": sensor_probe_all.get("validation", {}).get("probe_row_count", 0),
        "all_sensor_probe_allowed_count": sensor_probe_all.get("validation", {}).get("allowed_probe_count", 0),
        "runtime_gate_counts": runtime_bundle.get("runtime_gate_counts", {}),
        "runtime_research_query_row_count": research_query.get("validation", {}).get("query_row_count", research_query.get("query_row_count", 0)),
        "runtime_research_allowed_query_count": research_query.get("allowed_query_count", 0),
        "research_allowed_query_count": research_query.get("allowed_query_count", 0),
        "product_strict_status": "blocked_as_expected"
        if next((step for step in steps if step.get("step") == "runtime_query_product_strict_probe"), {}).get("returncode") == 2
        else "allowed_or_failed_unexpectedly",
        "steps": steps,
        "source_artifacts": {
            "readiness_report": repo_rel(readiness_dir / "camera_e2e_lut_readiness_report.json"),
            "runtime_bundle": repo_rel(bundle_json),
            "prior_seed_models": repo_rel(prior_seed_dir / "camera_e2e_prior_seed_models.json"),
            "fdtd_stack_cfa_sync": repo_rel(cfa_sync_dir / "fdtd_stack_cfa_sync.json"),
            "electrical_readout_tables": repo_rel(electrical_readout_dir / "camera_e2e_electrical_readout_tables.json"),
            "sensor_models": repo_rel(sensor_models_dir / "camera_e2e_sensor_models.json"),
            "color_response": repo_rel(color_response_dir / "camera_e2e_color_response.json"),
            "material_tables": repo_rel(material_tables_dir / "camera_e2e_material_tables.json"),
            "cfa_provenance": repo_rel(cfa_provenance_dir / "camera_e2e_cfa_provenance.json"),
            "cfa_db_tables": repo_rel(cfa_db_tables_dir / "camera_e2e_cfa_db_tables.json"),
            "module_coupling": repo_rel(module_coupling_dir / "camera_e2e_module_coupling.json"),
            "coverage_matrix": repo_rel(coverage_dir / "camera_e2e_coverage_matrix.json"),
            "capability_profile": repo_rel(capability_profile_dir / "camera_e2e_capability_profile.json"),
            "lut_trust_assessment": repo_rel(trust_assessment_dir / "camera_e2e_lut_trust_assessment.json"),
            "uncertainty_budget": repo_rel(uncertainty_budget_dir / "camera_e2e_uncertainty_budget.json"),
            "response_trace": repo_rel(response_trace_dir / "camera_e2e_response_trace.json"),
            "response_example": repo_rel(response_example_dir / "camera_e2e_response_example.json"),
            "method_provenance": repo_rel(method_provenance_dir / "camera_e2e_method_provenance.json"),
            "source_integrity": repo_rel(source_integrity_dir / "camera_e2e_lut_source_integrity.json"),
            "consumer_bundle": repo_rel(consumer_bundle_dir / "camera_e2e_consumer_bundle.json"),
            "use_scope_summary": repo_rel(use_scope_dir / "camera_e2e_use_scope_summary.json"),
            "flat_sensor_bundle": repo_rel(flat_sensor_bundle_dir / "camera_e2e_flat_sensor_bundle.json"),
            "import_contract": repo_rel(import_contract_dir / "camera_e2e_import_contract.json"),
            "canonical_payload": repo_rel(canonical_payload_dir / "camera_e2e_canonical_payload.json"),
            "flat_sensor_query": repo_rel(flat_sensor_query_dir / "camera_e2e_flat_sensor_query.json"),
            "flat_sensor_product_query": repo_rel(flat_sensor_product_query_dir / "camera_e2e_flat_sensor_query.json"),
            "analysis_report": repo_rel(analysis_report_dir / "camera_e2e_analysis_report.json"),
            "sensor_deliverable_summary": repo_rel(deliverable_summary_dir / "camera_e2e_sensor_deliverable_summary.json"),
            "consumer_query": repo_rel(consumer_query_dir / "camera_e2e_consumer_query.json"),
            "consumer_product_query": repo_rel(consumer_product_query_dir / "camera_e2e_consumer_query.json"),
            "mesh_confidence": repo_rel(mesh_confidence_dir / "camera_e2e_mesh_confidence.json"),
            "field_execution_pack": repo_rel(field_execution_pack_dir / "camera_e2e_field_execution_pack.json"),
            "crosstalk_support_audit": repo_rel(crosstalk_support_dir / "camera_e2e_crosstalk_support_audit.json"),
            "crosstalk_batch_priority": repo_rel(crosstalk_batch_priority_dir / "camera_e2e_crosstalk_batch_priority.json"),
            "crosstalk_execution_pack": repo_rel(crosstalk_execution_pack_dir / "camera_e2e_crosstalk_execution_pack.json"),
            "product_closure_summary": repo_rel(product_closure_summary_dir / "camera_e2e_product_closure_summary.json"),
            "usage_policy": repo_rel(usage_policy_dir / "camera_e2e_usage_policy.json"),
            "adapter_examples": repo_rel(adapter_examples_dir / "camera_e2e_adapter_examples.json"),
            "adapter_smoke": repo_rel(adapter_smoke_dir / "camera_e2e_adapter_smoke.json"),
            "objective_trace": repo_rel(objective_trace_dir / "camera_e2e_objective_trace.json"),
            "handoff_manifest": repo_rel(handoff_dir / "camera_e2e_handoff_manifest.json"),
            "handoff_loader_validation": repo_rel(handoff_loader_dir / "camera_e2e_handoff_loader_validation.json"),
            "objective_acceptance": repo_rel(objective_acceptance_dir / "camera_e2e_objective_acceptance.json"),
            "sensor_probe": repo_rel(sensor_probe_dir / "camera_e2e_sensor_probe.json"),
            "sensor_probe_all_sensors": repo_rel(sensor_probe_all_dir / "camera_e2e_sensor_probe.json"),
            "research_query": repo_rel(research_query_dir / "camera_e2e_runtime_query.json"),
            "product_strict_probe": repo_rel(product_probe_dir / "camera_e2e_runtime_query.json"),
            "closure_plan": repo_rel(closure_dir / "camera_e2e_closure_plan.json"),
        },
        "outputs": {
            "json": repo_rel(report_json),
            "steps_csv": repo_rel(steps_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(steps_csv, steps, STEP_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload)
    update_package_links(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-map-csv", type=Path, default=None)
    parser.add_argument("--slugs", default="")
    parser.add_argument("--query-slugs", default=DEFAULT_PREFER_SLUG)
    parser.add_argument("--query-field-x", default="-0.25,0,0.35")
    parser.add_argument("--query-field-z", default="0,0.4")
    parser.add_argument("--query-wavelength-nm", default="550")
    parser.add_argument("--combined-field-x", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--combined-field-z", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--combined-wavelength-nm", default="450,550,620")
    parser.add_argument("--prefer-slug", default="")
    parser.add_argument("--max-solver-points", type=int, default=16)
    parser.add_argument("--probe-incident-photons", type=float, default=8000.0)
    parser.add_argument("--probe-exposure-s", type=float, default=0.01)
    parser.add_argument("--probe-temperature-c", type=float, default=25.0)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true", help="Skip base package rebuild and only rerun exports, queries, audit, bundle, and closure plan.")
    parser.add_argument("--expect-product-blocked", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    payload = build_pipeline(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "product_lut_ready": payload["product_lut_ready"],
                "runtime_row_count": payload["runtime_row_count"],
                "kernel_row_count": payload["kernel_row_count"],
                "fdtd_stack_cfa_sync_status": payload["fdtd_stack_cfa_sync_status"],
                "fdtd_stack_cfa_sync_updated_count": payload["fdtd_stack_cfa_sync_updated_count"],
                "fdtd_stack_cfa_sync_fail_count": payload["fdtd_stack_cfa_sync_fail_count"],
                "runtime_research_allowed_query_count": payload["runtime_research_allowed_query_count"],
                "consumer_query_allowed_count": payload["consumer_query_allowed_count"],
                "consumer_bundle_requirement_count": payload["consumer_bundle_requirement_count"],
                "consumer_bundle_requirement_load_map_count": payload["consumer_bundle_requirement_load_map_count"],
                "flat_sensor_bundle_status": payload["flat_sensor_bundle_status"],
                "flat_sensor_bundle_sensor_count": payload["flat_sensor_bundle_sensor_count"],
                "import_contract_status": payload["import_contract_status"],
                "import_contract_requirement_row_count": payload["import_contract_requirement_row_count"],
                "import_contract_pointer_resolved_count": payload["import_contract_pointer_resolved_count"],
                "canonical_payload_status": payload["canonical_payload_status"],
                "canonical_payload_requirement_row_count": payload["canonical_payload_requirement_row_count"],
                "canonical_payload_pointer_resolved_count": payload["canonical_payload_pointer_resolved_count"],
                "flat_sensor_query_allowed_count": payload["flat_sensor_query_allowed_count"],
                "analysis_report_check_channel_row_count": payload["analysis_report_check_channel_row_count"],
                "usage_policy_status": payload["usage_policy_status"],
                "usage_policy_strict_product_filter_row_count": payload["usage_policy_strict_product_filter_row_count"],
                "adapter_examples_status": payload["adapter_examples_status"],
                "adapter_examples_file_count": payload["adapter_examples_file_count"],
                "adapter_smoke_status": payload["adapter_smoke_status"],
                "adapter_smoke_total_product_allowed_query_count": payload["adapter_smoke_total_product_allowed_query_count"],
                "objective_trace_status": payload["objective_trace_status"],
                "objective_trace_row_count": payload["objective_trace_row_count"],
                "product_strict_status": payload["product_strict_status"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not payload["validation"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
