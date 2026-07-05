#!/usr/bin/env python3
"""Audit CameraE2E LUT artifacts for research and production readiness.

This audit intentionally separates "internally valid research/trend package"
from "product accuracy LUT". Product readiness is strict: measured/calibrated
CRA input, measured stack/material, complete quantitative field coverage, and
converged finite-array crosstalk must all pass.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_readiness_audit"
PRODUCT_CRA_GATES = {"PASS"}
PRODUCT_STACK_GATES = {"PASS"}
PRODUCT_FIELD_GATES = {"PASS"}
PRODUCT_CROSSTALK_GATES = {"PASS"}
PASS_GATE_VALUES = {"PASS", "TRUE", "1", "YES"}

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "pixel_architecture",
    "ocl_mode_guess",
    "expected_field_rows",
    "field_rows",
    "field_pass_rows",
    "field_check_rows",
    "field_fail_rows",
    "cra_input_gate",
    "measured_stack_gate",
    "quantitative_field_gate",
    "finite_array_crosstalk_gate",
    "compact_crosstalk_gate",
    "combined_query_rows",
    "expected_combined_query_rows",
    "combined_query_gate",
    "research_ingest_gate",
    "production_lut_gate",
    "research_ingest_allowed",
    "production_ingest_allowed",
    "primary_blockers",
    "next_actions",
]

ISSUE_COLUMNS = [
    "severity",
    "scope",
    "slug",
    "code",
    "issue_code",
    "gate",
    "detail",
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
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def gate(value: Any, default: str = "MISSING") -> str:
    text = str(value if value is not None else "").strip().upper()
    return text or default


def boolish(value: Any) -> bool:
    return gate(value, "") in PASS_GATE_VALUES


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def combine_gate(values: list[str]) -> str:
    gates = {gate(value) for value in values}
    if "FAIL" in gates:
        return "FAIL"
    if "MISSING" in gates:
        return "MISSING"
    if "CHECK" in gates or "ASSUMED" in gates or "UNSUPPORTED" in gates or "BLOCKED" in gates:
        return "CHECK"
    if gates == {"PASS"}:
        return "PASS"
    return "CHECK"


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


def group_by(rows: list[dict[str, Any]], column: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(column, ""))].append(row)
    return groups


def latest_combined_json(package_dir: Path, package: dict[str, Any]) -> Path:
    latest = package.get("latest_camera_e2e_combined_query", {})
    if isinstance(latest, dict) and latest.get("json"):
        path = ROOT / latest["json"]
        if path.exists():
            return path
    fallback = package_dir / "camera_e2e_combined_query_all_sensors_5x5_no_fail" / "camera_e2e_combined_query.json"
    return fallback


def compact_kernel_normalization(rows: list[dict[str, str]], tolerance: float) -> tuple[dict[str, str], int, int]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    gates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                row.get("slug", ""),
                row.get("field_case", ""),
                row.get("wavelength_nm", ""),
                row.get("color_channel", ""),
            ]
        )
        sums[key] += finite_float(row.get("response_fraction"), 0.0)
        counts[key] += 1
        gates[row.get("slug", "")].append(gate(row.get("evidence_gate")))
    bad_count = sum(1 for value in sums.values() if abs(value - 1.0) > tolerance)
    slug_gate = {slug: combine_gate(values) for slug, values in gates.items()}
    return slug_gate, len(sums), bad_count


def issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    scope: str,
    slug: str = "",
    code: str = "",
    issue_code: str,
    gate_value: str,
    detail: str,
    next_action: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "scope": scope,
            "slug": slug,
            "code": code,
            "issue_code": issue_code,
            "gate": gate_value,
            "detail": detail,
            "next_action": next_action,
        }
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package_json_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json_path)
    sensor_rows = read_csv(package_dir / "camera_e2e_sensor_index.csv")
    field_design_rows = read_csv(package_dir / "camera_e2e_field_design_cases.csv")
    quantitative_queue_rows = read_csv(package_dir / "camera_e2e_quantitative_point_queue.csv")
    field_rows = read_csv(package_dir / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.csv")
    field_payload = read_json(package_dir / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.json")
    crosstalk_status_rows = read_csv(package_dir / "camera_e2e_ingest_export" / "camera_e2e_crosstalk_status_lut.csv")
    compact_kernel_rows = read_csv(package_dir / "camera_e2e_compact_crosstalk_lut" / "camera_e2e_compact_crosstalk_kernel_lut.csv")
    compact_report = read_json(package_dir / "camera_e2e_compact_crosstalk_lut" / "camera_e2e_compact_crosstalk_report.json")
    manifest = read_json(package_dir / "camera_e2e_ingest_export" / "camera_e2e_ingest_manifest.json")
    combined_json_path = latest_combined_json(package_dir, package)
    combined_payload = read_json(combined_json_path)
    combined_rows = combined_payload.get("rows", []) if isinstance(combined_payload.get("rows"), list) else []

    field_by_slug = group_by(field_rows, "slug")
    design_by_slug = group_by(field_design_rows, "slug")
    field_queue_required_by_slug = Counter(
        row.get("slug", "")
        for row in quantitative_queue_rows
        if row.get("solver") == "field" and row.get("slug")
    )
    crosstalk_by_slug = group_by(crosstalk_status_rows, "slug")
    combined_by_slug = group_by(combined_rows, "slug")
    compact_gate_by_slug, compact_kernel_count, compact_bad_sum_count = compact_kernel_normalization(
        compact_kernel_rows, args.tolerance
    )

    query_cfg = combined_payload.get("query", {}) if isinstance(combined_payload.get("query"), dict) else {}
    query_x_count = len(query_cfg.get("field_x_norm", [])) if isinstance(query_cfg.get("field_x_norm"), list) else 0
    query_z_count = len(query_cfg.get("field_z_norm", [])) if isinstance(query_cfg.get("field_z_norm"), list) else 0

    summary_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        code = sensor.get("code", "")
        s_field = field_by_slug.get(slug, [])
        s_design = design_by_slug.get(slug, [])
        s_xt = crosstalk_by_slug.get(slug, [])
        s_query = combined_by_slug.get(slug, [])
        wavelengths = sorted({row.get("wavelength_nm", "") for row in s_field if row.get("wavelength_nm", "")})
        expected_field_rows = max(len(s_design) * len(wavelengths), field_queue_required_by_slug.get(slug, 0))
        expected_query_rows = query_x_count * query_z_count * len(wavelengths)

        field_gate_counts = Counter(gate(row.get("evidence_gate")) for row in s_field)
        field_pass = field_gate_counts.get("PASS", 0)
        field_check = field_gate_counts.get("CHECK", 0) + field_gate_counts.get("MISSING", 0)
        field_fail = field_gate_counts.get("FAIL", 0)
        cra_gate = combine_gate([row.get("cra_input_gate", "") for row in s_field]) if s_field else "MISSING"
        stack_gate = gate(sensor.get("measured_stack_gate"))
        quantitative_field_gate = "PASS" if expected_field_rows > 0 and field_pass == expected_field_rows else combine_gate(
            [row.get("evidence_gate", "") for row in s_field]
        )
        finite_xt_gate = combine_gate([row.get("evidence_gate", "") for row in s_xt]) if s_xt else "MISSING"
        compact_gate = compact_gate_by_slug.get(slug, "MISSING")
        combined_gate = combine_gate([row.get("combined_evidence_gate", "") for row in s_query]) if s_query else "MISSING"

        blockers: list[str] = []
        actions: list[str] = []
        if cra_gate not in PRODUCT_CRA_GATES:
            blockers.append("measured CRA/ML-shift missing")
            actions.append("import measured/calibrated camera_module_field_map.csv")
            issue(
                issues,
                severity="blocker",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="cra_input_not_measured",
                gate_value=cra_gate,
                detail="CRA and microlens shift are assumed design priors, not measured/calibrated module data.",
                next_action="Provide camera_module_field_map.csv with MEASURED, CALIBRATED, or RAYTRACE_VALIDATED gate.",
            )
        if stack_gate not in PRODUCT_STACK_GATES:
            blockers.append("measured stack/n,k missing")
            actions.append("import measured stack geometry and measured n,k")
            issue(
                issues,
                severity="blocker",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="stack_material_not_measured",
                gate_value=stack_gate,
                detail="Measured stack geometry and wavelength-dependent n,k are not available.",
                next_action="Replace proxy stack/material with measured teardown/ellipsometry tables.",
            )
        if expected_field_rows == 0 or field_pass != expected_field_rows:
            blockers.append(f"field FDTD coverage {field_pass}/{expected_field_rows}")
            actions.append("run quantitative Meep field sweep for all field/color anchors")
            issue(
                issues,
                severity="blocker",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="field_fdtd_coverage_incomplete",
                gate_value=quantitative_field_gate,
                detail=f"Quantitative PASS field rows are {field_pass}/{expected_field_rows}; FAIL rows={field_fail}.",
                next_action="Run missing/failed field cases until convergence PASS, then re-export ingest LUTs.",
            )
        if finite_xt_gate not in PRODUCT_CROSSTALK_GATES:
            blockers.append("finite-array crosstalk not converged")
            actions.append("run finite-array Meep crosstalk convergence sweep")
            issue(
                issues,
                severity="blocker",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="finite_array_crosstalk_not_pass",
                gate_value=finite_xt_gate,
                detail="No converged finite-array crosstalk kernel is available for production use.",
                next_action="Run crosstalk sweeps with multi-resolution convergence and sufficient guard pixels.",
            )
        if compact_gate != "PASS" or compact_bad_sum_count:
            issue(
                issues,
                severity="warning" if compact_bad_sum_count == 0 else "blocker",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="compact_crosstalk_surrogate",
                gate_value=compact_gate,
                detail="Compact crosstalk is normalized but remains a CHECK-gated surrogate unless finite-array calibrated.",
                next_action="Use compact kernels only for research/trend until calibrated against finite-array FDTD or measurement.",
            )
        if expected_query_rows and len(s_query) < expected_query_rows:
            issue(
                issues,
                severity="warning",
                scope="sensor",
                slug=slug,
                code=code,
                issue_code="combined_query_rows_excluded",
                gate_value=combined_gate,
                detail=f"Combined query has {len(s_query)}/{expected_query_rows} rows after FAIL exclusion.",
                next_action="Fix failed field rows and regenerate combined query without excluding required field points.",
            )

        research_gate_inputs = [combined_gate, compact_gate]
        if compact_bad_sum_count:
            research_gate = "FAIL"
        elif expected_query_rows and len(s_query) < expected_query_rows:
            research_gate = "CHECK"
        else:
            research_gate = combine_gate(research_gate_inputs)
        production_gate = "PASS" if not blockers else "FAIL"

        summary_rows.append(
            {
                "slug": slug,
                "code": code,
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
                "pixel_architecture": sensor.get("pixel_architecture", ""),
                "ocl_mode_guess": sensor.get("ocl_mode_guess", ""),
                "expected_field_rows": expected_field_rows,
                "field_rows": len(s_field),
                "field_pass_rows": field_pass,
                "field_check_rows": field_check,
                "field_fail_rows": field_fail,
                "cra_input_gate": cra_gate,
                "measured_stack_gate": stack_gate,
                "quantitative_field_gate": quantitative_field_gate,
                "finite_array_crosstalk_gate": finite_xt_gate,
                "compact_crosstalk_gate": compact_gate,
                "combined_query_rows": len(s_query),
                "expected_combined_query_rows": expected_query_rows,
                "combined_query_gate": combined_gate,
                "research_ingest_gate": research_gate,
                "production_lut_gate": production_gate,
                "research_ingest_allowed": research_gate != "FAIL",
                "production_ingest_allowed": production_gate == "PASS",
                "primary_blockers": "; ".join(blockers),
                "next_actions": "; ".join(dict.fromkeys(actions)),
            }
        )

    if compact_bad_sum_count:
        issue(
            issues,
            severity="blocker",
            scope="package",
            issue_code="compact_kernel_normalization_failed",
            gate_value="FAIL",
            detail=f"{compact_bad_sum_count} compact crosstalk kernels do not sum to one within tolerance.",
            next_action="Regenerate compact crosstalk kernels and validate before any CameraE2E ingest.",
        )
    if not field_payload.get("npz_validation", {}).get("pass", False):
        issue(
            issues,
            severity="blocker",
            scope="package",
            issue_code="field_lut_npz_validation_failed",
            gate_value="FAIL",
            detail="Field LUT NPZ validation is missing or failed.",
            next_action="Re-run export_camera_e2e_ingest_luts.py and inspect NPZ validation issues.",
        )
    if not compact_report.get("npz_validation", {}).get("pass", False):
        issue(
            issues,
            severity="blocker",
            scope="package",
            issue_code="compact_crosstalk_npz_validation_failed",
            gate_value="FAIL",
            detail="Compact crosstalk NPZ validation is missing or failed.",
            next_action="Re-run build_camera_e2e_compact_crosstalk_lut.py and inspect NPZ validation issues.",
        )
    if manifest.get("product_lut_ready") not in {False, "False", "false", 0, "0"}:
        issue(
            issues,
            severity="blocker",
            scope="package",
            issue_code="manifest_product_ready_unexpected_true",
            gate_value="FAIL",
            detail="Manifest product_lut_ready is true despite strict readiness blockers.",
            next_action="Keep product_lut_ready false until all sensor-level production gates pass.",
        )

    production_gate_counts = Counter(row["production_lut_gate"] for row in summary_rows)
    research_gate_counts = Counter(row["research_ingest_gate"] for row in summary_rows)
    product_ready = bool(summary_rows) and all(row["production_lut_gate"] == "PASS" for row in summary_rows)
    research_valid = bool(summary_rows) and all(row["research_ingest_gate"] != "FAIL" for row in summary_rows)

    summary_csv = output_dir / "camera_e2e_lut_readiness_summary.csv"
    issues_csv = output_dir / "camera_e2e_lut_readiness_issues.csv"
    report_json = output_dir / "camera_e2e_lut_readiness_report.json"
    html_path = output_dir / "index.html"
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_csv(issues_csv, issues, ISSUE_COLUMNS)

    payload = {
        "schema": "camera_e2e_lut_readiness_audit_v1",
        "artifact_role": "camera_e2e_lut_product_and_research_gate_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "source_package_json": repo_rel(package_json_path),
        "source_field_lut_json": repo_rel(package_dir / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.json"),
        "source_combined_query_json": repo_rel(combined_json_path),
        "sensor_count": len(summary_rows),
        "product_lut_ready": product_ready,
        "research_ingest_valid": research_valid,
        "production_gate_counts": dict(production_gate_counts),
        "research_gate_counts": dict(research_gate_counts),
        "issue_counts": dict(Counter(row["severity"] for row in issues)),
        "compact_kernel_count": compact_kernel_count,
        "compact_kernel_normalization_bad_count": compact_bad_sum_count,
        "field_lut_npz_validation_pass": bool(field_payload.get("npz_validation", {}).get("pass", False)),
        "compact_crosstalk_npz_validation_pass": bool(compact_report.get("npz_validation", {}).get("pass", False)),
        "policy": {
            "production_requires": [
                "cra_input_gate PASS from measured/calibrated/raytrace-validated field map",
                "measured_stack_gate PASS",
                "all field/color anchors quantitative FDTD PASS",
                "finite-array crosstalk convergence PASS",
                "product_lut_ready true only after every strict gate passes",
            ],
            "research_ingest_note": "Rows may be used for CameraE2E trend experiments only when downstream preserves row gates and product_lut_ready=false.",
        },
        "rows": summary_rows,
        "issues": issues,
        "outputs": {
            "json": repo_rel(report_json),
            "summary_csv": repo_rel(summary_csv),
            "issues_csv": repo_rel(issues_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(report_json, payload)
    write_html(html_path, payload)
    update_package_links(package_json_path, payload)
    return payload


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    issues = payload["issues"]
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}
h1{font-size:30px;margin:0 0 6px} h2{margin-top:26px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.pass{color:#82f09d}.check{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22;position:sticky;top:0}.pill{display:inline-block;border:1px solid #2d6276;border-radius:999px;padding:3px 8px}
code{color:#9fe8ff}
"""
    product_class = "pass" if payload["product_lut_ready"] else "fail"
    research_class = "pass" if payload["research_ingest_valid"] else "fail"
    issue_counts = payload.get("issue_counts", {})
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E LUT Readiness Audit</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E LUT Readiness Audit</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This report gates research/trend ingest separately from product accuracy LUT use.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric {research_class}">{html_cell(payload.get("research_ingest_valid"))}</div><div class="muted">research ingest valid</div></div>
    <div class="card"><div class="metric {product_class}">{html_cell(payload.get("product_lut_ready"))}</div><div class="muted">product LUT ready</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("compact_kernel_count", 0))}</div><div class="muted">compact kernels</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("compact_kernel_normalization_bad_count", 0))}</div><div class="muted">bad kernel sums</div></div>
  </div>
  <p><span class="pill">Production gates: {html_cell(payload.get("production_gate_counts", {}))}</span>
  <span class="pill">Research gates: {html_cell(payload.get("research_gate_counts", {}))}</span>
  <span class="pill">Issues: {html_cell(issue_counts)}</span></p>
  <h2>Sensor Readiness</h2>
  {html_table(rows, SUMMARY_COLUMNS, limit=100)}
  <h2>Issues</h2>
  {html_table(issues, ISSUE_COLUMNS, limit=300)}
  <h2>Policy</h2>
  <p class="muted">Product use requires measured/calibrated CRA and microlens shift, measured stack/n,k, full quantitative FDTD field coverage, and finite-array crosstalk convergence. Current research rows must keep <code>product_lut_ready=false</code>.</p>
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_package_links(package_json_path: Path, payload: dict[str, Any]) -> None:
    package = read_json(package_json_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_readiness_audit_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_readiness_audit_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_readiness_audit_issues_csv"] = payload["outputs"]["issues_csv"]
    outputs["camera_e2e_readiness_audit_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_readiness_audit"] = {
        "schema": payload["schema"],
        "product_lut_ready": payload["product_lut_ready"],
        "research_ingest_valid": payload["research_ingest_valid"],
        "production_gate_counts": payload["production_gate_counts"],
        "research_gate_counts": payload["research_gate_counts"],
        "issue_counts": payload["issue_counts"],
        **payload["outputs"],
    }
    package_json_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


def main() -> None:
    payload = audit(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "sensor_count": payload["sensor_count"],
                "research_ingest_valid": payload["research_ingest_valid"],
                "product_lut_ready": payload["product_lut_ready"],
                "production_gate_counts": payload["production_gate_counts"],
                "research_gate_counts": payload["research_gate_counts"],
                "issue_counts": payload["issue_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
