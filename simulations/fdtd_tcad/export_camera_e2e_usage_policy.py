#!/usr/bin/env python3
"""Export CameraE2E ingest policy from LUT trust, mesh, and closure gates.

This artifact is intentionally separate from the LUT values. It tells a
CameraE2E loader which modes are allowed, which row filters are safe for
research, and why product mode must remain closed.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_usage_policy"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_profile",
    "research_ingest_allowed",
    "product_ingest_allowed",
    "allowed_camera_e2e_modes",
    "blocked_camera_e2e_modes",
    "recommended_runtime_filter_id",
    "recommended_bundle",
    "camera_e2e_use_scope",
    "camera_e2e_allowed_use",
    "trust_class",
    "mesh_confidence_class",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "runtime_row_count",
    "kernel_row_count",
    "flat_bundle_gate",
    "product_closure_status",
    "first_product_closure_action",
    "first_product_closure_local_feasibility",
    "policy_reason",
    "required_before_product_use",
]

DOMAIN_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "domain_profile",
    "research_ingest_allowed",
    "product_ingest_allowed",
    "domain_use_scope",
    "allowed_use",
    "blocked_use",
    "trust_class",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "requirement_count",
    "research_gate_counts",
    "product_gate_counts",
    "recommended_next_action",
]

FILTER_COLUMNS = [
    "filter_id",
    "target_table",
    "mode",
    "filter_expression",
    "row_count",
    "product_usable",
    "expected_behavior",
    "notes",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key)}


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def profile_for_scope(use_scope: str, cfa_class: str = "") -> str:
    text = f"{use_scope} {cfa_class}".upper()
    if "PRODUCT" in text and "BLOCKED" not in text:
        return "PRODUCT_READY"
    if "CFA_UNKNOWN" in text:
        return "SCHEMA_PLUMBING_CFA_UNKNOWN"
    if "RESEARCH_TREND" in text:
        return "RESEARCH_TREND"
    if "SINGLE_ANCHOR" in text or "ANCHOR" in text:
        return "RESEARCH_ANCHOR_OR_SMOKE"
    return "SCHEMA_PLUMBING_ONLY"


def domain_profile(domain_use_scope: str, trust_class: str) -> str:
    text = f"{domain_use_scope} {trust_class}".upper()
    if "PRODUCT" in text and "BLOCKED" not in text:
        return "PRODUCT_READY"
    if "PRIOR" in text or "SPARSE" in text or "PROXY" in text:
        return "RESEARCH_OR_PRIOR_ONLY"
    return "RESEARCH_CHECK"


def build_filter_rows(runtime_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    research_runtime_count = sum(1 for row in runtime_rows if boolish(row.get("research_ingest_allowed")))
    product_runtime_count = sum(
        1
        for row in runtime_rows
        if boolish(row.get("production_ingest_allowed")) and boolish(row.get("product_lut_ready"))
    )
    runtime_check_count = sum(1 for row in runtime_rows if str(row.get("combined_evidence_gate", "")).upper() == "CHECK")
    runtime_pass_count = sum(1 for row in runtime_rows if str(row.get("combined_evidence_gate", "")).upper() == "PASS")
    runtime_fail_count = sum(1 for row in runtime_rows if str(row.get("combined_evidence_gate", "")).upper() == "FAIL")
    kernel_research_count = sum(1 for row in kernel_rows if str(row.get("evidence_gate", "")).upper() in {"CHECK", "PASS"})
    kernel_product_count = sum(1 for row in kernel_rows if str(row.get("evidence_gate", "")).upper() == "PASS")
    return [
        {
            "filter_id": "research_runtime_rows",
            "target_table": "camera_e2e_runtime_lut.csv",
            "mode": "research",
            "filter_expression": "research_ingest_allowed == true",
            "row_count": research_runtime_count,
            "product_usable": False,
            "expected_behavior": "Allowed for CameraE2E plumbing, sensitivity, and trend studies only.",
            "notes": "Downstream must preserve evidence gates and product_lut_ready=false.",
        },
        {
            "filter_id": "strict_product_runtime_rows",
            "target_table": "camera_e2e_runtime_lut.csv",
            "mode": "product",
            "filter_expression": "production_ingest_allowed == true AND product_lut_ready == true",
            "row_count": product_runtime_count,
            "product_usable": product_runtime_count > 0,
            "expected_behavior": "Must be zero until measured stack/material/calibration and convergence gates pass.",
            "notes": "This is the only acceptable product LUT filter.",
        },
        {
            "filter_id": "research_runtime_check_rows",
            "target_table": "camera_e2e_runtime_lut.csv",
            "mode": "research",
            "filter_expression": "combined_evidence_gate == CHECK AND research_ingest_allowed == true",
            "row_count": runtime_check_count,
            "product_usable": False,
            "expected_behavior": "Usable for trend studies with uncertainty bands.",
            "notes": f"Runtime PASS rows in current package: {runtime_pass_count}; runtime FAIL rows: {runtime_fail_count}.",
        },
        {
            "filter_id": "research_crosstalk_kernel_rows",
            "target_table": "camera_e2e_runtime_crosstalk_kernel.csv",
            "mode": "research",
            "filter_expression": "evidence_gate in {CHECK, PASS}",
            "row_count": kernel_research_count,
            "product_usable": False,
            "expected_behavior": "Use as prior/proxy optical crosstalk only.",
            "notes": "Finite-array product crosstalk convergence is not complete.",
        },
        {
            "filter_id": "strict_product_crosstalk_kernel_rows",
            "target_table": "camera_e2e_runtime_crosstalk_kernel.csv",
            "mode": "product",
            "filter_expression": "evidence_gate == PASS AND product_crosstalk_ready == true",
            "row_count": 0,
            "product_usable": False,
            "expected_behavior": "Must stay zero until the crosstalk support/product convergence gate is open.",
            "notes": f"Raw kernel PASS-tag rows before product support gate: {kernel_product_count}.",
        },
    ]


def build_policy(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    use_scope = read_json(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_summary.json")
    trust = read_json(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_assessment.json")
    runtime_bundle = read_json(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_bundle.json")
    flat_bundle = read_json(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json")
    product_closure = read_json(package_dir / "camera_e2e_product_closure_summary" / "camera_e2e_product_closure_summary.json")

    use_sensor_rows = read_csv_rows(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_sensor.csv")
    use_domain_rows = read_csv_rows(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_domain.csv")
    trust_sensor_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv")
    trust_domain_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv")
    mesh_rows = read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv")
    flat_index_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv")
    product_closure_rows = read_csv_rows(package_dir / "camera_e2e_product_closure_summary" / "camera_e2e_product_closure_by_sensor.csv")
    runtime_rows = read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv")
    kernel_rows = read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv")

    trust_by_slug = index_by(trust_sensor_rows, "slug")
    mesh_by_slug = index_by(mesh_rows, "slug")
    flat_by_slug = index_by(flat_index_rows, "slug")
    closure_by_slug = index_by(product_closure_rows, "slug")
    runtime_by_slug = group_by(runtime_rows, "slug")
    kernel_by_slug = group_by(kernel_rows, "slug")
    trust_domain_by_key = {(row.get("slug", ""), row.get("domain", "")): row for row in trust_domain_rows}

    sensor_rows: list[dict[str, Any]] = []
    for row in use_sensor_rows:
        slug = row.get("slug", "")
        trust_row = trust_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        flat = flat_by_slug.get(slug, {})
        closure = closure_by_slug.get(slug, {})
        product_ready = boolish(row.get("product_ready"))
        profile = profile_for_scope(row.get("camera_e2e_use_scope", ""), row.get("cfa_provenance_class", ""))
        if profile == "RESEARCH_TREND":
            recommended_filter = "research_runtime_check_rows"
        elif profile == "SCHEMA_PLUMBING_CFA_UNKNOWN":
            recommended_filter = "research_runtime_rows"
        elif profile == "RESEARCH_ANCHOR_OR_SMOKE":
            recommended_filter = "research_runtime_rows"
        else:
            recommended_filter = "research_runtime_rows"
        blocked_modes = [
            "product_lut",
            "module_signoff",
            "crosstalk_correction_kernel",
            "calibrated_color_shading",
        ]
        allowed_modes = ["schema_loading", "pipeline_smoke"]
        if profile in {"RESEARCH_TREND", "RESEARCH_ANCHOR_OR_SMOKE"}:
            allowed_modes.append("research_trend")
        sensor_rows.append(
            {
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "camera_e2e_profile": profile,
                "research_ingest_allowed": True,
                "product_ingest_allowed": product_ready,
                "allowed_camera_e2e_modes": ";".join(allowed_modes),
                "blocked_camera_e2e_modes": ";".join(blocked_modes if not product_ready else []),
                "recommended_runtime_filter_id": recommended_filter,
                "recommended_bundle": "camera_e2e_flat_sensor_bundle for per-sensor load; camera_e2e_runtime_bundle for vectorized field/color rows",
                "camera_e2e_use_scope": row.get("camera_e2e_use_scope", ""),
                "camera_e2e_allowed_use": row.get("camera_e2e_allowed_use", ""),
                "trust_class": trust_row.get("trust_class", row.get("trust_class", "")),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", row.get("mesh_confidence_class", "")),
                "evidence_confidence_score_0_100": trust_row.get("evidence_confidence_score_0_100", ""),
                "product_calibration_score_0_100": trust_row.get("product_calibration_score_0_100", ""),
                "field_mesh_pass_points": mesh.get("field_pass_points", row.get("field_mesh_pass_points", "")),
                "field_mesh_required_points": mesh.get("field_required_points", row.get("field_mesh_required_points", "")),
                "crosstalk_mesh_pass_points": mesh.get("crosstalk_pass_points", row.get("crosstalk_mesh_pass_points", "")),
                "crosstalk_mesh_required_points": mesh.get("crosstalk_required_points", row.get("crosstalk_mesh_required_points", "")),
                "runtime_row_count": len(runtime_by_slug.get(slug, [])),
                "kernel_row_count": len(kernel_by_slug.get(slug, [])),
                "flat_bundle_gate": flat.get("loader_gate", ""),
                "product_closure_status": closure.get("product_closure_status", ""),
                "first_product_closure_action": closure.get("first_action_type", ""),
                "first_product_closure_local_feasibility": closure.get("first_action_local_feasibility", ""),
                "policy_reason": row.get("camera_e2e_allowed_use", ""),
                "required_before_product_use": row.get("required_before_product_use", ""),
            }
        )

    domain_rows: list[dict[str, Any]] = []
    for row in use_domain_rows:
        trust_row = trust_domain_by_key.get((row.get("slug", ""), row.get("domain", "")), {})
        product_ready = str(row.get("product_gate", "")).upper() == "PASS" and safe_float(row.get("product_calibration_score_0_100")) >= 100.0
        domain_rows.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "domain": row.get("domain", ""),
                "domain_profile": domain_profile(row.get("domain_use_scope", ""), trust_row.get("trust_class", "")),
                "research_ingest_allowed": True,
                "product_ingest_allowed": product_ready,
                "domain_use_scope": row.get("domain_use_scope", ""),
                "allowed_use": row.get("allowed_use", trust_row.get("camera_e2e_allowed_use", "")),
                "blocked_use": row.get("blocked_use", ""),
                "trust_class": trust_row.get("trust_class", ""),
                "evidence_confidence_score_0_100": row.get("evidence_confidence_score_0_100", trust_row.get("evidence_confidence_score_0_100", "")),
                "product_calibration_score_0_100": row.get("product_calibration_score_0_100", trust_row.get("product_calibration_score_0_100", "")),
                "requirement_count": row.get("requirement_count", ""),
                "research_gate_counts": row.get("research_gate_counts", ""),
                "product_gate_counts": row.get("product_gate_counts", ""),
                "recommended_next_action": row.get("recommended_next_action", trust_row.get("recommended_next_action", "")),
            }
        )

    filter_rows = build_filter_rows(runtime_rows, kernel_rows)
    product_filter_rows = [row for row in filter_rows if row["filter_id"].startswith("strict_product")]
    product_filter_row_count = sum(safe_int(row.get("row_count")) for row in product_filter_rows)
    product_ingest_allowed_count = sum(1 for row in sensor_rows if boolish(row.get("product_ingest_allowed")))
    profile_counts = dict(sorted(Counter(str(row.get("camera_e2e_profile", "")) for row in sensor_rows).items()))
    checks = [
        check_row(
            "sensor_policy_rows_present",
            len(sensor_rows) > 0,
            "PASS" if sensor_rows else "FAIL",
            {"sensor_policy_row_count": len(sensor_rows)},
            "Generate use-scope, trust, mesh, flat-bundle, and product-closure inputs.",
        ),
        check_row(
            "domain_policy_rows_present",
            len(domain_rows) > 0,
            "PASS" if domain_rows else "FAIL",
            {"domain_policy_row_count": len(domain_rows)},
            "Generate use-scope and trust domain rows.",
        ),
        check_row(
            "research_filter_rows_present",
            safe_int(next((row.get("row_count") for row in filter_rows if row["filter_id"] == "research_runtime_rows"), 0)) > 0,
            "PASS",
            {"runtime_row_count": len(runtime_rows), "kernel_row_count": len(kernel_rows)},
            "Regenerate runtime bundle before usage policy.",
        ),
        check_row(
            "product_filters_closed",
            product_filter_row_count == 0 and product_ingest_allowed_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_filter_row_count == 0 and product_ingest_allowed_count == 0 else "FAIL",
            {"product_filter_row_count": product_filter_row_count, "product_ingest_allowed_count": product_ingest_allowed_count},
            "Keep product filters closed until product_lut_ready and all product convergence/calibration gates pass.",
        ),
        check_row(
            "upstream_policy_inputs_valid",
            bool(use_scope.get("validation", {}).get("pass"))
            and bool(trust.get("validation", {}).get("pass"))
            and bool(runtime_bundle.get("validation", {}).get("pass"))
            and bool(flat_bundle.get("validation", {}).get("pass"))
            and bool(product_closure.get("validation", {}).get("pass")),
            "PASS",
            {
                "use_scope": use_scope.get("validation", {}).get("status"),
                "trust": trust.get("validation", {}).get("status"),
                "runtime": runtime_bundle.get("validation", {}).get("status"),
                "flat_bundle": flat_bundle.get("validation", {}).get("status"),
                "product_closure": product_closure.get("validation", {}).get("status"),
            },
            "Regenerate upstream policy inputs.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_USAGE_POLICY_READY_PRODUCT_BLOCKED" if error_count == 0 and product_ingest_allowed_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_csv = output_dir / "camera_e2e_usage_policy_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_usage_policy_by_domain.csv"
    filters_csv = output_dir / "camera_e2e_usage_policy_runtime_filters.csv"
    checks_csv = output_dir / "camera_e2e_usage_policy_checks.csv"
    json_path = output_dir / "camera_e2e_usage_policy.json"
    html_path = output_dir / "index.html"

    payload = {
        "schema": "camera_e2e_usage_policy_v1",
        "artifact_role": "camera_e2e_ingest_usage_policy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_policy_row_count": len(sensor_rows),
        "domain_policy_row_count": len(domain_rows),
        "runtime_filter_row_count": len(filter_rows),
        "runtime_row_count": len(runtime_rows),
        "kernel_row_count": len(kernel_rows),
        "profile_counts": profile_counts,
        "product_ingest_allowed_count": product_ingest_allowed_count,
        "strict_product_filter_row_count": product_filter_row_count,
        "validation": {
            "schema": "camera_e2e_usage_policy_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "loader_contract": {
            "research": "Use camera_e2e_usage_policy_runtime_filters.csv filter_id=research_runtime_rows or the per-sensor recommended filter, and keep evidence gates attached to every row.",
            "product": "Use filter_id=strict_product_runtime_rows only. In this package the expected count is zero.",
            "crosstalk": "Use research_crosstalk_kernel_rows only for proxy/research. Product crosstalk stays closed until finite-array convergence passes.",
        },
        "outputs": {
            "json": repo_rel(json_path),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "filters_csv": repo_rel(filters_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_csv(filters_csv, filter_rows, FILTER_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(json_path, payload)
    write_html(html_path, payload, sensor_rows, domain_rows, filter_rows, checks)
    update_package(package_dir, payload)
    return payload


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


def write_html(
    path: Path,
    payload: dict[str, Any],
    sensor_rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1440px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Usage Policy</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Usage Policy</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This is the loader policy for research/product filters and per-sensor use scope.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">policy status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_policy_row_count", 0))}</div><div class="muted">sensor policies</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("domain_policy_row_count", 0))}</div><div class="muted">domain policies</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("strict_product_filter_row_count", 0))}</div><div class="muted">strict product rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("runtime_row_count", 0))}</div><div class="muted">runtime rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("kernel_row_count", 0))}</div><div class="muted">kernel rows</div></div>
</div>
<h2>Loader Contract</h2>
<p><strong>Research:</strong> {html_cell(payload.get("loader_contract", {}).get("research", ""))}</p>
<p><strong>Product:</strong> {html_cell(payload.get("loader_contract", {}).get("product", ""))}</p>
<p><strong>Crosstalk:</strong> {html_cell(payload.get("loader_contract", {}).get("crosstalk", ""))}</p>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Runtime Filters</h2>{html_table(filter_rows, FILTER_COLUMNS)}
<h2>Sensor Policy</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Domain Policy</h2>{html_table(domain_rows, DOMAIN_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_usage_policy_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_usage_policy_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_usage_policy_by_domain_csv"] = payload["outputs"]["domain_csv"]
    outputs["camera_e2e_usage_policy_runtime_filters_csv"] = payload["outputs"]["filters_csv"]
    outputs["camera_e2e_usage_policy_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_usage_policy_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_usage_policy"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_policy_row_count": payload["sensor_policy_row_count"],
        "domain_policy_row_count": payload["domain_policy_row_count"],
        "strict_product_filter_row_count": payload["strict_product_filter_row_count"],
        "product_ingest_allowed_count": payload["product_ingest_allowed_count"],
        "profile_counts": payload["profile_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_policy(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_policy_row_count": payload["sensor_policy_row_count"],
                "domain_policy_row_count": payload["domain_policy_row_count"],
                "strict_product_filter_row_count": payload["strict_product_filter_row_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not payload["validation"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
