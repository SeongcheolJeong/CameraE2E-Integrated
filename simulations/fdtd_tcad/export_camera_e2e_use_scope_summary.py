#!/usr/bin/env python3
"""Export CameraE2E use-scope decision tables.

This artifact is the consumer-facing routing layer over the larger CameraE2E
package. It answers the practical question for each sensor and domain:

- what can CameraE2E use today;
- what must stay blocked for product accuracy;
- which next action should be run or imported first.

It does not create new physical simulation values. Product use remains blocked
until measured stack/material/CRA/electrical/readout/module data and product
mesh/convergence gates pass.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_use_scope_summary"

DOMAINS = ["Optical / Color", "Pixel / Electrical", "Readout / RAW", "Module Coupling"]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_use_scope",
    "camera_e2e_allowed_use",
    "product_gate",
    "product_ready",
    "trust_class",
    "mesh_confidence_class",
    "field_mesh_pass_fraction",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_fraction",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "optical_color_scope",
    "pixel_electrical_scope",
    "readout_raw_scope",
    "module_coupling_scope",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "cfa_db_row_count",
    "cfa_db_transmission_row_count",
    "crosstalk_support_gate",
    "crosstalk_support_best_neighborhood",
    "crosstalk_support_best_truncation_fraction",
    "first_crosstalk_action_type",
    "first_crosstalk_local_feasibility",
    "first_crosstalk_command",
    "primary_blockers",
    "required_before_product_use",
]

DOMAIN_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "domain_use_scope",
    "allowed_use",
    "blocked_use",
    "product_gate",
    "requirement_count",
    "research_gate_counts",
    "product_gate_counts",
    "row_count_sum",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "primary_blockers",
    "recommended_next_action",
]

ACTION_COLUMNS = [
    "priority_rank",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "action_class",
    "action_type",
    "source",
    "local_feasibility",
    "command_or_input",
    "expected_artifact",
    "why",
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def fraction(numerator: Any, denominator: Any) -> str:
    den = safe_float(denominator)
    if den <= 0:
        return "0.000000"
    return f"{max(0.0, min(1.0, safe_float(numerator) / den)):.6f}"


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            result[value].append(row)
    return dict(result)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "") or "MISSING") for row in rows).items()))


def compact_unique(texts: list[str], limit: int = 8) -> str:
    values: list[str] = []
    for text in texts:
        for item in str(text or "").split(";"):
            clean = item.strip()
            if clean and clean not in values:
                values.append(clean)
    return "; ".join(values[:limit])


def domain_scope(domain_row: dict[str, str], trust_row: dict[str, str]) -> str:
    if str(domain_row.get("product_gate", "")).upper() == "PASS":
        return "PRODUCT_READY"
    domain = domain_row.get("domain", "")
    evidence = safe_float(trust_row.get("evidence_confidence_score_0_100"))
    if domain == "Optical / Color":
        if evidence >= 8:
            return "PARTIAL_OPTICAL_TREND_ONLY"
        return "OPTICAL_PRIOR_OR_PROXY_ONLY"
    if domain == "Pixel / Electrical":
        return "ELECTRICAL_PRIOR_SEED_ONLY"
    if domain == "Readout / RAW":
        return "READOUT_PRIOR_SEED_ONLY"
    if domain == "Module Coupling":
        return "MODULE_PRIOR_FIELD_MAP_ONLY"
    return "RESEARCH_ONLY_PRODUCT_BLOCKED"


def domain_allowed_use(scope: str) -> str:
    if scope == "PRODUCT_READY":
        return "Product CameraE2E use allowed."
    if scope == "PARTIAL_OPTICAL_TREND_ONLY":
        return "Relative field/CRA/OCL trend studies around covered anchors; product mode blocked."
    if scope == "OPTICAL_PRIOR_OR_PROXY_ONLY":
        return "Optical plumbing, coarse sensitivity, and placeholder color/shading studies only."
    if scope == "ELECTRICAL_PRIOR_SEED_ONLY":
        return "Signal/noise pipeline smoke tests with prior CG/FWC/noise values only."
    if scope == "READOUT_PRIOR_SEED_ONLY":
        return "RAW path, gain, ADC, black-level, and mode plumbing tests with prior tables only."
    if scope == "MODULE_PRIOR_FIELD_MAP_ONLY":
        return "Field/CRA/vignetting plumbing with design-prior or placeholder module data only."
    return "Research-only use with explicit gates preserved."


def domain_blocked_use(scope: str) -> str:
    if scope == "PRODUCT_READY":
        return ""
    return "Do not use for calibrated product QE, color, crosstalk, noise, readout, or module shading decisions."


def sensor_scope(consumer: dict[str, str], trust: dict[str, str]) -> str:
    if boolish(consumer.get("product_ready")):
        return "PRODUCT_READY"
    trust_class = trust.get("trust_class", "")
    mesh = consumer.get("mesh_confidence_class", "")
    if trust_class == "PARTIAL_FIELD_TREND_PRODUCT_BLOCKED" or mesh == "MEDIUM_RESEARCH_FIELD_TREND":
        return "CAMERA_E2E_RESEARCH_TREND_ONLY"
    if trust_class == "SPARSE_FIELD_ANCHOR_PRODUCT_BLOCKED" or mesh == "LOW_RESEARCH_ANCHOR":
        return "CAMERA_E2E_SINGLE_ANCHOR_OR_SMOKE_ONLY"
    if consumer.get("cfa_assumption_gate") == "MISSING":
        return "CAMERA_E2E_SCHEMA_PRIOR_ONLY_CFA_UNKNOWN"
    return "CAMERA_E2E_SCHEMA_PRIOR_ONLY"


def required_before_product(scope: str) -> str:
    if scope == "PRODUCT_READY":
        return ""
    return (
        "measured stack geometry and n,k; measured or raytraced CRA/ML-shift map; "
        "product-resolution Meep field and finite-array crosstalk convergence PASS; "
        "measured CG/FWC/dark/DSNU/PRNU/noise/readout tables; module raytrace/vignetting/pupil calibration"
    )


def first_crosstalk_for_slug(rows: list[dict[str, str]], slug: str) -> dict[str, str]:
    slug_rows = [row for row in rows if row.get("slug") == slug]
    if not slug_rows:
        return {}
    return sorted(slug_rows, key=lambda row: safe_int(row.get("priority_rank"), 999999))[0]


def measured_input_action(sensor: dict[str, str], domain: str, blockers: str) -> dict[str, Any]:
    action_by_domain = {
        "Optical / Color": (
            "IMPORT_MEASURED_OPTICAL_STACK_AND_CRA",
            "image_sensor_db measured stack geometry, measured n,k, measured/raytraced CRA and ML/OCL shift maps",
        ),
        "Pixel / Electrical": (
            "IMPORT_ELECTRICAL_CALIBRATION",
            "measured CG/FWC/dark current/DSNU/PRNU/temporal-noise and calibrated TCAD collection targets",
        ),
        "Readout / RAW": (
            "IMPORT_READOUT_RAW_CALIBRATION",
            "measured analog/digital gain, black level, ADC, row/column FPN, timing, defect and mode calibration tables",
        ),
        "Module Coupling": (
            "IMPORT_MODULE_RAYTRACE_AND_POSE",
            "lens raytrace CRA/vignetting/pupil maps plus sensor tilt/decenter and assembly tolerance distributions",
        ),
    }
    action_type, input_hint = action_by_domain.get(domain, ("IMPORT_MEASURED_DATA", "measured calibration data"))
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "domain": domain,
        "action_class": "MEASURED_INPUT_REQUIRED",
        "action_type": action_type,
        "source": "coverage_matrix_product_blocker",
        "local_feasibility": "EXTERNAL_DATA_REQUIRED",
        "command_or_input": input_hint,
        "expected_artifact": "image_sensor_db measured/calibrated source tables",
        "why": compact_unique([blockers], limit=3),
    }


def build_use_scope(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    consumer_rows = read_csv_rows(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv")
    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    trust_sensor_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv")
    trust_domain_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv")
    crosstalk_priority_rows = read_csv_rows(package_dir / "camera_e2e_crosstalk_batch_priority" / "camera_e2e_crosstalk_batch_priority.csv")

    trust_by_slug = index_by(trust_sensor_rows, "slug")
    coverage_by_slug = group_by(coverage_rows, "slug")
    trust_domain_by_key = {(row.get("slug", ""), row.get("domain", "")): row for row in trust_domain_rows}

    sensor_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []

    for consumer in sorted(consumer_rows, key=lambda row: row.get("slug", "")):
        slug = consumer.get("slug", "")
        trust = trust_by_slug.get(slug, {})
        scope = sensor_scope(consumer, trust)
        first_xtalk = first_crosstalk_for_slug(crosstalk_priority_rows, slug)
        product_ready = boolish(consumer.get("product_ready"))
        sensor_row = {
            "slug": slug,
            "code": consumer.get("code", ""),
            "manufacturer": consumer.get("manufacturer", ""),
            "device_name": consumer.get("device_name", ""),
            "camera_e2e_use_scope": scope,
            "camera_e2e_allowed_use": trust.get("camera_e2e_allowed_use", consumer.get("camera_e2e_recommended_use", "")),
            "product_gate": "PASS" if product_ready else "FAIL",
            "product_ready": product_ready,
            "trust_class": trust.get("trust_class", ""),
            "mesh_confidence_class": consumer.get("mesh_confidence_class", ""),
            "field_mesh_pass_fraction": fraction(consumer.get("mesh_field_pass_points"), consumer.get("mesh_field_required_points")),
            "field_mesh_pass_points": consumer.get("mesh_field_pass_points", ""),
            "field_mesh_required_points": consumer.get("mesh_field_required_points", ""),
            "crosstalk_mesh_pass_fraction": fraction(consumer.get("mesh_crosstalk_pass_points"), consumer.get("mesh_crosstalk_required_points")),
            "crosstalk_mesh_pass_points": consumer.get("mesh_crosstalk_pass_points", ""),
            "crosstalk_mesh_required_points": consumer.get("mesh_crosstalk_required_points", ""),
            "optical_color_scope": consumer.get("capability_spectral_qe_scope", ""),
            "pixel_electrical_scope": "ELECTRICAL_PRIOR_SEED_ONLY",
            "readout_raw_scope": "READOUT_PRIOR_SEED_ONLY",
            "module_coupling_scope": "MODULE_PRIOR_FIELD_MAP_ONLY",
            "cfa_provenance_class": consumer.get("cfa_provenance_class", ""),
            "cfa_assumption_gate": consumer.get("cfa_assumption_gate", ""),
            "cfa_db_row_count": consumer.get("cfa_db_row_count", ""),
            "cfa_db_transmission_row_count": consumer.get("cfa_db_transmission_row_count", ""),
            "crosstalk_support_gate": consumer.get("crosstalk_support_gate", ""),
            "crosstalk_support_best_neighborhood": consumer.get("crosstalk_support_best_neighborhood", ""),
            "crosstalk_support_best_truncation_fraction": consumer.get("crosstalk_support_best_truncation_fraction", ""),
            "first_crosstalk_action_type": first_xtalk.get("action_type", ""),
            "first_crosstalk_local_feasibility": first_xtalk.get("local_feasibility", ""),
            "first_crosstalk_command": first_xtalk.get("command", ""),
            "primary_blockers": consumer.get("primary_blockers", trust.get("primary_blockers", "")),
            "required_before_product_use": required_before_product(scope),
        }
        sensor_rows.append(sensor_row)

        if first_xtalk:
            action_rows.append(
                {
                    "slug": slug,
                    "code": consumer.get("code", ""),
                    "manufacturer": consumer.get("manufacturer", ""),
                    "device_name": consumer.get("device_name", ""),
                    "domain": "Optical / Color",
                    "action_class": "SOLVER_BATCH_NEXT",
                    "action_type": first_xtalk.get("action_type", ""),
                    "source": "camera_e2e_crosstalk_batch_priority",
                    "local_feasibility": first_xtalk.get("local_feasibility", ""),
                    "command_or_input": first_xtalk.get("command", ""),
                    "expected_artifact": first_xtalk.get("expected_artifact", ""),
                    "why": first_xtalk.get("why_this_next", ""),
                }
            )

        for domain in DOMAINS:
            rows_for_domain = [row for row in coverage_by_slug.get(slug, []) if row.get("domain") == domain]
            trust_domain = trust_domain_by_key.get((slug, domain), {})
            product_gates = gate_counts(rows_for_domain, "product_gate")
            product_gate = "PASS" if product_gates and set(product_gates) <= {"PASS", "N/A"} else "FAIL"
            row_scope = domain_scope({"domain": domain, "product_gate": product_gate}, trust_domain)
            blockers = compact_unique([row.get("primary_blocker", "") for row in rows_for_domain])
            domain_row = {
                "slug": slug,
                "code": consumer.get("code", ""),
                "manufacturer": consumer.get("manufacturer", ""),
                "device_name": consumer.get("device_name", ""),
                "domain": domain,
                "domain_use_scope": row_scope,
                "allowed_use": domain_allowed_use(row_scope),
                "blocked_use": domain_blocked_use(row_scope),
                "product_gate": product_gate,
                "requirement_count": len(rows_for_domain),
                "research_gate_counts": json.dumps(gate_counts(rows_for_domain, "research_gate"), ensure_ascii=False),
                "product_gate_counts": json.dumps(product_gates, ensure_ascii=False),
                "row_count_sum": sum(safe_int(row.get("row_count")) for row in rows_for_domain),
                "evidence_confidence_score_0_100": trust_domain.get("evidence_confidence_score_0_100", ""),
                "product_calibration_score_0_100": trust_domain.get("product_calibration_score_0_100", ""),
                "primary_blockers": blockers,
                "recommended_next_action": trust_domain.get("recommended_next_action", ""),
            }
            domain_rows.append(domain_row)
            if product_gate != "PASS":
                action_rows.append(measured_input_action(consumer, domain, blockers))

    for index, row in enumerate(action_rows, start=1):
        row["priority_rank"] = index

    issues: list[dict[str, Any]] = []
    if not sensor_rows:
        issues.append({"severity": "error", "code": "no_sensor_rows", "message": "No consumer sensor rows were loaded."})
    missing_domains = [
        {"slug": row.get("slug"), "domain": domain}
        for row in sensor_rows
        for domain in DOMAINS
        if not any(item.get("slug") == row.get("slug") and item.get("domain") == domain for item in domain_rows)
    ]
    if missing_domains:
        issues.append({"severity": "error", "code": "missing_domain_rows", "message": json.dumps(missing_domains[:12], ensure_ascii=False)})
    if any(boolish(row.get("product_ready")) for row in sensor_rows):
        issues.append({"severity": "warning", "code": "product_ready_sensor_present", "message": "At least one sensor is marked product-ready; verify measured calibration evidence."})

    sensor_csv = output_dir / "camera_e2e_use_scope_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_use_scope_by_domain.csv"
    action_csv = output_dir / "camera_e2e_use_scope_next_actions.csv"
    json_path = output_dir / "camera_e2e_use_scope_summary.json"
    html_path = output_dir / "index.html"
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    payload = {
        "schema": "camera_e2e_use_scope_summary_v1",
        "artifact_role": "camera_e2e_consumer_use_scope_decision_table",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "domain_row_count": len(domain_rows),
        "next_action_row_count": len(action_rows),
        "product_ready_count": product_ready_count,
        "use_scope_counts": dict(Counter(row.get("camera_e2e_use_scope", "") for row in sensor_rows)),
        "domain_scope_counts": dict(Counter(row.get("domain_use_scope", "") for row in domain_rows)),
        "validation": {
            "schema": "camera_e2e_use_scope_summary_validation_v1",
            "pass": error_count == 0,
            "status": "RESEARCH_USE_SCOPE_READY_PRODUCT_BLOCKED" if error_count == 0 and product_ready_count == 0 else ("PRODUCT_SCOPE_PRESENT" if error_count == 0 else "FAIL"),
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "issues": issues,
        },
        "outputs": {
            "json": repo_rel(json_path),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "next_actions_csv": repo_rel(action_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_csv(action_csv, action_rows, ACTION_COLUMNS)
    write_json(json_path, payload)
    write_html(html_path, payload, sensor_rows, domain_rows, action_rows)
    update_package(package_dir, payload)
    return payload


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


def write_html(path: Path, payload: dict[str, Any], sensors: list[dict[str, Any]], domains: list[dict[str, Any]], actions: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:30px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Use Scope Summary</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Use Scope Summary</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This is a routing and gate table, not a product-accuracy certificate.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("domain_row_count", 0))}</div><div class="muted">domain rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Sensor Routing</h2>{html_table(sensors, SENSOR_COLUMNS)}
<h2>Domain Routing</h2>{html_table(domains, DOMAIN_COLUMNS)}
<h2>Next Actions</h2>{html_table(actions, ACTION_COLUMNS)}
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
    for key, value in payload.get("outputs", {}).items():
        outputs[f"camera_e2e_use_scope_{key}"] = value
    package["latest_camera_e2e_use_scope_summary"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        "use_scope_counts": payload["use_scope_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_json, package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    payload = build_use_scope(parser.parse_args())
    print(
        json.dumps(
            {
                "schema": payload.get("schema"),
                "validation": payload.get("validation"),
                "sensor_count": payload.get("sensor_count"),
                "domain_row_count": payload.get("domain_row_count"),
                "product_ready_count": payload.get("product_ready_count"),
                "outputs": payload.get("outputs"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
