#!/usr/bin/env python3
"""Export per-sensor CameraE2E capability/use profiles.

This artifact is a compact consumer-facing view over the larger coverage,
mesh-confidence, CFA provenance, electrical/readout, and module-coupling tables.
It does not create new simulation values. It tells CameraE2E code what each
existing row group can safely be used for: plumbing, research prior, partial
trend analysis, or product-blocked.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_capability_profile"

CAPABILITY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "overall_use_scope",
    "overall_product_gate",
    "product_ready",
    "mesh_confidence_class",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "color_accuracy_gate",
    "material_input_scope",
    "spectral_qe_scope",
    "color_response_scope",
    "optical_crosstalk_scope",
    "cra_response_scope",
    "microlens_shift_scope",
    "electrical_noise_scope",
    "readout_raw_scope",
    "binning_remosaic_scope",
    "module_coupling_scope",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "key_blockers",
    "next_actions",
]

CHECK_COLUMNS = ["check_id", "severity", "status", "evidence", "required_action"]


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
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def coverage_row(rows: list[dict[str, str]], requirement_id: str) -> dict[str, str]:
    return next((row for row in rows if row.get("requirement_id") == requirement_id), {})


def scope_for_material(cfa: dict[str, str], material: dict[str, str]) -> str:
    if cfa.get("cfa_assumption_gate") == "MISSING":
        return "PLUMBING_ONLY_CFA_UNKNOWN"
    if material.get("product_lut_gate") == "PASS":
        return "PRODUCT_READY"
    return "RESEARCH_PROXY_MATERIAL_INPUT"


def scope_for_spectral(mesh: dict[str, str], cfa: dict[str, str]) -> str:
    if cfa.get("cfa_assumption_gate") == "MISSING":
        return "PLUMBING_ONLY_CFA_UNKNOWN"
    mesh_class = mesh.get("mesh_confidence_class", "")
    if mesh_class == "MEDIUM_RESEARCH_FIELD_TREND":
        return "PARTIAL_RESEARCH_FIELD_TREND"
    if mesh_class == "LOW_RESEARCH_ANCHOR":
        return "SINGLE_ANCHOR_RESEARCH_FIELD_CHECK"
    if mesh_class == "LOW_RESEARCH_WITH_FAILED_POINT":
        return "ROUGH_RESEARCH_WITH_FAILED_MESH_POINT"
    return "STRUCTURAL_PRIOR_ONLY"


def scope_for_color(cfa: dict[str, str]) -> str:
    if cfa.get("cfa_assumption_gate") == "MISSING":
        return "PLUMBING_ONLY_CFA_UNKNOWN"
    if cfa.get("cfa_provenance_class") == "MONO_CLEAR_PROXY":
        return "NOT_APPLICABLE_MONO_CLEAR"
    return "RESEARCH_RGB_PROXY_SEED"


def scope_for_crosstalk(mesh: dict[str, str]) -> str:
    if safe_int(mesh.get("crosstalk_pass_points")) > 0:
        return "PARTIAL_RESEARCH_CROSSTALK_TREND"
    return "COMPACT_SURROGATE_ONLY_NO_FINITE_ARRAY_PASS"


def scope_for_cra(mesh: dict[str, str], module: dict[str, str]) -> str:
    if module.get("product_lut_gate") == "PASS" and safe_int(mesh.get("field_pass_points")) > 0:
        return "PRODUCT_READY"
    if mesh.get("mesh_confidence_class") == "MEDIUM_RESEARCH_FIELD_TREND":
        return "PARTIAL_RESEARCH_CRA_FIELD_TREND"
    if mesh.get("mesh_confidence_class") == "LOW_RESEARCH_ANCHOR":
        return "SINGLE_ANCHOR_RESEARCH_CRA_CHECK"
    return "CRA_PRIOR_ONLY"


def scope_for_prior(summary_row: dict[str, str], product_key: str = "product_lut_gate") -> str:
    if summary_row.get(product_key) == "PASS":
        return "PRODUCT_READY"
    if summary_row:
        return "RESEARCH_PRIOR_SEED_ONLY"
    return "MISSING"


def overall_scope(row: dict[str, Any]) -> str:
    if boolish(row.get("product_ready")):
        return "PRODUCT_READY"
    scopes = {str(value) for key, value in row.items() if key.endswith("_scope")}
    if "PLUMBING_ONLY_CFA_UNKNOWN" in scopes:
        return "PLUMBING_PLUS_PRIORS_CFA_UNKNOWN"
    if "PARTIAL_RESEARCH_FIELD_TREND" in scopes or "PARTIAL_RESEARCH_CRA_FIELD_TREND" in scopes:
        return "PARTIAL_RESEARCH_TREND"
    if "SINGLE_ANCHOR_RESEARCH_FIELD_CHECK" in scopes or "SINGLE_ANCHOR_RESEARCH_CRA_CHECK" in scopes:
        return "SINGLE_ANCHOR_RESEARCH_CHECK"
    return "RESEARCH_PRIOR_SCHEMA_PLUMBING"


def compact_blockers(*texts: str, limit: int = 8) -> str:
    blockers: list[str] = []
    for text in texts:
        for item in str(text or "").split(";"):
            clean = item.strip()
            if clean and clean not in blockers:
                blockers.append(clean)
    return "; ".join(blockers[:limit])


def build_profiles(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    sensor_rows = read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv")
    if not sensor_rows:
        sensor_rows = read_csv_rows(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv")

    coverage_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv"), "slug")
    coverage_summary_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_summary.csv"), "slug")
    mesh_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv"), "slug")
    cfa_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv"), "slug")
    material_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_summary.csv"), "slug")
    electrical_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_readout_summary.csv"), "slug")
    module_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_summary.csv"), "slug")

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        coverage_rows = coverage_by_slug.get(slug, [])
        coverage_summary = coverage_summary_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        cfa = cfa_by_slug.get(slug, {})
        material = material_by_slug.get(slug, {})
        electrical = electrical_by_slug.get(slug, {})
        module = module_by_slug.get(slug, {})

        product_ready = boolish(sensor.get("camera_e2e_product_ready") or sensor.get("product_ready"))
        row = {
            "slug": slug,
            "code": sensor.get("code", ""),
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "overall_use_scope": "",
            "overall_product_gate": "PASS" if product_ready else "FAIL",
            "product_ready": product_ready,
            "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
            "cfa_provenance_class": cfa.get("cfa_provenance_class", ""),
            "cfa_assumption_gate": cfa.get("cfa_assumption_gate", ""),
            "color_accuracy_gate": sensor.get("color_accuracy_gate", ""),
            "material_input_scope": scope_for_material(cfa, material),
            "spectral_qe_scope": scope_for_spectral(mesh, cfa),
            "color_response_scope": scope_for_color(cfa),
            "optical_crosstalk_scope": scope_for_crosstalk(mesh),
            "cra_response_scope": scope_for_cra(mesh, module),
            "microlens_shift_scope": "DESIGN_PRIOR_ONLY" if coverage_row(coverage_rows, "microlens_ocl_shift_map").get("research_gate") == "CHECK" else "MISSING",
            "electrical_noise_scope": scope_for_prior(electrical),
            "readout_raw_scope": scope_for_prior(electrical),
            "binning_remosaic_scope": scope_for_prior(electrical),
            "module_coupling_scope": "MODULE_PRIOR_ONLY" if module.get("research_use_gate") == "CHECK" else "MISSING",
            "field_mesh_pass_points": mesh.get("field_pass_points", ""),
            "field_mesh_required_points": mesh.get("field_required_points", ""),
            "crosstalk_mesh_pass_points": mesh.get("crosstalk_pass_points", ""),
            "crosstalk_mesh_required_points": mesh.get("crosstalk_required_points", ""),
            "coverage_research_gate_counts": coverage_summary.get("research_gate_counts", ""),
            "coverage_product_gate_counts": coverage_summary.get("product_gate_counts", ""),
            "key_blockers": compact_blockers(
                cfa.get("primary_blocker", ""),
                mesh.get("primary_limitations", ""),
                material.get("primary_blocker", ""),
                module.get("primary_blocker", ""),
                electrical.get("primary_blocker", ""),
                coverage_summary.get("primary_blockers", ""),
            ),
            "next_actions": compact_blockers(cfa.get("next_action", ""), mesh.get("next_action", ""), limit=5),
        }
        row["overall_use_scope"] = overall_scope(row)
        rows.append(row)

        if not cfa:
            checks.append({"check_id": f"{slug}:cfa_provenance_missing", "severity": "error", "status": "FAIL", "evidence": "{}", "required_action": "Regenerate CFA provenance before capability export."})
        if not mesh:
            checks.append({"check_id": f"{slug}:mesh_confidence_missing", "severity": "error", "status": "FAIL", "evidence": "{}", "required_action": "Regenerate mesh confidence before capability export."})
        if row["overall_use_scope"] == "PRODUCT_READY":
            checks.append({"check_id": f"{slug}:unexpected_product_ready", "severity": "error", "status": "FAIL", "evidence": json.dumps(row, sort_keys=True), "required_action": "Do not mark product-ready without measured/calibrated gates."})

    error_count = sum(1 for check in checks if check.get("severity") == "error")
    status = "FAIL" if error_count else "RESEARCH_CAPABILITY_PROFILE_READY_PRODUCT_BLOCKED"
    by_sensor_csv = output_dir / "camera_e2e_capability_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_capability_checks.csv"
    report_json = output_dir / "camera_e2e_capability_profile.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_capability_profile_v1",
        "artifact_role": "camera_e2e_per_sensor_use_scope_profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(rows),
        "row_count": len(rows),
        "product_ready_count": sum(1 for row in rows if boolish(row.get("product_ready"))),
        "scope_counts": {
            "overall_use_scope": dict(Counter(str(row.get("overall_use_scope", "")) for row in rows)),
            "spectral_qe_scope": dict(Counter(str(row.get("spectral_qe_scope", "")) for row in rows)),
            "color_response_scope": dict(Counter(str(row.get("color_response_scope", "")) for row in rows)),
            "optical_crosstalk_scope": dict(Counter(str(row.get("optical_crosstalk_scope", "")) for row in rows)),
        },
        "validation": {
            "schema": "camera_e2e_capability_profile_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": len(checks),
            "error_count": error_count,
            "warning_count": sum(1 for check in checks if check.get("severity") == "warning"),
            "issues": checks,
        },
        "policy": {
            "research": "Use scope strings must be propagated to CameraE2E. CHECK gates are not product accuracy.",
            "product": "Blocked until product_ready is true and all product/domain gates pass.",
        },
        "outputs": {
            "json": repo_rel(report_json),
            "by_sensor_csv": repo_rel(by_sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(by_sensor_csv, rows, CAPABILITY_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, rows, checks)
    update_package(package_dir, payload)
    return payload


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    if not shown:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1440px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    issue_html = html_table(checks, CHECK_COLUMNS) if checks else '<p class="pass">No capability structural errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Capability Profile</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Capability Profile</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Scope strings summarize what each sensor artifact can safely support.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">profile status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric fail">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Checks</h2>{issue_html}
<h2>Capability By Sensor</h2>{html_table(rows, CAPABILITY_COLUMNS)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_capability_profile_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_capability_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_capability_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_capability_profile_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_capability_profile"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        "scope_counts": payload["scope_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_profiles(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "product_ready_count": payload["product_ready_count"],
                "scope_counts": payload["scope_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
