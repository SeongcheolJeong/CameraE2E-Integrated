#!/usr/bin/env python3
"""Export CameraE2E LUT trust assessment tables.

The scores in this report are not sensor-accuracy percentages. They separate
three different questions:

* research_usability_score: are rows present and loadable for CameraE2E research?
* evidence_confidence_score: how much solver/measured evidence backs the values?
* product_calibration_score: how much of the product gate is actually closed?

This keeps trend/prototype usefulness visible while making product use fail
closed until measured stack/material/CRA/electrical/readout/module gates pass.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_lut_trust_assessment"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "trust_class",
    "camera_e2e_allowed_use",
    "product_ready",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "research_utility_grade_0_10",
    "solver_evidence_grade_0_10",
    "product_accuracy_grade_0_10",
    "field_mesh_pass_fraction",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_fraction",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "crosstalk_support_pilot_row_count",
    "crosstalk_support_product_gate_counts",
    "crosstalk_support_best_truncation_fraction",
    "crosstalk_support_worst_truncation_fraction",
    "crosstalk_support_recommended_kernel",
    "crosstalk_support_status",
    "coverage_requirement_count",
    "research_gate_counts",
    "product_gate_counts",
    "mesh_confidence_class",
    "capability_overall_use_scope",
    "primary_blockers",
    "recommended_next_action",
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
    "research_gate_counts",
    "product_gate_counts",
    "row_count_sum",
    "primary_blockers",
    "recommended_next_action",
]

REQUIREMENT_COLUMNS = [
    "slug",
    "domain",
    "requirement_id",
    "requirement",
    "trust_class",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "research_gate",
    "product_gate",
    "row_count",
    "primary_blocker",
    "camera_e2e_use",
    "source_artifacts",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]

OPTICAL_FIELD_REQUIREMENTS = {
    "spectral_response_qe",
    "angular_cra_response",
    "microlens_ocl_shift_map",
}
OPTICAL_CROSSTALK_REQUIREMENTS = {"optical_crosstalk_kernel"}
OPTICAL_PROXY_REQUIREMENTS = {"optical_material_nk_ri", "color_response_matrix"}


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
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def pct(value: float) -> str:
    return f"{max(0.0, min(100.0, value)):.2f}"


def fraction(numerator: Any, denominator: Any) -> float:
    den = safe_float(denominator)
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, safe_float(numerator) / den))


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


def gate_score(gate: str, *, na_value: float | None = None) -> float | None:
    normalized = str(gate or "").upper()
    if normalized == "PASS":
        return 100.0
    if normalized == "CHECK":
        return 65.0
    if normalized == "N/A":
        return na_value
    if normalized in {"FAIL", "MISSING", ""}:
        return 0.0
    return 0.0


def avg(values: list[float | None]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "") or "MISSING") for row in rows).items()))


def compact_unique(values: list[str], limit: int = 8) -> str:
    output: list[str] = []
    for value in values:
        for part in str(value or "").split(";"):
            clean = part.strip()
            if clean and clean not in output:
                output.append(clean)
    return "; ".join(output[:limit])


def grade_0_10(score_0_100: float) -> str:
    return f"{max(0.0, min(10.0, score_0_100 / 10.0)):.2f}"


def crosstalk_support_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "gate_counts": {},
            "best_truncation": "",
            "worst_truncation": "",
            "recommended_kernel": "",
            "status": "NO_FINITE_ARRAY_SUPPORT_PILOT",
        }

    neighborhoods = [safe_int(row.get("best_pilot_neighborhood")) for row in rows if safe_int(row.get("best_pilot_neighborhood")) > 0]
    truncations = [
        safe_float(row.get("best_pilot_truncation_fraction"), math.nan)
        for row in rows
        if math.isfinite(safe_float(row.get("best_pilot_truncation_fraction"), math.nan))
    ]
    gate_counter = gate_counts(rows, "product_crosstalk_gate")
    all_product_failed = bool(rows) and all(str(row.get("product_crosstalk_gate", "")).upper() == "FAIL" for row in rows)
    if all_product_failed:
        status = "LOW_RES_SUPPORT_PILOT_ONLY_PRODUCT_BLOCKED"
    else:
        status = "CROSSTALK_SUPPORT_REVIEW_REQUIRED"
    return {
        "row_count": len(rows),
        "gate_counts": gate_counter,
        "best_truncation": f"{min(truncations):.6f}" if truncations else "",
        "worst_truncation": f"{max(truncations):.6f}" if truncations else "",
        "recommended_kernel": f"{max(neighborhoods)}x{max(neighborhoods)}" if neighborhoods else "",
        "status": status,
    }


def evidence_score_for_requirement(row: dict[str, str], mesh: dict[str, str]) -> float:
    product_score = gate_score(row.get("product_gate", ""), na_value=None)
    if product_score == 100.0:
        return 100.0

    requirement_id = row.get("requirement_id", "")
    if requirement_id in OPTICAL_FIELD_REQUIREMENTS:
        return 100.0 * fraction(mesh.get("field_pass_points"), mesh.get("field_required_points"))
    if requirement_id in OPTICAL_CROSSTALK_REQUIREMENTS:
        return 100.0 * fraction(mesh.get("crosstalk_pass_points"), mesh.get("crosstalk_required_points"))
    if requirement_id in OPTICAL_PROXY_REQUIREMENTS:
        if row.get("research_gate") == "CHECK" and safe_int(row.get("row_count")) > 0:
            return 10.0
        return 0.0
    if row.get("research_gate") == "CHECK" and safe_int(row.get("row_count")) > 0:
        return 10.0
    return 0.0


def requirement_trust_class(research_score: float, evidence_score: float, product_score: float) -> str:
    if product_score >= 99.9 and evidence_score >= 95.0:
        return "PRODUCT_READY"
    if evidence_score >= 20.0:
        return "PARTIAL_NUMERICAL_TREND"
    if evidence_score > 0.0:
        return "SPARSE_OR_PROXY_EVIDENCE"
    if research_score > 0.0:
        return "RESEARCH_PRIOR_LOADABLE"
    return "MISSING_OR_BLOCKED"


def sensor_trust_class(evidence_score: float, product_score: float, field_pass: int, crosstalk_pass: int, research_score: float) -> str:
    if product_score >= 99.9 and evidence_score >= 95.0:
        return "PRODUCT_READY"
    if field_pass >= 9 and evidence_score >= 5.0:
        return "PARTIAL_FIELD_TREND_PRODUCT_BLOCKED"
    if field_pass > 0:
        return "SPARSE_FIELD_ANCHOR_PRODUCT_BLOCKED"
    if crosstalk_pass > 0:
        return "SPARSE_CROSSTALK_ANCHOR_PRODUCT_BLOCKED"
    if research_score > 0.0:
        return "STRUCTURAL_PRIOR_LOADABLE_PRODUCT_BLOCKED"
    return "MISSING_OR_BLOCKED"


def allowed_use_for_class(trust_class: str) -> str:
    if trust_class == "PRODUCT_READY":
        return "CameraE2E product LUT use allowed only when every row-level product gate also passes."
    if trust_class.startswith("PARTIAL_FIELD_TREND"):
        return "Research field/CRA trend studies around covered anchors; keep product mode blocked."
    if trust_class.startswith("SPARSE_FIELD_ANCHOR"):
        return "Local anchor or smoke comparison only; do not extrapolate as calibrated LUT."
    if trust_class.startswith("SPARSE_CROSSTALK"):
        return "Crosstalk smoke/trend anchor only; do not use as product kernel."
    if trust_class.startswith("STRUCTURAL_PRIOR"):
        return "CameraE2E loader/schema plumbing and coarse sensitivity tests only."
    if trust_class == "PARTIAL_NUMERICAL_TREND":
        return "Research trend use for this requirement; product calibration still blocked."
    if trust_class == "SPARSE_OR_PROXY_EVIDENCE":
        return "Proxy or sparse-evidence research use only."
    if trust_class == "RESEARCH_PRIOR_LOADABLE":
        return "Prior-seed research plumbing only."
    return "Do not use until required artifact exists."


def domain_next_action(domain: str) -> str:
    if domain == "Optical / Color":
        return "Run missing quantitative field/crosstalk points and replace proxy stack/material/CRA with measured or design-source data."
    if domain == "Pixel / Electrical":
        return "Import measured CG/FWC/dark/DSNU/PRNU/noise and calibrated TCAD charge-collection targets."
    if domain == "Readout / RAW":
        return "Import measured gain, black-level, ADC, timing, FPN, defect, and mode calibration tables."
    if domain == "Module Coupling":
        return "Import lens raytrace or measured module CRA, vignetting, sensor pose, and chromatic pupil maps."
    return "Close product gates listed in coverage matrix."


def build_requirement_rows(
    coverage_rows: list[dict[str, str]],
    mesh_by_slug: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in coverage_rows:
        mesh = mesh_by_slug.get(row.get("slug", ""), {})
        research_score = gate_score(row.get("research_gate", ""), na_value=None) or 0.0
        product_score = gate_score(row.get("product_gate", ""), na_value=None) or 0.0
        evidence_score = evidence_score_for_requirement(row, mesh)
        trust_class = requirement_trust_class(research_score, evidence_score, product_score)
        output.append(
            {
                "slug": row.get("slug", ""),
                "domain": row.get("domain", ""),
                "requirement_id": row.get("requirement_id", ""),
                "requirement": row.get("requirement", ""),
                "trust_class": trust_class,
                "research_usability_score_0_100": pct(research_score),
                "evidence_confidence_score_0_100": pct(evidence_score),
                "product_calibration_score_0_100": pct(product_score),
                "research_gate": row.get("research_gate", ""),
                "product_gate": row.get("product_gate", ""),
                "row_count": row.get("row_count", ""),
                "primary_blocker": row.get("primary_blocker", ""),
                "camera_e2e_use": row.get("camera_e2e_use", ""),
                "source_artifacts": row.get("source_artifacts", ""),
            }
        )
    return output


def build_domain_rows(requirement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (slug, domain), rows in sorted(group_rows(requirement_rows, "slug_domain").items()):
        product_score = avg([safe_float(row.get("product_calibration_score_0_100")) for row in rows])
        evidence_score = avg([safe_float(row.get("evidence_confidence_score_0_100")) for row in rows])
        research_score = avg([safe_float(row.get("research_usability_score_0_100")) for row in rows])
        trust_class = requirement_trust_class(research_score, evidence_score, product_score)
        output.append(
            {
                "slug": slug,
                "domain": domain,
                "trust_class": trust_class,
                "camera_e2e_allowed_use": allowed_use_for_class(trust_class),
                "research_usability_score_0_100": pct(research_score),
                "evidence_confidence_score_0_100": pct(evidence_score),
                "product_calibration_score_0_100": pct(product_score),
                "requirement_count": len(rows),
                "research_gate_counts": json.dumps(gate_counts(rows, "research_gate"), sort_keys=True),
                "product_gate_counts": json.dumps(gate_counts(rows, "product_gate"), sort_keys=True),
                "row_count_sum": sum(safe_int(row.get("row_count")) for row in rows),
                "primary_blockers": compact_unique([row.get("primary_blocker", "") for row in rows]),
                "recommended_next_action": domain_next_action(domain),
            }
        )
    return output


def add_group_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["slug_domain"] = (row.get("slug", ""), row.get("domain", ""))
    return rows


def build_sensor_rows(
    sensors: list[dict[str, str]],
    requirement_rows: list[dict[str, Any]],
    coverage_rows_by_slug: dict[str, list[dict[str, str]]],
    mesh_by_slug: dict[str, dict[str, str]],
    capability_by_slug: dict[str, dict[str, str]],
    crosstalk_support_by_slug: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    req_by_slug = group_rows(requirement_rows, "slug")
    output: list[dict[str, Any]] = []
    for sensor in sensors:
        slug = sensor.get("slug", "")
        req_rows = req_by_slug.get(slug, [])
        coverage_rows = coverage_rows_by_slug.get(slug, [])
        mesh = mesh_by_slug.get(slug, {})
        capability = capability_by_slug.get(slug, {})
        field_pass = safe_int(mesh.get("field_pass_points"))
        field_required = safe_int(mesh.get("field_required_points"))
        crosstalk_pass = safe_int(mesh.get("crosstalk_pass_points"))
        crosstalk_required = safe_int(mesh.get("crosstalk_required_points"))
        field_fraction = fraction(field_pass, field_required)
        crosstalk_fraction = fraction(crosstalk_pass, crosstalk_required)
        research_score = avg([safe_float(row.get("research_usability_score_0_100")) for row in req_rows])
        product_score = avg([safe_float(row.get("product_calibration_score_0_100")) for row in req_rows])
        calibration_fraction = product_score / 100.0
        evidence_score = 100.0 * (0.45 * field_fraction + 0.25 * crosstalk_fraction + 0.30 * calibration_fraction)
        trust_class = sensor_trust_class(evidence_score, product_score, field_pass, crosstalk_pass, research_score)
        product_ready = boolish(sensor.get("camera_e2e_ready") or sensor.get("product_ready"))
        support = crosstalk_support_summary(crosstalk_support_by_slug.get(slug, []))
        output.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "trust_class": trust_class,
                "camera_e2e_allowed_use": allowed_use_for_class(trust_class),
                "product_ready": product_ready,
                "research_usability_score_0_100": pct(research_score),
                "evidence_confidence_score_0_100": pct(evidence_score),
                "product_calibration_score_0_100": pct(product_score),
                "research_utility_grade_0_10": grade_0_10(research_score),
                "solver_evidence_grade_0_10": grade_0_10(evidence_score),
                "product_accuracy_grade_0_10": grade_0_10(product_score),
                "field_mesh_pass_fraction": f"{field_fraction:.6f}",
                "field_mesh_pass_points": field_pass,
                "field_mesh_required_points": field_required,
                "crosstalk_mesh_pass_fraction": f"{crosstalk_fraction:.6f}",
                "crosstalk_mesh_pass_points": crosstalk_pass,
                "crosstalk_mesh_required_points": crosstalk_required,
                "crosstalk_support_pilot_row_count": support["row_count"],
                "crosstalk_support_product_gate_counts": json.dumps(support["gate_counts"], sort_keys=True),
                "crosstalk_support_best_truncation_fraction": support["best_truncation"],
                "crosstalk_support_worst_truncation_fraction": support["worst_truncation"],
                "crosstalk_support_recommended_kernel": support["recommended_kernel"],
                "crosstalk_support_status": support["status"],
                "coverage_requirement_count": len(coverage_rows),
                "research_gate_counts": json.dumps(gate_counts(coverage_rows, "research_gate"), sort_keys=True),
                "product_gate_counts": json.dumps(gate_counts(coverage_rows, "product_gate"), sort_keys=True),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
                "capability_overall_use_scope": capability.get("overall_use_scope", ""),
                "primary_blockers": compact_unique([row.get("primary_blocker", "") for row in coverage_rows]),
                "recommended_next_action": "Close measured-data blockers and quantitative FDTD coverage before product LUT use.",
            }
        )
    return output


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def validate(
    *,
    sensor_rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
    requirement_rows: list[dict[str, Any]],
    coverage_row_count: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    allowed_product_score_count = sum(
        1
        for row in sensor_rows
        if safe_float(row.get("product_calibration_score_0_100")) >= 99.9
    )
    checks.append(check_row("sensor_rows_present", bool(sensor_rows), "PASS" if sensor_rows else "FAIL", {"sensor_row_count": len(sensor_rows)}, "Generate sensor index and mesh-confidence artifacts."))
    checks.append(check_row("domain_rows_present", bool(domain_rows), "PASS" if domain_rows else "FAIL", {"domain_row_count": len(domain_rows)}, "Generate coverage matrix domain rows."))
    checks.append(
        check_row(
            "requirement_rows_match_coverage",
            len(requirement_rows) == coverage_row_count,
            "PASS" if len(requirement_rows) == coverage_row_count else "FAIL",
            {"requirement_rows": len(requirement_rows), "coverage_rows": coverage_row_count},
            "Trust-by-requirement rows must map one-to-one with coverage matrix rows.",
        )
    )
    checks.append(
        check_row(
            "product_scores_fail_closed",
            allowed_product_score_count == product_ready_count,
            "PASS" if allowed_product_score_count == product_ready_count else "FAIL",
            {"product_ready_count": product_ready_count, "product_score_ready_count": allowed_product_score_count},
            "Do not mark high product calibration score unless product_ready is true.",
        )
    )
    failures = [row for row in checks if row.get("status") == "FAIL"]
    return {
        "schema": "camera_e2e_lut_trust_assessment_validation_v1",
        "pass": not failures,
        "status": "RESEARCH_TRUST_ASSESSMENT_READY_PRODUCT_BLOCKED" if not failures and product_ready_count == 0 else ("PRODUCT_TRUST_ASSESSMENT_READY" if not failures else "FAIL"),
        "issue_count": len(failures),
        "error_count": len(failures),
        "warning_count": 0,
        "issues": failures,
        "checks": checks,
    }


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
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
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:24px;font-weight:800}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:7px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}
"""
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E LUT Trust Assessment</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E LUT Trust Assessment</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Scores are research/evidence/gate confidence, not measured sensor accuracy percentages.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("validation", {}).get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product ready</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("mean_evidence_confidence_score_0_100", 0))}</div><div class="muted">mean evidence score</div></div>
  </div>
  <h2>Policy</h2>
  <p class="muted">{html_cell(payload.get("score_policy", {}).get("product_warning", ""))}</p>
  <h2>Sensor Trust</h2>
  {html_table(payload.get("sensor_rows", []), SENSOR_COLUMNS, limit=80)}
  <h2>Domain Trust</h2>
  {html_table(payload.get("domain_rows", []), DOMAIN_COLUMNS, limit=120)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_lut_trust_assessment_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_lut_trust_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_lut_trust_by_domain_csv"] = payload["outputs"]["domain_csv"]
    outputs["camera_e2e_lut_trust_by_requirement_csv"] = payload["outputs"]["requirement_csv"]
    outputs["camera_e2e_lut_trust_assessment_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_lut_trust_assessment"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        "mean_evidence_confidence_score_0_100": payload["mean_evidence_confidence_score_0_100"],
        "trust_class_counts": payload["trust_class_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_trust(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    mesh_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv"), "slug")
    capability_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv"), "slug")
    crosstalk_support_by_slug = group_rows(
        read_csv_rows(package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_by_sensor.csv"),
        "slug",
    )
    coverage_by_slug = group_rows(coverage_rows, "slug")

    requirement_rows = add_group_key(build_requirement_rows(coverage_rows, mesh_by_slug))
    domain_rows = build_domain_rows(requirement_rows)
    sensor_rows = build_sensor_rows(
        sensors,
        requirement_rows,
        coverage_by_slug,
        mesh_by_slug,
        capability_by_slug,
        crosstalk_support_by_slug,
    )
    validation = validate(
        sensor_rows=sensor_rows,
        domain_rows=domain_rows,
        requirement_rows=requirement_rows,
        coverage_row_count=len(coverage_rows),
    )

    sensor_csv = output_dir / "camera_e2e_lut_trust_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_lut_trust_by_domain.csv"
    requirement_csv = output_dir / "camera_e2e_lut_trust_by_requirement.csv"
    checks_csv = output_dir / "camera_e2e_lut_trust_checks.csv"
    report_json = output_dir / "camera_e2e_lut_trust_assessment.json"
    html_path = output_dir / "index.html"

    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_csv(requirement_csv, requirement_rows, REQUIREMENT_COLUMNS)
    write_csv(checks_csv, validation["checks"], CHECK_COLUMNS)

    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    mean_evidence_score = avg([safe_float(row.get("evidence_confidence_score_0_100")) for row in sensor_rows])
    payload = {
        "schema": "camera_e2e_lut_trust_assessment_v1",
        "artifact_role": "camera_e2e_lut_trust_and_usage_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "domain_row_count": len(domain_rows),
        "requirement_row_count": len(requirement_rows),
        "product_ready_count": product_ready_count,
        "mean_evidence_confidence_score_0_100": pct(mean_evidence_score),
        "trust_class_counts": dict(Counter(str(row.get("trust_class", "")) for row in sensor_rows)),
        "validation": {key: value for key, value in validation.items() if key != "checks"},
        "score_policy": {
            "research_usability_score": "Average row-gate loadability for research workflows: PASS=100, CHECK=65, FAIL/MISSING=0, N/A ignored.",
            "evidence_confidence_score": "Weighted solver/measured evidence score. Sensor score = 45% field mesh PASS fraction + 25% finite-array crosstalk PASS fraction + 30% product calibration fraction.",
            "product_calibration_score": "Average product gate score across applicable CameraE2E requirements. Current prior/proxy rows score 0 for product calibration.",
            "grade_0_10": "Convenience view of the corresponding 0-100 score. Product accuracy grade follows product_calibration_score, not research usability.",
            "crosstalk_support_status": "Low-resolution support pilots estimate kernel truncation and next support size only; they do not count as finite-array product crosstalk PASS.",
            "product_warning": "These scores are not physical accuracy percentages. Product use is blocked unless product_ready is true and row-level product gates pass.",
        },
        "sensor_rows": sensor_rows,
        "domain_rows": domain_rows,
        "outputs": {
            "json": repo_rel(report_json),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "requirement_csv": repo_rel(requirement_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(report_json, payload)
    write_html(html_path, payload)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = export_trust(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "sensor_count": payload["sensor_count"],
                "domain_row_count": payload["domain_row_count"],
                "requirement_row_count": payload["requirement_row_count"],
                "product_ready_count": payload["product_ready_count"],
                "mean_evidence_confidence_score_0_100": payload["mean_evidence_confidence_score_0_100"],
                "trust_class_counts": payload["trust_class_counts"],
                "validation": payload["validation"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
