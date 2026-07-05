#!/usr/bin/env python3
"""Audit mesh/convergence confidence for the CameraE2E sensor LUT package.

This report answers a narrower question than the readiness audit: how much of
the current LUT is backed by quantitative, grid-resolution-passing solver rows,
and therefore what it can be trusted for. It deliberately keeps product use
blocked unless the full field and finite-array crosstalk grids pass.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_mesh_confidence"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "ocl_mode_guess",
    "field_required_points",
    "field_pass_points",
    "field_fail_points",
    "field_check_points",
    "field_resource_limited_points",
    "field_coverage_fraction",
    "field_grid_pass_fraction",
    "field_min_actual_resolution_px_per_um",
    "field_min_target_resolution_px_per_um",
    "field_min_resolution_ratio",
    "field_signed_flux_nonpositive_points",
    "crosstalk_required_points",
    "crosstalk_pass_points",
    "crosstalk_check_points",
    "crosstalk_resource_limited_points",
    "crosstalk_coverage_fraction",
    "runtime_rows",
    "runtime_gate_counts",
    "mesh_confidence_class",
    "camera_e2e_recommended_use",
    "product_lut_ready",
    "primary_limitations",
    "next_action",
]

DOMAIN_COLUMNS = [
    "slug",
    "domain",
    "metric",
    "confidence_class",
    "evidence",
    "camera_e2e_allowed_use",
    "product_gate",
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def as_int(value: Any, default: int = 0) -> int:
    number = as_float(value, None)
    if number is None:
        return default
    return int(number)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") or "MISSING" for row in rows).items()))


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def coverage_for(rows: list[dict[str, str]], slug: str, solver: str) -> dict[str, str]:
    for row in rows:
        if row.get("slug") == slug and row.get("solver") == solver:
            return row
    return {}


def classify_sensor(
    field_required: int,
    field_pass: int,
    field_fail: int,
    field_resource_limited: int,
    field_grid_pass_fraction: float | None,
    crosstalk_required: int,
    crosstalk_pass: int,
    crosstalk_resource_limited: int,
) -> tuple[str, str, str, str]:
    if field_required and crosstalk_required and field_pass >= field_required and crosstalk_pass >= crosstalk_required:
        return (
            "PRODUCT_MESH_COVERAGE_CANDIDATE",
            "Potential product candidate only if measured stack/material/CRA/electrical gates also pass.",
            "Full field and crosstalk mesh coverage present; verify measured-data gates.",
            "Run measured-data/product readiness audit.",
        )
    if field_pass >= 9 and (field_grid_pass_fraction or 0.0) >= 0.95:
        return (
            "MEDIUM_RESEARCH_FIELD_TREND",
            "Use for relative CRA/OCL field-shape experiments around covered anchors; keep product mode blocked.",
            "Partial high-resolution field anchors exist, but color/wavelength/field coverage and finite-array crosstalk are incomplete.",
            "Complete missing RGB x wavelength x field anchors and finite-array crosstalk convergence.",
        )
    if field_pass > 0 and field_fail == 0:
        return (
            "LOW_RESEARCH_ANCHOR",
            "Use only as local optical anchor or smoke/trend seed; do not extrapolate as calibrated field LUT.",
            "Only sparse field solver anchors pass.",
            "Run the planned quantitative field queue for this sensor.",
        )
    if field_resource_limited > 0:
        return (
            "STRUCTURAL_PRIOR_FIELD_RESOURCE_LIMITED",
            "Use only proxy/prior rows for rough system plumbing; this sensor needs batch/HPC for quantitative field anchors.",
            "At least one quantitative field point timed out or exceeded local resources before KPI rows were produced.",
            "Move this field point to batch/HPC or reduce resolution only for non-product trend work.",
        )
    if field_fail > 0:
        return (
            "LOW_RESEARCH_WITH_FAILED_POINT",
            "Use only proxy/prior rows for rough system plumbing; inspect failed mesh point before trusting trends.",
            "At least one attempted field point failed the grid/solver gate.",
            "Fix failed resolution/convergence point and rerun before using as trend evidence.",
        )
    if crosstalk_resource_limited > 0:
        return (
            "STRUCTURAL_PRIOR_CROSSTALK_RESOURCE_LIMITED",
            "Use compact surrogate crosstalk only for sensitivity tests.",
            "Finite-array crosstalk was not run because the local mesh exceeded resource limits.",
            "Run crosstalk queue on a larger batch/cluster environment.",
        )
    return (
        "STRUCTURAL_PRIOR_ONLY",
        "Use only for CameraE2E loader/schema plumbing and very coarse prior sensitivity.",
        "No quantitative mesh-resolution-passing field anchors are available for this sensor.",
        "Start with one center RGB/wavelength field anchor, then expand to field and crosstalk queues.",
    )


def domain_rows_for_sensor(sensor_row: dict[str, Any]) -> list[dict[str, Any]]:
    slug = sensor_row["slug"]
    field_pass = as_int(sensor_row.get("field_pass_points"))
    field_required = as_int(sensor_row.get("field_required_points"))
    crosstalk_pass = as_int(sensor_row.get("crosstalk_pass_points"))
    crosstalk_required = as_int(sensor_row.get("crosstalk_required_points"))
    runtime_rows = as_int(sensor_row.get("runtime_rows"))
    product_ready = boolish(sensor_row.get("product_lut_ready"))
    field_fraction = as_float(sensor_row.get("field_coverage_fraction"), 0.0) or 0.0
    crosstalk_fraction = as_float(sensor_row.get("crosstalk_coverage_fraction"), 0.0) or 0.0

    product_gate = "PASS" if product_ready else "FAIL"
    if field_required and field_pass == field_required:
        field_class = "HIGH_NUMERICAL_FIELD_COVERAGE"
    elif field_pass >= 9:
        field_class = "MEDIUM_PARTIAL_FIELD_COVERAGE"
    elif field_pass > 0:
        field_class = "LOW_SPARSE_FIELD_ANCHOR"
    else:
        field_class = "PRIOR_ONLY_NO_FIELD_MESH_PASS"

    if crosstalk_required and crosstalk_pass == crosstalk_required:
        xt_class = "HIGH_FINITE_ARRAY_CROSSTALK_COVERAGE"
    elif crosstalk_pass > 0:
        xt_class = "LOW_PARTIAL_CROSSTALK_COVERAGE"
    else:
        xt_class = "SURROGATE_CROSSTALK_ONLY"

    return [
        {
            "slug": slug,
            "domain": "Optical / Color",
            "metric": "Spectral response / QE / CRA field response",
            "confidence_class": field_class,
            "evidence": f"{field_pass}/{field_required} field points PASS; runtime rows {runtime_rows}; field coverage {field_fraction:.3f}",
            "camera_e2e_allowed_use": "research trend" if field_pass else "research prior/plumbing",
            "product_gate": product_gate,
        },
        {
            "slug": slug,
            "domain": "Optical / Color",
            "metric": "Optical crosstalk kernel",
            "confidence_class": xt_class,
            "evidence": f"{crosstalk_pass}/{crosstalk_required} finite-array crosstalk points PASS; coverage {crosstalk_fraction:.3f}",
            "camera_e2e_allowed_use": "research compact-kernel sensitivity only",
            "product_gate": product_gate,
        },
        {
            "slug": slug,
            "domain": "Pixel / Electrical",
            "metric": "Charge collection / read noise / DSNU / PRNU",
            "confidence_class": "PRIOR_SEED_NOT_MESH_VALIDATED",
            "evidence": "DEVSIM/TCAD and readout rows are prior/proxy unless measured calibration is imported.",
            "camera_e2e_allowed_use": "research noise/readout sensitivity",
            "product_gate": product_gate,
        },
        {
            "slug": slug,
            "domain": "Module Coupling",
            "metric": "CRA map / pupil / vignetting",
            "confidence_class": "PRIOR_FIELD_MAP_NOT_RAYTRACE_VALIDATED",
            "evidence": "Module CRA/pupil rows are priors unless lens raytrace or measured field map is imported.",
            "camera_e2e_allowed_use": "research module mismatch sensitivity",
            "product_gate": product_gate,
        },
    ]


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    package = read_json(package_dir / "camera_e2e_lut_package.json")
    runtime_bundle = read_json(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_bundle.json")
    readiness = read_json(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json")
    sensor_index = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    if not sensor_index:
        sensor_index = read_csv_rows(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv")

    coverage_rows = read_csv_rows(package_dir / "camera_e2e_quantitative_coverage.csv")
    merged_rows = read_csv_rows(package_dir / "camera_e2e_quantitative_merged_summary.csv")
    runtime_rows = read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv")

    merged_by_slug = group_rows(merged_rows, "slug")
    runtime_by_slug = group_rows(runtime_rows, "slug")
    sensor_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for sensor in sensor_index:
        slug = sensor.get("slug", "")
        if not slug:
            continue
        field_cov = coverage_for(coverage_rows, slug, "field")
        xt_cov = coverage_for(coverage_rows, slug, "crosstalk")
        field_rows = [row for row in merged_by_slug.get(slug, []) if row.get("solver") == "field"]
        xt_rows = [row for row in merged_by_slug.get(slug, []) if row.get("solver") == "crosstalk"]

        field_pass_rows = [
            row
            for row in field_rows
            if row.get("solver_gate") == "PASS" and str(row.get("grid_resolution_gate_pass", "")).lower() == "true"
        ]
        field_grid_rows = [row for row in field_rows if row.get("grid_resolution_gate_pass") != ""]
        field_grid_pass_fraction = (
            sum(1 for row in field_grid_rows if str(row.get("grid_resolution_gate_pass", "")).lower() == "true") / len(field_grid_rows)
            if field_grid_rows
            else None
        )
        ratios: list[float] = []
        actual_values: list[float] = []
        target_values: list[float] = []
        for row in field_rows:
            actual = as_float(row.get("actual_resolution_px_per_um"))
            target = as_float(row.get("target_resolution_px_per_um"))
            if actual is not None:
                actual_values.append(actual)
            if target is not None:
                target_values.append(target)
            if actual is not None and target and target > 0:
                ratios.append(actual / target)
        signed_nonpositive = 0
        for row in field_rows:
            signed_flux = as_float(row.get("signed_flux_si_absorption_fraction_diagnostic"))
            if signed_flux is not None and signed_flux <= 0:
                signed_nonpositive += 1

        field_required = as_int(field_cov.get("required_points"))
        field_pass = as_int(field_cov.get("pass_points"))
        field_fail = as_int(field_cov.get("fail_points"))
        field_check = as_int(field_cov.get("check_points"))
        field_resource = as_int(field_cov.get("resource_limited_points"))
        crosstalk_required = as_int(xt_cov.get("required_points"))
        crosstalk_pass = as_int(xt_cov.get("pass_points"))
        crosstalk_check = as_int(xt_cov.get("check_points"))
        crosstalk_resource = as_int(xt_cov.get("resource_limited_points"))
        confidence, use, limits, next_action = classify_sensor(
            field_required=field_required,
            field_pass=field_pass,
            field_fail=field_fail,
            field_resource_limited=field_resource,
            field_grid_pass_fraction=field_grid_pass_fraction,
            crosstalk_required=crosstalk_required,
            crosstalk_pass=crosstalk_pass,
            crosstalk_resource_limited=crosstalk_resource,
        )
        row = {
            "slug": slug,
            "code": sensor.get("code", ""),
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
            "cfa_pattern": sensor.get("cfa_pattern", ""),
            "ocl_mode_guess": sensor.get("ocl_mode_guess", ""),
            "field_required_points": field_required,
            "field_pass_points": field_pass,
            "field_fail_points": field_fail,
            "field_check_points": field_check,
            "field_resource_limited_points": field_resource,
            "field_coverage_fraction": field_cov.get("coverage_fraction", "0"),
            "field_grid_pass_fraction": format_float(field_grid_pass_fraction),
            "field_min_actual_resolution_px_per_um": format_float(min(actual_values) if actual_values else None, 3),
            "field_min_target_resolution_px_per_um": format_float(min(target_values) if target_values else None, 3),
            "field_min_resolution_ratio": format_float(min(ratios) if ratios else None, 3),
            "field_signed_flux_nonpositive_points": signed_nonpositive,
            "crosstalk_required_points": crosstalk_required,
            "crosstalk_pass_points": crosstalk_pass,
            "crosstalk_check_points": crosstalk_check,
            "crosstalk_resource_limited_points": crosstalk_resource,
            "crosstalk_coverage_fraction": xt_cov.get("coverage_fraction", "0"),
            "runtime_rows": len(runtime_by_slug.get(slug, [])),
            "runtime_gate_counts": json.dumps(gate_counts(runtime_by_slug.get(slug, []), "combined_evidence_gate"), sort_keys=True),
            "mesh_confidence_class": confidence,
            "camera_e2e_recommended_use": use,
            "product_lut_ready": False,
            "primary_limitations": limits,
            "next_action": next_action,
        }
        sensor_rows.append(row)
        domain_rows.extend(domain_rows_for_sensor(row))

    class_counts = dict(sorted(Counter(row["mesh_confidence_class"] for row in sensor_rows).items()))
    field_pass_total = sum(as_int(row["field_pass_points"]) for row in sensor_rows)
    field_required_total = sum(as_int(row["field_required_points"]) for row in sensor_rows)
    crosstalk_pass_total = sum(as_int(row["crosstalk_pass_points"]) for row in sensor_rows)
    crosstalk_required_total = sum(as_int(row["crosstalk_required_points"]) for row in sensor_rows)
    product_ready_count = sum(1 for row in sensor_rows if boolish(row["product_lut_ready"]))
    status = "PRODUCT_MESH_CONFIDENCE_READY" if product_ready_count == len(sensor_rows) and sensor_rows else "RESEARCH_MESH_CONFIDENCE_LOW_PRODUCT_BLOCKED"

    check_rows = [
        check_row(
            "quantitative_coverage_present",
            bool(coverage_rows),
            "PASS" if coverage_rows else "FAIL",
            {"coverage_rows": len(coverage_rows)},
            "Run build_camera_e2e_sensor_luts.py to regenerate quantitative coverage.",
        ),
        check_row(
            "runtime_bundle_present",
            bool(runtime_rows) and bool(runtime_bundle.get("validation", {}).get("pass")),
            "PASS" if runtime_rows else "FAIL",
            {"runtime_rows": len(runtime_rows), "runtime_validation": runtime_bundle.get("validation", {})},
            "Export runtime bundle before mesh confidence audit.",
        ),
        check_row(
            "product_gate_preserved",
            not bool(runtime_bundle.get("product_lut_ready")) and not bool(readiness.get("product_lut_ready")),
            "PRODUCT_BLOCKED_AS_EXPECTED",
            {"runtime_product_lut_ready": runtime_bundle.get("product_lut_ready"), "readiness_product_lut_ready": readiness.get("product_lut_ready")},
            "Keep product gates blocked until full solver coverage and measured calibration pass.",
        ),
        check_row(
            "full_field_mesh_coverage",
            field_required_total > 0 and field_pass_total == field_required_total,
            "PASS" if field_required_total > 0 and field_pass_total == field_required_total else "FAIL",
            {"field_pass_total": field_pass_total, "field_required_total": field_required_total},
            "Complete every queued field/color/wavelength/CRA point with grid-resolution PASS.",
        ),
        check_row(
            "finite_array_crosstalk_mesh_coverage",
            crosstalk_required_total > 0 and crosstalk_pass_total == crosstalk_required_total,
            "PASS" if crosstalk_required_total > 0 and crosstalk_pass_total == crosstalk_required_total else "FAIL",
            {"crosstalk_pass_total": crosstalk_pass_total, "crosstalk_required_total": crosstalk_required_total},
            "Run finite-array crosstalk convergence jobs with sufficient guard pixels and resolution.",
        ),
    ]
    non_pass_checks = [row for row in check_rows if not boolish(row["pass"])]
    error_count = sum(1 for row in non_pass_checks if row["check_id"] in {"quantitative_coverage_present", "runtime_bundle_present"})
    warning_count = len(non_pass_checks) - error_count

    report_json = output_dir / "camera_e2e_mesh_confidence.json"
    sensor_csv = output_dir / "camera_e2e_mesh_confidence_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_mesh_confidence_by_domain.csv"
    checks_csv = output_dir / "camera_e2e_mesh_confidence_checks.csv"
    html_path = output_dir / "index.html"

    payload = {
        "schema": "camera_e2e_mesh_confidence_audit_v1",
        "artifact_role": "camera_e2e_mesh_resolution_confidence_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "status": status,
        "validation": {
            "schema": "camera_e2e_mesh_confidence_validation_v1",
            "pass": error_count == 0,
            "status": status if error_count == 0 else "FAIL",
            "issue_count": len(non_pass_checks),
            "error_count": error_count,
            "warning_count": warning_count,
            "issues": non_pass_checks,
        },
        "field_pass_total": field_pass_total,
        "field_required_total": field_required_total,
        "field_pass_fraction": field_pass_total / field_required_total if field_required_total else 0.0,
        "crosstalk_pass_total": crosstalk_pass_total,
        "crosstalk_required_total": crosstalk_required_total,
        "crosstalk_pass_fraction": crosstalk_pass_total / crosstalk_required_total if crosstalk_required_total else 0.0,
        "confidence_class_counts": class_counts,
        "product_ready_count": product_ready_count,
        "use_policy": {
            "research": "Allowed for CameraE2E sensitivity/prototyping only when row gates and confidence classes are preserved.",
            "product": "Blocked until full quantitative field coverage, finite-array crosstalk convergence, measured stack/n,k, measured CRA/ML shift, and electrical/readout calibration pass.",
        },
        "outputs": {
            "json": repo_rel(report_json),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_csv(checks_csv, check_rows, CHECK_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, check_rows, sensor_rows, domain_rows)
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
    check_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]],
    domain_rows: list[dict[str, Any]],
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
    status_class = "pass" if payload.get("product_ready_count") else "warn"
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Mesh Confidence Audit</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Mesh Confidence Audit</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This report estimates trust level from quantitative solver coverage and grid-resolution gates.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(payload.get("status", ""))}</div><div class="muted">mesh confidence status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("field_pass_total", 0))}/{html_cell(payload.get("field_required_total", 0))}</div><div class="muted">field mesh PASS points</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("crosstalk_pass_total", 0))}/{html_cell(payload.get("crosstalk_required_total", 0))}</div><div class="muted">finite-array crosstalk PASS points</div></div>
</div>
<h2>Interpretation</h2>
<p>Current PASS means the confidence audit ran and product gating is preserved. It does not mean product numerical accuracy. Full product use requires all field and finite-array crosstalk points to pass plus measured stack/material/CRA/electrical calibration.</p>
<h2>Checks</h2>{html_table(check_rows, CHECK_COLUMNS)}
<h2>Per-Sensor Mesh Confidence</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Domain Confidence</h2>{html_table(domain_rows, DOMAIN_COLUMNS)}
<h2>Payload</h2><pre><code>{html_cell(json.dumps({k:v for k,v in payload.items() if k not in {'outputs'}}, indent=2, ensure_ascii=False))}</code></pre>
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
    outputs["camera_e2e_mesh_confidence_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_mesh_confidence_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_mesh_confidence_domain_csv"] = payload["outputs"]["domain_csv"]
    outputs["camera_e2e_mesh_confidence_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_mesh_confidence_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_mesh_confidence"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "status": payload["status"],
        "field_pass_total": payload["field_pass_total"],
        "field_required_total": payload["field_required_total"],
        "crosstalk_pass_total": payload["crosstalk_pass_total"],
        "crosstalk_required_total": payload["crosstalk_required_total"],
        "confidence_class_counts": payload["confidence_class_counts"],
        "product_ready_count": payload["product_ready_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_audit(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "status": payload["status"],
                "field_pass_total": payload["field_pass_total"],
                "field_required_total": payload["field_required_total"],
                "crosstalk_pass_total": payload["crosstalk_pass_total"],
                "crosstalk_required_total": payload["crosstalk_required_total"],
                "confidence_class_counts": payload["confidence_class_counts"],
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
