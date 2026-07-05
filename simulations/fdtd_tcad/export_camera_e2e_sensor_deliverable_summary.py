#!/usr/bin/env python3
"""Export a per-sensor CameraE2E deliverable summary.

This is a loader-facing one-row-per-sensor index. It does not introduce new
physics. It joins the validated flat bundle, consumer bundle, source-integrity
matrix, analysis report, uncertainty budget, and objective acceptance audit so a
CameraE2E integrator can choose a sensor and immediately see what can be loaded,
what the values are based on, and why product use remains blocked.
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


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_sensor_deliverable_summary"

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "effective_ocl_mode",
    "recommended_loader",
    "flat_sensor_json",
    "consumer_manifest_json",
    "camera_e2e_use_scope",
    "recommended_camera_e2e_use",
    "deliverable_gate",
    "product_ready",
    "product_gate",
    "source_integrity_requirement_count",
    "source_integrity_gate_counts",
    "source_class_counts",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "field_mesh_pass_fraction",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "crosstalk_mesh_pass_fraction",
    "runtime_row_count",
    "kernel_row_count",
    "spectral_row_count",
    "material_row_count",
    "cfa_db_row_count",
    "cfa_db_transmission_row_count",
    "electrical_row_count",
    "readout_row_count",
    "binning_row_count",
    "module_field_row_count",
    "source_integrity_row_count",
    "response_example_row_count",
    "method_provenance_row_count",
    "query_row_count",
    "allowed_query_count",
    "edge_to_center_min",
    "edge_to_center_max",
    "max_output_crosstalk_fraction",
    "max_strongest_neighbor_fraction",
    "mean_signal_e",
    "mean_raw_dn_clipped",
    "min_snr_db",
    "max_snr_db",
    "qe_uncertainty_pct_min",
    "qe_uncertainty_pct_max",
    "cra_uncertainty_pct_min",
    "cra_uncertainty_pct_max",
    "optical_crosstalk_uncertainty_pct_min",
    "optical_crosstalk_uncertainty_pct_max",
    "temporal_noise_uncertainty_pct_min",
    "temporal_noise_uncertainty_pct_max",
    "readout_raw_uncertainty_pct_min",
    "readout_raw_uncertainty_pct_max",
    "module_coupling_uncertainty_pct_min",
    "module_coupling_uncertainty_pct_max",
    "uncertainty_product_gate",
    "crosstalk_support_status",
    "crosstalk_support_recommended_kernel",
    "crosstalk_support_summary",
    "primary_blockers",
    "required_before_product_use",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def safe_int(value: Any) -> int:
    try:
        if value in ("", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key)}


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, sort_keys=True, ensure_ascii=False) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


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


def build_rows(package_dir: Path) -> list[dict[str, Any]]:
    flat_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv")
    consumer_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv"), "slug")
    source_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_by_sensor.csv"), "slug")
    analysis_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_analysis_report" / "camera_e2e_analysis_by_sensor.csv"), "slug")
    uncertainty_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_by_sensor.csv"), "slug")
    acceptance_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_objective_acceptance" / "camera_e2e_objective_acceptance_sensors.csv"), "slug")

    rows: list[dict[str, Any]] = []
    for flat in flat_rows:
        slug = flat.get("slug", "")
        consumer = consumer_by_slug.get(slug, {})
        source = source_by_slug.get(slug, {})
        analysis = analysis_by_slug.get(slug, {})
        uncertainty = uncertainty_by_slug.get(slug, {})
        acceptance = acceptance_by_slug.get(slug, {})
        product_ready = boolish(flat.get("product_ready")) or boolish(acceptance.get("product_ready"))
        deliverable_gate = "PASS" if not product_ready and safe_int(flat.get("source_integrity_row_count")) > 0 and safe_int(flat.get("runtime_row_count")) > 0 else "FAIL"
        rows.append(
            {
                "slug": slug,
                "code": flat.get("code", ""),
                "manufacturer": flat.get("manufacturer", ""),
                "device_name": flat.get("device_name", ""),
                "pixel_pitch_um": flat.get("pixel_pitch_um", ""),
                "cfa_pattern": flat.get("cfa_pattern", ""),
                "effective_ocl_mode": flat.get("effective_ocl_mode", ""),
                "recommended_loader": "flat_sensor_json for one-sensor load; consumer_manifest_json for table joins",
                "flat_sensor_json": flat.get("flat_sensor_json", ""),
                "consumer_manifest_json": consumer.get("load_manifest_json", ""),
                "camera_e2e_use_scope": flat.get("camera_e2e_use_scope", consumer.get("camera_e2e_recommended_use", "")),
                "recommended_camera_e2e_use": analysis.get("recommended_camera_e2e_use", consumer.get("camera_e2e_recommended_use", "")),
                "deliverable_gate": deliverable_gate,
                "product_ready": product_ready,
                "product_gate": flat.get("product_gate", consumer.get("product_bundle_gate", "")),
                "source_integrity_requirement_count": source.get("requirement_count", ""),
                "source_integrity_gate_counts": source.get("source_integrity_gate_counts", ""),
                "source_class_counts": source.get("source_class_counts", ""),
                "coverage_research_gate_counts": acceptance.get("coverage_research_gate_counts", consumer.get("coverage_research_gate_counts", "")),
                "coverage_product_gate_counts": acceptance.get("coverage_product_gate_counts", consumer.get("coverage_product_gate_counts", "")),
                "field_mesh_pass_points": flat.get("field_mesh_pass_points", analysis.get("field_mesh_pass_points", "")),
                "field_mesh_required_points": flat.get("field_mesh_required_points", analysis.get("field_mesh_required_points", "")),
                "field_mesh_pass_fraction": analysis.get("field_mesh_pass_fraction", acceptance.get("field_mesh_pass_fraction", "")),
                "crosstalk_mesh_pass_points": source.get("crosstalk_pass_points", consumer.get("mesh_crosstalk_pass_points", "")),
                "crosstalk_mesh_required_points": source.get("crosstalk_required_points", consumer.get("mesh_crosstalk_required_points", "")),
                "crosstalk_mesh_pass_fraction": analysis.get("crosstalk_mesh_pass_fraction", acceptance.get("crosstalk_mesh_pass_fraction", "")),
                "runtime_row_count": flat.get("runtime_row_count", ""),
                "kernel_row_count": flat.get("kernel_row_count", ""),
                "spectral_row_count": flat.get("spectral_row_count", ""),
                "material_row_count": flat.get("material_row_count", ""),
                "cfa_db_row_count": flat.get("cfa_db_row_count", ""),
                "cfa_db_transmission_row_count": flat.get("cfa_db_transmission_row_count", ""),
                "electrical_row_count": flat.get("electrical_row_count", ""),
                "readout_row_count": flat.get("readout_row_count", ""),
                "binning_row_count": flat.get("binning_row_count", ""),
                "module_field_row_count": flat.get("module_field_row_count", ""),
                "source_integrity_row_count": flat.get("source_integrity_row_count", ""),
                "response_example_row_count": flat.get("response_example_row_count", ""),
                "method_provenance_row_count": flat.get("method_provenance_row_count", ""),
                "query_row_count": analysis.get("query_row_count", ""),
                "allowed_query_count": analysis.get("allowed_query_count", ""),
                "edge_to_center_min": analysis.get("edge_to_center_min", ""),
                "edge_to_center_max": analysis.get("edge_to_center_max", ""),
                "max_output_crosstalk_fraction": analysis.get("max_output_crosstalk_fraction", ""),
                "max_strongest_neighbor_fraction": analysis.get("max_strongest_neighbor_fraction", ""),
                "mean_signal_e": analysis.get("mean_signal_e", ""),
                "mean_raw_dn_clipped": analysis.get("mean_raw_dn_clipped", ""),
                "min_snr_db": analysis.get("min_snr_db", ""),
                "max_snr_db": analysis.get("max_snr_db", ""),
                "qe_uncertainty_pct_min": uncertainty.get("qe_absolute_uncertainty_pct_min", ""),
                "qe_uncertainty_pct_max": uncertainty.get("qe_absolute_uncertainty_pct_max", ""),
                "cra_uncertainty_pct_min": uncertainty.get("cra_edge_response_uncertainty_pct_min", ""),
                "cra_uncertainty_pct_max": uncertainty.get("cra_edge_response_uncertainty_pct_max", ""),
                "optical_crosstalk_uncertainty_pct_min": uncertainty.get("optical_crosstalk_uncertainty_pct_min", ""),
                "optical_crosstalk_uncertainty_pct_max": uncertainty.get("optical_crosstalk_uncertainty_pct_max", ""),
                "temporal_noise_uncertainty_pct_min": uncertainty.get("temporal_noise_uncertainty_pct_min", ""),
                "temporal_noise_uncertainty_pct_max": uncertainty.get("temporal_noise_uncertainty_pct_max", ""),
                "readout_raw_uncertainty_pct_min": uncertainty.get("readout_raw_uncertainty_pct_min", ""),
                "readout_raw_uncertainty_pct_max": uncertainty.get("readout_raw_uncertainty_pct_max", ""),
                "module_coupling_uncertainty_pct_min": uncertainty.get("module_coupling_uncertainty_pct_min", ""),
                "module_coupling_uncertainty_pct_max": uncertainty.get("module_coupling_uncertainty_pct_max", ""),
                "uncertainty_product_gate": flat.get("uncertainty_product_gate", uncertainty.get("uncertainty_product_gate", "")),
                "crosstalk_support_status": flat.get("crosstalk_support_status", consumer.get("lut_trust_crosstalk_support_status", "")),
                "crosstalk_support_recommended_kernel": flat.get("crosstalk_support_recommended_kernel", consumer.get("lut_trust_crosstalk_support_recommended_kernel", "")),
                "crosstalk_support_summary": flat.get("crosstalk_support_summary", consumer.get("crosstalk_support_summary", "")),
                "primary_blockers": flat.get("primary_blockers", acceptance.get("primary_blockers", "")),
                "required_before_product_use": analysis.get("required_before_product_use", ""),
            }
        )
    return rows


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    preview_cols = [
        "device_name",
        "pixel_pitch_um",
        "effective_ocl_mode",
        "camera_e2e_use_scope",
        "deliverable_gate",
        "product_gate",
        "source_integrity_requirement_count",
        "field_mesh_pass_fraction",
        "crosstalk_mesh_pass_fraction",
        "qe_uncertainty_pct_max",
        "primary_blockers",
    ]
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Sensor Deliverable Summary</title>
<style>{css}</style>
</head>
<body><main>
<h1>CameraE2E Sensor Deliverable Summary</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. One row per sensor for CameraE2E handoff. Product use remains blocked unless product gates pass.</p>
<div class="grid">
  <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
  <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
  <div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
  <div class="card"><div class="metric">{html_cell(payload.get("deliverable_gate_counts", {}))}</div><div class="muted">deliverable gates</div></div>
</div>
<h2>Sensor Rows</h2>
{html_table(rows, preview_cols)}
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
    outputs["camera_e2e_sensor_deliverable_summary_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_sensor_deliverable_summary_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_sensor_deliverable_summary_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_sensor_deliverable_summary"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "status": payload["validation"]["status"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        **payload["outputs"],
    }
    write_json(package_path, package)


def export_deliverable(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = build_rows(package_dir)
    product_ready_count = sum(1 for row in rows if boolish(row.get("product_ready")))
    missing_loader_count = sum(1 for row in rows if not row.get("flat_sensor_json") or not row.get("consumer_manifest_json"))
    missing_integrity_count = sum(1 for row in rows if safe_int(row.get("source_integrity_requirement_count")) <= 0)
    missing_allowed_query_count = sum(1 for row in rows if safe_int(row.get("allowed_query_count")) <= 0)
    checks = [
        check_row("deliverable_rows_present", bool(rows), "PASS" if rows else "FAIL", {"row_count": len(rows)}, "Regenerate flat and consumer bundles."),
        check_row(
            "loader_paths_present",
            missing_loader_count == 0,
            "PASS" if missing_loader_count == 0 else "FAIL",
            {"missing_loader_count": missing_loader_count},
            "Every sensor needs flat_sensor_json and consumer_manifest_json.",
        ),
        check_row(
            "source_integrity_present",
            missing_integrity_count == 0,
            "PASS" if missing_integrity_count == 0 else "FAIL",
            {"missing_integrity_count": missing_integrity_count},
            "Regenerate source-integrity matrix and flat bundle.",
        ),
        check_row(
            "research_query_available",
            missing_allowed_query_count == 0,
            "PASS" if missing_allowed_query_count == 0 else "FAIL",
            {"missing_allowed_query_count": missing_allowed_query_count},
            "Run flat sensor research query and analysis report.",
        ),
        check_row(
            "product_blocked",
            product_ready_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_ready_count == 0 else "FAIL",
            {"product_ready_count": product_ready_count},
            "Do not mark product-ready until measured/calibrated gates pass.",
        ),
    ]
    pass_all = all(boolish(row["pass"]) for row in checks)
    status = "SENSOR_DELIVERABLE_READY_PRODUCT_BLOCKED" if pass_all else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "camera_e2e_sensor_deliverable_summary.csv"
    checks_csv = output_dir / "camera_e2e_sensor_deliverable_checks.csv"
    report_json = output_dir / "camera_e2e_sensor_deliverable_summary.json"
    html_path = output_dir / "index.html"
    write_csv(summary_csv, rows, SUMMARY_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    payload = {
        "schema": "camera_e2e_sensor_deliverable_summary_v1",
        "artifact_role": "per_sensor_camera_e2e_deliverable_index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(rows),
        "product_ready_count": product_ready_count,
        "deliverable_gate_counts": dict(sorted(Counter(row.get("deliverable_gate", "") for row in rows).items())),
        "use_scope_counts": dict(sorted(Counter(row.get("camera_e2e_use_scope", "") for row in rows).items())),
        "validation": {
            "schema": "camera_e2e_sensor_deliverable_summary_validation_v1",
            "pass": pass_all,
            "status": status,
            "checks": checks,
        },
        "outputs": {
            "json": repo_rel(report_json),
            "summary_csv": repo_rel(summary_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research": "Use this as the first per-sensor selector for CameraE2E research/trend loading.",
            "product": "Product use remains blocked until product_ready is true and all product gates pass.",
        },
    }
    write_json(report_json, payload)
    write_html(html_path, payload, rows)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    payload = export_deliverable(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": payload["validation"]["status"],
                "sensor_count": payload["sensor_count"],
                "product_ready_count": payload["product_ready_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    return 0 if payload["validation"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
