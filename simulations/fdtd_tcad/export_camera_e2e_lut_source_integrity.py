#!/usr/bin/env python3
"""Export a joined source-integrity matrix for CameraE2E LUT handoff.

The package has separate coverage, method-provenance, and uncertainty artifacts.
This exporter joins them into one requirement-level table so a CameraE2E
consumer can see whether a value is solver-derived, external/local DB based, or
proxy/prior, and which uncertainty band must travel with it.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_lut_source_integrity"

MATRIX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "requirement_id",
    "requirement",
    "camera_e2e_use",
    "recommended_camera_e2e_use",
    "lut_source_class",
    "calculation_method",
    "source_priority",
    "solver_dependency",
    "external_info_dependency",
    "proxy_dependency",
    "structure_specialization",
    "primary_uncertainty_quantity",
    "primary_uncertainty_min",
    "primary_uncertainty_max",
    "primary_uncertainty_unit",
    "uncertainty_camera_e2e_use",
    "uncertainty_product_gate",
    "research_gate",
    "product_gate",
    "method_gate",
    "trust_class",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "row_count",
    "field_pass_points",
    "field_required_points",
    "crosstalk_pass_points",
    "crosstalk_required_points",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "mesh_confidence_class",
    "source_artifacts",
    "primary_blocker",
    "uncertainty_primary_blockers",
    "not_valid_for",
    "next_action",
    "source_integrity_gate",
]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "requirement_count",
    "source_class_counts",
    "research_gate_counts",
    "product_gate_counts",
    "source_integrity_gate_counts",
    "field_pass_points",
    "field_required_points",
    "crosstalk_pass_points",
    "crosstalk_required_points",
    "uncertainty_product_gate",
    "product_ready",
    "first_blocker",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]


UNCERTAINTY_MAP = {
    "optical_material_nk_ri": ("material RI n / CFA k-transmission", "material_ri_n_uncertainty_pct_min", "material_ri_n_uncertainty_pct_max", "pct"),
    "spectral_response_qe": ("QE absolute response", "qe_absolute_uncertainty_pct_min", "qe_absolute_uncertainty_pct_max", "pct"),
    "color_response_matrix": ("CFA k / color-response proxy", "cfa_k_transmission_uncertainty_pct_min", "cfa_k_transmission_uncertainty_pct_max", "pct"),
    "optical_crosstalk_kernel": ("optical crosstalk kernel", "optical_crosstalk_uncertainty_pct_min", "optical_crosstalk_uncertainty_pct_max", "pct"),
    "angular_cra_response": ("CRA edge response", "cra_edge_response_uncertainty_pct_min", "cra_edge_response_uncertainty_pct_max", "pct"),
    "microlens_ocl_shift_map": ("CRA / OCL shift response", "cra_edge_response_uncertainty_pct_min", "cra_edge_response_uncertainty_pct_max", "pct"),
    "conversion_gain_fwc_saturation_nonlinearity": ("conversion gain / FWC", "conversion_gain_fwc_uncertainty_pct_min", "conversion_gain_fwc_uncertainty_pct_max", "pct"),
    "dark_current_temperature_exposure": ("dark current", "dark_current_uncertainty_factor_min", "dark_current_uncertainty_factor_max", "factor"),
    "dsnu_prnu": ("DSNU / PRNU", "dsnu_prnu_uncertainty_pct_min", "dsnu_prnu_uncertainty_pct_max", "pct"),
    "temporal_noise": ("temporal/read/reset noise", "temporal_noise_uncertainty_pct_min", "temporal_noise_uncertainty_pct_max", "pct"),
    "charge_collection_electrical_crosstalk": ("electrical crosstalk prior", "conversion_gain_fwc_uncertainty_pct_min", "conversion_gain_fwc_uncertainty_pct_max", "pct"),
    "analog_digital_gain": ("readout RAW table", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "black_level_optical_black": ("readout RAW table", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "adc_clipping_quantization": ("readout RAW table", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "row_column_fpn_timing_direction": ("readout RAW / FPN", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "defect_hot_pixel_stats": ("readout RAW defects", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "binning_remosaic_modes": ("binning/remosaic readout", "readout_raw_uncertainty_pct_min", "readout_raw_uncertainty_pct_max", "pct"),
    "lens_raytrace_field_cra_map": ("module coupling", "module_coupling_uncertainty_pct_min", "module_coupling_uncertainty_pct_max", "pct"),
    "sensor_position_tilt_decenter": ("module coupling", "module_coupling_uncertainty_pct_min", "module_coupling_uncertainty_pct_max", "pct"),
    "vignetting_shading": ("module coupling", "module_coupling_uncertainty_pct_min", "module_coupling_uncertainty_pct_max", "pct"),
    "wavelength_dependent_cra_pupil": ("module coupling", "module_coupling_uncertainty_pct_min", "module_coupling_uncertainty_pct_max", "pct"),
}


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


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key)}


def index_by_pair(rows: list[dict[str, str]], key_a: str, key_b: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get(key_a, ""), row.get(key_b, "")): row for row in rows if row.get(key_a) and row.get(key_b)}


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


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def uncertainty_for(requirement_id: str, uncertainty: dict[str, str]) -> dict[str, str]:
    label, min_key, max_key, unit = UNCERTAINTY_MAP.get(requirement_id, ("", "", "", ""))
    return {
        "primary_uncertainty_quantity": label,
        "primary_uncertainty_min": uncertainty.get(min_key, ""),
        "primary_uncertainty_max": uncertainty.get(max_key, ""),
        "primary_uncertainty_unit": unit,
    }


def source_integrity_gate(row: dict[str, Any]) -> str:
    if row.get("research_gate") == "N/A":
        return "N/A"
    required = [
        "lut_source_class",
        "calculation_method",
        "source_priority",
        "primary_uncertainty_quantity",
        "primary_uncertainty_min",
        "primary_uncertainty_max",
        "uncertainty_product_gate",
    ]
    if any(not str(row.get(key, "")).strip() for key in required):
        return "FAIL"
    if str(row.get("product_gate", "")).upper() == "PASS":
        return "PASS"
    return "CHECK"


def build_rows(package_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    method_by_key = index_by_pair(
        read_csv_rows(package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_matrix.csv"),
        "slug",
        "requirement_id",
    )
    uncertainty_by_slug = index_by(
        read_csv_rows(package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_by_sensor.csv"),
        "slug",
    )

    matrix_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for coverage in coverage_rows:
        slug = coverage.get("slug", "")
        requirement_id = coverage.get("requirement_id", "")
        method = method_by_key.get((slug, requirement_id), {})
        uncertainty = uncertainty_by_slug.get(slug, {})
        row = {
            "slug": slug,
            "code": coverage.get("code", ""),
            "manufacturer": coverage.get("manufacturer", ""),
            "device_name": coverage.get("device_name", ""),
            "domain": coverage.get("domain", ""),
            "requirement_id": requirement_id,
            "requirement": coverage.get("requirement", ""),
            "camera_e2e_use": coverage.get("camera_e2e_use", ""),
            "recommended_camera_e2e_use": method.get("recommended_camera_e2e_use", ""),
            "lut_source_class": method.get("lut_source_class", ""),
            "calculation_method": method.get("calculation_method", ""),
            "source_priority": method.get("source_priority", ""),
            "solver_dependency": method.get("solver_dependency", ""),
            "external_info_dependency": method.get("external_info_dependency", ""),
            "proxy_dependency": method.get("proxy_dependency", ""),
            "structure_specialization": method.get("structure_specialization", ""),
            **uncertainty_for(requirement_id, uncertainty),
            "uncertainty_camera_e2e_use": uncertainty.get("camera_e2e_use", ""),
            "uncertainty_product_gate": uncertainty.get("uncertainty_product_gate", ""),
            "research_gate": coverage.get("research_gate", ""),
            "product_gate": coverage.get("product_gate", ""),
            "method_gate": method.get("method_gate", ""),
            "trust_class": method.get("trust_class", ""),
            "research_usability_score_0_100": method.get("research_usability_score_0_100", ""),
            "evidence_confidence_score_0_100": method.get("evidence_confidence_score_0_100", ""),
            "product_calibration_score_0_100": method.get("product_calibration_score_0_100", ""),
            "row_count": coverage.get("row_count", ""),
            "field_pass_points": method.get("field_pass_points", ""),
            "field_required_points": method.get("field_required_points", ""),
            "crosstalk_pass_points": method.get("crosstalk_pass_points", ""),
            "crosstalk_required_points": method.get("crosstalk_required_points", ""),
            "cfa_provenance_class": method.get("cfa_provenance_class", ""),
            "cfa_assumption_gate": method.get("cfa_assumption_gate", ""),
            "mesh_confidence_class": method.get("mesh_confidence_class", ""),
            "source_artifacts": coverage.get("source_artifacts", ""),
            "primary_blocker": coverage.get("primary_blocker", ""),
            "uncertainty_primary_blockers": uncertainty.get("primary_blockers", ""),
            "not_valid_for": method.get("not_valid_for", ""),
            "next_action": method.get("next_action", coverage.get("primary_blocker", "")),
        }
        row["source_integrity_gate"] = source_integrity_gate(row)
        matrix_rows.append(row)
        grouped[slug].append(row)

    sensor_rows: list[dict[str, Any]] = []
    for slug, rows in sorted(grouped.items()):
        first = rows[0]
        product_ready = all(row.get("product_gate") in {"PASS", "N/A"} for row in rows)
        blockers = [row.get("primary_blocker", "") for row in rows if str(row.get("primary_blocker", "")).strip()]
        sensor_rows.append(
            {
                "slug": slug,
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "requirement_count": len(rows),
                "source_class_counts": json.dumps(dict(sorted(Counter(row.get("lut_source_class", "") for row in rows).items())), sort_keys=True),
                "research_gate_counts": json.dumps(dict(sorted(Counter(row.get("research_gate", "") for row in rows).items())), sort_keys=True),
                "product_gate_counts": json.dumps(dict(sorted(Counter(row.get("product_gate", "") for row in rows).items())), sort_keys=True),
                "source_integrity_gate_counts": json.dumps(dict(sorted(Counter(row.get("source_integrity_gate", "") for row in rows).items())), sort_keys=True),
                "field_pass_points": first.get("field_pass_points", ""),
                "field_required_points": first.get("field_required_points", ""),
                "crosstalk_pass_points": first.get("crosstalk_pass_points", ""),
                "crosstalk_required_points": first.get("crosstalk_required_points", ""),
                "uncertainty_product_gate": first.get("uncertainty_product_gate", ""),
                "product_ready": product_ready,
                "first_blocker": blockers[0] if blockers else "",
            }
        )
    return matrix_rows, sensor_rows


def write_html(path: Path, payload: dict[str, Any], matrix_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]]) -> None:
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
        "domain",
        "requirement_id",
        "lut_source_class",
        "primary_uncertainty_quantity",
        "primary_uncertainty_min",
        "primary_uncertainty_max",
        "research_gate",
        "product_gate",
        "source_integrity_gate",
    ]
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E LUT Source Integrity</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E LUT Source Integrity</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This joins requirement coverage, method provenance, and uncertainty into one handoff table.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("matrix_row_count", 0))}</div><div class="muted">requirement rows</div></div>
    <div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
  </div>
  <h2>Source Classes</h2>
  <p><code>{html_cell(payload.get("source_class_counts", {}))}</code></p>
  <h2>Sensor Summary</h2>
  {html_table(sensor_rows, SENSOR_COLUMNS)}
  <h2>Requirement Preview</h2>
  {html_table(matrix_rows, preview_cols)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_lut_source_integrity_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_lut_source_integrity_matrix_csv"] = payload["outputs"]["matrix_csv"]
    outputs["camera_e2e_lut_source_integrity_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_lut_source_integrity_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_lut_source_integrity"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "status": payload["validation"]["status"],
        "sensor_count": payload["sensor_count"],
        "matrix_row_count": payload["matrix_row_count"],
        "product_ready_count": payload["product_ready_count"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def export_source_integrity(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    matrix_rows, sensor_rows = build_rows(package_dir)
    coverage_count = len(read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv"))
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    integrity_fail_count = sum(1 for row in matrix_rows if row.get("source_integrity_gate") == "FAIL")
    missing_uncertainty_count = sum(
        1
        for row in matrix_rows
        if row.get("research_gate") != "N/A"
        and (not row.get("primary_uncertainty_min") or not row.get("primary_uncertainty_max"))
    )
    checks = [
        check_row(
            "coverage_method_uncertainty_join_complete",
            bool(matrix_rows) and len(matrix_rows) == coverage_count,
            "PASS" if bool(matrix_rows) and len(matrix_rows) == coverage_count else "FAIL",
            {"matrix_rows": len(matrix_rows), "coverage_rows": coverage_count},
            "Regenerate coverage, method provenance, and uncertainty budget.",
        ),
        check_row(
            "source_integrity_no_fail_rows",
            integrity_fail_count == 0,
            "PASS" if integrity_fail_count == 0 else "FAIL",
            {"source_integrity_fail_count": integrity_fail_count},
            "Every applicable row needs source class, method, and uncertainty values.",
        ),
        check_row(
            "uncertainty_attached",
            missing_uncertainty_count == 0,
            "PASS" if missing_uncertainty_count == 0 else "FAIL",
            {"missing_uncertainty_count": missing_uncertainty_count},
            "Attach the sensor-level uncertainty budget to every applicable requirement.",
        ),
        check_row(
            "product_use_blocked",
            product_ready_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_ready_count == 0 else "FAIL",
            {"product_ready_count": product_ready_count},
            "Do not promote source-integrity rows to product use before measured data and solver convergence pass.",
        ),
    ]
    pass_all = all(boolish(row["pass"]) for row in checks)
    status = "SOURCE_INTEGRITY_READY_PRODUCT_BLOCKED" if pass_all else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_csv = output_dir / "camera_e2e_lut_source_integrity_matrix.csv"
    sensor_csv = output_dir / "camera_e2e_lut_source_integrity_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_lut_source_integrity_checks.csv"
    report_json = output_dir / "camera_e2e_lut_source_integrity.json"
    html_path = output_dir / "index.html"
    write_csv(matrix_csv, matrix_rows, MATRIX_COLUMNS)
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    payload = {
        "schema": "camera_e2e_lut_source_integrity_v1",
        "artifact_role": "requirement_source_method_uncertainty_join",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "matrix_row_count": len(matrix_rows),
        "product_ready_count": product_ready_count,
        "source_class_counts": dict(sorted(Counter(row.get("lut_source_class", "") for row in matrix_rows).items())),
        "source_integrity_gate_counts": dict(sorted(Counter(row.get("source_integrity_gate", "") for row in matrix_rows).items())),
        "uncertainty_product_gate_counts": dict(sorted(Counter(row.get("uncertainty_product_gate", "") for row in matrix_rows).items())),
        "validation": {
            "schema": "camera_e2e_lut_source_integrity_validation_v1",
            "pass": pass_all,
            "status": status,
            "checks": checks,
        },
        "outputs": {
            "json": repo_rel(report_json),
            "matrix_csv": repo_rel(matrix_csv),
            "sensor_csv": repo_rel(sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research": "Use these rows to route research/trend LUT data while preserving source and uncertainty.",
            "product": "Product use remains blocked until measured stack/material/CRA/electrical/readout/module data and solver convergence gates pass.",
        },
    }
    write_json(report_json, payload)
    write_html(html_path, payload, matrix_rows, sensor_rows)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    payload = export_source_integrity(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": payload["validation"]["status"],
                "matrix_row_count": payload["matrix_row_count"],
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
