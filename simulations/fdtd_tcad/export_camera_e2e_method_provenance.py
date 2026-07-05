#!/usr/bin/env python3
"""Export a method/source provenance matrix for CameraE2E LUT rows.

This artifact answers a practical integration question: for each sensor and
objective requirement, which values come from solver output, which come from
external/local sensor DB evidence, and which are proxy/prior estimates. It does
not change the LUT values. It makes the source hierarchy explicit so CameraE2E
can route the data without treating research priors as calibrated product data.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_method_provenance"

MATRIX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "requirement_id",
    "requirement",
    "camera_e2e_use",
    "lut_source_class",
    "calculation_method",
    "source_priority",
    "solver_dependency",
    "external_info_dependency",
    "proxy_dependency",
    "structure_specialization",
    "recommended_camera_e2e_use",
    "not_valid_for",
    "research_gate",
    "product_gate",
    "trust_class",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "coverage_row_count",
    "field_pass_points",
    "field_required_points",
    "crosstalk_pass_points",
    "crosstalk_required_points",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "mesh_confidence_class",
    "source_artifacts",
    "primary_blocker",
    "next_action",
    "method_gate",
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
    "method_gate_counts",
    "field_pass_points",
    "field_required_points",
    "crosstalk_pass_points",
    "crosstalk_required_points",
    "cfa_provenance_class",
    "mesh_confidence_class",
    "product_ready",
    "primary_method_note",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]

NOT_VALID_FOR = "calibrated product QE/color/crosstalk/noise/readout/module sign-off"


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def method_for_requirement(requirement_id: str, mesh: dict[str, str], cfa: dict[str, str]) -> dict[str, str]:
    field_pass = safe_int(mesh.get("field_pass_points"))
    field_required = safe_int(mesh.get("field_required_points"))
    crosstalk_pass = safe_int(mesh.get("crosstalk_pass_points"))
    crosstalk_required = safe_int(mesh.get("crosstalk_required_points"))
    cfa_class = cfa.get("cfa_provenance_class", "")

    if requirement_id == "optical_material_nk_ri":
        return {
            "lut_source_class": "external_db_plus_proxy_material",
            "calculation_method": "image_sensor_db optical/CFA stack rows joined with proxy CFA/OCL/passivation/Si n,k tables",
            "source_priority": "measured material table > sensor-specific extracted/derived DB > public/proxy material library",
            "solver_dependency": "FDTD consumes these rows; values are not solver-fitted",
            "external_info_dependency": "TechInsights/local image_sensor_db stack and CFA metadata",
            "proxy_dependency": "CFA/OCL/passivation proxy n,k where measured values are absent",
            "structure_specialization": "pixel pitch, CFA pattern, OCL mode, stack thickness if available",
        }
    if requirement_id == "spectral_response_qe":
        solver_text = f"partial Meep field anchors {field_pass}/{field_required}" if field_pass else "no converged field solver anchor for most rows"
        return {
            "lut_source_class": "solver_anchor_plus_cfa_si_proxy" if field_pass else "cfa_si_proxy_response_prior",
            "calculation_method": "runtime response from available FDTD field anchors or design field rolloff, cross-checked by CFA transmission x simple Si absorption",
            "source_priority": "converged Meep field point > merged low-res solver point > CFA x Si response prior normalized to runtime anchor",
            "solver_dependency": solver_text,
            "external_info_dependency": "image_sensor_db CFA/stack metadata and public/proxy Si n,k table",
            "proxy_dependency": "CFA transmission proxy, simple 1D Si absorption, runtime normalization where measured QE is missing",
            "structure_specialization": "field position, CRA, OCL mode, CFA channel, wavelength",
        }
    if requirement_id == "color_response_matrix":
        return {
            "lut_source_class": "derived_color_seed",
            "calculation_method": "RGB spectral response seed integrated to an equal-energy RGB-to-XYZ/CCM starting matrix",
            "source_priority": "measured spectral response/CCM target > runtime spectral response > CFA x Si proxy seed",
            "solver_dependency": "inherits field/QE solver availability indirectly",
            "external_info_dependency": "CFA provenance and sensor color/mono classification",
            "proxy_dependency": "generic RGB fallback when CFA pattern or measured spectra are absent",
            "structure_specialization": cfa_class or "CFA/color topology if known",
        }
    if requirement_id == "optical_crosstalk_kernel":
        return {
            "lut_source_class": "low_res_solver_or_compact_surrogate",
            "calculation_method": "finite-array crosstalk pilots where available plus compact kernel/surrogate rows for CameraE2E ingestion",
            "source_priority": "product-resolution finite-array Meep > low-res support pilot > topology compact surrogate",
            "solver_dependency": f"finite-array crosstalk pass {crosstalk_pass}/{crosstalk_required}; support pilots may still fail product gate",
            "external_info_dependency": "pixel pitch, OCL/binning topology, DTI/stack priors",
            "proxy_dependency": "compact kernel/surrogate for missing full-domain sweeps",
            "structure_specialization": "OCL/binning group support, neighborhood size, boundary truncation",
        }
    if requirement_id == "angular_cra_response":
        return {
            "lut_source_class": "design_cra_prior_plus_partial_solver",
            "calculation_method": "field/CRA response LUT from design CRA cases, runtime rolloff, and any merged FDTD field anchors",
            "source_priority": "lens raytrace CRA map + converged Meep > design CRA anchors > analytic rolloff prior",
            "solver_dependency": f"field pass {field_pass}/{field_required}",
            "external_info_dependency": "module/lens CRA map if imported; currently mostly absent",
            "proxy_dependency": "design field grid and CRA rolloff prior",
            "structure_specialization": "field_x/field_z, CRA_x/CRA_z, wavelength/color",
        }
    if requirement_id == "microlens_ocl_shift_map":
        return {
            "lut_source_class": "structural_shift_prior",
            "calculation_method": "field-indexed OCL/ML shift prior derived from design CRA compensation assumptions",
            "source_priority": "measured/raytrace ML shift map > design shift table > zero/analytic prior",
            "solver_dependency": "used by field response cases but not independently calibrated",
            "external_info_dependency": "field-specific ML/OCL shift map if supplied",
            "proxy_dependency": "analytic CRA compensation shift",
            "structure_specialization": "OCL mode and field position",
        }
    if requirement_id in {
        "conversion_gain_fwc_saturation_nonlinearity",
        "dark_current_temperature_exposure",
        "dsnu_prnu",
        "temporal_noise",
        "charge_collection_electrical_crosstalk",
    }:
        return {
            "lut_source_class": "electrical_prior_seed",
            "calculation_method": "area/architecture-based electrical priors with generic temperature/noise/gain distributions",
            "source_priority": "measured electrical calibration > calibrated TCAD/circuit model > pixel-area/architecture prior",
            "solver_dependency": "DEVSIM/TCAD proxy structures only; not calibrated pinned-PD/TG/FD transport",
            "external_info_dependency": "pixel pitch, architecture, HDR/PDAF/binned topology from image_sensor_db",
            "proxy_dependency": "CG/FWC/noise/dark/DSNU/PRNU priors",
            "structure_specialization": "pixel area, binning group, split/PD architecture where known",
        }
    if requirement_id in {
        "analog_digital_gain",
        "black_level_optical_black",
        "adc_clipping_quantization",
        "row_column_fpn_timing_direction",
        "defect_hot_pixel_stats",
        "binning_remosaic_modes",
    }:
        return {
            "lut_source_class": "readout_raw_prior_seed",
            "calculation_method": "generic CameraE2E readout-mode prior tables with topology-aware binning/remosaic rows",
            "source_priority": "sensor register/mode table > measured raw characterization > generic prior",
            "solver_dependency": "not solved by FDTD; no circuit-level readout simulation",
            "external_info_dependency": "sensor mode/topology metadata when available",
            "proxy_dependency": "gain, black level, ADC, FPN, timing, defect priors",
            "structure_specialization": "binning/remosaic mode and RAW bit-depth/gain assumptions",
        }
    if requirement_id in {
        "lens_raytrace_field_cra_map",
        "sensor_position_tilt_decenter",
        "vignetting_shading",
        "wavelength_dependent_cra_pupil",
    }:
        return {
            "lut_source_class": "module_coupling_prior",
            "calculation_method": "module field-map priors for CRA, vignetting, pupil, sensor pose; replaced by raytrace table when supplied",
            "source_priority": "module raytrace/measured field map > imported prior table > analytic module prior",
            "solver_dependency": "couples into optical LUT but is not generated by pixel FDTD",
            "external_info_dependency": "lens raytrace, assembly, sensor pose, and pupil data",
            "proxy_dependency": "cos4/vignetting and zero-centered assembly priors",
            "structure_specialization": "field position, wavelength, sensor pose, CRA mismatch if reference exists",
        }
    return {
        "lut_source_class": "unclassified_prior",
        "calculation_method": "coverage row exists but method classification needs review",
        "source_priority": "review required",
        "solver_dependency": "",
        "external_info_dependency": "",
        "proxy_dependency": "",
        "structure_specialization": "",
    }


def recommended_use(row: dict[str, str], method: dict[str, str]) -> str:
    if row.get("research_gate") == "N/A":
        return "Not applicable for this sensor/channel topology."
    source_class = method.get("lut_source_class", "")
    if "solver_anchor" in source_class:
        return "Use for research trend comparison and sensitivity studies; preserve gates and uncertainty."
    if "crosstalk" in row.get("requirement_id", ""):
        return "Use for CameraE2E crosstalk plumbing and relative topology comparison only."
    if "prior" in source_class or "proxy" in source_class:
        return "Use as a research prior or placeholder input; do not calibrate product decisions from it."
    return "Use only with the attached research/product gates."


def next_action(row: dict[str, str], method: dict[str, str]) -> str:
    blocker = row.get("primary_blocker", "")
    if row.get("product_gate") in {"PASS", "N/A"}:
        return "No product action for this requirement, or not applicable."
    if blocker:
        return blocker
    source_class = method.get("lut_source_class", "")
    if "module" in source_class:
        return "Import module raytrace or measured field map."
    if "electrical" in source_class or "readout" in source_class:
        return "Import measured electrical/readout calibration tables."
    if "material" in source_class or "cfa" in source_class:
        return "Import measured stack geometry and measured n,k/spectral data."
    return "Run converged quantitative solver points and import measured calibration data."


def build_rows(package_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    trust_by_key = index_by_pair(
        read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_requirement.csv"),
        "slug",
        "requirement_id",
    )
    capability_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv"), "slug")
    cfa_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv"), "slug")
    mesh_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv"), "slug")

    matrix_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for coverage in coverage_rows:
        slug = coverage.get("slug", "")
        rid = coverage.get("requirement_id", "")
        trust = trust_by_key.get((slug, rid), {})
        capability = capability_by_slug.get(slug, {})
        cfa = cfa_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        method = method_for_requirement(rid, mesh, cfa)
        method_gate = "CHECK" if coverage.get("research_gate") in {"CHECK", "PASS", "N/A"} else "FAIL"
        out = {
            "slug": slug,
            "code": coverage.get("code", ""),
            "manufacturer": coverage.get("manufacturer", ""),
            "device_name": coverage.get("device_name", ""),
            "domain": coverage.get("domain", ""),
            "requirement_id": rid,
            "requirement": coverage.get("requirement", ""),
            "camera_e2e_use": coverage.get("camera_e2e_use", ""),
            **method,
            "recommended_camera_e2e_use": recommended_use(coverage, method),
            "not_valid_for": NOT_VALID_FOR if coverage.get("product_gate") not in {"PASS", "N/A"} else "",
            "research_gate": coverage.get("research_gate", ""),
            "product_gate": coverage.get("product_gate", ""),
            "trust_class": trust.get("trust_class", ""),
            "research_usability_score_0_100": trust.get("research_usability_score_0_100", ""),
            "evidence_confidence_score_0_100": trust.get("evidence_confidence_score_0_100", ""),
            "product_calibration_score_0_100": trust.get("product_calibration_score_0_100", ""),
            "coverage_row_count": coverage.get("row_count", ""),
            "field_pass_points": mesh.get("field_pass_points", ""),
            "field_required_points": mesh.get("field_required_points", ""),
            "crosstalk_pass_points": mesh.get("crosstalk_pass_points", ""),
            "crosstalk_required_points": mesh.get("crosstalk_required_points", ""),
            "cfa_provenance_class": cfa.get("cfa_provenance_class", capability.get("cfa_provenance_class", "")),
            "cfa_assumption_gate": cfa.get("cfa_assumption_gate", capability.get("cfa_assumption_gate", "")),
            "mesh_confidence_class": mesh.get("mesh_confidence_class", capability.get("mesh_confidence_class", "")),
            "source_artifacts": coverage.get("source_artifacts", ""),
            "primary_blocker": coverage.get("primary_blocker", ""),
            "next_action": next_action(coverage, method),
            "method_gate": method_gate,
        }
        matrix_rows.append(out)
        grouped[slug].append(out)

    sensor_rows: list[dict[str, Any]] = []
    for slug, rows in sorted(grouped.items()):
        first = rows[0]
        product_ready = all(row.get("product_gate") in {"PASS", "N/A"} for row in rows)
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
                "method_gate_counts": json.dumps(dict(sorted(Counter(row.get("method_gate", "") for row in rows).items())), sort_keys=True),
                "field_pass_points": first.get("field_pass_points", ""),
                "field_required_points": first.get("field_required_points", ""),
                "crosstalk_pass_points": first.get("crosstalk_pass_points", ""),
                "crosstalk_required_points": first.get("crosstalk_required_points", ""),
                "cfa_provenance_class": first.get("cfa_provenance_class", ""),
                "mesh_confidence_class": first.get("mesh_confidence_class", ""),
                "product_ready": product_ready,
                "primary_method_note": "LUT is loadable for research/trend only; product gates remain blocked until measured inputs and convergence pass.",
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
    matrix_cols = [
        "device_name",
        "domain",
        "requirement_id",
        "lut_source_class",
        "calculation_method",
        "source_priority",
        "research_gate",
        "product_gate",
        "trust_class",
        "next_action",
    ]
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E LUT Method Provenance</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E LUT Method Provenance</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This matrix separates solver-derived, external-DB, and proxy/prior values before CameraE2E consumption.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("matrix_row_count", 0))}</div><div class="muted">requirement rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
  </div>
  <h2>Source Class Counts</h2>
  <p><code>{html_cell(payload.get("source_class_counts", {}))}</code></p>
  <h2>Sensor Summary</h2>
  {html_table(sensor_rows, SENSOR_COLUMNS)}
  <h2>Requirement Matrix</h2>
  {html_table(matrix_rows, matrix_cols)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    matrix_rows, sensor_rows = build_rows(package_dir)
    expected_rows = len(read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv"))
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    checks = [
        check_row(
            "coverage_join_complete",
            bool(matrix_rows) and len(matrix_rows) == expected_rows,
            "PASS" if bool(matrix_rows) and len(matrix_rows) == expected_rows else "FAIL",
            {"matrix_rows": len(matrix_rows), "coverage_rows": expected_rows},
            "Regenerate coverage and method provenance matrix.",
        ),
        check_row(
            "method_classified",
            all(str(row.get("lut_source_class", "")).strip() for row in matrix_rows),
            "PASS" if all(str(row.get("lut_source_class", "")).strip() for row in matrix_rows) else "FAIL",
            {"unclassified_count": sum(1 for row in matrix_rows if not str(row.get("lut_source_class", "")).strip())},
            "Add classification for every requirement_id.",
        ),
        check_row(
            "product_gate_closed",
            product_ready_count == 0,
            "PASS" if product_ready_count == 0 else "FAIL",
            {"product_ready_count": product_ready_count},
            "Do not expose product-ready source classes until measured data and convergence gates pass.",
        ),
    ]
    pass_all = all(boolish(row["pass"]) for row in checks)
    status = "METHOD_PROVENANCE_READY_PRODUCT_BLOCKED" if pass_all else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "camera_e2e_method_provenance.json"
    matrix_csv = output_dir / "camera_e2e_method_provenance_matrix.csv"
    sensors_csv = output_dir / "camera_e2e_method_provenance_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_method_provenance_checks.csv"
    html_path = output_dir / "index.html"

    write_csv(matrix_csv, matrix_rows, MATRIX_COLUMNS)
    write_csv(sensors_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)

    payload = {
        "schema": "camera_e2e_method_provenance_v1",
        "artifact_role": "solver_external_proxy_method_matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "matrix_row_count": len(matrix_rows),
        "product_ready_count": product_ready_count,
        "source_class_counts": dict(sorted(Counter(row.get("lut_source_class", "") for row in matrix_rows).items())),
        "research_gate_counts": dict(sorted(Counter(row.get("research_gate", "") for row in matrix_rows).items())),
        "product_gate_counts": dict(sorted(Counter(row.get("product_gate", "") for row in matrix_rows).items())),
        "validation": {
            "schema": "camera_e2e_method_provenance_validation_v1",
            "pass": pass_all,
            "status": status,
            "issue_count": sum(1 for row in checks if not boolish(row["pass"])),
            "issues": [row for row in checks if not boolish(row["pass"])],
        },
        "outputs": {
            "json": repo_rel(json_path),
            "matrix_csv": repo_rel(matrix_csv),
            "sensors_csv": repo_rel(sensors_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(json_path, payload)
    write_html(html_path, payload, matrix_rows, sensor_rows)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
