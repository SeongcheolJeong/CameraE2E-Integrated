#!/usr/bin/env python3
"""Export a consolidated CameraE2E runtime bundle.

The bundle joins field-response query rows, compact crosstalk kernels, and
readiness gates into a single integration artifact. It is designed so a camera
pipeline can ingest one manifest and still preserve every research/product gate.
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

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_runtime_bundle"

RUNTIME_COLUMNS = [
    "runtime_id",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "wavelength_nm",
    "color_channel",
    "response_nominal",
    "response_min",
    "response_max",
    "response_uncertainty_half_range",
    "cra_x_deg",
    "cra_z_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_x_deg",
    "cra_mismatch_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_tolerance_profile",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "cra_mismatch_gate",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "lens_shift_model",
    "cra_measurement_gate",
    "cra_input_gate",
    "crosstalk_kernel_id",
    "crosstalk_neighborhood",
    "crosstalk_center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "direct_signal_response",
    "neighbor_leakage_response",
    "crosstalk_uncertainty_min",
    "crosstalk_uncertainty_max",
    "field_evidence_gate",
    "crosstalk_evidence_gate",
    "combined_evidence_gate",
    "research_ingest_gate",
    "production_lut_gate",
    "research_ingest_allowed",
    "production_ingest_allowed",
    "product_lut_ready",
    "confidence_class",
    "uncertainty_policy",
    "field_source_cases",
    "cra_source",
]

RUNTIME_NUMERIC_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "wavelength_nm",
    "response_nominal",
    "response_min",
    "response_max",
    "response_uncertainty_half_range",
    "cra_x_deg",
    "cra_z_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_x_deg",
    "cra_mismatch_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "crosstalk_center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "direct_signal_response",
    "neighbor_leakage_response",
    "crosstalk_uncertainty_min",
    "crosstalk_uncertainty_max",
]

KERNEL_COLUMNS = [
    "runtime_id",
    "kernel_id",
    "slug",
    "field_x_norm",
    "field_z_norm",
    "wavelength_nm",
    "color_channel",
    "dx",
    "dz",
    "response_fraction",
    "response_fraction_min",
    "response_fraction_max",
    "color_relation",
    "evidence_gate",
    "source",
]

KERNEL_NUMERIC_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "wavelength_nm",
    "dx",
    "dz",
    "response_fraction",
    "response_fraction_min",
    "response_fraction_max",
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
            writer.writerow({column: csv_cell(row.get(column, "")) for column in columns})


def csv_cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def gate(value: Any, default: str = "MISSING") -> str:
    text = str(value if value is not None else "").strip().upper()
    return text or default


def runtime_string_columns() -> list[str]:
    return [column for column in RUNTIME_COLUMNS if column not in RUNTIME_NUMERIC_COLUMNS and column != "product_lut_ready"]


def kernel_string_columns() -> list[str]:
    return [column for column in KERNEL_COLUMNS if column not in KERNEL_NUMERIC_COLUMNS]


def latest_combined_query(package_dir: Path, package: dict[str, Any]) -> tuple[Path, Path]:
    latest = package.get("latest_camera_e2e_combined_query", {})
    if isinstance(latest, dict) and latest.get("csv") and latest.get("kernel_csv"):
        query_csv = ROOT / latest["csv"]
        kernel_csv = ROOT / latest["kernel_csv"]
        if query_csv.exists() and kernel_csv.exists():
            return query_csv, kernel_csv
    outputs = package.get("outputs", {})
    if isinstance(outputs, dict) and outputs.get("camera_e2e_combined_query_csv") and outputs.get("camera_e2e_combined_query_kernel_csv"):
        query_csv = ROOT / outputs["camera_e2e_combined_query_csv"]
        kernel_csv = ROOT / outputs["camera_e2e_combined_query_kernel_csv"]
        if query_csv.exists() and kernel_csv.exists():
            return query_csv, kernel_csv
    fallback = package_dir / "camera_e2e_combined_query_all_sensors_5x5_no_fail"
    return fallback / "camera_e2e_combined_query.csv", fallback / "camera_e2e_combined_query_kernel_rows.csv"


def confidence_class(row: dict[str, str], readiness: dict[str, str]) -> str:
    if gate(readiness.get("production_lut_gate")) == "PASS":
        return "product_calibrated"
    if gate(row.get("combined_evidence_gate")) == "PASS" and gate(row.get("cra_input_gate")) == "PASS":
        return "solver_pass_research"
    if gate(row.get("field_evidence_gate")) == "PASS":
        return "partial_solver_pass_trend"
    return "proxy_or_prior_trend"


def crosstalk_uncertainty_bounds(output_crosstalk_fraction: float, crosstalk_gate: str) -> tuple[float, float, str]:
    if crosstalk_gate == "PASS":
        width = max(0.002, 0.10 * output_crosstalk_fraction)
        return max(0.0, output_crosstalk_fraction - width), min(1.0, output_crosstalk_fraction + width), "finite_array_converged"
    upper = min(1.0, max(0.05, 3.0 * max(0.0, output_crosstalk_fraction) + 0.01))
    return 0.0, upper, "compact_surrogate_uncalibrated_wide_bound"


def kernel_fraction_bounds(response_fraction: float, evidence_gate: str) -> tuple[float, float]:
    if evidence_gate == "PASS":
        width = max(1e-5, 0.10 * response_fraction)
        return max(0.0, response_fraction - width), min(1.0, response_fraction + width)
    return 0.0, min(1.0, max(0.01, 3.0 * max(0.0, response_fraction) + 0.001))


def build_runtime_rows(
    query_rows: list[dict[str, str]],
    readiness_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    readiness_by_slug = {row.get("slug", ""): row for row in readiness_rows}
    output: list[dict[str, Any]] = []
    for row in query_rows:
        readiness = readiness_by_slug.get(row.get("slug", ""), {})
        nominal = finite_float(row.get("relative_qe_proxy"), 0.0)
        response_min = finite_float(row.get("relative_qe_min"), nominal)
        response_max = finite_float(row.get("relative_qe_max"), nominal)
        center_fraction = finite_float(row.get("crosstalk_center_fraction"), math.nan)
        output_xt = finite_float(row.get("output_crosstalk_fraction"), 0.0)
        if not math.isfinite(center_fraction):
            center_fraction = max(0.0, 1.0 - output_xt)
        xt_min, xt_max, xt_policy = crosstalk_uncertainty_bounds(output_xt, gate(row.get("crosstalk_evidence_gate")))
        response_min, response_max = sorted((response_min, response_max))
        output.append(
            {
                "runtime_id": row.get("query_id", ""),
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "field_x_norm": finite_float(row.get("field_x_norm")),
                "field_z_norm": finite_float(row.get("field_z_norm")),
                "field_radius_norm": finite_float(row.get("field_radius_norm")),
                "field_azimuth_deg": finite_float(row.get("field_azimuth_deg")),
                "wavelength_nm": finite_float(row.get("wavelength_nm")),
                "color_channel": row.get("color_channel", ""),
                "response_nominal": nominal,
                "response_min": response_min,
                "response_max": response_max,
                "response_uncertainty_half_range": 0.5 * max(0.0, response_max - response_min),
                "cra_x_deg": finite_float(row.get("cra_x_deg")),
                "cra_z_deg": finite_float(row.get("cra_z_deg")),
                "lens_cra_x_deg": finite_float(row.get("lens_cra_x_deg")),
                "lens_cra_z_deg": finite_float(row.get("lens_cra_z_deg")),
                "sensor_cra_x_deg": finite_float(row.get("sensor_cra_x_deg")),
                "sensor_cra_z_deg": finite_float(row.get("sensor_cra_z_deg")),
                "cra_mismatch_x_deg": finite_float(row.get("cra_mismatch_x_deg")),
                "cra_mismatch_z_deg": finite_float(row.get("cra_mismatch_z_deg")),
                "cra_mismatch_total_deg": finite_float(row.get("cra_mismatch_total_deg")),
                "cra_mismatch_tolerance_profile": row.get("cra_mismatch_tolerance_profile", ""),
                "cra_mismatch_pass_tolerance_deg": finite_float(row.get("cra_mismatch_pass_tolerance_deg")),
                "cra_mismatch_check_tolerance_deg": finite_float(row.get("cra_mismatch_check_tolerance_deg")),
                "cra_mismatch_gate": row.get("cra_mismatch_gate", ""),
                "lens_shift_x_um": finite_float(row.get("lens_shift_x_um")),
                "lens_shift_z_um": finite_float(row.get("lens_shift_z_um")),
                "lens_shift_model": row.get("lens_shift_model", ""),
                "cra_measurement_gate": row.get("cra_measurement_gate", ""),
                "cra_input_gate": row.get("cra_input_gate", ""),
                "crosstalk_kernel_id": row.get("crosstalk_kernel_id", ""),
                "crosstalk_neighborhood": row.get("crosstalk_neighborhood", ""),
                "crosstalk_center_fraction": center_fraction,
                "output_crosstalk_fraction": output_xt,
                "strongest_neighbor_fraction": finite_float(row.get("strongest_neighbor_fraction"), 0.0),
                "direct_signal_response": nominal * center_fraction,
                "neighbor_leakage_response": nominal * output_xt,
                "crosstalk_uncertainty_min": xt_min,
                "crosstalk_uncertainty_max": xt_max,
                "field_evidence_gate": row.get("field_evidence_gate", ""),
                "crosstalk_evidence_gate": row.get("crosstalk_evidence_gate", ""),
                "combined_evidence_gate": row.get("combined_evidence_gate", ""),
                "research_ingest_gate": readiness.get("research_ingest_gate", "MISSING"),
                "production_lut_gate": readiness.get("production_lut_gate", "MISSING"),
                "research_ingest_allowed": readiness.get("research_ingest_allowed", "False"),
                "production_ingest_allowed": readiness.get("production_ingest_allowed", "False"),
                "product_lut_ready": False,
                "confidence_class": confidence_class(row, readiness),
                "uncertainty_policy": xt_policy,
                "field_source_cases": row.get("field_source_cases", ""),
                "cra_source": row.get("cra_source", ""),
            }
        )
    return output


def build_kernel_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        response_fraction = finite_float(row.get("response_fraction"), 0.0)
        lower, upper = kernel_fraction_bounds(response_fraction, gate(row.get("evidence_gate")))
        output.append(
            {
                "runtime_id": row.get("query_id", ""),
                "kernel_id": row.get("kernel_id", ""),
                "slug": row.get("slug", ""),
                "field_x_norm": finite_float(row.get("field_x_norm")),
                "field_z_norm": finite_float(row.get("field_z_norm")),
                "wavelength_nm": finite_float(row.get("wavelength_nm")),
                "color_channel": row.get("color_channel", ""),
                "dx": finite_float(row.get("dx")),
                "dz": finite_float(row.get("dz")),
                "response_fraction": response_fraction,
                "response_fraction_min": lower,
                "response_fraction_max": upper,
                "color_relation": row.get("color_relation", ""),
                "evidence_gate": row.get("evidence_gate", ""),
                "source": row.get("source", ""),
            }
        )
    return output


def write_npz(path: Path, runtime_rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]]) -> None:
    arrays: dict[str, Any] = {
        "schema": np.array("camera_e2e_runtime_bundle_v1", dtype="U80"),
        "artifact_role": np.array("camera_e2e_runtime_lut_with_gates", dtype="U96"),
        "product_lut_ready": np.array(False, dtype=np.bool_),
        "runtime_row_count": np.array(len(runtime_rows), dtype=np.int64),
        "kernel_row_count": np.array(len(kernel_rows), dtype=np.int64),
        "runtime_columns": np.array(RUNTIME_COLUMNS, dtype="U80"),
        "runtime_numeric_columns": np.array(RUNTIME_NUMERIC_COLUMNS, dtype="U80"),
        "runtime_string_columns": np.array(runtime_string_columns(), dtype="U80"),
        "kernel_columns": np.array(KERNEL_COLUMNS, dtype="U80"),
        "kernel_numeric_columns": np.array(KERNEL_NUMERIC_COLUMNS, dtype="U80"),
        "kernel_string_columns": np.array(kernel_string_columns(), dtype="U80"),
    }
    for column in RUNTIME_NUMERIC_COLUMNS:
        arrays[f"runtime_{column}"] = np.array([finite_float(row.get(column)) for row in runtime_rows], dtype=np.float64)
    for column in runtime_string_columns():
        arrays[f"runtime_{column}"] = np.array([str(row.get(column, "")) for row in runtime_rows], dtype="U2048")
    arrays["runtime_product_lut_ready"] = np.array(
        [boolish(row.get("product_lut_ready")) for row in runtime_rows], dtype=np.bool_
    )
    for column in KERNEL_NUMERIC_COLUMNS:
        arrays[f"kernel_{column}"] = np.array([finite_float(row.get(column)) for row in kernel_rows], dtype=np.float64)
    for column in kernel_string_columns():
        arrays[f"kernel_{column}"] = np.array([str(row.get(column, "")) for row in kernel_rows], dtype="U2048")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def validate(
    runtime_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    npz_path: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    kernel_by_runtime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kernel_rows:
        runtime_id = str(row.get("runtime_id", ""))
        kernel_by_runtime[runtime_id].append(row)
        value = finite_float(row.get("response_fraction"))
        if not math.isfinite(value) or value < -tolerance:
            issues.append({"severity": "error", "code": "kernel_fraction_invalid", "runtime_id": runtime_id})
    for row in runtime_rows:
        runtime_id = str(row.get("runtime_id", ""))
        nominal = finite_float(row.get("response_nominal"))
        lower = finite_float(row.get("response_min"))
        upper = finite_float(row.get("response_max"))
        if not (math.isfinite(lower) and math.isfinite(nominal) and math.isfinite(upper) and lower <= nominal <= upper):
            issues.append({"severity": "error", "code": "response_bounds_invalid", "runtime_id": runtime_id})
        if boolish(row.get("product_lut_ready")):
            issues.append({"severity": "error", "code": "runtime_product_ready_true", "runtime_id": runtime_id})
        rows = kernel_by_runtime.get(runtime_id, [])
        if not rows:
            issues.append({"severity": "error", "code": "missing_kernel_rows", "runtime_id": runtime_id})
            continue
        total = sum(finite_float(item.get("response_fraction"), 0.0) for item in rows)
        if abs(total - 1.0) > tolerance:
            issues.append(
                {"severity": "error", "code": "kernel_sum_not_one", "runtime_id": runtime_id, "sum": total}
            )
    with np.load(npz_path, allow_pickle=False) as data:
        if str(data["schema"]) != "camera_e2e_runtime_bundle_v1":
            issues.append({"severity": "error", "code": "npz_schema_mismatch"})
        if int(data["runtime_row_count"]) != len(runtime_rows):
            issues.append({"severity": "error", "code": "npz_runtime_row_count_mismatch"})
        if int(data["kernel_row_count"]) != len(kernel_rows):
            issues.append({"severity": "error", "code": "npz_kernel_row_count_mismatch"})
        if bool(np.any(data["runtime_product_lut_ready"])):
            issues.append({"severity": "error", "code": "npz_runtime_product_ready_true"})
    return {
        "schema": "camera_e2e_runtime_bundle_validation_v1",
        "pass": not issues,
        "bad_count": len(issues),
        "runtime_row_count": len(runtime_rows),
        "kernel_row_count": len(kernel_rows),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 100) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    more = f"<p class=\"muted\">Showing {min(limit, len(rows))} of {len(rows)} rows.</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def write_html(path: Path, payload: dict[str, Any], runtime_rows: list[dict[str, Any]]) -> None:
    validation = payload.get("validation", {})
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.ok{color:#82f09d}.warn{color:#ffd36e}.bad{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}.pill{display:inline-block;border:1px solid #2d6276;border-radius:999px;padding:3px 8px;margin-right:6px}
code{color:#9fe8ff}
"""
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Runtime Bundle</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Runtime Bundle</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This bundle is for trend/research integration unless all production gates pass.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("runtime_row_count", 0))}</div><div class="muted">runtime rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("kernel_row_count", 0))}</div><div class="muted">kernel rows</div></div>
    <div class="card"><div class="metric {'ok' if validation.get('pass') else 'bad'}">{html_cell(validation.get("pass"))}</div><div class="muted">validation pass</div></div>
    <div class="card"><div class="metric bad">{html_cell(payload.get("product_lut_ready"))}</div><div class="muted">product LUT ready</div></div>
  </div>
  <p><span class="pill">Runtime gates: {html_cell(payload.get("runtime_gate_counts", {}))}</span>
  <span class="pill">Confidence: {html_cell(payload.get("confidence_class_counts", {}))}</span></p>
  <h2>Outputs</h2>
  <p><code>{html_cell(payload.get("outputs", {}).get("runtime_csv", ""))}</code><br>
  <code>{html_cell(payload.get("outputs", {}).get("kernel_csv", ""))}</code><br>
  <code>{html_cell(payload.get("outputs", {}).get("npz", ""))}</code></p>
  <h2>Sample Runtime Rows</h2>
  {html_table(runtime_rows, RUNTIME_COLUMNS, limit=80)}
  <h2>Use Policy</h2>
  <p class="muted">Research ingest is allowed only if CameraE2E preserves row-level gates, uncertainty bounds, and <code>product_lut_ready=false</code>. Production LUT use requires measured CRA/ML shift, measured stack/n,k, complete quantitative FDTD coverage, and finite-array crosstalk convergence.</p>
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
    outputs["camera_e2e_runtime_bundle_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_runtime_bundle_runtime_csv"] = payload["outputs"]["runtime_csv"]
    outputs["camera_e2e_runtime_bundle_kernel_csv"] = payload["outputs"]["kernel_csv"]
    outputs["camera_e2e_runtime_bundle_npz"] = payload["outputs"]["npz"]
    outputs["camera_e2e_runtime_bundle_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_runtime_bundle"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "product_lut_ready": payload["product_lut_ready"],
        "runtime_row_count": payload["runtime_row_count"],
        "kernel_row_count": payload["kernel_row_count"],
        "runtime_gate_counts": payload["runtime_gate_counts"],
        **payload["outputs"],
    }
    package_json_path.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    package_json_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json_path)
    query_csv, query_kernel_csv = latest_combined_query(package_dir, package)
    readiness_csv = package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_summary.csv"
    readiness_report = read_json(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json")
    query_rows = read_csv(query_csv)
    query_kernel_rows = read_csv(query_kernel_csv)
    readiness_rows = read_csv(readiness_csv)

    runtime_rows = build_runtime_rows(query_rows, readiness_rows)
    kernel_rows = build_kernel_rows(query_kernel_rows)

    runtime_csv = output_dir / "camera_e2e_runtime_lut.csv"
    kernel_csv = output_dir / "camera_e2e_runtime_crosstalk_kernel.csv"
    bundle_json = output_dir / "camera_e2e_runtime_bundle.json"
    bundle_npz = output_dir / "camera_e2e_runtime_bundle.npz"
    html_path = output_dir / "index.html"
    write_csv(runtime_csv, runtime_rows, RUNTIME_COLUMNS)
    write_csv(kernel_csv, kernel_rows, KERNEL_COLUMNS)
    write_npz(bundle_npz, runtime_rows, kernel_rows)
    validation = validate(runtime_rows, kernel_rows, bundle_npz, tolerance=args.tolerance)

    slugs = sorted({row.get("slug", "") for row in runtime_rows if row.get("slug")})
    payload = {
        "schema": "camera_e2e_runtime_bundle_v1",
        "artifact_role": "camera_e2e_runtime_lut_with_gates",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "source_combined_query_csv": repo_rel(query_csv),
        "source_combined_kernel_csv": repo_rel(query_kernel_csv),
        "source_readiness_report": repo_rel(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json"),
        "sensor_count": len(slugs),
        "runtime_row_count": len(runtime_rows),
        "kernel_row_count": len(kernel_rows),
        "product_lut_ready": False,
        "research_ingest_valid": bool(readiness_report.get("research_ingest_valid", False)),
        "runtime_gate_counts": dict(Counter(row.get("combined_evidence_gate", "") for row in runtime_rows)),
        "research_gate_counts": dict(Counter(row.get("research_ingest_gate", "") for row in runtime_rows)),
        "production_gate_counts": dict(Counter(row.get("production_lut_gate", "") for row in runtime_rows)),
        "confidence_class_counts": dict(Counter(row.get("confidence_class", "") for row in runtime_rows)),
        "validation": validation,
        "use_policy": {
            "allowed": "research_trend_only",
            "must_preserve": [
                "product_lut_ready",
                "combined_evidence_gate",
                "cra_input_gate",
                "research_ingest_gate",
                "production_lut_gate",
                "response_min/response_max",
                "crosstalk_uncertainty_min/crosstalk_uncertainty_max",
            ],
            "not_allowed": "Do not use as product accuracy LUT until readiness audit production_lut_gate PASS for every sensor.",
        },
        "outputs": {
            "json": repo_rel(bundle_json),
            "runtime_csv": repo_rel(runtime_csv),
            "kernel_csv": repo_rel(kernel_csv),
            "npz": repo_rel(bundle_npz),
            "html": repo_rel(html_path),
        },
    }
    write_json(bundle_json, payload)
    write_html(html_path, payload, runtime_rows)
    update_package_links(package_json_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


def main() -> None:
    payload = export(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation_pass": payload["validation"]["pass"],
                "runtime_row_count": payload["runtime_row_count"],
                "kernel_row_count": payload["kernel_row_count"],
                "product_lut_ready": payload["product_lut_ready"],
                "runtime_gate_counts": payload["runtime_gate_counts"],
                "confidence_class_counts": payload["confidence_class_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
