#!/usr/bin/env python3
"""Export canonical CameraE2E payloads per sensor.

Flat sensor models are complete, but their structure is still close to the
internal exporter layout. This exporter repackages each flat model into the
CameraE2E objective names: Optical/Color, Pixel/Electrical, Readout/RAW, and
Module Coupling. It preserves the import contract, source integrity, method
provenance, uncertainty, and product blockers.

No new physical values are created here.
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

from export_camera_e2e_import_contract import boolish, gate


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_canonical_payload"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "effective_ocl_mode",
    "canonical_payload_json",
    "flat_sensor_json",
    "sensor_import_contract_json",
    "optical_material_row_count",
    "spectral_response_row_count",
    "runtime_field_response_row_count",
    "optical_crosstalk_kernel_row_count",
    "electrical_noise_row_count",
    "readout_gain_row_count",
    "binning_remosaic_row_count",
    "module_field_row_count",
    "source_integrity_row_count",
    "method_provenance_row_count",
    "uncertainty_row_count",
    "import_requirement_row_count",
    "import_pointer_resolved_count",
    "research_allowed_requirement_count",
    "product_allowed_requirement_count",
    "product_ready",
    "camera_e2e_use_scope",
    "payload_gate",
    "payload_notes",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]

REQUIRED_DOMAIN_COUNTS = {
    "optical_material_row_count": "Optical/Color material n,k rows are missing.",
    "spectral_response_row_count": "Spectral response rows are missing.",
    "runtime_field_response_row_count": "Runtime field/CRA response rows are missing.",
    "optical_crosstalk_kernel_row_count": "Optical crosstalk kernel rows are missing.",
    "electrical_noise_row_count": "Pixel/electrical rows are missing.",
    "readout_gain_row_count": "Readout gain rows are missing.",
    "binning_remosaic_row_count": "Binning/remosaic rows are missing.",
    "module_field_row_count": "Module field/CRA rows are missing.",
    "source_integrity_row_count": "Source-integrity rows are missing.",
    "method_provenance_row_count": "Method-provenance rows are missing.",
    "uncertainty_row_count": "Uncertainty rows are missing.",
}


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def abs_from_repo(path: str | Path | None) -> Path:
    if not path:
        return Path("")
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_cell(row.get(column, "")) for column in columns})


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def count_rows(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return 1 if value else 0
    return 0


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def build_payload(
    *,
    flat_model: dict[str, Any],
    import_contract: dict[str, Any],
    flat_sensor_json: str,
    sensor_contract_json: str,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sensor = flat_model.get("sensor", {}) if isinstance(flat_model.get("sensor"), dict) else {}
    optical = flat_model.get("optical_color", {}) if isinstance(flat_model.get("optical_color"), dict) else {}
    electrical = flat_model.get("pixel_electrical", {}) if isinstance(flat_model.get("pixel_electrical"), dict) else {}
    readout = flat_model.get("readout_raw", {}) if isinstance(flat_model.get("readout_raw"), dict) else {}
    module = flat_model.get("module_coupling", {}) if isinstance(flat_model.get("module_coupling"), dict) else {}
    routing = flat_model.get("camera_e2e_routing", {}) if isinstance(flat_model.get("camera_e2e_routing"), dict) else {}
    uncertainty = flat_model.get("uncertainty_budget", {}) if isinstance(flat_model.get("uncertainty_budget"), dict) else {}
    response_trace = flat_model.get("response_trace", {}) if isinstance(flat_model.get("response_trace"), dict) else {}
    response_example = flat_model.get("response_example", {}) if isinstance(flat_model.get("response_example"), dict) else {}
    method_provenance = flat_model.get("method_provenance", {}) if isinstance(flat_model.get("method_provenance"), dict) else {}
    source_integrity = flat_model.get("source_integrity", {}) if isinstance(flat_model.get("source_integrity"), dict) else {}
    objective = flat_model.get("objective_fulfillment", {}) if isinstance(flat_model.get("objective_fulfillment"), dict) else {}
    import_gates = import_contract.get("gates", {}) if isinstance(import_contract.get("gates"), dict) else {}

    payload = {
        "schema": "camera_e2e_canonical_sensor_payload_v1",
        "artifact_role": "per_sensor_camera_e2e_canonical_payload",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor": sensor,
        "source_files": {
            "flat_sensor_json": flat_sensor_json,
            "sensor_import_contract_json": sensor_contract_json,
        },
        "gates": {
            "research_payload_gate": "PASS",
            "product_payload_gate": "PASS" if boolish(import_gates.get("product_ready")) and int(import_gates.get("product_allowed_requirement_count", 0) or 0) == int(import_gates.get("requirement_count", 0) or 0) else "FAIL",
            "product_ready": boolish(import_gates.get("product_ready")),
            "import_contract": import_gates,
            "flat_model": flat_model.get("gates", {}),
        },
        "camera_e2e_payload": {
            "optical_color": {
                "material_nk_ri": optical.get("material_nk_lut", []),
                "cfa": {
                    "by_sensor": optical.get("cfa_db_by_sensor", {}),
                    "transmission_lut": optical.get("cfa_db_transmission_lut", []),
                    "provenance": optical.get("cfa_provenance", {}),
                },
                "spectral_response_qe": {
                    "spectral_response": optical.get("spectral_response", []),
                    "runtime_field_response_lut": optical.get("runtime_field_response_lut", []),
                    "response_trace": response_trace,
                    "response_example": response_example,
                },
                "color_response_matrix": optical.get("color_matrix_seed", {}),
                "optical_crosstalk_kernel": {
                    "kernel_lut": optical.get("optical_crosstalk_kernel_lut", []),
                    "finite_array_support_rows": optical.get("crosstalk_support_rows", []),
                    "product_candidate_rows": optical.get("crosstalk_product_candidate_rows", []),
                    "batch_priority_rows": optical.get("crosstalk_batch_priority_rows", []),
                },
                "angular_cra_response": optical.get("angular_cra_response_rows", {}),
                "microlens_ocl_shift_map": optical.get("microlens_ocl_shift_rows", {}),
            },
            "pixel_electrical": {
                "conversion_gain_full_well_noise_dark_prnu_dsnu": electrical.get("electrical_noise_lut", []),
                "charge_collection_diffusion_electrical_crosstalk": {
                    "rows": electrical.get("electrical_noise_lut", []),
                    "columns": electrical.get("charge_collection_and_electrical_crosstalk_columns", []),
                },
            },
            "readout_raw": {
                "analog_digital_gain_black_adc_fpn_defect": readout.get("readout_gain_lut", []),
                "binning_remosaic_modes": readout.get("binning_remosaic_lut", []),
            },
            "module_coupling": {
                "lens_raytrace_field_cra_vignetting_pupil": module.get("module_field_lut", []),
                "cra_and_alignment_columns": module.get("cra_and_alignment_columns", []),
            },
        },
        "routing_and_evidence": {
            "import_contract_requirement_rows": import_contract.get("requirement_rows", []),
            "objective_fulfillment": objective,
            "source_integrity": source_integrity,
            "method_provenance": method_provenance,
            "uncertainty_budget": uncertainty,
            "capability_and_trust": {
                "use_scope_by_sensor": routing.get("use_scope_by_sensor", {}),
                "use_scope_by_domain": routing.get("use_scope_by_domain", []),
                "capability_profile": routing.get("capability_profile", {}),
                "lut_trust_by_sensor": routing.get("lut_trust_by_sensor", {}),
                "lut_trust_by_domain": routing.get("lut_trust_by_domain", []),
                "lut_trust_by_requirement": routing.get("lut_trust_by_requirement", []),
                "mesh_confidence": routing.get("mesh_confidence", {}),
            },
            "coverage_matrix": routing.get("coverage_matrix", []),
            "probe_summary": routing.get("probe_summary", []),
        },
        "policy": {
            "research_use": "Use for CameraE2E research/trend/sensitivity when research_payload_gate is PASS and row gates are preserved.",
            "product_use": "Blocked until product_payload_gate is PASS, measured inputs are imported, and high-resolution convergence passes.",
            "important_limitation": "Canonical payload is a repackaging of current LUT/proxy/prior rows. It is not a new physical calibration.",
        },
    }

    row_counts = {
        "optical_material_row_count": count_rows(optical.get("material_nk_lut", [])),
        "spectral_response_row_count": count_rows(optical.get("spectral_response", [])),
        "runtime_field_response_row_count": count_rows(optical.get("runtime_field_response_lut", [])),
        "optical_crosstalk_kernel_row_count": count_rows(optical.get("optical_crosstalk_kernel_lut", [])),
        "electrical_noise_row_count": count_rows(electrical.get("electrical_noise_lut", [])),
        "readout_gain_row_count": count_rows(readout.get("readout_gain_lut", [])),
        "binning_remosaic_row_count": count_rows(readout.get("binning_remosaic_lut", [])),
        "module_field_row_count": count_rows(module.get("module_field_lut", [])),
        "source_integrity_row_count": count_rows(source_integrity.get("source_integrity_matrix_rows", [])),
        "method_provenance_row_count": count_rows(method_provenance.get("method_provenance_matrix_rows", [])),
        "uncertainty_row_count": count_rows(uncertainty.get("uncertainty_budget_rows", [])),
        "import_requirement_row_count": count_rows(import_contract.get("requirement_rows", [])),
        "import_pointer_resolved_count": int(import_gates.get("pointer_resolved_count", 0) or 0),
        "research_allowed_requirement_count": int(import_gates.get("research_allowed_requirement_count", 0) or 0),
        "product_allowed_requirement_count": int(import_gates.get("product_allowed_requirement_count", 0) or 0),
    }
    payload["row_counts"] = row_counts
    notes = [message for key, message in REQUIRED_DOMAIN_COUNTS.items() if row_counts.get(key, 0) <= 0]
    if row_counts["import_pointer_resolved_count"] != row_counts["import_requirement_row_count"]:
        notes.append("Import contract pointer count does not match requirement count.")
    if row_counts["product_allowed_requirement_count"] != 0:
        notes.append("Product requirements are unexpectedly import-allowed.")
    payload["gates"]["research_payload_gate"] = "PASS" if not notes and row_counts["import_requirement_row_count"] > 0 else "FAIL"
    summary = {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
        "cfa_pattern": sensor.get("cfa_pattern", ""),
        "effective_ocl_mode": sensor.get("effective_ocl_mode", ""),
        "canonical_payload_json": repo_rel(output_path),
        "flat_sensor_json": flat_sensor_json,
        "sensor_import_contract_json": sensor_contract_json,
        **row_counts,
        "product_ready": boolish(import_gates.get("product_ready")),
        "camera_e2e_use_scope": routing.get("import_decision", {}).get("camera_e2e_use_scope", ""),
        "payload_gate": payload["gates"]["research_payload_gate"],
        "payload_notes": "; ".join(notes),
    }
    write_json(output_path, payload)
    return payload, summary


def build_canonical_payload(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensor_dir = output_dir / "sensors"
    flat_bundle_path = args.flat_bundle_json.resolve() if args.flat_bundle_json else package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json"
    import_contract_path = args.import_contract_json.resolve() if args.import_contract_json else package_dir / "camera_e2e_import_contract" / "camera_e2e_import_contract.json"
    flat_bundle = read_json(flat_bundle_path)
    import_contract_bundle = read_json(import_contract_path)
    import_contract_files = [str(path) for path in import_contract_bundle.get("sensor_import_contract_json_files", [])]
    import_by_slug: dict[str, tuple[str, dict[str, Any]]] = {}
    for rel_path in import_contract_files:
        path = abs_from_repo(rel_path)
        contract = read_json(path)
        slug = str(contract.get("sensor", {}).get("slug", ""))
        if slug:
            import_by_slug[slug] = (rel_path, contract)

    sensor_rows: list[dict[str, Any]] = []
    payload_files: list[str] = []
    missing_import_contracts: list[str] = []
    invalid_flat_models: list[str] = []

    for flat_sensor_json in [str(path) for path in flat_bundle.get("sensor_model_json_files", [])]:
        flat_path = abs_from_repo(flat_sensor_json)
        flat_model = read_json(flat_path)
        if flat_model.get("schema") != "camera_e2e_flat_sensor_model_v1":
            invalid_flat_models.append(flat_sensor_json)
            continue
        slug = str(flat_model.get("sensor", {}).get("slug", ""))
        contract_info = import_by_slug.get(slug)
        if not contract_info:
            missing_import_contracts.append(slug)
            continue
        sensor_import_contract_json, sensor_import_contract = contract_info
        output_path = sensor_dir / f"{slug}_camera_e2e_canonical_payload.json"
        _payload, summary = build_payload(
            flat_model=flat_model,
            import_contract=sensor_import_contract,
            flat_sensor_json=flat_sensor_json,
            sensor_contract_json=sensor_import_contract_json,
            output_path=output_path,
        )
        sensor_rows.append(summary)
        payload_files.append(repo_rel(output_path))

    payload_failures = [row for row in sensor_rows if row.get("payload_gate") != "PASS"]
    product_allowed_count = sum(int(row.get("product_allowed_requirement_count", 0) or 0) for row in sensor_rows)
    checks = [
        check_row(
            "flat_bundle_valid",
            flat_bundle.get("schema") == "camera_e2e_flat_sensor_bundle_v1" and bool(flat_bundle.get("validation", {}).get("pass")),
            flat_bundle.get("validation", {}).get("status", "MISSING"),
            {"sensor_count": flat_bundle.get("sensor_count", 0), "flat_bundle_json": repo_rel(flat_bundle_path)},
            "Regenerate flat sensor bundle.",
        ),
        check_row(
            "import_contract_valid",
            import_contract_bundle.get("schema") == "camera_e2e_import_contract_v1" and bool(import_contract_bundle.get("validation", {}).get("pass")),
            import_contract_bundle.get("validation", {}).get("status", "MISSING"),
            {"requirement_row_count": import_contract_bundle.get("requirement_row_count", 0), "import_contract_json": repo_rel(import_contract_path)},
            "Regenerate import contract.",
        ),
        check_row(
            "flat_models_valid",
            not invalid_flat_models,
            "PASS" if not invalid_flat_models else "FAIL",
            {"invalid_flat_models": invalid_flat_models},
            "Regenerate invalid flat sensor models.",
        ),
        check_row(
            "per_sensor_import_contracts_present",
            not missing_import_contracts,
            "PASS" if not missing_import_contracts else "FAIL",
            {"missing_import_contracts": missing_import_contracts},
            "Regenerate import contract per-sensor JSON files.",
        ),
        check_row(
            "canonical_payloads_present",
            len(sensor_rows) > 0 and len(sensor_rows) == int(flat_bundle.get("sensor_count", 0) or len(sensor_rows)),
            "PASS" if len(sensor_rows) > 0 and len(sensor_rows) == int(flat_bundle.get("sensor_count", 0) or len(sensor_rows)) else "FAIL",
            {"sensor_rows": len(sensor_rows), "flat_bundle_sensor_count": flat_bundle.get("sensor_count", 0)},
            "Generate one canonical payload per sensor.",
        ),
        check_row(
            "canonical_payloads_complete",
            not payload_failures,
            "PASS" if not payload_failures else "FAIL",
            {"payload_failures": [{key: row.get(key) for key in ("slug", "payload_notes")} for row in payload_failures]},
            "Inspect per-sensor canonical payload row counts.",
        ),
        check_row(
            "product_payloads_blocked",
            product_allowed_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_allowed_count == 0 else "FAIL",
            {"product_allowed_requirement_count": product_allowed_count},
            "Keep product payload import blocked until product gates pass.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_CANONICAL_PAYLOAD_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    package_json = output_dir / "camera_e2e_canonical_payload.json"
    by_sensor_csv = output_dir / "camera_e2e_canonical_payload_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_canonical_payload_checks.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_canonical_payload_v1",
        "artifact_role": "camera_e2e_canonical_payload_package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "flat_bundle_json": repo_rel(flat_bundle_path),
        "import_contract_json": repo_rel(import_contract_path),
        "sensor_count": len(sensor_rows),
        "canonical_payload_file_count": len(payload_files),
        "requirement_row_count": sum(int(row.get("import_requirement_row_count", 0) or 0) for row in sensor_rows),
        "pointer_resolved_count": sum(int(row.get("import_pointer_resolved_count", 0) or 0) for row in sensor_rows),
        "product_allowed_requirement_count": product_allowed_count,
        "payload_gate_counts": dict(sorted(Counter(str(row.get("payload_gate", "")) for row in sensor_rows).items())),
        "validation": {
            "schema": "camera_e2e_canonical_payload_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "policy": {
            "research_use": "Preferred adapter payload for CameraE2E research/trend/sensitivity workflows.",
            "product_use": "Blocked until product gates and measured/calibrated evidence pass.",
        },
        "canonical_payload_json_files": payload_files,
        "outputs": {
            "json": repo_rel(package_json),
            "by_sensor_csv": repo_rel(by_sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
            "sensor_payloads_dir": repo_rel(sensor_dir),
        },
    }
    write_csv(by_sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(package_json, payload)
    write_html(html_path, payload, sensor_rows, checks)
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


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
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
<title>CameraE2E Canonical Payload</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Canonical Payload</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Payloads are organized by CameraE2E objective domain and preserve source/proxy/simulation gates.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">payload status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_row_count", 0))}</div><div class="muted">requirement rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("pointer_resolved_count", 0))}</div><div class="muted">import pointers resolved</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_allowed_requirement_count", 0))}</div><div class="muted">product-allowed requirements</div></div>
</div>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Per-Sensor Payloads</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_canonical_payload_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_canonical_payload_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_canonical_payload_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_canonical_payload_html"] = payload["outputs"]["html"]
    outputs["camera_e2e_canonical_payload_sensor_dir"] = payload["outputs"]["sensor_payloads_dir"]
    package["latest_camera_e2e_canonical_payload"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "canonical_payload_file_count": payload["canonical_payload_file_count"],
        "requirement_row_count": payload["requirement_row_count"],
        "pointer_resolved_count": payload["pointer_resolved_count"],
        "product_allowed_requirement_count": payload["product_allowed_requirement_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flat-bundle-json", type=Path)
    parser.add_argument("--import-contract-json", type=Path)
    return parser


def main() -> None:
    payload = build_canonical_payload(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "canonical_payload_file_count": payload["canonical_payload_file_count"],
                "requirement_row_count": payload["requirement_row_count"],
                "pointer_resolved_count": payload["pointer_resolved_count"],
                "product_allowed_requirement_count": payload["product_allowed_requirement_count"],
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
