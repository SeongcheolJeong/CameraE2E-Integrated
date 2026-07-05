#!/usr/bin/env python3
"""Export a requirement-to-loader trace for the CameraE2E objective.

The coverage matrix says whether each objective requirement has evidence. This
trace adds the downstream loader path: flat sensor JSON section, table name,
usage-policy profile, adapter smoke proof, and product blocker status.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_objective_trace"

TRACE_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "requirement_id",
    "requirement",
    "camera_e2e_use",
    "camera_e2e_loader_section",
    "flat_json_pointer",
    "primary_loader_table",
    "secondary_loader_tables",
    "research_gate",
    "product_gate",
    "row_count",
    "camera_e2e_profile",
    "camera_e2e_use_scope",
    "recommended_runtime_filter_id",
    "adapter_smoke_gate",
    "adapter_research_allowed_query_count",
    "adapter_product_allowed_query_count",
    "flat_sensor_json",
    "flat_pointer_gate",
    "flat_pointer_notes",
    "source_artifacts",
    "evidence_summary",
    "primary_blocker",
    "required_before_product_use",
    "trace_gate",
    "trace_notes",
]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_profile",
    "requirement_count",
    "trace_pass_count",
    "trace_check_count",
    "trace_fail_count",
    "trace_missing_count",
    "research_gate_counts",
    "product_gate_counts",
    "adapter_smoke_gate",
    "adapter_research_allowed_query_count",
    "adapter_product_allowed_query_count",
    "product_ready",
    "sensor_trace_gate",
    "first_blocker",
]

REQUIREMENT_SUMMARY_COLUMNS = [
    "domain",
    "requirement_id",
    "requirement",
    "camera_e2e_use",
    "camera_e2e_loader_section",
    "flat_json_pointer",
    "primary_loader_table",
    "secondary_loader_tables",
    "sensor_count",
    "row_count_total",
    "trace_gate_counts",
    "flat_pointer_gate_counts",
    "research_gate_counts",
    "product_gate_counts",
    "camera_e2e_use_scope_counts",
    "source_integrity_gate_counts",
    "lut_source_class_counts",
    "calculation_method_counts",
    "uncertainty_product_gate_counts",
    "primary_uncertainty_quantity",
    "primary_uncertainty_min",
    "primary_uncertainty_max",
    "primary_uncertainty_unit",
    "product_ready_count",
    "product_blocked_count",
    "source_artifacts",
    "primary_blockers",
    "required_before_product_use",
    "recommended_camera_e2e_use",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]

REQUIREMENT_LOAD_MAP: dict[str, dict[str, str]] = {
    "optical_material_nk_ri": {
        "section": "optical_color",
        "pointer": "/optical_color/material_nk_lut",
        "primary": "material_nk_lut",
        "secondary": "cfa_db_by_sensor;cfa_db_transmission_lut;cfa_provenance",
    },
    "spectral_response_qe": {
        "section": "optical_color",
        "pointer": "/optical_color/runtime_field_response_lut",
        "primary": "runtime_field_response_lut",
        "secondary": "spectral_response;cfa_db_transmission_lut",
    },
    "color_response_matrix": {
        "section": "optical_color",
        "pointer": "/optical_color/color_matrix_seed",
        "primary": "color_matrix_seed",
        "secondary": "spectral_response;cfa_provenance",
    },
    "optical_crosstalk_kernel": {
        "section": "optical_color",
        "pointer": "/optical_color/optical_crosstalk_kernel_lut",
        "primary": "optical_crosstalk_kernel_lut",
        "secondary": "crosstalk_support_rows;crosstalk_product_candidate_rows",
    },
    "angular_cra_response": {
        "section": "optical_color",
        "pointer": "/optical_color/angular_cra_response_rows",
        "primary": "runtime_field_response_lut",
        "secondary": "module_coupling/module_field_lut",
    },
    "microlens_ocl_shift_map": {
        "section": "optical_color",
        "pointer": "/optical_color/runtime_field_response_lut/lens_shift_x_um",
        "primary": "runtime_field_response_lut",
        "secondary": "module_coupling/module_field_lut",
    },
    "conversion_gain_fwc_saturation_nonlinearity": {
        "section": "pixel_electrical",
        "pointer": "/pixel_electrical/electrical_noise_lut",
        "primary": "electrical_noise_lut",
        "secondary": "",
    },
    "dark_current_temperature_exposure": {
        "section": "pixel_electrical",
        "pointer": "/pixel_electrical/electrical_noise_lut/dark_current_e_per_s",
        "primary": "electrical_noise_lut",
        "secondary": "",
    },
    "dsnu_prnu": {
        "section": "pixel_electrical",
        "pointer": "/pixel_electrical/electrical_noise_lut/dsnu_e_rms",
        "primary": "electrical_noise_lut",
        "secondary": "",
    },
    "temporal_noise": {
        "section": "pixel_electrical",
        "pointer": "/pixel_electrical/electrical_noise_lut/total_noise_e_rms",
        "primary": "electrical_noise_lut",
        "secondary": "",
    },
    "charge_collection_electrical_crosstalk": {
        "section": "pixel_electrical",
        "pointer": "/pixel_electrical/electrical_noise_lut/electrical_crosstalk_fraction_prior",
        "primary": "electrical_noise_lut",
        "secondary": "charge_collection_and_electrical_crosstalk_columns",
    },
    "analog_digital_gain": {
        "section": "readout_raw",
        "pointer": "/readout_raw/readout_gain_lut",
        "primary": "readout_gain_lut",
        "secondary": "",
    },
    "black_level_optical_black": {
        "section": "readout_raw",
        "pointer": "/readout_raw/readout_gain_lut/black_level_dn",
        "primary": "readout_gain_lut",
        "secondary": "",
    },
    "adc_clipping_quantization": {
        "section": "readout_raw",
        "pointer": "/readout_raw/readout_gain_lut/adc_bit_depth",
        "primary": "readout_gain_lut",
        "secondary": "",
    },
    "row_column_fpn_timing_direction": {
        "section": "readout_raw",
        "pointer": "/readout_raw/readout_gain_lut/readout_direction",
        "primary": "readout_gain_lut",
        "secondary": "",
    },
    "defect_hot_pixel_stats": {
        "section": "readout_raw",
        "pointer": "/readout_raw/readout_gain_lut/hot_pixel_fraction",
        "primary": "readout_gain_lut",
        "secondary": "",
    },
    "binning_remosaic_modes": {
        "section": "readout_raw",
        "pointer": "/readout_raw/binning_remosaic_lut",
        "primary": "binning_remosaic_lut",
        "secondary": "",
    },
    "lens_raytrace_field_cra_map": {
        "section": "module_coupling",
        "pointer": "/module_coupling/module_field_lut/cra_x_deg",
        "primary": "module_field_lut",
        "secondary": "",
    },
    "sensor_position_tilt_decenter": {
        "section": "module_coupling",
        "pointer": "/module_coupling/module_field_lut/sensor_tilt_x_deg",
        "primary": "module_field_lut",
        "secondary": "",
    },
    "vignetting_shading": {
        "section": "module_coupling",
        "pointer": "/module_coupling/module_field_lut/relative_illumination_cos4",
        "primary": "module_field_lut",
        "secondary": "",
    },
    "wavelength_dependent_cra_pupil": {
        "section": "module_coupling",
        "pointer": "/module_coupling/module_field_lut/pupil_relative_transmission",
        "primary": "module_field_lut",
        "secondary": "",
    },
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
    return {row.get(key, ""): row for row in rows if row.get(key)}


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def gate_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        try:
            if value not in ("", None):
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def joined_values(rows: list[dict[str, Any]], key: str, limit: int = 8) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
        if len(values) >= limit:
            break
    return " | ".join(values)


def resolve_flat_pointer(document: dict[str, Any], pointer: str) -> tuple[str, str]:
    if not pointer:
        return "FAIL", "flat_json_pointer is empty"
    current: Any = document
    parts = [part for part in pointer.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if isinstance(current, dict):
            if part not in current:
                return "FAIL", f"missing key {part}"
            current = current[part]
            continue
        if isinstance(current, list):
            if current and all(isinstance(item, dict) for item in current):
                if any(part in item for item in current):
                    if index == len(parts) - 1:
                        nonempty_count = sum(1 for item in current if item.get(part) not in ("", None))
                        return ("PASS" if nonempty_count else "CHECK", f"column {part} in {len(current)} rows; nonempty={nonempty_count}")
                    return "FAIL", f"column {part} found but pointer has trailing elements"
                return "FAIL", f"column {part} missing from list rows"
            return "FAIL", f"cannot descend into non-dict list at {part}"
        return "FAIL", f"cannot descend into scalar at {part}"
    if isinstance(current, list):
        return ("PASS" if current else "CHECK", f"list rows={len(current)}")
    if isinstance(current, dict):
        return ("PASS" if current else "CHECK", f"dict keys={len(current)}")
    return ("PASS" if current not in ("", None) else "CHECK", "scalar")


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def trace_gate_for(row: dict[str, str], mapping: dict[str, str], smoke: dict[str, str], flat: dict[str, str], flat_pointer_gate: str) -> tuple[str, str]:
    notes: list[str] = []
    if not mapping:
        return "FAIL", "requirement_id is not mapped to a flat loader section"
    if not flat.get("flat_sensor_json"):
        notes.append("flat sensor JSON missing")
    if smoke.get("smoke_gate") != "PASS":
        notes.append("adapter smoke did not pass")
    if flat_pointer_gate == "FAIL":
        notes.append("flat_json_pointer does not resolve")
    research_gate = str(row.get("research_gate", "")).upper()
    if research_gate in {"FAIL", "MISSING", ""}:
        notes.append(f"research_gate={research_gate or 'MISSING'}")
    if str(row.get("product_gate", "")).upper() not in {"PASS"}:
        notes.append("product gate remains blocked")
    if not notes:
        return "PASS", ""
    if any(text.startswith("research_gate") or "missing" in text.lower() or "did not pass" in text.lower() or "does not resolve" in text.lower() for text in notes):
        return "FAIL", "; ".join(notes)
    return "CHECK", "; ".join(notes)


def build_objective_trace(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    coverage = read_json(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.json")
    usage_policy = read_json(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy.json")
    adapter_smoke = read_json(package_dir / "camera_e2e_adapter_smoke" / "camera_e2e_adapter_smoke.json")
    flat_bundle = read_json(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json")

    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    coverage_summary_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_summary.csv")
    source_integrity_rows = read_csv_rows(package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_matrix.csv")
    policy_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy_by_sensor.csv"), "slug")
    smoke_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_adapter_smoke" / "camera_e2e_adapter_smoke_by_sensor.csv"), "slug")
    flat_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv"), "slug")
    summary_by_slug = index_by(coverage_summary_rows, "slug")

    trace_rows: list[dict[str, Any]] = []
    unmapped_requirements: set[str] = set()
    for row in coverage_rows:
        slug = row.get("slug", "")
        req_id = row.get("requirement_id", "")
        mapping = REQUIREMENT_LOAD_MAP.get(req_id, {})
        if not mapping:
            unmapped_requirements.add(req_id)
        policy = policy_by_slug.get(slug, {})
        smoke = smoke_by_slug.get(slug, {})
        flat = flat_by_slug.get(slug, {})
        flat_json_path = ROOT / flat.get("flat_sensor_json", "")
        flat_document = read_json(flat_json_path)
        flat_pointer_gate, flat_pointer_notes = resolve_flat_pointer(flat_document, mapping.get("pointer", "")) if flat_document else ("FAIL", "flat sensor JSON missing or invalid")
        trace_gate, trace_notes = trace_gate_for(row, mapping, smoke, flat, flat_pointer_gate)
        trace_rows.append(
            {
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "domain": row.get("domain", ""),
                "requirement_id": req_id,
                "requirement": row.get("requirement", ""),
                "camera_e2e_use": row.get("camera_e2e_use", ""),
                "camera_e2e_loader_section": mapping.get("section", ""),
                "flat_json_pointer": mapping.get("pointer", ""),
                "primary_loader_table": mapping.get("primary", ""),
                "secondary_loader_tables": mapping.get("secondary", ""),
                "research_gate": row.get("research_gate", ""),
                "product_gate": row.get("product_gate", ""),
                "row_count": row.get("row_count", ""),
                "camera_e2e_profile": policy.get("camera_e2e_profile", ""),
                "camera_e2e_use_scope": policy.get("camera_e2e_use_scope", ""),
                "recommended_runtime_filter_id": policy.get("recommended_runtime_filter_id", ""),
                "adapter_smoke_gate": smoke.get("smoke_gate", ""),
                "adapter_research_allowed_query_count": smoke.get("research_allowed_query_count", ""),
                "adapter_product_allowed_query_count": smoke.get("product_allowed_query_count", ""),
                "flat_sensor_json": flat.get("flat_sensor_json", ""),
                "flat_pointer_gate": flat_pointer_gate,
                "flat_pointer_notes": flat_pointer_notes,
                "source_artifacts": row.get("source_artifacts", ""),
                "evidence_summary": row.get("evidence_summary", ""),
                "primary_blocker": row.get("primary_blocker", ""),
                "required_before_product_use": policy.get("required_before_product_use", ""),
                "trace_gate": trace_gate,
                "trace_notes": trace_notes,
            }
        )

    source_integrity_by_req = group_by(source_integrity_rows, "requirement_id")
    trace_by_req = group_by([{key: str(value) for key, value in row.items()} for row in trace_rows], "requirement_id")
    requirement_summary_rows: list[dict[str, Any]] = []
    for req_id, rows in sorted(trace_by_req.items()):
        first = rows[0]
        source_rows = source_integrity_by_req.get(req_id, [])
        row_count_values = [safe_int(row.get("row_count")) for row in rows]
        uncertainty_min_values = numeric_values(source_rows, "primary_uncertainty_min")
        uncertainty_max_values = numeric_values(source_rows, "primary_uncertainty_max")
        source_artifact_rows = source_rows if source_rows else rows
        product_ready_count_for_req = sum(1 for row in rows if str(row.get("product_gate", "")).upper() == "PASS")
        requirement_summary_rows.append(
            {
                "domain": first.get("domain", ""),
                "requirement_id": req_id,
                "requirement": first.get("requirement", ""),
                "camera_e2e_use": first.get("camera_e2e_use", ""),
                "camera_e2e_loader_section": first.get("camera_e2e_loader_section", ""),
                "flat_json_pointer": first.get("flat_json_pointer", ""),
                "primary_loader_table": first.get("primary_loader_table", ""),
                "secondary_loader_tables": first.get("secondary_loader_tables", ""),
                "sensor_count": len(rows),
                "row_count_total": sum(row_count_values),
                "trace_gate_counts": json.dumps(gate_counts(rows, "trace_gate"), sort_keys=True),
                "flat_pointer_gate_counts": json.dumps(gate_counts(rows, "flat_pointer_gate"), sort_keys=True),
                "research_gate_counts": json.dumps(gate_counts(rows, "research_gate"), sort_keys=True),
                "product_gate_counts": json.dumps(gate_counts(rows, "product_gate"), sort_keys=True),
                "camera_e2e_use_scope_counts": json.dumps(gate_counts(rows, "camera_e2e_use_scope"), sort_keys=True),
                "source_integrity_gate_counts": json.dumps(gate_counts(source_rows, "source_integrity_gate"), sort_keys=True),
                "lut_source_class_counts": json.dumps(gate_counts(source_rows, "lut_source_class"), sort_keys=True),
                "calculation_method_counts": json.dumps(gate_counts(source_rows, "calculation_method"), sort_keys=True),
                "uncertainty_product_gate_counts": json.dumps(gate_counts(source_rows, "uncertainty_product_gate"), sort_keys=True),
                "primary_uncertainty_quantity": joined_values(source_rows, "primary_uncertainty_quantity", limit=4),
                "primary_uncertainty_min": min(uncertainty_min_values) if uncertainty_min_values else "",
                "primary_uncertainty_max": max(uncertainty_max_values) if uncertainty_max_values else "",
                "primary_uncertainty_unit": joined_values(source_rows, "primary_uncertainty_unit", limit=4),
                "product_ready_count": product_ready_count_for_req,
                "product_blocked_count": len(rows) - product_ready_count_for_req,
                "source_artifacts": joined_values(source_artifact_rows, "source_artifacts", limit=10),
                "primary_blockers": joined_values(source_artifact_rows, "primary_blocker", limit=10),
                "required_before_product_use": joined_values(rows, "required_before_product_use", limit=4),
                "recommended_camera_e2e_use": joined_values(source_rows, "recommended_camera_e2e_use", limit=6),
            }
        )

    by_slug = group_by([{key: str(value) for key, value in row.items()} for row in trace_rows], "slug")
    sensor_rows: list[dict[str, Any]] = []
    for slug, rows in sorted(by_slug.items()):
        first = rows[0]
        summary = summary_by_slug.get(slug, {})
        policy = policy_by_slug.get(slug, {})
        smoke = smoke_by_slug.get(slug, {})
        trace_counts = gate_counts(rows, "trace_gate")
        sensor_gate = "FAIL" if trace_counts.get("FAIL", 0) else "CHECK" if trace_counts.get("CHECK", 0) else "PASS"
        first_blocker = next((row.get("primary_blocker", "") for row in rows if row.get("primary_blocker")), "")
        sensor_rows.append(
            {
                "slug": slug,
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "camera_e2e_profile": policy.get("camera_e2e_profile", ""),
                "requirement_count": len(rows),
                "trace_pass_count": trace_counts.get("PASS", 0),
                "trace_check_count": trace_counts.get("CHECK", 0),
                "trace_fail_count": trace_counts.get("FAIL", 0),
                "trace_missing_count": trace_counts.get("MISSING", 0),
                "research_gate_counts": summary.get("research_gate_counts", ""),
                "product_gate_counts": summary.get("product_gate_counts", ""),
                "adapter_smoke_gate": smoke.get("smoke_gate", ""),
                "adapter_research_allowed_query_count": smoke.get("research_allowed_query_count", ""),
                "adapter_product_allowed_query_count": smoke.get("product_allowed_query_count", ""),
                "product_ready": summary.get("product_ready", ""),
                "sensor_trace_gate": sensor_gate,
                "first_blocker": first_blocker,
            }
        )

    expected_sensor_count = safe_int(flat_bundle.get("sensor_count"))
    expected_requirement_count = safe_int(coverage.get("requirement_count_per_sensor"))
    trace_fail_count = sum(1 for row in trace_rows if row.get("trace_gate") == "FAIL")
    flat_pointer_fail_count = sum(1 for row in trace_rows if row.get("flat_pointer_gate") == "FAIL")
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    adapter_product_allowed = sum(safe_int(row.get("adapter_product_allowed_query_count")) for row in sensor_rows)
    checks = [
        check_row(
            "coverage_matrix_valid",
            coverage.get("schema") == "camera_e2e_coverage_matrix_export_v1" and bool(coverage.get("validation", {}).get("pass")),
            coverage.get("validation", {}).get("status", "MISSING"),
            {"coverage_row_count": coverage.get("coverage_row_count"), "requirement_count_per_sensor": expected_requirement_count},
            "Regenerate coverage matrix.",
        ),
        check_row(
            "usage_policy_valid",
            usage_policy.get("schema") == "camera_e2e_usage_policy_v1" and bool(usage_policy.get("validation", {}).get("pass")),
            usage_policy.get("validation", {}).get("status", "MISSING"),
            {"sensor_policy_row_count": usage_policy.get("sensor_policy_row_count")},
            "Regenerate usage policy.",
        ),
        check_row(
            "adapter_smoke_valid",
            adapter_smoke.get("schema") == "camera_e2e_adapter_smoke_v1" and bool(adapter_smoke.get("validation", {}).get("pass")),
            adapter_smoke.get("validation", {}).get("status", "MISSING"),
            {"total_research_allowed_query_count": adapter_smoke.get("total_research_allowed_query_count")},
            "Run adapter smoke.",
        ),
        check_row(
            "all_requirements_mapped",
            not unmapped_requirements,
            "PASS" if not unmapped_requirements else "FAIL",
            {"unmapped_requirement_ids": sorted(unmapped_requirements)},
            "Add missing requirement ids to REQUIREMENT_LOAD_MAP.",
        ),
        check_row(
            "trace_row_count_matches_objective",
            expected_sensor_count > 0
            and expected_requirement_count > 0
            and len(trace_rows) == expected_sensor_count * expected_requirement_count
            and len(sensor_rows) == expected_sensor_count,
            "PASS",
            {"trace_row_count": len(trace_rows), "sensor_count": len(sensor_rows), "expected_sensor_count": expected_sensor_count, "expected_requirement_count": expected_requirement_count},
            "Regenerate coverage and flat bundle before objective trace.",
        ),
        check_row(
            "requirement_summary_rows_match_objective",
            expected_requirement_count > 0 and len(requirement_summary_rows) == expected_requirement_count,
            "PASS" if expected_requirement_count > 0 and len(requirement_summary_rows) == expected_requirement_count else "FAIL",
            {"requirement_summary_row_count": len(requirement_summary_rows), "expected_requirement_count": expected_requirement_count},
            "Regenerate objective trace requirement summary.",
        ),
        check_row(
            "research_trace_loadable",
            trace_fail_count == 0,
            "PASS" if trace_fail_count == 0 else "FAIL",
            {"trace_fail_count": trace_fail_count},
            "Every objective row should be mapped to a loadable flat-bundle section with adapter smoke PASS.",
        ),
        check_row(
            "flat_json_pointers_resolve",
            flat_pointer_fail_count == 0,
            "PASS" if flat_pointer_fail_count == 0 else "FAIL",
            {"flat_pointer_fail_count": flat_pointer_fail_count},
            "Every objective flat_json_pointer must resolve inside every per-sensor flat JSON.",
        ),
        check_row(
            "product_trace_blocked",
            product_ready_count == 0 and adapter_product_allowed == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_ready_count == 0 and adapter_product_allowed == 0 else "FAIL",
            {"product_ready_count": product_ready_count, "adapter_product_allowed": adapter_product_allowed},
            "Keep product trace blocked until measured/calibrated product gates pass.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_OBJECTIVE_TRACE_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = output_dir / "camera_e2e_objective_trace_by_requirement.csv"
    requirement_summary_csv = output_dir / "camera_e2e_objective_trace_by_requirement_summary.csv"
    sensor_csv = output_dir / "camera_e2e_objective_trace_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_objective_trace_checks.csv"
    report_json = output_dir / "camera_e2e_objective_trace.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_objective_trace_v1",
        "artifact_role": "camera_e2e_objective_to_loader_trace",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "requirement_count_per_sensor": expected_requirement_count,
        "requirement_summary_row_count": len(requirement_summary_rows),
        "trace_row_count": len(trace_rows),
        "trace_gate_counts": gate_counts([{key: str(value) for key, value in row.items()} for row in trace_rows], "trace_gate"),
        "flat_pointer_gate_counts": gate_counts([{key: str(value) for key, value in row.items()} for row in trace_rows], "flat_pointer_gate"),
        "flat_pointer_fail_count": flat_pointer_fail_count,
        "product_ready_count": product_ready_count,
        "adapter_product_allowed_query_count": adapter_product_allowed,
        "validation": {
            "schema": "camera_e2e_objective_trace_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "outputs": {
            "json": repo_rel(report_json),
            "trace_csv": repo_rel(trace_csv),
            "requirement_summary_csv": repo_rel(requirement_summary_csv),
            "sensor_csv": repo_rel(sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(trace_csv, trace_rows, TRACE_COLUMNS)
    write_csv(requirement_summary_csv, requirement_summary_rows, REQUIREMENT_SUMMARY_COLUMNS)
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, trace_rows, requirement_summary_rows, sensor_rows, checks)
    update_package(package_dir, payload)
    return payload


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 220) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(
    path: Path,
    payload: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    requirement_summary_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1480px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
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
<title>CameraE2E Objective Trace</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Objective Trace</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Maps every objective requirement to flat-bundle loader sections, source artifacts, and adapter-smoke evidence.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">trace status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_summary_row_count", 0))}</div><div class="muted">requirement summaries</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("trace_row_count", 0))}</div><div class="muted">requirement rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("flat_pointer_fail_count", 0))}</div><div class="muted">flat pointer failures</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Requirement Summary</h2>{html_table(requirement_summary_rows, REQUIREMENT_SUMMARY_COLUMNS)}
<h2>Sensor Trace</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Requirement Trace</h2>{html_table(trace_rows, TRACE_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_objective_trace_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_objective_trace_by_requirement_csv"] = payload["outputs"]["trace_csv"]
    outputs["camera_e2e_objective_trace_by_requirement_summary_csv"] = payload["outputs"]["requirement_summary_csv"]
    outputs["camera_e2e_objective_trace_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_objective_trace_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_objective_trace_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_objective_trace"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "requirement_count_per_sensor": payload["requirement_count_per_sensor"],
        "requirement_summary_row_count": payload["requirement_summary_row_count"],
        "trace_row_count": payload["trace_row_count"],
        "trace_gate_counts": payload["trace_gate_counts"],
        "flat_pointer_gate_counts": payload["flat_pointer_gate_counts"],
        "flat_pointer_fail_count": payload["flat_pointer_fail_count"],
        "product_ready_count": payload["product_ready_count"],
        "adapter_product_allowed_query_count": payload["adapter_product_allowed_query_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_objective_trace(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "requirement_count_per_sensor": payload["requirement_count_per_sensor"],
                "requirement_summary_row_count": payload["requirement_summary_row_count"],
                "trace_row_count": payload["trace_row_count"],
                "trace_gate_counts": payload["trace_gate_counts"],
                "flat_pointer_gate_counts": payload["flat_pointer_gate_counts"],
                "flat_pointer_fail_count": payload["flat_pointer_fail_count"],
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
