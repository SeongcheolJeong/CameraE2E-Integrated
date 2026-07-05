#!/usr/bin/env python3
"""Export flat per-sensor CameraE2E load models.

The existing consumer bundle is a strong load contract, but it still points a
consumer at many CSV tables. This exporter creates one self-contained JSON per
sensor with the rows grouped into the CameraE2E domains requested by the active
objective: Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling.

No new simulation values are created here. Product use remains blocked unless
the upstream product gates are genuinely closed.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_flat_sensor_bundle"

INDEX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "effective_ocl_mode",
    "flat_sensor_json",
    "camera_e2e_use_scope",
    "product_gate",
    "product_ready",
    "mesh_confidence_class",
    "trust_class",
    "research_utility_grade_0_10",
    "solver_evidence_grade_0_10",
    "product_accuracy_grade_0_10",
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
    "uncertainty_budget_row_count",
    "uncertainty_product_gate",
    "response_trace_row_count",
    "response_example_row_count",
    "method_provenance_row_count",
    "source_integrity_row_count",
    "objective_fulfillment_row_count",
    "crosstalk_support_row_count",
    "crosstalk_support_gate",
    "crosstalk_product_gate",
    "crosstalk_support_status",
    "crosstalk_support_summary",
    "crosstalk_support_max_required_neighborhood",
    "crosstalk_support_recommended_kernel",
    "crosstalk_support_min_truncation_fraction",
    "crosstalk_support_max_truncation_fraction",
    "crosstalk_product_candidate_row_count",
    "crosstalk_batch_priority_row_count",
    "coverage_row_count",
    "total_embedded_row_count",
    "loader_gate",
    "primary_blockers",
]

OBJECTIVE_LOAD_MAP: dict[str, dict[str, str]] = {
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


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            grouped[value].append(row)
    return dict(grouped)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value and value not in result:
            result[value] = row
    return result


def unique_sorted(rows: list[dict[str, str]], key: str) -> list[str]:
    values = {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}

    def sort_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=sort_key)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def build_objective_fulfillment_rows(
    coverage_rows: list[dict[str, str]],
    source_integrity_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    integrity_by_requirement = index_by(source_integrity_rows, "requirement_id")
    rows: list[dict[str, Any]] = []
    for coverage in sorted(coverage_rows, key=lambda row: (row.get("domain", ""), row.get("requirement_id", ""))):
        requirement_id = coverage.get("requirement_id", "")
        mapping = OBJECTIVE_LOAD_MAP.get(requirement_id, {})
        integrity = integrity_by_requirement.get(requirement_id, {})
        rows.append(
            {
                "domain": coverage.get("domain", ""),
                "requirement_id": requirement_id,
                "requirement": coverage.get("requirement", ""),
                "camera_e2e_use": coverage.get("camera_e2e_use", ""),
                "camera_e2e_loader_section": mapping.get("section", ""),
                "flat_json_pointer": mapping.get("pointer", ""),
                "primary_loader_table": mapping.get("primary", ""),
                "secondary_loader_tables": mapping.get("secondary", ""),
                "research_gate": coverage.get("research_gate", ""),
                "product_gate": coverage.get("product_gate", ""),
                "row_count": coverage.get("row_count", ""),
                "source_integrity_gate": integrity.get("source_integrity_gate", ""),
                "lut_source_class": integrity.get("lut_source_class", ""),
                "calculation_method": integrity.get("calculation_method", ""),
                "source_priority": integrity.get("source_priority", ""),
                "solver_dependency": integrity.get("solver_dependency", ""),
                "external_info_dependency": integrity.get("external_info_dependency", ""),
                "proxy_dependency": integrity.get("proxy_dependency", ""),
                "structure_specialization": integrity.get("structure_specialization", ""),
                "primary_uncertainty_quantity": integrity.get("primary_uncertainty_quantity", ""),
                "primary_uncertainty_min": integrity.get("primary_uncertainty_min", ""),
                "primary_uncertainty_max": integrity.get("primary_uncertainty_max", ""),
                "primary_uncertainty_unit": integrity.get("primary_uncertainty_unit", ""),
                "uncertainty_camera_e2e_use": integrity.get("uncertainty_camera_e2e_use", ""),
                "uncertainty_product_gate": integrity.get("uncertainty_product_gate", ""),
                "recommended_camera_e2e_use": integrity.get("recommended_camera_e2e_use", ""),
                "source_artifacts": coverage.get("source_artifacts", "") or integrity.get("source_artifacts", ""),
                "primary_blocker": coverage.get("primary_blocker", "") or integrity.get("primary_blocker", ""),
                "next_action": integrity.get("next_action", ""),
            }
        )
    return rows


def crosstalk_support_index_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "support_gate": "MISSING",
            "product_gate": "MISSING",
            "product_gate_counts": {},
            "status": "NO_FINITE_ARRAY_SUPPORT_PILOT",
            "summary": "",
            "max_required_neighborhood": "",
            "recommended_kernel": "",
            "min_truncation_fraction": "",
            "max_truncation_fraction": "",
        }

    truncations = [
        value
        for value in (safe_float(row.get("best_pilot_truncation_fraction")) for row in rows)
        if value is not None
    ]
    neighborhoods = [safe_int(row.get("best_pilot_neighborhood")) for row in rows if row.get("best_pilot_neighborhood")]
    descriptors = []
    for row in sorted(rows, key=lambda item: (item.get("field_case", ""), safe_float(item.get("wavelength_nm"), 0.0) or 0.0, item.get("color_channel", ""))):
        channel = row.get("color_channel", "")
        wavelength = row.get("wavelength_nm", "")
        field = row.get("field_case", "")
        neighborhood = row.get("best_pilot_neighborhood", "")
        truncation = row.get("best_pilot_truncation_fraction", "")
        descriptors.append(f"{channel}@{wavelength}nm/{field}:n{neighborhood},trunc={truncation}")
    product_gate_counts = gate_counts(rows, "product_crosstalk_gate")
    product_gates = set(product_gate_counts)
    if "FAIL" in product_gates:
        product_gate = "FAIL"
    elif "MISSING" in product_gates:
        product_gate = "MISSING"
    elif "CHECK" in product_gates:
        product_gate = "CHECK"
    elif "PASS" in product_gates:
        product_gate = "PASS"
    else:
        product_gate = "MISSING"
    max_neighborhood = max(neighborhoods) if neighborhoods else ""
    status = (
        "LOW_RES_SUPPORT_PILOT_ONLY_PRODUCT_BLOCKED"
        if product_gate in {"FAIL", "MISSING"}
        else "FINITE_ARRAY_SUPPORT_REVIEW_REQUIRED"
    )
    return {
        "support_gate": "CHECK",
        "product_gate": product_gate,
        "product_gate_counts": product_gate_counts,
        "status": status,
        "summary": "; ".join(descriptors),
        "max_required_neighborhood": max_neighborhood,
        "recommended_kernel": f"{max_neighborhood}x{max_neighborhood}" if max_neighborhood else "",
        "min_truncation_fraction": min(truncations) if truncations else "",
        "max_truncation_fraction": max(truncations) if truncations else "",
    }


def source_paths(package_dir: Path) -> dict[str, Path]:
    return {
        "consumer_bundle": package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_bundle.json",
        "consumer_sensor_index": package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv",
        "sensor_model_summary": package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv",
        "runtime_lut": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv",
        "runtime_crosstalk_kernel": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv",
        "spectral_response": package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv",
        "color_matrix_seed": package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv",
        "material_nk_lut": package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv",
        "cfa_provenance_by_sensor": package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv",
        "cfa_db_by_sensor": package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_by_sensor.csv",
        "cfa_db_transmission_lut": package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_transmission_lut.csv",
        "electrical_noise_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_noise_lut.csv",
        "readout_gain_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_readout_gain_lut.csv",
        "binning_remosaic_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_binning_remosaic_lut.csv",
        "module_coupling_lut": package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_field_lut.csv",
        "uncertainty_budget": package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_budget.csv",
        "uncertainty_by_sensor": package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_by_sensor.csv",
        "response_trace": package_dir / "camera_e2e_response_trace" / "camera_e2e_response_trace.csv",
        "response_trace_summary": package_dir / "camera_e2e_response_trace" / "camera_e2e_response_trace_summary.csv",
        "response_example": package_dir / "camera_e2e_response_example" / "camera_e2e_response_example.csv",
        "response_example_summary": package_dir / "camera_e2e_response_example" / "camera_e2e_response_example_summary.csv",
        "method_provenance_matrix": package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_matrix.csv",
        "method_provenance_by_sensor": package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_by_sensor.csv",
        "source_integrity_matrix": package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_matrix.csv",
        "source_integrity_by_sensor": package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_by_sensor.csv",
        "coverage_matrix": package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv",
        "use_scope_by_sensor": package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_sensor.csv",
        "use_scope_by_domain": package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_domain.csv",
        "lut_trust_by_sensor": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv",
        "lut_trust_by_domain": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv",
        "lut_trust_by_requirement": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_requirement.csv",
        "mesh_confidence_by_sensor": package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv",
        "capability_profile_by_sensor": package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv",
        "crosstalk_support_by_sensor": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_by_sensor.csv",
        "crosstalk_product_candidates": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_product_candidates.csv",
        "crosstalk_batch_priority": package_dir / "camera_e2e_crosstalk_batch_priority" / "camera_e2e_crosstalk_batch_priority.csv",
        "probe_summary": package_dir / "camera_e2e_sensor_probe_all_sensors" / "camera_e2e_sensor_probe_summary.csv",
        "quantitative_point_queue": package_dir / "camera_e2e_quantitative_point_queue.csv",
        "resource_limited_batch_plan": package_dir / "camera_e2e_resource_limited_batch_plan.csv",
        "quantitative_coverage": package_dir / "camera_e2e_quantitative_coverage.csv",
    }


def count_rows_or_one(row_or_rows: dict[str, Any] | list[dict[str, Any]]) -> int:
    if isinstance(row_or_rows, list):
        return len(row_or_rows)
    return 1 if row_or_rows else 0


def build_sensor_payload(
    *,
    slug: str,
    consumer_row: dict[str, str],
    model_row: dict[str, str],
    paths: dict[str, Path],
    groups: dict[str, dict[str, list[dict[str, str]]]],
    indexes: dict[str, dict[str, dict[str, str]]],
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_rows = groups["runtime"].get(slug, [])
    runtime_ids = {row.get("runtime_id", "") for row in runtime_rows if row.get("runtime_id")}
    kernel_rows_by_slug = groups["kernel"].get(slug, [])
    kernel_rows = [row for row in kernel_rows_by_slug if not runtime_ids or row.get("runtime_id") in runtime_ids]
    spectral_rows = groups["spectral"].get(slug, [])
    material_rows = groups["material"].get(slug, [])
    cfa_db_transmission_rows = groups["cfa_db_transmission"].get(slug, [])
    electrical_rows = groups["electrical"].get(slug, [])
    readout_rows = groups["readout"].get(slug, [])
    binning_rows = groups["binning"].get(slug, [])
    module_rows = groups["module"].get(slug, [])
    uncertainty_rows = groups["uncertainty"].get(slug, [])
    response_trace_rows = groups["response_trace"].get(slug, [])
    response_example_rows = groups["response_example"].get(slug, [])
    method_provenance_rows = groups["method_provenance"].get(slug, [])
    source_integrity_rows = groups["source_integrity"].get(slug, [])
    coverage_rows = groups["coverage"].get(slug, [])
    use_scope_domain_rows = groups["use_scope_domain"].get(slug, [])
    trust_domain_rows = groups["trust_domain"].get(slug, [])
    trust_requirement_rows = groups["trust_requirement"].get(slug, [])
    support_rows = groups["crosstalk_support"].get(slug, [])
    product_candidate_rows = groups["crosstalk_product_candidates"].get(slug, [])
    batch_priority_rows = groups["crosstalk_batch_priority"].get(slug, [])
    probe_rows = groups["probe_summary"].get(slug, [])
    quantitative_queue_rows = groups["quantitative_queue"].get(slug, [])
    resource_limited_rows = groups["resource_limited"].get(slug, [])
    quantitative_coverage_rows = groups["quantitative_coverage"].get(slug, [])

    cfa_provenance = indexes["cfa_provenance"].get(slug, {})
    cfa_db = indexes["cfa_db"].get(slug, {})
    color_matrix = indexes["color_matrix"].get(slug, {})
    use_scope = indexes["use_scope_sensor"].get(slug, {})
    trust = indexes["trust_sensor"].get(slug, {})
    mesh = indexes["mesh"].get(slug, {})
    capability = indexes["capability"].get(slug, {})
    uncertainty_sensor = indexes["uncertainty_sensor"].get(slug, {})
    response_trace_summary = indexes["response_trace_summary"].get(slug, {})
    response_example_summary = indexes["response_example_summary"].get(slug, {})
    method_provenance_summary = indexes["method_provenance_sensor"].get(slug, {})
    source_integrity_summary = indexes["source_integrity_sensor"].get(slug, {})
    objective_fulfillment_rows = build_objective_fulfillment_rows(coverage_rows, source_integrity_rows)

    row_counts = {
        "runtime": len(runtime_rows),
        "kernel": len(kernel_rows),
        "spectral": len(spectral_rows),
        "color_matrix": count_rows_or_one(color_matrix),
        "material": len(material_rows),
        "cfa_provenance": count_rows_or_one(cfa_provenance),
        "cfa_db": count_rows_or_one(cfa_db),
        "cfa_db_transmission": len(cfa_db_transmission_rows),
        "electrical": len(electrical_rows),
        "readout": len(readout_rows),
        "binning": len(binning_rows),
        "module_field": len(module_rows),
        "uncertainty_budget": len(uncertainty_rows),
        "response_trace": len(response_trace_rows),
        "response_example": len(response_example_rows),
        "method_provenance": len(method_provenance_rows),
        "source_integrity": len(source_integrity_rows),
        "objective_fulfillment": len(objective_fulfillment_rows),
        "coverage": len(coverage_rows),
        "use_scope_sensor": count_rows_or_one(use_scope),
        "use_scope_domain": len(use_scope_domain_rows),
        "trust_sensor": count_rows_or_one(trust),
        "trust_domain": len(trust_domain_rows),
        "trust_requirement": len(trust_requirement_rows),
        "mesh_confidence": count_rows_or_one(mesh),
        "capability": count_rows_or_one(capability),
        "crosstalk_support": len(support_rows),
        "crosstalk_product_candidates": len(product_candidate_rows),
        "crosstalk_batch_priority": len(batch_priority_rows),
        "probe_summary": len(probe_rows),
        "quantitative_queue": len(quantitative_queue_rows),
        "resource_limited_batch": len(resource_limited_rows),
        "quantitative_coverage": len(quantitative_coverage_rows),
    }
    total_embedded_row_count = sum(row_counts.values())
    product_ready = boolish(use_scope.get("product_ready") or consumer_row.get("product_ready") or model_row.get("camera_e2e_product_ready"))

    source_tables = {key: repo_rel(path) for key, path in paths.items()}
    payload = {
        "schema": "camera_e2e_flat_sensor_model_v1",
        "artifact_role": "flat_per_sensor_camera_e2e_load_model",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor": {
            "slug": slug,
            "code": consumer_row.get("code") or model_row.get("code", ""),
            "manufacturer": consumer_row.get("manufacturer") or model_row.get("manufacturer", ""),
            "device_name": consumer_row.get("device_name") or model_row.get("device_name", ""),
            "pixel_pitch_um": model_row.get("pixel_pitch_um") or mesh.get("pixel_pitch_um", ""),
            "cfa_pattern": model_row.get("cfa_pattern") or mesh.get("cfa_pattern", ""),
            "effective_ocl_mode": model_row.get("effective_ocl_mode") or mesh.get("ocl_mode_guess", ""),
        },
        "gates": {
            "research_gate": "CHECK",
            "product_gate": use_scope.get("product_gate") or consumer_row.get("product_bundle_gate") or "FAIL",
            "product_ready": product_ready,
            "coverage_research_gate_counts": gate_counts(coverage_rows, "research_gate"),
            "coverage_product_gate_counts": gate_counts(coverage_rows, "product_gate"),
            "runtime_gate_counts": gate_counts(runtime_rows, "combined_evidence_gate"),
            "kernel_evidence_gate_counts": gate_counts(kernel_rows, "evidence_gate"),
        },
        "row_counts": row_counts,
        "total_embedded_row_count": total_embedded_row_count,
        "optical_color": {
            "cfa_db_by_sensor": cfa_db,
            "cfa_db_transmission_lut": cfa_db_transmission_rows,
            "cfa_provenance": cfa_provenance,
            "material_nk_lut": material_rows,
            "spectral_response": spectral_rows,
            "color_matrix_seed": color_matrix,
            "runtime_field_response_lut": runtime_rows,
            "optical_crosstalk_kernel_lut": kernel_rows,
            "crosstalk_support_rows": support_rows,
            "crosstalk_product_candidate_rows": product_candidate_rows,
            "crosstalk_batch_priority_rows": batch_priority_rows,
            "angular_cra_response_rows": {
                "runtime_response_rows": runtime_rows,
                "module_field_rows": module_rows,
            },
            "microlens_ocl_shift_rows": {
                "runtime_lens_shift_rows": runtime_rows,
                "module_field_rows": module_rows,
            },
        },
        "pixel_electrical": {
            "electrical_noise_lut": electrical_rows,
            "charge_collection_and_electrical_crosstalk_columns": [
                "charge_collection_electrical_crosstalk_gate",
                "electrical_collection_efficiency_prior",
                "electrical_crosstalk_fraction_prior",
                "electrical_crosstalk_fraction_min",
                "electrical_crosstalk_fraction_max",
                "electrical_diffusion_length_um",
            ],
        },
        "readout_raw": {
            "readout_gain_lut": readout_rows,
            "binning_remosaic_lut": binning_rows,
        },
        "module_coupling": {
            "module_field_lut": module_rows,
            "cra_and_alignment_columns": [
                "field_case",
                "field_x_norm",
                "field_z_norm",
                "cra_x_deg",
                "cra_z_deg",
                "lens_shift_x_um",
                "lens_shift_z_um",
                "sensor_decenter_x_um",
                "sensor_decenter_z_um",
                "sensor_tilt_x_deg",
                "sensor_tilt_z_deg",
                "relative_illumination_cos4",
                "pupil_relative_transmission",
            ],
        },
        "uncertainty_budget": {
            "uncertainty_by_sensor": uncertainty_sensor,
            "uncertainty_budget_rows": uncertainty_rows,
            "policy": "Use for CameraE2E sensitivity bands only; product use still requires measured/calibrated gates.",
        },
        "response_trace": {
            "response_trace_summary": response_trace_summary,
            "response_trace_rows": response_trace_rows,
            "policy": "Use to explain runtime response construction and material sanity inputs; cfa_times_si_simple_fraction is not a Meep replacement.",
        },
        "response_example": {
            "response_example_summary": response_example_summary,
            "response_example_rows": response_example_rows,
            "policy": "Use as a compact human-readable CFA-to-Si-to-QE example before interpreting runtime_response_nominal rows.",
        },
        "method_provenance": {
            "method_provenance_by_sensor": method_provenance_summary,
            "method_provenance_matrix_rows": method_provenance_rows,
            "policy": "Use before consuming numeric LUT rows to distinguish solver, external DB, structural topology, and proxy/prior values.",
        },
        "source_integrity": {
            "source_integrity_by_sensor": source_integrity_summary,
            "source_integrity_matrix_rows": source_integrity_rows,
            "policy": "Use as the single joined source/method/uncertainty table before routing values into CameraE2E.",
        },
        "objective_fulfillment": {
            "requirement_rows": objective_fulfillment_rows,
            "row_count": len(objective_fulfillment_rows),
            "join_key": "slug + requirement_id",
            "policy": (
                "Per-sensor fulfillment map for the active CameraE2E objective. "
                "Use flat_json_pointer and primary_loader_table to load values; preserve source class, uncertainty, and product gates."
            ),
        },
        "camera_e2e_routing": {
            "import_decision": {
                "camera_e2e_use_scope": use_scope.get("camera_e2e_use_scope", ""),
                "trust_class": trust.get("trust_class", ""),
                "research_utility_grade_0_10": trust.get("research_utility_grade_0_10", ""),
                "solver_evidence_grade_0_10": trust.get("solver_evidence_grade_0_10", ""),
                "product_accuracy_grade_0_10": trust.get("product_accuracy_grade_0_10", ""),
                "crosstalk_support_status": trust.get("crosstalk_support_status", ""),
                "crosstalk_support_recommended_kernel": trust.get("crosstalk_support_recommended_kernel", ""),
                "product_ready": product_ready,
                "product_gate": use_scope.get("product_gate") or consumer_row.get("product_bundle_gate") or "FAIL",
            },
            "use_scope_by_sensor": use_scope,
            "use_scope_by_domain": use_scope_domain_rows,
            "capability_profile": capability,
            "uncertainty_by_sensor": uncertainty_sensor,
            "response_trace_summary": response_trace_summary,
            "response_example_summary": response_example_summary,
            "response_example_rows": response_example_rows,
            "method_provenance_by_sensor": method_provenance_summary,
            "method_provenance_matrix": method_provenance_rows,
            "source_integrity_by_sensor": source_integrity_summary,
            "source_integrity_matrix": source_integrity_rows,
            "objective_fulfillment": objective_fulfillment_rows,
            "lut_trust_by_sensor": trust,
            "lut_trust_by_domain": trust_domain_rows,
            "lut_trust_by_requirement": trust_requirement_rows,
            "mesh_confidence": mesh,
            "coverage_matrix": coverage_rows,
            "probe_summary": probe_rows,
        },
        "solver_and_closure_inputs": {
            "quantitative_point_queue": quantitative_queue_rows,
            "resource_limited_batch_plan": resource_limited_rows,
            "quantitative_coverage": quantitative_coverage_rows,
        },
        "source_tables": source_tables,
        "join_keys": {
            "sensor_level": "slug",
            "runtime_to_kernel": "runtime_id",
            "runtime_to_color": "slug + color_channel + wavelength_nm",
            "runtime_to_material": "slug + material_family/material_key + color_channel + wavelength_nm",
            "cfa_db": "slug + color_channel + wavelength_nm",
            "coverage": "slug + requirement_id",
            "method_provenance": "slug + requirement_id",
            "source_integrity": "slug + requirement_id",
            "objective_fulfillment": "slug + requirement_id",
            "module_field": "slug + field_case + wavelength_nm",
            "uncertainty": "slug + domain + quantity",
            "response_trace": "slug + runtime_id",
            "response_example": "slug + color_channel",
            "electrical": "slug + temperature_c + exposure_s + signal_fraction",
            "readout": "slug + analog_gain_x + digital_gain_x + adc_bit_depth",
            "binning": "slug + mode_id",
        },
        "policy": {
            "research_use": "Allowed only for research/trend/plumbing according to camera_e2e_routing gates.",
            "product_use": "Blocked until measured stack/material/CRA/electrical/readout/module calibration and quantitative field/crosstalk convergence gates pass.",
            "important_limitation": (
                "This flat model improves loadability, not physical accuracy. "
                "Do not treat proxy/prior rows or incomplete mesh coverage as product LUT evidence."
            ),
        },
    }
    write_json(output_path, payload)

    loader_issues: list[str] = []
    for key in (
        "runtime",
        "kernel",
        "spectral",
        "material",
        "cfa_db",
        "cfa_db_transmission",
        "electrical",
        "readout",
        "binning",
        "module_field",
        "uncertainty_budget",
        "response_trace",
        "response_example",
        "method_provenance",
        "source_integrity",
        "objective_fulfillment",
        "coverage",
    ):
        if row_counts.get(key, 0) <= 0:
            loader_issues.append(f"{key} rows missing")
    if row_counts["objective_fulfillment"] != row_counts["coverage"]:
        loader_issues.append("objective_fulfillment row count does not match coverage")
    if row_counts["objective_fulfillment"] != row_counts["source_integrity"]:
        loader_issues.append("objective_fulfillment row count does not match source_integrity")
    if product_ready:
        loader_issues.append("unexpected product_ready true in research package")
    support_summary = crosstalk_support_index_summary(support_rows)
    index_row = {
        "slug": slug,
        "code": payload["sensor"]["code"],
        "manufacturer": payload["sensor"]["manufacturer"],
        "device_name": payload["sensor"]["device_name"],
        "pixel_pitch_um": payload["sensor"]["pixel_pitch_um"],
        "cfa_pattern": payload["sensor"]["cfa_pattern"],
        "effective_ocl_mode": payload["sensor"]["effective_ocl_mode"],
        "flat_sensor_json": repo_rel(output_path),
        "camera_e2e_use_scope": use_scope.get("camera_e2e_use_scope", ""),
        "product_gate": payload["gates"]["product_gate"],
        "product_ready": product_ready,
        "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
        "trust_class": trust.get("trust_class", ""),
        "research_utility_grade_0_10": trust.get("research_utility_grade_0_10", ""),
        "solver_evidence_grade_0_10": trust.get("solver_evidence_grade_0_10", ""),
        "product_accuracy_grade_0_10": trust.get("product_accuracy_grade_0_10", ""),
        "runtime_row_count": row_counts["runtime"],
        "kernel_row_count": row_counts["kernel"],
        "spectral_row_count": row_counts["spectral"],
        "material_row_count": row_counts["material"],
        "cfa_db_row_count": row_counts["cfa_db"],
        "cfa_db_transmission_row_count": row_counts["cfa_db_transmission"],
        "electrical_row_count": row_counts["electrical"],
        "readout_row_count": row_counts["readout"],
        "binning_row_count": row_counts["binning"],
        "module_field_row_count": row_counts["module_field"],
        "uncertainty_budget_row_count": row_counts["uncertainty_budget"],
        "uncertainty_product_gate": uncertainty_sensor.get("uncertainty_product_gate", ""),
        "response_trace_row_count": row_counts["response_trace"],
        "response_example_row_count": row_counts["response_example"],
        "method_provenance_row_count": row_counts["method_provenance"],
        "source_integrity_row_count": row_counts["source_integrity"],
        "objective_fulfillment_row_count": row_counts["objective_fulfillment"],
        "crosstalk_support_row_count": row_counts["crosstalk_support"],
        "crosstalk_support_gate": support_summary["support_gate"],
        "crosstalk_product_gate": support_summary["product_gate"],
        "crosstalk_support_status": support_summary["status"],
        "crosstalk_support_summary": support_summary["summary"],
        "crosstalk_support_max_required_neighborhood": support_summary["max_required_neighborhood"],
        "crosstalk_support_recommended_kernel": support_summary["recommended_kernel"],
        "crosstalk_support_min_truncation_fraction": support_summary["min_truncation_fraction"],
        "crosstalk_support_max_truncation_fraction": support_summary["max_truncation_fraction"],
        "crosstalk_product_candidate_row_count": row_counts["crosstalk_product_candidates"],
        "crosstalk_batch_priority_row_count": row_counts["crosstalk_batch_priority"],
        "coverage_row_count": row_counts["coverage"],
        "total_embedded_row_count": total_embedded_row_count,
        "loader_gate": "FAIL" if loader_issues else "PASS",
        "primary_blockers": use_scope.get("primary_blockers") or consumer_row.get("primary_blockers") or "; ".join(loader_issues),
    }
    return payload, index_row


def validate(index_rows: list[dict[str, Any]], sensor_paths: list[Path]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not index_rows:
        issues.append({"severity": "error", "code": "no_sensors", "message": "No flat sensor models were exported."})
    if len(index_rows) != len(sensor_paths):
        issues.append({"severity": "error", "code": "sensor_path_count_mismatch", "message": "Index rows and sensor JSON paths differ."})
    for row in index_rows:
        slug = row.get("slug", "")
        if row.get("loader_gate") != "PASS":
            issues.append({"severity": "error", "code": "flat_sensor_loader_gate_failed", "slug": slug, "message": row.get("primary_blockers", "")})
        if boolish(row.get("product_ready")):
            issues.append({"severity": "error", "code": "unexpected_product_ready", "slug": slug})
        for key in (
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
            "objective_fulfillment_row_count",
            "coverage_row_count",
        ):
            if safe_int(row.get(key)) <= 0:
                issues.append({"severity": "error", "code": "required_row_count_empty", "slug": slug, "row_count_key": key})
    product_ready_count = sum(1 for row in index_rows if boolish(row.get("product_ready")))
    if any(issue.get("severity") == "error" for issue in issues):
        status = "FAIL"
    elif product_ready_count:
        status = "PRODUCT_FLAT_SENSOR_BUNDLE_READY"
    else:
        status = "RESEARCH_FLAT_SENSOR_BUNDLE_READY_PRODUCT_BLOCKED"
    return {
        "schema": "camera_e2e_flat_sensor_bundle_validation_v1",
        "pass": not any(issue.get("severity") == "error" for issue in issues),
        "status": status,
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "issues": issues,
    }


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


def write_html(path: Path, payload: dict[str, Any], index_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1380px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
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
<title>CameraE2E Flat Sensor Bundle</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Flat Sensor Bundle</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. One JSON per sensor embeds the rows needed by CameraE2E research loaders.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">bundle status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("total_embedded_row_count", 0))}</div><div class="muted">embedded source rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Usage</h2>
<p>Load <code>camera_e2e_flat_sensor_bundle.json</code>, pick a sensor path from <code>sensor_model_json_files</code>, then read the domain groups: <code>optical_color</code>, <code>pixel_electrical</code>, <code>readout_raw</code>, and <code>module_coupling</code>.</p>
<p class="muted">This improves integration ergonomics only. It does not upgrade proxy material, sparse mesh, or prior electrical rows to product accuracy.</p>
<h2>Sensor Index</h2>
{html_table(index_rows, INDEX_COLUMNS)}
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
    outputs["camera_e2e_flat_sensor_bundle_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_flat_sensor_index_csv"] = payload["outputs"]["index_csv"]
    outputs["camera_e2e_flat_sensor_bundle_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_flat_sensor_bundle"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        "total_embedded_row_count": payload["total_embedded_row_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_flat_bundle(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensors_dir = output_dir / "sensors"
    paths = source_paths(package_dir)

    consumer_rows = read_csv_rows(paths["consumer_sensor_index"])
    model_rows = read_csv_rows(paths["sensor_model_summary"])
    model_by_slug = index_by(model_rows, "slug")

    groups = {
        "runtime": group_rows(read_csv_rows(paths["runtime_lut"]), "slug"),
        "kernel": group_rows(read_csv_rows(paths["runtime_crosstalk_kernel"]), "slug"),
        "spectral": group_rows(read_csv_rows(paths["spectral_response"]), "slug"),
        "material": group_rows(read_csv_rows(paths["material_nk_lut"]), "slug"),
        "cfa_db_transmission": group_rows(read_csv_rows(paths["cfa_db_transmission_lut"]), "slug"),
        "electrical": group_rows(read_csv_rows(paths["electrical_noise_lut"]), "slug"),
        "readout": group_rows(read_csv_rows(paths["readout_gain_lut"]), "slug"),
        "binning": group_rows(read_csv_rows(paths["binning_remosaic_lut"]), "slug"),
        "module": group_rows(read_csv_rows(paths["module_coupling_lut"]), "slug"),
        "uncertainty": group_rows(read_csv_rows(paths["uncertainty_budget"]), "slug"),
        "response_trace": group_rows(read_csv_rows(paths["response_trace"]), "slug"),
        "response_example": group_rows(read_csv_rows(paths["response_example"]), "slug"),
        "method_provenance": group_rows(read_csv_rows(paths["method_provenance_matrix"]), "slug"),
        "source_integrity": group_rows(read_csv_rows(paths["source_integrity_matrix"]), "slug"),
        "coverage": group_rows(read_csv_rows(paths["coverage_matrix"]), "slug"),
        "use_scope_domain": group_rows(read_csv_rows(paths["use_scope_by_domain"]), "slug"),
        "trust_domain": group_rows(read_csv_rows(paths["lut_trust_by_domain"]), "slug"),
        "trust_requirement": group_rows(read_csv_rows(paths["lut_trust_by_requirement"]), "slug"),
        "crosstalk_support": group_rows(read_csv_rows(paths["crosstalk_support_by_sensor"]), "slug"),
        "crosstalk_product_candidates": group_rows(read_csv_rows(paths["crosstalk_product_candidates"]), "slug"),
        "crosstalk_batch_priority": group_rows(read_csv_rows(paths["crosstalk_batch_priority"]), "slug"),
        "probe_summary": group_rows(read_csv_rows(paths["probe_summary"]), "slug"),
        "quantitative_queue": group_rows(read_csv_rows(paths["quantitative_point_queue"]), "slug"),
        "resource_limited": group_rows(read_csv_rows(paths["resource_limited_batch_plan"]), "slug"),
        "quantitative_coverage": group_rows(read_csv_rows(paths["quantitative_coverage"]), "slug"),
    }
    indexes = {
        "cfa_provenance": index_by(read_csv_rows(paths["cfa_provenance_by_sensor"]), "slug"),
        "cfa_db": index_by(read_csv_rows(paths["cfa_db_by_sensor"]), "slug"),
        "color_matrix": index_by(read_csv_rows(paths["color_matrix_seed"]), "slug"),
        "use_scope_sensor": index_by(read_csv_rows(paths["use_scope_by_sensor"]), "slug"),
        "trust_sensor": index_by(read_csv_rows(paths["lut_trust_by_sensor"]), "slug"),
        "mesh": index_by(read_csv_rows(paths["mesh_confidence_by_sensor"]), "slug"),
        "capability": index_by(read_csv_rows(paths["capability_profile_by_sensor"]), "slug"),
        "uncertainty_sensor": index_by(read_csv_rows(paths["uncertainty_by_sensor"]), "slug"),
        "response_trace_summary": index_by(read_csv_rows(paths["response_trace_summary"]), "slug"),
        "response_example_summary": index_by(read_csv_rows(paths["response_example_summary"]), "slug"),
        "method_provenance_sensor": index_by(read_csv_rows(paths["method_provenance_by_sensor"]), "slug"),
        "source_integrity_sensor": index_by(read_csv_rows(paths["source_integrity_by_sensor"]), "slug"),
    }

    index_rows: list[dict[str, Any]] = []
    sensor_paths: list[Path] = []
    total_embedded_row_count = 0
    for consumer_row in consumer_rows:
        slug = consumer_row.get("slug", "")
        if not slug:
            continue
        output_path = sensors_dir / f"{slug}.json"
        sensor_payload, index_row = build_sensor_payload(
            slug=slug,
            consumer_row=consumer_row,
            model_row=model_by_slug.get(slug, {}),
            paths=paths,
            groups=groups,
            indexes=indexes,
            output_path=output_path,
        )
        total_embedded_row_count += safe_int(sensor_payload.get("total_embedded_row_count"))
        index_rows.append(index_row)
        sensor_paths.append(output_path)

    validation = validate(index_rows, sensor_paths)
    index_csv = output_dir / "camera_e2e_flat_sensor_index.csv"
    bundle_json = output_dir / "camera_e2e_flat_sensor_bundle.json"
    html_path = output_dir / "index.html"
    product_ready_count = sum(1 for row in index_rows if boolish(row.get("product_ready")))
    payload = {
        "schema": "camera_e2e_flat_sensor_bundle_v1",
        "artifact_role": "self_contained_per_sensor_camera_e2e_load_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(index_rows),
        "product_ready_count": product_ready_count,
        "total_embedded_row_count": total_embedded_row_count,
        "row_count_summary": {
            "runtime": sum(safe_int(row.get("runtime_row_count")) for row in index_rows),
            "kernel": sum(safe_int(row.get("kernel_row_count")) for row in index_rows),
            "spectral": sum(safe_int(row.get("spectral_row_count")) for row in index_rows),
            "material": sum(safe_int(row.get("material_row_count")) for row in index_rows),
            "cfa_db_transmission": sum(safe_int(row.get("cfa_db_transmission_row_count")) for row in index_rows),
            "electrical": sum(safe_int(row.get("electrical_row_count")) for row in index_rows),
            "readout": sum(safe_int(row.get("readout_row_count")) for row in index_rows),
            "binning": sum(safe_int(row.get("binning_row_count")) for row in index_rows),
            "module_field": sum(safe_int(row.get("module_field_row_count")) for row in index_rows),
            "response_example": sum(safe_int(row.get("response_example_row_count")) for row in index_rows),
            "method_provenance": sum(safe_int(row.get("method_provenance_row_count")) for row in index_rows),
            "source_integrity": sum(safe_int(row.get("source_integrity_row_count")) for row in index_rows),
            "objective_fulfillment": sum(safe_int(row.get("objective_fulfillment_row_count")) for row in index_rows),
            "crosstalk_support": sum(safe_int(row.get("crosstalk_support_row_count")) for row in index_rows),
            "crosstalk_product_candidates": sum(safe_int(row.get("crosstalk_product_candidate_row_count")) for row in index_rows),
            "crosstalk_batch_priority": sum(safe_int(row.get("crosstalk_batch_priority_row_count")) for row in index_rows),
            "coverage": sum(safe_int(row.get("coverage_row_count")) for row in index_rows),
        },
        "validation": validation,
        "sensor_model_json_files": [repo_rel(path) for path in sensor_paths],
        "source_tables": {key: repo_rel(path) for key, path in paths.items()},
        "usage_policy": {
            "recommended_entrypoint": "Load this bundle, then one JSON from sensor_model_json_files.",
            "domain_groups": ["optical_color", "pixel_electrical", "readout_raw", "module_coupling"],
            "research": "Research/prototyping load is valid when validation.pass is true and per-row gates are preserved.",
            "product": "Product use remains blocked until product_ready is true for the target sensor and all upstream product gates pass.",
        },
        "outputs": {
            "json": repo_rel(bundle_json),
            "index_csv": repo_rel(index_csv),
            "sensors_dir": repo_rel(sensors_dir),
            "html": repo_rel(html_path),
        },
    }
    write_csv(index_csv, index_rows, INDEX_COLUMNS)
    write_json(bundle_json, payload)
    write_html(html_path, payload, index_rows)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = export_flat_bundle(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "product_ready_count": payload["product_ready_count"],
                "total_embedded_row_count": payload["total_embedded_row_count"],
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
