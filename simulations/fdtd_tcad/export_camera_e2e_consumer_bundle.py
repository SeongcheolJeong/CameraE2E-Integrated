#!/usr/bin/env python3
"""Export a CameraE2E consumer bundle.

This is the thin integration layer above the individual LUT exports. It creates
one per-sensor JSON manifest plus a package-level index so a CameraE2E consumer
can load the research package without reverse-engineering every artifact folder.

The exporter does not create new simulation values and does not promote prior
data to product accuracy. It only turns the existing, gated artifacts into a
consumer contract with explicit join keys, row counts, and blockers.
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

from export_camera_e2e_flat_sensor_bundle import OBJECTIVE_LOAD_MAP


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_consumer_bundle"

INDEX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "research_bundle_gate",
    "product_bundle_gate",
    "product_ready",
    "coverage_requirement_count",
    "objective_fulfillment_row_count",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "source_integrity_gate_counts",
    "mesh_confidence_class",
    "camera_e2e_recommended_use",
    "mesh_field_pass_points",
    "mesh_field_required_points",
    "mesh_crosstalk_pass_points",
    "mesh_crosstalk_required_points",
    "mesh_confidence_row_count",
    "crosstalk_support_gate",
    "crosstalk_support_best_neighborhood",
    "crosstalk_support_best_truncation_fraction",
    "crosstalk_support_summary",
    "crosstalk_support_max_required_neighborhood",
    "crosstalk_support_min_truncation_fraction",
    "crosstalk_support_max_truncation_fraction",
    "crosstalk_support_threshold",
    "crosstalk_support_recommendation",
    "crosstalk_support_row_count",
    "crosstalk_product_candidate_row_count",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "cfa_pattern_source_kind",
    "cfa_provenance_row_count",
    "cfa_db_row_count",
    "cfa_db_transmission_row_count",
    "capability_overall_use_scope",
    "capability_spectral_qe_scope",
    "capability_color_response_scope",
    "capability_crosstalk_scope",
    "capability_row_count",
    "lut_trust_class",
    "lut_trust_research_score_0_100",
    "lut_trust_evidence_score_0_100",
    "lut_trust_product_score_0_100",
    "lut_trust_research_grade_0_10",
    "lut_trust_solver_evidence_grade_0_10",
    "lut_trust_product_accuracy_grade_0_10",
    "lut_trust_crosstalk_support_status",
    "lut_trust_crosstalk_support_recommended_kernel",
    "lut_trust_row_count",
    "quantitative_queue_row_count",
    "resource_limited_batch_row_count",
    "runtime_row_count",
    "kernel_row_count",
    "spectral_row_count",
    "color_matrix_gate",
    "material_row_count",
    "electrical_row_count",
    "readout_row_count",
    "binning_row_count",
    "module_field_row_count",
    "probe_summary_row_count",
    "load_manifest_json",
    "primary_blockers",
]

ARTIFACT_COLUMNS = [
    "artifact_id",
    "role",
    "path",
    "exists",
    "schema",
    "row_count",
    "required_for_consumer",
    "notes",
]


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


def unique_values(rows: list[dict[str, str]], key: str) -> set[str]:
    return {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}


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
                "consumer_manifest_pointer": "/objective_fulfillment/requirement_rows",
                "flat_json_pointer": mapping.get("pointer", ""),
                "camera_e2e_loader_section": mapping.get("section", ""),
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


def build_requirement_load_map(coverage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for coverage in sorted(coverage_rows, key=lambda row: (row.get("domain", ""), row.get("requirement_id", ""))):
        requirement_id = coverage.get("requirement_id", "")
        if not requirement_id or requirement_id in seen:
            continue
        seen.add(requirement_id)
        mapping = OBJECTIVE_LOAD_MAP.get(requirement_id, {})
        result.append(
            {
                "domain": coverage.get("domain", ""),
                "requirement_id": requirement_id,
                "requirement": coverage.get("requirement", ""),
                "camera_e2e_use": coverage.get("camera_e2e_use", ""),
                "flat_json_pointer": mapping.get("pointer", ""),
                "camera_e2e_loader_section": mapping.get("section", ""),
                "primary_loader_table": mapping.get("primary", ""),
                "secondary_loader_tables": mapping.get("secondary", ""),
            }
        )
    return result


def crosstalk_support_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "gate": "MISSING",
            "status": "NO_FINITE_ARRAY_SUPPORT_PILOT",
            "best_neighborhood": "",
            "best_truncation_fraction": "",
            "summary": "",
            "max_required_neighborhood": "",
            "min_truncation_fraction": "",
            "max_truncation_fraction": "",
            "threshold": "",
            "recommendation": "No finite-array support pilot is available for this sensor.",
            "representative_row": {},
        }

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("field_case", ""),
            safe_float(row.get("wavelength_nm"), 0.0) or 0.0,
            row.get("color_channel", ""),
        ),
    )
    trunc_rows = [
        (safe_float(row.get("best_pilot_truncation_fraction")), row)
        for row in sorted_rows
        if safe_float(row.get("best_pilot_truncation_fraction")) is not None
    ]
    neighborhoods = [safe_int(row.get("best_pilot_neighborhood")) for row in sorted_rows if safe_int(row.get("best_pilot_neighborhood")) > 0]
    trunc_values = [value for value, _row in trunc_rows if value is not None]
    representative_row = max(trunc_rows, key=lambda pair: pair[0])[1] if trunc_rows else sorted_rows[0]
    gate_values = {row.get("product_crosstalk_gate", "") for row in sorted_rows}
    if "FAIL" in gate_values:
        gate = "FAIL"
    elif "CHECK" in gate_values:
        gate = "CHECK"
    elif "PASS" in gate_values:
        gate = "PASS"
    else:
        gate = "MISSING"

    summary_parts = []
    for row in sorted_rows:
        color = row.get("color_channel", "?")
        wavelength = row.get("wavelength_nm", "?")
        field = row.get("field_case", "?")
        neighborhood = row.get("best_pilot_neighborhood", "?")
        truncation = row.get("best_pilot_truncation_fraction", "?")
        summary_parts.append(f"{color}@{wavelength}nm/{field}:n{neighborhood},trunc={truncation}")

    max_neighborhood = max(neighborhoods) if neighborhoods else ""
    min_truncation = min(trunc_values) if trunc_values else ""
    max_truncation = max(trunc_values) if trunc_values else ""
    recommendation = representative_row.get("support_recommendation", "")
    if max_neighborhood and trunc_values:
        recommendation = (
            f"low-res support pilots cover {len(sorted_rows)} row(s); max required support is "
            f"{max_neighborhood}x{max_neighborhood}; truncation range is {min_truncation}..{max_truncation}; "
            "confirm at product resolution before product crosstalk use"
        )
    status = "LOW_RES_SUPPORT_PILOT_ONLY_PRODUCT_BLOCKED" if gate == "FAIL" else "CROSSTALK_SUPPORT_REVIEW_REQUIRED"

    return {
        "gate": gate,
        "status": status,
        "best_neighborhood": representative_row.get("best_pilot_neighborhood", ""),
        "best_truncation_fraction": representative_row.get("best_pilot_truncation_fraction", ""),
        "summary": "; ".join(summary_parts),
        "max_required_neighborhood": max_neighborhood,
        "min_truncation_fraction": min_truncation,
        "max_truncation_fraction": max_truncation,
        "threshold": representative_row.get("truncation_threshold", ""),
        "recommendation": recommendation,
        "representative_row": representative_row,
    }


def json_schema(path: Path) -> str:
    return str(read_json(path).get("schema", "")) if path.exists() and path.suffix.lower() == ".json" else ""


def row_count(path: Path) -> int | str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".csv":
        return len(read_csv_rows(path))
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        for key in ("sensor_count", "coverage_row_count", "runtime_row_count", "field_row_count", "material_row_count", "electrical_row_count"):
            if key in payload:
                return payload.get(key, "")
        for key in ("pilot_row_count", "candidate_row_count", "sensor_row_count"):
            if key in payload:
                return payload.get(key, "")
    return ""


def artifact_row(artifact_id: str, role: str, path: Path, *, required: bool, notes: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "role": role,
        "path": repo_rel(path),
        "exists": path.exists(),
        "schema": json_schema(path),
        "row_count": row_count(path),
        "required_for_consumer": required,
        "notes": notes,
    }


def build_artifact_rows(package_dir: Path) -> list[dict[str, Any]]:
    return [
        artifact_row(
            "runtime_lut",
            "optical_field_response_and_cra_lut",
            package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv",
            required=True,
            notes="Main response/QE proxy, CRA, OCL shift, and optical crosstalk summary rows.",
        ),
        artifact_row(
            "runtime_crosstalk_kernel",
            "optical_crosstalk_kernel",
            package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv",
            required=True,
            notes="Kernel rows joined by runtime_id.",
        ),
        artifact_row(
            "spectral_response",
            "spectral_color_response",
            package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv",
            required=True,
            notes="CFA/clear spectral response rows for color or mono sensors.",
        ),
        artifact_row(
            "color_matrix_seed",
            "rgb_to_xyz_seed",
            package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv",
            required=True,
            notes="RGB seed rows; monochrome sensors are N/A.",
        ),
        artifact_row(
            "material_nk_lut",
            "fdtd_material_nk_lut",
            package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv",
            required=True,
            notes="CFA/OCL/passivation/Si n,k or proxy material rows.",
        ),
        artifact_row(
            "cfa_provenance_by_sensor",
            "cfa_color_material_provenance_by_sensor",
            package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv",
            required=True,
            notes="Per-sensor CFA source/fallback class and CameraE2E color-use gate.",
        ),
        artifact_row(
            "cfa_provenance",
            "cfa_color_material_provenance_audit",
            package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance.json",
            required=True,
            notes="Package-level CFA provenance audit; flags generic RGB fallback for unknown CFA sensors.",
        ),
        artifact_row(
            "cfa_db_by_sensor",
            "dedicated_cfa_db_by_sensor",
            package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_by_sensor.csv",
            required=True,
            notes="Sensor-level CFA DB pattern, thickness, source, proxy channel, and product blocker rows.",
        ),
        artifact_row(
            "cfa_db_transmission_lut",
            "dedicated_cfa_db_transmission_lut",
            package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_transmission_lut.csv",
            required=True,
            notes="Wavelength x CFA channel n,k and absorption-only transmission rows for CameraE2E lookup.",
        ),
        artifact_row(
            "cfa_db_tables",
            "dedicated_cfa_db_table_manifest",
            package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_tables.json",
            required=True,
            notes="Package-level CFA DB table validation and fallback counts.",
        ),
        artifact_row(
            "electrical_noise_lut",
            "electrical_noise_and_collection_lut",
            package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_noise_lut.csv",
            required=True,
            notes="Conversion, FWC, dark, DSNU/PRNU, temporal noise, and electrical crosstalk priors.",
        ),
        artifact_row(
            "readout_gain_lut",
            "readout_raw_gain_lut",
            package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_readout_gain_lut.csv",
            required=True,
            notes="Gain, black level, ADC, FPN, timing, and defect priors.",
        ),
        artifact_row(
            "binning_remosaic_lut",
            "binning_remosaic_lut",
            package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_binning_remosaic_lut.csv",
            required=True,
            notes="Mode-level signal/noise/crosstalk redefinition.",
        ),
        artifact_row(
            "module_coupling_lut",
            "module_field_cra_vignetting_pupil_lut",
            package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_field_lut.csv",
            required=True,
            notes="CRA, OCL shift, vignetting, assembly, and chromatic pupil priors.",
        ),
        artifact_row(
            "uncertainty_budget",
            "camera_e2e_uncertainty_budget",
            package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_budget.csv",
            required=True,
            notes="Per-domain engineering uncertainty bands for RI, CFA k/transmission, QE, CRA, crosstalk, noise, readout, and module priors.",
        ),
        artifact_row(
            "uncertainty_by_sensor",
            "camera_e2e_uncertainty_by_sensor",
            package_dir / "camera_e2e_uncertainty_budget" / "camera_e2e_uncertainty_by_sensor.csv",
            required=True,
            notes="Sensor-level uncertainty summary and product-use guard.",
        ),
        artifact_row(
            "response_trace",
            "pixel_response_calculation_trace",
            package_dir / "camera_e2e_response_trace" / "camera_e2e_response_trace.csv",
            required=True,
            notes="Runtime response rows joined with CFA/OCL/passivation/Si constants and simple CFA x Si sanity calculations.",
        ),
        artifact_row(
            "response_trace_summary",
            "pixel_response_trace_summary_by_sensor",
            package_dir / "camera_e2e_response_trace" / "camera_e2e_response_trace_summary.csv",
            required=True,
            notes="Per-sensor response trace coverage and evidence gate summary.",
        ),
        artifact_row(
            "response_example",
            "readable_cfa_to_si_to_qe_examples",
            package_dir / "camera_e2e_response_example" / "camera_e2e_response_example.csv",
            required=True,
            notes="One center-field representative R/G/B or clear example per sensor showing CFA transmission, simple Si absorption, runtime scale, QE proxy, and crosstalk split.",
        ),
        artifact_row(
            "response_example_summary",
            "readable_qe_example_summary_by_sensor",
            package_dir / "camera_e2e_response_example" / "camera_e2e_response_example_summary.csv",
            required=True,
            notes="Per-sensor summary for the readable CFA-to-Si-to-QE examples.",
        ),
        artifact_row(
            "method_provenance_matrix",
            "solver_external_proxy_method_matrix",
            package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_matrix.csv",
            required=True,
            notes="Per-sensor and per-requirement source-method classification: solver output, external/local DB, structural topology, or proxy/prior.",
        ),
        artifact_row(
            "method_provenance_by_sensor",
            "method_source_summary_by_sensor",
            package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_by_sensor.csv",
            required=True,
            notes="Sensor-level counts of solver/external/proxy/prior source classes and product-use guard.",
        ),
        artifact_row(
            "source_integrity_matrix",
            "requirement_source_method_uncertainty_join",
            package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_matrix.csv",
            required=True,
            notes="Joined coverage, method provenance, and uncertainty row for each CameraE2E requirement.",
        ),
        artifact_row(
            "source_integrity_by_sensor",
            "source_integrity_summary_by_sensor",
            package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_by_sensor.csv",
            required=True,
            notes="Sensor-level summary of source class counts, uncertainty gate, and product-use guard.",
        ),
        artifact_row(
            "quantitative_execution_plan",
            "quantitative_fdtd_execution_plan",
            package_dir / "camera_e2e_quantitative_execution_plan.csv",
            required=True,
            notes="Full field/crosstalk point counts and runtime estimates, including finite-array crosstalk domain factors.",
        ),
        artifact_row(
            "quantitative_point_queue",
            "point_sized_solver_queue",
            package_dir / "camera_e2e_quantitative_point_queue.csv",
            required=True,
            notes="Concrete Meep point commands keyed by slug, solver, color, field case, and wavelength.",
        ),
        artifact_row(
            "resource_limited_batch_plan",
            "resource_limited_crosstalk_batch_plan",
            package_dir / "camera_e2e_resource_limited_batch_plan.csv",
            required=True,
            notes="Finite-array crosstalk points skipped by the local runner and intended for batch/cluster execution.",
        ),
        artifact_row(
            "quantitative_coverage",
            "quantitative_solver_coverage",
            package_dir / "camera_e2e_quantitative_coverage.csv",
            required=True,
            notes="Per-sensor field and finite-array crosstalk PASS/CHECK/FAIL/resource-limited coverage.",
        ),
        artifact_row(
            "coverage_matrix",
            "requirement_coverage_matrix",
            package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv",
            required=True,
            notes="Per-sensor requirement gates and blockers.",
        ),
        artifact_row(
            "mesh_confidence_by_sensor",
            "mesh_resolution_confidence_by_sensor",
            package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv",
            required=True,
            notes="Per-sensor numerical confidence class from quantitative mesh/convergence coverage.",
        ),
        artifact_row(
            "mesh_confidence",
            "mesh_resolution_confidence_audit",
            package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence.json",
            required=True,
            notes="Package-level mesh confidence audit with field/crosstalk PASS fractions and use policy.",
        ),
        artifact_row(
            "crosstalk_support_audit",
            "finite_array_crosstalk_support_audit",
            package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_audit.json",
            required=True,
            notes="Support-size and truncation guard for finite-array crosstalk kernels.",
        ),
        artifact_row(
            "crosstalk_support_by_sensor",
            "finite_array_crosstalk_support_by_sensor",
            package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_by_sensor.csv",
            required=True,
            notes="Per-sensor support recommendation; join by slug/color/wavelength/field case.",
        ),
        artifact_row(
            "crosstalk_support_pilots",
            "finite_array_crosstalk_support_pilots",
            package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_pilots.csv",
            required=True,
            notes="Low-resolution finite-array support pilot rows with truncation/grid/convergence gates.",
        ),
        artifact_row(
            "crosstalk_product_candidates",
            "finite_array_crosstalk_product_candidate_commands",
            package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_product_candidates.csv",
            required=True,
            notes="Product-resolution finite-array candidate commands and feasibility estimates.",
        ),
        artifact_row(
            "capability_profile_by_sensor",
            "camera_e2e_capability_profile_by_sensor",
            package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv",
            required=True,
            notes="Per-sensor use-scope summary for optical/color/crosstalk/electrical/readout/module domains.",
        ),
        artifact_row(
            "capability_profile",
            "camera_e2e_capability_profile_audit",
            package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_profile.json",
            required=True,
            notes="Package-level capability/use-profile manifest with product-blocked policy.",
        ),
        artifact_row(
            "lut_trust_assessment",
            "camera_e2e_lut_trust_assessment",
            package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_assessment.json",
            required=True,
            notes="Package-level trust assessment separating research usability, evidence confidence, and product calibration.",
        ),
        artifact_row(
            "lut_trust_by_sensor",
            "camera_e2e_lut_trust_by_sensor",
            package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv",
            required=True,
            notes="Per-sensor trust class and score split for CameraE2E routing.",
        ),
        artifact_row(
            "lut_trust_by_domain",
            "camera_e2e_lut_trust_by_domain",
            package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv",
            required=True,
            notes="Per-domain trust class for Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling.",
        ),
        artifact_row(
            "lut_trust_by_requirement",
            "camera_e2e_lut_trust_by_requirement",
            package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_requirement.csv",
            required=True,
            notes="Per-requirement trust class aligned one-to-one with the coverage matrix.",
        ),
        artifact_row(
            "probe_summary",
            "scalar_probe_summary",
            package_dir / "camera_e2e_sensor_probe_all_sensors" / "camera_e2e_sensor_probe_summary.csv",
            required=True,
            notes="Consumer-path scalar smoke summary.",
        ),
        artifact_row(
            "readiness_report",
            "product_readiness_gate_report",
            package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json",
            required=True,
            notes="Strict product gate blocker report.",
        ),
    ]


def source_paths(package_dir: Path) -> dict[str, Path]:
    return {
        "runtime_lut": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv",
        "runtime_crosstalk_kernel": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv",
        "spectral_response": package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv",
        "color_matrix_seed": package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv",
        "material_nk_lut": package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv",
        "cfa_provenance_by_sensor": package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv",
        "cfa_provenance": package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance.json",
        "cfa_db_by_sensor": package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_by_sensor.csv",
        "cfa_db_transmission_lut": package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_transmission_lut.csv",
        "cfa_db_tables": package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_tables.json",
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
        "quantitative_execution_plan": package_dir / "camera_e2e_quantitative_execution_plan.csv",
        "quantitative_point_queue": package_dir / "camera_e2e_quantitative_point_queue.csv",
        "resource_limited_batch_plan": package_dir / "camera_e2e_resource_limited_batch_plan.csv",
        "quantitative_coverage": package_dir / "camera_e2e_quantitative_coverage.csv",
        "coverage_matrix": package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv",
        "mesh_confidence_by_sensor": package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv",
        "mesh_confidence": package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence.json",
        "crosstalk_support_audit": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_audit.json",
        "crosstalk_support_by_sensor": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_by_sensor.csv",
        "crosstalk_support_pilots": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_pilots.csv",
        "crosstalk_product_candidates": package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_product_candidates.csv",
        "capability_profile_by_sensor": package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv",
        "capability_profile": package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_profile.json",
        "lut_trust_assessment": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_assessment.json",
        "lut_trust_by_sensor": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv",
        "lut_trust_by_domain": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_domain.csv",
        "lut_trust_by_requirement": package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_requirement.csv",
        "probe_summary": package_dir / "camera_e2e_sensor_probe_all_sensors" / "camera_e2e_sensor_probe_summary.csv",
        "readiness_report": package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json",
        "sensor_model_summary": package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv",
    }


def build_sensor_manifest(
    *,
    sensor: dict[str, str],
    model_row: dict[str, str],
    coverage_rows: list[dict[str, str]],
    source_integrity_rows: list[dict[str, str]],
    probe_rows: list[dict[str, str]],
    mesh_confidence_row: dict[str, str],
    crosstalk_support_rows: list[dict[str, str]],
    crosstalk_product_candidate_rows: list[dict[str, str]],
    cfa_provenance_row: dict[str, str],
    cfa_db_row: dict[str, str],
    cfa_db_transmission_rows: list[dict[str, str]],
    capability_row: dict[str, str],
    trust_row: dict[str, str],
    trust_domain_rows: list[dict[str, str]],
    trust_requirement_rows: list[dict[str, str]],
    quantitative_plan_rows: list[dict[str, str]],
    resource_limited_rows: list[dict[str, str]],
    counts: dict[str, int],
    paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    product_ready = boolish(model_row.get("camera_e2e_product_ready") or sensor.get("product_ready"))
    coverage_research_counts = gate_counts(coverage_rows, "research_gate")
    coverage_product_counts = gate_counts(coverage_rows, "product_gate")
    objective_fulfillment_rows = build_objective_fulfillment_rows(coverage_rows, source_integrity_rows)
    primary_blockers = []
    for row in coverage_rows:
        if row.get("product_gate") in {"FAIL", "MISSING"} and row.get("primary_blocker"):
            blocker = row["primary_blocker"]
            if blocker not in primary_blockers:
                primary_blockers.append(blocker)
    support_summary = crosstalk_support_summary(crosstalk_support_rows)
    manifest = {
        "schema": "camera_e2e_consumer_sensor_manifest_v1",
        "artifact_role": "per_sensor_camera_e2e_consumer_load_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor": {
            "slug": sensor.get("slug", ""),
            "code": sensor.get("code", ""),
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "pixel_pitch_um": model_row.get("pixel_pitch_um", ""),
            "cfa_pattern": model_row.get("cfa_pattern", ""),
            "effective_ocl_mode": model_row.get("effective_ocl_mode", ""),
        },
        "gates": {
            "research_bundle_gate": "CHECK",
            "product_bundle_gate": "PASS" if product_ready else "FAIL",
            "product_ready": product_ready,
            "coverage_research_gate_counts": coverage_research_counts,
            "coverage_product_gate_counts": coverage_product_counts,
        },
        "mesh_confidence": {
            "mesh_confidence_class": mesh_confidence_row.get("mesh_confidence_class", ""),
            "camera_e2e_recommended_use": mesh_confidence_row.get("camera_e2e_recommended_use", ""),
            "field_pass_points": mesh_confidence_row.get("field_pass_points", ""),
            "field_required_points": mesh_confidence_row.get("field_required_points", ""),
            "field_grid_pass_fraction": mesh_confidence_row.get("field_grid_pass_fraction", ""),
            "field_min_resolution_ratio": mesh_confidence_row.get("field_min_resolution_ratio", ""),
            "field_signed_flux_nonpositive_points": mesh_confidence_row.get("field_signed_flux_nonpositive_points", ""),
            "crosstalk_pass_points": mesh_confidence_row.get("crosstalk_pass_points", ""),
            "crosstalk_required_points": mesh_confidence_row.get("crosstalk_required_points", ""),
            "crosstalk_resource_limited_points": mesh_confidence_row.get("crosstalk_resource_limited_points", ""),
            "primary_limitations": mesh_confidence_row.get("primary_limitations", ""),
            "next_action": mesh_confidence_row.get("next_action", ""),
        },
        "crosstalk_support": {
            "support_rows": crosstalk_support_rows,
            "product_candidate_rows": crosstalk_product_candidate_rows,
            "aggregate": {
                "product_crosstalk_gate": support_summary["gate"],
                "support_status": support_summary["status"],
                "best_neighborhood": support_summary["best_neighborhood"],
                "best_truncation_fraction": support_summary["best_truncation_fraction"],
                "summary": support_summary["summary"],
                "max_required_neighborhood": support_summary["max_required_neighborhood"],
                "min_truncation_fraction": support_summary["min_truncation_fraction"],
                "max_truncation_fraction": support_summary["max_truncation_fraction"],
                "truncation_threshold": support_summary["threshold"],
                "support_recommendation": support_summary["recommendation"],
            },
            "best_available": support_summary["representative_row"],
            "product_crosstalk_gate": support_summary["gate"],
            "support_status": support_summary["status"],
            "support_recommendation": support_summary["recommendation"],
            "policy": (
                "This is a finite-array support-size guard. It does not replace product-resolution crosstalk mesh/convergence evidence."
            ),
        },
        "cfa_provenance": {
            "cfa_provenance_class": cfa_provenance_row.get("cfa_provenance_class", ""),
            "cfa_assumption_gate": cfa_provenance_row.get("cfa_assumption_gate", ""),
            "optical_cfa_pattern": cfa_provenance_row.get("optical_cfa_pattern", ""),
            "optical_cfa_pattern_source_kind": cfa_provenance_row.get("optical_cfa_pattern_source_kind", ""),
            "optical_cfa_thickness_um": cfa_provenance_row.get("optical_cfa_thickness_um", ""),
            "optical_cfa_thickness_source_kind": cfa_provenance_row.get("optical_cfa_thickness_source_kind", ""),
            "generic_rgb_fallback_detected": cfa_provenance_row.get("generic_rgb_fallback_detected", ""),
            "camera_e2e_recommended_use": cfa_provenance_row.get("camera_e2e_recommended_use", ""),
            "primary_blocker": cfa_provenance_row.get("primary_blocker", ""),
            "next_action": cfa_provenance_row.get("next_action", ""),
        },
        "cfa_db": {
            "by_sensor_row": cfa_db_row,
            "transmission_row_count": len(cfa_db_transmission_rows),
            "color_channels": sorted(unique_values(cfa_db_transmission_rows, "color_channel")),
            "wavelengths_nm": sorted(unique_values(cfa_db_transmission_rows, "wavelength_nm"), key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value),
            "source_tables": {
                "by_sensor": repo_rel(paths.get("cfa_db_by_sensor")),
                "transmission_lut": repo_rel(paths.get("cfa_db_transmission_lut")),
            },
            "join_keys": {
                "by_sensor": "slug",
                "transmission_lut": "slug + color_channel + wavelength_nm",
            },
            "policy": (
                "CFA DB rows are direct CameraE2E lookup tables for filter pattern, thickness, and proxy n,k/transmission. "
                "Rows remain product-blocked unless measured CFA material and spectral response calibration are imported."
            ),
        },
        "capability_profile": {
            "overall_use_scope": capability_row.get("overall_use_scope", ""),
            "spectral_qe_scope": capability_row.get("spectral_qe_scope", ""),
            "color_response_scope": capability_row.get("color_response_scope", ""),
            "optical_crosstalk_scope": capability_row.get("optical_crosstalk_scope", ""),
            "cra_response_scope": capability_row.get("cra_response_scope", ""),
            "electrical_noise_scope": capability_row.get("electrical_noise_scope", ""),
            "readout_raw_scope": capability_row.get("readout_raw_scope", ""),
            "module_coupling_scope": capability_row.get("module_coupling_scope", ""),
            "key_blockers": capability_row.get("key_blockers", ""),
            "next_actions": capability_row.get("next_actions", ""),
        },
        "lut_trust": {
            "trust_class": trust_row.get("trust_class", ""),
            "camera_e2e_allowed_use": trust_row.get("camera_e2e_allowed_use", ""),
            "research_usability_score_0_100": trust_row.get("research_usability_score_0_100", ""),
            "evidence_confidence_score_0_100": trust_row.get("evidence_confidence_score_0_100", ""),
            "product_calibration_score_0_100": trust_row.get("product_calibration_score_0_100", ""),
            "research_utility_grade_0_10": trust_row.get("research_utility_grade_0_10", ""),
            "solver_evidence_grade_0_10": trust_row.get("solver_evidence_grade_0_10", ""),
            "product_accuracy_grade_0_10": trust_row.get("product_accuracy_grade_0_10", ""),
            "field_mesh_pass_fraction": trust_row.get("field_mesh_pass_fraction", ""),
            "crosstalk_mesh_pass_fraction": trust_row.get("crosstalk_mesh_pass_fraction", ""),
            "crosstalk_support_pilot_row_count": trust_row.get("crosstalk_support_pilot_row_count", ""),
            "crosstalk_support_status": trust_row.get("crosstalk_support_status", ""),
            "crosstalk_support_recommended_kernel": trust_row.get("crosstalk_support_recommended_kernel", ""),
            "crosstalk_support_best_truncation_fraction": trust_row.get("crosstalk_support_best_truncation_fraction", ""),
            "crosstalk_support_worst_truncation_fraction": trust_row.get("crosstalk_support_worst_truncation_fraction", ""),
            "recommended_next_action": trust_row.get("recommended_next_action", ""),
            "domain_rows": trust_domain_rows,
            "requirement_rows": trust_requirement_rows,
            "policy": (
                "Trust scores are CameraE2E routing guards, not physical accuracy percentages. "
                "Product use is blocked unless product_ready is true and row-level product gates pass."
            ),
        },
        "quantitative_execution": {
            "execution_plan_rows": quantitative_plan_rows,
            "resource_limited_batch_rows": resource_limited_rows,
            "point_queue_source": repo_rel(paths.get("quantitative_point_queue")),
            "coverage_source": repo_rel(paths.get("quantitative_coverage")),
            "policy": (
                "Field and crosstalk queues are solver evidence requirements, not product data by themselves. "
                "Resource-limited rows require batch/cluster execution and merge before they can raise mesh confidence."
            ),
        },
        "row_counts": counts,
        "source_tables": {key: repo_rel(path) for key, path in paths.items() if key != "sensor_model_summary"},
        "join_keys": {
            "sensor": "slug",
            "runtime_to_kernel": "runtime_id",
            "runtime_to_color": "slug + color_channel + wavelength_nm",
            "runtime_to_material": "slug + material_family/material_key + color_channel + wavelength_nm",
            "cfa_provenance": "slug",
            "cfa_db_by_sensor": "slug",
            "cfa_db_transmission_lut": "slug + color_channel + wavelength_nm",
            "runtime_to_electrical": "slug + temperature_c + exposure_s + signal_fraction",
            "runtime_to_readout": "slug + analog_gain_x + digital_gain_x + adc_bit_depth",
            "runtime_to_binning": "slug + mode_id",
            "module_field": "slug + field_case + wavelength_nm",
            "crosstalk_support": "slug + color_channel + wavelength_nm + field_case",
            "crosstalk_product_candidates": "slug + mode + neighborhood + resolution_px_per_um",
            "quantitative_queue": "slug + solver + color + field_case + wavelength_nm",
            "resource_limited_batch": "queue_id",
            "coverage": "slug + requirement_id",
            "source_integrity": "slug + requirement_id",
            "objective_fulfillment": "slug + requirement_id",
            "capability_profile": "slug",
            "lut_trust": "slug",
            "lut_trust_domain": "slug + domain",
            "lut_trust_requirement": "slug + requirement_id",
        },
        "source_integrity": {
            "source_integrity_matrix_rows": source_integrity_rows,
            "gate_counts": gate_counts(source_integrity_rows, "source_integrity_gate"),
            "uncertainty_product_gate_counts": gate_counts(source_integrity_rows, "uncertainty_product_gate"),
            "policy": "Use before consuming numeric LUT rows to separate solver, external/local DB, and proxy/prior values.",
        },
        "objective_fulfillment": {
            "requirement_rows": objective_fulfillment_rows,
            "row_count": len(objective_fulfillment_rows),
            "join_key": "slug + requirement_id",
            "policy": (
                "Requirement-level load map for the CameraE2E objective. "
                "Consumers should use flat_json_pointer or primary_loader_table, and preserve row-level gates and uncertainty."
            ),
        },
        "coverage": coverage_rows,
        "probe_summary_rows": probe_rows,
        "primary_blockers": primary_blockers,
        "policy": {
            "research_use": "Allowed only if row-level gates and uncertainty columns are propagated.",
            "product_use": "Blocked unless product_ready is true and product_bundle_gate is PASS.",
        },
    }
    write_json(output_path, manifest)
    return manifest


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensor_dir = output_dir / "sensors"
    paths = source_paths(package_dir)
    package = read_json(package_dir / "camera_e2e_lut_package.json")
    sensor_rows = read_csv_rows(package_dir / "camera_e2e_handoff_manifest" / "camera_e2e_handoff_sensors.csv")
    if not sensor_rows:
        sensor_rows = read_csv_rows(paths["sensor_model_summary"])

    coverage_all = read_csv_rows(paths["coverage_matrix"])
    model_by_slug = index_by(read_csv_rows(paths["sensor_model_summary"]), "slug")
    runtime_by_slug = group_rows(read_csv_rows(paths["runtime_lut"]), "slug")
    kernel_by_slug = group_rows(read_csv_rows(paths["runtime_crosstalk_kernel"]), "slug")
    spectral_by_slug = group_rows(read_csv_rows(paths["spectral_response"]), "slug")
    material_by_slug = group_rows(read_csv_rows(paths["material_nk_lut"]), "slug")
    electrical_by_slug = group_rows(read_csv_rows(paths["electrical_noise_lut"]), "slug")
    readout_by_slug = group_rows(read_csv_rows(paths["readout_gain_lut"]), "slug")
    binning_by_slug = group_rows(read_csv_rows(paths["binning_remosaic_lut"]), "slug")
    module_by_slug = group_rows(read_csv_rows(paths["module_coupling_lut"]), "slug")
    quantitative_plan_by_slug = group_rows(read_csv_rows(paths["quantitative_execution_plan"]), "slug")
    quantitative_queue_by_slug = group_rows(read_csv_rows(paths["quantitative_point_queue"]), "slug")
    resource_limited_by_slug = group_rows(read_csv_rows(paths["resource_limited_batch_plan"]), "slug")
    coverage_by_slug = group_rows(coverage_all, "slug")
    source_integrity_by_slug = group_rows(read_csv_rows(paths["source_integrity_matrix"]), "slug")
    probe_by_slug = group_rows(read_csv_rows(paths["probe_summary"]), "slug")
    color_matrix_by_slug = index_by(read_csv_rows(paths["color_matrix_seed"]), "slug")
    mesh_confidence_by_slug = index_by(read_csv_rows(paths["mesh_confidence_by_sensor"]), "slug")
    crosstalk_support_by_slug = group_rows(read_csv_rows(paths["crosstalk_support_by_sensor"]), "slug")
    crosstalk_product_candidates_by_slug = group_rows(read_csv_rows(paths["crosstalk_product_candidates"]), "slug")
    cfa_provenance_by_slug = index_by(read_csv_rows(paths["cfa_provenance_by_sensor"]), "slug")
    cfa_db_by_slug = index_by(read_csv_rows(paths["cfa_db_by_sensor"]), "slug")
    cfa_db_transmission_by_slug = group_rows(read_csv_rows(paths["cfa_db_transmission_lut"]), "slug")
    capability_by_slug = index_by(read_csv_rows(paths["capability_profile_by_sensor"]), "slug")
    trust_by_slug = index_by(read_csv_rows(paths["lut_trust_by_sensor"]), "slug")
    trust_domain_by_slug = group_rows(read_csv_rows(paths["lut_trust_by_domain"]), "slug")
    trust_requirement_by_slug = group_rows(read_csv_rows(paths["lut_trust_by_requirement"]), "slug")

    sensor_index_rows: list[dict[str, Any]] = []
    sensor_manifest_paths: list[str] = []
    sensor_manifests: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        model = model_by_slug.get(slug, {})
        coverage_rows = coverage_by_slug.get(slug, [])
        mesh_confidence = mesh_confidence_by_slug.get(slug, {})
        crosstalk_support_rows = crosstalk_support_by_slug.get(slug, [])
        crosstalk_product_candidate_rows = crosstalk_product_candidates_by_slug.get(slug, [])
        crosstalk_summary = crosstalk_support_summary(crosstalk_support_rows)
        cfa_provenance = cfa_provenance_by_slug.get(slug, {})
        cfa_db = cfa_db_by_slug.get(slug, {})
        cfa_db_transmission_rows = cfa_db_transmission_by_slug.get(slug, [])
        capability = capability_by_slug.get(slug, {})
        trust = trust_by_slug.get(slug, {})
        trust_domain_rows = trust_domain_by_slug.get(slug, [])
        trust_requirement_rows = trust_requirement_by_slug.get(slug, [])
        counts = {
            "runtime": len(runtime_by_slug.get(slug, [])),
            "kernel": len(kernel_by_slug.get(slug, [])),
            "spectral": len(spectral_by_slug.get(slug, [])),
            "material": len(material_by_slug.get(slug, [])),
            "electrical": len(electrical_by_slug.get(slug, [])),
            "readout": len(readout_by_slug.get(slug, [])),
            "binning": len(binning_by_slug.get(slug, [])),
            "module_field": len(module_by_slug.get(slug, [])),
            "quantitative_plan": len(quantitative_plan_by_slug.get(slug, [])),
            "quantitative_queue": len(quantitative_queue_by_slug.get(slug, [])),
            "resource_limited_batch": len(resource_limited_by_slug.get(slug, [])),
            "coverage": len(coverage_rows),
            "source_integrity": len(source_integrity_by_slug.get(slug, [])),
            "objective_fulfillment": len(build_objective_fulfillment_rows(coverage_rows, source_integrity_by_slug.get(slug, []))),
            "mesh_confidence": 1 if mesh_confidence else 0,
            "crosstalk_support": len(crosstalk_support_rows),
            "crosstalk_product_candidates": len(crosstalk_product_candidate_rows),
            "cfa_provenance": 1 if cfa_provenance else 0,
            "cfa_db": 1 if cfa_db else 0,
            "cfa_db_transmission": len(cfa_db_transmission_rows),
            "capability": 1 if capability else 0,
            "lut_trust": 1 if trust else 0,
            "lut_trust_domain": len(trust_domain_rows),
            "lut_trust_requirement": len(trust_requirement_rows),
            "probe_summary": len(probe_by_slug.get(slug, [])),
        }
        sensor_json = sensor_dir / f"{slug}.json"
        sensor_manifest = build_sensor_manifest(
            sensor=sensor,
            model_row=model,
            coverage_rows=coverage_rows,
            source_integrity_rows=source_integrity_by_slug.get(slug, []),
            probe_rows=probe_by_slug.get(slug, []),
            mesh_confidence_row=mesh_confidence,
            crosstalk_support_rows=crosstalk_support_rows,
            crosstalk_product_candidate_rows=crosstalk_product_candidate_rows,
            cfa_provenance_row=cfa_provenance,
            cfa_db_row=cfa_db,
            cfa_db_transmission_rows=cfa_db_transmission_rows,
            capability_row=capability,
            trust_row=trust,
            trust_domain_rows=trust_domain_rows,
            trust_requirement_rows=trust_requirement_rows,
            quantitative_plan_rows=quantitative_plan_by_slug.get(slug, []),
            resource_limited_rows=resource_limited_by_slug.get(slug, []),
            counts=counts,
            paths=paths,
            output_path=sensor_json,
        )
        sensor_manifests.append(sensor_manifest)
        sensor_manifest_paths.append(repo_rel(sensor_json))
        primary_blockers = sensor_manifest["primary_blockers"]
        coverage_research_counts = sensor_manifest["gates"]["coverage_research_gate_counts"]
        coverage_product_counts = sensor_manifest["gates"]["coverage_product_gate_counts"]
        source_integrity_counts = gate_counts(source_integrity_by_slug.get(slug, []), "source_integrity_gate")
        sensor_index_rows.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "research_bundle_gate": "CHECK",
                "product_bundle_gate": "PASS" if sensor_manifest["gates"]["product_ready"] else "FAIL",
                "product_ready": sensor_manifest["gates"]["product_ready"],
                "coverage_requirement_count": counts["coverage"],
                "objective_fulfillment_row_count": counts["objective_fulfillment"],
                "coverage_research_gate_counts": json.dumps(coverage_research_counts, sort_keys=True),
                "coverage_product_gate_counts": json.dumps(coverage_product_counts, sort_keys=True),
                "source_integrity_gate_counts": json.dumps(source_integrity_counts, sort_keys=True),
                "mesh_confidence_class": mesh_confidence.get("mesh_confidence_class", ""),
                "camera_e2e_recommended_use": mesh_confidence.get("camera_e2e_recommended_use", ""),
                "mesh_field_pass_points": mesh_confidence.get("field_pass_points", ""),
                "mesh_field_required_points": mesh_confidence.get("field_required_points", ""),
                "mesh_crosstalk_pass_points": mesh_confidence.get("crosstalk_pass_points", ""),
                "mesh_crosstalk_required_points": mesh_confidence.get("crosstalk_required_points", ""),
                "mesh_confidence_row_count": counts["mesh_confidence"],
                "crosstalk_support_gate": crosstalk_summary["gate"],
                "crosstalk_support_best_neighborhood": crosstalk_summary["best_neighborhood"],
                "crosstalk_support_best_truncation_fraction": crosstalk_summary["best_truncation_fraction"],
                "crosstalk_support_summary": crosstalk_summary["summary"],
                "crosstalk_support_max_required_neighborhood": crosstalk_summary["max_required_neighborhood"],
                "crosstalk_support_min_truncation_fraction": crosstalk_summary["min_truncation_fraction"],
                "crosstalk_support_max_truncation_fraction": crosstalk_summary["max_truncation_fraction"],
                "crosstalk_support_threshold": crosstalk_summary["threshold"],
                "crosstalk_support_recommendation": crosstalk_summary["recommendation"],
                "crosstalk_support_row_count": counts["crosstalk_support"],
                "crosstalk_product_candidate_row_count": counts["crosstalk_product_candidates"],
                "cfa_provenance_class": cfa_provenance.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa_provenance.get("cfa_assumption_gate", ""),
                "cfa_pattern_source_kind": cfa_provenance.get("optical_cfa_pattern_source_kind", ""),
                "cfa_provenance_row_count": counts["cfa_provenance"],
                "cfa_db_row_count": counts["cfa_db"],
                "cfa_db_transmission_row_count": counts["cfa_db_transmission"],
                "capability_overall_use_scope": capability.get("overall_use_scope", ""),
                "capability_spectral_qe_scope": capability.get("spectral_qe_scope", ""),
                "capability_color_response_scope": capability.get("color_response_scope", ""),
                "capability_crosstalk_scope": capability.get("optical_crosstalk_scope", ""),
                "capability_row_count": counts["capability"],
                "lut_trust_class": trust.get("trust_class", ""),
                "lut_trust_research_score_0_100": trust.get("research_usability_score_0_100", ""),
                "lut_trust_evidence_score_0_100": trust.get("evidence_confidence_score_0_100", ""),
                "lut_trust_product_score_0_100": trust.get("product_calibration_score_0_100", ""),
                "lut_trust_research_grade_0_10": trust.get("research_utility_grade_0_10", ""),
                "lut_trust_solver_evidence_grade_0_10": trust.get("solver_evidence_grade_0_10", ""),
                "lut_trust_product_accuracy_grade_0_10": trust.get("product_accuracy_grade_0_10", ""),
                "lut_trust_crosstalk_support_status": trust.get("crosstalk_support_status", crosstalk_summary["status"]),
                "lut_trust_crosstalk_support_recommended_kernel": trust.get("crosstalk_support_recommended_kernel", ""),
                "lut_trust_row_count": counts["lut_trust"],
                "quantitative_queue_row_count": counts["quantitative_queue"],
                "resource_limited_batch_row_count": counts["resource_limited_batch"],
                "runtime_row_count": counts["runtime"],
                "kernel_row_count": counts["kernel"],
                "spectral_row_count": counts["spectral"],
                "color_matrix_gate": color_matrix_by_slug.get(slug, {}).get("gate", ""),
                "material_row_count": counts["material"],
                "electrical_row_count": counts["electrical"],
                "readout_row_count": counts["readout"],
                "binning_row_count": counts["binning"],
                "module_field_row_count": counts["module_field"],
                "probe_summary_row_count": counts["probe_summary"],
                "load_manifest_json": repo_rel(sensor_json),
                "primary_blockers": "; ".join(primary_blockers[:8]),
            }
        )

    artifact_rows = build_artifact_rows(package_dir)
    requirement_load_map = build_requirement_load_map(coverage_all)
    validation = validate_bundle(
        package=package,
        artifact_rows=artifact_rows,
        sensor_index_rows=sensor_index_rows,
        sensor_manifest_paths=sensor_manifest_paths,
    )

    index_csv = output_dir / "camera_e2e_consumer_sensor_index.csv"
    artifacts_csv = output_dir / "camera_e2e_consumer_artifacts.csv"
    bundle_json = output_dir / "camera_e2e_consumer_bundle.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_consumer_bundle_v1",
        "artifact_role": "camera_e2e_consumer_load_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_index_rows),
        "artifact_count": len(artifact_rows),
        "requirement_count": len(requirement_load_map),
        "product_ready_count": sum(1 for row in sensor_index_rows if boolish(row.get("product_ready"))),
        "validation": validation,
        "sensor_manifest_json_files": sensor_manifest_paths,
        "source_tables": {key: repo_rel(path) for key, path in paths.items()},
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
        "requirement_load_map": requirement_load_map,
        "consumer_contract": {
            "entrypoint": repo_rel(bundle_json),
            "preferred_entrypoint": "camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_bundle.json for self-contained per-sensor loading; this consumer bundle for table-oriented loading.",
            "load_order": [
                "camera_e2e_consumer_sensor_index.csv",
                "per-sensor JSON manifest",
                "objective_fulfillment.requirement_rows",
                "source_integrity.source_integrity_matrix_rows",
                "runtime_lut",
                "runtime_crosstalk_kernel",
                "crosstalk_support_by_sensor",
                "crosstalk_product_candidates",
                "material/color/electrical/readout/module tables as needed",
                "mesh_confidence_by_sensor",
                "quantitative_execution_plan",
                "resource_limited_batch_plan",
                "cfa_provenance_by_sensor",
                "cfa_db_by_sensor",
                "cfa_db_transmission_lut",
                "capability_profile_by_sensor",
                "lut_trust_by_sensor",
                "lut_trust_by_domain",
                "lut_trust_by_requirement",
            ],
            "policy": "Research bundle may load when validation.pass is true. Product ingest must require product_ready true per sensor.",
        },
        "outputs": {
            "json": repo_rel(bundle_json),
            "sensor_index_csv": repo_rel(index_csv),
            "artifacts_csv": repo_rel(artifacts_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(index_csv, sensor_index_rows, INDEX_COLUMNS)
    write_csv(artifacts_csv, artifact_rows, ARTIFACT_COLUMNS)
    write_json(bundle_json, payload)
    write_html(html_path, payload, sensor_index_rows, artifact_rows)
    update_package(package_dir, payload)
    return payload


def validate_bundle(
    *,
    package: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
    sensor_index_rows: list[dict[str, Any]],
    sensor_manifest_paths: list[str],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected_sensor_count = int(package.get("sensor_count", 0) or 0)
    if expected_sensor_count and len(sensor_index_rows) != expected_sensor_count:
        issues.append({"severity": "error", "code": "sensor_count_mismatch", "expected": expected_sensor_count, "actual": len(sensor_index_rows)})
    for row in artifact_rows:
        if boolish(row.get("required_for_consumer")) and not boolish(row.get("exists")):
            issues.append({"severity": "error", "code": "required_artifact_missing", "artifact_id": row.get("artifact_id"), "path": row.get("path")})
    for path in sensor_manifest_paths:
        payload = read_json(abs_from_repo(path))
        if payload.get("schema") != "camera_e2e_consumer_sensor_manifest_v1":
            issues.append({"severity": "error", "code": "sensor_manifest_schema_invalid", "path": path})
    for row in sensor_index_rows:
        slug = row.get("slug", "")
        if safe_int(row.get("coverage_requirement_count")) <= 0:
            issues.append({"severity": "error", "code": "coverage_missing", "slug": slug})
        coverage_count = safe_int(row.get("coverage_requirement_count"))
        objective_count = safe_int(row.get("objective_fulfillment_row_count"))
        if objective_count <= 0:
            issues.append({"severity": "error", "code": "objective_fulfillment_missing", "slug": slug})
        if coverage_count != objective_count:
            issues.append(
                {
                    "severity": "error",
                    "code": "objective_fulfillment_count_mismatch",
                    "slug": slug,
                    "coverage_requirement_count": coverage_count,
                    "objective_fulfillment_row_count": objective_count,
                }
            )
        for key in (
            "runtime_row_count",
            "kernel_row_count",
            "spectral_row_count",
            "material_row_count",
            "electrical_row_count",
            "readout_row_count",
            "binning_row_count",
            "module_field_row_count",
            "mesh_confidence_row_count",
            "cfa_provenance_row_count",
            "cfa_db_row_count",
            "cfa_db_transmission_row_count",
            "capability_row_count",
            "lut_trust_row_count",
            "objective_fulfillment_row_count",
            "probe_summary_row_count",
        ):
            if safe_int(row.get(key)) <= 0:
                issues.append({"severity": "error", "code": "sensor_required_rows_missing", "slug": slug, "field": key})
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    product_ready_count = sum(1 for row in sensor_index_rows if boolish(row.get("product_ready")))
    if error_count:
        status = "FAIL"
    elif product_ready_count == len(sensor_index_rows) and sensor_index_rows:
        status = "PRODUCT_CONSUMER_BUNDLE_READY"
    else:
        status = "RESEARCH_CONSUMER_BUNDLE_READY_PRODUCT_BLOCKED"
    return {
        "schema": "camera_e2e_consumer_bundle_validation_v1",
        "pass": error_count == 0,
        "status": status,
        "issue_count": len(issues),
        "error_count": error_count,
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


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> None:
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
    issue_html = html_table(validation.get("issues", []), ["severity", "code", "slug", "artifact_id", "path"]) if validation.get("issues") else '<p class="pass">No consumer bundle load errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Consumer Bundle</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Consumer Bundle</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This is the recommended research-load contract for CameraE2E consumers; product ingest remains blocked.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">bundle status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("artifact_count", 0))}</div><div class="muted">source artifacts</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_count", 0))}</div><div class="muted">CameraE2E requirements</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Consumer Contract</h2>
<ul>
<li>Use <code>camera_e2e_consumer_sensor_index.csv</code> to discover per-sensor manifests.</li>
<li>Use each per-sensor JSON's <code>objective_fulfillment.requirement_rows</code> to map every CameraE2E requirement to a loader section, table, and pointer.</li>
<li>Use each per-sensor JSON's <code>source_tables</code> and <code>join_keys</code> for table loading.</li>
<li>Preserve row-level gates; product use requires per-sensor <code>product_ready=true</code>.</li>
</ul>
<h2>Issues</h2>{issue_html}
<h2>Sensor Index</h2>{html_table(sensor_rows, INDEX_COLUMNS)}
<h2>Source Artifacts</h2>{html_table(artifact_rows, ARTIFACT_COLUMNS)}
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
    outputs["camera_e2e_consumer_bundle_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_consumer_sensor_index_csv"] = payload["outputs"]["sensor_index_csv"]
    outputs["camera_e2e_consumer_artifacts_csv"] = payload["outputs"]["artifacts_csv"]
    outputs["camera_e2e_consumer_bundle_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_consumer_bundle"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "artifact_count": payload["artifact_count"],
        "requirement_count": payload.get("requirement_count", 0),
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
    payload = build_bundle(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "artifact_count": payload["artifact_count"],
                "requirement_count": payload["requirement_count"],
                "product_ready_count": payload["product_ready_count"],
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
