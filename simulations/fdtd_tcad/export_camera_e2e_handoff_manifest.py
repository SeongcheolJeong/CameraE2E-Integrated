#!/usr/bin/env python3
"""Export a CameraE2E handoff manifest.

This is the integration-facing index for the generated sensor package. It does
not create new simulation values; it verifies and ties together the files that a
CameraE2E consumer should load:

- per-sensor model manifests;
- runtime response and crosstalk LUTs;
- color spectral response and RGB-to-XYZ seed;
- response calculation examples and method/source provenance;
- electrical/readout prior seeds;
- module-coupling field LUT;
- per-sensor requirement coverage matrix;
- mesh/convergence confidence audit;
- consumer-facing per-sensor load manifests;
- readiness and probe reports.

Product use remains blocked until the underlying product gates pass.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_handoff_manifest"

ARTIFACT_COLUMNS = [
    "artifact_id",
    "role",
    "path",
    "exists",
    "schema",
    "row_count",
    "sensor_scope",
    "gate",
    "product_usable",
    "loader_hint",
    "notes",
]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "model_json",
    "flat_sensor_json",
    "runtime_row_count",
    "kernel_row_count",
    "spectral_row_count",
    "material_row_count",
    "response_example_row_count",
    "method_provenance_row_count",
    "source_integrity_row_count",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "cfa_db_row_count",
    "cfa_db_transmission_row_count",
    "capability_overall_use_scope",
    "capability_spectral_qe_scope",
    "capability_color_response_scope",
    "capability_crosstalk_scope",
    "mesh_confidence_class",
    "mesh_field_pass_points",
    "mesh_field_required_points",
    "mesh_crosstalk_pass_points",
    "mesh_crosstalk_required_points",
    "mesh_crosstalk_resource_limited_points",
    "quantitative_queue_row_count",
    "resource_limited_batch_row_count",
    "color_matrix_gate",
    "electrical_row_count",
    "readout_row_count",
    "binning_row_count",
    "module_field_row_count",
    "prior_gate",
    "cra_mismatch_gate",
    "research_ingest_gate",
    "production_lut_gate",
    "product_ready",
    "primary_blockers",
    "loader_order",
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


def abs_from_repo(path: str | Path) -> Path:
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
        rows = list(csv.DictReader(handle))
    return [row for row in rows if next(iter(row.values()), "") != next(iter(row.keys()), "")]


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


def csv_count(path: Path) -> int:
    return len(read_csv_rows(path))


def json_schema(path: Path) -> str:
    return str(read_json(path).get("schema", ""))


def row_count_for(path: Path) -> int | str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".csv":
        return csv_count(path)
    if path.suffix.lower() != ".json":
        return ""
    payload = read_json(path)
    for key in (
        "row_count",
        "runtime_row_count",
        "kernel_row_count",
        "sensor_count",
        "field_row_count",
        "plan_row_count",
        "job_count",
        "priority_row_count",
        "trace_row_count",
        "example_row_count",
        "matrix_row_count",
        "domain_row_count",
    ):
        if key in payload:
            return payload.get(key, "")
    return ""


def group_count(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = row.get(key, "")
        if value:
            counts[value] += 1
    return dict(counts)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def artifact_row(
    package_dir: Path,
    artifact_id: str,
    role: str,
    rel_path: str,
    *,
    sensor_scope: str,
    gate: str,
    product_usable: bool,
    loader_hint: str,
    notes: str,
) -> dict[str, Any]:
    path = package_dir / rel_path
    return {
        "artifact_id": artifact_id,
        "role": role,
        "path": repo_rel(path),
        "exists": path.exists(),
        "schema": json_schema(path) if path.suffix.lower() == ".json" else "",
        "row_count": row_count_for(path),
        "sensor_scope": sensor_scope,
        "gate": gate,
        "product_usable": product_usable,
        "loader_hint": loader_hint,
        "notes": notes,
    }


def build_artifact_rows(package_dir: Path, package: dict[str, Any], pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    product_ready = bool(package.get("camera_e2e_ready_count")) and package.get("camera_e2e_ready_count") == package.get("sensor_count")
    gate = "PASS" if product_ready else "CHECK"
    return [
        artifact_row(
            package_dir,
            "handoff_sensor_models",
            "per_sensor_handoff_manifest",
            "camera_e2e_sensor_models/camera_e2e_sensor_models.json",
            sensor_scope="all_sensors",
            gate=gate,
            product_usable=product_ready,
            loader_hint="Load first to discover per-sensor JSON manifests and coverage matrix.",
            notes="Primary CameraE2E handoff index.",
        ),
        artifact_row(
            package_dir,
            "coverage_matrix",
            "camera_e2e_requirement_coverage_matrix",
            "camera_e2e_coverage_matrix/camera_e2e_coverage_matrix.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug and requirement_id to decide which CameraE2E inputs are research-loadable or product-blocked.",
            notes="Maps every requested CameraE2E requirement to concrete source artifacts, row counts, gates, and blockers.",
        ),
        artifact_row(
            package_dir,
            "consumer_bundle",
            "camera_e2e_consumer_load_contract",
            "camera_e2e_consumer_bundle/camera_e2e_consumer_bundle.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Recommended CameraE2E entrypoint after handoff manifest; load per-sensor JSON manifests from this bundle.",
            notes="Bundles source table paths, join keys, row counts, coverage gates, probe summaries, and product blockers per sensor.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_bundle",
            "self_contained_per_sensor_camera_e2e_load_bundle",
            "camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_bundle.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Recommended when CameraE2E wants one JSON per sensor with embedded domain rows instead of following many CSV references.",
            notes="Flat sensor models embed Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling rows while preserving product blockers.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_index",
            "self_contained_per_sensor_camera_e2e_load_index",
            "camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_index.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug and open flat_sensor_json for the self-contained per-sensor load model.",
            notes="One row per sensor with row counts and product-blocked use scope.",
        ),
        artifact_row(
            package_dir,
            "import_contract",
            "camera_e2e_downstream_import_contract",
            "camera_e2e_import_contract/camera_e2e_import_contract.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load after flat_sensor_bundle to verify every CameraE2E objective pointer resolves before adapter import.",
            notes="Package-level import contract with product-blocked validation and per-sensor contract file list.",
        ),
        artifact_row(
            package_dir,
            "import_contract_by_sensor",
            "camera_e2e_import_contract_by_sensor",
            "camera_e2e_import_contract/camera_e2e_import_contract_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug to find each per-sensor import contract JSON and pointer/loadability counts.",
            notes="One row per sensor with requirement count, pointer resolution count, research allowed count, and product-blocked status.",
        ),
        artifact_row(
            package_dir,
            "import_contract_by_requirement",
            "camera_e2e_import_contract_by_requirement",
            "camera_e2e_import_contract/camera_e2e_import_contract_by_requirement.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + requirement_id for the canonical CameraE2E objective import row.",
            notes="147-row downstream import table: 7 sensors x 21 requested CameraE2E requirements.",
        ),
        artifact_row(
            package_dir,
            "import_contract_checks",
            "camera_e2e_import_contract_checks",
            "camera_e2e_import_contract/camera_e2e_import_contract_checks.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Read before import to confirm all flat JSON pointers resolve and product import remains blocked.",
            notes="Validation checks for the downstream CameraE2E import contract.",
        ),
        artifact_row(
            package_dir,
            "canonical_payload",
            "camera_e2e_canonical_payload_package",
            "camera_e2e_canonical_payload/camera_e2e_canonical_payload.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Preferred CameraE2E adapter payload after import contract validation; load per-sensor payload JSONs from canonical_payload_json_files.",
            notes="Repackages flat models into Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling payload sections.",
        ),
        artifact_row(
            package_dir,
            "canonical_payload_by_sensor",
            "camera_e2e_canonical_payload_by_sensor",
            "camera_e2e_canonical_payload/camera_e2e_canonical_payload_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug to find each canonical per-sensor CameraE2E payload JSON.",
            notes="One row per sensor with canonical payload row counts and product-blocked gate.",
        ),
        artifact_row(
            package_dir,
            "canonical_payload_checks",
            "camera_e2e_canonical_payload_checks",
            "camera_e2e_canonical_payload/camera_e2e_canonical_payload_checks.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Read before canonical payload import to confirm section row counts and fail-closed product status.",
            notes="Validation checks for canonical payload completeness.",
        ),
        artifact_row(
            package_dir,
            "response_trace",
            "pixel_response_calculation_trace",
            "camera_e2e_response_trace/camera_e2e_response_trace.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + runtime_id to audit runtime response against CFA/OCL/passivation/Si material sanity inputs.",
            notes="Full row-level trace for response/QE proxy construction; product use remains blocked.",
        ),
        artifact_row(
            package_dir,
            "response_trace_summary",
            "pixel_response_trace_summary_by_sensor",
            "camera_e2e_response_trace/camera_e2e_response_trace_summary.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for response trace coverage and product gate summary.",
            notes="Compact per-sensor response trace summary.",
        ),
        artifact_row(
            package_dir,
            "response_example",
            "readable_cfa_to_si_to_qe_examples",
            "camera_e2e_response_example/camera_e2e_response_example.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug + color_channel as a compact human-readable example of CFA transmission, Si absorption, QE proxy, and crosstalk split.",
            notes="Explains representative center-field R/G/B or clear rows for review and debugging.",
        ),
        artifact_row(
            package_dir,
            "response_example_summary",
            "readable_qe_example_summary_by_sensor",
            "camera_e2e_response_example/camera_e2e_response_example_summary.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug to verify each sensor has representative response examples.",
            notes="Per-sensor summary of the readable CFA-to-Si-to-QE examples.",
        ),
        artifact_row(
            package_dir,
            "method_provenance_matrix",
            "solver_external_proxy_method_matrix",
            "camera_e2e_method_provenance/camera_e2e_method_provenance_matrix.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + requirement_id before consuming numeric rows to distinguish solver, external DB, structural topology, and proxy/prior values.",
            notes="Per-sensor requirement source-method matrix for CameraE2E risk routing.",
        ),
        artifact_row(
            package_dir,
            "method_provenance_by_sensor",
            "method_source_summary_by_sensor",
            "camera_e2e_method_provenance/camera_e2e_method_provenance_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for source-class counts and product-use guard.",
            notes="Sensor-level method provenance summary.",
        ),
        artifact_row(
            package_dir,
            "source_integrity_matrix",
            "requirement_source_method_uncertainty_join",
            "camera_e2e_lut_source_integrity/camera_e2e_lut_source_integrity_matrix.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + requirement_id to see source class, calculation method, proxy/external/solver dependency, and uncertainty band in one row.",
            notes="Combined source-integrity matrix for CameraE2E loader risk routing.",
        ),
        artifact_row(
            package_dir,
            "source_integrity_by_sensor",
            "source_integrity_summary_by_sensor",
            "camera_e2e_lut_source_integrity/camera_e2e_lut_source_integrity_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for source-integrity counts and uncertainty product gate.",
            notes="Sensor-level source-integrity summary.",
        ),
        artifact_row(
            package_dir,
            "sensor_deliverable_summary",
            "per_sensor_camera_e2e_deliverable_index",
            "camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load by slug as the first human-readable per-sensor selector for CameraE2E research/trend use.",
            notes="One-row-per-sensor summary of loader paths, source-integrity coverage, row counts, uncertainty bands, and product blockers.",
        ),
        artifact_row(
            package_dir,
            "sensor_deliverable_summary_json",
            "per_sensor_camera_e2e_deliverable_manifest",
            "camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Read validation.status and product_ready_count before loading per-sensor values.",
            notes="JSON validation manifest for the sensor deliverable summary.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_query",
            "self_contained_flat_sensor_camera_e2e_query_manifest",
            "camera_e2e_flat_sensor_query/camera_e2e_flat_sensor_query.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Example direct CameraE2E query using only flat per-sensor JSON files; product mode remains blocked.",
            notes="Validates that embedded flat rows can produce response, spectral/material/electrical/readout/module joins and scalar raw-DN/SNR outputs.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_query_csv",
            "self_contained_flat_sensor_camera_e2e_query_rows",
            "camera_e2e_flat_sensor_query/camera_e2e_flat_sensor_query.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use as a minimal example of CameraE2E rows generated from flat sensor JSONs.",
            notes="Rows include response/QE proxy, crosstalk kernel summary, CRA, material/color joins, electrical/readout/module joins, raw DN, SNR, and product blockers.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_query_summary",
            "self_contained_flat_sensor_camera_e2e_query_summary",
            "camera_e2e_flat_sensor_query/camera_e2e_flat_sensor_query_summary.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load for a compact per-sensor/color query summary.",
            notes="Summarizes allowed rows, signal, edge-to-center trend when present, raw DN, SNR, and crosstalk fractions.",
        ),
        artifact_row(
            package_dir,
            "flat_sensor_product_query_probe",
            "self_contained_flat_sensor_product_block_probe",
            "camera_e2e_flat_sensor_query_product_probe/camera_e2e_flat_sensor_query.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to verify product mode fails closed from the flat sensor bundle.",
            notes="Expected allowed_query_count is zero until product gates are closed.",
        ),
        artifact_row(
            package_dir,
            "analysis_report",
            "camera_e2e_design_facing_analysis_report",
            "camera_e2e_analysis_report/camera_e2e_analysis_report.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use after the flat sensor query to review per-sensor CameraE2E usability, edge/CRA trends, crosstalk, and CHECK warnings.",
            notes="Summarizes flat query rows into per-sensor and per-channel design-facing report tables while preserving product blockers.",
        ),
        artifact_row(
            package_dir,
            "analysis_by_sensor",
            "camera_e2e_analysis_by_sensor",
            "camera_e2e_analysis_report/camera_e2e_analysis_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for one-row-per-sensor analysis gate, edge/crosstalk ranges, and product blockers.",
            notes="Recommended compact table for design review.",
        ),
        artifact_row(
            package_dir,
            "analysis_by_channel",
            "camera_e2e_analysis_by_channel",
            "camera_e2e_analysis_report/camera_e2e_analysis_by_channel.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + color_channel + wavelength_nm to inspect field coverage, edge-to-center response, and summary gates.",
            notes="Flags partial field coverage and near-zero edge response as CHECK.",
        ),
        artifact_row(
            package_dir,
            "analysis_actions",
            "camera_e2e_analysis_actions",
            "camera_e2e_analysis_report/camera_e2e_analysis_actions.csv",
            sensor_scope="check_rows",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to schedule follow-up solver coverage or measured-input work for CHECK rows.",
            notes="Action list generated from design-facing analysis warnings.",
        ),
        artifact_row(
            package_dir,
            "use_scope_summary",
            "camera_e2e_use_scope_decision_summary",
            "camera_e2e_use_scope_summary/camera_e2e_use_scope_summary.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load before runtime LUTs to route each sensor into trend, anchor, schema-prior, or product-blocked workflows.",
            notes="Top-level CameraE2E use-scope router; product use remains blocked unless product gates pass.",
        ),
        artifact_row(
            package_dir,
            "use_scope_by_sensor",
            "camera_e2e_use_scope_by_sensor",
            "camera_e2e_use_scope_summary/camera_e2e_use_scope_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for camera_e2e_use_scope, allowed use, product gate, crosstalk next action, and required product inputs.",
            notes="One row per sensor with safe CameraE2E routing and first crosstalk batch command.",
        ),
        artifact_row(
            package_dir,
            "use_scope_by_domain",
            "camera_e2e_use_scope_by_domain",
            "camera_e2e_use_scope_summary/camera_e2e_use_scope_by_domain.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug and domain to route Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling separately.",
            notes="Domain-level allowed/blocked use table for CameraE2E integration.",
        ),
        artifact_row(
            package_dir,
            "use_scope_next_actions",
            "camera_e2e_use_scope_next_actions",
            "camera_e2e_use_scope_summary/camera_e2e_use_scope_next_actions.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use priority_rank and action_type to schedule solver batches or request measured calibration inputs.",
            notes="Actionable list combining crosstalk batch priorities and measured-input blockers.",
        ),
        artifact_row(
            package_dir,
            "runtime_lut",
            "field_response_runtime_lut",
            "camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv",
            sensor_scope="all_sensors",
            gate=gate,
            product_usable=product_ready,
            loader_hint="Join by slug, field, wavelength, and color channel.",
            notes="Response, CRA, ML shift, crosstalk summary, gates, and uncertainty columns.",
        ),
        artifact_row(
            package_dir,
            "runtime_crosstalk_kernel",
            "optical_crosstalk_kernel_lut",
            "camera_e2e_runtime_bundle/camera_e2e_runtime_crosstalk_kernel.csv",
            sensor_scope="all_sensors",
            gate=gate,
            product_usable=product_ready,
            loader_hint="Join to runtime rows by runtime_id.",
            notes="Compact crosstalk kernels; current kernels are research/trend surrogates.",
        ),
        artifact_row(
            package_dir,
            "runtime_npz",
            "typed_runtime_arrays",
            "camera_e2e_runtime_bundle/camera_e2e_runtime_bundle.npz",
            sensor_scope="all_sensors",
            gate=gate,
            product_usable=product_ready,
            loader_hint="Optional high-throughput loader for runtime and kernel arrays.",
            notes="Compressed typed-array export of runtime bundle.",
        ),
        artifact_row(
            package_dir,
            "color_spectral_response",
            "spectral_color_response",
            "camera_e2e_color_response/camera_e2e_spectral_response.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, wavelength_nm, and color_channel.",
            notes="CFA proxy spectral response scaled by runtime anchors where available.",
        ),
        artifact_row(
            package_dir,
            "color_matrix_seed",
            "rgb_to_xyz_seed",
            "camera_e2e_color_response/camera_e2e_color_matrix_seed.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use as a research CCM seed only; not a calibrated color matrix.",
            notes="Monochrome sensors are marked not applicable.",
        ),
        artifact_row(
            package_dir,
            "material_nk_lut",
            "fdtd_material_nk_and_cfa_proxy_lut",
            "camera_e2e_material_tables/camera_e2e_material_nk_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, material_family, material_key, color_channel, and wavelength_nm.",
            notes="Explicit CFA/OCL/passivation/Si n,k and CFA transmission proxy rows; product use needs measured material data.",
        ),
        artifact_row(
            package_dir,
            "cfa_provenance",
            "cfa_color_material_provenance_audit",
            "camera_e2e_cfa_provenance/camera_e2e_cfa_provenance.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load package summary and by-sensor CSV before trusting color/material proxy rows.",
            notes="Flags sensor-specific CFA proxy, default-thickness proxy, monochrome clear proxy, and generic RGB fallback for unknown CFA patterns.",
        ),
        artifact_row(
            package_dir,
            "cfa_provenance_by_sensor",
            "cfa_color_material_provenance_by_sensor",
            "camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug to preserve cfa_provenance_class and cfa_assumption_gate.",
            notes="Per-sensor CFA source/fallback class for CameraE2E consumer rows.",
        ),
        artifact_row(
            package_dir,
            "cfa_db_tables",
            "dedicated_cfa_db_table_manifest",
            "camera_e2e_cfa_db_tables/camera_e2e_cfa_db_tables.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load with cfa_db_by_sensor and cfa_db_transmission_lut before using CFA/color rows.",
            notes="Validates dedicated CFA DB lookup tables and flags unknown-CFA generic RGB fallback rows.",
        ),
        artifact_row(
            package_dir,
            "cfa_db_by_sensor",
            "dedicated_cfa_db_by_sensor",
            "camera_e2e_cfa_db_tables/camera_e2e_cfa_db_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for CFA pattern, thickness, source, channel availability, and product blockers.",
            notes="One row per packaged sensor from image_sensor_db/optical_qe_db.",
        ),
        artifact_row(
            package_dir,
            "cfa_db_transmission_lut",
            "dedicated_cfa_db_transmission_lut",
            "camera_e2e_cfa_db_tables/camera_e2e_cfa_db_transmission_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug + color_channel + wavelength_nm for CFA n,k and absorption-only transmission.",
            notes="Rows remain research/proxy until measured CFA material and spectral response pass product gates.",
        ),
        artifact_row(
            package_dir,
            "capability_profile",
            "camera_e2e_capability_profile_audit",
            "camera_e2e_capability_profile/camera_e2e_capability_profile.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load package-level use-scope profile to decide research/plumbing/trend suitability.",
            notes="Summarizes per-domain use scopes and preserves product-blocked policy.",
        ),
        artifact_row(
            package_dir,
            "capability_profile_by_sensor",
            "camera_e2e_capability_profile_by_sensor",
            "camera_e2e_capability_profile/camera_e2e_capability_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug for overall_use_scope and per-domain scope columns.",
            notes="Per-sensor CameraE2E use-scope matrix.",
        ),
        artifact_row(
            package_dir,
            "lut_trust_assessment",
            "camera_e2e_lut_trust_assessment",
            "camera_e2e_lut_trust_assessment/camera_e2e_lut_trust_assessment.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load to separate research usability, solver/measured evidence, and product calibration confidence.",
            notes="Trust scores are not physical accuracy percentages; they are CameraE2E usage guards.",
        ),
        artifact_row(
            package_dir,
            "lut_trust_by_sensor",
            "camera_e2e_lut_trust_by_sensor",
            "camera_e2e_lut_trust_assessment/camera_e2e_lut_trust_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug to route sensors into product-blocked, field-trend, sparse-anchor, or structural-prior workflows.",
            notes="Per-sensor trust class and research/evidence/product score split.",
        ),
        artifact_row(
            package_dir,
            "lut_trust_by_domain",
            "camera_e2e_lut_trust_by_domain",
            "camera_e2e_lut_trust_assessment/camera_e2e_lut_trust_by_domain.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug and domain to decide which CameraE2E domains can be used for research only.",
            notes="Domain-level trust class for Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling.",
        ),
        artifact_row(
            package_dir,
            "lut_trust_by_requirement",
            "camera_e2e_lut_trust_by_requirement",
            "camera_e2e_lut_trust_assessment/camera_e2e_lut_trust_by_requirement.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug and requirement_id when CameraE2E needs per-requirement confidence.",
            notes="One trust row per coverage-matrix requirement row.",
        ),
        artifact_row(
            package_dir,
            "prior_seed_models",
            "electrical_readout_module_prior_seed",
            "camera_e2e_prior_seed_models/camera_e2e_prior_seed_models.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load per-sensor JSONs from camera_e2e_prior_seed_models/models.",
            notes="Research-only seeds for noise, FWC, CG, dark current, readout, defects, and module alignment.",
        ),
        artifact_row(
            package_dir,
            "electrical_noise_lut",
            "electrical_noise_temperature_exposure_lut",
            "camera_e2e_electrical_readout_tables/camera_e2e_electrical_noise_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, temperature_c, exposure_s, and signal_fraction.",
            notes="Research prior table for FWC, CG, dark current, DSNU/PRNU, noise, and electrical crosstalk status.",
        ),
        artifact_row(
            package_dir,
            "readout_gain_lut",
            "readout_raw_gain_adc_lut",
            "camera_e2e_electrical_readout_tables/camera_e2e_readout_gain_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, analog_gain_x, digital_gain_x, and adc_bit_depth.",
            notes="Research prior table for gain, black level, ADC, FPN, timing, and defect statistics.",
        ),
        artifact_row(
            package_dir,
            "binning_remosaic_lut",
            "binning_remosaic_mode_lut",
            "camera_e2e_electrical_readout_tables/camera_e2e_binning_remosaic_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug and mode_id to redefine gain/noise/crosstalk for binned modes.",
            notes="Research prior table; measured mode calibration is still required.",
        ),
        artifact_row(
            package_dir,
            "module_coupling_lut",
            "module_field_cra_vignetting_lut",
            "camera_e2e_module_coupling/camera_e2e_module_coupling_field_lut.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, field_case, wavelength_nm for module coupling priors.",
            notes="Product use needs module raytrace/measured CRA, sensor CRA, pupil, and vignetting data.",
        ),
        artifact_row(
            package_dir,
            "quantitative_execution_plan",
            "quantitative_fdtd_execution_plan",
            "camera_e2e_quantitative_execution_plan.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to plan required field/crosstalk solver coverage and expected runtime.",
            notes="Includes periodic cell volume, finite-array crosstalk domain factor, full-sweep point counts, and runtime estimates.",
        ),
        artifact_row(
            package_dir,
            "quantitative_point_queue",
            "point_sized_solver_queue",
            "camera_e2e_quantitative_point_queue.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use by slug, solver, color, field case, and wavelength to schedule solver jobs.",
            notes="Concrete point commands for field response and finite-array crosstalk evidence generation.",
        ),
        artifact_row(
            package_dir,
            "resource_limited_batch_plan",
            "resource_limited_crosstalk_batch_plan",
            "camera_e2e_resource_limited_batch_plan.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run these commands outside the local interactive UX, then rerun the merge/build pipeline.",
            notes="Records local resource-limited finite-array crosstalk points and batch/cluster commands.",
        ),
        artifact_row(
            package_dir,
            "quantitative_coverage",
            "quantitative_solver_coverage",
            "camera_e2e_quantitative_coverage.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug and solver to distinguish PASS, CHECK, FAIL, and resource-limited coverage.",
            notes="Per-sensor quantitative field and finite-array crosstalk coverage summary.",
        ),
        artifact_row(
            package_dir,
            "closure_plan",
            "product_gate_closure_plan",
            "camera_e2e_closure_plan/camera_e2e_closure_plan.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use as the operational checklist for measured data import and remaining solver jobs.",
            notes="Validated closure manifest with measured optical input blockers, electrical/readout/module calibration blockers, quantitative queue rows, and resource-limited crosstalk batch rows.",
        ),
        artifact_row(
            package_dir,
            "closure_plan_csv",
            "product_gate_closure_plan_rows",
            "camera_e2e_closure_plan/camera_e2e_closure_plan.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Filter by track, slug, solver, and blocking_gate to schedule closure work.",
            notes="One row per measured optical blocker, measured calibration blocker, standard solver point, or resource-limited crosstalk batch point.",
        ),
        artifact_row(
            package_dir,
            "closure_checks",
            "product_gate_closure_validation_checks",
            "camera_e2e_closure_plan/camera_e2e_closure_checks.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load to verify closure plan coverage before starting product-gate closure work.",
            notes="Checks measured optical blocker coverage, measured calibration blocker coverage, resource-limited crosstalk coverage, commands, and batch rows.",
        ),
        artifact_row(
            package_dir,
            "readiness_report",
            "product_gate_audit",
            "camera_e2e_readiness_audit/camera_e2e_lut_readiness_report.json",
            sensor_scope="all_sensors",
            gate=str(pipeline.get("validation", {}).get("status", "CHECK")),
            product_usable=False,
            loader_hint="Use to block product ingest and list closure tasks.",
            notes="Separates research handoff validity from product LUT readiness.",
        ),
        artifact_row(
            package_dir,
            "mesh_confidence",
            "mesh_resolution_confidence_audit",
            "camera_e2e_mesh_confidence/camera_e2e_mesh_confidence.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to decide whether runtime rows are structural priors, sparse anchors, partial field trends, or product candidates.",
            notes="Summarizes quantitative field/crosstalk mesh coverage, grid-resolution gates, and recommended CameraE2E use.",
        ),
        artifact_row(
            package_dir,
            "field_execution_pack",
            "field_qe_execution_pack",
            "camera_e2e_field_execution_pack/camera_e2e_field_execution_pack.json",
            sensor_scope="all_field_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load after mesh confidence to schedule center spectral anchors, green CRA anchors, and full field/QE closure runs.",
            notes="Execution handoff only; product field/QE remains blocked until selected jobs pass product mesh/convergence gates.",
        ),
        artifact_row(
            package_dir,
            "field_execution_jobs_csv",
            "field_qe_execution_job_rows",
            "camera_e2e_field_execution_pack/camera_e2e_field_execution_jobs.csv",
            sensor_scope="all_field_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use priority_rank, action_type, local_feasibility, and command columns to schedule the next Meep field/QE runs.",
            notes="One execution row per remaining or stale quantitative field point.",
        ),
        artifact_row(
            package_dir,
            "field_execution_scripts_csv",
            "field_qe_execution_script_index",
            "camera_e2e_field_execution_pack/camera_e2e_field_execution_scripts.csv",
            sensor_scope="all_field_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Script index for center spectral/color anchors, green CRA anchors, failed reruns, full closure, and refresh scripts.",
            notes="Scripts are handoff helpers, not accuracy evidence.",
        ),
        artifact_row(
            package_dir,
            "field_center_spectral_anchor_script",
            "center_spectral_color_anchor_shell_script",
            "camera_e2e_field_execution_pack/run_center_spectral_color_anchors.sh",
            sensor_scope="center_color_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run first to improve per-sensor color response and center QE anchors.",
            notes="Long-running quantitative Meep field jobs; run selectively or through batch scheduling.",
        ),
        artifact_row(
            package_dir,
            "field_green_cra_anchor_script",
            "green_cra_field_anchor_shell_script",
            "camera_e2e_field_execution_pack/run_green_cra_field_anchors.sh",
            sensor_scope="green_cra_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run after center anchors to improve CRA/shading field-map confidence.",
            notes="Long-running quantitative Meep field jobs; product gates remain blocked after execution until package audits pass.",
        ),
        artifact_row(
            package_dir,
            "field_failed_or_stale_rerun_script",
            "failed_or_stale_field_rerun_shell_script",
            "camera_e2e_field_execution_pack/run_failed_or_stale_field_reruns.sh",
            sensor_scope="failed_or_stale_field_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Inspect or rerun failed/stale quantitative field points with the current stack and target resolution.",
            notes="Currently includes failed grid points such as stale low-resolution runs.",
        ),
        artifact_row(
            package_dir,
            "field_all_quantitative_script",
            "all_remaining_field_quantitative_shell_script",
            "camera_e2e_field_execution_pack/run_all_field_quantitative_remaining.sh",
            sensor_scope="all_remaining_field_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use for full field/QE closure through controlled batch scheduling.",
            notes="Do not launch blindly on a laptop; each quantitative field point can take tens of minutes.",
        ),
        artifact_row(
            package_dir,
            "field_refresh_after_solver_script",
            "post_field_solver_refresh_shell_script",
            "camera_e2e_field_execution_pack/refresh_after_field_jobs.sh",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run after selected field jobs finish to merge quantitative points and rebuild the CameraE2E package pipeline.",
            notes="Refresh helper only.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_support_audit",
            "finite_array_crosstalk_support_audit",
            "camera_e2e_crosstalk_support_audit/camera_e2e_crosstalk_support_audit.json",
            sensor_scope="pilot_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load before using compact crosstalk kernels to see finite-array support truncation and product-run feasibility.",
            notes="Low-resolution finite-array support sweep; current role is support-risk guard, not product crosstalk evidence.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_support_by_sensor",
            "finite_array_crosstalk_support_by_sensor",
            "camera_e2e_crosstalk_support_audit/camera_e2e_crosstalk_support_by_sensor.csv",
            sensor_scope="pilot_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Join by slug/color/wavelength/field_case for support-size recommendations.",
            notes="Records best available low-resolution support pilot and next support size to test.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_support_pilots",
            "finite_array_crosstalk_support_pilot_rows",
            "camera_e2e_crosstalk_support_audit/camera_e2e_crosstalk_support_pilots.csv",
            sensor_scope="pilot_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to inspect truncation, grid, convergence, and product-use gates for every support pilot.",
            notes="3x3 through expanded-neighborhood pilot rows where available.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_product_candidates",
            "finite_array_crosstalk_product_candidate_commands",
            "camera_e2e_crosstalk_support_audit/camera_e2e_crosstalk_product_candidates.csv",
            sensor_scope="pilot_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use commands and voxel estimates to schedule batch/HPC product-resolution crosstalk runs.",
            notes="Candidate quantitative finite-array crosstalk commands; local feasibility is estimated from voxel count.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_batch_priority",
            "support_aware_crosstalk_batch_priority",
            "camera_e2e_crosstalk_batch_priority/camera_e2e_crosstalk_batch_priority.json",
            sensor_scope="all_crosstalk_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load after crosstalk support audit to schedule product primary and support-discovery crosstalk jobs.",
            notes="One actionable row per crosstalk condition; product use remains blocked until selected jobs pass product mesh/convergence gates.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_batch_priority_csv",
            "support_aware_crosstalk_batch_priority_rows",
            "camera_e2e_crosstalk_batch_priority/camera_e2e_crosstalk_batch_priority.csv",
            sensor_scope="all_crosstalk_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use priority_rank, action_type, local_feasibility, and command columns to schedule the next finite-array crosstalk runs.",
            notes="Distinguishes product-resolution primary candidates from low-resolution support-discovery jobs.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_execution_pack",
            "crosstalk_hpc_and_support_execution_pack",
            "camera_e2e_crosstalk_execution_pack/camera_e2e_crosstalk_execution_pack.json",
            sensor_scope="all_crosstalk_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load after crosstalk batch priority to find runnable product-primary and support-discovery scripts.",
            notes="Execution handoff only; product crosstalk remains blocked until selected jobs pass product mesh/convergence gates.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_execution_jobs_csv",
            "crosstalk_execution_job_rows",
            "camera_e2e_crosstalk_execution_pack/camera_e2e_crosstalk_execution_jobs.csv",
            sensor_scope="all_crosstalk_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use execution_group to separate product_primary_hpc, support_discovery_local_candidate, and batch/reformulation jobs.",
            notes="One execution row per crosstalk priority condition.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_execution_scripts_csv",
            "crosstalk_execution_script_index",
            "camera_e2e_crosstalk_execution_pack/camera_e2e_crosstalk_execution_scripts.csv",
            sensor_scope="all_crosstalk_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Script index for product-primary HPC, local support candidate, batch/reformulation, and refresh scripts.",
            notes="Scripts are handoff helpers, not accuracy evidence.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_local_probe_evidence",
            "crosstalk_local_runtime_probe_evidence",
            "camera_e2e_crosstalk_execution_pack/camera_e2e_crosstalk_local_probe_evidence.csv",
            sensor_scope="probe_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use to understand local runtime cost before launching broad finite-array support discovery.",
            notes="Records interrupted local probe setup timing; not a completed solver result.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_product_primary_hpc_script",
            "product_primary_crosstalk_hpc_shell_script",
            "camera_e2e_crosstalk_execution_pack/run_product_primary_hpc.sh",
            sensor_scope="support_established_primary_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run on HPC/large workstation/domain-decomposition environment, then refresh audits.",
            notes="Executable helper for the 9 current product-primary finite-array crosstalk jobs.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_support_discovery_local_script",
            "local_candidate_crosstalk_support_discovery_shell_script",
            "camera_e2e_crosstalk_execution_pack/run_support_discovery_local_candidates.sh",
            sensor_scope="local_candidate_support_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run selectively; do not launch the full local candidate list blindly on a laptop.",
            notes="Prior local probe shows n15/res20 finite-array setup can be expensive.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_support_discovery_batch_script",
            "batch_or_reformulation_crosstalk_support_discovery_shell_script",
            "camera_e2e_crosstalk_execution_pack/run_support_discovery_batch_or_reformulation.sh",
            sensor_scope="batch_or_reformulation_support_conditions",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run through batch/HPC or after crosstalk solver reformulation.",
            notes="Broad support discovery work that should not be run as a laptop batch.",
        ),
        artifact_row(
            package_dir,
            "crosstalk_refresh_after_solver_script",
            "post_crosstalk_solver_refresh_shell_script",
            "camera_e2e_crosstalk_execution_pack/refresh_after_solver_jobs.sh",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Run after selected crosstalk jobs finish to rebuild support audit, priority, pipeline, and objective acceptance.",
            notes="Refresh helper only.",
        ),
        artifact_row(
            package_dir,
            "product_closure_summary",
            "camera_e2e_product_closure_summary",
            "camera_e2e_product_closure_summary/camera_e2e_product_closure_summary.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Start here for a compact per-sensor view of current use scope, product blockers, and first closure action.",
            notes="Product-readiness worklist summary; not an accuracy certificate.",
        ),
        artifact_row(
            package_dir,
            "product_closure_by_sensor",
            "camera_e2e_product_closure_by_sensor_rows",
            "camera_e2e_product_closure_summary/camera_e2e_product_closure_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="One row per sensor with mesh pass counts, blocker counts, crosstalk/HPC job counts, and first action.",
            notes="Use for design review and CameraE2E readiness routing.",
        ),
        artifact_row(
            package_dir,
            "product_closure_by_domain",
            "camera_e2e_product_closure_by_domain_rows",
            "camera_e2e_product_closure_summary/camera_e2e_product_closure_by_domain.csv",
            sensor_scope="all_sensors_by_domain",
            gate="CHECK",
            product_usable=False,
            loader_hint="Domain-level trust, product gate counts, blockers, and recommended next action.",
            notes="Derived from LUT trust assessment domain rows.",
        ),
        artifact_row(
            package_dir,
            "product_closure_checks",
            "camera_e2e_product_closure_validation_checks",
            "camera_e2e_product_closure_summary/camera_e2e_product_closure_checks.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Validation checks for product-closure summary completeness and product-gate blocking.",
            notes="Fails if sensor rows or closure actions are missing.",
        ),
        artifact_row(
            package_dir,
            "usage_policy",
            "camera_e2e_usage_policy",
            "camera_e2e_usage_policy/camera_e2e_usage_policy.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Load before runtime/flat bundle ingestion to choose research/product filters.",
            notes="Strict product filters are expected to have zero rows in the current package.",
        ),
        artifact_row(
            package_dir,
            "usage_policy_by_sensor",
            "camera_e2e_usage_policy_by_sensor_rows",
            "camera_e2e_usage_policy/camera_e2e_usage_policy_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Per-sensor recommended CameraE2E mode, allowed modes, blocked modes, and closure action.",
            notes="Use to prevent schema-prior sensors from being treated as calibrated LUTs.",
        ),
        artifact_row(
            package_dir,
            "usage_policy_by_domain",
            "camera_e2e_usage_policy_by_domain_rows",
            "camera_e2e_usage_policy/camera_e2e_usage_policy_by_domain.csv",
            sensor_scope="all_sensors_by_domain",
            gate="CHECK",
            product_usable=False,
            loader_hint="Domain-specific use restrictions for Optical/Color, Pixel/Electrical, Readout/RAW, and Module Coupling.",
            notes="Derived from use-scope and trust assessment domain rows.",
        ),
        artifact_row(
            package_dir,
            "usage_policy_runtime_filters",
            "camera_e2e_usage_policy_runtime_filters",
            "camera_e2e_usage_policy/camera_e2e_usage_policy_runtime_filters.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Apply filter_id=research_runtime_rows for research and strict_product_runtime_rows for product mode.",
            notes="Product filter row count must stay zero until product gates open.",
        ),
        artifact_row(
            package_dir,
            "usage_policy_checks",
            "camera_e2e_usage_policy_validation_checks",
            "camera_e2e_usage_policy/camera_e2e_usage_policy_checks.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Validation checks for policy completeness and product-filter blocking.",
            notes="Fails if product ingest rows are exposed while product gates are blocked.",
        ),
        artifact_row(
            package_dir,
            "adapter_examples",
            "camera_e2e_adapter_examples",
            "camera_e2e_adapter_examples/camera_e2e_adapter_examples.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Open after usage_policy to find one concrete CameraE2E load recipe per sensor.",
            notes="Per-sensor examples include research query command and fail-closed product probe command.",
        ),
        artifact_row(
            package_dir,
            "adapter_examples_by_sensor",
            "camera_e2e_adapter_examples_by_sensor_rows",
            "camera_e2e_adapter_examples/camera_e2e_adapter_examples_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Compact index of per-sensor adapter JSON files, query counts, gates, and loader paths.",
            notes="Product allowed query count is expected to be zero.",
        ),
        artifact_row(
            package_dir,
            "adapter_examples_checks",
            "camera_e2e_adapter_examples_validation_checks",
            "camera_e2e_adapter_examples/camera_e2e_adapter_examples_checks.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Validation checks for adapter example completeness and product query blocking.",
            notes="Fails if any per-sensor adapter recipe is missing or product mode opens.",
        ),
        artifact_row(
            package_dir,
            "adapter_smoke",
            "camera_e2e_adapter_smoke",
            "camera_e2e_adapter_smoke/camera_e2e_adapter_smoke.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Executable proof that per-sensor research queries load and product probes fail closed.",
            notes="Runs query_camera_e2e_flat_sensor_bundle.py for every sensor in research and product modes.",
        ),
        artifact_row(
            package_dir,
            "adapter_smoke_by_sensor",
            "camera_e2e_adapter_smoke_by_sensor_rows",
            "camera_e2e_adapter_smoke/camera_e2e_adapter_smoke_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Per-sensor adapter smoke return codes, output dirs, allowed query counts, and gate status.",
            notes="Product allowed query count is expected to be zero for every sensor.",
        ),
        artifact_row(
            package_dir,
            "adapter_smoke_checks",
            "camera_e2e_adapter_smoke_validation_checks",
            "camera_e2e_adapter_smoke/camera_e2e_adapter_smoke_checks.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Validation checks for executable adapter smoke.",
            notes="Fails if research queries do not load or product probes open.",
        ),
        artifact_row(
            package_dir,
            "objective_trace",
            "camera_e2e_objective_trace",
            "camera_e2e_objective_trace/camera_e2e_objective_trace.json",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Requirement-to-loader trace for every objective item and every sensor.",
            notes="Maps objective coverage rows to flat JSON sections, loader tables, and adapter smoke evidence.",
        ),
        artifact_row(
            package_dir,
            "objective_trace_by_requirement",
            "camera_e2e_objective_trace_by_requirement_rows",
            "camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement.csv",
            sensor_scope="all_sensors_by_requirement",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use this to find the flat JSON pointer and source artifacts for a specific objective requirement.",
            notes="Product gates remain blocked, but research trace rows should all be loadable.",
        ),
        artifact_row(
            package_dir,
            "objective_trace_by_requirement_summary",
            "camera_e2e_objective_trace_requirement_summary_rows",
            "camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement_summary.csv",
            sensor_scope="all_requirements",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use this 21-row summary to review each objective requirement's loader table, source class distribution, uncertainty band, and product blockers across sensors.",
            notes="One row per CameraE2E objective requirement; product gates remain blocked.",
        ),
        artifact_row(
            package_dir,
            "objective_trace_by_sensor",
            "camera_e2e_objective_trace_by_sensor_rows",
            "camera_e2e_objective_trace/camera_e2e_objective_trace_by_sensor.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Compact per-sensor objective trace counts and product blocker status.",
            notes="One row per sensor.",
        ),
        artifact_row(
            package_dir,
            "objective_trace_checks",
            "camera_e2e_objective_trace_validation_checks",
            "camera_e2e_objective_trace/camera_e2e_objective_trace_checks.csv",
            sensor_scope="package",
            gate="CHECK",
            product_usable=False,
            loader_hint="Validation checks for requirement mapping and product trace blocking.",
            notes="Fails if any objective requirement is unmapped or not loadable in research mode.",
        ),
        artifact_row(
            package_dir,
            "research_probe_all_sensors",
            "consumer_scalar_probe",
            "camera_e2e_sensor_probe_all_sensors/camera_e2e_sensor_probe_summary.csv",
            sensor_scope="all_sensors",
            gate="CHECK",
            product_usable=False,
            loader_hint="Use as a smoke test for signal/noise/raw-DN path.",
            notes="Generated with prior seeds; not a calibrated sensor characterization.",
        ),
    ]


def build_sensor_rows(package_dir: Path) -> list[dict[str, Any]]:
    summary_rows = read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv")
    runtime_counts = group_count(read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv"), "slug")
    kernel_counts = group_count(read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv"), "slug")
    spectral_counts = group_count(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv"), "slug")
    material_counts = group_count(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv"), "slug")
    response_example_counts = group_count(read_csv_rows(package_dir / "camera_e2e_response_example" / "camera_e2e_response_example.csv"), "slug")
    method_provenance_counts = group_count(read_csv_rows(package_dir / "camera_e2e_method_provenance" / "camera_e2e_method_provenance_matrix.csv"), "slug")
    source_integrity_counts = group_count(read_csv_rows(package_dir / "camera_e2e_lut_source_integrity" / "camera_e2e_lut_source_integrity_matrix.csv"), "slug")
    quantitative_queue_counts = group_count(read_csv_rows(package_dir / "camera_e2e_quantitative_point_queue.csv"), "slug")
    resource_limited_counts = group_count(read_csv_rows(package_dir / "camera_e2e_resource_limited_batch_plan.csv"), "slug")
    cfa_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv"), "slug")
    cfa_db_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_by_sensor.csv"), "slug")
    cfa_db_transmission_counts = group_count(read_csv_rows(package_dir / "camera_e2e_cfa_db_tables" / "camera_e2e_cfa_db_transmission_lut.csv"), "slug")
    capability_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_capability_profile" / "camera_e2e_capability_by_sensor.csv"), "slug")
    mesh_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv"), "slug")
    matrix_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv"), "slug")
    module_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_summary.csv"), "slug")
    prior_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_prior_seed_models" / "camera_e2e_prior_seed_summary.csv"), "slug")
    electrical_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_readout_summary.csv"), "slug")
    flat_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv"), "slug")

    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        slug = row.get("slug", "")
        module = module_by_slug.get(slug, {})
        prior = prior_by_slug.get(slug, {})
        electrical = electrical_by_slug.get(slug, {})
        cfa = cfa_by_slug.get(slug, {})
        cfa_db = cfa_db_by_slug.get(slug, {})
        capability = capability_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        flat = flat_by_slug.get(slug, {})
        product_ready = boolish(row.get("camera_e2e_product_ready"))
        rows.append(
            {
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "model_json": row.get("model_json", ""),
                "flat_sensor_json": flat.get("flat_sensor_json", ""),
                "runtime_row_count": runtime_counts.get(slug, 0),
                "kernel_row_count": kernel_counts.get(slug, 0),
                "spectral_row_count": spectral_counts.get(slug, 0),
                "material_row_count": material_counts.get(slug, 0),
                "response_example_row_count": response_example_counts.get(slug, 0),
                "method_provenance_row_count": method_provenance_counts.get(slug, 0),
                "source_integrity_row_count": source_integrity_counts.get(slug, 0),
                "cfa_provenance_class": cfa.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa.get("cfa_assumption_gate", ""),
                "cfa_db_row_count": 1 if cfa_db else 0,
                "cfa_db_transmission_row_count": cfa_db_transmission_counts.get(slug, 0),
                "capability_overall_use_scope": capability.get("overall_use_scope", ""),
                "capability_spectral_qe_scope": capability.get("spectral_qe_scope", ""),
                "capability_color_response_scope": capability.get("color_response_scope", ""),
                "capability_crosstalk_scope": capability.get("optical_crosstalk_scope", ""),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
                "mesh_field_pass_points": mesh.get("field_pass_points", ""),
                "mesh_field_required_points": mesh.get("field_required_points", ""),
                "mesh_crosstalk_pass_points": mesh.get("crosstalk_pass_points", ""),
                "mesh_crosstalk_required_points": mesh.get("crosstalk_required_points", ""),
                "mesh_crosstalk_resource_limited_points": mesh.get("crosstalk_resource_limited_points", ""),
                "quantitative_queue_row_count": quantitative_queue_counts.get(slug, 0),
                "resource_limited_batch_row_count": resource_limited_counts.get(slug, 0),
                "color_matrix_gate": matrix_by_slug.get(slug, {}).get("gate", row.get("color_matrix_gate", "")),
                "electrical_row_count": electrical.get("electrical_row_count", ""),
                "readout_row_count": electrical.get("readout_row_count", ""),
                "binning_row_count": electrical.get("binning_row_count", ""),
                "module_field_row_count": module.get("field_row_count", ""),
                "prior_gate": prior.get("prior_gate", ""),
                "cra_mismatch_gate": row.get("cra_mismatch_gate", ""),
                "research_ingest_gate": row.get("research_ingest_gate", ""),
                "production_lut_gate": row.get("production_lut_gate", ""),
                "product_ready": product_ready,
                "primary_blockers": row.get("primary_blockers", ""),
                "loader_order": "sensor_deliverable_summary -> usage_policy -> consumer_bundle -> use_scope_summary -> sensor_model_json/flat_sensor_json -> source_integrity -> method_provenance -> response_example/response_trace -> capability_profile -> lut_trust_assessment -> coverage_matrix -> quantitative_execution_plan -> quantitative_point_queue/resource_limited_batch_plan -> runtime_lut -> crosstalk_kernel -> color_response -> material_nk_lut -> cfa_provenance -> electrical/readout_luts -> prior_seed -> module_coupling",
                "notes": "Research handoff valid; product use blocked by gates."
                if not product_ready
                else "Product-ready gates pass.",
            }
        )
    return rows


def validate(artifact_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]], package: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for row in artifact_rows:
        if not boolish(row.get("exists")):
            issues.append({"severity": "error", "code": "artifact_missing", "artifact_id": row.get("artifact_id"), "path": row.get("path")})
    expected_sensor_count = int(package.get("sensor_count", 0) or 0)
    if expected_sensor_count and len(sensor_rows) != expected_sensor_count:
        issues.append(
            {
                "severity": "error",
                "code": "sensor_count_mismatch",
                "expected": expected_sensor_count,
                "actual": len(sensor_rows),
            }
        )
    for row in sensor_rows:
        if int(row.get("runtime_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_runtime_rows_missing", "slug": row.get("slug")})
        if int(row.get("kernel_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_kernel_rows_missing", "slug": row.get("slug")})
        if int(row.get("spectral_row_count") or 0) <= 0:
            issues.append({"severity": "warning", "code": "sensor_spectral_rows_missing", "slug": row.get("slug")})
        if int(row.get("material_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_material_rows_missing", "slug": row.get("slug")})
        if int(row.get("response_example_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_response_example_rows_missing", "slug": row.get("slug")})
        if int(row.get("method_provenance_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_method_provenance_rows_missing", "slug": row.get("slug")})
        if int(row.get("source_integrity_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_source_integrity_rows_missing", "slug": row.get("slug")})
        if int(row.get("electrical_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_electrical_rows_missing", "slug": row.get("slug")})
        if int(row.get("readout_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_readout_rows_missing", "slug": row.get("slug")})
        if int(row.get("binning_row_count") or 0) <= 0:
            issues.append({"severity": "error", "code": "sensor_binning_rows_missing", "slug": row.get("slug")})
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    status = "PRODUCT_READY" if product_ready_count == len(sensor_rows) and sensor_rows else "RESEARCH_HANDOFF_READY_PRODUCT_BLOCKED"
    if any(issue.get("severity") == "error" for issue in issues):
        status = "FAIL"
    return {
        "schema": "camera_e2e_handoff_validation_v1",
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


def write_html(path: Path, manifest: dict[str, Any], artifact_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1360px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = manifest.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Handoff Manifest</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Handoff Manifest</h1>
<p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. This is the integration index for research handoff; product gates are preserved.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">handoff status</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("artifact_count", 0))}</div><div class="muted">artifacts</div></div>
<div class="card"><div class="metric warn">{html_cell(manifest.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Sensor Load Index</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Artifact Load Index</h2>{html_table(artifact_rows, ARTIFACT_COLUMNS)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, manifest: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_handoff_manifest_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_handoff_artifacts_csv"] = manifest["outputs"]["artifacts_csv"]
    outputs["camera_e2e_handoff_sensors_csv"] = manifest["outputs"]["sensors_csv"]
    outputs["camera_e2e_handoff_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_handoff_manifest"] = {
        "schema": manifest["schema"],
        "validation": manifest["validation"],
        "sensor_count": manifest["sensor_count"],
        "artifact_count": manifest["artifact_count"],
        "product_ready_count": manifest["product_ready_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def export_handoff(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    package = read_json(package_dir / "camera_e2e_lut_package.json")
    pipeline = read_json(package_dir / "camera_e2e_pipeline_validation" / "camera_e2e_pipeline_validation.json")
    artifact_rows = build_artifact_rows(package_dir, package, pipeline)
    sensor_rows = build_sensor_rows(package_dir)
    validation = validate(artifact_rows, sensor_rows, package)

    artifacts_csv = output_dir / "camera_e2e_handoff_artifacts.csv"
    sensors_csv = output_dir / "camera_e2e_handoff_sensors.csv"
    manifest_json = output_dir / "camera_e2e_handoff_manifest.json"
    html_path = output_dir / "index.html"

    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    manifest = {
        "schema": "camera_e2e_handoff_manifest_v1",
        "artifact_role": "camera_e2e_integration_load_index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "artifact_count": len(artifact_rows),
        "product_ready_count": product_ready_count,
        "validation": validation,
        "usage_policy": {
            "research": "Load is allowed when validation.pass is true and row-level gates are propagated.",
            "product": "Blocked until every sensor product_ready is true and product LUT gates pass.",
            "entrypoint": "Start with camera_e2e_sensor_deliverable_summary for per-sensor selection, then camera_e2e_usage_policy before loading runtime or flat-bundle LUT values; then load camera_e2e_handoff_sensors.csv or the per-sensor JSON files referenced there.",
            "sensor_deliverable_summary_inputs": "Load camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.csv or JSON first to see each sensor's recommended loader path, available row counts, source-integrity coverage, uncertainty ranges, and product blockers.",
            "usage_policy_inputs": "Load camera_e2e_usage_policy/camera_e2e_usage_policy.json and camera_e2e_usage_policy_runtime_filters.csv to choose research_runtime_rows or strict_product_runtime_rows. The strict product filter is expected to have zero rows in this package.",
            "adapter_example_inputs": "Load camera_e2e_adapter_examples/camera_e2e_adapter_examples.json or the per-sensor JSON examples under camera_e2e_adapter_examples/sensors for executable research/product-probe query recipes.",
            "adapter_smoke_inputs": "Load camera_e2e_adapter_smoke/camera_e2e_adapter_smoke.json to confirm those recipes were executed: research queries return allowed rows and product probes return zero allowed rows.",
            "objective_trace_inputs": "Load camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement.csv to map every objective requirement to flat JSON sections, loader tables, source artifacts, and adapter-smoke evidence.",
            "flat_sensor_bundle": "For direct CameraE2E ingestion, load camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_bundle.json and then the per-sensor JSON from sensor_model_json_files.",
            "source_integrity_inputs": "Load camera_e2e_lut_source_integrity/camera_e2e_lut_source_integrity_matrix.csv first when reviewers need one row containing source class, calculation method, solver/external/proxy dependency, uncertainty band, and product-use guard.",
            "method_provenance_inputs": "Load camera_e2e_method_provenance/camera_e2e_method_provenance_matrix.csv, or the method_provenance section inside each flat sensor JSON, before treating any numeric LUT row as solver-derived.",
            "response_example_inputs": "Load camera_e2e_response_example/camera_e2e_response_example.csv, or the response_example section inside each flat sensor JSON, to inspect representative CFA-to-Si-to-QE calculations.",
            "response_trace_inputs": "Load camera_e2e_response_trace/camera_e2e_response_trace.csv for full runtime row-level CFA/OCL/passivation/Si response traceability.",
            "material_inputs": "Load camera_e2e_material_tables/camera_e2e_material_nk_lut.csv for explicit CFA/OCL/passivation/Si n,k and CFA transmission proxy rows.",
            "cfa_provenance_inputs": "Load camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv to distinguish sensor-confirmed CFA proxy from generic fallback.",
            "cfa_db_inputs": "Load camera_e2e_cfa_db_tables/camera_e2e_cfa_db_by_sensor.csv and camera_e2e_cfa_db_transmission_lut.csv for direct CFA DB pattern/thickness/transmission lookup.",
            "capability_profile_inputs": "Load camera_e2e_capability_profile/camera_e2e_capability_by_sensor.csv for per-domain CameraE2E use scopes.",
            "use_scope_inputs": "Load camera_e2e_use_scope_summary/camera_e2e_use_scope_by_sensor.csv and by-domain CSV to route each sensor/domain before consuming runtime values.",
            "trust_assessment_inputs": "Load camera_e2e_lut_trust_assessment/ to distinguish research usability from solver evidence and product calibration confidence.",
            "electrical_readout_inputs": "Load camera_e2e_electrical_readout_tables CSVs for prior electrical/noise/readout/binning rows.",
            "coverage_inputs": "Load camera_e2e_coverage_matrix/camera_e2e_coverage_matrix.csv to inspect per-requirement evidence, row counts, and product blockers.",
            "mesh_confidence_inputs": "Load camera_e2e_mesh_confidence/camera_e2e_mesh_confidence.json to preserve numerical confidence class and mesh coverage limitations.",
            "quantitative_execution_inputs": "Load camera_e2e_quantitative_execution_plan.csv, camera_e2e_quantitative_point_queue.csv, and camera_e2e_resource_limited_batch_plan.csv before treating missing crosstalk/QE coverage as complete.",
            "closure_plan_inputs": "Load camera_e2e_closure_plan/camera_e2e_closure_plan.json and camera_e2e_closure_plan.csv for the measured-data and solver-batch worklist required to move beyond research mode.",
            "consumer_bundle": "Load camera_e2e_consumer_bundle/camera_e2e_consumer_bundle.json for per-sensor table paths, join keys, and row-count contract.",
        },
        "gate_counts": {
            "production_lut_gate": dict(Counter(str(row.get("production_lut_gate", "")) for row in sensor_rows)),
            "cra_mismatch_gate": dict(Counter(str(row.get("cra_mismatch_gate", "")) for row in sensor_rows)),
            "color_matrix_gate": dict(Counter(str(row.get("color_matrix_gate", "")) for row in sensor_rows)),
            "cfa_assumption_gate": dict(Counter(str(row.get("cfa_assumption_gate", "")) for row in sensor_rows)),
        },
        "outputs": {
            "json": repo_rel(manifest_json),
            "artifacts_csv": repo_rel(artifacts_csv),
            "sensors_csv": repo_rel(sensors_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(artifacts_csv, artifact_rows, ARTIFACT_COLUMNS)
    write_csv(sensors_csv, sensor_rows, SENSOR_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, artifact_rows, sensor_rows)
    update_package(package_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    manifest = export_handoff(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "validation": manifest["validation"],
                "sensor_count": manifest["sensor_count"],
                "artifact_count": manifest["artifact_count"],
                "product_ready_count": manifest["product_ready_count"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not manifest["validation"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
