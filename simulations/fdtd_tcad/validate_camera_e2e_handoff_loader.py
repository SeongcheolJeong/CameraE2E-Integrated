#!/usr/bin/env python3
"""Validate that the CameraE2E handoff can be loaded as a consumer package.

The handoff manifest proves that the expected files were exported. This loader
validator goes one step further: it reads the manifest, loads the referenced
artifacts, and checks the join keys that a CameraE2E runtime would depend on.

It does not upgrade research/proxy data to product accuracy. Product readiness
still depends on measured stack, material, CRA, electrical, readout, and module
calibration gates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_HANDOFF_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_handoff_manifest"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_handoff_loader_validation"

REQUIRED_ARTIFACT_IDS = {
    "handoff_sensor_models",
    "coverage_matrix",
    "consumer_bundle",
    "flat_sensor_bundle",
    "flat_sensor_index",
    "import_contract",
    "import_contract_by_sensor",
    "import_contract_by_requirement",
    "import_contract_checks",
    "canonical_payload",
    "canonical_payload_by_sensor",
    "canonical_payload_checks",
    "flat_sensor_query",
    "flat_sensor_query_csv",
    "flat_sensor_query_summary",
    "flat_sensor_product_query_probe",
    "analysis_report",
    "analysis_by_sensor",
    "analysis_by_channel",
    "analysis_actions",
    "use_scope_summary",
    "use_scope_by_sensor",
    "use_scope_by_domain",
    "use_scope_next_actions",
    "runtime_lut",
    "runtime_crosstalk_kernel",
    "runtime_npz",
    "color_spectral_response",
    "color_matrix_seed",
    "material_nk_lut",
    "cfa_provenance",
    "cfa_provenance_by_sensor",
    "cfa_db_tables",
    "cfa_db_by_sensor",
    "cfa_db_transmission_lut",
    "capability_profile",
    "capability_profile_by_sensor",
    "lut_trust_assessment",
    "lut_trust_by_sensor",
    "lut_trust_by_domain",
    "lut_trust_by_requirement",
    "prior_seed_models",
    "electrical_noise_lut",
    "readout_gain_lut",
    "binning_remosaic_lut",
    "module_coupling_lut",
    "quantitative_execution_plan",
    "quantitative_point_queue",
    "resource_limited_batch_plan",
    "quantitative_coverage",
    "closure_plan",
    "closure_plan_csv",
    "closure_checks",
    "readiness_report",
    "mesh_confidence",
    "field_execution_pack",
    "field_execution_jobs_csv",
    "field_execution_scripts_csv",
    "field_center_spectral_anchor_script",
    "field_green_cra_anchor_script",
    "field_failed_or_stale_rerun_script",
    "field_all_quantitative_script",
    "field_refresh_after_solver_script",
    "crosstalk_support_audit",
    "crosstalk_support_by_sensor",
    "crosstalk_support_pilots",
    "crosstalk_product_candidates",
    "crosstalk_batch_priority",
    "crosstalk_batch_priority_csv",
    "crosstalk_execution_pack",
    "crosstalk_execution_jobs_csv",
    "crosstalk_execution_scripts_csv",
    "crosstalk_local_probe_evidence",
    "crosstalk_product_primary_hpc_script",
    "crosstalk_support_discovery_local_script",
    "crosstalk_support_discovery_batch_script",
    "crosstalk_refresh_after_solver_script",
    "product_closure_summary",
    "product_closure_by_sensor",
    "product_closure_by_domain",
    "product_closure_checks",
    "usage_policy",
    "usage_policy_by_sensor",
    "usage_policy_by_domain",
    "usage_policy_runtime_filters",
    "usage_policy_checks",
    "adapter_examples",
    "adapter_examples_by_sensor",
    "adapter_examples_checks",
    "adapter_smoke",
    "adapter_smoke_by_sensor",
    "adapter_smoke_checks",
    "objective_trace",
    "objective_trace_by_requirement",
    "objective_trace_by_requirement_summary",
    "objective_trace_by_sensor",
    "objective_trace_checks",
    "source_integrity_matrix",
    "source_integrity_by_sensor",
    "sensor_deliverable_summary",
    "sensor_deliverable_summary_json",
    "research_probe_all_sensors",
}

REQUIRED_CONSUMER_TRUST_SOURCE_TABLES = {
    "lut_trust_assessment",
    "lut_trust_by_sensor",
    "lut_trust_by_domain",
    "lut_trust_by_requirement",
}

REQUIRED_CONSUMER_CROSSTALK_SOURCE_TABLES = {
    "crosstalk_support_by_sensor",
    "crosstalk_product_candidates",
}

REQUIRED_CONSUMER_CFA_SOURCE_TABLES = {
    "cfa_db_by_sensor",
    "cfa_db_transmission_lut",
    "cfa_db_tables",
}

REQUIRED_CONSUMER_SOURCE_INTEGRITY_TABLES = {
    "source_integrity_matrix",
    "source_integrity_by_sensor",
}

REQUIRED_CONSUMER_TRUST_JOIN_KEYS = {
    "lut_trust",
    "lut_trust_domain",
    "lut_trust_requirement",
}

REQUIRED_CONSUMER_CROSSTALK_JOIN_KEYS = {
    "crosstalk_support",
    "crosstalk_product_candidates",
}

REQUIRED_CONSUMER_CFA_JOIN_KEYS = {
    "cfa_db_by_sensor",
    "cfa_db_transmission_lut",
}

REQUIRED_CONSUMER_SOURCE_INTEGRITY_JOIN_KEYS = {
    "source_integrity",
    "objective_fulfillment",
}

REQUIRED_CONSUMER_TRUST_FIELDS = {
    "trust_class",
    "camera_e2e_allowed_use",
    "research_usability_score_0_100",
    "evidence_confidence_score_0_100",
    "product_calibration_score_0_100",
    "field_mesh_pass_fraction",
    "crosstalk_mesh_pass_fraction",
    "recommended_next_action",
}

ARTIFACT_CHECK_COLUMNS = [
    "artifact_id",
    "role",
    "path",
    "exists",
    "declared_schema",
    "actual_schema",
    "declared_row_count",
    "actual_row_count",
    "loader_gate",
    "issue",
]

SENSOR_CHECK_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "sensor_model_schema",
    "flat_sensor_model_schema",
    "flat_sensor_total_embedded_row_count",
    "flat_objective_fulfillment_row_count",
    "runtime_row_count",
    "runtime_id_count",
    "runtime_ids_with_kernel_count",
    "missing_kernel_runtime_id_count",
    "kernel_row_count",
    "kernel_sum_gate",
    "spectral_row_count",
    "material_row_count",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "cfa_db_row_count",
    "cfa_db_transmission_row_count",
    "capability_overall_use_scope",
    "lut_trust_class",
    "lut_trust_evidence_score_0_100",
    "camera_e2e_use_scope",
    "use_scope_product_gate",
    "coverage_requirement_count",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "source_integrity_requirement_count",
    "source_integrity_gate_counts",
    "source_integrity_uncertainty_product_gate_counts",
    "quantitative_plan_row_count",
    "quantitative_queue_row_count",
    "quantitative_coverage_row_count",
    "resource_limited_batch_row_count",
    "electrical_row_count",
    "readout_row_count",
    "binning_row_count",
    "color_channels",
    "color_matrix_applicability",
    "color_matrix_gate",
    "prior_model_schema",
    "prior_gate",
    "module_field_row_count",
    "probe_summary_row_count",
    "cra_mismatch_gate",
    "production_lut_gate",
    "product_ready",
    "loader_gate",
    "issues",
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


def safe_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def json_schema(path: Path) -> str:
    return str(read_json(path).get("schema", ""))


def actual_row_count(path: Path) -> int | str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return len(read_csv_rows(path))
    if suffix == ".json":
        payload = read_json(path)
        for key in (
            "row_count",
            "runtime_row_count",
            "kernel_row_count",
            "sensor_count",
            "spectral_row_count",
            "matrix_row_count",
            "field_row_count",
            "summary_row_count",
            "artifact_count",
            "plan_row_count",
        ):
            if key in payload:
                return payload.get(key, "")
    if suffix == ".npz" and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return len(archive.namelist())
    return ""


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
    return {row.get(key, "") for row in rows if row.get(key, "")}


def build_artifact_checks(artifact_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    artifact_ids = {row.get("artifact_id", "") for row in artifact_rows}
    for required_id in sorted(REQUIRED_ARTIFACT_IDS - artifact_ids):
        issues.append({"severity": "error", "code": "required_artifact_not_listed", "artifact_id": required_id})
        checks.append(
            {
                "artifact_id": required_id,
                "role": "",
                "path": "",
                "exists": False,
                "declared_schema": "",
                "actual_schema": "",
                "declared_row_count": "",
                "actual_row_count": "",
                "loader_gate": "FAIL",
                "issue": "required artifact missing from handoff artifact CSV",
            }
        )

    for row in artifact_rows:
        path = abs_from_repo(row.get("path", ""))
        exists = path.exists()
        declared_schema = row.get("schema", "")
        declared_row_count = row.get("row_count", "")
        actual_schema_value = json_schema(path) if path.suffix.lower() == ".json" and exists else ""
        actual_count = actual_row_count(path)
        issue_parts: list[str] = []
        gate = "PASS"

        if not exists:
            gate = "FAIL"
            issue_parts.append("path does not exist")
            issues.append({"severity": "error", "code": "artifact_path_missing", "artifact_id": row.get("artifact_id"), "path": row.get("path")})
        if exists and path.suffix.lower() == ".npz" and not zipfile.is_zipfile(path):
            gate = "FAIL"
            issue_parts.append("npz is not a valid zip archive")
            issues.append({"severity": "error", "code": "artifact_npz_invalid", "artifact_id": row.get("artifact_id"), "path": row.get("path")})
        if declared_schema and actual_schema_value and declared_schema != actual_schema_value:
            gate = "FAIL"
            issue_parts.append(f"schema mismatch {declared_schema} != {actual_schema_value}")
            issues.append({"severity": "error", "code": "artifact_schema_mismatch", "artifact_id": row.get("artifact_id")})
        declared_count_int = safe_int(declared_row_count)
        actual_count_int = safe_int(actual_count)
        if declared_count_int is not None and actual_count_int is not None and declared_count_int != actual_count_int:
            gate = "FAIL"
            issue_parts.append(f"row count mismatch {declared_count_int} != {actual_count_int}")
            issues.append({"severity": "error", "code": "artifact_row_count_mismatch", "artifact_id": row.get("artifact_id")})

        checks.append(
            {
                "artifact_id": row.get("artifact_id", ""),
                "role": row.get("role", ""),
                "path": row.get("path", ""),
                "exists": exists,
                "declared_schema": declared_schema,
                "actual_schema": actual_schema_value,
                "declared_row_count": declared_row_count,
                "actual_row_count": actual_count,
                "loader_gate": gate,
                "issue": "; ".join(issue_parts),
            }
        )
    return checks, issues


def build_runtime_kernel_indexes(
    runtime_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
) -> dict[str, Any]:
    runtime_by_slug = group_rows(runtime_rows, "slug")
    kernel_by_slug = group_rows(kernel_rows, "slug")
    runtime_ids_by_slug: dict[str, set[str]] = {
        slug: unique_values(rows, "runtime_id") for slug, rows in runtime_by_slug.items()
    }
    kernel_runtime_ids_by_slug: dict[str, set[str]] = {
        slug: unique_values(rows, "runtime_id") for slug, rows in kernel_by_slug.items()
    }
    kernel_sums: dict[str, float] = defaultdict(float)
    for row in kernel_rows:
        runtime_id = row.get("runtime_id", "")
        value = safe_float(row.get("response_fraction"))
        if runtime_id and value is not None:
            kernel_sums[runtime_id] += value
    kernel_sum_bad = {
        runtime_id: total
        for runtime_id, total in kernel_sums.items()
        if total < 0.98 or total > 1.02
    }
    return {
        "runtime_by_slug": runtime_by_slug,
        "kernel_by_slug": kernel_by_slug,
        "runtime_ids_by_slug": runtime_ids_by_slug,
        "kernel_runtime_ids_by_slug": kernel_runtime_ids_by_slug,
        "kernel_sum_bad": kernel_sum_bad,
    }


def build_sensor_checks(
    handoff_sensor_rows: list[dict[str, str]],
    *,
    coverage_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
    spectral_rows: list[dict[str, str]],
    material_rows: list[dict[str, str]],
    cfa_provenance_rows: list[dict[str, str]],
    cfa_db_rows: list[dict[str, str]],
    cfa_db_transmission_rows: list[dict[str, str]],
    capability_rows: list[dict[str, str]],
    trust_sensor_rows: list[dict[str, str]],
    use_scope_rows: list[dict[str, str]],
    electrical_rows: list[dict[str, str]],
    readout_rows: list[dict[str, str]],
    binning_rows: list[dict[str, str]],
    color_matrix_rows: list[dict[str, str]],
    prior_summary_rows: list[dict[str, str]],
    module_field_rows: list[dict[str, str]],
    module_summary_rows: list[dict[str, str]],
    probe_summary_rows: list[dict[str, str]],
    quantitative_plan_rows: list[dict[str, str]],
    quantitative_queue_rows: list[dict[str, str]],
    quantitative_coverage_rows: list[dict[str, str]],
    resource_limited_rows: list[dict[str, str]],
    source_integrity_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    runtime_kernel = build_runtime_kernel_indexes(runtime_rows, kernel_rows)
    runtime_by_slug = runtime_kernel["runtime_by_slug"]
    kernel_by_slug = runtime_kernel["kernel_by_slug"]
    runtime_ids_by_slug = runtime_kernel["runtime_ids_by_slug"]
    kernel_runtime_ids_by_slug = runtime_kernel["kernel_runtime_ids_by_slug"]
    kernel_sum_bad = runtime_kernel["kernel_sum_bad"]

    spectral_by_slug = group_rows(spectral_rows, "slug")
    material_by_slug = group_rows(material_rows, "slug")
    cfa_by_slug = index_by(cfa_provenance_rows, "slug")
    cfa_db_by_slug = index_by(cfa_db_rows, "slug")
    cfa_db_transmission_by_slug = group_rows(cfa_db_transmission_rows, "slug")
    capability_by_slug = index_by(capability_rows, "slug")
    trust_by_slug = index_by(trust_sensor_rows, "slug")
    use_scope_by_slug = index_by(use_scope_rows, "slug")
    coverage_by_slug = group_rows(coverage_rows, "slug")
    coverage_requirement_ids = unique_values(coverage_rows, "requirement_id")
    source_integrity_by_slug = group_rows(source_integrity_rows, "slug")
    source_integrity_requirement_ids = unique_values(source_integrity_rows, "requirement_id")
    electrical_by_slug = group_rows(electrical_rows, "slug")
    readout_by_slug = group_rows(readout_rows, "slug")
    binning_by_slug = group_rows(binning_rows, "slug")
    matrix_by_slug = index_by(color_matrix_rows, "slug")
    prior_by_slug = index_by(prior_summary_rows, "slug")
    module_field_by_slug = group_rows(module_field_rows, "slug")
    module_summary_by_slug = index_by(module_summary_rows, "slug")
    probe_summary_by_slug = group_rows(probe_summary_rows, "slug")
    quantitative_plan_by_slug = group_rows(quantitative_plan_rows, "slug")
    quantitative_queue_by_slug = group_rows(quantitative_queue_rows, "slug")
    quantitative_coverage_by_slug = group_rows(quantitative_coverage_rows, "slug")
    resource_limited_by_slug = group_rows(resource_limited_rows, "slug")

    for row in handoff_sensor_rows:
        slug = row.get("slug", "")
        row_issues: list[str] = []
        model_path = abs_from_repo(row.get("model_json", ""))
        model_schema = json_schema(model_path)
        if model_schema != "camera_e2e_sensor_model_v1":
            row_issues.append("sensor model schema missing or invalid")
            issues.append({"severity": "error", "code": "sensor_model_schema_invalid", "slug": slug, "path": row.get("model_json", "")})

        flat_path = abs_from_repo(row.get("flat_sensor_json", ""))
        flat_payload = read_json(flat_path)
        flat_schema = str(flat_payload.get("schema", ""))
        flat_total_rows = safe_int(flat_payload.get("total_embedded_row_count")) or 0
        flat_counts = flat_payload.get("row_counts", {}) if isinstance(flat_payload.get("row_counts"), dict) else {}
        if flat_schema != "camera_e2e_flat_sensor_model_v1":
            row_issues.append("flat sensor model schema missing or invalid")
            issues.append({"severity": "error", "code": "flat_sensor_model_schema_invalid", "slug": slug, "path": row.get("flat_sensor_json", "")})
        else:
            if str(flat_payload.get("sensor", {}).get("slug", "")) != slug:
                row_issues.append("flat sensor slug mismatch")
                issues.append({"severity": "error", "code": "flat_sensor_slug_mismatch", "slug": slug, "path": row.get("flat_sensor_json", "")})
            for domain_key in ("optical_color", "pixel_electrical", "readout_raw", "module_coupling"):
                if not isinstance(flat_payload.get(domain_key), dict) or not flat_payload.get(domain_key):
                    row_issues.append(f"flat sensor {domain_key} domain missing")
                    issues.append({"severity": "error", "code": "flat_sensor_domain_missing", "slug": slug, "domain": domain_key})
            for count_key in ("runtime", "kernel", "spectral", "material", "electrical", "readout", "binning", "module_field", "coverage", "source_integrity", "objective_fulfillment"):
                if (safe_int(flat_counts.get(count_key)) or 0) <= 0:
                    row_issues.append(f"flat sensor {count_key} rows missing")
                    issues.append({"severity": "error", "code": "flat_sensor_row_count_empty", "slug": slug, "row_count_key": count_key})
            if boolish(flat_payload.get("gates", {}).get("product_ready")):
                row_issues.append("flat sensor unexpectedly product-ready")
                issues.append({"severity": "error", "code": "flat_sensor_unexpectedly_product_ready", "slug": slug})

        runtime_rows_for_slug = runtime_by_slug.get(slug, [])
        kernel_rows_for_slug = kernel_by_slug.get(slug, [])
        runtime_ids = runtime_ids_by_slug.get(slug, set())
        kernel_runtime_ids = kernel_runtime_ids_by_slug.get(slug, set())
        missing_kernel_ids = runtime_ids - kernel_runtime_ids
        bad_kernel_sum_ids = sorted(runtime_id for runtime_id in runtime_ids if runtime_id in kernel_sum_bad)
        kernel_sum_gate = "PASS" if not missing_kernel_ids and not bad_kernel_sum_ids and runtime_ids else "FAIL"

        if not runtime_rows_for_slug:
            row_issues.append("runtime rows missing")
            issues.append({"severity": "error", "code": "sensor_runtime_rows_missing", "slug": slug})
        if not kernel_rows_for_slug:
            row_issues.append("kernel rows missing")
            issues.append({"severity": "error", "code": "sensor_kernel_rows_missing", "slug": slug})
        if missing_kernel_ids:
            row_issues.append(f"{len(missing_kernel_ids)} runtime ids have no kernel rows")
            issues.append({"severity": "error", "code": "sensor_runtime_kernel_join_missing", "slug": slug, "count": len(missing_kernel_ids)})
        if bad_kernel_sum_ids:
            row_issues.append(f"{len(bad_kernel_sum_ids)} kernels are not normalized")
            issues.append({"severity": "error", "code": "sensor_kernel_sum_not_normalized", "slug": slug, "count": len(bad_kernel_sum_ids)})

        spectral_rows_for_slug = spectral_by_slug.get(slug, [])
        if not spectral_rows_for_slug:
            row_issues.append("spectral response rows missing")
            issues.append({"severity": "error", "code": "sensor_spectral_rows_missing", "slug": slug})

        material_rows_for_slug = material_by_slug.get(slug, [])
        if not material_rows_for_slug:
            row_issues.append("material n,k rows missing")
            issues.append({"severity": "error", "code": "sensor_material_rows_missing", "slug": slug})

        cfa = cfa_by_slug.get(slug, {})
        if not cfa:
            row_issues.append("CFA provenance row missing")
            issues.append({"severity": "error", "code": "sensor_cfa_provenance_missing", "slug": slug})
        elif cfa.get("cfa_assumption_gate") not in {"PASS", "CHECK", "MISSING", "FAIL"}:
            row_issues.append("CFA provenance gate invalid")
            issues.append({"severity": "error", "code": "sensor_cfa_provenance_gate_invalid", "slug": slug, "gate": cfa.get("cfa_assumption_gate", "")})

        cfa_db = cfa_db_by_slug.get(slug, {})
        cfa_db_transmission_for_slug = cfa_db_transmission_by_slug.get(slug, [])
        if not cfa_db:
            row_issues.append("CFA DB sensor row missing")
            issues.append({"severity": "error", "code": "sensor_cfa_db_row_missing", "slug": slug})
        if not cfa_db_transmission_for_slug:
            row_issues.append("CFA DB transmission rows missing")
            issues.append({"severity": "error", "code": "sensor_cfa_db_transmission_rows_missing", "slug": slug})

        capability = capability_by_slug.get(slug, {})
        if not capability:
            row_issues.append("capability profile row missing")
            issues.append({"severity": "error", "code": "sensor_capability_profile_missing", "slug": slug})

        trust = trust_by_slug.get(slug, {})
        if not trust:
            row_issues.append("LUT trust row missing")
            issues.append({"severity": "error", "code": "sensor_lut_trust_row_missing", "slug": slug})

        use_scope = use_scope_by_slug.get(slug, {})
        if not use_scope:
            row_issues.append("use-scope row missing")
            issues.append({"severity": "error", "code": "sensor_use_scope_row_missing", "slug": slug})
        elif boolish(use_scope.get("product_ready")) or str(use_scope.get("product_gate", "")).upper() == "PASS":
            row_issues.append("use-scope unexpectedly product-ready")
            issues.append({"severity": "error", "code": "sensor_use_scope_unexpectedly_product_ready", "slug": slug})

        coverage_rows_for_slug = coverage_by_slug.get(slug, [])
        coverage_ids_for_slug = unique_values(coverage_rows_for_slug, "requirement_id")
        missing_coverage_ids = coverage_requirement_ids - coverage_ids_for_slug
        if not coverage_rows_for_slug:
            row_issues.append("coverage requirement rows missing")
            issues.append({"severity": "error", "code": "sensor_coverage_rows_missing", "slug": slug})
        if missing_coverage_ids:
            row_issues.append(f"{len(missing_coverage_ids)} coverage requirement ids missing")
            issues.append({"severity": "error", "code": "sensor_coverage_requirement_ids_missing", "slug": slug, "count": len(missing_coverage_ids)})
        flat_objective_count = safe_int(flat_counts.get("objective_fulfillment")) if isinstance(flat_counts, dict) else 0
        if flat_objective_count != len(coverage_ids_for_slug):
            row_issues.append("flat objective_fulfillment row count mismatch")
            issues.append(
                {
                    "severity": "error",
                    "code": "flat_objective_fulfillment_count_mismatch",
                    "slug": slug,
                    "expected": len(coverage_ids_for_slug),
                    "actual": flat_objective_count,
                }
            )

        source_integrity_for_slug = source_integrity_by_slug.get(slug, [])
        source_integrity_ids_for_slug = unique_values(source_integrity_for_slug, "requirement_id")
        missing_source_integrity_ids = coverage_requirement_ids - source_integrity_ids_for_slug
        extra_source_integrity_ids = source_integrity_ids_for_slug - coverage_requirement_ids
        source_integrity_missing_uncertainty = [
            item
            for item in source_integrity_for_slug
            if item.get("research_gate") != "N/A"
            and (not str(item.get("primary_uncertainty_min", "")).strip() or not str(item.get("primary_uncertainty_max", "")).strip())
        ]
        if not source_integrity_for_slug:
            row_issues.append("source-integrity requirement rows missing")
            issues.append({"severity": "error", "code": "sensor_source_integrity_rows_missing", "slug": slug})
        if missing_source_integrity_ids:
            row_issues.append(f"{len(missing_source_integrity_ids)} source-integrity requirement ids missing")
            issues.append({"severity": "error", "code": "sensor_source_integrity_requirement_ids_missing", "slug": slug, "count": len(missing_source_integrity_ids)})
        if extra_source_integrity_ids:
            row_issues.append(f"{len(extra_source_integrity_ids)} source-integrity extra requirement ids")
            issues.append({"severity": "error", "code": "sensor_source_integrity_requirement_ids_extra", "slug": slug, "count": len(extra_source_integrity_ids)})
        if source_integrity_missing_uncertainty:
            row_issues.append(f"{len(source_integrity_missing_uncertainty)} source-integrity rows missing uncertainty")
            issues.append({"severity": "error", "code": "sensor_source_integrity_uncertainty_missing", "slug": slug, "count": len(source_integrity_missing_uncertainty)})
        if any(row.get("source_integrity_gate") == "FAIL" for row in source_integrity_for_slug):
            row_issues.append("source-integrity gate has FAIL rows")
            issues.append({"severity": "error", "code": "sensor_source_integrity_gate_fail", "slug": slug})

        quantitative_plan_for_slug = quantitative_plan_by_slug.get(slug, [])
        quantitative_queue_for_slug = quantitative_queue_by_slug.get(slug, [])
        quantitative_coverage_for_slug = quantitative_coverage_by_slug.get(slug, [])
        resource_limited_for_slug = resource_limited_by_slug.get(slug, [])
        if len(quantitative_plan_for_slug) < 2:
            row_issues.append("quantitative field/crosstalk execution plan rows missing")
            issues.append({"severity": "error", "code": "sensor_quantitative_plan_rows_missing", "slug": slug, "count": len(quantitative_plan_for_slug)})
        if not quantitative_queue_for_slug:
            row_issues.append("quantitative point queue rows missing")
            issues.append({"severity": "error", "code": "sensor_quantitative_queue_rows_missing", "slug": slug})
        if len(quantitative_coverage_for_slug) < 2:
            row_issues.append("quantitative field/crosstalk coverage rows missing")
            issues.append({"severity": "error", "code": "sensor_quantitative_coverage_rows_missing", "slug": slug, "count": len(quantitative_coverage_for_slug)})
        declared_queue_count = safe_int(row.get("quantitative_queue_row_count", ""))
        if declared_queue_count is not None and declared_queue_count != len(quantitative_queue_for_slug):
            row_issues.append("quantitative queue count mismatch")
            issues.append({"severity": "error", "code": "sensor_quantitative_queue_count_mismatch", "slug": slug})
        declared_resource_count = safe_int(row.get("resource_limited_batch_row_count", ""))
        if declared_resource_count is not None and declared_resource_count != len(resource_limited_for_slug):
            row_issues.append("resource-limited batch count mismatch")
            issues.append({"severity": "error", "code": "sensor_resource_limited_count_mismatch", "slug": slug})
        mesh_resource_count = safe_int(row.get("mesh_crosstalk_resource_limited_points", "")) or 0
        if mesh_resource_count > 0 and not resource_limited_for_slug:
            row_issues.append("mesh confidence reports resource-limited crosstalk but batch plan rows are missing")
            issues.append({"severity": "error", "code": "sensor_resource_limited_batch_rows_missing", "slug": slug, "mesh_resource_count": mesh_resource_count})

        electrical_rows_for_slug = electrical_by_slug.get(slug, [])
        if not electrical_rows_for_slug:
            row_issues.append("electrical/noise rows missing")
            issues.append({"severity": "error", "code": "sensor_electrical_rows_missing", "slug": slug})

        readout_rows_for_slug = readout_by_slug.get(slug, [])
        if not readout_rows_for_slug:
            row_issues.append("readout/gain rows missing")
            issues.append({"severity": "error", "code": "sensor_readout_rows_missing", "slug": slug})

        binning_rows_for_slug = binning_by_slug.get(slug, [])
        if not binning_rows_for_slug:
            row_issues.append("binning/remosaic rows missing")
            issues.append({"severity": "error", "code": "sensor_binning_rows_missing", "slug": slug})

        matrix = matrix_by_slug.get(slug, {})
        matrix_applicability = matrix.get("applicability", "")
        matrix_gate = matrix.get("gate", "")
        if not matrix:
            row_issues.append("color matrix seed row missing")
            issues.append({"severity": "warning", "code": "sensor_color_matrix_row_missing", "slug": slug})
        elif matrix_gate not in {"PASS", "CHECK", "MISSING"}:
            row_issues.append("color matrix gate invalid")
            issues.append({"severity": "warning", "code": "sensor_color_matrix_gate_invalid", "slug": slug, "gate": matrix_gate})

        prior = prior_by_slug.get(slug, {})
        prior_path = abs_from_repo(prior.get("model_json", ""))
        prior_schema = json_schema(prior_path)
        if prior_schema != "camera_e2e_prior_seed_model_v1":
            row_issues.append("prior model schema missing or invalid")
            issues.append({"severity": "error", "code": "sensor_prior_schema_invalid", "slug": slug, "path": prior.get("model_json", "")})

        module_rows_for_slug = module_field_by_slug.get(slug, [])
        module_summary = module_summary_by_slug.get(slug, {})
        if not module_rows_for_slug:
            row_issues.append("module coupling field rows missing")
            issues.append({"severity": "error", "code": "sensor_module_rows_missing", "slug": slug})
        declared_module_count = safe_int(module_summary.get("field_row_count", ""))
        if declared_module_count is not None and declared_module_count != len(module_rows_for_slug):
            row_issues.append("module coupling summary count mismatch")
            issues.append({"severity": "error", "code": "sensor_module_summary_count_mismatch", "slug": slug})

        probe_rows_for_slug = probe_summary_by_slug.get(slug, [])
        if not probe_rows_for_slug:
            row_issues.append("all-sensor probe summary rows missing")
            issues.append({"severity": "warning", "code": "sensor_probe_summary_missing", "slug": slug})

        runtime_channels = unique_values(runtime_rows_for_slug, "color_channel")
        spectral_channels = unique_values(spectral_rows_for_slug, "color_channel")
        missing_spectral_channels = runtime_channels - spectral_channels
        if missing_spectral_channels:
            row_issues.append("runtime color channels missing spectral rows: " + ",".join(sorted(missing_spectral_channels)))
            issues.append({"severity": "error", "code": "sensor_runtime_spectral_channel_join_missing", "slug": slug})

        loader_gate = "FAIL" if any(
            text
            for text in row_issues
            if not text.startswith("color matrix") and not text.startswith("all-sensor probe")
        ) else "PASS"
        checks.append(
            {
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "sensor_model_schema": model_schema,
                "flat_sensor_model_schema": flat_schema,
                "flat_sensor_total_embedded_row_count": flat_total_rows,
                "flat_objective_fulfillment_row_count": safe_int(flat_counts.get("objective_fulfillment")) if isinstance(flat_counts, dict) else 0,
                "runtime_row_count": len(runtime_rows_for_slug),
                "runtime_id_count": len(runtime_ids),
                "runtime_ids_with_kernel_count": len(runtime_ids & kernel_runtime_ids),
                "missing_kernel_runtime_id_count": len(missing_kernel_ids),
                "kernel_row_count": len(kernel_rows_for_slug),
                "kernel_sum_gate": kernel_sum_gate,
                "spectral_row_count": len(spectral_rows_for_slug),
                "material_row_count": len(material_rows_for_slug),
                "cfa_provenance_class": cfa.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa.get("cfa_assumption_gate", ""),
                "cfa_db_row_count": 1 if cfa_db else 0,
                "cfa_db_transmission_row_count": len(cfa_db_transmission_for_slug),
                "capability_overall_use_scope": capability.get("overall_use_scope", ""),
                "lut_trust_class": trust.get("trust_class", ""),
                "lut_trust_evidence_score_0_100": trust.get("evidence_confidence_score_0_100", ""),
                "camera_e2e_use_scope": use_scope.get("camera_e2e_use_scope", ""),
                "use_scope_product_gate": use_scope.get("product_gate", ""),
                "coverage_requirement_count": len(coverage_ids_for_slug),
                "coverage_research_gate_counts": json.dumps(dict(Counter(row.get("research_gate", "") for row in coverage_rows_for_slug)), sort_keys=True),
                "coverage_product_gate_counts": json.dumps(dict(Counter(row.get("product_gate", "") for row in coverage_rows_for_slug)), sort_keys=True),
                "source_integrity_requirement_count": len(source_integrity_ids_for_slug),
                "source_integrity_gate_counts": json.dumps(dict(Counter(row.get("source_integrity_gate", "") for row in source_integrity_for_slug)), sort_keys=True),
                "source_integrity_uncertainty_product_gate_counts": json.dumps(dict(Counter(row.get("uncertainty_product_gate", "") for row in source_integrity_for_slug)), sort_keys=True),
                "quantitative_plan_row_count": len(quantitative_plan_for_slug),
                "quantitative_queue_row_count": len(quantitative_queue_for_slug),
                "quantitative_coverage_row_count": len(quantitative_coverage_for_slug),
                "resource_limited_batch_row_count": len(resource_limited_for_slug),
                "electrical_row_count": len(electrical_rows_for_slug),
                "readout_row_count": len(readout_rows_for_slug),
                "binning_row_count": len(binning_rows_for_slug),
                "color_channels": ";".join(sorted(spectral_channels)),
                "color_matrix_applicability": matrix_applicability,
                "color_matrix_gate": matrix_gate,
                "prior_model_schema": prior_schema,
                "prior_gate": prior.get("prior_gate", ""),
                "module_field_row_count": len(module_rows_for_slug),
                "probe_summary_row_count": len(probe_rows_for_slug),
                "cra_mismatch_gate": row.get("cra_mismatch_gate", ""),
                "production_lut_gate": row.get("production_lut_gate", ""),
                "product_ready": row.get("product_ready", ""),
                "loader_gate": loader_gate,
                "issues": "; ".join(row_issues),
            }
        )
    return checks, issues


def validate_consumer_bundle(bundle_path: Path, handoff_sensor_rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    bundle = read_json(bundle_path)
    if bundle.get("schema") != "camera_e2e_consumer_bundle_v1":
        issues.append({"severity": "error", "code": "consumer_bundle_schema_invalid", "path": repo_rel(bundle_path)})
        return {}, issues
    requirement_count = safe_int(bundle.get("requirement_count")) or 0
    requirement_load_map = bundle.get("requirement_load_map", [])
    if requirement_count <= 0:
        issues.append({"severity": "error", "code": "consumer_requirement_count_missing", "path": repo_rel(bundle_path)})
    if not isinstance(requirement_load_map, list) or len(requirement_load_map) != requirement_count:
        issues.append(
            {
                "severity": "error",
                "code": "consumer_requirement_load_map_invalid",
                "path": repo_rel(bundle_path),
                "requirement_count": requirement_count,
                "load_map_count": len(requirement_load_map) if isinstance(requirement_load_map, list) else "",
            }
        )
    bundle_source_tables = bundle.get("source_tables", {})
    if not isinstance(bundle_source_tables, dict) or not bundle_source_tables.get("coverage_matrix") or not bundle_source_tables.get("source_integrity_matrix"):
        issues.append({"severity": "error", "code": "consumer_bundle_requirement_source_tables_missing", "path": repo_rel(bundle_path)})
    bundle_join_keys = bundle.get("join_keys", {})
    if not isinstance(bundle_join_keys, dict) or not bundle_join_keys.get("coverage") or not bundle_join_keys.get("source_integrity") or not bundle_join_keys.get("objective_fulfillment"):
        issues.append({"severity": "error", "code": "consumer_bundle_requirement_join_keys_missing", "path": repo_rel(bundle_path)})
    declared_paths = [str(path) for path in bundle.get("sensor_manifest_json_files", [])]
    expected_slugs = {row.get("slug", "") for row in handoff_sensor_rows if row.get("slug", "")}
    loaded_slugs: set[str] = set()
    for rel_path in declared_paths:
        sensor_path = abs_from_repo(rel_path)
        sensor_manifest = read_json(sensor_path)
        if sensor_manifest.get("schema") != "camera_e2e_consumer_sensor_manifest_v1":
            issues.append({"severity": "error", "code": "consumer_sensor_manifest_schema_invalid", "path": rel_path})
            continue
        slug = str(sensor_manifest.get("sensor", {}).get("slug", ""))
        loaded_slugs.add(slug)
        if not slug:
            issues.append({"severity": "error", "code": "consumer_sensor_manifest_slug_missing", "path": rel_path})
        if boolish(sensor_manifest.get("gates", {}).get("product_ready")):
            issues.append({"severity": "error", "code": "consumer_sensor_unexpectedly_product_ready", "slug": slug})
        source_tables = sensor_manifest.get("source_tables", {})
        if not isinstance(source_tables, dict):
            source_tables = {}
            issues.append({"severity": "error", "code": "consumer_source_tables_invalid", "slug": slug})
        for table_name in sorted(REQUIRED_CONSUMER_TRUST_SOURCE_TABLES):
            if not source_tables.get(table_name):
                issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_source_missing", "slug": slug, "table": table_name})
        for table_name in sorted(REQUIRED_CONSUMER_CROSSTALK_SOURCE_TABLES):
            if not source_tables.get(table_name):
                issues.append({"severity": "error", "code": "consumer_sensor_crosstalk_support_source_missing", "slug": slug, "table": table_name})
        for table_name in sorted(REQUIRED_CONSUMER_CFA_SOURCE_TABLES):
            if not source_tables.get(table_name):
                issues.append({"severity": "error", "code": "consumer_sensor_cfa_db_source_missing", "slug": slug, "table": table_name})
        for table_name in sorted(REQUIRED_CONSUMER_SOURCE_INTEGRITY_TABLES):
            if not source_tables.get(table_name):
                issues.append({"severity": "error", "code": "consumer_sensor_source_integrity_source_missing", "slug": slug, "table": table_name})
        join_keys = sensor_manifest.get("join_keys", {})
        if not isinstance(join_keys, dict):
            join_keys = {}
            issues.append({"severity": "error", "code": "consumer_join_keys_invalid", "slug": slug})
        for join_key in sorted(REQUIRED_CONSUMER_TRUST_JOIN_KEYS):
            if not join_keys.get(join_key):
                issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_join_key_missing", "slug": slug, "join_key": join_key})
        for join_key in sorted(REQUIRED_CONSUMER_CROSSTALK_JOIN_KEYS):
            if not join_keys.get(join_key):
                issues.append({"severity": "error", "code": "consumer_sensor_crosstalk_support_join_key_missing", "slug": slug, "join_key": join_key})
        for join_key in sorted(REQUIRED_CONSUMER_CFA_JOIN_KEYS):
            if not join_keys.get(join_key):
                issues.append({"severity": "error", "code": "consumer_sensor_cfa_db_join_key_missing", "slug": slug, "join_key": join_key})
        for join_key in sorted(REQUIRED_CONSUMER_SOURCE_INTEGRITY_JOIN_KEYS):
            if not join_keys.get(join_key):
                issues.append({"severity": "error", "code": "consumer_sensor_source_integrity_join_key_missing", "slug": slug, "join_key": join_key})
        trust = sensor_manifest.get("lut_trust", {})
        if not isinstance(trust, dict) or not trust:
            issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_missing", "slug": slug})
            trust = {}
        for field_name in sorted(REQUIRED_CONSUMER_TRUST_FIELDS):
            if str(trust.get(field_name, "")).strip() == "":
                issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_field_missing", "slug": slug, "field": field_name})
        if not isinstance(trust.get("domain_rows", []), list) or not trust.get("domain_rows"):
            issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_domain_rows_missing", "slug": slug})
        if not isinstance(trust.get("requirement_rows", []), list) or not trust.get("requirement_rows"):
            issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_requirement_rows_missing", "slug": slug})
        if safe_float(trust.get("evidence_confidence_score_0_100")) is None:
            issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_evidence_score_invalid", "slug": slug})
        if safe_float(trust.get("product_calibration_score_0_100")) is None:
            issues.append({"severity": "error", "code": "consumer_sensor_lut_trust_product_score_invalid", "slug": slug})
        crosstalk_support = sensor_manifest.get("crosstalk_support", {})
        if not isinstance(crosstalk_support, dict) or not crosstalk_support:
            issues.append({"severity": "error", "code": "consumer_sensor_crosstalk_support_missing", "slug": slug})
            crosstalk_support = {}
        if not str(crosstalk_support.get("product_crosstalk_gate", "")).strip():
            issues.append({"severity": "error", "code": "consumer_sensor_crosstalk_support_gate_missing", "slug": slug})
        if not str(crosstalk_support.get("support_recommendation", "")).strip():
            issues.append({"severity": "error", "code": "consumer_sensor_crosstalk_support_recommendation_missing", "slug": slug})
        for table_name, table_path in source_tables.items():
            if table_name == "readiness_report":
                continue
            if not abs_from_repo(table_path).exists():
                issues.append({"severity": "error", "code": "consumer_source_table_missing", "slug": slug, "table": table_name, "path": table_path})
        row_counts = sensor_manifest.get("row_counts", {})
        coverage_count = safe_int(row_counts.get("coverage")) or 0
        source_integrity_count = safe_int(row_counts.get("source_integrity")) or 0
        objective_count = safe_int(row_counts.get("objective_fulfillment")) or 0
        for key in (
            "runtime",
            "kernel",
            "spectral",
            "material",
            "cfa_provenance",
            "cfa_db",
            "cfa_db_transmission",
            "capability",
            "quantitative_plan",
            "quantitative_queue",
            "electrical",
            "readout",
            "binning",
            "module_field",
            "coverage",
            "source_integrity",
            "objective_fulfillment",
            "probe_summary",
        ):
            if safe_int(row_counts.get(key)) <= 0:
                issues.append({"severity": "error", "code": "consumer_sensor_row_count_empty", "slug": slug, "row_count_key": key})
        if requirement_count and objective_count != requirement_count:
            issues.append(
                {
                    "severity": "error",
                    "code": "consumer_sensor_objective_count_mismatch",
                    "slug": slug,
                    "expected": requirement_count,
                    "actual": objective_count,
                }
            )
        if objective_count != coverage_count or objective_count != source_integrity_count:
            issues.append(
                {
                    "severity": "error",
                    "code": "consumer_sensor_objective_source_count_mismatch",
                    "slug": slug,
                    "coverage": coverage_count,
                    "source_integrity": source_integrity_count,
                    "objective_fulfillment": objective_count,
                }
            )
        objective = sensor_manifest.get("objective_fulfillment", {})
        objective_rows = objective.get("requirement_rows", []) if isinstance(objective, dict) else []
        if not isinstance(objective_rows, list) or len(objective_rows) != objective_count:
            issues.append(
                {
                    "severity": "error",
                    "code": "consumer_sensor_objective_rows_invalid",
                    "slug": slug,
                    "expected": objective_count,
                    "actual": len(objective_rows) if isinstance(objective_rows, list) else "",
                }
            )
        source_integrity = sensor_manifest.get("source_integrity", {})
        source_integrity_rows = source_integrity.get("source_integrity_matrix_rows", []) if isinstance(source_integrity, dict) else []
        if not isinstance(source_integrity_rows, list) or len(source_integrity_rows) != source_integrity_count:
            issues.append(
                {
                    "severity": "error",
                    "code": "consumer_sensor_source_integrity_rows_invalid",
                    "slug": slug,
                    "expected": source_integrity_count,
                    "actual": len(source_integrity_rows) if isinstance(source_integrity_rows, list) else "",
                }
            )
        quantitative = sensor_manifest.get("quantitative_execution", {})
        if not isinstance(quantitative, dict) or len(quantitative.get("execution_plan_rows", [])) < 2:
            issues.append({"severity": "error", "code": "consumer_sensor_quantitative_plan_missing", "slug": slug})
        if safe_int(row_counts.get("resource_limited_batch")) and not quantitative.get("resource_limited_batch_rows"):
            issues.append({"severity": "error", "code": "consumer_sensor_resource_limited_rows_missing", "slug": slug})
    missing_slugs = expected_slugs - loaded_slugs
    extra_slugs = loaded_slugs - expected_slugs
    for slug in sorted(missing_slugs):
        issues.append({"severity": "error", "code": "consumer_sensor_manifest_missing", "slug": slug})
    for slug in sorted(extra_slugs):
        issues.append({"severity": "warning", "code": "consumer_sensor_manifest_extra", "slug": slug})
    return bundle, issues


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


def write_html(path: Path, payload: dict[str, Any], artifact_checks: list[dict[str, Any]], sensor_checks: list[dict[str, Any]]) -> None:
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
    issue_rows = validation.get("issues", [])
    issue_html = html_table(issue_rows, ["severity", "code", "slug", "artifact_id", "path"]) if issue_rows else '<p class="pass">No loader errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Handoff Loader Validation</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Handoff Loader Validation</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This validates consumer load paths and join keys; it does not certify product accuracy.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">loader status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors checked</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("runtime_row_count", 0))}</div><div class="muted">runtime rows loaded</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("kernel_row_count", 0))}</div><div class="muted">kernel rows loaded</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("import_contract_pointer_resolved_count", 0))}</div><div class="muted">import pointers resolved</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("canonical_payload_pointer_resolved_count", 0))}</div><div class="muted">canonical pointers resolved</div></div>
</div>
<h2>Consumer Contract</h2>
<ul>
<li>Runtime rows join to crosstalk kernels by <code>runtime_id</code>.</li>
<li>Sensor rows join to model/prior/color/module artifacts by <code>slug</code>.</li>
<li>Research-mode load is valid when this report passes; product use remains blocked until row-level gates pass.</li>
</ul>
<h2>Issues</h2>{issue_html}
<h2>Sensor Loader Checks</h2>{html_table(sensor_checks, SENSOR_CHECK_COLUMNS)}
<h2>Artifact Loader Checks</h2>{html_table(artifact_checks, ARTIFACT_CHECK_COLUMNS)}
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
    outputs["camera_e2e_handoff_loader_validation_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_handoff_loader_validation_sensor_csv"] = payload["outputs"]["sensor_checks_csv"]
    outputs["camera_e2e_handoff_loader_validation_artifact_csv"] = payload["outputs"]["artifact_checks_csv"]
    outputs["camera_e2e_handoff_loader_validation_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_handoff_loader_validation"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "runtime_row_count": payload["runtime_row_count"],
        "kernel_row_count": payload["kernel_row_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def validate_loader(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    handoff_dir = args.handoff_dir.resolve()
    output_dir = args.output_dir.resolve()

    handoff_manifest = read_json(handoff_dir / "camera_e2e_handoff_manifest.json")
    handoff_sensor_rows = read_csv_rows(abs_from_repo(handoff_manifest.get("outputs", {}).get("sensors_csv", handoff_dir / "camera_e2e_handoff_sensors.csv")))
    artifact_rows = read_csv_rows(abs_from_repo(handoff_manifest.get("outputs", {}).get("artifacts_csv", handoff_dir / "camera_e2e_handoff_artifacts.csv")))
    artifact_by_id = index_by(artifact_rows, "artifact_id")

    def artifact_path(artifact_id: str, fallback: str) -> Path:
        row = artifact_by_id.get(artifact_id, {})
        return abs_from_repo(row.get("path", fallback))

    runtime_rows = read_csv_rows(artifact_path("runtime_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv"))
    kernel_rows = read_csv_rows(artifact_path("runtime_crosstalk_kernel", "runs/camera_e2e_sensor_lut_package/camera_e2e_runtime_bundle/camera_e2e_runtime_crosstalk_kernel.csv"))
    consumer_bundle_path = artifact_path("consumer_bundle", "runs/camera_e2e_sensor_lut_package/camera_e2e_consumer_bundle/camera_e2e_consumer_bundle.json")
    import_contract_path = artifact_path("import_contract", "runs/camera_e2e_sensor_lut_package/camera_e2e_import_contract/camera_e2e_import_contract.json")
    canonical_payload_path = artifact_path("canonical_payload", "runs/camera_e2e_sensor_lut_package/camera_e2e_canonical_payload/camera_e2e_canonical_payload.json")
    flat_query_path = artifact_path("flat_sensor_query", "runs/camera_e2e_sensor_lut_package/camera_e2e_flat_sensor_query/camera_e2e_flat_sensor_query.json")
    flat_product_query_path = artifact_path("flat_sensor_product_query_probe", "runs/camera_e2e_sensor_lut_package/camera_e2e_flat_sensor_query_product_probe/camera_e2e_flat_sensor_query.json")
    analysis_report_path = artifact_path("analysis_report", "runs/camera_e2e_sensor_lut_package/camera_e2e_analysis_report/camera_e2e_analysis_report.json")
    objective_trace_path = artifact_path("objective_trace", "runs/camera_e2e_sensor_lut_package/camera_e2e_objective_trace/camera_e2e_objective_trace.json")
    deliverable_summary_path = artifact_path("sensor_deliverable_summary", "runs/camera_e2e_sensor_lut_package/camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.csv")
    deliverable_summary_json_path = artifact_path("sensor_deliverable_summary_json", "runs/camera_e2e_sensor_lut_package/camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.json")
    coverage_rows = read_csv_rows(artifact_path("coverage_matrix", "runs/camera_e2e_sensor_lut_package/camera_e2e_coverage_matrix/camera_e2e_coverage_matrix.csv"))
    deliverable_summary_rows = read_csv_rows(deliverable_summary_path)
    deliverable_summary = read_json(deliverable_summary_json_path)
    source_integrity_rows = read_csv_rows(artifact_path("source_integrity_matrix", "runs/camera_e2e_sensor_lut_package/camera_e2e_lut_source_integrity/camera_e2e_lut_source_integrity_matrix.csv"))
    spectral_rows = read_csv_rows(artifact_path("color_spectral_response", "runs/camera_e2e_sensor_lut_package/camera_e2e_color_response/camera_e2e_spectral_response.csv"))
    material_rows = read_csv_rows(artifact_path("material_nk_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_material_tables/camera_e2e_material_nk_lut.csv"))
    cfa_provenance_rows = read_csv_rows(artifact_path("cfa_provenance_by_sensor", "runs/camera_e2e_sensor_lut_package/camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv"))
    cfa_db_rows = read_csv_rows(artifact_path("cfa_db_by_sensor", "runs/camera_e2e_sensor_lut_package/camera_e2e_cfa_db_tables/camera_e2e_cfa_db_by_sensor.csv"))
    cfa_db_transmission_rows = read_csv_rows(artifact_path("cfa_db_transmission_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_cfa_db_tables/camera_e2e_cfa_db_transmission_lut.csv"))
    capability_rows = read_csv_rows(artifact_path("capability_profile_by_sensor", "runs/camera_e2e_sensor_lut_package/camera_e2e_capability_profile/camera_e2e_capability_by_sensor.csv"))
    trust_sensor_rows = read_csv_rows(artifact_path("lut_trust_by_sensor", "runs/camera_e2e_sensor_lut_package/camera_e2e_lut_trust_assessment/camera_e2e_lut_trust_by_sensor.csv"))
    use_scope_rows = read_csv_rows(artifact_path("use_scope_by_sensor", "runs/camera_e2e_sensor_lut_package/camera_e2e_use_scope_summary/camera_e2e_use_scope_by_sensor.csv"))
    electrical_rows = read_csv_rows(artifact_path("electrical_noise_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_electrical_readout_tables/camera_e2e_electrical_noise_lut.csv"))
    readout_rows = read_csv_rows(artifact_path("readout_gain_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_electrical_readout_tables/camera_e2e_readout_gain_lut.csv"))
    binning_rows = read_csv_rows(artifact_path("binning_remosaic_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_electrical_readout_tables/camera_e2e_binning_remosaic_lut.csv"))
    color_matrix_rows = read_csv_rows(artifact_path("color_matrix_seed", "runs/camera_e2e_sensor_lut_package/camera_e2e_color_response/camera_e2e_color_matrix_seed.csv"))
    prior_summary_rows = read_csv_rows(package_dir / "camera_e2e_prior_seed_models" / "camera_e2e_prior_seed_summary.csv")
    module_field_rows = read_csv_rows(artifact_path("module_coupling_lut", "runs/camera_e2e_sensor_lut_package/camera_e2e_module_coupling/camera_e2e_module_coupling_field_lut.csv"))
    module_summary_rows = read_csv_rows(package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_summary.csv")
    probe_summary_rows = read_csv_rows(artifact_path("research_probe_all_sensors", "runs/camera_e2e_sensor_lut_package/camera_e2e_sensor_probe_all_sensors/camera_e2e_sensor_probe_summary.csv"))
    quantitative_plan_rows = read_csv_rows(artifact_path("quantitative_execution_plan", "runs/camera_e2e_sensor_lut_package/camera_e2e_quantitative_execution_plan.csv"))
    quantitative_queue_rows = read_csv_rows(artifact_path("quantitative_point_queue", "runs/camera_e2e_sensor_lut_package/camera_e2e_quantitative_point_queue.csv"))
    quantitative_coverage_rows = read_csv_rows(artifact_path("quantitative_coverage", "runs/camera_e2e_sensor_lut_package/camera_e2e_quantitative_coverage.csv"))
    resource_limited_rows = read_csv_rows(artifact_path("resource_limited_batch_plan", "runs/camera_e2e_sensor_lut_package/camera_e2e_resource_limited_batch_plan.csv"))
    objective_requirement_summary_rows = read_csv_rows(
        artifact_path("objective_trace_by_requirement_summary", "runs/camera_e2e_sensor_lut_package/camera_e2e_objective_trace/camera_e2e_objective_trace_by_requirement_summary.csv")
    )

    artifact_checks, artifact_issues = build_artifact_checks(artifact_rows)
    consumer_bundle, consumer_issues = validate_consumer_bundle(consumer_bundle_path, handoff_sensor_rows)
    import_contract = read_json(import_contract_path)
    canonical_payload = read_json(canonical_payload_path)
    flat_query = read_json(flat_query_path)
    flat_product_query = read_json(flat_product_query_path)
    analysis_report = read_json(analysis_report_path)
    objective_trace = read_json(objective_trace_path)
    flat_query_issues: list[dict[str, Any]] = []
    if flat_query.get("schema") != "camera_e2e_flat_sensor_query_v1" or not bool(flat_query.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "flat_sensor_query_invalid", "path": repo_rel(flat_query_path)})
    if int(flat_query.get("allowed_query_count", 0) or 0) <= 0:
        flat_query_issues.append({"severity": "error", "code": "flat_sensor_query_no_allowed_rows", "path": repo_rel(flat_query_path)})
    if int(flat_query.get("product_ready_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "flat_sensor_query_unexpected_product_ready", "path": repo_rel(flat_query_path)})
    if flat_product_query.get("schema") != "camera_e2e_flat_sensor_query_v1" or not bool(flat_product_query.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "flat_sensor_product_query_invalid", "path": repo_rel(flat_product_query_path)})
    if int(flat_product_query.get("allowed_query_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "flat_sensor_product_query_unexpectedly_allowed", "path": repo_rel(flat_product_query_path)})
    if analysis_report.get("schema") != "camera_e2e_analysis_report_v1" or not bool(analysis_report.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "analysis_report_invalid", "path": repo_rel(analysis_report_path)})
    if int(analysis_report.get("sensor_count", 0) or 0) <= 0 or int(analysis_report.get("channel_row_count", 0) or 0) <= 0:
        flat_query_issues.append({"severity": "error", "code": "analysis_report_empty", "path": repo_rel(analysis_report_path)})
    if int(analysis_report.get("product_ready_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "analysis_report_unexpected_product_ready", "path": repo_rel(analysis_report_path)})
    if import_contract.get("schema") != "camera_e2e_import_contract_v1" or not bool(import_contract.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "import_contract_invalid", "path": repo_rel(import_contract_path)})
    if int(import_contract.get("sensor_count", 0) or 0) != len(handoff_sensor_rows):
        flat_query_issues.append(
            {
                "severity": "error",
                "code": "import_contract_sensor_count_mismatch",
                "path": repo_rel(import_contract_path),
                "expected": len(handoff_sensor_rows),
                "actual": import_contract.get("sensor_count", 0),
            }
        )
    if int(import_contract.get("requirement_row_count", 0) or 0) <= 0:
        flat_query_issues.append({"severity": "error", "code": "import_contract_empty", "path": repo_rel(import_contract_path)})
    if int(import_contract.get("pointer_resolved_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        flat_query_issues.append({"severity": "error", "code": "import_contract_unresolved_pointers", "path": repo_rel(import_contract_path)})
    if int(import_contract.get("research_allowed_requirement_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        flat_query_issues.append({"severity": "error", "code": "import_contract_research_not_loadable", "path": repo_rel(import_contract_path)})
    if int(import_contract.get("product_allowed_requirement_count", 0) or 0) != 0 or int(import_contract.get("product_ready_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "import_contract_unexpected_product_rows", "path": repo_rel(import_contract_path)})
    if canonical_payload.get("schema") != "camera_e2e_canonical_payload_v1" or not bool(canonical_payload.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "canonical_payload_invalid", "path": repo_rel(canonical_payload_path)})
    if int(canonical_payload.get("sensor_count", 0) or 0) != len(handoff_sensor_rows):
        flat_query_issues.append(
            {
                "severity": "error",
                "code": "canonical_payload_sensor_count_mismatch",
                "path": repo_rel(canonical_payload_path),
                "expected": len(handoff_sensor_rows),
                "actual": canonical_payload.get("sensor_count", 0),
            }
        )
    if int(canonical_payload.get("requirement_row_count", 0) or 0) != int(import_contract.get("requirement_row_count", 0) or 0):
        flat_query_issues.append({"severity": "error", "code": "canonical_payload_requirement_count_mismatch", "path": repo_rel(canonical_payload_path)})
    if int(canonical_payload.get("pointer_resolved_count", 0) or 0) != int(canonical_payload.get("requirement_row_count", 0) or 0):
        flat_query_issues.append({"severity": "error", "code": "canonical_payload_unresolved_pointers", "path": repo_rel(canonical_payload_path)})
    if int(canonical_payload.get("product_allowed_requirement_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "canonical_payload_unexpected_product_rows", "path": repo_rel(canonical_payload_path)})
    if objective_trace.get("schema") != "camera_e2e_objective_trace_v1" or not bool(objective_trace.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "objective_trace_invalid", "path": repo_rel(objective_trace_path)})
    if int(objective_trace.get("flat_pointer_fail_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "objective_trace_flat_pointer_fail", "path": repo_rel(objective_trace_path)})
    deliverable_slugs = {row.get("slug", "") for row in deliverable_summary_rows if row.get("slug")}
    handoff_slugs = {row.get("slug", "") for row in handoff_sensor_rows if row.get("slug")}
    if deliverable_summary.get("schema") != "camera_e2e_sensor_deliverable_summary_v1" or not bool(deliverable_summary.get("validation", {}).get("pass")):
        flat_query_issues.append({"severity": "error", "code": "sensor_deliverable_summary_invalid", "path": repo_rel(deliverable_summary_json_path)})
    if len(deliverable_summary_rows) != len(handoff_sensor_rows):
        flat_query_issues.append(
            {
                "severity": "error",
                "code": "sensor_deliverable_summary_row_count_mismatch",
                "path": repo_rel(deliverable_summary_path),
                "expected": len(handoff_sensor_rows),
                "actual": len(deliverable_summary_rows),
            }
        )
    if deliverable_slugs != handoff_slugs:
        flat_query_issues.append(
            {
                "severity": "error",
                "code": "sensor_deliverable_summary_slug_mismatch",
                "path": repo_rel(deliverable_summary_path),
                "missing": sorted(handoff_slugs - deliverable_slugs),
                "extra": sorted(deliverable_slugs - handoff_slugs),
            }
        )
    if int(deliverable_summary.get("product_ready_count", 0) or 0) != 0:
        flat_query_issues.append({"severity": "error", "code": "sensor_deliverable_summary_unexpected_product_ready", "path": repo_rel(deliverable_summary_json_path)})
    expected_objective_requirement_count = len({row.get("requirement_id", "") for row in coverage_rows if row.get("requirement_id")})
    if len(objective_requirement_summary_rows) != expected_objective_requirement_count:
        flat_query_issues.append(
            {
                "severity": "error",
                "code": "objective_requirement_summary_row_count_unexpected",
                "expected": expected_objective_requirement_count,
                "actual": len(objective_requirement_summary_rows),
            }
        )
    sensor_checks, sensor_issues = build_sensor_checks(
        handoff_sensor_rows,
        coverage_rows=coverage_rows,
        runtime_rows=runtime_rows,
        kernel_rows=kernel_rows,
        spectral_rows=spectral_rows,
        material_rows=material_rows,
        cfa_provenance_rows=cfa_provenance_rows,
        cfa_db_rows=cfa_db_rows,
        cfa_db_transmission_rows=cfa_db_transmission_rows,
        capability_rows=capability_rows,
        trust_sensor_rows=trust_sensor_rows,
        use_scope_rows=use_scope_rows,
        electrical_rows=electrical_rows,
        readout_rows=readout_rows,
        binning_rows=binning_rows,
        color_matrix_rows=color_matrix_rows,
        prior_summary_rows=prior_summary_rows,
        module_field_rows=module_field_rows,
        module_summary_rows=module_summary_rows,
        probe_summary_rows=probe_summary_rows,
        quantitative_plan_rows=quantitative_plan_rows,
        quantitative_queue_rows=quantitative_queue_rows,
        quantitative_coverage_rows=quantitative_coverage_rows,
        resource_limited_rows=resource_limited_rows,
        source_integrity_rows=source_integrity_rows,
    )
    issues = artifact_issues + consumer_issues + flat_query_issues + sensor_issues

    if handoff_manifest.get("schema") != "camera_e2e_handoff_manifest_v1":
        issues.append({"severity": "error", "code": "handoff_manifest_schema_invalid", "path": repo_rel(handoff_dir / "camera_e2e_handoff_manifest.json")})
    if not bool(handoff_manifest.get("validation", {}).get("pass")):
        issues.append({"severity": "error", "code": "handoff_manifest_validation_failed"})

    product_ready_count = sum(1 for row in sensor_checks if boolish(row.get("product_ready")))
    sensor_count = len(sensor_checks)
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    if error_count:
        status = "FAIL"
    elif product_ready_count == sensor_count and sensor_count:
        status = "PRODUCT_LOADER_VALID"
    else:
        status = "RESEARCH_LOADER_VALID_PRODUCT_BLOCKED"

    report_json = output_dir / "camera_e2e_handoff_loader_validation.json"
    sensor_csv = output_dir / "camera_e2e_handoff_loader_sensor_checks.csv"
    artifact_csv = output_dir / "camera_e2e_handoff_loader_artifact_checks.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_handoff_loader_validation_v1",
        "artifact_role": "camera_e2e_consumer_load_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "handoff_manifest": repo_rel(handoff_dir / "camera_e2e_handoff_manifest.json"),
        "sensor_count": sensor_count,
        "artifact_count": len(artifact_checks),
        "consumer_bundle_sensor_count": consumer_bundle.get("sensor_count", 0),
        "consumer_bundle_manifest_count": len(consumer_bundle.get("sensor_manifest_json_files", [])) if isinstance(consumer_bundle.get("sensor_manifest_json_files"), list) else 0,
        "import_contract_status": import_contract.get("validation", {}).get("status", ""),
        "import_contract_requirement_row_count": import_contract.get("requirement_row_count", 0),
        "import_contract_pointer_resolved_count": import_contract.get("pointer_resolved_count", 0),
        "import_contract_product_allowed_requirement_count": import_contract.get("product_allowed_requirement_count", 0),
        "canonical_payload_status": canonical_payload.get("validation", {}).get("status", ""),
        "canonical_payload_requirement_row_count": canonical_payload.get("requirement_row_count", 0),
        "canonical_payload_pointer_resolved_count": canonical_payload.get("pointer_resolved_count", 0),
        "canonical_payload_product_allowed_requirement_count": canonical_payload.get("product_allowed_requirement_count", 0),
        "flat_query_row_count": flat_query.get("query_row_count", 0),
        "flat_query_allowed_count": flat_query.get("allowed_query_count", 0),
        "flat_product_query_allowed_count": flat_product_query.get("allowed_query_count", 0),
        "analysis_report_sensor_count": analysis_report.get("sensor_count", 0),
        "analysis_report_channel_row_count": analysis_report.get("channel_row_count", 0),
        "analysis_report_check_channel_row_count": analysis_report.get("check_channel_row_count", 0),
        "sensor_deliverable_summary_row_count": len(deliverable_summary_rows),
        "sensor_deliverable_summary_status": deliverable_summary.get("validation", {}).get("status", ""),
        "sensor_deliverable_summary_product_ready_count": deliverable_summary.get("product_ready_count", 0),
        "sensor_deliverable_summary_gate_counts": deliverable_summary.get("deliverable_gate_counts", {}),
        "runtime_row_count": len(runtime_rows),
        "kernel_row_count": len(kernel_rows),
        "spectral_row_count": len(spectral_rows),
        "material_row_count": len(material_rows),
        "coverage_row_count": len(coverage_rows),
        "source_integrity_row_count": len(source_integrity_rows),
        "cfa_db_row_count": len(cfa_db_rows),
        "cfa_db_transmission_row_count": len(cfa_db_transmission_rows),
        "use_scope_row_count": len(use_scope_rows),
        "electrical_row_count": len(electrical_rows),
        "readout_row_count": len(readout_rows),
        "binning_row_count": len(binning_rows),
        "module_field_row_count": len(module_field_rows),
        "probe_summary_row_count": len(probe_summary_rows),
        "quantitative_plan_row_count": len(quantitative_plan_rows),
        "quantitative_queue_row_count": len(quantitative_queue_rows),
        "quantitative_coverage_row_count": len(quantitative_coverage_rows),
        "resource_limited_batch_row_count": len(resource_limited_rows),
        "objective_requirement_summary_row_count": len(objective_requirement_summary_rows),
        "objective_trace_flat_pointer_fail_count": objective_trace.get("flat_pointer_fail_count", 0),
        "objective_trace_flat_pointer_gate_counts": objective_trace.get("flat_pointer_gate_counts", {}),
        "product_ready_count": product_ready_count,
        "gate_counts": {
            "sensor_loader_gate": dict(Counter(str(row.get("loader_gate", "")) for row in sensor_checks)),
            "artifact_loader_gate": dict(Counter(str(row.get("loader_gate", "")) for row in artifact_checks)),
            "production_lut_gate": dict(Counter(str(row.get("production_lut_gate", "")) for row in sensor_checks)),
            "cra_mismatch_gate": dict(Counter(str(row.get("cra_mismatch_gate", "")) for row in sensor_checks)),
            "camera_e2e_use_scope": dict(Counter(str(row.get("camera_e2e_use_scope", "")) for row in sensor_checks)),
        },
        "consumer_contract": {
            "entrypoint": "camera_e2e_handoff_manifest/camera_e2e_handoff_manifest.json",
            "sensor_deliverable_summary": "camera_e2e_sensor_deliverable_summary/camera_e2e_sensor_deliverable_summary.csv",
            "consumer_bundle": "camera_e2e_consumer_bundle/camera_e2e_consumer_bundle.json",
            "import_contract": "camera_e2e_import_contract/camera_e2e_import_contract.json",
            "canonical_payload": "camera_e2e_canonical_payload/camera_e2e_canonical_payload.json",
            "join_keys": {
                "sensor_level": "slug",
                "sensor_deliverable_summary": "slug",
                "runtime_to_kernel": "runtime_id",
                "runtime_to_color": "slug + color_channel + wavelength_nm",
                "runtime_to_material": "slug + material_family/material_key + color_channel + wavelength_nm",
                "cfa_db": "slug + color_channel + wavelength_nm",
                "requirement_coverage": "slug + requirement_id",
                "source_integrity": "slug + requirement_id",
                "use_scope": "slug",
                "runtime_to_electrical": "slug + temperature_c + exposure_s + signal_fraction",
                "runtime_to_readout": "slug + analog_gain_x + digital_gain_x + adc_bit_depth",
                "runtime_to_binning": "slug + mode_id",
                "module_field": "slug + field_case + wavelength_nm",
            },
            "policy": "Research load may proceed when validation.pass is true; product ingest must still require product_lut_ready gates.",
        },
        "validation": {
            "schema": "camera_e2e_handoff_loader_validation_result_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": warning_count,
            "issues": issues,
        },
        "outputs": {
            "json": repo_rel(report_json),
            "sensor_checks_csv": repo_rel(sensor_csv),
            "artifact_checks_csv": repo_rel(artifact_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_checks, SENSOR_CHECK_COLUMNS)
    write_csv(artifact_csv, artifact_checks, ARTIFACT_CHECK_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, artifact_checks, sensor_checks)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = validate_loader(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "runtime_row_count": payload["runtime_row_count"],
                "kernel_row_count": payload["kernel_row_count"],
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
