#!/usr/bin/env python3
"""Audit whether the CameraE2E sensor package satisfies the current objective.

This audit is stricter than a file-exists check. It verifies that every sensor
has requirement coverage, the consumer bundle can be queried, and product use
stays blocked while the data is still prior/proxy/research-grade.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_objective_acceptance"

CHECK_COLUMNS = ["check_id", "scope", "pass", "status", "evidence", "required_action"]
SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "requirement_count",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "consumer_query_rows",
    "consumer_query_allowed_rows",
    "product_query_allowed_rows",
    "flat_query_rows",
    "flat_query_allowed_rows",
    "flat_product_query_allowed_rows",
    "analysis_gate",
    "mesh_confidence_class",
    "field_mesh_pass_fraction",
    "crosstalk_mesh_pass_fraction",
    "material_rows",
    "electrical_rows",
    "module_rows",
    "crosstalk_support_gate_counts",
    "crosstalk_support_row_count",
    "crosstalk_product_candidate_row_count",
    "lut_trust_class",
    "lut_trust_evidence_score_0_100",
    "lut_trust_product_score_0_100",
    "product_closure_status",
    "product_closure_measured_input_blocker_count",
    "product_closure_measured_calibration_blocker_count",
    "product_closure_field_solver_row_count",
    "product_closure_crosstalk_product_primary_row_count",
    "product_closure_crosstalk_support_discovery_row_count",
    "product_closure_first_action_class",
    "product_closure_first_action_type",
    "product_closure_first_action_source",
    "product_closure_first_action_local_feasibility",
    "acceptance_gate",
    "product_ready",
    "primary_blockers",
]


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


def safe_ratio(numerator: Any, denominator: Any) -> str:
    den = safe_int(denominator)
    if den <= 0:
        return ""
    return f"{safe_int(numerator) / den:.6f}"


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def check_row(check_id: str, scope: str, passed: bool, status: str, evidence: Any, action: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "scope": scope,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()

    package = read_json(package_dir / "camera_e2e_lut_package.json")
    coverage_json = read_json(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.json")
    consumer_bundle = read_json(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_bundle.json")
    consumer_query = read_json(package_dir / "camera_e2e_consumer_query" / "camera_e2e_consumer_query.json")
    product_query = read_json(package_dir / "camera_e2e_consumer_query_product_probe" / "camera_e2e_consumer_query.json")
    handoff_loader = read_json(package_dir / "camera_e2e_handoff_loader_validation" / "camera_e2e_handoff_loader_validation.json")
    readiness = read_json(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json")
    trust_assessment = read_json(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_assessment.json")
    flat_sensor_bundle = read_json(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json")
    flat_sensor_query = read_json(package_dir / "camera_e2e_flat_sensor_query" / "camera_e2e_flat_sensor_query.json")
    flat_product_query = read_json(package_dir / "camera_e2e_flat_sensor_query_product_probe" / "camera_e2e_flat_sensor_query.json")
    analysis_report = read_json(package_dir / "camera_e2e_analysis_report" / "camera_e2e_analysis_report.json")
    mesh_confidence = read_json(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence.json")
    product_closure_summary = read_json(package_dir / "camera_e2e_product_closure_summary" / "camera_e2e_product_closure_summary.json")
    usage_policy = read_json(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy.json")
    adapter_examples = read_json(package_dir / "camera_e2e_adapter_examples" / "camera_e2e_adapter_examples.json")
    adapter_smoke = read_json(package_dir / "camera_e2e_adapter_smoke" / "camera_e2e_adapter_smoke.json")
    objective_trace = read_json(package_dir / "camera_e2e_objective_trace" / "camera_e2e_objective_trace.json")

    coverage_rows = read_csv_rows(package_dir / "camera_e2e_coverage_matrix" / "camera_e2e_coverage_matrix.csv")
    consumer_index_rows = read_csv_rows(package_dir / "camera_e2e_consumer_bundle" / "camera_e2e_consumer_sensor_index.csv")
    consumer_query_rows = read_csv_rows(package_dir / "camera_e2e_consumer_query" / "camera_e2e_consumer_query.csv")
    product_query_rows = read_csv_rows(package_dir / "camera_e2e_consumer_query_product_probe" / "camera_e2e_consumer_query.csv")
    trust_rows = read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv")
    flat_index_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv")
    flat_query_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_query" / "camera_e2e_flat_sensor_query.csv")
    flat_product_query_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_query_product_probe" / "camera_e2e_flat_sensor_query.csv")
    analysis_sensor_rows = read_csv_rows(package_dir / "camera_e2e_analysis_report" / "camera_e2e_analysis_by_sensor.csv")
    mesh_sensor_rows = read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv")
    product_closure_sensor_rows = read_csv_rows(
        package_dir / "camera_e2e_product_closure_summary" / "camera_e2e_product_closure_by_sensor.csv"
    )
    usage_policy_sensor_rows = read_csv_rows(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy_by_sensor.csv")
    adapter_example_sensor_rows = read_csv_rows(package_dir / "camera_e2e_adapter_examples" / "camera_e2e_adapter_examples_by_sensor.csv")
    adapter_smoke_sensor_rows = read_csv_rows(package_dir / "camera_e2e_adapter_smoke" / "camera_e2e_adapter_smoke_by_sensor.csv")
    objective_trace_rows = read_csv_rows(package_dir / "camera_e2e_objective_trace" / "camera_e2e_objective_trace_by_requirement.csv")
    objective_trace_sensor_rows = read_csv_rows(package_dir / "camera_e2e_objective_trace" / "camera_e2e_objective_trace_by_sensor.csv")

    coverage_by_slug = group_rows(coverage_rows, "slug")
    query_by_slug = group_rows(consumer_query_rows, "slug")
    product_query_by_slug = group_rows(product_query_rows, "slug")
    trust_by_slug = {row.get("slug", ""): row for row in trust_rows if row.get("slug")}
    flat_index_by_slug = {row.get("slug", ""): row for row in flat_index_rows if row.get("slug")}
    flat_query_by_slug = group_rows(flat_query_rows, "slug")
    flat_product_query_by_slug = group_rows(flat_product_query_rows, "slug")
    analysis_by_slug = {row.get("slug", ""): row for row in analysis_sensor_rows if row.get("slug")}
    mesh_by_slug = {row.get("slug", ""): row for row in mesh_sensor_rows if row.get("slug")}
    product_closure_by_slug = {row.get("slug", ""): row for row in product_closure_sensor_rows if row.get("slug")}
    expected_requirement_count = safe_int(coverage_json.get("requirement_count_per_sensor"), 0)
    expected_sensor_count = safe_int(package.get("sensor_count"), 0)
    check_rows: list[dict[str, Any]] = []

    check_rows.append(
        check_row(
            "coverage_matrix_valid",
            "package",
            bool(coverage_json.get("validation", {}).get("pass")),
            coverage_json.get("validation", {}).get("status", "MISSING"),
            {"coverage_row_count": coverage_json.get("coverage_row_count"), "requirement_count_per_sensor": expected_requirement_count},
            "Regenerate coverage matrix.",
        )
    )
    check_rows.append(
        check_row(
            "consumer_bundle_valid",
            "package",
            bool(consumer_bundle.get("validation", {}).get("pass")),
            consumer_bundle.get("validation", {}).get("status", "MISSING"),
            {"sensor_count": consumer_bundle.get("sensor_count"), "manifest_count": len(consumer_bundle.get("sensor_manifest_json_files", [])) if isinstance(consumer_bundle.get("sensor_manifest_json_files"), list) else 0},
            "Regenerate consumer bundle.",
        )
    )
    check_rows.append(
        check_row(
            "consumer_research_query_valid",
            "package",
            bool(consumer_query.get("validation", {}).get("pass")) and safe_int(consumer_query.get("allowed_query_count")) > 0,
            consumer_query.get("validation", {}).get("status", "MISSING"),
            {"query_row_count": consumer_query.get("query_row_count"), "allowed_query_count": consumer_query.get("allowed_query_count")},
            "Run query_camera_e2e_consumer_bundle.py in research mode.",
        )
    )
    trust_missing_query_rows = [
        row.get("consumer_query_id", "")
        for row in consumer_query_rows
        if not str(row.get("lut_trust_class", "")).strip()
        or not str(row.get("lut_trust_evidence_score_0_100", "")).strip()
        or not str(row.get("lut_trust_product_score_0_100", "")).strip()
    ]
    check_rows.append(
        check_row(
            "consumer_query_trust_join_valid",
            "package",
            not trust_missing_query_rows and bool(consumer_query_rows),
            "PASS" if not trust_missing_query_rows and consumer_query_rows else "FAIL",
            {"missing_trust_query_ids": trust_missing_query_rows[:20], "query_row_count": len(consumer_query_rows)},
            "Regenerate consumer bundle query after trust assessment export.",
        )
    )
    support_missing_query_rows = [
        row.get("consumer_query_id", "")
        for row in consumer_query_rows
        if not str(row.get("crosstalk_support_gate", "")).strip()
    ]
    check_rows.append(
        check_row(
            "consumer_query_crosstalk_support_join_valid",
            "package",
            not support_missing_query_rows and bool(consumer_query_rows),
            "PASS" if not support_missing_query_rows and consumer_query_rows else "FAIL",
            {"missing_support_query_ids": support_missing_query_rows[:20], "query_row_count": len(consumer_query_rows)},
            "Regenerate consumer bundle query after crosstalk support audit export.",
        )
    )
    check_rows.append(
        check_row(
            "consumer_product_query_blocked",
            "package",
            bool(product_query.get("validation", {}).get("pass")) and safe_int(product_query.get("allowed_query_count")) == 0,
            product_query.get("validation", {}).get("status", "MISSING"),
            {"query_row_count": product_query.get("query_row_count"), "allowed_query_count": product_query.get("allowed_query_count")},
            "Product query must fail closed until measured/calibrated product gates pass.",
        )
    )
    check_rows.append(
        check_row(
            "handoff_loader_valid",
            "package",
            bool(handoff_loader.get("validation", {}).get("pass")),
            handoff_loader.get("validation", {}).get("status", "MISSING"),
            {"sensor_count": handoff_loader.get("sensor_count"), "artifact_count": handoff_loader.get("artifact_count")},
            "Regenerate handoff loader validation.",
        )
    )
    check_rows.append(
        check_row(
            "lut_trust_assessment_valid",
            "package",
            bool(trust_assessment.get("validation", {}).get("pass")) and safe_int(trust_assessment.get("sensor_count")) == expected_sensor_count,
            trust_assessment.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": trust_assessment.get("sensor_count"),
                "product_ready_count": trust_assessment.get("product_ready_count"),
                "trust_class_counts": trust_assessment.get("trust_class_counts"),
            },
            "Regenerate export_camera_e2e_lut_trust_assessment.py before objective acceptance.",
        )
    )
    check_rows.append(
        check_row(
            "flat_sensor_bundle_valid",
            "package",
            bool(flat_sensor_bundle.get("validation", {}).get("pass")) and safe_int(flat_sensor_bundle.get("sensor_count")) == expected_sensor_count,
            flat_sensor_bundle.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": flat_sensor_bundle.get("sensor_count"),
                "product_ready_count": flat_sensor_bundle.get("product_ready_count"),
                "total_embedded_row_count": flat_sensor_bundle.get("total_embedded_row_count"),
            },
            "Regenerate export_camera_e2e_flat_sensor_bundle.py.",
        )
    )
    check_rows.append(
        check_row(
            "flat_sensor_research_query_valid",
            "package",
            bool(flat_sensor_query.get("validation", {}).get("pass")) and safe_int(flat_sensor_query.get("allowed_query_count")) > 0,
            flat_sensor_query.get("validation", {}).get("status", "MISSING"),
            {"query_row_count": flat_sensor_query.get("query_row_count"), "allowed_query_count": flat_sensor_query.get("allowed_query_count")},
            "Run query_camera_e2e_flat_sensor_bundle.py in research mode.",
        )
    )
    check_rows.append(
        check_row(
            "flat_sensor_product_query_blocked",
            "package",
            bool(flat_product_query.get("validation", {}).get("pass")) and safe_int(flat_product_query.get("allowed_query_count")) == 0,
            flat_product_query.get("validation", {}).get("status", "MISSING"),
            {"query_row_count": flat_product_query.get("query_row_count"), "allowed_query_count": flat_product_query.get("allowed_query_count")},
            "Flat per-sensor product query must fail closed until product gates pass.",
        )
    )
    check_rows.append(
        check_row(
            "analysis_report_valid",
            "package",
            bool(analysis_report.get("validation", {}).get("pass"))
            and safe_int(analysis_report.get("sensor_count")) == expected_sensor_count
            and safe_int(analysis_report.get("channel_row_count")) > 0,
            analysis_report.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": analysis_report.get("sensor_count"),
                "channel_row_count": analysis_report.get("channel_row_count"),
                "check_channel_row_count": analysis_report.get("check_channel_row_count"),
                "mesh_confidence_class_counts": analysis_report.get("mesh_confidence_class_counts"),
            },
            "Regenerate export_camera_e2e_analysis_report.py.",
        )
    )
    check_rows.append(
        check_row(
            "mesh_confidence_valid_product_blocked",
            "package",
            bool(mesh_confidence.get("validation", {}).get("pass"))
            and safe_int(mesh_confidence.get("sensor_count")) == expected_sensor_count
            and safe_int(mesh_confidence.get("product_ready_count")) == 0,
            mesh_confidence.get("status", mesh_confidence.get("validation", {}).get("status", "MISSING")),
            {
                "field_pass_total": mesh_confidence.get("field_pass_total"),
                "field_required_total": mesh_confidence.get("field_required_total"),
                "crosstalk_pass_total": mesh_confidence.get("crosstalk_pass_total"),
                "crosstalk_required_total": mesh_confidence.get("crosstalk_required_total"),
                "confidence_class_counts": mesh_confidence.get("confidence_class_counts"),
            },
            "Regenerate audit_camera_e2e_mesh_confidence.py and keep product gates blocked until all required mesh points pass.",
        )
    )
    check_rows.append(
        check_row(
            "product_closure_summary_valid_product_blocked",
            "package",
            product_closure_summary.get("schema") == "camera_e2e_product_closure_summary_v1"
            and bool(product_closure_summary.get("validation", {}).get("pass"))
            and safe_int(product_closure_summary.get("sensor_count")) == expected_sensor_count
            and safe_int(product_closure_summary.get("product_ready_count")) == 0
            and len(product_closure_sensor_rows) == expected_sensor_count,
            product_closure_summary.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": product_closure_summary.get("sensor_count"),
                "product_ready_count": product_closure_summary.get("product_ready_count"),
                "domain_row_count": product_closure_summary.get("domain_row_count"),
                "sensor_row_count": len(product_closure_sensor_rows),
            },
            "Regenerate export_camera_e2e_product_closure_summary.py after closure plan and keep product gates blocked.",
        )
    )
    check_rows.append(
        check_row(
            "usage_policy_valid_product_blocked",
            "package",
            usage_policy.get("schema") == "camera_e2e_usage_policy_v1"
            and bool(usage_policy.get("validation", {}).get("pass"))
            and safe_int(usage_policy.get("sensor_policy_row_count")) == expected_sensor_count
            and safe_int(usage_policy.get("strict_product_filter_row_count")) == 0
            and safe_int(usage_policy.get("product_ingest_allowed_count")) == 0
            and len(usage_policy_sensor_rows) == expected_sensor_count,
            usage_policy.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_policy_row_count": usage_policy.get("sensor_policy_row_count"),
                "domain_policy_row_count": usage_policy.get("domain_policy_row_count"),
                "strict_product_filter_row_count": usage_policy.get("strict_product_filter_row_count"),
                "product_ingest_allowed_count": usage_policy.get("product_ingest_allowed_count"),
                "sensor_csv_row_count": len(usage_policy_sensor_rows),
            },
            "Regenerate export_camera_e2e_usage_policy.py and keep strict product filters closed.",
        )
    )
    check_rows.append(
        check_row(
            "adapter_examples_valid_product_blocked",
            "package",
            adapter_examples.get("schema") == "camera_e2e_adapter_examples_v1"
            and bool(adapter_examples.get("validation", {}).get("pass"))
            and safe_int(adapter_examples.get("sensor_count")) == expected_sensor_count
            and safe_int(adapter_examples.get("example_file_count")) == expected_sensor_count
            and safe_int(adapter_examples.get("product_allowed_query_count")) == 0
            and len(adapter_example_sensor_rows) == expected_sensor_count,
            adapter_examples.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": adapter_examples.get("sensor_count"),
                "example_file_count": adapter_examples.get("example_file_count"),
                "product_allowed_query_count": adapter_examples.get("product_allowed_query_count"),
                "sensor_csv_row_count": len(adapter_example_sensor_rows),
            },
            "Regenerate export_camera_e2e_adapter_examples.py and keep product query examples blocked.",
        )
    )
    check_rows.append(
        check_row(
            "adapter_smoke_valid_product_blocked",
            "package",
            adapter_smoke.get("schema") == "camera_e2e_adapter_smoke_v1"
            and bool(adapter_smoke.get("validation", {}).get("pass"))
            and safe_int(adapter_smoke.get("sensor_count")) == expected_sensor_count
            and safe_int(adapter_smoke.get("total_research_allowed_query_count")) > 0
            and safe_int(adapter_smoke.get("total_product_allowed_query_count")) == 0
            and len(adapter_smoke_sensor_rows) == expected_sensor_count,
            adapter_smoke.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": adapter_smoke.get("sensor_count"),
                "total_research_allowed_query_count": adapter_smoke.get("total_research_allowed_query_count"),
                "total_product_allowed_query_count": adapter_smoke.get("total_product_allowed_query_count"),
                "sensor_csv_row_count": len(adapter_smoke_sensor_rows),
            },
            "Run run_camera_e2e_adapter_smoke.py and keep product probes blocked.",
        )
    )
    check_rows.append(
        check_row(
            "objective_trace_valid_product_blocked",
            "package",
            objective_trace.get("schema") == "camera_e2e_objective_trace_v1"
            and bool(objective_trace.get("validation", {}).get("pass"))
            and safe_int(objective_trace.get("sensor_count")) == expected_sensor_count
            and safe_int(objective_trace.get("requirement_count_per_sensor")) == expected_requirement_count
            and safe_int(objective_trace.get("trace_row_count")) == expected_sensor_count * expected_requirement_count
            and safe_int(objective_trace.get("product_ready_count")) == 0
            and safe_int(objective_trace.get("adapter_product_allowed_query_count")) == 0
            and len(objective_trace_rows) == expected_sensor_count * expected_requirement_count
            and len(objective_trace_sensor_rows) == expected_sensor_count,
            objective_trace.get("validation", {}).get("status", "MISSING"),
            {
                "sensor_count": objective_trace.get("sensor_count"),
                "requirement_count_per_sensor": objective_trace.get("requirement_count_per_sensor"),
                "trace_row_count": objective_trace.get("trace_row_count"),
                "product_ready_count": objective_trace.get("product_ready_count"),
                "adapter_product_allowed_query_count": objective_trace.get("adapter_product_allowed_query_count"),
            },
            "Regenerate export_camera_e2e_objective_trace.py and keep product trace blocked.",
        )
    )
    check_rows.append(
        check_row(
            "product_gate_preserved",
            "package",
            not bool(package.get("camera_e2e_ready_count")) and not bool(consumer_query.get("product_ready_count")),
            "PRODUCT_BLOCKED_AS_EXPECTED",
            {
                "package_ready_count": package.get("camera_e2e_ready_count", 0),
                "consumer_query_product_ready_count": consumer_query.get("product_ready_count", 0),
                "readiness_product_lut_ready": readiness.get("product_lut_ready"),
            },
            "Do not open product gates without measured stack/material/CRA/electrical calibration and solver convergence.",
        )
    )

    sensor_rows: list[dict[str, Any]] = []
    for sensor in consumer_index_rows:
        slug = sensor.get("slug", "")
        coverage_for_slug = coverage_by_slug.get(slug, [])
        query_for_slug = query_by_slug.get(slug, [])
        product_for_slug = product_query_by_slug.get(slug, [])
        flat_for_slug = flat_query_by_slug.get(slug, [])
        flat_product_for_slug = flat_product_query_by_slug.get(slug, [])
        research_counts = gate_counts(coverage_for_slug, "research_gate")
        product_counts = gate_counts(coverage_for_slug, "product_gate")
        primary_blockers = sensor.get("primary_blockers", "")
        query_allowed = sum(1 for row in query_for_slug if boolish(row.get("query_allowed")))
        product_allowed = sum(1 for row in product_for_slug if boolish(row.get("query_allowed")))
        flat_allowed = sum(1 for row in flat_for_slug if boolish(row.get("query_allowed")))
        flat_product_allowed = sum(1 for row in flat_product_for_slug if boolish(row.get("query_allowed")))
        trust = trust_by_slug.get(slug, {})
        flat_index = flat_index_by_slug.get(slug, {})
        analysis = analysis_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        product_closure = product_closure_by_slug.get(slug, {})
        query_trust_rows = [row for row in query_for_slug if str(row.get("lut_trust_class", "")).strip()]
        query_support_rows = [row for row in query_for_slug if str(row.get("crosstalk_support_gate", "")).strip()]
        support_gate_counts = gate_counts(query_for_slug, "crosstalk_support_gate")
        product_ready = boolish(sensor.get("product_ready"))
        row_pass = (
            len(coverage_for_slug) == expected_requirement_count
            and research_counts.get("MISSING", 0) == 0
            and research_counts.get("FAIL", 0) == 0
            and query_allowed > 0
            and product_allowed == 0
            and safe_int(sensor.get("material_row_count")) > 0
            and safe_int(sensor.get("electrical_row_count")) > 0
            and safe_int(sensor.get("module_field_row_count")) > 0
            and safe_int(sensor.get("lut_trust_row_count")) > 0
            and bool(trust)
            and bool(flat_index)
            and flat_allowed > 0
            and flat_product_allowed == 0
            and bool(analysis)
            and bool(mesh)
            and bool(product_closure)
            and query_trust_rows
            and query_support_rows
            and not product_ready
        )
        sensor_rows.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "requirement_count": len(coverage_for_slug),
                "coverage_research_gate_counts": json.dumps(research_counts, sort_keys=True),
                "coverage_product_gate_counts": json.dumps(product_counts, sort_keys=True),
                "consumer_query_rows": len(query_for_slug),
                "consumer_query_allowed_rows": query_allowed,
                "product_query_allowed_rows": product_allowed,
                "flat_query_rows": len(flat_for_slug),
                "flat_query_allowed_rows": flat_allowed,
                "flat_product_query_allowed_rows": flat_product_allowed,
                "analysis_gate": analysis.get("camera_e2e_analysis_gate", ""),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
                "field_mesh_pass_fraction": safe_ratio(mesh.get("field_pass_points"), mesh.get("field_required_points")),
                "crosstalk_mesh_pass_fraction": safe_ratio(mesh.get("crosstalk_pass_points"), mesh.get("crosstalk_required_points")),
                "material_rows": sensor.get("material_row_count", ""),
                "electrical_rows": sensor.get("electrical_row_count", ""),
                "module_rows": sensor.get("module_field_row_count", ""),
                "crosstalk_support_gate_counts": json.dumps(support_gate_counts, sort_keys=True),
                "crosstalk_support_row_count": sensor.get("crosstalk_support_row_count", ""),
                "crosstalk_product_candidate_row_count": sensor.get("crosstalk_product_candidate_row_count", ""),
                "lut_trust_class": trust.get("trust_class", ""),
                "lut_trust_evidence_score_0_100": trust.get("evidence_confidence_score_0_100", ""),
                "lut_trust_product_score_0_100": trust.get("product_calibration_score_0_100", ""),
                "product_closure_status": product_closure.get("product_closure_status", ""),
                "product_closure_measured_input_blocker_count": product_closure.get("measured_input_blocker_count", ""),
                "product_closure_measured_calibration_blocker_count": product_closure.get("measured_calibration_blocker_count", ""),
                "product_closure_field_solver_row_count": product_closure.get("field_solver_row_count", ""),
                "product_closure_crosstalk_product_primary_row_count": product_closure.get("crosstalk_product_primary_row_count", ""),
                "product_closure_crosstalk_support_discovery_row_count": product_closure.get("crosstalk_support_discovery_row_count", ""),
                "product_closure_first_action_class": product_closure.get("first_action_class", ""),
                "product_closure_first_action_type": product_closure.get("first_action_type", ""),
                "product_closure_first_action_source": product_closure.get("first_action_source", ""),
                "product_closure_first_action_local_feasibility": product_closure.get("first_action_local_feasibility", ""),
                "acceptance_gate": "PASS" if row_pass else "FAIL",
                "product_ready": product_ready,
                "primary_blockers": primary_blockers,
            }
        )

    sensor_count_pass = expected_sensor_count == len(sensor_rows) and expected_sensor_count > 0
    check_rows.append(
        check_row(
            "sensor_count_matches_package",
            "package",
            sensor_count_pass,
            "PASS" if sensor_count_pass else "FAIL",
            {"expected": expected_sensor_count, "actual": len(sensor_rows)},
            "Regenerate package and consumer bundle.",
        )
    )
    failed_sensors = [row["slug"] for row in sensor_rows if row.get("acceptance_gate") != "PASS"]
    check_rows.append(
        check_row(
            "per_sensor_acceptance",
            "all_sensors",
            not failed_sensors,
            "PASS" if not failed_sensors else "FAIL",
            {"failed_sensors": failed_sensors, "sensor_count": len(sensor_rows)},
            "Inspect per-sensor rows in camera_e2e_objective_acceptance_sensors.csv.",
        )
    )

    error_count = sum(1 for row in check_rows if not boolish(row.get("pass")))
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    if error_count:
        status = "FAIL"
    elif product_ready_count:
        status = "PRODUCT_OBJECTIVE_ACCEPTED"
    else:
        status = "RESEARCH_OBJECTIVE_ACCEPTED_PRODUCT_BLOCKED"

    checks_csv = output_dir / "camera_e2e_objective_acceptance_checks.csv"
    sensors_csv = output_dir / "camera_e2e_objective_acceptance_sensors.csv"
    report_json = output_dir / "camera_e2e_objective_acceptance.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_objective_acceptance_audit_v1",
        "artifact_role": "camera_e2e_objective_completion_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "requirement_count_per_sensor": expected_requirement_count,
        "coverage_row_count": len(coverage_rows),
        "consumer_query_row_count": len(consumer_query_rows),
        "consumer_query_allowed_count": safe_int(consumer_query.get("allowed_query_count")),
        "product_query_allowed_count": safe_int(product_query.get("allowed_query_count")),
        "flat_sensor_bundle_status": flat_sensor_bundle.get("validation", {}).get("status", ""),
        "flat_sensor_bundle_sensor_count": flat_sensor_bundle.get("sensor_count", 0),
        "flat_sensor_bundle_total_embedded_row_count": flat_sensor_bundle.get("total_embedded_row_count", 0),
        "flat_query_row_count": len(flat_query_rows),
        "flat_query_allowed_count": safe_int(flat_sensor_query.get("allowed_query_count")),
        "flat_product_query_allowed_count": safe_int(flat_product_query.get("allowed_query_count")),
        "analysis_report_status": analysis_report.get("validation", {}).get("status", ""),
        "analysis_report_channel_row_count": analysis_report.get("channel_row_count", 0),
        "analysis_report_check_channel_row_count": analysis_report.get("check_channel_row_count", 0),
        "analysis_report_mesh_confidence_class_counts": analysis_report.get("mesh_confidence_class_counts", {}),
        "mesh_confidence_status": mesh_confidence.get("status", ""),
        "mesh_confidence_field_pass_total": mesh_confidence.get("field_pass_total", 0),
        "mesh_confidence_field_required_total": mesh_confidence.get("field_required_total", 0),
        "mesh_confidence_crosstalk_pass_total": mesh_confidence.get("crosstalk_pass_total", 0),
        "mesh_confidence_crosstalk_required_total": mesh_confidence.get("crosstalk_required_total", 0),
        "mesh_confidence_class_counts": mesh_confidence.get("confidence_class_counts", {}),
        "lut_trust_status": trust_assessment.get("validation", {}).get("status", ""),
        "lut_trust_sensor_count": trust_assessment.get("sensor_count", 0),
        "lut_trust_class_counts": trust_assessment.get("trust_class_counts", {}),
        "product_closure_summary_status": product_closure_summary.get("validation", {}).get("status", ""),
        "product_closure_summary_sensor_count": product_closure_summary.get("sensor_count", 0),
        "product_closure_summary_domain_row_count": product_closure_summary.get("domain_row_count", 0),
        "product_closure_summary_product_ready_count": product_closure_summary.get("product_ready_count", 0),
        "usage_policy_status": usage_policy.get("validation", {}).get("status", ""),
        "usage_policy_sensor_policy_row_count": usage_policy.get("sensor_policy_row_count", 0),
        "usage_policy_domain_policy_row_count": usage_policy.get("domain_policy_row_count", 0),
        "usage_policy_strict_product_filter_row_count": usage_policy.get("strict_product_filter_row_count", 0),
        "usage_policy_product_ingest_allowed_count": usage_policy.get("product_ingest_allowed_count", 0),
        "adapter_examples_status": adapter_examples.get("validation", {}).get("status", ""),
        "adapter_examples_sensor_count": adapter_examples.get("sensor_count", 0),
        "adapter_examples_file_count": adapter_examples.get("example_file_count", 0),
        "adapter_examples_product_allowed_query_count": adapter_examples.get("product_allowed_query_count", 0),
        "adapter_smoke_status": adapter_smoke.get("validation", {}).get("status", ""),
        "adapter_smoke_sensor_count": adapter_smoke.get("sensor_count", 0),
        "adapter_smoke_total_research_allowed_query_count": adapter_smoke.get("total_research_allowed_query_count", 0),
        "adapter_smoke_total_product_allowed_query_count": adapter_smoke.get("total_product_allowed_query_count", 0),
        "objective_trace_status": objective_trace.get("validation", {}).get("status", ""),
        "objective_trace_sensor_count": objective_trace.get("sensor_count", 0),
        "objective_trace_requirement_count_per_sensor": objective_trace.get("requirement_count_per_sensor", 0),
        "objective_trace_row_count": objective_trace.get("trace_row_count", 0),
        "objective_trace_gate_counts": objective_trace.get("trace_gate_counts", {}),
        "objective_trace_product_ready_count": objective_trace.get("product_ready_count", 0),
        "product_ready_count": product_ready_count,
        "validation": {
            "schema": "camera_e2e_objective_acceptance_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in check_rows if not boolish(row.get("pass"))],
        },
        "outputs": {
            "json": repo_rel(report_json),
            "checks_csv": repo_rel(checks_csv),
            "sensors_csv": repo_rel(sensors_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(checks_csv, check_rows, CHECK_COLUMNS)
    write_csv(sensors_csv, sensor_rows, SENSOR_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, check_rows, sensor_rows)
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


def write_html(path: Path, payload: dict[str, Any], check_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]]) -> None:
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
    issue_html = html_table(validation.get("issues", []), CHECK_COLUMNS) if validation.get("issues") else '<p class="pass">No acceptance issues.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Objective Acceptance Audit</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Objective Acceptance Audit</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This verifies research-package completeness and product-gate blocking against the active objective.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">acceptance status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_count_per_sensor", 0))}</div><div class="muted">requirements per sensor</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("flat_query_allowed_count", 0))}</div><div class="muted">flat research allowed rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("mesh_confidence_field_pass_total", 0))}/{html_cell(payload.get("mesh_confidence_field_required_total", 0))}</div><div class="muted">field mesh PASS</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("mesh_confidence_crosstalk_pass_total", 0))}/{html_cell(payload.get("mesh_confidence_crosstalk_required_total", 0))}</div><div class="muted">crosstalk mesh PASS</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_closure_summary_sensor_count", 0))}</div><div class="muted">closure-summary sensors</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("usage_policy_strict_product_filter_row_count", 0))}</div><div class="muted">strict product policy rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("adapter_examples_file_count", 0))}</div><div class="muted">adapter examples</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("adapter_smoke_total_research_allowed_query_count", 0))}</div><div class="muted">adapter research rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("adapter_smoke_total_product_allowed_query_count", 0))}</div><div class="muted">adapter product rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("objective_trace_row_count", 0))}</div><div class="muted">objective trace rows</div></div>
</div>
<h2>Policy</h2>
<p>PASS here means the package is complete enough for CameraE2E research loading and query tests. Product use remains blocked until measured/calibrated data and convergence gates pass.</p>
<h2>Issues</h2>{issue_html}
<h2>Checks</h2>{html_table(check_rows, CHECK_COLUMNS)}
<h2>Per-Sensor Acceptance</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
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
    outputs["camera_e2e_objective_acceptance_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_objective_acceptance_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_objective_acceptance_sensors_csv"] = payload["outputs"]["sensors_csv"]
    outputs["camera_e2e_objective_acceptance_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_objective_acceptance"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "requirement_count_per_sensor": payload["requirement_count_per_sensor"],
        "product_ready_count": payload["product_ready_count"],
        "product_closure_summary_status": payload.get("product_closure_summary_status", ""),
        "product_closure_summary_sensor_count": payload.get("product_closure_summary_sensor_count", 0),
        "usage_policy_status": payload.get("usage_policy_status", ""),
        "usage_policy_strict_product_filter_row_count": payload.get("usage_policy_strict_product_filter_row_count", 0),
        "adapter_examples_status": payload.get("adapter_examples_status", ""),
        "adapter_examples_file_count": payload.get("adapter_examples_file_count", 0),
        "adapter_smoke_status": payload.get("adapter_smoke_status", ""),
        "adapter_smoke_total_research_allowed_query_count": payload.get("adapter_smoke_total_research_allowed_query_count", 0),
        "adapter_smoke_total_product_allowed_query_count": payload.get("adapter_smoke_total_product_allowed_query_count", 0),
        "objective_trace_status": payload.get("objective_trace_status", ""),
        "objective_trace_row_count": payload.get("objective_trace_row_count", 0),
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_audit(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "requirement_count_per_sensor": payload["requirement_count_per_sensor"],
                "consumer_query_allowed_count": payload["consumer_query_allowed_count"],
                "product_query_allowed_count": payload["product_query_allowed_count"],
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
