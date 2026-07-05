#!/usr/bin/env python3
"""Export concrete CameraE2E adapter examples per sensor.

The examples are not a new source of LUT values. They are a downstream-facing
load recipe that binds usage policy, flat per-sensor JSON, representative query
summaries, and strict product-mode blocking evidence.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_adapter_examples"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_profile",
    "research_mode_allowed",
    "product_mode_allowed",
    "recommended_runtime_filter_id",
    "flat_sensor_json",
    "adapter_example_json",
    "research_summary_row_count",
    "research_allowed_query_count",
    "product_summary_row_count",
    "product_allowed_query_count",
    "runtime_row_count",
    "kernel_row_count",
    "electrical_row_count",
    "readout_row_count",
    "module_field_row_count",
    "objective_fulfillment_row_count",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "first_product_closure_action",
    "adapter_gate",
    "adapter_notes",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
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


def sum_int(rows: list[dict[str, str]], key: str) -> int:
    return sum(safe_int(row.get(key)) for row in rows)


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def representative_summaries(rows: list[dict[str, str]], limit: int = 6) -> list[dict[str, Any]]:
    def score(row: dict[str, str]) -> tuple[int, float, str]:
        color = str(row.get("color_channel", ""))
        wave = safe_float(row.get("wavelength_nm"))
        preferred = {
            ("green", 550.0): 0,
            ("red", 620.0): 1,
            ("blue", 450.0): 2,
        }.get((color, wave), 9)
        return preferred, abs(wave - 550.0), color

    compact: list[dict[str, Any]] = []
    for row in sorted(rows, key=score)[:limit]:
        compact.append(
            {
                "wavelength_nm": row.get("wavelength_nm", ""),
                "color_channel": row.get("color_channel", ""),
                "query_count": row.get("query_count", ""),
                "allowed_count": row.get("allowed_count", ""),
                "summary_gate": row.get("summary_gate", ""),
                "center_signal_e": row.get("center_signal_e", ""),
                "edge_to_center_signal_ratio": row.get("edge_to_center_signal_ratio", ""),
                "max_output_crosstalk_fraction": row.get("max_output_crosstalk_fraction", ""),
                "min_snr_db": row.get("min_snr_db", ""),
                "max_snr_db": row.get("max_snr_db", ""),
                "mesh_confidence_class": row.get("mesh_confidence_class", ""),
                "camera_e2e_use_scope": row.get("camera_e2e_use_scope", ""),
            }
        )
    return compact


def build_example_payload(
    *,
    package_dir: Path,
    usage_policy: dict[str, Any],
    policy_row: dict[str, str],
    flat_row: dict[str, str],
    research_rows: list[dict[str, str]],
    product_rows: list[dict[str, str]],
) -> dict[str, Any]:
    slug = policy_row.get("slug", "")
    flat_json = flat_row.get("flat_sensor_json", "")
    research_allowed = sum_int(research_rows, "allowed_count")
    product_allowed = sum_int(product_rows, "allowed_count")
    flat_abs = abs_from_repo(flat_json)
    flat_model = read_json(flat_abs)
    sensor = flat_model.get("sensor", {})
    gates = flat_model.get("gates", {})
    row_counts = flat_model.get("row_counts", {})
    objective_rows = flat_model.get("objective_fulfillment", {}).get("requirement_rows", [])
    if not isinstance(objective_rows, list):
        objective_rows = []
    return {
        "schema": "camera_e2e_sensor_adapter_example_v1",
        "artifact_role": "camera_e2e_per_sensor_load_recipe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor": {
            "slug": slug,
            "code": policy_row.get("code", ""),
            "manufacturer": policy_row.get("manufacturer", ""),
            "device_name": policy_row.get("device_name", ""),
            "pixel_pitch_um": sensor.get("pixel_pitch_um", flat_row.get("pixel_pitch_um", "")),
            "cfa_pattern": sensor.get("cfa_pattern", flat_row.get("cfa_pattern", "")),
            "effective_ocl_mode": sensor.get("effective_ocl_mode", flat_row.get("effective_ocl_mode", "")),
        },
        "policy": {
            "camera_e2e_profile": policy_row.get("camera_e2e_profile", ""),
            "camera_e2e_use_scope": policy_row.get("camera_e2e_use_scope", ""),
            "allowed_camera_e2e_modes": policy_row.get("allowed_camera_e2e_modes", ""),
            "blocked_camera_e2e_modes": policy_row.get("blocked_camera_e2e_modes", ""),
            "recommended_runtime_filter_id": policy_row.get("recommended_runtime_filter_id", ""),
            "product_ingest_allowed": boolish(policy_row.get("product_ingest_allowed")),
            "research_ingest_allowed": boolish(policy_row.get("research_ingest_allowed")),
            "strict_product_filter_row_count": usage_policy.get("strict_product_filter_row_count", 0),
            "loader_contract": usage_policy.get("loader_contract", {}),
        },
        "load_paths": {
            "usage_policy_json": repo_rel(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy.json"),
            "usage_policy_runtime_filters_csv": repo_rel(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy_runtime_filters.csv"),
            "flat_bundle_json": repo_rel(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json"),
            "flat_sensor_json": flat_json,
            "research_query_summary_csv": repo_rel(package_dir / "camera_e2e_flat_sensor_query" / "camera_e2e_flat_sensor_query_summary.csv"),
            "product_query_summary_csv": repo_rel(package_dir / "camera_e2e_flat_sensor_query_product_probe" / "camera_e2e_flat_sensor_query_summary.csv"),
        },
        "camera_e2e_section_map": {
            "optical_color": {
                "source": f"{flat_json}#/optical_color",
                "runtime_rows": row_counts.get("runtime", flat_row.get("runtime_row_count", "")),
                "kernel_rows": row_counts.get("kernel", flat_row.get("kernel_row_count", "")),
                "spectral_rows": row_counts.get("spectral", flat_row.get("spectral_row_count", "")),
                "material_rows": row_counts.get("material", flat_row.get("material_row_count", "")),
                "cfa_db_transmission_rows": row_counts.get("cfa_db_transmission", flat_row.get("cfa_db_transmission_row_count", "")),
            },
            "pixel_electrical": {
                "source": f"{flat_json}#/pixel_electrical",
                "electrical_rows": row_counts.get("electrical", flat_row.get("electrical_row_count", "")),
            },
            "readout_raw": {
                "source": f"{flat_json}#/readout_raw",
                "readout_rows": row_counts.get("readout", flat_row.get("readout_row_count", "")),
                "binning_rows": row_counts.get("binning", flat_row.get("binning_row_count", "")),
            },
            "module_coupling": {
                "source": f"{flat_json}#/module_coupling",
                "field_rows": row_counts.get("module_field", flat_row.get("module_field_row_count", "")),
            },
            "routing_and_gates": {
                "source": f"{flat_json}#/camera_e2e_routing",
                "mesh_confidence_class": policy_row.get("mesh_confidence_class", ""),
                "trust_class": policy_row.get("trust_class", ""),
                "product_gate": gates.get("product_gate", "FAIL"),
                "product_ready": gates.get("product_ready", False),
            },
        },
        "objective_fulfillment": {
            "row_count": len(objective_rows),
            "requirement_rows": objective_rows,
            "policy": (
                "This maps each requested CameraE2E item to the flat JSON pointer/table and the source/proxy/simulation gate. "
                "Do not strip these gates when loading research-mode rows."
            ),
        },
        "example_commands": {
            "research_query": (
                "python3 query_camera_e2e_flat_sensor_bundle.py "
                "--flat-bundle-json runs/camera_e2e_sensor_lut_package/camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_bundle.json "
                f"--output-dir runs/camera_e2e_sensor_lut_package/adapter_query_examples/{slug}/research "
                f"--slugs {slug} --field-x=-1,0,1 --field-z=-1,0,1 --wavelength-nm all --mode research"
            ),
            "product_block_probe": (
                "python3 query_camera_e2e_flat_sensor_bundle.py "
                "--flat-bundle-json runs/camera_e2e_sensor_lut_package/camera_e2e_flat_sensor_bundle/camera_e2e_flat_sensor_bundle.json "
                f"--output-dir runs/camera_e2e_sensor_lut_package/adapter_query_examples/{slug}/product_probe "
                f"--slugs {slug} --field-x=-1,0,1 --field-z=-1,0,1 --wavelength-nm all --mode product"
            ),
        },
        "representative_research_outputs": representative_summaries(research_rows),
        "product_block_evidence": {
            "product_summary_rows": len(product_rows),
            "product_allowed_query_count": product_allowed,
            "expected_product_allowed_query_count": 0,
            "representative_product_rows": representative_summaries(product_rows, limit=3),
        },
        "validation_expectation": {
            "research_allowed_query_count": research_allowed,
            "product_allowed_query_count": product_allowed,
            "adapter_gate": "PASS" if research_allowed > 0 and product_allowed == 0 and flat_abs.exists() else "FAIL",
            "product_ready": boolish(gates.get("product_ready")),
        },
        "required_before_product_use": policy_row.get("required_before_product_use", ""),
    }


def build_adapter_examples(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensor_output_dir = output_dir / "sensors"

    usage_policy = read_json(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy.json")
    policy_rows = read_csv_rows(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy_by_sensor.csv")
    flat_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_index.csv")
    research_summary_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_query" / "camera_e2e_flat_sensor_query_summary.csv")
    product_summary_rows = read_csv_rows(package_dir / "camera_e2e_flat_sensor_query_product_probe" / "camera_e2e_flat_sensor_query_summary.csv")

    flat_by_slug = index_by(flat_rows, "slug")
    research_by_slug = group_by(research_summary_rows, "slug")
    product_by_slug = group_by(product_summary_rows, "slug")

    sensor_rows: list[dict[str, Any]] = []
    example_files: list[str] = []
    missing_flat_json: list[str] = []
    missing_objective_map: list[str] = []
    research_blocked: list[str] = []
    product_open: list[str] = []

    for policy_row in policy_rows:
        slug = policy_row.get("slug", "")
        flat_row = flat_by_slug.get(slug, {})
        research_rows = research_by_slug.get(slug, [])
        product_rows = product_by_slug.get(slug, [])
        research_allowed = sum_int(research_rows, "allowed_count")
        product_allowed = sum_int(product_rows, "allowed_count")
        flat_json = flat_row.get("flat_sensor_json", "")
        flat_exists = abs_from_repo(flat_json).exists()
        if not flat_exists:
            missing_flat_json.append(slug)
        if research_allowed <= 0:
            research_blocked.append(slug)
        if product_allowed != 0 or boolish(policy_row.get("product_ingest_allowed")):
            product_open.append(slug)
        example_payload = build_example_payload(
            package_dir=package_dir,
            usage_policy=usage_policy,
            policy_row=policy_row,
            flat_row=flat_row,
            research_rows=research_rows,
            product_rows=product_rows,
        )
        objective_count = safe_int(example_payload.get("objective_fulfillment", {}).get("row_count"))
        if objective_count <= 0:
            missing_objective_map.append(slug)
        example_path = sensor_output_dir / f"{slug}_camera_e2e_adapter_example.json"
        write_json(example_path, example_payload)
        example_files.append(repo_rel(example_path))
        adapter_gate = example_payload.get("validation_expectation", {}).get("adapter_gate", "FAIL")
        sensor_rows.append(
            {
                "slug": slug,
                "code": policy_row.get("code", ""),
                "manufacturer": policy_row.get("manufacturer", ""),
                "device_name": policy_row.get("device_name", ""),
                "camera_e2e_profile": policy_row.get("camera_e2e_profile", ""),
                "research_mode_allowed": research_allowed > 0 and boolish(policy_row.get("research_ingest_allowed")),
                "product_mode_allowed": product_allowed > 0 or boolish(policy_row.get("product_ingest_allowed")),
                "recommended_runtime_filter_id": policy_row.get("recommended_runtime_filter_id", ""),
                "flat_sensor_json": flat_json,
                "adapter_example_json": repo_rel(example_path),
                "research_summary_row_count": len(research_rows),
                "research_allowed_query_count": research_allowed,
                "product_summary_row_count": len(product_rows),
                "product_allowed_query_count": product_allowed,
                "runtime_row_count": policy_row.get("runtime_row_count", flat_row.get("runtime_row_count", "")),
                "kernel_row_count": policy_row.get("kernel_row_count", flat_row.get("kernel_row_count", "")),
                "electrical_row_count": flat_row.get("electrical_row_count", ""),
                "readout_row_count": flat_row.get("readout_row_count", ""),
                "module_field_row_count": flat_row.get("module_field_row_count", ""),
                "objective_fulfillment_row_count": objective_count,
                "field_mesh_pass_points": policy_row.get("field_mesh_pass_points", ""),
                "field_mesh_required_points": policy_row.get("field_mesh_required_points", ""),
                "crosstalk_mesh_pass_points": policy_row.get("crosstalk_mesh_pass_points", ""),
                "crosstalk_mesh_required_points": policy_row.get("crosstalk_mesh_required_points", ""),
                "first_product_closure_action": policy_row.get("first_product_closure_action", ""),
                "adapter_gate": adapter_gate,
                "adapter_notes": policy_row.get("policy_reason", ""),
            }
        )

    adapter_failures = [row.get("slug", "") for row in sensor_rows if row.get("adapter_gate") != "PASS"]
    product_allowed_count = sum(safe_int(row.get("product_allowed_query_count")) for row in sensor_rows)
    checks = [
        check_row(
            "usage_policy_valid",
            usage_policy.get("schema") == "camera_e2e_usage_policy_v1" and bool(usage_policy.get("validation", {}).get("pass")),
            usage_policy.get("validation", {}).get("status", "MISSING"),
            {"strict_product_filter_row_count": usage_policy.get("strict_product_filter_row_count")},
            "Regenerate usage policy before adapter examples.",
        ),
        check_row(
            "adapter_examples_present",
            len(sensor_rows) > 0 and len(example_files) == len(sensor_rows),
            "PASS" if len(sensor_rows) > 0 and len(example_files) == len(sensor_rows) else "FAIL",
            {"sensor_count": len(sensor_rows), "example_file_count": len(example_files)},
            "Generate one adapter example JSON per sensor.",
        ),
        check_row(
            "flat_sensor_json_loadable",
            not missing_flat_json,
            "PASS" if not missing_flat_json else "FAIL",
            {"missing_flat_json_slugs": missing_flat_json},
            "Regenerate flat sensor bundle.",
        ),
        check_row(
            "objective_fulfillment_map_present",
            not missing_objective_map,
            "PASS" if not missing_objective_map else "FAIL",
            {"missing_objective_map_slugs": missing_objective_map},
            "Regenerate flat sensor bundle with objective_fulfillment rows.",
        ),
        check_row(
            "research_examples_allowed",
            not research_blocked,
            "PASS" if not research_blocked else "FAIL",
            {"research_blocked_slugs": research_blocked},
            "Run flat sensor research query before adapter examples.",
        ),
        check_row(
            "product_examples_blocked",
            product_allowed_count == 0 and not product_open,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_allowed_count == 0 and not product_open else "FAIL",
            {"product_allowed_query_count": product_allowed_count, "product_open_slugs": product_open},
            "Keep product examples blocked until product gates pass.",
        ),
        check_row(
            "per_sensor_adapter_gates",
            not adapter_failures,
            "PASS" if not adapter_failures else "FAIL",
            {"adapter_failures": adapter_failures},
            "Inspect per-sensor adapter rows.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_ADAPTER_EXAMPLES_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_json = output_dir / "camera_e2e_adapter_examples.json"
    by_sensor_csv = output_dir / "camera_e2e_adapter_examples_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_adapter_examples_checks.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_adapter_examples_v1",
        "artifact_role": "camera_e2e_downstream_adapter_examples",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "example_file_count": len(example_files),
        "product_allowed_query_count": product_allowed_count,
        "profile_counts": dict(sorted(Counter(str(row.get("camera_e2e_profile", "")) for row in sensor_rows).items())),
        "validation": {
            "schema": "camera_e2e_adapter_examples_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "usage": {
            "entrypoint": "Load camera_e2e_usage_policy first, then choose one per-sensor adapter example JSON.",
            "research_mode": "Use the research_query command and preserve row-level query_gate/product_ready values.",
            "product_mode": "Use the product_block_probe command only as a fail-closed test until product gates open.",
        },
        "example_json_files": example_files,
        "outputs": {
            "json": repo_rel(examples_json),
            "by_sensor_csv": repo_rel(by_sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
            "sensor_examples_dir": repo_rel(sensor_output_dir),
        },
    }
    write_csv(by_sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(examples_json, payload)
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
<title>CameraE2E Adapter Examples</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Adapter Examples</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Per-sensor load recipes bind usage policy, flat sensor JSON, query summaries, and product-blocking evidence.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">adapter status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("example_file_count", 0))}</div><div class="muted">example JSON files</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_allowed_query_count", 0))}</div><div class="muted">product-allowed query rows</div></div>
</div>
<h2>Usage</h2>
<p><strong>Entrypoint:</strong> {html_cell(payload.get("usage", {}).get("entrypoint", ""))}</p>
<p><strong>Research:</strong> {html_cell(payload.get("usage", {}).get("research_mode", ""))}</p>
<p><strong>Product:</strong> {html_cell(payload.get("usage", {}).get("product_mode", ""))}</p>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Per-Sensor Adapter Rows</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_adapter_examples_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_adapter_examples_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_adapter_examples_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_adapter_examples_html"] = payload["outputs"]["html"]
    outputs["camera_e2e_adapter_examples_sensor_dir"] = payload["outputs"]["sensor_examples_dir"]
    package["latest_camera_e2e_adapter_examples"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "example_file_count": payload["example_file_count"],
        "product_allowed_query_count": payload["product_allowed_query_count"],
        "profile_counts": payload["profile_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_adapter_examples(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "example_file_count": payload["example_file_count"],
                "product_allowed_query_count": payload["product_allowed_query_count"],
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
