#!/usr/bin/env python3
"""Export a strict CameraE2E import contract from flat per-sensor models.

This artifact is the downstream adapter contract. It opens every flat sensor
JSON, resolves every objective_fulfillment JSON pointer, and emits one compact
requirement row per sensor per CameraE2E objective item.

It does not create new simulation values or promote research/proxy data to
product accuracy. Product import remains blocked unless the embedded product
gates become true.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_import_contract"

REQUIREMENT_COLUMNS = [
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
    "pointer_resolves",
    "resolved_value_type",
    "resolved_value_count",
    "primary_loader_table",
    "secondary_loader_tables",
    "research_gate",
    "product_gate",
    "source_integrity_gate",
    "lut_source_class",
    "calculation_method",
    "source_priority",
    "solver_dependency",
    "external_info_dependency",
    "proxy_dependency",
    "structure_specialization",
    "primary_uncertainty_quantity",
    "primary_uncertainty_min",
    "primary_uncertainty_max",
    "primary_uncertainty_unit",
    "uncertainty_camera_e2e_use",
    "uncertainty_product_gate",
    "recommended_camera_e2e_use",
    "source_artifacts",
    "primary_blocker",
    "next_action",
    "product_ready",
    "import_research_allowed",
    "import_product_allowed",
    "import_gate",
    "import_notes",
]

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "effective_ocl_mode",
    "flat_sensor_json",
    "sensor_import_contract_json",
    "requirement_count",
    "pointer_resolved_count",
    "research_allowed_requirement_count",
    "product_allowed_requirement_count",
    "product_ready",
    "domain_counts",
    "research_gate_counts",
    "product_gate_counts",
    "source_integrity_gate_counts",
    "source_class_counts",
    "import_gate",
    "import_notes",
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


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def gate(value: Any, default: str = "MISSING") -> str:
    text = str(value if value is not None else "").strip().upper()
    return text or default


def json_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_pointer(payload: Any, pointer: str) -> tuple[bool, Any, str]:
    """Resolve normal JSON pointer plus list-column shorthand.

    A pointer such as /module_coupling/module_field_lut/cra_x_deg resolves to a
    list of cra_x_deg values when module_field_lut is a list of dict rows.
    """

    if pointer == "":
        return True, payload, ""
    if not pointer.startswith("/"):
        return False, None, "pointer does not start with /"
    current: Any = payload
    for raw_part in pointer.strip("/").split("/"):
        part = json_pointer_token(raw_part)
        if isinstance(current, dict):
            if part not in current:
                return False, None, f"dict key missing: {part}"
            current = current[part]
            continue
        if isinstance(current, list):
            if part.isdigit():
                index = int(part)
                if index >= len(current):
                    return False, None, f"list index out of range: {index}"
                current = current[index]
                continue
            if all(isinstance(row, dict) for row in current):
                values = [row.get(part) for row in current if part in row]
                if not values:
                    return False, None, f"list column missing: {part}"
                current = values
                continue
            return False, None, f"cannot select non-numeric token {part} from list"
        return False, None, f"cannot descend into {type(current).__name__}"
    return True, current, ""


def value_summary(value: Any) -> tuple[str, int]:
    if isinstance(value, list):
        return "list", len(value)
    if isinstance(value, dict):
        return "dict", len(value)
    if value in ("", None):
        return "empty_scalar", 0
    return "scalar", 1


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def import_product_allowed(row: dict[str, Any], product_ready: bool) -> bool:
    return (
        product_ready
        and gate(row.get("product_gate")) == "PASS"
        and gate(row.get("source_integrity_gate")) == "PASS"
        and gate(row.get("uncertainty_product_gate")) == "PASS"
    )


def import_research_allowed(row: dict[str, Any], pointer_resolves: bool) -> bool:
    return pointer_resolves and gate(row.get("research_gate")) in {"PASS", "CHECK", "N/A"}


def build_requirement_row(
    *,
    sensor: dict[str, Any],
    flat_payload: dict[str, Any],
    flat_json: str,
    requirement: dict[str, Any],
    product_ready: bool,
) -> dict[str, Any]:
    pointer = str(requirement.get("flat_json_pointer", ""))
    pointer_ok, resolved, pointer_error = resolve_pointer(flat_payload, pointer)
    resolved_type, resolved_count = value_summary(resolved) if pointer_ok else ("missing", 0)
    row: dict[str, Any] = {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "domain": requirement.get("domain", ""),
        "requirement_id": requirement.get("requirement_id", ""),
        "requirement": requirement.get("requirement", ""),
        "camera_e2e_use": requirement.get("camera_e2e_use", ""),
        "camera_e2e_loader_section": requirement.get("camera_e2e_loader_section", ""),
        "flat_json_pointer": pointer,
        "pointer_resolves": pointer_ok,
        "resolved_value_type": resolved_type,
        "resolved_value_count": resolved_count,
        "primary_loader_table": requirement.get("primary_loader_table", ""),
        "secondary_loader_tables": requirement.get("secondary_loader_tables", ""),
        "research_gate": requirement.get("research_gate", ""),
        "product_gate": requirement.get("product_gate", ""),
        "source_integrity_gate": requirement.get("source_integrity_gate", ""),
        "lut_source_class": requirement.get("lut_source_class", ""),
        "calculation_method": requirement.get("calculation_method", ""),
        "source_priority": requirement.get("source_priority", ""),
        "solver_dependency": requirement.get("solver_dependency", ""),
        "external_info_dependency": requirement.get("external_info_dependency", ""),
        "proxy_dependency": requirement.get("proxy_dependency", ""),
        "structure_specialization": requirement.get("structure_specialization", ""),
        "primary_uncertainty_quantity": requirement.get("primary_uncertainty_quantity", ""),
        "primary_uncertainty_min": requirement.get("primary_uncertainty_min", ""),
        "primary_uncertainty_max": requirement.get("primary_uncertainty_max", ""),
        "primary_uncertainty_unit": requirement.get("primary_uncertainty_unit", ""),
        "uncertainty_camera_e2e_use": requirement.get("uncertainty_camera_e2e_use", ""),
        "uncertainty_product_gate": requirement.get("uncertainty_product_gate", ""),
        "recommended_camera_e2e_use": requirement.get("recommended_camera_e2e_use", ""),
        "source_artifacts": requirement.get("source_artifacts", ""),
        "primary_blocker": requirement.get("primary_blocker", ""),
        "next_action": requirement.get("next_action", ""),
        "product_ready": product_ready,
    }
    row["import_research_allowed"] = import_research_allowed(row, pointer_ok)
    row["import_product_allowed"] = import_product_allowed(row, product_ready)
    notes = []
    if not pointer_ok:
        notes.append(pointer_error)
    if not row["import_product_allowed"]:
        notes.append(row.get("primary_blocker") or "product gate is not PASS")
    row["import_gate"] = "PASS" if pointer_ok and row["import_research_allowed"] else "FAIL"
    row["import_notes"] = "; ".join(note for note in notes if note)
    return row


def build_sensor_contract(
    *,
    flat_payload: dict[str, Any],
    flat_json: str,
    sensor_output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    sensor = flat_payload.get("sensor", {}) if isinstance(flat_payload.get("sensor"), dict) else {}
    gates = flat_payload.get("gates", {}) if isinstance(flat_payload.get("gates"), dict) else {}
    product_ready = boolish(gates.get("product_ready"))
    objective = flat_payload.get("objective_fulfillment", {}) if isinstance(flat_payload.get("objective_fulfillment"), dict) else {}
    objective_rows = objective.get("requirement_rows", []) if isinstance(objective.get("requirement_rows"), list) else []
    requirement_rows = [
        build_requirement_row(
            sensor=sensor,
            flat_payload=flat_payload,
            flat_json=flat_json,
            requirement=requirement,
            product_ready=product_ready,
        )
        for requirement in objective_rows
    ]
    domain_sections: dict[str, list[str]] = defaultdict(list)
    for row in requirement_rows:
        domain_sections[str(row.get("domain", ""))].append(str(row.get("requirement_id", "")))
    unresolved = [row for row in requirement_rows if not boolish(row.get("pointer_resolves"))]
    product_allowed = [row for row in requirement_rows if boolish(row.get("import_product_allowed"))]
    research_allowed = [row for row in requirement_rows if boolish(row.get("import_research_allowed"))]
    source_counts = Counter(str(row.get("lut_source_class", "")) for row in requirement_rows if str(row.get("lut_source_class", "")))
    contract = {
        "schema": "camera_e2e_sensor_import_contract_v1",
        "artifact_role": "per_sensor_camera_e2e_import_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flat_sensor_json": flat_json,
        "sensor": sensor,
        "gates": {
            "research_import_gate": "PASS" if not unresolved and research_allowed else "FAIL",
            "product_import_gate": "PASS" if product_ready and product_allowed and len(product_allowed) == len(requirement_rows) else "FAIL",
            "product_ready": product_ready,
            "requirement_count": len(requirement_rows),
            "pointer_resolved_count": len(requirement_rows) - len(unresolved),
            "research_allowed_requirement_count": len(research_allowed),
            "product_allowed_requirement_count": len(product_allowed),
            "research_gate_counts": dict(sorted(Counter(str(row.get("research_gate", "")) for row in requirement_rows).items())),
            "product_gate_counts": dict(sorted(Counter(str(row.get("product_gate", "")) for row in requirement_rows).items())),
            "source_integrity_gate_counts": dict(sorted(Counter(str(row.get("source_integrity_gate", "")) for row in requirement_rows).items())),
            "source_class_counts": dict(sorted(source_counts.items())),
        },
        "domain_sections": dict(sorted(domain_sections.items())),
        "requirement_rows": requirement_rows,
        "policy": {
            "research_use": "Allowed only when research_import_gate is PASS and row-level source/gate/uncertainty fields are propagated.",
            "product_use": "Blocked until every requirement row has product/source/uncertainty PASS and product_ready is true.",
            "important_limitation": "This import contract verifies loadability and gate propagation, not physical product accuracy.",
        },
    }
    write_json(sensor_output_path, contract)
    summary = {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
        "cfa_pattern": sensor.get("cfa_pattern", ""),
        "effective_ocl_mode": sensor.get("effective_ocl_mode", ""),
        "flat_sensor_json": flat_json,
        "sensor_import_contract_json": repo_rel(sensor_output_path),
        "requirement_count": len(requirement_rows),
        "pointer_resolved_count": len(requirement_rows) - len(unresolved),
        "research_allowed_requirement_count": len(research_allowed),
        "product_allowed_requirement_count": len(product_allowed),
        "product_ready": product_ready,
        "domain_counts": dict(sorted(Counter(str(row.get("domain", "")) for row in requirement_rows).items())),
        "research_gate_counts": contract["gates"]["research_gate_counts"],
        "product_gate_counts": contract["gates"]["product_gate_counts"],
        "source_integrity_gate_counts": contract["gates"]["source_integrity_gate_counts"],
        "source_class_counts": contract["gates"]["source_class_counts"],
        "import_gate": contract["gates"]["research_import_gate"],
        "import_notes": "; ".join(sorted({row.get("import_notes", "") for row in requirement_rows if row.get("import_notes")}))[:1200],
    }
    return contract, requirement_rows, summary


def build_import_contract(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    sensor_dir = output_dir / "sensors"
    flat_bundle_json = args.flat_bundle_json.resolve() if args.flat_bundle_json else package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json"
    flat_bundle = read_json(flat_bundle_json)
    sensor_files = [str(path) for path in flat_bundle.get("sensor_model_json_files", [])]

    requirement_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []
    contract_files: list[str] = []
    invalid_flat_models: list[str] = []
    missing_sensor_files: list[str] = []

    for flat_json in sensor_files:
        flat_path = abs_from_repo(flat_json)
        if not flat_path.exists():
            missing_sensor_files.append(flat_json)
            continue
        flat_payload = read_json(flat_path)
        if flat_payload.get("schema") != "camera_e2e_flat_sensor_model_v1":
            invalid_flat_models.append(flat_json)
            continue
        slug = str(flat_payload.get("sensor", {}).get("slug", flat_path.stem))
        sensor_output_path = sensor_dir / f"{slug}_camera_e2e_import_contract.json"
        _contract, rows, summary = build_sensor_contract(
            flat_payload=flat_payload,
            flat_json=flat_json,
            sensor_output_path=sensor_output_path,
        )
        requirement_rows.extend(rows)
        sensor_rows.append(summary)
        contract_files.append(repo_rel(sensor_output_path))

    unresolved = [row for row in requirement_rows if not boolish(row.get("pointer_resolves"))]
    research_blocked = [row for row in requirement_rows if not boolish(row.get("import_research_allowed"))]
    product_allowed_count = sum(1 for row in requirement_rows if boolish(row.get("import_product_allowed")))
    checks = [
        check_row(
            "flat_bundle_valid",
            flat_bundle.get("schema") == "camera_e2e_flat_sensor_bundle_v1" and bool(flat_bundle.get("validation", {}).get("pass")),
            flat_bundle.get("validation", {}).get("status", "MISSING"),
            {"flat_bundle_json": repo_rel(flat_bundle_json), "sensor_count": flat_bundle.get("sensor_count", 0)},
            "Regenerate flat sensor bundle.",
        ),
        check_row(
            "all_flat_sensor_files_present",
            not missing_sensor_files,
            "PASS" if not missing_sensor_files else "FAIL",
            {"missing_sensor_files": missing_sensor_files},
            "Regenerate flat sensor bundle index.",
        ),
        check_row(
            "flat_sensor_schemas_valid",
            not invalid_flat_models,
            "PASS" if not invalid_flat_models else "FAIL",
            {"invalid_flat_models": invalid_flat_models},
            "Regenerate invalid per-sensor flat JSON files.",
        ),
        check_row(
            "per_sensor_import_contracts_present",
            len(sensor_rows) > 0 and len(sensor_rows) == int(flat_bundle.get("sensor_count", 0) or len(sensor_rows)),
            "PASS" if len(sensor_rows) > 0 and len(sensor_rows) == int(flat_bundle.get("sensor_count", 0) or len(sensor_rows)) else "FAIL",
            {"sensor_rows": len(sensor_rows), "flat_bundle_sensor_count": flat_bundle.get("sensor_count", 0)},
            "Generate one import contract per flat sensor model.",
        ),
        check_row(
            "objective_pointers_resolve",
            not unresolved,
            "PASS" if not unresolved else "FAIL",
            {"unresolved_count": len(unresolved), "examples": unresolved[:5]},
            "Fix flat_json_pointer mappings or flat sensor JSON structure.",
        ),
        check_row(
            "research_import_loadable",
            not research_blocked and bool(requirement_rows),
            "PASS" if not research_blocked and bool(requirement_rows) else "FAIL",
            {"research_blocked_count": len(research_blocked), "requirement_row_count": len(requirement_rows)},
            "Every objective requirement should be loadable in research mode, including N/A rows.",
        ),
        check_row(
            "product_import_blocked",
            product_allowed_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_allowed_count == 0 else "FAIL",
            {"product_allowed_requirement_count": product_allowed_count},
            "Keep product import blocked until measured/calibrated product gates pass.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_IMPORT_CONTRACT_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_json = output_dir / "camera_e2e_import_contract.json"
    sensor_csv = output_dir / "camera_e2e_import_contract_by_sensor.csv"
    requirement_csv = output_dir / "camera_e2e_import_contract_by_requirement.csv"
    checks_csv = output_dir / "camera_e2e_import_contract_checks.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_import_contract_v1",
        "artifact_role": "camera_e2e_downstream_import_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "flat_bundle_json": repo_rel(flat_bundle_json),
        "sensor_count": len(sensor_rows),
        "requirement_row_count": len(requirement_rows),
        "requirement_count_per_sensor": max((int(row.get("requirement_count", 0) or 0) for row in sensor_rows), default=0),
        "pointer_resolved_count": sum(1 for row in requirement_rows if boolish(row.get("pointer_resolves"))),
        "research_allowed_requirement_count": sum(1 for row in requirement_rows if boolish(row.get("import_research_allowed"))),
        "product_allowed_requirement_count": product_allowed_count,
        "product_ready_count": sum(1 for row in sensor_rows if boolish(row.get("product_ready"))),
        "domain_counts": dict(sorted(Counter(str(row.get("domain", "")) for row in requirement_rows).items())),
        "source_class_counts": dict(sorted(Counter(str(row.get("lut_source_class", "")) for row in requirement_rows if str(row.get("lut_source_class", ""))).items())),
        "validation": {
            "schema": "camera_e2e_import_contract_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "policy": {
            "research_use": "Load by sensor_import_contract_json or by_requirement CSV; preserve gates and uncertainty fields.",
            "product_use": "Import must remain blocked until product_allowed_requirement_count equals requirement_row_count and product_ready_count equals sensor_count.",
        },
        "sensor_import_contract_json_files": contract_files,
        "outputs": {
            "json": repo_rel(contract_json),
            "by_sensor_csv": repo_rel(sensor_csv),
            "by_requirement_csv": repo_rel(requirement_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
            "sensor_contracts_dir": repo_rel(sensor_dir),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(requirement_csv, requirement_rows, REQUIREMENT_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(contract_json, payload)
    write_html(html_path, payload, sensor_rows, requirement_rows, checks)
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


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], requirement_rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
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
<title>CameraE2E Import Contract</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Import Contract</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This verifies every per-sensor objective pointer resolves from the flat CameraE2E model. It is an import/loadability contract, not product accuracy certification.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">contract status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_row_count", 0))}</div><div class="muted">requirement rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("pointer_resolved_count", 0))}</div><div class="muted">pointers resolved</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_allowed_requirement_count", 0))}</div><div class="muted">product-allowed requirements</div></div>
</div>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Per-Sensor Contract</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>Requirement Rows</h2>{html_table(requirement_rows, REQUIREMENT_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_import_contract_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_import_contract_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_import_contract_by_requirement_csv"] = payload["outputs"]["by_requirement_csv"]
    outputs["camera_e2e_import_contract_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_import_contract_html"] = payload["outputs"]["html"]
    outputs["camera_e2e_import_contract_sensor_dir"] = payload["outputs"]["sensor_contracts_dir"]
    package["latest_camera_e2e_import_contract"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "requirement_row_count": payload["requirement_row_count"],
        "pointer_resolved_count": payload["pointer_resolved_count"],
        "research_allowed_requirement_count": payload["research_allowed_requirement_count"],
        "product_allowed_requirement_count": payload["product_allowed_requirement_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flat-bundle-json", type=Path)
    return parser


def main() -> None:
    payload = build_import_contract(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "requirement_row_count": payload["requirement_row_count"],
                "pointer_resolved_count": payload["pointer_resolved_count"],
                "research_allowed_requirement_count": payload["research_allowed_requirement_count"],
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
