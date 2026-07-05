#!/usr/bin/env python3
"""Build a quantitative-evidence manifest for camera-system LUT readiness.

This is an audit/index artifact. It does not run heavy solvers and does not
upgrade proxy/reference data to product accuracy. It gathers existing optical,
crosstalk, electrical, field-LUT, and TCAD accuracy reports so downstream users
can see exactly which evidence is present and which gates still block product
LUT use.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "image_sensor_pixel_studio_reference.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_system_quantitative_evidence_reference"

EVIDENCE_COLUMNS = [
    "category",
    "id",
    "label",
    "status",
    "product_blocking",
    "research_blocking",
    "schema",
    "path",
    "summary",
]

BLOCKER_COLUMNS = [
    "category",
    "id",
    "status",
    "source",
    "details",
    "accuracy_blocking",
    "framework_blocking",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str | None, config_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    return (ROOT / path).resolve()


def status_from_bool(value: Any, missing: str = "CHECK") -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return missing


def add_evidence(
    rows: list[dict[str, Any]],
    *,
    category: str,
    evidence_id: str,
    label: str,
    status: str,
    path: Path | None,
    schema: str = "",
    summary: str = "",
    product_blocking: bool = False,
    research_blocking: bool = False,
) -> None:
    rows.append(
        {
            "category": category,
            "id": evidence_id,
            "label": label,
            "status": status,
            "product_blocking": bool(product_blocking),
            "research_blocking": bool(research_blocking),
            "schema": schema,
            "path": str(path) if path else "",
            "summary": summary,
        }
    )


def add_blocker(
    blockers: list[dict[str, Any]],
    *,
    category: str,
    blocker_id: str,
    status: str,
    source: str,
    details: str,
    accuracy_blocking: bool = True,
    framework_blocking: bool = False,
) -> None:
    blockers.append(
        {
            "category": category,
            "id": blocker_id,
            "status": status,
            "source": source,
            "details": details,
            "accuracy_blocking": bool(accuracy_blocking),
            "framework_blocking": bool(framework_blocking),
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def classify_camera_convergence(path: Path | None, evidence_id: str, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if not path or not path.exists():
        add_evidence(rows, category="optical_convergence", evidence_id=evidence_id, label=label, status="MISSING", path=path, product_blocking=True, research_blocking=False)
        add_blocker(blockers, category="optical_convergence", blocker_id=f"{evidence_id}_missing", status="MISSING", source=str(path or ""), details=f"{label} report is missing.")
        return rows[0], blockers
    data = read_json(path)
    schema = str(data.get("schema", ""))
    full_pass = bool(data.get("full_numerical_convergence_pass", False))
    grid_pass = bool(data.get("single_run_grid_qualification_pass", data.get("passed", False)))
    passed = bool(data.get("passed", False))
    status = "PASS" if full_pass or (passed and grid_pass and not data.get("unproven_axes")) else "CHECK" if passed or grid_pass else "FAIL"
    summary = (
        f"passed={passed}; full={full_pass}; spatial={data.get('spatial_convergence_pass')}; "
        f"time={data.get('time_convergence_pass')}; pml={data.get('pml_convergence_pass')}; "
        f"unproven_axes={','.join(data.get('unproven_axes') or [])}"
    )
    add_evidence(
        rows,
        category="optical_convergence",
        evidence_id=evidence_id,
        label=label,
        status=status,
        path=path,
        schema=schema,
        summary=summary,
        product_blocking=status != "PASS",
        research_blocking=False,
    )
    if status != "PASS":
        add_blocker(
            blockers,
            category="optical_convergence",
            blocker_id=f"{evidence_id}_not_full_pass",
            status=status,
            source=str(path),
            details=summary,
        )
    return rows[0], blockers


def classify_generic_status_report(
    path: Path | None,
    *,
    category: str,
    evidence_id: str,
    label: str,
    pass_values: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    pass_values = pass_values or {"PASS"}
    if not path or not path.exists():
        add_evidence(rows, category=category, evidence_id=evidence_id, label=label, status="MISSING", path=path, product_blocking=True)
        add_blocker(blockers, category=category, blocker_id=f"{evidence_id}_missing", status="MISSING", source=str(path or ""), details=f"{label} report is missing.")
        return rows[0], blockers
    data = read_json(path)
    raw_status = data.get("status")
    if raw_status is None and "passed" in data:
        status = status_from_bool(data.get("passed"))
    else:
        status = str(raw_status or "CHECK").upper()
    normalized = "PASS" if status in pass_values else "FAIL" if status == "FAIL" else "CHECK"
    checks = data.get("checks") if isinstance(data.get("checks"), list) else []
    fail_count = sum(1 for item in checks if str(item.get("status", "")).upper() == "FAIL")
    summary = f"status={status}; checks={len(checks)}; failed_checks={fail_count}"
    add_evidence(
        rows,
        category=category,
        evidence_id=evidence_id,
        label=label,
        status=normalized,
        path=path,
        schema=str(data.get("schema", "")),
        summary=summary,
        product_blocking=normalized != "PASS",
    )
    if normalized != "PASS":
        add_blocker(blockers, category=category, blocker_id=f"{evidence_id}_not_pass", status=normalized, source=str(path), details=summary)
    return rows[0], blockers


def classify_accuracy_gate(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if not path or not path.exists():
        add_evidence(evidence, category="accuracy_gate", evidence_id="tcad_accuracy_gate", label="TCAD accuracy gate", status="MISSING", path=path, product_blocking=True, research_blocking=True)
        add_blocker(blockers, category="accuracy_gate", blocker_id="tcad_accuracy_gate_missing", status="MISSING", source=str(path or ""), details="TCAD accuracy gate report is missing.", framework_blocking=True)
        return evidence, blockers, {"framework_ready": False, "accuracy_ready": False}
    data = read_json(path)
    framework_ready = bool(data.get("framework_ready", False))
    accuracy_ready = bool(data.get("accuracy_ready", False))
    add_evidence(
        evidence,
        category="accuracy_gate",
        evidence_id="framework_ready",
        label="Framework plumbing gate",
        status="PASS" if framework_ready else "FAIL",
        path=path,
        schema=str(data.get("schema", "")),
        summary=f"framework_blocking_failure_count={data.get('framework_blocking_failure_count')}",
        product_blocking=not framework_ready,
        research_blocking=not framework_ready,
    )
    add_evidence(
        evidence,
        category="accuracy_gate",
        evidence_id="product_accuracy_ready",
        label="Product accuracy gate",
        status="PASS" if accuracy_ready else "FAIL",
        path=path,
        schema=str(data.get("schema", "")),
        summary=f"accuracy_blocking_failure_count={data.get('accuracy_blocking_failure_count')}",
        product_blocking=not accuracy_ready,
        research_blocking=False,
    )
    for check in data.get("checks", []):
        status = str(check.get("status", "")).upper()
        if status == "FAIL" and (check.get("accuracy_blocking") or check.get("framework_blocking")):
            add_blocker(
                blockers,
                category="accuracy_gate",
                blocker_id=str(check.get("name", "accuracy_check")),
                status=status,
                source=str(path),
                details=str(check.get("details", "")),
                accuracy_blocking=bool(check.get("accuracy_blocking")),
                framework_blocking=bool(check.get("framework_blocking")),
            )
    return evidence, blockers, data


def classify_json_existence(path: Path | None, *, category: str, evidence_id: str, label: str, product_blocking: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if not path or not path.exists():
        add_evidence(rows, category=category, evidence_id=evidence_id, label=label, status="MISSING", path=path, product_blocking=product_blocking)
        if product_blocking:
            add_blocker(blockers, category=category, blocker_id=f"{evidence_id}_missing", status="MISSING", source=str(path or ""), details=f"{label} is missing.")
        return rows[0], blockers
    data = read_json(path)
    product_ready = bool(data.get("product_lut_ready", False))
    status = "PASS" if path.exists() else "MISSING"
    summary = f"schema={data.get('schema')}; product_lut_ready={product_ready}; rows={len(data.get('rows', [])) if isinstance(data.get('rows'), list) else data.get('row_count', '')}"
    add_evidence(rows, category=category, evidence_id=evidence_id, label=label, status=status, path=path, schema=str(data.get("schema", "")), summary=summary, product_blocking=product_blocking and not product_ready)
    if product_blocking and not product_ready:
        add_blocker(blockers, category=category, blocker_id=f"{evidence_id}_not_product_ready", status="FAIL", source=str(path), details=summary)
    return rows[0], blockers


def build_evidence(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    config_dir = config_path.parent
    views = config.get("views", {}) if isinstance(config.get("views"), dict) else {}
    inputs = config.get("inputs", {}) if isinstance(config.get("inputs"), dict) else {}
    accuracy_position = config.get("accuracy_position", {}) if isinstance(config.get("accuracy_position"), dict) else {}

    evidence: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for key, label in [
        ("cra3_optical_convergence_full_axes_report", "CRA3 full-axis optical convergence"),
        ("cra3_rgb_r84_convergence_report", "RGB r84 optical grid qualification"),
        ("cra3_optical_convergence_r60_r70_r80_report", "CRA3 r60/r70/r80 optical convergence"),
    ]:
        row, row_blockers = classify_camera_convergence(resolve_path(views.get(key), config_dir), key, label)
        evidence.append(row)
        blockers.extend(row_blockers)

    for key, category, label in [
        ("crosstalk_convergence_report", "crosstalk_convergence", "Full-array 3D crosstalk convergence"),
        ("crosstalk_xsection_convergence", "crosstalk_xsection_convergence", "High-resolution 2D crosstalk x-section convergence"),
        ("native_response_convergence_report", "electrical_convergence", "Native DEVSIM response convergence"),
    ]:
        row, row_blockers = classify_generic_status_report(resolve_path(views.get(key), config_dir), category=category, evidence_id=key, label=label)
        evidence.append(row)
        blockers.extend(row_blockers)

    for key, category, label in [
        ("camera_system_field_lut_query", "camera_lut_consumer_validation", "Dense field LUT query/validation"),
        ("camera_lut_spectral_coverage", "camera_lut_coverage", "Camera LUT spectral coverage"),
        ("camera_system_research_lut", "camera_lut_artifact", "Native-DEVSIM camera research LUT"),
        ("camera_system_uncertainty_lut", "camera_lut_artifact", "Camera-system uncertainty LUT"),
    ]:
        row, row_blockers = classify_json_existence(resolve_path(views.get(key), config_dir), category=category, evidence_id=key, label=label)
        evidence.append(row)
        blockers.extend(row_blockers)

    accuracy_path = resolve_path(inputs.get("accuracy_gate"), config_dir)
    accuracy_evidence, accuracy_blockers, accuracy_gate = classify_accuracy_gate(accuracy_path)
    evidence.extend(accuracy_evidence)
    blockers.extend(accuracy_blockers)

    product_blockers = [row for row in blockers if row.get("accuracy_blocking", True)]
    framework_blockers = [row for row in blockers if row.get("framework_blocking")]
    missing_or_failed_evidence = [
        row for row in evidence if row["status"] in {"MISSING", "FAIL"} and row.get("product_blocking")
    ]
    framework_ready = bool(accuracy_gate.get("framework_ready", False)) and not framework_blockers
    accuracy_ready = bool(accuracy_gate.get("accuracy_ready", False))
    quantitative_evidence_pass = not missing_or_failed_evidence
    product_lut_ready = bool(framework_ready and accuracy_ready and quantitative_evidence_pass)
    research_lut_ready = bool(accuracy_position.get("research_lut_ready", False)) and framework_ready
    status = "PRODUCT_READY" if product_lut_ready else "RESEARCH_READY_NOT_PRODUCT" if research_lut_ready else "CHECK"

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_csv = output_dir / "camera_system_quantitative_evidence.csv"
    blockers_csv = output_dir / "camera_system_quantitative_blockers.csv"
    manifest_json = output_dir / "camera_system_quantitative_evidence.json"
    report_md = output_dir / "camera_system_quantitative_evidence.md"
    write_csv(evidence_csv, evidence, EVIDENCE_COLUMNS)
    write_csv(blockers_csv, blockers, BLOCKER_COLUMNS)
    payload = {
        "schema": "camera_system_quantitative_evidence_v1",
        "status": status,
        "framework_ready": framework_ready,
        "research_lut_ready": research_lut_ready,
        "quantitative_evidence_pass": quantitative_evidence_pass,
        "accuracy_ready": accuracy_ready,
        "product_lut_ready": product_lut_ready,
        "source_config": str(config_path),
        "accuracy_position": accuracy_position,
        "evidence_count": len(evidence),
        "blocker_count": len(blockers),
        "product_blocker_count": len(product_blockers),
        "framework_blocker_count": len(framework_blockers),
        "evidence": evidence,
        "blockers": blockers,
        "outputs": {
            "manifest_json": str(manifest_json),
            "evidence_csv": str(evidence_csv),
            "blockers_csv": str(blockers_csv),
            "report_md": str(report_md),
        },
        "notes": [
            "This manifest indexes existing evidence; it does not run new solver jobs.",
            "A PASS here for individual convergence artifacts is numerical evidence only.",
            "Product LUT readiness remains false until measured stack/material/device calibration and all quantitative gates pass.",
        ],
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report_md, payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Camera-System Quantitative Evidence",
        "",
        f"- Status: `{payload['status']}`",
        f"- Framework ready: `{payload['framework_ready']}`",
        f"- Research LUT ready: `{payload['research_lut_ready']}`",
        f"- Quantitative evidence pass: `{payload['quantitative_evidence_pass']}`",
        f"- Accuracy ready: `{payload['accuracy_ready']}`",
        f"- Product LUT ready: `{payload['product_lut_ready']}`",
        f"- Evidence rows: `{payload['evidence_count']}`",
        f"- Blockers: `{payload['blocker_count']}`",
        "",
        "## Evidence",
        "",
        "| Category | ID | Status | Product Blocking | Summary |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["evidence"]:
        summary = str(row.get("summary", "")).replace("|", "\\|")
        lines.append(
            f"| {row['category']} | {row['id']} | `{row['status']}` | `{row['product_blocking']}` | {summary} |"
        )
    if payload["blockers"]:
        lines += ["", "## Blockers", "", "| Category | ID | Status | Details |", "|---|---|---:|---|"]
        for row in payload["blockers"]:
            details = str(row.get("details", "")).replace("|", "\\|")
            lines.append(f"| {row['category']} | {row['id']} | `{row['status']}` | {details} |")
    lines += [
        "",
        "This report is an evidence index. It does not make proxy/reference inputs product-accurate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-product-ready", action="store_true")
    args = parser.parse_args()
    payload = build_evidence(args.config.resolve(), args.output_dir.resolve())
    summary = {
        "schema": payload["schema"],
        "status": payload["status"],
        "framework_ready": payload["framework_ready"],
        "research_lut_ready": payload["research_lut_ready"],
        "quantitative_evidence_pass": payload["quantitative_evidence_pass"],
        "accuracy_ready": payload["accuracy_ready"],
        "product_lut_ready": payload["product_lut_ready"],
        "evidence_count": payload["evidence_count"],
        "blocker_count": payload["blocker_count"],
        "outputs": payload["outputs"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.require_product_ready and not payload["product_lut_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
