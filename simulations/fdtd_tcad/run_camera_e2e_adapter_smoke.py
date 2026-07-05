#!/usr/bin/env python3
"""Run per-sensor CameraE2E adapter smoke queries.

This executes the same flat-bundle query command that a downstream CameraE2E
adapter would call, once in research mode and once in product-probe mode for
each sensor. Product probe success means the command runs and returns zero
allowed product rows.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_adapter_smoke"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_profile",
    "research_output_dir",
    "research_returncode",
    "research_query_row_count",
    "research_allowed_query_count",
    "research_status",
    "product_output_dir",
    "product_returncode",
    "product_query_row_count",
    "product_allowed_query_count",
    "product_status",
    "flat_sensor_json",
    "adapter_example_json",
    "smoke_gate",
    "smoke_notes",
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


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def run_flat_query(
    *,
    flat_bundle_json: Path,
    output_dir: Path,
    slug: str,
    mode: str,
    field_x: str,
    field_z: str,
    wavelength_nm: str,
    temperature_c: float,
    exposure_s: float,
    incident_photons: float,
) -> tuple[int, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "query_camera_e2e_flat_sensor_bundle.py"),
        "--flat-bundle-json",
        str(flat_bundle_json),
        "--output-dir",
        str(output_dir),
        "--slugs",
        slug,
        f"--field-x={field_x}",
        f"--field-z={field_z}",
        "--wavelength-nm",
        wavelength_nm,
        "--mode",
        mode,
        "--temperature-c",
        str(temperature_c),
        "--exposure-s",
        str(exposure_s),
        "--incident-photons",
        str(incident_photons),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (output_dir / "command.json").write_text(json.dumps({"cmd": cmd, "returncode": proc.returncode}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return proc.returncode, read_json(output_dir / "camera_e2e_flat_sensor_query.json")


def build_smoke(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    flat_bundle_json = package_dir / "camera_e2e_flat_sensor_bundle" / "camera_e2e_flat_sensor_bundle.json"
    policy_json = read_json(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy.json")
    adapter_json = read_json(package_dir / "camera_e2e_adapter_examples" / "camera_e2e_adapter_examples.json")
    policy_rows = read_csv_rows(package_dir / "camera_e2e_usage_policy" / "camera_e2e_usage_policy_by_sensor.csv")
    adapter_rows = {row.get("slug", ""): row for row in read_csv_rows(package_dir / "camera_e2e_adapter_examples" / "camera_e2e_adapter_examples_by_sensor.csv")}

    sensor_rows: list[dict[str, Any]] = []
    research_failed: list[str] = []
    research_blocked: list[str] = []
    product_failed: list[str] = []
    product_open: list[str] = []
    missing_examples: list[str] = []

    for policy_row in policy_rows:
        slug = policy_row.get("slug", "")
        adapter_row = adapter_rows.get(slug, {})
        if not adapter_row:
            missing_examples.append(slug)
        sensor_root = output_dir / "sensors" / slug
        research_dir = sensor_root / "research"
        product_dir = sensor_root / "product_probe"
        research_rc, research_payload = run_flat_query(
            flat_bundle_json=flat_bundle_json,
            output_dir=research_dir,
            slug=slug,
            mode="research",
            field_x=args.field_x,
            field_z=args.field_z,
            wavelength_nm=args.wavelength_nm,
            temperature_c=args.temperature_c,
            exposure_s=args.exposure_s,
            incident_photons=args.incident_photons,
        )
        product_rc, product_payload = run_flat_query(
            flat_bundle_json=flat_bundle_json,
            output_dir=product_dir,
            slug=slug,
            mode="product",
            field_x=args.field_x,
            field_z=args.field_z,
            wavelength_nm=args.wavelength_nm,
            temperature_c=args.temperature_c,
            exposure_s=args.exposure_s,
            incident_photons=args.incident_photons,
        )
        research_allowed = safe_int(research_payload.get("allowed_query_count"))
        product_allowed = safe_int(product_payload.get("allowed_query_count"))
        research_status = research_payload.get("validation", {}).get("status", "MISSING")
        product_status = product_payload.get("validation", {}).get("status", "MISSING")
        if research_rc != 0 or not bool(research_payload.get("validation", {}).get("pass")):
            research_failed.append(slug)
        if research_allowed <= 0:
            research_blocked.append(slug)
        if product_rc != 0 or not bool(product_payload.get("validation", {}).get("pass")):
            product_failed.append(slug)
        if product_allowed != 0:
            product_open.append(slug)
        gate = "PASS" if research_rc == 0 and research_allowed > 0 and product_rc == 0 and product_allowed == 0 else "FAIL"
        notes = []
        if research_allowed <= 0:
            notes.append("research query returned no allowed rows")
        if product_allowed != 0:
            notes.append("product probe returned allowed rows")
        sensor_rows.append(
            {
                "slug": slug,
                "code": policy_row.get("code", ""),
                "manufacturer": policy_row.get("manufacturer", ""),
                "device_name": policy_row.get("device_name", ""),
                "camera_e2e_profile": policy_row.get("camera_e2e_profile", ""),
                "research_output_dir": repo_rel(research_dir),
                "research_returncode": research_rc,
                "research_query_row_count": research_payload.get("query_row_count", 0),
                "research_allowed_query_count": research_allowed,
                "research_status": research_status,
                "product_output_dir": repo_rel(product_dir),
                "product_returncode": product_rc,
                "product_query_row_count": product_payload.get("query_row_count", 0),
                "product_allowed_query_count": product_allowed,
                "product_status": product_status,
                "flat_sensor_json": adapter_row.get("flat_sensor_json", ""),
                "adapter_example_json": adapter_row.get("adapter_example_json", ""),
                "smoke_gate": gate,
                "smoke_notes": "; ".join(notes),
            }
        )

    smoke_failures = [row.get("slug", "") for row in sensor_rows if row.get("smoke_gate") != "PASS"]
    total_research_allowed = sum(safe_int(row.get("research_allowed_query_count")) for row in sensor_rows)
    total_product_allowed = sum(safe_int(row.get("product_allowed_query_count")) for row in sensor_rows)
    checks = [
        check_row(
            "usage_policy_valid",
            policy_json.get("schema") == "camera_e2e_usage_policy_v1" and bool(policy_json.get("validation", {}).get("pass")),
            policy_json.get("validation", {}).get("status", "MISSING"),
            {"strict_product_filter_row_count": policy_json.get("strict_product_filter_row_count")},
            "Regenerate usage policy.",
        ),
        check_row(
            "adapter_examples_valid",
            adapter_json.get("schema") == "camera_e2e_adapter_examples_v1" and bool(adapter_json.get("validation", {}).get("pass")),
            adapter_json.get("validation", {}).get("status", "MISSING"),
            {"example_file_count": adapter_json.get("example_file_count")},
            "Regenerate adapter examples.",
        ),
        check_row(
            "per_sensor_examples_present",
            not missing_examples,
            "PASS" if not missing_examples else "FAIL",
            {"missing_examples": missing_examples},
            "Generate adapter examples for every policy sensor.",
        ),
        check_row(
            "research_smoke_allowed",
            not research_failed and not research_blocked and total_research_allowed > 0,
            "PASS" if not research_failed and not research_blocked and total_research_allowed > 0 else "FAIL",
            {"research_failed": research_failed, "research_blocked": research_blocked, "total_research_allowed": total_research_allowed},
            "Research adapter smoke should run and return allowed rows.",
        ),
        check_row(
            "product_smoke_blocked",
            not product_failed and not product_open and total_product_allowed == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if not product_failed and not product_open and total_product_allowed == 0 else "FAIL",
            {"product_failed": product_failed, "product_open": product_open, "total_product_allowed": total_product_allowed},
            "Product adapter smoke should run and return zero allowed rows.",
        ),
        check_row(
            "per_sensor_smoke_gates",
            not smoke_failures,
            "PASS" if not smoke_failures else "FAIL",
            {"smoke_failures": smoke_failures},
            "Inspect per-sensor smoke rows.",
        ),
    ]
    error_count = sum(1 for row in checks if not boolish(row.get("pass")))
    status = "CAMERA_E2E_ADAPTER_SMOKE_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_json = output_dir / "camera_e2e_adapter_smoke.json"
    by_sensor_csv = output_dir / "camera_e2e_adapter_smoke_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_adapter_smoke_checks.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_adapter_smoke_v1",
        "artifact_role": "camera_e2e_adapter_executable_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "total_research_allowed_query_count": total_research_allowed,
        "total_product_allowed_query_count": total_product_allowed,
        "profile_counts": dict(sorted(Counter(str(row.get("camera_e2e_profile", "")) for row in sensor_rows).items())),
        "inputs": {
            "field_x": args.field_x,
            "field_z": args.field_z,
            "wavelength_nm": args.wavelength_nm,
            "temperature_c": args.temperature_c,
            "exposure_s": args.exposure_s,
            "incident_photons": args.incident_photons,
        },
        "validation": {
            "schema": "camera_e2e_adapter_smoke_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": error_count,
            "error_count": error_count,
            "warning_count": 0,
            "issues": [row for row in checks if not boolish(row.get("pass"))],
            "checks": checks,
        },
        "outputs": {
            "json": repo_rel(smoke_json),
            "by_sensor_csv": repo_rel(by_sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
            "sensor_smoke_dir": repo_rel(output_dir / "sensors"),
        },
    }
    write_csv(by_sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(smoke_json, payload)
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
<title>CameraE2E Adapter Smoke</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Adapter Smoke</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Executes per-sensor research and product-probe flat-bundle queries.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">smoke status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("total_research_allowed_query_count", 0))}</div><div class="muted">research allowed rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("total_product_allowed_query_count", 0))}</div><div class="muted">product allowed rows</div></div>
</div>
<h2>Checks</h2>{html_table(checks, CHECK_COLUMNS)}
<h2>Per-Sensor Smoke</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
</main></body></html>
"""
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_adapter_smoke_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_adapter_smoke_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_adapter_smoke_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_adapter_smoke_html"] = payload["outputs"]["html"]
    outputs["camera_e2e_adapter_smoke_sensor_dir"] = payload["outputs"]["sensor_smoke_dir"]
    package["latest_camera_e2e_adapter_smoke"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "total_research_allowed_query_count": payload["total_research_allowed_query_count"],
        "total_product_allowed_query_count": payload["total_product_allowed_query_count"],
        "profile_counts": payload["profile_counts"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-x", default="-1,0,1")
    parser.add_argument("--field-z", default="-1,0,1")
    parser.add_argument("--wavelength-nm", default="all")
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--exposure-s", type=float, default=0.01)
    parser.add_argument("--incident-photons", type=float, default=8000.0)
    return parser


def main() -> None:
    payload = build_smoke(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "total_research_allowed_query_count": payload["total_research_allowed_query_count"],
                "total_product_allowed_query_count": payload["total_product_allowed_query_count"],
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
