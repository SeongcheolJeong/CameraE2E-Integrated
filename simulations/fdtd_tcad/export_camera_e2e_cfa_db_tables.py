#!/usr/bin/env python3
"""Export dedicated CameraE2E CFA DB tables.

The material table already contains CFA proxy rows, but CameraE2E consumers need
a direct CFA lookup layer: pattern/thickness/source by sensor and wavelength x
channel transmission/n,k rows. This exporter exposes those rows explicitly while
preserving the existing research/product gates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_cfa_db_tables"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "optical_model_json",
    "sensor_index_cfa_pattern",
    "optical_cfa_pattern",
    "optical_cfa_pattern_source_kind",
    "optical_cfa_pattern_confidence",
    "optical_cfa_thickness_um",
    "optical_cfa_thickness_source_kind",
    "optical_cfa_thickness_confidence",
    "optical_cfa_thickness_min_um",
    "optical_cfa_thickness_max_um",
    "color_filter_pitch_um",
    "color_filter_pitch_source_kind",
    "cfa_proxy_enabled",
    "cfa_proxy_applicability",
    "cfa_proxy_library_id",
    "cfa_proxy_reference_thickness_um",
    "cfa_proxy_thickness_um",
    "cfa_proxy_channels",
    "transmission_lut_row_count",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "research_gate",
    "product_lut_gate",
    "camera_e2e_allowed_use",
    "primary_blocker",
    "required_before_product_use",
]

TRANSMISSION_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "color_channel",
    "wavelength_nm",
    "n",
    "k",
    "transmission_absorption_only",
    "thickness_um",
    "thickness_source_kind",
    "cfa_pattern",
    "cfa_pattern_source_kind",
    "cfa_channel_source",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "material_source_kind",
    "measured",
    "research_gate",
    "product_lut_gate",
    "source_path",
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


def value_info_value(info: Any, default: Any = "") -> Any:
    return info.get("value", default) if isinstance(info, dict) else default


def value_info_source(info: Any, default: str = "") -> str:
    return str(info.get("source_kind", default)) if isinstance(info, dict) else default


def value_info_confidence(info: Any, default: str = "") -> str:
    value = info.get("confidence", default) if isinstance(info, dict) else default
    return "" if value is None else str(value)


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def cfa_model_for(row: dict[str, str]) -> dict[str, Any]:
    path = abs_from_repo(row.get("model_json", ""))
    return read_json(path)


def build_sensor_rows(cfa_rows: list[dict[str, str]], material_by_slug: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    sensor_rows: list[dict[str, Any]] = []
    for row in cfa_rows:
        model = cfa_model_for(row)
        optical = model.get("optical", {}) if isinstance(model.get("optical"), dict) else {}
        cfa = optical.get("cfa", {}) if isinstance(optical.get("cfa"), dict) else {}
        proxy = model.get("cfa_proxy_nk", {}) if isinstance(model.get("cfa_proxy_nk"), dict) else {}
        channels = proxy.get("channels", {}) if isinstance(proxy.get("channels"), dict) else {}
        thickness_min = cfa.get("thickness_min_um", {}) if isinstance(cfa.get("thickness_min_um"), dict) else {}
        thickness_max = cfa.get("thickness_max_um", {}) if isinstance(cfa.get("thickness_max_um"), dict) else {}
        pitch = cfa.get("color_filter_pitch_um", {}) if isinstance(cfa.get("color_filter_pitch_um"), dict) else {}
        sensor_rows.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "optical_model_json": row.get("model_json", ""),
                "sensor_index_cfa_pattern": row.get("sensor_index_cfa_pattern", ""),
                "optical_cfa_pattern": row.get("optical_cfa_pattern", ""),
                "optical_cfa_pattern_source_kind": row.get("optical_cfa_pattern_source_kind", ""),
                "optical_cfa_pattern_confidence": row.get("optical_cfa_pattern_confidence", ""),
                "optical_cfa_thickness_um": row.get("optical_cfa_thickness_um", ""),
                "optical_cfa_thickness_source_kind": row.get("optical_cfa_thickness_source_kind", ""),
                "optical_cfa_thickness_confidence": row.get("optical_cfa_thickness_confidence", ""),
                "optical_cfa_thickness_min_um": value_info_value(thickness_min, ""),
                "optical_cfa_thickness_max_um": value_info_value(thickness_max, ""),
                "color_filter_pitch_um": value_info_value(pitch, ""),
                "color_filter_pitch_source_kind": value_info_source(pitch, ""),
                "cfa_proxy_enabled": row.get("cfa_proxy_enabled", ""),
                "cfa_proxy_applicability": row.get("cfa_proxy_applicability", ""),
                "cfa_proxy_library_id": row.get("cfa_proxy_library_id", ""),
                "cfa_proxy_reference_thickness_um": proxy.get("reference_thickness_um", ""),
                "cfa_proxy_thickness_um": row.get("material_cfa_proxy_thickness_um", ""),
                "cfa_proxy_channels": ";".join(sorted(channels)) or row.get("cfa_proxy_channels", ""),
                "transmission_lut_row_count": len(material_by_slug.get(row.get("slug", ""), [])),
                "cfa_provenance_class": row.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": row.get("cfa_assumption_gate", ""),
                "generic_rgb_fallback_detected": row.get("generic_rgb_fallback_detected", ""),
                "research_gate": "CHECK",
                "product_lut_gate": row.get("product_lut_gate", "FAIL"),
                "camera_e2e_allowed_use": row.get("camera_e2e_recommended_use", ""),
                "primary_blocker": row.get("primary_blocker", ""),
                "required_before_product_use": (
                    "CFA pattern/filter arrangement, CFA thickness, measured CFA/OCL/passivation/Si n,k, "
                    "measured spectral response/QE, and calibrated color targets."
                ),
            }
        )
    return sensor_rows


def build_transmission_rows(material_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in material_rows:
        if row.get("material_family") != "cfa_transmission_proxy":
            continue
        rows.append({column: row.get(column, "") for column in TRANSMISSION_COLUMNS})
    return rows


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], transmission_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E CFA DB Tables</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E CFA DB Tables</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Rows are research/proxy unless product gates pass.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("transmission_row_count", 0))}</div><div class="muted">transmission rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("generic_rgb_fallback_unknown_pattern_count", 0))}</div><div class="muted">generic RGB fallbacks</div></div>
</div>
<h2>Sensor CFA DB Summary</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>CFA Transmission LUT</h2>{html_table(transmission_rows, TRANSMISSION_COLUMNS)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_cfa_db_tables_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_cfa_db_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_cfa_db_transmission_lut_csv"] = payload["outputs"]["transmission_lut_csv"]
    outputs["camera_e2e_cfa_db_tables_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_cfa_db_tables"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "transmission_row_count": payload["transmission_row_count"],
        "generic_rgb_fallback_unknown_pattern_count": payload["generic_rgb_fallback_unknown_pattern_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_json, package)


def export_tables(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    cfa_rows = read_csv_rows(package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv")
    material_rows = read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv")
    transmission_rows = build_transmission_rows(material_rows)
    material_by_slug = group_rows(transmission_rows, "slug")
    sensor_rows = build_sensor_rows(cfa_rows, material_by_slug)

    issues: list[dict[str, Any]] = []
    if not sensor_rows:
        issues.append({"severity": "error", "code": "no_cfa_sensor_rows", "message": "No CFA provenance sensor rows were found."})
    if not transmission_rows:
        issues.append({"severity": "error", "code": "no_cfa_transmission_rows", "message": "No CFA transmission LUT rows were found."})
    missing_lut = [row.get("slug", "") for row in sensor_rows if not row.get("transmission_lut_row_count")]
    if missing_lut:
        issues.append({"severity": "error", "code": "sensor_missing_cfa_transmission_rows", "slugs": missing_lut})
    generic_count = sum(1 for row in sensor_rows if boolish(row.get("generic_rgb_fallback_detected")))
    if generic_count:
        issues.append({"severity": "warning", "code": "generic_rgb_fallback_present", "message": f"{generic_count} sensors use generic RGB fallback due to unknown CFA pattern."})

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    json_path = output_dir / "camera_e2e_cfa_db_tables.json"
    sensor_csv = output_dir / "camera_e2e_cfa_db_by_sensor.csv"
    transmission_csv = output_dir / "camera_e2e_cfa_db_transmission_lut.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_cfa_db_tables_v1",
        "artifact_role": "camera_e2e_cfa_db_lookup_tables",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "transmission_row_count": len(transmission_rows),
        "generic_rgb_fallback_unknown_pattern_count": generic_count,
        "cfa_provenance_class_counts": dict(Counter(row.get("cfa_provenance_class", "") for row in sensor_rows)),
        "cfa_assumption_gate_counts": dict(Counter(row.get("cfa_assumption_gate", "") for row in sensor_rows)),
        "product_ready_count": 0,
        "validation": {
            "schema": "camera_e2e_cfa_db_tables_validation_v1",
            "pass": error_count == 0,
            "status": "RESEARCH_CFA_DB_TABLES_READY_PRODUCT_BLOCKED" if error_count == 0 else "FAIL",
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "issues": issues,
        },
        "outputs": {
            "json": repo_rel(json_path),
            "by_sensor_csv": repo_rel(sensor_csv),
            "transmission_lut_csv": repo_rel(transmission_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(transmission_csv, transmission_rows, TRANSMISSION_COLUMNS)
    write_json(json_path, payload)
    write_html(html_path, payload, sensor_rows, transmission_rows)
    update_package(package_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    payload = export_tables(parser.parse_args())
    print(
        json.dumps(
            {
                "schema": payload.get("schema"),
                "validation": payload.get("validation"),
                "sensor_count": payload.get("sensor_count"),
                "transmission_row_count": payload.get("transmission_row_count"),
                "generic_rgb_fallback_unknown_pattern_count": payload.get("generic_rgb_fallback_unknown_pattern_count"),
                "outputs": payload.get("outputs"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
