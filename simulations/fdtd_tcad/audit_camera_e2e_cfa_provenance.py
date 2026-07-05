#!/usr/bin/env python3
"""Audit CFA provenance for CameraE2E color/material use.

The color and material exports intentionally keep research fallbacks alive so
CameraE2E plumbing can be exercised. This audit makes the fallback explicit per
sensor so downstream users do not mistake a generic RGB proxy for measured or
sensor-confirmed CFA data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_cfa_provenance"

CFA_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "sensor_index_cfa_pattern",
    "optical_cfa_pattern",
    "optical_cfa_pattern_source_kind",
    "optical_cfa_pattern_confidence",
    "optical_cfa_thickness_um",
    "optical_cfa_thickness_source_kind",
    "optical_cfa_thickness_confidence",
    "cfa_proxy_enabled",
    "cfa_proxy_applicability",
    "cfa_proxy_library_id",
    "cfa_proxy_channel_count",
    "cfa_proxy_channels",
    "material_cfa_proxy_row_count",
    "material_cfa_proxy_applicability",
    "material_cfa_proxy_thickness_um",
    "spectral_channel_count",
    "spectral_channels",
    "color_matrix_applicability",
    "color_matrix_gate",
    "generic_rgb_fallback_detected",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "camera_e2e_recommended_use",
    "product_lut_gate",
    "primary_blocker",
    "next_action",
    "model_json",
]

CHECK_COLUMNS = ["check_id", "severity", "status", "evidence", "required_action"]


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
            writer.writerow({column: csv_cell(row.get(column, "")) for column in columns})


def csv_cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def value_dict_value(payload: Any, default: Any = "") -> Any:
    if isinstance(payload, dict):
        value = payload.get("value", default)
        return default if value is None else value
    return default if payload is None else payload


def value_dict_source_kind(payload: Any, default: str = "") -> str:
    return str(payload.get("source_kind", default)) if isinstance(payload, dict) else default


def value_dict_confidence(payload: Any, default: Any = "") -> Any:
    return payload.get("confidence", default) if isinstance(payload, dict) else default


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        value = row.get(key, "")
        if value:
            result.setdefault(value, []).append(row)
    return result


def classify(
    *,
    pattern: str,
    proxy_enabled: bool,
    proxy_channels: list[str],
    thickness_source: str,
    generic_rgb_fallback: bool,
) -> tuple[str, str, str, str, str]:
    normalized_pattern = pattern.strip().lower()
    channel_set = set(proxy_channels)
    if generic_rgb_fallback:
        return (
            "GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN",
            "MISSING",
            "Use only for CameraE2E loader/plumbing; do not treat RGB color response as sensor evidence.",
            "CFA pattern is unavailable, but generic RGB fallback rows are present.",
            "Identify CFA pattern/filter arrangement or mark this sensor as non-color-specific before color-response use.",
        )
    if not normalized_pattern:
        return (
            "CFA_PATTERN_MISSING",
            "MISSING",
            "Use only for non-color-specific plumbing until CFA pattern is known.",
            "CFA pattern is unavailable in optical_qe_db and sensor index.",
            "Extract CFA/filter arrangement from source material or provide measured/imported filter model.",
        )
    if "mono" in normalized_pattern or "clear" in normalized_pattern or channel_set == {"clear"}:
        return (
            "MONO_CLEAR_PROXY",
            "CHECK",
            "Use as monochrome/clear-channel research prior; RGB CCM is not applicable.",
            "Monochrome/clear channel is proxy or default unless measured QE is attached.",
            "Attach measured mono spectral QE for product use.",
        )
    if proxy_enabled and {"red", "green", "blue"}.issubset(channel_set):
        if thickness_source in {"extracted", "derived_from_extracted_range"}:
            return (
                "SENSOR_SPECIFIC_RGB_PROXY",
                "CHECK",
                "Use for research color/material trend with explicit proxy gates.",
                "RGB pattern and CFA thickness are sensor-specific, but n,k/transmission remains proxy.",
                "Replace proxy RGB n,k with measured material table and measured spectral QE.",
            )
        return (
            "RGB_PROXY_DEFAULT_THICKNESS",
            "CHECK",
            "Use for coarse research trend only; thickness or n,k is default proxy.",
            "RGB pattern is known, but CFA thickness and/or n,k is default/inferred proxy.",
            "Extract CFA thickness and import measured material/color response.",
        )
    if proxy_enabled:
        return (
            "CFA_PROXY_ENABLED_UNUSUAL_CHANNELS",
            "CHECK",
            "Use only after manually inspecting channel mapping.",
            "CFA proxy is enabled but channels are not the standard RGB or clear set.",
            "Verify filter channel mapping and add an explicit CameraE2E color-mode mapping.",
        )
    return (
        "CFA_PROXY_DISABLED",
        "MISSING",
        "Do not use for color-specific CameraE2E response until CFA proxy or measured data exists.",
        "CFA proxy is disabled for this sensor.",
        "Add measured/imported CFA model or explicitly justify a proxy channel model.",
    )


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    optical_db = args.optical_qe_db.resolve()

    sensor_rows = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    optical_summary = index_by(read_csv_rows(optical_db / "optical_qe_summary.csv"), "code")
    material_summary = index_by(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_summary.csv"), "slug")
    spectral_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv"), "slug")
    matrix_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv"), "slug")

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        code = sensor.get("code", "")
        optical_row = optical_summary.get(code, {})
        model_rel = optical_row.get("model_json", "")
        optical_model = read_json(optical_db / model_rel) if model_rel else read_json(optical_db / "models" / f"{slug}.json")
        cfa = optical_model.get("optical", {}).get("cfa", {}) if isinstance(optical_model.get("optical", {}), dict) else {}
        pattern_info = cfa.get("pattern", {}) if isinstance(cfa, dict) else {}
        thickness_info = cfa.get("representative_thickness_um", {}) if isinstance(cfa, dict) else {}
        proxy = optical_model.get("cfa_proxy_nk", {}) if isinstance(optical_model, dict) else {}
        proxy_channels = sorted((proxy.get("channels") or {}).keys()) if isinstance(proxy.get("channels"), dict) else []
        pattern = str(value_dict_value(pattern_info, "") or sensor.get("cfa_pattern", "") or "").strip().lower()
        material = material_summary.get(slug, {})
        spectral_rows = spectral_by_slug.get(slug, [])
        spectral_channels = sorted({row.get("color_channel", "") for row in spectral_rows if row.get("color_channel", "")})
        matrix = matrix_by_slug.get(slug, {})
        generic_rgb_fallback = (
            not pattern
            and (
                material.get("cfa_proxy_applicability") == "unknown_cfa_pattern"
                and int(float(material.get("cfa_proxy_row_count", 0) or 0)) > 0
                or {"red", "green", "blue"}.issubset(set(spectral_channels))
                or matrix.get("applicability") == "rgb_proxy_seed"
            )
        )
        cls, gate, recommended_use, blocker, next_action = classify(
            pattern=pattern,
            proxy_enabled=boolish(proxy.get("enabled")),
            proxy_channels=proxy_channels,
            thickness_source=value_dict_source_kind(thickness_info, ""),
            generic_rgb_fallback=generic_rgb_fallback,
        )
        row = {
            "slug": slug,
            "code": code,
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "sensor_index_cfa_pattern": sensor.get("cfa_pattern", ""),
            "optical_cfa_pattern": pattern,
            "optical_cfa_pattern_source_kind": value_dict_source_kind(pattern_info, ""),
            "optical_cfa_pattern_confidence": value_dict_confidence(pattern_info, ""),
            "optical_cfa_thickness_um": value_dict_value(thickness_info, ""),
            "optical_cfa_thickness_source_kind": value_dict_source_kind(thickness_info, ""),
            "optical_cfa_thickness_confidence": value_dict_confidence(thickness_info, ""),
            "cfa_proxy_enabled": boolish(proxy.get("enabled")),
            "cfa_proxy_applicability": proxy.get("applicability", ""),
            "cfa_proxy_library_id": proxy.get("library_id", ""),
            "cfa_proxy_channel_count": len(proxy_channels),
            "cfa_proxy_channels": ";".join(proxy_channels),
            "material_cfa_proxy_row_count": material.get("cfa_proxy_row_count", ""),
            "material_cfa_proxy_applicability": material.get("cfa_proxy_applicability", ""),
            "material_cfa_proxy_thickness_um": material.get("cfa_proxy_thickness_um", ""),
            "spectral_channel_count": len(spectral_channels),
            "spectral_channels": ";".join(spectral_channels),
            "color_matrix_applicability": matrix.get("applicability", ""),
            "color_matrix_gate": matrix.get("gate", ""),
            "generic_rgb_fallback_detected": generic_rgb_fallback,
            "cfa_provenance_class": cls,
            "cfa_assumption_gate": gate,
            "camera_e2e_recommended_use": recommended_use,
            "product_lut_gate": "FAIL",
            "primary_blocker": blocker,
            "next_action": next_action,
            "model_json": repo_rel(optical_db / model_rel) if model_rel else repo_rel(optical_db / "models" / f"{slug}.json"),
        }
        rows.append(row)
        if generic_rgb_fallback:
            checks.append(
                {
                    "check_id": f"{slug}:generic_rgb_fallback",
                    "severity": "warning",
                    "status": "CHECK",
                    "evidence": json.dumps(
                        {
                            "material_cfa_proxy_applicability": material.get("cfa_proxy_applicability", ""),
                            "spectral_channels": spectral_channels,
                            "color_matrix_applicability": matrix.get("applicability", ""),
                        },
                        sort_keys=True,
                    ),
                    "required_action": "Do not use RGB spectral/color rows as sensor-confirmed data until CFA pattern is known.",
                }
            )
        if not spectral_rows:
            checks.append(
                {
                    "check_id": f"{slug}:spectral_rows_missing",
                    "severity": "error",
                    "status": "FAIL",
                    "evidence": "{}",
                    "required_action": "Regenerate color response export before CameraE2E bundle use.",
                }
            )

    error_count = sum(1 for check in checks if check.get("severity") == "error")
    class_counts = dict(Counter(str(row.get("cfa_provenance_class", "")) for row in rows))
    gate_counts = dict(Counter(str(row.get("cfa_assumption_gate", "")) for row in rows))
    status = "FAIL" if error_count else "RESEARCH_CFA_PROVENANCE_READY_PRODUCT_BLOCKED"

    by_sensor_csv = output_dir / "camera_e2e_cfa_provenance_by_sensor.csv"
    checks_csv = output_dir / "camera_e2e_cfa_provenance_checks.csv"
    report_json = output_dir / "camera_e2e_cfa_provenance.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_cfa_provenance_audit_v1",
        "artifact_role": "camera_e2e_cfa_color_material_provenance_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "source_optical_qe_db": repo_rel(optical_db),
        "sensor_count": len(rows),
        "row_count": len(rows),
        "status": status,
        "product_ready_count": 0,
        "generic_rgb_fallback_unknown_pattern_count": sum(1 for row in rows if boolish(row.get("generic_rgb_fallback_detected"))),
        "class_counts": class_counts,
        "assumption_gate_counts": gate_counts,
        "validation": {
            "schema": "camera_e2e_cfa_provenance_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": len(checks),
            "error_count": error_count,
            "warning_count": sum(1 for check in checks if check.get("severity") == "warning"),
            "issues": checks,
        },
        "policy": {
            "research": "Allowed only when cfa_assumption_gate and cfa_provenance_class are propagated to CameraE2E consumers.",
            "product": "Blocked until CFA pattern, thickness, n,k, and spectral QE are measured/imported and calibrated.",
            "important_warning": "GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN rows are plumbing priors, not sensor-confirmed color response.",
        },
        "outputs": {
            "json": repo_rel(report_json),
            "by_sensor_csv": repo_rel(by_sensor_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(by_sensor_csv, rows, CFA_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, rows, checks)
    update_package(package_dir, payload)
    return payload


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return html.escape(f"{value:.6g}" if math.isfinite(value) else str(value))
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    if not shown:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1440px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    issue_html = html_table(checks, CHECK_COLUMNS) if checks else '<p class="pass">No CFA provenance structural errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E CFA Provenance</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E CFA Provenance</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This separates sensor-confirmed CFA metadata from generic color fallback rows.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">audit status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("generic_rgb_fallback_unknown_pattern_count", 0))}</div><div class="muted">unknown CFA RGB fallbacks</div></div>
<div class="card"><div class="metric fail">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Checks</h2>{issue_html}
<h2>Sensor CFA Provenance</h2>{html_table(rows, CFA_COLUMNS)}
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
    outputs["camera_e2e_cfa_provenance_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_cfa_provenance_by_sensor_csv"] = payload["outputs"]["by_sensor_csv"]
    outputs["camera_e2e_cfa_provenance_checks_csv"] = payload["outputs"]["checks_csv"]
    outputs["camera_e2e_cfa_provenance_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_cfa_provenance"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "class_counts": payload["class_counts"],
        "assumption_gate_counts": payload["assumption_gate_counts"],
        "generic_rgb_fallback_unknown_pattern_count": payload["generic_rgb_fallback_unknown_pattern_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--optical-qe-db", type=Path, default=DEFAULT_OPTICAL_QE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = build_audit(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": payload["status"],
                "sensor_count": payload["sensor_count"],
                "generic_rgb_fallback_unknown_pattern_count": payload["generic_rgb_fallback_unknown_pattern_count"],
                "class_counts": payload["class_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
