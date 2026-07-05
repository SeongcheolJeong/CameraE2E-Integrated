#!/usr/bin/env python3
"""Export a CameraE2E per-sensor analysis report.

The flat query proves the package can be consumed. This report turns those query
rows into a design-facing summary: what each sensor can currently support in
CameraE2E, which channel/field/wavelength cases are incomplete, and which rows
must stay product-blocked.

It intentionally reports CHECK conditions instead of hiding them. The output is
a research analysis report, not a product accuracy certificate.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_analysis_report"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_use_scope",
    "mesh_confidence_class",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "field_mesh_pass_fraction",
    "field_mesh_coverage_fraction",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "crosstalk_mesh_pass_fraction",
    "crosstalk_mesh_coverage_fraction",
    "mesh_primary_limitations",
    "mesh_next_action",
    "trust_class",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "query_row_count",
    "allowed_query_count",
    "summary_row_count",
    "summary_pass_count",
    "summary_check_count",
    "field_coverage_min",
    "field_coverage_mean",
    "edge_to_center_min",
    "edge_to_center_max",
    "max_output_crosstalk_fraction",
    "max_strongest_neighbor_fraction",
    "mean_signal_e",
    "mean_raw_dn_clipped",
    "min_snr_db",
    "max_snr_db",
    "product_ready",
    "camera_e2e_analysis_gate",
    "recommended_camera_e2e_use",
    "primary_warnings",
    "required_before_product_use",
]

CHANNEL_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "wavelength_nm",
    "color_channel",
    "query_count",
    "requested_field_count",
    "field_coverage_fraction",
    "allowed_count",
    "edge_to_center_signal_ratio",
    "max_output_crosstalk_fraction",
    "max_strongest_neighbor_fraction",
    "mean_signal_e",
    "mean_raw_dn_clipped",
    "min_snr_db",
    "max_snr_db",
    "summary_gate",
    "summary_notes",
    "camera_e2e_use_scope",
    "mesh_confidence_class",
]

ACTION_COLUMNS = [
    "priority_rank",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "action_type",
    "domain",
    "reason",
    "affected_color_channel",
    "affected_wavelength_nm",
    "recommended_action",
    "source",
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


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def mean(values: list[float], default: float = math.nan) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else default


def safe_ratio(numerator: Any, denominator: Any) -> float:
    num = safe_float(numerator)
    den = safe_float(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return math.nan
    return num / den


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            grouped[value].append(row)
    return dict(grouped)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value and value not in result:
            result[value] = row
    return result


def compact_unique(values: list[str], limit: int = 8) -> str:
    output: list[str] = []
    for value in values:
        for item in str(value or "").split(";"):
            clean = item.strip()
            if clean and clean not in output:
                output.append(clean)
    return "; ".join(output[:limit])


def analysis_gate(summary_rows: list[dict[str, str]], use_scope: dict[str, str]) -> str:
    if boolish(use_scope.get("product_ready")):
        return "PRODUCT_READY"
    if not summary_rows:
        return "FAIL"
    if any(row.get("summary_gate") == "CHECK" for row in summary_rows):
        return "CHECK"
    return "RESEARCH_PASS_PRODUCT_BLOCKED"


def recommended_use(gate: str, use_scope: dict[str, str]) -> str:
    if gate == "PRODUCT_READY":
        return "Product CameraE2E ingest allowed by gates."
    if gate == "CHECK":
        return "Use only for research sensitivity; filter CHECK channel/field cases before CRA or color-shading conclusions."
    allowed = use_scope.get("camera_e2e_allowed_use", "")
    return allowed or "Research/plumbing use only; keep product mode blocked."


def build_sensor_rows(
    *,
    query_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    use_scope_by_slug: dict[str, dict[str, str]],
    mesh_by_slug: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    query_by_slug = group_by(query_rows, "slug")
    summary_by_slug = group_by(summary_rows, "slug")
    sensor_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    action_rank = 1

    for slug in sorted(set(query_by_slug) | set(summary_by_slug)):
        qrows = query_by_slug.get(slug, [])
        srows = summary_by_slug.get(slug, [])
        first = (srows or qrows or [{}])[0]
        use_scope = use_scope_by_slug.get(slug, {})
        mesh = mesh_by_slug.get(slug, {})
        coverage_values = [safe_float(row.get("field_coverage_fraction")) for row in srows]
        edge_values = [safe_float(row.get("edge_to_center_signal_ratio")) for row in srows]
        xtalk_values = [safe_float(row.get("max_output_crosstalk_fraction")) for row in srows]
        strongest_values = [safe_float(row.get("max_strongest_neighbor_fraction")) for row in srows]
        signal_values = [safe_float(row.get("signal_e")) for row in qrows]
        raw_values = [safe_float(row.get("raw_dn_clipped")) for row in qrows]
        snr_values = [safe_float(row.get("snr_db")) for row in qrows]
        check_rows = [row for row in srows if row.get("summary_gate") == "CHECK"]
        gate = analysis_gate(srows, use_scope)
        warnings = compact_unique(
            [row.get("summary_notes", "") for row in check_rows]
            + [
                mesh.get("primary_limitations", ""),
                use_scope.get("primary_blockers", ""),
                use_scope.get("required_before_product_use", ""),
            ]
        )
        sensor_rows.append(
            {
                "slug": slug,
                "code": first.get("code", use_scope.get("code", "")),
                "manufacturer": first.get("manufacturer", use_scope.get("manufacturer", "")),
                "device_name": first.get("device_name", use_scope.get("device_name", "")),
                "camera_e2e_use_scope": use_scope.get("camera_e2e_use_scope", first.get("camera_e2e_use_scope", "")),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", use_scope.get("mesh_confidence_class", first.get("mesh_confidence_class", ""))),
                "field_mesh_pass_points": mesh.get("field_pass_points", ""),
                "field_mesh_required_points": mesh.get("field_required_points", ""),
                "field_mesh_pass_fraction": safe_ratio(mesh.get("field_pass_points"), mesh.get("field_required_points")),
                "field_mesh_coverage_fraction": mesh.get("field_coverage_fraction", ""),
                "crosstalk_mesh_pass_points": mesh.get("crosstalk_pass_points", ""),
                "crosstalk_mesh_required_points": mesh.get("crosstalk_required_points", ""),
                "crosstalk_mesh_pass_fraction": safe_ratio(mesh.get("crosstalk_pass_points"), mesh.get("crosstalk_required_points")),
                "crosstalk_mesh_coverage_fraction": mesh.get("crosstalk_coverage_fraction", ""),
                "mesh_primary_limitations": mesh.get("primary_limitations", ""),
                "mesh_next_action": mesh.get("next_action", ""),
                "trust_class": use_scope.get("trust_class", ""),
                "cfa_provenance_class": use_scope.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": use_scope.get("cfa_assumption_gate", ""),
                "query_row_count": len(qrows),
                "allowed_query_count": sum(1 for row in qrows if boolish(row.get("query_allowed"))),
                "summary_row_count": len(srows),
                "summary_pass_count": sum(1 for row in srows if row.get("summary_gate") == "PASS"),
                "summary_check_count": len(check_rows),
                "field_coverage_min": min([value for value in coverage_values if math.isfinite(value)], default=math.nan),
                "field_coverage_mean": mean(coverage_values),
                "edge_to_center_min": min([value for value in edge_values if math.isfinite(value)], default=math.nan),
                "edge_to_center_max": max([value for value in edge_values if math.isfinite(value)], default=math.nan),
                "max_output_crosstalk_fraction": max([value for value in xtalk_values if math.isfinite(value)], default=math.nan),
                "max_strongest_neighbor_fraction": max([value for value in strongest_values if math.isfinite(value)], default=math.nan),
                "mean_signal_e": mean(signal_values),
                "mean_raw_dn_clipped": mean(raw_values),
                "min_snr_db": min([value for value in snr_values if math.isfinite(value)], default=math.nan),
                "max_snr_db": max([value for value in snr_values if math.isfinite(value)], default=math.nan),
                "product_ready": use_scope.get("product_ready", "False"),
                "camera_e2e_analysis_gate": gate,
                "recommended_camera_e2e_use": recommended_use(gate, use_scope),
                "primary_warnings": warnings,
                "required_before_product_use": use_scope.get("required_before_product_use", ""),
            }
        )
        for row in srows:
            channel_rows.append({column: row.get(column, "") for column in CHANNEL_COLUMNS})
            if row.get("summary_gate") == "CHECK":
                action_rows.append(
                    {
                        "priority_rank": action_rank,
                        "slug": slug,
                        "code": first.get("code", ""),
                        "manufacturer": first.get("manufacturer", ""),
                        "device_name": first.get("device_name", ""),
                        "action_type": "coverage_or_anomaly_review",
                        "domain": "Optical / Color",
                        "reason": row.get("summary_notes", ""),
                        "affected_color_channel": row.get("color_channel", ""),
                        "affected_wavelength_nm": row.get("wavelength_nm", ""),
                        "recommended_action": "Run missing high-resolution field/CRA optical points and inspect proxy assumptions before using this channel as CameraE2E trend.",
                        "source": "camera_e2e_flat_sensor_query_summary.csv",
                    }
                )
                action_rank += 1
    return sensor_rows, channel_rows, action_rows


def validate(
    sensor_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    flat_query: dict[str, Any],
    mesh_by_slug: dict[str, dict[str, str]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if flat_query.get("schema") != "camera_e2e_flat_sensor_query_v1" or not bool(flat_query.get("validation", {}).get("pass")):
        issues.append({"severity": "error", "code": "flat_query_invalid"})
    if not sensor_rows:
        issues.append({"severity": "error", "code": "no_sensor_rows"})
    if not channel_rows:
        issues.append({"severity": "error", "code": "no_channel_rows"})
    if not mesh_by_slug:
        issues.append({"severity": "error", "code": "mesh_confidence_rows_missing"})
    missing_mesh = [row.get("slug", "") for row in sensor_rows if row.get("slug") not in mesh_by_slug]
    if missing_mesh:
        issues.append({"severity": "error", "code": "sensor_mesh_confidence_missing", "slugs": missing_mesh})
    product_ready_count = sum(1 for row in sensor_rows if boolish(row.get("product_ready")))
    if product_ready_count:
        issues.append({"severity": "error", "code": "unexpected_product_ready", "count": product_ready_count})
    status = "FAIL" if any(issue.get("severity") == "error" for issue in issues) else "RESEARCH_ANALYSIS_READY_PRODUCT_BLOCKED"
    return {
        "schema": "camera_e2e_analysis_report_validation_v1",
        "pass": not any(issue.get("severity") == "error" for issue in issues),
        "status": status,
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float):
        return html.escape(f"{value:.6g}" if math.isfinite(value) else "")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 160) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1460px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
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
<title>CameraE2E Analysis Report</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Analysis Report</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This is a research handoff report; product accuracy remains blocked by gates.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">report status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("channel_row_count", 0))}</div><div class="muted">channel/wavelength summaries</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("check_channel_row_count", 0))}</div><div class="muted">CHECK summaries</div></div>
</div>
<h2>Per-Sensor Analysis</h2>{html_table(sensor_rows, SENSOR_COLUMNS)}
<h2>CHECK Actions</h2>{html_table(action_rows, ACTION_COLUMNS)}
<h2>Channel / Wavelength Summary</h2>{html_table(channel_rows, CHANNEL_COLUMNS)}
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
    outputs["camera_e2e_analysis_report_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_analysis_report_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_analysis_report_channel_csv"] = payload["outputs"]["channel_csv"]
    outputs["camera_e2e_analysis_report_actions_csv"] = payload["outputs"]["actions_csv"]
    outputs["camera_e2e_analysis_report_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_analysis_report"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "channel_row_count": payload["channel_row_count"],
        "check_channel_row_count": payload["check_channel_row_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_report(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    flat_query_dir = package_dir / "camera_e2e_flat_sensor_query"
    flat_query = read_json(flat_query_dir / "camera_e2e_flat_sensor_query.json")
    query_rows = read_csv_rows(flat_query_dir / "camera_e2e_flat_sensor_query.csv")
    summary_rows = read_csv_rows(flat_query_dir / "camera_e2e_flat_sensor_query_summary.csv")
    use_scope_rows = read_csv_rows(package_dir / "camera_e2e_use_scope_summary" / "camera_e2e_use_scope_by_sensor.csv")
    mesh_rows = read_csv_rows(package_dir / "camera_e2e_mesh_confidence" / "camera_e2e_mesh_confidence_by_sensor.csv")
    use_scope_by_slug = index_by(use_scope_rows, "slug")
    mesh_by_slug = index_by(mesh_rows, "slug")
    sensor_rows, channel_rows, action_rows = build_sensor_rows(
        query_rows=query_rows,
        summary_rows=summary_rows,
        use_scope_by_slug=use_scope_by_slug,
        mesh_by_slug=mesh_by_slug,
    )
    validation = validate(sensor_rows, channel_rows, flat_query, mesh_by_slug)
    sensor_csv = output_dir / "camera_e2e_analysis_by_sensor.csv"
    channel_csv = output_dir / "camera_e2e_analysis_by_channel.csv"
    actions_csv = output_dir / "camera_e2e_analysis_actions.csv"
    report_json = output_dir / "camera_e2e_analysis_report.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_analysis_report_v1",
        "artifact_role": "camera_e2e_design_facing_research_analysis_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "channel_row_count": len(channel_rows),
        "action_row_count": len(action_rows),
        "check_channel_row_count": sum(1 for row in channel_rows if row.get("summary_gate") == "CHECK"),
        "product_ready_count": sum(1 for row in sensor_rows if boolish(row.get("product_ready"))),
        "analysis_gate_counts": dict(Counter(str(row.get("camera_e2e_analysis_gate", "")) for row in sensor_rows)),
        "summary_gate_counts": dict(Counter(str(row.get("summary_gate", "")) for row in channel_rows)),
        "mesh_confidence_class_counts": dict(Counter(str(row.get("mesh_confidence_class", "")) for row in sensor_rows)),
        "field_mesh_pass_total": sum(int(safe_float(row.get("field_mesh_pass_points"), 0)) for row in sensor_rows),
        "field_mesh_required_total": sum(int(safe_float(row.get("field_mesh_required_points"), 0)) for row in sensor_rows),
        "crosstalk_mesh_pass_total": sum(int(safe_float(row.get("crosstalk_mesh_pass_points"), 0)) for row in sensor_rows),
        "crosstalk_mesh_required_total": sum(int(safe_float(row.get("crosstalk_mesh_required_points"), 0)) for row in sensor_rows),
        "validation": validation,
        "outputs": {
            "json": repo_rel(report_json),
            "sensor_csv": repo_rel(sensor_csv),
            "channel_csv": repo_rel(channel_csv),
            "actions_csv": repo_rel(actions_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research": "Use PASS summaries for CameraE2E sensitivity/plumbing and CHECK summaries only after reviewing warnings.",
            "product": "Blocked until measured inputs and quantitative solver gates pass.",
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(channel_csv, channel_rows, CHANNEL_COLUMNS)
    write_csv(actions_csv, action_rows, ACTION_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, sensor_rows, channel_rows, action_rows)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = export_report(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "channel_row_count": payload["channel_row_count"],
                "check_channel_row_count": payload["check_channel_row_count"],
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
