#!/usr/bin/env python3
"""Export readable CFA-to-Si-to-QE examples from the response trace.

The full response trace is row-oriented and intended for loaders. This exporter
creates a small review artifact: one center-field R/G/B (or clear) example per
sensor showing the exact intermediate values used to explain the CameraE2E
research QE proxy.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_response_example"

EXAMPLE_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "runtime_id",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "color_channel",
    "wavelength_nm",
    "cfa_transmission_proxy",
    "cfa_n",
    "cfa_k",
    "cfa_thickness_um",
    "si_n",
    "si_k",
    "si_thickness_um",
    "si_simple_absorption_fraction",
    "cfa_times_si_simple_fraction",
    "normalization_scale_to_runtime",
    "pixel_qe_proxy",
    "runtime_response_min",
    "runtime_response_max",
    "runtime_vs_cfa_si_simple_delta",
    "crosstalk_center_fraction",
    "output_crosstalk_fraction",
    "direct_signal_response",
    "neighbor_leakage_response",
    "combined_evidence_gate",
    "confidence_class",
    "calculation_formula",
    "calculation_path",
    "product_lut_ready",
    "example_note",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "example_row_count",
    "example_channels",
    "example_wavelengths_nm",
    "qe_proxy_min",
    "qe_proxy_max",
    "mean_normalization_scale_to_runtime",
    "gate_counts",
    "product_lut_ready_count",
    "example_gate",
    "primary_note",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]

PRIMARY_WAVELENGTH_NM = {
    "blue": 450.0,
    "green": 550.0,
    "red": 620.0,
    "clear": 550.0,
    "mono": 550.0,
}

CHANNEL_ORDER = ["red", "green", "blue", "clear", "mono"]
FORMULA = (
    "si_abs=1-exp(-(4*pi*si_k/lambda_um)*si_thickness_um); "
    "cfa_x_si=cfa_transmission_proxy*si_abs; "
    "pixel_qe_proxy=cfa_x_si*normalization_scale_to_runtime*field_cra_rolloff"
)


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def fmt(value: float, digits: int = 9) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def is_center(row: dict[str, str]) -> bool:
    return abs(safe_float(row.get("field_x_norm"), 1.0)) < 1e-12 and abs(safe_float(row.get("field_z_norm"), 1.0)) < 1e-12


def target_channel_rows(rows: list[dict[str, str]], channel: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("color_channel", "").lower() == channel]


def choose_example_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    center_rows = [row for row in rows if is_center(row)]
    source_rows = center_rows if center_rows else rows
    selected: list[dict[str, str]] = []
    seen_channels = {row.get("color_channel", "").lower() for row in source_rows}
    for channel in CHANNEL_ORDER:
        if channel not in seen_channels:
            continue
        target = PRIMARY_WAVELENGTH_NM.get(channel, 550.0)
        candidates = target_channel_rows(source_rows, channel)
        if not candidates:
            continue
        selected.append(min(candidates, key=lambda row: abs(safe_float(row.get("wavelength_nm"), target) - target)))
    return selected


def build_example_rows(package_dir: Path) -> list[dict[str, Any]]:
    trace_rows = read_csv_rows(package_dir / "camera_e2e_response_trace" / "camera_e2e_response_trace.csv")
    output: list[dict[str, Any]] = []
    for slug, rows in sorted(group_by(trace_rows, "slug").items()):
        for row in choose_example_rows(rows):
            cfa_si = safe_float(row.get("cfa_times_si_simple_fraction"))
            qe = safe_float(row.get("runtime_response_nominal"))
            scale = qe / cfa_si if math.isfinite(qe) and math.isfinite(cfa_si) and abs(cfa_si) > 1e-15 else math.nan
            channel = row.get("color_channel", "")
            note = "Center-field representative channel example."
            if not is_center(row):
                note = "No center-field row was available; nearest representative row is shown."
            if boolish(row.get("product_lut_ready")):
                note += " Product gate unexpectedly marked ready; inspect source gates."
            output.append(
                {
                    "slug": slug,
                    "code": row.get("code", ""),
                    "manufacturer": row.get("manufacturer", ""),
                    "device_name": row.get("device_name", ""),
                    "runtime_id": row.get("runtime_id", ""),
                    "field_x_norm": row.get("field_x_norm", ""),
                    "field_z_norm": row.get("field_z_norm", ""),
                    "cra_x_deg": row.get("cra_x_deg", ""),
                    "cra_z_deg": row.get("cra_z_deg", ""),
                    "color_channel": channel,
                    "wavelength_nm": row.get("wavelength_nm", ""),
                    "cfa_transmission_proxy": row.get("cfa_transmission_proxy", ""),
                    "cfa_n": row.get("cfa_n", ""),
                    "cfa_k": row.get("cfa_k", ""),
                    "cfa_thickness_um": row.get("cfa_thickness_um", ""),
                    "si_n": row.get("si_n", ""),
                    "si_k": row.get("si_k", ""),
                    "si_thickness_um": row.get("si_thickness_um", ""),
                    "si_simple_absorption_fraction": row.get("si_simple_absorption_fraction", ""),
                    "cfa_times_si_simple_fraction": row.get("cfa_times_si_simple_fraction", ""),
                    "normalization_scale_to_runtime": fmt(scale),
                    "pixel_qe_proxy": row.get("runtime_response_nominal", ""),
                    "runtime_response_min": row.get("runtime_response_min", ""),
                    "runtime_response_max": row.get("runtime_response_max", ""),
                    "runtime_vs_cfa_si_simple_delta": row.get("runtime_vs_cfa_si_simple_delta", ""),
                    "crosstalk_center_fraction": row.get("crosstalk_center_fraction", ""),
                    "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                    "direct_signal_response": row.get("direct_signal_response", ""),
                    "neighbor_leakage_response": row.get("neighbor_leakage_response", ""),
                    "combined_evidence_gate": row.get("combined_evidence_gate", ""),
                    "confidence_class": row.get("confidence_class", ""),
                    "calculation_formula": FORMULA,
                    "calculation_path": row.get("calculation_path", ""),
                    "product_lut_ready": row.get("product_lut_ready", ""),
                    "example_note": note,
                }
            )
    return output


def build_summary_rows(example_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for slug, rows in sorted(group_by([{k: str(v) for k, v in row.items()} for row in example_rows], "slug").items()):
        qes = [safe_float(row.get("pixel_qe_proxy")) for row in rows]
        scales = [safe_float(row.get("normalization_scale_to_runtime")) for row in rows]
        product_ready_count = sum(1 for row in rows if boolish(row.get("product_lut_ready")))
        gate_counts = dict(sorted(Counter(row.get("combined_evidence_gate", "") for row in rows if row.get("combined_evidence_gate")).items()))
        first = rows[0]
        output.append(
            {
                "slug": slug,
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "example_row_count": len(rows),
                "example_channels": ";".join(row.get("color_channel", "") for row in rows),
                "example_wavelengths_nm": ";".join(row.get("wavelength_nm", "") for row in rows),
                "qe_proxy_min": fmt(min((value for value in qes if math.isfinite(value)), default=math.nan)),
                "qe_proxy_max": fmt(max((value for value in qes if math.isfinite(value)), default=math.nan)),
                "mean_normalization_scale_to_runtime": fmt(mean(scales)),
                "gate_counts": json.dumps(gate_counts, sort_keys=True),
                "product_lut_ready_count": product_ready_count,
                "example_gate": "CHECK" if product_ready_count == 0 and rows else "FAIL",
                "primary_note": "Readable center-channel QE trace; not a replacement for measured stack/material or converged Meep.",
            }
        )
    return output


def write_html(path: Path, payload: dict[str, Any], example_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    example_cols = [
        "device_name",
        "color_channel",
        "wavelength_nm",
        "cfa_transmission_proxy",
        "si_simple_absorption_fraction",
        "cfa_times_si_simple_fraction",
        "normalization_scale_to_runtime",
        "pixel_qe_proxy",
        "direct_signal_response",
        "neighbor_leakage_response",
        "combined_evidence_gate",
    ]
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Response Example</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Response Example</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. These examples explain the research QE proxy construction; product use remains blocked unless measured stack/material and convergence gates pass.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("example_row_count", 0))}</div><div class="muted">example rows</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready rows</div></div>
  </div>
  <h2>Formula</h2>
  <p><code>{html_cell(FORMULA)}</code></p>
  <h2>Representative Examples</h2>
  {html_table(example_rows, example_cols)}
  <h2>Sensor Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    example_rows = build_example_rows(package_dir)
    summary_rows = build_summary_rows(example_rows)

    checks = [
        check_row(
            "response_trace_available",
            bool(example_rows),
            "PASS" if example_rows else "FAIL",
            {"example_row_count": len(example_rows)},
            "Run export_camera_e2e_response_trace.py first.",
        ),
        check_row(
            "product_gate_closed",
            sum(1 for row in example_rows if boolish(row.get("product_lut_ready"))) == 0,
            "PASS" if sum(1 for row in example_rows if boolish(row.get("product_lut_ready"))) == 0 else "FAIL",
            {"product_ready_count": sum(1 for row in example_rows if boolish(row.get("product_lut_ready")))},
            "Inspect source gates before exposing product examples.",
        ),
    ]
    pass_all = all(boolish(row["pass"]) for row in checks)
    status = "RESPONSE_EXAMPLES_READY_PRODUCT_BLOCKED" if pass_all else "FAIL"

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "camera_e2e_response_example.json"
    examples_csv = output_dir / "camera_e2e_response_example.csv"
    summary_csv = output_dir / "camera_e2e_response_example_summary.csv"
    checks_csv = output_dir / "camera_e2e_response_example_checks.csv"
    html_path = output_dir / "index.html"

    write_csv(examples_csv, example_rows, EXAMPLE_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_csv(checks_csv, checks, CHECK_COLUMNS)

    payload = {
        "schema": "camera_e2e_response_example_v1",
        "artifact_role": "readable_cfa_to_si_to_qe_examples",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(summary_rows),
        "example_row_count": len(example_rows),
        "product_ready_count": sum(1 for row in example_rows if boolish(row.get("product_lut_ready"))),
        "formula": FORMULA,
        "gate_counts": dict(sorted(Counter(row.get("combined_evidence_gate", "") for row in example_rows if row.get("combined_evidence_gate")).items())),
        "validation": {
            "schema": "camera_e2e_response_example_validation_v1",
            "pass": pass_all,
            "status": status,
            "issue_count": sum(1 for row in checks if not boolish(row["pass"])),
            "issues": [row for row in checks if not boolish(row["pass"])],
        },
        "outputs": {
            "json": repo_rel(json_path),
            "examples_csv": repo_rel(examples_csv),
            "summary_csv": repo_rel(summary_csv),
            "checks_csv": repo_rel(checks_csv),
            "html": repo_rel(html_path),
        },
    }
    write_json(json_path, payload)
    write_html(html_path, payload, example_rows, summary_rows)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_payload(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
