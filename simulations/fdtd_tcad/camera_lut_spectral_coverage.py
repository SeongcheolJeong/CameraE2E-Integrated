#!/usr/bin/env python3
"""Validate camera-system field LUT wavelength coverage."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_FIELD_LUT = Path("runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.json")
DEFAULT_OPTICAL_STACK = Path("runs/optical_stack_evidence_reference/optical_stack_summary.json")
DEFAULT_OUTPUT_DIR = Path("runs/camera_lut_spectral_coverage_reference")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float_csv(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = finite_float(item)
        if not math.isfinite(value):
            raise ValueError(f"invalid float: {item}")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, precision: int = 5) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1.0e-3 and abs(number) < 1.0e4:
        return f"{number:.{precision}g}"
    return f"{number:.{precision}e}"


def wavelength_present(wavelengths: list[float], target: float, tolerance_nm: float) -> float | None:
    for wavelength in wavelengths:
        if abs(wavelength - target) <= tolerance_nm:
            return wavelength
    return None


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    body = []
    for row in rows:
        status = "PASS" if row["present"] else "MISSING"
        body.append(
            "<tr>"
            f"<td>{fmt(row['required_wavelength_nm'], 3)}</td>"
            f"<td>{fmt(row['matched_wavelength_nm'], 3)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{row['field_coordinate_count']}</td>"
            f"<td>{html.escape(str(row['signed_field_grid']))}</td>"
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Camera LUT Spectral Coverage</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #dce7ef; background: #071017; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .panel {{ border: 1px solid #244254; border-radius: 8px; padding: 16px; margin: 16px 0; background: #0b1821; }}
    .muted {{ color: #9fb4c3; }}
    code {{ color: #f7d774; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #244254; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #8fd4ff; font-weight: 650; }}
  </style>
</head>
<body>
  <h1>Camera LUT Spectral Coverage</h1>
  <div class="muted">Coverage pass: <code>{payload['coverage_pass']}</code>. Product accuracy ready: <code>false</code>.</div>
  <div class="panel">
    <strong>Summary</strong>
    <div class="muted">Available wavelengths: {html.escape(', '.join(fmt(w, 3) for w in payload['available_wavelength_nm']))} nm</div>
    <div class="muted">Missing required wavelengths: {html.escape(', '.join(fmt(w, 3) for w in payload['missing_required_wavelength_nm']))} nm</div>
  </div>
  <div class="panel">
    <strong>Required Wavelengths</strong>
    <table>
      <thead><tr><th>Required nm</th><th>Matched nm</th><th>Status</th><th>Field Coords</th><th>Signed Grid</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    field_lut = read_json(args.field_lut_json)
    rows = field_lut.get("rows", [])
    required_nm = parse_float_csv(args.required_wavelengths_nm)
    tolerance_nm = float(args.tolerance_nm)
    by_wavelength: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        wavelength = finite_float(row.get("wavelength_nm"))
        if math.isfinite(wavelength):
            by_wavelength.setdefault(wavelength, []).append(row)
    available = sorted(by_wavelength)
    row_summaries: list[dict[str, Any]] = []
    for target in required_nm:
        matched = wavelength_present(available, target, tolerance_nm)
        group = by_wavelength.get(matched, []) if matched is not None else []
        coords = {
            (
                round(finite_float(row.get("field_x_norm")), 12),
                round(finite_float(row.get("field_z_norm")), 12),
            )
            for row in group
        }
        row_summaries.append(
            {
                "required_wavelength_nm": target,
                "matched_wavelength_nm": matched if matched is not None else math.nan,
                "present": matched is not None,
                "field_coordinate_count": len(coords),
                "signed_field_grid": bool(field_lut.get("signed_field_grid", False)),
            }
        )
    missing = [
        row["required_wavelength_nm"]
        for row in row_summaries
        if not bool(row["present"])
    ]
    coordinate_counts = [
        int(row["field_coordinate_count"])
        for row in row_summaries
        if bool(row["present"])
    ]
    uniform_grid = len(set(coordinate_counts)) <= 1 if coordinate_counts else False
    optical_stack_summary: dict[str, Any] = {}
    optical_stack_required_um: list[float] = []
    optical_stack_pass = None
    if args.optical_stack_summary and args.optical_stack_summary.exists():
        optical_stack_summary = read_json(args.optical_stack_summary)
        optical_stack_required_um = [
            finite_float(value)
            for value in optical_stack_summary.get("required_wavelengths_um", [])
            if math.isfinite(finite_float(value))
        ]
        optical_stack_pass = bool(optical_stack_summary.get("evidence_pass", False))
    coverage_pass = bool(rows) and not missing and uniform_grid
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "camera_lut_spectral_coverage.json"
    csv_path = args.output_dir / "camera_lut_spectral_coverage.csv"
    html_path = args.output_dir / "camera_lut_spectral_coverage.html"
    payload = {
        "schema": "camera_lut_spectral_coverage_v1",
        "product_lut_ready": False,
        "coverage_pass": coverage_pass,
        "field_lut_json": str(args.field_lut_json),
        "optical_stack_summary": str(args.optical_stack_summary) if args.optical_stack_summary else "",
        "required_wavelength_nm": required_nm,
        "available_wavelength_nm": available,
        "missing_required_wavelength_nm": missing,
        "wavelength_tolerance_nm": tolerance_nm,
        "uniform_field_grid_per_wavelength": uniform_grid,
        "signed_field_grid": bool(field_lut.get("signed_field_grid", False)),
        "field_grid_count": field_lut.get("field_grid_count"),
        "field_z_grid_count": field_lut.get("field_z_grid_count"),
        "row_count": len(rows),
        "optical_stack_evidence_pass": optical_stack_pass,
        "optical_stack_required_wavelength_um": optical_stack_required_um,
        "rows": row_summaries,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "html": str(html_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, row_summaries)
    write_html(html_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-lut-json", type=Path, default=DEFAULT_FIELD_LUT)
    parser.add_argument("--optical-stack-summary", type=Path, default=DEFAULT_OPTICAL_STACK)
    parser.add_argument("--required-wavelengths-nm", default="450,550,650")
    parser.add_argument("--tolerance-nm", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "coverage_pass": payload["coverage_pass"],
                "required_wavelength_nm": payload["required_wavelength_nm"],
                "available_wavelength_nm": payload["available_wavelength_nm"],
                "missing_required_wavelength_nm": payload["missing_required_wavelength_nm"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    if not payload["coverage_pass"] and args.fail_on_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
