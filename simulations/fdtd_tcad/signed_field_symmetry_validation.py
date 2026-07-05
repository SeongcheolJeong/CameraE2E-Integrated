#!/usr/bin/env python3
"""Validate signed field-LUT symmetry against direct negative-CRA native solves."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_FIELD_LUT = Path("runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.json")
DEFAULT_NEGATIVE_SUMMARY = Path(
    "runs/devsim_native_response_sweep_2d_cra_negative_r80_resolved_dti_pd_only_sidewall_liner/"
    "native_response_sweep_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("runs/signed_field_symmetry_validation_reference")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        round(finite_float(row.get("wavelength_nm")), 9),
        round(finite_float(row.get("field_x_norm")), 12),
        round(finite_float(row.get("field_z_norm")), 12),
    )


def relative_error(actual: float, predicted: float) -> float:
    scale = max(abs(actual), abs(predicted), 1.0e-30)
    return abs(actual - predicted) / scale


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, precision: int = 5) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1.0e-3 and abs(number) < 1.0e4:
        return f"{number:.{precision}g}"
    return f"{number:.{precision}e}"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    body = []
    for row in rows:
        status = "PASS" if row["case_pass"] else "FAIL"
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{fmt(row['field_x_norm'], 3)}</td>"
            f"<td>{fmt(row['field_z_norm'], 3)}</td>"
            f"<td>{fmt(row['direct_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['lut_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['total_response_rel_error'])}</td>"
            f"<td>{fmt(row['direct_split_phase_x'])}</td>"
            f"<td>{fmt(row['lut_split_phase_x'])}</td>"
            f"<td>{fmt(row['split_phase_abs_error'])}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Signed Field Symmetry Validation</title>
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
  <h1>Signed Field Symmetry Validation</h1>
  <div class="muted">Validation pass: <code>{payload['validation_pass']}</code>. Product accuracy ready: <code>false</code>.</div>
  <div class="panel">
    <strong>Gate</strong>
    <div class="muted">max total rel error {fmt(payload['max_total_response_rel_error'])} / {fmt(payload['max_total_rel_error_threshold'])}; max split abs error {fmt(payload['max_split_phase_abs_error'])} / {fmt(payload['max_split_abs_error_threshold'])}</div>
  </div>
  <div class="panel">
    <strong>Direct Negative-CRA Comparison</strong>
    <table>
      <thead>
        <tr>
          <th>Case</th><th>Field X</th><th>Field Z</th><th>Direct Total</th><th>LUT Total</th><th>Total Rel Err</th>
          <th>Direct Split</th><th>LUT Split</th><th>Split Abs Err</th><th>Status</th>
        </tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    field_lut = read_json(args.field_lut_json)
    direct_rows = read_csv(args.negative_native_summary_csv)
    lut_lookup = {row_key(row): row for row in field_lut.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for direct in direct_rows:
        key = row_key(direct)
        lut = lut_lookup.get(key)
        if lut is None:
            rows.append(
                {
                    "case": direct.get("case", ""),
                    "wavelength_nm": finite_float(direct.get("wavelength_nm")),
                    "field_x_norm": finite_float(direct.get("field_x_norm")),
                    "field_z_norm": finite_float(direct.get("field_z_norm")),
                    "direct_total_response_a_per_cm": finite_float(direct.get("photo_total_abs_delta_a_per_cm")),
                    "lut_total_response_a_per_cm": math.nan,
                    "total_response_rel_error": math.inf,
                    "direct_split_phase_x": finite_float(direct.get("photo_split_phase_x_proxy")),
                    "lut_split_phase_x": math.nan,
                    "split_phase_abs_error": math.inf,
                    "direct_left_response_a_per_cm": finite_float(direct.get("left_photo_delta_a_per_cm")),
                    "lut_left_response_a_per_cm": math.nan,
                    "direct_right_response_a_per_cm": finite_float(direct.get("right_photo_delta_a_per_cm")),
                    "lut_right_response_a_per_cm": math.nan,
                    "case_pass": False,
                    "status": "missing_lut_coordinate",
                }
            )
            continue
        direct_total = finite_float(direct.get("photo_total_abs_delta_a_per_cm"))
        lut_total = finite_float(lut.get("nominal_total_response_a_per_cm"))
        direct_split = finite_float(direct.get("photo_split_phase_x_proxy"))
        lut_split = finite_float(lut.get("nominal_split_phase_x"))
        total_rel = relative_error(direct_total, lut_total)
        split_abs = abs(direct_split - lut_split)
        case_pass = (
            total_rel <= args.max_total_rel_error
            and split_abs <= args.max_split_abs_error
        )
        rows.append(
            {
                "case": direct.get("case", ""),
                "wavelength_nm": finite_float(direct.get("wavelength_nm")),
                "field_x_norm": finite_float(direct.get("field_x_norm")),
                "field_z_norm": finite_float(direct.get("field_z_norm")),
                "direct_total_response_a_per_cm": direct_total,
                "lut_total_response_a_per_cm": lut_total,
                "total_response_rel_error": total_rel,
                "direct_split_phase_x": direct_split,
                "lut_split_phase_x": lut_split,
                "split_phase_abs_error": split_abs,
                "direct_left_response_a_per_cm": finite_float(direct.get("left_photo_delta_a_per_cm")),
                "lut_left_response_a_per_cm": finite_float(lut.get("nominal_left_response_a_per_cm")),
                "direct_right_response_a_per_cm": finite_float(direct.get("right_photo_delta_a_per_cm")),
                "lut_right_response_a_per_cm": finite_float(lut.get("nominal_right_response_a_per_cm")),
                "case_pass": case_pass,
                "status": "pass" if case_pass else "fail",
            }
        )
    max_total_rel = max((finite_float(row.get("total_response_rel_error")) for row in rows), default=math.inf)
    max_split_abs = max((finite_float(row.get("split_phase_abs_error")) for row in rows), default=math.inf)
    validation_pass = bool(rows) and all(bool(row["case_pass"]) for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "signed_field_symmetry_validation.json"
    csv_path = args.output_dir / "signed_field_symmetry_validation.csv"
    html_path = args.output_dir / "signed_field_symmetry_validation.html"
    payload = {
        "schema": "signed_field_symmetry_validation_v1",
        "product_lut_ready": False,
        "validation_pass": validation_pass,
        "field_lut_json": str(args.field_lut_json),
        "negative_native_summary_csv": str(args.negative_native_summary_csv),
        "case_count": len(rows),
        "max_total_response_rel_error": max_total_rel,
        "max_split_phase_abs_error": max_split_abs,
        "max_total_rel_error_threshold": args.max_total_rel_error,
        "max_split_abs_error_threshold": args.max_split_abs_error,
        "rows": rows,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "html": str(html_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_html(html_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-lut-json", type=Path, default=DEFAULT_FIELD_LUT)
    parser.add_argument("--negative-native-summary-csv", type=Path, default=DEFAULT_NEGATIVE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-total-rel-error", type=float, default=0.02)
    parser.add_argument("--max-split-abs-error", type=float, default=0.02)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation_pass": payload["validation_pass"],
                "case_count": payload["case_count"],
                "max_total_response_rel_error": payload["max_total_response_rel_error"],
                "max_split_phase_abs_error": payload["max_split_phase_abs_error"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )
    if not payload["validation_pass"] and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
