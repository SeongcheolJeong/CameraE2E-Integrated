#!/usr/bin/env python3
"""Validate and query CameraE2E multi-sensor ingest LUT rows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LUT_JSON = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "camera_e2e_ingest_query"

NUMERIC_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "wavelength_nm",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
]

QUERY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_radius_clamped_norm",
    "field_azimuth_deg",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "wavelength_nm",
    "color_channel",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "interpolation_method",
    "source_row_count",
    "source_field_cases",
    "source_evidence_levels",
    "evidence_gate",
    "product_lut_ready",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_float_list(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        value = finite_float(chunk.strip())
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("at least one finite query field value is required")
    return values


def sorted_unique(values: list[float]) -> list[float]:
    output: list[float] = []
    for value in sorted(value for value in values if math.isfinite(value)):
        if not output or abs(value - output[-1]) > 1e-12:
            output.append(value)
    return output


def axis_bounds(values: list[float], target: float) -> tuple[float, float]:
    values = sorted_unique(values)
    if not values:
        return math.nan, math.nan
    for value in values:
        if abs(target - value) <= 1e-12:
            return value, value
    if target <= values[0]:
        return values[0], values[0]
    if target >= values[-1]:
        return values[-1], values[-1]
    for left, right in zip(values, values[1:]):
        if left <= target <= right:
            return left, right
    return values[-1], values[-1]


def field_direction(field_x: float, field_z: float) -> tuple[float, float]:
    radius = math.hypot(field_x, field_z)
    azimuth = math.degrees(math.atan2(field_z, field_x)) if radius > 1e-12 else 0.0
    return radius, azimuth


def load_lut(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "camera_e2e_sensor_field_response_lut_v1":
        raise ValueError(f"{path} schema is {payload.get('schema')}, expected camera_e2e_sensor_field_response_lut_v1")
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} does not contain a rows array")
    return payload


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, **extra: Any) -> None:
    row = {"severity": severity, "code": code, "message": message}
    row.update(extra)
    issues.append(row)


def validate_lut(payload: dict[str, Any], tolerance: float) -> dict[str, Any]:
    rows = payload.get("rows", [])
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if payload.get("product_lut_ready") is not False:
        add_issue(warnings, "warning", "product_lut_ready_not_false", "current ingest export should not claim production readiness")
    seen: set[tuple[str, float, float, float]] = set()
    by_slug_wave: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        slug = str(row.get("slug", ""))
        wavelength = finite_float(row.get("wavelength_nm"))
        field_x = finite_float(row.get("field_x_norm"))
        field_z = finite_float(row.get("field_z_norm"))
        if not slug:
            add_issue(issues, "error", "missing_slug", "row slug is required", row_index=index)
        if not all(math.isfinite(value) for value in (wavelength, field_x, field_z)):
            add_issue(issues, "error", "bad_axis_value", "field_x_norm, field_z_norm, and wavelength_nm must be finite", row_index=index)
            continue
        key = (slug, wavelength, round(field_x, 12), round(field_z, 12))
        if key in seen:
            add_issue(issues, "error", "duplicate_slug_wavelength_field", "slug/wavelength/field coordinate must be unique", row_index=index)
        seen.add(key)
        by_slug_wave.setdefault((slug, wavelength), []).append(row)
        for column in NUMERIC_COLUMNS:
            value = finite_float(row.get(column))
            if not math.isfinite(value):
                add_issue(issues, "error", "nonfinite_numeric_value", f"{column} must be finite", row_index=index, column=column)
        lower = finite_float(row.get("relative_qe_min"))
        nominal = finite_float(row.get("relative_qe_proxy"))
        upper = finite_float(row.get("relative_qe_max"))
        if min(lower, nominal, upper) < -tolerance:
            add_issue(issues, "error", "negative_qe_bound", "relative QE bounds must be nonnegative", row_index=index)
        if lower - tolerance > nominal or nominal - tolerance > upper:
            add_issue(issues, "error", "qe_nominal_outside_bounds", "relative_qe_proxy must be inside min/max", row_index=index)
        for column in ("split_phase_x_proxy", "split_phase_z_proxy"):
            value = finite_float(row.get(column))
            if value < -1.0 - tolerance or value > 1.0 + tolerance:
                add_issue(issues, "error", "split_proxy_out_of_range", f"{column} must be inside [-1, 1]", row_index=index)
        if boolish(row.get("product_lut_ready")):
            add_issue(issues, "error", "row_product_lut_ready_true", "row-level product_lut_ready must stay false for current export", row_index=index)

    for (slug, wavelength), group in sorted(by_slug_wave.items()):
        xs = sorted_unique([finite_float(row.get("field_x_norm")) for row in group])
        zs = sorted_unique([finite_float(row.get("field_z_norm")) for row in group])
        if len(xs) < 2 or len(zs) < 2:
            add_issue(
                warnings,
                "warning",
                "sparse_field_grid",
                "query will clamp or interpolate on a sparse grid",
                slug=slug,
                wavelength_nm=wavelength,
                field_x_count=len(xs),
                field_z_count=len(zs),
            )
    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    return {
        "schema": "camera_e2e_ingest_lut_validation_v1",
        "pass": error_count == 0,
        "bad_count": error_count,
        "warning_count": len(warnings),
        "row_count": len(rows),
        "slug_count": len({str(row.get("slug", "")) for row in rows}),
        "slug_wavelength_grid_count": len(by_slug_wave),
        "issues": issues,
        "warnings": warnings,
    }


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row.get("slug", ""))
        wavelength = finite_float(row.get("wavelength_nm"))
        if slug and math.isfinite(wavelength):
            groups.setdefault((slug, wavelength), []).append(row)
    return groups


def row_lookup(group: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (
            round(finite_float(row.get("field_x_norm")), 12),
            round(finite_float(row.get("field_z_norm")), 12),
        ): row
        for row in group
    }


def source_rows(group: list[dict[str, Any]], field_x: float, field_z: float) -> tuple[list[dict[str, Any]], float, float, float, float]:
    xs = [finite_float(row.get("field_x_norm")) for row in group]
    zs = [finite_float(row.get("field_z_norm")) for row in group]
    x0, x1 = axis_bounds(xs, field_x)
    z0, z1 = axis_bounds(zs, field_z)
    lookup = row_lookup(group)
    rows = []
    for x_value in sorted_unique([x0, x1]):
        for z_value in sorted_unique([z0, z1]):
            row = lookup.get((round(x_value, 12), round(z_value, 12)))
            if row is not None:
                rows.append(row)
    return rows, x0, x1, z0, z1


def bilinear(group: list[dict[str, Any]], column: str, field_x: float, field_z: float) -> float:
    rows, x0, x1, z0, z1 = source_rows(group, field_x, field_z)
    lookup = row_lookup(group)
    if not all(math.isfinite(value) for value in (x0, x1, z0, z1)):
        return math.nan

    def at(x_value: float, z_value: float) -> float:
        row = lookup.get((round(x_value, 12), round(z_value, 12)))
        return finite_float(row.get(column)) if row else math.nan

    q00 = at(x0, z0)
    q10 = at(x1, z0)
    q01 = at(x0, z1)
    q11 = at(x1, z1)
    if not all(math.isfinite(value) for value in (q00, q10, q01, q11)):
        finite = [finite_float(row.get(column)) for row in rows if math.isfinite(finite_float(row.get(column)))]
        return sum(finite) / len(finite) if finite else math.nan
    if abs(x1 - x0) <= 1e-12 and abs(z1 - z0) <= 1e-12:
        return q00
    if abs(x1 - x0) <= 1e-12:
        t = (field_z - z0) / (z1 - z0)
        return q00 * (1.0 - t) + q01 * t
    if abs(z1 - z0) <= 1e-12:
        t = (field_x - x0) / (x1 - x0)
        return q00 * (1.0 - t) + q10 * t
    tx = (field_x - x0) / (x1 - x0)
    tz = (field_z - z0) / (z1 - z0)
    lower = q00 * (1.0 - tx) + q10 * tx
    upper = q01 * (1.0 - tx) + q11 * tx
    return lower * (1.0 - tz) + upper * tz


def combine_gate(rows: list[dict[str, Any]]) -> str:
    gates = {str(row.get("evidence_gate", "")).upper() for row in rows}
    if "FAIL" in gates:
        return "FAIL"
    if "CHECK" in gates or "MISSING" in gates or "" in gates:
        return "CHECK"
    if gates == {"PASS"}:
        return "PASS"
    return "CHECK"


def query_rows(
    payload: dict[str, Any],
    *,
    requested_slugs: list[str],
    field_x_values: list[float],
    field_z_values: list[float],
    wavelength_nm: str,
) -> list[dict[str, Any]]:
    groups = group_rows(payload["rows"])
    all_slugs = sorted({slug for slug, _wavelength in groups})
    slugs = all_slugs if not requested_slugs else requested_slugs
    requested_wavelength = None if wavelength_nm == "all" else finite_float(wavelength_nm)
    output: list[dict[str, Any]] = []
    for slug in slugs:
        wavelengths = sorted(w for s, w in groups if s == slug)
        if requested_wavelength is not None:
            wavelengths = [min(wavelengths, key=lambda value: abs(value - requested_wavelength))] if wavelengths else []
        for wavelength in wavelengths:
            group = groups.get((slug, wavelength), [])
            if not group:
                continue
            meta = group[0]
            xs = [finite_float(row.get("field_x_norm")) for row in group]
            zs = [finite_float(row.get("field_z_norm")) for row in group]
            for field_z in field_z_values:
                for field_x in field_x_values:
                    x_query = min(max(field_x, min(xs)), max(xs))
                    z_query = min(max(field_z, min(zs)), max(zs))
                    radius, azimuth = field_direction(field_x, field_z)
                    radius_clamped, _ = field_direction(x_query, z_query)
                    source, _x0, _x1, _z0, _z1 = source_rows(group, x_query, z_query)
                    nominal = max(0.0, bilinear(group, "relative_qe_proxy", x_query, z_query))
                    lower = min(max(0.0, bilinear(group, "relative_qe_min", x_query, z_query)), nominal)
                    upper = max(max(0.0, bilinear(group, "relative_qe_max", x_query, z_query)), nominal)
                    split_x = max(-1.0, min(1.0, bilinear(group, "split_phase_x_proxy", x_query, z_query)))
                    split_z = max(-1.0, min(1.0, bilinear(group, "split_phase_z_proxy", x_query, z_query)))
                    output.append(
                        {
                            "slug": slug,
                            "code": meta.get("code", ""),
                            "manufacturer": meta.get("manufacturer", ""),
                            "device_name": meta.get("device_name", ""),
                            "field_x_norm": field_x,
                            "field_z_norm": field_z,
                            "field_radius_norm": radius,
                            "field_radius_clamped_norm": radius_clamped,
                            "field_azimuth_deg": azimuth,
                            "cra_x_deg": bilinear(group, "cra_x_deg", x_query, z_query),
                            "cra_z_deg": bilinear(group, "cra_z_deg", x_query, z_query),
                            "lens_shift_x_um": bilinear(group, "lens_shift_x_um", x_query, z_query),
                            "lens_shift_z_um": bilinear(group, "lens_shift_z_um", x_query, z_query),
                            "wavelength_nm": wavelength,
                            "color_channel": meta.get("color_channel", ""),
                            "relative_qe_proxy": nominal,
                            "relative_qe_min": lower,
                            "relative_qe_max": upper,
                            "split_phase_x_proxy": split_x,
                            "split_phase_z_proxy": split_z,
                            "interpolation_method": "bilinear_camera_e2e_ingest_field_grid_clamped_v1",
                            "source_row_count": len(source),
                            "source_field_cases": ";".join(sorted({str(row.get("field_case", "")) for row in source})),
                            "source_evidence_levels": ";".join(sorted({str(row.get("evidence_level", "")) for row in source})),
                            "evidence_gate": combine_gate(source),
                            "product_lut_ready": False,
                        }
                    )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in QUERY_COLUMNS})


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if limit is not None and len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Ingest LUT Query</title>
  <style>
    body {{ margin: 24px; background: #081118; color: #e5f3ff; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    h1 {{ margin: 0 0 8px; }}
    .muted {{ color: #99b2c4; }}
    .note {{ border-left: 3px solid #ffd85a; padding-left: 12px; color: #e5f3ff; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }}
    th, td {{ border: 1px solid #244255; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #52e1ff; background: #102633; }}
    code {{ color: #d8f8ff; }}
  </style>
</head>
<body>
  <h1>CameraE2E Ingest LUT Query</h1>
  <div class="muted">Rows: <code>{len(rows)}</code>; validation pass: <code>{payload["validation"]["pass"]}</code>; product LUT ready: <code>{payload["product_lut_ready"]}</code>.</div>
  <p class="note">Query rows preserve the weakest source evidence gate from the neighboring LUT anchors. Current output is research/trend only.</p>
  {html_table(rows, QUERY_COLUMNS, limit=80)}
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_lut(args.lut_json)
    validation = validate_lut(payload, args.tolerance)
    slugs = [item.strip() for item in args.slugs.split(",") if item.strip()]
    field_x = parse_float_list(args.field_x)
    field_z = parse_float_list(args.field_z)
    rows = query_rows(
        payload,
        requested_slugs=slugs,
        field_x_values=field_x,
        field_z_values=field_z,
        wavelength_nm=args.wavelength_nm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "camera_e2e_ingest_query.json"
    csv_path = args.output_dir / "camera_e2e_ingest_query.csv"
    html_path = args.output_dir / "index.html"
    result = {
        "schema": "camera_e2e_ingest_lut_query_v1",
        "artifact_role": "camera_e2e_ingest_lut_validation_and_query",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product_lut_ready": False,
        "source_lut_json": str(args.lut_json),
        "query": {
            "slugs": slugs or "all",
            "field_x_norm": field_x,
            "field_z_norm": field_z,
            "wavelength_nm": args.wavelength_nm,
            "interpolation": "bilinear over slug/wavelength field grid, clamped to exported axes",
        },
        "validation": validation,
        "rows": rows,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "html": str(html_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_html(html_path, result)
    package_json = args.lut_json.parent.parent / "camera_e2e_lut_package.json"
    if package_json.exists():
        package = json.loads(package_json.read_text(encoding="utf-8"))
        outputs = package.setdefault("outputs", {})
        outputs["camera_e2e_ingest_query_json"] = repo_rel(json_path)
        outputs["camera_e2e_ingest_query_csv"] = repo_rel(csv_path)
        outputs["camera_e2e_ingest_query_html"] = repo_rel(html_path)
        package["latest_camera_e2e_ingest_query"] = {
            "schema": result["schema"],
            "product_lut_ready": result["product_lut_ready"],
            "validation_pass": result["validation"]["pass"],
            "validation_bad_count": result["validation"]["bad_count"],
            "query_row_count": len(result["rows"]),
            "json": repo_rel(json_path),
            "html": repo_rel(html_path),
        }
        package_json.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if validation["bad_count"] and not args.allow_validation_errors:
        raise SystemExit("validation failed; pass --allow-validation-errors to still write query output")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-json", type=Path, default=DEFAULT_LUT_JSON)
    parser.add_argument("--field-x", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--field-z", default="-1,0,1")
    parser.add_argument("--wavelength-nm", default="550")
    parser.add_argument("--slugs", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--allow-validation-errors", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "validation_pass": result["validation"]["pass"],
                "validation_bad_count": result["validation"]["bad_count"],
                "query_row_count": len(result["rows"]),
                "product_lut_ready": result["product_lut_ready"],
                "outputs": result["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
