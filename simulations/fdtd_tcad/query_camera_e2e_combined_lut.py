#!/usr/bin/env python3
"""Query field response and compact crosstalk together for CameraE2E."""

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
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_FIELD_JSON = DEFAULT_PACKAGE_DIR / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.json"
DEFAULT_COMPACT_XT_CSV = (
    DEFAULT_PACKAGE_DIR
    / "camera_e2e_compact_crosstalk_lut"
    / "camera_e2e_compact_crosstalk_kernel_lut.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_combined_query"

FIELD_NUMERIC_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "cra_x_deg",
    "cra_z_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_x_deg",
    "cra_mismatch_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
]

QUERY_COLUMNS = [
    "query_id",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_radius_clamped_norm",
    "field_azimuth_deg",
    "wavelength_nm",
    "color_channel",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "cra_x_deg",
    "cra_z_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_x_deg",
    "cra_mismatch_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_tolerance_profile",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "cra_mismatch_gate",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "lens_shift_model",
    "cra_measurement_gate",
    "cra_input_gate",
    "cra_source",
    "field_interpolation_method",
    "field_source_row_count",
    "field_source_cases",
    "field_evidence_gate",
    "crosstalk_kernel_id",
    "crosstalk_neighborhood",
    "crosstalk_center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "crosstalk_source_field_case",
    "crosstalk_distance_norm",
    "crosstalk_evidence_gate",
    "combined_evidence_gate",
    "product_lut_ready",
]

KERNEL_COLUMNS = [
    "query_id",
    "kernel_id",
    "slug",
    "field_x_norm",
    "field_z_norm",
    "wavelength_nm",
    "color_channel",
    "dx",
    "dz",
    "response_fraction",
    "color_relation",
    "evidence_gate",
    "source",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
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


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text or "").split(","):
        value = finite_float(item.strip())
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("at least one finite value is required")
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


def field_direction(x: float, z: float) -> tuple[float, float]:
    radius = math.hypot(x, z)
    azimuth = math.degrees(math.atan2(z, x)) if radius > 1e-12 else 0.0
    return radius, azimuth


def load_field_lut(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "camera_e2e_sensor_field_response_lut_v1":
        raise ValueError(f"{path} has schema {payload.get('schema')}")
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} does not contain rows")
    return payload


def group_field_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, float, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row.get("slug", ""))
        wavelength = finite_float(row.get("wavelength_nm"))
        color = str(row.get("color_channel", ""))
        if slug and math.isfinite(wavelength):
            groups.setdefault((slug, wavelength, color), []).append(row)
    return groups


def row_lookup(group: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (round(finite_float(row.get("field_x_norm")), 12), round(finite_float(row.get("field_z_norm")), 12)): row
        for row in group
    }


def source_rows(group: list[dict[str, Any]], x: float, z: float) -> tuple[list[dict[str, Any]], float, float, float, float]:
    xs = [finite_float(row.get("field_x_norm")) for row in group]
    zs = [finite_float(row.get("field_z_norm")) for row in group]
    x0, x1 = axis_bounds(xs, x)
    z0, z1 = axis_bounds(zs, z)
    lookup = row_lookup(group)
    rows = []
    for xv in sorted_unique([x0, x1]):
        for zv in sorted_unique([z0, z1]):
            row = lookup.get((round(xv, 12), round(zv, 12)))
            if row is not None:
                rows.append(row)
    return rows, x0, x1, z0, z1


def bilinear(group: list[dict[str, Any]], column: str, x: float, z: float) -> float:
    rows, x0, x1, z0, z1 = source_rows(group, x, z)
    lookup = row_lookup(group)

    def at(xv: float, zv: float) -> float:
        row = lookup.get((round(xv, 12), round(zv, 12)))
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
        t = (z - z0) / (z1 - z0)
        return q00 * (1.0 - t) + q01 * t
    if abs(z1 - z0) <= 1e-12:
        t = (x - x0) / (x1 - x0)
        return q00 * (1.0 - t) + q10 * t
    tx = (x - x0) / (x1 - x0)
    tz = (z - z0) / (z1 - z0)
    return (q00 * (1.0 - tx) + q10 * tx) * (1.0 - tz) + (q01 * (1.0 - tx) + q11 * tx) * tz


def combine_gate(gates: list[str]) -> str:
    normalized = {str(gate or "").upper() for gate in gates}
    if "FAIL" in normalized:
        return "FAIL"
    if "CHECK" in normalized or "MISSING" in normalized or "" in normalized:
        return "CHECK"
    if normalized == {"PASS"}:
        return "PASS"
    return "CHECK"


def joined_unique(rows: list[dict[str, Any]], column: str) -> str:
    values = sorted({str(row.get(column, "")).strip() for row in rows if str(row.get(column, "")).strip()})
    return ";".join(values)


def cra_input_gate(rows: list[dict[str, Any]]) -> str:
    true_gates = {"MEASURED", "CALIBRATED", "RAYTRACE_VALIDATED"}
    explicit = {str(row.get("cra_input_gate", "")).strip().upper() for row in rows if str(row.get("cra_input_gate", "")).strip()}
    if explicit:
        return "PASS" if explicit == {"PASS"} else "CHECK"
    gates = {str(row.get("cra_measurement_gate", "")).strip().upper() for row in rows}
    if not gates or "" in gates:
        return "CHECK"
    return "PASS" if gates.issubset(true_gates) else "CHECK"


def compact_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("slug", ""),
        row.get("wavelength_nm", ""),
        row.get("color_channel", ""),
        row.get("field_case", ""),
    )


def compact_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(compact_key(row), []).append(row)
    return groups


def nearest_compact_kernel(
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]],
    slug: str,
    wavelength: float,
    color: str,
    x: float,
    z: float,
) -> tuple[str, float, list[dict[str, str]]]:
    candidates = []
    wave_text = f"{wavelength:g}"
    for key, rows in groups.items():
        k_slug, k_wave, k_color, k_case = key
        if k_slug != slug or k_color != color:
            continue
        if abs(finite_float(k_wave) - wavelength) > 1e-6:
            continue
        first = rows[0]
        distance = math.hypot(finite_float(first.get("field_x_norm"), 0.0) - x, finite_float(first.get("field_z_norm"), 0.0) - z)
        candidates.append((distance, k_case, rows))
    if not candidates:
        for key, rows in groups.items():
            k_slug, _k_wave, k_color, k_case = key
            if k_slug == slug and k_color == color:
                first = rows[0]
                distance = math.hypot(finite_float(first.get("field_x_norm"), 0.0) - x, finite_float(first.get("field_z_norm"), 0.0) - z)
                candidates.append((distance, k_case, rows))
    if not candidates:
        return "", math.nan, []
    distance, case, rows = min(candidates, key=lambda item: (item[0], item[1]))
    return case, distance, rows


def query(
    field_payload: dict[str, Any],
    compact_rows: list[dict[str, str]],
    *,
    slugs: list[str],
    field_x_values: list[float],
    field_z_values: list[float],
    wavelength_nm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    field_groups = group_field_rows(field_payload["rows"])
    compact = compact_groups(compact_rows)
    all_slugs = sorted({slug for slug, _w, _color in field_groups})
    requested_slugs = all_slugs if not slugs else slugs
    requested_waves = None if wavelength_nm == "all" else parse_float_list(wavelength_nm)
    query_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    for slug in requested_slugs:
        waves = sorted_unique([w for s, w, _color in field_groups if s == slug])
        if requested_waves is not None and waves:
            waves = sorted_unique([min(waves, key=lambda value: abs(value - requested_wave)) for requested_wave in requested_waves])
        for wavelength in waves:
            colors = sorted({color for s, w, color in field_groups if s == slug and abs(w - wavelength) <= 1e-9})
            for color in colors:
                group = field_groups.get((slug, wavelength, color), [])
                if not group:
                    continue
                meta = group[0]
                xs = [finite_float(row.get("field_x_norm")) for row in group]
                zs = [finite_float(row.get("field_z_norm")) for row in group]
                unique_xs = sorted_unique(xs)
                unique_zs = sorted_unique(zs)
                for field_z in field_z_values:
                    if len(unique_zs) < 2 and not any(abs(field_z - value) <= 1e-9 for value in unique_zs):
                        continue
                    for field_x in field_x_values:
                        if len(unique_xs) < 2 and not any(abs(field_x - value) <= 1e-9 for value in unique_xs):
                            continue
                        x_query = min(max(field_x, min(xs)), max(xs))
                        z_query = min(max(field_z, min(zs)), max(zs))
                        source, _x0, _x1, _z0, _z1 = source_rows(group, x_query, z_query)
                        radius, azimuth = field_direction(field_x, field_z)
                        radius_clamped, _ = field_direction(x_query, z_query)
                        nominal = max(0.0, bilinear(group, "relative_qe_proxy", x_query, z_query))
                        lower = min(max(0.0, bilinear(group, "relative_qe_min", x_query, z_query)), nominal)
                        upper = max(max(0.0, bilinear(group, "relative_qe_max", x_query, z_query)), nominal)
                        split_x = max(-1.0, min(1.0, bilinear(group, "split_phase_x_proxy", x_query, z_query)))
                        split_z = max(-1.0, min(1.0, bilinear(group, "split_phase_z_proxy", x_query, z_query)))
                        color = str(meta.get("color_channel", ""))
                        xt_case, xt_distance, xt_rows = nearest_compact_kernel(compact, slug, wavelength, color, x_query, z_query)
                        xt_first = xt_rows[0] if xt_rows else {}
                        qid = f"{slug}_{color}_{wavelength:g}_{field_x:g}_{field_z:g}".replace("-", "m").replace(".", "p")
                        kernel_id = f"{slug}_{color}_{wavelength:g}_{xt_case}" if xt_case else ""
                        field_gate = combine_gate([str(row.get("evidence_gate", "")) for row in source])
                        cra_gate = cra_input_gate(source)
                        xt_gate = combine_gate([str(row.get("evidence_gate", "")) for row in xt_rows]) if xt_rows else "MISSING"
                        combined_gate = combine_gate([field_gate, cra_gate, xt_gate])
                        query_rows.append(
                        {
                            "query_id": qid,
                            "slug": slug,
                            "code": meta.get("code", ""),
                            "manufacturer": meta.get("manufacturer", ""),
                            "device_name": meta.get("device_name", ""),
                            "field_x_norm": field_x,
                            "field_z_norm": field_z,
                            "field_radius_norm": radius,
                            "field_radius_clamped_norm": radius_clamped,
                            "field_azimuth_deg": azimuth,
                            "wavelength_nm": wavelength,
                            "color_channel": color,
                            "relative_qe_proxy": nominal,
                            "relative_qe_min": lower,
                            "relative_qe_max": upper,
                            "split_phase_x_proxy": split_x,
                            "split_phase_z_proxy": split_z,
                            "cra_x_deg": bilinear(group, "cra_x_deg", x_query, z_query),
                            "cra_z_deg": bilinear(group, "cra_z_deg", x_query, z_query),
                            "lens_cra_x_deg": bilinear(group, "lens_cra_x_deg", x_query, z_query),
                            "lens_cra_z_deg": bilinear(group, "lens_cra_z_deg", x_query, z_query),
                            "sensor_cra_x_deg": bilinear(group, "sensor_cra_x_deg", x_query, z_query),
                            "sensor_cra_z_deg": bilinear(group, "sensor_cra_z_deg", x_query, z_query),
                            "cra_mismatch_x_deg": bilinear(group, "cra_mismatch_x_deg", x_query, z_query),
                            "cra_mismatch_z_deg": bilinear(group, "cra_mismatch_z_deg", x_query, z_query),
                            "cra_mismatch_total_deg": bilinear(group, "cra_mismatch_total_deg", x_query, z_query),
                            "cra_mismatch_tolerance_profile": joined_unique(source, "cra_mismatch_tolerance_profile"),
                            "cra_mismatch_pass_tolerance_deg": bilinear(group, "cra_mismatch_pass_tolerance_deg", x_query, z_query),
                            "cra_mismatch_check_tolerance_deg": bilinear(group, "cra_mismatch_check_tolerance_deg", x_query, z_query),
                            "cra_mismatch_gate": joined_unique(source, "cra_mismatch_gate"),
                            "lens_shift_x_um": bilinear(group, "lens_shift_x_um", x_query, z_query),
                            "lens_shift_z_um": bilinear(group, "lens_shift_z_um", x_query, z_query),
                            "lens_shift_model": joined_unique(source, "lens_shift_model"),
                            "cra_measurement_gate": joined_unique(source, "cra_measurement_gate"),
                            "cra_input_gate": cra_gate,
                            "cra_source": joined_unique(source, "cra_source"),
                            "field_interpolation_method": "bilinear_field_lut_clamped_v1",
                            "field_source_row_count": len(source),
                            "field_source_cases": ";".join(sorted({str(row.get("field_case", "")) for row in source})),
                            "field_evidence_gate": field_gate,
                            "crosstalk_kernel_id": kernel_id,
                            "crosstalk_neighborhood": xt_first.get("neighborhood", ""),
                            "crosstalk_center_fraction": xt_first.get("center_fraction", ""),
                            "output_crosstalk_fraction": xt_first.get("output_crosstalk_fraction", ""),
                            "strongest_neighbor_fraction": xt_first.get("strongest_neighbor_fraction", ""),
                            "crosstalk_source_field_case": xt_case,
                            "crosstalk_distance_norm": xt_distance,
                            "crosstalk_evidence_gate": xt_gate,
                            "combined_evidence_gate": combined_gate,
                            "product_lut_ready": False,
                        }
                    )
                        for row in xt_rows:
                            kernel_rows.append(
                                {
                                    "query_id": qid,
                                    "kernel_id": kernel_id,
                                    "slug": slug,
                                    "field_x_norm": field_x,
                                    "field_z_norm": field_z,
                                    "wavelength_nm": wavelength,
                                    "color_channel": color,
                                    "dx": row.get("dx", ""),
                                    "dz": row.get("dz", ""),
                                    "response_fraction": row.get("response_fraction", ""),
                                    "color_relation": row.get("color_relation", ""),
                                    "evidence_gate": row.get("evidence_gate", ""),
                                    "source": row.get("source", ""),
                                }
                            )
    return query_rows, kernel_rows


def validate(query_rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    by_query: dict[str, float] = {}
    for row in kernel_rows:
        qid = row.get("query_id", "")
        by_query[qid] = by_query.get(qid, 0.0) + finite_float(row.get("response_fraction"), 0.0)
    for row in query_rows:
        qid = row.get("query_id", "")
        if row.get("crosstalk_kernel_id") and qid not in by_query:
            issues.append({"severity": "error", "code": "missing_kernel_rows", "query_id": qid})
        if qid in by_query and abs(by_query[qid] - 1.0) > tolerance:
            issues.append({"severity": "error", "code": "kernel_sum_not_one", "query_id": qid, "sum": by_query[qid]})
        if boolish(row.get("product_lut_ready")):
            issues.append({"severity": "error", "code": "product_ready_true", "query_id": qid})
    return {
        "schema": "camera_e2e_combined_lut_query_validation_v1",
        "pass": not issues,
        "bad_count": len(issues),
        "query_row_count": len(query_rows),
        "kernel_row_count": len(kernel_rows),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
    shown = rows[:limit]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Combined LUT Query</title>
  <style>
    body {{ margin:24px; background:#081118; color:#e5f3ff; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    h1 {{ margin:0 0 8px; }}
    h2 {{ margin-top:26px; color:#52e1ff; }}
    .muted {{ color:#99b2c4; }}
    .note {{ border-left:3px solid #ffd85a; padding-left:12px; color:#e5f3ff; }}
    table {{ width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }}
    th,td {{ border:1px solid #244255; padding:7px 8px; text-align:left; vertical-align:top; }}
    th {{ color:#52e1ff; background:#102633; }}
    code {{ color:#d8f8ff; }}
  </style>
</head>
<body>
  <h1>CameraE2E Combined LUT Query</h1>
  <div class="muted">Query rows: <code>{len(payload["rows"])}</code>; kernel rows: <code>{len(payload["kernel_rows"])}</code>; validation pass: <code>{payload["validation"]["pass"]}</code>.</div>
  <p class="note">Field response and compact crosstalk keep separate evidence gates. Current combined output is research/trend only.</p>
  <h2>Query Summary</h2>
  {html_table(payload["rows"], QUERY_COLUMNS, limit=80)}
  <h2>Kernel Rows</h2>
  {html_table(payload["kernel_rows"], KERNEL_COLUMNS, limit=120)}
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def write_package_links(package_dir: Path, result: dict[str, Any], output_dir: Path) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    if not package_json.exists():
        return
    package = json.loads(package_json.read_text(encoding="utf-8"))
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_combined_query_json"] = repo_rel(output_dir / "camera_e2e_combined_query.json")
    outputs["camera_e2e_combined_query_csv"] = repo_rel(output_dir / "camera_e2e_combined_query.csv")
    outputs["camera_e2e_combined_query_kernel_csv"] = repo_rel(output_dir / "camera_e2e_combined_query_kernel_rows.csv")
    outputs["camera_e2e_combined_query_html"] = repo_rel(output_dir / "index.html")
    package["latest_camera_e2e_combined_query"] = {
        "schema": result["schema"],
        "product_lut_ready": result["product_lut_ready"],
        "validation_pass": result["validation"]["pass"],
        "query_row_count": len(result["rows"]),
        "kernel_row_count": len(result["kernel_rows"]),
        "json": outputs["camera_e2e_combined_query_json"],
        "csv": outputs["camera_e2e_combined_query_csv"],
        "kernel_csv": outputs["camera_e2e_combined_query_kernel_csv"],
        "html": outputs["camera_e2e_combined_query_html"],
    }
    package_json.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    field_payload = load_field_lut(args.field_lut_json)
    compact_rows = read_csv(args.compact_crosstalk_csv)
    slugs = [item.strip() for item in args.slugs.split(",") if item.strip()]
    query_rows, kernel_rows = query(
        field_payload,
        compact_rows,
        slugs=slugs,
        field_x_values=parse_float_list(args.field_x),
        field_z_values=parse_float_list(args.field_z),
        wavelength_nm=args.wavelength_nm,
    )
    excluded_fail_query_row_count = 0
    if args.exclude_fail:
        fail_query_ids = {
            str(row.get("query_id", ""))
            for row in query_rows
            if str(row.get("combined_evidence_gate", "")).upper() == "FAIL"
        }
        excluded_fail_query_row_count = len(fail_query_ids)
        query_rows = [row for row in query_rows if str(row.get("query_id", "")) not in fail_query_ids]
        kernel_rows = [row for row in kernel_rows if str(row.get("query_id", "")) not in fail_query_ids]
    validation = validate(query_rows, kernel_rows, args.tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_csv = args.output_dir / "camera_e2e_combined_query.csv"
    kernel_csv = args.output_dir / "camera_e2e_combined_query_kernel_rows.csv"
    query_json = args.output_dir / "camera_e2e_combined_query.json"
    html_path = args.output_dir / "index.html"
    result = {
        "schema": "camera_e2e_combined_lut_query_v1",
        "artifact_role": "camera_e2e_field_response_plus_compact_crosstalk_query",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product_lut_ready": False,
        "source_field_lut_json": repo_rel(args.field_lut_json),
        "source_compact_crosstalk_csv": repo_rel(args.compact_crosstalk_csv),
        "query": {
            "slugs": slugs or "all",
            "field_x_norm": parse_float_list(args.field_x),
            "field_z_norm": parse_float_list(args.field_z),
            "wavelength_nm": args.wavelength_nm,
            "exclude_fail": args.exclude_fail,
            "excluded_fail_query_row_count": excluded_fail_query_row_count,
            "field_interpolation": "bilinear over field LUT, clamped to exported axes",
            "crosstalk_selection": "nearest compact crosstalk field_case for slug/color/wavelength",
        },
        "validation": validation,
        "rows": query_rows,
        "kernel_rows": kernel_rows,
        "outputs": {
            "json": repo_rel(query_json),
            "csv": repo_rel(query_csv),
            "kernel_csv": repo_rel(kernel_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(query_csv, query_rows, QUERY_COLUMNS)
    write_csv(kernel_csv, kernel_rows, KERNEL_COLUMNS)
    query_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(html_path, result)
    write_package_links(DEFAULT_PACKAGE_DIR, result, args.output_dir)
    if validation["bad_count"] and not args.allow_validation_errors:
        raise SystemExit("combined query validation failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-lut-json", type=Path, default=DEFAULT_FIELD_JSON)
    parser.add_argument("--compact-crosstalk-csv", type=Path, default=DEFAULT_COMPACT_XT_CSV)
    parser.add_argument("--field-x", default="-1,0,1")
    parser.add_argument("--field-z", default="-1,0,1")
    parser.add_argument("--wavelength-nm", default="550")
    parser.add_argument("--slugs", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--exclude-fail", action="store_true")
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
                "kernel_row_count": len(result["kernel_rows"]),
                "product_lut_ready": result["product_lut_ready"],
                "outputs": result["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
