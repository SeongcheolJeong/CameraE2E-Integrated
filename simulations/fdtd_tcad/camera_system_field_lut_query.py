#!/usr/bin/env python3
"""Validate and query the dense camera-system field LUT.

This is a consumer-side contract check. It intentionally does not upgrade the
artifact to a product-accuracy LUT; it verifies that nominal/min/max channels are
numerically consistent and easy for camera simulation code to ingest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LUT_JSON = Path("runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.json")
DEFAULT_LUT_NPZ = Path("runs/camera_system_uncertainty_lut_reference/camera_system_field_lut.npz")
DEFAULT_OUTPUT_DIR = Path("runs/camera_system_field_lut_query_reference")

NUMERIC_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_radius_clamped_norm",
    "field_azimuth_deg",
    "cra_x_deg",
    "cra_z_deg",
    "wavelength_nm",
    "nominal_total_response_a_per_cm",
    "min_total_response_a_per_cm",
    "max_total_response_a_per_cm",
    "nominal_split_phase_x",
    "min_split_phase_x",
    "max_split_phase_x",
    "nominal_left_response_a_per_cm",
    "nominal_right_response_a_per_cm",
    "min_left_response_a_per_cm",
    "max_left_response_a_per_cm",
    "min_right_response_a_per_cm",
    "max_right_response_a_per_cm",
]

QUERY_COLUMNS = [
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_radius_clamped_norm",
    "field_azimuth_deg",
    "cra_x_deg",
    "cra_z_deg",
    "wavelength_nm",
    "nominal_total_response_a_per_cm",
    "min_total_response_a_per_cm",
    "max_total_response_a_per_cm",
    "nominal_split_phase_x",
    "min_split_phase_x",
    "max_split_phase_x",
    "nominal_left_response_a_per_cm",
    "nominal_right_response_a_per_cm",
    "min_left_response_a_per_cm",
    "max_left_response_a_per_cm",
    "min_right_response_a_per_cm",
    "max_right_response_a_per_cm",
    "interpolation_method",
    "source_row_count",
    "product_lut_ready",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_float_list(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = finite_float(chunk)
        if not math.isfinite(value):
            raise ValueError(f"invalid float value: {chunk}")
        values.append(value)
    if not values:
        raise ValueError("at least one query field value is required")
    return values


def responses_from_total_split(total: float, split: float) -> tuple[float, float]:
    split = max(-0.999999, min(0.999999, split))
    left = 0.5 * total * (1.0 - split)
    right = 0.5 * total * (1.0 + split)
    return left, right


def field_direction(field_x: float, field_z: float) -> tuple[float, float, float, float]:
    radius = math.hypot(field_x, field_z)
    if radius <= 1e-12:
        return radius, 0.0, 0.0, 0.0
    azimuth = math.degrees(math.atan2(field_z, field_x))
    return radius, field_x / radius, field_z / radius, azimuth


def scale_split_bounds(split_low: float, split_nominal: float, split_high: float, scale: float) -> tuple[float, float, float]:
    values = [
        max(-1.0, min(1.0, value * scale))
        for value in (split_low, split_nominal, split_high)
        if math.isfinite(value)
    ]
    if not values:
        return math.nan, math.nan, math.nan
    nominal = max(-1.0, min(1.0, split_nominal * scale)) if math.isfinite(split_nominal) else values[0]
    return min(values + [nominal]), nominal, max(values + [nominal])


def interpolate_value(points: list[tuple[float, float]], x_value: float) -> float:
    points = sorted((float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y))
    unique: list[tuple[float, float]] = []
    for point in points:
        if not unique or abs(point[0] - unique[-1][0]) > 1e-12:
            unique.append(point)
        else:
            unique[-1] = point
    if not unique:
        return math.nan
    if len(unique) == 1:
        return unique[0][1]
    if x_value <= unique[0][0]:
        return unique[0][1]
    if x_value >= unique[-1][0]:
        return unique[-1][1]
    for left, right in zip(unique, unique[1:]):
        x0, y0 = left
        x1, y1 = right
        if x0 <= x_value <= x1:
            if abs(x1 - x0) <= 1e-12:
                return y0
            t = (x_value - x0) / (x1 - x0)
            return y0 * (1.0 - t) + y1 * t
    return unique[-1][1]


def load_json_lut(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"{path} does not contain a rows array")
    return payload


def rows_from_npz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != "camera_system_field_lut_v1":
            raise ValueError(f"{path} schema is {str(data['schema'])}, expected camera_system_field_lut_v1")
        row_count = int(data["row_count"])
        for index in range(row_count):
            row: dict[str, Any] = {}
            for column in NUMERIC_COLUMNS:
                if column in data:
                    row[column] = float(data[column][index])
                elif column == "field_radius_norm":
                    row[column] = math.hypot(float(data["field_x_norm"][index]), float(data["field_z_norm"][index]))
                elif column == "field_radius_clamped_norm":
                    row[column] = min(1.0, math.hypot(float(data["field_x_norm"][index]), float(data["field_z_norm"][index])))
                elif column == "field_azimuth_deg":
                    row[column] = math.degrees(math.atan2(float(data["field_z_norm"][index]), float(data["field_x_norm"][index])))
                else:
                    row[column] = math.nan
            row["interpolation_method"] = str(data["interpolation_method"][index])
            row["anchor_case_count"] = int(data["anchor_case_count"][index])
            row["anchor_cases"] = str(data["anchor_cases"][index])
            row["bound_method"] = str(data["bound_method"][index])
            row["product_lut_ready"] = bool(data["row_product_lut_ready"][index])
            rows.append(row)
    return rows


def add_issue(issues: list[dict[str, Any]], severity: str, code: str, message: str, **extra: Any) -> None:
    issue = {"severity": severity, "code": code, "message": message}
    issue.update(extra)
    issues.append(issue)


def validate_rows(
    payload: dict[str, Any],
    npz_rows: list[dict[str, Any]] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    rows = payload.get("rows", [])
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if payload.get("schema") != "camera_system_field_lut_v1":
        add_issue(
            issues,
            "error",
            "schema_mismatch",
            "field LUT schema must be camera_system_field_lut_v1",
            actual=payload.get("schema"),
        )
    if not rows:
        add_issue(issues, "error", "empty_rows", "field LUT must contain at least one row")
    if payload.get("product_lut_ready"):
        add_issue(
            warnings,
            "warning",
            "product_lut_ready_true",
            "current reference workflow should only claim research/risk-range readiness",
        )

    by_wavelength: dict[float, list[dict[str, Any]]] = {}
    by_wavelength_z: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        wavelength = finite_float(row.get("wavelength_nm"))
        field_x = finite_float(row.get("field_x_norm"))
        field_z = finite_float(row.get("field_z_norm"))
        if not math.isfinite(wavelength):
            add_issue(issues, "error", "bad_wavelength", "wavelength_nm must be finite", row_index=index)
            continue
        if not math.isfinite(field_x):
            add_issue(issues, "error", "bad_field_x", "field_x_norm must be finite", row_index=index)
            continue
        if not math.isfinite(field_z):
            add_issue(issues, "error", "bad_field_z", "field_z_norm must be finite", row_index=index)
            continue
        by_wavelength.setdefault(wavelength, []).append(row)
        by_wavelength_z.setdefault((wavelength, field_z), []).append(row)

        for column in NUMERIC_COLUMNS:
            value = finite_float(row.get(column))
            if not math.isfinite(value):
                add_issue(
                    issues,
                    "error",
                    "nonfinite_numeric_value",
                    f"{column} must be finite",
                    row_index=index,
                    column=column,
                )

        nominal_total = finite_float(row.get("nominal_total_response_a_per_cm"))
        min_total = finite_float(row.get("min_total_response_a_per_cm"))
        max_total = finite_float(row.get("max_total_response_a_per_cm"))
        if min(nominal_total, min_total, max_total) < -tolerance:
            add_issue(issues, "error", "negative_total_response", "total response bounds must be nonnegative", row_index=index)
        if min_total - tolerance > nominal_total or nominal_total - tolerance > max_total:
            add_issue(issues, "error", "nominal_total_outside_bounds", "nominal total response must be inside min/max", row_index=index)

        nominal_split = finite_float(row.get("nominal_split_phase_x"))
        min_split = finite_float(row.get("min_split_phase_x"))
        max_split = finite_float(row.get("max_split_phase_x"))
        if min_split < -1.0 - tolerance or max_split > 1.0 + tolerance:
            add_issue(issues, "error", "split_bounds_out_of_range", "split phase bounds must be inside [-1, 1]", row_index=index)
        if min_split - tolerance > nominal_split or nominal_split - tolerance > max_split:
            add_issue(issues, "error", "nominal_split_outside_bounds", "nominal split phase must be inside min/max", row_index=index)

        nominal_left = finite_float(row.get("nominal_left_response_a_per_cm"))
        nominal_right = finite_float(row.get("nominal_right_response_a_per_cm"))
        if abs((nominal_left + nominal_right) - nominal_total) > max(tolerance, tolerance * max(1.0, nominal_total)):
            add_issue(issues, "error", "nominal_lr_sum_mismatch", "nominal left+right must equal total response", row_index=index)
        for side in ("left", "right"):
            nominal = finite_float(row.get(f"nominal_{side}_response_a_per_cm"))
            low = finite_float(row.get(f"min_{side}_response_a_per_cm"))
            high = finite_float(row.get(f"max_{side}_response_a_per_cm"))
            if low - tolerance > nominal or nominal - tolerance > high:
                add_issue(issues, "error", f"nominal_{side}_outside_bounds", f"nominal {side} response must be inside min/max", row_index=index)

    for wavelength, group in sorted(by_wavelength.items()):
        coords = [
            (round(finite_float(row.get("field_x_norm")), 12), round(finite_float(row.get("field_z_norm")), 12))
            for row in group
        ]
        if len(set(coords)) != len(coords):
            add_issue(issues, "error", "duplicate_field_coordinate", "field_x_norm/field_z_norm coordinate pairs must be unique within each wavelength", wavelength_nm=wavelength)
        xs = [coord[0] for coord in coords]
        zs = [coord[1] for coord in coords]
        if xs and (min(xs) > 0.0 + tolerance or max(xs) < 1.0 - tolerance):
            add_issue(
                warnings,
                "warning",
                "field_axis_not_full_range",
                "field_x_norm does not cover the full 0..1 range",
                wavelength_nm=wavelength,
                min_field_x_norm=min(xs),
                max_field_x_norm=max(xs),
            )
        if zs and (min(zs) > 0.0 + tolerance or max(zs) < 1.0 - tolerance):
            add_issue(
                warnings,
                "warning",
                "field_z_axis_not_full_range",
                "field_z_norm does not cover the full 0..1 range",
                wavelength_nm=wavelength,
                min_field_z_norm=min(zs),
                max_field_z_norm=max(zs),
            )
    for (wavelength, field_z), group in sorted(by_wavelength_z.items()):
        xs = [finite_float(row.get("field_x_norm")) for row in group]
        if xs != sorted(xs):
            add_issue(
                issues,
                "error",
                "field_x_axis_not_sorted_within_z_slice",
                "field_x_norm must be sorted within each wavelength/field_z_norm slice",
                wavelength_nm=wavelength,
                field_z_norm=field_z,
            )

    if npz_rows is not None:
        if len(npz_rows) != len(rows):
            add_issue(
                issues,
                "error",
                "npz_json_row_count_mismatch",
                "NPZ row count must match JSON row count",
                json_rows=len(rows),
                npz_rows=len(npz_rows),
            )
        else:
            for index, (json_row, npz_row) in enumerate(zip(rows, npz_rows)):
                for column in NUMERIC_COLUMNS:
                    json_value = finite_float(json_row.get(column))
                    npz_value = finite_float(npz_row.get(column))
                    if abs(json_value - npz_value) > max(tolerance, tolerance * max(1.0, abs(json_value))):
                        add_issue(
                            issues,
                            "error",
                            "npz_json_numeric_mismatch",
                            "NPZ numeric value must match JSON",
                            row_index=index,
                            column=column,
                            json_value=json_value,
                            npz_value=npz_value,
                        )
                        break

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    return {
        "schema": "camera_system_field_lut_validation_v1",
        "pass": error_count == 0,
        "bad_count": error_count,
        "warning_count": len(warnings),
        "row_count": len(rows),
        "wavelength_count": len(by_wavelength),
        "wavelength_nm": sorted(by_wavelength),
        "field_x_norm_min": min((finite_float(row.get("field_x_norm")) for row in rows), default=math.nan),
        "field_x_norm_max": max((finite_float(row.get("field_x_norm")) for row in rows), default=math.nan),
        "field_z_norm_min": min((finite_float(row.get("field_z_norm")) for row in rows), default=math.nan),
        "field_z_norm_max": max((finite_float(row.get("field_z_norm")) for row in rows), default=math.nan),
        "issues": issues,
        "warnings": warnings,
    }


def radial_anchors(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not group:
        return []
    min_abs_z = min(abs(finite_float(row.get("field_z_norm"), 0.0)) for row in group)
    anchors = [
        row
        for row in group
        if abs(abs(finite_float(row.get("field_z_norm"), 0.0)) - min_abs_z) <= 1e-12
    ]
    return sorted(anchors, key=lambda row: finite_float(row.get("field_x_norm")))


def sorted_unique(values: list[float]) -> list[float]:
    unique: list[float] = []
    for value in sorted(value for value in values if math.isfinite(value)):
        if not unique or abs(value - unique[-1]) > 1e-12:
            unique.append(value)
    return unique


def axis_bounds(values: list[float], target: float) -> tuple[float, float]:
    values = sorted_unique(values)
    if not values:
        return math.nan, math.nan
    if target <= values[0]:
        return values[0], values[0]
    if target >= values[-1]:
        return values[-1], values[-1]
    for left, right in zip(values, values[1:]):
        if left <= target <= right:
            return left, right
    return values[-1], values[-1]


def row_lookup(group: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (
            round(finite_float(row.get("field_x_norm")), 12),
            round(finite_float(row.get("field_z_norm")), 12),
        ): row
        for row in group
    }


def bilinear_value(group: list[dict[str, Any]], column: str, field_x: float, field_z: float) -> float:
    xs = [finite_float(row.get("field_x_norm")) for row in group]
    zs = [finite_float(row.get("field_z_norm")) for row in group]
    x0, x1 = axis_bounds(xs, field_x)
    z0, z1 = axis_bounds(zs, field_z)
    if not all(math.isfinite(value) for value in (x0, x1, z0, z1)):
        return math.nan
    lookup = row_lookup(group)

    def at(x_value: float, z_value: float) -> float:
        row = lookup.get((round(x_value, 12), round(z_value, 12)))
        return finite_float(row.get(column)) if row else math.nan

    q00 = at(x0, z0)
    q10 = at(x1, z0)
    q01 = at(x0, z1)
    q11 = at(x1, z1)
    if not all(math.isfinite(value) for value in (q00, q10, q01, q11)):
        nearest_z = min(sorted_unique(zs), key=lambda value: abs(value - field_z))
        return interpolate_value(
            [
                (finite_float(row.get("field_x_norm")), finite_float(row.get(column)))
                for row in group
                if abs(finite_float(row.get("field_z_norm")) - nearest_z) <= 1e-12
            ],
            field_x,
        )
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


def query_rows(
    rows: list[dict[str, Any]],
    field_x_values: list[float],
    field_z_values: list[float],
    wavelength_nm: float | None,
) -> list[dict[str, Any]]:
    by_wavelength: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        wavelength = finite_float(row.get("wavelength_nm"))
        if not math.isfinite(wavelength):
            continue
        if wavelength_nm is not None and abs(wavelength - wavelength_nm) > 1e-9:
            continue
        by_wavelength.setdefault(wavelength, []).append(row)
    output: list[dict[str, Any]] = []
    for wavelength, group in sorted(by_wavelength.items()):
        group = sorted(group, key=lambda row: finite_float(row.get("field_x_norm")))
        method = str(group[0].get("interpolation_method", "bilinear_dense_field_lut")) if group else ""
        min_x = min((finite_float(row.get("field_x_norm")) for row in group), default=0.0)
        max_x = max((finite_float(row.get("field_x_norm")) for row in group), default=1.0)
        min_z = min((finite_float(row.get("field_z_norm")) for row in group), default=0.0)
        max_z = max((finite_float(row.get("field_z_norm")) for row in group), default=1.0)

        for field_z in field_z_values:
            for field_x in field_x_values:
                radius, dir_x, dir_z, azimuth = field_direction(field_x, field_z)
                x_clamped = min(max_x, max(min_x, field_x))
                z_clamped = min(max_z, max(min_z, field_z))
                radius_clamped = math.hypot(x_clamped, z_clamped)
                cra_x = bilinear_value(group, "cra_x_deg", x_clamped, z_clamped)
                cra_z = bilinear_value(group, "cra_z_deg", x_clamped, z_clamped)
                nominal_total = max(0.0, bilinear_value(group, "nominal_total_response_a_per_cm", x_clamped, z_clamped))
                min_total = min(max(0.0, bilinear_value(group, "min_total_response_a_per_cm", x_clamped, z_clamped)), nominal_total)
                max_total = max(max(0.0, bilinear_value(group, "max_total_response_a_per_cm", x_clamped, z_clamped)), nominal_total)
                nominal_split = max(-1.0, min(1.0, bilinear_value(group, "nominal_split_phase_x", x_clamped, z_clamped)))
                min_split = min(max(-1.0, min(1.0, bilinear_value(group, "min_split_phase_x", x_clamped, z_clamped))), nominal_split)
                max_split = max(max(-1.0, min(1.0, bilinear_value(group, "max_split_phase_x", x_clamped, z_clamped))), nominal_split)
                nominal_left, nominal_right = responses_from_total_split(nominal_total, nominal_split)
                min_left, _ = responses_from_total_split(min_total, max_split)
                max_left, _ = responses_from_total_split(max_total, min_split)
                _, min_right = responses_from_total_split(min_total, min_split)
                _, max_right = responses_from_total_split(max_total, max_split)
                output.append(
                    {
                        "schema": "camera_system_field_lut_query_row_v1",
                        "field_x_norm": field_x,
                        "field_z_norm": field_z,
                        "field_radius_norm": radius,
                        "field_radius_clamped_norm": radius_clamped,
                        "field_azimuth_deg": azimuth,
                        "cra_x_deg": cra_x,
                        "cra_z_deg": cra_z,
                        "wavelength_nm": wavelength,
                        "nominal_total_response_a_per_cm": nominal_total,
                        "min_total_response_a_per_cm": min_total,
                        "max_total_response_a_per_cm": max_total,
                        "nominal_split_phase_x": nominal_split,
                        "min_split_phase_x": min_split,
                        "max_split_phase_x": max_split,
                        "nominal_left_response_a_per_cm": nominal_left,
                        "nominal_right_response_a_per_cm": nominal_right,
                        "min_left_response_a_per_cm": min_left,
                        "max_left_response_a_per_cm": max_left,
                        "min_right_response_a_per_cm": min_right,
                        "max_right_response_a_per_cm": max_right,
                        "interpolation_method": method,
                        "source_row_count": len(group),
                        "product_lut_ready": False,
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in QUERY_COLUMNS})


def run(args: argparse.Namespace) -> dict[str, Any]:
    lut_payload = load_json_lut(args.lut_json)
    npz_rows = rows_from_npz(args.lut_npz) if args.lut_npz and args.lut_npz.exists() else None
    validation = validate_rows(lut_payload, npz_rows=npz_rows, tolerance=args.tolerance)
    field_x_values = parse_float_list(args.field_x)
    field_z_values = parse_float_list(args.field_z)
    wavelength = None if args.wavelength_nm == "all" else finite_float(args.wavelength_nm)
    if wavelength is not None and not math.isfinite(wavelength):
        raise ValueError("--wavelength-nm must be 'all' or a finite number")
    query = query_rows(lut_payload["rows"], field_x_values, field_z_values, wavelength)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "field_lut_query.json"
    csv_path = args.output_dir / "field_lut_query.csv"
    payload = {
        "schema": "camera_system_field_lut_query_v1",
        "artifact_role": "field_lut_consumer_validation_and_query",
        "product_lut_ready": False,
        "source_lut_json": str(args.lut_json),
        "source_lut_npz": str(args.lut_npz) if args.lut_npz else "",
        "query": {
            "field_x_norm": field_x_values,
            "field_z_norm": field_z_values,
            "wavelength_nm": args.wavelength_nm,
            "interpolation": "bilinear interpolation over the exported dense field grid, clamped to the LUT field_x_norm/field_z_norm bounds",
        },
        "validation": validation,
        "accuracy_blockers": lut_payload.get("accuracy_blockers", []),
        "rows": query,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, query)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut-json", type=Path, default=DEFAULT_LUT_JSON)
    parser.add_argument("--lut-npz", type=Path, default=DEFAULT_LUT_NPZ)
    parser.add_argument("--field-x", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--field-z", default="0")
    parser.add_argument("--wavelength-nm", default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--allow-validation-errors", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    summary = {
        "schema": payload["schema"],
        "product_lut_ready": payload["product_lut_ready"],
        "validation": {
            "pass": payload["validation"]["pass"],
            "bad_count": payload["validation"]["bad_count"],
            "warning_count": payload["validation"]["warning_count"],
            "row_count": payload["validation"]["row_count"],
        },
        "query_row_count": len(payload["rows"]),
        "outputs": payload["outputs"],
    }
    print(json.dumps(summary, indent=2))
    if not payload["validation"]["pass"] and not args.allow_validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
