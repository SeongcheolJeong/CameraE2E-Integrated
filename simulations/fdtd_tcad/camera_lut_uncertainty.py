#!/usr/bin/env python3
"""Build an uncertainty envelope around the camera-system research LUT.

This does not certify product accuracy. It gives downstream camera-system
simulation a practical nominal/min/max artifact while measured stack, n,k,
implant, and transport calibration data are still missing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RESEARCH_LUT = Path(
    "runs/native_devsim_research_lut_cra3_r80_quant/camera_system_research_lut.json"
)
DEFAULT_VARIANT_COMPARISON = Path(
    "runs/image_sensor_design_variants_reference/variant_comparison.csv"
)
DEFAULT_ACCURACY_GATE = Path("runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json")
DEFAULT_OUTPUT_DIR = Path("runs/camera_system_uncertainty_lut_reference")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_variant_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            variant_id = str(raw.get("variant_id", ""))
            if variant_id in {"", "baseline_reference"}:
                continue
            rel_change = finite_float(raw.get("total_photo_delta_rel_change"))
            split_delta = finite_float(raw.get("split_phase_delta"))
            if not math.isfinite(rel_change) and not math.isfinite(split_delta):
                continue
            rows.append(
                {
                    "variant_id": variant_id,
                    "variant_label": raw.get("variant_label", variant_id),
                    "case": raw.get("case", ""),
                    "parameter_overrides": raw.get("parameter_overrides", ""),
                    "total_photo_delta_rel_change": rel_change,
                    "split_phase_delta": split_delta,
                    "summary_json": raw.get("summary_json", ""),
                }
            )
    return rows


def accuracy_blockers(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    data = read_json(path)
    blockers = []
    for check in data.get("checks", []):
        if check.get("accuracy_blocking") and check.get("status") == "FAIL":
            blockers.append(str(check.get("name", "")))
    return blockers


def envelope_from_rows(
    rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
) -> tuple[float, float, float, float, str, list[dict[str, Any]]]:
    source_rows = rows if rows else fallback_rows
    source = "case_specific_completed_variants" if rows else "global_completed_variant_fallback"
    rel_values = [
        row["total_photo_delta_rel_change"]
        for row in source_rows
        if math.isfinite(row.get("total_photo_delta_rel_change", math.nan))
    ]
    split_values = [
        row["split_phase_delta"]
        for row in source_rows
        if math.isfinite(row.get("split_phase_delta", math.nan))
    ]
    min_rel = min(rel_values) if rel_values else 0.0
    max_rel = max(rel_values) if rel_values else 0.0
    min_split = min(split_values) if split_values else 0.0
    max_split = max(split_values) if split_values else 0.0
    return min_rel, max_rel, min_split, max_split, source, source_rows


def responses_from_total_split(total: float, split: float) -> tuple[float, float]:
    split = max(-0.999999, min(0.999999, split))
    left = 0.5 * total * (1.0 - split)
    right = 0.5 * total * (1.0 + split)
    return left, right


def envelope_total(total: float, min_rel: float, max_rel: float) -> tuple[float, float]:
    candidates = [total]
    if math.isfinite(min_rel):
        candidates.append(total * (1.0 + min_rel))
    if math.isfinite(max_rel):
        candidates.append(total * (1.0 + max_rel))
    nonnegative = [max(0.0, value) for value in candidates if math.isfinite(value)]
    return min(nonnegative), max(nonnegative)


def envelope_split(split: float, min_delta: float, max_delta: float) -> tuple[float, float]:
    candidates = [split]
    if math.isfinite(min_delta):
        candidates.append(split + min_delta)
    if math.isfinite(max_delta):
        candidates.append(split + max_delta)
    clamped = [max(-1.0, min(1.0, value)) for value in candidates if math.isfinite(value)]
    return min(clamped), max(clamped)


def build_rows(
    research_lut: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    blockers: list[str],
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in variant_rows:
        by_case.setdefault(str(row.get("case", "")), []).append(row)
    rows = []
    for summary in research_lut.get("summary_rows", []):
        case = str(summary["case"])
        total = float(summary["total_response_a_per_cm"])
        split = float(summary["split_phase_x"])
        min_rel, max_rel, min_split, max_split, source, source_rows = envelope_from_rows(
            by_case.get(case, []),
            variant_rows,
        )
        total_low, total_high = envelope_total(total, min_rel, max_rel)
        split_low, split_high = envelope_split(split, min_split, max_split)
        # Total response and split phase come from independent stress envelopes.
        # Left response decreases as split increases, while right response increases.
        left_low, _ = responses_from_total_split(total_low, split_high)
        left_high, _ = responses_from_total_split(total_high, split_low)
        _, right_low = responses_from_total_split(total_low, split_low)
        _, right_high = responses_from_total_split(total_high, split_high)
        rows.append(
            {
                "schema": "camera_system_uncertainty_lut_row_v1",
                "case": case,
                "wavelength_nm": float(summary.get("wavelength_nm", math.nan)),
                "cra_x_deg": float(summary.get("cra_x_deg", math.nan)),
                "cra_z_deg": float(summary.get("cra_z_deg", math.nan)),
                "field_x_norm": float(summary.get("field_x_norm", math.nan)),
                "field_z_norm": float(summary.get("field_z_norm", math.nan)),
                "nominal_total_response_a_per_cm": total,
                "min_total_response_a_per_cm": total_low,
                "max_total_response_a_per_cm": total_high,
                "nominal_split_phase_x": split,
                "min_split_phase_x": split_low,
                "max_split_phase_x": split_high,
                "nominal_left_response_a_per_cm": float(summary["left_response_a_per_cm"]),
                "nominal_right_response_a_per_cm": float(summary["right_response_a_per_cm"]),
                "min_left_response_a_per_cm": left_low,
                "max_left_response_a_per_cm": left_high,
                "min_right_response_a_per_cm": right_low,
                "max_right_response_a_per_cm": right_high,
                "min_total_rel_change_from_stress": min_rel,
                "max_total_rel_change_from_stress": max_rel,
                "min_split_delta_from_stress": min_split,
                "max_split_delta_from_stress": max_split,
                "stress_row_count": len(source_rows),
                "stress_source": source,
                "bound_method": "independent_total_split_stress_envelope_v1",
                "stress_variants": [
                    {
                        "variant_id": row.get("variant_id", ""),
                        "parameter_overrides": row.get("parameter_overrides", ""),
                        "case": row.get("case", ""),
                        "total_photo_delta_rel_change": row.get("total_photo_delta_rel_change"),
                        "split_phase_delta": row.get("split_phase_delta"),
                    }
                    for row in source_rows
                ],
                "accuracy_blockers": blockers,
                "product_lut_ready": False,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "wavelength_nm",
        "cra_x_deg",
        "cra_z_deg",
        "field_x_norm",
        "field_z_norm",
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
        "min_total_rel_change_from_stress",
        "max_total_rel_change_from_stress",
        "min_split_delta_from_stress",
        "max_split_delta_from_stress",
        "stress_row_count",
        "stress_source",
        "bound_method",
        "product_lut_ready",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


FIELD_LUT_COLUMNS = [
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
    "anchor_case_count",
    "anchor_cases",
    "bound_method",
    "product_lut_ready",
]

FIELD_LUT_NUMERIC_COLUMNS = [
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


def interpolation_method(anchor_count: int) -> str:
    if anchor_count <= 1:
        return "constant_single_anchor"
    return f"piecewise_linear_{anchor_count}_anchor"


def field_grid(min_value: float, max_value: float, count: int) -> list[float]:
    count = max(2, int(count))
    if not (math.isfinite(min_value) and math.isfinite(max_value)) or abs(max_value - min_value) <= 1e-12:
        return [0.0]
    return [
        min_value + (max_value - min_value) * index / (count - 1)
        for index in range(count)
    ]


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


def axis_value(anchors: list[dict[str, Any]], axis_column: str, value: float, column: str) -> float:
    return interpolate_value(
        [(finite_float(row.get(axis_column)), finite_float(row.get(column))) for row in anchors],
        value,
    )


def clamp_field_axis(value: float, max_value: float) -> float:
    return min(max_value, max(0.0, value))


def combine_positive_axis_values(x_value: float, z_value: float, center_value: float) -> float:
    if not (math.isfinite(x_value) and math.isfinite(z_value) and math.isfinite(center_value)):
        return math.nan
    if abs(center_value) <= 1e-30:
        return max(0.0, 0.5 * (x_value + z_value))
    return max(0.0, x_value * z_value / center_value)


def combined_total_bounds(
    x_low: float,
    x_nominal: float,
    x_high: float,
    z_low: float,
    z_nominal: float,
    z_high: float,
    center_nominal: float,
) -> tuple[float, float, float]:
    nominal = combine_positive_axis_values(x_nominal, z_nominal, center_nominal)
    candidates = [
        combine_positive_axis_values(x_candidate, z_candidate, center_nominal)
        for x_candidate in (x_low, x_nominal, x_high)
        for z_candidate in (z_low, z_nominal, z_high)
    ]
    candidates = [value for value in candidates if math.isfinite(value)]
    if not candidates:
        return math.nan, nominal, math.nan
    candidates.append(nominal)
    return min(candidates), nominal, max(candidates)


def combined_split_bounds(
    x_low: float,
    x_nominal: float,
    x_high: float,
    z_low: float,
    z_nominal: float,
    z_high: float,
    center_split: float,
) -> tuple[float, float, float]:
    nominal = x_nominal + z_nominal - center_split
    candidates = [
        x_candidate + z_candidate - center_split
        for x_candidate in (x_low, x_nominal, x_high)
        for z_candidate in (z_low, z_nominal, z_high)
        if math.isfinite(x_candidate) and math.isfinite(z_candidate) and math.isfinite(center_split)
    ]
    candidates.append(nominal)
    clamped = [max(-1.0, min(1.0, value)) for value in candidates if math.isfinite(value)]
    if not clamped:
        return math.nan, math.nan, math.nan
    nominal = max(-1.0, min(1.0, nominal)) if math.isfinite(nominal) else clamped[0]
    return min(clamped + [nominal]), nominal, max(clamped + [nominal])


def signed_combined_split_bounds(
    x_low: float,
    x_nominal: float,
    x_high: float,
    z_low: float,
    z_nominal: float,
    z_high: float,
    center_split: float,
    x_sign: float,
) -> tuple[float, float, float]:
    x_sign = -1.0 if x_sign < 0.0 else 1.0
    nominal = center_split + x_sign * (x_nominal - center_split) + (z_nominal - center_split)
    candidates = [
        center_split + x_sign * (x_candidate - center_split) + (z_candidate - center_split)
        for x_candidate in (x_low, x_nominal, x_high)
        for z_candidate in (z_low, z_nominal, z_high)
        if math.isfinite(x_candidate) and math.isfinite(z_candidate) and math.isfinite(center_split)
    ]
    candidates.append(nominal)
    clamped = [max(-1.0, min(1.0, value)) for value in candidates if math.isfinite(value)]
    if not clamped:
        return math.nan, math.nan, math.nan
    nominal = max(-1.0, min(1.0, nominal)) if math.isfinite(nominal) else clamped[0]
    return min(clamped + [nominal]), nominal, max(clamped + [nominal])


def safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return default
    if abs(denominator) <= 1e-30:
        return default
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else default


def diagonal_axis_t(field_x: float, field_z: float, max_x: float, max_z: float) -> float:
    x_norm = field_x / max_x if math.isfinite(max_x) and max_x > 1e-12 else 0.0
    z_norm = field_z / max_z if math.isfinite(max_z) and max_z > 1e-12 else 0.0
    return max(0.0, min(1.0, max(x_norm, z_norm)))


def diagonal_alignment_weight(field_x: float, field_z: float, max_x: float, max_z: float) -> float:
    x_norm = field_x / max_x if math.isfinite(max_x) and max_x > 1e-12 else 0.0
    z_norm = field_z / max_z if math.isfinite(max_z) and max_z > 1e-12 else 0.0
    larger = max(x_norm, z_norm)
    if larger <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, min(x_norm, z_norm) / larger))


def apply_diagonal_correction(
    base_low: float,
    base_nominal: float,
    base_high: float,
    correction_points: list[dict[str, float]],
    field_x: float,
    field_z: float,
    max_x: float,
    max_z: float,
    low_key: str,
    nominal_key: str,
    high_key: str,
    *,
    is_split: bool = False,
    correction_scale: float = 1.0,
) -> tuple[float, float, float]:
    if not correction_points:
        return base_low, base_nominal, base_high
    weight = diagonal_alignment_weight(field_x, field_z, max_x, max_z)
    if weight <= 1e-12:
        return base_low, base_nominal, base_high
    t_value = diagonal_axis_t(field_x, field_z, max_x, max_z)
    low_delta = interpolate_value([(0.0, 0.0)] + [(row["t"], row[low_key]) for row in correction_points], t_value)
    nominal_delta = interpolate_value(
        [(0.0, 0.0)] + [(row["t"], row[nominal_key]) for row in correction_points],
        t_value,
    )
    high_delta = interpolate_value(
        [(0.0, 0.0)] + [(row["t"], row[high_key]) for row in correction_points],
        t_value,
    )
    if is_split:
        low = base_low + correction_scale * weight * low_delta
        nominal = base_nominal + correction_scale * weight * nominal_delta
        high = base_high + correction_scale * weight * high_delta
        values = [max(-1.0, min(1.0, value)) for value in (low, nominal, high) if math.isfinite(value)]
        nominal = max(-1.0, min(1.0, nominal)) if math.isfinite(nominal) else (values[0] if values else math.nan)
        return min(values + [nominal]), nominal, max(values + [nominal])
    low_ratio = 1.0 + weight * low_delta
    nominal_ratio = 1.0 + weight * nominal_delta
    high_ratio = 1.0 + weight * high_delta
    low = max(0.0, base_low * low_ratio) if math.isfinite(base_low) else math.nan
    nominal = max(0.0, base_nominal * nominal_ratio) if math.isfinite(base_nominal) else math.nan
    high = max(0.0, base_high * high_ratio) if math.isfinite(base_high) else math.nan
    values = [value for value in (low, nominal, high) if math.isfinite(value)]
    if not values:
        return math.nan, math.nan, math.nan
    nominal = nominal if math.isfinite(nominal) else values[0]
    return min(values + [nominal]), nominal, max(values + [nominal])


def build_field_lut_rows(
    anchor_rows: list[dict[str, Any]],
    grid_count: int,
    field_z_grid_count: int | None = None,
    z_anchor_rows: list[dict[str, Any]] | None = None,
    diagonal_anchor_rows: list[dict[str, Any]] | None = None,
    signed_field_grid: bool = False,
) -> list[dict[str, Any]]:
    valid = [
        row
        for row in anchor_rows
        if math.isfinite(finite_float(row.get("field_x_norm")))
        and math.isfinite(finite_float(row.get("wavelength_nm")))
    ]
    if not valid:
        return []
    z_valid = [
        row
        for row in (z_anchor_rows or [])
        if math.isfinite(finite_float(row.get("field_z_norm")))
        and math.isfinite(finite_float(row.get("wavelength_nm")))
    ]
    diagonal_valid = [
        row
        for row in (diagonal_anchor_rows or [])
        if math.isfinite(finite_float(row.get("field_x_norm")))
        and math.isfinite(finite_float(row.get("field_z_norm")))
        and math.isfinite(finite_float(row.get("wavelength_nm")))
    ]
    wavelengths = sorted({finite_float(row.get("wavelength_nm")) for row in valid})
    output_rows: list[dict[str, Any]] = []
    for wavelength in wavelengths:
        group = [
            row for row in valid if abs(finite_float(row.get("wavelength_nm")) - wavelength) <= 1e-9
        ]
        z_group = [
            row for row in z_valid if abs(finite_float(row.get("wavelength_nm")) - wavelength) <= 1e-9
        ]
        diagonal_group = [
            row
            for row in diagonal_valid
            if abs(finite_float(row.get("wavelength_nm")) - wavelength) <= 1e-9
            and finite_float(row.get("field_x_norm")) > 1e-12
            and finite_float(row.get("field_z_norm")) > 1e-12
        ]
        xs = [finite_float(row.get("field_x_norm")) for row in group]
        max_radius = max(xs) if xs else 0.0
        x_anchors = sorted(
            [row for row in group if abs(finite_float(row.get("field_z_norm"), 0.0)) <= 1e-12],
            key=lambda row: finite_float(row.get("field_x_norm")),
        )
        if not x_anchors:
            x_anchors = sorted(group, key=lambda row: finite_float(row.get("field_x_norm")))
        z_anchors = sorted(
            [row for row in z_group if abs(finite_float(row.get("field_x_norm"), 0.0)) <= 1e-12],
            key=lambda row: finite_float(row.get("field_z_norm")),
        )
        if not z_anchors:
            z_anchors = x_anchors
        anchor_cases = ",".join(str(row.get("case", "")) for row in x_anchors)
        z_anchor_cases = ",".join(str(row.get("case", "")) for row in z_anchors)
        diagonal_anchor_cases = ",".join(str(row.get("case", "")) for row in diagonal_group)
        has_direct_z_axis = len({round(finite_float(row.get("field_z_norm")), 12) for row in z_anchors}) > 1
        base_method = (
            f"separable_xz_{interpolation_method(len(x_anchors))}_x_"
            f"{interpolation_method(len(z_anchors))}_z"
            if has_direct_z_axis
            else f"radial_{interpolation_method(len(x_anchors))}_split_x_projection"
        )
        max_x = max((finite_float(row.get("field_x_norm")) for row in x_anchors), default=max_radius)
        max_z = max((finite_float(row.get("field_z_norm")) for row in z_anchors), default=max_radius)
        if not has_direct_z_axis:
            max_z = max_x
        center_total = axis_value(x_anchors, "field_x_norm", 0.0, "nominal_total_response_a_per_cm")
        center_split = axis_value(x_anchors, "field_x_norm", 0.0, "nominal_split_phase_x")

        def axis_response(x_clamped: float, z_clamped: float, x_sign: float = 1.0) -> dict[str, float]:
            total_low, nominal_total, total_high = combined_total_bounds(
                axis_value(x_anchors, "field_x_norm", x_clamped, "min_total_response_a_per_cm"),
                axis_value(x_anchors, "field_x_norm", x_clamped, "nominal_total_response_a_per_cm"),
                axis_value(x_anchors, "field_x_norm", x_clamped, "max_total_response_a_per_cm"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "min_total_response_a_per_cm"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "nominal_total_response_a_per_cm"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "max_total_response_a_per_cm"),
                center_total,
            )
            split_low, nominal_split, split_high = signed_combined_split_bounds(
                axis_value(x_anchors, "field_x_norm", x_clamped, "min_split_phase_x"),
                axis_value(x_anchors, "field_x_norm", x_clamped, "nominal_split_phase_x"),
                axis_value(x_anchors, "field_x_norm", x_clamped, "max_split_phase_x"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "min_split_phase_x"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "nominal_split_phase_x"),
                axis_value(z_anchors, "field_z_norm", z_clamped, "max_split_phase_x"),
                center_split,
                x_sign,
            )
            return {
                "cra_x": axis_value(x_anchors, "field_x_norm", x_clamped, "cra_x_deg"),
                "cra_z": axis_value(z_anchors, "field_z_norm", z_clamped, "cra_z_deg"),
                "total_low": total_low,
                "nominal_total": nominal_total,
                "total_high": total_high,
                "split_low": split_low,
                "nominal_split": nominal_split,
                "split_high": split_high,
            }

        diagonal_corrections: list[dict[str, float]] = []
        if has_direct_z_axis:
            for row in diagonal_group:
                diag_x = clamp_field_axis(finite_float(row.get("field_x_norm")), max_x)
                diag_z = clamp_field_axis(finite_float(row.get("field_z_norm")), max_z)
                t_value = diagonal_axis_t(diag_x, diag_z, max_x, max_z)
                if t_value <= 1e-12:
                    continue
                base = axis_response(diag_x, diag_z)
                diagonal_corrections.append(
                    {
                        "t": t_value,
                        "total_low_ratio_delta": safe_ratio(
                            finite_float(row.get("min_total_response_a_per_cm")),
                            base["total_low"],
                        )
                        - 1.0,
                        "total_nominal_ratio_delta": safe_ratio(
                            finite_float(row.get("nominal_total_response_a_per_cm")),
                            base["nominal_total"],
                        )
                        - 1.0,
                        "total_high_ratio_delta": safe_ratio(
                            finite_float(row.get("max_total_response_a_per_cm")),
                            base["total_high"],
                        )
                        - 1.0,
                        "split_low_delta": finite_float(row.get("min_split_phase_x"))
                        - base["split_low"],
                        "split_nominal_delta": finite_float(row.get("nominal_split_phase_x"))
                        - base["nominal_split"],
                        "split_high_delta": finite_float(row.get("max_split_phase_x"))
                        - base["split_high"],
                    }
                )
            diagonal_corrections.sort(key=lambda item: item["t"])
        method = (
            f"{base_method}_diag{len(diagonal_corrections)}"
            if has_direct_z_axis and diagonal_corrections
            else base_method
        )
        if signed_field_grid:
            method = f"signed_{method}"

        def interp_radius(column: str, radius_value: float) -> float:
            return interpolate_value(
                [(finite_float(row.get("field_x_norm")), finite_float(row.get(column))) for row in x_anchors],
                radius_value,
            )

        z_count = field_z_grid_count if field_z_grid_count is not None else grid_count
        x_grid = field_grid(-max_x, max_x, grid_count) if signed_field_grid else field_grid(0.0, max_x, grid_count)
        z_grid = field_grid(-max_z, max_z, z_count) if signed_field_grid else field_grid(0.0, max_z, z_count)
        for z_value in z_grid:
            for x_value in x_grid:
                radius, dir_x, dir_z, azimuth = field_direction(x_value, z_value)
                radius_clamped = min(max(max_x, max_z), max(0.0, radius))
                if has_direct_z_axis:
                    x_sign = -1.0 if x_value < -1e-12 else 1.0
                    z_sign = -1.0 if z_value < -1e-12 else 1.0
                    x_clamped = clamp_field_axis(abs(x_value), max_x)
                    z_clamped = clamp_field_axis(abs(z_value), max_z)
                    base = axis_response(x_clamped, z_clamped, x_sign)
                    cra_x = x_sign * base["cra_x"]
                    cra_z = z_sign * base["cra_z"]
                    total_low, nominal_total, total_high = apply_diagonal_correction(
                        base["total_low"],
                        base["nominal_total"],
                        base["total_high"],
                        diagonal_corrections,
                        x_clamped,
                        z_clamped,
                        max_x,
                        max_z,
                        "total_low_ratio_delta",
                        "total_nominal_ratio_delta",
                        "total_high_ratio_delta",
                    )
                    split_low, nominal_split, split_high = apply_diagonal_correction(
                        base["split_low"],
                        base["nominal_split"],
                        base["split_high"],
                        diagonal_corrections,
                        x_clamped,
                        z_clamped,
                        max_x,
                        max_z,
                        "split_low_delta",
                        "split_nominal_delta",
                        "split_high_delta",
                        is_split=True,
                        correction_scale=x_sign,
                    )
                    bound_method = (
                        (
                            "signed_separable_xz_native_axis_with_diagonal_native_anchor_correction_stress_envelope_v1"
                            if signed_field_grid
                            else "separable_xz_native_axis_with_diagonal_native_anchor_correction_stress_envelope_v1"
                        )
                        if diagonal_corrections
                        else (
                            "signed_separable_xz_native_axis_total_and_split_stress_envelope_v1"
                            if signed_field_grid
                            else "separable_xz_native_axis_total_and_split_stress_envelope_v1"
                        )
                    )
                else:
                    cra_radial = interp_radius("cra_x_deg", radius_clamped)
                    cra_x = cra_radial * dir_x
                    cra_z = cra_radial * dir_z
                    nominal_total = max(0.0, interp_radius("nominal_total_response_a_per_cm", radius_clamped))
                    total_low = max(0.0, interp_radius("min_total_response_a_per_cm", radius_clamped))
                    total_high = max(0.0, interp_radius("max_total_response_a_per_cm", radius_clamped))
                    total_low = min(total_low, nominal_total)
                    total_high = max(total_high, nominal_total)
                    radial_split = max(-1.0, min(1.0, interp_radius("nominal_split_phase_x", radius_clamped)))
                    radial_split_low = max(-1.0, min(1.0, interp_radius("min_split_phase_x", radius_clamped)))
                    radial_split_high = max(-1.0, min(1.0, interp_radius("max_split_phase_x", radius_clamped)))
                    split_low, nominal_split, split_high = scale_split_bounds(
                        radial_split_low,
                        radial_split,
                        radial_split_high,
                        dir_x,
                    )
                    bound_method = "radial_total_split_x_projection_stress_envelope_v1"
                nominal_left, nominal_right = responses_from_total_split(nominal_total, nominal_split)
                left_low, _ = responses_from_total_split(total_low, split_high)
                left_high, _ = responses_from_total_split(total_high, split_low)
                _, right_low = responses_from_total_split(total_low, split_low)
                _, right_high = responses_from_total_split(total_high, split_high)
                output_rows.append(
                    {
                        "schema": "camera_system_field_lut_row_v1",
                        "field_x_norm": x_value,
                        "field_z_norm": z_value,
                        "field_radius_norm": radius,
                        "field_radius_clamped_norm": radius_clamped,
                        "field_azimuth_deg": azimuth,
                        "cra_x_deg": cra_x,
                        "cra_z_deg": cra_z,
                        "wavelength_nm": wavelength,
                        "nominal_total_response_a_per_cm": nominal_total,
                        "min_total_response_a_per_cm": total_low,
                        "max_total_response_a_per_cm": total_high,
                        "nominal_split_phase_x": nominal_split,
                        "min_split_phase_x": split_low,
                        "max_split_phase_x": split_high,
                        "nominal_left_response_a_per_cm": nominal_left,
                        "nominal_right_response_a_per_cm": nominal_right,
                        "min_left_response_a_per_cm": left_low,
                        "max_left_response_a_per_cm": left_high,
                        "min_right_response_a_per_cm": right_low,
                        "max_right_response_a_per_cm": right_high,
                        "interpolation_method": method,
                        "anchor_case_count": (
                            len(x_anchors)
                            + (len(z_anchors) if has_direct_z_axis else 0)
                            + (len(diagonal_corrections) if has_direct_z_axis else 0)
                        ),
                        "anchor_cases": (
                            anchor_cases
                            + (f";z:{z_anchor_cases}" if has_direct_z_axis else "")
                            + (
                                f";diag:{diagonal_anchor_cases}"
                                if has_direct_z_axis and diagonal_corrections
                                else ""
                            )
                        ),
                        "bound_method": bound_method,
                        "product_lut_ready": False,
                    }
                )
    return output_rows


def write_field_lut_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_LUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELD_LUT_COLUMNS})


def write_field_lut_npz(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    arrays: dict[str, Any] = {
        "schema": np.array(payload["schema"], dtype="U64"),
        "artifact_role": np.array(payload["artifact_role"], dtype="U64"),
        "product_lut_ready": np.array(bool(payload["product_lut_ready"])),
        "source_uncertainty_lut_json": np.array(
            str(payload["source_uncertainty_lut_json"]), dtype="U512"
        ),
        "field_grid_count": np.array(int(payload["field_grid_count"]), dtype=np.int64),
        "row_count": np.array(len(rows), dtype=np.int64),
        "numeric_columns": np.array(FIELD_LUT_NUMERIC_COLUMNS, dtype="U80"),
        "columns": np.array(FIELD_LUT_COLUMNS, dtype="U80"),
        "scope": np.array(payload["scope"], dtype="U1024"),
        "accuracy_blockers": np.array(payload.get("accuracy_blockers", []), dtype="U128"),
    }
    for column in FIELD_LUT_NUMERIC_COLUMNS:
        arrays[column] = np.array(
            [finite_float(row.get(column)) for row in rows],
            dtype=np.float64,
        )
    arrays["interpolation_method"] = np.array(
        [str(row.get("interpolation_method", "")) for row in rows],
        dtype="U64",
    )
    arrays["anchor_case_count"] = np.array(
        [int(finite_float(row.get("anchor_case_count"), 0.0)) for row in rows],
        dtype=np.int64,
    )
    arrays["anchor_cases"] = np.array(
        [str(row.get("anchor_cases", "")) for row in rows],
        dtype="U256",
    )
    arrays["bound_method"] = np.array(
        [str(row.get("bound_method", "")) for row in rows],
        dtype="U96",
    )
    arrays["row_product_lut_ready"] = np.array(
        [bool(row.get("product_lut_ready", False)) for row in rows],
        dtype=np.bool_,
    )
    np.savez_compressed(path, **arrays)


def fmt(value: Any, precision: int = 4) -> str:
    number = finite_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1e-3 and abs(number) < 1e4:
        return f"{number:.{precision}g}"
    return f"{number:.{precision}e}"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['case'])}</td>"
            f"<td>{fmt(row['cra_x_deg'])}</td>"
            f"<td>{fmt(row['nominal_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['min_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['max_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['nominal_split_phase_x'])}</td>"
            f"<td>{fmt(row['min_split_phase_x'])}</td>"
            f"<td>{fmt(row['max_split_phase_x'])}</td>"
            f"<td>{html.escape(row['stress_source'])}</td>"
            "</tr>"
        )
    blocker_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in payload.get("accuracy_blockers", [])
    )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Camera-System Uncertainty LUT</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #dce7ef; background: #071017; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: #9fb4c3; }}
    .panel {{ border: 1px solid #244254; border-radius: 8px; padding: 16px; margin: 16px 0; background: #0b1821; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #244254; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, td:last-child {{ text-align: left; }}
    th {{ color: #8fd4ff; font-weight: 650; }}
    code {{ color: #f7d774; }}
  </style>
</head>
<body>
  <h1>Camera-System Uncertainty LUT</h1>
  <div class="muted">Nominal native-DEVSIM research LUT plus stress-variant envelope. Product LUT ready: <code>{payload['product_lut_ready']}</code>.</div>
  <div class="panel">
    <strong>Scope</strong>
    <div class="muted">{html.escape(payload['scope'])}</div>
  </div>
  <div class="panel">
    <strong>Remaining accuracy blockers</strong>
    <ul>{blocker_items}</ul>
  </div>
  <div class="panel">
    <strong>Response Envelope</strong>
    <table>
      <thead>
        <tr>
          <th>Case</th><th>CRA x</th><th>Total nominal</th><th>Total min</th><th>Total max</th>
          <th>Split nominal</th><th>Split min</th><th>Split max</th><th>Source</th>
        </tr>
      </thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def write_field_html(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{fmt(row['field_x_norm'], 3)}</td>"
            f"<td>{fmt(row['field_z_norm'], 3)}</td>"
            f"<td>{fmt(row['field_radius_clamped_norm'], 3)}</td>"
            f"<td>{fmt(row['cra_x_deg'], 3)}</td>"
            f"<td>{fmt(row['cra_z_deg'], 3)}</td>"
            f"<td>{fmt(row['nominal_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['min_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['max_total_response_a_per_cm'])}</td>"
            f"<td>{fmt(row['nominal_split_phase_x'])}</td>"
            f"<td>{fmt(row['min_split_phase_x'])}</td>"
            f"<td>{fmt(row['max_split_phase_x'])}</td>"
            f"<td>{html.escape(row['interpolation_method'])}</td>"
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Camera-System Field LUT</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #dce7ef; background: #071017; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .muted {{ color: #9fb4c3; }}
    .panel {{ border: 1px solid #244254; border-radius: 8px; padding: 16px; margin: 16px 0; background: #0b1821; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #244254; padding: 8px 10px; text-align: right; }}
    th:first-child, td:last-child {{ text-align: left; }}
    th {{ color: #8fd4ff; font-weight: 650; }}
    code {{ color: #f7d774; }}
  </style>
</head>
<body>
  <h1>Camera-System Field LUT</h1>
  <div class="muted">Dense field-axis interpolation from uncertainty LUT anchors. Product LUT ready: <code>{payload['product_lut_ready']}</code>.</div>
  <div class="panel">
    <strong>Scope</strong>
    <div class="muted">{html.escape(payload['scope'])}</div>
  </div>
  <div class="panel">
    <strong>Field Response Table</strong>
    <table>
      <thead>
        <tr>
          <th>Field x</th><th>Field z</th><th>Radius</th><th>CRA x</th><th>CRA z</th><th>Total nominal</th><th>Total min</th><th>Total max</th>
          <th>Split nominal</th><th>Split min</th><th>Split max</th><th>Method</th>
        </tr>
      </thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    research_lut = read_json(args.research_lut_json)
    variant_rows = read_variant_rows(args.variant_comparison_csv)
    blockers = sorted(set(accuracy_blockers(args.accuracy_gate_json) + args.accuracy_blocker))
    rows = build_rows(research_lut, variant_rows, blockers)
    z_rows: list[dict[str, Any]] = []
    diagonal_rows: list[dict[str, Any]] = []
    z_research_lut: dict[str, Any] | None = None
    if args.z_research_lut_json is not None:
        z_research_lut = read_json(args.z_research_lut_json)
        z_rows = build_rows(z_research_lut, variant_rows, blockers)
    if args.diagonal_research_lut_json is not None:
        diagonal_research_lut = read_json(args.diagonal_research_lut_json)
        diagonal_rows = build_rows(diagonal_research_lut, variant_rows, blockers)
    field_z_grid_count = args.field_z_grid_count if args.field_z_grid_count > 0 else args.field_grid_count
    field_rows = build_field_lut_rows(
        rows,
        args.field_grid_count,
        field_z_grid_count,
        z_anchor_rows=z_rows,
        diagonal_anchor_rows=diagonal_rows,
        signed_field_grid=args.signed_field_grid,
    )
    field_methods = sorted(
        {str(row.get("interpolation_method", "")) for row in field_rows if row.get("interpolation_method")}
    )
    field_methods_by_wavelength: dict[str, dict[str, int]] = {}
    for row in field_rows:
        wavelength = finite_float(row.get("wavelength_nm"))
        wavelength_key = f"{wavelength:g}" if math.isfinite(wavelength) else "nan"
        method = str(row.get("interpolation_method", ""))
        field_methods_by_wavelength.setdefault(wavelength_key, {})
        field_methods_by_wavelength[wavelength_key][method] = (
            field_methods_by_wavelength[wavelength_key].get(method, 0) + 1
        )
    uses_radial = any("radial" in method for method in field_methods)
    uses_separable = any("separable" in method for method in field_methods)
    signed_prefix = "signed_" if args.signed_field_grid else ""
    if uses_radial and uses_separable:
        field_interpolation_scope = (
            "2D field_x_norm/field_z_norm map with mixed spectral anchoring: wavelengths with "
            "direct CRA-z anchors use a separable x/z native-axis model"
            + (" with direct diagonal correction anchors" if diagonal_rows else "")
            + ", while wavelengths without direct CRA-z anchors use radial total response and "
            "dual-x split projection over the full x/z grid."
            + (
                " Signed field grid mirrors total response and flips x-split polarity."
                if args.signed_field_grid
                else ""
            )
        )
        field_axis_model_name = f"{signed_prefix}mixed_spectral_separable_xz_and_radial_projection_v1"
    elif uses_separable:
        field_interpolation_scope = (
            "2D field_x_norm/field_z_norm map from native-DEVSIM CRA-x and direct CRA-z anchors; "
            "total response uses a separable x/z native-axis model"
            + (" with direct diagonal CRA correction anchors." if diagonal_rows else ".")
            + (
                " Signed field grid mirrors total response and flips x-split polarity."
                if args.signed_field_grid
                else ""
            )
        )
        field_axis_model_name = (
            f"{signed_prefix}separable_xz_native_axis_with_diagonal_native_anchor_correction_v1"
            if diagonal_rows
            else f"{signed_prefix}separable_xz_native_axis_total_and_split_v1"
        )
    else:
        field_interpolation_scope = (
            "2D field_x_norm/field_z_norm map from native-DEVSIM CRA-x anchors; total response "
            "follows field radius and dual-x split is projected onto the x field component."
        )
        field_axis_model_name = f"{signed_prefix}radial_total_response_with_dual_x_split_projection_v1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "camera_system_uncertainty_lut.csv"
    json_path = args.output_dir / "camera_system_uncertainty_lut.json"
    html_path = args.output_dir / "camera_system_uncertainty_lut.html"
    field_csv_path = args.output_dir / "camera_system_field_lut.csv"
    field_json_path = args.output_dir / "camera_system_field_lut.json"
    field_html_path = args.output_dir / "camera_system_field_lut.html"
    field_npz_path = args.output_dir / "camera_system_field_lut.npz"
    write_csv(csv_path, rows)
    write_field_lut_csv(field_csv_path, field_rows)
    payload = {
        "schema": "camera_system_uncertainty_lut_v1",
        "artifact_role": "uncertainty_lut",
        "product_lut_ready": False,
        "nominal_research_lut_json": str(args.research_lut_json),
        "z_axis_research_lut_json": str(args.z_research_lut_json) if args.z_research_lut_json else "",
        "diagonal_research_lut_json": (
            str(args.diagonal_research_lut_json) if args.diagonal_research_lut_json else ""
        ),
        "variant_comparison_csv": str(args.variant_comparison_csv),
        "accuracy_gate_json": str(args.accuracy_gate_json) if args.accuracy_gate_json else "",
        "scope": (
            "Applies completed stress-variant response deltas to the current native-DEVSIM "
            "research LUT. This is an uncertainty envelope, not measured accuracy certification."
        ),
        "nominal_research_lut_status": research_lut.get("research_lut_status"),
        "full_numerical_convergence_pass": research_lut.get("numerical_convergence", {}).get(
            "full_numerical_convergence_pass"
        ),
        "accuracy_blockers": blockers,
        "stress_variant_row_count": len(variant_rows),
        "rows": rows,
        "field_lut": {
            "schema": "camera_system_field_lut_v1",
            "artifact_role": "dense_field_lut",
            "field_grid_count": args.field_grid_count,
            "field_z_grid_count": field_z_grid_count,
            "signed_field_grid": bool(args.signed_field_grid),
            "row_count": len(field_rows),
            "interpolation_scope": field_interpolation_scope,
            "field_axis_model": field_axis_model_name,
            "interpolation_methods": field_methods,
            "interpolation_methods_by_wavelength": field_methods_by_wavelength,
            "outputs": {
                "json": str(field_json_path),
                "csv": str(field_csv_path),
                "html": str(field_html_path),
                "npz": str(field_npz_path),
            },
        },
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "html": str(html_path),
            "field_lut_json": str(field_json_path),
            "field_lut_csv": str(field_csv_path),
            "field_lut_html": str(field_html_path),
            "field_lut_npz": str(field_npz_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(html_path, payload)
    field_payload = {
        "schema": "camera_system_field_lut_v1",
        "artifact_role": "dense_field_lut",
        "product_lut_ready": False,
        "source_uncertainty_lut_json": str(json_path),
        "source_anchor_cases": [row.get("case") for row in rows],
        "source_z_axis_anchor_cases": [row.get("case") for row in z_rows],
        "source_diagonal_anchor_cases": [row.get("case") for row in diagonal_rows],
        "field_grid_count": args.field_grid_count,
        "field_z_grid_count": field_z_grid_count,
        "signed_field_grid": bool(args.signed_field_grid),
        "scope": (
            "Interpolates nominal/min/max native-DEVSIM response over a 2D field_x_norm/"
            "field_z_norm grid. "
            + field_interpolation_scope
            + " This is for camera-system risk simulation, not measured product certification."
        ),
        "field_axis_model": {
            "schema": "camera_system_field_axis_model_v1",
            "model_name": field_axis_model_name,
            "interpolation_methods": field_methods,
            "interpolation_methods_by_wavelength": field_methods_by_wavelength,
            "total_response": (
                "mixed by wavelength: direct-z wavelengths use separable x/z native-axis total response"
                + (" with diagonal-anchor correction" if diagonal_rows else "")
                + "; wavelengths without direct-z anchors use radial interpolation from CRA-x anchors"
                if uses_radial and uses_separable
                else (
                (
                    "separable product of piecewise-linear native-DEVSIM CRA-x and CRA-z axes, "
                    "multiplied by interpolated direct diagonal-anchor correction"
                    + (
                        "; mirrored with even symmetry across signed field axes"
                        if args.signed_field_grid
                        else ""
                    )
                    if diagonal_rows
                    else "separable product of piecewise-linear native-DEVSIM CRA-x and CRA-z total-response axes"
                )
                if z_rows
                else "piecewise linear in field_radius_norm from native-DEVSIM CRA-x anchors"
                )
            ),
            "split_phase_x": (
                "mixed by wavelength: direct-z wavelengths combine x/z split terms"
                + (" with diagonal residual correction" if diagonal_rows else "")
                + "; wavelengths without direct-z anchors project radial split onto field_x/radius"
                if uses_radial and uses_separable
                else (
                (
                    "additive x-axis and z-axis split deviations from center, plus interpolated direct diagonal split residual"
                    + (
                        "; x-axis and diagonal split residuals change sign for negative field_x"
                        if args.signed_field_grid
                        else ""
                    )
                    if diagonal_rows
                    else "additive x-axis and z-axis split deviations from center; z-axis terms preserve measured residual asymmetry"
                )
                if z_rows
                else "radial split response multiplied by field_x_norm / field_radius_norm"
                )
            ),
            "cra_components": (
                "mixed by wavelength: direct-z wavelengths interpolate native CRA x/z axes; radial wavelengths project CRA magnitude onto x/z"
                if uses_radial and uses_separable
                else (
                "CRA x and CRA z interpolated from direct native-axis anchors"
                if z_rows
                else "radial CRA magnitude projected onto x/z components"
                )
            ),
            "radius_clamp": "field axes are clamped to the largest simulated anchor values",
        },
        "accuracy_blockers": blockers,
        "rows": field_rows,
        "outputs": {
            "json": str(field_json_path),
            "csv": str(field_csv_path),
            "html": str(field_html_path),
            "npz": str(field_npz_path),
        },
    }
    field_json_path.write_text(json.dumps(field_payload, indent=2), encoding="utf-8")
    write_field_html(field_html_path, field_payload)
    write_field_lut_npz(field_npz_path, field_payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-lut-json", type=Path, default=DEFAULT_RESEARCH_LUT)
    parser.add_argument(
        "--z-research-lut-json",
        type=Path,
        default=None,
        help="Optional direct CRA-z native-DEVSIM research LUT used to anchor the field_z axis.",
    )
    parser.add_argument(
        "--diagonal-research-lut-json",
        type=Path,
        default=None,
        help="Optional diagonal CRA native-DEVSIM research LUT used to correct off-axis field response.",
    )
    parser.add_argument("--variant-comparison-csv", type=Path, default=DEFAULT_VARIANT_COMPARISON)
    parser.add_argument("--accuracy-gate-json", type=Path, default=None)
    parser.add_argument(
        "--accuracy-blocker",
        action="append",
        default=[],
        help="Accuracy blocker label to embed without depending on the generated accuracy-gate artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-grid-count", type=int, default=21)
    parser.add_argument(
        "--field-z-grid-count",
        type=int,
        default=0,
        help="Number of field_z_norm samples. Defaults to --field-grid-count when <= 0.",
    )
    parser.add_argument(
        "--signed-field-grid",
        action="store_true",
        help="Export field_x_norm and field_z_norm over -max..+max using symmetry from positive-axis native anchors.",
    )
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "schema",
                    "product_lut_ready",
                    "stress_variant_row_count",
                    "field_lut",
                    "outputs",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
