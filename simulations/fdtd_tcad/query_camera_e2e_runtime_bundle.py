#!/usr/bin/env python3
"""Query the consolidated CameraE2E runtime bundle.

This is the preferred consumer-facing lookup tool. It interpolates runtime
response rows, blends the associated crosstalk kernels, and enforces research
vs product-use gates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_JSON = (
    ROOT
    / "runs"
    / "camera_e2e_sensor_lut_package"
    / "camera_e2e_runtime_bundle"
    / "camera_e2e_runtime_bundle.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "camera_e2e_runtime_query"

QUERY_COLUMNS = [
    "runtime_query_id",
    "mode",
    "query_allowed",
    "query_gate",
    "blockers",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "requested_field_x_norm",
    "requested_field_z_norm",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "requested_wavelength_nm",
    "wavelength_nm",
    "wavelength_distance_nm",
    "color_channel",
    "response_nominal",
    "response_min",
    "response_max",
    "response_uncertainty_half_range",
    "direct_signal_response",
    "neighbor_leakage_response",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "crosstalk_uncertainty_min",
    "crosstalk_uncertainty_max",
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
    "field_interpolation_method",
    "source_runtime_ids",
    "source_weight_count",
    "field_evidence_gate",
    "crosstalk_evidence_gate",
    "combined_evidence_gate",
    "research_ingest_gate",
    "production_lut_gate",
    "confidence_class",
    "uncertainty_policy",
    "product_lut_ready",
]

KERNEL_COLUMNS = [
    "runtime_query_id",
    "slug",
    "wavelength_nm",
    "color_channel",
    "dx",
    "dz",
    "response_fraction",
    "response_fraction_min",
    "response_fraction_max",
    "color_relation",
    "evidence_gate",
    "source_runtime_ids",
]

NUMERIC_RUNTIME_COLUMNS = [
    "field_radius_norm",
    "field_azimuth_deg",
    "response_nominal",
    "response_min",
    "response_max",
    "response_uncertainty_half_range",
    "direct_signal_response",
    "neighbor_leakage_response",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "crosstalk_uncertainty_min",
    "crosstalk_uncertainty_max",
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
]

PASS_VALUES = {"PASS", "TRUE", "1", "YES"}


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().upper() in PASS_VALUES


def gate(value: Any, default: str = "MISSING") -> str:
    text = str(value if value is not None else "").strip().upper()
    return text or default


def combine_gate(gates: list[str]) -> str:
    normalized = {gate(value) for value in gates}
    if "FAIL" in normalized:
        return "FAIL"
    if "MISSING" in normalized:
        return "MISSING"
    if normalized == {"PASS"}:
        return "PASS"
    return "CHECK"


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
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


def field_direction(x: float, z: float) -> tuple[float, float]:
    radius = math.hypot(x, z)
    azimuth = math.degrees(math.atan2(z, x)) if radius > 1e-12 else 0.0
    return radius, azimuth


def load_bundle(bundle_json: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    payload = read_json(bundle_json)
    if payload.get("schema") != "camera_e2e_runtime_bundle_v1":
        raise ValueError(f"{bundle_json} schema is {payload.get('schema')}")
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"{bundle_json} does not contain outputs")
    runtime_csv = ROOT / str(outputs.get("runtime_csv", ""))
    kernel_csv = ROOT / str(outputs.get("kernel_csv", ""))
    runtime_rows = read_csv(runtime_csv)
    kernel_rows = read_csv(kernel_csv)
    if not runtime_rows:
        raise ValueError(f"runtime CSV is empty or missing: {runtime_csv}")
    if not kernel_rows:
        raise ValueError(f"kernel CSV is empty or missing: {kernel_csv}")
    return payload, runtime_rows, kernel_rows


def runtime_groups(rows: list[dict[str, str]]) -> dict[tuple[str, float, str], list[dict[str, str]]]:
    groups: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        slug = row.get("slug", "")
        wavelength = finite_float(row.get("wavelength_nm"))
        color = str(row.get("color_channel", "")).strip()
        if slug and math.isfinite(wavelength):
            groups[(slug, wavelength, color)].append(row)
    return groups


def kernel_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("runtime_id", "")].append(row)
    return groups


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


def row_lookup(group: list[dict[str, str]]) -> dict[tuple[float, float], dict[str, str]]:
    return {
        (round(finite_float(row.get("field_x_norm")), 12), round(finite_float(row.get("field_z_norm")), 12)): row
        for row in group
    }


def interpolation_weights(group: list[dict[str, str]], x: float, z: float) -> tuple[list[tuple[dict[str, str], float]], float, float, str]:
    xs = [finite_float(row.get("field_x_norm")) for row in group]
    zs = [finite_float(row.get("field_z_norm")) for row in group]
    x0, x1 = axis_bounds(xs, x)
    z0, z1 = axis_bounds(zs, z)
    xq = min(max(x, min(xs)), max(xs))
    zq = min(max(z, min(zs)), max(zs))
    lookup = row_lookup(group)
    corners: list[tuple[float, float]] = []
    for xv in sorted_unique([x0, x1]):
        for zv in sorted_unique([z0, z1]):
            corners.append((xv, zv))
    rows = [(lookup[(round(xv, 12), round(zv, 12))], xv, zv) for xv, zv in corners if (round(xv, 12), round(zv, 12)) in lookup]
    if not rows:
        nearest = min(
            group,
            key=lambda row: math.hypot(finite_float(row.get("field_x_norm"), 0.0) - xq, finite_float(row.get("field_z_norm"), 0.0) - zq),
        )
        return [(nearest, 1.0)], xq, zq, "nearest_runtime_anchor"
    if len(rows) == 1:
        return [(rows[0][0], 1.0)], xq, zq, "exact_or_clamped_runtime_anchor"
    if len(rows) == 4 and abs(x1 - x0) > 1e-12 and abs(z1 - z0) > 1e-12:
        weights_by_coord = {
            (x0, z0): (x1 - xq) * (z1 - zq),
            (x1, z0): (xq - x0) * (z1 - zq),
            (x0, z1): (x1 - xq) * (zq - z0),
            (x1, z1): (xq - x0) * (zq - z0),
        }
        denom = (x1 - x0) * (z1 - z0)
        weighted = [(row, max(0.0, weights_by_coord[(xv, zv)] / denom)) for row, xv, zv in rows]
        total = sum(weight for _row, weight in weighted)
        if total > 0:
            return [(row, weight / total) for row, weight in weighted], xq, zq, "bilinear_runtime_grid"
    if len(rows) == 2:
        (_row_a, xa, za), (_row_b, xb, zb) = rows
        if abs(za - zb) <= 1e-12 and abs(xb - xa) > 1e-12:
            tb = min(1.0, max(0.0, (xq - xa) / (xb - xa)))
            return [(rows[0][0], 1.0 - tb), (rows[1][0], tb)], xq, zq, "linear_x_runtime_grid"
        if abs(xa - xb) <= 1e-12 and abs(zb - za) > 1e-12:
            tb = min(1.0, max(0.0, (zq - za) / (zb - za)))
            return [(rows[0][0], 1.0 - tb), (rows[1][0], tb)], xq, zq, "linear_z_runtime_grid"
    inv: list[tuple[dict[str, str], float]] = []
    for row, xv, zv in rows:
        distance = math.hypot(xv - xq, zv - zq)
        inv.append((row, 1.0 / max(distance, 1e-9)))
    total = sum(weight for _row, weight in inv)
    return [(row, weight / total) for row, weight in inv], xq, zq, "sparse_inverse_distance_runtime_grid"


def weighted_numeric(weights: list[tuple[dict[str, str], float]], column: str, default: float = math.nan) -> float:
    total = 0.0
    used = 0.0
    for row, weight in weights:
        value = finite_float(row.get(column))
        if math.isfinite(value):
            total += value * weight
            used += weight
    return total / used if used > 0 else default


def joined_values(weights: list[tuple[dict[str, str], float]], column: str) -> str:
    return ";".join(sorted({str(row.get(column, "")).strip() for row, _weight in weights if str(row.get(column, "")).strip()}))


def nearest_wavelength(waves: list[float], requested: float | None) -> float:
    if requested is None:
        raise ValueError("nearest_wavelength requires a requested value")
    return min(waves, key=lambda value: abs(value - requested))


def mode_gate(mode: str, bundle: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str, str]:
    if mode == "product":
        blockers = []
        if bundle.get("product_lut_ready") is not True:
            blockers.append("bundle product_lut_ready is false")
        if gate(row.get("production_lut_gate")) != "PASS":
            blockers.append(f"production_lut_gate is {row.get('production_lut_gate')}")
        if not boolish(row.get("production_ingest_allowed")):
            blockers.append("production_ingest_allowed is false")
        if blockers:
            return False, "FAIL", "; ".join(blockers)
        return True, "PASS", ""
    blockers = []
    if not boolish(row.get("research_ingest_allowed")):
        blockers.append("research_ingest_allowed is false")
    query_gate = combine_gate([row.get("combined_evidence_gate", ""), row.get("research_ingest_gate", "")])
    if blockers:
        return False, "FAIL", "; ".join(blockers)
    return True, query_gate, ""


def query_rows(
    bundle: dict[str, Any],
    runtime_rows: list[dict[str, str]],
    kernels: list[dict[str, str]],
    *,
    slugs: list[str],
    field_x_values: list[float],
    field_z_values: list[float],
    wavelength_nm: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = runtime_groups(runtime_rows)
    kernels_by_runtime = kernel_groups(kernels)
    all_slugs = sorted({slug for slug, _wave, _color in groups})
    requested_slugs = slugs or all_slugs
    query_all_waves = wavelength_nm == "all"
    requested_waves = None if query_all_waves else parse_float_list(wavelength_nm)
    query_output: list[dict[str, Any]] = []
    kernel_output: list[dict[str, Any]] = []
    for slug in requested_slugs:
        waves = sorted_unique([wave for group_slug, wave, _color in groups if group_slug == slug])
        if not waves:
            continue
        selected_waves = (
            waves
            if query_all_waves
            else sorted_unique([nearest_wavelength(waves, requested_wave) for requested_wave in requested_waves or []])
        )
        for wave in selected_waves:
            colors = sorted({color for group_slug, group_wave, color in groups if group_slug == slug and abs(group_wave - wave) <= 1e-9})
            for color in colors:
                group = groups.get((slug, wave, color), [])
                if not group:
                    continue
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
                        weights, resolved_x, resolved_z, method = interpolation_weights(group, field_x, field_z)
                        radius, azimuth = field_direction(field_x, field_z)
                        source_ids = [row.get("runtime_id", "") for row, _weight in weights]
                        weighted_row = {
                            "production_lut_gate": combine_gate([row.get("production_lut_gate", "") for row, _weight in weights]),
                            "production_ingest_allowed": all(boolish(row.get("production_ingest_allowed")) for row, _weight in weights),
                            "research_ingest_gate": combine_gate([row.get("research_ingest_gate", "") for row, _weight in weights]),
                            "research_ingest_allowed": all(boolish(row.get("research_ingest_allowed")) for row, _weight in weights),
                            "combined_evidence_gate": combine_gate([row.get("combined_evidence_gate", "") for row, _weight in weights]),
                        }
                        allowed, q_gate, blockers = mode_gate(mode, bundle, weighted_row)
                        qid = f"{slug}_{color}_{wave:g}_{field_x:g}_{field_z:g}_{mode}".replace("-", "m").replace(".", "p")
                        response_min = weighted_numeric(weights, "response_min", 0.0)
                        response_nominal = weighted_numeric(weights, "response_nominal", 0.0)
                        response_max = weighted_numeric(weights, "response_max", response_nominal)
                        response_min, response_max = sorted((response_min, response_max))
                        row = {
                            "runtime_query_id": qid,
                            "mode": mode,
                            "query_allowed": allowed,
                            "query_gate": q_gate,
                            "blockers": blockers,
                            "slug": slug,
                            "code": joined_values(weights, "code"),
                            "manufacturer": joined_values(weights, "manufacturer"),
                            "device_name": joined_values(weights, "device_name"),
                            "requested_field_x_norm": field_x,
                            "requested_field_z_norm": field_z,
                            "field_x_norm": resolved_x,
                            "field_z_norm": resolved_z,
                            "field_radius_norm": radius,
                            "field_azimuth_deg": azimuth,
                            "requested_wavelength_nm": "all" if query_all_waves else wavelength_nm,
                            "wavelength_nm": wave,
                            "wavelength_distance_nm": 0.0
                            if query_all_waves
                            else min(abs(wave - requested_wave) for requested_wave in requested_waves or [wave]),
                            "color_channel": color or joined_values(weights, "color_channel"),
                            "response_nominal": response_nominal,
                            "response_min": response_min,
                            "response_max": response_max,
                            "response_uncertainty_half_range": 0.5 * max(0.0, response_max - response_min),
                            "direct_signal_response": weighted_numeric(weights, "direct_signal_response", 0.0),
                            "neighbor_leakage_response": weighted_numeric(weights, "neighbor_leakage_response", 0.0),
                            "output_crosstalk_fraction": weighted_numeric(weights, "output_crosstalk_fraction", 0.0),
                            "strongest_neighbor_fraction": weighted_numeric(weights, "strongest_neighbor_fraction", 0.0),
                            "crosstalk_uncertainty_min": weighted_numeric(weights, "crosstalk_uncertainty_min", 0.0),
                            "crosstalk_uncertainty_max": weighted_numeric(weights, "crosstalk_uncertainty_max", 1.0),
                            "cra_x_deg": weighted_numeric(weights, "cra_x_deg", 0.0),
                            "cra_z_deg": weighted_numeric(weights, "cra_z_deg", 0.0),
                            "lens_cra_x_deg": weighted_numeric(weights, "lens_cra_x_deg", math.nan),
                            "lens_cra_z_deg": weighted_numeric(weights, "lens_cra_z_deg", math.nan),
                            "sensor_cra_x_deg": weighted_numeric(weights, "sensor_cra_x_deg", math.nan),
                            "sensor_cra_z_deg": weighted_numeric(weights, "sensor_cra_z_deg", math.nan),
                            "cra_mismatch_x_deg": weighted_numeric(weights, "cra_mismatch_x_deg", math.nan),
                            "cra_mismatch_z_deg": weighted_numeric(weights, "cra_mismatch_z_deg", math.nan),
                            "cra_mismatch_total_deg": weighted_numeric(weights, "cra_mismatch_total_deg", math.nan),
                            "cra_mismatch_tolerance_profile": joined_values(weights, "cra_mismatch_tolerance_profile"),
                            "cra_mismatch_pass_tolerance_deg": weighted_numeric(weights, "cra_mismatch_pass_tolerance_deg", math.nan),
                            "cra_mismatch_check_tolerance_deg": weighted_numeric(weights, "cra_mismatch_check_tolerance_deg", math.nan),
                            "cra_mismatch_gate": combine_gate([row.get("cra_mismatch_gate", "") for row, _weight in weights]),
                            "lens_shift_x_um": weighted_numeric(weights, "lens_shift_x_um", 0.0),
                            "lens_shift_z_um": weighted_numeric(weights, "lens_shift_z_um", 0.0),
                            "field_interpolation_method": method,
                            "source_runtime_ids": ";".join(source_ids),
                            "source_weight_count": len(weights),
                            "field_evidence_gate": combine_gate([row.get("field_evidence_gate", "") for row, _weight in weights]),
                            "crosstalk_evidence_gate": combine_gate([row.get("crosstalk_evidence_gate", "") for row, _weight in weights]),
                            "combined_evidence_gate": weighted_row["combined_evidence_gate"],
                            "research_ingest_gate": weighted_row["research_ingest_gate"],
                            "production_lut_gate": weighted_row["production_lut_gate"],
                            "confidence_class": joined_values(weights, "confidence_class"),
                            "uncertainty_policy": joined_values(weights, "uncertainty_policy"),
                            "product_lut_ready": False,
                        }
                        query_output.append(row)
                        kernel_output.extend(blend_kernels(qid, slug, wave, color or joined_values(weights, "color_channel"), weights, kernels_by_runtime))
    return query_output, kernel_output


def blend_kernels(
    query_id: str,
    slug: str,
    wavelength: float,
    color: str,
    weights: list[tuple[dict[str, str], float]],
    kernels_by_runtime: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    source_ids = [row.get("runtime_id", "") for row, _weight in weights]
    for runtime_row, weight in weights:
        runtime_id = runtime_row.get("runtime_id", "")
        for kernel in kernels_by_runtime.get(runtime_id, []):
            key = (finite_float(kernel.get("dx")), finite_float(kernel.get("dz")))
            item = grouped.setdefault(
                key,
                {
                    "runtime_query_id": query_id,
                    "slug": slug,
                    "wavelength_nm": wavelength,
                    "color_channel": color,
                    "dx": key[0],
                    "dz": key[1],
                    "response_fraction": 0.0,
                    "response_fraction_min": 0.0,
                    "response_fraction_max": 0.0,
                    "color_relation": kernel.get("color_relation", ""),
                    "evidence_gate": [],
                    "source_runtime_ids": ";".join(source_ids),
                },
            )
            item["response_fraction"] += finite_float(kernel.get("response_fraction"), 0.0) * weight
            item["response_fraction_min"] += finite_float(kernel.get("response_fraction_min"), 0.0) * weight
            item["response_fraction_max"] += finite_float(kernel.get("response_fraction_max"), 0.0) * weight
            item["evidence_gate"].append(kernel.get("evidence_gate", ""))
    total = sum(item["response_fraction"] for item in grouped.values())
    rows: list[dict[str, Any]] = []
    for item in grouped.values():
        fraction = item["response_fraction"] / total if total > 0 else 0.0
        rows.append(
            {
                "runtime_query_id": item["runtime_query_id"],
                "slug": item["slug"],
                "wavelength_nm": item["wavelength_nm"],
                "color_channel": item["color_channel"],
                "dx": item["dx"],
                "dz": item["dz"],
                "response_fraction": fraction,
                "response_fraction_min": item["response_fraction_min"],
                "response_fraction_max": item["response_fraction_max"],
                "color_relation": item["color_relation"],
                "evidence_gate": combine_gate(item["evidence_gate"]),
                "source_runtime_ids": item["source_runtime_ids"],
            }
        )
    return sorted(rows, key=lambda row: (finite_float(row.get("dz")), finite_float(row.get("dx"))))


def validate(rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]], *, mode: str, tolerance: float) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not rows:
        issues.append({"severity": "error", "code": "no_query_rows"})
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kernel_rows:
        by_query[str(row.get("runtime_query_id", ""))].append(row)
    for row in rows:
        qid = str(row.get("runtime_query_id", ""))
        nominal = finite_float(row.get("response_nominal"))
        lower = finite_float(row.get("response_min"))
        upper = finite_float(row.get("response_max"))
        if not (math.isfinite(lower) and math.isfinite(nominal) and math.isfinite(upper) and lower <= nominal <= upper):
            issues.append({"severity": "error", "code": "response_bounds_invalid", "runtime_query_id": qid})
        if boolish(row.get("product_lut_ready")):
            issues.append({"severity": "error", "code": "query_product_ready_true", "runtime_query_id": qid})
        if ";" in str(row.get("color_channel", "")):
            issues.append({"severity": "error", "code": "mixed_color_channel_query_row", "runtime_query_id": qid})
        if mode == "product" and boolish(row.get("query_allowed")):
            if gate(row.get("query_gate")) != "PASS" or gate(row.get("production_lut_gate")) != "PASS":
                issues.append({"severity": "error", "code": "product_query_allowed_without_pass", "runtime_query_id": qid})
        kernels = by_query.get(qid, [])
        if not kernels:
            issues.append({"severity": "error", "code": "missing_kernel_rows", "runtime_query_id": qid})
            continue
        total = sum(finite_float(item.get("response_fraction"), 0.0) for item in kernels)
        if abs(total - 1.0) > tolerance:
            issues.append({"severity": "error", "code": "kernel_sum_not_one", "runtime_query_id": qid, "sum": total})
    return {
        "schema": "camera_e2e_runtime_query_validation_v1",
        "pass": not issues,
        "bad_count": len(issues),
        "query_row_count": len(rows),
        "kernel_row_count": len(kernel_rows),
        "mode": mode,
        "allowed_query_count": sum(1 for row in rows if boolish(row.get("query_allowed"))),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 100) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    more = f"<p class=\"muted\">Showing {min(limit, len(rows))} of {len(rows)} rows.</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}
.muted{color:#9eb7c2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
.ok{color:#82f09d}.warn{color:#ffd36e}.bad{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}
"""
    validation = payload["validation"]
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Runtime Query</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Runtime Query</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. Product mode refuses rows unless production gates pass.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("mode", ""))}</div><div class="muted">mode</div></div>
    <div class="card"><div class="metric {'ok' if validation.get('pass') else 'bad'}">{html_cell(validation.get("pass"))}</div><div class="muted">validation pass</div></div>
    <div class="card"><div class="metric">{html_cell(validation.get("query_row_count", 0))}</div><div class="muted">query rows</div></div>
    <div class="card"><div class="metric">{html_cell(validation.get("allowed_query_count", 0))}</div><div class="muted">allowed rows</div></div>
    <div class="card"><div class="metric">{html_cell(validation.get("kernel_row_count", 0))}</div><div class="muted">kernel rows</div></div>
  </div>
  <h2>Query Rows</h2>
  {html_table(payload["rows"], QUERY_COLUMNS, limit=100)}
  <h2>Kernel Rows</h2>
  {html_table(payload["kernel_rows"], KERNEL_COLUMNS, limit=120)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_package_links(bundle: dict[str, Any], payload: dict[str, Any]) -> None:
    package_dir_text = str(bundle.get("package_dir", ""))
    if not package_dir_text:
        return
    package_json = ROOT / package_dir_text / "camera_e2e_lut_package.json"
    if not package_json.exists():
        return
    package = read_json(package_json)
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_runtime_query_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_runtime_query_csv"] = payload["outputs"]["csv"]
    outputs["camera_e2e_runtime_query_kernel_csv"] = payload["outputs"]["kernel_csv"]
    outputs["camera_e2e_runtime_query_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_runtime_query"] = {
        "schema": payload["schema"],
        "mode": payload["mode"],
        "validation_pass": payload["validation"]["pass"],
        "query_row_count": payload["validation"]["query_row_count"],
        "allowed_query_count": payload["validation"]["allowed_query_count"],
        "kernel_row_count": payload["validation"]["kernel_row_count"],
        "gate_counts": payload["gate_counts"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle, runtime_rows, kernel_rows = load_bundle(args.bundle_json)
    rows, kernels = query_rows(
        bundle,
        runtime_rows,
        kernel_rows,
        slugs=[item.strip() for item in args.slugs.split(",") if item.strip()],
        field_x_values=parse_float_list(args.field_x),
        field_z_values=parse_float_list(args.field_z),
        wavelength_nm=args.wavelength_nm,
        mode=args.mode,
    )
    validation = validate(rows, kernels, mode=args.mode, tolerance=args.tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_csv = args.output_dir / "camera_e2e_runtime_query.csv"
    kernel_csv = args.output_dir / "camera_e2e_runtime_query_kernel.csv"
    query_json = args.output_dir / "camera_e2e_runtime_query.json"
    html_path = args.output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_runtime_query_v1",
        "artifact_role": "camera_e2e_runtime_bundle_safe_lookup",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle_json": repo_rel(args.bundle_json),
        "mode": args.mode,
        "query": {
            "slugs": [item.strip() for item in args.slugs.split(",") if item.strip()] or "all",
            "field_x_norm": parse_float_list(args.field_x),
            "field_z_norm": parse_float_list(args.field_z),
            "wavelength_nm": args.wavelength_nm,
        },
        "gate_counts": dict(Counter(row["query_gate"] for row in rows)),
        "allowed_query_count": sum(1 for row in rows if boolish(row.get("query_allowed"))),
        "validation": validation,
        "rows": rows,
        "kernel_rows": kernels,
        "outputs": {
            "json": repo_rel(query_json),
            "csv": repo_rel(query_csv),
            "kernel_csv": repo_rel(kernel_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(query_csv, rows, QUERY_COLUMNS)
    write_csv(kernel_csv, kernels, KERNEL_COLUMNS)
    write_json(query_json, payload)
    write_html(html_path, payload)
    update_package_links(bundle, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", type=Path, default=DEFAULT_BUNDLE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="")
    parser.add_argument("--field-x", default="0")
    parser.add_argument("--field-z", default="0")
    parser.add_argument("--wavelength-nm", default="550")
    parser.add_argument("--mode", choices=("research", "product"), default="research")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--strict", action="store_true", help="exit nonzero if validation fails or any queried row is blocked")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "mode": payload["mode"],
                "validation_pass": payload["validation"]["pass"],
                "query_row_count": payload["validation"]["query_row_count"],
                "allowed_query_count": payload["validation"]["allowed_query_count"],
                "kernel_row_count": payload["validation"]["kernel_row_count"],
                "gate_counts": payload["gate_counts"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.strict and (not payload["validation"]["pass"] or payload["allowed_query_count"] != len(payload["rows"])):
        sys.exit(2)


if __name__ == "__main__":
    main()
