#!/usr/bin/env python3
"""Couple 3D FDTD generation with 3D QPD terminal weighting.

This is a practical QPD response surrogate:

    response_q ~= sum_nodes G(x,y,z) * W_q(x,y,z)

where G comes from the Meep 3D generation volume and W_q comes from the
DEVSIM pure-Laplace terminal weighting solve. It is more physically tied to the
3D optical field than average terminal weighting, but it is still not a
calibrated 3D drift-diffusion collection solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


ROOT = Path(__file__).resolve().parent
QPD_CONTACT_ORDER = (
    "cathode_q00_left_bottom",
    "cathode_q10_right_bottom",
    "cathode_q01_left_top",
    "cathode_q11_right_top",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_weighting_csv(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"weighting CSV has no rows: {path}")
    fields = rows[0].keys()
    columns = [
        f"w_{contact}_devsim_laplace"
        for contact in QPD_CONTACT_ORDER
        if f"w_{contact}_devsim_laplace" in fields
    ]
    if len(columns) != 4:
        raise RuntimeError(f"weighting CSV must contain four QPD weighting columns: {path}")

    x_um = np.asarray([float(row["x_um"]) for row in rows], dtype=float)
    depth_um = np.asarray([float(row["depth_um"]) for row in rows], dtype=float)
    z_um = np.asarray([float(row["z_um"]) for row in rows], dtype=float)
    weights = {
        contact: np.asarray([float(row[f"w_{contact}_devsim_laplace"]) for row in rows], dtype=float)
        for contact in QPD_CONTACT_ORDER
    }
    qsum = (
        np.asarray([float(row["w_qsum_devsim_laplace"]) for row in rows], dtype=float)
        if "w_qsum_devsim_laplace" in fields
        else sum(weights.values())
    )
    if not all(np.all(np.isfinite(array)) for array in (x_um, depth_um, z_um, qsum, *weights.values())):
        raise RuntimeError(f"weighting CSV contains non-finite values: {path}")
    return {
        "node_count": len(rows),
        "x_um": x_um,
        "depth_um": depth_um,
        "z_um": z_um,
        "weights": weights,
        "qsum": qsum,
    }


def sorted_axis_and_values(
    x_axis: np.ndarray,
    depth_axis: np.ndarray,
    z_axis: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_order = np.argsort(x_axis)
    depth_order = np.argsort(depth_axis)
    z_order = np.argsort(z_axis)
    return (
        x_axis[x_order],
        depth_axis[depth_order],
        z_axis[z_order],
        values[np.ix_(x_order, depth_order, z_order)],
    )


def trilinear_rectilinear(
    x_axis: np.ndarray,
    depth_axis: np.ndarray,
    z_axis: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    depth_query: np.ndarray,
    z_query: np.ndarray,
    *,
    outside_mode: str,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    x_axis, depth_axis, z_axis, values = sorted_axis_and_values(x_axis, depth_axis, z_axis, values)
    if values.shape != (len(x_axis), len(depth_axis), len(z_axis)):
        raise RuntimeError(
            f"generation grid shape {values.shape} does not match axes "
            f"({len(x_axis)}, {len(depth_axis)}, {len(z_axis)})"
        )
    if outside_mode not in {"clip", "zero"}:
        raise ValueError(f"outside_mode must be clip or zero, got {outside_mode!r}")

    outside = (
        (x_query < x_axis[0])
        | (x_query > x_axis[-1])
        | (depth_query < depth_axis[0])
        | (depth_query > depth_axis[-1])
        | (z_query < z_axis[0])
        | (z_query > z_axis[-1])
    )
    x = np.clip(x_query, x_axis[0], x_axis[-1])
    y = np.clip(depth_query, depth_axis[0], depth_axis[-1])
    z = np.clip(z_query, z_axis[0], z_axis[-1])

    ix1 = np.searchsorted(x_axis, x, side="right")
    iy1 = np.searchsorted(depth_axis, y, side="right")
    iz1 = np.searchsorted(z_axis, z, side="right")
    ix1 = np.clip(ix1, 1, len(x_axis) - 1)
    iy1 = np.clip(iy1, 1, len(depth_axis) - 1)
    iz1 = np.clip(iz1, 1, len(z_axis) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    iz0 = iz1 - 1

    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = depth_axis[iy0]
    y1 = depth_axis[iy1]
    z0 = z_axis[iz0]
    z1 = z_axis[iz1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))
    tz = np.divide(z - z0, z1 - z0, out=np.zeros_like(z), where=(z1 != z0))

    c000 = values[ix0, iy0, iz0]
    c100 = values[ix1, iy0, iz0]
    c010 = values[ix0, iy1, iz0]
    c110 = values[ix1, iy1, iz0]
    c001 = values[ix0, iy0, iz1]
    c101 = values[ix1, iy0, iz1]
    c011 = values[ix0, iy1, iz1]
    c111 = values[ix1, iy1, iz1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    out = c0 * (1.0 - tz) + c1 * tz
    if outside_mode == "zero":
        out[outside] = 0.0
    return out, {
        "outside_mode": outside_mode,
        "outside_node_count": int(np.count_nonzero(outside)),
        "outside_node_fraction": float(np.count_nonzero(outside) / max(len(outside), 1)),
        "grid_range_um": {
            "x": [float(x_axis[0]), float(x_axis[-1])],
            "depth": [float(depth_axis[0]), float(depth_axis[-1])],
            "z": [float(z_axis[0]), float(z_axis[-1])],
        },
    }, outside


def generation_cases(path: Path, requested_case: str, requested_wavelength: float | None) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        cases = np.asarray(data["case"]).astype(str)
        wavelengths = np.asarray(data["wavelength_nm"], dtype=float)
        result = []
        for index, case in enumerate(cases):
            wavelength = float(wavelengths[index])
            if requested_case != "all" and case != requested_case:
                continue
            if requested_wavelength is not None and not math.isclose(wavelength, requested_wavelength, abs_tol=1.0e-9):
                continue
            result.append(
                {
                    "index": index,
                    "case": str(case),
                    "wavelength_nm": wavelength,
                    "cra_x_deg": float(np.asarray(data["cra_x_deg"], dtype=float)[index]),
                    "cra_z_deg": float(np.asarray(data["cra_z_deg"], dtype=float)[index]),
                    "field_x_norm": float(np.asarray(data["field_x_norm"], dtype=float)[index]),
                    "field_z_norm": float(np.asarray(data["field_z_norm"], dtype=float)[index]),
                    "color_channel": str(np.asarray(data["color_channel"]).astype(str)[index])
                    if "color_channel" in data and np.asarray(data["color_channel"]).shape[0] == len(cases)
                    else (str(np.asarray(data["color_channel"]).astype(str)[0]) if "color_channel" in data else ""),
                }
            )
    if not result:
        raise RuntimeError(
            f"no generation volume entries match case={requested_case!r}, wavelength={requested_wavelength}"
        )
    return result


def load_generation_entry(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x_axis = np.asarray(data["x_um"], dtype=float)
        depth_axis = np.asarray(data["depth_um_from_si_top"], dtype=float)
        z_axis = np.asarray(data["z_um"], dtype=float)
        generation = np.asarray(data["generation_cm3_s"][index], dtype=float)
    if not np.all(np.isfinite(generation)):
        raise RuntimeError(f"generation volume contains non-finite values at index {index}")
    generation = np.clip(generation, 0.0, None)
    return x_axis, depth_axis, z_axis, generation


def interpolate_weighting_to_generation_grid(
    weighting: dict[str, Any],
    x_axis: np.ndarray,
    depth_axis: np.ndarray,
    z_axis: np.ndarray,
    generation: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    x_axis, depth_axis, z_axis, generation = sorted_axis_and_values(x_axis, depth_axis, z_axis, generation)
    xx, yy, zz = np.meshgrid(x_axis, depth_axis, z_axis, indexing="ij")
    query = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    source = np.column_stack([weighting["x_um"], weighting["depth_um"], weighting["z_um"]])
    fallback_counts: dict[str, int] = {}

    def interpolate(values: np.ndarray, name: str) -> np.ndarray:
        linear = LinearNDInterpolator(source, values, fill_value=np.nan)
        interpolated = np.asarray(linear(query), dtype=float)
        missing = ~np.isfinite(interpolated)
        fallback_counts[name] = int(np.count_nonzero(missing))
        if np.any(missing):
            nearest = NearestNDInterpolator(source, values)
            interpolated[missing] = np.asarray(nearest(query[missing]), dtype=float)
        return np.clip(interpolated, 0.0, 1.0)

    weight_values = {
        contact: interpolate(weighting["weights"][contact], contact)
        for contact in QPD_CONTACT_ORDER
    }
    qsum_values = interpolate(weighting["qsum"], "qsum")
    fallback_mask = np.zeros(len(query), dtype=bool)
    for count_name, count in fallback_counts.items():
        if count:
            values = weighting["qsum"] if count_name == "qsum" else weighting["weights"].get(count_name)
            # Recompute exact mask only for reporting/gating; interpolation itself already used nearest fallback.
            linear = LinearNDInterpolator(source, values, fill_value=np.nan)
            fallback_mask |= ~np.isfinite(np.asarray(linear(query), dtype=float))
    report = {
        "integration_grid": "generation_volume",
        "interpolation": "scipy_linear_nd_with_nearest_fallback",
        "generation_grid_shape": list(generation.shape),
        "integration_point_count": int(generation.size),
        "fallback_counts": fallback_counts,
        "fallback_point_count": int(np.count_nonzero(fallback_mask)),
        "fallback_point_fraction": float(np.count_nonzero(fallback_mask) / max(len(fallback_mask), 1)),
        "grid_range_um": {
            "x": [float(x_axis[0]), float(x_axis[-1])],
            "depth": [float(depth_axis[0]), float(depth_axis[-1])],
            "z": [float(z_axis[0]), float(z_axis[-1])],
        },
    }
    return generation.ravel(), weight_values, qsum_values, {**report, "_fallback_mask": fallback_mask}


def quadrant_metrics(raw: dict[str, float], total_generation: float, qsum_response: float) -> dict[str, Any]:
    raw_total = sum(raw.values())
    normalized = {contact: (value / raw_total if raw_total > 0.0 else 0.0) for contact, value in raw.items()}
    q00 = normalized["cathode_q00_left_bottom"]
    q10 = normalized["cathode_q10_right_bottom"]
    q01 = normalized["cathode_q01_left_top"]
    q11 = normalized["cathode_q11_right_top"]
    left = q00 + q01
    right = q10 + q11
    bottom = q00 + q10
    top = q01 + q11
    phase_x = (right - left) / (right + left) if right + left else 0.0
    phase_z = (top - bottom) / (top + bottom) if top + bottom else 0.0
    min_q = min(normalized.values()) if normalized else 0.0
    max_q = max(normalized.values()) if normalized else 0.0
    return {
        "raw_quadrant_response": raw,
        "raw_total_quadrant_response": raw_total,
        "normalized_quadrant_response": normalized,
        "generation_weighted_qsum_response": qsum_response,
        "generation_weighted_qsum_fraction": qsum_response / total_generation if total_generation > 0.0 else None,
        "left_response": left,
        "right_response": right,
        "bottom_response": bottom,
        "top_response": top,
        "phase_x_gw": phase_x,
        "phase_z_gw": phase_z,
        "phase_magnitude_gw": math.sqrt(phase_x * phase_x + phase_z * phase_z),
        "quadrant_uniformity_gw": min_q / max_q if max_q > 0.0 else None,
    }


def write_svg(path: Path, case_rows: list[dict[str, Any]]) -> None:
    width = 820
    height = 140 + max(1, len(case_rows)) * 170
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="#07131f"/>',
        '<text x="28" y="34" fill="#e2e8f0" font-family="Inter, Arial" font-size="18" font-weight="700">QPD 3D G*W Response Surrogate</text>',
        '<text x="28" y="58" fill="#94a3b8" font-family="Inter, Arial" font-size="12">3D FDTD generation volume multiplied by 3D Laplace terminal weighting potentials</text>',
    ]
    colors = {
        "cathode_q00_left_bottom": "#38bdf8",
        "cathode_q10_right_bottom": "#22c55e",
        "cathode_q01_left_top": "#818cf8",
        "cathode_q11_right_top": "#f59e0b",
    }
    labels = {
        "cathode_q00_left_bottom": "Q00 LB",
        "cathode_q10_right_bottom": "Q10 RB",
        "cathode_q01_left_top": "Q01 LT",
        "cathode_q11_right_top": "Q11 RT",
    }
    y0 = 92
    for row in case_rows:
        metrics = row["metrics"]
        weights = metrics["normalized_quadrant_response"]
        max_weight = max(weights.values()) if weights else 1.0
        lines.append(
            f'<text x="28" y="{y0}" fill="#e2e8f0" font-family="Inter, Arial" font-size="14" font-weight="700">'
            f'{row["case"]} · {row["wavelength_nm"]:.0f} nm · CRA x {row["cra_x_deg"]:.3g} z {row["cra_z_deg"]:.3g}</text>'
        )
        lines.append(
            f'<text x="28" y="{y0 + 20}" fill="#94a3b8" font-family="Inter, Arial" font-size="11">'
            f'phase x {metrics["phase_x_gw"]:.5g} · phase z {metrics["phase_z_gw"]:.5g} · uniformity {metrics.get("quadrant_uniformity_gw", 0):.5g} · qsum fraction {metrics.get("generation_weighted_qsum_fraction", 0):.5g}</text>'
        )
        for index, contact in enumerate(QPD_CONTACT_ORDER):
            y = y0 + 42 + index * 27
            value = weights.get(contact, 0.0)
            bar_width = 400 * value / max(max_weight, 1.0e-30)
            lines.append(f'<text x="44" y="{y + 17}" fill="#cbd5e1" font-family="Inter, Arial" font-size="12">{labels[contact]}</text>')
            lines.append(f'<rect x="142" y="{y}" width="{bar_width:.1f}" height="20" rx="4" fill="{colors[contact]}"/>')
            lines.append(f'<text x="565" y="{y + 15}" fill="#e2e8f0" font-family="Inter, Arial" font-size="12">{value:.6g}</text>')
        y0 += 170
    lines.append(
        f'<text x="28" y="{height - 24}" fill="#94a3b8" font-family="Inter, Arial" font-size="11">'
        'Surrogate only: not calibrated 3D drift-diffusion; use measured stack/material/device calibration for product LUT.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def field_response_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        return {}
    center = next((row for row in case_rows if row.get("case") == "center"), case_rows[0])
    center_metrics = center.get("metrics", {})
    center_total = float(center_metrics.get("raw_total_quadrant_response") or 0.0)
    center_phase_x = float(center_metrics.get("phase_x_gw") or 0.0)
    center_phase_z = float(center_metrics.get("phase_z_gw") or 0.0)
    curve = []
    phase_x_slopes = []
    phase_z_slopes = []
    response_ratios = []
    for row in case_rows:
        metrics = row.get("metrics", {})
        total = float(metrics.get("raw_total_quadrant_response") or 0.0)
        phase_x = float(metrics.get("phase_x_gw") or 0.0)
        phase_z = float(metrics.get("phase_z_gw") or 0.0)
        cra_mag = math.sqrt(float(row.get("cra_x_deg") or 0.0) ** 2 + float(row.get("cra_z_deg") or 0.0) ** 2)
        total_ratio = total / center_total if center_total > 0.0 else None
        if total_ratio is not None and row is not center:
            response_ratios.append(total_ratio)
        if cra_mag > 0.0:
            phase_x_slopes.append((phase_x - center_phase_x) / cra_mag)
            phase_z_slopes.append((phase_z - center_phase_z) / cra_mag)
        curve.append(
            {
                "case": row.get("case"),
                "wavelength_nm": row.get("wavelength_nm"),
                "cra_x_deg": row.get("cra_x_deg"),
                "cra_z_deg": row.get("cra_z_deg"),
                "cra_magnitude_deg": cra_mag,
                "total_response": total,
                "total_response_to_center": total_ratio,
                "phase_x_gw": phase_x,
                "phase_z_gw": phase_z,
                "phase_x_delta_to_center": phase_x - center_phase_x,
                "phase_z_delta_to_center": phase_z - center_phase_z,
                "quadrant_uniformity_gw": metrics.get("quadrant_uniformity_gw"),
            }
        )
    return {
        "center_case": center.get("case"),
        "curve": curve,
        "edge_to_center_response_ratio_min": min(response_ratios) if response_ratios else None,
        "edge_to_center_response_ratio_max": max(response_ratios) if response_ratios else None,
        "phase_x_slope_per_deg_max_abs": max((abs(value) for value in phase_x_slopes), default=None),
        "phase_z_slope_per_deg_max_abs": max((abs(value) for value in phase_z_slopes), default=None),
        "case_count": len(case_rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    weighting_csv = args.weighting_csv.resolve()
    generation_volume = args.generation_volume_npz.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weighting = load_weighting_csv(weighting_csv)
    case_entries = generation_cases(generation_volume, args.case, args.wavelength_nm)
    case_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for entry in case_entries:
        x_axis, depth_axis, z_axis, generation = load_generation_entry(generation_volume, int(entry["index"]))
        if args.integration_grid == "weighting":
            g_values, interp_report, outside_mask = trilinear_rectilinear(
                x_axis,
                depth_axis,
                z_axis,
                generation,
                weighting["x_um"],
                weighting["depth_um"],
                weighting["z_um"],
                outside_mode=args.outside_mode,
            )
            weight_values = weighting["weights"]
            qsum_values = weighting["qsum"]
            interp_report["integration_grid"] = "weighting_mesh_nodes"
            outside_label = "outside"
        else:
            g_values, weight_values, qsum_values, interp_report = interpolate_weighting_to_generation_grid(
                weighting,
                x_axis,
                depth_axis,
                z_axis,
                generation,
            )
            outside_mask = interp_report.pop("_fallback_mask")
            outside_label = "fallback"

        total_generation = float(np.sum(g_values))
        if total_generation <= 0.0:
            raise RuntimeError(f"interpolated generation sum is non-positive for case {entry['case']}")
        outside_generation = float(np.sum(g_values[outside_mask]))
        interp_report[f"{outside_label}_generation_sum"] = outside_generation
        interp_report[f"{outside_label}_generation_fraction"] = outside_generation / total_generation if total_generation > 0.0 else None
        raw = {
            contact: float(np.sum(g_values * weight_values[contact]))
            for contact in QPD_CONTACT_ORDER
        }
        qsum_response = float(np.sum(g_values * qsum_values))
        outside_weighted_response = float(np.sum(g_values[outside_mask] * qsum_values[outside_mask]))
        interp_report[f"{outside_label}_weighted_qsum_response"] = outside_weighted_response
        interp_report[f"{outside_label}_weighted_qsum_fraction"] = (
            outside_weighted_response / qsum_response if qsum_response > 0.0 else None
        )
        metrics = quadrant_metrics(raw, total_generation, qsum_response)
        finite = np.all(np.isfinite(g_values)) and all(math.isfinite(value) for value in raw.values())
        outside_weighted_fraction = interp_report[f"{outside_label}_weighted_qsum_fraction"] or 0.0
        gate = (
            "PASS"
            if finite
            and total_generation > 0.0
            and outside_weighted_fraction <= args.max_outside_weighted_response_fraction
            else "CHECK"
        )
        row = {
            **entry,
            "node_count": weighting["node_count"],
            "integration_point_count": int(len(g_values)),
            "generation_sum_on_integration_points": total_generation,
            "generation_min_on_integration_points": float(np.min(g_values)),
            "generation_max_on_integration_points": float(np.max(g_values)),
            "generation_mean_on_integration_points": float(np.mean(g_values)),
            "interpolation": interp_report,
            "metrics": metrics,
            "full_q1q4_gw_gate": gate,
            "full_q1q4_dd_gate": "CHECK",
        }
        case_rows.append(row)
        csv_rows.append(
            {
                "case": entry["case"],
                "wavelength_nm": entry["wavelength_nm"],
                "cra_x_deg": entry["cra_x_deg"],
                "cra_z_deg": entry["cra_z_deg"],
                "phase_x_gw": metrics["phase_x_gw"],
                "phase_z_gw": metrics["phase_z_gw"],
                "phase_magnitude_gw": metrics["phase_magnitude_gw"],
                "quadrant_uniformity_gw": metrics["quadrant_uniformity_gw"],
                "raw_total_quadrant_response": metrics["raw_total_quadrant_response"],
                "generation_weighted_qsum_fraction": metrics["generation_weighted_qsum_fraction"],
                "q00_left_bottom": metrics["normalized_quadrant_response"]["cathode_q00_left_bottom"],
                "q10_right_bottom": metrics["normalized_quadrant_response"]["cathode_q10_right_bottom"],
                "q01_left_top": metrics["normalized_quadrant_response"]["cathode_q01_left_top"],
                "q11_right_top": metrics["normalized_quadrant_response"]["cathode_q11_right_top"],
                "integration_grid": interp_report["integration_grid"],
                "outside_or_fallback_fraction": interp_report.get("outside_node_fraction", interp_report.get("fallback_point_fraction")),
                "outside_or_fallback_generation_fraction": interp_report.get(
                    "outside_generation_fraction",
                    interp_report.get("fallback_generation_fraction"),
                ),
                "outside_or_fallback_weighted_qsum_fraction": interp_report.get(
                    "outside_weighted_qsum_fraction",
                    interp_report.get("fallback_weighted_qsum_fraction"),
                ),
                "full_q1q4_gw_gate": gate,
                "full_q1q4_dd_gate": "CHECK",
            }
        )

    overall_status = "PASS" if case_rows and all(row["full_q1q4_gw_gate"] == "PASS" for row in case_rows) else "CHECK"
    response_summary = field_response_summary(case_rows)
    center_total = next((item["total_response"] for item in response_summary.get("curve", []) if item.get("case") == response_summary.get("center_case")), None)
    center_phase_x = next((item["phase_x_gw"] for item in response_summary.get("curve", []) if item.get("case") == response_summary.get("center_case")), None)
    center_phase_z = next((item["phase_z_gw"] for item in response_summary.get("curve", []) if item.get("case") == response_summary.get("center_case")), None)
    for row in csv_rows:
        matching = next(
            (
                item
                for item in response_summary.get("curve", [])
                if item.get("case") == row.get("case") and item.get("wavelength_nm") == row.get("wavelength_nm")
            ),
            {},
        )
        row["total_response_to_center"] = matching.get("total_response_to_center")
        row["phase_x_delta_to_center"] = (
            row["phase_x_gw"] - center_phase_x if center_phase_x is not None else None
        )
        row["phase_z_delta_to_center"] = (
            row["phase_z_gw"] - center_phase_z if center_phase_z is not None else None
        )
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "qpd_gw_3d_response.csv"
    svg_path = output_dir / "qpd_gw_3d_response.svg"
    write_csv(csv_path, csv_rows)
    write_svg(svg_path, case_rows)
    summary = {
        "schema": "qpd_3d_gw_response_v1",
        "status": overall_status,
        "method": "generation_weighted_laplace_terminal_weighting_surrogate_3d",
        "weighting_csv": rel_or_abs(weighting_csv),
        "generation_volume_npz": rel_or_abs(generation_volume),
        "case_count": len(case_rows),
        "node_count": weighting["node_count"],
        "integration_grid": args.integration_grid,
        "outside_mode": args.outside_mode,
        "cases": case_rows,
        "field_response_summary": response_summary,
        "full_q1q4_gw_gate": overall_status,
        "full_q1q4_dd_gate": "CHECK",
        "product_accuracy_ready": False,
        "outputs": {
            "summary_json": str(summary_path),
            "csv": str(csv_path),
            "plot_svg": str(svg_path),
        },
        "limitations": [
            "This multiplies 3D FDTD generation by pure-Laplace 3D terminal weighting potentials.",
            "It is not calibrated 3D drift-diffusion and does not include implant/trap/mobility/recombination calibration.",
            "If the generation volume is a smoke/proxy run, this result is trend-only and must not be used as a product LUT.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighting-csv", type=Path, required=True)
    parser.add_argument("--generation-volume-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--wavelength-nm", type=float, default=None)
    parser.add_argument("--integration-grid", choices=("generation", "weighting"), default="generation")
    parser.add_argument("--outside-mode", choices=("clip", "zero"), default="clip")
    parser.add_argument("--max-outside-generation-fraction", type=float, default=0.25)
    parser.add_argument("--max-outside-weighted-response-fraction", type=float, default=0.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
