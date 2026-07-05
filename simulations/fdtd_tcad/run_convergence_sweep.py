#!/usr/bin/env python3
"""Run resolution/time/PML convergence sweeps for the Meep camera LUT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def parse_number_list(raw: str, cast=float) -> list:
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty numeric list: {raw}")
    return values


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def mode_shape(mode: str) -> tuple[int, int]:
    if mode == "split-pd-1x1":
        return 1, 1
    if mode == "ocl-2x2":
        return 2, 2
    if mode == "ocl-3x3":
        return 3, 3
    raise ValueError(f"Unsupported mode: {mode}")


def stack_cell_y_um(geometry: dict[str, Any], pml_um: float) -> float:
    return (
        2.0 * pml_um
        + float(geometry.get("air_top", 0.55))
        + float(geometry.get("lens_height", 0.35))
        + float(geometry.get("cfa_thickness", 0.45))
        + float(geometry.get("passivation_thickness", 0.15))
        + float(geometry.get("si_thickness", 2.0))
        + float(geometry.get("bottom_air", 0.25))
    )


def snapped_cell_y_um(cell_y_um: float, bottom_air_um: float, resolution: int, mode: str) -> float:
    if mode == "off":
        return cell_y_um
    requested_pixels = cell_y_um * resolution
    if mode == "nearest":
        target_pixels = int(round(requested_pixels))
    elif mode == "ceil":
        target_pixels = int(math.ceil(requested_pixels - 1.0e-12))
    elif mode == "floor":
        target_pixels = int(math.floor(requested_pixels + 1.0e-12))
    else:
        raise ValueError("grid snap mode must be one of off, nearest, ceil, floor")
    target_cell_y = target_pixels / resolution
    if bottom_air_um + target_cell_y - cell_y_um < 0.05 and mode == "nearest":
        target_cell_y = math.ceil(requested_pixels - 1.0e-12) / resolution
    return target_cell_y


def grid_rounding_axis(axis: str, requested_um: float, resolution: int) -> dict[str, Any]:
    requested_pixels = requested_um * resolution
    rounded_pixels = int(round(requested_pixels))
    effective_um = rounded_pixels / resolution
    error_um = effective_um - requested_um
    return {
        "axis": axis,
        "requested_um": requested_um,
        "requested_pixels": requested_pixels,
        "rounded_pixels": rounded_pixels,
        "effective_um": effective_um,
        "rounding_error_um": error_um,
        "rounding_error_nm": error_um * 1000.0,
        "integer_grid": abs(requested_pixels - rounded_pixels) < 1.0e-9,
    }


def grid_rounding_checks(
    args,
    resolutions: list[int],
    pml_values: list[float],
    sparse_settings: list[tuple[int, float, float]] | None = None,
) -> list[dict[str, Any]]:
    stack_path = args.stack_config if args.stack_config.is_absolute() else (ROOT / args.stack_config)
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    geometry = stack.get("geometry_um", {})
    pitch = float(geometry.get("pitch", 1.4))
    nx, nz = mode_shape(args.mode)
    checks: list[dict[str, Any]] = []
    resolution_pml_pairs = (
        sorted({(resolution, pml_um) for resolution, _, pml_um in sparse_settings})
        if sparse_settings
        else [(resolution, pml_um) for resolution in resolutions for pml_um in pml_values]
    )
    for resolution, pml_um in resolution_pml_pairs:
        raw_cell_y = stack_cell_y_um(geometry, pml_um)
        cell_y = snapped_cell_y_um(
            raw_cell_y,
            float(geometry.get("bottom_air", 0.25)),
            resolution,
            args.grid_snap_y,
        )
        axes = [
            grid_rounding_axis("x", nx * pitch, resolution),
            grid_rounding_axis("y", cell_y, resolution),
            grid_rounding_axis("z", nz * pitch, resolution),
        ]
        lateral_axes = [axis for axis in axes if axis["axis"] in {"x", "z"}]
        checks.append(
            {
                "resolution_px_per_um": resolution,
                "pml_um": pml_um,
                "axes": axes,
                "all_axes_integer_grid": all(bool(axis["integer_grid"]) for axis in axes),
                "lateral_period_axes_integer_grid": all(
                    bool(axis["integer_grid"]) for axis in lateral_axes
                ),
                "max_abs_rounding_error_nm": max(
                    abs(float(axis["rounding_error_nm"])) for axis in axes
                ),
            }
        )
    return checks


def run_lut(args, resolution: int, after_source_time: float, pml_um: float, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "meep_supercell_lut.py"),
        "--mode",
        args.mode,
        "--split-mode",
        args.split_mode,
        "--split-gap-um",
        str(args.split_gap_um),
        "--wavelengths-nm",
        args.wavelengths_nm,
        "--cases",
        args.cases,
        "--resolution",
        str(resolution),
        "--after-source-time",
        str(after_source_time),
        "--decay-by",
        str(args.decay_by),
        "--decay-check-time",
        str(args.decay_check_time),
        "--pml-um",
        str(pml_um),
        "--grid-snap-y",
        args.grid_snap_y,
        "--min-feature-pixels",
        str(args.min_feature_pixels),
        "--min-si-wavelength-pixels",
        str(args.min_si_wavelength_pixels),
        "--stack-config",
        str(args.stack_config),
        "--color-channel",
        args.color_channel,
        "--f-number",
        str(args.f_number),
        "--pupil-samples",
        str(args.pupil_samples),
        "--output-dir",
        str(run_dir),
    ]
    print("running", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(result.returncode)


def load_summary_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "camera_lut_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_import_run(raw: str) -> tuple[int, float, float, Path]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "--import-run entries must use resolution:after_source_time:pml_um:path"
        )
    resolution = int(parts[0])
    after_source_time = float(parts[1])
    pml_um = float(parts[2])
    path = Path(parts[3])
    if not path.is_absolute():
        path = ROOT / path
    return resolution, after_source_time, pml_um, path


def parse_sparse_setting(raw: str) -> tuple[int, float, float]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("--sparse-setting entries must use resolution:after_source_time:pml_um")
    return int(parts[0]), float(parts[1]), float(parts[2])


def load_imported_runs(import_runs: list[str]) -> tuple[list[dict], set[tuple[int, float, float]]]:
    rows: list[dict] = []
    imported_settings: set[tuple[int, float, float]] = set()
    for raw in import_runs:
        resolution, after_source_time, pml_um, path = parse_import_run(raw)
        run_dir = path.parent if path.name == "camera_lut_summary.csv" else path
        summary_rows = load_summary_rows(run_dir)
        if not summary_rows:
            raise RuntimeError(f"No camera_lut_summary.csv rows found for imported run: {path}")
        setting = (resolution, after_source_time, pml_um)
        imported_settings.add(setting)
        for row in summary_rows:
            row["resolution_px_per_um"] = resolution
            row["after_source_time"] = after_source_time
            row["pml_um"] = pml_um
            row["run_dir"] = str(run_dir)
            row["return_code"] = 0
            row["imported_existing_run"] = True
        rows.extend(summary_rows)
    return rows, imported_settings


def fieldnames_for(rows: list[dict]) -> list[str]:
    preferred = [
        "mode",
        "color_channel",
        "wavelength_nm",
        "case",
        "resolution_px_per_um",
        "after_source_time",
        "pml_um",
        "total_response",
        "reference_total_response",
        "total_response_rel_delta_to_reference",
        "split_phase_x_proxy",
        "reference_split_phase_x_proxy",
        "split_phase_x_abs_delta_to_reference",
        "split_phase_z_proxy",
        "reference_split_phase_z_proxy",
        "split_phase_z_abs_delta_to_reference",
        "signed_flux_si_absorption_fraction_diagnostic",
        "signed_flux_negative",
        "si_internal_wavelength_pixels",
        "minimum_critical_feature_pixels",
        "recommended_min_resolution_px_per_um",
        "grid_resolution_gate_pass",
        "grid_resolution_notes",
        "run_dir",
        "return_code",
    ]
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys.difference(ordered)))
    return ordered


def group_key(row: dict) -> tuple:
    return (
        row.get("mode"),
        row.get("color_channel"),
        row.get("wavelength_nm"),
        row.get("case"),
        row.get("field_x_norm"),
        row.get("field_z_norm"),
        row.get("cra_x_deg"),
        row.get("cra_z_deg"),
    )


def annotate_reference(rows: list[dict], reference_resolution: int, reference_time: float, reference_pml: float) -> None:
    references = {}
    for row in rows:
        if (
            int(row["resolution_px_per_um"]) == reference_resolution
            and float(row["after_source_time"]) == reference_time
            and float(row["pml_um"]) == reference_pml
        ):
            references[group_key(row)] = row

    for row in rows:
        ref = references.get(group_key(row))
        if not ref:
            row["reference_total_response"] = ""
            row["total_response_rel_delta_to_reference"] = ""
            continue
        total = safe_float(row.get("total_response"))
        ref_total = safe_float(ref.get("total_response"))
        row["reference_total_response"] = ref_total
        row["total_response_rel_delta_to_reference"] = (
            abs(total - ref_total) / abs(ref_total) if ref_total else 0.0
        )
        for key in ("split_phase_x_proxy", "split_phase_z_proxy"):
            ref_key = f"reference_{key}"
            delta_key = f"{key.replace('_proxy', '')}_abs_delta_to_reference"
            value = safe_float(row.get(key))
            ref_value = safe_float(ref.get(key))
            row[ref_key] = ref_value if ref.get(key, "") != "" else ""
            row[delta_key] = abs(value - ref_value) if ref.get(key, "") != "" else ""
        signed_flux = safe_float(row.get("signed_flux_si_absorption_fraction_diagnostic"))
        row["signed_flux_negative"] = bool(signed_flux < 0)


def response_group_label(key: tuple) -> str:
    parts = [str(item) for item in key if item not in ("", None)]
    return "|".join(parts)


def axis_convergence_summary(
    rows: list[dict],
    *,
    axis: str,
    axis_key: str,
    fixed_keys: tuple[str, ...],
    relative_tolerance: float,
) -> dict[str, Any]:
    expected_groups = {group_key(row) for row in rows}
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get(axis_key) in ("", None):
            continue
        fixed = tuple(row.get(key) for key in fixed_keys)
        groups.setdefault((group_key(row), fixed), []).append(row)

    comparable_groups = []
    covered_groups = set()
    total_deltas: list[float] = []
    split_x_deltas: list[float] = []
    split_z_deltas: list[float] = []
    for (response_group, fixed), group_rows in groups.items():
        axis_values = {safe_float(row.get(axis_key)) for row in group_rows}
        if len(axis_values) < 2:
            continue
        reference = max(group_rows, key=lambda row: safe_float(row.get(axis_key)))
        reference_total = safe_float(reference.get("total_response"))
        group_total_deltas = []
        group_split_x_deltas = []
        group_split_z_deltas = []
        for row in group_rows:
            total = safe_float(row.get("total_response"))
            if reference_total:
                delta = abs(total - reference_total) / abs(reference_total)
                total_deltas.append(delta)
                group_total_deltas.append(delta)
            split_x = safe_float(row.get("split_phase_x_proxy"))
            reference_split_x = safe_float(reference.get("split_phase_x_proxy"))
            if math.isfinite(split_x) and math.isfinite(reference_split_x):
                delta_x = abs(split_x - reference_split_x)
                split_x_deltas.append(delta_x)
                group_split_x_deltas.append(delta_x)
            split_z = safe_float(row.get("split_phase_z_proxy"))
            reference_split_z = safe_float(reference.get("split_phase_z_proxy"))
            if math.isfinite(split_z) and math.isfinite(reference_split_z):
                delta_z = abs(split_z - reference_split_z)
                split_z_deltas.append(delta_z)
                group_split_z_deltas.append(delta_z)
        covered_groups.add(response_group)
        comparable_groups.append(
            {
                "response_group": response_group_label(response_group),
                "fixed_settings": {key: value for key, value in zip(fixed_keys, fixed)},
                "axis_values": sorted(axis_values),
                "reference_axis_value": safe_float(reference.get(axis_key)),
                "max_total_response_rel_delta": max(group_total_deltas)
                if group_total_deltas
                else float("nan"),
                "max_split_phase_x_abs_delta": max(group_split_x_deltas)
                if group_split_x_deltas
                else float("nan"),
                "max_split_phase_z_abs_delta": max(group_split_z_deltas)
                if group_split_z_deltas
                else float("nan"),
            }
        )

    missing_groups = sorted(expected_groups.difference(covered_groups))
    max_total_delta = max(total_deltas) if total_deltas else float("nan")
    max_split_x_delta = max(split_x_deltas) if split_x_deltas else float("nan")
    max_split_z_delta = max(split_z_deltas) if split_z_deltas else float("nan")
    passed = (
        bool(comparable_groups)
        and not missing_groups
        and math.isfinite(max_total_delta)
        and max_total_delta <= relative_tolerance
    )
    return {
        "axis": axis,
        "axis_key": axis_key,
        "passed": passed,
        "relative_tolerance": relative_tolerance,
        "comparable_group_count": len(comparable_groups),
        "expected_response_group_count": len(expected_groups),
        "covered_response_group_count": len(covered_groups),
        "missing_response_groups": [response_group_label(key) for key in missing_groups],
        "max_total_response_rel_delta": max_total_delta,
        "max_split_phase_x_abs_delta": max_split_x_delta,
        "max_split_phase_z_abs_delta": max_split_z_delta,
        "reference_policy": "max axis value within each matched response/fixed-settings group",
        "groups": comparable_groups,
    }


def write_report(rows: list[dict], args, output_dir: Path, grid_checks: list[dict[str, Any]]) -> None:
    rel_deltas = [
        safe_float(row.get("total_response_rel_delta_to_reference"))
        for row in rows
        if row.get("total_response_rel_delta_to_reference") not in ("", None)
    ]
    split_x_deltas = [
        safe_float(row.get("split_phase_x_abs_delta_to_reference"))
        for row in rows
        if row.get("split_phase_x_abs_delta_to_reference") not in ("", None)
    ]
    split_z_deltas = [
        safe_float(row.get("split_phase_z_abs_delta_to_reference"))
        for row in rows
        if row.get("split_phase_z_abs_delta_to_reference") not in ("", None)
    ]
    max_rel_delta = max(rel_deltas) if rel_deltas else float("nan")
    max_split_x_delta = max(split_x_deltas) if split_x_deltas else float("nan")
    max_split_z_delta = max(split_z_deltas) if split_z_deltas else float("nan")
    negative_flux_rows = [row for row in rows if row.get("signed_flux_negative") is True]
    grid_lateral_issue_count = sum(
        not bool(check["lateral_period_axes_integer_grid"]) for check in grid_checks
    )
    grid_any_issue_count = sum(not bool(check["all_axes_integer_grid"]) for check in grid_checks)
    grid_resolution_rows = [
        row for row in rows if safe_bool(row.get("grid_resolution_gate_pass")) is not None
    ]
    grid_resolution_fail_rows = [
        row for row in grid_resolution_rows if safe_bool(row.get("grid_resolution_gate_pass")) is False
    ]
    si_internal_wavelength_pixels = [
        safe_float(row.get("si_internal_wavelength_pixels"))
        for row in rows
        if row.get("si_internal_wavelength_pixels") not in ("", None)
    ]
    minimum_critical_feature_pixels = [
        safe_float(row.get("minimum_critical_feature_pixels"))
        for row in rows
        if row.get("minimum_critical_feature_pixels") not in ("", None)
    ]
    min_si_internal_wavelength_pixels = (
        min(si_internal_wavelength_pixels) if si_internal_wavelength_pixels else float("nan")
    )
    min_critical_feature_pixels = (
        min(minimum_critical_feature_pixels) if minimum_critical_feature_pixels else float("nan")
    )
    recommended_min_resolutions = [
        safe_float(row.get("recommended_min_resolution_px_per_um"))
        for row in rows
        if row.get("recommended_min_resolution_px_per_um") not in ("", None)
    ]
    recommended_min_resolution = (
        max(recommended_min_resolutions) if recommended_min_resolutions else float("nan")
    )
    sweep_setting_count = len(
        {
            (
                row.get("resolution_px_per_um"),
                row.get("after_source_time"),
                row.get("pml_um"),
            )
            for row in rows
        }
    )
    unique_resolutions = sorted({int(float(row.get("resolution_px_per_um"))) for row in rows})
    unique_after_source_times = sorted({safe_float(row.get("after_source_time")) for row in rows})
    unique_pml_values = sorted({safe_float(row.get("pml_um")) for row in rows})
    varied_axes = []
    if len(unique_resolutions) >= 2:
        varied_axes.append("resolution")
    if len(unique_after_source_times) >= 2:
        varied_axes.append("after_source_time")
    if len(unique_pml_values) >= 2:
        varied_axes.append("pml")
    unproven_axes = [
        axis
        for axis, count in (
            ("resolution", len(unique_resolutions)),
            ("after_source_time", len(unique_after_source_times)),
            ("pml", len(unique_pml_values)),
        )
        if count < 2
    ]
    axis_convergence = {
        "resolution": axis_convergence_summary(
            rows,
            axis="resolution",
            axis_key="resolution_px_per_um",
            fixed_keys=("after_source_time", "pml_um"),
            relative_tolerance=args.relative_tolerance,
        ),
        "after_source_time": axis_convergence_summary(
            rows,
            axis="after_source_time",
            axis_key="after_source_time",
            fixed_keys=("resolution_px_per_um", "pml_um"),
            relative_tolerance=args.relative_tolerance,
        ),
        "pml": axis_convergence_summary(
            rows,
            axis="pml",
            axis_key="pml_um",
            fixed_keys=("resolution_px_per_um", "after_source_time"),
            relative_tolerance=args.relative_tolerance,
        ),
    }
    comparable_axes = [
        axis for axis, summary in axis_convergence.items() if summary["comparable_group_count"] > 0
    ]
    unproven_axes = [
        axis
        for axis, summary in axis_convergence.items()
        if summary["comparable_group_count"] == 0 or summary["missing_response_groups"]
    ]
    failed_axes = [
        axis
        for axis, summary in axis_convergence.items()
        if summary["comparable_group_count"] > 0
        and not summary["missing_response_groups"]
        and not summary["passed"]
    ]
    axis_total_deltas = [
        summary["max_total_response_rel_delta"]
        for summary in axis_convergence.values()
        if summary["comparable_group_count"] > 0
        and math.isfinite(summary["max_total_response_rel_delta"])
    ]
    axis_split_x_deltas = [
        summary["max_split_phase_x_abs_delta"]
        for summary in axis_convergence.values()
        if summary["comparable_group_count"] > 0
        and math.isfinite(summary["max_split_phase_x_abs_delta"])
    ]
    axis_split_z_deltas = [
        summary["max_split_phase_z_abs_delta"]
        for summary in axis_convergence.values()
        if summary["comparable_group_count"] > 0
        and math.isfinite(summary["max_split_phase_z_abs_delta"])
    ]
    max_axis_total_delta = max(axis_total_deltas) if axis_total_deltas else float("nan")
    max_axis_split_x_delta = max(axis_split_x_deltas) if axis_split_x_deltas else float("nan")
    max_axis_split_z_delta = max(axis_split_z_deltas) if axis_split_z_deltas else float("nan")
    passed = bool(comparable_axes) and all(axis_convergence[axis]["passed"] for axis in comparable_axes)
    passed = passed and grid_lateral_issue_count == 0 and not grid_resolution_fail_rows
    if args.fail_on_negative_signed_flux:
        passed = passed and not negative_flux_rows
    spatial_convergence_pass = passed and axis_convergence["resolution"]["passed"]
    time_convergence_pass = passed and axis_convergence["after_source_time"]["passed"]
    pml_convergence_pass = passed and axis_convergence["pml"]["passed"]
    full_numerical_convergence_pass = (
        spatial_convergence_pass and time_convergence_pass and pml_convergence_pass
    )
    report = {
        "schema": "camera_lut_convergence_report_v1",
        "passed": passed,
        "spatial_convergence_pass": spatial_convergence_pass,
        "time_convergence_pass": time_convergence_pass,
        "pml_convergence_pass": pml_convergence_pass,
        "full_numerical_convergence_pass": full_numerical_convergence_pass,
        "varied_axes": comparable_axes,
        "unproven_axes": unproven_axes,
        "failed_axes": failed_axes,
        "axis_convergence": axis_convergence,
        "unique_resolution_count": len(unique_resolutions),
        "unique_after_source_time_count": len(unique_after_source_times),
        "unique_pml_count": len(unique_pml_values),
        "unique_resolutions_px_per_um": unique_resolutions,
        "unique_after_source_times": unique_after_source_times,
        "unique_pml_um": unique_pml_values,
        "relative_tolerance": args.relative_tolerance,
        "max_total_response_rel_delta_to_reference": max_axis_total_delta,
        "max_split_phase_x_abs_delta_to_reference": max_axis_split_x_delta,
        "max_split_phase_z_abs_delta_to_reference": max_axis_split_z_delta,
        "legacy_global_reference_max_total_response_rel_delta": max_rel_delta,
        "legacy_global_reference_max_split_phase_x_abs_delta": max_split_x_delta,
        "legacy_global_reference_max_split_phase_z_abs_delta": max_split_z_delta,
        "negative_signed_flux_count": len(negative_flux_rows),
        "fail_on_negative_signed_flux": args.fail_on_negative_signed_flux,
        "grid_snap_y": args.grid_snap_y,
        "grid_lateral_period_issue_count": grid_lateral_issue_count,
        "grid_any_axis_rounding_issue_count": grid_any_issue_count,
        "grid_rounding_checks": grid_checks,
        "grid_resolution_failure_count": len(grid_resolution_fail_rows),
        "min_si_internal_wavelength_pixels": min_si_internal_wavelength_pixels,
        "min_si_wavelength_pixels_required": args.min_si_wavelength_pixels,
        "min_critical_feature_pixels": min_critical_feature_pixels,
        "min_feature_pixels_required": args.min_feature_pixels,
        "recommended_min_resolution_px_per_um": recommended_min_resolution,
        "sweep_setting_count": sweep_setting_count,
        "row_count": len(rows),
        "notes": [
            "Reference is the max resolution, max after-source time, and max PML thickness in this sweep.",
            "At least two unique resolution/time/PML settings are required for this report to pass.",
            "The passed field means the supplied sweep settings are mutually stable; use full_numerical_convergence_pass to require resolution, time, and PML axes all varied.",
            "Lateral x/z period rounding is a blocking convergence issue because it changes the simulated pixel pitch or OCL supercell period.",
            "Y-axis rounding changes the air/PML stack extent and should be minimized for quantitative runs.",
            "Optical grid resolution is a blocking issue for quantitative use when Si internal wavelength or critical feature gates fail.",
            "A negative signed flux absorption diagnostic is a warning: inspect flux-plane orientation, sign convention, and numerical settling.",
            "The primary response is based on positive Si-volume absorption, not the signed local flux diagnostic.",
            "Passing this report is a numerical convergence check, not a product-accuracy guarantee.",
        ],
    }
    (output_dir / "convergence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Convergence Report",
        "",
        f"- passed: {passed}",
        f"- spatial convergence pass: {spatial_convergence_pass}",
        f"- time convergence pass: {time_convergence_pass}",
        f"- PML convergence pass: {pml_convergence_pass}",
        f"- full numerical convergence pass: {full_numerical_convergence_pass}",
        f"- varied axes: {', '.join(comparable_axes) if comparable_axes else 'none'}",
        f"- unproven axes: {', '.join(unproven_axes) if unproven_axes else 'none'}",
        f"- relative tolerance: {args.relative_tolerance:g}",
        f"- max axis total response relative delta: {max_axis_total_delta:g}",
        f"- max axis split-x absolute delta: {max_axis_split_x_delta:g}",
        f"- max axis split-z absolute delta: {max_axis_split_z_delta:g}",
        f"- legacy global-reference max total response relative delta: {max_rel_delta:g}",
        f"- negative signed flux rows: {len(negative_flux_rows)}",
        f"- fail on negative signed flux: {args.fail_on_negative_signed_flux}",
        f"- lateral grid-period issues: {grid_lateral_issue_count}",
        f"- any-axis grid rounding issues: {grid_any_issue_count}",
        f"- grid-resolution failures: {len(grid_resolution_fail_rows)}",
        f"- min Si internal-wavelength pixels: {min_si_internal_wavelength_pixels:g} / required {args.min_si_wavelength_pixels:g}",
        f"- min critical-feature pixels: {min_critical_feature_pixels:g} / required {args.min_feature_pixels:g}",
        f"- recommended minimum resolution: {recommended_min_resolution:g} px/um",
        f"- unique sweep settings: {sweep_setting_count}",
        f"- rows: {len(rows)}",
        "",
        "This checks numerical stability across resolution/time/PML settings. Signed flux negativity is a warning unless strict mode is enabled. This does not validate material or process accuracy.",
    ]
    (output_dir / "convergence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split-pd-1x1", "ocl-2x2", "ocl-3x3"), required=True)
    parser.add_argument("--split-mode", choices=("dual-x", "dual-z", "quad"), default="quad")
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--wavelengths-nm", default="550")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--resolutions", default="16,24")
    parser.add_argument("--after-source-times", default="25,40")
    parser.add_argument("--decay-by", type=float, default=0.0)
    parser.add_argument("--decay-check-time", type=float, default=50.0)
    parser.add_argument("--pml-um", default="0.45,0.60")
    parser.add_argument(
        "--grid-snap-y",
        choices=("off", "nearest", "ceil", "floor"),
        default="off",
        help="Forward y-cell grid snapping to meep_supercell_lut and include it in convergence grid checks.",
    )
    parser.add_argument("--stack-config", type=Path, default=ROOT / "configs" / "sensor_stack_proxy_1p4um.json")
    parser.add_argument("--color-channel", choices=("red", "green", "blue"), default="green")
    parser.add_argument("--f-number", type=float, default=0.0)
    parser.add_argument("--pupil-samples", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--min-feature-pixels",
        type=float,
        default=2.0,
        help="Minimum grid pixels required across critical optical features for quantitative use.",
    )
    parser.add_argument(
        "--min-si-wavelength-pixels",
        type=float,
        default=8.0,
        help="Minimum grid pixels per Si internal wavelength for quantitative use.",
    )
    parser.add_argument("--fail-on-negative-signed-flux", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--import-run",
        action="append",
        default=[],
        help=(
            "Import an existing Meep camera_lut_summary.csv as a convergence setting. "
            "Format: resolution:after_source_time:pml_um:path_to_run_dir_or_summary. "
            "May be repeated for split case runs."
        ),
    )
    parser.add_argument(
        "--skip-imported-settings",
        action="store_true",
        help="Do not execute settings already represented by --import-run rows.",
    )
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="Only build a report from --import-run rows; skip unimported Cartesian sweep settings.",
    )
    parser.add_argument(
        "--sparse-setting",
        action="append",
        default=[],
        help=(
            "Run only selected convergence settings instead of the full Cartesian sweep. "
            "Format: resolution:after_source_time:pml_um. May be repeated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "convergence_sweep")
    args = parser.parse_args()

    sparse_settings = [parse_sparse_setting(raw) for raw in args.sparse_setting]
    if sparse_settings:
        resolutions = sorted({setting[0] for setting in sparse_settings})
        times = sorted({setting[1] for setting in sparse_settings})
        pml_values = sorted({setting[2] for setting in sparse_settings})
        sweep_settings = sparse_settings
    else:
        resolutions = parse_number_list(args.resolutions, int)
        times = parse_number_list(args.after_source_times, float)
        pml_values = parse_number_list(args.pml_um, float)
        sweep_settings = [
            (resolution, after_source_time, pml_um)
            for resolution in resolutions
            for after_source_time in times
            for pml_um in pml_values
        ]
    grid_checks = grid_rounding_checks(args, resolutions, pml_values, sparse_settings or None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, imported_settings = load_imported_runs(args.import_run)
    for resolution, after_source_time, pml_um in sweep_settings:
        setting = (resolution, after_source_time, pml_um)
        if (args.skip_imported_settings or args.imports_only) and setting in imported_settings:
            print(
                "skipping imported setting "
                f"resolution={resolution} time={after_source_time:g} pml={pml_um:g}",
                flush=True,
            )
            continue
        if args.imports_only and setting not in imported_settings:
            print(
                "skipping unimported setting "
                f"resolution={resolution} time={after_source_time:g} pml={pml_um:g}",
                flush=True,
            )
            continue
        run_name = f"r{resolution}_t{after_source_time:g}_pml{pml_um:g}".replace(".", "p")
        run_dir = args.output_dir / run_name
        if args.skip_existing and (run_dir / "camera_lut_summary.csv").exists():
            return_code = 0
        else:
            return_code = run_lut(args, resolution, after_source_time, pml_um, run_dir)
        summary_rows = load_summary_rows(run_dir)
        for row in summary_rows:
            row["resolution_px_per_um"] = resolution
            row["after_source_time"] = after_source_time
            row["pml_um"] = pml_um
            row["run_dir"] = str(run_dir)
            row["return_code"] = return_code
        all_rows.extend(summary_rows)

    if not all_rows:
        raise RuntimeError("No convergence summary rows were produced")

    annotate_reference(all_rows, max(resolutions), max(times), max(pml_values))

    csv_path = args.output_dir / "convergence_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(all_rows))
        writer.writeheader()
        writer.writerows(all_rows)
    write_report(all_rows, args, args.output_dir, grid_checks)
    print(f"wrote {csv_path}")
    print(f"wrote {args.output_dir / 'convergence_report.json'}")
    print(f"wrote {args.output_dir / 'convergence_report.md'}")


if __name__ == "__main__":
    main()
