#!/usr/bin/env python3
"""Export a native-DEVSIM camera research LUT from one or more sweep manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from tcad_gw_coupling import (
    ROOT,
    eqe_proxy,
    response_axis_key,
    response_axis_label,
    write_camera_system_research_lut_outputs,
    write_native_devsim_response_outputs,
    write_product_lut_blocked_html,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    root_path = (ROOT / value).resolve()
    if root_path.exists() or base is None:
        return root_path
    return (base / value).resolve()


def rows_from_manifest(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"native sweep manifest rows must be a list: {path}")
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass"}:
        return True
    if normalized in {"false", "0", "no", "fail"}:
        return False
    return None


def optical_summary_evidence(
    paths: list[Path],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv_rows(path):
            row = dict(row)
            row["_source_csv"] = str(path)
            summary_rows.append(row)

    rows_by_axis_key = {response_axis_key(row): row for row in summary_rows}
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        rows_by_case.setdefault(str(row.get("case", "")), []).append(row)

    case_evidence = []
    missing_cases = []
    expected_case_names = [str(row.get("case", "")) for row in expected_rows]
    expected_case_keys = [response_axis_label(row) for row in expected_rows]
    for expected in expected_rows:
        case = str(expected.get("case", ""))
        evidence_key = response_axis_label(expected)
        row = rows_by_axis_key.get(response_axis_key(expected))
        if row is None:
            case_matches = rows_by_case.get(case, [])
            if len(case_matches) == 1:
                row = case_matches[0]
        if not row:
            missing_cases.append(evidence_key)
            continue
        case_evidence.append(
            {
                "case": case,
                "case_key": evidence_key,
                "source_csv": row.get("_source_csv", ""),
                "wavelength_nm": finite_float(row.get("wavelength_nm")),
                "cra_x_deg": finite_float(row.get("cra_x_deg")),
                "cra_z_deg": finite_float(row.get("cra_z_deg")),
                "field_x_norm": finite_float(row.get("field_x_norm")),
                "field_z_norm": finite_float(row.get("field_z_norm")),
                "total_response": finite_float(row.get("total_response")),
                "split_phase_x_proxy": finite_float(row.get("split_phase_x_proxy")),
                "grid_dx_um": finite_float(row.get("grid_dx_um")),
                "si_internal_wavelength_pixels": finite_float(row.get("si_internal_wavelength_pixels")),
                "minimum_critical_feature_pixels": finite_float(row.get("minimum_critical_feature_pixels")),
                "recommended_min_resolution_px_per_um": finite_float(
                    row.get("recommended_min_resolution_px_per_um")
                ),
                "si_wavelength_gate_pass": parse_bool(row.get("si_wavelength_gate_pass")),
                "critical_feature_gate_pass": parse_bool(row.get("critical_feature_gate_pass")),
                "grid_resolution_gate_pass": parse_bool(row.get("grid_resolution_gate_pass")),
                "grid_resolution_notes": row.get("grid_resolution_notes", ""),
            }
        )
    gate_values = [item.get("grid_resolution_gate_pass") for item in case_evidence]
    si_pixels = [
        item["si_internal_wavelength_pixels"]
        for item in case_evidence
        if math.isfinite(float(item["si_internal_wavelength_pixels"]))
    ]
    feature_pixels = [
        item["minimum_critical_feature_pixels"]
        for item in case_evidence
        if math.isfinite(float(item["minimum_critical_feature_pixels"]))
    ]
    all_grid_pass = bool(case_evidence) and not missing_cases and all(value is True for value in gate_values)
    return {
        "schema": "camera_system_research_lut_optical_generation_evidence_v1",
        "source_summary_csvs": [str(path) for path in paths],
        "case_count": len(case_evidence),
        "expected_cases": expected_case_names,
        "expected_case_keys": expected_case_keys,
        "missing_cases": missing_cases,
        "all_grid_resolution_gate_pass": all_grid_pass,
        "min_si_internal_wavelength_pixels": min(si_pixels) if si_pixels else math.nan,
        "min_critical_feature_pixels": min(feature_pixels) if feature_pixels else math.nan,
        "cases": case_evidence,
        "note": "Per-case Meep optical grid evidence imported from camera_lut_summary.csv. This proves configured per-case grid-resolution gates; full resolution/time/PML convergence is carried by optical_convergence_report when supplied.",
    }


def build_lut_rows(
    sweep_rows: list[dict[str, Any]],
    *,
    reference_case: str,
    pixel_pitch_um: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executed = [row for row in sweep_rows if row.get("status") == "executed"]
    if not executed:
        raise RuntimeError("No executed native-DEVSIM sweep rows were supplied")

    if not reference_case:
        reference_case = str(executed[0].get("case", "reference"))
    reference_rows = [row for row in executed if row.get("case") == reference_case]
    global_reference = reference_rows[0] if reference_rows else executed[0]
    references_by_wavelength: dict[float, dict[str, Any]] = {}
    for row in reference_rows:
        wavelength = finite_float(row.get("wavelength_nm"))
        if math.isfinite(wavelength):
            references_by_wavelength.setdefault(round(wavelength, 9), row)

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for row in executed:
        case = str(row.get("case", ""))
        wavelength = finite_float(row.get("wavelength_nm"))
        reference = references_by_wavelength.get(round(wavelength, 9), global_reference)
        reference_total = finite_float(reference.get("photo_total_abs_delta_a_per_cm"))
        reference_left = finite_float(reference.get("left_photo_delta_a_per_cm"))
        reference_right = finite_float(reference.get("right_photo_delta_a_per_cm"))
        reference_scope = (
            "same_wavelength"
            if math.isfinite(wavelength)
            and math.isfinite(finite_float(reference.get("wavelength_nm")))
            and round(wavelength, 9) == round(finite_float(reference.get("wavelength_nm")), 9)
            else "global"
        )
        photon_flux = finite_float(row.get("incident_photon_flux_cm2_s"), 1.0e20)
        left = finite_float(row.get("left_photo_delta_a_per_cm"))
        right = finite_float(row.get("right_photo_delta_a_per_cm"))
        total = finite_float(row.get("photo_total_abs_delta_a_per_cm"), abs(left) + abs(right))
        split_phase = finite_float(row.get("photo_split_phase_x_proxy"))
        summary = {
            "schema": "camera_system_diagnostic_response_v1",
            "artifact_role": "diagnostic_response",
            "mode": "split-pd-1x1",
            "split_mode": "dual-x",
            "color_channel": row.get("color_channel", ""),
            "wavelength_nm": wavelength,
            "case": case,
            "field_x_norm": finite_float(row.get("field_x_norm")),
            "field_z_norm": finite_float(row.get("field_z_norm")),
            "cra_x_deg": finite_float(row.get("cra_x_deg")),
            "cra_z_deg": finite_float(row.get("cra_z_deg")),
            "response_method": "native_devsim",
            "weighting_model": "native_terminal_current_delta",
            "reference_case": reference_case,
            "reference_wavelength_nm": finite_float(reference.get("wavelength_nm")),
            "reference_scope": reference_scope,
            "total_response_a_per_cm": total,
            "left_response_a_per_cm": left,
            "right_response_a_per_cm": right,
            "normalized_total_response_to_reference": total / reference_total
            if reference_total
            else math.nan,
            "max_region_response_a_per_cm": max(left, right),
            "min_region_response_a_per_cm": min(left, right),
            "split_phase_x": split_phase,
            "incident_photon_flux_cm2_s": photon_flux,
            "pixel_pitch_um": pixel_pitch_um,
            "eqe_proxy_total": eqe_proxy(total, photon_flux, pixel_pitch_um),
            "eqe_proxy_left": eqe_proxy(left, photon_flux, pixel_pitch_um),
            "eqe_proxy_right": eqe_proxy(right, photon_flux, pixel_pitch_um),
            "total_reference_scaled_rel_error": 0.0,
            "split_phase_error_to_native": 0.0,
            "source_native_sweep_summary": row.get("summary_json", ""),
            "source_native_sweep_output_dir": row.get("output_dir", ""),
            "product_lut_ready": False,
            "product_lut_block_reason": "native DEVSIM research response; measured stack/n,k and calibrated transport are required before product LUT use",
        }
        summary_rows.append(summary)
        for region_id, region_ix, response, ref_region in (
            ("pd_left", -1, left, reference_left),
            ("pd_right", 1, right, reference_right),
        ):
            long_rows.append(
                {
                    "schema": "camera_system_diagnostic_response_v1",
                    "artifact_role": "diagnostic_response",
                    "mode": "split-pd-1x1",
                    "split_mode": "dual-x",
                    "color_channel": row.get("color_channel", ""),
                    "wavelength_nm": wavelength,
                    "case": case,
                    "field_x_norm": finite_float(row.get("field_x_norm")),
                    "field_z_norm": finite_float(row.get("field_z_norm")),
                    "cra_x_deg": finite_float(row.get("cra_x_deg")),
                    "cra_z_deg": finite_float(row.get("cra_z_deg")),
                    "response_method": "native_devsim",
                    "weighting_model": "native_terminal_current_delta",
                    "reference_case": reference_case,
                    "reference_wavelength_nm": finite_float(reference.get("wavelength_nm")),
                    "reference_scope": reference_scope,
                    "region_id": region_id,
                    "region_kind": "subpd",
                    "region_ix": region_ix,
                    "region_iz": 0,
                    "response_a_per_cm": response,
                    "normalized_region_response_to_reference_same_region": response / ref_region
                    if ref_region
                    else math.nan,
                    "eqe_proxy_region": eqe_proxy(response, photon_flux, pixel_pitch_um),
                    "total_response_a_per_cm": total,
                    "split_phase_x": split_phase,
                    "source_native_sweep_summary": row.get("summary_json", ""),
                    "product_lut_ready": False,
                    "product_lut_block_reason": "native DEVSIM research response; measured stack/n,k and calibrated transport are required before product LUT use",
                }
            )
    return summary_rows, long_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, Any]] = []
    manifest_paths = [resolve_path(path) for path in args.native_sweep_manifest]
    for path in manifest_paths:
        sweep_rows.extend(rows_from_manifest(path))

    summary_rows, long_rows = build_lut_rows(
        sweep_rows,
        reference_case=args.reference_case,
        pixel_pitch_um=args.pixel_pitch_um,
    )
    native_outputs = write_native_devsim_response_outputs(args.output_dir, summary_rows, long_rows)
    research_outputs = write_camera_system_research_lut_outputs(
        args.output_dir,
        summary_rows,
        long_rows,
        optical_convergence_report=args.optical_convergence_report,
    )
    optical_evidence: dict[str, Any] = {}
    research_json = Path(research_outputs["json"])
    if args.optical_summary_csv:
        optical_paths = [resolve_path(path) for path in args.optical_summary_csv]
        optical_evidence = optical_summary_evidence(
            optical_paths,
            summary_rows,
        )
        research_data = read_json(research_json)
        research_data["optical_generation_evidence"] = optical_evidence
        if (
            optical_evidence.get("all_grid_resolution_gate_pass") is True
            and research_data.get("research_lut_status") == "READY_CONVERGENCE_NOT_PROVEN"
        ):
            research_data["research_lut_status"] = "GRID_QUALIFIED_CONVERGENCE_NOT_PROVEN"
        research_json.write_text(json.dumps(research_data, indent=2), encoding="utf-8")
    research_data_for_block = read_json(research_json)
    numerical_convergence = research_data_for_block.get("numerical_convergence", {})
    full_convergence_passed = (
        numerical_convergence.get("full_numerical_convergence_pass") is True
        or research_data_for_block.get("research_lut_status") == "READY_FULL_CONVERGENCE_PASS"
    )
    remaining_product_blockers = [
        "measured stack geometry",
        "measured optical n,k",
        "measured implant/profile sources",
        "calibrated electrical collection/transport targets",
    ]
    if not full_convergence_passed:
        remaining_product_blockers.append("full optical numerical convergence gates")
    product_json = args.output_dir / "camera_system_lut.json"
    product_html = args.output_dir / "camera_system_lut_report.html"
    product_block = {
        "schema": "camera_system_product_lut_blocked_v1",
        "artifact_role": "product_lut_block",
        "product_lut_ready": False,
        "status": "BLOCKED",
        "reason": "Product camera-system LUT export is blocked until " + ", ".join(remaining_product_blockers) + " pass.",
        "full_numerical_convergence_pass": full_convergence_passed,
        "remaining_product_blockers": remaining_product_blockers,
        "research_lut_json": research_outputs["json"],
        "native_devsim_response_json": native_outputs["json"],
    }
    product_json.write_text(json.dumps(product_block, indent=2), encoding="utf-8")
    write_product_lut_blocked_html(product_html, Path(research_outputs["json"]), [])
    manifest = {
        "schema": "native_devsim_research_lut_export_v1",
        "primary_response_method": "native_devsim",
        "response_methods": ["native_devsim"],
        "input_native_sweep_manifests": [str(path) for path in manifest_paths],
        "input_optical_summary_csvs": [str(resolve_path(path)) for path in args.optical_summary_csv],
        "case_count": len(summary_rows),
        "case_names": [row["case"] for row in summary_rows],
        "cases": [
            {
                "case": row["case"],
                "wavelength_nm": row["wavelength_nm"],
                "field_x_norm": row["field_x_norm"],
                "field_z_norm": row["field_z_norm"],
                "cra_x_deg": row["cra_x_deg"],
                "cra_z_deg": row["cra_z_deg"],
                "native_left_delta_a_per_cm": row["left_response_a_per_cm"],
                "native_right_delta_a_per_cm": row["right_response_a_per_cm"],
                "native_total_abs_delta_a_per_cm": row["total_response_a_per_cm"],
                "native_split_phase_x_proxy": row["split_phase_x"],
            }
            for row in summary_rows
        ],
        "reference_case": args.reference_case,
        "optical_generation_evidence": optical_evidence,
        "product_lut_ready": False,
        "outputs": {
            "native_devsim_response": native_outputs,
            "camera_system_research_lut": research_outputs,
            "camera_system_lut": {
                "json": str(product_json),
                "html": str(product_html),
            },
        },
    }
    manifest_path = args.output_dir / "native_devsim_research_lut_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-sweep-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-case", default="center")
    parser.add_argument("--pixel-pitch-um", type=float, default=1.4)
    parser.add_argument("--optical-convergence-report", type=Path, default=None)
    parser.add_argument(
        "--optical-summary-csv",
        type=Path,
        nargs="*",
        default=[],
        help="Optional Meep camera_lut_summary.csv files used to embed per-case optical grid evidence.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
