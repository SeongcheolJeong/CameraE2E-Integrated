#!/usr/bin/env python3
"""Query the flat per-sensor CameraE2E bundle.

This is the direct consumer path for CameraE2E code that wants to load one JSON
per sensor instead of following the full CSV table graph. It reads only the flat
sensor JSON files, performs the runtime optical lookup from embedded rows, joins
embedded color/material/electrical/readout/module rows, and emits scalar
CameraE2E probe values.

The query validates loadability and gate propagation. It does not certify
product accuracy.
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

from query_camera_e2e_runtime_bundle import (
    KERNEL_COLUMNS,
    boolish,
    finite_float,
    query_rows as query_runtime_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FLAT_BUNDLE_JSON = (
    ROOT
    / "runs"
    / "camera_e2e_sensor_lut_package"
    / "camera_e2e_flat_sensor_bundle"
    / "camera_e2e_flat_sensor_bundle.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "camera_e2e_flat_sensor_query"

FLAT_QUERY_COLUMNS = [
    "flat_query_id",
    "runtime_query_id",
    "mode",
    "query_allowed",
    "query_gate",
    "blockers",
    "source_flat_sensor_json",
    "camera_e2e_use_scope",
    "trust_class",
    "research_utility_grade_0_10",
    "solver_evidence_grade_0_10",
    "product_accuracy_grade_0_10",
    "crosstalk_support_status",
    "crosstalk_support_recommended_kernel",
    "crosstalk_condition_product_gate",
    "crosstalk_condition_best_truncation_fraction",
    "mesh_confidence_class",
    "field_mesh_pass_points",
    "field_mesh_required_points",
    "crosstalk_mesh_pass_points",
    "crosstalk_mesh_required_points",
    "response_lut_source_class",
    "response_calculation_method",
    "response_source_priority",
    "response_not_valid_for",
    "response_method_next_action",
    "crosstalk_lut_source_class",
    "crosstalk_calculation_method",
    "electrical_lut_source_class",
    "readout_lut_source_class",
    "module_lut_source_class",
    "response_example_cfa_times_si_simple_fraction",
    "response_example_normalization_scale_to_runtime",
    "response_example_pixel_qe_proxy",
    "response_example_direct_signal_response",
    "response_example_neighbor_leakage_response",
    "response_example_gate",
    "uncertainty_camera_e2e_use",
    "uncertainty_product_gate",
    "uncertainty_primary_blockers",
    "uncertainty_next_action",
    "material_ri_n_uncertainty_pct_min",
    "material_ri_n_uncertainty_pct_max",
    "cfa_k_transmission_uncertainty_pct_min",
    "cfa_k_transmission_uncertainty_pct_max",
    "qe_absolute_uncertainty_pct_min",
    "qe_absolute_uncertainty_pct_max",
    "cra_edge_response_uncertainty_pct_min",
    "cra_edge_response_uncertainty_pct_max",
    "optical_crosstalk_uncertainty_pct_min",
    "optical_crosstalk_uncertainty_pct_max",
    "conversion_gain_fwc_uncertainty_pct_min",
    "conversion_gain_fwc_uncertainty_pct_max",
    "temporal_noise_uncertainty_pct_min",
    "temporal_noise_uncertainty_pct_max",
    "dark_current_uncertainty_factor_min",
    "dark_current_uncertainty_factor_max",
    "dsnu_prnu_uncertainty_pct_min",
    "dsnu_prnu_uncertainty_pct_max",
    "readout_raw_uncertainty_pct_min",
    "readout_raw_uncertainty_pct_max",
    "module_coupling_uncertainty_pct_min",
    "module_coupling_uncertainty_pct_max",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_x_norm",
    "field_z_norm",
    "wavelength_nm",
    "color_channel",
    "response_nominal",
    "response_min",
    "response_max",
    "spectral_response",
    "spectral_response_normalized",
    "cfa_transmission_proxy",
    "material_row_count_at_wavelength",
    "material_research_gate",
    "material_product_gate",
    "material_keys",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "kernel_row_count",
    "kernel_sum",
    "cra_x_deg",
    "cra_z_deg",
    "cra_mismatch_gate",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "module_relative_illumination",
    "module_pupil_relative_transmission",
    "module_research_gate",
    "module_product_gate",
    "temperature_c",
    "exposure_s",
    "signal_fraction",
    "full_well_e",
    "conversion_gain_uv_per_e",
    "dark_current_e_per_s",
    "dark_signal_e",
    "dsnu_e_rms",
    "prnu_pct_rms",
    "read_noise_e_rms",
    "total_noise_e_rms",
    "electrical_collection_efficiency_prior",
    "electrical_crosstalk_fraction_prior",
    "electrical_crosstalk_gate",
    "analog_gain_x",
    "digital_gain_x",
    "adc_bit_depth",
    "black_level_dn",
    "clipping_dn",
    "dn_per_e_at_total_gain",
    "readout_direction",
    "hot_pixel_fraction",
    "defect_pixel_fraction",
    "binning_mode_id",
    "binning_group_size",
    "binning_signal_sum_gain",
    "binning_shot_noise_gain",
    "incident_photons_per_pixel",
    "signal_e",
    "direct_signal_e",
    "neighbor_leakage_e",
    "raw_dn",
    "raw_dn_clipped",
    "snr_db",
    "coverage_requirement_count",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
    "product_ready",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "wavelength_nm",
    "color_channel",
    "query_count",
    "requested_field_count",
    "field_coverage_fraction",
    "allowed_count",
    "query_gate_counts",
    "mean_signal_e",
    "min_signal_e",
    "max_signal_e",
    "center_signal_e",
    "min_edge_signal_e",
    "edge_to_center_signal_ratio",
    "mean_raw_dn_clipped",
    "min_snr_db",
    "max_snr_db",
    "max_output_crosstalk_fraction",
    "max_strongest_neighbor_fraction",
    "mesh_confidence_class",
    "camera_e2e_use_scope",
    "trust_class",
    "solver_evidence_grade_0_10",
    "product_accuracy_grade_0_10",
    "crosstalk_support_status",
    "summary_gate",
    "summary_notes",
    "product_ready",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def abs_from_repo(path: str | Path | None) -> Path:
    if not path:
        return Path("")
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text or "").replace(";", ",").split(","):
        value = finite_float(item.strip())
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("at least one finite value is required")
    return values


def gate(value: Any, default: str = "MISSING") -> str:
    text = str(value if value is not None else "").strip().upper()
    return text or default


def combine_gate(values: list[str]) -> str:
    gates = {gate(value) for value in values if str(value or "").strip()}
    if "FAIL" in gates:
        return "FAIL"
    if "MISSING" in gates:
        return "MISSING"
    if "CHECK" in gates:
        return "CHECK"
    if "PASS" in gates:
        return "PASS"
    return "MISSING"


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            grouped[value].append(row)
    return dict(grouped)


def nearest_row(
    rows: list[dict[str, Any]],
    *,
    numeric_targets: dict[str, float],
    exact: dict[str, str] | None = None,
    optional_exact: dict[str, str] | None = None,
) -> dict[str, Any]:
    exact = exact or {}
    optional_exact = optional_exact or {}
    candidates = [
        row
        for row in rows
        if all(str(row.get(key, "")).strip() == str(value).strip() for key, value in exact.items())
    ]
    for key, value in optional_exact.items():
        narrowed = [row for row in candidates if str(row.get(key, "")).strip() == str(value).strip()]
        if narrowed:
            candidates = narrowed
    if not candidates:
        return {}

    def score(row: dict[str, Any]) -> float:
        total = 0.0
        for key, target in numeric_targets.items():
            value = finite_float(row.get(key), math.nan)
            total += abs(value - target) if math.isfinite(value) else 1e9
        return total

    return min(candidates, key=score)


def gate_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def requirement_method(rows: list[dict[str, Any]], requirement_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("requirement_id", "")) == requirement_id:
            return row
    return {}


def response_example_row(rows: list[dict[str, Any]], color: str, wavelength: float) -> dict[str, Any]:
    same_color = [row for row in rows if str(row.get("color_channel", "")) == color]
    candidates = same_color if same_color else rows
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs(finite_float(row.get("wavelength_nm"), wavelength) - wavelength))


def field_case_from_norm(field_x: float, field_z: float) -> str:
    rounded = (round(field_x, 6), round(field_z, 6))
    mapping = {
        (0.0, 0.0): "center",
        (-1.0, 0.0): "x_minus_edge",
        (1.0, 0.0): "x_plus_edge",
        (0.0, -1.0): "z_minus_edge",
        (0.0, 1.0): "z_plus_edge",
        (-1.0, -1.0): "diag_minus_minus",
        (-1.0, 1.0): "diag_minus_plus",
        (1.0, -1.0): "diag_plus_minus",
        (1.0, 1.0): "diag_plus_plus",
    }
    return mapping.get(rounded, "")


def load_flat_models(bundle_path: Path, slugs: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    bundle = read_json(bundle_path)
    if bundle.get("schema") != "camera_e2e_flat_sensor_bundle_v1":
        raise ValueError(f"{bundle_path} is not camera_e2e_flat_sensor_bundle_v1")
    requested = set()
    if slugs.strip().lower() not in {"", "all", "*"}:
        requested = {slug.strip() for slug in slugs.split(",") if slug.strip()}
    models: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    for rel_path in bundle.get("sensor_model_json_files", []):
        path = abs_from_repo(rel_path)
        payload = read_json(path)
        slug = str(payload.get("sensor", {}).get("slug", ""))
        if not slug:
            continue
        if requested and slug not in requested:
            continue
        models[slug] = payload
        paths[slug] = repo_rel(path)
    missing = requested - set(models)
    if missing:
        raise ValueError(f"flat bundle is missing requested slugs: {', '.join(sorted(missing))}")
    if not models:
        raise ValueError("no flat sensor models selected")
    return bundle, models, paths


def material_rows_at_wavelength(rows: list[dict[str, Any]], slug: str, color: str, wavelength: float) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("slug") == slug
        and abs(finite_float(row.get("wavelength_nm"), wavelength) - wavelength) <= 1e-6
        and (
            row.get("color_channel") in {"", color}
            or row.get("material_family") not in {"cfa_transmission_proxy", "cfa_fdtd_material"}
        )
    ]
    if candidates:
        return candidates
    nearest = nearest_row(rows, numeric_targets={"wavelength_nm": wavelength}, exact={"slug": slug})
    nearest_wave = finite_float(nearest.get("wavelength_nm"), wavelength) if nearest else wavelength
    return [
        row
        for row in rows
        if row.get("slug") == slug
        and abs(finite_float(row.get("wavelength_nm"), nearest_wave) - nearest_wave) <= 1e-6
        and (
            row.get("color_channel") in {"", color}
            or row.get("material_family") not in {"cfa_transmission_proxy", "cfa_fdtd_material"}
        )
    ]


def raw_signal_row(
    runtime: dict[str, Any],
    electrical: dict[str, Any],
    readout: dict[str, Any],
    *,
    incident_photons: float,
) -> dict[str, float]:
    response = finite_float(runtime.get("response_nominal"), 0.0)
    direct_response = finite_float(runtime.get("direct_signal_response"), response)
    leakage_response = finite_float(runtime.get("neighbor_leakage_response"), 0.0)
    signal = max(0.0, response * incident_photons)
    direct_signal = max(0.0, direct_response * incident_photons)
    neighbor_leakage = max(0.0, leakage_response * incident_photons)
    dark_signal = max(0.0, finite_float(electrical.get("dark_signal_e"), 0.0))
    fwc = max(1.0, finite_float(electrical.get("full_well_e"), 1.0))
    total_noise = finite_float(electrical.get("total_noise_e_rms"), math.nan)
    if not math.isfinite(total_noise):
        total_noise = math.sqrt(max(signal, 0.0)) + finite_float(electrical.get("read_reset_sf_adc_noise_e_rms"), 0.0)
    clipped_signal = min(max(0.0, signal + dark_signal), fwc)
    black = finite_float(readout.get("black_level_dn"), 0.0)
    clip_dn = finite_float(readout.get("clipping_dn"), 4095.0)
    dn_per_e = finite_float(readout.get("dn_per_e_at_total_gain"), 1.0)
    raw_dn = black + clipped_signal * dn_per_e
    raw_dn_clipped = min(max(0.0, raw_dn), max(clip_dn, 0.0))
    snr = 20.0 * math.log10(signal / total_noise) if signal > 0 and total_noise > 0 else math.nan
    return {
        "signal_e": signal,
        "direct_signal_e": direct_signal,
        "neighbor_leakage_e": neighbor_leakage,
        "raw_dn": raw_dn,
        "raw_dn_clipped": raw_dn_clipped,
        "snr_db": snr,
    }


def build_flat_query(args: argparse.Namespace) -> dict[str, Any]:
    bundle_path = args.flat_bundle_json.resolve()
    bundle, models, model_paths = load_flat_models(bundle_path, args.slugs)
    runtime_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    for model in models.values():
        optical = model.get("optical_color", {})
        runtime_rows.extend(optical.get("runtime_field_response_lut", []))
        kernel_rows.extend(optical.get("optical_crosstalk_kernel_lut", []))

    runtime_bundle_stub = {
        "schema": "camera_e2e_runtime_bundle_v1",
        "product_lut_ready": False,
        "package_dir": bundle.get("package_dir", ""),
    }
    runtime_query_rows, runtime_query_kernels = query_runtime_rows(
        runtime_bundle_stub,
        runtime_rows,
        kernel_rows,
        slugs=[] if args.slugs.strip().lower() in {"", "all", "*"} else [slug.strip() for slug in args.slugs.split(",") if slug.strip()],
        field_x_values=parse_float_list(args.field_x),
        field_z_values=parse_float_list(args.field_z),
        wavelength_nm=args.wavelength_nm,
        mode=args.mode,
    )
    kernels_by_query = group_rows(runtime_query_kernels, "runtime_query_id")
    output_rows: list[dict[str, Any]] = []
    for runtime in runtime_query_rows:
        slug = str(runtime.get("slug", ""))
        model = models.get(slug, {})
        optical = model.get("optical_color", {})
        electrical_domain = model.get("pixel_electrical", {})
        readout_domain = model.get("readout_raw", {})
        module_domain = model.get("module_coupling", {})
        routing = model.get("camera_e2e_routing", {})
        color = str(runtime.get("color_channel", ""))
        wavelength = finite_float(runtime.get("wavelength_nm"), 0.0)
        field_x = finite_float(runtime.get("field_x_norm"), 0.0)
        field_z = finite_float(runtime.get("field_z_norm"), 0.0)
        field_case = field_case_from_norm(field_x, field_z)
        spectral = nearest_row(
            optical.get("spectral_response", []),
            numeric_targets={"wavelength_nm": wavelength},
            exact={"slug": slug},
            optional_exact={"color_channel": color},
        )
        material_rows = material_rows_at_wavelength(optical.get("material_nk_lut", []), slug, color, wavelength)
        electrical = nearest_row(
            electrical_domain.get("electrical_noise_lut", []),
            numeric_targets={
                "temperature_c": args.temperature_c,
                "exposure_s": args.exposure_s,
                "signal_fraction": args.signal_fraction,
            },
            exact={"slug": slug},
        )
        readout = nearest_row(
            readout_domain.get("readout_gain_lut", []),
            numeric_targets={
                "analog_gain_x": args.analog_gain,
                "digital_gain_x": args.digital_gain,
                "adc_bit_depth": args.adc_bit_depth,
            },
            exact={"slug": slug},
        )
        binning = nearest_row(readout_domain.get("binning_remosaic_lut", []), numeric_targets={}, exact={"slug": slug})
        module = nearest_row(
            module_domain.get("module_field_lut", []),
            numeric_targets={"field_x_norm": field_x, "field_z_norm": field_z, "wavelength_nm": wavelength},
            exact={"slug": slug},
        )
        support_rows = [
            row
            for row in optical.get("crosstalk_support_rows", [])
            if row.get("field_case") == field_case
            and row.get("color_channel") == color
            and abs(finite_float(row.get("wavelength_nm"), math.nan) - wavelength) <= 1e-6
        ]
        support = support_rows[0] if support_rows else {}
        coverage_rows = routing.get("coverage_matrix", [])
        use_scope = routing.get("use_scope_by_sensor", {})
        trust = routing.get("lut_trust_by_sensor", {})
        import_decision = routing.get("import_decision", {})
        mesh = routing.get("mesh_confidence", {})
        method_rows = model.get("method_provenance", {}).get("method_provenance_matrix_rows", [])
        response_method = requirement_method(method_rows, "spectral_response_qe")
        crosstalk_method = requirement_method(method_rows, "optical_crosstalk_kernel")
        electrical_method = requirement_method(method_rows, "temporal_noise")
        readout_method = requirement_method(method_rows, "analog_digital_gain")
        module_method = requirement_method(method_rows, "lens_raytrace_field_cra_map")
        example = response_example_row(model.get("response_example", {}).get("response_example_rows", []), color, wavelength)
        uncertainty = model.get("uncertainty_budget", {}).get("uncertainty_by_sensor", {})
        kernels = kernels_by_query.get(str(runtime.get("runtime_query_id", "")), [])
        kernel_sum = sum(finite_float(row.get("response_fraction"), 0.0) for row in kernels)
        scalar = raw_signal_row(runtime, electrical, readout, incident_photons=args.incident_photons)
        blocker_items = [item.strip() for item in str(runtime.get("blockers", "")).split(";") if item.strip()]
        if gate(support.get("product_crosstalk_gate", "MISSING")) in {"MISSING", "FAIL"}:
            blocker_items.append(
                f"finite-array crosstalk support {support.get('product_crosstalk_gate', 'MISSING')}: "
                f"{support.get('support_recommendation', 'support row missing for this condition')}"
            )
        if gate(module.get("product_lut_gate", "MISSING")) in {"MISSING", "FAIL"}:
            blocker_items.append(f"module product_lut_gate is {module.get('product_lut_gate', 'MISSING')}")
        output_rows.append(
            {
                "flat_query_id": f"{runtime.get('runtime_query_id')}_flat",
                "runtime_query_id": runtime.get("runtime_query_id", ""),
                "mode": args.mode,
                "query_allowed": runtime.get("query_allowed", ""),
                "query_gate": runtime.get("query_gate", ""),
                "blockers": "; ".join(blocker_items),
                "source_flat_sensor_json": model_paths.get(slug, ""),
                "camera_e2e_use_scope": use_scope.get("camera_e2e_use_scope", ""),
                "trust_class": trust.get("trust_class", ""),
                "research_utility_grade_0_10": trust.get("research_utility_grade_0_10", ""),
                "solver_evidence_grade_0_10": trust.get("solver_evidence_grade_0_10", ""),
                "product_accuracy_grade_0_10": trust.get("product_accuracy_grade_0_10", ""),
                "crosstalk_support_status": trust.get("crosstalk_support_status", import_decision.get("crosstalk_support_status", "")),
                "crosstalk_support_recommended_kernel": trust.get("crosstalk_support_recommended_kernel", import_decision.get("crosstalk_support_recommended_kernel", "")),
                "crosstalk_condition_product_gate": support.get("product_crosstalk_gate", "MISSING"),
                "crosstalk_condition_best_truncation_fraction": support.get("best_pilot_truncation_fraction", ""),
                "mesh_confidence_class": mesh.get("mesh_confidence_class", ""),
                "field_mesh_pass_points": mesh.get("field_pass_points", ""),
                "field_mesh_required_points": mesh.get("field_required_points", ""),
                "crosstalk_mesh_pass_points": mesh.get("crosstalk_pass_points", ""),
                "crosstalk_mesh_required_points": mesh.get("crosstalk_required_points", ""),
                "response_lut_source_class": response_method.get("lut_source_class", ""),
                "response_calculation_method": response_method.get("calculation_method", ""),
                "response_source_priority": response_method.get("source_priority", ""),
                "response_not_valid_for": response_method.get("not_valid_for", ""),
                "response_method_next_action": response_method.get("next_action", ""),
                "crosstalk_lut_source_class": crosstalk_method.get("lut_source_class", ""),
                "crosstalk_calculation_method": crosstalk_method.get("calculation_method", ""),
                "electrical_lut_source_class": electrical_method.get("lut_source_class", ""),
                "readout_lut_source_class": readout_method.get("lut_source_class", ""),
                "module_lut_source_class": module_method.get("lut_source_class", ""),
                "response_example_cfa_times_si_simple_fraction": example.get("cfa_times_si_simple_fraction", ""),
                "response_example_normalization_scale_to_runtime": example.get("normalization_scale_to_runtime", ""),
                "response_example_pixel_qe_proxy": example.get("pixel_qe_proxy", ""),
                "response_example_direct_signal_response": example.get("direct_signal_response", ""),
                "response_example_neighbor_leakage_response": example.get("neighbor_leakage_response", ""),
                "response_example_gate": example.get("combined_evidence_gate", ""),
                "uncertainty_camera_e2e_use": uncertainty.get("camera_e2e_use", ""),
                "uncertainty_product_gate": uncertainty.get("uncertainty_product_gate", ""),
                "uncertainty_primary_blockers": uncertainty.get("primary_blockers", ""),
                "uncertainty_next_action": uncertainty.get("recommended_next_action", ""),
                "material_ri_n_uncertainty_pct_min": uncertainty.get("material_ri_n_uncertainty_pct_min", ""),
                "material_ri_n_uncertainty_pct_max": uncertainty.get("material_ri_n_uncertainty_pct_max", ""),
                "cfa_k_transmission_uncertainty_pct_min": uncertainty.get("cfa_k_transmission_uncertainty_pct_min", ""),
                "cfa_k_transmission_uncertainty_pct_max": uncertainty.get("cfa_k_transmission_uncertainty_pct_max", ""),
                "qe_absolute_uncertainty_pct_min": uncertainty.get("qe_absolute_uncertainty_pct_min", ""),
                "qe_absolute_uncertainty_pct_max": uncertainty.get("qe_absolute_uncertainty_pct_max", ""),
                "cra_edge_response_uncertainty_pct_min": uncertainty.get("cra_edge_response_uncertainty_pct_min", ""),
                "cra_edge_response_uncertainty_pct_max": uncertainty.get("cra_edge_response_uncertainty_pct_max", ""),
                "optical_crosstalk_uncertainty_pct_min": uncertainty.get("optical_crosstalk_uncertainty_pct_min", ""),
                "optical_crosstalk_uncertainty_pct_max": uncertainty.get("optical_crosstalk_uncertainty_pct_max", ""),
                "conversion_gain_fwc_uncertainty_pct_min": uncertainty.get("conversion_gain_fwc_uncertainty_pct_min", ""),
                "conversion_gain_fwc_uncertainty_pct_max": uncertainty.get("conversion_gain_fwc_uncertainty_pct_max", ""),
                "temporal_noise_uncertainty_pct_min": uncertainty.get("temporal_noise_uncertainty_pct_min", ""),
                "temporal_noise_uncertainty_pct_max": uncertainty.get("temporal_noise_uncertainty_pct_max", ""),
                "dark_current_uncertainty_factor_min": uncertainty.get("dark_current_uncertainty_factor_min", ""),
                "dark_current_uncertainty_factor_max": uncertainty.get("dark_current_uncertainty_factor_max", ""),
                "dsnu_prnu_uncertainty_pct_min": uncertainty.get("dsnu_prnu_uncertainty_pct_min", ""),
                "dsnu_prnu_uncertainty_pct_max": uncertainty.get("dsnu_prnu_uncertainty_pct_max", ""),
                "readout_raw_uncertainty_pct_min": uncertainty.get("readout_raw_uncertainty_pct_min", ""),
                "readout_raw_uncertainty_pct_max": uncertainty.get("readout_raw_uncertainty_pct_max", ""),
                "module_coupling_uncertainty_pct_min": uncertainty.get("module_coupling_uncertainty_pct_min", ""),
                "module_coupling_uncertainty_pct_max": uncertainty.get("module_coupling_uncertainty_pct_max", ""),
                "slug": slug,
                "code": runtime.get("code", ""),
                "manufacturer": runtime.get("manufacturer", ""),
                "device_name": runtime.get("device_name", ""),
                "field_x_norm": runtime.get("field_x_norm", ""),
                "field_z_norm": runtime.get("field_z_norm", ""),
                "wavelength_nm": runtime.get("wavelength_nm", ""),
                "color_channel": color,
                "response_nominal": runtime.get("response_nominal", ""),
                "response_min": runtime.get("response_min", ""),
                "response_max": runtime.get("response_max", ""),
                "spectral_response": spectral.get("spectral_response", ""),
                "spectral_response_normalized": spectral.get("spectral_response_normalized", ""),
                "cfa_transmission_proxy": spectral.get("cfa_transmission_proxy", ""),
                "material_row_count_at_wavelength": len(material_rows),
                "material_research_gate": combine_gate([row.get("research_gate", "") for row in material_rows]),
                "material_product_gate": combine_gate([row.get("product_lut_gate", "") for row in material_rows]),
                "material_keys": ";".join(sorted({str(row.get("material_key", "")) for row in material_rows if row.get("material_key")})),
                "output_crosstalk_fraction": runtime.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": runtime.get("strongest_neighbor_fraction", ""),
                "kernel_row_count": len(kernels),
                "kernel_sum": kernel_sum,
                "cra_x_deg": runtime.get("cra_x_deg", ""),
                "cra_z_deg": runtime.get("cra_z_deg", ""),
                "cra_mismatch_gate": runtime.get("cra_mismatch_gate", ""),
                "lens_shift_x_um": runtime.get("lens_shift_x_um", ""),
                "lens_shift_z_um": runtime.get("lens_shift_z_um", ""),
                "module_relative_illumination": module.get("relative_illumination_cos4", ""),
                "module_pupil_relative_transmission": module.get("pupil_relative_transmission", ""),
                "module_research_gate": module.get("research_use_gate", ""),
                "module_product_gate": module.get("product_lut_gate", ""),
                "temperature_c": electrical.get("temperature_c", ""),
                "exposure_s": electrical.get("exposure_s", ""),
                "signal_fraction": electrical.get("signal_fraction", ""),
                "full_well_e": electrical.get("full_well_e", ""),
                "conversion_gain_uv_per_e": electrical.get("conversion_gain_uv_per_e", ""),
                "dark_current_e_per_s": electrical.get("dark_current_e_per_s", ""),
                "dark_signal_e": electrical.get("dark_signal_e", ""),
                "dsnu_e_rms": electrical.get("dsnu_e_rms", ""),
                "prnu_pct_rms": electrical.get("prnu_pct_rms", ""),
                "read_noise_e_rms": electrical.get("read_reset_sf_adc_noise_e_rms", ""),
                "total_noise_e_rms": electrical.get("total_noise_e_rms", ""),
                "electrical_collection_efficiency_prior": electrical.get("electrical_collection_efficiency_prior", ""),
                "electrical_crosstalk_fraction_prior": electrical.get("electrical_crosstalk_fraction_prior", ""),
                "electrical_crosstalk_gate": electrical.get("charge_collection_electrical_crosstalk_gate", ""),
                "analog_gain_x": readout.get("analog_gain_x", ""),
                "digital_gain_x": readout.get("digital_gain_x", ""),
                "adc_bit_depth": readout.get("adc_bit_depth", ""),
                "black_level_dn": readout.get("black_level_dn", ""),
                "clipping_dn": readout.get("clipping_dn", ""),
                "dn_per_e_at_total_gain": readout.get("dn_per_e_at_total_gain", ""),
                "readout_direction": readout.get("readout_direction", ""),
                "hot_pixel_fraction": readout.get("hot_pixel_fraction", ""),
                "defect_pixel_fraction": readout.get("defect_pixel_fraction", ""),
                "binning_mode_id": binning.get("mode_id", ""),
                "binning_group_size": binning.get("binning_group_size", ""),
                "binning_signal_sum_gain": binning.get("signal_sum_gain", ""),
                "binning_shot_noise_gain": binning.get("shot_noise_gain", ""),
                "incident_photons_per_pixel": args.incident_photons,
                **scalar,
                "coverage_requirement_count": len(coverage_rows),
                "coverage_research_gate_counts": gate_counts(coverage_rows, "research_gate"),
                "coverage_product_gate_counts": gate_counts(coverage_rows, "product_gate"),
                "product_ready": model.get("gates", {}).get("product_ready", False),
            }
        )
    expected_field_count = len(parse_float_list(args.field_x)) * len(parse_float_list(args.field_z))
    summary_rows = summarize(output_rows, expected_field_count=expected_field_count)
    validation = validate_query(output_rows, runtime_query_kernels, mode=args.mode)
    output_dir = args.output_dir.resolve()
    query_csv = output_dir / "camera_e2e_flat_sensor_query.csv"
    kernel_csv = output_dir / "camera_e2e_flat_sensor_query_kernel.csv"
    summary_csv = output_dir / "camera_e2e_flat_sensor_query_summary.csv"
    report_json = output_dir / "camera_e2e_flat_sensor_query.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_flat_sensor_query_v1",
        "artifact_role": "self_contained_flat_sensor_camera_e2e_query",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flat_bundle_json": repo_rel(bundle_path),
        "inputs": {
            "slugs": args.slugs,
            "field_x": args.field_x,
            "field_z": args.field_z,
            "wavelength_nm": args.wavelength_nm,
            "mode": args.mode,
            "temperature_c": args.temperature_c,
            "exposure_s": args.exposure_s,
            "signal_fraction": args.signal_fraction,
            "analog_gain": args.analog_gain,
            "digital_gain": args.digital_gain,
            "adc_bit_depth": args.adc_bit_depth,
            "incident_photons": args.incident_photons,
        },
        "query_row_count": len(output_rows),
        "kernel_row_count": len(runtime_query_kernels),
        "summary_row_count": len(summary_rows),
        "allowed_query_count": sum(1 for row in output_rows if boolish(row.get("query_allowed"))),
        "product_ready_count": sum(1 for row in output_rows if boolish(row.get("product_ready"))),
        "validation": validation,
        "summary_rows": summary_rows,
        "outputs": {
            "json": repo_rel(report_json),
            "query_csv": repo_rel(query_csv),
            "kernel_csv": repo_rel(kernel_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research": "The flat query is usable for research only when validation.pass is true and query_allowed is true.",
            "product": "Product use remains blocked until product_ready is true and product gates pass in the embedded sensor model.",
        },
    }
    write_csv(query_csv, output_rows, FLAT_QUERY_COLUMNS)
    write_csv(kernel_csv, runtime_query_kernels, KERNEL_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, output_rows, summary_rows)
    update_package(bundle, payload)
    return payload


def mean(values: list[float], default: float = math.nan) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else default


def summarize(rows: list[dict[str, Any]], *, expected_field_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("slug", "")), str(row.get("wavelength_nm", "")), str(row.get("color_channel", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (_slug, _wave, _color), group in sorted(groups.items()):
        first = group[0]
        signals = [finite_float(row.get("signal_e")) for row in group]
        raw_dns = [finite_float(row.get("raw_dn_clipped")) for row in group]
        snrs = [finite_float(row.get("snr_db")) for row in group]
        output_xtalk = [finite_float(row.get("output_crosstalk_fraction")) for row in group]
        strongest = [finite_float(row.get("strongest_neighbor_fraction")) for row in group]
        center_rows = [
            row
            for row in group
            if abs(finite_float(row.get("field_x_norm"), 999.0)) <= 1e-12
            and abs(finite_float(row.get("field_z_norm"), 999.0)) <= 1e-12
        ]
        center_signal = mean([finite_float(row.get("signal_e")) for row in center_rows])
        edge_signals = [
            finite_float(row.get("signal_e"))
            for row in group
            if math.hypot(finite_float(row.get("field_x_norm"), 0.0), finite_float(row.get("field_z_norm"), 0.0)) > 1e-12
        ]
        min_edge = min([value for value in edge_signals if math.isfinite(value)], default=math.nan)
        edge_ratio = min_edge / center_signal if math.isfinite(min_edge) and math.isfinite(center_signal) and center_signal > 0 else math.nan
        field_coverage = len(group) / max(expected_field_count, 1)
        notes: list[str] = []
        if len(group) < expected_field_count:
            notes.append(f"partial field coverage {len(group)}/{expected_field_count}")
        if math.isfinite(edge_ratio) and edge_ratio <= 0.05:
            notes.append("near-zero edge response; inspect proxy/solver coverage before using for CRA trend")
        summary_gate = "CHECK" if notes else "PASS"
        output.append(
            {
                "slug": first.get("slug", ""),
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "wavelength_nm": first.get("wavelength_nm", ""),
                "color_channel": first.get("color_channel", ""),
                "query_count": len(group),
                "requested_field_count": expected_field_count,
                "field_coverage_fraction": field_coverage,
                "allowed_count": sum(1 for row in group if boolish(row.get("query_allowed"))),
                "query_gate_counts": gate_counts(group, "query_gate"),
                "mean_signal_e": mean(signals),
                "min_signal_e": min([value for value in signals if math.isfinite(value)], default=math.nan),
                "max_signal_e": max([value for value in signals if math.isfinite(value)], default=math.nan),
                "center_signal_e": center_signal,
                "min_edge_signal_e": min_edge,
                "edge_to_center_signal_ratio": edge_ratio,
                "mean_raw_dn_clipped": mean(raw_dns),
                "min_snr_db": min([value for value in snrs if math.isfinite(value)], default=math.nan),
                "max_snr_db": max([value for value in snrs if math.isfinite(value)], default=math.nan),
                "max_output_crosstalk_fraction": max([value for value in output_xtalk if math.isfinite(value)], default=math.nan),
                "max_strongest_neighbor_fraction": max([value for value in strongest if math.isfinite(value)], default=math.nan),
                "mesh_confidence_class": first.get("mesh_confidence_class", ""),
                "camera_e2e_use_scope": first.get("camera_e2e_use_scope", ""),
                "trust_class": first.get("trust_class", ""),
                "solver_evidence_grade_0_10": first.get("solver_evidence_grade_0_10", ""),
                "product_accuracy_grade_0_10": first.get("product_accuracy_grade_0_10", ""),
                "crosstalk_support_status": first.get("crosstalk_support_status", ""),
                "summary_gate": summary_gate,
                "summary_notes": "; ".join(notes),
                "product_ready": any(boolish(row.get("product_ready")) for row in group),
            }
        )
    return output


def validate_query(rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not rows:
        issues.append({"severity": "error", "code": "no_flat_query_rows"})
    kernels_by_query = group_rows(kernel_rows, "runtime_query_id")
    for row in rows:
        qid = str(row.get("flat_query_id", ""))
        runtime_qid = str(row.get("runtime_query_id", ""))
        if boolish(row.get("product_ready")):
            issues.append({"severity": "error", "code": "flat_query_unexpectedly_product_ready", "flat_query_id": qid})
        if mode == "product" and boolish(row.get("query_allowed")):
            issues.append({"severity": "error", "code": "product_flat_query_unexpectedly_allowed", "flat_query_id": qid})
        for field in (
            "response_nominal",
            "spectral_response",
            "full_well_e",
            "conversion_gain_uv_per_e",
            "module_relative_illumination",
            "electrical_collection_efficiency_prior",
            "dn_per_e_at_total_gain",
            "camera_e2e_use_scope",
            "trust_class",
            "research_utility_grade_0_10",
            "solver_evidence_grade_0_10",
            "product_accuracy_grade_0_10",
            "crosstalk_support_status",
            "mesh_confidence_class",
        ):
            if str(row.get(field, "")).strip() == "":
                issues.append({"severity": "error", "code": "flat_query_joined_value_missing", "flat_query_id": qid, "field": field})
        if finite_float(row.get("material_row_count_at_wavelength"), 0.0) <= 0:
            issues.append({"severity": "error", "code": "flat_query_material_rows_missing", "flat_query_id": qid})
        if finite_float(row.get("kernel_row_count"), 0.0) <= 0:
            issues.append({"severity": "error", "code": "flat_query_kernel_rows_missing", "flat_query_id": qid})
        kernel_sum = sum(finite_float(item.get("response_fraction"), 0.0) for item in kernels_by_query.get(runtime_qid, []))
        if abs(kernel_sum - 1.0) > 1e-6:
            issues.append({"severity": "error", "code": "flat_query_kernel_sum_not_one", "flat_query_id": qid, "kernel_sum": kernel_sum})
        if finite_float(row.get("raw_dn_clipped"), 0.0) < 0:
            issues.append({"severity": "error", "code": "flat_query_negative_raw_dn", "flat_query_id": qid})
    if mode == "research" and rows and not any(boolish(row.get("query_allowed")) for row in rows):
        issues.append({"severity": "error", "code": "flat_research_query_no_allowed_rows"})
    status = "FAIL" if any(issue.get("severity") == "error" for issue in issues) else (
        "PRODUCT_FLAT_QUERY_BLOCKED_AS_EXPECTED" if mode == "product" else "RESEARCH_FLAT_QUERY_READY_PRODUCT_BLOCKED"
    )
    return {
        "schema": "camera_e2e_flat_sensor_query_validation_v1",
        "pass": not any(issue.get("severity") == "error" for issue in issues),
        "status": status,
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        return html.escape(f"{value:.6g}" if math.isfinite(value) else str(value))
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 160) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1480px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    issues = validation.get("issues", [])
    issue_html = html_table(issues, ["severity", "code", "flat_query_id", "field"]) if issues else '<p class="pass">No flat query load errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Flat Sensor Query</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Flat Sensor Query</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This query consumes only the flat per-sensor JSON rows and preserves product blockers.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">query status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("query_row_count", 0))}</div><div class="muted">query rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("allowed_query_count", 0))}</div><div class="muted">allowed rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready rows</div></div>
</div>
<h2>Issues</h2>{issue_html}
<h2>Summary</h2>{html_table(summary_rows, SUMMARY_COLUMNS, limit=180)}
<h2>Flat Query Rows</h2>{html_table(rows, FLAT_QUERY_COLUMNS, limit=180)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(bundle: dict[str, Any], payload: dict[str, Any]) -> None:
    package_dir = str(bundle.get("package_dir", ""))
    if not package_dir:
        return
    package_path = ROOT / package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_flat_sensor_query_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_flat_sensor_query_csv"] = payload["outputs"]["query_csv"]
    outputs["camera_e2e_flat_sensor_query_kernel_csv"] = payload["outputs"]["kernel_csv"]
    outputs["camera_e2e_flat_sensor_query_summary_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_flat_sensor_query_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_flat_sensor_query"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "query_row_count": payload["query_row_count"],
        "allowed_query_count": payload["allowed_query_count"],
        "product_ready_count": payload["product_ready_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-bundle-json", type=Path, default=DEFAULT_FLAT_BUNDLE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="all")
    parser.add_argument("--field-x", default="0")
    parser.add_argument("--field-z", default="0")
    parser.add_argument("--wavelength-nm", default="550")
    parser.add_argument("--mode", choices=["research", "product"], default="research")
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--exposure-s", type=float, default=0.01)
    parser.add_argument("--signal-fraction", type=float, default=0.5)
    parser.add_argument("--analog-gain", type=float, default=1.0)
    parser.add_argument("--digital-gain", type=float, default=1.0)
    parser.add_argument("--adc-bit-depth", type=float, default=12.0)
    parser.add_argument("--incident-photons", type=float, default=8000.0)
    return parser


def main() -> None:
    payload = build_flat_query(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "query_row_count": payload["query_row_count"],
                "allowed_query_count": payload["allowed_query_count"],
                "product_ready_count": payload["product_ready_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not payload["validation"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
