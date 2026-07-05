#!/usr/bin/env python3
"""Query the CameraE2E consumer bundle.

This is a consumer-facing smoke/query tool. It starts from
camera_e2e_consumer_bundle.json, loads the per-sensor manifest, uses the runtime
bundle interpolation for optical response/crosstalk, then joins the color,
material, electrical/readout, binning, module-coupling, and coverage tables.

It proves the exported package can be consumed through the bundle contract. It
does not certify product accuracy.
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
    QUERY_COLUMNS,
    boolish,
    finite_float,
    load_bundle,
    query_rows as query_runtime_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONSUMER_BUNDLE_JSON = (
    ROOT
    / "runs"
    / "camera_e2e_sensor_lut_package"
    / "camera_e2e_consumer_bundle"
    / "camera_e2e_consumer_bundle.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "camera_e2e_consumer_query"

CONSUMER_QUERY_COLUMNS = [
    "consumer_query_id",
    "runtime_query_id",
    "mode",
    "query_allowed",
    "query_gate",
    "blockers",
    "runtime_confidence_class",
    "mesh_confidence_class",
    "camera_e2e_recommended_use",
    "mesh_field_pass_points",
    "mesh_field_required_points",
    "mesh_crosstalk_pass_points",
    "mesh_crosstalk_required_points",
    "mesh_primary_limitations",
    "mesh_next_action",
    "crosstalk_support_gate",
    "crosstalk_support_best_neighborhood",
    "crosstalk_support_best_truncation_fraction",
    "crosstalk_support_summary",
    "crosstalk_support_max_required_neighborhood",
    "crosstalk_support_min_truncation_fraction",
    "crosstalk_support_max_truncation_fraction",
    "crosstalk_support_threshold",
    "crosstalk_support_recommendation",
    "crosstalk_support_next_action",
    "crosstalk_product_candidate_count",
    "crosstalk_product_candidate_lowest_available_neighborhood",
    "crosstalk_product_candidate_min_neighborhood",
    "crosstalk_product_candidate_min_feasibility",
    "crosstalk_product_candidate_recommended_priority",
    "crosstalk_product_candidate_recommended_role",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "cfa_pattern_source_kind",
    "cfa_primary_blocker",
    "cfa_next_action",
    "capability_overall_use_scope",
    "capability_spectral_qe_scope",
    "capability_color_response_scope",
    "capability_crosstalk_scope",
    "capability_cra_response_scope",
    "lut_trust_class",
    "lut_trust_allowed_use",
    "lut_trust_research_score_0_100",
    "lut_trust_evidence_score_0_100",
    "lut_trust_product_score_0_100",
    "lut_trust_field_mesh_pass_fraction",
    "lut_trust_crosstalk_mesh_pass_fraction",
    "lut_trust_next_action",
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
    "module_pupil_cra_shift_uncertainty_deg",
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
    "row_fpn_dn_rms",
    "column_fpn_dn_rms",
    "readout_direction",
    "hot_pixel_fraction",
    "defect_pixel_fraction",
    "binning_mode_id",
    "binning_group_size",
    "binning_signal_sum_gain",
    "binning_shot_noise_gain",
    "binning_remosaic_risk",
    "coverage_requirement_count",
    "coverage_research_gate_counts",
    "coverage_product_gate_counts",
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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
    values: list[float] = []
    for item in str(text or "").replace(";", ",").split(","):
        value = finite_float(item.strip())
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("at least one finite value is required")
    return values


def nearest_row(
    rows: list[dict[str, str]],
    *,
    numeric_targets: dict[str, float],
    exact: dict[str, str] | None = None,
    optional_exact: dict[str, str] | None = None,
) -> dict[str, str]:
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

    def score(row: dict[str, str]) -> float:
        total = 0.0
        for key, target in numeric_targets.items():
            value = finite_float(row.get(key), math.nan)
            if math.isfinite(value):
                total += abs(value - target)
            else:
                total += 1e9
        return total

    return min(candidates, key=score)


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def gate_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows if str(row.get(key, ""))).items()))


def requirement_method(rows: list[dict[str, str]], requirement_id: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("requirement_id", "")) == requirement_id:
            return row
    return {}


def response_example_row(rows: list[dict[str, str]], color: str, wavelength: float) -> dict[str, str]:
    same_color = [row for row in rows if str(row.get("color_channel", "")) == color]
    candidates = same_color if same_color else rows
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs(finite_float(row.get("wavelength_nm"), wavelength) - wavelength))


def combine_gate(values: list[str]) -> str:
    gates = {str(value or "").strip().upper() for value in values if str(value or "").strip()}
    if "FAIL" in gates:
        return "FAIL"
    if "MISSING" in gates:
        return "MISSING"
    if "CHECK" in gates:
        return "CHECK"
    if "PASS" in gates:
        return "PASS"
    return "MISSING"


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


def infer_package_dir(bundle: dict[str, Any], bundle_path: Path) -> Path:
    package_dir = bundle.get("package_dir", "")
    if package_dir:
        return abs_from_repo(package_dir)
    return bundle_path.resolve().parents[1]


def load_sensor_manifests(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for rel_path in bundle.get("sensor_manifest_json_files", []):
        payload = read_json(abs_from_repo(rel_path))
        slug = payload.get("sensor", {}).get("slug", "")
        if slug:
            result[slug] = payload
    return result


def query_consumer(args: argparse.Namespace) -> dict[str, Any]:
    bundle_path = args.consumer_bundle_json.resolve()
    consumer_bundle = read_json(bundle_path)
    if consumer_bundle.get("schema") != "camera_e2e_consumer_bundle_v1":
        raise ValueError(f"{bundle_path} is not camera_e2e_consumer_bundle_v1")
    package_dir = infer_package_dir(consumer_bundle, bundle_path)
    runtime_bundle_json = package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_bundle.json"
    runtime_bundle, runtime_rows, runtime_kernels = load_bundle(runtime_bundle_json)

    sensor_manifests = load_sensor_manifests(consumer_bundle)
    requested_slugs = [] if args.slugs.strip().lower() in {"", "all", "*"} else [slug.strip() for slug in args.slugs.split(",") if slug.strip()]
    if requested_slugs:
        missing = [slug for slug in requested_slugs if slug not in sensor_manifests]
        if missing:
            raise ValueError(f"unknown slugs in consumer bundle: {', '.join(missing)}")

    runtime_query, kernel_query = query_runtime_rows(
        runtime_bundle,
        runtime_rows,
        runtime_kernels,
        slugs=requested_slugs,
        field_x_values=parse_float_list(args.field_x),
        field_z_values=parse_float_list(args.field_z),
        wavelength_nm=args.wavelength_nm,
        mode=args.mode,
    )

    all_source_tables: dict[str, list[dict[str, str]]] = {}
    first_manifest = next(iter(sensor_manifests.values()), {})
    for key, rel_path in first_manifest.get("source_tables", {}).items():
        if key.endswith("_lut") or key in {
            "spectral_response",
            "color_matrix_seed",
            "coverage_matrix",
            "mesh_confidence_by_sensor",
            "crosstalk_support_by_sensor",
            "crosstalk_product_candidates",
            "cfa_provenance_by_sensor",
            "capability_profile_by_sensor",
            "probe_summary",
            "response_example",
            "response_example_summary",
            "method_provenance_matrix",
            "method_provenance_by_sensor",
            "response_trace",
            "response_trace_summary",
            "uncertainty_by_sensor",
            "uncertainty_budget",
        }:
            all_source_tables[key] = read_csv_rows(abs_from_repo(rel_path))

    spectral_rows = all_source_tables.get("spectral_response", [])
    material_rows = all_source_tables.get("material_nk_lut", [])
    electrical_rows = all_source_tables.get("electrical_noise_lut", [])
    readout_rows = all_source_tables.get("readout_gain_lut", [])
    binning_rows = all_source_tables.get("binning_remosaic_lut", [])
    module_rows = all_source_tables.get("module_coupling_lut", [])
    coverage_rows = all_source_tables.get("coverage_matrix", [])
    mesh_confidence_by_slug = {
        row.get("slug", ""): row
        for row in all_source_tables.get("mesh_confidence_by_sensor", [])
        if row.get("slug")
    }
    crosstalk_support_by_slug = group_rows(all_source_tables.get("crosstalk_support_by_sensor", []), "slug")
    crosstalk_product_candidates_by_slug = group_rows(all_source_tables.get("crosstalk_product_candidates", []), "slug")
    cfa_provenance_by_slug = {
        row.get("slug", ""): row
        for row in all_source_tables.get("cfa_provenance_by_sensor", [])
        if row.get("slug")
    }
    capability_by_slug = {
        row.get("slug", ""): row
        for row in all_source_tables.get("capability_profile_by_sensor", [])
        if row.get("slug")
    }
    method_provenance_by_slug = group_rows(all_source_tables.get("method_provenance_matrix", []), "slug")
    response_example_by_slug = group_rows(all_source_tables.get("response_example", []), "slug")
    uncertainty_by_slug = {
        row.get("slug", ""): row
        for row in all_source_tables.get("uncertainty_by_sensor", [])
        if row.get("slug")
    }

    kernels_by_query = group_rows([{k: str(v) for k, v in row.items()} for row in kernel_query], "runtime_query_id")
    coverage_by_slug = group_rows(coverage_rows, "slug")
    output_rows: list[dict[str, Any]] = []
    for runtime in runtime_query:
        slug = str(runtime.get("slug", ""))
        color = str(runtime.get("color_channel", ""))
        wavelength = finite_float(runtime.get("wavelength_nm"), 0.0)
        field_x = finite_float(runtime.get("field_x_norm"), 0.0)
        field_z = finite_float(runtime.get("field_z_norm"), 0.0)
        sensor_manifest = sensor_manifests.get(slug, {})
        sensor_support_aggregate = sensor_manifest.get("crosstalk_support", {}).get("aggregate", {})
        coverage_for_slug = coverage_by_slug.get(slug, [])
        mesh_confidence = mesh_confidence_by_slug.get(slug, sensor_manifest.get("mesh_confidence", {}))
        inferred_field_case = field_case_from_norm(field_x, field_z)
        support_candidates = crosstalk_support_by_slug.get(slug, [])
        if inferred_field_case:
            support_candidates = [row for row in support_candidates if row.get("field_case", "") == inferred_field_case]
        else:
            support_candidates = []
        if support_candidates:
            narrowed_support = [row for row in support_candidates if row.get("color_channel", "") == color]
            support_candidates = narrowed_support
        if support_candidates:
            support_candidates = [
                row for row in support_candidates if abs(finite_float(row.get("wavelength_nm"), math.nan) - wavelength) <= 1e-6
            ]
        crosstalk_support = nearest_row(
            support_candidates,
            numeric_targets={"wavelength_nm": wavelength},
            exact={},
        )
        product_candidates = [
            row
            for row in crosstalk_product_candidates_by_slug.get(slug, [])
            if row.get("color_channel", "") == color
            and row.get("field_case", "") == inferred_field_case
            and abs(finite_float(row.get("wavelength_nm"), math.nan) - wavelength) <= 1e-6
        ]
        candidate_neighborhoods = [
            int(finite_float(row.get("neighborhood"), math.nan))
            for row in product_candidates
            if math.isfinite(finite_float(row.get("neighborhood"), math.nan))
        ]
        lowest_candidate_neighborhood = min(candidate_neighborhoods) if candidate_neighborhoods else ""
        recommended_candidates = [
            row for row in product_candidates if row.get("candidate_support_role", "") == "RECOMMENDED_MINIMUM_SUPPORT"
        ]
        recommended_candidate_neighborhoods = [
            int(finite_float(row.get("neighborhood"), math.nan))
            for row in recommended_candidates
            if math.isfinite(finite_float(row.get("neighborhood"), math.nan))
        ]
        min_candidate_neighborhood = min(recommended_candidate_neighborhoods) if recommended_candidate_neighborhoods else ""
        min_candidate = next((row for row in recommended_candidates if str(row.get("neighborhood", "")) == str(min_candidate_neighborhood)), {})
        cfa_provenance = cfa_provenance_by_slug.get(slug, sensor_manifest.get("cfa_provenance", {}))
        capability = capability_by_slug.get(slug, sensor_manifest.get("capability_profile", {}))
        trust = sensor_manifest.get("lut_trust", {})
        method_rows = method_provenance_by_slug.get(slug, [])
        response_method = requirement_method(method_rows, "spectral_response_qe")
        crosstalk_method = requirement_method(method_rows, "optical_crosstalk_kernel")
        electrical_method = requirement_method(method_rows, "temporal_noise")
        readout_method = requirement_method(method_rows, "analog_digital_gain")
        module_method = requirement_method(method_rows, "lens_raytrace_field_cra_map")
        example = response_example_row(response_example_by_slug.get(slug, []), color, wavelength)
        uncertainty = uncertainty_by_slug.get(slug, sensor_manifest.get("uncertainty_budget", {}).get("uncertainty_by_sensor", {}))

        spectral = nearest_row(
            spectral_rows,
            numeric_targets={"wavelength_nm": wavelength},
            exact={"slug": slug},
            optional_exact={"color_channel": color},
        )
        material_candidates = [
            row
            for row in material_rows
            if row.get("slug") == slug
            and abs(finite_float(row.get("wavelength_nm"), wavelength) - wavelength) <= 1e-6
            and (row.get("color_channel") in {"", color} or row.get("material_family") not in {"cfa_transmission_proxy", "cfa_fdtd_material"})
        ]
        if not material_candidates:
            nearest_material_wave = nearest_row(material_rows, numeric_targets={"wavelength_nm": wavelength}, exact={"slug": slug})
            nearest_wave = finite_float(nearest_material_wave.get("wavelength_nm"), wavelength) if nearest_material_wave else wavelength
            material_candidates = [
                row
                for row in material_rows
                if row.get("slug") == slug
                and abs(finite_float(row.get("wavelength_nm"), nearest_wave) - nearest_wave) <= 1e-6
                and (row.get("color_channel") in {"", color} or row.get("material_family") not in {"cfa_transmission_proxy", "cfa_fdtd_material"})
            ]
        electrical = nearest_row(
            electrical_rows,
            numeric_targets={
                "temperature_c": args.temperature_c,
                "exposure_s": args.exposure_s,
                "signal_fraction": args.signal_fraction,
            },
            exact={"slug": slug},
        )
        readout = nearest_row(
            readout_rows,
            numeric_targets={
                "analog_gain_x": args.analog_gain,
                "digital_gain_x": args.digital_gain,
                "adc_bit_depth": args.adc_bit_depth,
            },
            exact={"slug": slug},
        )
        binning = nearest_row(binning_rows, numeric_targets={}, exact={"slug": slug})
        module = nearest_row(
            module_rows,
            numeric_targets={"field_x_norm": field_x, "field_z_norm": field_z, "wavelength_nm": wavelength},
            exact={"slug": slug},
        )
        kernel_rows = kernels_by_query.get(str(runtime.get("runtime_query_id", "")), [])
        kernel_sum = sum(finite_float(row.get("response_fraction"), 0.0) for row in kernel_rows)
        product_ready = boolish(sensor_manifest.get("gates", {}).get("product_ready"))
        blocker_items = [item.strip() for item in str(runtime.get("blockers", "")).split(";") if item.strip()]
        if str(cfa_provenance.get("cfa_assumption_gate", "")).upper() in {"MISSING", "FAIL"}:
            blocker_items.append(
                f"CFA provenance {cfa_provenance.get('cfa_assumption_gate', '')}: {cfa_provenance.get('primary_blocker', '')}"
            )
        support_gate = str(crosstalk_support.get("product_crosstalk_gate", "MISSING") if crosstalk_support else "MISSING")
        if support_gate.upper() in {"MISSING", "FAIL"}:
            support_note = crosstalk_support.get("support_recommendation", "") if crosstalk_support else "finite-array support pilot missing for this query field/color/wavelength"
            blocker_items.append(f"Crosstalk support {support_gate}: {support_note}")
        output_rows.append(
            {
                "consumer_query_id": f"{runtime.get('runtime_query_id')}_consumer",
                "runtime_query_id": runtime.get("runtime_query_id", ""),
                "mode": args.mode,
                "query_allowed": runtime.get("query_allowed", ""),
                "query_gate": runtime.get("query_gate", ""),
                "blockers": "; ".join(blocker_items),
                "runtime_confidence_class": runtime.get("confidence_class", ""),
                "mesh_confidence_class": mesh_confidence.get("mesh_confidence_class", ""),
                "camera_e2e_recommended_use": mesh_confidence.get("camera_e2e_recommended_use", ""),
                "mesh_field_pass_points": mesh_confidence.get("field_pass_points", ""),
                "mesh_field_required_points": mesh_confidence.get("field_required_points", ""),
                "mesh_crosstalk_pass_points": mesh_confidence.get("crosstalk_pass_points", ""),
                "mesh_crosstalk_required_points": mesh_confidence.get("crosstalk_required_points", ""),
                "mesh_primary_limitations": mesh_confidence.get("primary_limitations", ""),
                "mesh_next_action": mesh_confidence.get("next_action", ""),
                "crosstalk_support_gate": support_gate,
                "crosstalk_support_best_neighborhood": crosstalk_support.get("best_pilot_neighborhood", ""),
                "crosstalk_support_best_truncation_fraction": crosstalk_support.get("best_pilot_truncation_fraction", ""),
                "crosstalk_support_summary": sensor_support_aggregate.get("summary", ""),
                "crosstalk_support_max_required_neighborhood": sensor_support_aggregate.get("max_required_neighborhood", ""),
                "crosstalk_support_min_truncation_fraction": sensor_support_aggregate.get("min_truncation_fraction", ""),
                "crosstalk_support_max_truncation_fraction": sensor_support_aggregate.get("max_truncation_fraction", ""),
                "crosstalk_support_threshold": crosstalk_support.get("truncation_threshold", ""),
                "crosstalk_support_recommendation": crosstalk_support.get("support_recommendation", ""),
                "crosstalk_support_next_action": crosstalk_support.get("next_action", ""),
                "crosstalk_product_candidate_count": len(product_candidates),
                "crosstalk_product_candidate_lowest_available_neighborhood": lowest_candidate_neighborhood,
                "crosstalk_product_candidate_min_neighborhood": min_candidate_neighborhood,
                "crosstalk_product_candidate_min_feasibility": min_candidate.get("local_feasibility", ""),
                "crosstalk_product_candidate_recommended_priority": min_candidate.get("candidate_priority", ""),
                "crosstalk_product_candidate_recommended_role": min_candidate.get("candidate_support_role", ""),
                "cfa_provenance_class": cfa_provenance.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa_provenance.get("cfa_assumption_gate", ""),
                "cfa_pattern_source_kind": cfa_provenance.get("optical_cfa_pattern_source_kind", ""),
                "cfa_primary_blocker": cfa_provenance.get("primary_blocker", ""),
                "cfa_next_action": cfa_provenance.get("next_action", ""),
                "capability_overall_use_scope": capability.get("overall_use_scope", ""),
                "capability_spectral_qe_scope": capability.get("spectral_qe_scope", ""),
                "capability_color_response_scope": capability.get("color_response_scope", ""),
                "capability_crosstalk_scope": capability.get("optical_crosstalk_scope", ""),
                "capability_cra_response_scope": capability.get("cra_response_scope", ""),
                "lut_trust_class": trust.get("trust_class", ""),
                "lut_trust_allowed_use": trust.get("camera_e2e_allowed_use", ""),
                "lut_trust_research_score_0_100": trust.get("research_usability_score_0_100", ""),
                "lut_trust_evidence_score_0_100": trust.get("evidence_confidence_score_0_100", ""),
                "lut_trust_product_score_0_100": trust.get("product_calibration_score_0_100", ""),
                "lut_trust_field_mesh_pass_fraction": trust.get("field_mesh_pass_fraction", ""),
                "lut_trust_crosstalk_mesh_pass_fraction": trust.get("crosstalk_mesh_pass_fraction", ""),
                "lut_trust_next_action": trust.get("recommended_next_action", ""),
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
                "field_x_norm": field_x,
                "field_z_norm": field_z,
                "wavelength_nm": wavelength,
                "color_channel": color,
                "response_nominal": runtime.get("response_nominal", ""),
                "response_min": runtime.get("response_min", ""),
                "response_max": runtime.get("response_max", ""),
                "spectral_response": spectral.get("spectral_response", ""),
                "spectral_response_normalized": spectral.get("spectral_response_normalized", ""),
                "cfa_transmission_proxy": spectral.get("cfa_transmission_proxy", ""),
                "material_row_count_at_wavelength": len(material_candidates),
                "material_research_gate": combine_gate([row.get("research_gate", "") for row in material_candidates]),
                "material_product_gate": combine_gate([row.get("product_lut_gate", "") for row in material_candidates]),
                "material_keys": ";".join(sorted({row.get("material_key", "") for row in material_candidates if row.get("material_key", "")})),
                "output_crosstalk_fraction": runtime.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": runtime.get("strongest_neighbor_fraction", ""),
                "kernel_row_count": len(kernel_rows),
                "kernel_sum": kernel_sum,
                "cra_x_deg": runtime.get("cra_x_deg", ""),
                "cra_z_deg": runtime.get("cra_z_deg", ""),
                "cra_mismatch_gate": runtime.get("cra_mismatch_gate", ""),
                "lens_shift_x_um": runtime.get("lens_shift_x_um", ""),
                "lens_shift_z_um": runtime.get("lens_shift_z_um", ""),
                "module_relative_illumination": module.get("relative_illumination_cos4", ""),
                "module_pupil_relative_transmission": module.get("pupil_relative_transmission", ""),
                "module_pupil_cra_shift_uncertainty_deg": module.get("pupil_cra_shift_uncertainty_deg", ""),
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
                "row_fpn_dn_rms": readout.get("row_fpn_dn_rms", ""),
                "column_fpn_dn_rms": readout.get("column_fpn_dn_rms", ""),
                "readout_direction": readout.get("readout_direction", ""),
                "hot_pixel_fraction": readout.get("hot_pixel_fraction", ""),
                "defect_pixel_fraction": readout.get("defect_pixel_fraction", ""),
                "binning_mode_id": binning.get("mode_id", ""),
                "binning_group_size": binning.get("binning_group_size", ""),
                "binning_signal_sum_gain": binning.get("signal_sum_gain", ""),
                "binning_shot_noise_gain": binning.get("shot_noise_gain", ""),
                "binning_remosaic_risk": binning.get("remosaic_risk", ""),
                "coverage_requirement_count": len(coverage_for_slug),
                "coverage_research_gate_counts": json.dumps(gate_counts(coverage_for_slug, "research_gate"), sort_keys=True),
                "coverage_product_gate_counts": json.dumps(gate_counts(coverage_for_slug, "product_gate"), sort_keys=True),
                "product_ready": product_ready,
            }
        )

    validation = validate_query(output_rows, kernel_query, mode=args.mode)
    output_dir = args.output_dir.resolve()
    query_csv = output_dir / "camera_e2e_consumer_query.csv"
    kernel_csv = output_dir / "camera_e2e_consumer_query_kernel.csv"
    report_json = output_dir / "camera_e2e_consumer_query.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_consumer_query_v1",
        "artifact_role": "camera_e2e_consumer_bundle_join_query",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "consumer_bundle_json": repo_rel(bundle_path),
        "runtime_bundle_json": repo_rel(runtime_bundle_json),
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
        },
        "query_row_count": len(output_rows),
        "kernel_row_count": len(kernel_query),
        "allowed_query_count": sum(1 for row in output_rows if boolish(row.get("query_allowed"))),
        "product_ready_count": sum(1 for row in output_rows if boolish(row.get("product_ready"))),
        "validation": validation,
        "outputs": {
            "json": repo_rel(report_json),
            "query_csv": repo_rel(query_csv),
            "kernel_csv": repo_rel(kernel_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(query_csv, output_rows, CONSUMER_QUERY_COLUMNS)
    write_csv(kernel_csv, kernel_query, KERNEL_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, output_rows)
    update_package(package_dir, payload)
    return payload


def validate_query(rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not rows:
        issues.append({"severity": "error", "code": "no_consumer_query_rows"})
    for row in rows:
        qid = row.get("consumer_query_id", "")
        if boolish(row.get("product_ready")):
            issues.append({"severity": "error", "code": "consumer_query_unexpectedly_product_ready", "consumer_query_id": qid})
        if mode == "product" and boolish(row.get("query_allowed")):
            issues.append({"severity": "error", "code": "product_query_unexpectedly_allowed", "consumer_query_id": qid})
        for key in (
            "response_nominal",
            "spectral_response",
            "full_well_e",
            "conversion_gain_uv_per_e",
            "module_relative_illumination",
            "electrical_collection_efficiency_prior",
            "dn_per_e_at_total_gain",
            "mesh_confidence_class",
            "cfa_provenance_class",
            "cfa_assumption_gate",
            "capability_overall_use_scope",
            "lut_trust_class",
            "lut_trust_evidence_score_0_100",
            "lut_trust_product_score_0_100",
            "crosstalk_support_gate",
            "response_lut_source_class",
            "response_calculation_method",
            "crosstalk_lut_source_class",
            "electrical_lut_source_class",
            "readout_lut_source_class",
            "module_lut_source_class",
            "response_example_pixel_qe_proxy",
            "response_example_gate",
        ):
            if str(row.get(key, "")).strip() == "":
                issues.append({"severity": "error", "code": "joined_value_missing", "consumer_query_id": qid, "field": key})
        if finite_float(row.get("material_row_count_at_wavelength"), 0.0) <= 0:
            issues.append({"severity": "error", "code": "material_rows_missing_at_query", "consumer_query_id": qid})
        if finite_float(row.get("kernel_row_count"), 0.0) <= 0:
            issues.append({"severity": "error", "code": "kernel_rows_missing_at_query", "consumer_query_id": qid})
        if abs(finite_float(row.get("kernel_sum"), 0.0) - 1.0) > 1e-6:
            issues.append({"severity": "error", "code": "kernel_sum_not_one", "consumer_query_id": qid, "kernel_sum": row.get("kernel_sum")})
        if finite_float(row.get("coverage_requirement_count"), 0.0) <= 0:
            issues.append({"severity": "error", "code": "coverage_rows_missing_at_query", "consumer_query_id": qid})
    return {
        "schema": "camera_e2e_consumer_query_validation_v1",
        "pass": not any(issue.get("severity") == "error" for issue in issues),
        "status": "RESEARCH_CONSUMER_QUERY_READY_PRODUCT_BLOCKED"
        if mode == "research" and not any(issue.get("severity") == "error" for issue in issues)
        else ("PRODUCT_QUERY_BLOCKED_AS_EXPECTED" if mode == "product" and not any(issue.get("severity") == "error" for issue in issues) else "FAIL"),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return html.escape(f"{value:.6g}" if math.isfinite(value) else str(value))
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1480px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    issues = validation.get("issues", [])
    issue_html = html_table(issues, ["severity", "code", "consumer_query_id", "field"]) if issues else '<p class="pass">No consumer query join errors.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Consumer Query</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Consumer Query</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This joins optical, color, material, electrical, readout, module, and coverage rows through the consumer bundle contract.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">query status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("query_row_count", 0))}</div><div class="muted">query rows</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("allowed_query_count", 0))}</div><div class="muted">allowed rows</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready rows</div></div>
</div>
<h2>Issues</h2>{issue_html}
<h2>Joined Rows</h2>{html_table(rows, CONSUMER_QUERY_COLUMNS, limit=160)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_consumer_query_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_consumer_query_csv"] = payload["outputs"]["query_csv"]
    outputs["camera_e2e_consumer_query_kernel_csv"] = payload["outputs"]["kernel_csv"]
    outputs["camera_e2e_consumer_query_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_consumer_query"] = {
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
    parser.add_argument("--consumer-bundle-json", type=Path, default=DEFAULT_CONSUMER_BUNDLE_JSON)
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
    parser.add_argument("--adc-bit-depth", type=float, default=10.0)
    return parser


def main() -> None:
    payload = query_consumer(build_parser().parse_args())
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
