#!/usr/bin/env python3
"""Export a per-sensor CameraE2E requirement coverage matrix.

The package already has runtime LUTs, material tables, color response,
electrical/readout priors, module-coupling priors, and probe results. This
exporter ties those artifacts back to the actual CameraE2E requirements so a
consumer can see which data is loadable for research use and which items remain
blocked for product accuracy.

It is intentionally strict about wording: proxy, prior, assumed, or unmeasured
values are never promoted to product-ready status.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_coverage_matrix"

COVERAGE_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "requirement_id",
    "requirement",
    "camera_e2e_use",
    "research_status",
    "research_gate",
    "product_gate",
    "row_count",
    "source_artifacts",
    "evidence_summary",
    "primary_blocker",
    "notes",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "requirement_count",
    "research_pass_count",
    "research_check_count",
    "research_missing_count",
    "research_fail_count",
    "research_na_count",
    "product_pass_count",
    "product_check_count",
    "product_missing_count",
    "product_fail_count",
    "product_na_count",
    "research_gate_counts",
    "product_gate_counts",
    "product_ready",
    "primary_blockers",
]

REQUIREMENTS = [
    {
        "domain": "Optical / Color",
        "id": "optical_material_nk_ri",
        "label": "CFA/OCL/passivation/Si n,k or RI",
        "use": "FDTD material input tables for stack and color-filter propagation.",
    },
    {
        "domain": "Optical / Color",
        "id": "spectral_response_qe",
        "label": "Spectral response / QE",
        "use": "wavelength x color x field x CRA photon-to-electron response input.",
    },
    {
        "domain": "Optical / Color",
        "id": "color_response_matrix",
        "label": "Color response matrix",
        "use": "RGB spectral sensitivity and RGB-to-XYZ/CCM seed for ISP tests.",
    },
    {
        "domain": "Optical / Color",
        "id": "optical_crosstalk_kernel",
        "label": "Optical crosstalk kernel",
        "use": "Target-to-neighbor leakage kernel for raw-domain and binning tests.",
    },
    {
        "domain": "Optical / Color",
        "id": "angular_cra_response",
        "label": "Angular response / CRA response",
        "use": "Field-dependent roll-off, lens shading, and color shading input.",
    },
    {
        "domain": "Optical / Color",
        "id": "microlens_ocl_shift_map",
        "label": "Microlens/OCL shift map",
        "use": "Field-dependent OCL shift prior for CRA compensation analysis.",
    },
    {
        "domain": "Pixel / Electrical",
        "id": "conversion_gain_fwc_saturation_nonlinearity",
        "label": "Conversion gain, FWC, saturation, nonlinearity",
        "use": "Signal capacity and DN conversion model for exposure simulation.",
    },
    {
        "domain": "Pixel / Electrical",
        "id": "dark_current_temperature_exposure",
        "label": "Dark current vs temperature/exposure",
        "use": "Temperature and exposure dependent dark signal seed.",
    },
    {
        "domain": "Pixel / Electrical",
        "id": "dsnu_prnu",
        "label": "DSNU and PRNU",
        "use": "Dark offset and pixel gain variation seed.",
    },
    {
        "domain": "Pixel / Electrical",
        "id": "temporal_noise",
        "label": "Temporal noise",
        "use": "Shot, dark-shot, read/reset/source-follower/ADC noise seed.",
    },
    {
        "domain": "Pixel / Electrical",
        "id": "charge_collection_electrical_crosstalk",
        "label": "Charge collection / diffusion / electrical crosstalk",
        "use": "Electrical collection and diffusion risk marker separate from optical crosstalk.",
    },
    {
        "domain": "Readout / RAW",
        "id": "analog_digital_gain",
        "label": "Analog/digital gain table",
        "use": "Gain-dependent DN conversion and saturation input.",
    },
    {
        "domain": "Readout / RAW",
        "id": "black_level_optical_black",
        "label": "Black level and optical black behavior",
        "use": "Raw offset and optical-black correction seed.",
    },
    {
        "domain": "Readout / RAW",
        "id": "adc_clipping_quantization",
        "label": "ADC bit depth, clipping, quantization",
        "use": "ADC clipping and quantization model for raw pipeline tests.",
    },
    {
        "domain": "Readout / RAW",
        "id": "row_column_fpn_timing_direction",
        "label": "Row/column FPN, rolling shutter timing, readout direction",
        "use": "Readout artifact and timing seed for CameraE2E temporal tests.",
    },
    {
        "domain": "Readout / RAW",
        "id": "defect_hot_pixel_stats",
        "label": "Defect pixel / hot pixel statistics",
        "use": "Defect and hot-pixel injection seed.",
    },
    {
        "domain": "Readout / RAW",
        "id": "binning_remosaic_modes",
        "label": "Binning/remosaic mode gain, noise, crosstalk",
        "use": "Mode-specific gain/noise/crosstalk reinterpretation for binning and remosaic.",
    },
    {
        "domain": "Module Coupling",
        "id": "lens_raytrace_field_cra_map",
        "label": "Lens raytrace based field CRA map",
        "use": "Field CRA and relative illumination map for module-level shading.",
    },
    {
        "domain": "Module Coupling",
        "id": "sensor_position_tilt_decenter",
        "label": "Sensor position, tilt, decenter",
        "use": "Assembly tolerance seed for field asymmetry and CRA mismatch studies.",
    },
    {
        "domain": "Module Coupling",
        "id": "vignetting_shading",
        "label": "Vignetting / shading",
        "use": "Field relative illumination seed.",
    },
    {
        "domain": "Module Coupling",
        "id": "wavelength_dependent_cra_pupil",
        "label": "Wavelength-dependent chief ray / pupil behavior",
        "use": "Chromatic pupil and CRA behavior marker for color shading work.",
    },
]

REQ_BY_ID = {item["id"]: item for item in REQUIREMENTS}


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
            writer.writerow({column: row.get(column, "") for column in columns})


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def unique_values(rows: list[dict[str, str]], key: str) -> set[str]:
    return {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}


def count_nonempty(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")).strip())


def compact_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def join_paths(*paths: Path) -> str:
    return "; ".join(repo_rel(path) for path in paths if path)


def gate_from_rows(rows: list[dict[str, str]], gate_key: str, default: str = "MISSING") -> str:
    gates = {str(row.get(gate_key, "")).strip().upper() for row in rows if str(row.get(gate_key, "")).strip()}
    if "FAIL" in gates:
        return "FAIL"
    if "MISSING" in gates:
        return "MISSING"
    if "CHECK" in gates:
        return "CHECK"
    if "PASS" in gates:
        return "PASS"
    return default


def source_exists_list(source_artifacts: str) -> list[str]:
    missing: list[str] = []
    for part in [item.strip() for item in source_artifacts.split(";") if item.strip()]:
        if not abs_from_repo(part).exists():
            missing.append(part)
    return missing


def coverage_row(
    sensor: dict[str, str],
    requirement_id: str,
    *,
    research_status: str,
    research_gate: str,
    product_gate: str,
    row_count: int,
    source_artifacts: str,
    evidence_summary: dict[str, Any] | str,
    primary_blocker: str,
    notes: str,
) -> dict[str, Any]:
    req = REQ_BY_ID[requirement_id]
    evidence_text = evidence_summary if isinstance(evidence_summary, str) else json.dumps(evidence_summary, sort_keys=True)
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "domain": req["domain"],
        "requirement_id": req["id"],
        "requirement": req["label"],
        "camera_e2e_use": req["use"],
        "research_status": research_status,
        "research_gate": research_gate,
        "product_gate": product_gate,
        "row_count": row_count,
        "source_artifacts": source_artifacts,
        "evidence_summary": evidence_text,
        "primary_blocker": primary_blocker,
        "notes": notes,
    }


def build_sensor_coverage(
    sensor: dict[str, str],
    *,
    paths: dict[str, Path],
    runtime_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
    spectral_rows: list[dict[str, str]],
    color_matrix: dict[str, str],
    material_rows: list[dict[str, str]],
    material_summary: dict[str, str],
    cfa_provenance: dict[str, str],
    electrical_rows: list[dict[str, str]],
    readout_rows: list[dict[str, str]],
    binning_rows: list[dict[str, str]],
    electrical_summary: dict[str, str],
    module_rows: list[dict[str, str]],
    module_summary: dict[str, str],
    prior_summary: dict[str, str],
    probe_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    production_gate = sensor.get("production_lut_gate", "FAIL") or "FAIL"
    sensor_primary_blockers = sensor.get("primary_blockers", "")
    cfa_assumption_gate = cfa_provenance.get("cfa_assumption_gate", "")
    cfa_blocker = cfa_provenance.get("primary_blocker", "") if cfa_assumption_gate in {"MISSING", "FAIL"} else ""
    cfa_blocker_prefix = f"{cfa_blocker}; " if cfa_blocker else ""

    rows.append(
        coverage_row(
            sensor,
            "optical_material_nk_ri",
            research_status="proxy_material_table_ready",
            research_gate=material_summary.get("research_gate", "CHECK") or "CHECK",
            product_gate=material_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(material_rows),
            source_artifacts=join_paths(paths["material_lut"], paths["material_summary"], paths["cfa_provenance"]),
            evidence_summary={
                "material_rows": len(material_rows),
                "material_families": compact_counts(material_rows, "material_family"),
                "measured_material_count": material_summary.get("measured_material_count", "0"),
                "proxy_material_count": material_summary.get("proxy_material_count", ""),
                "cfa_proxy_library_id": material_summary.get("cfa_proxy_library_id", ""),
                "cfa_provenance_class": cfa_provenance.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa_assumption_gate,
                "generic_rgb_fallback_detected": cfa_provenance.get("generic_rgb_fallback_detected", ""),
            },
            primary_blocker=cfa_blocker_prefix + material_summary.get("primary_blocker", "measured material n,k and stack geometry missing"),
            notes="Explicit rows exist for FDTD input, but they are proxy/prior unless measured source flags are present. CFA assumption gate must be checked for color-specific use.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "spectral_response_qe",
            research_status="runtime_response_plus_cfa_spectral_seed",
            research_gate="CHECK" if runtime_rows and spectral_rows else "MISSING",
            product_gate=production_gate,
            row_count=len(runtime_rows) + len(spectral_rows) + len(probe_rows),
            source_artifacts=join_paths(paths["runtime_lut"], paths["spectral_response"], paths["probe_summary"], paths["cfa_provenance"]),
            evidence_summary={
                "runtime_rows": len(runtime_rows),
                "spectral_rows": len(spectral_rows),
                "probe_summary_rows": len(probe_rows),
                "field_points": len({(row.get("field_x_norm", ""), row.get("field_z_norm", "")) for row in runtime_rows}),
                "wavelengths_nm": sorted(unique_values(runtime_rows, "wavelength_nm")),
                "color_channels": sorted(unique_values(runtime_rows, "color_channel")),
                "runtime_gate_counts": compact_counts(runtime_rows, "combined_evidence_gate"),
                "spectral_cfa_assumption_gate_counts": compact_counts(spectral_rows, "cfa_assumption_gate"),
                "cfa_provenance_class": cfa_provenance.get("cfa_provenance_class", ""),
                "cfa_assumption_gate": cfa_assumption_gate,
            },
            primary_blocker=cfa_blocker_prefix + sensor_primary_blockers,
            notes="Loadable as research/trend response. Product QE and color-specific use need measured stack/material, CFA provenance, and high-resolution converged Meep sweeps.",
        )
    )

    matrix_applicability = color_matrix.get("applicability", "")
    is_mono_na = matrix_applicability == "monochrome_not_applicable" or sensor.get("color_matrix_gate", "") == "MISSING"
    rows.append(
        coverage_row(
            sensor,
            "color_response_matrix",
            research_status="not_applicable_monochrome" if is_mono_na else "rgb_to_xyz_seed_ready",
            research_gate="N/A" if is_mono_na else color_matrix.get("gate", "CHECK") or "CHECK",
            product_gate="N/A" if is_mono_na else "FAIL",
            row_count=1 if color_matrix else 0,
            source_artifacts=join_paths(paths["color_matrix"], paths["cfa_provenance"]),
            evidence_summary={
                "applicability": matrix_applicability,
                "matrix_role": color_matrix.get("matrix_role", ""),
                "gate": color_matrix.get("gate", ""),
                "cfa_provenance_class": color_matrix.get("cfa_provenance_class", cfa_provenance.get("cfa_provenance_class", "")),
                "cfa_assumption_gate": color_matrix.get("cfa_assumption_gate", cfa_assumption_gate),
                "color_accuracy_gate": color_matrix.get("color_accuracy_gate", ""),
                "generic_rgb_fallback_detected": color_matrix.get("generic_rgb_fallback_detected", cfa_provenance.get("generic_rgb_fallback_detected", "")),
                "spectral_rows": len(spectral_rows),
            },
            primary_blocker="monochrome sensor has no RGB color matrix requirement"
            if is_mono_na
            else cfa_blocker_prefix + "CCM/color calibration target, illuminant set, and measured spectral response missing",
            notes="RGB sensors get an equal-energy plumbing seed. Monochrome sensors keep spectral response and mark matrix as not applicable. Color-accuracy workflows must also require cfa_assumption_gate and color_accuracy_gate.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "optical_crosstalk_kernel",
            research_status="compact_optical_kernel_ready",
            research_gate=gate_from_rows(kernel_rows, "evidence_gate", "CHECK") if kernel_rows else "MISSING",
            product_gate=production_gate,
            row_count=len(kernel_rows),
            source_artifacts=join_paths(paths["kernel_lut"], paths["runtime_lut"]),
            evidence_summary={
                "kernel_rows": len(kernel_rows),
                "runtime_ids": len(unique_values(kernel_rows, "runtime_id")),
                "neighborhoods": sorted(unique_values(runtime_rows, "crosstalk_neighborhood")),
                "max_output_crosstalk_fraction": max([row.get("output_crosstalk_fraction", "") for row in runtime_rows] or [""]),
                "kernel_gate_counts": compact_counts(kernel_rows, "evidence_gate"),
            },
            primary_blocker="finite-array crosstalk convergence and measured boundary conditions missing",
            notes="Kernel is loadable for system sensitivity studies, not product crosstalk calibration.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "angular_cra_response",
            research_status="field_cra_prior_response_ready",
            research_gate="CHECK" if runtime_rows and count_nonempty(runtime_rows, "cra_x_deg") else "MISSING",
            product_gate=module_summary.get("cra_mismatch_gate", "MISSING") or "MISSING",
            row_count=len(runtime_rows),
            source_artifacts=join_paths(paths["runtime_lut"], paths["module_lut"]),
            evidence_summary={
                "runtime_rows": len(runtime_rows),
                "field_points": len({(row.get("field_x_norm", ""), row.get("field_z_norm", "")) for row in runtime_rows}),
                "cra_mismatch_gate_counts": compact_counts(runtime_rows, "cra_mismatch_gate"),
                "max_chief_ray_total_deg": module_summary.get("max_chief_ray_total_deg", ""),
                "max_cra_mismatch_total_deg": module_summary.get("max_cra_mismatch_total_deg", ""),
            },
            primary_blocker="measured lens CRA map and sensor CRA acceptance/ML-shift reference missing",
            notes="CRA priors are present. Mismatch cannot pass until lens and sensor CRA are both measured or raytraced.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "microlens_ocl_shift_map",
            research_status="design_prior_shift_map_ready",
            research_gate="CHECK" if runtime_rows and count_nonempty(runtime_rows, "lens_shift_x_um") else "MISSING",
            product_gate="MISSING",
            row_count=count_nonempty(runtime_rows, "lens_shift_x_um"),
            source_artifacts=join_paths(paths["runtime_lut"], paths["module_lut"]),
            evidence_summary={
                "shift_rows": count_nonempty(runtime_rows, "lens_shift_x_um"),
                "shift_models": compact_counts(runtime_rows, "lens_shift_model"),
                "module_shift_rows": count_nonempty(module_rows, "lens_shift_x_um"),
            },
            primary_blocker="field-specific measured microlens/OCL shift map missing",
            notes="Uses tan(CRA)-based design prior. Replace with measured or design database OCL offsets.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "conversion_gain_fwc_saturation_nonlinearity",
            research_status="electrical_prior_seed_ready",
            research_gate=electrical_summary.get("research_gate", "CHECK") or "CHECK",
            product_gate=electrical_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(electrical_rows),
            source_artifacts=join_paths(paths["electrical_lut"], paths["electrical_summary"], paths["prior_summary"]),
            evidence_summary={
                "full_well_e": electrical_summary.get("full_well_e", ""),
                "conversion_gain_uv_per_e": electrical_summary.get("conversion_gain_uv_per_e", ""),
                "electrical_rows": len(electrical_rows),
                "signal_fractions": sorted(unique_values(electrical_rows, "signal_fraction")),
            },
            primary_blocker=electrical_summary.get("primary_blocker", "measured electrical calibration missing"),
            notes="Prior values enable CameraE2E signal path tests. They are not sensor-characterization values.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "dark_current_temperature_exposure",
            research_status="dark_current_temperature_exposure_prior_ready",
            research_gate=electrical_summary.get("research_gate", "CHECK") or "CHECK",
            product_gate=electrical_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(electrical_rows),
            source_artifacts=join_paths(paths["electrical_lut"]),
            evidence_summary={
                "temperatures_c": sorted(unique_values(electrical_rows, "temperature_c")),
                "exposures_s": sorted(unique_values(electrical_rows, "exposure_s")),
                "dark_current_25c_e_per_s": electrical_summary.get("dark_current_25c_e_per_s", ""),
            },
            primary_blocker="dark current and DSNU need measured temperature/exposure characterization",
            notes="Uses a prior temperature model and exposure grid.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "dsnu_prnu",
            research_status="dsnu_prnu_prior_ready",
            research_gate=electrical_summary.get("research_gate", "CHECK") or "CHECK",
            product_gate=electrical_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(electrical_rows),
            source_artifacts=join_paths(paths["electrical_lut"], paths["electrical_summary"]),
            evidence_summary={
                "dsnu_e_rms": electrical_summary.get("dsnu_e_rms", ""),
                "prnu_pct_rms": electrical_summary.get("prnu_pct_rms", ""),
            },
            primary_blocker="DSNU/PRNU are fixed-pattern measured calibration items, currently priors only",
            notes="Can be used to test ISP/noise plumbing, not to predict product non-uniformity.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "temporal_noise",
            research_status="temporal_noise_prior_ready",
            research_gate=electrical_summary.get("research_gate", "CHECK") or "CHECK",
            product_gate=electrical_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(electrical_rows),
            source_artifacts=join_paths(paths["electrical_lut"], paths["probe_summary"]),
            evidence_summary={
                "read_noise_e_rms": electrical_summary.get("read_noise_e_rms", ""),
                "noise_terms": ["shot_noise_e_rms", "dark_shot_noise_e_rms", "read_reset_sf_adc_noise_e_rms"],
                "probe_rows": len(probe_rows),
            },
            primary_blocker="measured temporal-noise decomposition and gain/readout calibration missing",
            notes="Noise components are explicit prior seeds for end-to-end testing.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "charge_collection_electrical_crosstalk",
            research_status="electrical_crosstalk_marker_only",
            research_gate=gate_from_rows(electrical_rows, "charge_collection_electrical_crosstalk_gate", "CHECK")
            if electrical_rows
            else "MISSING",
            product_gate=electrical_summary.get("product_lut_gate", "FAIL") or "FAIL",
            row_count=len(electrical_rows),
            source_artifacts=join_paths(paths["electrical_lut"]),
            evidence_summary={
                "electrical_crosstalk_gates": compact_counts(electrical_rows, "charge_collection_electrical_crosstalk_gate"),
                "models": compact_counts(electrical_rows, "charge_collection_electrical_crosstalk_model"),
            },
            primary_blocker="calibrated TCAD charge collection/diffusion model missing",
            notes="This is a risk marker and prior, separate from optical crosstalk kernels.",
        )
    )

    readout_product_gate = electrical_summary.get("product_lut_gate", "FAIL") or "FAIL"
    for requirement_id, fields, blocker, note in [
        (
            "analog_digital_gain",
            {"analog_gain_x": sorted(unique_values(readout_rows, "analog_gain_x")), "digital_gain_x": sorted(unique_values(readout_rows, "digital_gain_x"))},
            "measured analog/digital gain and saturation calibration missing",
            "Gain rows are suitable for raw pipeline tests.",
        ),
        (
            "black_level_optical_black",
            {"black_level_dn": electrical_summary.get("black_level_dn", ""), "optical_black_models": compact_counts(readout_rows, "optical_black_model")},
            "optical-black behavior needs measured black-level calibration",
            "Black-level behavior is a fixed-offset seed.",
        ),
        (
            "adc_clipping_quantization",
            {"adc_bit_depth": electrical_summary.get("adc_bit_depth", ""), "clipping_values": sorted(unique_values(readout_rows, "clipping_dn"))},
            "ADC transfer, clipping, and quantization need readout calibration",
            "ADC rows expose bit depth and clipping for raw-DN tests.",
        ),
        (
            "row_column_fpn_timing_direction",
            {
                "row_fpn_present": count_nonempty(readout_rows, "row_fpn_dn_rms") > 0,
                "column_fpn_present": count_nonempty(readout_rows, "column_fpn_dn_rms") > 0,
                "readout_direction": sorted(unique_values(readout_rows, "readout_direction")),
                "line_time_us": sorted(unique_values(readout_rows, "line_time_us")),
            },
            "row/column FPN and timing require sensor readout measurements or datasheet timing",
            "FPN and timing are prior seeds with unknown readout direction where source data is absent.",
        ),
        (
            "defect_hot_pixel_stats",
            {"hot_pixel_fraction": sorted(unique_values(readout_rows, "hot_pixel_fraction")), "defect_pixel_fraction": sorted(unique_values(readout_rows, "defect_pixel_fraction"))},
            "defect and hot-pixel statistics require measured production distribution",
            "Defect statistics are placeholder priors for defect-injection tests.",
        ),
    ]:
        rows.append(
            coverage_row(
                sensor,
                requirement_id,
                research_status="readout_prior_seed_ready",
                research_gate=gate_from_rows(readout_rows, "research_gate", "CHECK") if readout_rows else "MISSING",
                product_gate=readout_product_gate,
                row_count=len(readout_rows),
                source_artifacts=join_paths(paths["readout_lut"], paths["electrical_summary"]),
                evidence_summary={"readout_rows": len(readout_rows), **fields},
                primary_blocker=blocker,
                notes=note,
            )
        )

    rows.append(
        coverage_row(
            sensor,
            "binning_remosaic_modes",
            research_status="binning_remosaic_prior_ready",
            research_gate=gate_from_rows(binning_rows, "research_gate", "CHECK") if binning_rows else "MISSING",
            product_gate=readout_product_gate,
            row_count=len(binning_rows),
            source_artifacts=join_paths(paths["binning_lut"], paths["kernel_lut"]),
            evidence_summary={
                "binning_rows": len(binning_rows),
                "mode_ids": sorted(unique_values(binning_rows, "mode_id")),
                "binning_group_size": prior_summary.get("binning_group_size", electrical_summary.get("binning_group_size", "")),
                "remosaic_risk": sorted(unique_values(binning_rows, "remosaic_risk")),
            },
            primary_blocker="measured mode-specific gain/noise/crosstalk and remosaic calibration missing",
            notes="Rows state how single-pixel kernels must be reinterpreted for binned outputs.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "lens_raytrace_field_cra_map",
            research_status="module_cra_prior_ready",
            research_gate=module_summary.get("research_use_gate", "CHECK") or "CHECK",
            product_gate=module_summary.get("product_lut_gate", "MISSING") or "MISSING",
            row_count=len(module_rows),
            source_artifacts=join_paths(paths["module_lut"], paths["module_summary"]),
            evidence_summary={
                "module_rows": len(module_rows),
                "field_cases": sorted(unique_values(module_rows, "field_case")),
                "max_chief_ray_total_deg": module_summary.get("max_chief_ray_total_deg", ""),
                "measurement_gate": sorted(unique_values(module_rows, "measurement_gate")),
            },
            primary_blocker=module_summary.get("primary_blocker", "module raytrace/measured CRA missing"),
            notes="Default camera field priors are present. Replace with lens raytrace per module.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "sensor_position_tilt_decenter",
            research_status="assembly_tolerance_prior_ready",
            research_gate=module_summary.get("research_use_gate", "CHECK") or "CHECK",
            product_gate=module_summary.get("product_lut_gate", "MISSING") or "MISSING",
            row_count=len(module_rows),
            source_artifacts=join_paths(paths["module_lut"], paths["prior_summary"]),
            evidence_summary={
                "sensor_decenter_sigma_prior_um": sorted(unique_values(module_rows, "sensor_decenter_sigma_prior_um")),
                "sensor_tilt_sigma_prior_deg": sorted(unique_values(module_rows, "sensor_tilt_sigma_prior_deg")),
                "module_rows": len(module_rows),
            },
            primary_blocker="module assembly tolerance distribution and sensor pose calibration missing",
            notes="Position/tilt/decenter are priors for sensitivity analysis.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "vignetting_shading",
            research_status="vignetting_prior_ready",
            research_gate=module_summary.get("research_use_gate", "CHECK") or "CHECK",
            product_gate=module_summary.get("product_lut_gate", "MISSING") or "MISSING",
            row_count=len(module_rows),
            source_artifacts=join_paths(paths["module_lut"]),
            evidence_summary={
                "min_relative_illumination": module_summary.get("min_relative_illumination", ""),
                "vignetting_models": compact_counts(module_rows, "vignetting_model"),
            },
            primary_blocker="measured or raytraced vignetting/shading map missing",
            notes="Uses a cos4-style seed for module shading tests.",
        )
    )

    rows.append(
        coverage_row(
            sensor,
            "wavelength_dependent_cra_pupil",
            research_status="chromatic_pupil_prior_ready",
            research_gate=gate_from_rows(module_rows, "wavelength_dependent_pupil_gate", "MISSING"),
            product_gate="PASS"
            if boolish(module_summary.get("product_lut_ready")) and module_summary.get("pupil_gate") == "PASS"
            else "MISSING",
            row_count=len(module_rows),
            source_artifacts=join_paths(paths["module_lut"]),
            evidence_summary={
                "pupil_gate": module_summary.get("pupil_gate", ""),
                "pupil_status": compact_counts(module_rows, "wavelength_dependent_pupil_status"),
                "wavelengths_nm": sorted(unique_values(module_rows, "wavelength_nm")),
            },
            primary_blocker="wavelength-dependent pupil and raytrace data missing",
            notes="This requirement is explicitly tracked, but current data does not model chromatic pupil behavior.",
        )
    )

    return rows


def build_summary_rows(coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slug = group_rows([{key: str(value) for key, value in row.items()} for row in coverage_rows], "slug")
    rows: list[dict[str, Any]] = []
    for slug, sensor_rows in sorted(by_slug.items()):
        research_counts = Counter(row.get("research_gate", "") for row in sensor_rows)
        product_counts = Counter(row.get("product_gate", "") for row in sensor_rows)
        blockers = []
        for row in sensor_rows:
            if row.get("product_gate") in {"FAIL", "MISSING"} and row.get("primary_blocker"):
                blocker = row["primary_blocker"]
                if blocker not in blockers:
                    blockers.append(blocker)
        first = sensor_rows[0]
        product_ready = (
            product_counts.get("FAIL", 0) == 0
            and product_counts.get("MISSING", 0) == 0
            and product_counts.get("CHECK", 0) == 0
            and product_counts.get("PASS", 0) > 0
        )
        rows.append(
            {
                "slug": slug,
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "requirement_count": len(sensor_rows),
                "research_pass_count": research_counts.get("PASS", 0),
                "research_check_count": research_counts.get("CHECK", 0),
                "research_missing_count": research_counts.get("MISSING", 0),
                "research_fail_count": research_counts.get("FAIL", 0),
                "research_na_count": research_counts.get("N/A", 0),
                "product_pass_count": product_counts.get("PASS", 0),
                "product_check_count": product_counts.get("CHECK", 0),
                "product_missing_count": product_counts.get("MISSING", 0),
                "product_fail_count": product_counts.get("FAIL", 0),
                "product_na_count": product_counts.get("N/A", 0),
                "research_gate_counts": json.dumps(dict(sorted(research_counts.items())), sort_keys=True),
                "product_gate_counts": json.dumps(dict(sorted(product_counts.items())), sort_keys=True),
                "product_ready": product_ready,
                "primary_blockers": "; ".join(blockers[:8]),
            }
        )
    return rows


def validate(
    *,
    coverage_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    package: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    sensor_count = int(package.get("sensor_count", 0) or 0)
    expected_row_count = sensor_count * len(REQUIREMENTS)
    if sensor_count and len(summary_rows) != sensor_count:
        issues.append({"severity": "error", "code": "sensor_count_mismatch", "expected": sensor_count, "actual": len(summary_rows)})
    if sensor_count and len(coverage_rows) != expected_row_count:
        issues.append(
            {
                "severity": "error",
                "code": "coverage_row_count_mismatch",
                "expected": expected_row_count,
                "actual": len(coverage_rows),
            }
        )

    seen: set[tuple[str, str]] = set()
    expected_req_ids = set(REQ_BY_ID)
    by_slug: dict[str, set[str]] = defaultdict(set)
    for row in coverage_rows:
        slug = str(row.get("slug", ""))
        requirement_id = str(row.get("requirement_id", ""))
        key = (slug, requirement_id)
        if key in seen:
            issues.append({"severity": "error", "code": "duplicate_requirement_row", "slug": slug, "requirement_id": requirement_id})
        seen.add(key)
        by_slug[slug].add(requirement_id)
        research_gate = str(row.get("research_gate", ""))
        product_gate = str(row.get("product_gate", ""))
        if research_gate not in {"PASS", "CHECK", "MISSING", "FAIL", "N/A"}:
            issues.append({"severity": "error", "code": "invalid_research_gate", "slug": slug, "requirement_id": requirement_id, "gate": research_gate})
        if product_gate not in {"PASS", "CHECK", "MISSING", "FAIL", "N/A"}:
            issues.append({"severity": "error", "code": "invalid_product_gate", "slug": slug, "requirement_id": requirement_id, "gate": product_gate})
        if research_gate in {"PASS", "CHECK"} and safe_int(row.get("row_count")) <= 0:
            issues.append({"severity": "error", "code": "covered_row_has_zero_count", "slug": slug, "requirement_id": requirement_id})
        for missing_source in source_exists_list(str(row.get("source_artifacts", ""))):
            issues.append(
                {
                    "severity": "error",
                    "code": "source_artifact_missing",
                    "slug": slug,
                    "requirement_id": requirement_id,
                    "path": missing_source,
                }
            )

    for slug, req_ids in by_slug.items():
        missing_reqs = sorted(expected_req_ids - req_ids)
        if missing_reqs:
            issues.append({"severity": "error", "code": "sensor_requirement_rows_missing", "slug": slug, "missing": ",".join(missing_reqs)})

    product_ready_count = sum(1 for row in summary_rows if boolish(row.get("product_ready")))
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
    if error_count:
        status = "FAIL"
    elif product_ready_count == len(summary_rows) and summary_rows:
        status = "PRODUCT_COVERAGE_READY"
    else:
        status = "RESEARCH_COVERAGE_READY_PRODUCT_BLOCKED"
    return {
        "schema": "camera_e2e_coverage_matrix_validation_v1",
        "pass": error_count == 0,
        "status": status,
        "issue_count": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], summary_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> None:
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
    issue_rows = validation.get("issues", [])
    issue_html = html_table(issue_rows, ["severity", "code", "slug", "requirement_id", "path"]) if issue_rows else '<p class="pass">No structural coverage issues.</p>'
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Requirement Coverage Matrix</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Requirement Coverage Matrix</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This maps each sensor requirement to concrete artifacts and preserves research/product gate separation.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">coverage status</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("requirement_count_per_sensor", 0))}</div><div class="muted">requirements per sensor</div></div>
<div class="card"><div class="metric warn">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Policy</h2>
<ul>
<li>Research use means the artifact is structurally loadable and useful for trend/system tests.</li>
<li>Product use remains blocked when the row depends on proxy material, prior electrical/readout data, assumed CRA, or unconverged optical/electrical solvers.</li>
<li>Monochrome sensors mark RGB color matrix as <code>N/A</code>, while keeping spectral/clear response rows.</li>
</ul>
<h2>Issues</h2>{issue_html}
<h2>Sensor Summary</h2>{html_table(summary_rows, SUMMARY_COLUMNS)}
<h2>Requirement Coverage</h2>{html_table(coverage_rows, COVERAGE_COLUMNS)}
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
    outputs["camera_e2e_coverage_matrix_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_coverage_matrix_csv"] = payload["outputs"]["coverage_csv"]
    outputs["camera_e2e_coverage_summary_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_coverage_matrix_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_coverage_matrix"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "sensor_count": payload["sensor_count"],
        "coverage_row_count": payload["coverage_row_count"],
        "product_ready_count": payload["product_ready_count"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def export_coverage(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    package = read_json(package_dir / "camera_e2e_lut_package.json")

    paths = {
        "runtime_lut": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv",
        "kernel_lut": package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv",
        "spectral_response": package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv",
        "color_matrix": package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv",
        "material_lut": package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv",
        "material_summary": package_dir / "camera_e2e_material_tables" / "camera_e2e_material_summary.csv",
        "cfa_provenance": package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv",
        "electrical_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_noise_lut.csv",
        "readout_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_readout_gain_lut.csv",
        "binning_lut": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_binning_remosaic_lut.csv",
        "electrical_summary": package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_readout_summary.csv",
        "module_lut": package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_field_lut.csv",
        "module_summary": package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_summary.csv",
        "prior_summary": package_dir / "camera_e2e_prior_seed_models" / "camera_e2e_prior_seed_summary.csv",
        "probe_summary": package_dir / "camera_e2e_sensor_probe_all_sensors" / "camera_e2e_sensor_probe_summary.csv",
        "sensor_summary": package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv",
    }

    sensor_rows = read_csv_rows(paths["sensor_summary"])
    runtime_by_slug = group_rows(read_csv_rows(paths["runtime_lut"]), "slug")
    kernel_by_slug = group_rows(read_csv_rows(paths["kernel_lut"]), "slug")
    spectral_by_slug = group_rows(read_csv_rows(paths["spectral_response"]), "slug")
    color_matrix_by_slug = index_by(read_csv_rows(paths["color_matrix"]), "slug")
    material_by_slug = group_rows(read_csv_rows(paths["material_lut"]), "slug")
    material_summary_by_slug = index_by(read_csv_rows(paths["material_summary"]), "slug")
    cfa_provenance_by_slug = index_by(read_csv_rows(paths["cfa_provenance"]), "slug")
    electrical_by_slug = group_rows(read_csv_rows(paths["electrical_lut"]), "slug")
    readout_by_slug = group_rows(read_csv_rows(paths["readout_lut"]), "slug")
    binning_by_slug = group_rows(read_csv_rows(paths["binning_lut"]), "slug")
    electrical_summary_by_slug = index_by(read_csv_rows(paths["electrical_summary"]), "slug")
    module_by_slug = group_rows(read_csv_rows(paths["module_lut"]), "slug")
    module_summary_by_slug = index_by(read_csv_rows(paths["module_summary"]), "slug")
    prior_summary_by_slug = index_by(read_csv_rows(paths["prior_summary"]), "slug")
    probe_by_slug = group_rows(read_csv_rows(paths["probe_summary"]), "slug")

    coverage_rows: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        coverage_rows.extend(
            build_sensor_coverage(
                sensor,
                paths=paths,
                runtime_rows=runtime_by_slug.get(slug, []),
                kernel_rows=kernel_by_slug.get(slug, []),
                spectral_rows=spectral_by_slug.get(slug, []),
                color_matrix=color_matrix_by_slug.get(slug, {}),
                material_rows=material_by_slug.get(slug, []),
                material_summary=material_summary_by_slug.get(slug, {}),
                cfa_provenance=cfa_provenance_by_slug.get(slug, {}),
                electrical_rows=electrical_by_slug.get(slug, []),
                readout_rows=readout_by_slug.get(slug, []),
                binning_rows=binning_by_slug.get(slug, []),
                electrical_summary=electrical_summary_by_slug.get(slug, {}),
                module_rows=module_by_slug.get(slug, []),
                module_summary=module_summary_by_slug.get(slug, {}),
                prior_summary=prior_summary_by_slug.get(slug, {}),
                probe_rows=probe_by_slug.get(slug, []),
            )
        )

    summary_rows = build_summary_rows(coverage_rows)
    validation = validate(coverage_rows=coverage_rows, summary_rows=summary_rows, package=package)

    coverage_csv = output_dir / "camera_e2e_coverage_matrix.csv"
    summary_csv = output_dir / "camera_e2e_coverage_summary.csv"
    report_json = output_dir / "camera_e2e_coverage_matrix.json"
    html_path = output_dir / "index.html"
    product_ready_count = sum(1 for row in summary_rows if boolish(row.get("product_ready")))
    payload = {
        "schema": "camera_e2e_coverage_matrix_export_v1",
        "artifact_role": "camera_e2e_requirement_to_artifact_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(summary_rows),
        "requirement_count_per_sensor": len(REQUIREMENTS),
        "coverage_row_count": len(coverage_rows),
        "summary_row_count": len(summary_rows),
        "product_ready_count": product_ready_count,
        "validation": validation,
        "requirement_ids": [req["id"] for req in REQUIREMENTS],
        "gate_policy": {
            "research_PASS": "Measured or structurally reliable research artifact.",
            "research_CHECK": "Loadable research/proxy/prior artifact; validate before design decisions.",
            "research_MISSING": "Requirement is tracked but current artifact has no useful rows.",
            "product_PASS": "Product LUT use allowed for this requirement.",
            "product_FAIL_or_MISSING": "Do not use as product sensor characterization.",
            "N/A": "Requirement is not applicable to this sensor, for example RGB matrix on monochrome sensors.",
        },
        "outputs": {
            "json": repo_rel(report_json),
            "coverage_csv": repo_rel(coverage_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
    }

    write_csv(coverage_csv, coverage_rows, COVERAGE_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(report_json, payload)
    write_html(html_path, payload, summary_rows, coverage_rows)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    payload = export_coverage(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "sensor_count": payload["sensor_count"],
                "coverage_row_count": payload["coverage_row_count"],
                "product_ready_count": payload["product_ready_count"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not payload["validation"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
