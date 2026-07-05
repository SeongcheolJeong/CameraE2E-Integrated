#!/usr/bin/env python3
"""Export per-sensor CameraE2E model manifests.

The existing CameraE2E package already contains solver LUTs, compact
crosstalk kernels, readiness gates, stack configs, and TechInsights-derived
optical/CFA evidence. This exporter makes those pieces easier to consume by a
camera-system simulator:

- one JSON manifest per selected sensor;
- a sensor-level summary CSV;
- an item-level coverage matrix for Optical/Color, Pixel/Electrical,
  Readout/RAW, and Module Coupling requirements;
- an HTML review page.

The exporter does not promote proxy data to measured data. Missing measured
CRA maps, measured n,k, noise/readout targets, and calibrated electrical
targets remain explicit blockers in the generated manifests.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_sensor_models"

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "ocl_mode_guess",
    "effective_ocl_mode",
    "optical_readiness",
    "cfa_proxy_nk_enabled",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "color_accuracy_gate",
    "quantitative_field_pass_points",
    "quantitative_field_required_points",
    "field_runtime_rows",
    "runtime_rows",
    "kernel_rows",
    "field_coverage_fraction",
    "field_coverage_gate",
    "crosstalk_coverage_gate",
    "cra_mismatch_gate",
    "cra_mismatch_profiles",
    "max_cra_mismatch_total_deg",
    "color_spectral_rows",
    "color_matrix_gate",
    "optical_color_gate",
    "pixel_electrical_gate",
    "readout_raw_gate",
    "module_coupling_gate",
    "research_ingest_gate",
    "production_lut_gate",
    "camera_e2e_product_ready",
    "primary_blockers",
    "model_json",
]

MATRIX_COLUMNS = [
    "slug",
    "code",
    "section",
    "item",
    "status",
    "gate",
    "source",
    "notes",
    "value",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


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
        rows = list(csv.DictReader(handle))
    cleaned: list[dict[str, str]] = []
    for row in rows:
        first_key = next(iter(row), "")
        if first_key and row.get(first_key) == first_key:
            continue
        cleaned.append(row)
    return cleaned


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: finite_csv(row.get(column, "")) for column in columns})


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def finite_csv(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return ""
    return value


def gate_rank(gate: str) -> int:
    order = {"PASS": 0, "CHECK": 1, "MISSING": 2, "FAIL": 3, "BLOCKED": 3}
    return order.get(str(gate).upper(), 2)


def worst_gate(*gates: str) -> str:
    normalized = [str(g or "MISSING").upper() for g in gates if str(g or "").strip()]
    if not normalized:
        return "MISSING"
    return max(normalized, key=gate_rank)


def item(section: str, name: str, status: str, gate: str, source: str, notes: str, value: Any = "") -> dict[str, Any]:
    return {
        "section": section,
        "item": name,
        "status": status,
        "gate": gate,
        "source": source,
        "notes": notes,
        "value": value,
    }


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in result:
            result[value] = row
    return result


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return grouped


def compact_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = str(row.get(column, "") or "").strip()
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def summarize_cra_mismatch(runtime_rows: list[dict[str, str]]) -> dict[str, Any]:
    gates = compact_counts(runtime_rows, "cra_mismatch_gate")
    profiles = compact_counts(runtime_rows, "cra_mismatch_tolerance_profile")
    values = [as_float(row.get("cra_mismatch_total_deg")) for row in runtime_rows]
    finite_values = [value for value in values if value is not None]
    return {
        "gate": worst_gate(*gates.keys()) if gates else "MISSING",
        "gate_counts": gates,
        "tolerance_profile_counts": profiles,
        "max_total_mismatch_deg": max(finite_values) if finite_values else None,
        "rows_with_sensor_cra_reference": sum(
            1
            for row in runtime_rows
            if row.get("sensor_cra_x_deg", "") not in {"", "nan", "NaN"}
            and row.get("sensor_cra_z_deg", "") not in {"", "nan", "NaN"}
        ),
        "row_count": len(runtime_rows),
    }


def load_optical_maps(optical_db: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    summary_rows = read_csv_rows(optical_db / "optical_qe_summary.csv")
    by_code = index_by(summary_rows, "code")
    models: dict[str, dict[str, Any]] = {}
    for code, row in by_code.items():
        rel = row.get("model_json", "")
        if not rel:
            continue
        model_path = optical_db / rel
        models[code] = read_json(model_path)
    return by_code, models


def summarize_runtime(rows: list[dict[str, str]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if row.get("field_evidence_gate") == "PASS"]
    colors = sorted({row.get("color_channel", "") for row in rows if row.get("color_channel")})
    wavelengths = sorted({row.get("wavelength_nm", "") for row in rows if row.get("wavelength_nm")}, key=lambda v: as_float(v, 0) or 0)
    fields = sorted({(row.get("field_x_norm", ""), row.get("field_z_norm", "")) for row in rows})
    return {
        "row_count": len(rows),
        "pass_row_count": len(pass_rows),
        "colors": colors,
        "wavelengths_nm": wavelengths,
        "field_point_count": len(fields),
        "field_points": [{"field_x_norm": x, "field_z_norm": z} for x, z in fields[:16]],
    }


def build_color_response_matrix(runtime_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    center_rows = [
        row
        for row in runtime_rows
        if as_float(row.get("field_x_norm"), 999) == 0.0 and as_float(row.get("field_z_norm"), 999) == 0.0
    ]
    if not center_rows:
        center_rows = runtime_rows
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for row in center_rows:
        key = (row.get("wavelength_nm", ""), row.get("color_channel", ""))
        response = as_float(row.get("response_nominal"))
        if response is None:
            continue
        matrix[key] = {
            "wavelength_nm": as_float(row.get("wavelength_nm")),
            "color_channel": row.get("color_channel"),
            "response_nominal": response,
            "response_min": as_float(row.get("response_min")),
            "response_max": as_float(row.get("response_max")),
            "field_evidence_gate": row.get("field_evidence_gate"),
            "confidence_class": row.get("confidence_class"),
        }
    return sorted(matrix.values(), key=lambda r: (r.get("wavelength_nm") or 0, str(r.get("color_channel"))))


def build_shift_map(runtime_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in runtime_rows:
        key = (
            row.get("field_x_norm", ""),
            row.get("field_z_norm", ""),
            row.get("cra_x_deg", ""),
            row.get("cra_z_deg", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "field_x_norm": as_float(row.get("field_x_norm")),
                "field_z_norm": as_float(row.get("field_z_norm")),
                "cra_x_deg": as_float(row.get("cra_x_deg")),
                "cra_z_deg": as_float(row.get("cra_z_deg")),
                "lens_cra_x_deg": as_float(row.get("lens_cra_x_deg")),
                "lens_cra_z_deg": as_float(row.get("lens_cra_z_deg")),
                "sensor_cra_x_deg": as_float(row.get("sensor_cra_x_deg")),
                "sensor_cra_z_deg": as_float(row.get("sensor_cra_z_deg")),
                "cra_mismatch_total_deg": as_float(row.get("cra_mismatch_total_deg")),
                "cra_mismatch_gate": row.get("cra_mismatch_gate", "MISSING"),
                "cra_mismatch_tolerance_profile": row.get("cra_mismatch_tolerance_profile", ""),
                "cra_mismatch_pass_tolerance_deg": as_float(row.get("cra_mismatch_pass_tolerance_deg")),
                "cra_mismatch_check_tolerance_deg": as_float(row.get("cra_mismatch_check_tolerance_deg")),
                "lens_shift_x_um": as_float(row.get("lens_shift_x_um")),
                "lens_shift_z_um": as_float(row.get("lens_shift_z_um")),
                "measurement_gate": row.get("cra_measurement_gate") or "ASSUMED_NOT_MEASURED",
                "source": row.get("cra_source"),
            }
        )
    return output


def resolve_tcad_structure(slug: str) -> dict[str, Any]:
    path = ROOT / "image_sensor_db" / "tcad_structure_db" / "models" / f"{slug}.json"
    return read_json(path)


def build_model(
    *,
    sensor_index_row: dict[str, str],
    sensor_lut: dict[str, Any],
    optical_row: dict[str, str],
    optical_model: dict[str, Any],
    stack_config: dict[str, Any],
    tcad_profile: dict[str, Any],
    tcad_structure: dict[str, Any],
    prior_model: dict[str, Any],
    readiness_row: dict[str, Any],
    coverage_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
    color_spectral_rows: list[dict[str, str]],
    color_matrix_seed_row: dict[str, str],
    cfa_provenance_row: dict[str, str],
    field_map_validation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    slug = sensor_index_row["slug"]
    code = sensor_index_row["code"]
    runtime_summary = summarize_runtime(runtime_rows)
    color_matrix = build_color_response_matrix(runtime_rows)
    cra_mismatch_summary = summarize_cra_mismatch(runtime_rows)
    shift_map = build_shift_map(runtime_rows)
    field_cov = next((row for row in coverage_rows if row.get("solver") == "field"), {})
    crosstalk_cov = next((row for row in coverage_rows if row.get("solver") == "crosstalk"), {})
    stack_materials = stack_config.get("materials", {}) if isinstance(stack_config.get("materials"), dict) else {}
    stack_measured = bool(stack_config.get("calibration_status", {}).get("is_measured"))
    optical_readiness = optical_row.get("optical_readiness") or optical_model.get("readiness", {}).get("level", "missing")
    cfa_proxy = optical_model.get("cfa_proxy_nk", {})
    cfa_proxy_enabled = boolish(optical_row.get("cfa_proxy_nk_enabled")) or bool(cfa_proxy.get("enabled"))
    measured_field_map = field_map_validation.get("gate") in {"MEASURED", "CALIBRATED", "RAYTRACE_VALIDATED", "PASS"}

    field_gate = field_cov.get("gate") or readiness_row.get("quantitative_field_gate") or "MISSING"
    crosstalk_gate = crosstalk_cov.get("gate") or readiness_row.get("finite_array_crosstalk_gate") or "MISSING"
    production_gate = readiness_row.get("production_lut_gate") or sensor_index_row.get("camera_e2e_usage_gate") or "FAIL"
    research_gate = readiness_row.get("research_ingest_gate") or "CHECK"
    cfa_assumption_gate = cfa_provenance_row.get("cfa_assumption_gate") or color_matrix_seed_row.get("cfa_assumption_gate") or "MISSING"
    cfa_provenance_class = cfa_provenance_row.get("cfa_provenance_class") or color_matrix_seed_row.get("cfa_provenance_class") or ""
    color_accuracy_gate = color_matrix_seed_row.get("color_accuracy_gate") or "MISSING"
    cfa_blocker = cfa_provenance_row.get("primary_blocker", "")

    tcad_available = bool(tcad_profile)
    tcad_calibrated = bool(tcad_profile.get("calibration_status", {}).get("is_measured")) if tcad_available else False
    readout_context = sensor_lut.get("sensor", {})
    pixel_prior = prior_model.get("pixel_electrical_prior", {}) if prior_model else {}
    readout_prior = prior_model.get("readout_raw_prior", {}) if prior_model else {}
    module_prior = prior_model.get("module_coupling_prior", {}) if prior_model else {}
    prior_source = (
        f"camera_e2e_prior_seed_models/models/{slug}.json"
        if prior_model
        else ""
    )
    effective_ocl_mode = (
        prior_model.get("sensor", {}).get("effective_ocl_mode")
        or sensor_index_row.get("ocl_mode_guess", "")
    )
    topology_note = ""
    if (
        effective_ocl_mode
        and sensor_index_row.get("ocl_mode_guess")
        and not str(effective_ocl_mode).startswith(str(sensor_index_row.get("ocl_mode_guess")))
    ):
        topology_note = (
            f"Package OCL guess {sensor_index_row.get('ocl_mode_guess')} differs from "
            f"CFA/OCL DB-derived effective mode {effective_ocl_mode}."
        )

    items = [
        item(
            "Optical / Color",
            "CFA/OCL/passivation/Si n,k or RI",
            "measured" if stack_measured else "proxy_prior",
            "PASS" if stack_measured else "CHECK",
            "; ".join(filter(None, [repo_rel(sensor_lut.get("source_artifacts", {}).get("stack_config")), repo_rel(optical_row.get("model_json"))])),
            "Stack config and CFA proxy n,k are FDTD-runnable, but product-specific measured n,k is not present."
            if not stack_measured
            else "Measured stack/material data is marked available.",
            {
                "stack_material_keys": sorted(stack_materials.keys()),
                "cfa_proxy_enabled": cfa_proxy_enabled,
                "cfa_proxy_library_id": cfa_proxy.get("library_id") or optical_row.get("cfa_proxy_library_id"),
            },
        ),
        item(
            "Optical / Color",
            "CFA provenance / color accuracy gate",
            cfa_provenance_class or "missing_cfa_provenance",
            cfa_assumption_gate,
            "camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv",
            cfa_blocker
            or "CFA rows are traceable for research use, but measured CFA n,k/spectral QE is still required for product color accuracy.",
            cfa_provenance_row,
        ),
        item(
            "Optical / Color",
            "Spectral response / QE",
            "partial_fdtd_plus_cfa_proxy_spectral" if runtime_rows and color_spectral_rows else ("partial_fdtd_simulation" if runtime_rows else "missing_solver_result"),
            "CHECK" if runtime_rows else "MISSING",
            "; ".join(
                filter(
                    None,
                    [
                        "camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv" if runtime_rows else "",
                        "camera_e2e_color_response/camera_e2e_spectral_response.csv" if color_spectral_rows else "",
                    ],
                )
            ),
            "Runtime rows are CameraE2E research/trend inputs. They are not calibrated product QE until measured stack/CRA gates pass.",
            {
                "runtime": runtime_summary,
                "color_response_spectral_rows": len(color_spectral_rows),
                "spectral_evidence_gates": compact_counts(color_spectral_rows, "evidence_gate"),
                "spectral_cfa_assumption_gates": compact_counts(color_spectral_rows, "cfa_assumption_gate"),
                "cfa_provenance_class": cfa_provenance_class,
            },
        ),
        item(
            "Optical / Color",
            "Color response matrix",
            "rgb_to_xyz_seed_plus_runtime_spectral_sensitivity" if color_matrix_seed_row else ("runtime_spectral_sensitivity" if color_matrix else "missing"),
            color_matrix_seed_row.get("gate", "CHECK" if color_matrix else "MISSING"),
            "camera_e2e_color_response/camera_e2e_color_matrix_seed.csv" if color_matrix_seed_row else "derived from runtime center/field rows",
            "Provides RGB spectral sensitivity rows for CameraE2E/ISP. XYZ/CCM fitting still needs target illuminants, lens transmission, and calibration.",
            {
                "runtime_center_row_count": len(color_matrix),
                "matrix_seed": color_matrix_seed_row,
                "cfa_provenance_class": cfa_provenance_class,
                "cfa_assumption_gate": cfa_assumption_gate,
                "color_accuracy_gate": color_accuracy_gate,
            },
        ),
        item(
            "Optical / Color",
            "Optical crosstalk kernel",
            "compact_surrogate_or_partial_solver" if kernel_rows else "missing",
            "CHECK" if kernel_rows else "MISSING",
            "camera_e2e_runtime_bundle/camera_e2e_runtime_crosstalk_kernel.csv",
            "Compact kernels are ingestible. Finite-array FDTD crosstalk convergence remains required for product use.",
            {"kernel_rows": len(kernel_rows), "coverage_gate": crosstalk_gate},
        ),
        item(
            "Optical / Color",
            "Angular response / CRA response",
            "design_prior_plus_partial_fdtd" if runtime_rows else "missing",
            "CHECK" if runtime_rows else "MISSING",
            "runtime CRA rows; camera_module_field_map.csv is missing unless imported",
            "CRA values are design priors when no measured/raytrace field map is provided.",
            {
                "field_coverage_fraction": field_cov.get("coverage_fraction", "0"),
                "field_gate": field_gate,
                "cra_mismatch": cra_mismatch_summary,
            },
        ),
        item(
            "Optical / Color",
            "Lens-vs-sensor CRA mismatch tolerance",
            "computed_if_sensor_cra_reference_present" if cra_mismatch_summary["rows_with_sensor_cra_reference"] else "missing_sensor_cra_reference",
            cra_mismatch_summary["gate"],
            "camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv",
            "Mismatch is computed only when lens CRA and sensor/ML/OCL optimized CRA are both supplied. Current DB lacks sensor CRA reference rows.",
            cra_mismatch_summary,
        ),
        item(
            "Optical / Color",
            "Microlens/OCL shift map",
            "measured_or_raytrace" if measured_field_map else "design_prior",
            "PASS" if measured_field_map else "MISSING",
            "camera_module_field_map.csv" if measured_field_map else "runtime design-prior lens shift model",
            "Measured or raytrace-validated ML/OCL shift map is absent; runtime shift values are priors.",
            {"shift_point_count": len(shift_map), "field_map_gate": field_map_validation.get("gate", "MISSING")},
        ),
        item(
            "Pixel / Electrical",
            "Conversion gain",
            "prior_seed" if pixel_prior.get("conversion_gain_uv_per_e") else "missing_measured",
            "CHECK" if pixel_prior.get("conversion_gain_uv_per_e") else "MISSING",
            prior_source or repo_rel(sensor_lut.get("source_artifacts", {}).get("tcad_profile")),
            "Research-only conversion-gain seed is available; replace with measured e-/DN or uV/e- calibration for product use."
            if pixel_prior.get("conversion_gain_uv_per_e")
            else "No calibrated e-/DN or uV/e- conversion-gain target is present. TCAD structure does not define readout conversion gain.",
            pixel_prior.get("conversion_gain_uv_per_e", ""),
        ),
        item(
            "Pixel / Electrical",
            "Full well capacity / saturation / nonlinearity",
            "prior_seed" if pixel_prior.get("full_well_capacity_e") else "missing_measured",
            "CHECK" if pixel_prior.get("full_well_capacity_e") else "MISSING",
            prior_source or repo_rel(sensor_lut.get("source_artifacts", {}).get("tcad_profile")),
            "Research-only FWC/saturation/nonlinearity seed is available; product use needs measured targets or calibrated pinned-PD/FD/TG model."
            if pixel_prior.get("full_well_capacity_e")
            else "FWC, saturation, and nonlinearity need measured targets or a calibrated pinned-PD/FD/TG model.",
            {
                "full_well_capacity_e": pixel_prior.get("full_well_capacity_e"),
                "saturation_signal_e": pixel_prior.get("saturation_signal_e"),
                "nonlinearity": pixel_prior.get("nonlinearity"),
            }
            if pixel_prior
            else "",
        ),
        item(
            "Pixel / Electrical",
            "Dark current vs temperature/exposure",
            "prior_seed" if pixel_prior.get("dark_current") else "missing_measured",
            "CHECK" if pixel_prior.get("dark_current") else "MISSING",
            prior_source or repo_rel(sensor_lut.get("source_artifacts", {}).get("tcad_profile")),
            "Research-only dark-current temperature seed is available; product use needs measured dark-current curves."
            if pixel_prior.get("dark_current")
            else "Dark-current temperature/exposure curves are not available in the current DB.",
            pixel_prior.get("dark_current", ""),
        ),
        item(
            "Pixel / Electrical",
            "DSNU",
            "prior_seed" if pixel_prior.get("dsnu") else "missing_measured",
            "CHECK" if pixel_prior.get("dsnu") else "MISSING",
            prior_source,
            "Research-only DSNU distribution seed is available; product DSNU still requires measured dark-frame statistics."
            if pixel_prior.get("dsnu")
            else "DSNU is a measured fixed-pattern dark offset distribution; it cannot be inferred from teardown geometry alone.",
            pixel_prior.get("dsnu", ""),
        ),
        item(
            "Pixel / Electrical",
            "PRNU",
            "prior_seed" if pixel_prior.get("prnu") else "missing_measured",
            "CHECK" if pixel_prior.get("prnu") else "MISSING",
            prior_source,
            "Research-only PRNU seed is available; product PRNU still requires flat-field statistics."
            if pixel_prior.get("prnu")
            else "PRNU is a measured pixel-gain distribution; current runtime response is deterministic and has no wafer/process variation model.",
            pixel_prior.get("prnu", ""),
        ),
        item(
            "Pixel / Electrical",
            "Temporal noise",
            "prior_seed" if pixel_prior.get("temporal_noise") else "missing_measured",
            "CHECK" if pixel_prior.get("temporal_noise") else "MISSING",
            prior_source,
            "Research-only temporal-noise seed is available. CameraE2E should still compute shot noise from signal; product read/reset/SF/ADC noise requires measured or circuit-calibrated parameters."
            if pixel_prior.get("temporal_noise")
            else "Shot-noise can be added by CameraE2E from signal level, but read/reset/source-follower/ADC noise needs measured or circuit-calibrated parameters.",
            pixel_prior.get("temporal_noise", ""),
        ),
        item(
            "Pixel / Electrical",
            "Charge collection / diffusion / electrical crosstalk",
            "tcad_proxy_structure" if tcad_available else "missing",
            "CHECK" if tcad_available and not tcad_calibrated else ("PASS" if tcad_calibrated else "MISSING"),
            repo_rel(sensor_lut.get("source_artifacts", {}).get("tcad_profile")),
            "DEVSIM/TCAD proxy geometry is available, but implant/TG/FD/interface/mobility calibration is not measured.",
            {
                "tcad_profile_available": tcad_available,
                "tcad_structure_model_available": bool(tcad_structure),
                "calibrated": tcad_calibrated,
            },
        ),
        item(
            "Readout / RAW",
            "Analog/digital gain table",
            "prior_seed" if readout_prior.get("gain_table") else "missing_measured",
            "CHECK" if readout_prior.get("gain_table") else "MISSING",
            prior_source,
            "Research-only gain sweep seed is available; replace with sensor mode/register table for product use."
            if readout_prior.get("gain_table")
            else "No register/gain table or calibrated mode table is present.",
            readout_prior.get("gain_table", ""),
        ),
        item(
            "Readout / RAW",
            "Black level / optical black behavior",
            "prior_seed" if readout_prior.get("black_level") else "missing_measured",
            "CHECK" if readout_prior.get("black_level") else "MISSING",
            prior_source,
            "Research-only black-level seed is available; product use needs optical-black calibration."
            if readout_prior.get("black_level")
            else "No optical-black statistics or black-level calibration is present.",
            readout_prior.get("black_level", ""),
        ),
        item(
            "Readout / RAW",
            "ADC bit depth / clipping / quantization",
            "prior_seed" if readout_prior.get("adc") else "missing_measured",
            "CHECK" if readout_prior.get("adc") else "MISSING",
            prior_source,
            "Research-only ADC/clipping/quantization seed is available; product use needs mode table and calibration."
            if readout_prior.get("adc")
            else "No ADC mode table or clipping model is present.",
            readout_prior.get("adc", ""),
        ),
        item(
            "Readout / RAW",
            "Row/column FPN, rolling shutter timing, readout direction",
            "prior_seed_plus_topology_context" if readout_prior.get("row_column_fpn") else ("partial_topology_context" if readout_context else "missing"),
            "CHECK" if readout_prior.get("row_column_fpn") or readout_context else "MISSING",
            prior_source or repo_rel(sensor_lut.get("source_artifacts", {}).get("tcad_profile")),
            "Research-only row/column FPN and rolling timing seeds are available; product timing/FPN still need mode documentation or measurement."
            if readout_prior.get("row_column_fpn")
            else "Pixel architecture context exists, but row/column FPN and timing need sensor mode documentation or measurement.",
            {
                "pixel_architecture": readout_context.get("pixel_architecture"),
                "has_hdr": readout_context.get("has_hdr"),
                "has_lofic": readout_context.get("has_lofic"),
                "has_pdaf": readout_context.get("has_pdaf"),
                "row_column_fpn": readout_prior.get("row_column_fpn"),
                "rolling_shutter": readout_prior.get("rolling_shutter"),
            },
        ),
        item(
            "Readout / RAW",
            "Defect pixel / hot pixel statistics",
            "prior_seed" if readout_prior.get("defect_pixels") else "missing_measured",
            "CHECK" if readout_prior.get("defect_pixels") else "MISSING",
            prior_source,
            "Research-only defect/hot-pixel distribution seed is available; production maps still require sensor test data."
            if readout_prior.get("defect_pixels")
            else "Defect/hot-pixel maps are production/test data and are not in the current DB.",
            readout_prior.get("defect_pixels", ""),
        ),
        item(
            "Readout / RAW",
            "Binning/remosaic mode gain/noise/crosstalk",
            "prior_seed_plus_topology",
            "CHECK",
            "; ".join(filter(None, [prior_source, "sensor LUT CFA/OCL/pixel architecture"])),
            "Topology and research-only binning/remosaic seeds are present; binning-mode gain/noise and remosaic calibration are not measured.",
            {
                "cfa_pattern": sensor_index_row.get("cfa_pattern"),
                "ocl_mode_guess": sensor_index_row.get("ocl_mode_guess"),
                "pixel_architecture": sensor_index_row.get("pixel_architecture"),
                "binning_remosaic": readout_prior.get("binning_remosaic"),
            },
        ),
        item(
            "Module Coupling",
            "Lens raytrace field CRA map",
            "measured_or_raytrace" if measured_field_map else "missing_measured_or_raytrace",
            "PASS" if measured_field_map else "MISSING",
            "image_sensor_db/camera_module_field_map.csv",
            "TechInsights-derived sensor metadata does not provide camera module lens CRA maps.",
            {"field_map_gate": field_map_validation.get("gate", "MISSING")},
        ),
        item(
            "Module Coupling",
            "Sensor position / tilt / decenter",
            "prior_seed" if module_prior.get("sensor_position_tilt_decenter") else "missing_module_data",
            "CHECK" if module_prior.get("sensor_position_tilt_decenter") else "MISSING",
            prior_source,
            "Zero-centered research module-alignment seed is available; product use needs assembly/raytrace/metrology data."
            if module_prior.get("sensor_position_tilt_decenter")
            else "Module assembly alignment data is not present.",
            module_prior.get("sensor_position_tilt_decenter", ""),
        ),
        item(
            "Module Coupling",
            "Vignetting / shading",
            "prior_seed_plus_partial_response" if module_prior.get("vignetting_shading") else ("partial_response_prior" if runtime_rows else "missing"),
            "CHECK" if runtime_rows or module_prior.get("vignetting_shading") else "MISSING",
            "; ".join(filter(None, [prior_source, "runtime response vs field" if runtime_rows else ""])),
            "Research vignetting/shading seed is available. Product shading requires lens transmission, measured CRA map, and calibration.",
            {"runtime_field_point_count": runtime_summary["field_point_count"]},
        ),
        item(
            "Module Coupling",
            "Wavelength-dependent chief ray / pupil behavior",
            "missing_raytrace",
            "MISSING",
            "",
            "No wavelength-dependent lens raytrace/pupil table is present.",
        ),
    ]

    optical_gate = worst_gate(*[entry["gate"] for entry in items if entry["section"] == "Optical / Color"])
    pixel_gate = worst_gate(*[entry["gate"] for entry in items if entry["section"] == "Pixel / Electrical"])
    readout_gate = worst_gate(*[entry["gate"] for entry in items if entry["section"] == "Readout / RAW"])
    module_gate = worst_gate(*[entry["gate"] for entry in items if entry["section"] == "Module Coupling"])

    model_path = f"models/{slug}.json"
    summary = {
        "slug": slug,
        "code": code,
        "manufacturer": sensor_index_row.get("manufacturer", ""),
        "device_name": sensor_index_row.get("device_name", ""),
        "pixel_pitch_um": sensor_index_row.get("pixel_pitch_um", ""),
        "cfa_pattern": sensor_index_row.get("cfa_pattern", ""),
        "ocl_mode_guess": sensor_index_row.get("ocl_mode_guess", ""),
        "effective_ocl_mode": effective_ocl_mode,
        "optical_readiness": optical_readiness,
        "cfa_proxy_nk_enabled": cfa_proxy_enabled,
        "cfa_provenance_class": cfa_provenance_class,
        "cfa_assumption_gate": cfa_assumption_gate,
        "color_accuracy_gate": color_accuracy_gate,
        "quantitative_field_pass_points": field_cov.get("pass_points", "0"),
        "quantitative_field_required_points": field_cov.get("required_points", "0"),
        "field_runtime_rows": runtime_summary["pass_row_count"],
        "runtime_rows": len(runtime_rows),
        "kernel_rows": len(kernel_rows),
        "field_coverage_fraction": field_cov.get("coverage_fraction", "0"),
        "field_coverage_gate": field_gate,
        "crosstalk_coverage_gate": crosstalk_gate,
        "cra_mismatch_gate": cra_mismatch_summary["gate"],
        "cra_mismatch_profiles": json.dumps(cra_mismatch_summary["tolerance_profile_counts"], sort_keys=True),
        "max_cra_mismatch_total_deg": cra_mismatch_summary["max_total_mismatch_deg"] if cra_mismatch_summary["max_total_mismatch_deg"] is not None else "",
        "color_spectral_rows": len(color_spectral_rows),
        "color_matrix_gate": color_matrix_seed_row.get("gate", ""),
        "optical_color_gate": optical_gate,
        "pixel_electrical_gate": pixel_gate,
        "readout_raw_gate": readout_gate,
        "module_coupling_gate": module_gate,
        "research_ingest_gate": research_gate,
        "production_lut_gate": production_gate,
        "camera_e2e_product_ready": production_gate == "PASS" and optical_gate == "PASS" and pixel_gate == "PASS" and readout_gate == "PASS" and module_gate == "PASS",
        "primary_blockers": readiness_row.get("primary_blockers", ""),
        "model_json": model_path,
    }

    model = {
        "schema": "camera_e2e_sensor_model_v1",
        "artifact_role": "per_sensor_camera_e2e_ingest_manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor": {
            "slug": slug,
            "code": code,
            "manufacturer": sensor_index_row.get("manufacturer"),
            "device_name": sensor_index_row.get("device_name"),
            "pixel_pitch_um": as_float(sensor_index_row.get("pixel_pitch_um")),
            "pixel_architecture": sensor_index_row.get("pixel_architecture"),
            "cfa_pattern": sensor_index_row.get("cfa_pattern"),
            "ocl_mode_guess": sensor_index_row.get("ocl_mode_guess"),
            "effective_ocl_mode": effective_ocl_mode,
            "topology_note": topology_note,
        },
        "camera_e2e_status": {
            "research_ingest_gate": research_gate,
            "production_lut_gate": production_gate,
            "product_ready": summary["camera_e2e_product_ready"],
            "primary_blockers": readiness_row.get("primary_blockers", ""),
            "next_actions": readiness_row.get("next_actions", ""),
        },
        "optical_color": {
            "gate": optical_gate,
            "optical_qe_db_readiness": optical_readiness,
            "stack_geometry_um": stack_config.get("geometry_um", {}),
            "stack_calibration_status": stack_config.get("calibration_status", {}),
            "materials": stack_materials,
            "cfa_proxy_nk": {
                "enabled": cfa_proxy_enabled,
                "library_id": cfa_proxy.get("library_id") or optical_row.get("cfa_proxy_library_id"),
                "source_kind": cfa_proxy.get("source_kind"),
                "applicability": cfa_proxy.get("applicability") or optical_row.get("cfa_proxy_applicability"),
                "thickness_um": cfa_proxy.get("thickness_um") or optical_row.get("cfa_proxy_thickness_um"),
                "channels": cfa_proxy.get("channels", {}),
            },
            "cfa_provenance": {
                "schema": "camera_e2e_cfa_provenance_sensor_row_v1",
                "csv": "camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv",
                "row": cfa_provenance_row,
                "cfa_provenance_class": cfa_provenance_class,
                "cfa_assumption_gate": cfa_assumption_gate,
                "color_accuracy_gate": color_accuracy_gate,
                "policy": "cfa_assumption_gate must be CHECK/PASS for research color trend use and PASS with measured material/spectral data for product color accuracy.",
            },
            "spectral_response_runtime_summary": runtime_summary,
            "spectral_response_color_table": {
                "schema": "camera_e2e_spectral_response_rows_v1",
                "row_count": len(color_spectral_rows),
                "csv": "camera_e2e_color_response/camera_e2e_spectral_response.csv",
                "evidence_gate_counts": compact_counts(color_spectral_rows, "evidence_gate"),
                "rows": color_spectral_rows,
                "limitations": [
                    "Rows are CFA proxy spectral responses scaled by runtime anchors where available.",
                    "They are not measured spectral QE without sensor/lens calibration.",
                ],
            },
            "color_response_matrix": {
                "schema": "camera_e2e_rgb_spectral_sensitivity_rows_v1",
                "rows": color_matrix,
                "rgb_to_xyz_seed": color_matrix_seed_row,
                "rgb_to_xyz_seed_csv": "camera_e2e_color_response/camera_e2e_color_matrix_seed.csv",
                "limitations": [
                    "Rows are generated from available runtime response points.",
                    "They are not a calibrated XYZ/CCM target without measured spectral QE, lens transmission, and color calibration.",
                ],
            },
            "optical_crosstalk": {
                "kernel_row_count": len(kernel_rows),
                "coverage": crosstalk_cov,
                "kernel_csv": "camera_e2e_runtime_bundle/camera_e2e_runtime_crosstalk_kernel.csv",
            },
            "angular_response": {
                "coverage": field_cov,
                "runtime_csv": "camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv",
                "cra_mismatch": cra_mismatch_summary,
            },
            "microlens_ocl_shift_map": {
                "source_gate": field_map_validation.get("gate", "MISSING"),
                "rows": shift_map,
                "limitations": ["Rows are design priors unless camera_module_field_map.csv is measured or raytrace-validated."],
            },
        },
        "pixel_electrical": {
            "gate": pixel_gate,
            "prior_seed": pixel_prior,
            "tcad_profile_available": tcad_available,
            "tcad_profile_calibrated": tcad_calibrated,
            "tcad_profile": tcad_profile,
            "tcad_structure_model": tcad_structure,
            "limitations": [
                "Conversion gain, FWC, saturation, dark current, DSNU, PRNU, and temporal noise are not measured in the current package.",
                "Proxy TCAD geometry is useful for structure review and trend experiments only until calibrated to measured electrical targets.",
            ],
        },
        "readout_raw": {
            "gate": readout_gate,
            "prior_seed": readout_prior,
            "mode_context": readout_context,
            "limitations": [
                "Analog/digital gain tables, black level, ADC, row/column FPN, timing, and defect statistics are missing measured readout data.",
                "Binning/remosaic topology is represented by CFA/OCL/pixel architecture only; noise/gain remap is not calibrated.",
            ],
        },
        "module_coupling": {
            "gate": module_gate,
            "prior_seed": module_prior,
            "field_map_validation": field_map_validation,
            "cra_mismatch": cra_mismatch_summary,
            "module_coupling_lut": {
                "csv": "camera_e2e_module_coupling/camera_e2e_module_coupling_field_lut.csv",
                "summary_csv": "camera_e2e_module_coupling/camera_e2e_module_coupling_summary.csv",
                "html": "camera_e2e_module_coupling/index.html",
            },
            "limitations": [
                "Lens raytrace CRA, pupil, vignetting, and module alignment are module-specific inputs and are not provided by sensor teardown metadata.",
            ],
        },
        "coverage_matrix": items,
        "source_artifacts": {
            "sensor_lut": sensor_index_row.get("lut_json"),
            "stack_config": sensor_lut.get("source_artifacts", {}).get("stack_config"),
            "tcad_profile": sensor_lut.get("source_artifacts", {}).get("tcad_profile"),
            "tcad_structure_model": repo_rel(ROOT / "image_sensor_db" / "tcad_structure_db" / "models" / f"{slug}.json") if tcad_structure else "",
            "prior_seed_model": prior_source,
            "optical_qe_model": repo_rel((DEFAULT_OPTICAL_QE_DB / optical_row.get("model_json", "")).resolve()) if optical_row.get("model_json") else "",
            "runtime_lut_csv": "camera_e2e_runtime_bundle/camera_e2e_runtime_lut.csv",
            "runtime_crosstalk_kernel_csv": "camera_e2e_runtime_bundle/camera_e2e_runtime_crosstalk_kernel.csv",
            "color_response_spectral_csv": "camera_e2e_color_response/camera_e2e_spectral_response.csv",
            "color_response_matrix_seed_csv": "camera_e2e_color_response/camera_e2e_color_matrix_seed.csv",
            "cfa_provenance_csv": "camera_e2e_cfa_provenance/camera_e2e_cfa_provenance_by_sensor.csv",
            "module_coupling_lut_csv": "camera_e2e_module_coupling/camera_e2e_module_coupling_field_lut.csv",
            "readiness_report": "camera_e2e_readiness_audit/camera_e2e_lut_readiness_report.json",
        },
    }
    return model, items, summary


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.check{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
code{color:#9fe8ff}
"""
    gate_counts: dict[str, int] = defaultdict(int)
    for row in matrix_rows:
        gate_counts[row.get("gate", "")] += 1
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Sensor Models</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Sensor Models</h1>
  <p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. This is an ingest manifest, not a measured product LUT sign-off.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric check">{html_cell(gate_counts.get("CHECK", 0))}</div><div class="muted">CHECK items</div></div>
    <div class="card"><div class="metric fail">{html_cell(gate_counts.get("MISSING", 0) + gate_counts.get("FAIL", 0))}</div><div class="muted">missing/fail items</div></div>
    <div class="card"><div class="metric">{html_cell(manifest.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
  </div>
  <h2>Sensor Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS)}
  <h2>Requirement Matrix</h2>
  {html_table(matrix_rows, MATRIX_COLUMNS)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, manifest: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_sensor_models_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_sensor_models_summary_csv"] = manifest["outputs"]["summary_csv"]
    outputs["camera_e2e_sensor_models_matrix_csv"] = manifest["outputs"]["matrix_csv"]
    outputs["camera_e2e_sensor_models_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_sensor_models"] = {
        "schema": manifest["schema"],
        "sensor_count": manifest["sensor_count"],
        "product_ready_count": manifest["product_ready_count"],
        "research_ready_count": manifest["research_ready_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def export_models(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    optical_db = args.optical_qe_db.resolve()
    output_dir = args.output_dir.resolve()
    models_dir = output_dir / "models"

    sensor_index = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    if args.slugs:
        wanted = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
        sensor_index = [row for row in sensor_index if row.get("slug") in wanted]

    readiness = read_json(package_dir / "camera_e2e_readiness_audit" / "camera_e2e_lut_readiness_report.json")
    readiness_by_slug = {row.get("slug", ""): row for row in readiness.get("rows", [])}
    coverage_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_quantitative_coverage.csv"), "slug")
    runtime_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv"), "slug")
    kernel_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_crosstalk_kernel.csv"), "slug")
    color_spectral_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_spectral_response.csv"), "slug")
    color_matrix_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_color_response" / "camera_e2e_color_matrix_seed.csv"), "slug")
    cfa_provenance_by_slug = index_by(read_csv_rows(package_dir / "camera_e2e_cfa_provenance" / "camera_e2e_cfa_provenance_by_sensor.csv"), "slug")
    field_map_validation = read_json(package_dir / "camera_module_field_map_validation.json")
    optical_by_code, optical_models_by_code = load_optical_maps(optical_db)
    prior_dir = package_dir / "camera_e2e_prior_seed_models" / "models"

    summary_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    model_paths: list[str] = []

    for row in sensor_index:
        slug = row.get("slug", "")
        code = row.get("code", "")
        sensor_lut_path = ROOT / row.get("lut_json", "")
        sensor_lut = read_json(sensor_lut_path)
        source_artifacts = sensor_lut.get("source_artifacts", {})
        stack_config = read_json(ROOT / source_artifacts.get("stack_config", ""))
        tcad_profile = read_json(ROOT / source_artifacts.get("tcad_profile", ""))
        tcad_structure = resolve_tcad_structure(slug)
        prior_model = read_json(prior_dir / f"{slug}.json")
        optical_row = optical_by_code.get(code, {})
        optical_model = optical_models_by_code.get(code, {})

        model, items, summary = build_model(
            sensor_index_row=row,
            sensor_lut=sensor_lut,
            optical_row=optical_row,
            optical_model=optical_model,
            stack_config=stack_config,
            tcad_profile=tcad_profile,
            tcad_structure=tcad_structure,
            prior_model=prior_model,
            readiness_row=readiness_by_slug.get(slug, {}),
            coverage_rows=coverage_by_slug.get(slug, []),
            runtime_rows=runtime_by_slug.get(slug, []),
            kernel_rows=kernel_by_slug.get(slug, []),
            color_spectral_rows=color_spectral_by_slug.get(slug, []),
            color_matrix_seed_row=color_matrix_by_slug.get(slug, {}),
            cfa_provenance_row=cfa_provenance_by_slug.get(slug, {}),
            field_map_validation=field_map_validation,
        )
        model_json_path = models_dir / f"{slug}.json"
        write_json(model_json_path, model)
        model_paths.append(repo_rel(model_json_path))
        summary["model_json"] = repo_rel(model_json_path)
        summary_rows.append(summary)
        for entry in items:
            matrix_rows.append(
                {
                    "slug": slug,
                    "code": code,
                    **entry,
                }
            )

    summary_csv = output_dir / "camera_e2e_sensor_model_summary.csv"
    matrix_csv = output_dir / "camera_e2e_sensor_model_matrix.csv"
    manifest_json = output_dir / "camera_e2e_sensor_models.json"
    html_path = output_dir / "index.html"

    manifest = {
        "schema": "camera_e2e_sensor_models_export_v1",
        "artifact_role": "camera_e2e_per_sensor_model_export",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "optical_qe_db": repo_rel(optical_db),
        "sensor_count": len(summary_rows),
        "research_ready_count": sum(1 for row in summary_rows if row.get("research_ingest_gate") in {"PASS", "CHECK"}),
        "product_ready_count": sum(1 for row in summary_rows if boolish(row.get("camera_e2e_product_ready"))),
        "policy": {
            "proxy_data": "allowed for research/trend use only",
            "production_gate": "requires measured/raytrace CRA map, measured stack n,k, calibrated electrical/noise/readout data, and crosstalk convergence",
        },
        "model_json_files": model_paths,
        "outputs": {
            "json": repo_rel(manifest_json),
            "summary_csv": repo_rel(summary_csv),
            "matrix_csv": repo_rel(matrix_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_csv(matrix_csv, matrix_rows, MATRIX_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, summary_rows, matrix_rows)
    update_package(package_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--optical-qe-db", type=Path, default=DEFAULT_OPTICAL_QE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="", help="Optional comma-separated slug filter.")
    return parser


def main() -> None:
    manifest = export_models(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "sensor_count": manifest["sensor_count"],
                "research_ready_count": manifest["research_ready_count"],
                "product_ready_count": manifest["product_ready_count"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
