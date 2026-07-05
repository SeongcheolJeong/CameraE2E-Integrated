#!/usr/bin/env python3
"""Export CameraE2E-ingestable research LUTs from the sensor package.

The exported rows are intentionally provenance-heavy. They make camera-system
experiments possible before full solver coverage exists, while preventing sparse
or prior-derived rows from being mistaken for product LUT data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import export_camera_e2e_material_tables as material_export


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_ingest_export"

FIELD_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "pixel_architecture",
    "cfa_pattern",
    "ocl_mode_guess",
    "field_case",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
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
    "lens_shift_model",
    "cra_measurement_gate",
    "cra_input_gate",
    "cra_source",
    "edge_cra_assumption_deg",
    "focus_target_depth_um",
    "lens_shift_cap_um",
    "wavelength_nm",
    "color_channel",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
    "response_model",
    "evidence_level",
    "evidence_gate",
    "source",
    "product_lut_ready",
]

FIELD_NUMERIC_COLUMNS = [
    "pixel_pitch_um",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
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
    "edge_cra_assumption_deg",
    "focus_target_depth_um",
    "lens_shift_cap_um",
    "wavelength_nm",
    "relative_qe_proxy",
    "relative_qe_min",
    "relative_qe_max",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "focal_centroid_shift_x_um",
    "focal_centroid_shift_z_um",
    "focal_rms_radius_um",
    "focal_target_fraction",
]

FIELD_STRING_COLUMNS = [
    column for column in FIELD_COLUMNS if column not in FIELD_NUMERIC_COLUMNS and column != "product_lut_ready"
]

CROSSTALK_STATUS_COLUMNS = [
    "slug",
    "code",
    "field_case",
    "wavelength_nm",
    "color_channel",
    "neighborhood",
    "simulation_neighborhood",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "resource_gate",
    "convergence_status",
    "evidence_gate",
    "source",
    "product_lut_ready",
]

MANIFEST_COLUMNS = [
    "artifact_id",
    "path",
    "artifact_role",
    "schema",
    "row_count",
    "research_ingest_allowed",
    "production_ingest_allowed",
    "gate",
    "notes",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "field_rows",
    "fdtd_pass_rows",
    "fdtd_fail_rows",
    "prior_rows",
    "tcad_proxy_rows",
    "crosstalk_rows",
    "field_gate",
    "crosstalk_gate",
    "production_ingest_allowed",
    "notes",
]

DEFAULT_WAVELENGTHS = (450.0, 550.0, 620.0)
SPECTRAL_PRIOR = {
    450.0: 0.54,
    550.0: 0.67,
    620.0: 0.61,
}
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_CFA_LIBRARY = ROOT / "cfa_proxy_nk_library.json"
TRUE_CRA_GATES = {"MEASURED", "CALIBRATED", "RAYTRACE_VALIDATED"}


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def cra_input_gate(value: Any, mismatch_gate: Any = "") -> str:
    measurement_pass = str(value or "").strip().upper() in TRUE_CRA_GATES
    mismatch = str(mismatch_gate or "").strip().upper()
    if not measurement_pass:
        return "CHECK"
    if mismatch == "PASS":
        return "PASS"
    if mismatch == "FAIL":
        return "FAIL"
    return "CHECK"


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def write_field_lut_npz(path: Path, rows: list[dict[str, Any]], source_json: Path, source_csv: Path) -> None:
    arrays: dict[str, Any] = {
        "schema": np.array("camera_e2e_sensor_field_response_lut_v1", dtype="U80"),
        "artifact_role": np.array("multi_sensor_relative_field_response_lut", dtype="U96"),
        "product_lut_ready": np.array(False, dtype=np.bool_),
        "row_count": np.array(len(rows), dtype=np.int64),
        "columns": np.array(FIELD_COLUMNS, dtype="U80"),
        "numeric_columns": np.array(FIELD_NUMERIC_COLUMNS, dtype="U80"),
        "string_columns": np.array(FIELD_STRING_COLUMNS, dtype="U80"),
        "source_json": np.array(repo_rel(source_json), dtype="U512"),
        "source_csv": np.array(repo_rel(source_csv), dtype="U512"),
    }
    for column in FIELD_NUMERIC_COLUMNS:
        arrays[column] = np.array([finite_float(row.get(column)) for row in rows], dtype=np.float64)
    for column in FIELD_STRING_COLUMNS:
        arrays[column] = np.array([str(row.get(column, "")) for row in rows], dtype="U1024")
    arrays["row_product_lut_ready"] = np.array([boolish(row.get("product_lut_ready")) for row in rows], dtype=np.bool_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def validate_field_lut_npz(path: Path, rows: list[dict[str, Any]], *, tolerance: float = 1e-10) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != "camera_e2e_sensor_field_response_lut_v1":
            issues.append({"severity": "error", "code": "schema_mismatch", "actual": schema})
        row_count = int(data["row_count"])
        if row_count != len(rows):
            issues.append({"severity": "error", "code": "row_count_mismatch", "npz": row_count, "rows": len(rows)})
        for column in FIELD_NUMERIC_COLUMNS:
            if column not in data:
                issues.append({"severity": "error", "code": "missing_numeric_array", "column": column})
                continue
            values = data[column]
            if len(values) != len(rows):
                issues.append({"severity": "error", "code": "numeric_array_length_mismatch", "column": column})
                continue
            for index, row in enumerate(rows):
                expected = finite_float(row.get(column))
                actual = float(values[index])
                if math.isnan(expected) and math.isnan(actual):
                    continue
                if abs(expected - actual) > max(tolerance, tolerance * max(1.0, abs(expected))):
                    issues.append(
                        {
                            "severity": "error",
                            "code": "numeric_value_mismatch",
                            "column": column,
                            "row_index": index,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                    break
        for column in FIELD_STRING_COLUMNS:
            if column not in data:
                issues.append({"severity": "error", "code": "missing_string_array", "column": column})
                continue
            values = data[column]
            if len(values) != len(rows):
                issues.append({"severity": "error", "code": "string_array_length_mismatch", "column": column})
                continue
            for index, row in enumerate(rows):
                expected = str(row.get(column, ""))
                actual = str(values[index])
                if expected != actual:
                    issues.append(
                        {
                            "severity": "error",
                            "code": "string_value_mismatch",
                            "column": column,
                            "row_index": index,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                    break
        if "row_product_lut_ready" in data and bool(np.any(data["row_product_lut_ready"])):
            issues.append({"severity": "error", "code": "row_product_lut_ready_true"})
    return {
        "schema": "camera_e2e_field_lut_npz_validation_v1",
        "pass": not issues,
        "bad_count": len(issues),
        "row_count": len(rows),
        "numeric_column_count": len(FIELD_NUMERIC_COLUMNS),
        "string_column_count": len(FIELD_STRING_COLUMNS),
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if limit is not None and len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def color_for_wavelength(wavelength_nm: float) -> str:
    if wavelength_nm <= 500:
        return "blue"
    if wavelength_nm >= 590:
        return "red"
    return "green"


def is_monochrome_sensor(sensor: dict[str, str]) -> bool:
    text = " ".join(
        [
            str(sensor.get("cfa_pattern", "")),
            str(sensor.get("sensor_modality", "")),
            str(sensor.get("device_name", "")),
        ]
    ).lower()
    return "mono" in text or "clear" in text


def color_for_sensor_wavelength(sensor: dict[str, str], wavelength_nm: float) -> str:
    if is_monochrome_sensor(sensor):
        return "clear"
    return color_for_wavelength(wavelength_nm)


def parse_wavelengths(text: str) -> list[float]:
    values = []
    for item in str(text or "").replace(";", ",").split(","):
        value = finite_float(item)
        if math.isfinite(value):
            values.append(value)
    return values or list(DEFAULT_WAVELENGTHS)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def nearest_row(rows: list[dict[str, Any]], wavelength_nm: float, *, color_channel: str = "") -> dict[str, Any]:
    candidates = rows
    if color_channel:
        same_color = [row for row in rows if str(row.get("color_channel", "")) == color_channel]
        if same_color:
            candidates = same_color
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs(finite_float(row.get("wavelength_nm"), wavelength_nm) - wavelength_nm))


def simple_si_absorption(si_k: float, wavelength_nm: float, thickness_um: float) -> float:
    if not (math.isfinite(si_k) and math.isfinite(wavelength_nm) and math.isfinite(thickness_um)):
        return math.nan
    if wavelength_nm <= 0 or thickness_um <= 0:
        return math.nan
    alpha_per_um = 4.0 * math.pi * max(0.0, si_k) / (wavelength_nm / 1000.0)
    return max(0.0, min(1.0, 1.0 - math.exp(-alpha_per_um * thickness_um)))


def cfa_si_fraction(material_rows: list[dict[str, Any]], wavelength_nm: float, color_channel: str) -> float:
    cfa_rows = [row for row in material_rows if row.get("material_family") == "cfa_transmission_proxy"]
    si_rows = [row for row in material_rows if row.get("material_family") == "silicon_fdtd_material"]
    cfa = nearest_row(cfa_rows, wavelength_nm, color_channel=color_channel)
    si = nearest_row(si_rows, wavelength_nm)
    cfa_t = 1.0 if color_channel == "clear" and not cfa else finite_float(cfa.get("transmission_absorption_only"))
    si_abs = simple_si_absorption(
        finite_float(si.get("k")),
        finite_float(si.get("wavelength_nm"), wavelength_nm),
        finite_float(si.get("thickness_um")),
    )
    if math.isfinite(cfa_t) and math.isfinite(si_abs):
        return max(0.0, cfa_t * si_abs)
    return math.nan


def material_spectral_shape(
    material_rows: list[dict[str, Any]],
    wavelength_nm: float,
    color_channel: str,
    *,
    reference_channel: str,
) -> tuple[float, str]:
    current = cfa_si_fraction(material_rows, wavelength_nm, color_channel)
    reference = cfa_si_fraction(material_rows, 550.0, reference_channel)
    if math.isfinite(current) and math.isfinite(reference) and reference > 1e-12:
        return max(0.0, current / reference), "CFA transmission proxy x simple Si absorption normalized to 550nm"
    spectral = SPECTRAL_PRIOR.get(wavelength_nm, SPECTRAL_PRIOR[550.0]) / SPECTRAL_PRIOR[550.0]
    return max(0.0, spectral), "default spectral prior normalized to 550nm"


def build_material_response_context(sensor_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    cfa_library = read_json_optional(DEFAULT_CFA_LIBRARY)
    context: dict[str, list[dict[str, Any]]] = {}
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        if not slug:
            continue
        optical_model = read_json_optional(DEFAULT_OPTICAL_QE_DB / "models" / f"{slug}.json")
        stack_path = DEFAULT_STACK_DIR / f"{slug}.json"
        stack = read_json_optional(stack_path)
        rows: list[dict[str, Any]] = []
        rows.extend(material_export.rows_from_cfa_proxy(sensor, optical_model, cfa_library))
        materials = stack.get("materials", {}) if isinstance(stack.get("materials"), dict) else {}
        for key in ("silicon", "lens", "passivation"):
            material = materials.get(key, {})
            if isinstance(material, dict):
                rows.extend(
                    material_export.rows_from_stack_material(
                        sensor=sensor,
                        stack_path=stack_path,
                        stack=stack,
                        material_key=key,
                        material=material,
                    )
                )
        context[slug] = rows
    return context


def field_direction(field_x: float, field_z: float) -> tuple[float, float]:
    radius = math.hypot(field_x, field_z)
    azimuth = math.degrees(math.atan2(field_z, field_x)) if radius > 1e-12 else 0.0
    return radius, azimuth


def group_by(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        output.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    return output


def scalar_by_x(points: list[tuple[float, float]], x_value: float, default: float) -> float:
    clean = sorted((x, y) for x, y in points if math.isfinite(x) and math.isfinite(y))
    if not clean:
        return default
    if x_value <= clean[0][0]:
        return clean[0][1]
    if x_value >= clean[-1][0]:
        return clean[-1][1]
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        if x0 <= x_value <= x1 and abs(x1 - x0) > 1e-12:
            t = (x_value - x0) / (x1 - x0)
            return y0 * (1.0 - t) + y1 * t
    return clean[-1][1]


def response_bounds(nominal: float, evidence_level: str) -> tuple[float, float]:
    spreads = {
        "fdtd_quantitative_pass": 0.10,
        "fdtd_quantitative_fail_numeric_reference": 0.35,
        "tcad_lateral_proxy_scaled": 0.45,
        "material_cfa_si_sanity_scaled": 0.50,
        "design_prior_spectral_rolloff": 0.60,
    }
    spread = spreads.get(evidence_level, 0.75)
    return max(0.0, nominal * (1.0 - spread)), min(1.0, nominal * (1.0 + spread))


def build_field_rows(
    *,
    sensor_rows: list[dict[str, str]],
    field_design_rows: list[dict[str, str]],
    quantitative_field_rows: list[dict[str, str]],
    proxy_rows: list[dict[str, str]],
    material_response_context: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    sensors = {row["slug"]: row for row in sensor_rows}
    q_by_point = {
        (
            row.get("slug", ""),
            row.get("field_case", ""),
            f"{finite_float(row.get('wavelength_nm')):g}",
            row.get("color", ""),
        ): row
        for row in quantitative_field_rows
        if row.get("slug")
    }
    pass_center_by_slug: dict[str, float] = {}
    for row in quantitative_field_rows:
        if row.get("field_case") == "center" and row.get("solver_gate") == "PASS":
            value = finite_float(row.get("total_response"))
            if math.isfinite(value) and value >= 0:
                pass_center_by_slug[row.get("slug", "")] = value

    proxy_by_slug = group_by(proxy_rows, "slug")
    design_by_slug_case = {(row.get("slug", ""), row.get("field_case", "")): row for row in field_design_rows}
    first_design_by_slug: dict[str, dict[str, str]] = {}
    for row in field_design_rows:
        first_design_by_slug.setdefault(row.get("slug", ""), row)
    output: list[dict[str, Any]] = []
    emitted_keys: set[tuple[str, str, str, str]] = set()
    for design in field_design_rows:
        slug = design.get("slug", "")
        sensor = sensors.get(slug, {})
        field_x = finite_float(design.get("field_x_norm"), 0.0)
        field_z = finite_float(design.get("field_z_norm"), 0.0)
        radius, azimuth = field_direction(field_x, field_z)
        wavelengths = parse_wavelengths(design.get("wavelength_set_nm", ""))
        proxy_points = []
        split_points = []
        for row in proxy_by_slug.get((slug,), []):
            proxy_x = finite_float(row.get("field_x_norm"))
            rel = finite_float(row.get("relative_response_to_center"))
            split = finite_float(row.get("split_phase_x_proxy"))
            if math.isfinite(proxy_x) and math.isfinite(rel):
                proxy_points.append((proxy_x, max(0.0, rel)))
            if math.isfinite(proxy_x) and math.isfinite(split):
                split_points.append((proxy_x, max(-1.0, min(1.0, split))))

        for wavelength in wavelengths:
            color = color_for_sensor_wavelength(sensor, wavelength)
            key = (slug, design.get("field_case", ""), f"{wavelength:g}", color)
            qrow = q_by_point.get(key)
            response_model = ""
            evidence_level = ""
            source = ""
            solver_gate = ""
            focal_shift_x = ""
            focal_shift_z = ""
            focal_rms = ""
            focal_fraction = ""
            nominal = math.nan
            split_x = scalar_by_x(split_points, field_x, 0.0)
            split_z = 0.0
            if qrow is not None:
                nominal = finite_float(qrow.get("total_response"))
                solver_gate = qrow.get("solver_gate", "")
                focal_shift_x = qrow.get("focal_centroid_shift_x_um", "")
                focal_shift_z = qrow.get("focal_centroid_shift_z_um", "")
                focal_rms = qrow.get("focal_rms_radius_um", "")
                focal_fraction = qrow.get("focal_target_fraction", "")
                if solver_gate == "PASS" and math.isfinite(nominal):
                    evidence_level = "fdtd_quantitative_pass"
                    response_model = "direct quantitative Meep FDTD point"
                elif math.isfinite(nominal):
                    evidence_level = "fdtd_quantitative_fail_numeric_reference"
                    response_model = "failed quantitative point retained as non-production numeric reference"
                source = qrow.get("source_summary_csv", "")

            if not math.isfinite(nominal):
                center = pass_center_by_slug.get(slug, SPECTRAL_PRIOR[550.0])
                material_shape, material_shape_basis = material_spectral_shape(
                    material_response_context.get(slug, []),
                    wavelength,
                    color,
                    reference_channel="clear" if color == "clear" else "green",
                )
                rolloff = scalar_by_x(proxy_points, field_x, max(0.25, 1.0 - 0.18 * min(radius, 1.414) ** 2))
                nominal = max(0.0, min(1.0, center * material_shape * rolloff))
                if proxy_points:
                    evidence_level = "tcad_lateral_proxy_scaled"
                    response_model = f"{material_shape_basis} scaled by DEVSIM lateral-generation proxy"
                    source = "camera_e2e_field_response_proxy.csv"
                    solver_gate = "CHECK"
                elif material_shape_basis.startswith("CFA"):
                    evidence_level = "material_cfa_si_sanity_scaled"
                    response_model = f"{material_shape_basis} with radius rolloff"
                    source = "image_sensor_db CFA/stack proxy material tables"
                    solver_gate = "CHECK"
                else:
                    evidence_level = "design_prior_spectral_rolloff"
                    response_model = "default spectral response prior with radius rolloff"
                    source = "generated design prior"
                    solver_gate = "MISSING"

            lower, upper = response_bounds(nominal, evidence_level)
            emitted_keys.add(key)
            output.append(
                {
                    "slug": slug,
                    "code": design.get("code", ""),
                    "manufacturer": sensor.get("manufacturer", ""),
                    "device_name": sensor.get("device_name", ""),
                    "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
                    "pixel_architecture": sensor.get("pixel_architecture", ""),
                    "cfa_pattern": sensor.get("cfa_pattern", ""),
                    "ocl_mode_guess": sensor.get("ocl_mode_guess", ""),
                    "field_case": design.get("field_case", ""),
                    "field_x_norm": field_x,
                    "field_z_norm": field_z,
                    "field_radius_norm": radius,
                    "field_azimuth_deg": azimuth,
                    "cra_x_deg": finite_float(design.get("cra_x_deg"), 0.0),
                    "cra_z_deg": finite_float(design.get("cra_z_deg"), 0.0),
                    "lens_cra_x_deg": finite_float(design.get("lens_cra_x_deg")),
                    "lens_cra_z_deg": finite_float(design.get("lens_cra_z_deg")),
                    "sensor_cra_x_deg": finite_float(design.get("sensor_cra_x_deg")),
                    "sensor_cra_z_deg": finite_float(design.get("sensor_cra_z_deg")),
                    "cra_mismatch_x_deg": finite_float(design.get("cra_mismatch_x_deg")),
                    "cra_mismatch_z_deg": finite_float(design.get("cra_mismatch_z_deg")),
                    "cra_mismatch_total_deg": finite_float(design.get("cra_mismatch_total_deg")),
                    "cra_mismatch_tolerance_profile": design.get("cra_mismatch_tolerance_profile", ""),
                    "cra_mismatch_pass_tolerance_deg": finite_float(design.get("cra_mismatch_pass_tolerance_deg")),
                    "cra_mismatch_check_tolerance_deg": finite_float(design.get("cra_mismatch_check_tolerance_deg")),
                    "cra_mismatch_gate": design.get("cra_mismatch_gate", ""),
                    "lens_shift_x_um": finite_float(design.get("lens_shift_x_um"), 0.0),
                    "lens_shift_z_um": finite_float(design.get("lens_shift_z_um"), 0.0),
                    "lens_shift_model": design.get("lens_shift_model", ""),
                    "cra_measurement_gate": design.get("measurement_gate", ""),
                    "cra_input_gate": cra_input_gate(design.get("measurement_gate", ""), design.get("cra_mismatch_gate", "")),
                    "cra_source": design.get("source", ""),
                    "edge_cra_assumption_deg": design.get("edge_cra_assumption_deg", ""),
                    "focus_target_depth_um": design.get("focus_target_depth_um", ""),
                    "lens_shift_cap_um": design.get("lens_shift_cap_um", ""),
                    "wavelength_nm": wavelength,
                    "color_channel": color,
                    "relative_qe_proxy": nominal,
                    "relative_qe_min": lower,
                    "relative_qe_max": upper,
                    "split_phase_x_proxy": split_x,
                    "split_phase_z_proxy": split_z,
                    "focal_centroid_shift_x_um": focal_shift_x,
                    "focal_centroid_shift_z_um": focal_shift_z,
                    "focal_rms_radius_um": focal_rms,
                    "focal_target_fraction": focal_fraction,
                    "response_model": response_model,
                    "evidence_level": evidence_level,
                    "evidence_gate": solver_gate if solver_gate in {"PASS", "FAIL"} else "CHECK",
                    "source": source,
                    "product_lut_ready": False,
                }
            )

    for qrow in quantitative_field_rows:
        slug = qrow.get("slug", "")
        field_case = qrow.get("field_case", "")
        wavelength = finite_float(qrow.get("wavelength_nm"))
        color = qrow.get("color", "")
        if not slug or not field_case or not math.isfinite(wavelength) or not color:
            continue
        key = (slug, field_case, f"{wavelength:g}", color)
        if key in emitted_keys:
            continue
        sensor = sensors.get(slug, {})
        design = design_by_slug_case.get((slug, field_case), first_design_by_slug.get(slug, {}))
        field_x = finite_float(qrow.get("field_x_norm"), finite_float(design.get("field_x_norm"), 0.0))
        field_z = finite_float(qrow.get("field_z_norm"), finite_float(design.get("field_z_norm"), 0.0))
        radius, azimuth = field_direction(field_x, field_z)
        nominal = finite_float(qrow.get("total_response"))
        if not math.isfinite(nominal):
            continue
        solver_gate = qrow.get("solver_gate", "")
        if solver_gate == "PASS":
            evidence_level = "fdtd_quantitative_pass"
            response_model = "direct quantitative Meep FDTD supplementary color/wavelength point"
        else:
            evidence_level = "fdtd_quantitative_fail_numeric_reference"
            response_model = "supplementary failed quantitative point retained as non-production numeric reference"
        lower, upper = response_bounds(nominal, evidence_level)
        output.append(
            {
                "slug": slug,
                "code": qrow.get("code") or design.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
                "pixel_architecture": sensor.get("pixel_architecture", ""),
                "cfa_pattern": sensor.get("cfa_pattern", ""),
                "ocl_mode_guess": sensor.get("ocl_mode_guess", ""),
                "field_case": field_case,
                "field_x_norm": field_x,
                "field_z_norm": field_z,
                "field_radius_norm": radius,
                "field_azimuth_deg": azimuth,
                "cra_x_deg": finite_float(qrow.get("cra_x_deg"), finite_float(design.get("cra_x_deg"), 0.0)),
                "cra_z_deg": finite_float(qrow.get("cra_z_deg"), finite_float(design.get("cra_z_deg"), 0.0)),
                "lens_cra_x_deg": finite_float(design.get("lens_cra_x_deg")),
                "lens_cra_z_deg": finite_float(design.get("lens_cra_z_deg")),
                "sensor_cra_x_deg": finite_float(design.get("sensor_cra_x_deg")),
                "sensor_cra_z_deg": finite_float(design.get("sensor_cra_z_deg")),
                "cra_mismatch_x_deg": finite_float(design.get("cra_mismatch_x_deg")),
                "cra_mismatch_z_deg": finite_float(design.get("cra_mismatch_z_deg")),
                "cra_mismatch_total_deg": finite_float(design.get("cra_mismatch_total_deg")),
                "cra_mismatch_tolerance_profile": design.get("cra_mismatch_tolerance_profile", ""),
                "cra_mismatch_pass_tolerance_deg": finite_float(design.get("cra_mismatch_pass_tolerance_deg")),
                "cra_mismatch_check_tolerance_deg": finite_float(design.get("cra_mismatch_check_tolerance_deg")),
                "cra_mismatch_gate": design.get("cra_mismatch_gate", ""),
                "lens_shift_x_um": finite_float(
                    qrow.get("lens_shift_x_um"),
                    finite_float(design.get("lens_shift_x_um"), 0.0),
                ),
                "lens_shift_z_um": finite_float(
                    qrow.get("lens_shift_z_um"),
                    finite_float(design.get("lens_shift_z_um"), 0.0),
                ),
                "lens_shift_model": design.get("lens_shift_model", "quantitative FDTD row geometry"),
                "cra_measurement_gate": design.get("measurement_gate", ""),
                "cra_input_gate": cra_input_gate(design.get("measurement_gate", ""), design.get("cra_mismatch_gate", "")),
                "cra_source": design.get("source", ""),
                "edge_cra_assumption_deg": design.get("edge_cra_assumption_deg", ""),
                "focus_target_depth_um": design.get("focus_target_depth_um", ""),
                "lens_shift_cap_um": design.get("lens_shift_cap_um", ""),
                "wavelength_nm": wavelength,
                "color_channel": color,
                "relative_qe_proxy": nominal,
                "relative_qe_min": lower,
                "relative_qe_max": upper,
                "split_phase_x_proxy": finite_float(qrow.get("split_phase_x_proxy"), 0.0),
                "split_phase_z_proxy": finite_float(qrow.get("split_phase_z_proxy"), 0.0),
                "focal_centroid_shift_x_um": qrow.get("focal_centroid_shift_x_um", ""),
                "focal_centroid_shift_z_um": qrow.get("focal_centroid_shift_z_um", ""),
                "focal_rms_radius_um": qrow.get("focal_rms_radius_um", ""),
                "focal_target_fraction": qrow.get("focal_target_fraction", ""),
                "response_model": response_model,
                "evidence_level": evidence_level,
                "evidence_gate": solver_gate if solver_gate in {"PASS", "FAIL"} else "CHECK",
                "source": qrow.get("source_summary_csv", ""),
                "product_lut_ready": False,
            }
        )
    return output


def build_crosstalk_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        gate = row.get("solver_gate") or row.get("resource_gate") or "MISSING"
        output.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "field_case": row.get("field_case", ""),
                "wavelength_nm": row.get("wavelength_nm", ""),
                "color_channel": row.get("color", ""),
                "neighborhood": row.get("neighborhood", ""),
                "simulation_neighborhood": row.get("simulation_neighborhood", ""),
                "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                "strongest_neighbor_fraction": row.get("strongest_neighbor_fraction", ""),
                "truncation_response_fraction": row.get("truncation_response_fraction", ""),
                "resource_gate": row.get("resource_gate", ""),
                "convergence_status": row.get("convergence_status", ""),
                "evidence_gate": gate,
                "source": row.get("source_summary_csv", ""),
                "product_lut_ready": False,
            }
        )
    return output


def sensor_summary(
    sensor_rows: list[dict[str, str]],
    field_rows: list[dict[str, Any]],
    crosstalk_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_slug: dict[str, list[dict[str, Any]]] = {}
    by_slug_xt: dict[str, list[dict[str, Any]]] = {}
    for row in field_rows:
        by_slug.setdefault(row["slug"], []).append(row)
    for row in crosstalk_rows:
        by_slug_xt.setdefault(row["slug"], []).append(row)

    output = []
    for sensor in sensor_rows:
        slug = sensor["slug"]
        rows = by_slug.get(slug, [])
        xt = by_slug_xt.get(slug, [])
        fdtd_pass = sum(1 for row in rows if row.get("evidence_level") == "fdtd_quantitative_pass")
        fdtd_fail = sum(1 for row in rows if row.get("evidence_level") == "fdtd_quantitative_fail_numeric_reference")
        tcad_proxy = sum(1 for row in rows if row.get("evidence_level") == "tcad_lateral_proxy_scaled")
        prior = sum(1 for row in rows if row.get("evidence_level") == "design_prior_spectral_rolloff")
        if fdtd_fail:
            field_gate = "FAIL"
        elif fdtd_pass and fdtd_pass == len(rows):
            field_gate = "PASS"
        else:
            field_gate = "CHECK"
        crosstalk_gate = "MISSING"
        if xt:
            gates = {str(row.get("evidence_gate", "")).upper() for row in xt}
            if "FAIL" in gates:
                crosstalk_gate = "FAIL"
            elif gates == {"PASS"}:
                crosstalk_gate = "PASS"
            else:
                crosstalk_gate = "CHECK"
        output.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "field_rows": len(rows),
                "fdtd_pass_rows": fdtd_pass,
                "fdtd_fail_rows": fdtd_fail,
                "prior_rows": prior,
                "tcad_proxy_rows": tcad_proxy,
                "crosstalk_rows": len(xt),
                "field_gate": field_gate,
                "crosstalk_gate": crosstalk_gate,
                "production_ingest_allowed": False,
                "notes": "Research/trend ingest only. Product LUT requires measured CRA/ML shift, measured stack n,k, full FDTD field coverage, and crosstalk convergence.",
            }
        )
    return output


def write_html_report(
    output_dir: Path,
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    crosstalk_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> None:
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Ingest Export</title>
  <style>
    body {{ margin: 0; background: #081118; color: #e5f3ff; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 10px; }}
    h2 {{ color: #52e1ff; margin-top: 26px; }}
    p, .muted {{ color: #99b2c4; }}
    .note {{ border-left: 3px solid #ffd85a; padding-left: 12px; color: #e5f3ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #244255; background: #0e1b25; border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 26px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }}
    th, td {{ border: 1px solid #244255; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #52e1ff; background: #102633; }}
    code {{ color: #d8f8ff; }}
  </style>
</head>
<body>
<main>
  <h1>CameraE2E Ingest Export</h1>
  <p>Generated: <code>{html_cell(payload.get("generated_at", ""))}</code></p>
  <p class="note">This is a research/trend ingest package. It deliberately keeps <code>production_ingest_allowed=false</code> until measured CRA/ML shift, measured stack/material, full quantitative FDTD coverage, and crosstalk convergence pass.</p>
  <p class="muted">NPZ validation pass: <code>{html_cell(payload.get("field_lut_npz_validation", {}).get("pass", ""))}</code>; bad count: <code>{html_cell(payload.get("field_lut_npz_validation", {}).get("bad_count", ""))}</code>.</p>
  <div class="grid">
    <div class="card"><div class="metric">{payload.get("sensor_count", 0)}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{payload.get("field_row_count", 0)}</div><div class="muted">field rows</div></div>
    <div class="card"><div class="metric">{payload.get("fdtd_pass_field_row_count", 0)}</div><div class="muted">direct FDTD PASS rows</div></div>
    <div class="card"><div class="metric">{payload.get("production_ready_sensor_count", 0)}</div><div class="muted">production-ready sensors</div></div>
  </div>
  <h2>Manifest</h2>
  {html_table(manifest_rows, MANIFEST_COLUMNS)}
  <h2>Sensor Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS)}
  <h2>Field Response LUT Preview</h2>
  {html_table(field_rows, FIELD_COLUMNS, limit=24)}
  <h2>Crosstalk Status Preview</h2>
  {html_table(crosstalk_rows, CROSSTALK_STATUS_COLUMNS, limit=24) if crosstalk_rows else "<p>No crosstalk kernel rows are complete enough for ingest; see the batch plan.</p>"}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensor_rows = read_csv(package_dir / "camera_e2e_sensor_index.csv")
    field_design_rows = read_csv(package_dir / "camera_e2e_field_design_cases.csv")
    quantitative_field_rows = read_csv(package_dir / "camera_e2e_quantitative_field_lut.csv")
    quantitative_crosstalk_rows = read_csv(package_dir / "camera_e2e_quantitative_crosstalk_lut.csv")
    proxy_rows = read_csv(package_dir / "camera_e2e_field_response_proxy.csv")

    if args.slugs:
        requested = {item.strip() for item in args.slugs.split(",") if item.strip()}
        sensor_rows = [row for row in sensor_rows if row.get("slug") in requested]
        field_design_rows = [row for row in field_design_rows if row.get("slug") in requested]
        quantitative_field_rows = [row for row in quantitative_field_rows if row.get("slug") in requested]
        quantitative_crosstalk_rows = [row for row in quantitative_crosstalk_rows if row.get("slug") in requested]
        proxy_rows = [row for row in proxy_rows if row.get("slug") in requested]

    material_response_context = build_material_response_context(sensor_rows)
    field_rows = build_field_rows(
        sensor_rows=sensor_rows,
        field_design_rows=field_design_rows,
        quantitative_field_rows=quantitative_field_rows,
        proxy_rows=proxy_rows,
        material_response_context=material_response_context,
    )
    crosstalk_rows = build_crosstalk_rows(quantitative_crosstalk_rows)
    summary_rows = sensor_summary(sensor_rows, field_rows, crosstalk_rows)

    field_csv = output_dir / "camera_e2e_field_response_lut.csv"
    field_json = output_dir / "camera_e2e_field_response_lut.json"
    field_npz = output_dir / "camera_e2e_field_response_lut.npz"
    crosstalk_csv = output_dir / "camera_e2e_crosstalk_status_lut.csv"
    summary_csv = output_dir / "camera_e2e_ingest_sensor_summary.csv"
    manifest_csv = output_dir / "camera_e2e_ingest_manifest.csv"
    manifest_json = output_dir / "camera_e2e_ingest_manifest.json"
    html_report = output_dir / "index.html"
    compact_xt_dir = package_dir / "camera_e2e_compact_crosstalk_lut"
    compact_xt_report = compact_xt_dir / "camera_e2e_compact_crosstalk_report.json"
    compact_xt_payload = read_json(compact_xt_report) if compact_xt_report.exists() else {}
    compact_xt_kernel_csv = compact_xt_dir / "camera_e2e_compact_crosstalk_kernel_lut.csv"
    compact_xt_kernel_npz = compact_xt_dir / "camera_e2e_compact_crosstalk_kernel_lut.npz"
    compact_xt_summary_csv = compact_xt_dir / "camera_e2e_compact_crosstalk_summary.csv"
    compact_xt_html = compact_xt_dir / "index.html"

    field_payload = {
        "schema": "camera_e2e_sensor_field_response_lut_v1",
        "artifact_role": "multi_sensor_relative_field_response_lut",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product_lut_ready": False,
        "response_unit": "relative_qe_proxy_or_fdtd_absorption_fraction",
        "field_row_count": len(field_rows),
        "accuracy_blockers": [
            "TechInsights-derived DB has no measured camera-module CRA/ML-shift map.",
            "Most rows are TCAD/proxy/prior rows, not quantitative Meep FDTD PASS rows.",
            "Measured stack geometry and measured n,k are not present.",
            "Crosstalk convergence is incomplete or resource-limited.",
        ],
        "rows": field_rows,
    }
    write_csv(field_csv, field_rows, FIELD_COLUMNS)
    write_json(field_json, field_payload)
    write_field_lut_npz(field_npz, field_rows, field_json, field_csv)
    npz_validation = validate_field_lut_npz(field_npz, field_rows)
    field_payload["npz_validation"] = npz_validation
    field_payload["outputs"] = {
        "csv": repo_rel(field_csv),
        "json": repo_rel(field_json),
        "npz": repo_rel(field_npz),
    }
    write_json(field_json, field_payload)
    write_csv(crosstalk_csv, crosstalk_rows, CROSSTALK_STATUS_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)

    manifest_rows = [
        {
            "artifact_id": "field_response_lut",
            "path": repo_rel(field_csv),
            "artifact_role": "camera_e2e_relative_field_response_lut",
            "schema": "camera_e2e_sensor_field_response_lut_v1",
            "row_count": len(field_rows),
            "research_ingest_allowed": True,
            "production_ingest_allowed": False,
            "gate": "CHECK",
            "notes": "Rows include response evidence and separate CRA/ML-shift provenance; use fdtd_quantitative_pass rows only with measured CRA inputs for production.",
        },
        {
            "artifact_id": "field_response_lut_json",
            "path": repo_rel(field_json),
            "artifact_role": "camera_e2e_relative_field_response_lut_json",
            "schema": "camera_e2e_sensor_field_response_lut_v1",
            "row_count": len(field_rows),
            "research_ingest_allowed": True,
            "production_ingest_allowed": False,
            "gate": "CHECK",
            "notes": "JSON version of the field response LUT with row-level provenance.",
        },
        {
            "artifact_id": "field_response_lut_npz",
            "path": repo_rel(field_npz),
            "artifact_role": "camera_e2e_relative_field_response_lut_npz",
            "schema": "camera_e2e_sensor_field_response_lut_v1_npz",
            "row_count": len(field_rows),
            "research_ingest_allowed": True,
            "production_ingest_allowed": False,
            "gate": "CHECK" if npz_validation["pass"] else "FAIL",
            "notes": "Compressed typed-array export for high-throughput CameraE2E ingestion; validated against JSON/CSV rows.",
        },
        {
            "artifact_id": "crosstalk_status_lut",
            "path": repo_rel(crosstalk_csv),
            "artifact_role": "camera_e2e_crosstalk_status_lut",
            "schema": "camera_e2e_crosstalk_status_lut_v1",
            "row_count": len(crosstalk_rows),
            "research_ingest_allowed": bool(crosstalk_rows),
            "production_ingest_allowed": False,
            "gate": "CHECK" if crosstalk_rows else "MISSING",
            "notes": "This is status/evidence only unless output_crosstalk_fraction is populated from a converged run.",
        },
        {
            "artifact_id": "compact_crosstalk_kernel_lut",
            "path": repo_rel(compact_xt_kernel_csv) if compact_xt_kernel_csv.exists() else "",
            "artifact_role": "camera_e2e_compact_crosstalk_kernel_lut",
            "schema": "camera_e2e_compact_crosstalk_lut_v1",
            "row_count": compact_xt_payload.get("kernel_row_count", 0) if compact_xt_payload else 0,
            "research_ingest_allowed": bool(compact_xt_payload),
            "production_ingest_allowed": False,
            "gate": "CHECK" if compact_xt_payload else "MISSING",
            "notes": "CHECK-gated Gaussian/DTI compact crosstalk surrogate for CameraE2E trend runs; not finite-array FDTD product evidence.",
        },
        {
            "artifact_id": "compact_crosstalk_kernel_lut_npz",
            "path": repo_rel(compact_xt_kernel_npz) if compact_xt_kernel_npz.exists() else "",
            "artifact_role": "camera_e2e_compact_crosstalk_kernel_lut_npz",
            "schema": "camera_e2e_compact_crosstalk_lut_v1_npz",
            "row_count": compact_xt_payload.get("kernel_row_count", 0) if compact_xt_payload else 0,
            "research_ingest_allowed": bool(compact_xt_payload),
            "production_ingest_allowed": False,
            "gate": "CHECK" if compact_xt_payload.get("npz_validation", {}).get("pass") else "FAIL" if compact_xt_payload else "MISSING",
            "notes": "Compressed typed-array compact crosstalk export; validated against compact kernel CSV rows.",
        },
        {
            "artifact_id": "sensor_summary",
            "path": repo_rel(summary_csv),
            "artifact_role": "sensor_ingest_readiness_summary",
            "schema": "camera_e2e_ingest_sensor_summary_csv_v1",
            "row_count": len(summary_rows),
            "research_ingest_allowed": True,
            "production_ingest_allowed": False,
            "gate": "CHECK",
            "notes": "Per-sensor evidence counts and gates.",
        },
        {
            "artifact_id": "html_report",
            "path": repo_rel(html_report),
            "artifact_role": "human_review_report",
            "schema": "html",
            "row_count": "",
            "research_ingest_allowed": True,
            "production_ingest_allowed": False,
            "gate": "CHECK",
            "notes": "Review report for designers and CameraE2E integrators.",
        },
    ]
    write_csv(manifest_csv, manifest_rows, MANIFEST_COLUMNS)

    payload = {
        "schema": "camera_e2e_ingest_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_package_dir": repo_rel(package_dir),
        "product_lut_ready": False,
        "sensor_count": len(sensor_rows),
        "production_ready_sensor_count": 0,
        "field_row_count": len(field_rows),
        "fdtd_pass_field_row_count": sum(1 for row in field_rows if row.get("evidence_level") == "fdtd_quantitative_pass"),
        "crosstalk_row_count": len(crosstalk_rows),
        "compact_crosstalk_kernel_row_count": compact_xt_payload.get("kernel_row_count", 0) if compact_xt_payload else 0,
        "outputs": {
            "manifest_json": repo_rel(manifest_json),
            "manifest_csv": repo_rel(manifest_csv),
            "html_report": repo_rel(html_report),
            "field_lut_csv": repo_rel(field_csv),
            "field_lut_json": repo_rel(field_json),
            "field_lut_npz": repo_rel(field_npz),
            "crosstalk_status_csv": repo_rel(crosstalk_csv),
            "compact_crosstalk_kernel_csv": repo_rel(compact_xt_kernel_csv) if compact_xt_kernel_csv.exists() else "",
            "compact_crosstalk_kernel_npz": repo_rel(compact_xt_kernel_npz) if compact_xt_kernel_npz.exists() else "",
            "compact_crosstalk_summary_csv": repo_rel(compact_xt_summary_csv) if compact_xt_summary_csv.exists() else "",
            "compact_crosstalk_html": repo_rel(compact_xt_html) if compact_xt_html.exists() else "",
            "sensor_summary_csv": repo_rel(summary_csv),
        },
        "field_lut_npz_validation": npz_validation,
        "accuracy_blockers": [
            "Production CameraE2E ingest is disabled because measured CRA/ML shift, measured stack n,k, complete FDTD field coverage, and converged crosstalk are not all available.",
            "Rows with evidence_level other than fdtd_quantitative_pass are trend/sensitivity priors.",
        ],
        "rows": manifest_rows,
    }
    write_json(manifest_json, payload)
    write_html_report(output_dir, payload, summary_rows, field_rows, crosstalk_rows, manifest_rows)
    package_json = package_dir / "camera_e2e_lut_package.json"
    if package_json.exists():
        package = json.loads(package_json.read_text(encoding="utf-8"))
        outputs = package.setdefault("outputs", {})
        outputs["camera_e2e_ingest_manifest_json"] = repo_rel(manifest_json)
        outputs["camera_e2e_ingest_manifest_csv"] = repo_rel(manifest_csv)
        outputs["camera_e2e_ingest_html"] = repo_rel(html_report)
        outputs["camera_e2e_ingest_field_lut_csv"] = repo_rel(field_csv)
        outputs["camera_e2e_ingest_field_lut_json"] = repo_rel(field_json)
        outputs["camera_e2e_ingest_field_lut_npz"] = repo_rel(field_npz)
        outputs["camera_e2e_ingest_crosstalk_status_csv"] = repo_rel(crosstalk_csv)
        if compact_xt_payload:
            outputs["camera_e2e_compact_crosstalk_kernel_csv"] = repo_rel(compact_xt_kernel_csv)
            outputs["camera_e2e_compact_crosstalk_kernel_npz"] = repo_rel(compact_xt_kernel_npz)
            outputs["camera_e2e_compact_crosstalk_summary_csv"] = repo_rel(compact_xt_summary_csv)
            outputs["camera_e2e_compact_crosstalk_html"] = repo_rel(compact_xt_html)
        package["latest_camera_e2e_ingest_export"] = {
            "schema": payload["schema"],
            "product_lut_ready": payload["product_lut_ready"],
            "sensor_count": payload["sensor_count"],
            "field_row_count": payload["field_row_count"],
            "fdtd_pass_field_row_count": payload["fdtd_pass_field_row_count"],
            "crosstalk_row_count": payload["crosstalk_row_count"],
            "compact_crosstalk_kernel_row_count": payload["compact_crosstalk_kernel_row_count"],
            "field_lut_npz_validation_pass": npz_validation["pass"],
            "manifest": repo_rel(manifest_json),
            "html": repo_rel(html_report),
        }
        package_json.write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({key: payload[key] for key in ["schema", "sensor_count", "field_row_count", "fdtd_pass_field_row_count", "crosstalk_row_count", "product_lut_ready"]}, indent=2))


if __name__ == "__main__":
    main()
