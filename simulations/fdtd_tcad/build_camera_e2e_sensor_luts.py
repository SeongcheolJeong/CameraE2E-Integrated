#!/usr/bin/env python3
"""Build per-sensor CameraE2E LUT packages from image_sensor_db evidence.

The generated files are intentionally gate-driven. They collect sensor metadata,
stack/proxy material provenance, current TCAD electrical-response rows, and any
available crosstalk evidence without pretending that proxy data is calibrated.
CameraE2E can ingest these packages for research/trend experiments, while the
readiness gates keep product-use blockers explicit.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SENSOR_CATALOG = ROOT / "image_sensor_db" / "sensor_catalog.csv"
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_TCAD_PROFILE_DIR = ROOT / "image_sensor_db" / "generated_tcad_profiles"
DEFAULT_TCAD_MAJOR_REPORT = ROOT / "runs" / "image_sensor_db_tcad_major_sim" / "tcad_major_sim_report.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_FIELD_MAP_CSV = ROOT / "image_sensor_db" / "camera_module_field_map.csv"

ARCHETYPE_CROSSTALK = {
    "quad_2x2": ROOT
    / "runs"
    / "replay_jobs"
    / "cad_quad_2x2_ocl_5x5_crosstalk_fdtd_replay_1782240785_f5d99406"
    / "crosstalk_kernel.json",
    "bayer_1x1": ROOT / "runs" / "crosstalk_kernel_smoke_guard_res8" / "crosstalk_kernel.json",
    "nona_3x3": ROOT / "runs" / "supercell_lut_ocl_3x3_smoke" / "camera_lut.json",
}

INDEX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "pixel_architecture",
    "cfa_pattern",
    "ocl_mode_guess",
    "cra_input_gate",
    "fdtd_field_sweep_gate",
    "tcad_gate",
    "crosstalk_gate",
    "cra_field_gate",
    "measured_stack_gate",
    "camera_e2e_usage_gate",
    "camera_e2e_ready",
    "reason",
    "lut_json",
]

FIELD_COLUMNS = [
    "slug",
    "code",
    "field_x_norm",
    "field_z_norm",
    "photo_shift_x_um",
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "total_response_proxy",
    "relative_response_to_center",
    "split_phase_x_proxy",
    "solver_gate",
    "attempt_id",
    "source",
]

CROSSTALK_COLUMNS = [
    "slug",
    "code",
    "archetype",
    "source_path",
    "archetype_gate",
    "executed_sweep_gate",
    "executed_sweep_summary_csv",
    "mode",
    "neighborhood",
    "simulation_neighborhood",
    "raw_pd_kernel_shape",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "truncation_response_fraction",
    "grid_resolution_gate_pass",
    "convergence_status",
    "usage_gate",
]

FIELD_DESIGN_COLUMNS = [
    "slug",
    "code",
    "field_case",
    "field_x_norm",
    "field_z_norm",
    "wavelength_set_nm",
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
    "edge_cra_assumption_deg",
    "focus_target_depth_um",
    "lens_shift_cap_um",
    "ocl_mode_guess",
    "ocl_supercell_pitch_um",
    "measurement_gate",
    "source",
]

FIELD_MAP_IMPORT_COLUMNS = [
    "slug",
    "code",
    "field_case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "wavelength_set_nm",
    "measurement_gate",
    "source",
]

OPTIONAL_FIELD_MAP_IMPORT_COLUMNS = [
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_tolerance_profile",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
]

FIELD_MAP_OUTPUT_COLUMNS = FIELD_MAP_IMPORT_COLUMNS + OPTIONAL_FIELD_MAP_IMPORT_COLUMNS

REQUIRED_RUN_COLUMNS = [
    "slug",
    "code",
    "requirement_id",
    "solver",
    "priority",
    "status",
    "current_gate",
    "why_required",
    "color_channels",
    "field_case_count",
    "wavelength_set_nm",
    "target_resolution_px_per_um",
    "fdtd_cell_volume_um3",
    "estimated_volume_factor",
    "fdtd_domain_factor",
    "required_neighborhood",
    "required_simulation_neighborhood",
    "raw_pixel_domain",
    "guard_cells",
    "command_hint",
]

SOLVER_COVERAGE_COLUMNS = [
    "slug",
    "code",
    "fdtd_field_gate",
    "fdtd_field_tier",
    "fdtd_field_summary_rows",
    "fdtd_field_summary_csv",
    "fdtd_field_html",
    "crosstalk_gate",
    "crosstalk_tier",
    "crosstalk_summary_rows",
    "crosstalk_summary_csv",
    "crosstalk_html",
    "product_lut_ready",
]

QUANTITATIVE_PLAN_COLUMNS = [
    "slug",
    "code",
    "requirement_id",
    "target_resolution_px_per_um",
    "fdtd_cell_volume_um3",
    "estimated_volume_factor",
    "fdtd_domain_factor",
    "color_channels",
    "color_count",
    "field_case_count",
    "wavelength_count",
    "full_sweep_points",
    "recommended_batch_points",
    "reference_resolution_px_per_um",
    "reference_seconds_per_point",
    "estimated_seconds",
    "estimated_hours",
    "estimate_model",
    "notes",
]

QUANTITATIVE_QUEUE_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "requirement_id",
    "solver",
    "color",
    "field_case",
    "wavelength_nm",
    "target_resolution_px_per_um",
    "fdtd_cell_volume_um3",
    "estimated_volume_factor",
    "fdtd_domain_factor",
    "estimated_seconds",
    "estimated_hours",
    "command",
]

QUANTITATIVE_COVERAGE_COLUMNS = [
    "slug",
    "code",
    "solver",
    "required_points",
    "attempted_points",
    "completed_points",
    "resource_limited_points",
    "pass_points",
    "check_points",
    "fail_points",
    "coverage_fraction",
    "attempted_fraction",
    "gate",
    "product_lut_ready",
    "summary_csv",
    "notes",
]

RESOURCE_LIMITED_BATCH_COLUMNS = [
    "queue_id",
    "slug",
    "code",
    "solver",
    "color",
    "field_case",
    "wavelength_nm",
    "target_resolution_px_per_um",
    "fdtd_domain_factor",
    "estimated_voxels",
    "resource_gate",
    "convergence_status",
    "local_summary_csv",
    "batch_command",
    "notes",
]

FIELD_MAP_VALIDATION_COLUMNS = [
    "scope",
    "slug",
    "code",
    "field_case",
    "gate",
    "severity",
    "issue_code",
    "issue",
    "measurement_gate",
    "source",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_tolerance_profile",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "cra_mismatch_gate",
    "lens_shift_x_um",
    "lens_shift_z_um",
]

CAMERA_E2E_EXPORT_MANIFEST_COLUMNS = [
    "artifact_id",
    "path",
    "artifact_role",
    "schema",
    "intended_scope",
    "research_ingest_allowed",
    "production_ingest_allowed",
    "actual_gate",
    "required_gate_for_production",
    "notes",
]

DEFAULT_FIELD_COORDS = [
    ("center", 0.0, 0.0),
    ("x_minus_edge", -1.0, 0.0),
    ("x_plus_edge", 1.0, 0.0),
    ("z_minus_edge", 0.0, -1.0),
    ("z_plus_edge", 0.0, 1.0),
    ("diag_minus_minus", -1.0, -1.0),
    ("diag_minus_plus", -1.0, 1.0),
    ("diag_plus_minus", 1.0, -1.0),
    ("diag_plus_plus", 1.0, 1.0),
]
DEFAULT_WAVELENGTHS_NM = (450, 550, 620)
DEFAULT_EDGE_CRA_DEG = 20.0
DEFAULT_FDTD_RESOLUTION_PX_PER_UM = 80
MIN_FEATURE_PIXELS_REQUIRED = 2.0
MIN_SI_WAVELENGTH_PIXELS_REQUIRED = 8.0
WORST_CASE_SI_N_FOR_VISIBLE = 4.7
MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"
TRUE_CRA_GATES = {"MEASURED", "CALIBRATED", "RAYTRACE_VALIDATED"}
DEFAULT_REFERENCE_SECONDS_PER_POINT = 1200.0
DEFAULT_REFERENCE_RESOLUTION_PX_PER_UM = 80
# SC550XS 1.0um pilot-like periodic field cell:
# pitch^2 * (2*pml + air_top + lens + cfa + passivation + si + bottom_air).
DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3 = 6.70


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
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
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def html_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return html.escape(f"{value:.6g}")
        return ""
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if limit is not None and len(rows) > limit:
        body.append(
            "<tr><td colspan=\"%d\">... %d more rows in CSV</td></tr>"
            % (len(columns), len(rows) - limit)
        )
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normalized_gate(value: Any, default: str) -> str:
    text = str(value or "").strip().upper()
    return text if text else default


def load_field_map_overrides(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in FIELD_MAP_IMPORT_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required field-map columns: {', '.join(missing)}")
        rows = list(reader)
    overrides: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        slug = str(row.get("slug") or "").strip()
        field_case = str(row.get("field_case") or "").strip()
        if not slug or not field_case:
            continue
        overrides.setdefault(slug, []).append(dict(row))
    return overrides


def load_quantitative_coverage(output_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = output_dir / "camera_e2e_quantitative_coverage.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coverage: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        slug = str(row.get("slug") or "")
        solver = str(row.get("solver") or "")
        if slug and solver:
            coverage[(slug, solver)] = row
    return coverage


def quantitative_evidence_from_coverage(
    row: dict[str, str] | None,
    *,
    schema: str,
) -> dict[str, Any] | None:
    if not row:
        return None
    completed = int(safe_float(row.get("completed_points"), 0))
    attempted = int(safe_float(row.get("attempted_points"), completed))
    required = int(safe_float(row.get("required_points"), 0))
    resource_limited = int(safe_float(row.get("resource_limited_points"), 0))
    gate_counts = {
        "PASS": int(safe_float(row.get("pass_points"), 0)),
        "CHECK": int(safe_float(row.get("check_points"), 0)),
        "FAIL": int(safe_float(row.get("fail_points"), 0)),
    }
    gate_counts = {key: value for key, value in gate_counts.items() if value}
    if attempted == 0 and not gate_counts:
        return {
            "schema": schema,
            "tier": "quantitative",
            "summary_row_count": 0,
            "required_point_count": required,
            "attempted_point_count": 0,
            "completed_point_count": 0,
            "resource_limited_point_count": 0,
            "coverage_fraction": row.get("coverage_fraction", "0"),
            "attempted_fraction": row.get("attempted_fraction", "0"),
            "solver_gate_counts": {},
            "summary_csv": row.get("summary_csv", ""),
            "html_report": "",
            "product_lut_ready": False,
            "coverage_gate": row.get("gate", "MISSING"),
            "coverage_notes": row.get("notes", ""),
        }
    return {
        "schema": schema,
        "tier": "quantitative",
        "summary_row_count": attempted,
        "required_point_count": required,
        "attempted_point_count": attempted,
        "completed_point_count": completed,
        "resource_limited_point_count": resource_limited,
        "coverage_fraction": row.get("coverage_fraction", ""),
        "attempted_fraction": row.get("attempted_fraction", ""),
        "solver_gate_counts": dict(sorted(gate_counts.items())),
        "summary_csv": row.get("summary_csv", ""),
        "html_report": "",
        "product_lut_ready": str(row.get("product_lut_ready", "")).lower() == "true",
        "coverage_gate": row.get("gate", ""),
        "coverage_notes": row.get("notes", ""),
    }


def field_input_gate(
    rows: list[dict[str, Any]],
    required_cases: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str]:
    if not rows:
        return "MISSING", "no CRA/field input rows"
    required = set(required_cases or [item[0] for item in DEFAULT_FIELD_COORDS])
    actual = {str(row.get("field_case") or "").strip() for row in rows if str(row.get("field_case") or "").strip()}
    missing = sorted(required - actual)
    if missing:
        return "CHECK", "field CRA input is missing standard anchor cases: " + ",".join(missing)
    nonfinite_columns = []
    for row in rows:
        for column in ("field_x_norm", "field_z_norm", "cra_x_deg", "cra_z_deg", "lens_shift_x_um", "lens_shift_z_um"):
            if not math.isfinite(safe_float(row.get(column))):
                nonfinite_columns.append(f"{row.get('field_case', '')}:{column}")
    if nonfinite_columns:
        return "CHECK", "field CRA input has non-finite numeric values: " + ",".join(nonfinite_columns[:8])
    gates = {normalized_gate(row.get("measurement_gate"), "ASSUMED_NOT_MEASURED") for row in rows}
    if gates and gates.issubset(TRUE_CRA_GATES):
        mismatch_gates = {normalized_gate(row.get("cra_mismatch_gate"), "MISSING") for row in rows}
        if "FAIL" in mismatch_gates:
            return "FAIL", "field CRA input has CRA mismatch rows outside tolerance"
        if "MISSING" in mismatch_gates or "" in mismatch_gates:
            return "CHECK", "field CRA input is measured/calibrated but lens-vs-sensor CRA mismatch reference is missing"
        if "CHECK" in mismatch_gates:
            return "CHECK", "field CRA input is measured/calibrated but some CRA mismatch rows are outside PASS tolerance"
        return "PASS", "field CRA, sensor acceptance CRA, microlens shift, and mismatch tolerance all pass"
    if "ASSUMED_NOT_MEASURED" in gates:
        return "ASSUMED", "field CRA and microlens shift are generated design priors, not measured module data"
    return "CHECK", "field CRA input was imported but is not marked measured/calibrated/validated"


def validation_issue(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    slug: str = "",
    code: str = "",
    field_case: str = "",
    gate: str,
    severity: str,
    issue_code: str,
    issue: str,
    source_row: dict[str, Any] | None = None,
) -> None:
    source_row = source_row or {}
    rows.append(
        {
            "scope": scope,
            "slug": slug or source_row.get("slug", ""),
            "code": code or source_row.get("code", ""),
            "field_case": field_case or source_row.get("field_case", ""),
            "gate": gate,
            "severity": severity,
            "issue_code": issue_code,
            "issue": issue,
            "measurement_gate": source_row.get("measurement_gate", ""),
            "source": source_row.get("source", ""),
            "field_x_norm": source_row.get("field_x_norm", ""),
            "field_z_norm": source_row.get("field_z_norm", ""),
            "cra_x_deg": source_row.get("cra_x_deg", ""),
            "cra_z_deg": source_row.get("cra_z_deg", ""),
            "lens_cra_x_deg": source_row.get("lens_cra_x_deg", ""),
            "lens_cra_z_deg": source_row.get("lens_cra_z_deg", ""),
            "sensor_cra_x_deg": source_row.get("sensor_cra_x_deg", ""),
            "sensor_cra_z_deg": source_row.get("sensor_cra_z_deg", ""),
            "cra_mismatch_total_deg": source_row.get("cra_mismatch_total_deg", ""),
            "cra_mismatch_tolerance_profile": source_row.get("cra_mismatch_tolerance_profile", ""),
            "cra_mismatch_pass_tolerance_deg": source_row.get("cra_mismatch_pass_tolerance_deg", ""),
            "cra_mismatch_check_tolerance_deg": source_row.get("cra_mismatch_check_tolerance_deg", ""),
            "cra_mismatch_gate": source_row.get("cra_mismatch_gate", ""),
            "lens_shift_x_um": source_row.get("lens_shift_x_um", ""),
            "lens_shift_z_um": source_row.get("lens_shift_z_um", ""),
        }
    )


def field_map_validation_report(
    *,
    field_map_csv: Path,
    selected: list[str],
    catalog: dict[str, dict[str, Any]],
    field_map_overrides: dict[str, list[dict[str, Any]]],
    field_design_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    expected_cases = {case for case, _field_x, _field_z in DEFAULT_FIELD_COORDS}
    design_by_slug_case = {
        (str(row.get("slug") or ""), str(row.get("field_case") or "")): row for row in field_design_rows
    }
    selected_set = set(selected)
    all_input_rows = [row for rows in field_map_overrides.values() for row in rows]

    if not field_map_csv.exists():
        validation_issue(
            validation_rows,
            scope="field_map_csv",
            gate="MISSING",
            severity="blocker",
            issue_code="field_map_csv_missing",
            issue="No camera_module_field_map.csv was supplied; CRA and microlens shifts are generated design priors.",
        )
        return (
            {
                "schema": "camera_module_field_map_validation_v1",
                "path": repo_rel(field_map_csv),
                "exists": False,
                "gate": "MISSING",
                "product_lut_ready": False,
                "selected_sensor_count": len(selected),
                "input_row_count": 0,
                "validation_row_count": len(validation_rows),
                "issue_count": len(validation_rows),
                "blocker_count": len(validation_rows),
                "warning_count": 0,
                "required_cases": sorted(expected_cases),
                "pass_gates": sorted(TRUE_CRA_GATES),
                "notes": [
                    "TechInsights-derived sensor metadata does not provide measured CRA or microlens-shift maps.",
                    "Provide image_sensor_db/camera_module_field_map.csv from module raytrace, lab measurement, or calibrated optical design.",
                ],
            },
            validation_rows,
        )

    unknown_slugs = sorted(set(field_map_overrides) - selected_set)
    for slug in unknown_slugs:
        for row in field_map_overrides[slug]:
            validation_issue(
                validation_rows,
                scope="field_map_csv",
                gate="CHECK",
                severity="warning",
                issue_code="slug_not_selected",
                issue="Field-map row is for a sensor not selected in this package run.",
                source_row=row,
            )

    for slug in selected:
        code = catalog.get(slug, {}).get("code", "")
        rows = field_map_overrides.get(slug, [])
        if not rows:
            validation_issue(
                validation_rows,
                scope="field_map_csv",
                slug=slug,
                code=code,
                gate="MISSING",
                severity="blocker",
                issue_code="sensor_field_map_missing",
                issue="Selected sensor has no imported CRA/ML-shift field-map rows.",
            )
            continue

        seen_cases: set[str] = set()
        actual_cases = {str(row.get("field_case") or "").strip() for row in rows}
        missing_cases = sorted(expected_cases - actual_cases)
        if missing_cases:
            validation_issue(
                validation_rows,
                scope="field_map_csv",
                slug=slug,
                code=code,
                gate="CHECK",
                severity="blocker",
                issue_code="standard_field_anchors_missing",
                issue="Missing standard field anchors required for production CameraE2E LUT: "
                + ",".join(missing_cases),
            )

        for row in rows:
            field_case = str(row.get("field_case") or "").strip()
            if field_case in seen_cases:
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="warning",
                    issue_code="duplicate_field_case",
                    issue="Duplicate field_case for this sensor; downstream sweeps need one stable row per anchor.",
                    source_row=row,
                )
            seen_cases.add(field_case)

            gate = normalized_gate(row.get("measurement_gate"), "IMPORTED_NOT_VALIDATED")
            if gate not in TRUE_CRA_GATES:
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="blocker",
                    issue_code="measurement_gate_not_product_usable",
                    issue="measurement_gate must be MEASURED, CALIBRATED, or RAYTRACE_VALIDATED for production use.",
                    source_row=row,
                )

            for column in ("field_x_norm", "field_z_norm", "cra_x_deg", "cra_z_deg", "lens_shift_x_um", "lens_shift_z_um"):
                value = safe_float(row.get(column))
                if not math.isfinite(value):
                    validation_issue(
                        validation_rows,
                        scope="field_map_csv",
                        slug=slug,
                        code=code,
                        field_case=field_case,
                        gate="CHECK",
                        severity="blocker",
                        issue_code="nonfinite_numeric_value",
                        issue=f"{column} must be finite for solver and CameraE2E interpolation.",
                        source_row=row,
                    )

            for column in OPTIONAL_FIELD_MAP_IMPORT_COLUMNS:
                raw_value = str(row.get(column, "") or "").strip()
                if raw_value and column not in {"cra_mismatch_tolerance_profile"} and not math.isfinite(safe_float(raw_value)):
                    validation_issue(
                        validation_rows,
                        scope="field_map_csv",
                        slug=slug,
                        code=code,
                        field_case=field_case,
                        gate="CHECK",
                        severity="blocker",
                        issue_code="nonfinite_optional_numeric_value",
                        issue=f"{column} must be finite when supplied.",
                        source_row=row,
                    )

            lens_cra_x = safe_float(row.get("lens_cra_x_deg"), safe_float(row.get("cra_x_deg")))
            lens_cra_z = safe_float(row.get("lens_cra_z_deg"), safe_float(row.get("cra_z_deg")))
            sensor_cra_x = safe_float(row.get("sensor_cra_x_deg"))
            sensor_cra_z = safe_float(row.get("sensor_cra_z_deg"))
            tolerance_profile, pass_tol, check_tol = cra_mismatch_tolerance_policy(
                catalog.get(slug, {}),
                lens_cra_total_deg=math.hypot(lens_cra_x, lens_cra_z),
                profile_override=row.get("cra_mismatch_tolerance_profile", ""),
                pass_override=row.get("cra_mismatch_pass_tolerance_deg", ""),
                check_override=row.get("cra_mismatch_check_tolerance_deg", ""),
            )
            mismatch = cra_mismatch_fields(
                lens_cra_x_deg=lens_cra_x,
                lens_cra_z_deg=lens_cra_z,
                sensor_cra_x_deg=sensor_cra_x,
                sensor_cra_z_deg=sensor_cra_z,
                tolerance_profile=tolerance_profile,
                pass_tolerance_deg=pass_tol,
                check_tolerance_deg=check_tol,
            )
            row_with_mismatch = {**row, **mismatch}
            product_usable_gate = gate in TRUE_CRA_GATES
            if product_usable_gate and mismatch["cra_mismatch_gate"] == "MISSING":
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="blocker",
                    issue_code="cra_mismatch_reference_missing",
                    issue="Product field-map rows need sensor_cra_x_deg/sensor_cra_z_deg so lens CRA can be compared to sensor/ML acceptance CRA.",
                    source_row=row_with_mismatch,
                )
            elif mismatch["cra_mismatch_gate"] == "FAIL":
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="FAIL",
                    severity="blocker",
                    issue_code="cra_mismatch_out_of_tolerance",
                    issue="Lens CRA and sensor/ML acceptance CRA mismatch is outside the configured tolerance.",
                    source_row=row_with_mismatch,
                )
            elif product_usable_gate and mismatch["cra_mismatch_gate"] == "CHECK":
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="warning",
                    issue_code="cra_mismatch_check_tolerance",
                    issue="Lens CRA and sensor/ML acceptance CRA mismatch is outside PASS tolerance but inside CHECK tolerance.",
                    source_row=row_with_mismatch,
                )

            field_x = safe_float(row.get("field_x_norm"))
            field_z = safe_float(row.get("field_z_norm"))
            if math.isfinite(field_x) and abs(field_x) > 1.25:
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="warning",
                    issue_code="field_x_outside_nominal_range",
                    issue="field_x_norm is outside the expected -1..1 image-field range.",
                    source_row=row,
                )
            if math.isfinite(field_z) and abs(field_z) > 1.25:
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="warning",
                    issue_code="field_z_outside_nominal_range",
                    issue="field_z_norm is outside the expected -1..1 image-field range.",
                    source_row=row,
                )

            design_row = design_by_slug_case.get((slug, field_case), {})
            cap = safe_float(design_row.get("lens_shift_cap_um"), math.nan)
            shift_x = abs(safe_float(row.get("lens_shift_x_um")))
            shift_z = abs(safe_float(row.get("lens_shift_z_um")))
            if math.isfinite(cap) and ((math.isfinite(shift_x) and shift_x > cap) or (math.isfinite(shift_z) and shift_z > cap)):
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="warning",
                    issue_code="lens_shift_exceeds_prior_cap",
                    issue="Imported lens shift exceeds the prior cap; this can be valid but should be traceable to module raytrace/calibration.",
                    source_row=row,
                )

            source = str(row.get("source") or "").strip().lower()
            if not source or "replace" in source or "example" in source:
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="CHECK",
                    severity="blocker",
                    issue_code="source_provenance_missing",
                    issue="source must identify the module raytrace, measurement, or calibration provenance.",
                    source_row=row,
                )

            if not any(
                issue_row
                for issue_row in validation_rows
                if issue_row.get("slug") == slug
                and issue_row.get("field_case") == field_case
                and issue_row.get("severity") in {"blocker", "warning"}
            ):
                validation_issue(
                    validation_rows,
                    scope="field_map_csv",
                    slug=slug,
                    code=code,
                    field_case=field_case,
                    gate="PASS",
                    severity="info",
                    issue_code="row_valid",
                    issue="Field-map row has required numeric values, product-usable measurement_gate, and source provenance.",
                    source_row=row,
                )

    blocker_count = sum(1 for row in validation_rows if row.get("severity") == "blocker")
    warning_count = sum(1 for row in validation_rows if row.get("severity") == "warning")
    if blocker_count:
        gate = "CHECK"
    elif warning_count:
        gate = "CHECK"
    else:
        gate = "PASS"
    return (
        {
            "schema": "camera_module_field_map_validation_v1",
            "path": repo_rel(field_map_csv),
            "exists": True,
            "gate": gate,
            "product_lut_ready": gate == "PASS",
            "selected_sensor_count": len(selected),
            "input_row_count": len(all_input_rows),
            "validation_row_count": len(validation_rows),
            "blocker_count": blocker_count,
            "warning_count": warning_count,
            "required_cases": sorted(expected_cases),
            "pass_gates": sorted(TRUE_CRA_GATES),
        },
        validation_rows,
    )


def write_field_map_validation_html(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Module Field Map Validation</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #081118; color: #e5f3ff; }}
    h1 {{ margin: 0 0 8px; }}
    .panel {{ border: 1px solid #244255; background: #0e1b25; border-radius: 8px; padding: 14px; margin: 16px 0; }}
    .muted {{ color: #99b2c4; }}
    code {{ color: #d8f8ff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid #244255; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #52e1ff; background: #102633; }}
  </style>
</head>
<body>
  <h1>Camera Module Field Map Validation</h1>
  <div class="muted">Gate: <code>{html_cell(report.get("gate", ""))}</code>; product LUT ready: <code>{html_cell(report.get("product_lut_ready", False))}</code></div>
  <div class="panel">
    <strong>Input</strong>
    <div class="muted"><code>{html_cell(report.get("path", ""))}</code>; exists: <code>{html_cell(report.get("exists", ""))}</code>; rows: <code>{html_cell(report.get("input_row_count", ""))}</code></div>
  </div>
  <div class="panel">
    <strong>Validation Rows</strong>
    {html_table(rows, FIELD_MAP_VALIDATION_COLUMNS) if rows else "<p>No validation rows.</p>"}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def camera_e2e_export_manifest_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = package.get("outputs", {}) if isinstance(package.get("outputs"), dict) else {}
    all_ready = bool(package.get("sensor_count")) and package.get("camera_e2e_ready_count") == package.get("sensor_count")
    production_gate = "PASS" if all_ready else "BLOCKED"

    def row(
        artifact_id: str,
        key: str,
        role: str,
        schema: str,
        scope: str,
        research_allowed: bool,
        production_allowed: bool,
        actual_gate: str,
        required_gate: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_id,
            "path": outputs.get(key, ""),
            "artifact_role": role,
            "schema": schema,
            "intended_scope": scope,
            "research_ingest_allowed": research_allowed,
            "production_ingest_allowed": production_allowed,
            "actual_gate": actual_gate,
            "required_gate_for_production": required_gate,
            "notes": notes,
        }

    return [
        row(
            "sensor_index",
            "index_csv",
            "gate_index",
            "camera_e2e_sensor_index_csv_v1",
            "research_and_gate_review",
            True,
            all_ready,
            production_gate,
            "all sensor camera_e2e_usage_gate PASS",
            "Primary gate table. Production consumers must filter to camera_e2e_ready=true.",
        ),
        row(
            "field_design_cases",
            "field_design_cases_csv",
            "cra_ml_shift_input_cases",
            "camera_e2e_field_design_cases_v1",
            "solver_input_and_review",
            True,
            False,
            package.get("field_map_validation", {}).get("gate", package.get("field_map_input", {}).get("exists", False)),
            "camera_module_field_map_validation PASS plus FDTD field convergence PASS",
            "Contains either imported field map rows or generated design priors; not itself a solved response LUT.",
        ),
        row(
            "field_map_validation",
            "field_map_validation_json",
            "cra_ml_shift_input_validation",
            "camera_module_field_map_validation_v1",
            "gate_review",
            True,
            package.get("field_map_validation", {}).get("product_lut_ready", False),
            package.get("field_map_validation", {}).get("gate", "MISSING"),
            "PASS",
            "Validates CRA and microlens-shift provenance before production use.",
        ),
        row(
            "quantitative_field_lut",
            "quantitative_field_lut_csv",
            "field_response_kpi_lut",
            "camera_e2e_quantitative_field_lut_csv_v1",
            "camera_e2e_response_input",
            True,
            all_ready,
            production_gate,
            "field coverage PASS, measured CRA PASS, measured stack/material PASS",
            "Sparse quantitative FDTD KPI rows; current rows can be research-only until all gates pass.",
        ),
        row(
            "quantitative_crosstalk_lut",
            "quantitative_crosstalk_lut_csv",
            "crosstalk_kernel_kpi_lut",
            "camera_e2e_quantitative_crosstalk_lut_csv_v1",
            "camera_e2e_crosstalk_input",
            True,
            all_ready,
            production_gate,
            "crosstalk quantitative coverage PASS",
            "Binning/OCL CameraE2E kernels need full-domain or validated patch crosstalk convergence.",
        ),
        row(
            "resource_limited_batch_plan",
            "resource_limited_batch_plan_csv",
            "batch_execution_plan",
            "camera_e2e_resource_limited_batch_plan_csv_v1",
            "execution_planning",
            True,
            False,
            "CHECK",
            "no resource-limited points remaining",
            "Rows here are not completed solver evidence; execute on a larger machine or cluster.",
        ),
        row(
            "required_runs",
            "required_runs_csv",
            "remaining_work_plan",
            "camera_e2e_required_runs_csv_v1",
            "execution_planning",
            True,
            False,
            "CHECK",
            "no REQUIRED items remaining",
            "Tracks P0/P1 work required before CameraE2E product use.",
        ),
    ]


def solver_evidence_gate(evidence: dict[str, Any] | None, *, product_requires_quantitative: bool = True) -> tuple[str, str]:
    if not evidence:
        return "MISSING", "no executed solver evidence"
    gate_counts = evidence.get("solver_gate_counts", {})
    if not isinstance(gate_counts, dict) or not gate_counts:
        return "MISSING", "solver evidence has no gate counts"
    normalized_counts = {str(key).upper(): int(value) for key, value in gate_counts.items()}
    tier = str(evidence.get("tier") or "").lower()
    coverage_gate = str(evidence.get("coverage_gate") or "").upper()
    required_points = int(safe_float(evidence.get("required_point_count"), 0))
    completed_points = int(safe_float(evidence.get("completed_point_count"), 0))
    resource_limited_points = int(safe_float(evidence.get("resource_limited_point_count"), 0))
    if normalized_counts.get("FAIL", 0) > 0:
        return "FAIL", f"{tier or 'unknown'} run has FAIL rows"
    if resource_limited_points > 0:
        return "CHECK", f"{resource_limited_points} point(s) are resource-limited and need batch/cluster execution"
    if normalized_counts.get("CHECK", 0) > 0:
        return "CHECK", f"{tier or 'unknown'} run is check-only"
    if normalized_counts.get("PASS", 0) <= 0:
        return "MISSING", "solver evidence has no PASS/CHECK/FAIL rows"
    if product_requires_quantitative and tier != "quantitative":
        return "CHECK", f"{tier or 'unknown'} run has PASS rows but is not quantitative tier"
    if coverage_gate == "FAIL":
        return "FAIL", "quantitative coverage gate failed"
    if coverage_gate == "MISSING":
        return "MISSING", "quantitative coverage is missing"
    if coverage_gate == "CHECK":
        return "CHECK", str(evidence.get("coverage_notes") or "quantitative coverage is incomplete")
    if required_points > 0 and completed_points < required_points:
        return "CHECK", f"quantitative coverage is incomplete: {completed_points}/{required_points} points"
    return "PASS", "quantitative solver evidence has PASS rows only"


def cra_field_gate(cra_input_status: str, fdtd_gate: str) -> tuple[str, bool, str]:
    if fdtd_gate == "MISSING":
        return "MISSING", False, "FDTD field sweep is missing"
    if fdtd_gate == "FAIL":
        return "FAIL", False, "FDTD field sweep failed"
    if cra_input_status != "PASS":
        return "CHECK", False, f"CRA input gate is {cra_input_status}"
    if fdtd_gate != "PASS":
        return "CHECK", False, f"FDTD field sweep gate is {fdtd_gate}"
    return "PASS", True, "CRA input and FDTD field sweep both pass"


def solver_coverage_row(index_row: dict[str, Any], lut: dict[str, Any]) -> dict[str, Any]:
    field_evidence = lut.get("fdtd_field_sweep_evidence")
    if not isinstance(field_evidence, dict):
        field_evidence = {}
    crosstalk_evidence = lut.get("crosstalk_sweep_evidence")
    if not isinstance(crosstalk_evidence, dict):
        crosstalk_evidence = {}
    field_gate, _field_reason = solver_evidence_gate(field_evidence or None)
    crosstalk_gate, _crosstalk_reason = solver_evidence_gate(crosstalk_evidence or None)
    return {
        "slug": index_row.get("slug", ""),
        "code": index_row.get("code", ""),
        "fdtd_field_gate": field_gate,
        "fdtd_field_tier": field_evidence.get("tier", ""),
        "fdtd_field_summary_rows": field_evidence.get("summary_row_count", ""),
        "fdtd_field_summary_csv": field_evidence.get("summary_csv", ""),
        "fdtd_field_html": field_evidence.get("html_report", ""),
        "crosstalk_gate": crosstalk_gate,
        "crosstalk_tier": crosstalk_evidence.get("tier", ""),
        "crosstalk_summary_rows": crosstalk_evidence.get("summary_row_count", ""),
        "crosstalk_summary_csv": crosstalk_evidence.get("summary_csv", ""),
        "crosstalk_html": crosstalk_evidence.get("html_report", ""),
        "product_lut_ready": bool(field_evidence.get("product_lut_ready"))
        and bool(crosstalk_evidence.get("product_lut_ready")),
    }


def quantitative_reference(output_dir: Path) -> tuple[float, int, str]:
    pilot_manifest = (
        output_dir
        / "fdtd_field_sweep_quantitative_hp5_green_center_550_pilot"
        / "fdtd_field_sweep_manifest.csv"
    )
    if pilot_manifest.exists():
        rows = []
        with pilot_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        done_rows = [row for row in rows if str(row.get("status")) == "DONE"]
        if done_rows:
            row = done_rows[0]
            duration = safe_float(row.get("duration_s"), DEFAULT_REFERENCE_SECONDS_PER_POINT)
            points = max(1, int(safe_float(row.get("sweep_point_count"), 1)))
            resolution = int(safe_float(row.get("resolution_px_per_um"), DEFAULT_REFERENCE_RESOLUTION_PX_PER_UM))
            return duration / points, resolution, repo_rel(pilot_manifest)
    return DEFAULT_REFERENCE_SECONDS_PER_POINT, DEFAULT_REFERENCE_RESOLUTION_PX_PER_UM, "default_estimate"


def quantitative_plan_rows(
    required_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    reference_seconds, reference_resolution, reference_source = quantitative_reference(output_dir)
    rows = []
    for required in required_rows:
        if required.get("solver") != "Meep FDTD":
            continue
        field_cases = int(safe_float(required.get("field_case_count"), len(DEFAULT_FIELD_COORDS)))
        wavelengths = len([item for item in str(required.get("wavelength_set_nm", "")).split(";") if item])
        wavelengths = wavelengths or len(DEFAULT_WAVELENGTHS_NM)
        color_channels = [item for item in str(required.get("color_channels", "")).split(";") if item]
        color_channels = color_channels or ["red", "green", "blue"]
        colors = len(color_channels)
        points = field_cases * wavelengths * colors
        target_resolution = int(safe_float(required.get("target_resolution_px_per_um"), DEFAULT_FDTD_RESOLUTION_PX_PER_UM))
        cell_volume = safe_float(required.get("fdtd_cell_volume_um3"), DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3)
        volume_factor = max(0.05, safe_float(required.get("estimated_volume_factor"), cell_volume / DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3))
        domain_factor = max(1.0, safe_float(required.get("fdtd_domain_factor"), 1.0))
        scale = (target_resolution / max(1, reference_resolution)) ** 4
        estimated_seconds = reference_seconds * points * scale * volume_factor * domain_factor
        rows.append(
            {
                "slug": required.get("slug", ""),
                "code": required.get("code", ""),
                "requirement_id": required.get("requirement_id", ""),
                "target_resolution_px_per_um": target_resolution,
                "fdtd_cell_volume_um3": f"{cell_volume:.6g}",
                "estimated_volume_factor": f"{volume_factor:.3f}",
                "fdtd_domain_factor": f"{domain_factor:.3f}",
                "color_channels": ";".join(color_channels),
                "color_count": colors,
                "field_case_count": field_cases,
                "wavelength_count": wavelengths,
                "full_sweep_points": points,
                "recommended_batch_points": 1 if target_resolution >= 120 else 3,
                "reference_resolution_px_per_um": reference_resolution,
                "reference_seconds_per_point": f"{reference_seconds:.3f}",
                "estimated_seconds": f"{estimated_seconds:.1f}",
                "estimated_hours": f"{estimated_seconds / 3600.0:.2f}",
                "estimate_model": (
                    f"reference={reference_source}; seconds_per_point scales as "
                    f"(resolution/reference)^4 * periodic_cell_volume/reference_volume * finite_array_domain_factor"
                ),
                "notes": "Local full quantitative sweep is expensive; run point/batch slices and merge evidence.",
            }
        )
    return rows


def quantitative_queue_rows(
    required_rows: list[dict[str, Any]],
    field_design_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    reference_seconds, reference_resolution, _reference_source = quantitative_reference(output_dir)
    fields_by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in field_design_rows:
        fields_by_slug.setdefault(str(row.get("slug")), []).append(row)
    queue = []
    for required in required_rows:
        requirement_id = str(required.get("requirement_id", ""))
        if requirement_id not in {"fdtd_cra_rgb_field_sweep", "fdtd_crosstalk_kernel_convergence"}:
            continue
        slug = str(required.get("slug", ""))
        colors = tuple(item for item in str(required.get("color_channels", "")).split(";") if item)
        if not colors:
            colors = ("red", "green", "blue")
        target_resolution = int(safe_float(required.get("target_resolution_px_per_um"), DEFAULT_FDTD_RESOLUTION_PX_PER_UM))
        cell_volume = safe_float(required.get("fdtd_cell_volume_um3"), DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3)
        volume_factor = max(0.05, safe_float(required.get("estimated_volume_factor"), cell_volume / DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3))
        domain_factor = max(1.0, safe_float(required.get("fdtd_domain_factor"), 1.0))
        seconds_per_point = reference_seconds * (target_resolution / max(1, reference_resolution)) ** 4 * volume_factor * domain_factor
        wavelengths = [item for item in str(required.get("wavelength_set_nm", "")).split(";") if item]
        wavelengths = wavelengths or [str(value) for value in DEFAULT_WAVELENGTHS_NM]
        runner = (
            "run_camera_e2e_fdtd_field_sweep.py"
            if requirement_id == "fdtd_cra_rgb_field_sweep"
            else "run_camera_e2e_crosstalk_sweep.py"
        )
        solver_name = "field" if requirement_id == "fdtd_cra_rgb_field_sweep" else "crosstalk"
        for field in fields_by_slug.get(slug, []):
            field_case = str(field.get("field_case", ""))
            for color in colors:
                for wavelength in wavelengths:
                    queue_id = f"{slug}_{solver_name}_{color}_{field_case}_{wavelength}"
                    point_output = (
                        output_dir
                        / "quantitative_point_runs"
                        / slug
                        / solver_name
                        / color
                        / field_case
                        / f"{wavelength}nm"
                    )
                    command = (
                        f"python3 {runner} --tier quantitative --slugs {slug} "
                        f"--colors {color} --field-cases {field_case} --wavelengths-nm {wavelength} "
                        f"--output-dir {repo_rel(point_output)}"
                    )
                    queue.append(
                        {
                            "queue_id": queue_id,
                            "slug": slug,
                            "code": required.get("code", ""),
                            "requirement_id": requirement_id,
                            "solver": solver_name,
                            "color": color,
                            "field_case": field_case,
                            "wavelength_nm": wavelength,
                            "target_resolution_px_per_um": target_resolution,
                            "fdtd_cell_volume_um3": f"{cell_volume:.6g}",
                            "estimated_volume_factor": f"{volume_factor:.3f}",
                            "fdtd_domain_factor": f"{domain_factor:.3f}",
                            "estimated_seconds": f"{seconds_per_point:.1f}",
                            "estimated_hours": f"{seconds_per_point / 3600.0:.2f}",
                            "command": command,
                        }
                    )
    return queue


def resource_limited_batch_rows(
    output_dir: Path,
    quantitative_queue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_path = output_dir / "camera_e2e_quantitative_merged_summary.csv"
    if not merged_path.exists():
        return []
    queue_by_id = {str(row.get("queue_id", "")): row for row in quantitative_queue}
    rows = []

    def command_with_option(command: str, option: str, value: str) -> str:
        parts = shlex.split(command)
        if option in parts:
            index = parts.index(option)
            if index + 1 < len(parts):
                parts[index + 1] = value
            else:
                parts.append(value)
        else:
            parts.extend([option, value])
        return shlex.join(parts)

    for row in read_csv(merged_path):
        if str(row.get("convergence_status", "")).upper() != "RESOURCE_LIMIT":
            continue
        queue_id = str(row.get("queue_id", ""))
        queue_row = queue_by_id.get(queue_id, {})
        base_command = str(queue_row.get("command", ""))
        if not base_command:
            continue
        solver = str(row.get("solver", ""))
        batch_command = command_with_option(base_command, "--timeout-s", "86400")
        if solver == "crosstalk":
            batch_command = command_with_option(batch_command, "--max-local-voxels", "0")
        notes = "Run outside the local interactive UX, then rerun merge_camera_e2e_quantitative_points.py and build_camera_e2e_sensor_luts.py."
        if solver == "field":
            notes = "Run this long field point on batch/HPC; the field sweep runner does not support --max-local-voxels."
        rows.append(
            {
                "queue_id": queue_id,
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "solver": solver,
                "color": row.get("color", ""),
                "field_case": row.get("field_case", ""),
                "wavelength_nm": row.get("wavelength_nm", ""),
                "target_resolution_px_per_um": row.get("target_resolution_px_per_um", ""),
                "fdtd_domain_factor": queue_row.get("fdtd_domain_factor", ""),
                "estimated_voxels": row.get("estimated_voxels", ""),
                "resource_gate": row.get("resource_gate", ""),
                "convergence_status": row.get("convergence_status", ""),
                "local_summary_csv": row.get("source_summary_csv", ""),
                "batch_command": batch_command,
                "notes": notes,
            }
        )
    return rows


def safe_slug_from_path(path_text: str) -> str:
    if not path_text:
        return ""
    return Path(path_text).stem


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = safe_slug_from_path(str(row.get("stack_config", "")))
        if not slug:
            profile = str(row.get("tcad_profile", ""))
            if profile:
                slug = Path(profile).parent.name
        if slug:
            output[slug] = row
    return output


def load_tcad_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    report = read_json(path)
    by_sensor: dict[str, list[dict[str, Any]]] = {}
    for row in report.get("run_rows", []):
        by_sensor.setdefault(str(row.get("slug")), []).append(row)
    summary = {str(row.get("slug")): row for row in report.get("sensor_summary_rows", [])}
    return by_sensor, summary


def ocl_mode_guess(catalog_row: dict[str, Any], stack: dict[str, Any]) -> str:
    cfa_pattern = str(catalog_row.get("cfa_pattern") or "").lower()
    architecture = str(catalog_row.get("pixel_architecture") or "").lower()
    microlens = str(catalog_row.get("microlens_type") or stack.get("techinsights_source", {}).get("derived_specs", {}).get("microlens_type", "")).lower()
    if "nona" in cfa_pattern or "9" in cfa_pattern:
        return "nona_3x3"
    if "quad" in cfa_pattern or "quad" in architecture or "qpd" in architecture:
        return "quad_2x2"
    if "high_precision" in microlens and safe_float(catalog_row.get("pixel_pitch_um")) <= 0.8:
        return "quad_2x2"
    return "bayer_1x1"


def material_measurement_gate(stack: dict[str, Any]) -> tuple[str, list[str]]:
    materials = stack.get("materials") if isinstance(stack.get("materials"), dict) else {}
    unmeasured = [
        name
        for name, value in materials.items()
        if isinstance(value, dict) and value.get("measured") is not True
    ]
    calibration = stack.get("calibration_status", {})
    if calibration.get("geometry_measured") is True and not unmeasured:
        return "PASS", []
    reasons = []
    if calibration.get("geometry_measured") is not True:
        reasons.append("stack geometry is TechInsights/proxy, not measured process geometry")
    if unmeasured:
        reasons.append("unmeasured n,k/material models: " + ",".join(sorted(unmeasured)))
    return "FAIL", reasons


def tcad_gate(rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> tuple[str, str]:
    if summary and summary.get("status_all"):
        status = str(summary.get("status_all"))
        if status in {"PASS", "CHECK", "UNSUPPORTED"}:
            return status, str(summary.get("center_failure_reason") or "")
    if not rows:
        return "MISSING", "no TCAD run rows"
    gates = {str(row.get("solver_gate")) for row in rows}
    if "FAIL" in gates:
        return "FAIL", "at least one TCAD row failed"
    if "UNSUPPORTED" in gates:
        return "UNSUPPORTED", "sensor architecture unsupported by current TCAD deck"
    if gates == {"PASS"}:
        return "PASS", ""
    return "CHECK", "TCAD solved with relaxed/check gate"


def field_rows_from_tcad(slug: str, catalog_row: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    center = next((row for row in rows if row.get("shift_label") == "center"), None)
    center_total = safe_float(center.get("total_photo_delta_e_a_per_cm")) if center else math.nan
    output = []
    width = safe_float(catalog_row.get("pixel_pitch_um"))
    if not math.isfinite(width) or width <= 0:
        widths = [safe_float(row.get("width_um")) for row in rows]
        width = next((value for value in widths if math.isfinite(value) and value > 0), 1.0)
    for row in rows:
        total = safe_float(row.get("total_photo_delta_e_a_per_cm"))
        shift = safe_float(row.get("photo_shift_x_um"))
        field_x = shift / (0.5 * width) if width > 0 and math.isfinite(shift) else math.nan
        output.append(
            {
                "slug": slug,
                "code": catalog_row.get("code", ""),
                "field_x_norm": max(-1.0, min(1.0, field_x)) if math.isfinite(field_x) else "",
                "field_z_norm": 0.0,
                "photo_shift_x_um": shift if math.isfinite(shift) else "",
                "wavelength_nm": 550.0,
                "cra_x_deg": "",
                "cra_z_deg": "",
                "total_response_proxy": total if math.isfinite(total) else "",
                "relative_response_to_center": total / center_total if math.isfinite(total) and math.isfinite(center_total) and center_total else "",
                "split_phase_x_proxy": row.get("photo_split_phase_x_proxy", ""),
                "solver_gate": row.get("solver_gate", ""),
                "attempt_id": row.get("attempt_id", ""),
                "source": "tcad_lateral_generation_shift_proxy",
            }
        )
    return output


def ocl_supercell_factor(mode: str) -> int:
    if mode == "quad_2x2":
        return 2
    if mode == "nona_3x3":
        return 3
    return 1


def estimated_focus_depth_um(catalog_row: dict[str, Any], stack: dict[str, Any]) -> float:
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    lens = safe_float(geometry.get("lens_height"), math.nan)
    cfa = safe_float(geometry.get("cfa_thickness"), math.nan)
    passivation = safe_float(geometry.get("passivation_thickness"), math.nan)
    si = safe_float(geometry.get("si_thickness"), math.nan)
    parts = [value for value in (lens, cfa, passivation) if math.isfinite(value) and value > 0]
    if math.isfinite(si) and si > 0:
        parts.append(min(0.40, 0.20 * si))
    if parts:
        return sum(parts)
    optical_stack = safe_float(catalog_row.get("optical_stack_height_um"), math.nan)
    if math.isfinite(optical_stack) and optical_stack > 0:
        return optical_stack
    pitch = safe_float(catalog_row.get("pixel_pitch_um"), 1.0)
    return max(0.8, 1.5 * pitch)


def lens_index_prior(stack: dict[str, Any]) -> float:
    materials = stack.get("materials", {}) if isinstance(stack.get("materials"), dict) else {}
    lens = materials.get("lens", {}) if isinstance(materials.get("lens"), dict) else {}
    # The generated stack currently uses a public patent anchor near n=1.61.
    return safe_float(lens.get("n_at_550nm"), 1.61)


def recommended_fdtd_resolution_px_per_um(catalog_row: dict[str, Any], stack: dict[str, Any]) -> tuple[int, str]:
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    specs = stack.get("techinsights_source", {}).get("derived_specs", {})
    passivation = safe_float(geometry.get("passivation_thickness"), math.nan)
    lens_edge_gap = safe_float(geometry.get("lens_edge_gap"), math.nan)
    dti_width = safe_float(specs.get("dti_width_um") or catalog_row.get("dti_width_um"), math.nan)
    feature_candidates = [
        value
        for value in (passivation, lens_edge_gap, dti_width)
        if math.isfinite(value) and value > 0
    ]
    min_feature_um = min(feature_candidates) if feature_candidates else math.nan
    feature_resolution = (
        math.ceil(MIN_FEATURE_PIXELS_REQUIRED / min_feature_um)
        if math.isfinite(min_feature_um) and min_feature_um > 0
        else DEFAULT_FDTD_RESOLUTION_PX_PER_UM
    )
    min_wavelength_um = min(DEFAULT_WAVELENGTHS_NM) / 1000.0
    si_resolution = math.ceil(
        MIN_SI_WAVELENGTH_PIXELS_REQUIRED * WORST_CASE_SI_N_FOR_VISIBLE / min_wavelength_um
    )
    recommended = max(DEFAULT_FDTD_RESOLUTION_PX_PER_UM, feature_resolution, si_resolution)
    reason = (
        f"max(default={DEFAULT_FDTD_RESOLUTION_PX_PER_UM}, "
        f"feature={feature_resolution} for min_feature_um={min_feature_um:.6g}, "
        f"si_visible={si_resolution} using n={WORST_CASE_SI_N_FOR_VISIBLE} at 450nm)"
    )
    return int(recommended), reason


def lens_shift_prior_um(cra_deg: float, focus_depth_um: float, pitch_um: float, n_eff: float) -> tuple[float, float]:
    cap = 0.35 * pitch_um if math.isfinite(pitch_um) and pitch_um > 0 else 0.35
    if not math.isfinite(cra_deg):
        return 0.0, cap
    raw_shift = math.tan(math.radians(cra_deg)) * focus_depth_um / max(n_eff, 1.0)
    return max(-cap, min(cap, raw_shift)), cap


def finite_or_blank(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def is_truthy_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def is_monochrome_sensor(catalog_row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(catalog_row.get("cfa_pattern", "")),
            str(catalog_row.get("sensor_modality", "")),
            str(catalog_row.get("device_name", "")),
        ]
    ).lower()
    return "mono" in text or "clear" in text


def estimated_fdtd_periodic_cell_volume_um3(stack: dict[str, Any]) -> float:
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    pitch = safe_float(geometry.get("pitch"), 1.0)
    pml = safe_float(geometry.get("pml"), 0.45)
    cell_y = (
        2.0 * pml
        + safe_float(geometry.get("air_top"), 0.55)
        + safe_float(geometry.get("lens_height"), 0.35)
        + safe_float(geometry.get("cfa_thickness"), 0.45)
        + safe_float(geometry.get("passivation_thickness"), 0.15)
        + safe_float(geometry.get("si_thickness"), 2.0)
        + safe_float(geometry.get("bottom_air"), 0.25)
    )
    if not math.isfinite(pitch) or pitch <= 0 or not math.isfinite(cell_y) or cell_y <= 0:
        return DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3
    return pitch * pitch * cell_y


def sensor_solver_channels(catalog_row: dict[str, Any]) -> tuple[str, ...]:
    if is_monochrome_sensor(catalog_row):
        return ("clear",)
    return ("red", "green", "blue")


def is_pdaf_or_split_sensor(catalog_row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(catalog_row.get("pixel_architecture", "")),
            str(catalog_row.get("microlens_type", "")),
            str(catalog_row.get("device_name", "")),
        ]
    ).lower()
    return is_truthy_text(catalog_row.get("has_pdaf")) or "dual" in text or "split" in text or "qpd" in text


def cra_mismatch_tolerance_policy(
    catalog_row: dict[str, Any],
    *,
    lens_cra_total_deg: float,
    profile_override: Any = "",
    pass_override: Any = "",
    check_override: Any = "",
) -> tuple[str, float, float]:
    pass_tol = safe_float(pass_override, math.nan)
    check_tol = safe_float(check_override, math.nan)
    profile = str(profile_override or "").strip()
    if math.isfinite(pass_tol) and math.isfinite(check_tol) and check_tol >= pass_tol:
        return profile or "custom_field_map_tolerance", pass_tol, check_tol

    pitch = safe_float(catalog_row.get("pixel_pitch_um"), math.nan)
    if is_pdaf_or_split_sensor(catalog_row):
        return "pdaf_split_strict", 2.0, 4.0
    if is_monochrome_sensor(catalog_row):
        return "mono_relaxed", 8.0, 12.0
    if (math.isfinite(pitch) and pitch <= 1.4) or (math.isfinite(lens_cra_total_deg) and lens_cra_total_deg > 20.0):
        return "rgb_small_pixel_or_high_cra", 3.0, 5.0
    return "rgb_mid_cra", 5.0, 7.0


def cra_mismatch_gate(total_deg: float, pass_tolerance_deg: float, check_tolerance_deg: float) -> str:
    if not all(math.isfinite(value) for value in (total_deg, pass_tolerance_deg, check_tolerance_deg)):
        return "MISSING"
    if total_deg <= pass_tolerance_deg:
        return "PASS"
    if total_deg <= check_tolerance_deg:
        return "CHECK"
    return "FAIL"


def cra_mismatch_fields(
    *,
    lens_cra_x_deg: float,
    lens_cra_z_deg: float,
    sensor_cra_x_deg: float,
    sensor_cra_z_deg: float,
    tolerance_profile: str,
    pass_tolerance_deg: float,
    check_tolerance_deg: float,
) -> dict[str, Any]:
    if all(math.isfinite(value) for value in (lens_cra_x_deg, lens_cra_z_deg, sensor_cra_x_deg, sensor_cra_z_deg)):
        dx = lens_cra_x_deg - sensor_cra_x_deg
        dz = lens_cra_z_deg - sensor_cra_z_deg
        total = math.hypot(dx, dz)
    else:
        dx = dz = total = math.nan
    return {
        "lens_cra_x_deg": finite_or_blank(lens_cra_x_deg),
        "lens_cra_z_deg": finite_or_blank(lens_cra_z_deg),
        "sensor_cra_x_deg": finite_or_blank(sensor_cra_x_deg),
        "sensor_cra_z_deg": finite_or_blank(sensor_cra_z_deg),
        "cra_mismatch_x_deg": finite_or_blank(dx),
        "cra_mismatch_z_deg": finite_or_blank(dz),
        "cra_mismatch_total_deg": finite_or_blank(total),
        "cra_mismatch_tolerance_profile": tolerance_profile,
        "cra_mismatch_pass_tolerance_deg": pass_tolerance_deg,
        "cra_mismatch_check_tolerance_deg": check_tolerance_deg,
        "cra_mismatch_gate": cra_mismatch_gate(total, pass_tolerance_deg, check_tolerance_deg),
    }


def field_design_cases(
    slug: str,
    catalog_row: dict[str, Any],
    stack: dict[str, Any],
    mode: str,
    field_overrides: list[dict[str, Any]] | None = None,
    field_override_source: str = "",
) -> list[dict[str, Any]]:
    pitch = safe_float(catalog_row.get("pixel_pitch_um"), math.nan)
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    if not math.isfinite(pitch) or pitch <= 0:
        pitch = safe_float(geometry.get("pitch"), 1.0)
    focus_depth = estimated_focus_depth_um(catalog_row, stack)
    n_eff = lens_index_prior(stack)
    group_factor = ocl_supercell_factor(mode)
    wavelength_set = ";".join(str(value) for value in DEFAULT_WAVELENGTHS_NM)
    prior_rows = []
    for label, field_x, field_z in DEFAULT_FIELD_COORDS:
        cra_x = DEFAULT_EDGE_CRA_DEG * field_x
        cra_z = DEFAULT_EDGE_CRA_DEG * field_z
        lens_cra_total = math.hypot(cra_x, cra_z)
        tolerance_profile, pass_tol, check_tol = cra_mismatch_tolerance_policy(
            catalog_row,
            lens_cra_total_deg=lens_cra_total,
        )
        mismatch = cra_mismatch_fields(
            lens_cra_x_deg=cra_x,
            lens_cra_z_deg=cra_z,
            sensor_cra_x_deg=math.nan,
            sensor_cra_z_deg=math.nan,
            tolerance_profile=tolerance_profile,
            pass_tolerance_deg=pass_tol,
            check_tolerance_deg=check_tol,
        )
        shift_x, cap = lens_shift_prior_um(cra_x, focus_depth, pitch, n_eff)
        shift_z, _cap = lens_shift_prior_um(cra_z, focus_depth, pitch, n_eff)
        prior_rows.append(
            {
                "slug": slug,
                "code": catalog_row.get("code", ""),
                "field_case": label,
                "field_x_norm": field_x,
                "field_z_norm": field_z,
                "wavelength_set_nm": wavelength_set,
                "cra_x_deg": cra_x,
                "cra_z_deg": cra_z,
                **mismatch,
                "lens_shift_x_um": shift_x,
                "lens_shift_z_um": shift_z,
                "lens_shift_model": "design_prior: tan(CRA)*focus_depth/n_eff clipped_to_0.35_pitch",
                "edge_cra_assumption_deg": DEFAULT_EDGE_CRA_DEG,
                "focus_target_depth_um": focus_depth,
                "lens_shift_cap_um": cap,
                "ocl_mode_guess": mode,
                "ocl_supercell_pitch_um": pitch * group_factor,
                "measurement_gate": "ASSUMED_NOT_MEASURED",
                "source": "TechInsights metadata plus default camera field prior; replace with measured CRA and ML offset map.",
            }
        )
    if not field_overrides:
        return prior_rows

    prior_by_case = {str(row["field_case"]): row for row in prior_rows}
    override_rows = []
    for override in field_overrides:
        field_case = str(override.get("field_case") or "").strip()
        base = dict(prior_by_case.get(field_case) or prior_rows[0])
        field_x = safe_float(override.get("field_x_norm"), safe_float(base.get("field_x_norm"), 0.0))
        field_z = safe_float(override.get("field_z_norm"), safe_float(base.get("field_z_norm"), 0.0))
        cra_x = safe_float(override.get("cra_x_deg"), safe_float(base.get("cra_x_deg"), 0.0))
        cra_z = safe_float(override.get("cra_z_deg"), safe_float(base.get("cra_z_deg"), 0.0))
        lens_cra_x = safe_float(override.get("lens_cra_x_deg"), cra_x)
        lens_cra_z = safe_float(override.get("lens_cra_z_deg"), cra_z)
        sensor_cra_x = safe_float(override.get("sensor_cra_x_deg"), math.nan)
        sensor_cra_z = safe_float(override.get("sensor_cra_z_deg"), math.nan)
        tolerance_profile, pass_tol, check_tol = cra_mismatch_tolerance_policy(
            catalog_row,
            lens_cra_total_deg=math.hypot(lens_cra_x, lens_cra_z),
            profile_override=override.get("cra_mismatch_tolerance_profile", ""),
            pass_override=override.get("cra_mismatch_pass_tolerance_deg", ""),
            check_override=override.get("cra_mismatch_check_tolerance_deg", ""),
        )
        mismatch = cra_mismatch_fields(
            lens_cra_x_deg=lens_cra_x,
            lens_cra_z_deg=lens_cra_z,
            sensor_cra_x_deg=sensor_cra_x,
            sensor_cra_z_deg=sensor_cra_z,
            tolerance_profile=tolerance_profile,
            pass_tolerance_deg=pass_tol,
            check_tolerance_deg=check_tol,
        )
        shift_x = safe_float(override.get("lens_shift_x_um"), math.nan)
        shift_z = safe_float(override.get("lens_shift_z_um"), math.nan)
        cap = safe_float(base.get("lens_shift_cap_um"), 0.35 * pitch)
        if not math.isfinite(shift_x):
            shift_x, cap = lens_shift_prior_um(cra_x, focus_depth, pitch, n_eff)
        if not math.isfinite(shift_z):
            shift_z, _cap = lens_shift_prior_um(cra_z, focus_depth, pitch, n_eff)
        gate = normalized_gate(override.get("measurement_gate"), "IMPORTED_NOT_VALIDATED")
        source = str(override.get("source") or field_override_source or "camera_module_field_map_csv")
        override_rows.append(
            {
                "slug": slug,
                "code": override.get("code") or catalog_row.get("code", ""),
                "field_case": field_case or base.get("field_case", ""),
                "field_x_norm": field_x,
                "field_z_norm": field_z,
                "wavelength_set_nm": override.get("wavelength_set_nm") or wavelength_set,
                "cra_x_deg": cra_x,
                "cra_z_deg": cra_z,
                **mismatch,
                "lens_shift_x_um": max(-cap, min(cap, shift_x)),
                "lens_shift_z_um": max(-cap, min(cap, shift_z)),
                "lens_shift_model": override.get("lens_shift_model")
                or "camera_module_field_map import; missing shifts derived by tan(CRA)*focus_depth/n_eff clipped_to_0.35_pitch",
                "edge_cra_assumption_deg": override.get("edge_cra_assumption_deg") or "",
                "focus_target_depth_um": override.get("focus_target_depth_um") or focus_depth,
                "lens_shift_cap_um": cap,
                "ocl_mode_guess": override.get("ocl_mode_guess") or mode,
                "ocl_supercell_pitch_um": override.get("ocl_supercell_pitch_um") or pitch * group_factor,
                "measurement_gate": gate,
                "source": source,
            }
        )
    return override_rows


def crosstalk_requirement_for_mode(mode: str) -> tuple[int, int, str, int]:
    if mode == "quad_2x2":
        return 5, 9, "18x18 raw pixels / 9x9 2x2-OCL cells for 5x5 output kernel plus 2-cell guard", 2
    if mode == "nona_3x3":
        return 7, 11, "33x33 raw pixels / 11x11 3x3-OCL cells for 7x7 output kernel plus 2-cell guard", 2
    return 3, 5, "5x5 raw pixels for 3x3 output kernel plus 1-cell guard", 1


def crosstalk_fdtd_domain_factor(mode: str, simulation_neighborhood: int) -> float:
    raw_pixel_width = max(1, int(simulation_neighborhood)) * ocl_supercell_factor(mode)
    return float(raw_pixel_width * raw_pixel_width)


def required_runs(
    slug: str,
    catalog_row: dict[str, Any],
    mode: str,
    tcad_status: str,
    crosstalk_status: str,
    material_status: str,
    cra_input_status: str,
    fdtd_status: str,
    field_case_count: int,
    target_resolution_px_per_um: int,
    target_resolution_reason: str,
    fdtd_cell_volume_um3: float,
) -> list[dict[str, Any]]:
    wavelength_set = ";".join(str(value) for value in DEFAULT_WAVELENGTHS_NM)
    wavelength_cli = ",".join(str(value) for value in DEFAULT_WAVELENGTHS_NM)
    color_channels = sensor_solver_channels(catalog_row)
    color_channel_set = ";".join(color_channels)
    color_cli = ",".join(color_channels)
    volume_factor = max(0.05, fdtd_cell_volume_um3 / DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3)
    neighborhood, simulation_neighborhood, raw_domain, guard_cells = crosstalk_requirement_for_mode(mode)
    field_domain_factor = 1.0
    crosstalk_domain_factor = crosstalk_fdtd_domain_factor(mode, simulation_neighborhood)
    command_case_format = (
        "case format: name:cra_x:cra_z:field_x:field_z:lens_shift_x:lens_shift_z "
        "from camera_e2e_field_design_cases.csv"
    )
    rows = [
        {
            "slug": slug,
            "code": catalog_row.get("code", ""),
            "requirement_id": "fdtd_cra_rgb_field_sweep",
            "solver": "Meep FDTD",
            "priority": "P0",
            "status": "DONE" if fdtd_status == "PASS" and cra_input_status == "PASS" else "REQUIRED",
            "current_gate": f"FDTD_{fdtd_status}; CRA_INPUT_{cra_input_status}",
            "why_required": "CameraE2E needs field-dependent response versus CRA and microlens shift, not only TCAD lateral generation shifts.",
            "color_channels": color_channel_set,
            "field_case_count": field_case_count,
            "wavelength_set_nm": wavelength_set,
            "target_resolution_px_per_um": target_resolution_px_per_um,
            "fdtd_cell_volume_um3": f"{fdtd_cell_volume_um3:.6g}",
            "estimated_volume_factor": f"{volume_factor:.3f}",
            "fdtd_domain_factor": f"{field_domain_factor:.3f}",
            "required_neighborhood": "",
            "required_simulation_neighborhood": "",
            "raw_pixel_domain": "single/supercell optical domain per field case",
            "guard_cells": "",
            "command_hint": (
                f"python3 run_camera_e2e_fdtd_field_sweep.py --tier quantitative --slugs {slug}; "
                f"target resolution {target_resolution_px_per_um} px/um ({target_resolution_reason}); "
                f"channels={color_cli}; volume_factor={volume_factor:.2f}; "
                f"domain_factor={field_domain_factor:.0f}; "
                f"underlying cases use {command_case_format}."
            ),
        },
        {
            "slug": slug,
            "code": catalog_row.get("code", ""),
            "requirement_id": "fdtd_crosstalk_kernel_convergence",
            "solver": "Meep FDTD",
            "priority": "P0",
            "status": "REQUIRED" if crosstalk_status != "PASS" else "DONE",
            "current_gate": crosstalk_status,
            "why_required": "Binning/OCL CameraE2E kernels must include neighboring pixels and pass truncation/grid convergence.",
            "color_channels": color_channel_set,
            "field_case_count": field_case_count,
            "wavelength_set_nm": wavelength_set,
            "target_resolution_px_per_um": target_resolution_px_per_um,
            "fdtd_cell_volume_um3": f"{fdtd_cell_volume_um3:.6g}",
            "estimated_volume_factor": f"{volume_factor:.3f}",
            "fdtd_domain_factor": f"{crosstalk_domain_factor:.3f}",
            "required_neighborhood": neighborhood,
            "required_simulation_neighborhood": simulation_neighborhood,
            "raw_pixel_domain": raw_domain,
            "guard_cells": guard_cells,
            "command_hint": (
                f"python3 run_camera_e2e_crosstalk_sweep.py --tier quantitative --slugs {slug}; "
                f"target resolution {target_resolution_px_per_um} px/um ({target_resolution_reason}); "
                f"underlying crosstalk target uses {mode}, neighborhood={neighborhood}, "
                f"simulation_neighborhood target {simulation_neighborhood}, guard={guard_cells}, "
                f"channels={color_cli}; wavelengths={wavelength_cli}; volume_factor={volume_factor:.2f}; "
                f"domain_factor={crosstalk_domain_factor:.0f}."
            ),
        },
        {
            "slug": slug,
            "code": catalog_row.get("code", ""),
            "requirement_id": "tcad_electrical_collection_response",
            "solver": "DEVSIM drift-diffusion",
            "priority": "P1",
            "status": "DONE_CHECK_GATED" if tcad_status in {"PASS", "CHECK"} else "BLOCKED",
            "current_gate": tcad_status,
            "why_required": "Optical absorption must be coupled to electrical collection for split-PD/PDAF and depth-dependent QE.",
            "color_channels": "",
            "field_case_count": "",
            "wavelength_set_nm": "",
            "target_resolution_px_per_um": "",
            "fdtd_cell_volume_um3": "",
            "estimated_volume_factor": "",
            "fdtd_domain_factor": "",
            "required_neighborhood": "",
            "required_simulation_neighborhood": "",
            "raw_pixel_domain": "2D proxy profile currently; real deck needed for product use",
            "guard_cells": "",
            "command_hint": "Use run_image_sensor_db_tcad_major.py results as current proxy; replace with calibrated measured TCAD deck when available.",
        },
        {
            "slug": slug,
            "code": catalog_row.get("code", ""),
            "requirement_id": "measured_stack_material_import",
            "solver": "data import / validation",
            "priority": "P0",
            "status": "REQUIRED" if material_status != "PASS" else "DONE",
            "current_gate": material_status,
            "why_required": "Without measured geometry and measured n,k, CameraE2E LUT cannot be treated as sensor-accurate.",
            "color_channels": color_channel_set,
            "field_case_count": "",
            "wavelength_set_nm": wavelength_set,
            "target_resolution_px_per_um": "",
            "fdtd_cell_volume_um3": f"{fdtd_cell_volume_um3:.6g}",
            "estimated_volume_factor": f"{volume_factor:.3f}",
            "fdtd_domain_factor": "",
            "required_neighborhood": "",
            "required_simulation_neighborhood": "",
            "raw_pixel_domain": "",
            "guard_cells": "",
            "command_hint": "Import measured layer geometry and RGB/OCL/passivation/silicon n,k tables, then rerun FDTD/TCAD gates.",
        },
    ]
    return rows


def write_html_report(
    output_dir: Path,
    package: dict[str, Any],
    index_rows: list[dict[str, Any]],
    solver_coverage_rows: list[dict[str, Any]],
    quantitative_coverage_rows: list[dict[str, Any]],
    quantitative_plan: list[dict[str, Any]],
    quantitative_queue: list[dict[str, Any]],
    resource_limited_batch: list[dict[str, Any]],
    field_design_rows: list[dict[str, Any]],
    xt_rows: list[dict[str, Any]],
    required_rows: list[dict[str, Any]],
) -> None:
    status_counts: dict[str, int] = {}
    for row in required_rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    blockers = [
        "TechInsights-derived data does not contain measured CRA/field angle or microlens offset maps.",
        "Current crosstalk evidence is low-resolution/check-only and fails convergence for CameraE2E product use.",
        "Measured stack geometry and measured RGB/OCL/passivation/silicon n,k are not present.",
        "OX08D10 is LOFIC/HDR/TheiaCel-like and needs a dedicated TCAD deck beyond the current split-PD proxy.",
    ]
    latest_rows = []
    field_lut_rows = read_csv(output_dir / "camera_e2e_quantitative_field_lut.csv")
    crosstalk_lut_rows = read_csv(output_dir / "camera_e2e_quantitative_crosstalk_lut.csv")
    field_map_validation_rows = read_csv(output_dir / "camera_module_field_map_validation.csv")
    export_manifest_rows = read_csv(output_dir / "camera_e2e_export_manifest.csv")
    latest_field = package.get("latest_fdtd_field_sweep")
    if isinstance(latest_field, dict):
        latest_rows.append(
            {
                "evidence": "field_response_fdtd",
                "tier": latest_field.get("tier", ""),
                "completed_jobs": latest_field.get("completed_job_count", ""),
                "summary_rows": latest_field.get("summary_row_count", ""),
                "product_lut_ready": latest_field.get("product_lut_ready", False),
                "html": package.get("outputs", {}).get("latest_fdtd_field_sweep_html", ""),
            }
        )
    latest_crosstalk = package.get("latest_crosstalk_sweep")
    if isinstance(latest_crosstalk, dict):
        latest_rows.append(
            {
                "evidence": "crosstalk_fdtd",
                "tier": latest_crosstalk.get("tier", ""),
                "completed_jobs": latest_crosstalk.get("completed_job_count", ""),
                "summary_rows": latest_crosstalk.get("summary_row_count", ""),
                "product_lut_ready": latest_crosstalk.get("product_lut_ready", False),
                "html": package.get("outputs", {}).get("latest_crosstalk_sweep_html", ""),
            }
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Sensor LUT Package</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #081118;
      --panel: #0e1b25;
      --line: #244255;
      --text: #e5f3ff;
      --muted: #99b2c4;
      --cyan: #52e1ff;
      --yellow: #ffd85a;
      --red: #ff7b7b;
      --green: #7de782;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 10px; letter-spacing: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 19px; margin-top: 28px; color: var(--cyan); }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 26px; font-weight: 800; }}
    .label {{ color: var(--muted); font-size: 13px; }}
    .blocked {{ color: var(--red); }}
    .check {{ color: var(--yellow); }}
    .pass {{ color: var(--green); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--cyan); background: #102633; position: sticky; top: 0; }}
    code {{ color: #d8f8ff; }}
    ul {{ color: var(--muted); line-height: 1.55; }}
    .note {{ border-left: 3px solid var(--yellow); padding-left: 12px; color: var(--text); }}
  </style>
</head>
<body>
<main>
  <h1>CameraE2E Sensor LUT Package</h1>
  <p>Generated: <code>{html_cell(package.get("generated_at", ""))}</code></p>
  <div class="grid">
    <div class="card"><div class="metric">{len(index_rows)}</div><div class="label">sensors packaged</div></div>
    <div class="card"><div class="metric blocked">{package.get("camera_e2e_ready_count", 0)}</div><div class="label">CameraE2E-ready sensors</div></div>
    <div class="card"><div class="metric">{len(field_design_rows)}</div><div class="label">CRA/ML-shift field cases</div></div>
    <div class="card"><div class="metric">{len(required_rows)}</div><div class="label">required run items</div></div>
  </div>

  <h2>Readiness</h2>
  <p class="note">Current package is an ingestible research scaffold, not a product-ready CameraE2E LUT. Field CRA and microlens shifts come from <code>--field-map-csv</code> when supplied; otherwise they are generated design priors because the TechInsights-derived DB has no measured CRA or field-offset map.</p>
  <ul>
    {''.join(f"<li>{html.escape(item)}</li>" for item in blockers)}
  </ul>

  <h2>CRA / ML-Shift Input</h2>
  <p>Input path: <code>{html_cell(package.get("field_map_input", {}).get("path", ""))}</code>;
  exists: <code>{html_cell(package.get("field_map_input", {}).get("exists", ""))}</code>;
  overridden sensors: <code>{html_cell(package.get("field_map_input", {}).get("sensor_override_count", ""))}</code>.
  PASS requires all imported rows for a sensor to use one of
	  <code>{html_cell(package.get("field_map_input", {}).get("pass_gates", []))}</code>.</p>
	  <p>Validation gate: <code>{html_cell(package.get("field_map_validation", {}).get("gate", ""))}</code>;
	  validation report: <code>{html_cell(package.get("outputs", {}).get("field_map_validation_html", ""))}</code>;
	  prior seed CSV: <code>{html_cell(package.get("outputs", {}).get("field_map_prior_seed_csv", ""))}</code>.</p>
	  {html_table(field_map_validation_rows, FIELD_MAP_VALIDATION_COLUMNS, limit=16) if field_map_validation_rows else "<p>No field-map validation rows have been generated.</p>"}

	  <h2>CameraE2E Export Manifest</h2>
	  <p>Camera-system consumers should use this manifest to separate research-ingest artifacts from production-ingest artifacts.</p>
	  {html_table(export_manifest_rows, CAMERA_E2E_EXPORT_MANIFEST_COLUMNS) if export_manifest_rows else "<p>No export manifest has been generated.</p>"}

  <h2>Latest Solver Evidence</h2>
  <p>These are the latest executed solver runs linked into the package. Dry-run plans are intentionally not treated as latest evidence.</p>
  {html_table(latest_rows, ["evidence", "tier", "completed_jobs", "summary_rows", "product_lut_ready", "html"]) if latest_rows else "<p>No executed field/crosstalk sweep evidence is linked yet.</p>"}

  <h2>Sensor Gates</h2>
  {html_table(index_rows, INDEX_COLUMNS)}

  <h2>Solver Coverage</h2>
  <p>This table shows per-sensor executed solver evidence. A smoke-tier CHECK means the pipeline ran, but grid/convergence is not product-ready.</p>
  {html_table(solver_coverage_rows, SOLVER_COVERAGE_COLUMNS)}

  <h2>Merged Quantitative Coverage</h2>
  <p>This table is generated by <code>merge_camera_e2e_quantitative_points.py</code>. Product gates use this merged quantitative coverage when present.</p>
  {html_table(quantitative_coverage_rows, QUANTITATIVE_COVERAGE_COLUMNS) if quantitative_coverage_rows else "<p>No merged quantitative coverage file has been generated yet.</p>"}

  <h2>Quantitative KPI LUTs</h2>
  <p>These CSVs carry the numerical field-response and crosstalk KPI values that CameraE2E can ingest after the corresponding gates pass.</p>
  <p>Field LUT: <code>{html_cell(package.get("outputs", {}).get("quantitative_field_lut_csv", ""))}</code><br>
  Crosstalk LUT: <code>{html_cell(package.get("outputs", {}).get("quantitative_crosstalk_lut_csv", ""))}</code></p>
  {html_table(field_lut_rows, [column for column in ["slug", "color", "field_case", "wavelength_nm", "cra_x_deg", "lens_shift_x_um", "total_response", "focal_centroid_shift_x_um", "solver_gate"]], limit=12) if field_lut_rows else "<p>No quantitative field KPI rows have been merged yet.</p>"}
  {html_table(crosstalk_lut_rows, [column for column in ["slug", "color", "field_case", "wavelength_nm", "output_crosstalk_fraction", "estimated_voxels", "resource_gate", "convergence_status", "solver_gate"]], limit=12) if crosstalk_lut_rows else "<p>No quantitative crosstalk KPI rows have been merged yet.</p>"}

  <h2>Quantitative Execution Plan</h2>
  <p>This estimate uses the completed HP5 550nm center-point pilot when available and scales runtime roughly as resolution to the fourth power. Treat it as scheduling guidance, not a physics result.</p>
  {html_table(quantitative_plan, QUANTITATIVE_PLAN_COLUMNS)}

  <h2>Quantitative Point Queue</h2>
  <p>Full quantitative work is split into point-sized commands so long runs can be scheduled, resumed, and merged without losing prior evidence.</p>
  {html_table(quantitative_queue, QUANTITATIVE_QUEUE_COLUMNS, limit=24)}

  <h2>Resource-Limited Batch Plan</h2>
  <p>These points exceeded the local interactive voxel limit. They are not completed solver points; run the batch command on a larger workstation or cluster, then merge the resulting evidence.</p>
  {html_table(resource_limited_batch, RESOURCE_LIMITED_BATCH_COLUMNS) if resource_limited_batch else "<p>No resource-limited quantitative points have been recorded.</p>"}

  <h2>Required Runs</h2>
  <p>Status counts: <code>{html_cell(dict(sorted(status_counts.items())))}</code></p>
  {html_table(required_rows, REQUIRED_RUN_COLUMNS)}

  <h2>Crosstalk Evidence</h2>
  {html_table(xt_rows, CROSSTALK_COLUMNS)}

  <h2>Field CRA / Microlens Shift Design Cases</h2>
  <p>Each sensor receives field anchors for the Meep sweep. If no module field map is imported, these are default prior cases. Replace them with camera-module CRA and microlens shift data before using the output as a CameraE2E product LUT.</p>
  {html_table(field_design_rows, FIELD_DESIGN_COLUMNS, limit=36)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def crosstalk_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    summaries = payload.get("summaries")
    if isinstance(summaries, list) and summaries:
        summary = dict(summaries[0])
    else:
        summary = {}
    convergence = payload.get("convergence")
    if isinstance(convergence, dict):
        summary["convergence_status"] = convergence.get("status", payload.get("convergence_status", ""))
    else:
        summary["convergence_status"] = payload.get("convergence_status", "")
    return summary


def crosstalk_gate(archetype: str) -> tuple[str, dict[str, Any] | None, str]:
    path = ARCHETYPE_CROSSTALK.get(archetype)
    if not path:
        return "MISSING", None, ""
    summary = crosstalk_summary(path)
    if not summary:
        return "MISSING", None, repo_rel(path)
    grid_pass = summary.get("grid_resolution_gate_pass") is True
    convergence_status = str(summary.get("convergence_status") or "").upper()
    truncation = safe_float(summary.get("truncation_response_fraction"))
    if grid_pass and convergence_status == "PASS" and math.isfinite(truncation) and truncation <= 0.015:
        return "PASS", summary, repo_rel(path)
    return "FAIL" if convergence_status == "FAIL" or not grid_pass else "CHECK", summary, repo_rel(path)


def camera_e2e_gate(
    *,
    tcad: str,
    crosstalk: str,
    material: str,
    has_true_cra: bool,
) -> tuple[str, bool, str]:
    blockers = []
    if tcad not in {"PASS", "CHECK"}:
        blockers.append(f"TCAD gate is {tcad}")
    if crosstalk != "PASS":
        blockers.append(f"crosstalk gate is {crosstalk}")
    if material != "PASS":
        blockers.append("measured stack/material gate is not PASS")
    if not has_true_cra:
        blockers.append("true CRA/field sweep LUT is missing")
    if blockers:
        return "BLOCKED", False, "; ".join(blockers)
    return "PASS", True, ""


def build_sensor_lut(
    *,
    slug: str,
    catalog_row: dict[str, Any],
    stack: dict[str, Any],
    profile_path: Path | None,
    tcad_rows: list[dict[str, Any]],
    tcad_summary: dict[str, Any] | None,
    output_dir: Path,
    field_overrides: list[dict[str, Any]],
    field_override_source: str,
    quantitative_coverage: dict[tuple[str, str], dict[str, str]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sensor_dir = output_dir / "sensors" / slug
    lut_path = sensor_dir / "camera_e2e_lut.json"
    previous_lut = read_json(lut_path) if lut_path.exists() else {}
    previous_field_evidence = previous_lut.get("fdtd_field_sweep_evidence")
    if not isinstance(previous_field_evidence, dict):
        previous_field_evidence = None
    previous_crosstalk_evidence = previous_lut.get("crosstalk_sweep_evidence")
    if not isinstance(previous_crosstalk_evidence, dict):
        previous_crosstalk_evidence = None
    quantitative_field_evidence = quantitative_evidence_from_coverage(
        quantitative_coverage.get((slug, "field")),
        schema="camera_e2e_fdtd_field_sweep_evidence_v1",
    )
    quantitative_crosstalk_evidence = quantitative_evidence_from_coverage(
        quantitative_coverage.get((slug, "crosstalk")),
        schema="camera_e2e_crosstalk_sweep_evidence_v1",
    )
    product_field_evidence = quantitative_field_evidence or previous_field_evidence
    product_crosstalk_evidence = quantitative_crosstalk_evidence or previous_crosstalk_evidence

    mode = ocl_mode_guess(catalog_row, stack)
    tcad_status, tcad_reason = tcad_gate(tcad_rows, tcad_summary)
    material_status, material_reasons = material_measurement_gate(stack)
    archetype_crosstalk_status, xt_summary, xt_path = crosstalk_gate(mode)
    crosstalk_evidence_status, crosstalk_evidence_reason = solver_evidence_gate(product_crosstalk_evidence)
    if quantitative_crosstalk_evidence is not None:
        crosstalk_status = crosstalk_evidence_status
    elif crosstalk_evidence_status != "MISSING":
        crosstalk_status = crosstalk_evidence_status
    else:
        crosstalk_status = archetype_crosstalk_status
    target_resolution, target_resolution_reason = recommended_fdtd_resolution_px_per_um(catalog_row, stack)
    fields = field_rows_from_tcad(slug, catalog_row, tcad_rows)
    field_design = field_design_cases(
        slug,
        catalog_row,
        stack,
        mode,
        field_overrides=field_overrides,
        field_override_source=field_override_source,
    )
    cra_input_status, cra_input_reason = field_input_gate(field_design)
    fdtd_field_status, fdtd_field_reason = solver_evidence_gate(product_field_evidence)
    cra_status, has_true_cra, cra_status_reason = cra_field_gate(cra_input_status, fdtd_field_status)
    fdtd_cell_volume_um3 = estimated_fdtd_periodic_cell_volume_um3(stack)
    required = required_runs(
        slug,
        catalog_row,
        mode,
        tcad_status,
        crosstalk_status,
        material_status,
        cra_input_status,
        fdtd_field_status,
        len(field_design),
        target_resolution,
        target_resolution_reason,
        fdtd_cell_volume_um3,
    )
    usage_gate, ready, reason = camera_e2e_gate(
        tcad=tcad_status,
        crosstalk=crosstalk_status,
        material=material_status,
        has_true_cra=has_true_cra,
    )
    if not reason and tcad_reason:
        reason = tcad_reason
    geometry = stack.get("geometry_um", {})
    source = stack.get("techinsights_source", {})
    specs = source.get("derived_specs", {})
    lut = {
        "schema": "camera_e2e_sensor_lut_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage": {
            "camera_e2e_ready": ready,
            "usage_gate": usage_gate,
            "intended_scope": "camera_e2e_research_trend" if not ready else "camera_e2e_lut",
            "reason": reason,
        },
        "sensor": {
            "slug": slug,
            "code": catalog_row.get("code", ""),
            "manufacturer": catalog_row.get("manufacturer", ""),
            "device_name": catalog_row.get("device_name", ""),
            "pixel_pitch_um": safe_float(catalog_row.get("pixel_pitch_um"), ""),
            "pixel_architecture": catalog_row.get("pixel_architecture", ""),
            "cfa_pattern": catalog_row.get("cfa_pattern", ""),
            "ocl_mode_guess": mode,
            "has_hdr": catalog_row.get("has_hdr", ""),
            "has_lofic": catalog_row.get("has_lofic", ""),
            "has_pdaf": catalog_row.get("has_pdaf", ""),
        },
        "geometry": {
            "pitch_um": geometry.get("pitch", specs.get("pixel_pitch_um", "")),
            "si_thickness_um": geometry.get("si_thickness", specs.get("active_si_thickness_um", "")),
            "cfa_thickness_um": geometry.get("cfa_thickness", specs.get("cfa_thickness_um", "")),
            "lens_height_um": geometry.get("lens_height", ""),
            "passivation_thickness_um": geometry.get("passivation_thickness", ""),
        },
        "fdtd_resolution_plan": {
            "target_resolution_px_per_um": target_resolution,
            "reason": target_resolution_reason,
            "min_feature_pixels_required": MIN_FEATURE_PIXELS_REQUIRED,
            "min_si_wavelength_pixels_required": MIN_SI_WAVELENGTH_PIXELS_REQUIRED,
            "estimated_periodic_cell_volume_um3": f"{fdtd_cell_volume_um3:.6g}",
            "estimated_volume_factor_vs_reference": f"{fdtd_cell_volume_um3 / DEFAULT_REFERENCE_FDTD_CELL_VOLUME_UM3:.3f}",
        },
        "gates": {
            "cra_input_gate": cra_input_status,
            "cra_input_reason": cra_input_reason,
            "fdtd_field_sweep_gate": fdtd_field_status,
            "fdtd_field_sweep_reason": fdtd_field_reason,
            "tcad_solver_gate": tcad_status,
            "tcad_reason": tcad_reason,
            "measured_stack_material_gate": material_status,
            "measured_stack_material_reasons": material_reasons,
            "cra_field_lut_gate": cra_status,
            "cra_field_lut_reason": cra_status_reason,
            "crosstalk_sweep_evidence_gate": crosstalk_evidence_status,
            "crosstalk_sweep_evidence_reason": crosstalk_evidence_reason,
            "crosstalk_archetype_gate": archetype_crosstalk_status,
            "crosstalk_kernel_gate": crosstalk_status,
            "camera_e2e_usage_gate": usage_gate,
        },
        "field_response_lut": {
            "schema": "camera_e2e_field_response_proxy_v1",
            "axis_warning": "field_x_norm is derived from TCAD optical-generation lateral shift, not a true CRA ray/field sweep.",
            "rows": fields,
        },
        "field_design_cases": {
            "schema": "camera_e2e_field_design_cases_v1",
            "measurement_gate": cra_input_status,
            "measurement_reason": cra_input_reason,
            "purpose": "Run these CRA and microlens-shift cases in Meep before CameraE2E product use.",
            "rows": field_design,
        },
        "crosstalk_lut": {
            "schema": "camera_e2e_crosstalk_reference_v1",
            "archetype": mode,
            "source_path": xt_path,
            "archetype_gate": archetype_crosstalk_status,
            "executed_sweep_gate": crosstalk_evidence_status,
            "summary": xt_summary or {},
            "warning": "Archetype crosstalk is not sensor-specific unless measured/CAD stack and convergence gate pass.",
        },
        "source_artifacts": {
            "stack_config": repo_rel(DEFAULT_STACK_DIR / f"{slug}.json"),
            "tcad_profile": repo_rel(profile_path) if profile_path and profile_path.exists() else "",
            "tcad_major_report": repo_rel(DEFAULT_TCAD_MAJOR_REPORT) if DEFAULT_TCAD_MAJOR_REPORT.exists() else "",
        },
        "limitations": [
            "TechInsights-derived stack/profile values are metadata/proxy values, not measured process decks.",
            "Current field response rows are electrical lateral-shift proxies, not full optical CRA G(x,y,z) LUT rows.",
            "The field_design_cases CRA and microlens shifts are product-usable only when cra_input_gate PASS and FDTD field sweep convergence PASS.",
            "CameraE2E product use requires measured stack n,k, true CRA field sweep, crosstalk convergence pass, and calibration targets.",
        ],
    }
    if product_field_evidence:
        lut["fdtd_field_sweep_evidence"] = product_field_evidence
    if previous_field_evidence and previous_field_evidence is not product_field_evidence:
        lut["latest_fdtd_field_sweep_evidence"] = previous_field_evidence
    if product_crosstalk_evidence:
        lut["crosstalk_sweep_evidence"] = product_crosstalk_evidence
    if previous_crosstalk_evidence and previous_crosstalk_evidence is not product_crosstalk_evidence:
        lut["latest_crosstalk_sweep_evidence"] = previous_crosstalk_evidence
    write_json(lut_path, lut)
    index_row = {
        "slug": slug,
        "code": catalog_row.get("code", ""),
        "manufacturer": catalog_row.get("manufacturer", ""),
        "device_name": catalog_row.get("device_name", ""),
        "pixel_pitch_um": catalog_row.get("pixel_pitch_um", ""),
        "pixel_architecture": catalog_row.get("pixel_architecture", ""),
        "cfa_pattern": catalog_row.get("cfa_pattern", ""),
        "ocl_mode_guess": mode,
        "cra_input_gate": cra_input_status,
        "fdtd_field_sweep_gate": fdtd_field_status,
        "tcad_gate": tcad_status,
        "crosstalk_gate": crosstalk_status,
        "cra_field_gate": cra_status,
        "measured_stack_gate": material_status,
        "camera_e2e_usage_gate": usage_gate,
        "camera_e2e_ready": ready,
        "reason": reason,
        "lut_json": repo_rel(lut_path),
    }
    xt_row = {
        "slug": slug,
        "code": catalog_row.get("code", ""),
        "archetype": mode,
        "source_path": xt_path,
        "archetype_gate": archetype_crosstalk_status,
        "executed_sweep_gate": crosstalk_evidence_status,
        "executed_sweep_summary_csv": (previous_crosstalk_evidence or {}).get("summary_csv", ""),
        "mode": (xt_summary or {}).get("mode", ""),
        "neighborhood": (xt_summary or {}).get("neighborhood", ""),
        "simulation_neighborhood": (xt_summary or {}).get("simulation_neighborhood", ""),
        "raw_pd_kernel_shape": (xt_summary or {}).get("raw_pd_kernel_shape", ""),
        "output_crosstalk_fraction": (xt_summary or {}).get("output_crosstalk_fraction", ""),
        "strongest_neighbor_fraction": (xt_summary or {}).get("strongest_neighbor_fraction", ""),
        "truncation_response_fraction": (xt_summary or {}).get("truncation_response_fraction", ""),
        "grid_resolution_gate_pass": (xt_summary or {}).get("grid_resolution_gate_pass", ""),
        "convergence_status": (xt_summary or {}).get("convergence_status", ""),
        "usage_gate": crosstalk_status,
    }
    return index_row, fields, xt_row, lut, field_design, required


def selected_slugs(catalog: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[str]:
    if args.slugs:
        return [item.strip() for item in args.slugs.split(",") if item.strip()]
    if args.major_only and DEFAULT_TCAD_MAJOR_REPORT.exists():
        report = read_json(DEFAULT_TCAD_MAJOR_REPORT)
        return [str(item) for item in report.get("settings", {}).get("slugs", [])]
    return sorted(catalog)


def build(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.sensor_catalog)
    tcad_by_sensor, tcad_summary = load_tcad_rows(args.tcad_major_report)
    field_map_overrides = load_field_map_overrides(args.field_map_csv)
    field_map_source = repo_rel(args.field_map_csv) if args.field_map_csv.exists() else ""
    output_dir = args.output_dir.resolve()
    quantitative_coverage = load_quantitative_coverage(output_dir)
    index_rows: list[dict[str, Any]] = []
    solver_coverage_rows: list[dict[str, Any]] = []
    all_field_rows: list[dict[str, Any]] = []
    all_field_design_rows: list[dict[str, Any]] = []
    all_required_rows: list[dict[str, Any]] = []
    xt_rows: list[dict[str, Any]] = []
    quantitative_coverage_rows = [
        row for _key, row in sorted(quantitative_coverage.items(), key=lambda item: item[0])
    ]
    selected = selected_slugs(catalog, args)
    for slug in selected:
        stack_path = args.stack_dir / f"{slug}.json"
        if not stack_path.exists():
            continue
        stack = read_json(stack_path)
        catalog_row = catalog.get(slug, {})
        profile_path = args.tcad_profile_dir / slug / "profile.json"
        index_row, field_rows, xt_row, lut, field_design_rows, required_rows = build_sensor_lut(
            slug=slug,
            catalog_row=catalog_row,
            stack=stack,
            profile_path=profile_path,
            tcad_rows=tcad_by_sensor.get(slug, []),
            tcad_summary=tcad_summary.get(slug),
            output_dir=output_dir,
            field_overrides=field_map_overrides.get(slug, []),
            field_override_source=field_map_source,
            quantitative_coverage=quantitative_coverage,
        )
        index_rows.append(index_row)
        solver_coverage_rows.append(solver_coverage_row(index_row, lut))
        all_field_rows.extend(field_rows)
        all_field_design_rows.extend(field_design_rows)
        all_required_rows.extend(required_rows)
        xt_rows.append(xt_row)
    ready_count = sum(1 for row in index_rows if row.get("camera_e2e_ready") is True)
    gate_counts: dict[str, int] = {}
    for row in index_rows:
        gate = str(row.get("camera_e2e_usage_gate", ""))
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    package = {
        "schema": "camera_e2e_sensor_lut_package_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor_count": len(index_rows),
        "camera_e2e_ready_count": ready_count,
        "usage_gate_counts": dict(sorted(gate_counts.items())),
        "selected_slugs": selected,
        "outputs": {
            "html_report": repo_rel(output_dir / "index.html"),
            "index_csv": repo_rel(output_dir / "camera_e2e_sensor_index.csv"),
            "solver_coverage_csv": repo_rel(output_dir / "camera_e2e_solver_coverage.csv"),
            "quantitative_execution_plan_csv": repo_rel(output_dir / "camera_e2e_quantitative_execution_plan.csv"),
            "quantitative_point_queue_csv": repo_rel(output_dir / "camera_e2e_quantitative_point_queue.csv"),
            "resource_limited_batch_plan_csv": repo_rel(output_dir / "camera_e2e_resource_limited_batch_plan.csv"),
            "quantitative_progress_json": repo_rel(output_dir / "camera_e2e_quantitative_progress.json"),
            "quantitative_coverage_csv": repo_rel(output_dir / "camera_e2e_quantitative_coverage.csv"),
            "quantitative_merged_summary_csv": repo_rel(output_dir / "camera_e2e_quantitative_merged_summary.csv"),
            "quantitative_field_lut_csv": repo_rel(output_dir / "camera_e2e_quantitative_field_lut.csv"),
            "quantitative_crosstalk_lut_csv": repo_rel(output_dir / "camera_e2e_quantitative_crosstalk_lut.csv"),
            "field_response_csv": repo_rel(output_dir / "camera_e2e_field_response_proxy.csv"),
            "field_design_cases_csv": repo_rel(output_dir / "camera_e2e_field_design_cases.csv"),
            "field_map_import_template_csv": repo_rel(output_dir / "camera_module_field_map_import_template.csv"),
            "field_map_prior_seed_csv": repo_rel(output_dir / "camera_module_field_map_prior_seed.csv"),
            "field_map_validation_json": repo_rel(output_dir / "camera_module_field_map_validation.json"),
            "field_map_validation_csv": repo_rel(output_dir / "camera_module_field_map_validation.csv"),
            "field_map_validation_html": repo_rel(output_dir / "camera_module_field_map_validation.html"),
            "crosstalk_index_csv": repo_rel(output_dir / "camera_e2e_crosstalk_index.csv"),
            "required_runs_csv": repo_rel(output_dir / "camera_e2e_required_runs.csv"),
            "camera_e2e_export_manifest_json": repo_rel(output_dir / "camera_e2e_export_manifest.json"),
            "camera_e2e_export_manifest_csv": repo_rel(output_dir / "camera_e2e_export_manifest.csv"),
        },
        "field_map_input": {
            "path": repo_rel(args.field_map_csv),
            "exists": args.field_map_csv.exists(),
            "sensor_override_count": len(field_map_overrides),
            "pass_gates": sorted(TRUE_CRA_GATES),
            "required_columns": FIELD_MAP_IMPORT_COLUMNS,
            "optional_columns": OPTIONAL_FIELD_MAP_IMPORT_COLUMNS,
        },
        "notes": [
            "This package is an ingestible CameraE2E scaffold with strict readiness gates.",
            "Field CRA and microlens-shift cases can be imported from --field-map-csv; otherwise they are generated as design priors because the TechInsights-derived DB has no measured CRA/field-offset map.",
            "camera_e2e_ready remains false until measured materials/stack, true CRA field LUT, and crosstalk convergence pass.",
        ],
    }
    field_map_validation, field_map_validation_rows = field_map_validation_report(
        field_map_csv=args.field_map_csv,
        selected=selected,
        catalog=catalog,
        field_map_overrides=field_map_overrides,
        field_design_rows=all_field_design_rows,
    )
    package["field_map_validation"] = {
        "schema": field_map_validation.get("schema"),
        "gate": field_map_validation.get("gate"),
        "product_lut_ready": field_map_validation.get("product_lut_ready"),
        "input_row_count": field_map_validation.get("input_row_count"),
        "validation_row_count": field_map_validation.get("validation_row_count"),
        "blocker_count": field_map_validation.get("blocker_count", ""),
        "warning_count": field_map_validation.get("warning_count", ""),
    }
    latest_fdtd_path = output_dir / "camera_e2e_fdtd_field_sweep_latest.json"
    if latest_fdtd_path.exists():
        latest_fdtd = read_json(latest_fdtd_path)
        package["outputs"]["latest_fdtd_field_sweep_report"] = repo_rel(latest_fdtd_path)
        package["outputs"]["latest_fdtd_field_sweep_html"] = latest_fdtd.get("html_report", "")
        report = latest_fdtd.get("report", {}) if isinstance(latest_fdtd.get("report"), dict) else {}
        package["latest_fdtd_field_sweep"] = {
            "tier": report.get("tier"),
            "summary_row_count": report.get("summary_row_count"),
            "completed_job_count": report.get("completed_job_count"),
            "product_lut_ready": False,
        }
    latest_crosstalk_path = output_dir / "camera_e2e_crosstalk_sweep_latest.json"
    if latest_crosstalk_path.exists():
        latest_crosstalk = read_json(latest_crosstalk_path)
        package["outputs"]["latest_crosstalk_sweep_report"] = repo_rel(latest_crosstalk_path)
        package["outputs"]["latest_crosstalk_sweep_html"] = latest_crosstalk.get("html_report", "")
        report = latest_crosstalk.get("report", {}) if isinstance(latest_crosstalk.get("report"), dict) else {}
        package["latest_crosstalk_sweep"] = {
            "tier": report.get("tier"),
            "summary_row_count": report.get("summary_row_count"),
            "completed_job_count": report.get("completed_job_count"),
            "product_lut_ready": False,
        }
    export_manifest_rows = camera_e2e_export_manifest_rows(package)
    write_json(output_dir / "camera_module_field_map_validation.json", field_map_validation)
    write_csv(
        output_dir / "camera_module_field_map_validation.csv",
        field_map_validation_rows,
        FIELD_MAP_VALIDATION_COLUMNS,
    )
    write_field_map_validation_html(
        output_dir / "camera_module_field_map_validation.html",
        field_map_validation,
        field_map_validation_rows,
    )
    write_json(
        output_dir / "camera_e2e_export_manifest.json",
        {
            "schema": "camera_e2e_export_manifest_v1",
            "generated_at": package.get("generated_at", ""),
            "sensor_count": package.get("sensor_count", 0),
            "camera_e2e_ready_count": package.get("camera_e2e_ready_count", 0),
            "field_map_validation_gate": field_map_validation.get("gate", ""),
            "rows": export_manifest_rows,
        },
    )
    write_csv(
        output_dir / "camera_e2e_export_manifest.csv",
        export_manifest_rows,
        CAMERA_E2E_EXPORT_MANIFEST_COLUMNS,
    )
    write_json(output_dir / "camera_e2e_lut_package.json", package)
    write_csv(output_dir / "camera_e2e_sensor_index.csv", index_rows, INDEX_COLUMNS)
    write_csv(output_dir / "camera_e2e_solver_coverage.csv", solver_coverage_rows, SOLVER_COVERAGE_COLUMNS)
    quantitative_plan = quantitative_plan_rows(all_required_rows, output_dir)
    write_csv(
        output_dir / "camera_e2e_quantitative_execution_plan.csv",
        quantitative_plan,
        QUANTITATIVE_PLAN_COLUMNS,
    )
    quantitative_queue = quantitative_queue_rows(all_required_rows, all_field_design_rows, output_dir)
    write_csv(
        output_dir / "camera_e2e_quantitative_point_queue.csv",
        quantitative_queue,
        QUANTITATIVE_QUEUE_COLUMNS,
    )
    resource_limited_batch = resource_limited_batch_rows(output_dir, quantitative_queue)
    write_csv(
        output_dir / "camera_e2e_resource_limited_batch_plan.csv",
        resource_limited_batch,
        RESOURCE_LIMITED_BATCH_COLUMNS,
    )
    write_csv(output_dir / "camera_e2e_field_response_proxy.csv", all_field_rows, FIELD_COLUMNS)
    write_csv(output_dir / "camera_e2e_field_design_cases.csv", all_field_design_rows, FIELD_DESIGN_COLUMNS)
    prior_seed_rows = []
    for row in all_field_design_rows:
        prior_seed_rows.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "field_case": row.get("field_case", ""),
                "field_x_norm": row.get("field_x_norm", ""),
                "field_z_norm": row.get("field_z_norm", ""),
                "cra_x_deg": row.get("cra_x_deg", ""),
                "cra_z_deg": row.get("cra_z_deg", ""),
                "lens_cra_x_deg": row.get("lens_cra_x_deg", ""),
                "lens_cra_z_deg": row.get("lens_cra_z_deg", ""),
                "sensor_cra_x_deg": row.get("sensor_cra_x_deg", ""),
                "sensor_cra_z_deg": row.get("sensor_cra_z_deg", ""),
                "cra_mismatch_tolerance_profile": row.get("cra_mismatch_tolerance_profile", ""),
                "cra_mismatch_pass_tolerance_deg": row.get("cra_mismatch_pass_tolerance_deg", ""),
                "cra_mismatch_check_tolerance_deg": row.get("cra_mismatch_check_tolerance_deg", ""),
                "lens_shift_x_um": row.get("lens_shift_x_um", ""),
                "lens_shift_z_um": row.get("lens_shift_z_um", ""),
                "wavelength_set_nm": row.get("wavelength_set_nm", ""),
                "measurement_gate": row.get("measurement_gate", ""),
                "source": row.get("source", ""),
            }
        )
    write_csv(output_dir / "camera_module_field_map_prior_seed.csv", prior_seed_rows, FIELD_MAP_OUTPUT_COLUMNS)
    import_template_rows = []
    for row in all_field_design_rows:
        import_template_rows.append(
            {
                "slug": row.get("slug", ""),
                "code": row.get("code", ""),
                "field_case": row.get("field_case", ""),
                "field_x_norm": row.get("field_x_norm", ""),
                "field_z_norm": row.get("field_z_norm", ""),
                "cra_x_deg": "",
                "cra_z_deg": "",
                "lens_cra_x_deg": "",
                "lens_cra_z_deg": "",
                "sensor_cra_x_deg": "",
                "sensor_cra_z_deg": "",
                "cra_mismatch_tolerance_profile": row.get("cra_mismatch_tolerance_profile", ""),
                "cra_mismatch_pass_tolerance_deg": row.get("cra_mismatch_pass_tolerance_deg", ""),
                "cra_mismatch_check_tolerance_deg": row.get("cra_mismatch_check_tolerance_deg", ""),
                "lens_shift_x_um": "",
                "lens_shift_z_um": "",
                "wavelength_set_nm": row.get("wavelength_set_nm", ""),
                "measurement_gate": "MEASURED_OR_CALIBRATED_REQUIRED",
                "source": "replace with module raytrace/measurement/calibration source",
            }
        )
    write_csv(output_dir / "camera_module_field_map_import_template.csv", import_template_rows, FIELD_MAP_OUTPUT_COLUMNS)
    write_csv(output_dir / "camera_e2e_crosstalk_index.csv", xt_rows, CROSSTALK_COLUMNS)
    write_csv(output_dir / "camera_e2e_required_runs.csv", all_required_rows, REQUIRED_RUN_COLUMNS)
    write_html_report(
        output_dir,
        package,
        index_rows,
        solver_coverage_rows,
        quantitative_coverage_rows,
        quantitative_plan,
        quantitative_queue,
        resource_limited_batch,
        all_field_design_rows,
        xt_rows,
        all_required_rows,
    )
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-catalog", type=Path, default=DEFAULT_SENSOR_CATALOG)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--tcad-profile-dir", type=Path, default=DEFAULT_TCAD_PROFILE_DIR)
    parser.add_argument("--tcad-major-report", type=Path, default=DEFAULT_TCAD_MAJOR_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-map-csv", type=Path, default=DEFAULT_FIELD_MAP_CSV)
    parser.add_argument("--slugs", default="")
    parser.add_argument("--major-only", action="store_true")
    args = parser.parse_args()
    package = build(args)
    print(json.dumps(package, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
