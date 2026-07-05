#!/usr/bin/env python3
"""Validate and index camera-system suite export packages.

The workbench suite export is intentionally research/trend evidence. This tool
checks that downstream camera simulation can ingest the package without silently
treating smoke/proxy data as a product LUT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

FIELD_QUERY_COLUMNS = [
    "query_field_x_norm",
    "query_field_z_norm",
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "total_response",
    "edge_to_center",
    "split_phase_x_proxy",
    "split_phase_z_proxy",
    "source_row_count",
    "interpolation_method",
    "product_lut_ready",
]

CROSSTALK_INDEX_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "source_case",
    "region_count",
    "region_ids",
    "cell_count",
    "center_response_fraction",
    "off_center_response_fraction",
    "total_response_fraction",
    "max_neighbor_response_fraction",
    "output_dx_min",
    "output_dx_max",
    "output_dz_min",
    "output_dz_max",
    "summary_output_crosstalk_fraction",
    "summary_strongest_neighbor_fraction",
    "summary_truncation_response_fraction",
    "grid_gate_pass",
    "kpi_status",
    "product_lut_ready",
]

GATE_SUMMARY_COLUMNS = [
    "suite_id",
    "tier",
    "case_id",
    "label",
    "runner",
    "case_status",
    "kpi_status",
    "grid_gate_pass",
    "gate_available",
    "convergence_status",
    "negative_signed_flux_count",
    "measured_accuracy",
    "consumer_gate",
    "product_lut_ready",
    "reason",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = finite_float(chunk)
        if not math.isfinite(value):
            raise ValueError(f"invalid float value: {chunk}")
        values.append(value)
    if not values:
        raise ValueError("at least one field value is required")
    return values


def add_issue(items: list[dict[str, Any]], severity: str, code: str, message: str, **extra: Any) -> None:
    issue = {"severity": severity, "code": code, "message": message}
    issue.update(extra)
    items.append(issue)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def local_path_from_artifact(value: Any, base_dir: Path) -> Path | None:
    text = str(value or "")
    if not text:
        return None
    if text.startswith("/"):
        return (ROOT / text.lstrip("/")).resolve()
    return (base_dir / text).resolve()


def interpolate(points: list[tuple[float, float]], target: float) -> float:
    clean = sorted((x, y) for x, y in points if math.isfinite(x) and math.isfinite(y))
    if not clean:
        return math.nan
    unique: list[tuple[float, float]] = []
    for x_value, y_value in clean:
        if unique and abs(x_value - unique[-1][0]) <= 1e-12:
            unique[-1] = (x_value, y_value)
        else:
            unique.append((x_value, y_value))
    if len(unique) == 1 or target <= unique[0][0]:
        return unique[0][1]
    if target >= unique[-1][0]:
        return unique[-1][1]
    for left, right in zip(unique, unique[1:]):
        x0, y0 = left
        x1, y1 = right
        if x0 <= target <= x1:
            t = (target - x0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else 0.0
            return y0 * (1.0 - t) + y1 * t
    return unique[-1][1]


def validate_export_payload(
    payload: dict[str, Any],
    export_path: Path,
    *,
    require_field: bool = False,
    require_pdaf: bool = False,
    require_crosstalk: bool = False,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    row_keys = [
        "field_response_rows",
        "pdaf_rows",
        "crosstalk_summary_rows",
        "crosstalk_cell_rows",
        "gate_rows",
    ]
    if payload.get("schema") != "camera_system_suite_export_v1":
        add_issue(issues, "error", "schema_mismatch", "expected camera_system_suite_export_v1", actual=payload.get("schema"))
    if payload.get("product_lut_ready") is not False:
        add_issue(issues, "error", "product_lut_ready_not_false", "suite export must not claim product LUT readiness")
    if payload.get("status") != "RESEARCH_ONLY":
        add_issue(warnings, "warning", "status_not_research_only", "suite export is expected to be RESEARCH_ONLY", actual=payload.get("status"))

    row_counts = payload.get("row_counts") if isinstance(payload.get("row_counts"), dict) else {}
    for key in row_keys:
        rows = payload.get(key)
        if not isinstance(rows, list):
            add_issue(issues, "error", "missing_rows", f"{key} must be a list", row_key=key)
            continue
        expected = row_counts.get(key)
        if expected is not None and int(expected) != len(rows):
            add_issue(issues, "error", "row_count_mismatch", f"{key} length does not match row_counts", row_key=key, expected=expected, actual=len(rows))

    if require_field and not payload.get("field_response_rows"):
        add_issue(issues, "error", "required_field_rows_missing", "field response rows were required but are absent")
    if require_pdaf and not payload.get("pdaf_rows"):
        add_issue(issues, "error", "required_pdaf_rows_missing", "PDAF rows were required but are absent")
    if require_crosstalk and not payload.get("crosstalk_summary_rows") and not payload.get("crosstalk_cell_rows"):
        add_issue(issues, "error", "required_crosstalk_rows_missing", "crosstalk rows were required but are absent")

    for index, row in enumerate(payload.get("field_response_rows") or []):
        if row.get("product_lut_ready") not in {False, "False", "false", 0, "0"}:
            add_issue(issues, "error", "field_row_product_ready", "field rows must not claim product readiness", row_index=index)
        total = finite_float(row.get("total_response"))
        if not math.isfinite(total):
            add_issue(issues, "error", "field_total_nonfinite", "field total_response must be finite", row_index=index)
        elif total < -tolerance:
            add_issue(issues, "error", "field_total_negative", "field total_response must be nonnegative", row_index=index, value=total)
        for column in ("field_x_norm", "field_z_norm", "cra_x_deg", "cra_z_deg"):
            value = finite_float(row.get(column))
            if row.get(column) not in {None, ""} and not math.isfinite(value):
                add_issue(issues, "error", "field_axis_nonfinite", f"{column} must be finite when present", row_index=index, column=column)

    for index, row in enumerate(payload.get("pdaf_rows") or []):
        for column in ("split_phase_x_proxy", "split_phase_z_proxy"):
            value = finite_float(row.get(column))
            if row.get(column) not in {None, ""} and abs(value) > 1.0 + tolerance:
                add_issue(issues, "error", "pdaf_split_out_of_range", f"{column} must stay within [-1, 1]", row_index=index, column=column, value=value)
        amplitude = finite_float(row.get("split_phase_amplitude"))
        if row.get("split_phase_amplitude") not in {None, ""} and amplitude < -tolerance:
            add_issue(issues, "error", "pdaf_amplitude_negative", "split phase amplitude must be nonnegative", row_index=index, value=amplitude)

    for index, row in enumerate(payload.get("crosstalk_summary_rows") or []):
        for column in ("output_crosstalk_fraction", "strongest_neighbor_fraction", "truncation_response_fraction"):
            value = finite_float(row.get(column))
            if not math.isfinite(value):
                add_issue(issues, "error", "crosstalk_fraction_nonfinite", f"{column} must be finite", row_index=index, column=column)
            elif value < -tolerance or value > 1.0 + tolerance:
                add_issue(issues, "error", "crosstalk_fraction_out_of_range", f"{column} must be in [0, 1]", row_index=index, column=column, value=value)

    grouped_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for index, row in enumerate(payload.get("crosstalk_cell_rows") or []):
        value = finite_float(row.get("response_fraction"))
        if not math.isfinite(value):
            add_issue(issues, "error", "crosstalk_cell_nonfinite", "cell response_fraction must be finite", row_index=index)
        elif value < -tolerance or value > 1.0 + tolerance:
            add_issue(issues, "error", "crosstalk_cell_out_of_range", "cell response_fraction must be in [0, 1]", row_index=index, value=value)
        grouped_cells.setdefault(
            (
                row.get("suite_id"),
                row.get("tier"),
                row.get("case_id"),
                row.get("source_case"),
            ),
            [],
        ).append(row)
    for key, rows in grouped_cells.items():
        has_center = any(
            abs(finite_float(row.get("output_dx"), 99.0)) <= tolerance
            and abs(finite_float(row.get("output_dz"), 99.0)) <= tolerance
            for row in rows
        )
        if not has_center:
            add_issue(warnings, "warning", "crosstalk_center_cell_missing", "kernel group has no center output cell", group_key=list(key))

    gate_rows = payload.get("gate_rows") or []
    gate_check_count = 0
    gate_fail_count = 0
    negative_flux_count = 0
    for index, row in enumerate(gate_rows):
        kpi_status = str(row.get("kpi_status") or "").upper()
        grid_gate = row.get("grid_gate_pass")
        if kpi_status == "FAIL":
            gate_fail_count += 1
        elif kpi_status != "PASS" or grid_gate in {False, "False", "false", 0, "0"}:
            gate_check_count += 1
        negative_flux = finite_float(row.get("negative_signed_flux_count"), 0.0)
        if negative_flux > 0:
            negative_flux_count += 1
        if row.get("product_lut_ready") not in {False, "False", "false", 0, "0"}:
            add_issue(issues, "error", "gate_row_product_ready", "gate rows must not claim product readiness", row_index=index)
    if not gate_rows:
        add_issue(warnings, "warning", "gate_rows_missing", "no gate rows are present")

    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    missing_artifacts = []
    for name, value in artifacts.items():
        local_path = local_path_from_artifact(value, export_path.parent)
        if local_path and not local_path.exists():
            missing_artifacts.append({"name": name, "path": str(local_path)})
    if missing_artifacts:
        add_issue(warnings, "warning", "artifact_targets_missing", "some artifact links do not exist locally", artifacts=missing_artifacts)

    error_count = sum(1 for item in issues if item.get("severity") == "error")
    return {
        "schema": "camera_system_suite_export_validation_report_v1",
        "pass": error_count == 0,
        "bad_count": error_count,
        "warning_count": len(warnings),
        "gate_check_count": gate_check_count,
        "gate_fail_count": gate_fail_count,
        "negative_flux_gate_count": negative_flux_count,
        "row_counts": {key: len(payload.get(key) or []) for key in row_keys},
        "coverage": {
            "has_field_response": bool(payload.get("field_response_rows")),
            "has_pdaf": bool(payload.get("pdaf_rows")),
            "has_crosstalk": bool(payload.get("crosstalk_summary_rows") or payload.get("crosstalk_cell_rows")),
        },
        "issues": issues,
        "warnings": warnings,
    }


def query_field_rows(
    field_rows: list[dict[str, Any]],
    field_x_values: list[float],
    field_z_values: list[float],
    wavelength_nm: float | None,
) -> list[dict[str, Any]]:
    by_wavelength: dict[float | None, list[dict[str, Any]]] = {}
    for row in field_rows:
        wavelength = finite_float(row.get("wavelength_nm"))
        key: float | None = wavelength if math.isfinite(wavelength) else None
        if wavelength_nm is not None and (key is None or abs(key - wavelength_nm) > 1e-9):
            continue
        by_wavelength.setdefault(key, []).append(row)
    query_rows: list[dict[str, Any]] = []
    for wavelength, group in sorted(by_wavelength.items(), key=lambda item: float("-inf") if item[0] is None else item[0]):
        z_values = sorted({finite_float(row.get("field_z_norm")) for row in group if math.isfinite(finite_float(row.get("field_z_norm")))})
        for field_z in field_z_values:
            selected_z = min(z_values, key=lambda value: abs(value - field_z)) if z_values else math.nan
            z_group = [
                row
                for row in group
                if not math.isfinite(selected_z) or abs(finite_float(row.get("field_z_norm")) - selected_z) <= 1e-12
            ]
            for field_x in field_x_values:
                def column_value(column: str) -> float:
                    return interpolate(
                        [(finite_float(row.get("field_x_norm")), finite_float(row.get(column))) for row in z_group],
                        field_x,
                    )

                query_rows.append(
                    {
                        "query_field_x_norm": field_x,
                        "query_field_z_norm": field_z,
                        "wavelength_nm": wavelength if wavelength is not None else "",
                        "cra_x_deg": column_value("cra_x_deg"),
                        "cra_z_deg": column_value("cra_z_deg"),
                        "total_response": max(0.0, column_value("total_response")),
                        "edge_to_center": column_value("edge_to_center"),
                        "split_phase_x_proxy": column_value("split_phase_x_proxy"),
                        "split_phase_z_proxy": column_value("split_phase_z_proxy"),
                        "source_row_count": len(z_group),
                        "interpolation_method": "piecewise_linear_field_x_nearest_field_z_suite_export_v1",
                        "product_lut_ready": False,
                    }
                )
    return query_rows


def crosstalk_index_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = payload.get("crosstalk_summary_rows") or []
    summary_lookup: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in summaries:
        summary_lookup[(row.get("case_id"), row.get("source_case"))] = row
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in payload.get("crosstalk_cell_rows") or []:
        groups.setdefault(
            (
                row.get("suite_id"),
                row.get("tier"),
                row.get("case_id"),
                row.get("source_case"),
            ),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (suite_id, tier, case_id, source_case), rows in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        center_values = [
            finite_float(row.get("response_fraction"))
            for row in rows
            if abs(finite_float(row.get("output_dx"), 99.0)) <= 1e-12
            and abs(finite_float(row.get("output_dz"), 99.0)) <= 1e-12
        ]
        fractions = [finite_float(row.get("response_fraction")) for row in rows if math.isfinite(finite_float(row.get("response_fraction")))]
        dx_values = [finite_float(row.get("output_dx")) for row in rows if math.isfinite(finite_float(row.get("output_dx")))]
        dz_values = [finite_float(row.get("output_dz")) for row in rows if math.isfinite(finite_float(row.get("output_dz")))]
        center = sum(center_values) if center_values else math.nan
        total = sum(fractions) if fractions else math.nan
        neighbor_values = [
            finite_float(row.get("response_fraction"))
            for row in rows
            if not (
                abs(finite_float(row.get("output_dx"), 99.0)) <= 1e-12
                and abs(finite_float(row.get("output_dz"), 99.0)) <= 1e-12
            )
        ]
        summary = summary_lookup.get((case_id, source_case), {})
        region_ids = sorted({str(row.get("region_id")) for row in rows if row.get("region_id") not in {None, ""}})
        output.append(
            {
                "suite_id": suite_id,
                "tier": tier,
                "case_id": case_id,
                "source_case": source_case,
                "region_count": len(region_ids),
                "region_ids": ",".join(region_ids),
                "cell_count": len(rows),
                "center_response_fraction": center,
                "off_center_response_fraction": total - center if math.isfinite(total) and math.isfinite(center) else math.nan,
                "total_response_fraction": total,
                "max_neighbor_response_fraction": max(neighbor_values) if neighbor_values else 0.0,
                "output_dx_min": min(dx_values) if dx_values else "",
                "output_dx_max": max(dx_values) if dx_values else "",
                "output_dz_min": min(dz_values) if dz_values else "",
                "output_dz_max": max(dz_values) if dz_values else "",
                "summary_output_crosstalk_fraction": summary.get("output_crosstalk_fraction"),
                "summary_strongest_neighbor_fraction": summary.get("strongest_neighbor_fraction"),
                "summary_truncation_response_fraction": summary.get("truncation_response_fraction"),
                "grid_gate_pass": summary.get("grid_gate_pass"),
                "kpi_status": summary.get("kpi_status"),
                "product_lut_ready": False,
            }
        )
    return output


def gate_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("gate_rows") or []:
        kpi_status = str(row.get("kpi_status") or "").upper()
        grid_gate = row.get("grid_gate_pass")
        if kpi_status == "FAIL":
            consumer_gate = "FAIL"
        elif kpi_status != "PASS" or grid_gate in {False, "False", "false", 0, "0"}:
            consumer_gate = "CHECK"
        else:
            consumer_gate = "PASS"
        rows.append({**row, "consumer_gate": consumer_gate, "product_lut_ready": False})
    return rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    validation = payload["validation"]
    lines = [
        "# Camera-System Suite Export Validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Product LUT ready: `{payload['product_lut_ready']}`",
        f"- Structural pass: `{validation['pass']}`",
        f"- Errors: `{validation['bad_count']}`",
        f"- Warnings: `{validation['warning_count']}`",
        f"- Gate CHECK rows: `{validation['gate_check_count']}`",
        f"- Gate FAIL rows: `{validation['gate_fail_count']}`",
        "",
        "This validation only proves that the package is safe to ingest as research/trend data. It does not certify product accuracy.",
        "",
        "## Coverage",
        "",
        f"- Field response rows: `{validation['row_counts']['field_response_rows']}`",
        f"- PDAF rows: `{validation['row_counts']['pdaf_rows']}`",
        f"- Crosstalk summary rows: `{validation['row_counts']['crosstalk_summary_rows']}`",
        f"- Crosstalk cell rows: `{validation['row_counts']['crosstalk_cell_rows']}`",
        f"- Gate rows: `{validation['row_counts']['gate_rows']}`",
    ]
    if validation["issues"]:
        lines += ["", "## Errors", ""]
        lines += [f"- `{item['code']}`: {item['message']}" for item in validation["issues"]]
    if validation["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- `{item['code']}`: {item['message']}" for item in validation["warnings"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_export_package(
    export_json: Path,
    output_dir: Path,
    *,
    field_x_values: list[float] | None = None,
    field_z_values: list[float] | None = None,
    wavelength_nm: float | None = None,
    require_field: bool = False,
    require_pdaf: bool = False,
    require_crosstalk: bool = False,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    payload = read_json(export_json)
    validation = validate_export_payload(
        payload,
        export_json,
        require_field=require_field,
        require_pdaf=require_pdaf,
        require_crosstalk=require_crosstalk,
        tolerance=tolerance,
    )
    field_x_values = field_x_values if field_x_values is not None else [0.0, 0.5, 1.0]
    field_z_values = field_z_values if field_z_values is not None else [0.0]
    output_dir.mkdir(parents=True, exist_ok=True)
    field_query = query_field_rows(payload.get("field_response_rows") or [], field_x_values, field_z_values, wavelength_nm)
    crosstalk_index = crosstalk_index_rows(payload)
    gates = gate_summary_rows(payload)

    validation_json = output_dir / "camera_system_suite_export_validation.json"
    validation_md = output_dir / "camera_system_suite_export_validation.md"
    field_query_csv = output_dir / "camera_system_suite_export_field_query.csv"
    crosstalk_index_csv = output_dir / "camera_system_suite_export_crosstalk_index.csv"
    gate_summary_csv = output_dir / "camera_system_suite_export_gate_summary.csv"
    result = {
        "schema": "camera_system_suite_export_validation_v1",
        "status": "PASS" if validation["pass"] else "FAIL",
        "artifact_role": "camera_system_suite_export_consumer_validation",
        "product_lut_ready": False,
        "source_export_json": str(export_json),
        "source_suite_id": payload.get("suite_id"),
        "source_tier": payload.get("tier"),
        "usage_scope": "camera_system_research_trend_not_product_accuracy",
        "validation": validation,
        "query": {
            "field_x_norm": field_x_values,
            "field_z_norm": field_z_values,
            "wavelength_nm": wavelength_nm if wavelength_nm is not None else "all",
            "field_query_row_count": len(field_query),
            "crosstalk_index_row_count": len(crosstalk_index),
            "gate_summary_row_count": len(gates),
        },
        "outputs": {
            "validation_json": str(validation_json),
            "validation_md": str(validation_md),
            "field_query_csv": str(field_query_csv),
            "crosstalk_index_csv": str(crosstalk_index_csv),
            "gate_summary_csv": str(gate_summary_csv),
        },
        "notes": [
            "This package can be consumed as research/trend evidence when validation passes.",
            "Product LUT use remains blocked until measured stack/material/device calibration and quantitative convergence pass.",
        ],
    }
    validation_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(field_query_csv, field_query, FIELD_QUERY_COLUMNS)
    write_csv(crosstalk_index_csv, crosstalk_index, CROSSTALK_INDEX_COLUMNS)
    write_csv(gate_summary_csv, gates, GATE_SUMMARY_COLUMNS)
    write_markdown(validation_md, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/camera_system_suite_export_validation_reference"))
    parser.add_argument("--field-x", default="0,0.5,1")
    parser.add_argument("--field-z", default="0")
    parser.add_argument("--wavelength-nm", default="all")
    parser.add_argument("--require-field", action="store_true")
    parser.add_argument("--require-pdaf", action="store_true")
    parser.add_argument("--require-crosstalk", action="store_true")
    parser.add_argument("--allow-validation-errors", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    wavelength = None if str(args.wavelength_nm).lower() == "all" else finite_float(args.wavelength_nm)
    if wavelength is not None and not math.isfinite(wavelength):
        raise ValueError("--wavelength-nm must be 'all' or a finite value")
    result = validate_export_package(
        args.export_json,
        args.output_dir,
        field_x_values=parse_float_list(args.field_x),
        field_z_values=parse_float_list(args.field_z),
        wavelength_nm=wavelength,
        require_field=args.require_field,
        require_pdaf=args.require_pdaf,
        require_crosstalk=args.require_crosstalk,
        tolerance=args.tolerance,
    )
    summary = {
        "schema": result["schema"],
        "status": result["status"],
        "product_lut_ready": result["product_lut_ready"],
        "validation": {
            "pass": result["validation"]["pass"],
            "bad_count": result["validation"]["bad_count"],
            "warning_count": result["validation"]["warning_count"],
            "row_counts": result["validation"]["row_counts"],
        },
        "query": result["query"],
        "outputs": result["outputs"],
    }
    print(json.dumps(summary, indent=2))
    if result["status"] != "PASS" and not args.allow_validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
