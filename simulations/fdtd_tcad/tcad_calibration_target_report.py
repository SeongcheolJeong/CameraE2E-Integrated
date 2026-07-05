#!/usr/bin/env python3
"""Validate calibration targets and best-fit residuals.

The calibration loop proves only that an optimizer ran. This report checks the
consumer-facing contract: every target row has a matching simulation row, target
sources are explicit, and best-fit current/split residuals pass configured
tolerances.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION_RESULT = Path("runs/tcad_calibration_reference_profile/calibration_result.json")
DEFAULT_TARGETS_CSV = Path("measured_profiles/reference_cmos_ppd_1p4um/calibration_targets_synthetic.csv")
DEFAULT_OUTPUT_DIR = Path("runs/tcad_calibration_target_report_reference")

CSV_COLUMNS = [
    "case",
    "wavelength_nm",
    "target_source",
    "target_measured",
    "target_total_photo_current_a_per_cm",
    "sim_total_photo_current_a_per_cm",
    "current_relative_error",
    "current_relative_tolerance",
    "current_pass",
    "target_split_phase_x_proxy",
    "sim_split_phase_x_proxy",
    "split_phase_abs_error",
    "split_phase_abs_tolerance",
    "split_phase_pass",
    "row_pass",
    "summary_json",
]

METRIC_CSV_COLUMNS = [
    "case",
    "wavelength_nm",
    "metric",
    "target",
    "simulated",
    "scale",
    "weight",
    "residual",
    "abs_residual",
    "max_abs_normalized_residual",
    "metric_pass",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def first_float(row: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        if row.get(key, "") not in {"", None}:
            value = finite_float(row.get(key))
            if math.isfinite(value):
                return value
    return default


def load_targets(
    path: Path,
    default_current_relative_tolerance: float,
    default_split_phase_abs_tolerance: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    required = {
        "case",
        "wavelength_nm",
        "target_source",
        "target_total_photo_current_a_per_cm",
        "target_split_phase_x_proxy",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        rows = []
        for index, row in enumerate(reader):
            target_source = str(row.get("target_source", "")).strip()
            rows.append(
                {
                    "row_index": index,
                    "case": str(row.get("case", "")).strip(),
                    "wavelength_nm": finite_float(row.get("wavelength_nm")),
                    "target_total_photo_current_a_per_cm": finite_float(
                        row.get("target_total_photo_current_a_per_cm")
                    ),
                    "target_split_phase_x_proxy": finite_float(
                        row.get("target_split_phase_x_proxy")
                    ),
                    "target_source": target_source,
                    "target_measured": target_source.lower() == "measured",
                    "current_relative_tolerance": first_float(
                        row,
                        [
                            "current_relative_tolerance",
                            "total_current_relative_tolerance",
                            "target_total_photo_current_relative_tolerance",
                        ],
                        default_current_relative_tolerance,
                    ),
                    "split_phase_abs_tolerance": first_float(
                        row,
                        [
                            "split_phase_abs_tolerance",
                            "phase_absolute_tolerance",
                            "split_phase_x_abs_tolerance",
                        ],
                        default_split_phase_abs_tolerance,
                    ),
                    "weight_current": finite_float(row.get("weight_current"), 1.0),
                    "weight_phase": finite_float(row.get("weight_phase"), 1.0),
                    "notes": row.get("notes", ""),
                }
            )
    if not rows:
        missing.append("nonempty_target_rows")
    return rows, missing


def target_key(row: dict[str, Any]) -> tuple[str, float]:
    return (str(row.get("case", "")), round(finite_float(row.get("wavelength_nm")), 9))


def best_history_entry(result: dict[str, Any]) -> dict[str, Any] | None:
    history = [entry for entry in result.get("history", []) if isinstance(entry, dict)]
    if not history:
        return None
    return min(history, key=lambda entry: finite_float(entry.get("residual_norm"), math.inf))


def build_rows(
    targets: list[dict[str, Any]],
    best_entry: dict[str, Any] | None,
    current_floor_a_per_cm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    sim_rows = {}
    if best_entry is None:
        issues.append(
            {
                "severity": "error",
                "code": "missing_calibration_history",
                "message": "calibration_result.json has no history rows",
            }
        )
    else:
        for row in best_entry.get("rows", []):
            sim_rows[target_key(row)] = row

    output_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for target in targets:
        key = target_key(target)
        if key in seen:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_target",
                    "message": "duplicate case/wavelength target row",
                    "case": target["case"],
                    "wavelength_nm": target["wavelength_nm"],
                }
            )
        seen.add(key)
        sim = sim_rows.get(key)
        if sim is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_simulation_row",
                    "message": "best calibration entry has no matching simulation row",
                    "case": target["case"],
                    "wavelength_nm": target["wavelength_nm"],
                }
            )
            sim_total = math.nan
            sim_phase = math.nan
            summary_json = ""
        else:
            sim_total = finite_float(sim.get("sim_total_photo_current_a_per_cm"))
            sim_phase = finite_float(sim.get("sim_split_phase_x_proxy"))
            summary_json = str(sim.get("summary_json", ""))
        target_total = finite_float(target.get("target_total_photo_current_a_per_cm"))
        target_phase = finite_float(target.get("target_split_phase_x_proxy"))
        denom = max(abs(target_total), current_floor_a_per_cm)
        current_relative_error = (
            (sim_total - target_total) / denom
            if math.isfinite(sim_total) and math.isfinite(target_total)
            else math.nan
        )
        split_phase_abs_error = (
            abs(sim_phase - target_phase)
            if math.isfinite(sim_phase) and math.isfinite(target_phase)
            else math.nan
        )
        current_pass = (
            math.isfinite(current_relative_error)
            and abs(current_relative_error) <= target["current_relative_tolerance"]
        )
        split_pass = (
            math.isfinite(split_phase_abs_error)
            and split_phase_abs_error <= target["split_phase_abs_tolerance"]
        )
        output_rows.append(
            {
                "case": target["case"],
                "wavelength_nm": target["wavelength_nm"],
                "target_source": target["target_source"],
                "target_measured": bool(target["target_measured"]),
                "target_total_photo_current_a_per_cm": target_total,
                "sim_total_photo_current_a_per_cm": sim_total,
                "current_relative_error": current_relative_error,
                "current_relative_tolerance": target["current_relative_tolerance"],
                "current_pass": current_pass,
                "target_split_phase_x_proxy": target_phase,
                "sim_split_phase_x_proxy": sim_phase,
                "split_phase_abs_error": split_phase_abs_error,
                "split_phase_abs_tolerance": target["split_phase_abs_tolerance"],
                "split_phase_pass": split_pass,
                "row_pass": bool(current_pass and split_pass),
                "summary_json": summary_json,
            }
        )
    return output_rows, issues


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def metric_residual_rows(
    best_entry: dict[str, Any] | None,
    max_abs_normalized_residual: float,
) -> tuple[list[dict[str, Any]], bool | None]:
    if best_entry is None:
        return [], None
    terms = best_entry.get("residual_terms", [])
    if not terms:
        return [], None
    rows = []
    for term in terms:
        residual = finite_float(term.get("residual"))
        abs_residual = abs(residual) if math.isfinite(residual) else math.nan
        metric_pass = math.isfinite(abs_residual) and abs_residual <= max_abs_normalized_residual
        rows.append(
            {
                "case": term.get("case", ""),
                "wavelength_nm": term.get("wavelength_nm", ""),
                "metric": term.get("metric", ""),
                "target": term.get("target", ""),
                "simulated": term.get("simulated", ""),
                "scale": term.get("scale", ""),
                "weight": term.get("weight", ""),
                "residual": residual,
                "abs_residual": abs_residual,
                "max_abs_normalized_residual": max_abs_normalized_residual,
                "metric_pass": metric_pass,
            }
        )
    return rows, bool(rows) and all(bool(row["metric_pass"]) for row in rows)


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in METRIC_CSV_COLUMNS})


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.calibration_result.read_text(encoding="utf-8"))
    targets, missing_columns = load_targets(
        args.targets_csv,
        args.default_current_relative_tolerance,
        args.default_split_phase_abs_tolerance,
    )
    best = best_history_entry(result)
    rows, issues = build_rows(targets, best, args.current_floor_a_per_cm)
    metric_rows, metric_residual_pass = metric_residual_rows(
        best,
        args.max_abs_normalized_residual,
    )
    for column in missing_columns:
        issues.append(
            {
                "severity": "error",
                "code": "missing_target_column",
                "message": f"targets CSV is missing {column}",
            }
        )
    all_targets_measured = bool(targets) and all(bool(row["target_measured"]) for row in targets)
    legacy_residual_pass = bool(rows) and all(bool(row["row_pass"]) for row in rows)
    residual_pass = (
        legacy_residual_pass
        and (metric_residual_pass is not False)
        and not issues
    )
    product_accuracy_ready = all_targets_measured and residual_pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "calibration_target_report.json"
    csv_path = args.output_dir / "calibration_target_report.csv"
    metric_csv_path = args.output_dir / "calibration_target_metric_residuals.csv"
    payload = {
        "schema": "tcad_calibration_target_report_v1",
        "artifact_role": "calibration_target_validation",
        "calibration_result": str(args.calibration_result),
        "targets_csv": str(args.targets_csv),
        "best_eval_index": None if best is None else best.get("eval_index"),
        "best_residual_norm": None if best is None else best.get("residual_norm"),
        "row_count": len(rows),
        "metric_residual_row_count": len(metric_rows),
        "all_targets_measured": all_targets_measured,
        "residual_pass": residual_pass,
        "legacy_current_split_residual_pass": legacy_residual_pass,
        "metric_residual_pass": metric_residual_pass,
        "max_abs_normalized_residual": args.max_abs_normalized_residual,
        "product_accuracy_ready": product_accuracy_ready,
        "default_current_relative_tolerance": args.default_current_relative_tolerance,
        "default_split_phase_abs_tolerance": args.default_split_phase_abs_tolerance,
        "current_floor_a_per_cm": args.current_floor_a_per_cm,
        "issues": issues,
        "non_measured_targets": [
            f"{row['case']}:{row['wavelength_nm']}" for row in targets if not bool(row["target_measured"])
        ],
        "rows": rows,
        "metric_residual_rows": metric_rows,
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "metric_csv": str(metric_csv_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_metric_csv(metric_csv_path, metric_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-result", type=Path, default=DEFAULT_CALIBRATION_RESULT)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--default-current-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--default-split-phase-abs-tolerance", type=float, default=0.005)
    parser.add_argument("--current-floor-a-per-cm", type=float, default=1.0e-12)
    parser.add_argument("--max-abs-normalized-residual", type=float, default=0.05)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "all_targets_measured": payload["all_targets_measured"],
                "residual_pass": payload["residual_pass"],
                "product_accuracy_ready": payload["product_accuracy_ready"],
                "row_count": payload["row_count"],
                "metric_residual_row_count": payload["metric_residual_row_count"],
                "metric_residual_pass": payload["metric_residual_pass"],
                "non_measured_targets": payload["non_measured_targets"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
