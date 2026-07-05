#!/usr/bin/env python3
"""Validate camera-module CRA and microlens-shift field-map input.

This is the fast preflight for replacing generated CameraE2E field priors with
module raytrace, measured, or calibrated CRA/ML-shift rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_camera_e2e_sensor_luts import (
    DEFAULT_FIELD_MAP_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SENSOR_CATALOG,
    DEFAULT_STACK_DIR,
    FIELD_MAP_OUTPUT_COLUMNS,
    FIELD_MAP_VALIDATION_COLUMNS,
    field_design_cases,
    field_map_validation_report,
    load_catalog,
    load_field_map_overrides,
    ocl_mode_guess,
    read_json,
    selected_slugs,
    write_csv,
    write_field_map_validation_html,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-catalog", type=Path, default=DEFAULT_SENSOR_CATALOG)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--field-map-csv", type=Path, default=DEFAULT_FIELD_MAP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="")
    parser.add_argument("--major-only", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog(args.sensor_catalog)
    selected = selected_slugs(catalog, args)
    field_map_overrides = load_field_map_overrides(args.field_map_csv)
    field_design_rows = []

    field_map_source = str(args.field_map_csv) if args.field_map_csv.exists() else ""
    for slug in selected:
        stack_path = args.stack_dir / f"{slug}.json"
        if not stack_path.exists():
            continue
        stack = read_json(stack_path)
        catalog_row = catalog.get(slug, {})
        mode = ocl_mode_guess(catalog_row, stack)
        field_design_rows.extend(
            field_design_cases(
                slug,
                catalog_row,
                stack,
                mode,
                field_overrides=field_map_overrides.get(slug, []),
                field_override_source=field_map_source,
            )
        )

    report, rows = field_map_validation_report(
        field_map_csv=args.field_map_csv,
        selected=selected,
        catalog=catalog,
        field_map_overrides=field_map_overrides,
        field_design_rows=field_design_rows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "camera_module_field_map_validation.json", report)
    write_csv(args.output_dir / "camera_module_field_map_validation.csv", rows, FIELD_MAP_VALIDATION_COLUMNS)
    write_field_map_validation_html(args.output_dir / "camera_module_field_map_validation.html", report, rows)

    seed_rows = [
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
        for row in field_design_rows
    ]
    write_csv(args.output_dir / "camera_module_field_map_prior_seed.csv", seed_rows, FIELD_MAP_OUTPUT_COLUMNS)
    print(f"{report.get('gate')} {report.get('path')} rows={report.get('input_row_count', 0)}")


if __name__ == "__main__":
    main()
