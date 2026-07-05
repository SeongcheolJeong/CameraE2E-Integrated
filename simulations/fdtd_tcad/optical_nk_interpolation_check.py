#!/usr/bin/env python3
"""Smoke-check optical n,k table parsing, interpolation, and evidence gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optical_stack_evidence import build_evidence
from sensor_stack_config import load_nk_table


DEFAULT_OUTPUT_DIR = Path("runs/optical_nk_interpolation_check")


def write_fixture(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nk_path = output_dir / "unsorted_measured_nk.csv"
    nk_path.write_text(
        "\n".join(
            [
                "# wavelength_um,n,k",
                "0.70,1.70,0.030",
                "0.40,1.40,0.000",
                "0.55,1.55,0.015",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    duplicate_path = output_dir / "duplicate_nk.csv"
    duplicate_path.write_text(
        "\n".join(
            [
                "0.40,1.40,0.000",
                "0.55,1.55,0.015",
                "0.55,1.56,0.016",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stack_path = output_dir / "measured_stack_fixture.json"
    stack = {
        "schema": "sensor_stack_config_v1",
        "name": "measured_nk_interpolation_fixture",
        "calibration_status": {
            "is_measured": True,
            "geometry_measured": True,
            "mode": "measured_fixture",
        },
        "geometry_um": {
            "pitch": 1.4,
            "lens_height": 0.657,
            "cfa_thickness": 0.8,
            "passivation_thickness": 0.08,
            "si_thickness": 2.8,
        },
        "materials": {
            role: {
                "nk_table": nk_path.name,
                "measured": True,
                "source": "fixture measured-style n,k table",
                "usage": role,
            }
            for role in ("silicon", "cfa_red", "cfa_green", "cfa_blue", "passivation", "lens")
        },
    }
    stack_path.write_text(json.dumps(stack, indent=2), encoding="utf-8")
    return nk_path, duplicate_path, stack_path


def run(args: argparse.Namespace) -> dict:
    nk_path, duplicate_path, stack_path = write_fixture(args.output_dir)
    n_mid, k_mid = load_nk_table(nk_path, 0.475)
    expected_n = 1.475
    expected_k = 0.0075
    duplicate_failed = False
    duplicate_error = ""
    try:
        load_nk_table(duplicate_path, 0.50)
    except ValueError as exc:
        duplicate_failed = True
        duplicate_error = str(exc)
    evidence = build_evidence(
        stack_path,
        required_wavelengths_um=[0.45, 0.55, 0.65],
    )
    summary = evidence["summary"]
    pass_status = (
        abs(n_mid - expected_n) <= args.absolute_tolerance
        and abs(k_mid - expected_k) <= args.absolute_tolerance
        and duplicate_failed
        and summary["all_nk_tables_exist"]
        and summary["all_nk_tables_valid"]
        and summary["all_required_wavelengths_covered"]
        and summary["accuracy_ready"]
    )
    report = {
        "schema": "optical_nk_interpolation_check_v1",
        "pass": pass_status,
        "nk_table": str(nk_path),
        "interpolated": {"wavelength_um": 0.475, "n": n_mid, "k": k_mid},
        "expected": {"n": expected_n, "k": expected_k},
        "duplicate_table_failed": duplicate_failed,
        "duplicate_error": duplicate_error,
        "evidence_summary": summary,
        "sample_material": evidence["materials"][0],
    }
    report_path = args.output_dir / "optical_nk_interpolation_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "pass": report["pass"],
                "interpolated": report["interpolated"],
                "duplicate_table_failed": report["duplicate_table_failed"],
                "all_nk_tables_valid": report["evidence_summary"]["all_nk_tables_valid"],
                "all_required_wavelengths_covered": report["evidence_summary"][
                    "all_required_wavelengths_covered"
                ],
                "output": str(args.output_dir / "optical_nk_interpolation_check.json"),
            },
            indent=2,
        )
    )
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
