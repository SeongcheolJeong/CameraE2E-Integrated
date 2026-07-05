#!/usr/bin/env python3
"""Smoke-check measured TCAD scattered implant interpolation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from measured_tcad_profile import doping_from_profile_with_summary, load_measured_profile


DEFAULT_OUTPUT_DIR = Path("runs/measured_profile_interpolation_check")


def donor_fn(x_um: float, depth_um: float) -> float:
    return 1.0e15 + 2.0e14 * x_um + 3.0e14 * depth_um


def acceptor_fn(x_um: float, depth_um: float) -> float:
    return 5.0e14 + 1.0e14 * x_um + 4.0e14 * depth_um


def write_fixture(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "linear_implant_grid.csv"
    rows = []
    for x_um in (0.0, 1.0):
        for depth_um in (0.0, 1.0):
            rows.append(
                {
                    "x_um": x_um,
                    "depth_um": depth_um,
                    "donor_cm3": donor_fn(x_um, depth_um),
                    "acceptor_cm3": acceptor_fn(x_um, depth_um),
                }
            )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    profile = {
        "schema": "measured_tcad_profile_v1",
        "profile_name": "interpolation_smoke",
        "units": {"length": "um", "doping": "cm^-3"},
        "geometry": {"width_um": 1.0, "depth_um": 1.0, "split_gap_um": 0.1},
        "background": {"donor_cm3": 0.0, "acceptor_cm3": 0.0},
        "implants": [
            {
                "name": "linear_scattered_reference",
                "type": "csv_scattered",
                "file": csv_path.name,
                "interpolation": "linear_nearest",
                "measured": True,
            }
        ],
        "calibration_status": {
            "is_measured": True,
            "mode": "interpolation_smoke",
        },
    }
    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile_path


def run(args: argparse.Namespace) -> dict:
    profile_path = write_fixture(args.output_dir)
    profile = load_measured_profile(profile_path)
    x_um = np.asarray([0.25, 0.75, 1.25], dtype=float)
    depth_um = np.asarray([0.25, 0.50, 0.50], dtype=float)
    donors, acceptors, summary = doping_from_profile_with_summary(
        profile,
        x_um * 1.0e-4,
        depth_um * 1.0e-4,
    )
    expected_donor_inside = np.asarray(
        [donor_fn(float(x_um[0]), float(depth_um[0])), donor_fn(float(x_um[1]), float(depth_um[1]))]
    )
    expected_acceptor_inside = np.asarray(
        [
            acceptor_fn(float(x_um[0]), float(depth_um[0])),
            acceptor_fn(float(x_um[1]), float(depth_um[1])),
        ]
    )
    donor_error = np.max(np.abs(donors[:2] - expected_donor_inside) / expected_donor_inside)
    acceptor_error = np.max(
        np.abs(acceptors[:2] - expected_acceptor_inside) / expected_acceptor_inside
    )
    fallback_finite = bool(np.isfinite(donors[2]) and np.isfinite(acceptors[2]))
    fallback_count = int(summary[0]["donor"]["outside_hull_nearest_fallback_count"])
    pass_status = (
        donor_error <= args.relative_tolerance
        and acceptor_error <= args.relative_tolerance
        and fallback_finite
        and fallback_count >= 1
    )
    report = {
        "schema": "measured_profile_interpolation_check_v1",
        "pass": pass_status,
        "profile": str(profile_path),
        "query_x_um": x_um.tolist(),
        "query_depth_um": depth_um.tolist(),
        "donor_cm3": donors.tolist(),
        "acceptor_cm3": acceptors.tolist(),
        "donor_inside_relative_error": float(donor_error),
        "acceptor_inside_relative_error": float(acceptor_error),
        "outside_hull_fallback_finite": fallback_finite,
        "outside_hull_nearest_fallback_count": fallback_count,
        "implant_summary": summary,
    }
    report_path = args.output_dir / "interpolation_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "pass": report["pass"],
                "donor_inside_relative_error": report["donor_inside_relative_error"],
                "acceptor_inside_relative_error": report["acceptor_inside_relative_error"],
                "outside_hull_nearest_fallback_count": report[
                    "outside_hull_nearest_fallback_count"
                ],
                "output": str(args.output_dir / "interpolation_check.json"),
            },
            indent=2,
        )
    )
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
