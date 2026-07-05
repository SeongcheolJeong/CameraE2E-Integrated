#!/usr/bin/env python3
"""Merge high-resolution 2D crosstalk x-section runs into one reference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meep_crosstalk_kernel import cases_from_summaries, convergence_status, read_csv_rows, unique_values
from meep_crosstalk_xsection_2d import save_line_plot, write_csv


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    ROOT / "runs/crosstalk_xsection_2d_split_n9_r72_84",
    ROOT / "runs/crosstalk_xsection_2d_ocl2_n5_r72_84",
    ROOT / "runs/crosstalk_xsection_2d_ocl3_n5_r72_84",
]


def build_kernel(summary: dict, output_rows: list[dict]) -> np.ndarray:
    neighborhood = int(summary["neighborhood"])
    half = neighborhood // 2
    resolution = int(summary["resolution_px_per_um"])
    kernel = np.zeros(neighborhood, dtype=float)
    for row in output_rows:
        if (
            row.get("mode") == summary.get("mode")
            and row.get("case") == summary.get("case")
            and int(row.get("neighborhood", 0)) == neighborhood
            and int(row.get("resolution_px_per_um", 0)) == resolution
        ):
            kernel[int(row["output_dx"]) + half] = float(row["response_fraction"])
    return kernel


def run(args: argparse.Namespace) -> dict:
    inputs = [path.resolve() for path in args.inputs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    output_rows: list[dict] = []
    raw_rows: list[dict] = []
    for input_dir in inputs:
        summary_rows += read_csv_rows(input_dir / "crosstalk_xsection_summary.csv")
        output_rows += read_csv_rows(input_dir / "crosstalk_xsection_output_kernel.csv")
        raw_rows += read_csv_rows(input_dir / "crosstalk_xsection_raw_pd_kernel.csv")

    output_csv = output_dir / "crosstalk_xsection_output_kernel.csv"
    raw_csv = output_dir / "crosstalk_xsection_raw_pd_kernel.csv"
    summary_csv = output_dir / "crosstalk_xsection_summary.csv"
    plot_png = output_dir / "crosstalk_xsection_kernel_lines.png"
    convergence_json = output_dir / "crosstalk_xsection_convergence.json"
    manifest_json = output_dir / "crosstalk_xsection_kernel.json"

    write_csv(output_csv, output_rows)
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    convergence = convergence_status(
        summary_rows,
        args.truncation_threshold,
        args.delta_threshold,
        args.min_feature_pixels,
        args.min_si_wavelength_pixels,
    )
    convergence_json.write_text(json.dumps(convergence, indent=2), encoding="utf-8")

    max_resolution = max(int(row["resolution_px_per_um"]) for row in summary_rows) if summary_rows else 0
    plot_kernels = [
        (summary, build_kernel(summary, output_rows))
        for summary in summary_rows
        if int(summary["resolution_px_per_um"]) == max_resolution
    ]
    save_line_plot(plot_png, plot_kernels)

    manifest = {
        "schema": "camera_crosstalk_xsection_fdtd_v1",
        "solver": "Meep 2D high-resolution FDTD",
        "source_model": "center OCL finite-aperture impulse x-section",
        "accuracy_status": "xsection_fdtd_numerical_gate_pass_not_3d_product_lut",
        "measured_accuracy_blocked": True,
        "convergence_status": convergence["status"],
        "configuration": {
            "modes": unique_values(summary_rows, "mode", str),
            "neighborhoods": unique_values(summary_rows, "neighborhood", int),
            "simulation_neighborhoods": unique_values(summary_rows, "simulation_neighborhood", int),
            "resolutions_px_per_um": unique_values(summary_rows, "resolution_px_per_um", int),
            "wavelengths_nm": unique_values(summary_rows, "wavelength_nm", float),
            "cases": cases_from_summaries(summary_rows),
            "source_dirs": [str(path) for path in inputs],
            "min_feature_pixels": args.min_feature_pixels,
            "min_si_wavelength_pixels": args.min_si_wavelength_pixels,
        },
        "scope": {
            "primary_kernel": "binned output-cell x-line crosstalk",
            "diagnostic_kernel": "raw physical-PD x-line crosstalk",
            "resolves": [
                "Si internal wavelength grid gate",
                "DTI/passivation/lens-edge critical feature grid gate",
                "CRA-x lateral crosstalk convergence for split-PD, 2x2 OCL, and 3x3 OCL",
            ],
            "does_not_resolve": [
                "full 3D OCL footprint coupling",
                "measured target-product optical n,k",
                "calibrated carrier collection after absorption",
            ],
        },
        "summaries": summary_rows,
        "convergence": convergence,
        "outputs": {
            "output_kernel_csv": str(output_csv),
            "raw_pd_kernel_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
            "plot_png": str(plot_png),
            "convergence_json": str(convergence_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "schema": manifest["schema"],
        "output_dir": str(output_dir),
        "summary_count": len(summary_rows),
        "convergence_status": convergence["status"],
        "outputs": manifest["outputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/crosstalk_xsection_2d_reference")
    parser.add_argument("--truncation-threshold", type=float, default=0.015)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--min-feature-pixels", type=float, default=2.0)
    parser.add_argument("--min-si-wavelength-pixels", type=float, default=8.0)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
