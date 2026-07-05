#!/usr/bin/env python3
"""High-resolution 2D FDTD x-section crosstalk kernel.

The existing 3D full-array runner is useful for OCL footprint geometry, but the
smoke resolutions are far below what 550 nm light inside silicon requires. This
runner keeps the same optical stack and finite-neighborhood source model, then
solves an x-y cross-section at high resolution so the grid and convergence gates
can be exercised on a practical local machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from meep_crosstalk_kernel import (
    DEFAULT_TCAD_PROFILE,
    CrosstalkCase,
    OpticalDtiConfig,
    Region,
    axis_centers,
    bayer_color_for_output,
    cases_from_summaries,
    centered_positions,
    convergence_status,
    grid_resolution_metadata,
    layout_label,
    lens_radius_for_layout,
    lens_sphere_radius,
    mode_layout_size,
    optical_dti_from_profile,
    parse_cases,
    parse_csv_floats,
    parse_csv_ints,
    source_amplitude,
    transverse_k,
    unique_values,
)
from meep_microlens_array_3d import MicrolensArrayGeometry
from sensor_stack_config import (
    DEFAULT_STACK_CONFIG,
    geometry_from_config,
    load_stack_config,
    material_role_for_color,
    medium_for_role,
    nk_from_material,
    shield_config_for_stack,
)


ROOT = Path(__file__).resolve().parent


def distance_to_pitch_boundary_x(value_um: float, pitch_um: float) -> float:
    local = (((value_um / pitch_um) + 0.5) % 1.0 - 0.5) * pitch_um
    return 0.5 * pitch_um - abs(local)


def in_dti_trench_x(point: mp.Vector3, geom: MicrolensArrayGeometry, dti: OpticalDtiConfig, half_array: float) -> bool:
    if not dti.enabled:
        return False
    if abs(point.x) > half_array:
        return False
    depth_from_si_top = geom.si_top - point.y
    if depth_from_si_top < 0.0 or depth_from_si_top > dti.depth_um:
        return False
    return distance_to_pitch_boundary_x(point.x, geom.pitch) <= 0.5 * dti.width_um


def nearest_index(value: float, centers: list[float], pitch: float) -> int | None:
    if not centers:
        return None
    distances = [abs(value - center) for center in centers]
    index = int(np.argmin(distances))
    return index if distances[index] <= 0.5 * pitch + 1.0e-9 else None


def build_regions_1d(
    geom: MicrolensArrayGeometry,
    layout_size: int,
    neighborhood: int,
) -> tuple[list[Region], list[Region]]:
    supercell_pitch = layout_size * geom.pitch
    output_centers = centered_positions(neighborhood, supercell_pitch)
    pd_count = neighborhood * layout_size
    pd_centers = centered_positions(pd_count, geom.pitch)
    half = neighborhood // 2
    pd_half = pd_count // 2
    output_regions = [
        Region(
            region_id=f"out_dx{ix - half:+d}",
            kind="output_cell_xline",
            ix=ix - half,
            iz=0,
            x_um=xc,
            z_um=0.0,
            sx_um=supercell_pitch,
            sz_um=0.0,
        )
        for ix, xc in enumerate(output_centers)
    ]
    raw_pd_regions = [
        Region(
            region_id=f"pd_ix{ix - pd_half:+d}",
            kind="raw_pd_xline",
            ix=ix - pd_half,
            iz=0,
            x_um=xc,
            z_um=0.0,
            sx_um=geom.pitch,
            sz_um=0.0,
        )
        for ix, xc in enumerate(pd_centers)
    ]
    return output_regions, raw_pd_regions


def make_material_function_2d(
    geom: MicrolensArrayGeometry,
    layout_size: int,
    simulation_neighborhood: int,
    case: CrosstalkCase,
    color_channel: str,
    silicon: mp.Medium,
    cfa_media: dict[str, mp.Medium],
    passivation: mp.Medium,
    lens: mp.Medium,
    dti_medium: mp.Medium,
    dti: OpticalDtiConfig,
):
    supercell_pitch = layout_size * geom.pitch
    output_centers = centered_positions(simulation_neighborhood, supercell_pitch)
    lens_radius = lens_radius_for_layout(geom, layout_size)
    sphere_radius = lens_sphere_radius(lens_radius, geom.lens_height)
    lens_center_y = geom.lens_top - sphere_radius
    half_array = 0.5 * simulation_neighborhood * supercell_pitch

    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        if abs(x) > half_array:
            return mp.air
        if geom.si_bottom <= y < geom.si_top:
            if in_dti_trench_x(point, geom, dti, half_array):
                return dti_medium
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            ix = nearest_index(x, output_centers, supercell_pitch)
            if ix is None:
                return mp.air
            offset_x = ix - simulation_neighborhood // 2
            color = bayer_color_for_output(offset_x, 0, color_channel)
            return cfa_media[color]
        if geom.lens_bottom <= y <= geom.lens_top:
            for xc in output_centers:
                dx = x - (xc + case.lens_shift_x_um)
                if abs(dx) > lens_radius:
                    continue
                if dx * dx + (y - lens_center_y) ** 2 <= sphere_radius * sphere_radius:
                    return lens
        return mp.air

    return material


def integrate_regions_1d(
    density: np.ndarray,
    geom: MicrolensArrayGeometry,
    span_x: float,
    regions: list[Region],
) -> dict[str, float]:
    if density.ndim != 2:
        raise RuntimeError(f"expected 2D density array, got shape={density.shape}")
    dx = span_x / density.shape[0]
    dy = geom.si_thickness / density.shape[1]
    x_values = axis_centers(span_x, density.shape[0])
    values = {}
    for region in regions:
        x_mask = np.abs(x_values - region.x_um) <= 0.5 * region.sx_um
        values[region.region_id] = float(np.sum(density[x_mask, :]) * dx * dy)
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_line_plot(path: Path, kernels: list[tuple[dict[str, Any], np.ndarray]]) -> None:
    if not kernels:
        return
    fig, axes = plt.subplots(len(kernels), 1, figsize=(7.2, 2.2 * len(kernels)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (summary, kernel) in zip(axes, kernels):
        xs = np.arange(kernel.size) - kernel.size // 2
        axis.bar(xs, kernel, width=0.84, color="#38bdf8")
        axis.set_yscale("log")
        axis.set_ylim(max(np.min(kernel[kernel > 0]) * 0.5, 1e-6) if np.any(kernel > 0) else 1e-6, max(np.max(kernel) * 1.5, 1.0))
        axis.set_xlabel("output dx")
        axis.set_ylabel("response fraction")
        axis.set_title(
            f"{summary['layout_label']} {summary['case']} 2D x-section, "
            f"N={summary['neighborhood']}, res={summary['resolution_px_per_um']} px/um, "
            f"XT={100 * summary['output_crosstalk_fraction']:.2f}%"
        )
        axis.grid(True, which="both", color="#d7dde5", linewidth=0.6)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_one(
    geom: MicrolensArrayGeometry,
    stack_config: dict[str, Any],
    mode: str,
    neighborhood: int,
    case: CrosstalkCase,
    wavelength_nm: float,
    resolution: int,
    after_source_time: float,
    color_channel: str,
    source_scale: float,
    source_profile: str,
    source_sigma_scale: float,
    dti: OpticalDtiConfig,
    guard_cells: int,
    min_feature_pixels: float,
    min_si_wavelength_pixels: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], np.ndarray]:
    layout_size = mode_layout_size(mode)
    simulation_neighborhood = neighborhood + 2 * guard_cells
    if simulation_neighborhood % 2 != 1:
        raise ValueError("simulation neighborhood must be odd")
    wavelength_um = wavelength_nm / 1000.0
    frequency = 1.0 / wavelength_um
    kx, kz = transverse_k(frequency, case.cra_x_deg, case.cra_z_deg)
    if abs(kz) > 1.0e-12:
        raise ValueError("2D x-section runner only supports cra_z_deg=0")
    supercell_pitch = layout_size * geom.pitch
    simulation_span = simulation_neighborhood * supercell_pitch
    source_size = source_scale * supercell_pitch
    source_sigma = max(source_sigma_scale * supercell_pitch, 1.0e-6)

    silicon, silicon_spec = medium_for_role(stack_config, "silicon", wavelength_um, frequency)
    passivation, passivation_spec = medium_for_role(stack_config, "passivation", wavelength_um, frequency)
    lens, lens_spec = medium_for_role(stack_config, "lens", wavelength_um, frequency)
    cfa_media = {}
    cfa_specs = {}
    for color in ("red", "green", "blue"):
        medium, spec = medium_for_role(stack_config, material_role_for_color(color), wavelength_um, frequency)
        cfa_media[color] = medium
        cfa_specs[color] = spec
    si_n, si_k, _ = nk_from_material(stack_config, "silicon", wavelength_um)
    eps_imag = 2.0 * si_n * si_k

    sim = mp.Simulation(
        cell_size=mp.Vector3(simulation_span + 2 * geom.pml, geom.cell_y, 0),
        boundary_layers=[mp.PML(geom.pml)],
        sources=[
            mp.Source(
                src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
                component=mp.Ez,
                center=mp.Vector3(0, geom.source_y, 0),
                size=mp.Vector3(source_size, 0, 0),
                amp_func=source_amplitude(kx, 0.0, source_profile, source_sigma),
            )
        ],
        resolution=resolution,
        force_complex_fields=True,
        default_material=mp.air,
        extra_materials=[silicon, passivation, lens, *cfa_media.values()],
        material_function=make_material_function_2d(
            geom,
            layout_size,
            simulation_neighborhood,
            case,
            color_channel,
            silicon,
            cfa_media,
            passivation,
            lens,
            passivation,
            dti,
        ),
        dimensions=2,
    )
    fields = sim.add_dft_fields(
        [mp.Ez],
        frequency,
        0,
        1,
        center=mp.Vector3(0, 0.5 * (geom.si_top + geom.si_bottom), 0),
        size=mp.Vector3(simulation_span, geom.si_thickness, 0),
    )
    sim.run(until_after_sources=after_source_time)
    ez = np.asarray(sim.get_dft_array(fields, mp.Ez, 0))
    density = eps_imag * np.abs(ez) ** 2
    total_raw = float(np.sum(density) * (simulation_span / density.shape[0]) * (geom.si_thickness / density.shape[1]))

    output_regions, raw_pd_regions = build_regions_1d(geom, layout_size, neighborhood)
    output_raw = integrate_regions_1d(density, geom, simulation_span, output_regions)
    pd_raw = integrate_regions_1d(density, geom, simulation_span, raw_pd_regions)
    normalize = total_raw if total_raw > 0 else 1.0

    output_rows = []
    output_kernel = np.zeros(neighborhood, dtype=float)
    for region in output_regions:
        value = output_raw[region.region_id] / normalize
        output_kernel[region.ix + neighborhood // 2] = value
        color = bayer_color_for_output(region.ix, 0, color_channel)
        output_rows.append(
            {
                "schema": "camera_crosstalk_xsection_fdtd_v1",
                "simulation_dimension": "2d_xsection",
                "mode": mode,
                "layout_label": layout_label(mode),
                "layout_size": layout_size,
                "neighborhood": neighborhood,
                "simulation_neighborhood": simulation_neighborhood,
                "guard_cells": guard_cells,
                "kernel_scope": "binned_output_xline",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "cra_x_deg": case.cra_x_deg,
                "cra_z_deg": case.cra_z_deg,
                "output_dx": region.ix,
                "output_dz": 0,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": 0.0,
                "response_fraction": value,
                "color": color,
                "color_relation": "target_color" if region.ix == 0 else ("same_color" if color == color_channel else "cross_color"),
                "source_model": "finite_array_center_ocl_impulse_fdtd_2d_xsection",
            }
        )

    raw_rows = []
    for region in raw_pd_regions:
        raw_rows.append(
            {
                "schema": "camera_crosstalk_xsection_fdtd_v1",
                "simulation_dimension": "2d_xsection",
                "mode": mode,
                "layout_label": layout_label(mode),
                "layout_size": layout_size,
                "neighborhood": neighborhood,
                "simulation_neighborhood": simulation_neighborhood,
                "guard_cells": guard_cells,
                "kernel_scope": "raw_pd_xline",
                "case": case.name,
                "wavelength_nm": wavelength_nm,
                "resolution_px_per_um": resolution,
                "color_channel": color_channel,
                "raw_pd_ix": region.ix,
                "raw_pd_iz": 0,
                "region_id": region.region_id,
                "region_x_um": region.x_um,
                "region_z_um": 0.0,
                "response_fraction": pd_raw[region.region_id] / normalize,
                "source_model": "finite_array_center_ocl_impulse_fdtd_2d_xsection",
            }
        )

    center = float(output_kernel[neighborhood // 2])
    border = float(output_kernel[0] + output_kernel[-1])
    off_center = float(np.sum(output_kernel) - center)
    outside = max(0.0, float(1.0 - np.sum(output_kernel)))
    neighbor_kernel = output_kernel.copy()
    neighbor_kernel[neighborhood // 2] = 0.0
    summary = {
        "schema": "camera_crosstalk_xsection_fdtd_v1",
        "simulation_dimension": "2d_xsection",
        "mode": mode,
        "layout_label": layout_label(mode),
        "layout_size": layout_size,
        "neighborhood": neighborhood,
        "simulation_neighborhood": simulation_neighborhood,
        "guard_cells": guard_cells,
        "output_cell_count": neighborhood,
        "raw_pd_kernel_shape": f"{neighborhood * layout_size}x1",
        "raw_pd_count": neighborhood * layout_size,
        "case": case.name,
        "wavelength_nm": wavelength_nm,
        "resolution_px_per_um": resolution,
        "color_channel": color_channel,
        "cra_x_deg": case.cra_x_deg,
        "cra_z_deg": case.cra_z_deg,
        "center_response_fraction": center,
        "output_crosstalk_fraction": off_center,
        "border_response_fraction": border,
        "outside_output_kernel_fraction": outside,
        "truncation_response_fraction": outside,
        "support_edge_response_fraction": border,
        "strongest_neighbor_fraction": float(np.max(neighbor_kernel)),
        "total_integrated_response_fraction": float(np.sum(output_kernel)),
        "total_absorption_raw": total_raw,
        "source_model": "finite_array_center_ocl_impulse_fdtd_2d_xsection",
        "source_profile": source_profile,
        "source_sigma_um": source_sigma,
        "optical_dti_enabled": dti.enabled,
        "optical_dti_width_um": dti.width_um,
        "optical_dti_depth_um": dti.depth_um,
        "optical_dti_measured": dti.measured,
        "accuracy_status": "xsection_fdtd_numerical_gate_not_3d_product_lut",
        "measured_accuracy_blocked": True,
        "notes": (
            "High-resolution 2D FDTD x-section for numerical convergence of CRA-x/DTI lateral crosstalk. "
            "Use the 3D runner for full OCL footprint effects."
        ),
        "materials": {
            "silicon": silicon_spec,
            "passivation": passivation_spec,
            "lens": lens_spec,
            "cfa": cfa_specs,
        },
        "optical_dti": asdict(dti),
    }
    summary.update(
        grid_resolution_metadata(
            geom,
            dti,
            stack_config,
            wavelength_nm,
            resolution,
            min_feature_pixels,
            min_si_wavelength_pixels,
        )
    )
    return output_rows, raw_rows, summary, output_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="split-pd-1x1,ocl-2x2,ocl-3x3")
    parser.add_argument("--neighborhoods", default="5")
    parser.add_argument("--resolutions", default="60,72")
    parser.add_argument("--wavelengths-nm", default="550")
    parser.add_argument("--cases", default="center:0:0:0:0,edge20x:20:0:1:0")
    parser.add_argument("--color-channel", choices=("red", "green", "blue"), default="green")
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--tcad-profile", type=Path, default=DEFAULT_TCAD_PROFILE)
    parser.add_argument("--pml-um", type=float, default=0.45)
    parser.add_argument("--after-source-time", type=float, default=12.0)
    parser.add_argument("--source-scale", type=float, default=0.92)
    parser.add_argument("--source-profile", choices=("gaussian", "rect"), default="gaussian")
    parser.add_argument("--source-sigma-scale", type=float, default=0.30)
    parser.add_argument("--guard-cells", type=int, default=1)
    parser.add_argument("--truncation-threshold", type=float, default=0.015)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--min-feature-pixels", type=float, default=2.0)
    parser.add_argument("--min-si-wavelength-pixels", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/crosstalk_xsection_2d_reference")
    args = parser.parse_args()

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    neighborhoods = parse_csv_ints(args.neighborhoods)
    resolutions = parse_csv_ints(args.resolutions)
    wavelengths = parse_csv_floats(args.wavelengths_nm)
    cases = parse_cases(args.cases)
    stack_config = load_stack_config(args.stack_config)
    shield = shield_config_for_stack(stack_config)
    if shield["enabled"]:
        raise ValueError("2D crosstalk x-section currently supports baseline imaging pixels with shield.mode=off")
    geom = geometry_from_config(stack_config, pml_um=args.pml_um)
    dti = optical_dti_from_profile(args.tcad_profile, geom)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    plot_kernels: list[tuple[dict[str, Any], np.ndarray]] = []
    for mode in modes:
        for neighborhood in neighborhoods:
            if neighborhood % 2 != 1:
                raise ValueError("neighborhoods must be odd")
            for resolution in resolutions:
                for wavelength_nm in wavelengths:
                    for case in cases:
                        print(
                            f"running 2D x-section mode={mode}, neighborhood={neighborhood}, "
                            f"res={resolution}, wavelength={wavelength_nm:g}nm, case={case.name}",
                            flush=True,
                        )
                        out, raw, summary, kernel = run_one(
                            geom,
                            stack_config,
                            mode,
                            neighborhood,
                            case,
                            wavelength_nm,
                            resolution,
                            args.after_source_time,
                            args.color_channel,
                            args.source_scale,
                            args.source_profile,
                            args.source_sigma_scale,
                            dti,
                            args.guard_cells,
                            args.min_feature_pixels,
                            args.min_si_wavelength_pixels,
                        )
                        output_rows.extend(out)
                        raw_rows.extend(raw)
                        summaries.append(summary)
                        if resolution == max(resolutions):
                            plot_kernels.append((summary, kernel))

    output_csv = args.output_dir / "crosstalk_xsection_output_kernel.csv"
    raw_csv = args.output_dir / "crosstalk_xsection_raw_pd_kernel.csv"
    summary_csv = args.output_dir / "crosstalk_xsection_summary.csv"
    plot_png = args.output_dir / "crosstalk_xsection_kernel_lines.png"
    write_csv(output_csv, output_rows)
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summaries)
    save_line_plot(plot_png, plot_kernels)
    convergence = convergence_status(
        summaries,
        args.truncation_threshold,
        args.delta_threshold,
        args.min_feature_pixels,
        args.min_si_wavelength_pixels,
    )
    convergence_path = args.output_dir / "crosstalk_xsection_convergence.json"
    convergence_path.write_text(json.dumps(convergence, indent=2), encoding="utf-8")
    manifest = {
        "schema": "camera_crosstalk_xsection_fdtd_v1",
        "solver": "Meep 2D high-resolution FDTD",
        "source_model": "center OCL finite-aperture impulse x-section",
        "accuracy_status": "xsection_fdtd_numerical_gate_not_3d_product_lut",
        "measured_accuracy_blocked": True,
        "convergence_status": convergence["status"],
        "configuration": {
            "modes": unique_values(summaries, "mode", str),
            "neighborhoods": unique_values(summaries, "neighborhood", int),
            "simulation_neighborhoods": unique_values(summaries, "simulation_neighborhood", int),
            "resolutions_px_per_um": unique_values(summaries, "resolution_px_per_um", int),
            "wavelengths_nm": unique_values(summaries, "wavelength_nm", float),
            "cases": cases_from_summaries(summaries),
            "color_channel": args.color_channel,
            "guard_cells": args.guard_cells,
            "stack_config": str(args.stack_config),
            "tcad_profile": str(args.tcad_profile) if args.tcad_profile else "",
            "geometry_um": asdict(geom),
            "optical_dti": asdict(dti),
            "source_scale": args.source_scale,
            "source_profile": args.source_profile,
            "source_sigma_scale": args.source_sigma_scale,
            "after_source_time": args.after_source_time,
            "min_feature_pixels": args.min_feature_pixels,
            "min_si_wavelength_pixels": args.min_si_wavelength_pixels,
        },
        "scope": {
            "primary_kernel": "binned output-cell x-line crosstalk",
            "diagnostic_kernel": "raw physical-PD x-line crosstalk",
            "resolves": [
                "Si internal wavelength grid gate",
                "DTI/passivation/lens-edge critical feature grid gate",
                "CRA-x lateral crosstalk convergence on a practical local mesh",
            ],
            "does_not_resolve": [
                "full 3D OCL footprint coupling",
                "measured target-product optical n,k",
                "calibrated carrier collection after absorption",
            ],
        },
        "summaries": summaries,
        "convergence": convergence,
        "outputs": {
            "output_kernel_csv": str(output_csv),
            "raw_pd_kernel_csv": str(raw_csv),
            "summary_csv": str(summary_csv),
            "plot_png": str(plot_png),
            "convergence_json": str(convergence_path),
        },
    }
    manifest_path = args.output_dir / "crosstalk_xsection_kernel.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output_dir": str(args.output_dir),
                "summary_count": len(summaries),
                "convergence_status": convergence["status"],
                "outputs": manifest["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
