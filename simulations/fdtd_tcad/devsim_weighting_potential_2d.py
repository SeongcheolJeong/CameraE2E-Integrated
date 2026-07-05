#!/usr/bin/env python3
"""Export DEVSIM-native 2D terminal weighting potentials on a Gmsh pixel mesh.

This solves a pure Laplace equation with Dirichlet contact values. It is a
solver-native weighting-potential export, not a calibrated drift-diffusion
adjoint collection probability.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from devsim import (
    add_gmsh_contact,
    add_gmsh_region,
    contact_equation,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_contact_list,
    get_node_model_values,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import (
    CreateContactNodeModel,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateSolution,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WeightingConfig:
    mesh: Path
    output_dir: Path
    device: str = "SplitPDWeighting2D"
    region: str = "Silicon"
    contacts: tuple[str, ...] = ("anode", "cathode_left", "cathode_right")
    target_contacts: tuple[str, ...] = ("anode", "cathode_left", "cathode_right")
    absolute_error: float = 1.0e-12
    relative_error: float = 1.0e-12
    maximum_iterations: int = 80


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def import_mesh(config: WeightingConfig) -> None:
    mesh_name = "splitpd_weighting_gmsh_mesh"
    create_gmsh_mesh(mesh=mesh_name, file=str(config.mesh))
    add_gmsh_region(mesh=mesh_name, gmsh_name="silicon", region=config.region, material="Silicon")
    for contact in config.contacts:
        add_gmsh_contact(
            mesh=mesh_name,
            gmsh_name=contact,
            region=config.region,
            material="metal",
            name=contact,
        )
    finalize_mesh(mesh=mesh_name)
    create_device(mesh=mesh_name, device=config.device)


def create_laplace_equation(config: WeightingConfig) -> None:
    CreateSolution(config.device, config.region, "Potential")
    flux = "(Potential@n0-Potential@n1)*EdgeInverseLength"
    CreateEdgeModel(config.device, config.region, "WeightingPotentialFlux", flux)
    CreateEdgeModelDerivatives(config.device, config.region, "WeightingPotentialFlux", flux, "Potential")
    # A weighting potential is a pure Laplace solve. Do not use
    # simple_physics.CreateSiliconPotentialOnly here because it adds node charge.
    from devsim import equation

    equation(
        device=config.device,
        region=config.region,
        name="PotentialEquation",
        variable_name="Potential",
        edge_model="WeightingPotentialFlux",
        variable_update="default",
    )
    for contact in get_contact_list(device=config.device):
        parameter_name = f"{contact}_weighting_bias"
        set_parameter(device=config.device, name=parameter_name, value=0.0)
        model_name = f"{contact}_weighting_bc"
        CreateContactNodeModel(config.device, contact, model_name, f"Potential - {parameter_name}")
        CreateContactNodeModel(config.device, contact, f"{model_name}:Potential", "1")
        contact_equation(
            device=config.device,
            contact=contact,
            name="PotentialEquation",
            node_model=model_name,
            edge_model="",
            node_charge_model="",
            edge_charge_model="",
            node_current_model="",
            edge_current_model="",
        )


def solve_for_contact(config: WeightingConfig, target_contact: str) -> np.ndarray:
    for contact in get_contact_list(device=config.device):
        set_parameter(
            device=config.device,
            name=f"{contact}_weighting_bias",
            value=1.0 if contact == target_contact else 0.0,
        )
    # A neutral initial guess reduces cross-contamination between target solves.
    node_count = len(get_node_model_values(device=config.device, region=config.region, name="x"))
    set_node_values(
        device=config.device,
        region=config.region,
        name="Potential",
        values=([1.0 if target_contact == "anode" else 0.0] * node_count),
    )
    solve(
        type="dc",
        absolute_error=config.absolute_error,
        relative_error=config.relative_error,
        maximum_iterations=config.maximum_iterations,
    )
    values = np.asarray(
        get_node_model_values(device=config.device, region=config.region, name="Potential"),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite weighting potential for {target_contact}")
    return np.clip(values, 0.0, 1.0)


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    x = np.asarray([row["x_um"] for row in rows], dtype=float)
    y = np.asarray([row["depth_um"] for row in rows], dtype=float)
    w_left = np.asarray([row["w_cathode_left_devsim_laplace"] for row in rows], dtype=float)
    w_right = np.asarray([row["w_cathode_right_devsim_laplace"] for row in rows], dtype=float)
    w_total = np.asarray([row["w_total_devsim_laplace"] for row in rows], dtype=float)
    plots = [
        (w_left, "W left"),
        (w_right, "W right"),
        (w_right - w_left, "W right-left"),
        (w_total, "W total"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    for axis, (values, title) in zip(axes, plots):
        sc = axis.scatter(x, y, c=values, s=12, cmap="viridis", vmin=-1.0 if "right-left" in title else 0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xlabel("x (um)")
        axis.set_ylabel("depth (um)")
        axis.invert_yaxis()
        fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: WeightingConfig) -> dict[str, Any]:
    if not config.mesh.exists():
        raise FileNotFoundError(f"Gmsh mesh not found: {config.mesh}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import_mesh(config)
    create_laplace_equation(config)

    contact_values = {
        contact: solve_for_contact(config, contact)
        for contact in config.target_contacts
    }
    x_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"), dtype=float)
    y_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"), dtype=float)
    rows: list[dict[str, Any]] = []
    for index, (x_value, y_value) in enumerate(zip(x_cm, y_cm)):
        row: dict[str, Any] = {
            "node_index": index,
            "x_cm": float(x_value),
            "x_um": float(x_value * 1.0e4),
            "y_cm": float(y_value),
            "depth_um": float(y_value * 1.0e4),
        }
        for contact, values in contact_values.items():
            row[f"w_{contact}_devsim_laplace"] = float(values[index])
        row["w_total_devsim_laplace"] = float(
            row.get("w_cathode_left_devsim_laplace", 0.0)
            + row.get("w_cathode_right_devsim_laplace", 0.0)
        )
        row["w_sum_all_contacts_devsim_laplace"] = float(
            sum(row.get(f"w_{contact}_devsim_laplace", 0.0) for contact in config.target_contacts)
        )
        rows.append(row)

    csv_path = config.output_dir / "weighting_potential_2d.csv"
    json_path = config.output_dir / "weighting_potential_2d_summary.json"
    png_path = config.output_dir / "weighting_potential_2d.png"
    dat_path = config.output_dir / "weighting_potential_2d.dat"
    write_csv(csv_path, rows)
    write_devices(file=str(dat_path), type="tecplot")
    save_plot(png_path, rows)

    values_summary = {}
    for contact in config.target_contacts:
        values = contact_values[contact]
        values_summary[contact] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    sum_all = np.asarray([row["w_sum_all_contacts_devsim_laplace"] for row in rows], dtype=float)
    summary = {
        "schema": "devsim_weighting_potential_2d_v1",
        "devsim_version": version("devsim"),
        "method": "pure_laplace_dirichlet_terminal_weighting",
        "config": {**asdict(config), "mesh": str(config.mesh), "output_dir": str(config.output_dir)},
        "contacts": list(get_contact_list(device=config.device)),
        "node_count": len(rows),
        "weighting_summary": values_summary,
        "sum_all_contacts_max_abs_error_to_one": float(np.max(np.abs(sum_all - 1.0))),
        "outputs": {
            "weighting_csv": str(csv_path),
            "tecplot": str(dat_path),
            "plot_png": str(png_path),
            "summary_json": str(json_path),
        },
        "limitations": [
            "This is a DEVSIM-native pure Laplace terminal weighting-potential solve.",
            "It is not a calibrated drift-diffusion adjoint collection probability.",
            "It does not include recombination, mobility, trap occupancy, TG transfer, or FD transient behavior.",
            "Use as W_devsim_laplace trend evidence only until measured calibration targets pass.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=ROOT / "runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/devsim_weighting_potential_2d_reference")
    parser.add_argument("--absolute-error", type=float, default=1.0e-12)
    parser.add_argument("--relative-error", type=float, default=1.0e-12)
    parser.add_argument("--maximum-iterations", type=int, default=80)
    args = parser.parse_args()
    run(
        WeightingConfig(
            mesh=args.mesh,
            output_dir=args.output_dir,
            absolute_error=args.absolute_error,
            relative_error=args.relative_error,
            maximum_iterations=args.maximum_iterations,
        )
    )


if __name__ == "__main__":
    main()
