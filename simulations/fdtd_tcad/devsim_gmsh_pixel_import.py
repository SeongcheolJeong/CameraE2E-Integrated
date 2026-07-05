#!/usr/bin/env python3
"""Import Gmsh pixel meshes into DEVSIM and run a potential smoke solve."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from devsim import (
    add_gmsh_contact,
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_contact_list,
    get_node_model_values,
    get_region_list,
    node_model,
    node_solution,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages import simple_physics
from devsim.python_packages.model_create import CreateSolution

from measured_tcad_profile import electrical_terms_from_profile, load_measured_profile


def import_mesh(mesh_path: Path, dimension: int, device: str, region: str) -> None:
    mesh_name = f"gmsh_pixel_{dimension}d"
    create_gmsh_mesh(mesh=mesh_name, file=str(mesh_path))
    add_gmsh_region(mesh=mesh_name, gmsh_name="silicon", region=region, material="Silicon")
    for contact in ("anode", "cathode_left", "cathode_right"):
        add_gmsh_contact(
            mesh=mesh_name,
            gmsh_name=contact,
            region=region,
            material="metal",
            name=contact,
        )
    finalize_mesh(mesh=mesh_name)
    create_device(mesh=mesh_name, device=device)


def set_measured_doping(device: str, region: str, profile_path: Path | None) -> dict:
    x = np.asarray(get_node_model_values(device=device, region=region, name="x"), dtype=float)
    y = np.asarray(get_node_model_values(device=device, region=region, name="y"), dtype=float)
    try:
        z = np.asarray(get_node_model_values(device=device, region=region, name="z"), dtype=float)
    except Exception:
        z = None

    if profile_path:
        profile = load_measured_profile(profile_path)
        donors, acceptors, fixed_charge, feature_summary = electrical_terms_from_profile(
            profile, x, y, z
        )
        source = str(profile_path)
    else:
        donors = 5.0e15 * (y >= 0.08e-4)
        acceptors = 5.0e16 * (y < 0.08e-4) + 1.0e14
        fixed_charge = np.zeros_like(donors, dtype=float)
        feature_summary = {
            "profile": None,
            "applied_features": [],
            "metadata_only_features": [],
        }
        source = "built_in_proxy"

    node_solution(device=device, region=region, name="Donors")
    node_solution(device=device, region=region, name="Acceptors")
    node_solution(device=device, region=region, name="FixedChargeDoping")
    set_node_values(device=device, region=region, name="Donors", values=donors.tolist())
    set_node_values(device=device, region=region, name="Acceptors", values=acceptors.tolist())
    set_node_values(
        device=device,
        region=region,
        name="FixedChargeDoping",
        values=fixed_charge.tolist(),
    )
    node_model(
        device=device,
        region=region,
        name="NetDoping",
        equation="Donors - Acceptors + FixedChargeDoping",
    )
    net_doping = donors - acceptors + fixed_charge
    return {
        "source": source,
        "donor_min_cm3": float(np.min(donors)),
        "donor_max_cm3": float(np.max(donors)),
        "acceptor_min_cm3": float(np.min(acceptors)),
        "acceptor_max_cm3": float(np.max(acceptors)),
        "fixed_charge_doping_min_cm3": float(np.min(fixed_charge)),
        "fixed_charge_doping_max_cm3": float(np.max(fixed_charge)),
        "net_min_cm3": float(np.min(net_doping)),
        "net_max_cm3": float(np.max(net_doping)),
        "feature_summary": feature_summary,
    }


def potential_solve(device: str, region: str, anode_bias_v: float) -> dict:
    simple_physics.SetSiliconParameters(device, region, 300.0)
    CreateSolution(device, region, "Potential")
    simple_physics.CreateSiliconPotentialOnly(device, region)
    for contact in get_contact_list(device=device):
        value = anode_bias_v if contact == "anode" else 0.0
        set_parameter(
            device=device,
            name=simple_physics.GetContactBiasName(contact),
            value=value,
        )
        simple_physics.CreateSiliconPotentialOnlyContact(device, region, contact)
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=80)
    potential = np.asarray(
        get_node_model_values(device=device, region=region, name="Potential"),
        dtype=float,
    )
    return {
        "anode_bias_v": anode_bias_v,
        "potential_min_v": float(np.min(potential)),
        "potential_max_v": float(np.max(potential)),
    }


def run(args: argparse.Namespace) -> dict:
    device = f"PixelGmsh{args.dimension}D"
    region = "Silicon"
    import_mesh(args.mesh, args.dimension, device, region)
    doping_summary = set_measured_doping(device, region, args.measured_profile)
    solve_summary = potential_solve(device, region, args.anode_bias_v)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tecplot_path = args.output_dir / f"gmsh_pixel_{args.dimension}d_potential.dat"
    write_devices(file=str(tecplot_path), type="tecplot")

    x = get_node_model_values(device=device, region=region, name="x")
    summary = {
        "schema": "devsim_gmsh_pixel_import_smoke_v1",
        "devsim_version": version("devsim"),
        "mesh": str(args.mesh),
        "dimension": args.dimension,
        "device": device,
        "regions": list(get_region_list(device=device)),
        "contacts": list(get_contact_list(device=device)),
        "node_count": len(x),
        "doping": doping_summary,
        "solve": solve_summary,
        "outputs": {"tecplot": str(tecplot_path)},
        "notes": [
            "This verifies Gmsh physical groups, DEVSIM mesh import, contacts, and measured-profile doping plumbing.",
            "Potential-only convergence is not a product accuracy claim.",
        ],
    }
    summary_path = args.output_dir / f"gmsh_pixel_{args.dimension}d_import_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--measured-profile", type=Path, default=None)
    parser.add_argument("--anode-bias-v", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/devsim_gmsh_pixel_import"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
