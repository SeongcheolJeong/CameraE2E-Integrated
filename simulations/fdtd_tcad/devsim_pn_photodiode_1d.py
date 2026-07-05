#!/usr/bin/env python3
"""DEVSIM 1D PN photodiode smoke simulation.

Units follow DEVSIM semiconductor examples: length in cm, densities in cm^-3,
generation in cm^-3 s^-1. This is a pipeline smoke test, not a calibrated image
sensor TCAD model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from devsim import (
    add_1d_contact,
    add_1d_mesh_line,
    add_1d_region,
    create_1d_mesh,
    create_device,
    delete_node_model,
    edge_average_model,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_edge_model_values,
    get_node_model_list,
    get_node_model_values,
    node_solution,
    set_node_values,
    set_parameter,
    solve,
)
from devsim.python_packages.model_create import (
    CreateNodeModel,
    CreateNodeModelDerivative,
    CreateSolution,
)
from devsim.python_packages import simple_physics


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Photodiode1DConfig:
    device: str = "PNPhotodiode1D"
    region: str = "Si"
    length_um: float = 2.8
    junction_um: float = 0.35
    mesh_um: float = 0.04
    junction_mesh_um: float = 0.01
    acceptor_cm3: float = 2.0e17
    donor_cm3: float = 5.0e15
    temperature_k: float = 300.0
    tau_n_s: float = 1.0e-6
    tau_p_s: float = 1.0e-6
    photo_g0_cm3_s: float = 1.0e20
    photo_sigma_um: float = 0.25
    generation_profile_csv: str = ""
    generation_profile_scale: float = 1.0
    generation_profile_case: str = ""
    generation_profile_wavelength_nm: float = 0.0
    reverse_bias_stop_v: float = -2.0
    reverse_bias_step_v: float = -0.25

    @property
    def length_cm(self) -> float:
        return self.length_um * 1.0e-4

    @property
    def junction_cm(self) -> float:
        return self.junction_um * 1.0e-4

    @property
    def mesh_cm(self) -> float:
        return self.mesh_um * 1.0e-4

    @property
    def junction_mesh_cm(self) -> float:
        return self.junction_mesh_um * 1.0e-4

    @property
    def photo_sigma_cm(self) -> float:
        return self.photo_sigma_um * 1.0e-4


def create_mesh(config: Photodiode1DConfig) -> None:
    create_1d_mesh(mesh="pn1d")
    add_1d_mesh_line(mesh="pn1d", pos=0.0, ps=config.mesh_cm, tag="anode")
    add_1d_mesh_line(
        mesh="pn1d",
        pos=config.junction_cm,
        ps=config.junction_mesh_cm,
        tag="junction",
    )
    add_1d_mesh_line(
        mesh="pn1d",
        pos=config.length_cm,
        ps=config.mesh_cm,
        tag="cathode",
    )
    add_1d_contact(mesh="pn1d", name="anode", tag="anode", material="metal")
    add_1d_contact(mesh="pn1d", name="cathode", tag="cathode", material="metal")
    add_1d_region(
        mesh="pn1d",
        material="Si",
        region=config.region,
        tag1="anode",
        tag2="cathode",
    )
    finalize_mesh(mesh="pn1d")
    create_device(mesh="pn1d", device=config.device)


def set_physical_parameters(config: Photodiode1DConfig) -> None:
    simple_physics.SetSiliconParameters(
        config.device, config.region, config.temperature_k
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="AcceptorsP",
        value=config.acceptor_cm3,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="DonorsN",
        value=config.donor_cm3,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="JunctionX",
        value=config.junction_cm,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="PhotoG0",
        value=0.0,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="PhotoSigma",
        value=config.photo_sigma_cm,
    )
    set_parameter(
        device=config.device, region=config.region, name="taun", value=config.tau_n_s
    )
    set_parameter(
        device=config.device, region=config.region, name="taup", value=config.tau_p_s
    )


def create_doping(config: Photodiode1DConfig) -> None:
    CreateNodeModel(
        config.device,
        config.region,
        "Acceptors",
        "AcceptorsP*step(JunctionX - x)",
    )
    CreateNodeModel(
        config.device,
        config.region,
        "Donors",
        "DonorsN*step(x - JunctionX)",
    )
    CreateNodeModel(config.device, config.region, "NetDoping", "Donors - Acceptors")


def set_contact_bias(config: Photodiode1DConfig, anode_bias_v: float) -> None:
    set_parameter(
        device=config.device,
        name=simple_physics.GetContactBiasName("anode"),
        value=anode_bias_v,
    )
    set_parameter(
        device=config.device,
        name=simple_physics.GetContactBiasName("cathode"),
        value=0.0,
    )


def solve_potential_only(config: Photodiode1DConfig) -> None:
    CreateSolution(config.device, config.region, "Potential")
    simple_physics.CreateSiliconPotentialOnly(config.device, config.region)
    for contact in get_contact_list(device=config.device):
        set_parameter(
            device=config.device,
            name=simple_physics.GetContactBiasName(contact),
            value=0.0,
        )
        simple_physics.CreateSiliconPotentialOnlyContact(
            config.device, config.region, contact
        )
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=60)


def replace_node_model(config: Photodiode1DConfig, model: str, expression: str) -> None:
    existing = set(get_node_model_list(device=config.device, region=config.region))
    for candidate in (
        f"{model}:Electrons",
        f"{model}:Holes",
        f"{model}:Potential",
        model,
    ):
        if candidate in existing:
            delete_node_model(
                device=config.device, region=config.region, name=candidate
            )
    CreateNodeModel(config.device, config.region, model, expression)


def imported_generation_enabled(config: Photodiode1DConfig) -> bool:
    return bool(config.generation_profile_csv)


def selected_generation_profile_rows(config: Photodiode1DConfig) -> list[dict]:
    path = Path(config.generation_profile_csv)
    if not path.exists():
        raise FileNotFoundError(f"generation profile not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"generation profile is empty: {path}")

    if config.generation_profile_case:
        rows = [row for row in rows if row.get("case") == config.generation_profile_case]
    if config.generation_profile_wavelength_nm > 0:
        rows = [
            row
            for row in rows
            if abs(float(row.get("wavelength_nm", 0.0)) - config.generation_profile_wavelength_nm)
            < 1e-9
        ]
    if not rows:
        raise RuntimeError(
            "No generation profile rows match requested case/wavelength filters"
        )
    rows.sort(key=lambda row: float(row["depth_um_from_si_top"]))
    return rows


def imported_generation_values(config: Photodiode1DConfig) -> list[float]:
    rows = selected_generation_profile_rows(config)
    profile_depth_um = np.asarray(
        [float(row["depth_um_from_si_top"]) for row in rows], dtype=float
    )
    profile_generation = np.asarray(
        [float(row["generation_cm3_s"]) for row in rows], dtype=float
    )
    x_cm = np.asarray(
        get_node_model_values(device=config.device, region=config.region, name="x"),
        dtype=float,
    )
    x_um = x_cm * 1.0e4
    values = np.interp(
        x_um,
        profile_depth_um,
        profile_generation,
        left=0.0,
        right=0.0,
    )
    return (values * config.generation_profile_scale).tolist()


def set_optical_generation(config: Photodiode1DConfig, illuminated: bool) -> None:
    node_count = len(
        get_node_model_values(device=config.device, region=config.region, name="x")
    )
    if imported_generation_enabled(config):
        values = imported_generation_values(config) if illuminated else [0.0] * node_count
        set_node_values(
            device=config.device,
            region=config.region,
            name="OpticalGenerationRate",
            values=values,
        )
    else:
        set_parameter(
            device=config.device,
            region=config.region,
            name="PhotoG0",
            value=config.photo_g0_cm3_s if illuminated else 0.0,
        )


def create_photo_generation_models(config: Photodiode1DConfig) -> None:
    usrh = "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    electron_generation = (
        f"-ElectronCharge*({usrh}) + ElectronCharge*OpticalGenerationRate"
    )
    hole_generation = f"+ElectronCharge*({usrh}) - ElectronCharge*OpticalGenerationRate"
    if imported_generation_enabled(config):
        node_solution(
            device=config.device,
            region=config.region,
            name="OpticalGenerationRate",
        )
        set_optical_generation(config, illuminated=False)
    else:
        CreateNodeModel(
            config.device,
            config.region,
            "OpticalGenerationRate",
            "PhotoG0*exp(-((x-JunctionX)^2)/(2*PhotoSigma^2))",
        )
    replace_node_model(
        config,
        "ElectronGeneration",
        electron_generation,
    )
    replace_node_model(
        config,
        "HoleGeneration",
        hole_generation,
    )
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "ElectronGeneration",
        electron_generation,
        "Electrons",
        "Holes",
    )
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "HoleGeneration",
        hole_generation,
        "Electrons",
        "Holes",
    )


def solve_drift_diffusion(config: Photodiode1DConfig) -> None:
    CreateSolution(config.device, config.region, "Electrons")
    CreateSolution(config.device, config.region, "Holes")
    set_node_values(
        device=config.device,
        region=config.region,
        name="Electrons",
        init_from="IntrinsicElectrons",
    )
    set_node_values(
        device=config.device,
        region=config.region,
        name="Holes",
        init_from="IntrinsicHoles",
    )
    simple_physics.CreatePE(config.device, config.region)
    simple_physics.CreateBernoulli(config.device, config.region)
    simple_physics.CreateSRH(config.device, config.region)
    create_photo_generation_models(config)
    simple_physics.CreateECE(config.device, config.region, "mu_n")
    simple_physics.CreateHCE(config.device, config.region, "mu_p")
    for contact in get_contact_list(device=config.device):
        simple_physics.CreateSiliconDriftDiffusionAtContact(
            config.device, config.region, contact
        )
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=80)


def contact_currents(config: Photodiode1DConfig, contact: str) -> dict[str, float]:
    electron_current = get_contact_current(
        device=config.device,
        contact=contact,
        equation=simple_physics.ece_name,
    )
    hole_current = get_contact_current(
        device=config.device,
        contact=contact,
        equation=simple_physics.hce_name,
    )
    return {
        f"{contact}_electron_current_a_per_cm2": electron_current,
        f"{contact}_hole_current_a_per_cm2": hole_current,
        f"{contact}_total_current_a_per_cm2": electron_current + hole_current,
    }


def bias_values(config: Photodiode1DConfig) -> list[float]:
    values = [0.0]
    value = config.reverse_bias_step_v
    while value >= config.reverse_bias_stop_v - 1e-12:
        values.append(value)
        value += config.reverse_bias_step_v
    return values


def run_bias_sweep(config: Photodiode1DConfig, condition: str, photo_g0: float) -> list[dict]:
    illuminated = condition != "dark" and photo_g0 != 0.0
    set_optical_generation(config, illuminated=illuminated)
    rows = []
    for bias in bias_values(config):
        set_contact_bias(config, bias)
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=80)
        row = {
            "condition": condition,
            "anode_bias_v": bias,
            "cathode_bias_v": 0.0,
            "photo_g0_cm3_s": photo_g0,
            "generation_source": "imported_profile" if imported_generation_enabled(config) else "analytic_gaussian",
            "generation_profile_csv": config.generation_profile_csv,
            "generation_profile_scale": config.generation_profile_scale,
        }
        row.update(contact_currents(config, "anode"))
        row.update(contact_currents(config, "cathode"))
        row["terminal_current_balance_a_per_cm2"] = (
            row["anode_total_current_a_per_cm2"]
            + row["cathode_total_current_a_per_cm2"]
        )
        rows.append(row)
    return rows


def get_node_profile(config: Photodiode1DConfig) -> list[dict]:
    names = [
        "Potential",
        "Electrons",
        "Holes",
        "NetDoping",
        "Acceptors",
        "Donors",
        "OpticalGenerationRate",
        "IntrinsicElectrons",
        "IntrinsicHoles",
    ]
    values = {
        "x_cm": get_node_model_values(device=config.device, region=config.region, name="x")
    }
    values["x_um"] = [x * 1.0e4 for x in values["x_cm"]]
    for name in names:
        values[name] = get_node_model_values(
            device=config.device, region=config.region, name=name
        )
    edge_average_model(
        device=config.device,
        region=config.region,
        node_model="x",
        edge_model="xmid",
    )
    xmid = np.asarray(
        get_edge_model_values(device=config.device, region=config.region, name="xmid")
    )
    electric_field = np.asarray(
        get_edge_model_values(
            device=config.device, region=config.region, name="ElectricField"
        )
    )
    efield_on_nodes = np.interp(values["x_cm"], xmid, electric_field)
    values["ElectricField_v_per_cm"] = efield_on_nodes.tolist()

    rows = []
    count = len(values["x_cm"])
    for index in range(count):
        rows.append({key: float(value[index]) for key, value in values.items()})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: Path, iv_rows: list[dict], profile_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for condition in sorted({row["condition"] for row in iv_rows}):
        rows = [row for row in iv_rows if row["condition"] == condition]
        axis.plot(
            [row["anode_bias_v"] for row in rows],
            [row["anode_total_current_a_per_cm2"] for row in rows],
            marker="o",
            label=condition,
        )
    axis.set_xlabel("Anode bias vs cathode (V)")
    axis.set_ylabel("Anode current density proxy (A/cm^2)")
    axis.set_title("1D PN photodiode IV smoke")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "iv_curve.png", dpi=180)
    plt.close(fig)

    x = np.asarray([row["x_um"] for row in profile_rows])
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    axes[0].semilogy(x, np.abs([row["NetDoping"] for row in profile_rows]), label="|NetDoping|")
    axes[0].semilogy(x, [row["Electrons"] for row in profile_rows], label="Electrons")
    axes[0].semilogy(x, [row["Holes"] for row in profile_rows], label="Holes")
    axes[0].set_ylabel("Density (cm^-3)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, [row["Potential"] for row in profile_rows], label="Potential")
    axes[1].set_ylabel("Potential (V)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, [row["ElectricField_v_per_cm"] for row in profile_rows], label="ElectricField")
    axes[2].plot(
        x,
        np.asarray([row["OpticalGenerationRate"] for row in profile_rows])
        / max(max(row["OpticalGenerationRate"] for row in profile_rows), 1.0)
        * max(np.abs([row["ElectricField_v_per_cm"] for row in profile_rows])),
        label="OpticalGeneration normalized",
        alpha=0.7,
    )
    axes[2].set_xlabel("Depth x (um)")
    axes[2].set_ylabel("Electric field (V/cm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.savefig(output_dir / "final_profile.png", dpi=180)
    plt.close(fig)


def run(config: Photodiode1DConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    create_mesh(config)
    set_physical_parameters(config)
    create_doping(config)
    set_contact_bias(config, 0.0)
    solve_potential_only(config)
    solve_drift_diffusion(config)

    dark_rows = run_bias_sweep(config, "dark", 0.0)
    photo_rows = run_bias_sweep(config, "illuminated", config.photo_g0_cm3_s)
    iv_rows = dark_rows + photo_rows
    profile_rows = get_node_profile(config)

    write_csv(output_dir / "iv.csv", iv_rows)
    write_csv(output_dir / "final_node_profile.csv", profile_rows)
    save_plots(output_dir, iv_rows, profile_rows)

    last_dark = dark_rows[-1]
    last_photo = photo_rows[-1]
    summary = {
        "schema": "devsim_pn_photodiode_1d_smoke_v1",
        "devsim_version": version("devsim"),
        "config": asdict(config),
        "generation_source": "imported_profile" if imported_generation_enabled(config) else "analytic_gaussian",
        "generation_profile_csv": config.generation_profile_csv or None,
        "generation_profile_scale": config.generation_profile_scale,
        "reverse_bias_stop_v": config.reverse_bias_stop_v,
        "dark_current_at_stop_a_per_cm2": last_dark["anode_total_current_a_per_cm2"],
        "illuminated_current_at_stop_a_per_cm2": last_photo[
            "anode_total_current_a_per_cm2"
        ],
        "photo_delta_current_at_stop_a_per_cm2": (
            last_photo["anode_total_current_a_per_cm2"]
            - last_dark["anode_total_current_a_per_cm2"]
        ),
        "terminal_current_balance_at_stop_a_per_cm2": last_photo[
            "terminal_current_balance_a_per_cm2"
        ],
        "node_count": len(profile_rows),
        "outputs": {
            "iv_csv": str(output_dir / "iv.csv"),
            "final_node_profile_csv": str(output_dir / "final_node_profile.csv"),
            "iv_curve_png": str(output_dir / "iv_curve.png"),
            "final_profile_png": str(output_dir / "final_profile.png"),
        },
        "notes": [
            "This is a DEVSIM smoke test for PN-junction drift-diffusion with a simple optical generation term.",
            "The structure is not a calibrated CMOS image-sensor pixel.",
            "Current is reported as a 1D unit-area current-density proxy.",
            "Imported generation profiles are depth-only 1D collapses of FDTD Si absorption unless a richer mesh is supplied.",
            "Next step is a 2D/3D pixel mesh for lateral split-PD/OCL collection.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "devsim_pn_photodiode_1d")
    parser.add_argument("--length-um", type=float, default=2.8)
    parser.add_argument("--junction-um", type=float, default=0.35)
    parser.add_argument("--mesh-um", type=float, default=0.04)
    parser.add_argument("--junction-mesh-um", type=float, default=0.01)
    parser.add_argument("--acceptor-cm3", type=float, default=2.0e17)
    parser.add_argument("--donor-cm3", type=float, default=5.0e15)
    parser.add_argument("--photo-g0-cm3-s", type=float, default=1.0e20)
    parser.add_argument("--photo-sigma-um", type=float, default=0.25)
    parser.add_argument(
        "--generation-profile-csv",
        type=Path,
        default=None,
        help="FDTD-exported tcad_generation_profile_1d.csv to use instead of analytic Gaussian generation.",
    )
    parser.add_argument("--generation-profile-scale", type=float, default=1.0)
    parser.add_argument("--generation-profile-case", default="")
    parser.add_argument("--generation-profile-wavelength-nm", type=float, default=0.0)
    parser.add_argument("--reverse-bias-stop-v", type=float, default=-2.0)
    parser.add_argument("--reverse-bias-step-v", type=float, default=-0.25)
    args = parser.parse_args()

    config = Photodiode1DConfig(
        length_um=args.length_um,
        junction_um=args.junction_um,
        mesh_um=args.mesh_um,
        junction_mesh_um=args.junction_mesh_um,
        acceptor_cm3=args.acceptor_cm3,
        donor_cm3=args.donor_cm3,
        photo_g0_cm3_s=args.photo_g0_cm3_s,
        photo_sigma_um=args.photo_sigma_um,
        generation_profile_csv=str(args.generation_profile_csv) if args.generation_profile_csv else "",
        generation_profile_scale=args.generation_profile_scale,
        generation_profile_case=args.generation_profile_case,
        generation_profile_wavelength_nm=args.generation_profile_wavelength_nm,
        reverse_bias_stop_v=args.reverse_bias_stop_v,
        reverse_bias_step_v=args.reverse_bias_step_v,
    )
    summary = run(config, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
