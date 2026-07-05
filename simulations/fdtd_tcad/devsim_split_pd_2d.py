#!/usr/bin/env python3
"""DEVSIM 2D split-PD smoke simulation with imported FDTD generation.

This is a lateral-collection smoke model:
  x: lateral pixel coordinate in cm
  y: depth from Si top in cm

The structure is a p+/n diode with one top anode and two bottom cathodes. It is
not a calibrated pinned-photodiode pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from devsim import (
    add_2d_contact,
    add_2d_mesh_line,
    add_2d_region,
    add_circuit_node,
    add_gmsh_contact,
    add_gmsh_interface,
    add_gmsh_region,
    circuit_alter,
    circuit_element,
    contact_equation,
    create_2d_mesh,
    create_device,
    create_gmsh_mesh,
    delete_node_model,
    edge_average_model,
    finalize_mesh,
    get_contact_current,
    get_contact_list,
    get_circuit_node_list,
    get_edge_model_values,
    get_interface_list,
    get_node_model_list,
    get_node_model_values,
    get_region_list,
    node_solution,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)
from devsim.python_packages.model_create import (
    CreateContactNodeModel,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateNodeModel,
    CreateNodeModelDerivative,
    CreateSolution,
    InEdgeModelList,
)
from devsim.python_packages import simple_physics
from measured_tcad_profile import electrical_terms_from_profile, load_measured_profile


ROOT = Path(__file__).resolve().parent
Q_E = 1.602176634e-19


@dataclass(frozen=True)
class SplitPD2DConfig:
    device: str = "SplitPD2D"
    region: str = "Si"
    oxide_region: str = "Oxide"
    width_um: float = 1.4
    depth_um: float = 2.8
    junction_um: float = 0.35
    mesh_x_um: float = 0.10
    mesh_y_um: float = 0.08
    junction_mesh_um: float = 0.025
    split_gap_um: float = 0.04
    acceptor_cm3: float = 2.0e17
    donor_cm3: float = 5.0e15
    temperature_k: float = 300.0
    tau_n_s: float = 1.0e-6
    tau_p_s: float = 1.0e-6
    photo_g0_cm3_s: float = 1.0e20
    photo_sigma_y_um: float = 0.25
    photo_sigma_x_um: float = 0.35
    photo_shift_x_um: float = 0.0
    generation_profile_csv: str = ""
    generation_profile_scale: float = 1.0
    generation_profile_case: str = ""
    generation_profile_wavelength_nm: float = 0.0
    generation_lateral_mode: str = "uniform"
    generation_map_npz: str = ""
    generation_map_scale: float = 1.0
    normalize_generation_map_integral: bool = True
    generation_probe_g0_cm3_s: float = 0.0
    generation_probe_x_um: float = 0.0
    generation_probe_depth_um: float = 0.0
    generation_probe_sigma_x_um: float = 0.07
    generation_probe_sigma_y_um: float = 0.10
    electrical_model: str = "proxy-pinned-split-pd"
    measured_profile: str = ""
    fixed_charge_sheet_thickness_um: float = 0.02
    pinning_depth_um: float = 0.08
    pinning_acceptor_cm3: float = 5.0e16
    substrate_acceptor_cm3: float = 1.0e14
    collection_donor_cm3: float = 5.0e15
    isolation_acceptor_cm3: float = 2.0e16
    dti_width_um: float = 0.05
    dti_acceptor_cm3: float = 2.0e16
    interface_trap_energy_width_ev: float = 0.56
    interface_trap_reference_potential_v: float = 0.0
    interface_trap_broadening_v: float = 0.02585
    interface_trap_thermal_velocity_cm_s: float = 1.0e7
    electron_mobility_scale: float = 1.0
    hole_mobility_scale: float = 1.0
    lifetime_scale: float = 1.0
    transport_override: str = "profile"
    disable_field_mobility: bool = False
    fixed_charge_scale: float = 1.0
    interface_trap_density_scale: float = 1.0
    interface_trap_recombination_scale: float = 1.0
    floating_diffusion_feature_scale: float = 1.0
    transfer_gate_barrier_feature_scale: float = 1.0
    bdti_liner_feature_scale: float = 1.0
    resolved_bdti_sidewall_liner: bool = False
    mu_n_cm2_v_s: float = 400.0
    mu_p_cm2_v_s: float = 200.0
    reverse_bias_v: float = -1.0
    floating_diffusion_bias_v: float = 0.0
    floating_diffusion_circuit: bool = False
    floating_diffusion_capacitance_f_per_cm: float = 1.0e-11
    floating_diffusion_reset_on_resistance_ohm_cm: float = 1.0e3
    floating_diffusion_reset_off_resistance_ohm_cm: float = 1.0e15
    transfer_gate_bias_v: float = 0.0
    transfer_gate_capacitive_coupling: bool = False
    transfer_gate_coupling_sign: float = -1.0
    dd_absolute_error: float = 1.0e10
    dd_relative_error: float = 1.0e-9
    dd_max_iterations: int = 160
    mesh_source: str = "internal"
    gmsh_mesh: str = ""

    @property
    def half_width_cm(self) -> float:
        return 0.5 * self.width_um * 1.0e-4

    @property
    def depth_cm(self) -> float:
        return self.depth_um * 1.0e-4

    @property
    def junction_cm(self) -> float:
        return self.junction_um * 1.0e-4

    @property
    def mesh_x_cm(self) -> float:
        return self.mesh_x_um * 1.0e-4

    @property
    def mesh_y_cm(self) -> float:
        return self.mesh_y_um * 1.0e-4

    @property
    def junction_mesh_cm(self) -> float:
        return self.junction_mesh_um * 1.0e-4

    @property
    def split_gap_cm(self) -> float:
        return self.split_gap_um * 1.0e-4

    @property
    def photo_sigma_x_cm(self) -> float:
        return self.photo_sigma_x_um * 1.0e-4

    @property
    def photo_sigma_y_cm(self) -> float:
        return self.photo_sigma_y_um * 1.0e-4

    @property
    def photo_shift_x_cm(self) -> float:
        return self.photo_shift_x_um * 1.0e-4

    @property
    def generation_probe_x_cm(self) -> float:
        return self.generation_probe_x_um * 1.0e-4

    @property
    def generation_probe_depth_cm(self) -> float:
        return self.generation_probe_depth_um * 1.0e-4

    @property
    def generation_probe_sigma_x_cm(self) -> float:
        return self.generation_probe_sigma_x_um * 1.0e-4

    @property
    def generation_probe_sigma_y_cm(self) -> float:
        return self.generation_probe_sigma_y_um * 1.0e-4

    @property
    def pinning_depth_cm(self) -> float:
        return self.pinning_depth_um * 1.0e-4

    @property
    def dti_width_cm(self) -> float:
        return self.dti_width_um * 1.0e-4


def create_mesh(config: SplitPD2DConfig) -> None:
    mesh = "splitpd2d_mesh"
    xmin = -config.half_width_cm
    xmax = config.half_width_cm
    ymin = 0.0
    ymax = config.depth_cm
    gap = 0.5 * config.split_gap_cm
    contact_dummy_thickness = max(0.25 * config.mesh_y_cm, 1.0e-7)

    create_2d_mesh(mesh=mesh)
    x_positions = {
        xmin,
        xmax,
        -gap,
        0.0,
        gap,
    }
    lateral_steps = max(int(np.ceil(config.half_width_cm / config.mesh_x_cm)), 1)
    for index in range(lateral_steps + 1):
        x = config.half_width_cm * index / lateral_steps
        x_positions.add(x)
        x_positions.add(-x)
    for x in sorted(x_positions):
        spacing = config.junction_mesh_cm if abs(x) <= gap + config.mesh_x_cm else config.mesh_x_cm
        add_2d_mesh_line(mesh=mesh, dir="x", pos=x, ps=spacing, ns=spacing)
    for y, spacing in (
        (ymin - contact_dummy_thickness, config.mesh_y_cm),
        (ymin, config.mesh_y_cm),
        (config.junction_cm, config.junction_mesh_cm),
        (ymax, config.mesh_y_cm),
        (ymax + contact_dummy_thickness, config.mesh_y_cm),
    ):
        add_2d_mesh_line(mesh=mesh, dir="y", pos=y, ps=spacing, ns=spacing)

    # DEVSIM's 2D mesh generator assigns contacts through neighboring region
    # ownership. Thin dummy regions make the top and split bottom contacts
    # explicit while keeping equations only on the silicon region.
    add_2d_region(
        mesh=mesh,
        region="top_contact_dummy",
        material="metal",
        xl=xmin,
        xh=xmax,
        yl=ymin - contact_dummy_thickness,
        yh=ymin,
    )
    add_2d_region(
        mesh=mesh,
        region="bottom_left_contact_dummy",
        material="metal",
        xl=xmin,
        xh=-gap,
        yl=ymax,
        yh=ymax + contact_dummy_thickness,
    )
    add_2d_region(
        mesh=mesh,
        region="bottom_gap_dummy",
        material="metal",
        xl=-gap,
        xh=gap,
        yl=ymax,
        yh=ymax + contact_dummy_thickness,
    )
    add_2d_region(
        mesh=mesh,
        region="bottom_right_contact_dummy",
        material="metal",
        xl=gap,
        xh=xmax,
        yl=ymax,
        yh=ymax + contact_dummy_thickness,
    )
    add_2d_region(
        mesh=mesh,
        region=config.region,
        material="Si",
        xl=xmin,
        xh=xmax,
        yl=ymin,
        yh=ymax,
    )
    add_2d_contact(
        mesh=mesh,
        name="anode",
        region=config.region,
        material="metal",
        xl=xmin,
        xh=xmax,
        yl=ymin,
        yh=ymin,
        bloat=1.0e-10,
    )
    add_2d_contact(
        mesh=mesh,
        name="cathode_left",
        region=config.region,
        material="metal",
        xl=xmin,
        xh=-gap,
        yl=ymax,
        yh=ymax,
        bloat=1.0e-10,
    )
    add_2d_contact(
        mesh=mesh,
        name="cathode_right",
        region=config.region,
        material="metal",
        xl=gap,
        xh=xmax,
        yl=ymax,
        yh=ymax,
        bloat=1.0e-10,
    )
    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=config.device)


def import_gmsh_mesh(config: SplitPD2DConfig) -> None:
    if not config.gmsh_mesh:
        raise ValueError("--gmsh-mesh is required when --mesh-source=gmsh")
    mesh_path = Path(config.gmsh_mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Gmsh mesh not found: {mesh_path}")
    mesh = "splitpd2d_gmsh_mesh"
    create_gmsh_mesh(mesh=mesh, file=str(mesh_path))
    add_gmsh_region(mesh=mesh, gmsh_name="silicon", region=config.region, material="Silicon")
    physical_names = gmsh_physical_names(mesh_path)
    if "oxide" in physical_names.get(2, []):
        add_gmsh_region(
            mesh=mesh,
            gmsh_name="oxide",
            region=config.oxide_region,
            material="Oxide",
        )
    contact_names = gmsh_contact_names(mesh_path)
    transfer_gate_on_oxide = "silicon_oxide_interface" in physical_names.get(1, [])
    for contact in contact_names:
        contact_region = (
            config.oxide_region
            if contact == "transfer_gate" and has_gmsh_region(mesh_path, "oxide") and transfer_gate_on_oxide
            else config.region
        )
        add_gmsh_contact(
            mesh=mesh,
            gmsh_name=contact,
            region=contact_region,
            material="metal",
            name=contact,
        )
    if "silicon_oxide_interface" in physical_names.get(1, []):
        add_gmsh_interface(
            mesh=mesh,
            gmsh_name="silicon_oxide_interface",
            region0=config.region,
            region1=config.oxide_region,
            name="silicon_oxide",
        )
    finalize_mesh(mesh=mesh)
    create_device(mesh=mesh, device=config.device)


def gmsh_physical_names(path: Path) -> dict[int, list[str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names: dict[int, list[str]] = {}
    if "$PhysicalNames" not in lines:
        return names
    start = lines.index("$PhysicalNames") + 1
    count = int(lines[start])
    for line in lines[start + 1 : start + 1 + count]:
        dim_text, _tag_text, raw_name = line.split(maxsplit=2)
        dim = int(dim_text)
        names.setdefault(dim, []).append(raw_name.strip('"'))
    return names


def has_gmsh_region(path: Path, name: str) -> bool:
    return name in gmsh_physical_names(path).get(2, [])


def gmsh_contact_names(path: Path) -> list[str]:
    names = [
        name
        for name in gmsh_physical_names(path).get(1, [])
        if name != "insulated" and not name.endswith("_interface")
    ]
    if not names:
        return ["anode", "cathode_left", "cathode_right"]
    required = ["anode", "cathode_left", "cathode_right"]
    ordered = [name for name in required if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def create_or_import_mesh(config: SplitPD2DConfig) -> None:
    if config.mesh_source == "internal":
        create_mesh(config)
    elif config.mesh_source == "gmsh":
        import_gmsh_mesh(config)
    else:
        raise ValueError(f"Unsupported mesh_source: {config.mesh_source}")


def set_parameters(config: SplitPD2DConfig) -> None:
    simple_physics.SetSiliconParameters(config.device, config.region, config.temperature_k)
    for name, value in (
        ("AcceptorsP", config.acceptor_cm3),
        ("DonorsN", config.donor_cm3),
        ("JunctionY", config.junction_cm),
        ("XMin", -config.half_width_cm),
        ("XMax", config.half_width_cm),
        ("DepthY", config.depth_cm),
        ("GapHalfWidth", 0.5 * config.split_gap_cm),
        ("PinningDepth", config.pinning_depth_cm),
        ("PinningAcceptors", config.pinning_acceptor_cm3),
        ("SubstrateAcceptors", config.substrate_acceptor_cm3),
        ("CollectionDonors", config.collection_donor_cm3),
        ("IsolationAcceptors", config.isolation_acceptor_cm3),
        ("DtiWidth", config.dti_width_cm),
        ("DtiAcceptors", config.dti_acceptor_cm3),
        ("PhotoG0", 0.0),
        ("PhotoSigmaX", config.photo_sigma_x_cm),
        ("PhotoSigmaY", config.photo_sigma_y_cm),
        ("PhotoShiftX", config.photo_shift_x_cm),
        ("mu_n", config.mu_n_cm2_v_s * config.electron_mobility_scale),
        ("mu_p", config.mu_p_cm2_v_s * config.hole_mobility_scale),
        ("taun", config.tau_n_s * config.lifetime_scale),
        ("taup", config.tau_p_s * config.lifetime_scale),
    ):
        set_parameter(device=config.device, region=config.region, name=name, value=value)
    if config.measured_profile and config.electrical_model == "profile-ppd":
        profile = load_measured_profile(config.measured_profile)
        transport = profile.data.get("mobility_recombination", {})
        for profile_key, parameter_name, scale in (
            ("mu_n_cm2_v_s", "mu_n", config.electron_mobility_scale),
            ("mu_p_cm2_v_s", "mu_p", config.hole_mobility_scale),
            ("tau_n_s", "taun", config.lifetime_scale),
            ("tau_p_s", "taup", config.lifetime_scale),
        ):
            if profile_key in transport:
                set_parameter(
                    device=config.device,
                    region=config.region,
                    name=parameter_name,
                    value=float(transport[profile_key]) * float(scale),
                )


def profile_electrical_terms(config: SplitPD2DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not config.measured_profile:
        raise ValueError("profile-ppd electrical model requires --measured-profile")
    profile = load_measured_profile(config.measured_profile)
    x = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"))
    y = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"))
    move_bdti_to_sidewall = resolved_oxide_enabled(config) and config.resolved_bdti_sidewall_liner
    donors, acceptors, fixed_charge, summary = electrical_terms_from_profile(
        profile,
        x,
        y,
        None,
        default_sheet_thickness_um=config.fixed_charge_sheet_thickness_um,
        feature_role_scales={
            "floating_diffusion": config.floating_diffusion_feature_scale,
            "transfer_gate_barrier": config.transfer_gate_barrier_feature_scale,
            "bdti_liner": 0.0 if move_bdti_to_sidewall else config.bdti_liner_feature_scale,
        },
    )
    if move_bdti_to_sidewall:
        apply_resolved_bdti_sidewall_liner(
            profile,
            x,
            y,
            donors,
            acceptors,
            summary,
            config.bdti_liner_feature_scale,
        )
    fixed_charge *= config.fixed_charge_scale
    summary.setdefault("runtime_calibration_scales", {})[
        "fixed_charge_scale"
    ] = config.fixed_charge_scale
    return donors, acceptors, fixed_charge, summary


def bdti_liner_feature_doses(profile) -> dict[str, dict[str, float]]:
    doses = {
        "left": {"donor_cm3": 0.0, "acceptor_cm3": 0.0, "scale": 1.0},
        "right": {"donor_cm3": 0.0, "acceptor_cm3": 0.0, "scale": 1.0},
    }
    for feature in profile.electrical_features:
        if feature.get("role") != "bdti_liner":
            continue
        x_mid = 0.5 * (float(feature.get("x_min_um", 0.0)) + float(feature.get("x_max_um", 0.0)))
        side = "left" if x_mid < 0.0 else "right"
        doses[side] = {
            "donor_cm3": float(feature.get("donor_cm3", 0.0)),
            "acceptor_cm3": float(feature.get("acceptor_cm3", 0.0)),
            "scale": float(feature.get("scale", 1.0)),
        }
    return doses


def apply_resolved_bdti_sidewall_liner(
    profile,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    donors: np.ndarray,
    acceptors: np.ndarray,
    summary: dict,
    runtime_scale: float,
) -> None:
    """Move proxy BDTI liner support to the silicon sidewall for oxide meshes."""
    bdti = profile.geometry.get("bdti", {})
    if not bdti.get("enabled", False):
        return
    liner_width_um = float(bdti.get("liner_width_um", 0.0))
    if liner_width_um <= 0.0:
        return
    depth_min_um = float(bdti.get("depth_min_um", 0.0))
    depth_max_um = float(bdti.get("depth_max_um", profile.geometry.get("depth_um", 0.0)))
    left_inner_um = float(bdti.get("x_left_max_um", -0.5 * float(profile.geometry["width_um"])))
    right_inner_um = float(bdti.get("x_right_min_um", 0.5 * float(profile.geometry["width_um"])))
    x_um = np.asarray(x_cm, dtype=float) * 1.0e4
    depth_um = np.asarray(y_cm, dtype=float) * 1.0e4
    depth_mask = (depth_um >= depth_min_um) & (depth_um <= depth_max_um)
    side_masks = {
        "left": depth_mask & (x_um >= left_inner_um) & (x_um <= left_inner_um + liner_width_um),
        "right": depth_mask & (x_um >= right_inner_um - liner_width_um) & (x_um <= right_inner_um),
    }
    doses = bdti_liner_feature_doses(profile)
    applied = []
    for side, mask in side_masks.items():
        dose = doses[side]
        donor_cm3 = dose["donor_cm3"] * dose["scale"] * runtime_scale
        acceptor_cm3 = dose["acceptor_cm3"] * dose["scale"] * runtime_scale
        donors += mask * donor_cm3
        acceptors += mask * acceptor_cm3
        applied.append(
            {
                "name": f"{side}_resolved_bdti_sidewall_liner",
                "role": "bdti_liner",
                "type": "sidewall_liner_from_profile_bdti",
                "measured": False,
                "active_node_count": int(np.count_nonzero(mask)),
                "scale": runtime_scale,
                "donor_cm3": donor_cm3,
                "acceptor_cm3": acceptor_cm3,
                "runtime_role_scale": runtime_scale,
                "x_inner_um": left_inner_um if side == "left" else right_inner_um,
                "liner_width_um": liner_width_um,
                "depth_min_um": depth_min_um,
                "depth_max_um": depth_max_um,
            }
        )
    feature_summary = summary.get("feature_summary") if isinstance(summary.get("feature_summary"), dict) else summary
    feature_summary.setdefault("applied_features", []).extend(applied)
    feature_summary.setdefault("notes", []).append(
        "Resolved oxide DTI meshes apply the proxy BDTI p-liner on the silicon sidewall, not inside the oxide trench volume."
    )


def interface_location_depth_um(item: dict) -> float:
    if "location_depth_um" in item:
        return float(item["location_depth_um"])
    location = str(item.get("location", "depth_um=0"))
    if location.startswith("depth_um="):
        return float(location.split("=", 1)[1])
    raise ValueError(f"unsupported interface location: {location}")


def create_interface_trap_sources(config: SplitPD2DConfig) -> dict:
    x = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"))
    y = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"))
    trap_density = np.zeros_like(x, dtype=float)
    recombination_coeff = np.zeros_like(x, dtype=float)
    applied: list[dict] = []
    metadata_only: list[str] = []

    if config.measured_profile and config.electrical_model == "profile-ppd":
        profile = load_measured_profile(config.measured_profile)
        x_um = x * 1.0e4
        depth_um = y * 1.0e4
        for interface in profile.interfaces:
            if not interface.get("enabled", True) or "dit_cm2_ev" not in interface:
                continue
            name = interface.get("name", "interface")
            dit_cm2_ev = float(interface.get("dit_cm2_ev", 0.0))
            if dit_cm2_ev <= 0.0:
                metadata_only.append(name)
                continue
            thickness_um = float(
                interface.get("sheet_thickness_um", config.fixed_charge_sheet_thickness_um)
            )
            if thickness_um <= 0.0:
                raise ValueError(f"{name}: sheet_thickness_um must be > 0")
            half_thickness_um = 0.5 * thickness_um
            center_depth_um = interface_location_depth_um(interface)
            mask = np.abs(depth_um - center_depth_um) <= half_thickness_um
            if "x_min_um" in interface and "x_max_um" in interface:
                mask &= (x_um >= float(interface["x_min_um"])) & (
                    x_um <= float(interface["x_max_um"])
                )
            active_count = int(np.count_nonzero(mask))
            if active_count == 0:
                metadata_only.append(name)
                continue
            energy_width_ev = float(
                interface.get("trap_energy_width_ev", config.interface_trap_energy_width_ev)
            )
            sheet_density_cm2_unscaled = dit_cm2_ev * energy_width_ev
            sheet_density_cm2 = sheet_density_cm2_unscaled * config.interface_trap_density_scale
            recombination_sheet_density_cm2 = (
                sheet_density_cm2_unscaled * config.interface_trap_recombination_scale
            )
            thickness_cm = thickness_um * 1.0e-4
            density_cm3 = sheet_density_cm2 / thickness_cm
            sigma_n = float(interface.get("sigma_n_cm2", 0.0))
            sigma_p = float(interface.get("sigma_p_cm2", 0.0))
            sigma_eff = (max(sigma_n, 0.0) * max(sigma_p, 0.0)) ** 0.5
            surface_velocity_cm_s = (
                config.interface_trap_thermal_velocity_cm_s
                * sigma_eff
                * recombination_sheet_density_cm2
            )
            trap_density[mask] += density_cm3
            recombination_coeff[mask] += surface_velocity_cm_s / thickness_cm
            applied.append(
                {
                    "name": name,
                    "role": "interface_trap_occupancy",
                    "type": "interface_trap_srh_sheet",
                    "measured": bool(interface.get("measured", False)),
                    "active_node_count": active_count,
                    "dit_cm2_ev": dit_cm2_ev,
                    "energy_width_ev": energy_width_ev,
                    "density_scale": config.interface_trap_density_scale,
                    "recombination_scale": config.interface_trap_recombination_scale,
                    "sheet_density_cm2": sheet_density_cm2,
                    "unscaled_sheet_density_cm2": sheet_density_cm2_unscaled,
                    "sheet_thickness_um": thickness_um,
                    "sigma_n_cm2": sigma_n,
                    "sigma_p_cm2": sigma_p,
                    "surface_velocity_cm_s": surface_velocity_cm_s,
                }
            )

    for model, values in (
        ("InterfaceTrapDensityVolume", trap_density),
        ("InterfaceTrapRecombinationCoeff", recombination_coeff),
    ):
        node_solution(device=config.device, region=config.region, name=model)
        set_node_values(device=config.device, region=config.region, name=model, values=values.tolist())

    set_parameter(
        device=config.device,
        region=config.region,
        name="InterfaceTrapReferencePotential",
        value=config.interface_trap_reference_potential_v,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="InterfaceTrapBroadeningVoltage",
        value=config.interface_trap_broadening_v,
    )
    return {
        "model": "potential_dependent_interface_trap_charge_and_srh_sheet_v1",
        "applied_interface_traps": applied,
        "metadata_only_interface_traps": metadata_only,
        "trap_density_min_cm3": float(np.min(trap_density)),
        "trap_density_max_cm3": float(np.max(trap_density)),
        "recombination_coeff_min_s1": float(np.min(recombination_coeff)),
        "recombination_coeff_max_s1": float(np.max(recombination_coeff)),
        "reference_potential_v": config.interface_trap_reference_potential_v,
        "broadening_v": config.interface_trap_broadening_v,
        "thermal_velocity_cm_s": config.interface_trap_thermal_velocity_cm_s,
        "runtime_calibration_scales": {
            "interface_trap_density_scale": config.interface_trap_density_scale,
            "interface_trap_recombination_scale": config.interface_trap_recombination_scale,
        },
    }


def attach_interface_trap_summary(doping_summary: dict, trap_summary: dict) -> None:
    feature_summary = doping_summary.get("feature_summary", {})
    applied_traps = trap_summary.get("applied_interface_traps", [])
    if not applied_traps:
        return
    applied_names = {str(item.get("name", "")) for item in applied_traps}
    metadata_only = [
        name
        for name in feature_summary.get("metadata_only_features", [])
        if str(name) not in applied_names
    ]
    feature_summary["metadata_only_features"] = metadata_only
    feature_summary.setdefault("applied_features", []).extend(applied_traps)
    notes = feature_summary.setdefault("notes", [])
    notes.append(
        "Interface Dit terms are coupled into DEVSIM through potential-dependent trap charge and SRH sheet recombination proxies."
    )


def create_interface_trap_models(config: SplitPD2DConfig) -> None:
    traps_disabled = (
        config.interface_trap_density_scale == 0.0
        and config.interface_trap_recombination_scale == 0.0
    )
    if traps_disabled:
        replace_node_model(config, "InterfaceTrapOccupancy", "0")
        replace_node_model(config, "InterfaceTrapChargeDoping", "0")
        potential_charge = "-ElectronCharge*kahan3(Holes, -Electrons, NetDoping)"
        replace_node_model(config, "PotentialNodeCharge", potential_charge)
        CreateNodeModelDerivative(
            config.device,
            config.region,
            "PotentialNodeCharge",
            potential_charge,
            "Electrons",
            "Holes",
            "Potential",
        )
        replace_node_model(config, "InterfaceTrapRecombinationRate", "0")
        CreateNodeModelDerivative(
            config.device,
            config.region,
            "InterfaceTrapRecombinationRate",
            "0",
            "Electrons",
            "Holes",
        )
        return

    occupancy = (
        "1/(1 + exp((InterfaceTrapReferencePotential - Potential)"
        "/InterfaceTrapBroadeningVoltage))"
    )
    trap_charge = "-InterfaceTrapDensityVolume*InterfaceTrapOccupancy"
    potential_charge = "-ElectronCharge*kahan3(Holes, -Electrons, NetDoping + InterfaceTrapChargeDoping)"
    recombination = (
        "InterfaceTrapRecombinationCoeff*(Electrons*Holes - n_i^2)"
        "/(Electrons + Holes + 2*n_i)"
    )
    replace_node_model(config, "InterfaceTrapOccupancy", occupancy)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "InterfaceTrapOccupancy",
        occupancy,
        "Potential",
    )
    replace_node_model(config, "InterfaceTrapChargeDoping", trap_charge)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "InterfaceTrapChargeDoping",
        trap_charge,
        "Potential",
    )
    replace_node_model(config, "PotentialNodeCharge", potential_charge)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "PotentialNodeCharge",
        potential_charge,
        "Electrons",
        "Holes",
        "Potential",
    )
    replace_node_model(config, "InterfaceTrapRecombinationRate", recombination)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "InterfaceTrapRecombinationRate",
        recombination,
        "Electrons",
        "Holes",
    )


def transfer_gate_geometry(config: SplitPD2DConfig) -> dict[str, float]:
    geometry: dict[str, float] = {
        "x_min_um": -0.18,
        "x_max_um": 0.18,
        "oxide_thickness_um": 0.006,
        "sheet_thickness_um": config.fixed_charge_sheet_thickness_um,
    }
    if config.measured_profile:
        profile = load_measured_profile(config.measured_profile)
        tg = profile.geometry.get("transfer_gate", {})
        geometry["x_min_um"] = float(tg.get("x_min_um", geometry["x_min_um"]))
        geometry["x_max_um"] = float(tg.get("x_max_um", geometry["x_max_um"]))
        geometry["oxide_thickness_um"] = float(
            tg.get("oxide_thickness_um", geometry["oxide_thickness_um"])
        )
    return geometry


def create_transfer_gate_capacitive_models(config: SplitPD2DConfig) -> dict[str, float | str | bool]:
    if not config.transfer_gate_capacitive_coupling:
        replace_node_model(config, "TransferGateChargeDoping", "0")
        return {"enabled": False}
    geometry = transfer_gate_geometry(config)
    eps0_f_per_cm = 8.8541878128e-14
    eps_ox = 3.9
    oxide_thickness_cm = max(float(geometry["oxide_thickness_um"]) * 1.0e-4, 1.0e-9)
    sheet_thickness_cm = max(float(geometry["sheet_thickness_um"]) * 1.0e-4, 1.0e-9)
    cox_f_per_cm2 = eps0_f_per_cm * eps_ox / oxide_thickness_cm
    doping_per_volt = cox_f_per_cm2 / (Q_E * sheet_thickness_cm)
    x_min_cm = float(geometry["x_min_um"]) * 1.0e-4
    x_max_cm = float(geometry["x_max_um"]) * 1.0e-4
    y_max_cm = sheet_thickness_cm
    set_parameter(device=config.device, region=config.region, name="TransferGateBias", value=config.transfer_gate_bias_v)
    set_parameter(
        device=config.device,
        region=config.region,
        name="TransferGateDopingPerVolt",
        value=doping_per_volt,
    )
    set_parameter(
        device=config.device,
        region=config.region,
        name="TransferGateCouplingSign",
        value=config.transfer_gate_coupling_sign,
    )
    mask = (
        f"step(x - ({x_min_cm:.16e}))*step(({x_max_cm:.16e}) - x)"
        f"*step(({y_max_cm:.16e}) - y)"
    )
    charge = (
        "TransferGateCouplingSign*TransferGateDopingPerVolt"
        f"*({mask})*(TransferGateBias - Potential)"
    )
    replace_node_model(config, "TransferGateChargeDoping", charge)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "TransferGateChargeDoping",
        charge,
        "Potential",
    )
    potential_charge = (
        "-ElectronCharge*kahan3(Holes, -Electrons, "
        "NetDoping + InterfaceTrapChargeDoping + TransferGateChargeDoping)"
    )
    replace_node_model(config, "PotentialNodeCharge", potential_charge)
    CreateNodeModelDerivative(
        config.device,
        config.region,
        "PotentialNodeCharge",
        potential_charge,
        "Electrons",
        "Holes",
        "Potential",
    )
    return {
        "enabled": True,
        "model": "oxide_capacitance_sheet_charge_proxy_v1",
        "x_min_um": geometry["x_min_um"],
        "x_max_um": geometry["x_max_um"],
        "oxide_thickness_um": geometry["oxide_thickness_um"],
        "sheet_thickness_um": geometry["sheet_thickness_um"],
        "cox_f_per_cm2": cox_f_per_cm2,
        "doping_per_volt_cm3": doping_per_volt,
        "coupling_sign": config.transfer_gate_coupling_sign,
    }


def create_profile_doping(config: SplitPD2DConfig) -> dict:
    donors, acceptors, fixed_charge, feature_summary = profile_electrical_terms(config)
    node_solution(device=config.device, region=config.region, name="Donors")
    node_solution(device=config.device, region=config.region, name="Acceptors")
    node_solution(device=config.device, region=config.region, name="FixedChargeDoping")
    set_node_values(device=config.device, region=config.region, name="Donors", values=donors.tolist())
    set_node_values(
        device=config.device,
        region=config.region,
        name="Acceptors",
        values=acceptors.tolist(),
    )
    set_node_values(
        device=config.device,
        region=config.region,
        name="FixedChargeDoping",
        values=fixed_charge.tolist(),
    )
    CreateNodeModel(
        config.device,
        config.region,
        "NetDoping",
        "Donors - Acceptors + FixedChargeDoping",
    )
    net_doping = donors - acceptors + fixed_charge
    return {
        "source": config.measured_profile,
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


def create_doping(config: SplitPD2DConfig) -> dict:
    if config.electrical_model == "simple-pn":
        acceptors = "AcceptorsP*step(JunctionY - y)"
        donors = "DonorsN*step(y - JunctionY)"
        fixed_charge = "0"
    elif config.electrical_model == "proxy-pinned-split-pd":
        gap_mask = "step(x + GapHalfWidth)*step(GapHalfWidth - x)"
        dti_mask = (
            "step(x - XMin)*step(XMin + DtiWidth - x)"
            " + step(x - (XMax - DtiWidth))*step(XMax - x)"
        )
        collection_depth_mask = "step(y - PinningDepth)*step(DepthY - y)"
        acceptors = (
            "SubstrateAcceptors"
            " + PinningAcceptors*step(PinningDepth - y)"
            f" + IsolationAcceptors*({gap_mask})*{collection_depth_mask}"
            f" + DtiAcceptors*({dti_mask})"
        )
        donors = (
            f"CollectionDonors*(1 - ({gap_mask}))*{collection_depth_mask}"
        )
        fixed_charge = "0"
    elif config.electrical_model == "profile-ppd":
        return create_profile_doping(config)
    else:
        raise ValueError(f"Unsupported electrical_model: {config.electrical_model}")
    CreateNodeModel(config.device, config.region, "Acceptors", acceptors)
    CreateNodeModel(config.device, config.region, "Donors", donors)
    CreateNodeModel(config.device, config.region, "FixedChargeDoping", fixed_charge)
    CreateNodeModel(
        config.device,
        config.region,
        "NetDoping",
        "Donors - Acceptors + FixedChargeDoping",
    )
    return {
        "source": "analytic_proxy",
        "feature_summary": {
            "applied_features": [],
            "metadata_only_features": [],
        },
    }


def transport_settings(config: SplitPD2DConfig) -> dict:
    settings = {
        "source": "cli_defaults",
        "model": "constant_reference_v1",
        "measured": False,
        "calibrated": False,
        "mu_n_model": "constant_reference",
        "mu_p_model": "constant_reference",
        "mu_n_cm2_v_s": config.mu_n_cm2_v_s,
        "mu_p_cm2_v_s": config.mu_p_cm2_v_s,
        "mu_n_min_cm2_v_s": config.mu_n_cm2_v_s,
        "mu_n_max_cm2_v_s": config.mu_n_cm2_v_s,
        "mu_n_ref_doping_cm3": 1.0e17,
        "mu_n_alpha": 1.0,
        "mu_p_min_cm2_v_s": config.mu_p_cm2_v_s,
        "mu_p_max_cm2_v_s": config.mu_p_cm2_v_s,
        "mu_p_ref_doping_cm3": 1.0e17,
        "mu_p_alpha": 1.0,
        "tau_n_s": config.tau_n_s,
        "tau_p_s": config.tau_p_s,
        "tau_n_min_s": config.tau_n_s,
        "tau_n_max_s": config.tau_n_s,
        "tau_p_min_s": config.tau_p_s,
        "tau_p_max_s": config.tau_p_s,
        "lifetime_ref_doping_cm3": 1.0e17,
        "lifetime_alpha": 1.0,
        "field_mobility_model": "none",
        "field_mobility_enabled": False,
        "electron_saturation_velocity_cm_s": 1.0e7,
        "hole_saturation_velocity_cm_s": 8.0e6,
        "field_mobility_beta_n": 2.0,
        "field_mobility_beta_p": 1.0,
        "field_mobility_floor_v_per_cm": 1.0,
    }
    use_profile_transport = (
        config.measured_profile
        and config.electrical_model == "profile-ppd"
        and config.transport_override != "constant-reference"
    )
    if use_profile_transport:
        profile = load_measured_profile(config.measured_profile)
        transport = profile.data.get("mobility_recombination", {})
        settings["source"] = config.measured_profile
        for key, value in transport.items():
            settings[key] = value
        settings["measured"] = bool(
            transport.get("measured", transport.get("transport_model_measured", False))
        )
        settings["calibrated"] = bool(
            transport.get("calibrated", transport.get("transport_model_calibrated", False))
        )
    if config.transport_override == "constant-reference":
        settings["source"] = "cli_defaults_transport_override"
        settings["model"] = "constant_reference_v1"
        settings["transport_model"] = "constant_reference_v1"
        settings["mu_n_model"] = "constant_reference"
        settings["mu_p_model"] = "constant_reference"
        settings["recombination_model"] = "constant_reference_lifetime"
        settings["field_mobility_model"] = "none"
        settings["field_mobility_enabled"] = False
        settings["measured"] = False
        settings["calibrated"] = False
    if config.disable_field_mobility:
        settings["field_mobility_model"] = "disabled_by_runtime_gate"
        settings["field_mobility_enabled"] = False
    settings["model"] = str(settings.get("transport_model", settings.get("model", "")))
    if not settings["model"]:
        label = f"{settings.get('mu_n_model', '')} {settings.get('mu_p_model', '')}".lower()
        settings["model"] = (
            "caughey_thomas_doping_dependent_reference_v1"
            if "caughey" in label or "doping" in label
            else "constant_reference_v1"
        )
    for key in ("mu_n_cm2_v_s", "mu_n_min_cm2_v_s", "mu_n_max_cm2_v_s"):
        if key in settings:
            settings[key] = float(settings[key]) * config.electron_mobility_scale
    for key in ("mu_p_cm2_v_s", "mu_p_min_cm2_v_s", "mu_p_max_cm2_v_s"):
        if key in settings:
            settings[key] = float(settings[key]) * config.hole_mobility_scale
    for key in ("tau_n_s", "tau_p_s", "tau_n_min_s", "tau_n_max_s", "tau_p_min_s", "tau_p_max_s"):
        if key in settings:
            settings[key] = float(settings[key]) * config.lifetime_scale
    settings["runtime_calibration_scales"] = {
        "electron_mobility_scale": config.electron_mobility_scale,
        "hole_mobility_scale": config.hole_mobility_scale,
        "lifetime_scale": config.lifetime_scale,
        "transport_override": config.transport_override,
        "disable_field_mobility": config.disable_field_mobility,
    }
    return settings


def set_transport_parameter(config: SplitPD2DConfig, name: str, value: float) -> None:
    set_parameter(device=config.device, region=config.region, name=name, value=float(value))


def node_model_min_max(config: SplitPD2DConfig, name: str) -> tuple[float, float]:
    values = np.asarray(get_node_model_values(device=config.device, region=config.region, name=name))
    return float(np.min(values)), float(np.max(values))


def edge_model_min_max(config: SplitPD2DConfig, name: str) -> tuple[float, float]:
    values = np.asarray(get_edge_model_values(device=config.device, region=config.region, name=name))
    return float(np.min(values)), float(np.max(values))


def create_transport_models(config: SplitPD2DConfig) -> dict:
    settings = transport_settings(config)
    model = str(settings["model"]).lower()
    use_doping_dependent = "constant" not in model

    replace_node_model(
        config,
        "TotalDopingForMobility",
        "abs(Donors) + abs(Acceptors) + 1.0",
    )
    if use_doping_dependent:
        for name, key in (
            ("MuNMin", "mu_n_min_cm2_v_s"),
            ("MuNMax", "mu_n_max_cm2_v_s"),
            ("MuNRefDoping", "mu_n_ref_doping_cm3"),
            ("MuNAlpha", "mu_n_alpha"),
            ("MuPMin", "mu_p_min_cm2_v_s"),
            ("MuPMax", "mu_p_max_cm2_v_s"),
            ("MuPRefDoping", "mu_p_ref_doping_cm3"),
            ("MuPAlpha", "mu_p_alpha"),
        ):
            set_transport_parameter(config, name, float(settings[key]))
        replace_node_model(
            config,
            "ElectronMobility",
            "MuNMin + (MuNMax - MuNMin)/(1 + (TotalDopingForMobility/MuNRefDoping)^MuNAlpha)",
        )
        replace_node_model(
            config,
            "HoleMobility",
            "MuPMin + (MuPMax - MuPMin)/(1 + (TotalDopingForMobility/MuPRefDoping)^MuPAlpha)",
        )
    else:
        replace_node_model(config, "ElectronMobility", "mu_n")
        replace_node_model(config, "HoleMobility", "mu_p")

    lifetime_model = str(settings.get("recombination_model", "")).lower()
    use_doping_lifetime = use_doping_dependent and "doping" in lifetime_model
    if use_doping_lifetime:
        for name, key in (
            ("TauNMin", "tau_n_min_s"),
            ("TauNMax", "tau_n_max_s"),
            ("TauPMin", "tau_p_min_s"),
            ("TauPMax", "tau_p_max_s"),
            ("LifetimeRefDoping", "lifetime_ref_doping_cm3"),
            ("LifetimeAlpha", "lifetime_alpha"),
        ):
            set_transport_parameter(config, name, float(settings[key]))
        replace_node_model(
            config,
            "SRHTauN",
            "TauNMin + (TauNMax - TauNMin)/(1 + (TotalDopingForMobility/LifetimeRefDoping)^LifetimeAlpha)",
        )
        replace_node_model(
            config,
            "SRHTauP",
            "TauPMin + (TauPMax - TauPMin)/(1 + (TotalDopingForMobility/LifetimeRefDoping)^LifetimeAlpha)",
        )
    else:
        replace_node_model(config, "SRHTauN", "taun")
        replace_node_model(config, "SRHTauP", "taup")

    edge_average_model(
        device=config.device,
        region=config.region,
        node_model="ElectronMobility",
        edge_model="ElectronMobilityLowField_Edge",
    )
    edge_average_model(
        device=config.device,
        region=config.region,
        node_model="HoleMobility",
        edge_model="HoleMobilityLowField_Edge",
    )

    total_doping_min, total_doping_max = node_model_min_max(config, "TotalDopingForMobility")
    electron_mu_min, electron_mu_max = node_model_min_max(config, "ElectronMobility")
    hole_mu_min, hole_mu_max = node_model_min_max(config, "HoleMobility")
    tau_n_min, tau_n_max = node_model_min_max(config, "SRHTauN")
    tau_p_min, tau_p_max = node_model_min_max(config, "SRHTauP")
    return {
        "model": settings["model"],
        "source": settings["source"],
        "measured": bool(settings.get("measured", False)),
        "calibrated": bool(settings.get("calibrated", False)),
        "electron_mobility_node_model": "ElectronMobility",
        "electron_mobility_low_field_edge_model": "ElectronMobilityLowField_Edge",
        "electron_mobility_edge_model": "ElectronMobility_Edge",
        "hole_mobility_node_model": "HoleMobility",
        "hole_mobility_low_field_edge_model": "HoleMobilityLowField_Edge",
        "hole_mobility_edge_model": "HoleMobility_Edge",
        "srh_tau_n_node_model": "SRHTauN",
        "srh_tau_p_node_model": "SRHTauP",
        "field_mobility_model": settings.get("field_mobility_model", "none"),
        "field_mobility_enabled": bool(settings.get("field_mobility_enabled", False)),
        "runtime_calibration_scales": settings.get("runtime_calibration_scales", {}),
        "total_doping_min_cm3": total_doping_min,
        "total_doping_max_cm3": total_doping_max,
        "electron_mobility_min_cm2_v_s": electron_mu_min,
        "electron_mobility_max_cm2_v_s": electron_mu_max,
        "hole_mobility_min_cm2_v_s": hole_mu_min,
        "hole_mobility_max_cm2_v_s": hole_mu_max,
        "tau_n_min_s": tau_n_min,
        "tau_n_max_s": tau_n_max,
        "tau_p_min_s": tau_p_min,
        "tau_p_max_s": tau_p_max,
        "parameters": {
            key: settings.get(key)
            for key in (
                "mu_n_model",
                "mu_p_model",
                "mu_n_min_cm2_v_s",
                "mu_n_max_cm2_v_s",
                "mu_n_ref_doping_cm3",
                "mu_n_alpha",
                "mu_p_min_cm2_v_s",
                "mu_p_max_cm2_v_s",
                "mu_p_ref_doping_cm3",
                "mu_p_alpha",
                "recombination_model",
                "tau_n_min_s",
                "tau_n_max_s",
                "tau_p_min_s",
                "tau_p_max_s",
                "lifetime_ref_doping_cm3",
                "lifetime_alpha",
                "field_mobility_model",
                "field_mobility_enabled",
                "electron_saturation_velocity_cm_s",
                "hole_saturation_velocity_cm_s",
                "field_mobility_beta_n",
                "field_mobility_beta_p",
                "field_mobility_floor_v_per_cm",
            )
            if key in settings
        },
    }


def refresh_transport_summary(config: SplitPD2DConfig, summary: dict) -> None:
    for model, low_key, high_key in (
        ("TotalDopingForMobility", "total_doping_min_cm3", "total_doping_max_cm3"),
        ("ElectronMobility", "electron_mobility_min_cm2_v_s", "electron_mobility_max_cm2_v_s"),
        ("HoleMobility", "hole_mobility_min_cm2_v_s", "hole_mobility_max_cm2_v_s"),
        ("SRHTauN", "tau_n_min_s", "tau_n_max_s"),
        ("SRHTauP", "tau_p_min_s", "tau_p_max_s"),
    ):
        low, high = node_model_min_max(config, model)
        summary[low_key] = low
        summary[high_key] = high
    for model, low_key, high_key in (
        (
            "ElectronMobilityLowField_Edge",
            "electron_mobility_low_field_edge_min_cm2_v_s",
            "electron_mobility_low_field_edge_max_cm2_v_s",
        ),
        (
            "HoleMobilityLowField_Edge",
            "hole_mobility_low_field_edge_min_cm2_v_s",
            "hole_mobility_low_field_edge_max_cm2_v_s",
        ),
        (
            "ElectronMobility_Edge",
            "electron_mobility_effective_edge_min_cm2_v_s",
            "electron_mobility_effective_edge_max_cm2_v_s",
        ),
        (
            "HoleMobility_Edge",
            "hole_mobility_effective_edge_min_cm2_v_s",
            "hole_mobility_effective_edge_max_cm2_v_s",
        ),
    ):
        low, high = edge_model_min_max(config, model)
        summary[low_key] = low
        summary[high_key] = high


def floating_diffusion_circuit_node(config: SplitPD2DConfig) -> str:
    return simple_physics.GetContactBiasName("floating_diffusion")


def setup_floating_diffusion_circuit(config: SplitPD2DConfig) -> None:
    if not config.floating_diffusion_circuit:
        return
    contact_names = set(get_contact_list(device=config.device))
    if "floating_diffusion" not in contact_names:
        return
    node = floating_diffusion_circuit_node(config)
    if node in set(get_circuit_node_list()):
        return
    add_circuit_node(name=node, variable_update="log_damp")
    circuit_element(
        name="RFDRESET",
        n1=node,
        n2=0,
        value=config.floating_diffusion_reset_on_resistance_ohm_cm,
    )
    circuit_element(
        name="CFD",
        n1=node,
        n2=0,
        value=config.floating_diffusion_capacitance_f_per_cm,
    )


def set_floating_diffusion_reset(config: SplitPD2DConfig, enabled: bool) -> None:
    if not config.floating_diffusion_circuit:
        return
    resistance = (
        config.floating_diffusion_reset_on_resistance_ohm_cm
        if enabled
        else config.floating_diffusion_reset_off_resistance_ohm_cm
    )
    circuit_alter(name="RFDRESET", value=resistance)


def create_transport_current_edges(config: SplitPD2DConfig, summary: dict) -> None:
    settings = transport_settings(config)
    field_enabled = bool(settings.get("field_mobility_enabled", False))
    summary["field_mobility_enabled"] = field_enabled
    summary["field_mobility_model"] = settings.get("field_mobility_model", "none")
    if field_enabled:
        for name, key in (
            ("ElectronSaturationVelocity", "electron_saturation_velocity_cm_s"),
            ("HoleSaturationVelocity", "hole_saturation_velocity_cm_s"),
            ("FieldMobilityBetaN", "field_mobility_beta_n"),
            ("FieldMobilityBetaP", "field_mobility_beta_p"),
            ("FieldMobilityFloor", "field_mobility_floor_v_per_cm"),
        ):
            set_transport_parameter(config, name, float(settings[key]))
        electric_field_abs = "(ElectricField*ElectricField + FieldMobilityFloor^2)^0.5"
        electron_edge = (
            "ElectronMobilityLowField_Edge"
            f"/(1 + (ElectronMobilityLowField_Edge*({electric_field_abs})"
            "/ElectronSaturationVelocity)^FieldMobilityBetaN)^(1/FieldMobilityBetaN)"
        )
        hole_edge = (
            "HoleMobilityLowField_Edge"
            f"/(1 + (HoleMobilityLowField_Edge*({electric_field_abs})"
            "/HoleSaturationVelocity)^FieldMobilityBetaP)^(1/FieldMobilityBetaP)"
        )
        CreateEdgeModel(config.device, config.region, "ElectronMobility_Edge", electron_edge)
        CreateEdgeModelDerivatives(
            config.device,
            config.region,
            "ElectronMobility_Edge",
            electron_edge,
            "Potential",
        )
        CreateEdgeModel(config.device, config.region, "HoleMobility_Edge", hole_edge)
        CreateEdgeModelDerivatives(
            config.device,
            config.region,
            "HoleMobility_Edge",
            hole_edge,
            "Potential",
        )
    else:
        edge_average_model(
            device=config.device,
            region=config.region,
            node_model="ElectronMobility",
            edge_model="ElectronMobility_Edge",
        )
        edge_average_model(
            device=config.device,
            region=config.region,
            node_model="HoleMobility",
            edge_model="HoleMobility_Edge",
        )
    refresh_transport_summary(config, summary)


def set_contact_bias(config: SplitPD2DConfig, anode_bias_v: float) -> None:
    for contact in get_contact_list(device=config.device):
        if contact == "floating_diffusion" and config.floating_diffusion_circuit:
            continue
        if contact == "anode":
            bias = anode_bias_v
        elif contact == "floating_diffusion":
            bias = config.floating_diffusion_bias_v
        elif contact == "transfer_gate":
            bias = config.transfer_gate_bias_v
        else:
            bias = 0.0
        set_parameter(
            device=config.device,
            name=simple_physics.GetContactBiasName(contact),
            value=bias,
        )


def set_transfer_gate_bias(config: SplitPD2DConfig, bias_v: float) -> None:
    set_parameter(
        device=config.device,
        region=config.region,
        name="TransferGateBias",
        value=float(bias_v),
    )
    if "transfer_gate" not in set(get_contact_list(device=config.device)):
        return
    set_parameter(
        device=config.device,
        name=simple_physics.GetContactBiasName("transfer_gate"),
        value=bias_v,
    )


def device_regions(config: SplitPD2DConfig) -> set[str]:
    return set(get_region_list(device=config.device))


def resolved_oxide_enabled(config: SplitPD2DConfig) -> bool:
    return config.oxide_region in device_regions(config)


def region_for_contact(config: SplitPD2DConfig, contact: str) -> str:
    regions = get_region_list(device=config.device, contact=contact)
    if not regions:
        return config.region
    return str(regions[0])


def create_gate_potential_contact(config: SplitPD2DConfig, contact: str) -> None:
    if not InEdgeModelList(config.device, config.region, "contactcharge_edge"):
        CreateEdgeModel(config.device, config.region, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(
            config.device,
            config.region,
            "contactcharge_edge",
            "Permittivity*ElectricField",
            "Potential",
        )
    bias_name = simple_physics.GetContactBiasName(contact)
    model_name = f"{contact}gatepotential"
    CreateContactNodeModel(config.device, contact, model_name, f"Potential - {bias_name}")
    CreateContactNodeModel(config.device, contact, f"{model_name}:Potential", "1")
    contact_equation(
        device=config.device,
        contact=contact,
        name="PotentialEquation",
        node_model=model_name,
        edge_charge_model="contactcharge_edge",
    )


def solve_potential_only(config: SplitPD2DConfig) -> None:
    CreateSolution(config.device, config.region, "Potential")
    simple_physics.CreateSiliconPotentialOnly(config.device, config.region)
    if resolved_oxide_enabled(config):
        simple_physics.SetOxideParameters(config.device, config.oxide_region, config.temperature_k)
        CreateSolution(config.device, config.oxide_region, "Potential")
        simple_physics.CreateOxidePotentialOnly(config.device, config.oxide_region, "log_damp")
    setup_floating_diffusion_circuit(config)
    for contact in get_contact_list(device=config.device):
        contact_region = region_for_contact(config, contact)
        is_circuit = contact == "floating_diffusion" and config.floating_diffusion_circuit
        if not is_circuit:
            set_parameter(
                device=config.device,
                name=simple_physics.GetContactBiasName(contact),
                value=0.0,
            )
        if contact_region == config.oxide_region:
            bias = config.transfer_gate_bias_v if contact == "transfer_gate" else 0.0
            set_parameter(
                device=config.device,
                name=simple_physics.GetContactBiasName(contact),
                value=bias,
            )
            simple_physics.CreateOxideContact(config.device, config.oxide_region, contact)
            continue
        if contact == "transfer_gate":
            set_parameter(
                device=config.device,
                name=simple_physics.GetContactBiasName(contact),
                value=config.transfer_gate_bias_v,
            )
            create_gate_potential_contact(config, contact)
            continue
        simple_physics.CreateSiliconPotentialOnlyContact(
            config.device, config.region, contact, is_circuit=is_circuit
        )
    for interface in get_interface_list(device=config.device):
        if str(interface) == "silicon_oxide":
            simple_physics.CreateSiliconOxideInterface(config.device, interface)
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=80)


def replace_node_model(config: SplitPD2DConfig, model: str, expression: str) -> None:
    existing = set(get_node_model_list(device=config.device, region=config.region))
    for candidate in (f"{model}:Electrons", f"{model}:Holes", f"{model}:Potential", model):
        if candidate in existing:
            delete_node_model(device=config.device, region=config.region, name=candidate)
    CreateNodeModel(config.device, config.region, model, expression)


def selected_generation_profile_rows(config: SplitPD2DConfig) -> list[dict]:
    path = Path(config.generation_profile_csv)
    if not path.exists():
        raise FileNotFoundError(f"generation profile not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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
        raise RuntimeError("No generation profile rows match requested filters")
    rows.sort(key=lambda row: float(row["depth_um_from_si_top"]))
    return rows


def selected_generation_map(config: SplitPD2DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    path = Path(config.generation_map_npz)
    if not path.exists():
        raise FileNotFoundError(f"generation map not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        cases = np.asarray(data["case"]).astype(str)
        wavelengths = np.asarray(data["wavelength_nm"], dtype=float)
        candidates = np.ones(cases.shape[0], dtype=bool)
        if config.generation_profile_case:
            candidates &= cases == config.generation_profile_case
        if config.generation_profile_wavelength_nm > 0:
            candidates &= np.isclose(
                wavelengths,
                config.generation_profile_wavelength_nm,
                rtol=0.0,
                atol=1.0e-9,
            )
        indices = np.flatnonzero(candidates)
        if indices.size == 0:
            raise RuntimeError("No generation map entry matches requested filters")
        index = int(indices[0])
        x_um = np.asarray(data["x_um"], dtype=float)
        depth_um = np.asarray(data["depth_um_from_si_top"], dtype=float)
        generation = np.asarray(data["generation_cm3_s"][index], dtype=float)
    return x_um, depth_um, generation, index


def bilinear_rectilinear(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    x_order = np.argsort(x_axis)
    y_order = np.argsort(y_axis)
    x_axis = x_axis[x_order]
    y_axis = y_axis[y_order]
    values = values[np.ix_(x_order, y_order)]

    result = np.zeros_like(x_query, dtype=float)
    inside = (
        (x_query >= x_axis[0])
        & (x_query <= x_axis[-1])
        & (y_query >= y_axis[0])
        & (y_query <= y_axis[-1])
    )
    if not np.any(inside):
        return result

    x = x_query[inside]
    y = y_query[inside]
    ix1 = np.searchsorted(x_axis, x, side="right")
    iy1 = np.searchsorted(y_axis, y, side="right")
    ix1 = np.clip(ix1, 1, len(x_axis) - 1)
    iy1 = np.clip(iy1, 1, len(y_axis) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    y0 = y_axis[iy0]
    y1 = y_axis[iy1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))

    v00 = values[ix0, iy0]
    v10 = values[ix1, iy0]
    v01 = values[ix0, iy1]
    v11 = values[ix1, iy1]
    result[inside] = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return result


def coordinate_key(x_cm: float, y_cm: float, digits: int = 14) -> tuple[float, float]:
    return (round(float(x_cm), digits), round(float(y_cm), digits))


def triangle_area_cm2(points: list[tuple[float, float]]) -> float:
    (x0, y0), (x1, y1), (x2, y2) = points
    return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5


def parse_gmsh_triangles(path_text: str) -> tuple[dict[int, tuple[float, float]], list[tuple[int, int, int]]]:
    path = Path(path_text)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    node_start = lines.index("$Nodes") + 1
    node_count = int(lines[node_start])
    nodes: dict[int, tuple[float, float]] = {}
    for line in lines[node_start + 1 : node_start + 1 + node_count]:
        parts = line.split()
        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
    elem_start = lines.index("$Elements") + 1
    elem_count = int(lines[elem_start])
    triangles: list[tuple[int, int, int]] = []
    for line in lines[elem_start + 1 : elem_start + 1 + elem_count]:
        parts = line.split()
        elem_type = int(parts[1])
        tag_count = int(parts[2])
        node_ids = [int(value) for value in parts[3 + tag_count :]]
        if elem_type == 2 and len(node_ids) == 3:
            triangles.append((node_ids[0], node_ids[1], node_ids[2]))
    return nodes, triangles


def rectilinear_generation_map_current_a_per_cm(config: SplitPD2DConfig) -> float:
    x_axis, depth_axis, generation, _ = selected_generation_map(config)
    integral_cm2 = float(
        np.trapezoid(np.trapezoid(generation, depth_axis, axis=1), x_axis) * 1.0e-8
    )
    return Q_E * integral_cm2 * config.generation_map_scale


def mesh_generation_current_a_per_cm_from_values(
    config: SplitPD2DConfig,
    values: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
) -> float | None:
    if config.mesh_source != "gmsh" or not config.gmsh_mesh:
        return None
    nodes, triangles = parse_gmsh_triangles(config.gmsh_mesh)
    by_xy = {
        coordinate_key(float(x), float(y)): float(value)
        for x, y, value in zip(x_cm, y_cm, values)
    }
    integral = 0.0
    for triangle in triangles:
        area = triangle_area_cm2([nodes[node_id] for node_id in triangle])
        triangle_values = [
            by_xy.get(coordinate_key(*nodes[node_id]), 0.0)
            for node_id in triangle
        ]
        integral += float(np.mean(triangle_values)) * area
    return Q_E * integral


def generation_map_normalization_factor(
    config: SplitPD2DConfig,
    values: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
) -> float:
    if not config.normalize_generation_map_integral:
        return config.generation_map_scale
    target_current = rectilinear_generation_map_current_a_per_cm(config)
    mesh_current = mesh_generation_current_a_per_cm_from_values(config, values, x_cm, y_cm)
    if mesh_current is None or not math.isfinite(mesh_current) or mesh_current <= 0.0:
        return config.generation_map_scale
    raw_target_current = target_current / config.generation_map_scale if config.generation_map_scale else 0.0
    return raw_target_current / mesh_current * config.generation_map_scale


def imported_generation_map_values(config: SplitPD2DConfig) -> list[float]:
    x_axis, depth_axis, generation, _ = selected_generation_map(config)
    x_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"))
    y_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"))
    x_um = x_cm * 1.0e4
    depth_um = y_cm * 1.0e4
    values = bilinear_rectilinear(x_axis, depth_axis, generation, x_um, depth_um)
    scale = generation_map_normalization_factor(config, values, x_cm, y_cm)
    return (values * scale).tolist()


def generation_map_integral_summary(config: SplitPD2DConfig, profiles: list[dict]) -> dict:
    if not config.generation_map_npz:
        return {}
    x_cm = np.asarray([row["x_cm"] for row in profiles], dtype=float)
    y_cm = np.asarray([row["y_cm"] for row in profiles], dtype=float)
    values = np.asarray([row["OpticalGenerationRate"] for row in profiles], dtype=float)
    target_current = rectilinear_generation_map_current_a_per_cm(config)
    mesh_current = mesh_generation_current_a_per_cm_from_values(config, values, x_cm, y_cm)
    rel_error = (
        abs(mesh_current - target_current) / abs(target_current)
        if mesh_current is not None and target_current
        else math.nan
    )
    return {
        "method": "rectilinear_map_integral_preserving_node_scale_v1"
        if config.normalize_generation_map_integral
        else "raw_node_interpolation_no_integral_normalization",
        "enabled": bool(config.normalize_generation_map_integral),
        "target_rectilinear_current_a_per_cm": target_current,
        "mesh_integrated_current_a_per_cm": mesh_current,
        "mesh_to_target_rel_error": rel_error,
    }


def imported_generation_values(config: SplitPD2DConfig) -> list[float]:
    rows = selected_generation_profile_rows(config)
    depth_um = np.asarray([float(row["depth_um_from_si_top"]) for row in rows])
    generation = np.asarray([float(row["generation_cm3_s"]) for row in rows])
    x_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"))
    y_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"))
    y_um = y_cm * 1.0e4
    base = np.interp(y_um, depth_um, generation, left=0.0, right=0.0)

    if config.generation_lateral_mode == "uniform":
        lateral = np.ones_like(base)
    elif config.generation_lateral_mode == "gaussian":
        sigma = max(config.photo_sigma_x_cm, 1e-30)
        lateral = np.exp(-((x_cm - config.photo_shift_x_cm) ** 2) / (2.0 * sigma * sigma))
    else:
        raise ValueError(f"Unsupported generation_lateral_mode: {config.generation_lateral_mode}")
    return (base * lateral * config.generation_profile_scale).tolist()


def generation_probe_values(config: SplitPD2DConfig) -> np.ndarray:
    node_count = len(get_node_model_values(device=config.device, region=config.region, name="x"))
    if config.generation_probe_g0_cm3_s == 0.0:
        return np.zeros(node_count, dtype=float)
    x_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"))
    y_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"))
    sigma_x = max(config.generation_probe_sigma_x_cm, 1.0e-30)
    sigma_y = max(config.generation_probe_sigma_y_cm, 1.0e-30)
    return config.generation_probe_g0_cm3_s * np.exp(
        -((x_cm - config.generation_probe_x_cm) ** 2) / (2.0 * sigma_x * sigma_x)
        - ((y_cm - config.generation_probe_depth_cm) ** 2) / (2.0 * sigma_y * sigma_y)
    )


def set_optical_generation(
    config: SplitPD2DConfig,
    illuminated: bool,
    ramp_scale: float = 1.0,
) -> None:
    node_count = len(get_node_model_values(device=config.device, region=config.region, name="x"))
    if config.generation_map_npz:
        values = imported_generation_map_values(config) if illuminated else [0.0] * node_count
        if illuminated:
            values = (
                np.asarray(values, dtype=float) * ramp_scale
                + generation_probe_values(config) * ramp_scale
            ).tolist()
        set_node_values(
            device=config.device,
            region=config.region,
            name="OpticalGenerationRate",
            values=values,
        )
    elif config.generation_profile_csv:
        values = imported_generation_values(config) if illuminated else [0.0] * node_count
        if illuminated:
            values = (
                np.asarray(values, dtype=float) * ramp_scale
                + generation_probe_values(config) * ramp_scale
            ).tolist()
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
            value=config.photo_g0_cm3_s * ramp_scale if illuminated else 0.0,
        )


def create_photo_generation_models(config: SplitPD2DConfig) -> None:
    usrh = "(Electrons*Holes - n_i^2)/(SRHTauP*(Electrons + n1) + SRHTauN*(Holes + p1))"
    interface_trap_rate = "InterfaceTrapRecombinationRate"
    electron_generation = (
        f"-ElectronCharge*({usrh}) - ElectronCharge*{interface_trap_rate} "
        "+ ElectronCharge*OpticalGenerationRate"
    )
    hole_generation = (
        f"+ElectronCharge*({usrh}) + ElectronCharge*{interface_trap_rate} "
        "- ElectronCharge*OpticalGenerationRate"
    )

    if config.generation_map_npz or config.generation_profile_csv:
        node_solution(device=config.device, region=config.region, name="OpticalGenerationRate")
        set_optical_generation(config, illuminated=False)
    else:
        CreateNodeModel(
            config.device,
            config.region,
            "OpticalGenerationRate",
            "PhotoG0*exp(-((x-PhotoShiftX)^2)/(2*PhotoSigmaX^2))*exp(-((y-JunctionY)^2)/(2*PhotoSigmaY^2))",
        )
    replace_node_model(config, "USRH", usrh)
    CreateNodeModelDerivative(config.device, config.region, "USRH", usrh, "Electrons", "Holes")
    replace_node_model(config, "ElectronGeneration", electron_generation)
    replace_node_model(config, "HoleGeneration", hole_generation)
    CreateNodeModelDerivative(
        config.device, config.region, "ElectronGeneration", electron_generation, "Electrons", "Holes"
    )
    CreateNodeModelDerivative(
        config.device, config.region, "HoleGeneration", hole_generation, "Electrons", "Holes"
    )


def solve_drift_diffusion(config: SplitPD2DConfig, transport_summary: dict) -> None:
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
    create_interface_trap_models(config)
    transport_summary["transfer_gate_capacitive_coupling"] = create_transfer_gate_capacitive_models(config)
    simple_physics.CreateBernoulli(config.device, config.region)
    create_transport_current_edges(config, transport_summary)
    simple_physics.CreateSRH(config.device, config.region)
    create_photo_generation_models(config)
    simple_physics.CreateECE(config.device, config.region, "ElectronMobility_Edge")
    simple_physics.CreateHCE(config.device, config.region, "HoleMobility_Edge")
    for contact in get_contact_list(device=config.device):
        if contact == "transfer_gate":
            continue
        is_circuit = contact == "floating_diffusion" and config.floating_diffusion_circuit
        simple_physics.CreateSiliconDriftDiffusionAtContact(
            config.device, config.region, contact, is_circuit=is_circuit
        )
    solve(
        type="dc",
        absolute_error=config.dd_absolute_error,
        relative_error=config.dd_relative_error,
        maximum_iterations=config.dd_max_iterations,
    )


def contact_currents(config: SplitPD2DConfig, contact: str) -> dict[str, float]:
    if contact == "transfer_gate":
        return {
            f"{contact}_electron_current_a_per_cm": 0.0,
            f"{contact}_hole_current_a_per_cm": 0.0,
            f"{contact}_total_current_a_per_cm": 0.0,
        }
    electron_current = get_contact_current(
        device=config.device, contact=contact, equation=simple_physics.ece_name
    )
    hole_current = get_contact_current(
        device=config.device, contact=contact, equation=simple_physics.hce_name
    )
    return {
        f"{contact}_electron_current_a_per_cm": electron_current,
        f"{contact}_hole_current_a_per_cm": hole_current,
        f"{contact}_total_current_a_per_cm": electron_current + hole_current,
    }


def generation_source_name(config: SplitPD2DConfig) -> str:
    if config.generation_map_npz:
        return "imported_2d_map"
    if config.generation_profile_csv:
        return "imported_1d_profile"
    return "analytic_2d_gaussian"


def solve_dc(config: SplitPD2DConfig) -> None:
    solve(
        type="dc",
        absolute_error=config.dd_absolute_error,
        relative_error=config.dd_relative_error,
        maximum_iterations=config.dd_max_iterations,
    )


def ramp_reverse_bias(config: SplitPD2DConfig, target_bias_v: float) -> None:
    if target_bias_v == 0.0:
        set_contact_bias(config, 0.0)
        return
    for fraction in (0.25, 0.5, 0.75, 1.0):
        set_contact_bias(config, target_bias_v * fraction)
        solve_dc(config)


def solve_condition(config: SplitPD2DConfig, condition: str, illuminated: bool) -> dict:
    if illuminated:
        set_contact_bias(config, config.reverse_bias_v)
        for ramp_scale in (0.25, 0.5, 1.0):
            set_optical_generation(config, illuminated=True, ramp_scale=ramp_scale)
            solve_dc(config)
    else:
        set_optical_generation(config, illuminated=False)
        ramp_reverse_bias(config, config.reverse_bias_v)
    row = {
        "condition": condition,
        "generation_source": generation_source_name(config),
        "generation_lateral_mode": config.generation_lateral_mode,
        "photo_shift_x_um": config.photo_shift_x_um,
        "electrical_model": config.electrical_model,
    }
    contacts = list(get_contact_list(device=config.device))
    for contact in contacts:
        if contact == "anode":
            row[f"{contact}_bias_v"] = config.reverse_bias_v
        elif contact == "floating_diffusion":
            row[f"{contact}_bias_v"] = config.floating_diffusion_bias_v
        elif contact == "transfer_gate":
            row[f"{contact}_bias_v"] = config.transfer_gate_bias_v
        else:
            row[f"{contact}_bias_v"] = 0.0
    for contact in contacts:
        row.update(contact_currents(config, contact))
    row["signal_carrier"] = "electron"
    row["cathode_left_signal_current_a_per_cm"] = row[
        "cathode_left_electron_current_a_per_cm"
    ]
    row["cathode_right_signal_current_a_per_cm"] = row[
        "cathode_right_electron_current_a_per_cm"
    ]
    signal_denom = abs(row["cathode_left_signal_current_a_per_cm"]) + abs(
        row["cathode_right_signal_current_a_per_cm"]
    )
    row["split_phase_x_signal"] = (
        (
            row["cathode_right_signal_current_a_per_cm"]
            - row["cathode_left_signal_current_a_per_cm"]
        )
        / signal_denom
        if signal_denom
        else 0.0
    )
    row["total_cathode_signal_current_a_per_cm"] = (
        row["cathode_left_signal_current_a_per_cm"]
        + row["cathode_right_signal_current_a_per_cm"]
    )
    left = row["cathode_left_total_current_a_per_cm"]
    right = row["cathode_right_total_current_a_per_cm"]
    denom = abs(left) + abs(right)
    row["split_phase_x_proxy"] = (right - left) / denom if denom else 0.0
    row["total_cathode_current_a_per_cm"] = left + right
    row["total_terminal_current_a_per_cm"] = sum(
        row[f"{contact}_total_current_a_per_cm"]
        for contact in contacts
    )
    row["terminal_current_balance_a_per_cm"] = (
        row["total_terminal_current_a_per_cm"]
    )
    return row


def profile_rows(config: SplitPD2DConfig) -> list[dict]:
    names = [
        "x",
        "y",
        "Potential",
        "Electrons",
        "Holes",
        "NetDoping",
        "FixedChargeDoping",
        "OpticalGenerationRate",
    ]
    optional_names = [
        "TotalDopingForMobility",
        "ElectronMobility",
        "HoleMobility",
        "SRHTauN",
        "SRHTauP",
        "InterfaceTrapDensityVolume",
        "InterfaceTrapOccupancy",
        "InterfaceTrapChargeDoping",
        "InterfaceTrapRecombinationCoeff",
        "InterfaceTrapRecombinationRate",
        "TransferGateChargeDoping",
    ]
    available_models = set(get_node_model_list(device=config.device, region=config.region))
    names.extend(name for name in optional_names if name in available_models)
    values = {
        name: get_node_model_values(device=config.device, region=config.region, name=name)
        for name in names
    }
    edge_average_model(device=config.device, region=config.region, node_model="x", edge_model="xmid")
    edge_average_model(device=config.device, region=config.region, node_model="y", edge_model="ymid")
    electric_field = np.asarray(
        get_edge_model_values(device=config.device, region=config.region, name="ElectricField")
    )
    x = np.asarray(values["x"])
    y = np.asarray(values["y"])
    xmid = np.asarray(get_edge_model_values(device=config.device, region=config.region, name="xmid"))
    ymid = np.asarray(get_edge_model_values(device=config.device, region=config.region, name="ymid"))
    # Nearest-edge electric-field proxy for plotting only.
    edge_points = np.column_stack([xmid, ymid])
    node_points = np.column_stack([x, y])
    ef_on_nodes = []
    for point in node_points:
        idx = int(np.argmin(np.sum((edge_points - point) ** 2, axis=1)))
        ef_on_nodes.append(float(electric_field[idx]))

    rows = []
    for index in range(len(x)):
        row = {
            "x_cm": float(values["x"][index]),
            "x_um": float(values["x"][index] * 1e4),
            "y_cm": float(values["y"][index]),
            "y_um": float(values["y"][index] * 1e4),
            "Potential": float(values["Potential"][index]),
            "Electrons": float(values["Electrons"][index]),
            "Holes": float(values["Holes"][index]),
            "NetDoping": float(values["NetDoping"][index]),
            "FixedChargeDoping": float(values["FixedChargeDoping"][index]),
            "OpticalGenerationRate": float(values["OpticalGenerationRate"][index]),
            "ElectricField_proxy_v_per_cm": ef_on_nodes[index],
        }
        for name in optional_names:
            if name in values:
                row[name] = float(values[name][index])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir: Path, rows: list[dict], profiles: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [row["condition"] for row in rows]
    left = [row.get("cathode_left_signal_current_a_per_cm", row["cathode_left_total_current_a_per_cm"]) for row in rows]
    right = [row.get("cathode_right_signal_current_a_per_cm", row["cathode_right_total_current_a_per_cm"]) for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.bar(x - 0.18, left, width=0.36, label="left cathode")
    axis.bar(x + 0.18, right, width=0.36, label="right cathode")
    axis.set_xticks(x)
    axis.set_xticklabels(conditions)
    axis.set_ylabel("Electron signal current (A/cm)")
    axis.set_title("2D split-PD cathode signal currents")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.savefig(output_dir / "split_currents.png", dpi=180)
    plt.close(fig)

    xs = np.asarray([row["x_um"] for row in profiles])
    ys = np.asarray([row["y_um"] for row in profiles])
    generation = np.asarray([row["OpticalGenerationRate"] for row in profiles])
    potential = np.asarray([row["Potential"] for row in profiles])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sc0 = axes[0].scatter(xs, ys, c=np.log10(np.maximum(generation, 1.0)), s=12, cmap="inferno")
    axes[0].set_title("log10 OpticalGenerationRate")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("depth y (um)")
    axes[0].invert_yaxis()
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    sc1 = axes[1].scatter(xs, ys, c=potential, s=12, cmap="viridis")
    axes[1].set_title("Potential")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("depth y (um)")
    axes[1].invert_yaxis()
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "node_maps.png", dpi=180)
    plt.close(fig)


def run(config: SplitPD2DConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    create_or_import_mesh(config)
    set_parameters(config)
    doping_summary = create_doping(config)
    transport_summary = create_transport_models(config)
    interface_trap_summary = create_interface_trap_sources(config)
    attach_interface_trap_summary(doping_summary, interface_trap_summary)
    set_contact_bias(config, 0.0)
    solve_potential_only(config)
    solve_drift_diffusion(config, transport_summary)

    rows = [
        solve_condition(config, "dark", illuminated=False),
        solve_condition(config, "illuminated", illuminated=True),
    ]
    refresh_transport_summary(config, transport_summary)
    profiles = profile_rows(config)
    generation_integral_summary = generation_map_integral_summary(config, profiles)
    write_csv(output_dir / "split_currents.csv", rows)
    write_csv(output_dir / "node_profile_2d.csv", profiles)
    device_tecplot = output_dir / "split_pd_2d_device.dat"
    write_devices(file=str(device_tecplot), type="tecplot")
    save_plots(output_dir, rows, profiles)

    dark = rows[0]
    illum = rows[1]
    left_delta_electron = (
        illum["cathode_left_electron_current_a_per_cm"]
        - dark["cathode_left_electron_current_a_per_cm"]
    )
    right_delta_electron = (
        illum["cathode_right_electron_current_a_per_cm"]
        - dark["cathode_right_electron_current_a_per_cm"]
    )
    left_delta_hole = (
        illum["cathode_left_hole_current_a_per_cm"]
        - dark["cathode_left_hole_current_a_per_cm"]
    )
    right_delta_hole = (
        illum["cathode_right_hole_current_a_per_cm"]
        - dark["cathode_right_hole_current_a_per_cm"]
    )
    left_delta_total = (
        illum["cathode_left_total_current_a_per_cm"]
        - dark["cathode_left_total_current_a_per_cm"]
    )
    right_delta_total = (
        illum["cathode_right_total_current_a_per_cm"]
        - dark["cathode_right_total_current_a_per_cm"]
    )
    contact_photo_deltas: dict[str, dict[str, float]] = {}
    for contact in get_contact_list(device=config.device):
        electron_key = f"{contact}_electron_current_a_per_cm"
        hole_key = f"{contact}_hole_current_a_per_cm"
        total_key = f"{contact}_total_current_a_per_cm"
        contact_photo_deltas[contact] = {
            "electron_delta_a_per_cm": illum[electron_key] - dark[electron_key],
            "hole_delta_a_per_cm": illum[hole_key] - dark[hole_key],
            "total_delta_a_per_cm": illum[total_key] - dark[total_key],
        }
    left_delta = left_delta_electron
    right_delta = right_delta_electron
    denom = abs(left_delta) + abs(right_delta)
    photo_split_phase_x_proxy = (right_delta - left_delta) / denom if denom else 0.0
    total_denom = abs(left_delta_total) + abs(right_delta_total)
    photo_split_phase_x_total_current = (
        (right_delta_total - left_delta_total) / total_denom if total_denom else 0.0
    )
    summary = {
        "schema": "devsim_split_pd_2d_smoke_v1",
        "devsim_version": version("devsim"),
        "config": asdict(config),
        "generation_source": generation_source_name(config),
        "generation_integral_summary": generation_integral_summary,
        "electrical_model": config.electrical_model,
        "doping_summary": doping_summary,
        "transport_summary": transport_summary,
        "interface_trap_summary": interface_trap_summary,
        "mesh_source": config.mesh_source,
        "gmsh_mesh": config.gmsh_mesh,
        "dark": dark,
        "illuminated": illum,
        "photo_signal_carrier": "electron",
        "left_photo_delta_a_per_cm": left_delta,
        "right_photo_delta_a_per_cm": right_delta,
        "left_photo_delta_electron_a_per_cm": left_delta_electron,
        "right_photo_delta_electron_a_per_cm": right_delta_electron,
        "left_photo_delta_hole_a_per_cm": left_delta_hole,
        "right_photo_delta_hole_a_per_cm": right_delta_hole,
        "left_photo_delta_total_current_a_per_cm": left_delta_total,
        "right_photo_delta_total_current_a_per_cm": right_delta_total,
        "contact_photo_deltas": contact_photo_deltas,
        "photo_split_phase_x_proxy": photo_split_phase_x_proxy,
        "photo_split_phase_x_total_current": photo_split_phase_x_total_current,
        "terminal_current_balance_illuminated_a_per_cm": illum["terminal_current_balance_a_per_cm"],
        "node_count": len(profiles),
        "outputs": {
            "split_currents_csv": str(output_dir / "split_currents.csv"),
            "node_profile_2d_csv": str(output_dir / "node_profile_2d.csv"),
            "device_tecplot": str(device_tecplot),
            "split_currents_png": str(output_dir / "split_currents.png"),
            "node_maps_png": str(output_dir / "node_maps.png"),
        },
        "notes": [
            "This 2D model is a lateral-collection smoke test, not a calibrated image sensor TCAD deck.",
            "The device_tecplot output is written from DEVSIM and preserves the solver mesh connectivity for VTK/VTU conversion.",
            "When generation_source is imported_2d_map, optical generation is interpolated from Meep G(x,depth) without analytic lateral shaping.",
            "The proxy-pinned-split-pd electrical model adds analytic pinning, split collection columns, center isolation, and side-DTI doping proxies.",
            "The profile-ppd electrical model imports donor/acceptor and fixed-charge proxy terms from measured_tcad_profile_v1.",
            "Split-PD signal deltas use cathode electron-current deltas; cathode hole-current and total-current deltas remain diagnostic fields.",
            "Carrier current uses DEVSIM effective edge mobility models, including optional field-dependent velocity saturation; SRH uses node lifetime models from the profile transport block.",
            "Interface Dit is represented by a potential-dependent trap charge and SRH sheet recombination proxy; optional floating_diffusion and resolved oxide transfer_gate contacts can be imported from Gmsh, with full TG transient charge transfer handled by devsim_tg_fd_transient_2d.py --tg-drive-mode resolved_gate.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "devsim_split_pd_2d")
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=2.8)
    parser.add_argument("--junction-um", type=float, default=0.35)
    parser.add_argument("--mesh-x-um", type=float, default=0.10)
    parser.add_argument("--mesh-y-um", type=float, default=0.08)
    parser.add_argument("--junction-mesh-um", type=float, default=0.025)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--acceptor-cm3", type=float, default=2.0e17)
    parser.add_argument("--donor-cm3", type=float, default=5.0e15)
    parser.add_argument("--photo-g0-cm3-s", type=float, default=1.0e20)
    parser.add_argument("--photo-shift-x-um", type=float, default=0.0)
    parser.add_argument("--photo-sigma-x-um", type=float, default=0.35)
    parser.add_argument("--photo-sigma-y-um", type=float, default=0.25)
    parser.add_argument("--generation-profile-csv", type=Path, default=None)
    parser.add_argument("--generation-profile-scale", type=float, default=1.0)
    parser.add_argument("--generation-profile-case", default="")
    parser.add_argument("--generation-profile-wavelength-nm", type=float, default=0.0)
    parser.add_argument("--generation-lateral-mode", choices=("uniform", "gaussian"), default="uniform")
    parser.add_argument("--generation-map-npz", type=Path, default=None)
    parser.add_argument("--generation-map-scale", type=float, default=1.0)
    parser.add_argument("--disable-generation-map-normalization", action="store_true")
    parser.add_argument("--generation-probe-g0-cm3-s", type=float, default=0.0)
    parser.add_argument("--generation-probe-x-um", type=float, default=0.0)
    parser.add_argument("--generation-probe-depth-um", type=float, default=0.0)
    parser.add_argument("--generation-probe-sigma-x-um", type=float, default=0.07)
    parser.add_argument("--generation-probe-sigma-y-um", type=float, default=0.10)
    parser.add_argument(
        "--electrical-model",
        choices=("simple-pn", "proxy-pinned-split-pd", "profile-ppd"),
        default="proxy-pinned-split-pd",
    )
    parser.add_argument("--measured-profile", type=Path, default=None)
    parser.add_argument("--fixed-charge-sheet-thickness-um", type=float, default=0.02)
    parser.add_argument("--pinning-depth-um", type=float, default=0.08)
    parser.add_argument("--pinning-acceptor-cm3", type=float, default=5.0e16)
    parser.add_argument("--substrate-acceptor-cm3", type=float, default=1.0e14)
    parser.add_argument("--collection-donor-cm3", type=float, default=5.0e15)
    parser.add_argument("--isolation-acceptor-cm3", type=float, default=2.0e16)
    parser.add_argument("--dti-width-um", type=float, default=0.05)
    parser.add_argument("--dti-acceptor-cm3", type=float, default=2.0e16)
    parser.add_argument("--interface-trap-energy-width-ev", type=float, default=0.56)
    parser.add_argument("--interface-trap-reference-potential-v", type=float, default=0.0)
    parser.add_argument("--interface-trap-broadening-v", type=float, default=0.02585)
    parser.add_argument("--interface-trap-thermal-velocity-cm-s", type=float, default=1.0e7)
    parser.add_argument("--electron-mobility-scale", type=float, default=1.0)
    parser.add_argument("--hole-mobility-scale", type=float, default=1.0)
    parser.add_argument("--lifetime-scale", type=float, default=1.0)
    parser.add_argument(
        "--transport-override",
        choices=("profile", "constant-reference"),
        default="profile",
        help="Use profile transport by default; constant-reference is a convergence diagnostic fallback.",
    )
    parser.add_argument(
        "--disable-field-mobility",
        action="store_true",
        help="Disable runtime field-dependent mobility saturation while keeping the selected low-field transport model.",
    )
    parser.add_argument("--fixed-charge-scale", type=float, default=1.0)
    parser.add_argument("--interface-trap-density-scale", type=float, default=1.0)
    parser.add_argument("--interface-trap-recombination-scale", type=float, default=1.0)
    parser.add_argument("--floating-diffusion-feature-scale", type=float, default=1.0)
    parser.add_argument("--transfer-gate-barrier-feature-scale", type=float, default=1.0)
    parser.add_argument("--bdti-liner-feature-scale", type=float, default=1.0)
    parser.add_argument(
        "--resolved-bdti-sidewall-liner",
        action="store_true",
        help="Apply proxy BDTI liner to silicon sidewall nodes on resolved oxide DTI meshes.",
    )
    parser.add_argument("--mu-n-cm2-v-s", type=float, default=400.0)
    parser.add_argument("--mu-p-cm2-v-s", type=float, default=200.0)
    parser.add_argument("--tau-n-s", type=float, default=1.0e-6)
    parser.add_argument("--tau-p-s", type=float, default=1.0e-6)
    parser.add_argument("--reverse-bias-v", type=float, default=-1.0)
    parser.add_argument("--floating-diffusion-bias-v", type=float, default=0.0)
    parser.add_argument("--dd-absolute-error", type=float, default=1.0e10)
    parser.add_argument("--dd-relative-error", type=float, default=1.0e-9)
    parser.add_argument("--dd-max-iterations", type=int, default=160)
    parser.add_argument("--mesh-source", choices=("internal", "gmsh"), default="internal")
    parser.add_argument("--gmsh-mesh", type=Path, default=None)
    args = parser.parse_args()

    config = SplitPD2DConfig(
        width_um=args.width_um,
        depth_um=args.depth_um,
        junction_um=args.junction_um,
        mesh_x_um=args.mesh_x_um,
        mesh_y_um=args.mesh_y_um,
        junction_mesh_um=args.junction_mesh_um,
        split_gap_um=args.split_gap_um,
        acceptor_cm3=args.acceptor_cm3,
        donor_cm3=args.donor_cm3,
        photo_g0_cm3_s=args.photo_g0_cm3_s,
        photo_shift_x_um=args.photo_shift_x_um,
        photo_sigma_x_um=args.photo_sigma_x_um,
        photo_sigma_y_um=args.photo_sigma_y_um,
        generation_profile_csv=str(args.generation_profile_csv) if args.generation_profile_csv else "",
        generation_profile_scale=args.generation_profile_scale,
        generation_profile_case=args.generation_profile_case,
        generation_profile_wavelength_nm=args.generation_profile_wavelength_nm,
        generation_lateral_mode=args.generation_lateral_mode,
        generation_map_npz=str(args.generation_map_npz) if args.generation_map_npz else "",
        generation_map_scale=args.generation_map_scale,
        normalize_generation_map_integral=not args.disable_generation_map_normalization,
        generation_probe_g0_cm3_s=args.generation_probe_g0_cm3_s,
        generation_probe_x_um=args.generation_probe_x_um,
        generation_probe_depth_um=args.generation_probe_depth_um,
        generation_probe_sigma_x_um=args.generation_probe_sigma_x_um,
        generation_probe_sigma_y_um=args.generation_probe_sigma_y_um,
        electrical_model=args.electrical_model,
        measured_profile=str(args.measured_profile) if args.measured_profile else "",
        fixed_charge_sheet_thickness_um=args.fixed_charge_sheet_thickness_um,
        pinning_depth_um=args.pinning_depth_um,
        pinning_acceptor_cm3=args.pinning_acceptor_cm3,
        substrate_acceptor_cm3=args.substrate_acceptor_cm3,
        collection_donor_cm3=args.collection_donor_cm3,
        isolation_acceptor_cm3=args.isolation_acceptor_cm3,
        dti_width_um=args.dti_width_um,
        dti_acceptor_cm3=args.dti_acceptor_cm3,
        interface_trap_energy_width_ev=args.interface_trap_energy_width_ev,
        interface_trap_reference_potential_v=args.interface_trap_reference_potential_v,
        interface_trap_broadening_v=args.interface_trap_broadening_v,
        interface_trap_thermal_velocity_cm_s=args.interface_trap_thermal_velocity_cm_s,
        electron_mobility_scale=args.electron_mobility_scale,
        hole_mobility_scale=args.hole_mobility_scale,
        lifetime_scale=args.lifetime_scale,
        transport_override=args.transport_override,
        disable_field_mobility=args.disable_field_mobility,
        fixed_charge_scale=args.fixed_charge_scale,
        interface_trap_density_scale=args.interface_trap_density_scale,
        interface_trap_recombination_scale=args.interface_trap_recombination_scale,
        floating_diffusion_feature_scale=args.floating_diffusion_feature_scale,
        transfer_gate_barrier_feature_scale=args.transfer_gate_barrier_feature_scale,
        bdti_liner_feature_scale=args.bdti_liner_feature_scale,
        resolved_bdti_sidewall_liner=args.resolved_bdti_sidewall_liner,
        mu_n_cm2_v_s=args.mu_n_cm2_v_s,
        mu_p_cm2_v_s=args.mu_p_cm2_v_s,
        tau_n_s=args.tau_n_s,
        tau_p_s=args.tau_p_s,
        reverse_bias_v=args.reverse_bias_v,
        floating_diffusion_bias_v=args.floating_diffusion_bias_v,
        dd_absolute_error=args.dd_absolute_error,
        dd_relative_error=args.dd_relative_error,
        dd_max_iterations=args.dd_max_iterations,
        mesh_source=args.mesh_source,
        gmsh_mesh=str(args.gmsh_mesh) if args.gmsh_mesh else "",
    )
    summary = run(config, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
