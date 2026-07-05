#!/usr/bin/env python3
"""Native DEVSIM transient TG/FD diagnostic for the split-PD deck.

This script keeps the existing open-source stack, but moves the TG/FD diagnostic
from a quasi-static scale sweep to an actual DEVSIM transient solve.  It runs a
dark reset, an illuminated fill, then a transfer phase where the transfer-gate
barrier feature is ramped open while FD/cathode currents are integrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from devsim import (
    get_circuit_node_value,
    get_contact_list,
    get_node_model_values,
    set_node_values,
    solve,
    write_devices,
)

from devsim_split_pd_2d import (
    Q_E,
    SplitPD2DConfig,
    attach_interface_trap_summary,
    contact_currents,
    create_doping,
    create_interface_trap_sources,
    create_or_import_mesh,
    create_transport_models,
    generation_map_integral_summary,
    parse_gmsh_triangles,
    profile_rows,
    floating_diffusion_circuit_node,
    set_floating_diffusion_reset,
    set_contact_bias,
    set_transfer_gate_bias,
    set_optical_generation,
    set_parameters,
    solve_drift_diffusion,
    solve_potential_only,
    triangle_area_cm2,
)
from measured_tcad_profile import (
    electrical_terms_from_profile,
    load_measured_profile,
    validate_measured_profile,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TGFDTransientConfig:
    generation_map_npz: Path
    measured_profile: Path
    output_dir: Path
    cases: tuple[str, ...] = ("center", "edge20x")
    wavelength_nm: float = 550.0
    width_um: float = 1.4
    depth_um: float = 3.0
    split_gap_um: float = 0.04
    mesh_um: float = 0.10
    fine_mesh_um: float = 0.025
    reverse_bias_v: float = -1.0
    floating_diffusion_bias_v: float = 0.0
    closed_tg_barrier_scale: float = 1.0
    open_tg_barrier_scale: float = 0.0
    fill_steps: int = 3
    fill_step_s: float = 1.0e-9
    transfer_step_s: float = 1.0e-8
    transfer_barrier_scales: tuple[float, ...] = (
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    charge_error: float = 1.0
    dd_relative_error: float = 2.0e-5
    dd_max_iterations: int = 400
    bias_ramp_steps: int = 20
    fd_terminal_mode: str = "ohmic"
    fd_capacitance_f_per_cm: float = 1.0e-11
    fd_reset_on_resistance_ohm_cm: float = 1.0e3
    fd_reset_off_resistance_ohm_cm: float = 1.0e15
    tg_drive_mode: str = "barrier_scale"
    transfer_gate_closed_bias_v: float = -1.0
    transfer_gate_open_bias_v: float = 1.5
    transfer_gate_coupling_sign: float = -1.0
    transfer_open_hold_steps: int = -1
    sequence_mode: str = "paired"
    force: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_float_csv(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def effective_transfer_open_hold_steps(config: TGFDTransientConfig) -> int:
    if config.transfer_open_hold_steps >= 0:
        return config.transfer_open_hold_steps
    return 100 if config.tg_drive_mode == "resolved_gate" else 0


def effective_transfer_barrier_scales(config: TGFDTransientConfig) -> tuple[float, ...]:
    return (
        *config.transfer_barrier_scales,
        *(0.0 for _ in range(effective_transfer_open_hold_steps(config))),
    )


def safe_label(value: str | float) -> str:
    text = f"{value:.3f}" if isinstance(value, float) else str(value)
    return text.replace(".", "p").replace("-", "m")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"command failed; see {log_path}\n{tail}")


def build_fd_mesh(config: TGFDTransientConfig, output_dir: Path) -> Path:
    mesh_dir = output_dir / "gmsh_fd_contact_mesh"
    mesh_path = mesh_dir / "split_pixel_2d.msh"
    if mesh_path.exists() and not config.force:
        return mesh_path
    command = [
        sys.executable,
        str(ROOT / "tcad_gmsh_pixel_mesh.py"),
        "--dimension",
        "2",
        "--measured-profile",
        str(config.measured_profile),
        "--width-um",
        f"{config.width_um:g}",
        "--depth-um",
        f"{config.depth_um:g}",
        "--split-gap-um",
        f"{config.split_gap_um:g}",
        "--mesh-um",
        f"{config.mesh_um:g}",
        "--fine-mesh-um",
        f"{config.fine_mesh_um:g}",
        "--include-fd-contact",
        "--output-dir",
        str(mesh_dir),
    ]
    if config.tg_drive_mode == "gate_contact":
        command.insert(-2, "--include-tg-contact")
    if config.tg_drive_mode == "resolved_gate":
        command.insert(-2, "--include-tg-oxide")
    run_command(command, mesh_dir / "mesh.log")
    return mesh_path


def scaled_profile_path(config: TGFDTransientConfig, output_dir: Path, scale: float) -> Path:
    return output_dir / "profiles" / f"profile_tg_scale_{safe_label(scale)}.json"


def write_scaled_profile(config: TGFDTransientConfig, output_dir: Path, scale: float) -> Path:
    path = scaled_profile_path(config, output_dir, scale)
    if path.exists() and not config.force:
        return path
    source = load_measured_profile(config.measured_profile)
    data = deepcopy(source.data)
    scaled_features = []
    for feature in data.get("electrical_features", []):
        if str(feature.get("role", "")).lower() != "transfer_gate_barrier":
            continue
        for key in ("acceptor_cm3", "donor_cm3"):
            if key in feature:
                feature[key] = float(feature[key]) * scale
        feature["tg_barrier_scale_applied"] = scale
        scaled_features.append(feature.get("name", "transfer_gate_barrier"))
    data["tg_fd_transient_state"] = {
        "source_profile": str(config.measured_profile),
        "transfer_gate_barrier_scale": scale,
        "scaled_features": scaled_features,
    }
    data.setdefault("notes", []).append(
        f"Generated by devsim_tg_fd_transient_2d.py with transfer_gate_barrier scale={scale:g}."
    )
    validate_measured_profile(data, path)
    write_json(path, data)
    return path


def split_config_for_case(
    config: TGFDTransientConfig,
    *,
    case: str,
    mesh_path: Path,
    profile_path: Path,
) -> SplitPD2DConfig:
    return SplitPD2DConfig(
        width_um=config.width_um,
        depth_um=config.depth_um,
        split_gap_um=config.split_gap_um,
        generation_map_npz=str(config.generation_map_npz),
        generation_profile_case=case,
        generation_profile_wavelength_nm=config.wavelength_nm,
        electrical_model="profile-ppd",
        measured_profile=str(profile_path),
        reverse_bias_v=config.reverse_bias_v,
        floating_diffusion_bias_v=config.floating_diffusion_bias_v,
        floating_diffusion_circuit=config.fd_terminal_mode == "circuit",
        floating_diffusion_capacitance_f_per_cm=config.fd_capacitance_f_per_cm,
        floating_diffusion_reset_on_resistance_ohm_cm=config.fd_reset_on_resistance_ohm_cm,
        floating_diffusion_reset_off_resistance_ohm_cm=config.fd_reset_off_resistance_ohm_cm,
        transfer_gate_bias_v=config.transfer_gate_closed_bias_v
        if config.tg_drive_mode in {"gate_contact", "gate_capacitance", "resolved_gate"}
        else 0.0,
        transfer_gate_capacitive_coupling=config.tg_drive_mode == "gate_capacitance",
        transfer_gate_coupling_sign=config.transfer_gate_coupling_sign,
        dd_relative_error=config.dd_relative_error,
        dd_max_iterations=config.dd_max_iterations,
        mesh_source="gmsh",
        gmsh_mesh=str(mesh_path),
    )


def transfer_gate_bias_for_scale(config: TGFDTransientConfig, scale: float) -> float:
    return config.transfer_gate_open_bias_v + scale * (
        config.transfer_gate_closed_bias_v - config.transfer_gate_open_bias_v
    )


def tg_drive_method(config: TGFDTransientConfig) -> str:
    if config.tg_drive_mode == "gate_contact":
        return "native_devsim_transient_bdf1_tg_gate_contact_bias_ramp_with_fd_terminal"
    if config.tg_drive_mode == "gate_capacitance":
        return "native_devsim_transient_bdf1_tg_oxide_capacitance_bias_ramp_with_fd_terminal"
    if config.tg_drive_mode == "resolved_gate":
        return "native_devsim_transient_bdf1_resolved_si_oxide_tg_bias_ramp_with_fd_terminal"
    return "native_devsim_transient_bdf1_tg_barrier_ramp_with_fd_terminal"


def tg_drive_limitations(config: TGFDTransientConfig) -> list[str]:
    if config.tg_drive_mode == "gate_contact":
        return [
            "TG opening is driven by a semiconductor-surface transfer_gate potential contact without a resolved oxide; this direct Dirichlet gate can be numerically stiff and is experimental.",
            "Single-sequence transfer_integrals include TG-ramp dark transient components unless sequence_mode=paired is used by the parent report.",
            "The deck still lacks a resolved oxide/poly gate stack and calibrated measured implant/interface/transport targets, so it is not a product-accuracy LUT.",
        ]
    if config.tg_drive_mode == "gate_capacitance":
        return [
            "TG opening is driven by an oxide-capacitance sheet-charge proxy Cox*(Vg-Psi_s), not a resolved oxide/poly gate mesh.",
            "Single-sequence transfer_integrals include TG-ramp dark transient components unless sequence_mode=paired is used by the parent report.",
            "The deck still lacks calibrated measured implant/interface/transport targets, so it is not a product-accuracy LUT.",
        ]
    if config.tg_drive_mode == "resolved_gate":
        return [
            "TG opening is driven by a resolved Si/oxide interface and metal gate boundary condition in the DEVSIM mesh.",
            "Single-sequence transfer_integrals include TG-ramp dark transient components unless sequence_mode=paired is used by the parent report.",
            "Implant, interface, and mobility/recombination values still come from the supplied profile or public/default references; product accuracy still requires calibration targets.",
        ]
    return [
        "This is a real DEVSIM transient drift-diffusion solve, but the TG opening is represented by ramping the configured transfer_gate_barrier doping feature.",
        "Single-sequence transfer_integrals include TG-ramp dark transient components unless sequence_mode=paired is used by the parent report.",
        "The deck still lacks a resolved oxide/poly gate stack and calibrated measured implant/interface/transport targets, so it is not a product-accuracy LUT.",
    ]


def solve_dc(
    config: SplitPD2DConfig,
    *,
    relative_error: float | None = None,
    maximum_iterations: int | None = None,
) -> None:
    solve(
        type="dc",
        absolute_error=config.dd_absolute_error,
        relative_error=config.dd_relative_error if relative_error is None else relative_error,
        maximum_iterations=config.dd_max_iterations
        if maximum_iterations is None
        else maximum_iterations,
    )


def solve_bias_ramp(
    config: SplitPD2DConfig,
    target_anode_bias_v: float,
    *,
    steps: int,
) -> None:
    if target_anode_bias_v == 0.0:
        set_contact_bias(config, 0.0)
        solve_dc(config)
        return
    for scale in np.linspace(0.0, 1.0, max(1, steps) + 1)[1:]:
        set_contact_bias(config, target_anode_bias_v * scale)
        solve_dc(config)


def solve_transient(
    config: SplitPD2DConfig,
    *,
    kind: str,
    tdelta_s: float,
    charge_error: float,
) -> None:
    solve(
        type=kind,
        tdelta=tdelta_s,
        absolute_error=config.dd_absolute_error,
        relative_error=config.dd_relative_error,
        charge_error=charge_error,
        maximum_iterations=config.dd_max_iterations,
    )


def apply_profile_doping(
    split_config: SplitPD2DConfig,
    profile_path: Path,
) -> dict[str, Any]:
    profile = load_measured_profile(profile_path)
    x = np.asarray(get_node_model_values(device=split_config.device, region=split_config.region, name="x"))
    y = np.asarray(get_node_model_values(device=split_config.device, region=split_config.region, name="y"))
    donors, acceptors, fixed_charge, feature_summary = electrical_terms_from_profile(
        profile,
        x,
        y,
        None,
        default_sheet_thickness_um=split_config.fixed_charge_sheet_thickness_um,
    )
    set_node_values(
        device=split_config.device,
        region=split_config.region,
        name="Donors",
        values=donors.tolist(),
    )
    set_node_values(
        device=split_config.device,
        region=split_config.region,
        name="Acceptors",
        values=acceptors.tolist(),
    )
    set_node_values(
        device=split_config.device,
        region=split_config.region,
        name="FixedChargeDoping",
        values=fixed_charge.tolist(),
    )
    net = donors - acceptors + fixed_charge
    return {
        "profile": str(profile_path),
        "donor_min_cm3": float(np.min(donors)),
        "donor_max_cm3": float(np.max(donors)),
        "acceptor_min_cm3": float(np.min(acceptors)),
        "acceptor_max_cm3": float(np.max(acceptors)),
        "fixed_charge_doping_min_cm3": float(np.min(fixed_charge)),
        "fixed_charge_doping_max_cm3": float(np.max(fixed_charge)),
        "net_min_cm3": float(np.min(net)),
        "net_max_cm3": float(np.max(net)),
        "feature_summary": feature_summary,
    }


def contact_current_fields(split_config: SplitPD2DConfig) -> dict[str, float]:
    row: dict[str, float] = {}
    for contact in get_contact_list(device=split_config.device):
        row.update(contact_currents(split_config, contact))
    row["total_terminal_current_a_per_cm"] = sum(
        row[f"{contact}_total_current_a_per_cm"]
        for contact in get_contact_list(device=split_config.device)
    )
    return row


def sample_row(
    split_config: SplitPD2DConfig,
    *,
    phase: str,
    step_index: int,
    time_s: float,
    dt_s: float,
    tg_barrier_scale: float,
    illuminated: bool,
    dark_baseline: dict[str, float] | None,
    transfer_gate_bias_v: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "step_index": step_index,
        "time_s": time_s,
        "dt_s": dt_s,
        "tg_barrier_scale": tg_barrier_scale,
        "illuminated": illuminated,
    }
    if transfer_gate_bias_v is not None:
        row["transfer_gate_bias_v"] = transfer_gate_bias_v
    row.update(contact_current_fields(split_config))
    if split_config.floating_diffusion_circuit:
        try:
            row["floating_diffusion_circuit_voltage_v"] = get_circuit_node_value(
                node=floating_diffusion_circuit_node(split_config),
                solution="dcop",
            )
        except Exception:
            row["floating_diffusion_circuit_voltage_v"] = math.nan
    if dark_baseline:
        for contact in get_contact_list(device=split_config.device):
            for carrier in ("electron", "hole", "total"):
                key = f"{contact}_{carrier}_current_a_per_cm"
                row[f"{contact}_{carrier}_delta_from_dark_a_per_cm"] = (
                    float(row[key]) - float(dark_baseline[key])
                )
    return row


def integrate_transfer_rows(rows: list[dict[str, Any]], contacts: list[str]) -> dict[str, Any]:
    transfer_rows = [row for row in rows if row["phase"] == "transfer"]
    integrated: dict[str, float] = {}
    abs_integrated: dict[str, float] = {}
    for contact in contacts:
        for carrier in ("electron", "hole", "total"):
            delta_key = f"{contact}_{carrier}_delta_from_dark_a_per_cm"
            charge_key = f"{contact}_{carrier}_charge_c_per_cm"
            abs_key = f"{contact}_{carrier}_abs_charge_c_per_cm"
            value = sum(float(row.get(delta_key, 0.0)) * float(row["dt_s"]) for row in transfer_rows)
            abs_value = sum(abs(float(row.get(delta_key, 0.0))) * float(row["dt_s"]) for row in transfer_rows)
            integrated[charge_key] = value
            abs_integrated[abs_key] = abs_value
    fd_e = abs_integrated.get("floating_diffusion_electron_abs_charge_c_per_cm", math.nan)
    left_e = abs_integrated.get("cathode_left_electron_abs_charge_c_per_cm", math.nan)
    right_e = abs_integrated.get("cathode_right_electron_abs_charge_c_per_cm", math.nan)
    denom = sum(value for value in (fd_e, left_e, right_e) if math.isfinite(value))
    fd_fraction = fd_e / denom if denom else math.nan
    tail_count = min(2, len(transfer_rows))
    fd_tail = sum(
        abs(float(row.get("floating_diffusion_electron_delta_from_dark_a_per_cm", 0.0)))
        * float(row["dt_s"])
        for row in transfer_rows[-tail_count:]
    )
    fd_currents = [
        abs(float(row.get("floating_diffusion_electron_delta_from_dark_a_per_cm", 0.0)))
        for row in transfer_rows
    ]
    fd_peak_current = max(fd_currents) if fd_currents else math.nan
    fd_last_current = fd_currents[-1] if fd_currents else math.nan
    fd_last_two_mean_current = (
        sum(fd_currents[-tail_count:]) / tail_count if tail_count and fd_currents else math.nan
    )
    fd_last_to_peak = (
        fd_last_current / fd_peak_current
        if fd_peak_current and math.isfinite(fd_peak_current)
        else math.nan
    )
    fd_last_two_mean_to_peak = (
        fd_last_two_mean_current / fd_peak_current
        if fd_peak_current and math.isfinite(fd_peak_current)
        else math.nan
    )
    return {
        "integrated_charge": integrated,
        "integrated_abs_charge": abs_integrated,
        "floating_diffusion_electron_abs_fraction_of_fd_plus_cathodes": fd_fraction,
        "floating_diffusion_electrons_per_cm": (
            integrated.get("floating_diffusion_electron_charge_c_per_cm", math.nan) / Q_E
        ),
        "floating_diffusion_abs_electrons_per_cm": fd_e / Q_E if math.isfinite(fd_e) else math.nan,
        "floating_diffusion_tail_abs_charge_fraction_last_two_steps": fd_tail / fd_e
        if fd_e and math.isfinite(fd_e)
        else math.nan,
        "floating_diffusion_peak_abs_current_a_per_cm": fd_peak_current,
        "floating_diffusion_last_abs_current_a_per_cm": fd_last_current,
        "floating_diffusion_last_abs_current_to_peak": fd_last_to_peak,
        "floating_diffusion_last_two_mean_abs_current_to_peak": fd_last_two_mean_to_peak,
    }


def run_single_sequence(config: TGFDTransientConfig, case: str, output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "tg_fd_transient_report.json"
    if report_path.exists() and not config.force:
        report = read_json(report_path)
        print(json.dumps(report, indent=2))
        return report
    if config.sequence_mode not in {"photo", "dark"}:
        raise ValueError(f"run_single_sequence needs sequence_mode photo or dark, got {config.sequence_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = build_fd_mesh(config, output_dir)
    profile_scale = (
        config.closed_tg_barrier_scale
        if config.tg_drive_mode == "barrier_scale"
        else config.open_tg_barrier_scale
    )
    closed_profile = write_scaled_profile(config, output_dir, profile_scale)
    split_config = split_config_for_case(
        config,
        case=case,
        mesh_path=mesh_path,
        profile_path=closed_profile,
    )

    create_or_import_mesh(split_config)
    set_parameters(split_config)
    doping_summary = create_doping(split_config)
    transport_summary = create_transport_models(split_config)
    interface_trap_summary = create_interface_trap_sources(split_config)
    attach_interface_trap_summary(doping_summary, interface_trap_summary)
    current_tg_bias = (
        config.transfer_gate_closed_bias_v
        if config.tg_drive_mode in {"gate_contact", "gate_capacitance", "resolved_gate"}
        else None
    )
    if current_tg_bias is not None:
        set_transfer_gate_bias(split_config, current_tg_bias)
    set_contact_bias(split_config, 0.0)
    solve_potential_only(split_config)
    solve_drift_diffusion(split_config, transport_summary)

    set_optical_generation(split_config, illuminated=False)
    solve_bias_ramp(
        split_config,
        split_config.reverse_bias_v,
        steps=config.bias_ramp_steps,
    )
    dark_baseline = contact_current_fields(split_config)

    rows: list[dict[str, Any]] = [
        sample_row(
            split_config,
            phase="dark_reset",
            step_index=0,
            time_s=0.0,
            dt_s=0.0,
            tg_barrier_scale=config.closed_tg_barrier_scale,
            transfer_gate_bias_v=current_tg_bias,
            illuminated=False,
            dark_baseline=dark_baseline,
        )
    ]

    set_floating_diffusion_reset(split_config, enabled=False)
    time_s = 0.0
    fill_illuminated = config.sequence_mode == "photo"
    set_optical_generation(split_config, illuminated=fill_illuminated)
    for step_index in range(1, config.fill_steps + 1):
        kind = "transient_dc" if step_index == 1 else "transient_bdf1"
        solve_transient(
            split_config,
            kind=kind,
            tdelta_s=config.fill_step_s,
            charge_error=config.charge_error,
        )
        time_s += config.fill_step_s
        rows.append(
            sample_row(
                split_config,
                phase="fill",
                step_index=step_index,
                time_s=time_s,
                dt_s=config.fill_step_s,
                tg_barrier_scale=config.closed_tg_barrier_scale,
                transfer_gate_bias_v=current_tg_bias,
                illuminated=fill_illuminated,
                dark_baseline=dark_baseline,
            )
        )

    fill_profiles = profile_rows(split_config)
    fill_generation_integral = generation_map_integral_summary(split_config, fill_profiles)

    set_optical_generation(split_config, illuminated=False)
    scale_summaries: dict[str, Any] = {}
    for step_index, scale in enumerate(effective_transfer_barrier_scales(config), start=1):
        if config.tg_drive_mode in {"gate_contact", "gate_capacitance", "resolved_gate"}:
            current_tg_bias = transfer_gate_bias_for_scale(config, scale)
            set_transfer_gate_bias(split_config, current_tg_bias)
            scale_summaries[safe_label(scale)] = {
                "method": f"transfer_gate_{config.tg_drive_mode}_bias_ramp",
                "transfer_gate_bias_v": current_tg_bias,
                "fixed_profile_tg_barrier_scale": profile_scale,
            }
        else:
            profile_path = write_scaled_profile(config, output_dir, scale)
            scale_summaries[safe_label(scale)] = apply_profile_doping(split_config, profile_path)
        solve_transient(
            split_config,
            kind="transient_bdf1",
            tdelta_s=config.transfer_step_s,
            charge_error=config.charge_error,
        )
        time_s += config.transfer_step_s
        rows.append(
            sample_row(
                split_config,
                phase="transfer",
                step_index=step_index,
                time_s=time_s,
                dt_s=config.transfer_step_s,
                tg_barrier_scale=scale,
                transfer_gate_bias_v=current_tg_bias,
                illuminated=False,
                dark_baseline=dark_baseline,
            )
        )

    contacts = list(get_contact_list(device=split_config.device))
    transfer_integrals = integrate_transfer_rows(rows, contacts)
    balances = [
        abs(float(row.get("total_terminal_current_a_per_cm", 0.0)))
        for row in rows
        if math.isfinite(float(row.get("total_terminal_current_a_per_cm", 0.0)))
    ]
    profiles = profile_rows(split_config)
    csv_path = output_dir / "tg_fd_transient_timeseries.csv"
    fill_node_csv_path = output_dir / "node_profile_after_fill.csv"
    node_csv_path = output_dir / "node_profile_after_transfer.csv"
    tecplot_path = output_dir / "tg_fd_transient_device_after_transfer.dat"
    write_csv(csv_path, rows)
    write_csv(fill_node_csv_path, fill_profiles)
    write_csv(node_csv_path, profiles)
    write_devices(file=str(tecplot_path), type="tecplot")

    report = {
        "schema": "devsim_tg_fd_transient_2d_v1",
        "method": tg_drive_method(config),
        "devsim_version": version("devsim"),
        "product_accuracy_ready": False,
        "sequence_mode": config.sequence_mode,
        "case": case,
        "wavelength_nm": config.wavelength_nm,
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "measured_profile": str(config.measured_profile),
            "output_dir": str(output_dir),
            "transfer_open_hold_steps_effective": effective_transfer_open_hold_steps(config),
        },
        "mesh": str(mesh_path),
        "contacts": contacts,
        "initial_doping_summary": doping_summary,
        "transfer_scale_doping_summaries": scale_summaries,
        "transport_summary": transport_summary,
        "interface_trap_summary": interface_trap_summary,
        "fill_generation_integral_summary": fill_generation_integral,
        "final_generation_integral_summary": generation_map_integral_summary(split_config, profiles),
        "row_count": len(rows),
        "terminal_balance_max_abs_a_per_cm": max(balances) if balances else math.nan,
        "transfer_integrals": transfer_integrals,
        "outputs": {
            "report_json": str(report_path),
            "timeseries_csv": str(csv_path),
            "node_profile_after_fill_csv": str(fill_node_csv_path),
            "node_profile_after_transfer_csv": str(node_csv_path),
            "device_tecplot_after_transfer": str(tecplot_path),
        },
        "rows": rows,
        "limitations": tg_drive_limitations(config),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return report


def sequence_command(
    config: TGFDTransientConfig,
    *,
    case: str,
    output_dir: Path,
    sequence_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--generation-map-npz",
        str(config.generation_map_npz),
        "--measured-profile",
        str(config.measured_profile),
        "--output-dir",
        str(output_dir),
        "--cases",
        case,
        "--sequence-mode",
        sequence_mode,
        "--wavelength-nm",
        f"{config.wavelength_nm:g}",
        "--width-um",
        f"{config.width_um:g}",
        "--depth-um",
        f"{config.depth_um:g}",
        "--split-gap-um",
        f"{config.split_gap_um:g}",
        "--mesh-um",
        f"{config.mesh_um:g}",
        "--fine-mesh-um",
        f"{config.fine_mesh_um:g}",
        "--reverse-bias-v",
        f"{config.reverse_bias_v:g}",
        "--floating-diffusion-bias-v",
        f"{config.floating_diffusion_bias_v:g}",
        "--closed-tg-barrier-scale",
        f"{config.closed_tg_barrier_scale:g}",
        "--open-tg-barrier-scale",
        f"{config.open_tg_barrier_scale:g}",
        "--fill-steps",
        str(config.fill_steps),
        "--fill-step-s",
        f"{config.fill_step_s:g}",
        "--transfer-step-s",
        f"{config.transfer_step_s:g}",
        "--transfer-barrier-scales",
        ",".join(f"{value:g}" for value in config.transfer_barrier_scales),
        "--transfer-open-hold-steps",
        str(config.transfer_open_hold_steps),
        "--charge-error",
        f"{config.charge_error:g}",
        "--dd-relative-error",
        f"{config.dd_relative_error:g}",
        "--dd-max-iterations",
        str(config.dd_max_iterations),
        "--bias-ramp-steps",
        str(config.bias_ramp_steps),
        "--fd-terminal-mode",
        config.fd_terminal_mode,
        "--fd-capacitance-f-per-cm",
        f"{config.fd_capacitance_f_per_cm:g}",
        "--fd-reset-on-resistance-ohm-cm",
        f"{config.fd_reset_on_resistance_ohm_cm:g}",
        "--fd-reset-off-resistance-ohm-cm",
        f"{config.fd_reset_off_resistance_ohm_cm:g}",
        "--tg-drive-mode",
        config.tg_drive_mode,
        "--transfer-gate-closed-bias-v",
        f"{config.transfer_gate_closed_bias_v:g}",
        "--transfer-gate-open-bias-v",
        f"{config.transfer_gate_open_bias_v:g}",
        "--transfer-gate-coupling-sign",
        f"{config.transfer_gate_coupling_sign:g}",
    ]
    if config.force:
        command.append("--force")
    return command


def transfer_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report.get("rows", []) if row.get("phase") == "transfer"]


def read_node_csv(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return {
        (round(row["x_cm"], 14), round(row["y_cm"], 14)): row
        for row in rows
    }


def inventory_regions(profile_path: Path) -> dict[str, dict[str, float]]:
    profile = load_measured_profile(profile_path)
    geometry = profile.geometry
    half_width = 0.5 * float(geometry["width_um"])
    split_half = 0.5 * float(geometry["split_gap_um"])
    depth = float(geometry["depth_um"])
    pinning_depth = float(geometry.get("pinning_depth_um", 0.08))
    fd = geometry.get("floating_diffusion", {})
    return {
        "silicon": {
            "x_min_um": -half_width,
            "x_max_um": half_width,
            "y_min_um": 0.0,
            "y_max_um": depth,
        },
        "pd_left": {
            "x_min_um": -half_width,
            "x_max_um": -split_half,
            "y_min_um": pinning_depth,
            "y_max_um": depth,
        },
        "pd_right": {
            "x_min_um": split_half,
            "x_max_um": half_width,
            "y_min_um": pinning_depth,
            "y_max_um": depth,
        },
        "pd_total": {
            "x_min_um": -half_width,
            "x_max_um": half_width,
            "exclude_x_min_um": -split_half,
            "exclude_x_max_um": split_half,
            "y_min_um": pinning_depth,
            "y_max_um": depth,
        },
        "floating_diffusion": {
            "x_min_um": float(fd.get("x_min_um", 0.0)),
            "x_max_um": float(fd.get("x_max_um", 0.0)),
            "y_min_um": float(fd.get("depth_min_um", 0.0)),
            "y_max_um": float(fd.get("depth_max_um", 0.0)),
        },
    }


def point_in_region(x_um: float, y_um: float, region: dict[str, float]) -> bool:
    inside = (
        x_um >= region["x_min_um"]
        and x_um <= region["x_max_um"]
        and y_um >= region["y_min_um"]
        and y_um <= region["y_max_um"]
    )
    if not inside:
        return False
    if "exclude_x_min_um" in region and "exclude_x_max_um" in region:
        if x_um >= region["exclude_x_min_um"] and x_um <= region["exclude_x_max_um"]:
            return False
    return True


def integrate_inventory(mesh_path: Path, node_csv: Path, profile_path: Path) -> dict[str, dict[str, float]]:
    nodes, triangles = parse_gmsh_triangles(str(mesh_path))
    rows_by_xy = read_node_csv(node_csv)
    regions = inventory_regions(profile_path)
    inventory = {
        name: {
            "area_cm2": 0.0,
            "electron_count_per_cm": 0.0,
            "hole_count_per_cm": 0.0,
            "electron_charge_c_per_cm": 0.0,
            "hole_charge_c_per_cm": 0.0,
        }
        for name in regions
    }
    for triangle in triangles:
        points = [nodes[node_id] for node_id in triangle]
        rows = []
        for x, y in points:
            row = rows_by_xy.get((round(x, 14), round(y, 14)))
            if row is None:
                rows = []
                break
            rows.append(row)
        if not rows:
            continue
        area = triangle_area_cm2(points)
        x_um = sum(point[0] for point in points) / 3.0 * 1.0e4
        y_um = sum(point[1] for point in points) / 3.0 * 1.0e4
        electrons = sum(row["Electrons"] for row in rows) / 3.0
        holes = sum(row["Holes"] for row in rows) / 3.0
        for name, region in regions.items():
            if not point_in_region(x_um, y_um, region):
                continue
            inventory[name]["area_cm2"] += area
            inventory[name]["electron_count_per_cm"] += electrons * area
            inventory[name]["hole_count_per_cm"] += holes * area
    for item in inventory.values():
        item["electron_charge_c_per_cm"] = -Q_E * item["electron_count_per_cm"]
        item["hole_charge_c_per_cm"] = Q_E * item["hole_count_per_cm"]
    return inventory


def subtract_inventory(
    lhs: dict[str, dict[str, float]],
    rhs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for region in lhs:
        result[region] = {}
        for key, value in lhs[region].items():
            result[region][key] = float(value) - float(rhs.get(region, {}).get(key, 0.0))
    return result


def inventory_delta(
    final: dict[str, dict[str, float]],
    initial: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return subtract_inventory(final, initial)


def paired_inventory_summary(
    config: TGFDTransientConfig,
    photo_report: dict[str, Any],
    dark_report: dict[str, Any],
) -> dict[str, Any]:
    mesh_path = Path(str(photo_report["mesh"]))
    photo_outputs = photo_report.get("outputs", {})
    dark_outputs = dark_report.get("outputs", {})
    photo_fill = integrate_inventory(
        mesh_path,
        Path(photo_outputs["node_profile_after_fill_csv"]),
        config.measured_profile,
    )
    photo_final = integrate_inventory(
        mesh_path,
        Path(photo_outputs["node_profile_after_transfer_csv"]),
        config.measured_profile,
    )
    dark_fill = integrate_inventory(
        mesh_path,
        Path(dark_outputs["node_profile_after_fill_csv"]),
        config.measured_profile,
    )
    dark_final = integrate_inventory(
        mesh_path,
        Path(dark_outputs["node_profile_after_transfer_csv"]),
        config.measured_profile,
    )
    excess_fill = subtract_inventory(photo_fill, dark_fill)
    excess_final = subtract_inventory(photo_final, dark_final)
    excess_delta = inventory_delta(excess_final, excess_fill)
    pd_fill = excess_fill.get("pd_total", {}).get("electron_count_per_cm", math.nan)
    pd_final = excess_final.get("pd_total", {}).get("electron_count_per_cm", math.nan)
    fd_fill = excess_fill.get("floating_diffusion", {}).get("electron_count_per_cm", math.nan)
    fd_final = excess_final.get("floating_diffusion", {}).get("electron_count_per_cm", math.nan)
    return {
        "method": "gmsh_triangle_integrated_photo_minus_dark_carrier_inventory",
        "regions": inventory_regions(config.measured_profile),
        "photo_fill": photo_fill,
        "photo_after_transfer": photo_final,
        "dark_fill": dark_fill,
        "dark_after_transfer": dark_final,
        "photo_minus_dark_fill": excess_fill,
        "photo_minus_dark_after_transfer": excess_final,
        "photo_minus_dark_after_minus_fill": excess_delta,
        "pd_total_excess_electron_fraction_remaining": pd_final / pd_fill
        if pd_fill and math.isfinite(pd_fill)
        else math.nan,
        "floating_diffusion_excess_electron_gain_per_cm": fd_final - fd_fill
        if math.isfinite(fd_final) and math.isfinite(fd_fill)
        else math.nan,
    }


def paired_photo_minus_dark_rows(
    photo_report: dict[str, Any],
    dark_report: dict[str, Any],
) -> list[dict[str, Any]]:
    contacts = [str(contact) for contact in photo_report.get("contacts", [])]
    photo_rows = transfer_rows(photo_report)
    dark_rows = transfer_rows(dark_report)
    if len(photo_rows) != len(dark_rows):
        raise RuntimeError("photo and dark transfer row counts differ")
    rows: list[dict[str, Any]] = []
    for photo_row, dark_row in zip(photo_rows, dark_rows):
        if int(photo_row["step_index"]) != int(dark_row["step_index"]):
            raise RuntimeError("photo and dark transfer step indices differ")
        row: dict[str, Any] = {
            "phase": "transfer",
            "signal_mode": "photo_minus_dark",
            "step_index": int(photo_row["step_index"]),
            "time_s": float(photo_row["time_s"]),
            "dt_s": float(photo_row["dt_s"]),
            "tg_barrier_scale": float(photo_row["tg_barrier_scale"]),
            "illuminated": False,
        }
        if "transfer_gate_bias_v" in photo_row:
            row["transfer_gate_bias_v"] = float(photo_row["transfer_gate_bias_v"])
        for contact in contacts:
            for carrier in ("electron", "hole", "total"):
                key = f"{contact}_{carrier}_delta_from_dark_a_per_cm"
                photo_value = float(photo_row.get(key, 0.0))
                dark_value = float(dark_row.get(key, 0.0))
                isolated = photo_value - dark_value
                row[f"{contact}_{carrier}_photo_delta_from_dark_a_per_cm"] = photo_value
                row[f"{contact}_{carrier}_dark_delta_from_dark_a_per_cm"] = dark_value
                row[f"{contact}_{carrier}_photo_minus_dark_delta_a_per_cm"] = isolated
                row[key] = isolated
        if "floating_diffusion_circuit_voltage_v" in photo_row or "floating_diffusion_circuit_voltage_v" in dark_row:
            photo_voltage = float(photo_row.get("floating_diffusion_circuit_voltage_v", math.nan))
            dark_voltage = float(dark_row.get("floating_diffusion_circuit_voltage_v", math.nan))
            row["floating_diffusion_photo_circuit_voltage_v"] = photo_voltage
            row["floating_diffusion_dark_circuit_voltage_v"] = dark_voltage
            row["floating_diffusion_photo_minus_dark_circuit_voltage_v"] = (
                photo_voltage - dark_voltage
                if math.isfinite(photo_voltage) and math.isfinite(dark_voltage)
                else math.nan
            )
        rows.append(row)
    return rows


def paired_circuit_voltage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["floating_diffusion_photo_minus_dark_circuit_voltage_v"])
        for row in rows
        if isinstance(row.get("floating_diffusion_photo_minus_dark_circuit_voltage_v"), (int, float))
        and math.isfinite(float(row["floating_diffusion_photo_minus_dark_circuit_voltage_v"]))
    ]
    if not values:
        return {}
    return {
        "method": "paired_photo_minus_dark_fd_circuit_node_voltage",
        "final_voltage_delta_v": values[-1],
        "min_voltage_delta_v": min(values),
        "max_voltage_delta_v": max(values),
        "peak_abs_voltage_delta_v": max(abs(value) for value in values),
        "last_abs_voltage_to_peak": abs(values[-1]) / max(abs(value) for value in values)
        if max(abs(value) for value in values) > 0.0
        else math.nan,
    }


def run_paired_case(config: TGFDTransientConfig, case: str, output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "tg_fd_transient_report.json"
    if report_path.exists() and not config.force:
        report = read_json(report_path)
        print(json.dumps(report, indent=2))
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_root = output_dir / "sequences"
    photo_dir = sequence_root / "photo"
    dark_dir = sequence_root / "dark"
    for mode, target_dir in (("photo", photo_dir), ("dark", dark_dir)):
        target_report = target_dir / "tg_fd_transient_report.json"
        if config.force or not target_report.exists():
            run_command(
                sequence_command(config, case=case, output_dir=target_dir, sequence_mode=mode),
                output_dir / "logs" / f"{mode}.log",
            )
    photo_report = read_json(photo_dir / "tg_fd_transient_report.json")
    dark_report = read_json(dark_dir / "tg_fd_transient_report.json")
    contacts = [str(contact) for contact in photo_report.get("contacts", [])]
    paired_rows = paired_photo_minus_dark_rows(photo_report, dark_report)
    transfer_integrals = integrate_transfer_rows(paired_rows, contacts)
    inventory_summary = paired_inventory_summary(config, photo_report, dark_report)
    circuit_voltage_summary = paired_circuit_voltage_summary(paired_rows)
    csv_path = output_dir / "tg_fd_transient_photo_minus_dark_timeseries.csv"
    write_csv(csv_path, paired_rows)
    balances = [
        float(value)
        for value in (
            photo_report.get("terminal_balance_max_abs_a_per_cm", math.nan),
            dark_report.get("terminal_balance_max_abs_a_per_cm", math.nan),
        )
        if math.isfinite(float(value))
    ]
    report = {
        "schema": "devsim_tg_fd_transient_2d_v1",
        "method": f"{tg_drive_method(config)}_photo_minus_dark",
        "devsim_version": photo_report.get("devsim_version"),
        "product_accuracy_ready": False,
        "sequence_mode": "paired",
        "signal_mode": "photo_minus_dark",
        "case": case,
        "wavelength_nm": config.wavelength_nm,
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "measured_profile": str(config.measured_profile),
            "output_dir": str(output_dir),
            "transfer_open_hold_steps_effective": effective_transfer_open_hold_steps(config),
        },
        "mesh": photo_report.get("mesh"),
        "contacts": contacts,
        "fill_generation_integral_summary": photo_report.get("fill_generation_integral_summary", {}),
        "row_count": len(paired_rows),
        "terminal_balance_max_abs_a_per_cm": max(balances) if balances else math.nan,
        "transfer_integrals": transfer_integrals,
        "floating_diffusion_circuit_voltage": circuit_voltage_summary,
        "carrier_inventory": inventory_summary,
        "raw_photo_transfer_integrals": photo_report.get("transfer_integrals", {}),
        "raw_dark_transfer_integrals": dark_report.get("transfer_integrals", {}),
        "outputs": {
            "report_json": str(report_path),
            "photo_minus_dark_timeseries_csv": str(csv_path),
            "photo_report_json": str(photo_dir / "tg_fd_transient_report.json"),
            "dark_report_json": str(dark_dir / "tg_fd_transient_report.json"),
            "photo_timeseries_csv": str(photo_dir / "tg_fd_transient_timeseries.csv"),
            "dark_timeseries_csv": str(dark_dir / "tg_fd_transient_timeseries.csv"),
        },
        "rows": paired_rows,
        "limitations": [
            "Photo transfer is isolated by subtracting a matching dark TG-ramp transient from the illuminated-fill transient.",
            *tg_drive_limitations(config),
        ],
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return report


def run_single_case(config: TGFDTransientConfig, case: str, output_dir: Path) -> dict[str, Any]:
    if config.sequence_mode in {"photo", "dark"}:
        return run_single_sequence(config, case, output_dir)
    if config.sequence_mode == "paired":
        return run_paired_case(config, case, output_dir)
    raise ValueError(f"Unsupported sequence_mode: {config.sequence_mode}")


def run(config: TGFDTransientConfig) -> dict[str, Any]:
    if len(config.cases) == 1:
        return run_single_case(config, config.cases[0], config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    case_reports = []
    for case in config.cases:
        case_dir = config.output_dir / "cases" / case
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--generation-map-npz",
            str(config.generation_map_npz),
            "--measured-profile",
            str(config.measured_profile),
            "--output-dir",
            str(case_dir),
            "--cases",
            case,
            "--sequence-mode",
            config.sequence_mode,
            "--wavelength-nm",
            f"{config.wavelength_nm:g}",
            "--width-um",
            f"{config.width_um:g}",
            "--depth-um",
            f"{config.depth_um:g}",
            "--split-gap-um",
            f"{config.split_gap_um:g}",
            "--mesh-um",
            f"{config.mesh_um:g}",
            "--fine-mesh-um",
            f"{config.fine_mesh_um:g}",
            "--reverse-bias-v",
            f"{config.reverse_bias_v:g}",
            "--floating-diffusion-bias-v",
            f"{config.floating_diffusion_bias_v:g}",
            "--closed-tg-barrier-scale",
            f"{config.closed_tg_barrier_scale:g}",
            "--open-tg-barrier-scale",
            f"{config.open_tg_barrier_scale:g}",
            "--fill-steps",
            str(config.fill_steps),
            "--fill-step-s",
            f"{config.fill_step_s:g}",
            "--transfer-step-s",
            f"{config.transfer_step_s:g}",
            "--transfer-barrier-scales",
            ",".join(f"{value:g}" for value in config.transfer_barrier_scales),
            "--transfer-open-hold-steps",
            str(config.transfer_open_hold_steps),
            "--charge-error",
            f"{config.charge_error:g}",
            "--dd-relative-error",
            f"{config.dd_relative_error:g}",
            "--dd-max-iterations",
            str(config.dd_max_iterations),
            "--bias-ramp-steps",
            str(config.bias_ramp_steps),
            "--fd-terminal-mode",
            config.fd_terminal_mode,
            "--fd-capacitance-f-per-cm",
            f"{config.fd_capacitance_f_per_cm:g}",
            "--fd-reset-on-resistance-ohm-cm",
            f"{config.fd_reset_on_resistance_ohm_cm:g}",
            "--fd-reset-off-resistance-ohm-cm",
            f"{config.fd_reset_off_resistance_ohm_cm:g}",
            "--tg-drive-mode",
            config.tg_drive_mode,
            "--transfer-gate-closed-bias-v",
            f"{config.transfer_gate_closed_bias_v:g}",
            "--transfer-gate-open-bias-v",
            f"{config.transfer_gate_open_bias_v:g}",
            "--transfer-gate-coupling-sign",
            f"{config.transfer_gate_coupling_sign:g}",
        ]
        if config.force:
            command.append("--force")
        run_command(command, config.output_dir / "logs" / f"{case}.log")
        case_reports.append(read_json(case_dir / "tg_fd_transient_report.json"))

    rows: list[dict[str, Any]] = []
    for report in case_reports:
        for row in report.get("rows", []):
            rows.append({"case": report.get("case"), **row})
    csv_path = config.output_dir / "tg_fd_transient_timeseries.csv"
    write_csv(csv_path, rows)
    report_path = config.output_dir / "tg_fd_transient_report.json"
    balances = [
        float(report.get("terminal_balance_max_abs_a_per_cm", math.nan))
        for report in case_reports
        if math.isfinite(float(report.get("terminal_balance_max_abs_a_per_cm", math.nan)))
    ]
    aggregate = {
        "schema": "devsim_tg_fd_transient_sweep_2d_v1",
        "method": tg_drive_method(config),
        "product_accuracy_ready": False,
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "measured_profile": str(config.measured_profile),
            "output_dir": str(config.output_dir),
            "transfer_open_hold_steps_effective": effective_transfer_open_hold_steps(config),
        },
        "case_count": len(case_reports),
        "row_count": len(rows),
        "terminal_balance_max_abs_a_per_cm": max(balances) if balances else math.nan,
        "cases": [
            {
                "case": report.get("case"),
                "report_json": report.get("outputs", {}).get("report_json"),
                "timeseries_csv": report.get("outputs", {}).get("timeseries_csv"),
                "photo_minus_dark_timeseries_csv": report.get("outputs", {}).get(
                    "photo_minus_dark_timeseries_csv"
                ),
                "transfer_integrals": report.get("transfer_integrals", {}),
                "floating_diffusion_circuit_voltage": report.get(
                    "floating_diffusion_circuit_voltage", {}
                ),
                "signal_mode": report.get("signal_mode", report.get("sequence_mode")),
                "terminal_balance_max_abs_a_per_cm": report.get("terminal_balance_max_abs_a_per_cm"),
                "carrier_inventory": {
                    "method": report.get("carrier_inventory", {}).get("method"),
                    "pd_total_excess_electron_fraction_remaining": report.get(
                        "carrier_inventory", {}
                    ).get("pd_total_excess_electron_fraction_remaining"),
                    "floating_diffusion_excess_electron_gain_per_cm": report.get(
                        "carrier_inventory", {}
                    ).get("floating_diffusion_excess_electron_gain_per_cm"),
                    "floating_diffusion_terminal_electrons_per_cm": report.get(
                        "transfer_integrals", {}
                    ).get("floating_diffusion_electrons_per_cm"),
                    "floating_diffusion_terminal_abs_electrons_per_cm": report.get(
                        "transfer_integrals", {}
                    ).get("floating_diffusion_abs_electrons_per_cm"),
                    "floating_diffusion_terminal_abs_fraction_of_fd_plus_cathodes": report.get(
                        "transfer_integrals", {}
                    ).get("floating_diffusion_electron_abs_fraction_of_fd_plus_cathodes"),
                    "floating_diffusion_last_abs_current_to_peak": report.get(
                        "transfer_integrals", {}
                    ).get("floating_diffusion_last_abs_current_to_peak"),
                },
            }
            for report in case_reports
        ],
        "outputs": {
            "report_json": str(report_path),
            "timeseries_csv": str(csv_path),
        },
        "limitations": [
            "Each case is run in a fresh DEVSIM process because DEVSIM device state is global.",
            *tg_drive_limitations(config),
        ],
    }
    write_json(report_path, aggregate)
    print(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-map-npz",
        type=Path,
        default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
    )
    parser.add_argument(
        "--measured-profile",
        type=Path,
        default=ROOT / "measured_profiles/reference_cmos_ppd_1p4um/profile.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/devsim_tg_fd_transient_2d_reference")
    parser.add_argument("--cases", default="center,edge20x")
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--mesh-um", type=float, default=0.10)
    parser.add_argument("--fine-mesh-um", type=float, default=0.025)
    parser.add_argument("--reverse-bias-v", type=float, default=-1.0)
    parser.add_argument("--floating-diffusion-bias-v", type=float, default=0.0)
    parser.add_argument("--closed-tg-barrier-scale", type=float, default=1.0)
    parser.add_argument("--open-tg-barrier-scale", type=float, default=0.0)
    parser.add_argument("--fill-steps", type=int, default=3)
    parser.add_argument("--fill-step-s", type=float, default=1.0e-9)
    parser.add_argument("--transfer-step-s", type=float, default=1.0e-8)
    parser.add_argument(
        "--transfer-barrier-scales",
        default="0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
    )
    parser.add_argument("--charge-error", type=float, default=1.0)
    parser.add_argument("--dd-relative-error", type=float, default=2.0e-5)
    parser.add_argument("--dd-max-iterations", type=int, default=400)
    parser.add_argument("--bias-ramp-steps", type=int, default=20)
    parser.add_argument("--fd-terminal-mode", choices=("ohmic", "circuit"), default="ohmic")
    parser.add_argument("--fd-capacitance-f-per-cm", type=float, default=1.0e-11)
    parser.add_argument("--fd-reset-on-resistance-ohm-cm", type=float, default=1.0e3)
    parser.add_argument("--fd-reset-off-resistance-ohm-cm", type=float, default=1.0e15)
    parser.add_argument(
        "--tg-drive-mode",
        choices=("barrier_scale", "gate_contact", "gate_capacitance", "resolved_gate"),
        default="barrier_scale",
    )
    parser.add_argument("--transfer-gate-closed-bias-v", type=float, default=-1.0)
    parser.add_argument("--transfer-gate-open-bias-v", type=float, default=1.5)
    parser.add_argument("--transfer-gate-coupling-sign", type=float, default=-1.0)
    parser.add_argument(
        "--transfer-open-hold-steps",
        type=int,
        default=-1,
        help="Extra open-TG hold steps appended after the requested ramp; -1 uses a resolved_gate default.",
    )
    parser.add_argument("--sequence-mode", choices=("paired", "photo", "dark"), default="paired")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        TGFDTransientConfig(
            generation_map_npz=args.generation_map_npz,
            measured_profile=args.measured_profile,
            output_dir=args.output_dir,
            cases=parse_csv(args.cases),
            wavelength_nm=args.wavelength_nm,
            width_um=args.width_um,
            depth_um=args.depth_um,
            split_gap_um=args.split_gap_um,
            mesh_um=args.mesh_um,
            fine_mesh_um=args.fine_mesh_um,
            reverse_bias_v=args.reverse_bias_v,
            floating_diffusion_bias_v=args.floating_diffusion_bias_v,
            closed_tg_barrier_scale=args.closed_tg_barrier_scale,
            open_tg_barrier_scale=args.open_tg_barrier_scale,
            fill_steps=args.fill_steps,
            fill_step_s=args.fill_step_s,
            transfer_step_s=args.transfer_step_s,
            transfer_barrier_scales=parse_float_csv(args.transfer_barrier_scales),
            charge_error=args.charge_error,
            dd_relative_error=args.dd_relative_error,
            dd_max_iterations=args.dd_max_iterations,
            bias_ramp_steps=args.bias_ramp_steps,
            fd_terminal_mode=args.fd_terminal_mode,
            fd_capacitance_f_per_cm=args.fd_capacitance_f_per_cm,
            fd_reset_on_resistance_ohm_cm=args.fd_reset_on_resistance_ohm_cm,
            fd_reset_off_resistance_ohm_cm=args.fd_reset_off_resistance_ohm_cm,
            tg_drive_mode=args.tg_drive_mode,
            transfer_gate_closed_bias_v=args.transfer_gate_closed_bias_v,
            transfer_gate_open_bias_v=args.transfer_gate_open_bias_v,
            transfer_gate_coupling_sign=args.transfer_gate_coupling_sign,
            transfer_open_hold_steps=args.transfer_open_hold_steps,
            sequence_mode=args.sequence_mode,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
