#!/usr/bin/env python3
"""Gate report for deciding whether TCAD outputs are accuracy-LUT ready.

The checks are intentionally strict. A reference/proxy deck can pass framework
plumbing checks while still failing product-accuracy gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from measured_tcad_profile import load_measured_profile


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"
ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    details: str,
    *,
    accuracy_blocking: bool = False,
    framework_blocking: bool = False,
    evidence: Any = None,
) -> None:
    row = {
        "name": name,
        "status": status,
        "details": details,
        "accuracy_blocking": accuracy_blocking,
        "framework_blocking": framework_blocking,
    }
    if evidence is not None:
        row["evidence"] = evidence
    checks.append(row)


def names_with_token(items: list[dict[str, Any]], token: str) -> list[str]:
    token = token.lower()
    matches = []
    for item in items:
        name = str(item.get("name", "")).lower()
        role = str(item.get("role", "")).lower()
        if token in name or token in role:
            matches.append(str(item.get("name", item.get("role", token))))
    return matches


def feature_roles_from_summaries(summaries: list[dict[str, Any]]) -> set[str]:
    roles: set[str] = set()
    for summary in summaries:
        features = (
            summary.get("doping_summary", {})
            .get("feature_summary", {})
            .get("applied_features", [])
        )
        for feature in features:
            roles.add(str(feature.get("role", "")).lower())
    return roles


def feature_role_active_counts_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary in summaries:
        features = (
            summary.get("doping_summary", {})
            .get("feature_summary", {})
            .get("applied_features", [])
        )
        for feature in features:
            role = str(feature.get("role", "")).lower()
            if not role:
                continue
            counts[role] = counts.get(role, 0) + int(feature.get("active_node_count", 0) or 0)
    return counts


def feature_role_effective_counts_from_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary in summaries:
        features = (
            summary.get("doping_summary", {})
            .get("feature_summary", {})
            .get("applied_features", [])
        )
        for feature in features:
            role = str(feature.get("role", "")).lower()
            if not role:
                continue
            scale_value = feature.get("scale", feature.get("runtime_role_scale", 1.0))
            try:
                scale = float(scale_value)
            except (TypeError, ValueError):
                scale = 1.0
            active_count = int(feature.get("active_node_count", 0) or 0)
            if abs(scale) > 0.0:
                counts[role] = counts.get(role, 0) + active_count
            else:
                counts.setdefault(role, 0)
    return counts


def feature_role_scales_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, list[float]]:
    scales: dict[str, set[float]] = {}
    for summary in summaries:
        features = (
            summary.get("doping_summary", {})
            .get("feature_summary", {})
            .get("applied_features", [])
        )
        for feature in features:
            role = str(feature.get("role", "")).lower()
            if not role:
                continue
            scale_value = feature.get("scale", feature.get("runtime_role_scale", 1.0))
            try:
                scale = float(scale_value)
            except (TypeError, ValueError):
                continue
            scales.setdefault(role, set()).add(scale)
    return {role: sorted(values) for role, values in scales.items()}


def metadata_only_from_summaries(summaries: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for summary in summaries:
        result.extend(
            summary.get("doping_summary", {})
            .get("feature_summary", {})
            .get("metadata_only_features", [])
        )
    return sorted(set(str(value) for value in result))


def check_profile(profile_path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    profile = load_measured_profile(profile_path)
    add_check(
        checks,
        "profile_load",
        PASS,
        "measured_tcad_profile_v1 loaded and schema validation passed.",
        framework_blocking=True,
        evidence=str(profile.path),
    )

    calibration = profile.calibration_status
    measured = bool(calibration.get("is_measured", False))
    add_check(
        checks,
        "profile_calibration_status",
        PASS if measured else FAIL,
        "Geometry, implants, interfaces, and transport parameters are marked measured."
        if measured
        else "Profile is marked reference/proxy, so it must not be used as a product accuracy LUT.",
        accuracy_blocking=not measured,
        evidence=calibration,
    )

    if profile.data.get("reference_mode", False):
        add_check(
            checks,
            "reference_mode",
            FAIL,
            "reference_mode=true keeps public anchors and stable defaults for exploration only.",
            accuracy_blocking=True,
        )

    unmeasured_implants = [
        implant.get("name", "implant")
        for implant in profile.implants
        if not bool(implant.get("measured", False))
    ]
    add_check(
        checks,
        "implant_sources_measured",
        PASS if not unmeasured_implants else FAIL,
        "All implant profiles are measured."
        if not unmeasured_implants
        else "One or more implant profiles are unmeasured reference/analytic profiles.",
        accuracy_blocking=bool(unmeasured_implants),
        evidence=unmeasured_implants,
    )

    all_feature_items = profile.electrical_features + profile.interfaces + profile.implants
    tg = names_with_token(profile.electrical_features, "transfer_gate")
    fd = names_with_token(profile.electrical_features, "floating_diffusion")
    dti = names_with_token(all_feature_items, "dti")
    bdti = names_with_token(all_feature_items, "bdti")
    fixed_charge = [
        feature.get("name", "feature")
        for feature in profile.electrical_features + profile.interfaces
        if float(feature.get("fixed_charge_cm2", 0.0))
        or float(feature.get("effective_trap_charge_cm2", 0.0))
    ]
    dit = [
        interface.get("name", "interface")
        for interface in profile.interfaces
        if "dit_cm2_ev" in interface
    ]

    add_check(
        checks,
        "transfer_gate_proxy_present",
        PASS if tg else FAIL,
        "Transfer-gate barrier feature is present in the profile."
        if tg
        else "No transfer-gate barrier/equation feature is present.",
        framework_blocking=not tg,
        accuracy_blocking=not tg,
        evidence=tg,
    )
    add_check(
        checks,
        "floating_diffusion_proxy_present",
        PASS if fd else FAIL,
        "Floating-diffusion doping feature is present in the profile."
        if fd
        else "No floating-diffusion feature is present.",
        framework_blocking=not fd,
        accuracy_blocking=not fd,
        evidence=fd,
    )
    add_check(
        checks,
        "dti_geometry_present",
        PASS if dti else FAIL,
        "DTI/liner geometry proxy is present."
        if dti
        else "No DTI or DTI liner geometry is present.",
        framework_blocking=not dti,
        accuracy_blocking=not dti,
        evidence=dti,
    )
    add_check(
        checks,
        "bdti_geometry_present",
        PASS if bdti else FAIL,
        "BDTI geometry is present."
        if bdti
        else "No BDTI geometry is present. For BSI sensors with BDTI this remains an accuracy blocker.",
        accuracy_blocking=not bdti,
        evidence=bdti,
    )
    add_check(
        checks,
        "fixed_charge_applied",
        PASS if fixed_charge else WARN,
        "At least one fixed/effective interface charge term is executable."
        if fixed_charge
        else "No fixed/effective interface charge term is active.",
        accuracy_blocking=False,
        evidence=fixed_charge,
    )
    add_check(
        checks,
        "interface_trap_density_declared",
        INFO,
        "Interface trap density is supplied; solver implementation is checked from DEVSIM summaries."
        if dit
        else "No interface trap density was supplied.",
        accuracy_blocking=False,
        evidence=dit,
    )

    transport = profile.data.get("mobility_recombination", {})
    transport_measured = bool(
        transport.get("measured", transport.get("transport_model_measured", False))
    )
    transport_calibrated = bool(
        transport.get("calibrated", transport.get("transport_model_calibrated", False))
    )
    model_labels = {
        "transport_model": transport.get("transport_model"),
        "mu_n_model": transport.get("mu_n_model"),
        "mu_p_model": transport.get("mu_p_model"),
        "recombination_model": transport.get("recombination_model"),
        "field_mobility_model": transport.get("field_mobility_model"),
        "field_mobility_enabled": transport.get("field_mobility_enabled"),
        "electron_saturation_velocity_cm_s": transport.get("electron_saturation_velocity_cm_s"),
        "hole_saturation_velocity_cm_s": transport.get("hole_saturation_velocity_cm_s"),
        "transport_model_measured": transport_measured,
        "transport_model_calibrated": transport_calibrated,
        "tau_n_s": transport.get("tau_n_s"),
        "tau_p_s": transport.get("tau_p_s"),
    }
    constant_transport = "constant_reference" in str(model_labels).lower()
    calibrated_transport = (
        measured and transport_measured and transport_calibrated and not constant_transport
    )
    if calibrated_transport:
        transport_details = "Transport/recombination model is marked measured/calibrated."
    elif constant_transport:
        transport_details = (
            "Transport uses reference constants/lifetimes and has not been calibrated to the target sensor."
        )
    else:
        transport_details = (
            "Doping-dependent transport is configured, but it is still reference/unmeasured and lacks target-sensor calibration."
        )
    add_check(
        checks,
        "mobility_recombination_calibrated",
        PASS if calibrated_transport else FAIL,
        transport_details,
        accuracy_blocking=not calibrated_transport,
        evidence=model_labels,
    )
    return profile.data


def check_split_summaries(paths: list[Path], checks: list[dict[str, Any]], terminal_tol: float) -> None:
    if not paths:
        add_check(
            checks,
            "split_devsim_runs",
            WARN,
            "No split-PD DEVSIM summaries were supplied to the gate.",
            framework_blocking=False,
        )
        return
    summaries = [read_json(path) for path in paths]
    bad_models = [
        str(path)
        for path, summary in zip(paths, summaries)
        if summary.get("electrical_model") != "profile-ppd"
    ]
    add_check(
        checks,
        "split_devsim_profile_model",
        PASS if not bad_models else FAIL,
        "All split-PD summaries used electrical_model=profile-ppd."
        if not bad_models
        else "At least one split-PD summary did not use profile-ppd.",
        framework_blocking=bool(bad_models),
        accuracy_blocking=bool(bad_models),
        evidence=bad_models,
    )

    non_imported = [
        str(path)
        for path, summary in zip(paths, summaries)
        if summary.get("generation_source") != "imported_2d_map"
    ]
    add_check(
        checks,
        "optical_generation_map_imported",
        PASS if not non_imported else FAIL,
        "All split-PD summaries used imported Meep G(x,depth)."
        if not non_imported
        else "At least one split-PD summary did not use imported Meep G(x,depth).",
        framework_blocking=bool(non_imported),
        accuracy_blocking=bool(non_imported),
        evidence=non_imported,
    )

    transport_failures = {}
    transport_evidence = {}
    for path, summary in zip(paths, summaries):
        transport = summary.get("transport_summary", {})
        path_key = str(path)
        evidence = {
            "model": transport.get("model"),
            "electron_mobility_edge_model": transport.get("electron_mobility_edge_model"),
            "hole_mobility_edge_model": transport.get("hole_mobility_edge_model"),
            "electron_mobility_min_cm2_v_s": transport.get("electron_mobility_min_cm2_v_s"),
            "electron_mobility_max_cm2_v_s": transport.get("electron_mobility_max_cm2_v_s"),
            "hole_mobility_min_cm2_v_s": transport.get("hole_mobility_min_cm2_v_s"),
            "hole_mobility_max_cm2_v_s": transport.get("hole_mobility_max_cm2_v_s"),
            "tau_n_min_s": transport.get("tau_n_min_s"),
            "tau_n_max_s": transport.get("tau_n_max_s"),
            "tau_p_min_s": transport.get("tau_p_min_s"),
            "tau_p_max_s": transport.get("tau_p_max_s"),
            "field_mobility_enabled": transport.get("field_mobility_enabled"),
            "field_mobility_model": transport.get("field_mobility_model"),
            "electron_mobility_effective_edge_min_cm2_v_s": transport.get(
                "electron_mobility_effective_edge_min_cm2_v_s"
            ),
            "electron_mobility_effective_edge_max_cm2_v_s": transport.get(
                "electron_mobility_effective_edge_max_cm2_v_s"
            ),
            "hole_mobility_effective_edge_min_cm2_v_s": transport.get(
                "hole_mobility_effective_edge_min_cm2_v_s"
            ),
            "hole_mobility_effective_edge_max_cm2_v_s": transport.get(
                "hole_mobility_effective_edge_max_cm2_v_s"
            ),
        }
        transport_evidence[path_key] = evidence
        failures = []
        if not transport:
            failures.append("missing transport_summary")
        if "constant" in str(transport.get("model", "")).lower():
            failures.append("transport model is constant")
        for low_key, high_key in (
            ("electron_mobility_min_cm2_v_s", "electron_mobility_max_cm2_v_s"),
            ("hole_mobility_min_cm2_v_s", "hole_mobility_max_cm2_v_s"),
            ("tau_n_min_s", "tau_n_max_s"),
            ("tau_p_min_s", "tau_p_max_s"),
        ):
            low = transport.get(low_key)
            high = transport.get(high_key)
            try:
                low_f = float(low)
                high_f = float(high)
            except (TypeError, ValueError):
                failures.append(f"{low_key}/{high_key} are not finite")
                continue
            if not (math.isfinite(low_f) and math.isfinite(high_f) and high_f > 0.0):
                failures.append(f"{low_key}/{high_key} are invalid")
        if failures:
            transport_failures[path_key] = failures
    add_check(
        checks,
        "transport_models_applied",
        PASS if not transport_failures else FAIL,
        "DEVSIM split summaries used doping-dependent mobility/lifetime models in the solver path."
        if not transport_failures
        else "At least one supplied split summary lacks executable doping-dependent transport evidence.",
        framework_blocking=bool(transport_failures),
        accuracy_blocking=bool(transport_failures),
        evidence={
            "summary_evidence": transport_evidence,
            "failures": transport_failures,
        },
    )

    field_transport_failures = {}
    field_transport_evidence = {}
    for path, summary in zip(paths, summaries):
        transport = summary.get("transport_summary", {})
        path_key = str(path)
        enabled = bool(transport.get("field_mobility_enabled", False))
        evidence = {
            "field_mobility_enabled": enabled,
            "field_mobility_model": transport.get("field_mobility_model"),
            "electron_mobility_edge_model": transport.get("electron_mobility_edge_model"),
            "hole_mobility_edge_model": transport.get("hole_mobility_edge_model"),
            "electron_effective_edge_min": transport.get(
                "electron_mobility_effective_edge_min_cm2_v_s"
            ),
            "electron_effective_edge_max": transport.get(
                "electron_mobility_effective_edge_max_cm2_v_s"
            ),
            "hole_effective_edge_min": transport.get(
                "hole_mobility_effective_edge_min_cm2_v_s"
            ),
            "hole_effective_edge_max": transport.get(
                "hole_mobility_effective_edge_max_cm2_v_s"
            ),
        }
        field_transport_evidence[path_key] = evidence
        failures = []
        if not enabled:
            failures.append("field mobility is not enabled")
        for low_key, high_key in (
            ("electron_mobility_effective_edge_min_cm2_v_s", "electron_mobility_effective_edge_max_cm2_v_s"),
            ("hole_mobility_effective_edge_min_cm2_v_s", "hole_mobility_effective_edge_max_cm2_v_s"),
        ):
            try:
                low = float(transport.get(low_key))
                high = float(transport.get(high_key))
            except (TypeError, ValueError):
                failures.append(f"{low_key}/{high_key} are missing")
                continue
            if not (math.isfinite(low) and math.isfinite(high) and high > 0.0):
                failures.append(f"{low_key}/{high_key} are invalid")
        if failures:
            field_transport_failures[path_key] = failures
    add_check(
        checks,
        "field_dependent_mobility_applied",
        PASS if not field_transport_failures else FAIL,
        "DEVSIM split summaries include field-dependent effective edge mobility for velocity saturation."
        if not field_transport_failures
        else "At least one split summary lacks field-dependent mobility evidence.",
        framework_blocking=bool(field_transport_failures),
        accuracy_blocking=bool(field_transport_failures),
        evidence={
            "summary_evidence": field_transport_evidence,
            "failures": field_transport_failures,
        },
    )

    balances = {
        str(path): float(summary.get("terminal_current_balance_illuminated_a_per_cm", 0.0))
        for path, summary in zip(paths, summaries)
    }
    balance_fail = {
        path: value for path, value in balances.items() if abs(value) > terminal_tol
    }
    add_check(
        checks,
        "terminal_current_balance",
        PASS if not balance_fail else FAIL,
        f"Illuminated terminal-current balance is within {terminal_tol:g} A/cm."
        if not balance_fail
        else "Illuminated terminal-current balance exceeds tolerance.",
        framework_blocking=bool(balance_fail),
        accuracy_blocking=bool(balance_fail),
        evidence=balances,
    )

    roles = feature_roles_from_summaries(summaries)
    role_active_counts = feature_role_active_counts_from_summaries(summaries)
    role_effective_counts = feature_role_effective_counts_from_summaries(summaries)
    role_scales = feature_role_scales_from_summaries(summaries)
    required_roles = {
        "transfer_gate_barrier",
        "floating_diffusion",
        "front_si_oxide_fixed_charge",
        "bdti_liner",
    }
    missing_roles = sorted(
        role for role in required_roles if role not in roles or role_active_counts.get(role, 0) <= 0
    )
    add_check(
        checks,
        "profile_features_applied_to_netdoping",
        PASS if not missing_roles else FAIL,
        "Executable TG, FD, BDTI, and fixed-charge feature masks are present; nonzero-effect counts show runtime-disabled roles such as FD scale=0 for PD-only response sweeps."
        if not missing_roles
        else "Some executable profile feature roles were not present on the DEVSIM mesh.",
        framework_blocking=bool(missing_roles),
        accuracy_blocking=bool(missing_roles),
        evidence={
            "roles": sorted(roles),
            "mask_node_counts": role_active_counts,
            "nonzero_effect_node_counts": role_effective_counts,
            "runtime_scales": role_scales,
            "disabled_by_scale": sorted(
                role
                for role in roles
                if role_active_counts.get(role, 0) > 0 and role_effective_counts.get(role, 0) == 0
            ),
            "missing": missing_roles,
        },
    )

    metadata_only = metadata_only_from_summaries(summaries)
    applied_trap_names = sorted(
        {
            str(trap.get("name", "interface_trap"))
            for summary in summaries
            for trap in summary.get("interface_trap_summary", {}).get(
                "applied_interface_traps", []
            )
        }
    )
    metadata_traps = sorted(
        {
            str(name)
            for name in metadata_only
            if "trap" in str(name).lower() or "interface" in str(name).lower()
        }
    )
    trap_model_evidence = {}
    trap_model_failures = {}
    trap_required = bool(applied_trap_names or metadata_traps)
    for path, summary in zip(paths, summaries):
        trap_summary = summary.get("interface_trap_summary", {})
        traps = trap_summary.get("applied_interface_traps", [])
        density_max = float(trap_summary.get("trap_density_max_cm3", 0.0) or 0.0)
        recomb_max = float(trap_summary.get("recombination_coeff_max_s1", 0.0) or 0.0)
        path_key = str(path)
        trap_model_evidence[path_key] = {
            "applied_interface_traps": [
                str(trap.get("name", "interface_trap")) for trap in traps
            ],
            "trap_density_max_cm3": density_max,
            "recombination_coeff_max_s1": recomb_max,
        }
        if trap_required:
            failures = []
            if not traps:
                failures.append("no applied interface trap entries")
            if not math.isfinite(density_max) or density_max <= 0.0:
                failures.append("trap_density_max_cm3 is not positive")
            if not math.isfinite(recomb_max) or recomb_max <= 0.0:
                failures.append("recombination_coeff_max_s1 is not positive")
            if failures:
                trap_model_failures[path_key] = failures
    trap_model_pass = trap_required and not trap_model_failures
    add_check(
        checks,
        "interface_trap_occupancy_equations",
        PASS if trap_model_pass else (FAIL if trap_required else INFO),
        "DEVSIM summaries include potential-dependent interface-trap charge and SRH sheet recombination proxies."
        if trap_model_pass
        else (
            "At least one supplied DEVSIM summary lacks nonzero interface-trap charge/recombination evidence."
            if trap_required
            else "No interface-trap implementation was required by the supplied summaries."
        ),
        accuracy_blocking=bool(trap_required and not trap_model_pass),
        framework_blocking=bool(trap_required and not trap_model_pass),
        evidence={
            "applied_interface_traps": applied_trap_names,
            "metadata_only_interface_traps": metadata_traps,
            "summary_evidence": trap_model_evidence,
            "failures": trap_model_failures,
        },
    )
    add_check(
        checks,
        "metadata_only_electrical_terms",
        WARN if metadata_only else PASS,
        "Some electrical terms remain metadata-only in the DEVSIM run."
        if metadata_only
        else "No metadata-only electrical terms were reported by the supplied split summaries.",
        accuracy_blocking=False,
        evidence=metadata_only,
    )


def check_gmsh_summaries(paths: list[Path], checks: list[dict[str, Any]]) -> None:
    if not paths:
        add_check(checks, "gmsh_devsim_import", INFO, "No Gmsh import summaries were supplied.")
        return
    summaries = [read_json(path) for path in paths]
    bad_contacts = {}
    for path, summary in zip(paths, summaries):
        contacts = set(summary.get("contacts", []))
        missing = sorted({"anode", "cathode_left", "cathode_right"} - contacts)
        if missing:
            bad_contacts[str(path)] = missing
    add_check(
        checks,
        "gmsh_contacts_imported",
        PASS if not bad_contacts else FAIL,
        "Gmsh import summaries contain silicon and all three contacts."
        if not bad_contacts
        else "One or more Gmsh import summaries are missing required contacts.",
        framework_blocking=bool(bad_contacts),
        evidence=bad_contacts,
    )
    dimensions = sorted({int(summary.get("dimension", 0)) for summary in summaries})
    add_check(
        checks,
        "gmsh_dimensions_smoked",
        PASS if {2, 3}.issubset(set(dimensions)) else WARN,
        "Both 2D and 3D Gmsh import smoke solves were supplied."
        if {2, 3}.issubset(set(dimensions))
        else "Only a subset of 2D/3D Gmsh import smoke solves was supplied.",
        framework_blocking=False,
        evidence=dimensions,
    )


def check_resolved_dti_oxide(
    mesh_metadata_path: Path | None,
    split_summary_path: Path | None,
    checks: list[dict[str, Any]],
    terminal_tol: float,
) -> None:
    if mesh_metadata_path is None and split_summary_path is None:
        add_check(
            checks,
            "resolved_dti_oxide_solver_path",
            INFO,
            "No resolved DTI/BDTI oxide mesh or DEVSIM run was supplied.",
            framework_blocking=False,
            accuracy_blocking=False,
        )
        return

    metadata: dict[str, Any] = {}
    mesh_entry: dict[str, Any] = {}
    mesh_ok = False
    mesh_failures: list[str] = []
    if mesh_metadata_path is None or not mesh_metadata_path.exists():
        mesh_failures.append("missing mesh metadata")
    else:
        metadata = read_json(mesh_metadata_path)
        meshes = metadata.get("meshes", [])
        for item in meshes:
            if int(item.get("dimension", 0) or 0) != 2:
                continue
            if item.get("resolved_dti_oxide"):
                mesh_entry = item
                break
        if not mesh_entry:
            mesh_failures.append("no 2D mesh entry with resolved_dti_oxide")
        else:
            if mesh_entry.get("oxide_region_physical") != "oxide":
                mesh_failures.append("oxide physical region missing")
            if "silicon_oxide_interface" not in set(mesh_entry.get("interfaces", [])):
                mesh_failures.append("silicon_oxide_interface physical group missing")
            contacts = set(str(contact) for contact in mesh_entry.get("contacts", []))
            missing_contacts = sorted({"anode", "cathode_left", "cathode_right"} - contacts)
            if missing_contacts:
                mesh_failures.append(f"missing contacts: {','.join(missing_contacts)}")
            line_counts = mesh_entry.get("line_counts", {})
            if int(line_counts.get("silicon_oxide_interface", 0) or 0) <= 0:
                mesh_failures.append("no interface line count")
            if int(mesh_entry.get("surface_counts", {}).get("oxide", 0) or 0) <= 0:
                mesh_failures.append("no oxide surface count")
    mesh_ok = not mesh_failures

    split_ok = False
    split_failures: list[str] = []
    split_evidence: dict[str, Any] = {}
    if split_summary_path is None or not split_summary_path.exists():
        split_failures.append("missing resolved-DTI DEVSIM split summary")
    else:
        summary = read_json(split_summary_path)
        generation_error = summary.get("generation_integral_summary", {}).get("mesh_to_target_rel_error")
        terminal_balance = summary.get("terminal_current_balance_illuminated_a_per_cm")
        split_evidence = {
            "path": str(split_summary_path),
            "mesh_source": summary.get("mesh_source"),
            "gmsh_mesh": summary.get("gmsh_mesh"),
            "electrical_model": summary.get("electrical_model"),
            "generation_source": summary.get("generation_source"),
            "node_count": summary.get("node_count"),
            "generation_integral_rel_error": generation_error,
            "terminal_current_balance_illuminated_a_per_cm": terminal_balance,
            "left_photo_delta_a_per_cm": summary.get("left_photo_delta_a_per_cm"),
            "right_photo_delta_a_per_cm": summary.get("right_photo_delta_a_per_cm"),
            "photo_split_phase_x_proxy": summary.get("photo_split_phase_x_proxy"),
        }
        if summary.get("mesh_source") != "gmsh":
            split_failures.append("split summary did not use gmsh mesh")
        if summary.get("electrical_model") != "profile-ppd":
            split_failures.append("split summary did not use profile-ppd")
        if summary.get("generation_source") != "imported_2d_map":
            split_failures.append("split summary did not use imported FDTD generation map")
        if int(summary.get("node_count", 0) or 0) <= 0:
            split_failures.append("split summary has no nodes")
        try:
            if abs(float(terminal_balance)) > terminal_tol:
                split_failures.append("terminal balance exceeds tolerance")
        except (TypeError, ValueError):
            split_failures.append("terminal balance is not finite")
        try:
            if abs(float(generation_error)) > 1.0e-6:
                split_failures.append("generation map integral was not preserved on resolved DTI mesh")
        except (TypeError, ValueError):
            split_failures.append("generation integral error is not finite")
    split_ok = not split_failures

    ok = mesh_ok and split_ok
    add_check(
        checks,
        "resolved_dti_oxide_solver_path",
        PASS if ok else FAIL,
        "Resolved side DTI/BDTI oxide regions and Si/oxide interface were exported to Gmsh and solved by DEVSIM."
        if ok
        else "Resolved DTI/BDTI oxide mesh or DEVSIM run is missing or malformed.",
        framework_blocking=not ok,
        accuracy_blocking=False,
        evidence={
            "mesh_metadata": str(mesh_metadata_path) if mesh_metadata_path else None,
            "mesh_ok": mesh_ok,
            "mesh_failures": mesh_failures,
            "resolved_dti_oxide": mesh_entry.get("resolved_dti_oxide", []),
            "line_counts": mesh_entry.get("line_counts", {}),
            "surface_counts": mesh_entry.get("surface_counts", {}),
            "split_ok": split_ok,
            "split_failures": split_failures,
            "split_evidence": split_evidence,
        },
    )


def check_convergence_report(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "fdtd_convergence_report",
            FAIL,
            "No FDTD convergence report was supplied.",
            accuracy_blocking=True,
        )
        return
    report = read_json(path)
    passed = bool(report.get("passed", False))
    full_pass = report.get("full_numerical_convergence_pass")
    negative_count = int(report.get("negative_signed_flux_count", 0))
    fail_on_negative = bool(report.get("fail_on_negative_signed_flux", False))
    status = PASS if passed and negative_count == 0 and full_pass is not False else FAIL
    details = "FDTD convergence passed with no negative signed-flux diagnostics."
    if not passed:
        details = "FDTD convergence report did not pass."
    elif negative_count:
        details = (
            "FDTD convergence numerically passed, but negative signed-flux diagnostics "
            "remain and must be resolved for accuracy use."
        )
    elif full_pass is False:
        details = (
            "FDTD grid/convergence report is partial; full resolution, time, and PML "
            "coverage has not all passed for accuracy use."
        )
    add_check(
        checks,
        "fdtd_convergence_report",
        status,
        details,
        accuracy_blocking=status == FAIL,
        evidence={
            "path": str(path),
            "passed": passed,
            "full_numerical_convergence_pass": full_pass,
            "spatial_convergence_pass": report.get("spatial_convergence_pass"),
            "time_convergence_pass": report.get("time_convergence_pass"),
            "pml_convergence_pass": report.get("pml_convergence_pass"),
            "varied_axes": report.get("varied_axes"),
            "unproven_axes": report.get("unproven_axes"),
            "negative_signed_flux_count": negative_count,
            "fail_on_negative_signed_flux": fail_on_negative,
        },
    )


def check_crosstalk_xsection_convergence(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if not path:
        add_check(
            checks,
            "fdtd_crosstalk_xsection_convergence",
            WARN,
            "No high-resolution 2D crosstalk x-section convergence report was supplied.",
            accuracy_blocking=False,
        )
        return
    data = read_json(path)
    status = str(data.get("status", "")).upper()
    check_rows = data.get("checks", [])
    failing_checks = [
        check
        for check in check_rows
        if str(check.get("status", "")).upper() not in {PASS, INFO}
    ]
    pass_rows = [check for check in check_rows if str(check.get("status", "")).upper() == PASS]
    resolution_values = [
        float(check.get("resolution_px_per_um"))
        for check in check_rows
        if check.get("resolution_px_per_um") not in ("", None)
    ]
    min_resolution = min(resolution_values) if resolution_values else None
    max_resolution = max(resolution_values) if resolution_values else None
    passed = status == PASS and not failing_checks
    add_check(
        checks,
        "fdtd_crosstalk_xsection_convergence",
        PASS if passed else FAIL,
        "High-resolution 2D FDTD crosstalk x-section convergence passed for configured split/OCL cases."
        if passed
        else "High-resolution 2D FDTD crosstalk x-section convergence failed or has failing subchecks.",
        accuracy_blocking=not passed,
        evidence={
            "path": str(path),
            "status": status,
            "pass_check_count": len(pass_rows),
            "fail_check_count": len(failing_checks),
            "min_resolution_px_per_um": min_resolution,
            "max_resolution_px_per_um": max_resolution,
            "failing_checks": failing_checks,
        },
    )


def check_native_response_convergence(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "native_devsim_response_convergence",
            WARN,
            "No native DEVSIM direct-response mesh convergence report was supplied.",
            accuracy_blocking=False,
        )
        return
    report = read_json(path)
    passed = bool(report.get("passed", False))
    add_check(
        checks,
        "native_devsim_response_convergence",
        PASS if passed else FAIL,
        "Native DEVSIM direct response passed mesh-refinement convergence."
        if passed
        else "Native DEVSIM direct response failed mesh-refinement convergence.",
        accuracy_blocking=not passed,
        evidence={
            "path": str(path),
            "passed": passed,
            "max_total_response_rel_delta_to_reference": report.get(
                "max_total_response_rel_delta_to_reference"
            ),
            "max_split_phase_abs_delta_to_reference": report.get(
                "max_split_phase_abs_delta_to_reference"
            ),
            "total_rel_tol": report.get("total_rel_tol"),
            "split_abs_tol": report.get("split_abs_tol"),
            "reference_level": report.get("reference_level"),
        },
    )


def check_signed_field_symmetry_validation(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "signed_field_symmetry_validation",
            WARN,
            "No direct negative-CRA signed-field symmetry validation report was supplied.",
            accuracy_blocking=False,
            framework_blocking=False,
        )
        return
    report = read_json(path)
    passed = bool(report.get("validation_pass", False))
    case_count = int(report.get("case_count", 0) or 0)
    max_total_rel = report.get("max_total_response_rel_error")
    max_split_abs = report.get("max_split_phase_abs_error")
    total_threshold = report.get("max_total_rel_error_threshold")
    split_threshold = report.get("max_split_abs_error_threshold")
    rows = report.get("rows", [])
    row_failures = [
        row for row in rows if not bool(row.get("case_pass", False))
    ]
    ok = (
        report.get("schema") == "signed_field_symmetry_validation_v1"
        and passed
        and case_count > 0
        and not row_failures
    )
    add_check(
        checks,
        "signed_field_symmetry_validation",
        PASS if ok else FAIL,
        "Signed field LUT was validated against direct negative-CRA FDTD/native-DEVSIM anchors."
        if ok
        else "Signed field LUT failed or lacks direct negative-CRA validation.",
        framework_blocking=not ok,
        accuracy_blocking=not ok,
        evidence={
            "path": str(path),
            "validation_pass": passed,
            "case_count": case_count,
            "max_total_response_rel_error": max_total_rel,
            "max_split_phase_abs_error": max_split_abs,
            "max_total_rel_error_threshold": total_threshold,
            "max_split_abs_error_threshold": split_threshold,
            "failed_rows": row_failures,
        },
    )


def check_camera_lut_spectral_coverage(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "camera_lut_spectral_coverage",
            WARN,
            "No camera-system field-LUT wavelength coverage report was supplied.",
            accuracy_blocking=False,
            framework_blocking=False,
        )
        return
    report = read_json(path)
    coverage_pass = bool(report.get("coverage_pass", False))
    missing = report.get("missing_required_wavelength_nm", [])
    available = report.get("available_wavelength_nm", [])
    required = report.get("required_wavelength_nm", [])
    uniform_grid = bool(report.get("uniform_field_grid_per_wavelength", False))
    signed_grid = bool(report.get("signed_field_grid", False))
    ok = (
        report.get("schema") == "camera_lut_spectral_coverage_v1"
        and coverage_pass
        and not missing
        and uniform_grid
        and signed_grid
    )
    add_check(
        checks,
        "camera_lut_spectral_coverage",
        PASS if ok else FAIL,
        "Camera-system field LUT covers the required RGB wavelength anchors on a uniform signed field grid."
        if ok
        else "Camera-system field LUT lacks required RGB wavelength anchors or a uniform signed field grid.",
        framework_blocking=False,
        accuracy_blocking=not ok,
        evidence={
            "path": str(path),
            "coverage_pass": coverage_pass,
            "required_wavelength_nm": required,
            "available_wavelength_nm": available,
            "missing_required_wavelength_nm": missing,
            "uniform_field_grid_per_wavelength": uniform_grid,
            "signed_field_grid": signed_grid,
            "row_count": report.get("row_count"),
            "field_grid_count": report.get("field_grid_count"),
            "field_z_grid_count": report.get("field_z_grid_count"),
            "optical_stack_evidence_pass": report.get("optical_stack_evidence_pass"),
        },
    )


def check_camera_system_field_lut(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "camera_system_field_lut_direct_z_diag_coverage",
            WARN,
            "No dense camera-system field LUT was supplied for direct z/diagonal anchor inspection.",
            accuracy_blocking=False,
            framework_blocking=False,
        )
        return

    report = read_json(path)
    rows = report.get("rows", [])
    axis_model = report.get("field_axis_model", {})
    methods = set(str(item) for item in axis_model.get("interpolation_methods", []))
    methods.update(str(row.get("interpolation_method", "")) for row in rows)
    by_wavelength = axis_model.get("interpolation_methods_by_wavelength", {})
    for method_counts in by_wavelength.values():
        if isinstance(method_counts, dict):
            methods.update(str(method) for method in method_counts.keys())

    row_wavelengths = sorted(
        {
            int(round(float(row.get("wavelength_nm"))))
            for row in rows
            if isinstance(row.get("wavelength_nm"), (int, float))
            and math.isfinite(float(row.get("wavelength_nm")))
        }
    )
    method_wavelengths = sorted(
        {
            int(round(float(wavelength)))
            for wavelength in by_wavelength.keys()
            if str(wavelength).replace(".", "", 1).isdigit()
        }
    )
    required_wavelengths = {450, 550, 650}
    available_wavelengths = set(row_wavelengths) | set(method_wavelengths)

    source_anchor_cases = {str(case) for case in report.get("source_anchor_cases", [])}
    source_z_cases = {str(case) for case in report.get("source_z_axis_anchor_cases", [])}
    source_diag_cases = {str(case) for case in report.get("source_diagonal_anchor_cases", [])}
    blockers = {str(blocker) for blocker in report.get("accuracy_blockers", [])}

    direct_method = "signed_separable_xz_piecewise_linear_3_anchor_x_piecewise_linear_3_anchor_z_diag2"
    fallback_methods = [
        method
        for method in methods
        if "fallback" in method.lower() or "radial" in method.lower()
    ]
    model_name = str(axis_model.get("model_name", ""))
    row_count = len(rows)
    expected_min_rows = 3 * int(report.get("field_grid_count", 0) or 0) * int(
        report.get("field_z_grid_count", 0) or 0
    )
    if expected_min_rows == 0:
        expected_min_rows = 1

    failures: list[str] = []
    if report.get("schema") != "camera_system_field_lut_v1":
        failures.append("schema is not camera_system_field_lut_v1")
    if row_count < expected_min_rows:
        failures.append("row count is smaller than the declared field grid")
    if not required_wavelengths.issubset(available_wavelengths):
        failures.append("missing required RGB wavelengths")
    if direct_method not in methods:
        failures.append("direct x/z/diagonal interpolation method is absent")
    if fallback_methods:
        failures.append("fallback/radial interpolation method is still present")
    if "diagonal_native_anchor_correction" not in model_name:
        failures.append("field axis model does not include diagonal native-anchor correction")
    if not {"center", "cra10x", "edge20x"}.issubset(source_anchor_cases):
        failures.append("x-axis native anchor cases are incomplete")
    if not {"center", "cra10z", "edge20z"}.issubset(source_z_cases):
        failures.append("z-axis native anchor cases are incomplete")
    if not {"cra10x10z", "edge20x20z"}.issubset(source_diag_cases):
        failures.append("diagonal native anchor cases are incomplete")
    if "spectral_z_diag_not_directly_anchored" in blockers:
        failures.append("stale spectral z/diagonal fallback blocker is still present")

    ok = not failures
    add_check(
        checks,
        "camera_system_field_lut_direct_z_diag_coverage",
        PASS if ok else FAIL,
        "Dense field LUT uses direct RGB x-axis, z-axis, and diagonal native-DEVSIM anchors without radial fallback."
        if ok
        else "Dense field LUT does not yet prove direct RGB x/z/diagonal native-anchor coverage.",
        framework_blocking=False,
        accuracy_blocking=not ok,
        evidence={
            "path": str(path),
            "row_count": row_count,
            "field_grid_count": report.get("field_grid_count"),
            "field_z_grid_count": report.get("field_z_grid_count"),
            "available_wavelength_nm": row_wavelengths,
            "interpolation_methods": sorted(method for method in methods if method),
            "source_anchor_cases": sorted(source_anchor_cases),
            "source_z_axis_anchor_cases": sorted(source_z_cases),
            "source_diagonal_anchor_cases": sorted(source_diag_cases),
            "accuracy_blockers": sorted(blockers),
            "failures": failures,
        },
    )


def expand_split_summary_paths(paths: list[Path], dirs: list[Path]) -> list[Path]:
    expanded: list[Path] = list(paths)
    for directory in dirs:
        case_dir = directory / "cases"
        if case_dir.exists():
            expanded.extend(sorted(case_dir.glob("*/summary.json")))
        else:
            expanded.extend(sorted(directory.glob("*/summary.json")))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in expanded:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def check_tg_fd_diagnostic(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "tg_fd_terminal_diagnostic",
            INFO,
            "No TG/FD terminal diagnostic report was supplied.",
            accuracy_blocking=False,
        )
        return
    report = read_json(path)
    rows = report.get("rows", [])
    balance = report.get("terminal_balance_max_abs_a_per_cm")
    fd_max = report.get("max_fd_abs_fraction")
    ok = (
        report.get("schema") == "devsim_tg_fd_transfer_sweep_2d_v1"
        and bool(rows)
        and isinstance(balance, (int, float))
        and math.isfinite(float(balance))
        and float(balance) <= 1.0e-9
        and isinstance(fd_max, (int, float))
        and math.isfinite(float(fd_max))
    )
    add_check(
        checks,
        "tg_fd_terminal_diagnostic",
        PASS if ok else FAIL,
        "TG/FD diagnostic has a real floating-diffusion terminal, finite FD response rows, and terminal balance closure."
        if ok
        else "TG/FD diagnostic is missing, malformed, or numerically imbalanced.",
        framework_blocking=not ok,
        accuracy_blocking=False,
        evidence={
            "path": str(path),
            "schema": report.get("schema"),
            "row_count": report.get("row_count"),
            "max_fd_abs_fraction": fd_max,
            "terminal_balance_max_abs_a_per_cm": balance,
            "product_accuracy_ready": report.get("product_accuracy_ready"),
        },
    )


def check_tg_fd_transient(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "tg_fd_transient_diagnostic",
            INFO,
            "No TG/FD transient diagnostic report was supplied.",
            accuracy_blocking=False,
        )
        return
    def load_case_report(case: dict[str, Any]) -> dict[str, Any]:
        report_path_text = str(case.get("report_json", ""))
        if not report_path_text:
            return case
        report_path = Path(report_path_text)
        candidates = [report_path]
        if not report_path.is_absolute():
            candidates.append(path.parent / report_path)
            candidates.append(Path.cwd() / report_path)
        for candidate in candidates:
            if candidate.exists():
                try:
                    return read_json(candidate)
                except (OSError, json.JSONDecodeError):
                    return case
        return case

    def inventory_metrics(case_report: dict[str, Any]) -> dict[str, Any]:
        inventory = case_report.get("carrier_inventory", {})
        transfer = case_report.get("transfer_integrals", {})
        return {
            "case": case_report.get("case"),
            "signal_mode": case_report.get("signal_mode", case_report.get("sequence_mode")),
            "method": inventory.get("method"),
            "pd_total_excess_electron_fraction_remaining": inventory.get(
                "pd_total_excess_electron_fraction_remaining"
            ),
            "floating_diffusion_excess_electron_gain_per_cm": inventory.get(
                "floating_diffusion_excess_electron_gain_per_cm"
            ),
            "floating_diffusion_terminal_electrons_per_cm": transfer.get(
                "floating_diffusion_electrons_per_cm"
            ),
            "floating_diffusion_terminal_abs_electrons_per_cm": transfer.get(
                "floating_diffusion_abs_electrons_per_cm"
            ),
            "floating_diffusion_terminal_abs_fraction_of_fd_plus_cathodes": transfer.get(
                "floating_diffusion_electron_abs_fraction_of_fd_plus_cathodes"
            ),
            "floating_diffusion_last_abs_current_to_peak": transfer.get(
                "floating_diffusion_last_abs_current_to_peak"
            ),
        }

    report = read_json(path)
    schema = report.get("schema")
    if schema == "devsim_tg_fd_transient_sweep_2d_v1":
        cases = report.get("cases", [])
        case_reports = [load_case_report(case) for case in cases]
        fd_fractions = [
            case.get("transfer_integrals", {}).get(
                "floating_diffusion_electron_abs_fraction_of_fd_plus_cathodes"
            )
            for case in case_reports
        ]
        row_count = int(report.get("row_count", 0) or 0)
        case_count = int(report.get("case_count", 0) or 0)
    elif schema == "devsim_tg_fd_transient_2d_v1":
        case_reports = [report]
        fd_fractions = [
            report.get("transfer_integrals", {}).get(
                "floating_diffusion_electron_abs_fraction_of_fd_plus_cathodes"
            )
        ]
        row_count = int(report.get("row_count", 0) or 0)
        case_count = 1
    else:
        case_reports = []
        fd_fractions = []
        row_count = 0
        case_count = 0
    finite_fd = [
        float(value)
        for value in fd_fractions
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    inventory = [inventory_metrics(case_report) for case_report in case_reports]
    inventory_present = bool(inventory) and all(item.get("method") for item in inventory)
    signal_modes = [str(item.get("signal_mode", "")) for item in inventory]
    ok = (
        schema in {"devsim_tg_fd_transient_sweep_2d_v1", "devsim_tg_fd_transient_2d_v1"}
        and row_count > 0
        and case_count > 0
        and len(finite_fd) == len(fd_fractions)
        and all(value >= 0.0 for value in finite_fd)
        and inventory_present
        and all(mode == "photo_minus_dark" for mode in signal_modes)
    )
    add_check(
        checks,
        "tg_fd_transient_diagnostic",
        PASS if ok else FAIL,
        "TG/FD diagnostic uses paired photo-minus-dark DEVSIM transient time stepping and reports carrier inventory."
        if ok
        else "TG/FD transient diagnostic is missing paired photo-minus-dark carrier-inventory evidence.",
        framework_blocking=not ok,
        accuracy_blocking=False,
        evidence={
            "path": str(path),
            "schema": schema,
            "case_count": case_count,
            "row_count": row_count,
            "fd_abs_fractions": finite_fd,
            "signal_modes": signal_modes,
            "carrier_inventory": inventory,
            "terminal_balance_max_abs_a_per_cm": report.get("terminal_balance_max_abs_a_per_cm"),
            "product_accuracy_ready": report.get("product_accuracy_ready"),
        },
    )
    finite_inventory = [
        item
        for item in inventory
        if isinstance(item.get("pd_total_excess_electron_fraction_remaining"), (int, float))
        and math.isfinite(float(item["pd_total_excess_electron_fraction_remaining"]))
        and isinstance(item.get("floating_diffusion_terminal_abs_electrons_per_cm"), (int, float))
        and math.isfinite(float(item["floating_diffusion_terminal_abs_electrons_per_cm"]))
    ]
    transfer_effect_ok = (
        len(finite_inventory) == case_count
        and case_count > 0
        and all(float(item["pd_total_excess_electron_fraction_remaining"]) < 0.95 for item in finite_inventory)
        and all(float(item["floating_diffusion_terminal_abs_electrons_per_cm"]) > 0.0 for item in finite_inventory)
    )
    add_check(
        checks,
        "tg_fd_transfer_inventory_effect",
        PASS if transfer_effect_ok else FAIL,
        "Carrier inventory shows photo-generated electrons leaving the PD region and FD terminal integration collects positive electron charge."
        if transfer_effect_ok
        else "Carrier inventory and FD terminal integration do not yet show a physically useful PD-to-FD transfer effect.",
        framework_blocking=not transfer_effect_ok,
        accuracy_blocking=False,
        evidence={
            "pd_remaining_pass_threshold": "< 0.95",
            "fd_terminal_collection_pass_threshold": "> 0 abs(electrons/cm)",
            "fd_inventory_gain_note": "A voltage-clamped FD ohmic terminal can collect electrons while the local FD silicon inventory decreases; FD volume gain is evidence only, not the pass criterion.",
            "carrier_inventory": inventory,
        },
    )
    finite_tail = [
        float(item["floating_diffusion_last_abs_current_to_peak"])
        for item in inventory
        if isinstance(item.get("floating_diffusion_last_abs_current_to_peak"), (int, float))
        and math.isfinite(float(item["floating_diffusion_last_abs_current_to_peak"]))
    ]
    settling_ok = len(finite_tail) == case_count and case_count > 0 and all(value <= 0.10 for value in finite_tail)
    add_check(
        checks,
        "tg_fd_transfer_settling",
        PASS if settling_ok else FAIL,
        "FD terminal photo-minus-dark current decays below 10% of its peak by the end of the transfer pulse."
        if settling_ok
        else "FD terminal photo-minus-dark current has not settled by the end of the transfer pulse.",
        framework_blocking=not settling_ok,
        accuracy_blocking=False,
        evidence={
            "last_abs_current_to_peak_pass_threshold": "<= 0.10",
            "last_abs_current_to_peak": finite_tail,
            "carrier_inventory": inventory,
        },
    )


def check_calibration(
    result_path: Path | None,
    targets_path: Path | None,
    target_report_path: Path | None,
    checks: list[dict[str, Any]],
) -> None:
    if result_path is None:
        add_check(
            checks,
            "calibration_result",
            FAIL,
            "No calibration result was supplied.",
            accuracy_blocking=True,
        )
    else:
        result = read_json(result_path)
        success = bool(result.get("success", False))
        add_check(
            checks,
            "calibration_result",
            PASS if success else FAIL,
            "Calibration optimizer reported success."
            if success
            else "Calibration optimizer did not report success.",
            accuracy_blocking=not success,
            evidence={
                "path": str(result_path),
                "cost": result.get("cost"),
                "best_parameters": result.get("best_parameters"),
            },
        )

    if targets_path is None:
        add_check(
            checks,
            "calibration_targets_measured",
            FAIL,
            "No calibration target CSV was supplied.",
            accuracy_blocking=True,
        )
        return
    with targets_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    synthetic = [
        row.get("case", f"row{index}")
        for index, row in enumerate(rows)
        if str(row.get("target_source", "")).lower() != "measured"
    ]
    add_check(
        checks,
        "calibration_targets_measured",
        PASS if rows and not synthetic else FAIL,
        "Calibration targets are marked measured."
        if rows and not synthetic
        else "Calibration targets are missing target_source=measured or are explicitly synthetic.",
        accuracy_blocking=bool(synthetic) or not rows,
        evidence={"path": str(targets_path), "non_measured_cases": synthetic},
    )

    if target_report_path is None:
        add_check(
            checks,
            "calibration_target_residuals",
            INFO,
            "No calibration target residual report was supplied.",
            accuracy_blocking=False,
            evidence=None,
        )
        return
    report = read_json(target_report_path)
    residual_pass = bool(report.get("residual_pass", False))
    all_targets_measured = bool(report.get("all_targets_measured", False))
    issues = list(report.get("issues", []))
    add_check(
        checks,
        "calibration_target_residuals",
        PASS if residual_pass else FAIL,
        "Best calibration entry passes per-target current and split residual tolerances."
        if residual_pass
        else "Best calibration entry does not pass per-target residual tolerances.",
        accuracy_blocking=not residual_pass,
        framework_blocking=False,
        evidence={
            "path": str(target_report_path),
            "schema": report.get("schema"),
            "row_count": report.get("row_count"),
            "best_eval_index": report.get("best_eval_index"),
            "best_residual_norm": report.get("best_residual_norm"),
            "all_targets_measured": all_targets_measured,
            "non_measured_targets": report.get("non_measured_targets", []),
            "product_accuracy_ready": report.get("product_accuracy_ready", False),
            "issues": issues,
        },
    )


def check_transport_sensitivity(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "transport_runtime_calibration_controls",
            INFO,
            "No transport sensitivity report was supplied.",
            framework_blocking=False,
            accuracy_blocking=False,
        )
        return
    report = read_json(path)
    wiring_pass = bool(report.get("solver_parameter_wiring_pass", False))
    response_pass = bool(report.get("response_sensitivity_pass", False))
    add_check(
        checks,
        "transport_runtime_calibration_controls",
        PASS if wiring_pass else FAIL,
        "Runtime mobility, lifetime, fixed-charge, and interface-trap calibration controls are wired into DEVSIM summaries."
        if wiring_pass
        else "Runtime transport/interface calibration controls are missing from one or more DEVSIM summaries.",
        framework_blocking=not wiring_pass,
        accuracy_blocking=False,
        evidence={
            "path": str(path),
            "schema": report.get("schema"),
            "row_count": report.get("row_count"),
            "scenario_count": report.get("scenario_count"),
            "case_count": report.get("case_count"),
            "response_sensitivity_pass": response_pass,
            "changed_row_count": report.get("changed_row_count"),
            "wiring_issues": report.get("wiring_issues", []),
        },
    )


def check_transport_calibration_report(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "transport_multi_parameter_calibration",
            INFO,
            "No multi-parameter transport calibration target report was supplied.",
            framework_blocking=False,
            accuracy_blocking=False,
        )
        return
    report = read_json(path)
    residual_pass = bool(report.get("residual_pass", False))
    metric_pass = report.get("metric_residual_pass")
    metric_count = int(report.get("metric_residual_row_count", 0) or 0)
    pass_status = residual_pass and (metric_pass is not False) and metric_count > 0
    add_check(
        checks,
        "transport_multi_parameter_calibration",
        PASS if pass_status else FAIL,
        "Multi-parameter calibration report passes current/split and metric residual checks."
        if pass_status
        else "Multi-parameter calibration report is missing or failing residual checks.",
        framework_blocking=not pass_status,
        accuracy_blocking=False,
        evidence={
            "path": str(path),
            "schema": report.get("schema"),
            "row_count": report.get("row_count"),
            "metric_residual_row_count": metric_count,
            "metric_residual_pass": metric_pass,
            "residual_pass": residual_pass,
            "all_targets_measured": report.get("all_targets_measured", False),
            "product_accuracy_ready": report.get("product_accuracy_ready", False),
            "max_abs_normalized_residual": report.get("max_abs_normalized_residual"),
            "non_measured_targets": report.get("non_measured_targets", []),
        },
    )


def check_optical_stack(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        add_check(
            checks,
            "measured_optical_stack_nk",
            FAIL,
            "No measured optical-stack geometry/n,k evidence was supplied.",
            accuracy_blocking=True,
        )
        return
    data = read_json(path)
    measured = bool(data.get("calibration_status", {}).get("is_measured", False)) or bool(
        data.get("measured", False)
    )
    summary = data.get("summary", {})
    missing_tables = list(summary.get("missing_nk_tables", []))
    invalid_tables = list(summary.get("invalid_nk_tables", []))
    coverage_failures = list(summary.get("nk_coverage_failures", []))
    missing_geometry = list(data.get("geometry", {}).get("missing_required_keys", []))
    non_measured = list(summary.get("non_measured_materials", []))
    proxy_materials = list(summary.get("proxy_materials", []))
    evidence = {
        "path": str(path),
        "schema": data.get("schema"),
        "stack_name": data.get("stack_name"),
        "calibration_status": data.get("calibration_status", {}),
        "missing_required_geometry": missing_geometry,
        "missing_nk_tables": missing_tables,
        "invalid_nk_tables": invalid_tables,
        "nk_coverage_failures": coverage_failures,
        "required_wavelengths_um": summary.get("required_wavelengths_um", []),
        "non_measured_materials": non_measured,
        "proxy_materials": proxy_materials,
    }
    if missing_tables or missing_geometry or invalid_tables or coverage_failures:
        add_check(
            checks,
            "measured_optical_stack_nk",
            FAIL,
            "Optical stack evidence is incomplete: required geometry or valid wavelength-covered n,k tables are missing.",
            accuracy_blocking=True,
            framework_blocking=True,
            evidence=evidence,
        )
        return
    add_check(
        checks,
        "measured_optical_stack_nk",
        PASS if measured else FAIL,
        "Optical stack is marked measured."
        if measured
        else "Optical stack evidence exists, but geometry and/or material n,k are not marked measured.",
        accuracy_blocking=not measured,
        evidence=evidence,
    )


def check_weighting_potential(
    summary_path: Path | None,
    csv_path: Path | None,
    split_summary_paths: list[Path],
    checks: list[dict[str, Any]],
    sum_tol: float,
    gw_manifest_path: Path | None = None,
) -> None:
    if summary_path is None and csv_path is None:
        add_check(
            checks,
            "devsim_weighting_laplace",
            WARN,
            "No DEVSIM-native Laplace weighting potential export was supplied.",
            framework_blocking=False,
        )
        return
    if summary_path is None or not summary_path.exists():
        add_check(
            checks,
            "devsim_weighting_summary",
            FAIL,
            "DEVSIM weighting summary JSON is missing.",
            framework_blocking=True,
            evidence=str(summary_path) if summary_path else None,
        )
        return
    summary = read_json(summary_path)
    schema_ok = summary.get("schema") == "devsim_weighting_potential_2d_v1"
    method_ok = summary.get("method") == "pure_laplace_dirichlet_terminal_weighting"
    contacts = set(str(contact) for contact in summary.get("contacts", []))
    missing_contacts = sorted({"anode", "cathode_left", "cathode_right"} - contacts)
    node_count = int(summary.get("node_count", 0))
    sum_error = float(summary.get("sum_all_contacts_max_abs_error_to_one", math.inf))
    summary_ok = schema_ok and method_ok and not missing_contacts and node_count > 0 and sum_error <= sum_tol
    add_check(
        checks,
        "devsim_weighting_laplace_summary",
        PASS if summary_ok else FAIL,
        "DEVSIM-native pure-Laplace terminal weighting summary is internally consistent."
        if summary_ok
        else "DEVSIM weighting summary failed schema/method/contact/node-count/sum checks.",
        framework_blocking=not summary_ok,
        evidence={
            "path": str(summary_path),
            "schema": summary.get("schema"),
            "method": summary.get("method"),
            "node_count": node_count,
            "missing_contacts": missing_contacts,
            "sum_all_contacts_max_abs_error_to_one": sum_error,
            "sum_tolerance": sum_tol,
        },
    )

    resolved_csv = csv_path
    if resolved_csv is None:
        output_value = summary.get("outputs", {}).get("weighting_csv")
        resolved_csv = Path(output_value) if output_value else None
    if resolved_csv is None or not resolved_csv.exists():
        add_check(
            checks,
            "devsim_weighting_csv",
            FAIL,
            "DEVSIM weighting CSV is missing.",
            framework_blocking=True,
            evidence=str(resolved_csv) if resolved_csv else None,
        )
        return
    with resolved_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "x_cm",
        "y_cm",
        "w_anode_devsim_laplace",
        "w_cathode_left_devsim_laplace",
        "w_cathode_right_devsim_laplace",
        "w_total_devsim_laplace",
    }
    columns = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(required_columns - columns)
    csv_ok = len(rows) == node_count and not missing_columns
    add_check(
        checks,
        "devsim_weighting_csv",
        PASS if csv_ok else FAIL,
        "DEVSIM weighting CSV row count and required columns are valid."
        if csv_ok
        else "DEVSIM weighting CSV row count or required columns are invalid.",
        framework_blocking=not csv_ok,
        evidence={
            "path": str(resolved_csv),
            "row_count": len(rows),
            "expected_node_count": node_count,
            "missing_columns": missing_columns,
        },
    )

    split_node_counts = {
        str(path): int(read_json(path).get("node_count", 0))
        for path in split_summary_paths
        if path.exists()
    }
    mismatches = {
        path: count for path, count in split_node_counts.items() if count and count != node_count
    }
    if split_node_counts:
        primary_native_only = response_manifest_is_native_only(gw_manifest_path)
        if not mismatches:
            mismatch_status = PASS
            mismatch_details = "DEVSIM weighting node count matches supplied split-PD run node counts."
        elif primary_native_only:
            mismatch_status = INFO
            mismatch_details = (
                "DEVSIM weighting node count does not match one or more split-PD runs, "
                "but the supplied primary response manifest is native_devsim-only; the "
                "weighting surrogate mesh is not used by this direct terminal-current LUT."
            )
        else:
            mismatch_status = FAIL
            mismatch_details = "DEVSIM weighting node count does not match one or more split-PD runs."
        add_check(
            checks,
            "devsim_weighting_mesh_matches_split_runs",
            mismatch_status,
            mismatch_details,
            framework_blocking=bool(mismatches) and not primary_native_only,
            evidence={
                "weighting_node_count": node_count,
                "split_node_counts": split_node_counts,
                "mismatches": mismatches,
                "primary_response_native_only": primary_native_only,
                "primary_response_manifest": str(gw_manifest_path) if gw_manifest_path else None,
            },
        )


def response_manifest_is_native_only(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    data = read_json(path)
    methods = {str(method) for method in data.get("response_methods", [])}
    outputs = data.get("outputs", {})
    if isinstance(outputs, dict):
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            methods.update(str(method) for method in output.get("response_methods", []))
    return bool(
        "native_devsim" in methods
        and not any(method.startswith("gw_") for method in methods)
        and (
            data.get("schema") == "native_devsim_research_lut_export_v1"
            or methods == {"native_devsim"}
        )
    )


def check_gw_manifest(
    path: Path | None,
    checks: list[dict[str, Any]],
    *,
    dd_probe_total_rel_error_max: float,
    dd_probe_split_abs_error_max: float,
) -> None:
    if path is None:
        add_check(checks, "gw_manifest_weighting_methods", INFO, "No G*W manifest was supplied.")
        return
    data = read_json(path)
    methods: set[str] = set()
    method_sources: list[str] = []
    outputs = data.get("outputs", {})

    def resolve_manifest_path(value: Any) -> Path:
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate
        root_candidate = ROOT / candidate
        if root_candidate.exists():
            return root_candidate
        return path.parent / candidate

    top_level_methods = data.get("response_methods", [])
    if top_level_methods:
        methods.update(str(method) for method in top_level_methods)
        method_sources.append("manifest.response_methods")

    for output_key in ("camera_system_diagnostic", "camera_system_lut"):
        output = outputs.get(output_key, {})
        if not isinstance(output, dict):
            continue
        inline_methods = output.get("response_methods", [])
        if inline_methods:
            methods.update(str(method) for method in inline_methods)
            method_sources.append(f"manifest.outputs.{output_key}.response_methods")
        for path_key in ("json", "path"):
            json_value = output.get(path_key)
            if not json_value:
                continue
            json_path = resolve_manifest_path(json_value)
            if not json_path.exists():
                continue
            json_methods = read_json(json_path).get("response_methods", [])
            if json_methods:
                methods.update(str(method) for method in json_methods)
                method_sources.append(str(json_path))

    has_native = "native_devsim" in methods
    has_devsim_laplace = "gw_devsim_laplace_ref_scaled" in methods
    has_devsim_dd_probe = "gw_devsim_dd_probe_ref_scaled" in methods
    cases = data.get("cases", [])
    native_cases = [
        case.get("case")
        for case in cases
        if isinstance(case.get("native_total_abs_delta_a_per_cm"), (int, float))
        and math.isfinite(float(case.get("native_total_abs_delta_a_per_cm")))
        and isinstance(case.get("native_split_phase_x_proxy"), (int, float))
        and math.isfinite(float(case.get("native_split_phase_x_proxy")))
    ]
    add_check(
        checks,
        "native_devsim_direct_method_present",
        PASS if has_native else FAIL,
        "Camera diagnostic includes native_devsim direct FDTD-generation DEVSIM responses."
        if has_native
        else "Camera diagnostic is missing native_devsim direct responses; proxy/surrogate G*W is not enough for accuracy-oriented LUT work.",
        accuracy_blocking=not has_native,
        framework_blocking=not has_native,
        evidence={
            "path": str(path),
            "response_methods": sorted(methods),
        },
    )
    add_check(
        checks,
        "native_devsim_direct_cases_present",
        PASS if len(native_cases) == len(cases) and cases else FAIL,
        "Every camera-system case has finite native DEVSIM terminal-current response and split phase."
        if len(native_cases) == len(cases) and cases
        else "One or more camera-system cases lack finite native DEVSIM terminal-current response or split phase.",
        accuracy_blocking=not (len(native_cases) == len(cases) and cases),
        framework_blocking=not (len(native_cases) == len(cases) and cases),
        evidence={"case_count": len(cases), "native_cases": native_cases},
    )
    research_lut_output = outputs.get("camera_system_research_lut", {})
    research_lut_json = None
    research_lut_npz = None
    if isinstance(research_lut_output, dict):
        json_value = research_lut_output.get("json")
        npz_value = research_lut_output.get("npz")
        if json_value:
            research_lut_json = resolve_manifest_path(json_value)
        if npz_value:
            research_lut_npz = resolve_manifest_path(npz_value)
    research_lut_data = read_json(research_lut_json) if research_lut_json and research_lut_json.exists() else {}
    research_lut_ok = bool(
        research_lut_json
        and research_lut_json.exists()
        and research_lut_npz
        and research_lut_npz.exists()
        and research_lut_data.get("primary_response_method") == "native_devsim"
        and research_lut_data.get("research_lut_ready") is True
        and research_lut_data.get("case_count") == len(cases)
    )
    add_check(
        checks,
        "camera_system_research_lut_native_export",
        PASS if research_lut_ok else FAIL,
        "Camera-system research LUT is exported from native_devsim direct responses."
        if research_lut_ok
        else "Camera-system research LUT is missing or is not marked as native_devsim direct response.",
        framework_blocking=not research_lut_ok,
        accuracy_blocking=False,
        evidence={
            "json": str(research_lut_json) if research_lut_json else "",
            "json_exists": bool(research_lut_json and research_lut_json.exists()),
            "npz": str(research_lut_npz) if research_lut_npz else "",
            "npz_exists": bool(research_lut_npz and research_lut_npz.exists()),
            "primary_response_method": research_lut_data.get("primary_response_method"),
            "research_lut_ready": research_lut_data.get("research_lut_ready"),
            "case_count": research_lut_data.get("case_count"),
        },
    )
    research_lut_convergence = research_lut_data.get("numerical_convergence", {})
    if isinstance(research_lut_convergence, dict) and research_lut_convergence:
        lut_convergence_pass = research_lut_convergence.get("passed") is True
        lut_convergence_fail = research_lut_convergence.get("passed") is False
        add_check(
            checks,
            "camera_system_research_lut_numerical_convergence",
            PASS if lut_convergence_pass else FAIL if lut_convergence_fail else WARN,
            "Research LUT embeds a passing optical numerical-convergence report."
            if lut_convergence_pass
            else "Research LUT numerical-convergence evidence is missing or did not pass.",
            framework_blocking=lut_convergence_fail,
            accuracy_blocking=lut_convergence_fail,
            evidence={
                "json": str(research_lut_json) if research_lut_json else "",
                "optical_convergence_report": research_lut_convergence.get(
                    "optical_convergence_report"
                ),
                "passed": research_lut_convergence.get("passed"),
                "spatial_convergence_pass": research_lut_convergence.get(
                    "spatial_convergence_pass"
                ),
                "time_convergence_pass": research_lut_convergence.get("time_convergence_pass"),
                "pml_convergence_pass": research_lut_convergence.get("pml_convergence_pass"),
                "full_numerical_convergence_pass": research_lut_convergence.get(
                    "full_numerical_convergence_pass"
                ),
                "varied_axes": research_lut_convergence.get("varied_axes"),
                "unproven_axes": research_lut_convergence.get("unproven_axes"),
                "failed_axes": research_lut_convergence.get("failed_axes"),
                "max_total_response_rel_delta_to_reference": research_lut_convergence.get(
                    "max_total_response_rel_delta_to_reference"
                ),
                "negative_signed_flux_count": research_lut_convergence.get(
                    "negative_signed_flux_count"
                ),
            },
        )
        full_lut_convergence_pass = (
            research_lut_convergence.get("full_numerical_convergence_pass") is True
        )
        full_lut_convergence_explicit_fail = (
            research_lut_convergence.get("full_numerical_convergence_pass") is False
        )
        add_check(
            checks,
            "camera_system_research_lut_full_numerical_coverage",
            PASS
            if full_lut_convergence_pass
            else FAIL
            if full_lut_convergence_explicit_fail
            else WARN,
            "Research LUT optical convergence varied resolution, time, and PML axes."
            if full_lut_convergence_pass
            else "Research LUT optical convergence is partial; one or more of resolution, time, or PML axes were not varied.",
            framework_blocking=False,
            accuracy_blocking=not full_lut_convergence_pass,
            evidence={
                "json": str(research_lut_json) if research_lut_json else "",
                "optical_convergence_report": research_lut_convergence.get(
                    "optical_convergence_report"
                ),
                "full_numerical_convergence_pass": research_lut_convergence.get(
                    "full_numerical_convergence_pass"
                ),
                "spatial_convergence_pass": research_lut_convergence.get(
                    "spatial_convergence_pass"
                ),
                "time_convergence_pass": research_lut_convergence.get("time_convergence_pass"),
                "pml_convergence_pass": research_lut_convergence.get("pml_convergence_pass"),
                "varied_axes": research_lut_convergence.get("varied_axes"),
                "unproven_axes": research_lut_convergence.get("unproven_axes"),
                "failed_axes": research_lut_convergence.get("failed_axes"),
                "unique_resolution_count": research_lut_convergence.get(
                    "unique_resolution_count"
                ),
                "unique_after_source_time_count": research_lut_convergence.get(
                    "unique_after_source_time_count"
                ),
                "unique_pml_count": research_lut_convergence.get("unique_pml_count"),
            },
        )
    else:
        add_check(
            checks,
            "camera_system_research_lut_numerical_convergence",
            WARN,
            "Research LUT does not embed optical numerical-convergence evidence.",
            framework_blocking=False,
            accuracy_blocking=False,
            evidence={"json": str(research_lut_json) if research_lut_json else ""},
        )
    optical_evidence = research_lut_data.get("optical_generation_evidence", {})
    if isinstance(optical_evidence, dict) and optical_evidence:
        optical_grid_pass = bool(optical_evidence.get("all_grid_resolution_gate_pass"))
        optical_missing_cases = optical_evidence.get("missing_cases", [])
        status = PASS if optical_grid_pass else FAIL
        add_check(
            checks,
            "camera_system_research_lut_optical_grid_evidence",
            status,
            "Research LUT includes per-case FDTD optical grid evidence and every case passes the configured grid-resolution gate."
            if optical_grid_pass
            else "Research LUT optical evidence is present, but one or more cases fail or are missing the grid-resolution gate.",
            framework_blocking=not optical_grid_pass,
            accuracy_blocking=not optical_grid_pass,
            evidence={
                "case_count": optical_evidence.get("case_count"),
                "missing_cases": optical_missing_cases,
                "all_grid_resolution_gate_pass": optical_evidence.get(
                    "all_grid_resolution_gate_pass"
                ),
                "min_si_internal_wavelength_pixels": optical_evidence.get(
                    "min_si_internal_wavelength_pixels"
                ),
                "min_critical_feature_pixels": optical_evidence.get(
                    "min_critical_feature_pixels"
                ),
            },
        )
    else:
        add_check(
            checks,
            "camera_system_research_lut_optical_grid_evidence",
            WARN,
            "Research LUT does not embed per-case FDTD optical grid evidence.",
            framework_blocking=False,
            accuracy_blocking=False,
            evidence={"json": str(research_lut_json) if research_lut_json else ""},
        )
    native_only_manifest = bool(
        has_native
        and not has_devsim_laplace
        and not has_devsim_dd_probe
        and (
            data.get("schema") == "native_devsim_research_lut_export_v1"
            or methods == {"native_devsim"}
        )
    )
    if native_only_manifest:
        add_check(
            checks,
            "gw_manifest_weighting_methods",
            INFO,
            "Native-only research LUT manifest supplied; G*W surrogate weighting is not part of this direct terminal-current export.",
            framework_blocking=False,
            evidence={
                "path": str(path),
                "schema": data.get("schema"),
                "response_methods": sorted(methods),
                "method_sources": method_sources,
            },
        )
        add_check(
            checks,
            "gw_devsim_dd_probe_method_present",
            INFO,
            "Native-only research LUT uses direct DEVSIM terminal-current deltas; sparse DD-probe surrogate is optional and not required for this export.",
            framework_blocking=False,
            evidence={
                "path": str(path),
                "schema": data.get("schema"),
                "response_methods": sorted(methods),
            },
        )
        return
    add_check(
        checks,
        "gw_manifest_weighting_methods",
        PASS if has_devsim_laplace else WARN,
        "G*W camera diagnostic includes gw_devsim_laplace_ref_scaled."
        if has_devsim_laplace
        else "G*W camera diagnostic does not include gw_devsim_laplace_ref_scaled.",
        framework_blocking=False,
        evidence={
            "path": str(path),
            "response_methods": sorted(methods),
            "method_sources": method_sources,
        },
    )
    add_check(
        checks,
        "gw_devsim_dd_probe_method_present",
        PASS if has_devsim_dd_probe else INFO,
        "G*W camera diagnostic includes sparse drift-diffusion DD-probe weighting."
        if has_devsim_dd_probe
        else "G*W camera diagnostic does not include sparse drift-diffusion DD-probe weighting.",
        framework_blocking=False,
        evidence={
            "path": str(path),
            "response_methods": sorted(methods),
        },
    )
    finite_errors = [
        case.get("case")
        for case in cases
        if isinstance(case.get("gw_devsim_laplace_total_reference_scaled_rel_error"), (int, float))
        and math.isfinite(float(case.get("gw_devsim_laplace_total_reference_scaled_rel_error")))
    ]
    add_check(
        checks,
        "gw_devsim_laplace_errors_present",
        PASS if len(finite_errors) == len(cases) and cases else WARN,
        "All G*W cases include finite W_devsim_laplace reference-scaled errors."
        if len(finite_errors) == len(cases) and cases
        else "Some G*W cases do not include finite W_devsim_laplace errors.",
        framework_blocking=False,
        evidence={"case_count": len(cases), "finite_error_cases": finite_errors},
    )
    dd_probe_finite_errors = [
        case.get("case")
        for case in cases
        if isinstance(case.get("gw_devsim_dd_probe_total_reference_scaled_rel_error"), (int, float))
        and math.isfinite(float(case.get("gw_devsim_dd_probe_total_reference_scaled_rel_error")))
    ]
    add_check(
        checks,
        "gw_devsim_dd_probe_errors_present",
        PASS if has_devsim_dd_probe and len(dd_probe_finite_errors) == len(cases) and cases else INFO,
        "All G*W cases include finite W_devsim_dd_probe reference-scaled errors."
        if has_devsim_dd_probe and len(dd_probe_finite_errors) == len(cases) and cases
        else "W_devsim_dd_probe errors are unavailable or incomplete.",
        framework_blocking=False,
        evidence={"case_count": len(cases), "finite_error_cases": dd_probe_finite_errors},
    )
    dd_probe_total_errors = [
        abs(float(case.get("gw_devsim_dd_probe_total_reference_scaled_rel_error")))
        for case in cases
        if isinstance(case.get("gw_devsim_dd_probe_total_reference_scaled_rel_error"), (int, float))
        and math.isfinite(float(case.get("gw_devsim_dd_probe_total_reference_scaled_rel_error")))
    ]
    dd_probe_split_errors = [
        abs(float(case.get("gw_devsim_dd_probe_split_phase_error")))
        for case in cases
        if isinstance(case.get("gw_devsim_dd_probe_split_phase_error"), (int, float))
        and math.isfinite(float(case.get("gw_devsim_dd_probe_split_phase_error")))
    ]
    total_gate_ok = bool(
        has_devsim_dd_probe
        and len(dd_probe_total_errors) == len(cases)
        and cases
        and max(dd_probe_total_errors) <= dd_probe_total_rel_error_max
    )
    split_gate_ok = bool(
        has_devsim_dd_probe
        and len(dd_probe_split_errors) == len(cases)
        and cases
        and max(dd_probe_split_errors) <= dd_probe_split_abs_error_max
    )
    add_check(
        checks,
        "gw_devsim_dd_probe_total_error_gate",
        PASS if total_gate_ok else FAIL if has_devsim_dd_probe else INFO,
        f"W_devsim_dd_probe total-response error is within {dd_probe_total_rel_error_max:g}."
        if total_gate_ok
        else f"W_devsim_dd_probe total-response error exceeds {dd_probe_total_rel_error_max:g} or is incomplete.",
        framework_blocking=False,
        accuracy_blocking=False,
        evidence={
            "threshold": dd_probe_total_rel_error_max,
            "max_abs_error": max(dd_probe_total_errors) if dd_probe_total_errors else None,
            "errors": dd_probe_total_errors,
        },
    )
    add_check(
        checks,
        "gw_devsim_dd_probe_split_error_gate",
        PASS if split_gate_ok else FAIL if has_devsim_dd_probe else INFO,
        f"W_devsim_dd_probe split-phase error is within {dd_probe_split_abs_error_max:g}."
        if split_gate_ok
        else f"W_devsim_dd_probe split-phase error exceeds {dd_probe_split_abs_error_max:g} or is incomplete, so it must not be promoted to an accuracy LUT surrogate.",
        framework_blocking=False,
        accuracy_blocking=False,
        evidence={
            "threshold": dd_probe_split_abs_error_max,
            "max_abs_error": max(dd_probe_split_errors) if dd_probe_split_errors else None,
            "errors": dd_probe_split_errors,
        },
    )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TCAD Accuracy Gate",
        "",
        f"- Framework ready: `{report['framework_ready']}`",
        f"- Accuracy LUT ready: `{report['accuracy_ready']}`",
        f"- Accuracy-blocking failures: `{report['accuracy_blocking_failure_count']}`",
        "",
        "| Check | Status | Accuracy Blocking | Framework Blocking | Details |",
        "|---|---:|---:|---:|---|",
    ]
    for check in report["checks"]:
        details = str(check["details"]).replace("\n", " ")
        lines.append(
            f"| {check['name']} | {check['status']} | "
            f"{check['accuracy_blocking']} | {check['framework_blocking']} | {details} |"
        )
    lines.append("")
    lines.append(
        "A PASS here only means the supplied open-source framework artifacts are internally "
        "consistent. Product accuracy requires all accuracy-blocking checks to pass with "
        "measured stack, measured electrical profile, convergence, and measured calibration."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    split_summary_paths = expand_split_summary_paths(args.split_summary, args.split_summary_dir)
    profile_data = check_profile(args.profile, checks)
    check_split_summaries(split_summary_paths, checks, args.terminal_balance_tol)
    check_gmsh_summaries(args.gmsh_summary, checks)
    check_resolved_dti_oxide(
        args.resolved_dti_mesh_metadata,
        args.resolved_dti_split_summary,
        checks,
        args.terminal_balance_tol,
    )
    check_convergence_report(args.convergence_report, checks)
    check_crosstalk_xsection_convergence(args.crosstalk_xsection_convergence, checks)
    check_native_response_convergence(args.native_response_convergence_report, checks)
    check_signed_field_symmetry_validation(args.signed_field_symmetry_validation, checks)
    check_camera_lut_spectral_coverage(args.camera_lut_spectral_coverage, checks)
    check_camera_system_field_lut(args.camera_system_field_lut, checks)
    check_tg_fd_diagnostic(args.tg_fd_report, checks)
    check_tg_fd_transient(args.tg_fd_transient_report, checks)
    check_calibration(args.calibration_result, args.targets_csv, args.calibration_target_report, checks)
    check_transport_sensitivity(args.transport_sensitivity_report, checks)
    check_transport_calibration_report(args.transport_calibration_target_report, checks)
    check_optical_stack(args.optical_stack_summary, checks)
    check_weighting_potential(
        args.weighting_summary,
        args.weighting_csv,
        split_summary_paths,
        checks,
        args.weighting_sum_tol,
        args.gw_manifest,
    )
    check_gw_manifest(
        args.gw_manifest,
        checks,
        dd_probe_total_rel_error_max=args.gw_dd_probe_total_rel_error_max,
        dd_probe_split_abs_error_max=args.gw_dd_probe_split_abs_error_max,
    )

    accuracy_failures = [
        check for check in checks if check["status"] == FAIL and check["accuracy_blocking"]
    ]
    framework_failures = [
        check for check in checks if check["status"] == FAIL and check["framework_blocking"]
    ]
    report = {
        "schema": "tcad_accuracy_gate_v1",
        "profile": str(args.profile),
        "profile_name": profile_data.get("profile_name"),
        "framework_ready": not framework_failures,
        "accuracy_ready": not accuracy_failures,
        "accuracy_blocking_failure_count": len(accuracy_failures),
        "framework_blocking_failure_count": len(framework_failures),
        "inputs": {
            "split_summary": [str(path) for path in split_summary_paths],
            "split_summary_dir": [str(path) for path in args.split_summary_dir],
            "gmsh_summary": [str(path) for path in args.gmsh_summary],
            "resolved_dti_mesh_metadata": str(args.resolved_dti_mesh_metadata)
            if args.resolved_dti_mesh_metadata
            else None,
            "resolved_dti_split_summary": str(args.resolved_dti_split_summary)
            if args.resolved_dti_split_summary
            else None,
            "convergence_report": str(args.convergence_report) if args.convergence_report else None,
            "crosstalk_xsection_convergence": str(args.crosstalk_xsection_convergence)
            if args.crosstalk_xsection_convergence
            else None,
            "native_response_convergence_report": str(args.native_response_convergence_report)
            if args.native_response_convergence_report
            else None,
            "signed_field_symmetry_validation": str(args.signed_field_symmetry_validation)
            if args.signed_field_symmetry_validation
            else None,
            "camera_lut_spectral_coverage": str(args.camera_lut_spectral_coverage)
            if args.camera_lut_spectral_coverage
            else None,
            "camera_system_field_lut": str(args.camera_system_field_lut)
            if args.camera_system_field_lut
            else None,
            "tg_fd_report": str(args.tg_fd_report) if args.tg_fd_report else None,
            "tg_fd_transient_report": str(args.tg_fd_transient_report)
            if args.tg_fd_transient_report
            else None,
            "calibration_result": str(args.calibration_result) if args.calibration_result else None,
            "targets_csv": str(args.targets_csv) if args.targets_csv else None,
            "calibration_target_report": str(args.calibration_target_report)
            if args.calibration_target_report
            else None,
            "transport_sensitivity_report": str(args.transport_sensitivity_report)
            if args.transport_sensitivity_report
            else None,
            "transport_calibration_target_report": str(args.transport_calibration_target_report)
            if args.transport_calibration_target_report
            else None,
            "optical_stack_summary": str(args.optical_stack_summary)
            if args.optical_stack_summary
            else None,
            "weighting_summary": str(args.weighting_summary) if args.weighting_summary else None,
            "weighting_csv": str(args.weighting_csv) if args.weighting_csv else None,
            "gw_manifest": str(args.gw_manifest) if args.gw_manifest else None,
        },
        "checks": checks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tcad_accuracy_gate.json"
    md_path = args.output_dir / "tcad_accuracy_gate.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps(report, indent=2))
    if args.fail_on_accuracy_fail and not report["accuracy_ready"]:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, action="append", default=[])
    parser.add_argument("--split-summary-dir", type=Path, action="append", default=[])
    parser.add_argument("--gmsh-summary", type=Path, action="append", default=[])
    parser.add_argument("--resolved-dti-mesh-metadata", type=Path, default=None)
    parser.add_argument("--resolved-dti-split-summary", type=Path, default=None)
    parser.add_argument("--convergence-report", type=Path, default=None)
    parser.add_argument("--crosstalk-xsection-convergence", type=Path, default=None)
    parser.add_argument("--native-response-convergence-report", type=Path, default=None)
    parser.add_argument("--signed-field-symmetry-validation", type=Path, default=None)
    parser.add_argument("--camera-lut-spectral-coverage", type=Path, default=None)
    parser.add_argument("--camera-system-field-lut", type=Path, default=None)
    parser.add_argument("--tg-fd-report", type=Path, default=None)
    parser.add_argument("--tg-fd-transient-report", type=Path, default=None)
    parser.add_argument("--calibration-result", type=Path, default=None)
    parser.add_argument("--targets-csv", type=Path, default=None)
    parser.add_argument("--calibration-target-report", type=Path, default=None)
    parser.add_argument("--transport-sensitivity-report", type=Path, default=None)
    parser.add_argument("--transport-calibration-target-report", type=Path, default=None)
    parser.add_argument("--optical-stack-summary", type=Path, default=None)
    parser.add_argument("--weighting-summary", type=Path, default=None)
    parser.add_argument("--weighting-csv", type=Path, default=None)
    parser.add_argument("--weighting-sum-tol", type=float, default=1.0e-9)
    parser.add_argument("--gw-manifest", type=Path, default=None)
    parser.add_argument("--gw-dd-probe-total-rel-error-max", type=float, default=1.0e-2)
    parser.add_argument("--gw-dd-probe-split-abs-error-max", type=float, default=2.0e-2)
    parser.add_argument("--terminal-balance-tol", type=float, default=1.0e-9)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tcad_accuracy_gate"))
    parser.add_argument("--fail-on-accuracy-fail", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
