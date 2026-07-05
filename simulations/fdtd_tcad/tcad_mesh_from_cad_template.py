#!/usr/bin/env python3
"""Generate a DEVSIM-oriented TCAD mesh bridge from a CAD template.

This bridges the reusable pixel CAD template library to the existing
Gmsh/DEVSIM split-PD mesh path. It derives a 2D electrical cross-section from
`template_parameters.json` and writes an explicit report describing what was
and was not preserved.

The resulting mesh is more traceable than a hard-coded bbox proxy, but it is
still not a measured product device mesh and it is not a full 3D CAD-solid
electrical solve.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tcad_gmsh_pixel_mesh import PixelMeshConfig, generate_2d


ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / "runs" / "pixel_cad_template_library_reference"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def template_dir_from_args(args: argparse.Namespace) -> Path:
    if args.template_dir:
        return args.template_dir.resolve()
    if not args.template_id:
        raise ValueError("--template-id or --template-dir is required")
    return (args.library_root / args.template_id).resolve()


def central_ocl_block(blocks: list[dict[str, Any]], nx: int, nz: int) -> dict[str, Any]:
    if not blocks:
        raise ValueError("template_parameters.json must define ocl_blocks")
    center_x = nx / 2.0
    center_z = nz / 2.0

    def key(block: dict[str, Any]) -> tuple[float, str]:
        ix = float(block.get("ix", 0))
        iz = float(block.get("iz", 0))
        sx = float(block.get("sx", 1))
        sz = float(block.get("sz", 1))
        bx = ix + sx / 2.0
        bz = iz + sz / 2.0
        return ((bx - center_x) ** 2 + (bz - center_z) ** 2, str(block.get("lens_id", "")))

    return min(blocks, key=key)


def physical_names(mesh_path: Path) -> dict[int, list[str]]:
    lines = [line.strip() for line in mesh_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    names: dict[int, list[str]] = {}
    if "$PhysicalNames" not in lines:
        return names
    start = lines.index("$PhysicalNames") + 1
    count = int(lines[start])
    for line in lines[start + 1 : start + 1 + count]:
        dim_text, _tag_text, raw_name = line.split(maxsplit=2)
        names.setdefault(int(dim_text), []).append(raw_name.strip('"'))
    return names


def normalize_split_mode(split_mode: str) -> str:
    return str(split_mode or "none").replace("-", "_")


def choose_section_axis(split_mode: str, requested_axis: str) -> str:
    if requested_axis in {"x", "z"}:
        return requested_axis
    normalized = normalize_split_mode(split_mode)
    if normalized == "dual_z":
        return "z"
    return "x"


def split_axis_capability(split_mode: str, section_axis: str) -> dict[str, Any]:
    normalized = str(split_mode or "none").replace("-", "_")
    axis = section_axis if section_axis in {"x", "z"} else "x"
    axis_labels = {
        "x": {"cathode_left": "left", "cathode_right": "right"},
        "z": {"cathode_left": "bottom", "cathode_right": "top"},
    }[axis]
    if normalized == "dual_x":
        gate = "PASS" if axis == "x" else "CHECK"
        return {
            "gate": gate,
            "requested_split_axis": "x",
            "section_axis": axis,
            "represented_split_axis": "x" if axis == "x" else None,
            "contact_axis_labels": axis_labels,
            "supported_phase_axes": ["x"] if axis == "x" else [],
            "phase_result_scope": "left/right x-axis split current" if axis == "x" else "z-section mesh/import only; x-axis dual-PD split phase is not represented",
            "unsupported_outputs": [] if axis == "x" else ["dual_x_left_right_phase"],
        }
    if normalized == "dual_z":
        gate = "PASS" if axis == "z" else "CHECK"
        return {
            "gate": gate,
            "requested_split_axis": "z",
            "section_axis": axis,
            "represented_split_axis": "z" if axis == "z" else None,
            "contact_axis_labels": axis_labels,
            "supported_phase_axes": ["z"] if axis == "z" else [],
            "phase_result_scope": "top/bottom z-axis split current mapped onto the 2D lateral solver" if axis == "z" else "x-section mesh/import only; z-axis split phase is not represented",
            "unsupported_outputs": [] if axis == "z" else ["dual_z_top_bottom_phase", "z_axis_pd_balance"],
        }
    if normalized == "quad":
        represented_axis = axis
        return {
            "gate": "PASS",
            "requested_split_axis": "x_and_z",
            "section_axis": axis,
            "represented_split_axis": represented_axis,
            "contact_axis_labels": axis_labels,
            "supported_phase_axes": [axis],
            "phase_result_scope": f"{axis}-axis QPD projection only; full Q1-Q4 balance needs a coupled 3D solve",
            "unsupported_outputs": ["full_qpd_q1_q4_balance", "orthogonal_axis_pd_balance"],
        }
    return {
        "gate": "PASS",
        "requested_split_axis": "none",
        "section_axis": axis,
        "represented_split_axis": axis,
        "contact_axis_labels": axis_labels,
        "supported_phase_axes": [axis],
        "phase_result_scope": f"generic split-PD {axis}-section smoke mesh",
        "unsupported_outputs": [],
    }


def derive_config(
    params: dict[str, Any],
    *,
    domain: str,
    mesh_um: float,
    fine_mesh_um: float,
    include_dti_oxide: bool,
    include_fd_contact: bool,
    include_tg_contact: bool,
    include_tg_oxide: bool,
    section_axis: str,
) -> tuple[PixelMeshConfig, dict[str, Any]]:
    nx = int(params.get("nx", 1))
    nz = int(params.get("nz", 1))
    pitch_um = float(params.get("pitch_um", 1.4))
    depth_um = float(params.get("si_thickness_um", params.get("dti_depth_um", 2.8)))
    dti_depth_um = float(params.get("dti_depth_um", depth_um))
    dti_width_um = float(params.get("dti_width_um", 0.06))
    split_gap_um = float(params.get("split_gap_um", 0.04))
    blocks = params.get("ocl_blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("ocl_blocks must be a list")
    target = central_ocl_block(blocks, nx, nz)
    split_mode = str(params.get("split_mode", "none"))
    resolved_axis = choose_section_axis(split_mode, section_axis)

    if domain == "full-template":
        x_width_um = nx * pitch_um
        z_width_um = nz * pitch_um
        domain_source = "full template nx/nz"
    else:
        x_width_um = float(target.get("sx", 1)) * pitch_um
        z_width_um = float(target.get("sz", 1)) * pitch_um
        domain_source = f"central OCL block {target.get('lens_id')}"
    lateral_width_um = z_width_um if resolved_axis == "z" else x_width_um
    orthogonal_width_um = x_width_um if resolved_axis == "z" else z_width_um

    config = PixelMeshConfig(
        width_um=lateral_width_um,
        depth_um=depth_um,
        z_width_um=orthogonal_width_um,
        split_gap_um=split_gap_um,
        mesh_um=mesh_um,
        fine_mesh_um=fine_mesh_um,
        include_fd_contact=include_fd_contact,
        include_tg_contact=include_tg_contact,
        include_tg_oxide=include_tg_oxide,
        include_dti_oxide=include_dti_oxide,
        dti_left_x_min_um=-0.5 * lateral_width_um,
        dti_left_x_max_um=-0.5 * lateral_width_um + dti_width_um,
        dti_right_x_min_um=0.5 * lateral_width_um - dti_width_um,
        dti_right_x_max_um=0.5 * lateral_width_um,
        dti_depth_min_um=0.0,
        dti_depth_max_um=min(dti_depth_um, depth_um),
    )
    derivation = {
        "template_id": params.get("template_id"),
        "label": params.get("label"),
        "domain": domain,
        "domain_source": domain_source,
        "section_axis": resolved_axis,
        "requested_section_axis": section_axis,
        "lateral_width_um": lateral_width_um,
        "orthogonal_width_um": orthogonal_width_um,
        "target_ocl_block": target,
        "split_mode": split_mode,
        "shield_mode": params.get("shield_mode", "off"),
        "electrical_capability": split_axis_capability(split_mode, resolved_axis),
        "source_fields": {
            "nx": nx,
            "nz": nz,
            "pitch_um": pitch_um,
            "x_width_um": x_width_um,
            "z_width_um": z_width_um,
            "si_thickness_um": depth_um,
            "dti_depth_um": dti_depth_um,
            "dti_width_um": dti_width_um,
            "split_gap_um": split_gap_um,
        },
    }
    return config, derivation


def run(args: argparse.Namespace) -> dict[str, Any]:
    template_dir = template_dir_from_args(args)
    params_path = template_dir / "template_parameters.json"
    geometry_path = template_dir / "geometry_import.json"
    step_path = template_dir / "model.step"
    mesh_out = args.output_dir / "split_pixel_2d.msh"
    report_path = args.output_dir / "tcad_bridge_report.json"
    config_path = args.output_dir / "derived_tcad_config.json"

    params = load_json(params_path)
    config, derivation = derive_config(
        params,
        domain=args.domain,
        mesh_um=args.mesh_um,
        fine_mesh_um=args.fine_mesh_um,
        include_dti_oxide=args.include_dti_oxide,
        include_fd_contact=args.include_fd_contact,
        include_tg_contact=args.include_tg_contact,
        include_tg_oxide=args.include_tg_oxide,
        section_axis=args.section_axis,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema": "cad_template_derived_tcad_config_v1",
        "template_parameters": str(params_path),
        "geometry_import": str(geometry_path),
        "step": str(step_path),
        "derivation": derivation,
        "pixel_mesh_config": asdict(config),
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    mesh_metadata = generate_2d(config, mesh_out)
    names = physical_names(mesh_out)
    contacts = names.get(1, [])
    regions = names.get(2, [])
    required_contacts = {"anode", "cathode_left", "cathode_right"}
    missing_contacts = sorted(required_contacts.difference(contacts))

    mesh_status = "PASS" if mesh_out.exists() and "silicon" in regions and not missing_contacts else "FAIL"
    capability = derivation.get("electrical_capability", {})
    capability_gate = capability.get("gate") if isinstance(capability, dict) else "PASS"
    status = "FAIL" if mesh_status == "FAIL" else "CHECK" if capability_gate == "CHECK" else "PASS"
    report = {
        "schema": "cad_template_tcad_mesh_bridge_v1",
        "status": status,
        "mesh_status": mesh_status,
        "bridge_type": "cad_template_parameter_derived_2d_electrical_mesh",
        "source_truth_level": "parametric_template_not_measured",
        "template_id": params.get("template_id"),
        "template_label": params.get("label"),
        "template_dir": str(template_dir),
        "template_parameters": str(params_path),
        "geometry_import": str(geometry_path),
        "step": str(step_path),
        "mesh": str(mesh_out),
        "derived_config": str(config_path),
        "mesh_metadata": mesh_metadata,
        "physical_names": names,
        "missing_required_contacts": missing_contacts,
        "electrical_capability": capability,
        "native_full_cad_electrical_mesh": False,
        "preserves_full_3d_cad_connectivity": False,
        "notes": [
            "The 2D electrical mesh dimensions are derived from template_parameters.json.",
            "This is more traceable than hard-coded bbox defaults, but it is not a measured product TCAD mesh.",
            "The mesh uses the existing DEVSIM split-PD physical names: silicon, anode, cathode_left, cathode_right, and optional oxide/TG/FD groups.",
            f"The current bridge is a {derivation.get('section_axis')}-depth 2D cross-section. It reports only the represented split axis in electrical_capability.",
            "Use measured implant, TG/FD, interface trap, DTI/BDTI, and calibrated transport data before claiming accuracy.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-id", default="")
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=("target-ocl", "full-template"), default="target-ocl")
    parser.add_argument("--section-axis", choices=("auto", "x", "z"), default="auto")
    parser.add_argument("--mesh-um", type=float, default=0.18)
    parser.add_argument("--fine-mesh-um", type=float, default=0.06)
    parser.add_argument("--include-dti-oxide", action="store_true")
    parser.add_argument("--include-fd-contact", action="store_true")
    parser.add_argument("--include-tg-contact", action="store_true")
    parser.add_argument("--include-tg-oxide", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
