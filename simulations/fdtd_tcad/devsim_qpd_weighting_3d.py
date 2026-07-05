#!/usr/bin/env python3
"""DEVSIM 3D QPD quadrant terminal weighting-potential smoke solve.

This creates a simple Gmsh 3D silicon mesh with one top anode and four bottom
quadrant cathodes, imports it into DEVSIM, and solves pure Laplace terminal
weighting potentials for Q00/Q10/Q01/Q11. It is not a calibrated 3D
drift-diffusion collection solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gmsh
import numpy as np
from devsim import (
    add_gmsh_contact,
    add_gmsh_region,
    contact_equation,
    create_device,
    create_gmsh_mesh,
    equation,
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
QPD_CONTACTS = (
    "cathode_q00_left_bottom",
    "cathode_q10_right_bottom",
    "cathode_q01_left_top",
    "cathode_q11_right_top",
)


@dataclass(frozen=True)
class QPD3DConfig:
    width_um: float = 2.8
    z_width_um: float = 2.8
    depth_um: float = 3.0
    split_gap_um: float = 0.04
    mesh_um: float = 0.38
    fine_mesh_um: float = 0.22
    output_dir: Path = ROOT / "runs" / "devsim_qpd_weighting_3d"
    device: str = "QPDWeighting3D"
    region: str = "Silicon"
    absolute_error: float = 1.0e-11
    relative_error: float = 1.0e-11
    maximum_iterations: int = 100

    @property
    def width_cm(self) -> float:
        return self.width_um * 1.0e-4

    @property
    def z_width_cm(self) -> float:
        return self.z_width_um * 1.0e-4

    @property
    def depth_cm(self) -> float:
        return self.depth_um * 1.0e-4

    @property
    def gap_half_cm(self) -> float:
        return 0.5 * self.split_gap_um * 1.0e-4

    @property
    def mesh_cm(self) -> float:
        return self.mesh_um * 1.0e-4

    @property
    def fine_mesh_cm(self) -> float:
        return self.fine_mesh_um * 1.0e-4


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_physical(dim: int, tags: list[int], name: str) -> int | None:
    clean = sorted(set(tags))
    if not clean:
        return None
    group = gmsh.model.addPhysicalGroup(dim, clean)
    gmsh.model.setPhysicalName(dim, group, name)
    return group


def generate_qpd_mesh(config: QPD3DConfig, mesh_path: Path) -> dict[str, Any]:
    gmsh.initialize()
    try:
        gmsh.model.add("qpd_weighting_3d")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", config.fine_mesh_cm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.mesh_cm)

        xmin = -0.5 * config.width_cm
        xmax = 0.5 * config.width_cm
        zmin = -0.5 * config.z_width_cm
        zmax = 0.5 * config.z_width_cm
        gap = config.gap_half_cm
        depth = config.depth_cm
        x_segments = ((xmin, -gap), (-gap, gap), (gap, xmax))
        z_segments = ((zmin, -gap), (-gap, gap), (gap, zmax))
        boxes: list[int] = []
        for x0, x1 in x_segments:
            for z0, z1 in z_segments:
                if x1 <= x0 or z1 <= z0:
                    continue
                boxes.append(gmsh.model.occ.addBox(x0, 0.0, z0, x1 - x0, depth, z1 - z0))
        gmsh.model.occ.fragment([(3, tag) for tag in boxes], [])
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        surface_tags: set[int] = set()
        for volume in volumes:
            for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False):
                if dim == 2:
                    surface_tags.add(tag)

        top_surfaces: list[int] = []
        insulated_surfaces: list[int] = []
        q_surfaces = {contact: [] for contact in QPD_CONTACTS}
        tol = max(config.fine_mesh_cm, 1.0e-12)
        for surface in sorted(surface_tags):
            sx0, sy0, sz0, sx1, sy1, sz1 = gmsh.model.getBoundingBox(2, surface)
            x_mid = 0.5 * (sx0 + sx1)
            z_mid = 0.5 * (sz0 + sz1)
            if abs(sy0) <= tol and abs(sy1) <= tol:
                top_surfaces.append(surface)
            elif abs(sy0 - depth) <= tol and abs(sy1 - depth) <= tol:
                if x_mid < -gap and z_mid < -gap:
                    q_surfaces["cathode_q00_left_bottom"].append(surface)
                elif x_mid > gap and z_mid < -gap:
                    q_surfaces["cathode_q10_right_bottom"].append(surface)
                elif x_mid < -gap and z_mid > gap:
                    q_surfaces["cathode_q01_left_top"].append(surface)
                elif x_mid > gap and z_mid > gap:
                    q_surfaces["cathode_q11_right_top"].append(surface)
                else:
                    insulated_surfaces.append(surface)
            else:
                insulated_surfaces.append(surface)

        add_physical(3, volumes, "silicon")
        add_physical(2, top_surfaces, "anode")
        for contact, tags in q_surfaces.items():
            add_physical(2, tags, contact)
        add_physical(2, insulated_surfaces, "insulated")

        gmsh.model.mesh.generate(3)
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(mesh_path))
        return {
            "dimension": 3,
            "mesh": str(mesh_path),
            "volume_count": len(volumes),
            "contacts": ["anode", *QPD_CONTACTS],
            "surface_counts": {
                "anode": len(top_surfaces),
                **{contact: len(tags) for contact, tags in q_surfaces.items()},
                "insulated": len(insulated_surfaces),
            },
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
        }
    finally:
        gmsh.finalize()


def import_mesh(config: QPD3DConfig, mesh_path: Path) -> None:
    mesh_name = "qpd_weighting_3d_mesh"
    create_gmsh_mesh(mesh=mesh_name, file=str(mesh_path))
    add_gmsh_region(mesh=mesh_name, gmsh_name="silicon", region=config.region, material="Silicon")
    for contact in ("anode", *QPD_CONTACTS):
        add_gmsh_contact(mesh=mesh_name, gmsh_name=contact, region=config.region, material="metal", name=contact)
    finalize_mesh(mesh=mesh_name)
    create_device(mesh=mesh_name, device=config.device)


def create_laplace_equation(config: QPD3DConfig) -> None:
    CreateSolution(config.device, config.region, "Potential")
    flux = "(Potential@n0-Potential@n1)*EdgeInverseLength"
    CreateEdgeModel(config.device, config.region, "WeightingPotentialFlux", flux)
    CreateEdgeModelDerivatives(config.device, config.region, "WeightingPotentialFlux", flux, "Potential")
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


def solve_for_contact(config: QPD3DConfig, target_contact: str) -> np.ndarray:
    for contact in get_contact_list(device=config.device):
        set_parameter(device=config.device, name=f"{contact}_weighting_bias", value=1.0 if contact == target_contact else 0.0)
    node_count = len(get_node_model_values(device=config.device, region=config.region, name="x"))
    set_node_values(device=config.device, region=config.region, name="Potential", values=[0.0] * node_count)
    solve(
        type="dc",
        absolute_error=config.absolute_error,
        relative_error=config.relative_error,
        maximum_iterations=config.maximum_iterations,
    )
    values = np.asarray(get_node_model_values(device=config.device, region=config.region, name="Potential"), dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite weighting potential for {target_contact}")
    return np.clip(values, 0.0, 1.0)


def quadrant_metrics(contact_values: dict[str, np.ndarray]) -> dict[str, Any]:
    means = {contact: float(np.mean(values)) for contact, values in contact_values.items()}
    total = sum(means.values())
    normalized = {contact: (value / total if total else 0.0) for contact, value in means.items()}
    q00 = normalized["cathode_q00_left_bottom"]
    q10 = normalized["cathode_q10_right_bottom"]
    q01 = normalized["cathode_q01_left_top"]
    q11 = normalized["cathode_q11_right_top"]
    left = q00 + q01
    right = q10 + q11
    bottom = q00 + q10
    top = q01 + q11
    min_q = min(normalized.values())
    max_q = max(normalized.values())
    phase_x = (right - left) / (right + left) if right + left else 0.0
    phase_z = (top - bottom) / (top + bottom) if top + bottom else 0.0
    return {
        "raw_mean_weighting": means,
        "normalized_quadrant_weight": normalized,
        "left_weight": left,
        "right_weight": right,
        "bottom_weight": bottom,
        "top_weight": top,
        "phase_x_weighting": phase_x,
        "phase_z_weighting": phase_z,
        "phase_magnitude_weighting": math.sqrt(phase_x * phase_x + phase_z * phase_z),
        "quadrant_uniformity": min_q / max_q if max_q else None,
    }


def write_svg(path: Path, metrics: dict[str, Any]) -> None:
    weights = metrics["normalized_quadrant_weight"]
    labels = [
        ("Q00 LB", "cathode_q00_left_bottom", "#38bdf8"),
        ("Q10 RB", "cathode_q10_right_bottom", "#22c55e"),
        ("Q01 LT", "cathode_q01_left_top", "#818cf8"),
        ("Q11 RT", "cathode_q11_right_top", "#f59e0b"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" viewBox="0 0 760 280">',
        '<rect width="760" height="280" rx="10" fill="#07131f"/>',
        '<text x="28" y="34" fill="#e2e8f0" font-family="Inter, Arial" font-size="18" font-weight="700">3D QPD Quadrant Weighting Smoke</text>',
        f'<text x="28" y="58" fill="#94a3b8" font-family="Inter, Arial" font-size="12">uniformity {metrics.get("quadrant_uniformity", 0):.5g} · phase x {metrics.get("phase_x_weighting", 0):.5g} · phase z {metrics.get("phase_z_weighting", 0):.5g}</text>',
    ]
    max_weight = max(weights.values()) if weights else 1.0
    for index, (label, key, color) in enumerate(labels):
        y = 88 + index * 38
        value = weights.get(key, 0.0)
        width = 420 * value / max(max_weight, 1.0e-30)
        lines.append(f'<text x="28" y="{y + 17}" fill="#cbd5e1" font-family="Inter, Arial" font-size="13">{label}</text>')
        lines.append(f'<rect x="150" y="{y}" width="{width:.1f}" height="24" rx="4" fill="{color}"/>')
        lines.append(f'<text x="590" y="{y + 17}" fill="#e2e8f0" font-family="Inter, Arial" font-size="13">{value:.6g}</text>')
    lines.append('<text x="28" y="252" fill="#94a3b8" font-family="Inter, Arial" font-size="11">Pure Laplace terminal weighting only; not calibrated 3D drift-diffusion collection.</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: QPD3DConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = config.output_dir / "qpd_weighting_3d.msh"
    mesh_summary = generate_qpd_mesh(config, mesh_path)
    import_mesh(config, mesh_path)
    create_laplace_equation(config)
    contact_values = {contact: solve_for_contact(config, contact) for contact in QPD_CONTACTS}

    x_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="x"), dtype=float)
    y_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="y"), dtype=float)
    z_cm = np.asarray(get_node_model_values(device=config.device, region=config.region, name="z"), dtype=float)
    rows: list[dict[str, Any]] = []
    for index, (x_value, y_value, z_value) in enumerate(zip(x_cm, y_cm, z_cm)):
        row: dict[str, Any] = {
            "node_index": index,
            "x_um": float(x_value * 1.0e4),
            "depth_um": float(y_value * 1.0e4),
            "z_um": float(z_value * 1.0e4),
        }
        for contact, values in contact_values.items():
            row[f"w_{contact}_devsim_laplace"] = float(values[index])
        row["w_qsum_devsim_laplace"] = float(sum(row[f"w_{contact}_devsim_laplace"] for contact in QPD_CONTACTS))
        rows.append(row)

    metrics = quadrant_metrics(contact_values)
    q_sum = np.asarray([row["w_qsum_devsim_laplace"] for row in rows], dtype=float)
    csv_path = config.output_dir / "qpd_weighting_3d.csv"
    dat_path = config.output_dir / "qpd_weighting_3d.dat"
    svg_path = config.output_dir / "qpd_weighting_3d.svg"
    summary_path = config.output_dir / "summary.json"
    write_csv(csv_path, rows)
    write_devices(file=str(dat_path), type="tecplot")
    write_svg(svg_path, metrics)
    summary = {
        "schema": "devsim_qpd_weighting_3d_v1",
        "status": "PASS",
        "devsim_version": version("devsim"),
        "method": "pure_laplace_dirichlet_quadrant_terminal_weighting_3d",
        "mesh": mesh_summary,
        "contacts": list(get_contact_list(device=config.device)),
        "node_count": len(rows),
        "metrics": metrics,
        "qsum_min": float(np.min(q_sum)),
        "qsum_max": float(np.max(q_sum)),
        "qsum_mean": float(np.mean(q_sum)),
        "full_q1q4_weighting_gate": "PASS",
        "full_q1q4_dd_gate": "CHECK",
        "product_accuracy_ready": False,
        "outputs": {
            "mesh": str(mesh_path),
            "csv": str(csv_path),
            "tecplot": str(dat_path),
            "plot_svg": str(svg_path),
            "summary_json": str(summary_path),
        },
        "limitations": [
            "This is a 3D terminal weighting-potential smoke solve, not a 3D drift-diffusion collection solve.",
            "Uniform quadrant weights are trend evidence only; real QPD balance needs calibrated generation, implants, traps, mobility/recombination, and convergence.",
            "The top anode is included as a grounded boundary, so the four cathode weighting potentials do not need to sum to one everywhere.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width-um", type=float, default=2.8)
    parser.add_argument("--z-width-um", type=float, default=2.8)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--mesh-um", type=float, default=0.38)
    parser.add_argument("--fine-mesh-um", type=float, default=0.22)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "devsim_qpd_weighting_3d")
    parser.add_argument("--absolute-error", type=float, default=1.0e-11)
    parser.add_argument("--relative-error", type=float, default=1.0e-11)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    args = parser.parse_args()
    run(QPD3DConfig(**vars(args)))


if __name__ == "__main__":
    main()
