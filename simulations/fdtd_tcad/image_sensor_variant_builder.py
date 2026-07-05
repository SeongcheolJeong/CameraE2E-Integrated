#!/usr/bin/env python3
"""Materialize Image Sensor Pixel Studio design variants.

The design-space registry describes candidate changes. This script turns those
changes into concrete stack/profile/project JSON files and stage-by-stage run
commands without mutating the baseline inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_CONFIG = ROOT / "configs" / "image_sensor_pixel_studio_reference.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "image_sensor_design_variants_reference"
MEEP_ENV = ROOT / ".meep-env"
CRA_CASE_SPECS = [
    ("center", "0:0:0:0:0:0"),
    ("cra10x", "10:0:0.5:0:0:0"),
    ("edge20x", "20:0:1:0:0:0"),
]
CRA_CASES = [case for case, _spec in CRA_CASE_SPECS]
CRA_CASE_STRING = ",".join(f"{case}:{spec}" for case, spec in CRA_CASE_SPECS)


STAGE_ORDER = [
    "meep_fdtd",
    "convergence_gate",
    "gmsh_mesh",
    "devsim_weighting",
    "devsim_electrical",
    "devsim_native_response_sweep",
    "design_viewer",
    "gw_lut",
    "studio",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def rel_from_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_to_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve()).replace(os.sep, "/")


def abs_path(config_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve() if not (config_dir / path).exists() else (config_dir / path).resolve()


def get_path(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def set_path(root: Any, dotted_path: str, value: Any) -> Any:
    current = root
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    leaf = parts[-1]
    if isinstance(current, list):
        old = current[int(leaf)]
        current[int(leaf)] = value
    else:
        old = current[leaf]
        current[leaf] = value
    return old


def split_override_path(path: str) -> tuple[str, str]:
    root, _, child = path.partition(".")
    if root not in {"stack", "profile", "project"} or not child:
        raise ValueError(f"override path must start with stack., profile., or project.: {path}")
    return root, child


def stage_list(required: list[str]) -> list[str]:
    stages = set(required)
    if "gmsh_mesh" in stages:
        stages.add("devsim_weighting")
    if "gw_lut" in stages:
        stages.add("devsim_native_response_sweep")
    if stages & {"meep_fdtd", "gmsh_mesh", "devsim_electrical"}:
        stages.add("design_viewer")
    if stages:
        stages.add("studio")
    return [stage for stage in STAGE_ORDER if stage in stages]


def material_path_source(source_stack_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (source_stack_path.parent / path).resolve()


def normalize_material_paths_for_variant(stack: dict[str, Any], source_stack_path: Path) -> None:
    for material in stack.get("materials", {}).values():
        if isinstance(material, dict) and material.get("nk_table"):
            material["nk_table"] = str(material_path_source(source_stack_path, str(material["nk_table"])))


def geometry_value(profile: dict[str, Any], key: str, fallback: float) -> float:
    try:
        return float(profile.get("geometry", {}).get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def micromamba_command() -> str:
    return f"{Path.home()}/.local/bin/micromamba run -p {MEEP_ENV}"


def command_rows(
    variant_id: str,
    project: dict[str, Any],
    profile: dict[str, Any],
    paths: dict[str, Path],
    stages: list[str],
) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    width = geometry_value(profile, "width_um", 1.4)
    depth = geometry_value(profile, "depth_um", 3.0)
    split_gap = geometry_value(profile, "split_gap_um", 0.04)
    generation_map = paths["generation_map_2d_variant"] if "meep_fdtd" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["inputs"]["generation_map_2d"]
    )
    generation_volume = paths["generation_volume_3d_variant"] if "meep_fdtd" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["inputs"]["generation_volume_3d"]
    )
    mesh_dir = paths["mesh_dir"] if "gmsh_mesh" in stages else ROOT / "runs/gmsh_split_pd_2d_reference_native"
    gmsh_mesh = mesh_dir / "split_pixel_2d.msh"
    weighting_csv = (
        paths["devsim_weighting_dir"] / "weighting_potential_2d.csv"
        if "devsim_weighting" in stages
        else ROOT / "runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv"
    )
    center_summary = paths["devsim_center_dir"] / "summary.json" if "devsim_electrical" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["center"]
    )
    edge_summary = paths["devsim_edge_dir"] / "summary.json" if "devsim_electrical" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["edge20x"]
    )
    native_sweep_manifest = paths["native_sweep_dir"] / "native_response_sweep_manifest.json"
    case_dirs = {
        "center": paths["devsim_center_dir"],
        "cra10x": paths["devsim_cra10x_dir"],
        "edge20x": paths["devsim_edge_dir"],
    }
    case_summary_paths = {
        "center": center_summary,
        "cra10x": paths["devsim_cra10x_dir"] / "summary.json"
        if "devsim_electrical" in stages
        else abs_path(DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["cra10x"]),
        "edge20x": edge_summary,
    }
    split_summary_args = " ".join(
        f"--split-summary {rel_from_root(path)}" for path in case_summary_paths.values()
    )
    report_summary_args = " ".join(
        f"--report-summary {rel_from_root(path)}" for path in case_summary_paths.values()
    )

    if "meep_fdtd" in stages:
        commands.append(
            {
                "stage": "meep_fdtd",
                "label": "Run Meep FDTD generation maps",
                "command": (
                    f"{micromamba_command()} python meep_supercell_lut.py "
                    "--mode split-pd-1x1 --split-mode dual-x --wavelengths-nm 550 "
                    f"--cases {CRA_CASE_STRING} "
                    "--resolution 80 --after-source-time 8 --pml-um 0.45 "
                    "--grid-snap-y nearest --min-feature-pixels 2 --min-si-wavelength-pixels 8 "
                    "--incident-photon-flux-cm2-s 1e20 "
                    f"--stack-config {rel_from_root(paths['stack_config'])} "
                    f"--output-dir {rel_from_root(paths['fdtd_dir'])}"
                ),
            }
        )
    if "convergence_gate" in stages:
        commands.append(
            {
                "stage": "convergence_gate",
                "label": "Run optical convergence gate",
                "command": (
                    f"{micromamba_command()} python run_convergence_sweep.py "
                    "--mode split-pd-1x1 --split-mode dual-x "
                    f"--cases {CRA_CASE_STRING} "
                    "--resolutions 6,8 --after-source-times 2,3 --pml-um 0.45 "
                    f"--stack-config {rel_from_root(paths['stack_config'])} "
                    f"--output-dir {rel_from_root(paths['convergence_dir'])}"
                ),
            }
        )
    if "gmsh_mesh" in stages:
        commands.append(
            {
                "stage": "gmsh_mesh",
                "label": "Build variant Gmsh split-PD mesh",
                "command": (
                    ".tcad-env/bin/python tcad_gmsh_pixel_mesh.py --dimension 2 "
                    f"--measured-profile {rel_from_root(paths['profile_config'])} "
                    "--mesh-um 0.10 --fine-mesh-um 0.025 "
                    f"--output-dir {rel_from_root(paths['mesh_dir'])}"
                ),
            }
        )
    if "devsim_weighting" in stages:
        commands.append(
            {
                "stage": "devsim_weighting",
                "label": "Export DEVSIM Laplace terminal weighting potentials",
                "command": (
                    ".tcad-env/bin/python devsim_weighting_potential_2d.py "
                    f"--mesh {rel_from_root(gmsh_mesh)} "
                    f"--output-dir {rel_from_root(paths['devsim_weighting_dir'])}"
                ),
            }
        )
    if "devsim_electrical" in stages:
        common = (
            ".tcad-env/bin/python devsim_split_pd_2d.py --mesh-source gmsh "
            f"--gmsh-mesh {rel_from_root(gmsh_mesh)} --width-um {width:g} --depth-um {depth:g} "
            f"--split-gap-um {split_gap:g} --generation-map-npz {rel_from_root(generation_map)} "
            "--generation-profile-wavelength-nm 550 --electrical-model profile-ppd "
            f"--measured-profile {rel_from_root(paths['profile_config'])}"
        )
        for case in CRA_CASES:
            commands.append(
                {
                    "stage": "devsim_electrical",
                    "label": f"Run DEVSIM {case} case",
                    "command": (
                        f"{common} --generation-profile-case {case} "
                        f"--output-dir {rel_from_root(case_dirs[case])}"
                    ),
                }
            )
    if "devsim_native_response_sweep" in stages:
        commands.append(
            {
                "stage": "devsim_native_response_sweep",
                "label": "Run native DEVSIM response sweep from FDTD generation map",
                "command": (
                    ".tcad-env/bin/python devsim_native_response_sweep_2d.py "
                    f"--generation-map-npz {rel_from_root(generation_map)} "
                    f"--mesh {rel_from_root(gmsh_mesh)} "
                    f"--measured-profile {rel_from_root(paths['profile_config'])} "
                    f"--width-um {width:g} --depth-um {depth:g} --split-gap-um {split_gap:g} "
                    f"--output-dir {rel_from_root(paths['native_sweep_dir'])} --force"
                ),
            }
        )
    if "design_viewer" in stages:
        commands.append(
            {
                "stage": "design_viewer",
                "label": "Regenerate variant 2D/3D viewers",
                "command": (
                    ".tcad-env/bin/python tcad_design_viewer.py "
                    f"--profile {rel_from_root(paths['profile_config'])} "
                    f"--stack-config {rel_from_root(paths['stack_config'])} "
                    f"--generation-map-npz {rel_from_root(generation_map)} "
                    f"--generation-volume-npz {rel_from_root(generation_volume)} "
                    f"{split_summary_args} "
                    f"{report_summary_args} "
                    f"--output-dir {rel_from_root(paths['design_viewer_dir'])}"
                ),
            }
        )
    if "gw_lut" in stages:
        commands.append(
            {
                "stage": "gw_lut",
                "label": "Regenerate variant G*W and camera diagnostic response",
                "command": (
                    ".tcad-env/bin/python tcad_gw_coupling.py "
                    f"--generation-map-npz {rel_from_root(generation_map)} "
                    f"--split-summary-manifest {rel_from_root(native_sweep_manifest)} "
                    f"--devsim-weighting-csv {rel_from_root(weighting_csv)} "
                    f"--pixel-pitch-um {width:g} "
                    f"--output-dir {rel_from_root(paths['gw_dir'])}"
                ),
            }
        )
    if "studio" in stages:
        commands.append(
            {
                "stage": "studio",
                "label": "Regenerate variant Studio",
                "command": (
                    ".tcad-env/bin/python image_sensor_pixel_studio.py "
                    f"--config {rel_from_root(paths['project_config'])} "
                    f"--output-dir {rel_from_root(paths['studio_dir'])}"
                ),
            }
        )
    for index, command in enumerate(commands):
        command["id"] = f"{variant_id}_{index:02d}_{command['stage']}"
    return commands


def build_variant(
    variant: dict[str, Any],
    project: dict[str, Any],
    stack: dict[str, Any],
    profile: dict[str, Any],
    output_dir: Path,
    source_stack_path: Path | None = None,
) -> dict[str, Any]:
    variant_id = variant["id"]
    variant_dir = output_dir / variant_id
    paths = {
        "variant_dir": variant_dir,
        "stack_config": variant_dir / "inputs" / "stack_config.json",
        "profile_config": variant_dir / "inputs" / "tcad_profile.json",
        "project_config": variant_dir / "inputs" / "studio_project.json",
        "fdtd_dir": variant_dir / "fdtd_generation",
        "generation_map_2d_variant": variant_dir / "fdtd_generation" / "tcad_generation_map_2d.npz",
        "generation_volume_3d_variant": variant_dir / "fdtd_generation" / "tcad_generation_volume_3d.npz",
        "convergence_dir": variant_dir / "convergence",
        "mesh_dir": variant_dir / "gmsh_mesh",
        "devsim_weighting_dir": variant_dir / "devsim_weighting",
        "devsim_center_dir": variant_dir / "devsim_center",
        "devsim_cra10x_dir": variant_dir / "devsim_cra10x",
        "devsim_edge_dir": variant_dir / "devsim_edge20x",
        "native_sweep_dir": variant_dir / "devsim_native_response_sweep",
        "design_viewer_dir": variant_dir / "design_viewer",
        "gw_dir": variant_dir / "gw_coupling",
        "studio_dir": variant_dir / "studio",
        "variant_manifest": variant_dir / "variant_manifest.json",
        "run_plan": variant_dir / "run_plan.sh",
    }
    variant_stack = deepcopy(stack)
    variant_profile = deepcopy(profile)
    variant_project = deepcopy(project)
    source_stack = source_stack_path or abs_path(DEFAULT_PROJECT_CONFIG.parent, project["inputs"]["stack_config"])
    normalize_material_paths_for_variant(variant_stack, source_stack)
    roots = {"stack": variant_stack, "profile": variant_profile, "project": variant_project}

    changes: list[dict[str, Any]] = []
    errors: list[str] = []
    for path, value in variant.get("parameter_overrides", {}).items():
        try:
            root_name, child_path = split_override_path(path)
            old_value = get_path(roots[root_name], child_path)
            set_path(roots[root_name], child_path, value)
            changes.append(
                {
                    "path": path,
                    "root": root_name,
                    "old_value": old_value,
                    "new_value": value,
                    "changed": old_value != value,
                }
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    stages = stage_list(list(variant.get("requires_rerun", [])))
    generation_map = paths["generation_map_2d_variant"] if "meep_fdtd" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["inputs"]["generation_map_2d"]
    )
    generation_volume = paths["generation_volume_3d_variant"] if "meep_fdtd" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["inputs"]["generation_volume_3d"]
    )
    weighting_csv = (
        paths["devsim_weighting_dir"] / "weighting_potential_2d.csv"
        if "devsim_weighting" in stages
        else ROOT / "runs/devsim_weighting_potential_2d_reference/weighting_potential_2d.csv"
    )
    center_summary = paths["devsim_center_dir"] / "summary.json" if "devsim_electrical" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["center"]
    )
    edge_summary = paths["devsim_edge_dir"] / "summary.json" if "devsim_electrical" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["edge20x"]
    )
    cra10x_summary = paths["devsim_cra10x_dir"] / "summary.json" if "devsim_electrical" in stages else abs_path(
        DEFAULT_PROJECT_CONFIG.parent, project["native_split_runs"]["cra10x"]
    )
    native_sweep_manifest = paths["native_sweep_dir"] / "native_response_sweep_manifest.json"

    variant_project["inputs"]["stack_config"] = rel_from_root(paths["stack_config"])
    variant_project["inputs"]["tcad_profile"] = rel_from_root(paths["profile_config"])
    variant_project["inputs"]["generation_map_2d"] = rel_from_root(generation_map)
    variant_project["inputs"]["generation_volume_3d"] = rel_from_root(generation_volume)
    if "design_viewer" in stages:
        variant_project["inputs"]["design_viewer_manifest"] = rel_from_root(paths["design_viewer_dir"] / "manifest.json")
        variant_project["views"]["cross_section_2d"] = rel_from_root(paths["design_viewer_dir"] / "viewers/cross_section_2d.html")
        variant_project["views"]["geometry_3d"] = rel_from_root(paths["design_viewer_dir"] / "viewers/geometry_3d.html")
        variant_project["views"]["parameter_report"] = rel_from_root(
            paths["design_viewer_dir"] / "reports/parameter_sweep_comparison.html"
        )
        variant_project["views"]["parameter_report_csv"] = rel_from_root(
            paths["design_viewer_dir"] / "reports/parameter_sweep_comparison.csv"
        )
    if "gw_lut" in stages:
        variant_project["views"]["gw_coupling_manifest"] = rel_from_root(paths["gw_dir"] / "gw_coupling_manifest.json")
        variant_project["views"]["gw_coupling_report"] = rel_from_root(paths["gw_dir"] / "gw_coupling_report.html")
        variant_project["views"]["gw_coupling_summary_csv"] = rel_from_root(paths["gw_dir"] / "gw_coupling_summary.csv")
        variant_project["views"]["camera_system_diagnostic_report"] = rel_from_root(
            paths["gw_dir"] / "camera_system_diagnostic_report.html"
        )
        variant_project["views"]["camera_system_lut_report"] = rel_from_root(paths["gw_dir"] / "camera_system_lut_report.html")
        variant_project["views"]["native_response_sweep_manifest"] = rel_from_root(native_sweep_manifest)
        variant_project["views"]["native_devsim_response"] = rel_from_root(
            paths["gw_dir"] / "camera_system_native_devsim_response.json"
        )
    variant_project["native_split_runs"]["center"] = rel_from_root(center_summary)
    variant_project["native_split_runs"]["cra10x"] = rel_from_root(cra10x_summary)
    variant_project["native_split_runs"]["edge20x"] = rel_from_root(edge_summary)
    if "devsim_native_response_sweep" in stages:
        for case in CRA_CASES:
            variant_project["native_split_runs"][case] = rel_from_root(
                paths["native_sweep_dir"] / "cases" / f"{case}_wl550nm" / "summary.json"
            )
    variant_project["project"]["name"] = f"{project['project']['name']} - {variant.get('label', variant_id)}"
    variant_project["project"]["short_name"] = f"{project['project']['short_name']}_{variant_id}"
    variant_project["project"]["status"] = "variant_plan_ready_accuracy_not_ready"

    write_json(paths["stack_config"], variant_stack)
    write_json(paths["profile_config"], variant_profile)
    write_json(paths["project_config"], variant_project)

    commands = command_rows(variant_id, project, variant_profile, paths, stages)
    run_plan_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        "",
    ]
    run_plan_lines.extend(command["command"] for command in commands)
    write_text_if_changed(paths["run_plan"], "\n".join(run_plan_lines) + "\n")

    warnings = list(variant.get("risks", []))
    if errors:
        warnings.append("One or more overrides failed to apply; inspect errors before running.")
    if not stages:
        warnings.append("No rerun stages are requested for this baseline/reference variant.")

    manifest = {
        "schema": "image_sensor_variant_manifest_v1",
        "id": variant_id,
        "label": variant.get("label", variant_id),
        "status": variant.get("status", ""),
        "goal": variant.get("goal", ""),
        "expected_effect": variant.get("expected_effect", ""),
        "product_lut_ready": False,
        "output_dir": str(variant_dir),
        "parameter_overrides": variant.get("parameter_overrides", {}),
        "applied_changes": changes,
        "errors": errors,
        "warnings": warnings,
        "required_stages": stages,
        "generated_files": {
            "stack_config": str(paths["stack_config"]),
            "tcad_profile": str(paths["profile_config"]),
            "project_config": str(paths["project_config"]),
            "run_plan": str(paths["run_plan"]),
            "variant_manifest": str(paths["variant_manifest"]),
        },
        "planned_outputs": {
            "fdtd_generation": str(paths["fdtd_dir"]),
            "gmsh_mesh": str(paths["mesh_dir"]),
            "devsim_weighting": str(paths["devsim_weighting_dir"]),
            "devsim_center": str(paths["devsim_center_dir"]),
            "devsim_cra10x": str(paths["devsim_cra10x_dir"]),
            "devsim_edge20x": str(paths["devsim_edge_dir"]),
            "devsim_native_response_sweep": str(paths["native_sweep_dir"]),
            "design_viewer": str(paths["design_viewer_dir"]),
            "gw_coupling": str(paths["gw_dir"]),
            "studio": str(paths["studio_dir"]),
        },
        "commands": commands,
        "notes": [
            "Generated configs are isolated copies; baseline inputs are not mutated.",
            "Run plans are not accuracy evidence until the listed stages are executed and their convergence/accuracy gates pass.",
        ],
    }
    write_json(paths["variant_manifest"], manifest)
    return manifest


def run(project_config: Path, output_dir: Path) -> dict[str, Any]:
    project = read_json(project_config)
    config_dir = project_config.parent
    stack_path = abs_path(config_dir, project["inputs"]["stack_config"])
    profile_path = abs_path(config_dir, project["inputs"]["tcad_profile"])
    design_space_path = abs_path(config_dir, project["inputs"]["design_space"])
    stack = read_json(stack_path)
    profile = read_json(profile_path)
    design_space = read_json(design_space_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        build_variant(variant, project, stack, profile, output_dir, stack_path)
        for variant in design_space.get("variants", [])
    ]
    root_manifest = {
        "schema": "image_sensor_variant_run_manifest_v1",
        "source_project_config": str(project_config),
        "source_stack_config": str(stack_path),
        "source_tcad_profile": str(profile_path),
        "source_design_space": str(design_space_path),
        "output_dir": str(output_dir),
        "variant_count": len(variants),
        "variants": variants,
        "summary": {
            "candidate_count": sum(1 for item in variants if item.get("status") != "simulated_reference"),
            "product_lut_ready": False,
            "all_override_errors": [
                {"variant": item["id"], "errors": item["errors"]}
                for item in variants
                if item.get("errors")
            ],
        },
    }
    manifest_path = output_dir / "variant_run_manifest.json"
    write_json(manifest_path, root_manifest)
    print(json.dumps({**root_manifest, "variants": [v["id"] for v in variants]}, indent=2, ensure_ascii=False))
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
