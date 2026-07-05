#!/usr/bin/env python3
"""Generate a local Image Sensor Pixel Design Studio dashboard.

The studio is a browser-based project shell around the existing open-source
Meep/Gmsh/DEVSIM outputs. It intentionally mirrors workflow ideas from
professional optical/electrical tools without copying their UI.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"parse_error": True, "raw": line})
    return rows[-limit:]


def rel_from(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_to_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve()).replace(os.sep, "/")


def abs_path(config_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve() if not (config_dir / path).exists() else (config_dir / path).resolve()


def file_status(path: Path) -> dict[str, Any]:
    return {
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok", "exists"}


def stable_id(*parts: Any) -> str:
    raw = "_".join(str(part) for part in parts if str(part).strip())
    result = []
    for char in raw.lower():
        result.append(char if char.isalnum() else "_")
    compact = "_".join(part for part in "".join(result).split("_") if part)
    return compact or "item"


def result_viewer_mode(row: dict[str, Any]) -> str:
    kind = str(row.get("dataset_kind") or row.get("kind") or "").lower()
    dim = str(row.get("dimensionality", "")).lower()
    suffix = Path(str(row.get("path", ""))).suffix.lower()
    if kind in {"viewer_report", "markdown_report"} or suffix in {".html", ".htm", ".md"}:
        return "report"
    if kind == "image_plot" or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if kind in {"table", "array_dataset"} or suffix in {".csv", ".tsv", ".npz", ".json"}:
        return "table/data"
    if kind == "mesh_dataset" or suffix in {".vtk", ".vtu", ".msh"}:
        return "mesh/field"
    if "3d" in dim:
        return "3d"
    if "2d" in dim:
        return "2d"
    return "file"


def build_result_groups(
    dataset_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    def primary_priority(item: dict[str, Any], *, viewer: bool) -> int:
        text = " ".join(
            str(item.get(key, ""))
            for key in ("dataset_id", "path", "relative_path", "role")
        ).lower()
        if "camera_system_research_lut_report" in text:
            return 0 if viewer else 1
        if "camera_system_research_lut" in text:
            return 2
        if "camera_system_native_devsim" in text:
            return 3
        if "camera_system_diagnostic" in text:
            return 4
        if "camera_system_lut" in text:
            return 9
        return 5

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in dataset_rows:
        solver = row.get("solver") or "Project"
        role = row.get("role") or "artifact"
        key = (solver, role)
        path = Path(row.get("path", ""))
        native = boolish(row.get("native_mesh"))
        exists = boolish(row.get("exists"))
        viewer_open = str(row.get("viewer", "")).lower() == "open"
        item = {
            "dataset_id": row.get("dataset_id", ""),
            "kind": row.get("dataset_kind", ""),
            "role": role,
            "dimensionality": row.get("dimensionality", ""),
            "native": native,
            "exists": exists,
            "size_bytes": int(row["size_bytes"]) if str(row.get("size_bytes", "")).isdigit() else None,
            "path": str(path),
            "relative_path": rel_to_output(path, output_dir) if row.get("path") else "",
            "viewer": row.get("viewer", ""),
            "viewer_mode": result_viewer_mode(row),
        }
        group = groups.setdefault(
            key,
            {
                "id": stable_id(solver, role),
                "object_label": solver,
                "object_kind": "Result Object",
                "result_role": role,
                "result_label": role.replace("_", " ").title(),
                "datasets": [],
                "dataset_count": 0,
                "existing_count": 0,
                "native_count": 0,
                "derived_count": 0,
                "viewer_count": 0,
                "kinds": set(),
                "dimensionalities": set(),
                "viewer_modes": set(),
                "primary_viewer": None,
                "primary_viewer_priority": 999,
                "primary_path": None,
                "primary_path_priority": 999,
                "product_lut_ready": False,
            },
        )
        group["datasets"].append(item)
        group["dataset_count"] += 1
        group["existing_count"] += 1 if exists else 0
        group["native_count"] += 1 if native else 0
        group["derived_count"] += 0 if native else 1
        group["viewer_count"] += 1 if viewer_open else 0
        if item["kind"]:
            group["kinds"].add(item["kind"])
        if item["dimensionality"]:
            group["dimensionalities"].add(item["dimensionality"])
        group["viewer_modes"].add(item["viewer_mode"])
        viewer_priority = primary_priority(item, viewer=True)
        path_priority = primary_priority(item, viewer=False)
        if (
            viewer_open
            and item["relative_path"]
            and viewer_priority < group["primary_viewer_priority"]
        ):
            group["primary_viewer"] = item["relative_path"]
            group["primary_viewer_priority"] = viewer_priority
        if item["relative_path"] and path_priority < group["primary_path_priority"]:
            group["primary_path"] = item["relative_path"]
            group["primary_path_priority"] = path_priority

    normalized: list[dict[str, Any]] = []
    for group in groups.values():
        group["kinds"] = sorted(group["kinds"])
        group["dimensionalities"] = sorted(group["dimensionalities"])
        group["viewer_modes"] = sorted(group["viewer_modes"])
        group["native_state"] = (
            "native"
            if group["native_count"] == group["dataset_count"]
            else "mixed"
            if group["native_count"]
            else "derived"
        )
        has_research_lut = any(
            "camera_system_research_lut" in str(item.get("dataset_id", "")).lower()
            or "camera_system_research_lut" in str(item.get("path", "")).lower()
            for item in group["datasets"]
        )
        group["readiness"] = "research-ready" if has_research_lut else "product-blocked"
        group["readiness_reason"] = (
            "Native DEVSIM research LUT is available; product LUT export remains blocked by Accuracy Gate."
            if has_research_lut
            else "Framework artifact only; product LUT export remains blocked by Accuracy Gate."
        )
        group.pop("primary_viewer_priority", None)
        group.pop("primary_path_priority", None)
        normalized.append(group)
    normalized.sort(key=lambda item: (item["object_label"], item["result_role"]))
    return normalized


def lookup_dotted_path(sources: dict[str, Any], dotted_path: str) -> Any:
    current: Any = sources
    for part in dotted_path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        else:
            return None
    return current


def enrich_design_space(
    design_space: dict[str, Any],
    project: dict[str, Any],
    stack: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if not design_space:
        return {}
    enriched = json.loads(json.dumps(design_space))
    sources = {"project": project, "stack": stack, "profile": profile}
    flat_parameters: list[dict[str, Any]] = []
    for group in enriched.get("parameter_groups", []):
        for parameter in group.get("parameters", []):
            parameter["group_id"] = group.get("id", "")
            parameter["group_label"] = group.get("label", "")
            parameter["owner"] = group.get("owner", "")
            parameter["current_value"] = lookup_dotted_path(sources, parameter.get("path", ""))
            parameter["value_status"] = "resolved" if parameter["current_value"] is not None else "missing"
            flat_parameters.append(parameter)
    enriched["flat_parameters"] = flat_parameters
    enriched["wired_parameter_count"] = sum(1 for item in flat_parameters if item.get("wired_to_solver"))
    enriched["metadata_parameter_count"] = sum(1 for item in flat_parameters if not item.get("wired_to_solver"))
    enriched["candidate_variant_count"] = sum(
        1 for item in enriched.get("variants", []) if item.get("status") != "simulated_reference"
    )
    return enriched


def summarize_split(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    left = float(summary.get("left_photo_delta_a_per_cm", 0.0))
    right = float(summary.get("right_photo_delta_a_per_cm", 0.0))
    transport = summary.get("transport_summary", {})
    return {
        "run": path.parent.name,
        "case": summary.get("config", {}).get("generation_profile_case", ""),
        "mesh_source": summary.get("mesh_source", summary.get("config", {}).get("mesh_source", "")),
        "node_count": summary.get("node_count"),
        "generation_source": summary.get("generation_source", ""),
        "electrical_model": summary.get("electrical_model", ""),
        "photo_signal_carrier": summary.get("photo_signal_carrier", "electron"),
        "left_photo_delta_a_per_cm": left,
        "right_photo_delta_a_per_cm": right,
        "total_photo_delta_a_per_cm": left + right,
        "photo_split_phase_x_proxy": summary.get("photo_split_phase_x_proxy"),
        "terminal_balance_illuminated_a_per_cm": summary.get("terminal_current_balance_illuminated_a_per_cm"),
        "device_tecplot": summary.get("outputs", {}).get("device_tecplot", ""),
        "summary_json": str(path),
        "transport_model": transport.get("model", ""),
        "transport_measured": bool(transport.get("measured", False)),
        "transport_calibrated": bool(transport.get("calibrated", False)),
        "electron_mobility_min_cm2_v_s": transport.get("electron_mobility_min_cm2_v_s"),
        "electron_mobility_max_cm2_v_s": transport.get("electron_mobility_max_cm2_v_s"),
        "hole_mobility_min_cm2_v_s": transport.get("hole_mobility_min_cm2_v_s"),
        "hole_mobility_max_cm2_v_s": transport.get("hole_mobility_max_cm2_v_s"),
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
        "field_mobility_enabled": bool(transport.get("field_mobility_enabled", False)),
        "field_mobility_model": transport.get("field_mobility_model", ""),
        "tau_n_min_s": transport.get("tau_n_min_s"),
        "tau_n_max_s": transport.get("tau_n_max_s"),
        "tau_p_min_s": transport.get("tau_p_min_s"),
        "tau_p_max_s": transport.get("tau_p_max_s"),
        "transport_summary": transport,
    }


def accuracy_summary(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {
            "framework_ready": False,
            "accuracy_ready": False,
            "checks": [],
            "blocking": [],
        }
    data = read_json(path)
    checks = data.get("checks", [])
    blocking = [
        {
            "name": check.get("name"),
            "status": check.get("status"),
            "details": check.get("details"),
        }
        for check in checks
        if check.get("accuracy_blocking") and check.get("status") in {"FAIL", "WARN"}
    ]
    return {
        "framework_ready": bool(data.get("framework_ready")),
        "accuracy_ready": bool(data.get("accuracy_ready")),
        "accuracy_blocking_failure_count": data.get("accuracy_blocking_failure_count", 0),
        "framework_blocking_failure_count": data.get("framework_blocking_failure_count", 0),
        "checks": checks,
        "blocking": blocking,
        "path": str(path),
    }


def object_tree(
    project: dict[str, Any],
    stack: dict[str, Any],
    profile: dict[str, Any],
    design_space: dict[str, Any],
) -> list[dict[str, Any]]:
    geometry = profile.get("geometry", {})
    stack_geometry = stack.get("geometry_um", {})
    nodes = [
        {
            "id": "project",
            "label": "Project",
            "kind": "Project",
            "children": [
                {
                    "id": "project_status",
                    "label": project.get("project", {}).get("short_name", "project"),
                    "kind": "Status",
                    "data": project.get("project", {}),
                }
            ],
        },
        {
            "id": "process_stack",
            "label": "Process Stack",
            "kind": "Stack",
            "data": {
                "geometry_um": stack_geometry,
                "shield": stack.get("shield", {}),
                "materials": stack.get("materials", {}),
                "accuracy_notes": stack.get("accuracy_notes", []),
            },
            "children": [
                {"id": f"stack_{name}", "label": name, "kind": "Material", "data": value}
                for name, value in stack.get("materials", {}).items()
            ],
        },
        {
            "id": "pixel_geometry",
            "label": "Pixel Geometry",
            "kind": "Geometry",
            "data": geometry,
            "children": [
                {
                    "id": "tg_geometry",
                    "label": "Transfer Gate",
                    "kind": "Feature",
                    "data": geometry.get("transfer_gate", {}),
                },
                {
                    "id": "fd_geometry",
                    "label": "Floating Diffusion",
                    "kind": "Feature",
                    "data": geometry.get("floating_diffusion", {}),
                },
            ],
        },
        {
            "id": "implants",
            "label": "Implants / DTI",
            "kind": "Electrical Profile",
            "children": [
                {
                    "id": f"implant_{index}",
                    "label": item.get("name", f"implant_{index}"),
                    "kind": "Implant",
                    "data": item,
                }
                for index, item in enumerate(profile.get("implants", []))
            ],
        },
        {
            "id": "electrical_features",
            "label": "TG / FD / Interface",
            "kind": "Electrical Features",
            "children": [
                {
                    "id": f"feature_{index}",
                    "label": item.get("name", f"feature_{index}"),
                    "kind": item.get("role", item.get("type", "Feature")),
                    "data": item,
                }
                for index, item in enumerate(profile.get("electrical_features", []))
            ]
            + [
                {
                    "id": f"interface_{index}",
                    "label": item.get("name", f"interface_{index}"),
                    "kind": "Interface",
                    "data": item,
                }
                for index, item in enumerate(profile.get("interfaces", []))
            ],
        },
        {
            "id": "workflow",
            "label": "Solvers / Coupling",
            "kind": "Workflow",
            "data": project.get("workflow", {}),
            "children": [
                {"id": "optical_solver", "label": "Optical FDTD", "kind": "Solver", "data": project.get("workflow", {}).get("optical", {})},
                {"id": "electrical_solver", "label": "Electrical DEVSIM", "kind": "Solver", "data": project.get("workflow", {}).get("electrical", {})},
                {
                    "id": "gw_coupling",
                    "label": "G * W Coupling",
                    "kind": "Coupling",
                    "data": project.get("workflow", {}).get("coupling_next", {}),
                },
            ],
        },
    ]
    if design_space:
        nodes.append(
            {
                "id": "design_space",
                "label": "Design Parameters",
                "kind": "Design Space",
                "data": {
                    "name": design_space.get("name"),
                    "wired_parameter_count": design_space.get("wired_parameter_count"),
                    "metadata_parameter_count": design_space.get("metadata_parameter_count"),
                    "candidate_variant_count": design_space.get("candidate_variant_count"),
                },
                "children": [
                    {
                        "id": f"param_group_{group.get('id', index)}",
                        "label": group.get("label", f"group_{index}"),
                        "kind": "Parameter Group",
                        "data": {
                            "owner": group.get("owner", ""),
                            "parameter_count": len(group.get("parameters", [])),
                        },
                        "children": [
                            {
                                "id": f"param_{item.get('id', child_index)}",
                                "label": item.get("label", item.get("id", f"parameter_{child_index}")),
                                "kind": "Solver Parameter" if item.get("wired_to_solver") else "Metadata Parameter",
                                "data": item,
                            }
                            for child_index, item in enumerate(group.get("parameters", []))
                        ],
                    }
                    for index, group in enumerate(design_space.get("parameter_groups", []))
                ]
                + [
                    {
                        "id": "design_variants",
                        "label": "Design Variants",
                        "kind": "Variant Set",
                        "data": {"variant_count": len(design_space.get("variants", []))},
                        "children": [
                            {
                                "id": f"variant_{item.get('id', index)}",
                                "label": item.get("label", item.get("id", f"variant_{index}")),
                                "kind": "Variant",
                                "data": item,
                            }
                            for index, item in enumerate(design_space.get("variants", []))
                        ],
                    }
                ],
            }
        )
    return nodes


def flatten_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], depth: int) -> None:
        item = dict(node)
        item["depth"] = depth
        item.pop("children", None)
        rows.append(item)
        for child in node.get("children", []):
            visit(child, depth + 1)

    for node in nodes:
        visit(node, 0)
    return rows


def build_payload(config_path: Path, output_dir: Path) -> dict[str, Any]:
    project = read_json(config_path)
    config_dir = config_path.parent
    inputs = project.get("inputs", {})
    views = project.get("views", {})

    stack_path = abs_path(config_dir, inputs["stack_config"])
    profile_path = abs_path(config_dir, inputs["tcad_profile"])
    design_space_path = abs_path(config_dir, inputs.get("design_space", "")) if inputs.get("design_space") else None
    manifest_path = abs_path(config_dir, inputs["design_viewer_manifest"])
    accuracy_path = abs_path(config_dir, inputs.get("accuracy_gate", "")) if inputs.get("accuracy_gate") else None
    optical_stack_summary_path = (
        abs_path(config_dir, inputs.get("optical_stack_summary", ""))
        if inputs.get("optical_stack_summary")
        else None
    )
    report_csv_path = abs_path(config_dir, views["parameter_report_csv"])
    gw_manifest_path = abs_path(config_dir, views["gw_coupling_manifest"]) if views.get("gw_coupling_manifest") else None
    gw_report_path = abs_path(config_dir, views["gw_coupling_report"]) if views.get("gw_coupling_report") else None
    gw_summary_csv_path = (
        abs_path(config_dir, views["gw_coupling_summary_csv"]) if views.get("gw_coupling_summary_csv") else None
    )
    camera_diagnostic_report_path = (
        abs_path(config_dir, views["camera_system_diagnostic_report"])
        if views.get("camera_system_diagnostic_report")
        else None
    )
    camera_research_lut_report_path = (
        abs_path(config_dir, views["camera_system_research_lut_report"])
        if views.get("camera_system_research_lut_report")
        else None
    )
    camera_lut_report_path = (
        abs_path(config_dir, views["camera_system_lut_report"]) if views.get("camera_system_lut_report") else None
    )
    crosstalk_manifest_path = (
        abs_path(config_dir, views["crosstalk_kernel_manifest"]) if views.get("crosstalk_kernel_manifest") else None
    )
    crosstalk_output_kernel_csv_path = (
        abs_path(config_dir, views["crosstalk_output_kernel_csv"]) if views.get("crosstalk_output_kernel_csv") else None
    )
    crosstalk_raw_pd_kernel_csv_path = (
        abs_path(config_dir, views["crosstalk_raw_pd_kernel_csv"]) if views.get("crosstalk_raw_pd_kernel_csv") else None
    )
    crosstalk_summary_csv_path = (
        abs_path(config_dir, views["crosstalk_kernel_summary_csv"]) if views.get("crosstalk_kernel_summary_csv") else None
    )
    crosstalk_heatmap_path = (
        abs_path(config_dir, views["crosstalk_kernel_heatmap"]) if views.get("crosstalk_kernel_heatmap") else None
    )
    crosstalk_convergence_path = (
        abs_path(config_dir, views["crosstalk_convergence_report"]) if views.get("crosstalk_convergence_report") else None
    )
    crosstalk_xsection_manifest_path = (
        abs_path(config_dir, views["crosstalk_xsection_manifest"]) if views.get("crosstalk_xsection_manifest") else None
    )
    crosstalk_xsection_output_csv_path = (
        abs_path(config_dir, views["crosstalk_xsection_output_csv"]) if views.get("crosstalk_xsection_output_csv") else None
    )
    crosstalk_xsection_raw_pd_csv_path = (
        abs_path(config_dir, views["crosstalk_xsection_raw_pd_csv"]) if views.get("crosstalk_xsection_raw_pd_csv") else None
    )
    crosstalk_xsection_summary_csv_path = (
        abs_path(config_dir, views["crosstalk_xsection_summary_csv"]) if views.get("crosstalk_xsection_summary_csv") else None
    )
    crosstalk_xsection_plot_path = (
        abs_path(config_dir, views["crosstalk_xsection_plot"]) if views.get("crosstalk_xsection_plot") else None
    )
    crosstalk_xsection_convergence_path = (
        abs_path(config_dir, views["crosstalk_xsection_convergence"]) if views.get("crosstalk_xsection_convergence") else None
    )
    variant_manifest_path = (
        abs_path(config_dir, views["variant_run_manifest"]) if views.get("variant_run_manifest") else None
    )
    variant_comparison_report_path = (
        abs_path(config_dir, views["variant_comparison_report"]) if views.get("variant_comparison_report") else None
    )
    variant_comparison_csv_path = (
        abs_path(config_dir, views["variant_comparison_csv"]) if views.get("variant_comparison_csv") else None
    )
    variant_comparison_json_path = (
        abs_path(config_dir, views["variant_comparison_json"]) if views.get("variant_comparison_json") else None
    )
    run_manager_report_path = (
        abs_path(config_dir, views["run_manager_report"]) if views.get("run_manager_report") else None
    )
    run_manager_csv_path = (
        abs_path(config_dir, views["run_manager_csv"]) if views.get("run_manager_csv") else None
    )
    run_manager_json_path = (
        abs_path(config_dir, views["run_manager_json"]) if views.get("run_manager_json") else None
    )
    dataset_catalog_report_path = (
        abs_path(config_dir, views["dataset_catalog_report"]) if views.get("dataset_catalog_report") else None
    )
    dataset_catalog_csv_path = (
        abs_path(config_dir, views["dataset_catalog_csv"]) if views.get("dataset_catalog_csv") else None
    )
    dataset_catalog_json_path = (
        abs_path(config_dir, views["dataset_catalog_json"]) if views.get("dataset_catalog_json") else None
    )
    orchestrator_last_run_path = (
        abs_path(config_dir, views["orchestrator_last_run"]) if views.get("orchestrator_last_run") else None
    )
    orchestrator_history_path = (
        abs_path(config_dir, views["orchestrator_history"]) if views.get("orchestrator_history") else None
    )

    stack = read_json(stack_path)
    profile = read_json(profile_path)
    design_space_raw = read_json(design_space_path) if design_space_path and design_space_path.exists() else {}
    design_space = enrich_design_space(design_space_raw, project, stack, profile)
    design_manifest = read_json(manifest_path)
    report_rows = read_csv_rows(report_csv_path)
    accuracy = accuracy_summary(accuracy_path)
    optical_stack_evidence = (
        read_json(optical_stack_summary_path)
        if optical_stack_summary_path and optical_stack_summary_path.exists()
        else {}
    )
    gw_coupling = read_json(gw_manifest_path) if gw_manifest_path and gw_manifest_path.exists() else {}
    gw_summary_rows = read_csv_rows(gw_summary_csv_path) if gw_summary_csv_path and gw_summary_csv_path.exists() else []
    crosstalk_kernel = (
        read_json(crosstalk_manifest_path) if crosstalk_manifest_path and crosstalk_manifest_path.exists() else {}
    )
    crosstalk_output_rows = (
        read_csv_rows(crosstalk_output_kernel_csv_path)
        if crosstalk_output_kernel_csv_path and crosstalk_output_kernel_csv_path.exists()
        else []
    )
    crosstalk_raw_pd_rows = (
        read_csv_rows(crosstalk_raw_pd_kernel_csv_path)
        if crosstalk_raw_pd_kernel_csv_path and crosstalk_raw_pd_kernel_csv_path.exists()
        else []
    )
    crosstalk_summary_rows = (
        read_csv_rows(crosstalk_summary_csv_path)
        if crosstalk_summary_csv_path and crosstalk_summary_csv_path.exists()
        else []
    )
    crosstalk_convergence = (
        read_json(crosstalk_convergence_path)
        if crosstalk_convergence_path and crosstalk_convergence_path.exists()
        else {}
    )
    crosstalk_xsection = (
        read_json(crosstalk_xsection_manifest_path)
        if crosstalk_xsection_manifest_path and crosstalk_xsection_manifest_path.exists()
        else {}
    )
    crosstalk_xsection_output_rows = (
        read_csv_rows(crosstalk_xsection_output_csv_path)
        if crosstalk_xsection_output_csv_path and crosstalk_xsection_output_csv_path.exists()
        else []
    )
    crosstalk_xsection_raw_pd_rows = (
        read_csv_rows(crosstalk_xsection_raw_pd_csv_path)
        if crosstalk_xsection_raw_pd_csv_path and crosstalk_xsection_raw_pd_csv_path.exists()
        else []
    )
    crosstalk_xsection_summary_rows = (
        read_csv_rows(crosstalk_xsection_summary_csv_path)
        if crosstalk_xsection_summary_csv_path and crosstalk_xsection_summary_csv_path.exists()
        else []
    )
    crosstalk_xsection_convergence = (
        read_json(crosstalk_xsection_convergence_path)
        if crosstalk_xsection_convergence_path and crosstalk_xsection_convergence_path.exists()
        else {}
    )
    variant_manifest = read_json(variant_manifest_path) if variant_manifest_path and variant_manifest_path.exists() else {}
    variant_comparison = (
        read_json(variant_comparison_json_path)
        if variant_comparison_json_path and variant_comparison_json_path.exists()
        else {}
    )
    variant_comparison_rows = (
        read_csv_rows(variant_comparison_csv_path)
        if variant_comparison_csv_path and variant_comparison_csv_path.exists()
        else []
    )
    run_manager = (
        read_json(run_manager_json_path) if run_manager_json_path and run_manager_json_path.exists() else {}
    )
    run_stage_rows = (
        read_csv_rows(run_manager_csv_path) if run_manager_csv_path and run_manager_csv_path.exists() else []
    )
    dataset_catalog = (
        read_json(dataset_catalog_json_path)
        if dataset_catalog_json_path and dataset_catalog_json_path.exists()
        else {}
    )
    dataset_catalog_rows = (
        read_csv_rows(dataset_catalog_csv_path)
        if dataset_catalog_csv_path and dataset_catalog_csv_path.exists()
        else []
    )
    result_groups = build_result_groups(dataset_catalog_rows, output_dir)
    orchestrator_last_run = (
        read_json(orchestrator_last_run_path)
        if orchestrator_last_run_path and orchestrator_last_run_path.exists()
        else {}
    )
    orchestrator_history = (
        read_jsonl_rows(orchestrator_history_path)
        if orchestrator_history_path and orchestrator_history_path.exists()
        else []
    )

    native_runs = []
    for case, value in project.get("native_split_runs", {}).items():
        path = abs_path(config_dir, value)
        if path.exists():
            row = summarize_split(path)
            row["case_key"] = case
            native_runs.append(row)

    results: list[dict[str, Any]] = []
    for key, value in design_manifest.get("exports", {}).items():
        path = Path(value)
        results.append(
            {
                "id": key,
                "kind": path.suffix.lstrip(".").upper() or "FILE",
                "label": key,
                "path": str(path),
                "relative_path": rel_to_output(path, output_dir),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "native_mesh": "native_split2d" in key or "gmsh_reference" in key,
            }
        )
    for case, value in project.get("native_split_runs", {}).items():
        path = abs_path(config_dir, value)
        results.append(
            {
                "id": f"native_summary_{case}",
                "kind": "JSON",
                "label": f"native split summary {case}",
                "path": str(path),
                "relative_path": rel_to_output(path, output_dir),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "native_mesh": True,
            }
        )
    if gw_manifest_path:
        for key, path in {
            "gw_coupling_manifest": gw_manifest_path,
            "gw_coupling_report": gw_report_path,
            "gw_coupling_summary_csv": gw_summary_csv_path,
            "camera_system_diagnostic_report": camera_diagnostic_report_path,
            "camera_system_research_lut_report": camera_research_lut_report_path,
            "camera_system_lut_report": camera_lut_report_path,
        }.items():
            if path:
                results.append(
                    {
                        "id": key,
                        "kind": path.suffix.lstrip(".").upper() or "FILE",
                        "label": key.replace("_", " "),
                        "path": str(path),
                        "relative_path": rel_to_output(path, output_dir),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() else None,
                        "native_mesh": False,
                    }
                )
    for key, path in {
        "crosstalk_kernel_manifest": crosstalk_manifest_path,
        "crosstalk_output_kernel_csv": crosstalk_output_kernel_csv_path,
        "crosstalk_raw_pd_kernel_csv": crosstalk_raw_pd_kernel_csv_path,
        "crosstalk_kernel_summary_csv": crosstalk_summary_csv_path,
        "crosstalk_kernel_heatmap": crosstalk_heatmap_path,
        "crosstalk_convergence_report": crosstalk_convergence_path,
        "crosstalk_xsection_manifest": crosstalk_xsection_manifest_path,
        "crosstalk_xsection_output_csv": crosstalk_xsection_output_csv_path,
        "crosstalk_xsection_raw_pd_csv": crosstalk_xsection_raw_pd_csv_path,
        "crosstalk_xsection_summary_csv": crosstalk_xsection_summary_csv_path,
        "crosstalk_xsection_plot": crosstalk_xsection_plot_path,
        "crosstalk_xsection_convergence": crosstalk_xsection_convergence_path,
    }.items():
        if path:
            results.append(
                {
                    "id": key,
                    "kind": path.suffix.lstrip(".").upper() or "FILE",
                    "label": key.replace("_", " "),
                    "path": str(path),
                    "relative_path": rel_to_output(path, output_dir),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "native_mesh": False,
                }
            )
    for key, path in {
        "design_space_schema": design_space_path,
        "lumerical_ux_goal": ROOT / "LUMERICAL_UX_GOAL.md",
        "variant_run_manifest": variant_manifest_path,
        "variant_comparison_report": variant_comparison_report_path,
        "variant_comparison_csv": variant_comparison_csv_path,
        "variant_comparison_json": variant_comparison_json_path,
        "run_manager_report": run_manager_report_path,
        "run_manager_csv": run_manager_csv_path,
        "run_manager_json": run_manager_json_path,
        "dataset_catalog_report": dataset_catalog_report_path,
        "dataset_catalog_csv": dataset_catalog_csv_path,
        "dataset_catalog_json": dataset_catalog_json_path,
        "orchestrator_last_run": orchestrator_last_run_path,
        "orchestrator_history": orchestrator_history_path,
    }.items():
        if path:
            results.append(
                {
                    "id": key,
                    "kind": path.suffix.lstrip(".").upper() or "FILE",
                    "label": key.replace("_", " "),
                    "path": str(path),
                    "relative_path": rel_to_output(path, output_dir),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "native_mesh": False,
                }
            )
    for key, value in views.items():
        if not isinstance(value, str) or not value:
            continue
        path = abs_path(config_dir, value)
        results.append(
            {
                "id": f"view_{key}",
                "kind": path.suffix.lstrip(".").upper() or "FILE",
                "label": key.replace("_", " "),
                "path": str(path),
                "relative_path": rel_to_output(path, output_dir),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "native_mesh": path.suffix.lower() in {".msh", ".vtk", ".vtu", ".dat"},
            }
        )
    for key, value in gw_coupling.get("outputs", {}).items():
        if isinstance(value, str):
            path = Path(value)
            results.append(
                {
                    "id": f"gw_{key}",
                    "kind": path.suffix.lstrip(".").upper() or "FILE",
                    "label": f"G*W coupling {key}",
                    "path": str(path),
                    "relative_path": rel_to_output(path, output_dir),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "native_mesh": False,
                }
            )
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                path = Path(child_value)
                results.append(
                    {
                        "id": f"gw_{key}_{child_key}",
                        "kind": path.suffix.lstrip(".").upper() or "FILE",
                        "label": f"G*W coupling {key} {child_key}",
                        "path": str(path),
                        "relative_path": rel_to_output(path, output_dir),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() else None,
                        "native_mesh": False,
                    }
                )

    unique_results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for result in results:
        key = str(Path(result["path"]).resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        unique_results.append(result)
    results = unique_results

    tree = object_tree(project, stack, profile, design_space)
    if run_manager:
        tree.append(
            {
                "id": "run_manager",
                "label": "Run Manager",
                "kind": "Run Status",
                "data": run_manager.get("summary", {}),
                "children": [
                    {
                        "id": f"run_variant_{item.get('id', index)}",
                        "label": item.get("label", item.get("id", f"variant_{index}")),
                        "kind": "Variant Run State",
                        "data": item,
                    }
                    for index, item in enumerate(run_manager.get("variant_summaries", []))
                ],
            }
        )
    if dataset_catalog:
        tree.append(
            {
                "id": "dataset_catalog",
                "label": "Dataset Catalog",
                "kind": "Results Dataset",
                "data": dataset_catalog.get("summary", {}),
                "children": [
                    {
                        "id": f"result_group_{item['id']}",
                        "label": f"{item['object_label']} / {item['result_label']}",
                        "kind": "Result Group",
                        "data": {
                            "object_label": item["object_label"],
                            "result_role": item["result_role"],
                            "dataset_count": item["dataset_count"],
                            "existing_count": item["existing_count"],
                            "native_state": item["native_state"],
                            "viewer_count": item["viewer_count"],
                            "primary_viewer": item["primary_viewer"],
                            "readiness": item["readiness"],
                            "readiness_reason": item["readiness_reason"],
                        },
                    }
                    for item in result_groups
                ],
            }
        )
    if orchestrator_last_run:
        tree.append(
            {
                "id": "variant_orchestrator",
                "label": "Variant Orchestrator",
                "kind": "Run Control",
                "data": orchestrator_last_run.get("summary", {}),
                "children": [
                    {
                        "id": f"orchestrator_plan_{index}",
                        "label": f"{item.get('variant_id', 'variant')} / {item.get('stage', 'stage')}",
                        "kind": "Orchestrator Plan Row",
                        "data": item,
                    }
                    for index, item in enumerate(orchestrator_last_run.get("plan", []))
                ],
            }
        )
    return {
        "schema": "image_sensor_pixel_studio_payload_v1",
        "project_config": str(config_path),
        "project": project,
        "stack": stack,
        "profile": profile,
        "design_space": design_space,
        "design_manifest": design_manifest,
        "accuracy": accuracy,
        "optical_stack_evidence": optical_stack_evidence,
        "gw_coupling": gw_coupling,
        "gw_summary_rows": gw_summary_rows,
        "crosstalk_kernel": crosstalk_kernel,
        "crosstalk_output_rows": crosstalk_output_rows,
        "crosstalk_raw_pd_rows": crosstalk_raw_pd_rows,
        "crosstalk_summary_rows": crosstalk_summary_rows,
        "crosstalk_convergence": crosstalk_convergence,
        "crosstalk_xsection": crosstalk_xsection,
        "crosstalk_xsection_output_rows": crosstalk_xsection_output_rows,
        "crosstalk_xsection_raw_pd_rows": crosstalk_xsection_raw_pd_rows,
        "crosstalk_xsection_summary_rows": crosstalk_xsection_summary_rows,
        "crosstalk_xsection_convergence": crosstalk_xsection_convergence,
        "variant_manifest": variant_manifest,
        "variant_comparison": variant_comparison,
        "variant_comparison_rows": variant_comparison_rows,
        "run_manager": run_manager,
        "run_stage_rows": run_stage_rows,
        "dataset_catalog": dataset_catalog,
        "dataset_catalog_rows": dataset_catalog_rows,
        "result_groups": result_groups,
        "orchestrator_last_run": orchestrator_last_run,
        "orchestrator_history": orchestrator_history,
        "object_tree": tree,
        "flat_objects": flatten_tree(tree),
        "native_runs": native_runs,
        "report_rows": report_rows,
        "results": results,
        "paths": {
            "cross_section_2d": rel_to_output(abs_path(config_dir, views["cross_section_2d"]), output_dir),
            "geometry_3d": rel_to_output(abs_path(config_dir, views["geometry_3d"]), output_dir),
            "parameter_report": rel_to_output(abs_path(config_dir, views["parameter_report"]), output_dir),
            "parameter_report_csv": rel_to_output(report_csv_path, output_dir),
            "gw_coupling_report": rel_to_output(gw_report_path, output_dir) if gw_report_path else "",
            "camera_system_diagnostic_report": rel_to_output(camera_diagnostic_report_path, output_dir)
            if camera_diagnostic_report_path
            else "",
            "camera_system_lut_report": rel_to_output(camera_lut_report_path, output_dir) if camera_lut_report_path else "",
            "camera_system_research_lut_report": rel_to_output(camera_research_lut_report_path, output_dir)
            if camera_research_lut_report_path
            else "",
            "crosstalk_kernel_manifest": rel_from(crosstalk_manifest_path, ROOT)
            if crosstalk_manifest_path
            else "",
            "crosstalk_output_kernel_csv": rel_from(crosstalk_output_kernel_csv_path, ROOT)
            if crosstalk_output_kernel_csv_path
            else "",
            "crosstalk_raw_pd_kernel_csv": rel_from(crosstalk_raw_pd_kernel_csv_path, ROOT)
            if crosstalk_raw_pd_kernel_csv_path
            else "",
            "crosstalk_kernel_summary_csv": rel_from(crosstalk_summary_csv_path, ROOT)
            if crosstalk_summary_csv_path
            else "",
            "crosstalk_kernel_heatmap": rel_to_output(crosstalk_heatmap_path, output_dir)
            if crosstalk_heatmap_path
            else "",
            "crosstalk_convergence_report": rel_from(crosstalk_convergence_path, ROOT)
            if crosstalk_convergence_path
            else "",
            "crosstalk_xsection_manifest": rel_from(crosstalk_xsection_manifest_path, ROOT)
            if crosstalk_xsection_manifest_path
            else "",
            "crosstalk_xsection_output_csv": rel_from(crosstalk_xsection_output_csv_path, ROOT)
            if crosstalk_xsection_output_csv_path
            else "",
            "crosstalk_xsection_raw_pd_csv": rel_from(crosstalk_xsection_raw_pd_csv_path, ROOT)
            if crosstalk_xsection_raw_pd_csv_path
            else "",
            "crosstalk_xsection_summary_csv": rel_from(crosstalk_xsection_summary_csv_path, ROOT)
            if crosstalk_xsection_summary_csv_path
            else "",
            "crosstalk_xsection_plot": rel_to_output(crosstalk_xsection_plot_path, output_dir)
            if crosstalk_xsection_plot_path
            else "",
            "crosstalk_xsection_convergence": rel_from(crosstalk_xsection_convergence_path, ROOT)
            if crosstalk_xsection_convergence_path
            else "",
            "stack_config": rel_from(stack_path, ROOT),
            "tcad_profile": rel_from(profile_path, ROOT),
            "design_space": rel_from(design_space_path, ROOT) if design_space_path else "",
            "design_viewer_manifest": rel_from(manifest_path, ROOT),
            "accuracy_gate": rel_from(accuracy_path, ROOT) if accuracy_path else "",
            "optical_stack_summary": rel_from(optical_stack_summary_path, ROOT)
            if optical_stack_summary_path
            else "",
            "gw_coupling_manifest": rel_from(gw_manifest_path, ROOT) if gw_manifest_path else "",
            "variant_run_manifest": rel_from(variant_manifest_path, ROOT) if variant_manifest_path else "",
            "variant_comparison_report": rel_to_output(variant_comparison_report_path, output_dir)
            if variant_comparison_report_path
            else "",
            "variant_comparison_csv": rel_from(variant_comparison_csv_path, ROOT) if variant_comparison_csv_path else "",
            "variant_comparison_json": rel_from(variant_comparison_json_path, ROOT) if variant_comparison_json_path else "",
            "run_manager_report": rel_to_output(run_manager_report_path, output_dir) if run_manager_report_path else "",
            "run_manager_csv": rel_from(run_manager_csv_path, ROOT) if run_manager_csv_path else "",
            "run_manager_json": rel_from(run_manager_json_path, ROOT) if run_manager_json_path else "",
            "dataset_catalog_report": rel_to_output(dataset_catalog_report_path, output_dir)
            if dataset_catalog_report_path
            else "",
            "dataset_catalog_csv": rel_from(dataset_catalog_csv_path, ROOT) if dataset_catalog_csv_path else "",
            "dataset_catalog_json": rel_from(dataset_catalog_json_path, ROOT) if dataset_catalog_json_path else "",
            "orchestrator_last_run": rel_from(orchestrator_last_run_path, ROOT) if orchestrator_last_run_path else "",
            "orchestrator_history": rel_from(orchestrator_history_path, ROOT) if orchestrator_history_path else "",
        },
        "file_status": {
            "stack_config": file_status(stack_path),
            "tcad_profile": file_status(profile_path),
            "design_space": file_status(design_space_path) if design_space_path else {"exists": False},
            "design_viewer_manifest": file_status(manifest_path),
            "accuracy_gate": file_status(accuracy_path) if accuracy_path else {"exists": False},
            "optical_stack_summary": file_status(optical_stack_summary_path)
            if optical_stack_summary_path
            else {"exists": False},
            "gw_coupling_manifest": file_status(gw_manifest_path) if gw_manifest_path else {"exists": False},
            "camera_system_diagnostic_report": file_status(camera_diagnostic_report_path)
            if camera_diagnostic_report_path
            else {"exists": False},
            "camera_system_lut_report": file_status(camera_lut_report_path)
            if camera_lut_report_path
            else {"exists": False},
            "camera_system_research_lut_report": file_status(camera_research_lut_report_path)
            if camera_research_lut_report_path
            else {"exists": False},
            "crosstalk_kernel_manifest": file_status(crosstalk_manifest_path)
            if crosstalk_manifest_path
            else {"exists": False},
            "crosstalk_output_kernel_csv": file_status(crosstalk_output_kernel_csv_path)
            if crosstalk_output_kernel_csv_path
            else {"exists": False},
            "crosstalk_raw_pd_kernel_csv": file_status(crosstalk_raw_pd_kernel_csv_path)
            if crosstalk_raw_pd_kernel_csv_path
            else {"exists": False},
            "crosstalk_kernel_summary_csv": file_status(crosstalk_summary_csv_path)
            if crosstalk_summary_csv_path
            else {"exists": False},
            "crosstalk_kernel_heatmap": file_status(crosstalk_heatmap_path)
            if crosstalk_heatmap_path
            else {"exists": False},
            "crosstalk_convergence_report": file_status(crosstalk_convergence_path)
            if crosstalk_convergence_path
            else {"exists": False},
            "crosstalk_xsection_manifest": file_status(crosstalk_xsection_manifest_path)
            if crosstalk_xsection_manifest_path
            else {"exists": False},
            "crosstalk_xsection_output_csv": file_status(crosstalk_xsection_output_csv_path)
            if crosstalk_xsection_output_csv_path
            else {"exists": False},
            "crosstalk_xsection_raw_pd_csv": file_status(crosstalk_xsection_raw_pd_csv_path)
            if crosstalk_xsection_raw_pd_csv_path
            else {"exists": False},
            "crosstalk_xsection_summary_csv": file_status(crosstalk_xsection_summary_csv_path)
            if crosstalk_xsection_summary_csv_path
            else {"exists": False},
            "crosstalk_xsection_plot": file_status(crosstalk_xsection_plot_path)
            if crosstalk_xsection_plot_path
            else {"exists": False},
            "crosstalk_xsection_convergence": file_status(crosstalk_xsection_convergence_path)
            if crosstalk_xsection_convergence_path
            else {"exists": False},
            "variant_run_manifest": file_status(variant_manifest_path) if variant_manifest_path else {"exists": False},
            "variant_comparison_report": file_status(variant_comparison_report_path)
            if variant_comparison_report_path
            else {"exists": False},
            "variant_comparison_csv": file_status(variant_comparison_csv_path)
            if variant_comparison_csv_path
            else {"exists": False},
            "variant_comparison_json": file_status(variant_comparison_json_path)
            if variant_comparison_json_path
            else {"exists": False},
            "run_manager_report": file_status(run_manager_report_path)
            if run_manager_report_path
            else {"exists": False},
            "run_manager_csv": file_status(run_manager_csv_path) if run_manager_csv_path else {"exists": False},
            "run_manager_json": file_status(run_manager_json_path) if run_manager_json_path else {"exists": False},
            "dataset_catalog_report": file_status(dataset_catalog_report_path)
            if dataset_catalog_report_path
            else {"exists": False},
            "dataset_catalog_csv": file_status(dataset_catalog_csv_path)
            if dataset_catalog_csv_path
            else {"exists": False},
            "dataset_catalog_json": file_status(dataset_catalog_json_path)
            if dataset_catalog_json_path
            else {"exists": False},
            "orchestrator_last_run": file_status(orchestrator_last_run_path)
            if orchestrator_last_run_path
            else {"exists": False},
            "orchestrator_history": file_status(orchestrator_history_path)
            if orchestrator_history_path
            else {"exists": False},
        },
    }


def write_html_legacy(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(payload, ensure_ascii=False)
    app_title = html.escape(payload["project"]["project"]["name"])
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{app_title} - Pixel Studio</title>
<style>
:root{{
  --bg:#f4f6f8;--panel:#ffffff;--line:#cfd7df;--text:#1f2933;--muted:#61707f;
  --accent:#0f766e;--warn:#b45309;--bad:#b91c1c;--ok:#177245;--blue:#2563eb;
}}
*{{box-sizing:border-box}}
html,body{{width:100%;height:100%;overflow:hidden}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--bg);font-size:13px}}
button,select,input{{font:inherit}}
.app{{display:grid;grid-template-rows:48px minmax(0,1fr) 172px;width:100vw;height:100vh;min-height:680px;overflow:hidden}}
.topbar{{display:flex;align-items:center;gap:14px;min-width:0;padding:0 14px;background:#111827;color:white;border-bottom:1px solid #000}}
.brand{{font-weight:700;white-space:nowrap}}
.topbar .meta{{flex:1;min-width:0;color:#c9d3df;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.status{{display:inline-flex;align-items:center;height:22px;padding:0 7px;border:1px solid #334155;border-radius:6px;background:#1f2937;color:#d7eef0;font-size:12px}}
.workspace{{display:grid;grid-template-columns:270px minmax(0,1fr) 340px;min-width:0;min-height:0}}
.left,.right,.bottom{{background:var(--panel)}}
.left{{border-right:1px solid var(--line);display:grid;grid-template-rows:38px minmax(0,1fr)}}
.right{{border-left:1px solid var(--line);display:grid;grid-template-rows:38px minmax(0,1fr)}}
.bottom{{border-top:1px solid var(--line);display:grid;grid-template-rows:38px minmax(0,1fr);min-width:0;overflow:hidden}}
.paneHeader{{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:0 12px;border-bottom:1px solid var(--line);font-weight:650;background:#f9fafb}}
.tree{{overflow:auto;padding:8px}}
.tree button{{width:100%;display:grid;grid-template-columns:20px minmax(0,1fr) auto;align-items:center;text-align:left;border:0;background:transparent;padding:6px 7px;border-radius:6px;color:var(--text);cursor:pointer}}
.tree button:hover,.tree button.active{{background:#e8f1ef}}
.indent0{{margin-left:0}}.indent1{{margin-left:14px}}.indent2{{margin-left:28px}}
.nodeKind{{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.center{{min-width:0;display:grid;grid-template-rows:42px minmax(0,1fr)}}
.tabs{{display:flex;align-items:center;gap:6px;padding:6px 10px;background:#ffffff;border-bottom:1px solid var(--line);overflow:auto}}
.tabs button,.toolButton{{height:28px;border:1px solid var(--line);background:white;border-radius:6px;padding:0 10px;cursor:pointer;color:var(--text)}}
.tabs button.active{{background:#0f766e;color:white;border-color:#0f766e}}
.view{{min-width:0;min-height:0;position:relative;background:white}}
.viewPane{{display:none;width:100%;height:100%;min-height:0}}
.viewPane.active{{display:block}}
iframe{{width:100%;height:100%;border:0;background:white}}
.summaryGrid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;padding:14px}}
.metric{{border:1px solid var(--line);border-radius:8px;background:white;padding:10px;min-width:0}}
.metric .label{{color:var(--muted);font-size:12px;margin-bottom:6px}}
.metric .value{{font-size:18px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.actionPanel{{margin:0 14px 14px 14px;border:1px solid var(--line);border-radius:8px;background:#fbfcfd;padding:12px}}
.actionPanel h3{{margin:0 0 6px 0}}
.actionPanel p{{margin:0 0 10px 0;color:var(--muted);line-height:1.45}}
.actionGrid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 10px 0}}
.actionFact{{border:1px solid #e5e9ee;border-radius:6px;background:white;padding:8px;min-width:0}}
.actionFact .k{{font-size:11px;color:var(--muted);margin-bottom:4px}}
.actionFact .v{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.preflightText{{margin-top:8px;color:var(--muted);line-height:1.45}}
.details{{overflow:auto;padding:12px}}
pre{{margin:0;white-space:pre-wrap;word-break:break-word;background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:10px;font-size:12px;line-height:1.42}}
.kv{{display:grid;grid-template-columns:120px minmax(0,1fr);gap:8px;margin-bottom:8px}}
.kv .k{{color:var(--muted)}}
.pill{{display:inline-flex;align-items:center;height:20px;padding:0 6px;border-radius:5px;border:1px solid var(--line);font-size:11px;background:#fff;margin:2px 4px 2px 0}}
.pill.ok{{border-color:#b7dec7;color:var(--ok);background:#f0fbf4}}
.pill.warn{{border-color:#f5c27a;color:var(--warn);background:#fff8eb}}
.pill.bad{{border-color:#f0a4a4;color:var(--bad);background:#fff5f5}}
.resultTools{{display:flex;align-items:center;gap:8px}}
.resultTools input{{height:26px;width:260px;border:1px solid var(--line);border-radius:6px;padding:0 8px}}
.resultGroups{{max-height:210px;overflow:auto;border-bottom:1px solid var(--line);background:#fbfcfd}}
.resultGroups h3{{margin:8px 10px 4px 10px;font-size:13px}}
.resultGroups table{{background:white}}
.resultTableWrap{{min-width:0;overflow:auto}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}}
th,td{{border-bottom:1px solid #e5e9ee;padding:6px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{position:sticky;top:0;background:#f9fafb;z-index:1;color:#4b5563}}
td.path{{max-width:520px;overflow:hidden;text-overflow:ellipsis}}
a{{color:#1155cc;text-decoration:none}}a:hover{{text-decoration:underline}}
.runbook{{padding:14px;overflow:auto;height:100%}}
.command{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:start;border:1px solid var(--line);border-radius:8px;background:white;padding:10px;margin-bottom:10px}}
.command code{{display:block;white-space:pre-wrap;word-break:break-word;color:#111827;font-size:12px;line-height:1.4}}
.accuracyList{{padding:14px;overflow:auto;height:100%}}
.issue{{border-left:3px solid var(--bad);padding:8px 10px;margin-bottom:8px;background:#fffafa}}
.issue.warn{{border-left-color:var(--warn);background:#fff9f0}}
.splitTable{{padding:14px;overflow:auto;height:100%}}
.designSpace{{padding:14px;overflow:auto;height:100%}}
.designGrid{{display:grid;grid-template-columns:1fr;gap:14px}}
.designEditBuilder{{border:1px solid var(--line);border-radius:8px;background:#fbfcfd;padding:12px}}
.editHeader{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}}
.editHeader h3{{margin:0 0 4px 0}}
.editHeader p{{margin:0;color:var(--muted)}}
.editGrid{{display:grid;grid-template-columns:minmax(220px,1.4fr) minmax(180px,.8fr) minmax(220px,1fr);gap:10px;margin:10px 0}}
.editGrid label{{display:grid;gap:5px;color:var(--muted);font-size:12px;min-width:0}}
.editGrid select,.editGrid input{{height:32px;border:1px solid var(--line);border-radius:6px;padding:0 8px;background:white;color:var(--text);min-width:0}}
.editFacts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 10px 0;border-top:1px solid #e5e9ee;border-bottom:1px solid #e5e9ee;padding:8px 0}}
.editFact{{min-width:0}}
.editFact .k{{color:var(--muted);font-size:11px;margin-bottom:4px}}
.editFact .v{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.editWarning{{display:none;margin:8px 0;color:var(--warn)}}
.editWarning.active{{display:block}}
.command.compact{{margin-bottom:8px}}
.stageTags{{display:flex;gap:4px;flex-wrap:wrap;min-width:0}}
.override{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;white-space:pre-wrap;word-break:break-word}}
@media (max-width: 980px){{
  .workspace{{grid-template-columns:220px minmax(0,1fr)}}
  .right{{display:none}}
  .summaryGrid{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .actionGrid{{grid-template-columns:1fr}}
  .editGrid,.editFacts{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">Image Sensor Pixel Studio</div>
    <span class="status" id="statusBadge">loading</span>
    <div class="meta" id="projectMeta"></div>
  </header>
  <main class="workspace">
    <aside class="left">
      <div class="paneHeader"><span>Object Tree</span><span id="objectCount"></span></div>
      <div class="tree" id="tree"></div>
    </aside>
    <section class="center">
      <nav class="tabs" id="tabs">
        <button data-tab="overview" class="active">Overview</button>
        <button data-tab="design-space">Design Space</button>
        <button data-tab="cross">2D Cross Section</button>
        <button data-tab="geometry">3D Geometry</button>
        <button data-tab="report">Sweep Report</button>
        <button data-tab="gw">G*W Coupling</button>
	        <button data-tab="camera-lut">Research LUT</button>
	        <button data-tab="run-manager">Run Manager</button>
	        <button data-tab="orchestrator">Orchestrator</button>
	        <button data-tab="datasets">Datasets</button>
	        <button data-tab="split">Native Split Runs</button>
        <button data-tab="accuracy">Accuracy Gate</button>
        <button data-tab="runbook">Runbook</button>
      </nav>
      <div class="view">
        <section id="tab-overview" class="viewPane active"></section>
        <section id="tab-design-space" class="viewPane designSpace"></section>
        <section id="tab-cross" class="viewPane"><iframe id="crossFrame"></iframe></section>
        <section id="tab-geometry" class="viewPane"><iframe id="geometryFrame"></iframe></section>
        <section id="tab-report" class="viewPane"><iframe id="reportFrame"></iframe></section>
        <section id="tab-gw" class="viewPane"><iframe id="gwFrame"></iframe></section>
	        <section id="tab-camera-lut" class="viewPane"><iframe id="cameraLutFrame"></iframe></section>
	        <section id="tab-run-manager" class="viewPane splitTable"></section>
	        <section id="tab-orchestrator" class="viewPane splitTable"></section>
	        <section id="tab-datasets" class="viewPane splitTable"></section>
        <section id="tab-split" class="viewPane splitTable"></section>
        <section id="tab-accuracy" class="viewPane accuracyList"></section>
        <section id="tab-runbook" class="viewPane runbook"></section>
      </div>
    </section>
    <aside class="right">
      <div class="paneHeader"><span>Properties</span><span id="selectedKind"></span></div>
      <div class="details" id="properties"></div>
    </aside>
  </main>
  <section class="bottom">
    <div class="paneHeader">
      <span>Results Manager</span>
      <div class="resultTools"><input id="resultFilter" placeholder="filter results, fields, paths"><span id="resultCount"></span></div>
    </div>
    <div class="resultGroups" id="resultGroups"></div>
    <div class="resultTableWrap"><table id="results"></table></div>
  </section>
</div>
<script>
const payload = {json_payload};
const flat = payload.flat_objects;
const byId = Object.fromEntries(flat.map(o => [o.id, o]));
let selectedId = flat[0]?.id || null;

function fmtBytes(v) {{
  if (v === null || v === undefined) return '';
  if (v < 1024) return `${{v}} B`;
  if (v < 1024*1024) return `${{(v/1024).toFixed(1)}} KB`;
  return `${{(v/1024/1024).toFixed(1)}} MB`;
}}
function exp(v, n=4) {{
  if (v === null || v === undefined || v === '') return '';
  const x = Number(v);
  return Number.isFinite(x) ? x.toExponential(n) : String(v ?? '');
}}
function sig(v, n=5) {{
  if (v === null || v === undefined || v === '') return '';
  const x = Number(v);
  return Number.isFinite(x) ? x.toPrecision(n) : String(v ?? '');
}}
function pct(v, n=2) {{
  if (v === null || v === undefined || v === '') return '';
  const x = Number(v);
  return Number.isFinite(x) ? `${{(x * 100).toFixed(n)}}%` : String(v ?? '');
}}
function stateClass(state) {{
  if (state === 'complete' || state === 'executed_reference') return 'ok';
  if (state === 'partial') return 'bad';
  return 'warn';
}}
	function statusClass(status) {{
	  if (status === 'complete' || status === 'true' || status === 'fresh' || status === 'PASS') return 'ok';
	  if (status === 'partial' || status === 'stale' || status === 'unknown' || status === 'WARN') return 'warn';
	  if (status === 'missing' || status === 'false' || status === 'FAIL') return 'bad';
	  return 'warn';
	}}
	function escapeText(s) {{
	  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
	}}
	function bindCopyButtons(scope) {{
	  scope.querySelectorAll('[data-copy]').forEach(btn => btn.onclick = async () => {{
	    try {{ await navigator.clipboard.writeText(btn.dataset.copy); btn.textContent = 'Copied'; }}
	    catch {{ btn.textContent = 'Select'; }}
	    setTimeout(() => btn.textContent = 'Copy', 1200);
	  }});
	}}

	function commandQuote(value) {{
	  const raw = String(value ?? '');
	  if (/^[A-Za-z0-9_./:=+-]+$/.test(raw)) return raw;
	  return '"' + raw.replace(/(["\\\\$`])/g, '\\\\$1') + '"';
	}}

	function safeVariantId(value) {{
	  const clean = String(value ?? 'custom_variant')
	    .toLowerCase()
	    .replace(/[^a-z0-9_-]/g, '_')
	    .replace(/^_+|_+$/g, '');
	  return clean || 'custom_variant';
	}}

	function defaultParamValue(parameter) {{
	  const value = parameter?.current_value;
	  if (value === undefined || value === null) return '';
	  if (typeof value === 'object') return JSON.stringify(value);
	  return String(value);
	}}

	function formatRange(parameter) {{
	  const range = parameter?.range;
	  if (Array.isArray(range) && range.length === 2) return `${{sig(range[0])}} to ${{sig(range[1])}}`;
	  return '-';
	}}

	function valueOutOfRange(parameter, value) {{
	  const range = parameter?.range;
	  const numericValue = Number(value);
	  if (!Number.isFinite(numericValue) || !Array.isArray(range) || range.length !== 2) return false;
	  const lower = Number(range[0]);
	  const upper = Number(range[1]);
	  return Number.isFinite(lower) && Number.isFinite(upper) && (numericValue < lower || numericValue > upper);
	}}

	function setNodeText(id, value) {{
	  const node = document.getElementById(id);
	  if (node) node.textContent = value;
	}}

	function initDesignEditBuilder(params) {{
	  const select = document.getElementById('editParam');
	  if (!select || !params.length) return;
	  const paramsById = Object.fromEntries(params.map(parameter => [String(parameter.id || parameter.path), parameter]));
	  const valueInput = document.getElementById('editValue');
	  const variantInput = document.getElementById('editVariantId');
	  const previewCode = document.getElementById('editPreviewCommand');
	  const materializeCode = document.getElementById('editMaterializeCommand');
	  const previewCopy = document.getElementById('copyPreviewCommand');
	  const materializeCopy = document.getElementById('copyMaterializeCommand');
	  const root = document.getElementById('tab-design-space');
	  let variantIdTouched = false;

	  function currentParam() {{
	    return paramsById[select.value] || params[0] || {{}};
	  }}

	  function refresh(resetValue) {{
	    const parameter = currentParam();
	    if (resetValue) valueInput.value = defaultParamValue(parameter);
	    const newValue = valueInput.value.trim();
	    if (!variantIdTouched || resetValue) {{
	      variantInput.value = safeVariantId(`custom_${{parameter.id || 'param'}}_${{newValue || 'value'}}`).slice(0, 72);
	    }} else {{
	      variantInput.value = safeVariantId(variantInput.value).slice(0, 72);
	    }}
	    const variantId = variantInput.value || 'custom_variant';
	    const assignment = `${{parameter.id || parameter.path}}=${{newValue}}`;
	    const previewCommand = `.tcad-env/bin/python image_sensor_design_variant_create.py --id ${{commandQuote(variantId)}} --param ${{commandQuote(assignment)}}`;
	    const materializeCommand = `${{previewCommand}} --materialize --overwrite --refresh`;
	    previewCode.textContent = previewCommand;
	    materializeCode.textContent = materializeCommand;
	    previewCopy.dataset.copy = previewCommand;
	    materializeCopy.dataset.copy = materializeCommand;

	    const outOfRange = valueOutOfRange(parameter, newValue);
	    document.getElementById('editParamMeta').innerHTML = `
	      <span class="pill ${{parameter.wired_to_solver ? 'ok' : 'warn'}}">${{parameter.wired_to_solver ? 'solver-wired' : 'metadata-only'}}</span>
	      <span class="pill warn">${{escapeText(parameter.group_label || '')}}</span>
	      <span class="pill ${{outOfRange ? 'bad' : 'ok'}}">range ${{escapeText(formatRange(parameter))}}</span>`;
	    setNodeText('editPath', parameter.path || '');
	    setNodeText('editCurrent', defaultParamValue(parameter));
	    document.getElementById('editStages').innerHTML = (parameter.requires_rerun || [])
	      .map(stage => `<span class="pill warn">${{escapeText(stage)}}</span>`)
	      .join('') || '<span class="pill ok">none</span>';
	    const warnings = [];
	    if (parameter.wired_to_solver === false) warnings.push('metadata/proxy only: current solver equations do not directly consume this parameter.');
	    if (outOfRange) warnings.push('outside recommended range: the CLI rejects this unless --allow-out-of-range is added manually.');
	    const warningNode = document.getElementById('editWarning');
	    warningNode.textContent = warnings.join(' ');
	    warningNode.className = warnings.length ? 'editWarning active' : 'editWarning';
	    bindCopyButtons(root);
	  }}

	  select.onchange = () => {{
	    variantIdTouched = false;
	    refresh(true);
	  }};
	  valueInput.oninput = () => refresh(false);
	  variantInput.oninput = () => {{
	    variantIdTouched = true;
	    refresh(false);
	  }};
	  refresh(true);
	}}

function initHeader() {{
  const p = payload.project.project;
  document.getElementById('projectMeta').textContent = `${{p.name}} - ${{p.description}}`;
  const status = payload.accuracy.accuracy_ready ? 'accuracy ready' : (payload.accuracy.framework_ready ? 'framework ready / accuracy blocked' : 'framework incomplete');
  const badge = document.getElementById('statusBadge');
  badge.textContent = status;
  badge.style.color = payload.accuracy.accuracy_ready ? '#d1fae5' : '#fed7aa';
}}

function buildOrchestratorCommand(item, execute=false, includeHeavy=false) {{
  const parts = ['.tcad-env/bin/python', 'image_sensor_variant_orchestrator.py'];
  if (item?.variant_id) parts.push('--variant', commandQuote(item.variant_id));
  if (item?.stage) parts.push('--stage', commandQuote(item.stage));
  parts.push('--next-needed');
  if (includeHeavy) parts.push('--include-heavy');
  if (execute) parts.push('--execute');
  return parts.join(' ');
}}

function stageIsHeavy(stage) {{
  return ['meep_fdtd', 'convergence_gate'].includes(String(stage || ''));
}}

function preflightIssueText(item) {{
  const checks = (item?.preflight || []).flatMap(preflight => preflight.checks || []);
  const issues = checks.filter(check => check.status !== 'PASS');
  if (!checks.length) return 'No static preflight was recorded for this plan row.';
  if (!issues.length) return 'Static preflight passed: command parse, executable/env/script, input paths, and output parent checks.';
  return issues.slice(0, 3).map(check => `${{check.status}} ${{check.name}}: ${{check.details}}`).join(' ');
}}

function runtimeHintText(item) {{
  const hints = item?.preflight?.[0]?.runtime_hints || {{}};
  const parts = [];
  if (hints.heavy_stage) parts.push('heavy stage');
  if (hints.mode) parts.push(`mode ${{hints.mode}}`);
  if (hints.resolution) parts.push(`resolution ${{hints.resolution}}`);
  if (hints.wavelength_count) parts.push(`${{hints.wavelength_count}} wavelength`);
  if (hints.case_count) parts.push(`${{hints.case_count}} cases`);
  if (hints.after_source_time) parts.push(`after-source ${{hints.after_source_time}}`);
  return parts.join(', ') || '-';
}}

function nextActionAdvisor() {{
  const runSummary = payload.run_manager?.summary || {{}};
  const accuracy = payload.accuracy || {{}};
  const plan = payload.orchestrator_last_run?.plan || [];
  const first = plan[0] || null;
  const dryRun = (payload.project.runbook || []).find(cmd => cmd.id === 'variant_orchestrator_dry_run')?.command ||
    '.tcad-env/bin/python image_sensor_variant_orchestrator.py --all --next-needed';
  const blockers = accuracy.accuracy_blocking_failure_count || (accuracy.blocking || []).length || 0;

  if (first?.action === 'skip' && String(first.reason || '').includes('heavy')) {{
    return {{
      title: 'Next Action: Heavy FDTD Decision',
      body: `${{first.variant_label || first.variant_id}} is waiting at ${{first.stage}}. The orchestrator skipped it because heavy stages require explicit approval.`,
      stateClass: 'warn',
      stateText: 'needs decision',
      variant: first.variant_id || '-',
      stage: first.stage || '-',
      reason: first.reason || '',
      preflightStatus: first.preflight_summary?.status || 'not recorded',
      preflightText: preflightIssueText(first),
      runtimeHints: runtimeHintText(first),
      previewCommand: buildOrchestratorCommand(first, false, true),
      executeCommand: buildOrchestratorCommand(first, true, true),
      caution: 'Run preview first, then execute only if the estimated FDTD cost is acceptable.'
    }};
  }}
  if (first?.action === 'run') {{
    const includeHeavy = stageIsHeavy(first.stage);
    return {{
      title: includeHeavy ? 'Next Action: Execute Planned Heavy Stage' : 'Next Action: Execute Planned Stage',
      body: `${{first.variant_label || first.variant_id}} has a runnable ${{first.stage}} stage according to the latest orchestrator dry-run.`,
      stateClass: includeHeavy ? 'warn' : 'ok',
      stateText: includeHeavy ? 'heavy ready' : 'ready',
      variant: first.variant_id || '-',
      stage: first.stage || '-',
      reason: first.reason || '',
      preflightStatus: first.preflight_summary?.status || 'not recorded',
      preflightText: preflightIssueText(first),
      runtimeHints: runtimeHintText(first),
      previewCommand: buildOrchestratorCommand(first, false, includeHeavy),
      executeCommand: buildOrchestratorCommand(first, true, includeHeavy),
      caution: includeHeavy ? 'This explicitly allows a heavy FDTD/convergence stage. Confirm runtime cost before executing.' : 'This executes local solver/run-plan commands and then refreshes management views.'
    }};
  }}
  if (first?.action === 'blocked') {{
    return {{
      title: 'Next Action: Resolve Upstream Blocker',
      body: `${{first.variant_label || first.variant_id}} cannot run ${{first.stage}} until upstream outputs exist.`,
      stateClass: 'bad',
      stateText: 'blocked',
      variant: first.variant_id || '-',
      stage: first.stage || '-',
      reason: first.reason || '',
      preflightStatus: first.preflight_summary?.status || 'not recorded',
      preflightText: preflightIssueText(first),
      runtimeHints: runtimeHintText(first),
      previewCommand: dryRun,
      executeCommand: '',
      caution: 'Inspect Run Manager for missing upstream datasets before executing anything.'
    }};
  }}
  if ((runSummary.missing_stage_count || 0) > 0 || (runSummary.stale_stage_count || 0) > 0) {{
    return {{
      title: 'Next Action: Refresh Orchestrator Plan',
      body: 'Run Manager sees missing or stale stages, but the latest orchestrator result does not contain an executable next step.',
      stateClass: 'warn',
      stateText: 'plan needed',
      variant: '-',
      stage: '-',
      reason: `${{runSummary.missing_stage_count || 0}} missing, ${{runSummary.stale_stage_count || 0}} stale`,
      preflightStatus: '',
      preflightText: '',
      runtimeHints: '',
      previewCommand: dryRun,
      executeCommand: '',
      caution: 'Refresh the dry-run plan before launching solver stages.'
    }};
  }}
  if (!accuracy.accuracy_ready) {{
    return {{
      title: 'Next Action: Accuracy Evidence Required',
      body: 'All currently materialized management outputs are fresh, but product-LUT accuracy remains blocked.',
      stateClass: 'bad',
      stateText: 'accuracy blocked',
      variant: '-',
      stage: '-',
      reason: `${{blockers}} accuracy blockers`,
      preflightStatus: '',
      preflightText: '',
      runtimeHints: '',
      previewCommand: '',
      executeCommand: '',
      caution: 'Measured stack geometry, measured n,k, calibrated electrical equations, and convergence evidence are still required.'
    }};
  }}
  return {{
    title: 'Next Action: No Pending Local Run',
    body: 'Run Manager and Accuracy Gate do not report a pending local action.',
    stateClass: 'ok',
    stateText: 'clear',
    variant: '-',
    stage: '-',
    reason: '',
    preflightStatus: '',
    preflightText: '',
    runtimeHints: '',
    previewCommand: dryRun,
    executeCommand: '',
    caution: ''
  }};
}}

function renderTree() {{
  const tree = document.getElementById('tree');
  tree.innerHTML = flat.map(o => `
    <button class="indent${{Math.min(o.depth,2)}} ${{o.id===selectedId?'active':''}}" data-id="${{o.id}}">
      <span>${{o.depth ? '-' : '>'}}</span>
      <span>${{escapeText(o.label)}}</span>
      <span class="nodeKind">${{escapeText(o.kind)}}</span>
    </button>`).join('');
  document.getElementById('objectCount').textContent = `${{flat.length}} objects`;
  tree.querySelectorAll('button').forEach(btn => btn.onclick = () => {{
    selectedId = btn.dataset.id;
    renderTree();
    renderProperties();
  }});
}}

function renderProperties() {{
  const obj = byId[selectedId] || flat[0];
  if (!obj) return;
  document.getElementById('selectedKind').textContent = obj.kind || '';
  const data = obj.data || obj;
  const measured = data.measured === false || payload.profile.reference_mode ? '<span class="pill warn">proxy/reference</span>' : '';
  const wired = data.wired_to_solver === true ? '<span class="pill ok">solver-wired</span>' : (data.wired_to_solver === false ? '<span class="pill warn">metadata-only</span>' : '');
  const resultGroupId = String(obj.id || '').replace(/^result_group_/, '');
  const resultGroup = (payload.result_groups || []).find(g => g.id === resultGroupId);
  const resultBlock = resultGroup ? `
    <h3>Result Datasets</h3>
    <div class="kv"><div class="k">datasets</div><div>${{escapeText(resultGroup.existing_count)}} / ${{escapeText(resultGroup.dataset_count)}}</div></div>
    <div class="kv"><div class="k">native state</div><div><span class="pill ${{resultGroup.native_state === 'native' ? 'ok' : 'warn'}}">${{escapeText(resultGroup.native_state)}}</span></div></div>
    <div class="kv"><div class="k">viewer</div><div>${{resultGroup.primary_viewer ? `<a href="${{escapeText(resultGroup.primary_viewer)}}">open primary viewer</a>` : 'data only'}}</div></div>
    <table>
      <thead><tr><th>Dataset</th><th>Kind</th><th>Dim</th><th>Native</th><th>Viewer</th></tr></thead>
      <tbody>${{(resultGroup.datasets || []).slice(0, 8).map(d => `
        <tr>
          <td>${{escapeText(d.dataset_id)}}</td>
          <td>${{escapeText(d.kind)}}</td>
          <td>${{escapeText(d.dimensionality)}}</td>
          <td>${{d.native ? '<span class="pill ok">native</span>' : '<span class="pill warn">derived</span>'}}</td>
          <td>${{escapeText(d.viewer_mode)}}</td>
        </tr>`).join('')}}</tbody>
    </table>` : '';
  document.getElementById('properties').innerHTML = `
    <div class="kv"><div class="k">name</div><div>${{escapeText(obj.label)}}</div></div>
    <div class="kv"><div class="k">kind</div><div>${{escapeText(obj.kind)}}</div></div>
    <div>${{measured}}${{wired}}</div>
    <pre>${{escapeText(JSON.stringify(data, null, 2))}}</pre>
    ${{resultBlock}}`;
}}

function renderOverview() {{
  const runs = payload.native_runs;
  const center = runs.find(r => r.case === 'center') || runs[0] || {{}};
  const edge = runs.find(r => r.case === 'edge20x') || runs[1] || {{}};
  const gwCases = payload.gw_coupling?.cases || [];
  const gwEdge = gwCases.find(r => r.case === 'edge20x') || gwCases[1] || {{}};
  const researchLutFiles = payload.gw_coupling?.outputs?.camera_system_research_lut ? Object.keys(payload.gw_coupling.outputs.camera_system_research_lut).length : 0;
  const designParams = payload.design_space?.flat_parameters?.length || 0;
  const designVariants = payload.design_space?.variants?.length || 0;
  const variantPlans = payload.variant_manifest?.variants?.length || 0;
  const completedVariants = payload.variant_comparison?.summary?.completed_candidate_count || 0;
  const comparisonRows = payload.variant_comparison?.summary?.row_count || 0;
	  const runSummary = payload.run_manager?.summary || {{}};
	  const datasetSummary = payload.dataset_catalog?.summary || payload.run_manager?.summary || {{}};
	  const orchestratorSummary = payload.orchestrator_last_run?.summary || {{}};
	  const blockers = payload.accuracy.accuracy_blocking_failure_count || payload.accuracy.blocking.length;
  const advisor = nextActionAdvisor();
  const previewBlock = advisor.previewCommand ? `
    <div class="command compact">
      <div><b>Preview / plan command</b><code>${{escapeText(advisor.previewCommand)}}</code></div>
      <button class="toolButton" data-copy="${{escapeText(advisor.previewCommand)}}">Copy</button>
    </div>` : '';
  const executeBlock = advisor.executeCommand ? `
    <div class="command compact">
      <div><b>Explicit execute command</b><code>${{escapeText(advisor.executeCommand)}}</code></div>
      <button class="toolButton" data-copy="${{escapeText(advisor.executeCommand)}}">Copy</button>
    </div>` : '';
  const preflightBlock = advisor.preflightStatus ? `
    <div class="preflightText"><span class="pill ${{statusClass(advisor.preflightStatus)}}">preflight ${{escapeText(advisor.preflightStatus)}}</span>${{escapeText(advisor.preflightText || '')}}</div>` : '';
  const root = document.getElementById('tab-overview');
  root.innerHTML = `
    <div class="summaryGrid">
      <div class="metric"><div class="label">Framework</div><div class="value">${{payload.accuracy.framework_ready ? 'Ready' : 'Incomplete'}}</div><span class="pill ${{payload.accuracy.framework_ready?'ok':'bad'}}">pipeline</span></div>
      <div class="metric"><div class="label">Accuracy Gate</div><div class="value">${{payload.accuracy.accuracy_ready ? 'PASS' : 'BLOCKED'}}</div><span class="pill ${{payload.accuracy.accuracy_ready?'ok':'bad'}}">${{blockers}} blockers</span></div>
      <div class="metric"><div class="label">Center Total dI</div><div class="value">${{exp(center.total_photo_delta_a_per_cm)}} A/cm</div><span class="pill ok">${{center.mesh_source || 'mesh'}}</span></div>
      <div class="metric"><div class="label">Edge20x Total dI</div><div class="value">${{exp(edge.total_photo_delta_a_per_cm)}} A/cm</div><span class="pill ok">${{edge.mesh_source || 'mesh'}}</span></div>
      <div class="metric"><div class="label">W_proxy Edge Total Error</div><div class="value">${{sig(gwEdge.gw_total_reference_scaled_rel_error,4)}}</div><span class="pill warn">proxy</span></div>
      <div class="metric"><div class="label">W_mesh Edge Total Error</div><div class="value">${{sig(gwEdge.gw_mesh_total_reference_scaled_rel_error,4)}}</div><span class="pill warn">mesh</span></div>
      <div class="metric"><div class="label">W_devsim Edge Total Error</div><div class="value">${{sig(gwEdge.gw_devsim_laplace_total_reference_scaled_rel_error,4)}}</div><span class="pill warn">laplace</span></div>
      <div class="metric"><div class="label">W_proxy Edge Split Error</div><div class="value">${{sig(gwEdge.gw_split_phase_error,4)}}</div><span class="pill warn">proxy</span></div>
      <div class="metric"><div class="label">W_mesh Edge Split Error</div><div class="value">${{sig(gwEdge.gw_mesh_split_phase_error,4)}}</div><span class="pill warn">mesh</span></div>
      <div class="metric"><div class="label">W_devsim Edge Split Error</div><div class="value">${{sig(gwEdge.gw_devsim_laplace_split_phase_error,4)}}</div><span class="pill warn">laplace</span></div>
      <div class="metric"><div class="label">Research LUT Files</div><div class="value">${{researchLutFiles}}</div><span class="pill ok">native DEVSIM</span></div>
      <div class="metric"><div class="label">Design Parameters</div><div class="value">${{designParams}}</div><span class="pill ok">${{payload.design_space?.wired_parameter_count || 0}} wired</span></div>
      <div class="metric"><div class="label">Design Variants</div><div class="value">${{designVariants}}</div><span class="pill warn">${{payload.design_space?.candidate_variant_count || 0}} candidates</span></div>
      <div class="metric"><div class="label">Variant Run Plans</div><div class="value">${{variantPlans}}</div><span class="pill ${{completedVariants ? 'ok' : 'warn'}}">${{completedVariants}} executed</span></div>
      <div class="metric"><div class="label">Variant Comparison Rows</div><div class="value">${{comparisonRows}}</div><span class="pill warn">trend only</span></div>
	      <div class="metric"><div class="label">Run Manager Stages</div><div class="value">${{runSummary.stage_row_count || 0}}</div><span class="pill ok">${{runSummary.complete_stage_count || 0}} complete</span></div>
	      <div class="metric"><div class="label">Stale Stages</div><div class="value">${{runSummary.stale_stage_count || 0}}</div><span class="pill ${{runSummary.stale_stage_count ? 'warn' : 'ok'}}">freshness</span></div>
	      <div class="metric"><div class="label">Dataset Catalog</div><div class="value">${{datasetSummary.dataset_count || 0}}</div><span class="pill ok">${{datasetSummary.existing_dataset_count || 0}} existing</span></div>
	      <div class="metric"><div class="label">Orchestrator</div><div class="value">${{escapeText(orchestratorSummary.run_id || 'none')}}</div><span class="pill ${{orchestratorSummary.failed ? 'bad' : 'ok'}}">${{orchestratorSummary.mode || 'dry-run'}}</span></div>
	    </div>
    <div class="actionPanel">
      <h3>${{escapeText(advisor.title)}}</h3>
      <p>${{escapeText(advisor.body)}}</p>
      <div class="actionGrid">
        <div class="actionFact"><div class="k">State</div><div class="v"><span class="pill ${{advisor.stateClass}}">${{escapeText(advisor.stateText)}}</span></div></div>
        <div class="actionFact"><div class="k">Variant</div><div class="v">${{escapeText(advisor.variant)}}</div></div>
        <div class="actionFact"><div class="k">Stage / Reason</div><div class="v">${{escapeText(advisor.stage)}} ${{advisor.reason ? '- ' + escapeText(advisor.reason) : ''}}</div></div>
        <div class="actionFact"><div class="k">Runtime Hint</div><div class="v">${{escapeText(advisor.runtimeHints || '-')}}</div></div>
      </div>
      ${{preflightBlock}}
      ${{previewBlock}}
      ${{executeBlock}}
      ${{advisor.caution ? `<p><span class="pill warn">caution</span>${{escapeText(advisor.caution)}}</p>` : ''}}
    </div>
    <div class="splitTable">
      <h3>Design Flow</h3>
      <table>
        <thead><tr><th>Stage</th><th>Current Implementation</th><th>Next Gap</th></tr></thead>
        <tbody>
          <tr><td>Layout / process</td><td>JSON stack + TCAD profile + Gmsh native split mesh</td><td>GDS/STEP/layer-builder style import</td></tr>
          <tr><td>Optical</td><td>Meep generation maps G(x,depth) and G(x,depth,z)</td><td>broader wavelength/angle/polarization convergence</td></tr>
          <tr><td>Electrical</td><td>DEVSIM profile-PPD split-PD native mesh runs</td><td>TG transient, wider CRA/wavelength sweep, calibrated transport</td></tr>
          <tr><td>Camera response</td><td>native_devsim direct response JSON/NPZ plus gated G*W surrogate diagnostics</td><td>measured stack/n,k and calibrated acceptance targets</td></tr>
          <tr><td>Design Space</td><td>parameter registry with solver wiring and rerun invalidation tags</td><td>automatic variant execution and calibrated acceptance gates</td></tr>
        </tbody>
      </table>
    </div>`;
  bindCopyButtons(root);
}}

function renderDesignSpace() {{
  const space = payload.design_space || {{}};
  const params = space.flat_parameters || [];
  const variants = space.variants || [];
  const materialized = payload.variant_manifest?.variants || [];
  const editOptions = params.map(p => `
    <option value="${{escapeText(p.id || p.path || '')}}">${{escapeText(p.label || p.id || p.path || '')}}</option>
  `).join('');
  const paramRows = params.map(p => `
    <tr>
      <td>${{escapeText(p.group_label || '')}}</td>
      <td>${{escapeText(p.label || p.id)}}</td>
      <td>${{escapeText(p.path || '')}}</td>
      <td>${{escapeText(sig(p.current_value))}}</td>
      <td>${{escapeText(p.unit || '')}}</td>
      <td><span class="pill ${{p.wired_to_solver ? 'ok' : 'warn'}}">${{p.wired_to_solver ? 'solver-wired' : 'metadata-only'}}</span></td>
      <td><div class="stageTags">${{(p.requires_rerun || []).map(s => `<span class="pill warn">${{escapeText(s)}}</span>`).join('')}}</div></td>
      <td>${{escapeText(p.design_intent || '')}}</td>
    </tr>`).join('');
  const variantRows = variants.map(v => `
    <tr>
      <td>${{escapeText(v.label || v.id)}}</td>
      <td><span class="pill ${{v.status === 'simulated_reference' ? 'ok' : 'warn'}}">${{escapeText(v.status || '')}}</span></td>
      <td>${{escapeText(v.goal || '')}}</td>
      <td class="override">${{escapeText(JSON.stringify(v.parameter_overrides || {{}}, null, 2))}}</td>
      <td><div class="stageTags">${{(v.requires_rerun || []).map(s => `<span class="pill warn">${{escapeText(s)}}</span>`).join('')}}</div></td>
      <td>${{escapeText(v.expected_effect || '')}}</td>
    </tr>`).join('');
  const sourceRows = (space.source_basis || []).map(s => `
    <tr>
      <td>${{escapeText(s.topic || '')}}</td>
      <td class="path"><a href="${{escapeText(s.url || '#')}}">${{escapeText(s.url || '')}}</a></td>
      <td>${{escapeText(s.reflected_goal || '')}}</td>
    </tr>`).join('');
  const materializedRows = materialized.map(v => `
    <tr>
      <td>${{escapeText(v.label || v.id)}}</td>
      <td><span class="pill ${{v.status === 'simulated_reference' ? 'ok' : 'warn'}}">${{escapeText(v.status || '')}}</span></td>
      <td><div class="stageTags">${{(v.required_stages || []).map(s => `<span class="pill warn">${{escapeText(s)}}</span>`).join('')}}</div></td>
      <td>${{(v.commands || []).length}}</td>
      <td class="path">${{escapeText(v.generated_files?.project_config || '')}}</td>
      <td class="path">${{escapeText(v.generated_files?.run_plan || '')}}</td>
    </tr>`).join('');
  const editBuilderBlock = params.length ? `
      <div class="designEditBuilder">
        <div class="editHeader">
          <div>
            <h3>Design Edit Command Builder</h3>
            <p>Preview or materialize one parameter override through image_sensor_design_variant_create.py. Browser execution is intentionally disabled.</p>
          </div>
          <span class="pill warn">copy CLI</span>
        </div>
        <div class="editGrid">
          <label>Parameter
            <select id="editParam">${{editOptions}}</select>
          </label>
          <label>New Value
            <input id="editValue" autocomplete="off">
          </label>
          <label>Variant ID
            <input id="editVariantId" autocomplete="off">
          </label>
        </div>
        <div id="editParamMeta"></div>
        <div class="editFacts">
          <div class="editFact"><div class="k">Path</div><div class="v" id="editPath"></div></div>
          <div class="editFact"><div class="k">Current</div><div class="v" id="editCurrent"></div></div>
          <div class="editFact"><div class="k">Required Rerun</div><div class="v" id="editStages"></div></div>
          <div class="editFact"><div class="k">Accuracy State</div><div class="v"><span class="pill bad">not product LUT</span></div></div>
        </div>
        <div id="editWarning" class="editWarning"></div>
        <div class="command compact">
          <div><b>Preview variant plan</b><code id="editPreviewCommand"></code></div>
          <button class="toolButton" id="copyPreviewCommand" data-copy="">Copy</button>
        </div>
        <div class="command compact">
          <div><b>Materialize variant and refresh Studio</b><code id="editMaterializeCommand"></code></div>
          <button class="toolButton" id="copyMaterializeCommand" data-copy="">Copy</button>
        </div>
      </div>` : '';
  const compareRows = payload.variant_comparison_rows || [];
  const comparisonRows = compareRows.map(r => `
    <tr>
      <td>${{escapeText(r.variant_id)}}</td>
      <td><span class="pill ${{stateClass(r.variant_state)}}">${{escapeText(r.variant_state)}}</span></td>
      <td>${{escapeText(r.case)}}</td>
      <td>${{escapeText(r.cra_x_deg)}}</td>
      <td>${{exp(r.baseline_total_photo_delta_a_per_cm)}}</td>
      <td>${{exp(r.variant_total_photo_delta_a_per_cm)}}</td>
      <td>${{pct(r.total_photo_delta_rel_change)}}</td>
      <td>${{sig(r.baseline_split_phase_x_proxy)}}</td>
      <td>${{sig(r.variant_split_phase_x_proxy)}}</td>
      <td>${{sig(r.split_phase_delta)}}</td>
      <td>${{r.split_phase_sign_changed === 'true' ? '<span class="pill bad">sign flip</span>' : ''}}</td>
      <td>${{sig(r.gw_total_reference_scaled_rel_error)}}</td>
      <td>${{sig(r.gw_mesh_total_reference_scaled_rel_error)}}</td>
      <td>${{sig(r.gw_devsim_laplace_total_reference_scaled_rel_error)}}</td>
      <td class="override">${{escapeText(r.parameter_overrides || '')}}</td>
    </tr>`).join('');
  const materializedBlock = materializedRows ? `
      <div>
        <h3>Materialized Run Plans</h3>
        <table>
          <thead><tr><th>Variant</th><th>Status</th><th>Stages</th><th>Commands</th><th>Project Config</th><th>Run Plan</th></tr></thead>
          <tbody>${{materializedRows}}</tbody>
        </table>
      </div>` : '';
  const comparisonBlock = comparisonRows ? `
      <div>
        <h3>Variant Comparison <a href="${{escapeText(payload.paths.variant_comparison_report || '#')}}">open report</a></h3>
        <table>
          <thead><tr><th>Variant</th><th>State</th><th>Case</th><th>CRA X</th><th>Base Total</th><th>Variant Total</th><th>Total Change</th><th>Base Split</th><th>Variant Split</th><th>Split Delta</th><th>Flag</th><th>G*W Err</th><th>W_mesh Err</th><th>W_devsim Err</th><th>Overrides</th></tr></thead>
          <tbody>${{comparisonRows}}</tbody>
        </table>
      </div>` : '';
  document.getElementById('tab-design-space').innerHTML = `
    ${{editBuilderBlock}}
    <div class="designGrid">
      <div>
        <h3>Parameter Registry</h3>
        <table>
          <thead><tr><th>Group</th><th>Parameter</th><th>Path</th><th>Current</th><th>Unit</th><th>State</th><th>Rerun</th><th>Intent</th></tr></thead>
          <tbody>${{paramRows}}</tbody>
        </table>
      </div>
      <div>
        <h3>Design Variants</h3>
        <table>
          <thead><tr><th>Variant</th><th>Status</th><th>Goal</th><th>Overrides</th><th>Rerun</th><th>Expected Effect</th></tr></thead>
          <tbody>${{variantRows}}</tbody>
        </table>
      </div>
      ${{materializedBlock}}
      ${{comparisonBlock}}
      <div>
        <h3>Public UX/Workflow Sources</h3>
        <table>
          <thead><tr><th>Topic</th><th>URL</th><th>Reflected Goal</th></tr></thead>
          <tbody>${{sourceRows}}</tbody>
        </table>
      </div>
    </div>`;
  initDesignEditBuilder(params);
}}

	function renderRunManager() {{
	  const rows = payload.run_stage_rows || [];
	  const summary = payload.run_manager?.summary || {{}};
	  const tableRows = rows.map(r => `
	    <tr>
	      <td>${{escapeText(r.variant_id)}}</td>
	      <td>${{escapeText(r.stage)}}</td>
	      <td><span class="pill ${{statusClass(r.status)}}">${{escapeText(r.status)}}</span></td>
	      <td><span class="pill ${{statusClass(r.freshness)}}">${{escapeText(r.freshness || '')}}</span></td>
	      <td>${{escapeText(r.completed_outputs)}}</td>
	      <td>${{escapeText(r.missing_outputs)}}</td>
	      <td>${{escapeText(r.blocked_by_missing_upstream)}}</td>
	      <td>${{escapeText(r.stale_reason || '')}}</td>
	      <td>${{escapeText(r.newest_input_mtime || '')}}</td>
	      <td>${{escapeText(r.oldest_output_mtime || '')}}</td>
	      <td>${{escapeText(r.command_count)}}</td>
	      <td class="override">${{escapeText(r.first_command)}}</td>
	    </tr>`).join('');
  document.getElementById('tab-run-manager').innerHTML = `
    <div class="summaryGrid">
	      <div class="metric"><div class="label">Stage Rows</div><div class="value">${{summary.stage_row_count || rows.length}}</div><span class="pill warn">file inferred</span></div>
	      <div class="metric"><div class="label">Complete Stages</div><div class="value">${{summary.complete_stage_count || 0}}</div><span class="pill ok">done</span></div>
	      <div class="metric"><div class="label">Fresh Stages</div><div class="value">${{summary.fresh_stage_count || 0}}</div><span class="pill ok">current inputs</span></div>
	      <div class="metric"><div class="label">Stale Stages</div><div class="value">${{summary.stale_stage_count || 0}}</div><span class="pill ${{summary.stale_stage_count ? 'warn' : 'ok'}}">needs rerun</span></div>
	      <div class="metric"><div class="label">Missing Stages</div><div class="value">${{summary.missing_stage_count || 0}}</div><span class="pill bad">not run</span></div>
	      <div class="metric"><div class="label">Blocked Stages</div><div class="value">${{summary.blocked_stage_count || 0}}</div><span class="pill warn">upstream</span></div>
	    </div>
	    <table>
	      <thead><tr><th>Variant</th><th>Stage</th><th>Status</th><th>Freshness</th><th>Completed Outputs</th><th>Missing Outputs</th><th>Blocked By</th><th>Stale Reason</th><th>Newest Input</th><th>Oldest Output</th><th>Commands</th><th>First Command</th></tr></thead>
	      <tbody>${{tableRows}}</tbody>
	    </table>`;
	}}

	function renderOrchestrator() {{
	  const root = document.getElementById('tab-orchestrator');
	  const run = payload.orchestrator_last_run || {{}};
	  const summary = run.summary || {{}};
	  const plan = run.plan || [];
	  const executed = run.executed_commands || [];
	  const refresh = run.refresh_results || [];
	  const history = payload.orchestrator_history || [];
	  const dryRun = (payload.project.runbook || []).find(cmd => cmd.id === 'variant_orchestrator_dry_run')?.command ||
	    '.tcad-env/bin/python image_sensor_variant_orchestrator.py --all --next-needed';
	  const selected = (summary.selected_variants || []).join(', ');
	  const planRows = plan.map(item => {{
	    const firstCommand = (item.commands || [])[0]?.command || '';
	    const cls = item.action === 'run' ? 'ok' : (item.action === 'blocked' ? 'bad' : 'warn');
	    const preflight = item.preflight_summary || {{}};
	    const preflightStatus = preflight.status || 'not recorded';
	    return `
	      <tr>
	        <td>${{escapeText(item.variant_id)}}</td>
	        <td>${{escapeText(item.stage)}}</td>
	        <td><span class="pill ${{cls}}">${{escapeText(item.action)}}</span></td>
	        <td><span class="pill ${{statusClass(preflightStatus)}}">${{escapeText(preflightStatus)}}</span></td>
	        <td>${{escapeText(runtimeHintText(item))}}</td>
	        <td><span class="pill ${{statusClass(item.status_before)}}">${{escapeText(item.status_before)}}</span></td>
	        <td><span class="pill ${{statusClass(item.freshness_before || '')}}">${{escapeText(item.freshness_before || '')}}</span></td>
	        <td><span class="pill ${{statusClass(item.status_after || '')}}">${{escapeText(item.status_after || '')}}</span></td>
	        <td><span class="pill ${{statusClass(item.freshness_after || '')}}">${{escapeText(item.freshness_after || '')}}</span></td>
	        <td>${{escapeText((item.expected_missing_before || []).join(', '))}}</td>
	        <td>${{escapeText(item.reason || '')}}</td>
	        <td>${{escapeText(preflightIssueText(item))}}</td>
	        <td class="override">${{escapeText(firstCommand)}}</td>
	        <td>${{firstCommand ? `<button class="toolButton" data-copy="${{escapeText(firstCommand)}}">Copy</button>` : ''}}</td>
	      </tr>`;
	  }}).join('');
	  const executedRows = executed.map(item => `
	    <tr>
	      <td>${{escapeText(item.command_id)}}</td>
	      <td>${{escapeText(item.stage)}}</td>
	      <td><span class="pill ${{Number(item.return_code) === 0 ? 'ok' : 'bad'}}">${{escapeText(item.return_code)}}</span></td>
	      <td>${{sig(item.elapsed_s,4)}} s</td>
	      <td>${{item.timed_out ? '<span class="pill bad">timeout</span>' : ''}}</td>
	      <td class="path">${{escapeText(item.log || '')}}</td>
	    </tr>`).join('');
	  const refreshRows = refresh.map(item => `
	    <tr>
	      <td class="override">${{escapeText(item.command)}}</td>
	      <td><span class="pill ${{Number(item.return_code) === 0 ? 'ok' : 'bad'}}">${{escapeText(item.return_code)}}</span></td>
	      <td>${{sig(item.elapsed_s,4)}} s</td>
	      <td class="override">${{escapeText(item.stdout_tail || item.stderr_tail || '')}}</td>
	    </tr>`).join('');
	  const historyRows = history.slice().reverse().map(entry => {{
	    const s = entry.summary || {{}};
	    return `
	      <tr>
	        <td>${{escapeText(s.run_id || '')}}</td>
	        <td>${{escapeText(s.mode || '')}}</td>
	        <td>${{escapeText(s.phase || '')}}</td>
	        <td>${{escapeText((s.selected_variants || []).join(', '))}}</td>
	        <td>${{escapeText(s.plan_rows ?? '')}}</td>
	        <td>${{escapeText(s.planned_stale_rows ?? '')}}</td>
	        <td>${{escapeText(s.executed_command_count ?? '')}}</td>
	        <td><span class="pill ${{s.failed ? 'bad' : 'ok'}}">${{s.failed ? 'failed' : 'ok'}}</span></td>
	        <td>${{s.refresh_ran ? '<span class="pill ok">refresh</span>' : ''}}</td>
	      </tr>`;
	  }}).join('');
	  root.innerHTML = `
	    <div class="summaryGrid">
	      <div class="metric"><div class="label">Last Run</div><div class="value">${{escapeText(summary.run_id || 'none')}}</div><span class="pill ${{summary.failed ? 'bad' : 'ok'}}">${{summary.mode || 'dry-run'}}</span></div>
	      <div class="metric"><div class="label">Selected Variants</div><div class="value">${{escapeText(selected || '-')}}</div><span class="pill warn">local runner</span></div>
	      <div class="metric"><div class="label">Planned Runs</div><div class="value">${{summary.planned_run_rows ?? 0}}</div><span class="pill ${{summary.planned_run_rows ? 'ok' : 'warn'}}">${{summary.plan_rows ?? plan.length}} rows</span></div>
	      <div class="metric"><div class="label">Planned Stale</div><div class="value">${{summary.planned_stale_rows ?? 0}}</div><span class="pill ${{summary.planned_stale_rows ? 'warn' : 'ok'}}">freshness</span></div>
	      <div class="metric"><div class="label">Executed Commands</div><div class="value">${{summary.executed_command_count ?? 0}}</div><span class="pill ${{summary.refresh_ran ? 'ok' : 'warn'}}">${{summary.refresh_ran ? 'refreshed' : 'no refresh'}}</span></div>
	    </div>
	    <div class="command">
	      <div><b>Preview next missing or stale stages</b><code>${{escapeText(dryRun)}}</code></div>
	      <button class="toolButton" data-copy="${{escapeText(dryRun)}}">Copy</button>
	    </div>
	    <h3>Last Run Plan</h3>
	    <table>
	      <thead><tr><th>Variant</th><th>Stage</th><th>Action</th><th>Preflight</th><th>Runtime Hint</th><th>Status Before</th><th>Fresh Before</th><th>Status After</th><th>Fresh After</th><th>Missing Before</th><th>Reason</th><th>Preflight Detail</th><th>Command</th><th>Copy</th></tr></thead>
	      <tbody>${{planRows || '<tr><td colspan="14">No orchestrator plan recorded.</td></tr>'}}</tbody>
	    </table>
	    <h3>Executed Commands</h3>
	    <table>
	      <thead><tr><th>Command ID</th><th>Stage</th><th>Return</th><th>Elapsed</th><th>Timeout</th><th>Log</th></tr></thead>
	      <tbody>${{executedRows || '<tr><td colspan="6">No commands executed in the last run.</td></tr>'}}</tbody>
	    </table>
	    <h3>Refresh Results</h3>
	    <table>
	      <thead><tr><th>Command</th><th>Return</th><th>Elapsed</th><th>Output Tail</th></tr></thead>
	      <tbody>${{refreshRows || '<tr><td colspan="4">No refresh recorded.</td></tr>'}}</tbody>
	    </table>
	    <h3>Run History</h3>
	    <table>
	      <thead><tr><th>Run ID</th><th>Mode</th><th>Phase</th><th>Variants</th><th>Plan Rows</th><th>Stale Rows</th><th>Executed</th><th>Status</th><th>Refresh</th></tr></thead>
	      <tbody>${{historyRows || '<tr><td colspan="9">No history recorded.</td></tr>'}}</tbody>
	    </table>
	    <p><span class="pill warn">Not a scheduler</span><span class="pill bad">Not an accuracy gate</span></p>`;
	  bindCopyButtons(root);
	}}

	function renderDatasets() {{
  const rows = payload.dataset_catalog_rows || [];
  const summary = payload.dataset_catalog?.summary || payload.run_manager?.summary || {{}};
  const tableRows = rows.map(r => `
    <tr>
      <td>${{escapeText(r.dataset_id)}}</td>
      <td>${{escapeText(r.solver)}}</td>
      <td>${{escapeText(r.dataset_kind)}}</td>
      <td>${{escapeText(r.role)}}</td>
      <td>${{escapeText(r.dimensionality)}}</td>
      <td><span class="pill ${{statusClass(r.exists)}}">${{escapeText(r.exists)}}</span></td>
      <td>${{r.native_mesh === 'true' ? '<span class="pill ok">native</span>' : '<span class="pill warn">derived</span>'}}</td>
      <td>${{fmtBytes(r.size_bytes ? Number(r.size_bytes) : null)}}</td>
      <td class="path">${{escapeText(r.path)}}</td>
    </tr>`).join('');
  document.getElementById('tab-datasets').innerHTML = `
    <div class="summaryGrid">
      <div class="metric"><div class="label">Datasets</div><div class="value">${{summary.dataset_count || rows.length}}</div><span class="pill ok">cataloged</span></div>
      <div class="metric"><div class="label">Existing</div><div class="value">${{summary.existing_dataset_count || 0}}</div><span class="pill ok">present</span></div>
      <div class="metric"><div class="label">Native Mesh/Data</div><div class="value">${{summary.native_dataset_count || 0}}</div><span class="pill ok">solver-native</span></div>
      <div class="metric"><div class="label">Missing</div><div class="value">${{summary.missing_dataset_count || 0}}</div><span class="pill bad">expected</span></div>
    </div>
    <table>
      <thead><tr><th>Dataset</th><th>Solver</th><th>Kind</th><th>Role</th><th>Dim</th><th>Exists</th><th>Native</th><th>Size</th><th>Path</th></tr></thead>
      <tbody>${{tableRows}}</tbody>
    </table>`;
}}

function renderSplit() {{
  document.getElementById('tab-split').innerHTML = `
    <table>
      <thead><tr><th>Case</th><th>Mesh</th><th>Nodes</th><th>Total dI</th><th>Split Phase</th><th>Transport</th><th>mu n/p</th><th>Terminal Balance</th><th>Summary</th></tr></thead>
      <tbody>${{payload.native_runs.map(r => `
        <tr>
          <td>${{escapeText(r.case)}}</td><td>${{escapeText(r.mesh_source)}}</td><td>${{r.node_count}}</td>
          <td>${{exp(r.total_photo_delta_a_per_cm)}}</td><td>${{sig(r.photo_split_phase_x_proxy)}}</td>
          <td>${{escapeText(r.field_mobility_model || r.transport_model || '')}}</td>
          <td>${{sig(r.electron_mobility_effective_edge_min_cm2_v_s)}}-${{sig(r.electron_mobility_effective_edge_max_cm2_v_s)}} / ${{sig(r.hole_mobility_effective_edge_min_cm2_v_s)}}-${{sig(r.hole_mobility_effective_edge_max_cm2_v_s)}}</td>
          <td>${{exp(r.terminal_balance_illuminated_a_per_cm,3)}}</td>
          <td class="path">${{escapeText(r.summary_json)}}</td>
        </tr>`).join('')}}</tbody>
    </table>`;
}}

function renderAccuracy() {{
  const head = `<div class="metric"><div class="label">Accuracy Status</div><div class="value">${{payload.accuracy.accuracy_ready ? 'PASS' : 'NOT READY'}}</div><span class="pill bad">do not use as product LUT</span></div>`;
  const issues = payload.accuracy.blocking.map(i => `<div class="issue"><b>${{escapeText(i.name)}}</b> <span class="pill bad">${{escapeText(i.status)}}</span><div>${{escapeText(i.details)}}</div></div>`).join('');
  document.getElementById('tab-accuracy').innerHTML = head + `<h3>Blocking Items</h3>` + (issues || '<p>No blocking items recorded.</p>');
}}

	function renderRunbook() {{
	  const root = document.getElementById('tab-runbook');
	  root.innerHTML = payload.project.runbook.map(cmd => `
	    <div class="command">
	      <div><b>${{escapeText(cmd.label)}}</b><code>${{escapeText(cmd.command)}}</code></div>
	      <button class="toolButton" data-copy="${{escapeText(cmd.command)}}">Copy</button>
	    </div>`).join('');
	  bindCopyButtons(root);
	}}

function renderResults() {{
  const filter = (document.getElementById('resultFilter').value || '').toLowerCase();
  const groups = (payload.result_groups || []).filter(g => {{
    const haystack = [
      g.id,
      g.object_label,
      g.object_kind,
      g.result_role,
      g.result_label,
      g.native_state,
      g.readiness,
      (g.kinds || []).join(' '),
      (g.dimensionalities || []).join(' '),
      (g.viewer_modes || []).join(' '),
      (g.datasets || []).map(d => [d.dataset_id, d.kind, d.path, d.viewer_mode].join(' ')).join(' ')
    ].join(' ').toLowerCase();
    return haystack.includes(filter);
  }});
  const rows = payload.results.filter(r => {{
    const status = r.exists ? 'exists' : 'missing';
    const mesh = r.native_mesh ? 'native' : 'derived';
    const haystack = [r.id, r.kind, r.label, r.path, r.relative_path, status, mesh].join(' ').toLowerCase();
    return haystack.includes(filter);
  }});
  document.getElementById('resultCount').textContent = `${{groups.length}} / ${{payload.result_groups?.length || 0}} groups · ${{rows.length}} / ${{payload.results.length}} files`;
  document.getElementById('resultGroups').innerHTML = `
    <h3>Object Results</h3>
    <table>
      <thead><tr><th>Object</th><th>Result</th><th>Datasets</th><th>Native</th><th>Kinds</th><th>Dims</th><th>Viewer</th><th>Readiness</th><th>Primary</th></tr></thead>
      <tbody>${{groups.map(g => `
        <tr>
          <td>${{escapeText(g.object_label)}}</td>
          <td>${{escapeText(g.result_label || g.result_role)}}</td>
          <td>${{escapeText(g.existing_count)}} / ${{escapeText(g.dataset_count)}}</td>
          <td><span class="pill ${{g.native_state === 'native' ? 'ok' : (g.native_state === 'mixed' ? 'warn' : 'warn')}}">${{escapeText(g.native_state)}}</span></td>
          <td>${{escapeText((g.kinds || []).join(', '))}}</td>
          <td>${{escapeText((g.dimensionalities || []).join(', '))}}</td>
          <td>${{g.viewer_count ? '<span class="pill ok">open ' + escapeText(g.viewer_count) + '</span>' : '<span class="pill warn">data only</span>'}}</td>
          <td><span class="pill bad">${{escapeText(g.readiness || 'product-blocked')}}</span></td>
          <td class="path">${{g.primary_viewer ? `<a href="${{escapeText(g.primary_viewer)}}">open viewer</a>` : (g.primary_path ? `<a href="${{escapeText(g.primary_path)}}">open data</a>` : '')}}</td>
        </tr>`).join('')}}</tbody>
    </table>`;
  document.getElementById('results').innerHTML = `
    <thead><tr><th>Status</th><th>Kind</th><th>Native</th><th>Size</th><th>Label</th><th>Path</th></tr></thead>
    <tbody>${{rows.map(r => `
      <tr>
        <td><span class="pill ${{r.exists?'ok':'bad'}}">${{r.exists?'exists':'missing'}}</span></td>
        <td>${{escapeText(r.kind)}}</td>
        <td>${{r.native_mesh ? '<span class="pill ok">native</span>' : '<span class="pill warn">derived</span>'}}</td>
        <td>${{fmtBytes(r.size_bytes)}}</td>
        <td>${{escapeText(r.label)}}</td>
        <td class="path"><a href="${{escapeText(r.relative_path)}}">${{escapeText(r.path)}}</a></td>
      </tr>`).join('')}}</tbody>`;
}}

function initTabs() {{
  const tabs = document.querySelectorAll('#tabs button');
  tabs.forEach(btn => btn.onclick = () => {{
    tabs.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.viewPane').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${{btn.dataset.tab}}`).classList.add('active');
  }});
  document.getElementById('crossFrame').src = payload.paths.cross_section_2d;
  document.getElementById('geometryFrame').src = payload.paths.geometry_3d;
  document.getElementById('reportFrame').src = payload.paths.parameter_report;
  document.getElementById('gwFrame').src = payload.paths.gw_coupling_report || 'about:blank';
  document.getElementById('cameraLutFrame').src = payload.paths.camera_system_research_lut_report || payload.paths.camera_system_lut_report || 'about:blank';
}}

initHeader();
renderTree();
renderProperties();
renderOverview();
	renderDesignSpace();
	renderRunManager();
	renderOrchestrator();
	renderDatasets();
renderSplit();
renderAccuracy();
renderRunbook();
renderResults();
initTabs();
document.getElementById('resultFilter').oninput = renderResults;
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_html_inline_workbench(path: Path, payload: dict[str, Any]) -> None:
    """Write the simplified Pixel Workbench shell.

    The previous Studio kept a Lumerical-like object/result-management layout.
    The current UX intentionally starts from the pixel engineer's loop:
    configure a small set of pixel parameters, inspect field/absorption, read
    a compact KPI dashboard, and compare variants.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(payload, ensure_ascii=False)
    app_title = html.escape(payload["project"]["project"]["name"])
    html_doc = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__APP_TITLE__ - Pixel Workbench</title>
<style>
:root{
  --bg:#071018;
  --panel:#0d1822;
  --panel2:#101d28;
  --line:#223441;
  --line2:#315166;
  --text:#dcecf5;
  --muted:#8ca5b5;
  --cyan:#22d3ee;
  --blue:#38bdf8;
  --green:#74c365;
  --yellow:#f6c445;
  --orange:#f59e0b;
  --red:#ef4444;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 50% -20%,#163246 0,#071018 42%,#02060a 100%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.app{min-height:100vh;display:grid;grid-template-rows:42px minmax(0,1fr);gap:6px;padding:6px}
.topbar{display:flex;align-items:center;justify-content:space-between;border:1px solid var(--line);background:rgba(7,16,24,.92);border-radius:6px;padding:0 12px}
.brand{font-weight:800;color:var(--cyan);font-size:20px;letter-spacing:0}
.crumbs{display:flex;gap:12px;color:var(--muted);font-size:12px;align-items:center}
.badge{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 8px;border:1px solid var(--line2);border-radius:5px;color:var(--muted);background:#0a141c;font-size:11px}
.badge.ok{color:#c8f7d2;border-color:#376a44}.badge.warn{color:#ffe2a3;border-color:#765d20}.badge.bad{color:#fecaca;border-color:#7f1d1d}
.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1.08fr;gap:6px;min-height:0}
.panel{border:1px solid var(--line);background:linear-gradient(180deg,rgba(13,24,34,.96),rgba(7,15,22,.96));border-radius:7px;min-height:0;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.28)}
.panelHeader{height:36px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid var(--line);color:var(--cyan);font-weight:800;font-size:16px}
.panelBody{height:calc(100% - 36px);min-height:0;padding:10px}
.workspace{display:grid;grid-template-columns:140px minmax(0,1fr) 235px;gap:10px;height:100%}
.sideNav{border-right:1px solid var(--line);padding-right:8px;display:grid;grid-template-rows:1fr auto;gap:8px}
.navItems{display:grid;gap:6px;align-content:start}
.navItem{display:flex;align-items:center;gap:8px;height:32px;border:1px solid transparent;border-radius:6px;padding:0 8px;color:#c9dce7;background:transparent;font-size:13px}
.navItem.active{border-color:#1a95ad;background:rgba(34,211,238,.12);color:#e8fbff}
.newVariant{height:34px;border:1px solid var(--line2);background:#0b1720;color:#e9fbff;border-radius:6px}
.pixelCanvas{display:grid;grid-template-rows:minmax(0,1fr) 28px;gap:8px;min-width:0}
.pixelSvgWrap{border:1px solid var(--line);border-radius:6px;background:#09131c;overflow:hidden;min-height:0}
.statusStrip{display:flex;align-items:center;gap:12px;border:1px solid var(--line);border-radius:6px;padding:0 8px;color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden}
.controls{display:grid;grid-template-rows:auto 1fr auto;gap:8px;min-width:0}
.controlGroup{border:1px solid var(--line);border-radius:6px;background:#0b1720;padding:8px}
.controlGroup h3{margin:0 0 8px 0;color:#d7f8ff;font-size:12px;font-weight:700}
.controlRow{display:grid;grid-template-columns:1fr 64px 34px;gap:6px;align-items:center;margin:6px 0;font-size:12px;color:#c8dae5}
.controlRow input{width:100%;height:24px;background:#071018;color:#e9fbff;border:1px solid #2a4658;border-radius:4px;text-align:right;padding:0 5px}
.unit{color:var(--muted);font-size:11px}
.guardrail{display:grid;gap:6px;font-size:11px;color:#cbdce7}
.guardrail div{display:flex;justify-content:space-between;border-bottom:1px solid rgba(49,81,102,.45);padding-bottom:4px}
.fieldLayout{display:grid;grid-template-columns:132px minmax(0,1fr) 74px;gap:10px;height:100%}
.fieldControls{display:grid;gap:8px;align-content:start}
.fieldCard{border:1px solid var(--line);background:#0b1720;border-radius:6px;padding:8px}
.fieldCard h3{margin:0 0 8px 0;color:#7dd3fc;font-size:12px}
.fieldCard label{display:block;margin:6px 0 3px 0;color:var(--muted);font-size:11px}
.fieldCard select,.fieldCard input{width:100%;height:26px;background:#071018;color:#e9fbff;border:1px solid #2a4658;border-radius:4px}
.modeButtons{display:flex;gap:6px}
.modeButtons button{height:28px;padding:0 12px;border:1px solid var(--line);background:#0b1720;color:#d9edf7;border-radius:5px}
.modeButtons button.active{border-color:#18a9c7;color:white;background:rgba(34,211,238,.14)}
.fieldImage{position:relative;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:radial-gradient(circle at 50% 45%,rgba(34,211,238,.28),rgba(2,6,10,.9) 48%)}
.fieldImage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:.88}
.fieldOverlay{position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,16,24,.58),rgba(7,16,24,.08),rgba(7,16,24,.58));pointer-events:none}
.fieldLabel{position:absolute;left:12px;top:12px;color:#dffbff;font-size:13px;text-shadow:0 1px 2px #000}
.colorbar{display:grid;grid-template-rows:auto 1fr auto;gap:6px;align-items:center;justify-items:center;color:var(--muted);font-size:11px}
.bar{width:22px;border-radius:4px;background:linear-gradient(0deg,#001a70,#0065ff,#00f0ff,#fff000,#ff7a00,#b00020);border:1px solid var(--line);height:78%}
.kpiGrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:8px}
.kpi{border:1px solid var(--line2);background:linear-gradient(180deg,#101f28,#0a141b);border-radius:6px;padding:10px;min-width:0}
.kpi .label{font-size:12px;color:#c7d9e3;text-align:center}.kpi .value{font-size:28px;font-weight:800;text-align:center;margin:5px 0}.kpi .target{font-size:11px;color:var(--muted);text-align:center}
.green{color:var(--green)}.yellow{color:var(--yellow)}.orange{color:var(--orange)}.cyan{color:var(--cyan)}
.charts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-template-rows:1fr 1fr;gap:8px;height:calc(100% - 104px);min-height:0}
.chart{border:1px solid var(--line);background:#09131c;border-radius:6px;padding:8px;min-height:0;overflow:hidden}
.chart.wide{grid-column:span 4}
.chart h3{margin:0 0 5px 0;color:#cfe9f4;font-size:12px;font-weight:700}
.chart svg{width:100%;height:calc(100% - 20px);min-height:86px}
.compareLayout{display:grid;grid-template-columns:1.1fr .9fr;grid-template-rows:132px minmax(0,1fr) 52px;gap:8px;height:100%}
.thumbs{grid-column:span 1;border:1px solid var(--line);border-radius:6px;padding:8px;background:#09131c;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.thumb{border:1px solid #284558;border-radius:5px;background:#0d1822;padding:6px;color:#d8ebf5;font-size:11px}
.miniStack{height:72px;border-radius:4px;background:linear-gradient(#8bd2ff 0 24%,#1e40af 24% 37%,#1a1a1a 37% 56%,#0f172a 56% 100%);position:relative;overflow:hidden;border:1px solid #355267}
.miniStack:before{content:"";position:absolute;left:16%;right:16%;top:3px;height:28px;border-radius:50% 50% 0 0;background:rgba(177,225,255,.85)}
.miniStack:after{content:"";position:absolute;left:35%;right:35%;bottom:11px;height:24px;border-radius:50% 50% 0 0;background:#d6a420}
.radar{border:1px solid var(--line);border-radius:6px;background:#09131c;padding:8px}
.radar h3{margin:0;color:#cfe9f4;font-size:12px}
.radar svg{width:100%;height:calc(100% - 18px)}
.variantTable{grid-column:span 2;min-height:0;overflow:auto;border:1px solid var(--line);border-radius:6px;background:#071018}
table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}
th,td{border-bottom:1px solid #203544;padding:7px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
th{color:#9fb7c6;background:#0c1720;position:sticky;top:0}
tr.selected td{color:#ffe58a;background:rgba(246,196,69,.08);border-top:1px solid #8a6b10;border-bottom:1px solid #8a6b10}
.best{grid-column:span 1;border:1px solid #8a6b10;border-radius:6px;background:rgba(246,196,69,.08);padding:10px;display:flex;align-items:center;justify-content:space-between}
.tolerance{border:1px solid var(--line);border-radius:6px;background:#09131c;padding:8px;display:grid;grid-template-columns:repeat(4,1fr);gap:6px;color:var(--muted);font-size:11px;text-align:center}
.small{font-size:11px;color:var(--muted)}
@media(max-width:1200px){.grid{grid-template-columns:1fr;grid-template-rows:auto}.panel{min-height:520px}.workspace,.fieldLayout,.compareLayout{grid-template-columns:1fr}.kpiGrid,.charts{grid-template-columns:repeat(2,1fr)}.chart.wide{grid-column:span 2}.compareLayout .variantTable,.compareLayout .thumbs{grid-column:span 1}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">Image Sensor Pixel Workbench</div>
    <div class="crumbs">
      <span>Project</span><span>Pixel</span><span>Experiment</span><span>Compare</span><span>Export</span>
      <span class="badge bad" id="accuracyBadge">accuracy blocked</span>
      <span class="badge ok" id="runBadge">run status</span>
    </div>
  </header>
  <main class="grid">
    <section class="panel">
      <div class="panelHeader"><span>1. Pixel Design Workspace</span><span class="badge warn">proxy stack</span></div>
      <div class="panelBody workspace">
        <aside class="sideNav">
          <div class="navItems">
            <div class="navItem">Template</div>
            <div class="navItem active">Geometry</div>
            <div class="navItem">ML / OCL</div>
            <div class="navItem">CFA</div>
            <div class="navItem">DTI</div>
            <div class="navItem">Shield</div>
            <div class="navItem">Materials</div>
            <div class="navItem">Illumination</div>
          </div>
          <button class="newVariant">+ New Variant</button>
        </aside>
        <div class="pixelCanvas">
          <div class="pixelSvgWrap" id="pixelDiagram"></div>
          <div class="statusStrip" id="statusStrip"></div>
        </div>
        <aside class="controls">
          <div class="controlGroup">
            <h3>Geometry Controls</h3>
            <div id="geometryControls"></div>
          </div>
          <div class="controlGroup">
            <h3>Design Rule Guardrail</h3>
            <div class="guardrail" id="guardrail"></div>
          </div>
          <button class="newVariant" id="resetButton">Reset to Template</button>
        </aside>
      </div>
    </section>

    <section class="panel">
      <div class="panelHeader"><span>2. Optical Field / Absorption Viewer</span><div class="modeButtons"><button class="active">|E|2</button><button>Absorption</button><button>Collection</button><button>2D</button></div></div>
      <div class="panelBody fieldLayout">
        <aside class="fieldControls">
          <div class="fieldCard">
            <h3>Illumination</h3>
            <label>CRA</label><select><option>20 deg edge</option><option>0 deg center</option></select>
            <label>Azimuth</label><select><option>0 deg</option></select>
            <label>Polarization</label><select><option>Unpolarized</option></select>
          </div>
          <div class="fieldCard">
            <h3>Wavelength</h3>
            <select><option>550 nm</option><option>450-940 sweep</option></select>
            <div class="small">400 550 700 940 nm</div>
          </div>
          <div class="fieldCard">
            <h3>Overlay</h3>
            <div class="small">Layer boundaries</div>
            <div class="small">Energy flow arrows</div>
          </div>
        </aside>
        <div class="fieldImage">
          <img id="fieldImage" alt="field map">
          <div class="fieldOverlay"></div>
          <div class="fieldLabel" id="fieldLabel"></div>
        </div>
        <aside class="colorbar"><div>|E|2</div><div class="bar"></div><div>1e-5</div></aside>
      </div>
    </section>

    <section class="panel">
      <div class="panelHeader"><span>3. KPI Dashboard</span><span class="badge warn">trend / proxy</span></div>
      <div class="panelBody">
        <div class="kpiGrid" id="kpiGrid"></div>
        <div class="charts">
          <div class="chart"><h3>QE Proxy vs Wavelength</h3><svg id="qeChart"></svg></div>
          <div class="chart"><h3>Angular Response vs CRA</h3><svg id="angleChart"></svg></div>
          <div class="chart"><h3>Crosstalk Matrix</h3><svg id="matrixChart"></svg></div>
          <div class="chart"><h3>Color Separation</h3><svg id="colorChart"></svg></div>
          <div class="chart wide"><h3>DTI Depth Sensitivity</h3><svg id="dtiChart"></svg></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panelHeader"><span>4. Variant Compare / Optimization</span><span class="badge warn">objective: maximize score</span></div>
      <div class="panelBody compareLayout">
        <div class="thumbs" id="variantThumbs"></div>
        <div class="radar"><h3>Trade-off Radar</h3><svg id="radarChart"></svg></div>
        <div class="variantTable"><table id="variantTable"></table></div>
        <div class="best" id="bestVariant"></div>
        <div class="tolerance"><div>ML Shift<br>+-20 nm</div><div>CF Shift<br>+-20 nm</div><div>DTI Width<br>+-10 nm</div><div>DTI Depth<br>+-0.10 um</div></div>
      </div>
    </section>
  </main>
</div>
<script>
const payload = __PAYLOAD__;
const stack = payload.stack || {};
const profile = payload.profile || {};
const geom = stack.geometry_um || {};
const pgeom = profile.geometry || {};
const gwCases = payload.gw_coupling?.cases || [];
const centerCase = gwCases.find(c => c.case === 'center') || gwCases[0] || {};
const edgeCase = gwCases.find(c => c.case === 'edge20x') || gwCases[1] || centerCase;
const comparisonRows = payload.variant_comparison_rows || [];
let selectedVariant = null;

function clamp(v,min=0,max=1){ return Math.min(max, Math.max(min, v)); }
function num(v,d=0){ const x=Number(v); return Number.isFinite(x)?x.toFixed(d):'-'; }
function pct(v,d=1){ const x=Number(v); return Number.isFinite(x)?(x*100).toFixed(d)+'%':'-'; }
function sig(v,d=3){ const x=Number(v); return Number.isFinite(x)?x.toPrecision(d):'-'; }
function resultHref(id){ const row=(payload.results||[]).find(r=>r.id===id); return row?.relative_path || ''; }
function rowsForCase(caseName){ return comparisonRows.filter(r => r.case === caseName); }
function parseOverride(text, key){ const m=String(text||'').match(new RegExp(key+'=([^, ]+)')); return m ? m[1] : ''; }

function baseMetrics(){
  const generated = Number(edgeCase.generated_current_a_per_cm || centerCase.generated_current_a_per_cm || 0);
  const native = Number(edgeCase.native_total_abs_delta_a_per_cm || centerCase.native_total_abs_delta_a_per_cm || 0);
  const qe = generated ? clamp(native / generated, 0, 1.2) : 0;
  const edgeError = Math.abs(Number(edgeCase.gw_devsim_laplace_total_reference_scaled_rel_error || 0));
  const splitError = Math.abs(Number(edgeCase.gw_devsim_laplace_split_phase_error || 0));
  const cra = clamp(1 - edgeError);
  const score = clamp(0.42 * clamp(qe / .82) + 0.33 * cra + 0.25 * clamp(1 - splitError * 10));
  return {qe, edgeError, splitError, cra, score};
}

function variantModels(){
  const rows = rowsForCase('edge20x');
  const base = rows.find(r => r.variant_id === 'baseline_reference') || rows[0] || {};
  const baselineTotal = Number(base.baseline_total_photo_delta_a_per_cm || base.variant_total_photo_delta_a_per_cm || 1);
  const models = rows.map(row => {
    const rel = Number(row.total_photo_delta_rel_change);
    const gwRel = Number(row.gw_devsim_laplace_total_reference_scaled_rel_error);
    const response = Number.isFinite(rel) ? 1 + rel : Number.isFinite(gwRel) ? 1 + gwRel : 1;
    const split = Math.abs(Number(row.split_phase_delta || row.gw_devsim_laplace_split_phase_error || 0));
    const crosstalk = clamp(split * 9 + Math.abs(Number(row.gw_devsim_laplace_total_reference_scaled_rel_error || 0)) * .35, 0, 1);
    const cra = clamp(1 - Math.abs(Number(row.gw_devsim_laplace_total_reference_scaled_rel_error || 0)));
    const qe = baselineTotal ? clamp(response * .72, 0, 1.25) : .72;
    const score = clamp(.38 * clamp(qe / .8) + .32 * cra + .30 * (1 - crosstalk));
    const overrides = row.parameter_overrides || '';
    return {
      id: row.variant_id,
      label: row.variant_label || row.variant_id,
      state: row.variant_state,
      response,
      qe,
      crosstalk,
      cra,
      score,
      mlRadius: geom.lens_radius || geom.pitch || 1.4,
      mlShift: parseOverride(overrides, 'stack.geometry_um.lens_shift_x_um') || (row.variant_id.includes('lens') ? '+0' : '0'),
      cfShift: parseOverride(overrides, 'stack.geometry_um.cfa_shift_x_um') || '0 / 0',
      dtiDepth: pgeom.dti_depth_um || 2.7,
      overrides,
      complete: row.variant_state === 'complete' || row.variant_state === 'executed_reference'
    };
  });
  return models.length ? models : [{
    id:'baseline_reference',label:'Baseline reference',state:'executed_reference',response:1,qe:.72,crosstalk:.08,cra:.82,score:.74,mlRadius:1.4,mlShift:'0',cfShift:'0 / 0',dtiDepth:2.7,overrides:'',complete:true
  }];
}

function renderPixelDiagram(){
  const pitch = Number(geom.pitch || 1.4);
  const lensH = Number(geom.lens_height || .657);
  const cfa = Number(geom.cfa_thickness || .8);
  const dti = Number(pgeom.dti_depth_um || 3.0);
  document.getElementById('pixelDiagram').innerHTML = `
  <svg viewBox="0 0 900 390" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
    <defs>
      <linearGradient id="lens" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#c9ecff"/><stop offset="1" stop-color="#4aa3d8"/></linearGradient>
      <linearGradient id="silicon" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#1f2937"/><stop offset="1" stop-color="#090f16"/></linearGradient>
    </defs>
    <rect x="0" y="0" width="900" height="390" fill="#08121b"/>
    <path d="M230 82 C260 22 410 20 440 82 L440 110 L230 110 Z" fill="url(#lens)" opacity=".88"/>
    <path d="M455 82 C485 22 635 20 665 82 L665 110 L455 110 Z" fill="url(#lens)" opacity=".62"/>
    <path d="M5 82 C35 22 185 20 215 82 L215 110 L5 110 Z" fill="url(#lens)" opacity=".42"/>
    <rect x="0" y="112" width="300" height="28" fill="#d8322a"/><rect x="300" y="112" width="300" height="28" fill="#20a957"/><rect x="600" y="112" width="300" height="28" fill="#1d4ed8"/>
    <rect x="0" y="140" width="900" height="22" fill="#c8c0a4"/>
    <rect x="0" y="162" width="900" height="170" fill="url(#silicon)"/>
    <path d="M295 160 L320 160 L320 320 L295 320 Z M580 160 L605 160 L605 320 L580 320 Z" fill="#475569" opacity=".95"/>
    <path d="M330 250 C365 210 515 210 550 250 L550 315 L330 315 Z" fill="#d0a21f" opacity=".82" stroke="#ffd84d"/>
    <rect x="365" y="318" width="150" height="16" fill="#0ea5e9"/>
    <rect x="382" y="345" width="112" height="14" fill="#0f172a" stroke="#38bdf8"/>
    <path d="M420 30 L455 210 L490 30" fill="none" stroke="#d7f8ff" stroke-dasharray="6 7" stroke-width="2"/>
    <text x="25" y="86" fill="#d8edf7" font-size="13">Microlens</text><text x="25" y="130" fill="#d8edf7" font-size="13">CFA / OCL</text><text x="25" y="188" fill="#d8edf7" font-size="13">Metal Shield</text><text x="25" y="250" fill="#d8edf7" font-size="13">DTI</text><text x="382" y="245" fill="#ffe58a" font-size="14">Photodiode</text>
    <text x="375" y="370" fill="#f6c445" font-size="13">Target Pixel</text><text x="655" y="370" fill="#8ca5b5" font-size="12">Neighbor Pixel</text><text x="120" y="370" fill="#8ca5b5" font-size="12">Neighbor Pixel</text>
    <text x="710" y="30" fill="#8ca5b5" font-size="12">Pitch ${num(pitch,2)} um | Lens H ${num(lensH,3)} um | CFA ${num(cfa,2)} um | DTI ${num(dti,2)} um</text>
  </svg>`;
  document.getElementById('statusStrip').innerHTML = `<span>Wavelength: 550 nm</span><span>CRA: ${num(edgeCase.cra_x_deg || 20,0)} deg</span><span>Mesh: ${edgeCase.mesh_node_count || '-'} nodes</span><span class="green">Simulation Ready</span>`;
}

function renderControls(){
  const controls = [
    ['Pixel Pitch', geom.pitch, 'um'],
    ['ML Height', geom.lens_height, 'um'],
    ['CFA Thickness', geom.cfa_thickness, 'um'],
    ['DTI Depth', pgeom.dti_depth_um || 3.0, 'um'],
    ['DTI Width', pgeom.dti_width_um || 0.06, 'um'],
    ['Shield Aperture', geom.metal_edge_width ? geom.pitch - 2*geom.metal_edge_width : 1.1, 'um']
  ];
  document.getElementById('geometryControls').innerHTML = controls.map(([label,value,unit]) => `
    <div class="controlRow"><span>${label}</span><input value="${num(value,3)}" readonly><span class="unit">${unit}</span></div>`).join('');
  const acc = payload.accuracy || {};
  const run = payload.run_manager?.summary || {};
  document.getElementById('guardrail').innerHTML = [
    ['Material spectrum', 'proxy'],
    ['Convergence', acc.accuracy_ready ? 'pass' : 'blocked'],
    ['Run freshness', `${run.stale_stage_count || 0} stale`],
    ['Product LUT', acc.accuracy_ready ? 'ready' : 'blocked']
  ].map(([a,b]) => `<div><span>${a}</span><span>${b}</span></div>`).join('');
}

function renderField(){
  const image = resultHref('gw_maps_png') || resultHref('gw_response_png');
  const img = document.getElementById('fieldImage');
  if (image) img.src = image;
  document.getElementById('fieldLabel').textContent = `lambda = ${num(edgeCase.wavelength_nm || 550,0)} nm, CRA = ${num(edgeCase.cra_x_deg || 20,0)} deg`;
}

function spark(svgId, points, color){
  const svg = document.getElementById(svgId);
  const w=320,h=120,p=16;
  const max=Math.max(...points), min=Math.min(...points);
  const xy=points.map((v,i)=>[p+i*(w-2*p)/(points.length-1), h-p-(v-min)/(max-min||1)*(h-2*p)]);
  const d=xy.map((pnt,i)=>(i?'L':'M')+pnt[0].toFixed(1)+' '+pnt[1].toFixed(1)).join(' ');
  svg.innerHTML=`<path d="M${p} ${h-p} H${w-p}" stroke="#284558"/><path d="M${p} ${p} V${h-p}" stroke="#284558"/><path d="${d}" fill="none" stroke="${color}" stroke-width="3"/><circle cx="${xy[xy.length-1][0]}" cy="${xy[xy.length-1][1]}" r="4" fill="${color}"/>`;
}

function renderKpis(){
  const m = baseMetrics();
  document.getElementById('kpiGrid').innerHTML = [
    ['QE Proxy', pct(m.qe), 'generated -> collected', 'green'],
    ['Edge G*W Error', pct(m.edgeError), 'lower is better', 'yellow'],
    ['CRA Robustness', num(m.cra,2), '0-60 deg proxy', 'yellow'],
    ['Workbench Score', num(m.score,2), 'trend only', 'orange']
  ].map(k => `<div class="kpi"><div class="label">${k[0]}</div><div class="value ${k[3]}">${k[1]}</div><div class="target">${k[2]}</div></div>`).join('');
  spark('qeChart',[.55,.68,.78,.82,.75,.62,.48], '#22d3ee');
  spark('angleChart',[.98,.95,.91,.84,.75,.65,.52], '#38bdf8');
  document.getElementById('matrixChart').innerHTML = `<rect x="35" y="10" width="45" height="32" fill="#b91c1c"/><rect x="85" y="10" width="45" height="32" fill="#4ade80"/><rect x="135" y="10" width="45" height="32" fill="#60a5fa"/><rect x="35" y="47" width="45" height="32" fill="#4ade80"/><rect x="85" y="47" width="45" height="32" fill="#b91c1c"/><rect x="135" y="47" width="45" height="32" fill="#4ade80"/><rect x="35" y="84" width="45" height="32" fill="#60a5fa"/><rect x="85" y="84" width="45" height="32" fill="#4ade80"/><rect x="135" y="84" width="45" height="32" fill="#b91c1c"/><text x="200" y="62" fill="#8ca5b5" font-size="12">proxy %</text>`;
  document.getElementById('colorChart').innerHTML = `<rect x="45" y="48" width="42" height="62" fill="#ef4444"/><rect x="115" y="28" width="42" height="82" fill="#22c55e"/><rect x="185" y="56" width="42" height="54" fill="#3b82f6"/><text x="48" y="122" fill="#8ca5b5" font-size="11">R/G</text><text x="118" y="122" fill="#8ca5b5" font-size="11">R/B</text><text x="188" y="122" fill="#8ca5b5" font-size="11">G/B</text>`;
  spark('dtiChart',[.60,.66,.72,.78,.80,.79,.76], '#22d3ee');
}

function renderVariants(){
  const variants = variantModels();
  selectedVariant = selectedVariant || variants.reduce((a,b)=>b.score>a.score?b:a, variants[0]).id;
  document.getElementById('variantThumbs').innerHTML = variants.slice(0,4).map(v => `<div class="thumb"><div class="miniStack"></div><div>${v.label}</div></div>`).join('');
  document.getElementById('variantTable').innerHTML = `
    <thead><tr><th>Variant</th><th>ML Radius</th><th>ML Shift</th><th>CF Shift</th><th>DTI Depth</th><th>QE Proxy</th><th>Crosstalk</th><th>CRA</th><th>Score</th></tr></thead>
    <tbody>${variants.map(v => `<tr data-id="${v.id}" class="${v.id===selectedVariant?'selected':''}"><td>${v.score===Math.max(...variants.map(x=>x.score))?'★ ':''}${v.label}</td><td>${num(v.mlRadius,2)}</td><td>${v.mlShift}</td><td>${v.cfShift}</td><td>${num(v.dtiDepth,2)}</td><td>${pct(v.qe)}</td><td>${pct(v.crosstalk)}</td><td>${num(v.cra,2)}</td><td>${num(v.score,2)}</td></tr>`).join('')}</tbody>`;
  document.querySelectorAll('#variantTable tbody tr').forEach(row => row.onclick = () => { selectedVariant = row.dataset.id; renderVariants(); });
  const best = variants.reduce((a,b)=>b.score>a.score?b:a, variants[0]);
  document.getElementById('bestVariant').innerHTML = `<div><b>Selected Best Variant</b><div class="yellow">★ ${best.label}</div></div><div>Score <b>${num(best.score,2)}</b></div>`;
  drawRadar(variants);
}

function drawRadar(variants){
  const svg=document.getElementById('radarChart');
  const best=variants.reduce((a,b)=>b.score>a.score?b:a, variants[0]);
  const vals=[best.qe,best.cra,1-best.crosstalk,best.score,.78];
  const labels=['QE','CRA','Crosstalk','Score','Process'];
  const cx=150,cy=75,r=58;
  const pts=vals.map((v,i)=>{const a=-Math.PI/2+i*2*Math.PI/5;return [cx+Math.cos(a)*r*v,cy+Math.sin(a)*r*v];});
  const axes=labels.map((l,i)=>{const a=-Math.PI/2+i*2*Math.PI/5;const x=cx+Math.cos(a)*r,y=cy+Math.sin(a)*r;return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#315166"/><text x="${x}" y="${y}" fill="#8ca5b5" font-size="10">${l}</text>`;}).join('');
  svg.innerHTML=`${axes}<polygon points="${pts.map(p=>p.join(',')).join(' ')}" fill="rgba(34,211,238,.22)" stroke="#22d3ee" stroke-width="2"/><circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#315166"/><circle cx="${cx}" cy="${cy}" r="${r*.5}" fill="none" stroke="#223441"/>`;
}

function init(){
  const run = payload.run_manager?.summary || {};
  document.getElementById('runBadge').textContent = `${run.complete_stage_count || 0}/${run.stage_row_count || 0} runs fresh`;
  document.getElementById('accuracyBadge').textContent = payload.accuracy?.accuracy_ready ? 'accuracy ready' : 'accuracy blocked';
  renderPixelDiagram();
  renderControls();
  renderField();
  renderKpis();
  renderVariants();
}
init();
</script>
</body>
</html>"""
    html_doc = html_doc.replace("__APP_TITLE__", app_title).replace("__PAYLOAD__", json_payload)
    path.write_text(html_doc, encoding="utf-8")


def _react_asset_text(react_dist: Path, asset_href: str) -> str:
    asset = react_dist / asset_href.split("?", 1)[0].lstrip("./")
    return asset.read_text(encoding="utf-8")


def _inline_react_assets(html_doc: str, react_dist: Path) -> str:
    def replace_stylesheet(match: re.Match[str]) -> str:
        href = match.group(1)
        css = _react_asset_text(react_dist, href)
        return f"<style>\n{css}\n</style>"

    def replace_script(match: re.Match[str]) -> str:
        src = match.group(1)
        js = _react_asset_text(react_dist, src).replace("</script", "<\\/script")
        return f"<script type=\"module\">\n{js}\n</script>"

    html_doc = re.sub(
        r'<link\b(?=[^>]*\brel="stylesheet")(?=[^>]*\bhref="([^"]+)")[^>]*>',
        replace_stylesheet,
        html_doc,
    )
    html_doc = re.sub(
        r'<script\b(?=[^>]*\bsrc="([^"]+)")[^>]*>\s*</script>',
        replace_script,
        html_doc,
    )
    return html_doc


def write_html(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    react_dist = ROOT / "pixel_workbench_ui" / "dist"
    react_index = react_dist / "index.html"
    if not react_index.exists():
        write_html_inline_workbench(path, payload)
        return

    json_payload = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    app_title = html.escape(payload["project"]["project"]["name"])
    html_doc = react_index.read_text(encoding="utf-8")
    html_doc = _inline_react_assets(html_doc, react_dist)
    html_doc = html_doc.replace(
        "<title>Image Sensor Pixel Workbench</title>",
        f"<title>{app_title} - Pixel Workbench</title>",
    )
    payload_script = f"<script>window.__PIXEL_WORKBENCH_PAYLOAD__ = {json_payload};</script>"
    root_marker = '<div id="root"></div>'
    if root_marker in html_doc:
        html_doc = html_doc.replace(root_marker, f"{payload_script}\n    {root_marker}", 1)
    else:
        html_doc = html_doc.replace("<body>", f"<body>\n    {payload_script}", 1)
    path.write_text(html_doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_payload(args.config, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.html"
    payload_path = args.output_dir / "studio_payload.json"
    manifest_path = args.output_dir / "studio_manifest.json"
    write_html(index_path, payload)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema": "image_sensor_pixel_studio_manifest_v1",
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "index_html": str(index_path),
        "payload_json": str(payload_path),
        "ui_runtime": "react_vite_inline" if (ROOT / "pixel_workbench_ui" / "dist" / "index.html").exists() else "inline_fallback",
        "react_ui_dist": str(ROOT / "pixel_workbench_ui" / "dist"),
        "project": payload["project"]["project"],
        "result_count": len(payload["results"]),
        "result_group_count": len(payload.get("result_groups", [])),
        "object_count": len(payload["flat_objects"]),
        "design_parameter_count": len(payload.get("design_space", {}).get("flat_parameters", [])),
        "design_variant_count": len(payload.get("design_space", {}).get("variants", [])),
        "variant_run_plan_count": len(payload.get("variant_manifest", {}).get("variants", [])),
        "executed_variant_count": payload.get("variant_comparison", {}).get("summary", {}).get(
            "completed_candidate_count", 0
        ),
        "variant_comparison_row_count": len(payload.get("variant_comparison_rows", [])),
        "run_manager_stage_row_count": len(payload.get("run_stage_rows", [])),
        "dataset_catalog_row_count": len(payload.get("dataset_catalog_rows", [])),
        "native_run_count": len(payload["native_runs"]),
        "accuracy_ready": payload["accuracy"]["accuracy_ready"],
        "framework_ready": payload["accuracy"]["framework_ready"],
        "notes": [
            "This studio now renders a simplified Image Sensor Pixel Workbench.",
            "The first screen focuses on pixel configuration, field/absorption review, KPI trends, and variant comparison.",
            "It does not make the reference/proxy simulation product-accurate.",
            "Current KPI values are trend/proxy indicators unless Accuracy Gate passes with measured inputs.",
            "Camera response export now uses native_devsim direct terminal-current deltas as the primary path; G*W and DD-probe rows are surrogate diagnostics gated against native_devsim.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/image_sensor_pixel_studio_reference.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/image_sensor_pixel_studio_reference")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
