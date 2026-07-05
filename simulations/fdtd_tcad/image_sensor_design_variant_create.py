#!/usr/bin/env python3
"""Create an ad-hoc image-sensor design variant from parameter overrides.

This is a design-edit entry point around image_sensor_variant_builder.py. It
validates parameter ids/paths against the design-space registry, materializes a
variant run plan when requested, and can refresh the Studio management artifacts.
It does not run Meep, Gmsh, or DEVSIM solver stages.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_sensor_variant_builder import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECT_CONFIG,
    abs_path,
    build_variant,
    get_path,
    read_json,
    rel_from_root,
    split_override_path,
    stage_list,
    write_json,
)


ROOT = Path(__file__).resolve().parent


def flatten_parameters(design_space: dict[str, Any]) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for group in design_space.get("parameter_groups", []):
        for parameter in group.get("parameters", []):
            item = dict(parameter)
            item["group_id"] = group.get("id", "")
            item["group_label"] = group.get("label", "")
            item["owner"] = group.get("owner", "")
            params.append(item)
    return params


def parse_value(text: str, current_value: Any) -> Any:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    if isinstance(current_value, bool):
        if isinstance(parsed, bool):
            return parsed
        if str(parsed).lower() in {"true", "1", "yes"}:
            return True
        if str(parsed).lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot coerce {text!r} to bool")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(parsed)
    if isinstance(current_value, float):
        return float(parsed)
    return parsed


def sanitize_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_").lower()
    return clean or datetime.now(timezone.utc).strftime("custom_%Y%m%dT%H%M%SZ")


def load_inputs(project_config: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    project = read_json(project_config)
    config_dir = project_config.parent
    stack = read_json(abs_path(config_dir, project["inputs"]["stack_config"]))
    profile = read_json(abs_path(config_dir, project["inputs"]["tcad_profile"]))
    design_space = read_json(abs_path(config_dir, project["inputs"]["design_space"]))
    return project, stack, profile, design_space


def resolve_override(
    assignment: str,
    params_by_id: dict[str, dict[str, Any]],
    params_by_path: dict[str, dict[str, Any]],
    roots: dict[str, Any],
    allow_out_of_range: bool,
) -> tuple[dict[str, Any], str, Any, list[str]]:
    if "=" not in assignment:
        raise ValueError(f"parameter override must be key=value: {assignment}")
    key, value_text = assignment.split("=", 1)
    key = key.strip()
    parameter = params_by_id.get(key) or params_by_path.get(key)
    if parameter:
        path = str(parameter["path"])
    else:
        path = key
        parameter = {
            "id": key,
            "label": key,
            "path": path,
            "requires_rerun": [],
            "wired_to_solver": None,
        }
    root_name, child_path = split_override_path(path)
    current_value = get_path(roots[root_name], child_path)
    new_value = parse_value(value_text.strip(), current_value)

    warnings: list[str] = []
    value_range = parameter.get("range")
    if isinstance(new_value, (int, float)) and isinstance(value_range, list) and len(value_range) == 2:
        lower, upper = float(value_range[0]), float(value_range[1])
        if not (lower <= float(new_value) <= upper):
            message = f"{parameter.get('id', path)}={new_value} is outside recommended range [{lower}, {upper}]"
            if allow_out_of_range:
                warnings.append(message)
            else:
                raise ValueError(message)
    if parameter.get("wired_to_solver") is False:
        warnings.append(f"{parameter.get('id', path)} is metadata/proxy only in the current solver path")
    return parameter, path, new_value, warnings


def build_variant_spec(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    project, stack, profile, design_space = load_inputs(args.config)
    roots = {"project": project, "stack": stack, "profile": profile}
    parameters = flatten_parameters(design_space)
    params_by_id = {str(item.get("id")): item for item in parameters}
    params_by_path = {str(item.get("path")): item for item in parameters}

    overrides: dict[str, Any] = {}
    resolved: list[dict[str, Any]] = []
    warnings: list[str] = []
    required_stages: set[str] = set()
    for assignment in args.param:
        parameter, path, value, item_warnings = resolve_override(
            assignment,
            params_by_id,
            params_by_path,
            roots,
            args.allow_out_of_range,
        )
        root_name, child_path = split_override_path(path)
        current_value = get_path(roots[root_name], child_path)
        overrides[path] = value
        warnings.extend(item_warnings)
        required_stages.update(parameter.get("requires_rerun", []))
        resolved.append(
            {
                "parameter_id": parameter.get("id", path),
                "label": parameter.get("label", path),
                "path": path,
                "old_value": current_value,
                "new_value": value,
                "wired_to_solver": parameter.get("wired_to_solver"),
                "requires_rerun": parameter.get("requires_rerun", []),
                "unit": parameter.get("unit", ""),
            }
        )

    if not overrides:
        raise ValueError("at least one --param key=value override is required")
    variant_id = sanitize_id(args.id or f"custom_{resolved[0]['parameter_id']}")
    variant = {
        "id": variant_id,
        "label": args.label or variant_id.replace("_", " "),
        "status": "candidate_not_simulated",
        "goal": args.goal or "Ad-hoc design edit generated from Studio design-space parameters.",
        "parameter_overrides": overrides,
        "expected_effect": args.expected_effect or "Generated candidate; inspect comparison results after running required stages.",
        "risks": list(args.risk or []) + warnings + ["Generated candidate is not product-accurate."],
        "requires_rerun": stage_list(sorted(required_stages)),
    }
    plan = {
        "schema": "image_sensor_design_variant_create_plan_v1",
        "variant": variant,
        "resolved_parameters": resolved,
        "warnings": warnings,
        "materialize_requested": bool(args.materialize),
        "refresh_requested": bool(args.refresh),
        "product_lut_ready": False,
        "accuracy_ready": False,
    }
    return variant, plan


def root_manifest_template(project_config: Path, output_dir: Path, project: dict[str, Any]) -> dict[str, Any]:
    config_dir = project_config.parent
    return {
        "schema": "image_sensor_variant_run_manifest_v1",
        "source_project_config": str(project_config),
        "source_stack_config": str(abs_path(config_dir, project["inputs"]["stack_config"])),
        "source_tcad_profile": str(abs_path(config_dir, project["inputs"]["tcad_profile"])),
        "source_design_space": str(abs_path(config_dir, project["inputs"]["design_space"])),
        "output_dir": str(output_dir),
        "variant_count": 0,
        "variants": [],
        "summary": {"candidate_count": 0, "product_lut_ready": False, "all_override_errors": []},
    }


def update_root_manifest(
    manifest_path: Path,
    project_config: Path,
    output_dir: Path,
    materialized_variant: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    project = read_json(project_config)
    if manifest_path.exists():
        root_manifest = read_json(manifest_path)
    else:
        root_manifest = root_manifest_template(project_config, output_dir, project)
    variants = list(root_manifest.get("variants", []))
    existing_index = next(
        (index for index, item in enumerate(variants) if item.get("id") == materialized_variant.get("id")),
        None,
    )
    if existing_index is not None:
        if not overwrite:
            raise ValueError(f"variant id already exists: {materialized_variant['id']} (pass --overwrite)")
        variants[existing_index] = materialized_variant
    else:
        variants.append(materialized_variant)
    root_manifest["variants"] = variants
    root_manifest["variant_count"] = len(variants)
    summary = root_manifest.setdefault("summary", {})
    summary["candidate_count"] = sum(1 for item in variants if item.get("status") != "simulated_reference")
    summary["all_override_errors"] = [
        {"variant": item.get("id"), "errors": item.get("errors", [])}
        for item in variants
        if item.get("errors")
    ]
    summary["product_lut_ready"] = False
    write_json(manifest_path, root_manifest)
    return root_manifest


def refresh_management(project_config: Path, output_dir: Path, studio_output_dir: Path) -> list[dict[str, Any]]:
    commands = [
        [sys.executable, "image_sensor_variant_compare.py", "--config", str(project_config), "--variant-manifest", str(output_dir / "variant_run_manifest.json"), "--output-dir", str(output_dir)],
        [sys.executable, "image_sensor_run_manager.py", "--config", str(project_config), "--variant-manifest", str(output_dir / "variant_run_manifest.json"), "--output-dir", str(output_dir), "--studio-output-dir", str(studio_output_dir)],
        [sys.executable, "image_sensor_pixel_studio.py", "--config", str(project_config), "--output-dir", str(studio_output_dir)],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": " ".join(command),
                "return_code": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        if completed.returncode != 0:
            break
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.studio_output_dir = args.studio_output_dir.resolve()
    variant, plan = build_variant_spec(args)
    result = dict(plan)

    if args.plan_json:
        write_json(args.plan_json.resolve(), plan)
    if args.materialize:
        project, stack, profile, _design_space = load_inputs(args.config)
        materialized = build_variant(variant, project, stack, profile, args.output_dir)
        manifest_path = args.output_dir / "variant_run_manifest.json"
        root_manifest = update_root_manifest(manifest_path, args.config, args.output_dir, materialized, args.overwrite)
        result["materialized_variant"] = materialized
        result["variant_manifest"] = str(manifest_path)
        result["variant_count"] = root_manifest.get("variant_count")
        if args.refresh:
            result["refresh_results"] = refresh_management(args.config, args.output_dir, args.studio_output_dir)
            result["refresh_failed"] = any(item["return_code"] != 0 for item in result["refresh_results"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("refresh_failed"):
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--studio-output-dir", type=Path, default=ROOT / "runs" / "image_sensor_pixel_studio_reference")
    parser.add_argument("--id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--goal", default="")
    parser.add_argument("--expected-effect", default="")
    parser.add_argument("--risk", action="append", default=[])
    parser.add_argument("--param", action="append", default=[], help="Parameter id/path assignment, e.g. split_gap_um=0.05")
    parser.add_argument("--allow-out-of-range", action="store_true")
    parser.add_argument("--materialize", action="store_true", help="Write variant files and update the root variant manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing materialized variant id.")
    parser.add_argument("--refresh", action="store_true", help="Refresh comparison/run-manager/Studio after materializing.")
    parser.add_argument("--plan-json", type=Path, default=None)
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    "schema": "image_sensor_design_variant_create_error_v1",
                    "error": str(exc),
                    "product_lut_ready": False,
                    "accuracy_ready": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
