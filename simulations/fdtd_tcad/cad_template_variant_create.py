#!/usr/bin/env python3
"""Create a CAD template variant from a base template plus parameter overrides.

This keeps design changes traceable: the generated STEP/BREP/mesh/footprint
files are accompanied by the exact base template and overrides that produced
them. It is intentionally limited to scalar TemplateSpec fields; topology
changes such as changing OCL block connectivity should be added as a new base
template, not hidden inside ad-hoc overrides.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from pixel_cad_template_library import (
    ROOT,
    OclBlock,
    TemplateSpec,
    template_specs,
    validate_library,
    write_template,
)


DEFAULT_LIBRARY = ROOT / "runs" / "pixel_cad_template_library_reference"
MANIFEST_NAME = "template_library_manifest.json"
VALIDATION_NAME = "cad_template_validation_report.json"
TOPOLOGY_OVERRIDE_FIELDS = {"nx", "nz", "cfa_pattern", "split_mode", "shield_mode"}
SCALAR_OVERRIDE_FIELDS = {
    field.name
    for field in fields(TemplateSpec)
    if field.name not in {"template_id", "label", "ocl_blocks", "notes", *TOPOLOGY_OVERRIDE_FIELDS}
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_").lower()
    if not clean:
        raise ValueError("variant id cannot be empty")
    return clean[:80]


def spec_from_parameters(path: Path) -> TemplateSpec:
    data = load_json(path)
    blocks = []
    for block in data.get("ocl_blocks", []):
        if not isinstance(block, dict):
            raise ValueError(f"Invalid OCL block in {path}: {block!r}")
        block_args = {key: block[key] for key in ("lens_id", "ix", "iz", "sx", "sz")}
        if block.get("height_um") is not None:
            block_args["height_um"] = block["height_um"]
        blocks.append(OclBlock(**block_args))
    data["ocl_blocks"] = tuple(blocks)
    if isinstance(data.get("notes"), list):
        data["notes"] = tuple(str(item) for item in data["notes"])
    return TemplateSpec(**data)


def load_base_spec(base_template: str, library_root: Path) -> TemplateSpec:
    parameter_path = library_root / base_template / "template_parameters.json"
    if parameter_path.exists():
        return spec_from_parameters(parameter_path)
    builtins = template_specs()
    if base_template in builtins:
        return builtins[base_template]
    raise KeyError(f"Unknown base CAD template: {base_template}")


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
        raise ValueError(f"Cannot coerce {text!r} to bool")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(parsed)
    if isinstance(current_value, float):
        return float(parsed)
    return str(parsed)


def parse_overrides(assignments: list[str], base: TemplateSpec) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Override must be key=value: {assignment}")
        key, raw_value = assignment.split("=", 1)
        key = key.strip()
        if key not in SCALAR_OVERRIDE_FIELDS:
            raise ValueError(
                f"Unsupported CAD template override {key!r}. "
                "Use scalar TemplateSpec fields only; create a new base template for topology changes."
            )
        overrides[key] = parse_value(raw_value.strip(), getattr(base, key))
    return overrides


def check_physical_ranges(spec: TemplateSpec) -> list[str]:
    warnings: list[str] = []
    if spec.pitch_um <= 0:
        warnings.append("pitch_um must be positive")
    if spec.si_thickness_um <= 0:
        warnings.append("si_thickness_um must be positive")
    if spec.passivation_thickness_um < 0:
        warnings.append("passivation_thickness_um cannot be negative")
    if spec.cfa_thickness_um <= 0:
        warnings.append("cfa_thickness_um should be positive")
    if spec.lens_height_um <= 0:
        warnings.append("lens_height_um should be positive")
    if spec.dti_width_um < 0 or spec.dti_width_um > spec.pitch_um * 0.35:
        warnings.append("dti_width_um is outside the practical guard range")
    if spec.pd_depth_min_um < 0 or spec.pd_depth_max_um <= spec.pd_depth_min_um:
        warnings.append("PD depth range is invalid")
    if spec.pd_depth_max_um > spec.si_thickness_um:
        warnings.append("PD depth exceeds silicon thickness")
    if spec.lens_edge_gap_um < 0 or spec.lens_edge_gap_um >= spec.pitch_um:
        warnings.append("lens_edge_gap_um is invalid for at least a 1x1 OCL")
    if spec.cfa_gap_um < 0 or spec.cfa_gap_um >= spec.pitch_um:
        warnings.append("cfa_gap_um is invalid")
    return warnings


def update_ledger(template_dir: Path, base_template: str, overrides: dict[str, Any], warnings: list[str]) -> None:
    ledger_path = template_dir / "assumption_ledger.json"
    ledger = load_json(ledger_path)
    ledger["variant_of"] = base_template
    ledger["parameter_overrides"] = overrides
    ledger["variant_validation_warnings"] = warnings
    ledger.setdefault("review_checklist", []).insert(
        0,
        "Compare variant_source.json against the base template before accepting this variant.",
    )
    write_json(ledger_path, ledger)


def append_or_replace_record(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    records = [item for item in manifest.get("templates", []) if isinstance(item, dict)]
    existing_index = next((index for index, item in enumerate(records) if item.get("template_id") == record["template_id"]), None)
    if existing_index is None:
        records.append(record)
    else:
        records[existing_index] = record
    manifest["templates"] = records
    manifest["template_count"] = len(records)
    manifest["accuracy_status"] = "parametric_templates_not_measured"
    return manifest


def read_or_make_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        return load_json(manifest_path)
    return {
        "schema": "pixel_cad_template_library_manifest_v1",
        "output_dir": str(output_dir),
        "template_count": 0,
        "generated_with": "Gmsh/OpenCASCADE",
        "freecad_role": "Open generated STEP/BREP files for 3D review; FreeCAD is not required for headless generation.",
        "mask_role": "Use geometry_import.json or downstream GDS export for solver footprints.",
        "mesh_role": "Optional model.msh files are coarse 3D CAD review meshes with physical volume groups; they are not calibrated DEVSIM electrical meshes.",
        "accuracy_status": "parametric_templates_not_measured",
        "templates": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = load_base_spec(args.base_template, args.library_root)
    variant_id = sanitize_id(args.id or f"{base.template_id}_variant")
    if variant_id == base.template_id:
        raise ValueError("variant id must differ from the base template id")
    overrides = parse_overrides(args.set, base)
    if not overrides:
        raise ValueError("At least one --set key=value override is required")
    notes = tuple(base.notes) + (
        f"Variant of {base.template_id}.",
        "Generated by cad_template_variant_create.py from explicit parameter overrides.",
    )
    variant = replace(
        base,
        template_id=variant_id,
        label=args.label or f"{base.label} variant",
        notes=notes,
        **overrides,
    )
    warnings = check_physical_ranges(variant)
    if warnings and not args.allow_warnings:
        raise ValueError("Variant validation warnings require --allow-warnings: " + "; ".join(warnings))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = write_template(variant, args.output_dir, mesh=not args.no_mesh)
    record["variant_of"] = base.template_id
    record["parameter_overrides"] = overrides
    record["validation_warnings"] = warnings
    template_dir = args.output_dir / variant.template_id
    source = {
        "schema": "cad_template_variant_source_v1",
        "template_id": variant.template_id,
        "label": variant.label,
        "base_template_id": base.template_id,
        "base_template_label": base.label,
        "parameter_overrides": overrides,
        "validation_warnings": warnings,
        "topology_changes_allowed": False,
        "product_accuracy_ready": False,
    }
    write_json(template_dir / "variant_source.json", source)
    update_ledger(template_dir, base.template_id, overrides, warnings)
    record["files"]["variant_source"] = str(template_dir / "variant_source.json")

    manifest = read_or_make_manifest(args.output_dir)
    append_or_replace_record(manifest, record)
    write_json(args.output_dir / MANIFEST_NAME, manifest)
    validation = validate_library(manifest["templates"], args.output_dir, mesh_expected=not args.no_mesh)
    write_json(args.output_dir / VALIDATION_NAME, validation)
    result = {
        "schema": "cad_template_variant_create_result_v1",
        "status": validation["status"],
        "template_id": variant.template_id,
        "base_template_id": base.template_id,
        "output_dir": str(template_dir),
        "variant_source": str(template_dir / "variant_source.json"),
        "assumption_ledger": str(template_dir / "assumption_ledger.json"),
        "step": record["files"]["step"],
        "brep": record["files"]["brep"],
        "mesh": record["files"]["mesh"],
        "geometry_import": record["files"]["geometry_import"],
        "manifest": str(args.output_dir / MANIFEST_NAME),
        "validation_report": str(args.output_dir / VALIDATION_NAME),
        "validation_warnings": warnings,
        "product_accuracy_ready": False,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-template", required=True, help="Existing template id to use as the variant base.")
    parser.add_argument("--id", required=True, help="New variant template id.")
    parser.add_argument("--label", default="", help="Human-readable variant label.")
    parser.add_argument("--set", action="append", default=[], help="Scalar TemplateSpec override, e.g. lens_height_um=0.71")
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY, help="Where to read the base template parameters.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LIBRARY, help="Where to write and register the variant.")
    parser.add_argument("--no-mesh", action="store_true", help="Skip coarse 3D model.msh generation.")
    parser.add_argument("--allow-warnings", action="store_true", help="Write the variant even if guard-range warnings are present.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
