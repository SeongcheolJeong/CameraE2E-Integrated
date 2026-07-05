#!/usr/bin/env python3
"""Review helper for the pixel CAD template library.

This script does not generate geometry and does not change solver inputs. It
helps inspect the existing template library by listing templates, printing the
remaining assumption ledger, and opening STEP/BREP artifacts in FreeCAD when it
is installed.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = ROOT / "runs" / "pixel_cad_template_library_reference"
FREECAD_VALIDATION_NAME = "freecad_validation_report.json"
FREECAD_APP_CANDIDATES = [
    Path.home() / "Applications" / "FreeCAD.app",
    Path("/Applications/FreeCAD.app"),
]

FREECAD_SHAPE_PROBE = r"""
import json
import sys
import traceback
from pathlib import Path

result = {
    "schema": "freecad_shape_probe_v1",
    "status": "FAIL",
}
output = Path(sys.argv[-3])

try:
    import FreeCAD as App
    import Part

    target = Path(sys.argv[-4])
    fcstd_text = sys.argv[-2]
    metadata_text = sys.argv[-1]
    fcstd_path = Path(fcstd_text) if fcstd_text else None
    metadata_path = Path(metadata_text) if metadata_text else None
    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def write_sheet(sheet, rows):
        for row_index, row in enumerate(rows, start=1):
            for col_index, value in enumerate(row, start=1):
                col = chr(ord("A") + col_index - 1)
                text = "" if value is None else str(value)
                sheet.set(f"{col}{row_index}", text[:240])

    shape = Part.Shape()
    shape.read(str(target))
    bbox = shape.BoundBox
    result.update(
        {
            "status": "PASS" if shape.isValid() and len(shape.Solids) > 0 and shape.Volume > 0 else "FAIL",
            "path": str(target),
            "shape_type": shape.ShapeType,
            "is_valid": bool(shape.isValid()),
            "solid_count": len(shape.Solids),
            "compound_count": len(shape.Compounds),
            "face_count": len(shape.Faces),
            "edge_count": len(shape.Edges),
            "volume_um3": float(shape.Volume),
            "area_um2": float(shape.Area),
            "bbox_um": {
                "xmin": float(bbox.XMin),
                "xmax": float(bbox.XMax),
                "ymin": float(bbox.YMin),
                "ymax": float(bbox.YMax),
                "zmin": float(bbox.ZMin),
                "zmax": float(bbox.ZMax),
                "xlen": float(bbox.XLength),
                "ylen": float(bbox.YLength),
                "zlen": float(bbox.ZLength),
            },
        }
    )
    if fcstd_path:
        doc = App.newDocument("PixelTemplateValidation")
        obj = doc.addObject("Part::Feature", "ImportedTemplateShape")
        obj.Shape = shape
        obj.Label = target.stem
        parameter_rows = [["parameter", "value"]]
        for key, value in sorted((metadata.get("parameters") or {}).items()):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            parameter_rows.append([key, value])
        if len(parameter_rows) > 1:
            sheet = doc.addObject("Spreadsheet::Sheet", "TemplateParameters")
            write_sheet(sheet, parameter_rows)

        validation_rows = [
            ["field", "value"],
            ["template_id", metadata.get("template_id")],
            ["label", metadata.get("label")],
            ["variant_of", metadata.get("variant_of")],
            ["source_truth_level", metadata.get("source_truth_level")],
            ["freecad_probe_status", result.get("status")],
            ["design_rule_status", metadata.get("design_rule_status")],
            ["design_rule_fail_count", metadata.get("design_rule_fail_count")],
            ["expected_bbox_um", json.dumps(metadata.get("expected_bbox_um", {}), ensure_ascii=False)],
        ]
        validation = metadata.get("design_rule_validation") if isinstance(metadata.get("design_rule_validation"), dict) else {}
        for rule in validation.get("rules", []):
            if isinstance(rule, dict):
                validation_rows.append([f"rule:{rule.get('name')}", rule.get("status")])
        summary_sheet = doc.addObject("Spreadsheet::Sheet", "ValidationSummary")
        write_sheet(summary_sheet, validation_rows)
        doc.recompute()
        fcstd_path.parent.mkdir(parents=True, exist_ok=True)
        doc.saveAs(str(fcstd_path))
        object_names = [item.Name for item in doc.Objects]
        App.closeDocument(doc.Name)
        result["fcstd"] = {
            "path": str(fcstd_path),
            "exists": fcstd_path.exists(),
            "size_bytes": fcstd_path.stat().st_size if fcstd_path.exists() else None,
            "contains_parameter_sheet": "TemplateParameters" in object_names,
            "contains_validation_sheet": "ValidationSummary" in object_names,
            "object_names": object_names,
        }
except Exception as exc:  # noqa: BLE001 - report FreeCAD import failure.
    result.update({"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()[-4000:]})

output.write_text(json.dumps(result, indent=2), encoding="utf-8")
"""

FREECAD_PARAMETER_EXTRACT_PROBE = r"""
import json
import sys
import traceback
from pathlib import Path

output = Path(sys.argv[-2])
target = Path(sys.argv[-1])
result = {
    "schema": "freecad_template_parameter_extract_v1",
    "status": "FAIL",
    "path": str(target),
}

try:
    import FreeCAD as App

    def coerce(value):
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        text = str(value)
        try:
            return json.loads(text)
        except Exception:
            return text

    doc = App.openDocument(str(target))
    objects = [obj.Name for obj in doc.Objects]
    sheet = doc.getObject("TemplateParameters")
    if sheet is None:
        raise RuntimeError("TemplateParameters spreadsheet not found")
    rows = []
    parameters = {}
    blank_count = 0
    for row_index in range(1, 300):
        try:
            key = sheet.get(f"A{row_index}")
            value = sheet.get(f"B{row_index}")
        except Exception:
            break
        if not key and not value:
            blank_count += 1
            if blank_count >= 12:
                break
            continue
        blank_count = 0
        key_text = str(key).strip() if key else ""
        if not key_text or key_text == "parameter":
            continue
        parsed = coerce(value)
        parameters[key_text] = parsed
        rows.append({"row": row_index, "key": key_text, "value": parsed})
    result.update(
        {
            "status": "PASS",
            "object_names": objects,
            "has_template_parameters": "TemplateParameters" in objects,
            "has_validation_summary": "ValidationSummary" in objects,
            "parameter_count": len(parameters),
            "parameters": parameters,
            "rows": rows,
        }
    )
    App.closeDocument(doc.Name)
except Exception as exc:  # noqa: BLE001 - report FreeCAD import failure.
    result.update({"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()[-4000:]})

output.write_text(json.dumps(result, indent=2), encoding="utf-8")
"""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_under_root(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def load_manifest(library_root: Path) -> dict[str, Any]:
    manifest_path = library_root / "template_library_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CAD template manifest not found: {manifest_path}")
    return load_json(manifest_path)


def templates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("templates", [])
    if not isinstance(records, list):
        raise ValueError("template_library_manifest.json has invalid templates field")
    return [item for item in records if isinstance(item, dict)]


def template_by_id(manifest: dict[str, Any], template_id: str) -> dict[str, Any]:
    for item in templates(manifest):
        if item.get("template_id") == template_id:
            return item
    raise KeyError(f"Unknown CAD template: {template_id}")


def artifact_path(record: dict[str, Any], artifact: str) -> Path:
    files = record.get("files", {})
    if not isinstance(files, dict):
        raise ValueError(f"Template {record.get('template_id')} has invalid files field")
    path = resolve_under_root(files.get(artifact))
    if path is None:
        raise FileNotFoundError(f"Template {record.get('template_id')} has no {artifact} artifact")
    if not path.exists():
        raise FileNotFoundError(f"Template artifact is missing: {path}")
    return path


def ledger_summary(record: dict[str, Any]) -> dict[str, Any]:
    try:
        ledger = load_json(artifact_path(record, "assumption_ledger"))
    except (FileNotFoundError, ValueError):
        return {"available": False}
    assumptions = ledger.get("assumptions", [])
    blockers = ledger.get("measured_blockers", [])
    return {
        "available": True,
        "product_accuracy_ready": ledger.get("product_accuracy_ready"),
        "assumption_count": len(assumptions) if isinstance(assumptions, list) else 0,
        "measured_blocker_count": len(blockers) if isinstance(blockers, list) else 0,
        "solver_mapping": ledger.get("solver_mapping", {}),
    }


def list_templates(manifest: dict[str, Any]) -> None:
    for record in templates(manifest):
        files = record.get("files", {}) if isinstance(record.get("files"), dict) else {}
        ledger = ledger_summary(record)
        print(
            json.dumps(
                {
                    "template_id": record.get("template_id"),
                    "label": record.get("label"),
                    "variant_of": record.get("variant_of"),
                    "override_count": len(record.get("parameter_overrides", {}))
                    if isinstance(record.get("parameter_overrides"), dict)
                    else 0,
                    "source_truth_level": record.get("source_truth_level"),
                    "step": files.get("step"),
                    "brep": files.get("brep"),
                    "mesh": files.get("mesh"),
                    "assumption_ledger": files.get("assumption_ledger"),
                    "product_accuracy_ready": ledger.get("product_accuracy_ready"),
                    "assumption_count": ledger.get("assumption_count"),
                    "measured_blocker_count": ledger.get("measured_blocker_count"),
                },
                ensure_ascii=False,
            )
        )


def freecad_app_path() -> Path | None:
    for candidate in FREECAD_APP_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def bundle_version(app_path: Path | None) -> str | None:
    if not app_path:
        return None
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        return None
    with plist_path.open("rb") as handle:
        info = plistlib.load(handle)
    return str(info.get("CFBundleVersion") or info.get("CFBundleShortVersionString") or "") or None


def freecad_status() -> dict[str, Any]:
    app_path = freecad_app_path()
    cli_path = app_path / "Contents" / "MacOS" / "FreeCAD" if app_path else None
    cmd_path = app_path / "Contents" / "Resources" / "bin" / "freecadcmd" if app_path else None
    shell_path = shutil.which("FreeCAD")
    shell_cmd_path = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
    return {
        "installed": bool(app_path or shell_path),
        "preferred_app": str(app_path) if app_path else None,
        "preferred_executable": str(cli_path) if cli_path and cli_path.exists() else shell_path,
        "preferred_command": str(cmd_path) if cmd_path and cmd_path.exists() else shell_cmd_path,
        "bundle_version": bundle_version(app_path),
        "checked_locations": [str(path) for path in FREECAD_APP_CANDIDATES],
        "shell_freecad": shell_path,
        "shell_freecadcmd": shell_cmd_path,
    }


def open_command(path: Path, *, prefer_freecad: bool) -> list[str]:
    app_path = freecad_app_path()
    if prefer_freecad and app_path:
        return ["open", "-a", str(app_path), str(path)]
    if prefer_freecad and shutil.which("FreeCAD"):
        return ["FreeCAD", str(path)]
    return ["open", str(path)]


def freecad_cmd_path() -> str | None:
    status = freecad_status()
    command = status.get("preferred_command")
    return str(command) if command else None


def json_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def expected_bbox_um(parameters: dict[str, Any]) -> dict[str, float | None]:
    pitch_um = json_number(parameters.get("pitch_um"))
    nx = json_number(parameters.get("nx"))
    nz = json_number(parameters.get("nz"))
    si_thickness_um = json_number(parameters.get("si_thickness_um"))
    passivation_um = json_number(parameters.get("passivation_thickness_um"))
    cfa_um = json_number(parameters.get("cfa_thickness_um"))
    lens_um = json_number(parameters.get("lens_height_um"))
    return {
        "xlen": pitch_um * nx if pitch_um is not None and nx is not None else None,
        "zlen": pitch_um * nz if pitch_um is not None and nz is not None else None,
        "ymin": -si_thickness_um if si_thickness_um is not None else None,
        "ymax": sum(value for value in [passivation_um, cfa_um, lens_um] if value is not None)
        if all(value is not None for value in [passivation_um, cfa_um, lens_um])
        else None,
    }


def bbox_checks(report: dict[str, Any], expected: dict[str, float | None], tolerance_um: float) -> dict[str, Any]:
    bbox = report.get("bbox_um") if isinstance(report.get("bbox_um"), dict) else {}
    checks: dict[str, Any] = {"tolerance_um": tolerance_um, "passed": True, "items": []}
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        actual = json_number(bbox.get(key))
        passed = actual is not None and abs(actual - expected_value) <= tolerance_um
        checks["items"].append(
            {
                "field": key,
                "actual_um": actual,
                "expected_um": expected_value,
                "delta_um": actual - expected_value if actual is not None else None,
                "passed": passed,
            }
        )
        if not passed:
            checks["passed"] = False
    return checks


def run_freecad_probe(target: Path, *, fcstd_path: Path | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    command_path = freecad_cmd_path()
    if not command_path:
        return {
            "schema": "freecad_shape_probe_v1",
            "status": "FAIL",
            "path": str(target),
            "error": "freecadcmd not found",
        }
    with tempfile.TemporaryDirectory(prefix="pixel_freecad_probe_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        script_path = tmp_dir / "shape_probe.py"
        output_path = tmp_dir / "shape_probe_result.json"
        metadata_path = tmp_dir / "metadata.json"
        script_path.write_text(textwrap.dedent(FREECAD_SHAPE_PROBE), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata or {}, indent=2, ensure_ascii=False), encoding="utf-8")
        command = [
            command_path,
            str(script_path),
            "--pass",
            str(target),
            str(output_path),
            str(fcstd_path) if fcstd_path else "",
            str(metadata_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        if output_path.exists():
            result = load_json(output_path)
        else:
            result = {
                "schema": "freecad_shape_probe_v1",
                "status": "FAIL",
                "path": str(target),
                "error": "FreeCAD probe did not write a JSON result",
            }
        result["command"] = command
        result["returncode"] = completed.returncode
        result["stdout_tail"] = completed.stdout[-1000:]
        result["stderr_tail"] = completed.stderr[-1000:]
        if completed.returncode != 0 and result.get("status") == "PASS":
            result["status"] = "FAIL"
        return result


def extract_fcstd_parameters(path: Path) -> dict[str, Any]:
    command_path = freecad_cmd_path()
    if not command_path:
        return {
            "schema": "freecad_template_parameter_extract_v1",
            "status": "FAIL",
            "path": str(path),
            "error": "freecadcmd not found",
        }
    with tempfile.TemporaryDirectory(prefix="pixel_freecad_extract_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        script_path = tmp_dir / "parameter_extract.py"
        output_path = tmp_dir / "parameter_extract_result.json"
        script_path.write_text(textwrap.dedent(FREECAD_PARAMETER_EXTRACT_PROBE), encoding="utf-8")
        command = [command_path, str(script_path), "--pass", str(output_path), str(path)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        if output_path.exists():
            result = load_json(output_path)
        else:
            result = {
                "schema": "freecad_template_parameter_extract_v1",
                "status": "FAIL",
                "path": str(path),
                "error": "FreeCAD parameter extract did not write a JSON result",
            }
        result["command"] = command
        result["returncode"] = completed.returncode
        result["stdout_tail"] = completed.stdout[-1000:]
        result["stderr_tail"] = completed.stderr[-1000:]
        if completed.returncode != 0 and result.get("status") == "PASS":
            result["status"] = "FAIL"
        return result


def validate_freecad_library(
    library_root: Path,
    *,
    template_id: str = "",
    write_fcstd: bool = False,
    output_path: Path | None = None,
    tolerance_um: float = 1.0e-6,
) -> dict[str, Any]:
    manifest = load_manifest(library_root)
    selected = [template_by_id(manifest, template_id)] if template_id else templates(manifest)
    library_validation_path = library_root / "cad_template_validation_report.json"
    library_validation = load_json(library_validation_path) if library_validation_path.exists() else {}
    design_by_template = {
        str(item.get("template_id")): item
        for item in library_validation.get("templates", [])
        if isinstance(item, dict) and item.get("template_id")
    } if isinstance(library_validation.get("templates"), list) else {}
    freecad = freecad_status()
    report_templates = []
    status = "PASS" if freecad.get("preferred_command") else "FAIL"
    for record in selected:
        files = record.get("files", {}) if isinstance(record.get("files"), dict) else {}
        parameters_path = resolve_under_root(files.get("parameters"))
        parameters = load_json(parameters_path) if parameters_path and parameters_path.exists() else {}
        expected = expected_bbox_um(parameters)
        design_validation = design_by_template.get(str(record.get("template_id") or ""), {})
        metadata = {
            "template_id": record.get("template_id"),
            "label": record.get("label"),
            "variant_of": record.get("variant_of"),
            "source_truth_level": record.get("source_truth_level"),
            "expected_bbox_um": expected,
            "parameters": parameters,
            "design_rule_status": design_validation.get("design_rule_status"),
            "design_rule_fail_count": design_validation.get("design_rule_fail_count"),
            "design_rule_validation": design_validation.get("design_rule_validation", {}),
        }
        template_dir = parameters_path.parent if parameters_path else library_root / str(record.get("template_id"))
        step_path = artifact_path(record, "step")
        brep_path = artifact_path(record, "brep")
        fcstd_path = template_dir / "model.FCStd" if write_fcstd else None
        step = run_freecad_probe(step_path, fcstd_path=fcstd_path, metadata=metadata)
        brep = run_freecad_probe(brep_path)
        step_checks = bbox_checks(step, expected, tolerance_um)
        brep_checks = bbox_checks(brep, expected, tolerance_um)
        fcstd_payload = step.get("fcstd") if isinstance(step.get("fcstd"), dict) else {}
        fcstd_ok = (
            not write_fcstd
            or (
                bool(fcstd_payload.get("exists"))
                and bool(fcstd_payload.get("contains_parameter_sheet"))
                and bool(fcstd_payload.get("contains_validation_sheet"))
            )
        )
        template_status = (
            "PASS"
            if step.get("status") == "PASS"
            and brep.get("status") == "PASS"
            and step_checks.get("passed")
            and brep_checks.get("passed")
            and fcstd_ok
            else "FAIL"
        )
        if template_status != "PASS":
            status = "FAIL"
        report_templates.append(
            {
                "template_id": record.get("template_id"),
                "label": record.get("label"),
                "status": template_status,
                "expected_bbox_um": expected,
                "step": step,
                "step_bbox_checks": step_checks,
                "brep": brep,
                "brep_bbox_checks": brep_checks,
                "fcstd": {
                    "requested": write_fcstd,
                    "path": str(fcstd_path) if fcstd_path else None,
                    "exists": bool(fcstd_path and fcstd_path.exists()),
                    "size_bytes": fcstd_path.stat().st_size if fcstd_path and fcstd_path.exists() else None,
                    "contains_parameter_sheet": step.get("fcstd", {}).get("contains_parameter_sheet")
                    if isinstance(step.get("fcstd"), dict)
                    else None,
                    "contains_validation_sheet": step.get("fcstd", {}).get("contains_validation_sheet")
                    if isinstance(step.get("fcstd"), dict)
                    else None,
                },
            }
        )
    report = {
        "schema": "cad_template_freecad_validation_report_v1",
        "status": status,
        "freecad": freecad,
        "template_count": len(report_templates),
        "template_id": template_id or None,
        "write_fcstd": write_fcstd,
        "tolerance_um": tolerance_um,
        "templates": report_templates,
        "notes": [
            "This validates that generated STEP/BREP files can be read by FreeCAD headlessly.",
            "Bounding box checks compare CAD dimensions against template_parameters.json.",
            "A PASS here validates CAD artifact integrity, not measured process accuracy.",
        ],
    }
    target = output_path or (library_root / FREECAD_VALIDATION_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> None:
    if args.check_tools:
        print(json.dumps({"freecad": freecad_status()}, indent=2))
        return

    if args.validate_freecad:
        report = validate_freecad_library(
            args.library_root,
            template_id=args.template,
            write_fcstd=args.write_fcstd,
            output_path=args.output,
            tolerance_um=args.tolerance_um,
        )
        print(json.dumps(report, indent=2))
        return

    manifest = load_manifest(args.library_root)
    if args.list:
        list_templates(manifest)
        return

    if not args.template:
        raise SystemExit("--template is required unless --list is used")

    record = template_by_id(manifest, args.template)
    if args.show_ledger:
        print(json.dumps(load_json(artifact_path(record, "assumption_ledger")), indent=2))
        return

    target = artifact_path(record, args.artifact)
    command = open_command(target, prefer_freecad=not args.default_viewer)
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--list", action="store_true", help="List available CAD templates and assumption ledger status.")
    parser.add_argument("--template", default="", help="Template id to review.")
    parser.add_argument("--artifact", choices=("step", "brep", "mesh", "geometry_import", "assumption_ledger"), default="step")
    parser.add_argument("--show-ledger", action="store_true", help="Print the full assumption ledger JSON for the selected template.")
    parser.add_argument("--check-tools", action="store_true", help="Print CAD viewer/tool availability.")
    parser.add_argument("--validate-freecad", action="store_true", help="Validate STEP/BREP artifacts through headless FreeCAD.")
    parser.add_argument("--write-fcstd", action="store_true", help="Write model.FCStd files while validating FreeCAD STEP imports.")
    parser.add_argument("--output", type=Path, default=None, help="Validation report path for --validate-freecad.")
    parser.add_argument("--tolerance-um", type=float, default=1.0e-6, help="FreeCAD bounding-box check tolerance in micrometers.")
    parser.add_argument("--print-command", action="store_true", help="Print the open command without executing it.")
    parser.add_argument("--default-viewer", action="store_true", help="Use the default macOS viewer instead of preferring FreeCAD.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
