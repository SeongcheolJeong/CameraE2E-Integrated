#!/usr/bin/env python3
"""Build reference-sensor CAD templates and smoke-analysis DB from sensor_db.

Input is the local ``sensor_db/sensor_catalog.json`` generated from licensed
reference material. Outputs are local research artifacts under ``runs/``:
sensor-specific CAD templates, solver command provenance, optional smoke
simulation results, and a compact analysis catalog.

The generated CAD is parameter-derived proxy geometry, not measured product
mask or process CAD.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from pixel_cad_template_library import OclBlock, TemplateSpec, template_specs, write_template


ROOT = Path(__file__).resolve().parent
DEFAULT_SENSOR_DB = ROOT / "sensor_db" / "sensor_catalog.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "reference_sensor_template_analysis"
MEEP_PYTHON = ROOT / ".meep-env" / "bin" / "python"
DEFAULT_WAVELENGTHS_NM = "550"
DEFAULT_CASES = "center:0:0:0:0:0:0"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_slug(value: str, *, fallback: str = "sensor") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sensor_catalog_records(sensor_db: Path) -> list[dict[str, Any]]:
    catalog = read_json(sensor_db)
    records = catalog.get("records")
    if not isinstance(records, list):
        raise ValueError(f"sensor catalog has no records list: {sensor_db}")
    return [record for record in records if isinstance(record, dict)]


def number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        if value != value:
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def int_number(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def sensor_score(record: dict[str, Any]) -> float:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    specs = record.get("derived_specs", {}) if isinstance(record.get("derived_specs"), dict) else {}
    pitch = number(specs.get("pixel_pitch_um"))
    if pitch is None:
        return -1.0
    manufacturer = str(metadata.get("manufacturer") or "").lower()
    year = int_number(metadata.get("analysis_year"))
    megapixels = number(specs.get("resolution_mp"), 0.0) or 0.0
    images = number(metadata.get("images_count"), 0.0) or 0.0
    docs = number(metadata.get("documents_count"), 0.0) or 0.0
    score = 0.0
    score += min(megapixels, 200.0) / 5.0
    score += max(0, year - 2018) * 2.0
    score += min(images, 500.0) / 50.0
    score += min(docs, 5.0)
    if any(name in manufacturer for name in ("sony", "samsung", "omnivision", "sk hynix", "smartsens", "canon", "stmicroelectronics")):
        score += 8.0
    if pitch <= 1.4:
        score += 5.0
    if pitch <= 1.0:
        score += 5.0
    if record.get("generated_files", {}).get("stack_config"):
        score += 2.0
    if record.get("generated_files", {}).get("tcad_profile"):
        score += 2.0
    return score


def dedupe_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    manufacturer = safe_slug(str(metadata.get("manufacturer") or "unknown"))
    device = safe_slug(str(metadata.get("device_name") or record.get("code") or "unknown"))
    return f"{manufacturer}:{device}"


def select_major_sensors(records: list[dict[str, Any]], max_sensors: int, include_codes: set[str]) -> list[dict[str, Any]]:
    by_key: dict[str, tuple[float, dict[str, Any]]] = {}
    for record in records:
        score = sensor_score(record)
        if score < 0:
            continue
        code = str(record.get("code") or "")
        key = dedupe_key(record)
        if code in include_codes:
            score += 1000.0
        current = by_key.get(key)
        if current is None or score > current[0]:
            by_key[key] = (score, record)
    selected = [record for _score, record in sorted(by_key.values(), key=lambda item: -item[0])]
    if include_codes:
        selected_codes = {str(record.get("code") or "") for record in selected[:max_sensors]}
        for record in sorted(records, key=lambda item: -sensor_score(item)):
            code = str(record.get("code") or "")
            if code in include_codes and code not in selected_codes:
                selected.insert(0, record)
                selected_codes.add(code)
    return selected[:max_sensors]


def selectable_sensor_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, tuple[float, dict[str, Any]]] = {}
    for record in records:
        score = sensor_score(record)
        if score < 0:
            continue
        key = dedupe_key(record)
        current = by_key.get(key)
        if current is None or score > current[0]:
            by_key[key] = (score, record)
    return [record for _score, record in sorted(by_key.values(), key=lambda item: -item[0])]


def selectable_all_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in sorted(records, key=lambda item: -sensor_score(item)) if sensor_score(record) >= 0]


def select_sensor_records(records: list[dict[str, Any]], args: argparse.Namespace, include_codes: set[str]) -> list[dict[str, Any]]:
    if args.all_records:
        selected = selectable_all_records(records)
        if include_codes:
            selected_codes = {str(record.get("code") or "") for record in selected}
            selected.extend(record for record in records if str(record.get("code") or "") in include_codes and str(record.get("code") or "") not in selected_codes)
        return selected[: args.max_sensors]
    return select_major_sensors(records, args.max_sensors, include_codes)


def write_selection_index(output_dir: Path, records: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected_codes = {str(record.get("code") or "") for record in selected}
    rows = []
    candidates = selectable_all_records(records) if args.all_records else selectable_sensor_candidates(records)
    for rank, record in enumerate(candidates, 1):
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        specs = record.get("derived_specs", {}) if isinstance(record.get("derived_specs"), dict) else {}
        code = str(record.get("code") or "")
        row = {
            "rank": rank,
            "selected": code in selected_codes,
            "selection_reason": "all pitch-qualified records" if args.all_records and code in selected_codes else "top scored main sensor" if code in selected_codes else "below selected rank window",
            "score": round(sensor_score(record), 4),
            "code": code,
            "manufacturer": metadata.get("manufacturer"),
            "device_name": metadata.get("device_name"),
            "analysis_year": metadata.get("analysis_year"),
            "pixel_pitch_um": specs.get("pixel_pitch_um"),
            "resolution_mp": specs.get("resolution_mp"),
            "pixel_architecture": specs.get("pixel_architecture"),
            "cfa_pattern": specs.get("cfa_pattern"),
            "source_template_id": topology_for_record(record),
        }
        rows.append(row)
    payload = {
        "schema": "reference_sensor_selection_index_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sensor_db": repo_rel(Path(args.sensor_db)),
        "source_record_count": len(records),
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "selection_policy": {
            "max_sensors": args.max_sensors,
            "all_records": args.all_records,
            "include_codes": args.include_codes,
            "score_inputs": [
                "resolution_mp",
                "analysis_year",
                "images_count",
                "documents_count",
                "major manufacturer bonus",
                "small pixel pitch bonus",
                "generated stack/tcad profile availability",
            ],
            "dedupe_key": "manufacturer + device_name",
        },
        "records": rows,
    }
    write_json(output_dir / "selection_index.json", payload)
    with (output_dir / "selection_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "selected",
            "selection_reason",
            "score",
            "code",
            "manufacturer",
            "device_name",
            "analysis_year",
            "pixel_pitch_um",
            "resolution_mp",
            "pixel_architecture",
            "cfa_pattern",
            "source_template_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def topology_for_record(record: dict[str, Any]) -> str:
    specs = record.get("derived_specs", {}) if isinstance(record.get("derived_specs"), dict) else {}
    text = " ".join(str(specs.get(key) or "").lower() for key in ("pixel_architecture", "cfa_pattern", "microlens_type"))
    if "super_qpd" in text or "qpd" in text:
        return "qpd_split_pd_no_shield_2x2"
    if "nona" in text or "nine" in text:
        return "nona_3x3_ocl"
    if any(token in text for token in ("quad", "tetracell", "four_shared", "eight_shared")):
        return "quad_2x2_ocl"
    return "bayer_1x1_3x3"


def stack_geometry(record: dict[str, Any]) -> dict[str, Any]:
    stack_path_text = record.get("generated_files", {}).get("stack_config") if isinstance(record.get("generated_files"), dict) else None
    if not stack_path_text:
        return {}
    stack_path = Path(str(stack_path_text))
    if not stack_path.is_absolute():
        stack_path = ROOT / stack_path
    if not stack_path.exists():
        return {}
    stack = read_json(stack_path)
    geometry = stack.get("geometry_um", {})
    return geometry if isinstance(geometry, dict) else {}


def clamped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def spec_for_record(record: dict[str, Any]) -> tuple[TemplateSpec, str]:
    specs = record.get("derived_specs", {}) if isinstance(record.get("derived_specs"), dict) else {}
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    base_id = topology_for_record(record)
    base = template_specs()[base_id]
    geometry = stack_geometry(record)
    pitch = number(specs.get("pixel_pitch_um"), number(geometry.get("pitch"), base.pitch_um)) or base.pitch_um
    si = number(specs.get("active_si_thickness_um"), number(geometry.get("si_thickness"), base.si_thickness_um)) or base.si_thickness_um
    cfa = number(specs.get("cfa_thickness_um"), number(geometry.get("cfa_thickness"), base.cfa_thickness_um)) or base.cfa_thickness_um
    lens = number(geometry.get("lens_height"), base.lens_height_um) or base.lens_height_um
    passivation = number(geometry.get("passivation_thickness"), base.passivation_thickness_um) or base.passivation_thickness_um
    dti_depth = number(specs.get("dti_depth_um"), number(geometry.get("si_thickness"), si)) or si
    dti_width = number(specs.get("dti_width_um"), base.dti_width_um) or base.dti_width_um
    code = str(record.get("code") or "unknown")
    manufacturer = str(metadata.get("manufacturer") or "Unknown")
    device = str(metadata.get("device_name") or "Unknown device")
    sensor_id = safe_slug(f"{code}_{manufacturer}_{device}")
    label = f"{manufacturer} {device} ({code})"
    notes = tuple(base.notes) + (
        f"Reference sensor derived from sensor_db record {code}.",
        "CAD dimensions are parameter-derived from extracted metadata and proxy stack config.",
        "Not a measured mask/process deck; use for benchmark trend studies only.",
    )
    spec = replace(
        base,
        template_id=sensor_id,
        label=label,
        pitch_um=round(pitch, 4),
        si_thickness_um=round(clamped(si, 0.5, 20.0), 4),
        cfa_thickness_um=round(clamped(cfa, 0.2, 2.0), 4),
        lens_height_um=round(clamped(lens, 0.12, 1.4), 4),
        passivation_thickness_um=round(clamped(passivation, 0.02, 1.5), 4),
        dti_depth_um=round(clamped(dti_depth, 0.2, 20.0), 4),
        dti_width_um=round(clamped(dti_width, 0.02, 0.35), 4),
        pd_depth_max_um=round(clamped(si * 0.42, 0.25, max(0.3, si - 0.05)), 4),
        pd_margin_um=round(clamped(pitch * 0.18, 0.05, max(0.055, pitch * 0.32)), 4),
        lens_edge_gap_um=round(clamped(pitch * 0.055, 0.015, 0.12), 4),
        cfa_gap_um=round(clamped(pitch * 0.014, 0.004, 0.035), 4),
        notes=notes,
    )
    return spec, base_id


def ocl_layout_string(spec: TemplateSpec) -> str:
    return ",".join(f"{block.lens_id}:{block.ix}:{block.iz}:{block.sx}:{block.sz}" for block in spec.ocl_blocks)


def central_block(spec: TemplateSpec) -> OclBlock:
    center_x = spec.nx / 2.0
    center_z = spec.nz / 2.0
    return min(
        spec.ocl_blocks,
        key=lambda block: (abs((block.ix + block.sx / 2.0) - center_x) + abs((block.iz + block.sz / 2.0) - center_z), block.lens_id),
    )


def solver_cfa_pattern(spec: TemplateSpec) -> tuple[str, str]:
    if spec.cfa_pattern.startswith("uniform_"):
        return "uniform", spec.cfa_pattern.split("_", 1)[1]
    return spec.cfa_pattern, "green"


def solver_command_for_template(
    spec: TemplateSpec,
    template_dir: Path,
    output_dir: Path,
    stack_config: Path | None,
    *,
    resolution: int,
    after_source_time: float,
    wavelengths_nm: str,
    cases: str,
) -> list[str]:
    cfa_pattern, color_channel = solver_cfa_pattern(spec)
    geometry_import = f"@{repo_rel(template_dir / 'geometry_import.json')}"
    command = [
        str(MEEP_PYTHON),
        "meep_supercell_lut.py",
        "--mode",
        "ocl-layout",
        "--layout-nx",
        str(spec.nx),
        "--layout-nz",
        str(spec.nz),
        "--ocl-layout",
        ocl_layout_string(spec),
        "--ocl-polygons",
        geometry_import,
        "--cfa-polygons",
        geometry_import,
        "--ocl-layout-name",
        spec.template_id[:80],
        "--target-lens-id",
        central_block(spec).lens_id,
        "--cfa-pattern",
        cfa_pattern,
        "--color-channel",
        color_channel,
        "--wavelengths-nm",
        wavelengths_nm,
        "--cases",
        cases,
        "--resolution",
        str(resolution),
        "--after-source-time",
        str(after_source_time),
        "--pml-um",
        "0.45",
        "--grid-snap-y",
        "nearest",
        "--output-dir",
        str(output_dir),
    ]
    split_mode = spec.split_mode.replace("_", "-")
    if split_mode in {"dual-x", "dual-z", "quad"}:
        command.extend(["--split-mode", split_mode, "--collection-mode", "split-pd"])
    else:
        command.extend(["--collection-mode", "pixel"])
    if spec.shield_mode in {"edge", "off", "pdaf_left", "pdaf_right", "pdaf_pair"}:
        command.extend(["--shield-mode", spec.shield_mode])
    if stack_config is not None:
        command.extend(["--stack-config", str(stack_config)])
    return command


def artifact_index(path: Path) -> dict[str, Any]:
    files = []
    if path.exists():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            files.append({"path": repo_rel(item), "bytes": item.stat().st_size})
    return {"file_count": len(files), "files": files[:80]}


def parse_simulation_metrics(output_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for json_path in sorted(output_dir.rglob("*.json")):
        try:
            payload = read_json(json_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        schema = payload.get("schema")
        if schema:
            metrics.setdefault("schemas", []).append({"path": repo_rel(json_path), "schema": schema})
        for key in ("kpi_summary", "summary", "convergence", "gates"):
            value = payload.get(key)
            if isinstance(value, dict) and key not in metrics:
                metrics[key] = value
    return metrics


def run_smoke(command: list[str], output_dir: Path, timeout_s: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stdout_path = output_dir / "solver_stdout.log"
    stderr_path = output_dir / "solver_stderr.log"
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        status = "PASS" if result.returncode == 0 else "FAIL"
        exit_code: int | str = result.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        status = "TIMEOUT"
        exit_code = "timeout"
        error = str(exc)
    elapsed = round(time.time() - started, 3)
    return {
        "status": status,
        "exit_code": exit_code,
        "elapsed_s": elapsed,
        "command": command,
        "stdout": repo_rel(stdout_path),
        "stderr": repo_rel(stderr_path),
        "error": error,
        "metrics": parse_simulation_metrics(output_dir),
        "artifacts": artifact_index(output_dir),
    }


def proxy_analysis(record: dict[str, Any], spec: TemplateSpec, source_template_id: str) -> dict[str, Any]:
    specs = record.get("derived_specs", {}) if isinstance(record.get("derived_specs"), dict) else {}
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    pitch = spec.pitch_um
    ocl_groups = len(spec.ocl_blocks)
    return {
        "schema": "reference_sensor_proxy_analysis_v1",
        "source_template_id": source_template_id,
        "pixel_pitch_um": pitch,
        "domain_pixels": spec.nx * spec.nz,
        "domain_size_um": {
            "x": round(spec.nx * pitch, 4),
            "z": round(spec.nz * pitch, 4),
        },
        "ocl_group_count": ocl_groups,
        "cfa_pattern": spec.cfa_pattern,
        "split_mode": spec.split_mode,
        "shield_mode": spec.shield_mode,
        "resolution_mp": specs.get("resolution_mp"),
        "dti_type": specs.get("dti_type"),
        "dti_depth_um": specs.get("dti_depth_um"),
        "optical_stack_height_um": specs.get("optical_stack_height_um"),
        "analysis_year": metadata.get("analysis_year"),
        "readiness": {
            "cad_generated": True,
            "fdtd_smoke_required_for_trend": True,
            "product_accuracy_ready": False,
            "reason": "Measured mask geometry, OCL profile, n,k, implant, trap, and convergence calibration are not present.",
        },
    }


def source_stack_path(record: dict[str, Any]) -> Path | None:
    text = record.get("generated_files", {}).get("stack_config") if isinstance(record.get("generated_files"), dict) else None
    if not text:
        return None
    path = Path(str(text))
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def source_tcad_profile(record: dict[str, Any]) -> Path | None:
    text = record.get("generated_files", {}).get("tcad_profile") if isinstance(record.get("generated_files"), dict) else None
    if not text:
        return None
    path = Path(str(text))
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def analyze_sensor(
    record: dict[str, Any],
    output_dir: Path,
    *,
    mesh: bool,
    run_smoke_solver: bool,
    smoke_enabled: bool,
    resolution: int,
    after_source_time: float,
    wavelengths_nm: str,
    cases: str,
    timeout_s: int,
) -> dict[str, Any]:
    spec, source_template_id = spec_for_record(record)
    template_result = write_template(spec, output_dir / "templates", mesh=mesh)
    template_dir = output_dir / "templates" / spec.template_id
    stack_config = source_stack_path(record)
    solver_output_dir = output_dir / "simulations" / spec.template_id / "fdtd_smoke"
    command = solver_command_for_template(
        spec,
        template_dir,
        solver_output_dir,
        stack_config,
        resolution=resolution,
        after_source_time=after_source_time,
        wavelengths_nm=wavelengths_nm,
        cases=cases,
    )
    simulation = {
        "status": "NOT_RUN",
        "reason": "smoke execution disabled",
        "command": command,
        "artifacts": {"file_count": 0, "files": []},
    }
    if run_smoke_solver and smoke_enabled:
        simulation = run_smoke(command, solver_output_dir, timeout_s)
    elif run_smoke_solver and not smoke_enabled:
        simulation["reason"] = "smoke limit reached"
    record_summary = {
        "code": record.get("code"),
        "metadata": record.get("metadata", {}),
        "derived_specs": record.get("derived_specs", {}),
        "generated_files": record.get("generated_files", {}),
    }
    write_json(template_dir / "source_sensor_record_summary.json", record_summary)
    analysis = proxy_analysis(record, spec, source_template_id)
    row = {
        "sensor_id": spec.template_id,
        "code": record.get("code"),
        "manufacturer": record.get("metadata", {}).get("manufacturer", ""),
        "device_name": record.get("metadata", {}).get("device_name", ""),
        "source_template_id": source_template_id,
        "template": template_result,
        "template_parameters": asdict(spec),
        "source_stack_config": repo_rel(stack_config) if stack_config else "",
        "source_tcad_profile": repo_rel(source_tcad_profile(record)) if source_tcad_profile(record) else "",
        "analysis": analysis,
        "simulation": simulation,
    }
    write_json(template_dir / "analysis_record.json", row)
    return row


def write_catalog(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, source_record_count: int) -> None:
    catalog = {
        "schema": "reference_sensor_template_analysis_db_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sensor_db": repo_rel(Path(args.sensor_db)),
        "output_dir": repo_rel(output_dir),
        "selection": {
            "max_sensors": args.max_sensors,
            "all_records": args.all_records,
            "include_codes": args.include_codes,
            "run_smoke": args.run_smoke,
            "smoke_count": args.smoke_count,
            "resolution": args.resolution,
            "after_source_time": args.after_source_time,
            "wavelengths_nm": args.wavelengths_nm,
            "selection_index_json": "selection_index.json",
            "selection_index_csv": "selection_index.csv",
        },
        "coverage": catalog_coverage(rows, source_record_count),
        "limitations": [
            "Reference sensor CAD is parameter-derived from sensor_db metadata and proxy stack configs.",
            "Generated STEP/BREP files are CAD review artifacts, not measured product mask CAD.",
            "Smoke simulations are research/trend checks and are not product-accuracy LUTs.",
        ],
        "record_count": len(rows),
        "records": rows,
    }
    write_json(output_dir / "analysis_catalog.json", catalog)
    csv_path = output_dir / "analysis_catalog.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sensor_id",
            "code",
            "manufacturer",
            "device_name",
            "source_template_id",
            "pitch_um",
            "resolution_mp",
            "cfa_pattern",
            "split_mode",
            "ocl_group_count",
            "simulation_status",
            "grid_resolution_gate_pass",
            "recommended_resolution_px_per_um",
            "total_response",
            "simulation_artifact_count",
            "template_step",
            "template_brep",
            "analysis_record",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            files = row.get("template", {}).get("files", {})
            analysis = row.get("analysis", {})
            checkpoint = simulation_checkpoint(row)
            writer.writerow(
                {
                    "sensor_id": row["sensor_id"],
                    "code": row["code"],
                    "manufacturer": row["manufacturer"],
                    "device_name": row["device_name"],
                    "source_template_id": row["source_template_id"],
                    "pitch_um": analysis.get("pixel_pitch_um"),
                    "resolution_mp": analysis.get("resolution_mp"),
                    "cfa_pattern": analysis.get("cfa_pattern"),
                    "split_mode": analysis.get("split_mode"),
                    "ocl_group_count": analysis.get("ocl_group_count"),
                    "simulation_status": row.get("simulation", {}).get("status"),
                    "grid_resolution_gate_pass": checkpoint.get("grid_resolution_gate_pass"),
                    "recommended_resolution_px_per_um": checkpoint.get("recommended_min_resolution_px_per_um"),
                    "total_response": checkpoint.get("total_response"),
                    "simulation_artifact_count": row.get("simulation", {}).get("artifacts", {}).get("file_count"),
                    "template_step": files.get("step"),
                    "template_brep": files.get("brep"),
                    "analysis_record": f"templates/{row['sensor_id']}/analysis_record.json",
                }
            )
    write_sqlite_catalog(output_dir / "analysis_catalog.sqlite", rows, catalog)
    write_html(output_dir / "index.html", rows)
    write_standalone_html(output_dir / "reference_sensor_template_analysis_standalone.html", rows, catalog)
    write_readme(output_dir / "README.md", catalog)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    body_rows = []
    for row in rows:
        files = row.get("template", {}).get("files", {})
        analysis = row.get("analysis", {})
        simulation = row.get("simulation", {})
        preview = files.get("footprint_preview")
        preview_html = f'<img src="{escape(os.path.relpath(preview, path.parent))}" alt="{escape(row["sensor_id"])} preview">' if preview else ""
        body_rows.append(
            "<tr>"
            f"<td>{preview_html}</td>"
            f"<td><strong>{escape(str(row['manufacturer']))}</strong><br>{escape(str(row['device_name']))}<br><code>{escape(str(row['code']))}</code></td>"
            f"<td>{escape(str(row['source_template_id']))}</td>"
            f"<td>{escape(str(analysis.get('pixel_pitch_um')))}</td>"
            f"<td>{escape(str(analysis.get('resolution_mp')))}</td>"
            f"<td>{escape(str(analysis.get('cfa_pattern')))}<br>{escape(str(analysis.get('split_mode')))}</td>"
            f"<td>{escape(str(simulation.get('status')))}</td>"
            f"<td><a href=\"{escape(os.path.relpath(files.get('step', ''), path.parent))}\">STEP</a><br><a href=\"{escape(os.path.relpath(files.get('brep', ''), path.parent))}\">BREP</a><br><a href=\"{escape(os.path.relpath('templates/' + row['sensor_id'] + '/analysis_record.json', '.'))}\">analysis</a></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reference Sensor Template Analysis DB</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7fafc; color: #17212b; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    p {{ color: #536273; margin: 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #d9e2ec; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eaf0f6; position: sticky; top: 0; }}
    img {{ width: 180px; max-width: 100%; border-radius: 6px; border: 1px solid #c7d2de; background: #06111a; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; background: #edf2f7; padding: 1px 4px; border-radius: 4px; }}
    a {{ color: #0967d2; }}
  </style>
</head>
<body>
  <h1>Reference Sensor Template Analysis DB</h1>
  <p>Sensor-specific proxy CAD templates, solver provenance, and smoke analysis records generated from local sensor_db metadata.</p>
  <table>
    <thead><tr><th>Preview</th><th>Sensor</th><th>Template</th><th>Pitch</th><th>MP</th><th>Pattern</th><th>Simulation</th><th>Artifacts</th></tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def resolve_output_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def inline_svg(path_text: str | None) -> str:
    path = resolve_output_path(path_text)
    if path is None or not path.exists():
        return '<div class="preview-empty">No preview</div>'
    try:
        svg = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return '<div class="preview-empty">Preview unavailable</div>'
    # Generated template previews are trusted local artifacts from this pipeline.
    return svg.replace("<svg ", '<svg class="sensor-preview" ', 1)


def simulation_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
    artifacts = row.get("simulation", {}).get("artifacts", {}).get("files", [])
    if not isinstance(artifacts, list):
        return {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("path") or "")
        if not path_text.endswith("camera_lut_checkpoint.json"):
            continue
        path = resolve_output_path(path_text)
        if path and path.exists():
            try:
                payload = read_json(path)
            except Exception:
                return {}
            point = payload.get("last_sweep_point")
            return point if isinstance(point, dict) else {}
    return {}


def gate_counter_key(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "CHECK"
    return "N/A"


def catalog_coverage(rows: list[dict[str, Any]], source_record_count: int) -> dict[str, Any]:
    pitches = [row.get("analysis", {}).get("pixel_pitch_um") for row in rows]
    numeric_pitches = [pitch for pitch in pitches if isinstance(pitch, (int, float))]
    grid_gates = Counter(gate_counter_key(simulation_checkpoint(row).get("grid_resolution_gate_pass")) for row in rows)
    simulation_statuses = Counter(str(row.get("simulation", {}).get("status") or "UNKNOWN") for row in rows)
    template_files = [row.get("template", {}).get("files", {}) for row in rows]
    return {
        "source_record_count": source_record_count,
        "selected_record_count": len(rows),
        "selection_fraction": round(len(rows) / source_record_count, 4) if source_record_count else None,
        "manufacturer_counts": dict(sorted(Counter(str(row.get("manufacturer") or "Unknown") for row in rows).items())),
        "source_template_counts": dict(sorted(Counter(str(row.get("source_template_id") or "unknown") for row in rows).items())),
        "simulation_status_counts": dict(sorted(simulation_statuses.items())),
        "grid_resolution_gate_counts": dict(sorted(grid_gates.items())),
        "cad_artifact_counts": {
            "step": sum(1 for files in template_files if files.get("step")),
            "brep": sum(1 for files in template_files if files.get("brep")),
            "geometry_import": sum(1 for files in template_files if files.get("geometry_import")),
            "footprint_preview": sum(1 for files in template_files if files.get("footprint_preview")),
        },
        "pitch_um_range": {
            "min": min(numeric_pitches) if numeric_pitches else None,
            "max": max(numeric_pitches) if numeric_pitches else None,
        },
        "accuracy_status": "research_trend_only",
        "product_lut_ready": False,
    }


def sqlite_bool(value: Any) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    return None


def sqlite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_sqlite_catalog(path: Path, rows: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE sensors (
                sensor_id TEXT PRIMARY KEY,
                code TEXT,
                manufacturer TEXT,
                device_name TEXT,
                source_template_id TEXT,
                pixel_pitch_um REAL,
                resolution_mp REAL,
                cfa_pattern TEXT,
                split_mode TEXT,
                domain_pixels INTEGER,
                ocl_group_count INTEGER,
                simulation_status TEXT,
                grid_resolution_gate_pass INTEGER,
                recommended_resolution_px_per_um REAL,
                total_response REAL,
                simulation_artifact_count INTEGER,
                step_path TEXT,
                brep_path TEXT,
                analysis_record_path TEXT,
                product_lut_ready INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE artifacts (
                sensor_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                bytes INTEGER,
                FOREIGN KEY(sensor_id) REFERENCES sensors(sensor_id)
            )
            """
        )
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema", catalog["schema"]))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("generated_at", catalog["generated_at"]))
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("coverage_json", json.dumps(catalog["coverage"], ensure_ascii=False)))
        for row in rows:
            analysis = row.get("analysis", {})
            simulation = row.get("simulation", {})
            checkpoint = simulation_checkpoint(row)
            files = row.get("template", {}).get("files", {})
            conn.execute(
                """
                INSERT INTO sensors (
                    sensor_id, code, manufacturer, device_name, source_template_id,
                    pixel_pitch_um, resolution_mp, cfa_pattern, split_mode,
                    domain_pixels, ocl_group_count, simulation_status,
                    grid_resolution_gate_pass, recommended_resolution_px_per_um,
                    total_response, simulation_artifact_count,
                    step_path, brep_path, analysis_record_path, product_lut_ready
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("sensor_id"),
                    row.get("code"),
                    row.get("manufacturer"),
                    row.get("device_name"),
                    row.get("source_template_id"),
                    sqlite_float(analysis.get("pixel_pitch_um")),
                    sqlite_float(analysis.get("resolution_mp")),
                    analysis.get("cfa_pattern"),
                    analysis.get("split_mode"),
                    analysis.get("domain_pixels"),
                    analysis.get("ocl_group_count"),
                    simulation.get("status"),
                    sqlite_bool(checkpoint.get("grid_resolution_gate_pass")),
                    sqlite_float(checkpoint.get("recommended_min_resolution_px_per_um")),
                    sqlite_float(checkpoint.get("total_response")),
                    simulation.get("artifacts", {}).get("file_count"),
                    files.get("step"),
                    files.get("brep"),
                    f"templates/{row.get('sensor_id')}/analysis_record.json",
                    0,
                ),
            )
            for kind, value in files.items():
                if value:
                    artifact_path = resolve_output_path(value)
                    conn.execute(
                        "INSERT INTO artifacts(sensor_id, kind, path, bytes) VALUES (?, ?, ?, ?)",
                        (row.get("sensor_id"), kind, value, artifact_path.stat().st_size if artifact_path and artifact_path.exists() else None),
                    )
            for artifact in simulation.get("artifacts", {}).get("files", []):
                if isinstance(artifact, dict) and artifact.get("path"):
                    conn.execute(
                        "INSERT INTO artifacts(sensor_id, kind, path, bytes) VALUES (?, ?, ?, ?)",
                        (row.get("sensor_id"), "simulation", artifact.get("path"), artifact.get("bytes")),
                    )
        conn.execute("CREATE INDEX idx_sensors_manufacturer ON sensors(manufacturer)")
        conn.execute("CREATE INDEX idx_sensors_template ON sensors(source_template_id)")
        conn.execute("CREATE INDEX idx_sensors_grid_gate ON sensors(grid_resolution_gate_pass)")
        conn.execute("CREATE INDEX idx_artifacts_sensor ON artifacts(sensor_id)")


def status_class(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value == "pass" or value == "true":
        return "pass"
    if value in {"fail", "false", "timeout"}:
        return "fail"
    if value in {"check", "not_run", "not run", "not_applicable", "not applicable"}:
        return "check"
    return "neutral"


def gate_label(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "CHECK"
    return "N/A"


def gate_status_class(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "check"
    return "neutral"


def compact_number(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}g}"
    return str(value) if value is not None else "-"


def write_standalone_html(path: Path, rows: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    smoke_pass = sum(1 for row in rows if row.get("simulation", {}).get("status") == "PASS")
    step_count = sum(1 for row in rows if row.get("template", {}).get("files", {}).get("step"))
    brep_count = sum(1 for row in rows if row.get("template", {}).get("files", {}).get("brep"))
    coverage = catalog.get("coverage", {})
    manufacturer_counts = ", ".join(f"{name} {count}" for name, count in coverage.get("manufacturer_counts", {}).items())
    template_counts = ", ".join(f"{name} {count}" for name, count in coverage.get("source_template_counts", {}).items())
    grid_counts = ", ".join(f"{name} {count}" for name, count in coverage.get("grid_resolution_gate_counts", {}).items())
    card_html = []
    review_records = []
    for row in rows:
        files = row.get("template", {}).get("files", {})
        analysis = row.get("analysis", {})
        simulation = row.get("simulation", {})
        params = row.get("template_parameters", {})
        checkpoint = simulation_checkpoint(row)
        gate = checkpoint.get("grid_resolution_gate_pass")
        sim_status = simulation.get("status")
        artifact_count = simulation.get("artifacts", {}).get("file_count", 0)
        template_status = row.get("template", {}).get("status", "generated")
        sensor_title = f"{row.get('manufacturer', '')} {row.get('device_name', '')}".strip()
        sensor_meta = [
            ("Code", row.get("code")),
            ("Base template", row.get("source_template_id")),
            ("Pixel pitch", f"{analysis.get('pixel_pitch_um')} um"),
            ("Resolution", f"{analysis.get('resolution_mp')} MP"),
            ("Domain", f"{analysis.get('domain_pixels')} px / {analysis.get('domain_size_um', {}).get('x')} x {analysis.get('domain_size_um', {}).get('z')} um"),
            ("OCL groups", analysis.get("ocl_group_count")),
            ("CFA / split", f"{analysis.get('cfa_pattern')} / {analysis.get('split_mode')}"),
            ("Si thickness", f"{params.get('si_thickness_um')} um"),
            ("CFA thickness", f"{params.get('cfa_thickness_um')} um"),
            ("Lens height", f"{params.get('lens_height_um')} um"),
            ("DTI", f"{params.get('dti_width_um')}w x {params.get('dti_depth_um')}d um"),
            ("FDTD response", compact_number(checkpoint.get("total_response"))),
            ("Grid gate", gate_label(gate)),
            ("Recommended res.", checkpoint.get("recommended_min_resolution_px_per_um")),
            ("Artifacts", artifact_count),
        ]
        meta_html = "".join(
            f"<div><span>{escape(str(label))}</span><strong>{escape(compact_number(value))}</strong></div>"
            for label, value in sensor_meta
        )
        paths_html = "".join(
            f"<li><span>{escape(label)}</span><code>{escape(str(value))}</code></li>"
            for label, value in (
                ("STEP", files.get("step")),
                ("BREP", files.get("brep")),
                ("Analysis", f"templates/{row.get('sensor_id')}/analysis_record.json"),
                ("Smoke LUT", next((item.get("path") for item in simulation.get("artifacts", {}).get("files", []) if str(item.get("path", "")).endswith("camera_lut.json")), "-")),
            )
            if value
        )
        notes = checkpoint.get("grid_resolution_notes") or analysis.get("readiness", {}).get("reason") or ""
        card_html.append(
            f"""
    <article class="sensor-card">
      <div class="card-top">
        <div class="preview-wrap">{inline_svg(files.get('footprint_preview'))}</div>
        <div class="card-summary">
          <div class="card-header">
            <div>
              <h2>{escape(sensor_title)}</h2>
              <p>{escape(str(row.get('code')))} · {escape(str(row.get('source_template_id')))}</p>
            </div>
            <div class="badges">
              <span class="badge {status_class(template_status)}">CAD {escape(str(template_status).upper())}</span>
              <span class="badge {status_class(sim_status)}">FDTD {escape(str(sim_status))}</span>
              <span class="badge {gate_status_class(gate)}">Grid {gate_label(gate)}</span>
            </div>
          </div>
          <div class="metric-grid">{meta_html}</div>
        </div>
      </div>
      <details>
        <summary>Local artifact paths</summary>
        <ul class="paths">{paths_html}</ul>
      </details>
      <p class="note">{escape(str(notes))}</p>
    </article>"""
        )
        review_records.append(
            {
                "sensor_id": row.get("sensor_id"),
                "code": row.get("code"),
                "manufacturer": row.get("manufacturer"),
                "device_name": row.get("device_name"),
                "source_template_id": row.get("source_template_id"),
                "pixel_pitch_um": analysis.get("pixel_pitch_um"),
                "resolution_mp": analysis.get("resolution_mp"),
                "cfa_pattern": analysis.get("cfa_pattern"),
                "split_mode": analysis.get("split_mode"),
                "domain_pixels": analysis.get("domain_pixels"),
                "simulation_status": sim_status,
                "grid_resolution_gate_pass": gate,
                "total_response": checkpoint.get("total_response"),
                "artifact_count": artifact_count,
                "step": files.get("step"),
                "brep": files.get("brep"),
            }
        )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in catalog.get("limitations", []))
    review_json = json.dumps(
        {
            "schema": "reference_sensor_template_analysis_standalone_v1",
            "generated_at": catalog.get("generated_at"),
            "source_sensor_db": catalog.get("source_sensor_db"),
            "record_count": len(rows),
            "records": review_records,
        },
        ensure_ascii=False,
        indent=2,
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reference Sensor Template Analysis - Standalone Review</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #081016;
      --panel: #0d1b24;
      --panel-2: #102736;
      --line: #24495a;
      --text: #e6f2f8;
      --muted: #8fb2c1;
      --cyan: #54d7ee;
      --green: #51d88a;
      --yellow: #f5c84c;
      --red: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    header {{ border: 1px solid var(--line); background: linear-gradient(135deg, #0b1a24, #0b2d3a); border-radius: 14px; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .summary div {{ border: 1px solid var(--line); background: rgba(255,255,255,0.035); border-radius: 10px; padding: 14px; }}
    .summary span, .metric-grid span, .paths span {{ display: block; color: var(--muted); font-size: 12px; }}
    .summary strong {{ display: block; margin-top: 4px; font-size: 22px; }}
    section {{ margin-top: 18px; border: 1px solid var(--line); background: var(--panel); border-radius: 14px; padding: 18px; }}
    .pipeline {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
    .pipeline div {{ border: 1px solid var(--line); border-radius: 10px; background: var(--panel-2); padding: 12px; min-height: 92px; }}
    .pipeline b {{ color: var(--cyan); }}
    ul {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); }}
    .sensor-list {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .sensor-card {{ border: 1px solid var(--line); border-radius: 14px; background: #091923; padding: 14px; }}
    .card-top {{ display: grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 16px; align-items: stretch; }}
    .preview-wrap {{ display: grid; place-items: center; min-height: 260px; border: 1px solid #183c4d; background: #050b10; border-radius: 12px; overflow: hidden; }}
    .sensor-preview {{ width: 100%; height: auto; display: block; }}
    .preview-empty {{ color: var(--muted); }}
    .card-summary {{ min-width: 0; }}
    .card-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 12px; }}
    .card-header p {{ margin: 4px 0 0; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; border: 1px solid var(--line); padding: 5px 9px; font-size: 12px; font-weight: 700; white-space: nowrap; }}
    .badge.pass {{ border-color: rgba(81,216,138,0.65); color: var(--green); background: rgba(81,216,138,0.1); }}
    .badge.check {{ border-color: rgba(245,200,76,0.65); color: var(--yellow); background: rgba(245,200,76,0.1); }}
    .badge.fail {{ border-color: rgba(255,107,107,0.75); color: var(--red); background: rgba(255,107,107,0.1); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 8px; }}
    .metric-grid div {{ border: 1px solid #183c4d; border-radius: 9px; padding: 10px; background: #07141d; min-height: 62px; }}
    .metric-grid strong {{ display: block; margin-top: 4px; overflow-wrap: anywhere; }}
    details {{ margin-top: 12px; border-top: 1px solid #17394a; padding-top: 10px; }}
    summary {{ cursor: pointer; color: var(--cyan); font-weight: 700; }}
    .paths {{ list-style: none; padding-left: 0; display: grid; gap: 6px; }}
    code {{ color: #dbeafe; overflow-wrap: anywhere; }}
    .note {{ margin: 10px 0 0; color: #d8b86b; }}
    .json-block {{ white-space: pre-wrap; max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; background: #050b10; padding: 12px; color: #bfd7e3; font-size: 12px; }}
    @media (max-width: 980px) {{
      main {{ padding: 16px; }}
      .summary, .pipeline {{ grid-template-columns: 1fr 1fr; }}
      .card-top {{ grid-template-columns: 1fr; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 620px) {{
      .summary, .pipeline, .metric-grid {{ grid-template-columns: 1fr; }}
      .card-header {{ display: block; }}
      .badges {{ justify-content: flex-start; margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Reference Sensor Template Analysis</h1>
      <p>sensor_db 기반 주요 이미지센서 템플릿을 CAD로 생성하고, Meep smoke solver 결과와 함께 설계 리뷰용으로 묶은 단일 HTML입니다. 이 파일은 preview와 요약 데이터를 내장하므로 별도 CSS/JS/SVG 없이 열 수 있습니다.</p>
      <div class="summary">
        <div><span>Sensor records</span><strong>{len(rows)}</strong></div>
        <div><span>STEP / BREP CAD</span><strong>{step_count} / {brep_count}</strong></div>
        <div><span>FDTD smoke pass</span><strong>{smoke_pass} / {len(rows)}</strong></div>
        <div><span>Accuracy status</span><strong>Research trend</strong></div>
      </div>
      <p>Coverage: {escape(manufacturer_counts or '-')} · Templates: {escape(template_counts or '-')} · Grid gates: {escape(grid_counts or '-')}</p>
    </header>

    <section>
      <h2>Pipeline</h2>
      <div class="pipeline">
        <div><b>1. Sensor DB</b><p>TechInsights 기반 로컬 metadata에서 pitch, CFA, topology, stack proxy를 추출합니다.</p></div>
        <div><b>2. Template Map</b><p>Bayer, Quad, Nona, QPD topology를 parametric CAD template에 매핑합니다.</p></div>
        <div><b>3. CAD Source</b><p>OpenCASCADE/FreeCAD에서 볼 수 있는 STEP/BREP와 FDTD footprint를 생성합니다.</p></div>
        <div><b>4. Solver Smoke</b><p>Meep 기반 FDTD smoke run으로 artifact 생성과 파이프라인 연결성을 확인합니다.</p></div>
        <div><b>5. Analysis DB</b><p>CAD 경로, simulation 상태, gate, sensor별 설계 요약을 DB화합니다.</p></div>
      </div>
    </section>

    <section>
      <h2>Important Limits</h2>
      <ul>{limitations}</ul>
      <p class="note">현재 smoke run은 빠른 검증 조건이므로 grid gate가 CHECK로 남을 수 있습니다. 이 상태는 “solver가 돌았다”는 증거이지, 제품 정확도 LUT로 쓸 수 있다는 뜻은 아닙니다.</p>
    </section>

    <section class="sensor-list">
      {''.join(card_html)}
    </section>

    <section>
      <h2>Embedded Review Data</h2>
      <pre class="json-block">{escape(review_json)}</pre>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_readme(path: Path, catalog: dict[str, Any]) -> None:
    text = f"""# Reference Sensor Template Analysis DB

Generated at `{catalog['generated_at']}` from `{catalog['source_sensor_db']}`.

This directory contains sensor-specific CAD templates and analysis records for
major image sensors selected from `sensor_db`.

## Contents

- `analysis_catalog.json`: full machine-readable DB.
- `analysis_catalog.csv`: spreadsheet-friendly summary.
- `analysis_catalog.sqlite`: queryable SQLite analysis DB.
- `selection_index.json` / `selection_index.csv`: ranked selectable source sensor list and selected flags.
- `index.html`: local browser review table.
- `reference_sensor_template_analysis_standalone.html`: single-file review package with embedded previews.
- `templates/<sensor_id>/model.step`: CAD source for FreeCAD review.
- `templates/<sensor_id>/model.brep`: OpenCASCADE CAD source.
- `templates/<sensor_id>/geometry_import.json`: FDTD footprint import.
- `simulations/<sensor_id>/fdtd_smoke/`: optional Meep smoke artifacts when `--run-smoke` is used.

## Accuracy Status

These are reference/trend artifacts, not production-accuracy LUTs. The CAD is
parameter-derived from extracted metadata and proxy stack configs. Product
accuracy still requires measured mask/profilometry, material n,k, implant/trap
calibration, and convergence pass.

## Selection

- Record count: {catalog['record_count']}
- Source DB records: {catalog['coverage']['source_record_count']}
- Manufacturers: {catalog['coverage']['manufacturer_counts']}
- Templates: {catalog['coverage']['source_template_counts']}
- Simulation statuses: {catalog['coverage']['simulation_status_counts']}
- Grid gates: {catalog['coverage']['grid_resolution_gate_counts']}
- Smoke enabled: {catalog['selection']['run_smoke']}
- Smoke count: {catalog['selection']['smoke_count']}
- Resolution: {catalog['selection']['resolution']}
- Wavelengths: {catalog['selection']['wavelengths_nm']} nm
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-db", type=Path, default=DEFAULT_SENSOR_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sensors", type=int, default=8)
    parser.add_argument("--all-records", action="store_true", help="Generate every pitch-qualified sensor record instead of deduping by manufacturer/device.")
    parser.add_argument("--include-codes", default="", help="Comma-separated sensor codes to force include.")
    parser.add_argument("--mesh", action="store_true", help="Also generate coarse 3D Gmsh meshes.")
    parser.add_argument("--run-smoke", action="store_true", help="Run Meep smoke simulations after CAD generation.")
    parser.add_argument("--smoke-count", type=int, default=3, help="Maximum selected sensors to execute with Meep smoke.")
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--after-source-time", type=float, default=0.5)
    parser.add_argument("--wavelengths-nm", default=DEFAULT_WAVELENGTHS_NM)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--timeout-s", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_db = args.sensor_db.resolve()
    output_dir = args.output_dir.resolve()
    include_codes = {code.strip() for code in args.include_codes.split(",") if code.strip()}
    records = sensor_catalog_records(sensor_db)
    selected = select_sensor_records(records, args, include_codes)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_selection_index(output_dir, records, selected, args)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        smoke_enabled = index < args.smoke_count
        row = analyze_sensor(
            record,
            output_dir,
            mesh=args.mesh,
            run_smoke_solver=args.run_smoke,
            smoke_enabled=smoke_enabled,
            resolution=args.resolution,
            after_source_time=args.after_source_time,
            wavelengths_nm=args.wavelengths_nm,
            cases=args.cases,
            timeout_s=args.timeout_s,
        )
        rows.append(row)
        print(f"{index + 1}/{len(selected)} {row['sensor_id']} CAD={row['template']['status']} SIM={row['simulation']['status']}")
    write_catalog(output_dir, rows, args, len(records))
    print(f"Wrote analysis DB: {output_dir / 'analysis_catalog.json'}")


if __name__ == "__main__":
    main()
