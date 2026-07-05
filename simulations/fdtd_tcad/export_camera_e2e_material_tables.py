#!/usr/bin/env python3
"""Export CameraE2E/FDTD material n,k tables per sensor.

This turns the TechInsights-derived stack configs and image_sensor_db
optical_qe_db CFA proxy models into explicit CSV/JSON artifacts:

- CFA transmission/n,k proxy rows from image_sensor_db/optical_qe_db;
- FDTD material n,k rows for CFA, OCL, passivation, and silicon from each
  sensor's stack config material references.

The rows are research/proxy inputs unless their source material is marked
measured. This exporter preserves that gate instead of promoting the data to a
product-ready LUT.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_CFA_LIBRARY = ROOT / "cfa_proxy_nk_library.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_material_tables"

MATERIAL_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "material_family",
    "material_key",
    "layer_role",
    "color_channel",
    "wavelength_nm",
    "n",
    "k",
    "transmission_absorption_only",
    "thickness_um",
    "thickness_source_kind",
    "cfa_pattern",
    "cfa_pattern_source_kind",
    "cfa_channel_source",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "material_source_kind",
    "measured",
    "research_gate",
    "product_lut_gate",
    "source_path",
    "source_url",
    "notes",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "material_row_count",
    "cfa_proxy_row_count",
    "fdtd_material_row_count",
    "measured_material_count",
    "proxy_material_count",
    "cfa_proxy_applicability",
    "cfa_proxy_library_id",
    "cfa_proxy_thickness_um",
    "cfa_pattern",
    "cfa_pattern_source_kind",
    "cfa_channel_source",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "stack_geometry_measured",
    "research_gate",
    "product_lut_gate",
    "primary_blocker",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_cell(row.get(column, "")) for column in columns})


def csv_cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def value_dict_value(obj: Any, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get("value", default)
    return obj if obj not in (None, "") else default


def value_dict_source_kind(obj: Any, default: str = "") -> str:
    if isinstance(obj, dict):
        return str(obj.get("source_kind", default))
    return default


def resolve_material_path(stack_path: Path, nk_table: str) -> Path:
    candidate = Path(nk_table)
    if candidate.is_absolute():
        return candidate
    return (stack_path.parent / candidate).resolve()


def parse_nk_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = [part.strip() for part in text.split(",")]
        if len(parts) < 3:
            continue
        wavelength_um = as_float(parts[0])
        n = as_float(parts[1])
        k = as_float(parts[2])
        if all(math.isfinite(value) for value in (wavelength_um, n, k)):
            rows.append({"wavelength_nm": wavelength_um * 1000.0, "n": n, "k": k})
    return rows


def parse_refractiveindex_yml_nk(path: Path, min_nm: float = 350.0, max_nm: float = 1000.0) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    in_data_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "data: |":
            in_data_block = True
            continue
        if not in_data_block:
            continue
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            if not line.startswith(" "):
                in_data_block = False
            continue
        wavelength_um = as_float(parts[0])
        n = as_float(parts[1])
        k = as_float(parts[2])
        wavelength_nm = wavelength_um * 1000.0
        if all(math.isfinite(value) for value in (wavelength_nm, n, k)) and min_nm <= wavelength_nm <= max_nm:
            rows.append({"wavelength_nm": wavelength_nm, "n": n, "k": k})
    return rows


def material_family_for_key(material_key: str) -> tuple[str, str, str]:
    key = material_key.lower()
    if key.startswith("cfa_"):
        return "cfa_fdtd_material", "color filter", key.replace("cfa_", "")
    if key == "lens":
        return "ocl_fdtd_material", "microlens / OCL", ""
    if key == "passivation":
        return "passivation_fdtd_material", "passivation / dielectric", ""
    if key == "silicon":
        return "silicon_fdtd_material", "photodiode silicon", ""
    if key == "metal":
        return "metal_model", "optional shield/aperture metal", ""
    return "other_fdtd_material", "", ""


def rows_from_stack_material(
    *,
    sensor: dict[str, str],
    stack_path: Path,
    stack: dict[str, Any],
    material_key: str,
    material: dict[str, Any],
) -> list[dict[str, Any]]:
    family, layer_role, color_channel = material_family_for_key(material_key)
    if material_key == "metal":
        return [
            {
                **sensor_identity(sensor),
                "material_family": family,
                "material_key": material_key,
                "layer_role": layer_role,
                "color_channel": color_channel,
                "wavelength_nm": "",
                "n": "",
                "k": "",
                "transmission_absorption_only": "",
                "thickness_um": "",
                "thickness_source_kind": "",
                "material_source_kind": "model",
                "measured": False,
                "research_gate": "CHECK",
                "product_lut_gate": "FAIL",
                "source_path": "",
                "source_url": material.get("source_url", ""),
                "notes": f"Metal model={material.get('model', '')}; {material.get('usage', '')}",
            }
        ]

    nk_table = str(material.get("nk_table", ""))
    nk_path = resolve_material_path(stack_path, nk_table) if nk_table else Path("")
    if nk_path.suffix.lower() in {".yml", ".yaml"}:
        nk_rows = parse_refractiveindex_yml_nk(nk_path)
    else:
        nk_rows = parse_nk_csv(nk_path)
    measured = boolish(material.get("measured"))
    gate = "PASS" if measured else "CHECK"
    product_gate = "PASS" if measured and boolish(stack.get("calibration_status", {}).get("geometry_measured")) else "FAIL"
    thickness = thickness_for_material(stack, material_key)
    return [
        {
            **sensor_identity(sensor),
            "material_family": family,
            "material_key": material_key,
            "layer_role": layer_role,
            "color_channel": color_channel,
            "wavelength_nm": row["wavelength_nm"],
            "n": row["n"],
            "k": row["k"],
            "transmission_absorption_only": "",
            "thickness_um": thickness,
            "thickness_source_kind": "stack_geometry_proxy",
            "material_source_kind": "measured_nk" if measured else "fdtd_proxy_nk",
            "measured": measured,
            "research_gate": gate,
            "product_lut_gate": product_gate,
            "source_path": repo_rel(nk_path) if nk_path else "",
            "source_url": material.get("source_url", ""),
            "notes": material.get("source", "") or material.get("usage", ""),
        }
        for row in nk_rows
    ]


def thickness_for_material(stack: dict[str, Any], material_key: str) -> float | str:
    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    mapping = {
        "cfa_red": "cfa_thickness",
        "cfa_green": "cfa_thickness",
        "cfa_blue": "cfa_thickness",
        "passivation": "passivation_thickness",
        "lens": "lens_height",
        "silicon": "si_thickness",
    }
    key = mapping.get(material_key)
    if not key:
        return ""
    value = as_float(geometry.get(key))
    return value if math.isfinite(value) else ""


def sensor_identity(sensor: dict[str, str]) -> dict[str, str]:
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
    }


def is_monochrome_sensor(sensor: dict[str, str], optical_model: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(sensor.get("cfa_pattern", "")),
            str(value_dict_value(optical_model.get("optical", {}).get("cfa", {}).get("pattern", {}), "")),
            str(sensor.get("device_name", "")),
        ]
    ).lower()
    return "mono" in text or "clear" in text


def cfa_context(sensor: dict[str, str], optical_model: dict[str, Any], channels: dict[str, Any], channel_source: str) -> dict[str, Any]:
    optical_cfa = optical_model.get("optical", {}).get("cfa", {}) if isinstance(optical_model.get("optical", {}), dict) else {}
    pattern_info = optical_cfa.get("pattern", {}) if isinstance(optical_cfa, dict) else {}
    pattern = str(value_dict_value(pattern_info, "") or sensor.get("cfa_pattern", "") or "").strip().lower()
    pattern_source = value_dict_source_kind(pattern_info, "sensor_index" if sensor.get("cfa_pattern") else "unavailable")
    cfa_proxy = optical_model.get("cfa_proxy_nk", {}) if isinstance(optical_model, dict) else {}
    thickness_source = value_dict_source_kind(cfa_proxy.get("thickness_um", {}), "")
    channel_set = set(channels.keys())
    generic_rgb_fallback = not pattern and channel_source == "generic_rgb_library_fallback" and {"red", "green", "blue"}.issubset(channel_set)
    if generic_rgb_fallback:
        provenance_class = "GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN"
        assumption_gate = "MISSING"
    elif "mono" in pattern or "clear" in pattern or channel_set == {"clear"}:
        provenance_class = "MONO_CLEAR_PROXY"
        assumption_gate = "CHECK"
    elif boolish(cfa_proxy.get("enabled")) and {"red", "green", "blue"}.issubset(channel_set):
        provenance_class = "SENSOR_SPECIFIC_RGB_PROXY" if thickness_source in {"extracted", "derived_from_extracted_range"} else "RGB_PROXY_DEFAULT_THICKNESS"
        assumption_gate = "CHECK"
    elif boolish(cfa_proxy.get("enabled")):
        provenance_class = "CFA_PROXY_ENABLED_UNUSUAL_CHANNELS"
        assumption_gate = "CHECK"
    else:
        provenance_class = "CFA_PROXY_DISABLED"
        assumption_gate = "MISSING"
    return {
        "cfa_pattern": pattern,
        "cfa_pattern_source_kind": pattern_source,
        "cfa_channel_source": channel_source,
        "cfa_provenance_class": provenance_class,
        "cfa_assumption_gate": assumption_gate,
        "generic_rgb_fallback_detected": generic_rgb_fallback,
    }


def rows_from_cfa_proxy(sensor: dict[str, str], optical_model: dict[str, Any], cfa_library: dict[str, Any]) -> list[dict[str, Any]]:
    cfa_proxy = optical_model.get("cfa_proxy_nk", {}) if isinstance(optical_model, dict) else {}
    channels = cfa_proxy.get("channels", {}) if isinstance(cfa_proxy.get("channels"), dict) else {}
    source_path = DEFAULT_OPTICAL_QE_DB / "models" / f"{sensor.get('slug', '')}.json"
    fallback_note = ""
    channel_source = "sensor_cfa_proxy_channels"
    if not channels:
        library_channels = cfa_library.get("channels", {}) if isinstance(cfa_library.get("channels"), dict) else {}
        selected = ["clear"] if is_monochrome_sensor(sensor, optical_model) else ["red", "green", "blue"]
        channels = {channel: library_channels[channel] for channel in selected if channel in library_channels}
        source_path = DEFAULT_CFA_LIBRARY
        channel_source = "generic_clear_library_fallback" if selected == ["clear"] else "generic_rgb_library_fallback"
        fallback_note = "generic CFA library fallback because sensor-specific CFA pattern/proxy channels are unavailable; "
    context = cfa_context(sensor, optical_model, channels, channel_source)
    thickness_info = cfa_proxy.get("thickness_um", {})
    thickness = value_dict_value(thickness_info, "")
    thickness_source = value_dict_source_kind(thickness_info, "")
    rows: list[dict[str, Any]] = []
    for channel, channel_payload in channels.items():
        data = channel_payload.get("data", []) if isinstance(channel_payload, dict) else []
        for point in data:
            transmission = as_float(point.get("transmission_absorption_only"))
            if not math.isfinite(transmission):
                transmission = as_float(point.get("target_transmission"))
            rows.append(
                {
                    **sensor_identity(sensor),
                    "material_family": "cfa_transmission_proxy",
                    "material_key": f"cfa_{channel}",
                    "layer_role": "color filter transmission proxy",
                    "color_channel": channel,
                    "wavelength_nm": as_float(point.get("wavelength_nm")),
                    "n": as_float(point.get("n")),
                    "k": as_float(point.get("k")),
                    "transmission_absorption_only": transmission,
                    "thickness_um": thickness,
                    "thickness_source_kind": thickness_source,
                    "cfa_pattern": context["cfa_pattern"],
                    "cfa_pattern_source_kind": context["cfa_pattern_source_kind"],
                    "cfa_channel_source": context["cfa_channel_source"],
                    "cfa_provenance_class": context["cfa_provenance_class"],
                    "cfa_assumption_gate": context["cfa_assumption_gate"],
                    "generic_rgb_fallback_detected": context["generic_rgb_fallback_detected"],
                    "material_source_kind": cfa_proxy.get("source_kind", "inferred_proxy_nk"),
                    "measured": False,
                    "research_gate": "CHECK",
                    "product_lut_gate": "FAIL",
                    "source_path": repo_rel(source_path),
                    "source_url": "",
                    "notes": (
                        f"{fallback_note}{cfa_proxy.get('applicability', '')}; "
                        f"{channel_payload.get('method', '') if isinstance(channel_payload, dict) else ''}"
                    ),
                }
            )
    return rows


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    if not shown:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if limit is not None and len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, Any]], material_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1360px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.warn{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = manifest.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Material Tables</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Material Tables</h1>
<p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. Rows expose FDTD n,k and CFA transmission proxy inputs; product use remains gate-blocked without measured material data.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">material table status</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("material_row_count", 0))}</div><div class="muted">material rows</div></div>
<div class="card"><div class="metric warn">{html_cell(manifest.get("product_ready_count", 0))}</div><div class="muted">product-ready sensors</div></div>
</div>
<h2>Sensor Summary</h2>{html_table(summary_rows, SUMMARY_COLUMNS)}
<h2>Material Rows</h2>{html_table(material_rows, MATERIAL_COLUMNS, limit=180)}
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, manifest: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_material_tables_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_material_nk_csv"] = manifest["outputs"]["material_csv"]
    outputs["camera_e2e_material_summary_csv"] = manifest["outputs"]["summary_csv"]
    outputs["camera_e2e_material_tables_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_material_tables"] = {
        "schema": manifest["schema"],
        "validation": manifest["validation"],
        "sensor_count": manifest["sensor_count"],
        "material_row_count": manifest["material_row_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def build_material_tables(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    optical_qe_db = args.optical_qe_db.resolve()
    stack_dir = args.stack_dir.resolve()
    cfa_library = read_json(args.cfa_library.resolve())
    sensor_rows = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    if args.slugs:
        wanted = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
        sensor_rows = [row for row in sensor_rows if row.get("slug") in wanted]

    material_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_issues: list[dict[str, Any]] = []
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        stack_path = stack_dir / f"{slug}.json"
        optical_path = optical_qe_db / "models" / f"{slug}.json"
        stack = read_json(stack_path)
        optical_model = read_json(optical_path)
        rows_for_sensor: list[dict[str, Any]] = []
        cfa_rows = rows_from_cfa_proxy(sensor, optical_model, cfa_library)
        rows_for_sensor.extend(cfa_rows)
        materials = stack.get("materials", {}) if isinstance(stack.get("materials"), dict) else {}
        for material_key, material in materials.items():
            if isinstance(material, dict):
                rows_for_sensor.extend(
                    rows_from_stack_material(
                        sensor=sensor,
                        stack_path=stack_path,
                        stack=stack,
                        material_key=material_key,
                        material=material,
                    )
                )
        if not rows_for_sensor:
            validation_issues.append({"severity": "error", "code": "sensor_material_rows_missing", "slug": slug})
        if not cfa_rows:
            validation_issues.append({"severity": "warning", "code": "sensor_cfa_proxy_rows_missing", "slug": slug})
        material_rows.extend(rows_for_sensor)

        measured_count = sum(1 for row in rows_for_sensor if boolish(row.get("measured")))
        proxy_count = len(rows_for_sensor) - measured_count
        cfa_proxy = optical_model.get("cfa_proxy_nk", {}) if isinstance(optical_model, dict) else {}
        thickness_info = cfa_proxy.get("thickness_um", {}) if isinstance(cfa_proxy, dict) else {}
        proxy_channels = cfa_proxy.get("channels", {}) if isinstance(cfa_proxy.get("channels"), dict) else {}
        channel_source = "sensor_cfa_proxy_channels"
        if not proxy_channels:
            selected = ["clear"] if is_monochrome_sensor(sensor, optical_model) else ["red", "green", "blue"]
            library_channels = cfa_library.get("channels", {}) if isinstance(cfa_library.get("channels"), dict) else {}
            proxy_channels = {channel: library_channels[channel] for channel in selected if channel in library_channels}
            channel_source = "generic_clear_library_fallback" if selected == ["clear"] else "generic_rgb_library_fallback"
        cfa_info = cfa_context(sensor, optical_model, proxy_channels, channel_source)
        geometry_measured = boolish(stack.get("calibration_status", {}).get("geometry_measured")) if isinstance(stack.get("calibration_status"), dict) else False
        product_ready = bool(rows_for_sensor) and proxy_count == 0 and geometry_measured
        summary_rows.append(
            {
                **sensor_identity(sensor),
                "material_row_count": len(rows_for_sensor),
                "cfa_proxy_row_count": len(cfa_rows),
                "fdtd_material_row_count": len(rows_for_sensor) - len(cfa_rows),
                "measured_material_count": measured_count,
                "proxy_material_count": proxy_count,
                "cfa_proxy_applicability": cfa_proxy.get("applicability", "") if isinstance(cfa_proxy, dict) else "",
                "cfa_proxy_library_id": cfa_proxy.get("library_id", "") if isinstance(cfa_proxy, dict) else "",
                "cfa_proxy_thickness_um": value_dict_value(thickness_info, ""),
                "cfa_pattern": cfa_info["cfa_pattern"],
                "cfa_pattern_source_kind": cfa_info["cfa_pattern_source_kind"],
                "cfa_channel_source": cfa_info["cfa_channel_source"],
                "cfa_provenance_class": cfa_info["cfa_provenance_class"],
                "cfa_assumption_gate": cfa_info["cfa_assumption_gate"],
                "generic_rgb_fallback_detected": cfa_info["generic_rgb_fallback_detected"],
                "stack_geometry_measured": geometry_measured,
                "research_gate": "CHECK" if rows_for_sensor else "FAIL",
                "product_lut_gate": "PASS" if product_ready else "FAIL",
                "primary_blocker": "" if product_ready else "material n,k and/or stack geometry are proxy/not measured",
            }
        )

    error_count = sum(1 for issue in validation_issues if issue.get("severity") == "error")
    product_ready_count = sum(1 for row in summary_rows if row.get("product_lut_gate") == "PASS")
    status = "PRODUCT_MATERIAL_READY" if product_ready_count == len(summary_rows) and summary_rows else "RESEARCH_MATERIAL_TABLE_READY_PRODUCT_BLOCKED"
    if error_count:
        status = "FAIL"

    material_csv = output_dir / "camera_e2e_material_nk_lut.csv"
    summary_csv = output_dir / "camera_e2e_material_summary.csv"
    manifest_json = output_dir / "camera_e2e_material_tables.json"
    html_path = output_dir / "index.html"
    manifest = {
        "schema": "camera_e2e_material_tables_export_v1",
        "artifact_role": "camera_e2e_fdtd_material_nk_and_cfa_proxy_tables",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "source_optical_qe_db": repo_rel(optical_qe_db),
        "source_stack_dir": repo_rel(stack_dir),
        "sensor_count": len(summary_rows),
        "material_row_count": len(material_rows),
        "summary_row_count": len(summary_rows),
        "product_ready_count": product_ready_count,
        "gate_counts": {
            "research_gate": dict(Counter(str(row.get("research_gate", "")) for row in summary_rows)),
            "product_lut_gate": dict(Counter(str(row.get("product_lut_gate", "")) for row in summary_rows)),
            "material_family": dict(Counter(str(row.get("material_family", "")) for row in material_rows)),
        },
        "validation": {
            "schema": "camera_e2e_material_tables_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": len(validation_issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in validation_issues if issue.get("severity") == "warning"),
            "issues": validation_issues,
        },
        "policy": {
            "research": "Rows are usable as explicit proxy material inputs when gates are preserved.",
            "product": "Blocked until material rows are measured/calibrated and stack geometry is measured.",
            "source_warning": "CFA transmission proxy and FDTD n,k file proxy may come from different inferred proxy models; do not treat them as measured material constants.",
        },
        "outputs": {
            "json": repo_rel(manifest_json),
            "material_csv": repo_rel(material_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(material_csv, material_rows, MATERIAL_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, summary_rows, material_rows)
    update_package(package_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--optical-qe-db", type=Path, default=DEFAULT_OPTICAL_QE_DB)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--cfa-library", type=Path, default=DEFAULT_CFA_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="")
    return parser


def main() -> None:
    manifest = build_material_tables(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "validation": manifest["validation"],
                "sensor_count": manifest["sensor_count"],
                "material_row_count": manifest["material_row_count"],
                "product_ready_count": manifest["product_ready_count"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not manifest["validation"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
