#!/usr/bin/env python3
"""Export CameraE2E color-response and rough RGB->XYZ seed artifacts.

This uses the optical_qe_db CFA proxy library plus available runtime response
anchors. It is intended to give CameraE2E/ISP code a consistent spectral
sensitivity table. It is not a measured color calibration and does not raise
product gates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import export_camera_e2e_material_tables as material_export


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_CFA_LIBRARY = ROOT / "cfa_proxy_nk_library.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_color_response"

SPECTRAL_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "cfa_pattern",
    "optical_cfa_pattern_source_kind",
    "cfa_proxy_applicability",
    "cfa_channel_source",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "effective_ocl_mode",
    "wavelength_nm",
    "color_channel",
    "cfa_transmission_proxy",
    "si_simple_absorption_fraction",
    "cfa_times_si_simple_fraction",
    "spectral_response",
    "spectral_response_normalized",
    "runtime_anchor_wavelength_nm",
    "runtime_anchor_response",
    "scale_factor",
    "spectral_response_basis",
    "evidence_level",
    "evidence_gate",
    "source",
]

MATRIX_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "cfa_pattern",
    "optical_cfa_pattern_source_kind",
    "cfa_proxy_applicability",
    "cfa_provenance_class",
    "cfa_assumption_gate",
    "generic_rgb_fallback_detected",
    "color_accuracy_gate",
    "matrix_role",
    "applicability",
    "gate",
    "m00",
    "m01",
    "m02",
    "m10",
    "m11",
    "m12",
    "m20",
    "m21",
    "m22",
    "condition",
    "source",
    "notes",
]


# 25 nm approximation of CIE 1931 2-deg color matching functions. This is a
# seed table for research CCM smoke tests, not color-calibration reference data.
CIE_1931_2DEG_25NM_APPROX = {
    400.0: (0.01431, 0.000396, 0.06785),
    425.0: (0.20915, 0.0078, 1.0156),
    450.0: (0.3362, 0.038, 1.7721),
    475.0: (0.1455, 0.115, 1.0503),
    500.0: (0.0049, 0.323, 0.272),
    525.0: (0.1144, 0.786, 0.0602),
    550.0: (0.4334, 0.995, 0.00875),
    575.0: (0.8392, 0.911, 0.001875),
    600.0: (1.0622, 0.631, 0.0008),
    625.0: (0.7484, 0.323, 0.00012),
    650.0: (0.2835, 0.107, 0.0),
    675.0: (0.0671, 0.0245, 0.0),
    700.0: (0.0114, 0.0041, 0.0),
}


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
        rows = list(csv.DictReader(handle))
    return [row for row in rows if next(iter(row.values()), "") != next(iter(row.keys()), "")]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


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


def value_dict_value(payload: Any, default: Any = "") -> Any:
    if isinstance(payload, dict):
        value = payload.get("value", default)
        return default if value is None else value
    return default if payload is None else payload


def value_dict_source_kind(payload: Any, default: str = "") -> str:
    return str(payload.get("source_kind", default)) if isinstance(payload, dict) else default


def cfa_context(
    sensor: dict[str, str],
    optical_model: dict[str, Any],
    selected_channels: list[str],
    cfa_channel_source: str,
) -> dict[str, Any]:
    optical_cfa = optical_model.get("optical", {}).get("cfa", {}) if isinstance(optical_model.get("optical", {}), dict) else {}
    pattern_info = optical_cfa.get("pattern", {}) if isinstance(optical_cfa, dict) else {}
    pattern = str(value_dict_value(pattern_info, "") or sensor.get("cfa_pattern", "") or "").strip().lower()
    pattern_source = value_dict_source_kind(pattern_info, "sensor_index" if sensor.get("cfa_pattern") else "unavailable")
    proxy = optical_model.get("cfa_proxy_nk", {}) if isinstance(optical_model, dict) else {}
    thickness_source = value_dict_source_kind(proxy.get("thickness_um", {}), "")
    selected_set = set(selected_channels)
    generic_rgb_fallback = not pattern and cfa_channel_source == "generic_cfa_proxy_library_fallback" and {"red", "green", "blue"}.issubset(selected_set)
    if generic_rgb_fallback:
        provenance_class = "GENERIC_RGB_FALLBACK_UNKNOWN_PATTERN"
        assumption_gate = "MISSING"
    elif "mono" in pattern or "clear" in pattern or selected_set == {"clear"}:
        provenance_class = "MONO_CLEAR_PROXY"
        assumption_gate = "CHECK"
    elif boolish(proxy.get("enabled")) and {"red", "green", "blue"}.issubset(selected_set):
        provenance_class = "SENSOR_SPECIFIC_RGB_PROXY" if thickness_source in {"extracted", "derived_from_extracted_range"} else "RGB_PROXY_DEFAULT_THICKNESS"
        assumption_gate = "CHECK"
    elif boolish(proxy.get("enabled")):
        provenance_class = "CFA_PROXY_ENABLED_UNUSUAL_CHANNELS"
        assumption_gate = "CHECK"
    else:
        provenance_class = "CFA_PROXY_DISABLED"
        assumption_gate = "MISSING"
    return {
        "optical_cfa_pattern": pattern,
        "optical_cfa_pattern_source_kind": pattern_source,
        "cfa_proxy_applicability": proxy.get("applicability", ""),
        "cfa_channel_source": cfa_channel_source,
        "cfa_provenance_class": provenance_class,
        "cfa_assumption_gate": assumption_gate,
        "generic_rgb_fallback_detected": generic_rgb_fallback,
    }


def channel_rows_from_model(optical_model: dict[str, Any], library: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], str, str, str]:
    cfa_proxy = optical_model.get("cfa_proxy_nk", {}) if optical_model else {}
    channels = cfa_proxy.get("channels", {}) if isinstance(cfa_proxy.get("channels"), dict) else {}
    if channels:
        return channels, "sensor_cfa_proxy_scaled", "CHECK", "sensor_cfa_proxy_channels"
    lib_channels = library.get("channels", {}) if isinstance(library.get("channels"), dict) else {}
    return lib_channels, "generic_cfa_proxy_library_fallback", "CHECK", "generic_cfa_proxy_library_fallback"


def channel_transmission(point: dict[str, Any]) -> float:
    if "transmission_absorption_only" in point:
        return as_float(point.get("transmission_absorption_only"), 0.0)
    return as_float(point.get("target_transmission"), 0.0)


def nearest_material(rows: list[dict[str, Any]], family: str, wavelength_nm: float) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("material_family") == family]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs(as_float(row.get("wavelength_nm"), wavelength_nm) - wavelength_nm))


def simple_si_absorption(si_k: float, wavelength_nm: float, thickness_um: float) -> float:
    if not (math.isfinite(si_k) and math.isfinite(wavelength_nm) and math.isfinite(thickness_um)):
        return math.nan
    if wavelength_nm <= 0 or thickness_um <= 0:
        return math.nan
    alpha_per_um = 4.0 * math.pi * max(0.0, si_k) / (wavelength_nm / 1000.0)
    return max(0.0, min(1.0, 1.0 - math.exp(-alpha_per_um * thickness_um)))


def point_cfa_si_response(point: dict[str, Any], material_rows: list[dict[str, Any]]) -> tuple[float, float]:
    wave = as_float(point.get("wavelength_nm"))
    transmission = channel_transmission(point)
    si = nearest_material(material_rows, "silicon_fdtd_material", wave)
    si_abs = simple_si_absorption(as_float(si.get("k")), wave, as_float(si.get("thickness_um")))
    if math.isfinite(si_abs):
        return si_abs, max(0.0, transmission * si_abs)
    return math.nan, transmission


def material_rows_for_sensor(sensor: dict[str, str], optical_model: dict[str, Any], cfa_library: dict[str, Any]) -> list[dict[str, Any]]:
    slug = sensor.get("slug", "")
    if not slug:
        return []
    stack_path = DEFAULT_STACK_DIR / f"{slug}.json"
    stack = read_json(stack_path)
    rows = material_export.rows_from_cfa_proxy(sensor, optical_model, cfa_library)
    materials = stack.get("materials", {}) if isinstance(stack.get("materials"), dict) else {}
    silicon = materials.get("silicon", {})
    if isinstance(silicon, dict):
        rows.extend(
            material_export.rows_from_stack_material(
                sensor=sensor,
                stack_path=stack_path,
                stack=stack,
                material_key="silicon",
                material=silicon,
            )
        )
    return rows


def group_runtime_anchors(runtime_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    anchors: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runtime_rows:
        if abs(as_float(row.get("field_x_norm"), 999.0)) <= 1e-12 and abs(as_float(row.get("field_z_norm"), 999.0)) <= 1e-12:
            anchors[(row.get("slug", ""), row.get("color_channel", ""))].append(row)
    return anchors


def group_runtime_anchors_by_slug(runtime_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    anchors: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runtime_rows:
        if abs(as_float(row.get("field_x_norm"), 999.0)) <= 1e-12 and abs(as_float(row.get("field_z_norm"), 999.0)) <= 1e-12:
            anchors[row.get("slug", "")].append(row)
    return anchors


def nearest_anchor(anchors: list[dict[str, str]], wavelength: float) -> dict[str, str] | None:
    if not anchors:
        return None
    return min(anchors, key=lambda row: abs(as_float(row.get("wavelength_nm"), 0.0) - wavelength))


def scale_for_channel(channel_points: list[dict[str, Any]], anchors: list[dict[str, str]], material_rows: list[dict[str, Any]]) -> float:
    if not anchors:
        return 1.0
    ratios: list[float] = []
    for anchor in anchors:
        wave = as_float(anchor.get("wavelength_nm"))
        response = as_float(anchor.get("response_nominal"))
        if not math.isfinite(wave) or not math.isfinite(response) or not channel_points:
            continue
        point = min(channel_points, key=lambda item: abs(as_float(item.get("wavelength_nm"), wave) - wave))
        basis = point_cfa_si_response(point, material_rows)[1]
        if basis > 1e-12:
            ratios.append(response / basis)
    return sum(ratios) / len(ratios) if ratios else 1.0


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def inverse_3x3(m: list[list[float]]) -> list[list[float]] | None:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    inv = [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
    return inv


def least_squares_rgb_to_xyz(rows: list[dict[str, Any]]) -> list[list[float]] | None:
    by_wave: dict[float, dict[str, float]] = defaultdict(dict)
    for row in rows:
        wave = as_float(row.get("wavelength_nm"))
        if wave in CIE_1931_2DEG_25NM_APPROX:
            by_wave[wave][str(row.get("color_channel"))] = as_float(row.get("spectral_response"), 0.0)
    s: list[list[float]] = []
    x: list[list[float]] = []
    for wave in sorted(by_wave):
        channels = by_wave[wave]
        if all(channel in channels for channel in ("red", "green", "blue")):
            s.append([channels["red"], channels["green"], channels["blue"]])
            x.append(list(CIE_1931_2DEG_25NM_APPROX[wave]))
    if len(s) < 3:
        return None
    st = transpose(s)
    sts = matmul(st, s)
    inv = inverse_3x3(sts)
    if inv is None:
        return None
    stx = matmul(st, x)
    return matmul(inv, stx)


def build_color_response(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    optical_db = args.optical_qe_db.resolve()
    cfa_library = read_json(args.cfa_library.resolve())
    sensor_rows = read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv")
    if args.slugs:
        wanted = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
        sensor_rows = [row for row in sensor_rows if row.get("slug") in wanted]
    optical_summary = {row.get("code", ""): row for row in read_csv_rows(optical_db / "optical_qe_summary.csv")}
    runtime_rows = read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv")
    runtime_anchors = group_runtime_anchors(runtime_rows)
    runtime_anchors_by_slug = group_runtime_anchors_by_slug(runtime_rows)

    spectral_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    per_sensor_json: list[str] = []
    models_dir = output_dir / "models"
    for sensor in sensor_rows:
        slug = sensor.get("slug", "")
        code = sensor.get("code", "")
        optical_row = optical_summary.get(code, {})
        optical_model = read_json(optical_db / optical_row.get("model_json", "")) if optical_row.get("model_json") else {}
        channels, evidence_level, evidence_gate, cfa_channel_source = channel_rows_from_model(optical_model, cfa_library)
        material_rows = material_rows_for_sensor(sensor, optical_model, cfa_library)
        sensor_spectral: list[dict[str, Any]] = []
        channel_max: dict[str, float] = {}
        cfa_pattern = str(sensor.get("cfa_pattern", "")).lower()
        selected_channels = ["clear"] if "monochrome" in cfa_pattern and "clear" in channels else ["red", "green", "blue"]
        cfa_info = cfa_context(sensor, optical_model, selected_channels, cfa_channel_source)
        if cfa_info["cfa_assumption_gate"] == "MISSING":
            evidence_level = "generic_rgb_fallback_unknown_cfa_pattern" if cfa_info["generic_rgb_fallback_detected"] else evidence_level
        for channel in selected_channels:
            points = channels.get(channel, {}).get("data", []) if isinstance(channels.get(channel), dict) else []
            anchors_for_scale = runtime_anchors_by_slug.get(slug, []) if channel == "clear" else runtime_anchors.get((slug, channel), [])
            scale = scale_for_channel(points, anchors_for_scale, material_rows)
            for point in points:
                wave = as_float(point.get("wavelength_nm"))
                transmission = channel_transmission(point)
                si_abs, cfa_si_response = point_cfa_si_response(point, material_rows)
                anchor = nearest_anchor(anchors_for_scale, wave)
                anchor_response = as_float(anchor.get("response_nominal")) if anchor else math.nan
                anchor_wave = as_float(anchor.get("wavelength_nm")) if anchor else math.nan
                response = cfa_si_response * scale
                row = {
                    "slug": slug,
                    "code": code,
                    "manufacturer": sensor.get("manufacturer", ""),
                    "device_name": sensor.get("device_name", ""),
                    "cfa_pattern": sensor.get("cfa_pattern", ""),
                    "optical_cfa_pattern_source_kind": cfa_info["optical_cfa_pattern_source_kind"],
                    "cfa_proxy_applicability": cfa_info["cfa_proxy_applicability"],
                    "cfa_channel_source": cfa_info["cfa_channel_source"],
                    "cfa_provenance_class": cfa_info["cfa_provenance_class"],
                    "cfa_assumption_gate": cfa_info["cfa_assumption_gate"],
                    "generic_rgb_fallback_detected": cfa_info["generic_rgb_fallback_detected"],
                    "effective_ocl_mode": sensor.get("effective_ocl_mode", ""),
                    "wavelength_nm": wave,
                    "color_channel": channel,
                    "cfa_transmission_proxy": transmission,
                    "si_simple_absorption_fraction": si_abs if math.isfinite(si_abs) else "",
                    "cfa_times_si_simple_fraction": cfa_si_response,
                    "spectral_response": response,
                    "spectral_response_normalized": 0.0,
                    "runtime_anchor_wavelength_nm": anchor_wave,
                    "runtime_anchor_response": anchor_response,
                    "scale_factor": scale,
                    "spectral_response_basis": "cfa_transmission_proxy_x_simple_si_absorption_scaled_to_runtime_anchor",
                    "evidence_level": evidence_level,
                    "evidence_gate": evidence_gate,
                    "source": optical_row.get("model_json", "") or repo_rel(args.cfa_library),
                }
                sensor_spectral.append(row)
                channel_max[channel] = max(channel_max.get(channel, 0.0), response)
        for row in sensor_spectral:
            peak = channel_max.get(str(row.get("color_channel")), 0.0)
            row["spectral_response_normalized"] = row["spectral_response"] / peak if peak > 0 else 0.0
        spectral_rows.extend(sensor_spectral)
        if "monochrome" in cfa_pattern:
            matrix = None
            matrix_gate = "MISSING"
            applicability = "not_applicable_monochrome_sensor"
            notes = "RGB->XYZ CCM is not physically meaningful for monochrome sensor metadata."
        else:
            matrix = least_squares_rgb_to_xyz(sensor_spectral)
            matrix_gate = "CHECK" if matrix else "MISSING"
            if cfa_info["generic_rgb_fallback_detected"]:
                applicability = "rgb_proxy_seed_unknown_cfa_pattern"
                notes = "Approximate equal-energy RGB->XYZ seed from generic RGB fallback plus simple Si absorption; CFA pattern is unknown, so this is plumbing only."
            else:
                applicability = "rgb_proxy_seed"
                notes = "Approximate equal-energy RGB->XYZ seed from CFA proxy x simple Si absorption and approximate CIE 1931 25nm table; not color calibration."
        matrix_row = {
            "slug": slug,
            "code": code,
            "manufacturer": sensor.get("manufacturer", ""),
            "device_name": sensor.get("device_name", ""),
            "cfa_pattern": sensor.get("cfa_pattern", ""),
            "optical_cfa_pattern_source_kind": cfa_info["optical_cfa_pattern_source_kind"],
            "cfa_proxy_applicability": cfa_info["cfa_proxy_applicability"],
            "cfa_provenance_class": cfa_info["cfa_provenance_class"],
            "cfa_assumption_gate": cfa_info["cfa_assumption_gate"],
            "generic_rgb_fallback_detected": cfa_info["generic_rgb_fallback_detected"],
            "color_accuracy_gate": "MISSING",
            "matrix_role": "rgb_to_xyz_equal_energy_seed",
            "applicability": applicability,
            "gate": matrix_gate,
            "m00": matrix[0][0] if matrix else "",
            "m01": matrix[0][1] if matrix else "",
            "m02": matrix[0][2] if matrix else "",
            "m10": matrix[1][0] if matrix else "",
            "m11": matrix[1][1] if matrix else "",
            "m12": matrix[1][2] if matrix else "",
            "m20": matrix[2][0] if matrix else "",
            "m21": matrix[2][1] if matrix else "",
            "m22": matrix[2][2] if matrix else "",
            "condition": "equal_energy_400_700nm_25nm",
            "source": "CFA proxy n,k/transmission x simple Si absorption plus runtime center anchors where available",
            "notes": notes,
        }
        matrix_rows.append(matrix_row)
        model = {
            "schema": "camera_e2e_color_response_model_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sensor": {key: sensor.get(key, "") for key in ("slug", "code", "manufacturer", "device_name", "cfa_pattern", "effective_ocl_mode")},
            "spectral_rows": sensor_spectral,
            "rgb_to_xyz_matrix": matrix_row,
            "policy": {
                "evidence_level": "proxy_cfa_si_or_runtime_anchor_scaled_color_response",
                "product_lut_ready": False,
                "note": "Measured spectral QE/color calibration is required before production use.",
            },
        }
        model_path = models_dir / f"{slug}.json"
        write_json(model_path, model)
        per_sensor_json.append(repo_rel(model_path))

    spectral_csv = output_dir / "camera_e2e_spectral_response.csv"
    matrix_csv = output_dir / "camera_e2e_color_matrix_seed.csv"
    manifest_json = output_dir / "camera_e2e_color_response.json"
    html_path = output_dir / "index.html"
    manifest = {
        "schema": "camera_e2e_color_response_export_v1",
        "artifact_role": "camera_e2e_color_spectral_response_seed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor_count": len(sensor_rows),
        "spectral_row_count": len(spectral_rows),
        "matrix_row_count": len(matrix_rows),
        "product_lut_ready": False,
        "model_json_files": per_sensor_json,
        "outputs": {
            "json": repo_rel(manifest_json),
            "spectral_csv": repo_rel(spectral_csv),
            "matrix_csv": repo_rel(matrix_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(spectral_csv, spectral_rows, SPECTRAL_COLUMNS)
    write_csv(matrix_csv, matrix_rows, MATRIX_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, matrix_rows, spectral_rows)
    update_package(package_dir, manifest)
    return manifest


def html_cell(value: Any) -> str:
    if isinstance(value, float):
        return html.escape(f"{value:.6g}" if math.isfinite(value) else str(value))
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(col)}</th>" for col in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, manifest: dict[str, Any], matrix_rows: list[dict[str, Any]], spectral_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CameraE2E Color Response</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Color Response</h1>
<p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. Proxy/runtime-anchor color response seed, not measured color calibration.</p>
<div class="grid">
<div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("spectral_row_count", 0))}</div><div class="muted">spectral rows</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("matrix_row_count", 0))}</div><div class="muted">matrix rows</div></div>
</div>
<h2>RGB to XYZ Seed Matrix</h2>
{html_table(matrix_rows, MATRIX_COLUMNS, limit=80)}
<h2>Spectral Response Rows</h2>
{html_table(spectral_rows, SPECTRAL_COLUMNS, limit=160)}
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
    outputs["camera_e2e_color_response_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_spectral_response_csv"] = manifest["outputs"]["spectral_csv"]
    outputs["camera_e2e_color_matrix_seed_csv"] = manifest["outputs"]["matrix_csv"]
    outputs["camera_e2e_color_response_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_color_response"] = {
        "schema": manifest["schema"],
        "sensor_count": manifest["sensor_count"],
        "spectral_row_count": manifest["spectral_row_count"],
        "matrix_row_count": manifest["matrix_row_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--optical-qe-db", type=Path, default=DEFAULT_OPTICAL_QE_DB)
    parser.add_argument("--cfa-library", type=Path, default=DEFAULT_CFA_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="", help="Optional comma-separated slug filter.")
    return parser


def main() -> None:
    manifest = build_color_response(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "sensor_count": manifest["sensor_count"],
                "spectral_row_count": manifest["spectral_row_count"],
                "matrix_row_count": manifest["matrix_row_count"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
