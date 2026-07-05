#!/usr/bin/env python3
"""Build CameraE2E research prior seeds for missing electrical/readout/module data.

These seeds are deliberately not calibration data. They exist so CameraE2E can
run end-to-end research scenarios while preserving strict product gates. Every
row and model emitted by this script is marked `prior_seed_not_measured`.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_SENSOR_CATALOG = ROOT / "image_sensor_db" / "sensor_catalog.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_prior_seed_models"

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "active_si_thickness_um",
    "pixel_architecture",
    "effective_ocl_mode",
    "shutter",
    "has_hdr",
    "has_lofic",
    "has_pdaf",
    "full_well_nominal_e",
    "conversion_gain_nominal_uv_per_e",
    "read_noise_nominal_e_rms",
    "dark_current_25c_nominal_e_per_s",
    "prnu_nominal_pct_rms",
    "dsnu_nominal_e_rms",
    "adc_bit_depth_nominal",
    "black_level_nominal_dn",
    "binning_group_size",
    "rolling_line_time_us",
    "prior_gate",
    "model_json",
]


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def range_value(nominal: float, lo_factor: float, hi_factor: float, *, digits: int = 6) -> dict[str, float]:
    return {
        "min": round(nominal * lo_factor, digits),
        "nominal": round(nominal, digits),
        "max": round(nominal * hi_factor, digits),
    }


def ocl_group_size(row: dict[str, str]) -> int:
    effective = str(row.get("effective_ocl_mode", "")).lower()
    if "3x3" in effective:
        return 9
    if "2x2" in effective:
        return 4
    text = " ".join(
        str(row.get(key, "")).lower()
        for key in ("ocl_mode_guess", "cfa_pattern", "pixel_architecture", "device_name")
    )
    if "nona" in text or "3x3" in text:
        return 9
    if "quad" in text or "2x2" in text or "four_shared" in text:
        return 4
    return 1


def effective_ocl_mode(row: dict[str, str]) -> str:
    pitch = as_float(row.get("pixel_pitch_um"))
    ocl_pitch = as_float(row.get("ocl_pitch_um"))
    color_filter_pitch = as_float(row.get("color_filter_pitch_um"))
    source_pitch = ocl_pitch or color_filter_pitch
    if pitch and source_pitch:
        ratio = source_pitch / pitch
        if ratio >= 2.65:
            return "nona_3x3_from_pitch_ratio"
        if ratio >= 1.65:
            return "quad_2x2_from_pitch_ratio"
        return "bayer_1x1_from_pitch_ratio"
    text = " ".join(str(row.get(key, "")).lower() for key in ("ocl_mode_guess", "microlens_type", "cfa_pattern"))
    if "3x3" in text or "nona" in text:
        return "nona_3x3_from_text"
    if "2x2" in text or "quad" in text:
        return "quad_2x2_from_text"
    return row.get("ocl_mode_guess") or "unknown"


def fwc_prior_e(pitch_um: float, si_um: float, has_lofic: bool) -> dict[str, Any]:
    area = pitch_um * pitch_um
    si_factor = clamp(math.sqrt(max(si_um, 1.0) / 3.0), 0.75, 1.6)
    nominal = clamp(6500.0 * area * si_factor, 1200.0, 90000.0)
    output = {
        "unit": "electron",
        "per_physical_pixel": range_value(nominal, 0.45, 1.9, digits=1),
        "model": "6500 e-/um^2 area seed scaled by sqrt(active_si_thickness/3um), clipped",
    }
    if has_lofic:
        output["lofic_or_hdr_extension"] = {
            "unit": "electron",
            "per_physical_pixel": range_value(nominal * 4.0, 1.5, 2.5, digits=1),
            "model": "research prior only; LOFIC/HDR capacity requires sensor-mode calibration",
        }
    return output


def conversion_gain_prior_uv_per_e(full_well_nominal_e: float) -> dict[str, Any]:
    nominal = 800000.0 / max(full_well_nominal_e, 1.0)
    nominal = clamp(nominal, 5.0, 450.0)
    return {
        "unit": "uV/e-",
        "range": range_value(nominal, 0.45, 2.0, digits=3),
        "model": "q/C seed using 0.8 V saturation swing; not a measured source-follower/FD conversion gain",
    }


def dark_current_prior_e_per_s(pitch_um: float, si_um: float) -> dict[str, Any]:
    area = pitch_um * pitch_um
    thickness_factor = clamp(si_um / 3.0, 0.7, 2.2)
    nominal = clamp(0.45 * area * thickness_factor, 0.04, 20.0)
    return {
        "unit": "e-/s/pixel",
        "at_25c": range_value(nominal, 0.1, 15.0, digits=6),
        "temperature_model": {
            "model": "doubling_interval",
            "doubling_interval_c": {"min": 6.0, "nominal": 8.0, "max": 10.0},
            "formula": "dark_current(T)=dark_current_25c*2^((T_C-25)/doubling_interval_C)",
        },
    }


def read_noise_prior_e(pitch_um: float, shutter: str, has_hdr: bool) -> dict[str, Any]:
    nominal = 1.15 if pitch_um <= 1.0 else 1.6
    if pitch_um >= 2.0:
        nominal = 2.0
    if "global" in shutter.lower():
        nominal += 0.7
    if has_hdr:
        nominal += 0.2
    return {
        "unit": "e- rms",
        "range": range_value(clamp(nominal, 0.8, 5.0), 0.65, 2.0, digits=4),
        "components": {
            "shot_noise": "CameraE2E should compute sqrt(signal_e) directly from signal level.",
            "read_reset_sf_adc_noise": "Seeded as a lumped Gaussian rms term only.",
        },
    }


def defect_prior() -> dict[str, Any]:
    return {
        "hot_pixel_fraction": {"min": 1e-7, "nominal": 1e-5, "max": 5e-4},
        "defect_pixel_fraction": {"min": 1e-6, "nominal": 5e-5, "max": 1e-3},
        "model": "research seed distribution only; production requires sensor test maps",
    }


def adc_prior(has_hdr: bool, shutter: str) -> dict[str, Any]:
    nominal_bits = 12 if has_hdr or "global" in shutter.lower() else 10
    if has_hdr:
        bit_options = [10, 12, 14]
    else:
        bit_options = [10, 12]
    black_level_dn = 64 if nominal_bits >= 12 else 16
    return {
        "bit_depth": {"options": bit_options, "nominal": nominal_bits},
        "black_level": {
            "unit": "DN",
            "range": {"min": black_level_dn // 2, "nominal": black_level_dn, "max": black_level_dn * 2},
            "model": "fixed offset seed; replace with optical-black calibration",
        },
        "clipping_dn": (2**nominal_bits) - 1,
        "quantization": {"unit": "DN", "lsb": 1},
    }


def rolling_timing_prior(row: dict[str, str], shutter: str) -> dict[str, Any]:
    y = as_float(row.get("resolution_y"))
    if "global" in shutter.lower():
        return {"shutter": "global", "line_time_us": None, "frame_time_ms": None}
    if not y or y <= 0:
        return {"shutter": "rolling_or_unknown", "line_time_us": None, "frame_time_ms": None}
    frame_time_ms = 33.333
    line_time_us = frame_time_ms * 1000.0 / y
    return {
        "shutter": shutter or "rolling_or_unknown",
        "line_time_us": round(line_time_us, 4),
        "frame_time_ms": frame_time_ms,
        "model": "30fps full-height rolling-shutter seed",
    }


def module_alignment_prior() -> dict[str, Any]:
    return {
        "sensor_decenter_um": {"x": 0.0, "z": 0.0, "sigma_prior_um": 10.0},
        "sensor_tilt_deg": {"x": 0.0, "z": 0.0, "sigma_prior_deg": 0.05},
        "gate": "CHECK",
        "source": "zero-centered module-alignment prior; replace with module assembly/raytrace data",
    }


def vignetting_prior() -> dict[str, Any]:
    return {
        "model": "cos4_cra_seed",
        "formula": "relative_illumination=cos(total_CRA_rad)^4",
        "gate": "CHECK",
        "note": "Use only until lens raytrace or measured shading data is imported.",
    }


def electrical_crosstalk_prior(pitch_um: float, active_si_um: float, group_size: int, has_pdaf: bool) -> dict[str, Any]:
    """Conservative diffusion prior for CameraE2E research plumbing.

    This is not a TCAD replacement. It gives CameraE2E a bounded scalar prior so
    electrical crosstalk is not silently omitted while product gates stay closed.
    """
    diffusion_length = max(0.035, min(0.22, 0.055 + 0.018 * active_si_um + (0.025 if pitch_um < 0.8 else 0.0)))
    geometry_factor = min(0.18, (diffusion_length / max(pitch_um, 0.35)) ** 2)
    binning_factor = 0.72 if group_size > 1 else 1.0
    pdaf_factor = 1.15 if has_pdaf else 1.0
    crosstalk_nominal = max(0.001, min(0.12, geometry_factor * binning_factor * pdaf_factor))
    collection_eff = max(0.88, min(0.997, 1.0 - 0.45 * crosstalk_nominal))
    return {
        "status": "diffusion_length_prior_seed",
        "gate": "CHECK",
        "model": "electrical_crosstalk_fraction ~= clamp((diffusion_length_um/pitch_um)^2 * architecture_factors)",
        "diffusion_length_um": {
            "min": round(diffusion_length * 0.55, 6),
            "nominal": round(diffusion_length, 6),
            "max": round(diffusion_length * 1.8, 6),
        },
        "electrical_crosstalk_fraction": {
            "min": round(max(0.0, crosstalk_nominal * 0.35), 8),
            "nominal": round(crosstalk_nominal, 8),
            "max": round(min(0.28, crosstalk_nominal * 2.8), 8),
        },
        "collection_efficiency": {
            "min": round(max(0.75, collection_eff - 0.08), 8),
            "nominal": round(collection_eff, 8),
            "max": round(min(1.0, collection_eff + 0.025), 8),
        },
        "source": "geometry-scaled diffusion prior; replace with calibrated DEVSIM/TCAD charge collection",
        "note": "Research seed only. Product use requires calibrated implants/TG/FD/interface traps/mobility/recombination.",
    }


def wavelength_pupil_prior(has_pdaf: bool, effective_ocl_mode: str) -> dict[str, Any]:
    cra_uncertainty = 1.4 if has_pdaf else 1.0
    if "3x3" in effective_ocl_mode:
        cra_uncertainty += 0.4
    elif "2x2" in effective_ocl_mode or "quad" in effective_ocl_mode:
        cra_uncertainty += 0.25
    return {
        "status": "zero_chromatic_shift_prior_seed",
        "gate": "CHECK",
        "model": "assume no wavelength-dependent pupil shift; expose uncertainty for CameraE2E sensitivity",
        "reference_wavelength_nm": 550.0,
        "wavelength_rows": [
            {"wavelength_nm": 450.0, "relative_pupil_transmission": 0.985, "cra_shift_x_deg": 0.0, "cra_shift_z_deg": 0.0},
            {"wavelength_nm": 550.0, "relative_pupil_transmission": 1.0, "cra_shift_x_deg": 0.0, "cra_shift_z_deg": 0.0},
            {"wavelength_nm": 620.0, "relative_pupil_transmission": 0.992, "cra_shift_x_deg": 0.0, "cra_shift_z_deg": 0.0},
        ],
        "cra_shift_uncertainty_deg": round(cra_uncertainty, 3),
        "source": "module-pupil placeholder prior; replace with lens raytrace wavelength-dependent pupil table",
        "note": "Research seed only. It prevents chromatic pupil behavior from being untracked, but it is not evidence of real lens behavior.",
    }


def build_prior_model(sensor_row: dict[str, str], catalog_row: dict[str, str]) -> dict[str, Any]:
    merged = {**catalog_row, **{k: v for k, v in sensor_row.items() if v not in ("", None)}}
    merged["effective_ocl_mode"] = effective_ocl_mode(merged)
    pitch = as_float(merged.get("pixel_pitch_um"), 1.0) or 1.0
    si = as_float(merged.get("active_si_thickness_um"), 3.0) or 3.0
    shutter = merged.get("shutter", "") or ("global" if "global" in (merged.get("sensor_modality", "") or "").lower() else "rolling_unknown")
    has_hdr = boolish(merged.get("has_hdr"))
    has_lofic = boolish(merged.get("has_lofic"))
    has_pdaf = boolish(merged.get("has_pdaf"))
    group_size = ocl_group_size(merged)
    fwc = fwc_prior_e(pitch, si, has_lofic)
    fwc_nom = fwc["per_physical_pixel"]["nominal"]
    cg = conversion_gain_prior_uv_per_e(fwc_nom)
    dark = dark_current_prior_e_per_s(pitch, si)
    read_noise = read_noise_prior_e(pitch, shutter, has_hdr)
    adc = adc_prior(has_hdr, shutter)
    rolling = rolling_timing_prior(merged, shutter)
    prnu = {"unit": "% rms", "range": {"min": 0.15, "nominal": 0.6, "max": 1.8}}
    dsnu = {"unit": "e- rms", "range": {"min": 0.2, "nominal": 1.2, "max": 6.0}}
    nonlinearity = {"unit": "% full-scale", "range": {"min": 0.2, "nominal": 1.0, "max": 3.0}}
    binning = {
        "group_size": group_size,
        "signal_sum_gain": group_size,
        "shot_noise_gain": round(math.sqrt(group_size), 6),
        "read_noise_combination": "sqrt(N) if summed before normalization; depends on analog vs digital binning",
        "remosaic_risk": "higher" if group_size > 1 else "not_applicable_for_1x1",
    }
    return {
        "schema": "camera_e2e_prior_seed_model_v1",
        "artifact_role": "research_only_missing_parameter_seed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "CHECK",
        "evidence_level": "prior_seed_not_measured",
        "sensor": {
            "slug": merged.get("slug"),
            "code": merged.get("code"),
            "manufacturer": merged.get("manufacturer"),
            "device_name": merged.get("device_name"),
            "pixel_pitch_um": pitch,
            "active_si_thickness_um": si,
            "pixel_architecture": merged.get("pixel_architecture"),
            "cfa_pattern": merged.get("cfa_pattern"),
            "ocl_mode_guess": merged.get("ocl_mode_guess"),
            "effective_ocl_mode": merged.get("effective_ocl_mode"),
            "ocl_pitch_um": as_float(merged.get("ocl_pitch_um")),
            "color_filter_pitch_um": as_float(merged.get("color_filter_pitch_um")),
            "shutter": shutter,
            "has_hdr": has_hdr,
            "has_lofic": has_lofic,
            "has_pdaf": has_pdaf,
        },
        "pixel_electrical_prior": {
            "conversion_gain_uv_per_e": cg,
            "full_well_capacity_e": fwc,
            "saturation_signal_e": fwc,
            "nonlinearity": nonlinearity,
            "dark_current": dark,
            "dsnu": dsnu,
            "prnu": prnu,
            "temporal_noise": read_noise,
            "charge_collection_diffusion_electrical_crosstalk": electrical_crosstalk_prior(pitch, si, group_size, has_pdaf),
        },
        "readout_raw_prior": {
            "gain_table": {
                "analog_gain_x": [1.0, 2.0, 4.0, 8.0],
                "digital_gain_x": [1.0, 2.0, 4.0],
                "model": "generic CameraE2E gain sweep seed; replace with mode/register table",
            },
            "black_level": adc["black_level"],
            "adc": {k: v for k, v in adc.items() if k != "black_level"},
            "row_column_fpn": {
                "row_fpn_dn_rms": {"min": 0.1, "nominal": 0.5, "max": 2.0},
                "column_fpn_dn_rms": {"min": 0.05, "nominal": 0.35, "max": 1.5},
                "model": "uncalibrated Gaussian fixed-pattern seed",
            },
            "rolling_shutter": rolling,
            "defect_pixels": defect_prior(),
            "binning_remosaic": binning,
        },
        "module_coupling_prior": {
            "sensor_position_tilt_decenter": module_alignment_prior(),
            "vignetting_shading": vignetting_prior(),
            "wavelength_dependent_chief_ray_pupil": wavelength_pupil_prior(has_pdaf, str(merged.get("effective_ocl_mode", ""))),
        },
        "use_policy": {
            "camera_e2e_research": "allowed only when propagated as prior_seed_not_measured",
            "camera_e2e_product": "blocked; replace with measured/readout/calibrated module data",
            "do_not_use_for": ["product LUT sign-off", "sensor vendor comparison claims", "noise calibration"],
        },
    }


def summary_row(model: dict[str, Any], model_json: str) -> dict[str, Any]:
    sensor = model["sensor"]
    pixel = model["pixel_electrical_prior"]
    readout = model["readout_raw_prior"]
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "pixel_pitch_um": sensor.get("pixel_pitch_um", ""),
        "active_si_thickness_um": sensor.get("active_si_thickness_um", ""),
        "pixel_architecture": sensor.get("pixel_architecture", ""),
        "effective_ocl_mode": sensor.get("effective_ocl_mode", ""),
        "shutter": sensor.get("shutter", ""),
        "has_hdr": sensor.get("has_hdr", ""),
        "has_lofic": sensor.get("has_lofic", ""),
        "has_pdaf": sensor.get("has_pdaf", ""),
        "full_well_nominal_e": pixel["full_well_capacity_e"]["per_physical_pixel"]["nominal"],
        "conversion_gain_nominal_uv_per_e": pixel["conversion_gain_uv_per_e"]["range"]["nominal"],
        "read_noise_nominal_e_rms": pixel["temporal_noise"]["range"]["nominal"],
        "dark_current_25c_nominal_e_per_s": pixel["dark_current"]["at_25c"]["nominal"],
        "prnu_nominal_pct_rms": pixel["prnu"]["range"]["nominal"],
        "dsnu_nominal_e_rms": pixel["dsnu"]["range"]["nominal"],
        "adc_bit_depth_nominal": readout["adc"]["bit_depth"]["nominal"],
        "black_level_nominal_dn": readout["black_level"]["range"]["nominal"],
        "binning_group_size": readout["binning_remosaic"]["group_size"],
        "rolling_line_time_us": readout["rolling_shutter"].get("line_time_us"),
        "prior_gate": model.get("gate", "CHECK"),
        "model_json": model_json,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.check{color:#ffd36e}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Prior Seed Models</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Prior Seed Models</h1>
  <p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. These values are research-only priors and do not raise product gates.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric check">CHECK</div><div class="muted">all prior gates</div></div>
  </div>
  {html_table(rows, SUMMARY_COLUMNS)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, manifest: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_prior_seed_models_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_prior_seed_summary_csv"] = manifest["outputs"]["summary_csv"]
    outputs["camera_e2e_prior_seed_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_prior_seed_models"] = {
        "schema": manifest["schema"],
        "sensor_count": manifest["sensor_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def build_models(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    models_dir = output_dir / "models"
    sensor_index = read_csv_rows(package_dir / "camera_e2e_sensor_index.csv")
    catalog_by_code = {row.get("code", ""): row for row in read_csv_rows(args.sensor_catalog.resolve())}
    if args.slugs:
        wanted = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
        sensor_index = [row for row in sensor_index if row.get("slug") in wanted]

    summary_rows: list[dict[str, Any]] = []
    model_paths: list[str] = []
    for sensor_row in sensor_index:
        model = build_prior_model(sensor_row, catalog_by_code.get(sensor_row.get("code", ""), {}))
        model_path = models_dir / f"{sensor_row.get('slug')}.json"
        write_json(model_path, model)
        rel = repo_rel(model_path)
        model_paths.append(rel)
        summary_rows.append(summary_row(model, rel))

    summary_csv = output_dir / "camera_e2e_prior_seed_summary.csv"
    manifest_json = output_dir / "camera_e2e_prior_seed_models.json"
    html_path = output_dir / "index.html"
    manifest = {
        "schema": "camera_e2e_prior_seed_models_export_v1",
        "artifact_role": "research_only_electrical_readout_module_seed_export",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_catalog": repo_rel(args.sensor_catalog),
        "sensor_count": len(summary_rows),
        "model_json_files": model_paths,
        "policy": {
            "gate": "CHECK",
            "evidence_level": "prior_seed_not_measured",
            "product_lut_ready": False,
            "note": "Use only to run CameraE2E research scenarios when measured data is absent.",
        },
        "outputs": {
            "json": repo_rel(manifest_json),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, summary_rows)
    update_package(package_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--sensor-catalog", type=Path, default=DEFAULT_SENSOR_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="", help="Optional comma-separated slug filter.")
    return parser


def main() -> None:
    manifest = build_models(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "sensor_count": manifest["sensor_count"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
