#!/usr/bin/env python3
"""Export CameraE2E electrical/noise/readout tables from prior seed models.

The prior seed model JSONs contain the right concepts, but CameraE2E consumers
need row-based LUTs. This exporter expands the per-sensor priors into:

- electrical/noise rows over temperature, exposure, and signal level;
- analog/digital gain and ADC rows;
- binning/remosaic mode rows.

All rows remain research-only. They are explicit enough for end-to-end camera
pipeline tests, but they are not measured sensor calibration data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_PRIOR_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_prior_seed_models"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_electrical_readout_tables"

TEMPERATURES_C = (-20.0, 0.0, 25.0, 40.0, 60.0, 85.0)
EXPOSURES_S = (0.001, 0.01, 0.033333, 0.1)
SIGNAL_FRACTIONS = (0.0, 0.18, 0.5, 0.9, 1.0)

ELECTRICAL_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "temperature_c",
    "exposure_s",
    "signal_fraction",
    "full_well_e",
    "saturation_signal_e",
    "conversion_gain_uv_per_e",
    "nonlinearity_pct_fs",
    "dark_current_e_per_s",
    "dark_signal_e",
    "dsnu_e_rms",
    "prnu_pct_rms",
    "photo_signal_e",
    "shot_noise_e_rms",
    "dark_shot_noise_e_rms",
    "read_reset_sf_adc_noise_e_rms",
    "total_noise_e_rms",
    "snr_db",
    "charge_collection_electrical_crosstalk_gate",
    "charge_collection_electrical_crosstalk_model",
    "electrical_collection_efficiency_prior",
    "electrical_crosstalk_fraction_prior",
    "electrical_crosstalk_fraction_min",
    "electrical_crosstalk_fraction_max",
    "electrical_diffusion_length_um",
    "electrical_crosstalk_source",
    "evidence_level",
    "research_gate",
    "product_lut_gate",
    "source_model_json",
]

READOUT_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "analog_gain_x",
    "digital_gain_x",
    "total_gain_x",
    "adc_bit_depth",
    "black_level_dn",
    "clipping_dn",
    "quantization_lsb_dn",
    "full_well_e",
    "effective_saturation_e_after_analog_gain",
    "e_per_dn_at_unity_digital_gain",
    "dn_per_e_at_total_gain",
    "row_fpn_dn_rms",
    "column_fpn_dn_rms",
    "readout_direction",
    "shutter_type",
    "line_time_us",
    "frame_time_ms",
    "hot_pixel_fraction",
    "defect_pixel_fraction",
    "optical_black_model",
    "evidence_level",
    "research_gate",
    "product_lut_gate",
    "source_model_json",
]

BINNING_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "mode_id",
    "effective_ocl_mode",
    "cfa_pattern",
    "binning_group_size",
    "signal_sum_gain",
    "shot_noise_gain",
    "read_noise_combination",
    "relative_read_noise_after_sum",
    "output_normalization",
    "remosaic_risk",
    "optical_crosstalk_redefinition",
    "electrical_crosstalk_gate",
    "electrical_crosstalk_fraction_prior",
    "evidence_level",
    "research_gate",
    "product_lut_gate",
    "source_model_json",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "electrical_row_count",
    "readout_row_count",
    "binning_row_count",
    "full_well_e",
    "conversion_gain_uv_per_e",
    "read_noise_e_rms",
    "dark_current_25c_e_per_s",
    "dsnu_e_rms",
    "prnu_pct_rms",
    "adc_bit_depth",
    "black_level_dn",
    "binning_group_size",
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


def nested(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def nominal_range_value(data: dict[str, Any], *keys: str, default: float = math.nan) -> float:
    value = nested(data, *keys, "nominal", default=default)
    return as_float(value, default)


def range_value(data: dict[str, Any], key: str, field: str, default: float = math.nan) -> float:
    value = data.get(key, {}) if isinstance(data, dict) else {}
    if isinstance(value, dict):
        return as_float(value.get(field), default)
    return default


def dark_current_at_temperature(dark_25c: float, temp_c: float, doubling_interval_c: float) -> float:
    if not all(math.isfinite(value) for value in (dark_25c, temp_c, doubling_interval_c)) or doubling_interval_c <= 0:
        return math.nan
    return dark_25c * (2.0 ** ((temp_c - 25.0) / doubling_interval_c))


def snr_db(signal_e: float, noise_e: float) -> float:
    if signal_e <= 0 or noise_e <= 0 or not math.isfinite(signal_e) or not math.isfinite(noise_e):
        return math.nan
    return 20.0 * math.log10(signal_e / noise_e)


def sensor_identity(model: dict[str, Any]) -> dict[str, Any]:
    sensor = model.get("sensor", {}) if isinstance(model.get("sensor"), dict) else {}
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
    }


def build_electrical_rows(model: dict[str, Any], source_model_json: str) -> list[dict[str, Any]]:
    pixel = model.get("pixel_electrical_prior", {}) if isinstance(model.get("pixel_electrical_prior"), dict) else {}
    ident = sensor_identity(model)
    full_well = nominal_range_value(pixel, "full_well_capacity_e", "per_physical_pixel")
    saturation = nominal_range_value(pixel, "saturation_signal_e", "per_physical_pixel", default=full_well)
    conversion_gain = nominal_range_value(pixel, "conversion_gain_uv_per_e", "range")
    nonlinearity = nominal_range_value(pixel, "nonlinearity", "range")
    dark_25 = nominal_range_value(pixel, "dark_current", "at_25c")
    doubling = nominal_range_value(pixel, "dark_current", "temperature_model", "doubling_interval_c", default=8.0)
    dsnu = nominal_range_value(pixel, "dsnu", "range")
    prnu = nominal_range_value(pixel, "prnu", "range")
    read_noise = nominal_range_value(pixel, "temporal_noise", "range")
    collection = pixel.get("charge_collection_diffusion_electrical_crosstalk", {})
    collection_gate = str(collection.get("gate", "MISSING") if isinstance(collection, dict) else "MISSING")
    collection_model = str(collection.get("model") or collection.get("note") or collection.get("status", "")) if isinstance(collection, dict) else ""
    collection_eff = range_value(collection, "collection_efficiency", "nominal")
    electrical_xtalk = range_value(collection, "electrical_crosstalk_fraction", "nominal")
    electrical_xtalk_min = range_value(collection, "electrical_crosstalk_fraction", "min")
    electrical_xtalk_max = range_value(collection, "electrical_crosstalk_fraction", "max")
    diffusion_length = range_value(collection, "diffusion_length_um", "nominal")
    collection_source = str(collection.get("source", "")) if isinstance(collection, dict) else ""
    rows: list[dict[str, Any]] = []
    for temp_c in TEMPERATURES_C:
        dark_current = dark_current_at_temperature(dark_25, temp_c, doubling)
        for exposure_s in EXPOSURES_S:
            dark_signal = dark_current * exposure_s if math.isfinite(dark_current) else math.nan
            dark_shot = math.sqrt(max(dark_signal, 0.0)) if math.isfinite(dark_signal) else math.nan
            for signal_fraction in SIGNAL_FRACTIONS:
                photo_signal = full_well * signal_fraction if math.isfinite(full_well) else math.nan
                shot_noise = math.sqrt(max(photo_signal, 0.0)) if math.isfinite(photo_signal) else math.nan
                prnu_noise = photo_signal * prnu / 100.0 if math.isfinite(photo_signal) and math.isfinite(prnu) else 0.0
                total_noise = math.sqrt(
                    sum(
                        value * value
                        for value in (shot_noise, dark_shot, read_noise, dsnu, prnu_noise)
                        if math.isfinite(value)
                    )
                )
                rows.append(
                    {
                        **ident,
                        "temperature_c": temp_c,
                        "exposure_s": exposure_s,
                        "signal_fraction": signal_fraction,
                        "full_well_e": full_well,
                        "saturation_signal_e": saturation,
                        "conversion_gain_uv_per_e": conversion_gain,
                        "nonlinearity_pct_fs": nonlinearity,
                        "dark_current_e_per_s": dark_current,
                        "dark_signal_e": dark_signal,
                        "dsnu_e_rms": dsnu,
                        "prnu_pct_rms": prnu,
                        "photo_signal_e": photo_signal,
                        "shot_noise_e_rms": shot_noise,
                        "dark_shot_noise_e_rms": dark_shot,
                        "read_reset_sf_adc_noise_e_rms": read_noise,
                        "total_noise_e_rms": total_noise,
                        "snr_db": snr_db(photo_signal, total_noise),
                        "charge_collection_electrical_crosstalk_gate": collection_gate,
                        "charge_collection_electrical_crosstalk_model": collection_model,
                        "electrical_collection_efficiency_prior": collection_eff,
                        "electrical_crosstalk_fraction_prior": electrical_xtalk,
                        "electrical_crosstalk_fraction_min": electrical_xtalk_min,
                        "electrical_crosstalk_fraction_max": electrical_xtalk_max,
                        "electrical_diffusion_length_um": diffusion_length,
                        "electrical_crosstalk_source": collection_source,
                        "evidence_level": "prior_seed_not_measured",
                        "research_gate": "CHECK",
                        "product_lut_gate": "FAIL",
                        "source_model_json": source_model_json,
                    }
                )
    return rows


def build_readout_rows(model: dict[str, Any], source_model_json: str) -> list[dict[str, Any]]:
    pixel = model.get("pixel_electrical_prior", {}) if isinstance(model.get("pixel_electrical_prior"), dict) else {}
    readout = model.get("readout_raw_prior", {}) if isinstance(model.get("readout_raw_prior"), dict) else {}
    ident = sensor_identity(model)
    full_well = nominal_range_value(pixel, "full_well_capacity_e", "per_physical_pixel")
    gain_table = readout.get("gain_table", {}) if isinstance(readout.get("gain_table"), dict) else {}
    analog_gains = gain_table.get("analog_gain_x", [1.0])
    digital_gains = gain_table.get("digital_gain_x", [1.0])
    adc = readout.get("adc", {}) if isinstance(readout.get("adc"), dict) else {}
    bit_options = nested(adc, "bit_depth", "options", default=[nested(adc, "bit_depth", "nominal", default=10)])
    if not isinstance(bit_options, list):
        bit_options = [bit_options]
    black = readout.get("black_level", {}) if isinstance(readout.get("black_level"), dict) else {}
    black_level = nominal_range_value(black, "range")
    fpn = readout.get("row_column_fpn", {}) if isinstance(readout.get("row_column_fpn"), dict) else {}
    row_fpn = nominal_range_value(fpn, "row_fpn_dn_rms")
    col_fpn = nominal_range_value(fpn, "column_fpn_dn_rms")
    timing = readout.get("rolling_shutter", {}) if isinstance(readout.get("rolling_shutter"), dict) else {}
    defects = readout.get("defect_pixels", {}) if isinstance(readout.get("defect_pixels"), dict) else {}
    hot_fraction = nominal_range_value(defects, "hot_pixel_fraction")
    defect_fraction = nominal_range_value(defects, "defect_pixel_fraction")
    rows: list[dict[str, Any]] = []
    for bit in bit_options:
        adc_bits = int(as_float(bit, 10))
        clipping = (2**adc_bits) - 1
        usable_dn = max(clipping - black_level, 1.0) if math.isfinite(black_level) else clipping
        e_per_dn_unity = full_well / usable_dn if math.isfinite(full_well) else math.nan
        for analog_gain in analog_gains:
            analog = as_float(analog_gain, 1.0)
            effective_saturation = full_well / max(analog, 1e-9) if math.isfinite(full_well) else math.nan
            for digital_gain in digital_gains:
                digital = as_float(digital_gain, 1.0)
                total_gain = analog * digital
                dn_per_e = total_gain / e_per_dn_unity if math.isfinite(e_per_dn_unity) and e_per_dn_unity > 0 else math.nan
                rows.append(
                    {
                        **ident,
                        "analog_gain_x": analog,
                        "digital_gain_x": digital,
                        "total_gain_x": total_gain,
                        "adc_bit_depth": adc_bits,
                        "black_level_dn": black_level,
                        "clipping_dn": clipping,
                        "quantization_lsb_dn": nested(adc, "quantization", "lsb", default=1),
                        "full_well_e": full_well,
                        "effective_saturation_e_after_analog_gain": effective_saturation,
                        "e_per_dn_at_unity_digital_gain": e_per_dn_unity,
                        "dn_per_e_at_total_gain": dn_per_e,
                        "row_fpn_dn_rms": row_fpn,
                        "column_fpn_dn_rms": col_fpn,
                        "readout_direction": "unknown",
                        "shutter_type": timing.get("shutter", ""),
                        "line_time_us": timing.get("line_time_us", ""),
                        "frame_time_ms": timing.get("frame_time_ms", ""),
                        "hot_pixel_fraction": hot_fraction,
                        "defect_pixel_fraction": defect_fraction,
                        "optical_black_model": black.get("model", "fixed offset seed; replace with optical-black calibration"),
                        "evidence_level": "prior_seed_not_measured",
                        "research_gate": "CHECK",
                        "product_lut_gate": "FAIL",
                        "source_model_json": source_model_json,
                    }
                )
    return rows


def build_binning_rows(model: dict[str, Any], source_model_json: str) -> list[dict[str, Any]]:
    readout = model.get("readout_raw_prior", {}) if isinstance(model.get("readout_raw_prior"), dict) else {}
    sensor = model.get("sensor", {}) if isinstance(model.get("sensor"), dict) else {}
    binning = readout.get("binning_remosaic", {}) if isinstance(readout.get("binning_remosaic"), dict) else {}
    pixel = model.get("pixel_electrical_prior", {}) if isinstance(model.get("pixel_electrical_prior"), dict) else {}
    collection = pixel.get("charge_collection_diffusion_electrical_crosstalk", {})
    ident = sensor_identity(model)
    group_size = as_float(binning.get("group_size"), 1.0)
    signal_sum_gain = as_float(binning.get("signal_sum_gain"), max(group_size, 1.0))
    shot_noise_gain = as_float(binning.get("shot_noise_gain"), math.sqrt(max(group_size, 1.0)))
    relative_read_noise_after_sum = math.sqrt(max(group_size, 1.0))
    mode_id = f"{int(group_size)}x_sum" if group_size > 1 else "1x1"
    return [
        {
            **ident,
            "mode_id": mode_id,
            "effective_ocl_mode": sensor.get("effective_ocl_mode", ""),
            "cfa_pattern": sensor.get("cfa_pattern", ""),
            "binning_group_size": group_size,
            "signal_sum_gain": signal_sum_gain,
            "shot_noise_gain": shot_noise_gain,
            "read_noise_combination": binning.get("read_noise_combination", ""),
            "relative_read_noise_after_sum": relative_read_noise_after_sum,
            "output_normalization": "sum_then_optional_divide_by_group_size",
            "remosaic_risk": binning.get("remosaic_risk", ""),
            "optical_crosstalk_redefinition": "use optical kernel after binning-group aggregation; do not reuse single-pixel kernel as final binned-output kernel",
            "electrical_crosstalk_gate": str(collection.get("gate", "MISSING") if isinstance(collection, dict) else "MISSING"),
            "electrical_crosstalk_fraction_prior": range_value(collection, "electrical_crosstalk_fraction", "nominal"),
            "evidence_level": "prior_seed_not_measured",
            "research_gate": "CHECK",
            "product_lut_gate": "FAIL",
            "source_model_json": source_model_json,
        }
    ]


def summary_row(
    model: dict[str, Any],
    *,
    electrical_count: int,
    readout_count: int,
    binning_count: int,
    source_model_json: str,
) -> dict[str, Any]:
    pixel = model.get("pixel_electrical_prior", {}) if isinstance(model.get("pixel_electrical_prior"), dict) else {}
    readout = model.get("readout_raw_prior", {}) if isinstance(model.get("readout_raw_prior"), dict) else {}
    return {
        **sensor_identity(model),
        "electrical_row_count": electrical_count,
        "readout_row_count": readout_count,
        "binning_row_count": binning_count,
        "full_well_e": nominal_range_value(pixel, "full_well_capacity_e", "per_physical_pixel"),
        "conversion_gain_uv_per_e": nominal_range_value(pixel, "conversion_gain_uv_per_e", "range"),
        "read_noise_e_rms": nominal_range_value(pixel, "temporal_noise", "range"),
        "dark_current_25c_e_per_s": nominal_range_value(pixel, "dark_current", "at_25c"),
        "dsnu_e_rms": nominal_range_value(pixel, "dsnu", "range"),
        "prnu_pct_rms": nominal_range_value(pixel, "prnu", "range"),
        "adc_bit_depth": nested(readout, "adc", "bit_depth", "nominal", default=""),
        "black_level_dn": nominal_range_value(readout, "black_level", "range"),
        "binning_group_size": nested(readout, "binning_remosaic", "group_size", default=1),
        "research_gate": "CHECK",
        "product_lut_gate": "FAIL",
        "primary_blocker": "electrical/readout/noise values are prior seeds, not measured calibration tables",
    }


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


def write_html(
    path: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    electrical_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
    binning_rows: list[dict[str, Any]],
) -> None:
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
<title>CameraE2E Electrical / Readout Tables</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Electrical / Readout Tables</h1>
<p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. These are research prior tables for CameraE2E plumbing, not measured sensor calibration data.</p>
<div class="grid">
<div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">table status</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("electrical_row_count", 0))}</div><div class="muted">electrical rows</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("readout_row_count", 0))}</div><div class="muted">readout rows</div></div>
</div>
<h2>Sensor Summary</h2>{html_table(summary_rows, SUMMARY_COLUMNS)}
<h2>Electrical / Noise Rows</h2>{html_table(electrical_rows, ELECTRICAL_COLUMNS, limit=120)}
<h2>Readout Gain Rows</h2>{html_table(readout_rows, READOUT_COLUMNS, limit=120)}
<h2>Binning / Remosaic Rows</h2>{html_table(binning_rows, BINNING_COLUMNS)}
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
    outputs["camera_e2e_electrical_readout_tables_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_electrical_lut_csv"] = manifest["outputs"]["electrical_csv"]
    outputs["camera_e2e_readout_lut_csv"] = manifest["outputs"]["readout_csv"]
    outputs["camera_e2e_binning_lut_csv"] = manifest["outputs"]["binning_csv"]
    outputs["camera_e2e_electrical_readout_summary_csv"] = manifest["outputs"]["summary_csv"]
    outputs["camera_e2e_electrical_readout_tables_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_electrical_readout_tables"] = {
        "schema": manifest["schema"],
        "validation": manifest["validation"],
        "sensor_count": manifest["sensor_count"],
        "electrical_row_count": manifest["electrical_row_count"],
        "readout_row_count": manifest["readout_row_count"],
        "binning_row_count": manifest["binning_row_count"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def build_tables(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    prior_dir = args.prior_dir.resolve()
    output_dir = args.output_dir.resolve()
    prior_manifest = read_json(prior_dir / "camera_e2e_prior_seed_models.json")
    model_paths = [ROOT / path for path in prior_manifest.get("model_json_files", [])]
    if args.slugs:
        wanted = {slug.strip() for slug in args.slugs.split(",") if slug.strip()}
        model_paths = [path for path in model_paths if path.stem in wanted]

    electrical_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    binning_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for model_path in model_paths:
        model = read_json(model_path)
        if model.get("schema") != "camera_e2e_prior_seed_model_v1":
            issues.append({"severity": "error", "code": "prior_model_schema_invalid", "path": repo_rel(model_path)})
            continue
        source_model_json = repo_rel(model_path)
        e_rows = build_electrical_rows(model, source_model_json)
        r_rows = build_readout_rows(model, source_model_json)
        b_rows = build_binning_rows(model, source_model_json)
        if not e_rows:
            issues.append({"severity": "error", "code": "electrical_rows_missing", "slug": model_path.stem})
        if not r_rows:
            issues.append({"severity": "error", "code": "readout_rows_missing", "slug": model_path.stem})
        electrical_rows.extend(e_rows)
        readout_rows.extend(r_rows)
        binning_rows.extend(b_rows)
        summary_rows.append(summary_row(model, electrical_count=len(e_rows), readout_count=len(r_rows), binning_count=len(b_rows), source_model_json=source_model_json))

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    status = "RESEARCH_ELECTRICAL_READOUT_READY_PRODUCT_BLOCKED" if not error_count else "FAIL"
    electrical_csv = output_dir / "camera_e2e_electrical_noise_lut.csv"
    readout_csv = output_dir / "camera_e2e_readout_gain_lut.csv"
    binning_csv = output_dir / "camera_e2e_binning_remosaic_lut.csv"
    summary_csv = output_dir / "camera_e2e_electrical_readout_summary.csv"
    manifest_json = output_dir / "camera_e2e_electrical_readout_tables.json"
    html_path = output_dir / "index.html"
    manifest = {
        "schema": "camera_e2e_electrical_readout_tables_export_v1",
        "artifact_role": "camera_e2e_prior_seed_electrical_noise_readout_tables",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "source_prior_seed_models": repo_rel(prior_dir / "camera_e2e_prior_seed_models.json"),
        "sensor_count": len(summary_rows),
        "electrical_row_count": len(electrical_rows),
        "readout_row_count": len(readout_rows),
        "binning_row_count": len(binning_rows),
        "summary_row_count": len(summary_rows),
        "product_ready_count": 0,
        "gate_counts": {
            "research_gate": dict(Counter(row.get("research_gate", "") for row in summary_rows)),
            "product_lut_gate": dict(Counter(row.get("product_lut_gate", "") for row in summary_rows)),
        },
        "axes": {
            "temperature_c": list(TEMPERATURES_C),
            "exposure_s": list(EXPOSURES_S),
            "signal_fraction": list(SIGNAL_FRACTIONS),
        },
        "validation": {
            "schema": "camera_e2e_electrical_readout_tables_validation_v1",
            "pass": error_count == 0,
            "status": status,
            "issue_count": len(issues),
            "error_count": error_count,
            "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
            "issues": issues,
        },
        "policy": {
            "research": "Allowed only as prior_seed_not_measured with row-level gates preserved.",
            "product": "Blocked until measured conversion gain, FWC, dark current, noise, FPN, defects, readout mode tables, and calibration targets are imported.",
        },
        "outputs": {
            "json": repo_rel(manifest_json),
            "electrical_csv": repo_rel(electrical_csv),
            "readout_csv": repo_rel(readout_csv),
            "binning_csv": repo_rel(binning_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(electrical_csv, electrical_rows, ELECTRICAL_COLUMNS)
    write_csv(readout_csv, readout_rows, READOUT_COLUMNS)
    write_csv(binning_csv, binning_rows, BINNING_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(manifest_json, manifest)
    write_html(html_path, manifest, summary_rows, electrical_rows, readout_rows, binning_rows)
    update_package(package_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="")
    return parser


def main() -> None:
    manifest = build_tables(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "validation": manifest["validation"],
                "sensor_count": manifest["sensor_count"],
                "electrical_row_count": manifest["electrical_row_count"],
                "readout_row_count": manifest["readout_row_count"],
                "binning_row_count": manifest["binning_row_count"],
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
