#!/usr/bin/env python3
"""Run a CameraE2E scalar probe from the runtime bundle and prior seed models.

This is a consumer-style smoke test for the generated sensor package. It asks:
"for this sensor, field, wavelength, and incident photon count, what would the
research CameraE2E signal/noise/crosstalk numbers look like?"

The calculation is intentionally scalar and transparent. It is not an ISP, not
a full image simulator, and not a product calibration path. Product mode still
fails closed through the runtime bundle gates.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from query_camera_e2e_runtime_bundle import (
    DEFAULT_BUNDLE_JSON,
    KERNEL_COLUMNS as QUERY_KERNEL_COLUMNS,
    QUERY_COLUMNS,
    boolish,
    finite_float,
    load_bundle,
    query_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_sensor_probe"

PROBE_COLUMNS = [
    "probe_id",
    "runtime_query_id",
    "mode",
    "query_allowed",
    "query_gate",
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_x_norm",
    "field_z_norm",
    "wavelength_nm",
    "color_channel",
    "incident_photons_per_pixel",
    "exposure_s",
    "temperature_c",
    "response_nominal",
    "signal_e",
    "direct_signal_e",
    "neighbor_leakage_e",
    "dark_signal_e",
    "full_well_e",
    "clipped_signal_e",
    "shot_noise_e_rms",
    "read_noise_e_rms",
    "dark_noise_e_rms",
    "dsnu_e_rms",
    "prnu_e_rms",
    "total_noise_e_rms",
    "snr_db",
    "conversion_gain_uv_per_e",
    "adc_bit_depth",
    "black_level_dn",
    "raw_dn",
    "raw_dn_clipped",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "field_evidence_gate",
    "crosstalk_evidence_gate",
    "combined_evidence_gate",
    "production_lut_gate",
    "confidence_class",
    "prior_seed_gate",
    "product_lut_ready",
]

KERNEL_PROBE_COLUMNS = [
    "probe_id",
    "runtime_query_id",
    "slug",
    "wavelength_nm",
    "color_channel",
    "dx",
    "dz",
    "response_fraction",
    "signal_e",
    "kernel_signal_e",
    "color_relation",
    "evidence_gate",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "wavelength_nm",
    "color_channel",
    "probe_count",
    "allowed_count",
    "query_gate",
    "field_evidence_gate",
    "crosstalk_evidence_gate",
    "production_lut_gate",
    "mean_signal_e",
    "min_signal_e",
    "max_signal_e",
    "center_signal_e",
    "min_edge_signal_e",
    "edge_to_center_signal_ratio",
    "mean_total_noise_e_rms",
    "min_snr_db",
    "max_snr_db",
    "mean_raw_dn",
    "max_output_crosstalk_fraction",
    "max_strongest_neighbor_fraction",
    "prior_seed_gate",
    "product_lut_ready",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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
            writer.writerow({column: row.get(column, "") for column in columns})


def range_nominal(payload: dict[str, Any], default: float = 0.0) -> float:
    if not isinstance(payload, dict):
        return default
    if "nominal" in payload:
        return finite_float(payload.get("nominal"), default)
    if isinstance(payload.get("range"), dict):
        return finite_float(payload["range"].get("nominal"), default)
    return default


def dark_current_at_temperature(prior: dict[str, Any], temperature_c: float) -> float:
    dark = prior.get("dark_current", {}) if isinstance(prior, dict) else {}
    at_25 = range_nominal(dark.get("at_25c", {}), 0.0)
    temp = dark.get("temperature_model", {}) if isinstance(dark.get("temperature_model"), dict) else {}
    interval = range_nominal(temp.get("doubling_interval_c", {}), 8.0)
    if interval <= 0:
        interval = 8.0
    return at_25 * (2.0 ** ((temperature_c - 25.0) / interval))


def load_prior_model(package_dir: Path, slug: str) -> dict[str, Any]:
    path = package_dir / "camera_e2e_prior_seed_models" / "models" / f"{slug}.json"
    return read_json(path)


def clipping_dn(adc: dict[str, Any]) -> float:
    value = finite_float(adc.get("clipping_dn"), math.nan)
    if math.isfinite(value):
        return value
    bits = finite_float(adc.get("bit_depth", {}).get("nominal"), 12.0)
    return (2 ** int(bits)) - 1


def probe_rows(
    *,
    package_dir: Path,
    runtime_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    incident_photons: float,
    exposure_s: float,
    temperature_c: float,
    analog_gain: float,
    digital_gain: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kernels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kernel_rows:
        kernels_by_query[str(row.get("runtime_query_id", ""))].append(row)

    output: list[dict[str, Any]] = []
    kernel_output: list[dict[str, Any]] = []
    prior_cache: dict[str, dict[str, Any]] = {}
    for row in runtime_rows:
        slug = str(row.get("slug", ""))
        prior = prior_cache.setdefault(slug, load_prior_model(package_dir, slug))
        pixel = prior.get("pixel_electrical_prior", {})
        readout = prior.get("readout_raw_prior", {})
        fwc = range_nominal(pixel.get("full_well_capacity_e", {}).get("per_physical_pixel", {}), 1.0)
        cg = range_nominal(pixel.get("conversion_gain_uv_per_e", {}), 0.0)
        read_noise = range_nominal(pixel.get("temporal_noise", {}), 0.0)
        dsnu = range_nominal(pixel.get("dsnu", {}), 0.0)
        prnu_pct = range_nominal(pixel.get("prnu", {}), 0.0)
        dark_current = dark_current_at_temperature(pixel, temperature_c)
        dark_signal = max(0.0, dark_current * exposure_s)
        response = finite_float(row.get("response_nominal"), 0.0)
        direct_response = finite_float(row.get("direct_signal_response"), response)
        leakage_response = finite_float(row.get("neighbor_leakage_response"), 0.0)
        signal = max(0.0, incident_photons * response)
        direct_signal = max(0.0, incident_photons * direct_response)
        neighbor_leakage = max(0.0, incident_photons * leakage_response)
        unclipped = signal + dark_signal
        clipped_signal = min(max(0.0, unclipped), max(fwc, 1.0))
        shot_noise = math.sqrt(max(signal, 0.0))
        dark_noise = math.sqrt(max(dark_signal, 0.0))
        prnu_noise = abs(signal) * prnu_pct / 100.0
        total_noise = math.sqrt(shot_noise**2 + read_noise**2 + dark_noise**2 + dsnu**2 + prnu_noise**2)
        snr_db = 20.0 * math.log10(signal / total_noise) if signal > 0 and total_noise > 0 else -math.inf
        adc = readout.get("adc", {})
        black = range_nominal(readout.get("black_level", {}), 0.0)
        clip_dn = clipping_dn(adc)
        bit_depth = finite_float(adc.get("bit_depth", {}).get("nominal"), 12.0)
        scaled = black + (clipped_signal / max(fwc, 1.0)) * max(0.0, clip_dn - black) * analog_gain * digital_gain
        raw_dn_clipped = min(max(0.0, scaled), clip_dn)
        qid = str(row.get("runtime_query_id", ""))
        probe_id = f"{qid}_photons_{incident_photons:g}_exp_{exposure_s:g}".replace(".", "p").replace("-", "m")
        out = {
            "probe_id": probe_id,
            "runtime_query_id": qid,
            "mode": row.get("mode", ""),
            "query_allowed": row.get("query_allowed", ""),
            "query_gate": row.get("query_gate", ""),
            "slug": slug,
            "code": row.get("code", ""),
            "manufacturer": row.get("manufacturer", ""),
            "device_name": row.get("device_name", ""),
            "field_x_norm": row.get("field_x_norm", ""),
            "field_z_norm": row.get("field_z_norm", ""),
            "wavelength_nm": row.get("wavelength_nm", ""),
            "color_channel": row.get("color_channel", ""),
            "incident_photons_per_pixel": incident_photons,
            "exposure_s": exposure_s,
            "temperature_c": temperature_c,
            "response_nominal": response,
            "signal_e": signal,
            "direct_signal_e": direct_signal,
            "neighbor_leakage_e": neighbor_leakage,
            "dark_signal_e": dark_signal,
            "full_well_e": fwc,
            "clipped_signal_e": clipped_signal,
            "shot_noise_e_rms": shot_noise,
            "read_noise_e_rms": read_noise,
            "dark_noise_e_rms": dark_noise,
            "dsnu_e_rms": dsnu,
            "prnu_e_rms": prnu_noise,
            "total_noise_e_rms": total_noise,
            "snr_db": snr_db,
            "conversion_gain_uv_per_e": cg,
            "adc_bit_depth": bit_depth,
            "black_level_dn": black,
            "raw_dn": scaled,
            "raw_dn_clipped": raw_dn_clipped,
            "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
            "strongest_neighbor_fraction": row.get("strongest_neighbor_fraction", ""),
            "field_evidence_gate": row.get("field_evidence_gate", ""),
            "crosstalk_evidence_gate": row.get("crosstalk_evidence_gate", ""),
            "combined_evidence_gate": row.get("combined_evidence_gate", ""),
            "production_lut_gate": row.get("production_lut_gate", ""),
            "confidence_class": row.get("confidence_class", ""),
            "prior_seed_gate": prior.get("gate", "MISSING"),
            "product_lut_ready": row.get("product_lut_ready", False),
        }
        output.append(out)
        for kernel in kernels_by_query.get(qid, []):
            frac = finite_float(kernel.get("response_fraction"), 0.0)
            kernel_output.append(
                {
                    "probe_id": probe_id,
                    "runtime_query_id": qid,
                    "slug": slug,
                    "wavelength_nm": row.get("wavelength_nm", ""),
                    "color_channel": row.get("color_channel", ""),
                    "dx": kernel.get("dx", ""),
                    "dz": kernel.get("dz", ""),
                    "response_fraction": frac,
                    "signal_e": signal,
                    "kernel_signal_e": signal * frac,
                    "color_relation": kernel.get("color_relation", ""),
                    "evidence_gate": kernel.get("evidence_gate", ""),
                }
            )
    return output, kernel_output


def mean(values: list[float], default: float = 0.0) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else default


def summarize_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("slug", "")), str(row.get("wavelength_nm", "")), str(row.get("color_channel", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (_slug, _wave, _color), group in sorted(groups.items()):
        signals = [finite_float(row.get("signal_e")) for row in group]
        noises = [finite_float(row.get("total_noise_e_rms")) for row in group]
        snrs = [finite_float(row.get("snr_db")) for row in group]
        raw_dns = [finite_float(row.get("raw_dn_clipped")) for row in group]
        output_xtalk = [finite_float(row.get("output_crosstalk_fraction")) for row in group]
        strongest = [finite_float(row.get("strongest_neighbor_fraction")) for row in group]
        center_rows = [
            row
            for row in group
            if abs(finite_float(row.get("field_x_norm"), 999.0)) <= 1e-12
            and abs(finite_float(row.get("field_z_norm"), 999.0)) <= 1e-12
        ]
        center_signal = mean([finite_float(row.get("signal_e")) for row in center_rows], math.nan)
        edge_signals = [
            finite_float(row.get("signal_e"))
            for row in group
            if math.hypot(finite_float(row.get("field_x_norm"), 0.0), finite_float(row.get("field_z_norm"), 0.0)) > 1e-12
        ]
        min_edge = min([value for value in edge_signals if math.isfinite(value)], default=math.nan)
        edge_ratio = min_edge / center_signal if math.isfinite(min_edge) and math.isfinite(center_signal) and center_signal > 0 else math.nan
        first = group[0]
        output.append(
            {
                "slug": first.get("slug", ""),
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "wavelength_nm": first.get("wavelength_nm", ""),
                "color_channel": first.get("color_channel", ""),
                "probe_count": len(group),
                "allowed_count": sum(1 for row in group if boolish(row.get("query_allowed"))),
                "query_gate": ";".join(sorted({str(row.get("query_gate", "")) for row in group if row.get("query_gate", "")})),
                "field_evidence_gate": ";".join(sorted({str(row.get("field_evidence_gate", "")) for row in group if row.get("field_evidence_gate", "")})),
                "crosstalk_evidence_gate": ";".join(sorted({str(row.get("crosstalk_evidence_gate", "")) for row in group if row.get("crosstalk_evidence_gate", "")})),
                "production_lut_gate": ";".join(sorted({str(row.get("production_lut_gate", "")) for row in group if row.get("production_lut_gate", "")})),
                "mean_signal_e": mean(signals),
                "min_signal_e": min([value for value in signals if math.isfinite(value)], default=math.nan),
                "max_signal_e": max([value for value in signals if math.isfinite(value)], default=math.nan),
                "center_signal_e": center_signal,
                "min_edge_signal_e": min_edge,
                "edge_to_center_signal_ratio": edge_ratio,
                "mean_total_noise_e_rms": mean(noises),
                "min_snr_db": min([value for value in snrs if math.isfinite(value)], default=math.nan),
                "max_snr_db": max([value for value in snrs if math.isfinite(value)], default=math.nan),
                "mean_raw_dn": mean(raw_dns),
                "max_output_crosstalk_fraction": max([value for value in output_xtalk if math.isfinite(value)], default=math.nan),
                "max_strongest_neighbor_fraction": max([value for value in strongest if math.isfinite(value)], default=math.nan),
                "prior_seed_gate": ";".join(sorted({str(row.get("prior_seed_gate", "")) for row in group if row.get("prior_seed_gate", "")})),
                "product_lut_ready": any(boolish(row.get("product_lut_ready")) for row in group),
            }
        )
    return output


def validate_probe(rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not rows:
        issues.append({"severity": "error", "code": "no_probe_rows"})
    for row in rows:
        if boolish(row.get("product_lut_ready")):
            issues.append({"severity": "error", "code": "product_ready_true", "probe_id": row.get("probe_id")})
        if finite_float(row.get("raw_dn_clipped"), 0.0) < 0:
            issues.append({"severity": "error", "code": "negative_raw_dn", "probe_id": row.get("probe_id")})
        if mode == "product" and boolish(row.get("query_allowed")):
            issues.append({"severity": "error", "code": "product_probe_unexpectedly_allowed", "probe_id": row.get("probe_id")})
    sums: dict[str, float] = defaultdict(float)
    for kernel in kernel_rows:
        sums[str(kernel.get("probe_id", ""))] += finite_float(kernel.get("response_fraction"), 0.0)
    for probe_id, total in sums.items():
        if abs(total - 1.0) > 1e-6:
            issues.append({"severity": "error", "code": "kernel_fraction_sum_not_one", "probe_id": probe_id, "sum": total})
    return {
        "schema": "camera_e2e_sensor_probe_validation_v1",
        "pass": not issues,
        "probe_row_count": len(rows),
        "kernel_probe_row_count": len(kernel_rows),
        "allowed_probe_count": sum(1 for row in rows if boolish(row.get("query_allowed"))),
        "mode": mode,
        "issues": issues,
    }


def html_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return html.escape(f"{value:.6g}")
        return html.escape(str(value))
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 100) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.check{color:#ffd36e}.fail{color:#ff8b8b}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    validation = payload["validation"]
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Sensor Probe</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Sensor Probe</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This is a scalar research probe using runtime optical rows and prior seeds.</p>
  <div class="grid">
    <div class="card"><div class="metric">{html_cell(payload.get("mode", ""))}</div><div class="muted">mode</div></div>
    <div class="card"><div class="metric {'pass' if validation.get('pass') else 'fail'}">{html_cell(validation.get("pass"))}</div><div class="muted">validation pass</div></div>
    <div class="card"><div class="metric">{html_cell(validation.get("probe_row_count", 0))}</div><div class="muted">probe rows</div></div>
    <div class="card"><div class="metric">{html_cell(validation.get("allowed_probe_count", 0))}</div><div class="muted">allowed probes</div></div>
  </div>
  <h2>Summary</h2>
  {html_table(payload["summary_rows"], SUMMARY_COLUMNS, limit=160)}
  <h2>Probe Rows</h2>
  {html_table(payload["probe_rows"], PROBE_COLUMNS, limit=120)}
  <h2>Crosstalk Kernel Probe Rows</h2>
  {html_table(payload["kernel_probe_rows"], KERNEL_PROBE_COLUMNS, limit=160)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package(package_dir: Path, payload: dict[str, Any]) -> None:
    package_path = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_path)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_sensor_probe_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_sensor_probe_summary_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_sensor_probe_csv"] = payload["outputs"]["probe_csv"]
    outputs["camera_e2e_sensor_probe_kernel_csv"] = payload["outputs"]["kernel_probe_csv"]
    outputs["camera_e2e_sensor_probe_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_sensor_probe"] = {
        "schema": payload["schema"],
        "validation": payload["validation"],
        "outputs": payload["outputs"],
    }
    write_json(package_path, package)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    bundle, runtime, kernels = load_bundle(args.bundle_json.resolve())
    requested_slugs = [] if args.slugs.strip().lower() in {"", "all", "*"} else [slug.strip() for slug in args.slugs.split(",") if slug.strip()]
    query, query_kernels = query_rows(
        bundle,
        runtime,
        kernels,
        slugs=requested_slugs,
        field_x_values=[finite_float(item.strip()) for item in args.field_x.split(",") if item.strip()],
        field_z_values=[finite_float(item.strip()) for item in args.field_z.split(",") if item.strip()],
        wavelength_nm=args.wavelength_nm,
        mode=args.mode,
    )
    probe, kernel_probe = probe_rows(
        package_dir=package_dir,
        runtime_rows=query,
        kernel_rows=query_kernels,
        incident_photons=args.incident_photons,
        exposure_s=args.exposure_s,
        temperature_c=args.temperature_c,
        analog_gain=args.analog_gain,
        digital_gain=args.digital_gain,
    )
    summary = summarize_probe_rows(probe)
    validation = validate_probe(probe, kernel_probe, mode=args.mode)
    json_path = output_dir / "camera_e2e_sensor_probe.json"
    summary_csv = output_dir / "camera_e2e_sensor_probe_summary.csv"
    probe_csv = output_dir / "camera_e2e_sensor_probe.csv"
    kernel_csv = output_dir / "camera_e2e_sensor_probe_kernel.csv"
    query_csv = output_dir / "camera_e2e_sensor_probe_runtime_query.csv"
    query_kernel_csv = output_dir / "camera_e2e_sensor_probe_runtime_query_kernel.csv"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_sensor_probe_v1",
        "artifact_role": "camera_e2e_scalar_research_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "inputs": {
            "slugs": args.slugs,
            "field_x": args.field_x,
            "field_z": args.field_z,
            "wavelength_nm": args.wavelength_nm,
            "incident_photons": args.incident_photons,
            "exposure_s": args.exposure_s,
            "temperature_c": args.temperature_c,
            "analog_gain": args.analog_gain,
            "digital_gain": args.digital_gain,
        },
        "validation": validation,
        "summary_rows": summary,
        "probe_rows": probe,
        "kernel_probe_rows": kernel_probe,
        "outputs": {
            "json": repo_rel(json_path),
            "summary_csv": repo_rel(summary_csv),
            "probe_csv": repo_rel(probe_csv),
            "kernel_probe_csv": repo_rel(kernel_csv),
            "runtime_query_csv": repo_rel(query_csv),
            "runtime_query_kernel_csv": repo_rel(query_kernel_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research_use": "allowed when query_allowed=true and evidence/prior gates are preserved",
            "product_use": "blocked unless runtime bundle production gates pass; current package is expected to fail closed",
        },
    }
    write_csv(summary_csv, summary, SUMMARY_COLUMNS)
    write_csv(probe_csv, probe, PROBE_COLUMNS)
    write_csv(kernel_csv, kernel_probe, KERNEL_PROBE_COLUMNS)
    write_csv(query_csv, query, QUERY_COLUMNS)
    write_csv(query_kernel_csv, query_kernels, QUERY_KERNEL_COLUMNS)
    write_json(json_path, payload)
    write_html(html_path, payload)
    update_package(package_dir, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--bundle-json", type=Path, default=DEFAULT_BUNDLE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="dep_2505_802_smartsens_sc550xs")
    parser.add_argument("--field-x", default="0")
    parser.add_argument("--field-z", default="0")
    parser.add_argument("--wavelength-nm", default="550")
    parser.add_argument("--mode", choices=["research", "product"], default="research")
    parser.add_argument("--incident-photons", type=float, default=8000.0)
    parser.add_argument("--exposure-s", type=float, default=0.01)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--analog-gain", type=float, default=1.0)
    parser.add_argument("--digital-gain", type=float, default=1.0)
    return parser


def main() -> None:
    payload = run_probe(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "validation": payload["validation"],
                "outputs": payload["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not payload["validation"]["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
