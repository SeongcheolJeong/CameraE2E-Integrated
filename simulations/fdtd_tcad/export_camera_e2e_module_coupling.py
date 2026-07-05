#!/usr/bin/env python3
"""Export CameraE2E module-coupling field LUT.

The sensor package has sensor-side field CRA and microlens-shift priors, but
CameraE2E consumers need a single module-facing table that also includes
vignetting/shading and assembly alignment terms. This exporter creates that
table while preserving the fact that current CRA/ML shift rows are not measured
module raytrace data.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_module_coupling"

FIELD_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_case",
    "field_x_norm",
    "field_z_norm",
    "field_radius_norm",
    "field_azimuth_deg",
    "wavelength_nm",
    "cra_x_deg",
    "cra_z_deg",
    "chief_ray_total_deg",
    "chief_ray_azimuth_deg",
    "lens_cra_x_deg",
    "lens_cra_z_deg",
    "sensor_cra_x_deg",
    "sensor_cra_z_deg",
    "cra_mismatch_x_deg",
    "cra_mismatch_z_deg",
    "cra_mismatch_total_deg",
    "cra_mismatch_tolerance_profile",
    "cra_mismatch_pass_tolerance_deg",
    "cra_mismatch_check_tolerance_deg",
    "cra_mismatch_gate",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "lens_shift_model",
    "sensor_decenter_x_um",
    "sensor_decenter_z_um",
    "sensor_decenter_sigma_prior_um",
    "sensor_tilt_x_deg",
    "sensor_tilt_z_deg",
    "sensor_tilt_sigma_prior_deg",
    "relative_illumination_cos4",
    "vignetting_model",
    "wavelength_dependent_pupil_gate",
    "wavelength_dependent_pupil_status",
    "pupil_relative_transmission",
    "pupil_cra_shift_x_deg",
    "pupil_cra_shift_z_deg",
    "pupil_cra_shift_uncertainty_deg",
    "pupil_model",
    "measurement_gate",
    "research_use_gate",
    "product_lut_gate",
    "module_coupling_gate",
    "product_lut_ready",
    "source",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "field_row_count",
    "wavelength_count",
    "min_relative_illumination",
    "max_chief_ray_total_deg",
    "max_cra_mismatch_total_deg",
    "field_measurement_gate",
    "cra_mismatch_gate",
    "research_use_gate",
    "product_lut_gate",
    "module_coupling_gate",
    "pupil_gate",
    "product_lut_ready",
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
        rows = list(csv.DictReader(handle))
    return [row for row in rows if next(iter(row.values()), "") != next(iter(row.keys()), "")]


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


def parse_wavelengths(text: str) -> list[float]:
    output: list[float] = []
    for item in str(text or "").replace(",", ";").split(";"):
        value = as_float(item.strip())
        if math.isfinite(value):
            output.append(value)
    return output or [450.0, 550.0, 620.0]


def gate_rank(gate: str) -> int:
    return {"PASS": 0, "CHECK": 1, "MISSING": 2, "FAIL": 3}.get(str(gate).upper(), 2)


def normalize_gate(value: str) -> str:
    clean = str(value or "").strip().upper()
    if clean in {"PASS", "CHECK", "MISSING", "FAIL"}:
        return clean
    if clean in {"MEASURED", "CALIBRATED", "RAYTRACE_VALIDATED", "VALIDATED"}:
        return "PASS"
    if clean in {"ASSUMED", "ASSUMED_NOT_MEASURED", "PRIOR", "DESIGN_PRIOR", "SIMULATED_NOT_MEASURED"}:
        return "CHECK"
    if clean in {"", "UNKNOWN", "NOT_AVAILABLE", "N/A", "NA"}:
        return "MISSING"
    return "MISSING"


def worst_gate(values: list[str]) -> str:
    clean = [normalize_gate(str(value or "MISSING")) for value in values if str(value or "").strip()]
    return max(clean or ["MISSING"], key=gate_rank)


def field_direction(x: float, z: float) -> tuple[float, float]:
    radius = math.hypot(x, z)
    azimuth = math.degrees(math.atan2(z, x)) if radius > 1e-12 else 0.0
    return radius, azimuth


def cra_direction(cra_x: float, cra_z: float) -> tuple[float, float]:
    total = math.hypot(cra_x, cra_z)
    azimuth = math.degrees(math.atan2(cra_z, cra_x)) if total > 1e-12 else 0.0
    return total, azimuth


def cos4_vignetting(cra_total_deg: float) -> float:
    return max(0.0, math.cos(math.radians(cra_total_deg))) ** 4


def wavelength_pupil_row(pupil: dict[str, Any], wavelength_nm: float) -> dict[str, float | str]:
    rows = pupil.get("wavelength_rows", []) if isinstance(pupil, dict) else []
    selected: dict[str, Any] = {}
    if isinstance(rows, list) and rows:
        selected = min(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: abs(as_float(row.get("wavelength_nm"), wavelength_nm) - wavelength_nm),
            default={},
        )
    return {
        "pupil_relative_transmission": as_float(selected.get("relative_pupil_transmission"), 1.0),
        "pupil_cra_shift_x_deg": as_float(selected.get("cra_shift_x_deg"), 0.0),
        "pupil_cra_shift_z_deg": as_float(selected.get("cra_shift_z_deg"), 0.0),
        "pupil_cra_shift_uncertainty_deg": as_float(pupil.get("cra_shift_uncertainty_deg"), math.nan) if isinstance(pupil, dict) else math.nan,
        "pupil_model": str(pupil.get("model", "")) if isinstance(pupil, dict) else "",
    }


def sensor_meta(package_dir: Path) -> dict[str, dict[str, str]]:
    return {row.get("slug", ""): row for row in read_csv_rows(package_dir / "camera_e2e_sensor_models" / "camera_e2e_sensor_model_summary.csv")}


def prior_models(package_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    models_dir = package_dir / "camera_e2e_prior_seed_models" / "models"
    for path in models_dir.glob("*.json"):
        payload = read_json(path)
        slug = payload.get("sensor", {}).get("slug") or path.stem
        out[slug] = payload
    return out


def build_module_coupling(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    fields = read_csv_rows(package_dir / "camera_e2e_field_design_cases.csv")
    meta = sensor_meta(package_dir)
    priors = prior_models(package_dir)
    field_validation = read_json(package_dir / "camera_module_field_map_validation.json")

    field_rows: list[dict[str, Any]] = []
    for field in fields:
        slug = field.get("slug", "")
        sensor = meta.get(slug, {})
        prior = priors.get(slug, {})
        module_prior = prior.get("module_coupling_prior", {})
        alignment = module_prior.get("sensor_position_tilt_decenter", {})
        decenter = alignment.get("sensor_decenter_um", {}) if isinstance(alignment.get("sensor_decenter_um"), dict) else {}
        tilt = alignment.get("sensor_tilt_deg", {}) if isinstance(alignment.get("sensor_tilt_deg"), dict) else {}
        pupil = module_prior.get("wavelength_dependent_chief_ray_pupil", {})
        vignetting = module_prior.get("vignetting_shading", {})
        wavelengths = parse_wavelengths(field.get("wavelength_set_nm", ""))
        fx = as_float(field.get("field_x_norm"), 0.0)
        fz = as_float(field.get("field_z_norm"), 0.0)
        radius, field_az = field_direction(fx, fz)
        cra_x = as_float(field.get("cra_x_deg"), 0.0)
        cra_z = as_float(field.get("cra_z_deg"), 0.0)
        cra_total, cra_az = cra_direction(cra_x, cra_z)
        meas_gate = field.get("measurement_gate", "MISSING")
        pupil_gate = str(pupil.get("gate", "MISSING"))
        mismatch_gate = str(field.get("cra_mismatch_gate", "MISSING") or "MISSING")
        research_gate = worst_gate(["CHECK", meas_gate])
        product_gate = worst_gate([meas_gate, pupil_gate, mismatch_gate])
        for wavelength in wavelengths:
            pupil_values = wavelength_pupil_row(pupil, wavelength)
            field_rows.append(
                {
                    "slug": slug,
                    "code": field.get("code", ""),
                    "manufacturer": sensor.get("manufacturer", ""),
                    "device_name": sensor.get("device_name", ""),
                    "field_case": field.get("field_case", ""),
                    "field_x_norm": fx,
                    "field_z_norm": fz,
                    "field_radius_norm": radius,
                    "field_azimuth_deg": field_az,
                    "wavelength_nm": wavelength,
                    "cra_x_deg": cra_x,
                    "cra_z_deg": cra_z,
                    "chief_ray_total_deg": cra_total,
                    "chief_ray_azimuth_deg": cra_az,
                    "lens_cra_x_deg": as_float(field.get("lens_cra_x_deg")),
                    "lens_cra_z_deg": as_float(field.get("lens_cra_z_deg")),
                    "sensor_cra_x_deg": as_float(field.get("sensor_cra_x_deg")),
                    "sensor_cra_z_deg": as_float(field.get("sensor_cra_z_deg")),
                    "cra_mismatch_x_deg": as_float(field.get("cra_mismatch_x_deg")),
                    "cra_mismatch_z_deg": as_float(field.get("cra_mismatch_z_deg")),
                    "cra_mismatch_total_deg": as_float(field.get("cra_mismatch_total_deg")),
                    "cra_mismatch_tolerance_profile": field.get("cra_mismatch_tolerance_profile", ""),
                    "cra_mismatch_pass_tolerance_deg": as_float(field.get("cra_mismatch_pass_tolerance_deg")),
                    "cra_mismatch_check_tolerance_deg": as_float(field.get("cra_mismatch_check_tolerance_deg")),
                    "cra_mismatch_gate": mismatch_gate,
                    "lens_shift_x_um": as_float(field.get("lens_shift_x_um"), 0.0),
                    "lens_shift_z_um": as_float(field.get("lens_shift_z_um"), 0.0),
                    "lens_shift_model": field.get("lens_shift_model", ""),
                    "sensor_decenter_x_um": as_float(decenter.get("x"), 0.0),
                    "sensor_decenter_z_um": as_float(decenter.get("z"), 0.0),
                    "sensor_decenter_sigma_prior_um": as_float(decenter.get("sigma_prior_um"), 0.0),
                    "sensor_tilt_x_deg": as_float(tilt.get("x"), 0.0),
                    "sensor_tilt_z_deg": as_float(tilt.get("z"), 0.0),
                    "sensor_tilt_sigma_prior_deg": as_float(tilt.get("sigma_prior_deg"), 0.0),
                    "relative_illumination_cos4": cos4_vignetting(cra_total),
                    "vignetting_model": vignetting.get("model", "cos4_cra_seed"),
                    "wavelength_dependent_pupil_gate": pupil_gate,
                    "wavelength_dependent_pupil_status": pupil.get("status", "missing_raytrace"),
                    **pupil_values,
                    "measurement_gate": meas_gate,
                    "research_use_gate": research_gate,
                    "product_lut_gate": product_gate,
                    "module_coupling_gate": product_gate,
                    "product_lut_ready": False,
                    "source": field.get("source", ""),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in field_rows:
        by_slug.setdefault(str(row.get("slug", "")), []).append(row)
    for slug, rows in sorted(by_slug.items()):
        sensor = meta.get(slug, {})
        rel = [as_float(row.get("relative_illumination_cos4")) for row in rows]
        cra = [as_float(row.get("chief_ray_total_deg")) for row in rows]
        mismatch = [as_float(row.get("cra_mismatch_total_deg")) for row in rows]
        gates = [str(row.get("module_coupling_gate", "")) for row in rows]
        research_gates = [str(row.get("research_use_gate", "")) for row in rows]
        product_gates = [str(row.get("product_lut_gate", "")) for row in rows]
        meas_gates = sorted({str(row.get("measurement_gate", "")) for row in rows if row.get("measurement_gate")})
        mismatch_gates = sorted({str(row.get("cra_mismatch_gate", "")) for row in rows if row.get("cra_mismatch_gate")})
        pupil_gates = sorted({str(row.get("wavelength_dependent_pupil_gate", "")) for row in rows if row.get("wavelength_dependent_pupil_gate")})
        summary_rows.append(
            {
                "slug": slug,
                "code": rows[0].get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "field_row_count": len(rows),
                "wavelength_count": len({row.get("wavelength_nm") for row in rows}),
                "min_relative_illumination": min([value for value in rel if math.isfinite(value)], default=math.nan),
                "max_chief_ray_total_deg": max([value for value in cra if math.isfinite(value)], default=math.nan),
                "max_cra_mismatch_total_deg": max([value for value in mismatch if math.isfinite(value)], default=math.nan),
                "field_measurement_gate": ";".join(meas_gates),
                "cra_mismatch_gate": ";".join(mismatch_gates),
                "research_use_gate": worst_gate(research_gates),
                "product_lut_gate": worst_gate(product_gates),
                "module_coupling_gate": worst_gate(gates),
                "pupil_gate": ";".join(pupil_gates),
                "product_lut_ready": False,
                "primary_blocker": "module raytrace/measured CRA and wavelength-dependent pupil data missing"
                if field_validation.get("gate") == "MISSING"
                else "verify module coupling gates before production use",
            }
        )

    field_csv = output_dir / "camera_e2e_module_coupling_field_lut.csv"
    summary_csv = output_dir / "camera_e2e_module_coupling_summary.csv"
    json_path = output_dir / "camera_e2e_module_coupling.json"
    html_path = output_dir / "index.html"
    validation = validate(field_rows, summary_rows)
    manifest = {
        "schema": "camera_e2e_module_coupling_export_v1",
        "artifact_role": "camera_e2e_module_coupling_research_lut",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sensor_count": len(summary_rows),
        "field_row_count": len(field_rows),
        "summary_row_count": len(summary_rows),
        "field_case_count": len({row.get("field_case") for row in field_rows}),
        "wavelengths_nm": sorted({row.get("wavelength_nm") for row in field_rows}),
        "product_lut_ready": False,
        "field_map_validation": field_validation,
        "validation": validation,
        "gate_counts": dict(Counter(row.get("module_coupling_gate", "") for row in field_rows)),
        "research_gate_counts": dict(Counter(row.get("research_use_gate", "") for row in field_rows)),
        "product_gate_counts": dict(Counter(row.get("product_lut_gate", "") for row in field_rows)),
        "outputs": {
            "json": repo_rel(json_path),
            "field_lut_csv": repo_rel(field_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
        "policy": {
            "research_use": "allowed if CHECK/MISSING gates are propagated",
            "product_use": "blocked until module raytrace/measured CRA, ML shift, vignetting, and pupil data are imported",
        },
    }
    write_csv(field_csv, field_rows, FIELD_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(json_path, manifest)
    write_html(html_path, manifest, summary_rows, field_rows)
    update_package(package_dir, manifest)
    return manifest


def validate(field_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not field_rows:
        issues.append({"severity": "error", "code": "no_module_coupling_rows"})
    if not summary_rows:
        issues.append({"severity": "error", "code": "no_module_coupling_summary"})
    for row in field_rows:
        rel = as_float(row.get("relative_illumination_cos4"))
        if not math.isfinite(rel) or rel < 0 or rel > 1:
            issues.append({"severity": "error", "code": "relative_illumination_out_of_range", "slug": row.get("slug")})
        if row.get("product_lut_ready") is True:
            issues.append({"severity": "error", "code": "product_lut_ready_true", "slug": row.get("slug")})
    return {
        "schema": "camera_e2e_module_coupling_validation_v1",
        "pass": not issues,
        "issue_count": len(issues),
        "field_row_count": len(field_rows),
        "summary_row_count": len(summary_rows),
        "issues": issues,
    }


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


def write_html(path: Path, manifest: dict[str, Any], summary_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    body = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CameraE2E Module Coupling</title><style>{css}</style></head>
<body><main>
<h1>CameraE2E Module Coupling</h1>
<p class="muted">Generated {html_cell(manifest.get("generated_at", ""))}. Module raytrace/measured CRA is not present unless field-map validation says otherwise.</p>
<div class="grid">
<div class="card"><div class="metric">{html_cell(manifest.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("field_row_count", 0))}</div><div class="muted">field rows</div></div>
<div class="card"><div class="metric">{html_cell(manifest.get("validation", {}).get("pass"))}</div><div class="muted">validation pass</div></div>
</div>
<h2>Summary</h2>{html_table(summary_rows, SUMMARY_COLUMNS, limit=80)}
<h2>Field LUT</h2>{html_table(field_rows, FIELD_COLUMNS, limit=160)}
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
    outputs["camera_e2e_module_coupling_json"] = manifest["outputs"]["json"]
    outputs["camera_e2e_module_coupling_field_lut_csv"] = manifest["outputs"]["field_lut_csv"]
    outputs["camera_e2e_module_coupling_summary_csv"] = manifest["outputs"]["summary_csv"]
    outputs["camera_e2e_module_coupling_html"] = manifest["outputs"]["html"]
    package["latest_camera_e2e_module_coupling"] = {
        "schema": manifest["schema"],
        "sensor_count": manifest["sensor_count"],
        "field_row_count": manifest["field_row_count"],
        "validation": manifest["validation"],
        "outputs": manifest["outputs"],
    }
    write_json(package_path, package)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    manifest = build_module_coupling(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "validation": manifest["validation"],
                "sensor_count": manifest["sensor_count"],
                "field_row_count": manifest["field_row_count"],
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
