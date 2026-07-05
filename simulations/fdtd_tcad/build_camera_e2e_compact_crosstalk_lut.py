#!/usr/bin/env python3
"""Build CameraE2E compact crosstalk kernels from field-FDTD and TCAD priors.

This is a fast surrogate for camera-system studies when full finite-array Meep
crosstalk is still resource-limited. It is intentionally gated as CHECK and
must not be treated as product crosstalk evidence.
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

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_INPUT_FIELD_LUT = DEFAULT_PACKAGE_DIR / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.csv"
DEFAULT_TCAD_PROFILE_DIR = ROOT / "image_sensor_db" / "generated_tcad_profiles"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_compact_crosstalk_lut"

KERNEL_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "pixel_pitch_um",
    "cfa_pattern",
    "ocl_mode_guess",
    "field_case",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "wavelength_nm",
    "color_channel",
    "kernel_scope",
    "neighborhood",
    "output_pitch_um",
    "dx",
    "dz",
    "response_fraction",
    "color_relation",
    "sigma_x_um",
    "sigma_z_um",
    "centroid_x_um",
    "centroid_z_um",
    "dti_width_um",
    "dti_depth_um",
    "dti_barrier_factor",
    "pre_barrier_truncation_fraction",
    "center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "model",
    "source",
    "evidence_level",
    "evidence_gate",
    "product_lut_ready",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "field_case",
    "wavelength_nm",
    "color_channel",
    "kernel_scope",
    "neighborhood",
    "center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
    "pre_barrier_truncation_fraction",
    "sigma_x_um",
    "sigma_z_um",
    "centroid_x_um",
    "centroid_z_um",
    "dti_width_um",
    "dti_depth_um",
    "dti_barrier_factor",
    "evidence_level",
    "evidence_gate",
    "source",
    "product_lut_ready",
]

KERNEL_NUMERIC_COLUMNS = [
    "pixel_pitch_um",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "wavelength_nm",
    "neighborhood",
    "output_pitch_um",
    "dx",
    "dz",
    "response_fraction",
    "sigma_x_um",
    "sigma_z_um",
    "centroid_x_um",
    "centroid_z_um",
    "dti_width_um",
    "dti_depth_um",
    "dti_barrier_factor",
    "pre_barrier_truncation_fraction",
    "center_fraction",
    "output_crosstalk_fraction",
    "strongest_neighbor_fraction",
]

KERNEL_STRING_COLUMNS = [
    column for column in KERNEL_COLUMNS if column not in KERNEL_NUMERIC_COLUMNS and column != "product_lut_ready"
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def read_csv(path: Path) -> list[dict[str, str]]:
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
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_kernel_npz(path: Path, rows: list[dict[str, Any]], source_csv: Path) -> None:
    arrays: dict[str, Any] = {
        "schema": np.array("camera_e2e_compact_crosstalk_lut_v1", dtype="U80"),
        "artifact_role": np.array("camera_e2e_compact_crosstalk_kernel_lut_npz", dtype="U96"),
        "product_lut_ready": np.array(False, dtype=np.bool_),
        "row_count": np.array(len(rows), dtype=np.int64),
        "columns": np.array(KERNEL_COLUMNS, dtype="U80"),
        "numeric_columns": np.array(KERNEL_NUMERIC_COLUMNS, dtype="U80"),
        "string_columns": np.array(KERNEL_STRING_COLUMNS, dtype="U80"),
        "source_csv": np.array(repo_rel(source_csv), dtype="U512"),
    }
    for column in KERNEL_NUMERIC_COLUMNS:
        arrays[column] = np.array([finite_float(row.get(column)) for row in rows], dtype=np.float64)
    for column in KERNEL_STRING_COLUMNS:
        arrays[column] = np.array([str(row.get(column, "")) for row in rows], dtype="U1024")
    arrays["row_product_lut_ready"] = np.array([boolish(row.get("product_lut_ready")) for row in rows], dtype=np.bool_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def validate_kernel_npz(path: Path, rows: list[dict[str, Any]], *, tolerance: float = 1e-10) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != "camera_e2e_compact_crosstalk_lut_v1":
            issues.append({"severity": "error", "code": "schema_mismatch", "actual": schema})
        row_count = int(data["row_count"])
        if row_count != len(rows):
            issues.append({"severity": "error", "code": "row_count_mismatch", "npz": row_count, "rows": len(rows)})
        for column in KERNEL_NUMERIC_COLUMNS:
            if column not in data:
                issues.append({"severity": "error", "code": "missing_numeric_array", "column": column})
                continue
            values = data[column]
            if len(values) != len(rows):
                issues.append({"severity": "error", "code": "numeric_array_length_mismatch", "column": column})
                continue
            for index, row in enumerate(rows):
                expected = finite_float(row.get(column))
                actual = float(values[index])
                if math.isnan(expected) and math.isnan(actual):
                    continue
                if abs(expected - actual) > max(tolerance, tolerance * max(1.0, abs(expected))):
                    issues.append(
                        {
                            "severity": "error",
                            "code": "numeric_value_mismatch",
                            "column": column,
                            "row_index": index,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                    break
        for column in KERNEL_STRING_COLUMNS:
            if column not in data:
                issues.append({"severity": "error", "code": "missing_string_array", "column": column})
                continue
            values = data[column]
            if len(values) != len(rows):
                issues.append({"severity": "error", "code": "string_array_length_mismatch", "column": column})
                continue
            for index, row in enumerate(rows):
                expected = str(row.get(column, ""))
                actual = str(values[index])
                if expected != actual:
                    issues.append(
                        {
                            "severity": "error",
                            "code": "string_value_mismatch",
                            "column": column,
                            "row_index": index,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                    break
        if "row_product_lut_ready" in data and bool(np.any(data["row_product_lut_ready"])):
            issues.append({"severity": "error", "code": "row_product_lut_ready_true"})
    return {
        "schema": "camera_e2e_compact_crosstalk_npz_validation_v1",
        "pass": not issues,
        "bad_count": len(issues),
        "row_count": len(rows),
        "numeric_column_count": len(KERNEL_NUMERIC_COLUMNS),
        "string_column_count": len(KERNEL_STRING_COLUMNS),
        "issues": issues,
    }


def repo_rel(path: Path | str | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def html_cell(value: Any) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return html.escape(f"{value:.6g}")
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
    shown = rows[:limit]
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - limit} more rows in CSV</td></tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def normal_cdf(value: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    return 0.5 * (1.0 + math.erf((value - mu) / (sigma * math.sqrt(2.0))))


def gaussian_cell_probability(dx: int, dz: int, pitch: float, mux: float, muz: float, sx: float, sz: float) -> float:
    x0 = dx * pitch - 0.5 * pitch
    x1 = dx * pitch + 0.5 * pitch
    z0 = dz * pitch - 0.5 * pitch
    z1 = dz * pitch + 0.5 * pitch
    px = max(0.0, normal_cdf(x1, mux, sx) - normal_cdf(x0, mux, sx))
    pz = max(0.0, normal_cdf(z1, muz, sz) - normal_cdf(z0, muz, sz))
    return px * pz


def ocl_group_factor(mode: str) -> int:
    text = str(mode or "").lower()
    if "3x3" in text or "nona" in text:
        return 3
    if "2x2" in text or "quad" in text:
        return 2
    return 1


def neighborhood_for_mode(mode: str) -> int:
    factor = ocl_group_factor(mode)
    if factor == 3:
        return 7
    if factor == 2:
        return 5
    return 3


def color_relation(dx: int, dz: int, color: str, cfa_pattern: str) -> str:
    if dx == 0 and dz == 0:
        return "target_color"
    pattern = str(cfa_pattern or "").lower()
    if "mono" in pattern:
        return "same_color"
    if "quad" in pattern or "nona" in pattern:
        return "same_color_group_or_binned_neighbor"
    if color == "green":
        same = (dx + dz) % 2 == 0
    else:
        same = dx % 2 == 0 and dz % 2 == 0
    return "same_color" if same else "cross_color"


def load_tcad_geometry(profile_dir: Path, slug: str) -> dict[str, Any]:
    path = profile_dir / slug / "profile.json"
    if not path.exists():
        return {"source": "", "geometry": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometry = payload.get("geometry", {}) if isinstance(payload.get("geometry"), dict) else {}
    return {"source": repo_rel(path), "geometry": geometry}


def dti_params(profile_dir: Path, slug: str, pitch: float) -> tuple[float, float, str]:
    payload = load_tcad_geometry(profile_dir, slug)
    geometry = payload["geometry"]
    width = finite_float(geometry.get("dti_width_um"), 0.0)
    depth = finite_float(geometry.get("bdti", {}).get("depth_max_um") if isinstance(geometry.get("bdti"), dict) else None, math.nan)
    if not math.isfinite(depth):
        depth = finite_float(geometry.get("dti_depth_um"), finite_float(geometry.get("depth_um"), 0.0) * 0.35)
    if width <= 0.0:
        width = 0.04 * pitch
    if depth <= 0.0:
        depth = 0.8 * pitch
    return width, depth, payload["source"]


def barrier_factor(width_um: float, depth_um: float, pitch_um: float) -> float:
    width_term = math.exp(-6.0 * max(width_um, 0.0) / max(pitch_um, 1e-9))
    depth_term = math.exp(-0.8 * max(depth_um, 0.0) / max(pitch_um, 1e-9))
    return max(0.03, min(1.0, width_term * depth_term))


def sigma_from_field(row: dict[str, str], output_pitch: float) -> tuple[float, float, str]:
    focal_fraction = finite_float(row.get("focal_target_fraction"))
    rms = finite_float(row.get("focal_rms_radius_um"))
    if math.isfinite(rms) and rms > 0.0:
        sigma = max(0.08 * output_pitch, rms / math.sqrt(2.0))
        return sigma, sigma, "compact gaussian from quantitative field-FDTD focal_rms_radius"
    evidence = str(row.get("evidence_level", ""))
    base = 0.24 * output_pitch
    if evidence == "design_prior_spectral_rolloff":
        base = 0.30 * output_pitch
    elif evidence == "tcad_lateral_proxy_scaled":
        base = 0.27 * output_pitch
    if math.isfinite(focal_fraction) and focal_fraction > 0:
        base *= max(0.75, min(1.45, 1.0 / math.sqrt(max(focal_fraction, 0.08))))
    return base, base, "compact gaussian stack/field prior"


def centroid_from_field(row: dict[str, str]) -> tuple[float, float]:
    x = finite_float(row.get("focal_centroid_shift_x_um"))
    z = finite_float(row.get("focal_centroid_shift_z_um"))
    if not math.isfinite(x):
        x = finite_float(row.get("lens_shift_x_um"), 0.0)
    if not math.isfinite(z):
        z = finite_float(row.get("lens_shift_z_um"), 0.0)
    return x, z


def build_for_row(row: dict[str, str], profile_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pitch = finite_float(row.get("pixel_pitch_um"), 1.0)
    group_factor = ocl_group_factor(row.get("ocl_mode_guess", ""))
    output_pitch = pitch * group_factor
    neighborhood = neighborhood_for_mode(row.get("ocl_mode_guess", ""))
    half = neighborhood // 2
    sigma_x, sigma_z, sigma_source = sigma_from_field(row, output_pitch)
    centroid_x, centroid_z = centroid_from_field(row)
    dti_width, dti_depth, tcad_source = dti_params(profile_dir, row.get("slug", ""), pitch)
    barrier = barrier_factor(dti_width, dti_depth, pitch)
    raw: dict[tuple[int, int], float] = {}
    pre_barrier_sum = 0.0
    for dz in range(-half, half + 1):
        for dx in range(-half, half + 1):
            value = gaussian_cell_probability(dx, dz, output_pitch, centroid_x, centroid_z, sigma_x, sigma_z)
            raw[(dx, dz)] = value
            pre_barrier_sum += value
    pre_barrier_truncation = max(0.0, 1.0 - pre_barrier_sum)
    weighted = {}
    for key, value in raw.items():
        weighted[key] = value if key == (0, 0) else value * barrier
    total = sum(weighted.values()) or 1.0
    normalized = {key: value / total for key, value in weighted.items()}
    center = normalized.get((0, 0), 0.0)
    neighbors = [value for key, value in normalized.items() if key != (0, 0)]
    output_xt = max(0.0, 1.0 - center)
    strongest = max(neighbors) if neighbors else 0.0
    evidence_source = row.get("source", "")
    source = "; ".join(item for item in [evidence_source, tcad_source, sigma_source] if item)
    rows: list[dict[str, Any]] = []
    for dz in range(-half, half + 1):
        for dx in range(-half, half + 1):
            rows.append(
                {
                    "slug": row.get("slug", ""),
                    "code": row.get("code", ""),
                    "manufacturer": row.get("manufacturer", ""),
                    "device_name": row.get("device_name", ""),
                    "pixel_pitch_um": pitch,
                    "cfa_pattern": row.get("cfa_pattern", ""),
                    "ocl_mode_guess": row.get("ocl_mode_guess", ""),
                    "field_case": row.get("field_case", ""),
                    "field_x_norm": row.get("field_x_norm", ""),
                    "field_z_norm": row.get("field_z_norm", ""),
                    "cra_x_deg": row.get("cra_x_deg", ""),
                    "cra_z_deg": row.get("cra_z_deg", ""),
                    "lens_shift_x_um": row.get("lens_shift_x_um", ""),
                    "lens_shift_z_um": row.get("lens_shift_z_um", ""),
                    "wavelength_nm": row.get("wavelength_nm", ""),
                    "color_channel": row.get("color_channel", ""),
                    "kernel_scope": "compact_output_crosstalk",
                    "neighborhood": neighborhood,
                    "output_pitch_um": output_pitch,
                    "dx": dx,
                    "dz": dz,
                    "response_fraction": normalized[(dx, dz)],
                    "color_relation": color_relation(dx, dz, row.get("color_channel", ""), row.get("cfa_pattern", "")),
                    "sigma_x_um": sigma_x,
                    "sigma_z_um": sigma_z,
                    "centroid_x_um": centroid_x,
                    "centroid_z_um": centroid_z,
                    "dti_width_um": dti_width,
                    "dti_depth_um": dti_depth,
                    "dti_barrier_factor": barrier,
                    "pre_barrier_truncation_fraction": pre_barrier_truncation,
                    "center_fraction": center,
                    "output_crosstalk_fraction": output_xt,
                    "strongest_neighbor_fraction": strongest,
                    "model": "separable gaussian focal spot plus DTI attenuation compact crosstalk surrogate",
                    "source": source,
                    "evidence_level": "compact_model_from_field_fdtd_or_stack_prior",
                    "evidence_gate": "CHECK",
                    "product_lut_ready": False,
                }
            )
    summary = {
        "slug": row.get("slug", ""),
        "code": row.get("code", ""),
        "field_case": row.get("field_case", ""),
        "wavelength_nm": row.get("wavelength_nm", ""),
        "color_channel": row.get("color_channel", ""),
        "kernel_scope": "compact_output_crosstalk",
        "neighborhood": neighborhood,
        "center_fraction": center,
        "output_crosstalk_fraction": output_xt,
        "strongest_neighbor_fraction": strongest,
        "pre_barrier_truncation_fraction": pre_barrier_truncation,
        "sigma_x_um": sigma_x,
        "sigma_z_um": sigma_z,
        "centroid_x_um": centroid_x,
        "centroid_z_um": centroid_z,
        "dti_width_um": dti_width,
        "dti_depth_um": dti_depth,
        "dti_barrier_factor": barrier,
        "evidence_level": "compact_model_from_field_fdtd_or_stack_prior",
        "evidence_gate": "CHECK",
        "source": source,
        "product_lut_ready": False,
    }
    return rows, summary


def write_html(output_dir: Path, report: dict[str, Any], summary_rows: list[dict[str, Any]], kernel_rows: list[dict[str, Any]]) -> None:
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraE2E Compact Crosstalk LUT</title>
  <style>
    body {{ margin:0; background:#081118; color:#e7f5ff; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1360px; margin:0 auto; padding:28px; }}
    h1 {{ margin:0 0 8px; }}
    h2 {{ color:#54e2ff; margin-top:26px; }}
    p {{ color:#9eb6c8; }}
    .note {{ border-left:3px solid #ffd95f; color:#e7f5ff; padding-left:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ border:1px solid #254255; border-radius:8px; background:#0e1b25; padding:14px; }}
    .metric {{ font-size:26px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:10px; }}
    th,td {{ border:1px solid #254255; padding:7px 8px; text-align:left; vertical-align:top; }}
    th {{ color:#54e2ff; background:#102633; }}
    code {{ color:#d8f8ff; }}
  </style>
</head>
<body>
<main>
  <h1>CameraE2E Compact Crosstalk LUT</h1>
  <p>Generated: <code>{html_cell(report.get("generated_at", ""))}</code></p>
  <p class="note">This artifact is a CHECK-gated compact surrogate. It is useful for CameraE2E sensitivity runs while full finite-array Meep crosstalk is resource-limited, but it is not product crosstalk evidence.</p>
  <p>NPZ validation pass: <code>{html_cell(report.get("npz_validation", {}).get("pass", ""))}</code>; bad count: <code>{html_cell(report.get("npz_validation", {}).get("bad_count", ""))}</code>.</p>
  <div class="grid">
    <div class="card"><div class="metric">{report.get("sensor_count", 0)}</div><div>sensors</div></div>
    <div class="card"><div class="metric">{report.get("summary_row_count", 0)}</div><div>point summaries</div></div>
    <div class="card"><div class="metric">{report.get("kernel_row_count", 0)}</div><div>kernel rows</div></div>
    <div class="card"><div class="metric">{report.get("production_ready_row_count", 0)}</div><div>production-ready rows</div></div>
  </div>
  <h2>Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS, limit=80)}
  <h2>Kernel Preview</h2>
  {html_table(kernel_rows, KERNEL_COLUMNS, limit=80)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    field_rows = read_csv(args.field_lut)
    kernel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in field_rows:
        rows, summary = build_for_row(row, args.tcad_profile_dir)
        kernel_rows.extend(rows)
        summary_rows.append(summary)
    output_dir = args.output_dir.resolve()
    kernel_csv = output_dir / "camera_e2e_compact_crosstalk_kernel_lut.csv"
    kernel_npz = output_dir / "camera_e2e_compact_crosstalk_kernel_lut.npz"
    summary_csv = output_dir / "camera_e2e_compact_crosstalk_summary.csv"
    report_json = output_dir / "camera_e2e_compact_crosstalk_report.json"
    write_csv(kernel_csv, kernel_rows, KERNEL_COLUMNS)
    write_kernel_npz(kernel_npz, kernel_rows, kernel_csv)
    npz_validation = validate_kernel_npz(kernel_npz, kernel_rows)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    report = {
        "schema": "camera_e2e_compact_crosstalk_lut_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "field_lut": repo_rel(args.field_lut),
        "kernel_csv": repo_rel(kernel_csv),
        "kernel_npz": repo_rel(kernel_npz),
        "summary_csv": repo_rel(summary_csv),
        "html_report": repo_rel(output_dir / "index.html"),
        "npz_validation": npz_validation,
        "sensor_count": len({row.get("slug", "") for row in field_rows if row.get("slug")}),
        "field_row_count": len(field_rows),
        "summary_row_count": len(summary_rows),
        "kernel_row_count": len(kernel_rows),
        "production_ready_row_count": 0,
        "product_lut_ready": False,
        "evidence_gate": "CHECK",
        "notes": [
            "Compact crosstalk uses Gaussian focal spot integration plus DTI attenuation priors.",
            "Use this for CameraE2E research/trend sensitivity only.",
            "Product crosstalk still requires finite-array Meep/measurement convergence PASS.",
        ],
    }
    write_json(report_json, report)
    write_html(output_dir, report, summary_rows, kernel_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-lut", type=Path, default=DEFAULT_INPUT_FIELD_LUT)
    parser.add_argument("--tcad-profile-dir", type=Path, default=DEFAULT_TCAD_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
