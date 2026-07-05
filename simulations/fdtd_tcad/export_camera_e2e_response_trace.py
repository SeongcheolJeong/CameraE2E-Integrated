#!/usr/bin/env python3
"""Export per-row pixel response calculation traces for CameraE2E review.

The runtime LUT already contains the CameraE2E response values. This exporter
adds auditability: for every runtime row it joins the nearest CFA/OCL/
passivation/Si material rows and computes a simple CFA x Si absorption sanity
check. The sanity check is not a replacement for Meep. It is a readable trace
that shows why a row is a research proxy and what measured inputs are missing.
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
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_response_trace"

TRACE_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "runtime_id",
    "field_x_norm",
    "field_z_norm",
    "cra_x_deg",
    "cra_z_deg",
    "color_channel",
    "wavelength_nm",
    "cfa_material_wavelength_nm",
    "cfa_transmission_proxy",
    "cfa_n",
    "cfa_k",
    "cfa_thickness_um",
    "cfa_material_source_kind",
    "cfa_product_lut_gate",
    "ocl_n",
    "ocl_k",
    "ocl_thickness_um",
    "passivation_n",
    "passivation_k",
    "passivation_thickness_um",
    "si_material_wavelength_nm",
    "si_n",
    "si_k",
    "si_thickness_um",
    "si_simple_absorption_fraction",
    "cfa_times_si_simple_fraction",
    "runtime_response_nominal",
    "runtime_response_min",
    "runtime_response_max",
    "runtime_vs_cfa_si_simple_delta",
    "crosstalk_center_fraction",
    "output_crosstalk_fraction",
    "direct_signal_response",
    "neighbor_leakage_response",
    "field_evidence_gate",
    "crosstalk_evidence_gate",
    "combined_evidence_gate",
    "confidence_class",
    "response_model",
    "evidence_level",
    "field_source",
    "calculation_path",
    "product_lut_ready",
]

SUMMARY_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "trace_row_count",
    "center_rgb_or_clear_row_count",
    "runtime_response_min",
    "runtime_response_max",
    "mean_runtime_response_nominal",
    "mean_cfa_times_si_simple_fraction",
    "max_runtime_vs_cfa_si_simple_abs_delta",
    "evidence_gate_counts",
    "confidence_class_counts",
    "product_lut_ready_count",
    "trace_product_gate",
    "primary_note",
]

CHECK_COLUMNS = ["check_id", "pass", "status", "evidence", "required_action"]


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
            writer.writerow({column: row.get(column, "") for column in columns})


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def float_key(value: Any) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.9g}"


def field_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("slug", ""),
        float_key(row.get("field_x_norm")),
        float_key(row.get("field_z_norm")),
        float_key(row.get("wavelength_nm")),
        row.get("color_channel", "") or row.get("color", ""),
    )


def nearest_material(
    rows: list[dict[str, str]],
    *,
    family: str,
    wavelength_nm: float,
    color_channel: str = "",
) -> dict[str, str]:
    candidates = [row for row in rows if row.get("material_family") == family]
    if color_channel:
        same_color = [row for row in candidates if row.get("color_channel") == color_channel]
        if same_color:
            candidates = same_color
    if not candidates:
        return {}
    return min(candidates, key=lambda row: abs(safe_float(row.get("wavelength_nm"), wavelength_nm) - wavelength_nm))


def simple_si_absorption(si_k: float, wavelength_nm: float, thickness_um: float) -> float:
    if not (math.isfinite(si_k) and math.isfinite(wavelength_nm) and math.isfinite(thickness_um)):
        return math.nan
    if wavelength_nm <= 0 or thickness_um <= 0:
        return math.nan
    alpha_per_um = 4.0 * math.pi * max(0.0, si_k) / (wavelength_nm / 1000.0)
    return max(0.0, min(1.0, 1.0 - math.exp(-alpha_per_um * thickness_um)))


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def build_trace_rows(package_dir: Path) -> list[dict[str, Any]]:
    runtime_rows = read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv")
    field_rows = read_csv_rows(package_dir / "camera_e2e_ingest_export" / "camera_e2e_field_response_lut.csv")
    material_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv"), "slug")
    field_by_key = {field_key(row): row for row in field_rows}

    output: list[dict[str, Any]] = []
    for row in runtime_rows:
        slug = row.get("slug", "")
        wavelength = safe_float(row.get("wavelength_nm"))
        color = row.get("color_channel", "")
        material_rows = material_by_slug.get(slug, [])
        cfa = nearest_material(material_rows, family="cfa_transmission_proxy", wavelength_nm=wavelength, color_channel=color)
        si = nearest_material(material_rows, family="silicon_fdtd_material", wavelength_nm=wavelength)
        ocl = nearest_material(material_rows, family="ocl_fdtd_material", wavelength_nm=wavelength)
        passivation = nearest_material(material_rows, family="passivation_fdtd_material", wavelength_nm=wavelength)
        field = field_by_key.get(field_key(row), {})

        cfa_t = safe_float(cfa.get("transmission_absorption_only"), 1.0 if color == "clear" else math.nan)
        si_k = safe_float(si.get("k"))
        si_thickness = safe_float(si.get("thickness_um"))
        si_abs = simple_si_absorption(si_k, wavelength, si_thickness)
        cfa_si = cfa_t * si_abs if math.isfinite(cfa_t) and math.isfinite(si_abs) else math.nan
        runtime_response = safe_float(row.get("response_nominal"))
        delta = runtime_response - cfa_si if math.isfinite(runtime_response) and math.isfinite(cfa_si) else math.nan

        if field.get("response_model"):
            calculation_path = field.get("response_model", "")
        elif row.get("confidence_class") == "product_calibrated":
            calculation_path = "product calibrated runtime response"
        else:
            calculation_path = "runtime response proxy; material rows provide optical sanity context"

        output.append(
            {
                "slug": slug,
                "code": row.get("code", ""),
                "manufacturer": row.get("manufacturer", ""),
                "device_name": row.get("device_name", ""),
                "runtime_id": row.get("runtime_id", ""),
                "field_x_norm": row.get("field_x_norm", ""),
                "field_z_norm": row.get("field_z_norm", ""),
                "cra_x_deg": row.get("cra_x_deg", ""),
                "cra_z_deg": row.get("cra_z_deg", ""),
                "color_channel": color,
                "wavelength_nm": row.get("wavelength_nm", ""),
                "cfa_material_wavelength_nm": cfa.get("wavelength_nm", ""),
                "cfa_transmission_proxy": cfa.get("transmission_absorption_only", "1.0" if color == "clear" else ""),
                "cfa_n": cfa.get("n", ""),
                "cfa_k": cfa.get("k", ""),
                "cfa_thickness_um": cfa.get("thickness_um", ""),
                "cfa_material_source_kind": cfa.get("material_source_kind", ""),
                "cfa_product_lut_gate": cfa.get("product_lut_gate", ""),
                "ocl_n": ocl.get("n", ""),
                "ocl_k": ocl.get("k", ""),
                "ocl_thickness_um": ocl.get("thickness_um", ""),
                "passivation_n": passivation.get("n", ""),
                "passivation_k": passivation.get("k", ""),
                "passivation_thickness_um": passivation.get("thickness_um", ""),
                "si_material_wavelength_nm": si.get("wavelength_nm", ""),
                "si_n": si.get("n", ""),
                "si_k": si.get("k", ""),
                "si_thickness_um": si.get("thickness_um", ""),
                "si_simple_absorption_fraction": f"{si_abs:.9g}" if math.isfinite(si_abs) else "",
                "cfa_times_si_simple_fraction": f"{cfa_si:.9g}" if math.isfinite(cfa_si) else "",
                "runtime_response_nominal": row.get("response_nominal", ""),
                "runtime_response_min": row.get("response_min", ""),
                "runtime_response_max": row.get("response_max", ""),
                "runtime_vs_cfa_si_simple_delta": f"{delta:.9g}" if math.isfinite(delta) else "",
                "crosstalk_center_fraction": row.get("crosstalk_center_fraction", ""),
                "output_crosstalk_fraction": row.get("output_crosstalk_fraction", ""),
                "direct_signal_response": row.get("direct_signal_response", ""),
                "neighbor_leakage_response": row.get("neighbor_leakage_response", ""),
                "field_evidence_gate": row.get("field_evidence_gate", ""),
                "crosstalk_evidence_gate": row.get("crosstalk_evidence_gate", ""),
                "combined_evidence_gate": row.get("combined_evidence_gate", ""),
                "confidence_class": row.get("confidence_class", ""),
                "response_model": field.get("response_model", ""),
                "evidence_level": field.get("evidence_level", ""),
                "field_source": field.get("source", ""),
                "calculation_path": calculation_path,
                "product_lut_ready": row.get("product_lut_ready", ""),
            }
        )
    return output


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def build_summary_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by([{k: str(v) for k, v in row.items()} for row in trace_rows], "slug")
    rows: list[dict[str, Any]] = []
    for slug, items in sorted(grouped.items()):
        first = items[0]
        runtime_values = [safe_float(row.get("runtime_response_nominal")) for row in items]
        sanity_values = [safe_float(row.get("cfa_times_si_simple_fraction")) for row in items]
        deltas = [abs(safe_float(row.get("runtime_vs_cfa_si_simple_delta"))) for row in items]
        center_rgb = [
            row
            for row in items
            if abs(safe_float(row.get("field_x_norm"), 999.0)) < 1e-12
            and abs(safe_float(row.get("field_z_norm"), 999.0)) < 1e-12
            and row.get("color_channel") in {"red", "green", "blue", "clear"}
        ]
        product_ready_count = sum(1 for row in items if boolish(row.get("product_lut_ready")))
        rows.append(
            {
                "slug": slug,
                "code": first.get("code", ""),
                "manufacturer": first.get("manufacturer", ""),
                "device_name": first.get("device_name", ""),
                "trace_row_count": len(items),
                "center_rgb_or_clear_row_count": len(center_rgb),
                "runtime_response_min": f"{min(runtime_values):.9g}" if runtime_values else "",
                "runtime_response_max": f"{max(runtime_values):.9g}" if runtime_values else "",
                "mean_runtime_response_nominal": f"{mean(runtime_values):.9g}" if runtime_values else "",
                "mean_cfa_times_si_simple_fraction": f"{mean(sanity_values):.9g}" if sanity_values else "",
                "max_runtime_vs_cfa_si_simple_abs_delta": f"{max(deltas):.9g}" if deltas else "",
                "evidence_gate_counts": json.dumps(dict(sorted(Counter(row.get("combined_evidence_gate", "") for row in items).items())), sort_keys=True),
                "confidence_class_counts": json.dumps(dict(sorted(Counter(row.get("confidence_class", "") for row in items).items())), sort_keys=True),
                "product_lut_ready_count": product_ready_count,
                "trace_product_gate": "PASS" if product_ready_count == len(items) and items else "FAIL",
                "primary_note": (
                    "CFA x Si simple fraction is a readable sanity check only; runtime response remains governed by FDTD/pass rows or research fallback gates."
                ),
            }
        )
    return rows


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 150) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], summary_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px}h1{margin:0 0 6px;font-size:30px}h2{margin-top:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}
.metric{font-size:26px;font-weight:800}.pass{color:#7dff9c}.fail{color:#ff8b8b}.warn{color:#ffd36e}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}
th{color:#9fe8ff;background:#0a1a22}code{color:#9fe8ff}
"""
    validation = payload.get("validation", {})
    status_class = "pass" if validation.get("pass") else "fail"
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CameraE2E Response Trace</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Response Trace</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This explains the runtime response rows; it does not upgrade proxy rows to product accuracy.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("trace_row_count", 0))}</div><div class="muted">trace rows</div></div>
    <div class="card"><div class="metric fail">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready rows</div></div>
  </div>
  <h2>Interpretation</h2>
  <p><code>cfa_times_si_simple_fraction</code> is computed from CFA transmission and a 1D Si absorption sanity formula. Use <code>runtime_response_nominal</code> for CameraE2E research runs, and keep product use blocked unless gates pass.</p>
  <h2>Sensor Summary</h2>
  {html_table(summary_rows, SUMMARY_COLUMNS)}
  <h2>Trace Rows</h2>
  {html_table(trace_rows, TRACE_COLUMNS)}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def update_package_links(package_dir: Path, payload: dict[str, Any]) -> None:
    package_json = package_dir / "camera_e2e_lut_package.json"
    package = read_json(package_json)
    if not package:
        return
    outputs = package.setdefault("outputs", {})
    outputs["camera_e2e_response_trace_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_response_trace_csv"] = payload["outputs"]["trace_csv"]
    outputs["camera_e2e_response_trace_summary_csv"] = payload["outputs"]["summary_csv"]
    outputs["camera_e2e_response_trace_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_response_trace"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "status": payload["validation"]["status"],
        "trace_row_count": payload["trace_row_count"],
        "product_ready_count": payload["product_ready_count"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def export_trace(package_dir: Path, output_dir: Path) -> dict[str, Any]:
    trace_rows = build_trace_rows(package_dir)
    summary_rows = build_summary_rows(trace_rows)
    product_ready_count = sum(1 for row in trace_rows if boolish(row.get("product_lut_ready")))
    checks = [
        check_row(
            "trace_rows_present",
            len(trace_rows) > 0,
            "PASS" if trace_rows else "FAIL",
            {"trace_rows": len(trace_rows)},
            "Regenerate runtime, material, and field response tables.",
        ),
        check_row(
            "summary_rows_present",
            len(summary_rows) > 0,
            "PASS" if summary_rows else "FAIL",
            {"summary_rows": len(summary_rows)},
            "Regenerate trace summary.",
        ),
        check_row(
            "product_gate_closed",
            product_ready_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_ready_count == 0 else "FAIL",
            {"product_ready_count": product_ready_count},
            "Only allow product response traces after measured/calibrated product gates pass.",
        ),
    ]
    validation_pass = all(boolish(row.get("pass")) for row in checks)
    validation = {
        "schema": "camera_e2e_response_trace_validation_v1",
        "pass": validation_pass,
        "status": "RESPONSE_TRACE_READY_PRODUCT_BLOCKED" if validation_pass else "FAIL",
        "issues": checks,
    }
    trace_csv = output_dir / "camera_e2e_response_trace.csv"
    summary_csv = output_dir / "camera_e2e_response_trace_summary.csv"
    json_path = output_dir / "camera_e2e_response_trace.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_response_trace_v1",
        "artifact_role": "pixel_response_calculation_trace_and_product_use_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(summary_rows),
        "trace_row_count": len(trace_rows),
        "product_ready_count": product_ready_count,
        "policy": {
            "runtime_response": "Use runtime_response_nominal for CameraE2E research mode according to gates.",
            "simple_sanity": "cfa_times_si_simple_fraction is a one-dimensional CFA x Si absorption sanity check, not Meep.",
            "product_use": "Blocked until measured material/stack, QE/CRA/crosstalk convergence, and electrical/readout calibration pass.",
        },
        "validation": validation,
        "outputs": {
            "json": repo_rel(json_path),
            "trace_csv": repo_rel(trace_csv),
            "summary_csv": repo_rel(summary_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(trace_csv, trace_rows, TRACE_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_json(json_path, {**payload, "summary_rows": summary_rows, "trace_rows": trace_rows})
    write_html(html_path, payload, summary_rows, trace_rows)
    update_package_links(package_dir, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = export_trace(args.package_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["validation"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
