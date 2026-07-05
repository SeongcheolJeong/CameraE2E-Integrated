#!/usr/bin/env python3
"""Export CameraE2E uncertainty budget rows.

This artifact makes the current error budget explicit for CameraE2E consumers.
It is not a measured confidence interval. It is a conservative engineering
budget derived from the source class of each table: proxy material, low-res
solver evidence, prior electrical/readout seed, or measured/product-ready data.
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


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package"
DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "camera_e2e_uncertainty_budget"

SENSOR_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "camera_e2e_use",
    "research_utility_grade_0_10",
    "solver_evidence_grade_0_10",
    "product_accuracy_grade_0_10",
    "material_ri_n_uncertainty_pct_min",
    "material_ri_n_uncertainty_pct_max",
    "cfa_k_transmission_uncertainty_pct_min",
    "cfa_k_transmission_uncertainty_pct_max",
    "qe_absolute_uncertainty_pct_min",
    "qe_absolute_uncertainty_pct_max",
    "cra_edge_response_uncertainty_pct_min",
    "cra_edge_response_uncertainty_pct_max",
    "optical_crosstalk_uncertainty_pct_min",
    "optical_crosstalk_uncertainty_pct_max",
    "conversion_gain_fwc_uncertainty_pct_min",
    "conversion_gain_fwc_uncertainty_pct_max",
    "temporal_noise_uncertainty_pct_min",
    "temporal_noise_uncertainty_pct_max",
    "dark_current_uncertainty_factor_min",
    "dark_current_uncertainty_factor_max",
    "dsnu_prnu_uncertainty_pct_min",
    "dsnu_prnu_uncertainty_pct_max",
    "readout_raw_uncertainty_pct_min",
    "readout_raw_uncertainty_pct_max",
    "module_coupling_uncertainty_pct_min",
    "module_coupling_uncertainty_pct_max",
    "uncertainty_product_gate",
    "primary_blockers",
    "recommended_next_action",
]

DOMAIN_COLUMNS = [
    "slug",
    "code",
    "manufacturer",
    "device_name",
    "domain",
    "quantity",
    "uncertainty_type",
    "uncertainty_min",
    "uncertainty_max",
    "unit",
    "evidence_class",
    "camera_e2e_use",
    "product_lut_gate",
    "basis",
    "blocker",
    "recommended_next_action",
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key)}


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return dict(grouped)


def compact(values: list[str], limit: int = 8) -> str:
    output: list[str] = []
    for value in values:
        for part in str(value or "").split(";"):
            clean = part.strip()
            if clean and clean not in output:
                output.append(clean)
    return "; ".join(output[:limit])


def check_row(check_id: str, passed: bool, status: str, evidence: Any, action: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "pass": passed,
        "status": status,
        "evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True) if isinstance(evidence, (dict, list)) else evidence,
        "required_action": action,
    }


def domain_row(
    sensor: dict[str, str],
    *,
    domain: str,
    quantity: str,
    uncertainty_type: str,
    low: float,
    high: float,
    unit: str,
    evidence_class: str,
    use: str,
    product_gate: str,
    basis: str,
    blocker: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "slug": sensor.get("slug", ""),
        "code": sensor.get("code", ""),
        "manufacturer": sensor.get("manufacturer", ""),
        "device_name": sensor.get("device_name", ""),
        "domain": domain,
        "quantity": quantity,
        "uncertainty_type": uncertainty_type,
        "uncertainty_min": f"{low:g}",
        "uncertainty_max": f"{high:g}",
        "unit": unit,
        "evidence_class": evidence_class,
        "camera_e2e_use": use,
        "product_lut_gate": product_gate,
        "basis": basis,
        "blocker": blocker,
        "recommended_next_action": next_action,
    }


def product_or_proxy_gate(*gates: str) -> str:
    normalized = [str(g or "").upper() for g in gates]
    if normalized and all(gate == "PASS" for gate in normalized):
        return "PASS"
    if any(gate in {"FAIL", "MISSING", ""} for gate in normalized):
        return "FAIL"
    return "CHECK"


def material_evidence(material: dict[str, str], material_rows: list[dict[str, str]]) -> dict[str, Any]:
    measured_count = safe_int(material.get("measured_material_count"))
    total = safe_int(material.get("material_row_count"), len(material_rows))
    product_gate = material.get("product_lut_gate", "FAIL")
    families = {row.get("material_family", "") for row in material_rows}
    if product_gate == "PASS" and measured_count >= total > 0:
        return {
            "ri": (0.5, 2.0, "measured_or_product_material"),
            "cfa": (5.0, 10.0, "measured_or_product_cfa"),
            "stack": (1.0, 3.0, "measured_stack_geometry"),
        }
    ri_high = 8.0 if any("cfa" in family for family in families) else 5.0
    cfa_high = 50.0 if safe_int(material.get("cfa_proxy_row_count")) > 0 else 25.0
    return {
        "ri": (0.5, ri_high, "proxy_mixed_material_nk"),
        "cfa": (10.0, cfa_high, "proxy_cfa_k_transmission"),
        "stack": (5.0, 20.0, "proxy_stack_geometry"),
    }


def qe_uncertainty(trust: dict[str, str], runtime_rows: list[dict[str, str]]) -> tuple[float, float, str]:
    if safe_float(trust.get("product_accuracy_grade_0_10")) >= 8.0:
        return (5.0, 10.0, "product_calibrated_qe")
    gates = {row.get("combined_evidence_gate", "") for row in runtime_rows}
    if "PASS" in gates:
        return (10.0, 25.0, "partial_quantitative_solver_qe")
    if runtime_rows:
        return (20.0, 50.0, "proxy_or_low_res_qe_trend")
    return (50.0, 100.0, "missing_qe_response")


def cra_uncertainty(module: dict[str, str], runtime_rows: list[dict[str, str]]) -> tuple[float, float, str]:
    if module.get("product_lut_gate") == "PASS":
        return (5.0, 15.0, "measured_module_cra")
    gates = {row.get("cra_mismatch_gate", "") for row in runtime_rows}
    if "PASS" in gates:
        return (10.0, 25.0, "partial_cra_match_context")
    return (20.0, 60.0, "assumed_cra_and_ocl_shift_prior")


def crosstalk_uncertainty(support_rows: list[dict[str, str]], runtime_rows: list[dict[str, str]]) -> tuple[float, float, str, str]:
    if not support_rows:
        if runtime_rows:
            return (50.0, 100.0, "compact_surrogate_without_finite_array_support", "No finite-array support pilot exists for this sensor.")
        return (75.0, 150.0, "missing_crosstalk_kernel", "No crosstalk rows exist for this sensor.")
    best = min((safe_float(row.get("best_pilot_truncation_fraction"), 1e9) for row in support_rows), default=1e9)
    threshold = min((safe_float(row.get("truncation_threshold"), 0.015) for row in support_rows), default=0.015)
    if best <= threshold:
        return (
            20.0,
            60.0,
            "low_res_finite_array_support_truncation_met",
            "Low-res finite-array support meets truncation threshold, but product resolution convergence is missing.",
        )
    return (
        30.0,
        80.0,
        "low_res_finite_array_support_truncation_not_met",
        "Low-res finite-array support still exceeds truncation threshold.",
    )


def electrical_uncertainty(electrical: dict[str, str]) -> dict[str, tuple[float, float, str]]:
    if electrical.get("product_lut_gate") == "PASS":
        return {
            "cg_fwc": (2.0, 5.0, "measured_electrical_calibration"),
            "temporal": (5.0, 15.0, "measured_noise_calibration"),
            "dark_factor": (1.2, 2.0, "measured_dark_curve"),
            "dsnu_prnu": (5.0, 20.0, "measured_fpn_statistics"),
            "readout": (2.0, 10.0, "measured_readout_table"),
        }
    return {
        "cg_fwc": (20.0, 50.0, "prior_seed_not_measured"),
        "temporal": (30.0, 100.0, "prior_seed_not_measured"),
        "dark_factor": (2.0, 10.0, "prior_seed_not_measured"),
        "dsnu_prnu": (30.0, 100.0, "prior_seed_not_measured"),
        "readout": (20.0, 50.0, "prior_seed_not_measured"),
    }


def module_uncertainty(module: dict[str, str]) -> tuple[float, float, str]:
    if module.get("product_lut_gate") == "PASS":
        return (5.0, 15.0, "raytrace_or_measured_module")
    return (20.0, 60.0, "module_field_prior_not_measured")


def build_rows(package_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    material_summary = index_by(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_summary.csv"), "slug")
    material_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_material_tables" / "camera_e2e_material_nk_lut.csv"), "slug")
    electrical_summary = index_by(
        read_csv_rows(package_dir / "camera_e2e_electrical_readout_tables" / "camera_e2e_electrical_readout_summary.csv"), "slug"
    )
    module_summary = index_by(read_csv_rows(package_dir / "camera_e2e_module_coupling" / "camera_e2e_module_coupling_summary.csv"), "slug")
    trust_summary = index_by(read_csv_rows(package_dir / "camera_e2e_lut_trust_assessment" / "camera_e2e_lut_trust_by_sensor.csv"), "slug")
    runtime_by_slug = group_by(read_csv_rows(package_dir / "camera_e2e_runtime_bundle" / "camera_e2e_runtime_lut.csv"), "slug")
    support_by_slug = group_by(
        read_csv_rows(package_dir / "camera_e2e_crosstalk_support_audit" / "camera_e2e_crosstalk_support_by_sensor.csv"), "slug"
    )

    slugs = sorted(set(material_summary) | set(electrical_summary) | set(module_summary) | set(trust_summary) | set(runtime_by_slug))
    sensor_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for slug in slugs:
        material = material_summary.get(slug, {})
        electrical = electrical_summary.get(slug, {})
        module = module_summary.get(slug, {})
        trust = trust_summary.get(slug, {})
        runtime_rows = runtime_by_slug.get(slug, [])
        support_rows = support_by_slug.get(slug, [])
        sensor = {**material, **electrical, **module, **trust, "slug": slug}
        for source in (trust, material, electrical, module):
            for key in ("code", "manufacturer", "device_name"):
                if not sensor.get(key) and source.get(key):
                    sensor[key] = source[key]

        mat = material_evidence(material, material_by_slug.get(slug, []))
        qe = qe_uncertainty(trust, runtime_rows)
        cra = cra_uncertainty(module, runtime_rows)
        xtalk = crosstalk_uncertainty(support_rows, runtime_rows)
        elec = electrical_uncertainty(electrical)
        module_unc = module_uncertainty(module)

        material_blocker = material.get("primary_blocker", "material n,k and stack geometry are not measured")
        electrical_blocker = electrical.get("primary_blocker", "electrical/noise values are prior seeds")
        module_blocker = module.get("primary_blocker", "module raytrace/measured CRA data missing")
        crosstalk_blocker = compact([row.get("primary_blockers", "") for row in support_rows]) or xtalk[3]
        next_action = (
            "Use these bands for sensitivity analysis only; replace with measured stack/n,k, high-res convergence, "
            "QE/CRA/crosstalk calibration, and measured electrical/readout/noise tables before product use."
        )

        ri_low, ri_high, ri_class = mat["ri"]
        cfa_low, cfa_high, cfa_class = mat["cfa"]
        stack_low, stack_high, stack_class = mat["stack"]
        qe_low, qe_high, qe_class = qe
        cra_low, cra_high, cra_class = cra
        xt_low, xt_high, xt_class, xt_basis = xtalk
        cg_low, cg_high, cg_class = elec["cg_fwc"]
        tn_low, tn_high, tn_class = elec["temporal"]
        dark_low, dark_high, dark_class = elec["dark_factor"]
        dp_low, dp_high, dp_class = elec["dsnu_prnu"]
        ro_low, ro_high, ro_class = elec["readout"]
        mod_low, mod_high, mod_class = module_unc

        product_gate = product_or_proxy_gate(
            material.get("product_lut_gate", "FAIL"),
            electrical.get("product_lut_gate", "FAIL"),
            module.get("product_lut_gate", "FAIL"),
            trust.get("product_ready", "FAIL"),
        )
        use = (
            "research_sensitivity_only"
            if product_gate != "PASS"
            else "calibrated_camera_e2e_use_allowed"
        )

        domain_specs = [
            ("Optical / Color", "material RI n", "relative_error", ri_low, ri_high, "pct", ri_class, material.get("product_lut_gate", "FAIL"), "Material family mix from FDTD n,k table.", material_blocker),
            ("Optical / Color", "CFA k / transmission", "relative_error", cfa_low, cfa_high, "pct", cfa_class, material.get("product_lut_gate", "FAIL"), "CFA proxy/source class and absorption-only transmission rows.", material_blocker),
            ("Optical / Color", "stack thickness / geometry", "relative_error", stack_low, stack_high, "pct", stack_class, material.get("product_lut_gate", "FAIL"), "Measured stack flag and source kind.", material_blocker),
            ("Optical / Color", "spectral QE / response", "relative_error", qe_low, qe_high, "pct", qe_class, trust.get("product_ready", "FAIL"), "Runtime response LUT evidence gate.", trust.get("primary_blockers", "")),
            ("Optical / Color", "CRA edge response / color shading", "relative_error", cra_low, cra_high, "pct", cra_class, module.get("product_lut_gate", "FAIL"), "Module CRA and runtime CRA mismatch gates.", module_blocker),
            ("Optical / Color", "microlens/OCL shift map", "response_equivalent_error", cra_low, cra_high, "pct", cra_class, module.get("product_lut_gate", "FAIL"), "OCL shift is derived from CRA prior unless measured shift map exists.", module_blocker),
            ("Optical / Color", "optical crosstalk kernel", "relative_error", xt_low, xt_high, "pct", xt_class, "FAIL", xt_basis, crosstalk_blocker),
            ("Pixel / Electrical", "conversion gain / full well / saturation", "relative_error", cg_low, cg_high, "pct", cg_class, electrical.get("product_lut_gate", "FAIL"), "Prior seed electrical model summary.", electrical_blocker),
            ("Pixel / Electrical", "temporal/read/reset/SF/ADC noise", "relative_error", tn_low, tn_high, "pct", tn_class, electrical.get("product_lut_gate", "FAIL"), "Noise rows are prior seed values; shot noise remains signal-derived.", electrical_blocker),
            ("Pixel / Electrical", "dark current vs temperature/exposure", "multiplicative_error", dark_low, dark_high, "factor", dark_class, electrical.get("product_lut_gate", "FAIL"), "Dark-current rows are prior seed temperature curves.", electrical_blocker),
            ("Pixel / Electrical", "DSNU / PRNU", "relative_error", dp_low, dp_high, "pct", dp_class, electrical.get("product_lut_gate", "FAIL"), "Fixed-pattern variation requires measured frame statistics.", electrical_blocker),
            ("Readout / RAW", "gain / black / ADC / row-column FPN", "relative_error", ro_low, ro_high, "pct", ro_class, electrical.get("product_lut_gate", "FAIL"), "Readout table is a prior seed, not measured register calibration.", electrical_blocker),
            ("Module Coupling", "field CRA / vignetting / pupil / assembly", "relative_error", mod_low, mod_high, "pct", mod_class, module.get("product_lut_gate", "FAIL"), "Module coupling is raytrace/measurement prior status.", module_blocker),
        ]
        for domain, quantity, kind, low, high, unit, evidence_class, gate, basis, blocker in domain_specs:
            domain_rows.append(
                domain_row(
                    sensor,
                    domain=domain,
                    quantity=quantity,
                    uncertainty_type=kind,
                    low=low,
                    high=high,
                    unit=unit,
                    evidence_class=evidence_class,
                    use=use,
                    product_gate=gate if str(gate).upper() == "PASS" else "FAIL",
                    basis=basis,
                    blocker=blocker,
                    next_action=next_action,
                )
            )

        blockers = compact([material_blocker, electrical_blocker, module_blocker, crosstalk_blocker, trust.get("primary_blockers", "")])
        sensor_rows.append(
            {
                "slug": slug,
                "code": sensor.get("code", ""),
                "manufacturer": sensor.get("manufacturer", ""),
                "device_name": sensor.get("device_name", ""),
                "camera_e2e_use": use,
                "research_utility_grade_0_10": trust.get("research_utility_grade_0_10", ""),
                "solver_evidence_grade_0_10": trust.get("solver_evidence_grade_0_10", ""),
                "product_accuracy_grade_0_10": trust.get("product_accuracy_grade_0_10", "0.00"),
                "material_ri_n_uncertainty_pct_min": ri_low,
                "material_ri_n_uncertainty_pct_max": ri_high,
                "cfa_k_transmission_uncertainty_pct_min": cfa_low,
                "cfa_k_transmission_uncertainty_pct_max": cfa_high,
                "qe_absolute_uncertainty_pct_min": qe_low,
                "qe_absolute_uncertainty_pct_max": qe_high,
                "cra_edge_response_uncertainty_pct_min": cra_low,
                "cra_edge_response_uncertainty_pct_max": cra_high,
                "optical_crosstalk_uncertainty_pct_min": xt_low,
                "optical_crosstalk_uncertainty_pct_max": xt_high,
                "conversion_gain_fwc_uncertainty_pct_min": cg_low,
                "conversion_gain_fwc_uncertainty_pct_max": cg_high,
                "temporal_noise_uncertainty_pct_min": tn_low,
                "temporal_noise_uncertainty_pct_max": tn_high,
                "dark_current_uncertainty_factor_min": dark_low,
                "dark_current_uncertainty_factor_max": dark_high,
                "dsnu_prnu_uncertainty_pct_min": dp_low,
                "dsnu_prnu_uncertainty_pct_max": dp_high,
                "readout_raw_uncertainty_pct_min": ro_low,
                "readout_raw_uncertainty_pct_max": ro_high,
                "module_coupling_uncertainty_pct_min": mod_low,
                "module_coupling_uncertainty_pct_max": mod_high,
                "uncertainty_product_gate": product_gate,
                "primary_blockers": blockers,
                "recommended_next_action": next_action,
            }
        )

    return sensor_rows, domain_rows


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], sensor_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]]) -> None:
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
<title>CameraE2E Uncertainty Budget</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>CameraE2E Uncertainty Budget</h1>
  <p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. These are conservative engineering budgets, not measured statistical confidence intervals.</p>
  <div class="grid">
    <div class="card"><div class="metric {status_class}">{html_cell(validation.get("status", ""))}</div><div class="muted">validation</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("sensor_count", 0))}</div><div class="muted">sensors</div></div>
    <div class="card"><div class="metric">{html_cell(payload.get("domain_row_count", 0))}</div><div class="muted">domain budgets</div></div>
    <div class="card"><div class="metric fail">{html_cell(payload.get("product_ready_count", 0))}</div><div class="muted">product-ready budgets</div></div>
  </div>
  <h2>How To Use</h2>
  <p>Use these rows to drive CameraE2E sensitivity bands, stochastic sweeps, and UI warnings. Do not treat them as product calibration data.</p>
  <h2>Sensor Summary</h2>
  {html_table(sensor_rows, SENSOR_COLUMNS)}
  <h2>Domain Budget Rows</h2>
  {html_table(domain_rows, DOMAIN_COLUMNS)}
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
    outputs["camera_e2e_uncertainty_budget_json"] = payload["outputs"]["json"]
    outputs["camera_e2e_uncertainty_by_sensor_csv"] = payload["outputs"]["sensor_csv"]
    outputs["camera_e2e_uncertainty_budget_csv"] = payload["outputs"]["domain_csv"]
    outputs["camera_e2e_uncertainty_budget_html"] = payload["outputs"]["html"]
    package["latest_camera_e2e_uncertainty_budget"] = {
        "schema": payload["schema"],
        "validation_pass": payload["validation"]["pass"],
        "status": payload["validation"]["status"],
        "sensor_count": payload["sensor_count"],
        "product_ready_count": payload["product_ready_count"],
        **payload["outputs"],
    }
    write_json(package_json, package)


def export_budget(package_dir: Path, output_dir: Path) -> dict[str, Any]:
    sensor_rows, domain_rows = build_rows(package_dir)
    product_ready_count = sum(1 for row in sensor_rows if str(row.get("uncertainty_product_gate", "")).upper() == "PASS")
    issues = []
    if not sensor_rows:
        issues.append(check_row("sensor_rows_present", False, "FAIL", {"sensor_rows": 0}, "Regenerate upstream CameraE2E tables."))
    else:
        issues.append(check_row("sensor_rows_present", True, "PASS", {"sensor_rows": len(sensor_rows)}, ""))
    if len(domain_rows) < len(sensor_rows) * 10:
        issues.append(
            check_row(
                "domain_budget_coverage",
                False,
                "FAIL",
                {"domain_rows": len(domain_rows), "sensor_rows": len(sensor_rows)},
                "Add missing uncertainty domains.",
            )
        )
    else:
        issues.append(
            check_row(
                "domain_budget_coverage",
                True,
                "PASS",
                {"domain_rows": len(domain_rows), "sensor_rows": len(sensor_rows)},
                "",
            )
        )
    issues.append(
        check_row(
            "product_gate_closed",
            product_ready_count == 0,
            "PRODUCT_BLOCKED_AS_EXPECTED" if product_ready_count == 0 else "FAIL",
            {"product_ready_count": product_ready_count},
            "Only allow product uncertainty rows after measured/calibrated product gates pass.",
        )
    )
    validation_pass = all(boolish(row.get("pass")) for row in issues)
    validation = {
        "schema": "camera_e2e_uncertainty_budget_validation_v1",
        "pass": validation_pass,
        "status": "UNCERTAINTY_BUDGET_READY_PRODUCT_BLOCKED" if validation_pass else "FAIL",
        "issues": issues,
    }

    sensor_csv = output_dir / "camera_e2e_uncertainty_by_sensor.csv"
    domain_csv = output_dir / "camera_e2e_uncertainty_budget.csv"
    json_path = output_dir / "camera_e2e_uncertainty_budget.json"
    html_path = output_dir / "index.html"
    payload = {
        "schema": "camera_e2e_uncertainty_budget_v1",
        "artifact_role": "camera_e2e_error_budget_and_product_use_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": repo_rel(package_dir),
        "sensor_count": len(sensor_rows),
        "domain_row_count": len(domain_rows),
        "product_ready_count": product_ready_count,
        "policy": {
            "meaning": "Conservative engineering uncertainty bands from source class and gate status.",
            "not_a_statistical_ci": True,
            "product_use": "Blocked until measured material/stack, QE/CRA/crosstalk convergence, and electrical/readout calibration pass.",
        },
        "validation": validation,
        "outputs": {
            "json": repo_rel(json_path),
            "sensor_csv": repo_rel(sensor_csv),
            "domain_csv": repo_rel(domain_csv),
            "html": repo_rel(html_path),
        },
    }
    write_csv(sensor_csv, sensor_rows, SENSOR_COLUMNS)
    write_csv(domain_csv, domain_rows, DOMAIN_COLUMNS)
    write_json(json_path, {**payload, "sensor_rows": sensor_rows, "domain_rows": domain_rows})
    write_html(html_path, payload, sensor_rows, domain_rows)
    update_package_links(package_dir, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = export_budget(args.package_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["validation"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
