"""Render physical sanity validation for FDTD-informed sensor LUTs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import (  # noqa: E402
    fdtd_sensor_cos4_relative_illumination,
    fdtd_sensor_default_lut_path,
    fdtd_sensor_lut_crosstalk_kernel,
    fdtd_sensor_lut_load,
    fdtd_sensor_physics_validate,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "fdtd_sensor"


def _fdtd_root() -> Path:
    return Path(os.environ.get("PYISETCAM_FDTD_ROOT", "/Users/seongcheoljeong/FDTD")).expanduser()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if value in {"", None}:
                    converted[key] = None
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
        return rows


def _status_class(status: str) -> str:
    return "ok" if status == "pass" else "bad" if status == "fail" else "warn"


def _render_ri_cos4(comparison_sets: list[tuple[str, list[dict[str, Any]]]], path: Path) -> None:
    labels: list[str] = []
    fdtd: list[float] = []
    cos4: list[float] = []
    for source, comparisons in comparison_sets:
        for item in comparisons:
            labels.append(f"{source}\n{item['case']}")
            fdtd.append(float(item["fdtd_response_norm"]))
            cos4.append(float(item["cos4_response"]))
    if not labels:
        labels = ["20 deg reference"]
        fdtd = [np.nan]
        cos4 = [fdtd_sensor_cos4_relative_illumination(20.0)]

    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(8.0, len(labels) * 1.4), 4.8), constrained_layout=True)
    axis.bar(x - 0.18, fdtd, width=0.36, label="FDTD normalized response", color="#2f6f88")
    axis.bar(x + 0.18, cos4, width=0.36, label="cos^4 CRA estimate", color="#b36b4f")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.15)
    axis.set_ylabel("Relative response")
    axis.set_title("Relative Illumination Sanity Check")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _render_ocl_pairs(pair_sets: list[tuple[str, list[dict[str, Any]]]], path: Path) -> None:
    labels: list[str] = []
    uncomp: list[float] = []
    comp: list[float] = []
    colors: list[str] = []
    for source, pairs in pair_sets:
        for pair in pairs:
            labels.append(source)
            uncomp.append(float(pair["uncompensated_response"]))
            comp.append(float(pair["compensated_response"]))
            colors.append("#2f8f62" if float(pair["improvement"]) > 0.0 else "#b34f4f")
    if not labels:
        labels = ["no OCL pair"]
        uncomp = [0.0]
        comp = [0.0]
        colors = ["#999999"]

    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(7.5, len(labels) * 1.5), 4.8), constrained_layout=True)
    axis.bar(x - 0.18, uncomp, width=0.36, label="uncompensated", color="#6b8fb3")
    axis.bar(x + 0.18, comp, width=0.36, label="compensated / OCL", color=colors)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("Normalized response")
    axis.set_title("OCL Shift Direction Check")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _render_wavelength_absorption(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    if not rows:
        return {"available": False, "monotonic_decreasing": None}
    wave = np.asarray([float(row["wavelength_nm"]) for row in rows], dtype=float)
    absorption = np.asarray([float(row["si_absorption_fraction_estimate"]) for row in rows], dtype=float)
    order = np.argsort(wave)
    wave = wave[order]
    absorption = absorption[order]
    monotonic = bool(np.all(np.diff(absorption) <= 1e-9))

    fig, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    axis.plot(wave, absorption, marker="o", color="#2f6f88")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Si absorption fraction estimate")
    axis.set_title("Silicon Absorption Wavelength Trend")
    axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "available": True,
        "monotonic_decreasing": monotonic,
        "wavelengths_nm": wave.tolist(),
        "absorption": absorption.tolist(),
    }


def _render_energy_budget(lut, path: Path) -> dict[str, Any]:
    rows = lut.long_rows if lut.long_rows else lut.summary_rows
    incident = []
    absorption = []
    response = []
    for row in rows:
        inc = row.get("incident_monitor_net_power_normalized")
        abs_value = row.get("total_si_absorption_fraction_estimate", row.get("si_absorption_fraction_estimate"))
        resp = row.get("response", row.get("total_response"))
        if inc is None or abs_value is None:
            continue
        incident.append(float(inc))
        absorption.append(float(abs_value))
        response.append(float(resp if resp is not None else abs_value))
    if not incident:
        return {"available": False}

    x = np.arange(len(incident))
    fig, axis = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
    axis.plot(x, incident, marker="o", label="incident monitor", color="#5b6770")
    axis.plot(x, absorption, marker="s", label="Si absorption", color="#2f6f88")
    axis.plot(x, response, marker="^", label="regional/total response", color="#b36b4f")
    axis.set_xlabel("LUT row index")
    axis.set_ylabel("Normalized power / response")
    axis.set_title("Energy Budget Sanity Check")
    axis.grid(alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "available": True,
        "max_incident": float(np.max(incident)),
        "max_absorption": float(np.max(absorption)),
        "max_response": float(np.max(response)),
        "absorption_over_incident": int(np.sum(np.asarray(absorption) > np.asarray(incident) + 1e-9)),
    }


def _render_kernel_locality(lut, path: Path) -> dict[str, Any]:
    kernel = fdtd_sensor_lut_crosstalk_kernel(lut, case="center", wavelength_nm=550.0)
    fig, axis = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    im = axis.imshow(kernel, cmap="viridis")
    axis.set_title("Crosstalk Proxy Locality")
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            axis.text(col, row, f"{kernel[row, col]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    mean = float(np.mean(kernel))
    return {
        "kernel_shape": list(kernel.shape),
        "center_weight": float(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]),
        "uniformity_cv": float(np.std(kernel) / max(abs(mean), 1e-12)),
    }


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f5ef; color: #1f2933; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 34px 24px 56px; }}
    h1 {{ font-size: 40px; margin: 0 0 10px; }}
    h2 {{ margin-top: 32px; border-bottom: 2px solid #ded6c8; padding-bottom: 8px; }}
    .card {{ background: #fff; border: 1px solid #e3dccf; border-radius: 18px; padding: 20px; margin: 18px 0; box-shadow: 0 8px 24px rgba(35, 44, 50, 0.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 14px; }}
    .metric {{ border-radius: 14px; padding: 14px; background: #eef6f8; }}
    .metric b {{ display: block; font-size: 13px; color: #52616b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric span {{ font-size: 24px; font-weight: 700; }}
    .ok {{ border-left: 5px solid #2f8f62; }}
    .warn {{ border-left: 5px solid #b8792f; background: #fff8ec; }}
    .bad {{ border-left: 5px solid #b34f4f; background: #fff1f1; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #ece6db; vertical-align: top; }}
    th {{ background: #f0ebe1; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid #e2ded5; background: #fff; }}
    code {{ background: #eee7da; padding: 2px 5px; border-radius: 5px; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _verdict_table(primary_validation: dict[str, Any], cra_validation: dict[str, Any] | None) -> str:
    rows = []
    for name, check in primary_validation["checks"].items():
        rows.append(("primary " + name, check["status"], "; ".join(check.get("failures", []) + check.get("warnings", [])) or "none"))
    if cra_validation is not None:
        rows.append(
            (
                "CRA LUT OCL shift",
                cra_validation["checks"]["ocl_shift"]["status"],
                "; ".join(
                    cra_validation["checks"]["ocl_shift"].get("failures", [])
                    + cra_validation["checks"]["ocl_shift"].get("warnings", [])
                )
                or "none",
            )
        )
    body = "".join(
        f"<tr><th>{escape(name)}</th><td class='{_status_class(status)}'>{escape(status.upper())}</td><td>{escape(detail)}</td></tr>"
        for name, status, detail in rows
    )
    return f"<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>{body}</tbody></table>"


def render_report(output_dir: Path = DEFAULT_OUTPUT_DIR, lut_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = _fdtd_root()
    primary_path = lut_path or fdtd_sensor_default_lut_path()
    if primary_path is None:
        raise FileNotFoundError("No primary FDTD camera LUT found. Pass --lut or set PYISETCAM_FDTD_ROOT.")
    primary_lut = fdtd_sensor_lut_load(primary_path)
    primary_validation = fdtd_sensor_physics_validate(primary_lut)

    cra_path = root / "runs" / "cra_response_lut" / "cra_response_lut.csv"
    cra_lut = fdtd_sensor_lut_load(cra_path) if cra_path.exists() else None
    cra_validation = fdtd_sensor_physics_validate(cra_lut) if cra_lut is not None else None

    absorption_rows = _read_csv(root / "runs" / "meep_pixel_absorption_sweep.csv")

    ri_png = output_dir / "physics_ri_cos4.png"
    ocl_png = output_dir / "physics_ocl_shift.png"
    wavelength_png = output_dir / "physics_wavelength_absorption.png"
    energy_png = output_dir / "physics_energy_budget.png"
    kernel_png = output_dir / "physics_kernel_locality.png"

    comparison_sets = [("3x3 smoke", primary_validation["checks"]["relative_illumination"]["comparisons"])]
    if cra_validation is not None:
        comparison_sets.append(("CRA LUT", cra_validation["checks"]["relative_illumination"]["comparisons"]))
    _render_ri_cos4(comparison_sets, ri_png)

    pair_sets = [("3x3 smoke", primary_validation["checks"]["ocl_shift"]["pairs"])]
    if cra_validation is not None:
        pair_sets.append(("CRA LUT", cra_validation["checks"]["ocl_shift"]["pairs"]))
    _render_ocl_pairs(pair_sets, ocl_png)

    wavelength_summary = _render_wavelength_absorption(absorption_rows, wavelength_png)
    energy_summary = _render_energy_budget(primary_lut, energy_png)
    kernel_summary = _render_kernel_locality(primary_lut, kernel_png)

    summary = {
        "report": str(output_dir / "physics_validation_report.html"),
        "primary_lut": str(primary_lut.source_path),
        "cra_lut": None if cra_lut is None else str(cra_lut.source_path),
        "primary_validation": primary_validation,
        "cra_validation": cra_validation,
        "wavelength_absorption": wavelength_summary,
        "energy_budget": energy_summary,
        "kernel_locality": kernel_summary,
        "figures": {
            "ri_cos4": str(ri_png),
            "ocl_shift": str(ocl_png),
            "wavelength_absorption": str(wavelength_png),
            "energy_budget": str(energy_png),
            "kernel_locality": str(kernel_png),
        },
    }
    (output_dir / "physics_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary_status = primary_validation["status"]
    cra_ocl_status = "missing" if cra_validation is None else cra_validation["checks"]["ocl_shift"]["status"]
    wavelength_status = "pass" if wavelength_summary.get("monotonic_decreasing") else "warn"
    body = f"""
<h1>FDTD Sensor Physics Validation Report</h1>
<p>This report checks whether the FDTD-informed sensor outputs are physically plausible, not just whether software integration runs.</p>
<div class="card {_status_class(primary_status)}">
  <h2>Executive Verdict</h2>
  <div class="grid">
    <div class="metric"><b>Primary LUT physics</b><span>{escape(primary_status.upper())}</span></div>
    <div class="metric"><b>3x3 OCL shift</b><span>{escape(primary_validation['checks']['ocl_shift']['status'].upper())}</span></div>
    <div class="metric"><b>CRA LUT OCL shift</b><span>{escape(cra_ocl_status.upper())}</span></div>
    <div class="metric"><b>Wavelength trend</b><span>{escape(wavelength_status.upper())}</span></div>
  </div>
  <p><b>Interpretation:</b> the current 3x3 smoke LUT is good enough to verify the data path, but not good enough to claim product-grade physical crosstalk or OCL improvement. The separate CRA LUT shows the expected compensation improvement, while the 3x3 smoke LUT does not.</p>
</div>
<div class="card">
  <h2>Source Data</h2>
  <table><tbody>
    <tr><th>Primary LUT</th><td><code>{escape(str(primary_lut.source_path))}</code></td></tr>
    <tr><th>CRA LUT</th><td><code>{escape(str(cra_path)) if cra_lut is not None else 'missing'}</code></td></tr>
    <tr><th>Absorption sweep</th><td><code>{escape(str(root / 'runs' / 'meep_pixel_absorption_sweep.csv'))}</code></td></tr>
  </tbody></table>
</div>
<div class="card">
  <h2>Validation Table</h2>
  {_verdict_table(primary_validation, cra_validation)}
</div>
<div class="card">
  <h2>Relative Illumination vs cos^4</h2>
  <p>The cos^4 curve is a first-order reference. FDTD may be lower because of stack, aperture, and absorption losses, but large excess over cos^4 or unexplained severe loss should be investigated.</p>
  <img src="{ri_png.name}" alt="Relative illumination cos4 comparison">
</div>
<div class="card">
  <h2>OCL Shift Direction</h2>
  <p>A correct OCL compensation should usually improve edge response versus uncompensated geometry. The current 3x3 smoke LUT fails this direction check, while the CRA response LUT passes.</p>
  <img src="{ocl_png.name}" alt="OCL shift comparison">
</div>
<div class="card">
  <h2>Wavelength Trend</h2>
  <p>For the simple pixel stack sweep, Si absorption decreases from blue to red in the available 450/550/650 nm data. This is directionally plausible.</p>
  <img src="{wavelength_png.name}" alt="Wavelength absorption trend">
</div>
<div class="card">
  <h2>Energy Budget</h2>
  <p>Responses should be non-negative and bounded by incident/absorbed power conventions. This plot exposes gross conservation failures.</p>
  <img src="{energy_png.name}" alt="Energy budget">
</div>
<div class="card">
  <h2>Crosstalk Locality</h2>
  <p>The current 3x3 smoke regional-response kernel is nearly uniform. That is a warning: it should not be treated as a localized crosstalk PSF without a better source/collection experiment.</p>
  <img src="{kernel_png.name}" alt="Crosstalk locality">
</div>
<div class="card warn">
  <h2>Required Improvement Before Product Use</h2>
  <p>Generate a product-grade LUT with convergence sweeps, signed CRA pairs, field/corner sweep, wavelength sweep, and a localized optical crosstalk source experiment. If photodiode collection physics is required, add TCAD or a calibrated compact collection model.</p>
</div>
"""
    html_path = output_dir / "physics_validation_report.html"
    html_path.write_text(_html_page("FDTD Sensor Physics Validation Report", body), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lut", type=Path, default=None)
    args = parser.parse_args()
    summary = render_report(args.output_dir, args.lut)
    print(summary["report"])


if __name__ == "__main__":
    main()
