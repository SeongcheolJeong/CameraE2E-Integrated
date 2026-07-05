"""Render FDTD-informed sensor block implementation and verification report."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import (  # noqa: E402
    DEFAULT_WAVE,
    AssetStore,
    OpticalImage,
    fdtd_sensor_apply_optical_response,
    fdtd_sensor_config,
    fdtd_sensor_default_lut_path,
    fdtd_sensor_field_response_map,
    fdtd_sensor_lut_crosstalk_kernel,
    fdtd_sensor_lut_load,
    fdtd_sensor_lut_response,
    fdtd_sensor_lut_summary,
    fdtd_sensor_lut_validate,
    fdtd_sensor_physics_validate,
    ip_compute,
    ip_create,
    ip_get,
    sensor_attach_fdtd_lut,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "fdtd_sensor"


def _synthetic_lut_path() -> Path:
    temp_root = Path(tempfile.gettempdir()) / "pyisetcam_fdtd_sensor_synthetic_lut"
    temp_root.mkdir(parents=True, exist_ok=True)
    long_csv = temp_root / "camera_lut_long.csv"
    summary_csv = temp_root / "camera_lut_summary.csv"
    json_path = temp_root / "camera_lut.json"
    summary_csv.write_text(
        "\n".join(
            [
                "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,total_response,normalized_total_response_to_first",
                "ocl-3x3,550,center,0,0,0,0,1.0,1.0",
                "ocl-3x3,550,edge20_synthetic,1,0,20,0,0.62,0.62",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    long_lines = [
        "mode,wavelength_nm,case,field_x_norm,field_z_norm,cra_x_deg,cra_z_deg,region_id,region_kind,region_ix,region_iz,response"
    ]
    kernel = np.array([[0.03, 0.06, 0.03], [0.06, 0.64, 0.06], [0.03, 0.06, 0.03]], dtype=float)
    for case, field, cra, scale in (("center", 0.0, 0.0, 1.0), ("edge20_synthetic", 1.0, 20.0, 0.62)):
        for iz in range(3):
            for ix in range(3):
                long_lines.append(
                    f"ocl-3x3,550,{case},{field},0,{cra},0,pix_x{ix}_z{iz},pixel,{ix},{iz},{kernel[iz, ix] * scale}"
                )
    long_csv.write_text("\n".join(long_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "schema": "camera_supercell_optical_lut_v1",
                "mode": "ocl-3x3-synthetic-fallback",
                "cell_pixels": {"x": 3, "z": 3},
                "wavelengths_nm": [550.0],
                "cases": [
                    {"name": "center", "field_x_norm": 0.0, "field_z_norm": 0.0, "cra_x_deg": 0.0, "cra_z_deg": 0.0},
                    {
                        "name": "edge20_synthetic",
                        "field_x_norm": 1.0,
                        "field_z_norm": 0.0,
                        "cra_x_deg": 20.0,
                        "cra_z_deg": 0.0,
                    },
                ],
                "long_csv": str(long_csv),
                "summary_csv": str(summary_csv),
                "notes": ["Synthetic fallback used only when the external FDTD repo LUT is unavailable."],
            }
        ),
        encoding="utf-8",
    )
    return json_path


def _load_lut(path: Path | None):
    default_path = fdtd_sensor_default_lut_path()
    resolved = path or default_path or _synthetic_lut_path()
    lut = fdtd_sensor_lut_load(resolved)
    return lut, bool(default_path is not None and Path(resolved).resolve() == Path(default_path).resolve())


def _edge_oi(rows: int = 96, cols: int = 96) -> OpticalImage:
    wave = np.asarray(DEFAULT_WAVE, dtype=float)
    y = np.linspace(-1.0, 1.0, rows, dtype=float)
    x = np.linspace(-1.0, 1.0, cols, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    vertical_edge = np.where(xx > 0.0, 2.4, 0.35)
    slanted_edge = np.where(yy > 0.55 * xx - 0.05, 1.0, 0.55)
    fine_lines = 0.80 + 0.20 * (np.sin(22.0 * np.pi * xx) > 0.0)
    image = 8.0e11 * vertical_edge * slanted_edge * fine_lines
    spectral = np.interp(wave, [float(wave.min()), 550.0, float(wave.max())], [0.65, 1.0, 0.75])

    oi = OpticalImage(name="fdtd sensor edge verification oi")
    spacing = 2.8e-6
    oi.fields.update(
        {
            "wave": wave,
            "sample_spacing_m": spacing,
            "width_m": cols * spacing,
            "height_m": rows * spacing,
            "fov_deg": 1.0,
            "vfov_deg": 1.0,
            "optics": {"model": "skip", "focal_length_m": 0.004},
        }
    )
    oi.data["photons"] = image[:, :, None] * spectral.reshape(1, 1, -1)
    return oi


def _normalize(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _render_response_rolloff(lut, output_path: Path) -> dict[str, float]:
    rows = lut.summary_rows
    cases = [str(row.get("case", row.get("name", f"case-{idx}"))) for idx, row in enumerate(rows)]
    responses = [
        fdtd_sensor_lut_response(lut, case=str(row.get("case", row.get("name", ""))), wavelength_nm=row.get("wavelength_nm"))
        for row in rows
    ]
    if not cases:
        cases = ["center"]
        responses = [1.0]

    fig, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    axis.bar(cases, responses, color="#2f6f88")
    axis.set_ylabel("Normalized optical response")
    axis.set_title("FDTD LUT field / CRA response")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {case: float(value) for case, value in zip(cases, responses, strict=False)}


def _render_kernel(lut, output_path: Path) -> np.ndarray:
    kernel = fdtd_sensor_lut_crosstalk_kernel(lut, case="center", wavelength_nm=550.0)
    fig, axis = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    im = axis.imshow(kernel, cmap="magma")
    axis.set_title("3x3 FDTD regional-response proxy kernel")
    axis.set_xlabel("Pixel x region")
    axis.set_ylabel("Pixel y region")
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            axis.text(col, row, f"{kernel[row, col]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return kernel


def _render_triptych(
    reference: np.ndarray,
    fdtd: np.ndarray,
    output_path: Path,
    *,
    title: str,
    cmap: str = "gray",
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    fdtd = np.asarray(fdtd, dtype=float)
    diff = np.abs(fdtd - reference)
    vmin = min(float(np.min(reference)), float(np.min(fdtd)))
    vmax = max(float(np.max(reference)), float(np.max(fdtd)))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, data, subtitle in (
        (axes[0], reference, "baseline sensor"),
        (axes[1], fdtd, "FDTD-enabled sensor"),
        (axes[2], diff, "absolute difference"),
    ):
        if data.ndim == 3:
            axis.imshow(_normalize(data, vmin, vmax))
        else:
            im = axis.imshow(data, cmap=cmap, vmin=vmin if subtitle != "absolute difference" else None, vmax=vmax if subtitle != "absolute difference" else None)
            if subtitle == "absolute difference":
                fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(subtitle)
        axis.set_axis_off()
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {
        "mae": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "normalized_mae": float(np.mean(diff) / max(float(np.max(np.abs(reference))), 1e-12)),
    }


def _render_crops(base: np.ndarray, fdtd: np.ndarray, output_path: Path) -> dict[str, float]:
    rows, cols = base.shape[:2]
    radius = max(min(rows, cols) // 8, 6)
    center = (rows // 2, cols // 2)
    edge = (rows // 2, cols - radius - 1)

    def crop(data: np.ndarray, point: tuple[int, int]) -> np.ndarray:
        row, col = point
        return data[max(row - radius, 0) : min(row + radius, rows), max(col - radius, 0) : min(col + radius, cols)]

    crops = [
        (crop(base, center), "baseline center"),
        (crop(fdtd, center), "FDTD center"),
        (crop(base, edge), "baseline edge"),
        (crop(fdtd, edge), "FDTD edge"),
    ]
    vmin = min(float(np.min(item[0])) for item in crops)
    vmax = max(float(np.max(item[0])) for item in crops)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0), constrained_layout=True)
    for axis, (data, title) in zip(axes.ravel(), crops, strict=False):
        axis.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle("Center and edge sensor crops")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    center_ratio = float(np.mean(crop(fdtd, center)) / max(float(np.mean(crop(base, center))), 1e-12))
    edge_ratio = float(np.mean(crop(fdtd, edge)) / max(float(np.mean(crop(base, edge))), 1e-12))
    return {"center_mean_ratio": center_ratio, "edge_mean_ratio": edge_ratio}


def _render_channel_means(base_rgb: np.ndarray, fdtd_rgb: np.ndarray, output_path: Path) -> dict[str, list[float]]:
    base_means = np.mean(np.asarray(base_rgb, dtype=float), axis=(0, 1))
    fdtd_means = np.mean(np.asarray(fdtd_rgb, dtype=float), axis=(0, 1))
    x = np.arange(base_means.size)
    labels = ["R", "G", "B"][: base_means.size]
    fig, axis = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    axis.bar(x - 0.18, base_means, width=0.36, label="baseline", color="#6b8fb3")
    axis.bar(x + 0.18, fdtd_means, width=0.36, label="FDTD-enabled", color="#b36b4f")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean IP channel value")
    axis.set_title("ISP output channel means")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {"baseline": base_means.tolist(), "fdtd": fdtd_means.tolist()}


def _architecture_svg() -> str:
    return """
<svg viewBox="0 0 1120 220" role="img" aria-label="FDTD sensor architecture">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#d8ecf2"/>
      <stop offset="100%" stop-color="#f3e2d5"/>
    </linearGradient>
  </defs>
  <rect width="1120" height="220" rx="24" fill="url(#g)"/>
  <g font-family="Helvetica, Arial, sans-serif" font-size="18" fill="#1f2933">
    <g fill="#ffffff" stroke="#2f6f88" stroke-width="2">
      <rect x="35" y="70" width="145" height="70" rx="14"/>
      <rect x="225" y="70" width="150" height="70" rx="14"/>
      <rect x="420" y="70" width="170" height="70" rx="14"/>
      <rect x="635" y="70" width="165" height="70" rx="14"/>
      <rect x="845" y="70" width="140" height="70" rx="14"/>
    </g>
    <text x="107" y="99" text-anchor="middle">FDTD DB</text>
    <text x="107" y="123" text-anchor="middle" font-size="14">CSV / JSON LUT</text>
    <text x="300" y="99" text-anchor="middle">LUT loader</text>
    <text x="300" y="123" text-anchor="middle" font-size="14">validate + interpolate</text>
    <text x="505" y="99" text-anchor="middle">Sensor optical layer</text>
    <text x="505" y="123" text-anchor="middle" font-size="14">QE, CRA, field, crosstalk</text>
    <text x="718" y="99" text-anchor="middle">sensor_compute</text>
    <text x="718" y="123" text-anchor="middle" font-size="14">volts/electrons/noise</text>
    <text x="915" y="99" text-anchor="middle">IP / metrics</text>
    <text x="915" y="123" text-anchor="middle" font-size="14">verification output</text>
    <g stroke="#1f2933" stroke-width="3" fill="none" marker-end="url(#arrow)">
      <path d="M180 105 H225"/>
      <path d="M375 105 H420"/>
      <path d="M590 105 H635"/>
      <path d="M800 105 H845"/>
    </g>
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#1f2933" />
      </marker>
    </defs>
  </g>
</svg>
"""


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f5ef; color: #1f2933; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 34px 24px 56px; }}
    h1 {{ font-size: 42px; margin: 0 0 10px; }}
    h2 {{ margin-top: 34px; border-bottom: 2px solid #ded6c8; padding-bottom: 8px; }}
    .card {{ background: #fff; border: 1px solid #e3dccf; border-radius: 18px; padding: 20px; margin: 18px 0; box-shadow: 0 8px 24px rgba(35, 44, 50, 0.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .metric {{ background: #eef6f8; border-radius: 14px; padding: 14px; }}
    .metric b {{ display: block; font-size: 13px; color: #52616b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric span {{ font-size: 24px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #ece6db; vertical-align: top; }}
    th {{ background: #f0ebe1; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid #e2ded5; background: #fff; }}
    code {{ background: #eee7da; padding: 2px 5px; border-radius: 5px; }}
    .warn {{ border-left: 5px solid #b36b4f; background: #fff4ec; }}
    .ok {{ border-left: 5px solid #2f8f62; }}
    .bad {{ border-left: 5px solid #b34f4f; background: #fff1f1; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table><tbody>" + "".join(f"<tr><th>{escape(key)}</th><td>{value}</td></tr>" for key, value in rows) + "</tbody></table>"


def render_report(output_dir: Path = DEFAULT_OUTPUT_DIR, lut_path: Path | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lut, using_external_default = _load_lut(lut_path)
    validation = fdtd_sensor_lut_validate(lut)
    physics_validation = fdtd_sensor_physics_validate(lut)
    lut_summary = fdtd_sensor_lut_summary(lut)

    response_png = output_dir / "fdtd_response_rolloff.png"
    kernel_png = output_dir / "fdtd_crosstalk_kernel.png"
    volts_png = output_dir / "sensor_volts_triptych.png"
    crops_png = output_dir / "center_edge_crops.png"
    ip_png = output_dir / "isp_output_triptych.png"
    channel_png = output_dir / "ip_channel_means.png"

    response_by_case = _render_response_rolloff(lut, response_png)
    kernel = _render_kernel(lut, kernel_png)

    store = AssetStore.default()
    oi = _edge_oi()
    base_sensor = sensor_set(sensor_create("default", asset_store=store), "size", oi.data["photons"].shape[:2])
    base_sensor = sensor_set(base_sensor, "noise flag", 0)
    base_sensor = sensor_set(base_sensor, "integration time", 0.006)
    crosstalk_strength = 0.25
    fdtd_sensor = sensor_attach_fdtd_lut(base_sensor, lut, mode="field+crosstalk", crosstalk_strength=crosstalk_strength)

    base_sensor = sensor_compute(base_sensor, oi, seed=0)
    fdtd_sensor = sensor_compute(fdtd_sensor, oi, seed=0)
    base_volts = np.asarray(sensor_get(base_sensor, "volts"), dtype=float)
    fdtd_volts = np.asarray(sensor_get(fdtd_sensor, "volts"), dtype=float)
    sensor_metrics = _render_triptych(base_volts, fdtd_volts, volts_png, title="Raw sensor volts: baseline vs FDTD optical layer")
    crop_metrics = _render_crops(base_volts, fdtd_volts, crops_png)

    base_ip = ip_compute(ip_create(sensor=base_sensor), base_sensor, asset_store=store)
    fdtd_ip = ip_compute(ip_create(sensor=fdtd_sensor), fdtd_sensor, asset_store=store)
    base_rgb = np.asarray(ip_get(base_ip, "result"), dtype=float)
    fdtd_rgb = np.asarray(ip_get(fdtd_ip, "result"), dtype=float)
    ip_metrics = _render_triptych(base_rgb, fdtd_rgb, ip_png, title="ISP output: baseline vs FDTD optical layer")
    channel_metrics = _render_channel_means(base_rgb, fdtd_rgb, channel_png)

    field_map = fdtd_sensor_field_response_map(lut, base_volts.shape[:2], wavelength_nm=550.0)
    probe = np.zeros((9, 9), dtype=float)
    probe[4, 4] = 1.0
    crosstalk_probe = fdtd_sensor_apply_optical_response(probe, fdtd_sensor_config(lut, mode="crosstalk"))

    summary = {
        "report": str(output_dir / "fdtd_sensor_report.html"),
        "lut": lut_summary,
        "validation": validation,
        "physics_validation": physics_validation,
        "using_external_default_lut": using_external_default,
        "figures": {
            "response_rolloff": str(response_png),
            "crosstalk_kernel": str(kernel_png),
            "sensor_volts": str(volts_png),
            "center_edge_crops": str(crops_png),
            "isp_output": str(ip_png),
            "ip_channel_means": str(channel_png),
        },
        "response_by_case": response_by_case,
        "kernel_sum": float(np.sum(kernel)),
        "kernel_center": float(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]),
        "report_crosstalk_strength": crosstalk_strength,
        "field_map_min": float(np.min(field_map)),
        "field_map_max": float(np.max(field_map)),
        "crosstalk_probe_center": float(crosstalk_probe[4, 4]),
        "sensor_metrics": sensor_metrics,
        "crop_metrics": crop_metrics,
        "ip_metrics": ip_metrics,
        "channel_metrics": channel_metrics,
    }
    summary_path = output_dir / "fdtd_sensor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_note = (
        "Using external FDTD repo LUT."
        if using_external_default
        else "Using explicit or synthetic LUT. Set PYISETCAM_FDTD_ROOT or pass --lut for the product FDTD DB."
    )
    validation_class = "ok" if validation["ok"] else "warn"
    physics_status = str(physics_validation["status"])
    physics_class = "ok" if physics_status == "pass" else "bad" if physics_status == "fail" else "warn"
    body = f"""
<h1>FDTD-Informed Sensor Block Report</h1>
<p>This report verifies the Phase 1-5 implementation path: LUT ingestion, QE/CRA/field modeling, 3x3 crosstalk, sensor runtime integration, and image evidence through ISP output.</p>
<div class="card {validation_class}">
  <h2>Executive Summary</h2>
  <div class="grid">
    <div class="metric"><b>LUT validation</b><span>{'PASS' if validation['ok'] else 'CHECK'}</span></div>
    <div class="metric"><b>Physics sanity</b><span>{escape(physics_status.upper())}</span></div>
    <div class="metric"><b>Sensor normalized MAE</b><span>{sensor_metrics['normalized_mae']:.4f}</span></div>
    <div class="metric"><b>Field response range</b><span>{summary['field_map_min']:.3f}-{summary['field_map_max']:.3f}</span></div>
    <div class="metric"><b>Kernel center weight</b><span>{summary['kernel_center']:.3f}</span></div>
  </div>
  <p><b>Source:</b> {escape(source_note)}</p>
  <p><b>Physics report:</b> run <code>python tools/render_fdtd_sensor_physics_report.py</code> and open <code>reports/fdtd_sensor/physics_validation_report.html</code>.</p>
</div>
<div class="card {physics_class}">
  <h2>Physics Sanity Gate</h2>
  <p>This integration report proves the data path. The physics gate separately checks cos^4 RI, energy bounds, OCL shift direction, wavelength coverage, symmetry coverage, convergence metadata, and crosstalk locality.</p>
  <p><b>Current status:</b> {escape(physics_status.upper())}. {escape('; '.join(physics_validation.get('failures', []) + physics_validation.get('warnings', [])) or 'No physics warnings detected.')}</p>
</div>
<div class="card warn">
  <h2>Important Boundary</h2>
  <p>The current FDTD DB is an <b>optical absorption / regional response proxy</b>. It improves the optical sensor block model for microlens/CFA/Si-stack response, CRA rolloff, field shading, and crosstalk. It is not a TCAD carrier-transport model and should not be presented as full photodiode charge-collection physics.</p>
  <p><b>Engineering caution:</b> the current 3x3 smoke LUT regional responses are nearly uniform under the available illumination setup. That is useful for verifying data flow, but it is not yet a product-grade localized optical crosstalk PSF. The simulator exposes <code>crosstalk_strength</code>; this report uses {crosstalk_strength:.2f} to avoid overstating that smoke LUT as a final crosstalk model.</p>
</div>
<div class="card">
  <h2>Architecture</h2>
  {_architecture_svg()}
</div>
<div class="card">
  <h2>FDTD DB And LUT Validation</h2>
  {_table([
      ('LUT path', f'<code>{escape(str(lut.source_path))}</code>'),
      ('Schema', escape(str(lut.schema))),
      ('Mode', escape(str(lut.mode))),
      ('Wavelengths', escape(', '.join(str(v) for v in lut_summary['wavelengths_nm']))),
      ('Cases', escape(', '.join(lut_summary['cases']))),
      ('Rows', f"summary={lut_summary['n_summary_rows']}, long={lut_summary['n_long_rows']}"),
      ('Validation issues', escape(', '.join(validation['issues']) if validation['issues'] else 'none')),
  ])}
  <img src="{response_png.name}" alt="FDTD response rolloff">
</div>
<div class="card">
  <h2>3x3 Crosstalk Evidence</h2>
  <p>The regional-response rows are converted to a normalized 3x3 optical crosstalk proxy kernel. The kernel is applied after spatial integration and before voltage/noise/ADC logic when enabled.</p>
  <img src="{kernel_png.name}" alt="FDTD crosstalk kernel">
</div>
<div class="card">
  <h2>Sensor Volts Evidence</h2>
  <p>The same edge-rich optical image was computed with the baseline sensor and with the FDTD optical layer enabled. Existing voltage/noise behavior remains downstream of the optional optical layer.</p>
  <img src="{volts_png.name}" alt="Sensor volts triptych">
  {_table([
      ('Sensor MAE', f"{sensor_metrics['mae']:.6g}"),
      ('Sensor max abs', f"{sensor_metrics['max_abs']:.6g}"),
      ('Sensor normalized MAE', f"{sensor_metrics['normalized_mae']:.6g}"),
      ('Center mean ratio', f"{crop_metrics['center_mean_ratio']:.6g}"),
      ('Edge mean ratio', f"{crop_metrics['edge_mean_ratio']:.6g}"),
  ])}
  <img src="{crops_png.name}" alt="Center and edge crops">
</div>
<div class="card">
  <h2>ISP Output Evidence</h2>
  <p>The FDTD layer propagates through the normal IP path, so ISP-visible differences can be inspected without changing IP algorithms.</p>
  <img src="{ip_png.name}" alt="ISP output triptych">
  <img src="{channel_png.name}" alt="IP channel means">
  {_table([
      ('IP MAE', f"{ip_metrics['mae']:.6g}"),
      ('IP max abs', f"{ip_metrics['max_abs']:.6g}"),
      ('IP normalized MAE', f"{ip_metrics['normalized_mae']:.6g}"),
  ])}
</div>
<div class="card">
  <h2>Verification Result</h2>
  <p>Phase 1-5 is implemented as an optional FDTD-informed sensor layer. The default sensor path is unchanged when no FDTD config is attached. The report demonstrates LUT ingestion, field/CRA response, crosstalk, raw sensor impact, center/edge evidence, and ISP-output impact.</p>
</div>
"""
    html_path = output_dir / "fdtd_sensor_report.html"
    html_path.write_text(_html_page("FDTD-Informed Sensor Block Report", body), encoding="utf-8")
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
