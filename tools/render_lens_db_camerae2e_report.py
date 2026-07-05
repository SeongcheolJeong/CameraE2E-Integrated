"""Render a CameraE2E Lens DB integration and smoke-test report."""

from __future__ import annotations

import json
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
    camera_compute,
    camera_create,
    camera_get,
    camera_set,
    lens_patent_camerae2e_manifest,
    lens_patent_companies,
    lens_patent_db_summary,
    lens_patent_default_data_dir,
    lens_patent_default_db_path,
    lens_patent_raytrace_optics,
    lens_patent_raytrace_psf_manifest,
    lens_patent_raytrace_psf_search,
    oi_create,
    scene_create,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "lens_db"
DEFAULT_SIMULATION_ID = "p0014:intermediate"


def main(output_dir: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    data_dir = lens_patent_default_data_dir()
    db_path = lens_patent_default_db_path()
    summary = lens_patent_db_summary()
    companies = lens_patent_companies()
    package_manifest = _optional_manifest()
    psf_manifest = lens_patent_raytrace_psf_manifest()
    highres_dir = data_dir / "raytrace_psf_highres"
    highres_manifest = lens_patent_raytrace_psf_manifest(highres_dir) if (highres_dir / "manifest.json").exists() else None

    sample_id = _choose_sample_id(highres_manifest, psf_manifest)
    camera_stats = _run_camera_smoke(sample_id)
    psf_stats = _render_psf_triptych(sample_id, out / "lens_db_sample_psf.png")
    readiness_png = _render_company_readiness(package_manifest, companies, out / "lens_db_company_readiness.png")
    generation_png = _render_psf_generation(psf_manifest, out / "lens_db_psf_generation.png")
    pipeline_png = _render_pipeline_outputs(camera_stats, out / "lens_db_camera_pipeline.png")

    report = {
        "data_dir": str(data_dir),
        "db_path": str(db_path),
        "summary": summary,
        "package_manifest": package_manifest,
        "psf_summary": psf_manifest.get("summary", {}),
        "highres_psf_summary": highres_manifest.get("summary", {}) if highres_manifest else None,
        "sample_simulation_id": sample_id,
        "psf_stats": psf_stats,
        "camera_smoke": _json_safe_camera_stats(camera_stats),
        "figures": {
            "company_readiness": str(readiness_png),
            "psf_generation": str(generation_png),
            "sample_psf": str(out / "lens_db_sample_psf.png"),
            "camera_pipeline": str(pipeline_png),
        },
        "caveats": _caveats(package_manifest),
    }

    summary_path = out / "lens_db_camerae2e_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = out / "lens_db_camerae2e_report.html"
    html_path.write_text(_render_html(report, html_path), encoding="utf-8")
    return {"html": html_path, "summary": summary_path, **report}


def _optional_manifest() -> dict[str, Any] | None:
    try:
        return lens_patent_camerae2e_manifest()
    except FileNotFoundError:
        return None


def _choose_sample_id(highres_manifest: dict[str, Any] | None, psf_manifest: dict[str, Any]) -> str:
    if highres_manifest is not None:
        for row in highres_manifest.get("rows", []):
            if row.get("simulation_id") == DEFAULT_SIMULATION_ID and row.get("status") == "generated":
                return DEFAULT_SIMULATION_ID
        for row in highres_manifest.get("rows", []):
            if row.get("status") == "generated":
                return str(row["simulation_id"])
    for row in psf_manifest.get("rows", []):
        if row.get("simulation_id") == DEFAULT_SIMULATION_ID and row.get("status") == "generated":
            return DEFAULT_SIMULATION_ID
    for row in psf_manifest.get("rows", []):
        if row.get("status") == "generated":
            return str(row["simulation_id"])
    raise RuntimeError("No generated RayOptics PSF asset is available for Lens DB smoke testing.")


def _run_camera_smoke(simulation_id: str) -> dict[str, Any]:
    data_dir = lens_patent_default_data_dir()
    highres_dir = data_dir / "raytrace_psf_highres"
    psf_dir = highres_dir if (highres_dir / "manifest.json").exists() else None
    target_psf_size = 64

    scene = scene_create("bar", 24)
    optics = lens_patent_raytrace_optics(
        simulation_id,
        psf_dir=psf_dir,
        target_psf_size=target_psf_size,
    )
    camera = camera_create()
    camera = camera_set(camera, "oi", oi_create("ray trace"))
    camera = camera_set(camera, "optics", optics)
    camera = camera_set(camera, "sensor size", [24, 24])
    result = camera_compute(camera, scene, sensor_resize=False)

    oi_photons = np.asarray(camera_get(result, "oi").data["photons"], dtype=float)
    sensor_volts = np.asarray(camera_get(result, "sensor volts"), dtype=float)
    image = np.asarray(camera_get(result, "image"), dtype=float)
    psf = np.asarray(optics["raytrace"]["psf"]["function"], dtype=float)

    return {
        "simulation_id": simulation_id,
        "scene": "bar",
        "psf_dir": str(psf_dir) if psf_dir is not None else str(data_dir / "raytrace_psf"),
        "target_psf_size": target_psf_size,
        "psf_shape": list(psf.shape),
        "oi_photons": oi_photons,
        "sensor_volts": sensor_volts,
        "image": image,
        "oi_shape": list(oi_photons.shape),
        "sensor_shape": list(sensor_volts.shape),
        "image_shape": list(image.shape),
        "sensor_min": float(np.min(sensor_volts)),
        "sensor_max": float(np.max(sensor_volts)),
        "sensor_mean": float(np.mean(sensor_volts)),
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "image_mean": float(np.mean(image)),
    }


def _render_company_readiness(
    package_manifest: dict[str, Any] | None,
    companies: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if package_manifest is not None:
        by_company = package_manifest.get("by_company", {})
        labels = sorted(by_company)
        raytrace = [float(by_company[name].get("raytrace_psf", 0)) for name in labels]
        proxy = [float(by_company[name].get("proxy_only", 0)) for name in labels]
        partial = [float(by_company[name].get("partial", 0)) for name in labels]
    else:
        labels = [str(row["company"]) for row in companies]
        raytrace = [float(row.get("ready_count", row.get("camerae2e_ready", 0)) or 0) for row in companies]
        proxy = [0.0 for _ in labels]
        partial = [0.0 for _ in labels]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.barh(y, raytrace, label="RayOptics PSF", color="#276678")
    ax.barh(y, proxy, left=raytrace, label="proxy-only", color="#f6a04d")
    ax.barh(y, partial, left=np.asarray(raytrace) + np.asarray(proxy), label="partial", color="#7f8c8d")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Lens configurations")
    ax.set_title("CameraE2E Lens DB readiness by company")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _render_psf_generation(psf_manifest: dict[str, Any], output_path: Path) -> Path:
    by_company = psf_manifest.get("summary", {}).get("by_company", {})
    labels = sorted(by_company)
    generated = np.asarray([float(by_company[name].get("generated", 0)) for name in labels], dtype=float)
    failed = np.asarray([float(by_company[name].get("failed", 0)) for name in labels], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(y, generated, label="generated", color="#2a9d8f")
    ax.barh(y, failed, left=generated, label="failed", color="#e76f51")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("RayOptics PSF rows")
    ax.set_title("RayOptics geometric PSF generation status")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _render_psf_triptych(simulation_id: str, output_path: Path) -> dict[str, Any]:
    data_dir = lens_patent_default_data_dir()
    highres_dir = data_dir / "raytrace_psf_highres"
    psf_dir = highres_dir if (highres_dir / "manifest.json").exists() else None
    optics = lens_patent_raytrace_optics(simulation_id, psf_dir=psf_dir, target_psf_size=128)
    psf = np.asarray(optics["raytrace"]["psf"]["function"], dtype=float)
    fields = np.asarray(optics["raytrace"]["psf"]["field_height_mm"], dtype=float)
    waves = np.asarray(optics["raytrace"]["psf"]["wavelength_nm"], dtype=float)
    wave_idx = int(np.argmin(np.abs(waves - 550.0)))
    field_indices = [0, psf.shape[2] // 2, psf.shape[2] - 1]
    titles = ["center field", "mid field", "edge field"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for ax, idx, title in zip(axes, field_indices, titles, strict=True):
        image = np.log10(np.maximum(psf[:, :, idx, wave_idx], 1.0e-12))
        im = ax.imshow(image, cmap="magma")
        ax.set_title(f"{title}\n{fields[idx]:.2f} mm")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{simulation_id}: log10 RayOptics PSF at {waves[wave_idx]:.0f} nm")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    return {
        "simulation_id": simulation_id,
        "psf_shape": list(psf.shape),
        "field_height_mm": fields.tolist(),
        "wavelength_nm": waves.tolist(),
        "center_peak": float(np.max(psf[:, :, 0, wave_idx])),
        "edge_peak": float(np.max(psf[:, :, -1, wave_idx])),
    }


def _render_pipeline_outputs(camera_stats: dict[str, Any], output_path: Path) -> Path:
    oi = _normalize_projection(np.asarray(camera_stats["oi_photons"], dtype=float))
    sensor = np.asarray(camera_stats["sensor_volts"], dtype=float)
    image = np.asarray(camera_stats["image"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    axes[0].imshow(oi, cmap="gray")
    axes[0].set_title("Lens output OI photons")
    axes[1].imshow(sensor, cmap="gray")
    axes[1].set_title("Sensor volts")
    axes[2].imshow(np.clip(image / max(float(np.max(image)), 1.0e-12), 0.0, 1.0))
    axes[2].set_title("IP output image")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"CameraE2E smoke pipeline: {camera_stats['simulation_id']}")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _normalize_projection(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _json_safe_camera_stats(camera_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in camera_stats.items()
        if key not in {"oi_photons", "sensor_volts", "image"}
    }


def _caveats(package_manifest: dict[str, Any] | None) -> list[str]:
    if package_manifest is not None:
        return [str(item) for item in package_manifest.get("caveats", [])]
    return [
        "Raytrace PSF rows are external package assets when available; bundled fallback data may be smaller.",
        "RayOptics PSFs are geometric ray-histogram PSFs, not diffraction wave-optics PSFs.",
    ]


def _render_html(report: dict[str, Any], html_path: Path) -> str:
    figures = report["figures"]
    summary = report["summary"]
    psf_summary = report["psf_summary"]
    highres_summary = report["highres_psf_summary"]
    camera = report["camera_smoke"]
    caveats = "".join(f"<li>{escape(item)}</li>" for item in report["caveats"])

    highres_html = (
        f"<p><b>Highres production PSFs:</b> {highres_summary.get('generated', 0)} generated / "
        f"{highres_summary.get('total', 0)} total</p>"
        if highres_summary
        else "<p><b>Highres production PSFs:</b> not available in the active Lens DB package.</p>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CameraE2E Lens DB Integration Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ color: #12343b; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 12px; padding: 18px; background: #fbfcfd; }}
    img {{ max-width: 100%; border: 1px solid #d8dee4; border-radius: 10px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    th {{ background: #edf2f7; }}
    code {{ background: #eef2f5; padding: 2px 5px; border-radius: 4px; }}
    .warn {{ border-left: 5px solid #e76f51; padding-left: 12px; }}
  </style>
</head>
<body>
  <h1>CameraE2E Lens DB Integration Report</h1>
  <p>This report verifies that CameraE2E is using the active RayOptics Lens DB package and can run a lens-through-sensor-through-IP smoke pipeline with a generated RayOptics PSF asset.</p>

  <h2>Active Data Source</h2>
  <table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Lens data directory</td><td><code>{escape(report["data_dir"])}</code></td></tr>
    <tr><td>SQLite DB</td><td><code>{escape(report["db_path"])}</code></td></tr>
    <tr><td>Total companies</td><td>{summary.get("companies", "")}</td></tr>
    <tr><td>Total lenses</td><td>{summary.get("lenses", "")}</td></tr>
    <tr><td>Total simulation rows</td><td>{summary.get("simulation_results", "")}</td></tr>
    <tr><td>CameraE2E-ready rows</td><td>{summary.get("status_counts", {}).get("camerae2e_ready", "")}</td></tr>
    <tr><td>Debug RayOptics PSFs</td><td>{psf_summary.get("generated", 0)} generated / {psf_summary.get("total", 0)} total</td></tr>
  </table>
  {highres_html}

  <h2>Coverage Figures</h2>
  <div class="grid">
    <div class="card"><h3>Company readiness</h3><img src="{escape(Path(figures["company_readiness"]).name)}" alt="company readiness"></div>
    <div class="card"><h3>PSF generation</h3><img src="{escape(Path(figures["psf_generation"]).name)}" alt="psf generation"></div>
  </div>

  <h2>Sample Lens Evidence</h2>
  <p>Sample simulation ID: <code>{escape(report["sample_simulation_id"])}</code></p>
  <div class="grid">
    <div class="card"><h3>RayOptics PSF field samples</h3><img src="{escape(Path(figures["sample_psf"]).name)}" alt="sample psf"></div>
    <div class="card"><h3>CameraE2E pipeline smoke output</h3><img src="{escape(Path(figures["camera_pipeline"]).name)}" alt="camera pipeline"></div>
  </div>

  <h2>Pipeline Smoke Metrics</h2>
    <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Smoke scene</td><td><code>{escape(camera["scene"])}</code></td></tr>
    <tr><td>PSF directory</td><td><code>{escape(camera["psf_dir"])}</code></td></tr>
    <tr><td>PSF shape</td><td>{escape(str(camera["psf_shape"]))}</td></tr>
    <tr><td>OI photons shape</td><td>{escape(str(camera["oi_shape"]))}</td></tr>
    <tr><td>Sensor volts shape</td><td>{escape(str(camera["sensor_shape"]))}</td></tr>
    <tr><td>IP image shape</td><td>{escape(str(camera["image_shape"]))}</td></tr>
    <tr><td>Sensor volts min / mean / max</td><td>{camera["sensor_min"]:.6g} / {camera["sensor_mean"]:.6g} / {camera["sensor_max"]:.6g}</td></tr>
    <tr><td>IP image min / mean / max</td><td>{camera["image_min"]:.6g} / {camera["image_mean"]:.6g} / {camera["image_max"]:.6g}</td></tr>
  </table>

  <h2>Important Limitations</h2>
  <div class="warn">
    <ul>{caveats}</ul>
    <p>Interpretation: this verifies Lens DB ingestion and CameraE2E execution. It does not prove wave-optics diffraction parity, because the RayOptics PSFs in this package are geometric ray-histogram PSFs.</p>
  </div>

  <p>Generated from <code>tools/render_lens_db_camerae2e_report.py</code>. JSON summary: <code>{escape((html_path.parent / "lens_db_camerae2e_summary.json").name)}</code></p>
</body>
</html>
"""


if __name__ == "__main__":
    result = main()
    print(result["html"])
