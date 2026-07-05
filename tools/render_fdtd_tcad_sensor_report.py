"""Render integrated FDTD + DEVSIM/TCAD sensor-block evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import (  # noqa: E402
    fdtd_sensor_default_lut_path,
    fdtd_sensor_lut_load,
    fdtd_sensor_lut_response,
    fdtd_sensor_lut_summary,
    fdtd_sensor_physics_validate,
    tcad_sensor_db_load,
    tcad_sensor_default_root,
    tcad_sensor_generation_map_slice,
    tcad_sensor_summary,
    tcad_sensor_validate,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "fdtd_sensor"


def _best_fdtd_lut_path(root: Path) -> Path | None:
    candidates = [
        root / "runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json",
        root / "runs/convergence_cra3z_rgb_r84_gridsnap_quant/camera_lut.json",
        root / "runs/convergence_cra_diag_rgb_r84_gridsnap_quant/camera_lut.json",
        root / "runs/fdtd_to_tcad_generation_2d_cra5_r6_t12/camera_lut.json",
        root / "runs/fdtd_to_tcad_generation_2d_cra5_smoke/camera_lut.json",
        root / "runs/fdtd_to_tcad_generation_2d_cra_smoke/camera_lut.json",
        root / "runs/supercell_lut_ocl_3x3_volume_smoke/camera_lut.json",
        root / "runs/supercell_lut_ocl_3x3_smoke/camera_lut.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return fdtd_sensor_default_lut_path()


def _response_value(row: dict) -> float:
    for key in (
        "normalized_total_response_to_first",
        "normalized_response_to_first_case",
        "total_response",
        "response",
        "total_si_absorption_fraction_estimate",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return 1.0


def _render_fdtd_response(lut, path: Path) -> dict[str, float]:
    rows = lut.summary_rows
    if not rows and lut.long_rows:
        cases = sorted({str(row.get("case", "")) for row in lut.long_rows if row.get("case")})
        values = [fdtd_sensor_lut_response(lut, case=case, wavelength_nm=550.0) for case in cases]
    else:
        cases = [str(row.get("case", row.get("name", f"case-{idx}"))) for idx, row in enumerate(rows)]
        values = [_response_value(row) for row in rows]
    if not cases:
        cases = ["center"]
        values = [1.0]

    fig, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    axis.bar(cases, values, color="#1f6f8b")
    axis.set_title("FDTD optical response by wavelength / CRA / field case")
    axis.set_ylabel("Response or normalized response")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {case: float(value) for case, value in zip(cases, values, strict=False)}


def _render_generation_maps(db, path: Path) -> list[str]:
    generation = db.generation_map
    if generation is None:
        return []
    cases = [str(item) for item in generation.cases[: min(2, generation.cases.size)]]
    if not cases:
        return []
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.6), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis, case in zip(axes_array, cases, strict=False):
        x_um, depth_um, values = tcad_sensor_generation_map_slice(db, case=case, wavelength_nm=550.0)
        log_values = np.log10(np.clip(values, 1.0, None))
        extent = [float(x_um.min()), float(x_um.max()), float(depth_um.max()), float(depth_um.min())]
        im = axis.imshow(log_values.T, cmap="viridis", aspect="auto", extent=extent)
        axis.set_title(f"G(x, depth), {case}")
        axis.set_xlabel("x (um)")
        axis.set_ylabel("Depth from Si top (um)")
        fig.colorbar(im, ax=axis, label="log10 generation (cm^-3 s^-1)")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return cases


def _render_devsim_currents(db, path: Path) -> None:
    summaries = list(db.collection_summaries)
    labels = [str(item.case or f"case-{idx}") for idx, item in enumerate(summaries)]
    left = [item.left_photo_delta_a_per_cm for item in summaries]
    right = [item.right_photo_delta_a_per_cm for item in summaries]
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.bar(x - width / 2.0, left, width, label="Left cathode", color="#5e8c61")
    axis.bar(x + width / 2.0, right, width, label="Right cathode", color="#c5794b")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Photo-current delta (A/cm)")
    axis.set_title("DEVSIM split-PD collection current")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _render_split_and_balance(db, path: Path) -> None:
    summaries = list(db.collection_summaries)
    labels = [str(item.case or f"case-{idx}") for idx, item in enumerate(summaries)]
    split = [item.photo_split_phase_x_proxy for item in summaries]
    balance = [abs(item.terminal_current_balance_illuminated_a_per_cm) for item in summaries]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].bar(labels, split, color="#6750a4")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Split-PD phase proxy")
    axes[0].set_ylabel("(right - left) / total")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, balance, color="#8b3a3a")
    axes[1].set_yscale("log")
    axes[1].set_title("Terminal current balance")
    axes[1].set_ylabel("|balance| (A/cm)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _accuracy_gate_rows(db) -> str:
    gate = db.accuracy_gate
    if gate is None:
        return "<tr><td colspan='4'>No accuracy gate artifact was loaded.</td></tr>"
    rows = []
    for check in gate.checks:
        status = str(check.get("status", ""))
        css = "pass" if status == "PASS" else "fail" if status == "FAIL" else "warn"
        rows.append(
            "<tr>"
            f"<td>{escape(str(check.get('name', '')))}</td>"
            f"<td class='{css}'>{escape(status)}</td>"
            f"<td>{escape(str(check.get('accuracy_blocking', '')))}</td>"
            f"<td>{escape(str(check.get('details', '')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _camerae2e_architecture_svg() -> str:
    return """<svg class="diagram" viewBox="0 0 1180 560" role="img" aria-label="FDTD and DEVSIM integration path into CameraE2E">
  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L12,4 L0,8 z" fill="#24424d"></path>
    </marker>
    <style>
      .box { fill: #ffffff; stroke: #24424d; stroke-width: 2; rx: 14; }
      .fdtd { fill: #e8f4f8; stroke: #1f6f8b; }
      .tcad { fill: #edf7ee; stroke: #4f7f52; }
      .cam { fill: #fff6e8; stroke: #b06000; }
      .meta { fill: #f5f5f5; stroke: #777; stroke-dasharray: 5 4; }
      .future { fill: #fff0f0; stroke: #a50e0e; stroke-dasharray: 5 4; }
      .text { font: 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #202124; }
      .small { font-size: 13px; fill: #4d5156; }
      .title { font-size: 18px; font-weight: 700; fill: #12343b; }
      .line { stroke: #24424d; stroke-width: 2.2; fill: none; marker-end: url(#arrow); }
      .dashed { stroke-dasharray: 6 5; }
    </style>
  </defs>

  <rect class="box" x="25" y="225" width="150" height="86"></rect>
  <text class="text title" x="100" y="255" text-anchor="middle">CameraE2E</text>
  <text class="text small" x="100" y="280" text-anchor="middle">OI photons</text>
  <text class="text small" x="100" y="300" text-anchor="middle">Scene + lens output</text>

  <rect class="box fdtd" x="230" y="70" width="275" height="165"></rect>
  <text class="text title" x="367" y="102" text-anchor="middle">FDTD / Meep</text>
  <text class="text small" x="367" y="128" text-anchor="middle">Measured/proxy n,k, RI stack</text>
  <text class="text small" x="367" y="150" text-anchor="middle">Microlens / CFA / Si absorption</text>
  <text class="text small" x="367" y="172" text-anchor="middle">CRA, field, OCL shift, pupil</text>
  <text class="text small" x="367" y="202" text-anchor="middle">Outputs: optical response LUT</text>
  <text class="text small" x="367" y="222" text-anchor="middle">and G(x, depth)</text>

  <rect class="box tcad" x="230" y="325" width="275" height="150"></rect>
  <text class="text title" x="367" y="357" text-anchor="middle">DEVSIM / TCAD</text>
  <text class="text small" x="367" y="383" text-anchor="middle">Consumes FDTD G(x, depth)</text>
  <text class="text small" x="367" y="405" text-anchor="middle">Electrical collection model</text>
  <text class="text small" x="367" y="427" text-anchor="middle">Left/right cathode currents</text>
  <text class="text small" x="367" y="455" text-anchor="middle">Outputs: collection efficiency</text>

  <rect class="box cam" x="585" y="70" width="260" height="165"></rect>
  <text class="text title" x="715" y="102" text-anchor="middle">pyisetcam fdtd_sensor</text>
  <text class="text small" x="715" y="130" text-anchor="middle">QE scale if mode has 'qe'</text>
  <text class="text small" x="715" y="152" text-anchor="middle">Field rolloff map</text>
  <text class="text small" x="715" y="174" text-anchor="middle">Regional crosstalk kernel</text>
  <text class="text small" x="715" y="205" text-anchor="middle">Explicit attach only</text>

  <rect class="box cam" x="585" y="325" width="260" height="150"></rect>
  <text class="text title" x="715" y="357" text-anchor="middle">pyisetcam tcad_sensor</text>
  <text class="text small" x="715" y="385" text-anchor="middle">Collection multiplier</text>
  <text class="text small" x="715" y="407" text-anchor="middle">Split phase metadata/report</text>
  <text class="text small" x="715" y="429" text-anchor="middle">Accuracy gate validation</text>
  <text class="text small" x="715" y="455" text-anchor="middle">Explicit attach only</text>

  <rect class="box cam" x="925" y="205" width="220" height="155"></rect>
  <text class="text title" x="1035" y="237" text-anchor="middle">Sensor compute</text>
  <text class="text small" x="1035" y="264" text-anchor="middle">Photons -> current</text>
  <text class="text small" x="1035" y="286" text-anchor="middle">Current/electrons -> volts</text>
  <text class="text small" x="1035" y="308" text-anchor="middle">Existing noise/ADC</text>
  <text class="text small" x="1035" y="337" text-anchor="middle">IP receives sensor volts</text>

  <rect class="box meta" x="925" y="405" width="220" height="95"></rect>
  <text class="text title" x="1035" y="435" text-anchor="middle">Report / metadata</text>
  <text class="text small" x="1035" y="462" text-anchor="middle">RI provenance, split phase,</text>
  <text class="text small" x="1035" y="482" text-anchor="middle">terminal balance, accuracy gate</text>

  <rect class="box future" x="585" y="500" width="560" height="42"></rect>
  <text class="text small" x="865" y="526" text-anchor="middle">Not v1 image-affecting: DEVSIM dark current/noise/lag/full-well unless future calibrated DB hook is added</text>

  <path class="line" d="M175 255 C205 220,205 170,230 150"></path>
  <path class="line" d="M505 150 L585 150"></path>
  <path class="line" d="M845 152 C880 175,900 215,925 250"></path>
  <path class="line dashed" d="M505 205 C535 250,545 335,585 380"></path>
  <path class="line" d="M505 400 L585 400"></path>
  <path class="line" d="M845 400 C885 380,900 335,925 315"></path>
  <path class="line dashed" d="M845 435 L925 450"></path>
</svg>"""


def _runtime_application_rows() -> str:
    rows = [
        (
            "RI / n,k",
            "Upstream FDTD generation only",
            "The refractive-index stack is consumed by Meep when building the LUT. CameraE2E does not solve RI/FDTD per frame; it consumes the resulting optical-response LUT.",
            "Metadata + indirect response",
            "warn",
        ),
        (
            "QE / spectral response",
            "Image-affecting when FDTD mode includes qe",
            "fdtd_sensor_qe_scale(...) multiplies sensor spectral QE before photon-to-current integration.",
            "Implemented",
            "pass",
        ),
        (
            "CRA / field shading",
            "Image-affecting when FDTD mode includes field",
            "fdtd_sensor_field_response_map(...) applies center-to-edge response after spatial integration.",
            "Implemented",
            "pass",
        ),
        (
            "Optical crosstalk",
            "Image-affecting when FDTD mode includes crosstalk",
            "fdtd_sensor_lut_crosstalk_kernel(...) applies a regional-response convolution. Current smoke LUT is a proxy, not product crosstalk PSF.",
            "Implemented as proxy",
            "warn",
        ),
        (
            "G(x, depth)",
            "DEVSIM input and report evidence",
            "FDTD-derived generation map is loaded and visualized; DEVSIM summaries are generated from it. CameraE2E does not integrate G(x,depth) per pixel directly.",
            "Report + DB source",
            "pass",
        ),
        (
            "Collection efficiency",
            "Image-affecting when TCAD hook is attached",
            "tcad_sensor_apply_collection_response(...) scales signal current/electron rate by DEVSIM total photo-current relative to center.",
            "Implemented",
            "pass",
        ),
        (
            "Split-PD phase",
            "Metadata/report only in v1",
            "DEVSIM left/right current imbalance is loaded and reported. It is not yet used by autofocus/PDAF logic.",
            "Metadata",
            "warn",
        ),
        (
            "Noise / dark current / lag / full-well",
            "Not image-affecting from DEVSIM in v1",
            "CameraE2E still uses existing sensor noise fields. DEVSIM dark current and terminal balance are evidence only until calibrated electrical hooks are added.",
            "Future calibrated hook",
            "fail",
        ),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{escape(item)}</td>"
        f"<td>{escape(path)}</td>"
        f"<td>{escape(detail)}</td>"
        f"<td class='{css}'>{escape(status)}</td>"
        "</tr>"
        for item, path, detail, status, css in rows
    )


def render_report(output_dir: Path = DEFAULT_OUTPUT_DIR, fdtd_lut_path: Path | None = None, fdtd_root: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = fdtd_root or tcad_sensor_default_root()
    lut_path = fdtd_lut_path or _best_fdtd_lut_path(root)
    if lut_path is None:
        raise FileNotFoundError("No FDTD LUT path was found.")
    lut = fdtd_sensor_lut_load(lut_path)
    preferred_generation_map = lut.source_path.with_name("tcad_generation_map_2d.npz")
    db = tcad_sensor_db_load(
        root=root,
        generation_map_path=preferred_generation_map if preferred_generation_map.exists() else None,
    )

    fdtd_response_path = output_dir / "fdtd_tcad_fdtd_response.png"
    generation_path = output_dir / "fdtd_tcad_generation_maps.png"
    currents_path = output_dir / "fdtd_tcad_devsim_currents.png"
    split_path = output_dir / "fdtd_tcad_split_balance.png"

    response = _render_fdtd_response(lut, fdtd_response_path)
    generated_cases = _render_generation_maps(db, generation_path)
    _render_devsim_currents(db, currents_path)
    _render_split_and_balance(db, split_path)

    fdtd_validation = fdtd_sensor_physics_validate(lut)
    fdtd_summary = fdtd_sensor_lut_summary(lut)
    tcad_validation = tcad_sensor_validate(db)
    tcad_summary = tcad_sensor_summary(db)
    summary = {
        "report": str(output_dir / "fdtd_tcad_sensor_report.html"),
        "fdtd_lut_path": str(lut.source_path),
        "fdtd_summary": fdtd_summary,
        "fdtd_physics_validation": fdtd_validation,
        "fdtd_response": response,
        "tcad_validation": tcad_validation,
        "tcad_summary": tcad_summary,
        "generated_map_cases": generated_cases,
        "figures": {
            "fdtd_response": str(fdtd_response_path),
            "generation_maps": str(generation_path),
            "devsim_currents": str(currents_path),
            "split_balance": str(split_path),
        },
    }
    summary_path = output_dir / "fdtd_tcad_sensor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    gate = db.accuracy_gate
    architecture_svg = _camerae2e_architecture_svg()
    runtime_application_rows = _runtime_application_rows()
    validation_warnings = [
        *[str(item) for item in tcad_validation.get("warnings", [])],
        *[str(item) for item in fdtd_validation.get("warnings", [])],
    ]
    warning_rows = (
        "\n".join(f"<tr><td>{escape(item)}</td></tr>" for item in validation_warnings)
        if validation_warnings
        else "<tr><td>No warnings.</td></tr>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FDTD + DEVSIM Sensor Block Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #202124; }}
    h1, h2 {{ color: #12343b; }}
    .callout {{ padding: 14px 16px; border-left: 5px solid #b06000; background: #fff4e5; margin: 18px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; border: 1px solid #ddd; padding: 12px; background: #fff; }}
    figcaption {{ font-size: 0.92rem; color: #555; margin-top: 8px; }}
    img {{ width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d7d7d7; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f1f4f6; text-align: left; }}
    .pass {{ color: #116329; font-weight: 700; }}
    .warn {{ color: #8a5a00; font-weight: 700; }}
    .fail {{ color: #a50e0e; font-weight: 700; }}
    .diagram {{ width: 100%; max-width: 1180px; height: auto; display: block; margin: 12px auto 22px; border: 1px solid #d7d7d7; background: #fbfcfd; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>FDTD + DEVSIM Sensor Block Report</h1>
  <div class="callout">
    <b>Status:</b> TCAD framework = <b>{escape(str(tcad_validation["framework_ready"]))}</b>,
    accuracy DB = <b>{escape(str(tcad_validation["accuracy_ready"]))}</b>.
    Current outputs must be labeled <b>proxy simulation</b> until measured optical stack n,k,
    measured implant/profile, and measured QE/split/dark/lag targets are supplied and the gate passes.
  </div>

  <h2>Architecture</h2>
  <pre>OI photons -> FDTD optical absorption LUT -> DEVSIM TCAD collection LUT -> pyisetcam sensor electrons/volts -> IP</pre>
  {architecture_svg}

  <h2>What Is Actually Reflected In CameraE2E?</h2>
  <table>
    <tr><th>Parameter / effect</th><th>How it enters CameraE2E</th><th>Current implementation detail</th><th>Status</th></tr>
    {runtime_application_rows}
  </table>

  <h2>Source Artifacts</h2>
  <table>
    <tr><th>Artifact</th><th>Path / Value</th></tr>
    <tr><td>FDTD LUT</td><td><code>{escape(str(lut.source_path))}</code></td></tr>
    <tr><td>Generation map</td><td><code>{escape(str(db.generation_map.source_path if db.generation_map else None))}</code></td></tr>
    <tr><td>DEVSIM summaries</td><td><code>{escape(', '.join(str(item.source_path) for item in db.collection_summaries))}</code></td></tr>
    <tr><td>Accuracy gate</td><td><code>{escape(str(gate.source_path if gate else None))}</code></td></tr>
    <tr><td>Summary JSON</td><td><code>{escape(str(summary_path))}</code></td></tr>
  </table>

  <h2>Integration Warnings</h2>
  <table>
    <tr><th>Warning</th></tr>
    {warning_rows}
  </table>

  <h2>FDTD Optical Evidence</h2>
  <p>FDTD remains the optical layer. It contributes wavelength/QE proxy response, CRA/relative-illumination behavior, OCL shift behavior, and regional-response/crosstalk proxy data.</p>
  <table>
    <tr><th>FDTD optical item</th><th>Evidence</th></tr>
    <tr><td>Schema / mode</td><td><code>{escape(str(fdtd_summary.get("schema")))}</code> / <code>{escape(str(fdtd_summary.get("mode")))}</code></td></tr>
    <tr><td>Wavelength / QE proxy samples</td><td>{escape(str(fdtd_summary.get("wavelengths_nm")))}</td></tr>
    <tr><td>CRA / relative illumination check</td><td>{escape(str(fdtd_validation["checks"]["relative_illumination"]["status"]))}</td></tr>
    <tr><td>OCL shift check</td><td>{escape(str(fdtd_validation["checks"]["ocl_shift"]["status"]))}</td></tr>
    <tr><td>FDTD physics status</td><td>{escape(str(fdtd_validation["status"]))}</td></tr>
  </table>
  <div class="grid">
    <figure><img src="{fdtd_response_path.name}" alt="FDTD response"><figcaption>FDTD response across available cases.</figcaption></figure>
  </div>

  <h2>FDTD-to-TCAD Generation Map</h2>
  <p>The DEVSIM input is the FDTD-derived <code>G(x, depth)</code> generation map. This preserves lateral and depth-dependent absorption before electrical collection.</p>
  <div class="grid">
    <figure><img src="{generation_path.name}" alt="Generation maps"><figcaption>log10 generation maps for representative center/edge cases.</figcaption></figure>
  </div>

  <h2>DEVSIM Collection Evidence</h2>
  <p>DEVSIM summaries provide left/right cathode photo-current deltas, split-PD phase proxy, and terminal current-balance checks.</p>
  <div class="grid">
    <figure><img src="{currents_path.name}" alt="DEVSIM currents"><figcaption>Left/right collected photo-current per selected case.</figcaption></figure>
    <figure><img src="{split_path.name}" alt="Split phase and terminal balance"><figcaption>Split phase and terminal-current balance.</figcaption></figure>
  </div>

  <h2>Accuracy Gate</h2>
  <table>
    <tr><th>Check</th><th>Status</th><th>Accuracy Blocking</th><th>Details</th></tr>
    {_accuracy_gate_rows(db)}
  </table>
</body>
</html>
"""
    html_path = output_dir / "fdtd_tcad_sensor_report.html"
    html_path.write_text(html, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fdtd-root", type=Path, default=None)
    parser.add_argument("--fdtd-lut", type=Path, default=None)
    args = parser.parse_args()
    summary = render_report(output_dir=args.output_dir, fdtd_lut_path=args.fdtd_lut, fdtd_root=args.fdtd_root)
    print(summary["report"])


if __name__ == "__main__":
    main()
