"""Render selectable image-sensor DB structure and response report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyisetcam import (  # noqa: E402
    DEFAULT_WAVE,
    OpticalImage,
    fdtd_sensor_config,
    fdtd_sensor_default_lut_path,
    fdtd_sensor_lut_load,
    image_sensor_db_get,
    image_sensor_db_parameters,
    image_sensor_db_records,
    image_sensor_db_summary,
    ip_compute,
    ip_create,
    ip_get,
    sensor_attach_fdtd_lut,
    sensor_compute,
    sensor_create,
    sensor_get,
    sensor_set,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "sensor_db"


def main(
    output_dir: str | Path | None = None,
    *,
    sensor_id: str | None = None,
    max_sensors: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records = [image_sensor_db_get(sensor_id)] if sensor_id else image_sensor_db_records(limit=max_sensors)
    if not records:
        raise RuntimeError("No image sensor DB records were found.")

    lut_path = fdtd_sensor_default_lut_path()
    lut = fdtd_sensor_lut_load(lut_path) if lut_path is not None else None
    ocl_image = image_dir / "active_fdtd_ocl_response.png"
    field_image = image_dir / "active_fdtd_field_response.png"
    impact_image = image_dir / "active_camerae2e_impact.png"
    if lut is not None:
        _render_ocl_response(lut, ocl_image)
        _render_field_response(lut, field_image)
        _render_camerae2e_impact(lut_path, impact_image)

    ui_records: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sensor_id"])
        stack = _load_json(record.get("stack_config_path"))
        profile = _load_json(record.get("tcad_profile_path"))
        structure_path = image_dir / f"{sid}_structure.png"
        tcad_path = image_dir / f"{sid}_tcad_collection.png"
        _render_structure(record, stack, structure_path)
        _render_tcad_profile(record, profile, tcad_path)
        cad_template = _cad_template_info(record)
        cad_template_image = None
        if cad_template is not None:
            preview_path = Path(cad_template["preview_svg"])
            if preview_path.exists():
                copied_preview = image_dir / f"{sid}_{preview_path.name}"
                shutil.copyfile(preview_path, copied_preview)
                cad_template_image = f"images/{copied_preview.name}"
        params = image_sensor_db_parameters(sid)
        ui_records.append(
            {
                **_ui_record(record),
                "images": {
                    "structure": f"images/{structure_path.name}",
                    "ocl_response": f"images/{ocl_image.name}",
                    "field_response": f"images/{field_image.name}",
                    "tcad_collection": f"images/{tcad_path.name}",
                    "camerae2e_impact": f"images/{impact_image.name}",
                    "cad_template": cad_template_image,
                },
                "cad_template": cad_template,
                "parameters": _jsonable(params),
                "notes": _sensor_notes(record, stack),
            }
        )

    payload = {
        "summary": image_sensor_db_summary(),
        "active_fdtd_lut_path": None if lut_path is None else str(lut_path),
        "record_count": len(ui_records),
        "records": ui_records,
        "panel_note": (
            "Sensor structure panels are metadata-derived unless a product CAD/GDS source is listed. "
            "Parametric CAD template previews are topology references, not measured product CAD; template OCL grouping can differ from extracted pitch metadata. "
            "OCL/field/CameraE2E impact panels use the active FDTD reference LUT and become sensor-specific "
            "only after a matching per-sensor FDTD LUT is generated."
        ),
    }
    summary_path = out / "sensor_db_summary.json"
    html_path = out / "sensor_db_overview.html"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return {"html": html_path, "summary": summary_path, "record_count": len(ui_records)}


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _render_structure(record: dict[str, Any], stack: dict[str, Any], output_path: Path) -> None:
    geom = dict(stack.get("geometry_um", {}))
    pitch = float(geom.get("pitch") or record.get("pixel_pitch_um") or 1.4)
    layer_specs = [
        ("Silicon / PD", float(geom.get("si_thickness", 2.8)), "#9bc4e2"),
        ("Passivation", float(geom.get("passivation_thickness", 0.08)), "#d9e2ec"),
        ("CFA", float(geom.get("cfa_thickness", 0.8)), "#77b255"),
        ("OCL / microlens", float(geom.get("lens_height", 0.657)), "#f6d365"),
        ("Air", float(geom.get("air_top", 0.55)), "#f7fbff"),
    ]
    total_height = sum(max(height, 0.02) for _, height, _ in layer_specs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.28, 1.0]}, constrained_layout=True)
    ax = axes[0]
    y = 0.0
    for label, height, color in layer_specs:
        height = max(float(height), 0.02)
        ax.add_patch(patches.Rectangle((-pitch / 2, y), pitch, height, facecolor=color, edgecolor="#263238", linewidth=0.9))
        ax.text(0, y + height / 2, f"{label}\n{height:.2f} um", ha="center", va="center", fontsize=9)
        y += height
    lens_height = max(float(geom.get("lens_height", 0.657)), 0.02)
    lens_base_y = sum(max(float(h), 0.02) for _, h, _ in layer_specs[:3])
    ax.add_patch(
        patches.Ellipse((0, lens_base_y + lens_height * 0.28), width=pitch * 0.92, height=lens_height * 1.6, fill=False, edgecolor="#8a6d00", linewidth=1.4)
    )
    pd_depth = min(float(geom.get("si_thickness", 2.8)) * 0.75, total_height * 0.38)
    ax.add_patch(patches.Rectangle((-pitch * 0.34, 0.18), pitch * 0.68, pd_depth, facecolor="none", edgecolor="#145da0", linewidth=1.6, linestyle="--"))
    ax.text(0, 0.18 + pd_depth / 2, "PD / depletion\nproxy", ha="center", va="center", color="#145da0", fontsize=8)
    metal_edge = float(geom.get("metal_edge_width", 0.0) or 0.0)
    if metal_edge > 0:
        ax.add_patch(patches.Rectangle((-pitch / 2, lens_base_y - 0.04), metal_edge, 0.08, facecolor="#4b5563"))
        ax.add_patch(patches.Rectangle((pitch / 2 - metal_edge, lens_base_y - 0.04), metal_edge, 0.08, facecolor="#4b5563"))
    ax.set_xlim(-pitch * 0.7, pitch * 0.7)
    ax.set_ylim(0, total_height * 1.05)
    ax.text(
        0.02,
        0.02,
        "metadata proxy - not product CAD",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#ddd"},
    )
    ax.set_title("Metadata-derived proxy cross-section")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("depth / stack height (um)")
    ax.grid(alpha=0.15)

    _draw_top_view_layout(axes[1], record, geom)

    fig.suptitle(_sensor_title(record))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_ocl_response(lut: Any, output_path: Path) -> None:
    wave = _nearest_wave(lut, 550.0)
    rows = [row for row in lut.long_rows if _float(row.get("wavelength_nm")) == wave]
    cases = sorted({str(row.get("case", "")) for row in rows}) or sorted({str(row.get("case", "")) for row in lut.summary_rows})
    fig, axes = plt.subplots(1, max(len(cases), 1), figsize=(4.2 * max(len(cases), 1), 4.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, case in zip(axes_arr, cases, strict=False):
        case_rows = [row for row in rows if str(row.get("case")) == case]
        heatmap = _region_heatmap(case_rows)
        im = ax.imshow(heatmap, cmap="magma", vmin=0.0)
        ax.set_title(f"{case} @ {wave:.0f} nm")
        ax.set_xticks(range(heatmap.shape[1]))
        ax.set_yticks(range(heatmap.shape[0]))
        for y, x in np.ndindex(heatmap.shape):
            ax.text(x, y, f"{heatmap[y, x]:.3g}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Active FDTD OCL / pixel response proxy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_field_response(lut: Any, output_path: Path) -> None:
    summary = list(lut.summary_rows)
    waves = sorted({_float(row.get("wavelength_nm")) for row in summary if _float(row.get("wavelength_nm")) is not None})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for wave in waves:
        rows = [row for row in summary if _float(row.get("wavelength_nm")) == wave]
        rows = sorted(rows, key=lambda row: _float(row.get("field_x_norm")) or 0.0)
        center = next((_float(row.get("total_response")) for row in rows if str(row.get("case")) == "center"), None)
        if center is None or center == 0:
            center = _float(rows[0].get("total_response")) if rows else 1.0
        x = np.asarray([_float(row.get("field_x_norm")) or 0.0 for row in rows], dtype=float)
        y = np.asarray([(_float(row.get("total_response")) or 0.0) / center for row in rows], dtype=float)
        axes[0].plot(x, y, marker="o", label=f"{wave:.0f} nm")
    x_ref = np.linspace(0, 1, 100)
    theta = np.deg2rad(20.0 * x_ref)
    axes[0].plot(x_ref, np.cos(theta) ** 4, "--", color="#555", label="cos4 reference")
    axes[0].set_xlabel("field_x_norm")
    axes[0].set_ylabel("center-normalized response")
    axes[0].set_title("Field / CRA response")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = []
    values = []
    for row in sorted(summary, key=lambda item: (_float(item.get("wavelength_nm")) or 0.0, str(item.get("case")))):
        labels.append(f"{_float(row.get('wavelength_nm')):.0f} {row.get('case')}")
        values.append(_float(row.get("total_response")) or 0.0)
    axes[1].bar(np.arange(len(values)), values, color="#2f6f88")
    axes[1].set_xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("total_response")
    axes[1].set_title("Wavelength / case response")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Active FDTD relative-illumination / QE-like proxy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_tcad_profile(record: dict[str, Any], profile: dict[str, Any], output_path: Path) -> None:
    geom = dict(profile.get("geometry", {}))
    width = float(geom.get("width_um") or record.get("pixel_pitch_um") or 1.4)
    depth = float(geom.get("depth_um") or record.get("active_si_thickness_um") or 4.0)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ax.add_patch(patches.Rectangle((-width / 2, 0), width, depth, facecolor="#eff6ff", edgecolor="#1f2933"))
    for implant in profile.get("implants", [])[:12]:
        x0 = _float(implant.get("x_min_um"))
        x1 = _float(implant.get("x_max_um"))
        z0 = _float(implant.get("depth_min_um"))
        z1 = _float(implant.get("depth_max_um"))
        if None in {x0, x1, z0, z1}:
            continue
        donor = _float(implant.get("donor_cm3")) or 0.0
        acceptor = _float(implant.get("acceptor_cm3")) or 0.0
        color = "#2b6cb0" if donor >= acceptor else "#c05621"
        alpha = 0.18 + 0.42 * min(np.log10(max(donor, acceptor, 1.0)) / 21.0, 1.0)
        ax.add_patch(patches.Rectangle((x0, z0), x1 - x0, z1 - z0, facecolor=color, edgecolor=color, alpha=alpha))
    bdti = dict(geom.get("bdti", {}))
    if bdti.get("enabled"):
        for x0, x1 in ((bdti.get("x_left_min_um"), bdti.get("x_left_max_um")), (bdti.get("x_right_min_um"), bdti.get("x_right_max_um"))):
            if x0 is not None and x1 is not None:
                ax.add_patch(
                    patches.Rectangle(
                        (float(x0), float(bdti.get("depth_min_um", 0.0))),
                        float(x1) - float(x0),
                        float(bdti.get("depth_max_um", depth)),
                        facecolor="#4a5568",
                        alpha=0.35,
                    )
                )
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(depth, 0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("depth from Si top (um)")
    ax.set_title("TCAD / charge-collection proxy geometry")
    ax.grid(alpha=0.2)
    ax.text(
        0.01,
        0.99,
        "blue: donor-like regions\norange: acceptor-like regions\ngray: BDTI proxy",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#ddd"},
    )
    fig.suptitle(_sensor_title(record))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_camerae2e_impact(lut_path: Path | str | None, output_path: Path) -> None:
    if lut_path is None:
        return
    wave = np.asarray(DEFAULT_WAVE, dtype=float)
    rows = cols = 72
    y = np.linspace(-1.0, 1.0, rows)
    x = np.linspace(-1.0, 1.0, cols)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    edge = np.where(xx + 0.35 * yy > 0, 2.2, 0.35)
    lines = 0.75 + 0.25 * (np.sin(16 * np.pi * xx) > 0)
    spectral = np.interp(wave, [float(wave.min()), 550.0, float(wave.max())], [0.65, 1.0, 0.75])
    oi = OpticalImage(name="sensor-db-impact-oi")
    oi.fields.update(
        {
            "wave": wave,
            "sample_spacing_m": 2.8e-6,
            "width_m": cols * 2.8e-6,
            "height_m": rows * 2.8e-6,
            "fov_deg": 1.0,
            "optics": {"model": "skip", "focal_length_m": 0.004},
        }
    )
    oi.data["photons"] = 7.0e11 * edge[:, :, None] * lines[:, :, None] * spectral.reshape(1, 1, -1)
    base_sensor = sensor_set(sensor_create(), "size", [rows, cols])
    base_sensor = sensor_set(base_sensor, "noise flag", -1)
    fdtd_sensor = sensor_attach_fdtd_lut(base_sensor.clone(), fdtd_sensor_config(lut_path))
    base = sensor_compute(base_sensor, oi)
    fdtd = sensor_compute(fdtd_sensor, oi)
    base_v = np.asarray(sensor_get(base, "volts"), dtype=float)
    fdtd_v = np.asarray(sensor_get(fdtd, "volts"), dtype=float)
    diff = np.abs(fdtd_v - base_v)
    ip = ip_compute(ip_create(sensor=fdtd), fdtd)
    rgb = np.asarray(ip_get(ip, "srgb"), dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    vmin, vmax = float(min(base_v.min(), fdtd_v.min())), float(max(base_v.max(), fdtd_v.max()))
    for ax, data, title in (
        (axes[0], base_v, "baseline volts"),
        (axes[1], fdtd_v, "FDTD-enabled volts"),
        (axes[2], diff, "absolute diff"),
    ):
        im = ax.imshow(data, cmap="gray" if title != "absolute diff" else "viridis", vmin=None if title == "absolute diff" else vmin, vmax=None if title == "absolute diff" else vmax)
        ax.set_title(title)
        ax.set_axis_off()
        if title == "absolute diff":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[3].imshow(np.clip(rgb, 0.0, 1.0))
    axes[3].set_title("IP sRGB")
    axes[3].set_axis_off()
    fig.suptitle("CameraE2E impact of active FDTD reference LUT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _region_heatmap(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((1, 1), dtype=float)
    xs = sorted({_int(row.get("region_ix")) or 0 for row in rows})
    zs = sorted({_int(row.get("region_iz")) or 0 for row in rows})
    x_index = {value: idx for idx, value in enumerate(xs)}
    z_index = {value: idx for idx, value in enumerate(zs)}
    heatmap = np.zeros((len(zs), len(xs)), dtype=float)
    for row in rows:
        x = _int(row.get("region_ix")) or 0
        z = _int(row.get("region_iz")) or 0
        heatmap[z_index[z], x_index[x]] = _float(row.get("response")) or 0.0
    return heatmap


def _render_html(payload: dict[str, Any]) -> str:
    records_json = json.dumps(payload["records"])
    options = "\n".join(
        f"<option value='{escape(record['sensor_id'])}'>{escape(record['label'])}</option>"
        for record in payload["records"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Sensor DB Selector</title>
  <style>
    body {{ margin: 28px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    h1, h2, h3 {{ color: #102a43; }}
    select {{ font-size: 16px; padding: 8px; max-width: 100%; }}
    .notice {{ border-left: 5px solid #f59e0b; padding: 10px 14px; background: #fffbeb; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 12px; padding: 16px; background: #fbfcfd; }}
    img {{ max-width: 100%; border: 1px solid #d8dee4; border-radius: 8px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px; text-align: left; vertical-align: top; }}
    th {{ background: #eef4f8; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; word-break: break-all; }}
    .small {{ color: #52606d; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Image Sensor DB Selector</h1>
  <p>Select an image sensor to inspect structure, OCL/pixel response, relative illumination, TCAD proxy geometry, CameraE2E impact, and parameter bundle.</p>
  <div class="notice">{escape(payload["panel_note"])}</div>
  <p><b>Active FDTD LUT:</b> <code>{escape(str(payload.get("active_fdtd_lut_path")))}</code></p>
  <label for="sensorSelect"><b>Image sensor:</b></label>
  <select id="sensorSelect">{options}</select>
  <h2 id="title"></h2>
  <div id="facts"></div>
  <div class="grid">
    <section class="card"><h3>1. Sensor Stack Structure</h3><img id="structure" alt="sensor structure"><p class="small">Cross-section and top-view are reconstructed from catalog metadata unless product CAD/GDS is explicitly listed.</p><div id="cadTemplateWrap" style="display:none"><h4>Parametric CAD Template Preview</h4><img id="cadTemplate" alt="cad template preview"><p id="cadTemplateNote" class="small"></p></div></section>
    <section class="card"><h3>2. OCL / Pixel Optical Response</h3><img id="ocl" alt="ocl response"><p class="small">Active FDTD optical absorption / collection proxy. Not measured QE.</p></section>
    <section class="card"><h3>3. Relative Illumination / QE-Like Response</h3><img id="field" alt="field response"><p class="small">Center-normalized active FDTD response versus cos4 reference.</p></section>
    <section class="card"><h3>4. TCAD / Charge Collection</h3><img id="tcad" alt="tcad collection"><p class="small">Generated TCAD profile proxy. Blue donor-like, orange acceptor-like regions.</p></section>
    <section class="card"><h3>5. CameraE2E Impact</h3><img id="impact" alt="camerae2e impact"><p class="small">Reference image impact from active FDTD LUT through sensor/IP.</p></section>
    <section class="card"><h3>6. Parameter Bundle</h3><div id="params"></div><h4>Notes</h4><ul id="notes"></ul></section>
  </div>
  <script>
    const records = {records_json};
    const byId = Object.fromEntries(records.map((record) => [record.sensor_id, record]));
    const select = document.getElementById("sensorSelect");
    function table(rows) {{
      return "<table>" + rows.map(([k, v]) => `<tr><th>${{k}}</th><td>${{v}}</td></tr>`).join("") + "</table>";
    }}
    function code(v) {{ return `<code>${{String(v ?? "-")}}</code>`; }}
    function update() {{
      const record = byId[select.value];
      document.getElementById("title").textContent = record.label;
      document.getElementById("facts").innerHTML = table([
        ["Code", code(record.code)],
        ["Manufacturer", record.manufacturer],
        ["Device", record.device_name],
        ["Pixel pitch", `${{record.pixel_pitch_um ?? "-"}} um`],
        ["Pixel architecture", record.pixel_architecture ?? "-"],
        ["CFA", record.cfa_pattern ?? "-"],
        ["Optical stack height", `${{record.optical_stack_height_um ?? "-"}} um`],
        ["Resolution", `${{record.resolution_mp ?? "-"}} MP`],
        ["CAD source status", record.cad_source_status ?? "-"],
      ]);
      document.getElementById("structure").src = record.images.structure;
      const cadWrap = document.getElementById("cadTemplateWrap");
      if (record.images.cad_template) {{
        cadWrap.style.display = "block";
        document.getElementById("cadTemplate").src = record.images.cad_template;
        document.getElementById("cadTemplateNote").textContent = `${{record.cad_template.template_id}}: ${{record.cad_template.truth_level}}`;
      }} else {{
        cadWrap.style.display = "none";
        document.getElementById("cadTemplate").removeAttribute("src");
      }}
      document.getElementById("ocl").src = record.images.ocl_response;
      document.getElementById("field").src = record.images.field_response;
      document.getElementById("tcad").src = record.images.tcad_collection;
      document.getElementById("impact").src = record.images.camerae2e_impact;
      document.getElementById("params").innerHTML = table(Object.entries(record.parameters).map(([k, v]) => [k, code(v)]));
      document.getElementById("notes").innerHTML = record.notes.map((note) => `<li>${{note}}</li>`).join("");
    }}
    select.addEventListener("change", update);
    update();
  </script>
</body>
</html>
"""


def _ui_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sensor_id": record["sensor_id"],
        "code": record["code"],
        "manufacturer": record["manufacturer"],
        "device_name": record["device_name"],
        "label": _sensor_title(record),
        "pixel_pitch_um": record.get("pixel_pitch_um"),
        "pixel_architecture": record.get("pixel_architecture"),
        "cfa_pattern": record.get("cfa_pattern"),
        "cad_source_status": _cad_source_status(record),
        "optical_stack_height_um": record.get("optical_stack_height_um"),
        "resolution_mp": record.get("resolution_mp"),
    }


def _sensor_notes(record: dict[str, Any], stack: dict[str, Any]) -> list[str]:
    notes = [
        "Generated stack configs are TechInsights-metadata proxy stacks, not calibrated process decks or product CAD.",
        _cad_source_status(record),
        "OCL response panel uses the active FDTD reference LUT unless a per-sensor LUT is generated.",
    ]
    cad_template = _cad_template_info(record)
    if cad_template is not None:
        notes.append(
            f"Matching parametric CAD template available: {cad_template['template_id']} "
            f"({cad_template['truth_level']})."
        )
    calibration = dict(stack.get("calibration_status", {}))
    if calibration.get("note"):
        notes.append(str(calibration["note"]))
    if record.get("has_pdaf"):
        notes.append("Sensor metadata indicates PDAF; generated optical masks remain off unless modeled separately.")
    if record.get("has_dti"):
        notes.append(f"Sensor metadata indicates DTI: {record.get('dti_type') or 'type not specified'}.")
    return notes


def _sensor_title(record: dict[str, Any]) -> str:
    return f"{record.get('code')} - {record.get('manufacturer')} {record.get('device_name')}".strip()


def _cfa_grid_colors(cfa: str) -> list[list[str]]:
    if "rccc" in cfa:
        return [["#ef4444", "#d1d5db"], ["#d1d5db", "#d1d5db"]]
    if "rgbw" in cfa:
        return [["#ef4444", "#22c55e"], ["#3b82f6", "#f9fafb"]]
    return [["#ef4444", "#22c55e"], ["#22c55e", "#3b82f6"]]


def _cfa_label(color: str) -> str:
    return {"#ef4444": "R", "#22c55e": "G", "#3b82f6": "B", "#f9fafb": "W", "#d1d5db": "C"}.get(color, "")


def _draw_top_view_layout(ax: Any, record: dict[str, Any], geom: dict[str, Any]) -> None:
    specs = _derived_specs(record)
    cfa = str(record.get("cfa_pattern") or specs.get("cfa_pattern") or "bayer").lower()
    arch = str(record.get("pixel_architecture") or specs.get("pixel_architecture") or "").lower()
    microlens = str(record.get("microlens_type") or specs.get("microlens_type") or "").lower()
    pitch = _first_float(record.get("pixel_pitch_um"), specs.get("pixel_pitch_um"), geom.get("pitch"), 1.4) or 1.4
    cfa_pitch = _first_float(specs.get("color_filter_pitch_um"), pitch) or pitch
    ocl_pitch = _first_float(specs.get("ocl_pitch_um"), geom.get("pitch"), pitch) or pitch

    if "quad" in cfa or "quad" in arch:
        grid_n = 4
        group = 2
        label_grid = _expanded_group_grid([["R", "G"], ["G", "B"]], group)
        layout_note = "quad Bayer: 2x2 same-color CFA groups"
    elif "nona" in cfa:
        grid_n = 6
        group = 3
        label_grid = _expanded_group_grid([["R", "G"], ["G", "B"]], group)
        layout_note = "nona: 3x3 same-color CFA groups"
    else:
        grid_n = 2
        group = 1
        label_grid = _cfa_label_grid(cfa)
        layout_note = f"{cfa or 'bayer'} CFA"

    for row in range(grid_n):
        for col in range(grid_n):
            label = label_grid[row][col]
            y = grid_n - 1 - row
            ax.add_patch(
                patches.Rectangle(
                    (col, y),
                    1,
                    1,
                    facecolor=_cfa_color(label),
                    edgecolor="#1f2933",
                    linewidth=0.8,
                    alpha=0.9,
                )
            )
            if grid_n <= 4:
                ax.text(col + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10, weight="bold")

    for col in range(grid_n + 1):
        ax.plot([col, col], [0, grid_n], color="#0f172a", linewidth=0.55, alpha=0.5)
    for row in range(grid_n + 1):
        ax.plot([0, grid_n], [row, row], color="#0f172a", linewidth=0.55, alpha=0.5)

    if group > 1:
        for row in range(0, grid_n, group):
            for col in range(0, grid_n, group):
                y = grid_n - group - row
                ax.add_patch(patches.Rectangle((col, y), group, group, facecolor="none", edgecolor="#111827", linewidth=2.0))
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (col + 0.08, y + 0.08),
                        group - 0.16,
                        group - 0.16,
                        boxstyle="round,pad=0.03,rounding_size=0.32",
                        facecolor="none",
                        edgecolor="#06b6d4",
                        linewidth=1.5,
                        linestyle="--",
                    )
                )
        ax.text(
            grid_n / 2,
            grid_n + 0.22,
            "black: CFA group boundary / cyan dashed: CAD-template OCL group",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )

    ax.annotate("", xy=(0, -0.22), xytext=(1, -0.22), arrowprops={"arrowstyle": "<->", "color": "#334155", "lw": 1.0})
    ax.text(0.5, -0.38, f"pixel pitch {pitch:.3g} um", ha="center", va="top", fontsize=8)
    if group > 1:
        ax.annotate("", xy=(0, -0.72), xytext=(group, -0.72), arrowprops={"arrowstyle": "<->", "color": "#334155", "lw": 1.0})
        ax.text(group / 2, -0.88, f"CFA group pitch {cfa_pitch:.3g} um", ha="center", va="top", fontsize=8)
    ax.text(
        grid_n / 2,
        -1.24 if group > 1 else -0.72,
        f"{layout_note}\nOCL pitch metadata: {ocl_pitch:.3g} um; microlens: {microlens or '-'}\nProduct CAD/GDS: not found in DB",
        ha="center",
        va="top",
        fontsize=8,
        color="#334155",
    )
    ax.set_xlim(-0.05, grid_n + 0.05)
    ax.set_ylim(-1.65 if group > 1 else -1.05, grid_n + 0.55)
    ax.set_aspect("equal")
    ax.set_title("Top-view metadata layout")
    ax.set_axis_off()


def _expanded_group_grid(groups: list[list[str]], group_size: int) -> list[list[str]]:
    grid: list[list[str]] = []
    for group_row in groups:
        expanded_row: list[str] = []
        for label in group_row:
            expanded_row.extend([label] * group_size)
        for _ in range(group_size):
            grid.append(list(expanded_row))
    return grid


def _cfa_label_grid(cfa: str) -> list[list[str]]:
    if "rccc" in cfa:
        return [["R", "C"], ["C", "C"]]
    if "rgbw" in cfa:
        return [["R", "G"], ["B", "W"]]
    return [["R", "G"], ["G", "B"]]


def _cfa_color(label: str) -> str:
    return {
        "R": "#ef4444",
        "G": "#22c55e",
        "B": "#3b82f6",
        "W": "#f9fafb",
        "C": "#d1d5db",
    }.get(label.upper(), "#e5e7eb")


def _derived_specs(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    return dict(raw.get("derived_specs", {}))


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _cad_source_status(record: dict[str, Any]) -> str:
    product_cad = _product_cad_paths(record)
    if product_cad:
        return f"Product CAD/layout source listed: {product_cad[0]}"
    return "Product CAD/GDS/OAS not found in DB; structure view is metadata-derived."


def _product_cad_paths(record: dict[str, Any]) -> list[str]:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    source_files = raw.get("source_files", [])
    cad_exts = {".gds", ".gdsii", ".oas", ".oasis", ".dxf", ".step", ".stp", ".brep", ".fcstd"}
    paths: list[str] = []
    for item in source_files if isinstance(source_files, list) else []:
        path = str(item)
        if Path(path).suffix.lower() in cad_exts:
            paths.append(path)
    return paths


def _cad_template_info(record: dict[str, Any]) -> dict[str, str] | None:
    specs = _derived_specs(record)
    cfa = str(record.get("cfa_pattern") or specs.get("cfa_pattern") or "").lower()
    arch = str(record.get("pixel_architecture") or specs.get("pixel_architecture") or "").lower()
    db_root = Path(str(record.get("db_root") or "")).expanduser()
    fdtd_root = db_root.parent if db_root.name in {"sensor_db", "image_sensor_db"} else db_root
    if "quad" in cfa or "quad" in arch:
        preview = fdtd_root / "review_assets" / "cad_quad_2x2_ocl_5x5.svg"
        return {
            "template_id": "quad_2x2_ocl_5x5_crosstalk",
            "preview_svg": str(preview),
            "truth_level": "parametric CAD template, not measured product CAD",
        }
    if "qpd" in cfa or "qpd" in arch:
        preview = fdtd_root / "review_assets" / "cad_qpd_split_pd_2x2.svg"
        return {
            "template_id": "qpd_split_pd_2x2",
            "preview_svg": str(preview),
            "truth_level": "parametric CAD template, not measured product CAD",
        }
    return None


def _nearest_wave(lut: Any, target: float) -> float:
    waves = np.asarray(lut.wavelengths_nm, dtype=float)
    if waves.size == 0:
        waves = np.asarray([_float(row.get("wavelength_nm")) for row in lut.summary_rows if _float(row.get("wavelength_nm"))], dtype=float)
    return float(waves[np.argmin(np.abs(waves - target))]) if waves.size else float(target)


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sensor-id")
    parser.add_argument("--max-sensors", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(args.output_dir, sensor_id=args.sensor_id, max_sensors=args.max_sensors)
    print(result["html"])
