"""Render CameraE2E asset registry and goal-readiness reports."""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyisetcam import (
    camerae2e_db_manifest,
    camerae2e_db_validate,
    camerae2e_physics_pipeline_plan,
    camerae2e_physics_simulation_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/camerae2e_goal"))
    args = parser.parse_args()
    outputs = render_report(args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


def render_report(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = camerae2e_db_manifest()
    validation = camerae2e_db_validate()
    physics_plan = camerae2e_physics_pipeline_plan()
    physics_simulation = camerae2e_physics_simulation_manifest()
    readiness = {
        "schema_version": "camerae2e_goal_readiness_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_pptx": (
            "/Users/seongcheoljeong/Documents/CameraE2E/outputs/"
            "camerae2e-technical-overview-images.pptx"
        ),
        "source_doc": "docs/camerae2e-technical-overview.md",
        "goal": (
            "Research-grade E2E optimization platform plus RAW data factory "
            "for perception training."
        ),
        "capability_matrix": _capability_matrix(manifest, validation),
        "asset_manifest": manifest,
        "validation": validation,
        "physics_pipeline_plan": physics_plan,
        "physics_simulation_manifest": physics_simulation,
    }
    readiness_json = output_dir / "readiness.json"
    readiness_html = output_dir / "readiness.html"
    registry_json = output_dir / "asset_registry.json"
    registry_html = output_dir / "asset_registry.html"
    physics_plan_json = output_dir / "physics_pipeline_plan.json"
    physics_simulation_json = output_dir / "physics_simulation_manifest.json"
    physics_simulation_html = output_dir / "physics_simulation_manifest.html"
    readiness_json.write_text(
        json.dumps(_jsonable(readiness), indent=2, sort_keys=True), encoding="utf-8"
    )
    registry_json.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    physics_plan_json.write_text(
        json.dumps(_jsonable(physics_plan), indent=2, sort_keys=True), encoding="utf-8"
    )
    physics_simulation_json.write_text(
        json.dumps(_jsonable(physics_simulation), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    physics_simulation_html.write_text(
        _render_physics_simulation_html(physics_simulation), encoding="utf-8"
    )
    readiness_html.write_text(_render_readiness_html(readiness), encoding="utf-8")
    registry_html.write_text(_render_registry_html(manifest, validation), encoding="utf-8")
    return {
        "readiness_json": readiness_json,
        "readiness_html": readiness_html,
        "asset_registry_json": registry_json,
        "asset_registry_html": registry_html,
        "physics_pipeline_plan_json": physics_plan_json,
        "physics_simulation_json": physics_simulation_json,
        "physics_simulation_html": physics_simulation_html,
    }


def _capability_matrix(
    manifest: Mapping[str, Any], validation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    status_by_name = {entry["name"]: entry for entry in manifest.get("entries", [])}
    has_stale = bool(validation.get("stale_dependency_count", 0))
    return [
        {
            "area": "Scene",
            "tier": "validated",
            "implemented": (
                "Spectral scene constructors, chart scenes, RGB/multispectral "
                "import, illuminant control."
            ),
            "remaining": "Measured scene/illumination capture remains external to v1.",
        },
        {
            "area": "Optics / RayOptics",
            "tier": status_by_name.get("lens_patents_active", {}).get("readiness_tier", "proxy"),
            "implemented": "Lens DB ingestion and RayOptics geometric PSF execution.",
            "remaining": (
                "Diffraction/wave-optics sign-off, flare/ghost/coating, "
                "manufacturing tolerance."
            ),
        },
        {
            "area": "Image Sensor",
            "tier": "proxy",
            "implemented": (
                "CFA/pixel/exposure/noise/RAW path, CFA preset/Quad Bayer selector, "
                "analytic shared-OCL group equalization proxy, and image-sensor selector DB."
            ),
            "remaining": (
                "Per-sensor calibrated process decks, measured n,k, CAD/GDS, "
                "sensor-specific FDTD LUTs."
            ),
        },
        {
            "area": "FDTD Optical LUT",
            "tier": status_by_name.get("fdtd_sensor_lut_active", {}).get("readiness_tier", "proxy"),
            "implemented": "QE/field/crosstalk proxy LUT ingestion and physics sanity checks.",
            "remaining": (
                "Full convergence, localized crosstalk source experiments, "
                "measured optical stack calibration."
            ),
        },
        {
            "area": "TCAD / DEVSIM",
            "tier": status_by_name.get("tcad_sensor_db_active", {}).get(
                "readiness_tier", "calibration_required"
            ),
            "implemented": (
                "Generation-map ingestion, split-PD collection-current proxy, "
                "accuracy gate."
            ),
            "remaining": (
                "Carrier transport calibration, dark/noise/lag/full-well hooks, "
                "active FDTD/TCAD lineage closure."
            ),
            "warning": status_by_name.get("tcad_sensor_db_active", {}).get("stale_reason"),
        },
        {
            "area": "HW ISP",
            "tier": status_by_name.get("hwisp_parameter_profiles", {}).get(
                "readiness_tier", "proxy"
            ),
            "implemented": "Rolling shutter, stage latency, queue, DMA, delayed AE/AWB simulation.",
            "remaining": (
                "Board/vendor trace calibration, AF/HDR/TNR/multi-camera "
                "contention detail."
            ),
        },
        {
            "area": "Metrics",
            "tier": "validated",
            "implemented": "MTF, ISO12233, Delta E, SCIELAB, VSNR, SQRI, comparison metrics.",
            "remaining": "Product-specific metric weighting and pass/fail thresholds.",
        },
        {
            "area": "Optimization",
            "tier": "validated",
            "implemented": (
                "Preset parameter-space catalog plus deterministic grid, random, "
                "Latin-hypercube candidate planning, and score-ranked evolutionary "
                "plus discrete surrogate-guided search over "
                "dot-path camera parameters using FACA metric objectives, constraints, "
                "Pareto front, selected scenarios, parameter-lineage evidence, "
                "pixel geometry/CFA preset/Quad Bayer/readout/binning/analytic OCL/"
                "FDTD-OCL/noise/optics-PSF configure "
                "targets, candidate budget controls, and preflight parameter-space "
                "validation."
            ),
            "remaining": (
                "True multi-factor readout/remosaic binning, OCL process-stack "
                "calibration, and closed-loop hardware calibration."
            ),
        },
        {
            "area": "Perception",
            "tier": status_by_name.get("task_perception_model_profiles", {}).get(
                "readiness_tier", "available"
            ),
            "implemented": (
                "Detection/segmentation/classification/pose/tracking adapters "
                "and robustness sweeps."
            ),
            "remaining": "Model-specific training loops and dataset-specific accuracy calibration.",
        },
        {
            "area": "RAW Data Factory",
            "tier": "validated",
            "implemented": (
                "Dataset manifest, metadata JSONL, deterministic RAW NPZ, "
                "optimization-case export, parameter-lineage evidence, split, checksum, "
                "RGB preview, caller-provided labels, RAW-aware perception training "
                "manifest, YOLO-style RGB preview/label view, ADAS/KITTI YOLO demo "
                "export, and focal-ratio pinhole crop/resize camera-spec variant "
                "re-capture."
            ),
            "remaining": "DNG writer and automatic label synthesis are intentionally outside v1.",
        },
        {
            "area": "DB/LUT Registry",
            "tier": "validated" if not has_stale else "calibration_required",
            "implemented": (
                "Manifest, readiness tier, provenance, dependency lineage, "
                "stale dependency detection, calibration evidence manifest, "
                "and readiness promotion plan."
            ),
            "remaining": (
                "Attach real measured evidence before promoting "
                "proxy/calibration_required assets."
            ),
        },
        {
            "area": "External Pipeline",
            "tier": "calibration_required" if has_stale else "proxy",
            "implemented": (
                "FDTD/TCAD and RayOptics source/config/small fixtures are merged "
                "under simulations/, active assets are discoverable from one "
                "registry, and the goal-level evidence gate covers registry, "
                "physics lineage, simulation manifest, FACA, optimization, RAW "
                "export, ADAS/KITTI demo, camera-spec variants, and sign-off guard."
            ),
            "remaining": (
                "External generated-run lineage closure and product-calibrated "
                "data generation."
            ),
        },
    ]


def _render_readiness_html(readiness: Mapping[str, Any]) -> str:
    rows = []
    for item in readiness["capability_matrix"]:
        warning = item.get("warning") or ""
        rows.append(
            "<tr>"
            f"<td>{_e(item['area'])}</td>"
            f"<td><span class='tier tier-{_e(item['tier'])}'>{_e(item['tier'])}</span></td>"
            f"<td>{_e(item['implemented'])}</td>"
            f"<td>{_e(item['remaining'])}</td>"
            f"<td>{_e(warning)}</td>"
            "</tr>"
        )
    return _html_page(
        "CameraE2E Goal Readiness",
        f"""
        <p class="lead">{_e(readiness["goal"])}</p>
        <p>Source: <code>{_e(readiness["source_pptx"])}</code>
        and <code>{_e(readiness["source_doc"])}</code></p>
        <table>
          <thead><tr><th>Area</th><th>Tier</th><th>Implemented</th><th>Remaining</th><th>Warning</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """,
    )


def _render_registry_html(manifest: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    rows = []
    for entry in manifest.get("entries", []):
        rows.append(
            "<tr>"
            f"<td>{_e(entry['name'])}</td>"
            f"<td>{_e(entry['family'])}</td>"
            f"<td>{_e(entry['role'])}</td>"
            f"<td>{_e(entry['status'])}</td>"
            f"<td><span class='tier tier-{_e(entry['readiness_tier'])}'>"
            f"{_e(entry['readiness_tier'])}</span></td>"
            f"<td><code>{_e(entry.get('path'))}</code></td>"
            f"<td>{_e(entry.get('stale_reason') or '')}</td>"
            "</tr>"
        )
    validation_banner = (
        f"<p class='warn'>Validation warnings: {validation['warning_count']}; "
        f"stale dependencies: {validation['stale_dependency_count']}</p>"
    )
    return _html_page(
        "CameraE2E Asset Registry",
        validation_banner
        + f"""
        <table>
          <thead><tr><th>Name</th><th>Family</th><th>Role</th>
          <th>Status</th><th>Tier</th><th>Path</th>
          <th>Stale reason</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """,
    )


def _render_physics_simulation_html(manifest: Mapping[str, Any]) -> str:
    rows = []
    for stage in manifest.get("stages", []):
        rows.append(
            "<tr>"
            f"<td>{_e(stage.get('stage_id'))}</td>"
            f"<td>{_e(stage.get('family'))}</td>"
            f"<td>{_e(stage.get('status'))}</td>"
            f"<td><span class='tier tier-{_e(stage.get('readiness_tier'))}'>"
            f"{_e(stage.get('readiness_tier'))}</span></td>"
            f"<td><code>{_e(stage.get('workspace_root'))}</code></td>"
            f"<td>{len(stage.get('commands', []))}</td>"
            f"<td>{_e(stage.get('truth_boundary'))}</td>"
            "</tr>"
        )
    summary = html.escape(json.dumps(manifest.get("summary", {}), sort_keys=True))
    return _html_page(
        "CameraE2E Physics Simulation Manifest",
        f"""
        <p class="lead">FDTD/TCAD/RayOptics source, fixtures, active artifacts,
        CameraE2E import modules, and refresh commands.</p>
        <p>Summary: <code>{summary}</code></p>
        <table>
          <thead><tr><th>Stage</th><th>Family</th><th>Status</th>
          <th>Tier</th><th>Workspace</th><th>Commands</th>
          <th>Truth Boundary</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """,
    )


def _html_page(title: str, body: str) -> str:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_e(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 32px; color: #17202a; }}
.lead {{ font-size: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
th, td {{ border: 1px solid #d5d8dc; padding: 8px 10px; vertical-align: top; font-size: 13px; }}
th {{ background: #f4f6f7; text-align: left; }}
code {{ font-size: 12px; word-break: break-all; }}
.tier {{ display: inline-block; padding: 2px 7px; border-radius: 4px;
  background: #edf2f7; font-weight: 600; }}
.tier-validated, .tier-calibrated {{ background: #d5f5e3; color: #145a32; }}
.tier-proxy, .tier-calibration_required {{ background: #fdebd0; color: #7d3c00; }}
.tier-missing {{ background: #fadbd8; color: #78281f; }}
.warn {{ color: #7d3c00; font-weight: 600; }}
</style>
</head>
<body>
<h1>{_e(title)}</h1>
{body}
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
