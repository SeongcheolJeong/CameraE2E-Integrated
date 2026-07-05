"""PPT-claim gap audit for the CameraE2E technical overview deck."""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile


def camerae2e_ppt_gap_audit(
    pptx_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the technical-overview PPT claims against current implementation.

    The referenced overview deck is image-only, so this function records deck
    hashes and uses a curated claim ledger derived from the visible slides.
    Claims are classified by implementation/readiness tier rather than by
    marketing intent, which prevents proxy assets from becoming sign-off claims.
    """

    pptx = Path(pptx_path).expanduser()
    source = _pptx_source_evidence(pptx)
    claims = _ppt_claims()
    status_counts = Counter(str(item["status"]) for item in claims)
    area_counts = Counter(str(item["area"]) for item in claims)
    gap_items = [
        item
        for item in claims
        if item["status"] in {"partial", "proxy", "missing", "blocked_external_evidence"}
    ]
    payload = {
        "schema_version": "camerae2e_ppt_gap_audit_v1",
        "source_pptx": str(pptx),
        "source": source,
        "summary": {
            "claim_count": len(claims),
            "gap_count": len(gap_items),
            "status_counts": dict(status_counts),
            "area_counts": dict(area_counts),
        },
        "claims": claims,
        "not_implemented_or_lacking": gap_items,
        "truth_boundary": (
            "This audit maps overview-slide claims to repository evidence. "
            "Items requiring measured calibration, vendor traces, process decks, "
            "or silicon data are intentionally blocked rather than simulated as sign-off."
        ),
    }
    if output_dir is not None:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "ppt_gap_audit.json"
        html_path = root / "ppt_gap_audit.html"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        html_path.write_text(_render_gap_audit_html(payload), encoding="utf-8")
        payload["reports"] = {"json": str(json_path), "html": str(html_path)}
    return payload


def _pptx_source_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "slide_count": 0,
            "media_count": 0,
            "sha256": None,
            "media": [],
        }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with ZipFile(path) as archive:
        names = archive.namelist()
        slides = sorted(
            [
                name
                for name in names
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ],
            key=_slide_sort_key,
        )
        media = sorted(name for name in names if name.startswith("ppt/media/"))
        media_entries = []
        for index, name in enumerate(media, start=1):
            data = archive.read(name)
            media_entries.append(
                {
                    "index": index,
                    "name": name,
                    "bytes": len(data),
                    "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
                }
            )
    return {
        "status": "loaded",
        "slide_count": len(slides),
        "media_count": len(media),
        "sha256": f"sha256:{digest}",
        "media": media_entries,
    }


def _slide_sort_key(name: str) -> int:
    stem = Path(name).stem
    try:
        return int(stem.replace("slide", ""))
    except ValueError:
        return 0


def _ppt_claims() -> list[dict[str, Any]]:
    return [
        _claim(
            1,
            "Scene / color",
            (
                "Spectral scenes, reflectance bases, calibration charts, spatial "
                "patterns, and ROI extraction."
            ),
            "implemented",
            "scene.py, illuminant.py, color.py, ROI helpers, chart/scene tests",
            "Scene capture calibration and measured illumination traces remain external evidence.",
            "Keep as validated research scene generation, not measured capture truth.",
        ),
        _claim(
            1,
            "Optics / lens",
            "Diffraction/WVF, raytrace PSF/OTF, distortion, cos4 falloff, depth/defocus.",
            "partial",
            "optics.py, lens_patents.py, RayOptics geometric PSF registry",
            (
                "RayOptics PSF is geometric; flare/coating/tolerance and measured "
                "PSF are not implemented."
            ),
            (
                "Use diffraction/WVF comparisons and measured PSF evidence before "
                "calibration promotion."
            ),
        ),
        _claim(
            1,
            "Sensor / RAW",
            (
                "CFA, pixel/photodiode, exposure, noise, FDTD optical LUT, TCAD "
                "collection proxy, RAW output."
            ),
            "partial",
            "sensor.py, fdtd_sensor.py, tcad_sensor.py, image_sensor_db.py, dataset.py",
            (
                "Sensor DB is metadata-derived; per-sensor calibrated process deck "
                "and sensor-specific LUTs are missing."
            ),
            (
                "Use analytic sweeps for exploration and FDTD/TCAD batches only for "
                "short-listed candidates."
            ),
        ),
        _claim(
            1,
            "IP / ISP",
            "Demosaic, sensor conversion, AWB, CCM, tone/gamma, display render.",
            "implemented",
            "ip.py, camera.py, system_faca.py",
            "Advanced production ISP blocks such as HDR/TNR tuning are proxy or not complete.",
            "Keep current ISP as research image-processing pipeline.",
        ),
        _claim(
            1,
            "HW ISP timing",
            "Rolling shutter, line timing, stage latency, AE/AWB delay, DMA/queue/bandwidth.",
            "proxy",
            "hwisp.py, hwisp_db.py, goal gate HW ISP smoke",
            (
                "Profiles are seed/public-derived; no board trace, hardware counter, "
                "or vendor BSP timing evidence."
            ),
            "Block latency sign-off until board traces and 3A telemetry are attached.",
        ),
        _claim(
            1,
            "Metrics / perception",
            "MTF/OTF, Delta E, SNR/VSNR, perceptual difference, mAP/mIoU task metrics.",
            "partial",
            "metrics.py, iso.py, scielab.py, perception.py, task_perception.py",
            (
                "Task model profiles exist, but no trained model loop or "
                "dataset-specific evaluation evidence is included."
            ),
            (
                "Treat model outputs as adapter metrics until training/evaluation "
                "manifests are provided."
            ),
        ),
        _claim(
            2,
            "DNG read/write",
            "Sensor and RAW module table includes DNG RAW read/write.",
            "partial",
            (
                "fileio.py supports DNG read and sensor_dng_read; dataset.py exports "
                "NPZ/TIFF/container"
            ),
            "Standards-compliant DNG writer remains intentionally missing.",
            (
                "Use optional DNG-like ZIP container for research packaging; keep "
                "true DNG writer as separate milestone."
            ),
        ),
        _claim(
            2,
            "Reports and dashboards",
            "HTML/PPT/JSON reports, plots, metric tables, task results.",
            "partial",
            "reports/camerae2e_goal, tools/render_* report scripts",
            "Reports are evidence snapshots and not automated product QA dashboards.",
            "Continue adding machine-readable gap/evidence reports.",
        ),
        _claim(
            3,
            "System FACA",
            "Field, angle, illuminant, color, artifact, control delay, use-case scenario analysis.",
            "implemented",
            "system_faca.py and tests/unit/test_system_faca.py",
            "Field/angle fidelity is only as strong as attached optics/sensor assets.",
            "Use FACA for ranking and sensitivity, not final accuracy without calibration.",
        ),
        _claim(
            3,
            "Optimization loops",
            (
                "Discrete/continuous design variables, constraints, Pareto frontier, "
                "sweep table, updated design loop."
            ),
            "implemented",
            (
                "optimization.py with grid/random/LHS/evolutionary/GP/surrogate, "
                "constraints, Pareto, escalation plan"
            ),
            "Hardware-in-loop and measured calibration objectives are still external.",
            "Route selected candidates through physics escalation and calibration evidence gates.",
        ),
        _claim(
            3,
            "Training dataset outputs",
            "RAW, labels, metadata, versioned/indexed training datasets.",
            "implemented",
            "dataset.py NPZ/metadata/labels/perception index/stage outputs/container/uncertainty",
            "Automatic label synthesis for arbitrary scenes is not implemented.",
            "Accept caller/synthetic-helper labels and preserve lineage.",
        ),
        _claim(
            4,
            "RAW data formats",
            (
                "Bayer RAW, OI photons/irradiance, sensor volts, sRGB, per-stage "
                "outputs, metadata, annotations."
            ),
            "partial",
            "dataset.py exports RAW NPZ, sensor_digital, RGB preview, per-stage NPZ, labels JSON",
            "OI photon/irradiance export and EXR/HDR scene files are not complete dataset formats.",
            "Add OI/stage exports incrementally without claiming measured RAW.",
        ),
        _claim(
            4,
            "Labels and metrics",
            (
                "Boxes, masks, semantic masks, keypoints, polylines, stage tags, "
                "quality and perception metrics."
            ),
            "partial",
            (
                "dataset.py preserves caller label payloads; task_perception.py "
                "supports boxes/masks/pose/tracking"
            ),
            "No general automatic label inference or mask generation from arbitrary RGB scenes.",
            "Use synthetic helpers or caller-provided labels only.",
        ),
        _claim(
            4,
            "Dataset split and regression baseline",
            (
                "Train/val/test split, reference images, calibrated pipeline "
                "baseline, uncertainty/confidence map."
            ),
            "partial",
            "dataset split policy, checksums, optional uncertainty proxy maps",
            "Calibrated baseline and true uncertainty posterior require measured references.",
            "Keep proxy confidence maps separate from calibration confidence.",
        ),
        _claim(
            5,
            "Physics fidelity ladder",
            (
                "Empirical proxy through ISETCam numerical, RayOptics, FDTD, TCAD, "
                "measured calibration."
            ),
            "implemented",
            "db_catalog.py, calibration.py, physics_pipeline.py, goal gate sign-off guard",
            "Measured silicon sign-off evidence is absent by design.",
            "Continue blocking calibrated tier until required external evidence validates.",
        ),
        _claim(
            5,
            "FDTD/TCAD lineage",
            "FDTD LUT and TCAD/DEVSIM collection proxy with accuracy gate.",
            "blocked_external_evidence",
            "physics_pipeline.py detects active FDTD/TCAD run mismatch",
            "Current active FDTD LUT and TCAD generation map are from different run roots.",
            "Regenerate TCAD generation map from active FDTD LUT or repoint both to one lineage.",
        ),
        _claim(
            6,
            "Repository evidence artifacts",
            "Unit tests, parity cases, HTML/JSON reports, design decisions, risk records.",
            "partial",
            "pytest suite, parity_report.py, render_* tools, reports/camerae2e_goal",
            "Risk/design-decision records are not yet first-class generated artifacts.",
            "Promote PPT gap audit and goal gate reports as evidence snapshots.",
        ),
        _claim(
            6,
            "External data catalogs",
            (
                "Lens DB, Sensor FDTD LUT, TCAD/DEVSIM DB, HW ISP profiles, task "
                "model profiles, MATLAB parity baselines."
            ),
            "partial",
            "db_catalog.py registry and bundled/external asset discovery",
            "Several catalogs are proxy, metadata-derived, stale, or missing measured evidence.",
            "Use registry readiness and calibration evidence plan before stronger claims.",
        ),
    ]


def _claim(
    slide: int,
    area: str,
    ppt_claim: str,
    status: str,
    evidence: str,
    gap: str,
    action: str,
) -> dict[str, Any]:
    tier = {
        "implemented": "validated",
        "partial": "available",
        "proxy": "proxy",
        "missing": "missing",
        "blocked_external_evidence": "calibration_required",
    }.get(status, "available")
    return {
        "slide": int(slide),
        "area": area,
        "ppt_claim": ppt_claim,
        "status": status,
        "readiness_tier": tier,
        "current_evidence": evidence,
        "gap_detail": gap,
        "recommended_action": action,
    }


def _render_gap_audit_html(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload.get("claims", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('slide')))}</td>"
            f"<td>{html.escape(str(item.get('area')))}</td>"
            f"<td>{html.escape(str(item.get('status')))}</td>"
            f"<td>{html.escape(str(item.get('ppt_claim')))}</td>"
            f"<td>{html.escape(str(item.get('gap_detail')))}</td>"
            f"<td>{html.escape(str(item.get('recommended_action')))}</td>"
            "</tr>"
        )
    summary = html.escape(json.dumps(payload.get("summary", {}), sort_keys=True))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CameraE2E PPT Gap Audit</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "margin:24px;color:#172033}table{border-collapse:collapse;width:100%;"
        "font-size:13px}td,th{border:1px solid #ccd3df;padding:7px;vertical-align:top}"
        "th{background:#eef3fb;text-align:left}code{background:#f6f8fb;padding:2px 4px}"
        "</style></head><body>"
        "<h1>CameraE2E PPT Gap Audit</h1>"
        f"<p><strong>Source:</strong> {html.escape(str(payload.get('source_pptx')))}</p>"
        f"<p><strong>Summary:</strong> <code>{summary}</code></p>"
        "<table><thead><tr><th>Slide</th><th>Area</th><th>Status</th>"
        "<th>PPT claim</th><th>Gap</th><th>Action</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


cameraE2EPPTGapAudit = camerae2e_ppt_gap_audit  # noqa: N816
