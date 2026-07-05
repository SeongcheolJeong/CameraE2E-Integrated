#!/usr/bin/env python3
"""Synchronize FDTD stack CFA thickness from image_sensor_db optical/CFA models.

The TechInsights-derived FDTD stack configs split a lumped optical stack into
microlens, CFA, and passivation proxy layers. The CameraE2E CFA DB can contain a
better CFA thickness estimate or a clearly tagged proxy default. This script
updates generated stack configs so solver runs use the same CFA thickness basis
as the CameraE2E material/color exports, while keeping all provenance explicit.
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


ROOT = Path(__file__).resolve().parent
DEFAULT_STACK_DIR = ROOT / "image_sensor_db" / "generated_stack_configs"
DEFAULT_OPTICAL_QE_DB = ROOT / "image_sensor_db" / "optical_qe_db"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "camera_e2e_sensor_lut_package" / "fdtd_stack_cfa_sync"

ROW_COLUMNS = [
    "slug",
    "stack_path",
    "optical_model_path",
    "status",
    "old_cfa_thickness_um",
    "new_cfa_thickness_um",
    "old_passivation_thickness_um",
    "new_passivation_thickness_um",
    "old_optical_top_um",
    "new_optical_top_um",
    "preserved_optical_stack_height",
    "cfa_thickness_source_kind",
    "cfa_thickness_confidence",
    "cfa_proxy_applicability",
    "cfa_proxy_library_id",
    "cfa_proxy_enabled",
    "product_lut_gate",
    "notes",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def value_info(obj: Any) -> tuple[float, str, str]:
    if isinstance(obj, dict):
        return (
            as_float(obj.get("value")),
            str(obj.get("source_kind", "")),
            "" if obj.get("confidence") is None else str(obj.get("confidence", "")),
        )
    return as_float(obj), "", ""


def cfa_thickness_from_model(model: dict[str, Any]) -> tuple[float, str, str, str]:
    proxy = model.get("cfa_proxy_nk", {}) if isinstance(model.get("cfa_proxy_nk"), dict) else {}
    value, source_kind, confidence = value_info(proxy.get("thickness_um"))
    if math.isfinite(value) and 0.05 <= value <= 3.0:
        return value, source_kind or "cfa_proxy_nk.thickness_um", confidence, "cfa_proxy_nk.thickness_um"
    optical = model.get("optical", {}) if isinstance(model.get("optical"), dict) else {}
    cfa = optical.get("cfa", {}) if isinstance(optical.get("cfa"), dict) else {}
    value, source_kind, confidence = value_info(cfa.get("representative_thickness_um"))
    if math.isfinite(value) and 0.05 <= value <= 3.0:
        return value, source_kind or "optical.cfa.representative_thickness_um", confidence, "optical.cfa.representative_thickness_um"
    return math.nan, "", "", ""


def optical_top_um(geometry: dict[str, Any]) -> float:
    return (
        as_float(geometry.get("lens_height"), 0.0)
        + as_float(geometry.get("cfa_thickness"), 0.0)
        + as_float(geometry.get("passivation_thickness"), 0.0)
    )


def should_preserve_lumped_optical_height(stack: dict[str, Any], old_top: float) -> bool:
    source = stack.get("techinsights_source", {}) if isinstance(stack.get("techinsights_source"), dict) else {}
    derived_sources = source.get("derived_geometry_sources", {}) if isinstance(source.get("derived_geometry_sources"), dict) else {}
    method = str(derived_sources.get("lumped_optical_stack", ""))
    optical_height = as_float(source.get("derived_specs", {}).get("optical_stack_height_um") if isinstance(source.get("derived_specs"), dict) else None)
    return "lumped_from_report_optical_stack_height" in method or (math.isfinite(optical_height) and abs(optical_height - old_top) <= 0.08)


MEASURED_OR_SENSOR_DERIVED_CFA_THICKNESS_SOURCES = {"extracted", "derived_from_extracted_range"}


def sync_one(
    stack_path: Path,
    optical_db: Path,
    *,
    dry_run: bool,
    include_proxy_defaults: bool,
) -> dict[str, Any]:
    slug = stack_path.stem
    stack = read_json(stack_path)
    model_path = optical_db / "models" / f"{slug}.json"
    model = read_json(model_path)
    row: dict[str, Any] = {
        "slug": slug,
        "stack_path": repo_rel(stack_path),
        "optical_model_path": repo_rel(model_path) if model_path.exists() else "",
        "status": "SKIP",
        "notes": "",
    }
    if not stack:
        row["status"] = "FAIL"
        row["notes"] = "stack JSON missing or invalid"
        return row
    if not model:
        row["notes"] = "no optical_qe_db model for stack slug"
        return row

    geometry = stack.get("geometry_um", {}) if isinstance(stack.get("geometry_um"), dict) else {}
    if not geometry:
        row["status"] = "FAIL"
        row["notes"] = "stack has no geometry_um"
        return row

    new_cfa, source_kind, confidence, source_field = cfa_thickness_from_model(model)
    proxy = model.get("cfa_proxy_nk", {}) if isinstance(model.get("cfa_proxy_nk"), dict) else {}
    row.update(
        {
            "cfa_thickness_source_kind": source_kind,
            "cfa_thickness_confidence": confidence,
            "cfa_proxy_applicability": proxy.get("applicability", ""),
            "cfa_proxy_library_id": proxy.get("library_id", ""),
            "cfa_proxy_enabled": proxy.get("enabled", ""),
            "product_lut_gate": "FAIL",
        }
    )
    if not math.isfinite(new_cfa):
        row["notes"] = "no usable CFA thickness in optical_qe_db model"
        return row
    if source_kind not in MEASURED_OR_SENSOR_DERIVED_CFA_THICKNESS_SOURCES and not include_proxy_defaults:
        row["notes"] = (
            f"CFA thickness source '{source_kind}' is not sensor-derived; "
            "leaving solver geometry unchanged unless --include-proxy-defaults is set"
        )
        return row

    old_cfa = as_float(geometry.get("cfa_thickness"))
    old_pass = as_float(geometry.get("passivation_thickness"))
    old_top = optical_top_um(geometry)
    preserve_top = should_preserve_lumped_optical_height(stack, old_top)
    new_pass = old_pass
    if preserve_top:
        lens = as_float(geometry.get("lens_height"), 0.25)
        new_pass = round(max(0.05, old_top - lens - new_cfa), 4)
    new_top = as_float(geometry.get("lens_height"), 0.0) + new_cfa + new_pass
    row.update(
        {
            "old_cfa_thickness_um": "" if not math.isfinite(old_cfa) else old_cfa,
            "new_cfa_thickness_um": round(new_cfa, 4),
            "old_passivation_thickness_um": "" if not math.isfinite(old_pass) else old_pass,
            "new_passivation_thickness_um": new_pass,
            "old_optical_top_um": round(old_top, 4),
            "new_optical_top_um": round(new_top, 4),
            "preserved_optical_stack_height": preserve_top,
        }
    )
    if math.isfinite(old_cfa) and abs(old_cfa - new_cfa) < 1e-6 and abs(old_pass - new_pass) < 1e-6:
        row["status"] = "UNCHANGED"
        row["notes"] = "stack already matches optical_qe_db CFA thickness"
        return row

    row["status"] = "DRY_RUN" if dry_run else "UPDATED"
    row["notes"] = f"geometry_um.cfa_thickness synchronized from {source_field}; product gate remains FAIL because n,k/QE are not measured"
    if not dry_run:
        geometry["cfa_thickness"] = round(new_cfa, 4)
        geometry["passivation_thickness"] = new_pass
        stack["geometry_um"] = geometry
        stack["cfa_db_sync"] = {
            "schema": "fdtd_stack_cfa_db_sync_v1",
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "source_optical_model": repo_rel(model_path),
            "source_field": source_field,
            "source_kind": source_kind,
            "confidence": confidence,
            "old_cfa_thickness_um": row["old_cfa_thickness_um"],
            "new_cfa_thickness_um": row["new_cfa_thickness_um"],
            "old_passivation_thickness_um": row["old_passivation_thickness_um"],
            "new_passivation_thickness_um": row["new_passivation_thickness_um"],
            "preserved_optical_stack_height": preserve_top,
            "product_lut_gate": "FAIL",
            "notes": row["notes"],
        }
        notes = list(stack.get("accuracy_notes", [])) if isinstance(stack.get("accuracy_notes"), list) else []
        sync_note = (
            "geometry_um.cfa_thickness is synchronized from image_sensor_db optical_qe_db CFA model; "
            "this is still proxy/research data unless measured CFA n,k and measured spectral QE are imported."
        )
        if sync_note not in notes:
            stack["accuracy_notes"] = [sync_note, *notes]
        write_json(stack_path, stack)
    return row


def html_cell(value: Any) -> str:
    return html.escape(str(value))


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(column)}</th>" for column in columns)
    body = []
    for row in rows[:240]:
        body.append("<tr>" + "".join(f"<td>{html_cell(row.get(column, ''))}</td>" for column in columns) + "</tr>")
    if len(rows) > 240:
        body.append(f"<tr><td colspan=\"{len(columns)}\">... {len(rows) - 240} more rows in CSV</td></tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    css = """
body{margin:0;background:#071116;color:#e8f4f8;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:28px}.muted{color:#9eb7c2}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.card{border:1px solid #1d4656;background:#0c1b23;border-radius:8px;padding:14px}.metric{font-size:26px;font-weight:800}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}th,td{border-bottom:1px solid #17303b;padding:8px;text-align:left;vertical-align:top}th{color:#9fe8ff;background:#0a1a22}
"""
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FDTD Stack CFA Sync</title><style>{css}</style></head><body><main>
<h1>FDTD Stack CFA Sync</h1>
<p class="muted">Generated {html_cell(payload.get("generated_at", ""))}. This sync aligns solver stack CFA thickness with image_sensor_db optical/CFA DB inputs.</p>
<div class="grid">
<div class="card"><div class="metric">{html_cell(payload.get("stack_count", 0))}</div><div class="muted">stacks checked</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("updated_count", 0))}</div><div class="muted">updated</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("unchanged_count", 0))}</div><div class="muted">unchanged</div></div>
<div class="card"><div class="metric">{html_cell(payload.get("fail_count", 0))}</div><div class="muted">failed</div></div>
</div>
{html_table(rows, ROW_COLUMNS)}
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def build_sync(args: argparse.Namespace) -> dict[str, Any]:
    stack_dir = args.stack_dir.resolve()
    optical_db = args.optical_qe_db.resolve()
    output_dir = args.output_dir.resolve()
    selected = {item.strip() for item in args.slugs.split(",") if item.strip()} if args.slugs else set()
    rows: list[dict[str, Any]] = []
    for stack_path in sorted(stack_dir.glob("*.json")):
        if selected and stack_path.stem not in selected:
            continue
        rows.append(
            sync_one(
                stack_path,
                optical_db,
                dry_run=args.dry_run,
                include_proxy_defaults=args.include_proxy_defaults,
            )
        )
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.get("status", "")] = status_counts.get(row.get("status", ""), 0) + 1
    payload = {
        "schema": "fdtd_stack_cfa_db_sync_report_v1",
        "artifact_role": "solver_stack_cfa_db_alignment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stack_dir": repo_rel(stack_dir),
        "optical_qe_db": repo_rel(optical_db),
        "dry_run": args.dry_run,
        "stack_count": len(rows),
        "updated_count": status_counts.get("UPDATED", 0),
        "dry_run_update_count": status_counts.get("DRY_RUN", 0),
        "unchanged_count": status_counts.get("UNCHANGED", 0),
        "skip_count": status_counts.get("SKIP", 0),
        "fail_count": status_counts.get("FAIL", 0),
        "status_counts": status_counts,
        "product_lut_ready": False,
        "validation": {
            "schema": "fdtd_stack_cfa_db_sync_validation_v1",
            "pass": status_counts.get("FAIL", 0) == 0 and len(rows) > 0,
            "status": "FDTD_STACK_CFA_SYNC_READY_PRODUCT_BLOCKED",
            "issues": [row for row in rows if row.get("status") == "FAIL"],
        },
        "outputs": {
            "json": repo_rel(output_dir / "fdtd_stack_cfa_sync.json"),
            "csv": repo_rel(output_dir / "fdtd_stack_cfa_sync.csv"),
            "html": repo_rel(output_dir / "index.html"),
        },
    }
    write_csv(output_dir / "fdtd_stack_cfa_sync.csv", rows, ROW_COLUMNS)
    write_json(output_dir / "fdtd_stack_cfa_sync.json", payload)
    write_html(output_dir / "index.html", payload, rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--optical-qe-db", type=Path, default=DEFAULT_OPTICAL_QE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slugs", default="", help="Comma-separated slug filter.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-proxy-defaults",
        action="store_true",
        help="Also overwrite stack CFA thickness with generic proxy defaults. Disabled by default to avoid adding stronger assumptions.",
    )
    args = parser.parse_args()
    print(json.dumps(build_sync(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
