#!/usr/bin/env python3
"""Compare materialized image-sensor design variants against the baseline.

This creates a sweep-style comparison table for the Pixel Studio. It only uses
completed local artifacts; missing variant outputs remain visible as planned
or partial runs instead of being silently treated as data.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_CONFIG = ROOT / "configs" / "image_sensor_pixel_studio_reference.json"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "image_sensor_design_variants_reference"
DEFAULT_VARIANT_MANIFEST = DEFAULT_OUTPUT_DIR / "variant_run_manifest.json"

CASE_OUTPUT_KEYS = {
    "center": "devsim_center",
    "cra10x": "devsim_cra10x",
    "edge20x": "devsim_edge20x",
}

STAGE_MARKERS = {
    "meep_fdtd": ("fdtd_generation", "tcad_generation_map_2d.npz"),
    "convergence_gate": ("convergence", "convergence_report.json"),
    "gmsh_mesh": ("gmsh_mesh", "split_pixel_2d.msh"),
    "devsim_weighting": ("devsim_weighting", "weighting_potential_2d_summary.json"),
    "devsim_native_response_sweep": ("devsim_native_response_sweep", "native_response_sweep_manifest.json"),
    "design_viewer": ("design_viewer", "manifest.json"),
    "gw_lut": ("gw_coupling", "gw_coupling_manifest.json"),
    "studio": ("studio", "studio_manifest.json"),
}

CSV_COLUMNS = [
    "variant_id",
    "variant_label",
    "variant_state",
    "case",
    "cra_x_deg",
    "baseline_total_photo_delta_a_per_cm",
    "variant_total_photo_delta_a_per_cm",
    "total_photo_delta_rel_change",
    "baseline_split_phase_x_proxy",
    "variant_split_phase_x_proxy",
    "split_phase_delta",
    "split_phase_sign_changed",
    "baseline_left_photo_delta_a_per_cm",
    "variant_left_photo_delta_a_per_cm",
    "baseline_right_photo_delta_a_per_cm",
    "variant_right_photo_delta_a_per_cm",
    "terminal_balance_illuminated_a_per_cm",
    "gw_total_reference_scaled_rel_error",
    "gw_mesh_total_reference_scaled_rel_error",
    "gw_devsim_laplace_total_reference_scaled_rel_error",
    "gw_split_phase_error",
    "gw_mesh_split_phase_error",
    "gw_devsim_laplace_split_phase_error",
    "parameter_overrides",
    "completed_stages",
    "missing_stages",
    "product_lut_ready",
    "summary_json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def abs_path(config_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    return (ROOT / path).resolve()


def rel_from_root(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def split_total(summary: dict[str, Any]) -> float | None:
    left = safe_float(summary.get("left_photo_delta_a_per_cm"))
    right = safe_float(summary.get("right_photo_delta_a_per_cm"))
    if left is None or right is None:
        return None
    return left + right


def split_summary(path: Path | None, fallback_case: str) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    data = read_json(path)
    case = data.get("config", {}).get("generation_profile_case") or fallback_case
    total = split_total(data)
    return {
        "case": case,
        "summary_json": str(path),
        "left": safe_float(data.get("left_photo_delta_a_per_cm")),
        "right": safe_float(data.get("right_photo_delta_a_per_cm")),
        "total": total,
        "split": safe_float(data.get("photo_split_phase_x_proxy")),
        "terminal_balance": safe_float(data.get("terminal_current_balance_illuminated_a_per_cm")),
        "cra_x_deg": safe_float(data.get("illuminated", {}).get("cra_x_deg"))
        or safe_float(data.get("config", {}).get("cra_x_deg")),
    }


def gw_cases(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = read_json(path)
    return {str(item.get("case", "")): item for item in data.get("cases", [])}


def stage_marker_path(variant: dict[str, Any], stage: str) -> Path | None:
    planned_outputs = variant.get("planned_outputs", {})
    if stage == "devsim_electrical":
        case_dirs = [planned_outputs.get(output_key) for output_key in CASE_OUTPUT_KEYS.values()]
        if any(not case_dir for case_dir in case_dirs):
            return None
        summaries = [Path(str(case_dir)) / "summary.json" for case_dir in case_dirs]
        return summaries[0] if all(summary.exists() for summary in summaries) else None
    output_key, file_name = STAGE_MARKERS[stage]
    output_dir = planned_outputs.get(output_key)
    if not output_dir:
        return None
    path = Path(output_dir) / file_name
    return path if path.exists() else None


def variant_state(variant: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    if variant.get("id") == "baseline_reference":
        return "executed_reference", [], []
    required = list(variant.get("required_stages", []))
    completed = [stage for stage in required if stage_marker_path(variant, stage)]
    missing = [stage for stage in required if stage not in completed]
    if required and not missing:
        return "complete", completed, []
    if completed:
        return "partial", completed, missing
    return "planned_only", [], required


def ratio_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (value - baseline) / baseline


def sign_changed(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return (a < 0 < b) or (b < 0 < a)


def override_string(variant: dict[str, Any]) -> str:
    overrides = variant.get("parameter_overrides", {})
    return "; ".join(f"{key}={value}" for key, value in overrides.items())


def variant_summary_path(variant: dict[str, Any], case: str) -> Path | None:
    native_sweep_dir = variant.get("planned_outputs", {}).get("devsim_native_response_sweep")
    if native_sweep_dir:
        native_path = Path(native_sweep_dir) / "cases" / f"{case}_wl550nm" / "summary.json"
        if native_path.exists():
            return native_path
    output_key = CASE_OUTPUT_KEYS.get(case)
    if not output_key:
        return None
    output_dir = variant.get("planned_outputs", {}).get(output_key)
    if not output_dir:
        return None
    return Path(output_dir) / "summary.json"


def build_rows(
    project_config: Path,
    variant_manifest_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    project = read_json(project_config)
    config_dir = project_config.parent
    views = project.get("views", {})
    root_manifest = read_json(variant_manifest_path)

    baseline_by_case: dict[str, dict[str, Any]] = {}
    for case, value in project.get("native_split_runs", {}).items():
        summary = split_summary(abs_path(config_dir, value), case)
        if summary:
            baseline_by_case[case] = summary

    baseline_gw_path = abs_path(config_dir, views.get("gw_coupling_manifest"))
    baseline_gw = gw_cases(baseline_gw_path)

    rows: list[dict[str, str]] = []
    variant_summaries: list[dict[str, Any]] = []

    for variant in root_manifest.get("variants", []):
        state, completed_stages, missing_stages = variant_state(variant)
        gw_path = Path(variant.get("planned_outputs", {}).get("gw_coupling", "")) / "gw_coupling_manifest.json"
        variant_gw = baseline_gw if variant.get("id") == "baseline_reference" else gw_cases(gw_path)

        max_abs_rel_change = 0.0
        has_numeric_change = False
        sign_changed_cases: list[str] = []

        for case, baseline in baseline_by_case.items():
            variant_result = (
                baseline
                if variant.get("id") == "baseline_reference"
                else split_summary(variant_summary_path(variant, case), case)
            )
            gw_row = variant_gw.get(case, {})
            total_change = ratio_delta(
                variant_result.get("total") if variant_result else None,
                baseline.get("total"),
            )
            split_delta = None
            if variant_result and variant_result.get("split") is not None and baseline.get("split") is not None:
                split_delta = variant_result["split"] - baseline["split"]
            changed_sign = sign_changed(baseline.get("split"), variant_result.get("split") if variant_result else None)
            if changed_sign:
                sign_changed_cases.append(case)
            if total_change is not None:
                has_numeric_change = True
                max_abs_rel_change = max(max_abs_rel_change, abs(total_change))

            rows.append(
                {
                    "variant_id": str(variant.get("id", "")),
                    "variant_label": str(variant.get("label", variant.get("id", ""))),
                    "variant_state": state,
                    "case": case,
                    "cra_x_deg": fmt_float(
                        safe_float(gw_row.get("cra_x_deg"))
                        if gw_row
                        else baseline.get("cra_x_deg")
                    ),
                    "baseline_total_photo_delta_a_per_cm": fmt_float(baseline.get("total")),
                    "variant_total_photo_delta_a_per_cm": fmt_float(
                        variant_result.get("total") if variant_result else None
                    ),
                    "total_photo_delta_rel_change": fmt_float(total_change),
                    "baseline_split_phase_x_proxy": fmt_float(baseline.get("split")),
                    "variant_split_phase_x_proxy": fmt_float(
                        variant_result.get("split") if variant_result else None
                    ),
                    "split_phase_delta": fmt_float(split_delta),
                    "split_phase_sign_changed": "true" if changed_sign else "false",
                    "baseline_left_photo_delta_a_per_cm": fmt_float(baseline.get("left")),
                    "variant_left_photo_delta_a_per_cm": fmt_float(
                        variant_result.get("left") if variant_result else None
                    ),
                    "baseline_right_photo_delta_a_per_cm": fmt_float(baseline.get("right")),
                    "variant_right_photo_delta_a_per_cm": fmt_float(
                        variant_result.get("right") if variant_result else None
                    ),
                    "terminal_balance_illuminated_a_per_cm": fmt_float(
                        variant_result.get("terminal_balance") if variant_result else None
                    ),
                    "gw_total_reference_scaled_rel_error": fmt_float(
                        safe_float(gw_row.get("gw_total_reference_scaled_rel_error"))
                    ),
                    "gw_mesh_total_reference_scaled_rel_error": fmt_float(
                        safe_float(gw_row.get("gw_mesh_total_reference_scaled_rel_error"))
                    ),
                    "gw_devsim_laplace_total_reference_scaled_rel_error": fmt_float(
                        safe_float(gw_row.get("gw_devsim_laplace_total_reference_scaled_rel_error"))
                    ),
                    "gw_split_phase_error": fmt_float(safe_float(gw_row.get("gw_split_phase_error"))),
                    "gw_mesh_split_phase_error": fmt_float(safe_float(gw_row.get("gw_mesh_split_phase_error"))),
                    "gw_devsim_laplace_split_phase_error": fmt_float(
                        safe_float(gw_row.get("gw_devsim_laplace_split_phase_error"))
                    ),
                    "parameter_overrides": override_string(variant),
                    "completed_stages": ",".join(completed_stages),
                    "missing_stages": ",".join(missing_stages),
                    "product_lut_ready": "false",
                    "summary_json": variant_result.get("summary_json", "") if variant_result else "",
                }
            )

        variant_summaries.append(
            {
                "id": variant.get("id"),
                "label": variant.get("label", variant.get("id")),
                "state": state,
                "completed_stages": completed_stages,
                "missing_stages": missing_stages,
                "has_numeric_change": has_numeric_change,
                "max_abs_total_photo_delta_rel_change": max_abs_rel_change if has_numeric_change else None,
                "split_phase_sign_changed_cases": sorted(set(sign_changed_cases)),
                "product_lut_ready": False,
            }
        )

    return rows, variant_summaries, root_manifest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def percent_text(value: str) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return ""
    return f"{numeric * 100:.2f}%"


def html_table(rows: list[dict[str, str]]) -> str:
    cells = []
    for row in rows:
        state = html.escape(row["variant_state"])
        sign = "yes" if row["split_phase_sign_changed"] == "true" else ""
        cells.append(
            "<tr>"
            f"<td>{html.escape(row['variant_id'])}</td>"
            f"<td><span class=\"pill {state}\">{state}</span></td>"
            f"<td>{html.escape(row['case'])}</td>"
            f"<td>{html.escape(row['cra_x_deg'])}</td>"
            f"<td>{html.escape(row['baseline_total_photo_delta_a_per_cm'])}</td>"
            f"<td>{html.escape(row['variant_total_photo_delta_a_per_cm'])}</td>"
            f"<td>{html.escape(percent_text(row['total_photo_delta_rel_change']))}</td>"
            f"<td>{html.escape(row['baseline_split_phase_x_proxy'])}</td>"
            f"<td>{html.escape(row['variant_split_phase_x_proxy'])}</td>"
            f"<td>{html.escape(row['split_phase_delta'])}</td>"
            f"<td>{sign}</td>"
            f"<td>{html.escape(row['gw_total_reference_scaled_rel_error'])}</td>"
            f"<td>{html.escape(row['gw_mesh_total_reference_scaled_rel_error'])}</td>"
            f"<td>{html.escape(row['gw_devsim_laplace_total_reference_scaled_rel_error'])}</td>"
            f"<td>{html.escape(row['parameter_overrides'])}</td>"
            "</tr>"
        )
    return "\n".join(cells)


def write_html(path: Path, rows: list[dict[str, str]], comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = comparison["summary"]
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Sensor Variant Comparison</title>
<style>
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1f2933;background:#f4f6f8;font-size:13px}}
header{{padding:18px 22px;background:#111827;color:white}}
h1{{font-size:18px;margin:0 0 6px}}
p{{margin:0;color:#c9d3df}}
main{{padding:16px 22px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.metric{{background:white;border:1px solid #cfd7df;border-radius:8px;padding:10px}}
.label{{color:#61707f;font-size:12px;margin-bottom:5px}}
.value{{font-weight:700;font-size:18px}}
.note{{background:#fff8eb;border:1px solid #f5c27a;color:#7c3f00;border-radius:8px;padding:10px;margin:12px 0}}
.tableWrap{{background:white;border:1px solid #cfd7df;border-radius:8px;overflow:auto}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px}}
th,td{{border-bottom:1px solid #e5e9ee;padding:7px 8px;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th{{position:sticky;top:0;background:#f9fafb;z-index:1;color:#4b5563}}
.pill{{display:inline-flex;height:20px;align-items:center;padding:0 6px;border-radius:5px;border:1px solid #cfd7df;background:#fff}}
.complete,.executed_reference{{border-color:#b7dec7;color:#177245;background:#f0fbf4}}
.planned_only{{border-color:#f5c27a;color:#b45309;background:#fff8eb}}
.partial{{border-color:#f0a4a4;color:#b91c1c;background:#fff5f5}}
footer{{padding:12px 22px;color:#61707f}}
</style>
</head>
<body>
<header>
  <h1>Image Sensor Variant Comparison</h1>
  <p>Baseline-relative sweep table for completed and planned local design variants.</p>
</header>
<main>
  <section class="metrics">
    <div class="metric"><div class="label">Variants</div><div class="value">{summary['variant_count']}</div></div>
    <div class="metric"><div class="label">Completed Candidate Variants</div><div class="value">{summary['completed_candidate_count']}</div></div>
    <div class="metric"><div class="label">Comparison Rows</div><div class="value">{summary['row_count']}</div></div>
    <div class="metric"><div class="label">Product LUT Ready</div><div class="value">No</div></div>
  </section>
  <div class="note">This comparison is useful for trend exploration only. The current fixed-charge and weighting models are still proxy models and are not calibrated to measured sensor targets.</div>
  <section class="tableWrap">
    <table>
      <thead>
        <tr>
          <th>Variant</th><th>State</th><th>Case</th><th>CRA X</th>
          <th>Baseline Total</th><th>Variant Total</th><th>Total Change</th>
          <th>Baseline Split</th><th>Variant Split</th><th>Split Delta</th><th>Sign Flip</th>
          <th>G*W Rel Err</th><th>W_mesh Rel Err</th><th>W_devsim Rel Err</th><th>Overrides</th>
        </tr>
      </thead>
      <tbody>
        {html_table(rows)}
      </tbody>
    </table>
  </section>
</main>
<footer>
Generated by image_sensor_variant_compare.py. Product LUT readiness remains false until measured stack, measured n,k, calibrated transport, and convergence gates pass.
</footer>
</body>
</html>
""",
        encoding="utf-8",
    )


def update_root_manifest(
    path: Path,
    root_manifest: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    root_manifest["comparison_outputs"] = comparison["outputs"]
    root_manifest.setdefault("summary", {})
    root_manifest["summary"]["completed_candidate_count"] = comparison["summary"]["completed_candidate_count"]
    root_manifest["summary"]["variant_comparison_row_count"] = comparison["summary"]["row_count"]
    root_manifest["summary"]["product_lut_ready"] = False
    write_json(path, root_manifest)


def run(project_config: Path, variant_manifest: Path, output_dir: Path, update_manifest: bool = True) -> dict[str, Any]:
    rows, variant_summaries, root_manifest = build_rows(project_config, variant_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "variant_comparison.csv"
    json_path = output_dir / "variant_comparison.json"
    html_path = output_dir / "variant_comparison_report.html"
    completed_candidate_count = sum(
        1
        for item in variant_summaries
        if item["id"] != "baseline_reference" and item["state"] == "complete"
    )
    comparison = {
        "schema": "image_sensor_variant_comparison_v1",
        "source_project_config": str(project_config),
        "source_variant_manifest": str(variant_manifest),
        "summary": {
            "variant_count": len(variant_summaries),
            "completed_candidate_count": completed_candidate_count,
            "row_count": len(rows),
            "product_lut_ready": False,
            "accuracy_ready": False,
        },
        "variant_summaries": variant_summaries,
        "rows": rows,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "html": str(html_path),
        },
        "limitations": [
            "Comparison rows use completed local artifacts only; missing outputs remain planned or partial.",
            "W_proxy, W_mesh, and W_devsim_laplace are not calibrated drift-diffusion collection probabilities.",
            "Fixed charge is represented through proxy terms until interface-trap equations and measured calibration targets exist.",
            "Use for design trend exploration, not product-accuracy camera LUT delivery.",
        ],
    }
    write_csv(csv_path, rows)
    write_json(json_path, comparison)
    write_html(html_path, rows, comparison)
    if update_manifest:
        update_root_manifest(variant_manifest, root_manifest, comparison)
    print(json.dumps(comparison["summary"], indent=2, ensure_ascii=False))
    print(f"csv: {rel_from_root(csv_path)}")
    print(f"html: {rel_from_root(html_path)}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--variant-manifest", type=Path, default=DEFAULT_VARIANT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-update-manifest", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.variant_manifest.resolve(),
        args.output_dir.resolve(),
        update_manifest=not args.no_update_manifest,
    )


if __name__ == "__main__":
    main()
