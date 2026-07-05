#!/usr/bin/env python3
"""Run DEVSIM TCAD smoke simulations for major image-sensor DB records.

The selected records come from image_sensor_db/tcad_structure_db readiness
ranking. Each run imports the generated measured_tcad_profile_v1 proxy profile
and sweeps three lateral optical-generation positions. This is a practical
solver smoke/trend suite, not a calibrated product TCAD deck.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "image_sensor_db_tcad_major_sim"
DEFAULT_TCAD_PYTHON = ROOT / ".tcad-env" / "bin" / "python"
DEFAULT_STRUCTURE_MANIFEST = ROOT / "image_sensor_db" / "tcad_structure_db" / "manifest.json"

DEFAULT_SLUGS = (
    "dep_2511_801_samsung_hp5",
    "dep_2506_801_samsung_jnp_s5kjnp",
    "dep_2504_801_sony_imx487_aamj_c",
    "dep_2505_802_smartsens_sc550xs",
    "dep_2412_801_sony_imx900",
    "dep_2511_802_stmicroelectronics_56g812a1a_cis",
    "dep_2411_801_omnivision_ox08d10",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.lower()).strip("_")


def compact(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if number != 0 and abs(number) < 0.001:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def manifest_by_slug(path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(path)
    output = {}
    for item in manifest.get("models", []):
        slug = Path(str(item.get("model_json", ""))).stem
        if slug:
            output[slug] = item
    return output


def classify_failure(text: str) -> str:
    if "Convergence failure" in text:
        return "DEVSIM drift-diffusion convergence failure"
    if "umfpack numeric failed" in text or "matrix is singular" in text:
        return "DEVSIM linear solver singular matrix"
    if "floating point exception" in text or "Overflow" in text:
        return "DEVSIM numerical overflow"
    if "TIMEOUT" in text:
        return "solver timeout"
    if "Traceback" in text:
        return "python/depsim traceback"
    return ""


def run_command(command: list[str], cwd: Path, log_path: Path, timeout_s: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False)
        elapsed = time.time() - started
        log_text = result.stdout + "\n" + result.stderr
        log_path.write_text(log_text, encoding="utf-8")
        if result.returncode == 0:
            status = "PASS"
        elif result.returncode < 0:
            status = "INTERRUPTED"
        else:
            status = "FAIL"
        return {
            "status": status,
            "returncode": result.returncode,
            "elapsed_s": elapsed,
            "log": repo_rel(log_path),
            "failure_reason": "" if status == "PASS" else classify_failure(log_text),
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.time() - started
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        log_text = stdout + "\n" + stderr + f"\nTIMEOUT after {timeout_s} s\n"
        log_path.write_text(log_text, encoding="utf-8")
        return {
            "status": "TIMEOUT",
            "returncode": None,
            "elapsed_s": elapsed,
            "log": repo_rel(log_path),
            "failure_reason": classify_failure(log_text),
        }


def sensor_paths(slug: str) -> dict[str, Path]:
    return {
        "profile": ROOT / "image_sensor_db" / "generated_tcad_profiles" / slug / "profile.json",
        "stack": ROOT / "image_sensor_db" / "generated_stack_configs" / f"{slug}.json",
        "structure": ROOT / "image_sensor_db" / "tcad_structure_db" / "models" / f"{slug}.json",
    }


def mesh_settings(geometry: dict[str, Any]) -> dict[str, float]:
    width = float(geometry["width_um"])
    depth = float(geometry["depth_um"])
    split_gap = float(geometry.get("split_gap_um", clamp(width * 0.04, 0.02, 0.12)))
    pinning_depth = float(geometry.get("pinning_depth_um", clamp(depth * 0.025, 0.05, 0.18)))
    return {
        "mesh_x_um": clamp(width / 12.0, 0.035, 0.18),
        "mesh_y_um": clamp(depth / 35.0, 0.08, 0.18),
        "junction_mesh_um": clamp(min(width / 24.0, pinning_depth / 3.0), 0.012, 0.05),
        "split_gap_um": split_gap,
        "photo_sigma_x_um": clamp(width * 0.16, 0.06, 0.35),
        "photo_sigma_y_um": clamp(depth * 0.045, 0.12, 0.30),
        "junction_um": clamp(pinning_depth + 0.24, 0.18, min(depth * 0.35, 0.8)),
    }


def unsupported_model_reason(profile: dict[str, Any]) -> str:
    source = profile.get("techinsights_source", {})
    specs = source.get("derived_specs", {})
    content = str(source.get("content_title", "")).lower()
    if bool(specs.get("has_lofic", False)) or "theiacel" in content:
        return (
            "unsupported by current split-PD profile-ppd deck: LOFIC/TheiaCel/HDR pixel "
            "needs a dedicated lateral overflow capacitor, transfer path, and readout node model"
        )
    return ""


def build_command(
    *,
    tcad_python: Path,
    output_dir: Path,
    profile_path: Path,
    geometry: dict[str, Any],
    settings: dict[str, float],
    shift_x_um: float,
    attempt: dict[str, Any],
) -> list[str]:
    command = [
        str(tcad_python),
        "devsim_split_pd_2d.py",
        "--output-dir",
        repo_rel(output_dir),
        "--width-um",
        f"{float(geometry['width_um']):g}",
        "--depth-um",
        f"{float(geometry['depth_um']):g}",
        "--junction-um",
        f"{settings['junction_um']:g}",
        "--mesh-x-um",
        f"{settings['mesh_x_um']:g}",
        "--mesh-y-um",
        f"{settings['mesh_y_um']:g}",
        "--junction-mesh-um",
        f"{settings['junction_mesh_um']:g}",
        "--split-gap-um",
        f"{settings['split_gap_um']:g}",
        "--photo-g0-cm3-s",
        f"{float(attempt['photo_g0_cm3_s']):g}",
        "--photo-shift-x-um",
        f"{shift_x_um:g}",
        "--photo-sigma-x-um",
        f"{settings['photo_sigma_x_um']:g}",
        "--photo-sigma-y-um",
        f"{settings['photo_sigma_y_um']:g}",
        "--electrical-model",
        "profile-ppd",
        "--measured-profile",
        repo_rel(profile_path),
        "--reverse-bias-v",
        f"{float(attempt['reverse_bias_v']):g}",
        "--dd-relative-error",
        f"{float(attempt['dd_relative_error']):g}",
        "--dd-max-iterations",
        str(int(attempt["dd_max_iterations"])),
        "--fixed-charge-scale",
        f"{float(attempt['fixed_charge_scale']):g}",
        "--interface-trap-density-scale",
        f"{float(attempt['interface_trap_density_scale']):g}",
        "--interface-trap-recombination-scale",
        f"{float(attempt['interface_trap_recombination_scale']):g}",
        "--floating-diffusion-feature-scale",
        f"{float(attempt['floating_diffusion_feature_scale']):g}",
        "--transfer-gate-barrier-feature-scale",
        f"{float(attempt['transfer_gate_barrier_feature_scale']):g}",
        "--bdti-liner-feature-scale",
        f"{float(attempt['bdti_liner_feature_scale']):g}",
    ]
    if attempt.get("transport_override", "profile") != "profile":
        command.extend(["--transport-override", str(attempt["transport_override"])])
    if attempt.get("disable_field_mobility", False):
        command.append("--disable-field-mobility")
    return command


def solver_attempts(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = {
        "photo_g0_cm3_s": args.photo_g0_cm3_s,
        "reverse_bias_v": args.reverse_bias_v,
        "fixed_charge_scale": 1.0,
        "interface_trap_density_scale": 1.0,
        "interface_trap_recombination_scale": 1.0,
        "floating_diffusion_feature_scale": 1.0,
        "transfer_gate_barrier_feature_scale": 1.0,
        "bdti_liner_feature_scale": 1.0,
        "transport_override": "profile",
        "disable_field_mobility": False,
    }
    return [
        {
            **base,
            "attempt_id": "nominal",
            "solver_gate_on_pass": "PASS",
            "dd_relative_error": args.dd_relative_error,
            "dd_max_iterations": args.dd_max_iterations,
            "note": "strict nominal drift-diffusion solve",
        },
        {
            **base,
            "attempt_id": "relaxed_dd",
            "solver_gate_on_pass": "CHECK",
            "dd_relative_error": args.relaxed_dd_relative_error,
            "dd_max_iterations": args.relaxed_dd_max_iterations,
            "note": "relaxed numerical tolerance for proxy profile smoke/trend only",
        },
        {
            **base,
            "attempt_id": "loose_dd",
            "solver_gate_on_pass": "CHECK",
            "dd_relative_error": args.loose_dd_relative_error,
            "dd_max_iterations": args.loose_dd_max_iterations,
            "note": "loose drift-diffusion tolerance for difficult proxy/deep-Si profiles; trend only",
        },
        {
            **base,
            "attempt_id": "no_field_mobility_no_traps",
            "solver_gate_on_pass": "CHECK",
            "dd_relative_error": args.loose_dd_relative_error,
            "dd_max_iterations": args.loose_dd_max_iterations,
            "interface_trap_density_scale": 0.0,
            "interface_trap_recombination_scale": 0.0,
            "disable_field_mobility": True,
            "note": "diagnostic fallback: profile doping with field mobility disabled and interface traps removed",
        },
        {
            **base,
            "attempt_id": "constant_transport_no_traps",
            "solver_gate_on_pass": "CHECK",
            "dd_relative_error": args.loose_dd_relative_error,
            "dd_max_iterations": args.loose_dd_max_iterations,
            "interface_trap_density_scale": 0.0,
            "interface_trap_recombination_scale": 0.0,
            "transport_override": "constant-reference",
            "note": "diagnostic fallback: profile doping with constant transport and interface traps removed",
        },
    ]


def summary_row(
    *,
    slug: str,
    manifest_item: dict[str, Any],
    profile_path: Path,
    stack_path: Path,
    structure_path: Path,
    shift_label: str,
    shift_x_um: float,
    run_result: dict[str, Any],
    attempt: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = read_json(summary_path)
    profile = read_json(profile_path)
    profile_geometry = profile.get("geometry", {})
    config = summary.get("config", {})
    if not config:
        config = {
            "width_um": profile_geometry.get("width_um"),
            "depth_um": profile_geometry.get("depth_um"),
        }
    doping = summary.get("doping_summary", {})
    transport = summary.get("transport_summary", {})
    calibration = (
        doping.get("feature_summary", {}).get("calibration_status", {})
        or profile.get("calibration_status", {})
    )
    run_status = run_result["status"]
    solver_gate = "FAIL"
    if run_status == "PASS":
        solver_gate = str(attempt.get("solver_gate_on_pass", "CHECK"))
    elif run_status == "UNSUPPORTED":
        solver_gate = "UNSUPPORTED"
    return {
        "slug": slug,
        "code": manifest_item.get("code", ""),
        "manufacturer": manifest_item.get("manufacturer", ""),
        "device_name": manifest_item.get("device_name", ""),
        "readiness_level": manifest_item.get("readiness_level", ""),
        "readiness_score": manifest_item.get("readiness_score", ""),
        "shift_label": shift_label,
        "photo_shift_x_um": shift_x_um,
        "run_status": run_status,
        "solver_gate": solver_gate,
        "attempt_id": attempt.get("attempt_id", ""),
        "attempt_note": attempt.get("note", ""),
        "failure_reason": run_result.get("failure_reason", ""),
        "dd_relative_error": attempt.get("dd_relative_error"),
        "dd_max_iterations": attempt.get("dd_max_iterations"),
        "reverse_bias_v": attempt.get("reverse_bias_v"),
        "photo_g0_cm3_s": attempt.get("photo_g0_cm3_s"),
        "transport_override": attempt.get("transport_override", "profile"),
        "disable_field_mobility": attempt.get("disable_field_mobility", False),
        "interface_trap_density_scale": attempt.get("interface_trap_density_scale"),
        "interface_trap_recombination_scale": attempt.get("interface_trap_recombination_scale"),
        "elapsed_s": run_result["elapsed_s"],
        "width_um": config.get("width_um"),
        "depth_um": config.get("depth_um"),
        "node_count": summary.get("node_count"),
        "generation_source": summary.get("generation_source", ""),
        "electrical_model": summary.get("electrical_model", ""),
        "calibration_mode": calibration.get("mode", ""),
        "geometry_measured": calibration.get("geometry_measured", False),
        "donor_max_cm3": doping.get("donor_max_cm3"),
        "acceptor_max_cm3": doping.get("acceptor_max_cm3"),
        "net_min_cm3": doping.get("net_min_cm3"),
        "net_max_cm3": doping.get("net_max_cm3"),
        "electron_mobility_min_cm2_v_s": transport.get("electron_mobility_effective_edge_min_cm2_v_s"),
        "electron_mobility_max_cm2_v_s": transport.get("electron_mobility_effective_edge_max_cm2_v_s"),
        "left_photo_delta_e_a_per_cm": summary.get("left_photo_delta_electron_a_per_cm"),
        "right_photo_delta_e_a_per_cm": summary.get("right_photo_delta_electron_a_per_cm"),
        "total_photo_delta_e_a_per_cm": (
            (summary.get("left_photo_delta_electron_a_per_cm") or 0.0)
            + (summary.get("right_photo_delta_electron_a_per_cm") or 0.0)
        )
        if summary
        else None,
        "photo_split_phase_x_proxy": summary.get("photo_split_phase_x_proxy"),
        "terminal_balance_illum_a_per_cm": summary.get("terminal_current_balance_illuminated_a_per_cm"),
        "summary_json": repo_rel(summary_path) if summary_path.exists() else "",
        "profile_json": repo_rel(profile_path),
        "stack_json": repo_rel(stack_path),
        "structure_json": repo_rel(structure_path),
        "log": run_result.get("log", ""),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sensor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sensor.setdefault(str(row["slug"]), []).append(row)
    output = []
    for slug, items in by_sensor.items():
        center = next((row for row in items if row["shift_label"] == "center"), items[0])
        left = next((row for row in items if row["shift_label"] == "left"), None)
        right = next((row for row in items if row["shift_label"] == "right"), None)
        def value(row: dict[str, Any] | None, key: str) -> float | None:
            if row is None:
                return None
            try:
                number = float(row.get(key))
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None
        phase_left = value(left, "photo_split_phase_x_proxy")
        phase_right = value(right, "photo_split_phase_x_proxy")
        slope = None
        if phase_left is not None and phase_right is not None:
            dx = float(right["photo_shift_x_um"]) - float(left["photo_shift_x_um"])
            slope = (phase_right - phase_left) / dx if dx else None
        output.append(
            {
                "slug": slug,
                "code": center.get("code"),
                "manufacturer": center.get("manufacturer"),
                "device_name": center.get("device_name"),
                "readiness_level": center.get("readiness_level"),
                "status_all": "PASS"
                if all(row.get("solver_gate") == "PASS" for row in items)
                else "FAIL"
                if any(row.get("solver_gate") == "FAIL" for row in items)
                else "UNSUPPORTED"
                if any(row.get("solver_gate") == "UNSUPPORTED" for row in items)
                else "CHECK",
                "width_um": center.get("width_um"),
                "depth_um": center.get("depth_um"),
                "center_total_photo_delta_e_a_per_cm": center.get("total_photo_delta_e_a_per_cm"),
                "center_phase_x": center.get("photo_split_phase_x_proxy"),
                "phase_x_left_shift": phase_left,
                "phase_x_right_shift": phase_right,
                "phase_slope_per_um": slope,
                "node_count_center": center.get("node_count"),
                "calibration_mode": center.get("calibration_mode"),
                "geometry_measured": center.get("geometry_measured"),
                "center_solver_gate": center.get("solver_gate"),
                "center_attempt": center.get("attempt_id"),
                "center_failure_reason": center.get("failure_reason"),
            }
        )
    return output


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(compact(row.get(column)))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, report: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Sensor DB TCAD Major Simulation</title>
  <style>
    body {{ margin:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#071017; color:#e8f6fa; }}
    h1,h2 {{ margin:0 0 10px; }}
    p,li {{ color:#a9c2cc; line-height:1.55; }}
    section {{ margin-top:18px; border:1px solid #24495a; border-radius:12px; padding:16px; background:#0d1b24; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border-bottom:1px solid #24495a; padding:7px; text-align:left; vertical-align:top; }}
    th {{ color:#58d9ee; }}
    code {{ color:#f7db70; }}
  </style>
</head>
<body>
  <h1>Image Sensor DB TCAD Major Simulation</h1>
  <p>Generated from <code>image_sensor_db</code> TCAD proxy profiles. Each sensor imports <code>measured_tcad_profile_v1</code> into DEVSIM <code>profile-ppd</code> and runs left/center/right lateral optical-generation positions.</p>
  <section>
    <h2>Run Scope</h2>
    <ul>
      <li>Electrical model: DEVSIM drift-diffusion <code>profile-ppd</code>.</li>
      <li>Generation source: analytic 2D Gaussian, not full Meep G(x,depth).</li>
      <li>Gate meaning: <code>PASS</code> = nominal strict solve, <code>CHECK</code> = relaxed numerical smoke/trend solve, <code>UNSUPPORTED</code> = sensor architecture is outside this 2D split-PD deck, <code>FAIL</code> = no stable DD result in this suite.</li>
      <li>Accuracy status: setup/trend only. These profiles are TechInsights metadata/SIMS-seed proxies, not calibrated process decks.</li>
    </ul>
  </section>
  <section>
    <h2>Sensor Summary</h2>
    {table_html(report["sensor_summary_rows"])}
  </section>
  <section>
    <h2>Run Details</h2>
    {table_html(report["run_rows"])}
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_by_slug(args.structure_manifest)
    slugs = [item.strip() for item in args.slugs.split(",") if item.strip()]
    attempts = solver_attempts(args)
    rows = []
    for slug in slugs:
        paths = sensor_paths(slug)
        for label, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{slug}: missing {label} path {path}")
        profile = read_json(paths["profile"])
        geometry = profile["geometry"]
        settings = mesh_settings(geometry)
        model_gate_reason = unsupported_model_reason(profile)
        width = float(geometry["width_um"])
        shifts = [
            ("left", -0.25 * width),
            ("center", 0.0),
            ("right", 0.25 * width),
        ]
        for shift_label, shift_x in shifts:
            run_dir = output_dir / slug / shift_label
            run_result: dict[str, Any] | None = None
            used_attempt: dict[str, Any] | None = None
            if args.reuse_existing and (run_dir / "summary.json").exists():
                used_attempt = attempts[0]
                run_result = {
                    "status": "PASS",
                    "returncode": 0,
                    "elapsed_s": 0.0,
                    "log": repo_rel(run_dir / "run_nominal.log"),
                    "failure_reason": "",
                }
            elif model_gate_reason and not args.force_unsupported:
                used_attempt = {
                    **attempts[0],
                    "attempt_id": "unsupported_model_gate",
                    "solver_gate_on_pass": "UNSUPPORTED",
                    "note": model_gate_reason,
                }
                run_result = {
                    "status": "UNSUPPORTED",
                    "returncode": None,
                    "elapsed_s": 0.0,
                    "log": "",
                    "failure_reason": model_gate_reason,
                }
            else:
                for attempt in attempts:
                    command = build_command(
                        tcad_python=args.tcad_python,
                        output_dir=run_dir,
                        profile_path=paths["profile"],
                        geometry=geometry,
                        settings=settings,
                        shift_x_um=shift_x,
                        attempt=attempt,
                    )
                    log_path = run_dir / f"run_{attempt['attempt_id']}.log"
                    run_result = run_command(command, ROOT, log_path, args.timeout_s)
                    used_attempt = attempt
                    if run_result["status"] == "PASS":
                        break
            if run_result is None or used_attempt is None:
                raise RuntimeError(f"{slug}/{shift_label}: no solver attempt was executed")
            rows.append(
                summary_row(
                    slug=slug,
                    manifest_item=manifest.get(slug, {}),
                    profile_path=paths["profile"],
                    stack_path=paths["stack"],
                    structure_path=paths["structure"],
                    shift_label=shift_label,
                    shift_x_um=shift_x,
                    run_result=run_result,
                    attempt=used_attempt,
                    output_dir=run_dir,
                )
            )
    sensor_summary_rows = aggregate_rows(rows)
    report = {
        "schema": "image_sensor_db_tcad_major_sim_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "slugs": slugs,
            "output_dir": repo_rel(output_dir),
            "tcad_python": repo_rel(args.tcad_python),
            "reverse_bias_v": args.reverse_bias_v,
            "photo_g0_cm3_s": args.photo_g0_cm3_s,
            "dd_relative_error": args.dd_relative_error,
            "dd_max_iterations": args.dd_max_iterations,
            "relaxed_dd_relative_error": args.relaxed_dd_relative_error,
            "relaxed_dd_max_iterations": args.relaxed_dd_max_iterations,
            "loose_dd_relative_error": args.loose_dd_relative_error,
            "loose_dd_max_iterations": args.loose_dd_max_iterations,
            "attempts": attempts,
        },
        "run_status_counts": count_by_key(rows, "run_status"),
        "solver_gate_counts": count_by_key(rows, "solver_gate"),
        "sensor_summary_rows": sensor_summary_rows,
        "run_rows": rows,
        "limitations": [
            "The imported profiles are generated TechInsights metadata/SIMS-seed proxies, not calibrated process decks.",
            "Generation is analytic 2D Gaussian in this suite. Use Meep tcad_generation_map_2d.npz for optical-to-electrical production studies.",
            "The model is a 2D lateral split-PD drift-diffusion smoke/trend simulation; it is not a full 3D pixel circuit model.",
            "LOFIC/TheiaCel/HDR pixels are model-gated as unsupported until a dedicated overflow capacitor, transfer path, and readout-node deck is implemented.",
        ],
    }
    write_json(output_dir / "tcad_major_sim_report.json", report)
    write_csv(output_dir / "tcad_major_sim_runs.csv", rows)
    write_csv(output_dir / "tcad_major_sensor_summary.csv", sensor_summary_rows)
    write_html(output_dir / "index.html", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tcad-python", type=Path, default=DEFAULT_TCAD_PYTHON)
    parser.add_argument("--structure-manifest", type=Path, default=DEFAULT_STRUCTURE_MANIFEST)
    parser.add_argument("--slugs", default=",".join(DEFAULT_SLUGS))
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--force-unsupported",
        action="store_true",
        help="Run profiles that are normally model-gated as unsupported by this 2D split-PD deck.",
    )
    parser.add_argument("--photo-g0-cm3-s", type=float, default=1.0e20)
    parser.add_argument("--reverse-bias-v", type=float, default=-1.0)
    parser.add_argument("--dd-relative-error", type=float, default=1.0e-9)
    parser.add_argument("--dd-max-iterations", type=int, default=100)
    parser.add_argument("--relaxed-dd-relative-error", type=float, default=1.0e-6)
    parser.add_argument("--relaxed-dd-max-iterations", type=int, default=180)
    parser.add_argument("--loose-dd-relative-error", type=float, default=1.0e-4)
    parser.add_argument("--loose-dd-max-iterations", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "sensor_count": len(report["sensor_summary_rows"]),
                "run_count": len(report["run_rows"]),
                "solver_gate_counts": report["solver_gate_counts"],
                "output_dir": report["settings"]["output_dir"],
                "html": str(Path(report["settings"]["output_dir"]) / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
