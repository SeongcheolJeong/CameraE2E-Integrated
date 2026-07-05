#!/usr/bin/env python3
"""Mesh-convergence check for native DEVSIM split-PD response sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MeshLevel:
    label: str
    mesh_um: float
    fine_mesh_um: float


@dataclass(frozen=True)
class NativeConvergenceConfig:
    generation_map_npz: Path
    measured_profile: Path
    output_dir: Path
    levels: tuple[MeshLevel, ...]
    width_um: float = 1.4
    depth_um: float = 3.0
    split_gap_um: float = 0.04
    total_rel_tol: float = 0.05
    split_abs_tol: float = 0.05
    include_dti_oxide: bool = False
    floating_diffusion_feature_scale: float = 1.0
    transfer_gate_barrier_feature_scale: float = 1.0
    bdti_liner_feature_scale: float = 1.0
    resolved_bdti_sidewall_liner: bool = False
    force: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_levels(text: str) -> tuple[MeshLevel, ...]:
    levels: list[MeshLevel] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "mesh levels must use label:mesh_um:fine_mesh_um, separated by commas"
            )
        levels.append(MeshLevel(parts[0], float(parts[1]), float(parts[2])))
    if len(levels) < 2:
        raise argparse.ArgumentTypeError("at least two mesh levels are required")
    return tuple(levels)


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        raise RuntimeError(f"command failed; see {log_path}\n{tail}")


def build_mesh(config: NativeConvergenceConfig, level: MeshLevel) -> Path:
    mesh_dir = config.output_dir / f"mesh_{level.label}"
    mesh_path = mesh_dir / "split_pixel_2d.msh"
    if mesh_path.exists() and not config.force:
        return mesh_path
    command = [
        sys.executable,
        str(ROOT / "tcad_gmsh_pixel_mesh.py"),
        "--dimension",
        "2",
        "--measured-profile",
        str(config.measured_profile),
        "--width-um",
        f"{config.width_um:g}",
        "--depth-um",
        f"{config.depth_um:g}",
        "--split-gap-um",
        f"{config.split_gap_um:g}",
        "--mesh-um",
        f"{level.mesh_um:g}",
        "--fine-mesh-um",
        f"{level.fine_mesh_um:g}",
        "--output-dir",
        str(mesh_dir),
    ]
    if config.include_dti_oxide:
        command.append("--include-dti-oxide")
    run_command(command, ROOT, mesh_dir / "mesh.log")
    return mesh_path


def run_sweep(config: NativeConvergenceConfig, level: MeshLevel, mesh_path: Path) -> Path:
    sweep_dir = config.output_dir / f"sweep_{level.label}"
    manifest_path = sweep_dir / "native_response_sweep_manifest.json"
    if manifest_path.exists() and not config.force:
        return manifest_path
    command = [
        sys.executable,
        str(ROOT / "devsim_native_response_sweep_2d.py"),
        "--generation-map-npz",
        str(config.generation_map_npz),
        "--mesh",
        str(mesh_path),
        "--measured-profile",
        str(config.measured_profile),
        "--width-um",
        f"{config.width_um:g}",
        "--depth-um",
        f"{config.depth_um:g}",
        "--split-gap-um",
        f"{config.split_gap_um:g}",
        "--floating-diffusion-feature-scale",
        f"{config.floating_diffusion_feature_scale:g}",
        "--transfer-gate-barrier-feature-scale",
        f"{config.transfer_gate_barrier_feature_scale:g}",
        "--bdti-liner-feature-scale",
        f"{config.bdti_liner_feature_scale:g}",
        "--output-dir",
        str(sweep_dir),
    ]
    if config.resolved_bdti_sidewall_liner:
        command.append("--resolved-bdti-sidewall-liner")
    if config.force:
        command.append("--force")
    run_command(command, ROOT, sweep_dir / "sweep.log")
    return manifest_path


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["case"]), float(row["wavelength_nm"])


def total_response(row: dict[str, Any]) -> float:
    value = row.get("photo_total_abs_delta_a_per_cm")
    if value is not None:
        return float(value)
    return abs(float(row["left_photo_delta_a_per_cm"])) + abs(float(row["right_photo_delta_a_per_cm"]))


def compare_manifests(config: NativeConvergenceConfig, manifests: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_label = config.levels[-1].label
    reference = read_json(manifests[reference_label])
    ref_rows = {row_key(row): row for row in reference.get("rows", [])}
    comparison_rows: list[dict[str, Any]] = []
    max_total_rel_delta = 0.0
    max_split_abs_delta = 0.0
    missing: list[str] = []
    for level in config.levels:
        manifest = read_json(manifests[level.label])
        for row in manifest.get("rows", []):
            key = row_key(row)
            ref = ref_rows.get(key)
            if ref is None:
                missing.append(f"{level.label}:{key[0]}:{key[1]}")
                continue
            total = total_response(row)
            ref_total = total_response(ref)
            split = float(row.get("photo_split_phase_x_proxy", math.nan))
            ref_split = float(ref.get("photo_split_phase_x_proxy", math.nan))
            rel_delta = abs(total - ref_total) / abs(ref_total) if ref_total else math.nan
            split_delta = split - ref_split
            if math.isfinite(rel_delta):
                max_total_rel_delta = max(max_total_rel_delta, rel_delta)
            if math.isfinite(split_delta):
                max_split_abs_delta = max(max_split_abs_delta, abs(split_delta))
            comparison_rows.append(
                {
                    "level": level.label,
                    "mesh_um": level.mesh_um,
                    "fine_mesh_um": level.fine_mesh_um,
                    "reference_level": reference_label,
                    "case": key[0],
                    "wavelength_nm": key[1],
                    "total_response_a_per_cm": total,
                    "reference_total_response_a_per_cm": ref_total,
                    "total_rel_delta_to_reference": rel_delta,
                    "split_phase_x": split,
                    "reference_split_phase_x": ref_split,
                    "split_abs_delta_to_reference": abs(split_delta)
                    if math.isfinite(split_delta)
                    else math.nan,
                    "summary_json": row.get("summary_json", ""),
                }
            )
    metrics = {
        "reference_level": reference_label,
        "max_total_response_rel_delta_to_reference": max_total_rel_delta,
        "max_split_phase_abs_delta_to_reference": max_split_abs_delta,
        "missing_comparisons": missing,
    }
    return comparison_rows, metrics


def run(config: NativeConvergenceConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths: dict[str, Path] = {}
    manifests: dict[str, Path] = {}
    for level in config.levels:
        mesh_path = build_mesh(config, level)
        mesh_paths[level.label] = mesh_path
        manifests[level.label] = run_sweep(config, level, mesh_path)
    comparison_rows, metrics = compare_manifests(config, manifests)
    passed = (
        not metrics["missing_comparisons"]
        and metrics["max_total_response_rel_delta_to_reference"] <= config.total_rel_tol
        and metrics["max_split_phase_abs_delta_to_reference"] <= config.split_abs_tol
    )
    summary_csv = config.output_dir / "native_response_convergence_summary.csv"
    report_json = config.output_dir / "native_response_convergence_report.json"
    write_csv(summary_csv, comparison_rows)
    report = {
        "schema": "devsim_native_response_convergence_2d_v1",
        "passed": passed,
        "method": "mesh_refinement_native_devsim_direct_response",
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "measured_profile": str(config.measured_profile),
            "output_dir": str(config.output_dir),
            "levels": [asdict(level) for level in config.levels],
        },
        "total_rel_tol": config.total_rel_tol,
        "split_abs_tol": config.split_abs_tol,
        **metrics,
        "level_outputs": {
            level.label: {
                "mesh": str(mesh_paths[level.label]),
                "sweep_manifest": str(manifests[level.label]),
            }
            for level in config.levels
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
        },
        "rows": comparison_rows,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-map-npz",
        type=Path,
        default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
    )
    parser.add_argument(
        "--measured-profile",
        type=Path,
        default=ROOT / "measured_profiles/reference_cmos_ppd_1p4um/profile.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/devsim_native_response_convergence_2d_reference")
    parser.add_argument(
        "--levels",
        type=parse_levels,
        default=parse_levels("coarse:0.12:0.03,reference:0.10:0.025"),
    )
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--total-rel-tol", type=float, default=0.05)
    parser.add_argument("--split-abs-tol", type=float, default=0.05)
    parser.add_argument("--include-dti-oxide", action="store_true")
    parser.add_argument("--floating-diffusion-feature-scale", type=float, default=1.0)
    parser.add_argument("--transfer-gate-barrier-feature-scale", type=float, default=1.0)
    parser.add_argument("--bdti-liner-feature-scale", type=float, default=1.0)
    parser.add_argument("--resolved-bdti-sidewall-liner", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        NativeConvergenceConfig(
            generation_map_npz=args.generation_map_npz,
            measured_profile=args.measured_profile,
            output_dir=args.output_dir,
            levels=args.levels,
            width_um=args.width_um,
            depth_um=args.depth_um,
            split_gap_um=args.split_gap_um,
            total_rel_tol=args.total_rel_tol,
            split_abs_tol=args.split_abs_tol,
            include_dti_oxide=args.include_dti_oxide,
            floating_diffusion_feature_scale=args.floating_diffusion_feature_scale,
            transfer_gate_barrier_feature_scale=args.transfer_gate_barrier_feature_scale,
            bdti_liner_feature_scale=args.bdti_liner_feature_scale,
            resolved_bdti_sidewall_liner=args.resolved_bdti_sidewall_liner,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
