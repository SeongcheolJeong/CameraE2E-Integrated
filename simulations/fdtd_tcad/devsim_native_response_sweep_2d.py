#!/usr/bin/env python3
"""Run native DEVSIM split-PD responses for every FDTD generation-map case.

This is the accuracy-oriented open-source path for camera-system response
tables: import each FDTD G(x,depth) map into DEVSIM and measure cathode
electron-current deltas directly.  G*W and DD-probe reductions can then be
checked against these native responses instead of being treated as primary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class NativeSweepConfig:
    generation_map_npz: Path
    mesh: Path
    measured_profile: Path
    output_dir: Path
    width_um: float = 1.4
    depth_um: float = 3.0
    split_gap_um: float = 0.04
    cases: tuple[str, ...] = ()
    wavelengths_nm: tuple[float, ...] = ()
    reverse_bias_v: float = -1.0
    dd_max_iterations: int = 160
    floating_diffusion_feature_scale: float = 1.0
    transfer_gate_barrier_feature_scale: float = 1.0
    bdti_liner_feature_scale: float = 1.0
    resolved_bdti_sidewall_liner: bool = False
    force: bool = False
    dry_run: bool = False


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


def scalar(value: Any, index: int | None = None) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if index is None:
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return arr.tolist()
    if arr.ndim == 0:
        return arr.item()
    if arr.shape[0] == 1:
        return arr.reshape(-1)[0].item()
    return arr[index].item() if hasattr(arr[index], "item") else arr[index]


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "case"


def wavelength_label(wavelength_nm: float) -> str:
    if abs(wavelength_nm - round(wavelength_nm)) < 1.0e-9:
        return f"{int(round(wavelength_nm))}nm"
    return f"{wavelength_nm:.3f}nm".replace(".", "p")


def parse_csv_list(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_float_csv_list(text: str) -> tuple[float, ...]:
    if not text:
        return ()
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def generation_entries(config: NativeSweepConfig) -> list[dict[str, Any]]:
    data = np.load(config.generation_map_npz, allow_pickle=True)
    cases = np.asarray(data["case"]).astype(str)
    wavelengths = np.asarray(data["wavelength_nm"], dtype=float)
    selected_cases = set(config.cases)
    selected_wavelengths = tuple(config.wavelengths_nm)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for index, case in enumerate(cases):
        wavelength_nm = float(wavelengths[index])
        if selected_cases and case not in selected_cases:
            continue
        if selected_wavelengths and not any(
            math.isclose(wavelength_nm, target, rel_tol=0.0, abs_tol=1.0e-9)
            for target in selected_wavelengths
        ):
            continue
        key = (case, wavelength_nm)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "case": case,
                "wavelength_nm": wavelength_nm,
                "cra_x_deg": float(scalar(data["cra_x_deg"], index)) if "cra_x_deg" in data else math.nan,
                "cra_z_deg": float(scalar(data["cra_z_deg"], index)) if "cra_z_deg" in data else math.nan,
                "field_x_norm": float(scalar(data["field_x_norm"], index))
                if "field_x_norm" in data
                else math.nan,
                "field_z_norm": float(scalar(data["field_z_norm"], index))
                if "field_z_norm" in data
                else math.nan,
                "color_channel": str(scalar(data["color_channel"], None))
                if "color_channel" in data
                else "",
                "incident_photon_flux_cm2_s": float(scalar(data["incident_photon_flux_cm2_s"], None))
                if "incident_photon_flux_cm2_s" in data
                else math.nan,
            }
        )
    if not entries:
        raise RuntimeError(
            f"no generation entries selected from {config.generation_map_npz}; "
            f"cases={config.cases or 'all'} wavelengths={config.wavelengths_nm or 'all'}"
        )
    return entries


def case_output_dir(config: NativeSweepConfig, entry: dict[str, Any]) -> Path:
    return (
        config.output_dir
        / "cases"
        / f"{safe_name(str(entry['case']))}_wl{wavelength_label(float(entry['wavelength_nm']))}"
    )


def run_case(config: NativeSweepConfig, entry: dict[str, Any], index: int) -> dict[str, Any]:
    run_dir = case_output_dir(config, entry)
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "run.log"
    command = [
        sys.executable,
        str(ROOT / "devsim_split_pd_2d.py"),
        "--mesh-source",
        "gmsh",
        "--gmsh-mesh",
        str(config.mesh),
        "--width-um",
        f"{config.width_um:g}",
        "--depth-um",
        f"{config.depth_um:g}",
        "--split-gap-um",
        f"{config.split_gap_um:g}",
        "--generation-map-npz",
        str(config.generation_map_npz),
        "--generation-profile-case",
        str(entry["case"]),
        "--generation-profile-wavelength-nm",
        f"{float(entry['wavelength_nm']):g}",
        "--electrical-model",
        "profile-ppd",
        "--measured-profile",
        str(config.measured_profile),
        "--reverse-bias-v",
        f"{config.reverse_bias_v:g}",
        "--dd-max-iterations",
        str(config.dd_max_iterations),
        "--floating-diffusion-feature-scale",
        f"{config.floating_diffusion_feature_scale:g}",
        "--transfer-gate-barrier-feature-scale",
        f"{config.transfer_gate_barrier_feature_scale:g}",
        "--bdti-liner-feature-scale",
        f"{config.bdti_liner_feature_scale:g}",
        "--output-dir",
        str(run_dir),
    ]
    if config.resolved_bdti_sidewall_liner:
        command.append("--resolved-bdti-sidewall-liner")
    status = "planned"
    if summary_path.exists() and not config.force:
        status = "reused"
    elif config.dry_run:
        status = "dry_run"
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(command) + "\n\n")
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
            raise RuntimeError(f"native DEVSIM case failed for {entry['case']} {entry['wavelength_nm']} nm\n{tail}")
        status = "executed"
    row = {
        "index": index,
        "case": entry["case"],
        "wavelength_nm": entry["wavelength_nm"],
        "cra_x_deg": entry["cra_x_deg"],
        "cra_z_deg": entry["cra_z_deg"],
        "field_x_norm": entry["field_x_norm"],
        "field_z_norm": entry["field_z_norm"],
        "color_channel": entry["color_channel"],
        "incident_photon_flux_cm2_s": entry["incident_photon_flux_cm2_s"],
        "status": status,
        "output_dir": str(run_dir),
        "summary_json": str(summary_path),
        "log": str(log_path),
        "command": " ".join(command),
    }
    if summary_path.exists():
        summary = read_json(summary_path)
        left_delta = float(summary.get("left_photo_delta_a_per_cm", math.nan))
        right_delta = float(summary.get("right_photo_delta_a_per_cm", math.nan))
        total_delta = (
            abs(left_delta) + abs(right_delta)
            if math.isfinite(left_delta) and math.isfinite(right_delta)
            else None
        )
        row.update(
            {
                "generated_current_a_per_cm": summary.get("generated_current_a_per_cm"),
                "left_photo_delta_a_per_cm": left_delta,
                "right_photo_delta_a_per_cm": right_delta,
                "photo_total_abs_delta_a_per_cm": summary.get("photo_total_abs_delta_a_per_cm", total_delta),
                "photo_split_phase_x_proxy": summary.get("photo_split_phase_x_proxy"),
                "terminal_balance_illuminated_a_per_cm": summary.get(
                    "terminal_current_balance_illuminated_a_per_cm"
                ),
                "mesh_node_count": summary.get("node_count"),
            }
        )
    return row


def run(config: NativeSweepConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    entries = generation_entries(config)
    rows = [run_case(config, entry, index) for index, entry in enumerate(entries)]
    completed = [row for row in rows if Path(str(row["summary_json"])).exists()]
    split_summaries = [row["summary_json"] for row in completed]
    summary_csv = config.output_dir / "native_response_sweep_summary.csv"
    split_summary_json = config.output_dir / "split_summaries.json"
    split_summary_txt = config.output_dir / "split_summaries.txt"
    manifest_path = config.output_dir / "native_response_sweep_manifest.json"
    write_csv(summary_csv, rows)
    split_summary_json.write_text(json.dumps(split_summaries, indent=2), encoding="utf-8")
    split_summary_txt.write_text("\n".join(split_summaries) + ("\n" if split_summaries else ""), encoding="utf-8")
    manifest = {
        "schema": "devsim_native_response_sweep_2d_v1",
        "method": "direct_fdtd_generation_map_to_devsim_split_pd",
        "config": {
            **asdict(config),
            "generation_map_npz": str(config.generation_map_npz),
            "mesh": str(config.mesh),
            "measured_profile": str(config.measured_profile),
            "output_dir": str(config.output_dir),
        },
        "entry_count": len(entries),
        "completed_count": len(completed),
        "case_count": len({row["case"] for row in rows}),
        "wavelength_count": len({float(row["wavelength_nm"]) for row in rows}),
        "split_summaries": split_summaries,
        "outputs": {
            "summary_csv": str(summary_csv),
            "split_summary_json": str(split_summary_json),
            "split_summary_txt": str(split_summary_txt),
            "manifest_json": str(manifest_path),
        },
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-map-npz",
        type=Path,
        default=ROOT / "runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=ROOT / "runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh",
    )
    parser.add_argument(
        "--measured-profile",
        type=Path,
        default=ROOT / "measured_profiles/reference_cmos_ppd_1p4um/profile.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/devsim_native_response_sweep_2d_reference")
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--cases", default="", help="Comma-separated case names. Default: all cases in NPZ.")
    parser.add_argument(
        "--wavelengths-nm",
        default="",
        help="Comma-separated wavelength values in nm. Default: all wavelengths in NPZ.",
    )
    parser.add_argument("--reverse-bias-v", type=float, default=-1.0)
    parser.add_argument("--dd-max-iterations", type=int, default=160)
    parser.add_argument("--floating-diffusion-feature-scale", type=float, default=1.0)
    parser.add_argument("--transfer-gate-barrier-feature-scale", type=float, default=1.0)
    parser.add_argument("--bdti-liner-feature-scale", type=float, default=1.0)
    parser.add_argument("--resolved-bdti-sidewall-liner", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(
        NativeSweepConfig(
            generation_map_npz=args.generation_map_npz,
            mesh=args.mesh,
            measured_profile=args.measured_profile,
            output_dir=args.output_dir,
            width_um=args.width_um,
            depth_um=args.depth_um,
            split_gap_um=args.split_gap_um,
            cases=parse_csv_list(args.cases),
            wavelengths_nm=parse_float_csv_list(args.wavelengths_nm),
            reverse_bias_v=args.reverse_bias_v,
            dd_max_iterations=args.dd_max_iterations,
            floating_diffusion_feature_scale=args.floating_diffusion_feature_scale,
            transfer_gate_barrier_feature_scale=args.transfer_gate_barrier_feature_scale,
            bdti_liner_feature_scale=args.bdti_liner_feature_scale,
            resolved_bdti_sidewall_liner=args.resolved_bdti_sidewall_liner,
            force=args.force,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
