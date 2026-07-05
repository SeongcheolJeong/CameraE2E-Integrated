#!/usr/bin/env python3
"""Build a sparse drift-diffusion local-generation response map.

Each probe is a real DEVSIM drift-diffusion solve with a narrow Gaussian optical
generation spot.  The resulting cathode electron-current deltas are normalized
by the generated current to produce a sparse collection-response table.  This is
not an adjoint solve and is not calibrated, but it does include the configured
doping, transport, SRH/trap, and bias equations.
"""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tcad_gw_coupling import Q_E, coordinate_key, parse_gmsh_msh22, triangle_area_cm2


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DDProbeConfig:
    mesh: Path
    measured_profile: Path
    output_dir: Path
    width_um: float = 1.4
    depth_um: float = 3.0
    split_gap_um: float = 0.04
    x_count: int = 5
    depth_count: int = 5
    x_min_um: float | None = None
    x_max_um: float | None = None
    depth_min_um: float | None = None
    depth_max_um: float | None = None
    photo_sigma_x_um: float = 0.07
    photo_sigma_y_um: float = 0.10
    photo_g0_cm3_s: float = 1.0e20
    baseline_generation_map_npz: Path | None = None
    baseline_cases: tuple[str, ...] = ()
    baseline_wavelength_nm: float = 550.0
    baseline_summary: tuple[Path, ...] = ()
    reverse_bias_v: float = -1.0
    dd_max_iterations: int = 160
    force: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_label(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def profile_geometry(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data.get("geometry", {})


def probe_grid(config: DDProbeConfig) -> list[tuple[float, float]]:
    geometry = profile_geometry(config.measured_profile)
    width_um = float(geometry.get("width_um", config.width_um))
    depth_um = float(geometry.get("depth_um", config.depth_um))
    dti_width_um = float(geometry.get("dti_width_um", 0.06))
    pinning_depth_um = float(geometry.get("pinning_depth_um", 0.08))
    x_min = config.x_min_um
    if x_min is None:
        x_min = -0.5 * width_um + dti_width_um + 0.08
    x_max = config.x_max_um
    if x_max is None:
        x_max = 0.5 * width_um - dti_width_um - 0.08
    depth_min = config.depth_min_um
    if depth_min is None:
        depth_min = pinning_depth_um + 0.12
    depth_max = config.depth_max_um
    if depth_max is None:
        depth_max = depth_um - 0.25
    xs = np.linspace(float(x_min), float(x_max), max(1, config.x_count))
    depths = np.linspace(float(depth_min), float(depth_max), max(1, config.depth_count))
    return [(float(x), float(depth)) for depth in depths for x in xs]


def baseline_summary_by_case(config: DDProbeConfig) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in config.baseline_summary:
        if not path.exists():
            raise FileNotFoundError(f"baseline summary not found: {path}")
        data = read_json(path)
        case = data.get("config", {}).get("generation_profile_case") or path.parent.name
        result[str(case)] = data
    return result


def run_probe(config: DDProbeConfig, x_um: float, depth_um: float, index: int, case: str = "") -> Path:
    case_part = f"{case}_" if case else ""
    probe_id = f"{case_part}probe_{index:03d}_x{safe_label(x_um)}_d{safe_label(depth_um)}"
    probe_dir = config.output_dir / "probes" / probe_id
    summary_path = probe_dir / "summary.json"
    if summary_path.exists() and not config.force:
        return summary_path
    probe_dir.mkdir(parents=True, exist_ok=True)
    log_path = probe_dir / "run.log"
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
        "--junction-um",
        f"{depth_um:g}",
        "--photo-shift-x-um",
        f"{x_um:g}",
        "--photo-sigma-x-um",
        f"{config.photo_sigma_x_um:g}",
        "--photo-sigma-y-um",
        f"{config.photo_sigma_y_um:g}",
        "--photo-g0-cm3-s",
        f"{config.photo_g0_cm3_s:g}",
        "--electrical-model",
        "profile-ppd",
        "--measured-profile",
        str(config.measured_profile),
        "--reverse-bias-v",
        f"{config.reverse_bias_v:g}",
        "--dd-max-iterations",
        str(config.dd_max_iterations),
        "--output-dir",
        str(probe_dir),
    ]
    if config.baseline_generation_map_npz:
        command.extend(
            [
                "--generation-map-npz",
                str(config.baseline_generation_map_npz),
                "--generation-profile-case",
                case,
                "--generation-profile-wavelength-nm",
                f"{config.baseline_wavelength_nm:g}",
                "--generation-probe-g0-cm3-s",
                f"{config.photo_g0_cm3_s:g}",
                "--generation-probe-x-um",
                f"{x_um:g}",
                "--generation-probe-depth-um",
                f"{depth_um:g}",
                "--generation-probe-sigma-x-um",
                f"{config.photo_sigma_x_um:g}",
                "--generation-probe-sigma-y-um",
                f"{config.photo_sigma_y_um:g}",
            ]
        )
    else:
        command.extend(
            [
                "--generation-profile-case",
                probe_id,
                "--generation-profile-wavelength-nm",
                "0",
            ]
        )
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
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"DD probe failed for {probe_id}; see {log_path}\n{tail}")
    if not summary_path.exists():
        raise RuntimeError(f"DD probe did not produce {summary_path}")
    return summary_path


def integrate_generation_current(mesh_path: Path, node_profile_csv: Path) -> float:
    mesh = parse_gmsh_msh22(mesh_path)
    rows = read_csv_rows(node_profile_csv)
    by_xy: dict[tuple[float, float], float] = {}
    for row in rows:
        by_xy[coordinate_key(float(row["x_cm"]), float(row["y_cm"]))] = float(
            row["OpticalGenerationRate"]
        )
    id_to_generation: dict[int, float] = {}
    missing: list[int] = []
    for node_id, xy in mesh.nodes.items():
        value = by_xy.get(coordinate_key(*xy))
        if value is None:
            missing.append(node_id)
            value = 0.0
        id_to_generation[node_id] = value
    if missing:
        raise RuntimeError(f"{len(missing)} mesh nodes are missing from {node_profile_csv}")
    integral = 0.0
    for triangle in mesh.triangles:
        area = triangle_area_cm2([mesh.nodes[node_id] for node_id in triangle])
        g = float(np.mean([id_to_generation[node_id] for node_id in triangle]))
        integral += g * area
    return Q_E * integral


def integrate_gaussian_probe_current(config: DDProbeConfig, mesh_path: Path, x_um: float, depth_um: float) -> float:
    mesh = parse_gmsh_msh22(mesh_path)
    sigma_x = max(config.photo_sigma_x_um, 1.0e-30)
    sigma_y = max(config.photo_sigma_y_um, 1.0e-30)
    integral = 0.0
    for triangle in mesh.triangles:
        points_um = [(mesh.nodes[node_id][0] * 1.0e4, mesh.nodes[node_id][1] * 1.0e4) for node_id in triangle]
        area = triangle_area_cm2([mesh.nodes[node_id] for node_id in triangle])
        values = [
            config.photo_g0_cm3_s
            * math.exp(
                -((px - x_um) ** 2) / (2.0 * sigma_x * sigma_x)
                - ((py - depth_um) ** 2) / (2.0 * sigma_y * sigma_y)
            )
            for px, py in points_um
        ]
        integral += float(np.mean(values)) * area
    return Q_E * integral


def summarize_probe(
    config: DDProbeConfig,
    summary_path: Path,
    x_um: float,
    depth_um: float,
    index: int,
    case: str = "",
    baseline_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    if baseline_summary is not None:
        generated_current = integrate_gaussian_probe_current(config, config.mesh, x_um, depth_um)
        left = float(summary.get("left_photo_delta_a_per_cm", math.nan)) - float(
            baseline_summary.get("left_photo_delta_a_per_cm", math.nan)
        )
        right = float(summary.get("right_photo_delta_a_per_cm", math.nan)) - float(
            baseline_summary.get("right_photo_delta_a_per_cm", math.nan)
        )
        mode = "baseline_illumination_plus_local_perturbation"
    else:
        generated_current = integrate_generation_current(
            config.mesh,
            Path(summary["outputs"]["node_profile_2d_csv"]),
        )
        left = float(summary.get("left_photo_delta_a_per_cm", math.nan))
        right = float(summary.get("right_photo_delta_a_per_cm", math.nan))
        mode = "local_generation_only"
    if generated_current > 0 and math.isfinite(generated_current):
        w_left = left / generated_current
        w_right = right / generated_current
    else:
        w_left = math.nan
        w_right = math.nan
    total = abs(w_left) + abs(w_right) if math.isfinite(w_left) and math.isfinite(w_right) else math.nan
    return {
        "probe_index": index,
        "probe_id": summary_path.parent.name,
        "case": case,
        "probe_mode": mode,
        "x_um": x_um,
        "depth_um": depth_um,
        "photo_sigma_x_um": config.photo_sigma_x_um,
        "photo_sigma_y_um": config.photo_sigma_y_um,
        "generated_current_a_per_cm": generated_current,
        "left_photo_delta_a_per_cm": left,
        "right_photo_delta_a_per_cm": right,
        "w_left_devsim_dd_probe": w_left,
        "w_right_devsim_dd_probe": w_right,
        "w_total_devsim_dd_probe": total,
        "split_phase_x": float(summary.get("photo_split_phase_x_proxy", math.nan)),
        "terminal_balance_illuminated_a_per_cm": float(
            summary.get("terminal_current_balance_illuminated_a_per_cm", math.nan)
        ),
        "transport_model": summary.get("transport_summary", {}).get("model", ""),
        "field_mobility_model": summary.get("transport_summary", {}).get("field_mobility_model", ""),
        "summary_json": str(summary_path),
    }


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    x = np.asarray([row["x_um"] for row in rows], dtype=float)
    depth = np.asarray([row["depth_um"] for row in rows], dtype=float)
    plots = [
        (np.asarray([row["w_left_devsim_dd_probe"] for row in rows], dtype=float), "W left DD probe"),
        (np.asarray([row["w_right_devsim_dd_probe"] for row in rows], dtype=float), "W right DD probe"),
        (np.asarray([row["w_total_devsim_dd_probe"] for row in rows], dtype=float), "W total DD probe"),
        (np.asarray([row["split_phase_x"] for row in rows], dtype=float), "Split phase"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    for axis, (values, title) in zip(axes, plots):
        sc = axis.scatter(x, depth, c=values, s=44, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x (um)")
        axis.set_ylabel("depth (um)")
        axis.invert_yaxis()
        fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config: DDProbeConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    points = probe_grid(config)
    baselines = baseline_summary_by_case(config)
    cases = list(config.baseline_cases) if config.baseline_cases else [""]
    rows: list[dict[str, Any]] = []
    for case in cases:
        baseline = baselines.get(case) if case else None
        if case and baseline is None:
            raise RuntimeError(f"missing baseline summary for case {case}")
        for index, (x_um, depth_um) in enumerate(points):
            summary_path = run_probe(config, x_um, depth_um, index, case=case)
            rows.append(
                summarize_probe(
                    config,
                    summary_path,
                    x_um,
                    depth_um,
                    index,
                    case=case,
                    baseline_summary=baseline,
                )
            )

    csv_path = config.output_dir / "dd_probe_response_2d.csv"
    json_path = config.output_dir / "dd_probe_response_2d_summary.json"
    plot_path = config.output_dir / "dd_probe_response_2d.png"
    write_csv(csv_path, rows)
    save_plot(plot_path, rows)
    finite_rows = [
        row
        for row in rows
        if math.isfinite(float(row["w_left_devsim_dd_probe"]))
        and math.isfinite(float(row["w_right_devsim_dd_probe"]))
    ]
    terminal_balances = [
        abs(float(row["terminal_balance_illuminated_a_per_cm"]))
        for row in rows
        if math.isfinite(float(row["terminal_balance_illuminated_a_per_cm"]))
    ]
    config_dict = asdict(config)
    config_dict["mesh"] = str(config.mesh)
    config_dict["measured_profile"] = str(config.measured_profile)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["baseline_generation_map_npz"] = (
        str(config.baseline_generation_map_npz) if config.baseline_generation_map_npz else ""
    )
    config_dict["baseline_cases"] = list(config.baseline_cases)
    config_dict["baseline_summary"] = [str(path) for path in config.baseline_summary]
    summary = {
        "schema": "devsim_dd_probe_response_2d_v1",
        "method": "direct_drift_diffusion_local_generation_probe",
        "config": config_dict,
        "probe_count": len(rows),
        "case_count": len(cases),
        "cases": cases,
        "finite_probe_count": len(finite_rows),
        "terminal_balance_max_abs_a_per_cm": max(terminal_balances) if terminal_balances else math.nan,
        "outputs": {
            "csv": str(csv_path),
            "summary_json": str(json_path),
            "plot_png": str(plot_path),
        },
        "limitations": [
            "This is a sparse set of direct DEVSIM drift-diffusion local-generation probe solves.",
            "When baseline_generation_map_npz is supplied, each probe is a local perturbation on top of the named full-illumination case.",
            "The CSV is interpolated in tcad_gw_coupling.py when used as W_devsim_dd_probe.",
            "It is not a mathematically exact adjoint collection probability and is not measured/calibrated.",
            "Probe spacing, Gaussian spot size, and operating bias must be convergence-checked before product LUT use.",
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=ROOT / "runs/gmsh_split_pd_2d_reference_native/split_pixel_2d.msh")
    parser.add_argument("--measured-profile", type=Path, default=ROOT / "measured_profiles/reference_cmos_ppd_1p4um/profile.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/devsim_dd_probe_response_2d_reference")
    parser.add_argument("--width-um", type=float, default=1.4)
    parser.add_argument("--depth-um", type=float, default=3.0)
    parser.add_argument("--split-gap-um", type=float, default=0.04)
    parser.add_argument("--x-count", type=int, default=5)
    parser.add_argument("--depth-count", type=int, default=5)
    parser.add_argument("--x-min-um", type=float, default=None)
    parser.add_argument("--x-max-um", type=float, default=None)
    parser.add_argument("--depth-min-um", type=float, default=None)
    parser.add_argument("--depth-max-um", type=float, default=None)
    parser.add_argument("--photo-sigma-x-um", type=float, default=0.07)
    parser.add_argument("--photo-sigma-y-um", type=float, default=0.10)
    parser.add_argument("--photo-g0-cm3-s", type=float, default=1.0e20)
    parser.add_argument("--baseline-generation-map-npz", type=Path, default=None)
    parser.add_argument("--baseline-cases", default="")
    parser.add_argument("--baseline-wavelength-nm", type=float, default=550.0)
    parser.add_argument("--baseline-summary", type=Path, nargs="*", default=[])
    parser.add_argument("--reverse-bias-v", type=float, default=-1.0)
    parser.add_argument("--dd-max-iterations", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        DDProbeConfig(
            mesh=args.mesh,
            measured_profile=args.measured_profile,
            output_dir=args.output_dir,
            width_um=args.width_um,
            depth_um=args.depth_um,
            split_gap_um=args.split_gap_um,
            x_count=args.x_count,
            depth_count=args.depth_count,
            x_min_um=args.x_min_um,
            x_max_um=args.x_max_um,
            depth_min_um=args.depth_min_um,
            depth_max_um=args.depth_max_um,
            photo_sigma_x_um=args.photo_sigma_x_um,
            photo_sigma_y_um=args.photo_sigma_y_um,
            photo_g0_cm3_s=args.photo_g0_cm3_s,
            baseline_generation_map_npz=args.baseline_generation_map_npz,
            baseline_cases=tuple(case for case in args.baseline_cases.split(",") if case),
            baseline_wavelength_nm=args.baseline_wavelength_nm,
            baseline_summary=tuple(args.baseline_summary),
            reverse_bias_v=args.reverse_bias_v,
            dd_max_iterations=args.dd_max_iterations,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
