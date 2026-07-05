#!/usr/bin/env python3
"""Run a small 2D FDTD smoke simulation with the open-source fdtd package."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parent / "fdtd"
sys.path.insert(0, str(SOURCE_ROOT))

import fdtd


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "runs" / "basic_2d"
    output_dir.mkdir(parents=True, exist_ok=True)

    fdtd.set_backend("numpy")

    grid = fdtd.Grid(shape=(24e-6, 14e-6, 1), grid_spacing=155e-9)
    wavelength = 1550e-9

    # Add a dielectric block so the wave interacts with a non-trivial object.
    grid[10e-6:14e-6, 5e-6:9e-6, 0] = fdtd.Object(
        permittivity=1.7**2,
        name="dielectric_block",
    )

    grid[3e-6, 4e-6:10e-6, 0] = fdtd.LineSource(
        period=wavelength / (3e8),
        amplitude=1.0,
        name="source",
    )
    grid[19e-6, :, 0] = fdtd.LineDetector(name="detector")

    pml_thickness = 10
    grid[0:pml_thickness, :, :] = fdtd.PML(name="pml_xlow")
    grid[-pml_thickness:, :, :] = fdtd.PML(name="pml_xhigh")
    grid[:, 0:pml_thickness, :] = fdtd.PML(name="pml_ylow")
    grid[:, -pml_thickness:, :] = fdtd.PML(name="pml_yhigh")

    total_steps = 160
    print(grid)
    grid.run(total_time=total_steps, progress_bar=True)

    detector_e = np.asarray(grid.detector.E)
    detector_h = np.asarray(grid.detector.H)
    final_energy = np.asarray(grid.E**2 + grid.H**2).sum(axis=-1)

    summary = {
        "backend": fdtd.backend.__class__.__name__,
        "grid_shape": [grid.Nx, grid.Ny, grid.Nz],
        "grid_spacing_m": grid.grid_spacing,
        "courant_number": grid.courant_number,
        "time_steps": total_steps,
        "detector_samples": list(detector_e.shape),
        "peak_abs_detector_ez": float(np.max(np.abs(detector_e[..., 2]))),
        "peak_abs_detector_h": float(np.max(np.abs(detector_h))),
        "final_total_field_energy": float(np.sum(final_energy)),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    field_fig = grid.visualize(z=0, cmap="inferno", show=False)
    field_fig.savefig(output_dir / "field_energy.png", dpi=160, bbox_inches="tight")
    plt.close(field_fig)

    center_index = detector_e.shape[1] // 2
    trace_fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(detector_e[:, center_index, 2], color="C0", linewidth=1.8)
    axis.set_title("Detector center Ez trace")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Ez")
    axis.grid(True, alpha=0.25)
    trace_fig.tight_layout()
    trace_fig.savefig(output_dir / "detector_ez_trace.png", dpi=160)
    plt.close(trace_fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
