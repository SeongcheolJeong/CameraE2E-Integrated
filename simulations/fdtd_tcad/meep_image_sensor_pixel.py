#!/usr/bin/env python3
"""2D Meep smoke model for an image-sensor pixel optical stack.

Units are microns. This is an optical FDTD model only; carrier transport and
photodiode collection physics are outside Meep's scope.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from meep.materials import Al

from meep_microlens_array_3d import VALID_SHIELD_MODES, shield_blocks_local_point


ROOT = Path(__file__).resolve().parent
SI_NK_PATH = ROOT / "materials" / "Si-Green-2008.yml"


@dataclass(frozen=True)
class PixelGeometry:
    pitch: float = 1.40
    pml: float = 0.50
    air_top: float = 0.55
    lens_height: float = 0.35
    cfa_thickness: float = 0.45
    passivation_thickness: float = 0.15
    si_thickness: float = 2.00
    bottom_air: float = 0.25
    metal_edge_width: float = 0.12

    @property
    def cell_y(self) -> float:
        return (
            2 * self.pml
            + self.air_top
            + self.lens_height
            + self.cfa_thickness
            + self.passivation_thickness
            + self.si_thickness
            + self.bottom_air
        )

    @property
    def usable_top(self) -> float:
        return 0.5 * self.cell_y - self.pml

    @property
    def lens_top(self) -> float:
        return self.usable_top - self.air_top

    @property
    def lens_bottom(self) -> float:
        return self.lens_top - self.lens_height

    @property
    def cfa_top(self) -> float:
        return self.lens_bottom

    @property
    def cfa_bottom(self) -> float:
        return self.cfa_top - self.cfa_thickness

    @property
    def pass_top(self) -> float:
        return self.cfa_bottom

    @property
    def pass_bottom(self) -> float:
        return self.pass_top - self.passivation_thickness

    @property
    def si_top(self) -> float:
        return self.pass_bottom

    @property
    def si_bottom(self) -> float:
        return self.si_top - self.si_thickness

    @property
    def source_y(self) -> float:
        return self.usable_top - 0.18

    @property
    def incident_monitor_y(self) -> float:
        return self.source_y - 0.14

    @property
    def si_top_monitor_y(self) -> float:
        return self.si_top + 0.02

    @property
    def si_bottom_monitor_y(self) -> float:
        return self.si_bottom + 0.08


def load_si_nk(path: Path, wavelength_um: float) -> tuple[float, float]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        try:
            rows.append(tuple(float(part) for part in parts))
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric n,k table found in {path}")

    table = np.asarray(rows)
    wavelengths = table[:, 0]
    if wavelength_um < wavelengths.min() or wavelength_um > wavelengths.max():
        raise ValueError(
            f"wavelength {wavelength_um:.4f} um is outside silicon nk table range "
            f"{wavelengths.min():.4f}-{wavelengths.max():.4f} um"
        )
    n = float(np.interp(wavelength_um, wavelengths, table[:, 1]))
    k = float(np.interp(wavelength_um, wavelengths, table[:, 2]))
    return n, k


def medium_from_nk(n: float, k: float, frequency: float) -> mp.Medium:
    eps_real = n * n - k * k
    eps_imag = 2 * n * k
    d_conductivity = 2 * math.pi * frequency * eps_imag / eps_real
    return mp.Medium(epsilon=eps_real, D_conductivity=d_conductivity)


def make_material_function(
    geom: PixelGeometry,
    silicon: mp.Medium,
    cfa: mp.Medium,
    passivation: mp.Medium,
    lens: mp.Medium,
    shield_mode: str,
):
    active_half_width = 0.5 * geom.pitch - geom.metal_edge_width
    cap_half_width = 0.5 * geom.pitch
    cap_height = geom.lens_height
    lens_radius = (cap_half_width**2 + cap_height**2) / (2 * cap_height)
    lens_center_y = geom.lens_top - lens_radius

    def material(point: mp.Vector3):
        x = point.x
        y = point.y

        if geom.si_bottom <= y < geom.si_top:
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            if shield_blocks_local_point(
                x,
                0.0,
                active_half_width,
                shield_mode,
                pair_index=0,
            ):
                return Al
            return cfa
        if geom.lens_bottom <= y <= geom.lens_top:
            in_cap = x * x + (y - lens_center_y) ** 2 <= lens_radius**2
            if in_cap:
                return lens
        return mp.air

    return material


def build_simulation(
    geom: PixelGeometry,
    wavelength_um: float,
    resolution: int,
    include_stack: bool,
    shield_mode: str,
    include_dft_fields: bool = False,
):
    frequency = 1 / wavelength_um
    cell_size = mp.Vector3(geom.pitch, geom.cell_y, 0)
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ez,
        center=mp.Vector3(0, geom.source_y),
        size=mp.Vector3(geom.pitch, 0),
    )

    kwargs = {}
    dft_materials = []
    if include_stack:
        si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)
        silicon = medium_from_nk(si_n, si_k, frequency)
        cfa_green = medium_from_nk(1.65, 0.015, frequency)
        passivation = mp.Medium(index=1.46)
        lens = mp.Medium(index=1.49)
        kwargs["material_function"] = make_material_function(
            geom, silicon, cfa_green, passivation, lens, shield_mode
        )
        dft_materials = [silicon, cfa_green]
        if shield_mode != "off":
            dft_materials.append(Al)

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml, direction=mp.Y)],
        sources=[source],
        resolution=resolution,
        default_material=mp.air,
        extra_materials=dft_materials,
        **kwargs,
    )

    incident_flux = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0, geom.incident_monitor_y),
            size=mp.Vector3(geom.pitch, 0),
        ),
    )
    si_top_flux = None
    si_bottom_flux = None
    dft_fields = None
    if include_stack:
        si_top_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_top_monitor_y),
                size=mp.Vector3(geom.pitch, 0),
            ),
        )
        si_bottom_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_bottom_monitor_y),
                size=mp.Vector3(geom.pitch, 0),
            ),
        )
        if include_dft_fields:
            dft_fields = sim.add_dft_fields(
                [mp.Ez],
                frequency,
                0,
                1,
                center=mp.Vector3(),
                size=cell_size,
            )

    return sim, incident_flux, si_top_flux, si_bottom_flux, dft_fields


def run_simulation(sim: mp.Simulation, after_source_time: float) -> None:
    sim.run(until_after_sources=after_source_time)


def material_id_map(
    geom: PixelGeometry,
    nx: int,
    ny: int,
    shield_mode: str,
) -> np.ndarray:
    xs = np.linspace(-0.5 * geom.pitch, 0.5 * geom.pitch, nx)
    ys = np.linspace(-0.5 * geom.cell_y, 0.5 * geom.cell_y, ny)
    active_half_width = 0.5 * geom.pitch - geom.metal_edge_width
    cap_half_width = 0.5 * geom.pitch
    lens_radius = (cap_half_width**2 + geom.lens_height**2) / (2 * geom.lens_height)
    lens_center_y = geom.lens_top - lens_radius
    labels = np.zeros((nx, ny), dtype=float)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if geom.si_bottom <= y < geom.si_top:
                labels[ix, iy] = 5
            elif geom.pass_bottom <= y < geom.pass_top:
                labels[ix, iy] = 4
            elif geom.cfa_bottom <= y < geom.cfa_top:
                labels[ix, iy] = (
                    3
                    if shield_blocks_local_point(
                        x,
                        0.0,
                        active_half_width,
                        shield_mode,
                        pair_index=0,
                    )
                    else 2
                )
            elif geom.lens_bottom <= y <= geom.lens_top:
                if x * x + (y - lens_center_y) ** 2 <= lens_radius**2:
                    labels[ix, iy] = 1
    return labels


def save_plots(
    output_dir: Path,
    geom: PixelGeometry,
    ez_dft: np.ndarray,
    wavelength_um: float,
    shield_mode: str,
) -> None:
    extent = [-0.5 * geom.pitch, 0.5 * geom.pitch, -0.5 * geom.cell_y, 0.5 * geom.cell_y]
    field = np.abs(ez_dft)
    field /= max(float(np.max(field)), 1e-30)
    labels = material_id_map(geom, field.shape[0], field.shape[1], shield_mode)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    material_image = axes[0].imshow(
        labels.T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=9,
    )
    axes[0].set_title("Pixel stack material map")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    cbar = fig.colorbar(material_image, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["air", "lens", "CFA", "Al", "pass", "Si"])

    field_image = axes[1].imshow(
        field.T,
        origin="lower",
        extent=extent,
        interpolation="spline16",
        cmap="inferno",
    )
    axes[1].set_title(f"|Ez| DFT field, lambda={wavelength_um * 1000:.0f} nm")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(field_image, ax=axes[1], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.axhline(geom.si_top, color="white", linewidth=0.8, alpha=0.7)
        axis.axhline(geom.si_bottom, color="white", linewidth=0.8, alpha=0.7)

    fig.savefig(output_dir / "pixel_stack_and_ez.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--after-source-time", type=float, default=80.0)
    parser.add_argument(
        "--shield-mode",
        choices=VALID_SHIELD_MODES,
        default="off",
        help="Optional optical mask mode. Default off means no metal shield in the imaging pixel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "meep_pixel_2d",
    )
    args = parser.parse_args()

    wavelength_um = args.wavelength_nm / 1000
    frequency = 1 / wavelength_um
    geom = PixelGeometry()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)
    print(f"Meep {mp.__version__}")
    print(
        f"lambda={args.wavelength_nm:.1f} nm, Si n={si_n:.4f}, "
        f"k={si_k:.5f}, shield={args.shield_mode}"
    )

    ref_sim, ref_incident, _, _, _ = build_simulation(
        geom,
        wavelength_um,
        args.resolution,
        include_stack=False,
        shield_mode=args.shield_mode,
    )
    run_simulation(ref_sim, args.after_source_time)
    incident_flux = mp.get_fluxes(ref_incident)[0]
    downward_sign = 1 if incident_flux >= 0 else -1
    incident_power = abs(incident_flux)

    sim, full_incident, si_top, si_bottom, dft_fields = build_simulation(
        geom,
        wavelength_um,
        args.resolution,
        include_stack=True,
        shield_mode=args.shield_mode,
        include_dft_fields=True,
    )
    run_simulation(sim, args.after_source_time)
    incident_full = mp.get_fluxes(full_incident)[0] * downward_sign
    si_top_power = mp.get_fluxes(si_top)[0] * downward_sign
    si_bottom_power = mp.get_fluxes(si_bottom)[0] * downward_sign
    si_absorbed = si_top_power - si_bottom_power
    ez_dft = sim.get_dft_array(dft_fields, mp.Ez, 0)

    save_plots(args.output_dir, geom, ez_dft, wavelength_um, args.shield_mode)

    summary = {
        "meep_version": mp.__version__,
        "wavelength_nm": args.wavelength_nm,
        "frequency_1_per_um": frequency,
        "resolution_px_per_um": args.resolution,
        "cell_size_um": [geom.pitch, geom.cell_y],
        "shield_mode": args.shield_mode,
        "shield_mask_edge_width_um": geom.metal_edge_width,
        "silicon_nk_source": str(SI_NK_PATH),
        "silicon_n": si_n,
        "silicon_k": si_k,
        "reference_incident_flux_raw": incident_flux,
        "incident_power_reference": incident_power,
        "incident_monitor_net_power_normalized": incident_full / incident_power,
        "si_top_net_downward_power_normalized": si_top_power / incident_power,
        "si_bottom_net_downward_power_normalized": si_bottom_power / incident_power,
        "si_absorption_fraction_estimate": si_absorbed / incident_power,
        "notes": [
            "2D optical FDTD only; no carrier generation, diffusion, or collection model.",
            "Default shield_mode=off has no metal optical shield; edge/PDAF modes are explicit variants.",
            "Silicon absorption is estimated from net flux difference between two monitor planes inside/above Si.",
            "Run resolution convergence and add measured material data before using results quantitatively.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
