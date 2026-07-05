#!/usr/bin/env python3
"""3D periodic Meep unit-cell model for an image-sensor microlens array.

Units are microns. The x and z directions are periodic, so this represents one
repeated microlens/pixel cell at normal incidence. The y direction uses PML.
This is an optical model only; carrier transport and charge collection are not
included.
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


ROOT = Path(__file__).resolve().parent
SI_NK_PATH = ROOT / "materials" / "Si-Green-2008.yml"
VALID_SHIELD_MODES = ("off", "edge", "pdaf_left", "pdaf_right", "pdaf_pair")


@dataclass(frozen=True)
class MicrolensArrayGeometry:
    pitch: float = 1.40
    pml: float = 0.45
    air_top: float = 0.55
    lens_height: float = 0.35
    lens_edge_gap: float = 0.04
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

    @property
    def focal_plane_y(self) -> float:
        return self.si_top + 0.02

    @property
    def active_half_width(self) -> float:
        return 0.5 * self.pitch - self.metal_edge_width

    @property
    def lens_aperture_radius(self) -> float:
        return 0.5 * self.pitch - self.lens_edge_gap

    @property
    def lens_sphere_radius(self) -> float:
        a = self.lens_aperture_radius
        h = self.lens_height
        return (a * a + h * h) / (2 * h)

    @property
    def lens_center_y(self) -> float:
        return self.lens_top - self.lens_sphere_radius


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


def shield_blocks_local_point(
    dx_um: float,
    dz_um: float,
    active_half_width_um: float,
    shield_mode: str,
    pair_index: int = 0,
) -> bool:
    mode = shield_mode.lower()
    if mode == "off":
        return False
    if mode == "edge":
        return abs(dx_um) > active_half_width_um or abs(dz_um) > active_half_width_um
    if mode == "pdaf_left":
        return dx_um < 0.0
    if mode == "pdaf_right":
        return dx_um > 0.0
    if mode == "pdaf_pair":
        return dx_um < 0.0 if pair_index % 2 == 0 else dx_um > 0.0
    raise ValueError(f"Unsupported shield mode: {shield_mode}")


def make_material_function(
    geom: MicrolensArrayGeometry,
    silicon: mp.Medium,
    cfa: mp.Medium,
    passivation: mp.Medium,
    lens: mp.Medium,
    metal: mp.Medium,
    shield_mode: str,
):
    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        z = point.z

        if geom.si_bottom <= y < geom.si_top:
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            if shield_blocks_local_point(
                x,
                z,
                geom.active_half_width,
                shield_mode,
                pair_index=0,
            ):
                return metal
            return cfa
        if geom.lens_bottom <= y <= geom.lens_top:
            r2 = x * x + z * z
            in_aperture = r2 <= geom.lens_aperture_radius**2
            on_sphere = r2 + (y - geom.lens_center_y) ** 2 <= geom.lens_sphere_radius**2
            if in_aperture and on_sphere:
                return lens
        return mp.air

    return material


def build_simulation(
    geom: MicrolensArrayGeometry,
    wavelength_um: float,
    resolution: int,
    metal_model: str,
    shield_mode: str,
    include_stack: bool,
    include_dft_fields: bool = False,
):
    frequency = 1 / wavelength_um
    cell_size = mp.Vector3(geom.pitch, geom.cell_y, geom.pitch)
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ex,
        center=mp.Vector3(0, geom.source_y, 0),
        size=mp.Vector3(geom.pitch, 0, geom.pitch),
    )

    kwargs = {}
    extra_materials = []
    if include_stack:
        si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)
        silicon = medium_from_nk(si_n, si_k, frequency)
        cfa_green = medium_from_nk(1.65, 0.015, frequency)
        passivation = mp.Medium(index=1.46)
        lens = mp.Medium(index=1.49)
        metal = Al if metal_model == "dispersive-al" else mp.metal
        kwargs["material_function"] = make_material_function(
            geom, silicon, cfa_green, passivation, lens, metal, shield_mode
        )
        extra_materials = [silicon, cfa_green]
        if shield_mode != "off" and metal_model == "dispersive-al":
            extra_materials.append(Al)
        elif shield_mode != "off":
            extra_materials.append(mp.metal)

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml, direction=mp.Y)],
        sources=[source],
        resolution=resolution,
        k_point=mp.Vector3(),
        default_material=mp.air,
        extra_materials=extra_materials,
        **kwargs,
    )

    incident_flux = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0, geom.incident_monitor_y, 0),
            size=mp.Vector3(geom.pitch, 0, geom.pitch),
        ),
    )

    si_top_flux = None
    si_bottom_flux = None
    xy_fields = None
    focal_fields = None
    if include_stack:
        si_top_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_top_monitor_y, 0),
                size=mp.Vector3(geom.pitch, 0, geom.pitch),
            ),
        )
        si_bottom_flux = sim.add_flux(
            frequency,
            0,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, geom.si_bottom_monitor_y, 0),
                size=mp.Vector3(geom.pitch, 0, geom.pitch),
            ),
        )
        if include_dft_fields:
            xy_fields = sim.add_dft_fields(
                [mp.Ex],
                frequency,
                0,
                1,
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(geom.pitch, geom.cell_y, 0),
            )
            focal_fields = sim.add_dft_fields(
                [mp.Ex],
                frequency,
                0,
                1,
                center=mp.Vector3(0, geom.focal_plane_y, 0),
                size=mp.Vector3(geom.pitch, 0, geom.pitch),
            )

    return sim, incident_flux, si_top_flux, si_bottom_flux, xy_fields, focal_fields


def material_id_xy(
    geom: MicrolensArrayGeometry,
    nx: int,
    ny: int,
    shield_mode: str,
) -> np.ndarray:
    xs = np.linspace(-0.5 * geom.pitch, 0.5 * geom.pitch, nx)
    ys = np.linspace(-0.5 * geom.cell_y, 0.5 * geom.cell_y, ny)
    labels = np.zeros((nx, ny), dtype=float)
    z = 0.0
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
                        z,
                        geom.active_half_width,
                        shield_mode,
                        pair_index=0,
                    )
                    else 2
                )
            elif geom.lens_bottom <= y <= geom.lens_top:
                r2 = x * x + z * z
                if (
                    r2 <= geom.lens_aperture_radius**2
                    and r2 + (y - geom.lens_center_y) ** 2 <= geom.lens_sphere_radius**2
                ):
                    labels[ix, iy] = 1
    return labels


def normalize_field(field: np.ndarray) -> np.ndarray:
    values = np.abs(np.squeeze(field))
    return values / max(float(np.max(values)), 1e-30)


def focal_metrics(geom: MicrolensArrayGeometry, focal_field: np.ndarray) -> dict:
    intensity = normalize_field(focal_field) ** 2
    nx, nz = intensity.shape
    xs = np.linspace(-0.5 * geom.pitch, 0.5 * geom.pitch, nx)
    zs = np.linspace(-0.5 * geom.pitch, 0.5 * geom.pitch, nz)
    x_grid, z_grid = np.meshgrid(xs, zs, indexing="ij")
    total = float(np.sum(intensity))
    if total <= 0:
        return {
            "focal_centroid_x_um": None,
            "focal_centroid_z_um": None,
            "focal_rms_radius_um": None,
            "active_area_fraction_at_focal_plane": None,
        }

    cx = float(np.sum(intensity * x_grid) / total)
    cz = float(np.sum(intensity * z_grid) / total)
    rms_radius = float(
        np.sqrt(np.sum(intensity * ((x_grid - cx) ** 2 + (z_grid - cz) ** 2)) / total)
    )
    active_mask = (np.abs(x_grid) <= geom.active_half_width) & (
        np.abs(z_grid) <= geom.active_half_width
    )
    active_fraction = float(np.sum(intensity[active_mask]) / total)
    return {
        "focal_centroid_x_um": cx,
        "focal_centroid_z_um": cz,
        "focal_rms_radius_um": rms_radius,
        "active_area_fraction_at_focal_plane": active_fraction,
    }


def save_plots(
    output_dir: Path,
    geom: MicrolensArrayGeometry,
    xy_field: np.ndarray,
    focal_field: np.ndarray,
    wavelength_um: float,
    shield_mode: str,
) -> None:
    xy = normalize_field(xy_field)
    focal = normalize_field(focal_field)
    xy_labels = material_id_xy(geom, xy.shape[0], xy.shape[1], shield_mode)
    xy_extent = [
        -0.5 * geom.pitch,
        0.5 * geom.pitch,
        -0.5 * geom.cell_y,
        0.5 * geom.cell_y,
    ]
    focal_extent = [
        -0.5 * geom.pitch,
        0.5 * geom.pitch,
        -0.5 * geom.pitch,
        0.5 * geom.pitch,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    material_image = axes[0].imshow(
        xy_labels.T,
        origin="lower",
        extent=xy_extent,
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=9,
    )
    axes[0].set_title("x-y material slice (z=0)")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    cbar = fig.colorbar(material_image, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["air", "lens", "CFA", "metal", "pass", "Si"])

    xy_image = axes[1].imshow(
        xy.T,
        origin="lower",
        extent=xy_extent,
        interpolation="spline16",
        cmap="inferno",
    )
    axes[1].set_title(f"x-y |Ex|, lambda={wavelength_um * 1000:.0f} nm")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    fig.colorbar(xy_image, ax=axes[1], fraction=0.046, pad=0.04)

    focal_image = axes[2].imshow(
        focal.T,
        origin="lower",
        extent=focal_extent,
        interpolation="spline16",
        cmap="viridis",
    )
    axes[2].set_title("focal plane |Ex| at Si top")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("z (um)")
    active = geom.active_half_width
    axes[2].plot(
        [-active, active, active, -active, -active],
        [-active, -active, active, active, -active],
        color="white",
        linewidth=1.0,
    )
    fig.colorbar(focal_image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes[:2]:
        axis.axhline(geom.si_top, color="white", linewidth=0.8, alpha=0.7)
        axis.axhline(geom.si_bottom, color="white", linewidth=0.8, alpha=0.7)

    fig.savefig(output_dir / "microlens_array_3d_slices.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--after-source-time", type=float, default=60.0)
    parser.add_argument(
        "--metal-model",
        choices=("pec", "dispersive-al"),
        default="pec",
        help="Metal material for optional optical shield modes; dispersive Al can require extra stability checks.",
    )
    parser.add_argument(
        "--shield-mode",
        choices=VALID_SHIELD_MODES,
        default="off",
        help="Optional optical mask mode. Default off means no metal shield in the imaging pixel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "meep_microlens_array_3d",
    )
    args = parser.parse_args()

    geom = MicrolensArrayGeometry()
    wavelength_um = args.wavelength_nm / 1000
    frequency = 1 / wavelength_um
    args.output_dir.mkdir(parents=True, exist_ok=True)

    si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)
    print(f"Meep {mp.__version__}")
    print(f"3D periodic microlens array unit cell")
    print(f"lambda={args.wavelength_nm:.1f} nm, Si n={si_n:.4f}, k={si_k:.5f}")
    print(
        f"cell=({geom.pitch:.3f}, {geom.cell_y:.3f}, {geom.pitch:.3f}) um, "
        f"resolution={args.resolution} px/um, shield={args.shield_mode}"
    )

    ref_sim, ref_incident, _, _, _, _ = build_simulation(
        geom,
        wavelength_um,
        args.resolution,
        args.metal_model,
        args.shield_mode,
        include_stack=False,
    )
    ref_sim.run(until_after_sources=args.after_source_time)
    incident_flux = mp.get_fluxes(ref_incident)[0]
    downward_sign = 1 if incident_flux >= 0 else -1
    incident_power = abs(incident_flux)

    sim, full_incident, si_top, si_bottom, xy_fields, focal_fields = build_simulation(
        geom,
        wavelength_um,
        args.resolution,
        args.metal_model,
        args.shield_mode,
        include_stack=True,
        include_dft_fields=True,
    )
    sim.run(until_after_sources=args.after_source_time)

    incident_full = mp.get_fluxes(full_incident)[0] * downward_sign
    si_top_power = mp.get_fluxes(si_top)[0] * downward_sign
    si_bottom_power = mp.get_fluxes(si_bottom)[0] * downward_sign
    si_absorbed = si_top_power - si_bottom_power

    xy_field = sim.get_dft_array(xy_fields, mp.Ex, 0)
    focal_field = sim.get_dft_array(focal_fields, mp.Ex, 0)
    save_plots(args.output_dir, geom, xy_field, focal_field, wavelength_um, args.shield_mode)

    metrics = focal_metrics(geom, focal_field)
    summary = {
        "meep_version": mp.__version__,
        "model": "3D periodic microlens array unit cell at normal incidence",
        "wavelength_nm": args.wavelength_nm,
        "frequency_1_per_um": frequency,
        "resolution_px_per_um": args.resolution,
        "cell_size_um": [geom.pitch, geom.cell_y, geom.pitch],
        "effective_rounded_cell_size_um": [
            round(geom.pitch * args.resolution) / args.resolution,
            round(geom.cell_y * args.resolution) / args.resolution,
            round(geom.pitch * args.resolution) / args.resolution,
        ],
        "metal_model": args.metal_model,
        "shield_mode": args.shield_mode,
        "shield_mask_edge_width_um": geom.metal_edge_width,
        "boundary_conditions": {
            "x": "periodic",
            "y": "PML",
            "z": "periodic",
        },
        "geometry_um": {
            "pitch": geom.pitch,
            "lens_height": geom.lens_height,
            "lens_aperture_radius": geom.lens_aperture_radius,
            "cfa_thickness": geom.cfa_thickness,
            "passivation_thickness": geom.passivation_thickness,
            "si_thickness": geom.si_thickness,
            "metal_edge_width": geom.metal_edge_width,
        },
        "silicon_nk_source": str(SI_NK_PATH),
        "silicon_n": si_n,
        "silicon_k": si_k,
        "reference_incident_flux_raw": incident_flux,
        "incident_power_reference": incident_power,
        "incident_monitor_net_power_normalized": incident_full / incident_power,
        "si_top_net_downward_power_normalized": si_top_power / incident_power,
        "si_bottom_net_downward_power_normalized": si_bottom_power / incident_power,
        "si_absorption_fraction_estimate": si_absorbed / incident_power,
        **metrics,
        "notes": [
            "This is one periodic unit cell, so it models an infinite regular microlens array.",
            "Default shield_mode=off has no metal optical shield; edge/PDAF modes are explicit variants.",
            "The result is an optical FDTD estimate only; it does not include carrier collection or TCAD physics.",
            "The default resolution is intentionally lightweight. Run convergence before using values quantitatively.",
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
