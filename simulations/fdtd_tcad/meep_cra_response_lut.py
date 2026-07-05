#!/usr/bin/env python3
"""Generate a CRA response LUT for camera-system simulation.

The model is a 3D periodic image-sensor microlens/pixel unit cell. Each case
represents a field point with a chief-ray angle and optional microlens/aperture
shift relative to the photodiode active area.

Units are microns. This is an optical proxy model; carrier transport and
electrical readout are not included.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from meep_microlens_array_3d import (
    SI_NK_PATH,
    VALID_SHIELD_MODES,
    MicrolensArrayGeometry,
    load_si_nk,
    medium_from_nk,
    shield_blocks_local_point,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CraCase:
    name: str
    cra_x_deg: float
    lens_shift_x_um: float
    field_norm: float = 0.0
    aperture_shift_x_um: float | None = None
    active_shift_x_um: float = 0.0

    @property
    def aperture_shift(self) -> float:
        return self.lens_shift_x_um if self.aperture_shift_x_um is None else self.aperture_shift_x_um


def parse_cases(raw: str) -> list[CraCase]:
    """Parse cases: name:cra_deg:lens_shift_um[:field_norm[:aperture_shift_um[:active_shift_um]]]."""
    cases = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError(
                "Each case must be name:cra_deg:lens_shift_um"
                "[:field_norm[:aperture_shift_um[:active_shift_um]]]"
            )
        name = parts[0]
        cra_x_deg = float(parts[1])
        lens_shift_x_um = float(parts[2])
        field_norm = float(parts[3]) if len(parts) >= 4 and parts[3] else 0.0
        aperture_shift = float(parts[4]) if len(parts) >= 5 and parts[4] else None
        active_shift = float(parts[5]) if len(parts) >= 6 and parts[5] else 0.0
        cases.append(
            CraCase(
                name=name,
                cra_x_deg=cra_x_deg,
                lens_shift_x_um=lens_shift_x_um,
                field_norm=field_norm,
                aperture_shift_x_um=aperture_shift,
                active_shift_x_um=active_shift,
            )
        )
    if not cases:
        raise ValueError("No cases parsed")
    return cases


def transverse_k_from_cra(frequency: float, cra_x_deg: float) -> tuple[float, float]:
    theta = math.radians(cra_x_deg)
    sx = math.sin(theta)
    if abs(sx) >= 1:
        raise ValueError(f"CRA angle is too large: {cra_x_deg}")
    return frequency * sx, 0.0


def phase_ramp(kx: float, kz: float):
    def amp(point: mp.Vector3):
        return np.exp(1j * 2 * np.pi * (kx * point.x + kz * point.z))

    return amp


def make_shifted_material_function(
    geom: MicrolensArrayGeometry,
    silicon: mp.Medium,
    cfa: mp.Medium,
    passivation: mp.Medium,
    lens: mp.Medium,
    metal: mp.Medium,
    case: CraCase,
    shield_mode: str,
):
    lens_shift = case.lens_shift_x_um
    aperture_shift = case.aperture_shift

    def material(point: mp.Vector3):
        x = point.x
        y = point.y
        z = point.z

        if geom.si_bottom <= y < geom.si_top:
            return silicon
        if geom.pass_bottom <= y < geom.pass_top:
            return passivation
        if geom.cfa_bottom <= y < geom.cfa_top:
            if shield_mode == "off":
                return cfa
            if shield_blocks_local_point(
                x - aperture_shift,
                z,
                geom.active_half_width,
                shield_mode,
                pair_index=0,
            ):
                return metal
            return cfa
        if geom.lens_bottom <= y <= geom.lens_top:
            dx = x - lens_shift
            r2 = dx * dx + z * z
            in_aperture = r2 <= geom.lens_aperture_radius**2
            on_sphere = r2 + (y - geom.lens_center_y) ** 2 <= geom.lens_sphere_radius**2
            if in_aperture and on_sphere:
                return lens
        return mp.air

    return material


def build_case_simulation(
    geom: MicrolensArrayGeometry,
    case: CraCase,
    wavelength_um: float,
    resolution: int,
    shield_mode: str,
    include_stack: bool,
    include_dft_fields: bool,
):
    frequency = 1 / wavelength_um
    kx, kz = transverse_k_from_cra(frequency, case.cra_x_deg)
    cell_size = mp.Vector3(geom.pitch, geom.cell_y, geom.pitch)
    source = mp.Source(
        src=mp.GaussianSource(frequency=frequency, fwidth=0.20 * frequency),
        component=mp.Ez,
        center=mp.Vector3(0, geom.source_y, 0),
        size=mp.Vector3(geom.pitch, 0, geom.pitch),
        amp_func=phase_ramp(kx, kz),
    )

    kwargs = {}
    extra_materials = []
    if include_stack:
        si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)
        silicon = medium_from_nk(si_n, si_k, frequency)
        cfa_green = medium_from_nk(1.65, 0.015, frequency)
        passivation = mp.Medium(index=1.46)
        lens = mp.Medium(index=1.49)
        metal = mp.metal
        kwargs["material_function"] = make_shifted_material_function(
            geom,
            silicon,
            cfa_green,
            passivation,
            lens,
            metal,
            case,
            shield_mode,
        )
        extra_materials = [silicon, cfa_green]
        if shield_mode != "off":
            extra_materials.append(metal)

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=[mp.PML(geom.pml, direction=mp.Y)],
        sources=[source],
        resolution=resolution,
        k_point=mp.Vector3(kx, 0, kz),
        force_complex_fields=True,
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
            focal_fields = sim.add_dft_fields(
                [mp.Ez],
                frequency,
                0,
                1,
                center=mp.Vector3(0, geom.focal_plane_y, 0),
                size=mp.Vector3(geom.pitch, 0, geom.pitch),
            )

    return sim, incident_flux, si_top_flux, si_bottom_flux, focal_fields


def normalized_intensity(field: np.ndarray) -> np.ndarray:
    values = np.abs(np.squeeze(field)) ** 2
    return values / max(float(np.max(values)), 1e-30)


def focal_metrics(geom: MicrolensArrayGeometry, case: CraCase, focal_field: np.ndarray) -> dict:
    intensity = normalized_intensity(focal_field)
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
    rms = float(
        np.sqrt(np.sum(intensity * ((x_grid - cx) ** 2 + (z_grid - cz) ** 2)) / total)
    )
    active_mask = (np.abs(x_grid - case.active_shift_x_um) <= geom.active_half_width) & (
        np.abs(z_grid) <= geom.active_half_width
    )
    active_fraction = float(np.sum(intensity[active_mask]) / total)
    return {
        "focal_centroid_x_um": cx,
        "focal_centroid_z_um": cz,
        "focal_rms_radius_um": rms,
        "active_area_fraction_at_focal_plane": active_fraction,
    }


def run_case(
    geom: MicrolensArrayGeometry,
    case: CraCase,
    wavelength_um: float,
    resolution: int,
    after_source_time: float,
    shield_mode: str,
):
    ref_sim, ref_incident, _, _, _ = build_case_simulation(
        geom,
        case,
        wavelength_um,
        resolution,
        shield_mode,
        include_stack=False,
        include_dft_fields=False,
    )
    ref_sim.run(until_after_sources=after_source_time)
    incident_flux = mp.get_fluxes(ref_incident)[0]
    downward_sign = 1 if incident_flux >= 0 else -1
    incident_power = abs(incident_flux)

    sim, full_incident, si_top, si_bottom, focal_fields = build_case_simulation(
        geom,
        case,
        wavelength_um,
        resolution,
        shield_mode,
        include_stack=True,
        include_dft_fields=True,
    )
    sim.run(until_after_sources=after_source_time)

    incident_full = mp.get_fluxes(full_incident)[0] * downward_sign
    si_top_power = mp.get_fluxes(si_top)[0] * downward_sign
    si_bottom_power = mp.get_fluxes(si_bottom)[0] * downward_sign
    si_absorption = (si_top_power - si_bottom_power) / incident_power
    focal_field = sim.get_dft_array(focal_fields, mp.Ez, 0)
    metrics = focal_metrics(geom, case, focal_field)
    collected_proxy = si_absorption * metrics["active_area_fraction_at_focal_plane"]

    row = {
        "case": case.name,
        "field_norm": case.field_norm,
        "cra_x_deg": case.cra_x_deg,
        "lens_shift_x_um": case.lens_shift_x_um,
        "aperture_shift_x_um": case.aperture_shift,
        "active_shift_x_um": case.active_shift_x_um,
        "shield_mode": shield_mode,
        "shield_mask_edge_width_um": geom.metal_edge_width,
        "incident_monitor_net_power_normalized": incident_full / incident_power,
        "si_top_net_downward_power_normalized": si_top_power / incident_power,
        "si_bottom_net_downward_power_normalized": si_bottom_power / incident_power,
        "si_absorption_fraction_estimate": si_absorption,
        "collected_response_proxy": collected_proxy,
        **metrics,
    }
    return row, normalized_intensity(focal_field)


def save_focal_maps(
    output_dir: Path,
    geom: MicrolensArrayGeometry,
    cases: list[CraCase],
    maps: list[np.ndarray],
    wavelength_nm: float,
) -> None:
    ncols = len(maps)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    extent = [-0.5 * geom.pitch, 0.5 * geom.pitch, -0.5 * geom.pitch, 0.5 * geom.pitch]
    for axis, case, fmap in zip(axes, cases, maps):
        image = axis.imshow(
            fmap.T,
            origin="lower",
            extent=extent,
            interpolation="spline16",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        active = geom.active_half_width
        cx = case.active_shift_x_um
        axis.plot(
            [cx - active, cx + active, cx + active, cx - active, cx - active],
            [-active, -active, active, active, -active],
            color="white",
            linewidth=1.0,
        )
        axis.set_title(f"{case.name}\nCRA x={case.cra_x_deg:g} deg")
        axis.set_xlabel("x (um)")
        axis.set_ylabel("z (um)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Focal-plane normalized |Ez|^2 at Si top, {wavelength_nm:g} nm")
    fig.savefig(output_dir / "cra_focal_maps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength-nm", type=float, default=550.0)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--after-source-time", type=float, default=35.0)
    parser.add_argument(
        "--cases",
        default="center:0:0:0,edge20_uncomp:20:0:1,edge20_comp:20:-0.18:1",
        help=(
            "Comma-separated cases as "
            "name:cra_deg:lens_shift_um[:field_norm[:aperture_shift_um[:active_shift_um]]]"
        ),
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
        default=ROOT / "runs" / "cra_response_lut",
    )
    args = parser.parse_args()

    geom = MicrolensArrayGeometry()
    wavelength_um = args.wavelength_nm / 1000
    frequency = 1 / wavelength_um
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(args.cases)
    si_n, si_k = load_si_nk(SI_NK_PATH, wavelength_um)

    print(f"Meep {mp.__version__}")
    print(
        f"CRA LUT at lambda={args.wavelength_nm:g} nm, "
        f"resolution={args.resolution} px/um, shield={args.shield_mode}"
    )
    print(f"Si n={si_n:.4f}, k={si_k:.5f}")

    rows = []
    maps = []
    for case in cases:
        print(f"running {case.name}: CRA={case.cra_x_deg:g} deg, lens_shift={case.lens_shift_x_um:g} um")
        row, focal_map = run_case(
            geom,
            case,
            wavelength_um,
            args.resolution,
            args.after_source_time,
            args.shield_mode,
        )
        rows.append(row)
        maps.append(focal_map)

    reference_response = rows[0]["collected_response_proxy"]
    for row in rows:
        row["normalized_response_to_first_case"] = (
            row["collected_response_proxy"] / reference_response
            if reference_response
            else float("nan")
        )

    csv_path = args.output_dir / "cra_response_lut.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "meep_version": mp.__version__,
        "model": "3D periodic CRA response unit cell",
        "wavelength_nm": args.wavelength_nm,
        "frequency_1_per_um": frequency,
        "resolution_px_per_um": args.resolution,
        "after_source_time": args.after_source_time,
        "geometry": asdict(geom),
        "shield_mode": args.shield_mode,
        "shield_mask_edge_width_um": geom.metal_edge_width,
        "silicon_nk_source": str(SI_NK_PATH),
        "silicon_n": si_n,
        "silicon_k": si_k,
        "cases": [asdict(case) for case in cases],
        "outputs": rows,
        "notes": [
            "CRA is modeled as an oblique plane wave in the x-y plane using Bloch k_point and a source phase ramp.",
            "The default edge20_comp case is illustrative; replace lens_shift_x_um and sign convention with the actual OCL/CFA shift design.",
            "Default shield_mode=off has no metal optical shield; edge/PDAF modes are explicit variants.",
            "collected_response_proxy = Si absorption estimate * focal-plane active-area fraction; it is not a TCAD charge-collection result.",
            "Use higher resolution and convergence sweeps before using this LUT quantitatively.",
        ],
    }
    json_path = args.output_dir / "cra_response_lut.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_focal_maps(args.output_dir, geom, cases, maps, args.wavelength_nm)

    print(json.dumps(rows, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
