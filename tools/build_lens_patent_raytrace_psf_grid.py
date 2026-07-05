#!/usr/bin/env python3
"""Build RayOptics-derived PSF grids for Lens_Patent_DB rows.

Run this with the RayOptics Python environment, for example:

    /Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \
        tools/build_lens_patent_raytrace_psf_grid.py --limit 1

The output is intentionally separate from the SQLite catalog because PSF arrays
can grow quickly. Each generated `.npz` file contains CameraE2E-compatible
raytrace arrays plus enough metadata for `pyisetcam.lens_patents` to load them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rayoptics.elem import profiles
from rayoptics.environment import FieldSpec, OpticalModel, PupilSpec, WvlSpec
from rayoptics.raytr.trace import trace_base

CAMERAE2E_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_DB = (
    CAMERAE2E_ROOT
    / "src"
    / "pyisetcam"
    / "data"
    / "lens_patents"
    / "lens_patent_simulation_v6.sqlite"
)
DEFAULT_LENS_DB_ROOT = Path.home() / "Lens_Patent_DB"
DEFAULT_EXPORT_ROOT = DEFAULT_LENS_DB_ROOT / "exports" / "prescriptions_expanded_v6"
DEFAULT_OUT_DIR = CAMERAE2E_ROOT / "src" / "pyisetcam" / "data" / "lens_patents" / "raytrace_psf"

PSF_PRESETS: dict[str, dict[str, Any]] = {
    "debug": {
        "field_count": 3,
        "wavelengths": "550",
        "pupil_samples": 13,
        "psf_size": 64,
    },
    "standard": {
        "field_count": 5,
        "wavelengths": "450,550,650",
        "pupil_samples": 49,
        "psf_size": 384,
    },
    "production": {
        "field_count": 5,
        "wavelengths": "450,550,650",
        "pupil_samples": 65,
        "psf_size": 512,
    },
    "golden": {
        "field_count": 7,
        "wavelengths": "450,500,550,600,650",
        "pupil_samples": 97,
        "psf_size": 768,
    },
}


@dataclass(frozen=True)
class Candidate:
    simulation_id: str
    lens_id: str
    company: str
    publication_number: str
    example_label: str
    configuration: str
    f_number: float
    focal_length_mm: float
    image_height_mm: float | None
    half_field_deg: float | None
    source_surfaces_csv: str
    optics_json: dict[str, Any]


@dataclass
class TraceBundle:
    psf_function: np.ndarray
    field_height_mm: np.ndarray
    wavelength_nm: np.ndarray
    sample_spacing_mm: np.ndarray
    geometry_function: np.ndarray
    relative_illumination_function: np.ndarray
    success_counts: np.ndarray
    requested_counts: np.ndarray
    chief_points_mm: np.ndarray
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PSF_PRESETS), default="debug")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--camerae2e-db", type=Path, default=DEFAULT_CAMERA_DB)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--simulation-id")
    parser.add_argument("--company")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--field-count", type=int)
    parser.add_argument("--wavelengths")
    parser.add_argument("--pupil-samples", type=int)
    parser.add_argument("--psf-size", type=int)
    parser.add_argument("--sample-spacing-um", type=float)
    parser.add_argument("--min-success-fraction", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-manifest", action="store_true")
    parser.add_argument("--include-non-ready", action="store_true")
    args = parser.parse_args()
    if not args.list_presets:
        apply_preset_defaults(args)
    return args


def main() -> int:
    args = parse_args()
    if args.list_presets:
        print(json.dumps(PSF_PRESETS, indent=2, sort_keys=True))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args)
    manifest_path = args.out_dir / "manifest.json"
    manifest_rows_by_id = {} if args.reset_manifest else load_existing_manifest_rows(manifest_path)
    build_settings = effective_build_settings(args)
    for candidate in candidates:
        output_path = args.out_dir / f"{safe_id(candidate.simulation_id)}.npz"
        row: dict[str, Any] = {
            "simulation_id": candidate.simulation_id,
            "lens_id": candidate.lens_id,
            "company": candidate.company,
            "publication_number": candidate.publication_number,
            "example_label": candidate.example_label,
            "configuration": candidate.configuration,
            "file": output_path.name,
            "status": "pending",
            "preset": args.preset,
            "built_at": datetime.now(UTC).isoformat(),
        }
        if output_path.exists() and not args.overwrite:
            row["status"] = "exists"
            manifest_rows_by_id[candidate.simulation_id] = merge_existing_row(
                manifest_rows_by_id.get(candidate.simulation_id),
                row,
            )
            continue

        try:
            surfaces = read_csv(args.export_root / candidate.source_surfaces_csv)
            bundle = build_raytrace_psf_bundle(
                candidate,
                surfaces,
                wavelengths_nm=parse_wavelengths(args.wavelengths),
                field_count=args.field_count,
                pupil_samples=args.pupil_samples,
                psf_size=args.psf_size,
                sample_spacing_um=args.sample_spacing_um,
                min_success_fraction=args.min_success_fraction,
            )
            optics_json = build_raytrace_optics_json(candidate, bundle, output_path.name)
            np.savez_compressed(
                output_path,
                psf_function=bundle.psf_function.astype(np.float32),
                field_height_mm=bundle.field_height_mm.astype(np.float64),
                wavelength_nm=bundle.wavelength_nm.astype(np.float64),
                sample_spacing_mm=bundle.sample_spacing_mm.astype(np.float64),
                geometry_function=bundle.geometry_function.astype(np.float64),
                relative_illumination_function=bundle.relative_illumination_function.astype(np.float64),
                success_counts=bundle.success_counts.astype(np.int32),
                requested_counts=bundle.requested_counts.astype(np.int32),
                chief_points_mm=bundle.chief_points_mm.astype(np.float64),
                optics_json=np.array(json.dumps(optics_json, sort_keys=True)),
                build_settings_json=np.array(json.dumps(build_settings, sort_keys=True)),
            )
            row.update(
                {
                    "status": "generated",
                    "psf_shape": list(bundle.psf_function.shape),
                    "sample_spacing_mm": bundle.sample_spacing_mm.tolist(),
                    "field_height_mm": bundle.field_height_mm.tolist(),
                    "wavelength_nm": bundle.wavelength_nm.tolist(),
                    "success_min": int(np.min(bundle.success_counts)),
                    "success_max": int(np.max(bundle.success_counts)),
                    "notes": bundle.notes,
                }
            )
        except Exception as exc:  # noqa: BLE001 - manifest should record all per-lens failures.
            row.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        manifest_rows_by_id[candidate.simulation_id] = row

    manifest_rows = sorted(
        manifest_rows_by_id.values(),
        key=lambda row: (str(row.get("company", "")), str(row.get("simulation_id", ""))),
    )
    write_manifest(manifest_path, args, manifest_rows)
    print(json.dumps(summarize_manifest(manifest_rows), indent=2, sort_keys=True))
    return 0


def load_existing_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rows = payload.get("rows", [])
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        simulation_id = row.get("simulation_id")
        if simulation_id:
            result[str(simulation_id)] = dict(row)
    return result


def merge_existing_row(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return incoming
    if existing.get("status") == "generated":
        return existing
    merged = dict(existing)
    merged.update(incoming)
    return merged


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = PSF_PRESETS[str(args.preset)]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if int(args.field_count) < 1:
        raise ValueError("--field-count must be positive.")
    if int(args.pupil_samples) < 1:
        raise ValueError("--pupil-samples must be positive.")
    if int(args.psf_size) < 8:
        raise ValueError("--psf-size must be at least 8.")
    if args.sample_spacing_um is not None and float(args.sample_spacing_um) <= 0.0:
        raise ValueError("--sample-spacing-um must be positive when supplied.")
    parse_wavelengths(str(args.wavelengths))


def effective_build_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "preset": str(args.preset),
        "field_count": int(args.field_count),
        "wavelengths": parse_wavelengths(str(args.wavelengths)).tolist(),
        "pupil_samples": int(args.pupil_samples),
        "psf_size": int(args.psf_size),
        "sample_spacing_um": args.sample_spacing_um,
        "min_success_fraction": float(args.min_success_fraction),
        "model": "rayoptics_geometric_histogram",
    }


def load_candidates(args: argparse.Namespace) -> list[Candidate]:
    con = sqlite3.connect(args.camerae2e_db)
    con.row_factory = sqlite3.Row
    clauses = []
    params: list[Any] = []
    if not args.include_non_ready:
        clauses.append("simulation_status = 'camerae2e_ready'")
    if args.simulation_id:
        clauses.append("simulation_id = ?")
        params.append(args.simulation_id)
    if args.company:
        clauses.append("lower(company) = lower(?)")
        params.append(args.company)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = "" if args.simulation_id else "LIMIT ?"
    if not args.simulation_id:
        params.append(max(int(args.limit), 1))
    rows = con.execute(
        f"""
        SELECT *
        FROM simulation_results
        {where}
        ORDER BY company, publication_number, example_label, configuration
        {limit_sql}
        """,
        params,
    ).fetchall()
    con.close()
    candidates: list[Candidate] = []
    for row in rows:
        if row["f_number"] is None or row["focal_length_mm"] is None:
            continue
        candidates.append(
            Candidate(
                simulation_id=str(row["simulation_id"]),
                lens_id=str(row["lens_id"]),
                company=str(row["company"]),
                publication_number=str(row["publication_number"]),
                example_label=str(row["example_label"]),
                configuration=str(row["configuration"]),
                f_number=float(row["f_number"]),
                focal_length_mm=float(row["focal_length_mm"]),
                image_height_mm=none_or_float(row["image_height_mm"]),
                half_field_deg=none_or_float(row["half_field_deg"]),
                source_surfaces_csv=str(row["source_surfaces_csv"]),
                optics_json=json.loads(str(row["optics_json"])),
            )
        )
    return candidates


def build_raytrace_psf_bundle(
    candidate: Candidate,
    surfaces: list[dict[str, str]],
    *,
    wavelengths_nm: np.ndarray,
    field_count: int,
    pupil_samples: int,
    psf_size: int,
    sample_spacing_um: float | None,
    min_success_fraction: float,
) -> TraceBundle:
    notes: list[str] = []
    model, field_height_mm = build_rayoptics_model(candidate, surfaces, field_count, wavelengths_nm)
    pupil_points = pupil_disk_samples(pupil_samples)
    if pupil_points.size == 0:
        raise ValueError("No pupil samples generated.")

    field_spec = model["optical_spec"]["fov"]
    fields = field_spec.fields
    n_field = len(fields)
    n_wave = wavelengths_nm.size
    requested_counts = np.full((n_field, n_wave), pupil_points.shape[0], dtype=int)
    success_counts = np.zeros((n_field, n_wave), dtype=int)
    chief_points = np.zeros((n_field, n_wave, 2), dtype=float)
    geometry = np.zeros((n_field, n_wave), dtype=float)
    rel_illum = np.zeros((n_field, n_wave), dtype=float)
    point_sets: list[list[np.ndarray]] = [
        [np.empty((0, 2), dtype=float) for _ in range(n_wave)]
        for _ in range(n_field)
    ]

    for field_index, field in enumerate(fields):
        for wave_index, wavelength_nm in enumerate(wavelengths_nm):
            chief_point = trace_image_point(model, [0.0, 0.0], field, float(wavelength_nm))
            chief_points[field_index, wave_index, :] = chief_point
            geometry[field_index, wave_index] = float(abs(chief_point[1]))
            points: list[list[float]] = []
            for pupil in pupil_points:
                try:
                    point = trace_image_point(model, pupil, field, float(wavelength_nm))
                except Exception:  # noqa: BLE001 - failed rays are represented by success_counts.
                    continue
                points.append([point[0] - chief_point[0], point[1] - chief_point[1]])
            if points:
                point_sets[field_index][wave_index] = np.asarray(points, dtype=float)
            success_counts[field_index, wave_index] = len(points)

    min_fraction = float(np.min(success_counts / np.maximum(requested_counts, 1)))
    if min_fraction < float(min_success_fraction):
        raise ValueError(
            f"Insufficient traced rays: min success fraction {min_fraction:.3f} "
            f"< {float(min_success_fraction):.3f}"
        )

    spacing_mm = choose_sample_spacing(point_sets, psf_size, sample_spacing_um)
    psf = np.zeros((psf_size, psf_size, n_field, n_wave), dtype=float)
    for field_index in range(n_field):
        for wave_index in range(n_wave):
            psf[:, :, field_index, wave_index] = histogram_psf(
                point_sets[field_index][wave_index],
                psf_size,
                spacing_mm,
            )

    center_counts = np.maximum(success_counts[0:1, :], 1)
    rel_illum = success_counts / center_counts
    notes.append("RayOptics geometric ray histogram PSF; diffraction is not included.")
    if any(row.get("is_aspheric", "").strip() in {"1", "true", "True"} for row in surfaces):
        notes.append("EvenPolynomial asphere coefficients were applied where parsed.")
    return TraceBundle(
        psf_function=psf,
        field_height_mm=field_height_mm,
        wavelength_nm=wavelengths_nm,
        sample_spacing_mm=np.array([spacing_mm, spacing_mm], dtype=float),
        geometry_function=geometry,
        relative_illumination_function=rel_illum,
        success_counts=success_counts,
        requested_counts=requested_counts,
        chief_points_mm=chief_points,
        notes=notes,
    )


def build_rayoptics_model(
    candidate: Candidate,
    surfaces: list[dict[str, str]],
    field_count: int,
    wavelengths_nm: np.ndarray,
) -> tuple[OpticalModel, np.ndarray]:
    max_field_height = candidate.image_height_mm
    if max_field_height is None and candidate.half_field_deg is not None:
        max_field_height = candidate.focal_length_mm * math.tan(
            math.radians(candidate.half_field_deg)
        )
    if max_field_height is None or max_field_height <= 0:
        max_field_height = max(candidate.focal_length_mm * 0.1, 1.0)
    field_height_mm = np.linspace(
        0.0,
        float(max_field_height),
        max(int(field_count), 1),
        dtype=float,
    )
    relative_fields = field_height_mm / max(float(max_field_height), 1e-12)

    model = OpticalModel()
    model.radius_mode = True
    seq_model = model["seq_model"]
    optical_spec = model["optical_spec"]
    seq_model.do_apertures = False
    seq_model.gaps[0].thi = 1.0e10
    optical_spec["pupil"] = PupilSpec(
        optical_spec,
        key=["image", "f/#"],
        value=float(candidate.f_number),
    )
    optical_spec["fov"] = FieldSpec(
        optical_spec,
        key=["image", "height"],
        value=float(max_field_height),
        flds=relative_fields.tolist(),
        is_relative=True,
    )
    optical_spec["wvls"] = WvlSpec([(float(w), 1.0) for w in wavelengths_nm], ref_wl=0)

    stop_surface = None
    min_aperture = math.inf
    for row in surfaces:
        surface_kind = str(row.get("surface_kind", "")).strip().lower()
        if surface_kind == "object":
            continue
        surf_data = surface_data(row)
        semi_diameter = positive(numeric(row.get("effective_aperture"))) 
        semi_diameter = None if semi_diameter is None else semi_diameter / 2.0
        seq_model.add_surface(surf_data, sd=semi_diameter)
        surface_index = int(seq_model.cur_surface)
        apply_asphere(seq_model.ifcs[surface_index], row)
        if surface_kind == "stop":
            stop_surface = surface_index
        elif surface_kind != "image" and semi_diameter is not None and semi_diameter < min_aperture:
            min_aperture = semi_diameter
            if stop_surface is None:
                stop_surface = surface_index

    if stop_surface is not None:
        seq_model.cur_surface = int(stop_surface)
        seq_model.set_stop()
    model.update_model()
    return model, field_height_mm


def surface_data(row: dict[str, str]) -> list[Any]:
    radius = numeric(row.get("radius"))
    thickness = numeric(row.get("thickness"))
    data: list[Any] = [0.0 if radius is None else radius, 0.0 if thickness is None else thickness]
    nd = positive(numeric(row.get("nd")))
    vd = positive(numeric(row.get("vd")))
    if nd is not None:
        data.extend([nd, 50.0 if vd is None else vd])
    return data


def apply_asphere(ifc: Any, row: dict[str, str]) -> None:
    if str(row.get("is_aspheric", "")).strip() not in {"1", "true", "True"}:
        conic = numeric(row.get("conic"))
        if conic is not None and hasattr(ifc, "profile"):
            ifc.profile = profiles.mutate_profile(ifc.profile, "Conic")
            ifc.profile.cc = conic
        return

    ifc.profile = profiles.mutate_profile(ifc.profile, "EvenPolynomial")
    conic = numeric(row.get("conic"))
    if conic is not None:
        ifc.profile.cc = conic
    coefficient_map = {
        "a4": 1,
        "a6": 2,
        "a8": 3,
        "a10": 4,
        "a12": 5,
        "a14": 6,
        "a16": 7,
        "a18": 8,
        "a20": 9,
    }
    if len(ifc.profile.coefs) < 10:
        ifc.profile.coefs = list(ifc.profile.coefs) + [0.0] * (10 - len(ifc.profile.coefs))
    for key, index in coefficient_map.items():
        value = numeric(row.get(key))
        if value is not None:
            ifc.profile.coefs[index] = value


def trace_image_point(
    model: OpticalModel,
    pupil: Any,
    field: Any,
    wavelength_nm: float,
) -> np.ndarray:
    ray, _op_delta, _wavelength = trace_base(model, pupil, field, wavelength_nm)
    point = np.asarray(ray[-1][0], dtype=float).reshape(3)
    return point[:2].copy()


def pupil_disk_samples(samples: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, max(int(samples), 1), dtype=float)
    points = []
    for y in axis:
        for x in axis:
            if (x * x + y * y) <= 1.0 + 1e-12:
                points.append([x, y])
    return np.asarray(points, dtype=float)


def choose_sample_spacing(
    point_sets: list[list[np.ndarray]],
    psf_size: int,
    sample_spacing_um: float | None,
) -> float:
    if sample_spacing_um is not None:
        return float(sample_spacing_um) / 1000.0
    max_extent = 0.0
    for row in point_sets:
        for points in row:
            if points.size:
                max_extent = max(max_extent, float(np.max(np.abs(points))))
    if max_extent <= 0.0:
        return 0.001
    return max_extent / max((psf_size / 2.0) * 0.85, 1.0)


def histogram_psf(points: np.ndarray, psf_size: int, sample_spacing_mm: float) -> np.ndarray:
    kernel = np.zeros((psf_size, psf_size), dtype=float)
    if points.size == 0:
        kernel[psf_size // 2, psf_size // 2] = 1.0
        return kernel
    center = (psf_size - 1) / 2.0
    cols = np.rint((points[:, 0] / sample_spacing_mm) + center).astype(int)
    rows = np.rint((points[:, 1] / sample_spacing_mm) + center).astype(int)
    valid = (rows >= 0) & (rows < psf_size) & (cols >= 0) & (cols < psf_size)
    for row, col in zip(rows[valid], cols[valid], strict=False):
        kernel[row, col] += 1.0
    total = float(np.sum(kernel))
    if total <= 0.0:
        kernel[psf_size // 2, psf_size // 2] = 1.0
    else:
        kernel /= total
    return kernel


def build_raytrace_optics_json(
    candidate: Candidate,
    bundle: TraceBundle,
    file_name: str,
) -> dict[str, Any]:
    base = dict(candidate.optics_json)
    base.update(
        {
            "model": "raytrace",
            "compute_method": "opticspsf",
            "offaxis_method": "skip",
            "focal_length_m": float(candidate.focal_length_mm) / 1000.0,
            "nominal_focal_length_m": float(candidate.focal_length_mm) / 1000.0,
            "f_number": float(candidate.f_number),
            "transmittance": {
                "wave": bundle.wavelength_nm.tolist(),
                "scale": [1.0] * int(bundle.wavelength_nm.size),
            },
            "raytrace": {
                "program": "RayOptics",
                "lens_file": candidate.source_surfaces_csv,
                "reference_wavelength_nm": float(bundle.wavelength_nm[0]),
                "object_distance_m": math.inf,
                "magnification": 0.0,
                "f_number": float(candidate.f_number),
                "effective_focal_length_m": float(candidate.focal_length_mm) / 1000.0,
                "effective_f_number": float(candidate.f_number),
                "max_fov_deg": (
                    None
                    if candidate.half_field_deg is None
                    else float(candidate.half_field_deg) * 2.0
                ),
                "geometry": {
                    "function": "__npz__:geometry_function",
                    "field_height_mm": bundle.field_height_mm.tolist(),
                    "wavelength_nm": bundle.wavelength_nm.tolist(),
                },
                "relative_illumination": {
                    "function": "__npz__:relative_illumination_function",
                    "field_height_mm": bundle.field_height_mm.tolist(),
                    "wavelength_nm": bundle.wavelength_nm.tolist(),
                },
                "psf": {
                    "function": "__npz__:psf_function",
                    "field_height_mm": bundle.field_height_mm.tolist(),
                    "wavelength_nm": bundle.wavelength_nm.tolist(),
                    "sample_spacing_mm": bundle.sample_spacing_mm.tolist(),
                },
                "computation": {
                    "psf_spacing_m": float(bundle.sample_spacing_mm[0]) / 1000.0,
                },
                "blocks_per_field_height": 4,
                "name": (
                    f"{candidate.company} {candidate.publication_number} "
                    f"{candidate.configuration}"
                ),
            },
            "lens_patent_raytrace_psf": {
                "file": file_name,
                "simulation_id": candidate.simulation_id,
                "model": "rayoptics_geometric_histogram",
                "notes": bundle.notes,
            },
        }
    )
    return base


def write_manifest(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "camerae2e_lens_patent_raytrace_psf_manifest_v1",
        "built_at": datetime.now(UTC).isoformat(),
        "camerae2e_db": str(args.camerae2e_db),
        "export_root": str(args.export_root),
        "settings": effective_build_settings(args),
        "available_presets": PSF_PRESETS,
        "rows": rows,
        "summary": summarize_manifest(rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(rows)}
    by_company: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row.get("status", "unknown"))
        summary[key] = int(summary.get(key, 0)) + 1
        company = str(row.get("company", "unknown") or "unknown")
        company_summary = by_company.setdefault(company, {"total": 0})
        company_summary["total"] += 1
        company_summary[key] = int(company_summary.get(key, 0)) + 1
    summary["by_company"] = by_company
    return summary


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_wavelengths(value: str) -> np.ndarray:
    wavelengths = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not wavelengths:
        raise ValueError("At least one wavelength is required.")
    return np.asarray(wavelengths, dtype=float)


def numeric(value: Any) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    lowered = text.lower().replace("∞", "infinity")
    if lowered in {"inf", "+inf", "infinite", "infinity", "-inf", "-infinity"}:
        return None
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        parsed = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def positive(value: float | None) -> float | None:
    if value is None or value <= 0.0 or not math.isfinite(value):
        return None
    return value


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = numeric(value)
    return None if parsed is None else float(parsed)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    raise SystemExit(main())
