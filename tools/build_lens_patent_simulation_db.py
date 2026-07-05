#!/usr/bin/env python3
"""Build a CameraE2E-ready lens patent simulation database.

The source Lens_Patent_DB files are normalized patent prescription staging data.
This builder preserves the source prescription tables and adds thin-lens /
paraxial proxy simulation summaries that CameraE2E can consume directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERAE2E_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LENS_DB_ROOT = Path.home() / "Lens_Patent_DB"
DEFAULT_OUT_DIR = CAMERAE2E_ROOT / "src" / "pyisetcam" / "data" / "lens_patents"
DEFAULT_SOURCE_VERSION = "expanded_v6"
WAVELENGTH_NM = 550.0
COC_DIAMETER_M = 10e-6


@dataclass(frozen=True)
class SimulationProfile:
    focal_length_mm: float | None
    focal_length_source: str
    f_number: float | None
    f_number_source: str
    aperture_diameter_mm: float | None
    aperture_source: str
    image_height_mm: float | None
    half_field_deg: float | None
    field_source: str
    simulation_status: str
    simulation_model: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-patent-db-root", type=Path, default=DEFAULT_LENS_DB_ROOT)
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--db-name", default="lens_patent_simulation_v6.sqlite")
    parser.add_argument("--include-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.lens_patent_db_root.expanduser().resolve()
    export_root = source_root / "exports" / f"prescriptions_{args.source_version}"
    data_root = source_root / "data" / args.source_version
    if not export_root.exists():
        raise FileNotFoundError(f"Prescription export directory not found: {export_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Normalized data directory not found: {data_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.out_dir / args.db_name
    tmp_path = db_path.with_suffix(".tmp.sqlite")
    if tmp_path.exists():
        tmp_path.unlink()

    manifest_rows = read_csv(export_root / "manifest.csv")
    readiness_by_id = {int(row["prescription_id"]): row for row in read_csv(export_root / "readiness.csv")}
    company_summary = read_csv(data_root / "company_summary.csv")

    con = sqlite3.connect(tmp_path)
    con.row_factory = sqlite3.Row
    create_schema(con)
    insert_companies(con, company_summary)

    build_info = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_version": args.source_version,
        "export_root": str(export_root),
        "include_blocked": bool(args.include_blocked),
    }
    con.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("build_info", json.dumps(build_info, sort_keys=True)),
    )

    counters: dict[str, int] = defaultdict(int)
    for manifest in manifest_rows:
        readiness = readiness_by_id.get(int(manifest["prescription_id"]), {})
        if manifest["readiness"] == "blocked" and not args.include_blocked:
            counters["skipped_blocked"] += 1
            continue
        insert_lens(con, export_root, manifest, readiness, counters)

    con.commit()
    create_indexes(con)
    summary = summarize(con)
    con.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("summary", json.dumps(summary, sort_keys=True)),
    )
    con.commit()
    con.close()

    tmp_path.replace(db_path)
    write_json(args.out_dir / "summary.json", summary | {"build": build_info})
    write_readme(args.out_dir / "README.md", summary, args.source_version)
    print(f"wrote {db_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE companies (
            company TEXT PRIMARY KEY,
            company_slug TEXT NOT NULL UNIQUE,
            publications INTEGER NOT NULL,
            prescriptions INTEGER NOT NULL,
            surfaces INTEGER NOT NULL,
            aspheres INTEGER NOT NULL,
            ready_staging INTEGER NOT NULL,
            ready_configured INTEGER NOT NULL,
            needs_variable_distances INTEGER NOT NULL,
            blocked INTEGER NOT NULL
        );

        CREATE TABLE lenses (
            lens_id TEXT PRIMARY KEY,
            prescription_id INTEGER NOT NULL,
            company TEXT NOT NULL,
            company_slug TEXT NOT NULL,
            publication_number TEXT NOT NULL,
            example_label TEXT NOT NULL,
            title TEXT NOT NULL,
            readiness TEXT NOT NULL,
            source_relative_dir TEXT NOT NULL,
            source_json_path TEXT NOT NULL,
            source_surfaces_csv TEXT NOT NULL,
            source_aspheres_csv TEXT NOT NULL,
            source_variable_distances_csv TEXT NOT NULL,
            source_configured_surfaces_dir TEXT NOT NULL,
            source_url TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            surface_count INTEGER NOT NULL,
            asphere_count INTEGER NOT NULL,
            variable_distance_count INTEGER NOT NULL,
            configured_surface_file_count INTEGER NOT NULL,
            notes TEXT NOT NULL
        );

        CREATE TABLE lens_surfaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lens_id TEXT NOT NULL,
            configuration TEXT NOT NULL,
            surface_order INTEGER NOT NULL,
            surface_label TEXT NOT NULL,
            surface_key TEXT NOT NULL,
            radius_mm REAL,
            thickness_mm REAL,
            nd REAL,
            vd REAL,
            material TEXT NOT NULL,
            effective_aperture_mm REAL,
            conic REAL,
            is_aspheric INTEGER NOT NULL,
            coefficients_json TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            FOREIGN KEY(lens_id) REFERENCES lenses(lens_id)
        );

        CREATE TABLE simulation_results (
            simulation_id TEXT PRIMARY KEY,
            lens_id TEXT NOT NULL,
            company TEXT NOT NULL,
            company_slug TEXT NOT NULL,
            publication_number TEXT NOT NULL,
            example_label TEXT NOT NULL,
            configuration TEXT NOT NULL,
            readiness TEXT NOT NULL,
            simulation_status TEXT NOT NULL,
            simulation_model TEXT NOT NULL,
            focal_length_mm REAL,
            focal_length_source TEXT NOT NULL,
            f_number REAL,
            f_number_source TEXT NOT NULL,
            aperture_diameter_mm REAL,
            aperture_source TEXT NOT NULL,
            image_height_mm REAL,
            half_field_deg REAL,
            field_of_view_deg REAL,
            field_source TEXT NOT NULL,
            airy_disk_diameter_um_550 REAL,
            diffraction_cutoff_lpmm_550 REAL,
            hyperfocal_m_coc_10um REAL,
            dof_m_at_1m_coc_10um REAL,
            surface_count INTEGER NOT NULL,
            asphere_count INTEGER NOT NULL,
            optics_json TEXT NOT NULL,
            source_surfaces_csv TEXT NOT NULL,
            notes TEXT NOT NULL,
            FOREIGN KEY(lens_id) REFERENCES lenses(lens_id)
        );
        """
    )


def create_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX idx_lenses_company ON lenses(company_slug, readiness);
        CREATE INDEX idx_surfaces_lens_configuration ON lens_surfaces(lens_id, configuration);
        CREATE INDEX idx_sim_company ON simulation_results(company_slug, simulation_status);
        CREATE INDEX idx_sim_readiness ON simulation_results(readiness, simulation_status);
        """
    )


def insert_companies(con: sqlite3.Connection, rows: list[dict[str, str]]) -> None:
    for row in rows:
        con.execute(
            """
            INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["company"],
                slugify(row["company"]),
                integer(row["publications"]),
                integer(row["prescriptions"]),
                integer(row["surfaces"]),
                integer(row["aspheres"]),
                integer(row["ready_staging"]),
                integer(row["ready_configured"]),
                integer(row["needs_variable_distances"]),
                integer(row["blocked"]),
            ),
        )


def insert_lens(
    con: sqlite3.Connection,
    export_root: Path,
    manifest: dict[str, str],
    readiness: dict[str, str],
    counters: dict[str, int],
) -> None:
    prescription_id = int(manifest["prescription_id"])
    lens_id = f"p{prescription_id:04d}"
    company = manifest["company"]
    company_slug = slugify(company)
    prescription_path = export_root / manifest["json_path"]
    prescription = json.loads(prescription_path.read_text(encoding="utf-8"))
    prescription_meta = prescription["prescription"]
    notes = readiness.get("notes", "")
    con.execute(
        """
        INSERT INTO lenses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lens_id,
            prescription_id,
            company,
            company_slug,
            manifest["publication_number"],
            manifest["example_label"],
            prescription_meta.get("title", ""),
            manifest["readiness"],
            manifest["relative_dir"],
            manifest["json_path"],
            manifest["surfaces_csv"],
            manifest["aspheres_csv"],
            manifest["variable_distances_csv"],
            manifest["configured_surfaces_dir"],
            prescription_meta.get("source_url", ""),
            prescription_meta.get("pdf_url", ""),
            integer(manifest["surface_count"]),
            integer(manifest["asphere_count"]),
            integer(manifest["variable_distance_count"]),
            integer(manifest["configured_surface_file_count"]),
            notes,
        ),
    )

    variable_rows = read_csv(export_root / manifest["variable_distances_csv"])
    variables_by_config = variables_by_configuration(variable_rows)
    surface_sets = configured_surface_sets(export_root, manifest)
    if not surface_sets:
        surface_sets = [("base", export_root / manifest["surfaces_csv"])]

    for configuration, surfaces_path in surface_sets:
        surfaces = read_csv(surfaces_path)
        config_variables = variables_for_configuration(variables_by_config, configuration)
        for surface in surfaces:
            insert_surface(con, lens_id, configuration, surface)
        profile = build_simulation_profile(surfaces, config_variables, manifest["readiness"])
        optics_json = build_optics_json(lens_id, manifest, configuration, profile)
        sim_id = f"{lens_id}:{slugify(configuration)}"
        con.execute(
            """
            INSERT INTO simulation_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sim_id,
                lens_id,
                company,
                company_slug,
                manifest["publication_number"],
                manifest["example_label"],
                configuration,
                manifest["readiness"],
                profile.simulation_status,
                profile.simulation_model,
                profile.focal_length_mm,
                profile.focal_length_source,
                profile.f_number,
                profile.f_number_source,
                profile.aperture_diameter_mm,
                profile.aperture_source,
                profile.image_height_mm,
                profile.half_field_deg,
                None if profile.half_field_deg is None else profile.half_field_deg * 2.0,
                profile.field_source,
                airy_disk_diameter_um(profile.f_number),
                diffraction_cutoff_lpmm(profile.f_number),
                hyperfocal_m(profile.focal_length_mm, profile.f_number),
                dof_at_1m(profile.focal_length_mm, profile.f_number),
                len(surfaces),
                integer(manifest["asphere_count"]),
                json.dumps(optics_json, sort_keys=True),
                str(surfaces_path.relative_to(export_root)),
                "; ".join(profile.notes),
            ),
        )
        counters[f"simulation_{profile.simulation_status}"] += 1


def insert_surface(
    con: sqlite3.Connection,
    lens_id: str,
    configuration: str,
    row: dict[str, str],
) -> None:
    coefficients = {
        key: numeric(row.get(key, ""))
        for key in ("k", "a4", "a6", "a8", "a10", "a12", "a14", "a16", "a18", "a20")
        if numeric(row.get(key, "")) is not None
    }
    con.execute(
        """
        INSERT INTO lens_surfaces(
            lens_id, configuration, surface_order, surface_label, surface_key, radius_mm,
            thickness_mm, nd, vd, material, effective_aperture_mm, conic, is_aspheric,
            coefficients_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lens_id,
            configuration,
            integer(row.get("surface_order", "0")),
            row.get("surface_label", ""),
            row.get("surface_key", ""),
            numeric(row.get("radius", "")),
            numeric(row.get("thickness", "")),
            numeric(row.get("nd", "")),
            numeric(row.get("vd", "")),
            row.get("material", ""),
            numeric(row.get("effective_aperture", "")),
            numeric(row.get("conic", "")),
            1 if str(row.get("is_aspheric", "")).strip() in {"1", "true", "True"} else 0,
            json.dumps(coefficients, sort_keys=True),
            json.dumps(row, sort_keys=True),
        ),
    )


def build_simulation_profile(
    surfaces: list[dict[str, str]],
    variables: dict[str, float],
    readiness: str,
) -> SimulationProfile:
    notes: list[str] = []
    focal_length = positive(variables.get("focal_length"))
    focal_source = "variable_distance:focal_length" if focal_length is not None else ""
    if focal_length is None:
        focal_length = paraxial_effective_focal_length_mm(surfaces)
        focal_source = "paraxial_matrix:surfaces" if focal_length is not None else ""
    if focal_length is None:
        notes.append("focal length unavailable")

    f_number = positive(variables.get("f_number"))
    f_number_source = "variable_distance:f_number" if f_number is not None else ""
    aperture = aperture_diameter_mm(surfaces)
    aperture_source = "derived:focal_length/f_number" if focal_length is not None and f_number is not None else ""
    if aperture is None and focal_length is not None and f_number is not None:
        aperture = focal_length / f_number
    elif aperture is not None:
        aperture_source = "surface:max_effective_aperture"
    if f_number is None and focal_length is not None and aperture is not None and aperture > 0:
        f_number = abs(focal_length) / aperture
        f_number_source = "proxy:focal_length/max_effective_aperture"
        notes.append("f-number inferred from max effective_aperture; verify aperture convention")
    if f_number is None:
        notes.append("f-number unavailable")

    image_height = positive(variables.get("image_height"))
    half_field = positive(variables.get("half_angle_of_view"))
    field_source = "variable_distance:half_angle_of_view" if half_field is not None else ""
    if half_field is None:
        half_field = positive(variables.get("half_angle_of_field"))
        field_source = "variable_distance:half_angle_of_field" if half_field is not None else ""
    if half_field is None and positive(variables.get("angle_of_view")) is not None:
        half_field = positive(variables.get("angle_of_view")) / 2.0
        field_source = "variable_distance:angle_of_view/2"
    if half_field is None and image_height is not None and focal_length is not None and focal_length > 0:
        half_field = math.degrees(math.atan(image_height / focal_length))
        field_source = "derived:atan(image_height/focal_length)"

    if readiness == "blocked":
        status = "blocked"
    elif focal_length is not None and f_number is not None:
        status = "camerae2e_ready"
    elif focal_length is not None:
        status = "partial"
    else:
        status = "metadata_only"
    model = "thin_lens_proxy" if status == "camerae2e_ready" else "metadata"
    if focal_source.startswith("paraxial"):
        model = "paraxial_proxy" if status != "metadata_only" else "metadata"
    return SimulationProfile(
        focal_length_mm=focal_length,
        focal_length_source=focal_source,
        f_number=f_number,
        f_number_source=f_number_source,
        aperture_diameter_mm=aperture,
        aperture_source=aperture_source,
        image_height_mm=image_height,
        half_field_deg=half_field,
        field_source=field_source,
        simulation_status=status,
        simulation_model=model,
        notes=notes,
    )


def build_optics_json(
    lens_id: str,
    manifest: dict[str, str],
    configuration: str,
    profile: SimulationProfile,
) -> dict[str, Any]:
    optics: dict[str, Any] = {
        "name": f"{manifest['company']} {manifest['publication_number']} {manifest['example_label'] or lens_id} {configuration}".strip(),
        "model": "diffractionlimited",
        "compute_method": "opticsotf",
        "offaxis_method": "cos4th",
        "lens_patent": {
            "lens_id": lens_id,
            "publication_number": manifest["publication_number"],
            "company": manifest["company"],
            "example_label": manifest["example_label"],
            "configuration": configuration,
            "readiness": manifest["readiness"],
            "simulation_status": profile.simulation_status,
            "focal_length_source": profile.focal_length_source,
            "f_number_source": profile.f_number_source,
        },
    }
    if profile.focal_length_mm is not None:
        optics["focal_length_m"] = profile.focal_length_mm / 1000.0
        optics["nominal_focal_length_m"] = profile.focal_length_mm / 1000.0
    if profile.f_number is not None:
        optics["f_number"] = profile.f_number
    if profile.half_field_deg is not None:
        optics["max_fov_deg"] = profile.half_field_deg * 2.0
    return optics


def configured_surface_sets(export_root: Path, manifest: dict[str, str]) -> list[tuple[str, Path]]:
    configured_dir = manifest.get("configured_surfaces_dir", "")
    if not configured_dir:
        return []
    directory = export_root / configured_dir
    if not directory.exists():
        return []
    return [(path.stem, path) for path in sorted(directory.glob("*.csv"))]


def variables_by_configuration(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = numeric(row.get("value", ""))
        if value is None:
            continue
        config = normalize_configuration(row.get("configuration", "base"))
        result[config][row.get("parameter", "")] = value
    return result


def variables_for_configuration(
    variables: dict[str, dict[str, float]],
    configuration: str,
) -> dict[str, float]:
    normalized = normalize_configuration(configuration)
    if normalized in variables:
        return variables[normalized]
    simplified = normalized.replace("wideangle", "wide").replace("telephoto", "tele")
    for key, values in variables.items():
        candidate = key.replace("wideangle", "wide").replace("telephoto", "tele")
        if candidate == simplified or simplified in candidate or candidate in simplified:
            return values
    if len(variables) == 1:
        return next(iter(variables.values()))
    return {}


def paraxial_effective_focal_length_mm(surfaces: list[dict[str, str]]) -> float | None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    n_before = 1.0
    used_power = False
    for row in surfaces:
        radius = numeric(row.get("radius", ""))
        n_after = numeric(row.get("nd", "")) or 1.0
        if radius is not None and abs(radius) > 1e-12 and math.isfinite(radius):
            phi = (n_after - n_before) / radius
            refraction = [[1.0, 0.0], [-phi / n_after, n_before / n_after]]
            matrix = matmul(refraction, matrix)
            used_power = True
        thickness = numeric(row.get("thickness", ""))
        if thickness is not None and math.isfinite(thickness):
            translation = [[1.0, thickness], [0.0, 1.0]]
            matrix = matmul(translation, matrix)
        n_before = n_after
    c = matrix[1][0]
    if not used_power or abs(c) < 1e-12:
        return None
    efl = -1.0 / c
    if not math.isfinite(efl) or abs(efl) <= 0.0 or abs(efl) > 1e6:
        return None
    return abs(efl)


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def aperture_diameter_mm(surfaces: list[dict[str, str]]) -> float | None:
    values = [positive(numeric(row.get("effective_aperture", ""))) for row in surfaces]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    aperture = max(finite)
    # Reject obvious theta_gF-style columns or non-aperture scalars for large lens systems.
    if aperture < 0.8:
        return None
    return aperture


def airy_disk_diameter_um(f_number: float | None) -> float | None:
    if f_number is None or f_number <= 0:
        return None
    return 2.44 * f_number * (WAVELENGTH_NM * 1e-3)


def diffraction_cutoff_lpmm(f_number: float | None) -> float | None:
    if f_number is None or f_number <= 0:
        return None
    return 1.0 / ((WAVELENGTH_NM * 1e-6) * f_number)


def hyperfocal_m(focal_length_mm: float | None, f_number: float | None) -> float | None:
    if focal_length_mm is None or f_number is None or focal_length_mm <= 0 or f_number <= 0:
        return None
    focal_length_m = focal_length_mm / 1000.0
    return (focal_length_m * focal_length_m) / (f_number * COC_DIAMETER_M) + focal_length_m


def dof_at_1m(focal_length_mm: float | None, f_number: float | None) -> float | None:
    if focal_length_mm is None or f_number is None or focal_length_mm <= 0 or f_number <= 0:
        return None
    focal_length_m = focal_length_mm / 1000.0
    return (2.0 * f_number * COC_DIAMETER_M) / max(focal_length_m * focal_length_m, 1e-30)


def summarize(con: sqlite3.Connection) -> dict[str, Any]:
    def one(query: str) -> Any:
        return con.execute(query).fetchone()[0]

    return {
        "companies": one("SELECT count(*) FROM companies"),
        "lenses": one("SELECT count(*) FROM lenses"),
        "surfaces": one("SELECT count(*) FROM lens_surfaces"),
        "simulation_results": one("SELECT count(*) FROM simulation_results"),
        "status_counts": {
            row["simulation_status"]: row["count"]
            for row in con.execute(
                "SELECT simulation_status, count(*) AS count FROM simulation_results GROUP BY simulation_status"
            )
        },
        "readiness_counts": {
            row["readiness"]: row["count"]
            for row in con.execute(
                "SELECT readiness, count(*) AS count FROM lenses GROUP BY readiness"
            )
        },
        "camerae2e_ready_by_company": {
            row["company"]: row["count"]
            for row in con.execute(
                """
                SELECT company, count(*) AS count
                FROM simulation_results
                WHERE simulation_status = 'camerae2e_ready'
                GROUP BY company
                ORDER BY company
                """
            )
        },
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any], source_version: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Lens Patent Simulation DB",
                "",
                f"Generated from Lens_Patent_DB `{source_version}` exports.",
                "",
                "Files:",
                "",
                "- `lens_patent_simulation_v6.sqlite`: CameraE2E-ready prescription and simulation result DB.",
                "- `summary.json`: generation summary and status counts.",
                "- `companies/`: company-specific SQLite DB subsets and manifest.",
                "- `raytrace_psf/`: optional RayOptics geometric PSF grid `.npz` files and manifest.",
                "",
                "Regenerate:",
                "",
                "```bash",
                "python tools/build_lens_patent_simulation_db.py",
                "python tools/export_lens_patent_company_sets.py --overwrite",
                "```",
                "",
                "Use:",
                "",
                "```python",
                "from pyisetcam import oi_create, oi_set",
                "from pyisetcam.lens_patents import lens_patent_search, lens_patent_optics",
                "",
                "row = lens_patent_search(company='Canon', require_camerae2e=True, limit=1)[0]",
                "optics = lens_patent_optics(row['simulation_id'])",
                "oi = oi_set(oi_create(), 'optics', optics)",
                "```",
                "",
                "Company-specific DB:",
                "",
                "```python",
                "from pyisetcam.lens_patents import lens_patent_company_db_path, lens_patent_search",
                "",
                "canon_db = lens_patent_company_db_path('Canon')",
                "canon_rows = lens_patent_search(db_path=canon_db, require_camerae2e=True)",
                "```",
                "",
                "Raytrace PSF grids:",
                "",
                "```bash",
                "/Users/seongcheoljeong/RayOptics/rayoptics-env/bin/python3.12 \\",
                "  tools/build_lens_patent_raytrace_psf_grid.py --limit 10000 --overwrite",
                "```",
                "",
                "Generated PSF grids are stored under `raytrace_psf/` as compressed `.npz` files "
                "with a `manifest.json`. Load them with `lens_patent_raytrace_optics(simulation_id)`. "
                "The PSF manifest records both generated and failed rows.",
                "",
                "Main API:",
                "",
                "- `lens_patent_db_summary()`",
                "- `lens_patent_companies()`",
                "- `lens_patent_company_sets_manifest()`",
                "- `lens_patent_company_db_path(company)`",
                "- `lens_patent_search(company=None, readiness=None, require_camerae2e=False, limit=None)`",
                "- `lens_patent_get(simulation_id)`",
                "- `lens_patent_surfaces(lens_id, configuration=None)`",
                "- `lens_patent_optics(simulation_id, default_f_number=None)`",
                "- `lens_patent_raytrace_psf_search(company=None, status=None)`",
                "- `lens_patent_raytrace_optics(simulation_id)`",
                "",
                "Caveat: these are patent-disclosed examples, not confirmed production lenses. "
                "Rows marked `paraxial_proxy` or with `proxy:` sources should be verified before "
                "being treated as ray-trace-accurate designs. This DB is a CameraE2E optics/proxy "
                "catalog, not a full Zemax/RayOptics sequential trace result set.",
                "",
                f"Summary: `{json.dumps(summary, sort_keys=True)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def positive(value: float | None) -> float | None:
    if value is None or value <= 0.0 or not math.isfinite(value):
        return None
    return value


def integer(value: Any) -> int:
    parsed = numeric(value)
    return 0 if parsed is None else int(parsed)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def normalize_configuration(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


if __name__ == "__main__":
    raise SystemExit(main())
