"""Unified CameraE2E database and parameter catalog.

The catalog is a runtime index over data sources that feed CameraE2E blocks:
lens prescriptions/PSFs, sensor optical/electrical LUTs, HW ISP profiles,
perception model profiles, upstream ISETCam assets, and parity baselines.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assets import DEFAULT_UPSTREAM_SHA
from .fdtd_sensor import (
    fdtd_sensor_default_lut_path,
    fdtd_sensor_lut_load,
    fdtd_sensor_lut_summary,
    fdtd_sensor_physics_validate,
)
from .hwisp_db import hw_isp_profile_names
from .lens_patents import (
    lens_patent_camerae2e_manifest,
    lens_patent_db_summary,
    lens_patent_default_data_dir,
    lens_patent_default_db_path,
    lens_patent_raytrace_psf_manifest,
)
from .task_perception import task_model_profile_names
from .tcad_sensor import tcad_sensor_db_load, tcad_sensor_default_paths, tcad_sensor_validate

_DEFAULT_CAMERA_DB_ROOT_ENV = "PYISETCAM_CAMERA_DB_ROOT"


@dataclass(frozen=True)
class CameraE2EDBEntry:
    """One CameraE2E database/LUT/model-profile catalog entry."""

    name: str
    family: str
    role: str
    status: str
    path: Path | None
    description: str
    parameter_hint: str
    parameters: dict[str, Any] = field(default_factory=dict)
    env_vars: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "camerae2e_db_entry_v2"
    artifact_id: str | None = None
    readiness_tier: str = "available"
    provenance: dict[str, Any] = field(default_factory=dict)
    source_hash: str | None = None
    dependencies: tuple[str, ...] = ()
    validation_gates: tuple[str, ...] = ()
    refresh_command: str | None = None
    stale_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "artifact_id": self.artifact_id or self.name,
            "family": self.family,
            "role": self.role,
            "status": self.status,
            "readiness_tier": self.readiness_tier,
            "path": None if self.path is None else str(self.path),
            "description": self.description,
            "parameter_hint": self.parameter_hint,
            "parameters": _jsonable(self.parameters),
            "env_vars": list(self.env_vars),
            "tags": list(self.tags),
            "metadata": _jsonable(self.metadata),
            "provenance": _jsonable(self.provenance),
            "source_hash": self.source_hash,
            "dependencies": list(self.dependencies),
            "validation_gates": list(self.validation_gates),
            "refresh_command": self.refresh_command,
            "stale_reason": self.stale_reason,
        }


def camerae2e_db_catalog(*, include_missing: bool = True) -> list[CameraE2EDBEntry]:
    """Return all known CameraE2E data sources as normalized catalog entries."""

    entries: list[CameraE2EDBEntry] = []
    entries.extend(_lens_entries())
    entries.extend(_sensor_entries())
    entries.extend(_hwisp_entries())
    entries.extend(_perception_entries())
    entries.extend(_upstream_entries())
    entries.extend(_parity_entries())
    if not include_missing:
        entries = [entry for entry in entries if entry.status != "missing"]
    return sorted(entries, key=lambda item: (item.family, item.role, item.name))


def camerae2e_db_search(
    query: str | None = None,
    *,
    family: str | None = None,
    role: str | None = None,
    status: str | None = None,
    tags: Iterable[str] | None = None,
    include_missing: bool = True,
) -> list[dict[str, Any]]:
    """Search CameraE2E DB entries and return JSON-friendly dictionaries."""

    tag_set = {str(tag).lower() for tag in tags or ()}
    query_text = "" if query is None else str(query).lower()
    results: list[CameraE2EDBEntry] = []
    for entry in camerae2e_db_catalog(include_missing=include_missing):
        if family is not None and entry.family != family:
            continue
        if role is not None and entry.role != role:
            continue
        if status is not None and entry.status != status:
            continue
        entry_tags = {tag.lower() for tag in entry.tags}
        if tag_set and not tag_set <= entry_tags:
            continue
        haystack = " ".join(
            [
                entry.name,
                entry.artifact_id or "",
                entry.family,
                entry.role,
                entry.status,
                entry.readiness_tier,
                entry.description,
                entry.parameter_hint,
                " ".join(entry.tags),
                " ".join(entry.dependencies),
            ]
        ).lower()
        if query_text and query_text not in haystack:
            continue
        results.append(entry)
    return [entry.to_dict() for entry in results]


def camerae2e_db_get(name: str) -> dict[str, Any]:
    """Return one catalog entry by name as a JSON-friendly dictionary."""

    entry = _entry_by_name(name)
    return entry.to_dict()


def camerae2e_db_parameters(name: str) -> dict[str, Any]:
    """Return parameter values intended for direct CameraE2E API use."""

    return dict(_entry_by_name(name).parameters)


def camerae2e_db_summary() -> dict[str, Any]:
    """Return a compact summary of available CameraE2E DB entries."""

    entries = camerae2e_db_catalog()
    families: dict[str, dict[str, int]] = {}
    statuses: dict[str, int] = {}
    readiness: dict[str, int] = {}
    for entry in entries:
        family_counts = families.setdefault(entry.family, {})
        family_counts[entry.status] = family_counts.get(entry.status, 0) + 1
        statuses[entry.status] = statuses.get(entry.status, 0) + 1
        readiness[entry.readiness_tier] = readiness.get(entry.readiness_tier, 0) + 1
    return {
        "total": len(entries),
        "schema_version": "camerae2e_db_manifest_v1",
        "families": families,
        "statuses": statuses,
        "readiness_tiers": readiness,
        "active": [entry.name for entry in entries if entry.status == "active"],
        "available": [
            entry.name for entry in entries if entry.status in {"active", "available", "fallback"}
        ],
        "stale_dependencies": [entry.name for entry in entries if entry.stale_reason],
    }


def camerae2e_db_manifest(*, include_missing: bool = True) -> dict[str, Any]:
    """Return the full CameraE2E data-asset manifest.

    This is the machine-readable source of truth for DB/LUT provenance,
    readiness tier, dependency edges, and validation gates.  It is intentionally
    an index over external assets rather than a copy of those assets.
    """

    entries = camerae2e_db_catalog(include_missing=include_missing)
    return {
        "schema_version": "camerae2e_db_manifest_v1",
        "seed": _registry_seed(),
        "summary": camerae2e_db_summary(),
        "entries": [entry.to_dict() for entry in entries],
    }


def camerae2e_db_validate(*, strict: bool = False, include_missing: bool = True) -> dict[str, Any]:
    """Validate the DB/LUT manifest and return structured issues.

    Non-strict validation accepts proxy and calibration-required assets as long
    as they are labeled truthfully.  Strict validation treats proxy,
    calibration-required, missing, and stale entries as failures.
    """

    entries = camerae2e_db_catalog(include_missing=include_missing)
    valid_tiers = {
        "missing",
        "available",
        "proxy",
        "validated",
        "calibration_required",
        "calibrated",
    }
    names = {entry.name for entry in entries}
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    stale_dependencies: list[dict[str, Any]] = []

    for entry in entries:
        if entry.readiness_tier not in valid_tiers:
            issues.append(
                {
                    "entry": entry.name,
                    "kind": "invalid_readiness_tier",
                    "message": f"Unknown readiness tier {entry.readiness_tier!r}.",
                }
            )
        if entry.status != "missing" and entry.path is not None and not Path(entry.path).exists():
            issues.append(
                {
                    "entry": entry.name,
                    "kind": "missing_path",
                    "message": f"Catalog path does not exist: {entry.path}",
                }
            )
        for dependency in entry.dependencies:
            if dependency not in names:
                issues.append(
                    {
                        "entry": entry.name,
                        "kind": "unknown_dependency",
                        "message": f"Dependency {dependency!r} is not a catalog entry.",
                    }
                )
        if entry.stale_reason:
            payload = {
                "entry": entry.name,
                "kind": "stale_dependency",
                "message": entry.stale_reason,
            }
            stale_dependencies.append(payload)
            (issues if strict else warnings).append(payload)
        if strict and entry.readiness_tier in {"missing", "proxy", "calibration_required"}:
            issues.append(
                {
                    "entry": entry.name,
                    "kind": "readiness_tier",
                    "message": f"{entry.readiness_tier} is not accepted by strict validation.",
                }
            )

    return {
        "schema_version": "camerae2e_db_validation_v1",
        "strict": bool(strict),
        "ok": not issues,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "stale_dependency_count": len(stale_dependencies),
        "issues": issues,
        "warnings": warnings,
        "stale_dependencies": stale_dependencies,
    }


def camerae2e_db_lineage(name: str, *, include_missing: bool = True) -> dict[str, Any]:
    """Return a dependency lineage graph rooted at one catalog entry."""

    entries = {entry.name: entry for entry in camerae2e_db_catalog(include_missing=include_missing)}
    if name not in entries:
        available = ", ".join(sorted(entries))
        raise KeyError(f"Unknown CameraE2E DB entry {name!r}. Available entries: {available}")

    visited: set[str] = set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    def visit(entry_name: str) -> None:
        if entry_name in visited:
            return
        visited.add(entry_name)
        entry = entries[entry_name]
        nodes.append(entry.to_dict())
        for dependency in entry.dependencies:
            edges.append({"from": entry.name, "to": dependency})
            if dependency in entries:
                visit(dependency)

    visit(str(name))
    return {
        "schema_version": "camerae2e_db_lineage_v1",
        "root": str(name),
        "nodes": nodes,
        "edges": edges,
    }


def _entry_by_name(name: str) -> CameraE2EDBEntry:
    key = str(name)
    for entry in camerae2e_db_catalog():
        if entry.name == key:
            return entry
    available = ", ".join(entry.name for entry in camerae2e_db_catalog(include_missing=False))
    raise KeyError(f"Unknown CameraE2E DB entry {name!r}. Available entries: {available}")


def _lens_entries() -> list[CameraE2EDBEntry]:
    entries: list[CameraE2EDBEntry] = []
    active_data_dir = lens_patent_default_data_dir()
    active_db = lens_patent_default_db_path()
    raytrace_dir = active_data_dir / "raytrace_psf"
    highres_dir = active_data_dir / "raytrace_psf_highres"
    lens_meta: dict[str, Any] = {}
    try:
        summary = lens_patent_db_summary(active_db)
        lens_meta["summary"] = summary
    except Exception as exc:  # pragma: no cover - defensive catalog path.
        lens_meta["load_error"] = str(exc)
    try:
        lens_meta["camerae2e_manifest"] = lens_patent_camerae2e_manifest(db_path=active_db)
    except Exception:
        pass
    try:
        lens_meta["raytrace_psf_summary"] = lens_patent_raytrace_psf_manifest(raytrace_dir).get(
            "summary", {}
        )
    except Exception:
        pass
    if (highres_dir / "manifest.json").exists():
        try:
            lens_meta["highres_psf_summary"] = lens_patent_raytrace_psf_manifest(highres_dir).get(
                "summary", {}
            )
        except Exception:
            pass

    entries.append(
        CameraE2EDBEntry(
            name="lens_patents_active",
            family="lens",
            role="optics-prescription-and-psf",
            status="active" if active_db.exists() else "missing",
            path=active_db,
            description=(
                "Active lens patent SQLite DB and RayOptics PSF package "
                "used by lens_patent_* APIs."
            ),
            parameter_hint=(
                "Use db_path for lens_patent_search/get/optics and "
                "psf_dir/highres_psf_dir for lens_patent_raytrace_optics."
            ),
            parameters={
                "data_dir": active_data_dir,
                "db_path": active_db,
                "psf_dir": raytrace_dir,
                "highres_psf_dir": highres_dir,
            },
            env_vars=(
                "PYISETCAM_LENS_PATENT_DB",
                "PYISETCAM_LENS_DB_ROOT",
                "PYISETCAM_LENS_PATENT_PSF_DIR",
            ),
            tags=("lens", "rayoptics", "psf", "sqlite", "active"),
            metadata=lens_meta,
            artifact_id="lens:patents:active",
            readiness_tier="proxy" if active_db.exists() else "missing",
            provenance={
                "source": "RayOptics Lens Patent DB package or bundled fallback",
                "truth_boundary": "geometric_psf_not_diffraction_wave_optics",
                "selected_data_dir": active_data_dir,
            },
            source_hash=_source_hash(active_db),
            validation_gates=(
                "sqlite_exists",
                "rayoptics_psf_manifest_exists",
                "geometric_psf_caveat",
            ),
            refresh_command="python tools/render_lens_db_camerae2e_report.py",
        )
    )

    bundled_dir = Path(__file__).resolve().parent / "data" / "lens_patents"
    bundled_db = bundled_dir / "lens_patent_simulation_v6.sqlite"
    entries.append(
        CameraE2EDBEntry(
            name="lens_patents_bundled_v6",
            family="lens",
            role="fallback-optics-prescription-and-psf",
            status="active"
            if active_db == bundled_db
            else "fallback"
            if bundled_db.exists()
            else "missing",
            path=bundled_db,
            description="Bundled v6 Lens DB fallback shipped with pyisetcam package data.",
            parameter_hint=(
                "Use when external RayOptics v9 package is unavailable "
                "or when a reproducible bundled fallback is needed."
            ),
            parameters={
                "data_dir": bundled_dir,
                "db_path": bundled_db,
                "psf_dir": bundled_dir / "raytrace_psf",
                "highres_psf_dir": bundled_dir / "raytrace_psf_highres",
            },
            env_vars=(),
            tags=("lens", "rayoptics", "fallback", "sqlite"),
            metadata=_lens_db_metadata(bundled_db, bundled_dir),
            artifact_id="lens:patents:bundled:v6",
            readiness_tier="proxy" if bundled_db.exists() else "missing",
            provenance={
                "source": "bundled CameraE2E fallback lens package",
                "truth_boundary": "geometric_psf_not_diffraction_wave_optics",
            },
            source_hash=_source_hash(bundled_db),
            validation_gates=(
                "sqlite_exists",
                "rayoptics_psf_manifest_exists",
                "geometric_psf_caveat",
            ),
            refresh_command="python tools/build_lens_patent_simulation_db.py",
        )
    )

    rayoptics_v9_dir = _first_existing_path(
        [
            _camera_db_root() / "lens_db/CameraE2E_Lens_DB_v9_20260627/data/lens_patents",
            _sibling_camera_db_root()
            / "lens_db/CameraE2E_Lens_DB_v9_20260627/data/lens_patents",
            Path.home()
            / "RayOptics"
            / "CameraE2E_Lens_DB_v9_20260627"
            / "data"
            / "lens_patents",
        ]
    )
    rayoptics_v9_db = rayoptics_v9_dir / "lens_patent_simulation_v9.sqlite"
    entries.append(
        CameraE2EDBEntry(
            name="rayoptics_lens_db_v9",
            family="lens",
            role="external-optics-prescription-and-psf",
            status="available" if rayoptics_v9_db.exists() else "missing",
            path=rayoptics_v9_db,
            description="External RayOptics v9 Lens DB package prepared for CameraE2E.",
            parameter_hint=(
                "Set PYISETCAM_LENS_DB_ROOT to this package root, "
                "or pass db_path/psf_dir explicitly."
            ),
            parameters={
                "data_dir": rayoptics_v9_dir,
                "db_path": rayoptics_v9_db,
                "psf_dir": rayoptics_v9_dir / "raytrace_psf",
                "highres_psf_dir": rayoptics_v9_dir / "raytrace_psf_highres",
            },
            env_vars=("PYISETCAM_LENS_DB_ROOT",),
            tags=("lens", "rayoptics", "external", "v9", "sqlite", "psf"),
            metadata={
                **_lens_db_metadata(rayoptics_v9_db, rayoptics_v9_dir),
                "selected_as_active": active_db == rayoptics_v9_db,
            },
            artifact_id="lens:rayoptics:v9",
            readiness_tier="proxy" if rayoptics_v9_db.exists() else "missing",
            provenance={
                "source": "external RayOptics CameraE2E Lens DB v9 package",
                "truth_boundary": "geometric_psf_not_diffraction_wave_optics",
            },
            source_hash=_source_hash(rayoptics_v9_db),
            validation_gates=(
                "sqlite_exists",
                "rayoptics_psf_manifest_exists",
                "geometric_psf_caveat",
            ),
            refresh_command="python tools/render_lens_db_camerae2e_report.py",
        )
    )
    return entries


def _lens_db_metadata(db_path: Path, data_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if db_path.exists():
        try:
            metadata["summary"] = lens_patent_db_summary(db_path)
        except Exception as exc:
            metadata["summary_error"] = str(exc)
    for key, manifest_path in {
        "raytrace_psf_summary": data_dir / "raytrace_psf" / "manifest.json",
        "highres_psf_summary": data_dir / "raytrace_psf_highres" / "manifest.json",
    }.items():
        if manifest_path.exists():
            try:
                metadata[key] = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "summary", {}
                )
            except Exception as exc:
                metadata[f"{key}_error"] = str(exc)
    return metadata


def _sensor_entries() -> list[CameraE2EDBEntry]:
    entries: list[CameraE2EDBEntry] = []
    lut_path = fdtd_sensor_default_lut_path()
    lut_meta: dict[str, Any] = {}
    if lut_path is not None and lut_path.exists():
        try:
            lut = fdtd_sensor_lut_load(lut_path)
            lut_meta["summary"] = fdtd_sensor_lut_summary(lut)
            lut_meta["physics_validation"] = fdtd_sensor_physics_validate(lut)
        except Exception as exc:
            lut_meta["load_error"] = str(exc)
    fdtd_stale = _fdtd_stale_reason(lut_path, lut_meta)
    entries.append(
        CameraE2EDBEntry(
            name="fdtd_sensor_lut_active",
            family="sensor",
            role="optical-response-lut",
            status="active" if lut_path is not None and lut_path.exists() else "missing",
            path=lut_path,
            description=(
                "Active FDTD-informed sensor optical-response LUT used for "
                "QE/RI/crosstalk proxy configuration."
            ),
            parameter_hint=(
                "Use lut_path with fdtd_sensor_config(lut_path) and "
                "sensor_attach_fdtd_lut."
            ),
            parameters={"lut_path": lut_path, "config": {"lut": lut_path, "enabled": True}},
            env_vars=("PYISETCAM_FDTD_LUT_PATH", "PYISETCAM_FDTD_ROOT"),
            tags=("sensor", "fdtd", "qe", "relative-illumination", "crosstalk", "lut"),
            metadata=lut_meta,
            artifact_id="sensor:fdtd:active_lut",
            readiness_tier="proxy" if lut_path is not None and lut_path.exists() else "missing",
            provenance={
                "source": "external FDTD workspace",
                "truth_boundary": "optical_absorption_and_regional_response_proxy",
            },
            source_hash=_source_hash(lut_path),
            dependencies=("fdtd_sensor_stack_catalog",),
            validation_gates=(
                "lut_schema",
                "physics_sanity",
                "convergence_metadata",
                "proxy_caveat",
            ),
            refresh_command="python tools/render_fdtd_sensor_physics_report.py",
            stale_reason=fdtd_stale,
        )
    )

    root = Path(tcad_sensor_default_paths()["root"]).expanduser()
    sensor_catalog = root / "sensor_db" / "sensor_catalog.json"
    stack_dir = root / "sensor_db" / "generated_stack_configs"
    entries.append(
        CameraE2EDBEntry(
            name="fdtd_sensor_stack_catalog",
            family="sensor",
            role="sensor-stack-parameter-catalog",
            status="available" if sensor_catalog.exists() else "missing",
            path=sensor_catalog,
            description="External FDTD sensor stack catalog and generated layer-stack configs.",
            parameter_hint=(
                "Use catalog_path to choose a sensor stack, then feed selected "
                "FDTD/TCAD artifacts into sensor config."
            ),
            parameters={"catalog_path": sensor_catalog, "stack_config_dir": stack_dir},
            env_vars=("PYISETCAM_FDTD_ROOT",),
            tags=("sensor", "fdtd", "stack", "catalog", "external"),
            metadata={
                "stack_config_count": _count_files(stack_dir, "*.json"),
                "csv_path": root / "sensor_db" / "sensor_catalog.csv",
                "validation_path": root / "sensor_db" / "validation.json",
            },
            artifact_id="sensor:fdtd:stack_catalog",
            readiness_tier="proxy" if sensor_catalog.exists() else "missing",
            provenance={
                "source": "external FDTD sensor_db package",
                "truth_boundary": "metadata_derived_stack_configs_unless_product_cad_listed",
            },
            source_hash=_source_hash(sensor_catalog),
            validation_gates=(
                "catalog_exists",
                "generated_stack_config_count",
                "metadata_proxy_caveat",
            ),
            refresh_command="python tools/render_sensor_db_overview.py",
        )
    )

    defaults = tcad_sensor_default_paths(root)
    validation: dict[str, Any] = {}
    try:
        db = tcad_sensor_db_load(root=root)
        validation = tcad_sensor_validate(db)
    except Exception as exc:
        validation = {"load_error": str(exc)}
    tcad_tier = _tcad_readiness_tier(validation)
    tcad_stale = _tcad_stale_reason(
        validation,
        active_fdtd_lut_path=lut_path,
        generation_map_path=defaults["generation_map_path"],
    )
    entries.append(
        CameraE2EDBEntry(
            name="tcad_sensor_db_active",
            family="sensor",
            role="electrical-collection-lut",
            status="active" if validation.get("ok") or validation.get("status") else "missing",
            path=Path(defaults["root"]),
            description=(
                "Active DEVSIM/TCAD sensor collection DB joining FDTD "
                "generation maps with split-PD electrical summaries."
            ),
            parameter_hint=(
                "Use these paths with tcad_sensor_db_load(...) or "
                "sensor_attach_tcad_lut."
            ),
            parameters=defaults,
            env_vars=("PYISETCAM_FDTD_ROOT",),
            tags=("sensor", "tcad", "devsim", "collection", "split-pd"),
            metadata={"validation": validation},
            artifact_id="sensor:tcad:active_collection",
            readiness_tier=tcad_tier,
            provenance={
                "source": "external FDTD/DEVSIM workspace",
                "truth_boundary": "carrier_collection_framework_not_product_calibrated_tcad",
            },
            source_hash=_source_hash(defaults["accuracy_gate_path"]),
            dependencies=("fdtd_sensor_lut_active", "fdtd_sensor_stack_catalog"),
            validation_gates=(
                "generation_map_schema",
                "devsim_summary_balance",
                "accuracy_gate",
                "lineage_match",
            ),
            refresh_command="python tools/render_fdtd_tcad_sensor_report.py",
            stale_reason=tcad_stale,
        )
    )
    return entries


def _hwisp_entries() -> list[CameraE2EDBEntry]:
    root = Path(
        os.environ.get("PYISETCAM_HWISP_DB", Path(__file__).resolve().parent / "data" / "hwisp")
    ).expanduser()
    names: list[str] = []
    try:
        names = hw_isp_profile_names(root)
    except Exception:
        names = []
    return [
        CameraE2EDBEntry(
            name="hwisp_parameter_profiles",
            family="isp",
            role="timing-and-3a-parameter-profiles",
            status="active" if names else "missing",
            path=root,
            description=(
                "HW ISP timing, transport, and 3A profile DB for "
                "system-level latency simulation."
            ),
            parameter_hint="Use profile_name with hw_isp_config_from_profile(profile_name).",
            parameters={
                "db_path": root,
                "profile_names": names,
                "default_profile": names[0] if names else None,
            },
            env_vars=("PYISETCAM_HWISP_DB",),
            tags=("isp", "hwisp", "latency", "3a", "profile"),
            metadata={"profile_count": len(names)},
            artifact_id="isp:hwisp:profiles",
            readiness_tier="proxy" if names else "missing",
            provenance={
                "source": "bundled or user-provided HW ISP profile DB",
                "truth_boundary": "seed_or_public_profile_not_vendor_trace_signoff",
            },
            source_hash=_source_hash(root),
            validation_gates=("profile_schema", "seed_profile_caveat", "timing_report"),
            refresh_command=(
                "python tools/render_hwisp_parameter_requirements_report.py "
                "--profile rpi_vc4_imx219_public_seed"
            ),
        )
    ]


def _perception_entries() -> list[CameraE2EDBEntry]:
    profiles_path = (
        Path(__file__).resolve().parent / "data" / "task_perception" / "model_profiles.json"
    )
    names = task_model_profile_names()
    cache_dir = Path(
        os.environ.get(
            "PYISETCAM_TASK_MODEL_CACHE",
            Path.home() / ".cache" / "pyisetcam" / "task_perception" / "yolo",
        )
    ).expanduser()
    return [
        CameraE2EDBEntry(
            name="task_perception_model_profiles",
            family="perception",
            role="model-profile-catalog",
            status="active" if profiles_path.exists() else "missing",
            path=profiles_path,
            description=(
                "Task perception model profile DB covering YOLO, tracking, "
                "detection, segmentation, pose, OBB, and optional adapters."
            ),
            parameter_hint=(
                "Use profile_name with task_model_config_from_profile(profile_name), "
                "then task_model_from_config."
            ),
            parameters={
                "profiles_path": profiles_path,
                "profile_names": names,
                "cache_dir": cache_dir,
            },
            env_vars=("PYISETCAM_TASK_MODEL_CACHE",),
            tags=("perception", "yolo", "detection", "segmentation", "tracking", "model-profile"),
            metadata={
                "profile_count": len(names),
                "cached_model_count": _count_files(cache_dir, "*.pt"),
            },
            artifact_id="perception:task_model_profiles",
            readiness_tier="available" if profiles_path.exists() else "missing",
            provenance={
                "source": "bundled task-perception model profile registry",
                "truth_boundary": "adapter_profiles_not_training_dataset_or_model_accuracy_claim",
            },
            source_hash=_source_hash(profiles_path),
            validation_gates=("profile_schema", "optional_backend_import"),
            refresh_command="python tools/render_task_perception_report.py",
        )
    ]


def _upstream_entries() -> list[CameraE2EDBEntry]:
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(os.environ.get("PYISETCAM_CACHE_ROOT", repo_root / ".cache")).expanduser()
    override = os.environ.get("PYISETCAM_UPSTREAM_ROOT")
    snapshot_root = (
        Path(override).expanduser()
        if override
        else cache_root / "upstream" / "isetcam" / DEFAULT_UPSTREAM_SHA
    )
    return [
        CameraE2EDBEntry(
            name="isetcam_upstream_snapshot",
            family="assets",
            role="upstream-matlab-asset-snapshot",
            status="active" if snapshot_root.exists() else "missing",
            path=snapshot_root,
            description=(
                "Pinned upstream ISETCam asset snapshot used by AssetStore "
                "for MATLAB-origin data files."
            ),
            parameter_hint=(
                "Use AssetStore.default(), or pass PYISETCAM_UPSTREAM_ROOT "
                "for an explicit snapshot."
            ),
            parameters={
                "snapshot_root": snapshot_root,
                "cache_root": cache_root,
                "sha": DEFAULT_UPSTREAM_SHA,
            },
            env_vars=("PYISETCAM_CACHE_ROOT", "PYISETCAM_UPSTREAM_ROOT"),
            tags=("assets", "upstream", "isetcam", "matlab", "snapshot"),
            metadata={"sha": DEFAULT_UPSTREAM_SHA},
            artifact_id="assets:isetcam_upstream_snapshot",
            readiness_tier="validated" if snapshot_root.exists() else "missing",
            provenance={
                "source": "pinned upstream ISETCam snapshot",
                "upstream_sha": DEFAULT_UPSTREAM_SHA,
            },
            source_hash=_source_hash(snapshot_root),
            validation_gates=("snapshot_exists", "pinned_sha"),
            refresh_command="python tools/fetch_upstream.py",
        )
    ]


def _parity_entries() -> list[CameraE2EDBEntry]:
    repo_root = Path(__file__).resolve().parents[2]
    cases_yaml = repo_root / "tests" / "parity" / "cases.yaml"
    baselines_dir = repo_root / "tests" / "parity" / "baselines"
    latest_json = repo_root / "reports" / "parity" / "latest.json"
    return [
        CameraE2EDBEntry(
            name="matlab_parity_baselines",
            family="parity",
            role="matlab-baseline-evidence",
            status="available" if cases_yaml.exists() and baselines_dir.exists() else "missing",
            path=baselines_dir,
            description=(
                "Curated MATLAB baseline .mat DB and parity case registry "
                "for regression/evidence reports."
            ),
            parameter_hint=(
                "Use cases_yaml and baselines_dir with tests/parity "
                "or tools/parity_report.py."
            ),
            parameters={
                "cases_yaml": cases_yaml,
                "baselines_dir": baselines_dir,
                "latest_report": latest_json,
            },
            env_vars=("PYISETCAM_RUN_PARITY",),
            tags=("parity", "matlab", "baseline", "evidence"),
            metadata={
                "baseline_count": _count_files(baselines_dir, "*.mat"),
                "latest_report_exists": latest_json.exists(),
            },
            artifact_id="parity:matlab_baselines",
            readiness_tier="validated"
            if cases_yaml.exists() and baselines_dir.exists()
            else "missing",
            provenance={"source": "curated MATLAB/Octave parity baselines"},
            source_hash=_source_hash(cases_yaml),
            validation_gates=("cases_yaml_exists", "baseline_count", "parity_report"),
            refresh_command="python tools/parity_report.py",
        )
    ]


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    return sum(1 for _ in root.glob(pattern))


def _registry_seed() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "data" / "camerae2e_registry_seed.json"
    if not path.exists():
        return {"schema_version": "camerae2e_registry_seed_v1", "readiness_tiers": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive packaging guard.
        return {"schema_version": "camerae2e_registry_seed_v1", "load_error": str(exc)}


def _source_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return None
    try:
        if resolved.is_dir():
            entries = sorted(item.name for item in resolved.iterdir())[:512]
            stat = resolved.stat()
            digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()[:16]
            return f"dir:{digest}:mtime_ns={stat.st_mtime_ns}:entries={len(entries)}"
        stat = resolved.stat()
        if stat.st_size > 64 * 1024 * 1024:
            return f"file:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except OSError as exc:
        return f"unavailable:{exc}"


def _fdtd_stale_reason(lut_path: Path | None, metadata: Mapping[str, Any]) -> str | None:
    if lut_path is None or not lut_path.exists():
        return None
    if "load_error" in metadata:
        return f"FDTD LUT could not be loaded: {metadata['load_error']}"
    validation = metadata.get("physics_validation", {})
    failures = validation.get("failures", []) if isinstance(validation, Mapping) else []
    if failures:
        return "FDTD physics validation has blocking failures: " + "; ".join(
            str(item) for item in failures
        )
    return None


def _tcad_readiness_tier(validation: Mapping[str, Any]) -> str:
    if validation.get("load_error"):
        return "missing"
    if bool(validation.get("accuracy_ready", False)):
        return "validated"
    if bool(validation.get("framework_ready", False)):
        return "calibration_required"
    if validation.get("status"):
        return "proxy"
    return "missing"


def _tcad_stale_reason(
    validation: Mapping[str, Any],
    *,
    active_fdtd_lut_path: str | Path | None = None,
    generation_map_path: str | Path | None = None,
) -> str | None:
    if validation.get("load_error"):
        return f"TCAD DB could not be loaded: {validation['load_error']}"
    warnings = [str(item) for item in validation.get("warnings", [])]
    lineage = [item for item in warnings if "was generated from" in item]
    if lineage:
        return "TCAD/FDTD lineage mismatch: " + "; ".join(lineage)
    if active_fdtd_lut_path is not None and generation_map_path is not None:
        fdtd_path = Path(active_fdtd_lut_path)
        generation_path = Path(generation_map_path)
        if (
            fdtd_path.exists()
            and generation_path.exists()
            and fdtd_path.parent.resolve() != generation_path.parent.resolve()
        ):
            return (
                "TCAD/FDTD active artifact mismatch: "
                f"active FDTD LUT is under {fdtd_path.parent}, "
                f"but TCAD generation map is under {generation_path.parent}"
            )
    if validation.get("status") == "invalid":
        issues = "; ".join(str(item) for item in validation.get("issues", []))
        return (
            f"TCAD DB validation is invalid: {issues}"
            if issues
            else "TCAD DB validation is invalid."
        )
    return None


def _camera_db_root() -> Path:
    explicit = os.environ.get(_DEFAULT_CAMERA_DB_ROOT_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return Path(__file__).resolve().parents[2] / "camerae2e_db"


def _sibling_camera_db_root() -> Path:
    return Path(__file__).resolve().parents[2].parent / "CameraE2E-DB"


def _first_existing_path(candidates: list[Path]) -> Path:
    fallback = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


cameraE2EDBCatalog = camerae2e_db_catalog  # noqa: N816
cameraE2EDBSearch = camerae2e_db_search  # noqa: N816
cameraE2EDBGet = camerae2e_db_get  # noqa: N816
cameraE2EDBParameters = camerae2e_db_parameters  # noqa: N816
cameraE2EDBSummary = camerae2e_db_summary  # noqa: N816
cameraE2EDBManifest = camerae2e_db_manifest  # noqa: N816
cameraE2EDBValidate = camerae2e_db_validate  # noqa: N816
cameraE2EDBLineage = camerae2e_db_lineage  # noqa: N816
