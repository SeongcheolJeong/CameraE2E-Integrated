"""Canonical CameraE2E asset catalog seeding from the integrated DB manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyisetcam import camerae2e_db_manifest

from .models import (
    AssetKind,
    CameraAssetRecord,
    FidelityLevel,
    ReadinessTier,
)
from .project import Project

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILTIN_ASSETS: dict[str, tuple[AssetKind, FidelityLevel]] = {
    "rayoptics_lens_db_v9": (AssetKind.LENS_DESIGN, FidelityLevel.LUT),
    "fdtd_sensor_stack_catalog": (AssetKind.SENSOR_PRODUCT, FidelityLevel.LUT),
    "fdtd_sensor_lut_active": (AssetKind.OPTICAL_LUT, FidelityLevel.LUT),
    "tcad_sensor_db_active": (AssetKind.PIXEL_STACK, FidelityLevel.SOLVER),
    "hwisp_parameter_profiles": (AssetKind.ISP_PROFILE, FidelityLevel.ANALYTIC),
    "task_perception_model_profiles": (AssetKind.PERCEPTION_MODEL, FidelityLevel.ANALYTIC),
}


def seed_builtin_camera_assets(project: Project) -> list[CameraAssetRecord]:
    """Register small immutable descriptors for repository-local runtime assets."""

    existing_sources = {item.source for item in project.store.list_camera_assets()}
    entries = {
        str(entry.get("name")): dict(entry)
        for entry in camerae2e_db_manifest(include_missing=False).get("entries", [])
    }
    seeded = []
    for name, (kind, fidelity) in _BUILTIN_ASSETS.items():
        source = f"camerae2e_builtin_registry:{name}"
        if source in existing_sources or name not in entries:
            continue
        entry = entries[name]
        path = _portable_path(entry.get("path"))
        descriptor = {
            "schema_version": "camerae2e_builtin_asset_descriptor_v2",
            "registry_name": name,
            "runtime_path": path,
            "source_hash": entry.get("source_hash"),
            "readiness_tier": entry.get("readiness_tier"),
            "provenance": entry.get("provenance", {}),
            "dependencies": entry.get("dependencies", []),
            "validation_gates": entry.get("validation_gates", []),
            "stale_reason": entry.get("stale_reason"),
            "refresh_command": entry.get("refresh_command"),
        }
        readiness = _readiness(entry.get("readiness_tier"))
        descriptor_artifact = project.artifacts.put_json(
            descriptor,
            artifact_type="camera_asset_descriptor",
            fidelity_level=fidelity,
            readiness_tier=readiness,
            source=source,
            validation={
                "runtime_path_exists": _runtime_path_exists(path),
                "stale_reason": entry.get("stale_reason"),
            },
        )
        asset = CameraAssetRecord(
            kind=kind,
            name=name,
            version=str(entry.get("schema_version", "1")),
            artifact_hashes=[descriptor_artifact.hash],
            fidelity_level=fidelity,
            readiness_tier=readiness,
            parameters={"runtime_path": path, "registry_artifact_id": entry.get("artifact_id")},
            valid_domain=dict(entry.get("parameters", {})),
            source=source,
            validation={
                "gates": entry.get("validation_gates", []),
                "stale_reason": entry.get("stale_reason"),
            },
        )
        project.store.put_camera_asset(asset)
        seeded.append(asset)
    return seeded


def _portable_path(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _runtime_path_exists(value: str | None) -> bool:
    if value is None:
        return False
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.exists()


def _readiness(value: Any) -> ReadinessTier:
    try:
        return ReadinessTier(str(value))
    except ValueError:
        return ReadinessTier.AVAILABLE
