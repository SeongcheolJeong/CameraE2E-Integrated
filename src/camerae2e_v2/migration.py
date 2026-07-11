"""One-way migration helpers from CameraE2E v1 settings and data roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .models import (
    AssetKind,
    CameraAssetRecord,
    CameraModule,
    FidelityLevel,
    ISPConfig,
    LensConfig,
    ReadinessTier,
    SceneCase,
    SceneType,
    SearchMethod,
    SensorConfig,
    StudyCreate,
)
from .project import Project


def migrate_v1_settings(project: Project, source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        payload = dict(source)
        source_name = "inline"
    else:
        path = Path(source).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(payload.get("settings", payload))
        source_name = str(path)
    module = CameraModule(
        name=str(payload.get("cameraPreset", "Migrated CameraE2E v1 module")),
        lens=LensConfig(
            hfov_deg=float(payload.get("fovDeg", 81.4)),
            f_number=float(payload.get("fNumber", 1.8)),
            psf_radius_um=float(payload.get("lensPsfRadiusUm", 2.0)),
        ),
        sensor=SensorConfig(
            pixel_size_um=float(payload.get("pixelSizeUm", 3.75)),
            cfa_preset=str(payload.get("cfaPreset", "bayer_rgb")),
            exposure_ms=float(payload.get("exposureMs", 4.0)),
            analog_gain=float(payload.get("analogGain", 1.0)),
            binning_factor=int(payload.get("binningFactor", 1)),
            ocl_mode=str(payload.get("oclMode", "centered")),
            ocl_group_shape=str(payload.get("oclGroupShape", "2x2")),
            ocl_equalization=float(payload.get("oclEqualization", 0.5)),
        ),
        isp=ISPConfig(
            demosaic_method=str(payload.get("demosaicMethod", "bilinear")),
            ccm_method=str(payload.get("sensorConversionMethod", "mcc_optimized")),
        ),
        hw_isp_enabled=bool(payload.get("hwIspEnabled", False)),
        hw_isp_frames=int(payload.get("hwIspFrames", 2)),
        metadata={
            "migrated_from": source_name,
            "legacy_settings": payload,
            "migration_boundary": (
                "Unrecognized v1 settings are preserved as metadata and are not silently "
                "treated as active v2 design variables."
            ),
        },
    )
    scene_type = str(payload.get("sceneType", "macbeth"))
    image_path = payload.get("kittiImagePath")
    label_path = payload.get("kittiLabelPath")
    if image_path and Path(str(image_path)).expanduser().is_file():
        scene = SceneCase(
            name="Migrated RGB scene",
            source_kind="display_rgb_proxy",
            scene_type="rgb_file",
            image_path=str(Path(str(image_path)).expanduser().resolve()),
            label_path=(
                str(Path(str(label_path)).expanduser().resolve())
                if label_path and Path(str(label_path)).expanduser().is_file()
                else None
            ),
        )
    else:
        mapped: SceneType = "slanted_bar" if "slanted" in scene_type.lower() else "macbeth"
        scene = SceneCase(name=f"Migrated {scene_type}", scene_type=mapped)
    target = str(payload.get("targetProfile", "raw_quality"))
    model = payload.get("yoloModelPath")
    perception_ready = bool(
        target == "adas_yolo_perception"
        and model
        and Path(str(model)).expanduser().is_file()
        and scene.label_path
    )
    raw_method = str(payload.get("optimizationMethod", "latin_hypercube"))
    allowed_methods = {
        "grid",
        "random",
        "latin_hypercube",
        "evolutionary",
        "surrogate",
        "gaussian_process",
    }
    search_method = cast(
        SearchMethod,
        raw_method if raw_method in allowed_methods else "latin_hypercube",
    )
    spec = StudyCreate(
        name=f"Migrated {payload.get('goal', 'CameraE2E v1')} study",
        baseline=module,
        scenes=[scene],
        target_profile="adas_yolo_perception" if perception_ready else "raw_quality",
        perception_model_path=(
            str(Path(str(model)).expanduser().resolve()) if perception_ready else None
        ),
        seed=int(payload.get("seed", 42)),
        search_budget=int(payload.get("maxCandidates", 24)),
        search_method=search_method,
    )
    study = project.create_study(spec)
    return {
        "schema_version": "camerae2e_v1_migration_v2",
        "source": source_name,
        "study": study.model_dump(mode="json"),
        "perception_preserved": perception_ready,
        "warnings": (
            []
            if perception_ready or target != "adas_yolo_perception"
            else [
                "v1 requested perception optimization but model/labels were unavailable; "
                "the migrated study uses raw_quality and does not create a fake perception score."
            ]
        ),
    }


def import_legacy_db(project: Project, source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve()
    if (root / "camerae2e_db").is_dir():
        root = root / "camerae2e_db"
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = project.artifacts.import_tree(
        root,
        artifact_type="legacy_camera_db_asset",
        source=f"camerae2e_v1_db:{root}",
    )
    assets = []
    for record in records:
        relative = str(record.metadata.get("source_relative_path", record.relative_path))
        asset = CameraAssetRecord(
            kind=_legacy_asset_kind(relative),
            name=Path(relative).stem,
            artifact_hashes=[record.hash],
            fidelity_level=FidelityLevel.LUT,
            readiness_tier=ReadinessTier.PROXY,
            source=f"camerae2e_v1_db:{root}",
            parameters={"source_relative_path": relative},
            validation={"migrated": True, "calibration_promoted": False},
        )
        project.store.put_camera_asset(asset)
        assets.append(asset)
    return {
        "schema_version": "camerae2e_db_migration_v2",
        "source": str(root),
        "artifact_count": len(records),
        "artifacts": [record.model_dump(mode="json") for record in records],
        "camera_assets": [asset.model_dump(mode="json") for asset in assets],
        "truth_boundary": (
            "Imported v1 DB files retain proxy/calibration-required status; migration does not "
            "promote their physical accuracy."
        ),
    }


def _legacy_asset_kind(relative_path: str) -> AssetKind:
    key = relative_path.lower()
    if "lens" in key or "psf" in key:
        return AssetKind.LENS_DESIGN
    if "/nk/" in f"/{key}" or "material" in key:
        return AssetKind.MATERIAL_NK
    if "camera_lut" in key or "crosstalk" in key or "generation_map" in key:
        return AssetKind.OPTICAL_LUT
    if "stack" in key or "pixel" in key or "tcad" in key:
        return AssetKind.PIXEL_STACK
    if "isp" in key:
        return AssetKind.ISP_PROFILE
    return AssetKind.SENSOR_PRODUCT
