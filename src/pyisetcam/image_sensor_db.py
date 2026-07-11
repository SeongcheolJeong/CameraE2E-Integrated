"""Image-sensor database selection helpers.

This module indexes the external FDTD ``sensor_db`` package so callers can
search image sensors, select one by stable ID, and pass its generated stack /
TCAD paths into CameraE2E configuration code.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .fdtd_sensor import fdtd_sensor_default_lut_path
from .tcad_sensor import tcad_sensor_default_paths

_DEFAULT_CAMERA_DB_ROOT_ENV = "PYISETCAM_CAMERA_DB_ROOT"


def image_sensor_db_root(root: str | Path | None = None) -> Path:
    """Return the active image-sensor DB root directory."""

    if root is not None:
        return Path(root).expanduser()
    explicit = os.environ.get("PYISETCAM_IMAGE_SENSOR_DB_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    camera_db_root = os.environ.get(_DEFAULT_CAMERA_DB_ROOT_ENV)
    if camera_db_root:
        return Path(camera_db_root).expanduser() / "fdtd_tcad/sensor_db"
    for candidate in _default_camera_db_sensor_roots():
        if candidate.exists():
            return candidate
    repo_root = Path(__file__).resolve().parents[2]
    fdtd_root = Path(
        os.environ.get("PYISETCAM_FDTD_ROOT", str(repo_root / "simulations/fdtd_tcad"))
    ).expanduser()
    return fdtd_root / "sensor_db"


def _default_camera_db_sensor_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "camerae2e_db/fdtd_tcad/sensor_db",
        repo_root / "simulations/fdtd_tcad/sensor_db",
        repo_root.parent / "CameraE2E-DB/fdtd_tcad/sensor_db",
    ]


def image_sensor_db_catalog_path(root: str | Path | None = None) -> Path:
    """Return the active image-sensor catalog JSON path."""

    return image_sensor_db_root(root) / "sensor_catalog.json"


def image_sensor_db_load(root: str | Path | None = None) -> dict[str, Any]:
    """Load the image-sensor catalog JSON."""

    path = image_sensor_db_catalog_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Image sensor DB catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def image_sensor_db_records(
    query: str | None = None,
    *,
    manufacturer: str | None = None,
    limit: int | None = None,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return normalized image-sensor records with optional text filters."""

    catalog = image_sensor_db_load(root)
    query_text = "" if query is None else str(query).lower()
    manufacturer_text = "" if manufacturer is None else str(manufacturer).lower()
    records: list[dict[str, Any]] = []
    for raw in catalog.get("records", []):
        record = _normalize_record(raw, image_sensor_db_root(root))
        if manufacturer_text and manufacturer_text not in record["manufacturer"].lower():
            continue
        haystack = " ".join(
            str(record.get(key, ""))
            for key in (
                "sensor_id",
                "code",
                "manufacturer",
                "device_name",
                "title",
                "pixel_architecture",
                "cfa_pattern",
                "illumination",
            )
        ).lower()
        if query_text and query_text not in haystack:
            continue
        records.append(record)
    records.sort(
        key=lambda item: (
            item["manufacturer"].lower(),
            item["device_name"].lower(),
            item["code"],
        )
    )
    if limit is not None:
        records = records[: int(limit)]
    return records


def image_sensor_db_get(sensor_id_or_code: str, root: str | Path | None = None) -> dict[str, Any]:
    """Return one normalized image-sensor record by ``sensor_id`` or report code."""

    key = str(sensor_id_or_code).lower()
    for record in image_sensor_db_records(root=root):
        if key in {str(record["sensor_id"]).lower(), str(record["code"]).lower()}:
            return record
    raise KeyError(f"Unknown image sensor DB record: {sensor_id_or_code}")


def image_sensor_db_parameters(
    sensor_id_or_code: str, root: str | Path | None = None
) -> dict[str, Any]:
    """Return path parameters for direct CameraE2E sensor configuration."""

    record = image_sensor_db_get(sensor_id_or_code, root=root)
    defaults = tcad_sensor_default_paths(image_sensor_db_root(root).parent)
    return {
        "sensor_id": record["sensor_id"],
        "code": record["code"],
        "catalog_path": image_sensor_db_catalog_path(root),
        "stack_config_path": (
            Path(record["stack_config_path"]) if record["stack_config_path"] else None
        ),
        "tcad_profile_path": (
            Path(record["tcad_profile_path"]) if record["tcad_profile_path"] else None
        ),
        "lut_path": fdtd_sensor_default_lut_path(),
        "generation_map_path": defaults["generation_map_path"],
        "collection_summary_paths": defaults["collection_summary_paths"],
        "accuracy_gate_path": defaults["accuracy_gate_path"],
    }


def image_sensor_db_config(
    sensor_id_or_code: str,
    *,
    strategy: str = "hybrid",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a CameraE2E scenario fragment from one image-sensor DB record.

    The returned ``scenario`` can be passed to ``camerae2e_run_scenario`` or used
    as the base scenario for parameter optimization.  The policy is explicit:
    DB/LUT paths are preferred when requested, while analytic sensor proxies are
    only filled from metadata that is available in the selected record.
    """

    record = image_sensor_db_get(sensor_id_or_code, root=root)
    params = image_sensor_db_parameters(sensor_id_or_code, root=root)
    resolved_strategy = _normalize_config_strategy(strategy)
    attach_luts = resolved_strategy in {"hybrid", "lut_only"}
    attach_analytic = resolved_strategy in {"hybrid", "analytic_only"}

    sensor_config = _analytic_sensor_config_from_record(
        record,
        include_ocl_proxy=attach_analytic,
    )
    fdtd_config: dict[str, Any] | None = None
    tcad_config: dict[str, Any] | None = None
    if attach_luts and _path_exists(params.get("lut_path")):
        fdtd_config = {
            "enabled": True,
            "lut": _path_string(params["lut_path"]),
            "mode": "qe+field+crosstalk",
            "crosstalk_strength": 1.0,
        }
    tcad_paths_available = (
        _path_exists(params.get("generation_map_path"))
        and _path_exists(params.get("accuracy_gate_path"))
        and all(_path_exists(item) for item in params.get("collection_summary_paths", []))
    )
    if attach_luts and tcad_paths_available:
        tcad_config = {
            "enabled": True,
            "db": {
                "root": _path_string(image_sensor_db_root(root).parent),
                "generation_map_path": _path_string(params["generation_map_path"]),
                "collection_summary_paths": [
                    _path_string(item) for item in params.get("collection_summary_paths", [])
                ],
                "accuracy_gate_path": _path_string(params["accuracy_gate_path"]),
            },
            "collection_mode": "collection",
            "allow_proxy_accuracy": True,
        }

    scenario: dict[str, Any] = {
        "name": f"image_sensor_db_{record['sensor_id']}",
        "sensor": sensor_config,
        "image_sensor_db": {
            "sensor_id": record["sensor_id"],
            "code": record["code"],
            "strategy": resolved_strategy,
        },
    }
    if fdtd_config is not None:
        scenario["fdtd"] = fdtd_config
    if tcad_config is not None:
        scenario["tcad"] = tcad_config

    component_tiers = {
        "sensor_db_metadata": "proxy",
        "analytic_sensor_proxy": "proxy" if attach_analytic and sensor_config else "missing",
        "fdtd_lut": "proxy" if fdtd_config is not None else "missing",
        "tcad_collection": "calibration_required" if tcad_config is not None else "missing",
    }
    return {
        "schema_version": "camerae2e_image_sensor_db_config_v1",
        "sensor_id": record["sensor_id"],
        "code": record["code"],
        "manufacturer": record["manufacturer"],
        "device_name": record["device_name"],
        "strategy": resolved_strategy,
        "readiness_tier": (
            "calibration_required" if tcad_config is not None else "proxy"
        ),
        "scenario": scenario,
        "sensor": sensor_config,
        "fdtd": fdtd_config,
        "tcad": tcad_config,
        "paths": _jsonable_parameters(params),
        "policy": {
            "schema_version": "camerae2e_sensor_db_policy_v1",
            "component_tiers": component_tiers,
            "db_lut_role": (
                "DB/LUT artifacts override or calibrate proxy defaults when attached."
            ),
            "analytic_role": (
                "Analytic proxy axes support broad free-configuration sweeps and "
                "should be calibrated or validated against DB/LUT/FDTD/TCAD evidence "
                "before stronger claims are made."
            ),
            "truth_boundary": (
                "Sensor DB records are metadata-derived unless measured stack, "
                "sensor-specific FDTD LUT, TCAD process deck, QE/noise data, and "
                "lineage evidence are attached."
            ),
        },
        "record_summary": _record_summary(record),
    }


def image_sensor_db_optimize_camera_parameters(
    sensor_id_or_code: str,
    *,
    strategy: str = "analytic_only",
    base_scenario: Mapping[str, Any] | None = None,
    preset: str = "raw_factory",
    parameter_space: Mapping[str, Any] | None = None,
    objective: Any | None = None,
    method: str = "grid",
    max_cases: int | None = None,
    constraints: Any | None = None,
    scene: Any | None = None,
    camera: Any | None = None,
    asset_store: Any | None = None,
    seed: int = 0,
    top_k: int = 5,
    include_arrays: bool = False,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Optimize camera parameters from an image-sensor DB selection.

    This is the direct bridge from sensor DB selection to the FACA optimizer.
    The selected DB record is preserved in ``source_image_sensor_db`` and in the
    selected scenario dictionaries so RAW dataset exports keep provenance.
    """

    from .optimization import camerae2e_optimize_camera_parameters

    config = image_sensor_db_config(sensor_id_or_code, strategy=strategy, root=root)
    scenario = _merge_scenario_dicts(config["scenario"], base_scenario or {})
    result = camerae2e_optimize_camera_parameters(
        scenario,
        preset=preset,
        parameter_space=parameter_space,
        objective=objective,
        method=method,
        max_cases=max_cases,
        constraints=constraints,
        scene=scene,
        camera=camera,
        asset_store=asset_store,
        seed=seed,
        top_k=top_k,
        include_arrays=include_arrays,
    )
    result["source_image_sensor_db"] = {
        "schema_version": "camerae2e_optimization_source_image_sensor_db_v1",
        "sensor_id": config["sensor_id"],
        "code": config["code"],
        "manufacturer": config["manufacturer"],
        "device_name": config["device_name"],
        "strategy": config["strategy"],
        "readiness_tier": config["readiness_tier"],
        "policy": config["policy"],
        "record_summary": config["record_summary"],
    }
    return result


def image_sensor_db_summary(root: str | Path | None = None) -> dict[str, Any]:
    """Return compact image-sensor DB summary facts."""

    catalog = image_sensor_db_load(root)
    records = image_sensor_db_records(root=root)
    manufacturers: dict[str, int] = {}
    architectures: dict[str, int] = {}
    for record in records:
        manufacturers[record["manufacturer"]] = manufacturers.get(record["manufacturer"], 0) + 1
        arch = str(record.get("pixel_architecture") or "unknown")
        architectures[arch] = architectures.get(arch, 0) + 1
    return {
        "schema": catalog.get("schema"),
        "generated_at": catalog.get("generated_at"),
        "record_count": len(records),
        "manufacturer_count": len(manufacturers),
        "manufacturers": dict(sorted(manufacturers.items(), key=lambda item: (-item[1], item[0]))),
        "pixel_architectures": dict(
            sorted(architectures.items(), key=lambda item: (-item[1], item[0]))
        ),
        "catalog_path": str(image_sensor_db_catalog_path(root)),
    }


def _normalize_config_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hybrid": "hybrid",
        "combo": "hybrid",
        "combination": "hybrid",
        "db_lut_preferred": "hybrid",
        "lut": "lut_only",
        "lut_only": "lut_only",
        "db_only": "lut_only",
        "analytic": "analytic_only",
        "analytic_only": "analytic_only",
        "proxy": "analytic_only",
        "free_configuration": "analytic_only",
    }
    if normalized not in aliases:
        raise ValueError("strategy must be one of: hybrid, lut_only, analytic_only.")
    return aliases[normalized]


def _merge_scenario_dicts(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = _deep_copy_dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            nested = _deep_copy_dict(merged[key])
            nested.update(_deep_copy_dict(value))
            merged[key] = nested
        else:
            merged[key] = _jsonable_value(value)
    return merged


def _deep_copy_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(item) for key, item in value.items()}


def _analytic_sensor_config_from_record(
    record: Mapping[str, Any], *, include_ocl_proxy: bool
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    pixel_pitch_um = _finite_float(record.get("pixel_pitch_um"))
    if pixel_pitch_um is not None and pixel_pitch_um > 0.0:
        config["pixel_size"] = pixel_pitch_um * 1.0e-6

    cfa_preset = _cfa_preset_from_record(record)
    if cfa_preset is not None:
        config["cfa_preset"] = cfa_preset

    if include_ocl_proxy:
        ocl_shape = _ocl_group_shape_from_record(record)
        if ocl_shape is not None:
            config["ocl_group_shape"] = ocl_shape
            config["ocl_group_equalization"] = 1.0 if ocl_shape == "2x2" else 0.0
    return config


def _cfa_preset_from_record(record: Mapping[str, Any]) -> str | None:
    pattern = str(record.get("cfa_pattern") or "").strip().lower()
    if pattern in {"bayer", "rggb_bayer"}:
        return "bayer_rgb"
    if pattern in {"quad_bayer", "tetracell_bayer"}:
        return "quad_bayer_rgb"
    return None


def _ocl_group_shape_from_record(record: Mapping[str, Any]) -> str | None:
    pattern = str(record.get("cfa_pattern") or "").strip().lower()
    architecture = str(record.get("pixel_architecture") or "").strip().lower()
    if pattern in {"quad_bayer", "tetracell_bayer"}:
        return "2x2"
    if architecture in {"four_shared", "quad_pixel", "super_qpd"}:
        return "2x2"
    if architecture in {"standalone", "two_shared", "dual_pixel", "eight_shared"}:
        return "1x1"
    return None


def _record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sensor_id",
        "code",
        "manufacturer",
        "device_name",
        "pixel_pitch_um",
        "pixel_architecture",
        "cfa_pattern",
        "illumination",
        "microlens_type",
        "dti_type",
        "has_dti",
        "has_pdaf",
        "has_hdr",
        "has_lofic",
        "stack_config_path",
        "tcad_profile_path",
    )
    return {key: record.get(key) for key in keys}


def _jsonable_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in params.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def _path_string(value: Any) -> str:
    return str(Path(value).expanduser()) if value is not None else ""


def _path_exists(value: Any) -> bool:
    return value is not None and Path(value).expanduser().exists()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _normalize_record(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata = dict(raw.get("metadata", {}))
    specs = dict(raw.get("derived_specs", {}))
    generated = dict(raw.get("generated_files", {}))
    stack_path = generated.get("stack_config")
    tcad_path = generated.get("tcad_profile")
    sensor_id = _sensor_id(raw, stack_path)
    return {
        "sensor_id": sensor_id,
        "code": str(raw.get("code", metadata.get("code", ""))),
        "manufacturer": str(metadata.get("manufacturer", "Unknown")),
        "device_name": str(metadata.get("device_name", "")),
        "title": str(metadata.get("title", raw.get("description_excerpt", ""))),
        "report_type": str(metadata.get("report_type", "")),
        "analysis_year": metadata.get("analysis_year"),
        "pixel_pitch_um": specs.get("pixel_pitch_um"),
        "pixel_architecture": specs.get("pixel_architecture"),
        "cfa_pattern": specs.get("cfa_pattern"),
        "illumination": specs.get("illumination"),
        "microlens_type": specs.get("microlens_type"),
        "optical_stack_height_um": specs.get("optical_stack_height_um"),
        "active_si_thickness_um": specs.get("active_si_thickness_um"),
        "dti_type": specs.get("dti_type"),
        "has_dti": specs.get("has_dti"),
        "has_pdaf": specs.get("has_pdaf"),
        "has_hdr": specs.get("has_hdr"),
        "has_lofic": specs.get("has_lofic"),
        "resolution_mp": specs.get("resolution_mp"),
        "optical_format": specs.get("optical_format"),
        "stack_config_path": _resolve_catalog_path(stack_path, root),
        "tcad_profile_path": _resolve_catalog_path(tcad_path, root),
        "raw": raw,
        "db_root": str(root),
    }


def _resolve_catalog_path(value: Any, root: Path) -> str | None:
    """Resolve generated catalog paths without depending on the producer machine."""

    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        parts = path.parts
        if "sensor_db" in parts:
            path = Path(*parts[parts.index("sensor_db") + 1 :])
        elif path.exists():
            return str(path)
    return str((root / path).resolve())


def _sensor_id(raw: dict[str, Any], stack_path: str | None) -> str:
    if stack_path:
        return Path(stack_path).stem
    metadata = dict(raw.get("metadata", {}))
    parts = [raw.get("code", ""), metadata.get("manufacturer", ""), metadata.get("device_name", "")]
    return _slugify("_".join(str(part) for part in parts if part))


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return "_".join(part for part in cleaned.split())


imageSensorDBRoot = image_sensor_db_root  # noqa: N816
imageSensorDBCatalogPath = image_sensor_db_catalog_path  # noqa: N816
imageSensorDBLoad = image_sensor_db_load  # noqa: N816
imageSensorDBRecords = image_sensor_db_records  # noqa: N816
imageSensorDBGet = image_sensor_db_get  # noqa: N816
imageSensorDBParameters = image_sensor_db_parameters  # noqa: N816
imageSensorDBConfig = image_sensor_db_config  # noqa: N816
imageSensorDBOptimizeCameraParameters = image_sensor_db_optimize_camera_parameters  # noqa: N816
imageSensorDBSummary = image_sensor_db_summary  # noqa: N816
