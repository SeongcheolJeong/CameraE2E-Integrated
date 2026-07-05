"""System-level Field/Angle/Color/Artifact/Control Analysis helpers."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from .assets import AssetStore
from .camera import camera_compute, camera_create, camera_get, camera_set
from .db_catalog import camerae2e_db_lineage, camerae2e_db_summary, camerae2e_db_validate
from .fdtd_sensor import sensor_attach_fdtd_lut
from .hwisp import HWIspConfig, hw_isp_config, hw_isp_simulate_sequence
from .ip import ip_get
from .optics import si_synthetic
from .scene import scene_create, scene_get
from .sensor import mlens_create, mlens_set, sensor_get, sensor_set, sensor_vignetting
from .tcad_sensor import sensor_attach_tcad_lut
from .types import Camera, Scene


def camerae2e_run_scenario(
    scenario: Mapping[str, Any] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_arrays: bool = True,
) -> dict[str, Any]:
    """Run one CameraE2E FACA scenario and collect stage evidence."""

    store = asset_store or AssetStore.default()
    config = dict(scenario or {})
    scenario_name = str(config.get("name", "camerae2e_faca_scenario"))
    resolved_scene = _resolve_scene(scene if scene is not None else config.get("scene"), store)
    resolved_camera = _resolve_camera(camera if camera is not None else config.get("camera"), store)
    resolved_camera, parameter_lineage = _apply_physics_and_sensor_overrides(
        resolved_camera, config, store
    )

    hw_payload = dict(config.get("hw_isp", {})) if isinstance(config.get("hw_isp"), Mapping) else {}
    use_hw_isp = bool(hw_payload.get("enabled", False))
    hw_sequence = None
    if use_hw_isp:
        hw_config = _resolve_hw_config(hw_payload.get("config", hw_payload))
        nframes = int(hw_payload.get("nframes", 3))
        hw_sequence = hw_isp_simulate_sequence(
            resolved_camera,
            resolved_scene,
            hw_config,
            nframes=nframes,
            asset_store=store,
        )
        computed_camera = hw_sequence.frames[-1].camera
    else:
        computed_camera = camera_compute(resolved_camera.clone(), resolved_scene, asset_store=store)

    stages = _collect_stages(resolved_scene, computed_camera, include_arrays=include_arrays)
    metrics = _faca_metrics(stages, hw_sequence.aggregate if hw_sequence is not None else None)
    return {
        "schema_version": "camerae2e_faca_scenario_v1",
        "name": scenario_name,
        "seed": int(seed),
        "scenario": _jsonable(_scenario_without_objects(config)),
        "camera": computed_camera,
        "hw_isp_sequence": hw_sequence,
        "stages": stages,
        "metrics": metrics,
        "parameter_lineage": parameter_lineage,
        "artifact_lineage": _artifact_lineage_summary(),
    }


def camerae2e_run_sweep(
    base_scenario: Mapping[str, Any] | None = None,
    sweep_axes: Mapping[str, Iterable[Any]] | None = None,
    *,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Run a deterministic Cartesian FACA sweep."""

    axes = {str(key): list(values) for key, values in dict(sweep_axes or {}).items()}
    if not axes:
        result = camerae2e_run_scenario(
            base_scenario,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=seed,
            include_arrays=include_arrays,
        )
        return {
            "schema_version": "camerae2e_faca_sweep_v1",
            "seed": int(seed),
            "axes": {},
            "cases": [camerae2e_faca_report(result)],
        }

    cases = []
    keys = list(axes)
    for index, values in enumerate(itertools.product(*(axes[key] for key in keys))):
        scenario = _deep_dict(base_scenario or {})
        scenario["name"] = str(scenario.get("name", "camerae2e_faca_sweep"))
        axis_payload = {}
        for key, value in zip(keys, values, strict=True):
            _assign_axis_value(scenario, key, value)
            axis_payload[key] = value
        result = camerae2e_run_scenario(
            scenario,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=int(seed) + index,
            include_arrays=include_arrays,
        )
        report = camerae2e_faca_report(result)
        report["axis_values"] = _jsonable(axis_payload)
        cases.append(report)

    return {
        "schema_version": "camerae2e_faca_sweep_v1",
        "seed": int(seed),
        "axes": _jsonable(axes),
        "case_count": len(cases),
        "cases": cases,
    }


def camerae2e_faca_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe FACA report for a scenario or sweep case."""

    if result.get("schema_version") == "camerae2e_faca_sweep_v1":
        return _jsonable(result)
    return {
        "schema_version": "camerae2e_faca_report_v1",
        "name": str(result.get("name", "camerae2e_faca_scenario")),
        "seed": int(result.get("seed", 0)),
        "scenario": _jsonable(result.get("scenario", {})),
        "stage_summaries": {
            str(name): _jsonable({key: value for key, value in payload.items() if key != "array"})
            for name, payload in dict(result.get("stages", {})).items()
        },
        "metrics": _jsonable(result.get("metrics", {})),
        "parameter_lineage": _jsonable(result.get("parameter_lineage", [])),
        "artifact_lineage": _jsonable(result.get("artifact_lineage", {})),
    }


def _resolve_scene(source: Scene | str | Mapping[str, Any] | None, store: AssetStore) -> Scene:
    if isinstance(source, Scene):
        return source.clone()
    if isinstance(source, Mapping):
        scene_type = str(source.get("type", source.get("name", "uniform ee")))
        args = list(source.get("args", [16]))
        kwargs = dict(source.get("kwargs", {}))
        kwargs.setdefault("asset_store", store)
        return scene_create(scene_type, *args, **kwargs)
    if isinstance(source, str):
        return scene_create(source, asset_store=store)
    return scene_create("uniform ee", 16, asset_store=store)


def _resolve_camera(source: Camera | Mapping[str, Any] | None, store: AssetStore) -> Camera:
    if isinstance(source, Camera):
        return source.clone()
    if isinstance(source, Mapping):
        camera_type = str(source.get("type", source.get("name", "default")))
        args = list(source.get("args", []))
        created = camera_create(camera_type, *args, asset_store=store)
    else:
        created = camera_create(asset_store=store)
    if isinstance(created, list):
        if not created:
            raise ValueError("camera_create returned an empty camera list.")
        return created[0].clone()
    return created.clone()


def _apply_physics_and_sensor_overrides(
    camera: Camera, config: Mapping[str, Any], store: AssetStore
) -> tuple[Camera, list[dict[str, Any]]]:
    updated = camera.clone()
    lineage: list[dict[str, Any]] = []
    parameter_overrides = {}
    for bucket_name in ("parameters", "camera_parameters"):
        bucket = config.get(bucket_name)
        if isinstance(bucket, Mapping):
            parameter_overrides.update(dict(bucket))
    for key, value in parameter_overrides.items():
        special = _apply_special_camera_parameter(updated, key, value)
        if special is not None:
            updated, parameter_name, before, after = special
            lineage.append(
                _parameter_lineage_entry(
                    str(key),
                    parameter_name,
                    value,
                    before,
                    after,
                    status="applied",
                )
            )
            continue
        parameter_name = _camera_parameter_name(key)
        before = _safe_camera_get(updated, parameter_name)
        updated = camera_set(updated, parameter_name, value)
        after = _safe_camera_get(updated, parameter_name)
        lineage.append(
            _parameter_lineage_entry(
                str(key),
                parameter_name,
                value,
                before,
                after,
                status="applied",
            )
        )

    sensor = camera_get(updated, "sensor").clone()
    sensor_overrides = (
        dict(config.get("sensor", {})) if isinstance(config.get("sensor"), Mapping) else {}
    )
    delayed_sensor_overrides: list[tuple[Any, Any]] = []
    for key, value in sensor_overrides.items():
        if _is_sensor_vignetting_override(key):
            delayed_sensor_overrides.append((key, value))
            continue
        special = _apply_special_sensor_override(sensor, key, value, store)
        if special is not None:
            sensor, sensor_parameter, before, after = special
            lineage.append(
                _parameter_lineage_entry(
                    f"sensor.{str(key).replace(' ', '_')}",
                    sensor_parameter,
                    value,
                    before,
                    after,
                    status="applied",
                )
            )
            continue
        sensor_parameter = _sensor_override_parameter_name(key)
        if sensor_parameter is None:
            raise KeyError(f"Unsupported CameraE2E sensor override: {key!r}")
        before = _safe_sensor_get(sensor, sensor_parameter)
        sensor = sensor_set(sensor, sensor_parameter, _sensor_override_value(key, value))
        after = _safe_sensor_get(sensor, sensor_parameter)
        lineage.append(
            _parameter_lineage_entry(
                f"sensor.{str(key).replace(' ', '_')}",
                sensor_parameter,
                value,
                before,
                after,
                status="applied",
            )
        )
    for key, value in delayed_sensor_overrides:
        special = _apply_special_sensor_override(sensor, key, value, store)
        if special is None:
            raise KeyError(f"Unsupported CameraE2E sensor override: {key!r}")
        sensor, sensor_parameter, before, after = special
        lineage.append(
            _parameter_lineage_entry(
                f"sensor.{str(key).replace(' ', '_')}",
                sensor_parameter,
                value,
                before,
                after,
                status="applied",
            )
        )

    fdtd = dict(config.get("fdtd", {})) if isinstance(config.get("fdtd"), Mapping) else {}
    if fdtd.get("lut") is not None:
        fdtd_kwargs = {key: value for key, value in fdtd.items() if key not in {"lut", "enabled"}}
        sensor = sensor_attach_fdtd_lut(sensor, fdtd["lut"], **fdtd_kwargs)
        lineage.append(
            _parameter_lineage_entry(
                "fdtd.lut",
                "fdtd lut",
                fdtd["lut"],
                None,
                {"attached": True, "options": fdtd_kwargs},
                status="attached",
            )
        )

    tcad = dict(config.get("tcad", {})) if isinstance(config.get("tcad"), Mapping) else {}
    if tcad.get("db") is not None:
        tcad_kwargs = {key: value for key, value in tcad.items() if key not in {"db", "enabled"}}
        if "collection_mode" in tcad_kwargs and "mode" not in tcad_kwargs:
            tcad_kwargs["mode"] = tcad_kwargs.pop("collection_mode")
        sensor = sensor_attach_tcad_lut(sensor, tcad["db"], **tcad_kwargs)
        lineage.append(
            _parameter_lineage_entry(
                "tcad.db",
                "tcad db",
                tcad["db"],
                None,
                {"attached": True, "options": tcad_kwargs},
                status="attached",
            )
        )

    updated.fields["sensor"] = sensor
    return updated, lineage


def _apply_special_camera_parameter(
    camera: Camera, key: Any, value: Any
) -> tuple[Camera, str, Any, Any] | None:
    normalized = str(key).strip().replace("_", " ").lower()
    if normalized in {
        "optics.psf radius um",
        "optics.si psf radius um",
        "optics.shift invariant psf radius um",
    }:
        radius_um = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if radius_um <= 0.0:
            raise ValueError("optics.si_psf_radius_um must be positive.")
        before = _safe_camera_get(camera, "optics model")
        oi = camera_get(camera, "oi")
        pillbox_diameter_mm = 2.0 * radius_um * 1e-3
        optics = si_synthetic("pillbox", oi, pillbox_diameter_mm)
        updated = camera_set(camera, "optics", optics)
        after = {
            "optics_model": _safe_camera_get(updated, "optics model"),
            "proxy_model": "shift_invariant_pillbox_psf",
            "radius_um": radius_um,
            "diameter_mm": pillbox_diameter_mm,
            "truth_boundary": (
                "synthetic_shift_invariant_proxy_not_rayoptics_or_wave_optics_signoff"
            ),
        }
        return updated, "optics si psf radius um", before, after
    return None


def _apply_special_sensor_override(
    sensor: Any, key: Any, value: Any, store: AssetStore
) -> tuple[Any, str, Any, Any] | None:
    normalized = str(key).strip().replace("_", " ").lower()
    if normalized in {
        "cfa preset",
        "cfa pattern preset",
        "cfa type",
        "cfa layout",
    }:
        before = {
            "pattern": _safe_sensor_get(sensor, "pattern"),
            "cfa_name": _safe_sensor_get(sensor, "cfa name"),
        }
        updated = sensor_set(sensor, "cfa preset", value)
        after = {
            "preset": value,
            "pattern": _safe_sensor_get(updated, "pattern"),
            "cfa_name": _safe_sensor_get(updated, "cfa name"),
        }
        return updated, "sensor cfa preset", before, after
    if normalized in {
        "ocl group proxy",
        "shared ocl aperture",
        "shared ocl aperture proxy",
    }:
        before = _safe_sensor_get(sensor, "ocl group proxy")
        updated = sensor_set(sensor, "ocl group proxy", value)
        after = _safe_sensor_get(updated, "ocl group proxy")
        return updated, "sensor ocl group proxy", before, after
    if normalized in {
        "ocl group shape",
        "shared ocl shape",
        "shared ocl aperture shape",
    }:
        before = _safe_sensor_get(sensor, "ocl group proxy")
        updated = sensor_set(sensor, "ocl group shape", value)
        after = _safe_sensor_get(updated, "ocl group proxy")
        return updated, "sensor ocl group shape", before, after
    if normalized in {
        "ocl group equalization",
        "ocl equalization",
        "shared ocl equalization",
        "shared ocl aperture equalization",
    }:
        before = _safe_sensor_get(sensor, "ocl group proxy")
        updated = sensor_set(sensor, "ocl group equalization", value)
        after = _safe_sensor_get(updated, "ocl group proxy")
        return updated, "sensor ocl group equalization", before, after
    if normalized in {
        "binning factor",
        "pixel binning factor",
        "readout binning factor",
    }:
        before = _safe_sensor_get(sensor, "sensor compute method")
        payload = _sensor_binning_factor_value(value)
        updated = sensor_set(sensor, "sensor compute method", payload)
        after = _safe_sensor_get(updated, "sensor compute method")
        return updated, "sensor compute method", before, after
    if normalized in {
        "ocl vignetting",
        "microlens vignetting",
        "pixel vignetting",
        "vignetting",
    }:
        before = {
            "vignetting": _safe_sensor_get(sensor, "vignetting"),
            "etendue": _array_summary(_safe_sensor_get(sensor, "etendue")),
        }
        updated = sensor_vignetting(sensor, value, asset_store=store)
        updated = sensor_set(updated, "vignetting", _sensor_vignetting_value(value))
        after = {
            "vignetting": _safe_sensor_get(updated, "vignetting"),
            "vignetting_name": _safe_sensor_get(updated, "vignetting name"),
            "etendue": _array_summary(_safe_sensor_get(updated, "etendue")),
        }
        return updated, "sensor ocl vignetting", before, after
    microlens_parameter = _microlens_override_parameter_name(normalized)
    if microlens_parameter is None:
        return None
    before = _safe_sensor_get(sensor, "microlens")
    microlens = before if isinstance(before, Mapping) else mlens_create(sensor, asset_store=store)
    if normalized in {"ocl focal length um", "microlens focal length um"}:
        updated_microlens = mlens_set(microlens, microlens_parameter, value, "microns")
    else:
        updated_microlens = mlens_set(microlens, microlens_parameter, value)
    updated = sensor_set(sensor, "microlens", updated_microlens)
    updated.fields["etendue"] = None
    after = _safe_sensor_get(updated, "microlens")
    return updated, f"microlens {microlens_parameter}", before, after


def _microlens_override_parameter_name(normalized: str) -> str | None:
    if normalized in {
        "ocl fnumber",
        "ocl f number",
        "microlens fnumber",
        "microlens f number",
    }:
        return "microlens fnumber"
    if normalized in {
        "ocl focal length um",
        "microlens focal length um",
    }:
        return "microlens focal length"
    if normalized in {
        "ocl refractive index",
        "microlens refractive index",
    }:
        return "microlens refractive index"
    if normalized in {
        "ocl chief ray angle deg",
        "microlens chief ray angle deg",
        "microlens ray angle deg",
    }:
        return "chief ray angle"
    return None


def _is_sensor_vignetting_override(key: Any) -> bool:
    normalized = str(key).strip().replace("_", " ").lower()
    return normalized in {
        "ocl vignetting",
        "microlens vignetting",
        "pixel vignetting",
        "vignetting",
    }


def _camera_parameter_name(key: Any) -> str:
    value = str(key).strip().replace("_", " ")
    if "." in value:
        prefix, remainder = value.split(".", 1)
        return f"{prefix} {remainder.replace('.', ' ')}"
    return value


def _sensor_override_parameter_name(key: Any) -> str | None:
    normalized = str(key).replace("_", " ").lower()
    if normalized in {"noise flag", "noise"}:
        return "noise flag"
    if normalized in {"integration time", "integration"}:
        return "integration time"
    if normalized in {"exposure duration", "exposure time", "exposure time s"}:
        return "exposure duration"
    if normalized in {"analog gain", "gain"}:
        return "analog gain"
    if normalized in {"pixel size", "pixel size m", "pixel pitch", "pixel pitch m"}:
        return "pixel size"
    if normalized in {"pixel fill factor", "fill factor", "pd fill factor"}:
        return "pixel fill factor"
    if normalized in {"n samples per pixel", "samples per pixel", "spatial samples per pixel"}:
        return "n samples per pixel"
    if normalized in {"cfa pattern", "pattern", "pattern and size"}:
        return "pattern and size"
    if normalized in {"cfa preset", "cfa pattern preset", "cfa type", "cfa layout"}:
        return "cfa preset"
    if normalized in {"cfa", "color filter array"}:
        return "cfa"
    if normalized in {"filter names", "filter name"}:
        return "filter names"
    if normalized in {"filter spectra", "color filters", "filter transmissivities"}:
        return "filter spectra"
    if normalized in {"pixel read noise v", "read noise", "read noise v"}:
        return "pixel read noise"
    if normalized in {"pixel dark voltage", "dark voltage", "dark voltage v per sec"}:
        return "pixel dark voltage"
    if normalized in {"pixel voltage swing", "voltage swing"}:
        return "pixel voltage swing"
    if normalized in {"pixel conversion gain", "conversion gain"}:
        return "pixel conversion gain"
    if normalized in {"sensor compute method", "compute method"}:
        return "sensor compute method"
    if normalized in {"binning", "binning method", "pixel binning", "pixel binning method"}:
        return "sensor compute method"
    if normalized in {"microlens", "micro lens", "ocl"}:
        return "microlens"
    if normalized in {"microlens offset", "microlens offset microns", "ocl offset um"}:
        return "microlens offset"
    if normalized in {"ocl group proxy", "shared ocl aperture", "shared ocl aperture proxy"}:
        return "ocl group proxy"
    if normalized in {"ocl group shape", "shared ocl shape", "shared ocl aperture shape"}:
        return "ocl group shape"
    if normalized in {
        "ocl group equalization",
        "ocl equalization",
        "shared ocl equalization",
        "shared ocl aperture equalization",
    }:
        return "ocl group equalization"
    return None


def _sensor_override_value(key: Any, value: Any) -> Any:
    normalized = str(key).replace("_", " ").lower()
    if normalized not in {"binning", "binning method", "pixel binning", "pixel binning method"}:
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("name", "binning")
        return payload
    normalized_value = str(value).strip().lower()
    if normalized_value in {"", "off", "none", "default", "disabled", "false", "0"}:
        return None
    return {"name": "binning", "method": str(value)}


def _sensor_binning_factor_value(value: Any) -> dict[str, Any] | None:
    factor = int(np.asarray(value, dtype=float).reshape(-1)[0])
    if factor <= 1:
        return None
    if factor != 2:
        raise ValueError("sensor.binning_factor currently supports 1/off or the legacy 2x proxy.")
    return {"name": "binning", "method": "kodak2008", "factor": factor}


def _sensor_vignetting_value(value: Any) -> Any:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", " ")
        if normalized in {"off", "skip", "none", "disabled", "false", "0"}:
            return 0
        if normalized in {"bare", "no microlens", "nomicrolens", "1"}:
            return 1
        if normalized in {"centered", "2"}:
            return 2
        if normalized in {"optimal", "optimized", "3"}:
            return 3
    return value


def _array_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return {"shape": list(array.shape), "available": False}
    return {
        "shape": list(array.shape),
        "available": True,
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
    }


def _parameter_lineage_entry(
    path: str,
    parameter: str,
    requested: Any,
    before: Any,
    after: Any,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "parameter": str(parameter),
        "requested": _jsonable(requested),
        "before": _jsonable(before),
        "after": _jsonable(after),
        "status": str(status),
    }


def _resolve_hw_config(raw: Any) -> HWIspConfig:
    if isinstance(raw, HWIspConfig):
        return raw
    if isinstance(raw, Mapping):
        return hw_isp_config(
            **{
                key: value
                for key, value in raw.items()
                if key not in {"enabled", "nframes", "config"}
            }
        )
    return hw_isp_config()


def _collect_stages(
    scene: Scene, camera: Camera, *, include_arrays: bool
) -> dict[str, dict[str, Any]]:
    oi = camera_get(camera, "oi")
    sensor = camera_get(camera, "sensor")
    ip = camera_get(camera, "ip")
    return {
        "scene_photons": _stage_payload(
            _safe_get(lambda: scene_get(scene, "photons")), include_arrays=include_arrays
        ),
        "oi_photons": _stage_payload(oi.data.get("photons"), include_arrays=include_arrays),
        "sensor_raw": _stage_payload(_sensor_raw(sensor), include_arrays=include_arrays),
        "sensor_digital": _stage_payload(_sensor_digital(sensor), include_arrays=include_arrays),
        "ip_result": _stage_payload(
            _safe_get(lambda: ip_get(ip, "result")), include_arrays=include_arrays
        ),
        "ip_srgb": _stage_payload(
            _safe_get(lambda: ip_get(ip, "srgb")), include_arrays=include_arrays
        ),
    }


def _sensor_raw(sensor: Any) -> Any:
    for key in ("volts", "dv", "electrons"):
        if key in getattr(sensor, "data", {}):
            return sensor.data[key]
    try:
        return sensor_get(sensor, "volts")
    except Exception:
        return None


def _sensor_digital(sensor: Any) -> Any:
    for key in ("digital_values", "dv"):
        if key in getattr(sensor, "data", {}):
            return sensor.data[key]
    try:
        return sensor_get(sensor, "digital values")
    except Exception:
        return None


def _stage_payload(values: Any, *, include_arrays: bool) -> dict[str, Any]:
    if values is None:
        return {"available": False, "shape": [], "dtype": None}
    array = np.asarray(values)
    payload: dict[str, Any] = {
        "available": bool(array.size),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size:
        finite = np.asarray(array, dtype=float)
        payload.update(
            {
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "std": float(np.nanstd(finite)),
            }
        )
    if include_arrays:
        payload["array"] = array.copy()
    return payload


def _faca_metrics(
    stages: Mapping[str, Mapping[str, Any]], hw_aggregate: Mapping[str, Any] | None
) -> dict[str, Any]:
    raw = dict(stages.get("sensor_raw", {}))
    rgb = dict(stages.get("ip_result", {}))
    rgb_array = np.asarray(rgb.get("array", np.empty(0)), dtype=float)
    raw_array = np.asarray(raw.get("array", np.empty(0)), dtype=float)
    raw_std = float(np.nanstd(raw_array)) if raw_array.size else raw.get("std")
    rgb_std = float(np.nanstd(rgb_array)) if rgb_array.size else rgb.get("std")
    rgb_mean = float(np.nanmean(rgb_array)) if rgb_array.size else rgb.get("mean")
    metrics: dict[str, Any] = {
        "field": {"available": bool(stages.get("oi_photons", {}).get("available", False))},
        "angle": {
            "available": False,
            "note": "CRA/field axes are recorded through attached FDTD scenario parameters.",
        },
        "color": {"rgb_mean": None if rgb_mean is None else float(rgb_mean)},
        "artifact": {
            "raw_std": None if raw_std is None else float(raw_std),
            "rgb_std": None if rgb_std is None else float(rgb_std),
            "rgb_clip_fraction": _clip_fraction(rgb_array) if rgb_array.size else None,
        },
        "control": dict(hw_aggregate or {}),
    }
    return metrics


def _clip_fraction(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean((values <= 0.0) | (values >= 1.0)))


def _artifact_lineage_summary() -> dict[str, Any]:
    validation = camerae2e_db_validate()
    summary = camerae2e_db_summary()
    lineage = {}
    for name in (
        "fdtd_sensor_lut_active",
        "tcad_sensor_db_active",
        "lens_patents_active",
        "hwisp_parameter_profiles",
    ):
        try:
            lineage[name] = camerae2e_db_lineage(name)
        except Exception as exc:
            lineage[name] = {"error": str(exc)}
    return {"summary": summary, "validation": validation, "lineage": lineage}


def _assign_axis_value(scenario: dict[str, Any], key: str, value: Any) -> None:
    parts = str(key).split(".", 1)
    if len(parts) == 2 and parts[0] in {"sensor", "fdtd", "tcad", "hw_isp"}:
        bucket = scenario.setdefault(parts[0], {})
        if not isinstance(bucket, dict):
            bucket = {}
            scenario[parts[0]] = bucket
        if parts[0] == "hw_isp" and parts[1] != "enabled":
            bucket.setdefault("enabled", True)
        bucket[parts[1]] = value
        return
    scenario.setdefault("parameters", {})[key] = value


def _scenario_without_objects(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in config.items():
        if isinstance(value, (Camera, Scene)):
            payload[key] = {"type": type(value).__name__, "name": value.name}
        else:
            payload[key] = value
    return payload


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _safe_get(callback: Any) -> Any:
    try:
        return callback()
    except Exception:
        return None


def _safe_camera_get(camera: Camera, parameter: str) -> Any:
    try:
        return camera_get(camera, parameter)
    except Exception:
        return None


def _safe_sensor_get(sensor: Any, parameter: str) -> Any:
    try:
        return sensor_get(sensor, parameter)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if key not in {"array", "camera", "hw_isp_sequence"}
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


cameraE2ERunScenario = camerae2e_run_scenario  # noqa: N816
cameraE2ERunSweep = camerae2e_run_sweep  # noqa: N816
cameraE2EFACAReport = camerae2e_faca_report  # noqa: N816
