"""DEVSIM/TCAD collection-response helpers for sensor simulation.

This module is intentionally separate from :mod:`pyisetcam.fdtd_sensor`.
FDTD describes optical absorption in silicon; TCAD/DEVSIM describes how that
generated charge is collected.  Both are optional LUT layers, so normal sensor
computation remains unchanged unless a caller explicitly attaches them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fdtd_sensor import FDTDSensorLUT, fdtd_sensor_config


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FDTD_ROOT = _REPO_ROOT / "simulations/fdtd_tcad"
_DEFAULT_CAMERA_DB_ROOT_ENV = "PYISETCAM_CAMERA_DB_ROOT"
_DEFAULT_GENERATION_MAP = Path("runs/fdtd_to_tcad_generation_2d_cra_smoke/tcad_generation_map_2d.npz")
_DEFAULT_COLLECTION_SUMMARIES = (
    Path("runs/devsim_split_pd_2d_fdtd_map_proxy_center_smoke/summary.json"),
    Path("runs/devsim_split_pd_2d_fdtd_map_proxy_edge20x_smoke/summary.json"),
)
_DEFAULT_ACCURACY_GATE = Path("runs/tcad_accuracy_gate_reference_profile/tcad_accuracy_gate.json")


@dataclass(frozen=True)
class TCADGenerationMap:
    """Loaded FDTD-to-TCAD generation map, usually ``G(x, depth)``."""

    source_path: Path
    schema: str
    generation_cm3_s: np.ndarray
    x_um: np.ndarray
    depth_um_from_si_top: np.ndarray
    cases: np.ndarray
    wavelength_nm: np.ndarray
    field_x_norm: np.ndarray
    field_z_norm: np.ndarray
    cra_x_deg: np.ndarray
    cra_z_deg: np.ndarray
    absorption_fraction_per_um2: np.ndarray | None = None
    y_um: np.ndarray | None = None
    color_channel: str | None = None
    incident_photon_flux_cm2_s: float | None = None
    method: str | None = None


@dataclass(frozen=True)
class TCADCollectionSummary:
    """Loaded DEVSIM split-PD collection summary."""

    source_path: Path
    schema: str
    devsim_version: str | None
    generation_source: str
    electrical_model: str
    case: str | None
    wavelength_nm: float | None
    left_photo_delta_a_per_cm: float
    right_photo_delta_a_per_cm: float
    photo_split_phase_x_proxy: float
    terminal_current_balance_illuminated_a_per_cm: float
    config: dict[str, Any] = field(default_factory=dict)
    dark: dict[str, Any] = field(default_factory=dict)
    illuminated: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_photo_delta_a_per_cm(self) -> float:
        return float(self.left_photo_delta_a_per_cm + self.right_photo_delta_a_per_cm)


@dataclass(frozen=True)
class TCADAccuracyGate:
    """Loaded TCAD accuracy-gate result."""

    source_path: Path
    schema: str
    profile_name: str | None
    framework_ready: bool
    accuracy_ready: bool
    accuracy_blocking_failure_count: int
    framework_blocking_failure_count: int
    checks: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TCADSensorDB:
    """Combined DEVSIM/TCAD sensor collection database."""

    generation_map: TCADGenerationMap | None = None
    collection_summaries: tuple[TCADCollectionSummary, ...] = ()
    accuracy_gate: TCADAccuracyGate | None = None
    source_root: Path | None = None


def tcad_sensor_default_root() -> Path:
    """Return the configured external FDTD/TCAD workspace root."""

    explicit_fdtd = os.environ.get("PYISETCAM_FDTD_ROOT")
    if explicit_fdtd:
        return Path(explicit_fdtd).expanduser()
    camera_db_root = os.environ.get(_DEFAULT_CAMERA_DB_ROOT_ENV)
    if camera_db_root:
        return Path(camera_db_root).expanduser() / "fdtd_tcad"
    for candidate in _default_camera_db_fdtd_roots():
        if candidate.exists():
            return candidate
    return _DEFAULT_FDTD_ROOT.expanduser()


def _default_camera_db_fdtd_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "camerae2e_db/fdtd_tcad",
        repo_root / "simulations/fdtd_tcad",
        repo_root.parent / "CameraE2E-DB/fdtd_tcad",
    ]


def tcad_sensor_default_paths(root: str | Path | None = None) -> dict[str, Any]:
    """Return the default FDTD/DEVSIM artifact paths used by the sensor DB."""

    resolved = Path(root).expanduser() if root is not None else tcad_sensor_default_root()
    return {
        "root": resolved,
        "generation_map_path": resolved / _DEFAULT_GENERATION_MAP,
        "collection_summary_paths": [resolved / item for item in _DEFAULT_COLLECTION_SUMMARIES],
        "accuracy_gate_path": resolved / _DEFAULT_ACCURACY_GATE,
    }


def tcad_generation_map_load(path: str | Path) -> TCADGenerationMap:
    """Load a ``tcad_generation_map_2d.npz`` artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    data = np.load(resolved, allow_pickle=True)
    generation = np.asarray(data["generation_cm3_s"], dtype=float)
    if generation.ndim != 3:
        raise ValueError("TCAD generation map must have shape [case, x, depth].")
    return TCADGenerationMap(
        source_path=resolved,
        schema=_npz_scalar_string(data, "schema", "unknown"),
        generation_cm3_s=generation,
        absorption_fraction_per_um2=_npz_optional_array(data, "absorption_fraction_per_um2"),
        x_um=np.asarray(data["x_um"], dtype=float).reshape(-1),
        depth_um_from_si_top=np.asarray(data["depth_um_from_si_top"], dtype=float).reshape(-1),
        y_um=_npz_optional_array(data, "y_um"),
        cases=np.asarray(data["case"]).astype(str).reshape(-1),
        wavelength_nm=np.asarray(data["wavelength_nm"], dtype=float).reshape(-1),
        field_x_norm=np.asarray(data["field_x_norm"], dtype=float).reshape(-1),
        field_z_norm=np.asarray(data["field_z_norm"], dtype=float).reshape(-1),
        cra_x_deg=np.asarray(data["cra_x_deg"], dtype=float).reshape(-1),
        cra_z_deg=np.asarray(data["cra_z_deg"], dtype=float).reshape(-1),
        color_channel=_npz_scalar_string(data, "color_channel", None),
        incident_photon_flux_cm2_s=_npz_optional_float(data, "incident_photon_flux_cm2_s"),
        method=_npz_scalar_string(data, "method", None),
    )


def tcad_collection_summary_load(path: str | Path) -> TCADCollectionSummary:
    """Load a DEVSIM split-PD ``summary.json`` artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    config = dict(payload.get("config", {}))
    return TCADCollectionSummary(
        source_path=resolved,
        schema=str(payload.get("schema", "unknown")),
        devsim_version=None if payload.get("devsim_version") is None else str(payload.get("devsim_version")),
        generation_source=str(payload.get("generation_source", "")),
        electrical_model=str(payload.get("electrical_model", "")),
        case=None if config.get("generation_profile_case") is None else str(config.get("generation_profile_case")),
        wavelength_nm=None
        if config.get("generation_profile_wavelength_nm") is None
        else float(config.get("generation_profile_wavelength_nm")),
        left_photo_delta_a_per_cm=float(payload.get("left_photo_delta_a_per_cm", 0.0)),
        right_photo_delta_a_per_cm=float(payload.get("right_photo_delta_a_per_cm", 0.0)),
        photo_split_phase_x_proxy=float(payload.get("photo_split_phase_x_proxy", 0.0)),
        terminal_current_balance_illuminated_a_per_cm=float(
            payload.get("terminal_current_balance_illuminated_a_per_cm", 0.0)
        ),
        config=config,
        dark=dict(payload.get("dark", {})),
        illuminated=dict(payload.get("illuminated", {})),
        notes=[str(item) for item in payload.get("notes", [])],
    )


def tcad_accuracy_gate_load(path: str | Path) -> TCADAccuracyGate:
    """Load a ``tcad_accuracy_gate.json`` artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return TCADAccuracyGate(
        source_path=resolved,
        schema=str(payload.get("schema", "unknown")),
        profile_name=None if payload.get("profile_name") is None else str(payload.get("profile_name")),
        framework_ready=bool(payload.get("framework_ready", False)),
        accuracy_ready=bool(payload.get("accuracy_ready", False)),
        accuracy_blocking_failure_count=int(payload.get("accuracy_blocking_failure_count", 0)),
        framework_blocking_failure_count=int(payload.get("framework_blocking_failure_count", 0)),
        checks=[dict(item) for item in payload.get("checks", [])],
        inputs=dict(payload.get("inputs", {})),
    )


def tcad_sensor_db_load(
    *,
    generation_map_path: str | Path | None = None,
    collection_summary_paths: Sequence[str | Path] | None = None,
    accuracy_gate_path: str | Path | None = None,
    root: str | Path | None = None,
) -> TCADSensorDB:
    """Load the default or explicitly supplied TCAD sensor collection DB."""

    defaults = tcad_sensor_default_paths(root)
    source_root = Path(defaults["root"])
    generation_path = Path(generation_map_path) if generation_map_path is not None else defaults["generation_map_path"]
    gate_path = Path(accuracy_gate_path) if accuracy_gate_path is not None else defaults["accuracy_gate_path"]
    summary_paths = (
        [Path(item) for item in collection_summary_paths]
        if collection_summary_paths is not None
        else [Path(item) for item in defaults["collection_summary_paths"]]
    )

    generation_map = tcad_generation_map_load(generation_path) if generation_path.exists() else None
    summaries = tuple(tcad_collection_summary_load(path) for path in summary_paths if path.exists())
    accuracy_gate = tcad_accuracy_gate_load(gate_path) if gate_path.exists() else None
    if generation_map is None and not summaries and accuracy_gate is None:
        raise FileNotFoundError(f"No TCAD sensor DB artifacts found under {source_root}")
    return TCADSensorDB(generation_map=generation_map, collection_summaries=summaries, accuracy_gate=accuracy_gate, source_root=source_root)


def tcad_sensor_validate(db: TCADSensorDB | Mapping[str, Any]) -> dict[str, Any]:
    """Validate TCAD sensor DB artifacts and separate framework from accuracy readiness."""

    loaded = _config_db(db) if isinstance(db, Mapping) else db
    issues: list[str] = []
    warnings: list[str] = []

    generation_check = _validate_generation_map(loaded.generation_map)
    collection_check = _validate_collection_summaries(loaded.collection_summaries, loaded.generation_map, loaded.source_root)
    accuracy_check = _validate_accuracy_gate(loaded.accuracy_gate)
    for check in (generation_check, collection_check, accuracy_check):
        issues.extend(str(item) for item in check.get("issues", []))
        warnings.extend(str(item) for item in check.get("warnings", []))

    framework_ready = not issues and bool(collection_check.get("ok", False))
    if loaded.accuracy_gate is not None:
        framework_ready = framework_ready and bool(loaded.accuracy_gate.framework_ready)
    accuracy_ready = framework_ready and bool(loaded.accuracy_gate and loaded.accuracy_gate.accuracy_ready)
    status = "accuracy-ready" if accuracy_ready else "proxy-framework" if framework_ready else "invalid"
    return {
        "ok": framework_ready,
        "status": status,
        "framework_ready": framework_ready,
        "accuracy_ready": accuracy_ready,
        "issues": issues,
        "warnings": warnings,
        "checks": {
            "generation_map": generation_check,
            "collection_summaries": collection_check,
            "accuracy_gate": accuracy_check,
        },
    }


def tcad_sensor_summary(db: TCADSensorDB | Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe summary of a TCAD sensor DB."""

    loaded = _config_db(db) if isinstance(db, Mapping) else db
    validation = tcad_sensor_validate(loaded)
    generation = loaded.generation_map
    return {
        "source_root": None if loaded.source_root is None else str(loaded.source_root),
        "status": validation["status"],
        "framework_ready": validation["framework_ready"],
        "accuracy_ready": validation["accuracy_ready"],
        "generation_map": None
        if generation is None
        else {
            "source_path": str(generation.source_path),
            "schema": generation.schema,
            "shape": list(generation.generation_cm3_s.shape),
            "cases": [str(item) for item in generation.cases],
            "wavelength_nm": [float(item) for item in generation.wavelength_nm],
            "cra_x_deg": [float(item) for item in generation.cra_x_deg],
            "generation_min": float(np.nanmin(generation.generation_cm3_s)),
            "generation_max": float(np.nanmax(generation.generation_cm3_s)),
            "color_channel": generation.color_channel,
        },
        "collection_summaries": [
            {
                "source_path": str(item.source_path),
                "case": item.case,
                "generation_source": item.generation_source,
                "electrical_model": item.electrical_model,
                "left_photo_delta_a_per_cm": item.left_photo_delta_a_per_cm,
                "right_photo_delta_a_per_cm": item.right_photo_delta_a_per_cm,
                "total_photo_delta_a_per_cm": item.total_photo_delta_a_per_cm,
                "photo_split_phase_x_proxy": item.photo_split_phase_x_proxy,
                "terminal_current_balance_illuminated_a_per_cm": item.terminal_current_balance_illuminated_a_per_cm,
            }
            for item in loaded.collection_summaries
        ],
        "accuracy_gate": None
        if loaded.accuracy_gate is None
        else {
            "source_path": str(loaded.accuracy_gate.source_path),
            "profile_name": loaded.accuracy_gate.profile_name,
            "framework_ready": loaded.accuracy_gate.framework_ready,
            "accuracy_ready": loaded.accuracy_gate.accuracy_ready,
            "accuracy_blocking_failure_count": loaded.accuracy_gate.accuracy_blocking_failure_count,
            "framework_blocking_failure_count": loaded.accuracy_gate.framework_blocking_failure_count,
        },
    }


def tcad_sensor_collection_efficiency(
    db: TCADSensorDB | Mapping[str, Any],
    *,
    case: str | None = None,
    normalize_to_center: bool = True,
) -> float:
    """Return a collection-efficiency multiplier from DEVSIM photo current."""

    loaded = _config_db(db) if isinstance(db, Mapping) else db
    selected = _select_collection_summary(loaded, case=case)
    if selected is None:
        return 1.0
    value = max(selected.total_photo_delta_a_per_cm, 0.0)
    if not normalize_to_center:
        return value
    center = _select_collection_summary(loaded, case="center") or loaded.collection_summaries[0]
    center_value = max(center.total_photo_delta_a_per_cm, 1e-30)
    return float(value / center_value)


def tcad_sensor_split_phase(db: TCADSensorDB | Mapping[str, Any], *, case: str | None = None) -> float:
    """Return the DEVSIM split-PD phase proxy for a selected case."""

    loaded = _config_db(db) if isinstance(db, Mapping) else db
    selected = _select_collection_summary(loaded, case=case)
    return 0.0 if selected is None else float(selected.photo_split_phase_x_proxy)


def tcad_sensor_generation_map_slice(
    db: TCADSensorDB | Mapping[str, Any],
    *,
    case: str | None = None,
    wavelength_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``x_um``, ``depth_um``, and selected ``G(x, depth)`` map."""

    loaded = _config_db(db) if isinstance(db, Mapping) else db
    generation = loaded.generation_map
    if generation is None:
        return np.empty(0), np.empty(0), np.empty((0, 0))
    index = _select_generation_index(generation, case=case, wavelength_nm=wavelength_nm)
    return (
        generation.x_um.copy(),
        generation.depth_um_from_si_top.copy(),
        np.asarray(generation.generation_cm3_s[index], dtype=float).copy(),
    )


def tcad_sensor_config(
    db: TCADSensorDB | Mapping[str, Any] | None = None,
    *,
    enabled: bool = True,
    mode: str = "collection",
    case: str | None = None,
    normalize_to_center: bool = True,
    collection_strength: float = 1.0,
    allow_proxy_accuracy: bool = True,
) -> dict[str, Any]:
    """Build a sensor-storable TCAD collection-response configuration."""

    if isinstance(db, Mapping):
        config = dict(db)
        config.setdefault("enabled", enabled)
        config.setdefault("mode", mode)
        config.setdefault("case", case)
        config.setdefault("normalize_to_center", normalize_to_center)
        config.setdefault("collection_strength", collection_strength)
        config.setdefault("allow_proxy_accuracy", allow_proxy_accuracy)
        return config
    return {
        "enabled": bool(enabled),
        "db": db,
        "mode": str(mode),
        "case": case,
        "normalize_to_center": bool(normalize_to_center),
        "collection_strength": float(collection_strength),
        "allow_proxy_accuracy": bool(allow_proxy_accuracy),
    }


def tcad_sensor_apply_collection_response(values: Any, config: Mapping[str, Any] | None) -> np.ndarray:
    """Apply the optional DEVSIM collection-efficiency multiplier."""

    array = np.asarray(values, dtype=float)
    if not _config_enabled(config) or not _mode_has(config, "collection"):
        return array.copy()
    db = _config_db(config)
    validation = tcad_sensor_validate(db)
    if not bool(config.get("allow_proxy_accuracy", True)) and not bool(validation.get("accuracy_ready", False)):
        raise ValueError("TCAD sensor DB is not accuracy-ready; set allow_proxy_accuracy=True for proxy simulation.")
    scale = tcad_sensor_collection_efficiency(
        db,
        case=config.get("case"),
        normalize_to_center=bool(config.get("normalize_to_center", True)),
    )
    strength = float(config.get("collection_strength", 1.0))
    effective_scale = 1.0 + (strength * (scale - 1.0))
    return np.asarray(array * effective_scale, dtype=float)


def sensor_attach_tcad_lut(sensor: Any, db: TCADSensorDB | Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
    """Attach a TCAD collection DB config to a sensor object."""

    updated = sensor.clone()
    updated.fields["tcad_sensor"] = tcad_sensor_config(db, **kwargs)
    return updated


def sensor_attach_physics_lut(
    sensor: Any,
    *,
    fdtd_lut: FDTDSensorLUT | str | Path | Mapping[str, Any] | None = None,
    tcad_db: TCADSensorDB | Mapping[str, Any] | None = None,
    fdtd_kwargs: Mapping[str, Any] | None = None,
    tcad_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Attach optional FDTD optical and TCAD collection LUTs to a sensor."""

    updated = sensor.clone()
    if fdtd_lut is not None:
        updated.fields["fdtd_sensor"] = fdtd_sensor_config(fdtd_lut, **dict(fdtd_kwargs or {}))
    if tcad_db is not None:
        updated.fields["tcad_sensor"] = tcad_sensor_config(tcad_db, **dict(tcad_kwargs or {}))
    return updated


def tcad_sensor_db_to_jsonable(db: TCADSensorDB | Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe representation of the TCAD sensor DB."""

    return tcad_sensor_summary(db)


def _npz_scalar_string(data: Any, key: str, default: str | None) -> str | None:
    if key not in data.files:
        return default
    values = np.asarray(data[key]).reshape(-1)
    if values.size == 0:
        return default
    return str(values[0])


def _npz_optional_float(data: Any, key: str) -> float | None:
    if key not in data.files:
        return None
    values = np.asarray(data[key], dtype=float).reshape(-1)
    if values.size == 0:
        return None
    return float(values[0])


def _npz_optional_array(data: Any, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    return np.asarray(data[key], dtype=float)


def _validate_generation_map(generation: TCADGenerationMap | None) -> dict[str, Any]:
    if generation is None:
        return {"ok": False, "issues": ["missing TCAD generation map"], "warnings": []}
    issues: list[str] = []
    warnings: list[str] = []
    if not generation.schema.startswith("tcad_generation_map_2d"):
        issues.append(f"unexpected generation-map schema: {generation.schema}")
    if generation.generation_cm3_s.shape[0] != generation.cases.size:
        issues.append("generation map case count does not match case axis")
    if generation.generation_cm3_s.shape[1] != generation.x_um.size:
        issues.append("generation map x-axis length does not match data")
    if generation.generation_cm3_s.shape[2] != generation.depth_um_from_si_top.size:
        issues.append("generation map depth-axis length does not match data")
    if not np.all(np.isfinite(generation.generation_cm3_s)):
        issues.append("generation map contains non-finite values")
    if np.nanmin(generation.generation_cm3_s) < -1e-12:
        issues.append("generation map contains negative values")
    if generation.x_um.size > 1 and not np.all(np.diff(generation.x_um) > 0):
        issues.append("generation x-axis is not strictly increasing")
    if generation.depth_um_from_si_top.size > 1 and not np.all(np.diff(generation.depth_um_from_si_top) > 0):
        issues.append("generation depth-axis is not strictly increasing")
    if generation.generation_cm3_s.shape[0] < 2:
        warnings.append("generation map has fewer than two field/CRA cases")
    return {
        "ok": not issues,
        "source_path": str(generation.source_path),
        "schema": generation.schema,
        "shape": list(generation.generation_cm3_s.shape),
        "cases": [str(item) for item in generation.cases],
        "generation_min": float(np.nanmin(generation.generation_cm3_s)),
        "generation_max": float(np.nanmax(generation.generation_cm3_s)),
        "issues": issues,
        "warnings": warnings,
    }


def _validate_collection_summaries(
    summaries: Sequence[TCADCollectionSummary],
    generation: TCADGenerationMap | None,
    source_root: Path | None,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    cases = {str(item) for item in generation.cases} if generation is not None else set()
    if not summaries:
        issues.append("missing DEVSIM collection summaries")
    for summary in summaries:
        if not summary.schema.startswith("devsim_split_pd_2d"):
            issues.append(f"{summary.source_path} has unexpected schema {summary.schema}")
        if summary.generation_source != "imported_2d_map":
            warnings.append(f"{summary.source_path} did not use imported Meep G(x,depth)")
        if summary.total_photo_delta_a_per_cm <= 0.0:
            issues.append(f"{summary.source_path} has non-positive total photo current")
        if abs(summary.terminal_current_balance_illuminated_a_per_cm) > 1e-9:
            issues.append(f"{summary.source_path} terminal current balance exceeds 1e-9 A/cm")
        if summary.case and cases and summary.case not in cases:
            issues.append(f"{summary.source_path} case {summary.case!r} is not present in the generation map")
        configured_map = _summary_generation_map_path(summary, source_root)
        if configured_map is not None and generation is not None:
            try:
                same_map = configured_map.resolve() == generation.source_path.resolve()
            except FileNotFoundError:
                same_map = str(configured_map) == str(generation.source_path)
            if not same_map:
                warnings.append(
                    f"{summary.source_path} was generated from {configured_map}, not the selected generation map {generation.source_path}"
                )
        if "proxy" in summary.electrical_model.lower():
            warnings.append(f"{summary.source_path} uses proxy electrical model {summary.electrical_model}")
    return {
        "ok": not issues,
        "n_summaries": len(summaries),
        "cases": [str(item.case) for item in summaries],
        "issues": issues,
        "warnings": warnings,
    }


def _summary_generation_map_path(summary: TCADCollectionSummary, source_root: Path | None) -> Path | None:
    raw = summary.config.get("generation_map_npz")
    if raw in {"", None}:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if source_root is not None:
        candidates.append(Path(source_root) / path)
    candidates.extend(parent / path for parent in summary.source_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _validate_accuracy_gate(gate: TCADAccuracyGate | None) -> dict[str, Any]:
    if gate is None:
        return {"ok": False, "issues": [], "warnings": ["missing TCAD accuracy gate"]}
    issues: list[str] = []
    warnings: list[str] = []
    if not gate.framework_ready:
        issues.append("TCAD framework gate is not ready")
    if not gate.accuracy_ready:
        warnings.append("TCAD accuracy gate is not ready; treat outputs as proxy simulation")
    return {
        "ok": gate.framework_ready,
        "source_path": str(gate.source_path),
        "profile_name": gate.profile_name,
        "framework_ready": gate.framework_ready,
        "accuracy_ready": gate.accuracy_ready,
        "accuracy_blocking_failure_count": gate.accuracy_blocking_failure_count,
        "framework_blocking_failure_count": gate.framework_blocking_failure_count,
        "issues": issues,
        "warnings": warnings,
    }


def _select_collection_summary(db: TCADSensorDB, *, case: str | None) -> TCADCollectionSummary | None:
    if not db.collection_summaries:
        return None
    if case is not None:
        for summary in db.collection_summaries:
            if summary.case == case:
                return summary
    return db.collection_summaries[0]


def _select_generation_index(generation: TCADGenerationMap, *, case: str | None, wavelength_nm: float | None) -> int:
    distances = []
    for index in range(generation.generation_cm3_s.shape[0]):
        distance = 0.0
        if case is not None:
            distance += 0.0 if str(generation.cases[index]) == str(case) else 1.0e6
        if wavelength_nm is not None and index < generation.wavelength_nm.size:
            distance += float(generation.wavelength_nm[index] - float(wavelength_nm)) ** 2
        distances.append(distance)
    return int(np.argmin(np.asarray(distances, dtype=float)))


def _config_enabled(config: Mapping[str, Any] | None) -> bool:
    return isinstance(config, Mapping) and bool(config.get("enabled", True))


def _mode_has(config: Mapping[str, Any] | None, token: str) -> bool:
    if not _config_enabled(config):
        return False
    mode = str(config.get("mode", "collection")).lower()
    return mode in {"all", "*"} or token in {part.strip() for part in mode.replace(",", "+").split("+")}


def _config_db(config_or_db: TCADSensorDB | Mapping[str, Any]) -> TCADSensorDB:
    if isinstance(config_or_db, TCADSensorDB):
        return config_or_db
    db = config_or_db.get("db")
    if isinstance(db, TCADSensorDB):
        return db
    if isinstance(db, Mapping):
        if "source_root" in db:
            loaded = tcad_sensor_db_load(root=db["source_root"])
        else:
            loaded = tcad_sensor_db_load(
                generation_map_path=db.get("generation_map_path"),
                collection_summary_paths=db.get("collection_summary_paths"),
                accuracy_gate_path=db.get("accuracy_gate_path"),
                root=db.get("root"),
            )
        if isinstance(config_or_db, dict):
            config_or_db["db"] = loaded
        return loaded
    if db is None:
        path_keys = {"generation_map_path", "collection_summary_paths", "accuracy_gate_path", "root", "source_root"}
        if any(key in config_or_db for key in path_keys):
            root = config_or_db.get("root", config_or_db.get("source_root"))
            loaded = tcad_sensor_db_load(
                generation_map_path=config_or_db.get("generation_map_path"),
                collection_summary_paths=config_or_db.get("collection_summary_paths"),
                accuracy_gate_path=config_or_db.get("accuracy_gate_path"),
                root=root,
            )
        else:
            loaded = tcad_sensor_db_load()
        if isinstance(config_or_db, dict):
            config_or_db["db"] = loaded
        return loaded
    raise TypeError("TCAD sensor config requires a TCADSensorDB or DB path mapping.")
