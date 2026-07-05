"""HW ISP parameter profile database.

Profiles provide named, provenance-tracked inputs for the HW ISP simulator. They
are intentionally normalized and small; vendor tuning dumps should be collected
or transformed into this schema rather than consumed directly by the simulator.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hwisp import HWIspConfig, hw_isp_config


@dataclass(frozen=True)
class HWIspParameterProfile:
    name: str
    description: str
    source: dict[str, Any]
    config: HWIspConfig
    calibration: dict[str, Any]
    notes: tuple[str, ...]
    confidence: str = "unknown"
    schema_version: int = 1
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "source": copy.deepcopy(self.source),
            "config": _config_payload_from_config(self.config),
            "calibration": copy.deepcopy(self.calibration),
            "notes": list(self.notes),
            "path": None if self.path is None else str(self.path),
        }


def _default_db_path() -> Path:
    env_path = os.environ.get("PYISETCAM_HWISP_DB")
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parent / "data" / "hwisp"


def _profile_files(db_path: str | Path | None = None) -> list[Path]:
    root = Path(db_path).expanduser() if db_path is not None else _default_db_path()
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if path.is_file())


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _config_payload_from_config(config: HWIspConfig) -> dict[str, Any]:
    return {
        "sensor_timing": config.sensor_timing.__dict__.copy(),
        "stages": [stage.__dict__.copy() for stage in config.stages],
        "control_path": config.control_path.__dict__.copy(),
        "transport": config.transport.__dict__.copy(),
        "global_latency_factor": float(config.global_latency_factor),
        "seed": int(config.seed),
    }


def _profile_from_payload(
    payload: dict[str, Any],
    path: Path | None = None,
) -> HWIspParameterProfile:
    if int(payload.get("schema_version", 1)) != 1:
        schema = payload.get("schema_version")
        raise ValueError(f"Unsupported HW ISP profile schema in {path}: {schema}")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"HW ISP profile missing name: {path}")
    config_payload = copy.deepcopy(payload.get("config", {}))
    config = hw_isp_config(**config_payload)
    return HWIspParameterProfile(
        schema_version=int(payload.get("schema_version", 1)),
        name=name,
        description=str(payload.get("description", "")),
        confidence=str(payload.get("confidence", "unknown")),
        source=copy.deepcopy(payload.get("source", {})),
        config=config,
        calibration=copy.deepcopy(payload.get("calibration", {})),
        notes=tuple(str(note) for note in payload.get("notes", [])),
        path=path,
    )


def hw_isp_parameter_db(db_path: str | Path | None = None) -> dict[str, HWIspParameterProfile]:
    """Load available HW ISP parameter profiles by name."""

    profiles: dict[str, HWIspParameterProfile] = {}
    for path in _profile_files(db_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = _profile_from_payload(payload, path)
        if profile.name in profiles:
            raise ValueError(f"Duplicate HW ISP profile name: {profile.name}")
        profiles[profile.name] = profile
    return profiles


def hw_isp_profile_names(db_path: str | Path | None = None) -> list[str]:
    """Return sorted HW ISP parameter profile names."""

    return sorted(hw_isp_parameter_db(db_path).keys())


def hw_isp_profile(name: str, db_path: str | Path | None = None) -> HWIspParameterProfile:
    """Return one named HW ISP parameter profile."""

    profiles = hw_isp_parameter_db(db_path)
    try:
        return profiles[str(name)]
    except KeyError as exc:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"Unknown HW ISP profile '{name}'. Available profiles: {available}") from exc


def hw_isp_config_from_profile(
    name: str,
    db_path: str | Path | None = None,
    **overrides: Any,
) -> HWIspConfig:
    """Create `HWIspConfig` from a named profile plus optional nested overrides."""

    profile = hw_isp_profile(name, db_path)
    payload = _config_payload_from_config(profile.config)
    payload = _deep_update(payload, overrides)
    return hw_isp_config(**payload)


hwISPParameterDB = hw_isp_parameter_db
hwISPProfileNames = hw_isp_profile_names
hwISPProfile = hw_isp_profile
hwISPConfigFromProfile = hw_isp_config_from_profile
