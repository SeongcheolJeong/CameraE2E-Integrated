"""Measured-to-simulated calibration fitting and evidence creation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .color import fit_constrained_ccm
from .models import CalibrationRequest, FidelityLevel, ReadinessTier
from .project import Project


def fit_calibration(project: Project, request: CalibrationRequest) -> dict[str, Any]:
    measured_path = Path(request.measured_path).expanduser().resolve()
    simulated_path = Path(request.simulated_path).expanduser().resolve()
    measured = _load_values(measured_path, request.measured_key)
    simulated = _load_values(simulated_path, request.simulated_key)
    if measured.shape != simulated.shape:
        raise ValueError(
            f"Measured and simulated arrays must have the same shape: "
            f"{measured.shape} != {simulated.shape}"
        )
    if request.kind == "color" and measured.ndim == 2 and measured.shape[1] == 3:
        return _fit_color_calibration(
            project,
            request,
            measured_path,
            simulated_path,
            measured,
            simulated,
        )
    finite = np.isfinite(measured) & np.isfinite(simulated)
    if request.valid_min is not None:
        finite &= simulated >= request.valid_min
    if request.valid_max is not None:
        finite &= simulated <= request.valid_max
    x = simulated[finite].reshape(-1)
    y = measured[finite].reshape(-1)
    if x.size < 3:
        raise ValueError("Calibration requires at least three finite paired samples")
    design = np.column_stack([x, np.ones_like(x)])
    gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = gain * x + offset
    residual = y - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    interval95 = float(1.96 * np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    measured_artifact = project.artifacts.put_file(
        measured_path,
        artifact_type=f"calibration_{request.kind}_measured",
        media_type=_media_type(measured_path),
        fidelity_level=FidelityLevel.CALIBRATED,
        readiness_tier=ReadinessTier.AVAILABLE,
        source="camerae2e_v2.calibration.measured",
    )
    simulated_artifact = project.artifacts.put_file(
        simulated_path,
        artifact_type=f"calibration_{request.kind}_simulated",
        media_type=_media_type(simulated_path),
        fidelity_level=FidelityLevel.SOLVER,
        readiness_tier=ReadinessTier.AVAILABLE,
        source="camerae2e_v2.calibration.simulated",
    )
    payload = {
        "schema_version": "camerae2e_calibration_fit_v2",
        "kind": request.kind,
        "sample_count": int(x.size),
        "model": {"type": "affine", "gain": float(gain), "offset": float(offset)},
        "residual": {"rmse": rmse, "mae": mae, "r2": r2, "interval95": interval95},
        "valid_domain": {"minimum": request.valid_min, "maximum": request.valid_max},
        "source_artifacts": [measured_artifact.hash, simulated_artifact.hash],
        "notes": request.notes,
        "promotion_scope": (
            f"Only the fitted {request.kind} response is calibrated. This does not promote "
            "the complete camera module or unrelated fidelity stages."
        ),
    }
    fit_artifact = project.artifacts.put_json(
        payload,
        artifact_type=f"calibration_{request.kind}_fit",
        fidelity_level=FidelityLevel.CALIBRATED,
        readiness_tier=ReadinessTier.CALIBRATED,
        source="camerae2e_v2.calibration.fit",
        dependencies=[measured_artifact.hash, simulated_artifact.hash],
        validation={"finite_samples": int(x.size), "r2": r2, "rmse": rmse},
    )
    return {**payload, "artifact": fit_artifact.model_dump(mode="json")}


def _fit_color_calibration(
    project: Project,
    request: CalibrationRequest,
    measured_path: Path,
    simulated_path: Path,
    measured: np.ndarray,
    simulated: np.ndarray,
) -> dict[str, Any]:
    ccm = fit_constrained_ccm(simulated, measured)
    measured_artifact = project.artifacts.put_file(
        measured_path,
        artifact_type="calibration_color_measured",
        media_type=_media_type(measured_path),
        fidelity_level=FidelityLevel.CALIBRATED,
        readiness_tier=ReadinessTier.AVAILABLE,
        source="camerae2e_v2.calibration.measured",
    )
    simulated_artifact = project.artifacts.put_file(
        simulated_path,
        artifact_type="calibration_color_simulated",
        media_type=_media_type(simulated_path),
        fidelity_level=FidelityLevel.SOLVER,
        readiness_tier=ReadinessTier.AVAILABLE,
        source="camerae2e_v2.calibration.simulated",
    )
    payload = {
        "schema_version": "camerae2e_color_calibration_v2",
        "kind": "color",
        "sample_count": ccm["sample_count"],
        "model": {"type": "constrained_ccm", "matrix": ccm["matrix"]},
        "validation": ccm,
        "source_artifacts": [measured_artifact.hash, simulated_artifact.hash],
        "notes": request.notes,
        "promotion_scope": (
            "Only the fitted CCM and supplied color domain are calibrated. Lens, sensor QE, "
            "noise, latency, and the complete camera module are not promoted."
        ),
    }
    readiness = ReadinessTier.CALIBRATED if ccm["validated"] else ReadinessTier.CALIBRATION_REQUIRED
    fit_artifact = project.artifacts.put_json(
        payload,
        artifact_type="calibration_color_ccm",
        fidelity_level=FidelityLevel.CALIBRATED,
        readiness_tier=readiness,
        source="camerae2e_v2.calibration.color",
        dependencies=[measured_artifact.hash, simulated_artifact.hash],
        validation={"validated": ccm["validated"], "gates": ccm["gates"]},
    )
    return {**payload, "artifact": fit_artifact.model_dump(mode="json")}


def _load_values(path: Path, key: str | None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected = key or next(iter(archive.files), None)
            if selected is None or selected not in archive:
                raise KeyError(f"Array key {selected!r} is unavailable in {path}")
            return np.asarray(archive[selected], dtype=float)
    if suffix == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if key:
            for part in key.split("."):
                payload = payload[part]
        return np.asarray(payload, dtype=float)
    if suffix in {".csv", ".txt"}:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        if not rows:
            return np.empty(0, dtype=float)
        if key and not _is_number(rows[0][0]):
            header = rows.pop(0)
            column = header.index(key)
            return np.asarray([float(row[column]) for row in rows], dtype=float)
        return np.asarray([[float(value) for value in row] for row in rows], dtype=float)
    raise ValueError(f"Unsupported calibration file format: {path.suffix}")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".npy": "application/x-npy",
        ".npz": "application/x-npz",
    }.get(path.suffix.lower(), "application/octet-stream")
