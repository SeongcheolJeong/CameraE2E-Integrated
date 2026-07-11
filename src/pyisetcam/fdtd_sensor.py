"""FDTD-informed optical response helpers for sensor simulation.

The FDTD data used here is an optical absorption proxy.  It is intentionally
kept as a LUT layer so normal camera runs do not execute Meep/FDTD.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.signal import convolve2d


_NUMERIC_KEYS = {
    "wavelength_nm",
    "field_x_norm",
    "field_z_norm",
    "field_norm",
    "cra_x_deg",
    "cra_z_deg",
    "lens_shift_x_um",
    "lens_shift_z_um",
    "aperture_shift_x_um",
    "aperture_shift_z_um",
    "active_shift_x_um",
    "total_response",
    "max_region_response",
    "min_region_response",
    "normalized_total_response_to_first",
    "normalized_response_to_first_case",
    "total_si_absorption_fraction_estimate",
    "si_absorption_fraction_estimate",
    "collected_response_proxy",
    "response",
    "focal_region_fraction",
    "regional_flux_response_diagnostic",
    "normalized_region_response_to_first_same_region",
    "region_ix",
    "region_iz",
    "region_x_um",
    "region_z_um",
    "region_sx_um",
    "region_sz_um",
}
_DEFAULT_CAMERA_DB_ROOT_ENV = "PYISETCAM_CAMERA_DB_ROOT"


@dataclass(frozen=True)
class FDTDSensorLUT:
    """Loaded camera-system FDTD optical LUT."""

    source_path: Path
    schema: str
    mode: str
    geometry: dict[str, Any] = field(default_factory=dict)
    cell_pixels: dict[str, int] = field(default_factory=dict)
    wavelengths_nm: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    cases: list[dict[str, Any]] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    long_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def fdtd_sensor_default_lut_path() -> Path | None:
    """Return the first known FDTD camera LUT path, if it exists."""

    explicit = os.environ.get("PYISETCAM_FDTD_LUT_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path

    candidates: list[Path] = []
    env_root = os.environ.get("PYISETCAM_FDTD_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    camera_db_root = os.environ.get(_DEFAULT_CAMERA_DB_ROOT_ENV)
    if camera_db_root:
        candidates.append(Path(camera_db_root).expanduser() / "fdtd_tcad")
    candidates.extend(_default_camera_db_fdtd_roots())

    relative_candidates = [
        Path("runs/convergence_cra3_rgb_r84_gridsnap_quant/camera_lut.json"),
        Path("runs/convergence_cra3z_rgb_r84_gridsnap_quant/camera_lut.json"),
        Path("runs/convergence_cra_diag_rgb_r84_gridsnap_quant/camera_lut.json"),
        Path("runs/fdtd_to_tcad_generation_2d_cra5_r6_t12/camera_lut.json"),
        Path("runs/fdtd_to_tcad_generation_2d_cra5_smoke/camera_lut.json"),
        Path("runs/fdtd_to_tcad_generation_2d_cra_smoke/camera_lut.json"),
        Path("runs/supercell_lut_ocl_3x3_volume_smoke/camera_lut.json"),
        Path("runs/supercell_lut_ocl_3x3_smoke/camera_lut.json"),
    ]
    for root in candidates:
        for relative in relative_candidates:
            path = root / relative
            if path.exists():
                return path
    return None


def _default_camera_db_fdtd_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "camerae2e_db/fdtd_tcad",
        repo_root / "simulations/fdtd_tcad",
        repo_root.parent / "CameraE2E-DB/fdtd_tcad",
    ]


def fdtd_sensor_config(
    lut: FDTDSensorLUT | str | Path | Mapping[str, Any] | None = None,
    *,
    enabled: bool = True,
    mode: str = "qe+field+crosstalk",
    field_model: str = "radial",
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
    normalize_to_center: bool = True,
    crosstalk_strength: float = 1.0,
) -> dict[str, Any]:
    """Build a sensor-storable FDTD optical response configuration."""

    if isinstance(lut, Mapping):
        config = dict(lut)
        config.setdefault("enabled", enabled)
        config.setdefault("mode", mode)
        config.setdefault("field_model", field_model)
        config.setdefault("normalize_to_center", normalize_to_center)
        config.setdefault("crosstalk_strength", crosstalk_strength)
        return config
    return {
        "enabled": bool(enabled),
        "lut": lut,
        "mode": str(mode),
        "field_model": str(field_model),
        "case": case,
        "field_x_norm": None if field_x_norm is None else float(field_x_norm),
        "field_z_norm": None if field_z_norm is None else float(field_z_norm),
        "cra_x_deg": None if cra_x_deg is None else float(cra_x_deg),
        "cra_z_deg": None if cra_z_deg is None else float(cra_z_deg),
        "normalize_to_center": bool(normalize_to_center),
        "crosstalk_strength": float(crosstalk_strength),
    }


def fdtd_sensor_lut_load(path: str | Path | None = None) -> FDTDSensorLUT:
    """Load an FDTD sensor camera LUT from JSON or CSV."""

    resolved = Path(path) if path is not None else fdtd_sensor_default_lut_path()
    if resolved is None:
        raise FileNotFoundError("No FDTD LUT path was provided and no default FDTD LUT was found.")
    resolved = resolved.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        long_csv = _resolve_lut_sidecar_path(resolved, payload.get("long_csv"), "camera_lut_long.csv")
        summary_csv = _resolve_lut_sidecar_path(
            resolved,
            payload.get("summary_csv"),
            "camera_lut_summary.csv",
        )
        long_rows = _read_csv_rows(long_csv) if long_csv.exists() else []
        summary_rows = _read_csv_rows(summary_csv) if summary_csv.exists() else list(payload.get("summaries", []))
        return FDTDSensorLUT(
            source_path=resolved,
            schema=str(payload.get("schema", "unknown")),
            mode=str(payload.get("mode", "")),
            geometry=dict(payload.get("geometry", {})),
            cell_pixels={str(key): int(value) for key, value in dict(payload.get("cell_pixels", {})).items()},
            wavelengths_nm=np.asarray(payload.get("wavelengths_nm", []), dtype=float),
            cases=[_numeric_row(dict(item)) for item in payload.get("cases", [])],
            regions=[_numeric_row(dict(item)) for item in payload.get("regions", [])],
            summary_rows=[_numeric_row(dict(item)) for item in summary_rows],
            long_rows=[_numeric_row(dict(item)) for item in long_rows],
            notes=[str(item) for item in payload.get("notes", [])],
            metadata={key: value for key, value in payload.items() if key not in {"cases", "regions", "summaries"}},
        )

    rows = _read_csv_rows(resolved)
    return FDTDSensorLUT(
        source_path=resolved,
        schema="camera_supercell_optical_lut_v1" if "camera_lut" in resolved.name else "cra_response_lut_v1",
        mode=str(rows[0].get("mode", "cra")) if rows else "",
        wavelengths_nm=np.asarray(sorted({float(row["wavelength_nm"]) for row in rows if "wavelength_nm" in row}), dtype=float),
        summary_rows=[row for row in rows if "region_id" not in row],
        long_rows=[row for row in rows if "region_id" in row],
        metadata={},
    )


def fdtd_sensor_lut_validate(lut: FDTDSensorLUT) -> dict[str, Any]:
    """Return validation facts for an FDTD LUT."""

    summary = fdtd_sensor_lut_summary(lut)
    issues: list[str] = []
    if summary["n_wavelengths"] == 0:
        issues.append("missing wavelength axis")
    if not lut.summary_rows and not lut.long_rows:
        issues.append("missing response rows")
    if lut.long_rows and not any("region_ix" in row and "region_iz" in row for row in lut.long_rows):
        issues.append("long rows do not include region indices")
    if not any(_response_value(row) is not None for row in [*lut.summary_rows, *lut.long_rows]):
        issues.append("missing numeric response column")
    return {**summary, "ok": not issues, "issues": issues}


def fdtd_sensor_cos4_relative_illumination(cra_x_deg: float, cra_z_deg: float = 0.0) -> float:
    """Return the first-order cos^4 chief-ray relative-illumination estimate."""

    tan_x = np.tan(np.deg2rad(float(cra_x_deg)))
    tan_z = np.tan(np.deg2rad(float(cra_z_deg)))
    theta = np.arctan(np.hypot(tan_x, tan_z))
    return float(np.cos(theta) ** 4)


def fdtd_sensor_physics_validate(lut: FDTDSensorLUT) -> dict[str, Any]:
    """Run physics sanity checks on an FDTD optical sensor LUT.

    These checks do not prove that the LUT is product-grade.  They flag common
    problems before the LUT is used as camera-system evidence.
    """

    rows = lut.summary_rows if lut.summary_rows else _summarize_long_rows(lut.long_rows)
    warnings: list[str] = []
    failures: list[str] = []
    energy = _physics_energy_check(lut)
    relative_illumination = _physics_relative_illumination_check(rows)
    ocl = _physics_ocl_shift_check(rows)
    wavelength = _physics_wavelength_check(lut)
    symmetry = _physics_symmetry_check(rows)
    crosstalk = _physics_crosstalk_check(lut)
    convergence = _physics_convergence_check(lut)

    for check in (energy, relative_illumination, ocl, wavelength, symmetry, crosstalk, convergence):
        warnings.extend(str(item) for item in check.get("warnings", []))
        failures.extend(str(item) for item in check.get("failures", []))

    return {
        "ok": not failures,
        "status": "pass" if not failures and not warnings else "fail" if failures else "warn",
        "failures": failures,
        "warnings": warnings,
        "checks": {
            "energy": energy,
            "relative_illumination": relative_illumination,
            "ocl_shift": ocl,
            "wavelength": wavelength,
            "symmetry": symmetry,
            "crosstalk": crosstalk,
            "convergence": convergence,
        },
    }


def fdtd_sensor_lut_summary(lut: FDTDSensorLUT) -> dict[str, Any]:
    """Return a compact JSON-safe LUT summary."""

    wavelengths = sorted({float(value) for value in np.asarray(lut.wavelengths_nm, dtype=float).reshape(-1)})
    if not wavelengths:
        wavelengths = sorted({float(row["wavelength_nm"]) for row in [*lut.summary_rows, *lut.long_rows] if "wavelength_nm" in row})
    case_names = sorted({str(row.get("case", row.get("name", ""))) for row in [*lut.summary_rows, *lut.long_rows, *lut.cases] if row.get("case", row.get("name", ""))})
    region_ix = [int(row["region_ix"]) for row in lut.long_rows if "region_ix" in row and _row_is_pixel_region(row)]
    region_iz = [int(row["region_iz"]) for row in lut.long_rows if "region_iz" in row and _row_is_pixel_region(row)]
    return {
        "source_path": str(lut.source_path),
        "schema": lut.schema,
        "response_model": lut.metadata.get("response_model"),
        "mode": lut.mode,
        "n_wavelengths": len(wavelengths),
        "wavelengths_nm": wavelengths,
        "n_cases": len(case_names),
        "cases": case_names,
        "n_regions": len(lut.regions) or len({row.get("region_id") for row in lut.long_rows if row.get("region_id")}),
        "kernel_shape": [max(region_iz) - min(region_iz) + 1, max(region_ix) - min(region_ix) + 1]
        if region_ix and region_iz
        else [0, 0],
        "n_summary_rows": len(lut.summary_rows),
        "n_long_rows": len(lut.long_rows),
        "convergence_report": _source_convergence_report(lut),
        "notes": list(lut.notes),
    }


def fdtd_sensor_lut_response(
    lut: FDTDSensorLUT,
    *,
    wavelength_nm: float | None = None,
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
    normalize_to_center: bool = True,
) -> float:
    """Return nearest-neighbor normalized FDTD optical response."""

    row = _select_summary_row(
        lut,
        wavelength_nm=wavelength_nm,
        case=case,
        field_x_norm=field_x_norm,
        field_z_norm=field_z_norm,
        cra_x_deg=cra_x_deg,
        cra_z_deg=cra_z_deg,
    )
    if row is None:
        return 1.0

    if normalize_to_center:
        center = _select_summary_row(lut, wavelength_nm=wavelength_nm, case="center")
        center_response = _response_value(center) if center is not None else None
        response = _response_value(row)
        if response is not None and center_response and center_response > 0.0:
            return max(float(response) / float(center_response), 0.0)
        # Some legacy exports only include a normalized column. Prefer an
        # explicit same-wavelength center ratio whenever absolute responses are
        # present: normalized_total_response_to_first is often normalized to
        # the first sweep row, which may be a different wavelength.
        direct = _normalized_response_value(row)
        if direct is not None:
            return max(float(direct), 0.0)

    response = _response_value(row)
    return 1.0 if response is None else max(float(response), 0.0)


def fdtd_sensor_qe_scale(config: Mapping[str, Any] | None, wave_nm: Any) -> np.ndarray:
    """Return wavelength-dependent QE scale for the FDTD config."""

    wave = np.asarray(wave_nm, dtype=float).reshape(-1)
    if wave.size == 0 or not _config_enabled(config) or not _mode_has(config, "qe"):
        return np.ones(wave.shape, dtype=float)
    lut = _config_lut(config)
    return np.asarray(
        [
            fdtd_sensor_lut_response(
                lut,
                wavelength_nm=float(value),
                case=config.get("case"),
                field_x_norm=config.get("field_x_norm"),
                field_z_norm=config.get("field_z_norm"),
                cra_x_deg=config.get("cra_x_deg"),
                cra_z_deg=config.get("cra_z_deg"),
                normalize_to_center=bool(config.get("normalize_to_center", True)),
            )
            for value in wave
        ],
        dtype=float,
    )


def fdtd_sensor_lut_crosstalk_kernel(
    lut: FDTDSensorLUT,
    *,
    wavelength_nm: float | None = None,
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Return a regional-response crosstalk kernel from a supercell LUT."""

    rows = _select_long_rows(
        lut,
        wavelength_nm=wavelength_nm,
        case=case,
        field_x_norm=field_x_norm,
        field_z_norm=field_z_norm,
        cra_x_deg=cra_x_deg,
        cra_z_deg=cra_z_deg,
    )
    if not rows or not all(_row_is_pixel_region(row) for row in rows):
        return np.ones((1, 1), dtype=float)

    min_ix = min(int(row.get("region_ix", 0)) for row in rows)
    max_ix = max(int(row.get("region_ix", 0)) for row in rows)
    min_iz = min(int(row.get("region_iz", 0)) for row in rows)
    max_iz = max(int(row.get("region_iz", 0)) for row in rows)
    kernel = np.zeros((max_iz - min_iz + 1, max_ix - min_ix + 1), dtype=float)
    for row in rows:
        iz = int(row.get("region_iz", 0)) - min_iz
        ix = int(row.get("region_ix", 0)) - min_ix
        kernel[iz, ix] += float(_response_value(row) or 0.0)

    if normalize_sum:
        total = float(np.sum(kernel))
        if total > 0.0:
            kernel = kernel / total
    return kernel


def fdtd_sensor_field_response_map(
    lut: FDTDSensorLUT,
    shape: tuple[int, int],
    *,
    wavelength_nm: float | None = None,
    normalize_to_center: bool = True,
) -> np.ndarray:
    """Build a radial field-response map over a sensor plane."""

    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        return np.empty((0, 0), dtype=float)
    samples = _radial_response_samples(lut, wavelength_nm=wavelength_nm, normalize_to_center=normalize_to_center)
    if not samples:
        return np.ones((rows, cols), dtype=float)
    radii = np.asarray([item[0] for item in samples], dtype=float)
    values = np.asarray([item[1] for item in samples], dtype=float)
    order = np.argsort(radii)
    radii = radii[order]
    values = values[order]
    y = np.linspace(-1.0, 1.0, rows, dtype=float)
    x = np.linspace(-1.0, 1.0, cols, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.clip(np.hypot(yy, xx) / np.sqrt(2.0), 0.0, 1.0)
    return np.interp(rr, radii, values, left=float(values[0]), right=float(values[-1]))


def fdtd_sensor_apply_optical_response(values: Any, config: Mapping[str, Any] | None) -> np.ndarray:
    """Apply field rolloff and crosstalk to sensor-domain values."""

    array = np.asarray(values, dtype=float)
    if not _config_enabled(config):
        return array.copy()
    lut = _config_lut(config)
    result = array.copy()

    if _mode_has(config, "field"):
        fmap = fdtd_sensor_field_response_map(
            lut,
            result.shape[:2],
            wavelength_nm=_config_wavelength(config),
            normalize_to_center=bool(config.get("normalize_to_center", True)),
        )
        result = result * fmap if result.ndim == 2 else result * fmap[:, :, None]

    if _mode_has(config, "crosstalk"):
        kernel = fdtd_sensor_lut_crosstalk_kernel(
            lut,
            wavelength_nm=_config_wavelength(config),
            case=config.get("case"),
            field_x_norm=config.get("field_x_norm"),
            field_z_norm=config.get("field_z_norm"),
            cra_x_deg=config.get("cra_x_deg"),
            cra_z_deg=config.get("cra_z_deg"),
        )
        strength = float(config.get("crosstalk_strength", 1.0))
        if kernel.shape != (1, 1) and strength > 0.0:
            center = np.zeros_like(kernel, dtype=float)
            center[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
            effective = ((1.0 - strength) * center) + (strength * kernel)
            effective = effective / max(float(np.sum(effective)), 1e-12)
            if result.ndim == 2:
                result = convolve2d(result, effective, mode="same", boundary="symm")
            else:
                convolved = np.empty_like(result, dtype=float)
                for index in range(result.shape[2]):
                    convolved[:, :, index] = convolve2d(result[:, :, index], effective, mode="same", boundary="symm")
                result = convolved
    return np.asarray(result, dtype=float)


def sensor_attach_fdtd_lut(sensor: Any, lut: FDTDSensorLUT | str | Path | Mapping[str, Any], **kwargs: Any) -> Any:
    """Attach an FDTD optical LUT config to a sensor object."""

    updated = sensor.clone()
    updated.fields["fdtd_sensor"] = fdtd_sensor_config(lut, **kwargs)
    return updated


def fdtd_sensor_lut_to_jsonable(lut: FDTDSensorLUT) -> dict[str, Any]:
    """Return a compact JSON-safe representation of a LUT."""

    return fdtd_sensor_lut_summary(lut)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [_numeric_row(dict(row)) for row in csv.DictReader(stream)]


def _resolve_lut_sidecar_path(json_path: Path, raw_path: Any, fallback_name: str) -> Path:
    """Resolve sidecar paths stored as absolute paths or FDTD-root-relative paths."""

    if raw_path in {"", None}:
        return json_path.with_name(fallback_name)
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    candidates = [json_path.parent / path, json_path.with_name(path.name)]
    candidates.extend(parent / path for parent in json_path.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _numeric_row(row: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if value in {"", "None", "none", "null", None}:
            converted[key] = None
        elif key in _NUMERIC_KEYS:
            try:
                if key in {"region_ix", "region_iz"}:
                    converted[key] = int(float(value))
                else:
                    converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        else:
            converted[key] = value
    return converted


def _config_enabled(config: Mapping[str, Any] | None) -> bool:
    return isinstance(config, Mapping) and bool(config.get("enabled", True))


def _mode_has(config: Mapping[str, Any] | None, token: str) -> bool:
    if not _config_enabled(config):
        return False
    mode = str(config.get("mode", "qe+field+crosstalk")).lower()
    return mode in {"all", "*"} or token in {part.strip() for part in mode.replace(",", "+").split("+")}


def _config_lut(config: Mapping[str, Any]) -> FDTDSensorLUT:
    lut = config.get("lut")
    if isinstance(lut, FDTDSensorLUT):
        return lut
    if isinstance(lut, (str, Path)) or lut is None:
        loaded = fdtd_sensor_lut_load(lut)
        if isinstance(config, dict):
            config["lut"] = loaded
        return loaded
    if isinstance(lut, Mapping) and "source_path" in lut:
        return fdtd_sensor_lut_load(lut["source_path"])
    raise TypeError("FDTD sensor config requires a LUT object or path.")


def _config_wavelength(config: Mapping[str, Any]) -> float | None:
    value = config.get("wavelength_nm")
    return None if value is None else float(value)


def _response_value(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    for key in (
        "total_response",
        "response",
        "collected_response_proxy",
        "total_si_absorption_fraction_estimate",
        "si_absorption_fraction_estimate",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _row_is_pixel_region(row: Mapping[str, Any]) -> bool:
    kind = str(row.get("region_kind", row.get("kind", ""))).lower()
    region_id = str(row.get("region_id", "")).lower()
    if kind:
        return kind == "pixel"
    if region_id:
        return region_id.startswith("pix_") or region_id.startswith("pixel")
    return True


def _normalized_response_value(row: Mapping[str, Any]) -> float | None:
    for key in ("normalized_total_response_to_first", "normalized_response_to_first_case"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _select_summary_row(
    lut: FDTDSensorLUT,
    *,
    wavelength_nm: float | None = None,
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
) -> dict[str, Any] | None:
    rows = lut.summary_rows
    if not rows and lut.long_rows:
        rows = _summarize_long_rows(lut.long_rows)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: _row_distance(
            row,
            wavelength_nm=wavelength_nm,
            case=case,
            field_x_norm=field_x_norm,
            field_z_norm=field_z_norm,
            cra_x_deg=cra_x_deg,
            cra_z_deg=cra_z_deg,
        ),
    )


def _select_long_rows(
    lut: FDTDSensorLUT,
    *,
    wavelength_nm: float | None = None,
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
) -> list[dict[str, Any]]:
    if not lut.long_rows:
        return []
    selected = min(
        lut.long_rows,
        key=lambda row: _row_distance(
            row,
            wavelength_nm=wavelength_nm,
            case=case,
            field_x_norm=field_x_norm,
            field_z_norm=field_z_norm,
            cra_x_deg=cra_x_deg,
            cra_z_deg=cra_z_deg,
        ),
    )
    keys = {
        "wavelength_nm": selected.get("wavelength_nm"),
        "case": selected.get("case"),
        "field_x_norm": selected.get("field_x_norm"),
        "field_z_norm": selected.get("field_z_norm"),
        "cra_x_deg": selected.get("cra_x_deg"),
        "cra_z_deg": selected.get("cra_z_deg"),
    }
    rows = []
    for row in lut.long_rows:
        if all(_same_or_missing(row.get(key), value) for key, value in keys.items()):
            rows.append(row)
    return rows


def _row_distance(
    row: Mapping[str, Any],
    *,
    wavelength_nm: float | None = None,
    case: str | None = None,
    field_x_norm: float | None = None,
    field_z_norm: float | None = None,
    cra_x_deg: float | None = None,
    cra_z_deg: float | None = None,
) -> float:
    distance = 0.0
    if case is not None:
        row_case = str(row.get("case", row.get("name", "")))
        distance += 0.0 if row_case == case else 1.0e6
    for key, target in (
        ("wavelength_nm", wavelength_nm),
        ("field_x_norm", field_x_norm),
        ("field_z_norm", field_z_norm),
        ("cra_x_deg", cra_x_deg),
        ("cra_z_deg", cra_z_deg),
    ):
        if target is None or row.get(key) is None:
            continue
        distance += float(row[key] - float(target)) ** 2
    return distance


def _same_or_missing(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, (float, int)) or isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right)))
    return left == right


def _summarize_long_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("wavelength_nm"),
            row.get("case"),
            row.get("field_x_norm"),
            row.get("field_z_norm"),
            row.get("cra_x_deg"),
            row.get("cra_z_deg"),
        )
        group = groups.setdefault(
            key,
            {
                "wavelength_nm": row.get("wavelength_nm"),
                "case": row.get("case"),
                "field_x_norm": row.get("field_x_norm"),
                "field_z_norm": row.get("field_z_norm"),
                "cra_x_deg": row.get("cra_x_deg"),
                "cra_z_deg": row.get("cra_z_deg"),
                "total_response": 0.0,
            },
        )
        group["total_response"] += float(_response_value(row) or 0.0)
    first = next(iter(groups.values()), None)
    first_response = float(first["total_response"]) if first is not None else 0.0
    for group in groups.values():
        group["normalized_total_response_to_first"] = (
            float(group["total_response"]) / first_response if first_response > 0.0 else 1.0
        )
    return list(groups.values())


def _radial_response_samples(
    lut: FDTDSensorLUT,
    *,
    wavelength_nm: float | None,
    normalize_to_center: bool,
) -> list[tuple[float, float]]:
    rows = lut.summary_rows if lut.summary_rows else _summarize_long_rows(lut.long_rows)
    if not rows:
        return []
    samples: list[tuple[float, float]] = []
    for row in rows:
        if wavelength_nm is not None and row.get("wavelength_nm") is not None:
            available = sorted({float(item.get("wavelength_nm")) for item in rows if item.get("wavelength_nm") is not None})
            nearest = min(available, key=lambda value: abs(value - float(wavelength_nm)))
            if not np.isclose(float(row["wavelength_nm"]), nearest):
                continue
        fx = float(row.get("field_x_norm") or row.get("field_norm") or 0.0)
        fz = float(row.get("field_z_norm") or 0.0)
        radius = float(np.clip(np.hypot(fx, fz), 0.0, 1.0))
        value = fdtd_sensor_lut_response(
            lut,
            wavelength_nm=float(row["wavelength_nm"]) if row.get("wavelength_nm") is not None else wavelength_nm,
            case=str(row.get("case")) if row.get("case") else None,
            normalize_to_center=normalize_to_center,
        )
        samples.append((radius, value))
    if not any(np.isclose(radius, 0.0) for radius, _ in samples):
        samples.append((0.0, 1.0))
    return samples


def _physics_energy_check(lut: FDTDSensorLUT) -> dict[str, Any]:
    rows = [*lut.summary_rows, *lut.long_rows]
    responses = [float(_response_value(row) or 0.0) for row in rows if _response_value(row) is not None]
    failures: list[str] = []
    warnings: list[str] = []
    negative = sum(1 for value in responses if value < -1e-12)
    over_unity = sum(1 for value in responses if value > 1.0 + 1e-9)
    absorption_over_incident = 0
    response_over_absorption = 0
    for row in rows:
        response = _response_value(row)
        if response is None:
            continue
        incident = row.get("incident_monitor_net_power_normalized")
        absorption = row.get("total_si_absorption_fraction_estimate", row.get("si_absorption_fraction_estimate"))
        if incident is not None and float(absorption if absorption is not None else response) > float(incident) + 1e-9:
            absorption_over_incident += 1
        if absorption is not None and "response" in row and float(row["response"] or 0.0) > float(absorption) + 1e-9:
            response_over_absorption += 1

    if negative:
        failures.append(f"{negative} response values are negative")
    if over_unity:
        failures.append(f"{over_unity} response values exceed unity")
    if absorption_over_incident:
        failures.append(f"{absorption_over_incident} absorption values exceed incident monitor power")
    if response_over_absorption:
        warnings.append(f"{response_over_absorption} regional response values exceed total absorption")

    return {
        "status": "fail" if failures else "warn" if warnings else "pass",
        "n_rows": len(rows),
        "n_response_values": len(responses),
        "n_negative": negative,
        "n_over_unity": over_unity,
        "n_absorption_over_incident": absorption_over_incident,
        "n_response_over_absorption": response_over_absorption,
        "max_response": float(max(responses)) if responses else None,
        "failures": failures,
        "warnings": warnings,
    }


def _physics_relative_illumination_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        response = _normalized_response_value(row)
        if response is None:
            response_raw = _response_value(row)
            if response_raw is None:
                continue
            center = _find_center_row(rows, row)
            center_response = _response_value(center) if center is not None else None
            if center_response is None or center_response <= 0:
                continue
            response = float(response_raw) / float(center_response)
        cra_x = float(row.get("cra_x_deg") or 0.0)
        cra_z = float(row.get("cra_z_deg") or 0.0)
        if np.isclose(cra_x, 0.0) and np.isclose(cra_z, 0.0):
            continue
        cos4 = fdtd_sensor_cos4_relative_illumination(cra_x, cra_z)
        ratio = float(response) / max(cos4, 1e-12)
        status = "pass"
        if ratio > 1.20:
            status = "warn"
            warnings.append(
                f"{row.get('case', row.get('name', 'case'))} response exceeds bare cos^4 by >20%; "
                "this can be valid with microlens/OCL concentration but needs calibration evidence"
            )
        elif ratio < 0.60:
            status = "warn"
            warnings.append(f"{row.get('case', row.get('name', 'case'))} response is much lower than cos^4 expectation")
        comparisons.append(
            {
                "case": str(row.get("case", row.get("name", ""))),
                "wavelength_nm": row.get("wavelength_nm"),
                "cra_x_deg": cra_x,
                "cra_z_deg": cra_z,
                "cra_deg": float(np.rad2deg(np.arctan(np.hypot(np.tan(np.deg2rad(cra_x)), np.tan(np.deg2rad(cra_z)))))),
                "fdtd_response_norm": float(response),
                "cos4_response": cos4,
                "ratio_to_cos4": ratio,
                "status": status,
            }
        )
    return {
        "status": "warn" if warnings or not comparisons else "pass",
        "comparisons": comparisons,
        "failures": [],
        "warnings": warnings if comparisons else ["no off-axis CRA rows available for cos^4 comparison"],
    }


def _physics_ocl_shift_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    pairs: list[dict[str, Any]] = []
    uncomp_rows = [row for row in rows if "uncomp" in str(row.get("case", row.get("name", ""))).lower()]
    compensated_rows = [
        row
        for row in rows
        if any(token in str(row.get("case", row.get("name", ""))).lower() for token in ("comp", "ocl", "shift"))
        and "uncomp" not in str(row.get("case", row.get("name", ""))).lower()
    ]
    for uncomp in uncomp_rows:
        best = None
        best_distance = float("inf")
        for comp in compensated_rows:
            distance = _row_distance(
                comp,
                wavelength_nm=uncomp.get("wavelength_nm"),
                field_x_norm=uncomp.get("field_x_norm", uncomp.get("field_norm")),
                field_z_norm=uncomp.get("field_z_norm", 0.0),
                cra_x_deg=uncomp.get("cra_x_deg"),
                cra_z_deg=uncomp.get("cra_z_deg"),
            )
            if distance < best_distance:
                best = comp
                best_distance = distance
        if best is None:
            continue
        uncomp_response = _normalized_response_value(uncomp) or _response_value(uncomp)
        comp_response = _normalized_response_value(best) or _response_value(best)
        if uncomp_response is None or comp_response is None:
            continue
        improvement = float(comp_response) - float(uncomp_response)
        status = "pass" if improvement > 0.0 else "fail"
        if improvement <= 0.0:
            failures.append(
                f"{best.get('case', best.get('name', 'compensated'))} response is not higher than {uncomp.get('case', 'uncompensated')}"
            )
        pairs.append(
            {
                "uncompensated_case": str(uncomp.get("case", uncomp.get("name", ""))),
                "compensated_case": str(best.get("case", best.get("name", ""))),
                "uncompensated_response": float(uncomp_response),
                "compensated_response": float(comp_response),
                "improvement": improvement,
                "status": status,
            }
        )
    if not pairs:
        warnings.append("no uncompensated/compensated OCL pair available")
    return {
        "status": "fail" if failures else "warn" if warnings else "pass",
        "pairs": pairs,
        "failures": failures,
        "warnings": warnings,
    }


def _physics_wavelength_check(lut: FDTDSensorLUT) -> dict[str, Any]:
    wavelengths = sorted({float(value) for value in np.asarray(lut.wavelengths_nm, dtype=float).reshape(-1)})
    if not wavelengths:
        wavelengths = sorted({float(row["wavelength_nm"]) for row in [*lut.summary_rows, *lut.long_rows] if row.get("wavelength_nm") is not None})
    has_rgb = any(value <= 470 for value in wavelengths) and any(520 <= value <= 580 for value in wavelengths) and any(value >= 620 for value in wavelengths)
    warnings = [] if has_rgb else ["LUT does not cover representative blue/green/red wavelengths"]
    return {
        "status": "pass" if has_rgb else "warn",
        "wavelengths_nm": wavelengths,
        "has_rgb_representative_wavelengths": has_rgb,
        "failures": [],
        "warnings": warnings,
    }


def _physics_symmetry_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        cra_x = float(row.get("cra_x_deg") or 0.0)
        cra_z = float(row.get("cra_z_deg") or 0.0)
        if np.isclose(cra_x, 0.0) and np.isclose(cra_z, 0.0):
            continue
        mirror = None
        for candidate in rows:
            if not np.isclose(float(candidate.get("cra_x_deg") or 0.0), -cra_x):
                continue
            if not np.isclose(float(candidate.get("cra_z_deg") or 0.0), -cra_z):
                continue
            mirror = candidate
            break
        if mirror is None:
            continue
        response = _normalized_response_value(row) or _response_value(row)
        mirror_response = _normalized_response_value(mirror) or _response_value(mirror)
        if response is None or mirror_response is None:
            continue
        delta = abs(float(response) - float(mirror_response))
        comparisons.append(
            {
                "case": str(row.get("case", row.get("name", ""))),
                "mirror_case": str(mirror.get("case", mirror.get("name", ""))),
                "response": float(response),
                "mirror_response": float(mirror_response),
                "abs_delta": float(delta),
            }
        )
    warnings = [] if comparisons else ["no +/- CRA mirror pairs available for symmetry validation"]
    return {
        "status": "pass" if comparisons else "warn",
        "comparisons": comparisons,
        "failures": [],
        "warnings": warnings,
    }


def _physics_crosstalk_check(lut: FDTDSensorLUT) -> dict[str, Any]:
    warnings: list[str] = []
    kernel = fdtd_sensor_lut_crosstalk_kernel(lut, case="center", wavelength_nm=550.0)
    if kernel.shape == (1, 1):
        warnings.append("no regional crosstalk kernel available")
        return {
            "status": "warn",
            "kernel_shape": list(kernel.shape),
            "center_weight": 1.0,
            "uniformity_cv": 0.0,
            "failures": [],
            "warnings": warnings,
        }
    mean = float(np.mean(kernel))
    std = float(np.std(kernel))
    cv = std / max(abs(mean), 1e-12)
    center = float(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2])
    if center < 0.25:
        warnings.append("center regional-response weight is low; current smoke LUT behaves like a near-uniform redistribution kernel")
    if cv < 0.05:
        warnings.append("regional-response kernel is nearly uniform; not a localized optical crosstalk PSF")
    return {
        "status": "warn" if warnings else "pass",
        "kernel_shape": list(kernel.shape),
        "center_weight": center,
        "uniformity_cv": cv,
        "kernel_sum": float(np.sum(kernel)),
        "failures": [],
        "warnings": warnings,
    }


def _physics_convergence_check(lut: FDTDSensorLUT) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    convergence = _load_source_convergence_report(lut)
    path_text = str(lut.source_path).lower()
    if "smoke" in path_text:
        warnings.append("LUT path is a smoke run; use convergence sweeps before quantitative product use")

    resolution = lut.metadata.get("resolution_px_per_um", lut.geometry.get("resolution_px_per_um", lut.geometry.get("resolution")))
    if convergence is not None:
        if not bool(convergence.get("passed", False)):
            failures.append("FDTD convergence report did not pass")
        elif not bool(convergence.get("full_numerical_convergence_pass", False)):
            warnings.append("FDTD report is grid-qualified but not full resolution/time/PML convergence")
        if int(convergence.get("negative_signed_flux_count", 0) or 0) > 0 and bool(convergence.get("fail_on_negative_signed_flux", False)):
            failures.append("FDTD convergence report has negative signed-flux diagnostics")
    elif resolution is None:
        warnings.append("LUT does not expose convergence metadata for FDTD resolution")

    return {
        "status": "fail" if failures else "warn" if warnings else "pass",
        "source_path": str(lut.source_path),
        "geometry": dict(lut.geometry),
        "resolution_px_per_um": resolution,
        "convergence_report": None if convergence is None else convergence,
        "failures": failures,
        "warnings": warnings,
    }


def _source_convergence_report(lut: FDTDSensorLUT) -> str | None:
    path = lut.source_path.with_name("convergence_report.json")
    return str(path) if path.exists() else None


def _load_source_convergence_report(lut: FDTDSensorLUT) -> dict[str, Any] | None:
    path = lut.source_path.with_name("convergence_report.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "error": str(exc), "path": str(path)}


def _find_center_row(rows: list[dict[str, Any]], row: Mapping[str, Any]) -> dict[str, Any] | None:
    wavelength = row.get("wavelength_nm")
    for candidate in rows:
        if wavelength is not None and candidate.get("wavelength_nm") is not None and not np.isclose(float(candidate["wavelength_nm"]), float(wavelength)):
            continue
        if np.isclose(float(candidate.get("cra_x_deg") or 0.0), 0.0) and np.isclose(float(candidate.get("cra_z_deg") or 0.0), 0.0):
            return candidate
    return None
