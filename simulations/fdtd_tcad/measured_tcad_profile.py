#!/usr/bin/env python3
"""Measured TCAD profile loader and interpolation helpers.

The schema is intentionally plain JSON/CSV so measured process data can be
versioned without a proprietary TCAD deck format.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError
except Exception:  # pragma: no cover - dependency fallback for minimal installs
    LinearNDInterpolator = None
    NearestNDInterpolator = None
    QhullError = Exception


@dataclass(frozen=True)
class MeasuredProfile:
    path: Path
    data: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def geometry(self) -> dict[str, Any]:
        return self.data["geometry"]

    @property
    def implants(self) -> list[dict[str, Any]]:
        return list(self.data.get("implants", []))

    @property
    def interfaces(self) -> list[dict[str, Any]]:
        return list(self.data.get("interfaces", []))

    @property
    def electrical_features(self) -> list[dict[str, Any]]:
        raw = self.data.get("electrical_features", [])
        if isinstance(raw, dict):
            return [
                {"name": name, **value}
                for name, value in raw.items()
                if isinstance(value, dict)
            ]
        return list(raw)

    @property
    def calibration_status(self) -> dict[str, Any]:
        return dict(self.data.get("calibration_status", {}))


def load_measured_profile(path: str | Path) -> MeasuredProfile:
    profile_path = Path(path).resolve()
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_measured_profile(data, profile_path)
    return MeasuredProfile(profile_path, data)


def validate_measured_profile(data: dict[str, Any], path: Path | None = None) -> None:
    label = str(path) if path else "profile"
    if data.get("schema") != "measured_tcad_profile_v1":
        raise ValueError(f"{label}: expected schema measured_tcad_profile_v1")
    geometry = data.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"{label}: missing geometry object")
    for key in ("width_um", "depth_um", "split_gap_um"):
        if key not in geometry:
            raise ValueError(f"{label}: geometry missing {key}")
        if float(geometry[key]) <= 0:
            raise ValueError(f"{label}: geometry {key} must be > 0")
    for implant in data.get("implants", []):
        if "name" not in implant or "type" not in implant:
            raise ValueError(f"{label}: each implant needs name and type")
        if implant["type"] == "csv_scattered":
            if "file" not in implant:
                raise ValueError(f"{label}: csv_scattered implant missing file")
            interpolation = str(implant.get("interpolation", "linear_nearest")).lower()
            if interpolation not in {"nearest", "linear", "linear_nearest", "idw"}:
                raise ValueError(
                    f"{label}: csv_scattered implant has unsupported interpolation {interpolation}"
                )
        elif implant["type"] in {"analytic_box", "analytic_smooth_box"}:
            for key in ("x_min_um", "x_max_um", "depth_min_um", "depth_max_um"):
                if key not in implant:
                    raise ValueError(f"{label}: {implant['type']} implant missing {key}")
        else:
            raise ValueError(f"{label}: unsupported implant type {implant['type']}")
    for feature in _feature_list(data):
        if "name" not in feature or "type" not in feature:
            raise ValueError(f"{label}: each electrical feature needs name and type")
        if feature["type"] == "doping_box":
            for key in ("x_min_um", "x_max_um", "depth_min_um", "depth_max_um"):
                if key not in feature:
                    raise ValueError(f"{label}: doping_box feature missing {key}")
        elif feature["type"] == "fixed_charge_sheet":
            if "fixed_charge_cm2" not in feature and "effective_trap_charge_cm2" not in feature:
                raise ValueError(
                    f"{label}: fixed_charge_sheet needs fixed_charge_cm2 or effective_trap_charge_cm2"
                )
        elif feature["type"] in {"metadata_only", "mobility_region"}:
            pass
        else:
            raise ValueError(f"{label}: unsupported electrical feature type {feature['type']}")


def _feature_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("electrical_features", [])
    if isinstance(raw, dict):
        return [
            {"name": name, **value}
            for name, value in raw.items()
            if isinstance(value, dict)
        ]
    return list(raw)


def _read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    if not rows:
        raise ValueError(f"empty implant CSV: {path}")
    return rows


def _scattered_points(
    rows: list[dict[str, float]],
    use_z: bool,
) -> np.ndarray:
    columns = ("x_um", "depth_um", "z_um") if use_z else ("x_um", "depth_um")
    return np.asarray(
        [[row.get(column, 0.0) for column in columns] for row in rows],
        dtype=float,
    )


def _scattered_queries(
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
) -> np.ndarray:
    if z_um is None:
        return np.column_stack([x_um.ravel(), depth_um.ravel()])
    return np.column_stack([x_um.ravel(), depth_um.ravel(), z_um.ravel()])


def _should_use_scattered_z(rows: list[dict[str, float]], z_um: np.ndarray | None) -> bool:
    if z_um is None:
        return False
    if not any("z_um" in row for row in rows):
        return False
    point_z = np.asarray([row.get("z_um", 0.0) for row in rows], dtype=float)
    return bool(np.ptp(point_z) > 1.0e-12)


def _nearest_values_dependency_free(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    chunk_size: int = 20000,
) -> np.ndarray:
    output = np.empty(queries.shape[0], dtype=float)
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        distances = np.sum((queries[start:stop, None, :] - points[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        output[start:stop] = values[nearest]
    return output


def _nearest_values(points: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if NearestNDInterpolator is not None:
        return np.asarray(NearestNDInterpolator(points, values)(queries), dtype=float)
    return _nearest_values_dependency_free(points, values, queries)


def _idw_values(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    power: float,
    chunk_size: int = 8000,
) -> np.ndarray:
    power = max(float(power), 1.0e-9)
    output = np.empty(queries.shape[0], dtype=float)
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        distances = np.sqrt(
            np.sum((queries[start:stop, None, :] - points[None, :, :]) ** 2, axis=2)
        )
        exact = distances <= 1.0e-15
        weights = 1.0 / np.maximum(distances, 1.0e-30) ** power
        values_chunk = np.sum(weights * values[None, :], axis=1) / np.sum(weights, axis=1)
        if np.any(exact):
            row_index, point_index = np.where(exact)
            for row, point in zip(row_index, point_index):
                values_chunk[row] = values[point]
        output[start:stop] = values_chunk
    return output


def _linear_nearest_values(
    points: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    if LinearNDInterpolator is None:
        return _nearest_values(points, values, queries), True, int(queries.shape[0])
    try:
        linear = np.asarray(
            LinearNDInterpolator(points, values, fill_value=np.nan)(queries),
            dtype=float,
        )
    except (QhullError, ValueError):
        return _nearest_values(points, values, queries), True, int(queries.shape[0])
    outside = ~np.isfinite(linear)
    fallback_count = int(np.count_nonzero(outside))
    if fallback_count:
        nearest = _nearest_values(points, values, queries[outside])
        linear[outside] = nearest
    return linear, False, fallback_count


def _interpolate_scattered_column(
    rows: list[dict[str, float]],
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
    column: str,
    interpolation: str,
    idw_power: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    use_z = _should_use_scattered_z(rows, z_um)
    points = _scattered_points(rows, use_z)
    queries = _scattered_queries(x_um, depth_um, z_um if use_z else None)
    values = np.asarray([row.get(column, 0.0) for row in rows], dtype=float)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite csv_scattered values for {column}")
    mode = interpolation.lower()
    fallback_to_nearest = False
    fallback_count = 0
    if mode == "nearest":
        interpolated = _nearest_values(points, values, queries)
    elif mode == "idw":
        interpolated = _idw_values(points, values, queries, idw_power)
    elif mode in {"linear", "linear_nearest"}:
        interpolated, fallback_to_nearest, fallback_count = _linear_nearest_values(
            points,
            values,
            queries,
        )
        if mode == "linear" and fallback_count:
            raise ValueError(
                f"linear interpolation for {column} produced {fallback_count} outside-hull query points; "
                "use interpolation=linear_nearest for nearest fallback"
            )
    else:
        raise ValueError(f"unsupported interpolation mode: {interpolation}")
    return interpolated.reshape(x_um.shape), {
        "column": column,
        "interpolation": mode,
        "point_count": int(points.shape[0]),
        "query_count": int(queries.shape[0]),
        "dimension": int(points.shape[1]),
        "scipy_available": bool(LinearNDInterpolator is not None),
        "linear_failed_to_nearest": bool(fallback_to_nearest),
        "outside_hull_nearest_fallback_count": fallback_count,
    }


def _scattered_doping(
    rows: list[dict[str, float]],
    implant: dict[str, Any],
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    interpolation = str(implant.get("interpolation", "linear_nearest")).lower()
    idw_power = float(implant.get("idw_power", 2.0))
    donor, donor_summary = _interpolate_scattered_column(
        rows,
        x_um,
        depth_um,
        z_um,
        "donor_cm3",
        interpolation,
        idw_power,
    )
    acceptor, acceptor_summary = _interpolate_scattered_column(
        rows,
        x_um,
        depth_um,
        z_um,
        "acceptor_cm3",
        interpolation,
        idw_power,
    )
    summary = {
        "name": implant.get("name", "csv_scattered_implant"),
        "type": implant.get("type", "csv_scattered"),
        "measured": bool(implant.get("measured", False)),
        "file": implant.get("file", ""),
        "interpolation": interpolation,
        "idw_power": idw_power,
        "donor": donor_summary,
        "acceptor": acceptor_summary,
        "donor_min_cm3": float(np.min(donor)),
        "donor_max_cm3": float(np.max(donor)),
        "acceptor_min_cm3": float(np.min(acceptor)),
        "acceptor_max_cm3": float(np.max(acceptor)),
    }
    return donor, acceptor, summary


def _nearest_scattered(
    rows: list[dict[str, float]],
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    donor, acceptor, _summary = _scattered_doping(
        rows,
        {"type": "csv_scattered", "interpolation": "nearest"},
        x_um,
        depth_um,
        z_um,
    )
    return donor, acceptor


def _smooth_window_1d(values: np.ndarray, lower: float, upper: float, rolloff: float) -> np.ndarray:
    if upper <= lower:
        raise ValueError("smooth implant upper bound must be larger than lower bound")
    if rolloff <= 0:
        return (values >= lower) & (values <= upper)
    left = 0.5 * (1.0 + np.tanh((values - lower) / rolloff))
    right = 0.5 * (1.0 + np.tanh((upper - values) / rolloff))
    return left * right


def _analytic_smooth_box(
    implant: dict[str, Any],
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
) -> np.ndarray:
    x_weight = _smooth_window_1d(
        x_um,
        float(implant["x_min_um"]),
        float(implant["x_max_um"]),
        float(implant.get("x_rolloff_um", implant.get("edge_rolloff_um", 0.02))),
    )
    depth_weight = _smooth_window_1d(
        depth_um,
        float(implant["depth_min_um"]),
        float(implant["depth_max_um"]),
        float(implant.get("depth_rolloff_um", implant.get("edge_rolloff_um", 0.04))),
    )
    weight = x_weight * depth_weight
    if z_um is not None and "z_min_um" in implant and "z_max_um" in implant:
        z_weight = _smooth_window_1d(
            z_um,
            float(implant["z_min_um"]),
            float(implant["z_max_um"]),
            float(implant.get("z_rolloff_um", implant.get("edge_rolloff_um", 0.02))),
        )
        weight *= z_weight
    if "x_peak_um" in implant and "x_sigma_um" in implant:
        sigma = float(implant["x_sigma_um"])
        if sigma <= 0:
            raise ValueError(f"{implant.get('name', 'implant')}: x_sigma_um must be > 0")
        weight *= np.exp(-0.5 * ((x_um - float(implant["x_peak_um"])) / sigma) ** 2)
    if "depth_peak_um" in implant and "depth_sigma_um" in implant:
        sigma = float(implant["depth_sigma_um"])
        if sigma <= 0:
            raise ValueError(f"{implant.get('name', 'implant')}: depth_sigma_um must be > 0")
        weight *= np.exp(-0.5 * ((depth_um - float(implant["depth_peak_um"])) / sigma) ** 2)
    return weight


def doping_from_profile_with_summary(
    profile: MeasuredProfile,
    x_cm: np.ndarray,
    depth_cm: np.ndarray,
    z_cm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_um = np.asarray(x_cm, dtype=float) * 1.0e4
    depth_um = np.asarray(depth_cm, dtype=float) * 1.0e4
    z_um = None if z_cm is None else np.asarray(z_cm, dtype=float) * 1.0e4
    donor_total = np.zeros_like(x_um, dtype=float)
    acceptor_total = np.zeros_like(x_um, dtype=float)
    implant_summary: list[dict[str, Any]] = []

    for implant in profile.implants:
        scale = float(implant.get("scale", 1.0))
        if implant["type"] == "analytic_box":
            mask = (
                (x_um >= float(implant["x_min_um"]))
                & (x_um <= float(implant["x_max_um"]))
                & (depth_um >= float(implant["depth_min_um"]))
                & (depth_um <= float(implant["depth_max_um"]))
            )
            if z_um is not None and "z_min_um" in implant and "z_max_um" in implant:
                mask &= (z_um >= float(implant["z_min_um"])) & (
                    z_um <= float(implant["z_max_um"])
                )
            donor_total += mask * float(implant.get("donor_cm3", 0.0)) * scale
            acceptor_total += mask * float(implant.get("acceptor_cm3", 0.0)) * scale
            implant_summary.append(
                {
                    "name": implant.get("name", "analytic_box_implant"),
                    "type": implant["type"],
                    "measured": bool(implant.get("measured", False)),
                    "interpolation": "analytic_box",
                    "active_node_count": int(np.count_nonzero(mask)),
                    "scale": scale,
                }
            )
        elif implant["type"] == "analytic_smooth_box":
            weight = _analytic_smooth_box(implant, x_um, depth_um, z_um)
            donor_total += weight * float(implant.get("donor_cm3", 0.0)) * scale
            acceptor_total += weight * float(implant.get("acceptor_cm3", 0.0)) * scale
            implant_summary.append(
                {
                    "name": implant.get("name", "analytic_smooth_box_implant"),
                    "type": implant["type"],
                    "measured": bool(implant.get("measured", False)),
                    "interpolation": "analytic_smooth_box",
                    "active_node_count": int(np.count_nonzero(weight > 1.0e-12)),
                    "weight_min": float(np.min(weight)),
                    "weight_max": float(np.max(weight)),
                    "scale": scale,
                }
            )
        elif implant["type"] == "csv_scattered":
            rows = _read_csv_rows(profile.base_dir / implant["file"])
            donor, acceptor, summary = _scattered_doping(
                implant=implant,
                rows=rows,
                x_um=x_um,
                depth_um=depth_um,
                z_um=z_um,
            )
            donor_total += donor * scale
            acceptor_total += acceptor * scale
            summary["scale"] = scale
            implant_summary.append(summary)
        else:
            raise ValueError(f"unsupported implant type {implant['type']}")

    background = profile.data.get("background", {})
    donor_total += float(background.get("donor_cm3", 0.0))
    acceptor_total += float(background.get("acceptor_cm3", 0.0))
    return donor_total, acceptor_total, implant_summary


def doping_from_profile(
    profile: MeasuredProfile,
    x_cm: np.ndarray,
    depth_cm: np.ndarray,
    z_cm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    donor_total, acceptor_total, _summary = doping_from_profile_with_summary(
        profile,
        x_cm,
        depth_cm,
        z_cm,
    )
    return donor_total, acceptor_total


def _box_mask(
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
    item: dict[str, Any],
) -> np.ndarray:
    mask = (
        (x_um >= float(item["x_min_um"]))
        & (x_um <= float(item["x_max_um"]))
        & (depth_um >= float(item["depth_min_um"]))
        & (depth_um <= float(item["depth_max_um"]))
    )
    if z_um is not None and "z_min_um" in item and "z_max_um" in item:
        mask &= (z_um >= float(item["z_min_um"])) & (z_um <= float(item["z_max_um"]))
    return mask


def _location_depth_um(item: dict[str, Any]) -> float:
    if "depth_um" in item:
        return float(item["depth_um"])
    if "location_depth_um" in item:
        return float(item["location_depth_um"])
    location = str(item.get("location", "depth_um=0"))
    if location.startswith("depth_um="):
        return float(location.split("=", 1)[1])
    raise ValueError(f"unsupported interface location: {location}")


def _fixed_charge_sheet(
    x_um: np.ndarray,
    depth_um: np.ndarray,
    z_um: np.ndarray | None,
    item: dict[str, Any],
    default_thickness_um: float,
) -> np.ndarray:
    sheet_charge_cm2 = float(item.get("fixed_charge_cm2", 0.0)) + float(
        item.get("effective_trap_charge_cm2", 0.0)
    )
    if sheet_charge_cm2 == 0.0:
        return np.zeros_like(x_um, dtype=float)
    thickness_um = float(item.get("sheet_thickness_um", default_thickness_um))
    if thickness_um <= 0:
        raise ValueError(f"{item.get('name', 'sheet')}: sheet_thickness_um must be > 0")
    half_thickness = 0.5 * thickness_um
    center_depth = _location_depth_um(item)
    mask = np.abs(depth_um - center_depth) <= half_thickness
    if "x_min_um" in item and "x_max_um" in item:
        mask &= (x_um >= float(item["x_min_um"])) & (x_um <= float(item["x_max_um"]))
    if z_um is not None and "z_min_um" in item and "z_max_um" in item:
        mask &= (z_um >= float(item["z_min_um"])) & (z_um <= float(item["z_max_um"]))
    thickness_cm = thickness_um * 1.0e-4
    return mask * (sheet_charge_cm2 / thickness_cm)


def electrical_terms_from_profile(
    profile: MeasuredProfile,
    x_cm: np.ndarray,
    depth_cm: np.ndarray,
    z_cm: np.ndarray | None = None,
    default_sheet_thickness_um: float = 0.02,
    feature_role_scales: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return donors, acceptors, fixed-charge equivalent doping, and summary.

    `fixed_charge_doping_cm3` is signed charge density in carrier-count units
    and is added to `NetDoping`. It is a sheet-charge approximation, not a full
    interface-trap occupancy model.
    """
    x_um = np.asarray(x_cm, dtype=float) * 1.0e4
    depth_um = np.asarray(depth_cm, dtype=float) * 1.0e4
    z_um = None if z_cm is None else np.asarray(z_cm, dtype=float) * 1.0e4
    donors, acceptors, implant_summary = doping_from_profile_with_summary(
        profile,
        x_cm,
        depth_cm,
        z_cm,
    )
    fixed_charge = np.zeros_like(donors, dtype=float)
    applied_features: list[dict[str, Any]] = []
    metadata_only: list[str] = []

    for feature in profile.electrical_features:
        if not feature.get("enabled", True):
            continue
        ftype = feature["type"]
        role = str(feature.get("role", "unspecified"))
        role_scale = float((feature_role_scales or {}).get(role, 1.0))
        scale = float(feature.get("scale", 1.0)) * role_scale
        if ftype == "doping_box":
            mask = _box_mask(x_um, depth_um, z_um, feature)
            donors += mask * float(feature.get("donor_cm3", 0.0)) * scale
            acceptors += mask * float(feature.get("acceptor_cm3", 0.0)) * scale
            applied_features.append(
                {
                    "name": feature["name"],
                    "role": role,
                    "type": ftype,
                    "measured": bool(feature.get("measured", False)),
                    "active_node_count": int(np.count_nonzero(mask)),
                    "scale": scale,
                    "runtime_role_scale": role_scale,
                }
            )
        elif ftype == "fixed_charge_sheet":
            charge = _fixed_charge_sheet(
                x_um,
                depth_um,
                z_um,
                feature,
                default_sheet_thickness_um,
            ) * scale
            fixed_charge += charge
            applied_features.append(
                {
                    "name": feature["name"],
                    "role": role,
                    "type": ftype,
                    "measured": bool(feature.get("measured", False)),
                    "active_node_count": int(np.count_nonzero(charge)),
                    "fixed_charge_cm2": float(feature.get("fixed_charge_cm2", 0.0)),
                    "effective_trap_charge_cm2": float(
                        feature.get("effective_trap_charge_cm2", 0.0)
                    ),
                    "scale": scale,
                    "runtime_role_scale": role_scale,
                }
            )
        elif ftype in {"metadata_only", "mobility_region"}:
            metadata_only.append(feature["name"])

    for interface in profile.interfaces:
        charge = _fixed_charge_sheet(
            x_um,
            depth_um,
            z_um,
            interface,
            default_sheet_thickness_um,
        )
        if np.any(charge):
            fixed_charge += charge
            applied_features.append(
                {
                    "name": interface.get("name", "interface"),
                    "role": "interface_fixed_charge",
                    "type": "fixed_charge_sheet",
                    "measured": bool(interface.get("measured", False)),
                    "active_node_count": int(np.count_nonzero(charge)),
                    "fixed_charge_cm2": float(interface.get("fixed_charge_cm2", 0.0)),
                    "effective_trap_charge_cm2": float(
                        interface.get("effective_trap_charge_cm2", 0.0)
                    ),
                }
            )
        elif "dit_cm2_ev" in interface:
            metadata_only.append(interface.get("name", "interface"))

    summary = {
        "profile": str(profile.path),
        "calibration_status": profile.calibration_status,
        "implant_summary": implant_summary,
        "applied_features": applied_features,
        "metadata_only_features": metadata_only,
        "notes": [
            "Donor/acceptor and fixed-charge proxy terms are applied to NetDoping.",
            "Interface traps are only electrical when fixed_charge_cm2 or effective_trap_charge_cm2 is supplied.",
            "Dit-only interface traps are exported as metadata for solver-specific trap-charge/recombination models.",
        ],
        "runtime_feature_role_scales": feature_role_scales or {},
    }
    return donors, acceptors, fixed_charge, summary


def write_example_implant_csv(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "x_um": x,
            "depth_um": depth,
            "donor_cm3": 5.0e15 if abs(x) > 0.04 and depth >= 0.08 else 0.0,
            "acceptor_cm3": 5.0e16 if depth < 0.08 else 1.0e14,
        }
        for x in (-0.6, -0.2, 0.0, 0.2, 0.6)
        for depth in (0.02, 0.12, 0.5, 1.5, 2.6)
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
