#!/usr/bin/env python3
"""Sensor-stack configuration and optical material helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import meep as mp
import numpy as np

from meep_microlens_array_3d import MicrolensArrayGeometry, medium_from_nk


ROOT = Path(__file__).resolve().parent
DEFAULT_STACK_CONFIG = ROOT / "configs" / "sensor_stack_proxy_1p4um.json"
VALID_SHIELD_MODES = {"off", "edge", "pdaf_left", "pdaf_right", "pdaf_pair"}


def load_stack_config(path: Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_STACK_CONFIG
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return data


def geometry_from_config(
    config: dict[str, Any],
    pml_um: float | None = None,
    grid_snap_y_resolution: int | None = None,
    grid_snap_y_mode: str = "off",
    min_bottom_air_um: float = 0.05,
) -> MicrolensArrayGeometry:
    geom_values = dict(config.get("geometry_um", {}))
    allowed = {item.name for item in fields(MicrolensArrayGeometry)}
    filtered = {key: float(value) for key, value in geom_values.items() if key in allowed}
    geom = MicrolensArrayGeometry(**filtered)
    if pml_um is not None:
        geom = replace(geom, pml=float(pml_um))
    if grid_snap_y_mode != "off":
        if grid_snap_y_resolution is None:
            raise ValueError("grid_snap_y_resolution is required when grid_snap_y_mode is enabled")
        geom = snap_geometry_y_to_grid(
            geom,
            resolution=int(grid_snap_y_resolution),
            mode=grid_snap_y_mode,
            min_bottom_air_um=min_bottom_air_um,
        )
    return geom


def snap_geometry_y_to_grid(
    geom: MicrolensArrayGeometry,
    resolution: int,
    mode: str = "nearest",
    min_bottom_air_um: float = 0.05,
) -> MicrolensArrayGeometry:
    """Adjust only bottom air padding so the y cell length lands on the FDTD grid."""
    if resolution <= 0:
        raise ValueError("resolution must be positive for y grid snapping")
    mode = mode.lower()
    requested_pixels = geom.cell_y * resolution
    if mode == "nearest":
        target_pixels = int(round(requested_pixels))
    elif mode == "ceil":
        target_pixels = int(math.ceil(requested_pixels - 1.0e-12))
    elif mode == "floor":
        target_pixels = int(math.floor(requested_pixels + 1.0e-12))
    else:
        raise ValueError("grid snap mode must be one of off, nearest, ceil, floor")

    target_cell_y = target_pixels / resolution
    delta_um = target_cell_y - geom.cell_y
    adjusted_bottom_air = geom.bottom_air + delta_um
    if adjusted_bottom_air < min_bottom_air_um:
        if mode == "nearest":
            target_pixels = int(math.ceil(requested_pixels - 1.0e-12))
            target_cell_y = target_pixels / resolution
            delta_um = target_cell_y - geom.cell_y
            adjusted_bottom_air = geom.bottom_air + delta_um
        if adjusted_bottom_air < min_bottom_air_um:
            raise ValueError(
                "y grid snapping would leave too little bottom air padding: "
                f"{adjusted_bottom_air:g} um < {min_bottom_air_um:g} um"
            )
    return replace(geom, bottom_air=adjusted_bottom_air)


def shield_config_for_stack(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized optional optical shield settings.

    `metal_edge_width` is kept as a geometric mask dimension for compatibility,
    but it no longer implies that a metal shield exists in the baseline stack.
    """
    geometry = config.get("geometry_um", {})
    shield = dict(config.get("shield", {}))
    shield.setdefault("mode", "off")
    shield.setdefault("mask_edge_width_um", geometry.get("metal_edge_width", 0.0))
    shield.setdefault("pdaf_axis", "x")
    shield["mode"] = str(shield["mode"]).lower()
    if shield["mode"] not in VALID_SHIELD_MODES:
        raise ValueError(
            f"Unsupported shield.mode={shield['mode']!r}; "
            f"expected one of {sorted(VALID_SHIELD_MODES)}"
        )
    shield["mask_edge_width_um"] = float(shield.get("mask_edge_width_um", 0.0))
    if shield["mask_edge_width_um"] < 0:
        raise ValueError("shield.mask_edge_width_um must be non-negative")
    shield["enabled"] = shield["mode"] != "off"
    return shield


def shield_active_half_width_um(
    geom: MicrolensArrayGeometry,
    shield: dict[str, Any],
) -> float:
    half_width = 0.5 * geom.pitch - float(shield.get("mask_edge_width_um", 0.0))
    if half_width <= 0:
        raise ValueError(
            "shield.mask_edge_width_um leaves no optical aperture; "
            f"got pitch={geom.pitch:g} um, mask_edge_width={shield.get('mask_edge_width_um')!r} um"
        )
    return half_width


def shield_blocks_local_point(
    dx_um: float,
    dz_um: float,
    active_half_width_um: float,
    shield_mode: str,
    pair_index: int = 0,
) -> bool:
    """Return whether an optional metal mask blocks a local CFA point."""
    mode = shield_mode.lower()
    if mode == "off":
        return False
    if mode == "edge":
        return abs(dx_um) > active_half_width_um or abs(dz_um) > active_half_width_um
    if mode == "pdaf_left":
        return dx_um < 0.0
    if mode == "pdaf_right":
        return dx_um > 0.0
    if mode == "pdaf_pair":
        return dx_um < 0.0 if pair_index % 2 == 0 else dx_um > 0.0
    raise ValueError(f"Unsupported shield mode: {shield_mode}")


def resolve_config_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(config.get("_config_dir", ROOT)).joinpath(path).resolve()


def load_nk_table(path: Path, wavelength_um: float) -> tuple[float, float]:
    rows: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = clean.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue

    if not rows:
        raise RuntimeError(f"No numeric wavelength,n,k rows found in {path}")

    table = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(table)):
        raise ValueError(f"Non-finite wavelength,n,k values found in {path}")
    order = np.argsort(table[:, 0])
    table = table[order]
    wavelengths = table[:, 0]
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError(f"Duplicate or non-increasing wavelength rows found in {path}")
    if np.any(table[:, 1] <= 0.0):
        raise ValueError(f"Non-positive refractive index n found in {path}")
    if np.any(table[:, 2] < 0.0):
        raise ValueError(f"Negative extinction coefficient k found in {path}")
    if wavelength_um < wavelengths.min() or wavelength_um > wavelengths.max():
        raise ValueError(
            f"wavelength {wavelength_um:.4f} um is outside {path.name} range "
            f"{wavelengths.min():.4f}-{wavelengths.max():.4f} um"
        )
    n = float(np.interp(wavelength_um, wavelengths, table[:, 1]))
    k = float(np.interp(wavelength_um, wavelengths, table[:, 2]))
    return n, k


def material_role_for_color(color_channel: str) -> str:
    channel = color_channel.lower()
    if channel in {"r", "red"}:
        return "cfa_red"
    if channel in {"g", "green", "gr", "gb"}:
        return "cfa_green"
    if channel in {"b", "blue"}:
        return "cfa_blue"
    raise ValueError(f"Unsupported color channel: {color_channel}")


def nk_from_material(
    config: dict[str, Any],
    role: str,
    wavelength_um: float,
) -> tuple[float, float, dict[str, Any]]:
    materials = config.get("materials", {})
    if role not in materials:
        raise KeyError(f"Material role '{role}' is missing from stack config")
    spec = dict(materials[role])
    if "nk_table" in spec:
        table_path = resolve_config_path(config, spec["nk_table"])
        n, k = load_nk_table(table_path, wavelength_um)
        spec["resolved_nk_table"] = str(table_path)
        return n, k, spec
    if "n" in spec:
        return float(spec["n"]), float(spec.get("k", 0.0)), spec
    if "index" in spec:
        return float(spec["index"]), 0.0, spec
    raise ValueError(f"Material role '{role}' needs nk_table, n/k, or index")


def medium_for_role(
    config: dict[str, Any],
    role: str,
    wavelength_um: float,
    frequency: float,
) -> tuple[mp.Medium, dict[str, Any]]:
    n, k, spec = nk_from_material(config, role, wavelength_um)
    spec["n_at_wavelength"] = n
    spec["k_at_wavelength"] = k
    return medium_from_nk(n, k, frequency), spec


def metal_for_stack(config: dict[str, Any]) -> tuple[mp.Medium, dict[str, Any]]:
    spec = dict(config.get("materials", {}).get("metal", {"model": "pec"}))
    model = spec.get("model", "pec")
    if model != "pec":
        raise ValueError("Only stable PEC metal is supported by the supercell LUT runner")
    return mp.metal, spec


def stack_metadata(
    config: dict[str, Any],
    geom: MicrolensArrayGeometry,
    color_channel: str,
) -> dict[str, Any]:
    return {
        "schema": config.get("schema"),
        "name": config.get("name"),
        "config_path": config.get("_config_path"),
        "geometry_um": asdict(geom),
        "shield": shield_config_for_stack(config),
        "color_channel": color_channel,
        "material_roles": config.get("materials", {}),
        "accuracy_notes": config.get("accuracy_notes", []),
    }
