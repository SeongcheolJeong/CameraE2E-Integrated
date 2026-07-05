"""CameraE2E parameter optimization helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from .assets import AssetStore
from .system_faca import camerae2e_faca_report, camerae2e_run_scenario
from .types import Camera, Scene

ObjectiveSpec = Mapping[str, Any] | str | Iterable[Mapping[str, Any]]
ObjectiveCallable = Callable[[Mapping[str, Any]], float]

_PARAMETER_AXIS_CATALOG: dict[str, dict[str, Any]] = {
    "sensor.integration_time": {
        "area": "sensor",
        "unit": "s",
        "readiness_tier": "validated",
        "values": [0.001, 0.002, 0.004],
        "description": "Sensor exposure/integration time for RAW signal and noise tradeoffs.",
    },
    "sensor.analog_gain": {
        "area": "sensor",
        "unit": "x",
        "readiness_tier": "validated",
        "values": [1.0, 2.0, 4.0],
        "description": "Analog gain before digital conversion.",
    },
    "sensor.noise_flag": {
        "area": "sensor",
        "unit": "enum",
        "readiness_tier": "validated",
        "values": [0, 2],
        "description": "Sensor noise model selector.",
    },
    "sensor.pixel_size": {
        "area": "sensor_geometry",
        "unit": "m",
        "readiness_tier": "validated",
        "values": [2.0e-6, 2.8e-6, 3.75e-6],
        "description": (
            "Pixel pitch used by sensor sampling, pixel area, FOV-driven sensor sizing, "
            "and RAW signal/noise tradeoffs."
        ),
    },
    "sensor.pixel_fill_factor": {
        "area": "sensor_geometry",
        "unit": "fraction",
        "readiness_tier": "validated",
        "values": [0.55, 0.75, 0.95],
        "description": "Photodiode fill factor used in pixel-area signal collection.",
    },
    "sensor.n_samples_per_pixel": {
        "area": "sensor_geometry",
        "unit": "samples/pixel",
        "readiness_tier": "validated",
        "values": [1, 3, 5],
        "description": (
            "Sub-pixel spatial sampling for sensor irradiance integration. "
            "This is not a readout-binning factor."
        ),
    },
    "sensor.cfa_pattern": {
        "area": "sensor_spectral",
        "unit": "index matrix",
        "readiness_tier": "validated",
        "values": [
            [[1, 2], [2, 3]],
            [[2, 1], [3, 2]],
        ],
        "description": (
            "CFA pattern-and-size matrix. Filter spectra/names must remain consistent "
            "with the selected pattern indices."
        ),
    },
    "sensor.cfa_preset": {
        "area": "sensor_spectral",
        "unit": "enum",
        "readiness_tier": "validated",
        "values": ["bayer_rgb", "quad_bayer_rgb", "quad_bayer_bggr"],
        "description": (
            "Named CFA pattern selector. Quad Bayer presets expand to 4x4 same-color "
            "2x2 groups while reusing the current RGB filter spectra."
        ),
    },
    "sensor.binning_method": {
        "area": "sensor_readout",
        "unit": "enum",
        "readiness_tier": "proxy",
        "values": ["off", "kodak2008"],
        "description": (
            "Legacy pixel-binning compute wrapper. Treat as a readout proxy until "
            "charge-domain, readout-domain, and ISP-domain binning are separated."
        ),
    },
    "sensor.binning_factor": {
        "area": "sensor_readout",
        "unit": "pixels",
        "readiness_tier": "proxy",
        "values": [1, 2],
        "description": (
            "Readout-binning factor selector. Current backend supports 1/off and "
            "the legacy 2x binning proxy; larger factors need a dedicated readout "
            "or remosaic implementation."
        ),
    },
    "sensor.ocl_vignetting": {
        "area": "sensor_ocl",
        "unit": "enum",
        "readiness_tier": "proxy",
        "values": ["off", "centered", "optimal"],
        "description": (
            "On-chip-lens/microlens etendue proxy applied through sensor vignetting. "
            "This changes sensor_compute via the etendue map, but is not a calibrated "
            "OCL process stack."
        ),
    },
    "sensor.ocl_group_shape": {
        "area": "sensor_ocl",
        "unit": "pixels",
        "readiness_tier": "proxy",
        "values": ["1x1", "2x2"],
        "description": (
            "Analytic shared-OCL aperture group shape. A 2x2 group equalizes the "
            "spatial optical sample per spectral plane before CFA selection; it "
            "does not create extra optical resolution."
        ),
    },
    "sensor.ocl_group_equalization": {
        "area": "sensor_ocl",
        "unit": "fraction",
        "readiness_tier": "proxy",
        "values": [0.0, 0.5, 1.0],
        "description": (
            "Strength of the analytic shared-OCL group equalization in [0, 1]. "
            "This is a fast proxy for CRA-matched shared aperture behavior, not a "
            "DTI/FDTD-calibrated process-stack model."
        ),
    },
    "sensor.ocl_fnumber": {
        "area": "sensor_ocl",
        "unit": "f/#",
        "readiness_tier": "proxy",
        "values": [1.2, 1.8, 2.4],
        "description": "Microlens f-number used by the OCL etendue proxy.",
    },
    "sensor.ocl_focal_length_um": {
        "area": "sensor_ocl",
        "unit": "um",
        "readiness_tier": "proxy",
        "values": [1.0, 1.8, 2.6],
        "description": "Microlens focal length used by the OCL etendue proxy.",
    },
    "sensor.ocl_refractive_index": {
        "area": "sensor_ocl",
        "unit": "index",
        "readiness_tier": "calibration_required",
        "values": [1.45, 1.55, 1.65],
        "description": (
            "Microlens refractive-index proxy. Treat as calibration-required unless "
            "matched to an optical stack or FDTD OCL LUT."
        ),
    },
    "sensor.pixel_read_noise_v": {
        "area": "sensor_noise",
        "unit": "V",
        "readiness_tier": "validated",
        "values": [5.0e-4, 1.0e-3, 2.0e-3],
        "description": "Pixel read-noise voltage used by the sensor noise model.",
    },
    "sensor.pixel_dark_voltage": {
        "area": "sensor_noise",
        "unit": "V/s",
        "readiness_tier": "validated",
        "values": [2.5e-4, 1.0e-3, 4.0e-3],
        "description": "Pixel dark voltage accumulation rate.",
    },
    "sensor.pixel_voltage_swing": {
        "area": "sensor_readout",
        "unit": "V",
        "readiness_tier": "validated",
        "values": [0.6, 1.0, 1.4],
        "description": "Pixel voltage swing/full-well proxy used by clipping and quantization.",
    },
    "sensor.pixel_conversion_gain": {
        "area": "sensor_readout",
        "unit": "V/e-",
        "readiness_tier": "validated",
        "values": [5.0e-5, 1.0e-4, 2.0e-4],
        "description": "Pixel conversion gain used for electron-to-voltage conversion.",
    },
    "optics.fnumber": {
        "area": "optics",
        "unit": "f/#",
        "readiness_tier": "validated",
        "values": [2.0, 2.8, 4.0],
        "description": "Optics f-number for irradiance and diffraction/blur tradeoffs.",
    },
    "optics.focal_length": {
        "area": "optics",
        "unit": "m",
        "readiness_tier": "validated",
        "values": [0.0028, 0.004, 0.006],
        "description": "Effective focal length for FOV, magnification, and ADAS crop tradeoffs.",
    },
    "optics.si_psf_radius_um": {
        "area": "optics_psf",
        "unit": "um",
        "readiness_tier": "proxy",
        "values": [1.0, 2.0, 4.0],
        "description": (
            "Synthetic shift-invariant pillbox PSF radius. This is a blur proxy, "
            "not RayOptics geometric PSF or diffraction/wave-optics sign-off."
        ),
    },
    "optics.psf_angle_step": {
        "area": "optics_psf",
        "unit": "deg",
        "readiness_tier": "proxy",
        "values": [5.0, 10.0, 15.0],
        "description": "RayOptics/geometric PSF angular sampling step.",
    },
    "optics.rt_compute_spacing": {
        "area": "optics_psf",
        "unit": "m",
        "readiness_tier": "proxy",
        "values": [1.0e-6, 2.5e-6, 5.0e-6],
        "description": "Raytrace PSF compute spacing for geometric PSF resampling.",
    },
    "fdtd.mode": {
        "area": "sensor_physics",
        "unit": "enum",
        "readiness_tier": "proxy",
        "values": ["qe", "qe+field", "qe+field+crosstalk"],
        "description": "FDTD optical-response mode. Requires an attached FDTD LUT.",
    },
    "fdtd.crosstalk_strength": {
        "area": "sensor_physics",
        "unit": "x",
        "readiness_tier": "proxy",
        "values": [0.0, 0.5, 1.0],
        "description": "Optical-crosstalk strength applied from an attached FDTD LUT.",
    },
    "fdtd.ocl_shift_um": {
        "area": "sensor_physics",
        "unit": "um",
        "readiness_tier": "proxy",
        "values": [-0.25, 0.0, 0.25],
        "description": (
            "FDTD OCL/microlens lateral-shift selector. Only affects the pipeline "
            "when the attached FDTD LUT contains matching OCL cases."
        ),
    },
    "fdtd.cra_x_deg": {
        "area": "sensor_physics",
        "unit": "deg",
        "readiness_tier": "proxy",
        "values": [-10.0, 0.0, 10.0],
        "description": "FDTD chief-ray-angle x-axis selector for attached LUT cases.",
    },
    "fdtd.cra_z_deg": {
        "area": "sensor_physics",
        "unit": "deg",
        "readiness_tier": "proxy",
        "values": [0.0, 10.0, 20.0],
        "description": "FDTD chief-ray-angle z-axis selector for attached LUT cases.",
    },
    "tcad.collection_mode": {
        "area": "sensor_physics",
        "unit": "enum",
        "readiness_tier": "calibration_required",
        "values": ["collection", "off"],
        "description": "TCAD collection-response mode. Requires an attached TCAD DB/LUT.",
    },
    "ip.demosaic_method": {
        "area": "isp",
        "unit": "enum",
        "readiness_tier": "validated",
        "values": ["bilinear", "nearest neighbor", "laplacian"],
        "description": "Demosaic method used by the image processor.",
    },
    "hw_isp.ae_apply_delay_frames": {
        "area": "hw_isp",
        "unit": "frames",
        "readiness_tier": "proxy",
        "values": [0, 1, 2],
        "description": "Delayed AE control application in the HW ISP simulator.",
    },
    "hw_isp.awb_apply_delay_frames": {
        "area": "hw_isp",
        "unit": "frames",
        "readiness_tier": "proxy",
        "values": [0, 1, 2],
        "description": "Delayed AWB control application in the HW ISP simulator.",
    },
    "hw_isp.global_latency_factor": {
        "area": "hw_isp",
        "unit": "x",
        "readiness_tier": "proxy",
        "values": [0.8, 1.0, 1.2],
        "description": "Global stage latency scale for system-control sweeps.",
    },
}

_PARAMETER_SPACE_PRESETS: dict[str, tuple[str, ...]] = {
    "exposure": ("sensor.integration_time", "sensor.analog_gain"),
    "optics": ("optics.fnumber",),
    "optics_psf": ("optics.fnumber", "optics.focal_length", "optics.si_psf_radius_um"),
    "raytrace_psf": ("optics.psf_angle_step", "optics.rt_compute_spacing"),
    "sensor_geometry": (
        "sensor.pixel_size",
        "sensor.pixel_fill_factor",
        "sensor.n_samples_per_pixel",
    ),
    "sensor_ocl": (
        "sensor.ocl_vignetting",
        "sensor.ocl_group_shape",
        "sensor.ocl_group_equalization",
        "sensor.ocl_fnumber",
        "sensor.ocl_focal_length_um",
        "sensor.ocl_refractive_index",
    ),
    "sensor_spectral": ("sensor.cfa_preset", "sensor.cfa_pattern"),
    "sensor_readout": (
        "sensor.integration_time",
        "sensor.analog_gain",
        "sensor.pixel_read_noise_v",
        "sensor.pixel_voltage_swing",
        "sensor.binning_method",
        "sensor.binning_factor",
    ),
    "physics_proxy": (
        "fdtd.mode",
        "fdtd.crosstalk_strength",
        "fdtd.ocl_shift_um",
        "fdtd.cra_x_deg",
        "fdtd.cra_z_deg",
        "tcad.collection_mode",
    ),
    "isp": ("ip.demosaic_method",),
    "hw_isp_control": (
        "hw_isp.ae_apply_delay_frames",
        "hw_isp.awb_apply_delay_frames",
        "hw_isp.global_latency_factor",
    ),
    "research_smoke": ("sensor.integration_time", "optics.fnumber"),
    "raw_factory": ("sensor.integration_time", "sensor.analog_gain", "optics.fnumber"),
    "adas_camera": (
        "optics.focal_length",
        "optics.fnumber",
        "sensor.pixel_size",
        "sensor.cfa_preset",
        "sensor.binning_factor",
        "sensor.ocl_vignetting",
        "sensor.ocl_group_shape",
        "sensor.ocl_group_equalization",
        "sensor.integration_time",
        "sensor.analog_gain",
    ),
}

_SENSOR_OPTIMIZATION_PARAMETERS = {
    "integration_time",
    "integration time",
    "integration",
    "exposure_duration",
    "exposure duration",
    "exposure_time",
    "exposure time",
    "analog_gain",
    "analog gain",
    "gain",
    "noise_flag",
    "noise flag",
    "noise",
    "pixel_size",
    "pixel size",
    "pixel_size_m",
    "pixel size m",
    "pixel_pitch",
    "pixel pitch",
    "pixel_fill_factor",
    "pixel fill factor",
    "fill_factor",
    "fill factor",
    "n_samples_per_pixel",
    "n samples per pixel",
    "samples_per_pixel",
    "samples per pixel",
    "cfa_pattern",
    "cfa pattern",
    "pattern",
    "pattern_and_size",
    "pattern and size",
    "cfa_preset",
    "cfa preset",
    "cfa_pattern_preset",
    "cfa pattern preset",
    "cfa_type",
    "cfa type",
    "cfa_layout",
    "cfa layout",
    "cfa",
    "color_filter_array",
    "color filter array",
    "filter_names",
    "filter names",
    "filter_spectra",
    "filter spectra",
    "color_filters",
    "color filters",
    "pixel_read_noise_v",
    "pixel read noise v",
    "read_noise",
    "read noise",
    "read_noise_v",
    "read noise v",
    "pixel_dark_voltage",
    "pixel dark voltage",
    "dark_voltage",
    "dark voltage",
    "pixel_voltage_swing",
    "pixel voltage swing",
    "voltage_swing",
    "voltage swing",
    "pixel_conversion_gain",
    "pixel conversion gain",
    "conversion_gain",
    "conversion gain",
    "sensor_compute_method",
    "sensor compute method",
    "binning",
    "binning_method",
    "binning method",
    "pixel_binning",
    "pixel binning",
    "binning_factor",
    "binning factor",
    "pixel_binning_factor",
    "pixel binning factor",
    "readout_binning_factor",
    "readout binning factor",
    "ocl_vignetting",
    "ocl vignetting",
    "microlens_vignetting",
    "microlens vignetting",
    "pixel_vignetting",
    "pixel vignetting",
    "ocl_group_shape",
    "ocl group shape",
    "shared_ocl_shape",
    "shared ocl shape",
    "shared_ocl_aperture_shape",
    "shared ocl aperture shape",
    "ocl_group_equalization",
    "ocl group equalization",
    "ocl_equalization",
    "ocl equalization",
    "shared_ocl_equalization",
    "shared ocl equalization",
    "shared_ocl_aperture_equalization",
    "shared ocl aperture equalization",
    "ocl_group_proxy",
    "ocl group proxy",
    "shared_ocl_aperture",
    "shared ocl aperture",
    "ocl_fnumber",
    "ocl fnumber",
    "ocl_f_number",
    "ocl f number",
    "microlens_fnumber",
    "microlens fnumber",
    "microlens_f_number",
    "microlens f number",
    "ocl_focal_length_um",
    "ocl focal length um",
    "microlens_focal_length_um",
    "microlens focal length um",
    "ocl_refractive_index",
    "ocl refractive index",
    "microlens_refractive_index",
    "microlens refractive index",
}

_OBJECTIVE_METRIC_CATALOG: dict[str, dict[str, Any]] = {
    "metrics.color.rgb_mean": {
        "area": "color",
        "direction_hint": "maximize",
        "description": "Mean rendered RGB response.",
    },
    "metrics.artifact.raw_std": {
        "area": "artifact",
        "direction_hint": "minimize_or_target",
        "description": "RAW variation/noise proxy from the sensor stage.",
    },
    "metrics.artifact.rgb_std": {
        "area": "artifact",
        "direction_hint": "minimize_or_target",
        "description": "Rendered RGB variation proxy.",
    },
    "metrics.artifact.rgb_clip_fraction": {
        "area": "artifact",
        "direction_hint": "minimize",
        "description": "Fraction of rendered RGB samples clipped to 0 or 1.",
    },
    "metrics.control.frame_count": {
        "area": "control",
        "direction_hint": "target",
        "description": "HW ISP sequence frame count when HW ISP simulation is enabled.",
    },
}


def camerae2e_optimize_parameters(
    base_scenario: Mapping[str, Any] | None = None,
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]] | None = None,
    objective: ObjectiveSpec | ObjectiveCallable | None = None,
    *,
    method: str = "grid",
    max_cases: int | None = None,
    constraints: Iterable[Mapping[str, Any]] | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    top_k: int = 5,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Optimize CameraE2E camera parameters with deterministic bounded search.

    Parameters use dot paths.  Examples:

    - ``sensor.integration_time``
    - ``sensor.analog_gain``
    - ``optics.fnumber``
    - ``ip.demosaic_method``

    The score is always maximized. Objective specs can maximize, minimize, or
    target any numeric path in the FACA report. Supported methods are grid,
    random, Latin-hypercube, score-ranked evolutionary, discrete surrogate,
    and Gaussian-process Bayesian search over the axis values supplied by the
    caller or preset catalog.
    """

    axes = _normalize_parameter_space(parameter_space or {})
    parameter_validation = camerae2e_parameter_space_validate(
        axes,
        base_scenario=base_scenario,
    )
    _validate_parameter_axis_names(axes, base_scenario)
    candidate_plan = camerae2e_parameter_candidate_plan(
        axes,
        method=method,
        max_cases=max_cases,
        seed=seed,
        base_scenario=base_scenario,
    )
    objective_specs = _normalize_objective(objective)
    constraint_specs = [dict(item) for item in constraints or []]
    if candidate_plan["method"] == "evolutionary":
        cases, candidate_plan = _run_evolutionary_parameter_search(
            candidate_plan,
            axes,
            base_scenario=base_scenario,
            objective_specs=objective_specs,
            objective=objective,
            constraint_specs=constraint_specs,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=seed,
            include_arrays=include_arrays,
        )
    elif candidate_plan["method"] == "surrogate":
        cases, candidate_plan = _run_surrogate_parameter_search(
            candidate_plan,
            axes,
            base_scenario=base_scenario,
            objective_specs=objective_specs,
            objective=objective,
            constraint_specs=constraint_specs,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=seed,
            include_arrays=include_arrays,
        )
    elif candidate_plan["method"] == "gaussian_process":
        cases, candidate_plan = _run_gaussian_process_parameter_search(
            candidate_plan,
            axes,
            base_scenario=base_scenario,
            objective_specs=objective_specs,
            objective=objective,
            constraint_specs=constraint_specs,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=seed,
            include_arrays=include_arrays,
        )
    else:
        cases = [
            _evaluate_parameter_candidate(
                axis_values,
                case_index=index,
                base_scenario=base_scenario,
                objective_specs=objective_specs,
                objective=objective,
                constraint_specs=constraint_specs,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=seed,
                include_arrays=include_arrays,
            )
            for index, axis_values in enumerate(candidate_plan["candidates"])
        ]

    ranked = sorted(
        cases,
        key=lambda item: (float(item["score"]), -int(item["case_index"])),
        reverse=True,
    )
    feasible_cases = [item for item in ranked if item["feasible"]]
    pareto = _pareto_cases(feasible_cases)
    best = feasible_cases[0] if feasible_cases else None
    return {
        "schema_version": "camerae2e_parameter_optimization_v1",
        "method": _optimization_method_name(candidate_plan),
        "search_method": candidate_plan["method"],
        "seed": int(seed),
        "parameter_space": _jsonable(axes),
        "parameter_space_validation": parameter_validation,
        "candidate_plan": _candidate_plan_summary(candidate_plan),
        "objective": _jsonable(_objective_description(objective, objective_specs)),
        "constraints": _jsonable(constraint_specs),
        "case_count": len(cases),
        "feasible_count": len(feasible_cases),
        "pareto_case_count": len(pareto),
        "best_case": _strip_case_report(best),
        "top_cases": [_strip_case_report(item) for item in feasible_cases[: max(int(top_k), 0)]],
        "pareto_front": [_strip_case_report(item) for item in pareto],
        "selected_scenarios": [
            _jsonable(item.get("scenario", {}))
            for item in feasible_cases[: max(int(top_k), 0)]
        ],
        "cases": cases,
    }


def camerae2e_parameter_space_catalog(preset: str | None = None) -> dict[str, Any]:
    """Return supported automated CameraE2E parameter axes and presets."""

    if preset is None:
        return {
            "schema_version": "camerae2e_parameter_space_catalog_v1",
            "axes": _jsonable(_PARAMETER_AXIS_CATALOG),
            "presets": {name: list(paths) for name, paths in _PARAMETER_SPACE_PRESETS.items()},
        }
    key = str(preset).strip().lower()
    if key not in _PARAMETER_SPACE_PRESETS:
        raise ValueError(f"Unknown CameraE2E parameter-space preset: {preset!r}.")
    paths = _PARAMETER_SPACE_PRESETS[key]
    return {
        "schema_version": "camerae2e_parameter_space_catalog_v1",
        "preset": key,
        "axes": {path: _jsonable(_PARAMETER_AXIS_CATALOG[path]) for path in paths},
        "parameter_space": {path: list(_PARAMETER_AXIS_CATALOG[path]["values"]) for path in paths},
    }


def camerae2e_parameter_candidate_plan(
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]],
    *,
    method: str = "grid",
    max_cases: int | None = None,
    seed: int = 0,
    base_scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic, budget-aware candidate plan for optimization.

    ``grid`` preserves Cartesian grid semantics.  ``random`` and
    ``latin_hypercube`` sample from the provided discrete axis values and default
    to a bounded budget when ``max_cases`` is omitted. ``evolutionary``,
    ``surrogate``, and ``gaussian_process`` return deterministic seed
    populations;
    :func:`camerae2e_optimize_parameters` expands them adaptively using
    evaluated FACA objective scores.
    """

    axes = _normalize_parameter_space(parameter_space)
    validation = camerae2e_parameter_space_validate(axes, base_scenario=base_scenario)
    method_key = _normalize_candidate_method(method)
    full_grid_count = _full_grid_count(axes)
    budget, implicit_default_budget = _candidate_budget(
        method_key,
        full_grid_count,
        max_cases,
    )
    warnings: list[dict[str, Any]] = []
    if not validation.get("ok", False):
        return {
            "schema_version": "camerae2e_parameter_candidate_plan_v1",
            "ok": False,
            "method": method_key,
            "seed": int(seed),
            "axis_count": len(axes),
            "axis_sizes": {key: len(values) for key, values in axes.items()},
            "full_grid_count": full_grid_count,
            "max_cases": budget,
            "implicit_default_budget": implicit_default_budget,
            "truncated": False,
            "case_count": 0,
            "parameter_space_validation": validation,
            "warnings": [],
            "candidates": [],
        }
    if method_key == "evolutionary":
        adaptive_seed_count = _evolutionary_population_size(axes, budget)
        candidates = _evolutionary_seed_candidates(axes, adaptive_seed_count, seed)
        warnings.append(
            {
                "kind": "evolutionary_seed_plan",
                "message": (
                    "Standalone evolutionary candidate plans contain only the seed "
                    "population; camerae2e_optimize_parameters expands the search "
                    "adaptively from evaluated objective scores."
                ),
            }
        )
    elif method_key == "surrogate":
        adaptive_seed_count = _surrogate_seed_count(axes, budget)
        candidates = _surrogate_seed_candidates(axes, adaptive_seed_count, seed)
        warnings.append(
            {
                "kind": "surrogate_seed_plan",
                "message": (
                    "Standalone surrogate candidate plans contain only the seed "
                    "population; camerae2e_optimize_parameters expands the search "
                    "with a discrete RBF acquisition proxy over candidate pools."
                ),
            }
        )
    elif method_key == "gaussian_process":
        adaptive_seed_count = _gp_seed_count(axes, budget)
        candidates = _gp_seed_candidates(axes, adaptive_seed_count, seed)
        warnings.append(
            {
                "kind": "gaussian_process_seed_plan",
                "message": (
                    "Standalone Gaussian-process candidate plans contain only the "
                    "initial design; camerae2e_optimize_parameters expands the "
                    "search with GP posterior expected improvement."
                ),
            }
        )
    else:
        adaptive_seed_count = None
        candidates = _generate_parameter_candidates(
            axes,
            method=method_key,
            max_cases=budget,
            seed=seed,
        )
    if implicit_default_budget and full_grid_count > budget:
        warnings.append(
            {
                "kind": "implicit_default_budget",
                "message": (
                    f"{method_key} search defaulted to max_cases={budget}; "
                    "pass max_cases explicitly to change the budget."
                ),
            }
        )
    return {
        "schema_version": "camerae2e_parameter_candidate_plan_v1",
        "ok": True,
        "method": method_key,
        "seed": int(seed),
        "axis_count": len(axes),
        "axis_sizes": {key: len(values) for key, values in axes.items()},
        "full_grid_count": full_grid_count,
        "max_cases": budget,
        "implicit_default_budget": implicit_default_budget,
        "truncated": len(candidates) < full_grid_count,
        "case_count": len(candidates),
        "search_config": _adaptive_search_config(
            method_key,
            axes,
            budget,
            adaptive_seed_count,
        ),
        "parameter_space_validation": validation,
        "warnings": warnings,
        "candidates": _jsonable(candidates),
    }


def camerae2e_optimization_config_catalog() -> dict[str, Any]:
    """Return all documented CameraE2E optimization configure targets.

    Registered axes are validated smoke-test targets.  Custom path rules
    describe additional dot-paths that the optimizer can assign, but callers
    should still verify parameter lineage and FACA metric movement.
    """

    return {
        "schema_version": "camerae2e_optimization_config_catalog_v1",
        "registered_axis_count": len(_PARAMETER_AXIS_CATALOG),
        "registered_axes": _jsonable(_PARAMETER_AXIS_CATALOG),
        "presets": {name: list(paths) for name, paths in _PARAMETER_SPACE_PRESETS.items()},
        "custom_path_rules": [
            {
                "path_pattern": "sensor.<name>",
                "assignment": "scenario.sensor -> sensor_set(...)",
                "readiness_tier": "validated",
                "allowed_suffixes": sorted(_SENSOR_OPTIMIZATION_PARAMETERS),
                "registered_examples": [
                    "sensor.integration_time",
                    "sensor.analog_gain",
                    "sensor.noise_flag",
                    "sensor.pixel_size",
                    "sensor.pixel_fill_factor",
                    "sensor.cfa_preset",
                    "sensor.cfa_pattern",
                    "sensor.binning_method",
                    "sensor.binning_factor",
                    "sensor.ocl_vignetting",
                    "sensor.ocl_group_shape",
                    "sensor.ocl_group_equalization",
                    "sensor.ocl_fnumber",
                ],
                "truth_boundary": (
                    "sensor.binning_method and sensor.binning_factor route to a legacy "
                    "binning compute proxy; "
                    "sensor.n_samples_per_pixel is sub-pixel integration sampling, "
                    "not readout binning. OCL/microlens axes use etendue/vignetting "
                    "and shared-aperture analytic proxies unless an FDTD OCL LUT is attached. "
                    "CFA presets change sampling layout but reuse the current filter spectra."
                ),
            },
            {
                "path_pattern": "hw_isp.<name>",
                "assignment": "scenario.hw_isp control dictionary",
                "readiness_tier": "proxy",
                "allowed_suffixes": "open, simulator-dependent",
                "side_effect": "enabled=True is inserted for HW ISP axes except hw_isp.enabled",
                "registered_examples": [
                    "hw_isp.ae_apply_delay_frames",
                    "hw_isp.awb_apply_delay_frames",
                    "hw_isp.global_latency_factor",
                ],
            },
            {
                "path_pattern": "fdtd.<name>",
                "assignment": "scenario.fdtd attachment options",
                "readiness_tier": "proxy",
                "allowed_suffixes": [
                    "lut",
                    "mode",
                    "field_model",
                    "case",
                    "cra_x_deg",
                    "cra_z_deg",
                    "ocl_shift_um",
                    "crosstalk_strength",
                ],
                "requirement": (
                    "FDTD configure axes only change the pipeline when "
                    "base_scenario.fdtd.lut is attached."
                ),
            },
            {
                "path_pattern": "tcad.<name>",
                "assignment": "scenario.tcad attachment options",
                "readiness_tier": "calibration_required",
                "allowed_suffixes": ["db", "collection_mode"],
                "requirement": (
                    "tcad.collection_mode only changes the pipeline when "
                    "base_scenario.tcad.db is attached."
                ),
            },
            {
                "path_pattern": "<camera_set dot path>",
                "assignment": "scenario.parameters/camera_parameters -> camera_set(...)",
                "readiness_tier": "available",
                "examples": [
                    "optics.fnumber",
                    "optics.focal_length",
                    "optics.si_psf_radius_um",
                    "optics.psf_angle_step",
                    "optics.rt_compute_spacing",
                    "pixel.size",
                    "sensor.bits",
                    "ip.demosaic_method",
                ],
                "truth_boundary": (
                    "The optimizer can assign these paths, but actual effect depends "
                    "on camera_set support and should be verified with parameter_lineage. "
                    "optics.si_psf_radius_um is a CameraE2E high-level synthetic PSF proxy, "
                    "not a product lens PSF calibration."
                ),
            },
        ],
        "objective_metrics": _jsonable(_OBJECTIVE_METRIC_CATALOG),
    }


def camerae2e_parameter_space_validate(
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]],
    *,
    base_scenario: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and classify optimization parameter-space paths without running a sweep."""

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        axes = _normalize_parameter_space(parameter_space)
    except ValueError as exc:
        return {
            "schema_version": "camerae2e_parameter_space_validation_v1",
            "ok": False,
            "axis_count": 0,
            "issue_count": 1,
            "warning_count": 0,
            "issues": [{"kind": "parameter_space", "message": str(exc)}],
            "warnings": [],
            "axes": {},
        }
    axis_payload: dict[str, dict[str, Any]] = {}
    for path, values in axes.items():
        axis_issues = _parameter_axis_issues(path, base_scenario)
        value_issues, value_warnings = _parameter_value_findings(path, values)
        issues.extend(axis_issues)
        issues.extend(value_issues)
        warnings.extend(value_warnings)
        classification = _parameter_axis_classification(path)
        if classification["status"] == "custom_passthrough":
            warnings.append(
                {
                    "path": path,
                    "kind": "custom_passthrough",
                    "message": (
                        "Axis is assignable through camera_set passthrough but is not "
                        "a registered validated optimization axis."
                    ),
                }
            )
        axis_payload[path] = {
            "values": _jsonable(values),
            "value_count": len(values),
            **classification,
            "issues": axis_issues + value_issues,
            "value_issues": value_issues,
            "value_warnings": value_warnings,
        }
    return {
        "schema_version": "camerae2e_parameter_space_validation_v1",
        "ok": not issues,
        "axis_count": len(axes),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "axes": axis_payload,
    }


def camerae2e_optimize_camera_parameters(
    base_scenario: Mapping[str, Any] | None = None,
    *,
    preset: str = "raw_factory",
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]] | None = None,
    objective: ObjectiveSpec | ObjectiveCallable | None = None,
    method: str = "grid",
    max_cases: int | None = None,
    constraints: Iterable[Mapping[str, Any]] | None = None,
    scene: Scene | str | Mapping[str, Any] | None = None,
    camera: Camera | None = None,
    asset_store: AssetStore | None = None,
    seed: int = 0,
    top_k: int = 5,
    include_arrays: bool = False,
) -> dict[str, Any]:
    """Run the automated CameraE2E camera-parameter optimization baseline.

    ``preset`` supplies a validated parameter-space starter set.  Passing
    ``parameter_space`` overrides or extends those axes while retaining the
    same deterministic optimizer, constraint, Pareto, and selected-scenario
    outputs as :func:`camerae2e_optimize_parameters`.
    """

    catalog = camerae2e_parameter_space_catalog(preset)
    axes = dict(catalog["parameter_space"])
    if parameter_space is not None:
        axes.update(_normalize_parameter_space(parameter_space))
    result = camerae2e_optimize_parameters(
        base_scenario,
        axes,
        objective,
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
    result["automation"] = {
        "schema_version": "camerae2e_parameter_optimization_automation_v1",
        "preset": str(preset).strip().lower(),
        "axis_count": len(axes),
        "search_method": result.get("search_method"),
        "max_cases": result.get("candidate_plan", {}).get("max_cases"),
        "candidate_count": result.get("candidate_plan", {}).get("case_count"),
        "axes": _jsonable(
            {
                path: _PARAMETER_AXIS_CATALOG.get(
                    path,
                    {
                        "readiness_tier": "available",
                        "description": "Caller-provided custom parameter axis.",
                    },
                )
                for path in axes
            }
        ),
    }
    return result


def camerae2e_pareto_front(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact non-dominated feasible cases from an optimization result."""

    if "pareto_front" in result:
        return _jsonable(result["pareto_front"])
    return [_strip_case_report(item) for item in _pareto_cases(list(result.get("cases", [])))]


def camerae2e_optimization_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe optimization report."""

    best = result.get("best_case")
    return {
        "schema_version": "camerae2e_parameter_optimization_report_v1",
        "method": result.get("method"),
        "search_method": result.get("search_method"),
        "seed": result.get("seed"),
        "case_count": result.get("case_count", 0),
        "feasible_count": result.get("feasible_count", 0),
        "pareto_case_count": result.get("pareto_case_count", 0),
        "candidate_plan": _jsonable(result.get("candidate_plan", {})),
        "objective": _jsonable(result.get("objective", {})),
        "constraints": _jsonable(result.get("constraints", [])),
        "best_case": _jsonable(best),
        "top_cases": _jsonable(result.get("top_cases", [])),
        "pareto_front": _jsonable(result.get("pareto_front", [])),
        "selected_scenarios": _jsonable(result.get("selected_scenarios", [])),
    }


def camerae2e_optimization_escalation_plan(
    result: Mapping[str, Any],
    *,
    selection: str = "pareto",
    max_cases: int | None = 5,
    strict: bool = False,
    include_physics_pipeline: bool = True,
    physics_pipeline_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan how optimized camera candidates should move to higher fidelity.

    The optimizer is intentionally allowed to run broad analytic/proxy sweeps.
    This planner keeps that useful speed while making the next evidence step
    explicit: DB/LUT anchoring, FDTD optical LUT validation, TCAD collection
    checks, RayOptics geometric PSF comparison, HW ISP trace calibration, and
    RAW/perception export.
    """

    if include_physics_pipeline and physics_pipeline_plan is None:
        from .physics_pipeline import camerae2e_physics_pipeline_plan

        physics_pipeline_plan = camerae2e_physics_pipeline_plan(strict=strict)
    selected_cases = _optimization_selected_cases(
        result,
        selection=selection,
        max_cases=max_cases,
    )
    axes = _optimization_axes_from_result(result, selected_cases)
    axis_summary = _optimization_axis_summary(axes)
    physics_summary = _physics_plan_summary(physics_pipeline_plan)
    stages = _escalation_stages(
        axis_summary,
        selected_case_count=len(selected_cases),
        physics_pipeline_plan=physics_pipeline_plan,
        strict=strict,
    )
    validation_jobs = [
        _escalation_validation_job(case, index, stages)
        for index, case in enumerate(selected_cases)
    ]
    blocking_stage_count = sum(
        1 for stage in stages if str(stage.get("status", "")).startswith("blocked")
    )
    return {
        "schema_version": "camerae2e_optimization_escalation_plan_v1",
        "ok": bool(selected_cases) and (not strict or blocking_stage_count == 0),
        "strict": bool(strict),
        "source": {
            "schema_version": result.get("schema_version"),
            "method": result.get("method"),
            "search_method": result.get("search_method"),
            "seed": result.get("seed"),
            "case_count": result.get("case_count", 0),
            "feasible_count": result.get("feasible_count", 0),
            "pareto_case_count": result.get("pareto_case_count", 0),
        },
        "selection": str(selection),
        "selected_case_count": len(selected_cases),
        "selected_cases": selected_cases,
        "axis_summary": axis_summary,
        "physics_pipeline": physics_summary,
        "stages": stages,
        "validation_jobs": validation_jobs,
        "acceptance_gates": _escalation_acceptance_gates(
            axis_summary,
            stages,
            selected_case_count=len(selected_cases),
        ),
        "truth_boundary": (
            "This is a research-fidelity escalation plan. Analytic and proxy "
            "optimization candidates are not promoted to calibrated/sign-off "
            "status until measured calibration evidence and closed lineage gates pass."
        ),
    }


def _optimization_selected_cases(
    result: Mapping[str, Any],
    *,
    selection: str,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    key = str(selection).strip().lower().replace("-", "_")
    if key in {"best", "winner"}:
        raw_cases = [result.get("best_case")] if result.get("best_case") else []
    elif key in {"top", "top_cases"}:
        raw_cases = list(result.get("top_cases", []))
    elif key in {"all", "cases"}:
        raw_cases = [_strip_case_report(case) for case in result.get("cases", [])]
    elif key in {"pareto", "pareto_front"}:
        raw_cases = list(result.get("pareto_front", [])) or camerae2e_pareto_front(result)
    else:
        raise ValueError("selection must be one of: best, top, pareto, all.")
    cases = [
        _escalation_case_summary(case, rank=index)
        for index, case in enumerate(raw_cases)
        if case is not None
    ]
    if max_cases is not None:
        cases = cases[: max(int(max_cases), 0)]
    return cases


def _optimization_axes_from_result(
    result: Mapping[str, Any], selected_cases: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    axes = result.get("parameter_space", {})
    if isinstance(axes, Mapping) and axes:
        return dict(axes)
    inferred: dict[str, list[Any]] = {}
    for case in selected_cases:
        for path, value in dict(case.get("parameters", {})).items():
            bucket = inferred.setdefault(str(path), [])
            if _value_signature(value) not in {_value_signature(item) for item in bucket}:
                bucket.append(value)
    return inferred


def _escalation_case_summary(case: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "score": case.get("score"),
        "feasible": bool(case.get("feasible", True)),
        "parameters": _jsonable(case.get("parameters", {})),
        "objective_values": _jsonable(case.get("objective_values", {})),
        "constraint_results": _jsonable(case.get("constraint_results", [])),
        "scenario": _jsonable(case.get("scenario", {})),
    }


def _optimization_axis_summary(axes: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    readiness_counts: dict[str, int] = {}
    area_counts: dict[str, int] = {}
    for path, values in axes.items():
        classification = _parameter_axis_classification(str(path))
        catalog_entry = dict(_PARAMETER_AXIS_CATALOG.get(str(path), {}))
        readiness = str(classification.get("readiness_tier", "available"))
        area = str(catalog_entry.get("area", _axis_area_from_path(str(path))))
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        area_counts[area] = area_counts.get(area, 0) + 1
        entries.append(
            {
                "path": str(path),
                "area": area,
                "readiness_tier": readiness,
                "status": classification.get("status"),
                "assignment": classification.get("assignment"),
                "value_count": _axis_value_count(values),
                "description": catalog_entry.get("description"),
            }
        )
    return {
        "axis_count": len(entries),
        "readiness_counts": readiness_counts,
        "area_counts": area_counts,
        "axes": entries,
        "needs": _escalation_needs(entries),
    }


def _axis_area_from_path(path: str) -> str:
    if "." not in path:
        return "camera"
    prefix = path.split(".", 1)[0]
    if prefix == "sensor":
        return "sensor_custom"
    if prefix == "optics":
        return "optics"
    if prefix == "fdtd":
        return "sensor_physics"
    if prefix == "tcad":
        return "sensor_physics"
    return prefix


def _axis_value_count(values: Any) -> int:
    if isinstance(values, Mapping) and "values" in values:
        return len(list(values.get("values", [])))
    if isinstance(values, Mapping):
        return 1
    try:
        return len(list(values))
    except TypeError:
        return 1


def _escalation_needs(entries: list[Mapping[str, Any]]) -> dict[str, bool]:
    paths = {str(entry.get("path", "")) for entry in entries}
    areas = {str(entry.get("area", "")) for entry in entries}
    return {
        "db_lut_anchor": any(path.startswith("sensor.") for path in paths),
        "fdtd_optical_lut": bool(
            areas & {"sensor_geometry", "sensor_spectral", "sensor_ocl", "sensor_physics"}
            or any(path.startswith("fdtd.") for path in paths)
        ),
        "tcad_collection": bool(
            areas & {"sensor_readout", "sensor_noise", "sensor_ocl", "sensor_physics"}
            or any(path.startswith("tcad.") for path in paths)
        ),
        "rayoptics_geometric_psf": bool(
            areas & {"optics", "optics_psf"} or any(path.startswith("optics.") for path in paths)
        ),
        "hw_isp_trace": any(path.startswith("hw_isp.") for path in paths),
        "raw_dataset_export": True,
        "perception_eval": True,
    }


def _physics_plan_summary(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {
            "included": False,
            "schema_version": None,
            "ok": None,
            "summary": {},
            "active_runs": {},
        }
    return {
        "included": True,
        "schema_version": plan.get("schema_version"),
        "ok": plan.get("ok"),
        "summary": _jsonable(plan.get("summary", {})),
        "active_runs": _jsonable(plan.get("active_runs", {})),
    }


def _escalation_stages(
    axis_summary: Mapping[str, Any],
    *,
    selected_case_count: int,
    physics_pipeline_plan: Mapping[str, Any] | None,
    strict: bool,
) -> list[dict[str, Any]]:
    needs = dict(axis_summary.get("needs", {}))
    stages = [
        _static_escalation_stage(
            "analytic_candidate_screen",
            needed=True,
            status="ready" if selected_case_count else "blocked_no_candidates",
            readiness_tier="validated",
            action="Use FACA-ranked optimization candidates as the screening set.",
            truth_boundary=(
                "Screening metrics are research-use FACA metrics; they do not prove "
                "calibrated sensor or lens performance."
            ),
        )
    ]
    stage_specs = [
        (
            "db_lut_anchor",
            "fdtd_sensor_stack_catalog",
            "proxy",
            "Bind candidates to selected Sensor DB records and DB/LUT provenance.",
            "Sensor DB metadata is not per-sensor calibrated physics by itself.",
        ),
        (
            "fdtd_optical_lut_batch",
            "fdtd_sensor_lut_active",
            "proxy",
            "Regenerate or attach FDTD optical-response LUTs for top candidates.",
            "FDTD LUT validation remains optical proxy until convergence and stack evidence pass.",
        ),
        (
            "tcad_collection_batch",
            "tcad_sensor_db_active",
            "calibration_required",
            "Run TCAD/DEVSIM collection checks for top sensor candidates.",
            "Current TCAD/DEVSIM summaries are not product-calibrated carrier transport.",
        ),
        (
            "rayoptics_geometric_psf_batch",
            "lens_patents_active",
            "proxy",
            "Attach RayOptics geometric PSF assets and compare against diffraction/WVF metrics.",
            "RayOptics PSFs are geometric ray histograms, not wave-optics sign-off.",
        ),
        (
            "hw_isp_trace_batch",
            "hwisp_parameter_profiles",
            "proxy",
            "Replay selected cases with HW ISP timing/control-delay profiles.",
            "Seed/public HW ISP profiles are not vendor latency sign-off without board traces.",
        ),
    ]
    need_key_by_stage = {
        "db_lut_anchor": "db_lut_anchor",
        "fdtd_optical_lut_batch": "fdtd_optical_lut",
        "tcad_collection_batch": "tcad_collection",
        "rayoptics_geometric_psf_batch": "rayoptics_geometric_psf",
        "hw_isp_trace_batch": "hw_isp_trace",
    }
    for stage_id, entry, tier, action, boundary in stage_specs:
        needed = bool(needs.get(need_key_by_stage[stage_id], False))
        stages.append(
            _physics_escalation_stage(
                stage_id,
                entry,
                needed=needed,
                readiness_tier=tier,
                action=action,
                truth_boundary=boundary,
                physics_pipeline_plan=physics_pipeline_plan,
                strict=strict,
            )
        )
    stages.extend(
        [
            _static_escalation_stage(
                "raw_dataset_export",
                needed=True,
                status="ready",
                readiness_tier="validated",
                action=(
                    "Export selected candidates with camerae2e_dataset_export_from_optimization."
                ),
                truth_boundary="RAW NPZ/metadata outputs are deterministic research artifacts.",
            ),
            _static_escalation_stage(
                "perception_eval_batch",
                needed=True,
                status="ready",
                readiness_tier="available",
                action=(
                    "Build RAW-aware and YOLO-view indexes for perception evaluation/training."
                ),
                truth_boundary=(
                    "Perception adapters do not imply dataset-specific model accuracy without "
                    "training/evaluation evidence."
                ),
            ),
        ]
    )
    return stages


def _physics_escalation_stage(
    stage_id: str,
    entry_name: str,
    *,
    needed: bool,
    readiness_tier: str,
    action: str,
    truth_boundary: str,
    physics_pipeline_plan: Mapping[str, Any] | None,
    strict: bool,
) -> dict[str, Any]:
    action_entry = _physics_action_for_entry(physics_pipeline_plan, entry_name)
    if not needed:
        status = "not_needed"
    elif action_entry is None:
        status = "planned_without_registry_evidence"
    else:
        kind = str(action_entry.get("kind", "ready"))
        severity = str(action_entry.get("severity", "info"))
        if severity == "blocking" or (strict and action_entry.get("blocks_strict_validation")):
            status = f"blocked_{kind}"
        elif kind in {"missing_asset", "stale_dependency", "calibration_required"}:
            status = f"needs_{kind}"
        else:
            status = "ready_proxy" if readiness_tier == "proxy" else "ready"
    return {
        "stage_id": stage_id,
        "needed": bool(needed),
        "status": status,
        "readiness_tier": readiness_tier,
        "registry_entry": entry_name,
        "action": action,
        "registry_action": None if action_entry is None else _jsonable(action_entry),
        "truth_boundary": truth_boundary,
    }


def _static_escalation_stage(
    stage_id: str,
    *,
    needed: bool,
    status: str,
    readiness_tier: str,
    action: str,
    truth_boundary: str,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "needed": bool(needed),
        "status": str(status),
        "readiness_tier": str(readiness_tier),
        "registry_entry": None,
        "action": str(action),
        "registry_action": None,
        "truth_boundary": str(truth_boundary),
    }


def _physics_action_for_entry(
    physics_pipeline_plan: Mapping[str, Any] | None, entry_name: str
) -> Mapping[str, Any] | None:
    if not isinstance(physics_pipeline_plan, Mapping):
        return None
    for action in physics_pipeline_plan.get("actions", []):
        if str(action.get("entry")) == entry_name:
            return action
    return None


def _escalation_validation_job(
    case: Mapping[str, Any],
    index: int,
    stages: list[Mapping[str, Any]],
) -> dict[str, Any]:
    needed_stages = [
        stage
        for stage in stages
        if stage.get("needed") and stage.get("stage_id") not in {"analytic_candidate_screen"}
    ]
    return {
        "job_id": f"candidate_{index:04d}",
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "parameters": _jsonable(case.get("parameters", {})),
        "score": case.get("score"),
        "scenario": _jsonable(case.get("scenario", {})),
        "planned_stage_ids": [str(stage.get("stage_id")) for stage in needed_stages],
        "registry_entries": [
            str(stage.get("registry_entry"))
            for stage in needed_stages
            if stage.get("registry_entry")
        ],
    }


def _escalation_acceptance_gates(
    axis_summary: Mapping[str, Any],
    stages: list[Mapping[str, Any]],
    *,
    selected_case_count: int,
) -> list[dict[str, Any]]:
    readiness_counts = dict(axis_summary.get("readiness_counts", {}))
    proxy_axis_count = int(readiness_counts.get("proxy", 0))
    calibration_axis_count = int(readiness_counts.get("calibration_required", 0))
    return [
        {
            "gate": "candidate_selection",
            "status": "pass" if selected_case_count > 0 else "fail",
            "evidence": {"selected_case_count": int(selected_case_count)},
        },
        {
            "gate": "proxy_axis_boundary",
            "status": "pass",
            "evidence": {
                "proxy_axis_count": proxy_axis_count,
                "calibration_required_axis_count": calibration_axis_count,
                "policy": (
                    "Proxy and calibration-required axes require downstream evidence "
                    "before calibrated claims."
                ),
            },
        },
        {
            "gate": "physics_stage_coverage",
            "status": "pass"
            if any(stage.get("stage_id") == "fdtd_optical_lut_batch" for stage in stages)
            else "fail",
            "evidence": {
                "needed_stage_ids": [
                    stage.get("stage_id") for stage in stages if stage.get("needed")
                ],
            },
        },
        {
            "gate": "dataset_export_path",
            "status": "pass"
            if any(stage.get("stage_id") == "raw_dataset_export" for stage in stages)
            else "fail",
            "evidence": {
                "export_api": "camerae2e_dataset_export_from_optimization",
            },
        },
    ]


def _normalize_parameter_space(
    parameter_space: Mapping[str, Iterable[Any] | Mapping[str, Any]]
) -> dict[str, list[Any]]:
    axes: dict[str, list[Any]] = {}
    for key, raw in parameter_space.items():
        values = _axis_values(raw)
        if not values:
            raise ValueError(f"Parameter axis {key!r} must contain at least one value.")
        axes[str(key)] = values
    return axes


def _normalize_candidate_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "deterministic_grid": "grid",
        "cartesian": "grid",
        "cartesian_grid": "grid",
        "grid": "grid",
        "random": "random",
        "random_search": "random",
        "sample": "random",
        "sampling": "random",
        "latin_hypercube": "latin_hypercube",
        "latin": "latin_hypercube",
        "lhs": "latin_hypercube",
        "budgeted_latin_hypercube": "latin_hypercube",
        "evolutionary": "evolutionary",
        "evolutionary_search": "evolutionary",
        "genetic": "evolutionary",
        "genetic_algorithm": "evolutionary",
        "ea": "evolutionary",
        "surrogate": "surrogate",
        "surrogate_search": "surrogate",
        "surrogate_guided": "surrogate",
        "model_guided": "surrogate",
        "bayesian_proxy": "surrogate",
        "inverse_distance_surrogate": "surrogate",
        "gaussian_process": "gaussian_process",
        "gp": "gaussian_process",
        "gp_bayesian": "gaussian_process",
        "bayesian": "gaussian_process",
        "bayesian_optimization": "gaussian_process",
        "expected_improvement": "gaussian_process",
    }
    if key not in aliases:
        raise ValueError(
            "Candidate generation method must be one of: grid, random, "
            "latin_hypercube, evolutionary, surrogate, gaussian_process."
        )
    return aliases[key]


def _candidate_budget(
    method: str, full_grid_count: int, max_cases: int | None
) -> tuple[int, bool]:
    if max_cases is None:
        if method == "grid":
            return full_grid_count, False
        return min(full_grid_count, 32), True
    budget = int(max_cases)
    if budget <= 0:
        raise ValueError("max_cases must be a positive integer.")
    return min(budget, full_grid_count), False


def _full_grid_count(axes: Mapping[str, list[Any]]) -> int:
    count = 1
    for values in axes.values():
        count *= len(values)
    return int(count)


def _generate_parameter_candidates(
    axes: Mapping[str, list[Any]],
    *,
    method: str,
    max_cases: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    if method == "grid":
        return _grid_candidates(axes, max_cases)
    if method == "random":
        return _random_candidates(axes, max_cases, seed)
    if method == "latin_hypercube":
        return _latin_hypercube_candidates(axes, max_cases, seed)
    if method == "evolutionary":
        return _evolutionary_seed_candidates(
            axes,
            _evolutionary_population_size(axes, max_cases),
            seed,
        )
    if method == "surrogate":
        return _surrogate_seed_candidates(
            axes,
            _surrogate_seed_count(axes, max_cases),
            seed,
        )
    if method == "gaussian_process":
        return _gp_seed_candidates(
            axes,
            _gp_seed_count(axes, max_cases),
            seed,
        )
    raise ValueError(f"Unsupported candidate generation method: {method!r}.")


def _grid_candidates(axes: Mapping[str, list[Any]], max_cases: int) -> list[dict[str, Any]]:
    full_grid_count = _full_grid_count(axes)
    indices = (
        range(full_grid_count)
        if max_cases >= full_grid_count
        else _even_grid_indices(full_grid_count, max_cases)
    )
    return [_candidate_from_grid_index(axes, index) for index in indices]


def _even_grid_indices(full_grid_count: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    indices = [
        int((index * (full_grid_count - 1)) // (count - 1))
        for index in range(count)
    ]
    result = []
    seen = set()
    for index in indices:
        if index not in seen:
            result.append(index)
            seen.add(index)
    fill = 0
    while len(result) < count and fill < full_grid_count:
        if fill not in seen:
            result.append(fill)
            seen.add(fill)
        fill += 1
    return result


def _candidate_from_grid_index(
    axes: Mapping[str, list[Any]], flat_index: int
) -> dict[str, Any]:
    keys = list(axes)
    remaining = int(flat_index)
    reversed_values: dict[str, Any] = {}
    for key in reversed(keys):
        values = axes[key]
        local_index = remaining % len(values)
        remaining //= len(values)
        reversed_values[key] = values[local_index]
    return {key: reversed_values[key] for key in keys}


def _random_candidates(
    axes: Mapping[str, list[Any]], max_cases: int, seed: int
) -> list[dict[str, Any]]:
    full_grid_count = _full_grid_count(axes)
    if max_cases >= full_grid_count:
        return _grid_candidates(axes, full_grid_count)
    rng = np.random.default_rng(int(seed))
    candidates = []
    seen = set()
    attempts = 0
    max_attempts = max(100, max_cases * 20)
    while len(candidates) < max_cases and attempts < max_attempts:
        candidate = {
            key: values[int(rng.integers(0, len(values)))]
            for key, values in axes.items()
        }
        attempts += 1
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        candidates.append(candidate)
        seen.add(signature)
    return _dedupe_and_fill_candidates(candidates, axes, max_cases)


def _latin_hypercube_candidates(
    axes: Mapping[str, list[Any]], max_cases: int, seed: int
) -> list[dict[str, Any]]:
    full_grid_count = _full_grid_count(axes)
    if max_cases >= full_grid_count:
        return _grid_candidates(axes, full_grid_count)
    rng = np.random.default_rng(int(seed))
    columns: dict[str, np.ndarray] = {}
    for key, values in axes.items():
        if max_cases == 1:
            strata = np.asarray([0.5], dtype=float)
        else:
            strata = (np.arange(max_cases, dtype=float) + rng.random(max_cases)) / max_cases
            rng.shuffle(strata)
        indices = np.floor(strata * len(values)).astype(int)
        columns[key] = np.clip(indices, 0, len(values) - 1)
    candidates = []
    for row_index in range(max_cases):
        candidates.append(
            {
                key: axes[key][int(columns[key][row_index])]
                for key in axes
            }
        )
    return _dedupe_and_fill_candidates(candidates, axes, max_cases)


def _evolutionary_population_size(axes: Mapping[str, list[Any]], budget: int) -> int:
    if not axes:
        return 1
    axis_count = max(len(axes), 1)
    full_grid_count = _full_grid_count(axes)
    requested = max(4, 2 * axis_count)
    return int(min(max(int(budget), 1), full_grid_count, requested))


def _evolutionary_search_config(
    axes: Mapping[str, list[Any]],
    budget: int,
    population_size: int | None = None,
) -> dict[str, Any]:
    if population_size is None:
        population_size = _evolutionary_population_size(axes, budget)
    axis_count = max(len(axes), 1)
    return {
        "population_size": int(population_size),
        "elite_count": int(max(1, min(population_size, math.ceil(population_size / 3.0)))),
        "mutation_probability": float(min(0.5, max(0.15, 1.0 / axis_count))),
        "selection": "score_ranked_elite",
        "crossover": "uniform_discrete_axis",
        "budget": int(budget),
    }


def _adaptive_search_config(
    method: str,
    axes: Mapping[str, list[Any]],
    budget: int,
    seed_count: int | None,
) -> dict[str, Any]:
    if method == "evolutionary":
        return _evolutionary_search_config(axes, budget, seed_count)
    if method == "surrogate":
        return _surrogate_search_config(axes, budget, seed_count)
    if method == "gaussian_process":
        return _gp_search_config(axes, budget, seed_count)
    return {}


def _evolutionary_seed_candidates(
    axes: Mapping[str, list[Any]], population_size: int, seed: int
) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    candidates: list[dict[str, Any]] = []
    full_grid_count = _full_grid_count(axes)
    if full_grid_count:
        candidates.append(_candidate_from_grid_index(axes, 0))
    if full_grid_count > 1:
        candidates.append(_candidate_from_grid_index(axes, full_grid_count - 1))
    candidates.extend(_latin_hypercube_candidates(axes, int(population_size), seed))
    candidates.extend(_random_candidates(axes, int(population_size), int(seed) + 991))
    return _dedupe_and_fill_candidates(candidates, axes, int(population_size))


def _surrogate_seed_count(axes: Mapping[str, list[Any]], budget: int) -> int:
    if not axes:
        return 1
    full_grid_count = _full_grid_count(axes)
    axis_count = max(len(axes), 1)
    requested = max(4, 2 * axis_count)
    return int(min(max(int(budget), 1), full_grid_count, requested))


def _surrogate_search_config(
    axes: Mapping[str, list[Any]],
    budget: int,
    seed_count: int | None = None,
) -> dict[str, Any]:
    if seed_count is None:
        seed_count = _surrogate_seed_count(axes, budget)
    full_grid_count = _full_grid_count(axes)
    candidate_pool_size = int(min(full_grid_count, max(256, int(budget) * 64)))
    return {
        "seed_count": int(seed_count),
        "candidate_pool_size": candidate_pool_size,
        "exploration_weight": 0.35,
        "length_scale": 0.75,
        "model": "discrete_rbf_inverse_distance_surrogate",
        "acquisition": "expected_improvement_proxy_plus_uncertainty",
        "truth_boundary": (
            "Surrogate search is a deterministic discrete-axis acquisition proxy, "
            "not a calibrated Gaussian-process Bayesian optimizer."
        ),
        "budget": int(budget),
    }


def _surrogate_seed_candidates(
    axes: Mapping[str, list[Any]], seed_count: int, seed: int
) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    candidates: list[dict[str, Any]] = []
    full_grid_count = _full_grid_count(axes)
    if full_grid_count:
        candidates.append(_candidate_from_grid_index(axes, 0))
    if full_grid_count > 1:
        candidates.append(_candidate_from_grid_index(axes, full_grid_count - 1))
    candidates.extend(_latin_hypercube_candidates(axes, int(seed_count), int(seed) + 101))
    candidates.extend(_random_candidates(axes, int(seed_count), int(seed) + 202))
    return _dedupe_and_fill_candidates(candidates, axes, int(seed_count))


def _gp_seed_count(axes: Mapping[str, list[Any]], budget: int) -> int:
    if not axes:
        return 1
    full_grid_count = _full_grid_count(axes)
    axis_count = max(len(axes), 1)
    requested = max(5, 2 * axis_count + 1)
    return int(min(max(int(budget), 1), full_grid_count, requested))


def _gp_search_config(
    axes: Mapping[str, list[Any]],
    budget: int,
    seed_count: int | None = None,
) -> dict[str, Any]:
    if seed_count is None:
        seed_count = _gp_seed_count(axes, budget)
    full_grid_count = _full_grid_count(axes)
    candidate_pool_size = int(min(full_grid_count, max(256, int(budget) * 64)))
    return {
        "seed_count": int(seed_count),
        "candidate_pool_size": candidate_pool_size,
        "kernel": "rbf",
        "model": "gaussian_process_rbf",
        "acquisition": "expected_improvement",
        "length_scale": 0.65,
        "signal_variance": 1.0,
        "noise_variance": 1.0e-6,
        "jitter": 1.0e-9,
        "xi": 0.01,
        "truth_boundary": (
            "Gaussian-process search models the discrete FACA objective surface; "
            "it is not a calibrated physical sensor, lens, or ISP model."
        ),
        "budget": int(budget),
    }


def _gp_seed_candidates(
    axes: Mapping[str, list[Any]], seed_count: int, seed: int
) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    candidates: list[dict[str, Any]] = []
    full_grid_count = _full_grid_count(axes)
    if full_grid_count:
        candidates.append(_candidate_from_grid_index(axes, 0))
    if full_grid_count > 1:
        candidates.append(_candidate_from_grid_index(axes, full_grid_count - 1))
    middle = full_grid_count // 2
    if full_grid_count > 2:
        candidates.append(_candidate_from_grid_index(axes, middle))
    candidates.extend(_latin_hypercube_candidates(axes, int(seed_count), int(seed) + 313))
    candidates.extend(_random_candidates(axes, int(seed_count), int(seed) + 626))
    return _dedupe_and_fill_candidates(candidates, axes, int(seed_count))


def _run_evolutionary_parameter_search(
    candidate_plan: Mapping[str, Any],
    axes: Mapping[str, list[Any]],
    *,
    base_scenario: Mapping[str, Any] | None,
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
    constraint_specs: list[dict[str, Any]],
    scene: Scene | str | Mapping[str, Any] | None,
    camera: Camera | None,
    asset_store: AssetStore | None,
    seed: int,
    include_arrays: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget = int(candidate_plan.get("max_cases", 1))
    config = dict(candidate_plan.get("search_config") or _evolutionary_search_config(axes, budget))
    population_size = int(
        config.get("population_size", _evolutionary_population_size(axes, budget))
    )
    elite_count = int(config.get("elite_count", max(1, population_size // 3)))
    mutation_probability = float(config.get("mutation_probability", 0.25))
    rng = np.random.default_rng(int(seed) + 1729)
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    search_trace: list[dict[str, Any]] = []
    pending = [dict(candidate) for candidate in candidate_plan.get("candidates", [])]
    generation = 0
    full_grid_count = _full_grid_count(axes)

    while len(cases) < budget and len(seen) < full_grid_count:
        generation_cases: list[dict[str, Any]] = []
        for candidate in pending:
            if len(cases) >= budget:
                break
            signature = _candidate_signature(candidate)
            if signature in seen:
                continue
            case = _evaluate_parameter_candidate(
                candidate,
                case_index=len(cases),
                base_scenario=base_scenario,
                objective_specs=objective_specs,
                objective=objective,
                constraint_specs=constraint_specs,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=seed,
                include_arrays=include_arrays,
            )
            cases.append(case)
            generation_cases.append(case)
            seen.add(signature)

        ranked = _rank_cases_for_search(cases)
        search_trace.append(
            _evolutionary_trace_entry(
                generation,
                generation_cases,
                ranked,
                evaluated_count=len(cases),
            )
        )
        if len(cases) >= budget or len(seen) >= full_grid_count:
            break
        pending = _evolutionary_next_candidates(
            axes,
            ranked,
            seen,
            population_size=min(population_size, budget - len(cases)),
            elite_count=elite_count,
            mutation_probability=mutation_probability,
            rng=rng,
        )
        if not pending:
            break
        generation += 1

    executed_plan = dict(candidate_plan)
    executed_plan["case_count"] = len(cases)
    executed_plan["truncated"] = len(cases) < full_grid_count
    executed_plan["candidates"] = [_jsonable(case.get("parameters", {})) for case in cases]
    executed_plan["search_config"] = _jsonable(config)
    executed_plan["search_trace"] = _jsonable(search_trace)
    executed_plan["executed_generation_count"] = len(search_trace)
    return cases, executed_plan


def _run_surrogate_parameter_search(
    candidate_plan: Mapping[str, Any],
    axes: Mapping[str, list[Any]],
    *,
    base_scenario: Mapping[str, Any] | None,
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
    constraint_specs: list[dict[str, Any]],
    scene: Scene | str | Mapping[str, Any] | None,
    camera: Camera | None,
    asset_store: AssetStore | None,
    seed: int,
    include_arrays: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget = int(candidate_plan.get("max_cases", 1))
    config = dict(candidate_plan.get("search_config") or _surrogate_search_config(axes, budget))
    pool = _surrogate_candidate_pool(
        axes,
        pool_size=int(config.get("candidate_pool_size", max(256, budget * 64))),
        seed=int(seed) + 303,
    )
    pending = [dict(candidate) for candidate in candidate_plan.get("candidates", [])]
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    search_trace: list[dict[str, Any]] = []
    generation = 0
    full_grid_count = _full_grid_count(axes)

    while len(cases) < budget and len(seen) < full_grid_count:
        generation_cases: list[dict[str, Any]] = []
        for candidate in pending:
            if len(cases) >= budget:
                break
            signature = _candidate_signature(candidate)
            if signature in seen:
                continue
            case = _evaluate_parameter_candidate(
                candidate,
                case_index=len(cases),
                base_scenario=base_scenario,
                objective_specs=objective_specs,
                objective=objective,
                constraint_specs=constraint_specs,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=seed,
                include_arrays=include_arrays,
            )
            cases.append(case)
            generation_cases.append(case)
            seen.add(signature)

        ranked = _rank_cases_for_search(cases)
        search_trace.append(
            _surrogate_trace_entry(
                generation,
                generation_cases,
                ranked,
                evaluated_count=len(cases),
                pool_size=len(pool),
            )
        )
        if len(cases) >= budget or len(seen) >= full_grid_count:
            break
        next_candidate = _surrogate_next_candidate(
            axes,
            pool,
            cases,
            seen,
            exploration_weight=float(config.get("exploration_weight", 0.35)),
            length_scale=float(config.get("length_scale", 0.75)),
        )
        if next_candidate is None:
            break
        pending = [next_candidate]
        generation += 1

    executed_plan = dict(candidate_plan)
    executed_plan["case_count"] = len(cases)
    executed_plan["truncated"] = len(cases) < full_grid_count
    executed_plan["candidates"] = [_jsonable(case.get("parameters", {})) for case in cases]
    executed_plan["search_config"] = _jsonable(config)
    executed_plan["search_trace"] = _jsonable(search_trace)
    executed_plan["executed_generation_count"] = len(search_trace)
    return cases, executed_plan


def _run_gaussian_process_parameter_search(
    candidate_plan: Mapping[str, Any],
    axes: Mapping[str, list[Any]],
    *,
    base_scenario: Mapping[str, Any] | None,
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
    constraint_specs: list[dict[str, Any]],
    scene: Scene | str | Mapping[str, Any] | None,
    camera: Camera | None,
    asset_store: AssetStore | None,
    seed: int,
    include_arrays: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget = int(candidate_plan.get("max_cases", 1))
    config = dict(candidate_plan.get("search_config") or _gp_search_config(axes, budget))
    pool = _surrogate_candidate_pool(
        axes,
        pool_size=int(config.get("candidate_pool_size", max(256, budget * 64))),
        seed=int(seed) + 919,
    )
    pending = [dict(candidate) for candidate in candidate_plan.get("candidates", [])]
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    search_trace: list[dict[str, Any]] = []
    generation = 0
    full_grid_count = _full_grid_count(axes)

    while len(cases) < budget and len(seen) < full_grid_count:
        generation_cases: list[dict[str, Any]] = []
        for candidate in pending:
            if len(cases) >= budget:
                break
            signature = _candidate_signature(candidate)
            if signature in seen:
                continue
            case = _evaluate_parameter_candidate(
                candidate,
                case_index=len(cases),
                base_scenario=base_scenario,
                objective_specs=objective_specs,
                objective=objective,
                constraint_specs=constraint_specs,
                scene=scene,
                camera=camera,
                asset_store=asset_store,
                seed=seed,
                include_arrays=include_arrays,
            )
            cases.append(case)
            generation_cases.append(case)
            seen.add(signature)

        ranked = _rank_cases_for_search(cases)
        acquisition_summary: dict[str, Any] = {}
        if len(cases) >= budget or len(seen) >= full_grid_count:
            next_candidate = None
        else:
            next_candidate, acquisition_summary = _gp_next_candidate(
                axes,
                pool,
                cases,
                seen,
                length_scale=float(config.get("length_scale", 0.65)),
                signal_variance=float(config.get("signal_variance", 1.0)),
                noise_variance=float(config.get("noise_variance", 1.0e-6)),
                jitter=float(config.get("jitter", 1.0e-9)),
                xi=float(config.get("xi", 0.01)),
            )
        search_trace.append(
            _gp_trace_entry(
                generation,
                generation_cases,
                ranked,
                evaluated_count=len(cases),
                pool_size=len(pool),
                acquisition_summary=acquisition_summary,
            )
        )
        if len(cases) >= budget or len(seen) >= full_grid_count:
            break
        if next_candidate is None:
            break
        pending = [next_candidate]
        generation += 1

    executed_plan = dict(candidate_plan)
    executed_plan["case_count"] = len(cases)
    executed_plan["truncated"] = len(cases) < full_grid_count
    executed_plan["candidates"] = [_jsonable(case.get("parameters", {})) for case in cases]
    executed_plan["search_config"] = _jsonable(config)
    executed_plan["search_trace"] = _jsonable(search_trace)
    executed_plan["executed_generation_count"] = len(search_trace)
    return cases, executed_plan


def _evaluate_parameter_candidate(
    axis_values: Mapping[str, Any],
    *,
    case_index: int,
    base_scenario: Mapping[str, Any] | None,
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
    constraint_specs: list[dict[str, Any]],
    scene: Scene | str | Mapping[str, Any] | None,
    camera: Camera | None,
    asset_store: AssetStore | None,
    seed: int,
    include_arrays: bool,
) -> dict[str, Any]:
    scenario = _deep_dict(base_scenario or {})
    scenario.setdefault("name", "camerae2e_parameter_optimization")
    for key, value in axis_values.items():
        _assign_parameter(scenario, key, value)
    report = camerae2e_faca_report(
        camerae2e_run_scenario(
            scenario,
            scene=scene,
            camera=camera,
            asset_store=asset_store,
            seed=int(seed) + int(case_index),
            include_arrays=include_arrays,
        )
    )
    objective_values, objective_utilities, score = _score_report(
        report, objective_specs, objective
    )
    constraint_results = [_evaluate_constraint(report, item) for item in constraint_specs]
    feasible = all(item["pass"] for item in constraint_results)
    if not feasible:
        score = -math.inf
    return {
        "case_index": int(case_index),
        "seed": int(seed) + int(case_index),
        "parameters": _jsonable(axis_values),
        "score": float(score),
        "feasible": bool(feasible),
        "objective_values": _jsonable(objective_values),
        "objective_utilities": _jsonable(objective_utilities),
        "constraint_results": _jsonable(constraint_results),
        "scenario": _jsonable(report.get("scenario", {})),
        "report": report,
    }


def _rank_cases_for_search(cases: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        cases,
        key=lambda item: (
            bool(item.get("feasible", True)),
            float(item.get("score", -math.inf)),
            -int(item.get("case_index", 0)),
        ),
        reverse=True,
    )


def _evolutionary_next_candidates(
    axes: Mapping[str, list[Any]],
    ranked_cases: list[Mapping[str, Any]],
    seen: set[str],
    *,
    population_size: int,
    elite_count: int,
    mutation_probability: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not axes or population_size <= 0 or not ranked_cases:
        return []
    elites = [dict(item.get("parameters", {})) for item in ranked_cases[: max(elite_count, 1)]]
    candidates: list[dict[str, Any]] = []
    for elite in elites[:population_size]:
        candidates.append(_mutate_candidate(elite, axes, mutation_probability, rng))
    attempts = 0
    while len(candidates) < population_size and attempts < max(100, population_size * 30):
        parent_a = elites[int(rng.integers(0, len(elites)))]
        parent_b = elites[int(rng.integers(0, len(elites)))]
        child = {}
        for key, values in axes.items():
            inherited = parent_a.get(key) if rng.random() < 0.5 else parent_b.get(key)
            if inherited is None or rng.random() < mutation_probability:
                inherited = values[int(rng.integers(0, len(values)))]
            child[key] = inherited
        candidates.append(child)
        attempts += 1
    return _dedupe_new_candidates(candidates, axes, population_size, seen)


def _surrogate_candidate_pool(
    axes: Mapping[str, list[Any]],
    *,
    pool_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    full_grid_count = _full_grid_count(axes)
    if full_grid_count <= int(pool_size):
        return _grid_candidates(axes, full_grid_count)
    candidates: list[dict[str, Any]] = []
    candidates.extend(_latin_hypercube_candidates(axes, int(pool_size), seed))
    candidates.extend(_random_candidates(axes, int(pool_size), int(seed) + 991))
    return _dedupe_and_fill_candidates(candidates, axes, int(pool_size))


def _surrogate_next_candidate(
    axes: Mapping[str, list[Any]],
    candidate_pool: Iterable[Mapping[str, Any]],
    cases: Iterable[Mapping[str, Any]],
    seen: set[str],
    *,
    exploration_weight: float,
    length_scale: float,
) -> dict[str, Any] | None:
    evaluated = [case for case in cases if case.get("parameters") is not None]
    if not evaluated:
        return None
    observed_vectors = np.asarray(
        [_candidate_vector(case["parameters"], axes) for case in evaluated],
        dtype=float,
    )
    observed_scores = _surrogate_score_array(evaluated)
    best_score = float(np.max(observed_scores)) if observed_scores.size else 0.0
    score_scale = float(np.std(observed_scores)) if observed_scores.size > 1 else 1.0
    score_scale = max(score_scale, 1.0e-9)
    best_candidate: dict[str, Any] | None = None
    best_acquisition = -math.inf
    for candidate in candidate_pool:
        payload = dict(candidate)
        signature = _candidate_signature(payload)
        if signature in seen:
            continue
        vector = _candidate_vector(payload, axes)
        distances = np.linalg.norm(observed_vectors - vector, axis=1)
        weights = np.exp(-np.square(distances) / max(2.0 * length_scale * length_scale, 1.0e-12))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-12:
            prediction = float(np.mean(observed_scores))
        else:
            prediction = float(np.dot(weights, observed_scores) / weight_sum)
        uncertainty = float(min(np.min(distances) / max(math.sqrt(max(len(axes), 1)), 1.0), 1.0))
        improvement_proxy = max(prediction - best_score, 0.0)
        acquisition = improvement_proxy + (float(exploration_weight) * score_scale * uncertainty)
        if acquisition > best_acquisition:
            best_acquisition = acquisition
            best_candidate = payload
    return best_candidate


def _gp_next_candidate(
    axes: Mapping[str, list[Any]],
    candidate_pool: Iterable[Mapping[str, Any]],
    cases: Iterable[Mapping[str, Any]],
    seen: set[str],
    *,
    length_scale: float,
    signal_variance: float,
    noise_variance: float,
    jitter: float,
    xi: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    evaluated = [case for case in cases if case.get("parameters") is not None]
    pool = [
        dict(candidate)
        for candidate in candidate_pool
        if _candidate_signature(dict(candidate)) not in seen
    ]
    if not evaluated or not pool:
        return None, {"status": "empty"}
    observed_vectors = np.asarray(
        [_candidate_vector(case["parameters"], axes) for case in evaluated],
        dtype=float,
    )
    observed_scores = _surrogate_score_array(evaluated)
    candidate_vectors = np.asarray(
        [_candidate_vector(candidate, axes) for candidate in pool],
        dtype=float,
    )
    mean, std = _gp_posterior(
        observed_vectors,
        observed_scores,
        candidate_vectors,
        length_scale=max(float(length_scale), 1.0e-9),
        signal_variance=max(float(signal_variance), 1.0e-12),
        noise_variance=max(float(noise_variance), 0.0),
        jitter=max(float(jitter), 0.0),
    )
    best_score = float(np.max(observed_scores)) if observed_scores.size else 0.0
    expected_improvement = _expected_improvement(
        mean,
        std,
        best_score=best_score,
        xi=float(xi),
    )
    if expected_improvement.size == 0:
        return None, {"status": "empty_acquisition"}
    order = np.lexsort((-std, -mean, -expected_improvement))
    best_index = int(order[0])
    selected = pool[best_index]
    return selected, {
        "status": "selected",
        "selected_parameters": _jsonable(selected),
        "selected_expected_improvement": float(expected_improvement[best_index]),
        "selected_predicted_mean": float(mean[best_index]),
        "selected_predicted_std": float(std[best_index]),
        "best_observed_score": best_score,
        "candidate_pool_size": len(pool),
    }


def _gp_posterior(
    observed_vectors: np.ndarray,
    observed_scores: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    length_scale: float,
    signal_variance: float,
    noise_variance: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    if observed_vectors.size == 0 or candidate_vectors.size == 0:
        return (
            np.zeros((candidate_vectors.shape[0],), dtype=float),
            np.zeros((candidate_vectors.shape[0],), dtype=float),
        )
    score_mean = float(np.mean(observed_scores))
    score_scale = float(np.std(observed_scores))
    score_scale = max(score_scale, 1.0e-9)
    y = (observed_scores - score_mean) / score_scale
    kernel = _rbf_kernel(
        observed_vectors,
        observed_vectors,
        length_scale=length_scale,
        signal_variance=signal_variance,
    )
    diag_noise = max(float(noise_variance), 0.0) + max(float(jitter), 0.0)
    for attempt in range(8):
        try:
            noise = diag_noise + (10.0**attempt * max(float(jitter), 1.0e-12))
            factor = np.linalg.cholesky(
                kernel + np.eye(kernel.shape[0], dtype=float) * noise
            )
            alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
            cross = _rbf_kernel(
                candidate_vectors,
                observed_vectors,
                length_scale=length_scale,
                signal_variance=signal_variance,
            )
            mean_norm = cross @ alpha
            solve = np.linalg.solve(factor, cross.T)
            variance_norm = np.maximum(
                float(signal_variance) - np.sum(np.square(solve), axis=0),
                0.0,
            )
            return (
                score_mean + score_scale * mean_norm,
                score_scale * np.sqrt(variance_norm),
            )
        except np.linalg.LinAlgError:
            continue
    fallback_mean = np.full((candidate_vectors.shape[0],), score_mean, dtype=float)
    fallback_std = np.full((candidate_vectors.shape[0],), score_scale, dtype=float)
    return fallback_mean, fallback_std


def _rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    length_scale: float,
    signal_variance: float,
) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    diff = left[:, None, :] - right[None, :, :]
    squared_distance = np.sum(np.square(diff), axis=2)
    return float(signal_variance) * np.exp(
        -0.5 * squared_distance / max(length_scale * length_scale, 1.0e-12)
    )


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best_score: float,
    xi: float,
) -> np.ndarray:
    improvement = np.asarray(mean, dtype=float) - float(best_score) - float(xi)
    std = np.maximum(np.asarray(std, dtype=float), 1.0e-12)
    z = improvement / std
    cdf = _normal_cdf(z)
    pdf = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
    return np.maximum(improvement * cdf + std * pdf, 0.0)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=float).reshape(-1)
    cdf = np.asarray(
        [0.5 * (1.0 + math.erf(float(item) / math.sqrt(2.0))) for item in flat],
        dtype=float,
    )
    return cdf.reshape(np.asarray(value).shape)


def _candidate_vector(candidate: Mapping[str, Any], axes: Mapping[str, list[Any]]) -> np.ndarray:
    values = []
    for key, axis_values in axes.items():
        value_to_index = {
            _value_signature(value): index
            for index, value in enumerate(axis_values)
        }
        index = value_to_index.get(_value_signature(candidate.get(key)), 0)
        denominator = max(len(axis_values) - 1, 1)
        values.append(float(index) / float(denominator))
    return np.asarray(values, dtype=float)


def _surrogate_score_array(cases: Iterable[Mapping[str, Any]]) -> np.ndarray:
    raw_scores = np.asarray(
        [float(case.get("score", -math.inf)) for case in cases],
        dtype=float,
    )
    finite = raw_scores[np.isfinite(raw_scores)]
    if finite.size == 0:
        return np.zeros(raw_scores.shape, dtype=float)
    floor = float(np.min(finite)) - max(float(np.std(finite)), 1.0)
    return np.where(np.isfinite(raw_scores), raw_scores, floor)


def _mutate_candidate(
    candidate: Mapping[str, Any],
    axes: Mapping[str, list[Any]],
    mutation_probability: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    mutated = dict(candidate)
    forced_key = list(axes)[int(rng.integers(0, len(axes)))] if axes else None
    for key, values in axes.items():
        if key == forced_key or rng.random() < mutation_probability:
            mutated[key] = values[int(rng.integers(0, len(values)))]
        else:
            mutated.setdefault(key, values[0])
    return mutated


def _dedupe_new_candidates(
    candidates: Iterable[Mapping[str, Any]],
    axes: Mapping[str, list[Any]],
    target_count: int,
    seen: set[str],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    local_seen = set(seen)
    for candidate in candidates:
        payload = dict(candidate)
        signature = _candidate_signature(payload)
        if signature in local_seen:
            continue
        unique.append(payload)
        local_seen.add(signature)
        if len(unique) >= target_count:
            return unique
    for flat_index in range(_full_grid_count(axes)):
        payload = _candidate_from_grid_index(axes, flat_index)
        signature = _candidate_signature(payload)
        if signature in local_seen:
            continue
        unique.append(payload)
        local_seen.add(signature)
        if len(unique) >= target_count:
            break
    return unique


def _evolutionary_trace_entry(
    generation: int,
    generation_cases: list[Mapping[str, Any]],
    ranked_cases: list[Mapping[str, Any]],
    *,
    evaluated_count: int,
) -> dict[str, Any]:
    best = ranked_cases[0] if ranked_cases else {}
    return {
        "generation": int(generation),
        "candidate_count": len(generation_cases),
        "evaluated_count": int(evaluated_count),
        "best_case_index": best.get("case_index"),
        "best_score": best.get("score"),
        "best_feasible": best.get("feasible"),
        "best_parameters": _jsonable(best.get("parameters", {})),
    }


def _surrogate_trace_entry(
    generation: int,
    generation_cases: list[Mapping[str, Any]],
    ranked_cases: list[Mapping[str, Any]],
    *,
    evaluated_count: int,
    pool_size: int,
) -> dict[str, Any]:
    best = ranked_cases[0] if ranked_cases else {}
    selected = generation_cases[-1] if generation_cases else {}
    return {
        "generation": int(generation),
        "candidate_count": len(generation_cases),
        "evaluated_count": int(evaluated_count),
        "candidate_pool_size": int(pool_size),
        "best_case_index": best.get("case_index"),
        "best_score": best.get("score"),
        "best_feasible": best.get("feasible"),
        "best_parameters": _jsonable(best.get("parameters", {})),
        "selected_case_index": selected.get("case_index"),
        "selected_score": selected.get("score"),
        "selected_parameters": _jsonable(selected.get("parameters", {})),
    }


def _gp_trace_entry(
    generation: int,
    generation_cases: list[Mapping[str, Any]],
    ranked_cases: list[Mapping[str, Any]],
    *,
    evaluated_count: int,
    pool_size: int,
    acquisition_summary: Mapping[str, Any],
) -> dict[str, Any]:
    best = ranked_cases[0] if ranked_cases else {}
    selected = generation_cases[-1] if generation_cases else {}
    return {
        "generation": int(generation),
        "candidate_count": len(generation_cases),
        "evaluated_count": int(evaluated_count),
        "candidate_pool_size": int(pool_size),
        "model": "gaussian_process_rbf",
        "acquisition": "expected_improvement",
        "best_case_index": best.get("case_index"),
        "best_score": best.get("score"),
        "best_feasible": best.get("feasible"),
        "best_parameters": _jsonable(best.get("parameters", {})),
        "selected_case_index": selected.get("case_index"),
        "selected_score": selected.get("score"),
        "selected_parameters": _jsonable(selected.get("parameters", {})),
        "next_candidate": _jsonable(dict(acquisition_summary)),
    }


def _dedupe_and_fill_candidates(
    candidates: Iterable[Mapping[str, Any]],
    axes: Mapping[str, list[Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        payload = dict(candidate)
        signature = _candidate_signature(payload)
        if signature in seen:
            continue
        unique.append(payload)
        seen.add(signature)
        if len(unique) >= target_count:
            return unique
    for flat_index in range(_full_grid_count(axes)):
        payload = _candidate_from_grid_index(axes, flat_index)
        signature = _candidate_signature(payload)
        if signature in seen:
            continue
        unique.append(payload)
        seen.add(signature)
        if len(unique) >= target_count:
            break
    return unique


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    return str(_jsonable(candidate))


def _value_signature(value: Any) -> str:
    return str(_jsonable(value))


def _candidate_plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in plan.items() if key != "candidates"}
    payload["candidate_preview"] = _jsonable(list(plan.get("candidates", []))[:5])
    return _jsonable(payload)


def _optimization_method_name(candidate_plan: Mapping[str, Any]) -> str:
    method = str(candidate_plan.get("method", "grid"))
    truncated = bool(candidate_plan.get("truncated", False))
    if method == "grid" and not truncated:
        return "deterministic_grid"
    if method == "grid":
        return "budgeted_grid"
    return f"budgeted_{method}"


def _validate_parameter_axis_names(
    axes: Mapping[str, Any], base_scenario: Mapping[str, Any] | None
) -> None:
    for key, values in axes.items():
        issues = _parameter_axis_issues(str(key), base_scenario)
        value_issues, _ = _parameter_value_findings(str(key), values)
        issues.extend(value_issues)
        if issues:
            raise ValueError(str(issues[0]["message"]))


def _parameter_axis_issues(
    path: str, base_scenario: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    if "." not in path:
        return []
    prefix, remainder = path.split(".", 1)
    if prefix == "sensor" and remainder not in _SENSOR_OPTIMIZATION_PARAMETERS:
        return [
            {
                "path": path,
                "kind": "unsupported_sensor_axis",
                "message": f"Unsupported sensor optimization parameter: {path!r}.",
            }
        ]
    if prefix == "fdtd" and remainder not in {"lut", "enabled"} and not _has_nested_value(
        base_scenario, "fdtd", "lut"
    ):
        return [
            {
                "path": path,
                "kind": "inactive_fdtd_axis",
                "message": (
                    f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                    "optimizing FDTD options alone would not change the camera pipeline."
                ),
            }
        ]
    if prefix == "tcad" and remainder == "collection_mode" and not _has_nested_value(
        base_scenario, "tcad", "db"
    ):
        return [
            {
                "path": path,
                "kind": "inactive_tcad_axis",
                "message": (
                    f"{path!r} needs an explicit LUT/DB attachment in the base scenario; "
                    "optimizing the mode alone would not change the camera pipeline."
                ),
            }
        ]
    return []


def _parameter_value_findings(
    path: str, values: Iterable[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    key = _canonical_parameter_path(path)

    positive_scalar_axes = {
        "sensor.integration_time",
        "sensor.analog_gain",
        "sensor.pixel_read_noise_v",
        "sensor.pixel_dark_voltage",
        "sensor.pixel_voltage_swing",
        "sensor.pixel_conversion_gain",
        "optics.fnumber",
        "optics.focal_length",
        "optics.si_psf_radius_um",
        "optics.psf_angle_step",
        "optics.rt_compute_spacing",
        "sensor.ocl_fnumber",
        "sensor.ocl_focal_length_um",
        "sensor.ocl_refractive_index",
        "hw_isp.global_latency_factor",
    }
    finite_scalar_axes = {
        "fdtd.ocl_shift_um",
        "fdtd.cra_x_deg",
        "fdtd.cra_z_deg",
    }
    nonnegative_scalar_axes = {"fdtd.crosstalk_strength"}
    fraction_axes = {"sensor.ocl_group_equalization"}
    nonnegative_integer_axes = {
        "hw_isp.ae_apply_delay_frames",
        "hw_isp.awb_apply_delay_frames",
    }

    for index, value in enumerate(values):
        if key in positive_scalar_axes:
            issues.extend(
                _validate_numeric_value(
                    path,
                    index,
                    value,
                    positive=True,
                    kind="invalid_positive_value",
                )
            )
        elif key in nonnegative_scalar_axes:
            issues.extend(
                _validate_numeric_value(
                    path,
                    index,
                    value,
                    nonnegative=True,
                    kind="invalid_nonnegative_value",
                )
            )
            number = _scalar_float(value)
            if number is not None and number > 1.0:
                warnings.append(
                    _value_finding(
                        path,
                        index,
                        "extrapolating_crosstalk_strength",
                        "FDTD crosstalk_strength > 1 extrapolates beyond the LUT kernel.",
                    )
                )
        elif key in nonnegative_integer_axes:
            issues.extend(
                _validate_integer_value(
                    path,
                    index,
                    value,
                    nonnegative=True,
                    kind="invalid_nonnegative_integer",
                )
            )
        elif key == "sensor.pixel_size":
            issues.extend(_validate_pixel_size(path, index, value))
        elif key == "sensor.pixel_fill_factor":
            issues.extend(_validate_fill_factor(path, index, value))
        elif key == "sensor.n_samples_per_pixel":
            issues.extend(_validate_n_samples_per_pixel(path, index, value))
        elif key == "sensor.cfa_pattern":
            cfa_issues, cfa_warnings = _validate_cfa_pattern(path, index, value)
            issues.extend(cfa_issues)
            warnings.extend(cfa_warnings)
        elif key == "sensor.cfa_preset":
            issues.extend(_validate_enum_value(path, index, value, _CFA_PRESETS))
        elif key == "sensor.binning_method":
            issues.extend(_validate_enum_value(path, index, value, _BINNING_METHODS))
        elif key == "sensor.binning_factor":
            issues.extend(_validate_binning_factor(path, index, value))
        elif key == "sensor.ocl_vignetting":
            issues.extend(_validate_enum_value(path, index, value, _OCL_VIGNETTING_MODES))
        elif key == "sensor.ocl_group_shape":
            issues.extend(_validate_ocl_group_shape(path, index, value))
        elif key in fraction_axes:
            issues.extend(_validate_fraction_value(path, index, value))
        elif key == "sensor.noise_flag":
            issues.extend(
                _validate_integer_value(
                    path,
                    index,
                    value,
                    kind="invalid_integer_value",
                )
            )
        elif key == "fdtd.mode":
            issues.extend(_validate_mode_tokens(path, index, value, {"qe", "field", "crosstalk"}))
        elif key == "tcad.collection_mode":
            issues.extend(_validate_mode_tokens(path, index, value, {"collection"}))
        elif key == "ip.demosaic_method":
            issues.extend(
                _validate_enum_value(
                    path,
                    index,
                    value,
                    {"bilinear", "nearestneighbor", "nearest neighbor", "laplacian"},
                )
            )
        elif key in finite_scalar_axes:
            issues.extend(
                _validate_numeric_value(
                    path,
                    index,
                    value,
                    kind="invalid_numeric_value",
                )
            )
    return issues, warnings


_BINNING_METHODS = {
    "off",
    "none",
    "disabled",
    "kodak2008",
    "addadjacentblocks",
    "averageadjacentdigitalblocks",
}

_OCL_VIGNETTING_MODES = {
    "0",
    "1",
    "2",
    "3",
    "off",
    "skip",
    "bare",
    "nomicrolens",
    "no microlens",
    "centered",
    "optimal",
    "optimized",
}

_CFA_PRESETS = {
    "bayer",
    "bayer_rgb",
    "bayer_rggb",
    "rggb",
    "bayer_grbg",
    "grbg",
    "bayer_bggr",
    "bggr",
    "bayer_gbrg",
    "gbrg",
    "quad_bayer",
    "quad_bayer_rgb",
    "quad_bayer_rggb",
    "quad_bayer_bggr",
    "quad_bayer_grbg",
    "quad_bayer_gbrg",
}

_OCL_GROUP_SHAPES = {"1x1", "2x2"}


def _canonical_parameter_path(path: str) -> str:
    value = str(path).strip()
    if "." not in value:
        return value.lower().replace(" ", "_")
    prefix, suffix = value.split(".", 1)
    return f"{prefix.lower()}.{suffix.strip().lower().replace(' ', '_')}"


def _validate_numeric_value(
    path: str,
    index: int,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    kind: str,
) -> list[dict[str, Any]]:
    number = _scalar_float(value)
    if number is None:
        return [
            _value_finding(
                path,
                index,
                kind,
                f"{path!r} values must be finite numeric scalars.",
            )
        ]
    if positive and number <= 0.0:
        return [
            _value_finding(path, index, kind, f"{path!r} values must be > 0.")
        ]
    if nonnegative and number < 0.0:
        return [
            _value_finding(path, index, kind, f"{path!r} values must be >= 0.")
        ]
    return []


def _validate_integer_value(
    path: str,
    index: int,
    value: Any,
    *,
    nonnegative: bool = False,
    kind: str,
) -> list[dict[str, Any]]:
    number = _scalar_float(value)
    if number is None or not float(number).is_integer():
        return [
            _value_finding(path, index, kind, f"{path!r} values must be integer scalars.")
        ]
    if nonnegative and number < 0:
        return [
            _value_finding(path, index, kind, f"{path!r} values must be >= 0.")
        ]
    return []


def _validate_pixel_size(path: str, index: int, value: Any) -> list[dict[str, Any]]:
    array = _numeric_array(value)
    if array is None or array.size not in {1, 2} or not np.all(array > 0.0):
        return [
            _value_finding(
                path,
                index,
                "invalid_pixel_size",
                f"{path!r} values must be positive scalar or two-element pixel sizes in meters.",
            )
        ]
    return []


def _validate_fill_factor(path: str, index: int, value: Any) -> list[dict[str, Any]]:
    number = _scalar_float(value)
    if number is None or not (0.0 < number <= 1.0):
        return [
            _value_finding(
                path,
                index,
                "invalid_fill_factor",
                f"{path!r} values must be in the interval (0, 1].",
            )
        ]
    return []


def _validate_fraction_value(path: str, index: int, value: Any) -> list[dict[str, Any]]:
    number = _scalar_float(value)
    if number is None or not (0.0 <= number <= 1.0):
        return [
            _value_finding(
                path,
                index,
                "invalid_fraction_value",
                f"{path!r} values must be in the interval [0, 1].",
            )
        ]
    return []


def _validate_n_samples_per_pixel(
    path: str, index: int, value: Any
) -> list[dict[str, Any]]:
    issues = _validate_integer_value(
        path,
        index,
        value,
        nonnegative=True,
        kind="invalid_samples_per_pixel",
    )
    if issues:
        return issues
    number = int(_scalar_float(value) or 0)
    if number <= 0 or number % 2 == 0:
        return [
            _value_finding(
                path,
                index,
                "unsupported_samples_per_pixel",
                "sensor.n_samples_per_pixel must be a positive odd integer.",
            )
        ]
    return []


def _validate_binning_factor(path: str, index: int, value: Any) -> list[dict[str, Any]]:
    issues = _validate_integer_value(
        path,
        index,
        value,
        nonnegative=True,
        kind="invalid_binning_factor",
    )
    if issues:
        return issues
    number = int(_scalar_float(value) or 0)
    if number not in {1, 2}:
        return [
            _value_finding(
                path,
                index,
                "unsupported_binning_factor",
                "sensor.binning_factor currently supports 1/off or the legacy 2x proxy.",
            )
        ]
    return []


def _validate_ocl_group_shape(path: str, index: int, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "")
        if normalized in _OCL_GROUP_SHAPES or normalized in {"off", "none", "disabled"}:
            return []
        return [
            _value_finding(
                path,
                index,
                "unsupported_ocl_group_shape",
                "sensor.ocl_group_shape currently supports '1x1' or '2x2'.",
            )
        ]
    array = _numeric_array(value)
    if (
        array is None
        or array.size not in {1, 2}
        or not np.all(np.isclose(array.reshape(-1), np.rint(array.reshape(-1))))
    ):
        return [
            _value_finding(
                path,
                index,
                "invalid_ocl_group_shape",
                (
                    "sensor.ocl_group_shape values must be '1x1', '2x2', "
                    "or positive integer dimensions."
                ),
            )
        ]
    dims = np.rint(array.reshape(-1)).astype(int)
    if dims.size == 1:
        dims = np.repeat(dims, 2)
    if np.any(dims <= 0) or tuple(dims[:2]) not in {(1, 1), (2, 2)}:
        return [
            _value_finding(
                path,
                index,
                "unsupported_ocl_group_shape",
                "sensor.ocl_group_shape currently supports '1x1' or '2x2'.",
            )
        ]
    return []


def _validate_cfa_pattern(
    path: str, index: int, value: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    array = _numeric_array(value)
    if (
        array is None
        or array.ndim != 2
        or array.size == 0
        or not np.all(np.isclose(array, np.rint(array)))
        or np.any(array < 1)
    ):
        return [
            _value_finding(
                path,
                index,
                "invalid_cfa_pattern",
                f"{path!r} values must be a non-empty 2-D positive integer matrix.",
            )
        ], []
    warnings = []
    if int(np.max(array)) > 3:
        warnings.append(
            _value_finding(
                path,
                index,
                "cfa_filter_dependency",
                "CFA pattern references filter index > 3; attach matching filter spectra/names.",
            )
        )
    return [], warnings


def _validate_enum_value(
    path: str, index: int, value: Any, allowed: set[str]
) -> list[dict[str, Any]]:
    key = str(value).strip().lower().replace(" ", "").replace("_", "")
    allowed_keys = {item.replace(" ", "").replace("_", "") for item in allowed}
    if key not in allowed_keys:
        return [
            _value_finding(
                path,
                index,
                "unsupported_enum_value",
                f"{path!r} value {value!r} is not supported.",
            )
        ]
    return []


def _validate_mode_tokens(
    path: str, index: int, value: Any, allowed_tokens: set[str]
) -> list[dict[str, Any]]:
    raw = str(value).strip().lower()
    if raw in {"", "off", "none", "disabled", "false", "0", "all", "*"}:
        return []
    tokens = {part.strip() for part in raw.replace(",", "+").split("+") if part.strip()}
    if not tokens <= allowed_tokens:
        return [
            _value_finding(
                path,
                index,
                "unsupported_mode_tokens",
                f"{path!r} contains unsupported mode token(s): {sorted(tokens - allowed_tokens)}.",
            )
        ]
    return []


def _scalar_float(value: Any) -> float | None:
    array = _numeric_array(value)
    if array is None or array.size != 1:
        return None
    return float(array.reshape(-1)[0])


def _numeric_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _value_finding(path: str, index: int, kind: str, message: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "value_index": int(index),
        "kind": str(kind),
        "message": str(message),
    }


def _parameter_axis_classification(path: str) -> dict[str, Any]:
    if path in _PARAMETER_AXIS_CATALOG:
        return {
            "status": "registered",
            "assignment": _assignment_for_path(path),
            "readiness_tier": _PARAMETER_AXIS_CATALOG[path]["readiness_tier"],
        }
    if "." not in path:
        return {
            "status": "custom_passthrough",
            "assignment": "scenario.parameters -> camera_set(...)",
            "readiness_tier": "available",
        }
    prefix, remainder = path.split(".", 1)
    if prefix == "sensor" and remainder in _SENSOR_OPTIMIZATION_PARAMETERS:
        return {
            "status": "assignable",
            "assignment": "scenario.sensor -> sensor_set(...)",
            "readiness_tier": "validated",
        }
    if prefix == "hw_isp":
        return {
            "status": "assignable",
            "assignment": "scenario.hw_isp control dictionary",
            "readiness_tier": "proxy",
        }
    if prefix == "fdtd":
        return {
            "status": "assignable",
            "assignment": "scenario.fdtd attachment options",
            "readiness_tier": "proxy",
        }
    if prefix == "tcad":
        return {
            "status": "assignable",
            "assignment": "scenario.tcad attachment options",
            "readiness_tier": "calibration_required",
        }
    return {
        "status": "custom_passthrough",
        "assignment": "scenario.parameters/camera_parameters -> camera_set(...)",
        "readiness_tier": "available",
    }


def _assignment_for_path(path: str) -> str:
    prefix = path.split(".", 1)[0]
    if prefix == "sensor":
        return "scenario.sensor -> sensor_set(...)"
    if prefix == "hw_isp":
        return "scenario.hw_isp control dictionary"
    if prefix in {"fdtd", "tcad"}:
        return f"scenario.{prefix} attachment options"
    return "scenario.parameters/camera_parameters -> camera_set(...)"


def _has_nested_value(payload: Mapping[str, Any] | None, bucket: str, name: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    value = payload.get(bucket)
    return isinstance(value, Mapping) and value.get(name) is not None


def _axis_values(raw: Iterable[Any] | Mapping[str, Any]) -> list[Any]:
    if isinstance(raw, Mapping):
        if "values" in raw:
            return list(raw["values"])
        if "linspace" in raw:
            start, stop, count = raw["linspace"]
            return np.linspace(float(start), float(stop), int(count)).tolist()
        if {"start", "stop", "num"} <= set(raw):
            return np.linspace(float(raw["start"]), float(raw["stop"]), int(raw["num"])).tolist()
        if {"start", "stop", "step"} <= set(raw):
            start = float(raw["start"])
            stop = float(raw["stop"])
            step = float(raw["step"])
            if step == 0:
                raise ValueError("Parameter axis step must be non-zero.")
            count = int(math.floor((stop - start) / step)) + 1
            return [start + index * step for index in range(max(count, 0))]
        raise ValueError("Parameter axis mapping must define values, linspace, or start/stop.")
    return list(raw)


def _normalize_objective(
    objective: ObjectiveSpec | ObjectiveCallable | None,
) -> list[dict[str, Any]]:
    if objective is None:
        return [{"metric": "metrics.color.rgb_mean", "direction": "maximize", "weight": 1.0}]
    if callable(objective):
        return [{"metric": "<callable>", "direction": "maximize", "weight": 1.0}]
    if isinstance(objective, str):
        return [{"metric": objective, "direction": "maximize", "weight": 1.0}]
    if isinstance(objective, Mapping):
        return [dict(objective)]
    return [dict(item) for item in objective]


def _score_report(
    report: Mapping[str, Any],
    objective_specs: list[dict[str, Any]],
    objective: ObjectiveSpec | ObjectiveCallable | None,
) -> tuple[dict[str, Any], dict[str, float], float]:
    if callable(objective):
        score = float(objective(report))
        return {"<callable>": score}, {"<callable>": score}, score
    values = {}
    utilities = {}
    score = 0.0
    for spec in objective_specs:
        metric = str(spec.get("metric", spec.get("path", "")))
        value = _numeric_path(report, metric)
        weight = float(spec.get("weight", 1.0))
        direction = str(spec.get("direction", "maximize")).lower()
        target = spec.get("target")
        if target is not None:
            component = -abs(value - float(target))
        elif direction in {"min", "minimize", "lower"}:
            component = -value
        elif direction in {"max", "maximize", "higher"}:
            component = value
        else:
            raise ValueError(f"Unsupported objective direction: {direction}")
        values[metric] = value
        utilities[metric] = float(component)
        score += weight * component
    return values, utilities, float(score)


def _evaluate_constraint(report: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    metric = str(spec.get("metric", spec.get("path", "")))
    op = str(spec.get("op", spec.get("operator", "<=")))
    limit = float(spec.get("value", spec.get("limit")))
    value = _numeric_path(report, metric)
    if op in {"<", "lt"}:
        passed = value < limit
    elif op in {"<=", "le"}:
        passed = value <= limit
    elif op in {">", "gt"}:
        passed = value > limit
    elif op in {">=", "ge"}:
        passed = value >= limit
    elif op in {"==", "eq"}:
        passed = math.isclose(value, limit)
    else:
        raise ValueError(f"Unsupported constraint operator: {op}")
    return {"metric": metric, "op": op, "limit": limit, "value": value, "pass": bool(passed)}


def _numeric_path(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            raise KeyError(f"Metric path {path!r} is not present in the FACA report.")
    if value is None:
        raise ValueError(f"Metric path {path!r} resolved to None.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric path {path!r} resolved to a non-finite value.")
    return number


def _assign_parameter(scenario: dict[str, Any], key: str, value: Any) -> None:
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


def _strip_case_report(case: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if case is None:
        return None
    return {
        "case_index": case.get("case_index"),
        "seed": case.get("seed"),
        "parameters": _jsonable(case.get("parameters", {})),
        "score": case.get("score"),
        "feasible": case.get("feasible"),
        "objective_values": _jsonable(case.get("objective_values", {})),
        "objective_utilities": _jsonable(case.get("objective_utilities", {})),
        "constraint_results": _jsonable(case.get("constraint_results", [])),
        "scenario": _jsonable(case.get("scenario", {})),
    }


def _pareto_cases(cases: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    feasible = [case for case in cases if case.get("feasible", True)]
    frontier = []
    for candidate in feasible:
        if not _objective_vector(candidate):
            continue
        dominated = False
        for other in feasible:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda item: (float(item.get("score", -math.inf)), -int(item.get("case_index", 0))),
        reverse=True,
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_vector = _objective_vector(left)
    right_vector = _objective_vector(right)
    if set(left_vector) != set(right_vector) or not left_vector:
        return False
    at_least_equal = all(left_vector[key] >= right_vector[key] for key in left_vector)
    strictly_better = any(left_vector[key] > right_vector[key] for key in left_vector)
    return bool(at_least_equal and strictly_better)


def _objective_vector(case: Mapping[str, Any]) -> dict[str, float]:
    values = dict(case.get("objective_utilities", {}))
    vector: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            vector[str(key)] = number
    return vector


def _objective_description(
    objective: ObjectiveSpec | ObjectiveCallable | None, specs: list[dict[str, Any]]
) -> Any:
    if callable(objective):
        return {"callable": getattr(objective, "__name__", "<callable>")}
    return specs


def _deep_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[str(key)] = _deep_dict(item) if isinstance(item, Mapping) else item
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


cameraE2EOptimizeParameters = camerae2e_optimize_parameters  # noqa: N816
cameraE2EOptimizeCameraParameters = camerae2e_optimize_camera_parameters  # noqa: N816
cameraE2EParetoFront = camerae2e_pareto_front  # noqa: N816
cameraE2EOptimizationReport = camerae2e_optimization_report  # noqa: N816
cameraE2EOptimizationEscalationPlan = camerae2e_optimization_escalation_plan  # noqa: N816
cameraE2EParameterSpaceCatalog = camerae2e_parameter_space_catalog  # noqa: N816
cameraE2EOptimizationConfigCatalog = camerae2e_optimization_config_catalog  # noqa: N816
cameraE2EParameterSpaceValidate = camerae2e_parameter_space_validate  # noqa: N816
