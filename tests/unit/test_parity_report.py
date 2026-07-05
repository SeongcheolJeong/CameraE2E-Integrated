from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pyisetcam.parity import run_python_case_with_context


def _load_parity_report_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "parity_report.py"
    spec = importlib.util.spec_from_file_location("parity_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_array_metrics_reports_border_and_max_index() -> None:
    parity_report = _load_parity_report_module()

    reference = np.ones((10, 10), dtype=float)
    actual = reference.copy()
    actual[0, 0] = 1.5
    actual[5, 5] = 1.1

    metrics = parity_report._array_metrics(reference, actual)

    assert metrics["max_index"] == [0, 0]
    assert np.isclose(metrics["max_index_ref"], 1.0)
    assert np.isclose(metrics["max_index_actual"], 1.5)
    assert np.isclose(metrics["edge_mean_rel"], 0.5 / 36.0)
    assert np.isclose(metrics["interior_mean_rel"], 0.0015625)


def test_array_metrics_reports_2x2_phase_breakdown() -> None:
    parity_report = _load_parity_report_module()

    reference = np.ones((4, 4), dtype=float)
    actual = reference.copy()
    actual[0::2, 1::2] = 1.4
    actual[1::2, 0::2] = 1.2

    metrics = parity_report._array_metrics(reference, actual)

    assert np.isclose(metrics["phase_2x2_mean_rel"]["r0c1"], 0.4)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r1c0"], 0.2)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r0c0"], 0.0)
    assert np.isclose(metrics["phase_2x2_mean_rel"]["r1c1"], 0.0)


def test_compare_supports_mean_rel_field_override() -> None:
    parity_report = _load_parity_report_module()

    reference = {"photons": np.ones((2, 2), dtype=float)}
    actual = {"photons": np.full((2, 2), 1.02, dtype=float)}

    comparison = parity_report._compare(
        reference,
        actual,
        rtol=1e-4,
        atol=1e-6,
        field_rules={"photons": {"mode": "mean_rel", "max_mean_rel": 0.03}},
    )

    photons = comparison["fields"]["photons"]
    assert photons["comparison_mode"] == "mean_rel"
    assert photons["pass"]
    assert np.isclose(photons["mean_rel"], 0.02)


def test_compare_supports_scale_invariant_field_override() -> None:
    parity_report = _load_parity_report_module()

    reference = {"photons": np.ones((2, 2), dtype=float)}
    actual = {"photons": np.full((2, 2), 0.1, dtype=float)}

    comparison = parity_report._compare(
        reference,
        actual,
        rtol=1e-4,
        atol=1e-6,
        field_rules={"photons": {"mode": "scale_invariant", "max_mean_rel": 1e-8}},
    )

    photons = comparison["fields"]["photons"]
    assert photons["comparison_mode"] == "scale_invariant"
    assert photons["pass"]
    assert np.isclose(photons["scale"], 10.0)
    assert np.isclose(photons["mean_rel"], 0.0)
    assert np.isclose(photons["unscaled_mean_rel"], 0.9)


def test_compare_treats_zero_dim_arrays_as_scalars() -> None:
    parity_report = _load_parity_report_module()

    comparison = parity_report._compare(np.array(1.5), np.array(1.5), rtol=1e-5, atol=1e-8)

    assert comparison["pass"]
    assert np.isclose(comparison["expected"], 1.5)
    assert np.isclose(comparison["actual"], 1.5)
    assert np.isclose(comparison["abs_diff"], 0.0)


def test_case_diagnostics_recompute_sensor_from_reference_oi(asset_store) -> None:
    parity_report = _load_parity_report_module()

    reference = parity_report._load_reference("sensor_bayer_noiseless")
    case = run_python_case_with_context("sensor_bayer_noiseless", asset_store=asset_store)
    diagnostics = parity_report._case_diagnostics(
        "sensor_bayer_noiseless",
        reference=reference,
        context=case.context,
        rtol=1e-5,
        atol=1e-8,
    )

    assert diagnostics["context"]["sensor_size"] == [72, 88]
    assert diagnostics["reference_recompute"]["reference_oi_case"] == "oi_diffraction_limited_default"
    assert diagnostics["reference_recompute"]["sensor_volts"]["pass"]
    assert diagnostics["reference_recompute"]["integration_time"]["pass"]


def test_case_diagnostics_show_camera_reference_recompute_match(asset_store) -> None:
    parity_report = _load_parity_report_module()

    reference = parity_report._load_reference("camera_default_pipeline")
    case = run_python_case_with_context("camera_default_pipeline", asset_store=asset_store)
    diagnostics = parity_report._case_diagnostics(
        "camera_default_pipeline",
        reference=reference,
        context=case.context,
        rtol=1e-3,
        atol=1e-3,
    )

    comparison = diagnostics["reference_recompute"]["sensor_volts_from_reference_oi"]

    assert diagnostics["context"]["oi_size"] == [80, 120, 31]
    assert comparison["pass"]
    assert comparison["mean_rel"] < 1e-6
