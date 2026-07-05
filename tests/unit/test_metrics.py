from __future__ import annotations

import numpy as np
import pyisetcam.metrics as metrics_module

from pyisetcam import (
    DEFAULT_WAVE,
    blackbody,
    cct_from_uv,
    comparison_metrics,
    correlated_color_temperature,
    deltaE2000,
    deltaE94,
    deltaEuv,
    delta_e_ab,
    metrics_spd,
    mired_difference,
    peak_signal_to_noise_ratio,
    sc_compute_difference,
    srgb_to_color_temp,
    xyz_from_energy,
    xyz_to_lab,
    xyz_to_luv,
    xyz_to_uv,
)


def test_metrics_module_human_cones_matlab_alias() -> None:
    assert metrics_module.humanCones is metrics_module.human_cones


def test_metrics_module_cct_matlab_alias() -> None:
    assert metrics_module.cct is metrics_module.cct_from_uv


def test_metrics_module_adjacent_matlab_aliases() -> None:
    assert metrics_module.cctFromUV is metrics_module.cct_from_uv
    assert metrics_module.chromaticityXY is metrics_module.chromaticity_xy
    assert metrics_module.comparisonMetrics is metrics_module.comparison_metrics
    assert metrics_module.correlatedColorTemperature is metrics_module.correlated_color_temperature
    assert metrics_module.exampleSPDPair is metrics_module.example_spd_pair
    assert metrics_module.meanAbsoluteError is metrics_module.mean_absolute_error
    assert metrics_module.meanRelativeError is metrics_module.mean_relative_error
    assert metrics_module.miredDifference is metrics_module.mired_difference
    assert metrics_module.peakSignalToNoiseRatio is metrics_module.peak_signal_to_noise_ratio
    assert metrics_module.rootMeanSquaredError is metrics_module.root_mean_squared_error
    assert metrics_module.spectralAngle is metrics_module.spectral_angle
    assert metrics_module.xyz2uv is metrics_module.xyz_to_uv


def test_xyz_to_lab_maps_white_point_to_neutral() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    lab = xyz_to_lab(white, white)

    assert np.allclose(lab, np.array([100.0, 0.0, 0.0]), atol=1e-6)


def test_xyz_to_luv_maps_white_point_to_neutral() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    luv = xyz_to_luv(white, white)

    assert np.allclose(luv, np.array([100.0, 0.0, 0.0]), atol=1e-6)


def test_xyz_to_luv_matches_cie_1976_reference_values() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    xyz = np.array([20.0, 30.0, 15.0], dtype=float)
    luv = xyz_to_luv(xyz, white)

    assert np.allclose(luv, np.array([61.65422221, -34.06397300, 44.83261015]), atol=1e-8)


def test_xyz_to_uv_matches_cie_1960_reference_values() -> None:
    xyz = np.array([20.0, 30.0, 15.0], dtype=float)
    uv = xyz_to_uv(xyz)

    assert np.allclose(uv, np.array([0.15533981, 0.34951456]), atol=1e-8)


def test_cct_from_uv_matches_upstream_lookup_table_value() -> None:
    uv = np.array([0.20029948, 0.31055768], dtype=float)
    cct = cct_from_uv(uv)

    assert np.isclose(cct, 6500.0, atol=200.0)


def test_delta_e_ab_is_zero_for_identical_xyz() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    xyz = np.array([20.0, 30.0, 15.0], dtype=float)

    assert np.isclose(delta_e_ab(xyz, xyz, white), 0.0)


def test_delta_e_ab_supports_component_modes_and_all_payload() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    xyz1 = np.array([[20.0, 30.0, 15.0], [22.0, 31.0, 18.0]], dtype=float)
    xyz2 = np.array([[19.0, 29.0, 14.0], [21.0, 32.0, 17.0]], dtype=float)

    lab1 = xyz_to_lab(xyz1, white)
    lab2 = xyz_to_lab(xyz2, white)
    delta_e_2000_value, components = deltaE2000(lab1, lab2)
    all_delta_e, all_components = delta_e_ab(xyz1, xyz2, white, "all")

    np.testing.assert_allclose(delta_e_ab(xyz1, xyz2, white, "luminance"), components["dL"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(delta_e_ab(xyz1, xyz2, white, "chrominance"), components["dC"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(delta_e_ab(xyz1, xyz2, white, "hue"), components["dH"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(all_delta_e, delta_e_2000_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(all_components["dL"], components["dL"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(all_components["dC"], components["dC"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(all_components["dH"], components["dH"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(all_components["RT"], components["RT"], rtol=1e-10, atol=1e-10)


def test_delta_e_ab_accepts_single_and_pair_white_points() -> None:
    xyz1 = np.array([[[20.0, 30.0, 15.0], [22.0, 31.0, 18.0]]], dtype=float)
    xyz2 = np.array([[[19.0, 29.0, 14.0], [21.0, 32.0, 17.0]]], dtype=float)
    white1 = np.array([95.047, 100.0, 108.883], dtype=float)
    white2 = np.array([96.0, 101.0, 109.0], dtype=float)

    shared = delta_e_ab(xyz1, xyz2, white1, "2000")
    paired = delta_e_ab(xyz1, xyz2, (white1, white2), "2000")

    expected_shared = deltaE2000(xyz_to_lab(xyz1, white1), xyz_to_lab(xyz2, white1))[0]
    expected_paired = deltaE2000(xyz_to_lab(xyz1, white1), xyz_to_lab(xyz2, white2))[0]

    np.testing.assert_allclose(shared, expected_shared, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(paired, expected_paired, rtol=1e-10, atol=1e-10)


def test_delta_e_2000_matches_skimage_and_reports_components() -> None:
    lab_std = np.array([[50.0, 2.6772, -79.7751], [50.0, 0.0, 0.0]], dtype=float)
    lab_sample = np.array([[50.0, 0.0, -82.7485], [50.0, -1.0, 2.0]], dtype=float)

    delta_e, components = deltaE2000(lab_std, lab_sample)

    from skimage.color import deltaE_ciede2000

    expected = deltaE_ciede2000(lab_std, lab_sample)
    np.testing.assert_allclose(delta_e, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        delta_e**2,
        components["dL"] ** 2 + components["dC"] ** 2 + components["dH"] ** 2 + components["RT"],
        rtol=1e-10,
        atol=1e-10,
    )


def test_delta_e_94_matches_skimage_and_reports_components() -> None:
    lab1 = np.array([[50.0, 20.0, 30.0], [65.0, -5.0, 15.0]], dtype=float)
    lab2 = np.array([[48.0, 18.0, 28.0], [60.0, -2.0, 10.0]], dtype=float)

    delta_e, components = deltaE94(lab1, lab2)

    from skimage.color import deltaE_ciede94

    expected = deltaE_ciede94(lab1, lab2)
    np.testing.assert_allclose(delta_e, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        delta_e**2,
        components["dL"] ** 2 + components["dC"] ** 2 + components["dH"] ** 2,
        rtol=1e-10,
        atol=1e-10,
    )


def test_delta_e_uv_accepts_single_and_pair_white_points() -> None:
    xyz1 = np.array([[[20.0, 30.0, 15.0], [22.0, 31.0, 18.0]]], dtype=float)
    xyz2 = np.array([[[19.0, 29.0, 14.0], [21.0, 32.0, 17.0]]], dtype=float)
    white1 = np.array([95.047, 100.0, 108.883], dtype=float)
    white2 = np.array([96.0, 101.0, 109.0], dtype=float)

    shared = deltaEuv(xyz1, xyz2, white1)
    paired = deltaEuv(xyz1, xyz2, (white1, white2))

    expected_shared = np.sqrt(np.sum((xyz_to_luv(xyz1, white1) - xyz_to_luv(xyz2, white1)) ** 2, axis=-1))
    expected_paired = np.sqrt(np.sum((xyz_to_luv(xyz1, white1) - xyz_to_luv(xyz2, white2)) ** 2, axis=-1))

    np.testing.assert_allclose(shared, expected_shared, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(paired, expected_paired, rtol=1e-10, atol=1e-10)


def test_sc_compute_difference_supports_component_modes_and_all_payload() -> None:
    white = np.array([95.047, 100.0, 108.883], dtype=float)
    xyz1 = np.array([[[20.0, 30.0, 15.0], [22.0, 31.0, 18.0]]], dtype=float)
    xyz2 = np.array([[[19.0, 29.0, 14.0], [21.0, 32.0, 17.0]]], dtype=float)

    lab1 = xyz_to_lab(xyz1, white)
    lab2 = xyz_to_lab(xyz2, white)
    delta_e_2000_value, components = deltaE2000(lab1, lab2)
    payload = sc_compute_difference(xyz1, xyz2, white, "all")

    np.testing.assert_allclose(sc_compute_difference(xyz1, xyz2, white, "luminance"), components["dL"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sc_compute_difference(xyz1, xyz2, white, "chrominance"), components["dC"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sc_compute_difference(xyz1, xyz2, white, "hue"), components["dH"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(payload["dE"], delta_e_2000_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(payload["components"]["dL"], components["dL"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(payload["components"]["dC"], components["dC"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(payload["components"]["dH"], components["dH"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(payload["components"]["RT"], components["RT"], rtol=1e-10, atol=1e-10)


def test_metrics_spd_angle_matches_direct_vector_angle() -> None:
    wave = np.array([500.0, 510.0, 520.0], dtype=float)
    spd1 = np.array([1.0, 0.0, 0.0], dtype=float)
    spd2 = np.array([0.0, 1.0, 0.0], dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="angle", wave=wave, return_params=True)

    assert np.isclose(value, 90.0)
    assert params == {}


def test_metrics_spd_cielab_matches_direct_xyz_and_lab_path() -> None:
    spd1 = np.asarray(blackbody(DEFAULT_WAVE, 6500.0, kind="energy"), dtype=float)
    spd2 = np.asarray(blackbody(DEFAULT_WAVE, 5000.0, kind="energy"), dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="cielab", return_params=True)

    xyz1 = xyz_from_energy(spd1 * (100.0 / xyz_from_energy(spd1, DEFAULT_WAVE)[1]), DEFAULT_WAVE)
    xyz2 = xyz_from_energy(spd2 * (100.0 / xyz_from_energy(spd2, DEFAULT_WAVE)[1]), DEFAULT_WAVE)
    white = xyz1 * (100.0 / xyz1[1])
    expected = np.linalg.norm(xyz_to_lab(xyz1, white) - xyz_to_lab(xyz2, white))

    assert np.isclose(value, expected, rtol=1e-8, atol=1e-10)
    assert params["lab1"].shape == (3,)
    assert params["lab2"].shape == (3,)


def test_metrics_spd_accepts_matlab_style_key_value_options() -> None:
    spd1 = np.asarray(blackbody(DEFAULT_WAVE, 6500.0, kind="energy"), dtype=float)
    spd2 = np.asarray(blackbody(DEFAULT_WAVE, 5000.0, kind="energy"), dtype=float)
    white = np.array([95.047, 100.0, 108.883], dtype=float)

    keyword_value, keyword_params = metrics_spd(
        spd1,
        spd2,
        metric="cielab",
        wave=DEFAULT_WAVE,
        white_point=white,
        return_params=True,
    )
    legacy_value, legacy_params = metrics_spd(
        spd1,
        spd2,
        "metric",
        "cielab",
        "wave",
        DEFAULT_WAVE,
        "white point",
        white,
        return_params=True,
    )

    assert np.isclose(legacy_value, keyword_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(legacy_params["xyz1"], keyword_params["xyz1"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(legacy_params["xyz2"], keyword_params["xyz2"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(legacy_params["lab1"], keyword_params["lab1"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(legacy_params["lab2"], keyword_params["lab2"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(legacy_params["white_point"], keyword_params["white_point"], rtol=1e-10, atol=1e-10)


def test_metrics_spd_mired_returns_estimated_ccts() -> None:
    spd1 = np.asarray(blackbody(DEFAULT_WAVE, 6500.0, kind="energy"), dtype=float)
    spd2 = np.asarray(blackbody(DEFAULT_WAVE, 5000.0, kind="energy"), dtype=float)

    value, params = metrics_spd(spd1, spd2, metric="mired", return_params=True)

    assert value > 0.0
    assert params["cct_k"].shape == (2,)
    assert np.isclose(value, mired_difference(params["cct_k"][0], params["cct_k"][1]))


def test_srgb_to_color_temp_accepts_matlab_style_method_key_value() -> None:
    rgb = np.array(
        [
            [[0.85, 0.78, 0.70], [0.80, 0.75, 0.68]],
            [[0.72, 0.70, 0.66], [0.68, 0.67, 0.65]],
        ],
        dtype=float,
    )

    positional = srgb_to_color_temp(rgb, "gray")
    key_value = srgb_to_color_temp(rgb, "method", "gray")

    assert positional == key_value


def test_correlated_color_temperature_tracks_d65_white_point() -> None:
    d65_white = np.array([95.047, 100.0, 108.883], dtype=float)
    cct = correlated_color_temperature(d65_white)

    assert 6000.0 <= cct <= 7000.0


def test_comparison_metrics_report_standard_error_values() -> None:
    reference = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=float)
    actual = np.array([[0.6, 0.5], [0.9, 0.5]], dtype=float)

    metrics = comparison_metrics(reference, actual, data_range=1.0)

    assert np.isclose(metrics["mae"], 0.05)
    assert np.isclose(metrics["rmse"], np.sqrt(0.005))
    assert np.isclose(metrics["mean_rel"], 0.075)
    assert np.isclose(metrics["max_abs"], 0.1)
    assert np.isclose(metrics["psnr"], peak_signal_to_noise_ratio(reference, actual, data_range=1.0))
