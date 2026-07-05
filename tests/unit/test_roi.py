from __future__ import annotations

import importlib

import numpy as np

from pyisetcam import (
    chromaticity_xy,
    ieLocs2Rect,
    ieRect2Locs,
    ieRect2Vertices,
    ieRoi2Locs,
    imageDataXYZ,
    ip_create,
    ip_get,
    ip_set,
    oi_create,
    oi_get,
    oi_set,
    scene_create,
    scene_get,
    scene_set,
    sensor_create,
    sensor_get,
    sensorROI,
    sensor_set,
    vcGetROIData,
    vcRect2Locs,
    xyz_from_energy,
)
from pyisetcam.display import display_get

roi_module = importlib.import_module("pyisetcam.roi")


def test_roi_module_helper_matlab_aliases() -> None:
    assert roi_module.sceneGet is roi_module.scene_get
    assert roi_module.oiGet is roi_module.oi_get
    assert roi_module.sensorGet is roi_module.sensor_get
    assert roi_module.ipGet is roi_module.ip_get
    assert roi_module.Quanta2Energy is roi_module.quanta_to_energy
    assert roi_module.imageDataXYZ is roi_module.image_data_xyz


def test_roi_rect_location_helpers_round_trip() -> None:
    rect = np.array([4, 2, 5, 7], dtype=int)

    roi_locs = ieRect2Locs(rect)

    assert np.array_equal(roi_locs[0], np.array([2, 4], dtype=int))
    assert np.array_equal(roi_locs[-1], np.array([9, 9], dtype=int))
    assert np.array_equal(vcRect2Locs(rect), roi_locs)
    assert np.array_equal(ieRoi2Locs(rect), roi_locs)
    assert np.array_equal(ieLocs2Rect(roi_locs), rect)


def test_ie_rect2_vertices_supports_closed_polygon() -> None:
    xv, yv = ieRect2Vertices([3, 5, 2, 4], close_flag=True)

    assert np.array_equal(xv, np.array([3, 3, 5, 5, 3], dtype=int))
    assert np.array_equal(yv, np.array([5, 9, 9, 5, 5], dtype=int))


def test_sensor_roi_returns_center_roi_locs(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "size", np.array([8, 12], dtype=int))

    roi_locs = sensorROI(sensor)

    expected = ieRect2Locs(np.array([3, 2, 6, 4], dtype=int))
    assert np.array_equal(roi_locs, expected)


def test_vc_get_roi_data_scene_rect_returns_xw_illuminant_energy(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)

    roi_data = vcGetROIData(scene, [2, 2, 1, 1], "illuminant energy")

    expected = np.tile(np.asarray(scene_get(scene, "illuminant energy"), dtype=float), (4, 1))
    assert roi_data.shape == expected.shape
    assert np.allclose(roi_data, expected)


def test_vc_get_roi_data_oi_clips_out_of_bounds(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)

    roi_data = vcGetROIData(oi, np.array([[0, 0], [9, 9]]), "photons")

    assert roi_data.shape == (2, 2)
    assert np.allclose(roi_data[0], photons[0, 0, :])
    assert np.allclose(roi_data[1], photons[-1, -1, :])


def test_vc_get_roi_data_sensor_mosaic_returns_nan_planes(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    volts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sensor = sensor_set(sensor, "volts", volts)

    roi_data = vcGetROIData(sensor, np.array([[1, 1], [1, 2], [2, 1], [2, 2]]), "volts")

    expected = np.array(
        [
            [np.nan, 1.0, np.nan],
            [2.0, np.nan, np.nan],
            [np.nan, np.nan, 3.0],
            [np.nan, 4.0, np.nan],
        ],
        dtype=float,
    )
    assert roi_data.shape == expected.shape
    assert np.allclose(roi_data, expected, equal_nan=True)


def test_vc_get_roi_data_ip_result_and_input(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip = ip_set(ip, "input", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    ip = ip_set(
        ip,
        "result",
        np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ],
            dtype=float,
        ),
    )

    result_roi = vcGetROIData(ip, np.array([[2, 2]]))
    input_roi = vcGetROIData(ip, np.array([[2, 1]]), "input")

    assert result_roi.shape == (1, 3)
    assert np.allclose(result_roi[0], np.array([1.0, 1.1, 1.2], dtype=float))
    assert input_roi.shape == (1, 1)
    assert np.allclose(input_roi[:, 0], np.array([3.0], dtype=float))


def test_sensor_get_roi_queries_and_means(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.5
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    sensor = sensor_set(sensor, "dv", np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float))
    sensor = sensor_set(sensor, "roirect", np.array([1, 1, 1, 1], dtype=int))

    roi_locs = sensor_get(sensor, "roilocs")
    roi_volts = sensor_get(sensor, "roivolts")
    roi_electrons = sensor_get(sensor, "roielectrons")
    roi_dv = sensor_get(sensor, "roidv")
    roi_volts_mean = sensor_get(sensor, "roivoltsmean")
    roi_electrons_mean = sensor_get(sensor, "roielectronsmean")

    assert np.array_equal(roi_locs, np.array([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=int))
    assert np.array_equal(sensor_get(sensor, "roirect"), np.array([1, 1, 1, 1], dtype=int))
    assert np.array_equal(sensor_get(sensor, "roi"), np.array([1, 1, 1, 1], dtype=int))
    assert np.allclose(
        roi_volts,
        np.array(
            [
                [np.nan, 1.0, np.nan],
                [2.0, np.nan, np.nan],
                [np.nan, np.nan, 3.0],
                [np.nan, 4.0, np.nan],
            ],
            dtype=float,
        ),
        equal_nan=True,
    )
    assert np.allclose(
        roi_electrons,
        np.array(
            [
                [np.nan, 2.0, np.nan],
                [6.0, np.nan, np.nan],
                [np.nan, np.nan, 10.0],
                [np.nan, 14.0, np.nan],
            ],
            dtype=float,
        ),
        equal_nan=True,
    )
    assert np.allclose(
        roi_dv,
        np.array(
            [
                [np.nan, 10.0, np.nan],
                [20.0, np.nan, np.nan],
                [np.nan, np.nan, 30.0],
                [np.nan, 40.0, np.nan],
            ],
            dtype=float,
        ),
        equal_nan=True,
    )
    assert np.allclose(roi_volts_mean, np.array([2.0, 2.5, 3.0], dtype=float))
    assert np.allclose(roi_electrons_mean, np.array([6.0, 8.0, 10.0], dtype=float))
    assert np.allclose(sensor_get(sensor, "roidata"), roi_volts, equal_nan=True)
    assert np.allclose(sensor_get(sensor, "roidatav"), roi_volts, equal_nan=True)
    assert np.allclose(sensor_get(sensor, "roidatavolts"), roi_volts, equal_nan=True)
    assert np.allclose(sensor_get(sensor, "roidatae"), roi_electrons, equal_nan=True)
    assert np.allclose(sensor_get(sensor, "roidataelectrons"), roi_electrons, equal_nan=True)
    assert np.allclose(sensor_get(sensor, "roidigitalcount"), roi_dv, equal_nan=True)


def test_sensor_get_electrons_direct_getter(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.25
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[0.5, 1.0], [1.5, 2.0]], dtype=float))

    electrons = sensor_get(sensor, "electrons")

    assert np.allclose(electrons, np.array([[0.0, 4.0], [8.0, 12.0]], dtype=float))


def test_sensor_get_line_profiles(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor.fields["pixel"]["conversion_gain_v_per_electron"] = 0.5
    sensor = sensor_set(sensor, "analog gain", 2.0)
    sensor = sensor_set(sensor, "analog offset", 1.0)
    sensor = sensor_set(sensor, "volts", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    sensor = sensor_set(sensor, "dv", np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float))

    hline_volts = sensor_get(sensor, "hlinevolts", 1)
    hline_electrons = sensor_get(sensor, "hlineelectrons", 1)
    vline_volts = sensor_get(sensor, "vlinevolts", 1)
    vline_electrons = sensor_get(sensor, "vlineelectrons", 1)
    hline_dv = sensor_get(sensor, "hlinedv", 1)
    vline_dv = sensor_get(sensor, "vlinedv", 1)
    support = sensor_get(sensor, "spatial support")

    assert set(hline_volts) == {"data", "pos", "pixPos"}
    assert np.allclose(hline_volts["data"][0], np.array([2.0], dtype=float))
    assert np.allclose(hline_volts["data"][1], np.array([1.0], dtype=float))
    assert hline_volts["data"][2].size == 0
    assert np.allclose(hline_volts["pos"][0], np.array([support["x"][1]], dtype=float))
    assert np.allclose(hline_volts["pos"][1], np.array([support["x"][0]], dtype=float))
    assert hline_volts["pos"][2].size == 0
    assert np.allclose(hline_volts["pixPos"][0], hline_volts["pos"][0])
    assert np.allclose(hline_volts["pixPos"][1], hline_volts["pos"][1])
    assert hline_volts["pixPos"][2].size == 0

    assert np.allclose(hline_electrons["data"][0], np.array([6.0], dtype=float))
    assert np.allclose(hline_electrons["data"][1], np.array([2.0], dtype=float))
    assert hline_electrons["data"][2].size == 0
    assert np.allclose(hline_electrons["pos"][0], np.array([support["x"][1]], dtype=float))
    assert np.allclose(hline_electrons["pos"][1], np.array([support["x"][0]], dtype=float))
    assert hline_electrons["pos"][2].size == 0

    assert vline_volts["data"][0].size == 0
    assert np.allclose(vline_volts["data"][1], np.array([1.0], dtype=float))
    assert np.allclose(vline_volts["data"][2], np.array([3.0], dtype=float))
    assert vline_volts["pos"][0].size == 0
    assert np.allclose(vline_volts["pos"][1], np.array([support["y"][0]], dtype=float))
    assert np.allclose(vline_volts["pos"][2], np.array([support["y"][1]], dtype=float))

    assert vline_electrons["data"][0].size == 0
    assert np.allclose(vline_electrons["data"][1], np.array([2.0], dtype=float))
    assert np.allclose(vline_electrons["data"][2], np.array([10.0], dtype=float))
    assert vline_electrons["pos"][0].size == 0
    assert np.allclose(vline_electrons["pos"][1], np.array([support["y"][0]], dtype=float))
    assert np.allclose(vline_electrons["pos"][2], np.array([support["y"][1]], dtype=float))

    assert np.allclose(hline_dv["data"][0], np.array([20.0], dtype=float))
    assert np.allclose(hline_dv["data"][1], np.array([10.0], dtype=float))
    assert hline_dv["data"][2].size == 0
    assert np.allclose(hline_dv["pos"][0], np.array([support["x"][1]], dtype=float))
    assert np.allclose(hline_dv["pos"][1], np.array([support["x"][0]], dtype=float))
    assert hline_dv["pos"][2].size == 0

    assert vline_dv["data"][0].size == 0
    assert np.allclose(vline_dv["data"][1], np.array([10.0], dtype=float))
    assert np.allclose(vline_dv["data"][2], np.array([30.0], dtype=float))


def test_sensor_get_chromaticity_and_roi_mean(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(
        sensor,
        "volts",
        np.array(
            [
                [2.0, 4.0, 2.0, 4.0],
                [6.0, 2.0, 6.0, 2.0],
                [2.0, 4.0, 2.0, 4.0],
                [6.0, 2.0, 6.0, 2.0],
            ],
            dtype=float,
        ),
    )

    chromaticity_vec = sensor_get(sensor, "chromaticity")
    chromaticity_matrix = sensor_get(sensor, "chromaticity", np.array([1, 1, 3, 3], dtype=int), "matrix")
    roi_chromaticity_mean = sensor_get(sensor, "roichromaticitymean", np.array([1, 1, 3, 3], dtype=int))

    expected_xy = np.array([4.0 / 12.0, 2.0 / 12.0], dtype=float)
    assert chromaticity_vec.shape == (16, 2)
    assert np.allclose(chromaticity_vec, np.tile(expected_xy, (16, 1)))
    assert chromaticity_matrix.shape == (4, 4, 2)
    assert np.allclose(chromaticity_matrix, np.broadcast_to(expected_xy.reshape(1, 1, 2), (4, 4, 2)))
    assert np.allclose(roi_chromaticity_mean, expected_xy)


def test_ip_get_roi_data_and_xyz(asset_store) -> None:
    ip = ip_create(display="default", asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)

    roi_locs = np.array([[1, 1], [2, 2]], dtype=int)
    roi_data = ip_get(ip, "roi data", roi_locs)
    roi_xyz = ip_get(ip, "roi xyz", roi_locs)
    full_xyz = imageDataXYZ(ip)

    spd = np.asarray(display_get(ip.fields["display"], "rgb spd"), dtype=float)
    wave = np.asarray(display_get(ip.fields["display"], "wave"), dtype=float)
    expected_roi_xyz = xyz_from_energy(roi_data @ spd.T, wave, asset_store=asset_store)
    expected_full_xyz = xyz_from_energy(result.reshape(-1, 3) @ spd.T, wave, asset_store=asset_store).reshape(2, 2, 3)

    assert np.allclose(roi_data, np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]], dtype=float))
    assert np.allclose(roi_xyz, expected_roi_xyz)
    assert np.allclose(full_xyz, expected_full_xyz)


def test_vc_get_roi_data_ip_xyz(asset_store) -> None:
    ip = ip_create(display="default", asset_store=asset_store)
    result = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)

    roi_locs = np.array([[1, 1], [2, 2]], dtype=int)
    roi_xyz = vcGetROIData(ip, roi_locs, "xyz")
    full_xyz = np.asarray(imageDataXYZ(ip), dtype=float)
    expected = np.vstack((full_xyz[0, 0, :], full_xyz[1, 1, :]))

    assert np.allclose(roi_xyz, expected)


def test_scene_get_roi_queries(asset_store) -> None:
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    roi = np.array([1, 1, 1, 1], dtype=int)

    roi_photons = scene_get(scene, "roi photons", roi)
    roi_energy = scene_get(scene, "roi energy", roi)
    roi_reflectance = scene_get(scene, "roi reflectance", roi)
    roi_luminance = scene_get(scene, "roi luminance", roi)
    roi_chromaticity = scene_get(scene, "chromaticity", roi)
    expected_scene_xy = chromaticity_xy(xyz_from_energy(roi_energy, np.asarray(scene_get(scene, "wave"), dtype=float), asset_store=asset_store))

    assert roi_photons.shape[0] == 4
    assert np.allclose(scene_get(scene, "roi mean photons", roi), np.mean(roi_photons, axis=0))
    assert np.allclose(scene_get(scene, "roi mean energy", roi), np.mean(roi_energy, axis=0))
    assert np.allclose(scene_get(scene, "roi mean reflectance", roi), np.mean(roi_reflectance, axis=0))
    assert roi_luminance.shape == (4, 1)
    assert np.isclose(scene_get(scene, "roi mean luminance", roi), float(np.mean(roi_luminance)))
    assert np.allclose(roi_chromaticity, expected_scene_xy)
    assert np.allclose(scene_get(scene, "roi chromaticity mean", roi), np.mean(expected_scene_xy, axis=0))
    assert np.allclose(roi_reflectance, np.ones_like(roi_reflectance))


def test_scene_get_line_profiles(asset_store) -> None:
    scene = scene_create("uniform ee", 4, asset_store=asset_store)
    scene = scene_set(scene, "wave", np.array([500.0, 600.0], dtype=float))
    scene = scene_set(
        scene,
        "photons",
        np.array(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            ],
            dtype=float,
        ),
    )

    radiance_hline = scene_get(scene, "radiance hline", np.array([2, 1], dtype=int))
    luminance_vline = scene_get(scene, "luminance vline", np.array([2, 1], dtype=int))
    support = scene_get(scene, "spatial support linear", "mm")

    assert np.allclose(radiance_hline["wave"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(radiance_hline["pos"], support["x"])
    assert np.allclose(radiance_hline["data"], np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=float))
    assert radiance_hline["unit"] == "mm"
    assert np.allclose(luminance_vline["pos"], support["y"])
    assert np.allclose(luminance_vline["data"], np.asarray(scene_get(scene, "luminance"), dtype=float)[:, 1])


def test_scene_get_illuminant_roi_and_line_queries(asset_store) -> None:
    scene = scene_create("uniform d65", 4, asset_store=asset_store)
    roi = np.array([1, 1, 1, 1], dtype=int)

    roi_illuminant_photons = scene_get(scene, "roi illuminant photons", roi)
    roi_illuminant_energy = scene_get(scene, "roi illuminant energy", roi)
    illuminant_hline_energy = scene_get(scene, "illuminant hline energy", np.array([2, 1], dtype=int))
    support = scene_get(scene, "spatial support linear", "mm")
    expected_photons = np.asarray(scene_get(scene, "illuminant photons"), dtype=float)
    expected_energy = np.asarray(scene_get(scene, "illuminant energy"), dtype=float)

    assert scene_get(scene, "illuminant comment") == "D65.mat"
    assert np.allclose(roi_illuminant_photons, np.tile(expected_photons, (4, 1)))
    assert np.allclose(roi_illuminant_energy, np.tile(expected_energy, (4, 1)))
    assert np.allclose(scene_get(scene, "roi mean illuminant photons", roi), expected_photons)
    assert np.allclose(scene_get(scene, "roi mean illuminant energy", roi), expected_energy)
    assert np.allclose(illuminant_hline_energy["pos"], support["x"])
    assert np.allclose(illuminant_hline_energy["wave"], np.asarray(scene_get(scene, "wave"), dtype=float))
    assert np.allclose(illuminant_hline_energy["data"], np.tile(expected_energy.reshape(-1, 1), (1, support["x"].size)))


def test_oi_get_roi_queries(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)
    roi = np.array([1, 1, 1, 1], dtype=int)

    roi_photons = oi_get(oi, "roi photons", roi)
    roi_energy = oi_get(oi, "roi energy", roi)
    roi_illuminance = oi_get(oi, "roi illuminance", roi)
    roi_chromaticity = oi_get(oi, "chromaticity", roi)
    expected_oi_xy = chromaticity_xy(xyz_from_energy(roi_energy, np.asarray(oi_get(oi, "wave"), dtype=float), asset_store=asset_store))

    assert np.allclose(roi_photons, photons.reshape(-1, 2))
    assert np.allclose(oi_get(oi, "roi mean photons", roi), np.mean(roi_photons, axis=0))
    assert np.allclose(oi_get(oi, "roi mean energy", roi), np.mean(roi_energy, axis=0))
    assert np.allclose(roi_illuminance[:, 0], oi_get(oi, "illuminance").reshape(-1))
    assert np.isclose(oi_get(oi, "roi mean illuminance", roi), float(np.mean(roi_illuminance)))
    assert np.allclose(roi_chromaticity, expected_oi_xy)
    assert np.allclose(oi_get(oi, "roi chromaticity mean", roi), np.mean(expected_oi_xy, axis=0))


def test_oi_get_line_profiles(asset_store) -> None:
    oi = oi_create(asset_store=asset_store)
    oi = oi_set(oi, "wave", np.array([500.0, 600.0], dtype=float))
    photons = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=float,
    )
    oi = oi_set(oi, "photons", photons)

    irradiance_hline = oi_get(oi, "irradiance hline", np.array([1, 1], dtype=int))
    illuminance_vline = oi_get(oi, "illuminance vline", np.array([2, 1], dtype=int))
    support = oi_get(oi, "spatial support linear", "um")

    assert np.allclose(irradiance_hline["wave"], np.array([500.0, 600.0], dtype=float))
    assert np.allclose(irradiance_hline["pos"], support["x"])
    assert np.allclose(irradiance_hline["data"], np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=float))
    assert irradiance_hline["unit"] == "um"
    assert np.allclose(illuminance_vline["pos"], support["y"])
    assert np.allclose(illuminance_vline["data"], np.asarray(oi_get(oi, "illuminance"), dtype=float)[:, 1])


def test_ip_get_chromaticity_queries(asset_store) -> None:
    ip = ip_create(display="default", asset_store=asset_store)
    result = np.array(
        [
            [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
            [[0.6, 0.7, 0.8], [0.8, 0.9, 1.0]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", result)
    roi_locs = np.array([[1, 1], [2, 2]], dtype=int)

    chromaticity = ip_get(ip, "chromaticity", roi_locs)
    expected_xy = chromaticity_xy(imageDataXYZ(ip, roi_locs))

    assert np.allclose(chromaticity, expected_xy)
    assert np.allclose(ip_get(ip, "roi chromaticity mean", roi_locs), np.mean(expected_xy, axis=0))
