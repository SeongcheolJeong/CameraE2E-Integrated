from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import pyisetcam.ip as ip_module
from scipy.signal import convolve2d

from pyisetcam.camera import _chart_rectangles, _chart_rects_data, _linear_srgb_to_xyz, _whole_chart_corner_points
from pyisetcam import (
    FaultyBilinear,
    FaultyNearestNeighbor,
    LFAutofocus,
    LFFiltShiftSum,
    LFConvertToFloat,
    LFDefaultField,
    LFDefaultVal,
    LFImage2buffer,
    LFToolboxVersion,
    LFbuffer2SubApertureViews,
    LFbuffer2image,
    Pocs,
    camera_create,
    camera_mtf,
    demosaic,
    demosaicMultichannel,
    demosaicRCCC,
    displayRender,
    display_get,
    faultyInsert,
    faultyList,
    ip2lightfield,
    ieColorTransform,
    ieInternal2Display,
    iePixelWellCapacity,
    ieRadiance2IP,
    imageColorBalance,
    imageDataXYZ,
    imageDistort,
    imageEsserTransform,
    imageIlluminantCorrection,
    imageMCCTransform,
    imageRGB2XYZ,
    imageShowImage,
    imageSensorConversion,
    imageSensorCorrection,
    imageSensorTransform,
    ie_reflectance_samples,
    ipMCCXYZ,
    ipHDRWhite,
    ip_compute,
    ip_create,
    ip_get,
    ip_set,
    ipClearData,
    ipSaveImage,
    ie_radiance_to_ip,
    oi_compute,
    oi_create,
    oi_get,
    scene_create,
    scene_get,
    scene_set,
    sensor_compute,
    vcimageClearData,
    vcimageISOMTF,
    vcimageMCCXYZ,
    vcimageSRGB,
    vcimageVSNR,
    sensor_create,
    sensor_get,
    sensor_set,
)
from pyisetcam.assets import ie_read_spectra
from pyisetcam.color import sensor_to_target_matrix, xyz_color_matching
from pyisetcam.scene import hdr_render
from pyisetcam.utils import energy_to_quanta, linear_to_srgb, rgb_to_xw_format, xw_to_rgb_format


def _sparse_cfa_planes(sensor, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pattern = np.asarray(sensor_get(sensor, "pattern"), dtype=int)
    rows, cols, bands = full.shape
    tiled = np.tile(
        pattern,
        (
            int(np.ceil(rows / pattern.shape[0])),
            int(np.ceil(cols / pattern.shape[1])),
        ),
    )[:rows, :cols]
    sparse = np.zeros_like(full, dtype=float)
    for band in range(bands):
        mask = tiled == (band + 1)
        sparse[:, :, band][mask] = full[:, :, band][mask]
    return sparse, tiled


def test_image_show_image_matches_headless_rgb_and_gray_render_paths() -> None:
    ip = ip_create()
    linear = np.array(
        [
            [[0.2, 0.4, 0.6], [0.1, 0.2, 0.3]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", linear)
    ip = ip_set(ip, "scale display", False)

    rgb = imageShowImage(ip, 1.0, False, 0)
    expected_rgb = linear_to_srgb(linear)
    assert np.allclose(rgb, expected_rgb)

    gray_ip = ip_set(ip, "render flag", "gray")
    gray = imageShowImage(gray_ip, 1.0, False, 0)
    expected_gray = np.mean(expected_rgb, axis=2, keepdims=True)
    expected_gray = np.repeat(expected_gray, 3, axis=2)
    assert np.allclose(gray, expected_gray)


def test_image_show_image_matches_hdr_render_headlessly() -> None:
    ip = ip_create()
    linear = np.array(
        [
            [[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]],
            [[0.2, 0.1, 0.4], [0.5, 0.6, 0.9]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", linear)
    ip = ip_set(ip, "scale display", False)
    ip = ip_set(ip, "render flag", "hdr")

    rendered = imageShowImage(ip, 1.0, False, 0)
    expected = hdr_render(linear_to_srgb(linear))
    assert np.allclose(rendered, expected)


def test_ip_create_empty_name_placeholder_uses_default_constructor(asset_store) -> None:
    default_ip = ip_create(asset_store=asset_store)
    placeholder_ip = ip_create([], asset_store=asset_store)

    assert placeholder_ip.name == default_ip.name
    np.testing.assert_allclose(ip_get(placeholder_ip, "wave"), ip_get(default_ip, "wave"))
    assert display_get(ip_get(placeholder_ip, "display"), "name") == display_get(ip_get(default_ip, "display"), "name")
    assert ip_get(placeholder_ip, "demosaic method") == ip_get(default_ip, "demosaic method")


def test_ip_compute_supports_rgb_bayer_demosaic_methods(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    mosaic = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ],
        dtype=float,
    )
    sensor = sensor_set(sensor, "volts", mosaic)

    expected = {
        "nearest neighbor": np.array(
            [
                [
                    [2.0, 1.0, 7.0],
                    [2.0, 1.0, 7.0],
                    [4.0, 3.0, 9.0],
                    [4.0, 3.0, 9.0],
                    [6.0, 5.0, 11.0],
                    [6.0, 5.0, 11.0],
                ],
                [
                    [2.0, 8.0, 7.0],
                    [2.0, 8.0, 7.0],
                    [4.0, 10.0, 9.0],
                    [4.0, 10.0, 9.0],
                    [6.0, 12.0, 11.0],
                    [6.0, 12.0, 11.0],
                ],
                [
                    [14.0, 13.0, 19.0],
                    [14.0, 13.0, 19.0],
                    [16.0, 15.0, 21.0],
                    [16.0, 15.0, 21.0],
                    [18.0, 17.0, 23.0],
                    [18.0, 17.0, 23.0],
                ],
                [
                    [14.0, 20.0, 19.0],
                    [14.0, 20.0, 19.0],
                    [16.0, 22.0, 21.0],
                    [16.0, 22.0, 21.0],
                    [18.0, 24.0, 23.0],
                    [18.0, 24.0, 23.0],
                ],
                [
                    [26.0, 25.0, 31.0],
                    [26.0, 25.0, 31.0],
                    [28.0, 27.0, 33.0],
                    [28.0, 27.0, 33.0],
                    [30.0, 29.0, 35.0],
                    [30.0, 29.0, 35.0],
                ],
                [
                    [26.0, 32.0, 31.0],
                    [26.0, 32.0, 31.0],
                    [28.0, 34.0, 33.0],
                    [28.0, 34.0, 33.0],
                    [30.0, 36.0, 35.0],
                    [30.0, 36.0, 35.0],
                ],
            ],
            dtype=float,
        ),
        "laplacian": np.array(
            [
                [
                    [0.625, 1.0, 7.5],
                    [2.0, 4.5, 9.875],
                    [1.625, 3.0, 9.75],
                    [4.0, 7.0, 12.125],
                    [3.625, 5.0, 11.5],
                    [6.0, 8.5, 12.875],
                ],
                [
                    [6.125, 4.5, 7.0],
                    [7.5, 8.0, 9.375],
                    [6.875, 6.0, 9.0],
                    [9.25, 10.0, 11.375],
                    [9.125, 8.5, 11.0],
                    [11.5, 12.0, 12.375],
                ],
                [
                    [14.125, 13.0, 13.5],
                    [14.0, 13.5, 14.375],
                    [15.125, 15.0, 15.75],
                    [16.0, 16.0, 16.625],
                    [17.125, 17.0, 17.5],
                    [18.0, 17.5, 17.375],
                ],
                [
                    [19.625, 19.5, 19.0],
                    [19.5, 20.0, 19.875],
                    [20.375, 21.0, 21.0],
                    [21.25, 22.0, 21.875],
                    [22.625, 23.5, 23.0],
                    [23.5, 24.0, 22.875],
                ],
                [
                    [24.625, 25.0, 25.5],
                    [26.0, 28.5, 27.875],
                    [25.625, 27.0, 27.75],
                    [28.0, 31.0, 30.125],
                    [27.625, 29.0, 29.5],
                    [30.0, 32.5, 30.875],
                ],
                [
                    [24.125, 28.5, 31.0],
                    [25.5, 32.0, 33.375],
                    [24.875, 30.0, 33.0],
                    [27.25, 34.0, 35.375],
                    [27.125, 32.5, 35.0],
                    [29.5, 36.0, 36.375],
                ],
            ],
            dtype=float,
        ),
        "adaptive laplacian": np.array(
            [
                [
                    [2.25, 1.0, 6.5],
                    [2.0, 1.0, 7.25],
                    [3.25, 3.0, 9.0],
                    [4.0, 4.0, 9.75],
                    [5.25, 5.0, 10.5],
                    [6.0, 5.0, 10.25],
                ],
                [
                    [8.75, 8.0, 7.0],
                    [8.5, 8.0, 7.75],
                    [9.25, 9.0, 9.0],
                    [10.0, 10.0, 9.75],
                    [11.75, 12.0, 11.0],
                    [12.5, 12.0, 10.75],
                ],
                [
                    [14.25, 13.0, 12.5],
                    [14.0, 13.0, 13.25],
                    [15.25, 15.0, 15.0],
                    [16.0, 16.0, 15.75],
                    [17.25, 17.0, 16.5],
                    [18.0, 17.0, 16.25],
                ],
                [
                    [20.75, 20.0, 19.0],
                    [20.5, 20.0, 19.75],
                    [21.25, 21.0, 21.0],
                    [22.0, 22.0, 21.75],
                    [23.75, 24.0, 23.0],
                    [24.5, 24.0, 22.75],
                ],
                [
                    [26.25, 25.0, 24.5],
                    [26.0, 25.0, 25.25],
                    [27.25, 27.0, 27.0],
                    [28.0, 28.0, 27.75],
                    [29.25, 29.0, 28.5],
                    [30.0, 29.0, 28.25],
                ],
                [
                    [26.75, 32.0, 31.0],
                    [26.5, 32.0, 31.75],
                    [27.25, 33.0, 33.0],
                    [28.0, 34.0, 33.75],
                    [29.75, 36.0, 35.0],
                    [30.5, 36.0, 34.75],
                ],
            ],
            dtype=float,
        ),
    }

    for method, expected_sensor_space in expected.items():
        ip = ip_create(asset_store=asset_store)
        ip = ip_set(ip, "demosaic method", method)
        computed = ip_compute(ip, sensor, asset_store=asset_store)

        assert np.allclose(
            np.asarray(ip_get(computed, "sensorspace"), dtype=float), expected_sensor_space
        )


def test_ip_compute_adaptive_laplacian_falls_back_to_bilinear_for_gbrg(asset_store) -> None:
    sensor = sensor_create("bayer-gbrg", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))

    bilinear = ip_compute(
        ip_set(ip_create(asset_store=asset_store), "demosaic method", "bilinear"),
        sensor,
        asset_store=asset_store,
    )
    adaptive = ip_compute(
        ip_set(ip_create(asset_store=asset_store), "demosaic method", "adaptive laplacian"),
        sensor,
        asset_store=asset_store,
    )

    assert np.allclose(
        np.asarray(ip_get(adaptive, "sensorspace"), dtype=float),
        np.asarray(ip_get(bilinear, "sensorspace"), dtype=float),
    )


def test_demosaic_matches_ip_compute_sensor_space(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))

    for method in ("bilinear", "nearest neighbor", "laplacian", "adaptive laplacian"):
        ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", method)
        computed = ip_compute(ip, sensor, asset_store=asset_store)

        assert np.allclose(
            np.asarray(demosaic(ip, sensor), dtype=float),
            np.asarray(ip_get(computed, "sensorspace"), dtype=float),
        )


def test_demosaic_prefers_ip_input_over_sensor_storage(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    override = np.arange(101, 137, dtype=float).reshape(6, 6)

    reference_sensor = sensor_create("default", asset_store=asset_store)
    reference_sensor = sensor_set(reference_sensor, "volts", override)
    reference_ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", "nearest neighbor")
    reference = ip_compute(reference_ip, reference_sensor, asset_store=asset_store)

    ip = ip_set(ip_create(asset_store=asset_store), "demosaic method", "nearest neighbor")
    ip.data["input"] = override

    assert np.allclose(
        np.asarray(demosaic(ip, sensor), dtype=float),
        np.asarray(ip_get(reference, "sensorspace"), dtype=float),
    )


def test_demosaic_returns_monochrome_sensor_plane(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    volts = np.arange(1, 13, dtype=float).reshape(3, 4)
    sensor = sensor_set(sensor, "volts", volts)
    ip = ip_create(asset_store=asset_store)

    result = np.asarray(demosaic(ip, sensor), dtype=float)

    assert result.shape == (3, 4, 1)
    assert np.allclose(result[:, :, 0], volts)


def test_demosaic_multichannel_fills_sparse_multifilter_planes(asset_store) -> None:
    sensor = sensor_create("rgbw", asset_store=asset_store)
    sensor = sensor_set(sensor, "size", (6, 6))

    rows, cols = np.indices((6, 6), dtype=float)
    full = np.stack(
        [
            0.10 + 0.05 * rows + 0.02 * cols,
            0.20 + 0.03 * rows + 0.04 * cols,
            0.30 + 0.02 * rows + 0.01 * cols,
            0.40 + 0.01 * rows + 0.03 * cols,
        ],
        axis=2,
    )
    sparse, tiled = _sparse_cfa_planes(sensor, full)

    interpolated = demosaicMultichannel(sparse, sensor, "interpolate")
    assert interpolated.shape == full.shape
    assert np.all(np.isfinite(interpolated))
    assert np.allclose(interpolated[1:-1, 1:-1, :], full[1:-1, 1:-1, :], atol=1.0e-10)

    mean_result = demosaicMultichannel(sparse, sensor, "mean")
    median_result = demosaicMultichannel(sparse, sensor, "median")
    for band in range(full.shape[2]):
        mask = tiled == (band + 1)
        assert np.allclose(interpolated[:, :, band][mask], full[:, :, band][mask])
        assert np.allclose(mean_result[:, :, band][mask], full[:, :, band][mask])
        assert np.allclose(median_result[:, :, band][mask], full[:, :, band][mask])
        assert np.all(np.isfinite(mean_result[:, :, band]))
        assert np.all(np.isfinite(median_result[:, :, band]))


def test_pocs_preserves_observed_samples_for_rggb_input(asset_store) -> None:
    sensor = sensor_create("bayer-rggb", asset_store=asset_store)
    sensor = sensor_set(sensor, "size", (6, 6))

    rows, cols = np.indices((6, 6), dtype=float)
    full = np.stack(
        [
            0.10 + 0.04 * rows + 0.03 * cols,
            0.20 + 0.02 * rows + 0.01 * cols,
            0.30 + 0.01 * rows + 0.05 * cols,
        ],
        axis=2,
    )
    sparse, tiled = _sparse_cfa_planes(sensor, full)

    result = Pocs(sparse, sensor_get(sensor, "pattern"), 4)
    assert result.shape == full.shape
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)

    for band in range(full.shape[2]):
        mask = tiled == (band + 1)
        assert np.allclose(result[:, :, band][mask], full[:, :, band][mask])


def test_faulty_pixel_helpers_match_legacy_grbg_rules() -> None:
    faulty = faultyList(8, 8, 3, 2, rng=np.random.default_rng(7))
    assert faulty.shape == (3, 2)
    assert np.unique(faulty, axis=0).shape[0] == 3
    assert np.all((faulty[:, 0] >= 1) & (faulty[:, 0] <= 8))
    assert np.all((faulty[:, 1] >= 1) & (faulty[:, 1] <= 8))
    distances = np.sqrt(np.sum((faulty[:, None, :] - faulty[None, :, :]) ** 2, axis=2))
    nearest = distances + np.eye(faulty.shape[0], dtype=float) * 1.0e9
    assert np.all(np.min(nearest, axis=1) >= 2.0)

    planes = np.zeros((6, 6, 3), dtype=float)
    planes[0::2, 1::2, 0] = 10.0 * np.arange(1, 10, dtype=float).reshape(3, 3)
    planes[0::2, 0::2, 1] = 20.0 * np.arange(1, 10, dtype=float).reshape(3, 3)
    planes[1::2, 1::2, 1] = 30.0 * np.arange(1, 10, dtype=float).reshape(3, 3)
    planes[1::2, 0::2, 2] = 40.0 * np.arange(1, 10, dtype=float).reshape(3, 3)

    faulty_sites = np.array([[4, 3], [3, 3], [3, 4]], dtype=int)
    inserted = faultyInsert(faulty_sites, planes, 0.0)
    assert np.allclose(inserted[2, 3, :], 0.0)
    assert np.allclose(inserted[2, 2, :], 0.0)
    assert np.allclose(inserted[3, 2, :], 0.0)

    nearest_fixed = FaultyNearestNeighbor(faulty_sites, inserted)
    bilinear_fixed = FaultyBilinear(faulty_sites, inserted)

    assert np.isclose(nearest_fixed[2, 3, 0], planes[4, 3, 0])
    assert np.isclose(
        bilinear_fixed[2, 3, 0],
        0.25 * (planes[0, 3, 0] + planes[2, 1, 0] + planes[4, 3, 0] + planes[2, 5, 0]),
    )
    assert np.isclose(nearest_fixed[2, 2, 1], planes[3, 3, 1])
    assert np.isclose(
        bilinear_fixed[2, 2, 1],
        0.25 * (planes[1, 1, 1] + planes[1, 3, 1] + planes[3, 1, 1] + planes[3, 3, 1]),
    )
    assert np.isclose(nearest_fixed[3, 2, 2], planes[5, 2, 2])
    assert np.isclose(
        bilinear_fixed[3, 2, 2],
        0.25 * (planes[1, 2, 2] + planes[3, 0, 2] + planes[5, 2, 2] + planes[3, 4, 2]),
    )


def test_lightfield_helper_utilities_match_expected_python_contract(asset_store) -> None:
    assert LFDefaultVal(None, 3) == 3
    assert LFDefaultVal(42, 3) == 42
    assert LFDefaultField(None, "cheese", "indeed") == {"cheese": "indeed"}
    assert LFDefaultField({"existing": 42}, "existing", 3) == {"existing": 42}
    assert LFToolboxVersion() == "v0.4 released 12-Feb-2015"

    int_lf = np.array([[0, 255]], dtype=np.uint8)
    converted = LFConvertToFloat(int_lf)
    assert converted.dtype == np.float32
    assert np.allclose(converted, np.array([[0.0, 1.0]], dtype=np.float32))
    assert LFConvertToFloat(np.array([[1.5]], dtype=np.float64), "double").dtype == np.float64

    image = np.arange(4 * 6 * 3, dtype=float).reshape(4, 6, 3)
    buffer_array = LFImage2buffer(image, 3, 2)
    assert buffer_array.shape == (2, 3, 2, 2, 3)
    assert np.allclose(LFbuffer2image(buffer_array), image)

    subaperture_array, corners = LFbuffer2SubApertureViews(buffer_array)
    assert subaperture_array.shape == (4, 6, 3)
    assert np.array_equal(corners[0, 0, :], np.array([1.0, 1.0], dtype=float))
    assert np.array_equal(corners[1, 1, :], np.array([3.0, 4.0], dtype=float))
    assert np.allclose(subaperture_array[0:2, 0:3, :], buffer_array[:, :, 0, 0, :])

    ip = ip_create(asset_store=asset_store)
    result = np.linspace(0.0, 1.0, 4 * 6 * 3, dtype=float).reshape(4, 6, 3)
    ip = ip_set(ip, "result", result)
    lightfield = ip2lightfield(ip, pinholes=[2, 3], colorspace="linear")
    assert lightfield.shape == (2, 2, 2, 3, 3)
    assert np.allclose(lightfield[:, :, 0, 0, :], result[0:2, 0:2, :])
    assert np.allclose(lightfield[:, :, 1, 2, :], result[2:4, 4:6, :])


def test_demosaic_rccc_matches_upstream_convolution_rule() -> None:
    mosaic = np.zeros((4, 5, 2), dtype=float)
    mosaic[:, :, 1] = np.array(
        [
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ],
        dtype=float,
    )
    mosaic[0::2, 1::2, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    mosaic[1::2, 0::2, 0] = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=float)

    kernel = np.array(
        [
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [-1.0, 2.0, 4.0, 2.0, -1.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
        ],
        dtype=float,
    ) / 8.0
    mosaic_ex = np.concatenate((mosaic[:, 1:2, :], mosaic, mosaic[:, -2:-1, :]), axis=1)
    mosaic_ex = np.concatenate((mosaic_ex[1:2, :, :], mosaic_ex, mosaic_ex[-2:-1, :, :]), axis=0)
    red = mosaic_ex[:, :, 0]
    clear = mosaic_ex[:, :, 1]
    red_mask = (red != 0).astype(float)
    expected = clear[1:-1, 1:-1] + (convolve2d(red + clear, kernel, mode="same") * red_mask)[1:-1, 1:-1]

    actual = demosaicRCCC(mosaic)

    assert actual.shape == mosaic.shape[:2]
    assert np.allclose(actual, expected)


def test_lf_shift_sum_and_autofocus_match_headless_focus_contract() -> None:
    lightfield = np.zeros((2, 2, 6, 6, 3), dtype=float)
    base = np.zeros((6, 6, 3), dtype=float)
    base[1:5, 2:4, 0] = 1.0
    base[2:4, 1:5, 1] = 0.5
    base[2:5, 2:5, 2] = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.8, 0.4],
            [0.3, 0.2, 0.1],
        ],
        dtype=float,
    )

    def shift_with_zeros(img: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
        out = np.zeros_like(img)
        src_row0 = max(-row_shift, 0)
        src_row1 = min(img.shape[0] - row_shift, img.shape[0])
        src_col0 = max(-col_shift, 0)
        src_col1 = min(img.shape[1] - col_shift, img.shape[1])
        dst_row0 = max(row_shift, 0)
        dst_row1 = dst_row0 + (src_row1 - src_row0)
        dst_col0 = max(col_shift, 0)
        dst_col1 = dst_col0 + (src_col1 - src_col0)
        out[dst_row0:dst_row1, dst_col0:dst_col1, :] = img[src_row0:src_row1, src_col0:src_col1, :]
        return out

    true_slope = 1.0
    v_offsets = np.linspace(-0.5, 0.5, 2) * true_slope * 2.0
    u_offsets = np.linspace(-0.5, 0.5, 2) * true_slope * 2.0
    for t_index, v_offset in enumerate(v_offsets):
        for s_index, u_offset in enumerate(u_offsets):
            lightfield[t_index, s_index, :, :, :] = shift_with_zeros(
                base,
                int(round(v_offset)),
                int(round(u_offset)),
            )

    mean_img, filt_options, shifted_lf = LFFiltShiftSum(lightfield, 0.0)
    assert mean_img.shape == (6, 6, 4)
    assert shifted_lf.shape == (2, 2, 6, 6, 4)
    assert np.allclose(mean_img[:, :, :3], np.mean(lightfield, axis=(0, 1)))
    assert np.allclose(mean_img[:, :, 3], 4.0)
    assert filt_options["FlattenMethod"] == "sum"

    focused = LFAutofocus(lightfield, None, [0.0, 1.0], 1.0)
    assert focused.shape == base.shape
    assert np.allclose(focused, base, atol=1.0e-6)


def test_image_sensor_conversion_matches_direct_formula(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    spectral_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)

    cmf = np.column_stack(
        (
            np.linspace(0.2, 0.7, wave.size),
            np.linspace(0.3, 0.9, wave.size),
            np.linspace(0.8, 0.1, wave.size),
        )
    )
    surfaces = np.column_stack(
        (
            np.linspace(0.1, 0.9, wave.size),
            np.linspace(0.9, 0.2, wave.size),
            np.linspace(0.3, 0.6, wave.size),
        )
    )
    illuminant = np.linspace(1.0, 2.0, wave.size)

    transform, actual, desired, white_cmf = imageSensorConversion(
        sensor,
        cmf,
        surfaces,
        illuminant,
        asset_store=asset_store,
    )

    weighted_surfaces = illuminant.reshape(-1, 1) * surfaces
    expected_actual = spectral_qe.T @ weighted_surfaces
    expected_desired = cmf.T @ weighted_surfaces
    expected_transform = expected_desired @ np.linalg.pinv(expected_actual)
    expected_white_cmf = cmf.T @ illuminant

    assert np.allclose(actual, expected_actual)
    assert np.allclose(desired, expected_desired)
    assert np.allclose(transform, expected_transform)
    assert np.allclose(white_cmf, expected_white_cmf)


def test_image_sensor_correction_matches_ip_compute_sensor_transform(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")

    sensor_space = demosaic(ip, sensor)
    corrected, corrected_ip, sensor_transform = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(corrected, np.asarray(ip_get(computed, "xyz"), dtype=float))
    assert np.allclose(
        sensor_transform,
        np.asarray(ip_get(computed, "conversion transform sensor"), dtype=float),
    )
    assert np.allclose(np.asarray(ip_get(corrected_ip, "xyz"), dtype=float), corrected)


def test_image_sensor_correction_preserves_2d_monochrome_input(asset_store) -> None:
    sensor = sensor_create("monochrome", asset_store=asset_store)
    img = np.arange(1, 13, dtype=float).reshape(3, 4)
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "none")

    corrected, corrected_ip, sensor_transform = imageSensorCorrection(
        img,
        ip,
        sensor,
        asset_store=asset_store,
    )

    assert corrected.shape == img.shape
    assert np.allclose(corrected, img / np.max(img))
    assert sensor_transform.shape == (1, 1)
    assert np.allclose(np.asarray(ip_get(corrected_ip, "sensorspace"), dtype=float)[:, :, 0], img)


def test_t_ip_tutorial_workflow_is_supported_headlessly(asset_store) -> None:
    scene = scene_create("macbeth tungsten", asset_store=asset_store)
    oi = oi_create(asset_store=asset_store)
    sensor = sensor_set(sensor_create("default", asset_store=asset_store), "size", [340, 420])

    fov = sensor_get(sensor, "fov", scene_get(scene, "distance"), oi)
    scene = scene_set(scene, "fov", fov)
    oi = oi_compute(oi, scene)
    sensor = sensor_compute(sensor, oi)

    ip = ip_set(ip_create(asset_store=asset_store), "name", "default")
    ip = ip_set(ip, "internal cs", "XYZ")
    ip = ip_set(ip, "conversion method sensor", "MCC Optimized")
    xyz_ip = ip_compute(ip, sensor, asset_store=asset_store)

    gray_world_ip = ip_set(xyz_ip, "illuminant correction method", "gray world")
    gray_world_ip = ip_compute(gray_world_ip, sensor, asset_store=asset_store)

    adaptive_ip = ip_set(gray_world_ip, "demosaic method", "Adaptive Laplacian")
    adaptive_ip = ip_compute(adaptive_ip, sensor, asset_store=asset_store)

    assert ip_get(xyz_ip, "name") == "default"
    assert str(ip_get(xyz_ip, "internal cs")) == "XYZ"
    assert str(ip_get(xyz_ip, "conversion method sensor")) == "MCC Optimized"
    assert tuple(np.asarray(ip_get(xyz_ip, "result"), dtype=float).shape) == (340, 420, 3)
    assert tuple(np.asarray(ip_get(gray_world_ip, "result"), dtype=float).shape) == (340, 420, 3)
    assert tuple(np.asarray(ip_get(adaptive_ip, "result"), dtype=float).shape) == (340, 420, 3)
    assert str(ip_get(gray_world_ip, "illuminant correction method")) == "gray world"
    assert str(ip_get(adaptive_ip, "demosaic method")) == "adaptive laplacian"
    assert np.asarray(ip_get(gray_world_ip, "illuminant correction transform"), dtype=float).shape == (3, 3)
    assert display_get(ip_get(adaptive_ip, "display"), "dpi") > 0


def test_image_illuminant_correction_matches_ip_compute_ics(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")
    ip = ip_set(ip, "correction method illuminant", "gray world")

    sensor_space = demosaic(ip, sensor)
    internal, corrected_ip, _ = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    corrected, illuminant_ip, illuminant_transform = imageIlluminantCorrection(
        internal,
        corrected_ip,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(corrected, np.asarray(ip_get(computed, "ics"), dtype=float))
    assert np.allclose(
        illuminant_transform,
        np.asarray(ip_get(computed, "illuminant correction transform"), dtype=float),
    )
    assert np.allclose(np.asarray(ip_get(illuminant_ip, "ics"), dtype=float), corrected)


def test_image_illuminant_correction_manual_matrix(asset_store) -> None:
    img = np.array(
        [
            [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
            [[0.8, 0.1, 0.2], [0.9, 0.3, 0.4]],
        ],
        dtype=float,
    )
    transform = np.diag([1.2, 0.8, 1.1])
    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "manual")
    ip = ip_set(ip, "illuminant correction transform", transform)

    corrected, corrected_ip, returned_transform = imageIlluminantCorrection(
        img,
        ip,
        asset_store=asset_store,
    )

    assert np.allclose(corrected, img @ transform)
    assert np.allclose(returned_transform, transform)
    assert np.allclose(
        np.asarray(ip_get(corrected_ip, "illuminant correction transform"), dtype=float),
        transform,
    )


def test_image_illuminant_correction_preserves_2d_input(asset_store) -> None:
    img = np.arange(1, 13, dtype=float).reshape(3, 4)
    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "none")

    corrected, corrected_ip, illuminant_transform = imageIlluminantCorrection(
        img,
        ip,
        asset_store=asset_store,
    )

    assert corrected.shape == img.shape
    assert np.allclose(corrected, img)
    assert illuminant_transform.shape == (1, 1)
    assert np.allclose(np.asarray(ip_get(corrected_ip, "ics"), dtype=float)[:, :, 0], img)


def test_image_color_balance_matches_image_illuminant_correction(asset_store) -> None:
    img = np.array(
        [
            [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
            [[0.8, 0.1, 0.2], [0.9, 0.3, 0.4]],
        ],
        dtype=float,
    )
    transform = np.diag([1.2, 0.8, 1.1])
    ip = ip_set(ip_create(asset_store=asset_store), "correction method illuminant", "manual")
    ip = ip_set(ip, "illuminant correction transform", transform)

    balanced, balanced_ip, balanced_transform = imageColorBalance(
        img,
        ip,
        asset_store=asset_store,
    )
    corrected, corrected_ip, corrected_transform = imageIlluminantCorrection(
        img,
        ip,
        asset_store=asset_store,
    )

    assert np.allclose(balanced, corrected)
    assert np.allclose(balanced_transform, corrected_transform)
    assert np.allclose(np.asarray(ip_get(balanced_ip, "ics"), dtype=float), corrected)
    assert np.allclose(
        np.asarray(ip_get(balanced_ip, "illuminant correction transform"), dtype=float),
        np.asarray(ip_get(corrected_ip, "illuminant correction transform"), dtype=float),
    )


def test_image_rgb_to_xyz_preserves_rgb_format(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb = np.array(
        [
            [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]],
            [[0.7, 0.2, 0.1], [0.9, 0.8, 0.2]],
        ],
        dtype=float,
    )

    xyz = np.asarray(imageRGB2XYZ(ip, rgb, asset_store=asset_store), dtype=float)

    assert xyz.shape == rgb.shape
    manual = imageRGB2XYZ(ip, rgb.reshape(-1, 3), asset_store=asset_store).reshape(rgb.shape)
    assert np.allclose(xyz, manual)


def test_image_rgb_to_xyz_preserves_xw_format(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb_xw = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5],
            [0.7, 0.2, 0.1],
            [0.9, 0.8, 0.2],
        ],
        dtype=float,
    )

    xyz_xw = np.asarray(imageRGB2XYZ(ip, rgb_xw, asset_store=asset_store), dtype=float)

    assert xyz_xw.shape == rgb_xw.shape
    assert np.allclose(xyz_xw, imageRGB2XYZ(ip, rgb_xw.copy(), asset_store=asset_store))


def test_image_data_xyz_uses_image_rgb_to_xyz_for_result(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    rgb = np.array(
        [
            [[0.25, 0.50, 0.75], [0.10, 0.20, 0.30]],
            [[0.60, 0.40, 0.20], [0.90, 0.70, 0.50]],
        ],
        dtype=float,
    )
    ip.data["result"] = rgb

    xyz_from_result = np.asarray(imageDataXYZ(ip, asset_store=asset_store), dtype=float)
    xyz_direct = np.asarray(imageRGB2XYZ(ip, rgb, asset_store=asset_store), dtype=float)

    assert np.allclose(xyz_from_result, xyz_direct)


def test_display_render_matches_ip_compute_result(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "conversion method sensor", "mcc optimized")
    ip = ip_set(ip, "correction method illuminant", "gray world")

    sensor_space = demosaic(ip, sensor)
    internal, corrected_ip, _ = imageSensorCorrection(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    ics, corrected_ip, _ = imageIlluminantCorrection(
        internal,
        corrected_ip,
        asset_store=asset_store,
    )
    display_linear, rendered_ip, display_transform = displayRender(
        ics,
        corrected_ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(display_linear, np.asarray(ip_get(computed, "result"), dtype=float))
    assert np.allclose(display_transform, np.asarray(ip_get(computed, "ics2display"), dtype=float))
    assert np.allclose(np.asarray(ip_get(rendered_ip, "result"), dtype=float), display_linear)


def test_display_render_matches_sensor_internal_space_path(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    ip = ip_set(ip_create(asset_store=asset_store), "internal cs", "sensor")
    ip = ip_set(ip, "conversion method sensor", "none")
    ip = ip_set(ip, "correction method illuminant", "none")

    sensor_space = demosaic(ip, sensor)
    display_linear, rendered_ip, display_transform = displayRender(
        sensor_space,
        ip,
        sensor,
        asset_store=asset_store,
    )
    computed = ip_compute(ip, sensor, asset_store=asset_store)

    assert np.allclose(display_linear, np.asarray(ip_get(computed, "result"), dtype=float))
    assert np.allclose(display_transform, np.asarray(ip_get(computed, "ics2display"), dtype=float))
    assert np.allclose(np.asarray(ip_get(rendered_ip, "result"), dtype=float), display_linear)


def test_ip_clear_data_clears_computed_payloads(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    cleared = ipClearData(computed)

    assert cleared is not computed
    assert ip_get(cleared, "result") is None
    assert ip_get(cleared, "input") is None
    assert ip_get(cleared, "xyz") is None
    assert ip_get(cleared, "ics") is None
    transforms = ip_get(cleared, "transforms")
    assert len(transforms) == 3
    assert transforms == [None, None, None]
    assert ip_get(cleared, "internal cs") == ip_get(computed, "internal cs")


def test_ip_set_data_empty_matches_ip_clear_data(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    cleared_by_set = ip_set(computed.clone(), "data", [])
    cleared_by_wrapper = ipClearData(computed)

    assert ip_get(cleared_by_set, "result") is None
    assert ip_get(cleared_by_set, "input") is None
    assert ip_get(cleared_by_set, "transforms") == [None, None, None]
    assert ip_get(cleared_by_wrapper, "transforms") == [None, None, None]


def test_vcimage_clear_data_matches_ip_clear_data(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    cleared_by_ip = ipClearData(computed)
    cleared_by_vcimage = vcimageClearData(computed)

    assert cleared_by_vcimage is not computed
    assert ip_get(cleared_by_vcimage, "result") is None
    assert ip_get(cleared_by_vcimage, "input") is None
    assert ip_get(cleared_by_vcimage, "xyz") is None
    assert ip_get(cleared_by_vcimage, "ics") is None
    assert ip_get(cleared_by_vcimage, "transforms") == [None, None, None]
    assert ip_get(cleared_by_vcimage, "internal cs") == ip_get(cleared_by_ip, "internal cs")
    assert cleared_by_vcimage.data == cleared_by_ip.data


def test_ip_get_white_point_aliases_match_upstream_fallbacks() -> None:
    ip = ip_create()
    display_white = np.asarray(display_get(ip_get(ip, "display"), "white point"), dtype=float)

    assert np.allclose(np.asarray(ip_get(ip, "data or display white point"), dtype=float), display_white)
    assert np.allclose(np.asarray(ip_get(ip, "data or monitor white point"), dtype=float), display_white)

    data_white = np.array([0.95, 1.0, 1.09], dtype=float)
    ip = ip_set(ip, "data white point", data_white)

    for alias in (
        "data white point",
        "data wp",
        "white point",
        "wp",
        "image white point",
        "image wp",
        "data or display white point",
        "data or monitor white point",
    ):
        assert np.allclose(np.asarray(ip_get(ip, alias), dtype=float), data_white)


def test_ip_module_core_matlab_aliases() -> None:
    assert ip_module.ipCreate is ip_module.ip_create
    assert ip_module.ipCompute is ip_module.ip_compute
    assert ip_module.ipGet is ip_module.ip_get
    assert ip_module.ipSet is ip_module.ip_set


def test_ip_get_scaled_result_and_primary_aliases_match_headless_helpers() -> None:
    ip = ip_create()
    linear = np.array(
        [
            [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]],
            [[0.6, 0.3, 0.1], [0.8, 0.4, 0.2]],
        ],
        dtype=float,
    )
    corrected_ics = np.array(
        [
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2], [0.3, 0.2, 0.1]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", linear)
    ip = ip_set(ip, "scale display output", False)
    ip.data["ics"] = corrected_ics

    expected_scaled = imageShowImage(ip_set(ip.clone(), "scale display output", True), 1.0, False, 0)
    assert np.allclose(np.asarray(ip_get(ip, "scaled result"), dtype=float), expected_scaled)
    assert np.allclose(np.asarray(ip_get(ip, "result scaled to max"), dtype=float), expected_scaled)
    assert np.allclose(np.asarray(ip_get(ip, "data intensities scaled"), dtype=float), expected_scaled)

    assert np.allclose(np.asarray(ip_get(ip, "data ics illuminant corrected"), dtype=float), corrected_ics)
    assert np.allclose(np.asarray(ip_get(ip, "result primary", 2), dtype=float), linear[:, :, 1])
    assert np.allclose(np.asarray(ip_get(ip, "data intensity", 2), dtype=float), linear[:, :, 1])


def test_ip_get_quantization_and_digital_value_aliases_match_upstream() -> None:
    ip = ip_create()
    linear = np.array(
        [
            [[0.25, 0.5, 0.75]],
        ],
        dtype=float,
    )
    ip = ip_set(ip, "result", linear)
    ip = ip_set(ip, "quantization", {"method": "12 bit"})
    ip = ip_set(ip, "nbits", 12)

    quantization = ip_get(ip, "quantization")
    assert isinstance(quantization, dict)
    assert quantization["method"] == "12 bit"
    assert int(ip_get(ip, "quantization nbits")) == 12
    assert int(ip_get(ip, "nbits")) == 12
    assert int(ip_get(ip, "bits")) == 12
    assert ip_get(ip, "quantization method") == "12 bit"
    assert float(ip_get(ip, "max digital value")) == 4096.0
    assert np.allclose(np.asarray(ip_get(ip, "data intensities dv"), dtype=float), linear * 4096.0)
    assert np.allclose(np.asarray(ip_get(ip, "data intensities digital values"), dtype=float), linear * 4096.0)

    empty_ip = ip_create()
    assert float(ip_get(empty_ip, "max digital value")) == 1.0


def test_ip_get_transform_gamma_and_sensor_space_aliases_match_upstream() -> None:
    ip = ip_create()
    transform_a = 2.0 * np.eye(3, dtype=float)
    transform_b = 3.0 * np.eye(3, dtype=float)
    transform_c = 4.0 * np.eye(3, dtype=float)
    sensor_space = np.arange(1, 13, dtype=float).reshape(2, 2, 3)

    ip = ip_set(ip, "transforms", [transform_a, transform_b, transform_c])
    ip = ip_set(ip, "gamma", 2.2)
    ip = ip_set(ip, "sensorspace", sensor_space)

    transform_list = ip_get(ip, "transform list")
    assert len(transform_list) == 3
    assert np.allclose(np.asarray(transform_list[0], dtype=float), transform_a)
    assert np.allclose(np.asarray(transform_list[1], dtype=float), transform_b)
    assert np.allclose(np.asarray(transform_list[2], dtype=float), transform_c)
    each_transform = ip_get(ip, "each transform")
    assert len(each_transform) == 3
    assert np.allclose(np.asarray(each_transform[2], dtype=float), transform_c)

    assert float(ip_get(ip, "gamma")) == 2.2
    assert float(ip_get(ip, "render gamma")) == 2.2

    for alias in ("data sensor", "sensor data", "sensor channels", "sensor space", "demosaic sensor"):
        assert np.allclose(np.asarray(ip_get(ip, alias), dtype=float), sensor_space)
    for alias in ("n input filters", "number sensor channels", "n sensor inputs", "n sensor channels"):
        assert int(ip_get(ip, alias)) == 3


def test_ip_set_renderwhitept_replays_legacy_sensor_transform_update() -> None:
    ip = ip_create()
    base_transform = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.0, 1.5, 0.1],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    sensor_qe = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.8],
        ],
        dtype=float,
    )
    illuminant = np.array([2.0, 1.0], dtype=float)

    ip = ip_set(ip, "sensor conversion matrix", base_transform)
    updated = ip_set(ip, "render whitept", illuminant, sensor_qe)

    sensor_light = (illuminant.reshape(1, -1) @ sensor_qe).reshape(-1)
    sensor_light /= np.max(sensor_light)
    expected_transform = base_transform @ np.diag(1.0 / (sensor_light @ base_transform))

    assert bool(ip_get(updated, "render whitept")) is True
    assert str(ip_get(updated, "transform method")) == "current"
    assert np.allclose(np.asarray(ip_get(updated, "sensor conversion matrix"), dtype=float), expected_transform)

    disabled = ip_set(updated, "render whitept", False)
    assert bool(ip_get(disabled, "render whitept")) is False


def test_ip_get_geometry_helpers_match_upstream_grids() -> None:
    ip = ip_create()
    linear = np.zeros((3, 5, 3), dtype=float)
    ip = ip_set(ip, "result", linear)

    center = np.asarray(ip_get(ip, "center"), dtype=float)
    assert np.allclose(center, np.array([2.0, 3.0], dtype=float))

    image_grid = ip_get(ip, "image grid")
    assert isinstance(image_grid, list)
    assert len(image_grid) == 2
    expected_x = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    expected_y = np.array(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    assert np.allclose(np.asarray(image_grid[0], dtype=float), expected_x)
    assert np.allclose(np.asarray(image_grid[1], dtype=float), expected_y)
    assert np.allclose(
        np.asarray(ip_get(ip, "distance2center"), dtype=float),
        np.sqrt(expected_x**2 + expected_y**2),
    )
    assert np.allclose(
        np.asarray(ip_get(ip, "angle"), dtype=float),
        np.arctan2(expected_y, expected_x),
    )


def test_ip_get_set_combination_method_matches_upstream_default() -> None:
    ip = ip_create()
    assert str(ip_get(ip, "combination method")) == "longest"
    assert str(ip_get(ip, "combine exposures")) == "longest"

    updated = ip_set(ip, "combination method", "sum")
    assert str(ip_get(updated, "combination method")) == "sum"
    assert str(ip_get(updated, "combine exposures")) == "sum"
    assert str(updated.fields["combination_method"]) == "sum"
    assert str(updated.fields["combine_exposures"]) == "sum"


def test_vcimage_srgb_matches_manual_pipeline(asset_store) -> None:
    generated = vcimageSRGB(asset_store=asset_store)

    scene = scene_create("macbethD65", asset_store=asset_store)
    oi = oi_compute(oi_create(asset_store=asset_store), scene)
    sensor = sensor_create(asset_store=asset_store)
    sensor = sensor_set(sensor, "size", [256, 256])
    sensor = sensor_set(sensor, "pixel size", np.array([3.0e-6, 3.0e-6], dtype=float))
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor = sensor_set(sensor, "color filters", ie_read_spectra("XYZ", wave, asset_store=asset_store))
    sensor = sensor_set(sensor, "filter names", ["x", "y", "z"])
    sensor = sensor_compute(sensor, oi)
    manual = ip_create(asset_store=asset_store)
    manual = ip_set(manual, "demosaicMethod", "Adaptive Laplacian")
    manual = ip_set(manual, "colorBalanceMethod", "Gray World")
    manual = ip_set(manual, "internalCS", "XYZ")
    manual = ip_set(manual, "colorconversionmethod", "MCC Optimized")
    manual = ip_compute(manual, sensor, asset_store=asset_store)

    assert tuple(np.asarray(ip_get(generated, "result"), dtype=float).shape) == (256, 256, 3)
    assert np.allclose(np.asarray(ip_get(generated, "result"), dtype=float), np.asarray(ip_get(manual, "result"), dtype=float))
    assert np.allclose(np.asarray(ip_get(generated, "srgb"), dtype=float), np.asarray(ip_get(manual, "srgb"), dtype=float))
    assert np.allclose(
        np.asarray(ip_get(generated, "illuminant correction transform"), dtype=float),
        np.asarray(ip_get(manual, "illuminant correction transform"), dtype=float),
    )
    assert np.allclose(
        np.asarray(ip_get(generated, "sensor conversion matrix"), dtype=float),
        np.asarray(ip_get(manual, "sensor conversion matrix"), dtype=float),
    )


def test_ie_radiance_to_ip_scene_matches_manual_pipeline(asset_store) -> None:
    scene = scene_create("macbethD65", asset_store=asset_store)

    generated_ip, generated_sensor = ieRadiance2IP(scene, asset_store=asset_store)
    assert generated_ip is not None

    manual_oi = oi_compute(oi_create("pinhole", scene_get(scene, "wave"), asset_store=asset_store), scene)
    pixel_size_um = float(oi_get(manual_oi, "width spatial resolution")) * 1e6
    well_capacity, _ = iePixelWellCapacity(pixel_size_um, asset_store=asset_store)

    manual_sensor = sensor_create(asset_store=asset_store)
    manual_sensor = sensor_set(manual_sensor, "pixel read noise volts", 2.0e-3)
    manual_sensor = sensor_set(manual_sensor, "pixel voltage swing", 1.0)
    manual_sensor = sensor_set(manual_sensor, "pixel dark voltage", 2.0e-3)
    manual_sensor = sensor_set(manual_sensor, "pixel conversion gain", 1.0 / float(well_capacity))
    manual_sensor = sensor_set(manual_sensor, "quantization method", "12 bit")
    manual_sensor = sensor_set(manual_sensor, "analog gain", 1.0)
    manual_sensor = sensor_set(manual_sensor, "pixel size same fill factor", pixel_size_um * 1e-6)
    manual_sensor = sensor_set(manual_sensor, "match oi", manual_oi)
    manual_sensor = sensor_set(manual_sensor, "auto exposure", True)
    manual_sensor = sensor_set(manual_sensor, "noise flag", 2)
    manual_sensor = sensor_compute(manual_sensor, manual_oi)

    manual_ip = ip_create(sensor=manual_sensor, asset_store=asset_store)
    manual_ip = ip_set(manual_ip, "conversion method sensor", "MCC Optimized")
    manual_ip = ip_set(manual_ip, "illuminant correction method", "gray world")
    manual_ip = ip_set(manual_ip, "demosaic method", "Adaptive Laplacian")
    manual_ip = ip_compute(manual_ip, manual_sensor, asset_store=asset_store)
    manual_ip.metadata["eT"] = float(sensor_get(manual_sensor, "integration time"))

    assert np.isclose(float(sensor_get(generated_sensor, "integration time")), float(sensor_get(manual_sensor, "integration time")))
    assert np.allclose(np.asarray(ip_get(generated_ip, "result"), dtype=float), np.asarray(ip_get(manual_ip, "result"), dtype=float))
    assert np.allclose(np.asarray(ip_get(generated_ip, "srgb"), dtype=float), np.asarray(ip_get(manual_ip, "srgb"), dtype=float))
    assert generated_ip.fields["demosaic_method"] == "adaptive laplacian"
    assert generated_ip.fields["illuminant_correction_method"] == "gray world"
    assert generated_ip.fields["conversion_method_sensor"] == "MCC Optimized"
    assert np.isclose(float(generated_ip.metadata["eT"]), float(sensor_get(generated_sensor, "integration time")))


def test_ie_radiance_to_ip_reuses_sensor_and_copies_metadata(asset_store) -> None:
    scene = scene_create("macbethD65", asset_store=asset_store)
    oi = oi_compute(oi_create("pinhole", scene_get(scene, "wave"), asset_store=asset_store), scene)
    provided_sensor = sensor_create(asset_store=asset_store)
    provided_sensor.metadata["source"] = "provided"

    generated_ip, generated_sensor = ie_radiance_to_ip(
        oi,
        "sensor",
        provided_sensor,
        "etime",
        0.01,
        "noise flag",
        0,
        asset_store=asset_store,
    )

    assert generated_ip is not None
    assert generated_sensor is not provided_sensor
    assert generated_sensor.metadata["source"] == "provided"
    assert np.isclose(float(sensor_get(generated_sensor, "integration time")), 0.01)
    assert generated_ip.metadata["source"] == "provided"
    assert np.isclose(float(generated_ip.metadata["eT"]), 0.01)
    assert generated_ip.fields["demosaic_method"] == "adaptive laplacian"


def test_vcimage_iso_mtf_matches_camera_mtf_vci(asset_store) -> None:
    camera = camera_create(asset_store=asset_store)

    generated = vcimageISOMTF(camera, asset_store=asset_store)
    reference = camera_mtf(camera, asset_store=asset_store).vci

    assert ip_get(generated, "name") == "iso12233"
    assert tuple(np.asarray(ip_get(generated, "size"), dtype=int)) == tuple(
        np.asarray(ip_get(reference, "size"), dtype=int)
    )
    assert np.allclose(
        np.asarray(ip_get(generated, "result"), dtype=float),
        np.asarray(ip_get(reference, "result"), dtype=float),
    )
    assert np.allclose(
        np.asarray(ip_get(generated, "srgb"), dtype=float),
        np.asarray(ip_get(reference, "srgb"), dtype=float),
    )


def test_vcimage_vsnr_replays_same_score_with_returned_rect(asset_store) -> None:
    ip = vcimageSRGB(asset_store=asset_store)

    vsnr, rect = vcimageVSNR(ip, asset_store=asset_store)
    replay_vsnr, replay_rect = vcimageVSNR(ip, rect=rect, asset_store=asset_store)

    assert rect.shape == (4,)
    assert rect[2] > 0
    assert rect[3] > 0
    assert np.isfinite(vsnr)
    assert vsnr > 0.0
    assert np.array_equal(replay_rect, rect)
    assert np.isclose(replay_vsnr, vsnr, rtol=1.0e-12, atol=1.0e-12)


def test_ip_mcc_xyz_matches_manual_srgb_and_custom_methods(asset_store) -> None:
    ip = vcimageSRGB(asset_store=asset_store)
    corners = _whole_chart_corner_points(*np.asarray(ip_get(ip, "size"), dtype=int)[:2])
    _, m_locs, p_size = _chart_rectangles(corners, 4, 6, 0.3)
    rgb_data = np.asarray(
        _chart_rects_data(ip, m_locs, float(np.asarray(p_size, dtype=float).reshape(-1)[0]), full_data=False, data_type="result"),
        dtype=float,
    )
    rgb_image = xw_to_rgb_format(rgb_data, 4, 6)
    expected_srgb, _, _, _ = rgb_to_xw_format(np.asarray(_linear_srgb_to_xyz(rgb_image), dtype=float))
    expected_custom, _, _, _ = rgb_to_xw_format(
        np.asarray(imageRGB2XYZ(ip, rgb_image, asset_store=asset_store), dtype=float)
    )

    srgb_xyz, srgb_white, returned_corners = ipMCCXYZ(ip, "whole chart", asset_store=asset_store)
    custom_xyz, custom_white, custom_corners = ipMCCXYZ(
        ip, returned_corners, "custom", asset_store=asset_store
    )
    vc_xyz, vc_white, vc_corners = vcimageMCCXYZ(ip, returned_corners, "custom", asset_store=asset_store)

    assert np.array_equal(returned_corners, corners)
    assert np.array_equal(custom_corners, corners)
    assert np.array_equal(vc_corners, corners)
    assert srgb_xyz.shape == (24, 3)
    assert custom_xyz.shape == (24, 3)
    assert np.allclose(srgb_xyz, expected_srgb)
    assert np.allclose(custom_xyz, expected_custom)
    assert np.allclose(vc_xyz, custom_xyz)
    assert np.allclose(srgb_white, srgb_xyz[3, :])
    assert np.allclose(custom_white, custom_xyz[3, :])
    assert np.allclose(vc_white, custom_white)


def test_ip_save_image_writes_png_and_appends_extension(tmp_path, asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    sensor = sensor_set(sensor, "volts", np.arange(1, 37, dtype=float).reshape(6, 6))
    computed = ip_compute(ip_create(asset_store=asset_store), sensor, asset_store=asset_store)

    output = ipSaveImage(computed, tmp_path / "ip-save-image")

    saved_path = tmp_path / "ip-save-image.png"
    assert output == str(saved_path.resolve())
    saved = iio.imread(saved_path)
    srgb = np.asarray(ip_get(computed, "srgb"), dtype=float)
    assert saved.shape == srgb.shape
    assert saved.dtype == np.uint8


def test_ip_save_image_cropborder_crops_black_frame(tmp_path, asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    srgb = np.zeros((6, 7, 3), dtype=float)
    srgb[1:5, 2:6, :] = np.array([0.25, 0.5, 0.75], dtype=float)
    ip.data["srgb"] = srgb

    output = ipSaveImage(ip, tmp_path / "cropped.png", cropborder=True)

    saved = iio.imread(output)
    assert saved.shape == (4, 4, 3)
    assert np.all(saved[0, 0] == np.array([64, 128, 191], dtype=np.uint8))


def test_image_mcc_transform_matches_sensor_to_target_matrix(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )

    transform = imageMCCTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)
    expected = sensor_to_target_matrix(
        wave,
        sensor_qe,
        target_space="xyz",
        illuminant="D65",
        surfaces="mcc",
        asset_store=asset_store,
    )

    assert np.allclose(transform, expected)


def test_image_mcc_transform_accepts_illuminant_vector(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )
    illuminant_energy = np.asarray(
        asset_store.load_illuminant("D65", wave_nm=wave)[1],
        dtype=float,
    )

    from_string = imageMCCTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)
    from_vector = imageMCCTransform(
        sensor_qe,
        target_qe,
        illuminant_energy,
        wave,
        asset_store=asset_store,
    )

    assert np.allclose(from_vector, from_string)


def test_image_sensor_transform_supports_multisurface_default(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )
    illuminant_energy = np.asarray(
        asset_store.load_illuminant("D65", wave_nm=wave)[1],
        dtype=float,
    )
    illuminant_quanta = np.asarray(energy_to_quanta(illuminant_energy, wave), dtype=float)
    reflectances = np.asarray(
        ie_reflectance_samples(None, None, wave, asset_store=asset_store)[0],
        dtype=float,
    )
    weighted_surfaces = reflectances * illuminant_quanta.reshape(-1, 1)
    sensor_response = weighted_surfaces.T @ sensor_qe
    target_response = weighted_surfaces.T @ target_qe
    expected, _, _, _ = np.linalg.lstsq(sensor_response, target_response, rcond=None)

    transform = imageSensorTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)

    assert np.allclose(transform, expected)


def test_ie_color_transform_matches_xyz_sensor_transform(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )

    expected = imageSensorTransform(
        sensor_qe,
        target_qe,
        "D65",
        wave,
        "multisurface",
        asset_store=asset_store,
    )
    generated = ieColorTransform(sensor, "XYZ", "D65", "multisurface", asset_store=asset_store)

    assert np.allclose(generated, expected)


def test_ie_color_transform_supports_stockman_and_sensor_identity(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    stockman_qe = np.asarray(
        ie_read_spectra("data/human/stockmanQuanta.mat", wave, asset_store=asset_store),
        dtype=float,
    )

    expected = imageSensorTransform(
        sensor_qe,
        stockman_qe,
        "D65",
        wave,
        "mcc",
        asset_store=asset_store,
    )
    generated = ieColorTransform(sensor, "Stockman", "D65", "mcc", asset_store=asset_store)
    identity = ieColorTransform(sensor, "sensor", asset_store=asset_store)

    assert np.allclose(generated, expected)
    assert np.array_equal(identity, np.eye(sensor_qe.shape[1], dtype=float))


def test_image_esser_transform_matches_direct_sensor_transform(asset_store) -> None:
    sensor = sensor_create("default", asset_store=asset_store)
    wave = np.asarray(sensor_get(sensor, "wave"), dtype=float)
    sensor_qe = np.asarray(sensor_get(sensor, "spectral qe"), dtype=float)
    target_qe = np.asarray(
        xyz_color_matching(wave, quanta=True, asset_store=asset_store),
        dtype=float,
    )

    expected = sensor_to_target_matrix(
        wave,
        sensor_qe,
        target_space="xyz",
        illuminant="D65",
        surfaces="esser",
        asset_store=asset_store,
    )
    direct = imageSensorTransform(
        sensor_qe,
        target_qe,
        "D65",
        wave,
        "esser",
        asset_store=asset_store,
    )
    esser = imageEsserTransform(sensor_qe, target_qe, "D65", wave, asset_store=asset_store)

    assert np.allclose(direct, expected)
    assert np.allclose(esser, expected)


def test_ie_internal_to_display_matches_direct_display_matrix(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)

    transform = ieInternal2Display(ip)
    internal_cmf = np.asarray(ip_get(ip, "internal cmf"), dtype=float)
    display_spd = np.asarray(ip_get(ip, "display rgb spd"), dtype=float)
    expected = np.linalg.inv(display_spd.T @ internal_cmf)

    assert np.allclose(transform, expected)


def test_ip_hdr_white_whitens_saturated_pixels_without_blur(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip.data["input"] = np.full((3, 3), 10.0, dtype=float)
    ip.data["result"] = np.full((3, 3, 3), 0.25, dtype=float)

    whitened_ip, weights = ipHDRWhite(
        ip,
        "saturation",
        10.0,
        "hdr level",
        0.5,
        "wgt blur",
        0.0,
    )

    assert np.allclose(weights, np.ones((3, 3), dtype=float))
    assert np.allclose(np.asarray(ip_get(whitened_ip, "result"), dtype=float), 1.0)


def test_ip_hdr_white_uses_input_max_when_saturation_omitted(asset_store) -> None:
    ip = ip_create(asset_store=asset_store)
    ip.data["input"] = np.array(
        [
            [0.0, 2.5, 5.0],
            [1.0, 5.0, 2.5],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    ip.data["result"] = np.zeros((3, 3, 3), dtype=float)

    whitened_ip, weights = ipHDRWhite(ip, "hdr level", 0.5, "wgt blur", 0.0)

    expected_weights = np.clip(ip.data["input"] / 5.0 - 0.5, 0.0, 1.0) / 0.5
    assert np.allclose(weights, expected_weights)
    assert np.allclose(np.asarray(ip_get(whitened_ip, "result"), dtype=float), expected_weights[:, :, np.newaxis])


def test_image_distort_gaussian_noise_zero_preserves_uint8_image() -> None:
    image = np.array(
        [[[0, 32, 64], [96, 128, 160]], [[192, 224, 255], [16, 48, 80]]],
        dtype=np.uint8,
    )

    distorted = imageDistort(image, "gaussian noise", 0)

    assert distorted.dtype == np.uint8
    assert np.array_equal(distorted, image)


def test_image_distort_scale_contrast_scales_float_image() -> None:
    image = np.array([[[0.25, 0.5, 0.75]]], dtype=float)

    distorted = imageDistort(image, "scale contrast", 0.2)

    assert np.allclose(distorted, image * 1.2)


def test_image_distort_jpeg_compress_roundtrip_constant_image() -> None:
    image = np.full((8, 8, 3), 128, dtype=np.uint8)

    distorted = imageDistort(image, "jpeg compress", 90)

    assert distorted.shape == image.shape
    assert distorted.dtype == np.uint8
    assert np.max(np.abs(distorted.astype(int) - image.astype(int))) <= 1
