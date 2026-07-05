from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat, savemat

from pyisetcam import (
    displayCompute,
    display_compute,
    displayConvert,
    displayPlot,
    displayPT2ISET,
    displayReflectance,
    display_create,
    display_get,
    display_set,
    render_lcd_samsung_rgbw,
    render_oled_samsung,
)
import pyisetcam.display as display_module


def test_display_create_lcd_example(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    assert display_get(display, "spd").shape[1] == 3
    assert display_get(display, "gamma").shape[1] == 3


def test_display_create_empty_name_placeholder_uses_default_constructor(asset_store) -> None:
    default_display = display_create(asset_store=asset_store)
    placeholder_display = display_create([], asset_store=asset_store)

    assert placeholder_display.name == default_display.name
    np.testing.assert_allclose(display_get(placeholder_display, "wave"), display_get(default_display, "wave"))
    np.testing.assert_allclose(display_get(placeholder_display, "spd"), display_get(default_display, "spd"))
    np.testing.assert_allclose(display_get(placeholder_display, "gamma"), display_get(default_display, "gamma"))


def test_display_module_matlab_aliases() -> None:
    assert display_module.displayConvert is display_module.display_convert
    assert display_module.displayPT2ISET is display_module.display_pt2iset
    assert display_module.displayReflectance is display_module.display_reflectance
    assert display_module.displayMaxContrast is display_module.display_max_contrast
    assert display_module.ieCalculateMonitorDPI is display_module.ie_calculate_monitor_dpi
    assert display_module.ieLUTDigital is display_module.ie_lut_digital
    assert display_module.mperdot2dpi is display_module.mperdot2dpi
    assert display_module.displayCreate is display_module.display_create
    assert display_module.displayList is display_module.display_list
    assert display_module.displayDescription is display_module.display_description
    assert display_module.displayShowImage is display_module.display_show_image
    assert display_module.displayPlot is display_module.display_plot
    assert display_module.displayCompute is display_module.display_compute
    assert display_module.displaySetMaxLuminance is display_module.display_set_max_luminance
    assert display_module.displaySetWhitePoint is display_module.display_set_white_point
    assert display_module.ieReadSpectra is display_module.ie_read_spectra
    assert display_module.ieUnitScaleFactor is display_module.ie_unit_scale_factor
    assert display_module.srgb2xyz is display_module.srgb_to_xyz
    assert display_module.displayGet is display_module.display_get
    assert display_module.displaySet is display_module.display_set
    assert display_module.xyz2lms is display_module.xyz_to_lms
    assert display_module.xyz2srgb is display_module.xyz_to_srgb


def test_display_wave_resample(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0)
    display = display_create("lcdExample.mat", wave, asset_store=asset_store)
    assert np.array_equal(display_get(display, "wave"), wave)
    assert display_get(display, "spd").shape == (wave.size, 3)


def test_display_get_reports_matlab_style_derived_values(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    display = display_set(display, "ambient spd", np.full(display_get(display, "n wave"), 0.01, dtype=float))

    inverse_gamma = display_get(display, "inverse gamma", 32)
    gamma = display_get(display, "gamma")
    dark_level = np.asarray(display_get(display, "dark level"), dtype=float)
    rgb2lms = np.asarray(display_get(display, "rgb2lms"), dtype=float)
    rgb2xyz = np.asarray(display_get(display, "rgb2xyz"), dtype=float)
    lrgb2xyz = np.asarray(display_get(display, "lrgb2xyz"), dtype=float)
    digital_rgb = np.array([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]], dtype=float)
    digital_xyz = np.asarray(display_get(display, "drgb2xyz", digital_rgb), dtype=float)
    white_lms = np.asarray(display_get(display, "white lms"), dtype=float)
    primaries_xyz = np.asarray(display_get(display, "primaries xyz"), dtype=float)
    primaries_srgb = np.asarray(display_get(display, "primaries srgb"), dtype=float)
    primaries_xy = np.asarray(display_get(display, "primaries xy"), dtype=float)
    black_xyz = np.asarray(display_get(display, "black xyz"), dtype=float)
    black_radiance = np.asarray(display_get(display, "black radiance"), dtype=float)
    dark_luminance = float(display_get(display, "dark luminance"))
    peak_luminance = float(display_get(display, "peak luminance"))
    peak_contrast = float(display_get(display, "peak contrast"))
    expected_bits = int(round(np.log2(gamma.shape[0])))

    assert inverse_gamma.shape == (32, 3)
    assert display_get(display, "is emissive") is True
    assert display_get(display, "bits") == expected_bits
    assert display_get(display, "n levels") == 2**expected_bits
    assert np.array_equal(display_get(display, "levels")[:4], np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(dark_level, gamma[0, :], rtol=1e-12, atol=1e-12)
    assert display_get(display, "n primaries") == 3
    assert display_get(display, "white spd").shape == (display_get(display, "n wave"),)
    assert display_get(display, "black spd").shape == (display_get(display, "n wave"),)
    np.testing.assert_allclose(lrgb2xyz, rgb2xyz, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        digital_xyz,
        gamma[digital_rgb.astype(int)[0, :, :], np.arange(3)].reshape(1, 2, 3) @ rgb2xyz,
        rtol=1e-12,
        atol=1e-12,
    )
    assert rgb2lms.shape == (3, 3)
    np.testing.assert_allclose(white_lms, np.sum(rgb2lms, axis=0), rtol=1e-12, atol=1e-12)
    assert primaries_xyz.shape == (3, 3)
    assert primaries_srgb.shape == (3, 3)
    assert np.all(primaries_srgb >= 0.0)
    assert np.all(primaries_srgb <= 1.0)
    assert primaries_xy.shape == (3, 2)
    np.testing.assert_allclose(
        primaries_xy,
        primaries_xyz[:, :2] / np.sum(primaries_xyz, axis=1, keepdims=True),
        rtol=1e-12,
        atol=1e-12,
    )
    assert black_xyz.shape == (3,)
    np.testing.assert_allclose(black_radiance, display_get(display, "black spd"), rtol=1e-12, atol=1e-12)
    assert np.isclose(dark_luminance, black_xyz[1], rtol=1e-12, atol=1e-12)
    assert np.isclose(peak_contrast, peak_luminance / dark_luminance, rtol=1e-12, atol=1e-12)
    assert display_get(display, "meters per dot") > 0.0
    assert display_get(display, "dots per deg") > 0.0
    assert np.isclose(display_get(display, "samp per deg"), display_get(display, "dots per deg"), rtol=1e-12, atol=1e-12)
    assert np.isclose(display_get(display, "distance"), display_get(display, "viewing distance"), rtol=1e-12, atol=1e-12)


def test_display_set_resamples_ambient_and_tracks_dixel_metadata(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    ambient = np.linspace(0.0, 1.0, display_get(display, "n wave"))
    intensity = np.ones((2, 3, 3), dtype=float)
    control = np.zeros((2, 3, 3), dtype=float)

    display = display_set(display, "ambient spd", ambient)
    display = display_set(display, "size", [0.30, 0.20])
    display = display_set(display, "pixels per dixel", [1, 1])
    display = display_set(display, "dixel image", intensity)
    display = display_set(display, "dixel control map", control)
    display = display_set(display, "wave", np.arange(420.0, 681.0, 20.0))

    assert display_get(display, "ambient spd").shape == display_get(display, "wave").shape
    assert np.allclose(display_get(display, "size"), np.array([0.30, 0.20]))
    assert display_get(display, "pixels per dixel") == [1, 1]
    assert display_get(display, "dixel size") == (2, 3)
    assert np.array_equal(display_get(display, "dixel control map"), control)
    np.testing.assert_allclose(display_get(display, "oversample"), np.array([2.0, 3.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        display_get(display, "sample spacing"),
        np.array(
            [
                float(display_get(display, "meters per dot")) / 2.0,
                float(display_get(display, "meters per dot")) / 3.0,
            ],
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        display_get(display, "sample spacing", "mm"),
        np.array(
            [
                float(display_get(display, "meters per dot", "mm")) / 2.0,
                float(display_get(display, "meters per dot", "mm")) / 3.0,
            ],
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(display_get(display, "fill factor"), np.ones(3, dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        display_get(display, "subpixel spd"),
        display_get(display, "spd"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_display_create_normalizes_render_function_names(asset_store) -> None:
    oled = display_create("OLED-Samsung.mat", asset_store=asset_store)
    rgbw = display_create("LCD-Samsung-RGBW.mat", asset_store=asset_store)
    barco = display_create("LED-BarcoC8.mat", asset_store=asset_store)

    assert display_get(oled, "render function") == "render_oled_samsung"
    assert display_get(rgbw, "render function") == "render_lcd_samsung_rgbw"
    assert display_get(barco, "render function") is None


def test_display_plot_returns_headless_payloads(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    dixel_image = np.array(
        [
            [[1.0, 0.5, 0.25], [0.25, 1.0, 0.5]],
            [[0.5, 0.25, 1.0], [1.0, 0.5, 0.25]],
        ],
        dtype=float,
    )
    display = display_set(display, "dixel image", dixel_image)
    display = display_set(display, "pixels per dixel", [1, 1])

    spd_payload, spd_handle = displayPlot(display, "spd")
    gamma_payload, gamma_handle = displayPlot(display, "gamma table")
    gamut_payload, gamut_handle = displayPlot(display, "gamut")
    psf_payload, psf_handle = displayPlot(display, "psf")

    assert spd_handle is None
    assert gamma_handle is None
    assert gamut_handle is None
    assert psf_handle is None
    np.testing.assert_allclose(spd_payload["wave"], display_get(display, "wave"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(spd_payload["spd"], display_get(display, "spd primaries"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gamma_payload, display_get(display, "gamma table"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gamut_payload["xy"][0, :], display_get(display, "primaries xy")[0, :], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gamut_payload["xy"][-1, :], gamut_payload["xy"][0, :], rtol=1e-12, atol=1e-12)
    assert psf_payload["psf"].shape == dixel_image.shape
    np.testing.assert_allclose(np.max(psf_payload["psf"], axis=(0, 1)), np.ones(3, dtype=float), rtol=1e-12, atol=1e-12)
    assert psf_payload["x"].shape == (2,)
    assert psf_payload["y"].shape == (2,)
    assert psf_payload["srgb"].shape == (3, 3)


def test_display_plot_gamut3d_returns_lab_cloud_and_hull(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)

    gamut_payload, gamut_handle = displayPlot(display, "gamut 3d")

    assert gamut_handle is None
    assert gamut_payload["LAB"].shape == (30**3, 3)
    assert gamut_payload["rgb"].shape == (30**3, 3)
    assert gamut_payload["hull"].ndim == 2
    assert gamut_payload["hull"].shape[1] == 3
    assert gamut_payload["hull"].shape[0] > 0
    assert np.all(gamut_payload["rgb"] >= 0.0)
    assert np.all(gamut_payload["rgb"] <= 1.0)


def test_display_compute_matches_nearest_neighbor_and_dixel_weighting(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    dixel_image = np.array(
        [
            [[1.0, 0.5, 0.25], [0.25, 1.0, 0.5]],
            [[0.5, 0.25, 1.0], [1.0, 0.5, 0.25]],
        ],
        dtype=float,
    )
    control = np.ones((2, 2, 3), dtype=float)
    image = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=float)

    display = display_set(display, "pixels per dixel", [1, 1])
    display = display_set(display, "dixel image", dixel_image)
    display = display_set(display, "dixel control map", control)

    resized_image = np.asarray(display_get(display, "dixel image", [4, 4]), dtype=float)
    resized_control = np.asarray(display_get(display, "dixel control map", [4, 4]), dtype=float)

    actual, returned = display_compute(display, image)
    alias_actual, alias_returned = displayCompute(display, image, [2, 2])

    repeated = np.repeat(np.repeat(np.repeat(image[:, :, None], 3, axis=2), 2, axis=0), 2, axis=1)
    expected = repeated * np.tile(dixel_image, (2, 2, 1))

    assert returned is display
    assert alias_returned is display
    assert resized_image.shape == (4, 4, 3)
    assert resized_control.shape == (4, 4, 3)
    np.testing.assert_allclose(np.sum(resized_image, axis=(0, 1)), np.full(3, 16.0, dtype=float), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(alias_actual, expected, rtol=1e-12, atol=1e-12)


def test_render_oled_samsung_matches_control_map_replay(asset_store) -> None:
    display = display_create("OLED-Samsung.mat", asset_store=asset_store)
    image = np.arange(1.0, 1.0 + 2 * 4 * 4, dtype=float).reshape(2, 4, 4, order="F")
    control_map = np.asarray(display_get(display, "dixel control map"), dtype=int)
    pixels_per_dixel = np.asarray(display_get(display, "pixels per dixel"), dtype=int).reshape(2)

    actual = render_oled_samsung(image, display)

    tile_rows, tile_cols = control_map.shape[:2]
    expected = np.zeros((tile_rows, tile_cols * 2, 4), dtype=float)
    for primary in range(image.shape[2]):
        control = control_map[:, :, primary] - 1
        for block_index, col in enumerate(range(0, image.shape[1], pixels_per_dixel[1])):
            block = image[:, col : col + pixels_per_dixel[1], primary]
            expected[:, block_index * tile_cols : (block_index + 1) * tile_cols, primary] = block.reshape(-1, order="F")[control]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_display_compute_uses_render_function_when_available(asset_store) -> None:
    display = display_create("OLED-Samsung.mat", asset_store=asset_store)
    image = np.arange(1.0, 1.0 + 2 * 4 * 4, dtype=float).reshape(2, 4, 4, order="F")

    actual, returned = display_compute(display, image)

    dixel_image = np.asarray(display_get(display, "dixel image"), dtype=float)
    pixels_per_dixel = np.asarray(display_get(display, "pixels per dixel"), dtype=int).reshape(2)
    rendered = render_oled_samsung(image, display, display_get(display, "dixel size"), asset_store=asset_store)
    expected = rendered * np.tile(
        dixel_image,
        (image.shape[0] // pixels_per_dixel[0], image.shape[1] // pixels_per_dixel[1], 1),
    )

    assert returned is display
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_render_lcd_samsung_rgbw_matches_white_extraction(asset_store) -> None:
    display = display_create("LCD-Samsung-RGBW.mat", asset_store=asset_store)
    image = np.array(
        [[[0.4, 0.1, 0.7, 0.9], [0.8, 0.3, 0.5, 0.2]]],
        dtype=float,
    )
    control_map = np.asarray(display_get(display, "dixel control map"), dtype=float)

    actual = render_lcd_samsung_rgbw(image, display)

    tile_rows, tile_cols = control_map.shape[:2]
    expected = np.zeros((tile_rows, tile_cols * image.shape[1], image.shape[2]), dtype=float)
    for col in range(image.shape[1]):
        rgb_levels = image[0, col, :3]
        white_level = float(np.min(rgb_levels))
        for primary in range(image.shape[2]):
            tile = white_level * control_map[:, :, primary]
            if primary < 3:
                tile = (rgb_levels[primary] - white_level) * control_map[:, :, primary]
            expected[:, col * tile_cols : (col + 1) * tile_cols, primary] = tile

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_display_conversion_wrappers_match_legacy_contracts(asset_store, tmp_path) -> None:
    ct_display = SimpleNamespace(
        m_strDisplayName="ct-example",
        sViewingContext=SimpleNamespace(m_fViewingDistance=0.7),
        sPhysicalDisplay=SimpleNamespace(
            m_fVerticalRefreshRate=120.0,
            m_objCDixelStructure=SimpleNamespace(
                m_aWaveLengthSamples=np.array([400.0, 500.0, 600.0], dtype=float),
                m_aSpectrumOfPrimaries=np.eye(3, dtype=float),
                m_cellGammaStructure=np.array(
                    [
                        SimpleNamespace(vGammaRampLUT=np.linspace(0.0, 1.0, 4, dtype=float))
                        for _ in range(3)
                    ],
                    dtype=object,
                ),
                m_cellPSFStructure=np.array(
                    [
                        SimpleNamespace(sCustomData=SimpleNamespace(aRawData=np.full((20, 20), idx + 1.0, dtype=float)))
                        for idx in range(3)
                    ],
                    dtype=object,
                ),
                m_fPixelSizeInMmX=0.254,
            ),
        ),
    )

    converted_path = tmp_path / "converted_display.mat"
    converted = displayConvert(
        ct_display,
        np.array([400.0, 450.0, 500.0, 550.0, 600.0], dtype=float),
        str(converted_path),
        True,
        "converted-display",
    )

    assert display_get(converted, "name") == "converted-display"
    assert display_get(converted, "wave").shape == (5,)
    assert display_get(converted, "spd").shape == (5, 3)
    assert display_get(converted, "gamma").shape == (4, 3)
    assert np.isclose(display_get(converted, "dpi"), 100.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(display_get(converted, "dist"), 0.7, atol=1e-12, rtol=1e-12)
    assert np.isclose(display_get(converted, "refresh rate"), 120.0, atol=1e-12, rtol=1e-12)
    psfs = np.asarray(display_get(converted, "psfs"), dtype=float)
    assert psfs.shape == (20, 20, 3)
    np.testing.assert_allclose(np.sum(psfs, axis=(0, 1)), np.ones(3, dtype=float), rtol=1e-12, atol=1e-12)
    assert converted_path.exists()
    reloaded = loadmat(converted_path, squeeze_me=True, struct_as_record=False)["d"]
    assert reloaded.name == "converted-display"

    pt_path = tmp_path / "pt_display.mat"
    savemat(
        pt_path,
        {
            "cals": np.array(
                [
                    {
                        "S_device": np.array([400.0, 10.0, 3.0], dtype=float),
                        "P_device": np.array(
                            [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=float,
                        ),
                        "gammaTable": np.linspace(0.0, 1.0, 12, dtype=float).reshape(4, 3),
                        "describe": {"dacsize": 2},
                    }
                ],
                dtype=object,
            )
        },
    )

    pt_display = displayPT2ISET(str(pt_path), np.array([400.0, 410.0, 420.0], dtype=float))
    assert display_get(pt_display, "wave").shape == (3,)
    assert display_get(pt_display, "spd").shape == (3, 3)
    assert display_get(pt_display, "gamma").shape == (4, 3)
    assert display_get(pt_display, "dacsize") == 2

    reflectance_display, rgb_primaries, ill_energy = displayReflectance(6500.0, np.arange(400.0, 701.0, 10.0), asset_store=asset_store)
    apple = display_create("LCD-Apple.mat", asset_store=asset_store)
    assert display_get(reflectance_display, "name") == "Natural (ill 6500K)"
    assert display_get(reflectance_display, "wave").shape == (31,)
    assert rgb_primaries.shape == (31, 3)
    assert ill_energy.shape == (31,)
    np.testing.assert_allclose(display_get(reflectance_display, "spd"), rgb_primaries, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(display_get(reflectance_display, "gamma"), display_get(apple, "gamma"), rtol=1e-12, atol=1e-12)
    assert np.isclose(display_get(reflectance_display, "max luminance"), 100.0, rtol=1e-6, atol=1e-6)
