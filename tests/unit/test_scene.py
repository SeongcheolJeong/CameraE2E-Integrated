from __future__ import annotations

import imageio.v3 as iio
import numpy as np
import pytest
import pyisetcam.scene as scene_module

from pyisetcam import (
    blackbody,
    buildPyramid,
    build_pyramid,
    display_create,
    display_get,
    exiftool_depth_from_file,
    exiftool_info,
    finalTouch,
    final_touch,
    font_create,
    getPFMraw,
    getpfmraw,
    haarPyramid,
    haar_pyramid,
    imNorm,
    im_norm,
    modulateFlip,
    modulateFlipShift,
    modulate_flip_shift,
    mo_target,
    padReflect,
    padReflectNeg,
    pad_reflect,
    pad_reflect_neg,
    qmfPyramid,
    qmf_pyramid,
    rangeCompressionLum,
    range_compression_lum,
    reconsHaarPyramid,
    reconsPyramid,
    reconsQmfPyramid,
    recons_haar_pyramid,
    recons_pyramid,
    recons_qmf_pyramid,
    scene_adjust_illuminant,
    scene_create,
    scene_from_ddf_file,
    scene_from_file,
    scene_get,
    scene_list,
    scene_pad,
    scene_set,
    scene_sdr,
    scenePad,
    scene_to_file,
)
from pyisetcam.types import Scene


def test_scene_module_core_matlab_aliases() -> None:
    assert scene_module.chartPatchData is scene_module.chart_patch_data
    assert scene_module.exiftoolDepthFromFile is scene_module.exiftool_depth_from_file
    assert scene_module.exiftoolInfo is scene_module.exiftool_info
    assert scene_module.FOTParams is scene_module.fot_params
    assert scene_module.gaborP is scene_module.gabor_p
    assert scene_module.hdrRender is scene_module.hdr_render
    assert scene_module.ieCheckerboard is scene_module.ie_checkerboard
    assert scene_module.ieCookTorrance is scene_module.ie_cook_torrance
    assert scene_module.ieReflectanceSamples is scene_module.ie_reflectance_samples
    assert scene_module.macbethChartCreate is scene_module.macbeth_chart_create
    assert scene_module.macbethDrawRects is scene_module.macbeth_draw_rects
    assert scene_module.macbethEvaluationGraphs is scene_module.macbeth_evaluation_graphs
    assert scene_module.macbethGretagSGCreate is scene_module.macbeth_gretag_sg_create
    assert scene_module.macbethIdealColor is scene_module.macbeth_ideal_color
    assert scene_module.macbethLuminanceNoise is scene_module.macbeth_luminance_noise
    assert scene_module.macbethPatchData is scene_module.macbeth_patch_data
    assert scene_module.macbethReadReflectance is scene_module.macbeth_read_reflectance
    assert scene_module.macbethRectangles is scene_module.macbeth_rectangles
    assert scene_module.macbethROIs is scene_module.macbeth_rois
    assert scene_module.macbethSelect is scene_module.macbeth_select
    assert scene_module.macbethSensorValues is scene_module.macbeth_sensor_values
    assert scene_module.MOTarget is scene_module.mo_target
    assert scene_module.sceneAdd is scene_module.scene_add
    assert scene_module.sceneAddGrid is scene_module.scene_add_grid
    assert scene_module.sceneAdjustIlluminant is scene_module.scene_adjust_illuminant
    assert scene_module.sceneAdjustLuminance is scene_module.scene_adjust_luminance
    assert scene_module.sceneAdjustPixelSize is scene_module.scene_adjust_pixel_size
    assert scene_module.sceneAdjustReflectance is scene_module.scene_adjust_reflectance
    assert scene_module.sceneCalculateLuminance is scene_module.scene_calculate_luminance
    assert scene_module.sceneClearData is scene_module.scene_clear_data
    assert scene_module.sceneCombine is scene_module.scene_combine
    assert scene_module.sceneCreate is scene_module.scene_create
    assert scene_module.sceneCrop is scene_module.scene_crop
    assert scene_module.sceneDescription is scene_module.scene_description
    assert scene_module.sceneEnergyFromVector is scene_module.scene_energy_from_vector
    assert scene_module.sceneExtractWaveband is scene_module.scene_extract_waveband
    assert scene_module.sceneFrequencySupport is scene_module.scene_frequency_support
    assert scene_module.sceneFromBasis is scene_module.scene_from_basis
    assert scene_module.sceneFromDDFFile is scene_module.scene_from_ddf_file
    assert scene_module.sceneFromFile is scene_module.scene_from_file
    assert scene_module.sceneGet is scene_module.scene_get
    assert scene_module.sceneHDRChart is scene_module.scene_hdr_chart
    assert scene_module.sceneHDRImage is scene_module.scene_hdr_image
    assert scene_module.sceneIlluminantPattern is scene_module.scene_illuminant_pattern
    assert scene_module.sceneIlluminantScale is scene_module.scene_illuminant_scale
    assert scene_module.sceneIlluminantSS is scene_module.scene_illuminant_ss
    assert scene_module.sceneInitGeometry is scene_module.scene_init_geometry
    assert scene_module.sceneInitSpatial is scene_module.scene_init_spatial
    assert scene_module.sceneInsert is scene_module.scene_insert
    assert scene_module.sceneInterpolateW is scene_module.scene_interpolate_w
    assert scene_module.sceneList is scene_module.scene_list
    assert scene_module.sceneMakeVideo is scene_module.scene_make_video
    assert scene_module.scenePad is scene_module.scene_pad
    assert scene_module.scenePhotonNoise is scene_module.scene_photon_noise
    assert scene_module.scenePhotonsFromVector is scene_module.scene_photons_from_vector
    assert scene_module.sceneRadianceChart is scene_module.scene_radiance_chart
    assert scene_module.sceneRadianceFromVector is scene_module.scene_radiance_from_vector
    assert scene_module.sceneReflectanceChart is scene_module.scene_reflectance_chart
    assert scene_module.sceneRamp is scene_module.scene_ramp
    assert scene_module.sceneRotate is scene_module.scene_rotate
    assert scene_module.sceneSaveImage is scene_module.scene_save_image
    assert scene_module.sceneSDR is scene_module.scene_sdr
    assert scene_module.sceneSet is scene_module.scene_set
    assert scene_module.sceneShowImage is scene_module.scene_show_image
    assert scene_module.sceneSPDScale is scene_module.scene_spd_scale
    assert scene_module.sceneSpatialResample is scene_module.scene_spatial_resample
    assert scene_module.sceneSpatialSupport is scene_module.scene_spatial_support
    assert scene_module.sceneThumbnail is scene_module.scene_thumbnail
    assert scene_module.sceneToFile is scene_module.scene_to_file
    assert scene_module.sceneTranslate is scene_module.scene_translate
    assert scene_module.sceneVernier is scene_module.scene_vernier
    assert scene_module.sceneWBCreate is scene_module.scene_wb_create


def test_scene_pad_replays_upstream_deprecation_error() -> None:
    assert scenePad is scene_pad
    with pytest.raises(ValueError, match=r"^Deprecated\. You probably want oiPad\.$"):
        scenePad(None, [8, 8, 0])


def test_scene_create_default_macbeth(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    assert photons.shape == (64, 96, wave.size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_scene_create_empty_name_placeholder_uses_default_constructor(asset_store) -> None:
    default_scene = scene_create(asset_store=asset_store)
    placeholder_scene = scene_create([], asset_store=asset_store)

    np.testing.assert_allclose(scene_get(placeholder_scene, "wave"), scene_get(default_scene, "wave"))
    np.testing.assert_allclose(scene_get(placeholder_scene, "photons"), scene_get(default_scene, "photons"))
    assert placeholder_scene.name == default_scene.name


def test_scene_create_blank_name_placeholder_uses_default_constructor(asset_store) -> None:
    default_scene = scene_create(asset_store=asset_store)
    blank_scene = scene_create("", asset_store=asset_store)

    np.testing.assert_allclose(scene_get(blank_scene, "wave"), scene_get(default_scene, "wave"))
    np.testing.assert_allclose(scene_get(blank_scene, "photons"), scene_get(default_scene, "photons"))
    assert blank_scene.name == default_scene.name


def test_scene_create_list_alias_matches_public_listing() -> None:
    listing = scene_create("scene list")
    assert isinstance(listing, str)
    assert listing == scene_list()
    assert "letter" in listing
    assert "multispectral" in listing


def test_scene_create_rgb_multispectral_and_monochrome_shells(asset_store) -> None:
    rgb = scene_create("rgb", asset_store=asset_store)
    multispectral = scene_create("hyperspectral", asset_store=asset_store)
    monochrome = scene_create("unispectral", asset_store=asset_store)

    assert scene_get(rgb, "name") == "rgb"
    assert scene_get(multispectral, "name") == "multispectral"
    assert scene_get(monochrome, "name") == "monochrome"
    assert scene_get(rgb, "photons").shape == (1, 1, 31)
    assert scene_get(multispectral, "photons").shape == (1, 1, 31)
    assert scene_get(monochrome, "photons").shape == (1, 1, 1)
    np.testing.assert_array_equal(scene_get(monochrome, "wave"), np.array([550.0]))
    assert scene_get(rgb, "illuminant comment") == "D65.mat"
    assert scene_get(multispectral, "illuminant comment") == "D65.mat"
    assert scene_get(monochrome, "illuminant comment") == "D65.mat"
    assert np.isclose(scene_get(rgb, "mean luminance", asset_store=asset_store), 0.0, atol=1e-12)

    resized = scene_set(rgb.clone(), "photons", np.ones((2, 3, scene_get(rgb, "nwave")), dtype=float))
    assert tuple(scene_get(resized, "size")) == (2, 3)
    assert scene_get(resized, "fov") == scene_get(rgb, "fov")


def test_scene_create_shells_accept_scene_seed(asset_store) -> None:
    seed = Scene(name="seed")
    seed.metadata["marker"] = "kept"
    seed.fields["custom_field"] = 17
    seed.fields["wave"] = np.array([500.0], dtype=float)
    seed.data["photons"] = np.ones((2, 2, 1), dtype=float)

    rgb = scene_create("rgb", seed, asset_store=asset_store)
    multispectral = scene_create("multispectral", seed, asset_store=asset_store)
    monochrome = scene_create("monochrome", seed, asset_store=asset_store)

    assert rgb.metadata["marker"] == "kept"
    assert multispectral.fields["custom_field"] == 17
    assert monochrome.fields["custom_field"] == 17
    assert tuple(scene_get(rgb, "photons").shape) == (1, 1, 31)
    assert tuple(scene_get(multispectral, "photons").shape) == (1, 1, 31)
    assert tuple(scene_get(monochrome, "photons").shape) == (2, 2, 1)
    np.testing.assert_array_equal(scene_get(rgb, "wave"), np.arange(400.0, 701.0, 10.0, dtype=float))
    np.testing.assert_array_equal(scene_get(multispectral, "wave"), np.arange(400.0, 701.0, 10.0, dtype=float))
    np.testing.assert_array_equal(scene_get(monochrome, "wave"), np.array([550.0], dtype=float))
    np.testing.assert_array_equal(scene_get(monochrome, "photons"), np.ones((2, 2, 1), dtype=float))
    np.testing.assert_array_equal(seed.fields["wave"], np.array([500.0], dtype=float))
    np.testing.assert_array_equal(seed.data["photons"], np.ones((2, 2, 1), dtype=float))


def test_scene_create_ramp_equal_photon_alias_matches_ramp(asset_store) -> None:
    ramp = scene_create("ramp", 32, 128.0, asset_store=asset_store)
    alias = scene_create("ramp equal photon", 32, 128.0, asset_store=asset_store)

    assert scene_get(alias, "name") == scene_get(ramp, "name")
    np.testing.assert_array_equal(scene_get(alias, "wave"), scene_get(ramp, "wave"))
    np.testing.assert_allclose(np.asarray(scene_get(alias, "photons"), dtype=float), np.asarray(scene_get(ramp, "photons"), dtype=float))
    assert np.isclose(
        scene_get(alias, "mean luminance", asset_store=asset_store),
        scene_get(ramp, "mean luminance", asset_store=asset_store),
        rtol=1e-10,
        atol=1e-10,
    )


def test_ramp_family_dispatch_replays_matlab_default_size(asset_store) -> None:
    ramp_default = scene_create("ramp", asset_store=asset_store)
    ramp_explicit = scene_create("ramp", 256, 256.0, asset_store=asset_store)
    linear_default = scene_create("linear intensity ramp", asset_store=asset_store)
    linear_explicit = scene_create("linear intensity ramp", 256, 256.0, asset_store=asset_store)
    exp_default = scene_create("exponential intensity ramp", asset_store=asset_store)
    exp_explicit = scene_create("exponential intensity ramp", 256, 256.0, asset_store=asset_store)

    np.testing.assert_allclose(np.asarray(scene_get(ramp_default, "photons"), dtype=float), np.asarray(scene_get(ramp_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(scene_get(linear_default, "photons"), dtype=float), np.asarray(scene_get(linear_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(scene_get(exp_default, "photons"), dtype=float), np.asarray(scene_get(exp_explicit, "photons"), dtype=float), rtol=0.0, atol=0.0)
    assert tuple(scene_get(ramp_default, "size")) == (256, 256)
    assert tuple(scene_get(linear_default, "size")) == (256, 256)
    assert tuple(scene_get(exp_default, "size")) == (256, 256)


def test_ramp_family_dispatch_accepts_empty_size_and_dynamic_range_placeholders(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    placeholder_cases = (
        ("ramp", ([], [], wave), (128, 256.0, wave)),
        ("linear intensity ramp", ([], [], wave), (128, 256.0, wave)),
        ("exponential intensity ramp", ([], [], wave), (128, 256.0, wave)),
    )

    for name, placeholder_args, explicit_args in placeholder_cases:
        placeholder = scene_create(name, *placeholder_args, asset_store=asset_store)
        explicit = scene_create(name, *explicit_args, asset_store=asset_store)

        np.testing.assert_allclose(
            np.asarray(scene_get(placeholder, "photons"), dtype=float),
            np.asarray(scene_get(explicit, "photons"), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        assert tuple(scene_get(placeholder, "size")) == (128, 128)
        np.testing.assert_array_equal(np.asarray(scene_get(placeholder, "wave"), dtype=float), wave)


def test_scene_adjust_illuminant_preserves_mean(asset_store) -> None:
    scene = scene_create(asset_store=asset_store)
    wave = scene_get(scene, "wave")
    changed = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), True, asset_store=asset_store)
    changed_no_preserve = scene_adjust_illuminant(scene.clone(), blackbody(wave, 3000.0), False, asset_store=asset_store)
    assert np.isclose(
        scene_get(changed, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=5e-2,
    )
    assert not np.isclose(
        scene_get(changed_no_preserve, "mean luminance", asset_store=asset_store),
        scene_get(scene, "mean luminance", asset_store=asset_store),
        rtol=1e-2,
    )


def test_scene_create_supports_macbeth_illuminant_variants(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 30.0, dtype=float)
    ir_wave = np.arange(700.0, 901.0, 40.0, dtype=float)

    d50 = scene_create("macbeth d50", 8, wave, asset_store=asset_store)
    illc = scene_create("macbeth illc", 8, wave, asset_store=asset_store)
    fluor = scene_create("macbeth fluor", 8, wave, asset_store=asset_store)
    custom = scene_create("macbeth custom reflectance", 8, wave, "macbethChart.mat", asset_store=asset_store)
    ee_ir = scene_create("macbeth ee_ir", 8, ir_wave, asset_store=asset_store)

    assert scene_get(d50, "name") == "Macbeth D50"
    assert scene_get(illc, "name") == "Macbeth Ill C"
    assert scene_get(fluor, "name") == "Macbeth Fluorescent"
    assert scene_get(custom, "name") == "Macbeth D65"
    assert scene_get(ee_ir, "name") == "Macbeth IR"
    assert scene_get(d50, "illuminant comment") == "D50.mat"
    assert scene_get(illc, "illuminant comment") == "illuminantC.mat"
    assert scene_get(fluor, "illuminant comment") == "Fluorescent.mat"
    assert scene_get(custom, "illuminant comment") == "D65.mat"
    assert scene_get(ee_ir, "illuminant comment") == "equalEnergy"
    assert scene_get(d50, "photons").shape == (32, 48, wave.size)
    assert scene_get(ee_ir, "photons").shape == (32, 48, ir_wave.size)
    np.testing.assert_array_equal(np.asarray(scene_get(ee_ir, "wave"), dtype=float), ir_wave)
    assert np.isclose(scene_get(d50, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(ee_ir, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert not np.allclose(scene_get(d50, "illuminant energy"), scene_get(custom, "illuminant energy"))
    assert not np.allclose(scene_get(illc, "illuminant energy"), scene_get(custom, "illuminant energy"))
    assert np.allclose(
        np.asarray(scene_get(ee_ir, "illuminant energy"), dtype=float),
        float(np.asarray(scene_get(ee_ir, "illuminant energy"), dtype=float).reshape(-1)[0]),
    )


def test_macbeth_dispatch_accepts_empty_patch_size_placeholder(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    default_placeholder = scene_create("default", None, wave, asset_store=asset_store)
    default_reference = scene_create("default", 16, wave, asset_store=asset_store)
    macbeth_placeholder = scene_create("macbeth", np.array([], dtype=float), wave, asset_store=asset_store)
    macbeth_reference = scene_create("macbeth", 16, wave, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(default_placeholder, "photons"), scene_get(default_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(macbeth_placeholder, "photons"), scene_get(macbeth_reference, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(default_placeholder, "size")) == (64, 96)
    assert tuple(scene_get(macbeth_placeholder, "size")) == (64, 96)
    assert np.array_equal(np.asarray(scene_get(default_placeholder, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(macbeth_placeholder, "wave"), dtype=float), wave)


def test_macbeth_dispatch_accepts_empty_surface_and_border_placeholders(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    default_placeholder = scene_create("default", 16, wave, [], [], asset_store=asset_store)
    default_reference = scene_create("default", 16, wave, "macbethChart.mat", False, asset_store=asset_store)
    tungsten_placeholder = scene_create("macbeth tungsten", 16, wave, [], [], asset_store=asset_store)
    tungsten_reference = scene_create(
        "macbeth tungsten",
        16,
        wave,
        "macbethChart.mat",
        False,
        asset_store=asset_store,
    )

    np.testing.assert_allclose(
        np.asarray(scene_get(default_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(default_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(tungsten_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(tungsten_reference, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert scene_get(default_placeholder, "illuminant comment") == "D65.mat"
    assert scene_get(tungsten_placeholder, "illuminant comment") == "tungsten"


def test_scene_create_moire_orient_replays_green_target_plane(asset_store) -> None:
    params = {"sceneSize": 96, "f": 1.0 / 1200.0}
    scene = scene_create("moire orient", params, asset_store=asset_store)
    positional = scene_create("moire orient", 96, 1.0 / 1200.0, asset_store=asset_store)
    placeholder_frequency = scene_create("moire orient", 96, [], asset_store=asset_store)
    default_frequency = scene_create("moire orient", {"sceneSize": 96}, asset_store=asset_store)
    empty_payload = scene_create("moire orient", [], asset_store=asset_store)
    default_payload = scene_create("moire orient", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    positional_photons = np.asarray(scene_get(positional, "photons"), dtype=float)
    placeholder_photons = np.asarray(scene_get(placeholder_frequency, "photons"), dtype=float)
    default_photons = np.asarray(scene_get(default_frequency, "photons"), dtype=float)
    empty_payload_photons = np.asarray(scene_get(empty_payload, "photons"), dtype=float)
    default_payload_photons = np.asarray(scene_get(default_payload, "photons"), dtype=float)
    expected = np.clip(np.asarray(mo_target("sinusoidalim", params), dtype=float)[:, :, 1], 1.0e-4, 1.0)
    expected = expected / np.max(expected)
    actual = photons[:, :, 0] / np.max(photons[:, :, 0])

    assert scene_get(scene, "name") == "MOTarget"
    assert scene_get(positional, "name") == "MOTarget"
    assert photons.shape[:2] == (96, 96)
    assert positional_photons.shape[:2] == (96, 96)
    np.testing.assert_allclose(photons[:, :, 0], photons[:, :, -1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(positional_photons, photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(placeholder_photons, default_photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(empty_payload_photons, default_payload_photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


@pytest.mark.parametrize(
    ("scene_name", "params"),
    [
        ("frequency orientation", {"blockSize": [], "contrast": [], "angles": [], "freqs": []}),
        ("harmonic", {"row": [], "col": [], "ang": [], "contrast": [], "freq": [], "ph": [], "center": [], "gaborFlag": []}),
        ("hdr lights", {"imageSize": [], "nCircles": [], "radius": [], "circleColors": [], "nLines": [], "lineLength": [], "lineColors": []}),
        ("moire orient", {"sceneSize": [], "f": []}),
    ],
)
def test_scene_create_empty_parameter_fields_match_default_dispatch(scene_name, params, asset_store) -> None:
    default_scene = scene_create(scene_name, asset_store=asset_store)
    placeholder_scene = scene_create(scene_name, params, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
    )


def test_scene_create_letter_branch_reuses_font_pipeline(asset_store) -> None:
    font = font_create("A", "Georgia", 18, asset_store=asset_store)
    scene = scene_create("letter", font, "LCD-Apple", asset_store=asset_store)
    shortcut = scene_create("letter", "A", 18, "Georgia", "LCD-Apple", asset_store=asset_store)
    photons = np.asarray(scene_get(scene, "photons"), dtype=float)
    shortcut_photons = np.asarray(scene_get(shortcut, "photons"), dtype=float)

    assert scene_get(scene, "name") == font["name"]
    assert scene_get(shortcut, "name") == font["name"]
    assert photons.ndim == 3
    assert photons.shape[2] == scene_get(scene, "nwave")
    assert float(np.max(photons)) > float(np.min(photons))
    assert scene_get(scene, "fov") > 0.0
    np.testing.assert_allclose(shortcut_photons, photons, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(scene_get(shortcut, "wave"), dtype=float),
        np.asarray(scene_get(scene, "wave"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )


def test_scene_create_letter_accepts_empty_font_placeholders(asset_store) -> None:
    default_scene = scene_create("letter", asset_store=asset_store)
    placeholder_scene = scene_create("letter", [], [], asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )
    assert scene_get(placeholder_scene, "name") == scene_get(default_scene, "name")


def test_scene_create_letter_shortcut_accepts_empty_optional_placeholders(asset_store) -> None:
    placeholder = scene_create("letter", "g", [], [], [], asset_store=asset_store)
    explicit = scene_create("letter", "g", 14, "Georgia", "LCD-Apple", asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder, "photons"), dtype=float),
        np.asarray(scene_get(explicit, "photons"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder, "wave"), dtype=float),
        np.asarray(scene_get(explicit, "wave"), dtype=float),
        atol=0.0,
        rtol=0.0,
    )
    assert scene_get(placeholder, "name") == scene_get(explicit, "name") == "g-georgia-14-96"


def test_uniform_family_accepts_empty_size_placeholders(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)

    uniform_placeholder = scene_create("uniform", [], wave, asset_store=asset_store)
    uniform_reference = scene_create("uniform", 32, wave, asset_store=asset_store)
    d65_placeholder = scene_create("uniform d65", np.array([], dtype=float), wave, asset_store=asset_store)
    d65_reference = scene_create("uniform d65", 32, wave, asset_store=asset_store)
    ep_placeholder = scene_create("uniform ep", [], wave, asset_store=asset_store)
    ep_reference = scene_create("uniform ep", 32, wave, asset_store=asset_store)
    bb_placeholder = scene_create("uniform bb", [], 4500, wave, asset_store=asset_store)
    bb_reference = scene_create("uniform bb", 32, 4500, wave, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(uniform_placeholder, "photons"), scene_get(uniform_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(d65_placeholder, "photons"), scene_get(d65_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(ep_placeholder, "photons"), scene_get(ep_reference, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(bb_placeholder, "photons"), scene_get(bb_reference, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(uniform_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(d65_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(ep_placeholder, "size")) == (32, 32)
    assert tuple(scene_get(bb_placeholder, "size")) == (32, 32)
    np.testing.assert_array_equal(np.asarray(scene_get(bb_placeholder, "wave"), dtype=float), wave)


def test_uniform_monochromatic_dispatch_accepts_size_first_docs_form(asset_store) -> None:
    size_first = scene_create("uniform monochromatic", 12, 550, asset_store=asset_store)
    wave_first = scene_create("uniform monochromatic", 550, 12, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(size_first, "photons"), scene_get(wave_first, "photons"), rtol=0.0, atol=0.0)
    assert tuple(scene_get(size_first, "size")) == (12, 12)
    np.testing.assert_array_equal(np.asarray(scene_get(size_first, "wave"), dtype=float), np.array([550.0], dtype=float))
    assert np.isclose(scene_get(size_first, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_uniform_monochromatic_dispatch_accepts_size_first_placeholders(asset_store) -> None:
    placeholder_wave = scene_create("uniform monochromatic", 12, [], asset_store=asset_store)
    explicit_default_wave = scene_create("uniform monochromatic", 12, 500, asset_store=asset_store)
    placeholder_size = scene_create("uniform monochromatic", [], 550, asset_store=asset_store)
    explicit_default_size = scene_create("uniform monochromatic", 128, 550, asset_store=asset_store)

    np.testing.assert_allclose(
        scene_get(placeholder_wave, "photons"),
        scene_get(explicit_default_wave, "photons"),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        scene_get(placeholder_size, "photons"),
        scene_get(explicit_default_size, "photons"),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(placeholder_wave, "size")) == (12, 12)
    assert tuple(scene_get(placeholder_size, "size")) == (128, 128)
    np.testing.assert_array_equal(
        np.asarray(scene_get(placeholder_wave, "wave"), dtype=float),
        np.array([500.0], dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(scene_get(placeholder_size, "wave"), dtype=float),
        np.array([550.0], dtype=float),
    )


def test_scene_from_file_supports_multispectral_mat_files(asset_store) -> None:
    scene = scene_from_file(
        asset_store.resolve("data/images/multispectral/Feng_Office-hdrs.mat"),
        "multispectral",
        200.0,
        asset_store=asset_store,
    )

    photons = scene_get(scene, "photons")
    wave = scene_get(scene, "wave")
    illuminant = scene_get(scene, "illuminant photons")

    assert photons.shape == (506, 759, 31)
    assert wave.shape == (31,)
    assert illuminant.shape == photons.shape
    assert scene_get(scene, "illuminant format") == "spatial spectral"
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 200.0, rtol=5e-2)


def test_scene_from_ddf_file_matches_scene_from_file_without_embedded_depth(tmp_path, asset_store) -> None:
    image = np.linspace(0.0, 1.0, 4 * 6 * 3, dtype=float).reshape(4, 6, 3)
    image_path = tmp_path / "ddf-source.png"
    iio.imwrite(image_path, np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8))

    expected = scene_from_file(image_path, "rgb", 75.0, "LCD-Apple.mat", asset_store=asset_store)
    actual = scene_from_ddf_file(image_path, "rgb", 75.0, "LCD-Apple.mat", asset_store=asset_store)

    np.testing.assert_array_equal(np.asarray(scene_get(actual, "size"), dtype=int), np.asarray(scene_get(expected, "size"), dtype=int))
    np.testing.assert_allclose(np.asarray(scene_get(actual, "photons"), dtype=float), np.asarray(scene_get(expected, "photons"), dtype=float))
    assert actual.name == expected.name
    assert np.isclose(
        scene_get(actual, "mean luminance", asset_store=asset_store),
        scene_get(expected, "mean luminance", asset_store=asset_store),
        rtol=1e-10,
        atol=1e-10,
    )


def test_exiftool_info_supports_json_format(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"jpg")

    class Result:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    def fake_run(command, check=False, capture_output=True, text=True):  # type: ignore[no-untyped-def]
        del check, capture_output, text
        assert command[:2] == ["/usr/bin/exiftool", "-j"]
        return Result('[{"Orientation":"Rotate 90 CW","DepthMapNear":1.0}]')

    monkeypatch.setattr(scene_module.shutil, "which", lambda name: "/usr/bin/exiftool")
    monkeypatch.setattr(scene_module.subprocess, "run", fake_run)

    info = exiftool_info(image_path, "format", "json")
    assert info["Orientation"] == "Rotate 90 CW"
    assert np.isclose(float(info["DepthMapNear"]), 1.0)


def test_exiftool_depth_from_file_decodes_meter_payload(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"jpg")
    payload_path = tmp_path / "depth.png"
    encoded = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    iio.imwrite(payload_path, encoded)

    monkeypatch.setattr(
        scene_module,
        "exiftool_info",
        lambda *args, **kwargs: {
            "DepthMapUnits": "Meters",
            "DepthMapNear": 1.0,
            "DepthMapFar": 5.0,
            "ImageHeight": 2,
            "ImageWidth": 2,
        },
    )
    monkeypatch.setattr(scene_module, "_exiftool_depth_payload", lambda path: payload_path.read_bytes())

    depth = exiftool_depth_from_file(image_path, "type", "GooglePixel")
    expected = 1.0 + (encoded.astype(float) / 255.0) * 4.0

    np.testing.assert_allclose(depth, expected, rtol=0.0, atol=1e-6)


def test_hdr_helper_wrappers_cover_padding_and_pyramid_reconstruction() -> None:
    small = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    expected_pad = np.array(
        [
            [4.0, 3.0, 4.0, 3.0],
            [2.0, 1.0, 2.0, 1.0],
            [4.0, 3.0, 4.0, 3.0],
            [2.0, 1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    expected_pad_neg = np.array(
        [
            [-4.0, 3.0, 4.0, -3.0],
            [-2.0, 1.0, 2.0, -1.0],
            [-4.0, 3.0, 4.0, -3.0],
            [-2.0, 1.0, 2.0, -1.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(pad_reflect(small, 1), expected_pad)
    np.testing.assert_allclose(padReflect(small, 1), expected_pad)
    np.testing.assert_allclose(pad_reflect_neg(small, 1, 1, 1, 0), expected_pad_neg)
    np.testing.assert_allclose(padReflectNeg(small, 1, 1, 1, 0), expected_pad_neg)

    image = np.arange(1.0, 65.0, dtype=float).reshape(8, 8) / 64.0
    haar = haar_pyramid(image, 2)
    alias_haar = haarPyramid(image, 2)
    built_haar, filt_num = build_pyramid(image, 2, "haar")
    alias_built_haar, alias_filt_num = buildPyramid(image, 2, "haar")

    assert filt_num == 3
    assert alias_filt_num == 3
    np.testing.assert_allclose(alias_haar, haar)
    np.testing.assert_allclose(built_haar, haar)
    np.testing.assert_allclose(alias_built_haar, haar)
    haar_reconstructed = recons_haar_pyramid(haar)
    alias_haar_reconstructed = reconsHaarPyramid(haar)
    dispatch_haar_reconstructed = recons_pyramid(haar, 3, "haar")
    alias_dispatch_haar_reconstructed = reconsPyramid(haar, 3, "haar")
    np.testing.assert_allclose(alias_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dispatch_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(alias_dispatch_haar_reconstructed, haar_reconstructed, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[1:, 1:], image[1:, 1:], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[0, 1:] - image[0, 1:], np.full(7, 1.0 / 16.0), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(haar_reconstructed[1:, 0] - image[1:, 0], np.full(7, 1.0 / 128.0), atol=1e-10, rtol=1e-10)
    assert np.isclose(float(haar_reconstructed[0, 0] - image[0, 0]), 9.0 / 128.0, atol=1e-10, rtol=1e-10)

    qmf = qmf_pyramid(image, 2)
    alias_qmf = qmfPyramid(image, 2)
    built_qmf, qmf_filt_num = build_pyramid(image, 2, "qmf")
    np.testing.assert_allclose(alias_qmf, qmf)
    np.testing.assert_allclose(built_qmf, qmf)
    assert qmf_filt_num == 3
    np.testing.assert_allclose(recons_qmf_pyramid(qmf), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(reconsQmfPyramid(qmf), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(recons_pyramid(qmf, 3, "qmf"), image, atol=6e-4, rtol=6e-4)
    np.testing.assert_allclose(reconsPyramid(qmf, 3, "qmf"), image, atol=6e-4, rtol=6e-4)

    with pytest.raises(NotImplementedError):
        build_pyramid(image, 2, "steerable")
    with pytest.raises(NotImplementedError):
        recons_pyramid(haar, 3, "steerable")


def test_hdr_helper_wrappers_cover_normalization_range_touch_filter_and_pfm(tmp_path) -> None:
    image = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=float)
    expected_norm = (image - 1.0) / 4.0
    expected_touch = image + 0.15 * scene_module._hist_equalize_global(np.real(image))
    grayscale = np.linspace(0.1, 1.6, 64, dtype=float).reshape(8, 8)
    expected_range = scene_module._range_compression_lum(grayscale, filt_type="haar", beta=0.6, alpha_a=0.2, ifsharp=0)

    np.testing.assert_allclose(im_norm(image), expected_norm)
    np.testing.assert_allclose(imNorm(image), expected_norm)
    np.testing.assert_allclose(final_touch(image), expected_touch)
    np.testing.assert_allclose(finalTouch(image), expected_touch)
    np.testing.assert_allclose(range_compression_lum(grayscale), expected_range)
    np.testing.assert_allclose(rangeCompressionLum(grayscale), expected_range)
    np.testing.assert_allclose(modulate_flip_shift([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))
    np.testing.assert_allclose(modulateFlip([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))
    np.testing.assert_allclose(modulateFlipShift([1.0, 2.0, 3.0, 4.0]), np.array([4.0, -3.0, 2.0, -1.0]))

    rgb = (np.arange(1.0, 13.0, dtype=np.float32).reshape(2, 2, 3)) / 12.0
    flipped = rgb[::-1, :, :]
    payload = np.concatenate([np.reshape(flipped[:, :, channel].T, -1, order="F") for channel in range(3)]).astype(np.float32)
    pfm_path = tmp_path / "sample.pfm"
    pfm_path.write_bytes(b"P7\n2 2\n1.0\n" + payload.tobytes())

    np.testing.assert_allclose(getpfmraw(pfm_path), rgb, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(getPFMraw(pfm_path), rgb, atol=1e-7, rtol=1e-7)


def test_scene_sdr_prefers_local_cache_for_mat_and_png(tmp_path, asset_store) -> None:
    cached_scene = scene_create("macbeth d65", asset_store=asset_store)
    scene_to_file(tmp_path / "local-scene.mat", cached_scene)

    loaded_scene = scene_sdr(
        "isetcam bitterli",
        "local-scene",
        download_dir=tmp_path,
        asset_store=asset_store,
    )
    assert tuple(scene_get(loaded_scene, "size")) == tuple(scene_get(cached_scene, "size"))
    np.testing.assert_array_equal(np.asarray(scene_get(loaded_scene, "wave"), dtype=float), np.asarray(scene_get(cached_scene, "wave"), dtype=float))
    assert loaded_scene.name == cached_scene.name

    cached_png = (np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 3) % 255
    png_path = tmp_path / "preview.png"
    iio.imwrite(png_path, cached_png)

    loaded_png = scene_sdr("isetcam pharr", "preview.png", download_dir=tmp_path)
    np.testing.assert_array_equal(np.asarray(loaded_png), cached_png)


def test_scene_sdr_rejects_unknown_deposit_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid deposit name"):
        scene_sdr("unknown deposit", "preview.png", download_dir=tmp_path)


def test_supported_pattern_scenes(asset_store) -> None:
    mackay = scene_create("rings rays", asset_store=asset_store)
    checkerboard = scene_create("checkerboard", 8, 4, asset_store=asset_store)
    slanted_bar = scene_create("slanted bar", 64, 0.6, 3.0, asset_store=asset_store)
    freq_orient = scene_create("frequency orientation", asset_store=asset_store)
    harmonic = scene_create("harmonic", asset_store=asset_store)
    sweep = scene_create("sweep frequency", asset_store=asset_store)
    star = scene_create("star pattern", 64, "ee", 6, asset_store=asset_store)
    line = scene_create("line ee", 32, 2, asset_store=asset_store)
    bar = scene_create("bar", 32, 5, asset_store=asset_store)
    point_array = scene_create("point array", 64, 16, "ep", 3, asset_store=asset_store)
    square_array = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    grid_lines = scene_create("grid lines", 64, 16, "ee", 2, asset_store=asset_store)
    white_noise = scene_create("white noise", 32, 20, asset_store=asset_store)
    lstar = scene_create("lstar", [80, 10], 20, 1, asset_store=asset_store)
    uniform_specify = scene_create("uniformEESpecify", 128, np.arange(380.0, 721.0, 10.0, dtype=float), asset_store=asset_store)
    assert scene_get(mackay, "photons").shape[:2] == (256, 256)
    assert scene_get(checkerboard, "photons").shape[:2] == (64, 64)
    assert scene_get(slanted_bar, "photons").shape[:2] == (65, 65)
    assert scene_get(freq_orient, "photons").shape[:2] == (256, 256)
    assert scene_get(harmonic, "photons").shape[:2] == (65, 65)
    assert scene_get(sweep, "photons").shape[:2] == (128, 128)
    assert scene_get(star, "photons").shape[:2] == (64, 64)
    assert scene_get(line, "photons").shape[:2] == (32, 32)
    assert scene_get(bar, "photons").shape[:2] == (32, 32)
    assert scene_get(point_array, "photons").shape[:2] == (64, 64)
    assert scene_get(square_array, "photons").shape[:2] == (64, 64)
    assert scene_get(grid_lines, "photons").shape[:2] == (64, 64)
    assert scene_get(white_noise, "photons").shape[:2] == (32, 32)
    assert scene_get(lstar, "photons").shape[:2] == (80, 200)
    assert scene_get(uniform_specify, "photons").shape[:2] == (128, 128)
    assert np.isclose(scene_get(freq_orient, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(harmonic, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(sweep, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(star, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(mackay, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(square_array, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(lstar, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(uniform_specify, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.isclose(scene_get(freq_orient, "fov"), 10.0)
    assert np.isclose(scene_get(harmonic, "fov"), 1.0)
    assert np.isclose(scene_get(sweep, "fov"), 10.0)
    assert np.isclose(scene_get(mackay, "fov"), 10.0)
    assert np.isclose(scene_get(point_array, "fov"), 40.0)
    assert np.isclose(scene_get(square_array, "fov"), 40.0)
    assert np.isclose(scene_get(grid_lines, "fov"), 40.0)
    assert np.isclose(scene_get(white_noise, "fov"), 1.0)


def test_iso12233_dispatch_accepts_empty_fov_placeholder(asset_store) -> None:
    wave = np.array([400.0, 500.0, 600.0], dtype=float)

    none_placeholder = scene_create("iso12233", 64, 1.33, None, wave, 0.3, asset_store=asset_store)
    empty_placeholder = scene_create("iso12233", 64, 1.33, [], wave, 0.3, asset_store=asset_store)
    explicit_default = scene_create("iso12233", 64, 1.33, 2.0, wave, 0.3, asset_store=asset_store)

    np.testing.assert_allclose(scene_get(none_placeholder, "photons"), scene_get(explicit_default, "photons"), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scene_get(empty_placeholder, "photons"), scene_get(explicit_default, "photons"), rtol=0.0, atol=0.0)
    assert np.isclose(float(scene_get(none_placeholder, "fov")), 2.0)
    assert np.isclose(float(scene_get(empty_placeholder, "fov")), 2.0)


def test_mackay_scene_has_center_mask_and_radial_alias(asset_store) -> None:
    rings = scene_create("rings rays", asset_store=asset_store)
    mackay = scene_create("mackay", asset_store=asset_store)

    rings_plane = scene_get(rings, "photons")[:, :, 0]
    mackay_plane = scene_get(mackay, "photons")[:, :, 0]

    assert np.array_equal(rings_plane, mackay_plane)
    assert rings_plane.shape == (256, 256)
    assert rings_plane[rings_plane.shape[0] // 2, rings_plane.shape[1] // 2] > np.min(rings_plane)


def test_lstar_scene_creates_monotonic_bar_steps(asset_store) -> None:
    scene = scene_create("lstar", [80, 10], 20, 1, asset_store=asset_store)
    luminance = np.asarray(scene_get(scene, "luminance", asset_store=asset_store), dtype=float)
    bar_means = np.array([np.mean(luminance[:, start : start + 10]) for start in range(0, 200, 10)], dtype=float)

    assert luminance.shape == (80, 200)
    assert np.all(np.diff(bar_means) > 0.0)


def test_lstar_dispatch_accepts_empty_placeholders(asset_store) -> None:
    default_scene = scene_create("l star", asset_store=asset_store)
    placeholder_scene = scene_create("l star", [], [], [], asset_store=asset_store)
    partial_placeholder_scene = scene_create("l star", [128, 20], [], [], asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(partial_placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(placeholder_scene, "size")) == tuple(scene_get(default_scene, "size"))
    assert tuple(scene_get(partial_placeholder_scene, "size")) == tuple(scene_get(default_scene, "size"))


def test_star_pattern_scene_matches_radial_lines_alias(asset_store) -> None:
    star = scene_create("star pattern", 64, "ee", 6, asset_store=asset_store)
    radial = scene_create("radial lines", 64, "ee", 6, asset_store=asset_store)

    star_photons = scene_get(star, "photons")
    radial_photons = scene_get(radial, "photons")

    assert np.array_equal(star_photons, radial_photons)
    assert star_photons.shape[:2] == (64, 64)
    assert np.isclose(scene_get(star, "fov"), 10.0)
    assert float(star_photons[32, 32, 0]) > float(star_photons[0, 0, 0])


def test_uniform_blackbody_and_monochromatic_scenes(asset_store) -> None:
    bb = scene_create("uniform bb", 16, 4500, asset_store=asset_store)
    mono = scene_create("uniform monochromatic", 550, 12, asset_store=asset_store)

    assert scene_get(bb, "photons").shape == (16, 16, scene_get(bb, "wave").size)
    assert scene_get(mono, "photons").shape == (12, 12, 1)
    assert np.array_equal(scene_get(mono, "wave"), np.array([550.0]))


def test_uniform_blackbody_dispatch_accepts_empty_temperature_placeholder(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    placeholder = scene_create("uniform bb", 32, [], wave, asset_store=asset_store)
    explicit = scene_create("uniform bb", 32, 5000.0, wave, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder, "photons"), dtype=float),
        np.asarray(scene_get(explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        np.asarray(scene_get(placeholder, "wave"), dtype=float),
        np.asarray(scene_get(explicit, "wave"), dtype=float),
    )


def test_line_and_bar_patterns_have_centered_bright_features(asset_store) -> None:
    line = scene_create("line ee", 33, 1, asset_store=asset_store)
    bar = scene_create("bar", 33, 3, asset_store=asset_store)

    line_plane = scene_get(line, "photons")[:, :, 0]
    bar_plane = scene_get(bar, "photons")[:, :, 0]
    line_column_energy = np.sum(line_plane, axis=0)
    bar_column_energy = np.sum(bar_plane, axis=0)

    assert int(np.argmax(line_column_energy)) == 16 + 1
    assert np.array_equal(np.sort(np.argsort(bar_column_energy)[-3:]), np.array([15, 16, 17]))


def test_bar_ee_alias_uses_equal_energy_spectrum(asset_store) -> None:
    canonical = scene_create("bar", 33, 3, asset_store=asset_store)
    alias = scene_create("bar ee", 33, 3, asset_store=asset_store)

    canonical_energy = np.asarray(scene_get(canonical, "illuminant energy"), dtype=float).reshape(-1)
    alias_energy = np.asarray(scene_get(alias, "illuminant energy"), dtype=float).reshape(-1)
    canonical_plane = np.asarray(scene_get(canonical, "photons"), dtype=float)[:, :, 0]
    alias_plane = np.asarray(scene_get(alias, "photons"), dtype=float)[:, :, 0]

    assert np.array_equal(
        np.sort(np.argsort(np.sum(canonical_plane, axis=0))[-3:]),
        np.array([15, 16, 17], dtype=int),
    )
    assert np.array_equal(
        np.sort(np.argsort(np.sum(alias_plane, axis=0))[-3:]),
        np.array([15, 16, 17], dtype=int),
    )
    assert tuple(scene_get(alias, "size")) == (33, 33)
    assert not np.allclose(alias_energy, canonical_energy, atol=0.0, rtol=1.0e-10)
    assert np.allclose(alias_energy / alias_energy[0], np.ones_like(alias_energy), atol=1.0e-12, rtol=1.0e-12)


def test_bar_dispatch_replays_matlab_default_width(asset_store) -> None:
    default_bar = scene_create("bar", asset_store=asset_store)
    explicit_bar = scene_create("bar", 64, 3, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(default_bar, "photons"), dtype=float),
        np.asarray(scene_get(explicit_bar, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(default_bar, "size")) == (64, 64)


def test_squares_alias_matches_square_array_scene(asset_store) -> None:
    canonical = scene_create("square array", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)
    alias = scene_create("squares", 64, 8, np.array([2, 2], dtype=int), asset_store=asset_store)

    np.testing.assert_allclose(
        scene_get(alias, "photons"),
        scene_get(canonical, "photons"),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(alias, "size")) == (64, 64)


def test_disk_array_dispatch_replays_matlab_default_radius(asset_store) -> None:
    default_scene = scene_create("disk array", asset_store=asset_store)
    explicit_scene = scene_create("disk array", 128, 128, np.array([1, 1], dtype=int), asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        np.asarray(scene_get(explicit_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(default_scene, "size")) == (128, 128)


def test_disk_and_square_array_dispatch_accept_empty_size_placeholders(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    disk_placeholder = scene_create("disk array", 96, [], [2, 1], wave, asset_store=asset_store)
    disk_explicit = scene_create("disk array", 96, 128, [2, 1], wave, asset_store=asset_store)
    square_placeholder = scene_create("square array", 96, [], [2, 1], wave, asset_store=asset_store)
    square_explicit = scene_create("square array", 96, 16, [2, 1], wave, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(disk_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(disk_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(square_placeholder, "photons"), dtype=float),
        np.asarray(scene_get(square_explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        np.asarray(scene_get(disk_placeholder, "wave"), dtype=float),
        np.asarray(scene_get(disk_explicit, "wave"), dtype=float),
    )
    assert np.array_equal(
        np.asarray(scene_get(square_placeholder, "wave"), dtype=float),
        np.asarray(scene_get(square_explicit, "wave"), dtype=float),
    )


def test_zone_plate_dispatch_accepts_optional_field_of_view_and_wave(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    default_zone = scene_create("zone plate", asset_store=asset_store)
    placeholder_size = scene_create("zone plate", [], asset_store=asset_store)
    explicit_fov = scene_create("zone plate", 96, 7.5, wave, asset_store=asset_store)
    placeholder_fov = scene_create("zone plate", 96, [], wave, asset_store=asset_store)
    wave_only = scene_create("zone plate", 96, wave, asset_store=asset_store)

    assert tuple(scene_get(default_zone, "size")) == (384, 384)
    assert tuple(scene_get(placeholder_size, "size")) == (256, 256)
    assert np.isclose(scene_get(default_zone, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(placeholder_size, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.asarray(scene_get(default_zone, "photons"), dtype=float).shape == (384, 384, 31)
    assert np.asarray(scene_get(placeholder_size, "photons"), dtype=float).shape == (256, 256, 31)
    assert tuple(scene_get(explicit_fov, "size")) == (96, 96)
    assert tuple(scene_get(placeholder_fov, "size")) == (96, 96)
    assert tuple(scene_get(wave_only, "size")) == (96, 96)
    assert np.isclose(scene_get(explicit_fov, "fov"), 7.5, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(placeholder_fov, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.isclose(scene_get(wave_only, "fov"), 4.0, atol=1e-12, rtol=1e-12)
    assert np.array_equal(np.asarray(scene_get(explicit_fov, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(placeholder_fov, "wave"), dtype=float), wave)
    assert np.array_equal(np.asarray(scene_get(wave_only, "wave"), dtype=float), wave)
    assert np.asarray(scene_get(explicit_fov, "photons"), dtype=float).shape == (96, 96, 3)
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_fov, "photons"), dtype=float),
        np.asarray(scene_get(wave_only, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_noise_mackay_and_slanted_bar_dispatch_accept_empty_optional_placeholders(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)

    noise_placeholder = scene_create("whitenoise", 128, [], wave, asset_store=asset_store)
    noise_explicit = scene_create("whitenoise", 128, 20.0, wave, asset_store=asset_store)
    mackay_radfreq_placeholder = scene_create("rings rays", [], 256, wave, asset_store=asset_store)
    mackay_radfreq_explicit = scene_create("rings rays", 8.0, 256, wave, asset_store=asset_store)
    mackay_size_placeholder = scene_create("rings rays", 8.0, [], wave, asset_store=asset_store)
    mackay_size_explicit = scene_create("rings rays", 8.0, 256, wave, asset_store=asset_store)
    slanted_placeholder = scene_create("slanted bar", [], [], [], wave, [], asset_store=asset_store)
    slanted_explicit = scene_create("slanted bar", 384, 2.6, 2.0, wave, 0.0, asset_store=asset_store)

    for placeholder, explicit in (
        (noise_placeholder, noise_explicit),
        (mackay_radfreq_placeholder, mackay_radfreq_explicit),
        (mackay_size_placeholder, mackay_size_explicit),
        (slanted_placeholder, slanted_explicit),
    ):
        np.testing.assert_allclose(
            np.asarray(scene_get(placeholder, "photons"), dtype=float),
            np.asarray(scene_get(explicit, "photons"), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        assert np.array_equal(
            np.asarray(scene_get(placeholder, "wave"), dtype=float),
            np.asarray(scene_get(explicit, "wave"), dtype=float),
        )


def test_point_array_and_grid_lines_follow_spacing(asset_store) -> None:
    point_array = scene_create("point array", 32, 8, "ep", 1, asset_store=asset_store)
    grid_lines = scene_create("grid lines", 32, 8, "ep", 1, asset_store=asset_store)

    point_plane = scene_get(point_array, "photons")[:, :, 0]
    grid_plane = scene_get(grid_lines, "photons")[:, :, 0]

    point_positions = np.argwhere(point_plane > 0.5)
    assert [int(point_positions[0, 0]), int(point_positions[0, 1])] == [3, 3]
    assert np.all((point_positions[:, 0] - 3) % 8 == 0)
    assert np.all((point_positions[:, 1] - 3) % 8 == 0)
    assert np.all(grid_plane[3::8, :] > 0.5)
    assert np.all(grid_plane[:, 3::8] > 0.5)


def test_frequency_orientation_scene_matches_upstream_parameterization(asset_store) -> None:
    params = {
        "angles": np.linspace(0.0, np.pi / 2.0, 5),
        "freqs": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        "blockSize": 64,
        "contrast": 0.8,
    }
    scene = scene_create("frequency orientation", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (320, 320)
    assert np.max(photons) > np.min(photons)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_frequency_orientation_scene_empty_params_match_default_dispatch(asset_store) -> None:
    default_scene = scene_create("frequency orientation", asset_store=asset_store)
    placeholder_scene = scene_create("frequency orientation", [], asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
    )


def test_harmonic_scene_empty_params_match_default_dispatch(asset_store) -> None:
    default_scene = scene_create("harmonic", asset_store=asset_store)
    placeholder_scene = scene_create("harmonic", [], asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_scene, "photons"), dtype=float),
        np.asarray(scene_get(default_scene, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
    )


def test_freq_orient_scalar_size_matches_tutorial_shape(asset_store) -> None:
    scene = scene_create("freq orient", 512, asset_store=asset_store)
    assert scene_get(scene, "photons").shape[:2] == (512, 512)


def test_harmonic_scene_supports_multiple_components_and_gabor_window(asset_store) -> None:
    params = {
        "freq": np.array([1.0, 5.0]),
        "contrast": np.array([0.2, 0.6]),
        "ph": np.array([0.0, np.pi / 3.0]),
        "ang": np.array([0.0, 0.0]),
        "row": 128,
        "col": 128,
        "GaborFlag": 0.2,
    }
    scene = scene_create("harmonic", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (128, 128)
    assert np.isclose(scene_get(scene, "fov"), 1.0)
    assert photons[64, 64] > photons[0, 0]


def test_sweep_frequency_scene_supports_custom_frequency_and_contrast_profile(asset_store) -> None:
    y_contrast = np.linspace(1.0, 0.25, 64, dtype=float)
    scene = scene_create("sweepFrequency", 64, 12, None, y_contrast, asset_store=asset_store)
    photons = scene_get(scene, "photons")[:, :, 0]

    assert photons.shape == (64, 64)
    assert np.ptp(photons[0, :]) > np.ptp(photons[-1, :])
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)


def test_sweep_frequency_dispatch_accepts_empty_placeholders(asset_store) -> None:
    placeholder = scene_create("sweep frequency", [], 12.0, [], [], asset_store=asset_store)
    explicit = scene_create("sweep frequency", 128, 12.0, None, None, asset_store=asset_store)
    placeholder_maxf = scene_create("sweep frequency", 64, [], [], [], asset_store=asset_store)
    explicit_maxf = scene_create("sweep frequency", 64, 4.0, None, None, asset_store=asset_store)

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder, "photons"), dtype=float),
        np.asarray(scene_get(explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(scene_get(placeholder, "size")) == (128, 128)
    assert np.array_equal(
        np.asarray(scene_get(placeholder, "wave"), dtype=float),
        np.asarray(scene_get(explicit, "wave"), dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder_maxf, "photons"), dtype=float),
        np.asarray(scene_get(explicit_maxf, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_dead_leaves_dispatch_ignores_empty_third_placeholder(asset_store) -> None:
    default_scene = scene_create("dead leaves", 96, 3.0, asset_store=asset_store)
    placeholder_scene = scene_create("dead leaves", 96, 3.0, [], asset_store=asset_store)

    assert np.array_equal(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
    )
    assert tuple(scene_get(placeholder_scene, "size")) == tuple(scene_get(default_scene, "size"))
    assert np.isclose(scene_get(placeholder_scene, "fov"), scene_get(default_scene, "fov"), atol=1e-12, rtol=1e-12)
    assert np.asarray(scene_get(placeholder_scene, "photons"), dtype=float).shape == np.asarray(
        scene_get(default_scene, "photons"),
        dtype=float,
    ).shape


def test_dead_leaves_dispatch_accepts_empty_size_and_sigma_placeholders(asset_store) -> None:
    default_scene = scene_create("dead leaves", asset_store=asset_store)
    placeholder_scene = scene_create("dead leaves", [], [], [], asset_store=asset_store)

    assert np.array_equal(
        np.asarray(scene_get(placeholder_scene, "wave"), dtype=float),
        np.asarray(scene_get(default_scene, "wave"), dtype=float),
    )
    assert tuple(scene_get(placeholder_scene, "size")) == tuple(scene_get(default_scene, "size"))
    assert np.isclose(scene_get(placeholder_scene, "fov"), scene_get(default_scene, "fov"), atol=1e-12, rtol=1e-12)
    assert np.asarray(scene_get(placeholder_scene, "photons"), dtype=float).shape == np.asarray(
        scene_get(default_scene, "photons"),
        dtype=float,
    ).shape


def test_pattern_scene_dispatches_accept_empty_wave_placeholders(asset_store) -> None:
    placeholder_cases = [
        ("line ee", (64, 2, []), (64, 2)),
        ("line ep", (64, 2, []), (64, 2)),
        ("bar", (64, 3, []), (64, 3)),
        ("bar ee", (64, 3, []), (64, 3)),
        ("point array", (64, 8, "ep", 1, []), (64, 8, "ep", 1)),
        ("grid lines", (64, 8, "ep", 1, []), (64, 8, "ep", 1)),
        ("checkerboard", (8, 4, "ep", []), (8, 4, "ep")),
        ("star pattern", (64, "ep", 8, []), (64, "ep", 8)),
    ]

    for name, placeholder_args, explicit_args in placeholder_cases:
        placeholder = scene_create(name, *placeholder_args, asset_store=asset_store)
        explicit = scene_create(name, *explicit_args, asset_store=asset_store)

        np.testing.assert_allclose(
            np.asarray(scene_get(placeholder, "photons"), dtype=float),
            np.asarray(scene_get(explicit, "photons"), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        assert np.array_equal(
            np.asarray(scene_get(placeholder, "wave"), dtype=float),
            np.asarray(scene_get(explicit, "wave"), dtype=float),
        )


def test_pattern_scene_dispatches_accept_empty_optional_placeholders(asset_store) -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    placeholder_cases = [
        ("line ee", (64, [], wave), (64, 0, wave)),
        ("line ep", (64, [], wave), (64, 0, wave)),
        ("bar", (64, [], wave), (64, 5, wave)),
        ("bar ee", (64, [], wave), (64, 5, wave)),
        ("point array", (128, 16, [], [], wave), (128, 16, "d65", 1, wave)),
        ("grid lines", (128, 16, [], [], wave), (128, 16, "ep", 1, wave)),
        ("checkerboard", ([], 8, "ep", wave), (16, 8, "ep", wave)),
        ("checkerboard", (16, [], "ep", wave), (16, 8, "ep", wave)),
        ("star pattern", (256, [], [], wave), (256, "ep", 8, wave)),
    ]

    for name, placeholder_args, explicit_args in placeholder_cases:
        placeholder = scene_create(name, *placeholder_args, asset_store=asset_store)
        explicit = scene_create(name, *explicit_args, asset_store=asset_store)

        np.testing.assert_allclose(
            np.asarray(scene_get(placeholder, "photons"), dtype=float),
            np.asarray(scene_get(explicit, "photons"), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        assert np.array_equal(
            np.asarray(scene_get(placeholder, "wave"), dtype=float),
            np.asarray(scene_get(explicit, "wave"), dtype=float),
        )


def test_reflectance_chart_scene_supports_explicit_sample_lists(asset_store) -> None:
    scene = scene_create(
        "reflectance chart",
        8,
        [[1, 2], [1, 2], [1]],
        [
            "MunsellSamples_Vhrel.mat",
            "Food_Vhrel.mat",
            "skin/HyspexSkinReflectance.mat",
        ],
        None,
        True,
        "without replacement",
        asset_store=asset_store,
    )

    photons = scene_get(scene, "photons")
    chart_parameters = scene_get(scene, "chart parameters")

    assert photons.shape == (24, 24, scene_get(scene, "wave").size)
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 100.0, rtol=5e-2)
    assert np.array_equal(chart_parameters["rowcol"], np.array([3, 3]))
    assert chart_parameters["rIdxMap"].shape == (24, 24)
    assert np.array_equal(chart_parameters["sSamples"][0], np.array([1, 2]))
    assert np.array_equal(chart_parameters["sSamples"][1], np.array([1, 2]))
    assert np.array_equal(chart_parameters["sSamples"][2], np.array([1]))


def test_reflectance_chart_dispatch_accepts_matlab_empty_placeholders(asset_store) -> None:
    s_files = [
        "MunsellSamples_Vhrel.mat",
        "Food_Vhrel.mat",
        "skin/HyspexSkinReflectance.mat",
    ]
    s_samples = [2, 2, 1]

    explicit = scene_create(
        "reflectance chart",
        24,
        s_samples,
        s_files,
        None,
        True,
        "r",
        asset_store=asset_store,
    )
    placeholder = scene_create(
        "reflectance chart",
        [],
        s_samples,
        s_files,
        [],
        [],
        [],
        asset_store=asset_store,
    )

    np.testing.assert_allclose(
        np.asarray(scene_get(placeholder, "photons"), dtype=float),
        np.asarray(scene_get(explicit, "photons"), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    placeholder_chart = scene_get(placeholder, "chart parameters")
    explicit_chart = scene_get(explicit, "chart parameters")
    assert placeholder_chart["pSize"] == explicit_chart["pSize"] == 24
    assert placeholder_chart["grayFlag"] == explicit_chart["grayFlag"] is True
    assert placeholder_chart["sampling"] == explicit_chart["sampling"] == "r"
    assert np.array_equal(np.asarray(scene_get(placeholder, "wave"), dtype=float), np.asarray(scene_get(explicit, "wave"), dtype=float))


def test_reflectance_chart_dispatch_accepts_empty_source_and_sample_placeholders(asset_store) -> None:
    wave = np.array([450.0, 550.0, 650.0], dtype=float)
    explicit = scene_create(
        "reflectance chart",
        24,
        [50, 40, 10],
        [
            "MunsellSamples_Vhrel.mat",
            "Food_Vhrel.mat",
            "skin/HyspexSkinReflectance.mat",
        ],
        wave,
        True,
        "r",
        asset_store=asset_store,
    )

    placeholder_cases = [
        scene_create("reflectance chart", 24, [], [], wave, [], [], asset_store=asset_store),
        scene_create("reflectance chart", 24, None, None, wave, None, None, asset_store=asset_store),
    ]
    explicit_chart = scene_get(explicit, "chart parameters")

    for placeholder in placeholder_cases:
        np.testing.assert_allclose(
            np.asarray(scene_get(placeholder, "photons"), dtype=float),
            np.asarray(scene_get(explicit, "photons"), dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        placeholder_chart = scene_get(placeholder, "chart parameters")
        assert placeholder_chart["sFiles"] == explicit_chart["sFiles"]
        assert placeholder_chart["grayFlag"] == explicit_chart["grayFlag"] is True
        assert placeholder_chart["sampling"] == explicit_chart["sampling"] == "r"
        assert all(
            np.array_equal(np.asarray(left, dtype=int), np.asarray(right, dtype=int))
            for left, right in zip(placeholder_chart["sSamples"], explicit_chart["sSamples"], strict=True)
        )
        assert np.array_equal(
            np.asarray(scene_get(placeholder, "wave"), dtype=float),
            np.asarray(scene_get(explicit, "wave"), dtype=float),
        )


def test_reflectance_chart_scene_supports_struct_parameters_and_absolute_paths(asset_store) -> None:
    params = {
        "pSize": 8,
        "sFiles": [
            asset_store.resolve("data/surfaces/reflectances/MunsellSamples_Vhrel.mat"),
            asset_store.resolve("data/surfaces/reflectances/Food_Vhrel.mat"),
            asset_store.resolve("data/surfaces/reflectances/skin/HyspexSkinReflectance.mat"),
        ],
        "sSamples": [[1], [1], [1]],
        "grayFlag": False,
        "sampling": "all",
    }

    scene = scene_create("reflectance chart", params, asset_store=asset_store)
    photons = scene_get(scene, "photons")
    chart_parameters = scene_get(scene, "chart parameters")

    assert photons.shape == (16, 16, scene_get(scene, "wave").size)
    assert np.array_equal(chart_parameters["rowcol"], np.array([2, 2]))
    assert all(np.array_equal(item, np.array([1])) for item in chart_parameters["sSamples"])
    assert all(path.endswith(".mat") for path in chart_parameters["sFiles"])


def test_scene_from_file_rgb_array_uses_display_geometry(asset_store) -> None:
    display = display_create("default", asset_store=asset_store)
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )

    scene = scene_from_file(image, "rgb", 50.0, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 2, scene_get(scene, "wave").size)
    assert np.isclose(scene_get(scene, "distance"), display_get(display, "viewing distance"))
    assert np.isclose(scene_get(scene, "fov"), 2.0 * display_get(display, "deg per dot"))
    assert scene_get(scene, "filename") == "numerical"
    assert scene_get(scene, "source type") == "rgb"
    assert np.isclose(scene_get(scene, "mean luminance", asset_store=asset_store), 50.0, rtol=5e-2)


def test_scene_from_file_monochrome_file_path(tmp_path, asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.array(
        [
            [0, 64, 128],
            [255, 192, 32],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "mono.png"
    iio.imwrite(image_path, image)

    scene = scene_from_file(image_path, "monochrome", None, display, asset_store=asset_store)

    photons = scene_get(scene, "photons")
    assert photons.shape == (2, 3, scene_get(scene, "wave").size)
    assert scene_get(scene, "filename") == str(image_path)
    assert scene.name.startswith("mono - ")
    assert np.all(np.isfinite(photons))
    assert float(np.mean(photons[1, 0, :])) > float(np.mean(photons[0, 0, :]))


def test_scene_from_file_preserves_display_wave_when_not_overridden(asset_store) -> None:
    display = display_create("lcdExample.mat", asset_store=asset_store)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scene = scene_from_file(image, "rgb", None, display, asset_store=asset_store)
    assert np.array_equal(scene_get(scene, "wave"), display_get(display, "wave"))
