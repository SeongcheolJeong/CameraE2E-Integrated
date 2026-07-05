from __future__ import annotations

import numpy as np
import pyisetcam.optics as optics_module

from pyisetcam import (
    colorTransformMatrix,
    conePlot,
    displayCreate,
    displayGet,
    humanAchromaticOTF,
    humanConeContrast,
    humanCones,
    humanConeMosaic,
    humanConeIsolating,
    humanCore,
    humanLSF,
    humanMacularTransmittance,
    humanOI,
    humanOpticalDensity,
    humanOTF,
    humanOTF_ibio,
    humanPupilSize,
    humanSpaceTime,
    humanUVSafety,
    humanWaveDefocus,
    imageLinearTransform,
    ieConePlot,
    ijspeert,
    kellySpaceTime,
    oiCompute,
    poirsonSpatioChromatic,
    sceneCreate,
    sceneFromFile,
    sceneGet,
    sensorCompute,
    sensorConePlot,
    sensorCreateConeMosaic,
    sensorGet,
    sensorHumanResize,
    sensorSet,
    sensorShowCFA,
    watsonImpulseResponse,
    watsonRGCSpacing,
    westheimerLSF,
    xyz2lms,
    xyz2srgb,
)
from pyisetcam.assets import AssetStore
from pyisetcam import oiCreate, oiGet
from pyisetcam.utils import energy_to_quanta


def test_human_pupil_size_models_match_analytic_forms() -> None:
    diameter_ms, area_ms = humanPupilSize(100.0, "ms")
    diameter_dg, area_dg = humanPupilSize(100.0, "dg")
    diameter_sd, area_sd = humanPupilSize(100.0, "sd", 10.0)
    diameter_wy, area_wy = humanPupilSize(100.0, "wy", {"age": 30, "area": 10.0, "eyeNum": 2})

    expected_ms = 4.9 - 3.0 * np.tanh(0.4 * np.log10(100.0) + 1.0)
    expected_dg = 10.0 ** (0.8558 - 0.000401 * (np.log10(100.0) + 8.6) ** 3)
    flux = 100.0 * 10.0
    expected_sd = 7.75 - 5.75 * (flux / 846.0) ** 0.41 / ((flux / 846.0) ** 0.41 + 2.0)
    expected_wy = expected_sd + (30.0 - 28.58) * (0.02132 - 0.009562 * expected_sd)

    assert np.isclose(diameter_ms, expected_ms)
    assert np.isclose(diameter_dg, expected_dg)
    assert np.isclose(diameter_sd, expected_sd)
    assert np.isclose(diameter_wy, expected_wy)
    assert np.isclose(area_ms, np.pi * (diameter_ms / 2.0) ** 2)
    assert np.isclose(area_dg, np.pi * (diameter_dg / 2.0) ** 2)
    assert np.isclose(area_sd, np.pi * (diameter_sd / 2.0) ** 2)
    assert np.isclose(area_wy, np.pi * (diameter_wy / 2.0) ** 2)


def test_human_optical_density_matches_stockman_profiles() -> None:
    fovea = humanOpticalDensity()
    periphery = humanOpticalDensity("stockman periphery", np.arange(400.0, 701.0, 10.0, dtype=float))
    subject = humanOpticalDensity("s1p")

    assert fovea["visfield"] == "fovea"
    assert np.isclose(fovea["macular"], 0.28)
    assert periphery["wave"].shape == (31,)
    assert np.isclose(periphery["LPOD"], 0.38)
    assert np.isclose(periphery["SPOD"], 0.3)
    assert subject["visfield"] == "p"
    assert np.isclose(subject["macular"], 0.0)
    assert np.isclose(subject["LPOD"], (0.4964 / 0.5) * 0.38)


def test_human_cone_contrast_energy_and_quanta_agree() -> None:
    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    background = np.full(wave.shape, 0.25, dtype=float)
    signal = np.column_stack((background * 0.1, background * 0.2))

    contrast_energy = humanConeContrast(signal, background, wave, "energy")
    contrast_quanta = humanConeContrast(
        energy_to_quanta(signal, wave),
        energy_to_quanta(background, wave),
        wave,
        "quanta",
    )

    assert contrast_energy.shape == (3, 2)
    np.testing.assert_allclose(contrast_quanta, contrast_energy)


def test_human_cone_isolating_returns_scaled_isolating_directions() -> None:
    display = displayCreate("LCD-Apple")
    cone_isolating, spd = humanConeIsolating(display)

    wave = np.asarray(displayGet(display, "wave"), dtype=float).reshape(-1)
    _, cones = AssetStore.default().load_spectra("stockman.mat", wave_nm=wave)
    isolation = np.asarray(cones, dtype=float).T @ spd
    off_diagonal = isolation - np.diag(np.diag(isolation))

    assert cone_isolating.shape == (3, 3)
    assert spd.shape == (wave.size, 3)
    np.testing.assert_allclose(np.max(np.abs(cone_isolating), axis=0), 0.5)
    assert np.max(np.abs(off_diagonal)) < 2.0e-4
    assert np.all(np.diag(isolation) > 0.0)


def test_human_cone_mosaic_replays_defaults_and_lms_density_patchup() -> None:
    default_xy, default_cone_type, default_densities, default_seed = humanConeMosaic([6, 8])
    placeholder_xy, placeholder_cone_type, placeholder_densities, placeholder_seed = humanConeMosaic(
        [6, 8],
        [],
        [],
        [],
    )
    blank_xy, blank_cone_type, blank_densities, blank_seed = humanConeMosaic(
        [6, 8],
        "",
        "",
        "",
    )

    assert default_xy.shape == (48, 2)
    assert default_cone_type.shape == (6, 8)
    np.testing.assert_allclose(placeholder_xy, default_xy)
    np.testing.assert_array_equal(placeholder_cone_type, default_cone_type)
    np.testing.assert_allclose(placeholder_densities, default_densities)
    np.testing.assert_allclose(blank_xy, default_xy)
    np.testing.assert_array_equal(blank_cone_type, default_cone_type)
    np.testing.assert_allclose(blank_densities, default_densities)
    assert np.isclose(np.sum(default_densities), 1.0)
    assert placeholder_seed == default_seed == 0
    assert blank_seed == default_seed == 0

    _, _, corrected_densities, corrected_seed = humanConeMosaic([4, 5], [1.0, 1.0, 1.0], [], 7)

    np.testing.assert_allclose(corrected_densities, np.array([0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    assert corrected_seed == 7


def test_human_wave_defocus_matches_thibos_formula() -> None:
    wave = np.array([400.0, 500.0, 700.0], dtype=float)
    expected = 1.7312 - (0.63346 / (wave * 1.0e-3 - 0.21410))

    np.testing.assert_allclose(humanWaveDefocus(wave), expected)


def test_optics_module_human_matlab_aliases() -> None:
    assert optics_module.humanWaveDefocus is optics_module.human_wave_defocus
    assert optics_module.humanAchromaticOTF is optics_module.human_achromatic_otf
    assert optics_module.humanCore is optics_module.human_core
    assert optics_module.humanOTF is optics_module.human_otf
    assert optics_module.humanOTF_ibio is optics_module.human_otf_ibio
    assert optics_module.humanLSF is optics_module.human_lsf
    assert optics_module.humanMacularTransmittance is optics_module.human_macular_transmittance
    assert optics_module.humanOI is optics_module.human_oi


def test_optics_module_native_matlab_aliases() -> None:
    assert optics_module.opticsBuild2Dotf is optics_module.optics_build_2d_otf
    assert optics_module.opticsCoC is optics_module.optics_coc
    assert optics_module.opticsDefocusCore is optics_module.optics_defocus_core
    assert optics_module.opticsDefocusDisplacement is optics_module.optics_defocus_displacement
    assert optics_module.opticsDepthDefocus is optics_module.optics_depth_defocus
    assert optics_module.opticsDoF is optics_module.optics_dof
    assert optics_module.opticsPSF2OTF is optics_module.optics_psf_to_otf
    assert optics_module.opticsRayTrace is optics_module.optics_ray_trace


def test_optics_module_oi_matlab_aliases() -> None:
    assert optics_module.oiBirefringentDiffuser is optics_module.oi_birefringent_diffuser
    assert optics_module.oiCalculateIlluminance is optics_module.oi_calculate_illuminance
    assert optics_module.oiCalculateOTF is optics_module.oi_calculate_otf
    assert optics_module.oiCameraMotion is optics_module.oi_camera_motion
    assert optics_module.oiClearData is optics_module.oi_clear_data
    assert optics_module.oiCombineDepths is optics_module.oi_combine_depths
    assert optics_module.oiCompute is optics_module.oi_compute
    assert optics_module.oiCreate is optics_module.oi_create
    assert optics_module.oiCrop is optics_module.oi_crop
    assert optics_module.oiCustomCompute is optics_module.oi_custom_compute
    assert optics_module.oiDepthCombine is optics_module.oi_depth_combine
    assert optics_module.oiDepthSegmentMap is optics_module.oi_depth_segment_map
    assert optics_module.oiDiffuser is optics_module.oi_diffuser
    assert optics_module.oiFrequencyResolution is optics_module.oi_frequency_resolution
    assert optics_module.oiGet is optics_module.oi_get
    assert optics_module.oiPadDepthMap is optics_module.oi_pad_depth_map
    assert optics_module.oiPreviewVideo is optics_module.oi_preview_video
    assert optics_module.oiSaveImage is optics_module.oi_save_image
    assert optics_module.oiSet is optics_module.oi_set
    assert optics_module.oiShowImage is optics_module.oi_show_image
    assert optics_module.oiSpace is optics_module.oi_space
    assert optics_module.oiSpatialResample is optics_module.oi_spatial_resample
    assert optics_module.oiSpatialSupport is optics_module.oi_spatial_support
    assert optics_module.oiWBCompute is optics_module.oi_wb_compute


def test_optics_module_wvf_matlab_aliases() -> None:
    assert optics_module.wvf2oi is optics_module.wvf_to_oi
    assert optics_module.wvf2optics is optics_module.wvf_to_optics
    assert optics_module.wvf2PSF is optics_module.wvf_to_psf
    assert optics_module.wvfAperture is optics_module.wvf_aperture
    assert optics_module.wvfApertureP is optics_module.wvf_aperture_params
    assert optics_module.wvfClearData is optics_module.wvf_clear_data
    assert optics_module.wvfCompute is optics_module.wvf_compute
    assert optics_module.wvfComputePSF is optics_module.wvf_compute_psf
    assert optics_module.wvfCreate is optics_module.wvf_create
    assert optics_module.wvfDefocusDioptersToMicrons is optics_module.wvf_defocus_diopters_to_microns
    assert optics_module.wvfDefocusMicronsToDiopters is optics_module.wvf_defocus_microns_to_diopters
    assert optics_module.wvfGet is optics_module.wvf_get
    assert optics_module.wvfLoadThibosVirtualEyes is optics_module.wvf_load_thibos_virtual_eyes
    assert optics_module.wvfPupilAmplitude is optics_module.wvf_pupil_amplitude
    assert optics_module.wvfPupilFunction is optics_module.wvf_pupil_function
    assert optics_module.wvfSet is optics_module.wvf_set


def test_optics_module_raytrace_matlab_aliases() -> None:
    assert optics_module.rtAngleLUT is optics_module.rt_angle_lut
    assert optics_module.rtBlockCenter is optics_module.rt_block_center
    assert optics_module.rtChooseBlockSize is optics_module.rt_choose_block_size
    assert optics_module.rtDIInterp is optics_module.rt_di_interp
    assert optics_module.rtExtractBlock is optics_module.rt_extract_block
    assert optics_module.rtFileNames is optics_module.rt_file_names
    assert optics_module.rtFilteredBlockSupport is optics_module.rt_filtered_block_support
    assert optics_module.rtGeometry is optics_module.rt_geometry
    assert optics_module.rtImageRotate is optics_module.rt_image_rotate
    assert optics_module.rtImportData is optics_module.rt_import_data
    assert optics_module.rtInsertBlock is optics_module.rt_insert_block
    assert optics_module.rtOTF is optics_module.rt_otf
    assert optics_module.rtPSFApply is optics_module.rt_psf_apply
    assert optics_module.rtPSFEdit is optics_module.rt_psf_edit
    assert optics_module.rtPSFGrid is optics_module.rt_psf_grid
    assert optics_module.rtPSFInterp is optics_module.rt_psf_interp
    assert optics_module.rtPrecomputePSF is optics_module.rt_precompute_psf
    assert optics_module.rtPrecomputePSFApply is optics_module.rt_precompute_psf_apply
    assert optics_module.rtRIInterp is optics_module.rt_ri_interp
    assert optics_module.rtRootPath is optics_module.rt_root_path
    assert optics_module.rtSampleHeights is optics_module.rt_sample_heights
    assert optics_module.rtSynthetic is optics_module.rt_synthetic


def test_optics_module_residual_matlab_aliases() -> None:
    assert optics_module.airyDisk is optics_module.airy_disk
    assert optics_module.ieFieldHeight2Index is optics_module.ie_field_height_to_index
    assert optics_module.sceCreate is optics_module.sce_create
    assert optics_module.sceGet is optics_module.sce_get
    assert optics_module.siSynthetic is optics_module.si_synthetic
    assert optics_module.zemaxLoad is optics_module.zemax_load
    assert optics_module.zemaxReadHeader is optics_module.zemax_read_header


def test_human_cones_replays_defaults_and_macular_density_adjustment() -> None:
    cones_default, correction_default, wave_default = humanCones()
    cones_placeholder, correction_placeholder, wave_placeholder = humanCones([], [], [], [])

    np.testing.assert_allclose(wave_default, np.arange(370.0, 731.0, 1.0, dtype=float))
    np.testing.assert_allclose(wave_placeholder, wave_default)
    np.testing.assert_allclose(cones_placeholder, cones_default)
    np.testing.assert_allclose(correction_default, np.ones_like(correction_default))
    np.testing.assert_allclose(correction_placeholder, correction_default)

    wave = np.arange(400.0, 701.0, 10.0, dtype=float)
    cones_adjusted, macular_correction, returned_wave = humanCones("stockmanAbs", wave, 0.0, 0.35)
    _, base_cones = AssetStore.default().load_spectra("stockmanAbs", wave_nm=wave)
    _, profile = AssetStore.default().load_spectra("macularPigment.mat", wave_nm=wave)
    expected_correction = np.power(10.0, -((np.asarray(profile, dtype=float).reshape(-1) / 0.3521) * (0.0 - 0.35)))

    np.testing.assert_allclose(returned_wave, wave)
    np.testing.assert_allclose(macular_correction, expected_correction)
    np.testing.assert_allclose(cones_adjusted, np.asarray(base_cones, dtype=float) * expected_correction[:, np.newaxis])


def test_watson_impulse_response_is_normalized() -> None:
    impulse_response, time, t_mtf, frequency = watsonImpulseResponse(np.linspace(0.0, 0.2, 11), 0.25)

    assert np.all(time > 0.0)
    assert np.isclose(np.sum(impulse_response), 1.0)
    assert t_mtf.shape == impulse_response.shape
    assert frequency.shape == impulse_response.shape


def test_kelly_space_time_and_human_space_time_dispatch() -> None:
    spatial = np.array([1.0, 2.0], dtype=float)
    temporal = np.array([4.0, 8.0], dtype=float)

    sensitivity, spatial_grid, temporal_grid = kellySpaceTime(spatial, temporal)
    dispatched, dispatched_spatial, dispatched_temporal = humanSpaceTime("kelly79", spatial, temporal)

    expected = (
        (6.1 + 7.3 * abs(np.log10((4.0 / 1.0) / 3.0)) ** 3)
        * (4.0 / 1.0)
        * (2.0 * np.pi * 1.0) ** 2
        * np.exp(-2.0 * (2.0 * np.pi * 1.0) / (45.9 / ((4.0 / 1.0) + 2.0)))
        / 2.0
    )

    assert sensitivity.shape == (2, 2)
    assert np.isclose(sensitivity[0, 0], expected)
    np.testing.assert_allclose(dispatched, sensitivity)
    np.testing.assert_allclose(dispatched_spatial, spatial_grid)
    np.testing.assert_allclose(dispatched_temporal, temporal_grid)


def test_poirson_spatio_chromatic_and_human_space_time_color_dispatch() -> None:
    lum_1d, rg_1d, by_1d, positions = poirsonSpatioChromatic(120.0, 1)
    lum_2d, rg_2d, by_2d, positions_2d = poirsonSpatioChromatic(120.0, 2)
    dispatched, support, _ = humanSpaceTime("poirsoncolor")

    assert lum_1d.ndim == 1
    assert lum_2d.ndim == 2
    assert np.isclose(np.sum(lum_1d), 1.0)
    assert np.isclose(np.sum(rg_1d), 1.0)
    assert np.isclose(np.sum(by_1d), 1.0)
    assert np.isclose(np.sum(lum_2d), 1.0)
    assert np.isclose(np.sum(rg_2d), 1.0)
    assert np.isclose(np.sum(by_2d), 1.0)
    assert np.isclose(positions[positions.size // 2], 0.0)
    np.testing.assert_allclose(positions, positions_2d)
    assert set(dispatched.keys()) == {"lum", "rg", "by"}
    assert support.ndim == 1


def test_westheimer_lsf_is_symmetric_and_normalized() -> None:
    support = np.arange(-300.0, 301.0, 1.0, dtype=float)
    line_spread = westheimerLSF(support)

    assert np.isclose(np.sum(line_spread), 1.0)
    np.testing.assert_allclose(line_spread, line_spread[::-1])


def test_watson_rgc_spacing_returns_expected_shapes() -> None:
    smf0, r, smf1d = watsonRGCSpacing(4)

    assert smf0.shape == (5, 5)
    assert r.shape == (1000,)
    assert smf1d.shape == (4, 1000)
    assert np.isfinite(smf0[2, 2])
    assert np.all(smf1d > 0.0)


def test_human_achromatic_otf_models_and_core_are_consistent() -> None:
    sample_sf = np.array([0.0, 10.0, 30.0], dtype=float)
    exp_otf = humanAchromaticOTF(sample_sf, "exp")
    dl_otf = humanAchromaticOTF(sample_sf, "dl", 3.0)
    watson_otf = humanAchromaticOTF(sample_sf, "watson", 3.0)
    core, achromatic = humanCore(np.array([500.0, 600.0], dtype=float), sample_sf, 0.0015, 60.0)

    assert np.isclose(exp_otf[0], 1.0)
    assert np.isclose(dl_otf[0], 1.0)
    assert np.isclose(watson_otf[0], 1.0)
    assert core.shape == (2, 3)
    np.testing.assert_allclose(achromatic, exp_otf)


def test_human_otf_and_lsf_return_headless_payloads() -> None:
    freq = np.linspace(-20.0, 20.0, 11, dtype=float)
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    support = np.dstack((fx, fy))

    otf, returned_support, wave = humanOTF(0.0015, 60.0, support, np.array([550.0, 650.0], dtype=float))
    line_spread_mm, x_mm, _ = humanLSF(0.0015, 60.0, "mm", np.array([550.0, 650.0], dtype=float))
    line_spread_um, x_um, _ = humanLSF(0.0015, 60.0, "um", np.array([550.0, 650.0], dtype=float))

    assert otf.shape == (11, 11, 2)
    np.testing.assert_allclose(returned_support, support)
    np.testing.assert_allclose(wave, np.array([550.0, 650.0], dtype=float))
    centered_otf = np.abs(np.fft.ifftshift(otf[:, :, 0]))
    assert np.isclose(centered_otf[centered_otf.shape[0] // 2, centered_otf.shape[1] // 2], 1.0)
    assert line_spread_mm.shape[0] == 2
    np.testing.assert_allclose(x_um / 1000.0, x_mm)
    assert line_spread_um.shape == line_spread_mm.shape


def test_human_otf_ibio_and_macular_transmittance_match_headless_contracts() -> None:
    freq = np.linspace(-20.0, 20.0, 11, dtype=float)
    fx, fy = np.meshgrid(freq, freq, indexing="xy")
    support = np.dstack((fx, fy))

    legacy_otf, returned_support, wave = humanOTF(0.0015, 60.0, support, np.array([550.0], dtype=float))
    ibio_otf, returned_support_ibio, wave_ibio = humanOTF_ibio(0.0015, 60.0, support, np.array([550.0], dtype=float))

    assert np.isclose(np.abs(ibio_otf[0, 0, 0]), 1.0)
    np.testing.assert_allclose(returned_support_ibio, returned_support)
    np.testing.assert_allclose(wave_ibio, wave)
    np.testing.assert_allclose(np.fft.ifftshift(np.fft.ifftshift(legacy_otf[:, :, 0])), ibio_otf[:, :, 0])

    oi = humanMacularTransmittance(oiCreate(), 0.35)
    current_wave = np.asarray(oiGet(oi, "wave"), dtype=float).reshape(-1)
    transmittance = np.asarray(oiGet(oi, "transmittance", current_wave), dtype=float).reshape(-1)
    assert transmittance.shape == current_wave.shape
    assert np.all(transmittance <= 1.0)
    assert np.all(transmittance >= 0.0)
    assert float(transmittance[np.argmin(np.abs(current_wave - 460.0))]) < float(
        transmittance[np.argmin(np.abs(current_wave - 650.0))]
    )


def test_human_oi_replays_scene_compute_with_human_otf() -> None:
    scene = sceneCreate("macbeth d65")
    oi = humanOI(scene, oiCreate("shift invariant"))

    photons = np.asarray(oiGet(oi, "photons"), dtype=float)
    illuminance = np.asarray(oiGet(oi, "illuminance"), dtype=float)

    assert photons.ndim == 3
    assert photons.shape[:2] == illuminance.shape
    assert photons.shape[2] == np.asarray(oiGet(oi, "wave"), dtype=float).size
    assert float(np.mean(illuminance)) > 0.0
    assert str(oiGet(oi, "compute method")) == "humanmw"


def test_s_human_color_blind_workflow_replays_brettel_projection() -> None:
    scene = sceneCreate("macbeth d65")
    xyz = np.asarray(sceneGet(scene, "xyz"), dtype=float)
    white_xyz = np.asarray(sceneGet(scene, "illuminant xyz"), dtype=float).reshape(-1)
    baseline_lms = np.asarray(xyz2lms(xyz), dtype=float)

    for cb_type, preserved in ((1, (1, 2)), (2, (0, 2)), (3, (0, 1))):
        lms = np.asarray(xyz2lms(xyz, cb_type, white_xyz), dtype=float)
        cb_xyz = np.asarray(imageLinearTransform(lms, colorTransformMatrix("lms2xyz")), dtype=float)
        cb_rgb = np.asarray(xyz2srgb(cb_xyz), dtype=float)

        assert cb_rgb.shape == xyz.shape
        assert np.all(np.isfinite(cb_rgb))
        np.testing.assert_allclose(lms[..., preserved], baseline_lms[..., preserved], atol=1.0e-7)
        assert float(np.max(np.abs(lms[..., cb_type - 1] - baseline_lms[..., cb_type - 1]))) > 1.0e-4


def test_s_human_display_psf_workflow_replays_headlessly(asset_store) -> None:
    display = displayCreate("LCD-Apple")
    image = np.zeros((51, 51, 3), dtype=float)
    image[25, 25, 1] = 1.0

    scene = sceneFromFile(image, "rgb", None, display, asset_store=asset_store)
    oi = oiCompute(oiCreate("wvf"), scene)
    sensor, _, cone_type, _, _ = sensorCreateConeMosaic(
        None,
        [128, 128],
        [0.0, 0.0, 1.0, 0.0],
        [1.0e-6, 1.0e-6],
        7,
        asset_store=asset_store,
    )
    sensor = sensorSet(sensor, "fov", sceneGet(scene, "fov"), oi)
    sensor = sensorSet(sensor, "noise flag", 0)
    sensor = sensorCompute(sensor, oi, seed=0)

    volts = np.asarray(sensorGet(sensor, "volts"), dtype=float)

    assert np.all(np.asarray(cone_type, dtype=int) == 3)
    assert volts.ndim == 2
    assert float(np.max(volts)) > 0.0
    assert np.isclose(
        float(volts[volts.shape[0] // 2, volts.shape[1] // 2]),
        float(np.max(volts)),
    )


def test_ie_cone_plot_returns_headless_mosaic_payload() -> None:
    _, xy, cone_type, _, _ = sensorCreateConeMosaic()
    payload = ieConePlot(xy, cone_type)

    assert set(payload.keys()) == {"support", "spread", "delta", "grid", "image"}
    assert payload["grid"].ndim == 2
    assert payload["image"].shape[:2] == payload["grid"].shape
    assert payload["image"].shape[2] == 3
    assert payload["spread"] > 0.0
    assert payload["delta"] > 0.0
    assert np.any(payload["image"][:, :, 0] > 0.0)
    assert np.any(payload["image"][:, :, 1] > 0.0)
    assert np.any(payload["image"][:, :, 2] > 0.0)


def test_cone_plot_alias_matches_ie_cone_plot_payload() -> None:
    _, xy, cone_type, _, _ = sensorCreateConeMosaic()
    payload = conePlot(xy, cone_type, 18.0, 1.1, 0.5)
    expected = ieConePlot(xy, cone_type, 18.0, 1.1, 0.5)

    assert payload.keys() == expected.keys()
    for key in payload:
        value = payload[key]
        expected_value = expected[key]
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, expected_value)
        else:
            assert value == expected_value


def test_sensor_cone_plot_matches_ie_cone_plot_payload() -> None:
    sensor, xy, cone_type, _, _ = sensorCreateConeMosaic()
    payload = sensorConePlot(sensor, 18.0, 1.1, 0.5)
    expected = ieConePlot(xy, cone_type, 18.0, 1.1, 0.5)

    assert payload.keys() == expected.keys()
    for key in payload:
        value = payload[key]
        expected_value = expected[key]
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, expected_value)
        else:
            assert value == expected_value


def test_sensor_cone_plot_falls_back_to_sensor_show_cfa_for_block_layout() -> None:
    sensor, _, _, _, _ = sensorCreateConeMosaic()
    block_sensor = sensorHumanResize(sensor, [0, 0], [0, 0])
    payload = sensorConePlot(block_sensor, [], [], [])
    _, expected_image = sensorShowCFA(block_sensor)

    assert payload["support"] is None
    assert payload["spread"] is None
    assert payload["delta"] is None
    assert payload["grid"] is None
    np.testing.assert_allclose(payload["image"], expected_image)


def test_human_uv_safety_methods_match_expected_thresholds() -> None:
    wave = np.arange(300.0, 701.0, 10.0, dtype=float)
    energy = np.full(wave.shape, 1.0e-3, dtype=float)

    safe_time, skin_level, skin_flag = humanUVSafety(energy, wave, method="skineye")
    eye_ok, eye_level, eye_flag = humanUVSafety(energy, wave, method="eye", duration=10.0)
    blue_time, blue_level, blue_flag = humanUVSafety(energy, wave, method="bluehazard", duration=10.0)
    thermal_value, thermal_level, thermal_flag = humanUVSafety(energy, wave, method="thermalskin", duration=1.0)
    threshold_value, threshold_level, threshold_flag = humanUVSafety(
        energy,
        wave,
        method="skin thermal threshold",
        duration=0.03,
    )

    assert np.isfinite(safe_time)
    assert skin_level > 0.0
    assert skin_flag is None
    assert isinstance(eye_ok, bool) and eye_level > 0.0 and eye_flag == eye_ok
    assert blue_level > 0.0 and isinstance(blue_flag, bool)
    assert np.isfinite(blue_time) or np.isinf(blue_time)
    assert thermal_value == thermal_level and isinstance(thermal_flag, bool)
    assert threshold_value == threshold_level and isinstance(threshold_flag, bool)


def test_ijspeert_returns_mtf_psf_and_lsf() -> None:
    sample_sf = np.arange(0.0, 61.0, 1.0, dtype=float)
    phi = np.deg2rad(np.linspace(0.0, 0.1, 10, dtype=float))

    mtf, psf, lsf = ijspeert(30.0, 3.0, 0.142, sample_sf, phi)

    assert mtf.shape == sample_sf.shape
    assert psf is not None and psf.shape == phi.shape
    assert lsf is not None and lsf.shape == phi.shape
    assert np.isclose(mtf[0], 1.0, atol=1e-6)
